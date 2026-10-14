"""CPU-only admitted maximum-load preparation for the C4 R8 rung."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import torch

from .c4_fixture import make_c4_fixture_v1
from .c4_population import TensorWorldC4BirthConfigV1
from .c4_state import TensorWorldStateBundleV4
from .c5_determinism import (
    TensorWorldC5ReplayContextV1,
    make_c5_pcg64_context_v1,
)
from .contracts import TensorWorldContractError


C4_RESIDENT_MAXLOAD_R8_SCHEMA_VERSION_V1: Final[str] = (
    "tensor_world_c4_resident_maxload_r8_preparation_v1"
)
C4_RESIDENT_MAXLOAD_R8_DECISION_V1: Final[str] = "NO_GATE_DECISION"
C4_RESIDENT_MAXLOAD_R8_IDENTITY_LIMITATION_V1: Final[str] = (
    "bounded_dense_world_local_identity_map_only; biological_ids_repeat_across_worlds"
)

_STAGING_INVENTORY_V1: Final[tuple[tuple[str, str, int], ...]] = (
    ("death_agent_ids", "int64", 8),
    ("death_slots", "int64", 8),
    ("death_reason_codes", "int32", 4),
    ("birth_parent_slots", "int32", 4),
    ("birth_child_slots", "int32", 4),
    ("birth_pos_y", "int32", 4),
    ("birth_pos_x", "int32", 4),
    ("birth_energy", "float32", 4),
    ("birth_child_agent_ids", "int64", 8),
    ("virtual_free_queue", "int64", 8),
)


def c4_resident_maxload_r8_contract_v1() -> dict[str, object]:
    """Freeze the admitted CPU preparation envelope without a resource claim."""

    batch, capacity = 4, 32
    slot_rows = batch * capacity
    inventory = [
        {
            "name": name,
            "shape": [batch, capacity],
            "dtype": dtype,
            "bytes": slot_rows * element_bytes,
        }
        for name, dtype, element_bytes in _STAGING_INVENTORY_V1
    ]
    batch_inventory = [
        {"name": name, "shape": [batch], "dtype": "int64", "bytes": batch * 8}
        for name in ("death_count", "birth_count", "free_count", "event_count")
    ]
    batch_inventory.append(
        {"name": "status", "shape": [batch], "dtype": "int32", "bytes": batch * 4}
    )
    inventory.extend(batch_inventory)
    inventory.append(
        {
            "name": "spawn_uniforms",
            "shape": [batch, capacity, 3],
            "dtype": "float64",
            "bytes": batch * capacity * 3 * 8,
        }
    )
    return {
        "schema_version": C4_RESIDENT_MAXLOAD_R8_SCHEMA_VERSION_V1,
        "route": "CPU_ORACLE_PREPARATION_ONLY",
        "batch": batch,
        "capacity": capacity,
        "initial_active_per_world": 16,
        "initial_inactive_per_world": 16,
        "admitted_max_deaths_per_world": 8,
        "admitted_max_births_per_world": 8,
        "admitted_event_rows_per_world": 16,
        "event_capacity_per_world": 64,
        "identity_next_before": 17,
        "identity_next_after": 25,
        "identity_map_headroom_after": 7,
        "staging_slot_rows": slot_rows,
        "staging_inventory": inventory,
        "staging_footprint_bytes": sum(int(row["bytes"]) for row in inventory),
        "staging_footprint_classification": (
            "logical_minimum_preparation_inventory_not_native_vram_measurement"
        ),
        "identity_map_limitation": C4_RESIDENT_MAXLOAD_R8_IDENTITY_LIMITATION_V1,
        "decision": C4_RESIDENT_MAXLOAD_R8_DECISION_V1,
        "gate_claim": False,
    }


@dataclass(slots=True)
class C4ResidentMaxloadR8FixtureV1:
    bundle: TensorWorldStateBundleV4
    replay: TensorWorldC5ReplayContextV1
    birth_config: TensorWorldC4BirthConfigV1
    spawn_uniforms: torch.Tensor
    contract: dict[str, object]
    schema_version: str = C4_RESIDENT_MAXLOAD_R8_SCHEMA_VERSION_V1
    decision: str = C4_RESIDENT_MAXLOAD_R8_DECISION_V1
    gate_claim: bool = False

    def validate(self) -> None:
        if self.schema_version != C4_RESIDENT_MAXLOAD_R8_SCHEMA_VERSION_V1:
            raise TensorWorldContractError("C4 R8 fixture schema is unsupported")
        if self.bundle.batch != 4 or self.bundle.capacity != 32:
            raise TensorWorldContractError("C4 R8 fixture requires B4/C32")
        if self.bundle.device.type != "cpu":
            raise TensorWorldContractError("C4 R8 preparation is CPU-only")
        if self.decision != C4_RESIDENT_MAXLOAD_R8_DECISION_V1 or self.gate_claim:
            raise TensorWorldContractError("C4 R8 preparation cannot claim a gate")
        self.bundle.validate()
        self.replay.validate_pcg64_for_bundle(self.bundle)
        if tuple(self.spawn_uniforms.shape) != (4, 32, 3) or (
            self.spawn_uniforms.dtype != torch.float64
            or not self.spawn_uniforms.is_contiguous()
        ):
            raise TensorWorldContractError("C4 R8 spawn tape ABI is invalid")


def make_c4_resident_maxload_r8_fixture_v1(
    *, seed: int = 53
) -> C4ResidentMaxloadR8FixtureV1:
    """Create maximum admitted 8-death/8-birth combined B4 turnover."""

    bundle, _ = make_c4_fixture_v1(
        batch=4,
        capacity=32,
        active=16,
        width=32,
        seed=seed,
        world_local_identity=True,
    )
    population = bundle.population
    population.free_queue.fill_(-1)
    population.free_queue_count.zero_()
    core = bundle.base.base.core
    patch = bundle.base.base.patch.agent_grid_i32
    core.occupancy.agent_id_grid.fill_(-1)
    core.occupancy.slot_grid.fill_(-1)
    patch.fill_(-1)
    # Separate all admitted parents so each has a disjoint adjacent placement
    # neighborhood under the source candidate order.
    for world in range(4):
        for slot in range(16):
            y = 2 + (slot // 4) * 6
            x = 2 + (slot % 4) * 6
            agent_id = int(core.agents.agent_id[world, slot])
            core.agents.pos_y[world, slot] = y
            core.agents.pos_x[world, slot] = x
            core.occupancy.agent_id_grid[world, y, x] = agent_id
            core.occupancy.slot_grid[world, y, x] = slot
            patch[world, y, x] = agent_id
    lifecycle = bundle.base.lifecycle
    lifecycle.death_candidate.zero_()
    lifecycle.death_candidate[:, 8:16] = True
    core.agents.health[:, 8:16:2] = 0.0
    core.agents.energy[:, :16] = 1000.0
    bundle.base.agent.policy_energy_observation[:, :16] = 1000.0
    bundle.base.agent.repro_threshold[:, :16] = 0.0
    replay = make_c5_pcg64_context_v1(bundle, streams=2, seed=seed)
    fixture = C4ResidentMaxloadR8FixtureV1(
        bundle=bundle,
        replay=replay,
        birth_config=TensorWorldC4BirthConfigV1(
            max_births=8,
            parent_energy_share=0.05,
            max_energy=1000.0,
            cooldown_steps=0,
            allow_mutation=False,
            require_spawn_rng_tape=True,
            generation=8,
        ),
        spawn_uniforms=torch.zeros((4, 32, 3), dtype=torch.float64),
        contract=c4_resident_maxload_r8_contract_v1(),
    )
    fixture.validate()
    return fixture


__all__ = [
    "C4_RESIDENT_MAXLOAD_R8_DECISION_V1",
    "C4_RESIDENT_MAXLOAD_R8_IDENTITY_LIMITATION_V1",
    "C4_RESIDENT_MAXLOAD_R8_SCHEMA_VERSION_V1",
    "C4ResidentMaxloadR8FixtureV1",
    "c4_resident_maxload_r8_contract_v1",
    "make_c4_resident_maxload_r8_fixture_v1",
]
