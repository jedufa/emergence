"""CPU/static R6 collision boundary mapped onto the resident R5 ABI.

The helper freezes adversarial B4 inputs and the exact authoritative tensor
groups consumed by R5.  It performs no CUDA allocation or native execution and
cannot provide resident parity or a C4 gate decision.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import torch

from .c4_mating import TensorWorldC4MateBirthConfigV1
from .c4_r6_adversarial import (
    make_c4_r6_adversarial_fixture_v1,
    validate_c4_r6_identity_slot_preflight_v1,
)
from .c4_resident_birth_r2 import (
    _half_tensors,
    _inherit_tensors,
    _minus_one_tensors,
    _one_tensors,
    _primary_tensors,
    _zero_tensors,
)
from .c4_resident_mutation_r4 import (
    _replay_tensors,
    _scalar_rng_tensors,
    _validate_replay_structure,
)
from .c4_resident_population_r5 import c4_resident_population_r5_source_receipt_v1
from .c4_resident_turnover_r3 import _death_authoritative
from .c4_state import TensorWorldStateBundleV4
from .c5_determinism import TensorWorldC5ReplayContextV1, make_c5_pcg64_context_v1
from .contracts import TensorWorldContractError


C4_RESIDENT_COLLISION_R6_SCHEMA_VERSION_V1: Final[str] = (
    "tensor_world_c4_resident_collision_r6_boundary_v1"
)
C4_RESIDENT_COLLISION_R6_DECISION_V1: Final[str] = "NO_GATE_DECISION"
C4_RESIDENT_COLLISION_R6_CASES_V1: Final[tuple[str, ...]] = (
    "b4_fragmentation_lineage_reuse",
    "duplicate_free_queue",
    "conflicting_occupancy",
    "conflicting_identity_map",
)
C4_RESIDENT_COLLISION_R6_ABI_COUNTS_V1: Final[dict[str, int]] = {
    "primary": 101,
    "death": 13,
    "inherit": 29,
    "zero": 55,
    "minus_one": 11,
    "half": 1,
    "one": 1,
    "scalar_rng": 8,
    "replay": 6,
}


@dataclass(slots=True)
class TensorWorldC4ResidentCollisionR6BoundaryV1:
    bundle: TensorWorldStateBundleV4
    config: TensorWorldC4MateBirthConfigV1
    rng_context: TensorWorldC5ReplayContextV1
    metadata: dict[str, object]
    schema_version: str = C4_RESIDENT_COLLISION_R6_SCHEMA_VERSION_V1


def c4_resident_collision_r6_source_receipt_v1() -> dict[str, object]:
    """Bind the static boundary to R5 and state the remaining native seam."""

    r5 = c4_resident_population_r5_source_receipt_v1()
    return {
        "schema_version": C4_RESIDENT_COLLISION_R6_SCHEMA_VERSION_V1,
        "route": "R6_CPU_STATIC_ADVERSARIAL_MAP_TO_R5_ABI",
        "decision": C4_RESIDENT_COLLISION_R6_DECISION_V1,
        "batch": 4,
        "r5_schema_version": r5["schema_version"],
        "r5_route": r5["route"],
        "authoritative_abi_counts": dict(C4_RESIDENT_COLLISION_R6_ABI_COUNTS_V1),
        "native_staging": "not_materialized_cpu_fallback_forbidden",
        "native_wrapper_required": True,
        "native_wrapper_reason": (
            "R5 validates identity-map and occupancy agreement for death "
            "candidates, not every live slot; R6 non-death collision cases "
            "require one batch-global semantic preflight before R5 commit"
        ),
        "gate_claim": False,
    }


def map_c4_resident_collision_r6_to_r5_abi_v1(
    bundle: TensorWorldStateBundleV4,
    rng_context: TensorWorldC5ReplayContextV1,
) -> dict[str, tuple[torch.Tensor, ...]]:
    """Return the exact CPU tensor groups corresponding to the R5 native ABI."""

    if not isinstance(bundle, TensorWorldStateBundleV4) or bundle.batch != 4:
        raise TensorWorldContractError("C4 R6 resident boundary requires a B4 V4 bundle")
    _validate_replay_structure(bundle, rng_context)
    groups = {
        "primary": tuple(_primary_tensors(bundle)),
        "death": tuple(_death_authoritative(bundle)),
        "inherit": tuple(_inherit_tensors(bundle)),
        "zero": tuple(_zero_tensors(bundle)),
        "minus_one": tuple(_minus_one_tensors(bundle)),
        "half": tuple(_half_tensors(bundle)),
        "one": tuple(_one_tensors(bundle)),
        "scalar_rng": tuple(_scalar_rng_tensors(bundle)),
        "replay": tuple(_replay_tensors(rng_context)),
    }
    actual_counts = {name: len(tensors) for name, tensors in groups.items()}
    if actual_counts != C4_RESIDENT_COLLISION_R6_ABI_COUNTS_V1:
        raise TensorWorldContractError("C4 R6 mapping drifted from the R5 native ABI")
    for tensors in groups.values():
        for tensor in tensors:
            if tensor.device != bundle.device or not tensor.is_contiguous():
                raise TensorWorldContractError(
                    "C4 R6 R5 ABI tensors must be contiguous and colocated"
                )
    return groups


def _place_r5_agents(bundle: TensorWorldStateBundleV4) -> None:
    core = bundle.base.base.core
    patch = bundle.base.base.patch.agent_grid_i32
    core.occupancy.agent_id_grid.fill_(-1)
    core.occupancy.slot_grid.fill_(-1)
    patch.fill_(-1)
    positions = (
        ((2, 2), (5, 5), (2, 1)),
        ((2, 2), (5, 5), (2, 1)),
        ((2, 2), (2, 1), (6, 6)),
        ((2, 2), (2, 1), (6, 6)),
    )
    for world, rows in enumerate(positions):
        for slot, (y, x) in enumerate(rows):
            agent_id = int(core.agents.agent_id[world, slot])
            core.agents.pos_y[world, slot] = y
            core.agents.pos_x[world, slot] = x
            core.occupancy.agent_id_grid[world, y, x] = agent_id
            core.occupancy.slot_grid[world, y, x] = slot
            patch[world, y, x] = agent_id


def make_c4_resident_collision_r6_boundary_v1(
    case: str,
    *,
    seed: int = 23,
) -> TensorWorldC4ResidentCollisionR6BoundaryV1:
    """Create one R5-compatible B4 mate/mutation boundary input."""

    if case not in C4_RESIDENT_COLLISION_R6_CASES_V1:
        raise ValueError(f"unknown C4 resident R6 case: {case}")
    bundle, _ordinary_config, _spawn_uniforms, base_metadata = (
        make_c4_r6_adversarial_fixture_v1(
            "b4_fragmentation_lineage_reuse", seed=seed
        )
    )
    _place_r5_agents(bundle)
    core = bundle.base.base.core
    core.agents.health[0, 1] = 0.0
    core.agents.health[1, 1] = 0.0
    bundle.base.genome.mutation_rate[:, :3] = 1.0
    if bundle.scalar_rng is None:
        raise TensorWorldContractError("C4 R6 requires scalar RNG state")
    bundle.scalar_rng.python_random_gauss_primary_tape.fill_(0.01)
    bundle.scalar_rng.python_random_gauss_secondary_tape.zero_()

    metadata = {
        **base_metadata,
        "case": case,
        "r5_expected_death_count": [1, 1, 0, 0],
        "r5_expected_birth_count": [1, 1, 1, 1],
        "r5_expected_child_slots": [6, 1, 3, 3],
        "r5_abi_counts": dict(C4_RESIDENT_COLLISION_R6_ABI_COUNTS_V1),
        "native_execution": False,
        "gate_claim": False,
    }
    if case == "duplicate_free_queue":
        bundle.population.free_queue[3].fill_(-1)
        bundle.population.free_queue[3, :2] = torch.tensor([3, 3])
        bundle.population.free_queue_count[3] = 2
        metadata["invalid_world"] = 3
    elif case == "conflicting_occupancy":
        y = int(core.agents.pos_y[2, 0])
        x = int(core.agents.pos_x[2, 0])
        core.occupancy.slot_grid[2, y, x] = 1
        metadata["invalid_world"] = 2
    elif case == "conflicting_identity_map":
        bundle.population.agent_id_to_slot[3, 1] = 2
        metadata["invalid_world"] = 3

    config = TensorWorldC4MateBirthConfigV1(
        max_births=1, allow_mutation=True, generation=6
    )
    rng_context = make_c5_pcg64_context_v1(bundle, streams=2, seed=seed)
    return TensorWorldC4ResidentCollisionR6BoundaryV1(
        bundle=bundle,
        config=config,
        rng_context=rng_context,
        metadata=metadata,
    )


def validate_c4_resident_collision_r6_boundary_v1(
    boundary: TensorWorldC4ResidentCollisionR6BoundaryV1,
) -> dict[str, tuple[torch.Tensor, ...]]:
    """Fail closed before any native load and return the frozen R5 ABI map."""

    if not isinstance(boundary, TensorWorldC4ResidentCollisionR6BoundaryV1):
        raise TensorWorldContractError("C4 R6 resident boundary type is unsupported")
    if boundary.schema_version != C4_RESIDENT_COLLISION_R6_SCHEMA_VERSION_V1:
        raise TensorWorldContractError("C4 R6 resident boundary schema is unsupported")
    if not boundary.config.allow_mutation or boundary.config.max_births != 1:
        raise TensorWorldContractError("C4 R6 requires the bounded R5 mate/mutation config")
    groups = map_c4_resident_collision_r6_to_r5_abi_v1(
        boundary.bundle, boundary.rng_context
    )
    validate_c4_r6_identity_slot_preflight_v1(boundary.bundle)
    return groups


__all__ = [
    "C4_RESIDENT_COLLISION_R6_ABI_COUNTS_V1",
    "C4_RESIDENT_COLLISION_R6_CASES_V1",
    "C4_RESIDENT_COLLISION_R6_DECISION_V1",
    "C4_RESIDENT_COLLISION_R6_SCHEMA_VERSION_V1",
    "TensorWorldC4ResidentCollisionR6BoundaryV1",
    "c4_resident_collision_r6_source_receipt_v1",
    "make_c4_resident_collision_r6_boundary_v1",
    "map_c4_resident_collision_r6_to_r5_abi_v1",
    "validate_c4_resident_collision_r6_boundary_v1",
]
