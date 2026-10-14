"""Frozen authority contract for the resident C1+C2+C3+C4 owner.

The historical R11 audit found several independent mutable projections.  This
module is the repair seam: it describes one canonical resident owner for each
cross-layer surface and makes every C3-facing projection explicitly derived
staging.  It contains no execution fallback and never authorizes the old
host-side C3 -> R9 loop.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Final

from .contracts import TensorWorldContractError


C4_RESIDENT_AUTHORITY_R11_SCHEMA_VERSION_V1: Final[str] = (
    "tensor_world_c4_resident_authority_r11_v1"
)


@dataclass(frozen=True, slots=True)
class TensorWorldResidentAuthorityFieldV1:
    """One canonical mutable field and its derived native projections."""

    name: str
    canonical_owner: str
    dtype: str
    projection: str
    mutation_phase: str
    derived: bool
    commit_order: int


# The order is the device transaction order, not Python field order.  A C3
# projection may be written by mechanics, but it is never an independent
# authority: the composed owner refreshes it from the canonical V4 field and
# copies its validated result back at the declared commit boundary.
_AUTHORITY_FIELDS: Final[tuple[TensorWorldResidentAuthorityFieldV1, ...]] = (
    TensorWorldResidentAuthorityFieldV1(
        "positions",
        "base.base.core.agents.pos_y/pos_x",
        "int64",
        "r11.c3_projection.pos_y_i32/pos_x_i32; device cast only",
        "c1_c2_c3_mechanics_then_position_commit",
        True,
        10,
    ),
    TensorWorldResidentAuthorityFieldV1(
        "alive_mask",
        "base.base.core.agents.alive",
        "bool",
        "r11.c3_projection.alive; device staging",
        "c3_eligibility_then_c4_commit",
        True,
        20,
    ),
    TensorWorldResidentAuthorityFieldV1(
        "agent_identity",
        "base.base.core.agents.agent_id",
        "int64",
        "r11.c3_projection.agent_id; device staging",
        "c4_identity_commit",
        True,
        30,
    ),
    TensorWorldResidentAuthorityFieldV1(
        "slot_tokens",
        "population.slot_tokens",
        "int64",
        "r11.c3_projection.slot_tokens; device read-only staging",
        "c4_identity_commit",
        True,
        31,
    ),
    TensorWorldResidentAuthorityFieldV1(
        "occupancy",
        "base.base.core.occupancy.agent_id_grid/slot_grid",
        "int64",
        "base.base.patch.agent_grid_i32; device projection",
        "c1_c2_mechanics_then_c4_commit",
        True,
        32,
    ),
    TensorWorldResidentAuthorityFieldV1(
        "lifecycle_eligibility",
        "base.lifecycle.death_candidate/death_reason",
        "bool/int32",
        "r11.c3_projection.death_outputs; device translation",
        "c3_eligibility",
        True,
        33,
    ),
    TensorWorldResidentAuthorityFieldV1(
        "typed_events",
        "events.*",
        "typed fixed-column tape",
        "c3 event rows are device staged, never a second tape",
        "c4_event_commit",
        True,
        40,
    ),
    TensorWorldResidentAuthorityFieldV1(
        "genome_execution",
        "base.genome.*",
        "source-declared dtypes",
        "r11.c3_projection.genome_views; device read-only/derived",
        "c3_mechanics_then_c4_birth_commit",
        True,
        41,
    ),
    TensorWorldResidentAuthorityFieldV1(
        "policy_and_adaptation",
        "base.base.policy.* + base.adaptation.*",
        "source-declared dtypes",
        "r11.c3_projection.policy_views; device derived",
        "c1_c2_c3_mechanics_then_c4_birth_commit",
        True,
        42,
    ),
    TensorWorldResidentAuthorityFieldV1(
        "pcg64_replay",
        "c5.rng_state_words/buffer_words/cursors/semantic_draw_counters",
        "int64",
        "r11.c3_projection.runner/env stream views; device derived",
        "c3_and_c4_rng_commit",
        True,
        43,
    ),
    TensorWorldResidentAuthorityFieldV1(
        "c4_scalar_rng",
        "scalar_rng.python_random_* + mutation_py_semantic_draws",
        "uint32/int64/bool/f64",
        "r11.c4_mutation_shadow; device write-ahead",
        "c4_rng_commit",
        True,
        44,
    ),
    TensorWorldResidentAuthorityFieldV1(
        "next_agent_id",
        "population.next_agent_id",
        "int64",
        "r11.c4_identity_shadow",
        "c4_identity_commit",
        True,
        45,
    ),
    TensorWorldResidentAuthorityFieldV1(
        "lineage",
        "population.lineage_* + patch_habitat_lineage_id",
        "int64/int32",
        "r11.c4_lineage_shadow",
        "c4_identity_commit",
        True,
        46,
    ),
    TensorWorldResidentAuthorityFieldV1(
        "free_slots",
        "population.free_queue/free_queue_count + pending_free_*",
        "int64",
        "r11.c4_virtual_queue_shadow",
        "c4_identity_commit",
        True,
        47,
    ),
)


def c4_resident_authority_fields_r11_v1() -> tuple[TensorWorldResidentAuthorityFieldV1, ...]:
    """Return the immutable authority map in transaction order."""

    return _AUTHORITY_FIELDS


def validate_c4_resident_authority_map_r11_v1() -> None:
    """Reject duplicate mutable owners and unsafe projection declarations."""

    fields = c4_resident_authority_fields_r11_v1()
    names = [field.name for field in fields]
    owners = [field.canonical_owner for field in fields]
    orders = [field.commit_order for field in fields]
    if len(names) != len(set(names)):
        raise TensorWorldContractError("R11 authority map contains duplicate surfaces")
    if len(owners) != len(set(owners)):
        raise TensorWorldContractError("R11 authority map contains duplicate mutable owners")
    if len(orders) != len(set(orders)):
        raise TensorWorldContractError("R11 authority commit order is not unique")
    if orders != sorted(orders):
        raise TensorWorldContractError("R11 authority fields are not in commit order")
    for field in fields:
        if not field.derived:
            raise TensorWorldContractError(
                f"R11 projection must be derived, not mutable: {field.name}"
            )
        if "c3" in field.projection and "device" not in field.projection:
            raise TensorWorldContractError(
                f"R11 C3 projection is not explicitly device-local: {field.name}"
            )
    positions = next(field for field in fields if field.name == "positions")
    if positions.dtype != "int64" or "pos_y_i32" not in positions.projection:
        raise TensorWorldContractError("R11 positions must retain V4 int64 authority")


def c4_resident_authority_map_receipt_r11_v1() -> dict[str, object]:
    """Return a serializable source-bound receipt for the frozen map."""

    validate_c4_resident_authority_map_r11_v1()
    return {
        "schema_version": C4_RESIDENT_AUTHORITY_R11_SCHEMA_VERSION_V1,
        "authority_count": len(_AUTHORITY_FIELDS),
        "canonical_mutable_owner_count": len(
            {field.canonical_owner for field in _AUTHORITY_FIELDS}
        ),
        "projection_policy": "all_native_projections_are_device_local_derived_staging",
        "commit_order": [field.name for field in _AUTHORITY_FIELDS],
        "fields": [asdict(field) for field in _AUTHORITY_FIELDS],
        "host_tick_authority": False,
        "host_snapshot_rollback": False,
        "gate_claim": False,
    }


__all__ = [
    "C4_RESIDENT_AUTHORITY_R11_SCHEMA_VERSION_V1",
    "TensorWorldResidentAuthorityFieldV1",
    "c4_resident_authority_fields_r11_v1",
    "c4_resident_authority_map_receipt_r11_v1",
    "validate_c4_resident_authority_map_r11_v1",
]
