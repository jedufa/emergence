"""Generic exact-state extraction and replay helpers for science requests.

This module only translates authenticated capture representations.  It never
changes an allele, fills a missing locus, or chooses a full genome from a
tracked projection.  Callers must select an explicit returned candidate before
building an exact natural source-state reference.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import struct
from typing import Any

from emergence_contracts import (
    canonical_sha256,
    full_genome_bytes_v1,
    full_genome_state_id_v1,
    tracked_genotype_state_id_v1,
)
from emergence_contracts.science_compact_capture import (
    reconstruct_compact_birth_genome_v1,
)


_NATIVE_GENOME_STRUCT_V1 = struct.Struct("<224b32h")
_PACKED_BIRTH_PROVENANCE_WIDTH_V1 = 68
_FULL_BIRTH_PROVENANCE_WIDTH_V1 = 132


class ExactNaturalSourceStateRecoveryError(ValueError):
    """An extraction or replay input is not exact."""


def native_genome_bytes_from_row_v1(row: Mapping[str, Any]) -> bytes:
    """Encode one capture genome row without changing its values."""

    policy = row.get("policy")
    trait = row.get("trait")
    if (
        isinstance(policy, (str, bytes))
        or not isinstance(policy, Sequence)
        or len(policy) != 224
        or isinstance(trait, (str, bytes))
        or not isinstance(trait, Sequence)
        or len(trait) != 32
    ):
        raise ExactNaturalSourceStateRecoveryError(
            "capture genome row does not contain the complete native genome"
        )
    try:
        encoded = _NATIVE_GENOME_STRUCT_V1.pack(
            *(int(value) for value in policy),
            *(int(value) for value in trait),
        )
    except (TypeError, ValueError, struct.error) as error:
        raise ExactNaturalSourceStateRecoveryError(
            "capture genome row contains values outside the native ABI"
        ) from error
    expected = row.get("genome_state_id")
    if expected is not None and expected != full_genome_state_id_v1(encoded):
        raise ExactNaturalSourceStateRecoveryError(
            "capture genome row identity does not bind its alleles"
        )
    return encoded


def enumerate_exact_natural_snapshot_candidates_v1(
    *,
    catalog_rows: Sequence[Mapping[str, Any]],
    active_rows: Sequence[Mapping[str, Any]],
    tracked_identity: str,
    policy_loci: Sequence[int],
    trait_loci: Sequence[int],
) -> tuple[dict[str, Any], ...]:
    """Return all exact active candidates for an explicit tracked projection.

    Multiple full states are deliberately returned.  The function does not
    collapse them into one tracked genotype and does not provide a default
    selection rule.
    """

    if not isinstance(tracked_identity, str) or not tracked_identity.startswith(
        "sha256:"
    ):
        raise ExactNaturalSourceStateRecoveryError("tracked identity is invalid")
    catalog: dict[str, bytes] = {}
    for row in catalog_rows:
        genome_state_id = row.get("genome_state_id")
        encoded = native_genome_bytes_from_row_v1(row)
        observed = full_genome_state_id_v1(encoded)
        if genome_state_id != observed or genome_state_id in catalog:
            raise ExactNaturalSourceStateRecoveryError(
                "snapshot catalog contains an ambiguous or unbound genome row"
            )
        catalog[genome_state_id] = encoded
    candidates: list[dict[str, Any]] = []
    seen_agents: set[int] = set()
    for row in active_rows:
        agent_id = row.get("agent_id")
        genome_state_id = row.get("genome_state_id")
        if type(agent_id) is not int or agent_id <= 0 or agent_id in seen_agents:
            raise ExactNaturalSourceStateRecoveryError(
                "active snapshot contains an ambiguous agent identity"
            )
        seen_agents.add(agent_id)
        encoded = catalog.get(genome_state_id)
        if encoded is None:
            raise ExactNaturalSourceStateRecoveryError(
                "active snapshot refers to an absent exact genome row"
            )
        if (
            tracked_genotype_state_id_v1(
                encoded, policy_loci=policy_loci, trait_loci=trait_loci
            )
            != tracked_identity
        ):
            continue
        candidates.append(
            {
                "agent": dict(row),
                "genome_state_id": genome_state_id,
                "genome_bytes": encoded,
                "tracked_genotype_state_id": tracked_identity,
            }
        )
    return tuple(
        sorted(
            candidates,
            key=lambda item: (
                item["genome_state_id"],
                int(item["agent"]["agent_id"]),
            ),
        )
    )


def expand_packed_birth_provenance_v1(value: bytes) -> bytes:
    """Expand the authenticated compact birth row to the replay ABI."""

    if not isinstance(value, bytes) or len(value) != _PACKED_BIRTH_PROVENANCE_WIDTH_V1:
        raise ExactNaturalSourceStateRecoveryError(
            "packed birth provenance must be exactly 68 bytes"
        )
    result = bytearray(_FULL_BIRTH_PROVENANCE_WIDTH_V1)
    for index, packed in enumerate(value[:64]):
        if packed & 0xF0:
            raise ExactNaturalSourceStateRecoveryError(
                "packed birth provenance uses reserved mutation bits"
            )
        result[2 * index] = packed & 0x0F
        result[2 * index + 1] = packed >> 4
    result[128:132] = value[64:68]
    return bytes(result)


def _compact_genome_from_bytes(value: bytes) -> dict[str, list[int]]:
    if not isinstance(value, bytes) or len(value) != _NATIVE_GENOME_STRUCT_V1.size:
        raise ExactNaturalSourceStateRecoveryError(
            "native genome bytes must be exactly 288 bytes"
        )
    values = _NATIVE_GENOME_STRUCT_V1.unpack(value)
    return {"policy": list(values[:224]), "trait": list(values[224:])}


def _genome_bytes_from_input(value: Mapping[str, Any]) -> bytes:
    if isinstance(value.get("genome"), Mapping):
        try:
            return full_genome_bytes_v1(value)
        except (TypeError, ValueError) as error:
            raise ExactNaturalSourceStateRecoveryError(
                "source reference carries an invalid exact genome"
            ) from error
    return native_genome_bytes_from_row_v1(value)


def reconstruct_exact_natural_birth_v1(
    *,
    parent_a: Mapping[str, Any],
    parent_b: Mapping[str, Any],
    packed_birth_provenance: bytes,
    mutation_step: int,
    expected_child_genome_state_id: str | None = None,
) -> dict[str, Any]:
    """Reconstruct and authenticate one later-born natural hereditary state."""

    parent_a_bytes = _genome_bytes_from_input(parent_a)
    parent_b_bytes = _genome_bytes_from_input(parent_b)
    try:
        reconstructed = reconstruct_compact_birth_genome_v1(
            parent_a_genome=_compact_genome_from_bytes(parent_a_bytes),
            parent_b_genome=_compact_genome_from_bytes(parent_b_bytes),
            birth_provenance=expand_packed_birth_provenance_v1(
                packed_birth_provenance
            ),
            mutation_step=mutation_step,
        )
        child = _NATIVE_GENOME_STRUCT_V1.pack(
            *(int(value) for value in reconstructed["child_genome"]["policy"]),
            *(int(value) for value in reconstructed["child_genome"]["trait"]),
        )
    except (TypeError, ValueError, struct.error) as error:
        raise ExactNaturalSourceStateRecoveryError(
            f"authenticated birth replay failed: {error}"
        ) from error
    child_identity = full_genome_state_id_v1(child)
    if (
        expected_child_genome_state_id is not None
        and child_identity != expected_child_genome_state_id
    ):
        raise ExactNaturalSourceStateRecoveryError(
            "replayed child genome identity differs from the expected state"
        )
    return {
        "child_genome_state_id": child_identity,
        "child_genome_bytes": child,
        "birth_provenance_sha256": hashlib.sha256(
            packed_birth_provenance
        ).hexdigest(),
        "mutation_origin_count": int(reconstructed["mutation_origin_count"]),
        "saturated_noop_count": int(reconstructed["saturated_noop_count"]),
    }


def replayed_hereditary_agent_state_id_v1(
    *,
    world_id: str,
    agent_id: int,
    parent_agent_ids: Sequence[int],
    birth_tick: int,
    genome_state_id: str,
) -> str:
    """Bind a replayed hereditary state when no dynamic birth snapshot exists.

    This is intentionally named as a replayed hereditary identity.  It is not
    presented as a native transient energy/position snapshot identity, which
    the compact capture does not retain at every birth boundary.
    """

    if (
        not isinstance(world_id, str)
        or type(agent_id) is not int
        or agent_id <= 0
        or type(birth_tick) is not int
        or birth_tick < 1
        or not isinstance(parent_agent_ids, Sequence)
        or len(parent_agent_ids) != 2
        or any(type(value) is not int or value < 0 for value in parent_agent_ids)
        or not isinstance(genome_state_id, str)
        or not genome_state_id.startswith("sha256:")
    ):
        raise ExactNaturalSourceStateRecoveryError(
            "replayed hereditary agent identity inputs are invalid"
        )
    core = {
        "schema_version": "eiel_replayed_hereditary_agent_state_v1",
        "world_id": world_id,
        "agent_id": agent_id,
        "parent_agent_ids": list(parent_agent_ids),
        "birth_tick": birth_tick,
        "genome_state_id": genome_state_id,
    }
    return "sha256:" + canonical_sha256(core).hex


__all__ = (
    "ExactNaturalSourceStateRecoveryError",
    "enumerate_exact_natural_snapshot_candidates_v1",
    "expand_packed_birth_provenance_v1",
    "native_genome_bytes_from_row_v1",
    "reconstruct_exact_natural_birth_v1",
    "replayed_hereditary_agent_state_id_v1",
)
