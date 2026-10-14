"""Campaign-neutral compact science capture ABI and genome reconstruction.

The native producer and offline evidence reducers share this low-dependency
contract.  Keeping the wire constants and hereditary reconstruction here
prevents an evidence reducer from importing a CUDA runtime implementation and
keeps physical execution details out of logical science ownership.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Final


COMPACT_SCIENCE_CAPTURE_SCHEMA_V1: Final[str] = (
    "eiel_full_evolution_cuda_compact_science_capture_v1"
)
COMPACT_SCIENCE_DISPATCH_SCHEMA_V1: Final[str] = (
    "eiel_cuda_v2_compact_science_dispatch_artifact_v1"
)
COMPACT_SCIENCE_SNAPSHOT_SCHEMA_V1: Final[str] = (
    "eiel_cuda_v2_compact_science_snapshot_artifact_v1"
)
COMPACT_SCIENCE_CHECKPOINT_FRONTIER_SCHEMA_V1: Final[str] = (
    "eiel_cuda_v2_compact_science_checkpoint_frontier_v1"
)
COMPACT_SCIENCE_AGENT_METRIC_FIELDS_V1: Final[tuple[str, ...]] = (
    "action_0_count",
    "action_1_count",
    "action_2_count",
    "action_3_count",
    "action_4_count",
    "action_5_count",
    "action_6_count",
    "alive_ticks",
    "reproduction_eligible_count",
    "mate_candidate_count",
    "birth_attempt_count",
    "successful_parent_count",
    "resource_uptake_total",
    "movement_cost_total",
    "movement_attempt_count",
    "movement_success_count",
    "health_damage_total",
    "energy_delta_total",
    "attack_attempt_count",
    "attack_hit_count",
    "births_as_parent_count",
)
COMPACT_SCIENCE_AGENT_METRIC_WIDTH_V1: Final[int] = len(
    COMPACT_SCIENCE_AGENT_METRIC_FIELDS_V1
)
COMPACT_SCIENCE_BIRTH_PROVENANCE_WIDTH_V1: Final[int] = 132
COMPACT_SCIENCE_BIRTH_PROVENANCE_ENCODING_V1: Final[str] = (
    "linked_locus_mutation_direction_u2_and_recombination_haplotype_u2_le_v1"
)
COMPACT_SCIENCE_DEATH_METRIC_ENCODING_V1: Final[str] = (
    "death_only_twenty_u32_plus_energy_delta_i64_le_v1"
)
COMPACT_SCIENCE_DEATH_METRIC_ROW_WIDTH_V1: Final[int] = 88
COMPACT_SCIENCE_PACKED_BIRTH_PROVENANCE_ENCODING_V1: Final[str] = (
    "birth_only_two_linked_loci_per_u8_plus_recombination_u32le_v1"
)
COMPACT_SCIENCE_PACKED_BIRTH_PROVENANCE_WIDTH_V1: Final[int] = 68
COMPACT_SCIENCE_DISPATCH_GZIP_ENCODING_V1: Final[str] = (
    "gzip_mtime0_level1_v1"
)


class ScienceCompactCaptureContractError(ValueError):
    """Compact science bytes do not satisfy their canonical ABI."""


def _compact_genome_projection(
    value: Mapping[str, Any], *, label: str
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    policy = value.get("policy")
    trait = value.get("trait")
    if (
        isinstance(policy, (str, bytes))
        or not isinstance(policy, Sequence)
        or len(policy) != 224
        or isinstance(trait, (str, bytes))
        or not isinstance(trait, Sequence)
        or len(trait) != 32
        or any(type(item) is not int or item < -128 or item > 127 for item in policy)
        or any(
            type(item) is not int or item < -32768 or item > 32767
            for item in trait
        )
    ):
        raise ScienceCompactCaptureContractError(
            f"{label} is not an exact native diploid genome projection"
        )
    return tuple(policy), tuple(trait)


def reconstruct_compact_birth_genome_v1(
    *,
    parent_a_genome: Mapping[str, Any],
    parent_b_genome: Mapping[str, Any],
    birth_provenance: Sequence[int] | Any,
    mutation_step: int,
) -> dict[str, Any]:
    """Reconstruct one exact child genome from generic compact observations."""

    if type(mutation_step) is not int or mutation_step < 1:
        raise ScienceCompactCaptureContractError("compact mutation step is invalid")
    parent_a_policy, parent_a_trait = _compact_genome_projection(
        parent_a_genome, label="parent A genome"
    )
    parent_b_policy, parent_b_trait = _compact_genome_projection(
        parent_b_genome, label="parent B genome"
    )
    try:
        raw = bytes(int(value) for value in birth_provenance)
    except Exception as exc:
        raise ScienceCompactCaptureContractError(
            "compact birth provenance is not a byte row"
        ) from exc
    if len(raw) != COMPACT_SCIENCE_BIRTH_PROVENANCE_WIDTH_V1:
        raise ScienceCompactCaptureContractError(
            "compact birth provenance has the wrong width"
        )
    mutation_bytes = raw[:128]
    if any(value & 0xF0 for value in mutation_bytes):
        raise ScienceCompactCaptureContractError(
            "compact birth provenance uses reserved mutation bits"
        )
    recombination = int.from_bytes(raw[128:132], "little")
    pre_policy = [0] * 224
    pre_trait = [0] * 32
    child_policy = [0] * 224
    child_trait = [0] * 32
    changes: list[dict[str, Any]] = []
    mutation_origin_count = 0
    saturated_noop_count = 0
    for linked_locus in range(128):
        block = linked_locus // 8
        haplotype_a = (recombination >> (2 * block)) & 1
        haplotype_b = (recombination >> (2 * block + 1)) & 1
        if linked_locus < 112:
            source_a = parent_a_policy
            source_b = parent_b_policy
            pre = pre_policy
            child = child_policy
            loci_per_haplotype = 112
            minimum = -127
            maximum = 127
            domain = "policy"
            locus = linked_locus
        else:
            source_a = parent_a_trait
            source_b = parent_b_trait
            pre = pre_trait
            child = child_trait
            loci_per_haplotype = 16
            minimum = -32767
            maximum = 32767
            domain = "trait"
            locus = linked_locus - 112
        inherited = (
            source_a[haplotype_a * loci_per_haplotype + locus],
            source_b[haplotype_b * loci_per_haplotype + locus],
        )
        mutation_byte = mutation_bytes[linked_locus]
        for allele, before in enumerate(inherited):
            index = allele * loci_per_haplotype + locus
            pre[index] = before
            code = (mutation_byte >> (2 * allele)) & 0x3
            if code == 3:
                raise ScienceCompactCaptureContractError(
                    "compact birth provenance uses the reserved mutation code"
                )
            direction = 0 if code == 0 else (1 if code == 1 else -1)
            after = max(minimum, min(maximum, before + direction * mutation_step))
            child[index] = after
            if direction:
                mutation_origin_count += 1
                saturated = after == before
                saturated_noop_count += int(saturated)
                changes.append(
                    {
                        "domain": domain,
                        "locus": locus,
                        "linked_locus": linked_locus,
                        "allele": allele,
                        "direction": direction,
                        "before": before,
                        "after": after,
                        "delta": after - before,
                        "saturated_noop": saturated,
                    }
                )
    return {
        "schema_version": "eiel_compact_birth_genome_reconstruction_v1",
        "birth_provenance_encoding": COMPACT_SCIENCE_BIRTH_PROVENANCE_ENCODING_V1,
        "recombination_choice_u32le": recombination,
        "pre_mutation_genome": {"policy": pre_policy, "trait": pre_trait},
        "child_genome": {"policy": child_policy, "trait": child_trait},
        "mutation_origin_count": mutation_origin_count,
        "changed_alleles": changes,
        "saturated_noop_count": saturated_noop_count,
    }


__all__ = (
    "COMPACT_SCIENCE_AGENT_METRIC_FIELDS_V1",
    "COMPACT_SCIENCE_AGENT_METRIC_WIDTH_V1",
    "COMPACT_SCIENCE_BIRTH_PROVENANCE_ENCODING_V1",
    "COMPACT_SCIENCE_BIRTH_PROVENANCE_WIDTH_V1",
    "COMPACT_SCIENCE_CAPTURE_SCHEMA_V1",
    "COMPACT_SCIENCE_CHECKPOINT_FRONTIER_SCHEMA_V1",
    "COMPACT_SCIENCE_DEATH_METRIC_ENCODING_V1",
    "COMPACT_SCIENCE_DEATH_METRIC_ROW_WIDTH_V1",
    "COMPACT_SCIENCE_DISPATCH_SCHEMA_V1",
    "COMPACT_SCIENCE_PACKED_BIRTH_PROVENANCE_ENCODING_V1",
    "COMPACT_SCIENCE_PACKED_BIRTH_PROVENANCE_WIDTH_V1",
    "COMPACT_SCIENCE_SNAPSHOT_SCHEMA_V1",
    "ScienceCompactCaptureContractError",
    "reconstruct_compact_birth_genome_v1",
)
