"""Frozen Genome V2 founder geometry for the RECOMB-72 successor."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Final, Mapping

from .genome_upgrade_v2 import (
    ALLELE_PARAMETERS,
    CONTEXT_COUNT,
    CONTRACT_SHA256,
    GENOME_UPGRADE_V2_INHERITANCE_VERSION,
    PARAMETER_COUNT,
    PRODUCT_CHANNELS,
    PRODUCT_COUNT,
    Q10_SCALE,
    GenomeUpgradeV2V1,
    express_reference_v1,
)


FOUNDER_CONTRACT_SCHEMA_VERSION: Final[str] = (
    "genome_upgrade_v2_recomb72_founder_contract_v1"
)
FOUNDER_PROFILE_PREFIX: Final[str] = "recomb72_gv2_v1"
FOUNDER_PATTERN_IDS: Final[tuple[str, ...]] = (
    "single_locus_extreme_pair",
    "context_conditional_hidden_pair",
    "correlated_multi_product_linkage_packages",
)
FOUNDER_STATE_IDS: Final[tuple[str, ...]] = (
    "all_heterozygous_ab",
    "balanced_homozygous_aa_bb",
    "homozygous_aa_control",
    "homozygous_bb_control",
)
FOUNDER_PROFILE_IDS: Final[tuple[str, ...]] = tuple(
    f"{FOUNDER_PROFILE_PREFIX}.{pattern_id}.{state_id}"
    for pattern_id in FOUNDER_PATTERN_IDS
    for state_id in FOUNDER_STATE_IDS
)


class GenomeUpgradeV2Recomb72FounderError(ValueError):
    """A frozen RECOMB-72-GV2 founder identity is invalid."""


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _neutral_haplotype() -> tuple[tuple[tuple[int, ...], ...], tuple[int, ...]]:
    neutral = GenomeUpgradeV2V1.neutral()
    return neutral.alleles[0], neutral.modifiers[0]


def _pattern_haplotypes(
    pattern_id: str,
) -> tuple[
    tuple[tuple[tuple[int, ...], ...], tuple[int, ...]],
    tuple[tuple[tuple[int, ...], ...], tuple[int, ...]],
    tuple[dict[str, Any], ...],
]:
    if pattern_id not in FOUNDER_PATTERN_IDS:
        raise GenomeUpgradeV2Recomb72FounderError(
            f"unsupported RECOMB-72-GV2 founder pattern: {pattern_id!r}"
        )
    neutral_alleles, neutral_modifiers = _neutral_haplotype()
    allele_a = [list(row) for row in neutral_alleles]
    allele_b = [list(row) for row in neutral_alleles]
    audited: list[dict[str, Any]] = []

    def vary(product: int, parameter: int, value_a: int, value_b: int) -> None:
        allele_a[product][parameter] = value_a
        allele_b[product][parameter] = value_b
        audited.append(
            {
                "locus_id": (
                    f"allele.{PRODUCT_CHANNELS[product]}."
                    f"{ALLELE_PARAMETERS[parameter]}"
                ),
                "product_index": product,
                "parameter_index": parameter,
                "linkage_block": product // 2,
                "endpoint_a_q10": value_a,
                "endpoint_b_q10": value_b,
            }
        )

    if pattern_id == "single_locus_extreme_pair":
        vary(0, 0, -Q10_SCALE, Q10_SCALE)
    elif pattern_id == "context_conditional_hidden_pair":
        vary(5, 3, -Q10_SCALE, Q10_SCALE)
    else:
        for product in range(PRODUCT_COUNT):
            value_a = Q10_SCALE // 2 if product % 2 == 0 else -(Q10_SCALE // 2)
            vary(product, 0, value_a, -value_a)

    haplotype_a = (
        tuple(tuple(row) for row in allele_a),
        tuple(neutral_modifiers),
    )
    haplotype_b = (
        tuple(tuple(row) for row in allele_b),
        tuple(neutral_modifiers),
    )
    return haplotype_a, haplotype_b, tuple(audited)


def _genome(
    left: tuple[tuple[tuple[int, ...], ...], tuple[int, ...]],
    right: tuple[tuple[tuple[int, ...], ...], tuple[int, ...]],
) -> GenomeUpgradeV2V1:
    return GenomeUpgradeV2V1(
        alleles=(left[0], right[0]),
        modifiers=(left[1], right[1]),
    )


def _flatten_haplotype(
    value: tuple[tuple[tuple[int, ...], ...], tuple[int, ...]],
) -> list[int]:
    return [
        *(
            item
            for product in value[0]
            for item in product
        ),
        *value[1],
    ]


def flatten_genome_q10_v1(genome: GenomeUpgradeV2V1) -> tuple[int, ...]:
    """Return the exact resident initializer layout for one diploid genome."""

    return tuple(
        [
            *(
                value
                for homolog in genome.alleles
                for product in homolog
                for value in product
            ),
            *(value for homolog in genome.modifiers for value in homolog),
        ]
    )


def _genome_from_flattened(values: list[int]) -> GenomeUpgradeV2V1:
    allele_width = 2 * PRODUCT_COUNT * PARAMETER_COUNT
    return GenomeUpgradeV2V1(
        alleles=tuple(
            tuple(
                tuple(
                    values[
                        (
                            homolog * PRODUCT_COUNT * PARAMETER_COUNT
                            + product * PARAMETER_COUNT
                        ):
                        (
                            homolog * PRODUCT_COUNT * PARAMETER_COUNT
                            + (product + 1) * PARAMETER_COUNT
                        )
                    ]
                )
                for product in range(PRODUCT_COUNT)
            )
            for homolog in range(2)
        ),
        modifiers=tuple(
            tuple(
                values[
                    allele_width + homolog * PRODUCT_COUNT:
                    allele_width + (homolog + 1) * PRODUCT_COUNT
                ]
            )
            for homolog in range(2)
        ),
    )


def founder_profile_id_v1(pattern_id: str, state_id: str) -> str:
    if pattern_id not in FOUNDER_PATTERN_IDS:
        raise GenomeUpgradeV2Recomb72FounderError(
            f"unsupported RECOMB-72-GV2 founder pattern: {pattern_id!r}"
        )
    if state_id not in FOUNDER_STATE_IDS:
        raise GenomeUpgradeV2Recomb72FounderError(
            f"unsupported RECOMB-72-GV2 founder state: {state_id!r}"
        )
    return f"{FOUNDER_PROFILE_PREFIX}.{pattern_id}.{state_id}"


def _profile_parts(profile_id: str) -> tuple[str, str]:
    if profile_id not in FOUNDER_PROFILE_IDS:
        raise GenomeUpgradeV2Recomb72FounderError(
            f"unsupported RECOMB-72-GV2 founder profile: {profile_id!r}"
        )
    prefix, pattern_id, state_id = profile_id.split(".", maxsplit=2)
    if prefix != FOUNDER_PROFILE_PREFIX:
        raise GenomeUpgradeV2Recomb72FounderError(
            "RECOMB-72-GV2 founder profile prefix differs"
        )
    return pattern_id, state_id


def _state_assignment(state_id: str) -> dict[str, Any]:
    if state_id == "all_heterozygous_ab":
        return {
            "algorithm": "every_active_slot_ab_v1",
            "requires_even_population": False,
            "genotype_fractions": {"ab": [1, 1]},
            "endpoint_copy_fractions": {"a": [1, 2], "b": [1, 2]},
        }
    if state_id == "balanced_homozygous_aa_bb":
        return {
            "algorithm": "ascending_active_slot_rank_first_half_aa_v1",
            "requires_even_population": True,
            "genotype_fractions": {"aa": [1, 2], "bb": [1, 2]},
            "endpoint_copy_fractions": {"a": [1, 2], "b": [1, 2]},
        }
    endpoint = "a" if state_id == "homozygous_aa_control" else "b"
    return {
        "algorithm": f"every_active_slot_{endpoint}{endpoint}_v1",
        "requires_even_population": False,
        "genotype_fractions": {f"{endpoint}{endpoint}": [1, 1]},
        "endpoint_copy_fractions": {
            "a": [1, 1] if endpoint == "a" else [0, 1],
            "b": [1, 1] if endpoint == "b" else [0, 1],
        },
    }


def resolve_recomb72_gv2_founder_profile_v1(
    profile_id: str,
) -> dict[str, Any]:
    """Resolve one immutable founder pattern and population-state identity."""

    pattern_id, state_id = _profile_parts(profile_id)
    haplotype_a, haplotype_b, audited = _pattern_haplotypes(pattern_id)
    aa = _genome(haplotype_a, haplotype_a)
    ab = _genome(haplotype_a, haplotype_b)
    bb = _genome(haplotype_b, haplotype_b)
    zero_context = (0,) * CONTEXT_COUNT
    zero_context_phenotypes = {
        "aa": list(express_reference_v1(aa, zero_context).phenotypes_q10),
        "ab": list(express_reference_v1(ab, zero_context).phenotypes_q10),
        "bb": list(express_reference_v1(bb, zero_context).phenotypes_q10),
    }
    payload: dict[str, Any] = {
        "schema_version": FOUNDER_CONTRACT_SCHEMA_VERSION,
        "profile_id": profile_id,
        "pattern_id": pattern_id,
        "state_id": state_id,
        "genome_contract_sha256": CONTRACT_SHA256,
        "inheritance_version": GENOME_UPGRADE_V2_INHERITANCE_VERSION,
        "linkage_block_by_product": [
            product // 2 for product in range(PRODUCT_COUNT)
        ],
        "haplotype_a_q10": _flatten_haplotype(haplotype_a),
        "haplotype_b_q10": _flatten_haplotype(haplotype_b),
        "haplotype_a_sha256": _canonical_sha256(
            _flatten_haplotype(haplotype_a)
        ),
        "haplotype_b_sha256": _canonical_sha256(
            _flatten_haplotype(haplotype_b)
        ),
        "genotype_q10": {
            "aa": list(flatten_genome_q10_v1(aa)),
            "ab": list(flatten_genome_q10_v1(ab)),
            "bb": list(flatten_genome_q10_v1(bb)),
        },
        "genotype_sha256": {
            "aa": aa.to_payload()["genotype_sha256"],
            "ab": ab.to_payload()["genotype_sha256"],
            "bb": bb.to_payload()["genotype_sha256"],
        },
        "audited_loci": [dict(row) for row in audited],
        "population_assignment": _state_assignment(state_id),
        "zero_context_q10": list(zero_context),
        "zero_context_phenotypes_q10": zero_context_phenotypes,
        "zero_context_exact_phenotype_equivalence": (
            len(
                {
                    tuple(values)
                    for values in zero_context_phenotypes.values()
                }
            )
            == 1
        ),
        "mutation_required": False,
        "raw_agent_rows_retained": False,
    }
    return {**payload, "payload_sha256": _canonical_sha256(payload)}


def founder_population_geometry_v1(
    profile_id: str,
    active_count: int,
) -> dict[str, Any]:
    """Project exact G0 counts for one active population without RNG."""

    if (
        isinstance(active_count, bool)
        or not isinstance(active_count, int)
        or active_count <= 0
    ):
        raise GenomeUpgradeV2Recomb72FounderError(
            "founder active_count must be a positive integer"
        )
    profile = resolve_recomb72_gv2_founder_profile_v1(profile_id)
    state_id = profile["state_id"]
    if state_id == "all_heterozygous_ab":
        genotype_counts = {"aa": 0, "ab": active_count, "bb": 0}
    elif state_id == "balanced_homozygous_aa_bb":
        if active_count % 2:
            raise GenomeUpgradeV2Recomb72FounderError(
                "balanced founder state requires an even active population"
            )
        genotype_counts = {
            "aa": active_count // 2,
            "ab": 0,
            "bb": active_count // 2,
        }
    elif state_id == "homozygous_aa_control":
        genotype_counts = {"aa": active_count, "ab": 0, "bb": 0}
    else:
        genotype_counts = {"aa": 0, "ab": 0, "bb": active_count}
    endpoint_a_copies = 2 * genotype_counts["aa"] + genotype_counts["ab"]
    endpoint_b_copies = 2 * genotype_counts["bb"] + genotype_counts["ab"]
    payload = {
        "schema_version": "genome_upgrade_v2_recomb72_founder_geometry_v1",
        "profile_id": profile_id,
        "profile_sha256": profile["payload_sha256"],
        "assignment_algorithm": profile["population_assignment"]["algorithm"],
        "active_count": active_count,
        "genotype_counts": genotype_counts,
        "endpoint_copy_counts": {
            "a": endpoint_a_copies,
            "b": endpoint_b_copies,
        },
        "total_endpoint_copies": 2 * active_count,
        "rng_draws": 0,
    }
    return {**payload, "payload_sha256": _canonical_sha256(payload)}


def founder_population_genotypes_v1(
    profile_id: str,
    active_count: int,
) -> tuple[GenomeUpgradeV2V1, ...]:
    """Materialize genomes in ascending active-slot rank order."""

    profile = resolve_recomb72_gv2_founder_profile_v1(profile_id)
    geometry = founder_population_geometry_v1(profile_id, active_count)
    genotypes = {
        name: _genome_from_flattened(values)
        for name, values in (
            (key, profile["genotype_q10"][key])
            for key in ("aa", "ab", "bb")
        )
    }
    counts: Mapping[str, int] = geometry["genotype_counts"]
    ordered: list[GenomeUpgradeV2V1] = []
    for genotype_id in ("aa", "ab", "bb"):
        ordered.extend([genotypes[genotype_id]] * counts[genotype_id])
    if len(ordered) != active_count:
        raise GenomeUpgradeV2Recomb72FounderError(
            "founder population materialization count differs"
        )
    return tuple(ordered)


__all__ = [
    "FOUNDER_CONTRACT_SCHEMA_VERSION",
    "FOUNDER_PATTERN_IDS",
    "FOUNDER_PROFILE_IDS",
    "FOUNDER_PROFILE_PREFIX",
    "FOUNDER_STATE_IDS",
    "GenomeUpgradeV2Recomb72FounderError",
    "flatten_genome_q10_v1",
    "founder_population_genotypes_v1",
    "founder_population_geometry_v1",
    "founder_profile_id_v1",
    "resolve_recomb72_gv2_founder_profile_v1",
]
