from __future__ import annotations

"""Bounded aggregate telemetry for diploid scalar genomes and receipts."""

from collections import Counter
import math
from typing import Any, Iterable, Mapping

from evolution.diploid_inheritance_v1 import (
    DIPLOID_LINKAGE_BLOCKS,
    DIPLOID_SCALAR_LOCI,
    diploid_genome_from_genome,
)


DIPLOID_TELEMETRY_SCHEMA_VERSION = "diploid_aggregate_telemetry_v1"


def _finite_mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def _variance(values: list[float]) -> float | None:
    if not values:
        return None
    mean = sum(values) / len(values)
    return float(sum((value - mean) ** 2 for value in values) / len(values))


def _frequency_metrics(values: list[float]) -> dict[str, Any]:
    if not values:
        return {
            "allele_count": 0,
            "effective_allele_count": None,
            "largest_allele_frequency": None,
            "fixed": None,
        }
    counts = Counter(values)
    total = float(len(values))
    frequencies = [count / total for count in counts.values()]
    return {
        "allele_count": len(counts),
        "effective_allele_count": float(
            1.0 / sum(frequency * frequency for frequency in frequencies)
        ),
        "largest_allele_frequency": float(max(frequencies)),
        "fixed": len(counts) == 1,
    }


def summarize_diploid_population(genomes: Iterable[Any]) -> dict[str, Any]:
    rows = [diploid_genome_from_genome(genome) for genome in genomes]
    loci: dict[str, Any] = {}
    for locus_index, locus in enumerate(DIPLOID_SCALAR_LOCI):
        allele_0 = [float(row.allele_0[locus_index]) for row in rows]
        allele_1 = [float(row.allele_1[locus_index]) for row in rows]
        alleles = allele_0 + allele_1
        phenotype = [
            0.5 * (left + right)
            for left, right in zip(allele_0, allele_1)
        ]
        heterozygotes = sum(
            left != right for left, right in zip(allele_0, allele_1)
        )
        genotype_variance = _variance(alleles)
        phenotype_variance = _variance(phenotype)
        loci[locus] = {
            **_frequency_metrics(alleles),
            "allele_mean": _finite_mean(alleles),
            "genotype_variance": genotype_variance,
            "phenotype_variance": phenotype_variance,
            "phenotype_genotype_variance_ratio": (
                float(phenotype_variance / genotype_variance)
                if genotype_variance not in {None, 0.0}
                and phenotype_variance is not None
                else None
            ),
            "homozygote_frequency": (
                float((len(rows) - heterozygotes) / len(rows))
                if rows
                else None
            ),
            "heterozygote_frequency": (
                float(heterozygotes / len(rows)) if rows else None
            ),
            "observed_heterozygosity": (
                float(heterozygotes / len(rows)) if rows else None
            ),
        }

    blocks: dict[str, Any] = {}
    locus_index = {
        locus: index for index, locus in enumerate(DIPLOID_SCALAR_LOCI)
    }
    for block_id, block_loci in DIPLOID_LINKAGE_BLOCKS:
        haplotypes: list[tuple[float, ...]] = []
        for row in rows:
            for homolog in (row.allele_0, row.allele_1):
                haplotypes.append(
                    tuple(homolog[locus_index[locus]] for locus in block_loci)
                )
        metrics = _frequency_metrics(haplotypes)
        blocks[block_id] = {
            "haplotype_count": metrics["allele_count"],
            "effective_haplotype_count": metrics["effective_allele_count"],
            "largest_haplotype_frequency": metrics[
                "largest_allele_frequency"
            ],
            "fixed": metrics["fixed"],
        }

    return {
        "schema_version": DIPLOID_TELEMETRY_SCHEMA_VERSION,
        "population_count": len(rows),
        "allele_copy_count": 2 * len(rows),
        "loci": loci,
        "linkage_blocks": blocks,
    }


def summarize_transmission_receipts(
    receipts: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    rows = [dict(receipt) for receipt in receipts]
    homolog_counts = Counter()
    parent_counts = Counter()
    semantic_mutations = 0
    mutation_events = 0
    for receipt in rows:
        for donor in receipt.get("meiosis", {}).get("donors", ()):
            homolog_counts[int(donor["homolog"])] += 1
            parent_counts[str(donor["parent"])] += 1
        for event in receipt.get("mutation", {}).get("events", ()):
            mutation_events += 1
            semantic_mutations += int(bool(event.get("semantic_change", False)))
    homolog_total = sum(homolog_counts.values())
    parent_total = sum(parent_counts.values())
    return {
        "schema_version": "diploid_transmission_telemetry_v1",
        "receipt_count": len(rows),
        "homolog_0_transmission_rate": (
            float(homolog_counts[0] / homolog_total)
            if homolog_total
            else None
        ),
        "homolog_1_transmission_rate": (
            float(homolog_counts[1] / homolog_total)
            if homolog_total
            else None
        ),
        "parent_a_contribution_rate": (
            float(parent_counts["A"] / parent_total) if parent_total else None
        ),
        "parent_b_contribution_rate": (
            float(parent_counts["B"] / parent_total) if parent_total else None
        ),
        "segregation_distortion_z": (
            float(
                (homolog_counts[1] - 0.5 * homolog_total)
                / math.sqrt(0.25 * homolog_total)
            )
            if homolog_total
            else None
        ),
        "mutation_event_count": mutation_events,
        "semantic_mutation_count": semantic_mutations,
    }


__all__ = [
    "DIPLOID_TELEMETRY_SCHEMA_VERSION",
    "summarize_diploid_population",
    "summarize_transmission_receipts",
]
