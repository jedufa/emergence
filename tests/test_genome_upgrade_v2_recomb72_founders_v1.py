from __future__ import annotations

import pytest
import torch

from engine.perf import (
    cuda_architecture_b_direct_resident_source_allocator_v1 as allocator,
)
from evolution.genome_upgrade_v2 import CONTEXT_COUNT, express_reference_v1
from evolution.genome_upgrade_v2_recomb72_founders_v1 import (
    FOUNDER_PATTERN_IDS,
    FOUNDER_PROFILE_IDS,
    FOUNDER_STATE_IDS,
    founder_population_genotypes_v1,
    founder_population_geometry_v1,
    founder_profile_id_v1,
    resolve_recomb72_gv2_founder_profile_v1,
)
from perf.cuda_genome_upgrade_v2_state_bank import (
    CudaGenomeUpgradeV2StateBank,
    GENOME_V2_FOUNDER_PROFILE_IDS,
    resolve_genome_v2_founder_profile_v1,
)


def test_all_frozen_founder_profiles_have_unique_canonical_identities() -> None:
    profiles = [
        resolve_recomb72_gv2_founder_profile_v1(profile_id)
        for profile_id in FOUNDER_PROFILE_IDS
    ]
    assert len(profiles) == 12
    assert len({row["payload_sha256"] for row in profiles}) == 12
    assert {
        (row["pattern_id"], row["state_id"]) for row in profiles
    } == {
        (pattern_id, state_id)
        for pattern_id in FOUNDER_PATTERN_IDS
        for state_id in FOUNDER_STATE_IDS
    }
    assert set(FOUNDER_PROFILE_IDS) < set(GENOME_V2_FOUNDER_PROFILE_IDS)
    wrappers = [
        resolve_genome_v2_founder_profile_v1(profile_id)
        for profile_id in FOUNDER_PROFILE_IDS
    ]
    assert all(
        row["founder_population"]["payload_sha256"]
        == row["source"]["population_payload_sha256"]
        for row in wrappers
    )


def test_matched_ab_and_balanced_states_have_identical_endpoint_counts() -> None:
    for pattern_id in FOUNDER_PATTERN_IDS:
        heterozygous = founder_population_geometry_v1(
            founder_profile_id_v1(pattern_id, "all_heterozygous_ab"),
            64,
        )
        balanced = founder_population_geometry_v1(
            founder_profile_id_v1(
                pattern_id,
                "balanced_homozygous_aa_bb",
            ),
            64,
        )
        assert heterozygous["endpoint_copy_counts"] == {"a": 64, "b": 64}
        assert balanced["endpoint_copy_counts"] == {"a": 64, "b": 64}
        assert heterozygous["rng_draws"] == balanced["rng_draws"] == 0


def test_hidden_pattern_is_exactly_phenotype_equivalent_at_zero_context() -> None:
    profile_id = founder_profile_id_v1(
        "context_conditional_hidden_pair",
        "balanced_homozygous_aa_bb",
    )
    profile = resolve_recomb72_gv2_founder_profile_v1(profile_id)
    assert profile["zero_context_exact_phenotype_equivalence"] is True
    genotypes = founder_population_genotypes_v1(profile_id, 2)
    assert (
        express_reference_v1(genotypes[0], (0,) * CONTEXT_COUNT).phenotypes_q10
        == express_reference_v1(
            genotypes[1],
            (0,) * CONTEXT_COUNT,
        ).phenotypes_q10
    )


def test_correlated_pattern_spans_every_fixed_linkage_block() -> None:
    profile = resolve_recomb72_gv2_founder_profile_v1(
        founder_profile_id_v1(
            "correlated_multi_product_linkage_packages",
            "all_heterozygous_ab",
        )
    )
    assert len(profile["audited_loci"]) == 12
    assert {
        row["linkage_block"] for row in profile["audited_loci"]
    } == set(range(6))
    assert all(
        row["endpoint_a_q10"] == -row["endpoint_b_q10"]
        for row in profile["audited_loci"]
    )


def test_resident_initializer_installs_exact_heterogeneous_populations() -> None:
    bank = CudaGenomeUpgradeV2StateBank.allocate(
        cells=2,
        capacity=8,
        device="cpu",
    )
    alive_mask = torch.zeros((2, 8), dtype=torch.bool)
    alive_mask[:, [0, 2, 4, 6]] = True
    pattern = "single_locus_extreme_pair"
    profile_ids = (
        founder_profile_id_v1(pattern, "all_heterozygous_ab"),
        founder_profile_id_v1(pattern, "balanced_homozygous_aa_bb"),
    )
    geometries = bank.initialize_founder_populations_by_cell_v1(
        profile_ids,
        alive_mask=alive_mask,
    )

    heterozygous = [
        bank.genotype(0, slot) for slot in (0, 2, 4, 6)
    ]
    balanced = [
        bank.genotype(1, slot) for slot in (0, 2, 4, 6)
    ]
    assert all(
        row.alleles[0][0][0] == -1024
        and row.alleles[1][0][0] == 1024
        for row in heterozygous
    )
    assert [
        (row.alleles[0][0][0], row.alleles[1][0][0])
        for row in balanced
    ] == [(-1024, -1024), (-1024, -1024), (1024, 1024), (1024, 1024)]
    assert geometries[0]["endpoint_copy_counts"] == {"a": 4, "b": 4}
    assert geometries[1]["endpoint_copy_counts"] == {"a": 4, "b": 4}


def test_source_allocator_initializes_mixed_uniform_and_population_profiles() -> None:
    bank = CudaGenomeUpgradeV2StateBank.allocate(
        cells=2,
        capacity=4,
        device="cpu",
    )
    alive_mask = torch.ones((2, 4), dtype=torch.bool)
    profiles = tuple(
        resolve_genome_v2_founder_profile_v1(
            founder_profile_id_v1(
                "single_locus_extreme_pair",
                state_id,
            )
        )
        for state_id in (
            "all_heterozygous_ab",
            "balanced_homozygous_aa_bb",
        )
    )
    geometries = allocator._initialize_genome_v2_fresh_founders_v1(
        bank,
        profiles,
        alive_mask=alive_mask,
    )
    assert len(geometries) == 2
    assert bank.genotype(0, 0).alleles[0][0][0] == -1024
    assert bank.genotype(0, 0).alleles[1][0][0] == 1024
    assert bank.genotype(1, 0).alleles[1][0][0] == -1024
    assert bank.genotype(1, 3).alleles[0][0][0] == 1024

    mixed_bank = CudaGenomeUpgradeV2StateBank.allocate(
        cells=2,
        capacity=4,
        device="cpu",
    )
    mixed_geometries = allocator._initialize_genome_v2_fresh_founders_v1(
        mixed_bank,
        (resolve_genome_v2_founder_profile_v1("neutral"), profiles[0]),
        alive_mask=alive_mask,
    )

    assert len(mixed_geometries) == 1
    assert mixed_bank.genotype(0, 0).alleles[0][0][0] == 0
    assert mixed_bank.genotype(0, 3).alleles[1][0][0] == 0
    assert mixed_bank.genotype(1, 0).alleles[0][0][0] == -1024
    assert mixed_bank.genotype(1, 0).alleles[1][0][0] == 1024
