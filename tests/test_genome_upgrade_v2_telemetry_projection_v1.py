from __future__ import annotations

import copy
import json

import pytest
import torch

from evolution.genome_upgrade_v2_telemetry import (
    CHRONICLE_SCHEMA_VERSION,
    COMPACT_SUMMARY_SCHEMA_VERSION,
    E1_METRIC_ATTRIBUTION_SCHEMA_VERSION,
    HEREDITARY_CAPACITY_SCHEMA_VERSION,
    HEREDITARY_LOCUS_COUNT,
    OBSERVABILITY_STORAGE_BUDGET_BYTES,
    RECOMB72_LONGITUDINAL_SCHEMA_VERSION,
    REPORTING_SCHEMA_VERSION,
    observability_storage_budget_v1,
    project_genome_upgrade_v2_e1_metric_attribution_v1,
    project_genome_upgrade_v2_checkpoint_summary_v1,
    reporting_schema_v1,
    summarize_genome_upgrade_v2_resident_tensors,
)
from perf.cuda_genome_upgrade_v2_state_bank import (
    CudaGenomeUpgradeV2StateBank,
    resolve_genome_v2_founder_profile_v1,
)
from evolution.genome_upgrade_v2_recomb72_founders_v1 import (
    founder_profile_id_v1,
)


def test_e1_metric_attribution_reduces_settled_sidecar_without_raw_arrays() -> None:
    bank = CudaGenomeUpgradeV2StateBank.allocate(
        cells=1,
        capacity=2,
        device="cpu",
    )
    alive = torch.tensor([[True, False]], dtype=torch.bool)
    context = torch.zeros((1, 2, 8), dtype=torch.int64)
    context[0, 0, 3] = 1024
    bank.express(context)
    summary = summarize_genome_upgrade_v2_resident_tensors(
        bank.named_tensors(),
        alive_mask=alive,
        context_q10=context,
        consequence_rows=(
            {
                "cell": 0,
                "surface": "energy",
                "applied_delta": -1.5,
                "cap_bound": True,
            },
        ),
    )
    checkpoint = {
        "genome_upgrade_v2_checkpoint_projection": {
            "identity": {
                "cell_id": "cell-a",
                "generation": 32,
                "tick": 0,
                "payload_sha256": "a" * 64,
            },
            "compact_generation_summary": summary,
        }
    }

    projected = project_genome_upgrade_v2_e1_metric_attribution_v1(
        checkpoint,
        expected_cell_id="cell-a",
    )

    assert projected is not None
    assert projected["schema_version"] == E1_METRIC_ATTRIBUTION_SCHEMA_VERSION
    assert projected["metrics"] == {
        "genome_v2.expression_observation_count": 1,
        "genome_v2.context_observation_count": 1,
        "genome_v2.cost_observation_count": 1,
        "genome_v2.cap_observation_count": 1,
    }
    assert projected["cap_binding_count_total"] == 1
    assert projected["applied_delta_by_surface"] == {"energy": -1.5}
    assert projected["raw_tick_arrays_retained"] is False
    assert projected["trajectory_influence"] == "forbidden"


def test_e1_metric_attribution_is_absent_when_genome_v2_is_disabled() -> None:
    assert (
        project_genome_upgrade_v2_e1_metric_attribution_v1(
            {},
            expected_cell_id="cell-a",
        )
        is None
    )


def test_resident_summary_reports_bounded_hidden_hereditary_capacity() -> None:
    bank = CudaGenomeUpgradeV2StateBank.allocate(
        cells=1,
        capacity=3,
        device="cpu",
    )
    bank.alleles_q10[0, 0, 0, 3, 1] = 512
    alive = torch.tensor([[True, True, False]], dtype=torch.bool)
    context = torch.zeros((1, 3, 8), dtype=torch.int64)
    bank.express(context)

    summary = summarize_genome_upgrade_v2_resident_tensors(
        bank.named_tensors(),
        alive_mask=alive,
        context_q10=context,
    )
    hereditary = summary["cells"][0]["hereditary_capacity"]

    assert summary["schema_version"] == COMPACT_SUMMARY_SCHEMA_VERSION
    assert hereditary["schema_version"] == HEREDITARY_CAPACITY_SCHEMA_VERSION
    assert hereditary["population_count"] == 2
    assert hereditary["locus_summary_count"] == HEREDITARY_LOCUS_COUNT == 60
    assert hereditary["allele_parameter_heterozygous_count"] == 1
    assert hereditary["allele_parameter_heterozygosity"] == 1 / 96
    assert hereditary["modifier_heterozygosity"] == 0.0
    assert hereditary["distinct_genotype_count"] == 2
    assert hereditary["distinct_phenotype_count"] == 1
    assert hereditary["hidden_genotype_retained_agent_count"] == 1
    assert hereditary["hidden_genotype_retention_fraction"] == 0.5
    assert hereditary["phenotype_equivalent_genotype_group_count"] == 1
    assert hereditary["phenotype_equivalent_distinct_genotype_count"] == 2
    assert hereditary["phenotype_equivalent_agent_count"] == 2
    assert hereditary["raw_genotype_rows_retained"] is False

    sensitivity = next(
        row
        for row in hereditary["loci"]
        if row["locus_id"] == "allele.energy_flux.sensitivity"
    )
    assert sensitivity["observed_copy_count"] == 4
    assert sensitivity["distinct_allele_count"] == 2
    assert sensitivity["heterozygous_agent_count"] == 1
    assert sensitivity["non_neutral_variant_retained"] is True
    assert sensitivity["variation_retained"] is True
    assert sensitivity["fixed"] is False


def test_empty_population_has_complete_bounded_locus_summaries() -> None:
    bank = CudaGenomeUpgradeV2StateBank.allocate(
        cells=1,
        capacity=2,
        device="cpu",
    )
    summary = summarize_genome_upgrade_v2_resident_tensors(
        bank.named_tensors(),
        alive_mask=torch.zeros((1, 2), dtype=torch.bool),
    )
    hereditary = summary["cells"][0]["hereditary_capacity"]

    assert hereditary["population_count"] == 0
    assert hereditary["distinct_genotype_count"] == 0
    assert hereditary["distinct_phenotype_count"] == 0
    assert hereditary["allele_parameter_heterozygosity"] is None
    assert hereditary["modifier_heterozygosity"] is None
    assert hereditary["hidden_genotype_retention_fraction"] is None
    assert len(hereditary["loci"]) == HEREDITARY_LOCUS_COUNT
    assert all(row["observed_copy_count"] == 0 for row in hereditary["loci"])
    assert all(row["fixed"] is False for row in hereditary["loci"])


def test_reporting_schema_bump_and_max_capacity_projection_are_bounded() -> None:
    reporting = reporting_schema_v1()
    budget = observability_storage_budget_v1()
    assert REPORTING_SCHEMA_VERSION == "genome_v2_reporting_v2"
    assert CHRONICLE_SCHEMA_VERSION == "genome_v2_chronicle_v2"
    assert reporting["source_schema_version"] == (
        "genome_upgrade_v2_compact_generation_summary_v2"
    )
    assert (
        reporting["hereditary_capacity"]["maximum_locus_summaries"]
        == HEREDITARY_LOCUS_COUNT
    )
    assert budget["maximum_hereditary_locus_summaries"] == (HEREDITARY_LOCUS_COUNT)
    assert reporting["compatibility"] == {
        "base_v2_contract_fields_stable": True,
        "optional_typed_cell_extensions": {
            "recomb72_gv2": RECOMB72_LONGITUDINAL_SCHEMA_VERSION,
        },
        "version_policy": (
            "typed_optional_profile_extensions_do_not_change_base_v2_"
            "reader_semantics"
        ),
    }

    bank = CudaGenomeUpgradeV2StateBank.allocate(
        cells=1,
        capacity=256,
        device="cpu",
    )
    summary = summarize_genome_upgrade_v2_resident_tensors(
        bank.named_tensors(),
        alive_mask=torch.ones((1, 256), dtype=torch.bool),
    )
    checkpoint = {
        "genome_upgrade_v2_checkpoint_projection": {
            "identity": {
                "cell_id": "capacity-cell",
                "generation": 1,
                "tick": 0,
                "payload_sha256": "a" * 64,
            },
            "compact_generation_summary": summary,
        }
    }
    projected = project_genome_upgrade_v2_checkpoint_summary_v1(
        checkpoint,
        expected_cell_id="capacity-cell",
    )

    assert projected is not None
    serialized = json.dumps(
        projected,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    assert len(serialized) < OBSERVABILITY_STORAGE_BUDGET_BYTES

    summary["cells"][0]["hereditary_capacity"]["loci"].pop()
    with pytest.raises(
        ValueError,
        match="hereditary-capacity reporting contract differs",
    ):
        project_genome_upgrade_v2_checkpoint_summary_v1(
            checkpoint,
            expected_cell_id="capacity-cell",
        )


@pytest.mark.parametrize(
    ("state_id", "expected"),
    (
        ("all_heterozygous_ab", {"aa": 0, "ab": 4, "ba": 0, "bb": 0}),
        ("balanced_homozygous_aa_bb", {"aa": 2, "ab": 0, "ba": 0, "bb": 2}),
        ("homozygous_aa_control", {"aa": 4, "ab": 0, "ba": 0, "bb": 0}),
        ("homozygous_bb_control", {"aa": 0, "ab": 0, "ba": 0, "bb": 4}),
    ),
)
def test_recomb72_longitudinal_summary_classifies_exact_founder_states(
    state_id: str,
    expected: dict[str, int],
) -> None:
    profile_id = founder_profile_id_v1(
        "correlated_multi_product_linkage_packages",
        state_id,
    )
    profile = resolve_genome_v2_founder_profile_v1(profile_id)
    bank = CudaGenomeUpgradeV2StateBank.allocate(
        cells=1,
        capacity=4,
        device="cpu",
    )
    alive = torch.ones((1, 4), dtype=torch.bool)
    bank.initialize_founder_populations_by_cell_v1(
        (profile_id,),
        alive_mask=alive,
    )
    bank.express(torch.zeros((1, 4, 8), dtype=torch.int64))

    summary = summarize_genome_upgrade_v2_resident_tensors(
        bank.named_tensors(),
        alive_mask=alive,
        founder_profiles=(
            {
                "schema_version": "genome_upgrade_v2_founder_profile_v1",
                "profile_id": profile_id,
                "payload_sha256": profile["payload_sha256"],
            },
        ),
    )
    recomb72 = summary["cells"][0]["recomb72_gv2"]

    assert recomb72["schema_version"] == RECOMB72_LONGITUDINAL_SCHEMA_VERSION
    assert recomb72["ordered_genotype_counts"] == {**expected, "other": 0}
    assert recomb72["homolog_package_counts"]["recombinant"] == 0
    assert recomb72["homolog_package_counts"]["neither"] == 0
    assert recomb72["novel_endpoint_copy_count"] == 0
    assert recomb72["unexpected_genome_copy_count"] == 0
    assert recomb72["changed_loci_counter_available"] is False
    assert len(recomb72["all_loci"]) == 60
    assert len(recomb72["audited_loci"]) == 12
    assert len(recomb72["audited_locus_pair_associations"]) == 66
    assert sum(recomb72["homolog_package_counts"].values()) == 8
    assert sum(recomb72["individual_package_counts"].values()) == 4
    if state_id == "all_heterozygous_ab":
        assert recomb72["genotype_distribution_sha256_counts"] == {
            profile["founder_population"]["genotype_sha256"]["ab"]: 4
        }
        first_pair = recomb72["audited_locus_pair_associations"][0]
        assert first_pair["phased_endpoint_copy_counts"] == {
            "aa": 4,
            "ab": 0,
            "ba": 0,
            "bb": 4,
        }
        assert first_pair["individual_b_dosage_counts"]["1,1"] == 4
        assert first_pair["phased_ld_d_rational"] == {
            "defined": True,
            "numerator": 16,
            "denominator": 64,
        }


def test_recomb72_longitudinal_summary_detects_cross_recombinant_package() -> None:
    profile_id = founder_profile_id_v1(
        "correlated_multi_product_linkage_packages",
        "all_heterozygous_ab",
    )
    profile = resolve_genome_v2_founder_profile_v1(profile_id)
    bank = CudaGenomeUpgradeV2StateBank.allocate(
        cells=1,
        capacity=2,
        device="cpu",
    )
    alive = torch.ones((1, 2), dtype=torch.bool)
    bank.initialize_founder_populations_by_cell_v1(
        (profile_id,),
        alive_mask=alive,
    )
    bank.alleles_q10[0, 0, 0, 0, 0] = profile["founder_population"]["haplotype_b_q10"][
        0
    ]
    bank.express(torch.zeros((1, 2, 8), dtype=torch.int64))

    summary = summarize_genome_upgrade_v2_resident_tensors(
        bank.named_tensors(),
        alive_mask=alive,
        founder_profiles=(
            {
                "schema_version": "genome_upgrade_v2_founder_profile_v1",
                "profile_id": profile_id,
                "payload_sha256": profile["payload_sha256"],
            },
        ),
    )
    recomb72 = summary["cells"][0]["recomb72_gv2"]

    assert recomb72["cross_recombinant_observed"] is True
    assert recomb72["homolog_package_counts"]["recombinant"] == 1
    assert recomb72["individual_package_counts"]["partial"] == 1
    assert recomb72["novel_endpoint_copy_count"] == 0


def test_recomb72_longitudinal_projection_is_identity_bound_and_bounded() -> None:
    profile_id = founder_profile_id_v1(
        "correlated_multi_product_linkage_packages",
        "all_heterozygous_ab",
    )
    profile = resolve_genome_v2_founder_profile_v1(profile_id)
    bank = CudaGenomeUpgradeV2StateBank.allocate(
        cells=1,
        capacity=256,
        device="cpu",
    )
    alive = torch.ones((1, 256), dtype=torch.bool)
    bank.initialize_founder_populations_by_cell_v1(
        (profile_id,),
        alive_mask=alive,
    )
    endpoints = profile["founder_population"]
    for slot in range(256):
        for locus_index, locus in enumerate(endpoints["audited_loci"][:8]):
            value = (
                locus["endpoint_b_q10"]
                if slot & (1 << locus_index)
                else locus["endpoint_a_q10"]
            )
            bank.alleles_q10[
                0,
                0,
                locus["product_index"],
                locus["parameter_index"],
                slot,
            ] = value
        bank.modifiers_q10[0, 0, 0, slot] = slot - 128
    bank.express(torch.zeros((1, 256, 8), dtype=torch.int64))
    identity = {
        "schema_version": "genome_upgrade_v2_founder_profile_v1",
        "profile_id": profile_id,
        "payload_sha256": profile["payload_sha256"],
    }
    summary = summarize_genome_upgrade_v2_resident_tensors(
        bank.named_tensors(),
        alive_mask=alive,
        founder_profiles=(identity,),
        selected_events=tuple(
            {"cell": 0, "event_ordinal": ordinal} for ordinal in range(16)
        ),
    )
    checkpoint = {
        "genome_upgrade_v2_checkpoint_projection": {
            "founder_profile": identity,
            "identity": {
                "cell_id": "recomb72-cell",
                "generation": 30,
                "tick": 0,
                "payload_sha256": "a" * 64,
            },
            "compact_generation_summary": summary,
        }
    }

    projected = project_genome_upgrade_v2_checkpoint_summary_v1(
        checkpoint,
        expected_cell_id="recomb72-cell",
    )

    assert projected is not None
    recomb72 = projected["summary"]["cells"][0]["recomb72_gv2"]
    assert len(recomb72["audited_locus_pair_associations"]) == 66
    assert len(recomb72["genotype_distribution_sha256_counts"]) == 256
    assert len(recomb72["phenotype_distribution"]) == 256
    assert recomb72["unexpected_genome_copy_count"] == 255
    assert (
        len(projected["summary"]["cells"][0]["selected_causal_events"]) == 16
    )
    assert (
        len(
            json.dumps(projected, sort_keys=True, separators=(",", ":")).encode("ascii")
        )
        < OBSERVABILITY_STORAGE_BUDGET_BYTES
    )
    checkpoint["genome_upgrade_v2_checkpoint_projection"]["founder_profile"][
        "payload_sha256"
    ] = "b" * 64
    with pytest.raises(
        ValueError,
        match="longitudinal reporting contract differs",
    ):
        project_genome_upgrade_v2_checkpoint_summary_v1(
            checkpoint,
            expected_cell_id="recomb72-cell",
        )


def test_recomb72_hidden_profile_reports_exact_phenotype_equivalence() -> None:
    profile_id = founder_profile_id_v1(
        "context_conditional_hidden_pair",
        "balanced_homozygous_aa_bb",
    )
    profile = resolve_genome_v2_founder_profile_v1(profile_id)
    bank = CudaGenomeUpgradeV2StateBank.allocate(
        cells=1,
        capacity=4,
        device="cpu",
    )
    alive = torch.ones((1, 4), dtype=torch.bool)
    bank.initialize_founder_populations_by_cell_v1(
        (profile_id,),
        alive_mask=alive,
    )
    bank.express(torch.zeros((1, 4, 8), dtype=torch.int64))
    identity = {
        "schema_version": "genome_upgrade_v2_founder_profile_v1",
        "profile_id": profile_id,
        "payload_sha256": profile["payload_sha256"],
    }

    recomb72 = summarize_genome_upgrade_v2_resident_tensors(
        bank.named_tensors(),
        alive_mask=alive,
        founder_profiles=(identity,),
    )["cells"][0]["recomb72_gv2"]

    assert len(recomb72["phenotype_distribution"]) == 1
    assert recomb72["phenotype_distribution"][0]["agent_count"] == 4
    assert len(recomb72["phenotype_equivalent_hidden_genotype_groups"]) == 1
    hidden = recomb72["phenotype_equivalent_hidden_genotype_groups"][0]
    assert hidden["genotype_count"] == 2
    assert hidden["agent_count"] == 4
    checkpoint_summary = summarize_genome_upgrade_v2_resident_tensors(
        bank.named_tensors(),
        alive_mask=alive,
        founder_profiles=(identity,),
    )
    checkpoint = {
        "genome_upgrade_v2_checkpoint_projection": {
            "founder_profile": identity,
            "identity": {
                "cell_id": "hidden-cell",
                "generation": 5,
                "tick": 0,
                "payload_sha256": "a" * 64,
            },
            "compact_generation_summary": checkpoint_summary,
        }
    }
    assert (
        project_genome_upgrade_v2_checkpoint_summary_v1(
            checkpoint,
            expected_cell_id="hidden-cell",
        )
        is not None
    )
    checkpoint_summary["cells"][0]["recomb72_gv2"][
        "phenotype_equivalent_hidden_genotype_groups"
    ] = []
    with pytest.raises(ValueError, match="hidden-genotype group differs"):
        project_genome_upgrade_v2_checkpoint_summary_v1(
            checkpoint,
            expected_cell_id="hidden-cell",
        )


def test_recomb72_projection_rejects_malformed_decision_partition() -> None:
    profile_id = founder_profile_id_v1(
        "single_locus_extreme_pair",
        "all_heterozygous_ab",
    )
    profile = resolve_genome_v2_founder_profile_v1(profile_id)
    bank = CudaGenomeUpgradeV2StateBank.allocate(
        cells=1,
        capacity=2,
        device="cpu",
    )
    alive = torch.ones((1, 2), dtype=torch.bool)
    bank.initialize_founder_populations_by_cell_v1(
        (profile_id,),
        alive_mask=alive,
    )
    bank.express(torch.zeros((1, 2, 8), dtype=torch.int64))
    identity = {
        "schema_version": "genome_upgrade_v2_founder_profile_v1",
        "profile_id": profile_id,
        "payload_sha256": profile["payload_sha256"],
    }
    summary = summarize_genome_upgrade_v2_resident_tensors(
        bank.named_tensors(),
        alive_mask=alive,
        founder_profiles=(identity,),
    )
    original = copy.deepcopy(summary)
    summary["cells"][0]["recomb72_gv2"]["homolog_package_counts"]["intact_a"] += 1
    checkpoint = {
        "genome_upgrade_v2_checkpoint_projection": {
            "founder_profile": identity,
            "identity": {
                "cell_id": "malformed-cell",
                "generation": 5,
                "tick": 0,
                "payload_sha256": "a" * 64,
            },
            "compact_generation_summary": summary,
        }
    }

    with pytest.raises(ValueError, match="homolog package partition differs"):
        project_genome_upgrade_v2_checkpoint_summary_v1(
            checkpoint,
            expected_cell_id="malformed-cell",
        )
    endpoint_tamper = copy.deepcopy(original)
    endpoint_tamper["cells"][0]["recomb72_gv2"]["all_loci"][0][
        "endpoint_a_q10"
    ] = 999
    checkpoint["genome_upgrade_v2_checkpoint_projection"][
        "compact_generation_summary"
    ] = endpoint_tamper
    with pytest.raises(ValueError, match="all-locus counts differ"):
        project_genome_upgrade_v2_checkpoint_summary_v1(
            checkpoint,
            expected_cell_id="malformed-cell",
        )


def test_recomb72_package_partition_covers_ba_recombinant_and_neither() -> None:
    profile_id = founder_profile_id_v1(
        "correlated_multi_product_linkage_packages",
        "all_heterozygous_ab",
    )
    profile = resolve_genome_v2_founder_profile_v1(profile_id)
    bank = CudaGenomeUpgradeV2StateBank.allocate(
        cells=1,
        capacity=3,
        device="cpu",
    )
    alive = torch.ones((1, 3), dtype=torch.bool)
    bank.initialize_founder_populations_by_cell_v1(
        (profile_id,),
        alive_mask=alive,
    )
    endpoints = profile["founder_population"]
    for locus in endpoints["audited_loci"]:
        product = locus["product_index"]
        parameter = locus["parameter_index"]
        bank.alleles_q10[0, 0, product, parameter, 0] = locus[
            "endpoint_b_q10"
        ]
        bank.alleles_q10[0, 1, product, parameter, 0] = locus[
            "endpoint_a_q10"
        ]
    first, second = endpoints["audited_loci"][:2]
    bank.alleles_q10[
        0,
        0,
        first["product_index"],
        first["parameter_index"],
        1,
    ] = first["endpoint_b_q10"]
    bank.alleles_q10[
        0,
        1,
        second["product_index"],
        second["parameter_index"],
        1,
    ] = second["endpoint_a_q10"]
    bank.modifiers_q10[0, 0, 0, 2] = 999
    bank.express(torch.zeros((1, 3, 8), dtype=torch.int64))
    identity = {
        "schema_version": "genome_upgrade_v2_founder_profile_v1",
        "profile_id": profile_id,
        "payload_sha256": profile["payload_sha256"],
    }

    recomb72 = summarize_genome_upgrade_v2_resident_tensors(
        bank.named_tensors(),
        alive_mask=alive,
        founder_profiles=(identity,),
    )["cells"][0]["recomb72_gv2"]

    assert recomb72["ordered_genotype_counts"] == {
        "aa": 0,
        "ab": 0,
        "ba": 1,
        "bb": 0,
        "other": 2,
    }
    assert recomb72["individual_package_counts"] == {
        "intact_aa": 0,
        "intact_ab": 0,
        "intact_ba": 1,
        "intact_bb": 0,
        "partial": 0,
        "recombinant": 1,
        "neither": 1,
    }
    assert recomb72["homolog_package_counts"] == {
        "intact_a": 1,
        "intact_b": 2,
        "recombinant": 2,
        "neither": 1,
    }
    assert recomb72["unexpected_genome_copy_count"] == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_recomb72_longitudinal_summary_is_exact_across_cpu_and_cuda() -> None:
    profile_id = founder_profile_id_v1(
        "correlated_multi_product_linkage_packages",
        "balanced_homozygous_aa_bb",
    )
    profile = resolve_genome_v2_founder_profile_v1(profile_id)
    identity = {
        "schema_version": "genome_upgrade_v2_founder_profile_v1",
        "profile_id": profile_id,
        "payload_sha256": profile["payload_sha256"],
    }
    summaries = []
    for device in ("cpu", "cuda"):
        bank = CudaGenomeUpgradeV2StateBank.allocate(
            cells=1,
            capacity=4,
            device=device,
        )
        alive = torch.ones((1, 4), dtype=torch.bool, device=device)
        bank.initialize_founder_populations_by_cell_v1(
            (profile_id,),
            alive_mask=alive,
        )
        context = torch.zeros((1, 4, 8), dtype=torch.int64, device=device)
        bank.express(context)
        summaries.append(
            summarize_genome_upgrade_v2_resident_tensors(
                bank.named_tensors(),
                alive_mask=alive,
                context_q10=context,
                founder_profiles=(identity,),
            )["cells"][0]["recomb72_gv2"]
        )

    assert summaries[0] == summaries[1]
