from __future__ import annotations

from copy import deepcopy

import pytest

from engine.authoritative_state.schema import content_sha256
from evolution.genome_upgrade_v2 import (
    PHENOTYPE_CHANNELS,
    PRODUCT_CHANNELS,
    Q10_SCALE,
    GenomeUpgradeV2Error,
)
from evolution.genome_upgrade_v2_bd_capabilities_v1 import (
    ADMITTED_RUNTIME_SURFACES,
    CAPABILITY_CONTRACT_SHA256,
    FamilyHeadPolicyRecordV1,
    MemoryRecordV1,
    PerceptionRecordV1,
    capability_contract_payload_v1,
    polygenic_decomposition_v1,
)


def test_contract_is_content_addressed_bounded_and_strategy_free() -> None:
    payload = capability_contract_payload_v1()
    assert content_sha256(payload) == CAPABILITY_CONTRACT_SHA256
    assert len(payload["candidate_locus_groups"]) == 4
    assert all(row["loci"] for row in payload["candidate_locus_groups"])
    assert all(row["bounds"] == [-Q10_SCALE, Q10_SCALE] for row in payload["candidate_locus_groups"])
    assert all(row["unit"] and row["cost"] and row["mutation"] for row in payload["candidate_locus_groups"])
    assert payload["candidate_admission"]["direct_strategy_encoding"] == "forbidden"
    assert payload["compact_records"]["runtime_admission"].startswith(
        "candidate_fixtures_only"
    )
    assert tuple(payload["admitted_runtime_surfaces"]) == tuple(
        {
            **row,
            "bounds_q10": [-Q10_SCALE, Q10_SCALE],
            "diploid_storage": (
                "two_homolog_product_parameter_and_modifier_loci"
            ),
            "expression": "fixed_sparse_product_to_phenotype_projection",
            "mutation": "bounded_parameter_and_modifier_mutation_v1",
            "recombination": (
                "genome_v2_biparental_gamete_v1_six_fixed_blocks"
            ),
            "neutral_migration": "homozygous_neutral_product_bundle",
            "checkpoint": "cuda_genome_upgrade_v2_state_bank_v1",
            "route_support": "reference_native_cuda_exact",
            "observability": (
                "compact_product_phenotype_contribution_attribution"
            ),
        }
        for row in ADMITTED_RUNTIME_SURFACES
    )
    changed = deepcopy(payload)
    changed["candidate_locus_groups"][0]["bounds"][1] -= 1
    assert content_sha256(changed) != CAPABILITY_CONTRACT_SHA256
    payload["candidate_locus_groups"][0]["loci"].clear()
    assert content_sha256(capability_contract_payload_v1()) == (
        CAPABILITY_CONTRACT_SHA256
    )


def test_family_head_policy_expression_preserves_cap_and_exact_zero_sum() -> None:
    record = FamilyHeadPolicyRecordV1(
        homolog_0_q10=(256, 256, -256, -256),
        homolog_1_q10=(256, -256, 256, -256),
    )
    expressed = record.express_q10()
    assert sum(expressed) == 0
    assert all(-256 <= value <= 256 for value in expressed)
    payload = record.payload()
    assert payload["record_sha256"] == content_sha256(
        {key: value for key, value in payload.items() if key != "record_sha256"}
    )


@pytest.mark.parametrize(
    ("record", "message"),
    [
        (
            FamilyHeadPolicyRecordV1,
            "homolog_0_q10",
        ),
    ],
)
def test_family_head_policy_rejects_wrong_shape(record: object, message: str) -> None:
    with pytest.raises(GenomeUpgradeV2Error, match=message):
        record(  # type: ignore[operator]
            homolog_0_q10=(0, 0, 0),
            homolog_1_q10=(0, 0, 0, 0),
        )


def test_perception_uses_explicit_union_and_radius_rules() -> None:
    record = PerceptionRecordV1(
        sensor_radius_steps=(2, 6),
        channel_masks=(0b0101, 0b1010),
    )
    assert record.express() == (6, 0b1111)
    with pytest.raises(GenomeUpgradeV2Error, match="channel_masks"):
        PerceptionRecordV1(sensor_radius_steps=(2, 3), channel_masks=(0, 16))


def test_memory_expression_cannot_create_illegal_intermediate_capacity() -> None:
    record = MemoryRecordV1(dimensions=(8, 32), module_masks=(0b0011, 0b1100))
    assert record.express() == (32, 0b1111)
    with pytest.raises(GenomeUpgradeV2Error, match="illegal ladder"):
        MemoryRecordV1(dimensions=(8, 24), module_masks=(0, 0))


def test_polygenic_decomposition_is_fixed_shape_exact_and_content_addressed() -> None:
    products = [0] * len(PRODUCT_CHANNELS)
    products[0] = Q10_SCALE
    products[1] = Q10_SCALE // 2
    result = polygenic_decomposition_v1(products)
    assert len(result["phenotypes"]) == len(PHENOTYPE_CHANNELS)
    for row in result["phenotypes"]:
        assert len(row["contributions_q10"]) == len(PRODUCT_CHANNELS)
        assert len(row["top_product_indices"]) == 3
        assert len(row["top_contributions_q10"]) == 3
        assert row["output_q10"] == max(
            -Q10_SCALE, min(Q10_SCALE, row["raw_sum_q10"])
        )
    unsigned = {
        key: value
        for key, value in result.items()
        if key != "decomposition_sha256"
    }
    assert result["decomposition_sha256"] == content_sha256(unsigned)
    assert result == polygenic_decomposition_v1(tuple(products))


def test_decomposition_reports_redundancy_and_signed_compensation() -> None:
    body_products = [0] * len(PRODUCT_CHANNELS)
    body_products[0] = Q10_SCALE
    body_products[1] = Q10_SCALE
    body_products[4] = Q10_SCALE
    body_size = polygenic_decomposition_v1(body_products)["phenotypes"][0]
    movement_products = list(body_products)
    movement_products[2] = Q10_SCALE
    movement = polygenic_decomposition_v1(movement_products)["phenotypes"][1]
    assert body_size["active_contributor_count"] == 3
    assert body_size["redundant_contributor_count"] == 2
    assert body_size["opposing_pair_count"] == 0
    assert movement["opposing_pair_count"] > 0
    assert 0 < movement["compensation_q10"] <= Q10_SCALE


@pytest.mark.parametrize(
    "products",
    [
        [0] * (len(PRODUCT_CHANNELS) - 1),
        [0] * (len(PRODUCT_CHANNELS) - 1) + [Q10_SCALE + 1],
        [0] * (len(PRODUCT_CHANNELS) - 1) + [True],
    ],
)
def test_decomposition_rejects_noncanonical_product_vectors(
    products: list[object],
) -> None:
    with pytest.raises(GenomeUpgradeV2Error, match="products_q10"):
        polygenic_decomposition_v1(products)
