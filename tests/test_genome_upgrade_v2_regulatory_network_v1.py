from dataclasses import replace

import pytest

from evolution.genome_upgrade_v2 import CONTEXT_COUNT, PRODUCT_COUNT, Q10_SCALE
from evolution.genome_upgrade_v2_regulatory_network_v1 import (
    DEFAULT_FIXED_REGULATORY_NETWORK_V1,
    EDGE_ACTIVATE,
    EDGE_INHIBIT,
    EDGE_MODULATE,
    MAX_FAN_IN,
    NODE_CONTEXT_INPUT,
    NODE_PRODUCT_INPUT,
    NODE_PRODUCT_OUTPUT,
    NODE_REGULATOR,
    PLEIOTROPIC_INFLUENCE_BUDGET_Q10,
    REGULATORY_CONTRACT_SHA256,
    REGULATORY_UPDATE_SCHEDULE,
    EpistaticModifierEdgeV1,
    FixedRegulatoryNetworkV1,
    GenomeUpgradeV2RegulatoryError,
    developmental_stage_from_context_v1,
    express_regulatory_reference_v1,
    pleiotropic_influence_budget_v1,
    regulatory_contract_payload_v1,
)


def _zeros(count: int) -> tuple[int, ...]:
    return (0,) * count


def test_contract_is_fixed_explicit_bounded_and_content_addressed() -> None:
    payload = regulatory_contract_payload_v1()
    kinds = [node["kind"] for node in payload["nodes"]]

    assert len(REGULATORY_CONTRACT_SHA256) == 64
    assert payload == regulatory_contract_payload_v1()
    assert kinds.count(NODE_PRODUCT_INPUT) == PRODUCT_COUNT
    assert kinds.count(NODE_CONTEXT_INPUT) == CONTEXT_COUNT
    assert kinds.count(NODE_REGULATOR) == 4
    assert kinds.count(NODE_PRODUCT_OUTPUT) == PRODUCT_COUNT
    assert payload["maximum_fan_in"] == MAX_FAN_IN
    assert payload["update_schedule"] == REGULATORY_UPDATE_SCHEDULE
    assert {edge["effect"] for edge in payload["edges"]} == {
        EDGE_ACTIVATE,
        EDGE_INHIBIT,
        EDGE_MODULATE,
    }
    assert payload["hidden_rng_draws"] == 0


def test_reference_execution_is_deterministic_and_bounded() -> None:
    products = tuple((index - 6) * 97 for index in range(PRODUCT_COUNT))
    context = (512, 256, 128, 0, 0, 768, 0, 0)
    modifiers = tuple((index - 6) * 53 for index in range(PRODUCT_COUNT))

    first = express_regulatory_reference_v1(products, context, modifiers)
    second = express_regulatory_reference_v1(products, context, modifiers)

    assert first == second
    assert all(-Q10_SCALE <= value <= Q10_SCALE for value in first.regulated_products_q10)
    assert all(0 <= value <= Q10_SCALE for value in first.regulator_nodes_q10)
    assert first.regulatory_cost_q10 <= (
        DEFAULT_FIXED_REGULATORY_NETWORK_V1.regulatory_budget_q10
    )


def test_activation_inhibition_and_modulation_have_distinct_effects() -> None:
    products = list(_zeros(PRODUCT_COUNT))
    products[10] = 512
    context = list(_zeros(CONTEXT_COUNT))
    context[5] = Q10_SCALE
    hazard = express_regulatory_reference_v1(
        products, context, _zeros(PRODUCT_COUNT)
    )
    assert hazard.regulated_products_q10[9] > 0
    assert hazard.regulated_products_q10[10] < products[10]

    context[5] = 0
    context[6] = Q10_SCALE
    juvenile = express_regulatory_reference_v1(
        products, context, _zeros(PRODUCT_COUNT)
    )
    assert juvenile.regulated_products_q10[10] > products[10]


def test_regulatory_budget_rejects_an_over_budget_execution() -> None:
    constrained = replace(
        DEFAULT_FIXED_REGULATORY_NETWORK_V1,
        regulatory_budget_q10=1,
    )
    context = list(_zeros(CONTEXT_COUNT))
    context[5] = Q10_SCALE
    with pytest.raises(
        GenomeUpgradeV2RegulatoryError,
        match="exceeds its fixed Q10 budget",
    ):
        express_regulatory_reference_v1(
            _zeros(PRODUCT_COUNT),
            context,
            _zeros(PRODUCT_COUNT),
            network=constrained,
        )


def test_cross_locus_epistasis_changes_a_different_product() -> None:
    products = list(_zeros(PRODUCT_COUNT))
    products[9] = 512
    modifiers = list(_zeros(PRODUCT_COUNT))
    modifiers[9] = Q10_SCALE
    result = express_regulatory_reference_v1(
        products, _zeros(CONTEXT_COUNT), modifiers
    )

    assert result.epistatic_delta_q10[3] == 256
    assert result.regulated_products_q10[3] == 256
    assert all(
        edge.source_product_index != edge.target_product_index
        for edge in DEFAULT_FIXED_REGULATORY_NETWORK_V1.epistatic_edges
    )


def test_bounded_modifiers_cover_stability_threshold_sensitivity_and_gains() -> None:
    products = list(_zeros(PRODUCT_COUNT))
    products[2] = 512
    products[7] = Q10_SCALE
    products[10] = Q10_SCALE
    context = list(_zeros(CONTEXT_COUNT))
    context[0] = 512
    context[5] = 512
    context[7] = Q10_SCALE
    neutral = express_regulatory_reference_v1(
        products, context, _zeros(PRODUCT_COUNT)
    )

    stability = list(_zeros(PRODUCT_COUNT))
    stability[2] = Q10_SCALE
    assert express_regulatory_reference_v1(
        products, context, stability
    ).regulated_products_q10[2] > neutral.regulated_products_q10[2]

    saturation = list(_zeros(PRODUCT_COUNT))
    saturation[11] = Q10_SCALE
    assert abs(
        express_regulatory_reference_v1(
            products, context, saturation
        ).regulated_products_q10[7]
    ) == 896

    threshold = list(_zeros(PRODUCT_COUNT))
    threshold[1] = Q10_SCALE
    assert express_regulatory_reference_v1(
        products, context, threshold
    ).regulator_nodes_q10[0] < neutral.regulator_nodes_q10[0]

    sensitivity = list(_zeros(PRODUCT_COUNT))
    sensitivity[6] = Q10_SCALE
    assert express_regulatory_reference_v1(
        products, context, sensitivity
    ).regulator_nodes_q10[2] > neutral.regulator_nodes_q10[2]

    activation = list(_zeros(PRODUCT_COUNT))
    activation[8] = Q10_SCALE
    assert express_regulatory_reference_v1(
        products, context, activation
    ).regulator_nodes_q10[0] > neutral.regulator_nodes_q10[0]

    inhibition = list(_zeros(PRODUCT_COUNT))
    inhibition[10] = Q10_SCALE
    assert express_regulatory_reference_v1(
        products, context, inhibition
    ).regulator_nodes_q10[3] < neutral.regulator_nodes_q10[3]


def test_developmental_switch_is_explicit_and_mutually_exclusive() -> None:
    products = list(_zeros(PRODUCT_COUNT))
    products[10] = 512
    mature_context = list(_zeros(CONTEXT_COUNT))
    juvenile_context = list(mature_context)
    juvenile_context[6] = Q10_SCALE
    senescent_context = list(mature_context)
    senescent_context[7] = Q10_SCALE

    mature = express_regulatory_reference_v1(
        products, mature_context, _zeros(PRODUCT_COUNT)
    )
    juvenile = express_regulatory_reference_v1(
        products, juvenile_context, _zeros(PRODUCT_COUNT)
    )
    senescent = express_regulatory_reference_v1(
        products, senescent_context, _zeros(PRODUCT_COUNT)
    )

    assert mature.developmental_stage == "mature"
    assert juvenile.developmental_stage == "juvenile"
    assert senescent.developmental_stage == "senescent"
    assert juvenile.regulated_products_q10[10] > mature.regulated_products_q10[10]
    assert senescent.regulated_products_q10[10] < mature.regulated_products_q10[10]

    contradictory = list(mature_context)
    contradictory[6] = contradictory[7] = Q10_SCALE
    with pytest.raises(GenomeUpgradeV2RegulatoryError, match="cannot both be active"):
        developmental_stage_from_context_v1(contradictory)


def test_pleiotropic_tradeoff_and_influence_budget_are_enforced() -> None:
    result = pleiotropic_influence_budget_v1()

    assert max(result["influence_q10_by_product"]) <= (
        PLEIOTROPIC_INFLUENCE_BUDGET_Q10
    )
    assert min(result["contribution_count_by_product"]) >= 2
    assert 0 in result["tradeoff_product_indices"]

    excessive = [[0] * PRODUCT_COUNT for _ in range(4)]
    for row in excessive:
        row[0] = Q10_SCALE
        row[1] = Q10_SCALE
    excessive[0][0] = -Q10_SCALE
    with pytest.raises(
        GenomeUpgradeV2RegulatoryError,
        match="exceeds its per-product budget",
    ):
        pleiotropic_influence_budget_v1(excessive)


def test_network_rejects_fan_in_above_the_fixed_bound() -> None:
    network = DEFAULT_FIXED_REGULATORY_NETWORK_V1
    repeated = network.edges + tuple(network.edges[:2]) * 2
    with pytest.raises(GenomeUpgradeV2RegulatoryError, match="fan-in exceeds"):
        FixedRegulatoryNetworkV1(
            nodes=network.nodes,
            edges=repeated,
            epistatic_edges=network.epistatic_edges,
        )


def test_epistatic_edges_reject_same_locus_shortcuts() -> None:
    with pytest.raises(GenomeUpgradeV2RegulatoryError, match="across product loci"):
        EpistaticModifierEdgeV1(2, 2, 2, EDGE_ACTIVATE, 128)
