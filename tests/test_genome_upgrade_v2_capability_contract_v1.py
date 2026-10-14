from __future__ import annotations

from copy import deepcopy

import pytest

from engine.authoritative_state.schema import content_sha256
from engine.evolution.genome_upgrade_v2_capability_contract_v1 import (
    ADMITTED_SURFACES,
    DEFERRED_ITEMS,
    EXCLUDED_SURFACES,
    GENOME_V2_P10_CAPABILITY_CONTRACT_SHA256,
    MANDATORY_P10_CAPABILITIES,
    GenomeV2P10CapabilityContract,
    GenomeV2P10CapabilityContractError,
    validate_genome_v2_p10_capability_contract_v1,
    validate_genome_v2_p10_surface_selection_v1,
)


def test_contract_freezes_exact_p10_capability_and_channel_envelopes() -> None:
    contract = GenomeV2P10CapabilityContract.v1()
    payload = contract.payload

    assert tuple(
        row["capability_id"] for row in payload["mandatory_capabilities"]
    ) == MANDATORY_P10_CAPABILITIES
    assert all(
        row["mandatory_for_p10_closeout"]
        for row in payload["mandatory_capabilities"]
    )
    assert len(payload["fixed_channel_envelope"]["products"]) == 12
    assert len(payload["fixed_channel_envelope"]["phenotypes"]) == 10
    assert len(payload["fixed_channel_envelope"]["contexts"]) == 8
    assert payload["fixed_channel_envelope"]["semantic_headroom"] == "none"
    assert (
        payload["fixed_channel_envelope"]["cardinality_policy"]
        == "exact_not_maximum"
    )


def test_contract_distinguishes_storage_from_arithmetic_width() -> None:
    numerical = GenomeV2P10CapabilityContract.v1().payload[
        "numerical_declaration"
    ]

    assert numerical["normative_value_width_bits"] == 32
    assert numerical["resident_storage_width_bits"] == 64
    assert numerical["intermediate_arithmetic_width_bits"] == 64
    assert numerical["storage_width_is_semantic_headroom"] is False
    assert (
        GenomeV2P10CapabilityContract.v1()
        .payload["recombination_declaration"]["homolog_selection"]
        == "genome_v2_biparental_gamete_v1"
    )


def test_contract_hash_is_canonical_and_validation_is_exact() -> None:
    contract = GenomeV2P10CapabilityContract.v1()
    bound = contract.to_dict()

    assert contract.contract_sha256 == content_sha256(contract.payload)
    assert contract.contract_sha256 == GENOME_V2_P10_CAPABILITY_CONTRACT_SHA256
    assert validate_genome_v2_p10_capability_contract_v1(bound) == contract

    tampered = deepcopy(bound)
    tampered["capacity_declaration"]["maximum_physical_cells"] = 73
    with pytest.raises(
        GenomeV2P10CapabilityContractError,
        match="unsupported or differs",
    ):
        validate_genome_v2_p10_capability_contract_v1(tampered)

    stale_hash = deepcopy(bound)
    stale_hash["contract_sha256"] = "0" * 64
    with pytest.raises(
        GenomeV2P10CapabilityContractError,
        match="contract_sha256",
    ):
        validate_genome_v2_p10_capability_contract_v1(stale_hash)


def test_surface_selection_admits_only_declared_supported_surfaces() -> None:
    selected = ADMITTED_SURFACES[:3]
    assert validate_genome_v2_p10_surface_selection_v1(selected) == selected

    for unsupported in (EXCLUDED_SURFACES[0], DEFERRED_ITEMS[0], "unknown"):
        with pytest.raises(
            GenomeV2P10CapabilityContractError,
            match="unsupported",
        ):
            validate_genome_v2_p10_surface_selection_v1([unsupported])

    with pytest.raises(
        GenomeV2P10CapabilityContractError,
        match="duplicates",
    ):
        validate_genome_v2_p10_surface_selection_v1(
            [ADMITTED_SURFACES[0], ADMITTED_SURFACES[0]]
        )


def test_regulator_route_cut_points_and_claims_are_explicitly_bounded() -> None:
    payload = GenomeV2P10CapabilityContract.v1().payload
    regulator = payload["fixed_regulator_envelope"]

    assert regulator["contract_state"] == "admitted_technical_surface"
    assert regulator["product_inputs"] == 12
    assert regulator["phenotype_outputs"] == 10
    assert regulator["dynamic_allocation"] is False
    assert payload["route_declaration"]["physical_route_qualification_cells"] == [
        1,
        2,
    ]
    assert payload["route_declaration"]["physical_48_cell_evidence_role"] == (
        "exact_b48_route_qualification_envelope"
    )
    assert payload["checkpoint_declaration"]["native_authority_contract"] == (
        "absolute_tick_v1"
    )
    assert payload["checkpoint_declaration"]["request_fields"] == [
        "total_ticks",
        "checkpoint_ticks",
    ]
    assert payload["checkpoint_declaration"]["restore_cursor_field"] == (
        "next_absolute_tick"
    )
    assert payload["checkpoint_declaration"]["span_semantics"] == (
        "half_open_absolute_tick_span"
    )
    assert payload["capacity_declaration"] == {
        "resident_layout": "soa_slot_last_contiguous_v1",
        "maximum_physical_cells": 48,
        "maximum_resident_capacity_per_cell": 512,
        "ordinary_evolution_can_resize": False,
        "overflow": "reject_before_allocation_no_truncation",
    }
    assert len(payload["cut_point_crosswalk"]) == 2
    assert {
        row["package_cut_point"] for row in payload["cut_point_crosswalk"]
    } == {"terminal_at_total_ticks", "cancellation_at_absolute_tick"}
    assert payload["claims_declaration"][
        "technical_admission_is_scientific_evidence"
    ] is False
    assert "intelligence" in payload["claims_declaration"]["excluded"]
