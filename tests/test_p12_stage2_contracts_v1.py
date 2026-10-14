from __future__ import annotations

import json

import pytest

from engine.evolution.p12_stage2_contracts_v1 import (
    CUDA_SHADOW_DISPOSITION_ID,
    EED_CURSOR_CONTRACT_ID,
    PUBLIC_IDENTITY_CONTRACT_ID,
    RECURRENT_ORACLE_CONTRACT_ID,
    RESPEC_CURSOR_CONTRACT_ID,
    STRUCTURAL_ORACLE_CONTRACT_ID,
    WDL_CAPACITY_CONTRACT_ID,
    P12Stage2ContractError,
    canonical_contract_bytes_v1,
    cuda_shadow_disposition_v1,
    cuda_shadow_route_audit_v1,
    cursor_contract_v1,
    dense_identity_route_envelope_v1,
    mutation_oracle_contracts_v1,
    public_identity_contract_v1,
    validate_cursor_token_v1,
    wdl_capacity_contract_v1,
)


def test_public_identity_contract_selects_slot_handle_abi() -> None:
    contract = public_identity_contract_v1()

    assert contract["contract_id"] == PUBLIC_IDENTITY_CONTRACT_ID
    assert contract["architecture"] == "resident_slot_handle_public_grid"
    assert contract["grid_storage"] == "int32 resident slot handle"
    assert contract["stable_biological_id_role"] == "history/reporting identity only"
    assert "implemented in Python reference" in contract["current_python_status"]
    assert contract["contract_sha256"]


def test_cuda_shadow_is_quarantined_not_authority() -> None:
    disposition = cuda_shadow_disposition_v1()

    assert disposition["contract_id"] == CUDA_SHADOW_DISPOSITION_ID
    assert disposition["field"] == "agent_id_to_slot"
    assert disposition["authoritative"] is False
    assert disposition["high_id_lookup_allowed"] is False
    assert disposition["production_authority_allowed"] is False
    assert "legacy/deprecated CUDA kernels" in disposition["current_route_status"]


def test_cuda_shadow_route_audit_quarantines_current_and_legacy_use() -> None:
    audit = cuda_shadow_route_audit_v1(".")

    assert audit["contract_id"] == CUDA_SHADOW_DISPOSITION_ID
    assert audit["disposition"] == "quarantine_non_authoritative_compatibility_projection"
    assert audit["production_shadow_exported"] is True
    assert audit["absence_changes_current_route_abi"] is True
    assert audit["current_route_dense_lookup_reads"] == ()
    assert {item["direct_access_count"] for item in audit["current_routes"]} == {2}
    assert set(audit["legacy_dense_lookup_debt_files"]) == {
        "engine/perf/cuda_architecture_b_phased_controller_v1_kernels.cu",
        "engine/perf/cuda_architecture_b_stackless_birth_generation_kernel.cu",
        "engine/perf/cuda_architecture_b_persistent_generation_kernel.cu",
    }


def test_wdl_capacity_projection_scales_by_cells_not_stable_id() -> None:
    contract = wdl_capacity_contract_v1()
    projections = contract["projections"]

    assert contract["contract_id"] == WDL_CAPACITY_CONTRACT_ID
    assert projections["b1"]["unique_agent_rows_max"] == 256
    assert projections["b72"]["unique_agent_rows_max"] == 72 * 256
    assert projections["b72"]["event_rows_max_per_generation"] == 72 * 256 * 4
    assert projections["b72"]["overflow_behavior"] == "fail_closed_not_evaluable"
    assert "stable biological id" in contract["capacity_basis"]


def test_dense_identity_route_envelope_scales_by_capacity_for_ragged_cells() -> None:
    low = dense_identity_route_envelope_v1(
        cells=3,
        active_counts=(4, 1, 0),
        max_stable_biological_id=10,
    )
    high = dense_identity_route_envelope_v1(
        cells=3,
        active_counts=(4, 1, 0),
        max_stable_biological_id=1_000_000_007,
    )

    assert low["contract_id"] == "p12-dense-id-route-envelope-v1"
    assert low["ragged_multicell"] is True
    assert low["slot_to_agent_id_rows"] == 3 * 256
    assert low["agent_id_to_slot_shadow_rows"] == 3 * 256
    assert low["active_identity_pair_rows"] == 5
    assert high["slot_to_agent_id_rows"] == low["slot_to_agent_id_rows"]
    assert high["active_identity_pair_rows"] == low["active_identity_pair_rows"]
    assert high["does_not_scale_with"] == "max(stable_biological_id)+1"


def test_dense_identity_route_envelope_fails_closed_on_int64_sentinel() -> None:
    with pytest.raises(P12Stage2ContractError, match="INT64_MAX"):
        dense_identity_route_envelope_v1(
            cells=1,
            active_counts=(1,),
            max_stable_biological_id=(1 << 63) - 1,
        )


def test_cursor_contracts_reject_stale_slot_tokens() -> None:
    eed = cursor_contract_v1(EED_CURSOR_CONTRACT_ID)
    respec = cursor_contract_v1(RESPEC_CURSOR_CONTRACT_ID)

    assert eed["owner_key"] == "resident_slot_plus_slot_token"
    assert respec["stale_token"] == "reject"
    validate_cursor_token_v1(current_token=4, supplied_token=4)
    with pytest.raises(P12Stage2ContractError, match="stale slot-token"):
        validate_cursor_token_v1(current_token=4, supplied_token=3)


def test_mutation_oracle_contracts_stop_before_native_cuda_promotion() -> None:
    contract = mutation_oracle_contracts_v1()

    assert contract["structural_contract_id"] == STRUCTURAL_ORACLE_CONTRACT_ID
    assert contract["recurrent_contract_id"] == RECURRENT_ORACLE_CONTRACT_ID
    assert contract["native_cuda_promotion_authorized"] is False
    assert contract["hidden_width_resize_allowed"] is False
    assert "python reference oracle implemented" in contract["structural_status"]
    assert "python reference oracle implemented" in contract["recurrent_status"]


def test_contract_serialization_is_canonical_ascii() -> None:
    wire = canonical_contract_bytes_v1(wdl_capacity_contract_v1())

    assert wire.endswith(b"\n")
    assert json.loads(wire.decode("ascii"))["contract_id"] == WDL_CAPACITY_CONTRACT_ID
