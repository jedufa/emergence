from __future__ import annotations

import pytest

from engine.authoritative_state.contracts.ordinary_turnover_v1 import (
    build_ordinary_turnover_contracts_v1,
    ordinary_birth_state_names_v1,
    reconcile_ordinary_turnover_contracts_v1,
)
from engine.authoritative_state.runtime_abi import (
    observe_prepared_direct_resident_abi_v1,
)
from engine.perf import (
    cuda_architecture_b_direct_resident_preparation_v1 as preparation,
)
from engine.perf.cuda_architecture_b_direct_resident_source_allocator_v1 import (
    TURNOVER_MUTABLE_STATE_FIELDS,
)
from tests.test_authoritative_state_runtime_abi_v1 import _prepared


@pytest.fixture(autouse=True)
def _allow_cpu_contract_fixture(monkeypatch: pytest.MonkeyPatch) -> None:
    original = preparation._require_turnover_selection_seed_by_cell

    def validate(value, **kwargs):
        if value.device.type == "cpu":
            return value
        return original(value, **kwargs)

    monkeypatch.setattr(
        preparation,
        "_require_turnover_selection_seed_by_cell",
        validate,
    )


def test_contracts_close_orders_and_genome_v2_shapes() -> None:
    contracts = build_ordinary_turnover_contracts_v1()
    ordinary = contracts["ordinary_birth_state"]
    turnover = contracts["turnover_mutable_state"]

    assert len(ordinary) == 139
    assert len(turnover) == 122
    assert tuple(row["name"] for row in turnover) == TURNOVER_MUTABLE_STATE_FIELDS
    turnover_by_name = {row["name"]: row for row in turnover}
    assert turnover_by_name["active_agent_slots"]["dtype"] == "int32"
    assert turnover_by_name["active_agent_slots"]["shape_formula"] == (
        "[cell,capacity]"
    )
    assert all(row["metadata_status"] == "supported_exact" for row in ordinary)
    assert all(row["metadata_status"] == "supported_exact" for row in turnover)
    assert all(
        row["signedness"]
        and isinstance(row["bit_width"], int)
        and row["valid_range"]
        and row["sentinels"]
        for row in (*ordinary, *turnover)
    )

    by_name = {row["name"]: row for row in ordinary}
    assert by_name["inheritance_mode"]["shape_formula"] == "[cell]"
    assert by_name["diploid_allele_storage"]["shape_formula"] == (
        "[cell,34,capacity]"
    )
    assert by_name["diploid_allele_shadow_storage"]["shape_formula"] == (
        "[cell,34,capacity]"
    )
    assert by_name["genome_resource_handling_axis"]["shape_formula"] == (
        "[cell,capacity]"
    )
    assert by_name["canonical_hereditary_digest"]["shape_formula"] == (
        "[cell,capacity,32]"
    )
    assert by_name["canonical_hereditary_digest"]["dtype"] == "uint8"
    assert by_name["genome_upgrade_v2_alleles_q10"]["shape_formula"] == (
        "[cell,2,12,4,capacity]"
    )
    assert by_name["genome_upgrade_v2_modifiers_q10"]["shape_formula"] == (
        "[cell,2,12,capacity]"
    )
    assert by_name["genome_upgrade_v2_phenotypes_q10"]["shape_formula"] == (
        "[cell,10,capacity]"
    )
    assert by_name["genome_upgrade_v2_expression_cost_q10"]["dtype"] == "int64"
    assert (
        by_name["genome_upgrade_v2_alleles_q10_shadow"]["mutability"]
        == "transaction_shadow_write_then_commit"
    )
    assert (
        by_name["genome_upgrade_v2_alleles_q10_shadow"][
            "checkpoint_participation"
        ]
        == "transaction shadow; not checkpoint authority"
    )
    assert tuple(row["name"] for row in ordinary) == (
        ordinary_birth_state_names_v1(139)
    )


def test_legacy_prepared_owner_reconciles_without_ambiguity() -> None:
    receipt = observe_prepared_direct_resident_abi_v1(_prepared("cpu"))
    result = reconcile_ordinary_turnover_contracts_v1(
        receipt,
        ordinary_variant_count=113,
    )

    # This older synthetic prepared-owner fixture intentionally represents all
    # unspecified lifecycle members as float32 rank-2 tensors.  A contradiction
    # must be reported rather than relabelled exact.
    assert result["ambiguities"]
    assert any(
        row["family"] == "ordinary_birth_state"
        and row["name"] == "slot_to_agent_id"
        and "dtype expected=int64 observed=float32" in row["reasons"]
        for row in result["ambiguities"]
    )


def test_turnover_disabled_variant_accepts_empty_prepared_rows() -> None:
    receipt = observe_prepared_direct_resident_abi_v1(
        _prepared(
            "cpu",
            generation_turnover_enabled=False,
            lifecycle_mode=1,
        )
    )
    result = reconcile_ordinary_turnover_contracts_v1(
        receipt,
        ordinary_variant_count=139,
        turnover_variant_count=0,
    )

    assert result["contracts"]["turnover_mutable_state"] == ()
    assert not any(
        row["family"] == "turnover_mutable_state"
        for row in result["ambiguities"]
    )


def test_exact_receipt_rows_reconcile_and_contradictions_fail_closed() -> None:
    static = build_ordinary_turnover_contracts_v1(ordinary_variant_count=113)
    tensors = []
    shape_values = {
        "cell": 3,
        "capacity": 64,
        "agent_id_capacity": 70,
        "height": 4,
        "width": 5,
        "policy_tick_stride": 80,
        "drive_channels": 4,
        "chemistry_channels": 3,
        "mlp_pack_width": 16,
        "gru_hidden": 8,
        "cpython_gauss_tape_capacity": 32,
        "mutation_rng_state_bytes": 64,
        "ordinary_birth_capacity": 6,
        "policy_crossover_packed_row_bytes": 2,
    }
    for family, prefix in (
        ("ordinary_birth_state", "ordinary_birth_state_tensors"),
        ("turnover_mutable_state", "turnover_state_tensors"),
    ):
        for row in static[family]:
            shape = [
                value if isinstance(value, int) else shape_values[value]
                for value in row["shape_dimensions"]
            ]
            tensors.append(
                {
                    "path": f"{prefix}.{row['name']}",
                    "dtype": row["dtype"],
                    "rank": row["rank"],
                    "shape": shape,
                    "capacity_count": 1,
                    "valid_count": 1,
                    "valid_count_semantics": "fixture_exact",
                    "mutability": (
                        "read_write"
                        if family == "turnover_mutable_state"
                        else "read_write"
                    ),
                }
            )
    receipt = {"tensors": tensors}
    result = reconcile_ordinary_turnover_contracts_v1(
        receipt,
        ordinary_variant_count=113,
    )
    assert result["ambiguities"] == ()
    assert all(
        row["prepared_owner_reconciliation"] == "exact_match"
        for rows in result["contracts"].values()
        for row in rows
    )

    receipt["tensors"][0]["dtype"] = "float32"
    with pytest.raises(ValueError, match="slot_to_agent_id"):
        build_ordinary_turnover_contracts_v1(
            receipt,
            ordinary_variant_count=113,
        )
