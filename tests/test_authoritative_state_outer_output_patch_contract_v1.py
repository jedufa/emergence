from __future__ import annotations

import json

import pytest
import torch

from engine.authoritative_state.contracts.outer_output_patch_v1 import (
    build_outer_output_patch_contracts_v1,
)
from engine.authoritative_state.runtime_abi import (
    observe_prepared_direct_resident_abi_v1,
)
from engine.perf import cuda_architecture_b_direct_resident_preparation_v1 as preparation
from engine.perf.cuda_architecture_b_direct_resident_preparation_v1 import (
    DIRECT_RESIDENT_ARGUMENT_NAMES_V1,
    PATCH_P2_STATE_FIELDS_V1,
)
from engine.perf.cuda_architecture_b_direct_resident_source_allocator_v1 import (
    GENOME_V2_CAUSAL_RECORD_CAPACITY_PER_CELL_V1,
    ORDINARY_BIRTH_OUTPUT_FIELDS,
    _allocate_birth_outputs,
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


REQUIRED = {
    "name",
    "kind",
    "dtype",
    "rank",
    "shape_formula",
    "capacity_formula",
    "valid_row_count_formula",
    "mutability",
    "stale_tail_semantics",
    "checkpoint_participation",
    "chronicle_participation",
    "provider_participation",
    "analysis_participation",
    "profile_availability",
    "semantic_owner",
    "source",
    "boundary",
    "signedness",
    "bit_width",
    "valid_range",
    "sentinels",
    "metadata_status",
}


def test_contract_orders_and_metadata_are_complete() -> None:
    contracts = build_outer_output_patch_contracts_v1()

    assert tuple(row["name"] for row in contracts["direct_resident_outer"]) == (
        DIRECT_RESIDENT_ARGUMENT_NAMES_V1
    )
    assert tuple(row["name"] for row in contracts["ordinary_birth_output"]) == (
        ORDINARY_BIRTH_OUTPUT_FIELDS
    )
    assert tuple(row["name"] for row in contracts["ordinary_birth_scratch"]) == (
        "lifecycle_scratch_storage",
    )
    assert tuple(row["name"] for row in contracts["patch_p2_state"]) == (
        PATCH_P2_STATE_FIELDS_V1
    )
    assert all(
        REQUIRED <= row.keys() and row["metadata_status"] == "supported_exact"
        for rows in contracts.values()
        for row in rows
    )
    json.dumps(contracts, allow_nan=False, sort_keys=True)


def test_patch_context_is_exact_prefix_and_full_mode_adds_three_grids() -> None:
    rows = build_outer_output_patch_contracts_v1()["patch_p2_state"]
    context = tuple(
        row["name"]
        for row in rows
        if "patch_p2_state_context_7" in row["variants"]
    )
    full = tuple(
        row["name"]
        for row in rows
        if "patch_p2_state_full_10" in row["variants"]
    )

    assert context == PATCH_P2_STATE_FIELDS_V1[:7]
    assert full == PATCH_P2_STATE_FIELDS_V1
    assert full[7:] == ("patch_cover", "patch_shelter", "patch_movement_cost")


def test_outer_nested_vectors_are_ordered_container_contracts() -> None:
    rows = {
        row["name"]: row
        for row in build_outer_output_patch_contracts_v1()["direct_resident_outer"]
    }

    assert rows["ordinary_birth_output_tensors"]["ordered_count"] == 63
    assert rows["turnover_output_tensors"]["ordered_count"] == 16
    assert rows["ordinary_birth_output_tensors"]["child_contract_family"] == (
        "ordinary_birth_output"
    )
    assert rows["ordinary_birth_scratch_tensors"]["ordered_count"] == 1
    assert rows["patch_p2_state_tensors"]["ordered_count"] == 10
    assert rows["event_state_tensors"]["ordered_count"] == 31
    assert rows["full_tick_state_tensors"]["ordered_count"] == 29
    assert rows["physical_full_tick_output_tensors"]["ordered_count"] == 20


def test_outer_move_cost_is_exact_batched_dispatch_tensor() -> None:
    row = {
        item["name"]: item
        for item in build_outer_output_patch_contracts_v1()["direct_resident_outer"]
    }["config.move_cost"]

    assert row["ordinal"] == DIRECT_RESIDENT_ARGUMENT_NAMES_V1.index(
        "config.move_cost"
    )
    assert row["kind"] == "tensor"
    assert row["dtype"] == "float64"
    assert row["rank"] == 1
    assert row["shape_formula"] == "[B]"
    assert row["capacity_formula"] == "B"
    assert row["valid_row_count_formula"] == "B"


def test_outer_contract_reconciles_with_prepared_receipt_fixture() -> None:
    contract = build_outer_output_patch_contracts_v1()["direct_resident_outer"]
    observed = observe_prepared_direct_resident_abi_v1(_prepared("cpu"))[
        "outer_arguments"
    ]

    assert len(contract) == len(observed) == len(DIRECT_RESIDENT_ARGUMENT_NAMES_V1)
    for expected, actual in zip(contract, observed, strict=True):
        assert expected["ordinal"] == actual["ordinal"]
        assert expected["name"] == actual["name"]
        if expected["kind"] == "ordered_tensor_container":
            assert actual["kind"] == "tensor_sequence"
            assert expected["ordered_count"] == actual["nested_count"]
        elif expected["kind"] == "ordered_mapping":
            assert actual["kind"] == "mapping"
        else:
            assert expected["kind"] == actual["kind"]


def test_life_history_outer_contract_slots_are_source_ordered_and_typed() -> None:
    rows = {
        row["name"]: row
        for row in build_outer_output_patch_contracts_v1()["direct_resident_outer"]
    }

    expected = {
        "config.life_history_mode": "int64",
        "config.life_history_maturity_start_ticks": "int64",
        "config.life_history_maturity_rise_ticks": "int64",
        "config.life_history_fertility_decline_start_ticks": "int64",
        "config.life_history_fertility_decline_ticks": "int64",
        "config.life_history_elder_fertility_floor": "float64",
        "config.life_history_senescence_start_ticks": "int64",
        "config.life_history_senescence_ramp_ticks": "int64",
        "config.life_history_senescence_energy_cost": "float64",
        "config.life_history_senescence_health_cost": "float64",
        "config.life_history_max_age_ticks": "int64",
    }

    for name, dtype in expected.items():
        row = rows[name]
        assert row["ordinal"] == DIRECT_RESIDENT_ARGUMENT_NAMES_V1.index(name)
        assert row["kind"] == "scalar"
        assert row["dtype"] == dtype
        assert row["mutability"] == "read_only"
        assert row["provider_participation"] == "dispatch_input"


def _shape(formula: str, **values: int) -> tuple[int, ...]:
    return tuple(
        values[token] if token in values else int(token)
        for token in formula.strip("[]").split(",")
    )


def test_birth_output_contract_matches_real_allocator_dtype_rank_and_shape() -> None:
    values = {"B": 2, "O": 5, "S": 64, "M": 3}
    allocated = _allocate_birth_outputs(
        capacity=values["S"],
        max_births_per_tick=values["M"],
        output_ticks=values["O"],
        batch_cells=values["B"],
        device=torch.device("cpu"),
        genome_v2_causal_record_capacity=(
            GENOME_V2_CAUSAL_RECORD_CAPACITY_PER_CELL_V1
        ),
    )
    rows = build_outer_output_patch_contracts_v1()["ordinary_birth_output"]

    for row in rows:
        tensor = allocated[row["name"]]
        assert str(tensor.dtype).removeprefix("torch.") == row["dtype"], row["name"]
        assert tensor.ndim == row["rank"], row["name"]
        assert tuple(tensor.shape) == _shape(row["shape_formula"], **values), row[
            "name"
        ]


def test_allocator_sentinels_and_physical_widths_are_exact() -> None:
    contracts = build_outer_output_patch_contracts_v1()
    birth = {row["name"]: row for row in contracts["ordinary_birth_output"]}
    allocated = _allocate_birth_outputs(
        capacity=64,
        max_births_per_tick=3,
        output_ticks=2,
        batch_cells=2,
        device=torch.device("cpu"),
    )

    for name, row in birth.items():
        tensor = allocated[name]
        if row["sentinels"] == [-1]:
            assert bool(tensor.eq(-1).all()), name
        elif row["sentinels"] == ["NaN"]:
            assert bool(torch.isnan(tensor).all()), name
        assert row["bit_width"] == tensor.element_size() * 8
        assert row["signedness"] in {"signed", "unsigned"}
        assert row["valid_range"]

    scratch = contracts["ordinary_birth_scratch"][0]
    assert scratch["dtype"] == "uint8"
    assert scratch["signedness"] == "unsigned"
    assert scratch["bit_width"] == 8
    assert scratch["valid_range"] == "[0,255]"
    assert scratch["sentinels"] == []

    patch = contracts["patch_p2_state"]
    assert all(row["signedness"] == "signed" for row in patch)
    assert all(row["bit_width"] == 32 for row in patch)
    assert all(row["sentinels"] == [] for row in patch)
