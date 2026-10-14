from __future__ import annotations

import pytest
import torch

from engine.authoritative_state.contracts.full_tick_event_v1 import (
    build_full_tick_event_contracts_v1,
    reconcile_full_tick_event_receipt_v1,
    unresolved_full_tick_event_entries_v1,
)
from engine.authoritative_state.errors import AuthoritativeStateContractError
from engine.authoritative_state.runtime_abi import (
    observe_prepared_direct_resident_abi_v1,
)
from tests.test_authoritative_state_runtime_abi_v1 import (
    BATCH,
    CAPACITY,
    EVENT_CAPACITY,
    _prepared,
)


def _exact_prepared():
    prepared = _prepared("cpu")
    source = prepared.source
    device = torch.device("cpu")
    drive_width = int(source.live_state["drives"].shape[2])
    noise_ticks = 5
    chemistry_width = 6

    full: list[torch.Tensor] = []
    for ordinal in range(29):
        dtype = torch.float32
        shape = (BATCH, CAPACITY)
        if ordinal in {0, 13, 14}:
            dtype = torch.int32
        elif ordinal == 6:
            shape = (BATCH, CAPACITY, drive_width)
        elif ordinal == 15:
            shape = (BATCH, CAPACITY, chemistry_width)
        elif ordinal in {16, 18}:
            dtype = torch.uint64
            shape = (BATCH, 4)
        elif ordinal in {17, 19}:
            dtype = torch.int64
            shape = (BATCH, 2)
        elif ordinal == 20:
            shape = (BATCH, noise_ticks, CAPACITY, drive_width)
        elif ordinal == 21:
            dtype = torch.bool
            shape = (BATCH, noise_ticks, CAPACITY)
        elif ordinal in {22, 24, 25}:
            dtype = torch.int64
        elif ordinal in {23, 26, 27, 28}:
            dtype = torch.int64
            shape = (BATCH,)
        full.append(torch.zeros(shape, dtype=dtype, device=device))
    full[21][0, 0, :3] = True
    full[21][1, 1, :2] = True
    full[21][2, 2, :1] = True
    full[22] = source.live_state["slot_free_queue"]
    full[23] = source.live_state["slot_free_queue_count"]
    full[26].copy_(torch.tensor([2, 1, 0], dtype=torch.int64))
    source.full_tick_state_tensors[:] = full

    event_dtypes = (
        torch.int32, torch.int64, torch.int64, torch.int64, torch.int64,
        torch.float32, torch.int32, torch.int64, torch.int64, torch.int64,
        torch.int64, torch.int64, torch.int64, torch.int64, torch.int64,
        torch.int64, torch.int64, torch.int32, torch.int32, torch.int32,
        torch.int8, torch.float32,
    )
    event = [
        torch.zeros((BATCH, EVENT_CAPACITY), dtype=dtype)
        for dtype in event_dtypes
    ]
    event_count = torch.tensor([3, 2, 1], dtype=torch.int64)
    initial_ids = torch.full(
        (BATCH, 4),
        torch.iinfo(torch.int64).max,
        dtype=torch.int64,
    )
    initial_ids[0, :4] = torch.arange(4)
    initial_ids[1, :3] = torch.arange(3)
    initial_ids[2, :2] = torch.arange(2)
    event.extend(
        [
            event_count,
            torch.zeros((BATCH,), dtype=torch.int32),
            torch.full((BATCH, CAPACITY), -1, dtype=torch.int32),
            torch.zeros((BATCH, CAPACITY), dtype=torch.int32),
            initial_ids,
            torch.zeros((BATCH, 4), dtype=torch.int32),
            torch.zeros((BATCH, 4), dtype=torch.int32),
            torch.zeros((BATCH, 160), dtype=torch.int32),
            torch.zeros((BATCH,), dtype=torch.int64),
        ]
    )
    source.event_state_tensors[:] = event
    return prepared


def _by_name(rows: tuple[dict[str, object], ...]) -> dict[str, dict[str, object]]:
    return {str(row["name"]): row for row in rows}


def _by_path(rows: tuple[dict[str, object], ...]) -> dict[str, dict[str, object]]:
    return {str(row["path"]): row for row in rows}


def test_contract_is_exact_complete_and_in_abi_order() -> None:
    contracts = build_full_tick_event_contracts_v1()

    assert len(contracts["full_tick_state"]) == 29
    assert len(contracts["event_state"]) == 31
    assert [row["ordinal"] for row in contracts["full_tick_state"]] == list(range(29))
    assert [row["ordinal"] for row in contracts["event_state"]] == list(range(31))
    assert all(
        row["metadata_status"] == "supported_exact"
        for rows in contracts.values()
        for row in rows
    )
    assert all(
        {"signedness", "bit_width", "valid_range", "sentinels"} <= row.keys()
        for rows in contracts.values()
        for row in rows
    )
    assert unresolved_full_tick_event_entries_v1() == ()

    full = _by_name(contracts["full_tick_state"])
    event = _by_name(contracts["event_state"])
    assert full["cost_charge_scalar"]["mutability"] == "read_only"
    assert full["cost_charge_maturation_scalar"]["mutability"] == "read_only"
    assert full["age"]["mutability"] == "read_write"
    assert event["initial_agent_ids_by_row"]["mutability"] == "read_only"
    assert event["event_tape_type"]["mutability"] == "read_write"
    assert full["runner_pcg64_state_words"]["signedness"] == "unsigned"
    assert full["runner_pcg64_state_words"]["bit_width"] == 64
    assert event["typed_lifecycle_event_success"]["valid_range"] == "-128..127"
    assert event["initial_agent_ids_by_row"]["sentinels"] == (
        {
            "value": 9223372036854775807,
            "meaning": "unused_reporting_row",
        },
    )
    assert full["residual_ids"]["valid_row_count_formula"] == (
        "sum(residual_count_by_cell)"
    )
    assert full["drive_noise_samples_by_tick_slot"][
        "valid_row_count_formula"
    ] == "count_true(drive_noise_samples_valid_by_tick_slot)*D"


def test_reconciliation_corrects_family_mutability_and_ragged_validity() -> None:
    prepared = _exact_prepared()
    receipt = observe_prepared_direct_resident_abi_v1(prepared)
    reconciled = reconcile_full_tick_event_receipt_v1(receipt, prepared=prepared)
    full = _by_path(reconciled["full_tick_state"])
    event = _by_path(reconciled["event_state"])

    assert full["full_tick_state_tensors.cost_charge_scalar"]["mutability"] == (
        "read_only"
    )
    assert full["full_tick_state_tensors.residual_ids"]["valid_count"] == 3
    assert full["full_tick_state_tensors.residual_slots"]["valid_count"] == 3
    assert full[
        "full_tick_state_tensors.drive_noise_samples_valid_by_tick_slot"
    ]["valid_count"] == 6
    assert full["full_tick_state_tensors.drive_noise_samples_by_tick_slot"][
        "valid_count"
    ] == 24
    assert event["event_state_tensors.event_tape_type"]["valid_count"] == 6
    assert event["event_state_tensors.initial_agent_ids_by_row"]["valid_count"] == 9
    assert event["event_state_tensors.terminal_survival"]["valid_count"] == 9
    assert event["event_state_tensors.initial_agent_ids_by_row"]["mutability"] == (
        "read_only"
    )


def test_reconciliation_ignores_declared_stale_tails_beyond_ragged_prefixes() -> None:
    prepared = _exact_prepared()
    source = prepared.source
    source.full_tick_state_tensors[22][0, 2:] = 2**31 - 1
    source.full_tick_state_tensors[24][0, 2:] = 2**63 - 1
    source.event_state_tensors[0][0, 3:] = 2**31 - 1

    receipt = observe_prepared_direct_resident_abi_v1(prepared)
    reconciled = reconcile_full_tick_event_receipt_v1(receipt, prepared=prepared)
    full = _by_path(reconciled["full_tick_state"])
    event = _by_path(reconciled["event_state"])

    assert full["full_tick_state_tensors.slot_free_queue"]["valid_count"] == 3
    assert full["full_tick_state_tensors.residual_ids"]["valid_count"] == 3
    assert event["event_state_tensors.event_tape_type"]["valid_count"] == 6
    assert (
        full["full_tick_state_tensors.slot_free_queue"]["stale_tail_semantics"]
        == "entries_after_each_cell_queue_count_are_stale"
    )
    assert (
        event["event_state_tensors.event_tape_type"]["stale_tail_semantics"]
        == "entries_after_each_cell_event_active_count_are_stale"
    )


def test_reconciliation_fails_closed_on_binding_dtype_mismatch() -> None:
    prepared = _exact_prepared()
    prepared.source.event_state_tensors[20] = prepared.source.event_state_tensors[
        20
    ].to(torch.int32)
    receipt = observe_prepared_direct_resident_abi_v1(prepared)

    with pytest.raises(
        AuthoritativeStateContractError,
        match="typed_lifecycle_event_success: dtype must be int8",
    ):
        reconcile_full_tick_event_receipt_v1(receipt, prepared=prepared)


def test_reconciliation_fails_closed_on_invalid_residual_count() -> None:
    prepared = _exact_prepared()
    prepared.source.full_tick_state_tensors[26][0] = CAPACITY + 1
    receipt = observe_prepared_direct_resident_abi_v1(prepared)

    with pytest.raises(
        AuthoritativeStateContractError,
        match=r"residual_count: values must be within \[0,64\]",
    ):
        reconcile_full_tick_event_receipt_v1(receipt, prepared=prepared)


def test_reconciliation_fails_closed_on_event_active_count_overflow() -> None:
    prepared = _exact_prepared()
    prepared.source.event_state_tensors[22][0] = EVENT_CAPACITY + 1
    receipt = observe_prepared_direct_resident_abi_v1(prepared)

    with pytest.raises(
        AuthoritativeStateContractError,
        match=(
            r"event_state_tensors\.event_active_count: "
            r"values must be within \[0,32\]"
        ),
    ):
        reconcile_full_tick_event_receipt_v1(receipt, prepared=prepared)
