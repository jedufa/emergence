from __future__ import annotations

import json
from typing import Any

import pytest

from engine.perf.cuda_architecture_b_resident_chunk_runtime_v1 import (
    aggregate_native_chunk_receipts,
    generation_slice,
    pointer_identity_receipt,
    tick_slice,
    validate_chunk_sequence,
)


TICKS = 80


def _receipt(
    generations: int,
    *,
    batch_cells: int = 1,
    generation_start: int = 0,
) -> dict[str, Any]:
    ticks = generations * TICKS
    return {
        "native_controller_invocations": 1,
        "native_dispatches": 1,
        "native_tick_iterations": ticks,
        "phase_1_launches": ticks,
        "birth_prepare_launches": ticks,
        "total_kernel_launches": ticks * 2,
        "final_host_synchronizations": 1,
        "intermediate_device_to_host_operations": 0,
        "final_device_to_host_operations": 2,
        "transfer_count": 2,
        "full_refill_ticks": generations,
        "requested_generations": generations,
        "completed_generations": generations,
        "completed_generations_by_cell": [
            generation_start + generations
        ] * batch_cells,
        "resident_chunk_generation_start": generation_start,
        "resident_chunk_generation_end": generation_start + generations,
        "python_generation_iterations": 0,
        "python_tick_iterations": 0,
        "host_synchronizations_inside_loop": 0,
        "runtime_jit": False,
        "hidden_fallback": False,
        "compact_checkpoint_hash_batches_per_generation": generations + 1,
        "compact_checkpoint_hash_status": "disabled",
    }


def _host_policy_receipt(
    generations: int,
    *,
    batch_cells: int = 1,
    generation_start: int = 0,
) -> dict[str, Any]:
    receipt = _receipt(
        generations,
        batch_cells=batch_cells,
        generation_start=generation_start,
    )
    calls = generations * TICKS
    receipt.update(
        {
            "host_synchronizations_inside_loop": calls * 2,
            "intermediate_device_to_host_operations": calls * 17,
            "intermediate_host_to_device_operations": calls * 13,
            "intermediate_host_synchronizations": calls * 2,
            "aten_operations_inside_tick_loop": calls * 30,
            "aten_operations_inside_tick_loop_count_semantics": (
                "lower_bound_from_explicit_native_host_transfers"
            ),
            "exact_cpu_policy_service_calls_inside_tick_loop": calls,
            "native_host_policy_service_used": True,
        }
    )
    return receipt


def test_chunk_sequences_admit_repeated_g2_and_g64() -> None:
    assert validate_chunk_sequence(4, (2, 2)) == (2, 2)
    assert validate_chunk_sequence(128, [64, 64]) == (64, 64)


@pytest.mark.parametrize("bad_total", [0, -1, True, 4.0])
def test_invalid_totals_fail_closed(bad_total: object) -> None:
    with pytest.raises(ValueError, match="total_generations"):
        validate_chunk_sequence(bad_total, (2, 2))  # type: ignore[arg-type]


@pytest.mark.parametrize("chunks", [(), (2, True), (2, 2.0), (4,), (2,)])
def test_invalid_chunk_sequences_fail_closed(chunks: tuple[object, ...]) -> None:
    with pytest.raises(ValueError):
        validate_chunk_sequence(4, chunks)  # type: ignore[arg-type]


def test_generation_and_tick_slices_are_exact_generation_major_windows() -> None:
    assert generation_slice(128, 64, 64) == slice(64, 128)
    assert tick_slice(128, TICKS, 64, 64) == slice(5_120, 10_240)

    with pytest.raises(ValueError, match="generation slice"):
        generation_slice(128, 65, 64)
    with pytest.raises(ValueError, match="ticks_per_generation"):
        tick_slice(128, True, 0, 64)


def test_additive_chunk_receipt_aggregation_is_exact_and_json_safe() -> None:
    receipts = [_receipt(64), _receipt(64, generation_start=64)]

    aggregate = aggregate_native_chunk_receipts(
        receipts,
        batch_cells=1,
        total_generations=128,
        ticks_per_generation=TICKS,
    )

    assert aggregate["native_controller_invocations"] == 2
    assert aggregate["native_dispatches"] == 2
    assert aggregate["native_chunk_submissions"] == 2
    assert aggregate["python_chunk_control_boundaries"] == 1
    assert aggregate["native_tick_iterations"] == 10_240
    assert aggregate["requested_generations"] == 128
    assert aggregate["completed_generations"] == 128
    assert aggregate["completed_generations_by_cell"] == [128]
    assert aggregate["initial_generation"] == 0
    assert aggregate["terminal_next_generation"] == 128
    assert aggregate["phase_1_launches"] == 10_240
    assert aggregate["birth_prepare_launches"] == 10_240
    assert aggregate["total_kernel_launches"] == 20_480
    assert aggregate["final_host_synchronizations"] == 2
    assert aggregate["final_device_to_host_operations"] == 4
    assert aggregate["transfer_count"] == 4
    assert aggregate["full_refill_ticks"] == 128
    assert aggregate["runtime_jit"] is False
    assert aggregate["hidden_fallback"] is False
    assert aggregate["resident_state_rebuilt_between_chunks"] is False
    assert aggregate["full_state_transfers_between_chunks"] == 0
    assert aggregate["production_native_chunk_loop"] is False
    assert aggregate["chunk_receipts"] == receipts
    assert "compact_checkpoint_hash_batches_per_generation" not in aggregate
    assert "compact_checkpoint_hash_status" not in aggregate
    assert json.loads(json.dumps(aggregate, allow_nan=False)) == aggregate


def test_declared_native_host_policy_receipts_aggregate_truthfully() -> None:
    receipts = [
        _host_policy_receipt(2),
        _host_policy_receipt(2, generation_start=2),
    ]

    aggregate = aggregate_native_chunk_receipts(
        receipts,
        batch_cells=1,
        total_generations=4,
        ticks_per_generation=TICKS,
    )

    assert aggregate["native_host_policy_service_used"] is True
    assert aggregate["exact_cpu_policy_service_calls_inside_tick_loop"] == 320
    assert aggregate["host_synchronizations_inside_loop"] == 640
    assert aggregate["intermediate_host_synchronizations"] == 640
    assert aggregate["intermediate_device_to_host_operations"] == 5_440
    assert aggregate["intermediate_host_to_device_operations"] == 4_160
    assert aggregate["aten_operations_inside_tick_loop"] == 9_600
    assert (
        aggregate["aten_operations_inside_tick_loop_count_semantics"]
        == "lower_bound_from_explicit_native_host_transfers"
    )


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("native_host_policy_service_used", False, "without declaring"),
        (
            "exact_cpu_policy_service_calls_inside_tick_loop",
            0,
            "without any service calls",
        ),
        (
            "host_synchronizations_inside_loop",
            159,
            "must equal two per exact CPU policy service call",
        ),
        (
            "intermediate_host_synchronizations",
            159,
            "must equal host_synchronizations_inside_loop",
        ),
        (
            "aten_operations_inside_tick_loop",
            2_399,
            "must equal the explicit native host-transfer lower bound",
        ),
        (
            "aten_operations_inside_tick_loop_count_semantics",
            "exact_zero",
            "must identify the explicit native host-transfer lower bound",
        ),
    ],
)
def test_native_host_policy_receipt_inconsistency_fails_closed(
    field: str,
    value: object,
    match: str,
) -> None:
    receipt = _host_policy_receipt(1)
    receipt[field] = value

    with pytest.raises(ValueError, match=match):
        aggregate_native_chunk_receipts(
            [receipt],
            batch_cells=1,
            total_generations=1,
            ticks_per_generation=TICKS,
        )


def test_host_policy_service_calls_must_equal_native_ticks() -> None:
    receipt = _host_policy_receipt(1)
    receipt["exact_cpu_policy_service_calls_inside_tick_loop"] = 79
    receipt["host_synchronizations_inside_loop"] = 158
    receipt["intermediate_host_synchronizations"] = 158

    with pytest.raises(ValueError, match="must equal native_tick_iterations"):
        aggregate_native_chunk_receipts(
            [receipt],
            batch_cells=1,
            total_generations=1,
            ticks_per_generation=TICKS,
        )


def test_host_synchronization_counter_remains_mandatory() -> None:
    receipt = _receipt(1)
    del receipt["host_synchronizations_inside_loop"]

    with pytest.raises(ValueError, match="host_synchronizations_inside_loop"):
        aggregate_native_chunk_receipts(
            [receipt],
            batch_cells=1,
            total_generations=1,
            ticks_per_generation=TICKS,
        )


def test_resident_chunks_cannot_change_host_policy_service_mode() -> None:
    receipts = [_receipt(1), _host_policy_receipt(1, generation_start=1)]

    with pytest.raises(ValueError, match="must not change"):
        aggregate_native_chunk_receipts(
            receipts,
            batch_cells=1,
            total_generations=2,
            ticks_per_generation=TICKS,
        )


@pytest.mark.parametrize(
    "receipts",
    [
        [],
        [object()],
        [{"native_dispatches": 1}],
    ],
)
def test_malformed_receipts_fail_closed(receipts: list[object]) -> None:
    with pytest.raises(ValueError):
        aggregate_native_chunk_receipts(  # type: ignore[arg-type]
            receipts,
            batch_cells=1,
            total_generations=2,
            ticks_per_generation=TICKS,
        )


def test_receipt_field_mismatch_fails_closed() -> None:
    receipts = [_receipt(2), _receipt(2, generation_start=2)]
    receipts[1]["completed_generations_by_cell"] = [1]

    with pytest.raises(ValueError, match="completed_generations_by_cell"):
        aggregate_native_chunk_receipts(
            receipts,
            batch_cells=1,
            total_generations=4,
            ticks_per_generation=TICKS,
        )

    receipts = [_receipt(2), _receipt(2, generation_start=2)]
    del receipts[1]["transfer_count"]
    with pytest.raises(ValueError, match="transfer_count"):
        aggregate_native_chunk_receipts(
            receipts,
            batch_cells=1,
            total_generations=4,
            ticks_per_generation=TICKS,
        )


def test_pointer_identity_receipt_reports_first_sorted_mismatch() -> None:
    exact = pointer_identity_receipt(
        {"state": 100, "output": 200},
        {"state": 100, "output": 200},
    )
    assert exact == {
        "all_exact": True,
        "pointer_count": 2,
        "first_mismatch": None,
        "mismatches": [],
    }

    mismatch = pointer_identity_receipt(
        {"state": 100, "output": 200},
        {"state": 101, "scratch": 300},
    )
    assert mismatch["all_exact"] is False
    assert mismatch["first_mismatch"] == {
        "surface": "output",
        "before": 200,
        "after": None,
    }
    assert [row["surface"] for row in mismatch["mismatches"]] == [
        "output",
        "scratch",
        "state",
    ]

    with pytest.raises(ValueError, match="before.state"):
        pointer_identity_receipt({"state": True}, {"state": 1})
