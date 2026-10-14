from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest

from engine.campaigns.tick_native_performance_timeline_v1 import (
    TICK_NATIVE_PERFORMANCE_TIMELINE_SCHEMA_VERSION,
    TickNativePerformanceTimelineError,
    derive_tick_native_performance_timeline_v1,
)


def _receipt() -> dict[str, Any]:
    return {
        "clock_domain": "perf_counter_ns",
        "wall_time": {"start_ns": 0, "end_ns": 1_000},
        "owner_preparation_receipts": [
            {
                "start_ns": 0,
                "end_ns": 50,
                "owner_id": "owner-0",
                "owner_span": {"start_next_tick": 0, "end_next_tick": 1120},
            }
        ],
        "simulation_receipts": [
            {
                "start_ns": 100,
                "end_ns": 300,
                "start_next_tick": 0,
                "end_next_tick": 1120,
            },
            {
                "start_ns": 500,
                "end_ns": 700,
                "start_next_tick": 1120,
                "end_next_tick": 2240,
            },
        ],
        "checkpoint_receipts": [
            {
                "checkpoint_tick": 1120,
                "owner_span": {"start_next_tick": 0, "end_next_tick": 1120},
                "detach": {"start_ns": 300, "end_ns": 350},
                "persist": {
                    "start_ns": 350,
                    "end_ns": 900,
                    "async": True,
                },
                "owner_preparation": {"start_ns": 400, "end_ns": 500},
            }
        ],
        "gpu_performance_window": {
            "start_ns": 100,
            "end_ns": 900,
            "terminal_checkpoint_tick": 2_240,
            "post_simulation_acs_tail_excluded": True,
        },
        "acs_tail_receipt": {"start_ns": 900, "end_ns": 1_000},
        "restore_receipt": {
            "status": "completed",
            "checkpoint_tick": 0,
            "start_ns": 50,
            "end_ns": 100,
        },
    }


def test_timeline_emits_strict_phase_accounting_and_explicit_async_overlap() -> None:
    timeline = derive_tick_native_performance_timeline_v1(_receipt())

    assert timeline["schema_version"] == (
        TICK_NATIVE_PERFORMANCE_TIMELINE_SCHEMA_VERSION
    )
    assert timeline["strict"] is True
    assert timeline["durations_ns"] == {
        "T_SIM": 400,
        "T_DETACH": 50,
        "T_PERSIST": 550,
        "T_ACS_TAIL": 100,
        "T_RESTORE": 50,
        "OWNER_PREPARATION": 150,
    }
    assert timeline["T_PERSIST_service_seconds"] == pytest.approx(0.00000055)
    assert timeline["checkpoint_coordinates"][0]["checkpoint_tick"] == 1120
    assert timeline["owner_preparation"]["root_interval_count"] == 1
    assert timeline["owner_preparation"]["checkpoint_compatibility_interval_count"] == 1
    assert timeline["owner_preparation"]["intervals"][0]["source"] == "root"
    assert (
        timeline["checkpoint_coordinates"][0]["OWNER_PREPARATION"]["source"]
        == "checkpoint_compatibility"
    )
    assert timeline["restore"]["checkpoint_tick"] == 0
    assert timeline["gpu_performance_window"] == {
        "start_ns": 100,
        "end_ns": 900,
        "duration_ns": 800,
        "seconds": pytest.approx(0.0000008),
        "terminal_checkpoint_tick": 2_240,
        "post_simulation_acs_tail_excluded": True,
        "measurement_scope": (
            "first_simulation_dispatch_through_terminal_gpu_safe_release_v1"
        ),
    }
    assert timeline["dispatch_overhead"] == {
        "definition": "unclassified_simulation_dispatch_domain_fraction_v1",
        "dispatch_domain_start_ns": 100,
        "dispatch_domain_end_ns": 700,
        "dispatch_span_ns": 600,
        "simulation_dispatch_union_ns": 400,
        "classified_non_dispatch_union_ns": 200,
        "capture_detach_union_ns": 50,
        "persistence_union_ns": 350,
        "owner_preparation_union_ns": 100,
        "dispatch_gap_ns": 0,
        "dispatch_domain_ns": 400,
        "acs_tail_excluded": True,
        "fraction": pytest.approx(0.0),
    }
    assert timeline["dispatch_overhead_fraction"] == pytest.approx(0.0)
    assert timeline["post_simulation_acs_tail"][
        "excluded_from_gpu_performance_metrics"
    ] is True
    assert timeline["wall_time_reconciliation"]["unclassified_ns"] == 0
    assert timeline["wall_time_reconciliation"]["reconciliation_error_ratio"] == 0


def test_not_requested_restore_is_explicit_and_zero_duration() -> None:
    receipt = _receipt()
    receipt["owner_preparation_receipts"] = []
    receipt["restore_receipt"] = {"status": "not_requested"}
    receipt["wall_time"]["start_ns"] = 100
    timeline = derive_tick_native_performance_timeline_v1(receipt)

    assert timeline["T_RESTORE_seconds"] == 0
    assert timeline["restore"] == {
        "status": "not_requested",
        "checkpoint_tick": None,
        "interval": None,
    }


def test_checkpoint_owner_preparation_is_optional_compatibility() -> None:
    receipt = _receipt()
    receipt["checkpoint_receipts"][0].pop("owner_preparation")

    timeline = derive_tick_native_performance_timeline_v1(receipt)

    assert timeline["owner_preparation"]["root_interval_count"] == 1
    assert timeline["owner_preparation"]["checkpoint_compatibility_interval_count"] == 0
    assert timeline["owner_preparation"]["duration_ns"] == 50
    assert timeline["checkpoint_coordinates"][0]["OWNER_PREPARATION"] is None


def test_root_and_checkpoint_owner_preparation_must_not_duplicate() -> None:
    receipt = _receipt()
    receipt["owner_preparation_receipts"] = [
        {"start_ns": 400, "end_ns": 500}
    ]

    with pytest.raises(TickNativePerformanceTimelineError, match="duplicate"):
        derive_tick_native_performance_timeline_v1(receipt)


def test_administrative_interval_fills_gap_without_entering_primary_phases() -> None:
    receipt = _receipt()
    receipt["acs_tail_receipt"] = {"start_ns": 900, "end_ns": 950}
    receipt["gpu_performance_window"]["end_ns"] = 900
    receipt["administrative_receipts"] = [
        {
            "start_ns": 950,
            "end_ns": 1_000,
            "reason": "provider bookkeeping finalization",
        }
    ]

    timeline = derive_tick_native_performance_timeline_v1(receipt)

    assert timeline["administrative"]["present"] is True
    assert timeline["administrative"]["duration_ns"] == 50
    assert timeline["administrative"]["reconciles_unclassified_wall_time"] is True
    assert "ADMINISTRATIVE" not in timeline["durations_ns"]
    assert timeline["wall_time_reconciliation"]["unclassified_ns"] == 0


def test_administrative_interval_is_rejected_when_no_gap_exists() -> None:
    receipt = _receipt()
    receipt["administrative_receipts"] = [
        {
            "start_ns": 950,
            "end_ns": 1_000,
            "reason": "unneeded bookkeeping",
        }
    ]

    with pytest.raises(TickNativePerformanceTimelineError, match="not needed"):
        derive_tick_native_performance_timeline_v1(receipt)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("simulation_receipts", 0, "end_ns"), None),
        (("simulation_receipts", 0, "start_ns"), -1),
        (("checkpoint_receipts", 0, "persist", "end_ns"), 340),
    ],
)
def test_missing_negative_or_non_monotonic_timestamp_fails_closed(
    path: tuple[object, ...], value: int | None
) -> None:
    receipt = deepcopy(_receipt())
    target: Any = receipt
    for key in path[:-1]:
        target = target[key]
    if value is None:
        del target[path[-1]]
    else:
        target[path[-1]] = value

    with pytest.raises(TickNativePerformanceTimelineError):
        derive_tick_native_performance_timeline_v1(receipt)


def test_exclusive_phase_overlap_fails_even_when_wall_reconciles() -> None:
    receipt = _receipt()
    receipt["checkpoint_receipts"][0]["detach"] = {
        "start_ns": 250,
        "end_ns": 350,
    }

    with pytest.raises(TickNativePerformanceTimelineError, match="overlap"):
        derive_tick_native_performance_timeline_v1(receipt)


def test_async_overlap_requires_explicit_async_flag() -> None:
    receipt = _receipt()
    receipt["checkpoint_receipts"][0]["persist"]["async"] = False

    with pytest.raises(TickNativePerformanceTimelineError, match="overlap"):
        derive_tick_native_performance_timeline_v1(receipt)


def test_reconciliation_error_over_two_percent_fails_closed() -> None:
    receipt = _receipt()
    receipt["wall_time"]["end_ns"] = 1_100

    with pytest.raises(TickNativePerformanceTimelineError, match="2%"):
        derive_tick_native_performance_timeline_v1(receipt)


def test_gpu_performance_window_must_end_where_post_simulation_tail_begins() -> None:
    receipt = _receipt()
    receipt["gpu_performance_window"]["end_ns"] = 899

    with pytest.raises(TickNativePerformanceTimelineError, match="tail boundary"):
        derive_tick_native_performance_timeline_v1(receipt)


def test_gpu_performance_window_terminal_tick_must_equal_simulation_completion() -> None:
    receipt = _receipt()
    receipt["gpu_performance_window"]["terminal_checkpoint_tick"] = 2_239

    with pytest.raises(
        TickNativePerformanceTimelineError,
        match="terminal checkpoint tick differs from simulation completion",
    ):
        derive_tick_native_performance_timeline_v1(receipt)
