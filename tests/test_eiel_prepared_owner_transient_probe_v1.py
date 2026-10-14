from __future__ import annotations

import pytest

from tools.scripts.eiel_prepared_owner_transient_probe_v1 import (
    summarize_trace_v1,
)


def _sample(
    stage: str,
    *,
    elapsed_ns: int,
    allocated: int,
    reserved: int,
    peak_allocated: int,
    peak_reserved: int,
    free: int,
) -> dict[str, int | str]:
    return {
        "schema_version": "eiel_resident_prepared_owner_cuda_memory_sample_v1",
        "stage": stage,
        "elapsed_ns": elapsed_ns,
        "allocated_bytes": allocated,
        "reserved_bytes": reserved,
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
        "free_bytes": free,
        "total_bytes": 1000,
    }


def test_summarize_trace_reports_peak_stages_and_deltas() -> None:
    summary = summarize_trace_v1(
        [
            _sample(
                "process_baseline",
                elapsed_ns=0,
                allocated=10,
                reserved=20,
                peak_allocated=10,
                peak_reserved=20,
                free=900,
            ),
            _sample(
                "probe:request_builder:1:after",
                elapsed_ns=1,
                allocated=50,
                reserved=200,
                peak_allocated=60,
                peak_reserved=220,
                free=700,
            ),
            _sample(
                "probe:batched_preparer:after",
                elapsed_ns=2,
                allocated=80,
                reserved=180,
                peak_allocated=120,
                peak_reserved=210,
                free=720,
            ),
        ]
    )

    assert summary["sample_count"] == 3
    assert summary["peak_reserved_stage"] == "probe:request_builder:1:after"
    assert summary["peak_reserved_delta_bytes"] == 200
    assert summary["peak_allocated_stage"] == "probe:batched_preparer:after"
    assert summary["peak_allocated_delta_bytes"] == 110
    assert summary["max_retained_reserved_stage"] == "probe:request_builder:1:after"
    assert summary["samples"][1]["reserved_delta_bytes"] == 180
    assert summary["samples"][1]["free_delta_bytes"] == -200


def test_summarize_trace_rejects_missing_required_fields() -> None:
    with pytest.raises(ValueError, match="missing fields"):
        summarize_trace_v1([{"stage": "process_baseline"}])


def test_summarize_trace_reports_first_peak_crossing_on_ties() -> None:
    summary = summarize_trace_v1(
        [
            _sample(
                "process_baseline",
                elapsed_ns=0,
                allocated=0,
                reserved=0,
                peak_allocated=0,
                peak_reserved=0,
                free=900,
            ),
            _sample(
                "probe:batched_preparer:after",
                elapsed_ns=1,
                allocated=100,
                reserved=200,
                peak_allocated=100,
                peak_reserved=200,
                free=700,
            ),
            _sample(
                "teardown",
                elapsed_ns=2,
                allocated=0,
                reserved=0,
                peak_allocated=100,
                peak_reserved=200,
                free=890,
            ),
        ]
    )

    assert summary["peak_reserved_stage"] == "probe:batched_preparer:after"
    assert summary["peak_allocated_stage"] == "probe:batched_preparer:after"
    assert summary["max_retained_reserved_stage"] == "probe:batched_preparer:after"
