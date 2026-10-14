from __future__ import annotations

from pathlib import Path

from tools.scripts.eiel_tensorworld_lifecycle_linear_native_parity_v1 import (
    _cases,
    _matched_source_commit,
)
import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_lifecycle_linear_microprototype_is_default_off_guarded_and_bounded() -> None:
    kernel = (
        ROOT
        / "engine/perf/"
        "cuda_architecture_b_flattened_wdl_controller_v2_kernels.cu"
    ).read_text(encoding="utf-8")
    candidate_builder = (
        ROOT
        / "tools/scripts/"
        "eiel_cuda_architecture_b_wdl_jungle_eed_"
        "lifecycle_linear_diagnostic_build.py"
    ).read_text(encoding="utf-8")
    baseline_builder = (
        ROOT
        / "tools/scripts/"
        "eiel_cuda_architecture_b_wdl_jungle_eed_"
        "lifecycle_baseline_diagnostic_build.py"
    ).read_text(encoding="utf-8")

    assert "#define EIEL_TW_LIFECYCLE_LINEAR_FAST_PATH_ENABLED 0" in kernel
    assert "valid_active_identity_pairs_v1(active_pairs)" in kernel
    assert "alive != assigned" in kernel
    assert "alive_assigned_count == active_count" in kernel
    assert "if (use_linear_fast_path)" in kernel
    assert "-DEIEL_TW_LIFECYCLE_LINEAR_FAST_PATH_ENABLED=1" in candidate_builder
    assert (
        "-DEIEL_TW_LIFECYCLE_LINEAR_FAST_PATH_TEST_SURFACE_ENABLED=1"
        in candidate_builder
    )
    assert (
        "-DEIEL_TW_LIFECYCLE_LINEAR_FAST_PATH_TEST_SURFACE_ENABLED=1"
        in baseline_builder
    )


def test_native_lifecycle_parity_matrix_covers_success_error_and_fallback() -> None:
    rows = _cases()
    names = {row["name"] for row in rows}

    assert len(rows) == 17
    assert {
        "empty",
        "no_death",
        "one_energy_death",
        "multiple_deaths_slot_order_differs",
        "health_reason_priority",
        "all_death",
        "pending_success_boundary",
        "pending_overflow_atomic",
        "invalid_pending_atomic",
        "out_of_order_active_fallback",
        "duplicate_active_id_fallback",
        "wrong_row_identity_fallback",
        "out_of_range_row_slot_fallback",
        "omitted_live_row_fallback",
        "alive_unassigned_fallback",
        "dead_assigned_fallback",
        "active_capacity_less_than_slots",
    } == names
    assert {
        row["name"]
        for row in rows
        if row.get("expect_no_mutation") is True
    } == {"pending_overflow_atomic", "invalid_pending_atomic"}


def test_native_lifecycle_parity_requires_exact_shared_source_commit() -> None:
    commit = "a" * 40
    assert _matched_source_commit(
        {"committed_source_snapshot": {"build_source_commit": commit}},
        {"committed_source_snapshot": {"build_source_commit": commit}},
    ) == commit

    with pytest.raises(RuntimeError, match="differ in source commit"):
        _matched_source_commit({}, {})
