from __future__ import annotations

from pathlib import Path

from tools.scripts.eiel_hapb_gate8_measurement_from_h4_v1 import (
    compare_h4_canaries_v1,
    measurement_row_from_h4_receipt_v1,
)


def _h4_receipt(*, cells: int = 60, exact: bool = True, wall_ns: int = 1_200_000_000_000) -> dict:
    return {
        "physical_cells": cells,
        "generations": 6,
        "continuation_exact": exact,
        "fresh_owner_restore": True,
        "checkpoint_restore_boundary_mismatch_count": 0 if exact else 2,
        "acs_checkpoint_slots": 0,
        "receipt_sha256": "abc",
        "artifact_sha256": "art",
        "midpoint_checkpoint_sha256": "mid",
        "secondary_boundary_timing": {
            "samples": [
                {"label": "g0_to_g3", "native_execute_wall_ns": wall_ns // 2},
                {"label": "g0_to_g6", "native_execute_wall_ns": wall_ns},
            ]
        },
        "memory": {
            "device_total_bytes": 6144 * 1024 * 1024,
            "minimum_free_at_peak_bytes": 400 * 1024 * 1024,
            "peak_used_bytes": 5700 * 1024 * 1024,
            "rule_pass": False,
        },
    }


def test_h4_measurement_row_is_engineering_only_and_not_a_rehearsal() -> None:
    row = measurement_row_from_h4_receipt_v1(
        _h4_receipt(),
        candidate_id="hapb_b60_canary",
        route="hapb",
        receipt_path=Path("/tmp/fake_h4.json"),
    )
    assert row["batch_cells"] == 60
    assert row["semantically_exact"] is True
    assert row["sustained_rehearsal_pass"] is False
    assert row["production_execution_admission"] is False
    assert row["qualification_claim"] is False
    assert row["amortized_production_seconds_per_generation"] == 200.0
    assert row["production_equivalent_cell_generations_per_hour"] == 60 * 3600.0 / 200.0
    assert row["memory"]["rule_pass"] is False


def test_h4_canary_compare_detects_midpoint_drift() -> None:
    run1 = _h4_receipt()
    run2 = _h4_receipt()
    assert compare_h4_canaries_v1(run1, run2)["agree"] is True
    run2["midpoint_checkpoint_sha256"] = "other"
    compared = compare_h4_canaries_v1(run1, run2)
    assert compared["agree"] is False
    assert "midpoint_checkpoint_sha256" in compared["mismatched_fields"]
