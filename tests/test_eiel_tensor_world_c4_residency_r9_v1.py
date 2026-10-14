"""Static source tests for the bounded C4 R9 residency classifier."""

from __future__ import annotations

from pathlib import Path

from engine.tensor_world.c4_residency_r9 import (
    C4_RESIDENCY_R9_DECISION_V1,
    C4_RESIDENCY_R9_SCHEMA_VERSION_V1,
    audit_c4_residency_r9_static_v1,
)


ROOT = Path(__file__).resolve().parents[1]


def test_c4_r9_static_receipt_is_source_bound_and_has_no_gate_claim() -> None:
    receipt = audit_c4_residency_r9_static_v1(ROOT)

    assert receipt["schema_version"] == C4_RESIDENCY_R9_SCHEMA_VERSION_V1
    assert receipt["decision"] == C4_RESIDENCY_R9_DECISION_V1
    assert receipt["status"] == "STATIC_RESIDENCY_BLOCKERS_OPEN"
    assert receipt["scope"] == "C4_R1_R9_ADAPTER_AND_NATIVE_SOURCE_ONLY"
    assert receipt["gate_claim"] is False
    assert len(receipt["rungs"]) == 9
    assert {row["rung"] for row in receipt["rungs"]} == {
        "R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8", "R9"
    }
    assert receipt["source_hashes"]
    assert all(len(digest) == 64 for digest in receipt["source_hashes"].values())


def test_c4_r9_execute_and_native_scans_do_not_promote_setup_or_test_boundaries() -> None:
    receipt = audit_c4_residency_r9_static_v1(ROOT)

    for row in receipt["rungs"]:
        assert row["source_transfer_hits"] == []
        assert row["source_sync_or_materialization_hits"] == []
        assert row["source_hot_allocation_hits"] == []
        assert row["source_cuda_graph_hits"] == []
        assert row["execute_per_world_python_loop_count"] == 0
        assert row["source_classification"] == (
            "NO_FORBIDDEN_BOUNDARY_TOKEN_OBSERVED"
        )

    # Allocation in reusable staging factories is setup, not a steady-state
    # execute-body hit; the audit keeps it as an unproven lifetime obligation.
    r6_source = (ROOT / "engine/tensor_world/c4_resident_preflight_r6.py").read_text()
    assert "status=torch.zeros" in r6_source
    rows = {row["rung"]: row for row in receipt["rungs"]}
    assert rows["R4"]["execute_python_loop_count"] == 1
    assert rows["R5"]["execute_python_loop_count"] == 1
    assert rows["R6"]["execute_python_loop_count"] == 1
    assert any(
        row["code"] == "PREALLOCATED_STAGING_LIFETIME_UNBOUND"
        for row in receipt["blockers"]
    )


def test_c4_r9_reports_exact_current_residency_blockers() -> None:
    receipt = audit_c4_residency_r9_static_v1(ROOT)
    blockers = {row["code"] for row in receipt["blockers"]}

    assert blockers == {
        "C1_C2_C3_C4_COMPOSED_TICK_NOT_WIRED",
        "CUDA_GRAPH_CLASSIFICATION_UNPROVEN",
        "R7_HOST_TICK_OWNERSHIP",
        "STEADY_STATE_COUNTERS_UNMEASURED",
        "PREALLOCATED_STAGING_LIFETIME_UNBOUND",
    }
    assert receipt["production_composed_callsites"] == []
    assert "component evidence only" in receipt["interpretation"]

    rows = {row["rung"]: row for row in receipt["rungs"]}
    assert rows["R2"]["admission"].startswith("B1_")
    assert rows["R3"]["admission"].startswith("B1_")
    assert rows["R4"]["admission"].startswith("B1_")
    assert rows["R5"]["admission"].startswith("B4_")
    assert rows["R6"]["admission"].startswith("B4_")
    assert rows["R7"]["admission"] == (
        "B4_C32_host_launched_five_tick_test_harness"
    )
    assert rows["R8"]["admission"].startswith("B4_C32_native")
    assert rows["R9"]["admission"] == "B48_native_one_tick_dynamic_batch_probe"
