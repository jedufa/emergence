"""Source-bound fail-closed tests for the unavailable composed R11 owner."""

from __future__ import annotations

from pathlib import Path

from engine.tensor_world.c4_resident_composition_r11 import (
    C4_RESIDENT_COMPOSITION_R11_DECISION_V1,
    C4_RESIDENT_COMPOSITION_R11_SCHEMA_VERSION_V1,
    audit_c4_resident_composition_r11_v1,
)


ROOT = Path(__file__).resolve().parents[1]


def test_r11_audit_binds_accepted_r10_and_refuses_residency_promotion() -> None:
    receipt = audit_c4_resident_composition_r11_v1(ROOT)

    assert receipt["schema_version"] == C4_RESIDENT_COMPOSITION_R11_SCHEMA_VERSION_V1
    assert receipt["decision"] == C4_RESIDENT_COMPOSITION_R11_DECISION_V1
    assert receipt["status"] == "FAIL_CLOSED_COMPOSITION_UNAVAILABLE"
    assert receipt["accepted_r10_route"] == "R10_B48_C32_FIVE_TICK_R9_DRIVER"
    assert receipt["accepted_r10_residency_classification"] == "C_host_dependent"
    assert receipt["r10_tick_python_loops"] == 1
    assert set(receipt["r10_host_boundary_hits"]) == {
        "detach().clone()",
        ".cpu()",
        ".tolist()",
        "torch.cuda.synchronize",
    }
    assert receipt["r10_zero_boundary_eligible"] is False
    assert receipt["gate_claim"] is False


def test_r11_exact_owner_rng_and_callsite_blockers_remain_source_bound() -> None:
    receipt = audit_c4_resident_composition_r11_v1(ROOT)
    blockers = {row["code"] for row in receipt["blockers"]}

    assert blockers == {
        "R10_HOST_STATUS_AND_ROLLBACK_AUTHORITY",
        "C4_OUTSIDE_C3_TICK_GRAPH",
        "C3_POSITION_AUTHORITY_IS_A_DTYPE_COPY",
        "C3_LIFECYCLE_AND_EVENT_OUTPUTS_NOT_BOUND_TO_V4",
        "C3_C5_RNG_CONTINUATION_UNMAPPED",
        "NO_COMPOSED_NATIVE_OR_GRAPH_CALLSITE",
    }
    assert receipt["composed_callsites"] == []
    assert receipt["sequential_host_fallback"] == "forbidden_not_implemented"
    assert receipt["runtime_boundary_counters"] == (
        "not_measured_for_nonexistent_composed_route"
    )
    assert set(receipt["sources"]) == {
        "r10_adapter",
        "r9_adapter",
        "r9_kernel",
        "c4_state_owner",
        "c3_persistent_owner",
        "c3_kernel",
        "c1_state_owner",
        "c5_replay_owner",
    }
    assert all(len(row["sha256"]) == 64 for row in receipt["sources"].values())
