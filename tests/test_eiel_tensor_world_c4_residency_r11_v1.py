"""Static residency evidence for the current composed R11 owner."""

from pathlib import Path

from engine.tensor_world.c4_residency_r11 import (
    C4_RESIDENCY_R11_DECISION_V1,
    C4_RESIDENCY_R11_SCHEMA_VERSION_V1,
    audit_c4_residency_r11_static_v1,
)


ROOT = Path(__file__).resolve().parents[1]


def test_r11_static_audit_classifies_the_composed_owner_without_gate_claim() -> None:
    receipt = audit_c4_residency_r11_static_v1(ROOT)
    assert receipt["schema_version"] == C4_RESIDENCY_R11_SCHEMA_VERSION_V1
    assert receipt["decision"] == C4_RESIDENCY_R11_DECISION_V1
    assert receipt["status"] == "R11_COMPONENT_RESIDENCY_READY_GATE_EVIDENCE_OPEN"
    assert receipt["execution_classification"] == (
        "B_FIXED_TOPOLOGY_CUDA_GRAPH_OWNER_WITH_DEVICE_TIME"
    )
    assert receipt["execute_python_loop_count"] == 0
    assert receipt["execute_per_world_python_loop_count"] == 0
    assert receipt["source_transfer_hits"] == []
    assert receipt["source_sync_or_materialization_hits"] == []
    assert receipt["source_hot_allocator_hits"] == []
    assert receipt["persistent_staging"] is True
    assert receipt["runtime_receipt_required"] is True
    assert receipt["gate_claim"] is False
    assert receipt["source_hashes"]


def test_r11_static_audit_keeps_final_gate_obligations_explicit() -> None:
    receipt = audit_c4_residency_r11_static_v1(ROOT)
    blockers = {row["code"] for row in receipt["blockers"]}
    assert blockers == {
        "R11_RUNTIME_COUNTERS_NOT_GATE_COMPLETE",
        "R11_FINAL_GEOMETRY_RESOURCE_OPEN",
        "R11_GRAPH_WINDOW_OPEN",
        "R11_LAUNCH_FAULT_ATOMICITY_UNCLAIMED",
    }
