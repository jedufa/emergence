"""Fail-stop runtime-fault policy contracts for the R11 owner."""

import pytest

from engine.tensor_world.c4_runtime_fault_r11 import (
    C4_RUNTIME_FAULT_R11_DECISION_V1,
    C4_RUNTIME_OWNER_ACTIVE_V1,
    C4_RUNTIME_OWNER_FATAL_INVALID_V1,
    TensorWorldC4RuntimeFaultStateR11V1,
    c4_runtime_fault_r11_source_receipt_v1,
)
from engine.tensor_world.contracts import TensorWorldContractError


def test_r11_runtime_fault_policy_is_fail_stop_not_fantasy_rollback() -> None:
    receipt = c4_runtime_fault_r11_source_receipt_v1()
    assert receipt["decision"] == C4_RUNTIME_FAULT_R11_DECISION_V1
    assert receipt["runtime_fault_policy"] == "irreversible_FATAL_INVALID_owner"
    assert receipt["in_memory_runtime_fault_rollback_claim"] is False
    assert receipt["next_tick_after_fault"] is False
    assert receipt["checkpoint_after_fault"] is False
    assert receipt["normal_tick_cpu_work"] is False
    assert receipt["control_boundary_only"] is True
    assert receipt["gate_claim"] is False


def test_r11_runtime_fault_state_retains_only_last_accepted_recovery_point() -> None:
    state = TensorWorldC4RuntimeFaultStateR11V1()
    state.validate()
    assert state.owner_state == C4_RUNTIME_OWNER_ACTIVE_V1
    digest = "a" * 64
    state.accept_checkpoint(digest)
    assert state.last_accepted_checkpoint_digest == digest

    state.mark_fatal(
        fault_class="CUDA_ILLEGAL_ACCESS",
        detail="illegal access detected while synchronizing the qualification boundary",
    )
    assert state.owner_state == C4_RUNTIME_OWNER_FATAL_INVALID_V1
    assert state.last_accepted_checkpoint_digest == digest
    with pytest.raises(TensorWorldContractError, match="FATAL/INVALID"):
        state.assert_usable_at_control_boundary()
    with pytest.raises(TensorWorldContractError, match="FATAL/INVALID"):
        state.accept_checkpoint("b" * 64)

    # Invalidity is irreversible for this owner; a second fault cannot
    # accidentally normalize it back to active or replace its first cause.
    first_fault = (state.fault_class, state.fault_detail)
    state.mark_fatal(fault_class="CUDA_CONTEXT_FAULT", detail="later fault")
    assert (state.fault_class, state.fault_detail) == first_fault


def test_r11_runtime_fault_state_rejects_invalid_checkpoint_identity() -> None:
    state = TensorWorldC4RuntimeFaultStateR11V1()
    with pytest.raises(TensorWorldContractError, match="SHA256"):
        state.accept_checkpoint("not-a-digest")
