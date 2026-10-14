"""Control-boundary integration for R11 fatal faults and C5 checkpoints.

The resident graph executes normal ticks without consulting this module.  A
host owner calls these operations only between complete captured graph windows:
accept a durable checkpoint, record a detected CUDA fault, or restore the last
accepted checkpoint into a distinct fresh owner.  CUDA graph executables are
ephemeral and a restored owner remains replay-ineligible until the caller has
captured a fresh executable and records its fingerprint here.

This adapter deliberately makes no in-memory rollback claim for a CUDA fault.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from pathlib import Path
from typing import Any, Final, Mapping

from .c4_runtime_fault_r11 import (
    C4_RUNTIME_OWNER_ACTIVE_V1,
    C4_RUNTIME_OWNER_FATAL_INVALID_V1,
    TensorWorldC4RuntimeFaultStateR11V1,
)
from .c4_state import TensorWorldStateBundleV4
from .c5_determinism import TensorWorldC5ReplayContextV1
from .c5_graph_checkpoint_r11 import (
    TensorWorldC5GraphCheckpointR11V1,
    TensorWorldC5GraphRunControlR11V1,
    capture_c5_graph_checkpoint_r11_v1,
    restore_c5_graph_checkpoint_r11_v1,
)
from .contracts import TensorWorldContractError


ROOT = Path(__file__).resolve().parents[2]
RUNTIME_FAULT_SOURCE = ROOT / "engine/tensor_world/c4_runtime_fault_r11.py"
GRAPH_CHECKPOINT_SOURCE = ROOT / "engine/tensor_world/c5_graph_checkpoint_r11.py"
GRAPH_OWNER_SOURCE = ROOT / "engine/tensor_world/c4_resident_composed_r11.py"

C5_GRAPH_FAULT_BOUNDARY_R11_SCHEMA_VERSION_V1: Final[str] = (
    "tensor_world_c5_graph_fault_boundary_r11_v1"
)
C5_GRAPH_FAULT_BOUNDARY_R11_DECISION_V1: Final[str] = "NO_GATE_DECISION"
C5_GRAPH_FAULT_BOUNDARY_R11_CONTROL_BOUNDARY_V1: Final[str] = (
    "between_complete_captured_graph_windows"
)
C5_GRAPH_FAULT_BOUNDARY_R11_WINDOW_TICKS_V1: Final[int] = 5


def _validate_sha256(value: str, *, name: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise TensorWorldContractError(f"R11 {name} is not lowercase SHA256")


def _validate_owner_id(owner_id: str) -> None:
    if (
        not isinstance(owner_id, str)
        or not owner_id
        or owner_id != owner_id.strip()
        or len(owner_id) > 128
    ):
        raise TensorWorldContractError("R11 graph owner identity is invalid")


@dataclass(slots=True)
class TensorWorldC5GraphFaultBoundaryOwnerR11V1:
    """Host-only lifecycle state sampled at graph control boundaries."""

    owner_id: str
    runtime_fault: TensorWorldC4RuntimeFaultStateR11V1 = field(
        default_factory=TensorWorldC4RuntimeFaultStateR11V1
    )
    pending_checkpoint_digest: str | None = None
    recovery_checkpoint_digest: str | None = None
    graph_capture_fingerprint: str | None = None
    schema_version: str = C5_GRAPH_FAULT_BOUNDARY_R11_SCHEMA_VERSION_V1

    def validate(self) -> None:
        if self.schema_version != C5_GRAPH_FAULT_BOUNDARY_R11_SCHEMA_VERSION_V1:
            raise TensorWorldContractError(
                "R11 graph fault-boundary schema is unsupported"
            )
        _validate_owner_id(self.owner_id)
        self.runtime_fault.validate()
        if self.pending_checkpoint_digest is not None:
            _validate_sha256(
                self.pending_checkpoint_digest,
                name="pending checkpoint digest",
            )
        if self.recovery_checkpoint_digest is not None:
            _validate_sha256(
                self.recovery_checkpoint_digest,
                name="recovery checkpoint digest",
            )
            if (
                self.runtime_fault.last_accepted_checkpoint_digest
                != self.recovery_checkpoint_digest
            ):
                raise TensorWorldContractError(
                    "R11 fresh-owner recovery digest differs from accepted checkpoint"
                )
        if self.graph_capture_fingerprint is not None:
            _validate_sha256(
                self.graph_capture_fingerprint,
                name="graph capture fingerprint",
            )
        if (
            self.runtime_fault.owner_state == C4_RUNTIME_OWNER_FATAL_INVALID_V1
            and (
                self.graph_capture_fingerprint is not None
                or self.pending_checkpoint_digest is not None
            )
        ):
            raise TensorWorldContractError(
                "R11 fatal owner cannot retain replayable or checkpoint candidates"
            )

    def assert_graph_window_replay_allowed_at_control_boundary(self) -> None:
        """Fail closed before dispatching the next complete graph window."""

        self.validate()
        self.runtime_fault.assert_usable_at_control_boundary()
        if self.graph_capture_fingerprint is None:
            raise TensorWorldContractError(
                "R11 owner requires fresh graph capture before window replay"
            )

    def record_graph_capture_at_control_boundary(
        self,
        *,
        graph_capture_fingerprint: str,
        recovery_checkpoint_digest: str | None = None,
    ) -> None:
        """Record caller-owned CUDA graph capture; no graph is built here."""

        self.runtime_fault.assert_usable_at_control_boundary()
        _validate_sha256(
            graph_capture_fingerprint,
            name="graph capture fingerprint",
        )
        if self.recovery_checkpoint_digest is None:
            if recovery_checkpoint_digest is not None:
                raise TensorWorldContractError(
                    "R11 initial graph capture cannot claim checkpoint recovery"
                )
        elif recovery_checkpoint_digest != self.recovery_checkpoint_digest:
            raise TensorWorldContractError(
                "R11 graph recapture is not bound to the restored checkpoint"
            )
        self.graph_capture_fingerprint = graph_capture_fingerprint
        self.recovery_checkpoint_digest = None
        self.validate()

    def mark_runtime_fault_at_control_boundary(
        self,
        *,
        fault_class: str,
        detail: str,
    ) -> None:
        """Irreversibly invalidate this owner after boundary fault detection."""

        self.runtime_fault.mark_fatal(fault_class=fault_class, detail=detail)
        self.pending_checkpoint_digest = None
        self.graph_capture_fingerprint = None
        self.validate()


def make_c5_graph_fault_boundary_owner_r11_v1(
    *, owner_id: str
) -> TensorWorldC5GraphFaultBoundaryOwnerR11V1:
    owner = TensorWorldC5GraphFaultBoundaryOwnerR11V1(owner_id=owner_id)
    owner.validate()
    return owner


def capture_c5_graph_checkpoint_at_control_boundary_r11_v1(
    owner: TensorWorldC5GraphFaultBoundaryOwnerR11V1,
    bundle: TensorWorldStateBundleV4,
    replay_context: TensorWorldC5ReplayContextV1,
    run_control: TensorWorldC5GraphRunControlR11V1,
    *,
    backend_id: str,
    source_commit: str,
    provider_contract_id: str,
    metadata: Mapping[str, Any] | None = None,
) -> TensorWorldC5GraphCheckpointR11V1:
    """Capture a candidate recovery point between complete graph windows."""

    owner.assert_graph_window_replay_allowed_at_control_boundary()
    checkpoint = capture_c5_graph_checkpoint_r11_v1(
        bundle,
        replay_context,
        run_control,
        backend_id=backend_id,
        source_commit=source_commit,
        provider_contract_id=provider_contract_id,
        metadata=metadata,
    )
    owner.pending_checkpoint_digest = checkpoint.digest
    owner.validate()
    return checkpoint


def accept_c5_graph_checkpoint_at_control_boundary_r11_v1(
    owner: TensorWorldC5GraphFaultBoundaryOwnerR11V1,
    checkpoint: TensorWorldC5GraphCheckpointR11V1,
) -> None:
    """Record acceptance only after the caller durably publishes checkpoint."""

    owner.assert_graph_window_replay_allowed_at_control_boundary()
    checkpoint.validate()
    if owner.pending_checkpoint_digest != checkpoint.digest:
        raise TensorWorldContractError(
            "R11 checkpoint acceptance does not match this owner's candidate"
        )
    owner.runtime_fault.accept_checkpoint(checkpoint.digest)
    owner.pending_checkpoint_digest = None
    owner.validate()


def restore_c5_graph_checkpoint_into_fresh_owner_at_control_boundary_r11_v1(
    failed_owner: TensorWorldC5GraphFaultBoundaryOwnerR11V1,
    fresh_owner: TensorWorldC5GraphFaultBoundaryOwnerR11V1,
    checkpoint: TensorWorldC5GraphCheckpointR11V1,
    destination: TensorWorldStateBundleV4,
    replay_context: TensorWorldC5ReplayContextV1,
    run_control: TensorWorldC5GraphRunControlR11V1,
    *,
    expected_backend_id: str,
    expected_source_commit: str,
    expected_provider_contract_id: str,
) -> str:
    """Restore the exact accepted recovery point into a distinct fresh owner."""

    failed_owner.validate()
    fresh_owner.validate()
    if failed_owner.runtime_fault.owner_state != C4_RUNTIME_OWNER_FATAL_INVALID_V1:
        raise TensorWorldContractError("R11 recovery source owner is not FATAL/INVALID")
    accepted_digest = failed_owner.runtime_fault.last_accepted_checkpoint_digest
    if accepted_digest is None or checkpoint.digest != accepted_digest:
        raise TensorWorldContractError(
            "R11 recovery checkpoint is not the failed owner's last accepted point"
        )
    if failed_owner.owner_id == fresh_owner.owner_id:
        raise TensorWorldContractError("R11 runtime-fault recovery requires a fresh owner")
    if (
        fresh_owner.runtime_fault.owner_state != C4_RUNTIME_OWNER_ACTIVE_V1
        or fresh_owner.runtime_fault.last_accepted_checkpoint_digest is not None
        or fresh_owner.pending_checkpoint_digest is not None
        or fresh_owner.recovery_checkpoint_digest is not None
        or fresh_owner.graph_capture_fingerprint is not None
    ):
        raise TensorWorldContractError("R11 recovery destination owner is not pristine")

    restored_digest = restore_c5_graph_checkpoint_r11_v1(
        checkpoint,
        destination,
        replay_context,
        run_control,
        expected_backend_id=expected_backend_id,
        expected_source_commit=expected_source_commit,
        expected_provider_contract_id=expected_provider_contract_id,
    )
    if restored_digest != accepted_digest:
        raise TensorWorldContractError("R11 fresh-owner restore digest differs")
    fresh_owner.runtime_fault.accept_checkpoint(restored_digest)
    fresh_owner.recovery_checkpoint_digest = restored_digest
    fresh_owner.validate()
    return restored_digest


def c5_graph_fault_boundary_r11_source_receipt_v1() -> dict[str, object]:
    sources = {
        "adapter": Path(__file__),
        "runtime_fault_policy": RUNTIME_FAULT_SOURCE,
        "graph_checkpoint": GRAPH_CHECKPOINT_SOURCE,
        "graph_owner": GRAPH_OWNER_SOURCE,
    }
    return {
        "schema_version": C5_GRAPH_FAULT_BOUNDARY_R11_SCHEMA_VERSION_V1,
        "decision": C5_GRAPH_FAULT_BOUNDARY_R11_DECISION_V1,
        "control_boundary": C5_GRAPH_FAULT_BOUNDARY_R11_CONTROL_BOUNDARY_V1,
        "graph_window_ticks": C5_GRAPH_FAULT_BOUNDARY_R11_WINDOW_TICKS_V1,
        "normal_tick_cpu_semantic_work": False,
        "between_tick_cpu_semantic_work": False,
        "checkpoint_cpu_work": "declared_control_boundaries_only",
        "checkpoint_acceptance": "caller_explicit_after_durable_publication",
        "fault_policy": "irreversible_FATAL_INVALID_old_owner",
        "recovery": (
            "exact_last_accepted_checkpoint_into_distinct_pristine_owner_then_"
            "fresh_graph_recapture"
        ),
        "graph_executable_serialized": False,
        "graph_recapture_execution": "caller_owned_not_performed_by_adapter",
        "in_memory_runtime_fault_rollback_claim": False,
        "gate_claim": False,
        "qualification_impact": {
            "existing_runtime_and_checkpoint_routes": "PRESERVED_UNCHANGED",
            "new_integration_boundary": "DELTA_QUALIFICATION_REQUIRED",
            "broad_requalification_required": False,
        },
        "sources": {
            name: {
                "path": str(path.relative_to(ROOT)),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for name, path in sources.items()
        },
    }


__all__ = [
    "C5_GRAPH_FAULT_BOUNDARY_R11_CONTROL_BOUNDARY_V1",
    "C5_GRAPH_FAULT_BOUNDARY_R11_DECISION_V1",
    "C5_GRAPH_FAULT_BOUNDARY_R11_SCHEMA_VERSION_V1",
    "C5_GRAPH_FAULT_BOUNDARY_R11_WINDOW_TICKS_V1",
    "TensorWorldC5GraphFaultBoundaryOwnerR11V1",
    "accept_c5_graph_checkpoint_at_control_boundary_r11_v1",
    "c5_graph_fault_boundary_r11_source_receipt_v1",
    "capture_c5_graph_checkpoint_at_control_boundary_r11_v1",
    "make_c5_graph_fault_boundary_owner_r11_v1",
    "restore_c5_graph_checkpoint_into_fresh_owner_at_control_boundary_r11_v1",
]
