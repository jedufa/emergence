"""Fail-stop runtime-fault policy for the resident R11 graph owner.

Semantic/capacity failures are handled by resident preflight and guarded
commit.  A CUDA launch or device fault is different: once detected at an
explicit synchronization/checkpoint/report boundary, the in-memory owner is
irreversibly invalid and recovery starts from the last accepted durable C5
checkpoint in a fresh owner (and, when required, a fresh CUDA context).

This module is control-plane state only.  It is never consulted from inside a
captured graph window and never promises in-memory rollback after a device
fault.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from .contracts import TensorWorldContractError


C4_RUNTIME_FAULT_R11_SCHEMA_VERSION_V1: Final[str] = (
    "tensor_world_c4_runtime_fault_r11_v1"
)
C4_RUNTIME_FAULT_R11_DECISION_V1: Final[str] = "NO_GATE_DECISION"
C4_RUNTIME_OWNER_ACTIVE_V1: Final[str] = "ACTIVE"
C4_RUNTIME_OWNER_FATAL_INVALID_V1: Final[str] = "FATAL_INVALID"
_FAULT_CLASSES: Final[frozenset[str]] = frozenset(
    {
        "CUDA_GRAPH_LAUNCH_FAULT",
        "CUDA_ILLEGAL_ACCESS",
        "CUDA_DEVICE_FAULT",
        "CUDA_CONTEXT_FAULT",
        "UNKNOWN_CUDA_RUNTIME_FAULT",
    }
)


@dataclass(slots=True)
class TensorWorldC4RuntimeFaultStateR11V1:
    """Host-visible state sampled only at explicit control boundaries."""

    owner_state: str = C4_RUNTIME_OWNER_ACTIVE_V1
    last_accepted_checkpoint_digest: str | None = None
    fault_class: str | None = None
    fault_detail: str | None = None
    schema_version: str = C4_RUNTIME_FAULT_R11_SCHEMA_VERSION_V1

    def validate(self) -> None:
        if self.schema_version != C4_RUNTIME_FAULT_R11_SCHEMA_VERSION_V1:
            raise TensorWorldContractError("R11 runtime-fault schema is unsupported")
        if self.owner_state not in {
            C4_RUNTIME_OWNER_ACTIVE_V1,
            C4_RUNTIME_OWNER_FATAL_INVALID_V1,
        }:
            raise TensorWorldContractError("R11 runtime owner state is invalid")
        digest = self.last_accepted_checkpoint_digest
        if digest is not None and (
            len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise TensorWorldContractError(
                "R11 accepted checkpoint digest is not SHA256"
            )
        if self.owner_state == C4_RUNTIME_OWNER_ACTIVE_V1:
            if self.fault_class is not None or self.fault_detail is not None:
                raise TensorWorldContractError(
                    "R11 active owner cannot retain a fatal fault"
                )
        elif self.fault_class not in _FAULT_CLASSES or not self.fault_detail:
            raise TensorWorldContractError(
                "R11 fatal owner requires a classified runtime fault"
            )

    def assert_usable_at_control_boundary(self) -> None:
        """Reject further windows/checkpoints once a runtime fault is known."""

        self.validate()
        if self.owner_state != C4_RUNTIME_OWNER_ACTIVE_V1:
            raise TensorWorldContractError(
                "R11 owner is FATAL/INVALID; restore the last accepted C5 "
                "checkpoint into a fresh owner/context"
            )

    def accept_checkpoint(self, digest: str) -> None:
        """Record the durable recovery point after checkpoint acceptance."""

        self.assert_usable_at_control_boundary()
        self.last_accepted_checkpoint_digest = digest
        self.validate()

    def mark_fatal(self, *, fault_class: str, detail: str) -> None:
        """Irreversibly invalidate this owner after a detected CUDA fault."""

        if self.owner_state == C4_RUNTIME_OWNER_FATAL_INVALID_V1:
            return
        if fault_class not in _FAULT_CLASSES:
            fault_class = "UNKNOWN_CUDA_RUNTIME_FAULT"
        normalized = " ".join(str(detail).split()).strip()
        if not normalized:
            normalized = "CUDA runtime fault detected at a control boundary"
        self.owner_state = C4_RUNTIME_OWNER_FATAL_INVALID_V1
        self.fault_class = fault_class
        self.fault_detail = normalized[:1_024]
        self.validate()


def c4_runtime_fault_r11_source_receipt_v1() -> dict[str, object]:
    return {
        "schema_version": C4_RUNTIME_FAULT_R11_SCHEMA_VERSION_V1,
        "decision": C4_RUNTIME_FAULT_R11_DECISION_V1,
        "semantic_failure_policy": (
            "device_preflight_then_batch_global_guarded_no_authoritative_C4_commit"
        ),
        "runtime_fault_policy": "irreversible_FATAL_INVALID_owner",
        "in_memory_runtime_fault_rollback_claim": False,
        "next_tick_after_fault": False,
        "checkpoint_after_fault": False,
        "recovery": (
            "fresh_owner_and_CUDA_context_if_required_from_last_accepted_"
            "durable_C5_checkpoint"
        ),
        "normal_tick_cpu_work": False,
        "control_boundary_only": True,
        "gate_claim": False,
    }


__all__ = [
    "C4_RUNTIME_FAULT_R11_DECISION_V1",
    "C4_RUNTIME_FAULT_R11_SCHEMA_VERSION_V1",
    "C4_RUNTIME_OWNER_ACTIVE_V1",
    "C4_RUNTIME_OWNER_FATAL_INVALID_V1",
    "TensorWorldC4RuntimeFaultStateR11V1",
    "c4_runtime_fault_r11_source_receipt_v1",
]
