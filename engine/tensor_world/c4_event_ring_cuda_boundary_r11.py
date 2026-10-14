"""Post-window acknowledgement boundary for the resident R11 CUDA owner.

Normal five-tick graph replay never imports or calls this module.  After a
complete window, a checkpoint or report boundary may materialize the resident
event ring once.  The resulting candidate is immutable and acknowledgement is
separate: only an accepted artifact may advance the device consumer cursor.

Bank rows/counts are not cleared by the host.  The next resident native window
uses the acknowledged cursor when it needs to reuse a physical bank, preserving
the existing fixed graph topology and absolute producer sequence.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Final

import torch

from .c4_event_ring_r11 import TensorWorldC4EventRingR11V1, _event_ring_tensors
from .c4_resident_composed_r11 import TensorWorldC4ResidentComposedR11OwnerV1
from .c5_determinism import _canonical_update, _clone_to_cpu
from .c5_graph_checkpoint_r11 import (
    TensorWorldC5GraphCheckpointR11V1,
    TensorWorldC5GraphRunControlR11V1,
    bind_c5_graph_run_control_r11_v1,
)
from .contracts import TensorWorldContractError


ROOT = Path(__file__).resolve().parents[2]
EVENT_RING_SOURCE = ROOT / "engine/tensor_world/c4_event_ring_r11.py"
GRAPH_OWNER_SOURCE = ROOT / "engine/tensor_world/c4_resident_composed_r11.py"
GRAPH_CHECKPOINT_SOURCE = ROOT / "engine/tensor_world/c5_graph_checkpoint_r11.py"
LIFETIME_SOURCE = ROOT / "engine/tensor_world/c4_event_ring_lifetime_r11.py"

C4_EVENT_RING_CUDA_BOUNDARY_R11_SCHEMA_VERSION_V1: Final[str] = (
    "tensor_world_c4_event_ring_cuda_boundary_r11_v1"
)
C4_EVENT_RING_CUDA_BOUNDARY_R11_DECISION_V1: Final[str] = "NO_GATE_DECISION"
C4_EVENT_RING_CUDA_BOUNDARY_R11_WINDOW_TICKS_V1: Final[int] = 5
C4_EVENT_RING_CUDA_BOUNDARY_R11_CHECKPOINT_V1: Final[str] = "checkpoint"
C4_EVENT_RING_CUDA_BOUNDARY_R11_REPORT_V1: Final[str] = "report"
_ACCEPTANCE_KINDS: Final[frozenset[str]] = frozenset(
    {
        C4_EVENT_RING_CUDA_BOUNDARY_R11_CHECKPOINT_V1,
        C4_EVENT_RING_CUDA_BOUNDARY_R11_REPORT_V1,
    }
)


def _validate_sha256(value: str, *, name: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise TensorWorldContractError(f"R11 {name} is not lowercase SHA256")


def _validate_completed_window_v1(
    control: TensorWorldC5GraphRunControlR11V1,
) -> None:
    control.validate(semantic=True)
    if control.device.type != "cpu":
        raise TensorWorldContractError(
            "R11 acknowledgement candidate run control must be materialized"
        )
    if not (
        bool((control.requested_ticks == 5).all())
        and bool((control.completed_ticks == 5).all())
        and bool((control.receipt_write_index == 5).all())
        and bool((control.global_status == 0).all())
        and bool((control.stop_requested == 0).all())
        and bool((control.current_tick + 1 == control.tick).all())
    ):
        raise TensorWorldContractError(
            "R11 acknowledgement requires one successful complete five-tick window"
        )


def _owner_binding_v1(owner: TensorWorldC4ResidentComposedR11OwnerV1) -> str:
    owner.assert_steady_state_ready()
    ring = owner.bundle.event_ring
    staging = owner.shadow_staging
    if ring is None or staging is None:
        raise TensorWorldContractError("R11 CUDA owner has no resident event boundary")
    hasher = hashlib.sha256()
    hasher.update(C4_EVENT_RING_CUDA_BOUNDARY_R11_SCHEMA_VERSION_V1.encode("ascii"))
    for value in (
        owner.bundle.batch,
        owner.bundle.capacity,
        owner.bundle.grid_size,
        ring.capacity,
        ring.bank_counts.data_ptr(),
        ring.bank_start_sequences.data_ptr(),
        ring.producer_cursor.data_ptr(),
        ring.consumer_cursor.data_ptr(),
        ring.active_bank.data_ptr(),
        staging.generation.data_ptr(),
        staging.tick.data_ptr(),
        staging.completed_ticks.data_ptr(),
        owner.replay.event_sequence.data_ptr(),
    ):
        hasher.update(int(value).to_bytes(16, "little", signed=False))
    return hasher.hexdigest()


def _candidate_digest_v1(
    *,
    ring: TensorWorldC4EventRingR11V1,
    run_control: TensorWorldC5GraphRunControlR11V1,
    owner_binding: str,
    acceptance_kind: str,
    artifact_digest: str | None,
) -> str:
    hasher = hashlib.sha256()
    hasher.update(C4_EVENT_RING_CUDA_BOUNDARY_R11_SCHEMA_VERSION_V1.encode("ascii"))
    _canonical_update(hasher, ring, "event_ring")
    _canonical_update(hasher, run_control, "run_control")
    _canonical_update(hasher, owner_binding, "owner_binding")
    _canonical_update(hasher, acceptance_kind, "acceptance_kind")
    _canonical_update(hasher, artifact_digest, "artifact_digest")
    return hasher.hexdigest()


@dataclass(frozen=True, slots=True)
class TensorWorldC4EventRingAcknowledgementCandidateR11V1:
    """Immutable CPU snapshot awaiting checkpoint/report acceptance."""

    event_ring: TensorWorldC4EventRingR11V1
    run_control: TensorWorldC5GraphRunControlR11V1
    owner_binding: str
    acceptance_kind: str
    artifact_digest: str | None
    digest: str
    schema_version: str = C4_EVENT_RING_CUDA_BOUNDARY_R11_SCHEMA_VERSION_V1

    @property
    def acceptance_digest(self) -> str:
        if self.acceptance_kind == C4_EVENT_RING_CUDA_BOUNDARY_R11_CHECKPOINT_V1:
            if self.artifact_digest is None:
                raise TensorWorldContractError(
                    "R11 checkpoint acknowledgement has no artifact digest"
                )
            return self.artifact_digest
        return self.digest

    def validate(self) -> None:
        if self.schema_version != C4_EVENT_RING_CUDA_BOUNDARY_R11_SCHEMA_VERSION_V1:
            raise TensorWorldContractError(
                "R11 event acknowledgement schema is unsupported"
            )
        if self.acceptance_kind not in _ACCEPTANCE_KINDS:
            raise TensorWorldContractError(
                "R11 event acknowledgement kind is unsupported"
            )
        _validate_sha256(self.owner_binding, name="event owner binding")
        if self.acceptance_kind == C4_EVENT_RING_CUDA_BOUNDARY_R11_CHECKPOINT_V1:
            if self.artifact_digest is None:
                raise TensorWorldContractError(
                    "R11 checkpoint acknowledgement requires an artifact digest"
                )
            _validate_sha256(self.artifact_digest, name="checkpoint artifact digest")
        elif self.artifact_digest is not None:
            raise TensorWorldContractError(
                "R11 report acknowledgement must bind its snapshot digest"
            )
        self.event_ring.validate(semantic=True)
        if self.event_ring.device.type != "cpu":
            raise TensorWorldContractError(
                "R11 event acknowledgement ring must be materialized"
            )
        _validate_completed_window_v1(self.run_control)
        if self.run_control.batch != self.event_ring.batch:
            raise TensorWorldContractError(
                "R11 event acknowledgement batch differs from run control"
            )
        expected = _candidate_digest_v1(
            ring=self.event_ring,
            run_control=self.run_control,
            owner_binding=self.owner_binding,
            acceptance_kind=self.acceptance_kind,
            artifact_digest=self.artifact_digest,
        )
        if self.digest != expected:
            raise TensorWorldContractError(
                "R11 event acknowledgement candidate digest is corrupted"
            )


def make_c4_event_ring_acknowledgement_candidate_r11_v1(
    event_ring: TensorWorldC4EventRingR11V1,
    run_control: TensorWorldC5GraphRunControlR11V1,
    *,
    owner_binding: str,
    acceptance_kind: str,
    artifact_digest: str | None = None,
) -> TensorWorldC4EventRingAcknowledgementCandidateR11V1:
    """Build a candidate from already-materialized boundary state."""

    event_ring.validate(semantic=True)
    _validate_completed_window_v1(run_control)
    if event_ring.device.type != "cpu" or run_control.device.type != "cpu":
        raise TensorWorldContractError(
            "R11 candidate construction requires materialized CPU state"
        )
    snapshot_ring = _clone_to_cpu(event_ring)
    snapshot_control = _clone_to_cpu(run_control)
    candidate = TensorWorldC4EventRingAcknowledgementCandidateR11V1(
        event_ring=snapshot_ring,
        run_control=snapshot_control,
        owner_binding=owner_binding,
        acceptance_kind=acceptance_kind,
        artifact_digest=artifact_digest,
        digest=_candidate_digest_v1(
            ring=snapshot_ring,
            run_control=snapshot_control,
            owner_binding=owner_binding,
            acceptance_kind=acceptance_kind,
            artifact_digest=artifact_digest,
        ),
    )
    candidate.validate()
    return candidate


def materialize_c4_event_ring_report_boundary_r11_v1(
    owner: TensorWorldC4ResidentComposedR11OwnerV1,
) -> TensorWorldC4EventRingAcknowledgementCandidateR11V1:
    """Perform the one permitted post-window D2H report materialization."""

    owner_binding = _owner_binding_v1(owner)
    ring = owner.bundle.event_ring
    staging = owner.shadow_staging
    if ring is None or staging is None:
        raise TensorWorldContractError("R11 CUDA owner boundary is incomplete")
    materialized_ring = _clone_to_cpu(ring)
    materialized_control = _clone_to_cpu(
        bind_c5_graph_run_control_r11_v1(staging)
    )
    return make_c4_event_ring_acknowledgement_candidate_r11_v1(
        materialized_ring,
        materialized_control,
        owner_binding=owner_binding,
        acceptance_kind=C4_EVENT_RING_CUDA_BOUNDARY_R11_REPORT_V1,
    )


def prepare_c4_event_ring_checkpoint_boundary_r11_v1(
    owner: TensorWorldC4ResidentComposedR11OwnerV1,
    checkpoint: TensorWorldC5GraphCheckpointR11V1,
) -> TensorWorldC4EventRingAcknowledgementCandidateR11V1:
    """Reuse an existing C5 checkpoint snapshot with zero additional D2H."""

    owner_binding = _owner_binding_v1(owner)
    checkpoint.validate()
    ring = checkpoint.base_checkpoint.state.event_ring
    if ring is None:
        raise TensorWorldContractError("R11 C5 checkpoint has no event ring")
    return make_c4_event_ring_acknowledgement_candidate_r11_v1(
        ring,
        checkpoint.run_control,
        owner_binding=owner_binding,
        acceptance_kind=C4_EVENT_RING_CUDA_BOUNDARY_R11_CHECKPOINT_V1,
        artifact_digest=checkpoint.digest,
    )


def _guarded_acknowledge_ring_v1(
    ring: TensorWorldC4EventRingR11V1,
    run_control: TensorWorldC5GraphRunControlR11V1,
    replay_event_sequence: torch.Tensor,
    candidate: TensorWorldC4EventRingAcknowledgementCandidateR11V1,
) -> torch.Tensor:
    """Advance consumer cursor all-or-none; return a device scalar status."""

    candidate.validate()
    device = ring.device
    expected_ring = candidate.event_ring
    expected_control = candidate.run_control
    expected_producer = expected_ring.producer_cursor.to(device=device)
    ring_ok = torch.ones((ring.batch,), dtype=torch.bool, device=device)
    for actual, expected in zip(
        _event_ring_tensors(ring),
        _event_ring_tensors(expected_ring),
        strict=True,
    ):
        expected_device = expected.to(device=device)
        if actual.dtype.is_floating_point:
            actual = actual.view(torch.int32)
            expected_device = expected_device.view(torch.int32)
        ring_ok &= (actual == expected_device).reshape(ring.batch, -1).all(dim=1)
    world_ok = (
        ring_ok
        & (replay_event_sequence == expected_producer)
        & (run_control.generation == expected_control.generation.to(device=device))
        & (run_control.tick == expected_control.tick.to(device=device))
        & (
            run_control.requested_ticks
            == expected_control.requested_ticks.to(device=device)
        )
        & (
            run_control.completed_ticks
            == expected_control.completed_ticks.to(device=device)
        )
        & (
            run_control.current_tick
            == expected_control.current_tick.to(device=device)
        )
        & (
            run_control.receipt_write_index
            == expected_control.receipt_write_index.to(device=device)
        )
        & (run_control.global_status == expected_control.global_status.to(device=device))
        & (run_control.stop_requested == expected_control.stop_requested.to(device=device))
    )
    global_ok = torch.all(world_ok)
    ring.consumer_cursor.copy_(
        torch.where(global_ok, expected_producer, ring.consumer_cursor)
    )
    return global_ok.to(dtype=torch.int32).reshape(1)


def accept_c4_event_ring_boundary_r11_v1(
    owner: TensorWorldC4ResidentComposedR11OwnerV1,
    candidate: TensorWorldC4EventRingAcknowledgementCandidateR11V1,
    *,
    accepted_artifact_digest: str,
) -> torch.Tensor:
    """Publish acknowledgement after artifact acceptance, without a D2H read."""

    candidate.validate()
    _validate_sha256(accepted_artifact_digest, name="accepted artifact digest")
    if accepted_artifact_digest != candidate.acceptance_digest:
        raise TensorWorldContractError(
            "R11 event acknowledgement artifact was not accepted"
        )
    if _owner_binding_v1(owner) != candidate.owner_binding:
        raise TensorWorldContractError(
            "R11 event acknowledgement candidate belongs to another CUDA owner"
        )
    ring = owner.bundle.event_ring
    staging = owner.shadow_staging
    if ring is None or staging is None:
        raise TensorWorldContractError("R11 CUDA owner boundary is incomplete")
    control = bind_c5_graph_run_control_r11_v1(staging)
    return _guarded_acknowledge_ring_v1(
        ring,
        control,
        owner.replay.event_sequence,
        candidate,
    )


def c4_event_ring_cuda_boundary_r11_source_receipt_v1() -> dict[str, object]:
    sources = {
        "adapter": Path(__file__),
        "event_ring": EVENT_RING_SOURCE,
        "graph_owner": GRAPH_OWNER_SOURCE,
        "graph_checkpoint": GRAPH_CHECKPOINT_SOURCE,
        "lifetime_evidence": LIFETIME_SOURCE,
    }
    return {
        "schema_version": C4_EVENT_RING_CUDA_BOUNDARY_R11_SCHEMA_VERSION_V1,
        "decision": C4_EVENT_RING_CUDA_BOUNDARY_R11_DECISION_V1,
        "boundary": "after_complete_five_tick_graph_window_only",
        "normal_tick_cpu_semantic_work": False,
        "between_tick_cpu_semantic_work": False,
        "report_materialization": "one_grouped_boundary_D2H",
        "checkpoint_materialization": "reuse_C5_snapshot_zero_extra_D2H",
        "acknowledgement": "device_guarded_consumer_cursor_publish_after_acceptance",
        "reclamation": "next_native_window_only",
        "bank_payload_host_mutation": False,
        "native_source_changed": False,
        "gate_claim": False,
        "qualification_impact": {
            "resident_native_graph_topology": "PRESERVED_UNCHANGED",
            "post_window_acknowledgement": "DELTA_QUALIFICATION_REQUIRED",
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
    "C4_EVENT_RING_CUDA_BOUNDARY_R11_CHECKPOINT_V1",
    "C4_EVENT_RING_CUDA_BOUNDARY_R11_DECISION_V1",
    "C4_EVENT_RING_CUDA_BOUNDARY_R11_REPORT_V1",
    "C4_EVENT_RING_CUDA_BOUNDARY_R11_SCHEMA_VERSION_V1",
    "C4_EVENT_RING_CUDA_BOUNDARY_R11_WINDOW_TICKS_V1",
    "TensorWorldC4EventRingAcknowledgementCandidateR11V1",
    "accept_c4_event_ring_boundary_r11_v1",
    "c4_event_ring_cuda_boundary_r11_source_receipt_v1",
    "make_c4_event_ring_acknowledgement_candidate_r11_v1",
    "materialize_c4_event_ring_report_boundary_r11_v1",
    "prepare_c4_event_ring_checkpoint_boundary_r11_v1",
]
