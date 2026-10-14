"""Preparation-bound C6 control, metric, and evidence boundaries.

The C6 provider is not admitted yet.  These contracts rehearse the domain
boundary which a future provider must satisfy: controls are applied only at an
explicit safe boundary, metrics are bounded scalar surfaces, and evidence is
bound to the exact route resolution.  They intentionally contain no tensors
and do not launch or select an execution backend.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import StrEnum
import hashlib
import json
import math
from typing import Any, Final, Mapping

from .c6_contract import (
    TensorWorldC6FailureCodeV1,
    TensorWorldC6RouteRequestV1,
    TensorWorldC6RouteResolutionV1,
    resolve_tensor_world_c6_route_v1,
)
from .contracts import TensorWorldContractError


C6_EXTERNAL_CONTRACT_SCHEMA_VERSION_V1: Final[str] = (
    "tensor_world_c6_external_boundary_v1"
)
C6_CONTROL_SCHEMA_VERSION_V1: Final[str] = "tensor_world_c6_control_request_v1"
C6_CONTROL_RECEIPT_SCHEMA_VERSION_V1: Final[str] = (
    "tensor_world_c6_control_receipt_v1"
)
C6_METRIC_SCHEMA_VERSION_V1: Final[str] = "tensor_world_c6_metric_surface_v1"
C6_EVIDENCE_MANIFEST_SCHEMA_VERSION_V1: Final[str] = (
    "tensor_world_c6_evidence_manifest_v1"
)


class TensorWorldC6SafeBoundaryV1(StrEnum):
    INITIALIZATION = "INITIALIZATION"
    TICK_BOUNDARY = "TICK_BOUNDARY"
    CHECKPOINT_BOUNDARY = "CHECKPOINT_BOUNDARY"
    RUN_END = "RUN_END"


class TensorWorldC6ControlKindV1(StrEnum):
    PRESSURE_UPDATE = "PRESSURE_UPDATE"
    PAUSE = "PAUSE"
    RESUME = "RESUME"
    STOP = "STOP"
    CHECKPOINT = "CHECKPOINT"
    RECONFIGURE = "RECONFIGURE"


def _canonical_json(value: Any, *, label: str) -> str:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise TensorWorldContractError(f"C6 {label} must be JSON-compatible") from exc


def canonical_c6_digest_v1(value: Any, *, label: str = "value") -> str:
    """Hash a JSON boundary value using one deterministic encoding."""

    encoded = _canonical_json(value, label=label)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class TensorWorldC6ControlRequestV1:
    sequence: int
    control_id: str
    kind: TensorWorldC6ControlKindV1
    boundary: TensorWorldC6SafeBoundaryV1
    payload: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = C6_CONTROL_SCHEMA_VERSION_V1

    def __post_init__(self) -> None:
        if self.schema_version != C6_CONTROL_SCHEMA_VERSION_V1:
            raise TensorWorldContractError("C6 control request schema version is unsupported")
        if type(self.sequence) is not int or self.sequence <= 0:
            raise TensorWorldContractError("C6 control sequence must be a positive int")
        if not self.control_id:
            raise TensorWorldContractError("C6 control identity is required")
        if not isinstance(self.kind, TensorWorldC6ControlKindV1):
            raise TensorWorldContractError("C6 control kind is invalid")
        if not isinstance(self.boundary, TensorWorldC6SafeBoundaryV1):
            raise TensorWorldContractError("C6 control boundary is invalid")
        if not isinstance(self.payload, Mapping):
            raise TensorWorldContractError("C6 control payload must be a mapping")
        if any(type(key) is not str or not key for key in self.payload):
            raise TensorWorldContractError("C6 control payload keys must be nonempty strings")
        _canonical_json(dict(self.payload), label="control payload")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "sequence": self.sequence,
            "control_id": self.control_id,
            "kind": self.kind.value,
            "boundary": self.boundary.value,
            "payload": dict(self.payload),
        }

    def fingerprint(self) -> str:
        return canonical_c6_digest_v1(self.to_dict(), label="control request")


@dataclass(frozen=True, slots=True)
class TensorWorldC6ControlStateV1:
    paused: bool = False
    stopped: bool = False
    pressure: float = 0.0
    checkpoint_requested: bool = False
    configuration_fingerprint: str = "0" * 64
    last_sequence: int = 0
    schema_version: str = C6_EXTERNAL_CONTRACT_SCHEMA_VERSION_V1

    def __post_init__(self) -> None:
        if self.schema_version != C6_EXTERNAL_CONTRACT_SCHEMA_VERSION_V1:
            raise TensorWorldContractError("C6 control state schema version is unsupported")
        for name in ("paused", "stopped", "checkpoint_requested"):
            if type(getattr(self, name)) is not bool:
                raise TensorWorldContractError(f"C6 {name} must be bool")
        if type(self.pressure) is not float or not math.isfinite(self.pressure):
            raise TensorWorldContractError("C6 pressure must be a finite float")
        if not 0.0 <= self.pressure <= 1.0:
            raise TensorWorldContractError("C6 pressure must be in [0, 1]")
        if len(self.configuration_fingerprint) != 64 or any(
            character not in "0123456789abcdef"
            for character in self.configuration_fingerprint
        ):
            raise TensorWorldContractError("C6 configuration fingerprint is not SHA256")
        if type(self.last_sequence) is not int or self.last_sequence < 0:
            raise TensorWorldContractError("C6 last control sequence must be nonnegative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "paused": self.paused,
            "stopped": self.stopped,
            "pressure": self.pressure,
            "checkpoint_requested": self.checkpoint_requested,
            "configuration_fingerprint": self.configuration_fingerprint,
            "last_sequence": self.last_sequence,
        }

    def fingerprint(self) -> str:
        return canonical_c6_digest_v1(self.to_dict(), label="control state")


@dataclass(frozen=True, slots=True)
class TensorWorldC6ControlReceiptV1:
    control_id: str
    sequence: int
    kind: TensorWorldC6ControlKindV1
    boundary: TensorWorldC6SafeBoundaryV1
    accepted: bool
    applied: bool
    state_fingerprint: str
    failure_code: TensorWorldC6FailureCodeV1 | None = None
    failure_reason: str | None = None
    schema_version: str = C6_CONTROL_RECEIPT_SCHEMA_VERSION_V1

    def __post_init__(self) -> None:
        if self.schema_version != C6_CONTROL_RECEIPT_SCHEMA_VERSION_V1:
            raise TensorWorldContractError("C6 control receipt schema version is unsupported")
        if not self.control_id or type(self.sequence) is not int or self.sequence <= 0:
            raise TensorWorldContractError("C6 control receipt identity is invalid")
        if not isinstance(self.kind, TensorWorldC6ControlKindV1) or not isinstance(
            self.boundary, TensorWorldC6SafeBoundaryV1
        ):
            raise TensorWorldContractError("C6 control receipt kind or boundary is invalid")
        if type(self.accepted) is not bool or type(self.applied) is not bool:
            raise TensorWorldContractError("C6 control receipt status is invalid")
        if self.applied and not self.accepted:
            raise TensorWorldContractError("C6 control cannot be applied when rejected")
        if len(self.state_fingerprint) != 64 or any(
            character not in "0123456789abcdef" for character in self.state_fingerprint
        ):
            raise TensorWorldContractError("C6 control state fingerprint is not SHA256")
        if self.accepted and (self.failure_code is not None or self.failure_reason is not None):
            raise TensorWorldContractError("accepted C6 control cannot carry a failure")
        if not self.accepted and (self.failure_code is None or not self.failure_reason):
            raise TensorWorldContractError("rejected C6 control requires a failure")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "control_id": self.control_id,
            "sequence": self.sequence,
            "kind": self.kind.value,
            "boundary": self.boundary.value,
            "accepted": self.accepted,
            "applied": self.applied,
            "state_fingerprint": self.state_fingerprint,
            "failure_code": None if self.failure_code is None else self.failure_code.value,
            "failure_reason": self.failure_reason,
        }

    def fingerprint(self) -> str:
        return canonical_c6_digest_v1(self.to_dict(), label="control receipt")


def _rejected_control_receipt(
    request: TensorWorldC6ControlRequestV1,
    state: TensorWorldC6ControlStateV1,
    reason: str,
) -> TensorWorldC6ControlReceiptV1:
    return TensorWorldC6ControlReceiptV1(
        control_id=request.control_id,
        sequence=request.sequence,
        kind=request.kind,
        boundary=request.boundary,
        accepted=False,
        applied=False,
        state_fingerprint=state.fingerprint(),
        failure_code=TensorWorldC6FailureCodeV1.CONTROL_BOUNDARY,
        failure_reason=reason,
    )


def apply_c6_control_oracle_v1(
    state: TensorWorldC6ControlStateV1,
    request: TensorWorldC6ControlRequestV1,
    *,
    actual_boundary: TensorWorldC6SafeBoundaryV1,
) -> tuple[TensorWorldC6ControlStateV1, TensorWorldC6ControlReceiptV1]:
    """Apply one control at a declared boundary without a per-tick host path."""

    if not isinstance(state, TensorWorldC6ControlStateV1):
        raise TensorWorldContractError("C6 control state type is invalid")
    if not isinstance(request, TensorWorldC6ControlRequestV1):
        raise TensorWorldContractError("C6 control request type is invalid")
    if not isinstance(actual_boundary, TensorWorldC6SafeBoundaryV1):
        raise TensorWorldContractError("C6 actual boundary is invalid")
    if request.boundary is not actual_boundary:
        return state, _rejected_control_receipt(
            request, state, "control was submitted at a different safe boundary"
        )
    if request.sequence <= state.last_sequence:
        return state, _rejected_control_receipt(
            request, state, "control sequence is not strictly increasing"
        )
    if state.stopped:
        return state, _rejected_control_receipt(request, state, "run is already stopped")
    if request.kind is TensorWorldC6ControlKindV1.RECONFIGURE and actual_boundary not in (
        TensorWorldC6SafeBoundaryV1.INITIALIZATION,
        TensorWorldC6SafeBoundaryV1.CHECKPOINT_BOUNDARY,
    ):
        return state, _rejected_control_receipt(
            request, state, "reconfiguration is only allowed at initialization or checkpoint"
        )
    next_state = replace(state, last_sequence=request.sequence)
    if request.kind is TensorWorldC6ControlKindV1.PRESSURE_UPDATE:
        value = request.payload.get("pressure")
        if type(value) is not float or not math.isfinite(value) or not 0.0 <= value <= 1.0:
            return state, _rejected_control_receipt(
                request, state, "pressure update requires finite float pressure in [0, 1]"
            )
        next_state = replace(next_state, pressure=value)
    elif request.kind is TensorWorldC6ControlKindV1.PAUSE:
        next_state = replace(next_state, paused=True)
    elif request.kind is TensorWorldC6ControlKindV1.RESUME:
        next_state = replace(next_state, paused=False)
    elif request.kind is TensorWorldC6ControlKindV1.STOP:
        next_state = replace(next_state, stopped=True, paused=True)
    elif request.kind is TensorWorldC6ControlKindV1.CHECKPOINT:
        next_state = replace(next_state, checkpoint_requested=True)
    elif request.kind is TensorWorldC6ControlKindV1.RECONFIGURE:
        next_state = replace(
            next_state,
            configuration_fingerprint=canonical_c6_digest_v1(
                dict(request.payload), label="reconfiguration payload"
            ),
        )
    receipt = TensorWorldC6ControlReceiptV1(
        control_id=request.control_id,
        sequence=request.sequence,
        kind=request.kind,
        boundary=request.boundary,
        accepted=True,
        applied=True,
        state_fingerprint=next_state.fingerprint(),
    )
    return next_state, receipt


@dataclass(frozen=True, slots=True)
class TensorWorldC6MetricSurfaceV1:
    route_fingerprint: str
    tick: int
    metrics: Mapping[str, int | float]
    event_counts: Mapping[str, int] = field(default_factory=dict)
    host_callback_count: int = 0
    device_to_host_bytes: int = 0
    host_to_device_bytes: int = 0
    schema_version: str = C6_METRIC_SCHEMA_VERSION_V1

    def __post_init__(self) -> None:
        if self.schema_version != C6_METRIC_SCHEMA_VERSION_V1:
            raise TensorWorldContractError("C6 metric schema version is unsupported")
        if len(self.route_fingerprint) != 64 or any(
            character not in "0123456789abcdef" for character in self.route_fingerprint
        ):
            raise TensorWorldContractError("C6 metric route fingerprint is not SHA256")
        if type(self.tick) is not int or self.tick < 0:
            raise TensorWorldContractError("C6 metric tick must be nonnegative")
        if not isinstance(self.metrics, Mapping) or not self.metrics:
            raise TensorWorldContractError("C6 metrics must be a nonempty mapping")
        if not isinstance(self.event_counts, Mapping):
            raise TensorWorldContractError("C6 event counts must be a mapping")
        for name, value in self.metrics.items():
            if type(name) is not str or not name:
                raise TensorWorldContractError("C6 metric names must be nonempty strings")
            if type(value) not in (int, float) or isinstance(value, bool):
                raise TensorWorldContractError("C6 metric values must be numeric scalars")
            if isinstance(value, float) and not math.isfinite(value):
                raise TensorWorldContractError("C6 metric floats must be finite")
        for name, value in self.event_counts.items():
            if type(name) is not str or not name or type(value) is not int or value < 0:
                raise TensorWorldContractError("C6 event counts must be nonnegative integers")
        for name in (
            "host_callback_count",
            "device_to_host_bytes",
            "host_to_device_bytes",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise TensorWorldContractError(f"C6 {name} must be nonnegative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "route_fingerprint": self.route_fingerprint,
            "tick": self.tick,
            "metrics": dict(self.metrics),
            "event_counts": dict(self.event_counts),
            "host_callback_count": self.host_callback_count,
            "device_to_host_bytes": self.device_to_host_bytes,
            "host_to_device_bytes": self.host_to_device_bytes,
        }

    def fingerprint(self) -> str:
        return canonical_c6_digest_v1(self.to_dict(), label="metric surface")


@dataclass(frozen=True, slots=True)
class TensorWorldC6EvidenceManifestV1:
    source_commit: str
    provider_contract_id: str
    backend_id: str
    route_fingerprint: str
    evidence_root: str
    route_accepted: bool
    control_receipt_digest: str
    metrics_digest: str
    control_count: int
    metric_surface_count: int
    control_boundary_rehearsal: bool
    metrics_schema_rehearsal: bool
    status: str = "PREPARATION_ONLY"
    gate_decision: str = "NO_GATE_DECISION"
    schema_version: str = C6_EVIDENCE_MANIFEST_SCHEMA_VERSION_V1

    def __post_init__(self) -> None:
        if self.schema_version != C6_EVIDENCE_MANIFEST_SCHEMA_VERSION_V1:
            raise TensorWorldContractError("C6 evidence manifest schema version is unsupported")
        if not self.source_commit or not self.provider_contract_id or not self.backend_id:
            raise TensorWorldContractError("C6 evidence identity is incomplete")
        if not self.evidence_root.startswith("/"):
            raise TensorWorldContractError("C6 evidence root must be absolute")
        for name in ("route_fingerprint", "control_receipt_digest", "metrics_digest"):
            value = getattr(self, name)
            if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
                raise TensorWorldContractError(f"C6 {name} is not SHA256")
        for name in ("route_accepted", "control_boundary_rehearsal", "metrics_schema_rehearsal"):
            if type(getattr(self, name)) is not bool:
                raise TensorWorldContractError(f"C6 {name} must be bool")
        for name in ("control_count", "metric_surface_count"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise TensorWorldContractError(f"C6 {name} must be nonnegative")
        if self.status != "PREPARATION_ONLY" or self.gate_decision != "NO_GATE_DECISION":
            raise TensorWorldContractError("C6 preparation manifest cannot claim a gate decision")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "phase": "TW-C6",
            "status": self.status,
            "gate_decision": self.gate_decision,
            "source_commit": self.source_commit,
            "provider_contract_id": self.provider_contract_id,
            "backend_id": self.backend_id,
            "route_fingerprint": self.route_fingerprint,
            "evidence_root": self.evidence_root,
            "route_accepted": self.route_accepted,
            "control_receipt_digest": self.control_receipt_digest,
            "metrics_digest": self.metrics_digest,
            "control_count": self.control_count,
            "metric_surface_count": self.metric_surface_count,
            "control_boundary_rehearsal": self.control_boundary_rehearsal,
            "metrics_schema_rehearsal": self.metrics_schema_rehearsal,
        }

    def fingerprint(self) -> str:
        return canonical_c6_digest_v1(self.to_dict(), label="C6 evidence manifest")


def build_c6_evidence_manifest_v1(
    resolution: TensorWorldC6RouteResolutionV1,
    controls: tuple[TensorWorldC6ControlReceiptV1, ...],
    metrics: tuple[TensorWorldC6MetricSurfaceV1, ...],
    *,
    source_commit: str,
    evidence_root: str,
) -> TensorWorldC6EvidenceManifestV1:
    """Bind preparation receipts to one route without admitting that route."""

    if not isinstance(resolution, TensorWorldC6RouteResolutionV1):
        raise TensorWorldContractError("C6 evidence resolution type is invalid")
    if not isinstance(controls, tuple) or not isinstance(metrics, tuple):
        raise TensorWorldContractError("C6 evidence surfaces must be tuples")
    if any(not isinstance(receipt, TensorWorldC6ControlReceiptV1) for receipt in controls):
        raise TensorWorldContractError("C6 evidence contains an invalid control receipt")
    if any(not isinstance(surface, TensorWorldC6MetricSurfaceV1) for surface in metrics):
        raise TensorWorldContractError("C6 evidence contains an invalid metric surface")
    sequences = [receipt.sequence for receipt in controls]
    if sequences != sorted(set(sequences)):
        raise TensorWorldContractError("C6 control receipts must have unique ordered sequences")
    if any(surface.route_fingerprint != resolution.route_fingerprint for surface in metrics):
        raise TensorWorldContractError("C6 metrics are bound to a different route")
    control_dicts = [receipt.to_dict() for receipt in controls]
    metric_dicts = [surface.to_dict() for surface in metrics]
    return TensorWorldC6EvidenceManifestV1(
        source_commit=source_commit,
        provider_contract_id="tensor_world_provider_contract_v1",
        backend_id=resolution.backend_id,
        route_fingerprint=resolution.route_fingerprint,
        evidence_root=evidence_root,
        route_accepted=resolution.accepted,
        control_receipt_digest=canonical_c6_digest_v1(control_dicts, label="control evidence"),
        metrics_digest=canonical_c6_digest_v1(metric_dicts, label="metric evidence"),
        control_count=len(controls),
        metric_surface_count=len(metrics),
        control_boundary_rehearsal=bool(controls) and all(
            receipt.accepted and receipt.applied for receipt in controls
        ),
        metrics_schema_rehearsal=bool(metrics),
    )


def run_c6_cpu_boundary_rehearsal_v1(
    request: TensorWorldC6RouteRequestV1,
    *,
    source_commit: str,
    evidence_root: str,
) -> TensorWorldC6EvidenceManifestV1:
    """Run a deterministic, provider-free C6 boundary rehearsal.

    This is an engineering-preparation oracle.  It intentionally resolves the
    current default manifest in non-strict mode, so the resulting evidence
    records the route rejection rather than inventing a provider launch.
    """

    if request.strict:
        request = replace(request, strict=False)
    resolution = resolve_tensor_world_c6_route_v1(request)
    state = TensorWorldC6ControlStateV1()
    controls: list[TensorWorldC6ControlReceiptV1] = []
    sequence = (
        (1, "pressure", TensorWorldC6ControlKindV1.PRESSURE_UPDATE,
         TensorWorldC6SafeBoundaryV1.TICK_BOUNDARY, {"pressure": 0.25}),
        (2, "checkpoint", TensorWorldC6ControlKindV1.CHECKPOINT,
         TensorWorldC6SafeBoundaryV1.CHECKPOINT_BOUNDARY, {}),
        (3, "pause", TensorWorldC6ControlKindV1.PAUSE,
         TensorWorldC6SafeBoundaryV1.TICK_BOUNDARY, {}),
    )
    for number, control_id, kind, boundary, payload in sequence:
        state, receipt = apply_c6_control_oracle_v1(
            state,
            TensorWorldC6ControlRequestV1(
                sequence=number,
                control_id=control_id,
                kind=kind,
                boundary=boundary,
                payload=payload,
            ),
            actual_boundary=boundary,
        )
        controls.append(receipt)
    metrics = (
        TensorWorldC6MetricSurfaceV1(
            route_fingerprint=resolution.route_fingerprint,
            tick=8,
            metrics={
                "active_population": request.initial_population,
                "host_callback_count": 0,
                "device_to_host_bytes": 0,
                "host_to_device_bytes": 0,
            },
            event_counts={"control_receipts": len(controls)},
        ),
    )
    return build_c6_evidence_manifest_v1(
        resolution,
        tuple(controls),
        metrics,
        source_commit=source_commit,
        evidence_root=evidence_root,
    )


__all__ = [
    "C6_CONTROL_RECEIPT_SCHEMA_VERSION_V1",
    "C6_CONTROL_SCHEMA_VERSION_V1",
    "C6_EVIDENCE_MANIFEST_SCHEMA_VERSION_V1",
    "C6_EXTERNAL_CONTRACT_SCHEMA_VERSION_V1",
    "C6_METRIC_SCHEMA_VERSION_V1",
    "TensorWorldC6ControlKindV1",
    "TensorWorldC6ControlReceiptV1",
    "TensorWorldC6ControlRequestV1",
    "TensorWorldC6ControlStateV1",
    "TensorWorldC6EvidenceManifestV1",
    "TensorWorldC6MetricSurfaceV1",
    "TensorWorldC6SafeBoundaryV1",
    "apply_c6_control_oracle_v1",
    "build_c6_evidence_manifest_v1",
    "canonical_c6_digest_v1",
    "run_c6_cpu_boundary_rehearsal_v1",
]
