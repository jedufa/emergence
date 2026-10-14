"""Explicit control-boundary checkpoint owner for C5/R11 graph continuation.

This module is deliberately absent from normal tick execution.  It snapshots
the existing C5 bundle/replay checkpoint together with the R11 graph owner's
trajectory-relevant run-control values.  CUDA graph executables are not
serialized; a restored owner must bind and capture fresh graphs before replay.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Final, Mapping

import torch

from .c4_identity_r11 import bind_c4_identity_authority_r11_v1
from .c4_resident_composed_r11 import (
    C4_RESIDENT_COMPOSED_R11_ADMITTED_CAPACITIES_V1,
)
from .c4_state import TensorWorldStateBundleV4
from .c5_determinism import (
    TensorWorldC5CheckpointV1,
    TensorWorldC5ReplayContextV1,
    _canonical_update,
    _clone_to_device,
    _clone_to_cpu,
    capture_c5_checkpoint_v1,
    restore_c5_checkpoint_v1,
)
from .contracts import TensorWorldContractError


ROOT = Path(__file__).resolve().parents[2]
C5_SOURCE = ROOT / "engine/tensor_world/c5_determinism.py"
R11_OWNER_SOURCE = ROOT / "engine/tensor_world/c4_resident_composed_r11.py"
R11_EVENT_RING_SOURCE = ROOT / "engine/tensor_world/c4_event_ring_r11.py"
R11_IDENTITY_SOURCE = ROOT / "engine/tensor_world/c4_identity_r11.py"

C5_GRAPH_CHECKPOINT_R11_SCHEMA_VERSION_V1: Final[str] = (
    "tensor_world_c5_graph_checkpoint_r11_v1"
)
C5_GRAPH_RUN_CONTROL_R11_SCHEMA_VERSION_V1: Final[str] = (
    "tensor_world_c5_graph_run_control_r11_v1"
)
C5_GRAPH_CHECKPOINT_R11_DECISION_V1: Final[str] = "NO_GATE_DECISION"


@dataclass(slots=True)
class TensorWorldC5GraphRunControlR11V1:
    """Aliasable device controls required to resume at a graph boundary."""

    generation: torch.Tensor
    tick: torch.Tensor
    requested_ticks: torch.Tensor
    completed_ticks: torch.Tensor
    current_tick: torch.Tensor
    global_status: torch.Tensor
    first_error_stage: torch.Tensor
    first_error_world: torch.Tensor
    first_error_code: torch.Tensor
    first_error_tick: torch.Tensor
    stop_requested: torch.Tensor
    receipt_write_index: torch.Tensor
    schema_version: str = C5_GRAPH_RUN_CONTROL_R11_SCHEMA_VERSION_V1

    @property
    def device(self) -> torch.device:
        return self.generation.device

    @property
    def batch(self) -> int:
        return int(self.generation.shape[0])

    def validate(self, *, semantic: bool = True) -> None:
        if self.schema_version != C5_GRAPH_RUN_CONTROL_R11_SCHEMA_VERSION_V1:
            raise TensorWorldContractError("C5 R11 graph run-control schema is unsupported")
        expected = (
            ("generation", self.generation, torch.int64),
            ("tick", self.tick, torch.int64),
            ("requested_ticks", self.requested_ticks, torch.int64),
            ("completed_ticks", self.completed_ticks, torch.int64),
            ("current_tick", self.current_tick, torch.int64),
            ("global_status", self.global_status, torch.int32),
            ("first_error_stage", self.first_error_stage, torch.int32),
            ("first_error_world", self.first_error_world, torch.int32),
            ("first_error_code", self.first_error_code, torch.int32),
            ("first_error_tick", self.first_error_tick, torch.int64),
            ("stop_requested", self.stop_requested, torch.int32),
            ("receipt_write_index", self.receipt_write_index, torch.int64),
        )
        devices = {self.device}
        for name, tensor, dtype in expected:
            if (
                not isinstance(tensor, torch.Tensor)
                or tuple(tensor.shape) != (self.batch,)
                or tensor.dtype != dtype
                or not tensor.is_contiguous()
            ):
                raise TensorWorldContractError(
                    f"C5 R11 graph run-control {name} ABI is invalid"
                )
            devices.add(tensor.device)
        if self.batch <= 0 or len(devices) != 1:
            raise TensorWorldContractError(
                "C5 R11 graph run-control tensors must share one device"
            )
        if not semantic or self.device.type != "cpu":
            return
        for _name, tensor, _dtype in expected:
            if not bool((tensor == tensor[0]).all().item()):
                raise TensorWorldContractError(
                    "C5 R11 graph run-control values must be batch-broadcast"
                )
        generation = int(self.generation[0])
        tick = int(self.tick[0])
        requested = int(self.requested_ticks[0])
        completed = int(self.completed_ticks[0])
        current = int(self.current_tick[0])
        receipt = int(self.receipt_write_index[0])
        status = int(self.global_status[0])
        stopped = int(self.stop_requested[0])
        if min(generation, tick, requested, completed, receipt) < 0:
            raise TensorWorldContractError("C5 R11 graph counters must be nonnegative")
        if completed > requested or receipt > completed:
            raise TensorWorldContractError("C5 R11 graph completion counters diverge")
        if current < -1 or current > tick or status not in (0, 1) or stopped not in (0, 1):
            raise TensorWorldContractError("C5 R11 graph control state is invalid")


def make_c5_graph_run_control_r11_v1(
    *, batch: int, device: torch.device | str = "cpu"
) -> TensorWorldC5GraphRunControlR11V1:
    if type(batch) is not int or batch <= 0:
        raise TensorWorldContractError("C5 R11 graph run-control batch must be positive")
    target = torch.device(device)
    control = TensorWorldC5GraphRunControlR11V1(
        generation=torch.zeros(batch, dtype=torch.int64, device=target),
        tick=torch.zeros(batch, dtype=torch.int64, device=target),
        requested_ticks=torch.zeros(batch, dtype=torch.int64, device=target),
        completed_ticks=torch.zeros(batch, dtype=torch.int64, device=target),
        current_tick=torch.full((batch,), -1, dtype=torch.int64, device=target),
        global_status=torch.zeros(batch, dtype=torch.int32, device=target),
        first_error_stage=torch.full((batch,), -1, dtype=torch.int32, device=target),
        first_error_world=torch.full((batch,), -1, dtype=torch.int32, device=target),
        first_error_code=torch.zeros(batch, dtype=torch.int32, device=target),
        first_error_tick=torch.full((batch,), -1, dtype=torch.int64, device=target),
        stop_requested=torch.zeros(batch, dtype=torch.int32, device=target),
        receipt_write_index=torch.zeros(batch, dtype=torch.int64, device=target),
    )
    control.validate(semantic=target.type == "cpu")
    return control


def bind_c5_graph_run_control_r11_v1(staging: object) -> TensorWorldC5GraphRunControlR11V1:
    """Alias the continuation fields of an R11 composed staging owner."""

    names = tuple(
        field.name
        for field in fields(TensorWorldC5GraphRunControlR11V1)
        if field.name != "schema_version"
    )
    try:
        control = TensorWorldC5GraphRunControlR11V1(
            **{name: getattr(staging, name) for name in names}
        )
    except AttributeError as exc:
        raise TensorWorldContractError(
            "C5 R11 graph owner does not expose complete run control"
        ) from exc
    control.validate(semantic=control.device.type == "cpu")
    return control


@dataclass(frozen=True, slots=True)
class TensorWorldC5GraphCheckpointR11V1:
    base_checkpoint: TensorWorldC5CheckpointV1
    run_control: TensorWorldC5GraphRunControlR11V1
    digest: str
    metadata: Mapping[str, Any]
    boundary: str = "between_captured_graph_windows"
    schema_version: str = C5_GRAPH_CHECKPOINT_R11_SCHEMA_VERSION_V1

    def validate(self) -> None:
        if self.schema_version != C5_GRAPH_CHECKPOINT_R11_SCHEMA_VERSION_V1:
            raise TensorWorldContractError("C5 R11 graph checkpoint schema is unsupported")
        if self.boundary != "between_captured_graph_windows":
            raise TensorWorldContractError("C5 R11 checkpoint boundary is unsupported")
        self.base_checkpoint.validate()
        self.base_checkpoint.replay_context.validate_typed_continuation_for_bundle(
            self.base_checkpoint.state
        )
        _validate_r11_bundle_boundary(self.base_checkpoint.state)
        self.run_control.validate(semantic=True)
        if self.run_control.device.type != "cpu":
            raise TensorWorldContractError("C5 R11 graph checkpoint controls must be CPU")
        try:
            json.dumps(self.metadata, sort_keys=True, separators=(",", ":"))
        except (TypeError, ValueError) as exc:
            raise TensorWorldContractError(
                "C5 R11 graph checkpoint metadata must be JSON-compatible"
            ) from exc
        if self.digest != _graph_checkpoint_digest_v1(
            self.base_checkpoint, self.run_control, self.metadata, self.boundary
        ):
            raise TensorWorldContractError("C5 R11 graph checkpoint digest is corrupted")


def _validate_r11_bundle_boundary(bundle: TensorWorldStateBundleV4) -> None:
    if (
        bundle.batch != 48
        or bundle.capacity not in C4_RESIDENT_COMPOSED_R11_ADMITTED_CAPACITIES_V1
    ):
        raise TensorWorldContractError(
            "C5 R11 graph checkpoint requires B48 and admitted C32/C512 capacity"
        )
    ring = bundle.event_ring
    if ring is None or not hasattr(ring, "validate"):
        raise TensorWorldContractError("C5 R11 checkpoint requires the typed event ring")
    ring.validate(semantic=bundle.device.type == "cpu")
    bind_c4_identity_authority_r11_v1(bundle, semantic=True)


def _graph_checkpoint_digest_v1(
    checkpoint: TensorWorldC5CheckpointV1,
    run_control: TensorWorldC5GraphRunControlR11V1,
    metadata: Mapping[str, Any],
    boundary: str,
) -> str:
    hasher = hashlib.sha256()
    hasher.update(C5_GRAPH_CHECKPOINT_R11_SCHEMA_VERSION_V1.encode("ascii"))
    _canonical_update(hasher, checkpoint.digest, "base_checkpoint.digest")
    _canonical_update(hasher, checkpoint.backend_id, "base_checkpoint.backend_id")
    _canonical_update(hasher, checkpoint.source_commit, "base_checkpoint.source_commit")
    _canonical_update(
        hasher, checkpoint.provider_contract_id, "base_checkpoint.provider_contract_id"
    )
    _canonical_update(hasher, checkpoint.metadata, "base_checkpoint.metadata")
    _canonical_update(hasher, run_control, "run_control")
    _canonical_update(hasher, metadata, "metadata")
    _canonical_update(hasher, boundary, "boundary")
    return hasher.hexdigest()


def capture_c5_graph_checkpoint_r11_v1(
    bundle: TensorWorldStateBundleV4,
    replay_context: TensorWorldC5ReplayContextV1,
    run_control: TensorWorldC5GraphRunControlR11V1,
    *,
    backend_id: str,
    source_commit: str,
    provider_contract_id: str,
    metadata: Mapping[str, Any] | None = None,
) -> TensorWorldC5GraphCheckpointR11V1:
    """Capture only at an explicit boundary between graph replays."""

    _validate_r11_bundle_boundary(bundle)
    replay_context.validate_typed_continuation_for_bundle(bundle)
    run_control.validate(semantic=run_control.device.type == "cpu")
    if run_control.batch != bundle.batch or run_control.device != bundle.device:
        raise TensorWorldContractError(
            "C5 R11 graph controls must match the canonical resident owner"
        )
    base = capture_c5_checkpoint_v1(
        bundle,
        replay_context,
        backend_id=backend_id,
        source_commit=source_commit,
        provider_contract_id=provider_contract_id,
        metadata={"control_boundary": "between_captured_graph_windows"},
    )
    control_snapshot = _clone_to_cpu(run_control)
    checkpoint_metadata = dict(metadata or {})
    boundary = "between_captured_graph_windows"
    checkpoint = TensorWorldC5GraphCheckpointR11V1(
        base_checkpoint=base,
        run_control=control_snapshot,
        digest=_graph_checkpoint_digest_v1(
            base, control_snapshot, checkpoint_metadata, boundary
        ),
        metadata=checkpoint_metadata,
        boundary=boundary,
    )
    checkpoint.validate()
    return checkpoint


def _copy_run_control_v1(
    destination: TensorWorldC5GraphRunControlR11V1,
    source: TensorWorldC5GraphRunControlR11V1,
) -> None:
    for field in fields(TensorWorldC5GraphRunControlR11V1):
        if field.name == "schema_version":
            continue
        target = getattr(destination, field.name)
        value = getattr(source, field.name)
        if target.shape != value.shape or target.dtype != value.dtype:
            raise TensorWorldContractError("C5 R11 graph run-control restore ABI differs")
        target.copy_(value.to(device=target.device))


def restore_c5_graph_checkpoint_r11_v1(
    checkpoint: TensorWorldC5GraphCheckpointR11V1,
    destination: TensorWorldStateBundleV4,
    replay_context: TensorWorldC5ReplayContextV1,
    run_control: TensorWorldC5GraphRunControlR11V1,
    *,
    expected_backend_id: str,
    expected_source_commit: str,
    expected_provider_contract_id: str,
) -> str:
    """Restore canonical state before a fresh graph bind/capture operation."""

    checkpoint.validate()
    _validate_r11_bundle_boundary(destination)
    replay_context.validate_typed_continuation_for_bundle(destination)
    run_control.validate(semantic=run_control.device.type == "cpu")
    if run_control.batch != destination.batch or run_control.device != destination.device:
        raise TensorWorldContractError("C5 R11 restore owner devices or batches differ")
    staged_destination = _clone_to_device(destination, destination.device)
    staged_replay = _clone_to_device(replay_context, replay_context.device)
    staged_control = _clone_to_device(run_control, run_control.device)
    restore_c5_checkpoint_v1(
        checkpoint.base_checkpoint,
        staged_destination,
        staged_replay,
        expected_backend_id=expected_backend_id,
        expected_source_commit=expected_source_commit,
        expected_provider_contract_id=expected_provider_contract_id,
    )
    _copy_run_control_v1(staged_control, checkpoint.run_control)
    _validate_r11_bundle_boundary(staged_destination)
    staged_replay.validate_typed_continuation_for_bundle(staged_destination)
    staged_control.validate(semantic=staged_control.device.type == "cpu")
    staged_base = capture_c5_checkpoint_v1(
        staged_destination,
        staged_replay,
        backend_id=checkpoint.base_checkpoint.backend_id,
        source_commit=checkpoint.base_checkpoint.source_commit,
        provider_contract_id=checkpoint.base_checkpoint.provider_contract_id,
        metadata=checkpoint.base_checkpoint.metadata,
    )
    staged_digest = _graph_checkpoint_digest_v1(
        staged_base, _clone_to_cpu(staged_control), checkpoint.metadata, checkpoint.boundary
    )
    if staged_digest != checkpoint.digest:
        raise TensorWorldContractError("C5 R11 staged restore digest differs")
    restore_c5_checkpoint_v1(
        checkpoint.base_checkpoint,
        destination,
        replay_context,
        expected_backend_id=expected_backend_id,
        expected_source_commit=expected_source_commit,
        expected_provider_contract_id=expected_provider_contract_id,
    )
    _copy_run_control_v1(run_control, checkpoint.run_control)
    return staged_digest


def _file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def write_c5_graph_checkpoint_r11_v1(
    checkpoint: TensorWorldC5GraphCheckpointR11V1,
    path: str | Path,
) -> dict[str, object]:
    """Atomically materialize one immutable control-boundary checkpoint."""

    checkpoint.validate()
    target = Path(path).resolve()
    if not target.parent.is_dir():
        raise TensorWorldContractError(
            "C5 R11 checkpoint output directory does not exist"
        )
    if target.exists():
        raise TensorWorldContractError(
            "C5 R11 checkpoint output already exists; overwrite is forbidden"
        )
    temporary = target.with_name(f".{target.name}.incomplete")
    if temporary.exists():
        raise TensorWorldContractError(
            "C5 R11 checkpoint incomplete output already exists"
        )
    try:
        with temporary.open("xb") as stream:
            torch.save(
                {
                    "schema_version": C5_GRAPH_CHECKPOINT_R11_SCHEMA_VERSION_V1,
                    "checkpoint": checkpoint,
                },
                stream,
            )
            stream.flush()
            os.fsync(stream.fileno())
        byte_count = temporary.stat().st_size
        sha256 = _file_sha256(temporary)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "schema_version": C5_GRAPH_CHECKPOINT_R11_SCHEMA_VERSION_V1,
        "operation": "write_c5_graph_checkpoint_r11_v1",
        "path": str(target),
        "bytes": byte_count,
        "sha256": sha256,
        "checkpoint_digest": checkpoint.digest,
        "atomic_publish": True,
        "overwrite_forbidden": True,
        "gate_claim": False,
    }


def load_c5_graph_checkpoint_r11_v1(
    path: str | Path,
    *,
    expected_sha256: str,
) -> TensorWorldC5GraphCheckpointR11V1:
    """Verify immutable bytes before loading and validating a CPU checkpoint."""

    source = Path(path).resolve()
    if not source.is_file():
        raise TensorWorldContractError("C5 R11 checkpoint file is unavailable")
    if (
        len(expected_sha256) != 64
        or any(character not in "0123456789abcdef" for character in expected_sha256)
        or _file_sha256(source) != expected_sha256
    ):
        raise TensorWorldContractError("C5 R11 checkpoint file digest differs")
    try:
        payload = torch.load(source, map_location="cpu", weights_only=False)
    except (OSError, RuntimeError, ValueError, TypeError) as exc:
        raise TensorWorldContractError("C5 R11 checkpoint load failed") from exc
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version")
        != C5_GRAPH_CHECKPOINT_R11_SCHEMA_VERSION_V1
        or not isinstance(
            payload.get("checkpoint"), TensorWorldC5GraphCheckpointR11V1
        )
    ):
        raise TensorWorldContractError("C5 R11 checkpoint file schema differs")
    checkpoint = payload["checkpoint"]
    checkpoint.validate()
    return checkpoint


def c5_graph_checkpoint_r11_source_receipt_v1() -> dict[str, object]:
    sources = {
        "adapter": Path(__file__),
        "base_checkpoint_owner": C5_SOURCE,
        "graph_owner": R11_OWNER_SOURCE,
        "event_ring": R11_EVENT_RING_SOURCE,
        "identity_authority": R11_IDENTITY_SOURCE,
    }
    return {
        "schema_version": C5_GRAPH_CHECKPOINT_R11_SCHEMA_VERSION_V1,
        "decision": C5_GRAPH_CHECKPOINT_R11_DECISION_V1,
        "boundary": "between_captured_graph_windows_only",
        "admitted_capacities": list(
            C4_RESIDENT_COMPOSED_R11_ADMITTED_CAPACITIES_V1
        ),
        "production_geometry": "B48_W32_P256_C512",
        "normal_tick_cpu_work": False,
        "included": [
            "TensorWorldStateBundleV4",
            "typed_C5_replay_context",
            "double_buffer_event_ring",
            "monotonic_identity_and_slot_tokens",
            "graph_run_control_values",
        ],
        "excluded_ephemeral": ["CUDA_graph_executables", "R9_transaction_staging"],
        "restore_requirement": "fresh_owner_bind_and_graph_recapture_before_replay",
        "gate_claim": False,
        "sources": {
            name: {
                "path": str(path.relative_to(ROOT)),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for name, path in sources.items()
        },
    }


__all__ = [
    "C5_GRAPH_CHECKPOINT_R11_DECISION_V1",
    "C5_GRAPH_CHECKPOINT_R11_SCHEMA_VERSION_V1",
    "C5_GRAPH_RUN_CONTROL_R11_SCHEMA_VERSION_V1",
    "TensorWorldC5GraphCheckpointR11V1",
    "TensorWorldC5GraphRunControlR11V1",
    "bind_c5_graph_run_control_r11_v1",
    "c5_graph_checkpoint_r11_source_receipt_v1",
    "capture_c5_graph_checkpoint_r11_v1",
    "make_c5_graph_run_control_r11_v1",
    "load_c5_graph_checkpoint_r11_v1",
    "restore_c5_graph_checkpoint_r11_v1",
    "write_c5_graph_checkpoint_r11_v1",
]
