from __future__ import annotations

"""Single-allocation envelope contract for the Stage 3 packed controller."""

from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Any, Final, Mapping

import torch

from perf.cuda_architecture_b_packed_state import (
    PACKED_STATE_ALIGNMENT_BYTES,
    PackedStateLayout,
    PackedStateLayoutError,
    validate_packed_state_layout,
)


PACKED_CONTROLLER_ENVELOPE_SCHEMA_VERSION: Final[str] = (
    "eiel_cuda_architecture_b_flat_packed_controller_envelope_v1"
)
PACKED_CONTROLLER_ABI_VERSION: Final[int] = 1
CAPTURE_MODES: Final[frozenset[str]] = frozenset({"exactness", "production"})

_TORCH_DTYPE_BY_NAME: Final[dict[str, torch.dtype]] = {
    "float32": torch.float32,
    "int32": torch.int32,
    "int64": torch.int64,
}
_DTYPE_BYTES: Final[dict[str, int]] = {
    name: torch.empty((), dtype=dtype).element_size()
    for name, dtype in _TORCH_DTYPE_BY_NAME.items()
}


def _align(value: int, alignment: int = PACKED_STATE_ALIGNMENT_BYTES) -> int:
    return (value + alignment - 1) // alignment * alignment


def _numel(shape: tuple[int, ...]) -> int:
    result = 1
    for size in shape:
        result *= size
    return result


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return sha256(encoded).hexdigest()


@dataclass(frozen=True)
class PackedControllerWorkspaceField:
    name: str
    dtype_name: str
    shape: tuple[int, ...]
    offset_bytes: int
    nbytes: int
    storage_nbytes: int

    @property
    def end_offset_bytes(self) -> int:
        return self.offset_bytes + self.storage_nbytes

    @property
    def torch_dtype(self) -> torch.dtype:
        return _TORCH_DTYPE_BY_NAME[self.dtype_name]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "dtype": self.dtype_name,
            "shape": list(self.shape),
            "offset_bytes": self.offset_bytes,
            "nbytes": self.nbytes,
            "storage_nbytes": self.storage_nbytes,
            "alignment_bytes": PACKED_STATE_ALIGNMENT_BYTES,
        }


@dataclass(frozen=True)
class FlatPackedControllerEnvelopeLayout:
    schema_version: str
    abi_version: int
    state_layout_manifest_sha256: str
    state_bytes: int
    envelope_bytes: int
    max_ticks: int
    observation_size: int
    action_count: int
    capture_mode: str
    workspace_fields: tuple[PackedControllerWorkspaceField, ...]
    manifest_sha256: str

    @property
    def workspace_fields_by_name(self) -> dict[str, PackedControllerWorkspaceField]:
        return {field.name: field for field in self.workspace_fields}

    def _payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "abi_version": self.abi_version,
            "state_layout_manifest_sha256": self.state_layout_manifest_sha256,
            "state_bytes": self.state_bytes,
            "envelope_bytes": self.envelope_bytes,
            "max_ticks": self.max_ticks,
            "observation_size": self.observation_size,
            "action_count": self.action_count,
            "capture_mode": self.capture_mode,
            "workspace_fields": [field.to_dict() for field in self.workspace_fields],
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self._payload()
        payload["manifest_sha256"] = self.manifest_sha256
        return payload


def _workspace_specs(
    *,
    cells: int,
    max_ticks: int,
    capacity: int,
    observation_size: int,
    action_count: int,
    capture_mode: str,
) -> tuple[tuple[str, str, tuple[int, ...]], ...]:
    cell_tick = (cells, max_ticks)
    tick_slot = (cells, max_ticks, capacity)
    specs: list[tuple[str, str, tuple[int, ...]]] = [
        ("status", "int32", (cells,)),
        ("food_gained", "float32", (cells, capacity)),
        ("active_count", "int64", cell_tick),
        ("policy_refill_rows", "int64", cell_tick),
        ("policy_cache_hit_rows", "int64", cell_tick),
        ("runner_shuffle_draws", "int64", cell_tick),
        ("runner_shuffle_status", "int32", cell_tick),
        ("drive_noise_draws", "int64", cell_tick),
        ("drive_noise_status", "int32", cell_tick),
        ("chemistry_draws", "int64", cell_tick),
        ("chemistry_status", "int32", cell_tick),
        ("active_mechanics_status", "int32", cell_tick),
        ("active_mechanics_counts", "int64", (*cell_tick, 5)),
    ]
    for prefix in ("action_death", "residual_death"):
        specs.extend(
            (
                (f"{prefix}_agent_ids", "int64", tick_slot),
                (f"{prefix}_slots", "int64", tick_slot),
                (f"{prefix}_reason_codes", "int32", tick_slot),
                (f"{prefix}_count", "int64", cell_tick),
            )
        )
    if capture_mode == "exactness":
        specs.extend(
            (
                ("ordered_agent_ids", "int64", tick_slot),
                ("ordered_slots", "int32", tick_slot),
                (
                    "observations",
                    "float32",
                    (*tick_slot, observation_size),
                ),
                ("logits", "float32", (*tick_slot, action_count)),
                ("row_actions", "int32", tick_slot),
                ("actions_by_slot", "int32", tick_slot),
                ("pre_energy", "float32", tick_slot),
                ("pre_health", "float32", tick_slot),
                ("pre_injury", "float32", tick_slot),
                ("pre_survival_steps", "int32", tick_slot),
            )
        )
    return tuple(specs)


def build_flat_packed_controller_envelope_layout(
    state_layout: PackedStateLayout,
    *,
    max_ticks: int,
    observation_size: int,
    action_count: int,
    capture_mode: str = "exactness",
) -> FlatPackedControllerEnvelopeLayout:
    """Append deterministic controller workspace without moving state offsets."""

    validate_packed_state_layout(state_layout)
    if type(max_ticks) is not int or max_ticks <= 0:
        raise PackedStateLayoutError("controller max_ticks must be a positive integer")
    if type(observation_size) is not int or observation_size <= 0:
        raise PackedStateLayoutError(
            "controller observation_size must be a positive integer"
        )
    if type(action_count) is not int or action_count < 7 or action_count > 16:
        raise PackedStateLayoutError("controller action_count must be in [7, 16]")
    if capture_mode not in CAPTURE_MODES:
        raise PackedStateLayoutError(
            f"unsupported controller capture mode: {capture_mode!r}"
        )
    bindings = dict(state_layout.shape_bindings)
    cells = bindings.get("cell")
    capacity = bindings.get("capacity")
    if not cells or not capacity:
        raise PackedStateLayoutError(
            "controller state layout must bind positive cell and capacity dimensions"
        )

    fields: list[PackedControllerWorkspaceField] = []
    cursor = _align(state_layout.total_bytes)
    for name, dtype_name, shape in _workspace_specs(
        cells=cells,
        max_ticks=max_ticks,
        capacity=capacity,
        observation_size=observation_size,
        action_count=action_count,
        capture_mode=capture_mode,
    ):
        cursor = _align(cursor)
        nbytes = _numel(shape) * _DTYPE_BYTES[dtype_name]
        fields.append(
            PackedControllerWorkspaceField(
                name=name,
                dtype_name=dtype_name,
                shape=shape,
                offset_bytes=cursor,
                nbytes=nbytes,
                storage_nbytes=nbytes if nbytes else PACKED_STATE_ALIGNMENT_BYTES,
            )
        )
        cursor += fields[-1].storage_nbytes
    envelope_bytes = _align(cursor)
    provisional = FlatPackedControllerEnvelopeLayout(
        schema_version=PACKED_CONTROLLER_ENVELOPE_SCHEMA_VERSION,
        abi_version=PACKED_CONTROLLER_ABI_VERSION,
        state_layout_manifest_sha256=state_layout.manifest_sha256,
        state_bytes=state_layout.total_bytes,
        envelope_bytes=envelope_bytes,
        max_ticks=max_ticks,
        observation_size=observation_size,
        action_count=action_count,
        capture_mode=capture_mode,
        workspace_fields=tuple(fields),
        manifest_sha256="",
    )
    layout = FlatPackedControllerEnvelopeLayout(
        **{
            **provisional.__dict__,
            "manifest_sha256": _canonical_sha256(provisional._payload()),
        }
    )
    validate_flat_packed_controller_envelope_layout(state_layout, layout)
    return layout


def validate_flat_packed_controller_envelope_layout(
    state_layout: PackedStateLayout,
    envelope_layout: FlatPackedControllerEnvelopeLayout,
) -> None:
    validate_packed_state_layout(state_layout)
    if not isinstance(envelope_layout, FlatPackedControllerEnvelopeLayout):
        raise PackedStateLayoutError(
            "controller envelope layout has the wrong object type"
        )
    if (
        envelope_layout.schema_version
        != PACKED_CONTROLLER_ENVELOPE_SCHEMA_VERSION
        or envelope_layout.abi_version != PACKED_CONTROLLER_ABI_VERSION
    ):
        raise PackedStateLayoutError("controller envelope schema or ABI mismatch")
    if envelope_layout.state_layout_manifest_sha256 != state_layout.manifest_sha256:
        raise PackedStateLayoutError("controller envelope state-layout identity mismatch")
    if envelope_layout.state_bytes != state_layout.total_bytes:
        raise PackedStateLayoutError("controller envelope state prefix size mismatch")
    if envelope_layout.capture_mode not in CAPTURE_MODES:
        raise PackedStateLayoutError("controller envelope capture mode is invalid")
    if type(envelope_layout.max_ticks) is not int or envelope_layout.max_ticks <= 0:
        raise PackedStateLayoutError("controller envelope max_ticks is invalid")
    if (
        type(envelope_layout.observation_size) is not int
        or envelope_layout.observation_size <= 0
    ):
        raise PackedStateLayoutError("controller envelope observation size is invalid")
    if (
        type(envelope_layout.action_count) is not int
        or envelope_layout.action_count < 7
        or envelope_layout.action_count > 16
    ):
        raise PackedStateLayoutError("controller envelope action count is invalid")
    if envelope_layout.manifest_sha256 != _canonical_sha256(envelope_layout._payload()):
        raise PackedStateLayoutError("controller envelope manifest hash mismatch")
    bindings = dict(state_layout.shape_bindings)
    expected_specs = _workspace_specs(
        cells=bindings["cell"],
        max_ticks=envelope_layout.max_ticks,
        capacity=bindings["capacity"],
        observation_size=envelope_layout.observation_size,
        action_count=envelope_layout.action_count,
        capture_mode=envelope_layout.capture_mode,
    )
    actual_specs = tuple(
        (field.name, field.dtype_name, field.shape)
        for field in envelope_layout.workspace_fields
    )
    if actual_specs != expected_specs:
        raise PackedStateLayoutError(
            "controller envelope workspace contract is incomplete or out of order"
        )
    names: set[str] = set()
    cursor = _align(state_layout.total_bytes)
    for field in envelope_layout.workspace_fields:
        if field.name in names:
            raise PackedStateLayoutError(
                f"duplicate controller workspace field: {field.name!r}"
            )
        names.add(field.name)
        if field.dtype_name not in _DTYPE_BYTES:
            raise PackedStateLayoutError(
                f"unsupported controller workspace dtype: {field.dtype_name!r}"
            )
        if any(type(size) is not int or size <= 0 for size in field.shape):
            raise PackedStateLayoutError(
                f"controller workspace field {field.name!r} has an invalid shape"
            )
        expected_offset = _align(cursor)
        expected_nbytes = _numel(field.shape) * _DTYPE_BYTES[field.dtype_name]
        if field.offset_bytes != expected_offset or field.offset_bytes % 64:
            raise PackedStateLayoutError(
                f"controller workspace field {field.name!r} has an invalid offset"
            )
        if field.nbytes != expected_nbytes or field.storage_nbytes != expected_nbytes:
            raise PackedStateLayoutError(
                f"controller workspace field {field.name!r} has an invalid byte size"
            )
        cursor = field.end_offset_bytes
    if envelope_layout.envelope_bytes != _align(cursor):
        raise PackedStateLayoutError("controller envelope total byte size mismatch")


def allocate_flat_packed_controller_envelope(
    state_layout: PackedStateLayout,
    state_buffer: torch.Tensor,
    envelope_layout: FlatPackedControllerEnvelopeLayout,
) -> torch.Tensor:
    """Copy a qualified packed state into an aligned single-allocation envelope."""

    validate_flat_packed_controller_envelope_layout(state_layout, envelope_layout)
    if (
        not isinstance(state_buffer, torch.Tensor)
        or state_buffer.dtype != torch.uint8
        or state_buffer.ndim != 1
        or not state_buffer.is_contiguous()
        or state_buffer.requires_grad
        or int(state_buffer.numel()) != state_layout.total_bytes
        or state_buffer.device.type != "cuda"
    ):
        raise PackedStateLayoutError(
            "controller state buffer must be detached contiguous CUDA uint8 with exact size"
        )
    envelope = torch.zeros(
        envelope_layout.envelope_bytes,
        dtype=torch.uint8,
        device=state_buffer.device,
    )
    envelope.narrow(0, 0, state_layout.total_bytes).copy_(state_buffer)
    return envelope


def packed_controller_workspace_views(
    state_layout: PackedStateLayout,
    envelope_layout: FlatPackedControllerEnvelopeLayout,
    envelope: torch.Tensor,
) -> Mapping[str, torch.Tensor]:
    """Expose zero-copy typed workspace views over the controller envelope."""

    validate_flat_packed_controller_envelope_layout(state_layout, envelope_layout)
    if (
        not isinstance(envelope, torch.Tensor)
        or envelope.dtype != torch.uint8
        or envelope.ndim != 1
        or not envelope.is_contiguous()
        or envelope.requires_grad
        or int(envelope.numel()) != envelope_layout.envelope_bytes
    ):
        raise PackedStateLayoutError(
            "controller envelope must be detached contiguous uint8 with exact size"
        )
    return {
        field.name: envelope.narrow(0, field.offset_bytes, field.nbytes)
        .view(field.torch_dtype)
        .reshape(field.shape)
        for field in envelope_layout.workspace_fields
    }
