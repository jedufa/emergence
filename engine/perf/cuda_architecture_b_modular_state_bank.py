from __future__ import annotations

"""Passive CUDA state-bank contract for the Phase E modular overlay.

This is a separate, optional contract.  It does not extend or replace the
accepted Architecture B 139-field manifest, execute modular state, or claim
checkpoint/parity closure.
"""

from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Any, Final, Mapping

import torch

from perf.cuda_architecture_b_modular_state_abi import (
    MODULAR_STORAGE_PROJECTION_PREFIX,
    MODULAR_STORAGE_PROJECTION_SCHEMA_VERSION,
    MODULAR_STORAGE_SCHEMA_VERSION,
    PHYSICAL_CONNECTION_SLOTS,
    PHYSICAL_MODULE_SLOTS,
    ModularStorageError,
    PackedModularStorageV1,
    build_modular_storage_layout,
    modular_storage_from_projection,
)


MODULAR_STATE_BANK_MANIFEST_SCHEMA_VERSION: Final[str] = (
    "eiel_cuda_architecture_b_modular_state_bank_manifest_v1"
)
MODULAR_STATE_BANK_RECEIPT_SCHEMA_VERSION: Final[str] = (
    "eiel_cuda_architecture_b_modular_state_bank_receipt_v1"
)
MODULAR_STATE_BANK_FIELD_COUNT: Final[int] = 57


class ModularStateBankError(RuntimeError):
    """The optional modular boundary overlay is malformed or contradictory."""


def _canonical_sha256(value: Any) -> str:
    return sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class ModularStateBankFieldContract:
    logical_name: str
    projected_name: str
    dtype_name: str
    shape: tuple[int, ...]
    fill_value: int | float | bool

    @property
    def torch_dtype(self) -> torch.dtype:
        return {
            "bool": torch.bool,
            "float32": torch.float32,
            "float64": torch.float64,
            "int8": torch.int8,
            "int32": torch.int32,
            "int64": torch.int64,
        }[self.dtype_name]

    def to_dict(self) -> dict[str, Any]:
        return {
            "logical_name": self.logical_name,
            "name": self.projected_name,
            "dtype": self.dtype_name,
            "shape": list(self.shape),
            "fill_value": self.fill_value,
        }


@dataclass(frozen=True)
class ModularStateBankManifest:
    schema_version: str
    storage_schema_version: str
    projection_schema_version: str
    projection_prefix: str
    profile_id: str
    agent_capacity: int
    logical_module_slots: int
    logical_connection_slots: int
    physical_module_slots: int
    physical_connection_slots: int
    fields: tuple[ModularStateBankFieldContract, ...]
    manifest_sha256: str

    @property
    def fields_by_projected_name(self) -> dict[str, ModularStateBankFieldContract]:
        return {field.projected_name: field for field in self.fields}

    def _payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "storage_schema_version": self.storage_schema_version,
            "projection_schema_version": self.projection_schema_version,
            "projection_prefix": self.projection_prefix,
            "profile_id": self.profile_id,
            "agent_capacity": self.agent_capacity,
            "logical_module_slots": self.logical_module_slots,
            "logical_connection_slots": self.logical_connection_slots,
            "physical_module_slots": self.physical_module_slots,
            "physical_connection_slots": self.physical_connection_slots,
            "field_count": len(self.fields),
            "fields": [field.to_dict() for field in self.fields],
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._payload(), "manifest_sha256": self.manifest_sha256}


@dataclass(frozen=True)
class ValidatedModularBoundaryOverlay:
    packed: PackedModularStorageV1
    manifest: ModularStateBankManifest


@dataclass(frozen=True)
class ModularStateBankValidationReceipt:
    manifest_sha256: str
    field_count: int
    profile_id: str
    agent_capacity: int
    device: str
    resident_qualified: bool
    device_to_host_synchronizations: int = 0
    simulation_executed: bool = False
    checkpoint_closure_claimed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": MODULAR_STATE_BANK_RECEIPT_SCHEMA_VERSION,
            "manifest_sha256": self.manifest_sha256,
            "field_count": self.field_count,
            "profile_id": self.profile_id,
            "agent_capacity": self.agent_capacity,
            "device": self.device,
            "resident_qualified": self.resident_qualified,
            "device_to_host_synchronizations": self.device_to_host_synchronizations,
            "simulation_executed": self.simulation_executed,
            "checkpoint_closure_claimed": self.checkpoint_closure_claimed,
        }


def build_modular_state_bank_manifest(
    agent_capacity: int,
    profile_id: str = "b2-12x32-v1",
) -> ModularStateBankManifest:
    layout = build_modular_storage_layout(agent_capacity, profile_id)
    fields = tuple(
        ModularStateBankFieldContract(
            logical_name=field.name,
            projected_name=f"{MODULAR_STORAGE_PROJECTION_PREFIX}{field.name}",
            dtype_name=field.dtype_name,
            shape=field.shape,
            fill_value=field.fill_value,
        )
        for field in layout.fields
    )
    if len(fields) != MODULAR_STATE_BANK_FIELD_COUNT:
        raise ModularStateBankError(
            "modular ABI field count disagrees with the state-bank contract"
        )
    provisional = ModularStateBankManifest(
        schema_version=MODULAR_STATE_BANK_MANIFEST_SCHEMA_VERSION,
        storage_schema_version=MODULAR_STORAGE_SCHEMA_VERSION,
        projection_schema_version=MODULAR_STORAGE_PROJECTION_SCHEMA_VERSION,
        projection_prefix=MODULAR_STORAGE_PROJECTION_PREFIX,
        profile_id=layout.profile_id,
        agent_capacity=layout.agent_capacity,
        logical_module_slots=layout.logical_module_slots,
        logical_connection_slots=layout.logical_connection_slots,
        physical_module_slots=PHYSICAL_MODULE_SLOTS,
        physical_connection_slots=PHYSICAL_CONNECTION_SLOTS,
        fields=fields,
        manifest_sha256="",
    )
    return ModularStateBankManifest(
        **{
            **provisional.__dict__,
            "manifest_sha256": _canonical_sha256(provisional._payload()),
        }
    )


def _validate_binding_tensors(
    packed: PackedModularStorageV1,
    *,
    slot_to_agent_id: torch.Tensor,
    slot_tokens: torch.Tensor,
    alive_mask: torch.Tensor,
) -> None:
    capacity = packed.layout.agent_capacity
    expected = (
        ("slot_to_agent_id", slot_to_agent_id, torch.int64),
        ("slot_tokens", slot_tokens, torch.int64),
        ("alive_mask", alive_mask, torch.bool),
    )
    for name, tensor, dtype in expected:
        if (
            not isinstance(tensor, torch.Tensor)
            or tensor.device.type != "cpu"
            or tensor.dtype != dtype
            or tuple(tensor.shape) != (capacity,)
            or not tensor.is_contiguous()
        ):
            raise ModularStateBankError(f"invalid CPU modular binding tensor: {name}")
    occupied = packed.tensors["agent_occupied"]
    if not torch.equal(occupied, alive_mask):
        raise ModularStateBankError("modular occupancy disagrees with alive_mask")
    if not torch.equal(packed.tensors["agent_id"], slot_to_agent_id):
        raise ModularStateBankError("modular agent_id disagrees with slot_to_agent_id")
    if not torch.equal(
        packed.tensors["agent_slot_token"][occupied],
        slot_tokens[occupied],
    ):
        raise ModularStateBankError("modular agent_slot_token disagrees with slot_tokens")


def validate_modular_cpu_boundary_projection(
    metadata: Mapping[str, Any],
    tensors: Mapping[str, Any],
    *,
    agent_capacity: int,
) -> ValidatedModularBoundaryOverlay | None:
    """Fully decode the optional CPU overlay and bind it to owner slots."""

    try:
        packed = modular_storage_from_projection(metadata, tensors)
    except ModularStorageError as exc:
        raise ModularStateBankError(f"invalid modular CPU projection: {exc}") from exc
    if packed is None:
        return None
    if packed.layout.agent_capacity != agent_capacity:
        raise ModularStateBankError(
            "modular agent capacity disagrees with CPU owner capacity"
        )
    _validate_binding_tensors(
        packed,
        slot_to_agent_id=tensors.get("slot_to_agent_id"),
        slot_tokens=tensors.get("slot_tokens"),
        alive_mask=tensors.get("alive_mask"),
    )
    return ValidatedModularBoundaryOverlay(
        packed=packed,
        manifest=build_modular_state_bank_manifest(
            packed.layout.agent_capacity,
            packed.layout.profile_id,
        ),
    )


def _device_binding_assertions(
    resident_tensors: Mapping[str, torch.Tensor],
) -> None:
    prefix = MODULAR_STORAGE_PROJECTION_PREFIX
    occupied = resident_tensors[f"{prefix}agent_occupied"]
    agent_id = resident_tensors[f"{prefix}agent_id"]
    slot_token = resident_tensors[f"{prefix}agent_slot_token"]
    try:
        torch._assert_async(
            torch.all(torch.eq(occupied, resident_tensors["alive_mask"])),
            "resident modular occupancy disagrees with alive_mask",
        )
        torch._assert_async(
            torch.all(torch.eq(agent_id, resident_tensors["slot_to_agent_id"])),
            "resident modular agent_id disagrees with slot_to_agent_id",
        )
        torch._assert_async(
            torch.all(
                torch.logical_or(
                    torch.logical_not(occupied),
                    torch.eq(slot_token, resident_tensors["slot_tokens"]),
                )
            ),
            "resident modular agent_slot_token disagrees with slot_tokens",
        )
    except RuntimeError as exc:
        raise ModularStateBankError(str(exc)) from exc


def _matches_target_device(actual: torch.device, target: torch.device) -> bool:
    return actual.type == target.type and (
        target.index is None or actual.index == target.index
    )


def validate_modular_resident_state_bank(
    overlay: ValidatedModularBoundaryOverlay,
    resident_tensors: Mapping[str, torch.Tensor],
    *,
    device: str | torch.device,
) -> ModularStateBankValidationReceipt:
    """Validate resident storage and enqueue binding assertions without D2H."""

    if not isinstance(overlay, ValidatedModularBoundaryOverlay):
        raise ModularStateBankError("validated modular overlay is required")
    target = torch.device(device)
    manifest = overlay.manifest
    expected_manifest = build_modular_state_bank_manifest(
        manifest.agent_capacity,
        manifest.profile_id,
    )
    if manifest != expected_manifest:
        raise ModularStateBankError("modular resident manifest is not exact")
    expected_names = set(manifest.fields_by_projected_name)
    actual_names = {
        str(name)
        for name in resident_tensors
        if str(name).startswith(MODULAR_STORAGE_PROJECTION_PREFIX)
    }
    if actual_names != expected_names:
        raise ModularStateBankError(
            "resident modular tensor field set mismatch "
            f"missing={sorted(expected_names-actual_names)!r};"
            f"extra={sorted(actual_names-expected_names)!r}"
        )
    for field in manifest.fields:
        tensor = resident_tensors[field.projected_name]
        if (
            not isinstance(tensor, torch.Tensor)
            or not _matches_target_device(tensor.device, target)
            or tensor.dtype != field.torch_dtype
            or tuple(tensor.shape) != field.shape
            or not tensor.is_contiguous()
            or tensor.requires_grad
        ):
            raise ModularStateBankError(
                f"invalid resident modular tensor: {field.projected_name}"
            )
    capacity = manifest.agent_capacity
    for name, dtype in (
        ("slot_to_agent_id", torch.int64),
        ("slot_tokens", torch.int64),
        ("alive_mask", torch.bool),
    ):
        tensor = resident_tensors.get(name)
        if (
            not isinstance(tensor, torch.Tensor)
            or not _matches_target_device(tensor.device, target)
            or tensor.dtype != dtype
            or tuple(tensor.shape) != (capacity,)
            or not tensor.is_contiguous()
        ):
            raise ModularStateBankError(f"invalid resident binding tensor: {name}")
    _device_binding_assertions(resident_tensors)
    return ModularStateBankValidationReceipt(
        manifest_sha256=manifest.manifest_sha256,
        field_count=len(manifest.fields),
        profile_id=manifest.profile_id,
        agent_capacity=manifest.agent_capacity,
        device=str(target),
        resident_qualified=True,
    )


__all__ = [
    "MODULAR_STATE_BANK_FIELD_COUNT",
    "MODULAR_STATE_BANK_MANIFEST_SCHEMA_VERSION",
    "MODULAR_STATE_BANK_RECEIPT_SCHEMA_VERSION",
    "ModularStateBankError",
    "ModularStateBankFieldContract",
    "ModularStateBankManifest",
    "ModularStateBankValidationReceipt",
    "ValidatedModularBoundaryOverlay",
    "build_modular_state_bank_manifest",
    "validate_modular_cpu_boundary_projection",
    "validate_modular_resident_state_bank",
]
