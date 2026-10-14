"""Strict CPU adapter for the prebuilt modular recurrent oracle artifact."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import struct
from typing import Final

import torch

from evolution.structural_gated_recurrent_v1 import (
    GATED_NUMERIC_FIELDS,
    GATED_PARAMETER_BOUNDS,
)
from perf.cuda_architecture_b_modular_state_abi import (
    PackedModularStorageV1,
    validate_modular_storage,
)
from engine.perf.recurrent_native.modular_recurrent_cpu_artifact_v1 import (
    ACTION_WIDTH,
    OBSERVATION_WIDTH,
    ROW_PLANE_FIELDS,
    ConstOwnerViewV1,
    MutationV1,
    OwnerViewV1,
    RowV1,
    TickReceiptV1,
    load_prebuilt_modular_recurrent_cpu_oracle,
)


production_qualified: Final[bool] = False
publication_qualified: Final[bool] = False
runtime_build: Final[bool] = False

_PROFILE_CODES: Final[dict[str, int]] = {
    "b2-12x32-v1": 0,
    "b2-16x48-v1": 1,
}
_PRIVATE_EMPTY_TOKEN_BASE: Final[int] = torch.iinfo(torch.int64).min
_MUTATE_GATED_PARAMETER: Final[int] = 1
_GATED_OPERATOR_CODE: Final[int] = 2
_INT64_MIN: Final[int] = torch.iinfo(torch.int64).min
_INT64_MAX: Final[int] = torch.iinfo(torch.int64).max


class ModularRecurrentCpuOracleAdapterError(RuntimeError):
    """The packed CPU oracle adapter rejected an input or C ABI result."""


@dataclass(frozen=True)
class ModularRecurrentCpuOracleInputIdentityV1:
    profile_id: str
    agent_capacity: int
    layout_manifest_sha256: str
    input_sidecar_sha256: str
    input_packed_sha256: str
    artifact_path: Path
    artifact_sha256: str


@dataclass(frozen=True)
class ModularRecurrentCpuOracleTickReceiptV1:
    tick_index: int
    committed_rows: int
    structured_agents: int
    neutral_agents: int
    executing_modules: int
    changing_state_modules: int
    rng_draws: int
    production_qualified: bool
    publication_qualified: bool


@dataclass(frozen=True)
class ModularRecurrentCpuOracleAdapterResultV1:
    """Runtime tensors are clones, not a newly digested packed authority."""

    updated_tensors: dict[str, torch.Tensor]
    action_residuals: torch.Tensor
    tick_receipts: tuple[ModularRecurrentCpuOracleTickReceiptV1, ...]
    input_identity: ModularRecurrentCpuOracleInputIdentityV1
    production_qualified: bool = False
    publication_qualified: bool = False
    runtime_build: bool = False


@dataclass(frozen=True)
class ModularRecurrentCpuOracleNewbornReceiptV1:
    destination_row: int
    expected_destination_token: int
    parent_a_row: int
    expected_parent_a_token: int
    parent_b_row: int
    expected_parent_b_token: int
    donor_draw_float32: float
    donor_parent_row: int
    child_agent_id: int
    publication_token: int
    gated_module_slot: int
    gated_field: str
    gated_field_index: int
    value_before_float32: float
    replacement_value_float32: float
    additive_delta_float32: float
    source_packed_sha256: str
    child_tensors_sha256: str
    artifact_sha256: str
    production_qualified: bool = False
    publication_qualified: bool = False


@dataclass(frozen=True)
class ModularRecurrentCpuOracleNewbornResultV1:
    """Exact staged tensor output for one shape-preserving newborn transaction."""

    updated_tensors: dict[str, torch.Tensor]
    child_tensors: dict[str, torch.Tensor]
    receipt: ModularRecurrentCpuOracleNewbornReceiptV1
    input_identity: ModularRecurrentCpuOracleInputIdentityV1
    production_qualified: bool = False
    publication_qualified: bool = False
    runtime_build: bool = False


@dataclass(frozen=True)
class _OwnerArrays:
    agent_ids: torch.Tensor
    slot_tokens: torch.Tensor
    alive_mask: torch.Tensor
    view: ConstOwnerViewV1


def _tensor_pointer(
    tensor: torch.Tensor,
    ctype: type[ctypes._SimpleCData],  # type: ignore[name-defined]
) -> ctypes._Pointer:  # type: ignore[name-defined]
    return ctypes.cast(tensor.data_ptr(), ctypes.POINTER(ctype))


def _validate_row_plane_layout(packed: PackedModularStorageV1) -> None:
    field_names = tuple(field.name for field in packed.layout.fields)
    if field_names != ROW_PLANE_FIELDS or len(ROW_PLANE_FIELDS) != 57:
        raise ModularRecurrentCpuOracleAdapterError(
            "packed storage does not match the exact RowV1 plane order"
        )
    row_count = packed.layout.agent_capacity
    for name, ctype in RowV1._fields_:
        tensor = packed.tensors[name]
        row_element_count = tensor.numel() // row_count
        if row_element_count * tensor.element_size() != ctypes.sizeof(ctype):
            raise ModularRecurrentCpuOracleAdapterError(
                f"packed plane byte extent does not match RowV1: {name}"
            )


def _rows_from_packed(packed: PackedModularStorageV1) -> ctypes.Array[RowV1]:
    row_count = packed.layout.agent_capacity
    rows = (RowV1 * row_count)()
    for name, ctype in RowV1._fields_:
        tensor = packed.tensors[name]
        row_bytes = ctypes.sizeof(ctype)
        field_offset = getattr(RowV1, name).offset
        source = tensor.data_ptr()
        for row_index in range(row_count):
            ctypes.memmove(
                ctypes.addressof(rows[row_index]) + field_offset,
                source + row_index * row_bytes,
                row_bytes,
            )
    return rows


def _tensors_from_rows(
    rows: ctypes.Array[RowV1], packed: PackedModularStorageV1
) -> dict[str, torch.Tensor]:
    row_count = packed.layout.agent_capacity
    result = {name: tensor.clone() for name, tensor in packed.tensors.items()}
    for name, ctype in RowV1._fields_:
        tensor = result[name]
        row_bytes = ctypes.sizeof(ctype)
        field_offset = getattr(RowV1, name).offset
        destination = tensor.data_ptr()
        for row_index in range(row_count):
            ctypes.memmove(
                destination + row_index * row_bytes,
                ctypes.addressof(rows[row_index]) + field_offset,
                row_bytes,
            )
    return result


def _owner_arrays(packed: PackedModularStorageV1) -> _OwnerArrays:
    row_count = packed.layout.agent_capacity
    agent_ids = packed.tensors["agent_id"].clone()
    slot_tokens = packed.tensors["agent_slot_token"].clone()
    alive_mask = packed.tensors["agent_occupied"].clone()
    empty_rows = ~alive_mask
    private_tokens = torch.arange(row_count, dtype=torch.int64)
    private_tokens.add_(_PRIVATE_EMPTY_TOKEN_BASE)
    slot_tokens[empty_rows] = private_tokens[empty_rows]
    view = ConstOwnerViewV1(
        _tensor_pointer(agent_ids, ctypes.c_int64),
        _tensor_pointer(slot_tokens, ctypes.c_int64),
        _tensor_pointer(alive_mask, ctypes.c_uint8),
        row_count,
    )
    return _OwnerArrays(agent_ids, slot_tokens, alive_mask, view)


def _mutable_owner_view(owners: _OwnerArrays) -> OwnerViewV1:
    return OwnerViewV1(
        _tensor_pointer(owners.agent_ids, ctypes.c_int64),
        _tensor_pointer(owners.slot_tokens, ctypes.c_int64),
        _tensor_pointer(owners.alive_mask, ctypes.c_uint8),
        owners.agent_ids.numel(),
    )


def _exact_int64(value: object, *, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum or value > _INT64_MAX:
        raise ModularRecurrentCpuOracleAdapterError(
            f"{name} must be an integer in [{minimum}, {_INT64_MAX}]"
        )
    return value


def _row_index(value: object, *, name: str, row_count: int) -> int:
    if type(value) is not int or value < 0 or value >= row_count:
        raise ModularRecurrentCpuOracleAdapterError(
            f"{name} must be an integer in [0,{row_count})"
        )
    return value


def _float32(value: object, *, name: str) -> float:
    if isinstance(value, bool):
        raise ModularRecurrentCpuOracleAdapterError(f"{name} must be finite")
    try:
        raw = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ModularRecurrentCpuOracleAdapterError(
            f"{name} must be finite"
        ) from exc
    converted = ctypes.c_float(raw).value
    if not math.isfinite(raw) or not math.isfinite(converted):
        raise ModularRecurrentCpuOracleAdapterError(f"{name} must be finite")
    return converted


def _float32_bits(value: float) -> bytes:
    return struct.pack("<f", value)


def _child_tensor_sha256(child_tensors: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    digest.update(b"eiel_modular_recurrent_child_tensors_v1\0")
    for name in ROW_PLANE_FIELDS:
        tensor = child_tensors[name].detach().contiguous()
        encoded_name = name.encode("ascii")
        encoded_dtype = str(tensor.dtype).removeprefix("torch.").encode("ascii")
        digest.update(struct.pack("<I", len(encoded_name)))
        digest.update(encoded_name)
        digest.update(struct.pack("<I", len(encoded_dtype)))
        digest.update(encoded_dtype)
        digest.update(struct.pack("<I", tensor.ndim))
        for extent in tensor.shape:
            digest.update(struct.pack("<q", extent))
        raw = tensor.numpy().tobytes()
        digest.update(struct.pack("<q", len(raw)))
        digest.update(raw)
    return digest.hexdigest()


def _input_identity(
    packed: PackedModularStorageV1,
    artifact_path: Path,
    artifact_sha256: str,
) -> ModularRecurrentCpuOracleInputIdentityV1:
    return ModularRecurrentCpuOracleInputIdentityV1(
        profile_id=packed.layout.profile_id,
        agent_capacity=packed.layout.agent_capacity,
        layout_manifest_sha256=packed.layout.manifest_sha256,
        input_sidecar_sha256=packed.sidecar_sha256,
        input_packed_sha256=packed.packed_sha256,
        artifact_path=artifact_path,
        artifact_sha256=artifact_sha256,
    )


def _validate_tensor(
    tensor: object,
    *,
    name: str,
    dtype: torch.dtype,
    shape: tuple[int, ...],
) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor):
        raise ModularRecurrentCpuOracleAdapterError(f"{name} must be a tensor")
    if (
        tensor.device.type != "cpu"
        or tensor.dtype != dtype
        or tuple(tensor.shape) != shape
        or not tensor.is_contiguous()
        or tensor.requires_grad
    ):
        raise ModularRecurrentCpuOracleAdapterError(
            f"{name} must be contiguous CPU {dtype} with shape {shape}"
        )
    return tensor


def _tick_inputs(
    observations: object,
    fatigue: object,
    stage_indices: object,
    *,
    row_count: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not isinstance(observations, torch.Tensor):
        raise ModularRecurrentCpuOracleAdapterError("observations must be a tensor")
    if observations.ndim == 2:
        observations = _validate_tensor(
            observations,
            name="observations",
            dtype=torch.float32,
            shape=(row_count, OBSERVATION_WIDTH),
        ).unsqueeze(0)
        fatigue = _validate_tensor(
            fatigue,
            name="fatigue",
            dtype=torch.float32,
            shape=(row_count,),
        ).unsqueeze(0)
        stage_indices = _validate_tensor(
            stage_indices,
            name="stage_indices",
            dtype=torch.int64,
            shape=(row_count,),
        ).unsqueeze(0)
    elif observations.ndim == 3:
        tick_count = int(observations.shape[0])
        if tick_count <= 0:
            raise ModularRecurrentCpuOracleAdapterError("at least one tick is required")
        observations = _validate_tensor(
            observations,
            name="observations",
            dtype=torch.float32,
            shape=(tick_count, row_count, OBSERVATION_WIDTH),
        )
        fatigue = _validate_tensor(
            fatigue,
            name="fatigue",
            dtype=torch.float32,
            shape=(tick_count, row_count),
        )
        stage_indices = _validate_tensor(
            stage_indices,
            name="stage_indices",
            dtype=torch.int64,
            shape=(tick_count, row_count),
        )
    else:
        raise ModularRecurrentCpuOracleAdapterError(
            "observations must have shape [rows,27] or [ticks,rows,27]"
        )
    if not bool(torch.isfinite(observations).all().item()):
        raise ModularRecurrentCpuOracleAdapterError("observations must be finite")
    if not bool(torch.isfinite(fatigue).all().item()) or bool(
        torch.any((fatigue < 0.0) | (fatigue > 1.0)).item()
    ):
        raise ModularRecurrentCpuOracleAdapterError(
            "fatigue must be finite and within [0,1]"
        )
    if bool(torch.any((stage_indices < 0) | (stage_indices > 2)).item()):
        raise ModularRecurrentCpuOracleAdapterError(
            "stage_indices must be within [0,2]"
        )
    return observations, fatigue, stage_indices


def execute_modular_recurrent_cpu_oracle_newborn_v1(
    packed: PackedModularStorageV1,
    artifact_path: str | Path,
    *,
    destination_row: int,
    expected_destination_token: int,
    parent_a_row: int,
    expected_parent_a_token: int,
    parent_b_row: int,
    expected_parent_b_token: int,
    donor_draw: float,
    child_agent_id: int,
    publication_token: int,
    gated_module_slot: int,
    gated_field: str,
    gated_replacement_value: float,
) -> ModularRecurrentCpuOracleNewbornResultV1:
    """Stage and publish one exact gated-parameter newborn transaction."""

    validate_modular_storage(packed)
    _validate_row_plane_layout(packed)
    profile = _PROFILE_CODES.get(packed.layout.profile_id)
    if profile is None:
        raise ModularRecurrentCpuOracleAdapterError(
            "packed profile is unsupported by the CPU oracle"
        )
    row_count = packed.layout.agent_capacity
    destination_row = _row_index(
        destination_row, name="destination_row", row_count=row_count
    )
    parent_a_row = _row_index(
        parent_a_row, name="parent_a_row", row_count=row_count
    )
    parent_b_row = _row_index(
        parent_b_row, name="parent_b_row", row_count=row_count
    )
    if bool(packed.tensors["agent_occupied"][destination_row].item()):
        raise ModularRecurrentCpuOracleAdapterError("destination_row must be empty")
    for name, row, expected_token in (
        ("parent_a", parent_a_row, expected_parent_a_token),
        ("parent_b", parent_b_row, expected_parent_b_token),
    ):
        if not bool(packed.tensors["agent_occupied"][row].item()):
            raise ModularRecurrentCpuOracleAdapterError(f"{name}_row must be occupied")
        expected = _exact_int64(expected_token, name=f"expected_{name}_token")
        actual = int(packed.tensors["agent_slot_token"][row].item())
        if expected != actual:
            raise ModularRecurrentCpuOracleAdapterError(
                f"expected_{name}_token does not match packed authority"
            )
    expected_destination_token = _exact_int64(
        expected_destination_token,
        name="expected_destination_token",
        minimum=_INT64_MIN,
    )
    child_agent_id = _exact_int64(child_agent_id, name="child_agent_id")
    publication_token = _exact_int64(
        publication_token, name="publication_token"
    )
    if bool(torch.any(packed.tensors["agent_id"] == child_agent_id).item()):
        raise ModularRecurrentCpuOracleAdapterError(
            "child_agent_id already exists in packed authority"
        )
    donor_draw_float32 = _float32(donor_draw, name="donor_draw")
    if not 0.0 <= donor_draw_float32 < 1.0:
        raise ModularRecurrentCpuOracleAdapterError(
            "donor_draw must convert to float32 within [0,1)"
        )
    donor_parent_row = parent_a_row if donor_draw_float32 < 0.5 else parent_b_row
    if (
        type(gated_module_slot) is not int
        or gated_module_slot < 0
        or gated_module_slot >= packed.layout.physical_module_slots
    ):
        raise ModularRecurrentCpuOracleAdapterError(
            "gated_module_slot is outside physical module capacity"
        )
    if not bool(
        packed.tensors["module_occupied"][
            donor_parent_row, gated_module_slot
        ].item()
    ) or int(
        packed.tensors["operator_code"][donor_parent_row, gated_module_slot].item()
    ) != _GATED_OPERATOR_CODE:
        raise ModularRecurrentCpuOracleAdapterError(
            "selected donor module is not an occupied gated module"
        )
    if type(gated_field) is not str or gated_field not in GATED_NUMERIC_FIELDS:
        raise ModularRecurrentCpuOracleAdapterError(
            "gated_field is not a canonical gated numeric field"
        )
    gated_field_index = GATED_NUMERIC_FIELDS.index(gated_field)
    value_before = float(
        packed.tensors["gated_parameters"][
            donor_parent_row, gated_module_slot, gated_field_index
        ].item()
    )
    replacement = _float32(
        gated_replacement_value, name="gated_replacement_value"
    )
    lower, upper = GATED_PARAMETER_BOUNDS[gated_field]
    if not float(lower) <= replacement <= float(upper):
        raise ModularRecurrentCpuOracleAdapterError(
            "gated_replacement_value is outside the field bounds"
        )
    additive_delta = ctypes.c_float(replacement - value_before).value
    projected = ctypes.c_float(value_before + additive_delta).value
    if _float32_bits(projected) != _float32_bits(replacement):
        raise ModularRecurrentCpuOracleAdapterError(
            "gated replacement is not exactly representable by one float32 delta"
        )

    artifact = load_prebuilt_modular_recurrent_cpu_oracle(artifact_path)
    rows = _rows_from_packed(packed)
    owners = _owner_arrays(packed)
    owners.slot_tokens[destination_row] = expected_destination_token
    mutable_owners = _mutable_owner_view(owners)
    mutation = MutationV1(
        _MUTATE_GATED_PARAMETER,
        gated_module_slot,
        gated_field_index,
        additive_delta,
    )
    status = int(
        artifact.library.eiel_modular_recurrent_newborn_inherit_v1(
            rows,
            row_count,
            profile,
            ctypes.byref(mutable_owners),
            destination_row,
            expected_destination_token,
            parent_a_row,
            expected_parent_a_token,
            parent_b_row,
            expected_parent_b_token,
            donor_draw_float32,
            child_agent_id,
            publication_token,
            ctypes.byref(mutation),
            1,
        )
    )
    if status != 0:
        raise ModularRecurrentCpuOracleAdapterError(
            f"C oracle newborn transaction failed: status={status}"
        )
    validation_status = int(
        artifact.library.eiel_modular_recurrent_validate_v1(
            rows,
            row_count,
            profile,
            ctypes.byref(owners.view),
        )
    )
    if validation_status != 0:
        raise ModularRecurrentCpuOracleAdapterError(
            "C oracle published a newborn bank that fails validation: "
            f"status={validation_status}"
        )
    if (
        owners.agent_ids[destination_row].item() != child_agent_id
        or owners.slot_tokens[destination_row].item() != publication_token
        or not bool(owners.alive_mask[destination_row].item())
    ):
        raise ModularRecurrentCpuOracleAdapterError(
            "C oracle newborn owner publication drifted"
        )

    updated_tensors = _tensors_from_rows(rows, packed)
    for name in ROW_PLANE_FIELDS:
        before = packed.tensors[name]
        after = updated_tensors[name]
        if destination_row > 0 and not torch.equal(
            before[:destination_row], after[:destination_row]
        ):
            raise ModularRecurrentCpuOracleAdapterError(
                f"C oracle changed a non-destination row: {name}"
            )
        if destination_row + 1 < row_count and not torch.equal(
            before[destination_row + 1 :], after[destination_row + 1 :]
        ):
            raise ModularRecurrentCpuOracleAdapterError(
                f"C oracle changed a non-destination row: {name}"
            )
    child_tensors = {
        name: updated_tensors[name][destination_row].clone()
        for name in ROW_PLANE_FIELDS
    }
    actual_replacement = float(
        child_tensors["gated_parameters"][
            gated_module_slot, gated_field_index
        ].item()
    )
    if _float32_bits(actual_replacement) != _float32_bits(replacement):
        raise ModularRecurrentCpuOracleAdapterError(
            "C oracle gated replacement does not match the staged transaction"
        )
    child_tensors_sha256 = _child_tensor_sha256(child_tensors)
    identity = _input_identity(
        packed, artifact.artifact_path, artifact.artifact_sha256
    )
    receipt = ModularRecurrentCpuOracleNewbornReceiptV1(
        destination_row=destination_row,
        expected_destination_token=expected_destination_token,
        parent_a_row=parent_a_row,
        expected_parent_a_token=expected_parent_a_token,
        parent_b_row=parent_b_row,
        expected_parent_b_token=expected_parent_b_token,
        donor_draw_float32=donor_draw_float32,
        donor_parent_row=donor_parent_row,
        child_agent_id=child_agent_id,
        publication_token=publication_token,
        gated_module_slot=gated_module_slot,
        gated_field=gated_field,
        gated_field_index=gated_field_index,
        value_before_float32=value_before,
        replacement_value_float32=replacement,
        additive_delta_float32=additive_delta,
        source_packed_sha256=packed.packed_sha256,
        child_tensors_sha256=child_tensors_sha256,
        artifact_sha256=artifact.artifact_sha256,
    )
    return ModularRecurrentCpuOracleNewbornResultV1(
        updated_tensors=updated_tensors,
        child_tensors=child_tensors,
        receipt=receipt,
        input_identity=identity,
    )


def execute_modular_recurrent_cpu_oracle_v1(
    packed: PackedModularStorageV1,
    artifact_path: str | Path,
    observations: torch.Tensor,
    fatigue: torch.Tensor,
    stage_indices: torch.Tensor,
) -> ModularRecurrentCpuOracleAdapterResultV1:
    """Execute one or more CPU oracle ticks without mutating packed authority."""

    validate_modular_storage(packed)
    _validate_row_plane_layout(packed)
    profile = _PROFILE_CODES.get(packed.layout.profile_id)
    if profile is None:
        raise ModularRecurrentCpuOracleAdapterError(
            "packed profile is unsupported by the CPU oracle"
        )
    row_count = packed.layout.agent_capacity
    observations, fatigue, stage_indices = _tick_inputs(
        observations,
        fatigue,
        stage_indices,
        row_count=row_count,
    )

    artifact = load_prebuilt_modular_recurrent_cpu_oracle(artifact_path)
    rows = _rows_from_packed(packed)
    owners = _owner_arrays(packed)
    validation_status = int(
        artifact.library.eiel_modular_recurrent_validate_v1(
            rows,
            row_count,
            profile,
            ctypes.byref(owners.view),
        )
    )
    if validation_status != 0:
        raise ModularRecurrentCpuOracleAdapterError(
            f"C oracle rejected the packed RowV1 bank: status={validation_status}"
        )

    tick_count = int(observations.shape[0])
    action_residuals = torch.empty(
        (tick_count, row_count, ACTION_WIDTH), dtype=torch.float32
    )
    scratch_rows = (RowV1 * row_count)()
    scratch_actions = torch.empty((row_count, ACTION_WIDTH), dtype=torch.float32)
    receipts: list[ModularRecurrentCpuOracleTickReceiptV1] = []
    for tick_index in range(tick_count):
        receipt = TickReceiptV1()
        status = int(
            artifact.library.eiel_modular_recurrent_tick_v1(
                rows,
                row_count,
                profile,
                ctypes.byref(owners.view),
                _tensor_pointer(observations[tick_index], ctypes.c_float),
                observations[tick_index].numel(),
                _tensor_pointer(fatigue[tick_index], ctypes.c_float),
                fatigue[tick_index].numel(),
                _tensor_pointer(stage_indices[tick_index], ctypes.c_int64),
                stage_indices[tick_index].numel(),
                scratch_rows,
                row_count,
                _tensor_pointer(scratch_actions, ctypes.c_float),
                scratch_actions.numel(),
                _tensor_pointer(action_residuals[tick_index], ctypes.c_float),
                action_residuals[tick_index].numel(),
                ctypes.byref(receipt),
            )
        )
        if status != 0:
            raise ModularRecurrentCpuOracleAdapterError(
                f"C oracle tick {tick_index} failed: status={status}"
            )
        if (
            receipt.committed_rows != row_count
            or receipt.rng_draws != 0
            or receipt.production_routing_qualified != 0
            or receipt.publication_qualified != 0
        ):
            raise ModularRecurrentCpuOracleAdapterError(
                f"C oracle tick {tick_index} receipt crossed its qualification boundary"
            )
        receipts.append(
            ModularRecurrentCpuOracleTickReceiptV1(
                tick_index=tick_index,
                committed_rows=int(receipt.committed_rows),
                structured_agents=int(receipt.structured_agents),
                neutral_agents=int(receipt.neutral_agents),
                executing_modules=int(receipt.executing_modules),
                changing_state_modules=int(receipt.changing_state_modules),
                rng_draws=int(receipt.rng_draws),
                production_qualified=False,
                publication_qualified=False,
            )
        )

    identity = _input_identity(
        packed, artifact.artifact_path, artifact.artifact_sha256
    )
    return ModularRecurrentCpuOracleAdapterResultV1(
        updated_tensors=_tensors_from_rows(rows, packed),
        action_residuals=action_residuals,
        tick_receipts=tuple(receipts),
        input_identity=identity,
    )


__all__ = [
    "ModularRecurrentCpuOracleAdapterError",
    "ModularRecurrentCpuOracleAdapterResultV1",
    "ModularRecurrentCpuOracleInputIdentityV1",
    "ModularRecurrentCpuOracleNewbornReceiptV1",
    "ModularRecurrentCpuOracleNewbornResultV1",
    "ModularRecurrentCpuOracleTickReceiptV1",
    "execute_modular_recurrent_cpu_oracle_newborn_v1",
    "execute_modular_recurrent_cpu_oracle_v1",
    "production_qualified",
    "publication_qualified",
    "runtime_build",
]
