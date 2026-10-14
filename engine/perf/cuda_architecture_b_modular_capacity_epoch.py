from __future__ import annotations

"""Integrity-only Phase E modular capacity-epoch migration overlay.

The transition changes the logical B2 profile from 12/32 to 16/48 while the
physical Architecture B planes remain A x 16 modules and A x 48 connections.
This module performs no simulation, campaign, policy, or RNG execution.
"""

from copy import deepcopy
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
import re
from typing import Any, Final, Mapping

import torch

from agents.modular_genome_v1 import B2_CAPACITY_PROFILES
from perf.cuda_architecture_b_modular_state_abi import (
    MODULAR_STORAGE_METADATA_KEY,
    MODULAR_STORAGE_PROJECTION_PREFIX,
    PackedModularStorageV1,
    modular_storage_from_projection,
    modular_storage_projection_payload,
    pack_modular_storage,
)
from perf.cuda_architecture_b_modular_state_bank import (
    ValidatedModularBoundaryOverlay,
    build_modular_state_bank_manifest,
    validate_modular_cpu_boundary_projection,
)
from simulation.capacity_epoch import (
    EXPANDED_CAPACITY_PROFILE_ID,
    LEGACY_CAPACITY_PROFILE_ID,
    SAFE_BOUNDARY_KIND,
)


MODULAR_CAPACITY_EPOCH_AUTHORIZATION_SCHEMA_VERSION: Final[str] = (
    "eiel_cuda_architecture_b_modular_capacity_epoch_authorization_v1"
)
MODULAR_CAPACITY_EPOCH_RECEIPT_SCHEMA_VERSION: Final[str] = (
    "eiel_cuda_architecture_b_modular_capacity_epoch_receipt_v1"
)
MODULAR_CAPACITY_EPOCH_MIGRATION_SCHEMA_VERSION: Final[str] = (
    "eiel_cuda_architecture_b_modular_capacity_epoch_migration_v1"
)

_AUTHORIZATION_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "authorization_receipt_id",
        "authorization_receipt_sha256",
        "authorized",
        "authorized_generation",
        "next_generation",
        "source_profile_id",
        "target_profile_id",
        "safe_boundary",
        "rollback_checkpoint_digest",
        "rollback_checkpoint_verified",
    }
)
_SHA256_RE: Final[re.Pattern[str]] = re.compile(r"^[0-9a-f]{64}$")


class ModularCapacityEpochError(RuntimeError):
    """The authorized neutral capacity transition is invalid or contradictory."""


@dataclass(frozen=True)
class ModularCapacityEpochReceipt:
    authorization_receipt_id: str
    authorization_receipt_sha256: str
    rollback_checkpoint_digest: str
    source_profile_id: str
    target_profile_id: str
    authorized_generation: int
    next_generation: int
    safe_boundary: str
    agent_capacity: int
    migrated_agent_count: int
    source_logical_module_slots: int
    target_logical_module_slots: int
    source_logical_connection_slots: int
    target_logical_connection_slots: int
    physical_module_slots: int
    physical_connection_slots: int
    source_manifest_sha256: str
    target_manifest_sha256: str
    source_packed_sha256: str
    target_packed_sha256: str
    physical_tensor_sha256: str
    rng_state_before_sha256: str
    rng_state_after_sha256: str
    rng_tensors_before_sha256: str
    rng_tensors_after_sha256: str
    rng_byte_delta: int = 0
    migration_rng_draws: int = 0
    physical_tensor_resize_count: int = 0
    old_occupied_subspace_exact: bool = True
    newly_legal_tail_canonical_empty: bool = True
    cold_sidecar_profile_switched: bool = True
    identity_topology_order_state_cost_exact: bool = True
    simulation_executed: bool = False
    campaign_executed: bool = False
    native_execution_performed: bool = False
    intelligence_or_emergence_claimed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": MODULAR_CAPACITY_EPOCH_RECEIPT_SCHEMA_VERSION,
            **asdict(self),
            "scope": "mechanical_storage_validation_only",
        }


@dataclass(frozen=True)
class ModularCapacityEpochMigration:
    metadata: dict[str, Any]
    tensors: dict[str, torch.Tensor]
    receipt: ModularCapacityEpochReceipt

    def projection(self) -> dict[str, Any]:
        return {"metadata": self.metadata, "tensors": self.tensors}


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return sha256(_canonical_bytes(value)).hexdigest()


def capacity_epoch_authorization_receipt_sha256(
    authorization: Mapping[str, Any],
) -> str:
    """Digest every authorization field except the self-authenticating digest."""

    if not isinstance(authorization, Mapping):
        raise TypeError("authorization must be a mapping")
    payload = dict(authorization)
    payload.pop("authorization_receipt_sha256", None)
    return _canonical_sha256(payload)


def _validate_authorization(authorization: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(authorization, Mapping):
        raise ModularCapacityEpochError("capacity migration authorization is missing")
    payload = dict(authorization)
    if set(payload) != set(_AUTHORIZATION_FIELDS):
        raise ModularCapacityEpochError(
            "capacity migration authorization receipt fields mismatch"
        )
    if payload.get("schema_version") != MODULAR_CAPACITY_EPOCH_AUTHORIZATION_SCHEMA_VERSION:
        raise ModularCapacityEpochError("capacity migration authorization schema mismatch")
    if payload.get("authorized") is not True:
        raise ModularCapacityEpochError("capacity migration is not authorized")
    if payload.get("rollback_checkpoint_verified") is not True:
        raise ModularCapacityEpochError("rollback checkpoint is not verified")
    receipt_id = payload.get("authorization_receipt_id")
    if type(receipt_id) is not str or not receipt_id.strip():
        raise ModularCapacityEpochError("authorization receipt id is missing")
    for field in ("rollback_checkpoint_digest", "authorization_receipt_sha256"):
        if type(payload.get(field)) is not str or _SHA256_RE.fullmatch(payload[field]) is None:
            raise ModularCapacityEpochError(f"{field} must be an exact SHA-256")
    if payload["authorization_receipt_sha256"] != capacity_epoch_authorization_receipt_sha256(
        payload
    ):
        raise ModularCapacityEpochError("authorization receipt digest mismatch")
    if payload.get("source_profile_id") != LEGACY_CAPACITY_PROFILE_ID:
        raise ModularCapacityEpochError("unsupported or reverse source profile")
    if payload.get("target_profile_id") != EXPANDED_CAPACITY_PROFILE_ID:
        raise ModularCapacityEpochError("unsupported or reverse target profile")
    if payload.get("safe_boundary") != SAFE_BOUNDARY_KIND:
        raise ModularCapacityEpochError("capacity migration is not at the safe boundary")
    authorized_generation = payload.get("authorized_generation")
    next_generation = payload.get("next_generation")
    if (
        type(authorized_generation) is not int
        or authorized_generation < 0
        or type(next_generation) is not int
        or next_generation != authorized_generation + 1
    ):
        raise ModularCapacityEpochError("authorization generation fields mismatch")
    return payload


def _encode_rng_value(value: Any, *, path: str) -> Any:
    if isinstance(value, Mapping):
        if any(type(key) is not str for key in value):
            raise ModularCapacityEpochError(f"{path} keys must be strings")
        return {
            "kind": "mapping",
            "items": [
                [key, _encode_rng_value(value[key], path=f"{path}.{key}")]
                for key in sorted(value)
            ],
        }
    if isinstance(value, torch.Tensor):
        if (
            value.device.type != "cpu"
            or not value.is_contiguous()
            or value.requires_grad
        ):
            raise ModularCapacityEpochError(
                f"{path} RNG tensor must be contiguous CPU authority"
            )
        return {
            "kind": "tensor",
            "dtype": str(value.dtype).removeprefix("torch."),
            "shape": [int(extent) for extent in value.shape],
            "bytes_hex": value.detach().numpy().tobytes().hex(),
        }
    if isinstance(value, (list, tuple)):
        return {
            "kind": "tuple" if isinstance(value, tuple) else "list",
            "items": [
                _encode_rng_value(item, path=f"{path}[{index}]")
                for index, item in enumerate(value)
            ],
        }
    if isinstance(value, bytes):
        return {"kind": "bytes", "bytes_hex": value.hex()}
    if isinstance(value, float) and not math.isfinite(value):
        raise ModularCapacityEpochError(f"{path} contains non-finite RNG state")
    if isinstance(value, (str, int, float, bool)) or value is None:
        return {"kind": type(value).__name__, "value": value}
    raise ModularCapacityEpochError(
        f"{path} contains unsupported RNG state type={type(value).__name__}"
    )


def _rng_authority_bytes(value: Mapping[str, Any], *, path: str) -> bytes:
    if not isinstance(value, Mapping):
        raise ModularCapacityEpochError(f"{path} must be a mapping")
    return _canonical_bytes(_encode_rng_value(value, path=path))


def _validate_zero_rng_delta(
    *,
    rng_state_before: Mapping[str, Any],
    rng_state_after: Mapping[str, Any],
    rng_tensors_before: Mapping[str, Any],
    rng_tensors_after: Mapping[str, Any],
) -> tuple[str, str, str, str]:
    state_before = _rng_authority_bytes(rng_state_before, path="rng_state_before")
    state_after = _rng_authority_bytes(rng_state_after, path="rng_state_after")
    tensors_before = _rng_authority_bytes(
        rng_tensors_before, path="rng_tensors_before"
    )
    tensors_after = _rng_authority_bytes(
        rng_tensors_after, path="rng_tensors_after"
    )
    if state_before != state_after or tensors_before != tensors_after:
        raise ModularCapacityEpochError("capacity migration RNG state changed")
    return (
        sha256(state_before).hexdigest(),
        sha256(state_after).hexdigest(),
        sha256(tensors_before).hexdigest(),
        sha256(tensors_after).hexdigest(),
    )


def _validated_overlay(
    metadata: Mapping[str, Any],
    tensors: Mapping[str, Any],
    *,
    expected_profile_id: str,
) -> ValidatedModularBoundaryOverlay:
    try:
        packed = modular_storage_from_projection(metadata, tensors)
    except Exception as exc:
        raise ModularCapacityEpochError(f"invalid modular projection: {exc}") from exc
    if packed is None:
        raise ModularCapacityEpochError("modular projection is missing")
    if packed.layout.profile_id != expected_profile_id:
        raise ModularCapacityEpochError(
            f"modular profile mismatch expected={expected_profile_id} "
            f"actual={packed.layout.profile_id}"
        )
    try:
        overlay = validate_modular_cpu_boundary_projection(
            metadata,
            tensors,
            agent_capacity=packed.layout.agent_capacity,
        )
    except Exception as exc:
        raise ModularCapacityEpochError(f"invalid modular state-bank binding: {exc}") from exc
    if overlay is None:
        raise ModularCapacityEpochError("modular state-bank overlay is missing")
    return overlay


def validate_legacy_unused_tail_empty(packed: PackedModularStorageV1) -> None:
    """Reject any occupied or non-neutral physical tail forbidden by 12/32."""

    if packed.layout.profile_id != LEGACY_CAPACITY_PROFILE_ID:
        raise ModularCapacityEpochError("legacy tail validator requires 12/32 source")
    fields = packed.layout.fields
    names = [field.name for field in fields]
    module_start = names.index("module_occupied")
    module_end = names.index("development_effective_gain")
    connection_start = names.index("connection_occupied")
    connection_end = names.index("connection_execution_order")
    if bool((packed.tensors["module_count"] > 12).any().item()):
        raise ModularCapacityEpochError("occupied forbidden module tail before migration")
    if bool((packed.tensors["connection_count"] > 32).any().item()):
        raise ModularCapacityEpochError("occupied forbidden connection tail before migration")
    for field in fields[module_start : module_end + 1]:
        tail = packed.tensors[field.name][:, 12:]
        if not torch.equal(tail, torch.full_like(tail, field.fill_value)):
            raise ModularCapacityEpochError(
                f"noncanonical forbidden module tail field={field.name}"
            )
    for field in fields[connection_start : connection_end + 1]:
        tail = packed.tensors[field.name][:, 32:]
        if not torch.equal(tail, torch.full_like(tail, field.fill_value)):
            raise ModularCapacityEpochError(
                f"noncanonical forbidden connection tail field={field.name}"
            )


def _records_without_profile(
    packed: PackedModularStorageV1,
) -> tuple[dict[str, Any] | None, ...]:
    records: list[dict[str, Any] | None] = []
    for record in packed.records_sidecar:
        if record is None:
            records.append(None)
            continue
        row = deepcopy(record)
        genome = dict(row["genome"])
        genome.pop("b2_caps", None)
        row["genome"] = genome
        records.append(row)
    return tuple(records)


def _expected_target_projection(
    source_metadata: Mapping[str, Any],
    source_tensors: Mapping[str, Any],
    source_packed: PackedModularStorageV1,
) -> tuple[dict[str, Any], dict[str, torch.Tensor], PackedModularStorageV1]:
    records = deepcopy(list(source_packed.records_sidecar))
    target_caps = deepcopy(B2_CAPACITY_PROFILES[EXPANDED_CAPACITY_PROFILE_ID])
    for record in records:
        if record is not None:
            record["genome"]["b2_caps"] = deepcopy(target_caps)
    target_packed = pack_modular_storage(
        records,
        agent_capacity=source_packed.layout.agent_capacity,
        profile_id=EXPANDED_CAPACITY_PROFILE_ID,
    )
    modular_payload = modular_storage_projection_payload(target_packed)
    target_metadata = deepcopy(dict(source_metadata))
    target_metadata[MODULAR_STORAGE_METADATA_KEY] = modular_payload["metadata"][
        MODULAR_STORAGE_METADATA_KEY
    ]
    target_tensors = {
        str(name): value.detach().clone()
        for name, value in source_tensors.items()
        if not str(name).startswith(MODULAR_STORAGE_PROJECTION_PREFIX)
    }
    target_tensors.update(modular_payload["tensors"])
    return target_metadata, target_tensors, target_packed


def _tensor_plane_digest(tensors: Mapping[str, torch.Tensor]) -> str:
    digest = sha256()
    for name in sorted(tensors):
        if not name.startswith(MODULAR_STORAGE_PROJECTION_PREFIX):
            continue
        value = tensors[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("utf-8"))
        digest.update(_canonical_bytes([int(extent) for extent in value.shape]))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _require_exact_projection(
    expected_metadata: Mapping[str, Any],
    expected_tensors: Mapping[str, torch.Tensor],
    actual_metadata: Mapping[str, Any],
    actual_tensors: Mapping[str, Any],
) -> None:
    if dict(actual_metadata) != dict(expected_metadata):
        raise ModularCapacityEpochError(
            "target metadata/cold sidecar/profile switch is not exact"
        )
    if set(actual_tensors) != set(expected_tensors):
        raise ModularCapacityEpochError("target tensor field set changed")
    for name in sorted(expected_tensors):
        expected = expected_tensors[name]
        actual = actual_tensors[name]
        if (
            not isinstance(actual, torch.Tensor)
            or actual.dtype != expected.dtype
            or tuple(actual.shape) != tuple(expected.shape)
            or not torch.equal(actual, expected)
        ):
            raise ModularCapacityEpochError(
                f"target identity/state/RNG-independent tensor changed field={name}"
            )


def _validate_transition(
    *,
    source_metadata: Mapping[str, Any],
    source_tensors: Mapping[str, Any],
    target_metadata: Mapping[str, Any],
    target_tensors: Mapping[str, Any],
    authorization: Mapping[str, Any],
    rng_state_before: Mapping[str, Any],
    rng_state_after: Mapping[str, Any],
    rng_tensors_before: Mapping[str, Any],
    rng_tensors_after: Mapping[str, Any],
) -> ModularCapacityEpochReceipt:
    authorized = _validate_authorization(authorization)
    rng_digests = _validate_zero_rng_delta(
        rng_state_before=rng_state_before,
        rng_state_after=rng_state_after,
        rng_tensors_before=rng_tensors_before,
        rng_tensors_after=rng_tensors_after,
    )
    source = _validated_overlay(
        source_metadata,
        source_tensors,
        expected_profile_id=LEGACY_CAPACITY_PROFILE_ID,
    )
    validate_legacy_unused_tail_empty(source.packed)
    expected_metadata, expected_tensors, expected_packed = _expected_target_projection(
        source_metadata,
        source_tensors,
        source.packed,
    )
    _require_exact_projection(
        expected_metadata,
        expected_tensors,
        target_metadata,
        target_tensors,
    )
    target = _validated_overlay(
        target_metadata,
        target_tensors,
        expected_profile_id=EXPANDED_CAPACITY_PROFILE_ID,
    )
    if _records_without_profile(source.packed) != _records_without_profile(target.packed):
        raise ModularCapacityEpochError("target cold sidecar old subspace changed")
    for field in source.packed.layout.fields:
        if not torch.equal(source.packed.tensors[field.name], target.packed.tensors[field.name]):
            raise ModularCapacityEpochError(
                f"target occupied identity/topology/order/state/cost changed field={field.name}"
            )
    # The newly legal target tail must still be empty immediately after the
    # migration; later growth is outside this validation-only boundary.
    legacy_view = PackedModularStorageV1(
        layout=source.packed.layout,
        tensors=target.packed.tensors,
        identity_table=target.packed.identity_table,
        records_sidecar=target.packed.records_sidecar,
        sidecar_sha256=target.packed.sidecar_sha256,
        packed_sha256=target.packed.packed_sha256,
    )
    validate_legacy_unused_tail_empty(legacy_view)

    source_manifest = build_modular_state_bank_manifest(
        source.packed.layout.agent_capacity,
        LEGACY_CAPACITY_PROFILE_ID,
    )
    target_manifest = build_modular_state_bank_manifest(
        target.packed.layout.agent_capacity,
        EXPANDED_CAPACITY_PROFILE_ID,
    )
    if source_manifest.manifest_sha256 == target_manifest.manifest_sha256:
        raise ModularCapacityEpochError("capacity profile manifest did not switch")
    return ModularCapacityEpochReceipt(
        authorization_receipt_id=authorized["authorization_receipt_id"],
        authorization_receipt_sha256=authorized["authorization_receipt_sha256"],
        rollback_checkpoint_digest=authorized["rollback_checkpoint_digest"],
        source_profile_id=LEGACY_CAPACITY_PROFILE_ID,
        target_profile_id=EXPANDED_CAPACITY_PROFILE_ID,
        authorized_generation=authorized["authorized_generation"],
        next_generation=authorized["next_generation"],
        safe_boundary=SAFE_BOUNDARY_KIND,
        agent_capacity=source.packed.layout.agent_capacity,
        migrated_agent_count=sum(record is not None for record in source.packed.records_sidecar),
        source_logical_module_slots=source.packed.layout.logical_module_slots,
        target_logical_module_slots=target.packed.layout.logical_module_slots,
        source_logical_connection_slots=source.packed.layout.logical_connection_slots,
        target_logical_connection_slots=target.packed.layout.logical_connection_slots,
        physical_module_slots=source.packed.layout.physical_module_slots,
        physical_connection_slots=source.packed.layout.physical_connection_slots,
        source_manifest_sha256=source_manifest.manifest_sha256,
        target_manifest_sha256=target_manifest.manifest_sha256,
        source_packed_sha256=source.packed.packed_sha256,
        target_packed_sha256=target.packed.packed_sha256,
        physical_tensor_sha256=_tensor_plane_digest(source_tensors),
        rng_state_before_sha256=rng_digests[0],
        rng_state_after_sha256=rng_digests[1],
        rng_tensors_before_sha256=rng_digests[2],
        rng_tensors_after_sha256=rng_digests[3],
    )


def validate_modular_capacity_epoch_transition(
    *,
    source_metadata: Mapping[str, Any],
    source_tensors: Mapping[str, Any],
    target_metadata: Mapping[str, Any],
    target_tensors: Mapping[str, Any],
    authorization: Mapping[str, Any],
    rng_state_before: Mapping[str, Any],
    rng_state_after: Mapping[str, Any],
    rng_tensors_before: Mapping[str, Any],
    rng_tensors_after: Mapping[str, Any],
) -> ModularCapacityEpochReceipt:
    """Validate one externally supplied safe-boundary migration candidate."""

    return _validate_transition(
        source_metadata=source_metadata,
        source_tensors=source_tensors,
        target_metadata=target_metadata,
        target_tensors=target_tensors,
        authorization=authorization,
        rng_state_before=rng_state_before,
        rng_state_after=rng_state_after,
        rng_tensors_before=rng_tensors_before,
        rng_tensors_after=rng_tensors_after,
    )


def migrate_modular_capacity_epoch_projection(
    *,
    source_metadata: Mapping[str, Any],
    source_tensors: Mapping[str, Any],
    authorization: Mapping[str, Any],
    rng_state_before: Mapping[str, Any],
    rng_state_after: Mapping[str, Any],
    rng_tensors_before: Mapping[str, Any],
    rng_tensors_after: Mapping[str, Any],
) -> ModularCapacityEpochMigration:
    """Build and validate the exact neutral target projection without RNG draws."""

    # Authorization is checked before even decoding or cloning the source.
    _validate_authorization(authorization)
    source = _validated_overlay(
        source_metadata,
        source_tensors,
        expected_profile_id=LEGACY_CAPACITY_PROFILE_ID,
    )
    validate_legacy_unused_tail_empty(source.packed)
    target_metadata, target_tensors, _target_packed = _expected_target_projection(
        source_metadata,
        source_tensors,
        source.packed,
    )
    receipt = _validate_transition(
        source_metadata=source_metadata,
        source_tensors=source_tensors,
        target_metadata=target_metadata,
        target_tensors=target_tensors,
        authorization=authorization,
        rng_state_before=rng_state_before,
        rng_state_after=rng_state_after,
        rng_tensors_before=rng_tensors_before,
        rng_tensors_after=rng_tensors_after,
    )
    return ModularCapacityEpochMigration(
        metadata=target_metadata,
        tensors=target_tensors,
        receipt=receipt,
    )


__all__ = [
    "MODULAR_CAPACITY_EPOCH_AUTHORIZATION_SCHEMA_VERSION",
    "MODULAR_CAPACITY_EPOCH_MIGRATION_SCHEMA_VERSION",
    "MODULAR_CAPACITY_EPOCH_RECEIPT_SCHEMA_VERSION",
    "ModularCapacityEpochError",
    "ModularCapacityEpochMigration",
    "ModularCapacityEpochReceipt",
    "capacity_epoch_authorization_receipt_sha256",
    "migrate_modular_capacity_epoch_projection",
    "validate_legacy_unused_tail_empty",
    "validate_modular_capacity_epoch_transition",
]
