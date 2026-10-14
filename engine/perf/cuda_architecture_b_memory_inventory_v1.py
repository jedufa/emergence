from __future__ import annotations

"""Device-neutral physical-storage inventory for CUDA Architecture B.

Callers describe tensor-like allocations with plain metadata.  The inventory
never imports a tensor framework and never infers storage identity, residency,
or checkpoint dispensability from a tensor name.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import re
from typing import Any, Final


SCHEMA_VERSION: Final[str] = "eiel_cuda_architecture_b_memory_inventory_v1"
MEMORY_INVENTORY_SCHEMA_VERSION_V1: Final[str] = SCHEMA_VERSION

POLICY_FAMILIES_V1: Final[tuple[str, ...]] = (
    "fixed_mlp",
    "gru",
    "non_policy",
)
ACTIVE_POLICY_FAMILIES_V1: Final[tuple[str, ...]] = ("fixed_mlp", "gru")
MEMORY_ROLES_V1: Final[tuple[str, ...]] = (
    "resident_active",
    "host_checkpoint_only",
    "rolling_scratch",
    "retained_history",
    "turnover_shadow",
    "immutable_tape",
)
MEMORY_TIERS_V1: Final[tuple[str, ...]] = ("device", "host")
ALLOCATION_SCOPES_V1: Final[tuple[str, ...]] = ("per_cell", "shared")

MAX_METADATA_ENTRIES_V1: Final[int] = 4_096
MAX_CELL_COUNT_V1: Final[int] = 1_000_000
MAX_ACCOUNTED_BYTES_V1: Final[int] = (1 << 63) - 1

AUTHORITY_BINDING_FIELDS_V1: Final[tuple[str, ...]] = (
    "world",
    "profile",
    "build_identity",
    "extension_sha256",
    "state_layout_sha256",
    "active_policy_family",
    "generation_count",
    "ticks_per_generation",
    "summary_mode",
    "tick_output_storage",
)
_AUTHORITY_STRING_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "world",
        "profile",
        "build_identity",
        "extension_sha256",
        "state_layout_sha256",
        "active_policy_family",
        "summary_mode",
        "tick_output_storage",
    }
)
_SHA256_FIELDS: Final[frozenset[str]] = frozenset(
    {"extension_sha256", "state_layout_sha256"}
)
_SHA256_RE: Final[re.Pattern[str]] = re.compile(r"[0-9a-f]{64}")

_REQUIRED_METADATA_FIELDS: Final[tuple[str, ...]] = (
    "name",
    "storage_id",
    "storage_nbytes",
    "allocation_scope",
    "memory_tier",
    "policy_family",
    "role",
    "canonical_checkpoint",
)


class MemoryInventoryContractError(ValueError):
    """Raised when an accepted inventory cannot safely feed another API."""


@dataclass(frozen=True, slots=True)
class TensorStorageMetadataV1:
    """Framework-neutral metadata for one logical tensor or tensor view.

    ``storage_nbytes`` is the size of the complete physical allocation observed
    at ``cell_count``.  Every logical alias of one ``storage_id`` must repeat
    identical storage-level metadata.
    """

    name: str
    storage_id: str | int
    storage_nbytes: int
    allocation_scope: str
    memory_tier: str
    policy_family: str
    role: str
    canonical_checkpoint: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "storage_id": self.storage_id,
            "storage_nbytes": self.storage_nbytes,
            "allocation_scope": self.allocation_scope,
            "memory_tier": self.memory_tier,
            "policy_family": self.policy_family,
            "role": self.role,
            "canonical_checkpoint": self.canonical_checkpoint,
        }


def _summary() -> dict[str, Any]:
    return {
        "storage_count": 0,
        "observed_bytes": 0,
        "per_cell_bytes": 0,
        "shared_bytes": 0,
        "by_memory_tier": {
            tier: {
                "observed_bytes": 0,
                "per_cell_bytes": 0,
                "shared_bytes": 0,
            }
            for tier in MEMORY_TIERS_V1
        },
    }


def _empty_policy_summaries() -> dict[str, Any]:
    return {
        family: {
            "totals": _summary(),
            "by_role": {role: _summary() for role in MEMORY_ROLES_V1},
        }
        for family in POLICY_FAMILIES_V1
    }


def _rejected_receipt(
    *,
    active_policy_family: object,
    cell_count: object,
    inventory_complete: object,
    errors: Sequence[str],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "rejected",
        "ok": False,
        "inventory_complete": inventory_complete is True,
        "active_policy_family": (
            active_policy_family
            if active_policy_family in ACTIVE_POLICY_FAMILIES_V1
            else None
        ),
        "cell_count": (
            cell_count
            if type(cell_count) is int and 1 <= cell_count <= MAX_CELL_COUNT_V1
            else None
        ),
        "authority_binding": None,
        "authority_binding_sha256": None,
        "logical_tensor_count": 0,
        "physical_storage_count": 0,
        "deduplicated_alias_count": 0,
        "deduplication_key": "caller_asserted_storage_id",
        "totals": _summary(),
        "canonical_checkpoint_totals": _summary(),
        "by_policy_family": _empty_policy_summaries(),
        "by_role": {role: _summary() for role in MEMORY_ROLES_V1},
        "storages": [],
        "automatic_batch_sizing": {
            "planning_usable": False,
            "production_usable": False,
            "per_cell_bytes": None,
            "shared_bytes": None,
            "basis": "rejected_inventory",
            "host_bytes_excluded": None,
        },
        "contract": _contract_receipt(),
        "errors": sorted(set(errors)),
    }


def _contract_receipt() -> dict[str, Any]:
    return {
        "physical_storage_identity_is_caller_asserted": True,
        "all_logical_aliases_must_agree_on_storage_metadata": True,
        "inactive_policy_storage_rule": "canonical_host_checkpoint_only",
        "inactive_gru_is_assumed_discardable": False,
        "automatic_batch_sizing_counts_device_tier_only": True,
        "allocator_and_framework_overhead_included": False,
        "maximum_metadata_entries": MAX_METADATA_ENTRIES_V1,
        "maximum_accounted_bytes": MAX_ACCOUNTED_BYTES_V1,
        "authority_binding_fields": list(AUTHORITY_BINDING_FIELDS_V1),
        "authority_binding_is_required": True,
        "batch_request_must_match_authority_binding": True,
        "automatic_batch_sizing_is_planning_only": True,
        "production_allocation_requires_native_runtime_authority": True,
    }


def _authority_binding_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _receipt_sha256(value: Mapping[str, Any]) -> str:
    payload = {
        key: child for key, child in value.items() if key != "receipt_sha256"
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _normalize_authority_binding(
    value: object,
    *,
    path: str,
    errors: list[str],
) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        errors.append(f"{path}: expected a mapping")
        return None

    expected_fields = set(AUTHORITY_BINDING_FIELDS_V1)
    actual_fields = set(value)
    missing = sorted(expected_fields - actual_fields)
    unexpected = sorted(actual_fields - expected_fields)
    if missing:
        errors.append(f"{path}: missing required fields: {', '.join(missing)}")
    if unexpected:
        errors.append(f"{path}: unexpected fields: {', '.join(unexpected)}")

    normalized: dict[str, Any] = {}
    for field in AUTHORITY_BINDING_FIELDS_V1:
        field_value = value.get(field)
        field_path = f"{path}.{field}"
        if field in _AUTHORITY_STRING_FIELDS:
            if (
                not isinstance(field_value, str)
                or not field_value
                or field_value != field_value.strip()
                or len(field_value) > 256
            ):
                errors.append(
                    f"{field_path}: expected a trimmed nonempty string "
                    "<= 256 chars"
                )
                continue
            if (
                field in _SHA256_FIELDS
                and _SHA256_RE.fullmatch(field_value) is None
            ):
                errors.append(
                    f"{field_path}: expected exactly 64 lowercase hex characters"
                )
                continue
            if (
                field == "active_policy_family"
                and field_value not in ACTIVE_POLICY_FAMILIES_V1
            ):
                errors.append(
                    f"{field_path}: expected exactly 'fixed_mlp' or 'gru'"
                )
                continue
        elif (
            type(field_value) is not int
            or field_value <= 0
            or field_value > MAX_ACCOUNTED_BYTES_V1
        ):
            errors.append(
                f"{field_path}: expected an integer in "
                f"[1, {MAX_ACCOUNTED_BYTES_V1}]"
            )
            continue
        normalized[field] = field_value

    if missing or unexpected or len(normalized) != len(AUTHORITY_BINDING_FIELDS_V1):
        return None
    return normalized


def _metadata_mapping(value: object) -> Mapping[str, Any] | None:
    if isinstance(value, TensorStorageMetadataV1):
        return value.to_dict()
    if isinstance(value, Mapping):
        return value
    return None


def _valid_bounded_string(
    value: object,
    *,
    field: str,
    path: str,
    errors: list[str],
) -> str | None:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or len(value) > 256
    ):
        errors.append(
            f"{path}.{field}: expected a trimmed nonempty string <= 256 chars"
        )
        return None
    return value


def _valid_storage_id(
    value: object,
    *,
    path: str,
    errors: list[str],
) -> str | int | None:
    if isinstance(value, str):
        return _valid_bounded_string(
            value, field="storage_id", path=path, errors=errors
        )
    if type(value) is int and 0 <= value <= MAX_ACCOUNTED_BYTES_V1:
        return value
    errors.append(
        f"{path}.storage_id: expected a nonnegative int64 or a trimmed "
        "nonempty string <= 256 chars"
    )
    return None


def _storage_sort_key(storage_id: str | int) -> tuple[int, str | int]:
    return (0, storage_id) if type(storage_id) is int else (1, storage_id)


def _normalize_metadata(
    raw: object,
    *,
    index: int,
    active_policy_family: str,
    cell_count: int,
    errors: list[str],
) -> dict[str, Any] | None:
    path = f"tensor_metadata[{index}]"
    value = _metadata_mapping(raw)
    if value is None:
        errors.append(f"{path}: expected a mapping or TensorStorageMetadataV1")
        return None

    missing = [field for field in _REQUIRED_METADATA_FIELDS if field not in value]
    if missing:
        errors.append(f"{path}: missing required fields: {', '.join(missing)}")
        return None

    name = _valid_bounded_string(
        value.get("name"), field="name", path=path, errors=errors
    )
    storage_id = _valid_storage_id(
        value.get("storage_id"), path=path, errors=errors
    )

    storage_nbytes = value.get("storage_nbytes")
    if (
        type(storage_nbytes) is not int
        or storage_nbytes < 0
        or storage_nbytes > MAX_ACCOUNTED_BYTES_V1
    ):
        errors.append(
            f"{path}.storage_nbytes: expected an integer in "
            f"[0, {MAX_ACCOUNTED_BYTES_V1}]"
        )
        storage_nbytes = None

    allocation_scope = value.get("allocation_scope")
    if allocation_scope not in ALLOCATION_SCOPES_V1:
        errors.append(
            f"{path}.allocation_scope: expected one of {ALLOCATION_SCOPES_V1!r}"
        )
        allocation_scope = None

    memory_tier = value.get("memory_tier")
    if memory_tier not in MEMORY_TIERS_V1:
        errors.append(f"{path}.memory_tier: expected one of {MEMORY_TIERS_V1!r}")
        memory_tier = None

    policy_family = value.get("policy_family")
    if policy_family not in POLICY_FAMILIES_V1:
        errors.append(
            f"{path}.policy_family: expected one of {POLICY_FAMILIES_V1!r}"
        )
        policy_family = None

    role = value.get("role")
    if role not in MEMORY_ROLES_V1:
        errors.append(f"{path}.role: expected one of {MEMORY_ROLES_V1!r}")
        role = None

    canonical_checkpoint = value.get("canonical_checkpoint")
    if type(canonical_checkpoint) is not bool:
        errors.append(f"{path}.canonical_checkpoint: expected an exact boolean")
        canonical_checkpoint = None

    if (
        storage_nbytes is not None
        and allocation_scope == "per_cell"
        and storage_nbytes % cell_count != 0
    ):
        errors.append(
            f"{path}.storage_nbytes: per_cell allocation is not divisible by "
            f"cell_count={cell_count}"
        )

    if role == "host_checkpoint_only" and (
        memory_tier != "host" or canonical_checkpoint is not True
    ):
        errors.append(
            f"{path}: host_checkpoint_only storage must be host tier and canonical"
        )
    if role == "resident_active" and memory_tier != "device":
        errors.append(f"{path}: resident_active storage must be device tier")
    if role in {"rolling_scratch", "turnover_shadow"} and memory_tier != "device":
        errors.append(f"{path}: {role} storage must be device tier")
    if (
        policy_family in ACTIVE_POLICY_FAMILIES_V1
        and policy_family != active_policy_family
        and (
            role != "host_checkpoint_only"
            or memory_tier != "host"
            or canonical_checkpoint is not True
        )
    ):
        errors.append(
            f"{path}: inactive {policy_family} storage must remain canonical "
            "host_checkpoint_only"
        )

    normalized_values = (
        name,
        storage_id,
        storage_nbytes,
        allocation_scope,
        memory_tier,
        policy_family,
        role,
        canonical_checkpoint,
    )
    if any(item is None for item in normalized_values):
        return None
    return dict(zip(_REQUIRED_METADATA_FIELDS, normalized_values, strict=True))


def _storage_signature(value: Mapping[str, Any]) -> tuple[Any, ...]:
    return tuple(
        value[field]
        for field in _REQUIRED_METADATA_FIELDS
        if field not in {"name", "storage_id"}
    )


def _checked_add(current: int, increment: int, *, path: str) -> int:
    result = current + increment
    if result > MAX_ACCOUNTED_BYTES_V1:
        raise MemoryInventoryContractError(
            f"{path}: accounted byte sum exceeds {MAX_ACCOUNTED_BYTES_V1}"
        )
    return result


def _add_storage(
    summary: dict[str, Any],
    *,
    storage_nbytes: int,
    bytes_per_cell: int,
    shared_bytes: int,
    memory_tier: str,
    path: str,
) -> None:
    summary["storage_count"] += 1
    summary["observed_bytes"] = _checked_add(
        summary["observed_bytes"], storage_nbytes, path=f"{path}.observed_bytes"
    )
    summary["per_cell_bytes"] = _checked_add(
        summary["per_cell_bytes"], bytes_per_cell, path=f"{path}.per_cell_bytes"
    )
    summary["shared_bytes"] = _checked_add(
        summary["shared_bytes"], shared_bytes, path=f"{path}.shared_bytes"
    )
    tier = summary["by_memory_tier"][memory_tier]
    tier["observed_bytes"] = _checked_add(
        tier["observed_bytes"],
        storage_nbytes,
        path=f"{path}.by_memory_tier.{memory_tier}.observed_bytes",
    )
    tier["per_cell_bytes"] = _checked_add(
        tier["per_cell_bytes"],
        bytes_per_cell,
        path=f"{path}.by_memory_tier.{memory_tier}.per_cell_bytes",
    )
    tier["shared_bytes"] = _checked_add(
        tier["shared_bytes"],
        shared_bytes,
        path=f"{path}.by_memory_tier.{memory_tier}.shared_bytes",
    )


def build_cuda_architecture_b_memory_inventory_v1(
    tensor_metadata: Sequence[TensorStorageMetadataV1 | Mapping[str, Any]],
    *,
    cell_count: int,
    active_policy_family: str,
    inventory_complete: bool,
    authority_binding: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a deterministic, fail-closed physical-storage receipt.

    Rejected receipts intentionally publish no batch-sizing byte values.  An
    accepted receipt counts host storage, including inactive canonical policy
    state, but excludes it from the device-memory inputs used by batch sizing.
    """

    errors: list[str] = []
    if type(cell_count) is not int or not 1 <= cell_count <= MAX_CELL_COUNT_V1:
        errors.append(
            f"cell_count: expected an integer in [1, {MAX_CELL_COUNT_V1}]"
        )
    if active_policy_family not in ACTIVE_POLICY_FAMILIES_V1:
        errors.append(
            "active_policy_family: expected exactly 'fixed_mlp' or 'gru'"
        )
    if inventory_complete is not True:
        errors.append("inventory_complete: must be exactly true")
    normalized_binding = _normalize_authority_binding(
        authority_binding,
        path="authority_binding",
        errors=errors,
    )
    if (
        normalized_binding is not None
        and normalized_binding["active_policy_family"] != active_policy_family
    ):
        errors.append(
            "authority_binding.active_policy_family: must match "
            "active_policy_family"
        )
    if not isinstance(tensor_metadata, Sequence) or isinstance(
        tensor_metadata, (str, bytes, bytearray)
    ):
        errors.append("tensor_metadata: expected a bounded sequence")
        metadata_values: Sequence[object] = ()
    else:
        metadata_values = tensor_metadata
        if not metadata_values:
            errors.append("tensor_metadata: expected at least one entry")
        if len(metadata_values) > MAX_METADATA_ENTRIES_V1:
            errors.append(
                f"tensor_metadata: exceeds {MAX_METADATA_ENTRIES_V1} entries"
            )

    if errors:
        return _rejected_receipt(
            active_policy_family=active_policy_family,
            cell_count=cell_count,
            inventory_complete=inventory_complete,
            errors=errors,
        )

    normalized: list[dict[str, Any]] = []
    for index, raw in enumerate(metadata_values):
        item = _normalize_metadata(
            raw,
            index=index,
            active_policy_family=active_policy_family,
            cell_count=cell_count,
            errors=errors,
        )
        if item is not None:
            normalized.append(item)

    names: dict[str, str | int] = {}
    storage_groups: dict[str | int, list[dict[str, Any]]] = {}
    for item in normalized:
        previous_storage = names.get(item["name"])
        if previous_storage is not None:
            errors.append(
                f"logical tensor name {item['name']!r} is duplicated across "
                f"storage ids {previous_storage!r} and {item['storage_id']!r}"
            )
        else:
            names[item["name"]] = item["storage_id"]
        storage_groups.setdefault(item["storage_id"], []).append(item)

    for storage_id, aliases in sorted(
        storage_groups.items(), key=lambda item: _storage_sort_key(item[0])
    ):
        signatures = {_storage_signature(alias) for alias in aliases}
        if len(signatures) != 1:
            errors.append(
                f"storage_id {storage_id!r}: logical aliases disagree on physical "
                "storage classification"
            )

    if errors:
        return _rejected_receipt(
            active_policy_family=active_policy_family,
            cell_count=cell_count,
            inventory_complete=inventory_complete,
            errors=errors,
        )

    totals = _summary()
    canonical_totals = _summary()
    policy_summaries = _empty_policy_summaries()
    role_summaries = {role: _summary() for role in MEMORY_ROLES_V1}
    storages: list[dict[str, Any]] = []

    try:
        for storage_id, aliases in sorted(
            storage_groups.items(), key=lambda item: _storage_sort_key(item[0])
        ):
            representative = aliases[0]
            storage_nbytes = representative["storage_nbytes"]
            allocation_scope = representative["allocation_scope"]
            bytes_per_cell = (
                storage_nbytes // cell_count
                if allocation_scope == "per_cell"
                else 0
            )
            shared_bytes = storage_nbytes if allocation_scope == "shared" else 0
            memory_tier = representative["memory_tier"]
            policy_family = representative["policy_family"]
            role = representative["role"]

            add_arguments = {
                "storage_nbytes": storage_nbytes,
                "bytes_per_cell": bytes_per_cell,
                "shared_bytes": shared_bytes,
                "memory_tier": memory_tier,
            }
            _add_storage(totals, path="totals", **add_arguments)
            _add_storage(
                policy_summaries[policy_family]["totals"],
                path=f"by_policy_family.{policy_family}.totals",
                **add_arguments,
            )
            _add_storage(
                policy_summaries[policy_family]["by_role"][role],
                path=f"by_policy_family.{policy_family}.by_role.{role}",
                **add_arguments,
            )
            _add_storage(
                role_summaries[role], path=f"by_role.{role}", **add_arguments
            )
            if representative["canonical_checkpoint"]:
                _add_storage(
                    canonical_totals,
                    path="canonical_checkpoint_totals",
                    **add_arguments,
                )

            storages.append(
                {
                    "storage_id": storage_id,
                    "logical_names": sorted(alias["name"] for alias in aliases),
                    "storage_nbytes": storage_nbytes,
                    "allocation_scope": allocation_scope,
                    "bytes_per_cell": bytes_per_cell,
                    "shared_bytes": shared_bytes,
                    "memory_tier": memory_tier,
                    "policy_family": policy_family,
                    "role": role,
                    "canonical_checkpoint": representative[
                        "canonical_checkpoint"
                    ],
                }
            )
    except MemoryInventoryContractError as exc:
        return _rejected_receipt(
            active_policy_family=active_policy_family,
            cell_count=cell_count,
            inventory_complete=inventory_complete,
            errors=[str(exc)],
        )

    device_totals = totals["by_memory_tier"]["device"]
    host_totals = totals["by_memory_tier"]["host"]
    batch_sizing_usable = device_totals["per_cell_bytes"] > 0
    assert normalized_binding is not None
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "status": "accepted",
        "ok": True,
        "inventory_complete": True,
        "active_policy_family": active_policy_family,
        "cell_count": cell_count,
        "authority_binding": normalized_binding,
        "authority_binding_sha256": _authority_binding_sha256(
            normalized_binding
        ),
        "logical_tensor_count": len(normalized),
        "physical_storage_count": len(storages),
        "deduplicated_alias_count": len(normalized) - len(storages),
        "deduplication_key": "caller_asserted_storage_id",
        "totals": totals,
        "canonical_checkpoint_totals": canonical_totals,
        "by_policy_family": policy_summaries,
        "by_role": role_summaries,
        "storages": storages,
        "automatic_batch_sizing": {
            "planning_usable": batch_sizing_usable,
            "production_usable": False,
            "per_cell_bytes": (
                device_totals["per_cell_bytes"] if batch_sizing_usable else None
            ),
            "shared_bytes": (
                device_totals["shared_bytes"] if batch_sizing_usable else None
            ),
            "basis": (
                "deduplicated_device_physical_storage"
                if batch_sizing_usable
                else "no_device_per_cell_storage"
            ),
            "host_bytes_excluded": host_totals["observed_bytes"],
        },
        "contract": _contract_receipt(),
        "errors": [],
    }
    receipt["receipt_sha256"] = _receipt_sha256(receipt)
    return receipt


def batch_sizing_bytes_from_memory_inventory_v1(
    receipt: Mapping[str, Any],
    *,
    expected_authority_binding: Mapping[str, Any],
) -> dict[str, int]:
    """Fail closed until the CUDA extension emits trusted runtime authority."""

    del receipt, expected_authority_binding
    raise MemoryInventoryContractError(
        "host memory inventories are planning evidence only; production batch "
        "authorization requires an opaque native CUDA runtime authority"
    )


__all__ = [
    "ACTIVE_POLICY_FAMILIES_V1",
    "ALLOCATION_SCOPES_V1",
    "AUTHORITY_BINDING_FIELDS_V1",
    "MAX_ACCOUNTED_BYTES_V1",
    "MAX_CELL_COUNT_V1",
    "MAX_METADATA_ENTRIES_V1",
    "MEMORY_INVENTORY_SCHEMA_VERSION_V1",
    "MEMORY_ROLES_V1",
    "MEMORY_TIERS_V1",
    "POLICY_FAMILIES_V1",
    "MemoryInventoryContractError",
    "TensorStorageMetadataV1",
    "batch_sizing_bytes_from_memory_inventory_v1",
    "build_cuda_architecture_b_memory_inventory_v1",
]
