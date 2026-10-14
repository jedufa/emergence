"""Generic prepare-only resource audit contracts for resident campaign owners."""

from __future__ import annotations

import dataclasses
import gc
import hashlib
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .canonical import canonical_sha256

REQUEST_SCHEMA_VERSION = "eiel_resident_prepared_owner_resource_audit_request_v1"
RECEIPT_SCHEMA_VERSION = "eiel_resident_prepared_owner_resource_audit_receipt_v1"
MEASUREMENT_AUTHORIZATION_SCHEMA_VERSION = (
    "eiel_resident_prepared_owner_measurement_authorization_v1"
)
STORAGE_INVENTORY_SCHEMA_VERSION = "eiel_resident_prepared_owner_storage_inventory_v1"
STORAGE_ATTRIBUTION_SCHEMA_VERSION = (
    "eiel_resident_prepared_owner_storage_attribution_v1"
)
SERIALIZATION_SCHEMA_VERSION = (
    "eiel_resident_prepared_owner_preparation_serialization_v1"
)
STAGE_SAMPLE_SCHEMA_VERSION = "eiel_resident_prepared_owner_cuda_memory_sample_v1"
LIFETIME_ANALYSIS_SCHEMA_VERSION = (
    "eiel_resident_prepared_owner_lifetime_analysis_v1"
)

PREPARE_ONLY_STAGES = (
    "process_baseline",
    "module_loaded",
    "prepared_owner",
    "preparation_projection",
    "serialized_projection",
    "teardown",
)


def _nonempty(value: str, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _nonnegative(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _runtime_assumptions(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("runtime_assumptions must be a mapping")
    normalized: dict[str, Any] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError("runtime_assumptions keys must be non-empty strings")
        if not (
            item is None
            or isinstance(item, str)
            or type(item) is bool
            or type(item) is int
            or type(item) is float
        ):
            raise ValueError("runtime_assumptions values must be scalar JSON values")
        normalized[key] = item
    return {key: normalized[key] for key in sorted(normalized)}


def _positive(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _sha256(value: str, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _artifact_hashes(value: Mapping[str, str]) -> dict[str, str]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError("artifact_sha256s must be a non-empty mapping")
    result: dict[str, str] = {}
    for key, digest in sorted(value.items()):
        result[_nonempty(key, name="artifact_sha256s key")] = _sha256(
            digest,
            name=f"artifact_sha256s[{key!r}]",
        )
    return result


def _hardware_identity(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("hardware_identity must be a mapping")
    required = {
        "device_index",
        "device_name",
        "device_uuid",
        "compute_capability",
        "total_memory_bytes",
        "cuda_runtime_version",
        "torch_version",
    }
    if set(value) != required:
        raise ValueError(
            "hardware_identity must contain exactly " + ", ".join(sorted(required))
        )
    capability = value["compute_capability"]
    if (
        not isinstance(capability, (list, tuple))
        or len(capability) != 2
        or any(type(item) is not int or item < 0 for item in capability)
    ):
        raise ValueError("hardware_identity.compute_capability is invalid")
    return {
        "device_index": _nonnegative(
            value["device_index"], name="hardware_identity.device_index"
        ),
        "device_name": _nonempty(
            value["device_name"], name="hardware_identity.device_name"
        ),
        "device_uuid": _nonempty(
            value["device_uuid"], name="hardware_identity.device_uuid"
        ),
        "compute_capability": [int(capability[0]), int(capability[1])],
        "total_memory_bytes": _positive(
            value["total_memory_bytes"],
            name="hardware_identity.total_memory_bytes",
        ),
        "cuda_runtime_version": _nonempty(
            value["cuda_runtime_version"],
            name="hardware_identity.cuda_runtime_version",
        ),
        "torch_version": _nonempty(
            value["torch_version"], name="hardware_identity.torch_version"
        ),
    }


def validate_cuda_hardware_identity_v1(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Return one canonical exact CUDA hardware identity or fail closed."""

    return _hardware_identity(value)


def _route_qualification_binding(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "authority_state",
        "receipt_path",
        "receipt_sha256",
        "artifact_conformance_exact",
    }:
        raise ValueError("route_qualification binding is incomplete")
    if value["authority_state"] != "route_qualified":
        raise ValueError("route_qualification is not qualified")
    if value["artifact_conformance_exact"] is not True:
        raise ValueError("route_qualification artifact conformance is not exact")
    return {
        "authority_state": "route_qualified",
        "receipt_path": _nonempty(
            value["receipt_path"], name="route_qualification.receipt_path"
        ),
        "receipt_sha256": _sha256(
            value["receipt_sha256"],
            name="route_qualification.receipt_sha256",
        ),
        "artifact_conformance_exact": True,
    }


def detect_cuda_hardware_identity_v1() -> dict[str, Any]:
    """Return the exact CUDA identity bound by measurement authorization."""

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    index = int(torch.cuda.current_device())
    properties = torch.cuda.get_device_properties(index)
    device_uuid = getattr(properties, "uuid", None)
    if device_uuid is None:
        raise RuntimeError("CUDA device UUID is unavailable")
    return _hardware_identity(
        {
            "device_index": index,
            "device_name": str(properties.name),
            "device_uuid": str(device_uuid),
            "compute_capability": [
                int(properties.major),
                int(properties.minor),
            ],
            "total_memory_bytes": int(properties.total_memory),
            "cuda_runtime_version": str(torch.version.cuda),
            "torch_version": str(torch.__version__),
        }
    )


def prepared_owner_configuration_sha256_v1(
    plan: Mapping[str, Any],
    batch_request: Mapping[str, Any],
) -> str:
    """Bind the exact campaign configuration and translated batch request."""

    return canonical_sha256(
        {
            "cells": plan.get("cells"),
            "execution": plan.get("execution"),
            "batch_request": dict(batch_request),
        }
    )


@dataclass(frozen=True)
class ResidentPreparedOwnerResourceAuditRequestV1:
    """Immutable bindings and admission policy for one prepare-only audit."""

    audit_id: str
    provider_id: str
    profile_id: str
    campaign_id: str
    plan_sha256: str
    qualification_key_sha256: str
    execution_route_sha256: str
    route_qualification: Mapping[str, Any]
    candidate_authority_sha256: str
    artifact_sha256s: Mapping[str, str]
    configuration_sha256: str
    resource_estimator_id: str
    conservative_vram_projection_bytes: int
    runtime_assumptions: Mapping[str, Any]
    measurement_authorization_sha256: str
    hardware_identity: Mapping[str, Any]
    source_projection_root_sha256: str
    ordered_cell_ids_sha256: str
    batch_request_sha256: str
    batch_cells: int
    resident_segment_generations: int
    physical_window_generations: int
    ticks_per_generation: int
    resident_capacity: int
    initial_population: int
    agent_id_allocation_generation_count: int
    declared_vram_budget_bytes: int
    minimum_physical_headroom_fraction: float = 0.04
    minimum_physical_headroom_bytes: int = 0
    minimum_residual_headroom_bytes: int = 0
    teardown_tolerance_bytes: int = 0
    schema_version: str = field(default=REQUEST_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        for name in ("audit_id", "provider_id", "profile_id", "campaign_id"):
            _nonempty(getattr(self, name), name=name)
        for name in (
            "plan_sha256",
            "qualification_key_sha256",
            "execution_route_sha256",
            "candidate_authority_sha256",
            "configuration_sha256",
            "measurement_authorization_sha256",
            "source_projection_root_sha256",
            "ordered_cell_ids_sha256",
            "batch_request_sha256",
        ):
            _sha256(getattr(self, name), name=name)
        object.__setattr__(
            self, "artifact_sha256s", _artifact_hashes(self.artifact_sha256s)
        )
        object.__setattr__(
            self,
            "route_qualification",
            _route_qualification_binding(self.route_qualification),
        )
        _nonempty(self.resource_estimator_id, name="resource_estimator_id")
        _positive(
            self.conservative_vram_projection_bytes,
            name="conservative_vram_projection_bytes",
        )
        object.__setattr__(
            self,
            "runtime_assumptions",
            _runtime_assumptions(self.runtime_assumptions),
        )
        object.__setattr__(
            self,
            "hardware_identity",
            _hardware_identity(self.hardware_identity),
        )
        for name in (
            "batch_cells",
            "resident_segment_generations",
            "physical_window_generations",
            "ticks_per_generation",
            "resident_capacity",
            "initial_population",
            "agent_id_allocation_generation_count",
            "declared_vram_budget_bytes",
        ):
            _positive(getattr(self, name), name=name)
        if self.initial_population > self.resident_capacity:
            raise ValueError("initial_population must not exceed resident_capacity")
        if self.physical_window_generations not in {1, 2}:
            raise ValueError("physical_window_generations must be 1 or 2")
        if (
            isinstance(self.minimum_physical_headroom_fraction, bool)
            or not isinstance(self.minimum_physical_headroom_fraction, (int, float))
            or not math.isfinite(self.minimum_physical_headroom_fraction)
            or not 0.0 <= self.minimum_physical_headroom_fraction < 1.0
        ):
            raise ValueError(
                "minimum_physical_headroom_fraction must be finite and in [0, 1)"
            )
        _nonnegative(
            self.minimum_physical_headroom_bytes,
            name="minimum_physical_headroom_bytes",
        )
        _nonnegative(
            self.minimum_residual_headroom_bytes,
            name="minimum_residual_headroom_bytes",
        )
        _nonnegative(self.teardown_tolerance_bytes, name="teardown_tolerance_bytes")

    def payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "audit_id": self.audit_id,
            "provider_id": self.provider_id,
            "profile_id": self.profile_id,
            "campaign_id": self.campaign_id,
            "plan_sha256": self.plan_sha256,
            "qualification_key_sha256": self.qualification_key_sha256,
            "execution_route_sha256": self.execution_route_sha256,
            "route_qualification": dict(self.route_qualification),
            "candidate_authority_sha256": self.candidate_authority_sha256,
            "artifact_sha256s": dict(self.artifact_sha256s),
            "configuration_sha256": self.configuration_sha256,
            "resource_estimator_id": self.resource_estimator_id,
            "conservative_vram_projection_bytes": (
                self.conservative_vram_projection_bytes
            ),
            "runtime_assumptions": dict(self.runtime_assumptions),
            "measurement_authorization_sha256": (self.measurement_authorization_sha256),
            "hardware_identity": dict(self.hardware_identity),
            "source_projection_root_sha256": self.source_projection_root_sha256,
            "ordered_cell_ids_sha256": self.ordered_cell_ids_sha256,
            "batch_request_sha256": self.batch_request_sha256,
            "dimensions": {
                "batch_cells": self.batch_cells,
                "resident_segment_generations": self.resident_segment_generations,
                "physical_window_generations": self.physical_window_generations,
                "ticks_per_generation": self.ticks_per_generation,
                "resident_capacity": self.resident_capacity,
                "initial_population": self.initial_population,
                "agent_id_allocation_generation_count": (
                    self.agent_id_allocation_generation_count
                ),
            },
            "policy": {
                "declared_vram_budget_bytes": self.declared_vram_budget_bytes,
                "minimum_physical_headroom_fraction": float(
                    self.minimum_physical_headroom_fraction
                ),
                "minimum_physical_headroom_bytes": (
                    self.minimum_physical_headroom_bytes
                ),
                "minimum_residual_headroom_bytes": (
                    self.minimum_residual_headroom_bytes
                ),
                "teardown_tolerance_bytes": self.teardown_tolerance_bytes,
            },
        }

    @property
    def request_sha256(self) -> str:
        return canonical_sha256(self.payload())


@dataclass(frozen=True)
class CudaMemorySampleV1:
    """One CUDA memory sample captured at a named prepare-only stage."""

    stage: str
    elapsed_ns: int
    allocated_bytes: int
    reserved_bytes: int
    peak_allocated_bytes: int
    peak_reserved_bytes: int
    free_bytes: int
    total_bytes: int
    schema_version: str = field(default=STAGE_SAMPLE_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        if self.stage not in PREPARE_ONLY_STAGES:
            raise ValueError(f"unknown prepare-only stage: {self.stage}")
        for name in (
            "elapsed_ns",
            "allocated_bytes",
            "reserved_bytes",
            "peak_allocated_bytes",
            "peak_reserved_bytes",
            "free_bytes",
            "total_bytes",
        ):
            _nonnegative(getattr(self, name), name=name)
        _positive(self.total_bytes, name="total_bytes")
        if self.peak_allocated_bytes < self.allocated_bytes:
            raise ValueError("peak_allocated_bytes must cover allocated_bytes")
        if self.peak_reserved_bytes < self.reserved_bytes:
            raise ValueError("peak_reserved_bytes must cover reserved_bytes")
        if self.free_bytes > self.total_bytes:
            raise ValueError("free_bytes must not exceed total_bytes")

    def payload(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class ResidentPreparedOwnerStorageInventoryV1:
    """De-aliased inventory of tensor views and their retained storages."""

    tensor_view_count: int
    unique_storage_count: int
    retained_storage_bytes: int
    device_storage_bytes: Mapping[str, int]
    storage_rows: tuple[Mapping[str, Any], ...]
    tensor_view_rows: tuple[Mapping[str, Any], ...] = ()
    schema_version: str = field(default=STORAGE_INVENTORY_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        for name in (
            "tensor_view_count",
            "unique_storage_count",
            "retained_storage_bytes",
        ):
            _nonnegative(getattr(self, name), name=name)
        if self.unique_storage_count > self.tensor_view_count:
            raise ValueError("unique_storage_count must not exceed tensor_view_count")
        checked_devices: dict[str, int] = {}
        for device, size in sorted(self.device_storage_bytes.items()):
            checked_devices[_nonempty(device, name="device")] = _nonnegative(
                size, name=f"device_storage_bytes[{device!r}]"
            )
        if sum(checked_devices.values()) != self.retained_storage_bytes:
            raise ValueError(
                "device storage totals do not match retained storage bytes"
            )
        object.__setattr__(self, "device_storage_bytes", checked_devices)
        object.__setattr__(
            self,
            "storage_rows",
            tuple(dict(row) for row in self.storage_rows),
        )
        object.__setattr__(
            self,
            "tensor_view_rows",
            tuple(dict(row) for row in self.tensor_view_rows),
        )

    def payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "attribution_schema_version": STORAGE_ATTRIBUTION_SCHEMA_VERSION,
            "tensor_view_count": self.tensor_view_count,
            "unique_storage_count": self.unique_storage_count,
            "retained_storage_bytes": self.retained_storage_bytes,
            "device_storage_bytes": dict(self.device_storage_bytes),
            "storage_rows": [dict(row) for row in self.storage_rows],
            "tensor_view_rows": [dict(row) for row in self.tensor_view_rows],
        }


def _tensor_shape(value: Any) -> list[int] | None:
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    try:
        return [int(item) for item in tuple(shape)]
    except (TypeError, ValueError):
        return None


def _is_tensor_like(value: Any) -> bool:
    return (
        hasattr(value, "untyped_storage")
        and callable(value.untyped_storage)
        and hasattr(value, "device")
    )


def _tensor_numel(value: Any) -> int | None:
    numel = getattr(value, "numel", None)
    if callable(numel):
        try:
            return int(numel())
        except (TypeError, ValueError, RuntimeError):
            return None
    shape = _tensor_shape(value)
    if shape is None:
        return None
    result = 1
    for size in shape:
        result *= int(size)
    return result


def _tensor_element_size(value: Any) -> int | None:
    element_size = getattr(value, "element_size", None)
    if callable(element_size):
        try:
            return int(element_size())
        except (TypeError, ValueError, RuntimeError):
            return None
    return None


def _tensor_stride(value: Any) -> list[int] | None:
    stride = getattr(value, "stride", None)
    if callable(stride):
        try:
            return [int(item) for item in tuple(stride())]
        except (TypeError, ValueError, RuntimeError):
            return None
    return None


def _tensor_storage_offset(value: Any) -> int | None:
    storage_offset = getattr(value, "storage_offset", None)
    if callable(storage_offset):
        try:
            return int(storage_offset())
        except (TypeError, ValueError, RuntimeError):
            return None
    return None


def _tensor_data_ptr(value: Any) -> int | None:
    data_ptr = getattr(value, "data_ptr", None)
    if callable(data_ptr):
        try:
            return int(data_ptr())
        except (TypeError, ValueError, RuntimeError):
            return None
    return None


def _tensor_is_contiguous(value: Any) -> bool | None:
    is_contiguous = getattr(value, "is_contiguous", None)
    if callable(is_contiguous):
        try:
            return bool(is_contiguous())
        except (TypeError, ValueError, RuntimeError):
            return None
    return None


def _owner_lifetime_category(path: str) -> str:
    lowered = path.lower()
    if "python_random_gauss" in lowered:
        return "immutable_gaussian_tape"
    if "policy_birth_post_rng_state" in lowered:
        return "immutable_torch_post_state_tape"
    if "policy_birth_crossover" in lowered:
        return "immutable_crossover_tape"
    if "turnover_shadow" in lowered:
        return "turnover_shadow"
    if "mlp_pack" in lowered:
        return "resident_mlp_pack"
    if "checkpoint" in lowered or "projection" in lowered:
        return "projection_or_checkpoint"
    if "summary" in lowered:
        return "compact_summary_or_reporting"
    return "resident_owner_tensor"


def inventory_resident_prepared_owner_storages_v1(
    owner: Any,
    *,
    tensor_predicate: Callable[[Any], bool] = _is_tensor_like,
    maximum_paths_per_storage: int = 8,
) -> ResidentPreparedOwnerStorageInventoryV1:
    """Recursively inventory tensor-like objects without double-counting aliases."""

    _positive(maximum_paths_per_storage, name="maximum_paths_per_storage")
    visited_containers: set[int] = set()
    storages: dict[tuple[str, int], dict[str, Any]] = {}
    tensor_view_rows: list[dict[str, Any]] = []
    tensor_view_count = 0

    def visit(value: Any, path: str) -> None:
        nonlocal tensor_view_count
        if tensor_predicate(value):
            storage = value.untyped_storage()
            size = int(storage.nbytes())
            _nonnegative(size, name=f"{path}.storage.nbytes")
            if size == 0:
                return
            tensor_view_count += 1
            pointer = int(storage.data_ptr())
            device = str(value.device)
            identity = (device, pointer if pointer != 0 else id(storage))
            alias_group_id = f"{device}:{pointer if pointer != 0 else id(storage):x}"
            shape = _tensor_shape(value)
            numel = _tensor_numel(value)
            element_size = _tensor_element_size(value)
            logical_bytes = (
                None if numel is None or element_size is None else numel * element_size
            )
            row = storages.setdefault(
                identity,
                {
                    "alias_group_id": alias_group_id,
                    "device": device,
                    "data_ptr": pointer,
                    "storage_data_ptr": pointer,
                    "nbytes": size,
                    "storage_nbytes": size,
                    "dtype": str(getattr(value, "dtype", "unknown")),
                    "shape": shape,
                    "encoding": "dense_tensor_storage_v1",
                    "paths": [],
                    "view_count": 0,
                    "logical_bytes_total": 0,
                },
            )
            if row["nbytes"] != size:
                raise ValueError(
                    "one storage identity reported inconsistent byte sizes"
                )
            if logical_bytes is not None:
                row["logical_bytes_total"] += int(logical_bytes)
            row["view_count"] += 1
            if len(row["paths"]) < maximum_paths_per_storage:
                row["paths"].append(path)
            tensor_view_rows.append(
                {
                    "alias_group_id": alias_group_id,
                    "path": path,
                    "device": device,
                    "dtype": str(getattr(value, "dtype", "unknown")),
                    "shape": shape,
                    "logical_bytes": logical_bytes,
                    "data_ptr": _tensor_data_ptr(value),
                    "storage_data_ptr": pointer,
                    "storage_nbytes": size,
                    "storage_offset": _tensor_storage_offset(value),
                    "stride": _tensor_stride(value),
                    "contiguous": _tensor_is_contiguous(value),
                    "python_object_id": id(value),
                    "has_base_view": getattr(value, "_base", None) is not None,
                    "owner_lifetime_category": _owner_lifetime_category(path),
                    "creation_stage": "prepared_owner_inventory",
                    "last_use_stage": "unknown_prepare_only",
                    "retained_after_stage": "prepared_owner",
                }
            )
            return
        if value is None or isinstance(value, (str, bytes, int, float, bool)):
            return
        identity = id(value)
        if identity in visited_containers:
            return
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            visited_containers.add(identity)
            for item in dataclasses.fields(value):
                visit(getattr(value, item.name), f"{path}.{item.name}")
            return
        if isinstance(value, Mapping):
            visited_containers.add(identity)
            for key in sorted(value, key=lambda item: repr(item)):
                visit(value[key], f"{path}[{key!r}]")
            return
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            visited_containers.add(identity)
            for index, item in enumerate(value):
                visit(item, f"{path}[{index}]")

    visit(owner, "$")
    rows = tuple(
        sorted(
            (dict(row) for row in storages.values()),
            key=lambda row: (
                -int(row["nbytes"]),
                str(row["device"]),
                int(row["data_ptr"]),
            ),
        )
    )
    device_bytes: dict[str, int] = {}
    for row in rows:
        device = str(row["device"])
        device_bytes[device] = device_bytes.get(device, 0) + int(row["nbytes"])
    return ResidentPreparedOwnerStorageInventoryV1(
        tensor_view_count=tensor_view_count,
        unique_storage_count=len(rows),
        retained_storage_bytes=sum(device_bytes.values()),
        device_storage_bytes=device_bytes,
        storage_rows=rows,
        tensor_view_rows=tuple(
            sorted(
                tensor_view_rows,
                key=lambda row: (
                    str(row["alias_group_id"]),
                    str(row["path"]),
                    int(row["python_object_id"]),
                ),
            )
        ),
    )


def _top_storage_rows_v1(
    inventory: ResidentPreparedOwnerStorageInventoryV1,
    *,
    limit: int = 12,
) -> list[dict[str, Any]]:
    _positive(limit, name="limit")
    rows: list[dict[str, Any]] = []
    for row in inventory.storage_rows[:limit]:
        rows.append(
            {
                "device": str(row["device"]),
                "nbytes": int(row["nbytes"]),
                "dtype": str(row.get("dtype", "unknown")),
                "shape": row.get("shape"),
                "encoding": str(row.get("encoding", "unknown")),
                "view_count": int(row["view_count"]),
                "paths": list(row["paths"]),
            }
        )
    return rows


def derive_prepared_owner_lifetime_analysis_v1(
    request: ResidentPreparedOwnerResourceAuditRequestV1,
    stage_samples: Sequence[CudaMemorySampleV1],
    storage_inventory: ResidentPreparedOwnerStorageInventoryV1,
    projection_storage_inventory: ResidentPreparedOwnerStorageInventoryV1 | None = None,
) -> dict[str, Any]:
    """Derive retained/transient attribution for one prepare-only owner peak."""

    samples = tuple(stage_samples)
    if tuple(sample.stage for sample in samples) != PREPARE_ONLY_STAGES:
        raise ValueError("all prepare-only stages must be present in order")
    baseline = samples[0]
    active_samples = samples[1:-1]
    if not active_samples:
        raise ValueError("prepare-only lifetime analysis requires active stages")
    retained_cuda_storage = sum(
        int(size)
        for device, size in storage_inventory.device_storage_bytes.items()
        if str(device).startswith("cuda")
    )
    projection_cuda_storage = 0
    projection_host_storage = 0
    if projection_storage_inventory is not None:
        for device, size in projection_storage_inventory.device_storage_bytes.items():
            if str(device).startswith("cuda"):
                projection_cuda_storage += int(size)
            else:
                projection_host_storage += int(size)

    def reserved_delta(sample: CudaMemorySampleV1) -> int:
        return max(0, int(sample.peak_reserved_bytes) - int(baseline.reserved_bytes))

    def allocated_delta(sample: CudaMemorySampleV1) -> int:
        return max(0, int(sample.peak_allocated_bytes) - int(baseline.allocated_bytes))

    peak_reserved_stage = max(
        active_samples,
        key=lambda sample: (reserved_delta(sample), sample.elapsed_ns, sample.stage),
    )
    peak_allocated_stage = max(
        active_samples,
        key=lambda sample: (allocated_delta(sample), sample.elapsed_ns, sample.stage),
    )
    peak_reserved_delta = reserved_delta(peak_reserved_stage)
    peak_allocated_delta = allocated_delta(peak_allocated_stage)
    peak_process_delta = max(peak_reserved_delta, peak_allocated_delta)
    retained_gap = max(0, peak_process_delta - retained_cuda_storage)
    per_cell_gap = retained_gap // request.batch_cells
    return {
        "schema_version": LIFETIME_ANALYSIS_SCHEMA_VERSION,
        "analysis_scope": "prepare_only_retained_vs_peak_v1",
        "batch_cells": request.batch_cells,
        "peak_reserved_stage": peak_reserved_stage.stage,
        "peak_allocated_stage": peak_allocated_stage.stage,
        "peak_process_delta_bytes": peak_process_delta,
        "peak_reserved_delta_bytes": peak_reserved_delta,
        "peak_allocated_delta_bytes": peak_allocated_delta,
        "retained_cuda_storage_bytes": retained_cuda_storage,
        "retained_owner_storage_bytes": storage_inventory.retained_storage_bytes,
        "cuda_retained_to_peak_gap_bytes": retained_gap,
        "cuda_retained_to_peak_gap_bytes_per_cell_floor": per_cell_gap,
        "gap_interpretation": (
            "peak_exceeds_retained_cuda_storage; attribute to transient staging, "
            "allocator reserve, non-tensor CUDA allocations, or overlapping "
            "lifetimes until later stage-aware instrumentation separates them"
            if retained_gap > 0
            else "retained_cuda_storage_covers_peak_process_delta"
        ),
        "stage_deltas": [
            {
                "stage": sample.stage,
                "allocated_delta_bytes": max(
                    0, int(sample.allocated_bytes) - int(baseline.allocated_bytes)
                ),
                "reserved_delta_bytes": max(
                    0, int(sample.reserved_bytes) - int(baseline.reserved_bytes)
                ),
                "peak_allocated_delta_bytes": allocated_delta(sample),
                "peak_reserved_delta_bytes": reserved_delta(sample),
                "free_delta_bytes": int(sample.free_bytes) - int(baseline.free_bytes),
            }
            for sample in samples
        ],
        "device_storage_bytes": dict(storage_inventory.device_storage_bytes),
        "projection_storage_summary": (
            None
            if projection_storage_inventory is None
            else {
                "tensor_view_count": projection_storage_inventory.tensor_view_count,
                "unique_storage_count": projection_storage_inventory.unique_storage_count,
                "retained_storage_bytes": (
                    projection_storage_inventory.retained_storage_bytes
                ),
                "cuda_storage_bytes": projection_cuda_storage,
                "host_storage_bytes": projection_host_storage,
                "top_storage_rows": _top_storage_rows_v1(
                    projection_storage_inventory
                ),
            }
        ),
        "stage_overlap_table": [
            {
                "stage": sample.stage,
                "retained_owner_cuda_storage_bytes": retained_cuda_storage,
                "projection_cuda_storage_bytes": (
                    projection_cuda_storage
                    if sample.stage in {"preparation_projection", "serialized_projection"}
                    else 0
                ),
                "projection_host_storage_bytes": (
                    projection_host_storage
                    if sample.stage in {"preparation_projection", "serialized_projection"}
                    else 0
                ),
                "reserved_delta_bytes": max(
                    0, int(sample.reserved_bytes) - int(baseline.reserved_bytes)
                ),
                "peak_reserved_delta_bytes": reserved_delta(sample),
            }
            for sample in samples
        ],
        "top_retained_storage_rows": _top_storage_rows_v1(storage_inventory),
        "estimator_schema": request.resource_estimator_id,
    }


@dataclass(frozen=True)
class PreparationProjectionSerializationV1:
    projection_sha256: str
    serialized_path: str
    serialized_sha256: str
    serialized_bytes: int
    serialization_time_ns: int
    format_id: str
    schema_version: str = field(default=SERIALIZATION_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        _sha256(self.projection_sha256, name="projection_sha256")
        path = Path(_nonempty(self.serialized_path, name="serialized_path"))
        if not path.is_absolute():
            raise ValueError("serialized_path must be absolute")
        _sha256(self.serialized_sha256, name="serialized_sha256")
        _nonnegative(self.serialized_bytes, name="serialized_bytes")
        _nonnegative(self.serialization_time_ns, name="serialization_time_ns")
        _nonempty(self.format_id, name="format_id")

    def payload(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def serialize_preparation_projection_v1(
    projection: Any,
    *,
    projection_sha256: str,
    destination: Path,
    format_id: str,
    serializer: Callable[[Any, Path], None],
    clock_ns: Callable[[], int] = time.perf_counter_ns,
) -> PreparationProjectionSerializationV1:
    """Serialize through an injected format writer and bind the resulting bytes."""

    _sha256(projection_sha256, name="projection_sha256")
    _nonempty(format_id, name="format_id")
    if destination.exists():
        raise FileExistsError(destination)
    started = clock_ns()
    serializer(projection, destination)
    finished = clock_ns()
    if not destination.is_file():
        raise ValueError("serializer did not create the requested destination")
    digest = hashlib.sha256()
    size = 0
    with destination.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)
    return PreparationProjectionSerializationV1(
        projection_sha256=projection_sha256,
        serialized_path=str(destination.resolve()),
        serialized_sha256=digest.hexdigest(),
        serialized_bytes=size,
        serialization_time_ns=max(0, finished - started),
        format_id=format_id,
    )


@dataclass(frozen=True)
class ResidentPreparedOwnerResourceAuditReceiptV1:
    request: ResidentPreparedOwnerResourceAuditRequestV1
    stage_samples: tuple[CudaMemorySampleV1, ...]
    storage_inventory: ResidentPreparedOwnerStorageInventoryV1
    preparation_serialization: PreparationProjectionSerializationV1
    executor_call_count: int
    scientific_tick_count: int
    decision: Mapping[str, Any]
    projection_storage_inventory: ResidentPreparedOwnerStorageInventoryV1 | None = None
    schema_version: str = field(default=RECEIPT_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        if tuple(sample.stage for sample in self.stage_samples) != PREPARE_ONLY_STAGES:
            raise ValueError("receipt must contain every prepare-only stage in order")
        _nonnegative(self.executor_call_count, name="executor_call_count")
        _nonnegative(self.scientific_tick_count, name="scientific_tick_count")
        if self.executor_call_count or self.scientific_tick_count:
            raise ValueError(
                "prepare-only receipt requires zero structural executor/tick counters"
            )

    def unsigned_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "request": self.request.payload(),
            "request_sha256": self.request.request_sha256,
            "stage_samples": [sample.payload() for sample in self.stage_samples],
            "storage_inventory": self.storage_inventory.payload(),
            "projection_storage_inventory": (
                None
                if self.projection_storage_inventory is None
                else self.projection_storage_inventory.payload()
            ),
            "preparation_serialization": self.preparation_serialization.payload(),
            "executor_call_count": self.executor_call_count,
            "scientific_tick_count": self.scientific_tick_count,
            "decision": dict(self.decision),
        }

    @property
    def receipt_sha256(self) -> str:
        return canonical_sha256(self.unsigned_payload())

    def payload(self) -> dict[str, Any]:
        return {**self.unsigned_payload(), "receipt_sha256": self.receipt_sha256}


def derive_prepared_owner_resource_decision_v1(
    request: ResidentPreparedOwnerResourceAuditRequestV1,
    stage_samples: Sequence[CudaMemorySampleV1],
    storage_inventory: ResidentPreparedOwnerStorageInventoryV1,
    projection_storage_inventory: ResidentPreparedOwnerStorageInventoryV1 | None = None,
) -> dict[str, Any]:
    """Derive the complete structural prepare-only resource decision."""

    samples = tuple(stage_samples)
    if tuple(sample.stage for sample in samples) != PREPARE_ONLY_STAGES:
        raise ValueError("all prepare-only stages must be present in order")
    if any(
        right.elapsed_ns < left.elapsed_ns
        for left, right in zip(samples, samples[1:])
    ):
        raise ValueError("stage elapsed_ns must be monotonic")
    baseline = samples[0]
    if any(sample.total_bytes != baseline.total_bytes for sample in samples):
        raise ValueError("CUDA total memory changed during one audit")
    if baseline.total_bytes != request.hardware_identity["total_memory_bytes"]:
        raise ValueError("stage samples do not bind the authorized CUDA hardware")
    teardown = samples[-1]
    active_samples = samples[1:-1]
    peak_allocated_delta = max(
        0,
        max(sample.peak_allocated_bytes for sample in active_samples)
        - baseline.allocated_bytes,
    )
    peak_reserved_delta = max(
        0,
        max(sample.peak_reserved_bytes for sample in active_samples)
        - baseline.reserved_bytes,
    )
    retained_cuda_storage = sum(
        int(size)
        for device, size in storage_inventory.device_storage_bytes.items()
        if str(device).startswith("cuda")
    )
    measured_owner_peak = max(
        peak_allocated_delta,
        peak_reserved_delta,
        retained_cuda_storage,
    )
    minimum_free = min(sample.free_bytes for sample in active_samples)
    required_headroom = max(
        request.minimum_physical_headroom_bytes,
        math.ceil(
            baseline.total_bytes * request.minimum_physical_headroom_fraction
        ),
    )
    tolerance = request.teardown_tolerance_bytes
    teardown_ok = teardown.allocated_bytes <= baseline.allocated_bytes + tolerance
    within_declared_budget = (
        measured_owner_peak <= request.declared_vram_budget_bytes
    )
    within_conservative_projection = (
        measured_owner_peak <= request.conservative_vram_projection_bytes
    )
    physical_headroom_ok = minimum_free >= required_headroom
    residual_headroom = minimum_free - required_headroom
    residual_headroom_ok = (
        residual_headroom >= request.minimum_residual_headroom_bytes
    )
    lifetime_analysis = derive_prepared_owner_lifetime_analysis_v1(
        request,
        samples,
        storage_inventory,
        projection_storage_inventory,
    )
    return {
        "measured_owner_peak_bytes": measured_owner_peak,
        "retained_owner_storage_bytes": storage_inventory.retained_storage_bytes,
        "retained_cuda_storage_bytes": retained_cuda_storage,
        "peak_process_allocated_delta_bytes": peak_allocated_delta,
        "peak_process_reserved_delta_bytes": peak_reserved_delta,
        "minimum_observed_free_bytes": minimum_free,
        "required_physical_headroom_bytes": required_headroom,
        "residual_headroom_bytes": residual_headroom,
        "minimum_residual_headroom_bytes": (
            request.minimum_residual_headroom_bytes
        ),
        "declared_vram_budget_bytes": request.declared_vram_budget_bytes,
        "within_declared_budget": within_declared_budget,
        "conservative_vram_projection_bytes": (
            request.conservative_vram_projection_bytes
        ),
        "within_conservative_projection": within_conservative_projection,
        "physical_headroom_ok": physical_headroom_ok,
        "residual_headroom_ok": residual_headroom_ok,
        "teardown_to_baseline": {
            "allocated_delta_bytes": teardown.allocated_bytes
            - baseline.allocated_bytes,
            "reserved_delta_bytes": teardown.reserved_bytes
            - baseline.reserved_bytes,
            "free_delta_bytes": teardown.free_bytes - baseline.free_bytes,
            "tolerance_bytes": tolerance,
            "allocator_cache_non_authoritative": True,
            "ok": teardown_ok,
        },
        "prepare_only_evidence_scope": (
            "structural_zero_dispatch_zero_tick_counter_v1"
        ),
        "prepare_only": True,
        "lifetime_analysis": lifetime_analysis,
        "ok": (
            within_declared_budget
            and within_conservative_projection
            and physical_headroom_ok
            and residual_headroom_ok
            and teardown_ok
        ),
    }


class ResidentPreparedOwnerAuditStageRecorderV1:
    """Fail-closed collector for one strictly prepare-only lifecycle."""

    def __init__(self, request: ResidentPreparedOwnerResourceAuditRequestV1) -> None:
        self.request = request
        self._samples: list[CudaMemorySampleV1] = []
        self._inventory: ResidentPreparedOwnerStorageInventoryV1 | None = None
        self._projection_inventory: ResidentPreparedOwnerStorageInventoryV1 | None = None
        self._serialization: PreparationProjectionSerializationV1 | None = None
        self._executor_call_count = 0
        self._scientific_tick_count = 0

    def record_sample(self, sample: CudaMemorySampleV1) -> None:
        expected = (
            PREPARE_ONLY_STAGES[len(self._samples)]
            if len(self._samples) < len(PREPARE_ONLY_STAGES)
            else None
        )
        if sample.stage != expected:
            raise ValueError(f"expected stage {expected!r}, received {sample.stage!r}")
        if self._samples and sample.elapsed_ns < self._samples[-1].elapsed_ns:
            raise ValueError("stage elapsed_ns must be monotonic")
        baseline = self._samples[0] if self._samples else sample
        if sample.total_bytes != baseline.total_bytes:
            raise ValueError("CUDA total memory changed during one audit")
        self._samples.append(sample)

    def record_storage_inventory(
        self, inventory: ResidentPreparedOwnerStorageInventoryV1
    ) -> None:
        if self._inventory is not None:
            raise ValueError("storage inventory was already recorded")
        self._inventory = inventory

    def record_projection_storage_inventory(
        self, inventory: ResidentPreparedOwnerStorageInventoryV1
    ) -> None:
        if self._projection_inventory is not None:
            raise ValueError("projection storage inventory was already recorded")
        self._projection_inventory = inventory

    def record_preparation_serialization(
        self, serialization: PreparationProjectionSerializationV1
    ) -> None:
        if self._serialization is not None:
            raise ValueError("preparation serialization was already recorded")
        self._serialization = serialization

    def record_executor_call(self) -> None:
        self._executor_call_count += 1

    def record_scientific_ticks(self, count: int) -> None:
        self._scientific_tick_count += _nonnegative(count, name="count")

    def finalize(self) -> ResidentPreparedOwnerResourceAuditReceiptV1:
        if tuple(sample.stage for sample in self._samples) != PREPARE_ONLY_STAGES:
            raise ValueError(
                "all prepare-only stages must be recorded before finalization"
            )
        if self._inventory is None:
            raise ValueError("storage inventory is required")
        if self._serialization is None:
            raise ValueError("preparation serialization is required")
        if self._executor_call_count or self._scientific_tick_count:
            raise ValueError(
                "audit counters record dispatch or ticks and cannot produce a receipt"
            )

        decision = derive_prepared_owner_resource_decision_v1(
            self.request,
            self._samples,
            self._inventory,
            self._projection_inventory,
        )
        return ResidentPreparedOwnerResourceAuditReceiptV1(
            request=self.request,
            stage_samples=tuple(self._samples),
            storage_inventory=self._inventory,
            preparation_serialization=self._serialization,
            executor_call_count=0,
            scientific_tick_count=0,
            decision=decision,
            projection_storage_inventory=self._projection_inventory,
        )


def cuda_memory_stage_sampler_v1(
    torch_module: Any,
    *,
    expected_hardware_identity: Mapping[str, Any] | None = None,
    clock_ns: Callable[[], int] = time.perf_counter_ns,
) -> Callable[[str], CudaMemorySampleV1]:
    """Create a synchronized CUDA sampler with one monotonic audit epoch."""

    cuda = getattr(torch_module, "cuda", None)
    if cuda is None or not bool(cuda.is_available()):
        raise ValueError("prepared-owner resource audit requires CUDA")
    expected = (
        None
        if expected_hardware_identity is None
        else validate_cuda_hardware_identity_v1(expected_hardware_identity)
    )

    def verify_device() -> None:
        if expected is None:
            return
        index = int(cuda.current_device())
        properties = cuda.get_device_properties(index)
        actual = validate_cuda_hardware_identity_v1(
            {
                "device_index": index,
                "device_name": str(properties.name),
                "device_uuid": str(getattr(properties, "uuid", "")),
                "compute_capability": [
                    int(properties.major),
                    int(properties.minor),
                ],
                "total_memory_bytes": int(properties.total_memory),
                "cuda_runtime_version": str(torch_module.version.cuda),
                "torch_version": str(torch_module.__version__),
            }
        )
        if actual != expected:
            raise ValueError("CUDA sampler device differs from authorization")

    verify_device()
    started = clock_ns()
    cuda.synchronize()
    cuda.reset_peak_memory_stats()

    def sample(stage: str) -> CudaMemorySampleV1:
        verify_device()
        cuda.synchronize()
        free_bytes, total_bytes = cuda.mem_get_info()
        return CudaMemorySampleV1(
            stage=stage,
            elapsed_ns=max(0, clock_ns() - started),
            allocated_bytes=int(cuda.memory_allocated()),
            reserved_bytes=int(cuda.memory_reserved()),
            peak_allocated_bytes=int(cuda.max_memory_allocated()),
            peak_reserved_bytes=int(cuda.max_memory_reserved()),
            free_bytes=int(free_bytes),
            total_bytes=int(total_bytes),
        )

    return sample


def execute_prepared_owner_resource_audit_v1(
    *,
    request_factory: Callable[[str], ResidentPreparedOwnerResourceAuditRequestV1],
    module_loader: Callable[[], Any],
    owner_preparer: Callable[[Any], Any],
    projection_exporter: Callable[[Any], tuple[Any, str]],
    projection_destination: Path,
    projection_format_id: str,
    projection_serializer: Callable[[Any, Path], None],
    stage_sampler: Callable[[str], CudaMemorySampleV1],
    resource_releaser: Callable[[Any, Any], None] | None = None,
    post_release_finalizer: Callable[[], None] | None = None,
) -> ResidentPreparedOwnerResourceAuditReceiptV1:
    """Measure one real prepared owner while exposing no execution callback."""

    samples = [stage_sampler("process_baseline")]
    module: Any = None
    owner: Any = None
    projection: Any = None
    try:
        module = module_loader()
        samples.append(stage_sampler("module_loaded"))
        owner = owner_preparer(module)
        samples.append(stage_sampler("prepared_owner"))
        inventory = inventory_resident_prepared_owner_storages_v1(owner)
        projection, projection_sha256 = projection_exporter(owner)
        projection_inventory = inventory_resident_prepared_owner_storages_v1(
            projection
        )
        request = request_factory(projection_sha256)
        samples.append(stage_sampler("preparation_projection"))
        serialization = serialize_preparation_projection_v1(
            projection,
            projection_sha256=projection_sha256,
            destination=projection_destination,
            format_id=projection_format_id,
            serializer=projection_serializer,
        )
        samples.append(stage_sampler("serialized_projection"))
    finally:
        try:
            if resource_releaser is not None:
                resource_releaser(owner, module)
        finally:
            # Release only references owned by this audit lifecycle. Mutating
            # containers reachable from the owner would corrupt projections
            # or configuration objects that may also be held by the caller.
            owner = None
            module = None
            projection = None
            gc.collect()
            if post_release_finalizer is not None:
                post_release_finalizer()
    samples.append(stage_sampler("teardown"))

    recorder = ResidentPreparedOwnerAuditStageRecorderV1(request)
    for sample in samples:
        recorder.record_sample(sample)
    recorder.record_storage_inventory(inventory)
    recorder.record_projection_storage_inventory(projection_inventory)
    recorder.record_preparation_serialization(serialization)
    return recorder.finalize()


def _exact_keys(
    value: Any,
    expected: set[str],
    *,
    name: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ValueError(f"{name} does not have the exact required fields")
    return dict(value)


def _deserialize_request_v1(
    value: Any,
) -> ResidentPreparedOwnerResourceAuditRequestV1:
    row = _exact_keys(
        value,
        {
            "schema_version",
            "audit_id",
            "provider_id",
            "profile_id",
            "campaign_id",
            "plan_sha256",
            "qualification_key_sha256",
            "execution_route_sha256",
            "route_qualification",
            "candidate_authority_sha256",
            "artifact_sha256s",
            "configuration_sha256",
            "resource_estimator_id",
            "conservative_vram_projection_bytes",
            "runtime_assumptions",
            "measurement_authorization_sha256",
            "hardware_identity",
            "source_projection_root_sha256",
            "ordered_cell_ids_sha256",
            "batch_request_sha256",
            "dimensions",
            "policy",
        },
        name="prepared-owner request",
    )
    if row.pop("schema_version") != REQUEST_SCHEMA_VERSION:
        raise ValueError("prepared-owner request schema is unsupported")
    dimensions = _exact_keys(
        row.pop("dimensions"),
        {
            "batch_cells",
            "resident_segment_generations",
            "physical_window_generations",
            "ticks_per_generation",
            "resident_capacity",
            "initial_population",
            "agent_id_allocation_generation_count",
        },
        name="prepared-owner request dimensions",
    )
    policy = _exact_keys(
        row.pop("policy"),
        {
            "declared_vram_budget_bytes",
            "minimum_physical_headroom_fraction",
            "minimum_physical_headroom_bytes",
            "minimum_residual_headroom_bytes",
            "teardown_tolerance_bytes",
        },
        name="prepared-owner request policy",
    )
    request = ResidentPreparedOwnerResourceAuditRequestV1(
        **row,
        **dimensions,
        **policy,
    )
    if request.payload() != value:
        raise ValueError("prepared-owner request is not canonical")
    return request


def _deserialize_stage_samples_v1(value: Any) -> tuple[CudaMemorySampleV1, ...]:
    if not isinstance(value, list) or len(value) != len(PREPARE_ONLY_STAGES):
        raise ValueError("prepared-owner receipt stage count is invalid")
    fields = {
        "schema_version",
        "stage",
        "elapsed_ns",
        "allocated_bytes",
        "reserved_bytes",
        "peak_allocated_bytes",
        "peak_reserved_bytes",
        "free_bytes",
        "total_bytes",
    }
    samples: list[CudaMemorySampleV1] = []
    for raw in value:
        row = _exact_keys(raw, fields, name="prepared-owner CUDA stage")
        if row.pop("schema_version") != STAGE_SAMPLE_SCHEMA_VERSION:
            raise ValueError("prepared-owner CUDA stage schema is unsupported")
        sample = CudaMemorySampleV1(**row)
        if sample.payload() != raw:
            raise ValueError("prepared-owner CUDA stage is not canonical")
        samples.append(sample)
    return tuple(samples)


def _deserialize_storage_inventory_v1(
    value: Any,
) -> ResidentPreparedOwnerStorageInventoryV1:
    row = _exact_keys(
        value,
        {
            "schema_version",
            "attribution_schema_version",
            "tensor_view_count",
            "unique_storage_count",
            "retained_storage_bytes",
            "device_storage_bytes",
            "storage_rows",
            "tensor_view_rows",
        },
        name="prepared-owner storage inventory",
    )
    if row.pop("schema_version") != STORAGE_INVENTORY_SCHEMA_VERSION:
        raise ValueError("prepared-owner storage inventory schema is unsupported")
    if row.pop("attribution_schema_version") != STORAGE_ATTRIBUTION_SCHEMA_VERSION:
        raise ValueError("prepared-owner storage attribution schema is unsupported")
    storage_rows = row.get("storage_rows")
    if not isinstance(storage_rows, list) or not storage_rows:
        raise ValueError("prepared-owner storage inventory rows are required")
    for raw in storage_rows:
        storage = _exact_keys(
            raw,
            {
                "alias_group_id",
                "data_ptr",
                "storage_data_ptr",
                "device",
                "nbytes",
                "storage_nbytes",
                "dtype",
                "shape",
                "encoding",
                "view_count",
                "paths",
                "logical_bytes_total",
            },
            name="prepared-owner storage row",
        )
        if (
            not isinstance(storage["alias_group_id"], str)
            or not storage["alias_group_id"]
            or type(storage["storage_data_ptr"]) is not int
            or storage["storage_data_ptr"] < 0
            or type(storage["data_ptr"]) is not int
            or storage["data_ptr"] < 0
            or not isinstance(storage["device"], str)
            or not storage["device"]
            or type(storage["nbytes"]) is not int
            or storage["nbytes"] <= 0
            or type(storage["storage_nbytes"]) is not int
            or storage["storage_nbytes"] != storage["nbytes"]
            or not isinstance(storage["dtype"], str)
            or not storage["dtype"]
            or not (
                storage["shape"] is None
                or (
                    isinstance(storage["shape"], list)
                    and all(type(item) is int and item >= 0 for item in storage["shape"])
                )
            )
            or storage["encoding"] != "dense_tensor_storage_v1"
            or type(storage["view_count"]) is not int
            or storage["view_count"] <= 0
            or not isinstance(storage["paths"], list)
            or not storage["paths"]
            or any(not isinstance(path, str) or not path for path in storage["paths"])
            or type(storage["logical_bytes_total"]) is not int
            or storage["logical_bytes_total"] < 0
        ):
            raise ValueError("prepared-owner storage row is invalid")
    tensor_view_rows = row.get("tensor_view_rows")
    if not isinstance(tensor_view_rows, list) or not tensor_view_rows:
        raise ValueError("prepared-owner tensor view rows are required")
    for raw in tensor_view_rows:
        view = _exact_keys(
            raw,
            {
                "alias_group_id",
                "path",
                "device",
                "dtype",
                "shape",
                "logical_bytes",
                "data_ptr",
                "storage_data_ptr",
                "storage_nbytes",
                "storage_offset",
                "stride",
                "contiguous",
                "python_object_id",
                "has_base_view",
                "owner_lifetime_category",
                "creation_stage",
                "last_use_stage",
                "retained_after_stage",
            },
            name="prepared-owner tensor view row",
        )
        if (
            not isinstance(view["alias_group_id"], str)
            or not view["alias_group_id"]
            or not isinstance(view["path"], str)
            or not view["path"]
            or not isinstance(view["device"], str)
            or not view["device"]
            or not isinstance(view["dtype"], str)
            or not view["dtype"]
            or not (
                view["shape"] is None
                or (
                    isinstance(view["shape"], list)
                    and all(type(item) is int and item >= 0 for item in view["shape"])
                )
            )
            or not (view["logical_bytes"] is None or type(view["logical_bytes"]) is int)
            or not (view["data_ptr"] is None or type(view["data_ptr"]) is int)
            or type(view["storage_data_ptr"]) is not int
            or view["storage_data_ptr"] < 0
            or type(view["storage_nbytes"]) is not int
            or view["storage_nbytes"] <= 0
            or not (view["storage_offset"] is None or type(view["storage_offset"]) is int)
            or not (
                view["stride"] is None
                or (
                    isinstance(view["stride"], list)
                    and all(type(item) is int for item in view["stride"])
                )
            )
            or not (view["contiguous"] is None or type(view["contiguous"]) is bool)
            or type(view["python_object_id"]) is not int
            or type(view["has_base_view"]) is not bool
            or not isinstance(view["owner_lifetime_category"], str)
            or not view["owner_lifetime_category"]
            or view["creation_stage"] != "prepared_owner_inventory"
            or view["last_use_stage"] != "unknown_prepare_only"
            or view["retained_after_stage"] != "prepared_owner"
        ):
            raise ValueError("prepared-owner tensor view row is invalid")
    inventory = ResidentPreparedOwnerStorageInventoryV1(**row)
    row_device_bytes: dict[str, int] = {}
    storage_identities: set[tuple[str, int]] = set()
    for storage in storage_rows:
        identity = (str(storage["device"]), int(storage["data_ptr"]))
        if identity in storage_identities:
            raise ValueError("prepared-owner storage identities are not unique")
        storage_identities.add(identity)
        row_device_bytes[identity[0]] = (
            row_device_bytes.get(identity[0], 0) + int(storage["nbytes"])
        )
    if (
        inventory.tensor_view_count <= 0
        or inventory.unique_storage_count <= 0
        or inventory.retained_storage_bytes <= 0
        or len(storage_rows) != inventory.unique_storage_count
        or sum(int(storage["view_count"]) for storage in storage_rows)
        != inventory.tensor_view_count
        or row_device_bytes != inventory.device_storage_bytes
        or inventory.payload() != value
    ):
        raise ValueError("prepared-owner storage inventory is not canonical")
    return inventory


def _deserialize_preparation_serialization_v1(
    value: Any,
) -> PreparationProjectionSerializationV1:
    row = _exact_keys(
        value,
        {
            "schema_version",
            "projection_sha256",
            "serialized_path",
            "serialized_sha256",
            "serialized_bytes",
            "serialization_time_ns",
            "format_id",
        },
        name="prepared-owner preparation serialization",
    )
    if row.pop("schema_version") != SERIALIZATION_SCHEMA_VERSION:
        raise ValueError("prepared-owner serialization schema is unsupported")
    serialization = PreparationProjectionSerializationV1(**row)
    serialized_path = Path(serialization.serialized_path)
    if (
        not serialized_path.is_file()
        or serialized_path.is_symlink()
        or serialization.serialized_bytes <= 0
        or serialized_path.stat().st_size != serialization.serialized_bytes
        or hashlib.sha256(serialized_path.read_bytes()).hexdigest()
        != serialization.serialized_sha256
        or serialization.payload() != value
    ):
        raise ValueError("prepared-owner serialization is not canonical")
    return serialization


def validate_prepared_owner_resource_audit_receipt_v1(
    receipt: Mapping[str, Any],
    *,
    expected_request_bindings: Mapping[str, Any],
    live_hardware_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate an exact prepare-only receipt for preflight promotion."""

    exact_receipt = _exact_keys(
        receipt,
        {
            "schema_version",
            "request",
            "request_sha256",
            "stage_samples",
            "storage_inventory",
            "projection_storage_inventory",
            "preparation_serialization",
            "executor_call_count",
            "scientific_tick_count",
            "decision",
            "receipt_sha256",
        },
        name="prepared-owner receipt",
    )
    if exact_receipt.get("schema_version") != RECEIPT_SCHEMA_VERSION:
        raise ValueError("prepared-owner receipt schema is unsupported")
    unsigned = {
        key: value
        for key, value in exact_receipt.items()
        if key != "receipt_sha256"
    }
    if exact_receipt.get("receipt_sha256") != canonical_sha256(unsigned):
        raise ValueError("prepared-owner receipt SHA-256 mismatch")
    request = _deserialize_request_v1(exact_receipt["request"])
    if exact_receipt.get("request_sha256") != request.request_sha256:
        raise ValueError("prepared-owner request SHA-256 mismatch")
    for key, expected in expected_request_bindings.items():
        if key == "policy" and isinstance(expected, Mapping):
            expected = {
                policy_key: policy_value
                for policy_key, policy_value in expected.items()
                if policy_key != "measurement_live_free_policy"
            }
        if request.payload().get(key) != expected:
            raise ValueError(f"prepared-owner request binding mismatch: {key}")

    hardware = _hardware_identity(live_hardware_identity)
    if request.hardware_identity != hardware:
        raise ValueError("prepared-owner receipt hardware identity mismatch")
    stages = _deserialize_stage_samples_v1(exact_receipt["stage_samples"])
    inventory = _deserialize_storage_inventory_v1(
        exact_receipt["storage_inventory"]
    )
    projection_inventory = (
        None
        if exact_receipt["projection_storage_inventory"] is None
        else _deserialize_storage_inventory_v1(
            exact_receipt["projection_storage_inventory"]
        )
    )
    serialization = _deserialize_preparation_serialization_v1(
        exact_receipt["preparation_serialization"]
    )
    if serialization.projection_sha256 != request.source_projection_root_sha256:
        raise ValueError("prepared-owner serialized projection binding mismatch")
    if exact_receipt.get("executor_call_count") != 0:
        raise ValueError("prepared-owner receipt counter records executor calls")
    if exact_receipt.get("scientific_tick_count") != 0:
        raise ValueError("prepared-owner receipt counter records scientific ticks")
    decision = exact_receipt.get("decision")
    if not isinstance(decision, Mapping):
        raise ValueError("prepared-owner receipt decision is missing")
    derived_decision = derive_prepared_owner_resource_decision_v1(
        request,
        stages,
        inventory,
        projection_inventory,
    )
    if dict(decision) != derived_decision:
        raise ValueError("prepared-owner receipt decision differs from evidence")
    if derived_decision["ok"] is not True:
        raise ValueError("prepared-owner receipt decision is not promotable")
    projection = request.conservative_vram_projection_bytes
    measured = derived_decision["measured_owner_peak_bytes"]
    return {
        "receipt_sha256": str(exact_receipt["receipt_sha256"]),
        "request_sha256": str(exact_receipt["request_sha256"]),
        "vram_bytes_upper_bound": projection,
        "measured_owner_peak_bytes": measured,
        "hardware_identity": hardware,
        "evidence_scope": "structural_zero_dispatch_zero_tick_counter_v1",
    }


__all__ = [
    "CudaMemorySampleV1",
    "LIFETIME_ANALYSIS_SCHEMA_VERSION",
    "MEASUREMENT_AUTHORIZATION_SCHEMA_VERSION",
    "PREPARE_ONLY_STAGES",
    "PreparationProjectionSerializationV1",
    "ResidentPreparedOwnerAuditStageRecorderV1",
    "ResidentPreparedOwnerResourceAuditReceiptV1",
    "ResidentPreparedOwnerResourceAuditRequestV1",
    "ResidentPreparedOwnerStorageInventoryV1",
    "inventory_resident_prepared_owner_storages_v1",
    "cuda_memory_stage_sampler_v1",
    "derive_prepared_owner_lifetime_analysis_v1",
    "derive_prepared_owner_resource_decision_v1",
    "detect_cuda_hardware_identity_v1",
    "execute_prepared_owner_resource_audit_v1",
    "prepared_owner_configuration_sha256_v1",
    "serialize_preparation_projection_v1",
    "validate_cuda_hardware_identity_v1",
    "validate_prepared_owner_resource_audit_receipt_v1",
]
