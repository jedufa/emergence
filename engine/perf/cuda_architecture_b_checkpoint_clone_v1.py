from __future__ import annotations

"""Device-neutral contract and host planner for resident checkpoint clones.

The planner only describes device-to-device byte copies at a completed
generation boundary. It does not import torch, launch CUDA, or inspect device
contents.
"""

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from hashlib import sha256
from math import prod
from pathlib import Path
from typing import Any, Final


CHECKPOINT_CLONE_PLAN_SCHEMA_VERSION_V1: Final[str] = (
    "eiel_cuda_architecture_b_checkpoint_clone_plan_v1"
)
CHECKPOINT_CLONE_ZERO_D2H_RECEIPT_SCHEMA_VERSION_V1: Final[str] = (
    "eiel_cuda_architecture_b_checkpoint_clone_zero_d2h_receipt_v1"
)
CHECKPOINT_CLONE_STATE_MANIFEST_SCHEMA_VERSION_V1: Final[str] = (
    "eiel_cuda_architecture_b_checkpoint_clone_state_manifest_v1"
)

STATE_SEMANTIC_CLASSIFICATION_V1: Final[str] = "state"
LINEAGE_SEMANTIC_CLASSIFICATION_V1: Final[str] = "lineage"
RNG_SEMANTIC_CLASSIFICATION_V1: Final[str] = "rng"

STATE_EXACT_COPY_CLASSIFICATION_V1: Final[str] = "state_exact_copy"
LINEAGE_EXACT_COPY_CLASSIFICATION_V1: Final[str] = "lineage_exact_copy"
RNG_EXACT_COPY_CLASSIFICATION_V1: Final[str] = "rng_exact_copy"
EXACT_COPY_CLASSIFICATIONS_V1: Final[frozenset[str]] = frozenset(
    {
        STATE_EXACT_COPY_CLASSIFICATION_V1,
        LINEAGE_EXACT_COPY_CLASSIFICATION_V1,
        RNG_EXACT_COPY_CLASSIFICATION_V1,
    }
)

_COPY_CLASSIFICATION_BY_SEMANTIC: Final[dict[str, str]] = {
    STATE_SEMANTIC_CLASSIFICATION_V1: STATE_EXACT_COPY_CLASSIFICATION_V1,
    LINEAGE_SEMANTIC_CLASSIFICATION_V1: LINEAGE_EXACT_COPY_CLASSIFICATION_V1,
    RNG_SEMANTIC_CLASSIFICATION_V1: RNG_EXACT_COPY_CLASSIFICATION_V1,
}
_ADDRESS_SPACE_SIZE: Final[int] = 1 << 64
_MAX_POINTER: Final[int] = _ADDRESS_SPACE_SIZE - 1
ACCEPTED_STATE_BANK_MANIFEST_SHA256_V1: Final[str] = (
    "e7981dc8d602f3a25b627881690ccdb231d82b7aaa98e4fdcd6afda6d715dfce"
)
_ACCEPTED_STATE_BANK_MANIFEST_PATH: Final[Path] = Path(__file__).with_name(
    "flat_patch_device_neutral_state_manifest_v4.json"
)
_ACCEPTED_STATE_BANK_SCHEMA_VERSION: Final[str] = (
    "eiel_flat_patch_device_neutral_state_manifest_v4"
)
_ACCEPTED_STATE_BANK_CONTRACT_STATUS: Final[str] = (
    "accepted_repository_bundled_flat_patch_state_bank_v4"
)
_DTYPE_BYTES: Final[dict[str, int]] = {
    "bool": 1,
    "float32": 4,
    "float64": 8,
    "int8": 1,
    "int32": 4,
    "int64": 8,
    "uint8": 1,
}
_STATE_MANIFEST_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "complete",
        "accepted_state_bank_manifest_sha256",
        "shape_bindings",
        "fields",
    }
)
_STATE_MANIFEST_FIELD_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "field_name",
        "shape",
        "element_size_bytes",
        "semantic_classification",
        "copy_classification",
    }
)

CHECKPOINT_CLONE_ZERO_D2H_RECEIPT_REQUIRED_FIELDS_V1: Final[
    tuple[str, ...]
] = (
    "schema_version",
    "plan_sha256",
    "generation_boundary_complete",
    "source_cell_index",
    "target_cell_indices",
    "physical_field_count",
    "expected_copy_operations",
    "expected_copy_bytes",
    "executed_copy_operations",
    "executed_copy_bytes",
    "device_to_host_operations",
    "device_to_host_bytes",
    "source_preserved",
    "hidden_host_materialization",
    "execution_performed",
    "zero_d2h_qualified",
    "simulation_parity_claimed",
)


class CheckpointCloneContractError(ValueError):
    """Raised when a clone request is unsafe or structurally ambiguous."""


@dataclass(frozen=True)
class CheckpointCloneFieldDescriptorV1:
    """One logical name for a cell-major resident physical allocation."""

    field_name: str
    pointer_identity: int
    shape: tuple[int, ...]
    cell_stride: int
    bytes_per_cell: int
    element_size_bytes: int
    semantic_classification: str | None = None
    copy_classification: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "field_name": self.field_name,
            "pointer_identity": self.pointer_identity,
            "shape": list(self.shape),
            "cell_stride": self.cell_stride,
            "bytes_per_cell": self.bytes_per_cell,
            "element_size_bytes": self.element_size_bytes,
            "semantic_classification": self.semantic_classification,
            "copy_classification": self.copy_classification,
        }


@dataclass(frozen=True)
class PhysicalCheckpointCloneFieldV1:
    """A validated physical allocation after logical alias deduplication."""

    physical_field_index: int
    field_name: str
    field_aliases: tuple[str, ...]
    pointer_identity: int
    shape: tuple[int, ...]
    cell_stride: int
    bytes_per_cell: int
    element_size_bytes: int
    allocation_end_exclusive: int
    semantic_classification: str
    copy_classification: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "physical_field_index": self.physical_field_index,
            "field_name": self.field_name,
            "field_aliases": list(self.field_aliases),
            "pointer_identity": self.pointer_identity,
            "shape": list(self.shape),
            "cell_stride": self.cell_stride,
            "bytes_per_cell": self.bytes_per_cell,
            "element_size_bytes": self.element_size_bytes,
            "allocation_end_exclusive": self.allocation_end_exclusive,
            "semantic_classification": self.semantic_classification,
            "copy_classification": self.copy_classification,
        }


@dataclass(frozen=True)
class CheckpointCloneEntryV1:
    """One future device-to-device copy from the source cell to one target."""

    clone_entry_index: int
    physical_field_index: int
    field_name: str
    field_aliases: tuple[str, ...]
    semantic_classification: str
    copy_classification: str
    source_cell_index: int
    target_cell_index: int
    source_pointer: int
    target_pointer: int
    bytes_to_copy: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "clone_entry_index": self.clone_entry_index,
            "physical_field_index": self.physical_field_index,
            "field_name": self.field_name,
            "field_aliases": list(self.field_aliases),
            "semantic_classification": self.semantic_classification,
            "copy_classification": self.copy_classification,
            "source_cell_index": self.source_cell_index,
            "target_cell_index": self.target_cell_index,
            "source_pointer": self.source_pointer,
            "target_pointer": self.target_pointer,
            "bytes_to_copy": self.bytes_to_copy,
        }


@dataclass(frozen=True, init=False)
class CheckpointClonePlanV1:
    state_manifest_sha256: str
    required_field_names: tuple[str, ...]
    generation_boundary_complete: bool
    source_cell_index: int
    target_cell_indices: tuple[int, ...]
    logical_field_count: int
    physical_fields: tuple[PhysicalCheckpointCloneFieldV1, ...]
    entries: tuple[CheckpointCloneEntryV1, ...]
    total_copy_bytes: int
    source_preserved_by_contract: bool = True
    cuda_execution_performed: bool = False
    simulation_parity_claimed: bool = False

    @classmethod
    def _from_validated(
        cls,
        *,
        state_manifest_sha256: str,
        required_field_names: tuple[str, ...],
        generation_boundary_complete: bool,
        source_cell_index: int,
        target_cell_indices: tuple[int, ...],
        logical_field_count: int,
        physical_fields: tuple[PhysicalCheckpointCloneFieldV1, ...],
        entries: tuple[CheckpointCloneEntryV1, ...],
        total_copy_bytes: int,
    ) -> CheckpointClonePlanV1:
        value = object.__new__(cls)
        for name, child in {
            "state_manifest_sha256": state_manifest_sha256,
            "required_field_names": required_field_names,
            "generation_boundary_complete": generation_boundary_complete,
            "source_cell_index": source_cell_index,
            "target_cell_indices": target_cell_indices,
            "logical_field_count": logical_field_count,
            "physical_fields": physical_fields,
            "entries": entries,
            "total_copy_bytes": total_copy_bytes,
            "source_preserved_by_contract": True,
            "cuda_execution_performed": False,
            "simulation_parity_claimed": False,
        }.items():
            object.__setattr__(value, name, child)
        return value

    @property
    def physical_field_count(self) -> int:
        return len(self.physical_fields)

    @property
    def clone_entry_count(self) -> int:
        return len(self.entries)

    def _payload(self) -> dict[str, Any]:
        return {
            "schema_version": CHECKPOINT_CLONE_PLAN_SCHEMA_VERSION_V1,
            "state_manifest_sha256": self.state_manifest_sha256,
            "required_field_names": list(self.required_field_names),
            "generation_boundary_complete": self.generation_boundary_complete,
            "source_cell_index": self.source_cell_index,
            "target_cell_indices": list(self.target_cell_indices),
            "logical_field_count": self.logical_field_count,
            "physical_field_count": self.physical_field_count,
            "clone_entry_count": self.clone_entry_count,
            "total_copy_bytes": self.total_copy_bytes,
            "physical_fields": [field.to_dict() for field in self.physical_fields],
            "entries": [entry.to_dict() for entry in self.entries],
            "source_preservation_contract": {
                "source_cell_index": self.source_cell_index,
                "source_cell_is_read_only": True,
                "source_write_entry_count": 0,
                "targets_exclude_source": True,
                "source_preserved_by_contract": self.source_preserved_by_contract,
            },
            "manifest_completeness_contract": {
                "authority_manifest_sha256": (
                    ACCEPTED_STATE_BANK_MANIFEST_SHA256_V1
                ),
                "required_logical_field_count": len(self.required_field_names),
                "provided_logical_field_count": self.logical_field_count,
                "accepted_tensor_state_complete": True,
                "accepted_non_tensor_state_surface_count": len(
                    _accepted_state_bank_contract_v1()["non_tensor_field_names"]
                ),
                "non_tensor_state_device_clone_qualified": False,
                "full_semantic_checkpoint_clone_qualified": False,
                "field_sets_exact": True,
                "shape_byte_widths_exact": True,
            },
            "transfer_contract": {
                "copy_kind": "device_to_device",
                "device_to_host_operations": 0,
                "device_to_host_bytes": 0,
                "hidden_host_materialization": False,
            },
            "zero_d2h_receipt_schema_version": (
                CHECKPOINT_CLONE_ZERO_D2H_RECEIPT_SCHEMA_VERSION_V1
            ),
            "cuda_execution_performed": self.cuda_execution_performed,
            "simulation_parity_claimed": self.simulation_parity_claimed,
        }

    @property
    def plan_sha256(self) -> str:
        return _canonical_sha256(self._payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self._payload()
        payload["plan_sha256"] = self.plan_sha256
        return payload


@dataclass(frozen=True, init=False)
class CheckpointCloneZeroD2HReceiptV1:
    """Execution receipt shape for a future CUDA clone implementation."""

    plan_sha256: str
    generation_boundary_complete: bool
    source_cell_index: int
    target_cell_indices: tuple[int, ...]
    physical_field_count: int
    expected_copy_operations: int
    expected_copy_bytes: int
    executed_copy_operations: int
    executed_copy_bytes: int
    device_to_host_operations: int
    device_to_host_bytes: int
    source_preserved: bool
    hidden_host_materialization: bool
    execution_performed: bool
    simulation_parity_claimed: bool = False

    @classmethod
    def _from_planning(
        cls,
        *,
        plan_sha256: str,
        generation_boundary_complete: bool,
        source_cell_index: int,
        target_cell_indices: tuple[int, ...],
        physical_field_count: int,
        expected_copy_operations: int,
        expected_copy_bytes: int,
    ) -> CheckpointCloneZeroD2HReceiptV1:
        value = object.__new__(cls)
        for name, child in {
            "plan_sha256": plan_sha256,
            "generation_boundary_complete": generation_boundary_complete,
            "source_cell_index": source_cell_index,
            "target_cell_indices": target_cell_indices,
            "physical_field_count": physical_field_count,
            "expected_copy_operations": expected_copy_operations,
            "expected_copy_bytes": expected_copy_bytes,
            "executed_copy_operations": 0,
            "executed_copy_bytes": 0,
            "device_to_host_operations": 0,
            "device_to_host_bytes": 0,
            "source_preserved": False,
            "hidden_host_materialization": False,
            "execution_performed": False,
            "simulation_parity_claimed": False,
        }.items():
            object.__setattr__(value, name, child)
        return value

    @property
    def zero_d2h_qualified(self) -> bool:
        # This host-only module cannot observe or authenticate CUDA execution.
        return False

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema_version": (
                CHECKPOINT_CLONE_ZERO_D2H_RECEIPT_SCHEMA_VERSION_V1
            ),
            "plan_sha256": self.plan_sha256,
            "generation_boundary_complete": self.generation_boundary_complete,
            "source_cell_index": self.source_cell_index,
            "target_cell_indices": list(self.target_cell_indices),
            "physical_field_count": self.physical_field_count,
            "expected_copy_operations": self.expected_copy_operations,
            "expected_copy_bytes": self.expected_copy_bytes,
            "executed_copy_operations": self.executed_copy_operations,
            "executed_copy_bytes": self.executed_copy_bytes,
            "device_to_host_operations": self.device_to_host_operations,
            "device_to_host_bytes": self.device_to_host_bytes,
            "source_preserved": self.source_preserved,
            "hidden_host_materialization": self.hidden_host_materialization,
            "execution_performed": self.execution_performed,
            "zero_d2h_qualified": self.zero_d2h_qualified,
            "simulation_parity_claimed": self.simulation_parity_claimed,
        }
        if tuple(payload) != CHECKPOINT_CLONE_ZERO_D2H_RECEIPT_REQUIRED_FIELDS_V1:
            raise AssertionError("zero-D2H receipt schema field order drift")
        return payload


@dataclass(frozen=True)
class _NormalizedField:
    field_name: str
    pointer_identity: int
    shape: tuple[int, ...]
    cell_stride: int
    bytes_per_cell: int
    element_size_bytes: int
    allocation_end_exclusive: int
    semantic_classification: str
    copy_classification: str


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return sha256(encoded).hexdigest()


def _fail(path: str, message: str) -> None:
    raise CheckpointCloneContractError(f"{path}: {message}")


def _required_int(value: Any, *, path: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        _fail(path, f"must be an integer >= {minimum}")
    return value


@lru_cache(maxsize=1)
def _accepted_state_bank_contract_v1() -> dict[str, Any]:
    try:
        encoded = _ACCEPTED_STATE_BANK_MANIFEST_PATH.read_bytes()
        payload = json.loads(encoded)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            "accepted Flat/Patch state-bank manifest is unavailable"
        ) from exc
    actual_sha256 = sha256(encoded).hexdigest()
    if actual_sha256 != ACCEPTED_STATE_BANK_MANIFEST_SHA256_V1:
        raise RuntimeError(
            "accepted Flat/Patch state-bank manifest hash drift: "
            f"expected {ACCEPTED_STATE_BANK_MANIFEST_SHA256_V1}, "
            f"got {actual_sha256}"
        )
    if payload.get("schema_version") != _ACCEPTED_STATE_BANK_SCHEMA_VERSION:
        raise RuntimeError("accepted Flat/Patch state-bank schema drift")
    if payload.get("contract_status") != _ACCEPTED_STATE_BANK_CONTRACT_STATUS:
        raise RuntimeError("accepted Flat/Patch state-bank status drift")
    surfaces = payload.get("tensor_surfaces")
    non_tensor = payload.get("non_tensor_state_surfaces")
    if not isinstance(surfaces, list) or not isinstance(non_tensor, list):
        raise RuntimeError("accepted Flat/Patch state-bank surface registry drift")

    fields: dict[str, dict[str, Any]] = {}
    shape_variables: set[str] = set()
    for index, surface in enumerate(surfaces):
        if not isinstance(surface, Mapping):
            raise RuntimeError(
                f"accepted state-bank tensor surface {index} is malformed"
            )
        name = surface.get("canonical_name")
        dtype = surface.get("dtype")
        variables = surface.get("device_neutral_shape_variables")
        rank = surface.get("device_neutral_rank")
        if (
            not isinstance(name, str)
            or name in fields
            or dtype not in _DTYPE_BYTES
            or not isinstance(variables, list)
            or not variables
            or any(not isinstance(value, str) or not value for value in variables)
            or type(rank) is not int
            or rank != len(variables)
            or variables[0] != "cell"
        ):
            raise RuntimeError(
                f"accepted state-bank tensor surface {index} is malformed"
            )
        shape_variables.update(variables)
        fields[name] = {
            "shape_variables": tuple(variables),
            "element_size_bytes": _DTYPE_BYTES[dtype],
        }
    non_tensor_names: list[str] = []
    for index, surface in enumerate(non_tensor):
        if not isinstance(surface, Mapping):
            raise RuntimeError(
                f"accepted state-bank non-tensor surface {index} is malformed"
            )
        name = surface.get("canonical_name")
        if not isinstance(name, str) or not name or name in non_tensor_names:
            raise RuntimeError(
                f"accepted state-bank non-tensor surface {index} is malformed"
            )
        non_tensor_names.append(name)
    if len(fields) != 139 or len(non_tensor_names) != 21:
        raise RuntimeError("accepted Flat/Patch state-bank field-count drift")
    return {
        "fields": fields,
        "shape_variables": tuple(sorted(shape_variables)),
        "non_tensor_field_names": tuple(sorted(non_tensor_names)),
    }


def build_accepted_checkpoint_clone_state_manifest_v1(
    shape_bindings: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the only accepted concrete tensor manifest for clone planning."""
    authority = _accepted_state_bank_contract_v1()
    if not isinstance(shape_bindings, Mapping):
        _fail("shape_bindings", "must be a mapping")
    required_variables = authority["shape_variables"]
    if set(shape_bindings) != set(required_variables):
        missing = sorted(set(required_variables) - set(shape_bindings))
        unexpected = sorted(set(shape_bindings) - set(required_variables))
        _fail(
            "shape_bindings",
            f"must exactly cover the accepted variables; "
            f"missing={missing}, unexpected={unexpected}",
        )
    normalized_bindings = {
        name: _required_int(shape_bindings[name], path=f"shape_bindings.{name}", minimum=1)
        for name in required_variables
    }
    fields: list[dict[str, Any]] = []
    for name, contract in sorted(authority["fields"].items()):
        semantic, copy_classification = classify_checkpoint_clone_field_v1(name)
        fields.append(
            {
                "field_name": name,
                "shape": [
                    normalized_bindings[variable]
                    for variable in contract["shape_variables"]
                ],
                "element_size_bytes": contract["element_size_bytes"],
                "semantic_classification": semantic,
                "copy_classification": copy_classification,
            }
        )
    return {
        "schema_version": CHECKPOINT_CLONE_STATE_MANIFEST_SCHEMA_VERSION_V1,
        "complete": True,
        "accepted_state_bank_manifest_sha256": (
            ACCEPTED_STATE_BANK_MANIFEST_SHA256_V1
        ),
        "shape_bindings": normalized_bindings,
        "fields": fields,
    }


def _mapping_alias(
    value: Mapping[str, Any],
    primary: str,
    alias: str,
    *,
    path: str,
) -> Any:
    primary_present = primary in value
    alias_present = alias in value
    if primary_present and alias_present and value[primary] != value[alias]:
        _fail(path, f"contradictory {primary!r} and {alias!r} values")
    if primary_present:
        return value[primary]
    if alias_present:
        return value[alias]
    _fail(path, f"missing required field {primary!r}")


def classify_checkpoint_clone_field_v1(field_name: str) -> tuple[str, str]:
    """Return the semantic and mandatory exact-copy class for a field name."""
    if not isinstance(field_name, str) or not field_name.strip():
        _fail("field_name", "must be a nonempty string")
    normalized = field_name.strip().lower()
    if "lineage" in normalized:
        semantic = LINEAGE_SEMANTIC_CLASSIFICATION_V1
    elif any(
        token in normalized
        for token in (
            "rng",
            "random",
            "pcg64",
            "semantic_draw",
            "generator_state",
        )
    ):
        semantic = RNG_SEMANTIC_CLASSIFICATION_V1
    else:
        semantic = STATE_SEMANTIC_CLASSIFICATION_V1
    return semantic, _COPY_CLASSIFICATION_BY_SEMANTIC[semantic]


def _normalize_state_manifest_v1(
    raw: Mapping[str, Any],
) -> tuple[dict[str, Any], str, tuple[str, ...], dict[str, dict[str, Any]]]:
    if not isinstance(raw, Mapping):
        _fail("state_manifest", "must be a mapping")
    if set(raw) != _STATE_MANIFEST_FIELDS:
        _fail("state_manifest", "fields do not match the frozen v1 schema")
    if raw.get("schema_version") != CHECKPOINT_CLONE_STATE_MANIFEST_SCHEMA_VERSION_V1:
        _fail("state_manifest.schema_version", "is unsupported")
    if raw.get("complete") is not True:
        _fail("state_manifest.complete", "must be exactly true")
    if (
        raw.get("accepted_state_bank_manifest_sha256")
        != ACCEPTED_STATE_BANK_MANIFEST_SHA256_V1
    ):
        _fail(
            "state_manifest.accepted_state_bank_manifest_sha256",
            "does not match the immutable accepted Flat/Patch state-bank manifest",
        )
    expected_manifest = build_accepted_checkpoint_clone_state_manifest_v1(
        raw.get("shape_bindings")
    )
    normalized_bindings = expected_manifest["shape_bindings"]
    expected_by_name = {
        field["field_name"]: field for field in expected_manifest["fields"]
    }
    fields = raw.get("fields")
    if not isinstance(fields, Sequence) or isinstance(
        fields, (str, bytes, bytearray)
    ) or not fields:
        _fail("state_manifest.fields", "must be a nonempty sequence")

    normalized_fields: list[dict[str, Any]] = []
    names: set[str] = set()
    for index, field in enumerate(fields):
        path = f"state_manifest.fields[{index}]"
        if not isinstance(field, Mapping) or set(field) != _STATE_MANIFEST_FIELD_FIELDS:
            _fail(path, "fields do not match the frozen field schema")
        name = field.get("field_name")
        if not isinstance(name, str) or not name or name != name.strip():
            _fail(f"{path}.field_name", "must be a trimmed nonempty string")
        if name in names:
            _fail(f"{path}.field_name", "duplicates an earlier manifest field")
        names.add(name)
        shape_value = field.get("shape")
        if not isinstance(shape_value, Sequence) or isinstance(
            shape_value, (str, bytes, bytearray)
        ) or not shape_value:
            _fail(f"{path}.shape", "must be a nonempty integer sequence")
        shape = tuple(shape_value)
        for dimension, extent in enumerate(shape):
            _required_int(extent, path=f"{path}.shape[{dimension}]", minimum=1)
        element_size_bytes = _required_int(
            field.get("element_size_bytes"),
            path=f"{path}.element_size_bytes",
            minimum=1,
        )
        semantic = field.get("semantic_classification")
        if semantic not in _COPY_CLASSIFICATION_BY_SEMANTIC:
            _fail(f"{path}.semantic_classification", "is unsupported")
        copy_classification = field.get("copy_classification")
        expected_copy = _COPY_CLASSIFICATION_BY_SEMANTIC[semantic]
        if copy_classification != expected_copy:
            _fail(
                f"{path}.copy_classification",
                f"must be {expected_copy!r} for semantic class {semantic!r}",
            )
        normalized_fields.append(
            {
                "field_name": name,
                "shape": list(shape),
                "element_size_bytes": element_size_bytes,
                "semantic_classification": semantic,
                "copy_classification": copy_classification,
            }
        )
    normalized_fields.sort(key=lambda field: field["field_name"])
    normalized_by_name = {
        field["field_name"]: field for field in normalized_fields
    }
    if set(normalized_by_name) != set(expected_by_name):
        missing = sorted(set(expected_by_name) - set(normalized_by_name))
        unexpected = sorted(set(normalized_by_name) - set(expected_by_name))
        _fail(
            "state_manifest.fields",
            "does not exactly cover the immutable accepted state-bank fields; "
            f"missing={missing}, unexpected={unexpected}",
        )
    for name, expected in expected_by_name.items():
        if normalized_by_name[name] != expected:
            _fail(
                f"state_manifest.fields[{name!r}]",
                "contradicts the immutable accepted state-bank layout",
            )
    cell_counts = {field["shape"][0] for field in normalized_fields}
    if len(cell_counts) != 1:
        _fail(
            "state_manifest.fields",
            "all fields must expose the same leading cell count",
        )
    normalized = {
        "schema_version": CHECKPOINT_CLONE_STATE_MANIFEST_SCHEMA_VERSION_V1,
        "complete": True,
        "accepted_state_bank_manifest_sha256": (
            ACCEPTED_STATE_BANK_MANIFEST_SHA256_V1
        ),
        "shape_bindings": normalized_bindings,
        "fields": normalized_fields,
    }
    by_name = {field["field_name"]: field for field in normalized_fields}
    return (
        normalized,
        _canonical_sha256(normalized),
        tuple(sorted(by_name)),
        by_name,
    )


def _normalize_descriptor(
    raw: CheckpointCloneFieldDescriptorV1 | Mapping[str, Any],
    *,
    index: int,
) -> _NormalizedField:
    path = f"field_descriptors[{index}]"
    if isinstance(raw, CheckpointCloneFieldDescriptorV1):
        descriptor = raw
    elif isinstance(raw, Mapping):
        descriptor = CheckpointCloneFieldDescriptorV1(
            field_name=_mapping_alias(
                raw, "field_name", "name", path=f"{path}.field_name"
            ),
            pointer_identity=raw.get("pointer_identity"),
            shape=raw.get("shape"),
            cell_stride=_mapping_alias(
                raw,
                "cell_stride",
                "cell_stride_bytes",
                path=f"{path}.cell_stride",
            ),
            bytes_per_cell=raw.get("bytes_per_cell"),
            element_size_bytes=raw.get("element_size_bytes"),
            semantic_classification=raw.get("semantic_classification"),
            copy_classification=raw.get("copy_classification"),
        )
    else:
        _fail(path, "must be a field descriptor or mapping")

    if not isinstance(descriptor.field_name, str) or not descriptor.field_name.strip():
        _fail(f"{path}.field_name", "must be a nonempty string")
    field_name = descriptor.field_name.strip()
    pointer = _required_int(
        descriptor.pointer_identity,
        path=f"{path}.pointer_identity",
        minimum=1,
    )
    if pointer > _MAX_POINTER:
        _fail(f"{path}.pointer_identity", "overflows the uint64 pointer range")

    shape_value = descriptor.shape
    if not isinstance(shape_value, Sequence) or isinstance(
        shape_value, (str, bytes, bytearray)
    ):
        _fail(f"{path}.shape", "must be a nonempty integer sequence")
    shape = tuple(shape_value)
    if not shape:
        _fail(f"{path}.shape", "must be nonempty")
    for dimension, extent in enumerate(shape):
        _required_int(extent, path=f"{path}.shape[{dimension}]", minimum=1)

    cell_stride = _required_int(
        descriptor.cell_stride,
        path=f"{path}.cell_stride",
        minimum=1,
    )
    bytes_per_cell = _required_int(
        descriptor.bytes_per_cell,
        path=f"{path}.bytes_per_cell",
        minimum=1,
    )
    element_size_bytes = _required_int(
        descriptor.element_size_bytes,
        path=f"{path}.element_size_bytes",
        minimum=1,
    )
    elements_per_cell = prod(shape[1:]) if len(shape) > 1 else 1
    expected_bytes_per_cell = elements_per_cell * element_size_bytes
    if expected_bytes_per_cell > _MAX_POINTER:
        _fail(path, "shape byte-width arithmetic overflows uint64")
    if bytes_per_cell != expected_bytes_per_cell:
        _fail(
            path,
            "bytes_per_cell does not equal the trailing shape extents times "
            "element_size_bytes",
        )
    if bytes_per_cell > cell_stride:
        _fail(
            path,
            "bytes_per_cell exceeds cell_stride and overlaps adjacent cells",
        )
    last_cell_offset = (shape[0] - 1) * cell_stride
    if last_cell_offset > _MAX_POINTER:
        _fail(path, "cell offset arithmetic overflows the uint64 pointer range")
    allocation_end = pointer + last_cell_offset + bytes_per_cell
    if allocation_end > _ADDRESS_SPACE_SIZE:
        _fail(path, "allocation address arithmetic overflows uint64")

    inferred_semantic, inferred_copy = classify_checkpoint_clone_field_v1(field_name)
    semantic = descriptor.semantic_classification or inferred_semantic
    if semantic not in _COPY_CLASSIFICATION_BY_SEMANTIC:
        _fail(f"{path}.semantic_classification", "is unsupported")
    if (
        inferred_semantic != STATE_SEMANTIC_CLASSIFICATION_V1
        and semantic != inferred_semantic
    ):
        _fail(
            f"{path}.semantic_classification",
            f"contradicts inferred {inferred_semantic!r} field semantics",
        )
    copy_classification = descriptor.copy_classification or (
        _COPY_CLASSIFICATION_BY_SEMANTIC[semantic]
    )
    expected_copy = _COPY_CLASSIFICATION_BY_SEMANTIC[semantic]
    if copy_classification != expected_copy:
        _fail(
            f"{path}.copy_classification",
            f"must be {expected_copy!r} for semantic class {semantic!r}",
        )
    if (
        descriptor.semantic_classification is None
        and copy_classification != inferred_copy
    ):
        _fail(f"{path}.copy_classification", "contradicts inferred field semantics")

    return _NormalizedField(
        field_name=field_name,
        pointer_identity=pointer,
        shape=shape,
        cell_stride=cell_stride,
        bytes_per_cell=bytes_per_cell,
        element_size_bytes=element_size_bytes,
        allocation_end_exclusive=allocation_end,
        semantic_classification=semantic,
        copy_classification=copy_classification,
    )


def _physical_fields(
    descriptors: Sequence[_NormalizedField],
) -> tuple[PhysicalCheckpointCloneFieldV1, ...]:
    pointer_by_name: dict[str, int] = {}
    for descriptor in descriptors:
        previous_pointer = pointer_by_name.get(descriptor.field_name)
        if (
            previous_pointer is not None
            and previous_pointer != descriptor.pointer_identity
        ):
            _fail(
                f"field_name[{descriptor.field_name!r}]",
                "alias ambiguity: one logical field resolves to multiple pointers",
            )
        pointer_by_name[descriptor.field_name] = descriptor.pointer_identity

    by_pointer: dict[int, list[_NormalizedField]] = {}
    for descriptor in descriptors:
        by_pointer.setdefault(descriptor.pointer_identity, []).append(descriptor)

    deduplicated: list[tuple[tuple[str, ...], _NormalizedField]] = []
    for pointer, aliases in by_pointer.items():
        reference = aliases[0]
        reference_contract = (
            reference.shape,
            reference.cell_stride,
            reference.bytes_per_cell,
            reference.element_size_bytes,
            reference.allocation_end_exclusive,
            reference.semantic_classification,
            reference.copy_classification,
        )
        if any(
            (
                alias.shape,
                alias.cell_stride,
                alias.bytes_per_cell,
                alias.element_size_bytes,
                alias.allocation_end_exclusive,
                alias.semantic_classification,
                alias.copy_classification,
            )
            != reference_contract
            for alias in aliases[1:]
        ):
            _fail(
                f"pointer_identity[{pointer}]",
                "alias ambiguity: one pointer has contradictory layout or semantics",
            )
        names = tuple(sorted({alias.field_name for alias in aliases}))
        deduplicated.append((names, reference))

    by_address = sorted(deduplicated, key=lambda item: item[1].pointer_identity)
    for previous, current in zip(by_address, by_address[1:]):
        previous_names, previous_field = previous
        current_names, current_field = current
        if previous_field.allocation_end_exclusive > current_field.pointer_identity:
            _fail(
                "field_descriptors",
                "physical allocation overlap between "
                f"{previous_names[0]!r} and {current_names[0]!r}",
            )

    canonical = sorted(
        deduplicated,
        key=lambda item: (item[0][0], item[1].pointer_identity),
    )
    return tuple(
        PhysicalCheckpointCloneFieldV1(
            physical_field_index=index,
            field_name=names[0],
            field_aliases=names,
            pointer_identity=field.pointer_identity,
            shape=field.shape,
            cell_stride=field.cell_stride,
            bytes_per_cell=field.bytes_per_cell,
            element_size_bytes=field.element_size_bytes,
            allocation_end_exclusive=field.allocation_end_exclusive,
            semantic_classification=field.semantic_classification,
            copy_classification=field.copy_classification,
        )
        for index, (names, field) in enumerate(canonical)
    )


def _target_indices(
    target_cell_indices: Iterable[int],
    *,
    source_cell_index: int,
    cell_count: int,
) -> tuple[int, ...]:
    if isinstance(target_cell_indices, (str, bytes, bytearray)):
        _fail("target_cell_indices", "must be an iterable of cell indices")
    try:
        raw_targets = tuple(target_cell_indices)
    except TypeError:
        _fail("target_cell_indices", "must be an iterable of cell indices")
    if not raw_targets:
        _fail("target_cell_indices", "must contain at least one target")
    targets: list[int] = []
    seen: set[int] = set()
    for index, raw_target in enumerate(raw_targets):
        target = _required_int(
            raw_target,
            path=f"target_cell_indices[{index}]",
            minimum=0,
        )
        if target >= cell_count:
            _fail(f"target_cell_indices[{index}]", "is outside descriptor cell bounds")
        if target == source_cell_index:
            _fail(
                f"target_cell_indices[{index}]",
                "creates a cyclic/self target and would overwrite the source",
            )
        if target in seen:
            _fail(f"target_cell_indices[{index}]", "duplicates an earlier target")
        seen.add(target)
        targets.append(target)
    return tuple(sorted(targets))


def plan_checkpoint_clone_v1(
    field_descriptors: Sequence[
        CheckpointCloneFieldDescriptorV1 | Mapping[str, Any]
    ],
    *,
    state_manifest: Mapping[str, Any],
    generation_boundary_complete: bool,
    source_cell_index: int,
    target_cell_indices: Iterable[int],
) -> CheckpointClonePlanV1:
    """Build a deterministic, source-preserving checkpoint clone plan."""
    if generation_boundary_complete is not True:
        _fail(
            "generation_boundary_complete",
            "must be exactly true before checkpoint cloning",
        )
    if not isinstance(field_descriptors, Sequence) or isinstance(
        field_descriptors, (str, bytes, bytearray)
    ):
        _fail("field_descriptors", "must be a nonempty sequence")
    if not field_descriptors:
        _fail("field_descriptors", "must be nonempty")
    (
        _normalized_manifest,
        state_manifest_sha256,
        canonical_required,
        manifest_by_name,
    ) = _normalize_state_manifest_v1(state_manifest)

    normalized = tuple(
        _normalize_descriptor(raw, index=index)
        for index, raw in enumerate(field_descriptors)
    )
    for field in normalized:
        expected = manifest_by_name.get(field.field_name)
        if expected is None:
            continue
        if (
            list(field.shape) != expected["shape"]
            or field.element_size_bytes != expected["element_size_bytes"]
            or field.semantic_classification
                != expected["semantic_classification"]
            or field.copy_classification != expected["copy_classification"]
        ):
            _fail(
                f"field_descriptors[{field.field_name!r}]",
                "does not match the frozen manifest layout and semantics",
            )
    cell_counts = {field.shape[0] for field in normalized}
    if len(cell_counts) != 1:
        _fail(
            "field_descriptors",
            "all physical fields must expose the same leading cell count",
        )
    cell_count = next(iter(cell_counts))
    source = _required_int(source_cell_index, path="source_cell_index", minimum=0)
    if source >= cell_count:
        _fail("source_cell_index", "is outside descriptor cell bounds")
    targets = _target_indices(
        target_cell_indices,
        source_cell_index=source,
        cell_count=cell_count,
    )
    physical_fields = _physical_fields(normalized)
    provided_names = sorted(field.field_name for field in normalized)
    if tuple(provided_names) != canonical_required:
        missing = sorted(set(canonical_required) - set(provided_names))
        unexpected = sorted(set(provided_names) - set(canonical_required))
        _fail(
            "field_descriptors",
            "does not exactly cover the frozen state manifest; "
            f"missing={missing}, unexpected={unexpected}",
        )

    entries: list[CheckpointCloneEntryV1] = []
    total_copy_bytes = 0
    for field in physical_fields:
        source_pointer = field.pointer_identity + source * field.cell_stride
        for target in targets:
            target_pointer = field.pointer_identity + target * field.cell_stride
            entries.append(
                CheckpointCloneEntryV1(
                    clone_entry_index=len(entries),
                    physical_field_index=field.physical_field_index,
                    field_name=field.field_name,
                    field_aliases=field.field_aliases,
                    semantic_classification=field.semantic_classification,
                    copy_classification=field.copy_classification,
                    source_cell_index=source,
                    target_cell_index=target,
                    source_pointer=source_pointer,
                    target_pointer=target_pointer,
                    bytes_to_copy=field.bytes_per_cell,
                )
            )
            total_copy_bytes += field.bytes_per_cell
            if total_copy_bytes > _MAX_POINTER:
                _fail("total_copy_bytes", "overflows uint64")

    if any(entry.target_cell_index == source for entry in entries):
        raise AssertionError("planner emitted a write into the source cell")
    return CheckpointClonePlanV1._from_validated(
        state_manifest_sha256=state_manifest_sha256,
        required_field_names=canonical_required,
        generation_boundary_complete=True,
        source_cell_index=source,
        target_cell_indices=targets,
        logical_field_count=len(normalized),
        physical_fields=physical_fields,
        entries=tuple(entries),
        total_copy_bytes=total_copy_bytes,
    )


def build_checkpoint_clone_zero_d2h_receipt_v1(
    plan: CheckpointClonePlanV1,
    *,
    executed_copy_operations: int,
    executed_copy_bytes: int,
    device_to_host_operations: int,
    device_to_host_bytes: int,
    source_preserved: bool,
    hidden_host_materialization: bool,
    execution_performed: bool,
) -> CheckpointCloneZeroD2HReceiptV1:
    """Build a planning receipt without claiming observed CUDA execution."""
    if not isinstance(plan, CheckpointClonePlanV1):
        _fail("plan", "must be CheckpointClonePlanV1")
    counters = {
        "executed_copy_operations": executed_copy_operations,
        "executed_copy_bytes": executed_copy_bytes,
        "device_to_host_operations": device_to_host_operations,
        "device_to_host_bytes": device_to_host_bytes,
    }
    for name, value in counters.items():
        _required_int(value, path=name, minimum=0)
    for name, value in {
        "source_preserved": source_preserved,
        "hidden_host_materialization": hidden_host_materialization,
        "execution_performed": execution_performed,
    }.items():
        if type(value) is not bool:
            _fail(name, "must be a boolean")
    if execution_performed:
        _fail(
            "execution_performed",
            "host planning code cannot authenticate CUDA execution",
        )
    if any(counters.values()):
        _fail(
            "execution_counters",
            "must remain zero until a trusted CUDA executor emits observations",
        )
    if source_preserved:
        _fail(
            "source_preserved",
            "cannot be observed by the host planning module",
        )
    if hidden_host_materialization:
        _fail(
            "hidden_host_materialization",
            "planning receipts cannot report execution-side materialization",
        )
    return CheckpointCloneZeroD2HReceiptV1._from_planning(
        plan_sha256=plan.plan_sha256,
        generation_boundary_complete=plan.generation_boundary_complete,
        source_cell_index=plan.source_cell_index,
        target_cell_indices=plan.target_cell_indices,
        physical_field_count=plan.physical_field_count,
        expected_copy_operations=plan.clone_entry_count,
        expected_copy_bytes=plan.total_copy_bytes,
    )


# The longer builder name is retained for callers that treat this as a plan object.
build_checkpoint_clone_plan_v1 = plan_checkpoint_clone_v1


__all__ = [
    "CHECKPOINT_CLONE_PLAN_SCHEMA_VERSION_V1",
    "ACCEPTED_STATE_BANK_MANIFEST_SHA256_V1",
    "CHECKPOINT_CLONE_STATE_MANIFEST_SCHEMA_VERSION_V1",
    "CHECKPOINT_CLONE_ZERO_D2H_RECEIPT_REQUIRED_FIELDS_V1",
    "CHECKPOINT_CLONE_ZERO_D2H_RECEIPT_SCHEMA_VERSION_V1",
    "EXACT_COPY_CLASSIFICATIONS_V1",
    "LINEAGE_EXACT_COPY_CLASSIFICATION_V1",
    "LINEAGE_SEMANTIC_CLASSIFICATION_V1",
    "RNG_EXACT_COPY_CLASSIFICATION_V1",
    "RNG_SEMANTIC_CLASSIFICATION_V1",
    "STATE_EXACT_COPY_CLASSIFICATION_V1",
    "STATE_SEMANTIC_CLASSIFICATION_V1",
    "CheckpointCloneContractError",
    "CheckpointCloneEntryV1",
    "CheckpointCloneFieldDescriptorV1",
    "CheckpointClonePlanV1",
    "CheckpointCloneZeroD2HReceiptV1",
    "PhysicalCheckpointCloneFieldV1",
    "build_checkpoint_clone_plan_v1",
    "build_accepted_checkpoint_clone_state_manifest_v1",
    "build_checkpoint_clone_zero_d2h_receipt_v1",
    "classify_checkpoint_clone_field_v1",
    "plan_checkpoint_clone_v1",
]
