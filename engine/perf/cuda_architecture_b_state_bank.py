"""Manifest-driven validation for the Architecture B CUDA state bank.

This module validates storage and checkpoint transport mechanics only. It does
not execute a simulation or make a simulation-parity claim.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Final, Mapping

import torch

from perf.cuda_architecture_b_checkpoint import (
    export_cuda_boundary_checkpoint,
    restore_cuda_boundary_checkpoint,
)


DEFAULT_ACCEPTED_MANIFEST: Final[Path] = Path(__file__).resolve().with_name(
    "flat_patch_device_neutral_state_manifest_v4.json"
)
ACCEPTED_MANIFEST_SCHEMA_VERSION: Final[str] = (
    "eiel_flat_patch_device_neutral_state_manifest_v4"
)
ACCEPTED_MANIFEST_CONTRACT_STATUS: Final[str] = (
    "accepted_repository_bundled_flat_patch_state_bank_v4"
)
ACCEPTED_MANIFEST_SHA256: Final[str] = (
    "e7981dc8d602f3a25b627881690ccdb231d82b7aaa98e4fdcd6afda6d715dfce"
)
ACCEPTED_TENSOR_FIELD_COUNT: Final[int] = 139
DEFAULT_WDL_MANIFEST: Final[Path] = Path(__file__).resolve().with_name(
    "flat_patch_wdl_device_neutral_state_manifest_v1.json"
)
WDL_MANIFEST_SCHEMA_VERSION: Final[str] = (
    "eiel_flat_patch_wdl_device_neutral_state_manifest_v1"
)
WDL_MANIFEST_CONTRACT_STATUS: Final[str] = (
    "accepted_repository_bundled_flat_patch_wdl_state_bank_v1"
)
WDL_MANIFEST_SHA256: Final[str] = (
    "f27f8c6418c7a2c883b40a5452df997cb33b48e00ffd5006e8e07677d306291b"
)
WDL_TENSOR_FIELD_COUNT: Final[int] = 146
CHECKPOINT_PROJECTION_SCHEMA_VERSION: Final[str] = (
    "cxxbackbone10_checkpoint_projection_v2"
)

EXACT_TENSOR_CHECKPOINT_MODE: Final[str] = "included_exact_tensor_clone"
CANONICAL_DISABLED_CHECKPOINT_MODE: Final[str] = (
    "included_canonical_disabled_tensor"
)
REFRESH_REQUIRED_CHECKPOINT_MODE: Final[str] = (
    "included_refreshed_derived_tensor"
)
ACTIVE_PREFIX_CHECKPOINT_MODE: Final[str] = "included_active_prefix"
BLOCKED_CHECKPOINT_MODE: Final[str] = "included_cpu_generator_byte_state"
KNOWN_CHECKPOINT_MODES: Final[frozenset[str]] = frozenset(
    {
        EXACT_TENSOR_CHECKPOINT_MODE,
        CANONICAL_DISABLED_CHECKPOINT_MODE,
        REFRESH_REQUIRED_CHECKPOINT_MODE,
        ACTIVE_PREFIX_CHECKPOINT_MODE,
        BLOCKED_CHECKPOINT_MODE,
    }
)
KNOWN_AUTHORITY_CLASSES: Final[frozenset[str]] = frozenset(
    {"immutable_snapshot_clone", "derived_recomputed"}
)
TORCH_DTYPE_BY_MANIFEST_NAME: Final[dict[str, torch.dtype]] = {
    "bool": torch.bool,
    "float32": torch.float32,
    "float64": torch.float64,
    "int8": torch.int8,
    "int32": torch.int32,
    "int64": torch.int64,
    "uint8": torch.uint8,
}


class StateBankManifestError(RuntimeError):
    """Raised when a supplied state-bank manifest is not an accepted contract."""


@dataclass(frozen=True)
class StateBankFieldContract:
    name: str
    dtype_name: str
    device_neutral_shape_variables: tuple[str, ...]
    device_neutral_rank: int
    authority_class: str
    persistence_class: str
    semantic_owner: str
    checkpoint_mode: str

    @property
    def torch_dtype(self) -> torch.dtype:
        return TORCH_DTYPE_BY_MANIFEST_NAME[self.dtype_name]


@dataclass(frozen=True)
class AcceptedStateBankManifest:
    source_path: str
    source_sha256: str
    schema_version: str
    contract_status: str
    fields: tuple[StateBankFieldContract, ...]
    non_tensor_field_names: tuple[str, ...]

    @property
    def fields_by_name(self) -> dict[str, StateBankFieldContract]:
        return {field.name: field for field in self.fields}


@dataclass(frozen=True)
class CudaResidentStateField:
    tensor: torch.Tensor
    authority_class: str


@dataclass(frozen=True, order=True)
class StateBankValidationIssue:
    field_name: str
    code: str
    expected: str
    actual: str

    def to_dict(self) -> dict[str, str]:
        return {
            "field_name": self.field_name,
            "code": self.code,
            "expected": self.expected,
            "actual": self.actual,
        }


@dataclass(frozen=True)
class CudaStateBankValidationReceipt:
    manifest_path: str
    manifest_sha256: str
    manifest_schema_version: str
    accepted_field_count: int
    provided_field_count: int
    present_accepted_field_count: int
    resident_field_count: int
    device: str | None
    shape_bindings: tuple[tuple[str, int], ...]
    missing_accepted_fields: tuple[str, ...]
    unexpected_fields: tuple[str, ...]
    invalid_accepted_fields: tuple[str, ...]
    issues: tuple[StateBankValidationIssue, ...]
    exact_tensor_checkpoint_fields: tuple[str, ...]
    canonical_disabled_checkpoint_fields: tuple[str, ...]
    refresh_required_checkpoint_fields: tuple[str, ...]
    active_prefix_checkpoint_fields: tuple[str, ...]
    blocked_checkpoint_fields: tuple[str, ...]
    manifest_non_tensor_fields: tuple[str, ...]
    resident_qualified: bool
    complete_semantic_checkpoint_transport_qualified: bool
    simulation_parity_claimed: bool = False

    def _payload(self) -> dict[str, Any]:
        return {
            "receipt_schema_version": "eiel_cuda_architecture_b_state_bank_receipt_v1",
            "manifest_path": self.manifest_path,
            "manifest_sha256": self.manifest_sha256,
            "manifest_schema_version": self.manifest_schema_version,
            "state_bank_scope": "accepted_tensor_surfaces_only",
            "accepted_field_count": self.accepted_field_count,
            "provided_field_count": self.provided_field_count,
            "present_accepted_field_count": self.present_accepted_field_count,
            "resident_field_count": self.resident_field_count,
            "device": self.device,
            "shape_bindings": dict(self.shape_bindings),
            "missing_accepted_fields": list(self.missing_accepted_fields),
            "unexpected_fields": list(self.unexpected_fields),
            "invalid_accepted_fields": list(self.invalid_accepted_fields),
            "issues": [issue.to_dict() for issue in self.issues],
            "exact_tensor_checkpoint_fields": list(
                self.exact_tensor_checkpoint_fields
            ),
            "canonical_disabled_checkpoint_fields": list(
                self.canonical_disabled_checkpoint_fields
            ),
            "refresh_required_checkpoint_fields": list(
                self.refresh_required_checkpoint_fields
            ),
            "active_prefix_checkpoint_fields": list(
                self.active_prefix_checkpoint_fields
            ),
            "blocked_checkpoint_fields": list(self.blocked_checkpoint_fields),
            "manifest_non_tensor_fields": list(self.manifest_non_tensor_fields),
            "resident_qualified": self.resident_qualified,
            "complete_semantic_checkpoint_transport_qualified": (
                self.complete_semantic_checkpoint_transport_qualified
            ),
            "simulation_parity_claimed": self.simulation_parity_claimed,
        }

    @property
    def receipt_sha256(self) -> str:
        return _canonical_sha256(self._payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self._payload()
        payload["receipt_sha256"] = self.receipt_sha256
        return payload


@dataclass(frozen=True)
class CudaStateBankCheckpointTransportReceipt:
    state_bank_receipt_sha256: str
    checkpoint_projection_digest: str
    source_device: str
    restored_device: str
    accepted_field_count: int
    mechanically_roundtripped_fields: tuple[str, ...]
    mismatched_fields: tuple[str, ...]
    exact_tensor_checkpoint_fields: tuple[str, ...]
    canonical_value_required_checkpoint_fields: tuple[str, ...]
    refresh_required_checkpoint_fields: tuple[str, ...]
    active_prefix_projection_required_fields: tuple[str, ...]
    blocked_semantic_checkpoint_fields: tuple[str, ...]
    device_to_host_operations: int
    device_to_host_bytes: int
    host_to_device_operations: int
    host_to_device_bytes: int
    checkpoint_api_device_synchronizations: int
    mechanical_roundtrip_qualified: bool
    complete_semantic_checkpoint_transport_qualified: bool
    simulation_parity_claimed: bool = False

    def _payload(self) -> dict[str, Any]:
        return {
            "receipt_schema_version": (
                "eiel_cuda_architecture_b_state_bank_checkpoint_transport_receipt_v1"
            ),
            "state_bank_receipt_sha256": self.state_bank_receipt_sha256,
            "checkpoint_projection_digest": self.checkpoint_projection_digest,
            "source_device": self.source_device,
            "restored_device": self.restored_device,
            "accepted_field_count": self.accepted_field_count,
            "mechanically_roundtripped_fields": list(
                self.mechanically_roundtripped_fields
            ),
            "mismatched_fields": list(self.mismatched_fields),
            "exact_tensor_checkpoint_fields": list(
                self.exact_tensor_checkpoint_fields
            ),
            "canonical_value_required_checkpoint_fields": list(
                self.canonical_value_required_checkpoint_fields
            ),
            "refresh_required_checkpoint_fields": list(
                self.refresh_required_checkpoint_fields
            ),
            "active_prefix_projection_required_fields": list(
                self.active_prefix_projection_required_fields
            ),
            "blocked_semantic_checkpoint_fields": list(
                self.blocked_semantic_checkpoint_fields
            ),
            "device_to_host_operations": self.device_to_host_operations,
            "device_to_host_bytes": self.device_to_host_bytes,
            "host_to_device_operations": self.host_to_device_operations,
            "host_to_device_bytes": self.host_to_device_bytes,
            "checkpoint_api_device_synchronizations": (
                self.checkpoint_api_device_synchronizations
            ),
            "mechanical_roundtrip_qualified": self.mechanical_roundtrip_qualified,
            "complete_semantic_checkpoint_transport_qualified": (
                self.complete_semantic_checkpoint_transport_qualified
            ),
            "simulation_parity_claimed": self.simulation_parity_claimed,
        }

    @property
    def receipt_sha256(self) -> str:
        return _canonical_sha256(self._payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self._payload()
        payload["receipt_sha256"] = self.receipt_sha256
        return payload


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _require_mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise StateBankManifestError(f"{label} must be a JSON object")
    return value


def _load_state_bank_manifest(
    manifest_path: str | Path,
    *,
    expected_sha256: str,
    expected_schema_version: str,
    expected_contract_status: str,
    expected_tensor_field_count: int,
    contract_label: str,
) -> AcceptedStateBankManifest:
    path = Path(manifest_path)
    try:
        encoded = path.read_bytes()
        raw = json.loads(encoded)
    except (OSError, json.JSONDecodeError) as exc:
        raise StateBankManifestError(
            f"unable to load accepted state manifest: {path}"
        ) from exc
    root = _require_mapping(raw, label="state manifest")
    manifest_sha256 = sha256(encoded).hexdigest()
    if manifest_sha256 != expected_sha256:
        raise StateBankManifestError(
            "state manifest digest mismatch: "
            f"expected {expected_sha256}, got {manifest_sha256}"
        )
    schema_version = str(root.get("schema_version", ""))
    if schema_version != expected_schema_version:
        raise StateBankManifestError(
            "state manifest schema mismatch: "
            f"expected {expected_schema_version}, got {schema_version or '<missing>'}"
        )
    contract_status = str(root.get("contract_status", ""))
    if contract_status != expected_contract_status:
        raise StateBankManifestError(
            f"state manifest is not the accepted repository-bundled {contract_label} contract"
        )
    global_layout = _require_mapping(
        root.get("global_layout_contract"), label="global_layout_contract"
    )
    if (
        global_layout.get("normative_tensor_form") != "[cell, ...state_dimensions...]"
        or global_layout.get("same_shape_batch") is not True
        or global_layout.get("storage")
        != "single contiguous row-major allocation per named bank with cell-major outer axis"
    ):
        raise StateBankManifestError(
            "state manifest global CUDA layout contract is not accepted"
        )

    raw_fields = root.get("tensor_surfaces")
    if (
        not isinstance(raw_fields, list)
        or len(raw_fields) != expected_tensor_field_count
    ):
        actual_count = len(raw_fields) if isinstance(raw_fields, list) else "not-a-list"
        raise StateBankManifestError(
            "accepted tensor field count mismatch: "
            f"expected {expected_tensor_field_count}, got {actual_count}"
        )

    fields: list[StateBankFieldContract] = []
    seen_names: set[str] = set()
    for index, raw_field in enumerate(raw_fields):
        row = _require_mapping(raw_field, label=f"tensor_surfaces[{index}]")
        name = str(row.get("canonical_name", ""))
        if not name or name in seen_names:
            raise StateBankManifestError(
                f"tensor field name is missing or duplicated at index {index}: {name!r}"
            )
        seen_names.add(name)
        dtype_name = str(row.get("dtype", ""))
        if dtype_name not in TORCH_DTYPE_BY_MANIFEST_NAME:
            raise StateBankManifestError(
                f"unsupported manifest dtype for {name}: {dtype_name!r}"
            )
        shape_value = row.get("device_neutral_shape_variables")
        if not isinstance(shape_value, list) or not all(
            isinstance(symbol, str) and symbol for symbol in shape_value
        ):
            raise StateBankManifestError(
                f"invalid device-neutral shape variables for {name}"
            )
        shape_variables = tuple(shape_value)
        rank = row.get("device_neutral_rank")
        if rank != len(shape_variables) or not shape_variables or shape_variables[0] != "cell":
            raise StateBankManifestError(
                f"invalid leading-cell rank contract for {name}"
            )
        layout = _require_mapping(
            row.get("strides_layout"), label=f"strides_layout for {name}"
        )
        if layout.get("required_layout") != "row_major_contiguous_cell_major":
            raise StateBankManifestError(
                f"unsupported storage layout for {name}: {layout.get('required_layout')!r}"
            )
        boundary = _require_mapping(
            row.get("boundary_projection_representation"),
            label=f"boundary projection for {name}",
        )
        authority_class = str(boundary.get("authority", ""))
        if authority_class not in KNOWN_AUTHORITY_CLASSES:
            raise StateBankManifestError(
                f"unsupported authority class for {name}: {authority_class!r}"
            )
        checkpoint = _require_mapping(
            row.get("checkpoint_representation"),
            label=f"checkpoint representation for {name}",
        )
        checkpoint_mode = str(checkpoint.get("mode", ""))
        if checkpoint_mode not in KNOWN_CHECKPOINT_MODES:
            raise StateBankManifestError(
                f"unsupported checkpoint mode for {name}: {checkpoint_mode!r}"
            )
        fields.append(
            StateBankFieldContract(
                name=name,
                dtype_name=dtype_name,
                device_neutral_shape_variables=shape_variables,
                device_neutral_rank=int(rank),
                authority_class=authority_class,
                persistence_class=str(row.get("persistence_class", "")),
                semantic_owner=str(row.get("semantic_owner", "")),
                checkpoint_mode=checkpoint_mode,
            )
        )

    raw_non_tensor_fields = root.get("non_tensor_state_surfaces")
    if not isinstance(raw_non_tensor_fields, list):
        raise StateBankManifestError("non_tensor_state_surfaces must be a list")
    non_tensor_names = []
    for index, raw_field in enumerate(raw_non_tensor_fields):
        row = _require_mapping(
            raw_field, label=f"non_tensor_state_surfaces[{index}]"
        )
        name = str(row.get("canonical_name", ""))
        if not name:
            raise StateBankManifestError(
                f"non-tensor field name is missing at index {index}"
            )
        non_tensor_names.append(name)
    if len(non_tensor_names) != len(set(non_tensor_names)):
        raise StateBankManifestError("non-tensor field names must be unique")

    return AcceptedStateBankManifest(
        source_path=str(path.resolve()),
        source_sha256=manifest_sha256,
        schema_version=schema_version,
        contract_status=contract_status,
        fields=tuple(fields),
        non_tensor_field_names=tuple(sorted(non_tensor_names)),
    )


def load_accepted_state_bank_manifest(
    manifest_path: str | Path = DEFAULT_ACCEPTED_MANIFEST,
) -> AcceptedStateBankManifest:
    return _load_state_bank_manifest(
        manifest_path,
        expected_sha256=ACCEPTED_MANIFEST_SHA256,
        expected_schema_version=ACCEPTED_MANIFEST_SCHEMA_VERSION,
        expected_contract_status=ACCEPTED_MANIFEST_CONTRACT_STATUS,
        expected_tensor_field_count=ACCEPTED_TENSOR_FIELD_COUNT,
        contract_label="Flat/Patch",
    )


def load_wdl_state_bank_manifest(
    manifest_path: str | Path = DEFAULT_WDL_MANIFEST,
) -> AcceptedStateBankManifest:
    return _load_state_bank_manifest(
        manifest_path,
        expected_sha256=WDL_MANIFEST_SHA256,
        expected_schema_version=WDL_MANIFEST_SCHEMA_VERSION,
        expected_contract_status=WDL_MANIFEST_CONTRACT_STATUS,
        expected_tensor_field_count=WDL_TENSOR_FIELD_COUNT,
        contract_label="Flat/Patch/WDL",
    )


def _normalize_bank_field(
    name: str,
    value: CudaResidentStateField | torch.Tensor | Any,
    authority_classes: Mapping[str, str],
) -> tuple[Any, str | None, StateBankValidationIssue | None]:
    declared_authority = authority_classes.get(name)
    if isinstance(value, CudaResidentStateField):
        if declared_authority is not None and declared_authority != value.authority_class:
            return (
                value.tensor,
                value.authority_class,
                StateBankValidationIssue(
                    field_name=name,
                    code="authority_declaration_conflict",
                    expected=value.authority_class,
                    actual=str(declared_authority),
                ),
            )
        return value.tensor, value.authority_class, None
    if isinstance(value, torch.Tensor):
        return value, declared_authority, None
    return value, declared_authority, None


def _storage_identity(tensor: torch.Tensor) -> tuple[str, int]:
    storage = tensor.untyped_storage()
    return str(tensor.device), int(storage._cdata)


def validate_cuda_resident_state_bank(
    bank: Mapping[str, CudaResidentStateField | torch.Tensor],
    *,
    authority_classes: Mapping[str, str] | None = None,
    expected_shape_bindings: Mapping[str, int] | None = None,
    manifest_path: str | Path = DEFAULT_ACCEPTED_MANIFEST,
    accepted_manifest: AcceptedStateBankManifest | None = None,
) -> CudaStateBankValidationReceipt:
    """Validate an exact CUDA resident tensor bank against the accepted manifest."""

    if accepted_manifest is not None and Path(manifest_path) != DEFAULT_ACCEPTED_MANIFEST:
        raise ValueError("accepted_manifest and manifest_path are mutually exclusive")
    manifest = accepted_manifest or load_accepted_state_bank_manifest(manifest_path)
    contracts = manifest.fields_by_name
    bank_names = set(bank)
    accepted_names = set(contracts)
    missing = tuple(sorted(accepted_names - bank_names))
    unexpected = tuple(sorted(bank_names - accepted_names))
    authorities = dict(authority_classes or {})
    expected_shapes = dict(expected_shape_bindings or {})
    known_shape_symbols = {
        symbol
        for contract in manifest.fields
        for symbol in contract.device_neutral_shape_variables
    }
    issues: list[StateBankValidationIssue] = []
    invalid_fields: set[str] = set()
    normalized_tensors: dict[str, torch.Tensor] = {}
    shape_observations: dict[str, dict[int, set[str]]] = {}
    cuda_devices: set[str] = set()

    for symbol, size in sorted(expected_shapes.items()):
        if not isinstance(symbol, str) or not symbol:
            raise ValueError("shape binding names must be non-empty strings")
        if symbol not in known_shape_symbols:
            raise ValueError(f"shape binding {symbol!r} is not declared by the manifest")
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise ValueError(f"shape binding {symbol!r} must be a non-negative integer")
        if symbol == "cell" and size == 0:
            raise ValueError("shape binding 'cell' must be positive")

    def add_issue(issue: StateBankValidationIssue, *field_names: str) -> None:
        issues.append(issue)
        invalid_fields.update(field_names or (issue.field_name,))

    for name in sorted(accepted_names & bank_names):
        contract = contracts[name]
        value, authority, normalization_issue = _normalize_bank_field(
            name, bank[name], authorities
        )
        if normalization_issue is not None:
            add_issue(normalization_issue)
        if authority is None:
            add_issue(
                StateBankValidationIssue(
                    name,
                    "authority_class_missing",
                    contract.authority_class,
                    "<missing>",
                )
            )
        elif authority != contract.authority_class:
            add_issue(
                StateBankValidationIssue(
                    name,
                    "authority_class_mismatch",
                    contract.authority_class,
                    str(authority),
                )
            )
        if not isinstance(value, torch.Tensor):
            add_issue(
                StateBankValidationIssue(
                    name,
                    "not_a_tensor",
                    "torch.Tensor",
                    type(value).__name__,
                )
            )
            continue
        normalized_tensors[name] = value
        if value.device.type != "cuda":
            add_issue(
                StateBankValidationIssue(
                    name,
                    "not_cuda_resident",
                    "cuda",
                    str(value.device),
                )
            )
        else:
            cuda_devices.add(str(value.device))
        if value.dtype != contract.torch_dtype:
            add_issue(
                StateBankValidationIssue(
                    name,
                    "dtype_mismatch",
                    contract.dtype_name,
                    str(value.dtype).removeprefix("torch."),
                )
            )
        if value.ndim != contract.device_neutral_rank:
            add_issue(
                StateBankValidationIssue(
                    name,
                    "rank_mismatch",
                    str(contract.device_neutral_rank),
                    str(value.ndim),
                )
            )
        else:
            for symbol, size in zip(
                contract.device_neutral_shape_variables, value.shape, strict=True
            ):
                shape_observations.setdefault(symbol, {}).setdefault(
                    int(size), set()
                ).add(name)
        if not value.is_contiguous(memory_format=torch.contiguous_format):
            add_issue(
                StateBankValidationIssue(
                    name,
                    "not_contiguous",
                    "row_major_contiguous_cell_major",
                    str(tuple(value.stride())),
                )
            )
        expected_storage_bytes = int(value.numel()) * int(value.element_size())
        if (
            value.storage_offset() != 0
            or value.untyped_storage().nbytes() != expected_storage_bytes
        ):
            add_issue(
                StateBankValidationIssue(
                    name,
                    "not_dedicated_named_bank_storage",
                    f"offset=0,nbytes={expected_storage_bytes}",
                    (
                        f"offset={value.storage_offset()},"
                        f"nbytes={value.untyped_storage().nbytes()}"
                    ),
                )
            )

    resolved_shapes: dict[str, int] = {}
    for symbol, sizes_to_fields in sorted(shape_observations.items()):
        observed_sizes = tuple(sorted(sizes_to_fields))
        if symbol in expected_shapes:
            expected_size = expected_shapes[symbol]
            resolved_shapes[symbol] = expected_size
            for actual_size, names in sorted(sizes_to_fields.items()):
                if actual_size == expected_size:
                    continue
                for name in sorted(names):
                    add_issue(
                        StateBankValidationIssue(
                            name,
                            "shape_symbol_mismatch",
                            f"{symbol}={expected_size}",
                            f"{symbol}={actual_size}",
                        )
                    )
        elif len(observed_sizes) == 1:
            resolved_shapes[symbol] = observed_sizes[0]
        else:
            affected_fields = tuple(
                sorted(
                    name
                    for names in sizes_to_fields.values()
                    for name in names
                )
            )
            add_issue(
                StateBankValidationIssue(
                    f"@shape:{symbol}",
                    "inconsistent_shape_symbol",
                    "one shared size",
                    ",".join(str(size) for size in observed_sizes),
                ),
                *affected_fields,
            )
    for symbol, expected_size in sorted(expected_shapes.items()):
        if symbol not in shape_observations:
            resolved_shapes[symbol] = expected_size

    if len(cuda_devices) > 1:
        add_issue(
            StateBankValidationIssue(
                "@bank",
                "multiple_cuda_devices",
                "one CUDA device",
                ",".join(sorted(cuda_devices)),
            ),
            *normalized_tensors,
        )

    storage_groups: dict[tuple[str, int], list[str]] = {}
    for name, tensor in normalized_tensors.items():
        storage_groups.setdefault(_storage_identity(tensor), []).append(name)
    for names in sorted(
        (tuple(sorted(group)) for group in storage_groups.values() if len(group) > 1)
    ):
        actual = ",".join(names)
        for name in names:
            add_issue(
                StateBankValidationIssue(
                    name,
                    "aliased_named_bank_storage",
                    "independent storage per named bank",
                    actual,
                )
            )

    for name in unexpected:
        issues.append(
            StateBankValidationIssue(
                name,
                "unexpected_field",
                "accepted manifest field",
                "not declared",
            )
        )

    exact_tensor = tuple(
        sorted(
            field.name
            for field in manifest.fields
            if field.checkpoint_mode == EXACT_TENSOR_CHECKPOINT_MODE
        )
    )
    canonical_disabled = tuple(
        sorted(
            field.name
            for field in manifest.fields
            if field.checkpoint_mode == CANONICAL_DISABLED_CHECKPOINT_MODE
        )
    )
    refresh_required = tuple(
        sorted(
            field.name
            for field in manifest.fields
            if field.checkpoint_mode == REFRESH_REQUIRED_CHECKPOINT_MODE
        )
    )
    active_prefix = tuple(
        sorted(
            field.name
            for field in manifest.fields
            if field.checkpoint_mode == ACTIVE_PREFIX_CHECKPOINT_MODE
        )
    )
    blocked = tuple(
        sorted(
            field.name
            for field in manifest.fields
            if field.checkpoint_mode == BLOCKED_CHECKPOINT_MODE
        )
    )
    resident_fields = accepted_names - set(missing) - invalid_fields
    resident_qualified = not missing and not unexpected and not issues
    device = next(iter(cuda_devices)) if len(cuda_devices) == 1 else None
    return CudaStateBankValidationReceipt(
        manifest_path=manifest.source_path,
        manifest_sha256=manifest.source_sha256,
        manifest_schema_version=manifest.schema_version,
        accepted_field_count=len(manifest.fields),
        provided_field_count=len(bank),
        present_accepted_field_count=len(accepted_names & bank_names),
        resident_field_count=len(resident_fields),
        device=device,
        shape_bindings=tuple(sorted(resolved_shapes.items())),
        missing_accepted_fields=missing,
        unexpected_fields=unexpected,
        invalid_accepted_fields=tuple(sorted(invalid_fields & accepted_names)),
        issues=tuple(sorted(issues)),
        exact_tensor_checkpoint_fields=exact_tensor,
        canonical_disabled_checkpoint_fields=canonical_disabled,
        refresh_required_checkpoint_fields=refresh_required,
        active_prefix_checkpoint_fields=active_prefix,
        blocked_checkpoint_fields=blocked,
        manifest_non_tensor_fields=manifest.non_tensor_field_names,
        resident_qualified=resident_qualified,
        complete_semantic_checkpoint_transport_qualified=(
            resident_qualified
            and not canonical_disabled
            and not refresh_required
            and not active_prefix
            and not blocked
        ),
        simulation_parity_claimed=False,
    )


def _extract_tensor_bank(
    bank: Mapping[str, CudaResidentStateField | torch.Tensor],
) -> dict[str, torch.Tensor]:
    return {
        name: value.tensor if isinstance(value, CudaResidentStateField) else value
        for name, value in sorted(bank.items())
    }


def _tensor_bytes_equal(left: torch.Tensor, right: torch.Tensor) -> bool:
    if left.dtype != right.dtype or left.shape != right.shape:
        return False
    return bool(torch.equal(left.view(torch.uint8), right.view(torch.uint8)))


def verify_cuda_state_bank_checkpoint_transport(
    bank: Mapping[str, CudaResidentStateField | torch.Tensor],
    *,
    validation_receipt: CudaStateBankValidationReceipt | None = None,
    authority_classes: Mapping[str, str] | None = None,
    expected_shape_bindings: Mapping[str, int] | None = None,
    manifest_path: str | Path = DEFAULT_ACCEPTED_MANIFEST,
    accepted_manifest: AcceptedStateBankManifest | None = None,
) -> CudaStateBankCheckpointTransportReceipt:
    """Prove byte-exact checkpoint API transport for an already valid bank.

    A successful mechanical round-trip does not resolve active-prefix export,
    derived-field refresh, or the opaque CPU-generator representation.
    """

    current_receipt = validate_cuda_resident_state_bank(
        bank,
        authority_classes=authority_classes,
        expected_shape_bindings=expected_shape_bindings,
        manifest_path=manifest_path,
        accepted_manifest=accepted_manifest,
    )
    if (
        validation_receipt is not None
        and validation_receipt.receipt_sha256 != current_receipt.receipt_sha256
    ):
        raise RuntimeError("supplied validation receipt does not match the current bank")
    receipt = current_receipt
    if not receipt.resident_qualified or receipt.device is None:
        raise RuntimeError(
            "checkpoint transport verification requires a qualified CUDA state bank"
        )
    tensors = _extract_tensor_bank(bank)
    checkpoint = export_cuda_boundary_checkpoint(
        schema_version=CHECKPOINT_PROJECTION_SCHEMA_VERSION,
        metadata={
            "state_bank_manifest_sha256": receipt.manifest_sha256,
            "state_bank_validation_receipt_sha256": receipt.receipt_sha256,
            "simulation_parity_claimed": False,
        },
        tensors=tensors,
    )
    restored = restore_cuda_boundary_checkpoint(checkpoint, device=receipt.device)
    restored_tensors = restored.projection["tensors"]
    mismatched = tuple(
        name
        for name in sorted(tensors)
        if name not in restored_tensors
        or not _tensor_bytes_equal(tensors[name], restored_tensors[name])
    )
    mechanically_roundtripped = tuple(
        name for name in sorted(tensors) if name not in set(mismatched)
    )
    mechanical_qualified = (
        not mismatched
        and len(mechanically_roundtripped) == receipt.accepted_field_count
    )
    complete_semantic = (
        mechanical_qualified
        and not receipt.canonical_disabled_checkpoint_fields
        and not receipt.refresh_required_checkpoint_fields
        and not receipt.active_prefix_checkpoint_fields
        and not receipt.blocked_checkpoint_fields
    )
    return CudaStateBankCheckpointTransportReceipt(
        state_bank_receipt_sha256=receipt.receipt_sha256,
        checkpoint_projection_digest=checkpoint.projection_digest,
        source_device=checkpoint.device,
        restored_device=restored.device,
        accepted_field_count=receipt.accepted_field_count,
        mechanically_roundtripped_fields=mechanically_roundtripped,
        mismatched_fields=mismatched,
        exact_tensor_checkpoint_fields=receipt.exact_tensor_checkpoint_fields,
        canonical_value_required_checkpoint_fields=(
            receipt.canonical_disabled_checkpoint_fields
        ),
        refresh_required_checkpoint_fields=(
            receipt.refresh_required_checkpoint_fields
        ),
        active_prefix_projection_required_fields=(
            receipt.active_prefix_checkpoint_fields
        ),
        blocked_semantic_checkpoint_fields=receipt.blocked_checkpoint_fields,
        device_to_host_operations=checkpoint.device_to_host_operations,
        device_to_host_bytes=checkpoint.device_to_host_bytes,
        host_to_device_operations=restored.host_to_device_operations,
        host_to_device_bytes=restored.host_to_device_bytes,
        checkpoint_api_device_synchronizations=(
            checkpoint.device_synchronizations + restored.device_synchronizations
        ),
        mechanical_roundtrip_qualified=mechanical_qualified,
        complete_semantic_checkpoint_transport_qualified=complete_semantic,
        simulation_parity_claimed=False,
    )


__all__ = [
    "ACCEPTED_MANIFEST_CONTRACT_STATUS",
    "ACCEPTED_MANIFEST_SHA256",
    "ACCEPTED_MANIFEST_SCHEMA_VERSION",
    "ACCEPTED_TENSOR_FIELD_COUNT",
    "AcceptedStateBankManifest",
    "CHECKPOINT_PROJECTION_SCHEMA_VERSION",
    "CudaResidentStateField",
    "CudaStateBankCheckpointTransportReceipt",
    "CudaStateBankValidationReceipt",
    "DEFAULT_ACCEPTED_MANIFEST",
    "DEFAULT_WDL_MANIFEST",
    "StateBankFieldContract",
    "StateBankManifestError",
    "StateBankValidationIssue",
    "load_accepted_state_bank_manifest",
    "load_wdl_state_bank_manifest",
    "validate_cuda_resident_state_bank",
    "verify_cuda_state_bank_checkpoint_transport",
    "WDL_MANIFEST_CONTRACT_STATUS",
    "WDL_MANIFEST_SCHEMA_VERSION",
    "WDL_MANIFEST_SHA256",
    "WDL_TENSOR_FIELD_COUNT",
]
