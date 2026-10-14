"""P1.2 RowV1/CUDA ordinary-birth receipt entry gate.

This module freezes the next receipt authority boundary without promoting CUDA.
It records a source-bound inventory, a canonical RowV1 semantic row, fixture
requirements, and fail-closed admission checks for RowV1 and CUDA routes.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Final, Mapping, Sequence

from engine.authoritative_state.schema import content_sha256, file_sha256
from engine.evolution.p12_native_cpu_parity_progress_v1 import (
    NATIVE_CORE_DISABLED_REASON_RECEIPT_BYTES_CONTRACT_ID,
    NATIVE_CORE_DISABLED_REASON_SEMANTICS_CONTRACT_ID,
    NATIVE_CORE_RECEIPT_REPEAT_CONTRACT_ID,
    native_core_ordinary_birth_disabled_reason_semantics_v1,
)
from engine.perf.recurrent_native.native_core_ordinary_birth_receipt_bridge_v1 import (
    DISABLED_REASON_BY_CODE,
    TRANSACTION_CAPACITY_UNAVAILABLE,
    TRANSACTION_COMMITTED,
    TRANSACTION_INVALID_DRAW,
    TRANSACTION_MUTATION_DISABLED,
    TRANSACTION_OWNER_MISMATCH,
    TRANSACTION_PUBLICATION_REJECTED,
    build_recurrent_receipt_from_native_core_ordinary_birth_v1,
)


SCHEMA_VERSION: Final[str] = "p12_rowv1_cuda_entry_gate_v1"
SURFACE_INVENTORY_CONTRACT_ID: Final[str] = (
    "p12-rowv1-cuda-ordinary-birth-receipt-surface-inventory-v1"
)
ROWV1_SCHEMA_CONTRACT_ID: Final[str] = (
    "p12-rowv1-ordinary-birth-semantic-receipt-row-v1"
)
ROWV1_PROJECTION_CONTRACT_ID: Final[str] = (
    "p12-rowv1-ordinary-birth-receipt-projection-v1"
)
FIXTURE_BANK_CONTRACT_ID: Final[str] = (
    "p12-rowv1-ordinary-birth-receipt-fixture-bank-v1"
)
ENTRY_GATE_CONTRACT_ID: Final[str] = (
    "p12-rowv1-cuda-ordinary-birth-failclosed-entry-gate-v1"
)
ROWV1_DECODER_ADMISSION_CONTRACT_ID: Final[str] = (
    "p12-rowv1-ordinary-birth-decoder-admission-v1"
)
CUDA_RECEIPT_SURFACE_ADMISSION_CONTRACT_ID: Final[str] = (
    "p12-cuda-ordinary-birth-receipt-surface-admission-v1"
)
CUDA_RAW_V2_ROWV1_PROJECTION_CONTRACT_ID: Final[str] = (
    "p12-cuda-raw-v2-ordinary-birth-rowv1-projection-v1"
)
CUDA_ROWV1_V3_KERNEL_MIRROR_PROJECTION_CONTRACT_ID: Final[str] = (
    "p12-cuda-rowv1-v3-kernel-mirror-projection-v1"
)
CUDA_ROWV1_V3_MINIMUM_ABI_CONTRACT_ID: Final[str] = (
    "p12-cuda-rowv1-ordinary-birth-minimum-receipt-abi-v3"
)
CUDA_ROWV1_DISABLED_EVENT_WRITER_CONTRACT_ID: Final[str] = (
    "p12-cuda-rowv1-disabled-event-writer-v1"
)
CUDA_ROWV1_MODULAR_DEVICE_WRITER_ADMISSION_CONTRACT_ID: Final[str] = (
    "p12-cuda-rowv1-modular-device-writer-admission-v1"
)
CUDA_ROWV1_COMPACT_CPU_CUDA_PARITY_CONTRACT_ID: Final[str] = (
    "p12-cuda-rowv1-compact-cpu-cuda-parity-v1"
)
CHECKPOINT_REPORT_ENVELOPE_CONTRACT_ID: Final[str] = (
    "p12-rowv1-ordinary-birth-checkpoint-report-envelope-v1"
)
CHECKPOINT_REPORT_COLLECTION_CONTRACT_ID: Final[str] = (
    "p12-rowv1-ordinary-birth-checkpoint-report-collection-v1"
)
DECISION: Final[str] = "P12-ROWV1-CUDA-ENTRY-A"
ROWV1_SEMANTIC_SCHEMA_VERSION: Final[str] = (
    "rowv1_ordinary_birth_receipt_semantic_v1"
)
TRANSACTION_SEQUENCE_SCOPE: Final[str] = (
    "per_cell_ordinary_birth_lifecycle_sequence_v1"
)
CHECKPOINT_REPORT_ENVELOPE_VERSION: Final[str] = (
    "rowv1_ordinary_birth_checkpoint_report_envelope_v1"
)
CHECKPOINT_REPORT_COLLECTION_VERSION: Final[str] = (
    "rowv1_ordinary_birth_checkpoint_report_collection_v1"
)
SUPPORTED_ROWV1_SOURCE_ABI_VERSIONS: Final[tuple[int, ...]] = (2, 3, 4)
CUDA_RAW_V2_RECEIPT_ROW_WIDTH: Final[int] = 21
CUDA_ROWV1_V3_RECEIPT_ABI_VERSION: Final[int] = 3
CUDA_ROWV1_V3_MINIMUM_RECEIPT_ROW_WIDTH: Final[int] = 64
CUDA_ROWV1_V3_SEQUENCE_CAPACITY: Final[int] = 3
SEMANTIC_MAPPING_CONTRACT_SHA256: Final[str] = (
    native_core_ordinary_birth_disabled_reason_semantics_v1()["contract_sha256"]
)
_SHA256_RE: Final[re.Pattern[str]] = re.compile(r"[0-9a-f]{64}\Z")
_INT32_MIN: Final[int] = -(1 << 31)
_INT32_MAX: Final[int] = (1 << 31) - 1
_INT64_MIN: Final[int] = -(1 << 63)
_INT64_MAX: Final[int] = (1 << 63) - 1

ROWV1_REQUIRED_FIELDS: Final[tuple[dict[str, str], ...]] = (
    {"name": "semantic_schema_version", "dtype": "string", "nullable": "false"},
    {"name": "source_receipt_abi_version", "dtype": "int32", "nullable": "false"},
    {"name": "source_receipt_sha256", "dtype": "sha256", "nullable": "false"},
    {"name": "semantic_mapping_contract_sha256", "dtype": "sha256", "nullable": "false"},
    {"name": "native_transaction_status", "dtype": "int32", "nullable": "false"},
    {"name": "raw_disabled_reason_code", "dtype": "int32", "nullable": "true"},
    {"name": "canonical_status", "dtype": "string", "nullable": "false"},
    {"name": "canonical_reason", "dtype": "string", "nullable": "false"},
    {"name": "transaction_id", "dtype": "uint64", "nullable": "false"},
    {
        "name": "authoritative_transaction_sequence",
        "dtype": "uint64",
        "nullable": "false",
    },
    {"name": "transaction_sequence_scope", "dtype": "string", "nullable": "false"},
    {"name": "parent_agent_id", "dtype": "int64", "nullable": "false"},
    {"name": "parent_slot", "dtype": "int32", "nullable": "false"},
    {"name": "parent_slot_token", "dtype": "int64", "nullable": "false"},
    {"name": "child_agent_id", "dtype": "int64", "nullable": "true"},
    {"name": "child_slot", "dtype": "int32", "nullable": "true"},
    {"name": "child_slot_token", "dtype": "int64", "nullable": "true"},
    {"name": "cell", "dtype": "int32", "nullable": "false"},
    {"name": "generation", "dtype": "int64", "nullable": "false"},
    {"name": "tick", "dtype": "int64", "nullable": "false"},
    {"name": "resident_slot", "dtype": "int32", "nullable": "false"},
    {"name": "mutation_family", "dtype": "string", "nullable": "true"},
    {"name": "operator_id", "dtype": "string", "nullable": "true"},
    {"name": "attempt_index", "dtype": "int32", "nullable": "false"},
    {"name": "rng_stream_id", "dtype": "string", "nullable": "false"},
    {"name": "rng_index_before", "dtype": "uint64", "nullable": "false"},
    {"name": "rng_index_after", "dtype": "uint64", "nullable": "false"},
    {"name": "proposal_digest", "dtype": "sha256", "nullable": "true"},
    {"name": "selected_identity", "dtype": "canonical_json_or_null", "nullable": "true"},
    {"name": "changed", "dtype": "bool", "nullable": "false"},
    {"name": "value_before", "dtype": "canonical_json_or_null", "nullable": "true"},
    {"name": "value_after", "dtype": "canonical_json_or_null", "nullable": "true"},
    {"name": "capacity_decision", "dtype": "string", "nullable": "false"},
    {"name": "publication_decision", "dtype": "string", "nullable": "false"},
    {"name": "rollback_decision", "dtype": "string", "nullable": "false"},
    {"name": "allocator_before", "dtype": "canonical_json", "nullable": "false"},
    {"name": "allocator_after", "dtype": "canonical_json", "nullable": "false"},
    {"name": "package_binding_sha256", "dtype": "sha256", "nullable": "false"},
    {"name": "schema_binding_sha256", "dtype": "sha256", "nullable": "false"},
    {"name": "mechanics_binding_sha256", "dtype": "sha256", "nullable": "false"},
)
ROWV1_FIELD_NAMES: Final[tuple[str, ...]] = tuple(
    row["name"] for row in ROWV1_REQUIRED_FIELDS
)

ROWV1_NEGATIVE_CASES: Final[tuple[str, ...]] = (
    "missing_source_receipt_abi_version",
    "missing_source_receipt_sha256",
    "v4_without_disabled_reason_code",
    "unknown_abi_version",
    "unknown_disabled_reason_code",
    "stale_mapping_contract",
    "changed_source_bytes",
    "changed_canonical_reason",
    "populated_mutation_fields_on_disabled_or_rejected_row",
    "dense_id_lookup_through_agent_id_to_slot_shadow",
    "stable_id_truncation",
    "stale_slot_token",
)

CUDA_FAIL_CLOSED_CASES: Final[tuple[str, ...]] = (
    "no_authoritative_cuda_receipt_abi",
    "cuda_receipt_abi_version_undeclared",
    "device_layout_cannot_preserve_rowv1_semantics",
    "host_device_size_or_alignment_differs",
    "decoder_width_stale",
    "checkpoint_schema_omits_source_abi_or_disabled_reason",
    "package_does_not_bind_receipt_abi",
    "route_does_not_bind_rowv1_mapping_contract",
    "sanitizer_coverage_absent",
    "dense_id_dependence_remaining",
)

ROWV1_DECODER_ADMISSION_FIELDS: Final[tuple[str, ...]] = (
    "decoder_contract_id",
    "decoder_schema_version",
    "supported_source_receipt_abi_versions",
    "decoder_row_width",
    "expected_row_width",
    "allow_mixed_source_abi_versions",
)

CUDA_RECEIPT_SURFACE_ADMISSION_FIELDS: Final[tuple[str, ...]] = (
    "admission_contract_id",
    "authoritative_cuda_receipt_abi_present",
    "declared_cuda_receipt_abi_version",
    "device_layout_preserves_rowv1_semantics",
    "receipt_surface_present",
    "receipt_surface_row_count",
    "host_receipt_size_bytes",
    "device_receipt_size_bytes",
    "host_receipt_alignment_bytes",
    "device_receipt_alignment_bytes",
    "row_dtype",
    "counts_dtype",
    "overflow_dtype",
    "receipt_row_width",
    "expected_receipt_row_width",
    "decoder_row_width",
    "checkpoint_manifest_receipt_width",
    "checkpoint_manifest_receipt_abi_version",
    "checkpoint_manifest_preserves_source_abi",
    "checkpoint_manifest_preserves_disabled_reason",
    "package_receipt_abi_sha256",
    "route_rowv1_mapping_contract_sha256",
    "sanitizer_bounds_cases",
    "dense_id_authority",
)

CHECKPOINT_REPORT_BOUNDARIES: Final[tuple[str, ...]] = (
    "checkpoint_export",
    "disk_publication",
    "fresh_owner_restore",
    "repeated_restore",
    "multicell_restore",
    "reporting_enabled",
    "reporting_disabled",
    "cancellation",
    "terminal_state",
)

CHECKPOINT_REPORT_ENVELOPE_FIELDS: Final[tuple[str, ...]] = (
    "envelope_schema_version",
    "contract_id",
    "rowv1_schema_contract_id",
    "rowv1_schema_sha256",
    "rowv1_projection_contract_id",
    "rowv1_projection_sha256",
    "semantic_mapping_contract_sha256",
    "boundary",
    "checkpoint_generation",
    "checkpoint_tick",
    "checkpoint_cell",
    "reporting_enabled",
    "cancellation_state",
    "terminal_state",
    "restore_scope",
    "disk_publication_state",
    "rowv1_receipt_sha256",
    "authoritative_transaction_sequence",
    "transaction_sequence_scope",
    "lifecycle_ledger_sha256",
    "source_checkpoint_boundary",
    "source_receipt_abi_version",
    "source_receipt_sha256",
    "raw_disabled_reason_code",
    "canonical_status",
    "canonical_reason",
    "rng_stream_id",
    "rng_index_before",
    "rng_index_after",
    "allocator_before",
    "allocator_after",
    "package_binding_sha256",
    "schema_binding_sha256",
    "mechanics_binding_sha256",
    "rowv1_receipt",
)

CHECKPOINT_REPORT_COLLECTION_SEGMENTS: Final[tuple[str, ...]] = (
    "uninterrupted",
    "checkpoint_prefix",
    "resumed_suffix",
    "checkpoint_plus_resume",
    "terminal",
)
EMPTY_COLLECTION_POLICIES: Final[tuple[str, ...]] = (
    "not_empty",
    "legitimate_empty",
)
EXPECTED_RECEIPT_COUNT_SOURCES: Final[tuple[str, ...]] = (
    "authoritative_lifecycle_ledger",
)
COLLECTION_CAPACITY_POLICIES: Final[tuple[str, ...]] = (
    "fail_closed_on_capacity_exceeded",
)

CHECKPOINT_REPORT_COLLECTION_FIELDS: Final[tuple[str, ...]] = (
    "collection_schema_version",
    "contract_id",
    "rowv1_checkpoint_report_envelope_contract_id",
    "rowv1_checkpoint_report_envelope_sha256",
    "run_id",
    "source_commit_sha",
    "route_binding_sha256",
    "package_binding_sha256",
    "boundary",
    "segment_kind",
    "checkpoint_lineage_id",
    "parent_collection_sha256",
    "prefix_collection_sha256",
    "join_binding_sha256",
    "receipt_count",
    "expected_receipt_count",
    "attempted_transaction_count",
    "expected_receipt_count_source",
    "lifecycle_ledger_sha256",
    "empty_collection_policy",
    "expected_cells",
    "cell_count",
    "transaction_sequence_scope",
    "sequence_first",
    "sequence_last",
    "last_prefix_sequence",
    "first_suffix_sequence",
    "ordered_transaction_sequences",
    "per_cell_sequence_ranges",
    "per_cell_collection_sha256s",
    "global_collection_sha256",
    "collector_success_state",
    "per_cell_expected_receipt_counts",
    "per_cell_receipt_counts",
    "generation_min",
    "generation_max",
    "tick_min",
    "tick_max",
    "collection_capacity_limit",
    "collection_capacity_policy",
    "overflow_count",
    "dropped_receipt_count",
    "cancellation_state",
    "sealed",
    "cancellation_seal_sha256",
    "checkpoint_state_sha256",
    "rng_state_sha256",
    "allocator_state_sha256",
    "identity_table_sha256",
    "owner_lineage_sha256",
    "deterministic_order_key",
    "transaction_identity_key",
    "receipt_envelope_sha256s",
    "receipt_envelopes",
)


class P12RowV1CudaEntryGateError(ValueError):
    """The RowV1/CUDA entry gate input is invalid."""


def _with_sha(payload: Mapping[str, Any], field: str = "contract_sha256") -> dict[str, Any]:
    result = dict(payload)
    result[field] = content_sha256(result)
    return result


def _source_row(root: Path, path: str, marker: str) -> dict[str, Any]:
    full = root / path
    text = full.read_text(encoding="utf-8")
    if marker not in text:
        raise P12RowV1CudaEntryGateError(f"missing source marker {marker!r} in {path}")
    return {
        "path": path,
        "marker_sha256": content_sha256({"path": path, "marker": marker}),
        "line": text[: text.index(marker)].count("\n") + 1,
        "source_sha256": file_sha256(full),
    }


def canonical_rowv1_cuda_entry_bytes_v1(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            dict(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")


def cuda_rowv1_v3_minimum_receipt_abi_contract_v1() -> dict[str, Any]:
    """Freeze the minimum CUDA receipt ABI needed for RowV1 admission."""

    payload = {
        "schema_version": SCHEMA_VERSION,
        "contract_id": CUDA_ROWV1_V3_MINIMUM_ABI_CONTRACT_ID,
        "decision": "P12-CUDA-ROWV1-V3-MINIMUM-ABI-PREFLIGHT",
        "declared_cuda_receipt_abi_version": CUDA_ROWV1_V3_RECEIPT_ABI_VERSION,
        "source_layout_superseded": "ordinary_birth_mutation_receipt_raw_layout_v2",
        "source_layout_superseded_width": CUDA_RAW_V2_RECEIPT_ROW_WIDTH,
        "minimum_receipt_row_width": CUDA_ROWV1_V3_MINIMUM_RECEIPT_ROW_WIDTH,
        "row_dtype": "int64",
        "counts_dtype": "int64",
        "overflow_dtype": "int32",
        "required_device_semantics": (
            "source_receipt_abi_version",
            "source_receipt_sha256_words_4x64",
            "semantic_mapping_contract_sha256_words_4x64",
            "raw_disabled_reason_code",
            "canonical_status_code",
            "canonical_reason_code",
            "authoritative_transaction_sequence",
            "parent_child_agent_ids_int64",
            "slot_and_slot_token_owner_identity",
            "rng_stream_and_indexes",
            "proposal_digest_sha256_words_4x64_or_zero",
            "selected_identity_codes",
            "value_before_after_bits",
            "allocator_before_after_state",
            "package_schema_mechanics_binding_sha256_words_12x64",
        ),
        "required_host_device_checks": (
            "host_device_size_match",
            "host_device_alignment_match",
            "decoder_width_matches_device_width",
            "checkpoint_manifest_width_matches_device_width",
            "checkpoint_manifest_abi_matches_device_abi",
            "checkpoint_manifest_preserves_source_abi",
            "checkpoint_manifest_preserves_disabled_reason",
            "package_binds_this_cuda_receipt_abi_contract",
            "route_binds_rowv1_semantic_mapping_contract",
            "sanitizer_covers_rows_counts_overflow",
            "dense_id_authority_is_sparse_identity_table_and_slot_token",
        ),
        "current_raw_v2_admissible": False,
        "current_entry_surface_present": True,
        "current_entry_surface_authoritative": False,
        "cuda_receipt_surface_currently_admitted": False,
        "cuda_abi_implementation_authorized": False,
        "cuda_parity_authorized": False,
        "route_package_qualification_authorized": False,
        "successor_campaign_authorized": False,
    }
    return _with_sha(payload)


def rowv1_source_receipt_sha256_v1(
    native_receipt: Mapping[str, Any],
    *,
    source_receipt_abi_version: int,
) -> str:
    """Bind one source native receipt envelope for RowV1 projection."""

    if type(source_receipt_abi_version) is not int:
        raise P12RowV1CudaEntryGateError("source receipt ABI version must be an int")
    return content_sha256(
        {
            "source_receipt_abi_version": source_receipt_abi_version,
            "native_receipt": dict(native_receipt),
        }
    )


def canonical_rowv1_ordinary_birth_receipt_bytes_v1(row: Mapping[str, Any]) -> bytes:
    """Return canonical RowV1 bytes only after strict validation."""

    return canonical_rowv1_cuda_entry_bytes_v1(
        validate_rowv1_ordinary_birth_receipt_v1(row)
    )


def rowv1_ordinary_birth_receipt_sha256_v1(row: Mapping[str, Any]) -> str:
    return content_sha256(validate_rowv1_ordinary_birth_receipt_v1(row))


def canonical_rowv1_checkpoint_report_envelope_bytes_v1(
    envelope: Mapping[str, Any],
) -> bytes:
    """Return canonical bytes for one validated checkpoint/report envelope."""

    return canonical_rowv1_cuda_entry_bytes_v1(
        validate_rowv1_checkpoint_report_envelope_v1(envelope)
    )


def rowv1_checkpoint_report_envelope_sha256_v1(envelope: Mapping[str, Any]) -> str:
    return content_sha256(validate_rowv1_checkpoint_report_envelope_v1(envelope))


def canonical_rowv1_checkpoint_report_collection_bytes_v1(
    collection: Mapping[str, Any],
) -> bytes:
    """Return canonical bytes for one validated receipt collection envelope."""

    return canonical_rowv1_cuda_entry_bytes_v1(
        validate_rowv1_checkpoint_report_collection_v1(collection)
    )


def rowv1_checkpoint_report_collection_sha256_v1(
    collection: Mapping[str, Any],
) -> str:
    return content_sha256(validate_rowv1_checkpoint_report_collection_v1(collection))


def _require_int(
    value: Any,
    *,
    field: str,
    lower: int | None = None,
    upper: int | None = None,
) -> int:
    if type(value) is not int:
        raise P12RowV1CudaEntryGateError(f"{field} must be an int")
    if lower is not None and value < lower:
        raise P12RowV1CudaEntryGateError(f"{field} is below supported range")
    if upper is not None and value > upper:
        raise P12RowV1CudaEntryGateError(f"{field} is above supported range")
    return value


def _require_int32(value: Any, *, field: str) -> int:
    return _require_int(value, field=field, lower=_INT32_MIN, upper=_INT32_MAX)


def _require_int64(value: Any, *, field: str) -> int:
    return _require_int(value, field=field, lower=_INT64_MIN, upper=_INT64_MAX)


def _require_uint64(value: Any, *, field: str) -> int:
    return _require_int(value, field=field, lower=0, upper=(1 << 64) - 1)


def _require_sha256(value: Any, *, field: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise P12RowV1CudaEntryGateError(f"{field} must be lowercase SHA-256 hex")
    return value


def _require_text(value: Any, *, field: str) -> str:
    if type(value) is not str or value == "":
        raise P12RowV1CudaEntryGateError(f"{field} must be a nonempty string")
    return value


def _require_bool(value: Any, *, field: str) -> bool:
    if type(value) is not bool:
        raise P12RowV1CudaEntryGateError(f"{field} must be bool")
    return value


def _require_choice(value: Any, *, field: str, choices: tuple[str, ...]) -> str:
    text = _require_text(value, field=field)
    if text not in choices:
        raise P12RowV1CudaEntryGateError(f"unsupported {field}")
    return text


def _require_sequence(value: Any, *, field: str) -> list[Any]:
    if not isinstance(value, (list, tuple)):
        raise P12RowV1CudaEntryGateError(f"{field} must be a list or tuple")
    return list(value)


def _require_git_commit(value: Any, *, field: str) -> str:
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{8,40}", value) is None:
        raise P12RowV1CudaEntryGateError(f"{field} must be lowercase git hex")
    return value


def _proposal_digest_sha256(
    native_receipt: Mapping[str, Any],
    *,
    proposal_digest_sha256: str | None,
) -> str | None:
    value = _require_uint64(native_receipt.get("proposal_digest", 0), field="proposal_digest")
    if value == 0:
        if proposal_digest_sha256 is not None:
            raise P12RowV1CudaEntryGateError(
                "proposal_digest_sha256 cannot be supplied for zero proposal digest"
            )
        return None
    if proposal_digest_sha256 is None:
        raise P12RowV1CudaEntryGateError(
            "nonzero proposal digest requires proposal_digest_sha256"
        )
    return _require_sha256(proposal_digest_sha256, field="proposal_digest_sha256")


def _transaction_decisions(transaction_status: int) -> tuple[str, str, str]:
    if transaction_status == TRANSACTION_COMMITTED:
        return "available", "published", "committed"
    if transaction_status == TRANSACTION_CAPACITY_UNAVAILABLE:
        return "unavailable", "not_attempted", "child_unchanged"
    if transaction_status == TRANSACTION_PUBLICATION_REJECTED:
        return "available", "rejected", "child_unchanged"
    if transaction_status == TRANSACTION_OWNER_MISMATCH:
        return "not_evaluated", "not_attempted", "child_unchanged"
    if transaction_status == TRANSACTION_INVALID_DRAW:
        return "not_evaluated", "not_attempted", "child_unchanged"
    if transaction_status == TRANSACTION_MUTATION_DISABLED:
        return "available", "not_attempted", "child_unchanged"
    return "not_evaluated", "not_attempted", "child_unchanged"


def _raw_disabled_reason_code(
    native_receipt: Mapping[str, Any],
    *,
    source_receipt_abi_version: int,
    transaction_status: int,
) -> int | None:
    value = native_receipt.get("disabled_reason_code")
    if source_receipt_abi_version == 4:
        if value is None:
            if transaction_status == TRANSACTION_MUTATION_DISABLED:
                raise P12RowV1CudaEntryGateError(
                    "v4 disabled row requires raw disabled reason code"
                )
            return None
        code = _require_int32(value, field="disabled_reason_code")
        if transaction_status == TRANSACTION_MUTATION_DISABLED:
            if code not in DISABLED_REASON_BY_CODE:
                raise P12RowV1CudaEntryGateError("unknown disabled reason code")
            return code
        if code != 0:
            raise P12RowV1CudaEntryGateError(
                "non-disabled v4 row cannot carry a disabled reason"
            )
        return 0
    if source_receipt_abi_version == 3 and transaction_status == TRANSACTION_MUTATION_DISABLED:
        if value is not None:
            raise P12RowV1CudaEntryGateError(
                "v3 disabled row must not invent raw disabled reason code"
            )
        return None
    if value not in (None, 0):
        raise P12RowV1CudaEntryGateError(
            "non-v4 row cannot carry a disabled reason code"
        )
    return None


def project_native_core_ordinary_birth_receipt_rowv1_v1(
    native_receipt: Mapping[str, Any],
    *,
    source_receipt_abi_version: int,
    transaction_id: int,
    cell: int,
    generation: int,
    tick: int,
    rng_stream_id: str,
    rng_index_before: int,
    rng_index_after: int,
    allocator_before: Mapping[str, Any],
    allocator_after: Mapping[str, Any],
    package_binding_sha256: str,
    schema_binding_sha256: str,
    mechanics_binding_sha256: str,
    selected_identity: Mapping[str, Any] | None = None,
    selected_instance_id_index: int = -1,
    selected_homology_id_index: int = -1,
    proposal_digest_sha256: str | None = None,
    expected_source_receipt_sha256: str | None = None,
    source_packed_sha256: str = "0" * 64,
    ancestry_tape_sha256: str = "0" * 64,
    profile_id: str = "b2-16x48-v1",
    authoritative_transaction_sequence: int | None = None,
) -> dict[str, Any]:
    """Project a native ordinary-birth receipt into the frozen RowV1 row."""

    if not isinstance(native_receipt, Mapping):
        raise P12RowV1CudaEntryGateError("native receipt must be a mapping")
    source_receipt_abi_version = _require_int32(
        source_receipt_abi_version,
        field="source_receipt_abi_version",
    )
    if source_receipt_abi_version not in SUPPORTED_ROWV1_SOURCE_ABI_VERSIONS:
        raise P12RowV1CudaEntryGateError("unsupported source receipt ABI version")
    source_receipt_sha256 = rowv1_source_receipt_sha256_v1(
        native_receipt,
        source_receipt_abi_version=source_receipt_abi_version,
    )
    if expected_source_receipt_sha256 is not None:
        _require_sha256(
            expected_source_receipt_sha256,
            field="expected_source_receipt_sha256",
        )
        if source_receipt_sha256 != expected_source_receipt_sha256:
            raise P12RowV1CudaEntryGateError("source receipt hash mismatch")

    transaction_status = _require_int32(
        native_receipt.get("transaction_status"),
        field="transaction_status",
    )
    raw_disabled_reason = _raw_disabled_reason_code(
        native_receipt,
        source_receipt_abi_version=source_receipt_abi_version,
        transaction_status=transaction_status,
    )
    recurrent_receipt = build_recurrent_receipt_from_native_core_ordinary_birth_v1(
        native_receipt,
        profile_id=profile_id,
        source_packed_sha256=_require_sha256(
            source_packed_sha256,
            field="source_packed_sha256",
        ),
        ancestry_tape_sha256=_require_sha256(
            ancestry_tape_sha256,
            field="ancestry_tape_sha256",
        ),
        selected_instance_id_index=selected_instance_id_index,
        selected_homology_id_index=selected_homology_id_index,
    )
    transaction = recurrent_receipt.transactions[0]
    if transaction.changed and selected_identity is None:
        raise P12RowV1CudaEntryGateError(
            "accepted changed row requires selected_identity"
        )
    if not transaction.changed and selected_identity is not None:
        raise P12RowV1CudaEntryGateError(
            "unchanged, rejected, or disabled row must clear selected_identity"
        )
    capacity_decision, publication_decision, rollback_decision = (
        _transaction_decisions(transaction_status)
    )
    row = {
        "semantic_schema_version": ROWV1_SEMANTIC_SCHEMA_VERSION,
        "source_receipt_abi_version": source_receipt_abi_version,
        "source_receipt_sha256": source_receipt_sha256,
        "semantic_mapping_contract_sha256": SEMANTIC_MAPPING_CONTRACT_SHA256,
        "native_transaction_status": transaction_status,
        "raw_disabled_reason_code": raw_disabled_reason,
        "canonical_status": transaction.status,
        "canonical_reason": transaction.reason,
        "transaction_id": _require_uint64(transaction_id, field="transaction_id"),
        "authoritative_transaction_sequence": _require_uint64(
            transaction_id
            if authoritative_transaction_sequence is None
            else authoritative_transaction_sequence,
            field="authoritative_transaction_sequence",
        ),
        "transaction_sequence_scope": TRANSACTION_SEQUENCE_SCOPE,
        "parent_agent_id": transaction.parent_agent_id,
        "parent_slot": transaction.parent_slot,
        "parent_slot_token": transaction.parent_slot_token,
        "child_agent_id": transaction.child_agent_id,
        "child_slot": transaction.child_slot,
        "child_slot_token": transaction.child_slot_token,
        "cell": _require_int32(cell, field="cell"),
        "generation": _require_int64(generation, field="generation"),
        "tick": _require_int64(tick, field="tick"),
        "resident_slot": transaction.child_slot,
        "mutation_family": "modular_recurrent",
        "operator_id": transaction.operation,
        "attempt_index": transaction.attempt_ordinal,
        "rng_stream_id": _require_text(rng_stream_id, field="rng_stream_id"),
        "rng_index_before": _require_uint64(rng_index_before, field="rng_index_before"),
        "rng_index_after": _require_uint64(rng_index_after, field="rng_index_after"),
        "proposal_digest": _proposal_digest_sha256(
            native_receipt,
            proposal_digest_sha256=proposal_digest_sha256,
        ),
        "selected_identity": dict(selected_identity) if selected_identity is not None else None,
        "changed": transaction.changed,
        "value_before": (
            {"float32": native_receipt.get("value_before_float32")}
            if transaction.changed
            else None
        ),
        "value_after": (
            {"float32": native_receipt.get("value_after_float32")}
            if transaction.changed
            else None
        ),
        "capacity_decision": capacity_decision,
        "publication_decision": publication_decision,
        "rollback_decision": rollback_decision,
        "allocator_before": dict(allocator_before),
        "allocator_after": dict(allocator_after),
        "package_binding_sha256": _require_sha256(
            package_binding_sha256,
            field="package_binding_sha256",
        ),
        "schema_binding_sha256": _require_sha256(
            schema_binding_sha256,
            field="schema_binding_sha256",
        ),
        "mechanics_binding_sha256": _require_sha256(
            mechanics_binding_sha256,
            field="mechanics_binding_sha256",
        ),
    }
    return validate_rowv1_ordinary_birth_receipt_v1(row)


def _cuda_raw_v2_operation(row: Sequence[int]) -> tuple[str, str, bool]:
    operation = {2: "refine_descendant_parameter"}.get(row[13])
    status = {1: "accepted", 3: "ineligible", 4: "noop"}.get(row[14])
    reason = {
        1: "proposal_valid",
        2: "no_duplicate_descendant_parameter",
        4: "refined_value_unchanged",
    }.get(row[15])
    dimension = {
        0: "none",
        1: "expression_scale",
        2: "live_retention",
        3: "live_input_gain",
        4: "live_output_gain",
        5: "reset_bias",
        6: "reset_input_gain",
        7: "reset_state_gain",
        8: "update_bias",
        9: "update_input_gain",
        10: "update_state_gain",
        11: "candidate_bias",
        12: "candidate_input_gain",
        13: "candidate_state_gain",
        14: "output_gain",
    }.get(row[19])
    changed = row[20] == 1
    if None in (operation, status, reason, dimension):
        raise P12RowV1CudaEntryGateError("CUDA raw v2 row has unsupported enums")
    if changed and status != "accepted":
        raise P12RowV1CudaEntryGateError("CUDA raw v2 non-accepted row changed")
    return operation, status, changed


def _cuda_rowv1_v3_transaction_authority(
    *,
    raw_disabled_reason: int,
    status: str,
    changed: bool,
) -> tuple[int, str, str]:
    if raw_disabled_reason == 0:
        return (
            TRANSACTION_COMMITTED,
            status,
            "cuda_rowv1_v3_kernel_mirror_" + status,
        )
    canonical_reason = DISABLED_REASON_BY_CODE.get(raw_disabled_reason)
    if canonical_reason is None:
        raise P12RowV1CudaEntryGateError("unknown CUDA RowV1 disabled reason code")
    if changed:
        raise P12RowV1CudaEntryGateError("disabled CUDA RowV1 v3 row changed")
    return TRANSACTION_MUTATION_DISABLED, "disabled", canonical_reason


def project_cuda_raw_v2_ordinary_birth_receipt_rowv1_v1(
    cuda_raw_row: Sequence[int],
    *,
    transaction_id: int,
    cell: int,
    rng_stream_id: str,
    rng_index_before: int,
    rng_index_after: int,
    allocator_before: Mapping[str, Any],
    allocator_after: Mapping[str, Any],
    package_binding_sha256: str,
    schema_binding_sha256: str,
    mechanics_binding_sha256: str,
    selected_identity: Mapping[str, Any] | None = None,
    authoritative_transaction_sequence: int | None = None,
) -> dict[str, Any]:
    """Project one current CUDA raw layout-v2 ordinary-birth row into RowV1.

    This is a host projection for the current accepted raw staging layout. It
    does not admit CUDA as a final authoritative receipt ABI because v2 still
    lacks disabled-reason and package/checkpoint ABI fields.
    """

    raw_values = tuple(
        _require_int(value, field="cuda_raw_row")
        for value in _require_sequence(cuda_raw_row, field="cuda_raw_row")
    )
    if len(raw_values) != CUDA_RAW_V2_RECEIPT_ROW_WIDTH:
        raise P12RowV1CudaEntryGateError("CUDA raw v2 row width changed")
    if raw_values[0] != 1:
        raise P12RowV1CudaEntryGateError("CUDA raw v2 row is not committed")
    if raw_values[1] < 0:
        raise P12RowV1CudaEntryGateError("CUDA raw v2 sequence index is invalid")
    if raw_values[4] < 0 or raw_values[5] < 0 or raw_values[7] < 0 or raw_values[8] < 0:
        raise P12RowV1CudaEntryGateError("CUDA raw v2 owner coordinates are invalid")
    if raw_values[10] < 0:
        raise P12RowV1CudaEntryGateError("CUDA raw v2 attempt index is invalid")

    operation, status, changed = _cuda_raw_v2_operation(raw_values)
    if changed and selected_identity is None:
        raise P12RowV1CudaEntryGateError(
            "accepted CUDA raw v2 changed row requires selected_identity"
        )
    if not changed and selected_identity is not None:
        raise P12RowV1CudaEntryGateError(
            "unchanged CUDA raw v2 row must clear selected_identity"
        )

    source_receipt = {
        "source_receipt_abi_version": 2,
        "cuda_raw_layout": "ordinary_birth_mutation_receipt_raw_layout_v2",
        "cuda_raw_row": list(raw_values),
    }
    source_receipt_sha256 = rowv1_source_receipt_sha256_v1(
        source_receipt,
        source_receipt_abi_version=2,
    )
    row = {
        "semantic_schema_version": ROWV1_SEMANTIC_SCHEMA_VERSION,
        "source_receipt_abi_version": 2,
        "source_receipt_sha256": source_receipt_sha256,
        "semantic_mapping_contract_sha256": SEMANTIC_MAPPING_CONTRACT_SHA256,
        "native_transaction_status": TRANSACTION_COMMITTED,
        "raw_disabled_reason_code": None,
        "canonical_status": status,
        "canonical_reason": "cuda_raw_v2_" + status,
        "transaction_id": _require_uint64(transaction_id, field="transaction_id"),
        "authoritative_transaction_sequence": _require_uint64(
            raw_values[1]
            if authoritative_transaction_sequence is None
            else authoritative_transaction_sequence,
            field="authoritative_transaction_sequence",
        ),
        "transaction_sequence_scope": TRANSACTION_SEQUENCE_SCOPE,
        "parent_agent_id": _require_int64(raw_values[4], field="parent_agent_id"),
        "parent_slot": _require_int32(raw_values[5], field="parent_slot"),
        "parent_slot_token": _require_int64(
            raw_values[6],
            field="parent_slot_token",
        ),
        "child_agent_id": _require_int64(raw_values[7], field="child_agent_id"),
        "child_slot": _require_int32(raw_values[8], field="child_slot"),
        "child_slot_token": _require_int64(raw_values[9], field="child_slot_token"),
        "cell": _require_int32(cell, field="cell"),
        "generation": _require_int64(raw_values[2], field="generation"),
        "tick": _require_int64(raw_values[3], field="tick"),
        "resident_slot": _require_int32(raw_values[8], field="resident_slot"),
        "mutation_family": "modular_recurrent",
        "operator_id": operation,
        "attempt_index": _require_int32(raw_values[10], field="attempt_index"),
        "rng_stream_id": _require_text(rng_stream_id, field="rng_stream_id"),
        "rng_index_before": _require_uint64(rng_index_before, field="rng_index_before"),
        "rng_index_after": _require_uint64(rng_index_after, field="rng_index_after"),
        "proposal_digest": None,
        "selected_identity": dict(selected_identity) if selected_identity is not None else None,
        "changed": changed,
        "value_before": None,
        "value_after": None,
        "capacity_decision": "available",
        "publication_decision": "published",
        "rollback_decision": "committed",
        "allocator_before": dict(allocator_before),
        "allocator_after": dict(allocator_after),
        "package_binding_sha256": _require_sha256(
            package_binding_sha256,
            field="package_binding_sha256",
        ),
        "schema_binding_sha256": _require_sha256(
            schema_binding_sha256,
            field="schema_binding_sha256",
        ),
        "mechanics_binding_sha256": _require_sha256(
            mechanics_binding_sha256,
            field="mechanics_binding_sha256",
        ),
    }
    return validate_rowv1_ordinary_birth_receipt_v1(row)


def project_cuda_rowv1_v3_kernel_mirror_receipt_rowv1_v1(
    cuda_rowv1_row: Sequence[int],
    *,
    transaction_id: int,
    cell: int,
    rng_stream_id: str,
    rng_index_before: int,
    rng_index_after: int,
    allocator_before: Mapping[str, Any],
    allocator_after: Mapping[str, Any],
    package_binding_sha256: str,
    schema_binding_sha256: str,
    mechanics_binding_sha256: str,
    selected_identity: Mapping[str, Any] | None = None,
    authoritative_transaction_sequence: int | None = None,
    expected_source_receipt_sha256: str | None = None,
) -> dict[str, Any]:
    """Project the CUDA RowV1 v3 kernel mirror into the RowV1 host schema.

    The device mirror is a source-bound v3 row carrying the raw-v2 semantic
    prefix and a 64-word width marker. This host projection proves the mirror
    can be decoded and hashed, but it still does not admit CUDA as the
    authoritative ordinary-birth receipt ABI.
    """

    rowv1_values = tuple(
        _require_int(value, field="cuda_rowv1_row")
        for value in _require_sequence(cuda_rowv1_row, field="cuda_rowv1_row")
    )
    if len(rowv1_values) != CUDA_ROWV1_V3_MINIMUM_RECEIPT_ROW_WIDTH:
        raise P12RowV1CudaEntryGateError("CUDA RowV1 v3 mirror row width changed")
    if rowv1_values[0] != CUDA_ROWV1_V3_RECEIPT_ABI_VERSION:
        raise P12RowV1CudaEntryGateError("CUDA RowV1 v3 mirror ABI marker changed")
    if rowv1_values[1] != 2:
        raise P12RowV1CudaEntryGateError("CUDA RowV1 v3 mirror source layout changed")

    raw_values = (1, *rowv1_values[2:22])
    if len(raw_values) != CUDA_RAW_V2_RECEIPT_ROW_WIDTH:
        raise P12RowV1CudaEntryGateError("CUDA RowV1 v3 mirror raw prefix changed")
    operation, status, changed = _cuda_raw_v2_operation(raw_values)
    if raw_values[1] < 0:
        raise P12RowV1CudaEntryGateError("CUDA RowV1 v3 mirror sequence is invalid")
    if raw_values[4] < 0 or raw_values[5] < 0 or raw_values[7] < 0 or raw_values[8] < 0:
        raise P12RowV1CudaEntryGateError("CUDA RowV1 v3 owner coordinates are invalid")
    if raw_values[10] < 0:
        raise P12RowV1CudaEntryGateError("CUDA RowV1 v3 attempt index is invalid")
    raw_disabled_reason = _require_int32(
        rowv1_values[22],
        field="raw_disabled_reason_code",
    )
    (
        transaction_status,
        canonical_status,
        canonical_reason,
    ) = _cuda_rowv1_v3_transaction_authority(
        raw_disabled_reason=raw_disabled_reason,
        status=status,
        changed=changed,
    )
    capacity_decision, publication_decision, rollback_decision = _transaction_decisions(
        transaction_status
    )
    if changed and selected_identity is None:
        raise P12RowV1CudaEntryGateError(
            "accepted CUDA RowV1 v3 changed row requires selected_identity"
        )
    if not changed and selected_identity is not None:
        raise P12RowV1CudaEntryGateError(
            "unchanged CUDA RowV1 v3 row must clear selected_identity"
        )

    source_receipt = {
        "source_receipt_abi_version": CUDA_ROWV1_V3_RECEIPT_ABI_VERSION,
        "cuda_rowv1_layout": "ordinary_birth_rowv1_receipt_kernel_mirror_v3",
        "mirrored_raw_layout_version": rowv1_values[1],
        "mirrored_raw_row": list(raw_values),
        "cuda_rowv1_row": list(rowv1_values),
    }
    source_receipt_sha256 = rowv1_source_receipt_sha256_v1(
        source_receipt,
        source_receipt_abi_version=CUDA_ROWV1_V3_RECEIPT_ABI_VERSION,
    )
    if expected_source_receipt_sha256 is not None:
        _require_sha256(
            expected_source_receipt_sha256,
            field="expected_source_receipt_sha256",
        )
        if source_receipt_sha256 != expected_source_receipt_sha256:
            raise P12RowV1CudaEntryGateError("source receipt hash mismatch")

    row = {
        "semantic_schema_version": ROWV1_SEMANTIC_SCHEMA_VERSION,
        "source_receipt_abi_version": CUDA_ROWV1_V3_RECEIPT_ABI_VERSION,
        "source_receipt_sha256": source_receipt_sha256,
        "semantic_mapping_contract_sha256": SEMANTIC_MAPPING_CONTRACT_SHA256,
        "native_transaction_status": transaction_status,
        "raw_disabled_reason_code": raw_disabled_reason,
        "canonical_status": canonical_status,
        "canonical_reason": canonical_reason,
        "transaction_id": _require_uint64(transaction_id, field="transaction_id"),
        "authoritative_transaction_sequence": _require_uint64(
            raw_values[1]
            if authoritative_transaction_sequence is None
            else authoritative_transaction_sequence,
            field="authoritative_transaction_sequence",
        ),
        "transaction_sequence_scope": TRANSACTION_SEQUENCE_SCOPE,
        "parent_agent_id": _require_int64(raw_values[4], field="parent_agent_id"),
        "parent_slot": _require_int32(raw_values[5], field="parent_slot"),
        "parent_slot_token": _require_int64(
            raw_values[6],
            field="parent_slot_token",
        ),
        "child_agent_id": _require_int64(raw_values[7], field="child_agent_id"),
        "child_slot": _require_int32(raw_values[8], field="child_slot"),
        "child_slot_token": _require_int64(raw_values[9], field="child_slot_token"),
        "cell": _require_int32(cell, field="cell"),
        "generation": _require_int64(raw_values[2], field="generation"),
        "tick": _require_int64(raw_values[3], field="tick"),
        "resident_slot": _require_int32(raw_values[8], field="resident_slot"),
        "mutation_family": "modular_recurrent",
        "operator_id": operation,
        "attempt_index": _require_int32(raw_values[10], field="attempt_index"),
        "rng_stream_id": _require_text(rng_stream_id, field="rng_stream_id"),
        "rng_index_before": _require_uint64(rng_index_before, field="rng_index_before"),
        "rng_index_after": _require_uint64(rng_index_after, field="rng_index_after"),
        "proposal_digest": None,
        "selected_identity": dict(selected_identity) if selected_identity is not None else None,
        "changed": changed,
        "value_before": None,
        "value_after": None,
        "capacity_decision": capacity_decision,
        "publication_decision": publication_decision,
        "rollback_decision": rollback_decision,
        "allocator_before": dict(allocator_before),
        "allocator_after": dict(allocator_after),
        "package_binding_sha256": _require_sha256(
            package_binding_sha256,
            field="package_binding_sha256",
        ),
        "schema_binding_sha256": _require_sha256(
            schema_binding_sha256,
            field="schema_binding_sha256",
        ),
        "mechanics_binding_sha256": _require_sha256(
            mechanics_binding_sha256,
            field="mechanics_binding_sha256",
        ),
    }
    return validate_rowv1_ordinary_birth_receipt_v1(row)


def validate_rowv1_ordinary_birth_receipt_v1(row: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalize a production RowV1 ordinary-birth receipt row."""

    if not isinstance(row, Mapping):
        raise P12RowV1CudaEntryGateError("RowV1 receipt row must be a mapping")
    actual = set(row)
    expected = set(ROWV1_FIELD_NAMES)
    if missing := sorted(expected - actual):
        raise P12RowV1CudaEntryGateError(f"missing RowV1 fields: {missing}")
    if extra := sorted(actual - expected):
        raise P12RowV1CudaEntryGateError(f"unsupported RowV1 fields: {extra}")
    normalized = {field: row[field] for field in ROWV1_FIELD_NAMES}
    if normalized["semantic_schema_version"] != ROWV1_SEMANTIC_SCHEMA_VERSION:
        raise P12RowV1CudaEntryGateError("unsupported semantic schema version")
    source_abi = _require_int32(
        normalized["source_receipt_abi_version"],
        field="source_receipt_abi_version",
    )
    if source_abi not in SUPPORTED_ROWV1_SOURCE_ABI_VERSIONS:
        raise P12RowV1CudaEntryGateError("unsupported source receipt ABI version")
    _require_sha256(normalized["source_receipt_sha256"], field="source_receipt_sha256")
    if normalized["semantic_mapping_contract_sha256"] != SEMANTIC_MAPPING_CONTRACT_SHA256:
        raise P12RowV1CudaEntryGateError("stale semantic mapping contract hash")
    transaction_status = _require_int32(
        normalized["native_transaction_status"],
        field="native_transaction_status",
    )
    raw_disabled_reason = normalized["raw_disabled_reason_code"]
    canonical_status = _require_text(normalized["canonical_status"], field="canonical_status")
    canonical_reason = _require_text(normalized["canonical_reason"], field="canonical_reason")
    if canonical_status == "disabled":
        if source_abi == 4:
            code = _require_int32(raw_disabled_reason, field="raw_disabled_reason_code")
            expected_reason = DISABLED_REASON_BY_CODE.get(code)
            if expected_reason is None:
                raise P12RowV1CudaEntryGateError("unknown disabled reason code")
            if canonical_reason != expected_reason:
                raise P12RowV1CudaEntryGateError("canonical reason mismatch")
        elif source_abi == 3:
            if raw_disabled_reason is None:
                expected_reason = "operator_disabled"
            else:
                code = _require_int32(raw_disabled_reason, field="raw_disabled_reason_code")
                expected_reason = DISABLED_REASON_BY_CODE.get(code)
                if expected_reason is None:
                    raise P12RowV1CudaEntryGateError("unknown disabled reason code")
            if canonical_reason != expected_reason:
                raise P12RowV1CudaEntryGateError("canonical reason mismatch")
        else:
            raise P12RowV1CudaEntryGateError(
                "source ABI cannot encode disabled reason authority"
            )
    elif raw_disabled_reason not in (None, 0):
        raise P12RowV1CudaEntryGateError(
            "non-disabled row cannot carry raw disabled reason code"
        )
    if transaction_status == TRANSACTION_MUTATION_DISABLED and canonical_status != "disabled":
        raise P12RowV1CudaEntryGateError("disabled transaction status mismatch")
    for field in (
        "transaction_id",
        "authoritative_transaction_sequence",
        "rng_index_before",
        "rng_index_after",
    ):
        _require_uint64(normalized[field], field=field)
    if normalized["transaction_sequence_scope"] != TRANSACTION_SEQUENCE_SCOPE:
        raise P12RowV1CudaEntryGateError("unsupported transaction sequence scope")
    for field in ("parent_agent_id", "parent_slot_token", "child_agent_id", "child_slot_token"):
        _require_int64(normalized[field], field=field)
    for field in ("parent_slot", "child_slot", "cell", "resident_slot"):
        _require_int32(normalized[field], field=field)
    for field in ("generation", "tick"):
        _require_int64(normalized[field], field=field)
    if normalized["rng_index_after"] < normalized["rng_index_before"]:
        raise P12RowV1CudaEntryGateError("rng index cannot move backward")
    if type(normalized["changed"]) is not bool:
        raise P12RowV1CudaEntryGateError("changed must be bool")
    if canonical_status in {"disabled", "rejected"}:
        if normalized["changed"] is not False:
            raise P12RowV1CudaEntryGateError(
                "rejected or disabled row cannot be marked changed"
            )
        for field in ("selected_identity", "value_before", "value_after"):
            if normalized[field] is not None:
                raise P12RowV1CudaEntryGateError(
                    "rejected or disabled row must clear mutation fields"
                )
    if canonical_status == "disabled" and normalized["proposal_digest"] is not None:
        raise P12RowV1CudaEntryGateError("disabled row must clear proposal digest")
    if normalized["proposal_digest"] is not None:
        _require_sha256(normalized["proposal_digest"], field="proposal_digest")
    for field in ("allocator_before", "allocator_after"):
        if not isinstance(normalized[field], Mapping):
            raise P12RowV1CudaEntryGateError(f"{field} must be a mapping")
        normalized[field] = dict(normalized[field])
    for field in ("package_binding_sha256", "schema_binding_sha256", "mechanics_binding_sha256"):
        _require_sha256(normalized[field], field=field)
    for field in (
        "rng_stream_id",
        "mutation_family",
        "operator_id",
        "capacity_decision",
        "publication_decision",
        "rollback_decision",
    ):
        _require_text(normalized[field], field=field)
    return normalized


def validate_rowv1_decoder_admission_v1(
    rows: tuple[Mapping[str, Any], ...] | list[Mapping[str, Any]],
    *,
    decoder_contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate that a RowV1 decoder can admit a homogeneous receipt batch."""

    if not isinstance(decoder_contract, Mapping):
        raise P12RowV1CudaEntryGateError("decoder contract must be a mapping")
    actual = set(decoder_contract)
    expected = set(ROWV1_DECODER_ADMISSION_FIELDS)
    if missing := sorted(expected - actual):
        raise P12RowV1CudaEntryGateError(f"missing decoder contract fields: {missing}")
    if extra := sorted(actual - expected):
        raise P12RowV1CudaEntryGateError(f"unsupported decoder contract fields: {extra}")
    if decoder_contract["decoder_contract_id"] != ROWV1_DECODER_ADMISSION_CONTRACT_ID:
        raise P12RowV1CudaEntryGateError("unsupported decoder contract")
    _require_text(decoder_contract["decoder_schema_version"], field="decoder_schema_version")
    supported_abis = tuple(
        _require_int32(value, field="supported_source_receipt_abi_versions")
        for value in _require_sequence(
            decoder_contract["supported_source_receipt_abi_versions"],
            field="supported_source_receipt_abi_versions",
        )
    )
    if not supported_abis:
        raise P12RowV1CudaEntryGateError("decoder supports no source receipt ABI")
    if any(abi not in SUPPORTED_ROWV1_SOURCE_ABI_VERSIONS for abi in supported_abis):
        raise P12RowV1CudaEntryGateError("unsupported decoder source receipt ABI")
    decoder_width = _require_uint64(
        decoder_contract["decoder_row_width"],
        field="decoder_row_width",
    )
    expected_width = _require_uint64(
        decoder_contract["expected_row_width"],
        field="expected_row_width",
    )
    if decoder_width != expected_width:
        raise P12RowV1CudaEntryGateError("stale decoder width")
    allow_mixed = _require_bool(
        decoder_contract["allow_mixed_source_abi_versions"],
        field="allow_mixed_source_abi_versions",
    )
    normalized_rows = [validate_rowv1_ordinary_birth_receipt_v1(row) for row in rows]
    if not normalized_rows:
        raise P12RowV1CudaEntryGateError("zero-length or absent receipt surface")
    row_abis = {int(row["source_receipt_abi_version"]) for row in normalized_rows}
    unsupported = sorted(row_abis - set(supported_abis))
    if unsupported:
        raise P12RowV1CudaEntryGateError(
            f"decoder does not support source receipt ABI versions: {unsupported}"
        )
    if len(row_abis) > 1 and not allow_mixed:
        raise P12RowV1CudaEntryGateError("mixed source receipt ABI versions in one batch")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "contract_id": ROWV1_DECODER_ADMISSION_CONTRACT_ID,
        "decoder_schema_version": decoder_contract["decoder_schema_version"],
        "source_receipt_abi_versions": sorted(row_abis),
        "receipt_count": len(normalized_rows),
        "decoder_row_width": decoder_width,
        "rowv1_schema_sha256": rowv1_semantic_receipt_schema_v1()["contract_sha256"],
    }
    return _with_sha(payload)


def validate_cuda_receipt_surface_admission_v1(
    admission: Mapping[str, Any],
) -> dict[str, Any]:
    """Fail closed until CUDA declares a receipt ABI that preserves RowV1."""

    if not isinstance(admission, Mapping):
        raise P12RowV1CudaEntryGateError("CUDA receipt admission must be a mapping")
    actual = set(admission)
    expected = set(CUDA_RECEIPT_SURFACE_ADMISSION_FIELDS)
    if missing := sorted(expected - actual):
        raise P12RowV1CudaEntryGateError(f"missing CUDA admission fields: {missing}")
    if extra := sorted(actual - expected):
        raise P12RowV1CudaEntryGateError(f"unsupported CUDA admission fields: {extra}")
    if admission["admission_contract_id"] != CUDA_RECEIPT_SURFACE_ADMISSION_CONTRACT_ID:
        raise P12RowV1CudaEntryGateError("unsupported CUDA receipt admission contract")
    if _require_bool(
        admission["authoritative_cuda_receipt_abi_present"],
        field="authoritative_cuda_receipt_abi_present",
    ) is not True:
        raise P12RowV1CudaEntryGateError("no authoritative CUDA receipt ABI")
    abi_version = admission["declared_cuda_receipt_abi_version"]
    if abi_version is None:
        raise P12RowV1CudaEntryGateError("CUDA receipt ABI version is undeclared")
    abi_version = _require_int32(abi_version, field="declared_cuda_receipt_abi_version")
    minimum_abi = cuda_rowv1_v3_minimum_receipt_abi_contract_v1()
    if abi_version != CUDA_ROWV1_V3_RECEIPT_ABI_VERSION:
        raise P12RowV1CudaEntryGateError("unsupported CUDA RowV1 receipt ABI version")
    if _require_bool(
        admission["device_layout_preserves_rowv1_semantics"],
        field="device_layout_preserves_rowv1_semantics",
    ) is not True:
        raise P12RowV1CudaEntryGateError("device layout cannot preserve RowV1 semantics")
    if _require_bool(
        admission["receipt_surface_present"],
        field="receipt_surface_present",
    ) is not True:
        raise P12RowV1CudaEntryGateError("zero-length or absent receipt surface")
    if _require_uint64(
        admission["receipt_surface_row_count"],
        field="receipt_surface_row_count",
    ) == 0:
        raise P12RowV1CudaEntryGateError("zero-length or absent receipt surface")
    host_size = _require_uint64(
        admission["host_receipt_size_bytes"],
        field="host_receipt_size_bytes",
    )
    device_size = _require_uint64(
        admission["device_receipt_size_bytes"],
        field="device_receipt_size_bytes",
    )
    if host_size != device_size:
        raise P12RowV1CudaEntryGateError("host/device receipt size mismatch")
    host_align = _require_uint64(
        admission["host_receipt_alignment_bytes"],
        field="host_receipt_alignment_bytes",
    )
    device_align = _require_uint64(
        admission["device_receipt_alignment_bytes"],
        field="device_receipt_alignment_bytes",
    )
    if host_align != device_align:
        raise P12RowV1CudaEntryGateError("host/device receipt alignment mismatch")
    if admission["row_dtype"] != "int64":
        raise P12RowV1CudaEntryGateError("unsupported receipt row dtype")
    if admission["counts_dtype"] != "int64":
        raise P12RowV1CudaEntryGateError("unsupported receipt counts dtype")
    if admission["overflow_dtype"] != "int32":
        raise P12RowV1CudaEntryGateError("unsupported receipt overflow dtype")
    row_width = _require_uint64(
        admission["receipt_row_width"],
        field="receipt_row_width",
    )
    expected_width = _require_uint64(
        admission["expected_receipt_row_width"],
        field="expected_receipt_row_width",
    )
    decoder_width = _require_uint64(
        admission["decoder_row_width"],
        field="decoder_row_width",
    )
    if row_width != expected_width or decoder_width != expected_width:
        raise P12RowV1CudaEntryGateError("stale decoder width")
    if expected_width != minimum_abi["minimum_receipt_row_width"]:
        raise P12RowV1CudaEntryGateError("CUDA RowV1 v3 receipt width mismatch")
    manifest_width = _require_uint64(
        admission["checkpoint_manifest_receipt_width"],
        field="checkpoint_manifest_receipt_width",
    )
    if manifest_width != row_width:
        raise P12RowV1CudaEntryGateError("checkpoint manifest receipt width mismatch")
    manifest_abi = _require_int32(
        admission["checkpoint_manifest_receipt_abi_version"],
        field="checkpoint_manifest_receipt_abi_version",
    )
    if manifest_abi != abi_version:
        raise P12RowV1CudaEntryGateError("checkpoint manifest receipt ABI mismatch")
    if _require_bool(
        admission["checkpoint_manifest_preserves_source_abi"],
        field="checkpoint_manifest_preserves_source_abi",
    ) is not True:
        raise P12RowV1CudaEntryGateError(
            "checkpoint manifest omits source receipt ABI"
        )
    if _require_bool(
        admission["checkpoint_manifest_preserves_disabled_reason"],
        field="checkpoint_manifest_preserves_disabled_reason",
    ) is not True:
        raise P12RowV1CudaEntryGateError(
            "checkpoint manifest omits disabled reason"
        )
    package_receipt_abi_sha256 = _require_sha256(
        admission["package_receipt_abi_sha256"],
        field="package_receipt_abi_sha256",
    )
    if package_receipt_abi_sha256 != minimum_abi["contract_sha256"]:
        raise P12RowV1CudaEntryGateError("package does not bind CUDA RowV1 v3 ABI")
    if admission["route_rowv1_mapping_contract_sha256"] != SEMANTIC_MAPPING_CONTRACT_SHA256:
        raise P12RowV1CudaEntryGateError("route does not bind RowV1 mapping contract")
    sanitizer_cases = set(
        _require_text(value, field="sanitizer_bounds_cases")
        for value in _require_sequence(
            admission["sanitizer_bounds_cases"],
            field="sanitizer_bounds_cases",
        )
    )
    required_sanitizers = {
        "receipt_rows_bounds",
        "receipt_counts_bounds",
        "receipt_overflow_bounds",
    }
    if not required_sanitizers <= sanitizer_cases:
        raise P12RowV1CudaEntryGateError("sanitizer coverage is absent")
    if admission["dense_id_authority"] != "sparse_identity_table_and_slot_token":
        raise P12RowV1CudaEntryGateError("dense-ID dependence remains")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "contract_id": CUDA_RECEIPT_SURFACE_ADMISSION_CONTRACT_ID,
        "cuda_receipt_abi_version": abi_version,
        "cuda_receipt_abi_contract_id": CUDA_ROWV1_V3_MINIMUM_ABI_CONTRACT_ID,
        "cuda_receipt_abi_contract_sha256": minimum_abi["contract_sha256"],
        "receipt_row_width": row_width,
        "package_receipt_abi_sha256": package_receipt_abi_sha256,
        "route_rowv1_mapping_contract_sha256": (
            admission["route_rowv1_mapping_contract_sha256"]
        ),
        "cuda_receipt_surface_admitted": True,
    }
    return _with_sha(payload)


def build_rowv1_checkpoint_report_envelope_v1(
    row: Mapping[str, Any],
    *,
    boundary: str,
    reporting_enabled: bool,
    lifecycle_ledger_sha256: str = "0" * 64,
    source_checkpoint_boundary: str = "not_checkpointed",
    cancellation_state: str = "not_cancelled",
    terminal_state: str = "not_terminal",
    restore_scope: str = "not_restore",
    disk_publication_state: str = "not_published",
) -> dict[str, Any]:
    """Wrap one validated RowV1 row in the frozen checkpoint/report envelope."""

    normalized_row = validate_rowv1_ordinary_birth_receipt_v1(row)
    schema = rowv1_semantic_receipt_schema_v1()
    projection = rowv1_ordinary_birth_projection_receipt_v1()
    payload = {
        "envelope_schema_version": CHECKPOINT_REPORT_ENVELOPE_VERSION,
        "contract_id": CHECKPOINT_REPORT_ENVELOPE_CONTRACT_ID,
        "rowv1_schema_contract_id": ROWV1_SCHEMA_CONTRACT_ID,
        "rowv1_schema_sha256": schema["contract_sha256"],
        "rowv1_projection_contract_id": ROWV1_PROJECTION_CONTRACT_ID,
        "rowv1_projection_sha256": projection["contract_sha256"],
        "semantic_mapping_contract_sha256": SEMANTIC_MAPPING_CONTRACT_SHA256,
        "boundary": _require_choice(
            boundary,
            field="boundary",
            choices=CHECKPOINT_REPORT_BOUNDARIES,
        ),
        "checkpoint_generation": normalized_row["generation"],
        "checkpoint_tick": normalized_row["tick"],
        "checkpoint_cell": normalized_row["cell"],
        "reporting_enabled": _require_bool(
            reporting_enabled,
            field="reporting_enabled",
        ),
        "cancellation_state": _require_text(
            cancellation_state,
            field="cancellation_state",
        ),
        "terminal_state": _require_text(terminal_state, field="terminal_state"),
        "restore_scope": _require_text(restore_scope, field="restore_scope"),
        "disk_publication_state": _require_text(
            disk_publication_state,
            field="disk_publication_state",
        ),
        "rowv1_receipt_sha256": rowv1_ordinary_birth_receipt_sha256_v1(
            normalized_row
        ),
        "authoritative_transaction_sequence": normalized_row[
            "authoritative_transaction_sequence"
        ],
        "transaction_sequence_scope": normalized_row["transaction_sequence_scope"],
        "lifecycle_ledger_sha256": _require_sha256(
            lifecycle_ledger_sha256,
            field="lifecycle_ledger_sha256",
        ),
        "source_checkpoint_boundary": _require_text(
            source_checkpoint_boundary,
            field="source_checkpoint_boundary",
        ),
        "source_receipt_abi_version": normalized_row["source_receipt_abi_version"],
        "source_receipt_sha256": normalized_row["source_receipt_sha256"],
        "raw_disabled_reason_code": normalized_row["raw_disabled_reason_code"],
        "canonical_status": normalized_row["canonical_status"],
        "canonical_reason": normalized_row["canonical_reason"],
        "rng_stream_id": normalized_row["rng_stream_id"],
        "rng_index_before": normalized_row["rng_index_before"],
        "rng_index_after": normalized_row["rng_index_after"],
        "allocator_before": dict(normalized_row["allocator_before"]),
        "allocator_after": dict(normalized_row["allocator_after"]),
        "package_binding_sha256": normalized_row["package_binding_sha256"],
        "schema_binding_sha256": normalized_row["schema_binding_sha256"],
        "mechanics_binding_sha256": normalized_row["mechanics_binding_sha256"],
        "rowv1_receipt": normalized_row,
    }
    return validate_rowv1_checkpoint_report_envelope_v1(payload)


def validate_rowv1_checkpoint_report_envelope_v1(
    envelope: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate and normalize the RowV1 checkpoint/report preservation envelope."""

    if not isinstance(envelope, Mapping):
        raise P12RowV1CudaEntryGateError(
            "RowV1 checkpoint/report envelope must be a mapping"
        )
    actual = set(envelope)
    expected = set(CHECKPOINT_REPORT_ENVELOPE_FIELDS)
    if missing := sorted(expected - actual):
        raise P12RowV1CudaEntryGateError(
            f"missing checkpoint/report envelope fields: {missing}"
        )
    if extra := sorted(actual - expected):
        raise P12RowV1CudaEntryGateError(
            f"unsupported checkpoint/report envelope fields: {extra}"
        )
    normalized = {field: envelope[field] for field in CHECKPOINT_REPORT_ENVELOPE_FIELDS}
    if normalized["envelope_schema_version"] != CHECKPOINT_REPORT_ENVELOPE_VERSION:
        raise P12RowV1CudaEntryGateError("unsupported checkpoint/report envelope version")
    if normalized["contract_id"] != CHECKPOINT_REPORT_ENVELOPE_CONTRACT_ID:
        raise P12RowV1CudaEntryGateError("unsupported checkpoint/report contract")
    if normalized["rowv1_schema_contract_id"] != ROWV1_SCHEMA_CONTRACT_ID:
        raise P12RowV1CudaEntryGateError("unsupported RowV1 schema contract")
    schema_sha = rowv1_semantic_receipt_schema_v1()["contract_sha256"]
    if normalized["rowv1_schema_sha256"] != schema_sha:
        raise P12RowV1CudaEntryGateError("stale RowV1 schema hash")
    if normalized["rowv1_projection_contract_id"] != ROWV1_PROJECTION_CONTRACT_ID:
        raise P12RowV1CudaEntryGateError("unsupported RowV1 projection contract")
    projection_sha = rowv1_ordinary_birth_projection_receipt_v1()["contract_sha256"]
    if normalized["rowv1_projection_sha256"] != projection_sha:
        raise P12RowV1CudaEntryGateError("stale RowV1 projection hash")
    if normalized["semantic_mapping_contract_sha256"] != SEMANTIC_MAPPING_CONTRACT_SHA256:
        raise P12RowV1CudaEntryGateError("stale semantic mapping contract hash")
    _require_choice(
        normalized["boundary"],
        field="boundary",
        choices=CHECKPOINT_REPORT_BOUNDARIES,
    )
    for field in ("checkpoint_generation", "checkpoint_tick"):
        _require_int64(normalized[field], field=field)
    _require_int32(normalized["checkpoint_cell"], field="checkpoint_cell")
    _require_bool(normalized["reporting_enabled"], field="reporting_enabled")
    for field in (
        "cancellation_state",
        "terminal_state",
        "restore_scope",
        "disk_publication_state",
    ):
        _require_text(normalized[field], field=field)
    _require_sha256(
        normalized["rowv1_receipt_sha256"],
        field="rowv1_receipt_sha256",
    )
    row = validate_rowv1_ordinary_birth_receipt_v1(normalized["rowv1_receipt"])
    expected_row_hash = rowv1_ordinary_birth_receipt_sha256_v1(row)
    if normalized["rowv1_receipt_sha256"] != expected_row_hash:
        raise P12RowV1CudaEntryGateError("RowV1 receipt hash mismatch")

    mirrored_fields = (
        "source_receipt_abi_version",
        "source_receipt_sha256",
        "raw_disabled_reason_code",
        "canonical_status",
        "canonical_reason",
        "authoritative_transaction_sequence",
        "transaction_sequence_scope",
        "rng_stream_id",
        "rng_index_before",
        "rng_index_after",
        "allocator_before",
        "allocator_after",
        "package_binding_sha256",
        "schema_binding_sha256",
        "mechanics_binding_sha256",
    )
    for field in mirrored_fields:
        if normalized[field] != row[field]:
            raise P12RowV1CudaEntryGateError(
                f"checkpoint/report envelope drifted from RowV1 field {field}"
            )
    _require_sha256(
        normalized["lifecycle_ledger_sha256"],
        field="lifecycle_ledger_sha256",
    )
    _require_text(
        normalized["source_checkpoint_boundary"],
        field="source_checkpoint_boundary",
    )
    if normalized["checkpoint_generation"] != row["generation"]:
        raise P12RowV1CudaEntryGateError(
            "checkpoint/report envelope drifted from RowV1 field generation"
        )
    if normalized["checkpoint_tick"] != row["tick"]:
        raise P12RowV1CudaEntryGateError(
            "checkpoint/report envelope drifted from RowV1 field tick"
        )
    if normalized["checkpoint_cell"] != row["cell"]:
        raise P12RowV1CudaEntryGateError(
            "checkpoint/report envelope drifted from RowV1 field cell"
        )
    normalized["rowv1_receipt"] = row
    normalized["allocator_before"] = dict(normalized["allocator_before"])
    normalized["allocator_after"] = dict(normalized["allocator_after"])
    return normalized


def _collection_transaction_identity(envelope: Mapping[str, Any]) -> tuple[int, int]:
    row = envelope["rowv1_receipt"]
    return (row["cell"], row["transaction_id"])


def _collection_sequence_identity(envelope: Mapping[str, Any]) -> tuple[int, int]:
    row = envelope["rowv1_receipt"]
    return (row["cell"], row["authoritative_transaction_sequence"])


def _collection_order_key(envelope: Mapping[str, Any]) -> tuple[int, int, int, int, str]:
    row = envelope["rowv1_receipt"]
    return (
        row["cell"],
        row["authoritative_transaction_sequence"],
        row["transaction_id"],
        row["attempt_index"],
        envelope["rowv1_receipt_sha256"],
    )


def _collection_sequence_ranges(
    envelopes: list[dict[str, Any]],
    *,
    expected_cells: list[int],
) -> list[dict[str, int | None]]:
    by_cell: dict[int, list[int]] = {cell: [] for cell in expected_cells}
    for envelope in envelopes:
        row = envelope["rowv1_receipt"]
        by_cell.setdefault(row["cell"], []).append(row["authoritative_transaction_sequence"])
    ranges = []
    for cell in expected_cells:
        sequences = sorted(by_cell.get(cell, []))
        ranges.append(
            {
                "cell": cell,
                "first_sequence": sequences[0] if sequences else None,
                "last_sequence": sequences[-1] if sequences else None,
                "receipt_count": len(sequences),
            }
        )
    return ranges


def _per_cell_collection_hashes(
    envelopes: list[dict[str, Any]],
    *,
    expected_cells: list[int],
    envelope_hashes: list[str],
) -> list[dict[str, Any]]:
    by_cell: dict[int, list[dict[str, Any]]] = {cell: [] for cell in expected_cells}
    for envelope, envelope_hash in zip(envelopes, envelope_hashes, strict=True):
        row = envelope["rowv1_receipt"]
        by_cell.setdefault(row["cell"], []).append(
            {
                "sequence": row["authoritative_transaction_sequence"],
                "transaction_id": row["transaction_id"],
                "envelope_sha256": envelope_hash,
            }
        )
    return [
        {
            "cell": cell,
            "collection_sha256": content_sha256(
                {
                    "cell": cell,
                    "transaction_sequence_scope": TRANSACTION_SEQUENCE_SCOPE,
                    "receipts": by_cell.get(cell, []),
                }
            ),
        }
        for cell in expected_cells
    ]


def _normalized_per_cell_counts(
    value: Mapping[int, int] | Mapping[str, int] | list[Mapping[str, Any]],
    *,
    field: str,
) -> list[dict[str, int]]:
    if isinstance(value, Mapping):
        rows = [
            {"cell": _require_int32(int(cell), field=field), "receipt_count": count}
            for cell, count in value.items()
        ]
    elif isinstance(value, list):
        rows = []
        for row in value:
            if not isinstance(row, Mapping):
                raise P12RowV1CudaEntryGateError(f"{field} rows must be mappings")
            rows.append(
                {
                    "cell": _require_int32(row.get("cell"), field=f"{field}.cell"),
                    "receipt_count": row.get("receipt_count"),
                }
            )
    else:
        raise P12RowV1CudaEntryGateError(f"{field} must be a mapping or list")
    normalized = [
        {
            "cell": row["cell"],
            "receipt_count": _require_uint64(
                row["receipt_count"],
                field=f"{field}.receipt_count",
            ),
        }
        for row in rows
    ]
    cells = [row["cell"] for row in normalized]
    if cells != sorted(set(cells)):
        raise P12RowV1CudaEntryGateError(f"{field} cells must be sorted and unique")
    return normalized


def _collection_join_binding_sha256(
    *,
    checkpoint_lineage_id: str,
    parent_collection_sha256: str | None,
    prefix_collection_sha256: str | None,
    checkpoint_state_sha256: str,
    rng_state_sha256: str,
    allocator_state_sha256: str,
    identity_table_sha256: str,
    owner_lineage_sha256: str,
) -> str | None:
    if parent_collection_sha256 is None and prefix_collection_sha256 is None:
        return None
    return content_sha256(
        {
            "checkpoint_lineage_id": checkpoint_lineage_id,
            "parent_collection_sha256": parent_collection_sha256,
            "prefix_collection_sha256": prefix_collection_sha256,
            "checkpoint_state_sha256": checkpoint_state_sha256,
            "rng_state_sha256": rng_state_sha256,
            "allocator_state_sha256": allocator_state_sha256,
            "identity_table_sha256": identity_table_sha256,
            "owner_lineage_sha256": owner_lineage_sha256,
        }
    )


def _collection_cancellation_seal_sha256(
    *,
    run_id: str,
    checkpoint_lineage_id: str,
    boundary: str,
    cancellation_state: str,
    receipt_envelope_sha256s: list[str],
    checkpoint_state_sha256: str,
    rng_state_sha256: str,
    allocator_state_sha256: str,
    identity_table_sha256: str,
    owner_lineage_sha256: str,
) -> str:
    return content_sha256(
        {
            "run_id": run_id,
            "checkpoint_lineage_id": checkpoint_lineage_id,
            "boundary": boundary,
            "cancellation_state": cancellation_state,
            "receipt_envelope_sha256s": receipt_envelope_sha256s,
            "checkpoint_state_sha256": checkpoint_state_sha256,
            "rng_state_sha256": rng_state_sha256,
            "allocator_state_sha256": allocator_state_sha256,
            "identity_table_sha256": identity_table_sha256,
            "owner_lineage_sha256": owner_lineage_sha256,
        }
    )


def _actual_per_cell_counts(
    envelopes: list[dict[str, Any]],
    *,
    expected_cells: list[int],
) -> list[dict[str, int]]:
    counts: dict[int, int] = {}
    for envelope in envelopes:
        cell = envelope["checkpoint_cell"]
        counts[cell] = counts.get(cell, 0) + 1
    return [
        {"cell": cell, "receipt_count": counts.get(cell, 0)}
        for cell in expected_cells
    ]


def build_rowv1_checkpoint_report_collection_v1(
    receipt_envelopes: tuple[Mapping[str, Any], ...] | list[Mapping[str, Any]],
    *,
    run_id: str,
    source_commit_sha: str,
    route_binding_sha256: str,
    package_binding_sha256: str,
    boundary: str,
    segment_kind: str,
    checkpoint_lineage_id: str,
    expected_receipt_count: int,
    expected_cells: tuple[int, ...] | list[int],
    lifecycle_ledger_sha256: str,
    per_cell_expected_receipt_counts: Mapping[int, int]
    | Mapping[str, int]
    | list[Mapping[str, Any]],
    checkpoint_state_sha256: str,
    rng_state_sha256: str,
    allocator_state_sha256: str,
    identity_table_sha256: str,
    owner_lineage_sha256: str,
    parent_collection_sha256: str | None = None,
    prefix_collection_sha256: str | None = None,
    last_prefix_sequence: int | None = None,
    collection_capacity_limit: int | None = None,
    overflow_count: int = 0,
    dropped_receipt_count: int = 0,
    cancellation_state: str = "not_cancelled",
    sealed: bool = False,
) -> dict[str, Any]:
    """Build the deterministic collection envelope for zero or more RowV1 rows."""

    envelopes = [
        validate_rowv1_checkpoint_report_envelope_v1(envelope)
        for envelope in receipt_envelopes
    ]
    envelopes.sort(key=_collection_order_key)
    cells = sorted({_require_int32(cell, field="expected_cells") for cell in expected_cells})
    per_cell_expected = _normalized_per_cell_counts(
        per_cell_expected_receipt_counts,
        field="per_cell_expected_receipt_counts",
    )
    per_cell_actual = _actual_per_cell_counts(envelopes, expected_cells=cells)
    row_generations = [envelope["checkpoint_generation"] for envelope in envelopes]
    row_ticks = [envelope["checkpoint_tick"] for envelope in envelopes]
    row_sequences = [
        envelope["authoritative_transaction_sequence"] for envelope in envelopes
    ]
    receipt_envelope_hashes = [
        rowv1_checkpoint_report_envelope_sha256_v1(envelope)
        for envelope in envelopes
    ]
    per_cell_ranges = _collection_sequence_ranges(envelopes, expected_cells=cells)
    per_cell_hashes = _per_cell_collection_hashes(
        envelopes,
        expected_cells=cells,
        envelope_hashes=receipt_envelope_hashes,
    )
    first_suffix_sequence = min(row_sequences) if row_sequences else None
    normalized_last_prefix_sequence = (
        _require_uint64(last_prefix_sequence, field="last_prefix_sequence")
        if last_prefix_sequence is not None
        else None
    )
    checkpoint_lineage_id = _require_text(
        checkpoint_lineage_id,
        field="checkpoint_lineage_id",
    )
    parent_collection_sha256 = (
        _require_sha256(parent_collection_sha256, field="parent_collection_sha256")
        if parent_collection_sha256 is not None
        else None
    )
    prefix_collection_sha256 = (
        _require_sha256(prefix_collection_sha256, field="prefix_collection_sha256")
        if prefix_collection_sha256 is not None
        else None
    )
    checkpoint_state_sha256 = _require_sha256(
        checkpoint_state_sha256,
        field="checkpoint_state_sha256",
    )
    rng_state_sha256 = _require_sha256(rng_state_sha256, field="rng_state_sha256")
    allocator_state_sha256 = _require_sha256(
        allocator_state_sha256,
        field="allocator_state_sha256",
    )
    identity_table_sha256 = _require_sha256(
        identity_table_sha256,
        field="identity_table_sha256",
    )
    owner_lineage_sha256 = _require_sha256(
        owner_lineage_sha256,
        field="owner_lineage_sha256",
    )
    cancellation_state = _require_text(cancellation_state, field="cancellation_state")
    capacity_limit = (
        _require_uint64(collection_capacity_limit, field="collection_capacity_limit")
        if collection_capacity_limit is not None
        else max(_require_uint64(expected_receipt_count, field="expected_receipt_count"), 1)
    )
    join_binding = _collection_join_binding_sha256(
        checkpoint_lineage_id=checkpoint_lineage_id,
        parent_collection_sha256=parent_collection_sha256,
        prefix_collection_sha256=prefix_collection_sha256,
        checkpoint_state_sha256=checkpoint_state_sha256,
        rng_state_sha256=rng_state_sha256,
        allocator_state_sha256=allocator_state_sha256,
        identity_table_sha256=identity_table_sha256,
        owner_lineage_sha256=owner_lineage_sha256,
    )
    cancellation_seal = (
        _collection_cancellation_seal_sha256(
            run_id=_require_text(run_id, field="run_id"),
            checkpoint_lineage_id=checkpoint_lineage_id,
            boundary=boundary,
            cancellation_state=cancellation_state,
            receipt_envelope_sha256s=receipt_envelope_hashes,
            checkpoint_state_sha256=checkpoint_state_sha256,
            rng_state_sha256=rng_state_sha256,
            allocator_state_sha256=allocator_state_sha256,
            identity_table_sha256=identity_table_sha256,
            owner_lineage_sha256=owner_lineage_sha256,
        )
        if cancellation_state != "not_cancelled" or boundary == "cancellation"
        else None
    )
    payload = {
        "collection_schema_version": CHECKPOINT_REPORT_COLLECTION_VERSION,
        "contract_id": CHECKPOINT_REPORT_COLLECTION_CONTRACT_ID,
        "rowv1_checkpoint_report_envelope_contract_id": (
            CHECKPOINT_REPORT_ENVELOPE_CONTRACT_ID
        ),
        "rowv1_checkpoint_report_envelope_sha256": (
            rowv1_checkpoint_report_envelope_receipt_v1()["contract_sha256"]
        ),
        "run_id": _require_text(run_id, field="run_id"),
        "source_commit_sha": _require_git_commit(
            source_commit_sha,
            field="source_commit_sha",
        ),
        "route_binding_sha256": _require_sha256(
            route_binding_sha256,
            field="route_binding_sha256",
        ),
        "package_binding_sha256": _require_sha256(
            package_binding_sha256,
            field="package_binding_sha256",
        ),
        "boundary": _require_choice(
            boundary,
            field="boundary",
            choices=CHECKPOINT_REPORT_BOUNDARIES,
        ),
        "segment_kind": _require_choice(
            segment_kind,
            field="segment_kind",
            choices=CHECKPOINT_REPORT_COLLECTION_SEGMENTS,
        ),
        "checkpoint_lineage_id": _require_text(
            checkpoint_lineage_id,
            field="checkpoint_lineage_id",
        ),
        "parent_collection_sha256": parent_collection_sha256,
        "prefix_collection_sha256": prefix_collection_sha256,
        "join_binding_sha256": join_binding,
        "receipt_count": len(envelopes),
        "expected_receipt_count": _require_uint64(
            expected_receipt_count,
            field="expected_receipt_count",
        ),
        "attempted_transaction_count": _require_uint64(
            expected_receipt_count,
            field="attempted_transaction_count",
        ),
        "expected_receipt_count_source": "authoritative_lifecycle_ledger",
        "lifecycle_ledger_sha256": _require_sha256(
            lifecycle_ledger_sha256,
            field="lifecycle_ledger_sha256",
        ),
        "empty_collection_policy": (
            "legitimate_empty" if len(envelopes) == 0 else "not_empty"
        ),
        "expected_cells": cells,
        "cell_count": len({envelope["checkpoint_cell"] for envelope in envelopes}),
        "transaction_sequence_scope": TRANSACTION_SEQUENCE_SCOPE,
        "sequence_first": min(row_sequences) if row_sequences else None,
        "sequence_last": max(row_sequences) if row_sequences else None,
        "last_prefix_sequence": normalized_last_prefix_sequence,
        "first_suffix_sequence": first_suffix_sequence,
        "ordered_transaction_sequences": row_sequences,
        "per_cell_sequence_ranges": per_cell_ranges,
        "per_cell_collection_sha256s": per_cell_hashes,
        "global_collection_sha256": content_sha256(
            {
                "run_id": run_id,
                "checkpoint_lineage_id": checkpoint_lineage_id,
                "transaction_sequence_scope": TRANSACTION_SEQUENCE_SCOPE,
                "receipt_envelope_sha256s": receipt_envelope_hashes,
                "per_cell_collection_sha256s": per_cell_hashes,
            }
        ),
        "collector_success_state": (
            "success_zero_attempts" if len(envelopes) == 0 else "success_complete"
        ),
        "per_cell_expected_receipt_counts": per_cell_expected,
        "per_cell_receipt_counts": per_cell_actual,
        "generation_min": min(row_generations) if row_generations else None,
        "generation_max": max(row_generations) if row_generations else None,
        "tick_min": min(row_ticks) if row_ticks else None,
        "tick_max": max(row_ticks) if row_ticks else None,
        "collection_capacity_limit": capacity_limit,
        "collection_capacity_policy": "fail_closed_on_capacity_exceeded",
        "overflow_count": _require_uint64(overflow_count, field="overflow_count"),
        "dropped_receipt_count": _require_uint64(
            dropped_receipt_count,
            field="dropped_receipt_count",
        ),
        "cancellation_state": cancellation_state,
        "sealed": _require_bool(sealed, field="sealed"),
        "cancellation_seal_sha256": cancellation_seal,
        "checkpoint_state_sha256": checkpoint_state_sha256,
        "rng_state_sha256": rng_state_sha256,
        "allocator_state_sha256": allocator_state_sha256,
        "identity_table_sha256": identity_table_sha256,
        "owner_lineage_sha256": owner_lineage_sha256,
        "deterministic_order_key": (
            "checkpoint_cell,authoritative_transaction_sequence,"
            "transaction_id,attempt_index,rowv1_receipt_sha256"
        ),
        "transaction_identity_key": "checkpoint_cell,transaction_id",
        "receipt_envelope_sha256s": receipt_envelope_hashes,
        "receipt_envelopes": envelopes,
    }
    return validate_rowv1_checkpoint_report_collection_v1(payload)


def validate_rowv1_checkpoint_report_collection_v1(
    collection: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate a deterministic collection of RowV1 checkpoint/report envelopes."""

    if not isinstance(collection, Mapping):
        raise P12RowV1CudaEntryGateError(
            "RowV1 checkpoint/report collection must be a mapping"
        )
    actual = set(collection)
    expected = set(CHECKPOINT_REPORT_COLLECTION_FIELDS)
    if missing := sorted(expected - actual):
        raise P12RowV1CudaEntryGateError(
            f"missing checkpoint/report collection fields: {missing}"
        )
    if extra := sorted(actual - expected):
        raise P12RowV1CudaEntryGateError(
            f"unsupported checkpoint/report collection fields: {extra}"
        )
    normalized = {
        field: collection[field] for field in CHECKPOINT_REPORT_COLLECTION_FIELDS
    }
    if normalized["collection_schema_version"] != CHECKPOINT_REPORT_COLLECTION_VERSION:
        raise P12RowV1CudaEntryGateError(
            "unsupported checkpoint/report collection version"
        )
    if normalized["contract_id"] != CHECKPOINT_REPORT_COLLECTION_CONTRACT_ID:
        raise P12RowV1CudaEntryGateError("unsupported checkpoint/report collection")
    if (
        normalized["rowv1_checkpoint_report_envelope_contract_id"]
        != CHECKPOINT_REPORT_ENVELOPE_CONTRACT_ID
    ):
        raise P12RowV1CudaEntryGateError("unsupported RowV1 envelope contract")
    envelope_contract_sha = rowv1_checkpoint_report_envelope_receipt_v1()[
        "contract_sha256"
    ]
    if normalized["rowv1_checkpoint_report_envelope_sha256"] != envelope_contract_sha:
        raise P12RowV1CudaEntryGateError("stale RowV1 envelope contract hash")

    _require_text(normalized["run_id"], field="run_id")
    _require_git_commit(normalized["source_commit_sha"], field="source_commit_sha")
    _require_sha256(normalized["route_binding_sha256"], field="route_binding_sha256")
    _require_sha256(normalized["package_binding_sha256"], field="package_binding_sha256")
    _require_choice(
        normalized["boundary"],
        field="boundary",
        choices=CHECKPOINT_REPORT_BOUNDARIES,
    )
    segment_kind = _require_choice(
        normalized["segment_kind"],
        field="segment_kind",
        choices=CHECKPOINT_REPORT_COLLECTION_SEGMENTS,
    )
    _require_text(normalized["checkpoint_lineage_id"], field="checkpoint_lineage_id")
    for field in ("parent_collection_sha256", "prefix_collection_sha256"):
        if normalized[field] is not None:
            _require_sha256(normalized[field], field=field)
    for field in (
        "checkpoint_state_sha256",
        "rng_state_sha256",
        "allocator_state_sha256",
        "identity_table_sha256",
        "owner_lineage_sha256",
    ):
        _require_sha256(normalized[field], field=field)
    expected_join_binding = _collection_join_binding_sha256(
        checkpoint_lineage_id=normalized["checkpoint_lineage_id"],
        parent_collection_sha256=normalized["parent_collection_sha256"],
        prefix_collection_sha256=normalized["prefix_collection_sha256"],
        checkpoint_state_sha256=normalized["checkpoint_state_sha256"],
        rng_state_sha256=normalized["rng_state_sha256"],
        allocator_state_sha256=normalized["allocator_state_sha256"],
        identity_table_sha256=normalized["identity_table_sha256"],
        owner_lineage_sha256=normalized["owner_lineage_sha256"],
    )
    if normalized["join_binding_sha256"] != expected_join_binding:
        raise P12RowV1CudaEntryGateError("checkpoint lineage join binding mismatch")
    if segment_kind == "resumed_suffix" and normalized["parent_collection_sha256"] is None:
        raise P12RowV1CudaEntryGateError("resumed suffix requires parent collection")
    if (
        segment_kind == "checkpoint_plus_resume"
        and normalized["prefix_collection_sha256"] is None
    ):
        raise P12RowV1CudaEntryGateError(
            "checkpoint plus resume requires prefix collection"
        )

    receipt_count = _require_uint64(normalized["receipt_count"], field="receipt_count")
    expected_count = _require_uint64(
        normalized["expected_receipt_count"],
        field="expected_receipt_count",
    )
    if receipt_count != expected_count:
        raise P12RowV1CudaEntryGateError("receipt collection count mismatch")
    attempted_count = _require_uint64(
        normalized["attempted_transaction_count"],
        field="attempted_transaction_count",
    )
    if attempted_count != expected_count:
        raise P12RowV1CudaEntryGateError("attempted transaction count mismatch")
    _require_choice(
        normalized["expected_receipt_count_source"],
        field="expected_receipt_count_source",
        choices=EXPECTED_RECEIPT_COUNT_SOURCES,
    )
    _require_sha256(normalized["lifecycle_ledger_sha256"], field="lifecycle_ledger_sha256")
    empty_policy = _require_choice(
        normalized["empty_collection_policy"],
        field="empty_collection_policy",
        choices=EMPTY_COLLECTION_POLICIES,
    )
    if receipt_count == 0:
        if expected_count != 0:
            raise P12RowV1CudaEntryGateError("empty collection contradicts ledger count")
        if empty_policy != "legitimate_empty":
            raise P12RowV1CudaEntryGateError("empty collection must be explicit")
        if normalized["collector_success_state"] != "success_zero_attempts":
            raise P12RowV1CudaEntryGateError("empty collection requires collector success")
    elif empty_policy != "not_empty":
        raise P12RowV1CudaEntryGateError("nonempty collection cannot use empty policy")
    elif normalized["collector_success_state"] != "success_complete":
        raise P12RowV1CudaEntryGateError("nonempty collection requires collector success")
    expected_cells_value = normalized["expected_cells"]
    if not isinstance(expected_cells_value, list):
        raise P12RowV1CudaEntryGateError("expected_cells must be a list")
    expected_cells = [
        _require_int32(cell, field="expected_cells") for cell in expected_cells_value
    ]
    if expected_cells != sorted(set(expected_cells)):
        raise P12RowV1CudaEntryGateError("expected_cells must be sorted and unique")
    _require_uint64(normalized["cell_count"], field="cell_count")
    per_cell_expected = _normalized_per_cell_counts(
        normalized["per_cell_expected_receipt_counts"],
        field="per_cell_expected_receipt_counts",
    )
    per_cell_actual_claim = _normalized_per_cell_counts(
        normalized["per_cell_receipt_counts"],
        field="per_cell_receipt_counts",
    )
    if [row["cell"] for row in per_cell_expected] != expected_cells:
        raise P12RowV1CudaEntryGateError("per-cell ledger counts omit expected cells")
    if sum(row["receipt_count"] for row in per_cell_expected) != expected_count:
        raise P12RowV1CudaEntryGateError("per-cell ledger count mismatch")

    overflow_count = _require_uint64(normalized["overflow_count"], field="overflow_count")
    dropped_count = _require_uint64(
        normalized["dropped_receipt_count"],
        field="dropped_receipt_count",
    )
    if overflow_count != 0 or dropped_count != 0:
        raise P12RowV1CudaEntryGateError("overflow or dropped receipts fail closed")
    capacity_limit = _require_uint64(
        normalized["collection_capacity_limit"],
        field="collection_capacity_limit",
    )
    if receipt_count > capacity_limit:
        raise P12RowV1CudaEntryGateError("receipt collection exceeds capacity limit")
    _require_choice(
        normalized["collection_capacity_policy"],
        field="collection_capacity_policy",
        choices=COLLECTION_CAPACITY_POLICIES,
    )
    cancellation_state = _require_text(
        normalized["cancellation_state"],
        field="cancellation_state",
    )
    sealed = _require_bool(normalized["sealed"], field="sealed")
    if cancellation_state != "not_cancelled" and not sealed:
        raise P12RowV1CudaEntryGateError("cancelled collection must be sealed")
    if normalized["boundary"] == "cancellation" and not sealed:
        raise P12RowV1CudaEntryGateError("cancellation boundary must be sealed")

    if normalized["deterministic_order_key"] != (
        "checkpoint_cell,authoritative_transaction_sequence,"
        "transaction_id,attempt_index,rowv1_receipt_sha256"
    ):
        raise P12RowV1CudaEntryGateError("unsupported deterministic order key")
    if normalized["transaction_identity_key"] != "checkpoint_cell,transaction_id":
        raise P12RowV1CudaEntryGateError("unsupported transaction identity key")
    if not isinstance(normalized["receipt_envelopes"], list):
        raise P12RowV1CudaEntryGateError("receipt_envelopes must be a list")
    envelopes = [
        validate_rowv1_checkpoint_report_envelope_v1(envelope)
        for envelope in normalized["receipt_envelopes"]
    ]
    if len(envelopes) != receipt_count:
        raise P12RowV1CudaEntryGateError("receipt envelope count mismatch")
    order_keys = [_collection_order_key(envelope) for envelope in envelopes]
    if order_keys != sorted(order_keys):
        raise P12RowV1CudaEntryGateError("receipt collection order is not deterministic")
    transaction_identities = [
        _collection_transaction_identity(envelope) for envelope in envelopes
    ]
    if len(set(transaction_identities)) != len(transaction_identities):
        raise P12RowV1CudaEntryGateError("duplicate transaction identity in collection")
    sequence_identities = [_collection_sequence_identity(envelope) for envelope in envelopes]
    if len(set(sequence_identities)) != len(sequence_identities):
        raise P12RowV1CudaEntryGateError("duplicate transaction sequence in collection")
    row_hashes = [envelope["rowv1_receipt_sha256"] for envelope in envelopes]
    if len(set(row_hashes)) != len(row_hashes):
        raise P12RowV1CudaEntryGateError("duplicate RowV1 receipt in collection")
    envelope_hashes = [
        rowv1_checkpoint_report_envelope_sha256_v1(envelope)
        for envelope in envelopes
    ]
    if normalized["receipt_envelope_sha256s"] != envelope_hashes:
        raise P12RowV1CudaEntryGateError("receipt envelope hash list mismatch")
    actual_cells = sorted({envelope["checkpoint_cell"] for envelope in envelopes})
    if any(cell not in expected_cells for cell in actual_cells):
        raise P12RowV1CudaEntryGateError("receipt cell outside expected run cells")
    if normalized["cell_count"] != len(actual_cells):
        raise P12RowV1CudaEntryGateError("cell count mismatch")
    actual_per_cell = _actual_per_cell_counts(envelopes, expected_cells=expected_cells)
    if per_cell_actual_claim != actual_per_cell:
        raise P12RowV1CudaEntryGateError("per-cell receipt count mismatch")
    if per_cell_expected != per_cell_actual_claim:
        raise P12RowV1CudaEntryGateError("per-cell completeness mismatch")
    if receipt_count == 0 and per_cell_expected != [
        {"cell": cell, "receipt_count": 0} for cell in expected_cells
    ]:
        raise P12RowV1CudaEntryGateError("empty collection contradicts per-cell ledger")
    if normalized["transaction_sequence_scope"] != TRANSACTION_SEQUENCE_SCOPE:
        raise P12RowV1CudaEntryGateError("unsupported transaction sequence scope")
    for nullable_field in (
        "sequence_first",
        "sequence_last",
        "last_prefix_sequence",
        "first_suffix_sequence",
    ):
        if normalized[nullable_field] is not None:
            _require_uint64(normalized[nullable_field], field=nullable_field)
    expected_cancellation_seal = (
        _collection_cancellation_seal_sha256(
            run_id=normalized["run_id"],
            checkpoint_lineage_id=normalized["checkpoint_lineage_id"],
            boundary=normalized["boundary"],
            cancellation_state=cancellation_state,
            receipt_envelope_sha256s=envelope_hashes,
            checkpoint_state_sha256=normalized["checkpoint_state_sha256"],
            rng_state_sha256=normalized["rng_state_sha256"],
            allocator_state_sha256=normalized["allocator_state_sha256"],
            identity_table_sha256=normalized["identity_table_sha256"],
            owner_lineage_sha256=normalized["owner_lineage_sha256"],
        )
        if cancellation_state != "not_cancelled" or normalized["boundary"] == "cancellation"
        else None
    )
    if normalized["cancellation_seal_sha256"] != expected_cancellation_seal:
        raise P12RowV1CudaEntryGateError("cancellation seal mismatch")

    generations = [envelope["checkpoint_generation"] for envelope in envelopes]
    ticks = [envelope["checkpoint_tick"] for envelope in envelopes]
    expected_ranges = {
        "generation_min": min(generations) if generations else None,
        "generation_max": max(generations) if generations else None,
        "tick_min": min(ticks) if ticks else None,
        "tick_max": max(ticks) if ticks else None,
    }
    for field, expected_value in expected_ranges.items():
        if expected_value is None:
            if normalized[field] is not None:
                raise P12RowV1CudaEntryGateError(f"{field} must be null")
        elif _require_int64(normalized[field], field=field) != expected_value:
            raise P12RowV1CudaEntryGateError(f"{field} mismatch")
    sequences = [envelope["authoritative_transaction_sequence"] for envelope in envelopes]
    if normalized["ordered_transaction_sequences"] != sequences:
        raise P12RowV1CudaEntryGateError("ordered transaction sequence list mismatch")
    expected_sequence_first = min(sequences) if sequences else None
    expected_sequence_last = max(sequences) if sequences else None
    if normalized["sequence_first"] != expected_sequence_first:
        raise P12RowV1CudaEntryGateError("sequence first mismatch")
    if normalized["sequence_last"] != expected_sequence_last:
        raise P12RowV1CudaEntryGateError("sequence last mismatch")
    expected_sequence_ranges = _collection_sequence_ranges(
        envelopes,
        expected_cells=expected_cells,
    )
    if normalized["per_cell_sequence_ranges"] != expected_sequence_ranges:
        raise P12RowV1CudaEntryGateError("per-cell sequence range mismatch")
    expected_cell_hashes = _per_cell_collection_hashes(
        envelopes,
        expected_cells=expected_cells,
        envelope_hashes=envelope_hashes,
    )
    if normalized["per_cell_collection_sha256s"] != expected_cell_hashes:
        raise P12RowV1CudaEntryGateError("per-cell collection hash mismatch")
    expected_global_hash = content_sha256(
        {
            "run_id": normalized["run_id"],
            "checkpoint_lineage_id": normalized["checkpoint_lineage_id"],
            "transaction_sequence_scope": TRANSACTION_SEQUENCE_SCOPE,
            "receipt_envelope_sha256s": envelope_hashes,
            "per_cell_collection_sha256s": expected_cell_hashes,
        }
    )
    if normalized["global_collection_sha256"] != expected_global_hash:
        raise P12RowV1CudaEntryGateError("global collection hash mismatch")
    if normalized["last_prefix_sequence"] is not None:
        if normalized["first_suffix_sequence"] is None:
            if receipt_count != 0:
                raise P12RowV1CudaEntryGateError("missing first suffix sequence")
        elif normalized["first_suffix_sequence"] != normalized["last_prefix_sequence"] + 1:
            raise P12RowV1CudaEntryGateError(
                "prefix/suffix sequence adjacency mismatch"
            )
    if segment_kind in {"resumed_suffix", "checkpoint_plus_resume"}:
        if normalized["last_prefix_sequence"] is None:
            raise P12RowV1CudaEntryGateError(
                "resumed history requires last prefix sequence"
            )
        if normalized["first_suffix_sequence"] is not None and (
            normalized["first_suffix_sequence"] <= normalized["last_prefix_sequence"]
        ):
            raise P12RowV1CudaEntryGateError("sequence regression after restore")

    normalized["expected_cells"] = expected_cells
    normalized["per_cell_expected_receipt_counts"] = per_cell_expected
    normalized["per_cell_receipt_counts"] = actual_per_cell
    normalized["receipt_envelopes"] = envelopes
    normalized["receipt_envelope_sha256s"] = envelope_hashes
    return normalized


def rowv1_cuda_surface_inventory_v1(source_root: str | Path = ".") -> dict[str, Any]:
    root = Path(source_root)
    surfaces = (
        {
            "surface_id": "native_core_receipt_abi_v4",
            "role": "authoritative_native_core_receipt_abi",
            "host_or_device": "host",
            "path": "engine/perf/native/recurrent/modular_recurrent_core_c_api_v1.h",
            "symbol": "eiel_modular_recurrent_core_ordinary_birth_receipt_v4",
            "version": "4",
            "dtype": "packed_c_struct",
            "width": "sizeof/alignof exported by C ABI",
            "disabled_reason_support": "raw_int32_code_0_to_6",
            "source_abi_version_support": "exported_layout_version_v4",
            "stable_id_semantics": "int64 agent ids plus slot/token owner fields",
            "dense_id_dependence": "none",
            "authority": "authoritative_for_native_core_fixtures_only",
            "authority_kind": "authoritative_native_abi",
            "persisted": False,
            "persistence_mode": "transient_call_output",
            "producer": "native recurrent C API ordinary-birth call",
            "consumer": "Python ctypes bridge and RowV1 host projection",
            "package_route_binding": "not a production CUDA/package route binding",
            "required_action": "project_into_RowV1_before_production_admission",
            "marker": "eiel_modular_recurrent_core_ordinary_birth_receipt_v4",
        },
        {
            "surface_id": "canonical_recurrent_receipt",
            "role": "canonical_semantic_receipt",
            "host_or_device": "host",
            "path": "engine/perf/recurrent_native/modular_recurrent_mutation_receipt_v1.py",
            "symbol": "ModularRecurrentMutationReceiptV1",
            "version": "modular_two_draws_v1",
            "dtype": "canonical_json",
            "width": "variable_canonical_json",
            "disabled_reason_support": "canonical_reason_string",
            "source_abi_version_support": "external_envelope_required",
            "stable_id_semantics": "int64 agent ids with slot/token transaction fields",
            "dense_id_dependence": "none",
            "authority": "canonical_replay_answer_key",
            "authority_kind": "canonical_semantic_receipt",
            "persisted": True,
            "persistence_mode": "canonical_json_receipt",
            "producer": "native-to-canonical bridge",
            "consumer": "fixture bank, replay, RowV1 projection",
            "package_route_binding": "external RowV1 envelope required",
            "required_action": "bind source receipt ABI and source receipt hash in RowV1",
            "marker": "ModularRecurrentMutationReceiptV1",
        },
        {
            "surface_id": "native_to_canonical_bridge",
            "role": "host_projection_bridge",
            "host_or_device": "host",
            "path": "engine/perf/recurrent_native/native_core_ordinary_birth_receipt_bridge_v1.py",
            "symbol": "DISABLED_REASON_BY_CODE",
            "version": "v4-aware",
            "dtype": "python_mapping",
            "width": "not_fixed_width",
            "disabled_reason_support": "raw_code_to_canonical_reason",
            "source_abi_version_support": "caller_must_supply_envelope",
            "stable_id_semantics": "maps native owner fields to canonical transaction",
            "dense_id_dependence": "none",
            "authority": "projection_only",
            "authority_kind": "host_projection_bridge",
            "persisted": False,
            "persistence_mode": "transient_mapping_logic",
            "producer": "Python bridge reader",
            "consumer": "canonical semantic receipt and RowV1 projection",
            "package_route_binding": "must be bound by semantic mapping hash",
            "required_action": "fail closed for unknown reason and stale mapping hash",
            "marker": "DISABLED_REASON_BY_CODE",
        },
        {
            "surface_id": "cuda_modular_ordinary_birth_mutation_receipt_tensor",
            "role": "device_staging_and_host_projection",
            "host_or_device": "device_and_host",
            "path": "engine/perf/cuda_architecture_b_direct_resident_modular_v1.py",
            "symbol": "ordinary_birth_mutation_receipt_rows",
            "version": "raw_layout_version_2",
            "dtype": "torch.int64",
            "width": "int64[cells,3,21]",
            "disabled_reason_support": "absent",
            "source_abi_version_support": "absent",
            "stable_id_semantics": "int64 agent ids plus slot/token row fields",
            "dense_id_dependence": "none_in_receipt_width",
            "authority": "bounded_cuda_mutation_receipt_projection_not_production_RowV1",
            "authority_kind": "diagnostic_device_staging",
            "persisted": False,
            "persistence_mode": "transient_device_and_host_tensor",
            "producer": "CUDA modular lifecycle kernel raw layout v2",
            "consumer": "host diagnostic projection only",
            "package_route_binding": "absent for RowV1/CUDA ordinary-birth admission",
            "required_action": "CUDA admission must fail closed for v4 RowV1 route",
            "marker": "ordinary_birth_mutation_receipt_rows",
        },
        {
            "surface_id": "cuda_modular_ordinary_birth_rowv1_v3_receipt_tensor",
            "role": "device_entry_surface_not_authoritative",
            "host_or_device": "device_and_host",
            "path": "engine/perf/cuda_architecture_b_direct_resident_modular_v1.py",
            "symbol": "ordinary_birth_rowv1_receipt_rows",
            "version": "rowv1_receipt_abi_version_3",
            "dtype": "torch.int64 rows plus int64 counts plus int32 overflow",
            "width": "int64[cells,3,64]",
            "disabled_reason_support": "reserved_rowv1_columns_not_kernel_qualified",
            "source_abi_version_support": "reserved_rowv1_columns_not_kernel_qualified",
            "stable_id_semantics": "int64 agent ids plus slot/token row fields",
            "dense_id_dependence": "none_in_receipt_shape",
            "authority": "kernel_mirror_present_not_authoritative",
            "authority_kind": "cuda_rowv1_v3_entry_surface",
            "persisted": False,
            "persistence_mode": "transient_device_and_host_tensor",
            "producer": "Python preparation, C++ binding validation, and CUDA kernel mirror",
            "consumer": "future CUDA RowV1 admission gate",
            "package_route_binding": "absent until artifact-bound ABI qualification",
            "required_action": "qualify authoritative fields before CUDA admission",
            "marker": "ordinary_birth_rowv1_receipt_rows",
        },
        {
            "surface_id": "cuda_modular_ordinary_birth_rowv1_v3_kernel_mirror",
            "role": "device_kernel_mirror_not_authoritative",
            "host_or_device": "device",
            "path": "engine/perf/cuda_architecture_b_flattened_modular_controller_v1.cuh",
            "symbol": "mutate_descendant_parameter_v1::ordinary_birth_rowv1_receipt_rows",
            "version": "rowv1_receipt_abi_version_3_mirrors_raw_layout_version_2",
            "dtype": "std::int64_t row words plus int64 counts plus int32 overflow",
            "width": "kOrdinaryBirthRowV1ReceiptWidthV3 == 64",
            "disabled_reason_support": "host_projection_rejects_nonzero_unqualified_reason",
            "source_abi_version_support": "rowv1_v3_source_hash_host_projection_bound",
            "stable_id_semantics": "int64 agent ids plus slot/token row fields",
            "dense_id_dependence": "none_in_receipt_shape",
            "authority": "kernel_mirror_projectable_not_authoritative",
            "authority_kind": "cuda_rowv1_v3_kernel_mirror",
            "persisted": False,
            "persistence_mode": "transient_device_tensor",
            "producer": "CUDA modular lifecycle mutation kernel",
            "consumer": "host RowV1 v3 mirror projection and future ABI admission",
            "package_route_binding": "absent until immutable CUDA artifact rebuild",
            "required_action": "bind disabled reason, package, checkpoint, and route before admission",
            "marker": "rowv1[0] = 3",
        },
        {
            "surface_id": "cuda_modular_receipt_binding",
            "role": "host_device_layout_check",
            "host_or_device": "host_binding",
            "path": "engine/perf/cuda_architecture_b_flattened_wdl_controller_v2_binding.cpp",
            "symbol": "kOrdinaryBirthRowV1ReceiptWidthV3",
            "version": "raw_layout_version_2_plus_rowv1_entry_surface_v3",
            "dtype": "torch::kInt64 rows plus int64 counts plus int32 overflow",
            "width": "cells*3*21 raw v2 and cells*3*64 RowV1 v3 entry storage",
            "disabled_reason_support": "raw v2 absent; RowV1 v3 reserved but not written",
            "source_abi_version_support": "capability_reports_rowv1_v3_surface_only",
            "stable_id_semantics": "cell-local slots and int64 agent ids",
            "dense_id_dependence": "none_in_receipt_shape",
            "authority": "layout_sanitizer_for_cuda_receipt_tensors",
            "authority_kind": "host_device_layout_sanitizer",
            "persisted": False,
            "persistence_mode": "transient_binding_capability_report",
            "producer": "CUDA binding capability report",
            "consumer": "host route admission checks",
            "package_route_binding": "RowV1 v3 surface not yet route-bound",
            "required_action": "reject until kernel writes and package binds RowV1 ABI",
            "marker": "ordinary_birth_rowv1_receipt_width",
        },
        {
            "surface_id": "cuda_modular_receipt_device_context",
            "role": "device_struct_pointer_surface",
            "host_or_device": "device",
            "path": "engine/perf/cuda_architecture_b_flattened_modular_controller_v1.hpp",
            "symbol": "FlatModularKernelContextV2::ordinary_birth_rowv1_receipt_rows",
            "version": "rowv1_entry_surface_v3",
            "dtype": "std::int64_t* rows, std::int64_t* counts, std::int32_t* overflow",
            "width": "kOrdinaryBirthRowV1ReceiptWidthV3 == 64",
            "disabled_reason_support": "reserved pointer surface not kernel qualified",
            "source_abi_version_support": "reserved pointer surface not kernel qualified",
            "stable_id_semantics": "row fields only",
            "dense_id_dependence": "none_in_receipt_shape",
            "authority": "device_staging_not_v4_authority",
            "authority_kind": "device_pointer_context",
            "persisted": False,
            "persistence_mode": "transient_kernel_context",
            "producer": "host binding passes raw tensor pointers",
            "consumer": "CUDA kernel writes bounded raw rows",
            "package_route_binding": "absent for RowV1/CUDA ordinary-birth admission",
            "required_action": "do not reuse padding; version any future CUDA ABI",
            "marker": "kTurnoverMutationReceiptWidthV1 = 21",
        },
        {
            "surface_id": "checkpoint_projection_mutation_receipt",
            "role": "checkpoint_projection_field",
            "host_or_device": "host",
            "path": "engine/perf/cuda_architecture_b_direct_resident_modular_v1.py",
            "symbol": "phase_e_active_modular_lifecycle_mutation_receipt",
            "version": "canonical_receipt_object",
            "dtype": "python_object_in_projection",
            "width": "not_fixed_width",
            "disabled_reason_support": "canonical_only_when_projected",
            "source_abi_version_support": "not_frozen_before_this_gate",
            "stable_id_semantics": "projection follows sparse identity authority",
            "dense_id_dependence": "none",
            "authority": "checkpoint_projection_consumer",
            "authority_kind": "checkpoint_projection_field",
            "persisted": True,
            "persistence_mode": "checkpoint_projection_object",
            "producer": "host projection",
            "consumer": "checkpoint/report publication and restore validation",
            "package_route_binding": "RowV1 envelope and collection binding required",
            "required_action": "freeze RowV1 checkpoint/report envelope before route admission",
            "marker": "phase_e_active_modular_lifecycle_mutation_receipt",
        },
    )
    audited = []
    for row in surfaces:
        marker = str(row["marker"])
        material = {key: value for key, value in row.items() if key != "marker"}
        material["source_evidence"] = _source_row(root, str(row["path"]), marker)
        audited.append(material)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "contract_id": SURFACE_INVENTORY_CONTRACT_ID,
        "decision": DECISION,
        "surface_count": len(audited),
        "surfaces": audited,
        "required_surface_fields": (
            "path",
            "symbol",
            "version",
            "dtype",
            "width",
            "authority_kind",
            "host_or_device",
            "persisted",
            "persistence_mode",
            "producer",
            "consumer",
            "disabled_reason_support",
            "source_abi_version_support",
            "stable_id_semantics",
            "dense_id_dependence",
            "package_route_binding",
            "required_action",
        ),
        "cuda_receipt_abi_present": False,
        "cuda_rowv1_v3_kernel_mirror_present": True,
        "cuda_rowv1_v3_kernel_mirror_projection_present": True,
        "cuda_existing_receipt_surface": (
            "raw_layout_v2_int64_device_staging_plus_rowv1_v3_kernel_mirror_projection"
        ),
        "cuda_disabled_reason_fits_without_layout_change": False,
        "cuda_layout_change_authorized": False,
        "dense_id_boundary": "receipt shapes are resident-capacity based; stable ids remain int64 fields",
        "required_next_action": "artifact-bound route/package/disabled-reason qualification before CUDA admission",
    }
    return _with_sha(payload, "inventory_sha256")


def rowv1_semantic_receipt_schema_v1() -> dict[str, Any]:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "contract_id": ROWV1_SCHEMA_CONTRACT_ID,
        "semantic_schema_version": "rowv1_ordinary_birth_receipt_semantic_v1",
        "field_order": ROWV1_REQUIRED_FIELDS,
        "nullability_policy": (
            "rejected and disabled rows must null selected identity, value, "
            "proposal digest, and child fields unless the source receipt proves them"
        ),
        "disabled_reason_policy": {
            "v4_disabled_receipt": "raw_disabled_reason_code_required_1_to_6",
            "v4_non_disabled_receipt": "raw_disabled_reason_code_zero_or_null",
            "v3_disabled_receipt": "raw_disabled_reason_code_null_defaults_operator_disabled",
            "unknown_code": "reject",
        },
        "source_binding": (
            "source receipt ABI version and source receipt SHA-256 are mandatory"
        ),
        "transaction_sequence_policy": {
            "scope": TRANSACTION_SEQUENCE_SCOPE,
            "initial_value": 0,
            "increment_rule": (
                "every admitted ordinary-birth transaction attempt consumes one "
                "per-cell sequence value from the authoritative lifecycle ledger"
            ),
            "accepted_rejected_disabled_invalid_behavior": (
                "accepted, capacity-rejected, publication-rejected, owner/token-"
                "rejected, disabled, and admitted invalid-draw attempts consume "
                "a sequence; malformed input before lifecycle admission does not"
            ),
            "rollback_behavior": (
                "rollback restores child state but never erases historical "
                "transaction order or reuses a sequence"
            ),
            "cancellation_behavior": "cancellation seals the last committed sequence",
            "checkpoint_persistence": (
                "checkpoint and report envelopes preserve sequence, scope, and "
                "lifecycle ledger hash"
            ),
            "fresh_owner_restore": (
                "restored owners resume from the checkpointed last sequence"
            ),
            "multicell_behavior": (
                "sequence is per cell; global collections are ordered by cell "
                "then authoritative transaction sequence"
            ),
            "not_generated_by": (
                "reporting, collection sort order, filesystem order, fixture "
                "enumeration, receipt hashes, or physical batch position"
            ),
        },
        "canonical_bytes": "canonical ASCII JSON with sorted keys and trailing newline",
        "dense_id_boundary": "stable biological IDs are int64 values, never array widths",
        "checkpoint_report_boundary": (
            "checkpoint and report rows persist source ABI version, source hash, "
            "raw disabled reason code, canonical reason, and mapping contract hash"
        ),
    }
    return _with_sha(payload)


def rowv1_fixture_bank_v1() -> dict[str, Any]:
    cases = (
        "accepted",
        "capacity_rejection",
        "publication_rejection",
        "owner_rejection",
        "legacy_v3_operator_disabled",
        "v4_disabled_operator_disabled",
        "v4_disabled_structural_mutation_disabled",
        "v4_disabled_recurrent_mutation_disabled",
        "v4_disabled_operator_family_disabled",
        "v4_disabled_zero_mutation_rate",
        "v4_disabled_mutation_attempt_budget_zero",
        "invalid_draw",
        "malformed_abi_version",
        "unknown_disabled_reason",
        "truncated_payload",
        "source_hash_mismatch",
        "canonical_reason_mismatch",
    )
    fixtures = tuple(
        {
            "case_id": case,
            "canonical_bytes_sha256": content_sha256(
                {"contract_id": FIXTURE_BANK_CONTRACT_ID, "case_id": case}
            ),
        }
        for case in cases
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "contract_id": FIXTURE_BANK_CONTRACT_ID,
        "rowv1_schema_contract_id": ROWV1_SCHEMA_CONTRACT_ID,
        "disabled_reason_receipt_contract_id": (
            NATIVE_CORE_DISABLED_REASON_RECEIPT_BYTES_CONTRACT_ID
        ),
        "disabled_reason_mapping_contract_id": NATIVE_CORE_DISABLED_REASON_SEMANTICS_CONTRACT_ID,
        "repeat_determinism_contract_id": NATIVE_CORE_RECEIPT_REPEAT_CONTRACT_ID,
        "fixture_count": len(fixtures),
        "fixtures": fixtures,
    }
    return _with_sha(payload, "fixture_bank_sha256")


def rowv1_ordinary_birth_projection_receipt_v1() -> dict[str, Any]:
    """Record the host RowV1 ordinary-birth projection implementation."""

    schema = rowv1_semantic_receipt_schema_v1()
    payload = {
        "schema_version": SCHEMA_VERSION,
        "contract_id": ROWV1_PROJECTION_CONTRACT_ID,
        "decision": "P12-ROWV1-W-ORDINARY-BIRTH-PROJECTION",
        "rowv1_schema_contract_id": ROWV1_SCHEMA_CONTRACT_ID,
        "rowv1_schema_sha256": schema["contract_sha256"],
        "semantic_mapping_contract_id": NATIVE_CORE_DISABLED_REASON_SEMANTICS_CONTRACT_ID,
        "semantic_mapping_contract_sha256": SEMANTIC_MAPPING_CONTRACT_SHA256,
        "projection_function": "project_native_core_ordinary_birth_receipt_rowv1_v1",
        "validation_function": "validate_rowv1_ordinary_birth_receipt_v1",
        "source_hash_function": "rowv1_source_receipt_sha256_v1",
        "canonical_bytes_function": "canonical_rowv1_ordinary_birth_receipt_bytes_v1",
        "covered_projection_cases": (
            "v4_disabled_reason_preserves_raw_code_and_canonical_reason",
            "legacy_v3_disabled_defaults_operator_disabled_without_raw_code",
            "accepted_changed_row_preserves_selected_identity_and_values",
            "publication_rejection_preserves_proposal_digest_sha256",
            "authoritative_transaction_sequence_preserved",
        ),
        "covered_negative_cases": (
            "v4_disabled_missing_raw_reason_rejected",
            "unknown_v4_disabled_reason_rejected",
            "nonzero_native_proposal_without_sha256_rejected",
            "source_receipt_hash_mismatch_rejected",
            "stale_semantic_mapping_contract_hash_rejected",
            "canonical_reason_mismatch_rejected",
            "rejected_or_disabled_mutation_fields_populated_rejected",
            "stable_id_outside_int64_rejected",
            "rng_index_rewind_rejected",
            "unsupported_transaction_sequence_scope_rejected",
        ),
        "limits": (
            "host_rowv1_projection_only",
            "not_checkpoint_resume_continuation_qualification",
            "not_cuda_abi_implementation",
            "not_cuda_parity",
            "not_route_package_qualification",
            "not_successor_campaign_authorization",
        ),
        "rowv1_projection_authorized": True,
        "cuda_abi_implementation_authorized": False,
        "cuda_parity_authorized": False,
        "route_package_qualification_authorized": False,
        "successor_campaign_authorized": False,
    }
    return _with_sha(payload)


def cuda_raw_v2_rowv1_projection_receipt_v1() -> dict[str, Any]:
    """Record bounded host RowV1 projection for current CUDA raw v2 rows."""

    minimum_abi = cuda_rowv1_v3_minimum_receipt_abi_contract_v1()
    payload = {
        "schema_version": SCHEMA_VERSION,
        "contract_id": CUDA_RAW_V2_ROWV1_PROJECTION_CONTRACT_ID,
        "decision": "P12-CUDA-RAW-V2-ROWV1-PROJECTION-PROGRESS",
        "rowv1_schema_contract_id": ROWV1_SCHEMA_CONTRACT_ID,
        "rowv1_schema_sha256": rowv1_semantic_receipt_schema_v1()["contract_sha256"],
        "projection_function": "project_cuda_raw_v2_ordinary_birth_receipt_rowv1_v1",
        "source_receipt_abi_version": 2,
        "source_layout": "ordinary_birth_mutation_receipt_raw_layout_v2",
        "source_row_width": CUDA_RAW_V2_RECEIPT_ROW_WIDTH,
        "minimum_authoritative_cuda_abi_contract_id": (
            CUDA_ROWV1_V3_MINIMUM_ABI_CONTRACT_ID
        ),
        "minimum_authoritative_cuda_abi_sha256": minimum_abi["contract_sha256"],
        "minimum_authoritative_cuda_abi_version": CUDA_ROWV1_V3_RECEIPT_ABI_VERSION,
        "minimum_authoritative_cuda_row_width": (
            CUDA_ROWV1_V3_MINIMUM_RECEIPT_ROW_WIDTH
        ),
        "covered_cases": (
            "accepted_cuda_raw_v2_row_projects_to_rowv1",
            "high_id_parent_and_child_fields_preserved_as_int64",
            "raw_row_source_sha256_bound_to_rowv1_source_receipt_sha256",
            "draw_bits_preserved_in_source_receipt_hash_input",
            "changed_row_requires_selected_identity",
            "row_width_change_rejected",
            "unsupported_cuda_enum_rejected",
            "invalid_owner_coordinates_rejected",
        ),
        "limits": (
            "host_projection_from_current_raw_v2_staging_layout_only",
            "accepted_policy_mutation_fixture_only",
            "raw_v2_lacks_disabled_reason_code",
            "raw_v2_lacks_source_abi_field_on_device",
            "raw_v2_lacks_proposal_digest_and_value_delta_fields",
            "not_authoritative_cuda_receipt_abi_admission",
            "not_cuda_route_qualification",
            "not_package_qualification",
            "not_successor_campaign_authorization",
        ),
        "cuda_rowv1_projection_progress": True,
        "cuda_receipt_surface_currently_admitted": False,
        "cuda_abi_implementation_authorized": False,
        "cuda_parity_authorized": False,
        "route_package_qualification_authorized": False,
        "successor_campaign_authorized": False,
    }
    return _with_sha(payload)


def cuda_rowv1_v3_kernel_mirror_projection_receipt_v1() -> dict[str, Any]:
    """Record bounded host projection for the CUDA RowV1 v3 kernel mirror."""

    minimum_abi = cuda_rowv1_v3_minimum_receipt_abi_contract_v1()
    payload = {
        "schema_version": SCHEMA_VERSION,
        "contract_id": CUDA_ROWV1_V3_KERNEL_MIRROR_PROJECTION_CONTRACT_ID,
        "decision": "P12-CUDA-ROWV1-V3-KERNEL-MIRROR-PROJECTION-PROGRESS",
        "rowv1_schema_contract_id": ROWV1_SCHEMA_CONTRACT_ID,
        "rowv1_schema_sha256": rowv1_semantic_receipt_schema_v1()["contract_sha256"],
        "projection_function": (
            "project_cuda_rowv1_v3_kernel_mirror_receipt_rowv1_v1"
        ),
        "source_receipt_abi_version": CUDA_ROWV1_V3_RECEIPT_ABI_VERSION,
        "source_layout": "ordinary_birth_rowv1_receipt_kernel_mirror_v3",
        "mirrored_source_layout": "ordinary_birth_mutation_receipt_raw_layout_v2",
        "source_row_width": CUDA_ROWV1_V3_MINIMUM_RECEIPT_ROW_WIDTH,
        "minimum_authoritative_cuda_abi_contract_id": (
            CUDA_ROWV1_V3_MINIMUM_ABI_CONTRACT_ID
        ),
        "minimum_authoritative_cuda_abi_sha256": minimum_abi["contract_sha256"],
        "minimum_authoritative_cuda_abi_version": CUDA_ROWV1_V3_RECEIPT_ABI_VERSION,
        "minimum_authoritative_cuda_row_width": (
            CUDA_ROWV1_V3_MINIMUM_RECEIPT_ROW_WIDTH
        ),
        "covered_cases": (
            "accepted_cuda_rowv1_v3_kernel_mirror_projects_to_rowv1",
            "kernel_mirror_abi_marker_bound",
            "kernel_mirror_raw_layout_marker_bound",
            "high_id_parent_and_child_fields_preserved_as_int64",
            "source_sha256_bound_to_full_64_word_row",
            "mirrored_raw_prefix_preserved_in_source_hash_input",
            "non_disabled_zero_disabled_reason_sentinel_bound",
            "disabled_reason_codes_1_to_6_project_to_canonical_reasons",
            "changed_row_requires_selected_identity",
            "row_width_change_rejected",
            "abi_marker_change_rejected",
            "source_layout_marker_change_rejected",
            "unsupported_cuda_enum_rejected",
            "unknown_disabled_reason_code_rejected",
            "changed_disabled_row_rejected",
        ),
        "limits": (
            "host_projection_from_current_rowv1_v3_kernel_mirror_only",
            "accepted_policy_mutation_fixture_only",
            "mirror_accepts_disabled_reason_codes_but_device_disabled_event_generation_is_not_qualified",
            "mirror_source_hash_bound_but_not_route_or_package_qualified",
            "not_authoritative_cuda_receipt_abi_admission",
            "not_cuda_route_qualification",
            "not_package_qualification",
            "not_successor_campaign_authorization",
        ),
        "cuda_rowv1_v3_kernel_mirror_projection_progress": True,
        "cuda_receipt_surface_currently_admitted": False,
        "cuda_abi_implementation_authorized": False,
        "cuda_parity_authorized": False,
        "route_package_qualification_authorized": False,
        "successor_campaign_authorized": False,
    }
    return _with_sha(payload)


def cuda_rowv1_disabled_event_writer_receipt_v1(
    *,
    source_root: str | Path = ".",
) -> dict[str, Any]:
    """Bind the device-side RowV1 disabled-event writer implementation."""

    root = Path(source_root)
    minimum_abi = cuda_rowv1_v3_minimum_receipt_abi_contract_v1()
    payload = {
        "schema_version": SCHEMA_VERSION,
        "contract_id": CUDA_ROWV1_DISABLED_EVENT_WRITER_CONTRACT_ID,
        "decision": "P12-CUDA-ROWV1-DISABLED-EVENT-WRITER-A",
        "minimum_authoritative_cuda_abi_contract_id": (
            CUDA_ROWV1_V3_MINIMUM_ABI_CONTRACT_ID
        ),
        "minimum_authoritative_cuda_abi_sha256": minimum_abi["contract_sha256"],
        "source_receipt_abi_version": CUDA_ROWV1_V3_RECEIPT_ABI_VERSION,
        "source_row_width": CUDA_ROWV1_V3_MINIMUM_RECEIPT_ROW_WIDTH,
        "sequence_capacity_contract": "runtime_rows_capacity_controls_admission",
        "device_sources": (
            _source_row(
                root,
                "engine/perf/p12_cuda_rowv1_disabled_receipts_v1.cuh",
                "write_cell",
            ),
            _source_row(
                root,
                "engine/perf/native/p12_cuda_rowv1_disabled_receipts_v1.cu",
                "eiel_p12_cuda_rowv1_disabled_receipts_write_v1",
            ),
            _source_row(
                root,
                "engine/perf/p12_cuda_rowv1_disabled_receipts_v1.py",
                "CUDA_DISABLED_RECEIPT_CAPABILITIES",
            ),
            _source_row(
                root,
                "tests/perf/test_p12_cuda_rowv1_disabled_receipts_v1_standalone.cu",
                "row[22] != reasons[input]",
            ),
            _source_row(
                root,
                "engine/perf/cuda_architecture_b_flattened_modular_controller_v1.cuh",
                "write_ordinary_birth_rowv1_disabled_event_v3",
            ),
            _source_row(
                root,
                "engine/perf/cuda_architecture_b_flattened_wdl_controller_v2_binding.cpp",
                'out["ordinary_birth_rowv1_receipt_kernel_write_qualified"] = true',
            ),
            _source_row(
                root,
                "tests/perf/test_p12_cuda_modular_rowv1_device_writer_v1.cu",
                "rowv1_device_writer_probe",
            ),
        ),
        "device_written_fields": (
            "source_receipt_abi_version",
            "raw_disabled_reason_code",
            "transaction_sequence",
            "transaction_id_attempt_index",
            "parent_agent_id",
            "parent_slot",
            "parent_slot_token",
            "child_agent_id",
            "child_slot",
            "child_slot_token",
            "generation",
            "tick",
            "changed_zero",
            "counts",
            "overflow",
        ),
        "disabled_reason_codes": (
            {"code": 1, "canonical_reason": "operator_disabled"},
            {"code": 2, "canonical_reason": "structural_mutation_disabled"},
            {"code": 3, "canonical_reason": "recurrent_mutation_disabled"},
            {"code": 4, "canonical_reason": "operator_family_disabled"},
            {"code": 5, "canonical_reason": "zero_mutation_rate"},
            {"code": 6, "canonical_reason": "mutation_attempt_budget_zero"},
        ),
        "capacity_semantics": (
            "zero_capacity_overflows_without_rows",
            "one_capacity_writes_one_row",
            "max_minus_one_capacity_writes_without_overflow",
            "max_capacity_writes_without_overflow",
            "max_plus_one_attempt_overflows_after_capacity_rows",
            "unknown_reason_sets_overflow",
        ),
        "validation_commands": (
            "nvcc -std=c++17 -O2 -I engine/perf tests/perf/test_p12_cuda_rowv1_disabled_receipts_v1_standalone.cu -o /tmp/p12_cuda_rowv1_disabled/test_p12_cuda_rowv1_disabled && /tmp/p12_cuda_rowv1_disabled/test_p12_cuda_rowv1_disabled",
            "compute-sanitizer --tool memcheck /tmp/p12_cuda_rowv1_disabled/test_p12_cuda_rowv1_disabled",
            "pytest -q tests/perf/test_p12_cuda_rowv1_disabled_receipts_v1.py tests/test_p12_rowv1_cuda_entry_gate_v1.py",
            "pytest -q tests/perf/test_p12_cuda_modular_rowv1_device_writer_v1.py",
        ),
        "torch_cuda_runtime_status": "pytest_cuda_extension_skips_when_active_torch_is_cpu_only",
        "standalone_nvcc_runtime_probe_required": True,
        "standalone_nvcc_runtime_probe_passed": True,
        "compute_sanitizer_tool": "memcheck",
        "compute_sanitizer_error_summary": "0_errors",
        "sanitizer_bounds_cases": (
            "receipt_rows_bounds",
            "receipt_counts_bounds",
            "receipt_overflow_bounds",
        ),
        "sanitizer_passed": True,
        "device_disabled_event_writer_present": True,
        "route_package_qualification_authorized": False,
        "successor_campaign_authorized": False,
    }
    return _with_sha(payload)


def cuda_rowv1_modular_device_writer_admission_descriptor_v1() -> dict[str, Any]:
    """Build the current bounded modular CUDA RowV1 ABI admission descriptor."""

    minimum_abi = cuda_rowv1_v3_minimum_receipt_abi_contract_v1()
    row_width = CUDA_ROWV1_V3_MINIMUM_RECEIPT_ROW_WIDTH
    return {
        "admission_contract_id": CUDA_RECEIPT_SURFACE_ADMISSION_CONTRACT_ID,
        "authoritative_cuda_receipt_abi_present": True,
        "declared_cuda_receipt_abi_version": CUDA_ROWV1_V3_RECEIPT_ABI_VERSION,
        "device_layout_preserves_rowv1_semantics": True,
        "receipt_surface_present": True,
        "receipt_surface_row_count": CUDA_ROWV1_V3_SEQUENCE_CAPACITY,
        "host_receipt_size_bytes": row_width * 8,
        "device_receipt_size_bytes": row_width * 8,
        "host_receipt_alignment_bytes": 8,
        "device_receipt_alignment_bytes": 8,
        "row_dtype": "int64",
        "counts_dtype": "int64",
        "overflow_dtype": "int32",
        "receipt_row_width": row_width,
        "expected_receipt_row_width": row_width,
        "decoder_row_width": row_width,
        "checkpoint_manifest_receipt_width": row_width,
        "checkpoint_manifest_receipt_abi_version": CUDA_ROWV1_V3_RECEIPT_ABI_VERSION,
        "checkpoint_manifest_preserves_source_abi": True,
        "checkpoint_manifest_preserves_disabled_reason": True,
        "package_receipt_abi_sha256": minimum_abi["contract_sha256"],
        "route_rowv1_mapping_contract_sha256": SEMANTIC_MAPPING_CONTRACT_SHA256,
        "sanitizer_bounds_cases": [
            "receipt_rows_bounds",
            "receipt_counts_bounds",
            "receipt_overflow_bounds",
        ],
        "dense_id_authority": "sparse_identity_table_and_slot_token",
    }


def cuda_rowv1_modular_device_writer_admission_receipt_v1(
    *,
    source_root: str | Path = ".",
) -> dict[str, Any]:
    """Bind the admitted bounded modular CUDA RowV1 device-writer ABI."""

    root = Path(source_root)
    descriptor = cuda_rowv1_modular_device_writer_admission_descriptor_v1()
    admission = validate_cuda_receipt_surface_admission_v1(descriptor)
    writer = cuda_rowv1_disabled_event_writer_receipt_v1(source_root=root)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "contract_id": CUDA_ROWV1_MODULAR_DEVICE_WRITER_ADMISSION_CONTRACT_ID,
        "decision": "P12-CUDA-ROWV1-MODULAR-DEVICE-WRITER-ABI-ADMITTED",
        "cuda_receipt_surface_admission_contract_id": (
            CUDA_RECEIPT_SURFACE_ADMISSION_CONTRACT_ID
        ),
        "cuda_receipt_surface_admission_sha256": admission["contract_sha256"],
        "cuda_rowv1_disabled_event_writer_contract_id": writer["contract_id"],
        "cuda_rowv1_disabled_event_writer_sha256": writer["contract_sha256"],
        "minimum_authoritative_cuda_abi_contract_id": (
            CUDA_ROWV1_V3_MINIMUM_ABI_CONTRACT_ID
        ),
        "minimum_authoritative_cuda_abi_sha256": (
            admission["cuda_receipt_abi_contract_sha256"]
        ),
        "source_evidence": (
            _source_row(
                root,
                "engine/perf/cuda_architecture_b_flattened_modular_controller_v1.cuh",
                "write_ordinary_birth_rowv1_receipt_v3",
            ),
            _source_row(
                root,
                "engine/perf/cuda_architecture_b_flattened_modular_controller_v1.cuh",
                "write_ordinary_birth_rowv1_disabled_event_v3",
            ),
            _source_row(
                root,
                "engine/perf/cuda_architecture_b_flattened_wdl_controller_v2_binding.cpp",
                'out["ordinary_birth_rowv1_receipt_device_writer_sanitized"] = true',
            ),
            _source_row(
                root,
                "tests/perf/test_p12_cuda_modular_rowv1_device_writer_v1.py",
                "test_modular_rowv1_device_writer_probe_passes_nvcc_and_memcheck",
            ),
        ),
        "admitted_fields": {
            "source_receipt_abi_version": "kernel_written",
            "raw_disabled_reason_code": "kernel_written",
            "canonical_disabled_reason": "exact_host_projection_from_kernel_reason_code",
            "transaction_sequence_and_id": "kernel_written_sequence_and_attempt_index",
            "accepted_rejected_disabled_state": "kernel_written_status_reason_changed_fields",
            "count_overflow_and_dropped_row_state": "kernel_written_count_and_overflow",
            "slot_token_and_stable_id_identity": "copied_from_authoritative_device_state",
            "checkpoint_manifest_width_and_abi": "bound_by_rowv1_checkpoint_report_contract",
            "dense_id_authority": "sparse_identity_table_and_slot_token",
        },
        "limits": (
            "bounded modular device-writer ABI admission only",
            "not compact CPU/CUDA parity",
            "not immutable CUDA artifact route qualification",
            "not final P12 package qualification",
            "not successor campaign authorization",
        ),
        "cuda_receipt_surface_admitted": True,
        "compact_cpu_cuda_parity_complete": False,
        "route_package_qualification_authorized": False,
        "successor_campaign_authorized": False,
    }
    return _with_sha(payload)


def _compact_cuda_rowv1_fixture_row_v1(
    *,
    status: int,
    reason: int,
    changed: int,
    raw_disabled_reason: int,
) -> list[int]:
    raw = [
        1,
        0,
        1,
        7,
        (1 << 40) + 10,
        0,
        100,
        (1 << 40) + 20,
        3,
        200,
        0,
        int("3fe3333333333333", 16),
        int("3fe5c00000000000", 16),
        2,
        status,
        reason,
        0 if changed else -1,
        0 if changed else -1,
        0 if changed else -1,
        5 if changed else 0,
        changed,
    ]
    row = [0] * CUDA_ROWV1_V3_MINIMUM_RECEIPT_ROW_WIDTH
    row[0] = CUDA_ROWV1_V3_RECEIPT_ABI_VERSION
    row[1] = 2
    row[2:22] = raw[1:21]
    row[22] = raw_disabled_reason
    row[23:] = [-1] * (CUDA_ROWV1_V3_MINIMUM_RECEIPT_ROW_WIDTH - 23)
    return row


def _compact_native_rowv1_fixture_receipt_v1(
    *,
    transaction_status: int,
    changed: int,
    disabled_reason_code: int | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "transaction_status": transaction_status,
        "native_status": 0,
        "operation": 2,
        "selected_slot": 0 if changed else -1,
        "changed_dimension": (1 << 8) if changed else 0,
        "changed": changed,
        "committed": changed,
        "capacity_available": 1,
        "publish_success": 1,
        "parent_agent_id": (1 << 40) + 10,
        "parent_slot": 0,
        "parent_slot_token": 100,
        "child_agent_id": (1 << 40) + 20,
        "child_slot": 3,
        "expected_child_slot_token": 200,
        "child_publication_token": 200,
        "draw0_float64_bits": int("3fe3333333333333", 16),
        "draw1_float64_bits": int("3fe5c00000000000", 16),
        "parent_digest_before": 11,
        "child_digest_before": 22,
        "proposal_digest": 123 if changed else 0,
        "child_digest_after": 23 if changed else 22,
        "selected_module_instance_id": 0,
        "selected_homology_id": 0,
        "value_before_float32": 0.0,
        "value_after_float32": 1.0 if changed else 0.0,
    }
    if disabled_reason_code is not None:
        payload["disabled_reason_code"] = disabled_reason_code
    return payload


def _compact_cpu_cuda_parity_projection_v1(
    native_row: Mapping[str, Any],
    cuda_row: Sequence[int],
    *,
    selected_identity: Mapping[str, Any] | None,
    authoritative_transaction_sequence: int = 0,
) -> dict[str, Any]:
    common = {
        "transaction_id": 1,
        "cell": 0,
        "rng_stream_id": "ordinary_birth_mutation_v1",
        "rng_index_before": 40,
        "rng_index_after": 42,
        "allocator_before": {"next_agent_id": (1 << 40) + 20},
        "allocator_after": {"next_agent_id": (1 << 40) + 21},
        "package_binding_sha256": "a" * 64,
        "schema_binding_sha256": "b" * 64,
        "mechanics_binding_sha256": "c" * 64,
        "selected_identity": selected_identity,
        "authoritative_transaction_sequence": authoritative_transaction_sequence,
    }
    native = project_native_core_ordinary_birth_receipt_rowv1_v1(
        native_row,
        source_receipt_abi_version=4,
        generation=1,
        tick=7,
        proposal_digest_sha256="d" * 64
        if int(native_row.get("proposal_digest", 0)) != 0
        else None,
        selected_instance_id_index=0 if selected_identity is not None else -1,
        selected_homology_id_index=0 if selected_identity is not None else -1,
        **common,
    )
    cuda = project_cuda_rowv1_v3_kernel_mirror_receipt_rowv1_v1(
        cuda_row,
        **common,
    )
    parity_fields = (
        "native_transaction_status",
        "canonical_status",
        "transaction_id",
        "authoritative_transaction_sequence",
        "transaction_sequence_scope",
        "parent_agent_id",
        "parent_slot",
        "parent_slot_token",
        "child_agent_id",
        "child_slot",
        "child_slot_token",
        "cell",
        "generation",
        "tick",
        "resident_slot",
        "mutation_family",
        "operator_id",
        "attempt_index",
        "rng_stream_id",
        "rng_index_before",
        "rng_index_after",
        "selected_identity",
        "changed",
        "capacity_decision",
        "publication_decision",
        "rollback_decision",
        "allocator_before",
        "allocator_after",
        "package_binding_sha256",
        "schema_binding_sha256",
        "mechanics_binding_sha256",
    )
    mismatches = [
        field for field in parity_fields if native.get(field) != cuda.get(field)
    ]
    return {
        "native_rowv1_sha256": rowv1_ordinary_birth_receipt_sha256_v1(native),
        "cuda_rowv1_sha256": rowv1_ordinary_birth_receipt_sha256_v1(cuda),
        "parity_fields": parity_fields,
        "mismatches": tuple(mismatches),
        "source_specific_fields": (
            "source_receipt_abi_version",
            "source_receipt_sha256",
            "raw_disabled_reason_code",
            "canonical_reason",
            "proposal_digest",
            "value_before",
            "value_after",
        ),
        "native_status": native["canonical_status"],
        "cuda_status": cuda["canonical_status"],
        "native_reason": native["canonical_reason"],
        "cuda_reason": cuda["canonical_reason"],
    }


def cuda_rowv1_compact_cpu_cuda_parity_receipt_v1(
    *,
    source_root: str | Path = ".",
) -> dict[str, Any]:
    """Bind compact native/CUDA RowV1 parity for the admitted device ABI."""

    root = Path(source_root)
    admission = cuda_rowv1_modular_device_writer_admission_receipt_v1(
        source_root=root
    )
    accepted = _compact_cpu_cuda_parity_projection_v1(
        _compact_native_rowv1_fixture_receipt_v1(
            transaction_status=TRANSACTION_COMMITTED,
            changed=1,
        ),
        _compact_cuda_rowv1_fixture_row_v1(
            status=1,
            reason=1,
            changed=1,
            raw_disabled_reason=0,
        ),
        selected_identity={
            "selected_module_instance_id": 0,
            "selected_homology_id": 0,
        },
    )
    disabled = _compact_cpu_cuda_parity_projection_v1(
        _compact_native_rowv1_fixture_receipt_v1(
            transaction_status=TRANSACTION_MUTATION_DISABLED,
            changed=0,
            disabled_reason_code=2,
        ),
        _compact_cuda_rowv1_fixture_row_v1(
            status=3,
            reason=2,
            changed=0,
            raw_disabled_reason=2,
        ),
        selected_identity=None,
    )
    if accepted["mismatches"] or disabled["mismatches"]:
        raise P12RowV1CudaEntryGateError("compact CPU/CUDA RowV1 parity mismatch")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "contract_id": CUDA_ROWV1_COMPACT_CPU_CUDA_PARITY_CONTRACT_ID,
        "decision": "P12-CUDA-ROWV1-COMPACT-CPU-CUDA-PARITY-A",
        "cuda_rowv1_modular_device_writer_admission_contract_id": (
            admission["contract_id"]
        ),
        "cuda_rowv1_modular_device_writer_admission_sha256": (
            admission["contract_sha256"]
        ),
        "covered_cases": (
            {
                "case_id": "accepted_refine_descendant_parameter_reset_bias",
                **accepted,
            },
            {
                "case_id": "disabled_structural_mutation_reason",
                **disabled,
            },
        ),
        "source_evidence": (
            _source_row(
                root,
                "engine/evolution/p12_rowv1_cuda_entry_gate_v1.py",
                "def cuda_rowv1_compact_cpu_cuda_parity_receipt_v1",
            ),
            _source_row(
                root,
                "tests/test_p12_rowv1_cuda_entry_gate_v1.py",
                "test_cuda_rowv1_compact_cpu_cuda_parity_receipt_is_positive_but_bounded",
            ),
            _source_row(
                root,
                "tests/perf/test_p12_cuda_modular_rowv1_device_writer_v1.cu",
                "rowv1_device_writer_probe",
            ),
        ),
        "limits": (
            "compact RowV1 ABI field parity for accepted and disabled fixture rows",
            "source-specific payload, proposal, value, reason-label, and source-hash fields are explicitly excluded",
            "not immutable CUDA artifact route qualification",
            "not final P12 package qualification",
            "not successor campaign authorization",
        ),
        "compact_cpu_cuda_parity_complete": True,
        "route_package_qualification_authorized": False,
        "successor_campaign_authorized": False,
    }
    return _with_sha(payload)


def rowv1_decoder_admission_receipt_v1() -> dict[str, Any]:
    """Record the RowV1 decoder admission contract and fail-closed cases."""

    payload = {
        "schema_version": SCHEMA_VERSION,
        "contract_id": ROWV1_DECODER_ADMISSION_CONTRACT_ID,
        "decision": "P12-ROWV1-Z-DECODER-ADMISSION",
        "rowv1_schema_contract_id": ROWV1_SCHEMA_CONTRACT_ID,
        "rowv1_schema_sha256": rowv1_semantic_receipt_schema_v1()["contract_sha256"],
        "field_order": ROWV1_DECODER_ADMISSION_FIELDS,
        "admission_function": "validate_rowv1_decoder_admission_v1",
        "covered_cases": (
            "homogeneous_v4_rows_admitted",
            "v4_row_presented_to_v3_only_decoder_rejected",
            "stale_decoder_width_rejected",
            "mixed_v3_v4_batch_rejected",
            "zero_length_or_absent_receipt_surface_rejected",
        ),
        "limits": (
            "host_decoder_admission_only",
            "not_cuda_abi_implementation",
            "not_cuda_parity",
            "not_route_package_qualification",
            "not_successor_campaign_authorization",
        ),
        "rowv1_decoder_admission_authorized": True,
        "cuda_abi_implementation_authorized": False,
        "cuda_parity_authorized": False,
        "route_package_qualification_authorized": False,
        "successor_campaign_authorized": False,
    }
    return _with_sha(payload)


def cuda_receipt_surface_admission_receipt_v1(
    *,
    source_root: str | Path = ".",
) -> dict[str, Any]:
    """Record the CUDA receipt admission contract for the current bounded ABI."""

    minimum_abi = cuda_rowv1_v3_minimum_receipt_abi_contract_v1()
    current_admission = validate_cuda_receipt_surface_admission_v1(
        cuda_rowv1_modular_device_writer_admission_descriptor_v1()
    )
    current_writer = cuda_rowv1_modular_device_writer_admission_receipt_v1(
        source_root=source_root
    )
    compact_parity = cuda_rowv1_compact_cpu_cuda_parity_receipt_v1(
        source_root=source_root
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "contract_id": CUDA_RECEIPT_SURFACE_ADMISSION_CONTRACT_ID,
        "decision": (
            "P12-CUDA-ORDINARY-BIRTH-RECEIPT-ABI-ADMITTED-ROUTE-PACKAGE-OPEN"
        ),
        "field_order": CUDA_RECEIPT_SURFACE_ADMISSION_FIELDS,
        "admission_function": "validate_cuda_receipt_surface_admission_v1",
        "minimum_authoritative_cuda_abi_contract_id": (
            CUDA_ROWV1_V3_MINIMUM_ABI_CONTRACT_ID
        ),
        "minimum_authoritative_cuda_abi_sha256": minimum_abi["contract_sha256"],
        "minimum_authoritative_cuda_abi_version": CUDA_ROWV1_V3_RECEIPT_ABI_VERSION,
        "minimum_authoritative_cuda_row_width": (
            CUDA_ROWV1_V3_MINIMUM_RECEIPT_ROW_WIDTH
        ),
        "current_cuda_receipt_abi_present": True,
        "current_cuda_rowv1_v3_entry_surface_present": True,
        "current_cuda_rowv1_v3_kernel_mirror_present": True,
        "current_cuda_rowv1_v3_kernel_mirror_projection_present": True,
        "current_cuda_rowv1_v3_entry_surface_width": (
            CUDA_ROWV1_V3_MINIMUM_RECEIPT_ROW_WIDTH
        ),
        "current_cuda_rowv1_v3_entry_surface_capacity": (
            CUDA_ROWV1_V3_SEQUENCE_CAPACITY
        ),
        "current_cuda_field_decision": (
            "bounded_device_writer_abi_and_compact_parity_complete_but_route_not_package_qualified"
        ),
        "current_cuda_receipt_surface_admission_sha256": (
            current_admission["contract_sha256"]
        ),
        "current_cuda_modular_device_writer_admission_contract_id": (
            current_writer["contract_id"]
        ),
        "current_cuda_modular_device_writer_admission_sha256": (
            current_writer["contract_sha256"]
        ),
        "current_cuda_compact_cpu_cuda_parity_contract_id": (
            compact_parity["contract_id"]
        ),
        "current_cuda_compact_cpu_cuda_parity_sha256": (
            compact_parity["contract_sha256"]
        ),
        "current_cuda_disabled_event_writer_contract_id": (
            CUDA_ROWV1_DISABLED_EVENT_WRITER_CONTRACT_ID
        ),
        "current_cuda_disabled_event_writer_sha256": (
            cuda_rowv1_disabled_event_writer_receipt_v1(source_root=source_root)[
                "contract_sha256"
            ]
        ),
        "current_cuda_disabled_event_writer_sanitizer": "memcheck_0_errors",
        "covered_failclosed_cases": CUDA_FAIL_CLOSED_CASES
        + (
            "zero_length_or_absent_receipt_surface",
            "unsupported_cuda_rowv1_receipt_abi_version",
            "wrong_cuda_rowv1_v3_receipt_width",
            "unknown_cuda_rowv1_v3_disabled_reason_code",
            "changed_cuda_rowv1_v3_disabled_row",
            "package_missing_cuda_rowv1_v3_abi_binding",
            "wrong_checkpoint_manifest_receipt_width",
            "wrong_checkpoint_manifest_receipt_abi_version",
        ),
        "limits": (
            "CUDA RowV1 v3 bounded device-writer ABI is admitted for the modular ordinary-birth receipt surface",
            "compact CPU/CUDA RowV1 ABI parity is complete only for accepted and disabled fixture fields",
            "the admitted ABI is not immutable CUDA artifact route qualification",
            "the admitted ABI is not final P12 package qualification",
            "not_route_package_qualification",
            "not_successor_campaign_authorization",
        ),
        "cuda_receipt_surface_currently_admitted": True,
        "cuda_abi_implementation_authorized": True,
        "compact_cpu_cuda_parity_complete": True,
        "cuda_parity_authorized": True,
        "route_package_qualification_authorized": False,
        "successor_campaign_authorized": False,
    }
    return _with_sha(payload)


def rowv1_checkpoint_report_envelope_receipt_v1() -> dict[str, Any]:
    """Record the frozen checkpoint/report representation for RowV1 receipts."""

    schema = rowv1_semantic_receipt_schema_v1()
    projection = rowv1_ordinary_birth_projection_receipt_v1()
    payload = {
        "schema_version": SCHEMA_VERSION,
        "contract_id": CHECKPOINT_REPORT_ENVELOPE_CONTRACT_ID,
        "decision": "P12-ROWV1-X-CHECKPOINT-REPORT-ENVELOPE",
        "envelope_schema_version": CHECKPOINT_REPORT_ENVELOPE_VERSION,
        "rowv1_schema_contract_id": ROWV1_SCHEMA_CONTRACT_ID,
        "rowv1_schema_sha256": schema["contract_sha256"],
        "rowv1_projection_contract_id": ROWV1_PROJECTION_CONTRACT_ID,
        "rowv1_projection_sha256": projection["contract_sha256"],
        "semantic_mapping_contract_sha256": SEMANTIC_MAPPING_CONTRACT_SHA256,
        "field_order": CHECKPOINT_REPORT_ENVELOPE_FIELDS,
        "covered_boundaries": CHECKPOINT_REPORT_BOUNDARIES,
        "preserved_identity_fields": (
            "source_receipt_abi_version",
            "source_receipt_sha256",
            "raw_disabled_reason_code",
            "canonical_status",
            "canonical_reason",
            "rng_stream_id",
            "rng_index_before",
            "rng_index_after",
            "allocator_before",
            "allocator_after",
            "package_binding_sha256",
            "schema_binding_sha256",
            "mechanics_binding_sha256",
            "rowv1_receipt_sha256",
            "authoritative_transaction_sequence",
            "transaction_sequence_scope",
            "lifecycle_ledger_sha256",
            "source_checkpoint_boundary",
        ),
        "failclosed_cases": (
            "missing_rowv1_receipt_hash_rejected",
            "stale_rowv1_schema_hash_rejected",
            "stale_projection_hash_rejected",
            "stale_mapping_hash_rejected",
            "source_receipt_abi_drift_rejected",
            "source_receipt_hash_drift_rejected",
            "raw_disabled_reason_drift_rejected",
            "canonical_reason_drift_rejected",
            "rng_index_drift_rejected",
            "allocator_state_drift_rejected",
            "package_schema_mechanics_binding_drift_rejected",
            "unsupported_boundary_rejected",
            "transaction_sequence_drift_rejected",
            "lifecycle_ledger_hash_missing_rejected",
        ),
        "limits": (
            "host_checkpoint_report_schema_only",
            "not_checkpoint_resume_continuation_qualification",
            "not_cuda_abi_implementation",
            "not_cuda_parity",
            "not_route_package_qualification",
            "not_successor_campaign_authorization",
        ),
        "rowv1_checkpoint_report_envelope_authorized": True,
        "cuda_abi_implementation_authorized": False,
        "cuda_parity_authorized": False,
        "route_package_qualification_authorized": False,
        "successor_campaign_authorized": False,
    }
    return _with_sha(payload)


def rowv1_checkpoint_report_collection_receipt_v1() -> dict[str, Any]:
    """Record the frozen zero/one/many receipt collection envelope."""

    envelope = rowv1_checkpoint_report_envelope_receipt_v1()
    payload = {
        "schema_version": SCHEMA_VERSION,
        "contract_id": CHECKPOINT_REPORT_COLLECTION_CONTRACT_ID,
        "decision": "P12-ROWV1-Y-CHECKPOINT-REPORT-COLLECTION",
        "collection_schema_version": CHECKPOINT_REPORT_COLLECTION_VERSION,
        "rowv1_checkpoint_report_envelope_contract_id": (
            CHECKPOINT_REPORT_ENVELOPE_CONTRACT_ID
        ),
        "rowv1_checkpoint_report_envelope_sha256": envelope["contract_sha256"],
        "field_order": CHECKPOINT_REPORT_COLLECTION_FIELDS,
        "covered_segments": CHECKPOINT_REPORT_COLLECTION_SEGMENTS,
        "deterministic_order_key": (
            "checkpoint_cell,authoritative_transaction_sequence,"
            "transaction_id,attempt_index,rowv1_receipt_sha256"
        ),
        "canonical_semantic_order": (
            "checkpoint_cell",
            "authoritative_transaction_sequence",
            "transaction_id",
            "attempt_index",
            "rowv1_receipt_sha256",
        ),
        "transaction_identity_key": "checkpoint_cell,transaction_id",
        "sequence_scope": TRANSACTION_SEQUENCE_SCOPE,
        "sequence_consumption_table": (
            {"case": "accepted_mutation", "consumes_sequence": True},
            {"case": "capacity_rejection", "consumes_sequence": True},
            {"case": "publication_rejection", "consumes_sequence": True},
            {"case": "owner_or_token_rejection", "consumes_sequence": True},
            {"case": "disabled_mutation", "consumes_sequence": True},
            {"case": "admitted_invalid_draw", "consumes_sequence": True},
            {"case": "malformed_input_before_admission", "consumes_sequence": False},
            {"case": "cancelled_before_admission", "consumes_sequence": False},
            {"case": "cancelled_after_admission", "consumes_sequence": True},
        ),
        "collection_cases": (
            "zero_receipts",
            "one_receipt",
            "many_receipts",
            "multicell_receipts",
            "checkpoint_prefix",
            "resumed_suffix",
            "checkpoint_plus_resume",
            "cancelled_sealed_collection",
        ),
        "failclosed_cases": (
            "receipt_count_mismatch_rejected",
            "duplicate_rowv1_receipt_rejected",
            "duplicate_transaction_identity_rejected",
            "nondeterministic_order_rejected",
            "wrong_run_cell_rejected",
            "stale_envelope_contract_hash_rejected",
            "receipt_envelope_hash_list_mismatch_rejected",
            "resumed_suffix_without_parent_rejected",
            "checkpoint_plus_resume_without_prefix_rejected",
            "reporting_derived_expected_count_rejected",
            "missing_or_failed_empty_collection_rejected",
            "per_cell_completeness_mismatch_rejected",
            "collection_capacity_limit_exceeded_rejected",
            "checkpoint_lineage_join_binding_mismatch_rejected",
            "checkpoint_rng_allocator_identity_or_lineage_drift_rejected",
            "overflow_count_rejected",
            "dropped_receipt_count_rejected",
            "unsealed_cancellation_rejected",
            "cancellation_seal_mismatch_rejected",
            "duplicate_transaction_sequence_rejected",
            "ordered_sequence_list_mismatch_rejected",
            "per_cell_sequence_range_mismatch_rejected",
            "per_cell_collection_hash_mismatch_rejected",
            "global_collection_hash_mismatch_rejected",
            "prefix_suffix_sequence_gap_or_overlap_rejected",
            "sequence_regression_after_restore_rejected",
            "missing_collector_success_state_rejected",
        ),
        "limits": (
            "host_collection_schema_only",
            "not_checkpoint_resume_continuation_qualification",
            "not_cuda_abi_implementation",
            "not_cuda_parity",
            "not_route_package_qualification",
            "not_successor_campaign_authorization",
        ),
        "rowv1_checkpoint_report_collection_authorized": True,
        "cuda_abi_implementation_authorized": False,
        "cuda_parity_authorized": False,
        "route_package_qualification_authorized": False,
        "successor_campaign_authorized": False,
    }
    return _with_sha(payload)


def rowv1_cuda_failclosed_entry_gate_v1(
    *,
    source_root: str | Path = ".",
) -> dict[str, Any]:
    inventory = rowv1_cuda_surface_inventory_v1(source_root)
    schema = rowv1_semantic_receipt_schema_v1()
    fixtures = rowv1_fixture_bank_v1()
    decoder_admission = rowv1_decoder_admission_receipt_v1()
    cuda_admission = cuda_receipt_surface_admission_receipt_v1(
        source_root=source_root
    )
    minimum_cuda_abi = cuda_rowv1_v3_minimum_receipt_abi_contract_v1()
    cuda_raw_v2_projection = cuda_raw_v2_rowv1_projection_receipt_v1()
    cuda_rowv1_v3_mirror_projection = (
        cuda_rowv1_v3_kernel_mirror_projection_receipt_v1()
    )
    cuda_disabled_writer = cuda_rowv1_disabled_event_writer_receipt_v1(
        source_root=source_root
    )
    cuda_modular_writer_admission = (
        cuda_rowv1_modular_device_writer_admission_receipt_v1(
            source_root=source_root
        )
    )
    cuda_compact_parity = cuda_rowv1_compact_cpu_cuda_parity_receipt_v1(
        source_root=source_root
    )
    checkpoint_report = rowv1_checkpoint_report_envelope_receipt_v1()
    checkpoint_collection = rowv1_checkpoint_report_collection_receipt_v1()
    payload = {
        "schema_version": SCHEMA_VERSION,
        "contract_id": ENTRY_GATE_CONTRACT_ID,
        "decision": DECISION,
        "surface_inventory_sha256": inventory["inventory_sha256"],
        "rowv1_schema_sha256": schema["contract_sha256"],
        "fixture_bank_sha256": fixtures["fixture_bank_sha256"],
        "rowv1_decoder_admission_contract_id": ROWV1_DECODER_ADMISSION_CONTRACT_ID,
        "rowv1_decoder_admission_sha256": decoder_admission["contract_sha256"],
        "rowv1_decoder_admission_fields": list(ROWV1_DECODER_ADMISSION_FIELDS),
        "cuda_receipt_surface_admission_contract_id": (
            CUDA_RECEIPT_SURFACE_ADMISSION_CONTRACT_ID
        ),
        "cuda_receipt_surface_admission_sha256": cuda_admission["contract_sha256"],
        "minimum_authoritative_cuda_abi_contract_id": (
            CUDA_ROWV1_V3_MINIMUM_ABI_CONTRACT_ID
        ),
        "minimum_authoritative_cuda_abi_sha256": minimum_cuda_abi["contract_sha256"],
        "minimum_authoritative_cuda_abi_version": CUDA_ROWV1_V3_RECEIPT_ABI_VERSION,
        "minimum_authoritative_cuda_row_width": (
            CUDA_ROWV1_V3_MINIMUM_RECEIPT_ROW_WIDTH
        ),
        "cuda_raw_v2_rowv1_projection_contract_id": (
            CUDA_RAW_V2_ROWV1_PROJECTION_CONTRACT_ID
        ),
        "cuda_raw_v2_rowv1_projection_sha256": cuda_raw_v2_projection[
            "contract_sha256"
        ],
        "cuda_rowv1_v3_kernel_mirror_projection_contract_id": (
            CUDA_ROWV1_V3_KERNEL_MIRROR_PROJECTION_CONTRACT_ID
        ),
        "cuda_rowv1_v3_kernel_mirror_projection_sha256": (
            cuda_rowv1_v3_mirror_projection["contract_sha256"]
        ),
        "cuda_rowv1_disabled_event_writer_contract_id": (
            CUDA_ROWV1_DISABLED_EVENT_WRITER_CONTRACT_ID
        ),
        "cuda_rowv1_disabled_event_writer_sha256": (
            cuda_disabled_writer["contract_sha256"]
        ),
        "cuda_rowv1_modular_device_writer_admission_contract_id": (
            CUDA_ROWV1_MODULAR_DEVICE_WRITER_ADMISSION_CONTRACT_ID
        ),
        "cuda_rowv1_modular_device_writer_admission_sha256": (
            cuda_modular_writer_admission["contract_sha256"]
        ),
        "cuda_rowv1_compact_cpu_cuda_parity_contract_id": (
            CUDA_ROWV1_COMPACT_CPU_CUDA_PARITY_CONTRACT_ID
        ),
        "cuda_rowv1_compact_cpu_cuda_parity_sha256": (
            cuda_compact_parity["contract_sha256"]
        ),
        "cuda_receipt_surface_admission_fields": (
            list(CUDA_RECEIPT_SURFACE_ADMISSION_FIELDS)
        ),
        "checkpoint_report_envelope_contract_id": CHECKPOINT_REPORT_ENVELOPE_CONTRACT_ID,
        "checkpoint_report_envelope_sha256": checkpoint_report["contract_sha256"],
        "checkpoint_report_collection_contract_id": (
            CHECKPOINT_REPORT_COLLECTION_CONTRACT_ID
        ),
        "checkpoint_report_collection_sha256": checkpoint_collection[
            "contract_sha256"
        ],
        "rowv1_negative_cases": list(ROWV1_NEGATIVE_CASES),
        "cuda_failclosed_cases": list(CUDA_FAIL_CLOSED_CASES),
        "cuda_field_decision": (
            "bounded_device_writer_abi_and_compact_parity_complete_but_route_not_package_qualified"
        ),
        "cuda_field_decision_reason": (
            "current CUDA now exposes validated RowV1 v3 int64[cells,3,64] "
            "entry tensors, a device writer for accepted and disabled RowV1 "
            "rows, and a host source-hash projection. Standalone nvcc and "
            "compute-sanitizer probes now validate the modular device writer "
            "for accepted rows, disabled-reason rows, counts, overflow, and "
            "tail clearing. Compact CPU/CUDA RowV1 ABI parity is complete for "
            "accepted and disabled fixture fields, but immutable artifact "
            "binding and final route/package qualification remain open"
        ),
        "cuda_raw_v2_projection_decision": (
            "accepted raw v2 CUDA rows may be projected to RowV1 on the host, "
            "but the CUDA surface remains unadmitted until the device ABI binds "
            "source version, disabled reason, checkpoint, package, and sanitizer "
            "contracts"
        ),
        "rowv1_implementation_authorized": True,
        "cuda_abi_implementation_authorized": True,
        "cuda_parity_authorized": True,
        "cuda_parity_scope": "compact_rowv1_abi_fixture_fields_only",
        "route_package_qualification_authorized": False,
        "successor_campaign_authorized": False,
        "checkpoint_report_schema_decision": (
            "frozen by p12-rowv1-ordinary-birth-checkpoint-report-envelope-v1 "
            "and p12-rowv1-ordinary-birth-checkpoint-report-collection-v1; "
            "route admission must bind both before execution"
        ),
        "no_claims": (
            "No CUDA route qualification, campaign, cognition, intelligence, "
            "emergence, or scientific-success claim is authorized."
        ),
    }
    return _with_sha(payload)


__all__ = [
    "CUDA_FAIL_CLOSED_CASES",
    "CUDA_RAW_V2_RECEIPT_ROW_WIDTH",
    "CUDA_RAW_V2_ROWV1_PROJECTION_CONTRACT_ID",
    "CUDA_RECEIPT_SURFACE_ADMISSION_CONTRACT_ID",
    "CUDA_RECEIPT_SURFACE_ADMISSION_FIELDS",
    "CUDA_ROWV1_COMPACT_CPU_CUDA_PARITY_CONTRACT_ID",
    "CUDA_ROWV1_V3_KERNEL_MIRROR_PROJECTION_CONTRACT_ID",
    "CUDA_ROWV1_DISABLED_EVENT_WRITER_CONTRACT_ID",
    "CUDA_ROWV1_V3_MINIMUM_ABI_CONTRACT_ID",
    "CUDA_ROWV1_V3_MINIMUM_RECEIPT_ROW_WIDTH",
    "CUDA_ROWV1_V3_RECEIPT_ABI_VERSION",
    "CUDA_ROWV1_V3_SEQUENCE_CAPACITY",
    "CHECKPOINT_REPORT_BOUNDARIES",
    "CHECKPOINT_REPORT_COLLECTION_CONTRACT_ID",
    "CHECKPOINT_REPORT_COLLECTION_FIELDS",
    "CHECKPOINT_REPORT_COLLECTION_SEGMENTS",
    "CHECKPOINT_REPORT_COLLECTION_VERSION",
    "CHECKPOINT_REPORT_ENVELOPE_CONTRACT_ID",
    "CHECKPOINT_REPORT_ENVELOPE_FIELDS",
    "CHECKPOINT_REPORT_ENVELOPE_VERSION",
    "DECISION",
    "ENTRY_GATE_CONTRACT_ID",
    "FIXTURE_BANK_CONTRACT_ID",
    "P12RowV1CudaEntryGateError",
    "ROWV1_FIELD_NAMES",
    "ROWV1_NEGATIVE_CASES",
    "ROWV1_DECODER_ADMISSION_CONTRACT_ID",
    "ROWV1_DECODER_ADMISSION_FIELDS",
    "ROWV1_PROJECTION_CONTRACT_ID",
    "ROWV1_SCHEMA_CONTRACT_ID",
    "ROWV1_SEMANTIC_SCHEMA_VERSION",
    "SCHEMA_VERSION",
    "SEMANTIC_MAPPING_CONTRACT_SHA256",
    "SURFACE_INVENTORY_CONTRACT_ID",
    "TRANSACTION_SEQUENCE_SCOPE",
    "build_rowv1_checkpoint_report_collection_v1",
    "build_rowv1_checkpoint_report_envelope_v1",
    "canonical_rowv1_checkpoint_report_collection_bytes_v1",
    "canonical_rowv1_checkpoint_report_envelope_bytes_v1",
    "canonical_rowv1_ordinary_birth_receipt_bytes_v1",
    "canonical_rowv1_cuda_entry_bytes_v1",
    "cuda_rowv1_v3_minimum_receipt_abi_contract_v1",
    "project_native_core_ordinary_birth_receipt_rowv1_v1",
    "project_cuda_raw_v2_ordinary_birth_receipt_rowv1_v1",
    "project_cuda_rowv1_v3_kernel_mirror_receipt_rowv1_v1",
    "cuda_raw_v2_rowv1_projection_receipt_v1",
    "cuda_rowv1_v3_kernel_mirror_projection_receipt_v1",
    "cuda_rowv1_disabled_event_writer_receipt_v1",
    "cuda_rowv1_modular_device_writer_admission_descriptor_v1",
    "cuda_rowv1_modular_device_writer_admission_receipt_v1",
    "cuda_rowv1_compact_cpu_cuda_parity_receipt_v1",
    "cuda_receipt_surface_admission_receipt_v1",
    "rowv1_cuda_failclosed_entry_gate_v1",
    "rowv1_cuda_surface_inventory_v1",
    "rowv1_checkpoint_report_collection_receipt_v1",
    "rowv1_checkpoint_report_collection_sha256_v1",
    "rowv1_checkpoint_report_envelope_receipt_v1",
    "rowv1_checkpoint_report_envelope_sha256_v1",
    "rowv1_fixture_bank_v1",
    "rowv1_decoder_admission_receipt_v1",
    "rowv1_ordinary_birth_receipt_sha256_v1",
    "rowv1_ordinary_birth_projection_receipt_v1",
    "rowv1_semantic_receipt_schema_v1",
    "rowv1_source_receipt_sha256_v1",
    "validate_rowv1_checkpoint_report_collection_v1",
    "validate_rowv1_checkpoint_report_envelope_v1",
    "validate_cuda_receipt_surface_admission_v1",
    "validate_rowv1_decoder_admission_v1",
    "validate_rowv1_ordinary_birth_receipt_v1",
]
