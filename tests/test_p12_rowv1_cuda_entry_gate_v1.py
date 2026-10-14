from __future__ import annotations

import json
from pathlib import Path
import struct

import pytest

from engine.evolution.p12_rowv1_cuda_entry_gate_v1 import (
    CHECKPOINT_REPORT_BOUNDARIES,
    CHECKPOINT_REPORT_COLLECTION_CONTRACT_ID,
    CHECKPOINT_REPORT_COLLECTION_FIELDS,
    CHECKPOINT_REPORT_COLLECTION_SEGMENTS,
    CHECKPOINT_REPORT_COLLECTION_VERSION,
    CHECKPOINT_REPORT_ENVELOPE_CONTRACT_ID,
    CHECKPOINT_REPORT_ENVELOPE_FIELDS,
    CHECKPOINT_REPORT_ENVELOPE_VERSION,
    CUDA_FAIL_CLOSED_CASES,
    CUDA_RAW_V2_RECEIPT_ROW_WIDTH,
    CUDA_RAW_V2_ROWV1_PROJECTION_CONTRACT_ID,
    CUDA_RECEIPT_SURFACE_ADMISSION_CONTRACT_ID,
    CUDA_RECEIPT_SURFACE_ADMISSION_FIELDS,
    CUDA_ROWV1_COMPACT_CPU_CUDA_PARITY_CONTRACT_ID,
    CUDA_ROWV1_DISABLED_EVENT_WRITER_CONTRACT_ID,
    CUDA_ROWV1_MODULAR_DEVICE_WRITER_ADMISSION_CONTRACT_ID,
    CUDA_ROWV1_V3_MINIMUM_RECEIPT_ROW_WIDTH,
    CUDA_ROWV1_V3_KERNEL_MIRROR_PROJECTION_CONTRACT_ID,
    CUDA_ROWV1_V3_RECEIPT_ABI_VERSION,
    CUDA_ROWV1_V3_SEQUENCE_CAPACITY,
    DECISION,
    ENTRY_GATE_CONTRACT_ID,
    FIXTURE_BANK_CONTRACT_ID,
    P12RowV1CudaEntryGateError,
    ROWV1_DECODER_ADMISSION_CONTRACT_ID,
    ROWV1_DECODER_ADMISSION_FIELDS,
    ROWV1_FIELD_NAMES,
    ROWV1_NEGATIVE_CASES,
    ROWV1_PROJECTION_CONTRACT_ID,
    ROWV1_SCHEMA_CONTRACT_ID,
    ROWV1_SEMANTIC_SCHEMA_VERSION,
    SEMANTIC_MAPPING_CONTRACT_SHA256,
    SURFACE_INVENTORY_CONTRACT_ID,
    TRANSACTION_SEQUENCE_SCOPE,
    TRANSACTION_MUTATION_DISABLED,
    build_rowv1_checkpoint_report_collection_v1,
    build_rowv1_checkpoint_report_envelope_v1,
    canonical_rowv1_checkpoint_report_collection_bytes_v1,
    canonical_rowv1_checkpoint_report_envelope_bytes_v1,
    canonical_rowv1_ordinary_birth_receipt_bytes_v1,
    canonical_rowv1_cuda_entry_bytes_v1,
    cuda_rowv1_v3_minimum_receipt_abi_contract_v1,
    cuda_rowv1_v3_kernel_mirror_projection_receipt_v1,
    cuda_rowv1_compact_cpu_cuda_parity_receipt_v1,
    cuda_rowv1_disabled_event_writer_receipt_v1,
    cuda_rowv1_modular_device_writer_admission_descriptor_v1,
    cuda_rowv1_modular_device_writer_admission_receipt_v1,
    cuda_raw_v2_rowv1_projection_receipt_v1,
    cuda_receipt_surface_admission_receipt_v1,
    project_cuda_rowv1_v3_kernel_mirror_receipt_rowv1_v1,
    project_cuda_raw_v2_ordinary_birth_receipt_rowv1_v1,
    project_native_core_ordinary_birth_receipt_rowv1_v1,
    rowv1_cuda_failclosed_entry_gate_v1,
    rowv1_cuda_surface_inventory_v1,
    rowv1_checkpoint_report_collection_receipt_v1,
    rowv1_checkpoint_report_collection_sha256_v1,
    rowv1_checkpoint_report_envelope_receipt_v1,
    rowv1_checkpoint_report_envelope_sha256_v1,
    rowv1_decoder_admission_receipt_v1,
    rowv1_fixture_bank_v1,
    rowv1_ordinary_birth_receipt_sha256_v1,
    rowv1_ordinary_birth_projection_receipt_v1,
    rowv1_semantic_receipt_schema_v1,
    rowv1_source_receipt_sha256_v1,
    validate_cuda_receipt_surface_admission_v1,
    validate_rowv1_checkpoint_report_collection_v1,
    validate_rowv1_checkpoint_report_envelope_v1,
    validate_rowv1_decoder_admission_v1,
    validate_rowv1_ordinary_birth_receipt_v1,
)


ROOT = Path(__file__).resolve().parents[1]
SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64
SHA_E = "e" * 64
SHA_F = "f" * 64
CUDA_RAW_V2_ROW = [
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
    1,
    1,
    0,
    0,
    0,
    5,
    1,
]


def _cuda_rowv1_v3_mirror_row() -> list[int]:
    row = [0] * CUDA_ROWV1_V3_MINIMUM_RECEIPT_ROW_WIDTH
    row[0] = CUDA_ROWV1_V3_RECEIPT_ABI_VERSION
    row[1] = 2
    row[2:22] = CUDA_RAW_V2_ROW[1:21]
    return row


def _float64_bits(value: float) -> int:
    return int.from_bytes(struct.pack(">d", value), "big")


def _native_receipt(**overrides: object) -> dict[str, object]:
    receipt: dict[str, object] = {
        "transaction_status": 6,
        "native_status": 0,
        "operation": 2,
        "selected_slot": -1,
        "changed_dimension": 0,
        "changed": 0,
        "committed": 0,
        "capacity_available": 1,
        "publish_success": 1,
        "parent_agent_id": (1 << 40) + 100,
        "parent_slot": 2,
        "parent_slot_token": 12,
        "child_agent_id": (1 << 40) + 200,
        "child_slot": 4,
        "expected_child_slot_token": 17,
        "child_publication_token": 18,
        "draw0_float64_bits": _float64_bits(0.625),
        "draw1_float64_bits": _float64_bits(0.5),
        "parent_digest_before": 11,
        "child_digest_before": 22,
        "proposal_digest": 0,
        "child_digest_after": 22,
        "selected_module_instance_id": 0,
        "selected_homology_id": 0,
        "value_before_float32": 0.0,
        "value_after_float32": 0.0,
        "disabled_reason_code": 2,
    }
    receipt.update(overrides)
    return receipt


def _project(
    native_receipt: dict[str, object],
    *,
    source_receipt_abi_version: int = 4,
    **overrides: object,
) -> dict[str, object]:
    kwargs: dict[str, object] = {
        "source_receipt_abi_version": source_receipt_abi_version,
        "transaction_id": 1,
        "cell": 0,
        "generation": 1,
        "tick": 7,
        "rng_stream_id": "ordinary_birth_mutation_v1",
        "rng_index_before": 40,
        "rng_index_after": 42,
        "allocator_before": {"next_agent_id": (1 << 40) + 200},
        "allocator_after": {"next_agent_id": (1 << 40) + 200},
        "package_binding_sha256": SHA_A,
        "schema_binding_sha256": SHA_B,
        "mechanics_binding_sha256": SHA_C,
    }
    kwargs.update(overrides)
    return project_native_core_ordinary_birth_receipt_rowv1_v1(
        native_receipt,
        **kwargs,
    )


def _envelope(
    row: dict[str, object],
    *,
    boundary: str = "checkpoint_export",
    reporting_enabled: bool = True,
    **overrides: object,
) -> dict[str, object]:
    kwargs: dict[str, object] = {
        "boundary": boundary,
        "reporting_enabled": reporting_enabled,
    }
    kwargs.update(overrides)
    return build_rowv1_checkpoint_report_envelope_v1(row, **kwargs)


def _collection(
    envelopes: list[dict[str, object]],
    *,
    expected_receipt_count: int | None = None,
    expected_cells: list[int] | None = None,
    segment_kind: str = "uninterrupted",
    boundary: str = "checkpoint_export",
    **overrides: object,
) -> dict[str, object]:
    cells = [0] if expected_cells is None else expected_cells
    actual_counts = {cell: 0 for cell in cells}
    for envelope in envelopes:
        cell = int(envelope["checkpoint_cell"])
        actual_counts[cell] = actual_counts.get(cell, 0) + 1
    kwargs: dict[str, object] = {
        "run_id": "rowv1-collection-test-run",
        "source_commit_sha": "f69180d78",
        "route_binding_sha256": SHA_C,
        "package_binding_sha256": SHA_A,
        "boundary": boundary,
        "segment_kind": segment_kind,
        "checkpoint_lineage_id": "lineage-a",
        "expected_receipt_count": (
            len(envelopes)
            if expected_receipt_count is None
            else expected_receipt_count
        ),
        "expected_cells": cells,
        "lifecycle_ledger_sha256": SHA_B,
        "per_cell_expected_receipt_counts": actual_counts,
        "checkpoint_state_sha256": SHA_C,
        "rng_state_sha256": SHA_D,
        "allocator_state_sha256": SHA_E,
        "identity_table_sha256": SHA_F,
        "owner_lineage_sha256": SHA_A,
    }
    kwargs.update(overrides)
    return build_rowv1_checkpoint_report_collection_v1(envelopes, **kwargs)


def _surfaces() -> dict[str, dict]:
    inventory = rowv1_cuda_surface_inventory_v1(ROOT)
    return {row["surface_id"]: row for row in inventory["surfaces"]}


def _decoder_contract(**overrides: object) -> dict[str, object]:
    contract: dict[str, object] = {
        "decoder_contract_id": ROWV1_DECODER_ADMISSION_CONTRACT_ID,
        "decoder_schema_version": "rowv1-decoder-test-v1",
        "supported_source_receipt_abi_versions": [4],
        "decoder_row_width": len(ROWV1_FIELD_NAMES),
        "expected_row_width": len(ROWV1_FIELD_NAMES),
        "allow_mixed_source_abi_versions": False,
    }
    contract.update(overrides)
    return contract


def _cuda_admission(**overrides: object) -> dict[str, object]:
    minimum_abi = cuda_rowv1_v3_minimum_receipt_abi_contract_v1()
    row_width = CUDA_ROWV1_V3_MINIMUM_RECEIPT_ROW_WIDTH
    admission: dict[str, object] = {
        "admission_contract_id": CUDA_RECEIPT_SURFACE_ADMISSION_CONTRACT_ID,
        "authoritative_cuda_receipt_abi_present": True,
        "declared_cuda_receipt_abi_version": CUDA_ROWV1_V3_RECEIPT_ABI_VERSION,
        "device_layout_preserves_rowv1_semantics": True,
        "receipt_surface_present": True,
        "receipt_surface_row_count": 1,
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
    admission.update(overrides)
    return admission


def test_surface_inventory_classifies_cuda_rowv1_surface_as_entry_only() -> None:
    inventory = rowv1_cuda_surface_inventory_v1(ROOT)
    surfaces = {row["surface_id"]: row for row in inventory["surfaces"]}
    cuda_tensor = surfaces["cuda_modular_ordinary_birth_mutation_receipt_tensor"]
    cuda_rowv1 = surfaces["cuda_modular_ordinary_birth_rowv1_v3_receipt_tensor"]
    cuda_binding = surfaces["cuda_modular_receipt_binding"]
    cuda_context = surfaces["cuda_modular_receipt_device_context"]

    assert inventory["contract_id"] == SURFACE_INVENTORY_CONTRACT_ID
    assert inventory["decision"] == DECISION
    assert inventory["surface_count"] >= 7
    assert {
        "producer",
        "consumer",
        "authority_kind",
        "persistence_mode",
        "package_route_binding",
    } <= set(inventory["required_surface_fields"])
    assert inventory["cuda_receipt_abi_present"] is False
    assert inventory["cuda_rowv1_v3_kernel_mirror_present"] is True
    assert inventory["cuda_rowv1_v3_kernel_mirror_projection_present"] is True
    assert inventory["cuda_disabled_reason_fits_without_layout_change"] is False
    assert inventory["cuda_layout_change_authorized"] is False
    assert cuda_tensor["dtype"] == "torch.int64"
    assert cuda_tensor["width"] == "int64[cells,3,21]"
    assert cuda_tensor["disabled_reason_support"] == "absent"
    assert cuda_tensor["source_abi_version_support"] == "absent"
    assert cuda_rowv1["width"] == "int64[cells,3,64]"
    assert cuda_rowv1["authority"] == "kernel_mirror_present_not_authoritative"
    assert cuda_rowv1["package_route_binding"] == (
        "absent until artifact-bound ABI qualification"
    )
    cuda_mirror = surfaces["cuda_modular_ordinary_birth_rowv1_v3_kernel_mirror"]
    assert cuda_mirror["authority_kind"] == "cuda_rowv1_v3_kernel_mirror"
    assert cuda_mirror["source_abi_version_support"] == (
        "rowv1_v3_source_hash_host_projection_bound"
    )
    assert cuda_mirror["package_route_binding"] == (
        "absent until immutable CUDA artifact rebuild"
    )
    assert "int32 overflow" in cuda_binding["dtype"]
    assert "64" in cuda_context["width"]
    assert all(row["source_evidence"]["line"] > 0 for row in surfaces.values())


def test_inventory_binds_native_v4_and_checkpoint_projection_surfaces() -> None:
    surfaces = _surfaces()

    native = surfaces["native_core_receipt_abi_v4"]
    checkpoint = surfaces["checkpoint_projection_mutation_receipt"]
    bridge = surfaces["native_to_canonical_bridge"]

    assert native["version"] == "4"
    assert native["disabled_reason_support"] == "raw_int32_code_0_to_6"
    assert native["source_abi_version_support"] == "exported_layout_version_v4"
    assert checkpoint["persisted"] is True
    assert "not_frozen_before_this_gate" in checkpoint["source_abi_version_support"]
    assert bridge["required_action"] == (
        "fail closed for unknown reason and stale mapping hash"
    )


def test_rowv1_schema_requires_source_abi_hash_and_raw_reason_code() -> None:
    schema = rowv1_semantic_receipt_schema_v1()
    fields = {row["name"]: row for row in schema["field_order"]}

    assert schema["contract_id"] == ROWV1_SCHEMA_CONTRACT_ID
    assert fields["source_receipt_abi_version"]["dtype"] == "int32"
    assert fields["source_receipt_sha256"]["dtype"] == "sha256"
    assert fields["semantic_mapping_contract_sha256"]["dtype"] == "sha256"
    assert fields["raw_disabled_reason_code"]["dtype"] == "int32"
    assert fields["parent_agent_id"]["dtype"] == "int64"
    assert fields["child_agent_id"]["dtype"] == "int64"
    assert fields["rng_index_before"]["dtype"] == "uint64"
    assert fields["rng_index_after"]["dtype"] == "uint64"
    assert schema["disabled_reason_policy"]["unknown_code"] == "reject"
    assert "stable biological IDs are int64" in schema["dense_id_boundary"]
    assert schema["contract_sha256"]


def test_fixture_bank_covers_required_positive_and_negative_cases() -> None:
    bank = rowv1_fixture_bank_v1()
    case_ids = {row["case_id"] for row in bank["fixtures"]}

    assert bank["contract_id"] == FIXTURE_BANK_CONTRACT_ID
    assert bank["fixture_count"] == 17
    assert {
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
    } == case_ids
    assert all(row["canonical_bytes_sha256"] for row in bank["fixtures"])
    assert bank["fixture_bank_sha256"]


def test_entry_gate_freezes_failclosed_rowv1_and_cuda_admission() -> None:
    gate = rowv1_cuda_failclosed_entry_gate_v1(source_root=ROOT)

    assert gate["contract_id"] == ENTRY_GATE_CONTRACT_ID
    assert gate["decision"] == DECISION
    assert set(gate["rowv1_negative_cases"]) == set(ROWV1_NEGATIVE_CASES)
    assert set(gate["cuda_failclosed_cases"]) == set(CUDA_FAIL_CLOSED_CASES)
    assert gate["rowv1_decoder_admission_contract_id"] == (
        ROWV1_DECODER_ADMISSION_CONTRACT_ID
    )
    assert gate["rowv1_decoder_admission_sha256"] == (
        rowv1_decoder_admission_receipt_v1()["contract_sha256"]
    )
    assert tuple(gate["rowv1_decoder_admission_fields"]) == (
        ROWV1_DECODER_ADMISSION_FIELDS
    )
    assert gate["cuda_receipt_surface_admission_contract_id"] == (
        CUDA_RECEIPT_SURFACE_ADMISSION_CONTRACT_ID
    )
    assert gate["cuda_receipt_surface_admission_sha256"] == (
        cuda_receipt_surface_admission_receipt_v1(source_root=ROOT)["contract_sha256"]
    )
    assert gate["cuda_raw_v2_rowv1_projection_contract_id"] == (
        CUDA_RAW_V2_ROWV1_PROJECTION_CONTRACT_ID
    )
    assert gate["cuda_raw_v2_rowv1_projection_sha256"] == (
        cuda_raw_v2_rowv1_projection_receipt_v1()["contract_sha256"]
    )
    assert gate["cuda_rowv1_v3_kernel_mirror_projection_contract_id"] == (
        CUDA_ROWV1_V3_KERNEL_MIRROR_PROJECTION_CONTRACT_ID
    )
    assert gate["cuda_rowv1_v3_kernel_mirror_projection_sha256"] == (
        cuda_rowv1_v3_kernel_mirror_projection_receipt_v1()["contract_sha256"]
    )
    assert gate["cuda_rowv1_disabled_event_writer_contract_id"] == (
        CUDA_ROWV1_DISABLED_EVENT_WRITER_CONTRACT_ID
    )
    assert gate["cuda_rowv1_disabled_event_writer_sha256"] == (
        cuda_rowv1_disabled_event_writer_receipt_v1(source_root=ROOT)[
            "contract_sha256"
        ]
    )
    assert gate["cuda_rowv1_modular_device_writer_admission_contract_id"] == (
        CUDA_ROWV1_MODULAR_DEVICE_WRITER_ADMISSION_CONTRACT_ID
    )
    assert gate["cuda_rowv1_modular_device_writer_admission_sha256"] == (
        cuda_rowv1_modular_device_writer_admission_receipt_v1(source_root=ROOT)[
            "contract_sha256"
        ]
    )
    assert gate["cuda_rowv1_compact_cpu_cuda_parity_contract_id"] == (
        CUDA_ROWV1_COMPACT_CPU_CUDA_PARITY_CONTRACT_ID
    )
    assert gate["cuda_rowv1_compact_cpu_cuda_parity_sha256"] == (
        cuda_rowv1_compact_cpu_cuda_parity_receipt_v1(source_root=ROOT)[
            "contract_sha256"
        ]
    )
    assert tuple(gate["cuda_receipt_surface_admission_fields"]) == (
        CUDA_RECEIPT_SURFACE_ADMISSION_FIELDS
    )
    assert gate["checkpoint_report_envelope_contract_id"] == (
        CHECKPOINT_REPORT_ENVELOPE_CONTRACT_ID
    )
    assert gate["checkpoint_report_envelope_sha256"]
    assert gate["checkpoint_report_collection_contract_id"] == (
        CHECKPOINT_REPORT_COLLECTION_CONTRACT_ID
    )
    assert gate["checkpoint_report_collection_sha256"]
    assert gate["cuda_field_decision"] == (
        "bounded_device_writer_abi_and_compact_parity_complete_but_route_not_package_qualified"
    )
    assert "int64[cells,3,64]" in gate["cuda_field_decision_reason"]
    assert "compute-sanitizer probes" in gate["cuda_field_decision_reason"]
    assert "Compact CPU/CUDA RowV1 ABI parity is complete" in (
        gate["cuda_field_decision_reason"]
    )
    assert "accepted raw v2 CUDA rows may be projected to RowV1" in gate[
        "cuda_raw_v2_projection_decision"
    ]
    assert gate["rowv1_implementation_authorized"] is True
    assert gate["cuda_abi_implementation_authorized"] is True
    assert gate["cuda_parity_authorized"] is True
    assert gate["cuda_parity_scope"] == "compact_rowv1_abi_fixture_fields_only"
    assert gate["route_package_qualification_authorized"] is False
    assert gate["successor_campaign_authorized"] is False
    assert "frozen by" in gate["checkpoint_report_schema_decision"]
    assert "No CUDA route qualification" in gate["no_claims"]
    assert gate["contract_sha256"]


def test_cuda_disabled_event_writer_receipt_binds_sanitizer_probe() -> None:
    receipt = cuda_rowv1_disabled_event_writer_receipt_v1(source_root=ROOT)

    assert receipt["contract_id"] == CUDA_ROWV1_DISABLED_EVENT_WRITER_CONTRACT_ID
    assert receipt["device_disabled_event_writer_present"] is True
    assert receipt["standalone_nvcc_runtime_probe_passed"] is True
    assert receipt["compute_sanitizer_tool"] == "memcheck"
    assert receipt["compute_sanitizer_error_summary"] == "0_errors"
    assert tuple(receipt["sanitizer_bounds_cases"]) == (
        "receipt_rows_bounds",
        "receipt_counts_bounds",
        "receipt_overflow_bounds",
    )
    assert any(
        "compute-sanitizer --tool memcheck" in command
        for command in receipt["validation_commands"]
    )
    assert receipt["route_package_qualification_authorized"] is False
    assert receipt["successor_campaign_authorized"] is False


def test_entry_gate_serialization_is_canonical_ascii() -> None:
    gate = rowv1_cuda_failclosed_entry_gate_v1(source_root=ROOT)
    wire = canonical_rowv1_cuda_entry_bytes_v1(gate)

    assert wire.endswith(b"\n")
    decoded = json.loads(wire.decode("ascii"))
    assert decoded == gate
    assert canonical_rowv1_cuda_entry_bytes_v1(decoded) == wire


def test_rowv1_decoder_admission_accepts_homogeneous_v4_rows() -> None:
    row = _project(_native_receipt(disabled_reason_code=2))

    receipt = validate_rowv1_decoder_admission_v1(
        [row],
        decoder_contract=_decoder_contract(),
    )

    assert receipt["contract_id"] == ROWV1_DECODER_ADMISSION_CONTRACT_ID
    assert receipt["source_receipt_abi_versions"] == [4]
    assert receipt["receipt_count"] == 1
    assert receipt["contract_sha256"]


def test_rowv1_decoder_admission_receipt_records_prompt_negative_cases() -> None:
    receipt = rowv1_decoder_admission_receipt_v1()

    assert receipt["contract_id"] == ROWV1_DECODER_ADMISSION_CONTRACT_ID
    assert "v4_row_presented_to_v3_only_decoder_rejected" in receipt["covered_cases"]
    assert "mixed_v3_v4_batch_rejected" in receipt["covered_cases"]
    assert "zero_length_or_absent_receipt_surface_rejected" in receipt["covered_cases"]
    assert receipt["cuda_abi_implementation_authorized"] is False
    assert receipt["successor_campaign_authorized"] is False
    assert receipt["contract_sha256"]


@pytest.mark.parametrize(
    ("decoder_overrides", "rows", "match"),
    (
        (
            {"supported_source_receipt_abi_versions": [3]},
            lambda: [_project(_native_receipt(disabled_reason_code=2))],
            "decoder does not support source receipt ABI",
        ),
        (
            {"decoder_row_width": len(ROWV1_FIELD_NAMES) - 1},
            lambda: [_project(_native_receipt(disabled_reason_code=2))],
            "stale decoder width",
        ),
        (
            {"supported_source_receipt_abi_versions": [3, 4]},
            lambda: [
                _project(_native_receipt(disabled_reason_code=2)),
                _project(
                    {k: v for k, v in _native_receipt().items() if k != "disabled_reason_code"},
                    source_receipt_abi_version=3,
                    transaction_id=2,
                ),
            ],
            "mixed source receipt ABI",
        ),
        (
            {},
            lambda: [],
            "zero-length or absent receipt surface",
        ),
    ),
)
def test_rowv1_decoder_admission_fails_closed_for_stale_or_mixed_surfaces(
    decoder_overrides: dict[str, object],
    rows,
    match: str,
) -> None:
    with pytest.raises(P12RowV1CudaEntryGateError, match=match):
        validate_rowv1_decoder_admission_v1(
            rows(),
            decoder_contract=_decoder_contract(**decoder_overrides),
        )


def test_cuda_receipt_surface_admission_accepts_only_complete_future_abi() -> None:
    receipt = validate_cuda_receipt_surface_admission_v1(_cuda_admission())

    assert receipt["contract_id"] == CUDA_RECEIPT_SURFACE_ADMISSION_CONTRACT_ID
    assert receipt["cuda_receipt_abi_version"] == CUDA_ROWV1_V3_RECEIPT_ABI_VERSION
    assert receipt["cuda_receipt_abi_contract_sha256"] == (
        cuda_rowv1_v3_minimum_receipt_abi_contract_v1()["contract_sha256"]
    )
    assert receipt["cuda_receipt_surface_admitted"] is True
    assert receipt["contract_sha256"]


def test_cuda_receipt_surface_admission_receipt_records_failclosed_boundary() -> None:
    receipt = cuda_receipt_surface_admission_receipt_v1()
    writer = cuda_rowv1_disabled_event_writer_receipt_v1(source_root=ROOT)
    admitted = cuda_rowv1_modular_device_writer_admission_receipt_v1(
        source_root=ROOT
    )

    assert receipt["contract_id"] == CUDA_RECEIPT_SURFACE_ADMISSION_CONTRACT_ID
    assert receipt["current_cuda_receipt_abi_present"] is True
    assert receipt["current_cuda_rowv1_v3_entry_surface_present"] is True
    assert receipt["current_cuda_rowv1_v3_kernel_mirror_present"] is True
    assert receipt["current_cuda_rowv1_v3_kernel_mirror_projection_present"] is True
    assert receipt["current_cuda_rowv1_v3_entry_surface_width"] == (
        CUDA_ROWV1_V3_MINIMUM_RECEIPT_ROW_WIDTH
    )
    assert receipt["current_cuda_rowv1_v3_entry_surface_capacity"] == (
        CUDA_ROWV1_V3_SEQUENCE_CAPACITY
    )
    assert receipt["current_cuda_field_decision"] == (
        "bounded_device_writer_abi_and_compact_parity_complete_but_route_not_package_qualified"
    )
    assert receipt["current_cuda_modular_device_writer_admission_contract_id"] == (
        admitted["contract_id"]
    )
    assert receipt["current_cuda_modular_device_writer_admission_sha256"] == (
        admitted["contract_sha256"]
    )
    compact_parity = cuda_rowv1_compact_cpu_cuda_parity_receipt_v1(source_root=ROOT)
    assert receipt["current_cuda_compact_cpu_cuda_parity_contract_id"] == (
        compact_parity["contract_id"]
    )
    assert receipt["current_cuda_compact_cpu_cuda_parity_sha256"] == (
        compact_parity["contract_sha256"]
    )
    assert receipt["current_cuda_disabled_event_writer_contract_id"] == (
        CUDA_ROWV1_DISABLED_EVENT_WRITER_CONTRACT_ID
    )
    assert receipt["current_cuda_disabled_event_writer_sha256"] == (
        writer["contract_sha256"]
    )
    assert receipt["current_cuda_disabled_event_writer_sanitizer"] == (
        "memcheck_0_errors"
    )
    assert set(CUDA_FAIL_CLOSED_CASES) <= set(receipt["covered_failclosed_cases"])
    assert "wrong_checkpoint_manifest_receipt_width" in receipt[
        "covered_failclosed_cases"
    ]
    assert receipt["cuda_receipt_surface_currently_admitted"] is True
    assert receipt["cuda_abi_implementation_authorized"] is True
    assert receipt["compact_cpu_cuda_parity_complete"] is True
    assert receipt["cuda_parity_authorized"] is True
    assert receipt["route_package_qualification_authorized"] is False
    assert receipt["successor_campaign_authorized"] is False
    assert receipt["contract_sha256"]


def test_cuda_modular_device_writer_admission_receipt_is_positive_but_bounded() -> None:
    descriptor = cuda_rowv1_modular_device_writer_admission_descriptor_v1()
    admission = validate_cuda_receipt_surface_admission_v1(descriptor)
    receipt = cuda_rowv1_modular_device_writer_admission_receipt_v1(
        source_root=ROOT
    )

    assert receipt["contract_id"] == (
        CUDA_ROWV1_MODULAR_DEVICE_WRITER_ADMISSION_CONTRACT_ID
    )
    assert receipt["decision"] == (
        "P12-CUDA-ROWV1-MODULAR-DEVICE-WRITER-ABI-ADMITTED"
    )
    assert receipt["cuda_receipt_surface_admission_sha256"] == (
        admission["contract_sha256"]
    )
    assert receipt["cuda_receipt_surface_admitted"] is True
    assert receipt["admitted_fields"]["raw_disabled_reason_code"] == "kernel_written"
    assert receipt["admitted_fields"]["count_overflow_and_dropped_row_state"] == (
        "kernel_written_count_and_overflow"
    )
    assert receipt["compact_cpu_cuda_parity_complete"] is False
    assert receipt["route_package_qualification_authorized"] is False
    assert receipt["successor_campaign_authorized"] is False
    assert all(row["line"] > 0 for row in receipt["source_evidence"])


def test_cuda_rowv1_compact_cpu_cuda_parity_receipt_is_positive_but_bounded() -> None:
    receipt = cuda_rowv1_compact_cpu_cuda_parity_receipt_v1(source_root=ROOT)

    assert receipt["contract_id"] == CUDA_ROWV1_COMPACT_CPU_CUDA_PARITY_CONTRACT_ID
    assert receipt["decision"] == "P12-CUDA-ROWV1-COMPACT-CPU-CUDA-PARITY-A"
    assert receipt["cuda_rowv1_modular_device_writer_admission_contract_id"] == (
        CUDA_ROWV1_MODULAR_DEVICE_WRITER_ADMISSION_CONTRACT_ID
    )
    assert receipt["compact_cpu_cuda_parity_complete"] is True
    assert receipt["route_package_qualification_authorized"] is False
    assert receipt["successor_campaign_authorized"] is False
    cases = {case["case_id"]: case for case in receipt["covered_cases"]}
    assert set(cases) == {
        "accepted_refine_descendant_parameter_reset_bias",
        "disabled_structural_mutation_reason",
    }
    for case in cases.values():
        assert case["mismatches"] == ()
        assert "source_receipt_sha256" in case["source_specific_fields"]
        assert case["native_rowv1_sha256"] != case["cuda_rowv1_sha256"]
    assert cases["disabled_structural_mutation_reason"]["native_reason"] == (
        "structural_mutation_disabled"
    )
    assert cases["disabled_structural_mutation_reason"]["cuda_reason"] == (
        "structural_mutation_disabled"
    )
    assert all(row["line"] > 0 for row in receipt["source_evidence"])


@pytest.mark.parametrize(
    ("overrides", "match"),
    (
        ({"authoritative_cuda_receipt_abi_present": False}, "no authoritative CUDA"),
        ({"declared_cuda_receipt_abi_version": None}, "undeclared"),
        (
            {"device_layout_preserves_rowv1_semantics": False},
            "device layout cannot preserve",
        ),
        ({"receipt_surface_present": False}, "zero-length or absent"),
        ({"receipt_surface_row_count": 0}, "zero-length or absent"),
        ({"device_receipt_size_bytes": 184}, "size mismatch"),
        ({"device_receipt_alignment_bytes": 4}, "alignment mismatch"),
        ({"decoder_row_width": 21}, "stale decoder width"),
        ({"checkpoint_manifest_receipt_width": 21}, "manifest receipt width"),
        ({"checkpoint_manifest_receipt_abi_version": 2}, "manifest receipt ABI"),
        ({"package_receipt_abi_sha256": SHA_A}, "package does not bind"),
        (
            {"checkpoint_manifest_preserves_source_abi": False},
            "omits source receipt ABI",
        ),
        (
            {"checkpoint_manifest_preserves_disabled_reason": False},
            "omits disabled reason",
        ),
        ({"route_rowv1_mapping_contract_sha256": SHA_B}, "RowV1 mapping contract"),
        ({"sanitizer_bounds_cases": []}, "sanitizer coverage"),
        ({"dense_id_authority": "agent_id_to_slot_shadow"}, "dense-ID dependence"),
    ),
)
def test_cuda_receipt_surface_admission_fails_closed_before_execution(
    overrides: dict[str, object],
    match: str,
) -> None:
    with pytest.raises(P12RowV1CudaEntryGateError, match=match):
        validate_cuda_receipt_surface_admission_v1(_cuda_admission(**overrides))


def test_inventory_fails_closed_when_source_marker_is_missing(tmp_path: Path) -> None:
    source = tmp_path / "source"
    for row in rowv1_cuda_surface_inventory_v1(ROOT)["surfaces"]:
        path = source / row["source_evidence"]["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("marker missing\n", encoding="utf-8")

    with pytest.raises(P12RowV1CudaEntryGateError, match="missing source marker"):
        rowv1_cuda_surface_inventory_v1(source)


def test_rowv1_projection_preserves_v4_disabled_reason_and_source_hash() -> None:
    native_receipt = _native_receipt(disabled_reason_code=2)
    row = _project(native_receipt)

    assert tuple(row) == ROWV1_FIELD_NAMES
    assert row["semantic_schema_version"] == ROWV1_SEMANTIC_SCHEMA_VERSION
    assert row["source_receipt_abi_version"] == 4
    assert row["source_receipt_sha256"] == rowv1_source_receipt_sha256_v1(
        native_receipt,
        source_receipt_abi_version=4,
    )
    assert row["authoritative_transaction_sequence"] == 1
    assert row["transaction_sequence_scope"] == TRANSACTION_SEQUENCE_SCOPE
    assert row["semantic_mapping_contract_sha256"] == SEMANTIC_MAPPING_CONTRACT_SHA256
    assert row["raw_disabled_reason_code"] == 2
    assert row["canonical_status"] == "disabled"
    assert row["canonical_reason"] == "structural_mutation_disabled"
    assert row["proposal_digest"] is None
    assert row["selected_identity"] is None
    assert row["changed"] is False
    assert row["allocator_before"] == row["allocator_after"]
    assert canonical_rowv1_ordinary_birth_receipt_bytes_v1(row).endswith(b"\n")
    assert rowv1_ordinary_birth_receipt_sha256_v1(row)


def test_rowv1_projection_defaults_v3_disabled_operator_without_raw_reason() -> None:
    native_receipt = _native_receipt()
    native_receipt.pop("disabled_reason_code")

    row = _project(native_receipt, source_receipt_abi_version=3)

    assert row["source_receipt_abi_version"] == 3
    assert row["raw_disabled_reason_code"] is None
    assert row["canonical_status"] == "disabled"
    assert row["canonical_reason"] == "operator_disabled"


def test_rowv1_projection_preserves_accepted_changed_identity_and_values() -> None:
    native_receipt = _native_receipt(
        transaction_status=0,
        selected_slot=0,
        changed_dimension=1 << 8,
        changed=1,
        committed=1,
        proposal_digest=44,
        child_digest_after=44,
        selected_module_instance_id=1001,
        selected_homology_id=2001,
        value_before_float32=0.25,
        value_after_float32=0.5,
        disabled_reason_code=0,
    )

    row = _project(
        native_receipt,
        selected_identity={"module_instance_id": "mi-1001", "homology_id": "h-2001"},
        selected_instance_id_index=0,
        selected_homology_id_index=1,
        proposal_digest_sha256=SHA_B,
        allocator_after={"next_agent_id": (1 << 40) + 201},
        authoritative_transaction_sequence=8,
    )

    assert row["canonical_status"] == "accepted"
    assert row["authoritative_transaction_sequence"] == 8
    assert row["canonical_reason"] == "native_core_committed"
    assert row["raw_disabled_reason_code"] == 0
    assert row["proposal_digest"] == SHA_B
    assert row["selected_identity"] == {
        "module_instance_id": "mi-1001",
        "homology_id": "h-2001",
    }
    assert row["changed"] is True
    assert row["value_before"] == {"float32": 0.25}
    assert row["value_after"] == {"float32": 0.5}
    assert row["publication_decision"] == "published"
    assert row["rollback_decision"] == "committed"


def test_rowv1_projection_preserves_publication_rejection_proposal_digest() -> None:
    native_receipt = _native_receipt(
        transaction_status=2,
        proposal_digest=55,
        disabled_reason_code=0,
    )

    row = _project(native_receipt, proposal_digest_sha256=SHA_C)

    assert row["canonical_status"] == "rejected"
    assert row["canonical_reason"] == "child_publication_failed"
    assert row["proposal_digest"] == SHA_C
    assert row["selected_identity"] is None
    assert row["changed"] is False
    assert row["publication_decision"] == "rejected"


def test_cuda_raw_v2_projection_preserves_accepted_rowv1_identity_and_source() -> None:
    row = project_cuda_raw_v2_ordinary_birth_receipt_rowv1_v1(
        CUDA_RAW_V2_ROW,
        transaction_id=9,
        cell=0,
        rng_stream_id="cuda_ordinary_birth_mutation_v2",
        rng_index_before=70,
        rng_index_after=72,
        allocator_before={"next_agent_id": (1 << 40) + 20},
        allocator_after={"next_agent_id": (1 << 40) + 21},
        package_binding_sha256=SHA_A,
        schema_binding_sha256=SHA_B,
        mechanics_binding_sha256=SHA_C,
        selected_identity={"module_instance_id_index": 0, "homology_id_index": 0},
    )

    assert row["source_receipt_abi_version"] == 2
    assert row["raw_disabled_reason_code"] is None
    assert row["canonical_status"] == "accepted"
    assert row["canonical_reason"] == "cuda_raw_v2_accepted"
    assert row["parent_agent_id"] == (1 << 40) + 10
    assert row["child_agent_id"] == (1 << 40) + 20
    assert row["parent_slot_token"] == 100
    assert row["child_slot_token"] == 200
    assert row["generation"] == 1
    assert row["tick"] == 7
    assert row["operator_id"] == "refine_descendant_parameter"
    assert row["changed"] is True
    assert row["selected_identity"] == {
        "module_instance_id_index": 0,
        "homology_id_index": 0,
    }
    assert row["proposal_digest"] is None
    assert row["rng_index_before"] == 70
    assert row["rng_index_after"] == 72
    assert row["source_receipt_sha256"] == rowv1_source_receipt_sha256_v1(
        {
            "source_receipt_abi_version": 2,
            "cuda_raw_layout": "ordinary_birth_mutation_receipt_raw_layout_v2",
            "cuda_raw_row": CUDA_RAW_V2_ROW,
        },
        source_receipt_abi_version=2,
    )
    assert canonical_rowv1_ordinary_birth_receipt_bytes_v1(row).endswith(b"\n")


def test_cuda_raw_v2_projection_receipt_records_progress_without_admission() -> None:
    receipt = cuda_raw_v2_rowv1_projection_receipt_v1()

    assert receipt["contract_id"] == CUDA_RAW_V2_ROWV1_PROJECTION_CONTRACT_ID
    assert receipt["source_receipt_abi_version"] == 2
    assert receipt["source_row_width"] == CUDA_RAW_V2_RECEIPT_ROW_WIDTH
    assert {
        "accepted_cuda_raw_v2_row_projects_to_rowv1",
        "draw_bits_preserved_in_source_receipt_hash_input",
        "changed_row_requires_selected_identity",
        "row_width_change_rejected",
    } <= set(receipt["covered_cases"])
    assert receipt["cuda_rowv1_projection_progress"] is True
    assert receipt["cuda_receipt_surface_currently_admitted"] is False
    assert receipt["cuda_abi_implementation_authorized"] is False
    assert receipt["successor_campaign_authorized"] is False
    assert receipt["contract_sha256"]


@pytest.mark.parametrize(
    ("mutator", "match"),
    (
        (lambda values: values.pop(), "row width"),
        (lambda values: values.__setitem__(13, 99), "unsupported enums"),
        (lambda values: values.__setitem__(4, -1), "owner coordinates"),
    ),
)
def test_cuda_raw_v2_projection_fails_closed_for_malformed_rows(mutator, match: str) -> None:
    values = list(CUDA_RAW_V2_ROW)
    mutator(values)

    with pytest.raises(P12RowV1CudaEntryGateError, match=match):
        project_cuda_raw_v2_ordinary_birth_receipt_rowv1_v1(
            values,
            transaction_id=9,
            cell=0,
            rng_stream_id="cuda_ordinary_birth_mutation_v2",
            rng_index_before=70,
            rng_index_after=72,
            allocator_before={"next_agent_id": (1 << 40) + 20},
            allocator_after={"next_agent_id": (1 << 40) + 21},
            package_binding_sha256=SHA_A,
            schema_binding_sha256=SHA_B,
            mechanics_binding_sha256=SHA_C,
            selected_identity={"module_instance_id_index": 0, "homology_id_index": 0},
        )


def test_cuda_raw_v2_projection_requires_identity_for_changed_row() -> None:
    with pytest.raises(P12RowV1CudaEntryGateError, match="requires selected_identity"):
        project_cuda_raw_v2_ordinary_birth_receipt_rowv1_v1(
            CUDA_RAW_V2_ROW,
            transaction_id=9,
            cell=0,
            rng_stream_id="cuda_ordinary_birth_mutation_v2",
            rng_index_before=70,
            rng_index_after=72,
            allocator_before={"next_agent_id": (1 << 40) + 20},
            allocator_after={"next_agent_id": (1 << 40) + 21},
            package_binding_sha256=SHA_A,
            schema_binding_sha256=SHA_B,
            mechanics_binding_sha256=SHA_C,
        )


def test_cuda_rowv1_v3_kernel_mirror_projection_binds_full_source_row() -> None:
    mirror = _cuda_rowv1_v3_mirror_row()
    row = project_cuda_rowv1_v3_kernel_mirror_receipt_rowv1_v1(
        mirror,
        transaction_id=9,
        cell=0,
        rng_stream_id="cuda_ordinary_birth_rowv1_v3_mirror",
        rng_index_before=70,
        rng_index_after=72,
        allocator_before={"next_agent_id": (1 << 40) + 20},
        allocator_after={"next_agent_id": (1 << 40) + 21},
        package_binding_sha256=SHA_A,
        schema_binding_sha256=SHA_B,
        mechanics_binding_sha256=SHA_C,
        selected_identity={"module_instance_id_index": 0, "homology_id_index": 0},
    )

    assert row["source_receipt_abi_version"] == CUDA_ROWV1_V3_RECEIPT_ABI_VERSION
    assert row["raw_disabled_reason_code"] == 0
    assert row["canonical_status"] == "accepted"
    assert row["canonical_reason"] == "cuda_rowv1_v3_kernel_mirror_accepted"
    assert row["parent_agent_id"] == (1 << 40) + 10
    assert row["child_agent_id"] == (1 << 40) + 20
    assert row["generation"] == 1
    assert row["tick"] == 7
    assert row["operator_id"] == "refine_descendant_parameter"
    assert row["changed"] is True
    assert row["selected_identity"] == {
        "module_instance_id_index": 0,
        "homology_id_index": 0,
    }
    assert row["source_receipt_sha256"] == rowv1_source_receipt_sha256_v1(
        {
            "source_receipt_abi_version": CUDA_ROWV1_V3_RECEIPT_ABI_VERSION,
            "cuda_rowv1_layout": "ordinary_birth_rowv1_receipt_kernel_mirror_v3",
            "mirrored_raw_layout_version": 2,
            "mirrored_raw_row": CUDA_RAW_V2_ROW,
            "cuda_rowv1_row": mirror,
        },
        source_receipt_abi_version=CUDA_ROWV1_V3_RECEIPT_ABI_VERSION,
    )
    assert canonical_rowv1_ordinary_birth_receipt_bytes_v1(row).endswith(b"\n")


def test_cuda_rowv1_v3_kernel_mirror_projection_preserves_disabled_reason() -> None:
    mirror = _cuda_rowv1_v3_mirror_row()
    mirror[15] = 4
    mirror[16] = 4
    mirror[21] = 0
    mirror[22] = 3

    row = project_cuda_rowv1_v3_kernel_mirror_receipt_rowv1_v1(
        mirror,
        transaction_id=10,
        cell=0,
        rng_stream_id="cuda_ordinary_birth_rowv1_v3_mirror",
        rng_index_before=80,
        rng_index_after=82,
        allocator_before={"next_agent_id": (1 << 40) + 20},
        allocator_after={"next_agent_id": (1 << 40) + 20},
        package_binding_sha256=SHA_A,
        schema_binding_sha256=SHA_B,
        mechanics_binding_sha256=SHA_C,
    )

    assert row["source_receipt_abi_version"] == CUDA_ROWV1_V3_RECEIPT_ABI_VERSION
    assert row["native_transaction_status"] == TRANSACTION_MUTATION_DISABLED
    assert row["raw_disabled_reason_code"] == 3
    assert row["canonical_status"] == "disabled"
    assert row["canonical_reason"] == "recurrent_mutation_disabled"
    assert row["selected_identity"] is None
    assert row["changed"] is False
    assert row["capacity_decision"] == "available"
    assert row["publication_decision"] == "not_attempted"
    assert row["rollback_decision"] == "child_unchanged"


def test_cuda_rowv1_v3_kernel_mirror_projection_receipt_records_progress() -> None:
    receipt = cuda_rowv1_v3_kernel_mirror_projection_receipt_v1()

    assert receipt["contract_id"] == CUDA_ROWV1_V3_KERNEL_MIRROR_PROJECTION_CONTRACT_ID
    assert receipt["source_receipt_abi_version"] == CUDA_ROWV1_V3_RECEIPT_ABI_VERSION
    assert receipt["source_row_width"] == CUDA_ROWV1_V3_MINIMUM_RECEIPT_ROW_WIDTH
    assert {
        "accepted_cuda_rowv1_v3_kernel_mirror_projects_to_rowv1",
        "source_sha256_bound_to_full_64_word_row",
        "non_disabled_zero_disabled_reason_sentinel_bound",
        "disabled_reason_codes_1_to_6_project_to_canonical_reasons",
        "changed_row_requires_selected_identity",
        "abi_marker_change_rejected",
        "unknown_disabled_reason_code_rejected",
    } <= set(receipt["covered_cases"])
    assert receipt["cuda_rowv1_v3_kernel_mirror_projection_progress"] is True
    assert receipt["cuda_receipt_surface_currently_admitted"] is False
    assert receipt["cuda_abi_implementation_authorized"] is False
    assert receipt["successor_campaign_authorized"] is False
    assert receipt["contract_sha256"]


def test_cuda_rowv1_disabled_event_writer_receipt_binds_device_sources() -> None:
    receipt = cuda_rowv1_disabled_event_writer_receipt_v1(source_root=ROOT)

    assert receipt["contract_id"] == CUDA_ROWV1_DISABLED_EVENT_WRITER_CONTRACT_ID
    assert receipt["decision"] == "P12-CUDA-ROWV1-DISABLED-EVENT-WRITER-A"
    assert receipt["device_disabled_event_writer_present"] is True
    assert receipt["standalone_nvcc_runtime_probe_required"] is True
    assert receipt["standalone_nvcc_runtime_probe_passed"] is True
    assert {
        "raw_disabled_reason_code",
        "transaction_sequence",
        "counts",
        "overflow",
    } <= set(receipt["device_written_fields"])
    assert {
        row["canonical_reason"] for row in receipt["disabled_reason_codes"]
    } == {
        "operator_disabled",
        "structural_mutation_disabled",
        "recurrent_mutation_disabled",
        "operator_family_disabled",
        "zero_mutation_rate",
        "mutation_attempt_budget_zero",
    }
    assert "max_plus_one_attempt_overflows_after_capacity_rows" in (
        receipt["capacity_semantics"]
    )
    assert {
        row["path"]
        for row in receipt["device_sources"]
    } >= {
        "engine/perf/cuda_architecture_b_flattened_modular_controller_v1.cuh",
        "engine/perf/cuda_architecture_b_flattened_wdl_controller_v2_binding.cpp",
        "tests/perf/test_p12_cuda_modular_rowv1_device_writer_v1.cu",
    }
    assert receipt["route_package_qualification_authorized"] is False
    assert receipt["successor_campaign_authorized"] is False
    assert receipt["contract_sha256"]


@pytest.mark.parametrize(
    ("mutator", "match"),
    (
        (lambda values: values.pop(), "row width"),
        (lambda values: values.__setitem__(0, 2), "ABI marker"),
        (lambda values: values.__setitem__(1, 3), "source layout"),
        (lambda values: values.__setitem__(14, 99), "unsupported enums"),
        (lambda values: values.__setitem__(5, -1), "owner coordinates"),
        (lambda values: values.__setitem__(22, 99), "disabled reason code"),
    ),
)
def test_cuda_rowv1_v3_kernel_mirror_projection_fails_closed_for_malformed_rows(
    mutator,
    match: str,
) -> None:
    values = _cuda_rowv1_v3_mirror_row()
    mutator(values)

    with pytest.raises(P12RowV1CudaEntryGateError, match=match):
        project_cuda_rowv1_v3_kernel_mirror_receipt_rowv1_v1(
            values,
            transaction_id=9,
            cell=0,
            rng_stream_id="cuda_ordinary_birth_rowv1_v3_mirror",
            rng_index_before=70,
            rng_index_after=72,
            allocator_before={"next_agent_id": (1 << 40) + 20},
            allocator_after={"next_agent_id": (1 << 40) + 21},
            package_binding_sha256=SHA_A,
            schema_binding_sha256=SHA_B,
            mechanics_binding_sha256=SHA_C,
            selected_identity={"module_instance_id_index": 0, "homology_id_index": 0},
        )


def test_cuda_rowv1_v3_kernel_mirror_projection_rejects_changed_disabled_row() -> None:
    mirror = _cuda_rowv1_v3_mirror_row()
    mirror[22] = 3

    with pytest.raises(P12RowV1CudaEntryGateError, match="disabled CUDA RowV1"):
        project_cuda_rowv1_v3_kernel_mirror_receipt_rowv1_v1(
            mirror,
            transaction_id=9,
            cell=0,
            rng_stream_id="cuda_ordinary_birth_rowv1_v3_mirror",
            rng_index_before=70,
            rng_index_after=72,
            allocator_before={"next_agent_id": (1 << 40) + 20},
            allocator_after={"next_agent_id": (1 << 40) + 21},
            package_binding_sha256=SHA_A,
            schema_binding_sha256=SHA_B,
            mechanics_binding_sha256=SHA_C,
            selected_identity={"module_instance_id_index": 0, "homology_id_index": 0},
        )


def test_cuda_rowv1_v3_kernel_mirror_projection_requires_identity() -> None:
    with pytest.raises(P12RowV1CudaEntryGateError, match="requires selected_identity"):
        project_cuda_rowv1_v3_kernel_mirror_receipt_rowv1_v1(
            _cuda_rowv1_v3_mirror_row(),
            transaction_id=9,
            cell=0,
            rng_stream_id="cuda_ordinary_birth_rowv1_v3_mirror",
            rng_index_before=70,
            rng_index_after=72,
            allocator_before={"next_agent_id": (1 << 40) + 20},
            allocator_after={"next_agent_id": (1 << 40) + 21},
            package_binding_sha256=SHA_A,
            schema_binding_sha256=SHA_B,
            mechanics_binding_sha256=SHA_C,
        )


def test_cuda_rowv1_v3_kernel_mirror_projection_rejects_source_hash_mismatch() -> None:
    with pytest.raises(P12RowV1CudaEntryGateError, match="source receipt hash"):
        project_cuda_rowv1_v3_kernel_mirror_receipt_rowv1_v1(
            _cuda_rowv1_v3_mirror_row(),
            transaction_id=9,
            cell=0,
            rng_stream_id="cuda_ordinary_birth_rowv1_v3_mirror",
            rng_index_before=70,
            rng_index_after=72,
            allocator_before={"next_agent_id": (1 << 40) + 20},
            allocator_after={"next_agent_id": (1 << 40) + 21},
            package_binding_sha256=SHA_A,
            schema_binding_sha256=SHA_B,
            mechanics_binding_sha256=SHA_C,
            selected_identity={"module_instance_id_index": 0, "homology_id_index": 0},
            expected_source_receipt_sha256=SHA_D,
        )


@pytest.mark.parametrize(
    ("mutator", "match"),
    (
        (
            lambda receipt: receipt.pop("disabled_reason_code"),
            "v4 disabled row requires raw disabled reason code",
        ),
        (
            lambda receipt: receipt.update(disabled_reason_code=99),
            "unknown disabled reason code",
        ),
        (
            lambda receipt: receipt.update(proposal_digest=55),
            "nonzero proposal digest requires proposal_digest_sha256",
        ),
    ),
)
def test_rowv1_projection_fails_closed_for_malformed_native_source(
    mutator,
    match: str,
) -> None:
    native_receipt = _native_receipt()
    mutator(native_receipt)

    with pytest.raises(P12RowV1CudaEntryGateError, match=match):
        _project(native_receipt)


def test_rowv1_projection_rejects_source_hash_mismatch() -> None:
    with pytest.raises(P12RowV1CudaEntryGateError, match="source receipt hash"):
        _project(_native_receipt(), expected_source_receipt_sha256=SHA_A)


def test_rowv1_validator_rejects_tampered_mapping_reason_and_mutation_fields() -> None:
    row = _project(_native_receipt())

    stale = {**row, "semantic_mapping_contract_sha256": SHA_A}
    with pytest.raises(P12RowV1CudaEntryGateError, match="stale semantic mapping"):
        validate_rowv1_ordinary_birth_receipt_v1(stale)

    changed_reason = {**row, "canonical_reason": "operator_disabled"}
    with pytest.raises(P12RowV1CudaEntryGateError, match="canonical reason mismatch"):
        validate_rowv1_ordinary_birth_receipt_v1(changed_reason)

    selected = {**row, "selected_identity": {"module_instance_id": "mi"}}
    with pytest.raises(P12RowV1CudaEntryGateError, match="clear mutation fields"):
        validate_rowv1_ordinary_birth_receipt_v1(selected)


def test_rowv1_validator_rejects_stable_id_truncation_and_rng_rewind() -> None:
    row = _project(_native_receipt())

    too_wide = {**row, "parent_agent_id": 1 << 80}
    with pytest.raises(P12RowV1CudaEntryGateError, match="parent_agent_id"):
        validate_rowv1_ordinary_birth_receipt_v1(too_wide)

    rewind = {**row, "rng_index_after": row["rng_index_before"] - 1}
    with pytest.raises(P12RowV1CudaEntryGateError, match="rng index"):
        validate_rowv1_ordinary_birth_receipt_v1(rewind)

    stale_scope = {**row, "transaction_sequence_scope": "reporting_order_v1"}
    with pytest.raises(P12RowV1CudaEntryGateError, match="sequence scope"):
        validate_rowv1_ordinary_birth_receipt_v1(stale_scope)


def test_rowv1_projection_receipt_records_authorized_host_boundary() -> None:
    receipt = rowv1_ordinary_birth_projection_receipt_v1()

    assert receipt["contract_id"] == ROWV1_PROJECTION_CONTRACT_ID
    assert receipt["decision"] == "P12-ROWV1-W-ORDINARY-BIRTH-PROJECTION"
    assert receipt["projection_function"] == (
        "project_native_core_ordinary_birth_receipt_rowv1_v1"
    )
    assert receipt["validation_function"] == "validate_rowv1_ordinary_birth_receipt_v1"
    assert {
        "v4_disabled_reason_preserves_raw_code_and_canonical_reason",
        "legacy_v3_disabled_defaults_operator_disabled_without_raw_code",
        "accepted_changed_row_preserves_selected_identity_and_values",
        "publication_rejection_preserves_proposal_digest_sha256",
    } <= set(receipt["covered_projection_cases"])
    assert {
        "source_receipt_hash_mismatch_rejected",
        "stale_semantic_mapping_contract_hash_rejected",
        "stable_id_outside_int64_rejected",
        "rng_index_rewind_rejected",
    } <= set(receipt["covered_negative_cases"])
    assert receipt["rowv1_projection_authorized"] is True
    assert receipt["cuda_abi_implementation_authorized"] is False
    assert receipt["successor_campaign_authorized"] is False
    assert receipt["contract_sha256"]


def test_checkpoint_report_envelope_preserves_rowv1_identity_fields() -> None:
    row = _project(_native_receipt(disabled_reason_code=3))
    envelope = build_rowv1_checkpoint_report_envelope_v1(
        row,
        boundary="checkpoint_export",
        reporting_enabled=True,
        lifecycle_ledger_sha256=SHA_D,
        source_checkpoint_boundary="checkpoint-bundle-a",
        restore_scope="fresh_owner_restore_ready",
        disk_publication_state="published_to_disk",
    )

    assert tuple(envelope) == CHECKPOINT_REPORT_ENVELOPE_FIELDS
    assert envelope["envelope_schema_version"] == CHECKPOINT_REPORT_ENVELOPE_VERSION
    assert envelope["contract_id"] == CHECKPOINT_REPORT_ENVELOPE_CONTRACT_ID
    assert envelope["rowv1_receipt_sha256"] == rowv1_ordinary_birth_receipt_sha256_v1(
        row
    )
    assert envelope["authoritative_transaction_sequence"] == 1
    assert envelope["transaction_sequence_scope"] == TRANSACTION_SEQUENCE_SCOPE
    assert envelope["lifecycle_ledger_sha256"] == SHA_D
    assert envelope["source_checkpoint_boundary"] == "checkpoint-bundle-a"
    assert envelope["source_receipt_abi_version"] == 4
    assert envelope["source_receipt_sha256"] == row["source_receipt_sha256"]
    assert envelope["raw_disabled_reason_code"] == 3
    assert envelope["canonical_reason"] == "recurrent_mutation_disabled"
    assert envelope["rng_stream_id"] == row["rng_stream_id"]
    assert envelope["rng_index_before"] == row["rng_index_before"]
    assert envelope["rng_index_after"] == row["rng_index_after"]
    assert envelope["allocator_before"] == row["allocator_before"]
    assert envelope["allocator_after"] == row["allocator_after"]
    assert envelope["package_binding_sha256"] == SHA_A
    assert envelope["schema_binding_sha256"] == SHA_B
    assert envelope["mechanics_binding_sha256"] == SHA_C
    assert envelope["rowv1_receipt"] == row
    assert canonical_rowv1_checkpoint_report_envelope_bytes_v1(envelope).endswith(
        b"\n"
    )
    assert rowv1_checkpoint_report_envelope_sha256_v1(envelope)


@pytest.mark.parametrize("boundary", CHECKPOINT_REPORT_BOUNDARIES)
def test_checkpoint_report_envelope_covers_all_stage6_boundaries(
    boundary: str,
) -> None:
    row = _project(_native_receipt(disabled_reason_code=4))
    envelope = build_rowv1_checkpoint_report_envelope_v1(
        row,
        boundary=boundary,
        reporting_enabled=boundary != "reporting_disabled",
        cancellation_state=(
            "cancelled_after_checkpoint" if boundary == "cancellation" else "not_cancelled"
        ),
        terminal_state="terminal" if boundary == "terminal_state" else "not_terminal",
        restore_scope=boundary if boundary.endswith("restore") else "not_restore",
        disk_publication_state=(
            "published_to_disk" if boundary == "disk_publication" else "not_published"
        ),
    )

    assert validate_rowv1_checkpoint_report_envelope_v1(envelope)["boundary"] == boundary


@pytest.mark.parametrize(
    ("field", "value", "match"),
    (
        ("source_receipt_abi_version", 3, "source_receipt_abi_version"),
        ("source_receipt_sha256", SHA_A, "source_receipt_sha256"),
        ("raw_disabled_reason_code", 6, "raw_disabled_reason_code"),
        ("canonical_reason", "operator_disabled", "canonical_reason"),
        ("authoritative_transaction_sequence", 2, "authoritative_transaction_sequence"),
        ("transaction_sequence_scope", "reporting_order_v1", "transaction_sequence_scope"),
        ("rng_index_before", 41, "rng_index_before"),
        ("rng_index_after", 43, "rng_index_after"),
        ("allocator_before", {"next_agent_id": 9}, "allocator_before"),
        ("allocator_after", {"next_agent_id": 10}, "allocator_after"),
        ("package_binding_sha256", SHA_B, "package_binding_sha256"),
        ("schema_binding_sha256", SHA_C, "schema_binding_sha256"),
        ("mechanics_binding_sha256", SHA_A, "mechanics_binding_sha256"),
    ),
)
def test_checkpoint_report_envelope_rejects_drift_from_rowv1(
    field: str,
    value: object,
    match: str,
) -> None:
    row = _project(_native_receipt(disabled_reason_code=5))
    envelope = build_rowv1_checkpoint_report_envelope_v1(
        row,
        boundary="disk_publication",
        reporting_enabled=False,
    )
    tampered = {**envelope, field: value}

    with pytest.raises(P12RowV1CudaEntryGateError, match=match):
        validate_rowv1_checkpoint_report_envelope_v1(tampered)


def test_checkpoint_report_envelope_rejects_hash_schema_and_boundary_tamper() -> None:
    row = _project(_native_receipt(disabled_reason_code=6))
    envelope = build_rowv1_checkpoint_report_envelope_v1(
        row,
        boundary="terminal_state",
        reporting_enabled=True,
        terminal_state="terminal",
    )

    stale_row_hash = {**envelope, "rowv1_receipt_sha256": SHA_A}
    with pytest.raises(P12RowV1CudaEntryGateError, match="RowV1 receipt hash"):
        validate_rowv1_checkpoint_report_envelope_v1(stale_row_hash)

    stale_schema = {**envelope, "rowv1_schema_sha256": SHA_B}
    with pytest.raises(P12RowV1CudaEntryGateError, match="stale RowV1 schema"):
        validate_rowv1_checkpoint_report_envelope_v1(stale_schema)

    stale_projection = {**envelope, "rowv1_projection_sha256": SHA_C}
    with pytest.raises(P12RowV1CudaEntryGateError, match="stale RowV1 projection"):
        validate_rowv1_checkpoint_report_envelope_v1(stale_projection)

    stale_mapping = {**envelope, "semantic_mapping_contract_sha256": SHA_A}
    with pytest.raises(P12RowV1CudaEntryGateError, match="stale semantic mapping"):
        validate_rowv1_checkpoint_report_envelope_v1(stale_mapping)

    unsupported_boundary = {**envelope, "boundary": "silent_cuda_restore"}
    with pytest.raises(P12RowV1CudaEntryGateError, match="unsupported boundary"):
        validate_rowv1_checkpoint_report_envelope_v1(unsupported_boundary)


def test_checkpoint_report_envelope_receipt_records_stage6_contract() -> None:
    receipt = rowv1_checkpoint_report_envelope_receipt_v1()

    assert receipt["contract_id"] == CHECKPOINT_REPORT_ENVELOPE_CONTRACT_ID
    assert receipt["decision"] == "P12-ROWV1-X-CHECKPOINT-REPORT-ENVELOPE"
    assert set(receipt["covered_boundaries"]) == set(CHECKPOINT_REPORT_BOUNDARIES)
    assert "source_receipt_abi_version" in receipt["preserved_identity_fields"]
    assert "raw_disabled_reason_code" in receipt["preserved_identity_fields"]
    assert "allocator_state_drift_rejected" in receipt["failclosed_cases"]
    assert receipt["rowv1_checkpoint_report_envelope_authorized"] is True
    assert receipt["cuda_abi_implementation_authorized"] is False
    assert receipt["successor_campaign_authorized"] is False
    assert receipt["contract_sha256"]


def test_checkpoint_report_collection_accepts_zero_receipts() -> None:
    collection = _collection(
        [],
        expected_receipt_count=0,
        expected_cells=[0, 1],
    )

    assert tuple(collection) == CHECKPOINT_REPORT_COLLECTION_FIELDS
    assert collection["collection_schema_version"] == CHECKPOINT_REPORT_COLLECTION_VERSION
    assert collection["contract_id"] == CHECKPOINT_REPORT_COLLECTION_CONTRACT_ID
    assert collection["receipt_count"] == 0
    assert collection["cell_count"] == 0
    assert collection["expected_receipt_count_source"] == "authoritative_lifecycle_ledger"
    assert collection["attempted_transaction_count"] == 0
    assert collection["empty_collection_policy"] == "legitimate_empty"
    assert collection["collector_success_state"] == "success_zero_attempts"
    assert collection["sequence_first"] is None
    assert collection["sequence_last"] is None
    assert collection["ordered_transaction_sequences"] == []
    assert collection["per_cell_expected_receipt_counts"] == [
        {"cell": 0, "receipt_count": 0},
        {"cell": 1, "receipt_count": 0},
    ]
    assert collection["per_cell_receipt_counts"] == [
        {"cell": 0, "receipt_count": 0},
        {"cell": 1, "receipt_count": 0},
    ]
    assert collection["generation_min"] is None
    assert collection["generation_max"] is None
    assert collection["receipt_envelope_sha256s"] == []
    assert collection["receipt_envelopes"] == []
    assert canonical_rowv1_checkpoint_report_collection_bytes_v1(collection).endswith(
        b"\n"
    )
    assert rowv1_checkpoint_report_collection_sha256_v1(collection)


def test_checkpoint_report_collection_orders_many_multicell_receipts() -> None:
    row_b = _project(
        _native_receipt(disabled_reason_code=2),
        transaction_id=2,
        authoritative_transaction_sequence=2,
        cell=1,
        generation=3,
        tick=4,
    )
    row_a = _project(
        _native_receipt(disabled_reason_code=3),
        transaction_id=1,
        authoritative_transaction_sequence=1,
        cell=0,
        generation=2,
        tick=9,
    )
    env_b = _envelope(row_b, boundary="disk_publication", reporting_enabled=True)
    env_a = _envelope(row_a, boundary="disk_publication", reporting_enabled=True)

    collection = _collection(
        [env_b, env_a],
        expected_cells=[0, 1],
        boundary="disk_publication",
    )

    assert collection["receipt_count"] == 2
    assert collection["cell_count"] == 2
    assert [env["checkpoint_cell"] for env in collection["receipt_envelopes"]] == [0, 1]
    assert collection["deterministic_order_key"] == (
        "checkpoint_cell,authoritative_transaction_sequence,"
        "transaction_id,attempt_index,rowv1_receipt_sha256"
    )
    assert collection["transaction_identity_key"] == "checkpoint_cell,transaction_id"
    assert collection["transaction_sequence_scope"] == TRANSACTION_SEQUENCE_SCOPE
    assert collection["ordered_transaction_sequences"] == [1, 2]
    assert collection["sequence_first"] == 1
    assert collection["sequence_last"] == 2
    assert collection["per_cell_sequence_ranges"] == [
        {"cell": 0, "first_sequence": 1, "last_sequence": 1, "receipt_count": 1},
        {"cell": 1, "first_sequence": 2, "last_sequence": 2, "receipt_count": 1},
    ]
    assert collection["per_cell_collection_sha256s"]
    assert collection["global_collection_sha256"]
    assert collection["per_cell_expected_receipt_counts"] == [
        {"cell": 0, "receipt_count": 1},
        {"cell": 1, "receipt_count": 1},
    ]
    assert collection["generation_min"] == 2
    assert collection["generation_max"] == 3
    assert collection["tick_min"] == 4
    assert collection["tick_max"] == 9
    assert collection["receipt_envelope_sha256s"] == [
        rowv1_checkpoint_report_envelope_sha256_v1(env_a),
        rowv1_checkpoint_report_envelope_sha256_v1(env_b),
    ]


@pytest.mark.parametrize("segment_kind", CHECKPOINT_REPORT_COLLECTION_SEGMENTS)
def test_checkpoint_report_collection_covers_segments(segment_kind: str) -> None:
    row = _project(_native_receipt(disabled_reason_code=4))
    envelope = _envelope(row, boundary="checkpoint_export")
    kwargs: dict[str, object] = {}
    if segment_kind == "resumed_suffix":
        kwargs["parent_collection_sha256"] = SHA_B
        kwargs["last_prefix_sequence"] = 0
    if segment_kind == "checkpoint_plus_resume":
        kwargs["prefix_collection_sha256"] = SHA_C
        kwargs["last_prefix_sequence"] = 0
    if segment_kind == "terminal":
        kwargs["boundary"] = "terminal_state"

    collection = _collection([envelope], segment_kind=segment_kind, **kwargs)

    assert validate_rowv1_checkpoint_report_collection_v1(collection)[
        "segment_kind"
    ] == segment_kind


@pytest.mark.parametrize(
    ("mutator", "match"),
    (
        (
            lambda collection: collection.update(expected_receipt_count=2),
            "count mismatch",
        ),
        (
            lambda collection: collection.update(receipt_envelopes=[]),
            "receipt envelope count",
        ),
        (
            lambda collection: collection.update(receipt_envelope_sha256s=[]),
            "hash list mismatch",
        ),
        (
            lambda collection: collection.update(
                rowv1_checkpoint_report_envelope_sha256=SHA_A
            ),
            "stale RowV1 envelope",
        ),
        (
            lambda collection: collection.update(expected_cells=[1]),
            "per-cell ledger counts omit expected cells",
        ),
        (
            lambda collection: collection.update(
                expected_receipt_count_source="reporting_input"
            ),
            "expected_receipt_count_source",
        ),
    ),
)
def test_checkpoint_report_collection_rejects_tampered_contract_and_counts(
    mutator,
    match: str,
) -> None:
    row = _project(_native_receipt(disabled_reason_code=5))
    envelope = _envelope(row)
    collection = _collection([envelope])
    mutator(collection)

    with pytest.raises(P12RowV1CudaEntryGateError, match=match):
        validate_rowv1_checkpoint_report_collection_v1(collection)


def test_checkpoint_report_collection_rejects_duplicate_and_order_drift() -> None:
    row_a = _project(
        _native_receipt(disabled_reason_code=2),
        transaction_id=1,
        cell=0,
        generation=1,
        tick=1,
    )
    row_b = _project(
        _native_receipt(disabled_reason_code=3),
        transaction_id=2,
        cell=1,
        generation=1,
        tick=1,
    )
    env_a = _envelope(row_a)
    env_b = _envelope(row_b)

    with pytest.raises(P12RowV1CudaEntryGateError, match="duplicate"):
        _collection([env_a, env_a], expected_cells=[0], expected_receipt_count=2)

    collection = _collection([env_a, env_b], expected_cells=[0, 1])
    reversed_collection = {
        **collection,
        "receipt_envelopes": [env_b, env_a],
        "receipt_envelope_sha256s": [
            rowv1_checkpoint_report_envelope_sha256_v1(env_b),
            rowv1_checkpoint_report_envelope_sha256_v1(env_a),
        ],
    }
    with pytest.raises(P12RowV1CudaEntryGateError, match="order"):
        validate_rowv1_checkpoint_report_collection_v1(reversed_collection)


def test_checkpoint_report_collection_rejects_duplicate_transaction_identity() -> None:
    row_a = _project(
        _native_receipt(disabled_reason_code=2),
        transaction_id=1,
        cell=0,
        generation=1,
        tick=1,
    )
    row_b = _project(
        _native_receipt(disabled_reason_code=3),
        transaction_id=1,
        cell=0,
        generation=1,
        tick=2,
    )
    env_a = _envelope(row_a)
    env_b = _envelope(row_b)

    with pytest.raises(P12RowV1CudaEntryGateError, match="transaction identity"):
        _collection([env_a, env_b], expected_receipt_count=2)


def test_checkpoint_report_collection_rejects_duplicate_transaction_sequence() -> None:
    row_a = _project(
        _native_receipt(disabled_reason_code=2),
        transaction_id=1,
        authoritative_transaction_sequence=4,
        cell=0,
        generation=1,
        tick=1,
    )
    row_b = _project(
        _native_receipt(disabled_reason_code=3),
        transaction_id=2,
        authoritative_transaction_sequence=4,
        cell=0,
        generation=1,
        tick=2,
    )
    env_a = _envelope(row_a)
    env_b = _envelope(row_b)

    with pytest.raises(P12RowV1CudaEntryGateError, match="transaction sequence"):
        _collection([env_a, env_b], expected_receipt_count=2)


@pytest.mark.parametrize(
    ("mutator", "match"),
    (
        (
            lambda collection: collection.update(
                ordered_transaction_sequences=[99]
            ),
            "ordered transaction sequence",
        ),
        (
            lambda collection: collection.update(sequence_first=99),
            "sequence first",
        ),
        (
            lambda collection: collection.update(
                per_cell_sequence_ranges=[
                    {
                        "cell": 0,
                        "first_sequence": 99,
                        "last_sequence": 99,
                        "receipt_count": 1,
                    }
                ]
            ),
            "per-cell sequence range",
        ),
        (
            lambda collection: collection.update(per_cell_collection_sha256s=[]),
            "per-cell collection hash",
        ),
        (
            lambda collection: collection.update(global_collection_sha256=SHA_D),
            "global collection hash",
        ),
        (
            lambda collection: collection.update(
                transaction_sequence_scope="global_batch_order_v1"
            ),
            "sequence scope",
        ),
    ),
)
def test_checkpoint_report_collection_rejects_sequence_authority_drift(
    mutator,
    match: str,
) -> None:
    row = _project(_native_receipt(disabled_reason_code=5))
    envelope = _envelope(row)
    collection = _collection([envelope])
    mutator(collection)

    with pytest.raises(P12RowV1CudaEntryGateError, match=match):
        validate_rowv1_checkpoint_report_collection_v1(collection)


def test_checkpoint_report_collection_rejects_prefix_suffix_gap_overlap_and_reset() -> None:
    row = _project(
        _native_receipt(disabled_reason_code=5),
        authoritative_transaction_sequence=5,
    )
    envelope = _envelope(row)

    gap = _collection(
        [envelope],
        parent_collection_sha256=SHA_B,
        segment_kind="resumed_suffix",
        last_prefix_sequence=4,
    )
    assert gap["first_suffix_sequence"] == 5
    assert gap["last_prefix_sequence"] == 4

    with pytest.raises(P12RowV1CudaEntryGateError, match="adjacency"):
        _collection(
            [envelope],
            parent_collection_sha256=SHA_B,
            segment_kind="resumed_suffix",
            last_prefix_sequence=3,
        )

    with pytest.raises(P12RowV1CudaEntryGateError, match="adjacency|regression"):
        _collection(
            [envelope],
            parent_collection_sha256=SHA_B,
            segment_kind="resumed_suffix",
            last_prefix_sequence=5,
        )


@pytest.mark.parametrize(
    ("mutator", "match"),
    (
        (
            lambda collection: collection.update(empty_collection_policy="legitimate_empty"),
            "nonempty collection",
        ),
        (
            lambda collection: collection.update(
                per_cell_expected_receipt_counts=[{"cell": 0, "receipt_count": 2}]
            ),
            "per-cell ledger count",
        ),
        (
            lambda collection: collection.update(
                per_cell_receipt_counts=[{"cell": 0, "receipt_count": 99}]
            ),
            "per-cell receipt count",
        ),
        (
            lambda collection: collection.update(collection_capacity_limit=0),
            "capacity",
        ),
        (
            lambda collection: collection.update(
                checkpoint_state_sha256=SHA_D,
                join_binding_sha256=SHA_D,
            ),
            "join binding",
        ),
    ),
)
def test_checkpoint_report_collection_rejects_bridge_integrity_drift(
    mutator,
    match: str,
) -> None:
    row = _project(_native_receipt(disabled_reason_code=5))
    envelope = _envelope(row)
    collection = _collection(
        [envelope],
        parent_collection_sha256=SHA_B,
        segment_kind="resumed_suffix",
        last_prefix_sequence=0,
    )
    mutator(collection)

    with pytest.raises(P12RowV1CudaEntryGateError, match=match):
        validate_rowv1_checkpoint_report_collection_v1(collection)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    (
        ({"overflow_count": 1}, "overflow or dropped"),
        ({"dropped_receipt_count": 1}, "overflow or dropped"),
        (
            {"boundary": "cancellation", "cancellation_state": "operator_cancelled"},
            "cancel",
        ),
        ({"segment_kind": "resumed_suffix"}, "parent collection"),
        ({"segment_kind": "checkpoint_plus_resume"}, "prefix collection"),
    ),
)
def test_checkpoint_report_collection_rejects_unqualified_lifecycle_states(
    kwargs: dict[str, object],
    match: str,
) -> None:
    row = _project(_native_receipt(disabled_reason_code=6))
    envelope = _envelope(row, boundary=str(kwargs.get("boundary", "checkpoint_export")))

    with pytest.raises(P12RowV1CudaEntryGateError, match=match):
        _collection([envelope], **kwargs)


def test_checkpoint_report_collection_accepts_sealed_cancellation() -> None:
    row = _project(_native_receipt(disabled_reason_code=2))
    envelope = _envelope(row, boundary="cancellation", cancellation_state="cancelled")
    collection = _collection(
        [envelope],
        boundary="cancellation",
        cancellation_state="cancelled",
        sealed=True,
    )

    assert collection["boundary"] == "cancellation"
    assert collection["cancellation_state"] == "cancelled"
    assert collection["sealed"] is True
    assert collection["cancellation_seal_sha256"]


def test_checkpoint_report_collection_rejects_cancellation_seal_drift() -> None:
    row = _project(_native_receipt(disabled_reason_code=2))
    envelope = _envelope(row, boundary="cancellation", cancellation_state="cancelled")
    collection = _collection(
        [envelope],
        boundary="cancellation",
        cancellation_state="cancelled",
        sealed=True,
    )
    tampered = {**collection, "cancellation_seal_sha256": SHA_B}

    with pytest.raises(P12RowV1CudaEntryGateError, match="cancellation seal"):
        validate_rowv1_checkpoint_report_collection_v1(tampered)


def test_checkpoint_report_collection_receipt_records_bridge_feedback_gate() -> None:
    receipt = rowv1_checkpoint_report_collection_receipt_v1()

    assert receipt["contract_id"] == CHECKPOINT_REPORT_COLLECTION_CONTRACT_ID
    assert receipt["decision"] == "P12-ROWV1-Y-CHECKPOINT-REPORT-COLLECTION"
    assert set(receipt["covered_segments"]) == set(CHECKPOINT_REPORT_COLLECTION_SEGMENTS)
    assert {"zero_receipts", "many_receipts", "checkpoint_plus_resume"} <= set(
        receipt["collection_cases"]
    )
    assert "duplicate_rowv1_receipt_rejected" in receipt["failclosed_cases"]
    assert "duplicate_transaction_identity_rejected" in receipt["failclosed_cases"]
    assert "reporting_derived_expected_count_rejected" in receipt["failclosed_cases"]
    assert "checkpoint_lineage_join_binding_mismatch_rejected" in receipt[
        "failclosed_cases"
    ]
    assert "unsealed_cancellation_rejected" in receipt["failclosed_cases"]
    assert receipt["sequence_scope"] == TRANSACTION_SEQUENCE_SCOPE
    assert receipt["canonical_semantic_order"] == (
        "checkpoint_cell",
        "authoritative_transaction_sequence",
        "transaction_id",
        "attempt_index",
        "rowv1_receipt_sha256",
    )
    assert "duplicate_transaction_sequence_rejected" in receipt["failclosed_cases"]
    assert "prefix_suffix_sequence_gap_or_overlap_rejected" in receipt[
        "failclosed_cases"
    ]
    assert receipt["rowv1_checkpoint_report_collection_authorized"] is True
    assert receipt["cuda_abi_implementation_authorized"] is False
    assert receipt["successor_campaign_authorized"] is False
    assert receipt["contract_sha256"]


def test_sequence_order_contract_fields_are_present_in_schema_envelope_and_collection() -> None:
    schema_fields = set(ROWV1_FIELD_NAMES)
    envelope_fields = set(CHECKPOINT_REPORT_ENVELOPE_FIELDS)
    collection_fields = set(CHECKPOINT_REPORT_COLLECTION_FIELDS)
    collection_receipt = rowv1_checkpoint_report_collection_receipt_v1()

    assert {"authoritative_transaction_sequence", "transaction_sequence_scope"} <= (
        schema_fields
    )
    assert {
        "authoritative_transaction_sequence",
        "transaction_sequence_scope",
        "lifecycle_ledger_sha256",
        "source_checkpoint_boundary",
    } <= envelope_fields
    assert {
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
    } <= collection_fields
    declared_order = tuple(collection_receipt["deterministic_order_key"].split(","))
    assert declared_order == collection_receipt["canonical_semantic_order"]
    for field in declared_order:
        assert (
            field in envelope_fields
            or field in schema_fields
            or field == "rowv1_receipt_sha256"
        )
