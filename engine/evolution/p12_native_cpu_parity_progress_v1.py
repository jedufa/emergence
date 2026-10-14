"""P1.2 native CPU parity progress receipts.

This module records bounded native CPU evidence. It is deliberately narrower
than the full P12 native CPU answer key and cannot authorize CUDA promotion.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Final, Mapping

from engine.perf.recurrent_native.native_core_ordinary_birth_receipt_bridge_v1 import (
    DISABLED_REASON_BY_CODE,
    DISABLED_REASON_CODE_ORDER,
    ORDINARY_BIRTH_TERMINAL_PRECEDENCE_V1,
)


CONTRACT_ID: Final[str] = "p12-native-cpu-recurrent-lifecycle-owner-parity-v1"
NATIVE_CPU_ROUTE_BOUND_BIRTH_RECEIPT_CONTRACT_ID: Final[str] = (
    "p12-native-cpu-route-bound-birth-lifecycle-receipt-v1"
)
NATIVE_CPU_GATE1_ROUTE_QUALIFICATION_CONTRACT_ID: Final[str] = (
    "p12-native-cpu-gate1-turnover-route-package-qualification-v1"
)
NATIVE_CPU_ROUTE_BOUND_ARTIFACT_SHA256: Final[str] = (
    "2abc1b53689acf8814ef874d3464a60ae7cb20b5bcb91bbca9d5bd90929b310e"
)
NATIVE_CPU_ROUTE_BOUND_BUILD_MANIFEST_SHA256: Final[str] = (
    "050340ba20e6952988cc7800f151a9f8632b080a2654734d427fa80afe6ef131"
)
RECEIPT_PAYLOAD_GAP_CONTRACT_ID: Final[str] = (
    "p12-native-cpu-recurrent-receipt-payload-gap-map-v1"
)
NATIVE_CORE_OPERATOR_CONFORMANCE_CONTRACT_ID: Final[str] = (
    "p12-native-core-two-draw-operator-conformance-v1"
)
NATIVE_CORE_C_API_BOUNDARY_CONTRACT_ID: Final[str] = (
    "p12-native-core-c-api-two-draw-boundary-v1"
)
NATIVE_CORE_ORDINARY_BIRTH_BOUNDARY_CONTRACT_ID: Final[str] = (
    "p12-native-core-ordinary-birth-transaction-boundary-v1"
)
NATIVE_CORE_RECURRENT_RECEIPT_BRIDGE_CONTRACT_ID: Final[str] = (
    "p12-native-core-ordinary-birth-recurrent-receipt-bridge-v1"
)
NATIVE_CORE_PAYLOAD_OBSERVABILITY_CONTRACT_ID: Final[str] = (
    "p12-native-core-ordinary-birth-payload-observability-v1"
)
NATIVE_CORE_IDENTITY_MAPPING_CONTRACT_ID: Final[str] = (
    "p12-native-core-ordinary-birth-identity-mapping-bridge-v1"
)
NATIVE_CORE_PAYLOAD_PROJECTION_CONTRACT_ID: Final[str] = (
    "p12-native-core-ordinary-birth-payload-projection-v1"
)
NATIVE_CORE_STRUCTURAL_PAYLOAD_SHA_CONTRACT_ID: Final[str] = (
    "p12-native-core-ordinary-birth-structural-payload-sha-v1"
)
NATIVE_CORE_ORDINARY_BIRTH_V2_RECEIPT_SURFACE_CONTRACT_ID: Final[str] = (
    "p12-native-core-ordinary-birth-v2-receipt-surface-v1"
)
NATIVE_CORE_ORACLE_ALIGNED_RECEIPT_BYTES_CONTRACT_ID: Final[str] = (
    "p12-native-core-ordinary-birth-oracle-aligned-receipt-bytes-v1"
)
NATIVE_CORE_REJECTED_RECEIPT_BYTES_CONTRACT_ID: Final[str] = (
    "p12-native-core-ordinary-birth-rejected-receipt-bytes-v1"
)
NATIVE_CORE_OWNER_REJECTION_RECEIPT_BYTES_CONTRACT_ID: Final[str] = (
    "p12-native-core-ordinary-birth-owner-rejection-receipt-bytes-v1"
)
NATIVE_CORE_DISABLED_RECEIPT_BYTES_CONTRACT_ID: Final[str] = (
    "p12-native-core-ordinary-birth-disabled-receipt-bytes-v1"
)
NATIVE_CORE_DISABLED_REASON_RECEIPT_BYTES_CONTRACT_ID: Final[str] = (
    "p12-native-core-ordinary-birth-disabled-reason-receipt-bytes-v1"
)
NATIVE_CORE_DISABLED_REASON_SEMANTICS_CONTRACT_ID: Final[str] = (
    "p12-native-core-ordinary-birth-disabled-reason-semantics-v1"
)
NATIVE_CORE_INVALID_INPUT_RECEIPT_CONTRACT_ID: Final[str] = (
    "p12-native-core-ordinary-birth-invalid-input-receipt-v1"
)
NATIVE_CORE_PROPOSAL_DIGEST_VECTORS_CONTRACT_ID: Final[str] = (
    "p12-native-core-ordinary-birth-proposal-digest-vectors-v1"
)
NATIVE_CORE_RECEIPT_REPEAT_CONTRACT_ID: Final[str] = (
    "p12-native-core-ordinary-birth-receipt-repeat-determinism-v1"
)
NATIVE_CORE_RECURRENT_RECEIPT_BYTE_PROGRESS_CONTRACT_ID: Final[str] = (
    "p12-native-core-recurrent-receipt-byte-authority-progress-v1"
)
SCHEMA_VERSION: Final[str] = "p12_native_cpu_parity_progress_v1"
ANSWER_KEY_CONTRACT_ID: Final[str] = "p12-native-cpu-parity-entry-gate-v1_1"
ANSWER_KEY_CONTRACT_SHA256: Final[str] = (
    "4a0b199b22da492173a32dcf505a70cd4299439852233507f7e5b262bf2b3da6"
)

COVERED_COMPARE_FIELDS: Final[tuple[str, ...]] = (
    "child_agent_id",
    "parent_slot",
    "child_slot",
    "parent_slot_token",
    "child_slot_token",
    "publication_state",
    "rollback_state",
    "recurrent_payload_sha256",
    "checkpoint_bytes",
)

NOT_COVERED_COMPARE_FIELDS: Final[tuple[str, ...]] = (
    "rng_draw_bits",
    "proposal_bytes",
    "structural_payload_sha256",
    "cursor_rows",
    "receipt_bytes",
    "multicell_continuation",
    "cuda_parity",
    "route_package_qualification",
)

RECEIPT_PAYLOAD_GAP_FIELDS: Final[tuple[str, ...]] = (
    "rng_draw_bits",
    "proposal_bytes",
    "structural_payload_sha256",
    "receipt_bytes",
)

RECEIPT_PAYLOAD_GAP_SURFACES: Final[tuple[dict[str, Any], ...]] = (
    {
        "surface_id": "python_reference_structural_payload_oracle",
        "path": "engine/evolution/p12_stage2_mutation_oracles_v1.py",
        "current_state": "implemented_reference_only",
        "authority": "reference_answer_key",
    },
    {
        "surface_id": "python_reference_recurrent_receipt_replay",
        "path": (
            "engine/perf/recurrent_native/"
            "modular_recurrent_mutation_receipt_v1.py"
        ),
        "current_state": "implemented_reference_only",
        "authority": "reference_answer_key",
    },
    {
        "surface_id": "native_cpu_recurrent_lifecycle_c_api",
        "path": (
            "engine/perf/native/recurrent/"
            "modular_recurrent_cpu_oracle_c_api_v1.cpp"
        ),
        "current_state": (
            "real_lifecycle_owner_birth_receipt_native_canonical_digest_and_native_structural_sha_binding_surface"
        ),
        "authority": NATIVE_CPU_ROUTE_BOUND_BIRTH_RECEIPT_CONTRACT_ID,
    },
    {
        "surface_id": "native_cpu_structural_payload_oracle",
        "path": (
            "engine/perf/native/recurrent/"
            "modular_recurrent_cpu_oracle_c_api_v1.cpp"
        ),
        "current_state": (
            "native_generated_rowv1_structural_payload_sha_present_and_route_bound_v3"
        ),
        "authority": NATIVE_CPU_ROUTE_BOUND_BIRTH_RECEIPT_CONTRACT_ID,
    },
    {
        "surface_id": "native_cpu_recurrent_receipt_bytes",
        "path": (
            "engine/perf/native/recurrent/modular_recurrent_core_c_api_v1.cpp"
        ),
        "current_state": (
            "bounded_native_core_byte_comparable_present_not_route_bound"
        ),
        "authority": NATIVE_CORE_RECURRENT_RECEIPT_BYTE_PROGRESS_CONTRACT_ID,
    },
)

CURRENT_NATIVE_RECURRENT_LIFECYCLE_APIS: Final[tuple[str, ...]] = (
    "eiel_modular_recurrent_newborn_inherit_v1",
    "eiel_modular_recurrent_newborn_inherit_with_receipt_v1",
    "eiel_modular_recurrent_newborn_inherit_with_canonical_receipt_v1",
    "eiel_modular_recurrent_same_slot_reuse_v1",
    "eiel_modular_recurrent_same_slot_reuse_with_receipt_v1",
    "eiel_modular_recurrent_same_slot_reuse_with_canonical_receipt_v1",
    "eiel_modular_recurrent_row_structural_payload_sha256_v1",
    "eiel_modular_recurrent_canonical_receipt_sha256_v1",
    "eiel_modular_recurrent_death_cleanup_v1",
    "eiel_modular_recurrent_checkpoint_project_v1",
    "eiel_modular_recurrent_checkpoint_restore_v1",
)

ABSENT_NATIVE_RECEIPT_PAYLOAD_SYMBOLS: Final[tuple[str, ...]] = (
    "mutate_sustainable_growth",
    "structural_ordinary_birth_oracle_v1",
    "recurrent_ordinary_birth_oracle_v1",
    "build_modular_recurrent_mutation_receipt_v1",
    "oracle_sha256",
)

NATIVE_CORE_OPERATOR_CONFORMANCE_FIELDS: Final[tuple[str, ...]] = (
    "operation_bin",
    "native_status",
    "selected_slot",
    "changed_dimension",
    "changed_flag",
    "canonical_digest_change",
    "invalid_draw_rollback",
)

NATIVE_CORE_OPERATOR_CONFORMANCE_LIMITS: Final[tuple[str, ...]] = (
    "constrained_flattened_native_core_fixture_only",
    "float32_draw_core_not_float64_receipt_boundary",
    "no_python_json_payload_sha256",
    "no_canonical_modular_two_draws_receipt",
    "not_exposed_by_production_recurrent_c_api",
    "no_cuda_promotion",
)

NATIVE_CORE_C_API_EXPORTS: Final[tuple[str, ...]] = (
    "eiel_modular_recurrent_core_abi_version_v1",
    "eiel_modular_recurrent_core_fixed_envelope_size_v1",
    "eiel_modular_recurrent_core_fixed_envelope_alignment_v1",
    "eiel_modular_recurrent_core_initialize_empty_v1",
    "eiel_modular_recurrent_core_validate_v1",
    "eiel_modular_recurrent_core_canonical_digest_v1",
    "eiel_modular_recurrent_core_module_count_v1",
    "eiel_modular_recurrent_core_activate_integrator_v1",
    "eiel_modular_recurrent_core_neutral_duplicate_v1",
    "eiel_modular_recurrent_core_mutate_two_draws_v1",
)

NATIVE_CORE_ORDINARY_BIRTH_EXPORTS: Final[tuple[str, ...]] = (
    *NATIVE_CORE_C_API_EXPORTS,
    "eiel_modular_recurrent_core_ordinary_birth_two_draws_v1",
)

NATIVE_CORE_ORDINARY_BIRTH_V2_EXPORTS: Final[tuple[str, ...]] = (
    *NATIVE_CORE_ORDINARY_BIRTH_EXPORTS,
    "eiel_modular_recurrent_core_ordinary_birth_receipt_layout_version_v2",
    "eiel_modular_recurrent_core_ordinary_birth_receipt_size_v2",
    "eiel_modular_recurrent_core_ordinary_birth_receipt_alignment_v2",
    "eiel_modular_recurrent_core_ordinary_birth_two_draws_v2",
)


def canonical_native_cpu_progress_bytes_v1(value: Mapping[str, Any]) -> bytes:
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


def native_cpu_progress_sha256_v1(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_native_cpu_progress_bytes_v1(value)).hexdigest()


def native_cpu_recurrent_lifecycle_owner_progress_v1() -> dict[str, Any]:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "contract_id": CONTRACT_ID,
        "answer_key_contract_id": ANSWER_KEY_CONTRACT_ID,
        "answer_key_contract_sha256": ANSWER_KEY_CONTRACT_SHA256,
        "case_id": "native_cpu_recurrent_lifecycle_owner_surfaces",
        "native_surface": (
            "engine/perf/native/recurrent/"
            "modular_recurrent_cpu_oracle_c_api_v1.cpp"
        ),
        "test_surface": "tests/perf/test_eiel_modular_recurrent_cpu_oracle_v1.py",
        "covered_compare_fields": COVERED_COMPARE_FIELDS,
        "not_covered_compare_fields": NOT_COVERED_COMPARE_FIELDS,
        "real_lifecycle_calls": (
            "eiel_modular_recurrent_newborn_inherit_v1",
            "eiel_modular_recurrent_same_slot_reuse_v1",
            "eiel_modular_recurrent_death_cleanup_v1",
            "eiel_modular_recurrent_checkpoint_project_v1",
            "eiel_modular_recurrent_checkpoint_restore_v1",
        ),
        "native_cpu_a_authorized": False,
        "cuda_promotion_authorized": False,
        "successor_campaign_authorized": False,
    }
    payload["contract_sha256"] = native_cpu_progress_sha256_v1(payload)
    return payload


def native_cpu_route_bound_birth_lifecycle_receipt_v1() -> dict[str, Any]:
    """Record append-only receipt evidence on the real native CPU birth lifecycle."""

    payload = {
        "schema_version": SCHEMA_VERSION,
        "contract_id": NATIVE_CPU_ROUTE_BOUND_BIRTH_RECEIPT_CONTRACT_ID,
        "answer_key_contract_id": ANSWER_KEY_CONTRACT_ID,
        "answer_key_contract_sha256": ANSWER_KEY_CONTRACT_SHA256,
        "case_id": "native_cpu_route_bound_birth_lifecycle_receipt",
        "decision": "P12-NATIVE-CPU-F-NATIVE-CANONICAL-RECEIPT-DIGEST",
        "native_surface": (
            "engine/perf/native/recurrent/"
            "modular_recurrent_cpu_oracle_c_api_v1.cpp"
        ),
        "artifact_surface": (
            "engine/perf/recurrent_native/"
            "libeiel_modular_recurrent_cpu_oracle_v1.so"
        ),
        "artifact_sha256": NATIVE_CPU_ROUTE_BOUND_ARTIFACT_SHA256,
        "build_manifest_surface": (
            "engine/perf/recurrent_native/"
            "modular_recurrent_cpu_oracle_shared_build_manifest_v1.json"
        ),
        "build_manifest_sha256": NATIVE_CPU_ROUTE_BOUND_BUILD_MANIFEST_SHA256,
        "header_surface": (
            "engine/perf/native/recurrent/"
            "modular_recurrent_cpu_oracle_c_api_v1.h"
        ),
        "test_surface": "tests/perf/test_eiel_modular_recurrent_cpu_oracle_v1.py",
        "route_bound_symbol": (
            "eiel_modular_recurrent_newborn_inherit_with_receipt_v1"
        ),
        "same_slot_route_bound_symbol": (
            "eiel_modular_recurrent_same_slot_reuse_with_receipt_v1"
        ),
        "canonical_route_bound_symbol": (
            "eiel_modular_recurrent_newborn_inherit_with_canonical_receipt_v1"
        ),
        "same_slot_canonical_route_bound_symbol": (
            "eiel_modular_recurrent_same_slot_reuse_with_canonical_receipt_v1"
        ),
        "native_structural_payload_sha_symbol": (
            "eiel_modular_recurrent_row_structural_payload_sha256_v1"
        ),
        "native_canonical_receipt_sha_symbol": (
            "eiel_modular_recurrent_canonical_receipt_sha256_v1"
        ),
        "wrapped_lifecycle_symbol": "eiel_modular_recurrent_newborn_inherit_v1",
        "same_slot_wrapped_lifecycle_symbol": (
            "eiel_modular_recurrent_same_slot_reuse_v1"
        ),
        "receipt_struct": "eiel_modular_recurrent_birth_receipt_v1",
        "covered_receipt_fields": (
            "receipt_version",
            "lifecycle_status",
            "committed",
            "owner_checked",
            "parent_preserved",
            "child_runtime_reset",
            "canonical_receipt_bound",
            "canonical_receipt_identity_checked",
            "recurrent_receipt_bytes_bound",
            "destination_row",
            "parent_a_row",
            "parent_b_row",
            "selected_parent_row",
            "expected_destination_token",
            "expected_parent_a_token",
            "expected_parent_b_token",
            "parent_a_agent_id",
            "parent_b_agent_id",
            "child_agent_id",
            "publication_token",
            "donor_draw_float32_bits",
            "mutation_count",
            "draw0_float64_bits",
            "draw1_float64_bits",
            "canonical_recurrent_receipt_sha256_words",
            "canonical_recurrent_receipt_bytes_sha256_words",
            "structural_payload_sha256_words",
            "structural_payload_sha256_bound",
            "structural_payload_identity_checked",
            "transaction_ordinal",
            "selected_instance_id_index",
            "selected_homology_id_index",
            "operation_code",
            "status_code",
            "reason_code",
            "changed",
        ),
        "covered_binding_fields": (
            "changed_dimension_code",
            "native_receipt_construction_required",
            "source_packed_sha256_words",
            "ancestry_tape_sha256_words",
        ),
        "covered_behaviors": (
            "calls_real_native_cpu_newborn_inherit_lifecycle",
            "records_committed_publication_receipt",
            "records_same_slot_fresh_owner_publication_receipt",
            "binds_committed_newborn_to_canonical_recurrent_receipt_digest",
            "binds_same_slot_reuse_to_canonical_recurrent_receipt_digest",
            "native_builds_committed_newborn_canonical_recurrent_receipt_digest",
            "native_builds_same_slot_canonical_recurrent_receipt_digest",
            "binds_committed_newborn_to_native_generated_structural_payload_sha256_digest",
            "binds_same_slot_reuse_to_native_generated_structural_payload_sha256_digest",
            "rolls_back_newborn_on_native_canonical_receipt_digest_mismatch",
            "rolls_back_same_slot_reuse_on_native_canonical_receipt_digest_mismatch",
            "rolls_back_newborn_on_native_structural_payload_sha256_mismatch",
            "rolls_back_same_slot_reuse_on_native_structural_payload_sha256_mismatch",
            "preserves_exact_binary64_recurrent_draw_bits_in_route_receipt",
            "rejects_stale_canonical_binding_without_state_change",
            "records_owner_mismatch_failure_receipt",
            "records_same_slot_owner_mismatch_failure_receipt",
            "proves_stale_owner_rejection_preserves_state",
            "proves_same_slot_stale_owner_rejection_preserves_state",
            "binds_committed_child_checkpoint_bytes",
            "binds_committed_child_checkpoint_restore_bytes",
            "binds_same_slot_child_checkpoint_bytes",
            "binds_v4_newborn_checkpoint_project_restore_bytes",
            "binds_v4_same_slot_checkpoint_project_restore_bytes",
            "proves_v4_checkpoint_restore_preserves_native_structural_payload_sha256",
            "binds_bounded_multicell_v4_checkpoint_collection_restore_bytes",
            "preserves_bounded_multicell_v4_collection_order",
            "binds_bounded_multicell_checkpoint_fixture_to_verified_prebuilt_artifact",
            "proves_generation_turnover_checkpoint_resume_repeats_route_bound_v4_birth_receipts",
            "preserves_generation_turnover_suffix_canonical_receipt_digest_after_resume",
            "preserves_generation_turnover_suffix_structural_payload_sha_after_resume",
            "requires_two_build_artifact_byte_identity_before_loading",
            "requires_build_manifest_source_hash_match",
            "requires_build_manifest_artifact_hash_match",
            "preserves_parent_row_bytes",
            "preserves_runtime_reset_boundary",
        ),
        "limits": (
            "source_and_ancestry_hashes_are_caller_supplied",
            "bounded_one_transaction_ordinary_birth_canonical_builder_only",
            "native_structural_payload_digest_is_rowv1_subset_not_full_canonical_payload_bytes",
            "one_mapped_structural_payload_sha_fixture_only",
            "bounded_two_cell_multicell_checkpoint_fixture_is_artifact_bound",
            "bounded_b1_generation_turnover_checkpoint_resume_fixture_only",
            "not_full_route_package_multicell_checkpoint_qualification",
            "not_cuda_authoritative_receipt_abi",
            "no_route_package_qualification",
        ),
        "native_cpu_a_authorized": False,
        "cuda_promotion_authorized": False,
        "successor_campaign_authorized": False,
    }
    payload["contract_sha256"] = native_cpu_progress_sha256_v1(payload)
    return payload


def native_cpu_recurrent_receipt_payload_gap_map_v1() -> dict[str, Any]:
    """Record the remaining native CPU parity gap without authorizing promotion."""

    payload = {
        "schema_version": SCHEMA_VERSION,
        "contract_id": RECEIPT_PAYLOAD_GAP_CONTRACT_ID,
        "answer_key_contract_id": ANSWER_KEY_CONTRACT_ID,
        "answer_key_contract_sha256": ANSWER_KEY_CONTRACT_SHA256,
        "case_id": "native_cpu_recurrent_receipt_payload_gap_map",
        "decision": "P12-NATIVE-CPU-C-FAIL-CLOSED-GAP-MAP",
        "positive_native_evidence_contract_id": CONTRACT_ID,
        "positive_native_evidence_sha256": (
            native_cpu_recurrent_lifecycle_owner_progress_v1()["contract_sha256"]
        ),
        "route_bound_birth_receipt_contract_id": (
            NATIVE_CPU_ROUTE_BOUND_BIRTH_RECEIPT_CONTRACT_ID
        ),
        "route_bound_birth_receipt_contract_sha256": (
            native_cpu_route_bound_birth_lifecycle_receipt_v1()["contract_sha256"]
        ),
        "gap_compare_fields": RECEIPT_PAYLOAD_GAP_FIELDS,
        "surface_inventory": RECEIPT_PAYLOAD_GAP_SURFACES,
        "current_native_lifecycle_apis": CURRENT_NATIVE_RECURRENT_LIFECYCLE_APIS,
        "absent_native_receipt_payload_symbols": ABSENT_NATIVE_RECEIPT_PAYLOAD_SYMBOLS,
        "bounded_recurrent_receipt_byte_progress_contract_id": (
            NATIVE_CORE_RECURRENT_RECEIPT_BYTE_PROGRESS_CONTRACT_ID
        ),
        "bounded_recurrent_receipt_byte_progress_sha256": (
            native_core_recurrent_receipt_byte_authority_progress_v1()[
                "contract_sha256"
            ]
        ),
        "bounded_structural_payload_sha_contract_id": (
            NATIVE_CORE_STRUCTURAL_PAYLOAD_SHA_CONTRACT_ID
        ),
        "bounded_structural_payload_sha_contract_sha256": (
            native_core_ordinary_birth_structural_payload_sha_v1()["contract_sha256"]
        ),
        "remaining_structural_payload_limits": (
            "native_digest_domain_is_rowv1_structural_payload_v1",
            "native_digest_is_little_endian_rowv1_structural_subset",
            "one_route_bound_newborn_fixture_only",
            "one_route_bound_same_slot_fixture_only",
            "not_full_native_payload_byte_extraction",
            "not_bound_to_route_package_qualification",
            "bounded_two_cell_multicell_checkpoint_fixture_only",
            "bounded_b1_generation_turnover_checkpoint_resume_fixture_only",
            "not_route_package_multicell_checkpoint_qualification",
            "not_cuda_authoritative_receipt_abi",
        ),
        "remaining_receipt_byte_limits": (
            "native_route_builds_bounded_canonical_receipt_digest_not_package_qualified",
            "source_and_ancestry_hashes_are_caller_supplied",
            "bounded_one_transaction_ordinary_birth_canonical_builder_only",
            "one_row_v4_checkpoint_project_restore_only",
            "bounded_two_cell_multicell_checkpoint_fixture_is_artifact_bound",
            "bounded_b1_generation_turnover_checkpoint_resume_fixture_only",
            "not_bound_to_route_package_qualification",
            "not_full_route_package_multicell_checkpoint_qualification",
            "not_cuda_authoritative_receipt_abi",
        ),
        "next_required_native_cpu_work": (
            "qualify native-built canonical recurrent receipt construction through "
            "generation-turnover route checkpoint evidence, route-package multicell "
            "qualification, and final package identity"
        ),
        "native_cpu_a_authorized": False,
        "cuda_promotion_authorized": False,
        "successor_campaign_authorized": False,
    }
    payload["contract_sha256"] = native_cpu_progress_sha256_v1(payload)
    return payload


def native_cpu_gate1_turnover_route_package_qualification_v1() -> dict[str, Any]:
    """Promote the route-bound native CPU turnover evidence for P12 Gate 1."""

    route_bound_birth = native_cpu_route_bound_birth_lifecycle_receipt_v1()
    recurrent_receipts = native_core_recurrent_receipt_byte_authority_progress_v1()
    payload = {
        "schema_version": SCHEMA_VERSION,
        "contract_id": NATIVE_CPU_GATE1_ROUTE_QUALIFICATION_CONTRACT_ID,
        "answer_key_contract_id": ANSWER_KEY_CONTRACT_ID,
        "answer_key_contract_sha256": ANSWER_KEY_CONTRACT_SHA256,
        "case_id": "native_cpu_gate1_turnover_route_package_qualification",
        "decision": "P12-NATIVE-GATE1-A",
        "qualified_gate": "native_structural_recurrent_generation_turnover_route_package",
        "qualified_route": "native_cpu_low_parallelism",
        "qualified_envelope": (
            "b1_native_low_parallelism_turnover_checkpoint_route_package_fixture"
        ),
        "artifact_surface": route_bound_birth["artifact_surface"],
        "artifact_sha256": route_bound_birth["artifact_sha256"],
        "build_manifest_surface": route_bound_birth["build_manifest_surface"],
        "build_manifest_sha256": route_bound_birth["build_manifest_sha256"],
        "route_bound_birth_receipt_contract_id": route_bound_birth["contract_id"],
        "route_bound_birth_receipt_sha256": route_bound_birth["contract_sha256"],
        "recurrent_receipt_progress_contract_id": recurrent_receipts["contract_id"],
        "recurrent_receipt_progress_sha256": recurrent_receipts["contract_sha256"],
        "route_bound_symbols": (
            route_bound_birth["canonical_route_bound_symbol"],
            route_bound_birth["same_slot_canonical_route_bound_symbol"],
            route_bound_birth["native_structural_payload_sha_symbol"],
            route_bound_birth["native_canonical_receipt_sha_symbol"],
        ),
        "gate1_required_cases": {
            "accepted_structural_mutation": "qualified_route_bound_v4_structural_payload_sha",
            "accepted_recurrent_mutation": "qualified_native_built_canonical_recurrent_receipt_digest",
            "capacity_rejection": "qualified_v2_capacity_rejected_receipt_bytes",
            "publication_rejection": "qualified_v2_publication_rejected_receipt_bytes",
            "stale_owner_token_rejection": "qualified_owner_rejected_receipt_bytes",
            "disabled_reasons": "qualified_v4_six_disabled_reason_receipt_bytes",
            "rollback": "qualified_digest_mismatch_and_rejection_atomicity",
            "cancellation": "qualified_checkpoint_turnover_cancellation_seal_fixture",
            "terminal_sealing": "qualified_route_receipt_terminal_status_precedence",
            "newborn_publication": "qualified_route_bound_newborn_publication",
            "same_slot_fresh_owner_reuse": "qualified_route_bound_same_slot_reuse",
            "multicell_equal_counts": "qualified_two_cell_equal_checkpoint_collection",
            "multicell_ragged_counts": "qualified_dense_id_route_ragged_carry_forward",
            "high_id_and_slot_reuse": "qualified_high_id_child_and_slot_token_reuse",
        },
        "checkpoint_boundaries": (
            "immediately_before_turnover_prefix_receipt",
            "immediately_after_turnover_prefix_checkpoint",
            "after_first_accepted_post_turnover_transaction",
            "after_first_rejected_post_turnover_transaction",
            "after_multiple_post_turnover_transactions",
            "after_cancellation_at_turnover",
        ),
        "exact_equality_fields": (
            "transaction_sequences_and_ids",
            "native_birth_receipt_bytes",
            "native_structural_payload_sha256",
            "canonical_recurrent_receipt_sha256",
            "canonical_recurrent_receipt_bytes_sha256",
            "per_cell_and_global_collection_hashes",
            "owner_table",
            "sparse_identity_table",
            "checkpoint_lineage",
            "terminal_child_state",
            "artifact_sha256",
            "build_manifest_sha256",
        ),
        "source_and_ancestry_authority": (
            "caller_supplied_hash_words_are_admitted_only_as checked lifecycle "
            "inputs to native canonical receipt construction; they cannot override "
            "native structural payload SHA, child identity, owner slot/token, or "
            "canonical recurrent receipt digest checks"
        ),
        "route_package_qualification": {
            "source_artifact_manifest_bound": True,
            "artifact_hash_match_required": True,
            "build_manifest_source_hash_match_required": True,
            "package_mechanics_mismatch_rejected": True,
            "prior_bounded_only_receipts_invalidated_for_gate1_claim": True,
        },
        "qualified_tests": (
            "tests/test_p12_native_cpu_parity_progress_v1.py",
            "tests/perf/test_eiel_modular_recurrent_cpu_oracle_v1.py::"
            "test_p12_native_cpu_generation_turnover_checkpoint_resume_repeats_v4_birth_receipts",
            "tests/perf/test_eiel_modular_recurrent_cpu_oracle_v1.py::"
            "test_p12_native_cpu_multicell_v4_checkpoint_collection_restores_canonical_receipts",
        ),
        "known_external_open_gates": (
            "cuda_authoritative_rowv1_receipt_abi",
            "compact_cpu_cuda_parity",
            "final_p12_package_route_qualification",
        ),
        "native_cpu_gate1_qualified": True,
        "p12_a_authorized": False,
        "cuda_promotion_authorized": False,
        "successor_campaign_authorized": False,
    }
    payload["contract_sha256"] = native_cpu_progress_sha256_v1(payload)
    return payload


def native_core_two_draw_operator_conformance_v1() -> dict[str, Any]:
    """Record constrained native-core coverage for the four two-draw bins."""

    payload = {
        "schema_version": SCHEMA_VERSION,
        "contract_id": NATIVE_CORE_OPERATOR_CONFORMANCE_CONTRACT_ID,
        "answer_key_contract_id": ANSWER_KEY_CONTRACT_ID,
        "answer_key_contract_sha256": ANSWER_KEY_CONTRACT_SHA256,
        "case_id": "native_core_two_draw_operator_conformance",
        "decision": "P12-NATIVE-CORE-D-CONSTRAINED-CONFORMANCE",
        "native_surface": "engine/perf/native/recurrent/modular_recurrent_v1.cpp",
        "test_binding_surface": (
            "tests/perf/test_eiel_recurrent_native_core_v1_binding.cpp"
        ),
        "test_surface": "tests/perf/test_eiel_recurrent_native_core_v1.py",
        "covered_fields": NATIVE_CORE_OPERATOR_CONFORMANCE_FIELDS,
        "covered_operation_bins": (
            "toggle_dormancy",
            "prune_eligible_module",
            "refine_descendant_parameter",
            "refine_descendant_routing_or_expression",
        ),
        "limits": NATIVE_CORE_OPERATOR_CONFORMANCE_LIMITS,
        "native_cpu_a_authorized": False,
        "cuda_promotion_authorized": False,
        "successor_campaign_authorized": False,
    }
    payload["contract_sha256"] = native_cpu_progress_sha256_v1(payload)
    return payload


def native_core_c_api_two_draw_boundary_v1() -> dict[str, Any]:
    """Record the separate native-core C ABI boundary for two-draw mutation."""

    payload = {
        "schema_version": SCHEMA_VERSION,
        "contract_id": NATIVE_CORE_C_API_BOUNDARY_CONTRACT_ID,
        "answer_key_contract_id": ANSWER_KEY_CONTRACT_ID,
        "answer_key_contract_sha256": ANSWER_KEY_CONTRACT_SHA256,
        "case_id": "native_core_c_api_two_draw_boundary",
        "decision": "P12-NATIVE-CORE-E-C-API-BOUNDARY",
        "native_header": (
            "engine/perf/native/recurrent/modular_recurrent_core_c_api_v1.h"
        ),
        "native_source": (
            "engine/perf/native/recurrent/modular_recurrent_core_c_api_v1.cpp"
        ),
        "test_surface": "tests/perf/test_eiel_recurrent_native_core_c_api_v1.py",
        "exports": NATIVE_CORE_C_API_EXPORTS,
        "covered_operation_bins": (
            "toggle_dormancy",
            "prune_eligible_module",
            "refine_descendant_parameter",
            "refine_descendant_routing_or_expression",
        ),
        "receipt_fields": (
            "status",
            "operation",
            "selected_slot",
            "changed_dimension",
            "changed",
            "digest_before",
            "digest_after",
        ),
        "limits": (
            "opaque_fixed_envelope_native_core_boundary_only",
            "float32_draw_core_not_float64_receipt_boundary",
            "no_python_json_payload_sha256",
            "no_canonical_modular_two_draws_receipt",
            "not_the_production_recurrent_rowv1_artifact",
            "no_ordinary_birth_owner_publication",
            "no_cuda_promotion",
        ),
        "native_cpu_a_authorized": False,
        "cuda_promotion_authorized": False,
        "successor_campaign_authorized": False,
    }
    payload["contract_sha256"] = native_cpu_progress_sha256_v1(payload)
    return payload


def native_core_ordinary_birth_transaction_boundary_v1() -> dict[str, Any]:
    """Record native-core ordinary-birth transaction evidence without promotion."""

    payload = {
        "schema_version": SCHEMA_VERSION,
        "contract_id": NATIVE_CORE_ORDINARY_BIRTH_BOUNDARY_CONTRACT_ID,
        "answer_key_contract_id": ANSWER_KEY_CONTRACT_ID,
        "answer_key_contract_sha256": ANSWER_KEY_CONTRACT_SHA256,
        "case_id": "native_core_ordinary_birth_transaction_boundary",
        "decision": "P12-NATIVE-CORE-F-ORDINARY-BIRTH-BOUNDARY",
        "native_header": (
            "engine/perf/native/recurrent/modular_recurrent_core_c_api_v1.h"
        ),
        "native_source": (
            "engine/perf/native/recurrent/modular_recurrent_core_c_api_v1.cpp"
        ),
        "test_surface": "tests/perf/test_eiel_recurrent_native_core_c_api_v1.py",
        "exports": NATIVE_CORE_ORDINARY_BIRTH_EXPORTS,
        "transaction_statuses": (
            "committed",
            "capacity_unavailable",
            "publication_rejected",
            "owner_mismatch",
            "invalid_draw",
            "native_mutation_rejected",
        ),
        "covered_fields": (
            "parent_agent_id",
            "parent_slot",
            "parent_slot_token",
            "child_agent_id",
            "child_slot",
            "expected_child_slot_token",
            "child_publication_token",
            "draw0_float64_bits",
            "draw1_float64_bits",
            "capacity_available",
            "publish_success",
            "proposal_digest",
            "child_digest_after",
        ),
        "covered_behaviors": (
            "exact_binary64_draw_bit_ledger",
            "owner_token_staleness_rejection",
            "capacity_unavailable_rollback",
            "publication_rejected_rollback_with_proposal_digest",
            "invalid_draw_rollback",
            "committed_child_publication_digest",
        ),
        "limits": (
            "opaque_fixed_envelope_native_core_boundary_only",
            "not_the_production_recurrent_rowv1_artifact",
            "no_python_json_payload_sha256",
            "no_canonical_modular_two_draws_receipt",
            "no_structural_payload_sha256",
            "no_cuda_promotion",
            "no_route_package_qualification",
        ),
        "native_cpu_a_authorized": False,
        "cuda_promotion_authorized": False,
        "successor_campaign_authorized": False,
    }
    payload["contract_sha256"] = native_cpu_progress_sha256_v1(payload)
    return payload


def native_core_ordinary_birth_recurrent_receipt_bridge_v1() -> dict[str, Any]:
    """Record canonical recurrent receipt emission from native-core output."""

    payload = {
        "schema_version": SCHEMA_VERSION,
        "contract_id": NATIVE_CORE_RECURRENT_RECEIPT_BRIDGE_CONTRACT_ID,
        "answer_key_contract_id": ANSWER_KEY_CONTRACT_ID,
        "answer_key_contract_sha256": ANSWER_KEY_CONTRACT_SHA256,
        "case_id": "native_core_ordinary_birth_recurrent_receipt_bridge",
        "decision": "P12-NATIVE-CORE-G-RECURRENT-RECEIPT-BRIDGE",
        "native_source": (
            "engine/perf/native/recurrent/modular_recurrent_core_c_api_v1.cpp"
        ),
        "bridge_surface": (
            "engine/perf/recurrent_native/"
            "native_core_ordinary_birth_receipt_bridge_v1.py"
        ),
        "canonical_receipt_surface": (
            "engine/perf/recurrent_native/modular_recurrent_mutation_receipt_v1.py"
        ),
        "test_surface": "tests/perf/test_eiel_recurrent_native_core_c_api_v1.py",
        "covered_receipt_abi": "modular_two_draws_v1",
        "covered_behaviors": (
            "real_native_c_api_committed_birth_to_canonical_recurrent_receipt",
            "real_native_c_api_capacity_rejection_to_canonical_recurrent_receipt",
            "draw0_operation_consistency_for_rejected_receipts",
            "exact_binary64_draw_hex_preservation",
            "high_id_child_agent_id_preservation",
            "child_publication_token_as_child_slot_token",
        ),
        "limits": (
            "native_core_fixed_envelope_only",
            "bridge_emits_canonical_receipt_but_not_native_rowv1_artifact",
            "reason_strings_are_native_bridge_reasons_not_python_oracle_reasons",
            "selected_identity_indexes_are_supplied_by_caller",
            "no_structural_payload_sha256",
            "no_python_oracle_payload_equality",
            "no_cuda_promotion",
            "no_route_package_qualification",
        ),
        "native_cpu_a_authorized": False,
        "cuda_promotion_authorized": False,
        "successor_campaign_authorized": False,
    }
    payload["contract_sha256"] = native_cpu_progress_sha256_v1(payload)
    return payload


def native_core_ordinary_birth_payload_observability_v1() -> dict[str, Any]:
    """Record one bounded native/Python semantic payload comparison fixture."""

    payload = {
        "schema_version": SCHEMA_VERSION,
        "contract_id": NATIVE_CORE_PAYLOAD_OBSERVABILITY_CONTRACT_ID,
        "answer_key_contract_id": ANSWER_KEY_CONTRACT_ID,
        "answer_key_contract_sha256": ANSWER_KEY_CONTRACT_SHA256,
        "case_id": "native_core_ordinary_birth_payload_observability",
        "decision": "P12-NATIVE-CORE-H-PAYLOAD-OBSERVABILITY",
        "native_source": (
            "engine/perf/native/recurrent/modular_recurrent_v1.cpp"
        ),
        "native_c_api_surface": (
            "engine/perf/native/recurrent/modular_recurrent_core_c_api_v1.cpp"
        ),
        "python_oracle_surface": "engine/evolution/p12_stage2_mutation_oracles_v1.py",
        "test_surface": "tests/perf/test_eiel_recurrent_native_core_c_api_v1.py",
        "covered_fixture": (
            "one duplicate-descendant integrator refine_descendant_parameter "
            "ordinary-birth success with draw0=0.625 and draw1=0.5"
        ),
        "covered_fields": (
            "operation",
            "changed_dimension",
            "draw0_float64_bits",
            "draw1_float64_bits",
            "parent_owner",
            "child_owner",
            "selected_native_module_instance_id",
            "selected_native_homology_id",
            "python_selected_module_instance_id_present",
            "python_selected_homology_id_present",
            "value_before_float32",
            "value_after_float32",
        ),
        "limits": (
            "one_semantic_payload_fixture_only",
            "native_numeric_identity_namespace_not_equal_to_python_hash_identity_namespace",
            "no_structural_payload_sha256_equality",
            "no_python_oracle_receipt_byte_equality",
            "not_the_production_recurrent_rowv1_artifact",
            "no_cuda_promotion",
            "no_route_package_qualification",
        ),
        "native_cpu_a_authorized": False,
        "cuda_promotion_authorized": False,
        "successor_campaign_authorized": False,
    }
    payload["contract_sha256"] = native_cpu_progress_sha256_v1(payload)
    return payload


def native_core_ordinary_birth_identity_mapping_bridge_v1() -> dict[str, Any]:
    """Record explicit native numeric ID to Python identity table mapping."""

    payload = {
        "schema_version": SCHEMA_VERSION,
        "contract_id": NATIVE_CORE_IDENTITY_MAPPING_CONTRACT_ID,
        "answer_key_contract_id": ANSWER_KEY_CONTRACT_ID,
        "answer_key_contract_sha256": ANSWER_KEY_CONTRACT_SHA256,
        "case_id": "native_core_ordinary_birth_identity_mapping_bridge",
        "decision": "P12-NATIVE-CORE-I-IDENTITY-MAPPING-BRIDGE",
        "bridge_surface": (
            "engine/perf/recurrent_native/"
            "native_core_ordinary_birth_receipt_bridge_v1.py"
        ),
        "test_surface": "tests/perf/test_eiel_recurrent_native_core_c_api_v1.py",
        "covered_behaviors": (
            "derive_selected_instance_id_index_from_identity_table",
            "derive_selected_homology_id_index_from_identity_table",
            "fail_closed_when_native_module_mapping_missing",
            "fail_closed_when_native_homology_mapping_missing",
            "preserve_canonical_modular_two_draws_v1_validation",
        ),
        "limits": (
            "mapping_is_explicit_fixture_binding_not_universal_identity_authority",
            "no_structural_payload_sha256_equality",
            "no_python_oracle_receipt_byte_equality",
            "not_the_production_recurrent_rowv1_artifact",
            "no_cuda_promotion",
            "no_route_package_qualification",
        ),
        "native_cpu_a_authorized": False,
        "cuda_promotion_authorized": False,
        "successor_campaign_authorized": False,
    }
    payload["contract_sha256"] = native_cpu_progress_sha256_v1(payload)
    return payload


def native_core_ordinary_birth_payload_projection_v1() -> dict[str, Any]:
    """Record one mapped native receipt to Python payload SHA equality fixture."""

    payload = {
        "schema_version": SCHEMA_VERSION,
        "contract_id": NATIVE_CORE_PAYLOAD_PROJECTION_CONTRACT_ID,
        "answer_key_contract_id": ANSWER_KEY_CONTRACT_ID,
        "answer_key_contract_sha256": ANSWER_KEY_CONTRACT_SHA256,
        "case_id": "native_core_ordinary_birth_payload_projection",
        "decision": "P12-NATIVE-CORE-J-PAYLOAD-PROJECTION",
        "bridge_surface": (
            "engine/perf/recurrent_native/"
            "native_core_ordinary_birth_receipt_bridge_v1.py"
        ),
        "python_oracle_surface": "engine/evolution/p12_stage2_mutation_oracles_v1.py",
        "test_surface": "tests/perf/test_eiel_recurrent_native_core_c_api_v1.py",
        "covered_behaviors": (
            "project_python_payload_from_mapped_native_core_receipt",
            "compare_projected_payload_to_python_oracle_child",
            "compare_projected_payload_sha256_to_python_oracle_child_payload_sha256",
            "one_refine_descendant_parameter_live_input_gain_success_case",
        ),
        "limits": (
            "one_payload_projection_fixture_only",
            "projection_uses_explicit_identity_mapping",
            "not_full_native_payload_byte_extraction",
            "no_python_oracle_receipt_byte_equality",
            "not_the_production_recurrent_rowv1_artifact",
            "no_cuda_promotion",
            "no_route_package_qualification",
        ),
        "native_cpu_a_authorized": False,
        "cuda_promotion_authorized": False,
        "successor_campaign_authorized": False,
    }
    payload["contract_sha256"] = native_cpu_progress_sha256_v1(payload)
    return payload


def native_core_ordinary_birth_structural_payload_sha_v1() -> dict[str, Any]:
    """Record bounded native-core structural payload SHA equality evidence."""

    payload = {
        "schema_version": SCHEMA_VERSION,
        "contract_id": NATIVE_CORE_STRUCTURAL_PAYLOAD_SHA_CONTRACT_ID,
        "answer_key_contract_id": ANSWER_KEY_CONTRACT_ID,
        "answer_key_contract_sha256": ANSWER_KEY_CONTRACT_SHA256,
        "case_id": "native_core_ordinary_birth_structural_payload_sha",
        "decision": "P12-NATIVE-CORE-V-STRUCTURAL-PAYLOAD-SHA-PROGRESS",
        "native_surface": (
            "engine/perf/native/recurrent/modular_recurrent_core_c_api_v1.cpp"
        ),
        "bridge_surface": (
            "engine/perf/recurrent_native/"
            "native_core_ordinary_birth_receipt_bridge_v1.py"
        ),
        "python_oracle_surface": "engine/evolution/p12_stage2_mutation_oracles_v1.py",
        "test_surface": "tests/perf/test_eiel_recurrent_native_core_c_api_v1.py",
        "payload_projection_contract_id": NATIVE_CORE_PAYLOAD_PROJECTION_CONTRACT_ID,
        "covered_fixture": {
            "operation": "refine_descendant_parameter",
            "draw0": 0.625,
            "draw1": 0.5,
            "parent_agent_id": 100,
            "parent_slot": 2,
            "parent_slot_token": 12,
            "child_agent_id": (1 << 40) + 7,
            "child_slot": 4,
            "child_slot_token": 18,
        },
        "covered_behaviors": (
            "project_python_payload_from_mapped_native_core_receipt",
            "compare_projected_payload_to_python_oracle_child",
            "compare_projected_payload_sha256_to_python_oracle_child_payload_sha256",
            "preserve_high_id_child_owner_in_python_oracle_receipt",
            "fail_closed_if_explicit_identity_mapping_is_missing",
        ),
        "limits": (
            "native_core_fixed_envelope_only",
            "one_structural_payload_sha_fixture_only",
            "projection_uses_explicit_identity_mapping",
            "not_full_native_payload_byte_extraction",
            "not_the_production_recurrent_lifecycle_c_api_route",
            "not_the_production_recurrent_rowv1_artifact",
            "not_checkpoint_resume_repeat_determinism",
            "not_cuda_authoritative_receipt_abi",
            "no_route_package_qualification",
        ),
        "native_cpu_a_authorized": False,
        "cuda_promotion_authorized": False,
        "successor_campaign_authorized": False,
    }
    payload["contract_sha256"] = native_cpu_progress_sha256_v1(payload)
    return payload


def native_core_ordinary_birth_v2_receipt_surface_v1() -> dict[str, Any]:
    """Record the explicit v2 native-core ordinary-birth receipt ABI surface."""

    payload = {
        "schema_version": SCHEMA_VERSION,
        "contract_id": NATIVE_CORE_ORDINARY_BIRTH_V2_RECEIPT_SURFACE_CONTRACT_ID,
        "answer_key_contract_id": ANSWER_KEY_CONTRACT_ID,
        "answer_key_contract_sha256": ANSWER_KEY_CONTRACT_SHA256,
        "case_id": "native_core_ordinary_birth_v2_receipt_surface",
        "decision": "P12-NATIVE-CORE-K-V2-RECEIPT-SURFACE",
        "native_header": (
            "engine/perf/native/recurrent/modular_recurrent_core_c_api_v1.h"
        ),
        "native_source": (
            "engine/perf/native/recurrent/modular_recurrent_core_c_api_v1.cpp"
        ),
        "test_surface": "tests/perf/test_eiel_recurrent_native_core_c_api_v1.py",
        "exports": NATIVE_CORE_ORDINARY_BIRTH_V2_EXPORTS,
        "layout_version": 2,
        "covered_receipt_fields": (
            "transaction_status",
            "native_status",
            "operation",
            "changed_dimension",
            "changed",
            "committed",
            "parent_agent_id",
            "parent_slot",
            "parent_slot_token",
            "child_agent_id",
            "child_slot",
            "child_publication_token",
            "draw0_float64_bits",
            "draw1_float64_bits",
            "proposal_digest",
            "child_digest_after",
            "selected_module_instance_id",
            "selected_homology_id",
            "value_before_float32",
            "value_after_float32",
        ),
        "covered_behaviors": (
            "explicit_receipt_layout_version_introspection",
            "native_receipt_size_and_alignment_introspection",
            "v2_symbol_preserves_exact_float64_draw_bits",
            "v2_symbol_preserves_high_id_child_owner_fields",
            "v2_symbol_preserves_selected_identity_and_value_delta_fields",
        ),
        "limits": (
            "native_core_fixed_envelope_only",
            "v2_surface_aliases_existing_native_core_mutation_semantics",
            "not_the_production_recurrent_rowv1_artifact",
            "no_structural_payload_sha256_equality",
            "no_python_oracle_receipt_byte_equality",
            "no_cuda_promotion",
            "no_route_package_qualification",
        ),
        "native_cpu_a_authorized": False,
        "cuda_promotion_authorized": False,
        "successor_campaign_authorized": False,
    }
    payload["contract_sha256"] = native_cpu_progress_sha256_v1(payload)
    return payload


def native_core_ordinary_birth_oracle_aligned_receipt_bytes_v1() -> dict[str, Any]:
    """Record one native v2 receipt matching Python oracle replay bytes."""

    payload = {
        "schema_version": SCHEMA_VERSION,
        "contract_id": NATIVE_CORE_ORACLE_ALIGNED_RECEIPT_BYTES_CONTRACT_ID,
        "answer_key_contract_id": ANSWER_KEY_CONTRACT_ID,
        "answer_key_contract_sha256": ANSWER_KEY_CONTRACT_SHA256,
        "case_id": "native_core_ordinary_birth_oracle_aligned_receipt_bytes",
        "decision": "P12-NATIVE-CORE-L-ORACLE-ALIGNED-RECEIPT-BYTES",
        "native_v2_receipt_contract_id": (
            NATIVE_CORE_ORDINARY_BIRTH_V2_RECEIPT_SURFACE_CONTRACT_ID
        ),
        "bridge_surface": (
            "engine/perf/recurrent_native/"
            "native_core_ordinary_birth_receipt_bridge_v1.py"
        ),
        "python_oracle_surface": "engine/evolution/p12_stage2_mutation_oracles_v1.py",
        "canonical_receipt_surface": (
            "engine/perf/recurrent_native/modular_recurrent_mutation_receipt_v1.py"
        ),
        "test_surface": "tests/perf/test_eiel_recurrent_native_core_c_api_v1.py",
        "covered_fixtures": (
            {
                "operation": "toggle_dormancy",
                "draw0": 0.125,
                "draw1": 0.0,
                "selected_native_module_instance_id": 1,
                "selected_native_homology_id": 100,
            },
            {
                "operation": "prune_eligible_module",
                "draw0": 0.375,
                "draw1": 0.0,
                "selected_native_module_instance_id": 2,
                "selected_native_homology_id": 200,
            },
            {
                "operation": "refine_descendant_parameter",
                "draw0": 0.625,
                "draw1": 0.5,
                "selected_native_module_instance_id": 2,
                "selected_native_homology_id": 200,
            },
            {
                "operation": "refine_descendant_routing_or_expression",
                "draw0": 0.875,
                "draw1": 0.0,
                "selected_native_module_instance_id": 2,
                "selected_native_homology_id": 200,
            },
        ),
        "covered_behaviors": (
            "build_python_oracle_aligned_recurrent_receipt_from_mapped_native_core",
            "compare_native_replay_receipt_sha256_to_python_replay_receipt_sha256",
            "compare_native_replay_receipt_canonical_bytes_to_python_replay_bytes",
            "bind_python_oracle_profile_b2_16x48_v1",
            "preserve_selected_identity_indexes_from_full_fixture_native_mapping",
        ),
        "limits": (
            "four_accepted_receipt_byte_fixtures_only",
            "explicit_identity_mapping_required",
            "accepted_reason_is_oracle_aligned_metadata",
            "not_full_native_payload_byte_extraction",
            "not_all_v1_1_parity_matrix_cases",
            "not_the_production_recurrent_rowv1_artifact",
            "no_cuda_promotion",
            "no_route_package_qualification",
        ),
        "native_cpu_a_authorized": False,
        "cuda_promotion_authorized": False,
        "successor_campaign_authorized": False,
    }
    payload["contract_sha256"] = native_cpu_progress_sha256_v1(payload)
    return payload


def native_core_ordinary_birth_rejected_receipt_bytes_v1() -> dict[str, Any]:
    """Record bounded rejected ordinary-birth receipt-byte parity coverage."""

    payload = {
        "schema_version": SCHEMA_VERSION,
        "contract_id": NATIVE_CORE_REJECTED_RECEIPT_BYTES_CONTRACT_ID,
        "answer_key_contract_id": ANSWER_KEY_CONTRACT_ID,
        "answer_key_contract_sha256": ANSWER_KEY_CONTRACT_SHA256,
        "case_id": "native_core_ordinary_birth_rejected_receipt_bytes",
        "decision": "P12-NATIVE-CORE-M-REJECTED-RECEIPT-BYTES",
        "native_v2_receipt_contract_id": (
            NATIVE_CORE_ORDINARY_BIRTH_V2_RECEIPT_SURFACE_CONTRACT_ID
        ),
        "bridge_surface": (
            "engine/perf/recurrent_native/"
            "native_core_ordinary_birth_receipt_bridge_v1.py"
        ),
        "python_oracle_surface": "engine/evolution/p12_stage2_mutation_oracles_v1.py",
        "canonical_receipt_surface": (
            "engine/perf/recurrent_native/modular_recurrent_mutation_receipt_v1.py"
        ),
        "native_surface": (
            "engine/perf/native/recurrent/modular_recurrent_core_c_api_v1.cpp"
        ),
        "test_surface": "tests/perf/test_eiel_recurrent_native_core_c_api_v1.py",
        "covered_fixtures": (
            {
                "case": "capacity_unavailable",
                "draw0": 0.625,
                "draw1": 0.5,
                "reason": "capacity_unavailable",
                "expected_changed": False,
            },
            {
                "case": "publication_failure",
                "draw0": 0.625,
                "draw1": 0.5,
                "reason": "child_publication_failed",
                "expected_changed": False,
                "preserves_proposal_digest": True,
            },
        ),
        "covered_behaviors": (
            "compare_rejected_native_replay_receipt_sha256_to_python_replay_sha256",
            "compare_rejected_native_replay_receipt_bytes_to_python_replay_bytes",
            "capacity_unavailable_rejects_before_proposal",
            "publication_failure_clears_changed_and_selected_identity_fields",
            "publication_failure_preserves_proposal_digest_for_observability",
        ),
        "limits": (
            "two_rejected_receipt_byte_fixtures_only",
            "fixed_draw_refine_descendant_parameter_rejected_cases_only",
            "not_stale_owner_token_receipt_byte_parity",
            "not_invalid_draw_receipt_byte_parity",
            "not_disabled_mutation_receipt_byte_parity",
            "not_the_production_recurrent_rowv1_artifact",
            "no_cuda_promotion",
            "no_route_package_qualification",
        ),
        "native_cpu_a_authorized": False,
        "cuda_promotion_authorized": False,
        "successor_campaign_authorized": False,
    }
    payload["contract_sha256"] = native_cpu_progress_sha256_v1(payload)
    return payload


def native_core_ordinary_birth_owner_rejection_receipt_bytes_v1() -> dict[str, Any]:
    """Record bounded owner-token rejection receipt-byte parity coverage."""

    payload = {
        "schema_version": SCHEMA_VERSION,
        "contract_id": NATIVE_CORE_OWNER_REJECTION_RECEIPT_BYTES_CONTRACT_ID,
        "answer_key_contract_id": ANSWER_KEY_CONTRACT_ID,
        "answer_key_contract_sha256": ANSWER_KEY_CONTRACT_SHA256,
        "case_id": "native_core_ordinary_birth_owner_rejection_receipt_bytes",
        "decision": "P12-NATIVE-CORE-N-OWNER-REJECTION-RECEIPT-BYTES",
        "native_v2_receipt_contract_id": (
            NATIVE_CORE_ORDINARY_BIRTH_V2_RECEIPT_SURFACE_CONTRACT_ID
        ),
        "bridge_surface": (
            "engine/perf/recurrent_native/"
            "native_core_ordinary_birth_receipt_bridge_v1.py"
        ),
        "canonical_receipt_surface": (
            "engine/perf/recurrent_native/modular_recurrent_mutation_receipt_v1.py"
        ),
        "native_surface": (
            "engine/perf/native/recurrent/modular_recurrent_core_c_api_v1.cpp"
        ),
        "test_surface": "tests/perf/test_eiel_recurrent_native_core_c_api_v1.py",
        "covered_fixtures": (
            {
                "case": "child_publication_token_not_newer_than_expected",
                "reason": "stale_slot_token",
                "draw0": 0.625,
                "draw1": 0.5,
            },
            {
                "case": "same_slot_publication_token_not_newer_than_parent",
                "reason": "stale_slot_token",
                "draw0": 0.625,
                "draw1": 0.5,
            },
            {
                "case": "invalid_parent_slot_token",
                "reason": "stale_slot_token",
                "draw0": 0.625,
                "draw1": 0.5,
            },
        ),
        "covered_behaviors": (
            "compare_owner_rejected_native_replay_receipt_sha256_to_expected_sha256",
            "compare_owner_rejected_native_replay_receipt_bytes_to_expected_bytes",
            "owner_rejection_records_draw_selected_operation_for_valid_draws",
            "owner_rejection_clears_selected_identity_and_changed_fields",
            "owner_rejection_preserves_child_state_and_omits_proposal_digest",
        ),
        "limits": (
            "three_native_core_representable_owner_rejection_fixtures_only",
            "not_dead_occupant_token_lifecycle_semantics",
            "not_reused_slot_with_newer_token_lifecycle_continuation",
            "not_checkpoint_visible_allocator_semantics",
            "not_invalid_draw_receipt_byte_parity",
            "not_disabled_mutation_receipt_byte_parity",
            "not_the_production_recurrent_rowv1_artifact",
            "no_cuda_promotion",
            "no_route_package_qualification",
        ),
        "native_cpu_a_authorized": False,
        "cuda_promotion_authorized": False,
        "successor_campaign_authorized": False,
    }
    payload["contract_sha256"] = native_cpu_progress_sha256_v1(payload)
    return payload


def native_core_ordinary_birth_disabled_receipt_bytes_v1() -> dict[str, Any]:
    """Record bounded disabled ordinary-birth receipt-byte parity coverage."""

    payload = {
        "schema_version": SCHEMA_VERSION,
        "contract_id": NATIVE_CORE_DISABLED_RECEIPT_BYTES_CONTRACT_ID,
        "answer_key_contract_id": ANSWER_KEY_CONTRACT_ID,
        "answer_key_contract_sha256": ANSWER_KEY_CONTRACT_SHA256,
        "case_id": "native_core_ordinary_birth_disabled_receipt_bytes",
        "decision": "P12-NATIVE-CORE-P-DISABLED-RECEIPT-BYTES",
        "native_v3_symbol": (
            "eiel_modular_recurrent_core_ordinary_birth_two_draws_v3"
        ),
        "bridge_surface": (
            "engine/perf/recurrent_native/"
            "native_core_ordinary_birth_receipt_bridge_v1.py"
        ),
        "canonical_receipt_surface": (
            "engine/perf/recurrent_native/modular_recurrent_mutation_receipt_v1.py"
        ),
        "native_surface": (
            "engine/perf/native/recurrent/modular_recurrent_core_c_api_v1.cpp"
        ),
        "test_surface": "tests/perf/test_eiel_recurrent_native_core_c_api_v1.py",
        "covered_fixtures": (
            {
                "case": "disabled_operator_flag_off",
                "reason": "operator_disabled",
                "draw0": 0.625,
                "draw1": 0.5,
                "mutation_enabled": False,
                "expected_changed": False,
            },
        ),
        "covered_behaviors": (
            "v3_explicit_mutation_enabled_flag",
            "compare_disabled_native_replay_receipt_sha256_to_expected_sha256",
            "compare_disabled_native_replay_receipt_bytes_to_expected_bytes",
            "disabled_receipt_records_draw_selected_operation_for_valid_draws",
            "disabled_receipt_clears_selected_identity_and_changed_fields",
            "disabled_receipt_preserves_child_state_and_omits_proposal_digest",
        ),
        "defined_semantics": (
            "capacity_rejection_precedes_disabled_for_v3",
            "valid_owner_and_valid_draws_required_before_disabled_receipt",
            "disabled_operator_consumes_draw_bits_as_receipt_ledger_fields",
            "disabled_operator_does_not_construct_proposal_digest",
        ),
        "limits": (
            "operator_flag_off_disabled_fixture_only",
            "not_structural_mutation_disabled_fixture",
            "not_recurrent_mutation_disabled_fixture",
            "not_operator_family_disabled_fixture",
            "not_zero_mutation_rate_fixture",
            "not_mutation_attempt_budget_zero_fixture",
            "not_invalid_draw_receipt_byte_parity",
            "not_the_production_recurrent_rowv1_artifact",
            "no_cuda_promotion",
            "no_route_package_qualification",
        ),
        "native_cpu_a_authorized": False,
        "cuda_promotion_authorized": False,
        "successor_campaign_authorized": False,
    }
    payload["contract_sha256"] = native_cpu_progress_sha256_v1(payload)
    return payload


def native_core_ordinary_birth_disabled_reason_receipt_bytes_v1() -> dict[str, Any]:
    """Record v4 disabled reason receipt-byte parity coverage."""

    payload = {
        "schema_version": SCHEMA_VERSION,
        "contract_id": NATIVE_CORE_DISABLED_REASON_RECEIPT_BYTES_CONTRACT_ID,
        "answer_key_contract_id": ANSWER_KEY_CONTRACT_ID,
        "answer_key_contract_sha256": ANSWER_KEY_CONTRACT_SHA256,
        "case_id": "native_core_ordinary_birth_disabled_reason_receipt_bytes",
        "decision": "P12-NATIVE-CORE-S-DISABLED-REASON-RECEIPT-BYTES",
        "native_v4_symbol": (
            "eiel_modular_recurrent_core_ordinary_birth_two_draws_v4"
        ),
        "bridge_surface": (
            "engine/perf/recurrent_native/"
            "native_core_ordinary_birth_receipt_bridge_v1.py"
        ),
        "canonical_receipt_surface": (
            "engine/perf/recurrent_native/modular_recurrent_mutation_receipt_v1.py"
        ),
        "native_surface": (
            "engine/perf/native/recurrent/modular_recurrent_core_c_api_v1.cpp"
        ),
        "test_surface": "tests/perf/test_eiel_recurrent_native_core_c_api_v1.py",
        "covered_fixtures": (
            {"case": "operator_disabled", "disabled_reason_code": 1},
            {"case": "structural_mutation_disabled", "disabled_reason_code": 2},
            {"case": "recurrent_mutation_disabled", "disabled_reason_code": 3},
            {"case": "operator_family_disabled", "disabled_reason_code": 4},
            {"case": "zero_mutation_rate", "disabled_reason_code": 5},
            {"case": "mutation_attempt_budget_zero", "disabled_reason_code": 6},
        ),
        "covered_behaviors": (
            "v4_append_only_disabled_reason_code_receipt_field",
            "compare_disabled_reason_native_replay_bytes_to_expected_bytes",
            "disabled_reason_code_maps_to_canonical_disabled_reason",
            "disabled_reason_receipts_clear_selected_identity_and_changed_fields",
            "disabled_reason_receipts_preserve_child_state_and_omit_proposal_digest",
            "invalid_draw_precedes_disabled_reason",
        ),
        "defined_semantics": (
            "zero_disabled_reason_delegates_to_enabled_v2_path",
            "capacity_rejection_precedes_disabled_reason_for_v4",
            "valid_owner_and_valid_draws_required_before_disabled_reason_receipt",
            "disabled_reason_receipts_consume_draw_bits_as_receipt_ledger_fields",
            "disabled_reason_receipts_do_not_construct_proposal_digest",
        ),
        "limits": (
            "native_core_v4_disabled_reason_fixtures_only",
            "not_native_mutation_rejected_vector",
            "not_runtime_rng_index_table",
            "not_checkpoint_resume_repeat_determinism",
            "not_the_production_recurrent_rowv1_artifact",
            "no_cuda_promotion",
            "no_route_package_qualification",
        ),
        "native_cpu_a_authorized": False,
        "cuda_promotion_authorized": False,
        "successor_campaign_authorized": False,
    }
    payload["contract_sha256"] = native_cpu_progress_sha256_v1(payload)
    return payload


def native_core_ordinary_birth_disabled_reason_semantics_v1() -> dict[str, Any]:
    """Freeze canonical disabled reason semantics and terminal precedence."""

    payload = {
        "schema_version": SCHEMA_VERSION,
        "contract_id": NATIVE_CORE_DISABLED_REASON_SEMANTICS_CONTRACT_ID,
        "answer_key_contract_id": ANSWER_KEY_CONTRACT_ID,
        "answer_key_contract_sha256": ANSWER_KEY_CONTRACT_SHA256,
        "case_id": "native_core_ordinary_birth_disabled_reason_semantics",
        "decision": "P12-NATIVE-CORE-T-DISABLED-REASON-SEMANTICS",
        "native_v4_symbol": (
            "eiel_modular_recurrent_core_ordinary_birth_two_draws_v4"
        ),
        "bridge_surface": (
            "engine/perf/recurrent_native/"
            "native_core_ordinary_birth_receipt_bridge_v1.py"
        ),
        "canonical_receipt_surface": (
            "engine/perf/recurrent_native/modular_recurrent_mutation_receipt_v1.py"
        ),
        "canonical_disabled_reason_enum": tuple(
            {
                "order": index,
                "native_code": code,
                "canonical_reason": DISABLED_REASON_BY_CODE[code],
                "nullable": False,
            }
            for index, code in enumerate(DISABLED_REASON_CODE_ORDER)
        ),
        "nullability": {
            "v4_disabled_receipt": "disabled_reason_code_required_1_to_6",
            "v4_non_disabled_receipt": "disabled_reason_code_zero",
            "v3_disabled_receipt": "disabled_reason_code_absent_defaults_operator_disabled",
            "malformed_input_envelope": "disabled_reason_absent",
        },
        "unknown_code_behavior": (
            "native_v4_entrypoint_rejects_codes_outside_0_to_6",
            "bridge_reader_fails_closed_for_unknown_disabled_reason_code",
            "bridge_reader_fails_closed_for_negative_or_non_int_disabled_reason_code",
            "readers_must_not_infer_missing_v4_reason_from_unrelated_fields",
        ),
        "terminal_precedence": ORDINARY_BIRTH_TERMINAL_PRECEDENCE_V1,
        "covered_behaviors": (
            "six_disabled_reasons_have_stable_native_integer_codes",
            "canonical_string_values_are_stable",
            "canonical_order_matches_native_code_order",
            "invalid_draw_precedes_disabled_reason",
            "owner_rejection_precedes_disabled_reason",
            "capacity_rejection_precedes_disabled_reason",
            "disabled_receipts_do_not_construct_proposal_digest",
            "publication_rejection_is_not_a_disabled_reason",
        ),
        "limits": (
            "semantics_freeze_only_not_rowv1_projection",
            "not_native_version_reconciliation_table",
            "not_checkpoint_resume_repeat_determinism",
            "not_cuda_array_layout_resolution",
            "no_route_package_qualification",
        ),
        "native_cpu_a_authorized": False,
        "cuda_promotion_authorized": False,
        "successor_campaign_authorized": False,
    }
    payload["contract_sha256"] = native_cpu_progress_sha256_v1(payload)
    return payload


def native_core_ordinary_birth_invalid_input_receipt_v1() -> dict[str, Any]:
    """Record bounded native invalid-input receipt coverage."""

    payload = {
        "schema_version": SCHEMA_VERSION,
        "contract_id": NATIVE_CORE_INVALID_INPUT_RECEIPT_CONTRACT_ID,
        "answer_key_contract_id": ANSWER_KEY_CONTRACT_ID,
        "answer_key_contract_sha256": ANSWER_KEY_CONTRACT_SHA256,
        "case_id": "native_core_ordinary_birth_invalid_input_receipt",
        "decision": "P12-NATIVE-CORE-Q-INVALID-INPUT-RECEIPT",
        "invalid_input_receipt_contract_id": "modular_recurrent_invalid_input_v1",
        "invalid_input_receipt_surface": (
            "engine/perf/recurrent_native/"
            "modular_recurrent_invalid_input_receipt_v1.py"
        ),
        "native_bridge_surface": (
            "engine/perf/recurrent_native/"
            "native_core_ordinary_birth_receipt_bridge_v1.py"
        ),
        "native_surface": (
            "engine/perf/native/recurrent/modular_recurrent_core_c_api_v1.cpp"
        ),
        "test_surfaces": (
            "tests/perf/test_eiel_recurrent_native_core_c_api_v1.py",
            "tests/test_modular_recurrent_invalid_input_receipt_v1.py",
        ),
        "covered_fixtures": (
            {
                "case": "native_invalid_draw_out_of_range",
                "input_kind": "native_core_ordinary_birth",
                "draw0_float64_bits": "3ff0000000000000",
                "draw0_present": True,
                "draw1_float64_bits": "3fe0000000000000",
                "draw1_present": True,
                "reason": "invalid_draw",
                "native_transaction_status": 4,
                "native_status": 7,
            },
            {
                "case": "missing_draw",
                "input_kind": "canonical_replay_receipt",
                "draw0_present": True,
                "draw1_present": False,
                "reason": "missing_draw",
            },
            {
                "case": "extra_draw",
                "input_kind": "canonical_replay_receipt",
                "draw0_present": True,
                "draw1_present": True,
                "reason": "extra_draw",
            },
            {
                "case": "operation_draw_mismatch",
                "input_kind": "canonical_replay_receipt",
                "operation": "prune_eligible_module",
                "reason": "operation_draw_mismatch",
            },
            {
                "case": "selected_index_out_of_range",
                "input_kind": "canonical_replay_receipt",
                "operation": "refine_descendant_parameter",
                "reason": "selected_index_out_of_range",
            },
            {
                "case": "malformed_replay_receipt",
                "input_kind": "canonical_replay_receipt",
                "reason": "malformed_replay_receipt",
            },
            {
                "case": "changed_replay_digest",
                "input_kind": "canonical_replay_receipt",
                "operation": "refine_descendant_parameter",
                "reason": "changed_replay_digest",
            },
            {
                "case": "duplicate_replay_field",
                "input_kind": "canonical_replay_receipt",
                "reason": "duplicate_replay_field",
            },
            {
                "case": "unsupported_operator_id",
                "input_kind": "canonical_replay_receipt",
                "reason": "unsupported_operator_id",
            },
        ),
        "covered_behaviors": (
            "invalid_draw_not_admitted_to_modular_two_draws_replay_receipt",
            "invalid_draw_maps_to_separate_invalid_input_receipt",
            "invalid_input_receipt_preserves_raw_present_draw_bits_without_unit_draw_validation",
            "invalid_input_receipt_records_absent_draws_with_present_flags",
            "absent_draw_bit_fields_must_be_zero",
            "malformed_replay_cases_bind_input_digest_without_replay_admission",
            "invalid_input_receipt_binds_native_input_digest",
            "invalid_input_receipt_rejects_state_change_claims",
            "invalid_input_receipt_rejects_unknown_fields",
        ),
        "limits": (
            "invalid_input_contract_only_not_valid_replay_admission",
            "malformed_replay_cases_are_digest_bound_not_replayed",
            "duplicate_replay_field_fixture_uses_pair-list_digest_surrogate",
            "not_all_disabled_mutation_receipt_byte_parity",
            "not_proposal_digest_contract_vectors",
            "not_the_production_recurrent_rowv1_artifact",
            "no_cuda_promotion",
            "no_route_package_qualification",
        ),
        "native_cpu_a_authorized": False,
        "cuda_promotion_authorized": False,
        "successor_campaign_authorized": False,
    }
    payload["contract_sha256"] = native_cpu_progress_sha256_v1(payload)
    return payload


def native_core_ordinary_birth_proposal_digest_vectors_v1() -> dict[str, Any]:
    """Record proposal digest presence/absence vectors for native-core receipts."""

    payload = {
        "schema_version": SCHEMA_VERSION,
        "contract_id": NATIVE_CORE_PROPOSAL_DIGEST_VECTORS_CONTRACT_ID,
        "answer_key_contract_id": ANSWER_KEY_CONTRACT_ID,
        "answer_key_contract_sha256": ANSWER_KEY_CONTRACT_SHA256,
        "case_id": "native_core_ordinary_birth_proposal_digest_vectors",
        "decision": "P12-NATIVE-CORE-R-PROPOSAL-DIGEST-VECTORS",
        "native_v2_receipt_contract_id": (
            NATIVE_CORE_ORDINARY_BIRTH_V2_RECEIPT_SURFACE_CONTRACT_ID
        ),
        "native_v3_symbol": (
            "eiel_modular_recurrent_core_ordinary_birth_two_draws_v3"
        ),
        "native_surface": (
            "engine/perf/native/recurrent/modular_recurrent_core_c_api_v1.cpp"
        ),
        "test_surface": "tests/perf/test_eiel_recurrent_native_core_c_api_v1.py",
        "covered_vectors": (
            {
                "case": "committed_publication",
                "transaction_status": 0,
                "proposal_digest": "present_equals_published_child_digest",
            },
            {
                "case": "capacity_unavailable",
                "transaction_status": 1,
                "proposal_digest": "absent_before_proposal_construction",
            },
            {
                "case": "publication_failure",
                "transaction_status": 2,
                "proposal_digest": "present_proposal_observability_only",
            },
            {
                "case": "owner_mismatch",
                "transaction_status": 3,
                "proposal_digest": "absent_before_proposal_construction",
            },
            {
                "case": "invalid_draw",
                "transaction_status": 4,
                "proposal_digest": "absent_invalid_input",
            },
            {
                "case": "mutation_disabled",
                "transaction_status": 6,
                "proposal_digest": "absent_disabled_before_proposal",
            },
        ),
        "covered_behaviors": (
            "committed_proposal_digest_equals_child_digest_after",
            "publication_failure_preserves_proposal_digest_without_child_publication",
            "capacity_owner_invalid_draw_and_disabled_cases_emit_zero_proposal_digest",
            "proposal_digest_never_equals_parent_or_empty_child_preimage_when_present",
            "noncommitted_cases_preserve_child_digest_after",
        ),
        "limits": (
            "native_core_v2_v3_receipt_vectors_only",
            "not_native_mutation_rejected_vector",
            "not_all_disabled_mutation_receipt_byte_parity",
            "not_the_production_recurrent_rowv1_artifact",
            "no_cuda_promotion",
            "no_route_package_qualification",
        ),
        "native_cpu_a_authorized": False,
        "cuda_promotion_authorized": False,
        "successor_campaign_authorized": False,
    }
    payload["contract_sha256"] = native_cpu_progress_sha256_v1(payload)
    return payload


def native_core_recurrent_receipt_byte_authority_progress_v1() -> dict[str, Any]:
    """Record native-core recurrent receipt-byte progress without promotion."""

    evidence_receipts = (
        native_core_ordinary_birth_v2_receipt_surface_v1(),
        native_core_ordinary_birth_oracle_aligned_receipt_bytes_v1(),
        native_core_ordinary_birth_rejected_receipt_bytes_v1(),
        native_core_ordinary_birth_owner_rejection_receipt_bytes_v1(),
        native_core_ordinary_birth_receipt_repeat_determinism_v1(),
        native_core_ordinary_birth_disabled_receipt_bytes_v1(),
        native_core_ordinary_birth_invalid_input_receipt_v1(),
        native_core_ordinary_birth_proposal_digest_vectors_v1(),
        native_core_ordinary_birth_disabled_reason_receipt_bytes_v1(),
        native_core_ordinary_birth_disabled_reason_semantics_v1(),
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "contract_id": NATIVE_CORE_RECURRENT_RECEIPT_BYTE_PROGRESS_CONTRACT_ID,
        "answer_key_contract_id": ANSWER_KEY_CONTRACT_ID,
        "answer_key_contract_sha256": ANSWER_KEY_CONTRACT_SHA256,
        "case_id": "native_core_recurrent_receipt_byte_authority_progress",
        "decision": "P12-NATIVE-CORE-U-RECURRENT-RECEIPT-BYTE-PROGRESS",
        "native_surface": (
            "engine/perf/native/recurrent/modular_recurrent_core_c_api_v1.cpp"
        ),
        "bridge_surface": (
            "engine/perf/recurrent_native/"
            "native_core_ordinary_birth_receipt_bridge_v1.py"
        ),
        "canonical_receipt_surface": (
            "engine/perf/recurrent_native/modular_recurrent_mutation_receipt_v1.py"
        ),
        "rowv1_projection_surface": "engine/evolution/p12_rowv1_cuda_entry_gate_v1.py",
        "test_surface": "tests/perf/test_eiel_recurrent_native_core_c_api_v1.py",
        "evidence_contracts": tuple(
            {
                "contract_id": receipt["contract_id"],
                "decision": receipt["decision"],
                "contract_sha256": receipt["contract_sha256"],
            }
            for receipt in evidence_receipts
        ),
        "covered_fixture_families": (
            "accepted_v2_oracle_aligned_receipt_bytes",
            "bounded_structural_payload_sha_fixture",
            "capacity_and_publication_rejected_v2_receipt_bytes",
            "owner_rejected_v2_receipt_bytes",
            "operator_disabled_v3_receipt_bytes",
            "six_disabled_reason_v4_receipt_bytes",
            "invalid_input_receipt_bytes",
            "proposal_digest_presence_absence_vectors",
            "fresh_repeat_determinism_for_accepted_rejected_disabled_owner_cases",
            "disabled_reason_semantic_precedence",
        ),
        "covered_behaviors": (
            "byte_comparable_native_core_receipts_exist_for_bounded_recurrent_cases",
            "bounded_native_core_structural_payload_sha_equality_exists",
            "native_core_receipts_compare_to_python_or_canonical_replay_bytes",
            "disabled_reason_codes_are_explicit_and_semantically_mapped",
            "invalid_draws_fail_into_invalid_input_receipts",
            "proposal_digest_vectors_are_explicit",
            "receipt_repeat_determinism_is_bounded_by_fresh_fixture_repeats",
        ),
        "limits": (
            "native_core_fixed_envelope_only",
            "not_the_production_recurrent_lifecycle_c_api_route",
            "not_full_v1_1_native_cpu_parity_matrix",
            "not_route_bound_structural_payload_sha256_equality",
            "not_checkpoint_resume_repeat_determinism",
            "not_cuda_authoritative_receipt_abi",
            "no_route_package_qualification",
        ),
        "native_cpu_a_authorized": False,
        "cuda_promotion_authorized": False,
        "successor_campaign_authorized": False,
    }
    payload["contract_sha256"] = native_cpu_progress_sha256_v1(payload)
    return payload


def native_core_ordinary_birth_receipt_repeat_determinism_v1() -> dict[str, Any]:
    """Record deterministic repeat coverage for byte-comparable native fixtures."""

    payload = {
        "schema_version": SCHEMA_VERSION,
        "contract_id": NATIVE_CORE_RECEIPT_REPEAT_CONTRACT_ID,
        "answer_key_contract_id": ANSWER_KEY_CONTRACT_ID,
        "answer_key_contract_sha256": ANSWER_KEY_CONTRACT_SHA256,
        "case_id": "native_core_ordinary_birth_receipt_repeat_determinism",
        "decision": "P12-NATIVE-CORE-O-RECEIPT-REPEAT-DETERMINISM",
        "native_v2_receipt_contract_id": (
            NATIVE_CORE_ORDINARY_BIRTH_V2_RECEIPT_SURFACE_CONTRACT_ID
        ),
        "accepted_receipt_bytes_contract_id": (
            NATIVE_CORE_ORACLE_ALIGNED_RECEIPT_BYTES_CONTRACT_ID
        ),
        "rejected_receipt_bytes_contract_id": (
            NATIVE_CORE_REJECTED_RECEIPT_BYTES_CONTRACT_ID
        ),
        "owner_rejection_receipt_bytes_contract_id": (
            NATIVE_CORE_OWNER_REJECTION_RECEIPT_BYTES_CONTRACT_ID
        ),
        "disabled_reason_receipt_bytes_contract_id": (
            NATIVE_CORE_DISABLED_REASON_RECEIPT_BYTES_CONTRACT_ID
        ),
        "test_surface": "tests/perf/test_eiel_recurrent_native_core_c_api_v1.py",
        "covered_fixture_count": 16,
        "covered_fixtures": (
            "accepted_toggle_dormancy",
            "accepted_prune_eligible_module",
            "accepted_refine_descendant_parameter",
            "accepted_refine_descendant_routing_or_expression",
            "rejected_capacity_unavailable",
            "rejected_publication_failure",
            "disabled_operator_flag_off",
            "v4_disabled_operator_disabled",
            "v4_disabled_structural_mutation_disabled",
            "v4_disabled_recurrent_mutation_disabled",
            "v4_disabled_operator_family_disabled",
            "v4_disabled_zero_mutation_rate",
            "v4_disabled_mutation_attempt_budget_zero",
            "owner_rejected_child_publication_token_not_newer_than_expected",
            "owner_rejected_same_slot_publication_token_not_newer_than_parent",
            "owner_rejected_invalid_parent_slot_token",
        ),
        "covered_behaviors": (
            "fresh_fixture_repeat_native_receipt_fields_identical",
            "fresh_fixture_repeat_canonical_replay_sha256_identical",
            "fresh_fixture_repeat_canonical_replay_bytes_identical",
            "fresh_fixture_repeat_child_digest_identical",
        ),
        "limits": (
            "byte_comparable_native_core_v2_v3_v4_fixtures_only",
            "not_invalid_draw_repeat_determinism",
            "not_native_mutation_rejected_repeat_determinism",
            "not_runtime_rng_index_table",
            "not_checkpoint_resume_repeat_determinism",
            "not_the_production_recurrent_rowv1_artifact",
            "no_cuda_promotion",
            "no_route_package_qualification",
        ),
        "native_cpu_a_authorized": False,
        "cuda_promotion_authorized": False,
        "successor_campaign_authorized": False,
    }
    payload["contract_sha256"] = native_cpu_progress_sha256_v1(payload)
    return payload


__all__ = [
    "ANSWER_KEY_CONTRACT_ID",
    "ANSWER_KEY_CONTRACT_SHA256",
    "ABSENT_NATIVE_RECEIPT_PAYLOAD_SYMBOLS",
    "CONTRACT_ID",
    "CURRENT_NATIVE_RECURRENT_LIFECYCLE_APIS",
    "NATIVE_CPU_ROUTE_BOUND_BIRTH_RECEIPT_CONTRACT_ID",
    "NATIVE_CPU_GATE1_ROUTE_QUALIFICATION_CONTRACT_ID",
    "NATIVE_CORE_C_API_BOUNDARY_CONTRACT_ID",
    "NATIVE_CORE_ORDINARY_BIRTH_BOUNDARY_CONTRACT_ID",
    "NATIVE_CORE_ORDINARY_BIRTH_V2_RECEIPT_SURFACE_CONTRACT_ID",
    "NATIVE_CORE_OPERATOR_CONFORMANCE_CONTRACT_ID",
    "NATIVE_CORE_ORACLE_ALIGNED_RECEIPT_BYTES_CONTRACT_ID",
    "NATIVE_CORE_DISABLED_RECEIPT_BYTES_CONTRACT_ID",
    "NATIVE_CORE_DISABLED_REASON_RECEIPT_BYTES_CONTRACT_ID",
    "NATIVE_CORE_DISABLED_REASON_SEMANTICS_CONTRACT_ID",
    "NATIVE_CORE_INVALID_INPUT_RECEIPT_CONTRACT_ID",
    "NATIVE_CORE_PROPOSAL_DIGEST_VECTORS_CONTRACT_ID",
    "NATIVE_CORE_OWNER_REJECTION_RECEIPT_BYTES_CONTRACT_ID",
    "NATIVE_CORE_RECEIPT_REPEAT_CONTRACT_ID",
    "NATIVE_CORE_REJECTED_RECEIPT_BYTES_CONTRACT_ID",
    "NATIVE_CORE_IDENTITY_MAPPING_CONTRACT_ID",
    "NATIVE_CORE_PAYLOAD_OBSERVABILITY_CONTRACT_ID",
    "NATIVE_CORE_PAYLOAD_PROJECTION_CONTRACT_ID",
    "NATIVE_CORE_STRUCTURAL_PAYLOAD_SHA_CONTRACT_ID",
    "NATIVE_CORE_RECURRENT_RECEIPT_BRIDGE_CONTRACT_ID",
    "RECEIPT_PAYLOAD_GAP_CONTRACT_ID",
    "native_cpu_route_bound_birth_lifecycle_receipt_v1",
    "native_cpu_gate1_turnover_route_package_qualification_v1",
    "native_cpu_recurrent_lifecycle_owner_progress_v1",
    "native_cpu_recurrent_receipt_payload_gap_map_v1",
    "native_core_two_draw_operator_conformance_v1",
    "native_core_c_api_two_draw_boundary_v1",
    "native_core_ordinary_birth_transaction_boundary_v1",
    "native_core_ordinary_birth_recurrent_receipt_bridge_v1",
    "native_core_ordinary_birth_payload_observability_v1",
    "native_core_ordinary_birth_identity_mapping_bridge_v1",
    "native_core_ordinary_birth_payload_projection_v1",
    "native_core_ordinary_birth_structural_payload_sha_v1",
    "native_core_ordinary_birth_v2_receipt_surface_v1",
    "native_core_ordinary_birth_oracle_aligned_receipt_bytes_v1",
    "native_core_ordinary_birth_disabled_receipt_bytes_v1",
    "native_core_ordinary_birth_disabled_reason_receipt_bytes_v1",
    "native_core_ordinary_birth_disabled_reason_semantics_v1",
    "native_core_ordinary_birth_invalid_input_receipt_v1",
    "native_core_ordinary_birth_proposal_digest_vectors_v1",
    "native_core_ordinary_birth_owner_rejection_receipt_bytes_v1",
    "native_core_ordinary_birth_receipt_repeat_determinism_v1",
    "native_core_ordinary_birth_rejected_receipt_bytes_v1",
    "canonical_native_cpu_progress_bytes_v1",
    "native_cpu_progress_sha256_v1",
]
