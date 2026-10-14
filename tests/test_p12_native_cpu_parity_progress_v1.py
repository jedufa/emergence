from __future__ import annotations

import json
from pathlib import Path

from engine.evolution.p12_native_cpu_parity_progress_v1 import (
    ANSWER_KEY_CONTRACT_SHA256,
    ABSENT_NATIVE_RECEIPT_PAYLOAD_SYMBOLS,
    CONTRACT_ID,
    CURRENT_NATIVE_RECURRENT_LIFECYCLE_APIS,
    NATIVE_CPU_ROUTE_BOUND_BIRTH_RECEIPT_CONTRACT_ID,
    NATIVE_CORE_C_API_BOUNDARY_CONTRACT_ID,
    NATIVE_CORE_DISABLED_REASON_RECEIPT_BYTES_CONTRACT_ID,
    NATIVE_CORE_DISABLED_REASON_SEMANTICS_CONTRACT_ID,
    NATIVE_CORE_DISABLED_RECEIPT_BYTES_CONTRACT_ID,
    NATIVE_CORE_IDENTITY_MAPPING_CONTRACT_ID,
    NATIVE_CORE_INVALID_INPUT_RECEIPT_CONTRACT_ID,
    NATIVE_CORE_ORDINARY_BIRTH_BOUNDARY_CONTRACT_ID,
    NATIVE_CORE_ORDINARY_BIRTH_V2_RECEIPT_SURFACE_CONTRACT_ID,
    NATIVE_CORE_OPERATOR_CONFORMANCE_CONTRACT_ID,
    NATIVE_CORE_ORACLE_ALIGNED_RECEIPT_BYTES_CONTRACT_ID,
    NATIVE_CORE_OWNER_REJECTION_RECEIPT_BYTES_CONTRACT_ID,
    NATIVE_CORE_PAYLOAD_OBSERVABILITY_CONTRACT_ID,
    NATIVE_CORE_PAYLOAD_PROJECTION_CONTRACT_ID,
    NATIVE_CORE_STRUCTURAL_PAYLOAD_SHA_CONTRACT_ID,
    NATIVE_CORE_PROPOSAL_DIGEST_VECTORS_CONTRACT_ID,
    NATIVE_CORE_RECEIPT_REPEAT_CONTRACT_ID,
    NATIVE_CORE_RECURRENT_RECEIPT_BYTE_PROGRESS_CONTRACT_ID,
    NATIVE_CORE_REJECTED_RECEIPT_BYTES_CONTRACT_ID,
    NATIVE_CORE_RECURRENT_RECEIPT_BRIDGE_CONTRACT_ID,
    NATIVE_CPU_GATE1_ROUTE_QUALIFICATION_CONTRACT_ID,
    NATIVE_CPU_ROUTE_BOUND_ARTIFACT_SHA256,
    NATIVE_CPU_ROUTE_BOUND_BUILD_MANIFEST_SHA256,
    RECEIPT_PAYLOAD_GAP_CONTRACT_ID,
    canonical_native_cpu_progress_bytes_v1,
    native_core_c_api_two_draw_boundary_v1,
    native_core_ordinary_birth_disabled_reason_receipt_bytes_v1,
    native_core_ordinary_birth_disabled_reason_semantics_v1,
    native_core_ordinary_birth_disabled_receipt_bytes_v1,
    native_core_ordinary_birth_invalid_input_receipt_v1,
    native_core_ordinary_birth_identity_mapping_bridge_v1,
    native_core_ordinary_birth_transaction_boundary_v1,
    native_core_ordinary_birth_recurrent_receipt_bridge_v1,
    native_core_ordinary_birth_payload_observability_v1,
    native_core_ordinary_birth_payload_projection_v1,
    native_core_ordinary_birth_structural_payload_sha_v1,
    native_core_ordinary_birth_proposal_digest_vectors_v1,
    native_core_ordinary_birth_v2_receipt_surface_v1,
    native_core_ordinary_birth_oracle_aligned_receipt_bytes_v1,
    native_core_ordinary_birth_owner_rejection_receipt_bytes_v1,
    native_core_ordinary_birth_receipt_repeat_determinism_v1,
    native_core_ordinary_birth_rejected_receipt_bytes_v1,
    native_core_recurrent_receipt_byte_authority_progress_v1,
    native_core_two_draw_operator_conformance_v1,
    native_cpu_gate1_turnover_route_package_qualification_v1,
    native_cpu_progress_sha256_v1,
    native_cpu_route_bound_birth_lifecycle_receipt_v1,
    native_cpu_recurrent_lifecycle_owner_progress_v1,
    native_cpu_recurrent_receipt_payload_gap_map_v1,
)


def test_native_cpu_progress_receipt_is_bounded_to_answer_key_subset() -> None:
    receipt = native_cpu_recurrent_lifecycle_owner_progress_v1()

    assert receipt["contract_id"] == CONTRACT_ID
    assert receipt["answer_key_contract_sha256"] == ANSWER_KEY_CONTRACT_SHA256
    assert {
        "child_agent_id",
        "parent_slot",
        "child_slot",
        "parent_slot_token",
        "child_slot_token",
        "publication_state",
        "rollback_state",
        "recurrent_payload_sha256",
        "checkpoint_bytes",
    } == set(receipt["covered_compare_fields"])
    assert {
        "rng_draw_bits",
        "proposal_bytes",
        "structural_payload_sha256",
        "cursor_rows",
        "receipt_bytes",
        "multicell_continuation",
        "cuda_parity",
        "route_package_qualification",
    } == set(receipt["not_covered_compare_fields"])
    assert receipt["native_cpu_a_authorized"] is False
    assert receipt["cuda_promotion_authorized"] is False
    assert receipt["successor_campaign_authorized"] is False


def test_native_cpu_progress_receipt_serialization_is_canonical() -> None:
    receipt = native_cpu_recurrent_lifecycle_owner_progress_v1()
    wire = canonical_native_cpu_progress_bytes_v1(receipt)

    assert wire.endswith(b"\n")
    assert json.loads(wire.decode("ascii"))["contract_id"] == CONTRACT_ID
    assert receipt["contract_sha256"] == native_cpu_progress_sha256_v1(
        {key: value for key, value in receipt.items() if key != "contract_sha256"}
    )


def test_native_cpu_receipt_payload_gap_map_is_fail_closed() -> None:
    receipt = native_cpu_recurrent_receipt_payload_gap_map_v1()

    assert receipt["contract_id"] == RECEIPT_PAYLOAD_GAP_CONTRACT_ID
    assert receipt["answer_key_contract_sha256"] == ANSWER_KEY_CONTRACT_SHA256
    assert receipt["decision"] == "P12-NATIVE-CPU-C-FAIL-CLOSED-GAP-MAP"
    assert receipt["positive_native_evidence_contract_id"] == CONTRACT_ID
    assert set(receipt["gap_compare_fields"]) == {
        "rng_draw_bits",
        "proposal_bytes",
        "structural_payload_sha256",
        "receipt_bytes",
    }
    current_states = {
        row["surface_id"]: row["current_state"]
        for row in receipt["surface_inventory"]
    }
    for surface_id, current_state in {
        "python_reference_structural_payload_oracle": "implemented_reference_only",
        "python_reference_recurrent_receipt_replay": "implemented_reference_only",
        "native_cpu_recurrent_lifecycle_c_api": (
            "real_lifecycle_owner_birth_receipt_native_canonical_digest_and_native_structural_sha_binding_surface"
        ),
        "native_cpu_structural_payload_oracle": (
            "native_generated_rowv1_structural_payload_sha_present_and_route_bound_v3"
        ),
        "native_cpu_recurrent_receipt_bytes": (
            "bounded_native_core_byte_comparable_present_not_route_bound"
        ),
    }.items():
        assert current_states[surface_id] == current_state
    assert receipt["bounded_recurrent_receipt_byte_progress_contract_id"] == (
        NATIVE_CORE_RECURRENT_RECEIPT_BYTE_PROGRESS_CONTRACT_ID
    )
    assert receipt["route_bound_birth_receipt_contract_id"] == (
        NATIVE_CPU_ROUTE_BOUND_BIRTH_RECEIPT_CONTRACT_ID
    )
    assert receipt["bounded_structural_payload_sha_contract_id"] == (
        NATIVE_CORE_STRUCTURAL_PAYLOAD_SHA_CONTRACT_ID
    )
    assert {
        "native_digest_domain_is_rowv1_structural_payload_v1",
        "native_digest_is_little_endian_rowv1_structural_subset",
        "one_route_bound_newborn_fixture_only",
        "one_route_bound_same_slot_fixture_only",
        "not_bound_to_route_package_qualification",
        "bounded_two_cell_multicell_checkpoint_fixture_only",
        "bounded_b1_generation_turnover_checkpoint_resume_fixture_only",
        "not_route_package_multicell_checkpoint_qualification",
    } <= set(receipt["remaining_structural_payload_limits"])
    assert "not_generation_turnover_checkpoint_resume_repeatability" not in set(
        receipt["remaining_structural_payload_limits"]
    )
    assert {
        "native_route_builds_bounded_canonical_receipt_digest_not_package_qualified",
        "source_and_ancestry_hashes_are_caller_supplied",
        "bounded_one_transaction_ordinary_birth_canonical_builder_only",
        "one_row_v4_checkpoint_project_restore_only",
        "bounded_two_cell_multicell_checkpoint_fixture_is_artifact_bound",
        "bounded_b1_generation_turnover_checkpoint_resume_fixture_only",
        "not_bound_to_route_package_qualification",
        "not_full_route_package_multicell_checkpoint_qualification",
        "not_cuda_authoritative_receipt_abi",
    } <= set(receipt["remaining_receipt_byte_limits"])
    assert "not_generation_turnover_checkpoint_resume_repeatability" not in set(
        receipt["remaining_receipt_byte_limits"]
    )
    assert tuple(receipt["current_native_lifecycle_apis"]) == (
        CURRENT_NATIVE_RECURRENT_LIFECYCLE_APIS
    )
    assert receipt["native_cpu_a_authorized"] is False
    assert receipt["cuda_promotion_authorized"] is False
    assert receipt["successor_campaign_authorized"] is False
    assert receipt["contract_sha256"] == native_cpu_progress_sha256_v1(
        {key: value for key, value in receipt.items() if key != "contract_sha256"}
    )


def test_native_cpu_route_bound_birth_lifecycle_receipt_is_bounded() -> None:
    receipt = native_cpu_route_bound_birth_lifecycle_receipt_v1()

    assert receipt["contract_id"] == NATIVE_CPU_ROUTE_BOUND_BIRTH_RECEIPT_CONTRACT_ID
    assert receipt["decision"] == "P12-NATIVE-CPU-F-NATIVE-CANONICAL-RECEIPT-DIGEST"
    assert (
        receipt["route_bound_symbol"]
        == "eiel_modular_recurrent_newborn_inherit_with_receipt_v1"
    )
    assert (
        receipt["same_slot_route_bound_symbol"]
        == "eiel_modular_recurrent_same_slot_reuse_with_receipt_v1"
    )
    assert receipt["canonical_route_bound_symbol"] == (
        "eiel_modular_recurrent_newborn_inherit_with_canonical_receipt_v1"
    )
    assert receipt["same_slot_canonical_route_bound_symbol"] == (
        "eiel_modular_recurrent_same_slot_reuse_with_canonical_receipt_v1"
    )
    assert receipt["native_structural_payload_sha_symbol"] == (
        "eiel_modular_recurrent_row_structural_payload_sha256_v1"
    )
    assert receipt["native_canonical_receipt_sha_symbol"] == (
        "eiel_modular_recurrent_canonical_receipt_sha256_v1"
    )
    assert receipt["artifact_surface"] == (
        "engine/perf/recurrent_native/libeiel_modular_recurrent_cpu_oracle_v1.so"
    )
    assert receipt["artifact_sha256"] == NATIVE_CPU_ROUTE_BOUND_ARTIFACT_SHA256
    assert receipt["build_manifest_surface"] == (
        "engine/perf/recurrent_native/modular_recurrent_cpu_oracle_shared_build_manifest_v1.json"
    )
    assert (
        receipt["build_manifest_sha256"]
        == NATIVE_CPU_ROUTE_BOUND_BUILD_MANIFEST_SHA256
    )
    assert receipt["wrapped_lifecycle_symbol"] == (
        "eiel_modular_recurrent_newborn_inherit_v1"
    )
    assert receipt["same_slot_wrapped_lifecycle_symbol"] == (
        "eiel_modular_recurrent_same_slot_reuse_v1"
    )
    assert {
        "lifecycle_status",
        "committed",
        "parent_preserved",
        "child_runtime_reset",
        "canonical_receipt_bound",
        "canonical_receipt_identity_checked",
        "recurrent_receipt_bytes_bound",
        "donor_draw_float32_bits",
        "draw0_float64_bits",
        "draw1_float64_bits",
        "canonical_recurrent_receipt_sha256_words",
        "canonical_recurrent_receipt_bytes_sha256_words",
        "structural_payload_sha256_words",
        "structural_payload_sha256_bound",
        "structural_payload_identity_checked",
        "mutation_count",
    } <= set(receipt["covered_receipt_fields"])
    assert {
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
        "proves_same_slot_stale_owner_rejection_preserves_state",
    } <= set(receipt["covered_behaviors"])
    assert {
        "source_and_ancestry_hashes_are_caller_supplied",
        "bounded_one_transaction_ordinary_birth_canonical_builder_only",
        "native_structural_payload_digest_is_rowv1_subset_not_full_canonical_payload_bytes",
        "bounded_two_cell_multicell_checkpoint_fixture_is_artifact_bound",
        "bounded_b1_generation_turnover_checkpoint_resume_fixture_only",
        "not_full_route_package_multicell_checkpoint_qualification",
        "not_cuda_authoritative_receipt_abi",
        "no_route_package_qualification",
    } <= set(receipt["limits"])
    assert "not_generation_turnover_checkpoint_resume_repeatability" not in set(
        receipt["limits"]
    )
    assert receipt["native_cpu_a_authorized"] is False
    assert receipt["cuda_promotion_authorized"] is False
    assert receipt["successor_campaign_authorized"] is False
    assert receipt["contract_sha256"] == native_cpu_progress_sha256_v1(
        {key: value for key, value in receipt.items() if key != "contract_sha256"}
    )


def test_native_core_recurrent_receipt_byte_progress_is_bounded() -> None:
    receipt = native_core_recurrent_receipt_byte_authority_progress_v1()

    assert (
        receipt["contract_id"]
        == NATIVE_CORE_RECURRENT_RECEIPT_BYTE_PROGRESS_CONTRACT_ID
    )
    assert (
        receipt["decision"]
        == "P12-NATIVE-CORE-U-RECURRENT-RECEIPT-BYTE-PROGRESS"
    )
    assert receipt["answer_key_contract_sha256"] == ANSWER_KEY_CONTRACT_SHA256
    assert {
        row["contract_id"] for row in receipt["evidence_contracts"]
    } == {
        NATIVE_CORE_ORDINARY_BIRTH_V2_RECEIPT_SURFACE_CONTRACT_ID,
        NATIVE_CORE_ORACLE_ALIGNED_RECEIPT_BYTES_CONTRACT_ID,
        NATIVE_CORE_REJECTED_RECEIPT_BYTES_CONTRACT_ID,
        NATIVE_CORE_OWNER_REJECTION_RECEIPT_BYTES_CONTRACT_ID,
        NATIVE_CORE_RECEIPT_REPEAT_CONTRACT_ID,
        NATIVE_CORE_DISABLED_RECEIPT_BYTES_CONTRACT_ID,
        NATIVE_CORE_INVALID_INPUT_RECEIPT_CONTRACT_ID,
        NATIVE_CORE_PROPOSAL_DIGEST_VECTORS_CONTRACT_ID,
        NATIVE_CORE_DISABLED_REASON_RECEIPT_BYTES_CONTRACT_ID,
        NATIVE_CORE_DISABLED_REASON_SEMANTICS_CONTRACT_ID,
    }
    assert {
        "accepted_v2_oracle_aligned_receipt_bytes",
        "bounded_structural_payload_sha_fixture",
        "capacity_and_publication_rejected_v2_receipt_bytes",
        "owner_rejected_v2_receipt_bytes",
        "six_disabled_reason_v4_receipt_bytes",
        "invalid_input_receipt_bytes",
    } <= set(receipt["covered_fixture_families"])
    assert {
        "byte_comparable_native_core_receipts_exist_for_bounded_recurrent_cases",
        "bounded_native_core_structural_payload_sha_equality_exists",
        "native_core_receipts_compare_to_python_or_canonical_replay_bytes",
        "disabled_reason_codes_are_explicit_and_semantically_mapped",
    } <= set(receipt["covered_behaviors"])
    assert {
        "not_the_production_recurrent_lifecycle_c_api_route",
        "not_full_v1_1_native_cpu_parity_matrix",
        "not_route_bound_structural_payload_sha256_equality",
        "not_cuda_authoritative_receipt_abi",
        "no_route_package_qualification",
    } <= set(receipt["limits"])
    assert receipt["native_cpu_a_authorized"] is False
    assert receipt["cuda_promotion_authorized"] is False
    assert receipt["successor_campaign_authorized"] is False
    assert receipt["contract_sha256"] == native_cpu_progress_sha256_v1(
        {key: value for key, value in receipt.items() if key != "contract_sha256"}
    )


def test_native_cpu_gate1_turnover_route_package_is_qualified() -> None:
    receipt = native_cpu_gate1_turnover_route_package_qualification_v1()

    assert receipt["contract_id"] == NATIVE_CPU_GATE1_ROUTE_QUALIFICATION_CONTRACT_ID
    assert receipt["decision"] == "P12-NATIVE-GATE1-A"
    assert receipt["qualified_route"] == "native_cpu_low_parallelism"
    assert receipt["native_cpu_gate1_qualified"] is True
    assert receipt["artifact_sha256"] == NATIVE_CPU_ROUTE_BOUND_ARTIFACT_SHA256
    assert (
        receipt["build_manifest_sha256"]
        == NATIVE_CPU_ROUTE_BOUND_BUILD_MANIFEST_SHA256
    )
    assert {
        "accepted_structural_mutation",
        "accepted_recurrent_mutation",
        "capacity_rejection",
        "publication_rejection",
        "stale_owner_token_rejection",
        "disabled_reasons",
        "rollback",
        "cancellation",
        "terminal_sealing",
        "newborn_publication",
        "same_slot_fresh_owner_reuse",
        "multicell_equal_counts",
        "multicell_ragged_counts",
        "high_id_and_slot_reuse",
    } == set(receipt["gate1_required_cases"])
    assert {
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
    } == set(receipt["exact_equality_fields"])
    assert receipt["route_package_qualification"] == {
        "source_artifact_manifest_bound": True,
        "artifact_hash_match_required": True,
        "build_manifest_source_hash_match_required": True,
        "package_mechanics_mismatch_rejected": True,
        "prior_bounded_only_receipts_invalidated_for_gate1_claim": True,
    }
    assert set(receipt["known_external_open_gates"]) == {
        "cuda_authoritative_rowv1_receipt_abi",
        "compact_cpu_cuda_parity",
        "final_p12_package_route_qualification",
    }
    assert receipt["p12_a_authorized"] is False
    assert receipt["cuda_promotion_authorized"] is False
    assert receipt["successor_campaign_authorized"] is False
    assert receipt["contract_sha256"] == native_cpu_progress_sha256_v1(
        {key: value for key, value in receipt.items() if key != "contract_sha256"}
    )


def test_native_cpu_gap_map_matches_current_recurrent_c_api_symbols() -> None:
    source = (
        "engine/perf/native/recurrent/modular_recurrent_cpu_oracle_c_api_v1.cpp"
    )
    text = Path(source).read_text(encoding="utf-8")

    for symbol in CURRENT_NATIVE_RECURRENT_LIFECYCLE_APIS:
        assert symbol in text
    for symbol in ABSENT_NATIVE_RECEIPT_PAYLOAD_SYMBOLS:
        assert symbol not in text


def test_native_core_two_draw_operator_conformance_receipt_is_bounded() -> None:
    receipt = native_core_two_draw_operator_conformance_v1()

    assert receipt["contract_id"] == NATIVE_CORE_OPERATOR_CONFORMANCE_CONTRACT_ID
    assert receipt["decision"] == "P12-NATIVE-CORE-D-CONSTRAINED-CONFORMANCE"
    assert receipt["answer_key_contract_sha256"] == ANSWER_KEY_CONTRACT_SHA256
    assert tuple(receipt["covered_operation_bins"]) == (
        "toggle_dormancy",
        "prune_eligible_module",
        "refine_descendant_parameter",
        "refine_descendant_routing_or_expression",
    )
    assert {
        "operation_bin",
        "native_status",
        "selected_slot",
        "changed_dimension",
        "changed_flag",
        "canonical_digest_change",
        "invalid_draw_rollback",
    } == set(receipt["covered_fields"])
    assert {
        "no_canonical_modular_two_draws_receipt",
        "not_exposed_by_production_recurrent_c_api",
        "no_cuda_promotion",
    } <= set(receipt["limits"])
    assert receipt["native_cpu_a_authorized"] is False
    assert receipt["cuda_promotion_authorized"] is False
    assert receipt["successor_campaign_authorized"] is False
    assert receipt["contract_sha256"] == native_cpu_progress_sha256_v1(
        {key: value for key, value in receipt.items() if key != "contract_sha256"}
    )


def test_native_core_c_api_two_draw_boundary_receipt_is_bounded() -> None:
    receipt = native_core_c_api_two_draw_boundary_v1()

    assert receipt["contract_id"] == NATIVE_CORE_C_API_BOUNDARY_CONTRACT_ID
    assert receipt["decision"] == "P12-NATIVE-CORE-E-C-API-BOUNDARY"
    assert receipt["answer_key_contract_sha256"] == ANSWER_KEY_CONTRACT_SHA256
    assert "eiel_modular_recurrent_core_mutate_two_draws_v1" in receipt["exports"]
    assert "eiel_modular_recurrent_newborn_inherit_v1" not in receipt["exports"]
    assert tuple(receipt["covered_operation_bins"]) == (
        "toggle_dormancy",
        "prune_eligible_module",
        "refine_descendant_parameter",
        "refine_descendant_routing_or_expression",
    )
    assert {
        "status",
        "operation",
        "selected_slot",
        "changed_dimension",
        "changed",
        "digest_before",
        "digest_after",
    } == set(receipt["receipt_fields"])
    assert {
        "not_the_production_recurrent_rowv1_artifact",
        "no_ordinary_birth_owner_publication",
        "no_cuda_promotion",
    } <= set(receipt["limits"])
    assert receipt["native_cpu_a_authorized"] is False
    assert receipt["cuda_promotion_authorized"] is False
    assert receipt["successor_campaign_authorized"] is False
    assert receipt["contract_sha256"] == native_cpu_progress_sha256_v1(
        {key: value for key, value in receipt.items() if key != "contract_sha256"}
    )


def test_native_core_ordinary_birth_transaction_boundary_is_bounded() -> None:
    receipt = native_core_ordinary_birth_transaction_boundary_v1()

    assert receipt["contract_id"] == NATIVE_CORE_ORDINARY_BIRTH_BOUNDARY_CONTRACT_ID
    assert receipt["decision"] == "P12-NATIVE-CORE-F-ORDINARY-BIRTH-BOUNDARY"
    assert receipt["answer_key_contract_sha256"] == ANSWER_KEY_CONTRACT_SHA256
    assert (
        "eiel_modular_recurrent_core_ordinary_birth_two_draws_v1"
        in receipt["exports"]
    )
    assert set(receipt["transaction_statuses"]) == {
        "committed",
        "capacity_unavailable",
        "publication_rejected",
        "owner_mismatch",
        "invalid_draw",
        "native_mutation_rejected",
    }
    assert {
        "draw0_float64_bits",
        "draw1_float64_bits",
        "proposal_digest",
        "child_digest_after",
    } <= set(receipt["covered_fields"])
    assert {
        "exact_binary64_draw_bit_ledger",
        "owner_token_staleness_rejection",
        "publication_rejected_rollback_with_proposal_digest",
    } <= set(receipt["covered_behaviors"])
    assert {
        "no_python_json_payload_sha256",
        "no_canonical_modular_two_draws_receipt",
        "no_cuda_promotion",
        "no_route_package_qualification",
    } <= set(receipt["limits"])
    assert receipt["native_cpu_a_authorized"] is False
    assert receipt["cuda_promotion_authorized"] is False
    assert receipt["successor_campaign_authorized"] is False
    assert receipt["contract_sha256"] == native_cpu_progress_sha256_v1(
        {key: value for key, value in receipt.items() if key != "contract_sha256"}
    )


def test_native_core_ordinary_birth_recurrent_receipt_bridge_is_bounded() -> None:
    receipt = native_core_ordinary_birth_recurrent_receipt_bridge_v1()

    assert receipt["contract_id"] == NATIVE_CORE_RECURRENT_RECEIPT_BRIDGE_CONTRACT_ID
    assert receipt["decision"] == "P12-NATIVE-CORE-G-RECURRENT-RECEIPT-BRIDGE"
    assert receipt["answer_key_contract_sha256"] == ANSWER_KEY_CONTRACT_SHA256
    assert receipt["covered_receipt_abi"] == "modular_two_draws_v1"
    assert {
        "real_native_c_api_committed_birth_to_canonical_recurrent_receipt",
        "real_native_c_api_capacity_rejection_to_canonical_recurrent_receipt",
        "exact_binary64_draw_hex_preservation",
        "high_id_child_agent_id_preservation",
    } <= set(receipt["covered_behaviors"])
    assert {
        "reason_strings_are_native_bridge_reasons_not_python_oracle_reasons",
        "selected_identity_indexes_are_supplied_by_caller",
        "no_structural_payload_sha256",
        "no_python_oracle_payload_equality",
        "no_cuda_promotion",
    } <= set(receipt["limits"])
    assert receipt["native_cpu_a_authorized"] is False
    assert receipt["cuda_promotion_authorized"] is False
    assert receipt["successor_campaign_authorized"] is False
    assert receipt["contract_sha256"] == native_cpu_progress_sha256_v1(
        {key: value for key, value in receipt.items() if key != "contract_sha256"}
    )


def test_native_core_ordinary_birth_payload_observability_is_bounded() -> None:
    receipt = native_core_ordinary_birth_payload_observability_v1()

    assert receipt["contract_id"] == NATIVE_CORE_PAYLOAD_OBSERVABILITY_CONTRACT_ID
    assert receipt["decision"] == "P12-NATIVE-CORE-H-PAYLOAD-OBSERVABILITY"
    assert receipt["answer_key_contract_sha256"] == ANSWER_KEY_CONTRACT_SHA256
    assert "refine_descendant_parameter" in receipt["covered_fixture"]
    assert {
        "operation",
        "changed_dimension",
        "value_before_float32",
        "value_after_float32",
        "python_selected_module_instance_id_present",
    } <= set(receipt["covered_fields"])
    assert {
        "one_semantic_payload_fixture_only",
        "native_numeric_identity_namespace_not_equal_to_python_hash_identity_namespace",
        "no_structural_payload_sha256_equality",
        "no_cuda_promotion",
        "no_route_package_qualification",
    } <= set(receipt["limits"])
    assert receipt["native_cpu_a_authorized"] is False
    assert receipt["cuda_promotion_authorized"] is False
    assert receipt["successor_campaign_authorized"] is False
    assert receipt["contract_sha256"] == native_cpu_progress_sha256_v1(
        {key: value for key, value in receipt.items() if key != "contract_sha256"}
    )


def test_native_core_ordinary_birth_identity_mapping_bridge_is_bounded() -> None:
    receipt = native_core_ordinary_birth_identity_mapping_bridge_v1()

    assert receipt["contract_id"] == NATIVE_CORE_IDENTITY_MAPPING_CONTRACT_ID
    assert receipt["decision"] == "P12-NATIVE-CORE-I-IDENTITY-MAPPING-BRIDGE"
    assert receipt["answer_key_contract_sha256"] == ANSWER_KEY_CONTRACT_SHA256
    assert {
        "derive_selected_instance_id_index_from_identity_table",
        "derive_selected_homology_id_index_from_identity_table",
        "fail_closed_when_native_homology_mapping_missing",
    } <= set(receipt["covered_behaviors"])
    assert {
        "mapping_is_explicit_fixture_binding_not_universal_identity_authority",
        "no_structural_payload_sha256_equality",
        "no_cuda_promotion",
        "no_route_package_qualification",
    } <= set(receipt["limits"])
    assert receipt["native_cpu_a_authorized"] is False
    assert receipt["cuda_promotion_authorized"] is False
    assert receipt["successor_campaign_authorized"] is False
    assert receipt["contract_sha256"] == native_cpu_progress_sha256_v1(
        {key: value for key, value in receipt.items() if key != "contract_sha256"}
    )


def test_native_core_ordinary_birth_payload_projection_is_bounded() -> None:
    receipt = native_core_ordinary_birth_payload_projection_v1()

    assert receipt["contract_id"] == NATIVE_CORE_PAYLOAD_PROJECTION_CONTRACT_ID
    assert receipt["decision"] == "P12-NATIVE-CORE-J-PAYLOAD-PROJECTION"
    assert receipt["answer_key_contract_sha256"] == ANSWER_KEY_CONTRACT_SHA256
    assert {
        "project_python_payload_from_mapped_native_core_receipt",
        "compare_projected_payload_sha256_to_python_oracle_child_payload_sha256",
        "one_refine_descendant_parameter_live_input_gain_success_case",
    } <= set(receipt["covered_behaviors"])
    assert {
        "one_payload_projection_fixture_only",
        "projection_uses_explicit_identity_mapping",
        "not_full_native_payload_byte_extraction",
        "no_cuda_promotion",
        "no_route_package_qualification",
    } <= set(receipt["limits"])
    assert receipt["native_cpu_a_authorized"] is False
    assert receipt["cuda_promotion_authorized"] is False
    assert receipt["successor_campaign_authorized"] is False
    assert receipt["contract_sha256"] == native_cpu_progress_sha256_v1(
        {key: value for key, value in receipt.items() if key != "contract_sha256"}
    )


def test_native_core_structural_payload_sha_is_bounded() -> None:
    receipt = native_core_ordinary_birth_structural_payload_sha_v1()

    assert receipt["contract_id"] == NATIVE_CORE_STRUCTURAL_PAYLOAD_SHA_CONTRACT_ID
    assert (
        receipt["decision"]
        == "P12-NATIVE-CORE-V-STRUCTURAL-PAYLOAD-SHA-PROGRESS"
    )
    assert receipt["payload_projection_contract_id"] == (
        NATIVE_CORE_PAYLOAD_PROJECTION_CONTRACT_ID
    )
    assert receipt["covered_fixture"]["child_agent_id"] == (1 << 40) + 7
    assert {
        "compare_projected_payload_sha256_to_python_oracle_child_payload_sha256",
        "fail_closed_if_explicit_identity_mapping_is_missing",
    } <= set(receipt["covered_behaviors"])
    assert {
        "one_structural_payload_sha_fixture_only",
        "not_the_production_recurrent_lifecycle_c_api_route",
        "not_checkpoint_resume_repeat_determinism",
        "no_route_package_qualification",
    } <= set(receipt["limits"])
    assert receipt["native_cpu_a_authorized"] is False
    assert receipt["cuda_promotion_authorized"] is False
    assert receipt["successor_campaign_authorized"] is False
    assert receipt["contract_sha256"] == native_cpu_progress_sha256_v1(
        {key: value for key, value in receipt.items() if key != "contract_sha256"}
    )


def test_native_core_ordinary_birth_v2_receipt_surface_is_bounded() -> None:
    receipt = native_core_ordinary_birth_v2_receipt_surface_v1()

    assert (
        receipt["contract_id"]
        == NATIVE_CORE_ORDINARY_BIRTH_V2_RECEIPT_SURFACE_CONTRACT_ID
    )
    assert receipt["decision"] == "P12-NATIVE-CORE-K-V2-RECEIPT-SURFACE"
    assert receipt["answer_key_contract_sha256"] == ANSWER_KEY_CONTRACT_SHA256
    assert receipt["layout_version"] == 2
    assert {
        "eiel_modular_recurrent_core_ordinary_birth_receipt_layout_version_v2",
        "eiel_modular_recurrent_core_ordinary_birth_receipt_size_v2",
        "eiel_modular_recurrent_core_ordinary_birth_receipt_alignment_v2",
        "eiel_modular_recurrent_core_ordinary_birth_two_draws_v2",
    } <= set(receipt["exports"])
    assert {
        "draw0_float64_bits",
        "draw1_float64_bits",
        "selected_module_instance_id",
        "selected_homology_id",
        "value_before_float32",
        "value_after_float32",
    } <= set(receipt["covered_receipt_fields"])
    assert {
        "explicit_receipt_layout_version_introspection",
        "native_receipt_size_and_alignment_introspection",
        "v2_symbol_preserves_selected_identity_and_value_delta_fields",
    } <= set(receipt["covered_behaviors"])
    assert {
        "v2_surface_aliases_existing_native_core_mutation_semantics",
        "not_the_production_recurrent_rowv1_artifact",
        "no_structural_payload_sha256_equality",
        "no_python_oracle_receipt_byte_equality",
        "no_cuda_promotion",
        "no_route_package_qualification",
    } <= set(receipt["limits"])
    assert receipt["native_cpu_a_authorized"] is False
    assert receipt["cuda_promotion_authorized"] is False
    assert receipt["successor_campaign_authorized"] is False
    assert receipt["contract_sha256"] == native_cpu_progress_sha256_v1(
        {key: value for key, value in receipt.items() if key != "contract_sha256"}
    )


def test_native_core_oracle_aligned_receipt_bytes_is_bounded() -> None:
    receipt = native_core_ordinary_birth_oracle_aligned_receipt_bytes_v1()

    assert receipt["contract_id"] == NATIVE_CORE_ORACLE_ALIGNED_RECEIPT_BYTES_CONTRACT_ID
    assert receipt["decision"] == "P12-NATIVE-CORE-L-ORACLE-ALIGNED-RECEIPT-BYTES"
    assert receipt["answer_key_contract_sha256"] == ANSWER_KEY_CONTRACT_SHA256
    assert (
        receipt["native_v2_receipt_contract_id"]
        == NATIVE_CORE_ORDINARY_BIRTH_V2_RECEIPT_SURFACE_CONTRACT_ID
    )
    assert {
        row["operation"] for row in receipt["covered_fixtures"]
    } == {
        "toggle_dormancy",
        "prune_eligible_module",
        "refine_descendant_parameter",
        "refine_descendant_routing_or_expression",
    }
    assert "b2_16x48" in "_".join(receipt["covered_behaviors"])
    assert {
        "compare_native_replay_receipt_sha256_to_python_replay_receipt_sha256",
        "compare_native_replay_receipt_canonical_bytes_to_python_replay_bytes",
        "preserve_selected_identity_indexes_from_full_fixture_native_mapping",
    } <= set(receipt["covered_behaviors"])
    assert {
        "four_accepted_receipt_byte_fixtures_only",
        "explicit_identity_mapping_required",
        "accepted_reason_is_oracle_aligned_metadata",
        "not_all_v1_1_parity_matrix_cases",
        "no_cuda_promotion",
        "no_route_package_qualification",
    } <= set(receipt["limits"])
    assert receipt["native_cpu_a_authorized"] is False
    assert receipt["cuda_promotion_authorized"] is False
    assert receipt["successor_campaign_authorized"] is False
    assert receipt["contract_sha256"] == native_cpu_progress_sha256_v1(
        {key: value for key, value in receipt.items() if key != "contract_sha256"}
    )


def test_native_core_rejected_receipt_bytes_is_bounded() -> None:
    receipt = native_core_ordinary_birth_rejected_receipt_bytes_v1()

    assert receipt["contract_id"] == NATIVE_CORE_REJECTED_RECEIPT_BYTES_CONTRACT_ID
    assert receipt["decision"] == "P12-NATIVE-CORE-M-REJECTED-RECEIPT-BYTES"
    assert receipt["answer_key_contract_sha256"] == ANSWER_KEY_CONTRACT_SHA256
    assert (
        receipt["native_v2_receipt_contract_id"]
        == NATIVE_CORE_ORDINARY_BIRTH_V2_RECEIPT_SURFACE_CONTRACT_ID
    )
    assert {row["case"] for row in receipt["covered_fixtures"]} == {
        "capacity_unavailable",
        "publication_failure",
    }
    assert {
        "compare_rejected_native_replay_receipt_sha256_to_python_replay_sha256",
        "compare_rejected_native_replay_receipt_bytes_to_python_replay_bytes",
        "publication_failure_clears_changed_and_selected_identity_fields",
        "publication_failure_preserves_proposal_digest_for_observability",
    } <= set(receipt["covered_behaviors"])
    assert {
        "two_rejected_receipt_byte_fixtures_only",
        "not_stale_owner_token_receipt_byte_parity",
        "not_invalid_draw_receipt_byte_parity",
        "not_disabled_mutation_receipt_byte_parity",
        "no_cuda_promotion",
        "no_route_package_qualification",
    } <= set(receipt["limits"])
    assert receipt["native_cpu_a_authorized"] is False
    assert receipt["cuda_promotion_authorized"] is False
    assert receipt["successor_campaign_authorized"] is False
    assert receipt["contract_sha256"] == native_cpu_progress_sha256_v1(
        {key: value for key, value in receipt.items() if key != "contract_sha256"}
    )


def test_native_core_owner_rejection_receipt_bytes_is_bounded() -> None:
    receipt = native_core_ordinary_birth_owner_rejection_receipt_bytes_v1()

    assert (
        receipt["contract_id"]
        == NATIVE_CORE_OWNER_REJECTION_RECEIPT_BYTES_CONTRACT_ID
    )
    assert (
        receipt["decision"]
        == "P12-NATIVE-CORE-N-OWNER-REJECTION-RECEIPT-BYTES"
    )
    assert receipt["answer_key_contract_sha256"] == ANSWER_KEY_CONTRACT_SHA256
    assert (
        receipt["native_v2_receipt_contract_id"]
        == NATIVE_CORE_ORDINARY_BIRTH_V2_RECEIPT_SURFACE_CONTRACT_ID
    )
    assert {row["case"] for row in receipt["covered_fixtures"]} == {
        "child_publication_token_not_newer_than_expected",
        "same_slot_publication_token_not_newer_than_parent",
        "invalid_parent_slot_token",
    }
    assert {
        "compare_owner_rejected_native_replay_receipt_sha256_to_expected_sha256",
        "compare_owner_rejected_native_replay_receipt_bytes_to_expected_bytes",
        "owner_rejection_records_draw_selected_operation_for_valid_draws",
        "owner_rejection_clears_selected_identity_and_changed_fields",
    } <= set(receipt["covered_behaviors"])
    assert {
        "three_native_core_representable_owner_rejection_fixtures_only",
        "not_dead_occupant_token_lifecycle_semantics",
        "not_reused_slot_with_newer_token_lifecycle_continuation",
        "not_checkpoint_visible_allocator_semantics",
        "not_disabled_mutation_receipt_byte_parity",
        "no_cuda_promotion",
        "no_route_package_qualification",
    } <= set(receipt["limits"])
    assert receipt["native_cpu_a_authorized"] is False
    assert receipt["cuda_promotion_authorized"] is False
    assert receipt["successor_campaign_authorized"] is False
    assert receipt["contract_sha256"] == native_cpu_progress_sha256_v1(
        {key: value for key, value in receipt.items() if key != "contract_sha256"}
    )


def test_native_core_receipt_repeat_determinism_is_bounded() -> None:
    receipt = native_core_ordinary_birth_receipt_repeat_determinism_v1()

    assert receipt["contract_id"] == NATIVE_CORE_RECEIPT_REPEAT_CONTRACT_ID
    assert receipt["decision"] == "P12-NATIVE-CORE-O-RECEIPT-REPEAT-DETERMINISM"
    assert receipt["answer_key_contract_sha256"] == ANSWER_KEY_CONTRACT_SHA256
    assert receipt["covered_fixture_count"] == 16
    assert {
        "accepted_toggle_dormancy",
        "accepted_refine_descendant_routing_or_expression",
        "rejected_capacity_unavailable",
        "rejected_publication_failure",
        "disabled_operator_flag_off",
        "v4_disabled_structural_mutation_disabled",
        "v4_disabled_recurrent_mutation_disabled",
        "v4_disabled_mutation_attempt_budget_zero",
        "owner_rejected_invalid_parent_slot_token",
    } <= set(receipt["covered_fixtures"])
    assert {
        "fresh_fixture_repeat_native_receipt_fields_identical",
        "fresh_fixture_repeat_canonical_replay_sha256_identical",
        "fresh_fixture_repeat_canonical_replay_bytes_identical",
        "fresh_fixture_repeat_child_digest_identical",
    } <= set(receipt["covered_behaviors"])
    assert {
        "byte_comparable_native_core_v2_v3_v4_fixtures_only",
        "not_invalid_draw_repeat_determinism",
        "not_native_mutation_rejected_repeat_determinism",
        "not_runtime_rng_index_table",
        "not_checkpoint_resume_repeat_determinism",
        "no_cuda_promotion",
        "no_route_package_qualification",
    } <= set(receipt["limits"])
    assert receipt["native_cpu_a_authorized"] is False
    assert receipt["cuda_promotion_authorized"] is False
    assert receipt["successor_campaign_authorized"] is False
    assert receipt["contract_sha256"] == native_cpu_progress_sha256_v1(
        {key: value for key, value in receipt.items() if key != "contract_sha256"}
    )


def test_native_core_disabled_receipt_bytes_is_bounded() -> None:
    receipt = native_core_ordinary_birth_disabled_receipt_bytes_v1()

    assert receipt["contract_id"] == NATIVE_CORE_DISABLED_RECEIPT_BYTES_CONTRACT_ID
    assert receipt["decision"] == "P12-NATIVE-CORE-P-DISABLED-RECEIPT-BYTES"
    assert receipt["answer_key_contract_sha256"] == ANSWER_KEY_CONTRACT_SHA256
    assert (
        receipt["native_v3_symbol"]
        == "eiel_modular_recurrent_core_ordinary_birth_two_draws_v3"
    )
    assert {row["case"] for row in receipt["covered_fixtures"]} == {
        "disabled_operator_flag_off",
    }
    assert {
        "v3_explicit_mutation_enabled_flag",
        "compare_disabled_native_replay_receipt_sha256_to_expected_sha256",
        "compare_disabled_native_replay_receipt_bytes_to_expected_bytes",
        "disabled_receipt_clears_selected_identity_and_changed_fields",
    } <= set(receipt["covered_behaviors"])
    assert {
        "capacity_rejection_precedes_disabled_for_v3",
        "valid_owner_and_valid_draws_required_before_disabled_receipt",
        "disabled_operator_does_not_construct_proposal_digest",
    } <= set(receipt["defined_semantics"])
    assert {
        "operator_flag_off_disabled_fixture_only",
        "not_structural_mutation_disabled_fixture",
        "not_recurrent_mutation_disabled_fixture",
        "not_zero_mutation_rate_fixture",
        "not_mutation_attempt_budget_zero_fixture",
        "no_cuda_promotion",
        "no_route_package_qualification",
    } <= set(receipt["limits"])
    assert receipt["native_cpu_a_authorized"] is False
    assert receipt["cuda_promotion_authorized"] is False
    assert receipt["successor_campaign_authorized"] is False
    assert receipt["contract_sha256"] == native_cpu_progress_sha256_v1(
        {key: value for key, value in receipt.items() if key != "contract_sha256"}
    )


def test_native_core_invalid_input_receipt_is_bounded() -> None:
    receipt = native_core_ordinary_birth_invalid_input_receipt_v1()

    assert receipt["contract_id"] == NATIVE_CORE_INVALID_INPUT_RECEIPT_CONTRACT_ID
    assert receipt["decision"] == "P12-NATIVE-CORE-Q-INVALID-INPUT-RECEIPT"
    assert receipt["answer_key_contract_sha256"] == ANSWER_KEY_CONTRACT_SHA256
    assert receipt["invalid_input_receipt_contract_id"] == (
        "modular_recurrent_invalid_input_v1"
    )
    assert {row["case"] for row in receipt["covered_fixtures"]} == {
        "native_invalid_draw_out_of_range",
        "missing_draw",
        "extra_draw",
        "operation_draw_mismatch",
        "selected_index_out_of_range",
        "malformed_replay_receipt",
        "changed_replay_digest",
        "duplicate_replay_field",
        "unsupported_operator_id",
    }
    assert {
        "invalid_draw_not_admitted_to_modular_two_draws_replay_receipt",
        "invalid_draw_maps_to_separate_invalid_input_receipt",
        "invalid_input_receipt_preserves_raw_present_draw_bits_without_unit_draw_validation",
        "invalid_input_receipt_records_absent_draws_with_present_flags",
        "malformed_replay_cases_bind_input_digest_without_replay_admission",
        "invalid_input_receipt_binds_native_input_digest",
    } <= set(receipt["covered_behaviors"])
    assert {
        "invalid_input_contract_only_not_valid_replay_admission",
        "malformed_replay_cases_are_digest_bound_not_replayed",
        "not_all_disabled_mutation_receipt_byte_parity",
        "not_proposal_digest_contract_vectors",
        "no_cuda_promotion",
        "no_route_package_qualification",
    } <= set(receipt["limits"])
    assert receipt["native_cpu_a_authorized"] is False
    assert receipt["cuda_promotion_authorized"] is False
    assert receipt["successor_campaign_authorized"] is False
    assert receipt["contract_sha256"] == native_cpu_progress_sha256_v1(
        {key: value for key, value in receipt.items() if key != "contract_sha256"}
    )


def test_native_core_disabled_reason_receipt_bytes_are_bounded() -> None:
    receipt = native_core_ordinary_birth_disabled_reason_receipt_bytes_v1()

    assert receipt["contract_id"] == NATIVE_CORE_DISABLED_REASON_RECEIPT_BYTES_CONTRACT_ID
    assert receipt["decision"] == "P12-NATIVE-CORE-S-DISABLED-REASON-RECEIPT-BYTES"
    assert receipt["answer_key_contract_sha256"] == ANSWER_KEY_CONTRACT_SHA256
    assert (
        receipt["native_v4_symbol"]
        == "eiel_modular_recurrent_core_ordinary_birth_two_draws_v4"
    )
    assert {row["case"] for row in receipt["covered_fixtures"]} == {
        "operator_disabled",
        "structural_mutation_disabled",
        "recurrent_mutation_disabled",
        "operator_family_disabled",
        "zero_mutation_rate",
        "mutation_attempt_budget_zero",
    }
    assert {
        "v4_append_only_disabled_reason_code_receipt_field",
        "compare_disabled_reason_native_replay_bytes_to_expected_bytes",
        "disabled_reason_code_maps_to_canonical_disabled_reason",
        "disabled_reason_receipts_clear_selected_identity_and_changed_fields",
        "invalid_draw_precedes_disabled_reason",
    } <= set(receipt["covered_behaviors"])
    assert {
        "zero_disabled_reason_delegates_to_enabled_v2_path",
        "capacity_rejection_precedes_disabled_reason_for_v4",
        "valid_owner_and_valid_draws_required_before_disabled_reason_receipt",
        "disabled_reason_receipts_do_not_construct_proposal_digest",
    } <= set(receipt["defined_semantics"])
    assert {
        "native_core_v4_disabled_reason_fixtures_only",
        "not_runtime_rng_index_table",
        "not_checkpoint_resume_repeat_determinism",
        "no_cuda_promotion",
        "no_route_package_qualification",
    } <= set(receipt["limits"])
    assert receipt["native_cpu_a_authorized"] is False
    assert receipt["cuda_promotion_authorized"] is False
    assert receipt["successor_campaign_authorized"] is False
    assert receipt["contract_sha256"] == native_cpu_progress_sha256_v1(
        {key: value for key, value in receipt.items() if key != "contract_sha256"}
    )


def test_native_core_disabled_reason_semantics_are_frozen() -> None:
    receipt = native_core_ordinary_birth_disabled_reason_semantics_v1()

    assert receipt["contract_id"] == NATIVE_CORE_DISABLED_REASON_SEMANTICS_CONTRACT_ID
    assert receipt["decision"] == "P12-NATIVE-CORE-T-DISABLED-REASON-SEMANTICS"
    assert receipt["answer_key_contract_sha256"] == ANSWER_KEY_CONTRACT_SHA256
    assert tuple(
        (row["order"], row["native_code"], row["canonical_reason"])
        for row in receipt["canonical_disabled_reason_enum"]
    ) == (
        (0, 1, "operator_disabled"),
        (1, 2, "structural_mutation_disabled"),
        (2, 3, "recurrent_mutation_disabled"),
        (3, 4, "operator_family_disabled"),
        (4, 5, "zero_mutation_rate"),
        (5, 6, "mutation_attempt_budget_zero"),
    )
    assert receipt["nullability"] == {
        "v4_disabled_receipt": "disabled_reason_code_required_1_to_6",
        "v4_non_disabled_receipt": "disabled_reason_code_zero",
        "v3_disabled_receipt": "disabled_reason_code_absent_defaults_operator_disabled",
        "malformed_input_envelope": "disabled_reason_absent",
    }
    assert {
        "native_v4_entrypoint_rejects_codes_outside_0_to_6",
        "bridge_reader_fails_closed_for_unknown_disabled_reason_code",
        "bridge_reader_fails_closed_for_negative_or_non_int_disabled_reason_code",
        "readers_must_not_infer_missing_v4_reason_from_unrelated_fields",
    } == set(receipt["unknown_code_behavior"])
    assert tuple(row["condition"] for row in receipt["terminal_precedence"]) == (
        "malformed_or_missing_input",
        "invalid_draw_bits",
        "owner_or_slot_token_mismatch",
        "capacity_unavailable",
        "mutation_disabled",
        "proposal_generation",
        "publication_rejected",
    )
    assert {
        "invalid_draw_precedes_disabled_reason",
        "owner_rejection_precedes_disabled_reason",
        "capacity_rejection_precedes_disabled_reason",
        "disabled_receipts_do_not_construct_proposal_digest",
        "publication_rejection_is_not_a_disabled_reason",
    } <= set(receipt["covered_behaviors"])
    assert {
        "semantics_freeze_only_not_rowv1_projection",
        "not_native_version_reconciliation_table",
        "not_checkpoint_resume_repeat_determinism",
        "not_cuda_array_layout_resolution",
        "no_route_package_qualification",
    } <= set(receipt["limits"])
    assert receipt["native_cpu_a_authorized"] is False
    assert receipt["cuda_promotion_authorized"] is False
    assert receipt["successor_campaign_authorized"] is False
    assert receipt["contract_sha256"] == native_cpu_progress_sha256_v1(
        {key: value for key, value in receipt.items() if key != "contract_sha256"}
    )


def test_native_core_proposal_digest_vectors_are_bounded() -> None:
    receipt = native_core_ordinary_birth_proposal_digest_vectors_v1()

    assert receipt["contract_id"] == NATIVE_CORE_PROPOSAL_DIGEST_VECTORS_CONTRACT_ID
    assert receipt["decision"] == "P12-NATIVE-CORE-R-PROPOSAL-DIGEST-VECTORS"
    assert receipt["answer_key_contract_sha256"] == ANSWER_KEY_CONTRACT_SHA256
    assert {row["case"] for row in receipt["covered_vectors"]} == {
        "committed_publication",
        "capacity_unavailable",
        "publication_failure",
        "owner_mismatch",
        "invalid_draw",
        "mutation_disabled",
    }
    assert {
        "committed_proposal_digest_equals_child_digest_after",
        "publication_failure_preserves_proposal_digest_without_child_publication",
        "capacity_owner_invalid_draw_and_disabled_cases_emit_zero_proposal_digest",
        "proposal_digest_never_equals_parent_or_empty_child_preimage_when_present",
        "noncommitted_cases_preserve_child_digest_after",
    } <= set(receipt["covered_behaviors"])
    assert {
        "native_core_v2_v3_receipt_vectors_only",
        "not_native_mutation_rejected_vector",
        "not_all_disabled_mutation_receipt_byte_parity",
        "no_cuda_promotion",
        "no_route_package_qualification",
    } <= set(receipt["limits"])
    assert receipt["native_cpu_a_authorized"] is False
    assert receipt["cuda_promotion_authorized"] is False
    assert receipt["successor_campaign_authorized"] is False
    assert receipt["contract_sha256"] == native_cpu_progress_sha256_v1(
        {key: value for key, value in receipt.items() if key != "contract_sha256"}
    )
