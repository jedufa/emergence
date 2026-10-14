"""P1.2 native CPU parity entry gates.

This module freezes the prerequisites for native CPU parity. It does not
implement native CPU parity and does not authorize CUDA promotion.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Final, Mapping


CONTRACT_ID: Final[str] = "p12-native-cpu-parity-entry-gate-v1_1"
SCHEMA_VERSION: Final[str] = "p12_native_cpu_parity_entry_gate_v1_1"
GOLDEN_FIXTURE_SHA256: Final[str] = (
    "1733234c0fa2b57774b4b1eac56d7331cea6d11bcb5aea3c3de2b985fb43f776"
)
GOLDEN_FIXTURE_FILE_SHA256: Final[str] = (
    "839f00a79d1fe2b80a4e2dd111ad99f950baa35acb3203ea0fbafbea2fd2b870"
)

REQUIRED_ENTRY_GATES: Final[tuple[str, ...]] = (
    "canonical_oracle_contract_and_hash",
    "byte_comparable_mutation_receipt_abi",
    "exact_rng_draw_table_per_operator",
    "complete_rollback_surface",
    "canonical_lineage_homology_ordering",
    "high_id_and_slot_reuse_fixtures",
    "planned_checkpoint_boundaries",
    "native_cpu_parity_matrix",
    "real_ordinary_birth_lifecycle_proof",
)

STRUCTURAL_OPERATOR_DRAW_TABLE: Final[tuple[dict[str, Any], ...]] = (
    {
        "operation": "toggle_dormancy",
        "draw0_range": "[0.00,0.25)",
        "draw_count": 2,
        "changed_dimensions": ("dormant",),
    },
    {
        "operation": "prune_eligible_module",
        "draw0_range": "[0.25,0.50)",
        "draw_count": 2,
        "changed_dimensions": ("module",),
    },
    {
        "operation": "refine_descendant_parameter",
        "draw0_range": "[0.50,0.75)",
        "draw_count": 2,
        "changed_dimensions": (
            "candidate_bias",
            "candidate_input_gain",
            "candidate_state_gain",
            "expression_scale",
            "live_input_gain",
            "live_output_gain",
            "live_retention",
            "output_gain",
            "reset_bias",
            "reset_input_gain",
            "reset_state_gain",
            "update_bias",
            "update_input_gain",
            "update_state_gain",
        ),
    },
    {
        "operation": "refine_descendant_routing_or_expression",
        "draw0_range": "[0.75,1.00)",
        "draw_count": 2,
        "changed_dimensions": (
            "expression_scale",
            "input_binding_gain",
            "output_binding_gain",
        ),
    },
)

ROLLBACK_CASES: Final[tuple[str, ...]] = (
    "capacity_unavailable",
    "child_publication_failed",
    "stale_slot_token",
    "invalid_homology",
    "receipt_replay_drift",
    "checkpoint_write_rejected",
)

COMMON_ORACLE_RECEIPT_FIELDS: Final[tuple[str, ...]] = (
    "schema_version",
    "contract_id",
    "profile_id",
    "transaction_kind",
    "parent_owner",
    "child_owner",
    "rng_draw_budget",
    "draw0_float64_bits",
    "draw1_float64_bits",
    "operation",
    "status",
    "reason",
    "changed_dimension",
    "changed",
    "selected_module_instance_id",
    "selected_homology_id",
    "committed",
    "pre_parent_payload_sha256",
    "child_payload_sha256",
    "parent_preserved",
    "native_cuda_promotion_authorized",
    "oracle_sha256",
)

RECURRENT_ORACLE_EXTENSION_FIELDS: Final[tuple[str, ...]] = (
    "identity_table_sha256",
    "replay_receipt_sha256",
)

REPLAY_RECEIPT_TOP_LEVEL_FIELDS: Final[tuple[str, ...]] = (
    "schema_version",
    "contract_id",
    "profile_id",
    "source_packed_sha256",
    "ancestry_tape_sha256",
    "transactions",
    "canonical_sha256",
)

REPLAY_RECEIPT_TRANSACTION_FIELDS: Final[tuple[str, ...]] = (
    "transaction_kind",
    "transaction_ordinal",
    "parent_agent_id",
    "parent_slot",
    "parent_slot_token",
    "child_agent_id",
    "child_slot",
    "child_slot_token",
    "attempt_ordinal",
    "draw0_float64_bits",
    "draw1_float64_bits",
    "operation",
    "status",
    "reason",
    "selected_instance_id_index",
    "selected_homology_id_index",
    "changed_dimension",
    "changed",
)

ROLLBACK_RESTORED_STATE_FIELDS: Final[tuple[str, ...]] = (
    "parent_payload",
    "child_payload",
    "child_id_allocator",
    "next_agent_id",
    "free_slot_allocator",
    "birth_publication_index",
    "slot_to_agent_id",
    "slot_tokens",
    "alive_mask",
    "agent_grid_occupancy",
    "lineage_table",
    "homology_table",
    "module_identity_table",
    "recurrent_identity_table",
    "eed_cursor_rows",
    "respec_cursor_rows",
    "wdl_compact_rows",
    "checkpoint_bundle",
    "rng_receipt_ledger",
    "ordinary_birth_receipt_ledger",
)

HOMOLOGY_ORDERING_RULES: Final[tuple[str, ...]] = (
    "identity_table is sorted by string value after collecting every "
    "module_instance_id and homology_id",
    "identity_table contains no duplicates",
    "module_instance_id values are unique per parent payload",
    "module_instance_id and homology_id values must be nonempty strings",
    "selected_instance_id_index and selected_homology_id_index must be -1 or "
    "valid indexes into identity_table",
    "accepted mutations must provide both selected indexes",
    "invalid, duplicate, unsorted, or missing identity entries fail closed",
)

CHECKPOINT_PLAN: Final[tuple[dict[str, Any], ...]] = (
    {
        "boundary": "before_mutation",
        "must_bind": ("parent_payload", "owner_slots", "slot_tokens", "rng_state"),
    },
    {
        "boundary": "after_child_publication",
        "must_bind": ("child_payload", "slot_publication", "receipt_commit"),
    },
    {
        "boundary": "after_rollback",
        "must_bind": ROLLBACK_RESTORED_STATE_FIELDS,
    },
    {
        "boundary": "after_slot_reuse_fresh_owner",
        "must_bind": ("new_slot_token", "empty_cursor_rows", "fresh_owner_restore"),
    },
    {
        "boundary": "multicell_continuation",
        "must_bind": ("cell_local_slots", "cell_id", "terminal_prefix_merge"),
    },
)

TRANSACTION_ROLLBACK_PROBE_POINTS: Final[tuple[dict[str, Any], ...]] = (
    {
        "probe": "after_proposal_before_publication",
        "persisted_checkpoint_allowed": False,
        "purpose": "compare proposal bytes and rollback image inside one transaction",
        "must_not_escape": True,
    },
)

CUDA_SHADOW_READER_WRITER_INVENTORY: Final[tuple[dict[str, Any], ...]] = (
    {
        "category": "current_flattened_cuda_routes",
        "paths": (
            "engine/perf/cuda_architecture_b_flattened_controller_v1_kernels.cu",
            "engine/perf/cuda_architecture_b_flattened_wdl_controller_v2_kernels.cu",
        ),
        "dense_lookup_reads": 0,
        "guarded_low_id_mirror_writes_per_route": 2,
        "authority": "non_authoritative_compatibility_projection",
    },
    {
        "category": "current_binding_exports",
        "paths": (
            "engine/perf/cuda_architecture_b_flattened_controller_v1_binding.cpp",
        ),
        "exports_shadow_pointer": True,
        "absence_changes_current_route_abi": True,
        "authority": "abi_compatibility_only",
    },
    {
        "category": "legacy_dense_lookup_debt",
        "paths": (
            "engine/perf/cuda_architecture_b_phased_controller_v1_kernels.cu",
            "engine/perf/cuda_architecture_b_stackless_birth_generation_kernel.cu",
            "engine/perf/cuda_architecture_b_persistent_generation_kernel.cu",
        ),
        "dense_lookup_reads": "present",
        "authority": "deprecated_debt_not_current_route_authority",
    },
    {
        "category": "native_cpu_and_reporting_readers",
        "readers": (
            {
                "path": "engine/perf/native/cpu_compact_summary_reporting_ledger_v1.cpp",
                "classification": "reporting-ledger compatibility reader",
                "authoritative_for_high_id": False,
            },
            {
                "path": "engine/perf/native/cxxrewrite4_upstream_pack_kernel.cpp",
                "classification": "legacy upstream-pack dense lookup reader",
                "authoritative_for_high_id": False,
            },
            {
                "path": "engine/perf/native/cxxbackbone8r_generation_island.cpp",
                "classification": "legacy generation-island dense lookup reader",
                "authoritative_for_high_id": False,
            },
            {
                "path": "engine/perf/native/cxxbackbone8r_generation_island_cuda.cu",
                "classification": "legacy CUDA generation-island dense lookup reader",
                "authoritative_for_high_id": False,
            },
        ),
        "dense_lookup_reads": "present",
        "authority": "must_not_be_used_for_high_id_public_authority",
    },
)

COMMON_PARITY_COMPARE_FIELDS: Final[tuple[str, ...]] = (
    "rng_draw_bits",
    "proposal_bytes",
    "child_agent_id",
    "parent_slot",
    "child_slot",
    "parent_slot_token",
    "child_slot_token",
    "publication_state",
    "rollback_state",
    "structural_payload_sha256",
    "recurrent_payload_sha256",
    "cursor_rows",
    "receipt_bytes",
    "checkpoint_bytes",
)

ROLLBACK_PARITY_COMPARE_FIELDS: Final[tuple[str, ...]] = COMMON_PARITY_COMPARE_FIELDS + (
    "child_id_allocator",
    "next_agent_id",
    "free_slot_allocator",
    "lineage_table",
    "homology_table",
)

PARITY_MATRIX: Final[tuple[dict[str, Any], ...]] = (
    {
        "case_id": "structural_toggle_dormancy_success",
        "oracle": "structural",
        "operation": "toggle_dormancy",
        "requires_real_lifecycle": True,
        "checkpoint_boundary": "after_child_publication",
        "compare_fields": COMMON_PARITY_COMPARE_FIELDS,
    },
    {
        "case_id": "structural_prune_eligible_module_success",
        "oracle": "structural",
        "operation": "prune_eligible_module",
        "requires_real_lifecycle": True,
        "checkpoint_boundary": "after_child_publication",
        "compare_fields": COMMON_PARITY_COMPARE_FIELDS,
    },
    {
        "case_id": "structural_refine_descendant_parameter_success",
        "oracle": "structural",
        "operation": "refine_descendant_parameter",
        "requires_real_lifecycle": True,
        "checkpoint_boundary": "after_child_publication",
        "compare_fields": COMMON_PARITY_COMPARE_FIELDS,
    },
    {
        "case_id": "structural_refine_routing_or_expression_success",
        "oracle": "structural",
        "operation": "refine_descendant_routing_or_expression",
        "requires_real_lifecycle": True,
        "checkpoint_boundary": "after_child_publication",
        "compare_fields": COMMON_PARITY_COMPARE_FIELDS,
    },
    {
        "case_id": "recurrent_receipt_replay_high_id_success",
        "oracle": "recurrent",
        "operation": "refine_descendant_parameter",
        "requires_real_lifecycle": True,
        "checkpoint_boundary": "after_child_publication",
        "compare_fields": COMMON_PARITY_COMPARE_FIELDS,
    },
    {
        "case_id": "recurrent_flags_off_disabled_clone",
        "oracle": "recurrent",
        "operation": "refine_descendant_parameter",
        "requires_real_lifecycle": True,
        "checkpoint_boundary": "after_child_publication",
        "compare_fields": COMMON_PARITY_COMPARE_FIELDS,
    },
    {
        "case_id": "structural_zero_mutation_rate_disabled_clone",
        "oracle": "structural",
        "operation": "refine_descendant_parameter",
        "requires_real_lifecycle": True,
        "checkpoint_boundary": "after_child_publication",
        "compare_fields": COMMON_PARITY_COMPARE_FIELDS,
    },
    {
        "case_id": "slot_reuse_new_token_success",
        "oracle": "cursor",
        "operation": "cursor_rebind",
        "requires_real_lifecycle": True,
        "checkpoint_boundary": "after_slot_reuse_fresh_owner",
        "compare_fields": COMMON_PARITY_COMPARE_FIELDS,
    },
    {
        "case_id": "capacity_unavailable_rollback",
        "oracle": "structural",
        "operation": "rollback",
        "requires_real_lifecycle": True,
        "checkpoint_boundary": "before_child_publication",
        "compare_fields": ROLLBACK_PARITY_COMPARE_FIELDS,
    },
    {
        "case_id": "publication_failure_rollback",
        "oracle": "structural",
        "operation": "rollback",
        "requires_real_lifecycle": True,
        "checkpoint_boundary": "before_child_publication",
        "compare_fields": ROLLBACK_PARITY_COMPARE_FIELDS,
    },
    {
        "case_id": "stale_slot_token_rejected_before_mutation",
        "oracle": "structural",
        "operation": "rollback",
        "requires_real_lifecycle": True,
        "checkpoint_boundary": "before_mutation",
        "compare_fields": ROLLBACK_PARITY_COMPARE_FIELDS,
    },
    {
        "case_id": "invalid_homology_rejected",
        "oracle": "recurrent",
        "operation": "rollback",
        "requires_real_lifecycle": True,
        "checkpoint_boundary": "before_child_publication",
        "compare_fields": ROLLBACK_PARITY_COMPARE_FIELDS,
    },
    {
        "case_id": "recurrent_duplicate_identity_table_rejected",
        "oracle": "recurrent",
        "operation": "rollback",
        "requires_real_lifecycle": True,
        "checkpoint_boundary": "before_child_publication",
        "compare_fields": ROLLBACK_PARITY_COMPARE_FIELDS,
    },
    {
        "case_id": "recurrent_missing_identity_table_entry_rejected",
        "oracle": "recurrent",
        "operation": "rollback",
        "requires_real_lifecycle": True,
        "checkpoint_boundary": "before_child_publication",
        "compare_fields": ROLLBACK_PARITY_COMPARE_FIELDS,
    },
    {
        "case_id": "recurrent_selected_index_out_of_range_rejected",
        "oracle": "recurrent",
        "operation": "rollback",
        "requires_real_lifecycle": True,
        "checkpoint_boundary": "before_child_publication",
        "compare_fields": ROLLBACK_PARITY_COMPARE_FIELDS,
    },
    {
        "case_id": "recurrent_receipt_operation_draw_mismatch_rejected",
        "oracle": "recurrent",
        "operation": "rollback",
        "requires_real_lifecycle": True,
        "checkpoint_boundary": "before_child_publication",
        "compare_fields": ROLLBACK_PARITY_COMPARE_FIELDS,
    },
    {
        "case_id": "recurrent_receipt_replay_drift_rejected",
        "oracle": "recurrent",
        "operation": "rollback",
        "requires_real_lifecycle": True,
        "checkpoint_boundary": "before_child_publication",
        "compare_fields": ROLLBACK_PARITY_COMPARE_FIELDS,
    },
    {
        "case_id": "deterministic_repeat_receipt_bytes",
        "oracle": "structural_and_recurrent",
        "operation": "repeat",
        "requires_real_lifecycle": True,
        "checkpoint_boundary": "after_child_publication",
        "compare_fields": COMMON_PARITY_COMPARE_FIELDS,
    },
    {
        "case_id": "checkpoint_before_mutation_resume",
        "oracle": "structural_and_recurrent",
        "operation": "checkpoint_resume",
        "requires_real_lifecycle": True,
        "checkpoint_boundary": "before_mutation",
        "compare_fields": COMMON_PARITY_COMPARE_FIELDS,
    },
    {
        "case_id": "checkpoint_after_mutation_resume",
        "oracle": "structural_and_recurrent",
        "operation": "checkpoint_resume",
        "requires_real_lifecycle": True,
        "checkpoint_boundary": "after_child_publication",
        "compare_fields": COMMON_PARITY_COMPARE_FIELDS,
    },
    {
        "case_id": "fresh_owner_restore_after_slot_reuse",
        "oracle": "cursor",
        "operation": "cursor_rebind",
        "requires_real_lifecycle": True,
        "checkpoint_boundary": "after_slot_reuse_fresh_owner",
        "compare_fields": COMMON_PARITY_COMPARE_FIELDS,
    },
    {
        "case_id": "multicell_cell_local_slot_remap",
        "oracle": "cursor",
        "operation": "multicell_remap",
        "requires_real_lifecycle": True,
        "checkpoint_boundary": "multicell_continuation",
        "compare_fields": COMMON_PARITY_COMPARE_FIELDS,
    },
    {
        "case_id": "high_id_sparse_gap_no_shadow_lookup",
        "oracle": "public_identity",
        "operation": "high_id_relabel",
        "requires_real_lifecycle": True,
        "checkpoint_boundary": "after_child_publication",
        "compare_fields": COMMON_PARITY_COMPARE_FIELDS,
    },
)


def canonical_entry_gate_bytes_v1(value: Mapping[str, Any]) -> bytes:
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


def entry_gate_sha256_v1(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_entry_gate_bytes_v1(value)).hexdigest()


def native_cpu_parity_entry_gate_payload_v1() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "contract_id": CONTRACT_ID,
        "decision": "P12-NATIVE-CPU-ENTRY-GATE",
        "required_entry_gates": REQUIRED_ENTRY_GATES,
        "oracle_contracts": {
            "structural": {
                "contract_id": "p12-structural-ordinary-birth-reference-oracle-v1",
                "contract_sha256": (
                    "dba904ef4b34cb8eca4db49ab3ddfe60854c8b3f641a5c9b4fee7fd375deea90"
                ),
                "contract_sha256_scope": "aggregate_stage2_mutation_oracles_v1",
            },
            "recurrent": {
                "contract_id": "p12-recurrent-ordinary-birth-reference-oracle-v1",
                "contract_sha256": (
                    "dba904ef4b34cb8eca4db49ab3ddfe60854c8b3f641a5c9b4fee7fd375deea90"
                ),
                "contract_sha256_scope": (
                    "aggregate_stage2_mutation_oracles_v1_shared_with_structural"
                ),
            },
            "cursor_eed": {
                "contract_id": "p12-eed-slot-token-cursor-v1",
                "contract_sha256": (
                    "c78afef8400c5599b977e07f5ac32491708617307c5411f709f09e308ecb8d6a"
                ),
            },
            "cursor_respec": {
                "contract_id": "p12-respec-slot-token-cursor-v1",
                "contract_sha256": (
                    "25c5c829a1fa5410c03eed640069024ca06db15fb42e78850564fb95fadaa245"
                ),
            },
        },
        "golden_fixtures": {
            "fixture_sha256": GOLDEN_FIXTURE_SHA256,
            "fixture_file_sha256": GOLDEN_FIXTURE_FILE_SHA256,
            "path": (
                "evidences/P12_SUBSTRATE_GATES_20260727/"
                "p12_stage2_mutation_oracle_golden_fixtures_v1.json"
            ),
        },
        "byte_comparable_receipt_abi": {
            "canonical_json": True,
            "ascii": True,
            "newline_terminated": True,
            "hash_algorithm": "sha256",
            "forbidden": "native-only receipt fields without Python oracle field mapping",
            "common_oracle_receipt_fields": COMMON_ORACLE_RECEIPT_FIELDS,
            "recurrent_oracle_extension_fields": RECURRENT_ORACLE_EXTENSION_FIELDS,
            "replay_receipt_top_level_fields": REPLAY_RECEIPT_TOP_LEVEL_FIELDS,
            "replay_receipt_transaction_fields": REPLAY_RECEIPT_TRANSACTION_FIELDS,
        },
        "rng_semantics": {
            "draw_owner": "caller supplies already-accounted unit draws",
            "internal_rng_draws": 0,
            "draw_encoding": "IEEE-754 binary64 lowercase hex in receipt fields",
            "structural_draw_table": STRUCTURAL_OPERATOR_DRAW_TABLE,
            "recurrent_draw_table": STRUCTURAL_OPERATOR_DRAW_TABLE,
            "draw1_selection": "ordinal=floor(draw1*eligible_count); fraction remainder",
            "invalid_draw_policy": "fail_closed_without_mutation",
        },
        "rollback_cases": ROLLBACK_CASES,
        "rollback_restored_state_fields": ROLLBACK_RESTORED_STATE_FIELDS,
        "lineage_homology_ordering": HOMOLOGY_ORDERING_RULES,
        "high_id_and_slot_reuse": {
            "high_id_child_required": True,
            "slot_reuse_requires_newer_token": True,
            "stale_token_rejected": True,
        },
        "checkpoint_plan": CHECKPOINT_PLAN,
        "transaction_rollback_probe_points": TRANSACTION_ROLLBACK_PROBE_POINTS,
        "parity_matrix": PARITY_MATRIX,
        "cuda_shadow_reader_writer_inventory": CUDA_SHADOW_READER_WRITER_INVENTORY,
        "real_lifecycle_requirement": (
            "native parity must execute the real ordinary-birth transaction; "
            "helper-only mutation equality is insufficient"
        ),
        "cuda_promotion_authorized": False,
    }


def native_cpu_parity_entry_gate_v1() -> dict[str, Any]:
    payload = native_cpu_parity_entry_gate_payload_v1()
    payload["contract_sha256"] = entry_gate_sha256_v1(payload)
    return payload


__all__ = [
    "CONTRACT_ID",
    "native_cpu_parity_entry_gate_v1",
    "native_cpu_parity_entry_gate_payload_v1",
    "canonical_entry_gate_bytes_v1",
    "entry_gate_sha256_v1",
]
