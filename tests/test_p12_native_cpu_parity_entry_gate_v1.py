from __future__ import annotations

import json

from engine.evolution.p12_native_cpu_parity_entry_gate_v1 import (
    CONTRACT_ID,
    canonical_entry_gate_bytes_v1,
    entry_gate_sha256_v1,
    native_cpu_parity_entry_gate_payload_v1,
    native_cpu_parity_entry_gate_v1,
)


def test_entry_gate_contains_bridge_required_topics() -> None:
    gate = native_cpu_parity_entry_gate_v1()

    assert gate["contract_id"] == CONTRACT_ID
    assert gate["schema_version"] == "p12_native_cpu_parity_entry_gate_v1_1"
    assert set(gate["required_entry_gates"]) == {
        "canonical_oracle_contract_and_hash",
        "byte_comparable_mutation_receipt_abi",
        "exact_rng_draw_table_per_operator",
        "complete_rollback_surface",
        "canonical_lineage_homology_ordering",
        "high_id_and_slot_reuse_fixtures",
        "planned_checkpoint_boundaries",
        "native_cpu_parity_matrix",
        "real_ordinary_birth_lifecycle_proof",
    }
    assert gate["cuda_promotion_authorized"] is False
    assert "real ordinary-birth transaction" in gate["real_lifecycle_requirement"]
    assert gate["contract_sha256"]
    assert gate["golden_fixtures"]["fixture_sha256"] == (
        "1733234c0fa2b57774b4b1eac56d7331cea6d11bcb5aea3c3de2b985fb43f776"
    )


def test_rng_draw_table_covers_all_structural_operator_bins() -> None:
    gate = native_cpu_parity_entry_gate_v1()

    assert [row["operation"] for row in gate["rng_semantics"]["structural_draw_table"]] == [
        "toggle_dormancy",
        "prune_eligible_module",
        "refine_descendant_parameter",
        "refine_descendant_routing_or_expression",
    ]
    assert {row["draw_count"] for row in gate["rng_semantics"]["structural_draw_table"]} == {2}
    assert gate["rng_semantics"]["recurrent_draw_table"] == gate["rng_semantics"][
        "structural_draw_table"
    ]
    assert gate["rng_semantics"]["internal_rng_draws"] == 0


def test_oracle_contract_hashes_and_receipt_abi_are_bound() -> None:
    gate = native_cpu_parity_entry_gate_v1()
    contracts = gate["oracle_contracts"]
    receipt_abi = gate["byte_comparable_receipt_abi"]

    assert contracts["structural"]["contract_id"] == (
        "p12-structural-ordinary-birth-reference-oracle-v1"
    )
    assert contracts["structural"]["contract_sha256"] == (
        "dba904ef4b34cb8eca4db49ab3ddfe60854c8b3f641a5c9b4fee7fd375deea90"
    )
    assert contracts["structural"]["contract_sha256_scope"] == (
        "aggregate_stage2_mutation_oracles_v1"
    )
    assert contracts["recurrent"]["contract_sha256_scope"] == (
        "aggregate_stage2_mutation_oracles_v1_shared_with_structural"
    )
    assert contracts["cursor_eed"]["contract_sha256"] == (
        "c78afef8400c5599b977e07f5ac32491708617307c5411f709f09e308ecb8d6a"
    )
    assert contracts["cursor_respec"]["contract_sha256"] == (
        "25c5c829a1fa5410c03eed640069024ca06db15fb42e78850564fb95fadaa245"
    )
    assert {
        "parent_owner",
        "child_owner",
        "draw0_float64_bits",
        "draw1_float64_bits",
        "pre_parent_payload_sha256",
        "child_payload_sha256",
        "oracle_sha256",
    } <= set(receipt_abi["common_oracle_receipt_fields"])
    assert {
        "identity_table_sha256",
        "replay_receipt_sha256",
    } == set(receipt_abi["recurrent_oracle_extension_fields"])
    assert {
        "selected_instance_id_index",
        "selected_homology_id_index",
    } <= set(receipt_abi["replay_receipt_transaction_fields"])


def test_rollback_homology_checkpoint_and_cuda_shadow_inventory_are_explicit() -> None:
    gate = native_cpu_parity_entry_gate_v1()

    assert {
        "parent_payload",
        "child_payload",
        "slot_to_agent_id",
        "slot_tokens",
        "child_id_allocator",
        "next_agent_id",
        "free_slot_allocator",
        "lineage_table",
        "homology_table",
        "agent_grid_occupancy",
        "eed_cursor_rows",
        "respec_cursor_rows",
        "checkpoint_bundle",
    } <= set(gate["rollback_restored_state_fields"])
    assert any("duplicates" in rule for rule in gate["lineage_homology_ordering"])
    assert {row["boundary"] for row in gate["checkpoint_plan"]} >= {
        "after_slot_reuse_fresh_owner",
        "multicell_continuation",
    }
    assert "after_proposal_before_publication" not in {
        row["boundary"] for row in gate["checkpoint_plan"]
    }
    assert gate["transaction_rollback_probe_points"][0] == {
        "probe": "after_proposal_before_publication",
        "persisted_checkpoint_allowed": False,
        "purpose": "compare proposal bytes and rollback image inside one transaction",
        "must_not_escape": True,
    }
    inventory = {row["category"]: row for row in gate["cuda_shadow_reader_writer_inventory"]}
    assert inventory["current_flattened_cuda_routes"]["dense_lookup_reads"] == 0
    assert inventory["current_binding_exports"]["exports_shadow_pointer"] is True
    assert inventory["legacy_dense_lookup_debt"]["dense_lookup_reads"] == "present"
    assert {
        row["path"] for row in inventory["native_cpu_and_reporting_readers"]["readers"]
    } >= {
        "engine/perf/native/cpu_compact_summary_reporting_ledger_v1.cpp",
        "engine/perf/native/cxxrewrite4_upstream_pack_kernel.cpp",
        "engine/perf/native/cxxbackbone8r_generation_island.cpp",
        "engine/perf/native/cxxbackbone8r_generation_island_cuda.cu",
    }
    assert all(
        row["authoritative_for_high_id"] is False
        for row in inventory["native_cpu_and_reporting_readers"]["readers"]
    )


def test_parity_matrix_requires_real_lifecycle_for_every_case() -> None:
    gate = native_cpu_parity_entry_gate_v1()

    assert len(gate["parity_matrix"]) >= 23
    assert all(row["requires_real_lifecycle"] is True for row in gate["parity_matrix"])
    assert all(row["compare_fields"] for row in gate["parity_matrix"])
    assert all("receipt_bytes" in row["compare_fields"] for row in gate["parity_matrix"])
    case_ids = {row["case_id"] for row in gate["parity_matrix"]}
    assert {
        "recurrent_flags_off_disabled_clone",
        "structural_zero_mutation_rate_disabled_clone",
        "invalid_homology_rejected",
        "recurrent_duplicate_identity_table_rejected",
        "recurrent_missing_identity_table_entry_rejected",
        "recurrent_selected_index_out_of_range_rejected",
        "recurrent_receipt_operation_draw_mismatch_rejected",
        "recurrent_receipt_replay_drift_rejected",
        "deterministic_repeat_receipt_bytes",
        "fresh_owner_restore_after_slot_reuse",
        "multicell_cell_local_slot_remap",
        "high_id_sparse_gap_no_shadow_lookup",
    } <= case_ids
    assert {
        "before_child_publication",
        "after_child_publication",
        "after_slot_reuse_fresh_owner",
        "multicell_continuation",
    } <= {row["checkpoint_boundary"] for row in gate["parity_matrix"]}


def test_entry_gate_serialization_is_byte_comparable() -> None:
    wire = canonical_entry_gate_bytes_v1(native_cpu_parity_entry_gate_v1())

    assert wire.endswith(b"\n")
    assert json.loads(wire.decode("ascii"))["contract_id"] == CONTRACT_ID


def test_contract_hash_excludes_self_hash_field() -> None:
    gate = native_cpu_parity_entry_gate_v1()
    payload = native_cpu_parity_entry_gate_payload_v1()

    assert gate["contract_sha256"] == entry_gate_sha256_v1(payload).strip()
    assert gate["contract_sha256"] != entry_gate_sha256_v1(gate)
