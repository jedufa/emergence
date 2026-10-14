"""Frozen promotion-test surfaces for the unified Genome V2 package."""

from __future__ import annotations

from typing import Any, Final

from evolution.genome_upgrade_v2_cancellation_v1 import (
    CANCELLATION_LIFECYCLE_CASES,
    cancellation_contract_v1,
    cancellation_lifecycle_case_v1,
)


SCHEMA_PREFIX: Final[str] = "genome_v2"


def _contract(
    name: str,
    *,
    owner: str,
    stages: tuple[str, ...],
    invariants: tuple[str, ...],
) -> dict[str, Any]:
    return {
        "schema_version": f"{SCHEMA_PREFIX}_{name}_contract_v1",
        "provider_id": "eiel-production-provider-v1",
        "runtime_profile_id": "eiel-production-runtime-v1",
        "global_genome_profile_id": "genome_upgrade_v2",
        "owner": owner,
        "ordered_stages": list(stages),
        "invariants": list(invariants),
        "evidence_policy": "runtime_receipt_required_no_static_pass",
    }


def promotion_contract_artifacts_v1() -> dict[str, dict[str, Any]]:
    birth = _contract(
        "birth_transaction",
        owner="unified_resident_ordinary_birth_transaction_v2",
        stages=(
            "parent_selection",
            "destination_reservation",
            "inherited_genotype_staging",
            "product_modifier_staging",
            "phenotype_staging",
            "context_initialization",
            "runtime_phenotype_projection",
            "parent_debit",
            "child_energy_assignment",
            "lineage_id_staging",
            "resident_shadow_write",
            "authoritative_commit",
            "chronicle_event_emission",
        ),
        invariants=(
            "failure_before_commit_has_no_visible_child",
            "failure_after_commit_has_one_complete_child_and_event",
            "no_slot_id_lineage_or_rng_leak",
            "all_seven_genome_v2_planes_commit_or_rollback_together",
        ),
    )
    turnover = _contract(
        "turnover_transaction",
        owner="unified_resident_generation_turnover_v1",
        stages=(
            "cull_result",
            "refill_parent_pool",
            "destination_allocation",
            "inherited_genotype",
            "mutation",
            "product_phenotype_projection",
            "state_initialization",
            "lineage",
            "checkpoint_boundary",
            "authoritative_commit",
            "chronicle_projection",
        ),
        invariants=(
            "all_seven_genome_v2_planes_commit_or_rollback_together",
            "failed_turnover_has_no_partial_refill_or_lineage",
            "post_commit_failure_preserves_one_complete_turnover",
        ),
    )
    checkpoint = _contract(
        "checkpoint_authority",
        owner="canonical_direct_resident_checkpoint_owner",
        stages=(
            "settle_authoritative_state",
            "export_live_arrays",
            "bind_cell_sidecar_identity",
            "bind_package_route_artifact_schema",
            "serialize",
            "atomic_disk_publish",
        ),
        invariants=(
            "live_arrays_not_reconstructed_projections_are_authority",
            "package_route_artifact_and_schema_identity_are_exact",
            "partial_or_mixed_sidecars_fail_closed",
            "transaction_scratch_is_never_checkpoint_authority",
        ),
    )
    multicell = _contract(
        "multicell_sidecar",
        owner="inheritance_sidecar_identity_v1",
        stages=(
            "collect_complete_cell_identity_set",
            "verify_each_payload_checksum",
            "map_by_explicit_cell_id",
            "allocate_fresh_banks",
            "apply_all_or_fail_without_publication",
        ),
        invariants=(
            "incidental_sidecar_order_has_no_authority",
            "one_failed_cell_cannot_mutate_another",
            "missing_duplicate_extra_stale_or_mixed_sets_fail_closed",
        ),
    )
    route = _contract(
        "route",
        owner="execution_route_qualification_v1",
        stages=(
            "package_candidate_complete",
            "route_requirement_resolution",
            "bounded_obligation_canary",
            "route_receipt_publish",
            "campaign_admission",
        ),
        invariants=(
            "route_qualification_never_runs_as_normal_campaign_side_effect",
            "route_receipt_requires_identical_package_and_artifact",
            "bound_surface_change_revokes_reuse",
        ),
    )
    ecological = {
        "schema_version": "genome_v2_ecological_campaign_boundary_v1",
        "admission_order": [
            "immutable_package_candidate",
            "exact_route_qualification",
            "fixed_genotype_causal_probes",
            "preregistered_bounded_ecological_campaign",
        ],
        "causal_probe_requirements": [
            "one_surface_at_a_time",
            "matched_non_genome_inputs",
            "trajectory_and_rng_identity_recorded",
            "consequence_and_cap_binding_observed",
            "checkpoint_resume_repeat",
        ],
        "campaign_claim_boundary": (
            "bounded ecological usefulness only; no intelligence emergence "
            "consciousness or broad cognition claim"
        ),
        "production_backend": "sm75_cuda_fail_closed",
        "default_workers": 6,
    }
    component_binding = {
        "schema_version": "genome_v2_component_binding_contract_v1",
        "binding_chain": [
            "campaign_registry",
            "global_genome_profile",
            "unified_component_selection",
            "runner_configuration",
            "resident_generation_configuration",
            "authoritative_state_inventory",
            "executable_package",
            "route_requirements",
            "route_qualification",
            "executed_artifact",
            "checkpoint",
            "chronicle_report",
        ],
        "cardinality": "exactly_one_global_genome_profile",
        "partial_match": "forbidden",
        "receipt_state": "contract_frozen_evidence_pending",
    }
    return {
        "genome_v2_birth_transaction_contract.json": birth,
        "genome_v2_turnover_transaction_contract.json": turnover,
        "genome_v2_checkpoint_authority_contract.json": checkpoint,
        "genome_v2_multicell_sidecar_contract.json": multicell,
        "genome_v2_cancellation_contract.json": (cancellation_contract_v1()),
        "genome_v2_route_contract.json": route,
        "genome_v2_ecological_campaign_boundary.json": ecological,
        "genome_v2_component_binding_contract.json": component_binding,
    }


def _failure_rows(
    cases: tuple[str, ...],
    *,
    boundary: str,
    post_commit: frozenset[str] = frozenset(),
) -> list[dict[str, str]]:
    return [
        {
            "case_id": case,
            "boundary": boundary,
            "injected": "required",
            "expected_authority": (
                "one_complete_committed_transaction"
                if case in post_commit
                else "exact_pre_transaction_state"
            ),
            "expected_terminal_state": "sticky_failure",
            "receipt": "required",
        }
        for case in cases
    ]


def promotion_matrix_rows_v1() -> dict[str, list[dict[str, str]]]:
    birth_cases = (
        "after_destination_reservation",
        "after_genotype_staging",
        "after_product_staging",
        "after_phenotype_staging",
        "after_parent_debit",
        "after_child_energy",
        "after_lineage_allocation",
        "after_shadow_write",
        "immediately_before_commit",
        "immediately_after_commit_before_reporting",
    )
    turnover_cases = (
        "mid_cull",
        "after_destination_allocation",
        "after_child_staging",
        "after_mutation",
        "after_phenotype_projection",
        "before_turnover_commit",
        "after_turnover_commit_before_checkpoint",
    )
    registry_cases = (
        "no_global_profile",
        "multiple_global_profiles",
        "profile_without_package",
        "profile_hash_mismatch",
        "provider_runtime_mismatch",
        "state_inventory_mismatch",
        "abi_mismatch",
        "obligation_list_incomplete",
        "evidence_receipt_stale",
        "artifact_mismatch",
        "checkpoint_schema_mismatch",
        "chronicle_schema_mismatch",
        "route_for_other_profile",
        "route_before_package_completion",
        "dirty_source_without_diagnostic_status",
    )
    checkpoint_cases = (
        "exact_same_package",
        "different_package",
        "different_route",
        "different_artifact",
        "different_schema",
        "partial_sidecar",
        "stale_sidecar",
        "checksum_mismatch",
    )
    fresh_owner_cases = (
        "new_allocation",
        "process_restart",
        "same_process_after_other_run",
        "different_prior_batch_size",
        "b1",
        "b4",
        "largest_safe_batch",
        "reporting_enabled_disabled",
        "after_cancellation",
        "after_ordinary_birth",
        "after_turnover",
        "repeated_save_restore_soak",
    )
    multicell_cases = (
        "missing_cell",
        "duplicate_cell",
        "wrong_order_identity_remap",
        "wrong_seed",
        "wrong_profile",
        "mixed_package",
        "mixed_route",
        "mixed_artifact",
        "mixed_schema",
        "stale_sidecar",
        "shape_mismatch",
        "checksum_mismatch",
        "extra_sidecar",
        "partial_restore_set",
    )
    cancellation_cases = CANCELLATION_LIFECYCLE_CASES
    route_cases = (
        "identical_package_route_artifact",
        "package_id_changed",
        "route_hash_changed",
        "artifact_hash_changed",
        "qualification_obligation_changed",
    )
    return {
        "genome_v2_birth_failure_injection_matrix.csv": _failure_rows(
            birth_cases,
            boundary="ordinary_birth",
            post_commit=frozenset({"immediately_after_commit_before_reporting"}),
        ),
        "genome_v2_turnover_failure_injection_matrix.csv": _failure_rows(
            turnover_cases,
            boundary="generation_turnover",
            post_commit=frozenset({"after_turnover_commit_before_checkpoint"}),
        ),
        "genome_v2_registry_binding_matrix.csv": [
            {
                "case_id": case,
                "expected": ("admit" if case == "exact_complete_binding" else "reject"),
                "receipt": "required",
            }
            for case in ("exact_complete_binding", *registry_cases)
        ],
        "genome_v2_fail_closed_admission_matrix.csv": [
            {
                "case_id": case,
                "expected": "reject_before_campaign_launch",
                "receipt": "required",
            }
            for case in registry_cases
        ],
        "genome_v2_checkpoint_compatibility_matrix.csv": [
            {
                "case_id": case,
                "expected": (
                    "exact_restore"
                    if case == "exact_same_package"
                    else "reject_without_mutation"
                ),
                "receipt": "required",
            }
            for case in checkpoint_cases
        ],
        "genome_v2_fresh_owner_restore_matrix.csv": [
            {
                "case_id": case,
                "expected": "exact_arrays_rng_lineage_and_first_transition",
                "receipt": "required",
            }
            for case in fresh_owner_cases
        ],
        "genome_v2_multicell_restore_matrix.csv": [
            {
                "case_id": case,
                "expected": (
                    "exact_identity_remap"
                    if case == "wrong_order_identity_remap"
                    else "reject_without_cross_cell_mutation"
                ),
                "receipt": "required",
            }
            for case in multicell_cases
        ],
        "genome_v2_cancellation_matrix.csv": [
            {
                **cancellation_lifecycle_case_v1(case).to_dict(),
                "receipt": "required",
            }
            for case in cancellation_cases
        ],
        "genome_v2_route_revocation_matrix.csv": [
            {
                "case_id": case,
                "expected": (
                    "reuse"
                    if case == "identical_package_route_artifact"
                    else "requalify"
                ),
                "receipt": "required",
            }
            for case in route_cases
        ],
    }


__all__ = [
    "promotion_contract_artifacts_v1",
    "promotion_matrix_rows_v1",
]
