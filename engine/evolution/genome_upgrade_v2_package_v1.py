"""Immutable promotion contract for the unified Genome V2 executable package."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final

from engine.authoritative_state.schema import content_sha256
from .genome_upgrade_v2_capability_contract_v1 import (
    GENOME_V2_P10_CAPABILITY_CONTRACT_SHA256,
)


PACKAGE_CONTRACT_SCHEMA_VERSION: Final[str] = (
    "genome_v2_executable_package_v1"
)
PACKAGE_IDENTITY_RECEIPT_SCHEMA_VERSION: Final[str] = (
    "genome_v2_package_identity_receipt_v1"
)
PACKAGE_PROMOTION_RECEIPT_SCHEMA_VERSION: Final[str] = (
    "genome_v2_package_promotion_receipt_v1"
)
CAPACITY_LAYOUT_CONTRACT_SCHEMA_VERSION: Final[str] = (
    "genome_v2_capacity_layout_contract_v1"
)
ROUTE_INPUT_CONTRACT_SCHEMA_VERSION: Final[str] = (
    "genome_v2_route_input_contract_v1"
)
CHECKPOINT_AUTHORITY_BINDING_SCHEMA_VERSION: Final[str] = (
    "genome_v2_checkpoint_authority_binding_v1"
)
NATIVE_CHECKPOINT_AUTHORITY_CONTRACT_SCHEMA_VERSION: Final[str] = (
    "genome_v2_absolute_tick_checkpoint_authority_v1"
)

PACKAGE_OBLIGATIONS: Final[tuple[str, ...]] = (
    "genome_v2_default_off_exact",
    "genome_v2_neutral_profile_exact",
    "genome_v2_reference_native_cuda_expression_parity",
    "genome_v2_inheritance_mutation_parity",
    "genome_v2_ordinary_birth_commit",
    "genome_v2_ordinary_birth_rollback",
    "genome_v2_turnover_commit",
    "genome_v2_turnover_rollback",
    "genome_v2_checkpoint_export",
    "genome_v2_disk_reload",
    "genome_v2_fresh_owner_restore",
    "genome_v2_multicell_restore",
    "genome_v2_cancellation",
    "genome_v2_checkpoint_on_cancel",
    "genome_v2_sticky_failure",
    "genome_v2_no_post_terminal_mutation",
    "genome_v2_chronicle_no_influence",
    "genome_v2_reporting_no_influence",
    "genome_v2_compact_observability_complete",
    "genome_v2_sanitizer_clean",
    "genome_v2_storage_admission",
    "genome_v2_memory_build_admission",
    "genome_v2_route_qualification",
    "genome_v2_clean_final_artifact_rebuild",
    "genome_v2_ecological_probe_readiness",
)
ROUTE_OWNED_OBLIGATIONS: Final[tuple[str, ...]] = (
    "genome_v2_route_qualification",
)
PRE_ROUTE_PACKAGE_OBLIGATIONS: Final[tuple[str, ...]] = tuple(
    obligation
    for obligation in PACKAGE_OBLIGATIONS
    if obligation not in ROUTE_OWNED_OBLIGATIONS
)

_OBLIGATION_DOMAINS: Final[tuple[str, ...]] = (
    "default_off",
    "neutral_profile",
    "expression_parity",
    "inheritance_mutation",
    "ordinary_birth",
    "ordinary_birth",
    "turnover",
    "turnover",
    "checkpoint",
    "checkpoint",
    "restore",
    "restore",
    "cancellation",
    "cancellation",
    "failure",
    "failure",
    "chronicle",
    "reporting",
    "observability",
    "sanitizer",
    "storage",
    "build",
    "route",
    "build",
    "ecological_admission",
)


def obligation_dictionary_v1() -> dict[str, Any]:
    rows = []
    for obligation, domain in zip(
        PACKAGE_OBLIGATIONS, _OBLIGATION_DOMAINS, strict=True
    ):
        route_owned = obligation in ROUTE_OWNED_OBLIGATIONS
        rows.append(
            {
                "obligation_id": obligation,
                "domain": domain,
                "mandatory": True,
                "evidence_owner": (
                    "execution_route_qualification_receipt_v1"
                    if route_owned
                    else PACKAGE_IDENTITY_RECEIPT_SCHEMA_VERSION
                ),
                "pre_route_required": not route_owned,
                "missing_status": "package_incomplete",
                "not_applicable_policy": (
                    "explicit_contract_reason_required"
                ),
            }
        )
    payload = {
        "schema_version": "genome_v2_package_obligation_dictionary_v1",
        "obligations": rows,
    }
    payload["obligation_set_sha256"] = content_sha256(rows)
    return payload


def invalidation_rules_v1() -> dict[str, Any]:
    bound_surfaces = tuple(package_invalidation_field_map_v1())
    return {
        "schema_version": "genome_v2_package_invalidation_rules_v1",
        "bound_surfaces": list(bound_surfaces),
        "identity_rule": (
            "any bound-surface change creates a new immutable package identity"
        ),
        "partial_reuse": "forbidden",
        "dirty_source": "diagnostic_only",
        "stale_evidence": "package_incomplete",
        "route_reuse": {
            "requires_identical_package_id": True,
            "requires_identical_route_hash": True,
            "requires_identical_artifact_sha256": True,
            "mismatch_action": "requalify",
        },
        "promotion_state_machine": {
            "candidate": (
                "all pre-route package obligations passed; qualification-only"
            ),
            "route_qualified": (
                "candidate package plus exact execution-route receipt"
            ),
            "promoted": (
                "route-qualified package plus completed causal-probe readiness"
            ),
        },
    }


def capacity_layout_contract_v1() -> dict[str, Any]:
    """Return the fixed resident-array and qualification envelope contract."""

    return {
        "schema_version": CAPACITY_LAYOUT_CONTRACT_SCHEMA_VERSION,
        "resident_layout": "soa_slot_last_contiguous_v1",
        "resident_dtype": "int64",
        "resident_planes": {
            "alleles_q10": ["cells", 2, 12, 4, "capacity"],
            "modifiers_q10": ["cells", 2, 12, "capacity"],
            "products_q10": ["cells", 12, "capacity"],
            "phenotypes_q10": ["cells", 10, "capacity"],
            "expression_cost_q10": ["cells", "capacity"],
            "active_product_count": ["cells", "capacity"],
            "saturated_product_count": ["cells", "capacity"],
        },
        "bytes_per_cell_slot": 1160,
        "qualification_envelope": {
            "maximum_physical_cells": 48,
            "maximum_resident_capacity": 512,
            "target_architecture": "sm_75",
        },
        "overflow_policy": "reject_before_allocation_no_truncation",
        "unused_capacity_policy": "neutral_initialized_and_dead_slot_sanitized",
    }


def route_input_contract_v1() -> dict[str, Any]:
    """Return package-owned inputs consumed by route qualification."""

    return {
        "schema_version": ROUTE_INPUT_CONTRACT_SCHEMA_VERSION,
        "provider_id": "eiel-production-provider-v1",
        "runtime_profile_id": "eiel-production-runtime-v1",
        "global_genome_profile_id": "genome_upgrade_v2",
        "capability_contract_sha256": (
            GENOME_V2_P10_CAPABILITY_CONTRACT_SHA256
        ),
        "target_architecture": "sm_75",
        "execution_backend": "unified_native_cuda_resident",
        "jit_allowed": False,
        "fallback_allowed": False,
        "package_before_route": True,
    }


def supported_world_controller_config_envelope_v1() -> dict[str, Any]:
    """Return the active absolute-tick world/controller envelope.

    The historical function name is retained for callers at the package
    boundary.  Its payload is deliberately tick-native; generation-shaped
    values belong only in an explicitly private compatibility adapter.
    """

    return {
        "world_base": "patch",
        "world_overlays": [
            "eed_resource_impact",
            "jungle",
            "none",
            "wdl_delayed_hazard",
        ],
        "controller_family": "fixed_mlp",
        **capacity_layout_contract_v1()["qualification_envelope"],
    }


def supported_world_controller_absolute_tick_envelope_v1() -> dict[str, Any]:
    """Return the canonical absolute-tick package support envelope."""

    return supported_world_controller_config_envelope_v1()


def supported_checkpoint_cut_ticks_v1() -> list[str]:
    """Return the checkpoint boundary forms supported by the package."""

    return [
        "terminal_at_total_ticks",
        "cancellation_at_absolute_tick",
    ]


def checkpoint_authority_contract_v1() -> dict[str, Any]:
    """Return the package's absolute-tick checkpoint authority contract."""

    return {
        "schema_version": NATIVE_CHECKPOINT_AUTHORITY_CONTRACT_SCHEMA_VERSION,
        "authority": "absolute_tick_v1",
        "request_fields": ["total_ticks", "checkpoint_ticks"],
        "restore_cursor_field": "next_absolute_tick",
        "span_semantics": "half_open_absolute_tick_span",
        "compatibility_policy": "legacy_compatibility_optional_non_authoritative",
    }


def validate_capacity_layout_admission_v1(
    *,
    cells: object,
    capacity: object,
    total_ticks: object,
) -> dict[str, int]:
    """Validate resident dimensions against the absolute-tick envelope."""

    values = {
        "physical_cells": cells,
        "resident_capacity": capacity,
        "total_ticks": total_ticks,
    }
    if any(type(value) is not int or value < 1 for value in values.values()):
        raise ValueError("Genome V2 capacity dimensions must be positive integers")
    envelope = capacity_layout_contract_v1()["qualification_envelope"]
    limits = {
        "physical_cells": envelope["maximum_physical_cells"],
        "resident_capacity": envelope["maximum_resident_capacity"],
    }
    exceeded = [
        name for name, value in values.items() if name in limits and value > limits[name]
    ]
    if exceeded:
        raise ValueError(
            "Genome V2 package envelope exceeded: " + ", ".join(exceeded)
        )
    return {name: int(value) for name, value in values.items()}


def package_invalidation_field_map_v1() -> dict[str, tuple[str, ...]]:
    """Map normative surfaces to the exact immutable identity fields."""

    return {
        "source": ("source_commit", "source_tree_sha256", "source_clean"),
        "provider_runtime": ("provider_id", "runtime_profile_id"),
        "authoritative_state_schema": (
            "authoritative_state_schema_version",
            "authoritative_state_schema_sha256",
        ),
        "global_genome_profile": (
            "global_genome_profile_id",
            "global_genome_profile_sha256",
            "mechanic_contract_sha256",
            "capability_contract_sha256",
        ),
        "genome_envelope": (
            "genome_envelope_sha256",
            "product_schema_sha256",
            "phenotype_schema_sha256",
            "context_schema_sha256",
            "capacity_layout_contract_sha256",
        ),
        "inheritance_and_mutation": (
            "linkage_map_sha256",
            "inheritance_mutation_draw_contract_sha256",
            "inheritance_authority_contract_sha256",
        ),
        "native_transaction_abi": (
            "ordinary_birth_abi_schema",
            "ordinary_birth_abi_sha256",
            "turnover_abi_schema",
            "turnover_abi_sha256",
        ),
        "cancellation_abi": (
            "cancellation_abi_schema",
            "cancellation_abi_sha256",
        ),
        "causal_authority": (
            "causal_settlement_schema_version",
            "causal_settlement_contract_sha256",
            "causal_gate_schema_version",
            "causal_gate_preregistration_sha256",
        ),
        "ecological_authority": (
            "ecological_e1_schema_version",
            "ecological_e1_preregistration_sha256",
        ),
        "terminal_science_authority": (
            "hereditary_capacity_schema_version",
            "hereditary_capacity_preregistration_sha256",
            "environment_switch_schema_version",
            "environment_switch_preregistration_sha256",
        ),
        "checkpoint_contract": (
            "checkpoint_schema_version",
            "checkpoint_schema_sha256",
        ),
        "checkpoint_authority": (
            "checkpoint_authority_contract_sha256",
            "supported_checkpoint_cut_ticks",
        ),
        "observability_contract": (
            "chronicle_schema_version",
            "chronicle_schema_sha256",
            "genome_v2_chronicle_schema_version",
            "genome_v2_chronicle_schema_sha256",
            "reporting_schema_version",
            "reporting_schema_sha256",
            "observability_storage_budget_schema_version",
            "observability_storage_budget_sha256",
        ),
        "package_contract": (
            "executable_contract_sha256",
            "obligation_set_sha256",
            "executable_package_core_sha256",
            "evidence_inventory_sha256",
        ),
        "executable_bytes": (
            "cuda_artifact_sha256",
            "loaded_artifact_sha256",
            "cuda_manifest_sha256",
            "elf_build_id",
        ),
        "toolchain_and_target": (
            "compiler_cuda_torch_environment_sha256",
            "critical_build_flags_sha256",
            "target_architecture",
            "no_jit",
            "no_fallback",
        ),
        "route_inputs": ("route_contract_inputs_sha256",),
        "supported_absolute_tick_envelope": (
            "supported_world_controller_absolute_tick_envelope",
            "supported_campaign_classes",
        ),
    }


def package_invalidation_decision_v1(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare two identity mappings and explain package reuse deterministically."""

    changed_surfaces: list[str] = []
    changed_fields: list[str] = []
    for surface, fields in package_invalidation_field_map_v1().items():
        surface_changes = [
            field for field in fields if baseline.get(field) != candidate.get(field)
        ]
        if surface_changes:
            changed_surfaces.append(surface)
            changed_fields.extend(surface_changes)
    return {
        "schema_version": "genome_v2_package_invalidation_decision_v1",
        "reusable": not changed_fields,
        "action": "reuse" if not changed_fields else "requalify",
        "changed_surfaces": changed_surfaces,
        "changed_fields": changed_fields,
        "unbound_metadata_ignored": True,
    }


def executable_package_contract_v1() -> dict[str, Any]:
    dictionary = obligation_dictionary_v1()
    invalidation = invalidation_rules_v1()
    required_identity_fields = (
        "source_commit",
        "source_tree_sha256",
        "source_clean",
        "diagnostic_only",
        "provider_id",
        "runtime_profile_id",
        "authoritative_state_schema_version",
        "authoritative_state_schema_sha256",
        "global_genome_profile_id",
        "global_genome_profile_sha256",
        "mechanic_contract_sha256",
        "capability_contract_sha256",
        "genome_envelope_sha256",
        "product_schema_sha256",
        "phenotype_schema_sha256",
        "context_schema_sha256",
        "linkage_map_sha256",
        "inheritance_mutation_draw_contract_sha256",
        "inheritance_authority_contract_sha256",
        "ordinary_birth_abi_schema",
        "ordinary_birth_abi_sha256",
        "turnover_abi_schema",
        "turnover_abi_sha256",
        "cancellation_abi_schema",
        "cancellation_abi_sha256",
        "causal_settlement_schema_version",
        "causal_settlement_contract_sha256",
        "causal_gate_schema_version",
        "causal_gate_preregistration_sha256",
        "ecological_e1_schema_version",
        "ecological_e1_preregistration_sha256",
        "hereditary_capacity_schema_version",
        "hereditary_capacity_preregistration_sha256",
        "environment_switch_schema_version",
        "environment_switch_preregistration_sha256",
        "checkpoint_schema_version",
        "checkpoint_schema_sha256",
        "chronicle_schema_version",
        "chronicle_schema_sha256",
        "genome_v2_chronicle_schema_version",
        "genome_v2_chronicle_schema_sha256",
        "reporting_schema_version",
        "reporting_schema_sha256",
        "observability_storage_budget_schema_version",
        "observability_storage_budget_sha256",
        "executable_contract_sha256",
        "obligation_set_sha256",
        "executable_package_core_sha256",
        "evidence_inventory_sha256",
        "cuda_artifact_sha256",
        "loaded_artifact_sha256",
        "cuda_manifest_sha256",
        "elf_build_id",
        "capacity_layout_contract_sha256",
        "checkpoint_authority_contract_sha256",
        "route_contract_inputs_sha256",
        "compiler_cuda_torch_environment",
        "compiler_cuda_torch_environment_sha256",
        "critical_build_flags_sha256",
        "target_architecture",
        "no_jit",
        "no_fallback",
        "supported_world_controller_absolute_tick_envelope",
        "supported_checkpoint_cut_ticks",
        "supported_campaign_classes",
    )
    return {
        "schema_version": PACKAGE_CONTRACT_SCHEMA_VERSION,
        "identity_receipt_schema_version": (
            PACKAGE_IDENTITY_RECEIPT_SCHEMA_VERSION
        ),
        "promotion_receipt_schema_version": (
            PACKAGE_PROMOTION_RECEIPT_SCHEMA_VERSION
        ),
        "provider_id": "eiel-production-provider-v1",
        "runtime_profile_id": "eiel-production-runtime-v1",
        "global_genome_profile_id": "genome_upgrade_v2",
        "required_identity_fields": list(required_identity_fields),
        "required_package_obligations": list(
            PRE_ROUTE_PACKAGE_OBLIGATIONS
        ),
        "route_owned_obligations": list(ROUTE_OWNED_OBLIGATIONS),
        "complete_promotion_obligations": list(PACKAGE_OBLIGATIONS),
        "obligation_set_sha256": dictionary["obligation_set_sha256"],
        "obligation_dictionary_sha256": content_sha256(dictionary),
        "invalidation_rules_sha256": content_sha256(invalidation),
        "qualification_cycle_resolution": (
            "the immutable candidate identity covers every pre-route "
            "obligation; the existing route receipt owns route qualification "
            "without changing the package identity"
        ),
        "causal_evidence_binding": {
            "pre_package": (
                "package identity binds the canonical causal settlement and "
                "C1 preregistration contracts only"
            ),
            "post_route": (
                "event, terminal, and strict-join evidence is mandatory at C1 "
                "with the exact issued package and execution route"
            ),
            "post_route_identity_fields": [
                "package_id_sha256",
                "package_receipt_sha256",
                "execution_route_sha256",
            ],
        },
        "terminal_science_evidence_binding": {
            "pre_package": (
                "package identity binds the hereditary-capacity and "
                "environment-switch preregistration contracts"
            ),
            "post_route": (
                "terminal evidence must join the exact issued package and "
                "qualified execution route"
            ),
        },
        "immutability": "content_addressed_no_partial_reuse",
    }


PACKAGE_CONTRACT_SHA256: Final[str] = content_sha256(
    executable_package_contract_v1()
)


def package_identity_sha256_v1(receipt: Mapping[str, Any]) -> str:
    """Hash every immutable receipt field except the self-address."""

    payload = dict(receipt)
    payload.pop("package_id_sha256", None)
    return content_sha256(payload)


def package_core_identity_sha256_v1(identity: Mapping[str, Any]) -> str:
    """Hash package-defining inputs without evidence inventory self-dependence."""

    payload = dict(identity)
    payload.pop("executable_package_core_sha256", None)
    payload.pop("evidence_inventory_sha256", None)
    return content_sha256(payload)


def checkpoint_authority_binding_v1(
    identity: Mapping[str, Any],
    *,
    package_id_sha256: str,
    package_receipt_sha256: str,
    execution_route_sha256: str,
) -> dict[str, str]:
    """Project exact package and route authority into a checkpoint sidecar."""

    fields = {
        "schema_version": CHECKPOINT_AUTHORITY_BINDING_SCHEMA_VERSION,
        "provider_id": identity.get("provider_id"),
        "runtime_profile_id": identity.get("runtime_profile_id"),
        "global_genome_profile_id": identity.get(
            "global_genome_profile_id"
        ),
        "source_commit": identity.get("source_commit"),
        "executable_package_core_sha256": identity.get(
            "executable_package_core_sha256"
        ),
        "package_id_sha256": package_id_sha256,
        "package_receipt_sha256": package_receipt_sha256,
        "artifact_sha256": identity.get("cuda_artifact_sha256"),
        "manifest_sha256": identity.get("cuda_manifest_sha256"),
        "state_schema_sha256": identity.get(
            "authoritative_state_schema_sha256"
        ),
        "mechanic_contract_sha256": identity.get(
            "mechanic_contract_sha256"
        ),
        "capability_contract_sha256": identity.get(
            "capability_contract_sha256"
        ),
        "cancellation_abi_sha256": identity.get(
            "cancellation_abi_sha256"
        ),
        "observability_schema_sha256": identity.get(
            "observability_storage_budget_sha256"
        ),
        "capacity_layout_contract_sha256": identity.get(
            "capacity_layout_contract_sha256"
        ),
        "execution_route_sha256": execution_route_sha256,
    }
    if any(
        not isinstance(value, str) or not value
        for value in fields.values()
    ):
        raise ValueError("Genome V2 checkpoint authority binding is incomplete")
    return fields


def validate_checkpoint_authority_binding_v1(
    binding: Any,
) -> dict[str, str]:
    """Validate the lossless production projection of package authority."""

    expected_fields = {
        "schema_version",
        "provider_id",
        "runtime_profile_id",
        "global_genome_profile_id",
        "source_commit",
        "executable_package_core_sha256",
        "package_id_sha256",
        "package_receipt_sha256",
        "artifact_sha256",
        "manifest_sha256",
        "state_schema_sha256",
        "mechanic_contract_sha256",
        "capability_contract_sha256",
        "cancellation_abi_sha256",
        "observability_schema_sha256",
        "capacity_layout_contract_sha256",
        "execution_route_sha256",
    }
    try:
        pairs = tuple(binding.items()) if isinstance(binding, Mapping) else tuple(binding)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Genome V2 checkpoint authority binding is malformed"
        ) from exc
    if (
        any(
            not isinstance(pair, tuple)
            or len(pair) != 2
            or not isinstance(pair[0], str)
            or not isinstance(pair[1], str)
            for pair in pairs
        )
        or len({name for name, _ in pairs}) != len(pairs)
    ):
        raise ValueError("Genome V2 checkpoint authority binding is malformed")
    fields = dict(pairs)
    if set(fields) != expected_fields:
        raise ValueError("Genome V2 checkpoint authority binding fields mismatch")
    expected_identity = {
        "schema_version": CHECKPOINT_AUTHORITY_BINDING_SCHEMA_VERSION,
        "provider_id": "eiel-production-provider-v1",
        "runtime_profile_id": "eiel-production-runtime-v1",
        "global_genome_profile_id": "genome_upgrade_v2",
    }
    if any(fields[name] != value for name, value in expected_identity.items()):
        raise ValueError("Genome V2 checkpoint authority identity mismatch")
    sha256_fields = {
        name for name in expected_fields if name.endswith("_sha256")
    }
    if any(
        len(fields[name]) != 64
        or any(character not in "0123456789abcdef" for character in fields[name])
        for name in sha256_fields
    ):
        raise ValueError("Genome V2 checkpoint authority digest mismatch")
    if (
        len(fields["source_commit"]) not in {40, 64}
        or any(
            character not in "0123456789abcdef"
            for character in fields["source_commit"]
        )
    ):
        raise ValueError("Genome V2 checkpoint authority source mismatch")
    return fields


__all__ = [
    "PACKAGE_CONTRACT_SCHEMA_VERSION",
    "PACKAGE_CONTRACT_SHA256",
    "CAPACITY_LAYOUT_CONTRACT_SCHEMA_VERSION",
    "CHECKPOINT_AUTHORITY_BINDING_SCHEMA_VERSION",
    "NATIVE_CHECKPOINT_AUTHORITY_CONTRACT_SCHEMA_VERSION",
    "PACKAGE_IDENTITY_RECEIPT_SCHEMA_VERSION",
    "PACKAGE_OBLIGATIONS",
    "PACKAGE_PROMOTION_RECEIPT_SCHEMA_VERSION",
    "PRE_ROUTE_PACKAGE_OBLIGATIONS",
    "ROUTE_OWNED_OBLIGATIONS",
    "ROUTE_INPUT_CONTRACT_SCHEMA_VERSION",
    "capacity_layout_contract_v1",
    "checkpoint_authority_contract_v1",
    "checkpoint_authority_binding_v1",
    "validate_checkpoint_authority_binding_v1",
    "executable_package_contract_v1",
    "invalidation_rules_v1",
    "obligation_dictionary_v1",
    "package_invalidation_decision_v1",
    "package_invalidation_field_map_v1",
    "package_identity_sha256_v1",
    "package_core_identity_sha256_v1",
    "route_input_contract_v1",
    "supported_checkpoint_cut_ticks_v1",
    "supported_world_controller_absolute_tick_envelope_v1",
    "supported_world_controller_config_envelope_v1",
    "validate_capacity_layout_admission_v1",
]
