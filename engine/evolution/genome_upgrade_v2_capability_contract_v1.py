"""Content-addressed capability contract for Genome Upgrade V2 P1.0."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Final

from engine.authoritative_state.schema import content_sha256
from .genome_upgrade_v2 import (
    CONTEXT_CHANNELS,
    PHENOTYPE_CHANNELS,
    PRODUCT_CHANNELS,
)
from .genome_upgrade_v2_bd_capabilities_v1 import (
    CAPABILITY_CONTRACT_SHA256 as BD_CAPABILITY_CONTRACT_SHA256,
    capability_contract_payload_v1 as bd_capability_contract_payload_v1,
)
from .genome_upgrade_v2_regulatory_network_v1 import (
    REGULATORY_CONTRACT_SHA256,
    regulatory_contract_payload_v1,
)


SCHEMA_VERSION: Final[str] = "genome_v2_p10_capability_contract_v1"
PROFILE_ID: Final[str] = "genome_upgrade_v2_fixed_q10_v1"
PROVIDER_ID: Final[str] = "eiel-production-provider-v1"
RUNTIME_PROFILE_ID: Final[str] = "eiel-production-runtime-v1"

MANDATORY_P10_CAPABILITIES: Final[tuple[str, ...]] = (
    "stable_diploid_inheritance",
    "broader_hereditary_traits_with_real_costs",
    "gene_product_output",
    "polygenic_phenotype_mapping",
    "bounded_regulatory_execution",
    "emergent_dominance_fixtures",
    "modifier_gene_interaction",
    "environmental_expression",
    "pleiotropic_tradeoffs",
    "reference_native_cuda_parity",
    "checkpoint_and_chronicle_authority",
    "ecological_usefulness_under_preregistered_campaigns",
)

ADMITTED_SURFACES: Final[tuple[str, ...]] = (
    "diploid_product_parameter_alleles",
    "diploid_product_modifier_alleles",
    "fixed_product_channels",
    "fixed_polygenic_phenotype_projection",
    "fixed_context_sensitive_expression",
    "fixed_sparse_pleiotropic_projection",
    "expression_energy_cost",
    "linked_product_pair_recombination",
    "bounded_parameter_and_modifier_mutation",
    "authoritative_checkpoint_sidecar",
    "compact_chronicle_projection",
    "unified_native_cuda_resident_route",
    "broader_runtime_trait_projection",
    "fixed_bounded_regulatory_network",
    "bounded_cross_locus_epistasis",
    "compact_polygenic_contribution_attribution",
)

EXCLUDED_SURFACES: Final[tuple[str, ...]] = (
    "direct_action_strategy_genes",
    "free_dominance_scalar",
    "runtime_tensor_resize",
    "dynamic_regulatory_graph_allocation",
    "unbounded_product_or_phenotype_channels",
    "categorical_allele_numeric_averaging",
    "runtime_jit",
    "cpu_fallback_for_production_multiworld",
    "provider_specific_genome_implementation",
    "hidden_rng_draws",
)

DEFERRED_ITEMS: Final[tuple[str, ...]] = (
    "evolvable_regulatory_nodes_and_edges",
    "evolvable_linkage_blocks",
    "linkage_block_fusion_and_splitting",
    "crossover_rate_modifiers",
    "recombination_cold_regions",
    "inversion_like_recombination_suppression",
    "module_duplication_and_splitting",
    "structural_diploidy",
    "graph_crossover",
    "capacity_epoch_expansion",
)


class GenomeV2P10CapabilityContractError(ValueError):
    """The P1.0 capability declaration is absent, stale, or unsupported."""


def _capability_rows() -> list[dict[str, Any]]:
    technical_state = {
        "stable_diploid_inheritance": "admitted_technical_surface",
        "broader_hereditary_traits_with_real_costs": (
            "admitted_technical_surface"
        ),
        "gene_product_output": "admitted_technical_surface",
        "polygenic_phenotype_mapping": "admitted_fixed_prototype",
        "bounded_regulatory_execution": "admitted_technical_surface",
        "emergent_dominance_fixtures": "admitted_technical_surface",
        "modifier_gene_interaction": "admitted_technical_surface",
        "environmental_expression": "admitted_technical_surface",
        "pleiotropic_tradeoffs": "admitted_technical_surface",
        "reference_native_cuda_parity": "admitted_technical_surface",
        "checkpoint_and_chronicle_authority": "admitted_technical_surface",
        "ecological_usefulness_under_preregistered_campaigns": (
            "programme_gate_open"
        ),
    }
    return [
        {
            "capability_id": capability_id,
            "mandatory_for_p10_closeout": True,
            "current_contract_state": technical_state[capability_id],
            "absence_disposition": "p10_not_promotable",
        }
        for capability_id in MANDATORY_P10_CAPABILITIES
    ]


def genome_v2_p10_capability_payload_v1() -> dict[str, Any]:
    """Return the unsigned, canonical P1.0 capability declaration."""

    return {
        "schema_version": SCHEMA_VERSION,
        "profile_id": PROFILE_ID,
        "provider_id": PROVIDER_ID,
        "runtime_profile_id": RUNTIME_PROFILE_ID,
        "programme_scope": "GENOME-UPGRADE-V2/P1.0",
        "mandatory_capabilities": _capability_rows(),
        "surface_admission": {
            "admitted": list(ADMITTED_SURFACES),
            "excluded": list(EXCLUDED_SURFACES),
            "selection_rule": (
                "only exact admitted identifiers are executable; excluded, "
                "deferred, missing, and unknown identifiers fail closed"
            ),
        },
        "deferred_items": [
            {
                "item_id": item_id,
                "state": "not_supported_by_p10",
                "activation_rule": (
                    "new versioned contract, capacity review, implementation, "
                    "and package plus route requalification required"
                ),
            }
            for item_id in DEFERRED_ITEMS
        ],
        "fixed_channel_envelope": {
            "products": list(PRODUCT_CHANNELS),
            "product_count": 12,
            "phenotypes": list(PHENOTYPE_CHANNELS),
            "phenotype_count": 10,
            "contexts": list(CONTEXT_CHANNELS),
            "context_count": 8,
            "cardinality_policy": "exact_not_maximum",
            "unused_or_unnamed_channel_semantics": "forbidden",
            "semantic_headroom": "none",
        },
        "fixed_regulator_envelope": {
            "contract_state": "admitted_technical_surface",
            "product_inputs": 12,
            "body_state_inputs": 2,
            "environmental_inputs": 4,
            "developmental_inputs": 2,
            "hidden_nodes": 4,
            "product_outputs": 12,
            "phenotype_outputs": 10,
            "fixed_connections": 18,
            "maximum_fan_in": 4,
            "update_stride_ticks": 1,
            "node_storage": (
                "stateless_combinational_recomputed_from_checkpointed_inputs"
            ),
            "allowed_edge_kinds": [
                "activate",
                "inhibit",
                "modulate",
            ],
            "dynamic_allocation": False,
            "hidden_rng": False,
            "regulatory_contract_sha256": REGULATORY_CONTRACT_SHA256,
            "regulatory_contract": regulatory_contract_payload_v1(),
        },
        "modifier_declaration": {
            "admitted": "one signed_q10 modifier per homolog and product",
            "effects": list(
                regulatory_contract_payload_v1()[
                    "modifier_mechanisms"
                ]
            ),
            "free_dominance_field": False,
            "epistasis": "two_fixed_bounded_cross_locus_edges",
        },
        "environment_declaration": {
            "admitted_contexts": list(CONTEXT_CHANNELS),
            "normalization": "clamped_signed_q10_truncation_toward_zero_v1",
            "context_can_change": [
                "product_expression",
                "phenotype_projection",
            ],
            "direct_strategy_switch": False,
            "unlisted_context": "unsupported",
        },
        "pleiotropy_declaration": {
            "admitted": "fixed_sparse_product_to_multiple_phenotype_weights",
            "hereditary_mapping_weights": False,
            "evolvable_fan_out": False,
            "required_science_before_closeout": "pleiotropic_tradeoff_campaigns",
        },
        "cost_declaration": {
            "expression": (
                "sum_absolute_active_product_output_times_product_cost_q10"
            ),
            "runtime_debit": "once_per_live_tick",
            "free_beneficial_surface": False,
            "regulatory_node_edge_cost": (
                "active_node_plus_active_edge_plus_signal_magnitude"
            ),
        },
        "numerical_declaration": {
            "encoding": "signed_q10",
            "scale": 1024,
            "normative_value_width_bits": 32,
            "resident_storage_width_bits": 64,
            "intermediate_arithmetic_width_bits": 64,
            "rounding": "integer_truncation_toward_zero",
            "overflow": "reject_or_clamp_at_named_semantic_boundary",
            "storage_width_is_semantic_headroom": False,
        },
        "mutation_declaration": {
            "scope": [
                "allele_activity",
                "allele_quantity",
                "allele_stability",
                "allele_sensitivity",
                "product_modifier",
            ],
            "semantic_draw_count": 180,
            "hidden_draws": 0,
            "bounds": "parameter_specific_q10_clamp",
            "structural_mutation": "not_supported",
        },
        "recombination_declaration": {
            "linkage_blocks": 6,
            "products_per_linkage_block": 2,
            "meiosis_draw_count": 12,
            "homolog_selection": "genome_v2_biparental_gamete_v1",
            "evolvable_linkage": "not_supported",
        },
        "checkpoint_declaration": {
            "authority": "canonical_direct_resident_checkpoint_owner",
            "native_authority_contract": "absolute_tick_v1",
            "request_fields": ["total_ticks", "checkpoint_ticks"],
            "restore_cursor_field": "next_absolute_tick",
            "span_semantics": "half_open_absolute_tick_span",
            "genome_sidecar_required": True,
            "fresh_owner_restore_required": True,
            "multicell_identity_mapping": "explicit_cell_id_all_or_fail",
            "supported_package_cut_ticks": [
                "terminal_at_total_ticks",
                "cancellation_at_absolute_tick",
            ],
        },
        "chronicle_declaration": {
            "projection": "compact_bounded_no_influence",
            "authoritative_genome_state": False,
            "lineage_attribution_required": True,
            "reporting_can_change_trajectory": False,
        },
        "route_declaration": {
            "provider_id": PROVIDER_ID,
            "runtime_profile_id": RUNTIME_PROFILE_ID,
            "backend": "unified_native_cuda_resident",
            "cuda_target": "sm_75",
            "jit": False,
            "fallback": False,
            "physical_route_qualification_cells": [1, 2],
            "physical_48_cell_evidence_role": (
                "exact_b48_route_qualification_envelope"
            ),
        },
        "capacity_declaration": {
            "resident_layout": "soa_slot_last_contiguous_v1",
            "maximum_physical_cells": 48,
            "maximum_resident_capacity_per_cell": 512,
            "ordinary_evolution_can_resize": False,
            "overflow": "reject_before_allocation_no_truncation",
        },
        "cut_point_crosswalk": [
            {
                "package_cut_point": "terminal_at_total_ticks",
                "transaction_owner": "native_cuda_tick_transaction",
                "checkpoint_state": "complete_authoritative_tick_span",
                "cancellation_lifecycle_cases": [
                    "before_resident_dispatch",
                    "before_ordinary_birth_transaction",
                    "during_ordinary_birth_transaction",
                    "immediately_after_ordinary_birth_commit",
                    "before_turnover_transaction",
                    "during_turnover_transaction",
                    "immediately_after_turnover_commit",
                    "before_absolute_tick_boundary_sealing",
                    "supported_absolute_tick_boundary_cancellation",
                ],
            },
            {
                "package_cut_point": "cancellation_at_absolute_tick",
                "transaction_owner": (
                    "canonical_direct_resident_checkpoint_owner"
                ),
                "checkpoint_state": (
                    "durable_exact_checkpoint_before_sticky_terminal_seal"
                ),
                "cancellation_lifecycle_cases": [
                    "checkpoint_on_cancel_preparation",
                    "after_checkpoint_publication_before_terminal_response",
                    "repeated_cancellation_after_acceptance",
                    "contradictory_operation_after_terminal_sealing",
                ],
            },
        ],
        "claims_declaration": {
            "admissible_after_matching_evidence": [
                "richer_hereditary_capacity",
                "hidden_allele_retention",
                "polygenic_genotype_to_phenotype_mapping",
                "emergent_dominance_mechanics",
                "regulatory_plasticity",
                "pleiotropic_tradeoffs",
                "bounded_ecological_usefulness",
            ],
            "excluded": [
                "intelligence",
                "consciousness",
                "general_cognition",
                "open_ended_evolution",
                "speciation",
                "universal_biological_realism",
                "superiority_over_natural_genetics",
                "superintelligence",
            ],
            "technical_admission_is_scientific_evidence": False,
        },
        "bd_capability_contract_sha256": BD_CAPABILITY_CONTRACT_SHA256,
        "bd_capability_contract": bd_capability_contract_payload_v1(),
    }


@dataclass(frozen=True, slots=True)
class GenomeV2P10CapabilityContract:
    """Validated immutable wrapper for the canonical contract payload."""

    payload: Mapping[str, Any]
    contract_sha256: str

    @classmethod
    def v1(cls) -> GenomeV2P10CapabilityContract:
        payload = genome_v2_p10_capability_payload_v1()
        return cls(
            payload=deepcopy(payload),
            contract_sha256=content_sha256(payload),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            **deepcopy(dict(self.payload)),
            "contract_sha256": self.contract_sha256,
        }


def validate_genome_v2_p10_capability_contract_v1(
    value: Mapping[str, Any],
) -> GenomeV2P10CapabilityContract:
    """Validate exact content and hash; extensions require a new contract."""

    if not isinstance(value, Mapping):
        raise GenomeV2P10CapabilityContractError("contract must be a mapping")
    supplied = deepcopy(dict(value))
    claimed_hash = supplied.pop("contract_sha256", None)
    expected = genome_v2_p10_capability_payload_v1()
    if supplied != expected:
        raise GenomeV2P10CapabilityContractError(
            "contract payload is unsupported or differs from P1.0"
        )
    expected_hash = content_sha256(expected)
    if claimed_hash != expected_hash:
        raise GenomeV2P10CapabilityContractError(
            "contract_sha256 does not match canonical P1.0 payload"
        )
    return GenomeV2P10CapabilityContract(
        payload=deepcopy(expected),
        contract_sha256=expected_hash,
    )


def validate_genome_v2_p10_surface_selection_v1(
    surface_ids: Sequence[str],
) -> tuple[str, ...]:
    """Admit an exact, duplicate-free subset of declared executable surfaces."""

    if isinstance(surface_ids, (str, bytes)):
        raise GenomeV2P10CapabilityContractError(
            "surface selection must be a sequence of identifiers"
        )
    selected = tuple(surface_ids)
    if any(not isinstance(item, str) or not item for item in selected):
        raise GenomeV2P10CapabilityContractError(
            "surface identifiers must be nonempty strings"
        )
    if len(set(selected)) != len(selected):
        raise GenomeV2P10CapabilityContractError(
            "surface selection contains duplicates"
        )
    unsupported = sorted(set(selected) - set(ADMITTED_SURFACES))
    if unsupported:
        raise GenomeV2P10CapabilityContractError(
            "unsupported Genome V2 P1.0 surfaces: " + ", ".join(unsupported)
        )
    return selected


GENOME_V2_P10_CAPABILITY_CONTRACT_SHA256: Final[str] = (
    GenomeV2P10CapabilityContract.v1().contract_sha256
)


__all__ = [
    "ADMITTED_SURFACES",
    "DEFERRED_ITEMS",
    "EXCLUDED_SURFACES",
    "GENOME_V2_P10_CAPABILITY_CONTRACT_SHA256",
    "MANDATORY_P10_CAPABILITIES",
    "PROFILE_ID",
    "PROVIDER_ID",
    "RUNTIME_PROFILE_ID",
    "SCHEMA_VERSION",
    "GenomeV2P10CapabilityContract",
    "GenomeV2P10CapabilityContractError",
    "genome_v2_p10_capability_payload_v1",
    "validate_genome_v2_p10_capability_contract_v1",
    "validate_genome_v2_p10_surface_selection_v1",
]
