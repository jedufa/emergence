"""Explicit, auditable registry for declarative campaign runtime profiles."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace
import json
from typing import Final

from .contracts import (
    CampaignRequirementsV1,
    MetricFamilyV1,
    ParameterContractV1,
    RuntimeProfileV1,
    UnsupportedReasonV1,
)
from engine.evolution.genome_upgrade_v2_recomb72_founders_v1 import (
    FOUNDER_PROFILE_IDS as RECOMB72_GV2_FOUNDER_PROFILE_IDS,
)
from .production_engine_identity import PRODUCTION_CPU_PROFILE_ID, PRODUCTION_PROFILE_ID
from .production_control_path_v1 import primitive_parameter_contracts_v1
from .production_engine_lifecycle_v1 import (
    LIFE_HISTORY_PROFILE_INERT_V1,
    LIFE_HISTORY_PROFILE_STAGE1R_QUALIFICATION_V1,
    LIFE_HISTORY_PROFILE_STAGE1S_OVERLAP_V1,
    PATCH_FIXED_MLP_TURNOVER_ONLY_W16P128C256_V1,
    PATCH_FIXED_MLP_TURNOVER_ONLY_W16P64C128_V1,
    PATCH_FIXED_MLP_TURNOVER_ONLY_W32P256C256_V1,
    PATCH_FIXED_MLP_TURNOVER_ONLY_W32P256C512_V1,
    PATCH_FIXED_MLP_TURNOVER_ONLY_W32P512C768_V1,
    PATCH_FIXED_MLP_TURNOVER_ONLY_W32P512C1024_V1,
)

PATCH_ECOLOGY_PROFILE_DEFAULT_V1: Final[str] = "patch_p2_default_v1"
PATCH_ECOLOGY_PROFILE_WD3_RESOURCE_THROUGHPUT_V1: Final[str] = (
    "wd3_resource_throughput_v1"
)
PATCH_INITIAL_SPATIAL_PLACEMENT_HERITAGE_V1: Final[str] = "heritage_random_v1"
PATCH_INITIAL_SPATIAL_PLACEMENT_DISPERSED_LATTICE_V1: Final[str] = (
    "dispersed_lattice_v1"
)
PATCH_INITIAL_SPATIAL_PLACEMENT_DISPERSED_LATTICE_FILL_V1: Final[str] = (
    "dispersed_lattice_fill_v1"
)


def _positive_int(value: int) -> bool:
    return value > 0


def _fraction(value: float) -> bool:
    return 0.0 <= value <= 1.0


def _q10(value: int) -> bool:
    return 0 <= value <= 1024


def _one_of(*values: str):
    accepted = frozenset(values)
    return lambda value: value in accepted


COMMON_PATCH_PARAMETERS: Final[dict[str, ParameterContractV1]] = {
    "patch.grid_size": ParameterContractV1(
        "patch.grid_size", int, _positive_int, "Square world side length", False, True
    ),
    "patch.initial_population": ParameterContractV1(
        "patch.initial_population", int, _positive_int, "Initial residents", False, True
    ),
    "patch.capacity": ParameterContractV1(
        "patch.capacity", int, _positive_int, "Maximum resident slots", False, True
    ),
    "patch.founder_placement": ParameterContractV1(
        "patch.founder_placement",
        str,
        _one_of("legacy", "balanced", "round_robin"),
        "Founder lineage-treatment mode (does not change spatial placement)",
        control_role="lineage_treatment",
        wrc_axis=False,
    ),
    "patch.ecology_profile": ParameterContractV1(
        "patch.ecology_profile",
        str,
        _one_of(
            PATCH_ECOLOGY_PROFILE_DEFAULT_V1,
            PATCH_ECOLOGY_PROFILE_WD3_RESOURCE_THROUGHPUT_V1,
        ),
        "Route-qualified Patch ecology profile",
        False,
        True,
        control_role="bundle_selector",
        wrc_axis=False,
    ),
    "patch.initial_live_count": ParameterContractV1(
        "patch.initial_live_count",
        int,
        lambda value: value in {64, 192, 256, 512},
        "Alias of patch.initial_population; retained for compatibility",
        False,
        True,
        control_role="alias",
        wrc_axis=False,
        alias_of="patch.initial_population",
    ),
    "patch.initial_spatial_placement_profile": ParameterContractV1(
        "patch.initial_spatial_placement_profile",
        str,
        _one_of(
            PATCH_INITIAL_SPATIAL_PLACEMENT_HERITAGE_V1,
            PATCH_INITIAL_SPATIAL_PLACEMENT_DISPERSED_LATTICE_V1,
            PATCH_INITIAL_SPATIAL_PLACEMENT_DISPERSED_LATTICE_FILL_V1,
        ),
        "Route-qualified generation-zero spatial placement profile",
        False,
        True,
    ),
}

GENOME_UPGRADE_V2_PARAMETERS: Final[dict[str, ParameterContractV1]] = {
    "genome_upgrade_v2.profile": ParameterContractV1(
        "genome_upgrade_v2.profile",
        str,
        _one_of("genome_upgrade_v2_fixed_q10_v1"),
        "Fixed Genome V2 genotype-to-phenotype contract",
        False,
        True,
        control_role="route_constant",
        wrc_axis=False,
    ),
    "genome_upgrade_v2.enable": ParameterContractV1(
        "genome_upgrade_v2.enable",
        bool,
        lambda value: value is True,
        "Explicit admission of Genome V2 mechanics",
        False,
        True,
        control_role="route_constant",
        wrc_axis=False,
    ),
    "genome_upgrade_v2.mutation_rate_q10": ParameterContractV1(
        "genome_upgrade_v2.mutation_rate_q10",
        int,
        _q10,
        "Per-locus Genome V2 mutation probability in Q10",
        True,
        True,
    ),
    "genome_upgrade_v2.mutation_step_q10": ParameterContractV1(
        "genome_upgrade_v2.mutation_step_q10",
        int,
        _q10,
        "Genome V2 mutation step in Q10",
        True,
        True,
    ),
    "genome_upgrade_v2.founder_profile": ParameterContractV1(
        "genome_upgrade_v2.founder_profile",
        str,
        _one_of(
            "neutral",
            "pleiotropic_tradeoffs",
            "regulatory_activation_inhibition",
            *RECOMB72_GV2_FOUNDER_PROFILE_IDS,
        ),
        "Frozen Genome V2 generation-zero founder genotype",
        True,
        False,
    ),
    "genome_upgrade_v2.e1_arm_label": ParameterContractV1(
        "genome_upgrade_v2.e1_arm_label",
        str,
        _one_of("control", "treatment"),
        "Non-mechanical E1 matched-arm provenance label",
        True,
        False,
        control_role="provenance_only",
        wrc_axis=False,
    ),
}

PRODUCTION_LIFECYCLE_PARAMETERS: Final[dict[str, ParameterContractV1]] = {
    "production_lifecycle.profile": ParameterContractV1(
        "production_lifecycle.profile",
        str,
        _one_of(
            PATCH_FIXED_MLP_TURNOVER_ONLY_W16P128C256_V1,
            PATCH_FIXED_MLP_TURNOVER_ONLY_W16P64C128_V1,
            PATCH_FIXED_MLP_TURNOVER_ONLY_W32P256C256_V1,
            PATCH_FIXED_MLP_TURNOVER_ONLY_W32P256C512_V1,
            PATCH_FIXED_MLP_TURNOVER_ONLY_W32P512C768_V1,
            PATCH_FIXED_MLP_TURNOVER_ONLY_W32P512C1024_V1,
        ),
        "Explicit bounded production lifecycle profile selector",
        True,
        True,
        control_role="bundle_selector",
        wrc_axis=False,
    ),
    "production_lifecycle.turnover_enabled": ParameterContractV1(
        "production_lifecycle.turnover_enabled",
        bool,
        lambda value: value is True,
        "Frozen production-route turnover admission constant",
        False,
        True,
        control_role="route_constant",
        wrc_axis=False,
    ),
    "production_lifecycle.ordinary_births_enabled": ParameterContractV1(
        "production_lifecycle.ordinary_births_enabled",
        bool,
        lambda value: type(value) is bool,
        "Enable or disable production-engine ordinary live births",
        False,
        True,
    ),
    "production_lifecycle.lifecycle_mode": ParameterContractV1(
        "production_lifecycle.lifecycle_mode",
        str,
        _one_of("legacy_synchronized_truncation_v1", "hybrid_overlapping_v1"),
        "Production lifecycle mode; defaults to qualified Lifecycle V2 when omitted",
        True,
        True,
    ),
    "production_lifecycle.life_history_profile": ParameterContractV1(
        "production_lifecycle.life_history_profile",
        str,
        _one_of(
            LIFE_HISTORY_PROFILE_INERT_V1,
            LIFE_HISTORY_PROFILE_STAGE1R_QUALIFICATION_V1,
            LIFE_HISTORY_PROFILE_STAGE1S_OVERLAP_V1,
        ),
        "Frozen tick-life-history profile for Lifecycle V2",
        True,
        True,
        control_role="bundle_selector",
        wrc_axis=False,
    ),
    "production_lifecycle.turnover_cull_fraction": ParameterContractV1(
        "production_lifecycle.turnover_cull_fraction",
        float,
        lambda value: value in {0.0, 0.5},
        "Mode-derived lifecycle cull fraction",
        False,
        True,
        control_role="derived",
        wrc_axis=False,
        alias_of="production_lifecycle.lifecycle_mode",
    ),
    "production_lifecycle.turnover_parent_fraction": ParameterContractV1(
        "production_lifecycle.turnover_parent_fraction",
        float,
        lambda value: value == 1.0,
        "Frozen share of refill children created by two-parent meiosis",
        False,
        True,
        control_role="route_constant",
        wrc_axis=False,
    ),
}

CUDA_ARCHITECTURE_B_PARAMETERS: Final[dict[str, ParameterContractV1]] = {
    "cuda_architecture_b.cpython_gauss_segment_capacity": ParameterContractV1(
        "cuda_architecture_b.cpython_gauss_segment_capacity",
        int,
        lambda value: value >= 0,
        "Route-bound exact CPython Gaussian values per bounded CUDA segment",
        False,
        True,
    ),
    "cuda_architecture_b.policy_birth_segment_refill_enable": ParameterContractV1(
        "cuda_architecture_b.policy_birth_segment_refill_enable",
        bool,
        lambda value: value is True or value is False,
        "Route-bound policy birth tape segment refill opt-in",
        False,
        True,
    ),
    "cuda_architecture_b.policy_birth_segment_capacity": ParameterContractV1(
        "cuda_architecture_b.policy_birth_segment_capacity",
        int,
        lambda value: value == -1 or value > 0,
        "Route-bound policy birth episodes per bounded CUDA segment",
        False,
        True,
    ),
    "cuda_architecture_b.turnover_mlp_shadow_elide_enable": ParameterContractV1(
        "cuda_architecture_b.turnover_mlp_shadow_elide_enable",
        bool,
        lambda value: value is True or value is False,
        "Route-bound turnover MLP policy shadow elision opt-in",
        False,
        True,
    ),
    "cuda_architecture_b.cpu_request_staging_enable": ParameterContractV1(
        "cuda_architecture_b.cpu_request_staging_enable",
        bool,
        lambda value: value is True or value is False,
        "Route-bound CPU staging for per-cell direct-resident source requests",
        False,
        True,
    ),
    "tw_rngv2.semantics_version": ParameterContractV1(
        "tw_rngv2.semantics_version",
        int,
        lambda value: value in (1, 2),
        "Default-off RNG semantics version; 1 remains production default",
        False,
        True,
    ),
    "tw_rngv2.world_uid": ParameterContractV1(
        "tw_rngv2.world_uid",
        int,
        lambda value: 0 <= value <= 0xFFFFFFFF,
        "Event-addressed V2 world uid for FE birth domains",
        False,
        True,
    ),
    "tw_rngv2.namespace_key64": ParameterContractV1(
        "tw_rngv2.namespace_key64",
        int,
        lambda value: value == 0xCA05117E450A51C0,
        "Explicit canonical event-addressed V2 Philox namespace key",
        False,
        True,
    ),
    "hapb_frag.identity_reuse": ParameterContractV1(
        "hapb_frag.identity_reuse",
        bool,
        lambda value: value is True or value is False,
        "Explicit HAPB-FRAG identity-reuse window path; default off",
        False,
        False,
    ),
    "hapb_frag.force_physical_refill": ParameterContractV1(
        "hapb_frag.force_physical_refill",
        bool,
        lambda value: value is True or value is False,
        "Explicit HAPB-FRAG force-physical policy refill; default off",
        False,
        False,
    ),
    "cuda_architecture_b.qualification_registry_path": ParameterContractV1(
        "cuda_architecture_b.qualification_registry_path",
        str,
        lambda value: bool(value),
        "Candidate CUDA qualification registry path",
        False,
        True,
    ),
    "cuda_architecture_b.promotion_registry_path": ParameterContractV1(
        "cuda_architecture_b.promotion_registry_path",
        str,
        lambda value: bool(value),
        "Candidate CUDA promotion registry path",
        False,
        True,
    ),
}

RECOMB72_PARAMETERS: Final[dict[str, ParameterContractV1]] = {
    **COMMON_PATCH_PARAMETERS,
    "recomb72.inheritance_mode": ParameterContractV1(
        "recomb72.inheritance_mode",
        str,
        _one_of(
            "current_authoritative",
            "parent_copy_per_locus",
            "bounded_interpolation",
            "whole_package_single_parent",
        ),
        "Frozen audit-only inheritance mode",
    ),
    "recomb72.founder_pattern": ParameterContractV1(
        "recomb72.founder_pattern",
        str,
        _one_of(
            "single_extreme_scalar",
            "resource_specialization_extremes",
            "correlated_multi_trait_packages",
        ),
        "Frozen 32/32 hereditary founder package",
    ),
    "recomb72.mutation_enabled": ParameterContractV1(
        "recomb72.mutation_enabled",
        bool,
        lambda value: value is False,
        "Mutation must remain disabled",
        False,
    ),
    "recomb72.ordinary_births_enabled": ParameterContractV1(
        "recomb72.ordinary_births_enabled",
        bool,
        lambda value: value is False,
        "Ordinary live births must remain disabled",
        False,
    ),
    "recomb72.neutral_fixture": ParameterContractV1(
        "recomb72.neutral_fixture",
        bool,
        lambda value: value is True,
        "Genotype-blind neutral turnover fixture",
        False,
    ),
}

EIEL_PRODUCTION_PROFILE_ID: Final[str] = PRODUCTION_PROFILE_ID
EIEL_PRODUCTION_CPU_PROFILE_ID: Final[str] = PRODUCTION_CPU_PROFILE_ID
# Compatibility name for callers migrating from the former provider-specific
# runtime identity. It resolves to the one neutral production runtime.
INHERITANCE_CROSS_PATH_PROFILE_ID: Final[str] = EIEL_PRODUCTION_PROFILE_ID
_DECLARATIVE_RUNTIME_IDENTITY_V1: Final[str] = (
    "campaign_requirements_bound_runtime_identity_v1"
)
_DECLARATIVE_IDENTITY_PLACEHOLDER: Final[str] = "__provider_registry_bound__"

# Physical admission and scalar-coherence are mechanics configuration, not
# execution-provider identity.  These rows preserve each qualified component
# envelope while every campaign resolves to the same production runtime.
_PRODUCTION_EXECUTION_CONFIGS: Final[
    dict[
        tuple[str, frozenset[str], str],
        tuple[int, int, tuple[str, ...], tuple[str, ...]],
    ]
] = {
    ("patch", frozenset(), "fixed_mlp"): (
        72,
        72,
        (),
        ("population", "lineage", "genome_v2"),
    ),
    (
        "patch",
        frozenset({"jungle"}),
        "fixed_mlp",
    ): (72, 1, (), ("population", "lineage")),
    (
        "patch",
        frozenset(),
        "stable_gru",
    ): (
        72,
        1,
        ("policy.hidden_width", "policy.reset_hidden_each_tick"),
        ("population", "policy", "lineage"),
    ),
    (
        "patch",
        frozenset({"wdl_delayed_hazard"}),
        "fixed_mlp",
    ): (
        72,
        72,
        (
            "wdl.delay_ticks",
            "wdl.cue_duration_ticks",
            "wdl.active_duration_ticks",
            "wdl.intensity",
            "wdl.cue_visible_at_decision",
        ),
        ("population", "policy", "lineage", "genome_v2"),
    ),
    (
        "patch",
        frozenset({"eed_resource_impact"}),
        "fixed_mlp",
    ): (
        72,
        48,
        (
            "eed.resource_impact_apply_enable",
            "eed.ecology_frozen",
            "eed.phenotype_fixed",
        ),
        ("population", "genome_v2", "eed"),
    ),
    (
        "patch",
        frozenset({"wdl_delayed_hazard"}),
        "stable_gru",
    ): (
        72,
        1,
        (
            "wdl.delay_ticks",
            "wdl.cue_duration_ticks",
            "wdl.active_duration_ticks",
            "wdl.intensity",
            "wdl.cue_visible_at_decision",
            "policy.hidden_width",
            "policy.reset_hidden_each_tick",
        ),
        ("population", "policy", "lineage"),
    ),
    (
        "patch",
        frozenset({"gmnw_memory_necessity"}),
        "gmnw_policy_arms",
    ): (
        72,
        3,
        ("gmnw.cue_ticks", "gmnw.consequence_ticks"),
        ("population", "policy", "lineage", "gmnw"),
    ),
    ("recomb72_neutral", frozenset(), "audit_only"): (
        72,
        72,
        (),
        ("population", "recomb72"),
    ),
}

WDL_PARAMETERS: Final[dict[str, ParameterContractV1]] = {
    "wdl.delay_ticks": ParameterContractV1(
        "wdl.delay_ticks",
        int,
        _positive_int,
        "Cue-to-consequence delay",
        variability_semantics="experiment_variable",
        variability_route_scopes=("wdl",),
        value_contract={"kind": "int", "minimum": 1, "minimum_inclusive": True},
    ),
    "wdl.cue_duration_ticks": ParameterContractV1(
        "wdl.cue_duration_ticks",
        int,
        _positive_int,
        "Cue duration",
        variability_semantics="experiment_variable",
        variability_route_scopes=("wdl",),
        value_contract={"kind": "int", "minimum": 1, "minimum_inclusive": True},
    ),
    "wdl.active_duration_ticks": ParameterContractV1(
        "wdl.active_duration_ticks",
        int,
        _positive_int,
        "Consequence duration",
        variability_semantics="experiment_variable",
        variability_route_scopes=("wdl",),
        value_contract={"kind": "int", "minimum": 1, "minimum_inclusive": True},
    ),
    "wdl.intensity": ParameterContractV1(
        "wdl.intensity",
        float,
        _fraction,
        "Hazard intensity",
        variability_semantics="experiment_variable",
        variability_route_scopes=("wdl",),
        value_contract={
            "kind": "float",
            "finite_only": True,
            "minimum": 0.0,
            "minimum_inclusive": True,
            "maximum": 1.0,
            "maximum_inclusive": True,
        },
    ),
    "wdl.cue_visible_at_decision": ParameterContractV1(
        "wdl.cue_visible_at_decision",
        bool,
        lambda _: True,
        "Immediate-control cue visibility",
        variability_semantics="experiment_variable",
        variability_route_scopes=("wdl",),
        value_contract={"kind": "bool"},
    ),
}

EED_PARAMETERS: Final[dict[str, ParameterContractV1]] = {
    "eed.founder_resource_axis_profile": ParameterContractV1(
        "eed.founder_resource_axis_profile",
        str,
        _one_of("neutral", "a_high", "b_high", "symmetric_mix"),
        "Generation-zero resource-handling phenotype profile",
        True,
        False,
    ),
    "eed.resource_impact_apply_enable": ParameterContractV1(
        "eed.resource_impact_apply_enable",
        bool,
        lambda _: True,
        "Apply reciprocal resource-depletion impact",
        True,
        True,
    ),
    "eed.ecology_frozen": ParameterContractV1(
        "eed.ecology_frozen",
        bool,
        lambda _: True,
        "Freeze EED ecological stocks while preserving immediate handling",
        True,
        True,
    ),
    "eed.phenotype_fixed": ParameterContractV1(
        "eed.phenotype_fixed",
        bool,
        lambda _: True,
        "Use the declared fixed EED resource-handling phenotype profile",
        True,
        True,
    ),
}

PRODUCTION_PRIMITIVE_PARAMETERS: Final[dict[str, ParameterContractV1]] = (
    primitive_parameter_contracts_v1()
)

POLICY_PARAMETERS: Final[dict[str, ParameterContractV1]] = {
    "policy.hidden_width": ParameterContractV1(
        "policy.hidden_width",
        int,
        lambda value: value == 64,
        "Qualified stable-GRU route constant",
        False,
        True,
        control_role="route_constant",
        wrc_axis=False,
    ),
    "policy.reset_hidden_each_tick": ParameterContractV1(
        "policy.reset_hidden_each_tick",
        bool,
        lambda value: value is False,
        "Qualified stable-GRU route constant",
        False,
        True,
        control_role="route_constant",
        wrc_axis=False,
    ),
}

GMNW_PARAMETERS: Final[dict[str, ParameterContractV1]] = {
    "gmnw.policy_arm": ParameterContractV1(
        "gmnw.policy_arm",
        str,
        _one_of("persistent_gru", "reset_gru", "matched_mlp", "mixed_selection"),
        "Coherent GMNW policy treatment arm",
    ),
    "gmnw.cue_alphabet": ParameterContractV1(
        "gmnw.cue_alphabet",
        int,
        _one_of(2, 4),
        "Native-supported task cue alphabet size",
    ),
    "gmnw.cue_ticks": ParameterContractV1(
        "gmnw.cue_ticks", int, _positive_int, "Cue presentation duration"
    ),
    "gmnw.delay_ticks": ParameterContractV1(
        "gmnw.delay_ticks", int, _positive_int, "Cue-to-consequence delay"
    ),
    "gmnw.consequence_ticks": ParameterContractV1(
        "gmnw.consequence_ticks", int, _positive_int, "Consequence duration"
    ),
    "gmnw.distractors_enabled": ParameterContractV1(
        "gmnw.distractors_enabled", bool, lambda _: True, "Delay-period distractors"
    ),
    "gmnw.cue_visible_at_decision": ParameterContractV1(
        "gmnw.cue_visible_at_decision",
        bool,
        lambda _: True,
        "Immediate-control cue visibility",
    ),
    "gmnw.gru_founder_count": ParameterContractV1(
        "gmnw.gru_founder_count",
        int,
        lambda value: 0 <= value <= 64,
        "Mixed-selection GRU founder allocation in [0,64]",
    ),
    "gmnw.venv_consequence_profile": ParameterContractV1(
        "gmnw.venv_consequence_profile",
        str,
        _one_of("legacy", "P1", "P2", "P3"),
        "Native consequence profile",
        False,
    ),
    "gmnw.terminal_low_population_no_reinit": ParameterContractV1(
        "gmnw.terminal_low_population_no_reinit",
        bool,
        lambda _: True,
        "Disable terminal low-population reinitialization",
        False,
    ),
}

POPULATION_METRICS = MetricFamilyV1(
    "population_v1",
    (
        "population.alive",
        "population.births",
        "population.deaths",
        "population.survival",
        "population.survival_auc",
        "population.terminal_alive",
        "population.successful_births_cumulative",
    ),
    ("population", "lifecycle"),
    "zero counts are measured; ratios with zero denominator are null",
)
GENOME_V2_METRICS = MetricFamilyV1(
    "genome_upgrade_v2_observation_v1",
    (
        "genome_v2.expression_observation_count",
        "genome_v2.context_observation_count",
        "genome_v2.cost_observation_count",
        "genome_v2.cap_observation_count",
    ),
    ("population", "inheritance"),
    "zero counts are measured; unavailable when Genome V2 is disabled",
)
EED_METRICS = MetricFamilyV1(
    "eed_resource_impact_v2",
    (
        "eed.resource_a_stock_total",
        "eed.resource_b_stock_total",
        "eed.resource_a_impact_load_total",
        "eed.resource_b_impact_load_total",
        "eed.resource_a_stock_per_capita",
        "eed.resource_b_stock_per_capita",
        "eed.resource_a_impact_load_per_capita",
        "eed.resource_b_impact_load_per_capita",
        "eed.reproductive_opportunity_successful_births",
        "eed.reproductive_opportunity_survival_auc",
        "eed.pre_birth_opportunity_count",
        "eed.pre_birth_opportunity_mass",
        "eed.birth_attempts",
        "eed.successful_births",
        "eed.unique_opportunity_bearing_parents",
        "eed.committed_mutation_events",
        "eed.hereditary_change_outcomes",
        "eed.population_abundance",
        "eed.opportunity_mass_per_living_agent",
        "eed.opportunity_mass_per_opportunity_parent",
        "eed.resource_axis_mean",
        "eed.resource_axis_positive_count",
        "eed.resource_axis_negative_count",
        "eed.resource_axis_neutral_count",
    ),
    ("world", "inheritance"),
    "terminal resident-state and typed-lifecycle opportunity metrics; ratios are null when their denominator is zero",
)
POLICY_METRICS = MetricFamilyV1(
    "policy_v1",
    ("policy.task_accuracy", "policy.recurrent_cost", "policy.family_share"),
    ("policy",),
    "unavailable when the resolved profile does not emit the metric",
)
LINEAGE_METRICS = MetricFamilyV1(
    "lineage_v1",
    ("lineage.active", "lineage.top1_share", "lineage.effective_count"),
    ("lineage",),
    "unavailable when lineage reporting is disabled",
)
RECOMB72_METRICS = MetricFamilyV1(
    "recomb72_inheritance_v1",
    (
        "recomb72.mutation_events",
        "recomb72.ordinary_births",
        "recomb72.variance",
        "recomb72.variance_retention",
        "recomb72.extreme_retention",
        "recomb72.intact_package_retention",
    ),
    ("population", "inheritance", "ancestry", "rng"),
    "all integrity counters are measured; package retention is zero outside Pattern 3",
)
GMNW_OUTCOME_METRICS = MetricFamilyV1(
    "gmnw_outcome_v1",
    (
        "outcome.decision_count",
        "outcome.correct_count",
        "outcome.task_death_count",
    ),
    ("gmnw", "policy"),
    "zero counts are measured; unavailable when native GMNW outcome metrics are absent",
)


@dataclass(slots=True)
class ProfileRegistryV1:
    """Small explicit registry; registration is localized and duplicates fail."""

    _profiles: dict[str, RuntimeProfileV1]

    def __init__(self) -> None:
        self._profiles = {}

    def register(self, profile: RuntimeProfileV1) -> None:
        if profile.profile_id in self._profiles:
            raise ValueError(f"duplicate runtime profile: {profile.profile_id}")
        duplicate_uniform = len(set(profile.uniform_batch_parameters)) != len(
            profile.uniform_batch_parameters
        )
        unknown_uniform = sorted(
            set(profile.uniform_batch_parameters) - set(profile.parameter_contracts)
        )
        if duplicate_uniform or unknown_uniform:
            raise ValueError(
                f"invalid uniform batch parameters for {profile.profile_id}: "
                f"duplicates={duplicate_uniform} unknown={unknown_uniform}"
            )
        if not (
            1 <= profile.batch_min <= profile.batch_max
            and profile.batch_min
            <= profile.qualified_batch_min
            <= profile.qualified_batch_max
            <= profile.batch_max
        ):
            raise ValueError(
                f"invalid planning/qualified batch ranges for {profile.profile_id}"
            )
        self._profiles[profile.profile_id] = profile

    def profiles(self) -> tuple[RuntimeProfileV1, ...]:
        return tuple(self._profiles[name] for name in sorted(self._profiles))

    def resolve(
        self, requirements: CampaignRequirementsV1
    ) -> tuple[RuntimeProfileV1 | None, tuple[UnsupportedReasonV1, ...]]:
        candidates: list[RuntimeProfileV1] = []
        reasons: list[UnsupportedReasonV1] = []
        for registered_profile in self.profiles():
            profile = _specialize_declarative_runtime_identity(
                registered_profile, requirements
            )
            profile_reasons = _profile_reasons(profile, requirements)
            if not profile_reasons:
                candidates.append(profile)
            else:
                reasons.extend(profile_reasons)
        if not candidates:
            return None, tuple(reasons)
        candidates.sort(
            key=lambda row: (
                (
                    0
                    if row.qualification_invariants.get("runtime_identity_contract")
                    == _DECLARATIVE_RUNTIME_IDENTITY_V1
                    else 1
                ),
                len(row.world_overlays),
                row.profile_id,
            )
        )
        return candidates[0], ()


def _specialize_declarative_runtime_identity(
    profile: RuntimeProfileV1,
    requirements: CampaignRequirementsV1,
) -> RuntimeProfileV1:
    """Bind a reusable provider template to one compiled runtime identity."""

    if (
        profile.qualification_invariants.get("runtime_identity_contract")
        != _DECLARATIVE_RUNTIME_IDENTITY_V1
    ):
        return profile
    invariants = {
        **profile.qualification_invariants,
        "runtime.world_base": requirements.world_base,
        "runtime.world_overlays": json.dumps(
            sorted(requirements.world_overlays),
            separators=(",", ":"),
        ),
        "runtime.policy_family": requirements.policy_family,
        "runtime.selection_mode": requirements.selection_mode,
    }
    execution_config = _PRODUCTION_EXECUTION_CONFIGS.get(
        (
            requirements.world_base,
            requirements.world_overlays,
            requirements.policy_family,
        )
    )
    if execution_config is None:
        batch_max, qualified_batch_max, uniform_parameters = (
            profile.batch_max,
            profile.qualified_batch_max,
            profile.uniform_batch_parameters,
        )
    else:
        (
            batch_max,
            qualified_batch_max,
            uniform_parameters,
            metric_family_ids,
        ) = execution_config
        if profile.backend == "cpu_native":
            batch_max = 1
            qualified_batch_max = 1
        metric_families = tuple(
            {
                "population": POPULATION_METRICS,
                "policy": POLICY_METRICS,
                "lineage": LINEAGE_METRICS,
                "genome_v2": GENOME_V2_METRICS,
                "eed": EED_METRICS,
                "gmnw": GMNW_OUTCOME_METRICS,
                "recomb72": RECOMB72_METRICS,
            }[family_id]
            for family_id in metric_family_ids
        )
        invariants["mechanics_configuration"] = (
            f"{requirements.world_base}:"
            f"{','.join(sorted(requirements.world_overlays))}:"
            f"{requirements.policy_family}"
        )
        invariants["rng_schedule"] = "architecture-b-resident-v1"
    return replace(
        profile,
        world_base=requirements.world_base,
        world_overlays=requirements.world_overlays,
        policy_family=requirements.policy_family,
        selection_modes=frozenset({requirements.selection_mode}),
        qualification_invariants=invariants,
        batch_max=batch_max,
        qualified_batch_max=qualified_batch_max,
        uniform_batch_parameters=uniform_parameters,
        metric_families=(
            profile.metric_families if execution_config is None else metric_families
        ),
    )


def _profile_reasons(
    profile: RuntimeProfileV1, requirements: CampaignRequirementsV1
) -> list[UnsupportedReasonV1]:
    checks = (
        (
            profile.backend == requirements.backend,
            "missing_backend",
            "execution.backend",
        ),
        (
            profile.world_base == requirements.world_base,
            "missing_world_base",
            "world.base",
        ),
        (
            requirements.world_overlays <= profile.world_overlays,
            "missing_world_overlay",
            "world.overlays",
        ),
        (
            profile.policy_family == requirements.policy_family,
            "missing_policy",
            "policy.family",
        ),
        (
            requirements.selection_mode in profile.selection_modes,
            "missing_selection_mode",
            "selection.mode",
        ),
        (
            requirements.parameter_names <= set(profile.parameter_contracts),
            "missing_parameter_contract",
            "parameters",
        ),
        (
            requirements.required_metrics <= profile.metric_names,
            "missing_metric_provider",
            "analysis.rules",
        ),
        (
            not requirements.require_continuation or profile.continuation_supported,
            "missing_continuation_provider",
            "comparisons.continuation",
        ),
        (
            not requirements.require_cancellation or profile.cancellation_supported,
            "missing_cancellation_provider",
            "execution.cancellation",
        ),
    )
    return [
        UnsupportedReasonV1(code, path, code.replace("_", " "), profile.profile_id)
        for ok, code, path in checks
        if not ok
    ]


def _register_builtin_profiles(registry: ProfileRegistryV1) -> None:
    cuda_profile = RuntimeProfileV1(
        profile_id=EIEL_PRODUCTION_PROFILE_ID,
        profile_version=2,
        backend="cuda_resident",
        world_base=_DECLARATIVE_IDENTITY_PLACEHOLDER,
        world_overlays=frozenset(),
        policy_family=_DECLARATIVE_IDENTITY_PLACEHOLDER,
        selection_modes=frozenset({_DECLARATIVE_IDENTITY_PLACEHOLDER}),
        observation_contract="eiel-production-engine-observation-v1",
        state_components=(
            "population",
            "lifecycle",
            "world",
            "controller",
            "inheritance",
            "ancestry",
            "identity",
            "rng",
        ),
        checkpoint_components=(
            "authoritative_population",
            "authoritative_world",
            "authoritative_controller",
            "authoritative_genome",
            "authoritative_identity",
            "authoritative_rng",
        ),
        parameter_contracts={
            **COMMON_PATCH_PARAMETERS,
            **WDL_PARAMETERS,
            **EED_PARAMETERS,
            **PRODUCTION_PRIMITIVE_PARAMETERS,
            **POLICY_PARAMETERS,
            **GMNW_PARAMETERS,
            **GENOME_UPGRADE_V2_PARAMETERS,
            **PRODUCTION_LIFECYCLE_PARAMETERS,
            **CUDA_ARCHITECTURE_B_PARAMETERS,
            **RECOMB72_PARAMETERS,
        },
        uniform_batch_parameters=(),
        metric_families=(
            POPULATION_METRICS,
            POLICY_METRICS,
            LINEAGE_METRICS,
            GMNW_OUTCOME_METRICS,
            RECOMB72_METRICS,
            GENOME_V2_METRICS,
            EED_METRICS,
        ),
        execution_owner="eiel_production_engine_v1",
        arbitrary_positive_generations=True,
        continuation_supported=True,
        cancellation_supported=True,
        batch_min=1,
        batch_max=65535,
        qualified_batch_min=1,
        qualified_batch_max=65535,
        generation_counter_bits=64,
        qualification_invariants={
            "semantic_registry": "eiel_global_genome_registry_v1",
            "device_owner": "cuda_resident_no_fallback",
            "checkpoint_schema": "authoritative_state_schema_bound",
            "runtime_identity_contract": _DECLARATIVE_RUNTIME_IDENTITY_V1,
        },
        qualification_status="qualified",
        limitations=(
            "the production engine configuration and route qualification "
            "receipt bind world, controller, genome, live batch, checkpoint, "
            "and artifact identities",
            "CPU execution and automatic device fallback are forbidden",
            "execution remains fail-closed unless mechanic, reporting, "
            "authoritative-state-schema, and execution-route receipts all "
            "match the live artifacts exactly",
        ),
    )
    registry.register(cuda_profile)
    registry.register(
        replace(
            cuda_profile,
            profile_id=EIEL_PRODUCTION_CPU_PROFILE_ID,
            backend="cpu_native",
            batch_max=1,
            qualified_batch_max=1,
            qualification_status="qualified",
            qualification_invariants={
                **cuda_profile.qualification_invariants,
                "device_owner": "cpu_native_b1_no_cuda_fallback",
                "prebuilt_resident_owner": "verified_no_runtime_jit",
            },
            limitations=(
                "the production engine configuration and route qualification "
                "receipt bind world, controller, genome, live batch, checkpoint, "
                "runtime, and prebuilt CPU resident-owner artifact identities",
                "CPU-native execution is admitted only for single-world B1 "
                "production after exact route and continuation authority are earned",
                "runtime JIT is forbidden for qualified CPU-native production",
                "CUDA execution and automatic device fallback are forbidden for "
                "this CPU profile",
            ),
        )
    )


def default_profile_registry(
    extra_profiles: Iterable[RuntimeProfileV1] = (),
) -> ProfileRegistryV1:
    registry = ProfileRegistryV1()
    _register_builtin_profiles(registry)
    for profile in extra_profiles:
        registry.register(profile)
    return registry


__all__ = [
    "EIEL_PRODUCTION_PROFILE_ID",
    "EIEL_PRODUCTION_CPU_PROFILE_ID",
    "INHERITANCE_CROSS_PATH_PROFILE_ID",
    "ProfileRegistryV1",
    "default_profile_registry",
]
