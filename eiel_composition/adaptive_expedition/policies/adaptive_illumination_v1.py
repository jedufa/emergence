"""First deterministic, manifest-bound Adaptive Illumination policy profile.

All scientific interpretation and allocation choices in this module are data
inside ``ADAPTIVE_ILLUMINATION_V1_PROFILE``.  The algorithm is a small profile
interpreter: it does not make those choices coordinator truth and never emits
execution authority.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Final

from emergence_contracts.canonical import canonical_sha256

from ..policy_protocol import (
    ExpeditionPolicyError,
    canonical_plain_value_v1,
    freeze_canonical_mapping_v1,
    policy_binding_v1,
    seal_policy_proposal_v1,
    sha256_counter_prf_index_v1,
    validate_bounded_probe_contracts_v1,
    validate_policy_input_authority_v1,
    validate_policy_input_v1,
)
from .adaptive_illumination_identity_v1 import (
    ADAPTIVE_ILLUMINATION_IMPLEMENTATION_SHA256_V1,
)


ADAPTIVE_ILLUMINATION_POLICY_ID_V1: Final[str] = (
    "EIEL_ADAPTIVE_ILLUMINATION_POLICY_V1"
)
ADAPTIVE_ILLUMINATION_POLICY_VERSION_V1: Final[str] = "1"
ADAPTIVE_ILLUMINATION_PROFILE_SCHEMA_VERSION_V1: Final[str] = (
    "eiel_adaptive_illumination_profile_v1"
)

ADAPTIVE_ILLUMINATION_ALGORITHM_SHA256_V1 = (
    ADAPTIVE_ILLUMINATION_IMPLEMENTATION_SHA256_V1
)


ADAPTIVE_ILLUMINATION_V1_PROFILE: Final[Mapping[str, object]] = (
    freeze_canonical_mapping_v1(
        {
            "profile_id": "EIEL_ADAPTIVE_ILLUMINATION_PROFILE_V1",
            "profile_version": "1.3",
            "policy_input_authority": {
                "schema_version": "eiel_adaptive_policy_input_authority_v1",
                "allowed_paths": [
                    "/atlas/entries/*/descriptor_values/active_fraction_milli",
                    "/atlas/entries/*/descriptor_values/birth_count",
                    "/atlas/entries/*/descriptor_values/maximum_lineage_depth",
                    "/atlas/entries/*/entry_id",
                    "/atlas/entries/*/niche_id",
                    "/atlas/entries/*/representative_role",
                    "/atlas/entries/*/state_ref_id",
                    "/authorized_search_space/controls/*/allowed_values",
                    "/authorized_search_space/controls/*/control_id",
                    "/available_state_refs/*/control_history_sha256",
                    "/available_state_refs/*/exact_state_sha256",
                    "/available_state_refs/*/next_absolute_tick",
                    "/available_state_refs/*/semantic_world_sha256",
                    "/available_state_refs/*/source_object_sha256",
                    "/available_state_refs/*/state_class",
                    "/available_state_refs/*/state_ref_id",
                    "/available_state_refs/*/world_trajectory_id",
                    "/decision_frontier/frontier_id",
                    "/decision_frontier/frontier_ordinal",
                    "/decision_frontier/observation_frames/*/absolute_tick",
                    "/decision_frontier/observation_frames/*/control_readback/controls/environment.resource.multiplier",
                    "/decision_frontier/observation_frames/*/control_readback/controls/mutation.rate",
                    "/decision_frontier/observation_frames/*/facts/active_count",
                    "/decision_frontier/observation_frames/*/facts/birth_count",
                    "/decision_frontier/observation_frames/*/facts/capacity",
                    "/decision_frontier/observation_frames/*/facts/integrity_ok",
                    "/decision_frontier/observation_frames/*/facts/maximum_lineage_depth",
                    "/decision_frontier/observation_frames/*/frame_id",
                    "/decision_frontier/observation_frames/*/state_ref_id",
                    "/decision_frontier/observation_frames/*/trajectory_id",
                    "/expedition_id",
                    "/logical_capacity",
                    "/manifest_sha256",
                    "/outer_rng/cursor",
                    "/outer_rng/seed",
                    "/policy_binding",
                    "/protocol_version",
                    "/retained_parent_bank/entries/*/divergence_signature",
                    "/retained_parent_bank/entries/*/reason_codes",
                    "/retained_parent_bank/entries/*/state_ref_id",
                    "/retained_parent_bank/maximum_entries",
                ],
            },
            "descriptor_definitions": [
                {
                    "descriptor_id": "active_fraction_milli",
                    "source": {
                        "kind": "ratio_milli",
                        "numerator_path": ["facts", "active_count"],
                        "denominator_path": ["facts", "capacity"],
                    },
                    "calibration": {
                        "kind": "declared_integer_range",
                        "minimum": 0,
                        "maximum": 1000,
                    },
                    "niche": {"kind": "fixed_equal_bins", "bin_count": 5},
                    "distance": {
                        "kind": "normalized_absolute_milli",
                        "scale": 1000,
                    },
                },
                {
                    "descriptor_id": "birth_count",
                    "source": {
                        "kind": "integer_fact",
                        "path": ["facts", "birth_count"],
                    },
                    "calibration": {
                        "kind": "declared_integer_range",
                        "minimum": 0,
                        "maximum": 64,
                    },
                    "niche": {"kind": "fixed_equal_bins", "bin_count": 4},
                    "distance": {
                        "kind": "normalized_absolute_milli",
                        "scale": 1000,
                    },
                },
                {
                    "descriptor_id": "maximum_lineage_depth",
                    "source": {
                        "kind": "integer_fact",
                        "path": ["facts", "maximum_lineage_depth"],
                    },
                    "calibration": {
                        "kind": "declared_integer_range",
                        "minimum": 0,
                        "maximum": 16,
                    },
                    "niche": {"kind": "fixed_equal_bins", "bin_count": 4},
                    "distance": {
                        "kind": "normalized_absolute_milli",
                        "scale": 1000,
                    },
                },
            ],
            "missing_fact_policy": {
                "class_id": "INVALID_OBSERVATION",
                "reason_code": "PROFILE_REQUIRED_FACT_MISSING",
            },
            "classifiers": [
                {
                    "class_id": "INVALID_OBSERVATION",
                    "predicates": [
                        {
                            "source": "fact",
                            "path": ["facts", "integrity_ok"],
                            "operator": "EQ",
                            "value": False,
                        }
                    ],
                },
                {
                    "class_id": "EXTINCTION_REGION",
                    "predicates": [
                        {
                            "source": "fact",
                            "path": ["facts", "active_count"],
                            "operator": "LE",
                            "value": 0,
                        }
                    ],
                },
                {
                    "class_id": "INFORMATIVE_FRONTIER",
                    "predicates": [
                        {
                            "source": "descriptor",
                            "descriptor_id": "active_fraction_milli",
                            "operator": "BETWEEN_INCLUSIVE",
                            "minimum": 1,
                            "maximum": 300,
                        },
                        {
                            "source": "fact",
                            "path": ["facts", "birth_count"],
                            "operator": "GE",
                            "value": 1,
                        },
                    ],
                },
                {
                    "class_id": "PERSISTENT_ACTIVITY",
                    "predicates": [
                        {
                            "source": "descriptor",
                            "descriptor_id": "active_fraction_milli",
                            "operator": "GE",
                            "value": 301,
                        },
                        {
                            "source": "fact",
                            "path": ["facts", "maximum_lineage_depth"],
                            "operator": "GE",
                            "value": 2,
                        },
                    ],
                },
                {
                    "class_id": "TRIVIAL_REGION",
                    "predicates": [
                        {
                            "source": "fact",
                            "path": ["facts", "birth_count"],
                            "operator": "EQ",
                            "value": 0,
                        }
                    ],
                },
            ],
            "fallback_class_id": "UNKNOWN",
            "archive_policy": {
                "maximum_representatives_per_niche": 3,
                "representative_priority": [
                    "INFORMATIVE_FRONTIER",
                    "PERSISTENT_ACTIVITY",
                    "EXTINCTION_REGION",
                    "TRIVIAL_REGION",
                    "UNKNOWN",
                    "INVALID_OBSERVATION",
                ],
            },
            "parent_bank_policy": {
                "maximum_entries": 32,
                "retained_classes": [
                    "INFORMATIVE_FRONTIER",
                    "PERSISTENT_ACTIVITY",
                    "EXTINCTION_REGION",
                ],
                "reason_by_class": {
                    "INFORMATIVE_FRONTIER": "PATH_DEPENDENT_STEPPING_STONE",
                    "PERSISTENT_ACTIVITY": "FUTURE_PARENT",
                    "EXTINCTION_REGION": "NEGATIVE_SENTINEL",
                },
                "retention_reason_priority": [
                    "PATH_DEPENDENT_STEPPING_STONE",
                    "FUTURE_PARENT",
                    "NEGATIVE_SENTINEL",
                ],
                "same_niche_divergence": [
                    {"axis_id": "trajectory", "path": ["world_trajectory_id"], "operator": "NOT_EQUAL"},
                    {"axis_id": "exact_state", "path": ["exact_state_sha256"], "operator": "NOT_EQUAL"},
                    {"axis_id": "semantic_world", "path": ["semantic_world_sha256"], "operator": "NOT_EQUAL"},
                    {"axis_id": "source_object", "path": ["source_object_sha256"], "operator": "NOT_EQUAL"},
                    {"axis_id": "history", "path": ["control_history_sha256"], "operator": "NOT_EQUAL"},
                ],
                "budget_exhaustion_reason": "PROFILE_PARENT_BANK_MAXIMUM_ENTRIES",
            },
            "novelty_definition": {
                "definition_id": "descriptor_l1_minimum_milli_v1",
                "kind": "minimum_normalized_descriptor_l1_milli",
                "threshold_milli": 350,
            },
            "frontier_definition": {
                "definition_id": "profile_class_membership_v1",
                "class_ids": ["INFORMATIVE_FRONTIER"],
            },
            "evolvability_definition": {
                "definition_id": "lineage_depth_observation_v1",
                "source_path": ["facts", "maximum_lineage_depth"],
                "minimum_observed_value": 2,
            },
            "allocation_quotas": {
                "buckets": [
                    {"bucket": "exploration", "weight": 4},
                    {"bucket": "frontier", "weight": 3},
                    {"bucket": "replication", "weight": 2},
                    {"bucket": "observation", "weight": 1},
                ],
                "spill_order": [
                    "exploration",
                    "frontier",
                    "replication",
                    "observation",
                ],
            },
            "candidate_generators": {
                "global_exploration": {
                    "enabled": True,
                    "kind": "sha256_counter_cartesian_index_without_replacement_v1",
                    "bucket": "exploration",
                    "action_kind": "EXPLORE_GLOBAL",
                    "slot_cost": 1,
                    "reason_code": "UNDEREXPLORED_AUTHORIZED_SPACE",
                    "founder_start_next_tick": 0,
                    "founder_execution": {
                        "schema_version": "eiel_adaptive_founder_execution_profile_v1",
                        "world_count": 64,
                        "seed_derivation": (
                            "sha256_manifest_action_replicate_ordinal_unique_u63_v1"
                        ),
                    },
                },
                "class_action_rules": [
                    {
                        "class_id": "INVALID_OBSERVATION",
                        "bucket": "observation",
                        "action_kind": "QUARANTINE",
                        "slot_cost": 0,
                        "requires_state_ref": False,
                        "reason_code": "INVALID_OBSERVATION",
                    },
                    {
                        "class_id": "EXTINCTION_REGION",
                        "bucket": "observation",
                        "action_kind": "PARK",
                        "slot_cost": 0,
                        "requires_state_ref": False,
                        "reason_code": "EXTINCTION_NEGATIVE_RECORDED",
                    },
                    {
                        "class_id": "INFORMATIVE_FRONTIER",
                        "bucket": "frontier",
                        "action_kind": "BRANCH_LOCAL",
                        "slot_cost": 1,
                        "requires_state_ref": True,
                        "reason_code": "INFORMATIVE_FRONTIER",
                    },
                    {
                        "class_id": "PERSISTENT_ACTIVITY",
                        "bucket": "replication",
                        "action_kind": "REPLICATE",
                        "slot_cost": 1,
                        "requires_state_ref": True,
                        "reason_code": "PERSISTENT_ACTIVITY_REQUIRES_CONFIRMATION",
                    },
                    {
                        "class_id": "TRIVIAL_REGION",
                        "bucket": "observation",
                        "action_kind": "PARK",
                        "slot_cost": 0,
                        "requires_state_ref": False,
                        "reason_code": "TRIVIAL_REGION_RECORDED",
                    },
                    {
                        "class_id": "UNKNOWN",
                        "bucket": "observation",
                        "action_kind": "REQUEST_DETAILED_OBSERVATION",
                        "slot_cost": 1,
                        "requires_state_ref": False,
                        "reason_code": "PROFILE_CLASSIFICATION_UNRESOLVED",
                    },
                ],
            },
            "historical_action_controls": {
                "schema_version": (
                    "eiel_adaptive_illumination_historical_action_controls_v1"
                ),
                "control_ids": [
                    "environment.resource.multiplier",
                    "mutation.rate",
                ],
                "control_readback_path": ["control_readback", "controls"],
                "branch_selection_rule": (
                    "first_profile_axis_next_authorized_value_v1"
                ),
                "replicate_selection_rule": (
                    "preserve_authenticated_parent_controls_v1"
                ),
                "ineligible_policy": "OMIT_ACTION_WITH_REASON",
            },
            "replication_policy": {
                "schema_version": "eiel_adaptive_historical_replicate_policy_v1",
                "replicate_count_per_action": 1,
                "parent_state_class": "BRANCHABLE_EXACT",
                "control_rule": "PRESERVE_AUTHENTICATED_PARENT_CONTROLS",
                "rng_event_rule": "PRESERVE_EXACT_PARENT_CONTINUATION",
                "child_identity_rule": "FRESH_RUN_ATTEMPT_NEVER_RETRY",
                "pairing_rule": "same_parent_occurrence_exact_suffix_v1",
            },
            "tie_breaks": {
                "candidate": "canonical_action_sha256",
                "atlas": "representative_priority_then_entry_id",
                "parent_bank": "reason_priority_then_state_ref_id",
                "quota_remainder": "declared_bucket_order",
            },
            "human_review_projection": {
                "schema_version": "eiel_adaptive_policy_review_presentation_v1",
                "atlas_role_sections": [
                    {
                        "representative_role": "EXTINCTION_REGION",
                        "section_key": "overwhelming_extinction_or_dead_end_regions",
                    },
                    {
                        "representative_role": "INFORMATIVE_FRONTIER",
                        "section_key": "response_or_frontier_regions",
                    },
                    {
                        "representative_role": "INVALID_OBSERVATION",
                        "section_key": "limiting_controls_observations_capabilities_or_budgets",
                    },
                    {
                        "representative_role": "PERSISTENT_ACTIVITY",
                        "section_key": "novel_persistent_or_divergent_observations",
                    },
                    {
                        "representative_role": "TRIVIAL_REGION",
                        "section_key": "trivial_or_unexpressed_regions",
                    },
                    {
                        "representative_role": "UNKNOWN",
                        "section_key": "unresolved_scientific_questions",
                    },
                ],
                "action_reason_sections": [
                    {
                        "reason_code": "EXTINCTION_NEGATIVE_RECORDED",
                        "section_key": "overwhelming_extinction_or_dead_end_regions",
                    },
                    {
                        "reason_code": "INVALID_OBSERVATION",
                        "section_key": "limiting_controls_observations_capabilities_or_budgets",
                    },
                    {
                        "reason_code": "PROFILE_CLASSIFICATION_UNRESOLVED",
                        "section_key": "unresolved_scientific_questions",
                    },
                    {
                        "reason_code": "TRIVIAL_REGION_RECORDED",
                        "section_key": "trivial_or_unexpressed_regions",
                    },
                ],
                "empty_status_by_section": {
                    "authorized_search_space": "AUTHORIZED_SPACE_NOT_INDEXED",
                    "exploration_coverage": "NO_EXPLORATION_FRONTIER_RECORDED",
                    "trivial_or_unexpressed_regions": "NO_PROFILE_TRIVIAL_ROLE_RECORDED",
                    "overwhelming_extinction_or_dead_end_regions": "NO_PROFILE_DEAD_END_ROLE_RECORDED",
                    "response_or_frontier_regions": "NO_PROFILE_FRONTIER_ROLE_RECORDED",
                    "novel_persistent_or_divergent_observations": "NO_PROFILE_PERSISTENT_OR_DIVERGENT_ROLE_RECORDED",
                    "failed_replication_or_causal_followups": "NO_POLICY_ASSESSED_FOLLOWUP_FAILURE_RECORDED",
                    "retained_parents": "NO_EXACT_PARENT_RETAINED",
                    "discarded_parent_candidates": "NO_PARENT_DISCARD_REASON_RECORDED",
                    "unresolved_scientific_questions": "NO_UNRESOLVED_ROW_RECORDED",
                    "limiting_controls_observations_capabilities_or_budgets": "NO_LIMITATION_ROW_RECORDED",
                    "recommended_next_expedition": "NO_POLICY_PRESENTATION_RECOMMENDATION",
                    "human_scientific_choices": "NO_HUMAN_CHOICE_ROW_RECORDED",
                },
                "recommendation_rules": [
                    {
                        "when_section_key": "response_or_frontier_regions",
                        "recommendation_id": "REVIEW_FRONTIER_ROWS_FOR_NEXT_MANIFEST",
                        "recommendation": "Review the referenced frontier rows and exact retained parents, then author one bounded follow-up manifest for the human-selected question.",
                    },
                    {
                        "when_section_key": "novel_persistent_or_divergent_observations",
                        "recommendation_id": "REVIEW_PERSISTENT_ROWS_FOR_NEXT_MANIFEST",
                        "recommendation": "Review the referenced persistent or divergent rows and replication history before authoring a bounded follow-up manifest.",
                    },
                    {
                        "when_section_key": "overwhelming_extinction_or_dead_end_regions",
                        "recommendation_id": "REVIEW_DEAD_END_BOUNDARIES_FOR_NEXT_MANIFEST",
                        "recommendation": "Review the referenced dead-end rows and adjacent authorized controls before deciding whether a bounded boundary-mapping Expedition is warranted.",
                    },
                ],
                "fallback_recommendation": {
                    "recommendation_id": "HUMAN_REVIEW_BEFORE_NEXT_MANIFEST",
                    "recommendation": "Review the Atlas, exact Parent Bank, discarded candidates, history, and unresolved rows before authoring the next immutable Expedition manifest.",
                },
                "human_scientific_choices": [
                    {
                        "choice_id": "AUTHOR_HIGH_LEVEL_QUESTION",
                        "choice": "Choose the next high-level scientific question and authorized search space.",
                    },
                    {
                        "choice_id": "AUTHOR_PROBE_CONTRACTS",
                        "choice": "Decide whether any bounded causal probe contract is scientifically justified.",
                    },
                    {
                        "choice_id": "SELECT_NEXT_SCALE_AND_BUDGET",
                        "choice": "Choose the next Expedition duration, compute budget, stop conditions, and evidence expectations.",
                    },
                    {
                        "choice_id": "SELECT_PARENT_STATES",
                        "choice": "Choose which exact retained parents may seed the next Expedition.",
                    },
                ],
                "released_parent_reason_code": "POLICY_RELEASED_BOUNDED_PARENT_REFERENCE",
            },
            "probe_contracts": [],
        }
    )
)


def _exact_workhorse_profile_v1() -> Mapping[str, object]:
    profile = canonical_plain_value_v1(ADAPTIVE_ILLUMINATION_V1_PROFILE)
    if type(profile) is not dict:
        raise AssertionError("reference profile did not detach to a mapping")
    profile["profile_id"] = "EIEL_ADAPTIVE_ILLUMINATION_EXACT_WORKHORSE_PROFILE_V1"
    profile["profile_version"] = "1.1"
    profile["descriptor_definitions"] = [
        {
            "descriptor_id": "active_fraction_milli",
            "source": {
                "kind": "ratio_milli",
                "numerator_path": ["facts", "active_count"],
                "denominator_path": ["facts", "capacity"],
            },
            "calibration": {
                "kind": "declared_integer_range",
                "minimum": 0,
                "maximum": 1000,
            },
            "niche": {"kind": "fixed_equal_bins", "bin_count": 5},
            "distance": {"kind": "normalized_absolute_milli", "scale": 1000},
        },
        {
            "descriptor_id": "births_per_tick_milli",
            "source": {
                "kind": "ratio_milli",
                "numerator_path": ["facts", "birth_count"],
                "denominator_path": ["facts", "terminal_absolute_tick"],
            },
            "calibration": {
                "kind": "declared_integer_range",
                "minimum": 0,
                "maximum": 4096,
            },
            "niche": {"kind": "fixed_equal_bins", "bin_count": 5},
            "distance": {"kind": "normalized_absolute_milli", "scale": 1000},
        },
        {
            "descriptor_id": "deaths_per_tick_milli",
            "source": {
                "kind": "ratio_milli",
                "numerator_path": ["facts", "death_count"],
                "denominator_path": ["facts", "terminal_absolute_tick"],
            },
            "calibration": {
                "kind": "declared_integer_range",
                "minimum": 0,
                "maximum": 4096,
            },
            "niche": {"kind": "fixed_equal_bins", "bin_count": 5},
            "distance": {"kind": "normalized_absolute_milli", "scale": 1000},
        },
    ]
    persistent = next(
        row
        for row in profile["classifiers"]
        if row["class_id"] == "PERSISTENT_ACTIVITY"
    )
    persistent["predicates"][1] = {
        "source": "fact",
        "path": ["facts", "birth_count"],
        "operator": "GE",
        "value": 1,
    }
    profile["evolvability_definition"] = {
        "definition_id": "observed_reproductive_activity_v1",
        "source_path": ["facts", "birth_count"],
        "minimum_observed_value": 1,
    }
    profile["candidate_generators"]["global_exploration"]["enabled"] = False
    authority_paths = profile["policy_input_authority"]["allowed_paths"]
    authority_paths[:] = [
        path
        for path in authority_paths
        if "maximum_lineage_depth" not in path
        and not path.startswith("/atlas/entries/*/descriptor_values/")
    ]
    authority_paths.extend(
        [
            "/atlas/entries/*/descriptor_values/active_fraction_milli",
            "/atlas/entries/*/descriptor_values/births_per_tick_milli",
            "/atlas/entries/*/descriptor_values/deaths_per_tick_milli",
            "/decision_frontier/observation_frames/*/facts/death_count",
            "/decision_frontier/observation_frames/*/facts/terminal_absolute_tick",
        ]
    )
    authority_paths.sort()
    return freeze_canonical_mapping_v1(profile)


ADAPTIVE_ILLUMINATION_EXACT_WORKHORSE_V1_PROFILE: Final[
    Mapping[str, object]
] = _exact_workhorse_profile_v1()


def _founder_workhorse_profile_v1() -> Mapping[str, object]:
    """Versioned founder-only profile for real closed-loop use.

    The aggregate Science Request route does not expose the exact selected
    world's lineage-depth fact.  Reuse the exact Workhorse descriptor set that
    is actually authenticated by the adapter, but keep global founder
    exploration enabled.  This is a replaceable policy-profile choice; the
    coordinator only sees the pure policy port.
    """

    profile = canonical_plain_value_v1(ADAPTIVE_ILLUMINATION_EXACT_WORKHORSE_V1_PROFILE)
    if type(profile) is not dict:
        raise AssertionError("exact Workhorse profile did not detach to a mapping")
    profile["profile_id"] = "EIEL_ADAPTIVE_ILLUMINATION_FOUNDER_WORKHORSE_PROFILE_V1"
    profile["profile_version"] = "1"
    profile["candidate_generators"]["global_exploration"]["enabled"] = True
    profile["candidate_generators"]["global_exploration"]["founder_execution"][
        "world_count"
    ] = 64
    return freeze_canonical_mapping_v1(profile)


ADAPTIVE_ILLUMINATION_FOUNDER_WORKHORSE_V1_PROFILE: Final[
    Mapping[str, object]
] = _founder_workhorse_profile_v1()


def founder_workhorse_profile_for_world_count_v1(
    world_count: int,
) -> Mapping[str, object]:
    """Return an explicit founder-only profile with a bounded world count.

    The count is policy-profile data, not coordinator truth.  Giving each
    count a distinct profile identity makes 1,024/10,000-world engineering
    campaigns replayable and prevents a manifest from silently changing the
    packed child shape after launch.
    """

    if type(world_count) is not int or not 1 <= world_count <= 1_000_000:
        raise ValueError("founder world_count must be within the bounded V1 range")
    profile = canonical_plain_value_v1(
        ADAPTIVE_ILLUMINATION_FOUNDER_WORKHORSE_V1_PROFILE
    )
    if type(profile) is not dict:
        raise AssertionError("founder Workhorse profile did not detach to a mapping")
    profile["profile_id"] = (
        "EIEL_ADAPTIVE_ILLUMINATION_FOUNDER_WORKHORSE_"
        f"W{world_count}_PROFILE_V1"
    )
    profile["profile_version"] = "1"
    profile["candidate_generators"]["global_exploration"]["founder_execution"][
        "world_count"
    ] = world_count
    return freeze_canonical_mapping_v1(profile)


def _exact_workhorse_axis_round_robin_profile_v1() -> Mapping[str, object]:
    """Versioned exact-only profile that exercises every authorized axis.

    This is profile data interpreted by the same pure policy.  The generic
    coordinator remains unaware of controls or axis-selection semantics.
    """

    profile = canonical_plain_value_v1(
        ADAPTIVE_ILLUMINATION_EXACT_WORKHORSE_V1_PROFILE
    )
    if type(profile) is not dict:
        raise AssertionError("exact Workhorse profile did not detach to a mapping")
    profile["profile_id"] = (
        "EIEL_ADAPTIVE_ILLUMINATION_EXACT_WORKHORSE_AXIS_ROUND_ROBIN_PROFILE_V1"
    )
    profile["profile_version"] = "1"
    profile["historical_action_controls"]["branch_selection_rule"] = (
        "frontier_ordinal_round_robin_axis_next_authorized_value_v1"
    )
    bounded_cycle = [
        {
            "action_kind": "BRANCH_LOCAL",
            "bucket": "frontier",
            "slot_cost": 1,
            "requires_state_ref": True,
            "reason_code": "BOUNDED_EXACT_AXIS_FOLLOWUP",
        },
        {
            "action_kind": "BRANCH_LOCAL",
            "bucket": "frontier",
            "slot_cost": 1,
            "requires_state_ref": True,
            "reason_code": "BOUNDED_EXACT_AXIS_FOLLOWUP",
        },
        {
            "action_kind": "REPLICATE",
            "bucket": "replication",
            "slot_cost": 1,
            "requires_state_ref": True,
            "reason_code": "BOUNDED_EXACT_REPLICATION_FOLLOWUP",
        },
    ]
    for rule in profile["candidate_generators"]["class_action_rules"]:
        if rule["class_id"] in {"INFORMATIVE_FRONTIER", "PERSISTENT_ACTIVITY"}:
            rule["frontier_action_cycle"] = canonical_plain_value_v1(bounded_cycle)
    return freeze_canonical_mapping_v1(profile)


ADAPTIVE_ILLUMINATION_EXACT_WORKHORSE_AXIS_ROUND_ROBIN_V1_PROFILE: Final[
    Mapping[str, object]
] = _exact_workhorse_axis_round_robin_profile_v1()


def _mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ExpeditionPolicyError(f"{field} must be a mapping")
    return value


def _sequence(value: object, *, field: str) -> Sequence[object]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ExpeditionPolicyError(f"{field} must be a sequence")
    return value


_MISSING = object()


def _at_path(value: object, path: Sequence[object]) -> object:
    current = value
    for raw_part in path:
        if type(raw_part) is not str or not isinstance(current, Mapping):
            return _MISSING
        if raw_part not in current:
            return _MISSING
        current = current[raw_part]
    return current


def _integer(value: object, *, field: str) -> int:
    if type(value) is not int:
        raise ExpeditionPolicyError(f"{field} must be an integer")
    return value


def _validate_profile_v1(profile: Mapping[str, object]) -> None:
    required = {
        "profile_id",
        "profile_version",
        "policy_input_authority",
        "descriptor_definitions",
        "missing_fact_policy",
        "classifiers",
        "fallback_class_id",
        "archive_policy",
        "parent_bank_policy",
        "novelty_definition",
        "frontier_definition",
        "evolvability_definition",
        "allocation_quotas",
        "candidate_generators",
        "historical_action_controls",
        "replication_policy",
        "tie_breaks",
        "human_review_projection",
        "probe_contracts",
    }
    if set(profile) != required:
        raise ExpeditionPolicyError("Adaptive Illumination profile fields must be exact")
    descriptors = _sequence(
        profile["descriptor_definitions"], field="profile.descriptor_definitions"
    )
    if not descriptors:
        raise ExpeditionPolicyError("profile requires at least one descriptor")
    descriptor_ids: list[str] = []
    for index, raw in enumerate(descriptors):
        row = _mapping(raw, field=f"profile.descriptor_definitions[{index}]")
        descriptor_id = row.get("descriptor_id")
        if type(descriptor_id) is not str or not descriptor_id:
            raise ExpeditionPolicyError("descriptor_id must be non-empty text")
        descriptor_ids.append(descriptor_id)
        calibration = _mapping(row.get("calibration"), field="descriptor.calibration")
        minimum = _integer(calibration.get("minimum"), field="calibration.minimum")
        maximum = _integer(calibration.get("maximum"), field="calibration.maximum")
        if maximum <= minimum:
            raise ExpeditionPolicyError("descriptor calibration range must be positive")
        niche = _mapping(row.get("niche"), field="descriptor.niche")
        if niche.get("kind") != "fixed_equal_bins" or _integer(
            niche.get("bin_count"), field="niche.bin_count"
        ) <= 0:
            raise ExpeditionPolicyError("descriptor niche definition is unsupported")
        distance = _mapping(row.get("distance"), field="descriptor.distance")
        if distance.get("kind") != "normalized_absolute_milli" or _integer(
            distance.get("scale"), field="distance.scale"
        ) <= 0:
            raise ExpeditionPolicyError("descriptor distance definition is unsupported")
    if descriptor_ids != list(dict.fromkeys(descriptor_ids)):
        raise ExpeditionPolicyError("descriptor_ids must be unique")
    classifiers = _sequence(profile["classifiers"], field="profile.classifiers")
    class_ids = [
        _mapping(row, field="classifier").get("class_id") for row in classifiers
    ]
    if any(type(item) is not str or not item for item in class_ids) or len(
        class_ids
    ) != len(set(class_ids)):
        raise ExpeditionPolicyError("classifier class_ids must be unique text")
    quotas = _mapping(profile["allocation_quotas"], field="profile.allocation_quotas")
    buckets = _sequence(quotas.get("buckets"), field="allocation_quotas.buckets")
    bucket_names = []
    for raw in buckets:
        row = _mapping(raw, field="allocation quota")
        bucket = row.get("bucket")
        weight = row.get("weight")
        if type(bucket) is not str or not bucket or type(weight) is not int or weight <= 0:
            raise ExpeditionPolicyError("allocation quota bucket/weight is invalid")
        bucket_names.append(bucket)
    if bucket_names != list(dict.fromkeys(bucket_names)):
        raise ExpeditionPolicyError("allocation quota buckets must be unique")
    spill = list(_sequence(quotas.get("spill_order"), field="allocation spill_order"))
    if set(spill) != set(bucket_names) or len(spill) != len(bucket_names):
        raise ExpeditionPolicyError("allocation spill_order must cover each bucket once")
    generators = _mapping(
        profile["candidate_generators"], field="profile.candidate_generators"
    )
    global_generator = _mapping(
        generators.get("global_exploration"), field="global_exploration"
    )
    if type(global_generator.get("enabled")) is not bool:
        raise ExpeditionPolicyError("global exploration enabled must be boolean")
    founder_execution = _mapping(
        global_generator.get("founder_execution"), field="founder_execution"
    )
    if set(founder_execution) != {
        "schema_version",
        "world_count",
        "seed_derivation",
    }:
        raise ExpeditionPolicyError("founder execution profile fields must be exact")
    if founder_execution.get("schema_version") != (
        "eiel_adaptive_founder_execution_profile_v1"
    ):
        raise ExpeditionPolicyError("founder execution profile version is unsupported")
    world_count = founder_execution.get("world_count")
    if type(world_count) is not int or not 1 <= world_count <= 1_000_000:
        raise ExpeditionPolicyError(
            "founder execution world_count must be within the bounded V1 range"
        )
    if founder_execution.get("seed_derivation") != (
        "sha256_manifest_action_replicate_ordinal_unique_u63_v1"
    ):
        raise ExpeditionPolicyError("founder seed derivation is unsupported")
    historical_controls = _mapping(
        profile["historical_action_controls"],
        field="profile.historical_action_controls",
    )
    if set(historical_controls) != {
        "schema_version",
        "control_ids",
        "control_readback_path",
        "branch_selection_rule",
        "replicate_selection_rule",
        "ineligible_policy",
    } or historical_controls.get("schema_version") != (
        "eiel_adaptive_illumination_historical_action_controls_v1"
    ):
        raise ExpeditionPolicyError(
            "historical action control profile fields/version must be exact"
        )
    historical_control_ids = list(
        _sequence(
            historical_controls.get("control_ids"),
            field="historical action control_ids",
        )
    )
    if (
        not historical_control_ids
        or any(type(item) is not str or not item for item in historical_control_ids)
        or historical_control_ids != list(dict.fromkeys(historical_control_ids))
    ):
        raise ExpeditionPolicyError(
            "historical action control_ids must be unique non-empty text"
        )
    readback_path = _sequence(
        historical_controls.get("control_readback_path"),
        field="historical action control_readback_path",
    )
    if not readback_path or any(type(item) is not str or not item for item in readback_path):
        raise ExpeditionPolicyError(
            "historical action control readback path must be non-empty text"
        )
    if historical_controls.get("branch_selection_rule") not in {
        "first_profile_axis_next_authorized_value_v1",
        "frontier_ordinal_round_robin_axis_next_authorized_value_v1",
    } or historical_controls.get("replicate_selection_rule") != (
        "preserve_authenticated_parent_controls_v1"
    ) or historical_controls.get("ineligible_policy") != (
        "OMIT_ACTION_WITH_REASON"
    ):
        raise ExpeditionPolicyError(
            "historical action control rules are unsupported"
        )
    replication = _mapping(
        profile["replication_policy"], field="profile.replication_policy"
    )
    expected_replication = {
        "schema_version": "eiel_adaptive_historical_replicate_policy_v1",
        "replicate_count_per_action": 1,
        "parent_state_class": "BRANCHABLE_EXACT",
        "control_rule": "PRESERVE_AUTHENTICATED_PARENT_CONTROLS",
        "rng_event_rule": "PRESERVE_EXACT_PARENT_CONTINUATION",
        "child_identity_rule": "FRESH_RUN_ATTEMPT_NEVER_RETRY",
        "pairing_rule": "same_parent_occurrence_exact_suffix_v1",
    }
    if canonical_plain_value_v1(replication) != expected_replication:
        raise ExpeditionPolicyError(
            "historical replication policy semantics are unsupported"
        )
    validate_bounded_probe_contracts_v1(profile)
    validate_policy_input_authority_v1(profile)

    def reject_hidden_authority(value: object, *, in_probe: bool = False) -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                child_probe = in_probe or key == "probe_contracts"
                if key in {"reward", "fitness", "scalar_reward"}:
                    raise ExpeditionPolicyError("profile cannot define scalar reward authority")
                if key in {"phenotype", "phenotype_id", "phenotype_label", "phenotype_name"} and not child_probe:
                    raise ExpeditionPolicyError(
                        "named phenotype is legal only inside a bounded probe contract"
                    )
                reject_hidden_authority(item, in_probe=child_probe)
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            for item in value:
                reject_hidden_authority(item, in_probe=in_probe)

    reject_hidden_authority(profile)


def _descriptor_value_v1(frame: Mapping[str, object], definition: Mapping[str, object]) -> int:
    source = _mapping(definition.get("source"), field="descriptor.source")
    kind = source.get("kind")
    if kind == "integer_fact":
        value = _at_path(frame, _sequence(source.get("path"), field="source.path"))
        return _integer(value, field=f"descriptor {definition.get('descriptor_id')}")
    if kind == "ratio_milli":
        numerator = _at_path(
            frame,
            _sequence(source.get("numerator_path"), field="source.numerator_path"),
        )
        denominator = _at_path(
            frame,
            _sequence(source.get("denominator_path"), field="source.denominator_path"),
        )
        numerator_int = _integer(numerator, field="ratio numerator")
        denominator_int = _integer(denominator, field="ratio denominator")
        if denominator_int <= 0:
            raise ExpeditionPolicyError("ratio descriptor denominator must be positive")
        return numerator_int * 1000 // denominator_int
    raise ExpeditionPolicyError(f"unsupported descriptor source kind {kind!r}")


def _descriptor_projection_v1(
    frame: Mapping[str, object], profile: Mapping[str, object]
) -> tuple[dict[str, int], str]:
    values: dict[str, int] = {}
    bins: list[str] = []
    for raw in _sequence(
        profile["descriptor_definitions"], field="profile.descriptor_definitions"
    ):
        definition = _mapping(raw, field="descriptor")
        descriptor_id = str(definition["descriptor_id"])
        value = _descriptor_value_v1(frame, definition)
        calibration = _mapping(definition["calibration"], field="calibration")
        minimum = int(calibration["minimum"])
        maximum = int(calibration["maximum"])
        clamped = min(maximum, max(minimum, value))
        values[descriptor_id] = clamped
        bin_count = int(_mapping(definition["niche"], field="niche")["bin_count"])
        bin_index = min(
            bin_count - 1,
            (clamped - minimum) * bin_count // (maximum - minimum + 1),
        )
        bins.append(f"{descriptor_id}:{bin_index}")
    return values, "|".join(bins)


def _predicate_v1(
    predicate: Mapping[str, object],
    *,
    frame: Mapping[str, object],
    descriptors: Mapping[str, int],
) -> bool:
    if predicate.get("source") == "descriptor":
        descriptor_id = predicate.get("descriptor_id")
        value = descriptors.get(str(descriptor_id), _MISSING)
    elif predicate.get("source") == "fact":
        value = _at_path(
            frame, _sequence(predicate.get("path"), field="predicate.path")
        )
    else:
        raise ExpeditionPolicyError("predicate source must be fact or descriptor")
    if value is _MISSING:
        return False
    operator = predicate.get("operator")
    if operator == "EQ":
        return value == predicate.get("value")
    if operator in {"LE", "GE"}:
        if type(value) is not int or type(predicate.get("value")) is not int:
            return False
        return value <= predicate["value"] if operator == "LE" else value >= predicate["value"]
    if operator == "BETWEEN_INCLUSIVE":
        return (
            type(value) is int
            and type(predicate.get("minimum")) is int
            and type(predicate.get("maximum")) is int
            and predicate["minimum"] <= value <= predicate["maximum"]
        )
    raise ExpeditionPolicyError(f"unsupported predicate operator {operator!r}")


def _classify_v1(
    frame: Mapping[str, object],
    descriptors: Mapping[str, int],
    profile: Mapping[str, object],
) -> str:
    for raw in _sequence(profile["classifiers"], field="profile.classifiers"):
        classifier = _mapping(raw, field="classifier")
        predicates = _sequence(classifier.get("predicates"), field="classifier.predicates")
        if predicates and all(
            _predicate_v1(_mapping(item, field="predicate"), frame=frame, descriptors=descriptors)
            for item in predicates
        ):
            return str(classifier["class_id"])
    return str(profile["fallback_class_id"])


def _descriptor_distance_v1(
    left: Mapping[str, int], right: Mapping[str, object], profile: Mapping[str, object]
) -> int | None:
    distances: list[int] = []
    for raw in _sequence(profile["descriptor_definitions"], field="descriptors"):
        definition = _mapping(raw, field="descriptor")
        descriptor_id = str(definition["descriptor_id"])
        other = right.get(descriptor_id)
        if type(other) is not int:
            return None
        calibration = _mapping(definition["calibration"], field="calibration")
        span = int(calibration["maximum"]) - int(calibration["minimum"])
        scale = int(_mapping(definition["distance"], field="distance")["scale"])
        distances.append(abs(left[descriptor_id] - other) * scale // span)
    return sum(distances) // len(distances) if distances else None


def _novelty_v1(
    descriptors: Mapping[str, int], atlas: Mapping[str, object], profile: Mapping[str, object]
) -> dict[str, object]:
    distances = []
    for raw in _sequence(atlas.get("entries", ()), field="atlas.entries"):
        entry = _mapping(raw, field="atlas entry")
        other = entry.get("descriptor_values")
        if isinstance(other, Mapping):
            distance = _descriptor_distance_v1(descriptors, other, profile)
            if distance is not None:
                distances.append(distance)
    definition = _mapping(profile["novelty_definition"], field="novelty_definition")
    minimum = min(distances) if distances else None
    return {
        "definition_id": definition["definition_id"],
        "minimum_distance_milli": minimum,
        "is_novel": minimum is None or minimum >= int(definition["threshold_milli"]),
    }


def _action_v1(core: Mapping[str, object]) -> dict[str, object]:
    plain = canonical_plain_value_v1(core)
    if type(plain) is not dict:
        raise ExpeditionPolicyError("action core must be a mapping")
    return {
        "action_id": f"A-{canonical_sha256(plain).hex[:24]}",
        **plain,
    }


def _state_refs_by_id_v1(
    checked_input: Mapping[str, object],
) -> dict[str, Mapping[str, object]]:
    result: dict[str, Mapping[str, object]] = {}
    for raw in _sequence(checked_input["available_state_refs"], field="state refs"):
        row = _mapping(raw, field="state ref")
        state_ref_id = row.get("state_ref_id")
        next_tick = row.get("next_absolute_tick")
        if type(state_ref_id) is not str or not state_ref_id or state_ref_id in result:
            raise ExpeditionPolicyError("state_ref_id must be unique text")
        if type(next_tick) is not int or next_tick < 0:
            raise ExpeditionPolicyError("state ref next_absolute_tick must be non-negative")
        result[state_ref_id] = row
    return result


def _frame_rows_v1(checked_input: Mapping[str, object]) -> list[Mapping[str, object]]:
    frontier = _mapping(checked_input["decision_frontier"], field="decision frontier")
    rows = []
    identities = set()
    for raw in _sequence(frontier["observation_frames"], field="observation frames"):
        frame = _mapping(raw, field="observation frame")
        frame_id = frame.get("frame_id")
        trajectory_id = frame.get("trajectory_id")
        absolute_tick = frame.get("absolute_tick")
        if (
            type(frame_id) is not str
            or not frame_id
            or type(trajectory_id) is not str
            or not trajectory_id
            or type(absolute_tick) is not int
            or absolute_tick < 0
            or frame_id in identities
        ):
            raise ExpeditionPolicyError("observation frame identity/tick is invalid")
        _mapping(frame.get("facts"), field="observation frame facts")
        identities.add(frame_id)
        rows.append(frame)
    return sorted(rows, key=lambda row: (str(row["trajectory_id"]), str(row["frame_id"])))


def _authorized_control_domains_v1(
    checked_input: Mapping[str, object],
) -> list[tuple[str, Sequence[object]]]:
    search_space = _mapping(
        checked_input.get("authorized_search_space", {"controls": ()}),
        field="authorized search space",
    )
    controls = [
        _mapping(row, field="authorized control")
        for row in _sequence(
            search_space.get("controls", ()), field="authorized controls"
        )
    ]
    ordered: list[tuple[str, Sequence[object]]] = []
    seen: set[str] = set()
    for control in sorted(controls, key=lambda row: str(row.get("control_id"))):
        control_id = control.get("control_id")
        values = _sequence(
            control.get("allowed_values"), field="allowed control values"
        )
        if (
            type(control_id) is not str
            or not control_id
            or control_id in seen
            or not values
            or len({canonical_sha256(value).hex for value in values}) != len(values)
        ):
            raise ExpeditionPolicyError("authorized control domain is invalid")
        seen.add(control_id)
        ordered.append((control_id, values))
    return ordered


def _historical_action_control_bindings_v1(
    *,
    frame: Mapping[str, object],
    state_ref: Mapping[str, object] | None,
    action_kind: str,
    control_domains: Mapping[str, Sequence[object]],
    profile: Mapping[str, object],
    frontier_ordinal: int,
) -> tuple[list[dict[str, object]] | None, str | None]:
    """Resolve one manifest-legal exact-parent action from profile data only."""

    if state_ref is None or state_ref.get("state_class") != "BRANCHABLE_EXACT":
        return None, "HISTORICAL_PARENT_NOT_BRANCHABLE_EXACT"
    policy = _mapping(
        profile["historical_action_controls"], field="historical action controls"
    )
    control_ids = [
        str(item)
        for item in _sequence(policy["control_ids"], field="historical control_ids")
    ]
    readback = _at_path(
        frame,
        _sequence(policy["control_readback_path"], field="control readback path"),
    )
    if not isinstance(readback, Mapping):
        return None, "AUTHENTICATED_PARENT_CONTROL_READBACK_MISSING"
    current: dict[str, object] = {}
    indexes: dict[str, int] = {}
    for control_id in control_ids:
        if control_id not in readback or control_id not in control_domains:
            return None, "AUTHENTICATED_PARENT_CONTROL_READBACK_INCOMPLETE"
        value = readback[control_id]
        domain = control_domains[control_id]
        matches = [
            index
            for index, candidate in enumerate(domain)
            if canonical_sha256(candidate) == canonical_sha256(value)
        ]
        if len(matches) != 1:
            return None, "PARENT_CONTROL_OUTSIDE_AUTHORIZED_DOMAIN"
        current[control_id] = canonical_plain_value_v1(value)
        indexes[control_id] = matches[0]
    if action_kind == "REPLICATE":
        return [
            {"control_id": control_id, "value": current[control_id]}
            for control_id in control_ids
        ], None
    if action_kind != "BRANCH_LOCAL":
        raise ExpeditionPolicyError("historical control binding requested for another action")
    eligible_control_ids = [
        control_id
        for control_id in control_ids
        if len(control_domains[control_id]) > 1
    ]
    if not eligible_control_ids:
        return None, "NO_AUTHORIZED_LOCAL_CONTROL_ALTERNATIVE"
    branch_rule = policy["branch_selection_rule"]
    if branch_rule == "first_profile_axis_next_authorized_value_v1":
        changed_control_id = eligible_control_ids[0]
    elif branch_rule == (
        "frontier_ordinal_round_robin_axis_next_authorized_value_v1"
    ):
        changed_control_id = eligible_control_ids[
            frontier_ordinal % len(eligible_control_ids)
        ]
    else:
        raise ExpeditionPolicyError("historical branch selection rule is unsupported")
    bindings = []
    for control_id in control_ids:
        value = current[control_id]
        if control_id == changed_control_id:
            domain = control_domains[control_id]
            value = canonical_plain_value_v1(
                domain[(indexes[control_id] + 1) % len(domain)]
            )
        bindings.append({"control_id": control_id, "value": value})
    return bindings, None


def _frontier_scheduled_class_rule_v1(
    rule: Mapping[str, object], *, frontier_ordinal: int
) -> Mapping[str, object]:
    """Apply an optional, profile-bound finite action cycle."""

    raw_cycle = rule.get("frontier_action_cycle")
    if raw_cycle is None:
        return rule
    cycle = [
        _mapping(row, field="frontier action cycle row")
        for row in _sequence(raw_cycle, field="frontier action cycle")
    ]
    expected = {
        "action_kind",
        "bucket",
        "slot_cost",
        "requires_state_ref",
        "reason_code",
    }
    if not cycle or any(set(row) != expected for row in cycle):
        raise ExpeditionPolicyError("frontier action cycle fields must be exact")
    selected = cycle[frontier_ordinal % len(cycle)]
    return {
        **{key: value for key, value in rule.items() if key != "frontier_action_cycle"},
        **selected,
    }


def _atlas_delta_v1(
    *,
    new_entries: Sequence[Mapping[str, object]],
    atlas: Mapping[str, object],
    profile: Mapping[str, object],
) -> tuple[dict[str, object], int]:
    archive = _mapping(profile["archive_policy"], field="archive policy")
    maximum = int(archive["maximum_representatives_per_niche"])
    priority = {
        name: index
        for index, name in enumerate(
            _sequence(archive["representative_priority"], field="representative priority")
        )
    }
    existing = [
        _mapping(row, field="atlas entry")
        for row in _sequence(atlas.get("entries", ()), field="atlas entries")
    ]
    new_ids = {str(row["entry_id"]) for row in new_entries}
    by_niche: dict[str, list[Mapping[str, object]]] = {}
    for row in (*existing, *new_entries):
        niche_id = row.get("niche_id")
        entry_id = row.get("entry_id")
        if type(niche_id) is not str or type(entry_id) is not str:
            raise ExpeditionPolicyError("atlas entry identity/niche is invalid")
        by_niche.setdefault(niche_id, []).append(row)
    keep_ids: set[str] = set()
    for rows in by_niche.values():
        ordered = sorted(
            rows,
            key=lambda row: (
                priority.get(str(row.get("representative_role")), len(priority)),
                str(row["entry_id"]),
            ),
        )
        keep_ids.update(str(row["entry_id"]) for row in ordered[:maximum])
    upserts = [row for row in new_entries if str(row["entry_id"]) in keep_ids]
    retire = sorted(
        str(row["entry_id"])
        for row in existing
        if str(row["entry_id"]) not in keep_ids
    )
    rejected = len(new_ids - keep_ids)
    return {"upsert_entries": upserts, "retire_entry_ids": retire}, rejected


def _divergence_signature_v1(
    *,
    niche_id: str,
    reason_code: str,
    state_ref: Mapping[str, object],
    profile: Mapping[str, object],
) -> tuple[object, ...]:
    bank = _mapping(profile["parent_bank_policy"], field="parent bank policy")
    values: list[object] = [niche_id, reason_code]
    for raw in _sequence(bank["same_niche_divergence"], field="divergence axes"):
        axis = _mapping(raw, field="divergence axis")
        values.append(
            _at_path(state_ref, _sequence(axis["path"], field="divergence path"))
        )
    return tuple(None if value is _MISSING else canonical_plain_value_v1(value) for value in values)


def _parent_bank_delta_v1(
    *,
    candidates: Sequence[Mapping[str, object]],
    bank_view: Mapping[str, object],
    profile: Mapping[str, object],
    atlas: Mapping[str, object],
    state_refs: Mapping[str, Mapping[str, object]],
) -> tuple[dict[str, object], dict[str, int]]:
    policy = _mapping(profile["parent_bank_policy"], field="parent bank policy")
    # The immutable manifest/bank revision is the hard capacity.  The profile
    # may choose a stricter scientific budget, but it cannot silently exceed
    # the manifest's bounded Parent Bank allowance.
    profile_maximum = int(policy["maximum_entries"])
    declared_maximum = bank_view.get("maximum_entries", profile_maximum)
    if type(declared_maximum) is not int or declared_maximum < 0:
        raise ExpeditionPolicyError("parent bank maximum_entries is invalid")
    maximum = min(profile_maximum, declared_maximum)
    existing = [
        _mapping(row, field="parent bank entry")
        for row in _sequence(bank_view.get("entries", ()), field="parent bank entries")
    ]
    existing_ids = {str(row.get("state_ref_id")) for row in existing}
    def signature_key(value: object) -> tuple[str, ...]:
        plain = canonical_plain_value_v1(value)
        if type(plain) is not list:
            raise ExpeditionPolicyError("parent divergence signature must be a sequence")
        return tuple(canonical_sha256(item).hex for item in plain)

    # A retained bank entry predates the current frontier.  Its durable
    # contract intentionally stores only generic StateRef/reason data, while
    # the policy owns the meaning of marginal divergence.  Reconstruct the
    # profile-defined signature from the authenticated StateRef and the Atlas
    # link when available.  If an Atlas representative has already been
    # retired, do not guess: treating the prior entry as divergent is the
    # conservative choice that preserves a possible stepping stone.
    atlas_niche_by_state_id: dict[str, str] = {}
    for raw in _sequence(atlas.get("entries", ()), field="atlas entries"):
        row = _mapping(raw, field="atlas entry")
        state_ref_id = row.get("state_ref_id")
        niche_id = row.get("niche_id")
        if (
            type(state_ref_id) is str
            and state_ref_id
            and type(niche_id) is str
            and niche_id
        ):
            atlas_niche_by_state_id[state_ref_id] = niche_id

    def existing_signature(row: Mapping[str, object]) -> tuple[str, ...] | None:
        state_ref_id = row.get("state_ref_id")
        reason_codes = row.get("reason_codes")
        if (
            type(state_ref_id) is str
            and state_ref_id
            and isinstance(reason_codes, Sequence)
            and not isinstance(reason_codes, (str, bytes))
            and reason_codes
            and state_ref_id in state_refs
            and state_ref_id in atlas_niche_by_state_id
        ):
            try:
                reconstructed = _divergence_signature_v1(
                    niche_id=atlas_niche_by_state_id[state_ref_id],
                    reason_code=str(reason_codes[0]),
                    state_ref=state_refs[state_ref_id],
                    profile=profile,
                )
                return signature_key(reconstructed)
            except (KeyError, ExpeditionPolicyError):
                # A malformed/legacy identity must not cause an unsafe
                # deduplication decision.  Fall through to the legacy
                # signature, if one is present.
                pass
        raw_signature = row.get("divergence_signature")
        if raw_signature is None:
            return None
        try:
            return signature_key(raw_signature)
        except ExpeditionPolicyError:
            return None

    signatures = {
        signature
        for row in existing
        if (signature := existing_signature(row)) is not None
    }
    reason_priority = {
        reason: index
        for index, reason in enumerate(
            _sequence(policy["retention_reason_priority"], field="reason priority")
        )
    }
    available = max(0, maximum - len(existing))
    retained: list[Mapping[str, object]] = []
    not_retained: list[dict[str, object]] = []
    ordered = sorted(
        candidates,
        key=lambda row: (
            reason_priority.get(str(row["survival_reason"]), len(reason_priority)),
            str(row["state_ref_id"]),
        ),
    )
    for row in ordered:
        state_ref_id = str(row["state_ref_id"])
        signature = signature_key(row["divergence_signature"])
        if state_ref_id in existing_ids:
            continue
        if signature in signatures:
            not_retained.append(
                {
                    "state_ref_id": state_ref_id,
                    "reason_code": "NO_PROFILE_DEFINED_MARGINAL_DIVERGENCE",
                }
            )
            continue
        if len(retained) >= available:
            not_retained.append(
                {
                    "state_ref_id": state_ref_id,
                    "reason_code": policy["budget_exhaustion_reason"],
                }
            )
            continue
        retained.append(row)
        existing_ids.add(state_ref_id)
        signatures.add(signature)
    counts: dict[str, int] = {}
    for row in not_retained:
        reason = str(row["reason_code"])
        counts[reason] = counts.get(reason, 0) + 1
    return {
        "retain_entries": retained,
        "release_state_ref_ids": [],
        "not_retained": not_retained,
    }, counts


def _quota_select_v1(
    candidates: Sequence[Mapping[str, object]],
    *,
    capacity: int,
    profile: Mapping[str, object],
) -> tuple[list[Mapping[str, object]], dict[str, int]]:
    zero_cost = [row for row in candidates if int(row["slot_cost"]) == 0]
    positive = [row for row in candidates if int(row["slot_cost"]) > 0]
    if any(int(row["slot_cost"]) != 1 for row in positive):
        raise ExpeditionPolicyError("V1 quota allocator supports unit slot costs only")
    quotas = _mapping(profile["allocation_quotas"], field="allocation quotas")
    bucket_rows = [
        _mapping(row, field="quota bucket")
        for row in _sequence(quotas["buckets"], field="quota buckets")
    ]
    total_weight = sum(int(row["weight"]) for row in bucket_rows)
    allocation = {
        str(row["bucket"]): capacity * int(row["weight"]) // total_weight
        for row in bucket_rows
    }
    remainder = capacity - sum(allocation.values())
    for row in bucket_rows[:remainder]:
        allocation[str(row["bucket"])] += 1
    by_bucket: dict[str, list[Mapping[str, object]]] = {
        str(row["bucket"]): [] for row in bucket_rows
    }
    for row in positive:
        bucket = str(row["bucket"])
        if bucket not in by_bucket:
            raise ExpeditionPolicyError(f"candidate bucket {bucket!r} is absent from profile")
        by_bucket[bucket].append(row)
    for rows in by_bucket.values():
        rows.sort(key=lambda row: str(row["action_id"]))
    selected = list(sorted(zero_cost, key=lambda row: str(row["action_id"])))
    selected_ids = {str(row["action_id"]) for row in selected}
    used = 0
    for bucket, amount in allocation.items():
        chosen = by_bucket[bucket][:amount]
        selected.extend(chosen)
        selected_ids.update(str(row["action_id"]) for row in chosen)
        used += len(chosen)
    spill_order = [str(item) for item in _sequence(quotas["spill_order"], field="spill order")]
    while used < capacity:
        added = False
        for bucket in spill_order:
            candidate = next(
                (
                    row
                    for row in by_bucket[bucket]
                    if str(row["action_id"]) not in selected_ids
                ),
                None,
            )
            if candidate is not None:
                selected.append(candidate)
                selected_ids.add(str(candidate["action_id"]))
                used += 1
                added = True
                if used == capacity:
                    break
        if not added:
            break
    rejected_counts: dict[str, int] = {}
    for row in positive:
        if str(row["action_id"]) not in selected_ids:
            bucket = str(row["bucket"])
            rejected_counts[bucket] = rejected_counts.get(bucket, 0) + 1
    return selected, rejected_counts


class AdaptiveIlluminationPolicyV1:
    """Pure interpreter for one immutable Adaptive Illumination profile."""

    def __init__(
        self, profile: Mapping[str, object] = ADAPTIVE_ILLUMINATION_V1_PROFILE
    ) -> None:
        self._profile = freeze_canonical_mapping_v1(profile)
        _validate_profile_v1(self._profile)
        self._binding = policy_binding_v1(
            policy_id=self.policy_id,
            policy_version=self.policy_version,
            algorithm_sha256=ADAPTIVE_ILLUMINATION_ALGORITHM_SHA256_V1,
            profile=self._profile,
            profile_schema_version=ADAPTIVE_ILLUMINATION_PROFILE_SCHEMA_VERSION_V1,
        )

    @property
    def policy_id(self) -> str:
        return ADAPTIVE_ILLUMINATION_POLICY_ID_V1

    @property
    def policy_version(self) -> str:
        return ADAPTIVE_ILLUMINATION_POLICY_VERSION_V1

    @property
    def profile(self) -> Mapping[str, object]:
        return self._profile

    @property
    def binding(self) -> Mapping[str, object]:
        return self._binding

    def decide(self, decision_input: Mapping[str, object]) -> Mapping[str, object]:
        checked = validate_policy_input_v1(
            decision_input, expected_binding=self.binding, profile=self.profile
        )
        state_refs = _state_refs_by_id_v1(checked)
        frames = _frame_rows_v1(checked)
        atlas = _mapping(checked["atlas"], field="atlas")
        bank_view = _mapping(checked["retained_parent_bank"], field="parent bank")
        generator = _mapping(
            self.profile["candidate_generators"], field="candidate generators"
        )
        class_rules = {
            str(row["class_id"]): row
            for row in (
                _mapping(raw, field="class action rule")
                for raw in _sequence(
                    generator["class_action_rules"], field="class action rules"
                )
            )
        }
        bank_policy = _mapping(
            self.profile["parent_bank_policy"], field="parent bank policy"
        )
        retained_classes = set(
            _sequence(bank_policy["retained_classes"], field="retained classes")
        )
        reason_by_class = _mapping(
            bank_policy["reason_by_class"], field="reason by class"
        )
        atlas_entries: list[dict[str, object]] = []
        bank_candidates: list[dict[str, object]] = []
        bank_not_retained: list[dict[str, object]] = []
        action_candidates: list[dict[str, object]] = []
        classification_counts: dict[str, int] = {}
        decision_frontier = _mapping(
            checked["decision_frontier"], field="decision frontier"
        )
        frontier_ordinal = int(decision_frontier["frontier_ordinal"])
        ordered_controls = _authorized_control_domains_v1(checked)
        control_domains = dict(ordered_controls)
        historical_action_ineligible_counts: dict[str, int] = {}

        for frame in frames:
            try:
                descriptors, niche_id = _descriptor_projection_v1(frame, self.profile)
                class_id = _classify_v1(frame, descriptors, self.profile)
                missing_reason = None
            except ExpeditionPolicyError:
                descriptors = {}
                niche_id = "INVALID:PROFILE_REQUIRED_FACT_MISSING"
                missing = _mapping(
                    self.profile["missing_fact_policy"], field="missing fact policy"
                )
                class_id = str(missing["class_id"])
                missing_reason = str(missing["reason_code"])
            classification_counts[class_id] = classification_counts.get(class_id, 0) + 1
            novelty = (
                _novelty_v1(descriptors, atlas, self.profile)
                if descriptors
                else {
                    "definition_id": _mapping(
                        self.profile["novelty_definition"], field="novelty definition"
                    )["definition_id"],
                    "minimum_distance_milli": None,
                    "is_novel": False,
                }
            )
            atlas_core = {
                "frame_id": frame["frame_id"],
                "trajectory_id": frame["trajectory_id"],
                "frontier_ordinal": frontier_ordinal,
                "observation_absolute_tick": frame["absolute_tick"],
                "descriptor_values": descriptors,
                "niche_id": niche_id,
                "representative_role": class_id,
                "novelty": novelty,
                "reason_codes": [missing_reason or class_id],
            }
            atlas_entries.append(
                {
                    "entry_id": f"EA-{canonical_sha256(atlas_core).hex[:24]}",
                    **atlas_core,
                }
            )
            state_ref_id = frame.get("state_ref_id")
            state_ref = state_refs.get(str(state_ref_id)) if state_ref_id is not None else None
            raw_rule = class_rules.get(class_id)
            rule = (
                None
                if raw_rule is None
                else _frontier_scheduled_class_rule_v1(
                    raw_rule,
                    frontier_ordinal=frontier_ordinal,
                )
            )
            action_control_bindings: list[dict[str, object]] | None = None
            ineligible_reason: str | None = None
            if (
                rule is not None
                and bool(rule.get("requires_state_ref"))
                and state_ref is None
            ):
                ineligible_reason = "REQUIRED_PARENT_STATE_REF_MISSING"
            if rule is not None and rule["action_kind"] in {
                "BRANCH_LOCAL",
                "REPLICATE",
            }:
                action_control_bindings, ineligible_reason = (
                    _historical_action_control_bindings_v1(
                        frame=frame,
                        state_ref=state_ref,
                        action_kind=str(rule["action_kind"]),
                        control_domains=control_domains,
                        profile=self.profile,
                        frontier_ordinal=frontier_ordinal,
                    )
                )
            if ineligible_reason is not None:
                historical_action_ineligible_counts[ineligible_reason] = (
                    historical_action_ineligible_counts.get(ineligible_reason, 0) + 1
                )
            if rule is not None and ineligible_reason is None:
                action_core: dict[str, object] = {
                    "action_kind": rule["action_kind"],
                    "bucket": rule["bucket"],
                    "slot_cost": rule["slot_cost"],
                    "trajectory_id": frame["trajectory_id"],
                    "source_frame_id": frame["frame_id"],
                    "frontier_ordinal": frontier_ordinal,
                    "observation_absolute_tick": frame["absolute_tick"],
                    "reason_codes": [rule["reason_code"]],
                }
                if state_ref is not None:
                    action_core["parent_state_ref_id"] = state_ref_id
                    action_core["parent_next_absolute_tick"] = state_ref[
                        "next_absolute_tick"
                    ]
                if action_control_bindings is not None:
                    action_core["control_bindings"] = action_control_bindings
                if rule["action_kind"] == "REPLICATE":
                    action_core["replication"] = canonical_plain_value_v1(
                        self.profile["replication_policy"]
                    )
                action_candidates.append(_action_v1(action_core))
            if class_id in retained_classes and state_ref is not None:
                survival_reason = str(reason_by_class[class_id])
                if state_ref.get("state_class") not in {
                    "BRANCHABLE_EXACT",
                    "REPLAY_DERIVABLE",
                }:
                    bank_not_retained.append(
                        {
                            "state_ref_id": state_ref_id,
                            "reason_code": "STATE_NOT_EXACTLY_CONTINUABLE",
                        }
                    )
                else:
                    signature = _divergence_signature_v1(
                        niche_id=niche_id,
                        reason_code=survival_reason,
                        state_ref=state_ref,
                        profile=self.profile,
                    )
                    bank_candidates.append(
                        {
                            "state_ref_id": state_ref_id,
                            "trajectory_id": frame["trajectory_id"],
                            "frontier_ordinal": frontier_ordinal,
                            "observation_absolute_tick": frame["absolute_tick"],
                            "parent_next_absolute_tick": state_ref["next_absolute_tick"],
                            "niche_id": niche_id,
                            "survival_reason": survival_reason,
                            "divergence_signature": signature,
                            "source_frame_id": frame["frame_id"],
                        }
                    )

        probes = validate_bounded_probe_contracts_v1(self.profile)
        for probe_id, contract in probes.items():
            template = _mapping(contract["action_template"], field="probe action template")
            matched = 0
            for frame in frames:
                if matched >= int(contract["maximum_actions"]):
                    break
                try:
                    descriptors, _ = _descriptor_projection_v1(frame, self.profile)
                except ExpeditionPolicyError:
                    continue
                predicates = _sequence(
                    contract["observable_predicates"], field="probe predicates"
                )
                if all(
                    _predicate_v1(
                        _mapping(raw, field="probe predicate"),
                        frame=frame,
                        descriptors=descriptors,
                    )
                    for raw in predicates
                ):
                    action_candidates.append(
                        _action_v1(
                            {
                                "action_kind": "CAUSAL_PROBE",
                                "bucket": template["bucket"],
                                "slot_cost": template["slot_cost"],
                                "trajectory_id": frame["trajectory_id"],
                                "source_frame_id": frame["frame_id"],
                                "frontier_ordinal": frontier_ordinal,
                                "observation_absolute_tick": frame["absolute_tick"],
                                "probe_contract_id": probe_id,
                                "reason_codes": ["EXPLICIT_BOUNDED_PROBE_MATCH"],
                            }
                        )
                    )
                    matched += 1

        outer_rng = _mapping(checked["outer_rng"], field="outer_rng")
        seed = int(outer_rng["seed"])
        cursor = int(outer_rng["cursor"])
        capacity = int(checked["logical_capacity"])
        global_generator = _mapping(
            generator["global_exploration"], field="global exploration generator"
        )
        cartesian_size = 1
        for _control_id, values in ordered_controls:
            cartesian_size *= len(values)
        used_cartesian_indices: set[int] = set()
        global_candidate_count = (
            min(capacity, cartesian_size)
            if global_generator["enabled"] is True
            else 0
        )
        for candidate_ordinal in range(global_candidate_count):
            cartesian_index, cursor = sha256_counter_prf_index_v1(
                seed=seed,
                counter=cursor,
                domain=f"{self.policy_id}:global-cartesian:{candidate_ordinal}",
                size=cartesian_size,
            )
            while cartesian_index in used_cartesian_indices:
                cartesian_index = (cartesian_index + 1) % cartesian_size
            used_cartesian_indices.add(cartesian_index)
            remainder = cartesian_index
            bindings = []
            for control_id, values in reversed(ordered_controls):
                value_index = remainder % len(values)
                remainder //= len(values)
                bindings.append(
                    {
                        "control_id": control_id,
                        "value": canonical_plain_value_v1(values[value_index]),
                    }
                )
            bindings.reverse()
            if ordered_controls:
                action_candidates.append(
                    _action_v1(
                        {
                            "action_kind": global_generator["action_kind"],
                            "bucket": global_generator["bucket"],
                            "slot_cost": global_generator["slot_cost"],
                            "child_start_next_tick": global_generator[
                                "founder_start_next_tick"
                            ],
                            "candidate_ordinal": candidate_ordinal,
                            "frontier_ordinal": frontier_ordinal,
                            "control_bindings": bindings,
                            "founder_execution": canonical_plain_value_v1(
                                global_generator["founder_execution"]
                            ),
                            "reason_codes": [global_generator["reason_code"]],
                        }
                    )
                )

        atlas_delta, atlas_rejected = _atlas_delta_v1(
            new_entries=atlas_entries, atlas=atlas, profile=self.profile
        )
        bank_delta, bank_rejected = _parent_bank_delta_v1(
            candidates=bank_candidates,
            bank_view=bank_view,
            profile=self.profile,
            atlas=atlas,
            state_refs=state_refs,
        )
        unique_ineligible = [
            {"state_ref_id": state_ref_id, "reason_code": reason_code}
            for state_ref_id, reason_code in sorted(
                {
                    (str(row["state_ref_id"]), str(row["reason_code"]))
                    for row in bank_not_retained
                }
            )
        ]
        bank_delta["not_retained"] = [
            *unique_ineligible,
            *bank_delta["not_retained"],
        ]
        if unique_ineligible:
            bank_rejected["STATE_NOT_EXACTLY_CONTINUABLE"] = len(
                unique_ineligible
            )
        selected_actions, action_rejected = _quota_select_v1(
            action_candidates, capacity=capacity, profile=self.profile
        )
        rejected = {
            "atlas_profile_limit": atlas_rejected,
            "parent_bank_by_reason": bank_rejected,
            "action_candidates_by_bucket": action_rejected,
            "classification_counts": dict(sorted(classification_counts.items())),
            "historical_action_ineligible_counts": dict(
                sorted(historical_action_ineligible_counts.items())
            ),
        }
        return seal_policy_proposal_v1(
            policy_binding=self.binding,
            profile=self.profile,
            decision_input=checked,
            rng_cursor_after=cursor,
            atlas_delta=atlas_delta,
            retained_parent_bank_delta=bank_delta,
            semantic_actions=selected_actions,
            rejected_candidate_summary=rejected,
        )


__all__ = (
    "ADAPTIVE_ILLUMINATION_ALGORITHM_SHA256_V1",
    "ADAPTIVE_ILLUMINATION_POLICY_ID_V1",
    "ADAPTIVE_ILLUMINATION_POLICY_VERSION_V1",
    "ADAPTIVE_ILLUMINATION_PROFILE_SCHEMA_VERSION_V1",
    "ADAPTIVE_ILLUMINATION_EXACT_WORKHORSE_AXIS_ROUND_ROBIN_V1_PROFILE",
    "ADAPTIVE_ILLUMINATION_EXACT_WORKHORSE_V1_PROFILE",
    "ADAPTIVE_ILLUMINATION_FOUNDER_WORKHORSE_V1_PROFILE",
    "founder_workhorse_profile_for_world_count_v1",
    "ADAPTIVE_ILLUMINATION_V1_PROFILE",
    "AdaptiveIlluminationPolicyV1",
)
