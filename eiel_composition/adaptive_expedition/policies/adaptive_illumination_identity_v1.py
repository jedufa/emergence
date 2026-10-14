"""Generated, pure implementation identity for Adaptive Illumination V1.

The source-bundle digest is verified by tests against the policy interpreter
and policy protocol source bytes.  The semantic manifest also binds a compact
golden replay vector whose expected digest is checked against live execution.
No runtime reads source files; this module is immutable build metadata only.
"""

from __future__ import annotations

from typing import Final

from emergence_contracts.canonical import canonical_sha256
from emergence_contracts.ids import ContentDigest


ADAPTIVE_ILLUMINATION_SOURCE_BUNDLE_SHA256_HEX_V1: Final[str] = (
    "04f85d1a4fc9e689c67b21bd94884a59ad2df42202a2eeaff029e9f6944c4df4"
)

ADAPTIVE_ILLUMINATION_GOLDEN_REPLAY_VECTOR_V1: Final[dict[str, object]] = {
    "vector_id": "adaptive-illumination-reference-semantic-vector-v1",
    "input_semantics": {
        "frontier_ordinal": 7,
        "outer_seed": 676302,
        "outer_cursor": 3,
        "logical_capacity": 5,
        "authorized_controls": {
            "environment.resource.multiplier": [1.0, 1.5],
            "mutation.rate": [0.125, 0.25],
        },
        "frame_classes": [
            "INFORMATIVE_FRONTIER",
            "PERSISTENT_ACTIVITY",
            "EXTINCTION_REGION",
            "INVALID_OBSERVATION",
        ],
    },
    "expected_semantic_output_sha256": {
        "algorithm": "sha256",
        "hex": "d48fda8e1d486df73a0d41d80fed3858685f99d059f5a26f3ef6ec106f4684ad",
    },
}

ADAPTIVE_ILLUMINATION_SEMANTIC_IMPLEMENTATION_MANIFEST_V1: Final[
    dict[str, object]
] = {
    "schema_version": "eiel_adaptive_policy_implementation_manifest_v1",
    "algorithm_id": "eiel_adaptive_illumination_profile_interpreter_v1",
    "policy_id": "EIEL_ADAPTIVE_ILLUMINATION_POLICY_V1",
    "policy_version": "1",
    "protocol_version": "eiel_adaptive_expedition_policy_protocol_v1",
    "source_bundle": {
        "schema_version": "eiel_policy_source_bundle_sha256_v1",
        "sha256": ADAPTIVE_ILLUMINATION_SOURCE_BUNDLE_SHA256_HEX_V1,
        "paths": [
            "eiel_composition/adaptive_expedition/policies/adaptive_illumination_v1.py",
            "eiel_composition/adaptive_expedition/policy_protocol.py",
        ],
    },
    "semantic_operations": [
        "positive_profile_declared_input_projection",
        "profile_defined_descriptor_projection",
        "profile_defined_rule_classification",
        "profile_defined_niche_projection",
        "profile_defined_vector_distance",
        "profile_defined_parent_divergence",
        "profile_defined_quota_allocation",
        "profile_defined_global_generator_enablement",
        "profile_defined_founder_world_count",
        "profile_defined_exact_historical_action_controls",
        "profile_defined_historical_axis_selection",
        "profile_defined_frontier_action_schedule",
        "profile_defined_exact_historical_replicate",
        "exact_continuation_parent_retention_gate",
        "sha256_counter_cartesian_generation_without_replacement",
        "canonical_digest_tie_break",
    ],
    "golden_replay_vectors": [ADAPTIVE_ILLUMINATION_GOLDEN_REPLAY_VECTOR_V1],
    "authority_effect": "NONE_PROPOSAL_ONLY",
}

ADAPTIVE_ILLUMINATION_IMPLEMENTATION_SHA256_V1: Final[ContentDigest] = (
    canonical_sha256(ADAPTIVE_ILLUMINATION_SEMANTIC_IMPLEMENTATION_MANIFEST_V1)
)


__all__ = (
    "ADAPTIVE_ILLUMINATION_GOLDEN_REPLAY_VECTOR_V1",
    "ADAPTIVE_ILLUMINATION_IMPLEMENTATION_SHA256_V1",
    "ADAPTIVE_ILLUMINATION_SEMANTIC_IMPLEMENTATION_MANIFEST_V1",
    "ADAPTIVE_ILLUMINATION_SOURCE_BUNDLE_SHA256_HEX_V1",
)
