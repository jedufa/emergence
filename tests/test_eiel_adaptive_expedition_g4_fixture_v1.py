from __future__ import annotations

from pathlib import Path

from emergence_contracts.canonical import canonical_sha256
from eiel_composition.adaptive_expedition.g4_fixture import (
    G4_EXPEDITION_ID_V1,
    G4_OBSERVATION_PROFILE_ID_V1,
    build_g4_exact_manifest_v1,
)
from eiel_composition.adaptive_expedition.policies.adaptive_illumination_v1 import (
    ADAPTIVE_ILLUMINATION_EXACT_WORKHORSE_V1_PROFILE,
)
from eiel_composition.adaptive_expedition.policy_protocol import (
    canonical_plain_value_v1,
)


FOUNDER = Path(
    "_spec/adaptive_science_expedition/v1/examples/"
    "EIEL_ADAPTIVE_EXPEDITION_G4_W64_FOUNDER_DECLARATION_V1.json"
)


def test_g4_draft_binds_exact_axis_profile_and_two_small_control_domains() -> None:
    manifest = build_g4_exact_manifest_v1(founder_request_path=FOUNDER)
    profile = manifest.policy_binding.profile.to_value()

    assert manifest.expedition_id == G4_EXPEDITION_ID_V1
    assert manifest.observation_profile_id == G4_OBSERVATION_PROFILE_ID_V1
    assert manifest.epoch_ticks == 2
    assert manifest.maximum_absolute_tick == 12
    assert manifest.maximum_frontiers == 3
    assert manifest.maximum_actions_per_frontier == 1
    assert len(manifest.founder_basis_refs) == 1
    assert profile["profile_id"] == (
        "EIEL_ADAPTIVE_ILLUMINATION_EXACT_WORKHORSE_AXIS_ROUND_ROBIN_PROFILE_V1"
    )
    assert profile["candidate_generators"]["global_exploration"]["enabled"] is False
    assert profile["historical_action_controls"]["branch_selection_rule"] == (
        "frontier_ordinal_round_robin_axis_next_authorized_value_v1"
    )
    assert manifest.observation_descriptor_set_sha256 == canonical_sha256(
        canonical_plain_value_v1(profile["descriptor_definitions"])
    )
    assert manifest.search_space.to_value() == {
        "controls": [
            {
                "control_id": "environment.resource.multiplier",
                "allowed_values": [1.0, 1.5],
            },
            {"control_id": "mutation.rate", "allowed_values": [0.125, 0.25]},
        ]
    }


def test_g4_fixture_can_freeze_one_frontier_replicate_profile() -> None:
    manifest = build_g4_exact_manifest_v1(
        founder_request_path=FOUNDER,
        policy_profile=ADAPTIVE_ILLUMINATION_EXACT_WORKHORSE_V1_PROFILE,
        maximum_frontiers=1,
        wall_time_budget_seconds=3600,
    )
    profile = manifest.policy_binding.profile.to_value()
    assert manifest.maximum_frontiers == 1
    assert manifest.wall_time_budget_seconds == 3600
    assert profile["profile_id"] == "EIEL_ADAPTIVE_ILLUMINATION_EXACT_WORKHORSE_PROFILE_V1"
    assert profile["candidate_generators"]["global_exploration"]["enabled"] is False
