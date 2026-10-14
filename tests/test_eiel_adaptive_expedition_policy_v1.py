from __future__ import annotations

import ast
import hashlib
from pathlib import Path

import pytest

from emergence_contracts.canonical import canonical_sha256
from emergence_contracts.ids import ContentDigest
from emergence_contracts.payloads import CanonicalPayloadV1
from eiel_composition.adaptive_expedition.policy_protocol import (
    ExpeditionPolicyError,
    ExpeditionPolicyV1,
    POLICY_PROTOCOL_VERSION_V1,
    canonical_plain_value_v1,
    project_policy_input_v1,
    seal_policy_proposal_v1,
    sha256_counter_prf_index_v1,
    sha256_counter_prf_u64_v1,
    validate_policy_input_v1,
    validate_policy_proposal_v1,
)
from eiel_composition.adaptive_expedition.policies.adaptive_illumination_v1 import (
    ADAPTIVE_ILLUMINATION_EXACT_WORKHORSE_AXIS_ROUND_ROBIN_V1_PROFILE,
    ADAPTIVE_ILLUMINATION_EXACT_WORKHORSE_V1_PROFILE,
    ADAPTIVE_ILLUMINATION_V1_PROFILE,
    AdaptiveIlluminationPolicyV1,
    founder_workhorse_profile_for_world_count_v1,
)
from eiel_composition.adaptive_expedition.policies.adaptive_illumination_identity_v1 import (
    ADAPTIVE_ILLUMINATION_GOLDEN_REPLAY_VECTOR_V1,
    ADAPTIVE_ILLUMINATION_SEMANTIC_IMPLEMENTATION_MANIFEST_V1,
    ADAPTIVE_ILLUMINATION_SOURCE_BUNDLE_SHA256_HEX_V1,
)
from eiel_composition.adaptive_expedition.policies.toy_round_robin_v1 import (
    ToyRoundRobinPolicyV1,
)


def _state_ref(
    state_ref_id: str,
    *,
    next_absolute_tick: int,
    lineage_id: str,
    genotype: str,
) -> dict[str, object]:
    return {
        "state_ref_id": state_ref_id,
        "next_absolute_tick": next_absolute_tick,
        "state_class": "BRANCHABLE_EXACT",
        "world_trajectory_id": lineage_id,
        "exact_state_sha256": genotype,
        "semantic_world_sha256": f"semantic-{lineage_id}",
        "source_object_sha256": f"source-{state_ref_id}",
        "control_history_sha256": f"history-{state_ref_id}",
    }


def _decision_input(policy: ExpeditionPolicyV1) -> dict[str, object]:
    frames = [
        {
            "frame_id": "frame-frontier",
            "trajectory_id": "trajectory-frontier",
            "absolute_tick": 10,
            "state_ref_id": "state-frontier",
            "control_readback": {
                "controls": {
                    "environment.resource.multiplier": 1.0,
                    "mutation.rate": 0.125,
                }
            },
            "facts": {
                "integrity_ok": True,
                "active_count": 2,
                "capacity": 10,
                "birth_count": 1,
                "maximum_lineage_depth": 1,
            },
        },
        {
            "frame_id": "frame-persistent",
            "trajectory_id": "trajectory-persistent",
            "absolute_tick": 18,
            "state_ref_id": "state-persistent",
            "control_readback": {
                "controls": {
                    "environment.resource.multiplier": 1.0,
                    "mutation.rate": 0.125,
                }
            },
            "facts": {
                "integrity_ok": True,
                "active_count": 8,
                "capacity": 10,
                "birth_count": 3,
                "maximum_lineage_depth": 3,
            },
        },
        {
            "frame_id": "frame-extinct",
            "trajectory_id": "trajectory-extinct",
            "absolute_tick": 6,
            "state_ref_id": "state-extinct",
            "control_readback": {
                "controls": {
                    "environment.resource.multiplier": 1.0,
                    "mutation.rate": 0.125,
                }
            },
            "facts": {
                "integrity_ok": True,
                "active_count": 0,
                "capacity": 10,
                "birth_count": 0,
                "maximum_lineage_depth": 0,
            },
        },
        {
            "frame_id": "frame-invalid",
            "trajectory_id": "trajectory-invalid",
            "absolute_tick": 4,
            "control_readback": {
                "controls": {
                    "environment.resource.multiplier": 1.0,
                    "mutation.rate": 0.125,
                }
            },
            "facts": {
                "integrity_ok": False,
                "active_count": 5,
                "capacity": 10,
                "birth_count": 0,
                "maximum_lineage_depth": 0,
            },
        },
    ]
    for frame in frames:
        frame["facts"]["death_count"] = max(
            0, int(frame["facts"]["birth_count"]) - 1
        )
        frame["facts"]["terminal_absolute_tick"] = frame["absolute_tick"]
    state_refs = [
        _state_ref(
            "state-frontier",
            next_absolute_tick=11,
            lineage_id="lineage-a",
            genotype="genotype-a",
        ),
        _state_ref(
            "state-persistent",
            next_absolute_tick=19,
            lineage_id="lineage-b",
            genotype="genotype-b",
        ),
        _state_ref(
            "state-extinct",
            next_absolute_tick=7,
            lineage_id="lineage-c",
            genotype="genotype-c",
        ),
    ]
    return {
        "protocol_version": POLICY_PROTOCOL_VERSION_V1,
        "expedition_id": "expedition-test",
        "manifest_sha256": canonical_sha256(
            {
                "expedition_id": "expedition-test",
                "policy_binding": canonical_plain_value_v1(policy.binding),
            }
        ).to_dict(),
        "policy_binding": canonical_plain_value_v1(policy.binding),
        "decision_frontier": {
            "frontier_id": "frontier-0007",
            "frontier_ordinal": 7,
            "observation_frames": frames,
            "bounded_history": [],
        },
        "atlas": {"entries": []},
        "retained_parent_bank": {"entries": []},
        "available_state_refs": state_refs,
        "logical_capacity": 5,
        "outer_rng": {"seed": 0xA51CE, "cursor": 3},
        "authorized_search_space": {
            "controls": [
                {
                    "control_id": "environment.resource.multiplier",
                    "allowed_values": [1.0, 1.5],
                },
                {"control_id": "mutation.rate", "allowed_values": [0.125, 0.25]},
            ]
        },
    }


def _proposal_core_kwargs(
    policy: ExpeditionPolicyV1, decision_input: dict[str, object]
) -> dict[str, object]:
    return {
        "policy_binding": policy.binding,
        "profile": policy.profile,
        "decision_input": decision_input,
        "rng_cursor_after": decision_input["outer_rng"]["cursor"],
        "atlas_delta": {"upsert_entries": [], "retire_entry_ids": []},
        "retained_parent_bank_delta": {
            "retain_entries": [],
            "release_state_ref_ids": [],
            "not_retained": [],
        },
        "rejected_candidate_summary": {},
    }


def test_sha256_counter_prf_is_replayable_and_explicitly_cursor_addressed() -> None:
    first = sha256_counter_prf_u64_v1(seed=91, counter=7, domain="candidate")
    assert first == sha256_counter_prf_u64_v1(
        seed=91, counter=7, domain="candidate"
    )
    assert first != sha256_counter_prf_u64_v1(
        seed=91, counter=8, domain="candidate"
    )
    assert first != sha256_counter_prf_u64_v1(seed=91, counter=7, domain="tie")
    index, cursor = sha256_counter_prf_index_v1(
        seed=91, counter=7, domain="candidate", size=13
    )
    assert index == first % 13
    assert cursor == 8


def test_policy_binding_is_byte_identical_to_manifest_policy_binding_shape() -> None:
    policy = AdaptiveIlluminationPolicyV1()
    binding = canonical_plain_value_v1(policy.binding)

    assert set(binding) == {
        "policy_id",
        "policy_version",
        "implementation_sha256",
        "profile_sha256",
        "profile",
    }
    payload = CanonicalPayloadV1.from_dict(binding["profile"])
    declared = binding["profile_sha256"]
    assert payload.content_digest() == ContentDigest(
        algorithm=declared["algorithm"], hex=declared["hex"]
    )
    assert payload.to_value() == canonical_plain_value_v1(policy.profile)
    assert binding["policy_version"] == policy.policy_version


def test_policy_implementation_identity_binds_source_and_golden_replay() -> None:
    source = ADAPTIVE_ILLUMINATION_SEMANTIC_IMPLEMENTATION_MANIFEST_V1[
        "source_bundle"
    ]
    rows = [
        {
            "path": path,
            "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        }
        for path in source["paths"]
    ]
    assert canonical_sha256(
        {
            "schema_version": "eiel_policy_source_bundle_sha256_v1",
            "files": rows,
        }
    ).hex == ADAPTIVE_ILLUMINATION_SOURCE_BUNDLE_SHA256_HEX_V1

    policy = AdaptiveIlluminationPolicyV1()
    result = policy.decide(_decision_input(policy))
    semantic_projection = {
        key: canonical_plain_value_v1(result[key])
        for key in (
            "outer_rng",
            "atlas_delta",
            "retained_parent_bank_delta",
            "semantic_actions",
            "rejected_candidate_summary",
        )
    }
    assert canonical_sha256(semantic_projection).to_dict() == (
        ADAPTIVE_ILLUMINATION_GOLDEN_REPLAY_VECTOR_V1[
            "expected_semantic_output_sha256"
        ]
    )


def test_adaptive_policy_is_immutable_replayable_and_frontier_ordinal_native() -> None:
    policy = AdaptiveIlluminationPolicyV1()
    assert isinstance(policy, ExpeditionPolicyV1)
    decision_input = _decision_input(policy)

    first = policy.decide(decision_input)
    second = policy.decide(decision_input)

    assert canonical_plain_value_v1(first) == canonical_plain_value_v1(second)
    assert first["proposal_sha256"] == second["proposal_sha256"]
    assert first["frontier_ordinal"] == 7
    assert "absolute_tick" not in first
    assert first["authority_effect"] == "NONE_PROPOSAL_ONLY"
    assert first["outer_rng"]["cursor_before"] == 3
    assert first["outer_rng"]["cursor_after"] == 7
    validate_policy_proposal_v1(
        first,
        profile=policy.profile,
        decision_input=decision_input,
        expected_binding=policy.binding,
    )
    with pytest.raises(TypeError):
        first["frontier_ordinal"] = 8
    with pytest.raises(TypeError):
        first["outer_rng"]["cursor_after"] = 14

    actions = list(first["semantic_actions"])
    founder_actions = [row for row in actions if row["action_kind"] == "EXPLORE_GLOBAL"]
    assert founder_actions
    assert {row["child_start_next_tick"] for row in founder_actions} == {0}
    assert {
        row["founder_execution"]["world_count"] for row in founder_actions
    } == {64}
    assert {
        row["founder_execution"]["seed_derivation"] for row in founder_actions
    } == {"sha256_manifest_action_replicate_ordinal_unique_u63_v1"}
    assert len(
        {
            canonical_sha256(
                canonical_plain_value_v1(row["control_bindings"])
            ).hex
            for row in founder_actions
        }
    ) == len(founder_actions)
    state_actions = [row for row in actions if "parent_state_ref_id" in row]
    assert state_actions
    assert {row["observation_absolute_tick"] for row in state_actions} <= {6, 10, 18}
    assert {row["parent_next_absolute_tick"] for row in state_actions} <= {7, 11, 19}
    branch = next(row for row in actions if row["action_kind"] == "BRANCH_LOCAL")
    replicate = next(row for row in actions if row["action_kind"] == "REPLICATE")
    assert branch["parent_state_ref_id"] == "state-frontier"
    assert branch["control_bindings"] == (
        {
            "control_id": "environment.resource.multiplier",
            "value": 1.5,
        },
        {"control_id": "mutation.rate", "value": 0.125},
    )
    assert replicate["parent_state_ref_id"] == "state-persistent"
    assert replicate["control_bindings"] == (
        {
            "control_id": "environment.resource.multiplier",
            "value": 1.0,
        },
        {"control_id": "mutation.rate", "value": 0.125},
    )
    assert replicate["replication"] == {
        "schema_version": "eiel_adaptive_historical_replicate_policy_v1",
        "replicate_count_per_action": 1,
        "parent_state_class": "BRANCHABLE_EXACT",
        "control_rule": "PRESERVE_AUTHENTICATED_PARENT_CONTROLS",
        "rng_event_rule": "PRESERVE_EXACT_PARENT_CONTINUATION",
        "child_identity_rule": "FRESH_RUN_ATTEMPT_NEVER_RETRY",
        "pairing_rule": "same_parent_occurrence_exact_suffix_v1",
    }


def test_policy_rejects_a_fake_single_frontier_absolute_tick() -> None:
    policy = AdaptiveIlluminationPolicyV1()
    decision_input = _decision_input(policy)
    decision_input["decision_frontier"]["absolute_tick"] = 18

    with pytest.raises(ExpeditionPolicyError, match="cannot claim one absolute tick"):
        policy.decide(decision_input)


def test_profile_threshold_change_is_manifest_bound_and_changes_policy_only() -> None:
    baseline = AdaptiveIlluminationPolicyV1()
    changed_profile = canonical_plain_value_v1(ADAPTIVE_ILLUMINATION_V1_PROFILE)
    changed_profile["profile_version"] = "1-test-frontier-100"
    frontier_classifier = next(
        row
        for row in changed_profile["classifiers"]
        if row["class_id"] == "INFORMATIVE_FRONTIER"
    )
    frontier_classifier["predicates"][0]["maximum"] = 100
    changed = AdaptiveIlluminationPolicyV1(changed_profile)
    baseline_input = _decision_input(baseline)
    changed_input = _decision_input(changed)

    baseline_result = baseline.decide(baseline_input)
    changed_result = changed.decide(changed_input)

    assert baseline.binding["profile_sha256"] != changed.binding["profile_sha256"]
    assert baseline_result["proposal_sha256"] != changed_result["proposal_sha256"]
    baseline_classes = baseline_result["rejected_candidate_summary"][
        "classification_counts"
    ]
    changed_classes = changed_result["rejected_candidate_summary"][
        "classification_counts"
    ]
    assert baseline_classes["INFORMATIVE_FRONTIER"] == 1
    assert "INFORMATIVE_FRONTIER" not in changed_classes
    with pytest.raises(ExpeditionPolicyError, match="binding differs"):
        changed.decide(baseline_input)


def test_historical_actions_fail_closed_without_exact_parent_control_authority() -> None:
    policy = AdaptiveIlluminationPolicyV1()
    decision_input = _decision_input(policy)
    persistent_state = next(
        row
        for row in decision_input["available_state_refs"]
        if row["state_ref_id"] == "state-persistent"
    )
    persistent_state["state_class"] = "OBSERVATION_ONLY"
    frontier_frame = next(
        row
        for row in decision_input["decision_frontier"]["observation_frames"]
        if row["frame_id"] == "frame-frontier"
    )
    frontier_frame["control_readback"]["controls"]["mutation.rate"] = 0.999

    result = policy.decide(decision_input)
    actions = result["semantic_actions"]

    assert not any(
        row["action_kind"] in {"BRANCH_LOCAL", "REPLICATE"} for row in actions
    )
    assert result["rejected_candidate_summary"][
        "historical_action_ineligible_counts"
    ] == {
        "HISTORICAL_PARENT_NOT_BRANCHABLE_EXACT": 1,
        "PARENT_CONTROL_OUTSIDE_AUTHORIZED_DOMAIN": 1,
    }


def test_founder_world_count_is_profile_bound_and_replaceable() -> None:
    baseline = AdaptiveIlluminationPolicyV1()
    profile = canonical_plain_value_v1(ADAPTIVE_ILLUMINATION_V1_PROFILE)
    profile["profile_version"] = "1-test-founder-world-count"
    profile["candidate_generators"]["global_exploration"][
        "founder_execution"
    ]["world_count"] = 3
    changed = AdaptiveIlluminationPolicyV1(profile)

    baseline_actions = baseline.decide(_decision_input(baseline))["semantic_actions"]
    changed_actions = changed.decide(_decision_input(changed))["semantic_actions"]
    baseline_founders = [
        row for row in baseline_actions if row["action_kind"] == "EXPLORE_GLOBAL"
    ]
    changed_founders = [
        row for row in changed_actions if row["action_kind"] == "EXPLORE_GLOBAL"
    ]

    assert {row["founder_execution"]["world_count"] for row in baseline_founders} == {
        64
    }
    assert {row["founder_execution"]["world_count"] for row in changed_founders} == {
        3
    }
    assert baseline.binding["profile_sha256"] != changed.binding["profile_sha256"]
    assert baseline_founders != changed_founders


def test_global_generator_can_be_profile_disabled_for_exact_only_expedition() -> None:
    profile = canonical_plain_value_v1(ADAPTIVE_ILLUMINATION_V1_PROFILE)
    profile["profile_version"] = "1-exact-only-test"
    profile["candidate_generators"]["global_exploration"]["enabled"] = False
    policy = AdaptiveIlluminationPolicyV1(profile)

    actions = policy.decide(_decision_input(policy))["semantic_actions"]

    assert not any(row["action_kind"] == "EXPLORE_GLOBAL" for row in actions)
    assert any(row["action_kind"] == "BRANCH_LOCAL" for row in actions)
    assert any(row["action_kind"] == "REPLICATE" for row in actions)


def test_exact_workhorse_profile_uses_only_common_authenticated_rate_facts() -> None:
    policy = AdaptiveIlluminationPolicyV1(
        ADAPTIVE_ILLUMINATION_EXACT_WORKHORSE_V1_PROFILE
    )

    result = policy.decide(_decision_input(policy))
    descriptor_ids = {
        descriptor["descriptor_id"] for descriptor in policy.profile["descriptor_definitions"]
    }

    assert descriptor_ids == {
        "active_fraction_milli",
        "births_per_tick_milli",
        "deaths_per_tick_milli",
    }
    assert not any(
        row["action_kind"] == "EXPLORE_GLOBAL"
        for row in result["semantic_actions"]
    )
    assert "maximum_lineage_depth" not in canonical_plain_value_v1(policy.profile)


def test_exact_workhorse_axis_round_robin_profile_alternates_bounded_controls() -> None:
    policy = AdaptiveIlluminationPolicyV1(
        ADAPTIVE_ILLUMINATION_EXACT_WORKHORSE_AXIS_ROUND_ROBIN_V1_PROFILE
    )
    resource_input = _decision_input(policy)
    resource_input["decision_frontier"]["frontier_id"] = "frontier-0000"
    resource_input["decision_frontier"]["frontier_ordinal"] = 0
    resource = policy.decide(resource_input)
    resource_branch = next(
        row
        for row in resource["semantic_actions"]
        if row["action_kind"] == "BRANCH_LOCAL"
    )
    resource_controls = {
        row["control_id"]: row["value"]
        for row in resource_branch["control_bindings"]
    }
    assert resource_controls == {
        "environment.resource.multiplier": 1.5,
        "mutation.rate": 0.125,
    }

    mutation_input = _decision_input(policy)
    mutation_input["decision_frontier"]["frontier_id"] = "frontier-0001"
    mutation_input["decision_frontier"]["frontier_ordinal"] = 1
    mutation = policy.decide(mutation_input)
    mutation_branch = next(
        row
        for row in mutation["semantic_actions"]
        if row["action_kind"] == "BRANCH_LOCAL"
    )
    mutation_controls = {
        row["control_id"]: row["value"]
        for row in mutation_branch["control_bindings"]
    }
    assert mutation_controls == {
        "environment.resource.multiplier": 1.0,
        "mutation.rate": 0.25,
    }
    assert policy.profile["profile_id"] != (
        ADAPTIVE_ILLUMINATION_EXACT_WORKHORSE_V1_PROFILE["profile_id"]
    )


def test_exact_workhorse_axis_profile_schedules_branch_branch_replicate_for_persistent_state() -> None:
    policy = AdaptiveIlluminationPolicyV1(
        ADAPTIVE_ILLUMINATION_EXACT_WORKHORSE_AXIS_ROUND_ROBIN_V1_PROFILE
    )
    expected = (
        (
            "BRANCH_LOCAL",
            {
                "environment.resource.multiplier": 1.5,
                "mutation.rate": 0.125,
            },
        ),
        (
            "BRANCH_LOCAL",
            {
                "environment.resource.multiplier": 1.0,
                "mutation.rate": 0.25,
            },
        ),
        (
            "REPLICATE",
            {
                "environment.resource.multiplier": 1.0,
                "mutation.rate": 0.125,
            },
        ),
    )

    observed_action_kinds: list[str] = []
    for frontier_ordinal, (action_kind, controls) in enumerate(expected):
        decision_input = _decision_input(policy)
        decision_input["decision_frontier"]["frontier_id"] = (
            f"frontier-{frontier_ordinal:04d}"
        )
        decision_input["decision_frontier"]["frontier_ordinal"] = frontier_ordinal
        result = policy.decide(decision_input)
        persistent = next(
            row
            for row in result["semantic_actions"]
            if row.get("source_frame_id") == "frame-persistent"
        )

        assert persistent["action_kind"] == action_kind
        observed_action_kinds.append(persistent["action_kind"])
        assert {
            row["control_id"]: row["value"]
            for row in persistent["control_bindings"]
        } == controls
    assert observed_action_kinds == ["BRANCH_LOCAL", "BRANCH_LOCAL", "REPLICATE"]
    assert observed_action_kinds[2] == "REPLICATE"


def test_toy_policy_proves_the_coordinator_port_is_replaceable() -> None:
    adaptive = AdaptiveIlluminationPolicyV1()
    toy = ToyRoundRobinPolicyV1()
    assert isinstance(toy, ExpeditionPolicyV1)

    adaptive_result = adaptive.decide(_decision_input(adaptive))
    toy_input = _decision_input(toy)
    toy_result = toy.decide(toy_input)

    assert set(adaptive_result) == set(toy_result)
    assert toy_result["policy_binding"]["policy_id"] == toy.policy_id
    assert adaptive_result["policy_binding"]["policy_id"] != toy.policy_id
    assert toy_result["atlas_delta"] == {
        "retire_entry_ids": (),
        "upsert_entries": (),
    }
    assert all(row["action_kind"] == "CONTINUE" for row in toy_result["semantic_actions"])
    validate_policy_proposal_v1(
        toy_result,
        profile=toy.profile,
        decision_input=toy_input,
        expected_binding=toy.binding,
    )


def test_hidden_reward_and_unbounded_named_phenotype_authority_are_rejected() -> None:
    policy = AdaptiveIlluminationPolicyV1()
    decision_input = _decision_input(policy)
    kwargs = _proposal_core_kwargs(policy, decision_input)

    with pytest.raises(ExpeditionPolicyError, match="hidden scalar/phenotype"):
        seal_policy_proposal_v1(
            **kwargs,
            semantic_actions=[
                {
                    "action_id": "A-hidden",
                    "action_kind": "CONTINUE",
                    "reward": 1,
                }
            ],
        )
    with pytest.raises(ExpeditionPolicyError, match="lacks bounded probe authority"):
        seal_policy_proposal_v1(
            **kwargs,
            semantic_actions=[
                {"action_id": "A-probe", "action_kind": "CAUSAL_PROBE"}
            ],
        )

    invalid_profile = canonical_plain_value_v1(ADAPTIVE_ILLUMINATION_V1_PROFILE)
    invalid_profile["profile_version"] = "invalid-phenotype"
    invalid_profile["phenotype_name"] = "memory"
    with pytest.raises(ExpeditionPolicyError):
        AdaptiveIlluminationPolicyV1(invalid_profile)


def test_positive_input_authority_drops_undeclared_aliases_before_policy_use() -> None:
    policy = AdaptiveIlluminationPolicyV1()
    baseline_input = _decision_input(policy)
    injected_input = canonical_plain_value_v1(baseline_input)
    injected_input["decision_frontier"]["observation_frames"][0]["facts"][
        "interesting_target_alias"
    ] = 10**12

    projected = project_policy_input_v1(injected_input, profile=policy.profile)
    facts = projected["decision_frontier"]["observation_frames"][0]["facts"]
    assert "interesting_target_alias" not in facts
    assert policy.decide(injected_input) == policy.decide(baseline_input)

    invalid_profile = canonical_plain_value_v1(ADAPTIVE_ILLUMINATION_V1_PROFILE)
    invalid_profile["profile_version"] = "invalid-reward-input-authority"
    paths = invalid_profile["policy_input_authority"]["allowed_paths"]
    paths.append(
        "/decision_frontier/observation_frames/*/facts/reward"
    )
    paths.sort()
    with pytest.raises(ExpeditionPolicyError, match="bounded probe contract"):
        AdaptiveIlluminationPolicyV1(invalid_profile)


def test_named_phenotype_can_only_activate_through_explicit_bounded_probe() -> None:
    profile = canonical_plain_value_v1(ADAPTIVE_ILLUMINATION_V1_PROFILE)
    profile["profile_version"] = "1-bounded-probe-test"
    profile["probe_contracts"] = [
        {
            "probe_contract_id": "probe-memory-candidate-v1",
            "phenotype_label": "memory_candidate",
            "observable_predicates": [
                {
                    "source": "descriptor",
                    "descriptor_id": "active_fraction_milli",
                    "operator": "GE",
                    "value": 300,
                }
            ],
            "matched_comparator": {
                "kind": "paired_control",
                "pairing": "same_source_state_and_seed_bindings",
            },
            "authorized_control_ids": ["agent.memory.enabled"],
            "action_template": {
                "action_kind": "CAUSAL_PROBE",
                "bucket": "observation",
                "slot_cost": 1,
            },
            "maximum_actions": 1,
            "maximum_replicates": 2,
            "maximum_absolute_ticks": 100,
            "stop_rules": ["budget_exhausted", "integrity_failure"],
            "claim_boundary": "candidate mechanism only; no phenotype claim",
        }
    ]
    policy = AdaptiveIlluminationPolicyV1(profile)
    decision_input = _decision_input(policy)
    decision_input["logical_capacity"] = 10

    result = policy.decide(decision_input)

    probe_actions = [
        row for row in result["semantic_actions"] if row["action_kind"] == "CAUSAL_PROBE"
    ]
    assert len(probe_actions) == 1
    assert probe_actions[0]["probe_contract_id"] == "probe-memory-candidate-v1"
    assert "phenotype_label" not in probe_actions[0]
    assert "phenotype_name" not in probe_actions[0]


def test_parent_bank_keeps_same_niche_states_with_declared_divergence() -> None:
    policy = AdaptiveIlluminationPolicyV1()
    decision_input = _decision_input(policy)
    second_frame = canonical_plain_value_v1(
        decision_input["decision_frontier"]["observation_frames"][0]
    )
    second_frame.update(
        {
            "frame_id": "frame-frontier-sibling",
            "trajectory_id": "trajectory-frontier-sibling",
            "absolute_tick": 12,
            "state_ref_id": "state-frontier-sibling",
        }
    )
    decision_input["decision_frontier"]["observation_frames"].append(second_frame)
    decision_input["available_state_refs"].append(
        _state_ref(
            "state-frontier-sibling",
            next_absolute_tick=13,
            lineage_id="lineage-sibling",
            genotype="genotype-sibling",
        )
    )

    result = policy.decide(decision_input)
    retained = result["retained_parent_bank_delta"]["retain_entries"]
    retained_ids = {row["state_ref_id"] for row in retained}

    assert {"state-frontier", "state-frontier-sibling"} <= retained_ids
    niches = {
        row["niche_id"]
        for row in retained
        if row["state_ref_id"] in {"state-frontier", "state-frontier-sibling"}
    }
    assert len(niches) == 1


def test_parent_bank_replays_cross_frontier_divergence_proof_without_atlas() -> None:
    policy = AdaptiveIlluminationPolicyV1()
    first_input = _decision_input(policy)
    first = canonical_plain_value_v1(policy.decide(first_input))
    retained = next(
        row
        for row in first["retained_parent_bank_delta"]["retain_entries"]
        if row["state_ref_id"] == "state-frontier"
    )

    second_input = _decision_input(policy)
    second_input["decision_frontier"]["frontier_ordinal"] = 8
    new_frame = canonical_plain_value_v1(
        second_input["decision_frontier"]["observation_frames"][0]
    )
    new_frame.update(
        {
            "frame_id": "frame-frontier-new",
            "absolute_tick": 12,
            "state_ref_id": "state-frontier-new",
        }
    )
    second_input["decision_frontier"]["observation_frames"].append(new_frame)
    new_state = _state_ref(
        "state-frontier-new",
        next_absolute_tick=13,
        lineage_id="lineage-a",
        genotype="genotype-a",
    )
    new_state["source_object_sha256"] = "source-state-frontier"
    new_state["control_history_sha256"] = "history-state-frontier"
    second_input["available_state_refs"].append(new_state)
    # The Atlas has intentionally evicted the representative.  The opaque
    # signature carried by the durable policy proposal is the authority for
    # the cross-frontier comparison.
    second_input["retained_parent_bank"] = {
        "maximum_entries": 32,
        "entries": [
            {
                "state_ref_id": retained["state_ref_id"],
                "reason_codes": [retained["survival_reason"]],
                "divergence_signature": retained["divergence_signature"],
            }
        ],
    }
    result = canonical_plain_value_v1(policy.decide(second_input))
    assert {
        row["reason_code"]
        for row in result["retained_parent_bank_delta"]["not_retained"]
        if row["state_ref_id"] == "state-frontier-new"
    } == {"NO_PROFILE_DEFINED_MARGINAL_DIVERGENCE"}

    changed_input = canonical_plain_value_v1(second_input)
    changed_state = next(
        row
        for row in changed_input["available_state_refs"]
        if row["state_ref_id"] == "state-frontier-new"
    )
    changed_state["control_history_sha256"] = "changed-history"
    changed_result = canonical_plain_value_v1(policy.decide(changed_input))
    assert any(
        row["state_ref_id"] == "state-frontier-new"
        for row in changed_result["retained_parent_bank_delta"]["retain_entries"]
    )


def test_policy_records_observation_only_state_as_not_retained() -> None:
    policy = AdaptiveIlluminationPolicyV1()
    decision_input = _decision_input(policy)
    frontier_state = next(
        row
        for row in decision_input["available_state_refs"]
        if row["state_ref_id"] == "state-frontier"
    )
    frontier_state["state_class"] = "OBSERVATION_ONLY"

    result = policy.decide(decision_input)
    delta = result["retained_parent_bank_delta"]

    assert not any(
        row["state_ref_id"] == "state-frontier"
        for row in delta["retain_entries"]
    )
    assert {
        "state_ref_id": "state-frontier",
        "reason_code": "STATE_NOT_EXACTLY_CONTINUABLE",
    } in delta["not_retained"]


def test_policy_modules_import_no_io_clock_random_provider_or_simulator() -> None:
    roots = [
        Path("eiel_composition/adaptive_expedition/policy_protocol.py"),
        Path(
            "eiel_composition/adaptive_expedition/policies/adaptive_illumination_v1.py"
        ),
        Path("eiel_composition/adaptive_expedition/policies/toy_round_robin_v1.py"),
    ]
    forbidden_roots = {
        "asyncio",
        "engine",
        "os",
        "pathlib",
        "random",
        "secrets",
        "socket",
        "subprocess",
        "time",
        "urllib",
    }
    for path in roots:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        assert imported.isdisjoint(forbidden_roots), (path, imported & forbidden_roots)


def test_policy_input_freezes_callers_data_before_use() -> None:
    policy = AdaptiveIlluminationPolicyV1()
    decision_input = _decision_input(policy)
    checked = validate_policy_input_v1(
        decision_input, expected_binding=policy.binding, profile=policy.profile
    )
    decision_input["decision_frontier"]["frontier_ordinal"] = 99

    assert checked["decision_frontier"]["frontier_ordinal"] == 7
    with pytest.raises(TypeError):
        checked["decision_frontier"]["frontier_ordinal"] = 8


@pytest.mark.parametrize("world_count", [1024, 10000])
def test_founder_world_count_profiles_are_distinct_manifest_bound_choices(
    world_count: int,
) -> None:
    profile = founder_workhorse_profile_for_world_count_v1(world_count)
    policy = AdaptiveIlluminationPolicyV1(profile)
    generator = profile["candidate_generators"]["global_exploration"]
    assert generator["founder_execution"]["world_count"] == world_count
    assert profile["profile_id"].endswith(f"W{world_count}_PROFILE_V1")
    assert policy.binding["profile_sha256"]["hex"]
    assert policy.binding["profile"]["payload"]["profile_id"] == profile[
        "profile_id"
    ]
