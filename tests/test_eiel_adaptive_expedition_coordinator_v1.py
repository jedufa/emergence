from __future__ import annotations

from collections.abc import Mapping

import pytest

from emergence_contracts.adaptive_science_expedition_v1 import (
    EvolutionaryAtlasV1,
    ExpeditionManifestV1,
    ObservationFrameV1,
    RetainedParentBankV1,
    StateRefV1,
)
from emergence_contracts.canonical import canonical_sha256
from eiel_composition.adaptive_expedition.coordinator import (
    ExpeditionCoordinatorError,
    prepare_decision_v1,
)
from eiel_composition.adaptive_expedition.policies.adaptive_illumination_v1 import (
    AdaptiveIlluminationPolicyV1,
)
from eiel_composition.adaptive_expedition.policies.toy_round_robin_v1 import (
    ToyRoundRobinPolicyV1,
)
from eiel_composition.adaptive_expedition.policy_protocol import (
    ExpeditionPolicyV1,
    canonical_plain_value_v1,
    seal_policy_proposal_v1,
)
from eiel_composition.adaptive_expedition.lifecycle import (
    default_expedition_lifecycle_policy_value_v1,
)


def _digest(label: str) -> dict[str, str]:
    return canonical_sha256({"label": label}).to_dict()


def _payload(schema_version: str, value: object) -> dict[str, object]:
    return {"schema_version": schema_version, "payload": value}


def _artifact(label: str, *, size_bytes: int = 128) -> dict[str, object]:
    return {
        "schema_version": "run_artifact_ref_v1",
        "artifact_id": f"artifact-{label}",
        "artifact_kind": "native_checkpoint",
        "artifact_schema_version": "native_checkpoint_v1",
        "locator": f"opaque://lifecycle/{label}",
        "content_digest": _digest(f"artifact-{label}"),
        "size_bytes": size_bytes,
    }


def _state() -> StateRefV1:
    artifact = _artifact("state")
    return StateRefV1.from_dict(
        {
            "schema_version": "eiel_adaptive_science_state_ref_v1",
            "expedition_id": "expedition-coordinator",
            "world_trajectory_id": "trajectory-a",
            "logical_world_id": 7,
            "next_absolute_tick": 480,
            "state_class": "BRANCHABLE_EXACT",
            "semantic_world_sha256": _digest("world"),
            "control_history_sha256": _digest("controls"),
            "source_object_sha256": artifact["content_digest"],
            "source_artifact_ref": artifact,
            "exact_state_sha256": _digest("exact-state"),
            "checkpoint_occurrence_sha256": _digest("checkpoint-occurrence"),
            "rng_state_sha256": _digest("rng"),
            "event_prefix_sha256": _digest("event-prefix"),
            "parent_occurrence_sha256": _digest("parent-occurrence"),
            "replay_recipe_sha256": None,
        }
    )


def _manifest(policy: ExpeditionPolicyV1) -> ExpeditionManifestV1:
    return ExpeditionManifestV1.from_dict(
        {
            "schema_version": "eiel_adaptive_science_expedition_manifest_v1",
            "expedition_id": "expedition-coordinator",
            "question_description": "Exercise a bounded deterministic policy port",
            "no_claims": True,
            "hypothesis_ids": [],
            "required_science_instrument": "EIEL_SCIENCE_INSTRUMENT_V1",
            "founder_basis_refs": [_artifact("founder")],
            "search_space": _payload(
                "adaptive_search_space_v1",
                {
                    "controls": [
                        {
                            "control_id": "environment.resource.multiplier",
                            "allowed_values": [1.0, 1.5],
                        },
                        {
                            "control_id": "mutation.rate",
                            "allowed_values": [0.125, 0.25],
                        },
                    ]
                },
            ),
            "epoch_ticks": 120,
            "maximum_absolute_tick": 2400,
            "maximum_frontiers": 20,
            "maximum_actions_per_frontier": 16,
            "wall_time_budget_seconds": None,
            "observation_profile_id": "coordinator-observation-profile",
            "observation_descriptor_set_sha256": _digest("descriptor-set"),
            "policy_binding": canonical_plain_value_v1(policy.binding),
            "outer_search_seed": 0xA51CE,
            "maximum_atlas_entries": 64,
            "maximum_retained_parent_entries": 32,
            "maximum_retained_parent_bytes": 4096,
            "artifact_lifecycle_policy": _payload(
                "eiel_adaptive_expedition_lifecycle_policy_v1",
                default_expedition_lifecycle_policy_value_v1(),
            ),
            "stop_conditions": ["integrity_failure", "operator_stop"],
        }
    )


def _frame() -> ObservationFrameV1:
    state = _state()
    return ObservationFrameV1.from_dict(
        {
            "schema_version": "eiel_adaptive_science_observation_frame_v1",
            "expedition_id": state.expedition_id,
            "world_trajectory_id": state.world_trajectory_id,
            "logical_world_id": state.logical_world_id,
            "next_absolute_tick": state.next_absolute_tick,
            "semantic_world_sha256": state.semantic_world_sha256.to_dict(),
            "control_history_sha256": state.control_history_sha256.to_dict(),
            "observation_profile_id": "coordinator-observation-profile",
            "observation_profile_sha256": _digest("observation-profile"),
            "observation_descriptor_set_sha256": _digest("descriptor-set"),
            "reducer_implementation_sha256": _digest("reducer"),
            "observations": _payload("coordinator_observation_v1", {"rows": []}),
            "control_readback": _payload(
                "coordinator_readback_v1",
                {
                    "controls": {
                        "environment.resource.multiplier": 1.0,
                        "mutation.rate": 0.125,
                    }
                },
            ),
            "policy_facing_facts": _payload(
                "coordinator_policy_facts_v1",
                {
                    "integrity_ok": True,
                    "active_count": 2,
                    "capacity": 10,
                    "birth_count": 1,
                    "maximum_lineage_depth": 1,
                },
            ),
            "descriptor_values": _payload("coordinator_descriptors_v1", {}),
            "calibration": _payload("coordinator_calibration_v1", {}),
            "source_state_ref": state.to_dict(),
            "previous_frame_sha256": None,
        }
    )


def _atlas(manifest: ExpeditionManifestV1) -> EvolutionaryAtlasV1:
    return EvolutionaryAtlasV1.from_dict(
        {
            "schema_version": "eiel_adaptive_science_evolutionary_atlas_v1",
            "expedition_id": manifest.expedition_id,
            "manifest_sha256": manifest.manifest_sha256().to_dict(),
            "revision": 0,
            "through_frontier_ordinal": 0,
            "descriptor_set_sha256": manifest.observation_descriptor_set_sha256.to_dict(),
            "maximum_entries": 64,
            "entries": [],
            "previous_atlas_sha256": None,
        }
    )


def _bank(manifest: ExpeditionManifestV1) -> RetainedParentBankV1:
    return RetainedParentBankV1.from_dict(
        {
            "schema_version": "eiel_adaptive_science_retained_parent_bank_v1",
            "expedition_id": manifest.expedition_id,
            "manifest_sha256": manifest.manifest_sha256().to_dict(),
            "revision": 0,
            "maximum_entries": manifest.maximum_retained_parent_entries,
            "maximum_total_bytes": manifest.maximum_retained_parent_bytes,
            "current_entry_count": 0,
            "current_total_bytes": 0,
            "entries": [],
            "previous_bank_sha256": None,
        }
    )


def _bank_with_parent(manifest: ExpeditionManifestV1) -> RetainedParentBankV1:
    state = _state()
    return RetainedParentBankV1.from_dict(
        {
            "schema_version": "eiel_adaptive_science_retained_parent_bank_v1",
            "expedition_id": manifest.expedition_id,
            "manifest_sha256": manifest.manifest_sha256().to_dict(),
            "revision": 0,
            "maximum_entries": manifest.maximum_retained_parent_entries,
            "maximum_total_bytes": manifest.maximum_retained_parent_bytes,
            "current_entry_count": 1,
            "current_total_bytes": state.source_artifact_ref.size_bytes,
            "entries": [
                {
                    "state_ref": state.to_dict(),
                    "selected_at_frontier_ordinal": 0,
                    "reason_codes": ["FUTURE_PARENT"],
                    "retained_bytes": state.source_artifact_ref.size_bytes,
                }
            ],
            "previous_bank_sha256": None,
        }
    )


def _prepare(policy: ExpeditionPolicyV1):
    manifest = _manifest(policy)
    return prepare_decision_v1(
        manifest=manifest,
        policy=policy,
        frontier_id="frontier-0000",
        frontier_ordinal=0,
        observation_frames=[_frame()],
        bounded_history=[],
        atlas=_atlas(manifest),
        retained_parent_bank=_bank(manifest),
        logical_capacity=4,
        outer_rng_cursor=0,
    )


def test_coordinator_seals_byte_identical_adaptive_decisions() -> None:
    policy = AdaptiveIlluminationPolicyV1()
    first = _prepare(policy)
    second = _prepare(policy)

    assert first.decision.to_dict() == second.decision.to_dict()
    assert first.decision.decision_sha256() == second.decision.decision_sha256()
    assert first.policy_input_sha256 == second.policy_input_sha256
    assert first.proposal_sha256 == second.proposal_sha256
    assert first.decision.policy_proposal.to_value()["authority_effect"] == (
        "NONE_PROPOSAL_ONLY"
    )
    assert first.decision.action_rows
    assert all(row.effective_next_absolute_tick >= 0 for row in first.decision.action_rows)


def test_same_coordinator_accepts_an_unrelated_policy_implementation() -> None:
    prepared = _prepare(ToyRoundRobinPolicyV1())
    assert prepared.decision.action_rows
    assert {row.quota_bucket for row in prepared.decision.action_rows} == {
        "round_robin"
    }


def test_manifest_binding_rejects_a_different_policy() -> None:
    manifest = _manifest(AdaptiveIlluminationPolicyV1())
    with pytest.raises(ExpeditionCoordinatorError, match="frozen manifest binding"):
        prepare_decision_v1(
            manifest=manifest,
            policy=ToyRoundRobinPolicyV1(),
            frontier_id="frontier-0000",
            frontier_ordinal=0,
            observation_frames=[_frame()],
            bounded_history=[],
            atlas=_atlas(manifest),
            retained_parent_bank=_bank(manifest),
            logical_capacity=1,
            outer_rng_cursor=0,
        )


def test_adaptive_policy_fails_closed_without_parent_divergence_proof() -> None:
    policy = AdaptiveIlluminationPolicyV1()
    manifest = _manifest(policy)
    with pytest.raises(
        ExpeditionCoordinatorError,
        match="requires durable policy divergence proofs",
    ):
        prepare_decision_v1(
            manifest=manifest,
            policy=policy,
            frontier_id="frontier-0000",
            frontier_ordinal=0,
            observation_frames=[_frame()],
            bounded_history=[],
            atlas=_atlas(manifest),
            retained_parent_bank=_bank_with_parent(manifest),
            logical_capacity=1,
            outer_rng_cursor=0,
        )


class _AlternatingPolicy:
    def __init__(self) -> None:
        self._base = AdaptiveIlluminationPolicyV1()
        self._calls = 0

    @property
    def policy_id(self) -> str:
        return self._base.policy_id

    @property
    def policy_version(self) -> str:
        return self._base.policy_version

    @property
    def profile(self) -> Mapping[str, object]:
        return self._base.profile

    @property
    def binding(self) -> Mapping[str, object]:
        return self._base.binding

    def decide(self, decision_input: Mapping[str, object]) -> Mapping[str, object]:
        self._calls += 1
        return seal_policy_proposal_v1(
            policy_binding=self.binding,
            profile=self.profile,
            decision_input=decision_input,
            rng_cursor_after=decision_input["outer_rng"]["cursor"],
            atlas_delta={"upsert_entries": [], "retire_entry_ids": []},
            retained_parent_bank_delta={
                "retain_entries": [],
                "release_state_ref_ids": [],
                "not_retained": [],
            },
            semantic_actions=[],
            rejected_candidate_summary={"alternating_value": self._calls},
        )


def test_coordinator_rejects_non_deterministic_policy_replay() -> None:
    with pytest.raises(ExpeditionCoordinatorError, match="not byte-identical"):
        _prepare(_AlternatingPolicy())
