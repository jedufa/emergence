"""Minimal alternate policy used to prove coordinator-level replaceability."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Final

from emergence_contracts.canonical import canonical_sha256
from emergence_contracts.ids import ContentDigest

from ..policy_protocol import (
    ExpeditionPolicyError,
    freeze_canonical_mapping_v1,
    policy_binding_v1,
    seal_policy_proposal_v1,
    sha256_counter_prf_index_v1,
    validate_policy_input_authority_v1,
    validate_policy_input_v1,
)


TOY_ROUND_ROBIN_POLICY_ID_V1: Final[str] = "EIEL_TOY_ROUND_ROBIN_POLICY_V1"
TOY_ROUND_ROBIN_POLICY_VERSION_V1: Final[str] = "1"
TOY_ROUND_ROBIN_PROFILE_SCHEMA_VERSION_V1: Final[str] = (
    "eiel_toy_round_robin_profile_v1"
)
TOY_ROUND_ROBIN_ALGORITHM_SHA256_V1: Final[ContentDigest] = canonical_sha256(
    {
        "algorithm_id": "eiel_toy_round_robin_policy_v1",
        "ordering": "trajectory_id_then_frame_id_circular",
        "start": "sha256_counter_prf_index_v1",
        "authority_effect": "NONE_PROPOSAL_ONLY",
    }
)

TOY_ROUND_ROBIN_V1_PROFILE: Final[Mapping[str, object]] = freeze_canonical_mapping_v1(
    {
        "profile_id": "EIEL_TOY_ROUND_ROBIN_PROFILE_V1",
        "profile_version": "1.1",
        "policy_input_authority": {
            "schema_version": "eiel_adaptive_policy_input_authority_v1",
            "allowed_paths": [
                "/available_state_refs/*/next_absolute_tick",
                "/available_state_refs/*/state_ref_id",
                "/decision_frontier/frontier_id",
                "/decision_frontier/frontier_ordinal",
                "/decision_frontier/observation_frames/*/absolute_tick",
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
            ],
        },
        "action_template": {
            "action_kind": "CONTINUE",
            "bucket": "round_robin",
            "slot_cost": 1,
            "reason_code": "TOY_ROUND_ROBIN_SELECTION",
            "requires_state_ref": True,
        },
        "tie_break": "trajectory_id_then_frame_id",
        "human_review_projection": {
            "schema_version": "eiel_adaptive_policy_review_presentation_v1",
            "atlas_role_sections": [],
            "action_reason_sections": [],
            "empty_status_by_section": {
                "authorized_search_space": "AUTHORIZED_SPACE_NOT_INDEXED",
                "exploration_coverage": "NO_EXPLORATION_FRONTIER_RECORDED",
                "trivial_or_unexpressed_regions": "NO_POLICY_ROLE_RECORDED",
                "overwhelming_extinction_or_dead_end_regions": "NO_POLICY_ROLE_RECORDED",
                "response_or_frontier_regions": "NO_POLICY_ROLE_RECORDED",
                "novel_persistent_or_divergent_observations": "NO_POLICY_ROLE_RECORDED",
                "failed_replication_or_causal_followups": "NO_POLICY_ASSESSED_FOLLOWUP_FAILURE_RECORDED",
                "retained_parents": "NO_EXACT_PARENT_RETAINED",
                "discarded_parent_candidates": "NO_PARENT_DISCARD_REASON_RECORDED",
                "unresolved_scientific_questions": "NO_UNRESOLVED_ROW_RECORDED",
                "limiting_controls_observations_capabilities_or_budgets": "NO_LIMITATION_ROW_RECORDED",
                "recommended_next_expedition": "NO_POLICY_PRESENTATION_RECOMMENDATION",
                "human_scientific_choices": "NO_HUMAN_CHOICE_ROW_RECORDED",
            },
            "recommendation_rules": [],
            "fallback_recommendation": {
                "recommendation_id": "HUMAN_REVIEW_BEFORE_NEXT_MANIFEST",
                "recommendation": "Review the history and exact retained parents before authoring the next immutable Expedition manifest.",
            },
            "human_scientific_choices": [
                {
                    "choice_id": "AUTHOR_HIGH_LEVEL_QUESTION",
                    "choice": "Choose the next high-level scientific question and authorized search space.",
                },
                {
                    "choice_id": "SELECT_NEXT_SCALE_AND_BUDGET",
                    "choice": "Choose the next Expedition scale, budget, and stop conditions.",
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


def _mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ExpeditionPolicyError(f"{field} must be a mapping")
    return value


def _sequence(value: object, *, field: str) -> Sequence[object]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ExpeditionPolicyError(f"{field} must be a sequence")
    return value


class ToyRoundRobinPolicyV1:
    """Select state-backed trajectories circularly using the same pure port."""

    def __init__(
        self, profile: Mapping[str, object] = TOY_ROUND_ROBIN_V1_PROFILE
    ) -> None:
        self._profile = freeze_canonical_mapping_v1(profile)
        required = {
            "profile_id",
            "profile_version",
            "policy_input_authority",
            "action_template",
            "tie_break",
            "human_review_projection",
            "probe_contracts",
        }
        if set(self._profile) != required:
            raise ExpeditionPolicyError("toy round-robin profile fields must be exact")
        validate_policy_input_authority_v1(self._profile)
        template = _mapping(self._profile["action_template"], field="action_template")
        if (
            template.get("action_kind") != "CONTINUE"
            or template.get("requires_state_ref") is not True
            or type(template.get("slot_cost")) is not int
            or int(template["slot_cost"]) != 1
        ):
            raise ExpeditionPolicyError("toy action template must be state-backed CONTINUE")
        self._binding = policy_binding_v1(
            policy_id=self.policy_id,
            policy_version=self.policy_version,
            algorithm_sha256=TOY_ROUND_ROBIN_ALGORITHM_SHA256_V1,
            profile=self._profile,
            profile_schema_version=TOY_ROUND_ROBIN_PROFILE_SCHEMA_VERSION_V1,
        )

    @property
    def policy_id(self) -> str:
        return TOY_ROUND_ROBIN_POLICY_ID_V1

    @property
    def policy_version(self) -> str:
        return TOY_ROUND_ROBIN_POLICY_VERSION_V1

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
        state_refs = {}
        for raw in _sequence(checked["available_state_refs"], field="state refs"):
            row = _mapping(raw, field="state ref")
            state_ref_id = row.get("state_ref_id")
            next_tick = row.get("next_absolute_tick")
            if (
                type(state_ref_id) is not str
                or not state_ref_id
                or state_ref_id in state_refs
                or type(next_tick) is not int
                or next_tick < 0
            ):
                raise ExpeditionPolicyError("toy state ref identity/tick is invalid")
            state_refs[state_ref_id] = row
        frontier = _mapping(checked["decision_frontier"], field="decision frontier")
        eligible = []
        for raw in _sequence(frontier["observation_frames"], field="observation frames"):
            frame = _mapping(raw, field="observation frame")
            frame_id = frame.get("frame_id")
            trajectory_id = frame.get("trajectory_id")
            absolute_tick = frame.get("absolute_tick")
            state_ref_id = frame.get("state_ref_id")
            if (
                type(frame_id) is not str
                or not frame_id
                or type(trajectory_id) is not str
                or not trajectory_id
                or type(absolute_tick) is not int
                or absolute_tick < 0
            ):
                raise ExpeditionPolicyError("toy observation frame identity/tick is invalid")
            if state_ref_id in state_refs:
                eligible.append(frame)
        eligible.sort(key=lambda row: (str(row["trajectory_id"]), str(row["frame_id"])))
        outer_rng = _mapping(checked["outer_rng"], field="outer_rng")
        cursor = int(outer_rng["cursor"])
        if eligible:
            start, cursor = sha256_counter_prf_index_v1(
                seed=int(outer_rng["seed"]),
                counter=cursor,
                domain=f"{self.policy_id}:frontier:{frontier['frontier_ordinal']}",
                size=len(eligible),
            )
            ordered = eligible[start:] + eligible[:start]
        else:
            ordered = []
        capacity = int(checked["logical_capacity"])
        template = _mapping(self.profile["action_template"], field="action template")
        actions = []
        for frame in ordered[:capacity]:
            state_ref_id = str(frame["state_ref_id"])
            state_ref = state_refs[state_ref_id]
            core = {
                "action_kind": template["action_kind"],
                "bucket": template["bucket"],
                "slot_cost": template["slot_cost"],
                "trajectory_id": frame["trajectory_id"],
                "source_frame_id": frame["frame_id"],
                "frontier_ordinal": frontier["frontier_ordinal"],
                "observation_absolute_tick": frame["absolute_tick"],
                "parent_state_ref_id": state_ref_id,
                "parent_next_absolute_tick": state_ref["next_absolute_tick"],
                "reason_codes": [template["reason_code"]],
            }
            actions.append(
                {
                    "action_id": f"A-{canonical_sha256(core).hex[:24]}",
                    **core,
                }
            )
        return seal_policy_proposal_v1(
            policy_binding=self.binding,
            profile=self.profile,
            decision_input=checked,
            rng_cursor_after=cursor,
            atlas_delta={"upsert_entries": [], "retire_entry_ids": []},
            retained_parent_bank_delta={
                "retain_entries": [],
                "release_state_ref_ids": [],
                "not_retained": [],
            },
            semantic_actions=actions,
            rejected_candidate_summary={
                "ineligible_without_state_ref": len(
                    _sequence(frontier["observation_frames"], field="observation frames")
                )
                - len(eligible),
                "capacity_deferred": max(0, len(eligible) - capacity),
            },
        )


__all__ = (
    "TOY_ROUND_ROBIN_ALGORITHM_SHA256_V1",
    "TOY_ROUND_ROBIN_POLICY_ID_V1",
    "TOY_ROUND_ROBIN_POLICY_VERSION_V1",
    "TOY_ROUND_ROBIN_PROFILE_SCHEMA_VERSION_V1",
    "TOY_ROUND_ROBIN_V1_PROFILE",
    "ToyRoundRobinPolicyV1",
)
