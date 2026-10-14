"""Generic deterministic decision coordinator for Adaptive Expedition V1.

The coordinator owns binding, replay, quota-independent action sealing, and
durable decision construction.  It deliberately does not interpret any
descriptor, niche, phenotype, novelty, retention reason, or policy bucket.
Policy output remains proposal-only until this module validates it against the
manifest and seals a :class:`DecisionV1` receipt.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

from emergence_contracts.adaptive_science_expedition_v1 import (
    DECISION_SCHEMA_VERSION,
    DecisionV1,
    EvolutionaryAtlasV1,
    ExpeditionManifestV1,
    ExperimentActionKindV1,
    ObservationFrameV1,
    RetainedParentBankV1,
    StateRefV1,
)
from emergence_contracts.canonical import canonical_sha256
from emergence_contracts.ids import ContentDigest, SchemaVersion
from emergence_contracts.payloads import CanonicalPayloadV1

from .policy_protocol import (
    POLICY_PROTOCOL_VERSION_V1,
    ExpeditionPolicyError,
    ExpeditionPolicyV1,
    canonical_plain_value_v1,
    project_policy_input_v1,
    validate_policy_proposal_v1,
)


POLICY_ACTION_PAYLOAD_SCHEMA_V1: Final[SchemaVersion] = SchemaVersion(
    "eiel_adaptive_science_policy_action_v1"
)
POLICY_PROPOSAL_PAYLOAD_SCHEMA_V1: Final[SchemaVersion] = SchemaVersion(
    "eiel_adaptive_science_policy_proposal_v1"
)

_RUN_ACTIONS: Final[frozenset[ExperimentActionKindV1]] = frozenset(
    {
        ExperimentActionKindV1.CONTINUE,
        ExperimentActionKindV1.BRANCH_LOCAL,
        ExperimentActionKindV1.EXPLORE_GLOBAL,
        ExperimentActionKindV1.REPLICATE,
        ExperimentActionKindV1.CAUSAL_PROBE,
        ExperimentActionKindV1.TRANSFER_CHALLENGE,
    }
)


class ExpeditionCoordinatorError(ValueError):
    """Raised when policy input or output cannot be sealed faithfully."""


@dataclass(frozen=True, slots=True)
class PreparedDecisionV1:
    """Internal result: one durable receipt and its replay cursor."""

    decision: DecisionV1
    policy_input_sha256: ContentDigest
    proposal_sha256: ContentDigest
    next_outer_rng_cursor: int


def _plain_mapping(value: object, *, field: str) -> dict[str, object]:
    plain = canonical_plain_value_v1(value)
    if type(plain) is not dict:
        raise ExpeditionCoordinatorError(f"{field} must be a canonical mapping")
    return plain


def _plain_sequence(value: object, *, field: str) -> list[object]:
    plain = canonical_plain_value_v1(value)
    if type(plain) is not list:
        raise ExpeditionCoordinatorError(f"{field} must be a canonical sequence")
    return plain


def _digest(value: object, *, field: str) -> ContentDigest:
    body = _plain_mapping(value, field=field)
    if set(body) != {"algorithm", "hex"}:
        raise ExpeditionCoordinatorError(f"{field} must be a ContentDigest mapping")
    try:
        return ContentDigest(str(body["algorithm"]), str(body["hex"]))
    except (TypeError, ValueError) as error:
        raise ExpeditionCoordinatorError(f"{field} is not a valid digest") from error


def _policy_state_ref_view_v1(state: StateRefV1) -> dict[str, object]:
    state_ref_sha256 = state.state_ref_sha256()
    return {
        "state_ref_id": f"SR-{state_ref_sha256.hex}",
        "state_ref_sha256": state_ref_sha256.to_dict(),
        "world_trajectory_id": state.world_trajectory_id,
        "logical_world_id": state.logical_world_id,
        "next_absolute_tick": state.next_absolute_tick,
        "state_class": state.state_class.value,
        "semantic_world_sha256": state.semantic_world_sha256.to_dict(),
        "control_history_sha256": state.control_history_sha256.to_dict(),
        "source_object_sha256": state.source_object_sha256.to_dict(),
        "exact_state_sha256": (
            None if state.exact_state_sha256 is None else state.exact_state_sha256.to_dict()
        ),
    }


def _policy_frame_view_v1(frame: ObservationFrameV1) -> dict[str, object]:
    frame_sha256 = frame.observation_frame_sha256()
    state_ref_sha256 = frame.source_state_ref.state_ref_sha256()
    facts = frame.policy_facing_facts.to_value()
    if type(facts) is not dict:
        raise ExpeditionCoordinatorError(
            "policy_facing_facts payload must contain a mapping"
        )
    return {
        "frame_id": f"OF-{frame_sha256.hex}",
        "frame_sha256": frame_sha256.to_dict(),
        "trajectory_id": frame.world_trajectory_id,
        "logical_world_id": frame.logical_world_id,
        # The policy port historically names a completed observation cursor
        # ``absolute_tick``.  It is the frame's per-trajectory next tick, never
        # an outer-frontier clock.
        "absolute_tick": frame.next_absolute_tick,
        "state_ref_id": f"SR-{state_ref_sha256.hex}",
        "facts": facts,
        "descriptor_values": frame.descriptor_values.to_value(),
        "calibration": frame.calibration.to_value(),
        "control_readback": frame.control_readback.to_value(),
    }


def _policy_niche_id_v1(value: Mapping[str, object]) -> str:
    payload = value.get("payload")
    if type(payload) is str and payload:
        return payload
    # Policies own niche semantics.  The coordinator only gives an opaque,
    # stable identity to Atlas keys that are not already plain identifiers.
    return "NK-" + canonical_sha256(value).hex


def _validate_manifest_policy_v1(
    manifest: ExpeditionManifestV1, policy: ExpeditionPolicyV1
) -> None:
    expected = canonical_plain_value_v1(manifest.policy_binding.to_dict())
    actual = canonical_plain_value_v1(policy.binding)
    if actual != expected:
        raise ExpeditionCoordinatorError(
            "policy implementation/profile differs from the frozen manifest binding"
        )


def _validate_frames_v1(
    manifest: ExpeditionManifestV1,
    frames: Sequence[ObservationFrameV1],
) -> None:
    frame_ids: set[str] = set()
    for frame in frames:
        if not isinstance(frame, ObservationFrameV1):
            raise ExpeditionCoordinatorError(
                "observation frontier contains a non-ObservationFrameV1 value"
            )
        if frame.expedition_id != manifest.expedition_id:
            raise ExpeditionCoordinatorError(
                "observation frame belongs to a different expedition"
            )
        if frame.observation_profile_id != manifest.observation_profile_id:
            raise ExpeditionCoordinatorError(
                "observation profile differs from the frozen manifest"
            )
        if (
            frame.observation_descriptor_set_sha256
            != manifest.observation_descriptor_set_sha256
        ):
            raise ExpeditionCoordinatorError(
                "observation descriptor set differs from the frozen manifest"
            )
        identity = str(frame.observation_frame_sha256())
        if identity in frame_ids:
            raise ExpeditionCoordinatorError("observation frame was duplicated")
        frame_ids.add(identity)


def _action_row_v1(
    raw_action: object,
    *,
    manifest: ExpeditionManifestV1,
    states_by_id: Mapping[str, StateRefV1],
) -> dict[str, object]:
    action = _plain_mapping(raw_action, field="policy semantic action")
    action_id = action.get("action_id")
    bucket = action.get("bucket")
    reason_codes = action.get("reason_codes")
    if type(action_id) is not str or not action_id:
        raise ExpeditionCoordinatorError("policy action_id must be non-empty text")
    if type(bucket) is not str or not bucket:
        raise ExpeditionCoordinatorError("policy action bucket must be non-empty text")
    reasons = _plain_sequence(reason_codes, field="policy action reason_codes")
    if not reasons or any(type(item) is not str or not item for item in reasons):
        raise ExpeditionCoordinatorError("policy action reason codes must be text")
    try:
        action_kind = ExperimentActionKindV1(action.get("action_kind"))
    except (TypeError, ValueError) as error:
        raise ExpeditionCoordinatorError("policy emitted an unknown action kind") from error

    parent_state_ref_sha256: ContentDigest | None = None
    parent_tick: int | None = None
    parent_id = action.get("parent_state_ref_id")
    if parent_id is not None:
        if type(parent_id) is not str or parent_id not in states_by_id:
            raise ExpeditionCoordinatorError(
                "policy action refers to an unavailable parent StateRef"
            )
        parent = states_by_id[parent_id]
        parent_state_ref_sha256 = parent.state_ref_sha256()
        parent_tick = parent.next_absolute_tick

    if action_kind is ExperimentActionKindV1.EXPLORE_GLOBAL:
        effective_tick = action.get("child_start_next_tick")
        if effective_tick != 0 or parent_id is not None:
            raise ExpeditionCoordinatorError(
                "global founder exploration must start at tick zero without a parent"
            )
    elif parent_tick is not None:
        effective_tick = parent_tick
        declared_parent_tick = action.get("parent_next_absolute_tick")
        if declared_parent_tick is not None and declared_parent_tick != parent_tick:
            raise ExpeditionCoordinatorError(
                "policy action parent tick differs from the exact StateRef"
            )
    else:
        # Observation-only actions need no restorable parent.  Their effective
        # cursor is the exact per-trajectory observation cursor.
        effective_tick = action.get("observation_absolute_tick", 0)
    if type(effective_tick) is not int or effective_tick < 0:
        raise ExpeditionCoordinatorError("policy action effective tick is invalid")

    action_payload = CanonicalPayloadV1.from_value(
        POLICY_ACTION_PAYLOAD_SCHEMA_V1, action
    )
    return {
        "action_id": action_id,
        "action_kind": action_kind.value,
        "quota_bucket": bucket,
        "parent_state_ref_sha256": (
            None
            if parent_state_ref_sha256 is None
            else parent_state_ref_sha256.to_dict()
        ),
        "effective_next_absolute_tick": effective_tick,
        "next_epoch_ticks": (
            manifest.epoch_ticks if action_kind in _RUN_ACTIONS else None
        ),
        "reason_codes": sorted(set(reasons)),
        "action_payload": action_payload.to_dict(),
    }


def prepare_decision_v1(
    *,
    manifest: ExpeditionManifestV1,
    policy: ExpeditionPolicyV1,
    frontier_id: str,
    frontier_ordinal: int,
    observation_frames: Sequence[ObservationFrameV1],
    bounded_history: Sequence[Mapping[str, object]],
    atlas: EvolutionaryAtlasV1,
    retained_parent_bank: RetainedParentBankV1,
    logical_capacity: int,
    outer_rng_cursor: int,
    retained_parent_divergence_signatures: Mapping[str, object] | None = None,
) -> PreparedDecisionV1:
    """Run, replay-check, and seal one policy decision without dispatching it."""

    if type(frontier_id) is not str or not frontier_id:
        raise ExpeditionCoordinatorError("frontier_id must be non-empty text")
    if type(frontier_ordinal) is not int or frontier_ordinal < 0:
        raise ExpeditionCoordinatorError("frontier_ordinal must be non-negative")
    if type(logical_capacity) is not int or logical_capacity < 0:
        raise ExpeditionCoordinatorError("logical_capacity must be non-negative")
    if logical_capacity > manifest.maximum_actions_per_frontier:
        raise ExpeditionCoordinatorError(
            "logical_capacity exceeds the immutable manifest frontier budget"
        )
    if type(outer_rng_cursor) is not int or not 0 <= outer_rng_cursor < (1 << 64):
        raise ExpeditionCoordinatorError("outer RNG cursor must be a uint64")
    _validate_manifest_policy_v1(manifest, policy)
    _validate_frames_v1(manifest, observation_frames)
    retained_parent_bank.validate_against_manifest(manifest)
    if atlas.expedition_id != manifest.expedition_id:
        raise ExpeditionCoordinatorError("Atlas belongs to a different expedition")
    if atlas.manifest_sha256 != manifest.manifest_sha256():
        raise ExpeditionCoordinatorError("Atlas differs from the frozen manifest")
    if atlas.descriptor_set_sha256 != manifest.observation_descriptor_set_sha256:
        raise ExpeditionCoordinatorError("Atlas descriptor set differs from the manifest")
    if atlas.maximum_entries != manifest.maximum_atlas_entries:
        raise ExpeditionCoordinatorError(
            "Atlas capacity differs from the immutable manifest budget"
        )

    ordered_frames = tuple(
        sorted(
            observation_frames,
            key=lambda frame: (
                frame.world_trajectory_id,
                frame.logical_world_id,
                frame.observation_frame_sha256().hex,
            ),
        )
    )
    states_by_id: dict[str, StateRefV1] = {}

    def register_state(state: StateRefV1) -> None:
        state_id = f"SR-{state.state_ref_sha256().hex}"
        prior = states_by_id.get(state_id)
        if prior is not None and prior != state:
            raise ExpeditionCoordinatorError("StateRef digest collision")
        states_by_id[state_id] = state

    for frame in ordered_frames:
        register_state(frame.source_state_ref)
    # Retained exact parents remain policy-visible after their observation
    # frontier has passed.  This is the generic historical-action seam; the
    # coordinator does not interpret why the policy retained or selected one.
    for entry in retained_parent_bank.entries:
        register_state(entry.state_ref)

    policy_frames = [_policy_frame_view_v1(frame) for frame in ordered_frames]
    policy_states = [
        _policy_state_ref_view_v1(states_by_id[state_id])
        for state_id in sorted(states_by_id)
    ]
    history = _plain_sequence(bounded_history, field="bounded_history")
    atlas_view = {
        "entries": [
            {
                "entry_id": row["atlas_entry_id"],
                "trajectory_id": row["world_trajectory_id"],
                # Keep the generic StateRef identity available to the bound
                # policy.  The coordinator does not interpret the Atlas
                # niche; this link lets a policy reconstruct its own
                # cross-frontier retention identity from authenticated state.
                "state_ref_id": (
                    None
                    if row["state_ref_sha256"] is None
                    else f"SR-{_digest(row['state_ref_sha256'], field='Atlas state_ref_sha256').hex}"
                ),
                "niche_id": _policy_niche_id_v1(row["niche_key"]),
                "descriptor_values": row["descriptor_values"]["payload"],
                "representative_role": (
                    row["representative_roles"][0]
                    if row["representative_roles"]
                    else "UNSPECIFIED"
                ),
            }
            for row in atlas.to_dict()["entries"]
        ]
    }
    bank_entry_ids = {
        f"SR-{entry.state_ref.state_ref_sha256().hex}"
        for entry in retained_parent_bank.entries
    }
    profile_authority = canonical_plain_value_v1(policy.profile).get(
        "policy_input_authority", {}
    )
    authority_paths = (
        profile_authority.get("allowed_paths", [])
        if isinstance(profile_authority, Mapping)
        else []
    )
    policy_uses_divergence_proofs = (
        "/retained_parent_bank/entries/*/divergence_signature" in authority_paths
    )
    if retained_parent_divergence_signatures is None:
        if bank_entry_ids and policy_uses_divergence_proofs:
            raise ExpeditionCoordinatorError(
                "non-empty Retained Parent Bank requires durable policy divergence proofs"
            )
        divergence_signatures: dict[str, object] = {}
    else:
        divergence_signatures = {
            str(state_ref_id): canonical_plain_value_v1(signature)
            for state_ref_id, signature in retained_parent_divergence_signatures.items()
        }
        if set(divergence_signatures) != bank_entry_ids and policy_uses_divergence_proofs:
            raise ExpeditionCoordinatorError(
                "Retained Parent Bank divergence proofs do not match its entries"
            )
    bank_view = {
        "maximum_entries": retained_parent_bank.maximum_entries,
        "entries": [
            {
                "state_ref_id": state_ref_id,
                "reason_codes": list(entry.reason_codes),
                "divergence_signature": divergence_signatures[state_ref_id]
                if divergence_signatures
                else [entry.state_ref.state_ref_sha256().to_dict()],
            }
            for entry in retained_parent_bank.entries
            for state_ref_id in [f"SR-{entry.state_ref.state_ref_sha256().hex}"]
        ]
    }
    decision_input: dict[str, object] = {
        "protocol_version": POLICY_PROTOCOL_VERSION_V1,
        "expedition_id": manifest.expedition_id,
        "manifest_sha256": manifest.manifest_sha256().to_dict(),
        "policy_binding": manifest.policy_binding.to_dict(),
        "decision_frontier": {
            "frontier_id": frontier_id,
            "frontier_ordinal": frontier_ordinal,
            "observation_frames": policy_frames,
            "bounded_history": history,
        },
        "atlas": atlas_view,
        "retained_parent_bank": bank_view,
        "available_state_refs": policy_states,
        "logical_capacity": logical_capacity,
        "outer_rng": {
            "seed": manifest.outer_search_seed,
            "cursor": outer_rng_cursor,
        },
        "authorized_search_space": manifest.search_space.to_value(),
    }

    try:
        policy_input = project_policy_input_v1(
            decision_input, profile=policy.profile
        )
        first = policy.decide(policy_input)
        second = policy.decide(policy_input)
        checked = validate_policy_proposal_v1(
            first,
            profile=policy.profile,
            decision_input=policy_input,
            expected_binding=policy.binding,
        )
    except ExpeditionPolicyError as error:
        raise ExpeditionCoordinatorError(f"policy decision failed: {error}") from error
    if canonical_plain_value_v1(first) != canonical_plain_value_v1(second):
        raise ExpeditionCoordinatorError(
            "policy replay from identical canonical input was not byte-identical"
        )

    proposal = _plain_mapping(checked, field="policy proposal")
    semantic_actions = _plain_sequence(
        proposal.get("semantic_actions"), field="policy semantic_actions"
    )
    action_rows = [
        _action_row_v1(raw, manifest=manifest, states_by_id=states_by_id)
        for raw in semantic_actions
    ]
    action_ids = [str(row["action_id"]) for row in action_rows]
    if len(action_ids) != len(set(action_ids)):
        raise ExpeditionCoordinatorError("policy emitted duplicate action identities")

    proposal_payload = CanonicalPayloadV1.from_value(
        POLICY_PROPOSAL_PAYLOAD_SCHEMA_V1, proposal
    )
    rng_before = {
        "seed": manifest.outer_search_seed,
        "cursor": outer_rng_cursor,
    }
    proposal_rng = _plain_mapping(proposal.get("outer_rng"), field="proposal outer_rng")
    cursor_after = proposal_rng.get("cursor_after")
    if type(cursor_after) is not int or not 0 <= cursor_after < (1 << 64):
        raise ExpeditionCoordinatorError("policy returned an invalid RNG cursor")
    rng_after = {"seed": manifest.outer_search_seed, "cursor": cursor_after}
    observation_frontier = {
        "frontier_id": frontier_id,
        "frontier_ordinal": frontier_ordinal,
        "observation_frame_sha256s": [
            frame.observation_frame_sha256().to_dict() for frame in ordered_frames
        ],
        "bounded_history": history,
    }
    binding = manifest.policy_binding.to_dict()
    decision = DecisionV1.from_dict(
        {
            "schema_version": str(DECISION_SCHEMA_VERSION),
            "expedition_id": manifest.expedition_id,
            "frontier_ordinal": frontier_ordinal,
            "manifest_sha256": manifest.manifest_sha256().to_dict(),
            "observation_frontier_sha256": canonical_sha256(
                observation_frontier
            ).to_dict(),
            "atlas_before_sha256": atlas.atlas_sha256().to_dict(),
            "policy_implementation_sha256": binding["implementation_sha256"],
            "policy_profile_sha256": binding["profile_sha256"],
            "outer_rng_before_sha256": canonical_sha256(rng_before).to_dict(),
            "outer_rng_after_sha256": canonical_sha256(rng_after).to_dict(),
            "policy_proposal": proposal_payload.to_dict(),
            "policy_proposal_sha256": proposal_payload.content_digest().to_dict(),
            "action_rows": action_rows,
            "integrity_warnings": [],
        }
    )
    return PreparedDecisionV1(
        decision=decision,
        policy_input_sha256=canonical_sha256(canonical_plain_value_v1(policy_input)),
        proposal_sha256=_digest(
            proposal["proposal_sha256"], field="proposal_sha256"
        ),
        next_outer_rng_cursor=cursor_after,
    )


__all__ = (
    "ExpeditionCoordinatorError",
    "POLICY_ACTION_PAYLOAD_SCHEMA_V1",
    "POLICY_PROPOSAL_PAYLOAD_SCHEMA_V1",
    "PreparedDecisionV1",
    "prepare_decision_v1",
)
