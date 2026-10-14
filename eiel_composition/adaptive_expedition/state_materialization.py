"""Apply proposal-only policy deltas to the bounded public Atlas and Parent Bank.

The policy chooses scientific representatives and retention reasons.  This
module supplies only generic integrity checks and immutable revision building:
an Atlas reference never creates branch authority, while the Parent Bank can
contain only exact or exactly replayable ``StateRefV1`` values.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from emergence_contracts import canonical_json_bytes
from emergence_contracts.adaptive_science_expedition_v1 import (
    DecisionV1,
    EvolutionaryAtlasV1,
    ExpeditionManifestV1,
    ObservationFrameV1,
    RetainedParentBankV1,
    StateRefV1,
)
from emergence_contracts.canonical import canonical_sha256
from emergence_contracts.ids import SchemaVersion
from emergence_contracts.payloads import CanonicalPayloadV1

from .policy_protocol import canonical_plain_value_v1


NICHE_KEY_PAYLOAD_SCHEMA_V1 = SchemaVersion(
    "eiel_adaptive_science_policy_niche_key_v1"
)
DESCRIPTOR_VALUES_PAYLOAD_SCHEMA_V1 = SchemaVersion(
    "eiel_adaptive_science_policy_descriptor_values_v1"
)
ATLAS_POLICY_FACTS_PROJECTION_SCHEMA_V1 = SchemaVersion(
    "eiel_adaptive_science_atlas_policy_facts_projection_v1"
)
OPAQUE_POLICY_FACT_SUBTREE_SCHEMA_V1 = SchemaVersion(
    "eiel_adaptive_science_opaque_policy_fact_subtree_v1"
)

# Atlas entries are repeated in every recoverable Decision revision.  Keep
# their policy-facts projection bounded; the exact ObservationFrame remains
# available through observation_frame_sha256 and the retained package.
ATLAS_POLICY_FACT_VALUE_BUDGET_BYTES = 4_096
ATLAS_POLICY_FACTS_BUDGET_BYTES = 8_192


class ExpeditionStateMaterializationError(ValueError):
    """A policy delta cannot become a faithful bounded state revision."""


@dataclass(frozen=True, slots=True)
class AppliedPolicyStateDeltaV1:
    """Internal result folded into the neighboring durable Decision event."""

    atlas: EvolutionaryAtlasV1
    retained_parent_bank: RetainedParentBankV1
    retention_outcomes: Mapping[str, object]


def _mapping(value: object, *, field: str) -> dict[str, object]:
    plain = canonical_plain_value_v1(value)
    if type(plain) is not dict:
        raise ExpeditionStateMaterializationError(f"{field} must be a mapping")
    return plain


def _sequence(value: object, *, field: str) -> list[object]:
    plain = canonical_plain_value_v1(value)
    if type(plain) is not list:
        raise ExpeditionStateMaterializationError(f"{field} must be a sequence")
    return plain


def _opaque_policy_fact_subtree_v1(value: object) -> dict[str, object]:
    plain = canonical_plain_value_v1(value)
    return {
        "schema_version": str(OPAQUE_POLICY_FACT_SUBTREE_SCHEMA_V1),
        "source_sha256": canonical_sha256(plain).to_dict(),
        "source_bytes": len(canonical_json_bytes(plain)),
    }


def _compact_policy_facing_facts_v1(
    frame: ObservationFrameV1,
) -> CanonicalPayloadV1:
    """Keep a bounded opaque projection while preserving exact source identity.

    The coordinator's policy input is built from the live ObservationFrame,
    not this Atlas field.  Atlas therefore stores compact facts for review and
    replay indexing only.  Large or over-budget subtrees are represented by a
    content digest; the linked exact frame and Parent Bank package retain the
    complete authenticated facts.
    """

    source = canonical_plain_value_v1(frame.policy_facing_facts.to_value())
    if type(source) is not dict:
        raise ExpeditionStateMaterializationError(
            "policy-facing facts payload must contain a mapping"
        )
    source_bytes = len(canonical_json_bytes(source))
    projected: dict[str, object] = {}
    for key in sorted(source):
        value = source[key]
        if len(canonical_json_bytes(value)) <= ATLAS_POLICY_FACT_VALUE_BUDGET_BYTES:
            projected[key] = value
        else:
            projected[key] = _opaque_policy_fact_subtree_v1(value)

    result: dict[str, object] = {
        "source_schema_version": str(frame.policy_facing_facts.schema_version),
        "source_payload_sha256": canonical_sha256(source).to_dict(),
        "source_payload_bytes": source_bytes,
        "facts": projected,
    }
    if len(canonical_json_bytes(result)) > ATLAS_POLICY_FACTS_BUDGET_BYTES:
        # A profile with many independently small fields must still remain
        # bounded.  Preserve the complete source digest and one opaque digest
        # rather than silently dropping fields.
        result["facts"] = {
            "_all_fields": _opaque_policy_fact_subtree_v1(source)
        }
    return CanonicalPayloadV1.from_value(
        ATLAS_POLICY_FACTS_PROJECTION_SCHEMA_V1,
        result,
    )


def _frame_maps_v1(
    frames: Sequence[ObservationFrameV1],
) -> tuple[dict[str, ObservationFrameV1], dict[str, StateRefV1]]:
    by_frame: dict[str, ObservationFrameV1] = {}
    by_state: dict[str, StateRefV1] = {}
    for frame in frames:
        if not isinstance(frame, ObservationFrameV1):
            raise ExpeditionStateMaterializationError(
                "state materialization received a non-ObservationFrameV1"
            )
        frame_id = f"OF-{frame.observation_frame_sha256().hex}"
        state = frame.source_state_ref
        state_id = f"SR-{state.state_ref_sha256().hex}"
        if frame_id in by_frame and by_frame[frame_id] != frame:
            raise ExpeditionStateMaterializationError("observation identity collision")
        if state_id in by_state and by_state[state_id] != state:
            raise ExpeditionStateMaterializationError("StateRef identity collision")
        by_frame[frame_id] = frame
        by_state[state_id] = state
    return by_frame, by_state


def _atlas_entry_v1(
    raw: object,
    *,
    frontier_ordinal: int,
    frames_by_id: Mapping[str, ObservationFrameV1],
) -> dict[str, object]:
    row = _mapping(raw, field="atlas upsert entry")
    entry_id = row.get("entry_id")
    frame_id = row.get("frame_id")
    trajectory_id = row.get("trajectory_id")
    niche_id = row.get("niche_id")
    representative_role = row.get("representative_role")
    if any(
        type(value) is not str or not value
        for value in (
            entry_id,
            frame_id,
            trajectory_id,
            niche_id,
            representative_role,
        )
    ):
        raise ExpeditionStateMaterializationError(
            "atlas upsert lacks its generic identity, niche, or role"
        )
    frame = frames_by_id.get(str(frame_id))
    if frame is None:
        raise ExpeditionStateMaterializationError(
            "atlas upsert refers to an unavailable observation frame"
        )
    if (
        row.get("frontier_ordinal") != frontier_ordinal
        or row.get("observation_absolute_tick") != frame.next_absolute_tick
        or trajectory_id != frame.world_trajectory_id
    ):
        raise ExpeditionStateMaterializationError(
            "atlas upsert differs from its authenticated observation identity"
        )
    descriptor_values = _mapping(
        row.get("descriptor_values"), field="atlas descriptor values"
    )
    parent_entry_ids = row.get("parent_entry_ids", [])
    parent_ids = _sequence(parent_entry_ids, field="atlas parent entry IDs")
    if any(type(value) is not str or not value for value in parent_ids):
        raise ExpeditionStateMaterializationError(
            "atlas parent entry identities must be non-empty text"
        )
    return {
        "atlas_entry_id": entry_id,
        "world_trajectory_id": frame.world_trajectory_id,
        "observation_frame_sha256": frame.observation_frame_sha256().to_dict(),
        # This is an evidence link only.  Atlas itself has no retention or
        # continuation authority, including for observation-only StateRefs.
        "state_ref_sha256": frame.source_state_ref.state_ref_sha256().to_dict(),
        "niche_key": CanonicalPayloadV1.from_value(
            NICHE_KEY_PAYLOAD_SCHEMA_V1, niche_id
        ).to_dict(),
        "descriptor_values": CanonicalPayloadV1.from_value(
            DESCRIPTOR_VALUES_PAYLOAD_SCHEMA_V1, descriptor_values
        ).to_dict(),
        "policy_facing_facts": _compact_policy_facing_facts_v1(frame).to_dict(),
        "representative_roles": [representative_role],
        "parent_entry_ids": sorted(set(parent_ids)),
    }


def _apply_atlas_delta_v1(
    *,
    decision: DecisionV1,
    frames_by_id: Mapping[str, ObservationFrameV1],
    atlas: EvolutionaryAtlasV1,
    delta: Mapping[str, object],
) -> EvolutionaryAtlasV1:
    if decision.atlas_before_sha256 != atlas.atlas_sha256():
        raise ExpeditionStateMaterializationError(
            "Decision Atlas predecessor differs from the supplied revision"
        )
    if set(delta) != {"upsert_entries", "retire_entry_ids"}:
        raise ExpeditionStateMaterializationError("Atlas delta fields must be exact")
    retire_ids = _sequence(delta["retire_entry_ids"], field="Atlas retire IDs")
    if any(type(value) is not str or not value for value in retire_ids) or len(
        retire_ids
    ) != len(set(retire_ids)):
        raise ExpeditionStateMaterializationError("Atlas retire IDs are invalid")
    existing = {
        entry.atlas_entry_id: entry.to_dict() for entry in atlas.entries
    }
    unknown = set(retire_ids) - set(existing)
    if unknown:
        raise ExpeditionStateMaterializationError(
            "Atlas delta retires an unavailable representative"
        )
    for entry_id in retire_ids:
        del existing[str(entry_id)]
    for raw in _sequence(delta["upsert_entries"], field="Atlas upserts"):
        entry = _atlas_entry_v1(
            raw,
            frontier_ordinal=decision.frontier_ordinal,
            frames_by_id=frames_by_id,
        )
        entry_id = str(entry["atlas_entry_id"])
        prior = existing.get(entry_id)
        if prior is not None and prior != entry:
            raise ExpeditionStateMaterializationError(
                "Atlas upsert changes an existing representative identity"
            )
        existing[entry_id] = entry
    entries = [existing[key] for key in sorted(existing)]
    if len(entries) > atlas.maximum_entries:
        raise ExpeditionStateMaterializationError(
            "policy Atlas delta exceeds the bounded Atlas entry budget"
        )
    return EvolutionaryAtlasV1.from_dict(
        {
            "schema_version": "eiel_adaptive_science_evolutionary_atlas_v1",
            "expedition_id": atlas.expedition_id,
            "manifest_sha256": atlas.manifest_sha256.to_dict(),
            "revision": atlas.revision + 1,
            "through_frontier_ordinal": decision.frontier_ordinal,
            "descriptor_set_sha256": atlas.descriptor_set_sha256.to_dict(),
            "maximum_entries": atlas.maximum_entries,
            "entries": entries,
            "previous_atlas_sha256": atlas.atlas_sha256().to_dict(),
        }
    )


def _apply_parent_bank_delta_v1(
    *,
    manifest: ExpeditionManifestV1,
    decision: DecisionV1,
    states_by_id: Mapping[str, StateRefV1],
    bank: RetainedParentBankV1,
    delta: Mapping[str, object],
) -> tuple[RetainedParentBankV1, dict[str, object]]:
    bank.validate_against_manifest(manifest)
    if set(delta) != {
        "retain_entries",
        "release_state_ref_ids",
        "not_retained",
    }:
        raise ExpeditionStateMaterializationError(
            "Retained Parent Bank delta fields must be exact"
        )
    release_ids = _sequence(
        delta["release_state_ref_ids"], field="Parent Bank release IDs"
    )
    if any(type(value) is not str or not value for value in release_ids) or len(
        release_ids
    ) != len(set(release_ids)):
        raise ExpeditionStateMaterializationError("Parent Bank release IDs are invalid")
    existing = {
        f"SR-{entry.state_ref.state_ref_sha256().hex}": entry.to_dict()
        for entry in bank.entries
    }
    if set(release_ids) - set(existing):
        raise ExpeditionStateMaterializationError(
            "Parent Bank delta releases an unavailable exact state"
        )
    for state_ref_id in release_ids:
        del existing[str(state_ref_id)]

    retained_ids: list[str] = []
    for raw in _sequence(delta["retain_entries"], field="Parent Bank retains"):
        row = _mapping(raw, field="Parent Bank retain entry")
        state_ref_id = row.get("state_ref_id")
        survival_reason = row.get("survival_reason")
        if (
            type(state_ref_id) is not str
            or not state_ref_id
            or type(survival_reason) is not str
            or not survival_reason
        ):
            raise ExpeditionStateMaterializationError(
                "retained parent lacks an exact state identity or survival reason"
            )
        if state_ref_id in release_ids:
            raise ExpeditionStateMaterializationError(
                "one Parent Bank delta both retains and releases a state"
            )
        state = states_by_id.get(state_ref_id)
        if state is None:
            raise ExpeditionStateMaterializationError(
                "retained parent refers to an unavailable StateRef"
            )
        if not state.supports_exact_continuation:
            raise ExpeditionStateMaterializationError(
                "observation-only state cannot enter the Retained Parent Bank"
            )
        size_bytes = state.source_artifact_ref.size_bytes
        if size_bytes is None or size_bytes <= 0:
            raise ExpeditionStateMaterializationError(
                "retained parent lacks exact artifact byte accounting"
            )
        if row.get("frontier_ordinal") != decision.frontier_ordinal:
            raise ExpeditionStateMaterializationError(
                "retained parent frontier differs from the Decision"
            )
        new_entry = {
            "state_ref": state.to_dict(),
            "selected_at_frontier_ordinal": decision.frontier_ordinal,
            "reason_codes": [survival_reason],
            "retained_bytes": size_bytes,
        }
        prior = existing.get(state_ref_id)
        if prior is not None and prior != new_entry:
            raise ExpeditionStateMaterializationError(
                "retained StateRef identity changed its survival receipt"
            )
        existing[state_ref_id] = new_entry
        retained_ids.append(state_ref_id)

    entries = [existing[key] for key in sorted(existing)]
    updated = RetainedParentBankV1.from_dict(
        {
            "schema_version": "eiel_adaptive_science_retained_parent_bank_v1",
            "expedition_id": bank.expedition_id,
            "manifest_sha256": bank.manifest_sha256.to_dict(),
            "revision": bank.revision + 1,
            "maximum_entries": bank.maximum_entries,
            "maximum_total_bytes": bank.maximum_total_bytes,
            "current_entry_count": len(entries),
            "current_total_bytes": sum(
                int(entry["retained_bytes"]) for entry in entries
            ),
            "entries": entries,
            "previous_bank_sha256": bank.bank_sha256().to_dict(),
        }
    )
    updated.validate_against_manifest(manifest)
    policy_not_retained = _sequence(
        delta["not_retained"], field="Parent Bank not-retained rows"
    )
    outcomes = {
        "schema_version": "eiel_adaptive_parent_retention_outcomes_v1",
        "retained_state_ref_ids": sorted(set(retained_ids)),
        "released_state_ref_ids": sorted(str(value) for value in release_ids),
        "policy_not_retained": policy_not_retained,
        "observation_only_retention_authority": False,
        "deletion_authority": False,
    }
    # Force a canonical form now; the Decision event will bind these outcomes.
    canonical_sha256(outcomes)
    return updated, outcomes


def apply_policy_state_deltas_v1(
    *,
    manifest: ExpeditionManifestV1,
    decision: DecisionV1,
    observation_frames: Sequence[ObservationFrameV1],
    atlas: EvolutionaryAtlasV1,
    retained_parent_bank: RetainedParentBankV1,
) -> AppliedPolicyStateDeltaV1:
    """Build the next immutable Atlas and Parent Bank revisions."""

    if decision.expedition_id != manifest.expedition_id:
        raise ExpeditionStateMaterializationError(
            "Decision and manifest belong to different Expeditions"
        )
    if decision.manifest_sha256 != manifest.manifest_sha256():
        raise ExpeditionStateMaterializationError(
            "Decision does not bind the supplied manifest"
        )
    if atlas.expedition_id != manifest.expedition_id or (
        atlas.manifest_sha256 != manifest.manifest_sha256()
    ):
        raise ExpeditionStateMaterializationError(
            "Atlas does not bind the supplied manifest"
        )
    if atlas.descriptor_set_sha256 != manifest.observation_descriptor_set_sha256:
        raise ExpeditionStateMaterializationError(
            "Atlas descriptor identity differs from the manifest"
        )
    frames_by_id, states_by_id = _frame_maps_v1(observation_frames)
    proposal = _mapping(decision.policy_proposal.to_value(), field="policy proposal")
    atlas_delta = _mapping(proposal.get("atlas_delta"), field="Atlas delta")
    bank_delta = _mapping(
        proposal.get("retained_parent_bank_delta"), field="Parent Bank delta"
    )
    updated_atlas = _apply_atlas_delta_v1(
        decision=decision,
        frames_by_id=frames_by_id,
        atlas=atlas,
        delta=atlas_delta,
    )
    updated_bank, outcomes = _apply_parent_bank_delta_v1(
        manifest=manifest,
        decision=decision,
        states_by_id=states_by_id,
        bank=retained_parent_bank,
        delta=bank_delta,
    )
    return AppliedPolicyStateDeltaV1(
        atlas=updated_atlas,
        retained_parent_bank=updated_bank,
        retention_outcomes=outcomes,
    )


__all__ = (
    "AppliedPolicyStateDeltaV1",
    "ExpeditionStateMaterializationError",
    "apply_policy_state_deltas_v1",
)
