"""Explicit recursive-genealogy projections for Adaptive Expeditions.

The canonical StateRef and Parent Bank contracts remain the authorities for
exact state and retention.  This module only binds the observation/result
association and derives a compact experiment DAG from the existing append-only
ledger.  It deliberately contains no provider, scheduler, database, or
artifact-lifecycle authority.

In particular, a state reference's semantic identity does not include its
filesystem locator.  The existing ``StateRefV1`` digest still authenticates
the complete public reference (including its locator), while the separate
semantic digest below makes the non-semantic nature of packing and placement
explicit for genealogy comparisons.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Final

from emergence_contracts import ContentDigest, canonical_sha256
from emergence_contracts.adaptive_science_expedition_v1 import (
    ExpeditionManifestV1,
    ObservationFrameV1,
    RetainedParentBankV1,
    StateClassV1,
    StateRefV1,
)


GENEALOGY_ASSOCIATION_SCHEMA_V1: Final[str] = (
    "eiel_adaptive_state_ref_association_v1"
)
GENEALOGY_RETENTION_BINDING_SCHEMA_V1: Final[str] = (
    "eiel_adaptive_parent_retention_binding_v1"
)
GENEALOGY_DAG_SCHEMA_V1: Final[str] = "eiel_adaptive_experiment_dag_v1"


class GenealogyIntegrityError(ValueError):
    """A genealogy projection cannot be authenticated without guessing."""


class BranchabilityClassV1(str, Enum):
    """The explicit continuation class of one authenticated observation."""

    OBSERVATION_ONLY = "OBSERVATION_ONLY"
    EXACT_BRANCHABLE = "EXACT_BRANCHABLE"
    EXACT_REPLAYABLE = "EXACT_REPLAYABLE"
    UNAVAILABLE = "UNAVAILABLE"
    INVALID = "INVALID"


def _digest(value: object, *, field: str) -> ContentDigest:
    if isinstance(value, ContentDigest):
        return value
    if not isinstance(value, Mapping) or set(value) != {"algorithm", "hex"}:
        raise GenealogyIntegrityError(f"{field} is not a digest")
    try:
        return ContentDigest(str(value["algorithm"]), str(value["hex"]))
    except (TypeError, ValueError) as error:
        raise GenealogyIntegrityError(f"{field} is invalid") from error


def _optional_digest(value: object, *, field: str) -> ContentDigest | None:
    return None if value is None else _digest(value, field=field)


def _text(value: object, *, field: str, allow_none: bool = False) -> str | None:
    if value is None and allow_none:
        return None
    if type(value) is not str or not value:
        raise GenealogyIntegrityError(f"{field} must be non-empty text")
    return value


def _nonnegative_int(value: object, *, field: str) -> int:
    if type(value) is not int or value < 0:
        raise GenealogyIntegrityError(f"{field} must be a non-negative integer")
    return value


def _canonical_mapping(value: object, *, field: str) -> dict[str, object]:
    if not isinstance(value, Mapping) or any(type(key) is not str for key in value):
        raise GenealogyIntegrityError(f"{field} must be a mapping")
    return dict(value)


def branchability_class_for_state_ref_v1(
    state_ref: StateRefV1 | None,
) -> BranchabilityClassV1:
    """Map the canonical StateRef class without inferring from observations."""

    if state_ref is None:
        return BranchabilityClassV1.UNAVAILABLE
    if state_ref.state_class is StateClassV1.OBSERVATION_ONLY:
        return BranchabilityClassV1.OBSERVATION_ONLY
    if state_ref.state_class is StateClassV1.BRANCHABLE_EXACT:
        return BranchabilityClassV1.EXACT_BRANCHABLE
    if state_ref.state_class is StateClassV1.REPLAY_DERIVABLE:
        return BranchabilityClassV1.EXACT_REPLAYABLE
    if state_ref.state_class is StateClassV1.TERMINAL_ONLY:
        return BranchabilityClassV1.UNAVAILABLE
    return BranchabilityClassV1.INVALID


def classify_state_ref_value_v1(value: object) -> BranchabilityClassV1:
    """Classify malformed input as INVALID rather than granting authority."""

    if value is None:
        return BranchabilityClassV1.UNAVAILABLE
    if isinstance(value, StateRefV1):
        return branchability_class_for_state_ref_v1(value)
    try:
        return branchability_class_for_state_ref_v1(StateRefV1.from_dict(value))
    except (KeyError, TypeError, ValueError):
        return BranchabilityClassV1.INVALID


def semantic_state_identity_sha256_v1(state_ref: StateRefV1) -> ContentDigest:
    """Hash logical state identity while excluding physical placement.

    ``source_artifact_ref.locator`` and ``artifact_id`` are intentionally not
    included.  Content, schema, and byte accounting remain bound, as do every
    exact continuation digest.  Thus relocating the same authenticated state
    package or changing B/slot metadata outside this value cannot change the
    semantic identity.
    """

    if not isinstance(state_ref, StateRefV1):
        raise GenealogyIntegrityError("semantic state identity requires StateRefV1")
    artifact = state_ref.source_artifact_ref
    body = {
        "schema_version": "eiel_adaptive_semantic_state_identity_v1",
        "expedition_id": state_ref.expedition_id,
        "world_trajectory_id": state_ref.world_trajectory_id,
        "logical_world_id": state_ref.logical_world_id,
        "next_absolute_tick": state_ref.next_absolute_tick,
        "state_class": state_ref.state_class.value,
        "semantic_world_sha256": state_ref.semantic_world_sha256.to_dict(),
        "control_history_sha256": state_ref.control_history_sha256.to_dict(),
        "source_object_sha256": state_ref.source_object_sha256.to_dict(),
        "source_artifact_content_sha256": artifact.content_digest.to_dict(),
        "source_artifact_kind": artifact.artifact_kind,
        "source_artifact_schema_version": str(artifact.artifact_schema_version),
        "source_artifact_size_bytes": artifact.size_bytes,
        "exact_state_sha256": (
            None
            if state_ref.exact_state_sha256 is None
            else state_ref.exact_state_sha256.to_dict()
        ),
        "checkpoint_occurrence_sha256": (
            None
            if state_ref.checkpoint_occurrence_sha256 is None
            else state_ref.checkpoint_occurrence_sha256.to_dict()
        ),
        "rng_state_sha256": (
            None
            if state_ref.rng_state_sha256 is None
            else state_ref.rng_state_sha256.to_dict()
        ),
        "event_prefix_sha256": (
            None
            if state_ref.event_prefix_sha256 is None
            else state_ref.event_prefix_sha256.to_dict()
        ),
        "parent_occurrence_sha256": (
            None
            if state_ref.parent_occurrence_sha256 is None
            else state_ref.parent_occurrence_sha256.to_dict()
        ),
        "replay_recipe_sha256": (
            None
            if state_ref.replay_recipe_sha256 is None
            else state_ref.replay_recipe_sha256.to_dict()
        ),
    }
    return canonical_sha256(body)


def _child_occurrence_identity_v1(
    *,
    child_run_id: str,
    child_request_sha256: ContentDigest,
    source_result_sha256: ContentDigest,
    logical_world_id: int,
    state_ref_sha256: ContentDigest,
) -> str:
    return "occurrence-" + canonical_sha256(
        {
            "schema_version": "eiel_adaptive_child_occurrence_identity_v1",
            "child_run_id": child_run_id,
            "child_request_sha256": child_request_sha256.to_dict(),
            "source_result_sha256": source_result_sha256.to_dict(),
            "logical_world_id": logical_world_id,
            "state_ref_sha256": state_ref_sha256.to_dict(),
        }
    ).hex[:40]


def _resolver_identity_v1(
    state_ref: StateRefV1,
    branchability: BranchabilityClassV1,
) -> ContentDigest:
    return canonical_sha256(
        {
            "schema_version": "eiel_adaptive_state_ref_resolver_identity_v1",
            "resolver_id": (
                "eiel_workhorse_selected_world_continuation_resolver_v1"
                if branchability
                in {
                    BranchabilityClassV1.EXACT_BRANCHABLE,
                    BranchabilityClassV1.EXACT_REPLAYABLE,
                }
                else "observation_only_no_continuation_resolver_v1"
            ),
            "state_semantic_identity_sha256": semantic_state_identity_sha256_v1(
                state_ref
            ).to_dict(),
            "state_class": state_ref.state_class.value,
        }
    )


def _materialization_identity_v1(state_ref: StateRefV1) -> ContentDigest:
    return canonical_sha256(
        {
            "schema_version": "eiel_adaptive_state_ref_materialization_identity_v1",
            "materialization_id": (
                "eiel_workhorse_selected_world_state_materialization_v1"
            ),
            "source_object_sha256": state_ref.source_object_sha256.to_dict(),
            "exact_state_sha256": (
                None
                if state_ref.exact_state_sha256 is None
                else state_ref.exact_state_sha256.to_dict()
            ),
            "checkpoint_occurrence_sha256": (
                None
                if state_ref.checkpoint_occurrence_sha256 is None
                else state_ref.checkpoint_occurrence_sha256.to_dict()
            ),
        }
    )


@dataclass(frozen=True, slots=True)
class StateRefAssociationV1:
    """One content-bound observation/result to StateRef association."""

    expedition_id: str
    world_trajectory_id: str
    logical_world_id: int
    child_run_id: str | None
    child_occurrence_id: str | None
    child_request_sha256: ContentDigest | None
    source_result_sha256: ContentDigest
    observation_frame_sha256: ContentDigest
    state_ref_sha256: ContentDigest
    branchability_class: BranchabilityClassV1
    next_absolute_tick: int
    semantic_state_identity_sha256: ContentDigest
    semantic_world_sha256: ContentDigest
    control_history_sha256: ContentDigest
    source_object_sha256: ContentDigest
    exact_state_sha256: ContentDigest | None
    checkpoint_occurrence_sha256: ContentDigest | None
    rng_state_sha256: ContentDigest | None
    event_prefix_sha256: ContentDigest | None
    parent_occurrence_sha256: ContentDigest | None
    replay_recipe_sha256: ContentDigest | None
    resolver_identity_sha256: ContentDigest
    materialization_identity_sha256: ContentDigest
    source_frontier_ordinal: int
    lifecycle_hold_state: str
    parent_state_ref_sha256: ContentDigest | None = None

    def __post_init__(self) -> None:
        _text(self.expedition_id, field="expedition_id")
        _text(self.world_trajectory_id, field="world_trajectory_id")
        _nonnegative_int(self.logical_world_id, field="logical_world_id")
        _nonnegative_int(self.next_absolute_tick, field="next_absolute_tick")
        _nonnegative_int(self.source_frontier_ordinal, field="source_frontier_ordinal")
        _text(self.lifecycle_hold_state, field="lifecycle_hold_state")
        if not isinstance(self.branchability_class, BranchabilityClassV1):
            raise GenealogyIntegrityError("branchability_class is invalid")
        for name in (
            "source_result_sha256",
            "observation_frame_sha256",
            "state_ref_sha256",
            "semantic_state_identity_sha256",
            "semantic_world_sha256",
            "control_history_sha256",
            "source_object_sha256",
            "resolver_identity_sha256",
            "materialization_identity_sha256",
        ):
            if not isinstance(getattr(self, name), ContentDigest):
                raise GenealogyIntegrityError(f"{name} is invalid")
        for name in (
            "child_request_sha256",
            "exact_state_sha256",
            "checkpoint_occurrence_sha256",
            "rng_state_sha256",
            "event_prefix_sha256",
            "parent_occurrence_sha256",
            "replay_recipe_sha256",
            "parent_state_ref_sha256",
        ):
            value = getattr(self, name)
            if value is not None and not isinstance(value, ContentDigest):
                raise GenealogyIntegrityError(f"{name} is invalid")
        if self.child_run_id is None:
            if self.child_occurrence_id is not None or self.child_request_sha256 is not None:
                raise GenealogyIntegrityError(
                    "child occurrence/request cannot exist without child run identity"
                )
        elif self.child_occurrence_id is None or self.child_request_sha256 is None:
            raise GenealogyIntegrityError(
                "child association requires run, occurrence, and request identities"
            )
        exact_fields = (
            self.exact_state_sha256,
            self.checkpoint_occurrence_sha256,
            self.rng_state_sha256,
            self.event_prefix_sha256,
            self.parent_occurrence_sha256,
        )
        if self.branchability_class in {
            BranchabilityClassV1.EXACT_BRANCHABLE,
            BranchabilityClassV1.EXACT_REPLAYABLE,
        } and any(value is None for value in exact_fields):
            raise GenealogyIntegrityError(
                "exact StateRef association lacks exact continuation evidence"
            )
        if self.branchability_class is BranchabilityClassV1.EXACT_BRANCHABLE and (
            self.replay_recipe_sha256 is not None
        ):
            raise GenealogyIntegrityError(
                "direct exact StateRef association carries a replay recipe"
            )
        if self.branchability_class is BranchabilityClassV1.EXACT_REPLAYABLE and (
            self.replay_recipe_sha256 is None
        ):
            raise GenealogyIntegrityError(
                "replay-derived StateRef association lacks its replay recipe"
            )
        if self.branchability_class in {
            BranchabilityClassV1.OBSERVATION_ONLY,
            BranchabilityClassV1.UNAVAILABLE,
        } and any(
            value is not None for value in (*exact_fields, self.replay_recipe_sha256)
        ):
            raise GenealogyIntegrityError(
                "non-branchable StateRef association carries exact authority"
            )
        if self.branchability_class is BranchabilityClassV1.INVALID:
            raise GenealogyIntegrityError("INVALID cannot be persisted as an association")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": GENEALOGY_ASSOCIATION_SCHEMA_V1,
            "expedition_id": self.expedition_id,
            "world_trajectory_id": self.world_trajectory_id,
            "logical_world_id": self.logical_world_id,
            "child_run_id": self.child_run_id,
            "child_occurrence_id": self.child_occurrence_id,
            "child_request_sha256": (
                None
                if self.child_request_sha256 is None
                else self.child_request_sha256.to_dict()
            ),
            "source_result_sha256": self.source_result_sha256.to_dict(),
            "observation_frame_sha256": self.observation_frame_sha256.to_dict(),
            "state_ref_sha256": self.state_ref_sha256.to_dict(),
            "branchability_class": self.branchability_class.value,
            "next_absolute_tick": self.next_absolute_tick,
            "semantic_state_identity_sha256": self.semantic_state_identity_sha256.to_dict(),
            "semantic_world_sha256": self.semantic_world_sha256.to_dict(),
            "control_history_sha256": self.control_history_sha256.to_dict(),
            "source_object_sha256": self.source_object_sha256.to_dict(),
            "exact_state_sha256": (
                None if self.exact_state_sha256 is None else self.exact_state_sha256.to_dict()
            ),
            "checkpoint_occurrence_sha256": (
                None
                if self.checkpoint_occurrence_sha256 is None
                else self.checkpoint_occurrence_sha256.to_dict()
            ),
            "rng_state_sha256": (
                None if self.rng_state_sha256 is None else self.rng_state_sha256.to_dict()
            ),
            "event_prefix_sha256": (
                None if self.event_prefix_sha256 is None else self.event_prefix_sha256.to_dict()
            ),
            "parent_occurrence_sha256": (
                None
                if self.parent_occurrence_sha256 is None
                else self.parent_occurrence_sha256.to_dict()
            ),
            "replay_recipe_sha256": (
                None
                if self.replay_recipe_sha256 is None
                else self.replay_recipe_sha256.to_dict()
            ),
            "resolver_identity_sha256": self.resolver_identity_sha256.to_dict(),
            "materialization_identity_sha256": self.materialization_identity_sha256.to_dict(),
            "source_frontier_ordinal": self.source_frontier_ordinal,
            "lifecycle_hold_state": self.lifecycle_hold_state,
            "parent_state_ref_sha256": (
                None
                if self.parent_state_ref_sha256 is None
                else self.parent_state_ref_sha256.to_dict()
            ),
        }

    def association_sha256(self) -> ContentDigest:
        return canonical_sha256(self.to_dict())

    @classmethod
    def from_dict(cls, value: object) -> "StateRefAssociationV1":
        body = _canonical_mapping(value, field="state_ref_association")
        expected = {
            "schema_version",
            "expedition_id",
            "world_trajectory_id",
            "logical_world_id",
            "child_run_id",
            "child_occurrence_id",
            "child_request_sha256",
            "source_result_sha256",
            "observation_frame_sha256",
            "state_ref_sha256",
            "branchability_class",
            "next_absolute_tick",
            "semantic_state_identity_sha256",
            "semantic_world_sha256",
            "control_history_sha256",
            "source_object_sha256",
            "exact_state_sha256",
            "checkpoint_occurrence_sha256",
            "rng_state_sha256",
            "event_prefix_sha256",
            "parent_occurrence_sha256",
            "replay_recipe_sha256",
            "resolver_identity_sha256",
            "materialization_identity_sha256",
            "source_frontier_ordinal",
            "lifecycle_hold_state",
            "parent_state_ref_sha256",
        }
        if set(body) != expected or body.get("schema_version") != GENEALOGY_ASSOCIATION_SCHEMA_V1:
            raise GenealogyIntegrityError("state_ref_association fields are inexact")
        return cls(
            expedition_id=_text(body["expedition_id"], field="expedition_id"),  # type: ignore[arg-type]
            world_trajectory_id=_text(body["world_trajectory_id"], field="world_trajectory_id"),  # type: ignore[arg-type]
            logical_world_id=_nonnegative_int(body["logical_world_id"], field="logical_world_id"),
            child_run_id=_text(body["child_run_id"], field="child_run_id", allow_none=True),
            child_occurrence_id=_text(body["child_occurrence_id"], field="child_occurrence_id", allow_none=True),
            child_request_sha256=_optional_digest(body["child_request_sha256"], field="child_request_sha256"),
            source_result_sha256=_digest(body["source_result_sha256"], field="source_result_sha256"),
            observation_frame_sha256=_digest(body["observation_frame_sha256"], field="observation_frame_sha256"),
            state_ref_sha256=_digest(body["state_ref_sha256"], field="state_ref_sha256"),
            branchability_class=BranchabilityClassV1(body["branchability_class"]),  # type: ignore[arg-type]
            next_absolute_tick=_nonnegative_int(body["next_absolute_tick"], field="next_absolute_tick"),
            semantic_state_identity_sha256=_digest(body["semantic_state_identity_sha256"], field="semantic_state_identity_sha256"),
            semantic_world_sha256=_digest(body["semantic_world_sha256"], field="semantic_world_sha256"),
            control_history_sha256=_digest(body["control_history_sha256"], field="control_history_sha256"),
            source_object_sha256=_digest(body["source_object_sha256"], field="source_object_sha256"),
            exact_state_sha256=_optional_digest(body["exact_state_sha256"], field="exact_state_sha256"),
            checkpoint_occurrence_sha256=_optional_digest(body["checkpoint_occurrence_sha256"], field="checkpoint_occurrence_sha256"),
            rng_state_sha256=_optional_digest(body["rng_state_sha256"], field="rng_state_sha256"),
            event_prefix_sha256=_optional_digest(body["event_prefix_sha256"], field="event_prefix_sha256"),
            parent_occurrence_sha256=_optional_digest(body["parent_occurrence_sha256"], field="parent_occurrence_sha256"),
            replay_recipe_sha256=_optional_digest(body["replay_recipe_sha256"], field="replay_recipe_sha256"),
            resolver_identity_sha256=_digest(body["resolver_identity_sha256"], field="resolver_identity_sha256"),
            materialization_identity_sha256=_digest(body["materialization_identity_sha256"], field="materialization_identity_sha256"),
            source_frontier_ordinal=_nonnegative_int(body["source_frontier_ordinal"], field="source_frontier_ordinal"),
            lifecycle_hold_state=_text(body["lifecycle_hold_state"], field="lifecycle_hold_state"),  # type: ignore[arg-type]
            parent_state_ref_sha256=_optional_digest(body["parent_state_ref_sha256"], field="parent_state_ref_sha256"),
        )


def validate_state_ref_association_v1(
    association: StateRefAssociationV1,
    *,
    frame: ObservationFrameV1,
) -> StateRefAssociationV1:
    """Re-open the association against the exact frame/state pair."""

    if not isinstance(association, StateRefAssociationV1) or not isinstance(
        frame, ObservationFrameV1
    ):
        raise GenealogyIntegrityError("association validation requires typed inputs")
    state = frame.source_state_ref
    expected_class = branchability_class_for_state_ref_v1(state)
    expected_semantic = semantic_state_identity_sha256_v1(state)
    if (
        association.expedition_id != frame.expedition_id
        or association.world_trajectory_id != frame.world_trajectory_id
        or association.logical_world_id != frame.logical_world_id
        or association.logical_world_id != state.logical_world_id
        or association.next_absolute_tick != frame.next_absolute_tick
        or association.next_absolute_tick != state.next_absolute_tick
        or association.observation_frame_sha256 != frame.observation_frame_sha256()
        or association.state_ref_sha256 != state.state_ref_sha256()
        or association.branchability_class is not expected_class
        or association.semantic_state_identity_sha256 != expected_semantic
        or association.semantic_world_sha256 != state.semantic_world_sha256
        or association.control_history_sha256 != state.control_history_sha256
        or association.source_object_sha256 != state.source_object_sha256
        or association.exact_state_sha256 != state.exact_state_sha256
        or association.checkpoint_occurrence_sha256 != state.checkpoint_occurrence_sha256
        or association.rng_state_sha256 != state.rng_state_sha256
        or association.event_prefix_sha256 != state.event_prefix_sha256
        or association.parent_occurrence_sha256 != state.parent_occurrence_sha256
        or association.replay_recipe_sha256 != state.replay_recipe_sha256
        or association.resolver_identity_sha256 != _resolver_identity_v1(
            state, expected_class
        )
        or association.materialization_identity_sha256
        != _materialization_identity_v1(state)
        or association.source_object_sha256
        != state.source_artifact_ref.content_digest
    ):
        raise GenealogyIntegrityError("state_ref association differs from its frame")
    if expected_class is BranchabilityClassV1.OBSERVATION_ONLY and any(
        value is not None
        for value in (
            state.exact_state_sha256,
            state.checkpoint_occurrence_sha256,
            state.rng_state_sha256,
            state.event_prefix_sha256,
            state.parent_occurrence_sha256,
            state.replay_recipe_sha256,
        )
    ):
        raise GenealogyIntegrityError("observation-only association carries exact authority")
    if association.child_run_id is not None:
        assert association.child_request_sha256 is not None
        expected_occurrence = _child_occurrence_identity_v1(
            child_run_id=association.child_run_id,
            child_request_sha256=association.child_request_sha256,
            source_result_sha256=association.source_result_sha256,
            logical_world_id=association.logical_world_id,
            state_ref_sha256=association.state_ref_sha256,
        )
        if association.child_occurrence_id != expected_occurrence:
            raise GenealogyIntegrityError("child occurrence identity was rerolled")
    return association


def associate_observation_state_ref_v1(
    *,
    frame: ObservationFrameV1,
    source_result_sha256: ContentDigest | str,
    source_frontier_ordinal: int,
    child_run_id: str | None = None,
    child_request_sha256: ContentDigest | None = None,
    parent_state_ref_sha256: ContentDigest | None = None,
    lifecycle_hold_state: str = "OBSERVED_FRONTIER",
) -> StateRefAssociationV1:
    """Create an explicit association; no source fact is inferred."""

    if isinstance(source_result_sha256, str):
        source_result_sha256 = _digest(
            {"algorithm": "sha256", "hex": source_result_sha256},
            field="source_result_sha256",
        )
    if not isinstance(source_result_sha256, ContentDigest):
        raise GenealogyIntegrityError("source_result_sha256 is invalid")
    state = frame.source_state_ref
    branchability = branchability_class_for_state_ref_v1(state)
    state_digest = state.state_ref_sha256()
    occurrence = None
    if child_run_id is not None:
        if not isinstance(child_request_sha256, ContentDigest):
            raise GenealogyIntegrityError("child request digest is required for a child")
        occurrence = _child_occurrence_identity_v1(
            child_run_id=child_run_id,
            child_request_sha256=child_request_sha256,
            source_result_sha256=source_result_sha256,
            logical_world_id=frame.logical_world_id,
            state_ref_sha256=state_digest,
        )
    association = StateRefAssociationV1(
        expedition_id=frame.expedition_id,
        world_trajectory_id=frame.world_trajectory_id,
        logical_world_id=frame.logical_world_id,
        child_run_id=child_run_id,
        child_occurrence_id=occurrence,
        child_request_sha256=child_request_sha256,
        source_result_sha256=source_result_sha256,
        observation_frame_sha256=frame.observation_frame_sha256(),
        state_ref_sha256=state_digest,
        branchability_class=branchability,
        next_absolute_tick=frame.next_absolute_tick,
        semantic_state_identity_sha256=semantic_state_identity_sha256_v1(state),
        semantic_world_sha256=state.semantic_world_sha256,
        control_history_sha256=state.control_history_sha256,
        source_object_sha256=state.source_object_sha256,
        exact_state_sha256=state.exact_state_sha256,
        checkpoint_occurrence_sha256=state.checkpoint_occurrence_sha256,
        rng_state_sha256=state.rng_state_sha256,
        event_prefix_sha256=state.event_prefix_sha256,
        parent_occurrence_sha256=state.parent_occurrence_sha256,
        replay_recipe_sha256=state.replay_recipe_sha256,
        resolver_identity_sha256=_resolver_identity_v1(state, branchability),
        materialization_identity_sha256=_materialization_identity_v1(state),
        source_frontier_ordinal=_nonnegative_int(
            source_frontier_ordinal, field="source_frontier_ordinal"
        ),
        lifecycle_hold_state=_text(lifecycle_hold_state, field="lifecycle_hold_state"),  # type: ignore[arg-type]
        parent_state_ref_sha256=parent_state_ref_sha256,
    )
    return validate_state_ref_association_v1(association, frame=frame)


def association_from_dict_for_frame_v1(
    value: object,
    *,
    frame: ObservationFrameV1,
) -> StateRefAssociationV1:
    return validate_state_ref_association_v1(
        StateRefAssociationV1.from_dict(value), frame=frame
    )


def _same_state_association_v1(
    left: StateRefAssociationV1,
    right: StateRefAssociationV1,
) -> bool:
    """Compare the state binding while allowing different result contexts."""

    return (
        left.expedition_id == right.expedition_id
        and left.world_trajectory_id == right.world_trajectory_id
        and left.logical_world_id == right.logical_world_id
        and left.observation_frame_sha256 == right.observation_frame_sha256
        and left.state_ref_sha256 == right.state_ref_sha256
        and left.branchability_class is right.branchability_class
        and left.next_absolute_tick == right.next_absolute_tick
        and left.semantic_state_identity_sha256 == right.semantic_state_identity_sha256
        and left.semantic_world_sha256 == right.semantic_world_sha256
        and left.control_history_sha256 == right.control_history_sha256
        and left.source_object_sha256 == right.source_object_sha256
        and left.exact_state_sha256 == right.exact_state_sha256
        and left.checkpoint_occurrence_sha256 == right.checkpoint_occurrence_sha256
        and left.rng_state_sha256 == right.rng_state_sha256
        and left.event_prefix_sha256 == right.event_prefix_sha256
        and left.parent_occurrence_sha256 == right.parent_occurrence_sha256
        and left.replay_recipe_sha256 == right.replay_recipe_sha256
        and left.resolver_identity_sha256 == right.resolver_identity_sha256
        and left.materialization_identity_sha256 == right.materialization_identity_sha256
    )


def _policy_identity_v1(manifest: ExpeditionManifestV1) -> dict[str, object]:
    binding = manifest.policy_binding
    return {
        "policy_id": binding.policy_id,
        "policy_version": binding.policy_version,
        "implementation_sha256": binding.implementation_sha256.to_dict(),
        "profile_sha256": binding.profile_sha256.to_dict(),
    }


def build_parent_retention_binding_v1(
    *,
    manifest: ExpeditionManifestV1,
    state_ref: StateRefV1,
    association: StateRefAssociationV1,
    source_frontier_ordinal: int,
    reason_codes: Sequence[str],
    lifecycle_hold_state: str,
) -> dict[str, object]:
    """Bind one Parent Bank entry to policy, provenance, and lifecycle hold."""

    if not isinstance(manifest, ExpeditionManifestV1):
        raise GenealogyIntegrityError("manifest is invalid")
    if association.state_ref_sha256 != state_ref.state_ref_sha256():
        raise GenealogyIntegrityError("retention association names another StateRef")
    reasons = sorted(set(reason_codes))
    if not reasons or any(type(value) is not str or not value for value in reasons):
        raise GenealogyIntegrityError("retention reasons must be non-empty text")
    source_frontier_ordinal = _nonnegative_int(
        source_frontier_ordinal, field="source_frontier_ordinal"
    )
    lifecycle_hold_state = _text(
        lifecycle_hold_state, field="lifecycle_hold_state"
    )
    core = {
        "schema_version": GENEALOGY_RETENTION_BINDING_SCHEMA_V1,
        "expedition_id": manifest.expedition_id,
        "manifest_sha256": manifest.manifest_sha256().to_dict(),
        "state_ref_id": f"SR-{state_ref.state_ref_sha256().hex}",
        "state_ref_sha256": state_ref.state_ref_sha256().to_dict(),
        "state_semantic_identity_sha256": semantic_state_identity_sha256_v1(
            state_ref
        ).to_dict(),
        "source_frontier_ordinal": source_frontier_ordinal,
        "reason_codes": reasons,
        "policy_identity": _policy_identity_v1(manifest),
        "policy_binding_sha256": canonical_sha256(_policy_identity_v1(manifest)).to_dict(),
        "association_sha256": association.association_sha256().to_dict(),
        "provenance": {
            "world_trajectory_id": state_ref.world_trajectory_id,
            "logical_world_id": state_ref.logical_world_id,
            "next_absolute_tick": state_ref.next_absolute_tick,
            "semantic_world_sha256": state_ref.semantic_world_sha256.to_dict(),
            "control_history_sha256": state_ref.control_history_sha256.to_dict(),
            "source_object_sha256": state_ref.source_object_sha256.to_dict(),
            "exact_state_sha256": (
                None
                if state_ref.exact_state_sha256 is None
                else state_ref.exact_state_sha256.to_dict()
            ),
            "checkpoint_occurrence_sha256": (
                None
                if state_ref.checkpoint_occurrence_sha256 is None
                else state_ref.checkpoint_occurrence_sha256.to_dict()
            ),
            "rng_state_sha256": (
                None
                if state_ref.rng_state_sha256 is None
                else state_ref.rng_state_sha256.to_dict()
            ),
            "event_prefix_sha256": (
                None
                if state_ref.event_prefix_sha256 is None
                else state_ref.event_prefix_sha256.to_dict()
            ),
            "parent_occurrence_sha256": (
                None
                if state_ref.parent_occurrence_sha256 is None
                else state_ref.parent_occurrence_sha256.to_dict()
            ),
        },
        "branchability_class": association.branchability_class.value,
        "lifecycle_hold_state": lifecycle_hold_state,
        "deletion_authority": False,
    }
    return {**core, "binding_sha256": canonical_sha256(core).to_dict()}


def _state_associations_from_ledger_v1(
    ledger: Mapping[str, object],
) -> dict[str, StateRefAssociationV1]:
    """Collect only explicitly persisted associations from ledger events."""

    found: dict[str, StateRefAssociationV1] = {}
    for raw_event in ledger.get("events", ()):
        if not isinstance(raw_event, Mapping):
            continue
        payload = raw_event.get("payload")
        if not isinstance(payload, Mapping):
            continue
        values: list[object] = []
        raw_associations = payload.get("state_ref_associations")
        if isinstance(raw_associations, Sequence) and not isinstance(
            raw_associations, (str, bytes)
        ):
            values.extend(raw_associations)
        # The same field name is intentionally accepted in both event types.
        # A frontier projection and its later child settlement can describe
        # the same state with different result/occurrence context; their
        # state identity must agree, while the later settlement is preferred.
        for raw in values:
            try:
                association = StateRefAssociationV1.from_dict(raw)
            except (KeyError, TypeError, ValueError) as error:
                raise GenealogyIntegrityError(
                    "ledger contains an invalid StateRef association"
                ) from error
            state_id = f"SR-{association.state_ref_sha256.hex}"
            prior = found.get(state_id)
            if prior is not None and not _same_state_association_v1(
                prior, association
            ):
                raise GenealogyIntegrityError(
                    "one StateRef has conflicting genealogy associations"
                )
            found[state_id] = association
    return found


def _binding_validation_v1(value: object) -> dict[str, object]:
    binding = _canonical_mapping(value, field="retention binding")
    expected = {
        "schema_version",
        "expedition_id",
        "manifest_sha256",
        "state_ref_id",
        "state_ref_sha256",
        "state_semantic_identity_sha256",
        "source_frontier_ordinal",
        "reason_codes",
        "policy_identity",
        "policy_binding_sha256",
        "association_sha256",
        "provenance",
        "branchability_class",
        "lifecycle_hold_state",
        "deletion_authority",
        "binding_sha256",
    }
    if set(binding) != expected or binding.get("schema_version") != GENEALOGY_RETENTION_BINDING_SCHEMA_V1:
        raise GenealogyIntegrityError("retention binding fields are inexact")
    core = {key: item for key, item in binding.items() if key != "binding_sha256"}
    if binding.get("binding_sha256") != canonical_sha256(core).to_dict():
        raise GenealogyIntegrityError("retention binding digest changed")
    if binding.get("deletion_authority") is not False:
        raise GenealogyIntegrityError("retention binding cannot grant deletion authority")
    return binding


def build_parent_retention_bindings_v1(
    *,
    manifest: ExpeditionManifestV1,
    bank: RetainedParentBankV1,
    ledger: Mapping[str, object],
    observation_frames: Sequence[ObservationFrameV1] = (),
    frontier_ordinal: int,
    lifecycle_hold_state: str = "R4_ACTIVE_PARENT",
) -> list[dict[str, object]]:
    """Reconstruct compact policy/lifecycle bindings for the active Bank."""

    bank.validate_against_manifest(manifest)
    association_by_state = _state_associations_from_ledger_v1(ledger)
    for frame in observation_frames:
        association = associate_observation_state_ref_v1(
            frame=frame,
            source_result_sha256=frame.observation_frame_sha256(),
            source_frontier_ordinal=frontier_ordinal,
            lifecycle_hold_state="OBSERVED_FRONTIER",
        )
        state_id = f"SR-{association.state_ref_sha256.hex}"
        prior = association_by_state.get(state_id)
        if prior is not None and prior != association:
            raise GenealogyIntegrityError(
                "current frontier changes an existing StateRef association"
            )
        association_by_state[state_id] = association

    result: list[dict[str, object]] = []
    for entry in bank.entries:
        state = entry.state_ref
        state_id = f"SR-{state.state_ref_sha256().hex}"
        association = association_by_state.get(state_id)
        if association is None:
            # A pre-association bank revision can be replayed without
            # inventing a child occurrence.  Its exact StateRef remains the
            # authority; this explicit legacy resolver label keeps the gap
            # visible until a fresh observed association replaces it.
            association = associate_observation_state_ref_v1(
                frame=_frame_from_state_ref_for_legacy_v1(state),
                source_result_sha256=state.source_object_sha256,
                source_frontier_ordinal=entry.selected_at_frontier_ordinal,
                lifecycle_hold_state="R4_LEGACY_EXACT_STATE",
            )
        result.append(
            build_parent_retention_binding_v1(
                manifest=manifest,
                state_ref=state,
                association=association,
                source_frontier_ordinal=association.source_frontier_ordinal,
                reason_codes=entry.reason_codes,
                lifecycle_hold_state=lifecycle_hold_state,
            )
        )
    return sorted(result, key=lambda row: str(row["state_ref_id"]))


def _frame_from_state_ref_for_legacy_v1(state: StateRefV1) -> ObservationFrameV1:
    """Make a typed identity-only frame for old exact Bank revisions.

    This is not used to create execution evidence.  It only lets the compact
    retention binding validate the already-public StateRef fields during
    restart of a pre-genealogy ledger.
    """

    from emergence_contracts.ids import SchemaVersion
    from emergence_contracts.payloads import CanonicalPayloadV1

    descriptor = canonical_sha256(
        {"schema_version": "eiel_legacy_exact_state_identity_frame_v1"}
    )
    empty = CanonicalPayloadV1.from_value(
        SchemaVersion("eiel_legacy_exact_state_identity_payload_v1"),
        {"authority": "IDENTITY_ONLY_NO_NEW_OBSERVATION"},
    )
    return ObservationFrameV1(
        expedition_id=state.expedition_id,
        world_trajectory_id=state.world_trajectory_id,
        logical_world_id=state.logical_world_id,
        next_absolute_tick=state.next_absolute_tick,
        semantic_world_sha256=state.semantic_world_sha256,
        control_history_sha256=state.control_history_sha256,
        observation_profile_id="legacy-exact-state-identity",
        observation_profile_sha256=descriptor,
        observation_descriptor_set_sha256=descriptor,
        reducer_implementation_sha256=descriptor,
        observations=empty,
        control_readback=empty,
        policy_facing_facts=empty,
        descriptor_values=empty,
        calibration=empty,
        source_state_ref=state,
        previous_frame_sha256=None,
    )


def _append_node_v1(nodes: dict[str, dict[str, object]], node: dict[str, object]) -> None:
    node_id = str(node["node_id"])
    prior = nodes.get(node_id)
    if prior is not None and prior != node:
        raise GenealogyIntegrityError(f"DAG node {node_id} has conflicting definitions")
    nodes[node_id] = node


def _state_node_v1(
    *,
    state_ref_sha256: ContentDigest,
    semantic_state_identity_sha256: ContentDigest,
    logical_world_id: int,
    world_trajectory_id: str,
    next_absolute_tick: int,
    branchability_class: BranchabilityClassV1,
    source_object_sha256: ContentDigest,
    exact_state_sha256: ContentDigest | None,
    checkpoint_occurrence_sha256: ContentDigest | None,
    rng_state_sha256: ContentDigest | None,
    event_prefix_sha256: ContentDigest | None,
    parent_occurrence_sha256: ContentDigest | None,
    replay_recipe_sha256: ContentDigest | None,
) -> dict[str, object]:
    if branchability_class is BranchabilityClassV1.INVALID:
        raise GenealogyIntegrityError("DAG cannot contain an invalid StateRef class")
    return {
        "node_id": f"SR-{state_ref_sha256.hex}",
        "node_kind": "STATE_REF",
        "state_ref_sha256": state_ref_sha256.to_dict(),
        "semantic_state_identity_sha256": semantic_state_identity_sha256.to_dict(),
        "logical_world_id": logical_world_id,
        "world_trajectory_id": world_trajectory_id,
        "next_absolute_tick": next_absolute_tick,
        "branchability_class": branchability_class.value,
        "source_object_sha256": source_object_sha256.to_dict(),
        "exact_state_sha256": (
            None if exact_state_sha256 is None else exact_state_sha256.to_dict()
        ),
        "checkpoint_occurrence_sha256": (
            None
            if checkpoint_occurrence_sha256 is None
            else checkpoint_occurrence_sha256.to_dict()
        ),
        "rng_state_sha256": (
            None if rng_state_sha256 is None else rng_state_sha256.to_dict()
        ),
        "event_prefix_sha256": (
            None if event_prefix_sha256 is None else event_prefix_sha256.to_dict()
        ),
        "parent_occurrence_sha256": (
            None
            if parent_occurrence_sha256 is None
            else parent_occurrence_sha256.to_dict()
        ),
        "replay_recipe_sha256": (
            None if replay_recipe_sha256 is None else replay_recipe_sha256.to_dict()
        ),
    }


def _state_node_from_association_v1(
    association: StateRefAssociationV1,
) -> dict[str, object]:
    return _state_node_v1(
        state_ref_sha256=association.state_ref_sha256,
        semantic_state_identity_sha256=association.semantic_state_identity_sha256,
        logical_world_id=association.logical_world_id,
        world_trajectory_id=association.world_trajectory_id,
        next_absolute_tick=association.next_absolute_tick,
        branchability_class=association.branchability_class,
        source_object_sha256=association.source_object_sha256,
        exact_state_sha256=association.exact_state_sha256,
        checkpoint_occurrence_sha256=association.checkpoint_occurrence_sha256,
        rng_state_sha256=association.rng_state_sha256,
        event_prefix_sha256=association.event_prefix_sha256,
        parent_occurrence_sha256=association.parent_occurrence_sha256,
        replay_recipe_sha256=association.replay_recipe_sha256,
    )


def _state_node_from_state_ref_v1(state_ref: StateRefV1) -> dict[str, object]:
    branchability = branchability_class_for_state_ref_v1(state_ref)
    return _state_node_v1(
        state_ref_sha256=state_ref.state_ref_sha256(),
        semantic_state_identity_sha256=semantic_state_identity_sha256_v1(state_ref),
        logical_world_id=state_ref.logical_world_id,
        world_trajectory_id=state_ref.world_trajectory_id,
        next_absolute_tick=state_ref.next_absolute_tick,
        branchability_class=branchability,
        source_object_sha256=state_ref.source_object_sha256,
        exact_state_sha256=state_ref.exact_state_sha256,
        checkpoint_occurrence_sha256=state_ref.checkpoint_occurrence_sha256,
        rng_state_sha256=state_ref.rng_state_sha256,
        event_prefix_sha256=state_ref.event_prefix_sha256,
        parent_occurrence_sha256=state_ref.parent_occurrence_sha256,
        replay_recipe_sha256=state_ref.replay_recipe_sha256,
    )


def build_experiment_dag_v1(ledger: Mapping[str, object]) -> dict[str, object]:
    """Derive a deterministic parent/action/child DAG from the ledger."""

    if not isinstance(ledger, Mapping):
        raise GenealogyIntegrityError("ledger must be a mapping")
    nodes: dict[str, dict[str, object]] = {}
    edges: set[tuple[str, str, str]] = set()
    association_by_state = _state_associations_from_ledger_v1(ledger)
    action_to_parent: dict[str, str] = {}
    action_request_by_id: dict[str, ContentDigest] = {}
    action_nodes: set[str] = set()
    settled_actions: set[str] = set()
    child_state_by_action: dict[str, set[str]] = {}
    state_producers: dict[str, set[str]] = {}
    occurrence_producers: dict[str, str] = {}
    missing_state_ref_associations: set[str] = set()

    for raw_event in ledger.get("events", ()):
        if not isinstance(raw_event, Mapping):
            raise GenealogyIntegrityError("DAG source ledger event is invalid")
        event_type = raw_event.get("event_type")
        payload = raw_event.get("payload")
        if not isinstance(payload, Mapping):
            raise GenealogyIntegrityError("DAG source event payload is invalid")
        if event_type == "FRONTIER_OBSERVED":
            raw_values = payload.get("state_ref_associations", ())
            if isinstance(raw_values, Sequence) and not isinstance(raw_values, (str, bytes)):
                for raw in raw_values:
                    association = StateRefAssociationV1.from_dict(raw)
                    state_id = f"SR-{association.state_ref_sha256.hex}"
                    _append_node_v1(nodes, _state_node_from_association_v1(association))
        elif event_type == "ACTION_DISPATCH_BOUND":
            action_id = _text(payload.get("action_id"), field="action_id")
            assert action_id is not None
            action_node_id = f"ACTION-{action_id}"
            if action_node_id in action_nodes:
                raise GenealogyIntegrityError("historical action node was duplicated")
            action_nodes.add(action_node_id)
            raw_request_digest = payload.get("child_request_sha256")
            request_digest = _digest(
                raw_request_digest
                if isinstance(raw_request_digest, Mapping)
                else {"algorithm": "sha256", "hex": raw_request_digest},
                field=f"action {action_id} child_request_sha256",
            )
            action_request_by_id[action_id] = request_digest
            continuation = payload.get("continuation")
            parent_digest = None
            parent_state: StateRefV1 | None = None
            if isinstance(continuation, Mapping):
                parent = continuation.get("parent_state_ref")
                if isinstance(parent, Mapping):
                    try:
                        parent_state = StateRefV1.from_dict(parent)
                        parent_digest = parent_state.state_ref_sha256().hex
                    except (KeyError, TypeError, ValueError) as error:
                        raise GenealogyIntegrityError("DAG continuation parent is invalid") from error
            if parent_state is not None:
                _append_node_v1(nodes, _state_node_from_state_ref_v1(parent_state))
            _append_node_v1(
                nodes,
                {
                    "node_id": action_node_id,
                    "node_kind": "ACTION",
                    "action_id": action_id,
                    "child_request_sha256": request_digest.to_dict(),
                    "historical_continuation": parent_digest is not None,
                },
            )
            if parent_digest is not None:
                parent_id = f"SR-{parent_digest}"
                action_to_parent[action_id] = parent_id
                edges.add((parent_id, action_node_id, "PARENT_TO_ACTION"))
        elif event_type == "ACTION_SETTLED":
            action_id = _text(payload.get("action_id"), field="action_id")
            assert action_id is not None
            if action_id not in action_request_by_id:
                raise GenealogyIntegrityError("settlement has no dispatched action")
            if action_id in settled_actions:
                raise GenealogyIntegrityError("historical action was settled twice")
            settled_actions.add(action_id)
            raw_values = payload.get("state_ref_associations", ())
            if not isinstance(raw_values, Sequence) or isinstance(raw_values, (str, bytes)):
                missing_state_ref_associations.add(action_id)
                continue
            for raw in raw_values:
                association = StateRefAssociationV1.from_dict(raw)
                state_id = f"SR-{association.state_ref_sha256.hex}"
                prior = association_by_state.get(state_id)
                if prior is not None and not _same_state_association_v1(
                    prior, association
                ):
                    raise GenealogyIntegrityError("settlement association changed")
                association_by_state[state_id] = association
                if association.child_run_id is None:
                    raise GenealogyIntegrityError(
                        "settled StateRef association lacks a child occurrence"
                    )
                expected_request = action_request_by_id[action_id]
                if association.child_request_sha256 != expected_request:
                    raise GenealogyIntegrityError(
                        "settled StateRef association rerolled its child request"
                    )
                occurrence = association.child_occurrence_id
                assert occurrence is not None
                prior_occurrence_action = occurrence_producers.get(occurrence)
                if prior_occurrence_action is not None:
                    raise GenealogyIntegrityError(
                        "historical child occurrence identity was duplicated"
                    )
                occurrence_producers[occurrence] = action_id
                expected_parent = action_to_parent.get(action_id)
                actual_parent = (
                    None
                    if association.parent_state_ref_sha256 is None
                    else f"SR-{association.parent_state_ref_sha256.hex}"
                )
                if actual_parent != expected_parent:
                    raise GenealogyIntegrityError(
                        "settled StateRef association changed its historical parent"
                    )
                prior_producers = state_producers.setdefault(state_id, set())
                prior_producers.add(action_id)
                if len(prior_producers) > 1:
                    raise GenealogyIntegrityError(
                        "one child StateRef was produced by multiple historical actions"
                    )
                _append_node_v1(nodes, _state_node_from_association_v1(association))
                child_state_by_action.setdefault(action_id, set()).add(state_id)
                edges.add((f"ACTION-{action_id}", state_id, "ACTION_TO_CHILD"))

    for action_id, child_states in child_state_by_action.items():
        if len(child_states) > 1:
            # Multiple logical worlds from one founder action are valid, but
            # each must have its own state node and exact occurrence identity.
            associations = [association_by_state.get(state_id) for state_id in child_states]
            if any(item is None for item in associations):
                raise GenealogyIntegrityError("DAG child state association is missing")
            occurrences = [item.child_occurrence_id for item in associations if item is not None]
            if len(occurrences) != len(set(occurrences)):
                raise GenealogyIntegrityError("DAG child occurrence identity was duplicated")

    # Every historical action must point to an extant parent and every exact
    # child with a declared parent must point back to the action that produced
    # it.  Observation-only founder outputs remain valid roots/candidates.
    for action_id, parent_id in action_to_parent.items():
        if parent_id not in nodes:
            raise GenealogyIntegrityError("DAG historical action has an orphan parent")
        if (
            action_id in settled_actions
            and not child_state_by_action.get(action_id)
            and action_id not in missing_state_ref_associations
        ):
            raise GenealogyIntegrityError("DAG historical action has no settled child")
    for state_id, association in association_by_state.items():
        if association.parent_state_ref_sha256 is not None:
            parent_id = f"SR-{association.parent_state_ref_sha256.hex}"
            if parent_id not in nodes:
                raise GenealogyIntegrityError("DAG descendant has an orphan parent StateRef")

    edge_rows = [
        {"source": source, "target": target, "edge_kind": kind}
        for source, target, kind in sorted(edges)
    ]
    node_rows = [nodes[key] for key in sorted(nodes)]
    core = {
        "schema_version": GENEALOGY_DAG_SCHEMA_V1,
        "expedition_id": ledger.get("expedition_id"),
        "manifest_sha256": ledger.get("manifest_sha256"),
        "nodes": node_rows,
        "edges": edge_rows,
        "integrity": {
            "duplicate_historical_actions": 0,
            "duplicate_child_roots": 0,
            "orphaned_descendant_state_refs": 0,
            "missing_state_ref_associations": len(missing_state_ref_associations),
            "source": "append_only_expedition_ledger_only",
        },
    }
    return {**core, "dag_sha256": canonical_sha256(core).to_dict()}


def validate_experiment_dag_v1(value: object) -> dict[str, object]:
    dag = _canonical_mapping(value, field="experiment_dag")
    expected = {
        "schema_version",
        "expedition_id",
        "manifest_sha256",
        "nodes",
        "edges",
        "integrity",
        "dag_sha256",
    }
    if set(dag) != expected or dag.get("schema_version") != GENEALOGY_DAG_SCHEMA_V1:
        raise GenealogyIntegrityError("experiment DAG fields are inexact")
    _text(dag.get("expedition_id"), field="experiment_dag.expedition_id")
    _digest(
        {"algorithm": "sha256", "hex": dag.get("manifest_sha256")},
        field="experiment_dag.manifest_sha256",
    )
    raw_nodes = dag.get("nodes")
    raw_edges = dag.get("edges")
    if not isinstance(raw_nodes, list) or not isinstance(raw_edges, list):
        raise GenealogyIntegrityError("experiment DAG nodes and edges must be arrays")
    node_ids: set[str] = set()
    node_kinds: dict[str, str] = {}
    state_node_fields = {
        "node_id",
        "node_kind",
        "state_ref_sha256",
        "semantic_state_identity_sha256",
        "logical_world_id",
        "world_trajectory_id",
        "next_absolute_tick",
        "branchability_class",
        "source_object_sha256",
        "exact_state_sha256",
        "checkpoint_occurrence_sha256",
        "rng_state_sha256",
        "event_prefix_sha256",
        "parent_occurrence_sha256",
        "replay_recipe_sha256",
    }
    action_node_fields = {
        "node_id",
        "node_kind",
        "action_id",
        "child_request_sha256",
        "historical_continuation",
    }
    for index, raw_node in enumerate(raw_nodes):
        node = _canonical_mapping(raw_node, field=f"experiment_dag.nodes[{index}]")
        kind = node.get("node_kind")
        if kind == "STATE_REF":
            if set(node) != state_node_fields:
                raise GenealogyIntegrityError(
                    f"experiment DAG state node {index} fields are inexact"
                )
            node_id = _text(node.get("node_id"), field=f"state node {index}.node_id")
            assert node_id is not None
            state_digest = _digest(
                node["state_ref_sha256"], field=f"state node {index}.state_ref_sha256"
            )
            if node_id != f"SR-{state_digest.hex}":
                raise GenealogyIntegrityError("experiment DAG StateRef node ID changed")
            semantic_digest = _digest(
                node["semantic_state_identity_sha256"],
                field=f"state node {index}.semantic_state_identity_sha256",
            )
            if not isinstance(semantic_digest, ContentDigest):
                raise GenealogyIntegrityError("state node semantic identity is invalid")
            _nonnegative_int(node["logical_world_id"], field=f"state node {index}.logical_world_id")
            _text(node["world_trajectory_id"], field=f"state node {index}.world_trajectory_id")
            _nonnegative_int(node["next_absolute_tick"], field=f"state node {index}.next_absolute_tick")
            try:
                BranchabilityClassV1(node["branchability_class"])
            except (TypeError, ValueError) as error:
                raise GenealogyIntegrityError("state node branchability class is invalid") from error
            for field in (
                "source_object_sha256",
                "exact_state_sha256",
                "checkpoint_occurrence_sha256",
                "rng_state_sha256",
                "event_prefix_sha256",
                "parent_occurrence_sha256",
                "replay_recipe_sha256",
            ):
                _optional_digest(node[field], field=f"state node {index}.{field}")
        elif kind == "ACTION":
            if set(node) != action_node_fields:
                raise GenealogyIntegrityError(
                    f"experiment DAG action node {index} fields are inexact"
                )
            node_id = _text(node.get("node_id"), field=f"action node {index}.node_id")
            action_id = _text(node.get("action_id"), field=f"action node {index}.action_id")
            assert node_id is not None and action_id is not None
            if node_id != f"ACTION-{action_id}" or type(node["historical_continuation"]) is not bool:
                raise GenealogyIntegrityError("experiment DAG action identity is invalid")
            _digest(node["child_request_sha256"], field=f"action node {index}.child_request_sha256")
        else:
            raise GenealogyIntegrityError("experiment DAG contains an unknown node kind")
        if node_id in node_ids:
            raise GenealogyIntegrityError("experiment DAG contains duplicate node IDs")
        node_ids.add(node_id)
        node_kinds[node_id] = str(kind)
    canonical_nodes = sorted(raw_nodes, key=lambda node: str(node["node_id"]))
    if raw_nodes != canonical_nodes:
        raise GenealogyIntegrityError("experiment DAG nodes are not deterministically ordered")

    edge_keys: set[tuple[str, str, str]] = set()
    adjacency: dict[str, set[str]] = {node_id: set() for node_id in node_ids}
    incoming_child: dict[str, str] = {}
    for index, raw_edge in enumerate(raw_edges):
        edge = _canonical_mapping(raw_edge, field=f"experiment_dag.edges[{index}]")
        if set(edge) != {"source", "target", "edge_kind"}:
            raise GenealogyIntegrityError("experiment DAG edge fields are inexact")
        source = _text(edge["source"], field=f"DAG edge {index}.source")
        target = _text(edge["target"], field=f"DAG edge {index}.target")
        edge_kind = _text(edge["edge_kind"], field=f"DAG edge {index}.edge_kind")
        assert source is not None and target is not None and edge_kind is not None
        if source not in node_ids or target not in node_ids:
            raise GenealogyIntegrityError("experiment DAG edge points to an unknown node")
        if edge_kind == "PARENT_TO_ACTION":
            if node_kinds[source] != "STATE_REF" or node_kinds[target] != "ACTION":
                raise GenealogyIntegrityError("PARENT_TO_ACTION edge has invalid endpoints")
        elif edge_kind == "ACTION_TO_CHILD":
            if node_kinds[source] != "ACTION" or node_kinds[target] != "STATE_REF":
                raise GenealogyIntegrityError("ACTION_TO_CHILD edge has invalid endpoints")
            prior_source = incoming_child.get(target)
            if prior_source is not None and prior_source != source:
                raise GenealogyIntegrityError("a child StateRef has multiple producer actions")
            incoming_child[target] = source
        else:
            raise GenealogyIntegrityError("experiment DAG edge kind is invalid")
        key = (source, target, edge_kind)
        if key in edge_keys:
            raise GenealogyIntegrityError("experiment DAG contains duplicate edges")
        edge_keys.add(key)
        adjacency[source].add(target)
    canonical_edges = sorted(raw_edges, key=lambda edge: (str(edge["source"]), str(edge["target"]), str(edge["edge_kind"])))
    if raw_edges != canonical_edges:
        raise GenealogyIntegrityError("experiment DAG edges are not deterministically ordered")

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node_id: str) -> None:
        if node_id in visiting:
            raise GenealogyIntegrityError("experiment DAG contains a cycle")
        if node_id in visited:
            return
        visiting.add(node_id)
        for child_id in sorted(adjacency[node_id]):
            visit(child_id)
        visiting.remove(node_id)
        visited.add(node_id)

    for node_id in sorted(node_ids):
        visit(node_id)

    integrity = _canonical_mapping(dag["integrity"], field="experiment_dag.integrity")
    if set(integrity) != {
        "duplicate_historical_actions",
        "duplicate_child_roots",
        "orphaned_descendant_state_refs",
        "missing_state_ref_associations",
        "source",
    }:
        raise GenealogyIntegrityError("experiment DAG integrity fields are inexact")
    for field in (
        "duplicate_historical_actions",
        "duplicate_child_roots",
        "orphaned_descendant_state_refs",
        "missing_state_ref_associations",
    ):
        _nonnegative_int(integrity[field], field=f"experiment_dag.integrity.{field}")
    if integrity["source"] != "append_only_expedition_ledger_only":
        raise GenealogyIntegrityError("experiment DAG source is not canonical")
    core = {key: item for key, item in dag.items() if key != "dag_sha256"}
    if dag.get("dag_sha256") != canonical_sha256(core).to_dict():
        raise GenealogyIntegrityError("experiment DAG digest changed")
    return dag


__all__ = (
    "BranchabilityClassV1",
    "GENEALOGY_ASSOCIATION_SCHEMA_V1",
    "GENEALOGY_DAG_SCHEMA_V1",
    "GENEALOGY_RETENTION_BINDING_SCHEMA_V1",
    "GenealogyIntegrityError",
    "StateRefAssociationV1",
    "associate_observation_state_ref_v1",
    "association_from_dict_for_frame_v1",
    "branchability_class_for_state_ref_v1",
    "build_experiment_dag_v1",
    "build_parent_retention_binding_v1",
    "build_parent_retention_bindings_v1",
    "classify_state_ref_value_v1",
    "semantic_state_identity_sha256_v1",
    "validate_experiment_dag_v1",
    "validate_state_ref_association_v1",
)
