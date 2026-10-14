"""Durable, route-neutral contracts for Adaptive Science Expedition V1.

The module deliberately owns data contracts only.  It does not import the
composition layer, select a runtime route, or grant checkpoint/continuation
authority.  Every persisted object is immutable, rejects unknown fields, uses
absolute-tick cursors, and derives identity with the repository canonical
SHA-256 implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from ._validation import require_instance, require_int, require_text
from .artifacts import RunArtifactRefV1
from .canonical import canonical_sha256
from .errors import contract_invalid
from .ids import ContentDigest, RunId, SchemaVersion
from .payloads import CanonicalPayloadV1


EXPEDITION_MANIFEST_SCHEMA_VERSION = SchemaVersion(
    "eiel_adaptive_science_expedition_manifest_v1"
)
OBSERVATION_FRAME_SCHEMA_VERSION = SchemaVersion(
    "eiel_adaptive_science_observation_frame_v1"
)
STATE_REF_SCHEMA_VERSION = SchemaVersion("eiel_adaptive_science_state_ref_v1")
DECISION_SCHEMA_VERSION = SchemaVersion("eiel_adaptive_science_decision_v1")
GENERATED_CHILD_CONTINUATION_SCHEMA_VERSION = SchemaVersion(
    "eiel_adaptive_science_generated_child_continuation_v1"
)
JOURNAL_RECORD_SCHEMA_VERSION = SchemaVersion(
    "eiel_adaptive_science_journal_record_v1"
)
EVOLUTIONARY_ATLAS_SCHEMA_VERSION = SchemaVersion(
    "eiel_adaptive_science_evolutionary_atlas_v1"
)
RETAINED_PARENT_BANK_SCHEMA_VERSION = SchemaVersion(
    "eiel_adaptive_science_retained_parent_bank_v1"
)

_UINT64_MAX = (1 << 64) - 1
_EXPEDITION_LIFECYCLE_POLICY_SCHEMA_VERSION_V1 = SchemaVersion(
    "eiel_adaptive_expedition_lifecycle_policy_v1"
)
_RUN_ARTIFACT_LIFECYCLE_SCHEMA_VERSION_V1 = (
    "eiel_run_artifact_lifecycle_contract_v1"
)
_EXPEDITION_CHILD_LIFECYCLE_PROFILES_V1 = frozenset(
    {"CHECKPOINT_TRANSIT", "SCIENCE_COMPACT", "DEBUG_FORENSIC"}
)
# Stop-condition names are manifest vocabulary, not coordinator policy.  They
# are intentionally a small closed set so a typo or an unconsumed declaration
# cannot cross the durable/API boundary silently.
SUPPORTED_EXPEDITION_STOP_CONDITIONS_V1 = frozenset(
    {
        "integrity_failure",
        "operator_stop",
        "resource_budget_exhausted",
        "wall_time_budget",
        "maximum_frontiers",
        "search_space_exhausted",
    }
)
SCIENCE_EXPEDITION_INSTRUMENT_V1 = "EIEL_SCIENCE_INSTRUMENT_V1"
_SHA256_HEX_V1 = frozenset("0123456789abcdef")


class StateClassV1(str, Enum):
    OBSERVATION_ONLY = "OBSERVATION_ONLY"
    BRANCHABLE_EXACT = "BRANCHABLE_EXACT"
    REPLAY_DERIVABLE = "REPLAY_DERIVABLE"
    TERMINAL_ONLY = "TERMINAL_ONLY"


class ExperimentActionKindV1(str, Enum):
    CONTINUE = "CONTINUE"
    BRANCH_LOCAL = "BRANCH_LOCAL"
    EXPLORE_GLOBAL = "EXPLORE_GLOBAL"
    REPLICATE = "REPLICATE"
    CAUSAL_PROBE = "CAUSAL_PROBE"
    TRANSFER_CHALLENGE = "TRANSFER_CHALLENGE"
    REQUEST_DETAILED_OBSERVATION = "REQUEST_DETAILED_OBSERVATION"
    PARK = "PARK"
    RETIRE = "RETIRE"
    QUARANTINE = "QUARANTINE"


class ChildRealizationKindV1(str, Enum):
    FOUNDER_BASIS_REQUEST = "FOUNDER_BASIS_REQUEST"
    EXACT_STATE_REQUEST = "EXACT_STATE_REQUEST"


class JournalRecordKindV1(str, Enum):
    OBSERVATION_ACCEPTED = "OBSERVATION_ACCEPTED"
    DECISION_COMMITTED = "DECISION_COMMITTED"
    CHILD_REQUEST_GENERATED = "CHILD_REQUEST_GENERATED"
    ATLAS_UPDATED = "ATLAS_UPDATED"
    PARENT_BANK_UPDATED = "PARENT_BANK_UPDATED"
    FRONTIER_SETTLED = "FRONTIER_SETTLED"
    EXPEDITION_TERMINATED = "EXPEDITION_TERMINATED"
    INTEGRITY_FAILURE = "INTEGRITY_FAILURE"


def _object(value: object, required: set[str], *, field: str) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != required:
        raise contract_invalid(
            "object has unknown or missing fields",
            field=field,
        )
    return value


def _array(value: object, *, field: str) -> list[object]:
    if not isinstance(value, list):
        raise contract_invalid("value must be an array", field=field)
    return value


def _digest(value: object, *, field: str) -> ContentDigest:
    body = _object(value, {"algorithm", "hex"}, field=field)
    try:
        return ContentDigest(body["algorithm"], body["hex"])  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise contract_invalid("digest object is invalid", field=field) from exc


def _optional_digest(value: object, *, field: str) -> ContentDigest | None:
    return None if value is None else _digest(value, field=field)


def _payload(value: object, *, field: str) -> CanonicalPayloadV1:
    try:
        return CanonicalPayloadV1.from_dict(value)
    except (TypeError, ValueError, KeyError) as exc:
        raise contract_invalid("canonical payload is invalid", field=field) from exc


def _validate_expedition_lifecycle_policy_envelope_v1(
    value: CanonicalPayloadV1,
) -> None:
    """Validate the public manifest envelope without duplicating runtime law.

    The existing artifact-lifecycle implementation remains the semantic
    authority for the embedded outer contract.  This contract-layer check
    merely prevents an opaque or unused payload from crossing the API/replay
    boundary as if it were an Expedition lifecycle declaration.
    """

    if value.schema_version != _EXPEDITION_LIFECYCLE_POLICY_SCHEMA_VERSION_V1:
        raise contract_invalid(
            "unsupported Expedition lifecycle policy schema",
            field="artifact_lifecycle_policy",
        )
    payload = value.to_value()
    if not isinstance(payload, dict) or set(payload) != {
        "child_contract_class",
        "lifecycle_authority",
        "outer_contract",
        "retention_authority_created",
    }:
        raise contract_invalid(
            "Expedition lifecycle policy has unknown or missing fields",
            field="artifact_lifecycle_policy",
        )
    if payload["child_contract_class"] not in (
        _EXPEDITION_CHILD_LIFECYCLE_PROFILES_V1
    ):
        raise contract_invalid(
            "unsupported child lifecycle contract class",
            field="artifact_lifecycle_policy",
        )
    if payload["lifecycle_authority"] != "canonical_run_artifact_lifecycle":
        raise contract_invalid(
            "Expedition lifecycle policy must use canonical lifecycle authority",
            field="artifact_lifecycle_policy",
        )
    if payload["retention_authority_created"] is not False:
        raise contract_invalid(
            "Expedition manifest cannot create retention authority",
            field="artifact_lifecycle_policy",
        )
    outer = payload["outer_contract"]
    if (
        not isinstance(outer, dict)
        or outer.get("schema_version")
        != _RUN_ARTIFACT_LIFECYCLE_SCHEMA_VERSION_V1
        or outer.get("lifecycle_profile") != "SCIENCE_COMPACT"
        or not isinstance(outer.get("contract_sha256"), str)
        or len(outer["contract_sha256"]) != 64
        or any(character not in _SHA256_HEX_V1 for character in outer["contract_sha256"])
    ):
        raise contract_invalid(
            "Expedition outer lifecycle contract identity is invalid",
            field="artifact_lifecycle_policy",
        )


def _artifact_ref(value: object, *, field: str) -> RunArtifactRefV1:
    body = _object(
        value,
        {
            "schema_version",
            "artifact_id",
            "artifact_kind",
            "artifact_schema_version",
            "locator",
            "content_digest",
            "size_bytes",
        },
        field=field,
    )
    _digest(body["content_digest"], field=f"{field}.content_digest")
    try:
        return RunArtifactRefV1.from_dict(body)
    except (TypeError, ValueError, KeyError) as exc:
        raise contract_invalid("run artifact reference is invalid", field=field) from exc


def _text_tuple(
    value: object,
    *,
    field: str,
    nonempty: bool = False,
    sorted_unique: bool = True,
) -> tuple[str, ...]:
    rows = _array(value, field=field)
    if any(not isinstance(item, str) for item in rows):
        raise contract_invalid("array entries must be strings", field=field)
    result = tuple(rows)  # type: ignore[arg-type]
    _require_text_tuple(
        result,
        field=field,
        nonempty=nonempty,
        sorted_unique=sorted_unique,
    )
    return result


def _require_text_tuple(
    value: object,
    *,
    field: str,
    nonempty: bool = False,
    sorted_unique: bool = True,
) -> tuple[str, ...]:
    if not isinstance(value, tuple) or any(not isinstance(item, str) for item in value):
        raise contract_invalid("canonical string collection must be a tuple", field=field)
    if nonempty and not value:
        raise contract_invalid("canonical string collection must not be empty", field=field)
    for item in value:
        require_text(item, field=field)
    if sorted_unique and value != tuple(sorted(set(value))):
        raise contract_invalid("string collection must be sorted and unique", field=field)
    return value


def _require_digest_tuple(value: object, *, field: str) -> tuple[ContentDigest, ...]:
    if not isinstance(value, tuple) or any(not isinstance(item, ContentDigest) for item in value):
        raise contract_invalid("digest collection must be a tuple", field=field)
    if value != tuple(sorted(set(value), key=str)):
        raise contract_invalid("digest collection must be sorted and unique", field=field)
    return value


def _digest_tuple(value: object, *, field: str) -> tuple[ContentDigest, ...]:
    rows = tuple(_digest(item, field=field) for item in _array(value, field=field))
    return _require_digest_tuple(rows, field=field)


@dataclass(frozen=True, slots=True)
class _PolicyBindingV1:
    policy_id: str
    policy_version: str
    implementation_sha256: ContentDigest
    profile_sha256: ContentDigest
    profile: CanonicalPayloadV1

    def __post_init__(self) -> None:
        require_text(self.policy_id, field="policy_id")
        require_text(self.policy_version, field="policy_version")
        require_instance(
            self.implementation_sha256,
            ContentDigest,
            field="implementation_sha256",
        )
        require_instance(self.profile_sha256, ContentDigest, field="profile_sha256")
        require_instance(self.profile, CanonicalPayloadV1, field="profile")
        if self.profile_sha256 != self.profile.content_digest():
            raise contract_invalid(
                "policy profile digest does not bind the canonical versioned payload",
                field="profile_sha256",
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "policy_id": self.policy_id,
            "policy_version": self.policy_version,
            "implementation_sha256": self.implementation_sha256.to_dict(),
            "profile_sha256": self.profile_sha256.to_dict(),
            "profile": self.profile.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: object) -> "_PolicyBindingV1":
        body = _object(
            value,
            {
                "policy_id",
                "policy_version",
                "implementation_sha256",
                "profile_sha256",
                "profile",
            },
            field="policy_binding",
        )
        return cls(
            policy_id=body["policy_id"],  # type: ignore[arg-type]
            policy_version=body["policy_version"],  # type: ignore[arg-type]
            implementation_sha256=_digest(
                body["implementation_sha256"], field="implementation_sha256"
            ),
            profile_sha256=_digest(body["profile_sha256"], field="profile_sha256"),
            profile=_payload(body["profile"], field="profile"),
        )


@dataclass(frozen=True, slots=True)
class StateRefV1:
    expedition_id: str
    world_trajectory_id: str
    logical_world_id: int
    next_absolute_tick: int
    state_class: StateClassV1
    semantic_world_sha256: ContentDigest
    control_history_sha256: ContentDigest
    source_object_sha256: ContentDigest
    source_artifact_ref: RunArtifactRefV1
    exact_state_sha256: ContentDigest | None
    checkpoint_occurrence_sha256: ContentDigest | None
    rng_state_sha256: ContentDigest | None
    event_prefix_sha256: ContentDigest | None
    parent_occurrence_sha256: ContentDigest | None
    replay_recipe_sha256: ContentDigest | None
    schema_version: SchemaVersion = STATE_REF_SCHEMA_VERSION

    def __post_init__(self) -> None:
        require_text(self.expedition_id, field="expedition_id")
        require_text(self.world_trajectory_id, field="world_trajectory_id")
        require_int(self.logical_world_id, field="logical_world_id")
        require_int(self.next_absolute_tick, field="next_absolute_tick")
        require_instance(self.state_class, StateClassV1, field="state_class")
        for field_name in (
            "semantic_world_sha256",
            "control_history_sha256",
            "source_object_sha256",
        ):
            require_instance(getattr(self, field_name), ContentDigest, field=field_name)
        require_instance(
            self.source_artifact_ref, RunArtifactRefV1, field="source_artifact_ref"
        )
        for field_name in (
            "exact_state_sha256",
            "checkpoint_occurrence_sha256",
            "rng_state_sha256",
            "event_prefix_sha256",
            "parent_occurrence_sha256",
            "replay_recipe_sha256",
        ):
            item = getattr(self, field_name)
            if item is not None:
                require_instance(item, ContentDigest, field=field_name)
        if self.source_object_sha256 != self.source_artifact_ref.content_digest:
            raise contract_invalid(
                "source object digest must equal the wrapped RunArtifactRef digest",
                field="source_object_sha256",
            )
        exact_evidence = (
            self.exact_state_sha256,
            self.checkpoint_occurrence_sha256,
            self.rng_state_sha256,
            self.event_prefix_sha256,
            self.parent_occurrence_sha256,
        )
        if self.state_class in {
            StateClassV1.BRANCHABLE_EXACT,
            StateClassV1.REPLAY_DERIVABLE,
        }:
            if any(item is None for item in exact_evidence):
                raise contract_invalid(
                    "exactly continuable state lacks exact state/RNG/event/occurrence evidence",
                    field="state_class",
                )
            if (
                self.state_class is StateClassV1.BRANCHABLE_EXACT
                and self.replay_recipe_sha256 is not None
            ):
                raise contract_invalid(
                    "direct exact state must not masquerade as replay-derived",
                    field="replay_recipe_sha256",
                )
            if (
                self.state_class is StateClassV1.REPLAY_DERIVABLE
                and self.replay_recipe_sha256 is None
            ):
                raise contract_invalid(
                    "replay-derived state requires a replay recipe",
                    field="replay_recipe_sha256",
                )
        elif any(item is not None for item in (*exact_evidence, self.replay_recipe_sha256)):
            raise contract_invalid(
                "observation/terminal state cannot carry implicit continuation authority",
                field="state_class",
            )
        if self.schema_version != STATE_REF_SCHEMA_VERSION:
            raise contract_invalid("unsupported state reference schema", field="schema_version")

    @property
    def supports_exact_continuation(self) -> bool:
        return self.state_class in {
            StateClassV1.BRANCHABLE_EXACT,
            StateClassV1.REPLAY_DERIVABLE,
        }

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": str(self.schema_version),
            "expedition_id": self.expedition_id,
            "world_trajectory_id": self.world_trajectory_id,
            "logical_world_id": self.logical_world_id,
            "next_absolute_tick": self.next_absolute_tick,
            "state_class": self.state_class.value,
            "semantic_world_sha256": self.semantic_world_sha256.to_dict(),
            "control_history_sha256": self.control_history_sha256.to_dict(),
            "source_object_sha256": self.source_object_sha256.to_dict(),
            "source_artifact_ref": self.source_artifact_ref.to_dict(),
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
                None
                if self.event_prefix_sha256 is None
                else self.event_prefix_sha256.to_dict()
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
        }

    def state_ref_sha256(self) -> ContentDigest:
        return canonical_sha256(self.to_dict())

    @classmethod
    def from_dict(cls, value: object) -> "StateRefV1":
        body = _object(
            value,
            {
                "schema_version",
                "expedition_id",
                "world_trajectory_id",
                "logical_world_id",
                "next_absolute_tick",
                "state_class",
                "semantic_world_sha256",
                "control_history_sha256",
                "source_object_sha256",
                "source_artifact_ref",
                "exact_state_sha256",
                "checkpoint_occurrence_sha256",
                "rng_state_sha256",
                "event_prefix_sha256",
                "parent_occurrence_sha256",
                "replay_recipe_sha256",
            },
            field="state_ref",
        )
        return cls(
            expedition_id=body["expedition_id"],  # type: ignore[arg-type]
            world_trajectory_id=body["world_trajectory_id"],  # type: ignore[arg-type]
            logical_world_id=body["logical_world_id"],  # type: ignore[arg-type]
            next_absolute_tick=body["next_absolute_tick"],  # type: ignore[arg-type]
            state_class=StateClassV1(body["state_class"]),  # type: ignore[arg-type]
            semantic_world_sha256=_digest(
                body["semantic_world_sha256"], field="semantic_world_sha256"
            ),
            control_history_sha256=_digest(
                body["control_history_sha256"], field="control_history_sha256"
            ),
            source_object_sha256=_digest(
                body["source_object_sha256"], field="source_object_sha256"
            ),
            source_artifact_ref=_artifact_ref(
                body["source_artifact_ref"], field="source_artifact_ref"
            ),
            exact_state_sha256=_optional_digest(
                body["exact_state_sha256"], field="exact_state_sha256"
            ),
            checkpoint_occurrence_sha256=_optional_digest(
                body["checkpoint_occurrence_sha256"],
                field="checkpoint_occurrence_sha256",
            ),
            rng_state_sha256=_optional_digest(
                body["rng_state_sha256"], field="rng_state_sha256"
            ),
            event_prefix_sha256=_optional_digest(
                body["event_prefix_sha256"], field="event_prefix_sha256"
            ),
            parent_occurrence_sha256=_optional_digest(
                body["parent_occurrence_sha256"], field="parent_occurrence_sha256"
            ),
            replay_recipe_sha256=_optional_digest(
                body["replay_recipe_sha256"], field="replay_recipe_sha256"
            ),
            schema_version=SchemaVersion(body["schema_version"]),  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class ExpeditionManifestV1:
    expedition_id: str
    question_description: str
    no_claims: bool
    hypothesis_ids: tuple[str, ...]
    required_science_instrument: str
    founder_basis_refs: tuple[RunArtifactRefV1, ...]
    search_space: CanonicalPayloadV1
    epoch_ticks: int
    maximum_absolute_tick: int
    maximum_frontiers: int
    maximum_actions_per_frontier: int
    wall_time_budget_seconds: int | None
    observation_profile_id: str
    observation_descriptor_set_sha256: ContentDigest
    policy_binding: _PolicyBindingV1
    outer_search_seed: int
    maximum_atlas_entries: int
    maximum_retained_parent_entries: int
    maximum_retained_parent_bytes: int
    artifact_lifecycle_policy: CanonicalPayloadV1
    stop_conditions: tuple[str, ...]
    schema_version: SchemaVersion = EXPEDITION_MANIFEST_SCHEMA_VERSION

    def __post_init__(self) -> None:
        require_text(self.expedition_id, field="expedition_id")
        require_text(self.question_description, field="question_description")
        if self.no_claims is not True:
            raise contract_invalid(
                "an expedition manifest cannot grant scientific claims", field="no_claims"
            )
        _require_text_tuple(self.hypothesis_ids, field="hypothesis_ids")
        require_text(self.required_science_instrument, field="required_science_instrument")
        if self.required_science_instrument != SCIENCE_EXPEDITION_INSTRUMENT_V1:
            raise contract_invalid(
                "unsupported Expedition science instrument",
                field="required_science_instrument",
            )
        if (
            not isinstance(self.founder_basis_refs, tuple)
            or not self.founder_basis_refs
            or any(not isinstance(item, RunArtifactRefV1) for item in self.founder_basis_refs)
        ):
            raise contract_invalid(
                "founder basis must be a nonempty tuple of RunArtifactRefV1",
                field="founder_basis_refs",
            )
        founder_ids = tuple(str(item.artifact_id) for item in self.founder_basis_refs)
        if founder_ids != tuple(sorted(set(founder_ids))):
            raise contract_invalid(
                "founder basis references must be sorted and unique by artifact_id",
                field="founder_basis_refs",
            )
        require_instance(self.search_space, CanonicalPayloadV1, field="search_space")
        require_int(self.epoch_ticks, field="epoch_ticks", minimum=1)
        require_int(self.maximum_absolute_tick, field="maximum_absolute_tick", minimum=1)
        if self.maximum_absolute_tick < self.epoch_ticks:
            raise contract_invalid(
                "maximum absolute tick precedes the first epoch boundary",
                field="maximum_absolute_tick",
            )
        require_int(self.maximum_frontiers, field="maximum_frontiers", minimum=1)
        require_int(
            self.maximum_actions_per_frontier,
            field="maximum_actions_per_frontier",
            minimum=1,
        )
        if self.wall_time_budget_seconds is not None:
            require_int(
                self.wall_time_budget_seconds,
                field="wall_time_budget_seconds",
                minimum=1,
            )
        require_text(self.observation_profile_id, field="observation_profile_id")
        require_instance(
            self.observation_descriptor_set_sha256,
            ContentDigest,
            field="observation_descriptor_set_sha256",
        )
        require_instance(self.policy_binding, _PolicyBindingV1, field="policy_binding")
        require_int(
            self.outer_search_seed,
            field="outer_search_seed",
            maximum=_UINT64_MAX,
        )
        require_int(
            self.maximum_atlas_entries,
            field="maximum_atlas_entries",
            minimum=1,
        )
        require_int(
            self.maximum_retained_parent_entries,
            field="maximum_retained_parent_entries",
            minimum=1,
        )
        require_int(
            self.maximum_retained_parent_bytes,
            field="maximum_retained_parent_bytes",
            minimum=1,
        )
        require_instance(
            self.artifact_lifecycle_policy,
            CanonicalPayloadV1,
            field="artifact_lifecycle_policy",
        )
        _validate_expedition_lifecycle_policy_envelope_v1(
            self.artifact_lifecycle_policy
        )
        _require_text_tuple(
            self.stop_conditions,
            field="stop_conditions",
            nonempty=True,
        )
        if any(
            condition not in SUPPORTED_EXPEDITION_STOP_CONDITIONS_V1
            for condition in self.stop_conditions
        ):
            raise contract_invalid(
                "stop_conditions contains an unsupported or unconsumed name",
                field="stop_conditions",
            )
        if self.schema_version != EXPEDITION_MANIFEST_SCHEMA_VERSION:
            raise contract_invalid("unsupported expedition manifest schema", field="schema_version")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": str(self.schema_version),
            "expedition_id": self.expedition_id,
            "question_description": self.question_description,
            "no_claims": self.no_claims,
            "hypothesis_ids": list(self.hypothesis_ids),
            "required_science_instrument": self.required_science_instrument,
            "founder_basis_refs": [item.to_dict() for item in self.founder_basis_refs],
            "search_space": self.search_space.to_dict(),
            "epoch_ticks": self.epoch_ticks,
            "maximum_absolute_tick": self.maximum_absolute_tick,
            "maximum_frontiers": self.maximum_frontiers,
            "maximum_actions_per_frontier": self.maximum_actions_per_frontier,
            "wall_time_budget_seconds": self.wall_time_budget_seconds,
            "observation_profile_id": self.observation_profile_id,
            "observation_descriptor_set_sha256": (
                self.observation_descriptor_set_sha256.to_dict()
            ),
            "policy_binding": self.policy_binding.to_dict(),
            "outer_search_seed": self.outer_search_seed,
            "maximum_atlas_entries": self.maximum_atlas_entries,
            "maximum_retained_parent_entries": self.maximum_retained_parent_entries,
            "maximum_retained_parent_bytes": self.maximum_retained_parent_bytes,
            "artifact_lifecycle_policy": self.artifact_lifecycle_policy.to_dict(),
            "stop_conditions": list(self.stop_conditions),
        }

    def manifest_sha256(self) -> ContentDigest:
        return canonical_sha256(self.to_dict())

    def founder_basis_sha256(self) -> ContentDigest:
        return canonical_sha256([item.to_dict() for item in self.founder_basis_refs])

    @classmethod
    def from_dict(cls, value: object) -> "ExpeditionManifestV1":
        body = _object(
            value,
            {
                "schema_version",
                "expedition_id",
                "question_description",
                "no_claims",
                "hypothesis_ids",
                "required_science_instrument",
                "founder_basis_refs",
                "search_space",
                "epoch_ticks",
                "maximum_absolute_tick",
                "maximum_frontiers",
                "maximum_actions_per_frontier",
                "wall_time_budget_seconds",
                "observation_profile_id",
                "observation_descriptor_set_sha256",
                "policy_binding",
                "outer_search_seed",
                "maximum_atlas_entries",
                "maximum_retained_parent_entries",
                "maximum_retained_parent_bytes",
                "artifact_lifecycle_policy",
                "stop_conditions",
            },
            field="expedition_manifest",
        )
        founder_rows = _array(body["founder_basis_refs"], field="founder_basis_refs")
        return cls(
            expedition_id=body["expedition_id"],  # type: ignore[arg-type]
            question_description=body["question_description"],  # type: ignore[arg-type]
            no_claims=body["no_claims"],  # type: ignore[arg-type]
            hypothesis_ids=_text_tuple(body["hypothesis_ids"], field="hypothesis_ids"),
            required_science_instrument=body["required_science_instrument"],  # type: ignore[arg-type]
            founder_basis_refs=tuple(
                _artifact_ref(item, field="founder_basis_refs") for item in founder_rows
            ),
            search_space=_payload(body["search_space"], field="search_space"),
            epoch_ticks=body["epoch_ticks"],  # type: ignore[arg-type]
            maximum_absolute_tick=body["maximum_absolute_tick"],  # type: ignore[arg-type]
            maximum_frontiers=body["maximum_frontiers"],  # type: ignore[arg-type]
            maximum_actions_per_frontier=body["maximum_actions_per_frontier"],  # type: ignore[arg-type]
            wall_time_budget_seconds=body["wall_time_budget_seconds"],  # type: ignore[arg-type]
            observation_profile_id=body["observation_profile_id"],  # type: ignore[arg-type]
            observation_descriptor_set_sha256=_digest(
                body["observation_descriptor_set_sha256"],
                field="observation_descriptor_set_sha256",
            ),
            policy_binding=_PolicyBindingV1.from_dict(body["policy_binding"]),
            outer_search_seed=body["outer_search_seed"],  # type: ignore[arg-type]
            maximum_atlas_entries=body["maximum_atlas_entries"],  # type: ignore[arg-type]
            maximum_retained_parent_entries=body["maximum_retained_parent_entries"],  # type: ignore[arg-type]
            maximum_retained_parent_bytes=body["maximum_retained_parent_bytes"],  # type: ignore[arg-type]
            artifact_lifecycle_policy=_payload(
                body["artifact_lifecycle_policy"], field="artifact_lifecycle_policy"
            ),
            stop_conditions=_text_tuple(
                body["stop_conditions"], field="stop_conditions", nonempty=True
            ),
            schema_version=SchemaVersion(body["schema_version"]),  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class ObservationFrameV1:
    expedition_id: str
    world_trajectory_id: str
    logical_world_id: int
    next_absolute_tick: int
    semantic_world_sha256: ContentDigest
    control_history_sha256: ContentDigest
    observation_profile_id: str
    observation_profile_sha256: ContentDigest
    observation_descriptor_set_sha256: ContentDigest
    reducer_implementation_sha256: ContentDigest
    observations: CanonicalPayloadV1
    control_readback: CanonicalPayloadV1
    policy_facing_facts: CanonicalPayloadV1
    descriptor_values: CanonicalPayloadV1
    calibration: CanonicalPayloadV1
    source_state_ref: StateRefV1
    previous_frame_sha256: ContentDigest | None
    schema_version: SchemaVersion = OBSERVATION_FRAME_SCHEMA_VERSION

    def __post_init__(self) -> None:
        require_text(self.expedition_id, field="expedition_id")
        require_text(self.world_trajectory_id, field="world_trajectory_id")
        require_int(self.logical_world_id, field="logical_world_id")
        require_int(self.next_absolute_tick, field="next_absolute_tick")
        for field_name in (
            "semantic_world_sha256",
            "control_history_sha256",
            "observation_profile_sha256",
            "observation_descriptor_set_sha256",
            "reducer_implementation_sha256",
        ):
            require_instance(getattr(self, field_name), ContentDigest, field=field_name)
        require_text(self.observation_profile_id, field="observation_profile_id")
        require_instance(self.observations, CanonicalPayloadV1, field="observations")
        require_instance(
            self.control_readback, CanonicalPayloadV1, field="control_readback"
        )
        require_instance(
            self.policy_facing_facts,
            CanonicalPayloadV1,
            field="policy_facing_facts",
        )
        require_instance(
            self.descriptor_values,
            CanonicalPayloadV1,
            field="descriptor_values",
        )
        require_instance(self.calibration, CanonicalPayloadV1, field="calibration")
        require_instance(self.source_state_ref, StateRefV1, field="source_state_ref")
        if self.previous_frame_sha256 is not None:
            require_instance(
                self.previous_frame_sha256, ContentDigest, field="previous_frame_sha256"
            )
        state = self.source_state_ref
        for field_name in (
            "expedition_id",
            "world_trajectory_id",
            "logical_world_id",
            "next_absolute_tick",
            "semantic_world_sha256",
            "control_history_sha256",
        ):
            if getattr(self, field_name) != getattr(state, field_name):
                raise contract_invalid(
                    "observation identity differs from its source state",
                    field=field_name,
                )
        if self.schema_version != OBSERVATION_FRAME_SCHEMA_VERSION:
            raise contract_invalid("unsupported observation frame schema", field="schema_version")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": str(self.schema_version),
            "expedition_id": self.expedition_id,
            "world_trajectory_id": self.world_trajectory_id,
            "logical_world_id": self.logical_world_id,
            "next_absolute_tick": self.next_absolute_tick,
            "semantic_world_sha256": self.semantic_world_sha256.to_dict(),
            "control_history_sha256": self.control_history_sha256.to_dict(),
            "observation_profile_id": self.observation_profile_id,
            "observation_profile_sha256": self.observation_profile_sha256.to_dict(),
            "observation_descriptor_set_sha256": (
                self.observation_descriptor_set_sha256.to_dict()
            ),
            "reducer_implementation_sha256": self.reducer_implementation_sha256.to_dict(),
            "observations": self.observations.to_dict(),
            "control_readback": self.control_readback.to_dict(),
            "policy_facing_facts": self.policy_facing_facts.to_dict(),
            "descriptor_values": self.descriptor_values.to_dict(),
            "calibration": self.calibration.to_dict(),
            "source_state_ref": self.source_state_ref.to_dict(),
            "previous_frame_sha256": (
                None
                if self.previous_frame_sha256 is None
                else self.previous_frame_sha256.to_dict()
            ),
        }

    def observation_frame_sha256(self) -> ContentDigest:
        return canonical_sha256(self.to_dict())

    @classmethod
    def from_dict(cls, value: object) -> "ObservationFrameV1":
        body = _object(
            value,
            {
                "schema_version",
                "expedition_id",
                "world_trajectory_id",
                "logical_world_id",
                "next_absolute_tick",
                "semantic_world_sha256",
                "control_history_sha256",
                "observation_profile_id",
                "observation_profile_sha256",
                "observation_descriptor_set_sha256",
                "reducer_implementation_sha256",
                "observations",
                "control_readback",
                "policy_facing_facts",
                "descriptor_values",
                "calibration",
                "source_state_ref",
                "previous_frame_sha256",
            },
            field="observation_frame",
        )
        return cls(
            expedition_id=body["expedition_id"],  # type: ignore[arg-type]
            world_trajectory_id=body["world_trajectory_id"],  # type: ignore[arg-type]
            logical_world_id=body["logical_world_id"],  # type: ignore[arg-type]
            next_absolute_tick=body["next_absolute_tick"],  # type: ignore[arg-type]
            semantic_world_sha256=_digest(
                body["semantic_world_sha256"], field="semantic_world_sha256"
            ),
            control_history_sha256=_digest(
                body["control_history_sha256"], field="control_history_sha256"
            ),
            observation_profile_id=body["observation_profile_id"],  # type: ignore[arg-type]
            observation_profile_sha256=_digest(
                body["observation_profile_sha256"], field="observation_profile_sha256"
            ),
            observation_descriptor_set_sha256=_digest(
                body["observation_descriptor_set_sha256"],
                field="observation_descriptor_set_sha256",
            ),
            reducer_implementation_sha256=_digest(
                body["reducer_implementation_sha256"],
                field="reducer_implementation_sha256",
            ),
            observations=_payload(body["observations"], field="observations"),
            control_readback=_payload(
                body["control_readback"], field="control_readback"
            ),
            policy_facing_facts=_payload(
                body["policy_facing_facts"], field="policy_facing_facts"
            ),
            descriptor_values=_payload(
                body["descriptor_values"], field="descriptor_values"
            ),
            calibration=_payload(body["calibration"], field="calibration"),
            source_state_ref=StateRefV1.from_dict(body["source_state_ref"]),
            previous_frame_sha256=_optional_digest(
                body["previous_frame_sha256"], field="previous_frame_sha256"
            ),
            schema_version=SchemaVersion(body["schema_version"]),  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class _ActionRowV1:
    action_id: str
    action_kind: ExperimentActionKindV1
    quota_bucket: str
    parent_state_ref_sha256: ContentDigest | None
    effective_next_absolute_tick: int
    next_epoch_ticks: int | None
    reason_codes: tuple[str, ...]
    action_payload: CanonicalPayloadV1

    def __post_init__(self) -> None:
        require_text(self.action_id, field="action_id")
        require_instance(self.action_kind, ExperimentActionKindV1, field="action_kind")
        require_text(self.quota_bucket, field="quota_bucket")
        if self.parent_state_ref_sha256 is not None:
            require_instance(
                self.parent_state_ref_sha256,
                ContentDigest,
                field="parent_state_ref_sha256",
            )
        require_int(
            self.effective_next_absolute_tick,
            field="effective_next_absolute_tick",
        )
        run_actions = {
            ExperimentActionKindV1.CONTINUE,
            ExperimentActionKindV1.BRANCH_LOCAL,
            ExperimentActionKindV1.EXPLORE_GLOBAL,
            ExperimentActionKindV1.REPLICATE,
            ExperimentActionKindV1.CAUSAL_PROBE,
            ExperimentActionKindV1.TRANSFER_CHALLENGE,
        }
        if self.action_kind in run_actions:
            require_int(self.next_epoch_ticks, field="next_epoch_ticks", minimum=1)
        elif self.next_epoch_ticks is not None:
            raise contract_invalid(
                "non-running action cannot allocate an epoch", field="next_epoch_ticks"
            )
        _require_text_tuple(self.reason_codes, field="reason_codes", nonempty=True)
        require_instance(self.action_payload, CanonicalPayloadV1, field="action_payload")

    def to_dict(self) -> dict[str, object]:
        return {
            "action_id": self.action_id,
            "action_kind": self.action_kind.value,
            "quota_bucket": self.quota_bucket,
            "parent_state_ref_sha256": (
                None
                if self.parent_state_ref_sha256 is None
                else self.parent_state_ref_sha256.to_dict()
            ),
            "effective_next_absolute_tick": self.effective_next_absolute_tick,
            "next_epoch_ticks": self.next_epoch_ticks,
            "reason_codes": list(self.reason_codes),
            "action_payload": self.action_payload.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: object) -> "_ActionRowV1":
        body = _object(
            value,
            {
                "action_id",
                "action_kind",
                "quota_bucket",
                "parent_state_ref_sha256",
                "effective_next_absolute_tick",
                "next_epoch_ticks",
                "reason_codes",
                "action_payload",
            },
            field="action_row",
        )
        return cls(
            action_id=body["action_id"],  # type: ignore[arg-type]
            action_kind=ExperimentActionKindV1(body["action_kind"]),  # type: ignore[arg-type]
            quota_bucket=body["quota_bucket"],  # type: ignore[arg-type]
            parent_state_ref_sha256=_optional_digest(
                body["parent_state_ref_sha256"], field="parent_state_ref_sha256"
            ),
            effective_next_absolute_tick=body["effective_next_absolute_tick"],  # type: ignore[arg-type]
            next_epoch_ticks=body["next_epoch_ticks"],  # type: ignore[arg-type]
            reason_codes=_text_tuple(
                body["reason_codes"], field="reason_codes", nonempty=True
            ),
            action_payload=_payload(body["action_payload"], field="action_payload"),
        )


@dataclass(frozen=True, slots=True)
class DecisionV1:
    expedition_id: str
    frontier_ordinal: int
    manifest_sha256: ContentDigest
    observation_frontier_sha256: ContentDigest
    atlas_before_sha256: ContentDigest
    policy_implementation_sha256: ContentDigest
    policy_profile_sha256: ContentDigest
    outer_rng_before_sha256: ContentDigest
    outer_rng_after_sha256: ContentDigest
    policy_proposal: CanonicalPayloadV1
    policy_proposal_sha256: ContentDigest
    action_rows: tuple[_ActionRowV1, ...]
    integrity_warnings: tuple[str, ...]
    schema_version: SchemaVersion = DECISION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        require_text(self.expedition_id, field="expedition_id")
        require_int(self.frontier_ordinal, field="frontier_ordinal")
        for field_name in (
            "manifest_sha256",
            "observation_frontier_sha256",
            "atlas_before_sha256",
            "policy_implementation_sha256",
            "policy_profile_sha256",
            "outer_rng_before_sha256",
            "outer_rng_after_sha256",
            "policy_proposal_sha256",
        ):
            require_instance(getattr(self, field_name), ContentDigest, field=field_name)
        require_instance(self.policy_proposal, CanonicalPayloadV1, field="policy_proposal")
        if self.policy_proposal_sha256 != self.policy_proposal.content_digest():
            raise contract_invalid(
                "policy proposal digest differs from its canonical policy output",
                field="policy_proposal_sha256",
            )
        if (
            not isinstance(self.action_rows, tuple)
            or any(not isinstance(item, _ActionRowV1) for item in self.action_rows)
        ):
            raise contract_invalid("action rows must be a canonical tuple", field="action_rows")
        action_ids = tuple(item.action_id for item in self.action_rows)
        if len(action_ids) != len(set(action_ids)):
            raise contract_invalid("action IDs must be unique", field="action_rows")
        _require_text_tuple(self.integrity_warnings, field="integrity_warnings")
        if self.schema_version != DECISION_SCHEMA_VERSION:
            raise contract_invalid("unsupported decision schema", field="schema_version")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": str(self.schema_version),
            "expedition_id": self.expedition_id,
            "frontier_ordinal": self.frontier_ordinal,
            "manifest_sha256": self.manifest_sha256.to_dict(),
            "observation_frontier_sha256": self.observation_frontier_sha256.to_dict(),
            "atlas_before_sha256": self.atlas_before_sha256.to_dict(),
            "policy_implementation_sha256": self.policy_implementation_sha256.to_dict(),
            "policy_profile_sha256": self.policy_profile_sha256.to_dict(),
            "outer_rng_before_sha256": self.outer_rng_before_sha256.to_dict(),
            "outer_rng_after_sha256": self.outer_rng_after_sha256.to_dict(),
            "policy_proposal": self.policy_proposal.to_dict(),
            "policy_proposal_sha256": self.policy_proposal_sha256.to_dict(),
            "action_rows": [item.to_dict() for item in self.action_rows],
            "integrity_warnings": list(self.integrity_warnings),
        }

    def decision_sha256(self) -> ContentDigest:
        return canonical_sha256(self.to_dict())

    @classmethod
    def from_dict(cls, value: object) -> "DecisionV1":
        body = _object(
            value,
            {
                "schema_version",
                "expedition_id",
                "frontier_ordinal",
                "manifest_sha256",
                "observation_frontier_sha256",
                "atlas_before_sha256",
                "policy_implementation_sha256",
                "policy_profile_sha256",
                "outer_rng_before_sha256",
                "outer_rng_after_sha256",
                "policy_proposal",
                "policy_proposal_sha256",
                "action_rows",
                "integrity_warnings",
            },
            field="decision",
        )
        return cls(
            expedition_id=body["expedition_id"],  # type: ignore[arg-type]
            frontier_ordinal=body["frontier_ordinal"],  # type: ignore[arg-type]
            manifest_sha256=_digest(body["manifest_sha256"], field="manifest_sha256"),
            observation_frontier_sha256=_digest(
                body["observation_frontier_sha256"], field="observation_frontier_sha256"
            ),
            atlas_before_sha256=_digest(
                body["atlas_before_sha256"], field="atlas_before_sha256"
            ),
            policy_implementation_sha256=_digest(
                body["policy_implementation_sha256"],
                field="policy_implementation_sha256",
            ),
            policy_profile_sha256=_digest(
                body["policy_profile_sha256"], field="policy_profile_sha256"
            ),
            outer_rng_before_sha256=_digest(
                body["outer_rng_before_sha256"], field="outer_rng_before_sha256"
            ),
            outer_rng_after_sha256=_digest(
                body["outer_rng_after_sha256"], field="outer_rng_after_sha256"
            ),
            policy_proposal=_payload(
                body["policy_proposal"], field="policy_proposal"
            ),
            policy_proposal_sha256=_digest(
                body["policy_proposal_sha256"], field="policy_proposal_sha256"
            ),
            action_rows=tuple(
                _ActionRowV1.from_dict(item)
                for item in _array(body["action_rows"], field="action_rows")
            ),
            integrity_warnings=_text_tuple(
                body["integrity_warnings"], field="integrity_warnings"
            ),
            schema_version=SchemaVersion(body["schema_version"]),  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class GeneratedChildContinuationV1:
    expedition_id: str
    manifest_sha256: ContentDigest
    decision_sha256: ContentDigest
    action_id: str
    action_kind: ExperimentActionKindV1
    realization_kind: ChildRealizationKindV1
    parent_state_ref: StateRefV1 | None
    founder_basis_sha256: ContentDigest
    child_trajectory_id: str
    child_run_id: RunId
    retry_of_run_id: None
    effective_next_absolute_tick: int
    control_schedule_sha256: ContentDigest
    child_request: CanonicalPayloadV1
    child_request_sha256: ContentDigest
    schema_version: SchemaVersion = GENERATED_CHILD_CONTINUATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        require_text(self.expedition_id, field="expedition_id")
        for field_name in (
            "manifest_sha256",
            "decision_sha256",
            "founder_basis_sha256",
            "control_schedule_sha256",
            "child_request_sha256",
        ):
            require_instance(getattr(self, field_name), ContentDigest, field=field_name)
        require_text(self.action_id, field="action_id")
        require_instance(self.action_kind, ExperimentActionKindV1, field="action_kind")
        require_instance(
            self.realization_kind, ChildRealizationKindV1, field="realization_kind"
        )
        if self.parent_state_ref is not None:
            require_instance(self.parent_state_ref, StateRefV1, field="parent_state_ref")
        if (
            self.parent_state_ref is not None
            and self.parent_state_ref.expedition_id != self.expedition_id
        ):
            raise contract_invalid(
                "child continuation and parent state belong to different expeditions",
                field="parent_state_ref",
            )
        require_text(self.child_trajectory_id, field="child_trajectory_id")
        require_instance(self.child_run_id, RunId, field="child_run_id")
        if self.retry_of_run_id is not None:
            raise contract_invalid(
                "generated child is a new run occurrence, never a retry",
                field="retry_of_run_id",
            )
        require_int(
            self.effective_next_absolute_tick,
            field="effective_next_absolute_tick",
        )
        if self.realization_kind is ChildRealizationKindV1.FOUNDER_BASIS_REQUEST:
            if self.parent_state_ref is not None:
                raise contract_invalid(
                    "founder-basis child must not fabricate a historical parent state",
                    field="parent_state_ref",
                )
            if self.effective_next_absolute_tick != 0:
                raise contract_invalid(
                    "founder-basis child requests restart scientific time at tick zero",
                    field="effective_next_absolute_tick",
                )
        else:
            if (
                self.parent_state_ref is None
                or not self.parent_state_ref.supports_exact_continuation
            ):
                raise contract_invalid(
                    "exact-state child requires an exactly continuable parent StateRef",
                    field="parent_state_ref",
                )
            if self.effective_next_absolute_tick < self.parent_state_ref.next_absolute_tick:
                raise contract_invalid(
                    "changed controls cannot become effective before the parent branch tick",
                    field="effective_next_absolute_tick",
                )
        require_instance(self.child_request, CanonicalPayloadV1, field="child_request")
        if self.child_request_sha256 != self.child_request.content_digest():
            raise contract_invalid(
                "child request digest does not bind the canonical versioned request",
                field="child_request_sha256",
            )
        child_action_kinds = {
            ExperimentActionKindV1.CONTINUE,
            ExperimentActionKindV1.BRANCH_LOCAL,
            ExperimentActionKindV1.EXPLORE_GLOBAL,
            ExperimentActionKindV1.REPLICATE,
            ExperimentActionKindV1.CAUSAL_PROBE,
            ExperimentActionKindV1.TRANSFER_CHALLENGE,
        }
        if self.action_kind not in child_action_kinds:
            raise contract_invalid(
                "action kind cannot generate a child run", field="action_kind"
            )
        if self.schema_version != GENERATED_CHILD_CONTINUATION_SCHEMA_VERSION:
            raise contract_invalid(
                "unsupported generated child continuation schema", field="schema_version"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": str(self.schema_version),
            "expedition_id": self.expedition_id,
            "manifest_sha256": self.manifest_sha256.to_dict(),
            "decision_sha256": self.decision_sha256.to_dict(),
            "action_id": self.action_id,
            "action_kind": self.action_kind.value,
            "realization_kind": self.realization_kind.value,
            "parent_state_ref": (
                None if self.parent_state_ref is None else self.parent_state_ref.to_dict()
            ),
            "founder_basis_sha256": self.founder_basis_sha256.to_dict(),
            "child_trajectory_id": self.child_trajectory_id,
            "child_run_id": str(self.child_run_id),
            "retry_of_run_id": self.retry_of_run_id,
            "effective_next_absolute_tick": self.effective_next_absolute_tick,
            "control_schedule_sha256": self.control_schedule_sha256.to_dict(),
            "child_request": self.child_request.to_dict(),
            "child_request_sha256": self.child_request_sha256.to_dict(),
        }

    def continuation_sha256(self) -> ContentDigest:
        return canonical_sha256(self.to_dict())

    @classmethod
    def from_dict(cls, value: object) -> "GeneratedChildContinuationV1":
        body = _object(
            value,
            {
                "schema_version",
                "expedition_id",
                "manifest_sha256",
                "decision_sha256",
                "action_id",
                "action_kind",
                "realization_kind",
                "parent_state_ref",
                "founder_basis_sha256",
                "child_trajectory_id",
                "child_run_id",
                "retry_of_run_id",
                "effective_next_absolute_tick",
                "control_schedule_sha256",
                "child_request",
                "child_request_sha256",
            },
            field="generated_child_continuation",
        )
        return cls(
            expedition_id=body["expedition_id"],  # type: ignore[arg-type]
            manifest_sha256=_digest(body["manifest_sha256"], field="manifest_sha256"),
            decision_sha256=_digest(body["decision_sha256"], field="decision_sha256"),
            action_id=body["action_id"],  # type: ignore[arg-type]
            action_kind=ExperimentActionKindV1(body["action_kind"]),  # type: ignore[arg-type]
            realization_kind=ChildRealizationKindV1(body["realization_kind"]),  # type: ignore[arg-type]
            parent_state_ref=(
                None
                if body["parent_state_ref"] is None
                else StateRefV1.from_dict(body["parent_state_ref"])
            ),
            founder_basis_sha256=_digest(
                body["founder_basis_sha256"], field="founder_basis_sha256"
            ),
            child_trajectory_id=body["child_trajectory_id"],  # type: ignore[arg-type]
            child_run_id=RunId(body["child_run_id"]),  # type: ignore[arg-type]
            retry_of_run_id=body["retry_of_run_id"],  # type: ignore[arg-type]
            effective_next_absolute_tick=body["effective_next_absolute_tick"],  # type: ignore[arg-type]
            control_schedule_sha256=_digest(
                body["control_schedule_sha256"], field="control_schedule_sha256"
            ),
            child_request=_payload(body["child_request"], field="child_request"),
            child_request_sha256=_digest(
                body["child_request_sha256"], field="child_request_sha256"
            ),
            schema_version=SchemaVersion(body["schema_version"]),  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class JournalRecordV1:
    expedition_id: str
    record_sequence: int
    frontier_ordinal: int
    record_kind: JournalRecordKindV1
    manifest_sha256: ContentDigest
    subject_sha256: ContentDigest
    payload: CanonicalPayloadV1
    payload_sha256: ContentDigest
    previous_record_sha256: ContentDigest | None
    schema_version: SchemaVersion = JOURNAL_RECORD_SCHEMA_VERSION

    def __post_init__(self) -> None:
        require_text(self.expedition_id, field="expedition_id")
        require_int(self.record_sequence, field="record_sequence")
        require_int(self.frontier_ordinal, field="frontier_ordinal")
        require_instance(self.record_kind, JournalRecordKindV1, field="record_kind")
        for field_name in ("manifest_sha256", "subject_sha256", "payload_sha256"):
            require_instance(getattr(self, field_name), ContentDigest, field=field_name)
        require_instance(self.payload, CanonicalPayloadV1, field="payload")
        if self.payload_sha256 != self.payload.content_digest():
            raise contract_invalid(
                "journal payload digest differs from its canonical payload",
                field="payload_sha256",
            )
        if self.previous_record_sha256 is not None:
            require_instance(
                self.previous_record_sha256,
                ContentDigest,
                field="previous_record_sha256",
            )
        if (self.record_sequence == 0) != (self.previous_record_sha256 is None):
            raise contract_invalid(
                "only the first journal record may omit its predecessor",
                field="previous_record_sha256",
            )
        if self.schema_version != JOURNAL_RECORD_SCHEMA_VERSION:
            raise contract_invalid("unsupported journal record schema", field="schema_version")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": str(self.schema_version),
            "expedition_id": self.expedition_id,
            "record_sequence": self.record_sequence,
            "frontier_ordinal": self.frontier_ordinal,
            "record_kind": self.record_kind.value,
            "manifest_sha256": self.manifest_sha256.to_dict(),
            "subject_sha256": self.subject_sha256.to_dict(),
            "payload": self.payload.to_dict(),
            "payload_sha256": self.payload_sha256.to_dict(),
            "previous_record_sha256": (
                None
                if self.previous_record_sha256 is None
                else self.previous_record_sha256.to_dict()
            ),
        }

    def record_sha256(self) -> ContentDigest:
        return canonical_sha256(self.to_dict())

    @classmethod
    def from_dict(cls, value: object) -> "JournalRecordV1":
        body = _object(
            value,
            {
                "schema_version",
                "expedition_id",
                "record_sequence",
                "frontier_ordinal",
                "record_kind",
                "manifest_sha256",
                "subject_sha256",
                "payload",
                "payload_sha256",
                "previous_record_sha256",
            },
            field="journal_record",
        )
        return cls(
            expedition_id=body["expedition_id"],  # type: ignore[arg-type]
            record_sequence=body["record_sequence"],  # type: ignore[arg-type]
            frontier_ordinal=body["frontier_ordinal"],  # type: ignore[arg-type]
            record_kind=JournalRecordKindV1(body["record_kind"]),  # type: ignore[arg-type]
            manifest_sha256=_digest(body["manifest_sha256"], field="manifest_sha256"),
            subject_sha256=_digest(body["subject_sha256"], field="subject_sha256"),
            payload=_payload(body["payload"], field="payload"),
            payload_sha256=_digest(body["payload_sha256"], field="payload_sha256"),
            previous_record_sha256=_optional_digest(
                body["previous_record_sha256"], field="previous_record_sha256"
            ),
            schema_version=SchemaVersion(body["schema_version"]),  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class _AtlasEntryV1:
    atlas_entry_id: str
    world_trajectory_id: str
    observation_frame_sha256: ContentDigest
    state_ref_sha256: ContentDigest | None
    niche_key: CanonicalPayloadV1
    descriptor_values: CanonicalPayloadV1
    policy_facing_facts: CanonicalPayloadV1
    representative_roles: tuple[str, ...]
    parent_entry_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        require_text(self.atlas_entry_id, field="atlas_entry_id")
        require_text(self.world_trajectory_id, field="world_trajectory_id")
        require_instance(
            self.observation_frame_sha256,
            ContentDigest,
            field="observation_frame_sha256",
        )
        if self.state_ref_sha256 is not None:
            require_instance(self.state_ref_sha256, ContentDigest, field="state_ref_sha256")
        require_instance(self.niche_key, CanonicalPayloadV1, field="niche_key")
        require_instance(
            self.descriptor_values, CanonicalPayloadV1, field="descriptor_values"
        )
        require_instance(
            self.policy_facing_facts,
            CanonicalPayloadV1,
            field="policy_facing_facts",
        )
        _require_text_tuple(
            self.representative_roles,
            field="representative_roles",
            nonempty=True,
        )
        _require_text_tuple(self.parent_entry_ids, field="parent_entry_ids")

    def to_dict(self) -> dict[str, object]:
        return {
            "atlas_entry_id": self.atlas_entry_id,
            "world_trajectory_id": self.world_trajectory_id,
            "observation_frame_sha256": self.observation_frame_sha256.to_dict(),
            "state_ref_sha256": (
                None if self.state_ref_sha256 is None else self.state_ref_sha256.to_dict()
            ),
            "niche_key": self.niche_key.to_dict(),
            "descriptor_values": self.descriptor_values.to_dict(),
            "policy_facing_facts": self.policy_facing_facts.to_dict(),
            "representative_roles": list(self.representative_roles),
            "parent_entry_ids": list(self.parent_entry_ids),
        }

    @classmethod
    def from_dict(cls, value: object) -> "_AtlasEntryV1":
        body = _object(
            value,
            {
                "atlas_entry_id",
                "world_trajectory_id",
                "observation_frame_sha256",
                "state_ref_sha256",
                "niche_key",
                "descriptor_values",
                "policy_facing_facts",
                "representative_roles",
                "parent_entry_ids",
            },
            field="atlas_entry",
        )
        return cls(
            atlas_entry_id=body["atlas_entry_id"],  # type: ignore[arg-type]
            world_trajectory_id=body["world_trajectory_id"],  # type: ignore[arg-type]
            observation_frame_sha256=_digest(
                body["observation_frame_sha256"], field="observation_frame_sha256"
            ),
            state_ref_sha256=_optional_digest(
                body["state_ref_sha256"], field="state_ref_sha256"
            ),
            niche_key=_payload(body["niche_key"], field="niche_key"),
            descriptor_values=_payload(
                body["descriptor_values"], field="descriptor_values"
            ),
            policy_facing_facts=_payload(
                body["policy_facing_facts"], field="policy_facing_facts"
            ),
            representative_roles=_text_tuple(
                body["representative_roles"],
                field="representative_roles",
                nonempty=True,
            ),
            parent_entry_ids=_text_tuple(
                body["parent_entry_ids"], field="parent_entry_ids"
            ),
        )


@dataclass(frozen=True, slots=True)
class EvolutionaryAtlasV1:
    expedition_id: str
    manifest_sha256: ContentDigest
    revision: int
    through_frontier_ordinal: int
    descriptor_set_sha256: ContentDigest
    maximum_entries: int
    entries: tuple[_AtlasEntryV1, ...]
    previous_atlas_sha256: ContentDigest | None
    schema_version: SchemaVersion = EVOLUTIONARY_ATLAS_SCHEMA_VERSION

    def __post_init__(self) -> None:
        require_text(self.expedition_id, field="expedition_id")
        require_instance(self.manifest_sha256, ContentDigest, field="manifest_sha256")
        require_int(self.revision, field="revision")
        require_int(self.through_frontier_ordinal, field="through_frontier_ordinal")
        require_instance(
            self.descriptor_set_sha256, ContentDigest, field="descriptor_set_sha256"
        )
        require_int(self.maximum_entries, field="maximum_entries", minimum=1)
        if (
            not isinstance(self.entries, tuple)
            or any(not isinstance(item, _AtlasEntryV1) for item in self.entries)
        ):
            raise contract_invalid("atlas entries must be a canonical tuple", field="entries")
        entry_ids = tuple(item.atlas_entry_id for item in self.entries)
        if entry_ids != tuple(sorted(set(entry_ids))):
            raise contract_invalid(
                "atlas entries must be sorted and unique by atlas_entry_id",
                field="entries",
            )
        if len(self.entries) > self.maximum_entries:
            raise contract_invalid("atlas entry budget exceeded", field="entries")
        if self.previous_atlas_sha256 is not None:
            require_instance(
                self.previous_atlas_sha256,
                ContentDigest,
                field="previous_atlas_sha256",
            )
        if (self.revision == 0) != (self.previous_atlas_sha256 is None):
            raise contract_invalid(
                "only atlas revision zero may omit its predecessor",
                field="previous_atlas_sha256",
            )
        if self.schema_version != EVOLUTIONARY_ATLAS_SCHEMA_VERSION:
            raise contract_invalid("unsupported evolutionary atlas schema", field="schema_version")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": str(self.schema_version),
            "expedition_id": self.expedition_id,
            "manifest_sha256": self.manifest_sha256.to_dict(),
            "revision": self.revision,
            "through_frontier_ordinal": self.through_frontier_ordinal,
            "descriptor_set_sha256": self.descriptor_set_sha256.to_dict(),
            "maximum_entries": self.maximum_entries,
            "entries": [item.to_dict() for item in self.entries],
            "previous_atlas_sha256": (
                None
                if self.previous_atlas_sha256 is None
                else self.previous_atlas_sha256.to_dict()
            ),
        }

    def atlas_sha256(self) -> ContentDigest:
        return canonical_sha256(self.to_dict())

    @classmethod
    def from_dict(cls, value: object) -> "EvolutionaryAtlasV1":
        body = _object(
            value,
            {
                "schema_version",
                "expedition_id",
                "manifest_sha256",
                "revision",
                "through_frontier_ordinal",
                "descriptor_set_sha256",
                "maximum_entries",
                "entries",
                "previous_atlas_sha256",
            },
            field="evolutionary_atlas",
        )
        return cls(
            expedition_id=body["expedition_id"],  # type: ignore[arg-type]
            manifest_sha256=_digest(body["manifest_sha256"], field="manifest_sha256"),
            revision=body["revision"],  # type: ignore[arg-type]
            through_frontier_ordinal=body["through_frontier_ordinal"],  # type: ignore[arg-type]
            descriptor_set_sha256=_digest(
                body["descriptor_set_sha256"], field="descriptor_set_sha256"
            ),
            maximum_entries=body["maximum_entries"],  # type: ignore[arg-type]
            entries=tuple(
                _AtlasEntryV1.from_dict(item)
                for item in _array(body["entries"], field="entries")
            ),
            previous_atlas_sha256=_optional_digest(
                body["previous_atlas_sha256"], field="previous_atlas_sha256"
            ),
            schema_version=SchemaVersion(body["schema_version"]),  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class _RetainedParentEntryV1:
    state_ref: StateRefV1
    selected_at_frontier_ordinal: int
    reason_codes: tuple[str, ...]
    retained_bytes: int

    def __post_init__(self) -> None:
        require_instance(self.state_ref, StateRefV1, field="state_ref")
        if not self.state_ref.supports_exact_continuation:
            raise contract_invalid(
                "retained parent must be exactly continuable",
                field="state_ref",
            )
        require_int(
            self.selected_at_frontier_ordinal,
            field="selected_at_frontier_ordinal",
        )
        _require_text_tuple(self.reason_codes, field="reason_codes", nonempty=True)
        require_int(self.retained_bytes, field="retained_bytes", minimum=1)
        artifact_bytes = self.state_ref.source_artifact_ref.size_bytes
        if artifact_bytes is None or self.retained_bytes != artifact_bytes:
            raise contract_invalid(
                "retained byte accounting must equal the wrapped artifact size",
                field="retained_bytes",
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "state_ref": self.state_ref.to_dict(),
            "selected_at_frontier_ordinal": self.selected_at_frontier_ordinal,
            "reason_codes": list(self.reason_codes),
            "retained_bytes": self.retained_bytes,
        }

    @classmethod
    def from_dict(cls, value: object) -> "_RetainedParentEntryV1":
        body = _object(
            value,
            {
                "state_ref",
                "selected_at_frontier_ordinal",
                "reason_codes",
                "retained_bytes",
            },
            field="retained_parent_entry",
        )
        return cls(
            state_ref=StateRefV1.from_dict(body["state_ref"]),
            selected_at_frontier_ordinal=body["selected_at_frontier_ordinal"],  # type: ignore[arg-type]
            reason_codes=_text_tuple(
                body["reason_codes"], field="reason_codes", nonempty=True
            ),
            retained_bytes=body["retained_bytes"],  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class RetainedParentBankV1:
    expedition_id: str
    manifest_sha256: ContentDigest
    revision: int
    maximum_entries: int
    maximum_total_bytes: int
    current_entry_count: int
    current_total_bytes: int
    entries: tuple[_RetainedParentEntryV1, ...]
    previous_bank_sha256: ContentDigest | None
    schema_version: SchemaVersion = RETAINED_PARENT_BANK_SCHEMA_VERSION

    def __post_init__(self) -> None:
        require_text(self.expedition_id, field="expedition_id")
        require_instance(self.manifest_sha256, ContentDigest, field="manifest_sha256")
        require_int(self.revision, field="revision")
        require_int(self.maximum_entries, field="maximum_entries", minimum=1)
        require_int(self.maximum_total_bytes, field="maximum_total_bytes", minimum=1)
        require_int(self.current_entry_count, field="current_entry_count")
        require_int(self.current_total_bytes, field="current_total_bytes")
        if (
            not isinstance(self.entries, tuple)
            or any(not isinstance(item, _RetainedParentEntryV1) for item in self.entries)
        ):
            raise contract_invalid(
                "retained parent entries must be a canonical tuple", field="entries"
            )
        ordered = tuple(sorted(self.entries, key=lambda item: str(item.state_ref.state_ref_sha256())))
        if self.entries != ordered:
            raise contract_invalid(
                "retained parent entries must be ordered by StateRef digest",
                field="entries",
            )
        state_ids = tuple(str(item.state_ref.state_ref_sha256()) for item in self.entries)
        if len(state_ids) != len(set(state_ids)):
            raise contract_invalid("retained StateRefs must be unique", field="entries")
        calculated_bytes = sum(item.retained_bytes for item in self.entries)
        if self.current_entry_count != len(self.entries):
            raise contract_invalid(
                "current entry count differs from bank contents",
                field="current_entry_count",
            )
        if self.current_total_bytes != calculated_bytes:
            raise contract_invalid(
                "current byte count differs from bank contents",
                field="current_total_bytes",
            )
        if self.current_entry_count > self.maximum_entries:
            raise contract_invalid(
                "manifest retained-parent entry cap exceeded",
                field="current_entry_count",
            )
        if self.current_total_bytes > self.maximum_total_bytes:
            raise contract_invalid(
                "manifest retained-parent byte cap exceeded",
                field="current_total_bytes",
            )
        if self.previous_bank_sha256 is not None:
            require_instance(
                self.previous_bank_sha256, ContentDigest, field="previous_bank_sha256"
            )
        if (self.revision == 0) != (self.previous_bank_sha256 is None):
            raise contract_invalid(
                "only parent-bank revision zero may omit its predecessor",
                field="previous_bank_sha256",
            )
        if self.schema_version != RETAINED_PARENT_BANK_SCHEMA_VERSION:
            raise contract_invalid(
                "unsupported retained parent bank schema", field="schema_version"
            )

    def validate_against_manifest(self, manifest: ExpeditionManifestV1) -> None:
        require_instance(manifest, ExpeditionManifestV1, field="manifest")
        if self.expedition_id != manifest.expedition_id:
            raise contract_invalid("bank expedition differs from manifest", field="expedition_id")
        if self.manifest_sha256 != manifest.manifest_sha256():
            raise contract_invalid("bank does not bind the manifest", field="manifest_sha256")
        if self.maximum_entries != manifest.maximum_retained_parent_entries:
            raise contract_invalid(
                "bank entry cap differs from manifest hard cap", field="maximum_entries"
            )
        if self.maximum_total_bytes != manifest.maximum_retained_parent_bytes:
            raise contract_invalid(
                "bank byte cap differs from manifest hard cap", field="maximum_total_bytes"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": str(self.schema_version),
            "expedition_id": self.expedition_id,
            "manifest_sha256": self.manifest_sha256.to_dict(),
            "revision": self.revision,
            "maximum_entries": self.maximum_entries,
            "maximum_total_bytes": self.maximum_total_bytes,
            "current_entry_count": self.current_entry_count,
            "current_total_bytes": self.current_total_bytes,
            "entries": [item.to_dict() for item in self.entries],
            "previous_bank_sha256": (
                None
                if self.previous_bank_sha256 is None
                else self.previous_bank_sha256.to_dict()
            ),
        }

    def bank_sha256(self) -> ContentDigest:
        return canonical_sha256(self.to_dict())

    @classmethod
    def from_dict(cls, value: object) -> "RetainedParentBankV1":
        body = _object(
            value,
            {
                "schema_version",
                "expedition_id",
                "manifest_sha256",
                "revision",
                "maximum_entries",
                "maximum_total_bytes",
                "current_entry_count",
                "current_total_bytes",
                "entries",
                "previous_bank_sha256",
            },
            field="retained_parent_bank",
        )
        return cls(
            expedition_id=body["expedition_id"],  # type: ignore[arg-type]
            manifest_sha256=_digest(body["manifest_sha256"], field="manifest_sha256"),
            revision=body["revision"],  # type: ignore[arg-type]
            maximum_entries=body["maximum_entries"],  # type: ignore[arg-type]
            maximum_total_bytes=body["maximum_total_bytes"],  # type: ignore[arg-type]
            current_entry_count=body["current_entry_count"],  # type: ignore[arg-type]
            current_total_bytes=body["current_total_bytes"],  # type: ignore[arg-type]
            entries=tuple(
                _RetainedParentEntryV1.from_dict(item)
                for item in _array(body["entries"], field="entries")
            ),
            previous_bank_sha256=_optional_digest(
                body["previous_bank_sha256"], field="previous_bank_sha256"
            ),
            schema_version=SchemaVersion(body["schema_version"]),  # type: ignore[arg-type]
        )


__all__ = (
    "DecisionV1",
    "ExpeditionManifestV1",
    "EvolutionaryAtlasV1",
    "GeneratedChildContinuationV1",
    "JournalRecordV1",
    "ObservationFrameV1",
    "RetainedParentBankV1",
    "StateRefV1",
)
