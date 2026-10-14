"""Policy-bound, compact human review indexes for terminal Expeditions.

The coordinator supplies authenticated identities and counts.  A frozen policy
profile supplies the opaque role/reason-to-section presentation mapping and the
short recommendation rows.  This module never assigns scientific meaning to a
role, descriptor, niche, action, or retention reason.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from typing import Any, Final

from emergence_contracts import canonical_json_bytes
from emergence_contracts.adaptive_science_expedition_v1 import (
    EvolutionaryAtlasV1,
    ExpeditionManifestV1,
    RetainedParentBankV1,
)
from emergence_contracts.canonical import canonical_sha256

from .ledger import reconstruct_expedition_frontier_v1
from .policy_protocol import canonical_plain_value_v1

POLICY_REVIEW_PRESENTATION_SCHEMA_V1: Final[str] = (
    "eiel_adaptive_policy_review_presentation_v1"
)
REVIEW_SECTION_SCHEMA_V1: Final[str] = "eiel_adaptive_expedition_review_section_v1"
REVIEW_SECTION_KEYS_V1: Final[tuple[str, ...]] = (
    "authorized_search_space",
    "exploration_coverage",
    "trivial_or_unexpressed_regions",
    "overwhelming_extinction_or_dead_end_regions",
    "response_or_frontier_regions",
    "novel_persistent_or_divergent_observations",
    "failed_replication_or_causal_followups",
    "retained_parents",
    "discarded_parent_candidates",
    "unresolved_scientific_questions",
    "limiting_controls_observations_capabilities_or_budgets",
    "recommended_next_expedition",
    "human_scientific_choices",
)
MAX_INDEX_IDENTITIES_PER_ROW_V1: Final[int] = 24
MAX_RETAINED_PARENT_ROWS_V1: Final[int] = 64
MAX_DISCARD_REASON_ROWS_V1: Final[int] = 128
MAX_CONTROL_COORDINATE_ROWS_V1: Final[int] = 64
MAX_REVIEW_ROWS_PER_SECTION_V1: Final[int] = 256
MAX_REVIEW_ROW_BYTES_V1: Final[int] = 16 * 1024
MAX_EVIDENCE_REFS_PER_SECTION_V1: Final[int] = 8
MAX_REVIEW_TEXT_ARRAY_ITEMS_V1: Final[int] = 128
_FORBIDDEN_REVIEW_FIELD_FRAGMENTS_V1: Final[tuple[str, ...]] = (
    "claim",
    "qualification",
    "authority",
)
_EVIDENCE_REF_PATHS_V1: Final[Mapping[str, str]] = {
    "EXPEDITION_MANIFEST": "../control/expedition_manifest.json",
    "EXPEDITION_LEDGER": "../durable/ledger",
    "EVOLUTIONARY_ATLAS": "evolutionary_atlas.json",
    "RETAINED_PARENT_BANK": "retained_parent_bank.json",
    "EXPEDITION_JOURNAL": "expedition_journal.json",
}
_REQUIRED_TEXT_ROW_FIELDS_V1: Final[frozenset[str]] = frozenset(
    {
        "question_ref",
        "search_space_ref",
        "complete_search_space_ref",
        "complete_history_ref",
        "complete_parent_bank_ref",
        "control_coordinate",
        "representative_role",
        "reason_code",
        "world_trajectory_id",
        "source_artifact_id",
        "question_description",
        "resolution_status",
        "programme_stage",
        "terminal_status",
        "required_science_instrument",
        "observation_profile_id",
        "recommendation_id",
        "recommendation",
        "choice_id",
        "choice",
    }
)
_OPTIONAL_TEXT_ROW_FIELDS_V1: Final[frozenset[str]] = frozenset(
    {"stop_reason", "trigger_section"}
)
_EXACT_REVIEW_PATH_FIELDS_V1: Final[Mapping[str, str]] = {
    "question_ref": "../control/expedition_manifest.json#/question_description",
    "search_space_ref": "../control/expedition_manifest.json#/search_space",
    "complete_search_space_ref": "../control/expedition_manifest.json",
    "complete_history_ref": "expedition_journal.json",
    "complete_parent_bank_ref": "retained_parent_bank.json",
}
_REVIEW_ROW_FIELDS_V1: Final[Mapping[str, frozenset[str]]] = {
    "AUTHORIZED_MANIFEST_INDEX": frozenset(
        {
            "row_kind",
            "manifest_sha256",
            "question_ref",
            "hypothesis_ids",
            "founder_basis_sha256",
            "founder_artifact_ids",
            "search_space_sha256",
            "search_space_ref",
            "epoch_ticks",
            "maximum_absolute_tick",
            "maximum_frontiers",
            "maximum_actions_per_frontier",
            "wall_time_budget_seconds",
            "stop_conditions",
        }
    ),
    "EXPERIMENT_DAG_AND_COVERAGE_INDEX": frozenset(
        {
            "row_kind",
            "frontier_count",
            "decision_count",
            "proposed_action_kind_counts",
            "dispatched_action_count",
            "settled_action_count",
            "atlas_entry_count",
            "journal_record_count",
            "observation_frame_sha256_samples",
            "observation_frame_sha256_omitted",
            "decision_sha256_samples",
            "decision_sha256_omitted",
            "action_id_samples",
            "action_ids_omitted",
            "complete_history_ref",
        }
    ),
    "OPAQUE_CONTROL_COORDINATE_COVERAGE_INDEX": frozenset(
        {
            "row_kind",
            "control_coordinate",
            "declared_in_authorized_search_space",
            "authorized_value_count",
            "authorized_value_sha256_samples",
            "authorized_value_sha256s_omitted",
            "selected_value_count",
            "selected_value_sha256_samples",
            "selected_value_sha256s_omitted",
            "action_binding_count",
        }
    ),
    "OPAQUE_CONTROL_COORDINATE_INDEX_OMISSION": frozenset(
        {
            "row_kind",
            "omitted_control_coordinate_count",
            "complete_search_space_ref",
            "complete_history_ref",
        }
    ),
    "POLICY_ATLAS_ROLE_INDEX": frozenset(
        {
            "row_kind",
            "representative_role",
            "entry_count",
            "atlas_entry_id_samples",
            "atlas_entry_ids_omitted",
            "observation_frame_sha256_samples",
            "observation_frame_sha256s_omitted",
            "state_ref_sha256_samples",
            "state_ref_sha256s_omitted",
        }
    ),
    "POLICY_ACTION_REASON_INDEX": frozenset(
        {
            "row_kind",
            "reason_code",
            "action_count",
            "action_id_samples",
            "action_ids_omitted",
        }
    ),
    "EXACT_RETAINED_PARENT_INDEX": frozenset(
        {
            "row_kind",
            "state_ref_sha256",
            "world_trajectory_id",
            "logical_world_id",
            "next_absolute_tick",
            "exact_state_sha256",
            "checkpoint_occurrence_sha256",
            "source_artifact_id",
            "source_object_sha256",
            "selected_at_frontier_ordinal",
            "reason_codes",
            "retained_bytes",
        }
    ),
    "RETAINED_PARENT_INDEX_OMISSION": frozenset(
        {
            "row_kind",
            "omitted_parent_count",
            "complete_parent_bank_ref",
        }
    ),
    "POLICY_PARENT_DISCARD_REASON_INDEX": frozenset(
        {
            "row_kind",
            "reason_code",
            "distinct_state_count",
            "state_ref_id_samples",
            "state_ref_ids_omitted",
            "frontier_ordinal_samples",
            "frontier_ordinals_omitted",
            "decision_sha256_samples",
            "decision_sha256s_omitted",
        }
    ),
    "DISCARDED_PARENT_REASON_INDEX_OMISSION": frozenset(
        {
            "row_kind",
            "omitted_reason_count",
            "complete_history_ref",
        }
    ),
    "MANIFEST_QUESTION_UNRESOLVED_NO_CLAIMS": frozenset(
        {
            "row_kind",
            "question_description",
            "hypothesis_ids",
            "resolution_status",
        }
    ),
    "EXPEDITION_BOUNDARY_INDEX": frozenset(
        {
            "row_kind",
            "programme_stage",
            "terminal_status",
            "stop_reason",
            "selected_world_historical_continuation_available",
            "required_science_instrument",
            "observation_profile_id",
            "observation_descriptor_set_sha256",
            "maximum_absolute_tick",
            "maximum_frontiers",
            "maximum_actions_per_frontier",
            "maximum_atlas_entries",
            "maximum_retained_parent_entries",
            "maximum_retained_parent_bytes",
        }
    ),
    "POLICY_PRESENTATION_RECOMMENDATION": frozenset(
        {
            "row_kind",
            "recommendation_id",
            "recommendation",
            "trigger_section",
            "requires_new_immutable_manifest",
            "automatic_continuation",
        }
    ),
    "POLICY_PRESENTATION_HUMAN_CHOICE": frozenset(
        {"row_kind", "choice_id", "choice", "must_remain_human"}
    ),
}
_POLICY_INDEX_SECTIONS_V1: Final[frozenset[str]] = frozenset(
    {
        "trivial_or_unexpressed_regions",
        "overwhelming_extinction_or_dead_end_regions",
        "response_or_frontier_regions",
        "novel_persistent_or_divergent_observations",
        "failed_replication_or_causal_followups",
        "unresolved_scientific_questions",
        "limiting_controls_observations_capabilities_or_budgets",
    }
)
_ALLOWED_ROW_KINDS_BY_SECTION_V1: Final[Mapping[str, frozenset[str]]] = {
    "authorized_search_space": frozenset({"AUTHORIZED_MANIFEST_INDEX"}),
    "exploration_coverage": frozenset(
        {
            "EXPERIMENT_DAG_AND_COVERAGE_INDEX",
            "OPAQUE_CONTROL_COORDINATE_COVERAGE_INDEX",
            "OPAQUE_CONTROL_COORDINATE_INDEX_OMISSION",
        }
    ),
    "trivial_or_unexpressed_regions": frozenset(
        {"POLICY_ATLAS_ROLE_INDEX", "POLICY_ACTION_REASON_INDEX"}
    ),
    "overwhelming_extinction_or_dead_end_regions": frozenset(
        {"POLICY_ATLAS_ROLE_INDEX", "POLICY_ACTION_REASON_INDEX"}
    ),
    "response_or_frontier_regions": frozenset(
        {"POLICY_ATLAS_ROLE_INDEX", "POLICY_ACTION_REASON_INDEX"}
    ),
    "novel_persistent_or_divergent_observations": frozenset(
        {"POLICY_ATLAS_ROLE_INDEX", "POLICY_ACTION_REASON_INDEX"}
    ),
    "failed_replication_or_causal_followups": frozenset(
        {"POLICY_ATLAS_ROLE_INDEX", "POLICY_ACTION_REASON_INDEX"}
    ),
    "retained_parents": frozenset(
        {"EXACT_RETAINED_PARENT_INDEX", "RETAINED_PARENT_INDEX_OMISSION"}
    ),
    "discarded_parent_candidates": frozenset(
        {
            "POLICY_PARENT_DISCARD_REASON_INDEX",
            "DISCARDED_PARENT_REASON_INDEX_OMISSION",
        }
    ),
    "unresolved_scientific_questions": frozenset(
        {
            "POLICY_ATLAS_ROLE_INDEX",
            "POLICY_ACTION_REASON_INDEX",
            "MANIFEST_QUESTION_UNRESOLVED_NO_CLAIMS",
        }
    ),
    "limiting_controls_observations_capabilities_or_budgets": frozenset(
        {
            "POLICY_ATLAS_ROLE_INDEX",
            "POLICY_ACTION_REASON_INDEX",
            "EXPEDITION_BOUNDARY_INDEX",
        }
    ),
    "recommended_next_expedition": frozenset({"POLICY_PRESENTATION_RECOMMENDATION"}),
    "human_scientific_choices": frozenset({"POLICY_PRESENTATION_HUMAN_CHOICE"}),
}


class ExpeditionReviewProjectionError(ValueError):
    """A policy presentation or review index is incomplete or unsafe."""


def _plain(value: object, *, field: str) -> Any:
    try:
        detached = canonical_plain_value_v1(value)
        return json.loads(canonical_json_bytes(detached))
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise ExpeditionReviewProjectionError(
            f"{field} is not canonical JSON"
        ) from error


def _mapping(value: object, *, field: str) -> dict[str, Any]:
    plain = _plain(value, field=field)
    if type(plain) is not dict:
        raise ExpeditionReviewProjectionError(f"{field} must be a mapping")
    return plain


def _sequence(value: object, *, field: str) -> list[Any]:
    plain = _plain(value, field=field)
    if type(plain) is not list:
        raise ExpeditionReviewProjectionError(f"{field} must be a sequence")
    return plain


def _text(value: object, *, field: str) -> str:
    if type(value) is not str or not value or len(value) > 4096:
        raise ExpeditionReviewProjectionError(f"{field} must be bounded non-empty text")
    return value


def _reject_forbidden_review_fields_v1(value: object, *, path: str) -> None:
    if type(value) is dict:
        for key, item in value.items():
            if type(key) is not str:
                raise ExpeditionReviewProjectionError(
                    f"{path} contains a non-text field"
                )
            lowered = key.lower()
            if any(
                fragment in lowered for fragment in _FORBIDDEN_REVIEW_FIELD_FRAGMENTS_V1
            ):
                raise ExpeditionReviewProjectionError(
                    f"{path}.{key} is forbidden in human-review presentation data"
                )
            _reject_forbidden_review_fields_v1(item, path=f"{path}.{key}")
    elif type(value) is list:
        for index, item in enumerate(value):
            _reject_forbidden_review_fields_v1(item, path=f"{path}[{index}]")


def _sha256_digest_v1(value: object, *, field: str) -> dict[str, str]:
    digest = _mapping(value, field=field)
    hexadecimal = digest.get("hex")
    if (
        set(digest) != {"algorithm", "hex"}
        or digest.get("algorithm") != "sha256"
        or type(hexadecimal) is not str
        or len(hexadecimal) != 64
        or any(character not in "0123456789abcdef" for character in hexadecimal)
    ):
        raise ExpeditionReviewProjectionError(f"{field} is not SHA-256")
    return {"algorithm": "sha256", "hex": hexadecimal}


def _validate_evidence_ref_v1(value: object, *, field: str) -> dict[str, Any]:
    ref = _mapping(value, field=field)
    artifact_id = ref.get("artifact_id")
    expected_path = (
        None
        if type(artifact_id) is not str
        else _EVIDENCE_REF_PATHS_V1.get(artifact_id)
    )
    expected_fields = {"artifact_id", "relative_path", "content_sha256"}
    if artifact_id == "EXPEDITION_LEDGER":
        expected_fields.add("head_event_sha256")
    if (
        set(ref) != expected_fields
        or expected_path is None
        or ref.get("relative_path") != expected_path
    ):
        raise ExpeditionReviewProjectionError(
            f"{field} fields or direct artifact path differ"
        )
    _sha256_digest_v1(ref["content_sha256"], field=f"{field}.content_sha256")
    if artifact_id == "EXPEDITION_LEDGER":
        head = ref.get("head_event_sha256")
        if (
            type(head) is not str
            or len(head) != 64
            or any(character not in "0123456789abcdef" for character in head)
        ):
            raise ExpeditionReviewProjectionError(
                f"{field}.head_event_sha256 is invalid"
            )
    return ref


def _validate_review_row_v1(value: object, *, field: str) -> dict[str, Any]:
    row = _mapping(value, field=field)
    _reject_forbidden_review_fields_v1(row, path=field)
    row_kind = row.get("row_kind")
    expected = (
        None if type(row_kind) is not str else _REVIEW_ROW_FIELDS_V1.get(row_kind)
    )
    if expected is None or set(row) != set(expected):
        raise ExpeditionReviewProjectionError(
            f"{field} has an unknown kind or non-exact fields"
        )
    if len(canonical_json_bytes(row)) > MAX_REVIEW_ROW_BYTES_V1:
        raise ExpeditionReviewProjectionError(f"{field} exceeds its compact bound")
    for key, item in row.items():
        if (
            key.endswith("_count")
            or key.endswith("_omitted")
            or key
            in {
                "epoch_ticks",
                "maximum_absolute_tick",
                "maximum_frontiers",
                "maximum_actions_per_frontier",
                "maximum_atlas_entries",
                "maximum_retained_parent_entries",
                "maximum_retained_parent_bytes",
                "logical_world_id",
                "next_absolute_tick",
                "retained_bytes",
                "selected_at_frontier_ordinal",
            }
        ):
            if type(item) is not int or item < 0:
                raise ExpeditionReviewProjectionError(
                    f"{field}.{key} must be a non-negative integer"
                )
        if (
            key == "wall_time_budget_seconds"
            and item is not None
            and (type(item) is not int or item < 1)
        ):
            raise ExpeditionReviewProjectionError(
                f"{field}.{key} must be null or a positive integer"
            )
        if key == "frontier_ordinal_samples":
            rows = _sequence(item, field=f"{field}.{key}")
            if len(rows) > MAX_INDEX_IDENTITIES_PER_ROW_V1 or any(
                type(entry) is not int or entry < 0 for entry in rows
            ):
                raise ExpeditionReviewProjectionError(
                    f"{field}.{key} exceeds its bounded ordinal-array shape"
                )
        elif key.endswith("_samples") or key == "reason_codes":
            rows = _sequence(item, field=f"{field}.{key}")
            if len(rows) > MAX_INDEX_IDENTITIES_PER_ROW_V1 or any(
                type(entry) is not str or not entry for entry in rows
            ):
                raise ExpeditionReviewProjectionError(
                    f"{field}.{key} exceeds its bounded text-array shape"
                )
            if "sha256" in key and any(
                len(entry) != 64
                or any(character not in "0123456789abcdef" for character in entry)
                for entry in rows
            ):
                raise ExpeditionReviewProjectionError(
                    f"{field}.{key} contains a non-SHA-256 identity"
                )
        if key in {"hypothesis_ids", "founder_artifact_ids", "stop_conditions"}:
            rows = _sequence(item, field=f"{field}.{key}")
            if len(rows) > MAX_REVIEW_TEXT_ARRAY_ITEMS_V1 or any(
                type(entry) is not str or not entry for entry in rows
            ):
                raise ExpeditionReviewProjectionError(
                    f"{field}.{key} exceeds its bounded text-array shape"
                )
    if "proposed_action_kind_counts" in row:
        counts = _mapping(
            row["proposed_action_kind_counts"],
            field=f"{field}.proposed_action_kind_counts",
        )
        if any(
            type(key) is not str or not key or type(count) is not int or count < 0
            for key, count in counts.items()
        ):
            raise ExpeditionReviewProjectionError(
                f"{field}.proposed_action_kind_counts is invalid"
            )
    for key in (
        "declared_in_authorized_search_space",
        "selected_world_historical_continuation_available",
        "requires_new_immutable_manifest",
        "automatic_continuation",
        "must_remain_human",
    ):
        if key in row and type(row[key]) is not bool:
            raise ExpeditionReviewProjectionError(f"{field}.{key} must be boolean")
    for key in _REQUIRED_TEXT_ROW_FIELDS_V1 & set(row):
        _text(row[key], field=f"{field}.{key}")
    for key in _OPTIONAL_TEXT_ROW_FIELDS_V1 & set(row):
        if row[key] is not None:
            _text(row[key], field=f"{field}.{key}")
    for key, expected_path in _EXACT_REVIEW_PATH_FIELDS_V1.items():
        if key in row and row[key] != expected_path:
            raise ExpeditionReviewProjectionError(
                f"{field}.{key} is not the direct bounded artifact path"
            )
    for key in (
        "manifest_sha256",
        "founder_basis_sha256",
        "search_space_sha256",
        "state_ref_sha256",
        "source_object_sha256",
        "observation_descriptor_set_sha256",
    ):
        if key in row:
            _sha256_digest_v1(row[key], field=f"{field}.{key}")
    for key in ("exact_state_sha256", "checkpoint_occurrence_sha256"):
        if key in row and row[key] is not None:
            _sha256_digest_v1(row[key], field=f"{field}.{key}")
    return row


def _validate_presentation_row_v1(
    value: object,
    *,
    field: str,
    identity_field: str,
) -> dict[str, str]:
    row = _mapping(value, field=field)
    if set(row) != {identity_field, "section_key"}:
        raise ExpeditionReviewProjectionError(f"{field} fields must be exact")
    identity = _text(row[identity_field], field=f"{field}.{identity_field}")
    section = _text(row["section_key"], field=f"{field}.section_key")
    if section not in _POLICY_INDEX_SECTIONS_V1:
        raise ExpeditionReviewProjectionError(
            f"{field} names a section that cannot receive policy index rows"
        )
    return {identity_field: identity, "section_key": section}


def _validate_recommendation_v1(value: object, *, field: str) -> dict[str, str]:
    row = _mapping(value, field=field)
    if set(row) != {"recommendation_id", "recommendation"}:
        raise ExpeditionReviewProjectionError(f"{field} fields must be exact")
    return {
        "recommendation_id": _text(
            row["recommendation_id"], field=f"{field}.recommendation_id"
        ),
        "recommendation": _text(row["recommendation"], field=f"{field}.recommendation"),
    }


def validate_policy_review_presentation_v1(value: object) -> dict[str, Any]:
    """Validate semantic-neutral presentation data bound inside a policy profile."""

    presentation = _mapping(value, field="policy human_review_projection")
    _reject_forbidden_review_fields_v1(
        presentation, path="policy human_review_projection"
    )
    required = {
        "schema_version",
        "atlas_role_sections",
        "action_reason_sections",
        "empty_status_by_section",
        "recommendation_rules",
        "fallback_recommendation",
        "human_scientific_choices",
        "released_parent_reason_code",
    }
    if set(presentation) != required:
        raise ExpeditionReviewProjectionError(
            "policy human-review presentation fields must be exact"
        )
    if presentation["schema_version"] != POLICY_REVIEW_PRESENTATION_SCHEMA_V1:
        raise ExpeditionReviewProjectionError(
            "policy human-review presentation version is unsupported"
        )

    raw_atlas_rows = _sequence(
        presentation["atlas_role_sections"], field="atlas_role_sections"
    )
    if len(raw_atlas_rows) > MAX_REVIEW_TEXT_ARRAY_ITEMS_V1:
        raise ExpeditionReviewProjectionError(
            "atlas role presentation exceeds its bounded row count"
        )
    atlas_rows = [
        _validate_presentation_row_v1(
            row,
            field=f"atlas_role_sections[{index}]",
            identity_field="representative_role",
        )
        for index, row in enumerate(raw_atlas_rows)
    ]
    role_ids = [row["representative_role"] for row in atlas_rows]
    if role_ids != sorted(set(role_ids)):
        raise ExpeditionReviewProjectionError(
            "atlas role presentation rows must be sorted and unique"
        )

    raw_action_rows = _sequence(
        presentation["action_reason_sections"],
        field="action_reason_sections",
    )
    if len(raw_action_rows) > MAX_REVIEW_TEXT_ARRAY_ITEMS_V1:
        raise ExpeditionReviewProjectionError(
            "action reason presentation exceeds its bounded row count"
        )
    action_rows = [
        _validate_presentation_row_v1(
            row,
            field=f"action_reason_sections[{index}]",
            identity_field="reason_code",
        )
        for index, row in enumerate(raw_action_rows)
    ]
    reason_ids = [row["reason_code"] for row in action_rows]
    if reason_ids != sorted(set(reason_ids)):
        raise ExpeditionReviewProjectionError(
            "action reason presentation rows must be sorted and unique"
        )

    empty_status = _mapping(
        presentation["empty_status_by_section"], field="empty_status_by_section"
    )
    if set(empty_status) != set(REVIEW_SECTION_KEYS_V1):
        raise ExpeditionReviewProjectionError(
            "empty review status must cover exactly the thirteen sections"
        )
    for key in REVIEW_SECTION_KEYS_V1:
        _text(empty_status[key], field=f"empty_status_by_section.{key}")

    raw_rules = _sequence(
        presentation["recommendation_rules"], field="recommendation_rules"
    )
    if len(raw_rules) > MAX_REVIEW_TEXT_ARRAY_ITEMS_V1:
        raise ExpeditionReviewProjectionError(
            "recommendation rules exceed their bounded row count"
        )
    rules = []
    for index, raw in enumerate(raw_rules):
        row = _mapping(raw, field=f"recommendation_rules[{index}]")
        if set(row) != {
            "when_section_key",
            "recommendation_id",
            "recommendation",
        }:
            raise ExpeditionReviewProjectionError(
                f"recommendation_rules[{index}] fields must be exact"
            )
        section = _text(
            row["when_section_key"],
            field=f"recommendation_rules[{index}].when_section_key",
        )
        if section not in REVIEW_SECTION_KEYS_V1:
            raise ExpeditionReviewProjectionError(
                f"recommendation_rules[{index}] names an unknown section"
            )
        rules.append(
            {
                "when_section_key": section,
                **_validate_recommendation_v1(
                    {
                        "recommendation_id": row["recommendation_id"],
                        "recommendation": row["recommendation"],
                    },
                    field=f"recommendation_rules[{index}]",
                ),
            }
        )
    rule_ids = [row["recommendation_id"] for row in rules]
    if len(rule_ids) != len(set(rule_ids)):
        raise ExpeditionReviewProjectionError(
            "recommendation identities must be unique"
        )

    raw_choices = _sequence(
        presentation["human_scientific_choices"],
        field="human_scientific_choices",
    )
    if len(raw_choices) > MAX_REVIEW_TEXT_ARRAY_ITEMS_V1:
        raise ExpeditionReviewProjectionError(
            "human scientific choices exceed their bounded row count"
        )
    choices = []
    for index, raw in enumerate(raw_choices):
        row = _mapping(raw, field=f"human_scientific_choices[{index}]")
        if set(row) != {"choice_id", "choice"}:
            raise ExpeditionReviewProjectionError(
                f"human_scientific_choices[{index}] fields must be exact"
            )
        choices.append(
            {
                "choice_id": _text(
                    row["choice_id"],
                    field=f"human_scientific_choices[{index}].choice_id",
                ),
                "choice": _text(
                    row["choice"],
                    field=f"human_scientific_choices[{index}].choice",
                ),
            }
        )
    choice_ids = [row["choice_id"] for row in choices]
    if choice_ids != sorted(set(choice_ids)):
        raise ExpeditionReviewProjectionError(
            "human scientific choices must be sorted and unique"
        )

    return {
        "schema_version": POLICY_REVIEW_PRESENTATION_SCHEMA_V1,
        "atlas_role_sections": atlas_rows,
        "action_reason_sections": action_rows,
        "empty_status_by_section": empty_status,
        "recommendation_rules": rules,
        "fallback_recommendation": _validate_recommendation_v1(
            presentation["fallback_recommendation"],
            field="fallback_recommendation",
        ),
        "human_scientific_choices": choices,
        "released_parent_reason_code": _text(
            presentation["released_parent_reason_code"],
            field="released_parent_reason_code",
        ),
    }


def _sample(values: Sequence[str]) -> tuple[list[str], int]:
    ordered = sorted(set(values))
    selected = ordered[:MAX_INDEX_IDENTITIES_PER_ROW_V1]
    return selected, len(ordered) - len(selected)


def _sample_ordinals(values: Sequence[int]) -> tuple[list[int], int]:
    ordered = sorted(set(values))
    selected = ordered[:MAX_INDEX_IDENTITIES_PER_ROW_V1]
    return selected, len(ordered) - len(selected)


def _opaque_control_domains_v1(
    manifest: ExpeditionManifestV1,
) -> dict[str, list[str]]:
    value = _plain(manifest.search_space.to_value(), field="manifest search_space")
    if type(value) is not dict or type(value.get("controls")) is not list:
        return {}
    result: dict[str, list[str]] = {}
    for raw in value["controls"]:
        if type(raw) is not dict:
            continue
        coordinate = raw.get("control_id")
        allowed = raw.get("allowed_values")
        if type(coordinate) is not str or not coordinate or type(allowed) is not list:
            continue
        result[coordinate] = [canonical_sha256(item).hex for item in allowed]
    return result


def _opaque_selected_control_values_v1(
    decisions: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, list[str]], dict[str, int]]:
    selected: dict[str, list[str]] = {}
    counts: dict[str, int] = {}
    for record in decisions:
        decision = _mapping(record["decision"], field="coverage decision")
        for raw_action in _sequence(
            decision.get("action_rows"), field="coverage action_rows"
        ):
            action = _mapping(raw_action, field="coverage action")
            envelope = action.get("action_payload")
            if type(envelope) is not dict:
                continue
            payload = envelope.get("payload")
            if type(payload) is not dict:
                continue
            bindings = payload.get("control_bindings")
            if type(bindings) is not list:
                continue
            for raw_binding in bindings:
                if type(raw_binding) is not dict or set(raw_binding) != {
                    "control_id",
                    "value",
                }:
                    continue
                coordinate = raw_binding.get("control_id")
                if type(coordinate) is not str or not coordinate:
                    continue
                selected.setdefault(coordinate, []).append(
                    canonical_sha256(raw_binding["value"]).hex
                )
                counts[coordinate] = counts.get(coordinate, 0) + 1
    return selected, counts


def _artifact_ref(
    artifact_index: Mapping[str, object], artifact_id: str
) -> dict[str, Any]:
    value = artifact_index.get(artifact_id)
    if value is None:
        raise ExpeditionReviewProjectionError(f"artifact index lacks {artifact_id}")
    return _mapping(value, field=f"artifact_index.{artifact_id}")


def _new_sections(
    empty_status_by_section: Mapping[str, str],
) -> dict[str, dict[str, Any]]:
    return {
        key: {
            "schema_version": REVIEW_SECTION_SCHEMA_V1,
            "question_ordinal": ordinal,
            "section_key": key,
            "status_code": empty_status_by_section[key],
            "rows": [],
            "evidence_refs": [],
        }
        for ordinal, key in enumerate(REVIEW_SECTION_KEYS_V1, start=1)
    }


def _append_row(
    sections: dict[str, dict[str, Any]],
    *,
    section_key: str,
    row: Mapping[str, object],
    evidence_refs: Sequence[Mapping[str, object]] = (),
) -> None:
    section = sections[section_key]
    if len(section["rows"]) >= MAX_REVIEW_ROWS_PER_SECTION_V1:
        raise ExpeditionReviewProjectionError(
            f"review.{section_key} exceeds its bounded row count"
        )
    section["rows"].append(
        _validate_review_row_v1(row, field=f"review.{section_key}.row")
    )
    section["status_code"] = "RECORDED_INDEX_ROWS_NO_CLAIMS"
    known = {canonical_sha256(item).hex for item in section["evidence_refs"]}
    for raw_ref in evidence_refs:
        ref = _validate_evidence_ref_v1(
            raw_ref, field=f"review.{section_key}.evidence_ref"
        )
        identity = canonical_sha256(ref).hex
        if identity not in known:
            if len(section["evidence_refs"]) >= MAX_EVIDENCE_REFS_PER_SECTION_V1:
                raise ExpeditionReviewProjectionError(
                    f"review.{section_key} exceeds its evidence-reference bound"
                )
            section["evidence_refs"].append(ref)
            known.add(identity)


def build_human_review_projection_v1(
    *,
    manifest: ExpeditionManifestV1,
    policy_profile: Mapping[str, object],
    ledger: Mapping[str, object],
    atlas: EvolutionaryAtlasV1,
    retained_parent_bank: RetainedParentBankV1,
    public_journal: Sequence[Mapping[str, object]],
    artifact_index: Mapping[str, object],
    programme_stage: str,
    historical_continuation_authority: bool,
) -> dict[str, object]:
    """Build the thirteen compact indexes without interpreting policy meanings."""

    if not isinstance(manifest, ExpeditionManifestV1):
        raise ExpeditionReviewProjectionError("manifest contract is invalid")
    if not isinstance(atlas, EvolutionaryAtlasV1):
        raise ExpeditionReviewProjectionError("Atlas contract is invalid")
    if not isinstance(retained_parent_bank, RetainedParentBankV1):
        raise ExpeditionReviewProjectionError("Parent Bank contract is invalid")
    if type(programme_stage) is not str or not programme_stage:
        raise ExpeditionReviewProjectionError("programme_stage must be text")
    if type(historical_continuation_authority) is not bool:
        raise ExpeditionReviewProjectionError(
            "historical continuation authority must be boolean"
        )
    retained_parent_bank.validate_against_manifest(manifest)
    if (
        atlas.expedition_id != manifest.expedition_id
        or atlas.manifest_sha256 != manifest.manifest_sha256()
        or atlas.descriptor_set_sha256 != manifest.observation_descriptor_set_sha256
    ):
        raise ExpeditionReviewProjectionError(
            "Atlas differs from the immutable manifest"
        )
    profile = _mapping(policy_profile, field="policy_profile")
    presentation = validate_policy_review_presentation_v1(
        profile.get("human_review_projection")
    )
    projection = reconstruct_expedition_frontier_v1(ledger)
    journal = _sequence(public_journal, field="public_journal")
    artifacts = _mapping(artifact_index, field="artifact_index")
    manifest_ref = _artifact_ref(artifacts, "manifest")
    atlas_ref = _artifact_ref(artifacts, "atlas")
    bank_ref = _artifact_ref(artifacts, "retained_parent_bank")
    journal_ref = _artifact_ref(artifacts, "journal")
    ledger_ref = _artifact_ref(artifacts, "ledger")
    sections = _new_sections(presentation["empty_status_by_section"])

    _append_row(
        sections,
        section_key="authorized_search_space",
        row={
            "row_kind": "AUTHORIZED_MANIFEST_INDEX",
            "manifest_sha256": manifest.manifest_sha256().to_dict(),
            "question_ref": "../control/expedition_manifest.json#/question_description",
            "hypothesis_ids": list(manifest.hypothesis_ids),
            "founder_basis_sha256": manifest.founder_basis_sha256().to_dict(),
            "founder_artifact_ids": [
                str(item.artifact_id) for item in manifest.founder_basis_refs
            ],
            "search_space_sha256": manifest.search_space.content_digest().to_dict(),
            "search_space_ref": "../control/expedition_manifest.json#/search_space",
            "epoch_ticks": manifest.epoch_ticks,
            "maximum_absolute_tick": manifest.maximum_absolute_tick,
            "maximum_frontiers": manifest.maximum_frontiers,
            "maximum_actions_per_frontier": (manifest.maximum_actions_per_frontier),
            "wall_time_budget_seconds": manifest.wall_time_budget_seconds,
            "stop_conditions": list(manifest.stop_conditions),
        },
        evidence_refs=(manifest_ref,),
    )

    action_kind_counts: dict[str, int] = {}
    dispatched_action_ids: list[str] = []
    settled_action_ids: list[str] = []
    observation_ids: list[str] = []
    decision_sha256s: list[str] = []
    decisions: list[dict[str, Any]] = []
    for event in ledger["events"]:
        payload = event["payload"]
        event_type = event["event_type"]
        if event_type == "FRONTIER_OBSERVED":
            observation_ids.extend(payload.get("observation_frame_sha256s", []))
        elif event_type == "DECISION_PREPARED":
            decision = _mapping(payload["decision"], field="ledger decision")
            decisions.append(
                {
                    "frontier_ordinal": event["frontier_ordinal"],
                    "decision_sha256": payload["decision_sha256"],
                    "event_sha256": event["event_sha256"],
                    "decision": decision,
                    "retention_outcomes": _mapping(
                        payload["retention_outcomes"],
                        field="retention_outcomes",
                    ),
                }
            )
            decision_sha256s.append(payload["decision_sha256"])
            for row in decision["action_rows"]:
                kind = str(row["action_kind"])
                action_kind_counts[kind] = action_kind_counts.get(kind, 0) + 1
        elif event_type == "ACTION_DISPATCH_BOUND":
            dispatched_action_ids.append(payload["action_id"])
        elif event_type == "ACTION_SETTLED":
            settled_action_ids.append(payload["action_id"])
    sampled_observations, omitted_observations = _sample(observation_ids)
    sampled_decisions, omitted_decisions = _sample(decision_sha256s)
    sampled_actions, omitted_actions = _sample(dispatched_action_ids)
    _append_row(
        sections,
        section_key="exploration_coverage",
        row={
            "row_kind": "EXPERIMENT_DAG_AND_COVERAGE_INDEX",
            "frontier_count": len(projection["frontiers"]),
            "decision_count": len(decisions),
            "proposed_action_kind_counts": dict(sorted(action_kind_counts.items())),
            "dispatched_action_count": len(set(dispatched_action_ids)),
            "settled_action_count": len(set(settled_action_ids)),
            "atlas_entry_count": len(atlas.entries),
            "journal_record_count": len(journal),
            "observation_frame_sha256_samples": sampled_observations,
            "observation_frame_sha256_omitted": omitted_observations,
            "decision_sha256_samples": sampled_decisions,
            "decision_sha256_omitted": omitted_decisions,
            "action_id_samples": sampled_actions,
            "action_ids_omitted": omitted_actions,
            "complete_history_ref": journal_ref["relative_path"],
        },
        evidence_refs=(journal_ref, ledger_ref, atlas_ref),
    )
    authorized_controls = _opaque_control_domains_v1(manifest)
    selected_controls, selected_control_counts = _opaque_selected_control_values_v1(
        decisions
    )
    control_coordinates = sorted(set(authorized_controls) | set(selected_controls))
    for coordinate in control_coordinates[:MAX_CONTROL_COORDINATE_ROWS_V1]:
        authorized_samples, authorized_omitted = _sample(
            authorized_controls.get(coordinate, [])
        )
        selected_samples, selected_omitted = _sample(
            selected_controls.get(coordinate, [])
        )
        _append_row(
            sections,
            section_key="exploration_coverage",
            row={
                "row_kind": "OPAQUE_CONTROL_COORDINATE_COVERAGE_INDEX",
                "control_coordinate": coordinate,
                "declared_in_authorized_search_space": (
                    coordinate in authorized_controls
                ),
                "authorized_value_count": len(
                    set(authorized_controls.get(coordinate, []))
                ),
                "authorized_value_sha256_samples": authorized_samples,
                "authorized_value_sha256s_omitted": authorized_omitted,
                "selected_value_count": len(set(selected_controls.get(coordinate, []))),
                "selected_value_sha256_samples": selected_samples,
                "selected_value_sha256s_omitted": selected_omitted,
                "action_binding_count": selected_control_counts.get(coordinate, 0),
            },
            evidence_refs=(manifest_ref, ledger_ref),
        )
    if len(control_coordinates) > MAX_CONTROL_COORDINATE_ROWS_V1:
        _append_row(
            sections,
            section_key="exploration_coverage",
            row={
                "row_kind": "OPAQUE_CONTROL_COORDINATE_INDEX_OMISSION",
                "omitted_control_coordinate_count": (
                    len(control_coordinates) - MAX_CONTROL_COORDINATE_ROWS_V1
                ),
                "complete_search_space_ref": manifest_ref["relative_path"],
                "complete_history_ref": journal_ref["relative_path"],
            },
            evidence_refs=(manifest_ref, journal_ref, ledger_ref),
        )

    role_to_section = {
        row["representative_role"]: row["section_key"]
        for row in presentation["atlas_role_sections"]
    }
    atlas_groups: dict[tuple[str, str], list[Any]] = {}
    for entry in atlas.entries:
        for role in entry.representative_roles:
            section_key = role_to_section.get(role)
            if section_key is not None:
                atlas_groups.setdefault((section_key, role), []).append(entry)
    for (section_key, role), entries in sorted(atlas_groups.items()):
        entry_ids, omitted = _sample([entry.atlas_entry_id for entry in entries])
        observation_refs, observation_omitted = _sample(
            [entry.observation_frame_sha256.hex for entry in entries]
        )
        state_refs, state_omitted = _sample(
            [
                entry.state_ref_sha256.hex
                for entry in entries
                if entry.state_ref_sha256 is not None
            ]
        )
        _append_row(
            sections,
            section_key=section_key,
            row={
                "row_kind": "POLICY_ATLAS_ROLE_INDEX",
                "representative_role": role,
                "entry_count": len(entries),
                "atlas_entry_id_samples": entry_ids,
                "atlas_entry_ids_omitted": omitted,
                "observation_frame_sha256_samples": observation_refs,
                "observation_frame_sha256s_omitted": observation_omitted,
                "state_ref_sha256_samples": state_refs,
                "state_ref_sha256s_omitted": state_omitted,
            },
            evidence_refs=(atlas_ref,),
        )

    reason_to_section = {
        row["reason_code"]: row["section_key"]
        for row in presentation["action_reason_sections"]
    }
    action_reason_groups: dict[tuple[str, str], list[str]] = {}
    for record in decisions:
        for row in record["decision"]["action_rows"]:
            for reason in row["reason_codes"]:
                section_key = reason_to_section.get(reason)
                if section_key is not None:
                    action_reason_groups.setdefault((section_key, reason), []).append(
                        row["action_id"]
                    )
    for (section_key, reason), action_ids in sorted(action_reason_groups.items()):
        samples, omitted = _sample(action_ids)
        _append_row(
            sections,
            section_key=section_key,
            row={
                "row_kind": "POLICY_ACTION_REASON_INDEX",
                "reason_code": reason,
                "action_count": len(set(action_ids)),
                "action_id_samples": samples,
                "action_ids_omitted": omitted,
            },
            evidence_refs=(ledger_ref,),
        )

    bank_rows = list(retained_parent_bank.entries)
    for entry in bank_rows[:MAX_RETAINED_PARENT_ROWS_V1]:
        state = entry.state_ref
        _append_row(
            sections,
            section_key="retained_parents",
            row={
                "row_kind": "EXACT_RETAINED_PARENT_INDEX",
                "state_ref_sha256": state.state_ref_sha256().to_dict(),
                "world_trajectory_id": state.world_trajectory_id,
                "logical_world_id": state.logical_world_id,
                "next_absolute_tick": state.next_absolute_tick,
                "exact_state_sha256": (
                    None
                    if state.exact_state_sha256 is None
                    else state.exact_state_sha256.to_dict()
                ),
                "checkpoint_occurrence_sha256": (
                    None
                    if state.checkpoint_occurrence_sha256 is None
                    else state.checkpoint_occurrence_sha256.to_dict()
                ),
                "source_artifact_id": str(state.source_artifact_ref.artifact_id),
                "source_object_sha256": state.source_object_sha256.to_dict(),
                "selected_at_frontier_ordinal": entry.selected_at_frontier_ordinal,
                "reason_codes": list(entry.reason_codes),
                "retained_bytes": entry.retained_bytes,
            },
            evidence_refs=(bank_ref,),
        )
    if len(bank_rows) > MAX_RETAINED_PARENT_ROWS_V1:
        _append_row(
            sections,
            section_key="retained_parents",
            row={
                "row_kind": "RETAINED_PARENT_INDEX_OMISSION",
                "omitted_parent_count": (len(bank_rows) - MAX_RETAINED_PARENT_ROWS_V1),
                "complete_parent_bank_ref": bank_ref["relative_path"],
            },
            evidence_refs=(bank_ref,),
        )

    discarded: dict[str, dict[str, Any]] = {}
    released_reason = presentation["released_parent_reason_code"]
    for record in decisions:
        outcomes = record["retention_outcomes"]
        for raw in outcomes["policy_not_retained"]:
            row = _mapping(raw, field="policy_not_retained")
            reason = _text(row.get("reason_code"), field="discard reason_code")
            state_ref_id = _text(row.get("state_ref_id"), field="discard state_ref_id")
            group = discarded.setdefault(
                reason,
                {
                    "state_ref_ids": [],
                    "frontier_ordinals": [],
                    "decision_sha256s": [],
                },
            )
            group["state_ref_ids"].append(state_ref_id)
            group["frontier_ordinals"].append(record["frontier_ordinal"])
            group["decision_sha256s"].append(record["decision_sha256"])
        for raw_state_ref_id in outcomes["released_state_ref_ids"]:
            state_ref_id = _text(
                raw_state_ref_id, field="released discard state_ref_id"
            )
            group = discarded.setdefault(
                released_reason,
                {
                    "state_ref_ids": [],
                    "frontier_ordinals": [],
                    "decision_sha256s": [],
                },
            )
            group["state_ref_ids"].append(state_ref_id)
            group["frontier_ordinals"].append(record["frontier_ordinal"])
            group["decision_sha256s"].append(record["decision_sha256"])
    discarded_rows = sorted(discarded.items())
    for reason, values in discarded_rows[:MAX_DISCARD_REASON_ROWS_V1]:
        state_ids, omitted = _sample(values["state_ref_ids"])
        decision_ids, decisions_omitted = _sample(values["decision_sha256s"])
        frontier_ordinals, frontier_ordinals_omitted = _sample_ordinals(
            values["frontier_ordinals"]
        )
        _append_row(
            sections,
            section_key="discarded_parent_candidates",
            row={
                "row_kind": "POLICY_PARENT_DISCARD_REASON_INDEX",
                "reason_code": reason,
                "distinct_state_count": len(set(values["state_ref_ids"])),
                "state_ref_id_samples": state_ids,
                "state_ref_ids_omitted": omitted,
                "frontier_ordinal_samples": frontier_ordinals,
                "frontier_ordinals_omitted": frontier_ordinals_omitted,
                "decision_sha256_samples": decision_ids,
                "decision_sha256s_omitted": decisions_omitted,
            },
            evidence_refs=(ledger_ref,),
        )
    if len(discarded_rows) > MAX_DISCARD_REASON_ROWS_V1:
        _append_row(
            sections,
            section_key="discarded_parent_candidates",
            row={
                "row_kind": "DISCARDED_PARENT_REASON_INDEX_OMISSION",
                "omitted_reason_count": (
                    len(discarded_rows) - MAX_DISCARD_REASON_ROWS_V1
                ),
                "complete_history_ref": journal_ref["relative_path"],
            },
            evidence_refs=(journal_ref, ledger_ref),
        )

    _append_row(
        sections,
        section_key="unresolved_scientific_questions",
        row={
            "row_kind": "MANIFEST_QUESTION_UNRESOLVED_NO_CLAIMS",
            "question_description": manifest.question_description,
            "hypothesis_ids": list(manifest.hypothesis_ids),
            "resolution_status": "REQUIRES_HUMAN_SCIENTIFIC_REVIEW",
        },
        evidence_refs=(manifest_ref, atlas_ref, journal_ref),
    )

    terminal = projection["terminal"] or {}
    stop = projection["stop"] or {}
    _append_row(
        sections,
        section_key="limiting_controls_observations_capabilities_or_budgets",
        row={
            "row_kind": "EXPEDITION_BOUNDARY_INDEX",
            "programme_stage": programme_stage,
            "terminal_status": terminal.get("terminal_status"),
            "stop_reason": stop.get("reason"),
            "selected_world_historical_continuation_available": (
                historical_continuation_authority
            ),
            "required_science_instrument": manifest.required_science_instrument,
            "observation_profile_id": manifest.observation_profile_id,
            "observation_descriptor_set_sha256": (
                manifest.observation_descriptor_set_sha256.to_dict()
            ),
            "maximum_absolute_tick": manifest.maximum_absolute_tick,
            "maximum_frontiers": manifest.maximum_frontiers,
            "maximum_actions_per_frontier": (manifest.maximum_actions_per_frontier),
            "maximum_atlas_entries": manifest.maximum_atlas_entries,
            "maximum_retained_parent_entries": (
                manifest.maximum_retained_parent_entries
            ),
            "maximum_retained_parent_bytes": (manifest.maximum_retained_parent_bytes),
        },
        evidence_refs=(manifest_ref, ledger_ref, atlas_ref, bank_ref),
    )

    recommendation = presentation["fallback_recommendation"]
    trigger_section: str | None = None
    for rule in presentation["recommendation_rules"]:
        candidate = sections[rule["when_section_key"]]
        if candidate["rows"]:
            recommendation = {
                "recommendation_id": rule["recommendation_id"],
                "recommendation": rule["recommendation"],
            }
            trigger_section = rule["when_section_key"]
            break
    recommendation_refs = (
        (manifest_ref, atlas_ref, bank_ref, journal_ref)
        if trigger_section is None
        else tuple(sections[trigger_section]["evidence_refs"])
    )
    _append_row(
        sections,
        section_key="recommended_next_expedition",
        row={
            "row_kind": "POLICY_PRESENTATION_RECOMMENDATION",
            "recommendation_id": recommendation["recommendation_id"],
            "recommendation": recommendation["recommendation"],
            "trigger_section": trigger_section,
            "requires_new_immutable_manifest": True,
            "automatic_continuation": False,
        },
        evidence_refs=recommendation_refs,
    )

    for choice in presentation["human_scientific_choices"]:
        _append_row(
            sections,
            section_key="human_scientific_choices",
            row={
                "row_kind": "POLICY_PRESENTATION_HUMAN_CHOICE",
                "choice_id": choice["choice_id"],
                "choice": choice["choice"],
                "must_remain_human": True,
            },
            evidence_refs=(manifest_ref, atlas_ref, bank_ref, journal_ref),
        )

    result = {key: sections[key] for key in REVIEW_SECTION_KEYS_V1}
    validate_human_review_projection_v1(result)
    return result


def validate_human_review_projection_v1(value: object) -> dict[str, Any]:
    review = _mapping(value, field="human review projection")
    _reject_forbidden_review_fields_v1(review, path="human review projection")
    if set(review) != set(REVIEW_SECTION_KEYS_V1):
        raise ExpeditionReviewProjectionError(
            "human review must answer exactly the thirteen sections"
        )
    for ordinal, key in enumerate(REVIEW_SECTION_KEYS_V1, start=1):
        section = _mapping(review[key], field=f"review.{key}")
        if set(section) != {
            "schema_version",
            "question_ordinal",
            "section_key",
            "status_code",
            "rows",
            "evidence_refs",
        }:
            raise ExpeditionReviewProjectionError(
                f"review section {key} fields must be exact"
            )
        if (
            section["schema_version"] != REVIEW_SECTION_SCHEMA_V1
            or section["question_ordinal"] != ordinal
            or section["section_key"] != key
        ):
            raise ExpeditionReviewProjectionError(
                f"review section {key} identity differs"
            )
        status = _text(section["status_code"], field=f"review.{key}.status_code")
        rows = _sequence(section["rows"], field=f"review.{key}.rows")
        evidence_refs = _sequence(
            section["evidence_refs"], field=f"review.{key}.evidence_refs"
        )
        if len(rows) > MAX_REVIEW_ROWS_PER_SECTION_V1:
            raise ExpeditionReviewProjectionError(
                f"review.{key} exceeds its bounded row count"
            )
        if len(evidence_refs) > MAX_EVIDENCE_REFS_PER_SECTION_V1:
            raise ExpeditionReviewProjectionError(
                f"review.{key} exceeds its evidence-reference bound"
            )
        validated_rows = [
            _validate_review_row_v1(row, field=f"review.{key}.rows[{index}]")
            for index, row in enumerate(rows)
        ]
        if any(
            row["row_kind"] not in _ALLOWED_ROW_KINDS_BY_SECTION_V1[key]
            for row in validated_rows
        ):
            raise ExpeditionReviewProjectionError(
                f"review.{key} contains a row kind assigned to another question"
            )
        validated_refs = [
            _validate_evidence_ref_v1(ref, field=f"review.{key}.evidence_refs[{index}]")
            for index, ref in enumerate(evidence_refs)
        ]
        if len({canonical_sha256(row).hex for row in validated_rows}) != len(
            validated_rows
        ):
            raise ExpeditionReviewProjectionError(
                f"review.{key} contains duplicate rows"
            )
        if len({canonical_sha256(ref).hex for ref in validated_refs}) != len(
            validated_refs
        ):
            raise ExpeditionReviewProjectionError(
                f"review.{key} contains duplicate evidence references"
            )
        if bool(rows) is (status != "RECORDED_INDEX_ROWS_NO_CLAIMS"):
            raise ExpeditionReviewProjectionError(
                f"review.{key} status does not match its row presence"
            )
        if bool(rows) != bool(evidence_refs):
            raise ExpeditionReviewProjectionError(
                f"review.{key} rows and direct evidence references must co-occur"
            )
    return review


__all__ = (
    "ExpeditionReviewProjectionError",
    "MAX_INDEX_IDENTITIES_PER_ROW_V1",
    "POLICY_REVIEW_PRESENTATION_SCHEMA_V1",
    "REVIEW_SECTION_KEYS_V1",
    "REVIEW_SECTION_SCHEMA_V1",
    "build_human_review_projection_v1",
    "validate_human_review_projection_v1",
    "validate_policy_review_presentation_v1",
)
