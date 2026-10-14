"""Legacy test-only Atlas compatibility helpers.

Atlas entries are derived references and bounded summaries.  They never carry a
checkpoint, replay recipe, lifecycle object, filesystem path, or retention
authority; exact reusable state belongs exclusively to the canonical
``RetainedParentBankV1`` contract.  This module is private and is not a
production/public Expedition contract; new code must use
``emergence_contracts.adaptive_science_expedition_v1``.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import math
import re
from typing import Any, Final

from emergence_contracts import canonical_json_bytes


ATLAS_SCHEMA_V1: Final[str] = "eiel_adaptive_expedition_atlas_v1"
ATLAS_ENTRY_SCHEMA_V1: Final[str] = "eiel_adaptive_expedition_atlas_entry_v1"
MAX_EPISTEMIC_SUMMARY_BYTES_V1: Final[int] = 16 * 1024

_ENTRY_INPUT_FIELDS_V1: Final[frozenset[str]] = frozenset(
    {
        "entry_id",
        "candidate_id",
        "frontier_ordinal",
        "niche_id",
        "observation_frame_sha256",
        "semantic_world_sha256",
        "acquisition_sha256",
        "reducer_result_sha256",
        "descriptor_sha256",
        "source_settlement_sha256",
        "epistemic",
    }
)
_FORBIDDEN_EPISTEMIC_KEYS_V1: Final[frozenset[str]] = frozenset(
    {
        "artifact_path",
        "canonical_state",
        "checkpoint",
        "checkpoint_path",
        "checkpoint_ref",
        "content_bytes",
        "lifecycle_manifest",
        "lifecycle_ref",
        "payload",
        "raw_payload",
        "replay_state_ref",
        "run_root",
        "state_ref",
    }
)


class AdaptiveExpeditionAtlasError(ValueError):
    """An Atlas value is non-canonical, unsafe, or internally inconsistent."""


def _canonical_clone(value: Any, *, label: str) -> Any:
    try:
        return json.loads(canonical_json_bytes(value))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise AdaptiveExpeditionAtlasError(
            f"{label} is not canonical JSON data"
        ) from exc


def _sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _require_sha256(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or re.fullmatch(r"[0-9a-f]{64}", value) is None
    ):
        raise AdaptiveExpeditionAtlasError(
            f"{field} must be a lowercase SHA-256"
        )
    return value


def _require_identifier(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 512:
        raise AdaptiveExpeditionAtlasError(
            f"{field} must be a bounded non-empty string"
        )
    return value


def _reject_state_material_v1(value: Any, *, path: str = "epistemic") -> None:
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            if not isinstance(raw_key, str):
                raise AdaptiveExpeditionAtlasError(
                    f"{path} contains a non-string key"
                )
            key = raw_key.lower()
            if (
                key in _FORBIDDEN_EPISTEMIC_KEYS_V1
                or key.endswith("_path")
                or key.endswith("_bytes")
                or key.endswith("_state_ref")
                or key.endswith("_lifecycle_ref")
            ):
                raise AdaptiveExpeditionAtlasError(
                    f"Atlas epistemic summary cannot retain state material: {path}.{raw_key}"
                )
            _reject_state_material_v1(child, path=f"{path}.{raw_key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_state_material_v1(child, path=f"{path}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise AdaptiveExpeditionAtlasError(
            f"{path} contains a non-finite number"
        )


def _atlas_body_v1(
    *,
    expedition_id: str,
    manifest_sha256: str,
    descriptor_set_sha256: str,
    entries: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": ATLAS_SCHEMA_V1,
        "expedition_id": expedition_id,
        "manifest_sha256": manifest_sha256,
        "descriptor_set_sha256": descriptor_set_sha256,
        "authority_effect": "none",
        "state_retention_authority": False,
        "entry_count": len(entries),
        "entries": entries,
    }


def new_expedition_atlas_v1(
    *, expedition_id: str, manifest_sha256: str, descriptor_set_sha256: str
) -> dict[str, Any]:
    """Create an empty derived Atlas."""

    body = _atlas_body_v1(
        expedition_id=_require_identifier(expedition_id, field="expedition_id"),
        manifest_sha256=_require_sha256(
            manifest_sha256, field="manifest_sha256"
        ),
        descriptor_set_sha256=_require_sha256(
            descriptor_set_sha256, field="descriptor_set_sha256"
        ),
        entries=[],
    )
    return {**body, "atlas_sha256": _sha256(body)}


def _normalize_entry_input_v1(value: Mapping[str, Any]) -> dict[str, Any]:
    if set(value) != _ENTRY_INPUT_FIELDS_V1:
        raise AdaptiveExpeditionAtlasError("Atlas entry input fields differ")
    entry = _canonical_clone(dict(value), label="Atlas entry")
    for field in ("entry_id", "candidate_id", "niche_id"):
        _require_identifier(entry[field], field=field)
    if type(entry["frontier_ordinal"]) is not int or entry["frontier_ordinal"] < 0:
        raise AdaptiveExpeditionAtlasError(
            "frontier_ordinal must be a non-negative integer"
        )
    for field in (
        "observation_frame_sha256",
        "semantic_world_sha256",
        "acquisition_sha256",
        "reducer_result_sha256",
        "descriptor_sha256",
        "source_settlement_sha256",
    ):
        _require_sha256(entry[field], field=field)
    if not isinstance(entry["epistemic"], Mapping):
        raise AdaptiveExpeditionAtlasError("epistemic summary must be an object")
    _reject_state_material_v1(entry["epistemic"])
    if len(canonical_json_bytes(entry["epistemic"])) > MAX_EPISTEMIC_SUMMARY_BYTES_V1:
        raise AdaptiveExpeditionAtlasError(
            "epistemic summary exceeds its compact byte bound"
        )
    return entry


def _validate_atlas_entry_v1(
    raw: object, *, expected_ordinal: int, previous_entry_sha256: str
) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise AdaptiveExpeditionAtlasError("Atlas entry must be an object")
    entry = _canonical_clone(dict(raw), label="Atlas entry")
    expected_fields = {
        *_ENTRY_INPUT_FIELDS_V1,
        "schema_version",
        "entry_ordinal",
        "previous_entry_sha256",
        "entry_sha256",
    }
    if set(entry) != expected_fields:
        raise AdaptiveExpeditionAtlasError("sealed Atlas entry fields differ")
    input_value = {field: entry[field] for field in _ENTRY_INPUT_FIELDS_V1}
    _normalize_entry_input_v1(input_value)
    if (
        entry.get("schema_version") != ATLAS_ENTRY_SCHEMA_V1
        or entry.get("entry_ordinal") != expected_ordinal
        or entry.get("previous_entry_sha256") != previous_entry_sha256
    ):
        raise AdaptiveExpeditionAtlasError("Atlas entry chain is inconsistent")
    body = {key: value for key, value in entry.items() if key != "entry_sha256"}
    supplied = _require_sha256(entry.get("entry_sha256"), field="entry_sha256")
    if supplied != _sha256(body):
        raise AdaptiveExpeditionAtlasError("Atlas entry digest changed")
    return entry


def validate_expedition_atlas_v1(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate every compact entry and recompute the Atlas revision digest."""

    if not isinstance(value, Mapping):
        raise AdaptiveExpeditionAtlasError("Atlas must be an object")
    atlas = _canonical_clone(dict(value), label="Atlas")
    expected_fields = {
        "schema_version",
        "expedition_id",
        "manifest_sha256",
        "descriptor_set_sha256",
        "authority_effect",
        "state_retention_authority",
        "entry_count",
        "entries",
        "atlas_sha256",
    }
    if set(atlas) != expected_fields:
        raise AdaptiveExpeditionAtlasError("Atlas fields differ")
    if (
        atlas.get("schema_version") != ATLAS_SCHEMA_V1
        or atlas.get("authority_effect") != "none"
        or atlas.get("state_retention_authority") is not False
    ):
        raise AdaptiveExpeditionAtlasError("Atlas authority boundary is invalid")
    _require_identifier(atlas.get("expedition_id"), field="expedition_id")
    _require_sha256(atlas.get("manifest_sha256"), field="manifest_sha256")
    _require_sha256(
        atlas.get("descriptor_set_sha256"), field="descriptor_set_sha256"
    )
    raw_entries = atlas.get("entries")
    if not isinstance(raw_entries, list) or atlas.get("entry_count") != len(raw_entries):
        raise AdaptiveExpeditionAtlasError("Atlas entry count is inconsistent")
    previous = atlas["descriptor_set_sha256"]
    entries: list[dict[str, Any]] = []
    entry_ids: set[str] = set()
    for ordinal, raw_entry in enumerate(raw_entries):
        entry = _validate_atlas_entry_v1(
            raw_entry,
            expected_ordinal=ordinal,
            previous_entry_sha256=previous,
        )
        if entry["entry_id"] in entry_ids:
            raise AdaptiveExpeditionAtlasError("Atlas entry identity was duplicated")
        entry_ids.add(entry["entry_id"])
        entries.append(entry)
        previous = entry["entry_sha256"]
    body = _atlas_body_v1(
        expedition_id=atlas["expedition_id"],
        manifest_sha256=atlas["manifest_sha256"],
        descriptor_set_sha256=atlas["descriptor_set_sha256"],
        entries=entries,
    )
    supplied = _require_sha256(atlas.get("atlas_sha256"), field="atlas_sha256")
    if supplied != _sha256(body):
        raise AdaptiveExpeditionAtlasError("Atlas revision digest changed")
    return atlas


def add_atlas_entry_v1(
    value: Mapping[str, Any], entry: Mapping[str, Any]
) -> dict[str, Any]:
    """Return a new Atlas revision containing one compact epistemic entry."""

    atlas = validate_expedition_atlas_v1(value)
    normalized = _normalize_entry_input_v1(entry)
    if any(row["entry_id"] == normalized["entry_id"] for row in atlas["entries"]):
        raise AdaptiveExpeditionAtlasError("Atlas entry identity was duplicated")
    previous = (
        atlas["entries"][-1]["entry_sha256"]
        if atlas["entries"]
        else atlas["descriptor_set_sha256"]
    )
    entry_body = {
        "schema_version": ATLAS_ENTRY_SCHEMA_V1,
        "entry_ordinal": len(atlas["entries"]),
        "previous_entry_sha256": previous,
        **normalized,
    }
    sealed_entry = {**entry_body, "entry_sha256": _sha256(entry_body)}
    entries = [*atlas["entries"], sealed_entry]
    body = _atlas_body_v1(
        expedition_id=atlas["expedition_id"],
        manifest_sha256=atlas["manifest_sha256"],
        descriptor_set_sha256=atlas["descriptor_set_sha256"],
        entries=entries,
    )
    candidate = {**body, "atlas_sha256": _sha256(body)}
    return validate_expedition_atlas_v1(candidate)


def atlas_entries_for_niche_v1(
    value: Mapping[str, Any], *, niche_id: str
) -> tuple[dict[str, Any], ...]:
    """Return derived entries for one niche without exposing retained state."""

    atlas = validate_expedition_atlas_v1(value)
    requested = _require_identifier(niche_id, field="niche_id")
    return tuple(
        dict(entry) for entry in atlas["entries"] if entry["niche_id"] == requested
    )


__all__ = [
    "AdaptiveExpeditionAtlasError",
    "add_atlas_entry_v1",
    "atlas_entries_for_niche_v1",
    "new_expedition_atlas_v1",
    "validate_expedition_atlas_v1",
]
