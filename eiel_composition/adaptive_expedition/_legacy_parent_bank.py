"""Legacy test-only Retained Parent Bank compatibility helpers.

The bank owns references and protected holds only.  Removing or evicting a bank
entry drops that reference; it never grants filesystem deletion, lifecycle
release, archival, execution, or qualification authority.  This private module
is retained only for legacy safety fixtures; the production/public contract is
``RetainedParentBankV1`` in ``emergence_contracts``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import re
from typing import Any, Final

from emergence_contracts import canonical_json_bytes


PARENT_BANK_SCHEMA_V1: Final[str] = "eiel_adaptive_expedition_parent_bank_v1"
PARENT_BANK_ENTRY_SCHEMA_V1: Final[str] = (
    "eiel_adaptive_expedition_retained_parent_v1"
)
PARENT_BANK_UPDATE_SCHEMA_V1: Final[str] = (
    "eiel_adaptive_expedition_parent_bank_update_v1"
)
STATE_REF_KINDS_V1: Final[frozenset[str]] = frozenset(
    {"BRANCHABLE_EXACT", "REPLAY_DERIVABLE"}
)

_CAP_FIELDS_V1: Final[frozenset[str]] = frozenset(
    {"max_entries", "max_total_bytes"}
)
_STATE_REF_FIELDS_V1: Final[frozenset[str]] = frozenset(
    {
        "kind",
        "state_ref_sha256",
        "state_bytes",
        "lifecycle_manifest_sha256",
        "lifecycle_object_id",
        "next_absolute_tick",
        "semantic_world_sha256",
    }
)
_HOLD_FIELDS_V1: Final[frozenset[str]] = frozenset(
    {"hold_id", "hold_kind", "reference_sha256"}
)
_ENTRY_INPUT_FIELDS_V1: Final[frozenset[str]] = frozenset(
    {
        "parent_id",
        "candidate_id",
        "niche_id",
        "frontier_ordinal",
        "retention_priority",
        "state_ref",
        "holds",
    }
)


class AdaptiveExpeditionParentBankError(ValueError):
    """Base failure for an invalid Retained Parent Bank."""


class ParentBankCapacityError(AdaptiveExpeditionParentBankError):
    """Manifest caps cannot be satisfied without evicting a protected entry."""


class ParentBankIntegrityError(AdaptiveExpeditionParentBankError):
    """A bank, retained entry, state reference, or hold is inconsistent."""


def _canonical_clone(value: Any, *, label: str) -> Any:
    try:
        return json.loads(canonical_json_bytes(value))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ParentBankIntegrityError(f"{label} is not canonical JSON data") from exc


def _sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _require_sha256(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or re.fullmatch(r"[0-9a-f]{64}", value) is None
    ):
        raise ParentBankIntegrityError(f"{field} must be a lowercase SHA-256")
    return value


def _require_identifier(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 512:
        raise ParentBankIntegrityError(
            f"{field} must be a bounded non-empty string"
        )
    return value


def _normalize_caps_v1(value: Mapping[str, Any]) -> dict[str, int]:
    if not isinstance(value, Mapping) or set(value) != _CAP_FIELDS_V1:
        raise ParentBankIntegrityError("parent bank caps fields differ")
    caps = _canonical_clone(dict(value), label="parent bank caps")
    if (
        type(caps["max_entries"]) is not int
        or caps["max_entries"] <= 0
        or type(caps["max_total_bytes"]) is not int
        or caps["max_total_bytes"] <= 0
    ):
        raise ParentBankIntegrityError("parent bank caps must be positive")
    return caps


def _normalize_state_ref_v1(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _STATE_REF_FIELDS_V1:
        raise ParentBankIntegrityError("retained state_ref fields differ")
    state_ref = _canonical_clone(dict(value), label="retained state_ref")
    if state_ref["kind"] not in STATE_REF_KINDS_V1:
        raise ParentBankIntegrityError("retained state_ref kind is unsupported")
    for field in (
        "state_ref_sha256",
        "lifecycle_manifest_sha256",
        "semantic_world_sha256",
    ):
        _require_sha256(state_ref[field], field=field)
    _require_identifier(
        state_ref["lifecycle_object_id"], field="lifecycle_object_id"
    )
    if type(state_ref["state_bytes"]) is not int or state_ref["state_bytes"] <= 0:
        raise ParentBankIntegrityError("state_bytes must be an exact positive integer")
    if (
        type(state_ref["next_absolute_tick"]) is not int
        or state_ref["next_absolute_tick"] < 0
    ):
        raise ParentBankIntegrityError(
            "state_ref next_absolute_tick must be a non-negative integer"
        )
    return state_ref


def _normalize_hold_v1(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _HOLD_FIELDS_V1:
        raise ParentBankIntegrityError("parent hold fields differ")
    hold = _canonical_clone(dict(value), label="parent hold")
    _require_identifier(hold["hold_id"], field="hold_id")
    _require_identifier(hold["hold_kind"], field="hold_kind")
    _require_sha256(hold["reference_sha256"], field="reference_sha256")
    return hold


def _normalize_holds_v1(value: object) -> list[dict[str, Any]]:
    if (
        isinstance(value, (str, bytes, bytearray))
        or not isinstance(value, Sequence)
    ):
        raise ParentBankIntegrityError("parent holds must be an array")
    holds = [_normalize_hold_v1(item) for item in value]
    hold_ids = [hold["hold_id"] for hold in holds]
    if len(set(hold_ids)) != len(hold_ids):
        raise ParentBankIntegrityError("parent hold identity was duplicated")
    return sorted(holds, key=lambda hold: hold["hold_id"])


def _normalize_entry_input_v1(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _ENTRY_INPUT_FIELDS_V1:
        raise ParentBankIntegrityError("retained parent input fields differ")
    entry = _canonical_clone(dict(value), label="retained parent input")
    for field in ("parent_id", "candidate_id", "niche_id"):
        _require_identifier(entry[field], field=field)
    if type(entry["frontier_ordinal"]) is not int or entry["frontier_ordinal"] < 0:
        raise ParentBankIntegrityError(
            "frontier_ordinal must be a non-negative integer"
        )
    if type(entry["retention_priority"]) is not int:
        raise ParentBankIntegrityError("retention_priority must be an integer")
    entry["state_ref"] = _normalize_state_ref_v1(entry["state_ref"])
    entry["holds"] = _normalize_holds_v1(entry["holds"])
    return entry


def _seal_entry_v1(value: Mapping[str, Any], *, retained_ordinal: int) -> dict[str, Any]:
    normalized = _normalize_entry_input_v1(value)
    if type(retained_ordinal) is not int or retained_ordinal < 0:
        raise ParentBankIntegrityError("retained_ordinal must be non-negative")
    body = {
        "schema_version": PARENT_BANK_ENTRY_SCHEMA_V1,
        "retained_ordinal": retained_ordinal,
        **normalized,
    }
    return {**body, "entry_sha256": _sha256(body)}


def _entry_input_v1(entry: Mapping[str, Any]) -> dict[str, Any]:
    return {field: entry[field] for field in _ENTRY_INPUT_FIELDS_V1}


def _validate_entry_v1(raw: object) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise ParentBankIntegrityError("retained parent entry must be an object")
    entry = _canonical_clone(dict(raw), label="retained parent entry")
    expected_fields = {
        *_ENTRY_INPUT_FIELDS_V1,
        "schema_version",
        "retained_ordinal",
        "entry_sha256",
    }
    if set(entry) != expected_fields:
        raise ParentBankIntegrityError("retained parent entry fields differ")
    if entry.get("schema_version") != PARENT_BANK_ENTRY_SCHEMA_V1:
        raise ParentBankIntegrityError("retained parent schema version is invalid")
    _normalize_entry_input_v1(_entry_input_v1(entry))
    if type(entry["retained_ordinal"]) is not int or entry["retained_ordinal"] < 0:
        raise ParentBankIntegrityError("retained parent ordinal is invalid")
    body = {key: value for key, value in entry.items() if key != "entry_sha256"}
    supplied = _require_sha256(entry.get("entry_sha256"), field="entry_sha256")
    if supplied != _sha256(body):
        raise ParentBankIntegrityError("retained parent entry digest changed")
    return entry


def _bank_body_v1(
    *,
    expedition_id: str,
    manifest_sha256: str,
    caps: Mapping[str, int],
    bank_revision: int,
    next_retained_ordinal: int,
    entries: list[dict[str, Any]],
) -> dict[str, Any]:
    ordered = sorted(entries, key=lambda entry: entry["retained_ordinal"])
    return {
        "schema_version": PARENT_BANK_SCHEMA_V1,
        "expedition_id": expedition_id,
        "manifest_sha256": manifest_sha256,
        "caps": dict(caps),
        "bank_revision": bank_revision,
        "next_retained_ordinal": next_retained_ordinal,
        "entry_count": len(ordered),
        "total_state_bytes": sum(
            int(entry["state_ref"]["state_bytes"]) for entry in ordered
        ),
        "entries": ordered,
        "authority_effect": "reference_retention_only",
        "deletion_authority": False,
        "lifecycle_release_authority": False,
    }


def new_retained_parent_bank_v1(
    *, expedition_id: str, manifest_sha256: str, caps: Mapping[str, Any]
) -> dict[str, Any]:
    """Create an empty manifest-capped Retained Parent Bank."""

    body = _bank_body_v1(
        expedition_id=_require_identifier(expedition_id, field="expedition_id"),
        manifest_sha256=_require_sha256(
            manifest_sha256, field="manifest_sha256"
        ),
        caps=_normalize_caps_v1(caps),
        bank_revision=0,
        next_retained_ordinal=0,
        entries=[],
    )
    return {**body, "bank_sha256": _sha256(body)}


def validate_retained_parent_bank_v1(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate caps, exact state/lifecycle refs, entries, and bank digest."""

    if not isinstance(value, Mapping):
        raise ParentBankIntegrityError("parent bank must be an object")
    bank = _canonical_clone(dict(value), label="parent bank")
    expected_fields = {
        "schema_version",
        "expedition_id",
        "manifest_sha256",
        "caps",
        "bank_revision",
        "next_retained_ordinal",
        "entry_count",
        "total_state_bytes",
        "entries",
        "authority_effect",
        "deletion_authority",
        "lifecycle_release_authority",
        "bank_sha256",
    }
    if set(bank) != expected_fields:
        raise ParentBankIntegrityError("parent bank fields differ")
    if (
        bank.get("schema_version") != PARENT_BANK_SCHEMA_V1
        or bank.get("authority_effect") != "reference_retention_only"
        or bank.get("deletion_authority") is not False
        or bank.get("lifecycle_release_authority") is not False
    ):
        raise ParentBankIntegrityError("parent bank authority boundary is invalid")
    expedition_id = _require_identifier(bank["expedition_id"], field="expedition_id")
    manifest_sha256 = _require_sha256(
        bank["manifest_sha256"], field="manifest_sha256"
    )
    caps = _normalize_caps_v1(bank["caps"])
    if (
        type(bank["bank_revision"]) is not int
        or bank["bank_revision"] < 0
        or type(bank["next_retained_ordinal"]) is not int
        or bank["next_retained_ordinal"] < 0
    ):
        raise ParentBankIntegrityError("parent bank revision cursor is invalid")
    raw_entries = bank["entries"]
    if not isinstance(raw_entries, list):
        raise ParentBankIntegrityError("parent bank entries must be an array")
    entries = [_validate_entry_v1(entry) for entry in raw_entries]
    ordinals = [entry["retained_ordinal"] for entry in entries]
    if (
        ordinals != sorted(ordinals)
        or len(set(ordinals)) != len(ordinals)
        or any(ordinal >= bank["next_retained_ordinal"] for ordinal in ordinals)
    ):
        raise ParentBankIntegrityError("retained parent ordinals are inconsistent")
    parent_ids = [entry["parent_id"] for entry in entries]
    state_refs = [entry["state_ref"]["state_ref_sha256"] for entry in entries]
    if len(set(parent_ids)) != len(parent_ids):
        raise ParentBankIntegrityError("retained parent identity was duplicated")
    if len(set(state_refs)) != len(state_refs):
        raise ParentBankIntegrityError("retained state identity was duplicated")
    total_bytes = sum(entry["state_ref"]["state_bytes"] for entry in entries)
    if (
        bank["entry_count"] != len(entries)
        or bank["total_state_bytes"] != total_bytes
        or len(entries) > caps["max_entries"]
        or total_bytes > caps["max_total_bytes"]
    ):
        raise ParentBankIntegrityError("parent bank manifest caps are violated")
    body = _bank_body_v1(
        expedition_id=expedition_id,
        manifest_sha256=manifest_sha256,
        caps=caps,
        bank_revision=bank["bank_revision"],
        next_retained_ordinal=bank["next_retained_ordinal"],
        entries=entries,
    )
    supplied = _require_sha256(bank.get("bank_sha256"), field="bank_sha256")
    if supplied != _sha256(body):
        raise ParentBankIntegrityError("parent bank revision digest changed")
    return bank


def _eviction_reference_v1(
    entry: Mapping[str, Any], *, reason: str
) -> dict[str, Any]:
    state_ref = entry["state_ref"]
    return {
        "parent_id": entry["parent_id"],
        "niche_id": entry["niche_id"],
        "state_ref_sha256": state_ref["state_ref_sha256"],
        "state_bytes": state_ref["state_bytes"],
        "lifecycle_manifest_sha256": state_ref["lifecycle_manifest_sha256"],
        "lifecycle_object_id": state_ref["lifecycle_object_id"],
        "reason": reason,
        "effect": "DROP_BANK_REFERENCE_ONLY",
        "deletion_authority": False,
        "lifecycle_release_authority": False,
    }


def _bank_update_v1(
    *,
    bank: Mapping[str, Any],
    retained_parent_id: str | None,
    removed: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    body = {
        "schema_version": PARENT_BANK_UPDATE_SCHEMA_V1,
        "bank": dict(bank),
        "retained_parent_id": retained_parent_id,
        "removed_references": [dict(item) for item in removed],
        "effect": "REFERENCE_SET_UPDATE_ONLY",
        "deletion_authority": False,
        "lifecycle_release_authority": False,
    }
    return {**body, "update_sha256": _sha256(body)}


def _over_capacity_v1(
    entries: Sequence[Mapping[str, Any]], caps: Mapping[str, int]
) -> bool:
    return (
        len(entries) > caps["max_entries"]
        or sum(entry["state_ref"]["state_bytes"] for entry in entries)
        > caps["max_total_bytes"]
    )


def retain_parent_v1(
    value: Mapping[str, Any], parent: Mapping[str, Any]
) -> dict[str, Any]:
    """Retain one branchable/replay state ref and deterministically evict old refs."""

    bank = validate_retained_parent_bank_v1(value)
    normalized = _normalize_entry_input_v1(parent)
    if any(entry["parent_id"] == normalized["parent_id"] for entry in bank["entries"]):
        raise ParentBankIntegrityError("retained parent identity was duplicated")
    if any(
        entry["state_ref"]["state_ref_sha256"]
        == normalized["state_ref"]["state_ref_sha256"]
        for entry in bank["entries"]
    ):
        raise ParentBankIntegrityError("retained state identity was duplicated")
    caps = bank["caps"]
    if normalized["state_ref"]["state_bytes"] > caps["max_total_bytes"]:
        raise ParentBankCapacityError("one retained state exceeds the bank byte cap")
    new_entry = _seal_entry_v1(
        normalized, retained_ordinal=bank["next_retained_ordinal"]
    )
    entries = [*bank["entries"], new_entry]
    removed: list[dict[str, Any]] = []

    while True:
        if not _over_capacity_v1(entries, caps):
            break
        pool = [
            entry
            for entry in entries
            if entry["parent_id"] != new_entry["parent_id"]
            and not entry["holds"]
        ]
        if not pool:
            raise ParentBankCapacityError(
                "parent bank caps require eviction but every eligible prior entry is held"
            )
        victim = min(
            pool,
            key=lambda entry: (
                entry["retention_priority"],
                entry["frontier_ordinal"],
                entry["retained_ordinal"],
                entry["parent_id"],
            ),
        )
        entries = [
            entry for entry in entries if entry["parent_id"] != victim["parent_id"]
        ]
        removed.append(_eviction_reference_v1(victim, reason="CAPACITY_EVICTION"))

    body = _bank_body_v1(
        expedition_id=bank["expedition_id"],
        manifest_sha256=bank["manifest_sha256"],
        caps=caps,
        bank_revision=bank["bank_revision"] + 1,
        next_retained_ordinal=bank["next_retained_ordinal"] + 1,
        entries=entries,
    )
    updated = validate_retained_parent_bank_v1(
        {**body, "bank_sha256": _sha256(body)}
    )
    return _bank_update_v1(
        bank=updated,
        retained_parent_id=new_entry["parent_id"],
        removed=removed,
    )


def _replace_entry_holds_v1(
    bank: Mapping[str, Any], *, parent_id: str, holds: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    entries = list(bank["entries"])
    selected = next(
        (entry for entry in entries if entry["parent_id"] == parent_id), None
    )
    if selected is None:
        raise ParentBankIntegrityError(f"unknown retained parent: {parent_id}")
    input_value = _entry_input_v1(selected)
    input_value["holds"] = [dict(hold) for hold in holds]
    replacement = _seal_entry_v1(
        input_value, retained_ordinal=selected["retained_ordinal"]
    )
    entries = [
        replacement if entry["parent_id"] == parent_id else entry for entry in entries
    ]
    body = _bank_body_v1(
        expedition_id=bank["expedition_id"],
        manifest_sha256=bank["manifest_sha256"],
        caps=bank["caps"],
        bank_revision=bank["bank_revision"] + 1,
        next_retained_ordinal=bank["next_retained_ordinal"],
        entries=entries,
    )
    return validate_retained_parent_bank_v1(
        {**body, "bank_sha256": _sha256(body)}
    )


def add_parent_hold_v1(
    value: Mapping[str, Any], *, parent_id: str, hold: Mapping[str, Any]
) -> dict[str, Any]:
    """Add one exact protected hold without changing lifecycle authority."""

    bank = validate_retained_parent_bank_v1(value)
    identifier = _require_identifier(parent_id, field="parent_id")
    normalized = _normalize_hold_v1(hold)
    entry = next(
        (item for item in bank["entries"] if item["parent_id"] == identifier), None
    )
    if entry is None:
        raise ParentBankIntegrityError(f"unknown retained parent: {identifier}")
    if any(item["hold_id"] == normalized["hold_id"] for item in entry["holds"]):
        raise ParentBankIntegrityError("parent hold identity was duplicated")
    return _replace_entry_holds_v1(
        bank,
        parent_id=identifier,
        holds=[*entry["holds"], normalized],
    )


def release_parent_hold_v1(
    value: Mapping[str, Any], *, parent_id: str, hold_id: str
) -> dict[str, Any]:
    """Release one bank hold; this does not release or delete lifecycle bytes."""

    bank = validate_retained_parent_bank_v1(value)
    identifier = _require_identifier(parent_id, field="parent_id")
    requested_hold = _require_identifier(hold_id, field="hold_id")
    entry = next(
        (item for item in bank["entries"] if item["parent_id"] == identifier), None
    )
    if entry is None:
        raise ParentBankIntegrityError(f"unknown retained parent: {identifier}")
    if not any(item["hold_id"] == requested_hold for item in entry["holds"]):
        raise ParentBankIntegrityError(f"unknown parent hold: {requested_hold}")
    return _replace_entry_holds_v1(
        bank,
        parent_id=identifier,
        holds=[item for item in entry["holds"] if item["hold_id"] != requested_hold],
    )


def drop_parent_reference_v1(
    value: Mapping[str, Any], *, parent_id: str, reason: str
) -> dict[str, Any]:
    """Drop an unheld bank reference and emit an explicit no-delete update."""

    bank = validate_retained_parent_bank_v1(value)
    identifier = _require_identifier(parent_id, field="parent_id")
    drop_reason = _require_identifier(reason, field="reason")
    entry = next(
        (item for item in bank["entries"] if item["parent_id"] == identifier), None
    )
    if entry is None:
        raise ParentBankIntegrityError(f"unknown retained parent: {identifier}")
    if entry["holds"]:
        raise ParentBankCapacityError("a protected retained parent cannot be dropped")
    entries = [item for item in bank["entries"] if item["parent_id"] != identifier]
    body = _bank_body_v1(
        expedition_id=bank["expedition_id"],
        manifest_sha256=bank["manifest_sha256"],
        caps=bank["caps"],
        bank_revision=bank["bank_revision"] + 1,
        next_retained_ordinal=bank["next_retained_ordinal"],
        entries=entries,
    )
    updated = validate_retained_parent_bank_v1(
        {**body, "bank_sha256": _sha256(body)}
    )
    removed = [_eviction_reference_v1(entry, reason=drop_reason)]
    return _bank_update_v1(
        bank=updated,
        retained_parent_id=None,
        removed=removed,
    )


__all__ = [
    "AdaptiveExpeditionParentBankError",
    "ParentBankCapacityError",
    "ParentBankIntegrityError",
    "add_parent_hold_v1",
    "drop_parent_reference_v1",
    "new_retained_parent_bank_v1",
    "release_parent_hold_v1",
    "retain_parent_v1",
    "validate_retained_parent_bank_v1",
]
