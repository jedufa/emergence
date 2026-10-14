"""Compact, digest-bound human review projection for expedition closeout."""

from __future__ import annotations

from collections.abc import Mapping
import json
from typing import Final

from emergence_contracts import canonical_json_bytes, canonical_sha256

from .review_projection import (
    REVIEW_SECTION_KEYS_V1,
    validate_human_review_projection_v1,
)


CLOSEOUT_SCHEMA_VERSION_V1: Final[str] = (
    "eiel_adaptive_science_expedition_closeout_v1"
)
ENGINEERING_G5_TERMINAL_STATUS_V1: Final[str] = (
    "ENGINEERING_G5_COMPLETE_AWAITING_HUMAN_REVIEW"
)
MAX_CLOSEOUT_BYTES_V1: Final[int] = 256 * 1024


class ExpeditionCloseoutError(ValueError):
    """Raised when a terminal human-review projection is incomplete or unsafe."""


def _mapping(value: object, *, field: str) -> dict[str, object]:
    if not isinstance(value, Mapping) or any(type(key) is not str for key in value):
        raise ExpeditionCloseoutError(f"{field} must be a canonical mapping")
    try:
        return dict(json.loads(canonical_json_bytes(dict(value))))
    except (TypeError, ValueError) as error:
        raise ExpeditionCloseoutError(f"{field} is not canonical JSON") from error


def _digest(value: object, *, field: str) -> dict[str, str]:
    parsed = _mapping(value, field=field)
    if set(parsed) != {"algorithm", "hex"}:
        raise ExpeditionCloseoutError(f"{field} must be a content digest")
    if parsed["algorithm"] != "sha256":
        raise ExpeditionCloseoutError(f"{field} must use SHA-256")
    hexadecimal = parsed["hex"]
    if (
        type(hexadecimal) is not str
        or len(hexadecimal) != 64
        or any(character not in "0123456789abcdef" for character in hexadecimal)
    ):
        raise ExpeditionCloseoutError(f"{field} must be lowercase SHA-256")
    return {"algorithm": "sha256", "hex": hexadecimal}


def build_expedition_closeout_v1(
    *,
    expedition_id: str,
    terminal_status: str,
    programme_stage: str,
    source_commit: str | None,
    manifest_sha256: Mapping[str, object],
    policy_sha256: Mapping[str, object],
    ledger_sha256: Mapping[str, object],
    atlas_sha256: Mapping[str, object],
    retained_parent_bank_sha256: Mapping[str, object],
    review_projection: Mapping[str, object],
    performance_summary: Mapping[str, object],
    lifecycle_summary: Mapping[str, object],
    validation_summary: Mapping[str, object],
) -> dict[str, object]:
    """Build the small terminal package without interpreting policy science."""

    if type(expedition_id) is not str or not expedition_id:
        raise ExpeditionCloseoutError("expedition_id must be non-empty text")
    if type(terminal_status) is not str or not terminal_status:
        raise ExpeditionCloseoutError("terminal_status must be non-empty text")
    if type(programme_stage) is not str or not programme_stage:
        raise ExpeditionCloseoutError("programme_stage must be non-empty text")
    if source_commit is not None and (
        type(source_commit) is not str
        or len(source_commit) != 40
        or any(character not in "0123456789abcdef" for character in source_commit)
    ):
        raise ExpeditionCloseoutError(
            "source_commit must be null or a full Git object ID"
        )
    if terminal_status not in {
        "COMPLETED",
        "STOPPED",
        "FAILED",
        "QUARANTINED",
        ENGINEERING_G5_TERMINAL_STATUS_V1,
    }:
        raise ExpeditionCloseoutError(
            "terminal_status is not an Expedition terminal state"
        )
    try:
        review = validate_human_review_projection_v1(review_projection)
    except ValueError as error:
        raise ExpeditionCloseoutError(str(error)) from error
    body: dict[str, object] = {
        "schema_version": CLOSEOUT_SCHEMA_VERSION_V1,
        "expedition_id": expedition_id,
        "terminal_status": terminal_status,
        "programme_stage": programme_stage,
        "source_commit": source_commit,
        "manifest_sha256": _digest(manifest_sha256, field="manifest_sha256"),
        "policy_sha256": _digest(policy_sha256, field="policy_sha256"),
        "ledger_sha256": _digest(ledger_sha256, field="ledger_sha256"),
        "atlas_sha256": _digest(atlas_sha256, field="atlas_sha256"),
        "retained_parent_bank_sha256": _digest(
            retained_parent_bank_sha256, field="retained_parent_bank_sha256"
        ),
        "review": review,
        "performance_summary": _mapping(
            performance_summary, field="performance_summary"
        ),
        "lifecycle_summary": _mapping(lifecycle_summary, field="lifecycle_summary"),
        "validation_summary": _mapping(
            validation_summary, field="validation_summary"
        ),
        "no_claims": True,
        "qualification_status": "NOT_REQUESTED",
        "authority_effect": "NONE_HUMAN_REVIEW_ONLY",
    }
    result = {**body, "closeout_sha256": canonical_sha256(body).to_dict()}
    if len(canonical_json_bytes(result)) > MAX_CLOSEOUT_BYTES_V1:
        raise ExpeditionCloseoutError("closeout exceeds the compact 256 KiB bound")
    return result


def validate_expedition_closeout_v1(value: object) -> dict[str, object]:
    parsed = _mapping(value, field="closeout")
    expected = {
        "schema_version",
        "expedition_id",
        "terminal_status",
        "programme_stage",
        "source_commit",
        "manifest_sha256",
        "policy_sha256",
        "ledger_sha256",
        "atlas_sha256",
        "retained_parent_bank_sha256",
        "review",
        "performance_summary",
        "lifecycle_summary",
        "validation_summary",
        "no_claims",
        "qualification_status",
        "authority_effect",
        "closeout_sha256",
    }
    if set(parsed) != expected:
        raise ExpeditionCloseoutError("closeout fields must be exact")
    rebuilt = build_expedition_closeout_v1(
        expedition_id=parsed["expedition_id"],
        terminal_status=parsed["terminal_status"],
        programme_stage=parsed["programme_stage"],
        source_commit=parsed["source_commit"],
        manifest_sha256=parsed["manifest_sha256"],
        policy_sha256=parsed["policy_sha256"],
        ledger_sha256=parsed["ledger_sha256"],
        atlas_sha256=parsed["atlas_sha256"],
        retained_parent_bank_sha256=parsed["retained_parent_bank_sha256"],
        review_projection=parsed["review"],
        performance_summary=parsed["performance_summary"],
        lifecycle_summary=parsed["lifecycle_summary"],
        validation_summary=parsed["validation_summary"],
    )
    if parsed != rebuilt:
        raise ExpeditionCloseoutError("closeout content or digest changed")
    return parsed


__all__ = (
    "CLOSEOUT_SCHEMA_VERSION_V1",
    "ENGINEERING_G5_TERMINAL_STATUS_V1",
    "ExpeditionCloseoutError",
    "MAX_CLOSEOUT_BYTES_V1",
    "REVIEW_SECTION_KEYS_V1",
    "build_expedition_closeout_v1",
    "validate_expedition_closeout_v1",
)
