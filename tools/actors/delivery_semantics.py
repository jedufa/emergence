"""Compatibility accessors for generic delivery terminology.

Current writers use ``delivery_*`` names. Historical V2/V3 Session and Review
artifacts used ``implementation_*`` names even for architecture and engineering
deliveries. These accessors keep those old bytes readable without allowing the
legacy vocabulary to remain authoritative in newly written records.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def delivery_session_id(raw: Mapping[str, Any] | None) -> str:
    value = dict(raw or {})
    return str(value.get("delivery_session_id") or value.get("implementation_session_id") or "")


def delivery_commit(raw: Mapping[str, Any] | None) -> str:
    value = dict(raw or {})
    return str(value.get("delivery_commit") or value.get("implementation_commit") or "")


def delivery_sufficiency(raw: Mapping[str, Any] | None) -> str:
    value = dict(raw or {})
    return str(value.get("delivery_sufficiency") or value.get("implementation_sufficiency") or "")


def delivering_codex_identities(raw: Mapping[str, Any] | None) -> list[str]:
    value = dict(raw or {})
    rows = value.get("delivering_codex_identities")
    if rows is None:
        rows = value.get("implementing_codex_identities")
    return [str(item) for item in (rows or []) if str(item).strip()]


def delivery_identity_assurance(raw: Mapping[str, Any] | None) -> Any:
    value = dict(raw or {})
    return value.get("delivery_identity_assurance", value.get("implementing_identity_assurance"))


def delivery_model_assignment(raw: Mapping[str, Any] | None) -> Any:
    value = dict(raw or {})
    return value.get("delivery_model_assignment", value.get("implementation_model_assignment"))


def canonicalize_delivery_evidence(raw: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(raw)
    value.setdefault("delivery_session_id", delivery_session_id(value))
    value.setdefault("delivery_commit", delivery_commit(value))
    value.setdefault("delivery_sufficiency", delivery_sufficiency(value) or "UNASSESSED")
    return value


def canonicalize_review_target(raw: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(raw)
    value.setdefault("delivery_session_id", delivery_session_id(value))
    value.setdefault("delivering_codex_identities", delivering_codex_identities(value))
    value.setdefault("delivery_identity_assurance", delivery_identity_assurance(value))
    material = value.get("delivery_material")
    if isinstance(material, Mapping):
        material_value = dict(material)
        material_value.setdefault("delivery_session_id", delivery_session_id(material_value))
        material_value.setdefault("delivery_commit", delivery_commit(material_value))
        value["delivery_material"] = material_value
    return value


def canonicalize_review_evidence(raw: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(raw)
    value.setdefault("delivery_session_id", delivery_session_id(value))
    value.setdefault("delivering_codex_identities", delivering_codex_identities(value))
    value.setdefault("delivery_identity_assurance", delivery_identity_assurance(value))
    value.setdefault("delivery_model_assignment", delivery_model_assignment(value))
    value.setdefault("delivery_sufficiency", delivery_sufficiency(value) or "UNASSESSED")
    return value


__all__ = [
    "canonicalize_delivery_evidence",
    "canonicalize_review_evidence",
    "canonicalize_review_target",
    "delivery_commit",
    "delivery_identity_assurance",
    "delivery_model_assignment",
    "delivery_session_id",
    "delivery_sufficiency",
    "delivering_codex_identities",
]
