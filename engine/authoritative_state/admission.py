"""Fail-closed admission helpers for authoritative-state component schemas."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import re
from typing import Any

from .identity import (
    canonical_json_bytes as _identity_canonical_json_bytes,
    canonical_sha256 as _identity_canonical_sha256,
)


FIELD_ID_PATTERN = re.compile(
    r"^[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)+\.v[1-9][0-9]*$"
)
HASH_PATTERN = re.compile(r"^[0-9a-f]{64}$")
MIGRATION_CATEGORIES = frozenset(
    {
        "identity",
        "rename_only",
        "add_optional_with_default",
        "add_required_with_constant",
        "lossless_widening",
        "reject",
    }
)


class SchemaAdmissionError(ValueError):
    """A schema component or admission receipt is incomplete or contradictory."""


def canonical_json_bytes(value: Any) -> bytes:
    """Encode canonical JSON, rejecting non-finite numbers."""

    try:
        return _identity_canonical_json_bytes(value)
    except (TypeError, ValueError) as exc:
        raise SchemaAdmissionError(f"value is not canonical JSON: {exc}") from exc


def canonical_sha256(value: Any) -> str:
    try:
        return _identity_canonical_sha256(value)
    except (TypeError, ValueError) as exc:
        raise SchemaAdmissionError(f"value is not canonical JSON: {exc}") from exc


def contract_hash(contract: Mapping[str, Any]) -> str:
    payload = dict(contract)
    payload.pop("contract_hash", None)
    return canonical_sha256(payload)


def require_sha256(value: Any, *, path: str) -> str:
    if not isinstance(value, str) or HASH_PATTERN.fullmatch(value) is None:
        raise SchemaAdmissionError(f"{path} must be a lowercase SHA-256")
    return value


def require_field_id(value: Any, *, path: str) -> str:
    if not isinstance(value, str) or FIELD_ID_PATTERN.fullmatch(value) is None:
        raise SchemaAdmissionError(f"{path} is not a stable versioned field ID")
    return value


def validate_contract_identity(contract: Mapping[str, Any]) -> str:
    """Validate a component contract's version and self-hash."""

    version = contract.get("schema_version")
    if not isinstance(version, str) or not version.endswith("_v1"):
        raise SchemaAdmissionError("schema_version must be a nonempty v1 identifier")
    declared = require_sha256(contract.get("contract_hash"), path="contract_hash")
    observed = contract_hash(contract)
    if declared != observed:
        raise SchemaAdmissionError(
            f"contract_hash mismatch: declared={declared} observed={observed}"
        )
    return declared


def _field_rows(rows: Any, *, path: str) -> dict[str, Mapping[str, Any]]:
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise SchemaAdmissionError(f"{path} must be an ordered array")
    indexed: dict[str, Mapping[str, Any]] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise SchemaAdmissionError(f"{path}[{index}] must be an object")
        field_id = require_field_id(
            row.get("field_id"), path=f"{path}[{index}].field_id"
        )
        if field_id in indexed:
            raise SchemaAdmissionError(f"{path} contains duplicate {field_id}")
        dtype = row.get("dtype")
        shape = row.get("shape")
        if not isinstance(dtype, str) or not dtype:
            raise SchemaAdmissionError(f"{path}[{index}].dtype must be nonempty")
        if not isinstance(shape, list) or any(
            not isinstance(item, (int, str)) for item in shape
        ):
            raise SchemaAdmissionError(
                f"{path}[{index}].shape must be an array of dimensions"
            )
        indexed[field_id] = row
    return indexed


def validate_field_manifest(
    contract_fields: Any,
    observed_fields: Any,
    *,
    allow_optional_omission: bool = False,
) -> tuple[str, ...]:
    """Require exact field IDs, dtypes, shapes, and order."""

    expected = _field_rows(contract_fields, path="contract.fields")
    observed = _field_rows(observed_fields, path="manifest.fields")
    required = {
        field_id
        for field_id, row in expected.items()
        if row.get("required", True) is True
    }
    missing = required - observed.keys()
    extra = observed.keys() - expected.keys()
    omitted_optional = expected.keys() - observed.keys() - required
    if missing or extra or (omitted_optional and not allow_optional_omission):
        raise SchemaAdmissionError(
            "field manifest mismatch: "
            f"missing={sorted(missing)} extra={sorted(extra)} "
            f"omitted_optional={sorted(omitted_optional)}"
        )
    expected_order = [field_id for field_id in expected if field_id in observed]
    if list(observed) != expected_order:
        raise SchemaAdmissionError("field manifest order mismatch")
    for field_id, actual in observed.items():
        wanted = expected[field_id]
        if actual.get("dtype") != wanted.get("dtype"):
            raise SchemaAdmissionError(f"{field_id} dtype mismatch")
        if actual.get("shape") != wanted.get("shape"):
            raise SchemaAdmissionError(f"{field_id} shape mismatch")
    return tuple(observed)


def validate_exact_bindings(
    observed: Any,
    expected: Mapping[str, str],
    *,
    path: str = "bindings",
) -> None:
    if not isinstance(observed, Mapping):
        raise SchemaAdmissionError(f"{path} must be an object")
    if set(observed) != set(expected):
        raise SchemaAdmissionError(
            f"{path} keys mismatch: expected={sorted(expected)} "
            f"observed={sorted(observed)}"
        )
    for key, wanted in expected.items():
        require_sha256(wanted, path=f"expected.{key}")
        actual = require_sha256(observed.get(key), path=f"{path}.{key}")
        if actual != wanted:
            raise SchemaAdmissionError(f"{path}.{key} mismatch")


def validate_migration_rules(document: Mapping[str, Any]) -> str:
    declared_hash = validate_contract_identity(document)
    rules = document.get("rules")
    if not isinstance(rules, list):
        raise SchemaAdmissionError("migration rules must be an ordered array")
    identities: set[tuple[str, str, str]] = set()
    for index, rule in enumerate(rules):
        if not isinstance(rule, Mapping):
            raise SchemaAdmissionError(f"rules[{index}] must be an object")
        component = rule.get("component")
        category = rule.get("category")
        if not isinstance(component, str) or not component:
            raise SchemaAdmissionError(f"rules[{index}].component is invalid")
        if category not in MIGRATION_CATEGORIES:
            raise SchemaAdmissionError(f"rules[{index}].category is unsupported")
        source = require_sha256(
            rule.get("from_contract_hash"), path=f"rules[{index}].from_contract_hash"
        )
        target = require_sha256(
            rule.get("to_contract_hash"), path=f"rules[{index}].to_contract_hash"
        )
        identity = (component, source, target)
        if identity in identities:
            raise SchemaAdmissionError(f"duplicate migration rule {identity}")
        identities.add(identity)
        details = rule.get("details")
        if not isinstance(details, Mapping):
            raise SchemaAdmissionError(f"rules[{index}].details must be an object")
        if category == "rename_only":
            require_field_id(details.get("from_field_id"), path="from_field_id")
            require_field_id(details.get("to_field_id"), path="to_field_id")
        elif category in {"add_optional_with_default", "add_required_with_constant"}:
            require_field_id(details.get("field_id"), path="field_id")
            if "value" not in details:
                raise SchemaAdmissionError(f"rules[{index}] lacks deterministic value")
        elif category == "lossless_widening":
            require_field_id(details.get("field_id"), path="field_id")
            if not all(
                isinstance(details.get(key), str) and details.get(key)
                for key in ("from_dtype", "to_dtype")
            ):
                raise SchemaAdmissionError(
                    f"rules[{index}] lacks explicit widening dtypes"
                )
        elif category in {"identity", "reject"} and details:
            raise SchemaAdmissionError(
                f"rules[{index}] category {category} must have empty details"
            )
    return declared_hash


def resolve_migration(
    document: Mapping[str, Any],
    *,
    component: str,
    from_contract_hash: str,
    to_contract_hash: str,
) -> Mapping[str, Any]:
    """Select one exact deterministic rule; absence and ambiguity fail closed."""

    validate_migration_rules(document)
    source = require_sha256(from_contract_hash, path="from_contract_hash")
    target = require_sha256(to_contract_hash, path="to_contract_hash")
    matches = [
        rule
        for rule in document["rules"]
        if rule["component"] == component
        and rule["from_contract_hash"] == source
        and rule["to_contract_hash"] == target
    ]
    if len(matches) != 1:
        raise SchemaAdmissionError(
            f"expected one migration rule for {component}, found {len(matches)}"
        )
    if matches[0]["category"] == "reject":
        raise SchemaAdmissionError(f"migration for {component} is explicitly rejected")
    return matches[0]


def validate_compatibility_matrix(
    document: Mapping[str, Any],
    *,
    observed: Mapping[str, str],
) -> Mapping[str, Any]:
    """Admit only an exact component-hash tuple listed by the matrix."""

    validate_contract_identity(document)
    if not observed:
        raise SchemaAdmissionError("observed compatibility tuple is empty")
    for component, digest in observed.items():
        if not isinstance(component, str) or not component:
            raise SchemaAdmissionError("compatibility component name is invalid")
        require_sha256(digest, path=f"observed.{component}")
    variants = document.get("compatible_sets")
    if not isinstance(variants, list):
        raise SchemaAdmissionError("compatible_sets must be an ordered array")
    matches = []
    for index, variant in enumerate(variants):
        if not isinstance(variant, Mapping):
            raise SchemaAdmissionError(f"compatible_sets[{index}] must be an object")
        bindings = variant.get("bindings")
        if not isinstance(bindings, Mapping):
            raise SchemaAdmissionError(
                f"compatible_sets[{index}].bindings must be an object"
            )
        if dict(bindings) == dict(observed):
            matches.append(variant)
    if len(matches) != 1:
        raise SchemaAdmissionError(
            f"compatibility tuple must match exactly one set; matches={len(matches)}"
        )
    return matches[0]
