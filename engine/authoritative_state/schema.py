"""Canonical EIEL authoritative-state schema v1 loader and validator.

The JSON package is the source of truth.  This module deliberately contains no
parallel field inventory: it verifies the content-addressed package and exposes
immutable lookups to boundary validators.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any, Mapping, NoReturn


SCHEMA_VERSION_V1 = "eiel_authoritative_state_v1"
SCHEMA_ROOT = (
    Path(__file__).resolve().parents[2] / "_spec" / "authoritative_state" / "v1"
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class AuthoritativeStateSchemaError(ValueError):
    """Stable fail-closed error raised by authoritative-state boundaries."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"{code}: {message}")


def _fail(code: str, message: str) -> NoReturn:
    raise AuthoritativeStateSchemaError(code, message)


def canonical_value(value: Any, *, path: str = "$") -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            _fail("EIEL_SCHEMA_NONFINITE", f"{path} is non-finite")
        return 0.0 if value == 0.0 else value
    if isinstance(value, Mapping):
        return {
            str(key): canonical_value(item, path=f"{path}.{key}")
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [
            canonical_value(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    _fail(
        "EIEL_SCHEMA_UNSUPPORTED_VALUE",
        f"{path} has unsupported type {type(value).__name__}",
    )


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        canonical_value(value),
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def content_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        _fail("EIEL_SCHEMA_UNREADABLE", f"cannot read {path}: {exc}")
    if not isinstance(value, dict):
        _fail("EIEL_SCHEMA_INVALID_ROOT", f"{path} root must be an object")
    return value


@dataclass(frozen=True, slots=True)
class AuthoritativeStateSchemaV1:
    root: Mapping[str, Any]
    fields: tuple[Mapping[str, Any], ...]
    fields_by_id: Mapping[str, Mapping[str, Any]]
    abi_variants: Mapping[str, Any]
    components: Mapping[str, Mapping[str, Any]]
    schema_hash: str

    def field(self, field_id: str) -> Mapping[str, Any]:
        try:
            return self.fields_by_id[field_id]
        except KeyError:
            _fail("EIEL_SCHEMA_UNKNOWN_FIELD", f"unknown field ID {field_id!r}")


def _validate_field(field: Mapping[str, Any], *, index: int) -> None:
    required = {
        "field_id",
        "name",
        "meaning",
        "semantic_owner",
        "producer",
        "consumers",
        "logical_type",
        "physical_type",
        "signedness",
        "bit_width",
        "scalar_or_tensor",
        "rank",
        "shape_formula",
        "capacity_formula",
        "valid_range",
        "sentinels",
        "optional",
        "mutable",
        "lifecycle",
        "birth_reset",
        "checkpoint_encoding",
        "chronicle_encoding",
        "provider_projection",
        "analysis_projection",
        "abi_variants",
        "artifact_participation",
        "compatibility_version",
        "migration_rule",
        "boundary_test_profile",
        "source_anchors",
    }
    missing = sorted(required - set(field))
    if missing:
        _fail(
            "EIEL_SCHEMA_FIELD_MISSING",
            f"fields[{index}] missing keys {missing}",
        )
    if not str(field["semantic_owner"]).strip():
        _fail("EIEL_SCHEMA_OWNER_UNRESOLVED", f"{field['field_id']} has no owner")
    if str(field["semantic_owner"]).lower() in {"unknown", "unresolved"}:
        _fail(
            "EIEL_SCHEMA_OWNER_UNRESOLVED",
            f"{field['field_id']} owner is unresolved",
        )
    if not isinstance(field["consumers"], list):
        _fail("EIEL_SCHEMA_FIELD_INVALID", f"{field['field_id']} consumers")
    if not isinstance(field["source_anchors"], list) or not field["source_anchors"]:
        _fail("EIEL_SCHEMA_FIELD_INVALID", f"{field['field_id']} source anchors")


def _validate_migration_families(component: Mapping[str, Any]) -> None:
    if (
        component.get("schema_version")
        != "eiel_authoritative_state_migration_families_v1"
    ):
        _fail(
            "EIEL_SCHEMA_MIGRATION_FAMILY_INVALID",
            "migration family schema version differs",
        )
    claimed = component.get("contract_hash")
    material = dict(component)
    material.pop("contract_hash", None)
    if (
        not isinstance(claimed, str)
        or _SHA256.fullmatch(claimed) is None
        or content_sha256(material) != claimed
    ):
        _fail(
            "EIEL_SCHEMA_MIGRATION_FAMILY_INVALID",
            "migration family contract hash differs",
        )
    categories = component.get("categories")
    expected_categories = {
        "compatible_no_op",
        "deterministic_conversion",
        "checkpoint_rewrite",
        "fresh_campaign",
        "unsupported",
    }
    if (
        not isinstance(categories, list)
        or set(categories) != expected_categories
        or len(categories) != len(expected_categories)
    ):
        _fail(
            "EIEL_SCHEMA_MIGRATION_FAMILY_INVALID",
            "migration family categories differ",
        )
    families = component.get("families")
    required = {
        "family_id",
        "surface_kind",
        "source_schema_version",
        "target_component",
        "target_contract_hash",
        "route_scope",
        "category",
        "operation",
        "rng_behavior",
        "validation",
        "rollback",
        "evidence",
        "legacy_exact_preservation",
        "admission",
    }
    if not isinstance(families, list) or not families:
        _fail(
            "EIEL_SCHEMA_MIGRATION_FAMILY_INVALID",
            "migration family rows are absent",
        )
    identities: set[str] = set()
    for index, family in enumerate(families):
        if not isinstance(family, Mapping) or set(family) != required:
            _fail(
                "EIEL_SCHEMA_MIGRATION_FAMILY_INVALID",
                f"migration family row {index} fields differ",
            )
        family_id = family["family_id"]
        if (
            not isinstance(family_id, str)
            or not family_id
            or family_id in identities
        ):
            _fail(
                "EIEL_SCHEMA_MIGRATION_FAMILY_INVALID",
                f"migration family row {index} identity is invalid",
            )
        identities.add(family_id)
        if family["category"] not in expected_categories:
            _fail(
                "EIEL_SCHEMA_MIGRATION_FAMILY_INVALID",
                f"migration family row {index} category is invalid",
            )
        if any(
            not isinstance(family[key], str) or not family[key]
            for key in required - {"family_id", "category"}
        ):
            _fail(
                "EIEL_SCHEMA_MIGRATION_FAMILY_INVALID",
                f"migration family row {index} contains an empty contract",
            )


def load_schema_v1(root: str | Path = SCHEMA_ROOT) -> AuthoritativeStateSchemaV1:
    package = Path(root)
    schema = _load_json(package / "schema.json")
    if schema.get("schema_version") != SCHEMA_VERSION_V1:
        _fail(
            "EIEL_SCHEMA_VERSION_MISMATCH",
            f"expected {SCHEMA_VERSION_V1}, got {schema.get('schema_version')!r}",
        )
    component_hashes = schema.get("component_hashes")
    if not isinstance(component_hashes, dict) or not component_hashes:
        _fail("EIEL_SCHEMA_COMPONENTS_MISSING", "component hash map is empty")
    components: dict[str, Mapping[str, Any]] = {}
    for filename, expected in sorted(component_hashes.items()):
        if not isinstance(filename, str) or "/" in filename or filename == "schema.json":
            _fail("EIEL_SCHEMA_COMPONENT_INVALID", f"invalid component {filename!r}")
        if not isinstance(expected, str) or _SHA256.fullmatch(expected) is None:
            _fail("EIEL_SCHEMA_COMPONENT_INVALID", f"invalid hash for {filename}")
        path = package / filename
        actual = file_sha256(path)
        if actual != expected:
            _fail(
                "EIEL_SCHEMA_COMPONENT_HASH_MISMATCH",
                f"{filename}: expected {expected}, got {actual}",
            )
        component = _load_json(path)
        if filename == "migration_families.json":
            _validate_migration_families(component)
        components[filename] = MappingProxyType(component)
    claimed = schema.get("schema_hash")
    material = dict(schema)
    material.pop("schema_hash", None)
    actual_schema_hash = content_sha256(material)
    if claimed != actual_schema_hash:
        _fail(
            "EIEL_SCHEMA_HASH_MISMATCH",
            f"expected {claimed}, computed {actual_schema_hash}",
        )
    registry = components.get("field_registry.json")
    if registry is None or not isinstance(registry.get("fields"), list):
        _fail("EIEL_SCHEMA_FIELDS_MISSING", "field_registry.json has no fields")
    fields = registry["fields"]
    identifiers: set[str] = set()
    by_id: dict[str, Mapping[str, Any]] = {}
    for index, field in enumerate(fields):
        if not isinstance(field, dict):
            _fail("EIEL_SCHEMA_FIELD_INVALID", f"fields[{index}] is not an object")
        _validate_field(field, index=index)
        field_id = str(field["field_id"])
        if field_id in identifiers:
            _fail("EIEL_SCHEMA_FIELD_DUPLICATE", field_id)
        identifiers.add(field_id)
        frozen = MappingProxyType(field)
        by_id[field_id] = frozen
    if registry.get("unresolved_supported_field_count") != 0:
        _fail(
            "EIEL_SCHEMA_SUPPORTED_FIELD_UNRESOLVED",
            "supported production fields remain unresolved",
        )
    abi = components.get("abi_variants.json")
    if abi is None:
        _fail("EIEL_SCHEMA_ABI_MISSING", "abi_variants.json is absent")
    return AuthoritativeStateSchemaV1(
        root=MappingProxyType(schema),
        fields=tuple(by_id[field_id] for field_id in sorted(by_id)),
        fields_by_id=MappingProxyType(by_id),
        abi_variants=abi,
        components=MappingProxyType(components),
        schema_hash=actual_schema_hash,
    )


def validate_boundary_v1(
    field_id: str,
    *,
    physical_type: str,
    rank: int,
    shape: tuple[int, ...],
    schema: AuthoritativeStateSchemaV1 | None = None,
) -> None:
    resolved = schema or load_schema_v1()
    field = resolved.field(field_id)
    if physical_type != field["physical_type"]:
        _fail(
            "EIEL_SCHEMA_DTYPE_MISMATCH",
            f"{field_id}: expected {field['physical_type']}, got {physical_type}",
        )
    if rank != field["rank"] or len(shape) != rank:
        _fail(
            "EIEL_SCHEMA_RANK_MISMATCH",
            f"{field_id}: expected rank {field['rank']}, got {rank}/{shape}",
        )
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in shape):
        _fail("EIEL_SCHEMA_SHAPE_INVALID", f"{field_id}: invalid shape {shape}")


__all__ = [
    "AuthoritativeStateSchemaError",
    "AuthoritativeStateSchemaV1",
    "SCHEMA_ROOT",
    "SCHEMA_VERSION_V1",
    "canonical_json_bytes",
    "canonical_value",
    "content_sha256",
    "file_sha256",
    "load_schema_v1",
    "validate_boundary_v1",
]
