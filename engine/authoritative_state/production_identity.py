"""Exact admission identity shared by request, provider, manifest, and binary."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import copy
import re
from typing import Any, NoReturn

from .identity import AbiVariantIdentityV1, canonical_json_bytes, canonical_sha256
from .schema import AuthoritativeStateSchemaError
from .schema import AuthoritativeStateSchemaV1


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_FIELDS = frozenset(
    {
        "contract_schema_version",
        "state_schema_version",
        "state_schema_sha256",
        "source_commit",
        "source_inventory_sha256",
        "build_identity_sha256",
        "abi_set_sha256",
        "abi_variants",
        "checkpoint_schema_version",
        "checkpoint_schema_sha256",
        "chronicle_schema_version",
        "chronicle_schema_sha256",
        "provider_capability_identity",
        "overlay_version",
        "supported_cache_widths",
        "supported_profiles",
        "build_toolchain_identity",
        "compile_flags",
        "artifact_build_id",
    }
)


def _fail(code: str, message: str) -> NoReturn:
    raise AuthoritativeStateSchemaError(code, message)


def _sha(value: Any, *, path: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        _fail("EIEL_ADMISSION_INVALID_HASH", path)
    return value


def validate_production_identity_v1(value: Any, *, path: str) -> dict[str, Any]:
    observed_fields = set(value) if isinstance(value, Mapping) else set()
    if observed_fields != _FIELDS:
        _fail(
            "EIEL_ADMISSION_IDENTITY_FIELDS",
            f"{path}: missing={sorted(_FIELDS - observed_fields)} "
            f"extra={sorted(observed_fields - _FIELDS)}",
        )
    result = copy.deepcopy(dict(value))
    if result["contract_schema_version"] != "eiel_authoritative_state_binary_identity_v1":
        _fail("EIEL_ADMISSION_IDENTITY_VERSION", path)
    for field in (
        "state_schema_sha256",
        "source_inventory_sha256",
        "build_identity_sha256",
        "abi_set_sha256",
        "checkpoint_schema_sha256",
        "chronicle_schema_sha256",
    ):
        _sha(result[field], path=f"{path}.{field}")
    if not isinstance(result["source_commit"], str) or _COMMIT.fullmatch(
        result["source_commit"]
    ) is None:
        _fail("EIEL_ADMISSION_SOURCE_COMMIT", path)
    variants = result["abi_variants"]
    if not isinstance(variants, Sequence) or isinstance(variants, (str, bytes)):
        _fail("EIEL_ADMISSION_ABI_INVALID", path)
    parsed = tuple(
        AbiVariantIdentityV1.from_mapping(row, path=f"{path}.abi_variants[{index}]")
        for index, row in enumerate(variants)
    )
    keys = tuple((row.entrypoint, row.abi_version) for row in parsed)
    if keys != tuple(sorted(set(keys))):
        _fail("EIEL_ADMISSION_ABI_ORDER", path)
    observed_abi_hash = canonical_sha256([row.payload() for row in parsed])
    if result["abi_set_sha256"] != observed_abi_hash:
        _fail("EIEL_ADMISSION_ABI_HASH", path)
    widths = result["supported_cache_widths"]
    if (
        not isinstance(widths, (list, tuple))
        or not widths
        or any(type(width) is not int or width <= 0 for width in widths)
        or tuple(widths) != tuple(sorted(set(widths)))
    ):
        _fail("EIEL_ADMISSION_CACHE_WIDTHS", path)
    profiles = result["supported_profiles"]
    if (
        not isinstance(profiles, (list, tuple))
        or not profiles
        or any(not isinstance(profile, str) or not profile for profile in profiles)
        or tuple(profiles) != tuple(sorted(set(profiles)))
    ):
        _fail("EIEL_ADMISSION_PROFILES", path)
    for field in (
        "state_schema_version",
        "checkpoint_schema_version",
        "chronicle_schema_version",
        "provider_capability_identity",
        "overlay_version",
        "build_toolchain_identity",
        "compile_flags",
        "artifact_build_id",
    ):
        if not isinstance(result[field], str) or not result[field]:
            _fail("EIEL_ADMISSION_IDENTITY_VALUE", f"{path}.{field}")
    return result


def validate_admission_identity_v1(
    *,
    request: Mapping[str, Any],
    provider: Mapping[str, Any],
    manifest: Mapping[str, Any],
    binary: Mapping[str, Any],
    checkpoint: Mapping[str, Any] | None = None,
) -> str:
    surfaces = {
        "request": request,
        "provider": provider,
        "manifest": manifest,
        "binary": binary,
    }
    if checkpoint is not None:
        surfaces["checkpoint"] = checkpoint
    validated = {
        name: validate_production_identity_v1(value, path=name)
        for name, value in surfaces.items()
    }
    identities = {
        name: canonical_json_bytes(value) for name, value in validated.items()
    }
    if len(set(identities.values())) != 1:
        first = next(iter(validated))
        for name in tuple(validated)[1:]:
            if identities[name] != identities[first]:
                keys = sorted(
                    key
                    for key in _FIELDS
                    if canonical_json_bytes(validated[name][key])
                    != canonical_json_bytes(validated[first][key])
                )
                _fail(
                    "EIEL_ADMISSION_IDENTITY_MISMATCH",
                    f"{first} != {name}: {keys}",
                )
    return canonical_sha256(validated["request"])


def validate_identity_against_local_schema_v1(
    value: Mapping[str, Any],
    *,
    schema: AuthoritativeStateSchemaV1,
    provider_id: str,
    profile_id: str,
) -> dict[str, Any]:
    identity = validate_production_identity_v1(value, path="artifact_identity")
    supported_surfaces = schema.root.get("supported_surfaces")
    if not isinstance(supported_surfaces, Sequence) or isinstance(
        supported_surfaces, (str, bytes)
    ):
        _fail(
            "EIEL_ADMISSION_SUPPORTED_SURFACES_INVALID",
            "canonical supported_surfaces must be an ordered array",
        )
    supported_pairs = {
        (row.get("provider_id"), row.get("profile_id"))
        for row in supported_surfaces
        if isinstance(row, Mapping)
    }
    if (provider_id, profile_id) not in supported_pairs:
        _fail(
            "EIEL_ADMISSION_SURFACE_UNSUPPORTED",
            f"provider={provider_id} profile={profile_id}",
        )
    expected_components = schema.root["component_hashes"]
    expected_scalars = {
        "state_schema_version": schema.root["schema_version"],
        "state_schema_sha256": schema.schema_hash,
        "source_commit": schema.root["source_commit"],
        "source_inventory_sha256": expected_components["field_registry.json"],
        "checkpoint_schema_sha256": expected_components["checkpoint_schema.json"],
        "chronicle_schema_sha256": expected_components["chronicle_schema.json"],
        "provider_capability_identity": provider_id,
    }
    for field, expected in expected_scalars.items():
        if identity[field] != expected:
            _fail(
                "EIEL_ADMISSION_LOCAL_SCHEMA_MISMATCH",
                f"{field}: expected {expected}, got {identity[field]}",
            )
    if profile_id not in identity["supported_profiles"]:
        _fail("EIEL_ADMISSION_PROFILE_UNSUPPORTED", profile_id)
    expected_variants = [
        {
            "entrypoint": row["variant_id"],
            "abi_version": row["variant_id"],
            "argument_count": row["count"],
            "argument_order_sha256": row["order_sha256"],
        }
        for row in schema.abi_variants["variants"]
        if row["count"] > 0
    ]
    expected_variants.sort(key=lambda row: (row["entrypoint"], row["abi_version"]))
    if identity["abi_variants"] != expected_variants:
        _fail(
            "EIEL_ADMISSION_LOCAL_ABI_MISMATCH",
            "compiled ABI inventory differs from canonical nonempty variants",
        )
    if identity["abi_set_sha256"] != canonical_sha256(expected_variants):
        _fail("EIEL_ADMISSION_LOCAL_ABI_HASH", "compiled ABI set hash differs")
    return identity


__all__ = [
    "validate_admission_identity_v1",
    "validate_production_identity_v1",
    "validate_identity_against_local_schema_v1",
]
