"""Strict field manifest for direct resident physical-batch ACS capture.

The manifest is deliberately a classification contract, not a capture
implementation.  Its field order is inherited from the existing resident
source allocator and then extended by the authoritative event, WDL, and EED
field constants.  The source groups overlap by design; the first occurrence
is authoritative and the resulting surface is fixed at 191 fields.

No tensor is inspected or allocated here.  In particular, a zero-fill entry
means that a later producer may omit that field only under the recorded
contract; it does not create a replacement tensor or alter simulation state.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

from emergence_contracts.direct_resident_field_surface import (
    BASE_BATCHED_FIELDS,
    ORDINARY_BIRTH_DRIVE_FIELDS,
    ORDINARY_BIRTH_GENOME_FLOAT32_FIELDS,
    ORDINARY_BIRTH_GENOME_INT32_FIELDS,
    ORDINARY_BIRTH_IDENTITY_FIELDS,
    ORDINARY_BIRTH_POLICY_FIELDS,
    ORDINARY_BIRTH_SCALAR_RNG_FIELDS,
    PATCH_RESIDENT_FIELDS,
    TURNOVER_STATE_FIELDS,
)
from engine.native_runtime.checkpoint_digests import DEFAULT_FIELD_POLICIES
from engine.perf.cuda_architecture_b_boundary import (
    EVENT_COLUMN_NAMES,
    POLICY_GENERATION_ALIASES,
)
from engine.perf.cuda_architecture_b_direct_resident_eed_v1 import (
    EED_STATE_FIELDS_V1,
)
from engine.perf.cuda_architecture_b_direct_resident_wdl_v2 import (
    WDL_STATE_FIELDS_V2,
)


SCHEMA_VERSION: Final[str] = "eiel_acs_direct_resident_field_manifest_v1"
EXPECTED_FIELD_COUNT_V1: Final[int] = 191

BATCHED_TENSOR_ROW: Final[str] = "batched_tensor_row"
EVENT_COLUMN_ROW: Final[str] = "event_column_row"
WDL_OVERLAY_ROW: Final[str] = "wdl_overlay_row"
EED_OVERLAY_ROW: Final[str] = "eed_overlay_row"
HOST_AUTHORITATIVE_ROW: Final[str] = "host_authoritative_row"
ZERO_FILL: Final[str] = "zero_fill"
DERIVED_BATCHED: Final[str] = "derived_batched"

PROVIDER_KINDS_V1: Final[frozenset[str]] = frozenset(
    {
        BATCHED_TENSOR_ROW,
        EVENT_COLUMN_ROW,
        WDL_OVERLAY_ROW,
        EED_OVERLAY_ROW,
        HOST_AUTHORITATIVE_ROW,
        ZERO_FILL,
        DERIVED_BATCHED,
    }
)

FIXED_MLP_CONTRACT_V1: Final[str] = "fixed_mlp_contract_v1"
WORLD_RECONSTRUCTION_AUTHORITY_V1: Final[str] = (
    "existing_world_authority_reconstructible_v1"
)
EVENT_INDEXING_AUTHORITY_V1: Final[str] = "active_event_prefix_index_v1"
OPAQUE_CPU_RNG_AUTHORITY_V1: Final[str] = "opaque_cpu_rng_host_source_v1"
POLICY_ALIAS_AUTHORITY_V1: Final[str] = "policy_alias_byte_equivalent_v1"
DERIVED_BATCHED_AUTHORITY_V1: Final[str] = "existing_derived_batched_authority_v1"

_CANONICAL_SOURCE_GROUPS_V1: Final[tuple[tuple[str, tuple[str, ...]], ...]] = (
    ("base_batched", tuple(BASE_BATCHED_FIELDS)),
    ("ordinary_birth_identity", tuple(ORDINARY_BIRTH_IDENTITY_FIELDS)),
    ("ordinary_birth_scalar_rng", tuple(ORDINARY_BIRTH_SCALAR_RNG_FIELDS)),
    ("ordinary_birth_genome_int32", tuple(ORDINARY_BIRTH_GENOME_INT32_FIELDS)),
    ("ordinary_birth_genome_float32", tuple(ORDINARY_BIRTH_GENOME_FLOAT32_FIELDS)),
    ("ordinary_birth_drive", tuple(ORDINARY_BIRTH_DRIVE_FIELDS)),
    ("ordinary_birth_policy", tuple(ORDINARY_BIRTH_POLICY_FIELDS)),
    ("turnover_state", tuple(TURNOVER_STATE_FIELDS)),
    ("patch_resident", tuple(PATCH_RESIDENT_FIELDS)),
    ("event_columns", tuple(EVENT_COLUMN_NAMES)),
    ("wdl_state", tuple(WDL_STATE_FIELDS_V2)),
    ("eed_state", tuple(EED_STATE_FIELDS_V1)),
)

_WORLD_RECONSTRUCTION_FIELDS_V1: Final[frozenset[str]] = frozenset(
    {"biome_grid", "hazard_rate_map"}
)
_FIXED_MLP_ZERO_FIELDS_V1: Final[frozenset[str]] = frozenset(
    {"gru_pack", "hmem"}
)
_OPAQUE_CPU_RNG_FIELDS_V1: Final[frozenset[str]] = frozenset(
    {"rng_manager_mutation_torch_cpu_state"}
)
_DERIVED_BATCHED_FIELDS_V1: Final[frozenset[str]] = frozenset(
    {
        "action_cache_next",
        "patch_density",
    }
)

# This digest intentionally covers only the source-derived surface.  It is a
# cheap import-time sentinel for accidental source-schema drift; the manifest
# digest below additionally covers provider kinds and contract metadata.
EXPECTED_CANONICAL_FIELD_SURFACE_DIGEST_V1: Final[str] = (
    "bdca0a7b7513be33191f355c135835bd5204d5568e0c518413a6d5926536ddc6"
)
EXPECTED_CANONICAL_MANIFEST_DIGEST_V1: Final[str] = (
    "162e4be63c5e8f99284d2491eb33d3f0015d8861106131ef5e1c963315fa11bb"
)


class DirectResidentFieldManifestError(ValueError):
    """Raised when the direct-resident field contract is contradictory."""


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise DirectResidentFieldManifestError(
            "manifest canonical payload is not deterministic JSON"
        ) from exc


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _deduplicate_source_groups(
    groups: Sequence[tuple[str, Sequence[str]]],
) -> tuple[str, ...]:
    seen: set[str] = set()
    ordered: list[str] = []
    for group_name, fields in groups:
        if not isinstance(group_name, str) or not group_name:
            raise DirectResidentFieldManifestError(
                "authoritative source group names must be nonempty strings"
            )
        local_seen: set[str] = set()
        for name in fields:
            if not isinstance(name, str) or not name:
                raise DirectResidentFieldManifestError(
                    f"authoritative source group {group_name!r} contains an invalid field"
                )
            if name in local_seen:
                raise DirectResidentFieldManifestError(
                    f"authoritative source group {group_name!r} contains duplicate field {name!r}"
                )
            local_seen.add(name)
            if name not in seen:
                seen.add(name)
                ordered.append(name)
    return tuple(ordered)


def _surface_payload(field_names: Sequence[str]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "field_count": len(field_names),
        "field_names": list(field_names),
    }


CANONICAL_FIELD_NAMES_V1: Final[tuple[str, ...]] = _deduplicate_source_groups(
    _CANONICAL_SOURCE_GROUPS_V1
)
CANONICAL_FIELD_COUNT_V1: Final[int] = len(CANONICAL_FIELD_NAMES_V1)
CANONICAL_FIELD_SURFACE_DIGEST_V1: Final[str] = _sha256(
    _surface_payload(CANONICAL_FIELD_NAMES_V1)
)
if CANONICAL_FIELD_COUNT_V1 != EXPECTED_FIELD_COUNT_V1:
    raise DirectResidentFieldManifestError(
        "authoritative prepared-owner field count changed: "
        f"expected {EXPECTED_FIELD_COUNT_V1}, got {CANONICAL_FIELD_COUNT_V1}"
    )
if CANONICAL_FIELD_SURFACE_DIGEST_V1 != EXPECTED_CANONICAL_FIELD_SURFACE_DIGEST_V1:
    raise DirectResidentFieldManifestError(
        "authoritative prepared-owner field surface digest changed: "
        f"expected {EXPECTED_CANONICAL_FIELD_SURFACE_DIGEST_V1}, "
        f"got {CANONICAL_FIELD_SURFACE_DIGEST_V1}"
    )

_CANONICAL_FIELD_SET_V1: Final[frozenset[str]] = frozenset(
    CANONICAL_FIELD_NAMES_V1
)
_CANONICAL_ORDINAL_V1: Final[dict[str, int]] = {
    name: ordinal for ordinal, name in enumerate(CANONICAL_FIELD_NAMES_V1)
}
_SOURCE_GROUP_BY_FIELD_V1: Final[dict[str, str]] = {}
for _group_name, _group_fields in _CANONICAL_SOURCE_GROUPS_V1:
    for _field_name in _group_fields:
        _SOURCE_GROUP_BY_FIELD_V1.setdefault(_field_name, _group_name)


def _validate_exact_bool(value: Any, *, path: str) -> bool:
    if type(value) is not bool:
        raise DirectResidentFieldManifestError(f"{path} must be an exact boolean")
    return value


def _validate_field_names(field_names: Sequence[str]) -> tuple[str, ...]:
    if isinstance(field_names, (str, bytes, bytearray)):
        raise DirectResidentFieldManifestError("field_names must be a sequence")
    try:
        supplied = tuple(field_names)
    except TypeError as exc:
        raise DirectResidentFieldManifestError("field_names must be a sequence") from exc
    if any(not isinstance(name, str) or not name for name in supplied):
        raise DirectResidentFieldManifestError(
            "field_names must contain only nonempty strings"
        )
    duplicates = sorted(
        name for name in set(supplied) if supplied.count(name) > 1
    )
    if duplicates:
        raise DirectResidentFieldManifestError(
            f"field_names contain duplicate fields: {duplicates}"
        )
    supplied_set = frozenset(supplied)
    unsupported = sorted(supplied_set - _CANONICAL_FIELD_SET_V1)
    missing = sorted(_CANONICAL_FIELD_SET_V1 - supplied_set)
    if unsupported:
        raise DirectResidentFieldManifestError(
            f"field_names contain unsupported fields: {unsupported}"
        )
    if missing:
        raise DirectResidentFieldManifestError(
            f"field_names are missing authoritative fields: {missing}"
        )
    # Callers may supply an equivalent set in a different order; the manifest
    # always returns the source-authoritative first-occurrence order.
    return CANONICAL_FIELD_NAMES_V1


def validate_direct_resident_field_names_v1(
    field_names: Sequence[str],
) -> tuple[str, ...]:
    """Validate and canonicalize an exact 191-name prepared-owner surface."""

    return _validate_field_names(field_names)


def _world_reconstruction_is_authorized(
    field_name: str,
    authority: Mapping[str, str] | None,
) -> bool:
    if authority is None or field_name not in authority:
        return False
    marker = authority[field_name]
    if marker != WORLD_RECONSTRUCTION_AUTHORITY_V1:
        raise DirectResidentFieldManifestError(
            f"{field_name} has an unsupported reconstruction authority marker"
        )
    policy = DEFAULT_FIELD_POLICIES.get(field_name)
    if policy is None or policy.reconstructible is not True:
        raise DirectResidentFieldManifestError(
            f"existing authority does not permit reconstructing {field_name}"
        )
    return True


def _validate_world_authority(
    authority: Mapping[str, str] | None,
) -> dict[str, str]:
    if authority is None:
        return {}
    if not isinstance(authority, Mapping):
        raise DirectResidentFieldManifestError(
            "world_reconstruction_authority must be a mapping"
        )
    normalized = dict(authority)
    unsupported = sorted(set(normalized) - _WORLD_RECONSTRUCTION_FIELDS_V1)
    if unsupported:
        raise DirectResidentFieldManifestError(
            f"world reconstruction authority has unsupported fields: {unsupported}"
        )
    for name, marker in normalized.items():
        if not isinstance(name, str) or not name:
            raise DirectResidentFieldManifestError(
                "world reconstruction authority keys must be nonempty strings"
            )
        if marker != WORLD_RECONSTRUCTION_AUTHORITY_V1:
            raise DirectResidentFieldManifestError(
                f"world reconstruction authority for {name!r} is not recognized"
            )
    return dict(sorted(normalized.items()))


@dataclass(frozen=True, slots=True)
class DirectResidentFieldV1:
    """One source-derived field classification."""

    ordinal: int
    name: str
    source_group: str
    provider_kind: str
    alias_of: str | None = None
    indexing: str | None = None
    authority: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ordinal": self.ordinal,
            "name": self.name,
            "source_group": self.source_group,
            "provider_kind": self.provider_kind,
            "alias_of": self.alias_of,
            "indexing": self.indexing,
            "authority": self.authority,
        }


@dataclass(frozen=True, slots=True)
class DirectResidentFieldManifestV1:
    """Immutable, hash-bound classification of the prepared-owner surface."""

    schema_version: str
    field_count: int
    fixed_mlp_contract: bool
    world_reconstruction_authority: tuple[tuple[str, str], ...]
    fields: tuple[DirectResidentFieldV1, ...]
    canonical_digest: str

    def _payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "field_count": self.field_count,
            "contract": {
                "fixed_mlp_contract": self.fixed_mlp_contract,
                "world_reconstruction_authority": dict(
                    self.world_reconstruction_authority
                ),
            },
            "fields": [field.to_dict() for field in self.fields],
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self._payload()
        payload["canonical_digest"] = self.canonical_digest
        return payload


def _classify_field(
    name: str,
    *,
    fixed_mlp_contract: bool,
    world_reconstruction_authority: Mapping[str, str],
) -> DirectResidentFieldV1:
    source_group = _SOURCE_GROUP_BY_FIELD_V1[name]
    alias_of = POLICY_GENERATION_ALIASES.get(name)
    if name in EVENT_COLUMN_NAMES:
        return DirectResidentFieldV1(
            ordinal=_CANONICAL_ORDINAL_V1[name],
            name=name,
            source_group=source_group,
            provider_kind=EVENT_COLUMN_ROW,
            indexing=EVENT_INDEXING_AUTHORITY_V1,
            authority=EVENT_INDEXING_AUTHORITY_V1,
        )
    if name in WDL_STATE_FIELDS_V2:
        return DirectResidentFieldV1(
            ordinal=_CANONICAL_ORDINAL_V1[name],
            name=name,
            source_group=source_group,
            provider_kind=WDL_OVERLAY_ROW,
            authority="wdl_overlay_state_v2",
        )
    if name in EED_STATE_FIELDS_V1:
        return DirectResidentFieldV1(
            ordinal=_CANONICAL_ORDINAL_V1[name],
            name=name,
            source_group=source_group,
            provider_kind=EED_OVERLAY_ROW,
            authority="eed_overlay_state_v1",
        )
    if name in _OPAQUE_CPU_RNG_FIELDS_V1:
        return DirectResidentFieldV1(
            ordinal=_CANONICAL_ORDINAL_V1[name],
            name=name,
            source_group=source_group,
            provider_kind=HOST_AUTHORITATIVE_ROW,
            authority=OPAQUE_CPU_RNG_AUTHORITY_V1,
        )
    if alias_of is not None:
        if alias_of not in _CANONICAL_FIELD_SET_V1:
            raise DirectResidentFieldManifestError(
                f"policy alias {name!r} targets unsupported field {alias_of!r}"
            )
        return DirectResidentFieldV1(
            ordinal=_CANONICAL_ORDINAL_V1[name],
            name=name,
            source_group=source_group,
            provider_kind=DERIVED_BATCHED,
            alias_of=alias_of,
            authority=POLICY_ALIAS_AUTHORITY_V1,
        )
    if name in _FIXED_MLP_ZERO_FIELDS_V1 and fixed_mlp_contract:
        return DirectResidentFieldV1(
            ordinal=_CANONICAL_ORDINAL_V1[name],
            name=name,
            source_group=source_group,
            provider_kind=ZERO_FILL,
            authority=FIXED_MLP_CONTRACT_V1,
        )
    if name in _WORLD_RECONSTRUCTION_FIELDS_V1 and _world_reconstruction_is_authorized(
        name, world_reconstruction_authority
    ):
        return DirectResidentFieldV1(
            ordinal=_CANONICAL_ORDINAL_V1[name],
            name=name,
            source_group=source_group,
            provider_kind=DERIVED_BATCHED,
            authority=WORLD_RECONSTRUCTION_AUTHORITY_V1,
        )
    if name in _DERIVED_BATCHED_FIELDS_V1:
        return DirectResidentFieldV1(
            ordinal=_CANONICAL_ORDINAL_V1[name],
            name=name,
            source_group=source_group,
            provider_kind=DERIVED_BATCHED,
            authority=DERIVED_BATCHED_AUTHORITY_V1,
        )
    return DirectResidentFieldV1(
        ordinal=_CANONICAL_ORDINAL_V1[name],
        name=name,
        source_group=source_group,
        provider_kind=BATCHED_TENSOR_ROW,
    )


def build_direct_resident_field_manifest_v1(
    field_names: Sequence[str] | None = None,
    *,
    fixed_mlp_contract: bool = False,
    world_reconstruction_authority: Mapping[str, str] | None = None,
    schema_version: str = SCHEMA_VERSION,
) -> DirectResidentFieldManifestV1:
    """Build the exact field manifest under explicit route contracts."""

    if schema_version != SCHEMA_VERSION:
        raise DirectResidentFieldManifestError(
            f"schema version mismatch: expected {SCHEMA_VERSION!r}, got {schema_version!r}"
        )
    fixed_mlp = _validate_exact_bool(
        fixed_mlp_contract, path="fixed_mlp_contract"
    )
    canonical_names = (
        CANONICAL_FIELD_NAMES_V1
        if field_names is None
        else _validate_field_names(field_names)
    )
    authority = _validate_world_authority(world_reconstruction_authority)
    fields = tuple(
        _classify_field(
            name,
            fixed_mlp_contract=fixed_mlp,
            world_reconstruction_authority=authority,
        )
        for name in canonical_names
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "field_count": len(fields),
        "contract": {
            "fixed_mlp_contract": fixed_mlp,
            "world_reconstruction_authority": authority,
        },
        "fields": [field.to_dict() for field in fields],
    }
    manifest = DirectResidentFieldManifestV1(
        schema_version=SCHEMA_VERSION,
        field_count=len(fields),
        fixed_mlp_contract=fixed_mlp,
        world_reconstruction_authority=tuple(authority.items()),
        fields=fields,
        canonical_digest=_sha256(payload),
    )
    validate_direct_resident_field_manifest_v1(manifest, _check_digest=False)
    return manifest


def _payload_to_manifest(value: Mapping[str, Any]) -> DirectResidentFieldManifestV1:
    expected_keys = {
        "schema_version",
        "field_count",
        "contract",
        "fields",
        "canonical_digest",
    }
    if set(value) != expected_keys:
        raise DirectResidentFieldManifestError(
            "manifest has unsupported or missing top-level fields: "
            f"expected {sorted(expected_keys)}, got {sorted(value)}"
        )
    schema_version = value["schema_version"]
    if schema_version != SCHEMA_VERSION:
        raise DirectResidentFieldManifestError("manifest schema version mismatch")
    field_count = value["field_count"]
    if type(field_count) is not int:
        raise DirectResidentFieldManifestError("manifest field_count must be an exact integer")
    contract = value["contract"]
    if not isinstance(contract, Mapping):
        raise DirectResidentFieldManifestError("manifest contract must be a mapping")
    if set(contract) != {"fixed_mlp_contract", "world_reconstruction_authority"}:
        raise DirectResidentFieldManifestError("manifest contract fields are incomplete or unsupported")
    fixed_mlp = _validate_exact_bool(
        contract["fixed_mlp_contract"], path="manifest.contract.fixed_mlp_contract"
    )
    authority = _validate_world_authority(
        contract["world_reconstruction_authority"]
    )
    raw_fields = value["fields"]
    if not isinstance(raw_fields, (list, tuple)):
        raise DirectResidentFieldManifestError("manifest fields must be a sequence")
    fields: list[DirectResidentFieldV1] = []
    field_keys = {
        "ordinal",
        "name",
        "source_group",
        "provider_kind",
        "alias_of",
        "indexing",
        "authority",
    }
    for index, raw in enumerate(raw_fields):
        if not isinstance(raw, Mapping) or set(raw) != field_keys:
            raise DirectResidentFieldManifestError(
                f"manifest field {index} has unsupported or missing members"
            )
        fields.append(
            DirectResidentFieldV1(
                ordinal=raw["ordinal"],
                name=raw["name"],
                source_group=raw["source_group"],
                provider_kind=raw["provider_kind"],
                alias_of=raw["alias_of"],
                indexing=raw["indexing"],
                authority=raw["authority"],
            )
        )
    digest = value["canonical_digest"]
    if not isinstance(digest, str):
        raise DirectResidentFieldManifestError("manifest canonical_digest must be a string")
    return DirectResidentFieldManifestV1(
        schema_version=schema_version,
        field_count=field_count,
        fixed_mlp_contract=fixed_mlp,
        world_reconstruction_authority=tuple(authority.items()),
        fields=tuple(fields),
        canonical_digest=digest,
    )


def validate_direct_resident_field_manifest_v1(
    manifest: DirectResidentFieldManifestV1 | Mapping[str, Any],
    *,
    _check_digest: bool = True,
) -> DirectResidentFieldManifestV1:
    """Fail closed on schema, source surface, classification, or digest drift."""

    if isinstance(manifest, DirectResidentFieldManifestV1):
        candidate = manifest
    elif isinstance(manifest, Mapping):
        candidate = _payload_to_manifest(manifest)
    else:
        raise DirectResidentFieldManifestError("manifest must be a manifest object or mapping")

    if candidate.schema_version != SCHEMA_VERSION:
        raise DirectResidentFieldManifestError("manifest schema version mismatch")
    if candidate.field_count != EXPECTED_FIELD_COUNT_V1:
        raise DirectResidentFieldManifestError(
            f"manifest field_count must be {EXPECTED_FIELD_COUNT_V1}"
        )
    names = tuple(field.name for field in candidate.fields)
    _validate_field_names(names)
    if names != CANONICAL_FIELD_NAMES_V1:
        raise DirectResidentFieldManifestError(
            "manifest fields are not in authoritative canonical order"
        )
    if tuple(field.ordinal for field in candidate.fields) != tuple(
        range(EXPECTED_FIELD_COUNT_V1)
    ):
        raise DirectResidentFieldManifestError("manifest field ordinals are not canonical")
    if any(field.provider_kind not in PROVIDER_KINDS_V1 for field in candidate.fields):
        raise DirectResidentFieldManifestError("manifest contains an unsupported provider kind")

    if _check_digest:
        expected = build_direct_resident_field_manifest_v1(
            field_names=names,
            fixed_mlp_contract=candidate.fixed_mlp_contract,
            world_reconstruction_authority=dict(candidate.world_reconstruction_authority),
        )
        if [field.to_dict() for field in candidate.fields] != [
            field.to_dict() for field in expected.fields
        ]:
            raise DirectResidentFieldManifestError(
                "manifest field classification or authority mapping is not canonical"
            )
        expected_payload = candidate._payload()
        expected_digest = _sha256(expected_payload)
        if candidate.canonical_digest != expected_digest:
            raise DirectResidentFieldManifestError(
                "manifest canonical digest mismatch"
            )
    return candidate


DIRECT_RESIDENT_FIELD_MANIFEST_V1: Final[DirectResidentFieldManifestV1] = (
    build_direct_resident_field_manifest_v1(fixed_mlp_contract=True)
)
CANONICAL_MANIFEST_DIGEST_V1: Final[str] = (
    DIRECT_RESIDENT_FIELD_MANIFEST_V1.canonical_digest
)
if CANONICAL_MANIFEST_DIGEST_V1 != EXPECTED_CANONICAL_MANIFEST_DIGEST_V1:
    raise DirectResidentFieldManifestError(
        "canonical direct-resident field manifest digest changed: "
        f"expected {EXPECTED_CANONICAL_MANIFEST_DIGEST_V1}, "
        f"got {CANONICAL_MANIFEST_DIGEST_V1}"
    )
CANONICAL_DIGEST_V1: Final[str] = CANONICAL_MANIFEST_DIGEST_V1


__all__ = [
    "BATCHED_TENSOR_ROW",
    "CANONICAL_DIGEST_V1",
    "CANONICAL_FIELD_COUNT_V1",
    "CANONICAL_FIELD_NAMES_V1",
    "CANONICAL_FIELD_SURFACE_DIGEST_V1",
    "CANONICAL_MANIFEST_DIGEST_V1",
    "DERIVED_BATCHED",
    "DirectResidentFieldManifestError",
    "DirectResidentFieldManifestV1",
    "DirectResidentFieldV1",
    "DIRECT_RESIDENT_FIELD_MANIFEST_V1",
    "EED_OVERLAY_ROW",
    "EVENT_COLUMN_ROW",
    "EXPECTED_CANONICAL_FIELD_SURFACE_DIGEST_V1",
    "EXPECTED_CANONICAL_MANIFEST_DIGEST_V1",
    "EXPECTED_FIELD_COUNT_V1",
    "FIXED_MLP_CONTRACT_V1",
    "HOST_AUTHORITATIVE_ROW",
    "OPAQUE_CPU_RNG_AUTHORITY_V1",
    "POLICY_ALIAS_AUTHORITY_V1",
    "PROVIDER_KINDS_V1",
    "SCHEMA_VERSION",
    "WDL_OVERLAY_ROW",
    "WORLD_RECONSTRUCTION_AUTHORITY_V1",
    "ZERO_FILL",
    "build_direct_resident_field_manifest_v1",
    "validate_direct_resident_field_manifest_v1",
    "validate_direct_resident_field_names_v1",
]
