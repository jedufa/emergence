from __future__ import annotations

"""Deterministic biological identity for bounded modular structure.

``module_id`` remains a compact, genome-local storage slot.  It is deliberately
not an ancestry key.  Native module instances and structural loci use separate
hash-derived identifiers that consume no RNG draws and survive canonical
packing, checkpointing, and replay.
"""

from copy import deepcopy
import hashlib
import json
import math
from typing import Any, Mapping


STRUCTURAL_IDENTITY_SCHEMA_VERSION = "structural_identity_v1"
STRUCTURAL_ANCESTRY_SCHEMA_VERSION = "structural_ancestry_v1"
TOPOLOGY_POSITION_SCHEMA_VERSION = "topology_position_v1"
STRUCTURAL_LEGACY_MIGRATION_VERSION = "structural_identity_legacy_migration_v1"

IDENTITY_ORIGIN_NATIVE = "native"
IDENTITY_ORIGIN_LEGACY = "legacy_derived_unresolved"


class StructuralIdentityError(RuntimeError):
    pass


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _hash_id(prefix: str, material: Mapping[str, Any]) -> str:
    return f"{prefix}_{hashlib.sha256(_canonical_bytes(dict(material))).hexdigest()}"


def _finite_unit(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        out = 0.0
    if not math.isfinite(out):
        out = 0.0
    return float(round(min(1.0, max(0.0, out)), 8))


def canonical_topology_position(value: Any) -> list[float]:
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return [_finite_unit(value[0]), _finite_unit(value[1])]
    if isinstance(value, Mapping):
        return [_finite_unit(value.get("x", 0.0)), _finite_unit(value.get("y", 0.0))]
    return [0.0, 0.0]


def topology_position_from_material(material: Mapping[str, Any]) -> list[float]:
    digest = hashlib.sha256(_canonical_bytes(dict(material))).digest()
    den = float(2**64 - 1)
    return canonical_topology_position(
        [
            int.from_bytes(digest[0:8], "big", signed=False) / den,
            int.from_bytes(digest[8:16], "big", signed=False) / den,
        ]
    )


def nearby_topology_position(parent_position: Any, material: Mapping[str, Any]) -> list[float]:
    """Derive a stable nearby position, keeping duplication safely local."""

    parent = canonical_topology_position(parent_position)
    digest = hashlib.sha256(_canonical_bytes(dict(material))).digest()
    dx = ((int(digest[0]) / 255.0) - 0.5) * 0.10
    dy = ((int(digest[1]) / 255.0) - 0.5) * 0.10
    # Reflect at the unit-square boundary rather than wrapping across it.
    return canonical_topology_position([parent[0] + dx, parent[1] + dy])


def canonical_identity_context(context: Mapping[str, Any] | None) -> dict[str, Any]:
    raw = dict(context or {})
    parent_ids = raw.get("parent_agent_ids", [])
    if not isinstance(parent_ids, (list, tuple)):
        parent_ids = []
    child_raw = raw.get("child_agent_id", -1)
    child_agent_id = -1 if child_raw is None or child_raw == "" else int(child_raw)
    return {
        "campaign_id": str(raw.get("campaign_id", "") or ""),
        "branch_id": str(raw.get("branch_id", "default") or "default"),
        "generation": int(raw.get("generation", 0) or 0),
        "tick": int(raw.get("tick", 0) or 0),
        "child_agent_id": child_agent_id,
        "parent_agent_ids": sorted(int(v) for v in parent_ids),
        "lineage_token": str(raw.get("lineage_token", "") or ""),
    }


def validate_native_creation_context(context: Mapping[str, Any] | None) -> dict[str, Any]:
    out = canonical_identity_context(context)
    if out["child_agent_id"] < 0 and not out["lineage_token"]:
        raise StructuralIdentityError(
            "native structural creation requires child_agent_id or an explicit lineage_token"
        )
    return out


def mutation_event_identity(
    context: Mapping[str, Any] | None,
    *,
    operator_id: str,
    attempt_index: int,
) -> str:
    ctx = validate_native_creation_context(context)
    return _hash_id(
        "sme",
        {
            "schema_version": STRUCTURAL_ANCESTRY_SCHEMA_VERSION,
            "context": ctx,
            "operator_id": str(operator_id),
            "attempt_index": int(attempt_index),
        },
    )


def _canonical_ancestry(value: Any) -> dict[str, Any]:
    raw = dict(value) if isinstance(value, Mapping) else {}
    parent_ids = raw.get("parent_module_instance_ids", [])
    if not isinstance(parent_ids, (list, tuple)):
        parent_ids = []
    return {
        "schema_version": STRUCTURAL_ANCESTRY_SCHEMA_VERSION,
        "parent_module_instance_ids": sorted(
            {str(v) for v in parent_ids if str(v).strip()}
        ),
        "origin_operator": str(raw.get("origin_operator", "legacy_migration") or "legacy_migration"),
        "origin_generation": int(raw.get("origin_generation", 0) or 0),
        "origin_mutation_event_id": str(raw.get("origin_mutation_event_id", "") or ""),
        "inherited_homology_id": str(raw.get("inherited_homology_id", "") or ""),
        "provenance_id": str(raw.get("provenance_id", "") or ""),
    }


def _legacy_material(record: Mapping[str, Any], *, record_index: int) -> dict[str, Any]:
    return {
        "migration_version": STRUCTURAL_LEGACY_MIGRATION_VERSION,
        "module_id": str(record.get("module_id", record.get("id", "")) or ""),
        "module_type": str(record.get("module_type", record.get("kind", "generic")) or "generic"),
        "params": record.get("params", {}) if isinstance(record.get("params", {}), Mapping) else {},
        "io": record.get("io", {}) if isinstance(record.get("io", {}), Mapping) else {},
    }


def canonicalize_module_identity(
    record: Mapping[str, Any],
    *,
    record_index: int,
    strict_partial_native: bool = False,
) -> dict[str, Any]:
    """Return identity fields, conservatively migrating legacy records.

    Legacy-derived homology identifiers are diagnostic only.  C2 must never
    align them across parents because common ancestry cannot be established.
    """

    raw = dict(record)
    identity_keys = {
        "module_instance_id",
        "homology_id",
        "identity_origin",
        "identity_schema_version",
    }
    any_identity = any(key in raw for key in identity_keys)
    instance_id = str(raw.get("module_instance_id", "") or "")
    homology_id = str(raw.get("homology_id", "") or "")
    origin = str(raw.get("identity_origin", "") or "")
    schema = str(raw.get("identity_schema_version", "") or "")
    native_complete = bool(
        instance_id
        and homology_id
        and origin == IDENTITY_ORIGIN_NATIVE
        and schema == STRUCTURAL_IDENTITY_SCHEMA_VERSION
    )
    if any_identity and not native_complete and strict_partial_native:
        # Complete legacy migrations are valid inputs; partial claims of native
        # identity are ambiguous and fail closed.
        legacy_complete = bool(
            instance_id
            and homology_id
            and origin == IDENTITY_ORIGIN_LEGACY
            and schema == STRUCTURAL_IDENTITY_SCHEMA_VERSION
        )
        if not legacy_complete:
            raise StructuralIdentityError("partial or unsupported structural identity metadata")

    if native_complete:
        ancestry = _canonical_ancestry(raw.get("ancestry", {}))
        pos_source = raw.get("topology_pos", raw.get("latent_pos", None))
        position = (
            canonical_topology_position(pos_source)
            if pos_source is not None
            else topology_position_from_material(
                {"schema_version": TOPOLOGY_POSITION_SCHEMA_VERSION, "homology_id": homology_id}
            )
        )
        return {
            "identity_schema_version": STRUCTURAL_IDENTITY_SCHEMA_VERSION,
            "module_instance_id": instance_id,
            "homology_id": homology_id,
            "identity_origin": IDENTITY_ORIGIN_NATIVE,
            "ancestry": ancestry,
            "topology_position_schema_version": TOPOLOGY_POSITION_SCHEMA_VERSION,
            "topology_pos": position,
            "legacy_identity_source": None,
        }

    legacy = _legacy_material(raw, record_index=record_index)
    legacy_digest = hashlib.sha256(_canonical_bytes(legacy)).hexdigest()
    # Old ``latent_pos`` values were derived cache fields rather than owned
    # topology.  Recompute the historical rule once, then freeze the result in
    # explicit ``topology_pos``.  A genuinely explicit new topology field wins.
    position_source = raw.get("topology_pos", None)
    position = (
        canonical_topology_position(position_source)
        if position_source is not None
        else topology_position_from_material(
            {
                "module_id": legacy["module_id"],
                "module_type": legacy["module_type"],
                "params": legacy["params"],
            }
        )
    )
    return {
        "identity_schema_version": STRUCTURAL_IDENTITY_SCHEMA_VERSION,
        "module_instance_id": f"mi_legacy_{legacy_digest}",
        "homology_id": f"mh_legacy_{legacy_digest}",
        "identity_origin": IDENTITY_ORIGIN_LEGACY,
        "ancestry": {
            "schema_version": STRUCTURAL_ANCESTRY_SCHEMA_VERSION,
            "parent_module_instance_ids": [],
            "origin_operator": "legacy_migration",
            "origin_generation": 0,
            "origin_mutation_event_id": "",
            "inherited_homology_id": "",
            "provenance_id": f"legacy_{legacy_digest}",
        },
        "topology_position_schema_version": TOPOLOGY_POSITION_SCHEMA_VERSION,
        "topology_pos": position,
        "legacy_identity_source": {
            "migration_version": STRUCTURAL_LEGACY_MIGRATION_VERSION,
            "source_module_id": str(raw.get("module_id", raw.get("id", "")) or ""),
            "source_digest": legacy_digest,
            "homology_status": "unresolved_do_not_cross_align",
        },
    }


def new_native_identity(
    *,
    context: Mapping[str, Any] | None,
    operator_id: str,
    attempt_index: int,
    operator_role: str,
    parent_module: Mapping[str, Any] | None = None,
    inherited_homology: bool = False,
) -> dict[str, Any]:
    ctx = validate_native_creation_context(context)
    event_id = mutation_event_identity(
        ctx,
        operator_id=str(operator_id),
        attempt_index=int(attempt_index),
    )
    parent_instance = str((parent_module or {}).get("module_instance_id", "") or "")
    parent_homology = str((parent_module or {}).get("homology_id", "") or "")
    role_material = {
        "schema_version": STRUCTURAL_IDENTITY_SCHEMA_VERSION,
        "context": ctx,
        "event_id": event_id,
        "operator_id": str(operator_id),
        "operator_role": str(operator_role),
        "parent_module_instance_id": parent_instance,
        "parent_homology_id": parent_homology,
    }
    instance_id = _hash_id("mi", role_material)
    if inherited_homology and parent_homology:
        homology_id = parent_homology
    else:
        homology_id = _hash_id("mh", {**role_material, "kind": "new_structural_locus"})
    if parent_module is None:
        position = topology_position_from_material(
            {"schema_version": TOPOLOGY_POSITION_SCHEMA_VERSION, "homology_id": homology_id}
        )
    else:
        position = nearby_topology_position(
            parent_module.get("topology_pos", parent_module.get("latent_pos", [0.0, 0.0])),
            {"schema_version": TOPOLOGY_POSITION_SCHEMA_VERSION, "identity": role_material},
        )
    provenance_id = _hash_id(
        "sp",
        {
            "campaign_id": ctx["campaign_id"],
            "branch_id": ctx["branch_id"],
            "lineage_token": ctx["lineage_token"],
            "child_agent_id": ctx["child_agent_id"],
        },
    )
    return {
        "identity_schema_version": STRUCTURAL_IDENTITY_SCHEMA_VERSION,
        "module_instance_id": instance_id,
        "homology_id": homology_id,
        "identity_origin": IDENTITY_ORIGIN_NATIVE,
        "ancestry": {
            "schema_version": STRUCTURAL_ANCESTRY_SCHEMA_VERSION,
            "parent_module_instance_ids": [parent_instance] if parent_instance else [],
            "origin_operator": str(operator_id),
            "origin_generation": int(ctx["generation"]),
            "origin_mutation_event_id": event_id,
            "inherited_homology_id": parent_homology if inherited_homology else "",
            "provenance_id": provenance_id,
        },
        "topology_position_schema_version": TOPOLOGY_POSITION_SCHEMA_VERSION,
        "topology_pos": position,
        "legacy_identity_source": None,
    }


def module_is_native_homology(record: Mapping[str, Any]) -> bool:
    return bool(
        str(record.get("identity_origin", "")) == IDENTITY_ORIGIN_NATIVE
        and str(record.get("identity_schema_version", "")) == STRUCTURAL_IDENTITY_SCHEMA_VERSION
        and str(record.get("module_instance_id", ""))
        and str(record.get("homology_id", ""))
    )


def migrate_legacy_structural_payload(
    payload: Mapping[str, Any],
    *,
    strict: bool = False,
) -> dict[str, Any]:
    """Diagnostic migration envelope preserving the exact source for rollback."""

    source = deepcopy(dict(payload))
    modules = source.get("modules", [])
    if not isinstance(modules, list):
        raise StructuralIdentityError("legacy structural payload modules must be a list")
    migrated_modules: list[dict[str, Any]] = []
    for index, record in enumerate(modules):
        if not isinstance(record, Mapping):
            if strict:
                raise StructuralIdentityError(f"legacy module at index {index} is not a mapping")
            continue
        migrated = dict(record)
        migrated.update(
            canonicalize_module_identity(
                record,
                record_index=index,
                strict_partial_native=bool(strict),
            )
        )
        migrated["latent_pos"] = list(migrated["topology_pos"])
        migrated_modules.append(migrated)
    migrated_payload = deepcopy(source)
    migrated_payload["modules"] = migrated_modules
    return {
        "schema_version": STRUCTURAL_LEGACY_MIGRATION_VERSION,
        "source_payload": source,
        "migrated_payload": migrated_payload,
        "rng_draws": 0,
        "homology_policy": "legacy_modules_remain_unresolved_and_unmatched_across_parents",
    }
