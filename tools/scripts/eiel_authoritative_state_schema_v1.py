#!/usr/bin/env python3
"""Generate and check the canonical EIEL authoritative-state schema v1."""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import time
from typing import Any, Iterable, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENGINE_ROOT = PROJECT_ROOT / "engine"
PACKAGE_ROOT = PROJECT_ROOT / "_spec" / "authoritative_state" / "v1"
SCHEMA_PATH = PACKAGE_ROOT / "schema.json"
SOURCE_COMMIT_PATH = PACKAGE_ROOT / "source_commit.txt"
GENERATED_ROOT = ENGINE_ROOT / "authoritative_state" / "generated"
RUNTIME_ABI_CONTRACTS_PATH = (
    GENERATED_ROOT / "runtime_abi_contracts_v1.py"
)
BINARY_CONSTANTS_PATH = (
    ENGINE_ROOT
    / "native_runtime"
    / "authoritative_state"
    / "binary_identity_constants_v1.hpp"
)
COMPONENT_NAMES = (
    "field_registry.json",
    "abi_variants.json",
    "checkpoint_schema.json",
    "chronicle_schema.json",
    "provider_projection_schema.json",
    "analysis_projection_schema.json",
    "compatibility_matrix.json",
    "migration_rules.json",
    "migration_families.json",
    "generated_hashes.json",
)
RETAINED_PRODUCTION_SURFACE = {
    "provider_id": "eiel-production-provider-v1",
    "profile_id": "eiel-production-runtime-v1",
    "required_abi_variants": [
        "direct_resident_outer_172",
        "event_state_31",
        "full_tick_state_29",
        "ordinary_birth_diploid_122",
        "ordinary_birth_output_63",
        "ordinary_birth_scratch_1",
        "patch_p2_state_full_10",
        "turnover_mutable_state_122",
        "turnover_output_16",
    ],
}
BOUNDARY_TOKENS = frozenset(
    {
        "agent_id",
        "parent",
        "child",
        "lineage",
        "generation",
        "tick",
        "cursor",
        "count",
        "capacity",
        "energy",
        "health",
        "drive",
        "recurrent",
        "policy_cache",
        "rng",
        "pcg64",
        "birth",
        "event",
        "checkpoint",
        "chronicle",
        "schema",
        "abi",
        "artifact",
        "route",
    }
)
NATIVE_SOURCE_SUFFIXES = frozenset({".c", ".cc", ".cpp", ".cu", ".cuh", ".h", ".hpp"})
NATIVE_LEXEME = re.compile(
    r'"(?P<string>(?:\\.|[^"\\])*)"|(?P<symbol>\b[A-Za-z_]\w*\b)'
)


class GenerationError(RuntimeError):
    pass


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def value_hash(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise GenerationError(f"{path} root is not an object")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_bytes(value) + b"\n")


def normalize_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _source_anchors(paths: Iterable[str]) -> list[dict[str, Any]]:
    anchors: list[dict[str, Any]] = []
    for relative in paths:
        path = PROJECT_ROOT / relative
        if not path.is_file():
            raise GenerationError(f"source anchor is absent: {relative}")
        anchors.append(
            {
                "path": relative,
                "source_sha256": file_hash(path),
            }
        )
    return anchors


def _validate_abi_source_contract(abi: Mapping[str, Any]) -> None:
    tool = PROJECT_ROOT / "tools/scripts/eiel_authoritative_state_abi_v1.py"
    spec = importlib.util.spec_from_file_location(
        "eiel_authoritative_state_abi_v1_for_schema", tool
    )
    if spec is None or spec.loader is None:
        raise GenerationError("cannot load authoritative-state ABI generator")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    expected = module.build_contract()
    if dict(abi) != expected:
        raise GenerationError(
            "abi_variants.json differs from current production source contract"
        )


def _expand_fields(source: Mapping[str, Any], abi: Mapping[str, Any]) -> list[dict[str, Any]]:
    defaults = dict(source["field_defaults"])
    fields: dict[str, dict[str, Any]] = {}
    for group in source["field_groups"]:
        base = {**defaults, **{key: value for key, value in group.items() if key != "names"}}
        domain = normalize_name(str(base.pop("domain")))
        anchors = _source_anchors(base.pop("source_paths"))
        for raw_name in group["names"]:
            name = normalize_name(str(raw_name))
            field_id = f"{domain}.{name}.v1"
            if field_id in fields:
                raise GenerationError(f"duplicate canonical field {field_id}")
            field = copy.deepcopy(base)
            field.update(
                {
                    "field_id": field_id,
                    "name": name,
                    "meaning": f"Canonical {domain} authority for {name.replace('_', ' ')}.",
                    "source_anchors": anchors,
                    "resolution_status": "resolved",
                    "production_admission": "admitted",
                }
            )
            fields[field_id] = field

    # ABI entries are boundary fields too. Every ordered entry is represented.
    # Exact entries become resolved fields; incomplete families remain explicit
    # deferred rows that cannot be admitted by production.
    supported_variants = {
        str(row["variant_id"])
        for row in abi.get("variants", [])
        if row.get("support_status") == "supported_exact_metadata"
    }
    unsupported_reasons = {
        str(row["abi_family"]): str(row["reason"])
        for row in abi.get("unsupported_families", [])
    }
    abi_anchors = _source_anchors(
        source_row["path"] for source_row in abi["sources"]
    )
    for variant in abi.get("variants", []):
        variant_id = str(variant["variant_id"])
        family = str(variant["abi_family"])
        supported = variant_id in supported_variants
        for entry in variant.get("entries", []):
            field_id = str(entry["semantic_field_id"])
            if field_id in fields:
                existing = fields[field_id]
                variants = set(existing["abi_variants"])
                variants.add(variant_id)
                existing["abi_variants"] = sorted(variants)
                continue
            if supported and entry.get("metadata_status") == "supported_exact":
                dtype = str(entry["dtype"])
                fields[field_id] = {
                    **defaults,
                    "field_id": field_id,
                    "name": str(entry["name"]),
                    "meaning": f"Canonical ABI authority for {entry['name']}.",
                    "semantic_owner": str(entry["semantic_owner"]),
                    "producer": "canonical ABI generator",
                    "consumers": ["native CPU", "CUDA", "pybind"],
                    "logical_type": str(
                        entry.get("authority_contract", {}).get(
                            "kind", "source-bound ABI value"
                        )
                    ),
                    "physical_type": dtype,
                    "storage_representation": str(entry["physical_type"]),
                    "signedness": str(entry["signedness"]),
                    "bit_width": entry["bit_width"],
                    "scalar_or_tensor": str(
                        entry.get("authority_contract", {}).get("kind", "tensor")
                    ),
                    "rank": entry["rank"],
                    "shape_formula": str(entry["shape_formula"]),
                    "valid_row_count_formula": str(
                        entry["valid_row_count_formula"]
                    ),
                    "capacity_formula": str(entry["capacity_formula"]),
                    "valid_range": copy.deepcopy(entry["valid_range"]),
                    "sentinels": copy.deepcopy(entry["sentinels"]),
                    "optional": False,
                    "optionality_rule": str(entry["optionality"]),
                    "mutable": entry["mutability"] != "read_only",
                    "mutability_rule": str(entry["mutability"]),
                    "stale_tail_policy": str(entry["stale_tail_policy"]),
                    "checkpoint_encoding": str(
                        entry["checkpoint_participation"]
                    ),
                    "chronicle_encoding": str(entry["chronicle_participation"]),
                    "provider_projection": str(entry["provider_participation"]),
                    "analysis_projection": str(entry["analysis_participation"]),
                    "profile_availability": list(entry["profile_availability"]),
                    "boundary_test_profile": str(
                        entry["boundary_test_profile"]
                    ),
                    "source_declaration": str(entry["source_declaration"]),
                    "abi_variants": [variant_id],
                    "source_anchors": abi_anchors,
                    "resolution_status": "resolved",
                    "production_admission": "admitted",
                }
            else:
                reason = unsupported_reasons.get(
                    family, "ABI metadata is not closed"
                )
                fields[field_id] = {
                    **defaults,
                    "field_id": field_id,
                    "name": str(entry["name"]),
                    "meaning": (
                        f"Deferred ABI field {entry['name']} in unsupported "
                        f"family {family}."
                    ),
                    "semantic_owner": f"deferred ABI family {family}",
                    "producer": "canonical ABI order inventory",
                    "consumers": ["schema audit only"],
                    "logical_type": "deferred unsupported ABI field",
                    "physical_type": "unsupported_not_proven",
                    "signedness": "unsupported_not_proven",
                    "bit_width": "unsupported_not_proven",
                    "scalar_or_tensor": "unsupported_not_proven",
                    "rank": "unsupported_not_proven",
                    "shape_formula": "unsupported_not_proven",
                    "valid_row_count_formula": "unsupported_not_proven",
                    "capacity_formula": "unsupported_not_proven",
                    "valid_range": "unsupported_not_proven",
                    "sentinels": [],
                    "optional": "unsupported_not_proven",
                    "mutable": "unsupported_not_proven",
                    "lifecycle": "unsupported production surface",
                    "birth_reset": "unsupported_not_proven",
                    "checkpoint_encoding": "unsupported_not_proven",
                    "chronicle_encoding": "unsupported_not_proven",
                    "provider_projection": "unsupported_not_proven",
                    "analysis_projection": "unsupported_not_proven",
                    "abi_variants": [variant_id],
                    "artifact_participation": False,
                    "migration_rule": "unsupported_until_metadata_closed",
                    "boundary_test_profile": "source_order_and_hash_only",
                    "source_anchors": abi_anchors,
                    "resolution_status": "deferred_unsupported",
                    "production_admission": "forbidden",
                    "unsupported_family": family,
                    "unsupported_reason": reason,
                    "source_metadata": {
                        key: copy.deepcopy(entry[key])
                        for key in (
                            "physical_type",
                            "dtype",
                            "signedness",
                            "rank",
                            "shape_formula",
                            "valid_row_count_formula",
                            "capacity_formula",
                            "mutability",
                            "semantic_owner",
                            "optionality",
                            "checkpoint_participation",
                            "chronicle_participation",
                            "profile_availability",
                            "metadata_status",
                        )
                    },
                }
    for field in fields.values():
        field["abi_variants"] = sorted(set(field["abi_variants"]))
        field["contract_sha256"] = value_hash(field)
    return [fields[field_id] for field_id in sorted(fields)]


def _python_generated(fields: list[Mapping[str, Any]], schema_hash: str) -> str:
    ids = tuple(str(field["field_id"]) for field in fields)
    resolved = tuple(
        str(field["field_id"])
        for field in fields
        if field.get("resolution_status") == "resolved"
    )
    deferred = tuple(
        str(field["field_id"])
        for field in fields
        if field.get("resolution_status") == "deferred_unsupported"
    )
    physical = {str(field["field_id"]): str(field["physical_type"]) for field in fields}
    admission = {
        str(field["field_id"]): str(field["production_admission"])
        for field in fields
    }
    return (
        '"""Generated from _spec/authoritative_state/v1/schema.json; do not edit."""\n\n'
        f'SCHEMA_VERSION = "eiel_authoritative_state_v1"\n'
        'SCHEMA_IDENTITY_SOURCE = "_spec/authoritative_state/v1/schema.json"\n'
        f"FIELD_IDS = {ids!r}\n"
        f"RESOLVED_FIELD_IDS = {resolved!r}\n"
        f"DEFERRED_UNSUPPORTED_FIELD_IDS = {deferred!r}\n"
        f"PHYSICAL_TYPES = {physical!r}\n"
        f"PRODUCTION_ADMISSION = {admission!r}\n"
    )


def _runtime_abi_contracts_generated(abi: Mapping[str, Any]) -> str:
    prefixes = {
        "full_tick_state": "full_tick_state_tensors",
        "event_state": "event_state_tensors",
        "ordinary_birth_state": "ordinary_birth_state_tensors",
        "ordinary_birth_output": "ordinary_birth_output_tensors",
        "ordinary_birth_scratch": "ordinary_birth_scratch_tensors",
        "turnover_mutable_state": "turnover_state_tensors",
        "turnover_output": "turnover_output_tensors",
        "patch_p2_state": "patch_p2_state_tensors",
    }
    contracts: dict[str, dict[str, str]] = {}
    for variant in abi.get("variants", []):
        if variant.get("support_status") != "supported_exact_metadata":
            continue
        family = str(variant["abi_family"])
        for entry in variant.get("entries", []):
            name = str(entry["name"])
            path = (
                name
                if family == "direct_resident_outer"
                else f"{prefixes[family]}.{name}"
            )
            contract = {
                "shape_formula": str(entry["shape_formula"]),
                "valid_row_count_formula": str(
                    entry["valid_row_count_formula"]
                ),
            }
            prior = contracts.get(path)
            if prior is not None and prior != contract:
                raise GenerationError(
                    f"runtime ABI path has conflicting contracts: {path}"
                )
            contracts[path] = contract
    return (
        '"""Generated from abi_variants.json; do not edit."""\n\n'
        f"RUNTIME_ABI_PATH_CONTRACTS_V1 = {dict(sorted(contracts.items()))!r}\n"
    )


def _cpp_generated(fields: list[Mapping[str, Any]], schema_hash: str) -> str:
    rows = "\n".join(
        f'    "{field["field_id"]}",' for field in fields
    )
    resolved_rows = "\n".join(
        f'    "{field["field_id"]}",'
        for field in fields
        if field.get("resolution_status") == "resolved"
    )
    deferred_rows = "\n".join(
        f'    "{field["field_id"]}",'
        for field in fields
        if field.get("resolution_status") == "deferred_unsupported"
    )
    resolved_count = sum(
        field.get("resolution_status") == "resolved" for field in fields
    )
    deferred_count = sum(
        field.get("resolution_status") == "deferred_unsupported"
        for field in fields
    )
    return f"""// Generated from schema.json; do not edit.
#pragma once
#include <array>
#include <string_view>

namespace eiel::authoritative_state::v1 {{
inline constexpr std::string_view kSchemaVersion = "eiel_authoritative_state_v1";
inline constexpr std::string_view kSchemaIdentitySource =
    "_spec/authoritative_state/v1/schema.json";
inline constexpr std::array<std::string_view, {len(fields)}> kFieldIds = {{
{rows}
}};
inline constexpr std::array<std::string_view, {resolved_count}>
    kResolvedFieldIds = {{
{resolved_rows}
}};
inline constexpr std::array<std::string_view, {deferred_count}>
    kDeferredUnsupportedFieldIds = {{
{deferred_rows}
}};
}}  // namespace eiel::authoritative_state::v1
"""


def _binary_constants_generated(
    schema: Mapping[str, Any],
    abi: Mapping[str, Any],
) -> str:
    variants = sorted(
        (
            {
                "entrypoint": str(row["variant_id"]),
                "abi_version": str(row["variant_id"]),
                "argument_count": int(row["count"]),
                "argument_order_sha256": str(row["order_sha256"]),
            }
            for row in abi["variants"]
            if int(row["count"]) > 0
        ),
        key=lambda row: (row["entrypoint"], row["abi_version"]),
    )
    abi_set_sha256 = value_hash(variants)
    components = schema["component_hashes"]
    build_identity_sha256 = value_hash(
        {
            "schema_sha256": schema["schema_hash"],
            "abi_set_sha256": abi_set_sha256,
            "provider_capability_identity": (
                "eiel-production-provider-v1"
            ),
            "overlay_version": "inheritance-cross-path-overlay-v1",
            "supported_cache_widths": [1, 2],
            "supported_profiles": [
                "eiel-production-runtime-v1"
            ],
        }
    )
    variant_rows = "\n".join(
        "    AbiVariantConstantV1{"
        f'"{row["entrypoint"]}", '
        f'{row["argument_count"]}U, '
        f'"{row["argument_order_sha256"]}"'
        "},"
        for row in variants
    )
    return f"""// Generated by eiel_authoritative_state_schema_v1.py; do not edit.
#pragma once

#include <array>
#include <cstdint>
#include <string_view>

namespace eiel::authoritative_state::constants_v1 {{

struct AbiVariantConstantV1 {{
    std::string_view variant_id;
    std::uint32_t argument_count;
    std::string_view argument_order_sha256;
}};

inline constexpr std::string_view kSchemaVersion =
    "{schema["schema_version"]}";
inline constexpr std::string_view kSchemaSha256 =
    "{schema["schema_hash"]}";
inline constexpr std::string_view kSourceCommit =
    "{schema["source_commit"]}";
inline constexpr std::string_view kAbiSetSha256 =
    "{abi_set_sha256}";
inline constexpr std::string_view kSourceInventorySha256 =
    "{components["field_registry.json"]}";
inline constexpr std::string_view kBuildIdentitySha256 =
    "{build_identity_sha256}";
inline constexpr std::string_view kCheckpointSchemaSha256 =
    "{components["checkpoint_schema.json"]}";
inline constexpr std::string_view kChronicleSchemaSha256 =
    "{components["chronicle_schema.json"]}";
inline constexpr std::string_view kProviderCapabilityIdentity =
    "eiel-production-provider-v1";
inline constexpr std::string_view kArtifactBuildId =
    "state-schema-v1-sm75-{schema["schema_hash"][:12]}";
inline constexpr std::array<AbiVariantConstantV1, {len(variants)}>
    kAbiVariants = {{{{
{variant_rows}
}}}};

}}  // namespace eiel::authoritative_state::constants_v1
"""


def _render(source: dict[str, Any]) -> dict[Path, bytes]:
    abi = load_json(PACKAGE_ROOT / "abi_variants.json")
    _validate_abi_source_contract(abi)
    fields = _expand_fields(source, abi)
    resolved_count = sum(
        field.get("resolution_status") == "resolved" for field in fields
    )
    deferred_count = sum(
        field.get("resolution_status") == "deferred_unsupported"
        for field in fields
    )
    registry = {
        "schema_version": "eiel_authoritative_state_field_registry_v1",
        "canonical_schema_version": source["schema_version"],
        "source_commit": source["source_commit"],
        "field_count": len(fields),
        "resolved_field_count": resolved_count,
        "deferred_unsupported_field_count": deferred_count,
        "unresolved_supported_field_count": 0,
        "fields": fields,
    }
    registry["registry_sha256"] = value_hash(registry)
    rendered: dict[Path, bytes] = {
        PACKAGE_ROOT / "field_registry.json": canonical_bytes(registry) + b"\n",
        RUNTIME_ABI_CONTRACTS_PATH: _runtime_abi_contracts_generated(abi).encode(
            "utf-8"
        ),
    }
    component_hashes: dict[str, str] = {}
    for name in COMPONENT_NAMES:
        if name in {"field_registry.json", "generated_hashes.json"}:
            continue
        path = PACKAGE_ROOT / name
        if not path.is_file():
            raise GenerationError(f"required component is absent: {name}")
        component_hashes[name] = file_hash(path)
    component_hashes["field_registry.json"] = hashlib.sha256(
        rendered[PACKAGE_ROOT / "field_registry.json"]
    ).hexdigest()

    schema_material = copy.deepcopy(source)
    schema_material["component_hashes"] = {}
    schema_material["schema_hash"] = ""
    provisional = value_hash(schema_material)
    py_path = GENERATED_ROOT / "schema_v1.py"
    hpp_path = GENERATED_ROOT / "schema_v1.hpp"
    rendered[py_path] = _python_generated(fields, provisional).encode("utf-8")
    rendered[hpp_path] = _cpp_generated(fields, provisional).encode("utf-8")
    generated_hashes = {
        "schema_version": "eiel_authoritative_state_generated_hashes_v1",
        "source_commit": source["source_commit"],
        "generated": {
            str(path.relative_to(PROJECT_ROOT)): hashlib.sha256(payload).hexdigest()
            for path, payload in sorted(rendered.items(), key=lambda row: str(row[0]))
            if path.suffix in {".py", ".hpp"}
        },
        "components_excluding_self": dict(sorted(component_hashes.items())),
    }
    rendered[PACKAGE_ROOT / "generated_hashes.json"] = (
        canonical_bytes(generated_hashes) + b"\n"
    )
    component_hashes["generated_hashes.json"] = hashlib.sha256(
        rendered[PACKAGE_ROOT / "generated_hashes.json"]
    ).hexdigest()
    final_schema = copy.deepcopy(source)
    final_schema["component_hashes"] = dict(sorted(component_hashes.items()))
    final_schema.pop("schema_hash", None)
    final_schema["schema_hash"] = value_hash(final_schema)
    rendered[SCHEMA_PATH] = canonical_bytes(final_schema) + b"\n"

    # Re-render language constants with the final root identity.
    rendered[py_path] = _python_generated(fields, final_schema["schema_hash"]).encode(
        "utf-8"
    )
    rendered[hpp_path] = _cpp_generated(fields, final_schema["schema_hash"]).encode(
        "utf-8"
    )
    generated_hashes["generated"] = {
        str(path.relative_to(PROJECT_ROOT)): hashlib.sha256(payload).hexdigest()
        for path, payload in sorted(rendered.items(), key=lambda row: str(row[0]))
        if path.suffix in {".py", ".hpp"}
    }
    rendered[PACKAGE_ROOT / "generated_hashes.json"] = (
        canonical_bytes(generated_hashes) + b"\n"
    )
    final_schema["component_hashes"]["generated_hashes.json"] = hashlib.sha256(
        rendered[PACKAGE_ROOT / "generated_hashes.json"]
    ).hexdigest()
    material = dict(final_schema)
    material.pop("schema_hash", None)
    final_schema["schema_hash"] = value_hash(material)
    rendered[SCHEMA_PATH] = canonical_bytes(final_schema) + b"\n"
    # Schema hash changed only because generated_hashes changed.  One final
    # language render does not feed back into component hashes: its hashes are
    # evidence, not a schema component, avoiding a self-referential cycle.
    rendered[py_path] = _python_generated(fields, final_schema["schema_hash"]).encode(
        "utf-8"
    )
    rendered[hpp_path] = _cpp_generated(fields, final_schema["schema_hash"]).encode(
        "utf-8"
    )
    # This header embeds the final schema identity and therefore cannot
    # participate in the schema's own component-hash cycle. The immutable
    # extension build manifest independently hashes it as a build source.
    rendered[BINARY_CONSTANTS_PATH] = _binary_constants_generated(
        final_schema,
        abi,
    ).encode("utf-8")
    return rendered


def generate(*, check: bool) -> dict[str, Any]:
    started = time.perf_counter()
    source = load_json(SCHEMA_PATH)
    source_commit = SOURCE_COMMIT_PATH.read_text(encoding="ascii").strip()
    if re.fullmatch(r"[0-9a-f]{40}", source_commit) is None:
        raise GenerationError("source_commit.txt must contain one lowercase Git commit")
    source["source_commit"] = source_commit
    abi = load_json(PACKAGE_ROOT / "abi_variants.json")
    supported_surfaces = source.get("supported_surfaces")
    expected_surfaces = [RETAINED_PRODUCTION_SURFACE]
    legacy_160_surface = copy.deepcopy(RETAINED_PRODUCTION_SURFACE)
    legacy_160_surface["required_abi_variants"] = [
        "direct_resident_outer_160"
        if variant == "direct_resident_outer_172"
        else variant
        for variant in legacy_160_surface["required_abi_variants"]
    ]
    if supported_surfaces not in (expected_surfaces, [legacy_160_surface]):
        raise GenerationError(
            "supported_surfaces must contain exactly the canonical "
            "production engine surface"
        )
    support_by_variant = {
        str(row["variant_id"]): str(row.get("support_status"))
        for row in abi["variants"]
    }
    incomplete = [
        variant
        for variant in RETAINED_PRODUCTION_SURFACE["required_abi_variants"]
        if support_by_variant.get(variant) != "supported_exact_metadata"
    ]
    if incomplete:
        raise GenerationError(
            "canonical production admission requires exact metadata for ABI "
            f"variants {sorted(incomplete)}"
        )
    source["supported_surfaces"] = [copy.deepcopy(RETAINED_PRODUCTION_SURFACE)]
    source["unsupported_surfaces"] = []
    source["terminal_target"] = "STATE-SCHEMA-V1-B"
    rendered = _render(source)
    changed: list[str] = []
    for path, payload in rendered.items():
        current = path.read_bytes() if path.is_file() else None
        if current != payload:
            changed.append(str(path.relative_to(PROJECT_ROOT)))
            if not check:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(payload)
    if check and changed:
        raise GenerationError(f"generated schema drift: {changed}")
    return {
        "ok": True,
        "mode": "check" if check else "generate",
        "changed": changed,
        "elapsed_seconds": time.perf_counter() - started,
        "schema_hash": json.loads(rendered[SCHEMA_PATH])["schema_hash"],
        "field_count": json.loads(
            rendered[PACKAGE_ROOT / "field_registry.json"]
        )["field_count"],
    }


class _InventoryVisitor(ast.NodeVisitor):
    def __init__(self, relative: str) -> None:
        self.relative = relative
        self.rows: list[dict[str, Any]] = []

    def _record(self, name: str, line: int, context: str) -> None:
        normalized = normalize_name(name)
        if not normalized or not any(token in normalized for token in BOUNDARY_TOKENS):
            return
        self.rows.append(
            {
                "name": normalized,
                "path": self.relative,
                "line": line,
                "context": context,
            }
        )

    def visit_Constant(self, node: ast.Constant) -> None:
        if isinstance(node.value, str):
            self._record(node.value, node.lineno, "string_literal")

    def visit_Name(self, node: ast.Name) -> None:
        self._record(node.id, node.lineno, "symbol")


def _inventory_row(
    name: str, *, relative: str, line: int, context: str, language: str
) -> dict[str, Any] | None:
    normalized = normalize_name(name)
    if not normalized or not any(
        token in normalized for token in BOUNDARY_TOKENS
    ):
        return None
    return {
        "name": normalized,
        "path": relative,
        "line": line,
        "context": context,
        "language": language,
    }


def _native_inventory_rows(path: Path, *, relative: str) -> list[dict[str, Any]]:
    try:
        source = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return []
    source = re.sub(
        r"/\*.*?\*/",
        lambda match: "\n" * match.group(0).count("\n"),
        source,
        flags=re.DOTALL,
    )
    source = re.sub(r"//[^\n]*", "", source)
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(source.splitlines(), start=1):
        for match in NATIVE_LEXEME.finditer(line):
            context = "string_literal" if match.group("string") is not None else "symbol"
            value = match.group(context.split("_")[0])
            row = _inventory_row(
                value,
                relative=relative,
                line=line_number,
                context=context,
                language="native",
            )
            if row is not None:
                rows.append(row)
    return rows


def inventory() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    source_files: list[dict[str, str]] = []
    for path in sorted(ENGINE_ROOT.rglob("*.py")):
        relative = str(path.relative_to(PROJECT_ROOT))
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=relative)
        except (OSError, SyntaxError):
            continue
        source_files.append(
            {"path": relative, "language": "python", "sha256": file_hash(path)}
        )
        visitor = _InventoryVisitor(relative)
        visitor.visit(tree)
        rows.extend(
            {**row, "language": "python"}
            for row in visitor.rows
        )
    for path in sorted(
        path
        for path in ENGINE_ROOT.rglob("*")
        if path.is_file() and path.suffix in NATIVE_SOURCE_SUFFIXES
    ):
        relative = str(path.relative_to(PROJECT_ROOT))
        source_files.append(
            {"path": relative, "language": "native", "sha256": file_hash(path)}
        )
        rows.extend(_native_inventory_rows(path, relative=relative))
    rows.sort(
        key=lambda row: (
            row["path"],
            row["line"],
            row["context"],
            row["name"],
        )
    )
    return {
        "schema_version": "eiel_authoritative_state_context_inventory_v1",
        "method": "python_ast_and_native_lexical_symbols_and_string_literals",
        "source_file_count": len(source_files),
        "source_files_sha256": value_hash(source_files),
        "source_files": source_files,
        "declaration_count": len(rows),
        "declarations": rows,
    }


def migration_plan(from_path: Path, to_path: Path) -> dict[str, Any]:
    old = load_json(from_path)
    new = load_json(to_path)
    old_fields = {row["field_id"]: row for row in old["fields"]}
    new_fields = {row["field_id"]: row for row in new["fields"]}
    added = sorted(set(new_fields) - set(old_fields))
    removed = sorted(set(old_fields) - set(new_fields))
    changed = sorted(
        field_id
        for field_id in set(old_fields) & set(new_fields)
        if old_fields[field_id].get("contract_sha256")
        != new_fields[field_id].get("contract_sha256")
    )
    category = (
        "compatible_no_op"
        if not added and not removed and not changed
        else "requires_checkpoint_rewrite"
        if removed or changed
        else "compatible_with_deterministic_conversion"
    )
    return {
        "schema_version": "eiel_authoritative_state_migration_plan_v1",
        "category": category,
        "added": added,
        "removed": removed,
        "changed": changed,
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    sub = result.add_subparsers(dest="command", required=True)
    sub.add_parser("generate")
    sub.add_parser("check")
    inv = sub.add_parser("inventory")
    inv.add_argument("--output")
    diff = sub.add_parser("diff")
    diff.add_argument("from_path", type=Path)
    diff.add_argument("to_path", type=Path)
    migrate = sub.add_parser("migration-plan")
    migrate.add_argument("from_path", type=Path)
    migrate.add_argument("to_path", type=Path)
    return result


def main() -> int:
    args = parser().parse_args()
    try:
        if args.command in {"generate", "check"}:
            output = generate(check=args.command == "check")
        elif args.command == "inventory":
            output = inventory()
            if args.output:
                write_json(Path(args.output), output)
        else:
            output = migration_plan(args.from_path, args.to_path)
    except (GenerationError, OSError, ValueError, KeyError) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, sort_keys=True))
        return 1
    print(json.dumps(output, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
