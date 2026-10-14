#!/usr/bin/env python3
from __future__ import annotations

"""Validate the Architecture B compact-generation-summary v1 sidecars and rows."""

import argparse
import hashlib
import json
import math
import re
import struct
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final


ROOT: Final[Path] = Path(__file__).resolve().parents[2]
CONTRACT_ROOT: Final[Path] = ROOT / "docs" / "40_eiel_design"
DEFAULT_SCHEMA_PATH: Final[Path] = CONTRACT_ROOT / "compact_generation_summary_v1.json"
DEFAULT_ABI_PATH: Final[Path] = CONTRACT_ROOT / "compact_summary_native_abi_v1.json"
DEFAULT_ABI_V2_PATH: Final[Path] = CONTRACT_ROOT / "compact_summary_native_abi_v2.json"
DEFAULT_PLANE_LAYOUT_V2_PATH: Final[Path] = (
    CONTRACT_ROOT / "compact_summary_plane_layout_v2.json"
)
DEFAULT_PROJECTION_V2_PATH: Final[Path] = (
    CONTRACT_ROOT / "compact_summary_projection_contract_v2.json"
)
DEFAULT_INVALID_ROW_V2_PATH: Final[Path] = (
    CONTRACT_ROOT / "compact_summary_invalid_row_contract_v2.json"
)
DEFAULT_OWNERSHIP_PATH: Final[Path] = CONTRACT_ROOT / "compact_summary_source_ownership.json"
DEFAULT_SCHEDULE_PATH: Final[Path] = CONTRACT_ROOT / "compact_summary_checkpoint_schedule.json"
DEFAULT_EXACTNESS_PATH: Final[Path] = CONTRACT_ROOT / "compact_summary_exactness_rules.json"
DEFAULT_DIGEST_PREIMAGES_PATH: Final[Path] = (
    CONTRACT_ROOT / "compact_summary_digest_preimages_v1.json"
)
DEFAULT_DOCUMENT_PATH: Final[Path] = CONTRACT_ROOT / "compact_generation_summary_v1.md"

WIRE_SCHEMA_VERSION: Final[str] = "eiel_compact_generation_summary_v1"
VALIDATOR_SCHEMA_VERSION: Final[str] = "eiel_compact_generation_summary_validator_v1"
PHASE_A_LAYOUT_VERSION: Final[str] = (
    "eiel_cuda_architecture_b_generation_summary_scalar_v1"
)
PHASE_A_COLUMNS: Final[tuple[str, ...]] = (
    "generation",
    "event_cursor_start",
    "event_cursor_pre_turnover",
    "event_cursor_end",
    "alive_after_turnover",
    "energy_sum_after_turnover",
    "births",
    "deaths",
    "failed_births",
    "action_total",
    "move_total",
    "gather_total",
    "moved_total",
    "gathered_total",
    "movement_attempts",
    "movement_blocked_occupied",
    "movement_blocked_bounds",
    "peak_injury",
    "policy_loop_tick_calls",
    "policy_refill_calls",
    "policy_refill_rows",
    "policy_cache_hit_calls",
    "policy_cache_rows_requested",
    "policy_cache_rows_hit",
    "policy_cache_rows_missed",
    "policy_cache_rows_refilled",
    "policy_cache_full_refill_ticks",
    "policy_cache_partial_refill_ticks",
    "patch_food_stock_after_turnover",
    "energy_collected_exact_sum_after_turnover",
    "policy_energy_live_sum_after_turnover",
)


class ContractValidationError(ValueError):
    """The contract bundle or candidate row is not exact compact summary v1."""


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ContractValidationError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_json(path: Path) -> Any:
    """Load strict JSON while rejecting duplicate keys and non-finite constants."""

    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_object_without_duplicate_keys,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ContractValidationError(f"non-finite JSON constant: {value}")
            ),
        )
    except ContractValidationError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ContractValidationError(f"cannot load strict JSON {path}: {exc}") from exc


def canonical_json_bytes(value: Any) -> bytes:
    try:
        payload = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ContractValidationError(f"value is not canonical JSON: {exc}") from exc
    return (payload + "\n").encode("utf-8")


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _ordered_fingerprint(value: Any) -> str:
    try:
        payload = json.dumps(
            value,
            sort_keys=False,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ContractValidationError(f"value is not strict JSON: {exc}") from exc
    return hashlib.sha256(payload).hexdigest()


def _fail(path: str, message: str) -> None:
    raise ContractValidationError(f"{path}: {message}")


def _require_mapping(value: Any, *, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(path, "must be an object")
    return value


def _require_exact_keys(
    value: Mapping[str, Any], expected: Sequence[str], *, path: str
) -> None:
    expected_set = set(expected)
    missing = sorted(expected_set - set(value))
    extra = sorted(set(value) - expected_set)
    if missing:
        _fail(path, f"missing required field(s): {missing}")
    if extra:
        _fail(path, f"additional field(s) forbidden: {extra}")


def _resolve_ref(schema: Mapping[str, Any], root: Mapping[str, Any]) -> Mapping[str, Any]:
    ref = schema.get("$ref")
    if ref is None:
        return schema
    prefix = "#/$defs/"
    if not isinstance(ref, str) or not ref.startswith(prefix):
        raise ContractValidationError(f"unsupported schema reference: {ref!r}")
    name = ref[len(prefix) :]
    definitions = root.get("$defs")
    if not isinstance(definitions, Mapping) or name not in definitions:
        raise ContractValidationError(f"unresolved schema reference: {ref}")
    target = definitions[name]
    if not isinstance(target, Mapping):
        raise ContractValidationError(f"schema reference is not an object: {ref}")
    return target


def _matches_type(value: Any, expected: str) -> bool:
    if expected == "null":
        return value is None
    if expected == "boolean":
        return type(value) is bool
    if expected == "integer":
        return type(value) is int
    if expected == "number":
        return type(value) in {int, float} and math.isfinite(value)
    if expected == "string":
        return type(value) is str
    if expected == "array":
        return type(value) is list
    if expected == "object":
        return isinstance(value, Mapping)
    raise ContractValidationError(f"unsupported schema type: {expected}")


def _validate_schema_value(
    value: Any,
    schema: Mapping[str, Any],
    root: Mapping[str, Any],
    *,
    path: str,
) -> None:
    schema = _resolve_ref(schema, root)
    branches = schema.get("oneOf")
    if branches is not None:
        if not isinstance(branches, list):
            _fail(path, "schema oneOf must be an array")
        matches = 0
        errors: list[str] = []
        for branch in branches:
            if not isinstance(branch, Mapping):
                _fail(path, "schema oneOf entries must be objects")
            try:
                _validate_schema_value(value, branch, root, path=path)
            except ContractValidationError as exc:
                errors.append(str(exc))
            else:
                matches += 1
        if matches != 1:
            _fail(path, f"must match exactly one schema branch, matched {matches}: {errors}")
        return

    if "const" in schema:
        constant = schema["const"]
        if type(value) is not type(constant) or value != constant:
            _fail(path, f"must equal {constant!r}")
    if "enum" in schema and not any(
        type(value) is type(item) and value == item for item in schema["enum"]
    ):
        _fail(path, f"must be one of {schema['enum']!r}")

    expected_type = schema.get("type")
    if expected_type is not None:
        expected_types = expected_type if isinstance(expected_type, list) else [expected_type]
        if not all(isinstance(item, str) for item in expected_types):
            _fail(path, "schema type must be a string or string array")
        if not any(_matches_type(value, item) for item in expected_types):
            _fail(path, f"must have type {expected_type!r}")
    if (
        schema.get("x-canonical-python-type") == "float"
        and value is not None
        and type(value) is not float
    ):
        _fail(path, "must be a canonical Python float")

    if type(value) in {int, float}:
        if not math.isfinite(value):
            _fail(path, "must be finite")
        minimum = schema.get("minimum")
        maximum = schema.get("maximum")
        if minimum is not None and value < minimum:
            _fail(path, f"must be at least {minimum}")
        if maximum is not None and value > maximum:
            _fail(path, f"must be at most {maximum}")

    if type(value) is str:
        minimum_length = schema.get("minLength")
        if minimum_length is not None and len(value) < minimum_length:
            _fail(path, f"must contain at least {minimum_length} character(s)")
        pattern = schema.get("pattern")
        if pattern is not None and re.fullmatch(pattern, value) is None:
            _fail(path, f"must match {pattern!r}")
        maximum_utf8_bytes = schema.get("x-max-utf8-bytes")
        if (
            maximum_utf8_bytes is not None
            and len(value.encode("utf-8")) > maximum_utf8_bytes
        ):
            _fail(path, f"must use at most {maximum_utf8_bytes} UTF-8 bytes")

    if type(value) is list:
        item_schema = schema.get("items")
        if item_schema is not None:
            if not isinstance(item_schema, Mapping):
                _fail(path, "schema items must be an object")
            for index, item in enumerate(value):
                _validate_schema_value(item, item_schema, root, path=f"{path}[{index}]")
        if schema.get("uniqueItems"):
            fingerprints = [canonical_json_bytes(item) for item in value]
            if len(fingerprints) != len(set(fingerprints)):
                _fail(path, "must contain unique items")

    if isinstance(value, Mapping):
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        if not isinstance(properties, Mapping) or not isinstance(required, list):
            _fail(path, "schema object properties/required are malformed")
        missing = [key for key in required if key not in value]
        if missing:
            _fail(path, f"missing required field(s): {missing}")
        extras = [key for key in value if key not in properties]
        if extras and schema.get("additionalProperties") is False:
            _fail(path, f"additional field(s) forbidden: {extras}")
        for key, child in value.items():
            child_schema = properties.get(key)
            if child_schema is not None:
                if not isinstance(child_schema, Mapping):
                    _fail(f"{path}.{key}", "schema property must be an object")
                _validate_schema_value(child, child_schema, root, path=f"{path}.{key}")


def validate_schema_definition(schema: Mapping[str, Any]) -> None:
    if schema.get("$id") != WIRE_SCHEMA_VERSION:
        _fail("$schema", f"unexpected $id {schema.get('$id')!r}")
    if schema.get("additionalProperties") is not False:
        _fail("$schema", "top-level additionalProperties must be false")
    properties = _require_mapping(schema.get("properties"), path="$schema.properties")
    required = schema.get("required")
    if not isinstance(required, list) or len(required) != len(set(required)):
        _fail("$schema.required", "must be a unique array")
    if set(properties) != set(required):
        _fail("$schema", "every top-level property must be required")
    for field, definition in properties.items():
        if not isinstance(definition, Mapping):
            _fail(f"$schema.properties.{field}", "must be an object")
        owner = definition.get("x-owner")
        if not isinstance(owner, str) or not owner:
            _fail(f"$schema.properties.{field}", "x-owner is required")

    definitions = _require_mapping(schema.get("$defs"), path="$schema.$defs")
    for name, definition in definitions.items():
        if not isinstance(definition, Mapping):
            _fail(f"$schema.$defs.{name}", "must be an object")
        nested = definition.get("properties")
        if nested is None:
            continue
        if not isinstance(nested, Mapping):
            _fail(f"$schema.$defs.{name}.properties", "must be an object")
        if definition.get("additionalProperties") is not False:
            _fail(f"$schema.$defs.{name}", "wire objects must be closed")
        nested_required = definition.get("required")
        if not isinstance(nested_required, list) or set(nested_required) != set(nested):
            _fail(f"$schema.$defs.{name}", "every object property must be required")


def validate_ownership(
    schema: Mapping[str, Any], ownership: Mapping[str, Any]
) -> dict[str, Any]:
    if ownership.get("schema_version") != "eiel_compact_summary_source_ownership_v1":
        _fail("$ownership.schema_version", "unexpected value")
    if ownership.get("wire_schema_version") != WIRE_SCHEMA_VERSION:
        _fail("$ownership.wire_schema_version", "wire schema mismatch")
    producers = _require_mapping(ownership.get("producers"), path="$ownership.producers")
    field_owners = _require_mapping(
        ownership.get("field_owners"), path="$ownership.field_owners"
    )
    properties = _require_mapping(schema.get("properties"), path="$schema.properties")
    _require_exact_keys(field_owners, list(properties), path="$ownership.field_owners")
    for field, definition in properties.items():
        assert isinstance(definition, Mapping)
        expected_owner = definition["x-owner"]
        actual_owner = field_owners[field]
        if actual_owner not in producers:
            _fail(
                f"$ownership.field_owners.{field}",
                f"unknown producer {actual_owner!r}",
            )
        if actual_owner != expected_owner:
            _fail(
                f"$ownership.field_owners.{field}",
                f"misowned field: expected {expected_owner!r}, found {actual_owner!r}",
            )
    unresolved = ownership.get("unresolved_native_producer_fields")
    if (
        not isinstance(unresolved, list)
        or unresolved != sorted(unresolved)
        or len(unresolved) != len(set(unresolved))
        or not unresolved
    ):
        _fail("$ownership.unresolved_native_producer_fields", "must be sorted, unique, and nonempty")
    claims = _require_mapping(ownership.get("claims"), path="$ownership.claims")
    if claims.get("canonical_native_producer_complete") is not False:
        _fail("$ownership.claims", "canonical producer must remain false")
    if claims.get("canonical_runtime_capability") is not False:
        _fail("$ownership.claims", "canonical capability must remain false")
    if claims.get("phase_a_equals_canonical_v1") is not False:
        _fail("$ownership.claims", "Phase A must not equal canonical v1")
    return {
        "owned_field_count": len(field_owners),
        "producer_count": len(producers),
        "unresolved_native_producer_fields": list(unresolved),
    }


def validate_native_abi(abi: Mapping[str, Any]) -> dict[str, Any]:
    if abi.get("schema_version") != "eiel_compact_summary_native_abi_v1":
        _fail("$abi.schema_version", "unexpected value")
    if abi.get("wire_schema_version") != WIRE_SCHEMA_VERSION:
        _fail("$abi.wire_schema_version", "wire schema mismatch")
    phase_a = _require_mapping(abi.get("implemented_phase_a"), path="$abi.implemented_phase_a")
    if phase_a.get("layout_version") != PHASE_A_LAYOUT_VERSION:
        _fail("$abi.implemented_phase_a.layout_version", "Phase A layout mismatch")
    if tuple(phase_a.get("columns", ())) != PHASE_A_COLUMNS:
        _fail("$abi.implemented_phase_a.columns", "must match the exact 31-column source layout")
    if phase_a.get("shape") != ["generation_count", "cells", len(PHASE_A_COLUMNS)]:
        _fail("$abi.implemented_phase_a.shape", "must be [generation_count,cells,31]")
    for field, expected in (
        ("implemented", True),
        ("canonical_compact_v1", False),
        ("preallocated_by_caller", True),
        ("allocated_in_generation_loop", False),
        ("intermediate_host_transfer", False),
        ("intermediate_host_synchronization", False),
    ):
        if phase_a.get(field) is not expected:
            _fail(f"$abi.implemented_phase_a.{field}", f"must be {expected}")

    projection = _require_mapping(
        abi.get("canonical_projection_from_phase_a"),
        path="$abi.canonical_projection_from_phase_a",
    )
    direct = _require_mapping(
        projection.get("directly_usable_inputs"),
        path="$abi.canonical_projection_from_phase_a.directly_usable_inputs",
    )
    _require_exact_keys(direct, PHASE_A_COLUMNS, path="$abi.canonical_projection_from_phase_a.directly_usable_inputs")
    missing = projection.get("not_derivable_from_phase_a_alone")
    if not isinstance(missing, list) or not missing:
        _fail("$abi.canonical_projection_from_phase_a", "canonical gaps must remain explicit")

    target = _require_mapping(abi.get("target_canonical_v1"), path="$abi.target_canonical_v1")
    if target.get("implemented") is not False:
        _fail("$abi.target_canonical_v1.implemented", "must remain false")
    plane_rows = target.get("planes")
    if not isinstance(plane_rows, list):
        _fail("$abi.target_canonical_v1.planes", "must be an array")
    plane_names = [row.get("name") for row in plane_rows if isinstance(row, Mapping)]
    expected_planes = [
        "validity",
        "status_codes",
        "integer_values",
        "float_values",
        "digest_values",
        "optionality_mask",
        "text_records",
    ]
    if plane_names != expected_planes:
        _fail("$abi.target_canonical_v1.planes", f"must be {expected_planes}")
    expected_plane_layouts = {
        "validity": ("uint8", ["generation_count", "cells"]),
        "status_codes": ("int32", ["generation_count", "cells", 10]),
        "integer_values": ("int64", ["generation_count", "cells", 76]),
        "float_values": ("float64", ["generation_count", "cells", 21]),
        "digest_values": (
            "uint8",
            ["generation_count", "cells", 10, 32],
        ),
        "optionality_mask": ("uint32", ["generation_count", "cells"]),
        "text_records": (
            "uint8",
            ["generation_count", "cells", 3, 65],
        ),
    }
    for row in plane_rows:
        if not isinstance(row, Mapping):
            _fail("$abi.target_canonical_v1.planes", "plane entries must be objects")
        name = row.get("name")
        expected_dtype, expected_shape = expected_plane_layouts[name]
        if row.get("dtype") != expected_dtype or row.get("shape") != expected_shape:
            _fail(
                f"$abi.target_canonical_v1.planes.{name}",
                f"must be {expected_dtype} {expected_shape}",
            )

    registry = _require_mapping(
        target.get("field_registry"), path="$abi.target_canonical_v1.field_registry"
    )
    registry_counts = {
        "status_codes": 10,
        "integer_values": 76,
        "float_values": 21,
        "digest_values": 10,
        "text_records": 3,
    }
    _require_exact_keys(
        registry,
        tuple(registry_counts),
        path="$abi.target_canonical_v1.field_registry",
    )
    for name, count in registry_counts.items():
        fields = registry[name]
        if (
            not isinstance(fields, list)
            or len(fields) != count
            or len(set(fields)) != count
            or any(type(field) is not str or not field for field in fields)
        ):
            _fail(
                f"$abi.target_canonical_v1.field_registry.{name}",
                f"must contain {count} unique non-empty paths",
            )

    codebooks = _require_mapping(
        target.get("status_codebooks"),
        path="$abi.target_canonical_v1.status_codebooks",
    )
    _require_exact_keys(
        codebooks,
        registry["status_codes"],
        path="$abi.target_canonical_v1.status_codebooks",
    )
    for name, codebook_value in codebooks.items():
        codebook = _require_mapping(
            codebook_value,
            path=f"$abi.target_canonical_v1.status_codebooks.{name}",
        )
        codes = list(codebook.values())
        if (
            not codes
            or 0 not in codes
            or len(codes) != len(set(codes))
            or any(type(code) is not int or code < 0 for code in codes)
        ):
            _fail(
                f"$abi.target_canonical_v1.status_codebooks.{name}",
                "must reserve unique nonnegative code zero",
            )

    optionality = target.get("optionality_bits")
    if (
        not isinstance(optionality, list)
        or len(optionality) != 32
        or len(set(optionality)) != 32
        or optionality[17] != "patch_metrics"
        or any(not field.startswith("patch_metrics.") for field in optionality[18:])
    ):
        _fail(
            "$abi.target_canonical_v1.optionality_bits",
            "must contain the exact 17+1+14 global optionality layout",
        )
    if target.get("logical_bytes_per_generation_cell") != 1336:
        _fail(
            "$abi.target_canonical_v1.logical_bytes_per_generation_cell",
            "must equal 1336",
        )
    text_rules = _require_mapping(
        target.get("text_rules"), path="$abi.target_canonical_v1.text_rules"
    )
    if (
        text_rules.get("record_width") != 65
        or text_rules.get("length_byte") != 0
        or text_rules.get("payload_bytes") != [1, 64]
        or text_rules.get("encoding") != "strict UTF-8"
    ):
        _fail("$abi.target_canonical_v1.text_rules", "bounded text ABI changed")
    float_rules = _require_mapping(
        target.get("float_rules"), path="$abi.target_canonical_v1.float_rules"
    )
    if (
        float_rules.get("bit_preserving_decode") is not True
        or float_rules.get("negative_zero_preserved") is not True
        or float_rules.get("arithmetic_during_decode") is not False
    ):
        _fail("$abi.target_canonical_v1.float_rules", "float bits must remain exact")
    sticky = _require_mapping(target.get("sticky_failure_rule"), path="$abi.target_canonical_v1.sticky_failure_rule")
    if sticky.get("later_mutation") is not False:
        _fail("$abi.target_canonical_v1.sticky_failure_rule", "later mutation must be false")

    capability = _require_mapping(abi.get("capability_truth"), path="$abi.capability_truth")
    expected_capability = {
        "resident_generation_summaries_supported": False,
        "eiel_compact_generation_summary_v1_supported": False,
        "phase_a_generation_summary_scalar_supported": True,
        "phase_a_is_canonical_v1": False,
    }
    if dict(capability) != expected_capability:
        _fail("$abi.capability_truth", f"must equal {expected_capability}")
    missing_producer = abi.get("precise_missing_native_producer")
    if not isinstance(missing_producer, str) or len(missing_producer.strip()) < 40:
        _fail("$abi.precise_missing_native_producer", "must be precise and nonempty")
    return {
        "phase_a_layout_version": PHASE_A_LAYOUT_VERSION,
        "phase_a_width": len(PHASE_A_COLUMNS),
        "phase_a_is_canonical_v1": False,
        "canonical_native_capability": False,
        "precise_missing_native_producer": missing_producer,
    }


def validate_native_abi_v2_contracts(
    abi: Mapping[str, Any],
    plane_layout: Mapping[str, Any],
    projection: Mapping[str, Any],
    invalid_row: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the lossless ABI v2 envelope without claiming qualification."""

    if abi.get("schema_version") != "eiel_compact_summary_native_abi_v2":
        _fail("$abi_v2.schema_version", "unexpected value")
    if abi.get("wire_schema_version") != WIRE_SCHEMA_VERSION:
        _fail("$abi_v2.wire_schema_version", "wire schema mismatch")
    expected_planes = {
        "validity": ("uint8", ["generation_count", "cells"], 1),
        "defined_mask": ("uint64", ["generation_count", "cells", 1], 8),
        "status_codes": ("int32", ["generation_count", "cells", 12], 48),
        "integer_values": ("int64", ["generation_count", "cells", 76], 608),
        "float_values": ("float64", ["generation_count", "cells", 21], 168),
        "digest_values": (
            "uint8",
            ["generation_count", "cells", 10, 32],
            320,
        ),
        "text_values": (
            "uint8",
            ["generation_count", "cells", 1, 65],
            65,
        ),
    }
    planes = abi.get("planes")
    if not isinstance(planes, list):
        _fail("$abi_v2.planes", "must be an array")
    if [row.get("name") for row in planes if isinstance(row, Mapping)] != list(
        expected_planes
    ):
        _fail("$abi_v2.planes", f"must be ordered as {list(expected_planes)}")
    for row in planes:
        if not isinstance(row, Mapping):
            _fail("$abi_v2.planes", "entries must be objects")
        name = row.get("name")
        dtype, shape, row_bytes = expected_planes[name]
        if (
            row.get("dtype") != dtype
            or row.get("shape") != shape
            or row.get("logical_bytes_per_row") != row_bytes
        ):
            _fail(
                f"$abi_v2.planes.{name}",
                f"must be {dtype} {shape} with {row_bytes} logical bytes",
            )
    if abi.get("logical_bytes_per_generation_cell") != sum(
        value[2] for value in expected_planes.values()
    ):
        _fail("$abi_v2.logical_bytes_per_generation_cell", "must equal 1218")

    capability = _require_mapping(abi.get("capability"), path="$abi_v2.capability")
    for field in (
        "resident_generation_summaries_supported",
        "eiel_compact_generation_summary_v1_supported",
    ):
        if capability.get(field) is not False:
            _fail(f"$abi_v2.capability.{field}", "must remain false before qualification")
    decisions = _require_mapping(
        abi.get("representation_decisions"),
        path="$abi_v2.representation_decisions",
    )
    if decisions.get("general_defined_mask") is not True:
        _fail("$abi_v2.representation_decisions.general_defined_mask", "must be true")
    if decisions.get("definedness_inferred_from_value") is not False:
        _fail(
            "$abi_v2.representation_decisions.definedness_inferred_from_value",
            "must be false",
        )
    if decisions.get("bounded_text_fields") != [
        "lineage_metrics.dominant_lineage_id"
    ]:
        _fail(
            "$abi_v2.representation_decisions.bounded_text_fields",
            "only the proven-unbounded lineage identifier may use text storage",
        )

    if plane_layout.get("schema_version") != "eiel_compact_summary_plane_layout_v2":
        _fail("$plane_layout_v2.schema_version", "unexpected value")
    if plane_layout.get("logical_bytes_per_generation_cell") != 1218:
        _fail("$plane_layout_v2.logical_bytes_per_generation_cell", "must equal 1218")
    if plane_layout.get("validity_written_last") is not True:
        _fail("$plane_layout_v2.validity_written_last", "must be true")
    if [row.get("name") for row in plane_layout.get("planes", ())] != list(
        expected_planes
    ):
        _fail("$plane_layout_v2.planes", "plane ordering disagrees with ABI v2")

    if projection.get("schema_version") != (
        "eiel_compact_summary_projection_contract_v2"
    ):
        _fail("$projection_v2.schema_version", "unexpected value")
    if projection.get("projection_boundary") != (
        "after_one_final_device_synchronization"
    ):
        _fail("$projection_v2.projection_boundary", "must use one final synchronization")
    if projection.get("projection_consumes_rng") is not False:
        _fail("$projection_v2.projection_consumes_rng", "must be false")
    if projection.get("projection_mutates_simulation_state") is not False:
        _fail("$projection_v2.projection_mutates_simulation_state", "must be false")
    canonical = _require_mapping(
        projection.get("canonical_bytes"), path="$projection_v2.canonical_bytes"
    )
    if canonical.get("terminal_newline") != "LF":
        _fail("$projection_v2.canonical_bytes.terminal_newline", "must preserve frozen LF")

    if invalid_row.get("schema_version") != (
        "eiel_compact_summary_invalid_row_contract_v2"
    ):
        _fail("$invalid_row_v2.schema_version", "unexpected value")
    invalid_planes = _require_mapping(
        invalid_row.get("planes"), path="$invalid_row_v2.planes"
    )
    _require_exact_keys(invalid_planes, tuple(expected_planes), path="$invalid_row_v2.planes")
    if invalid_row.get("canonical_projection_permitted") is not False:
        _fail("$invalid_row_v2.canonical_projection_permitted", "must be false")
    if invalid_row.get("completed_rows_before_failure_remain_valid_and_immutable") is not True:
        _fail(
            "$invalid_row_v2.completed_rows_before_failure_remain_valid_and_immutable",
            "must be true",
        )

    return {
        "schema_version": abi["schema_version"],
        "wire_schema_version": abi["wire_schema_version"],
        "logical_bytes_per_generation_cell": 1218,
        "defined_field_count": 32,
        "status_field_count": 12,
        "integer_field_count": 76,
        "float_field_count": 21,
        "digest_field_count": 10,
        "bounded_text_field_count": 1,
        "capability_promoted": False,
    }


def _schedule_checkpoint_coordinates(
    *, origin: int, next_generations: Sequence[int], explicit: Sequence[int] = ()
) -> list[int]:
    explicit_set = set(explicit)
    return [
        generation
        for generation in next_generations
        if (generation - origin) > 0
        and ((generation - origin) % 8 == 0 or generation in explicit_set)
    ]


def validate_schedule(schedule: Mapping[str, Any]) -> dict[str, Any]:
    if schedule.get("schema_version") != "eiel_compact_summary_checkpoint_schedule_v1":
        _fail("$schedule.schema_version", "unexpected value")
    if schedule.get("wire_schema_version") != WIRE_SCHEMA_VERSION:
        _fail("$schedule.wire_schema_version", "wire schema mismatch")
    channels = _require_mapping(schedule.get("channels"), path="$schedule.channels")
    _require_exact_keys(channels, ("S1", "S8", "explicit_request"), path="$schedule.channels")
    for name, cadence in (("S1", 1), ("S8", 8)):
        channel = _require_mapping(channels[name], path=f"$schedule.channels.{name}")
        if channel.get("cadence_generations") != cadence:
            _fail(f"$schedule.channels.{name}", f"cadence must be {cadence}")
        if channel.get("schedule_math_validator_implemented") is not True:
            _fail(f"$schedule.channels.{name}", "schedule math validator must be true")
        if channel.get("canonical_runtime_qualified") is not False:
            _fail(f"$schedule.channels.{name}", "runtime qualification must remain false")
    explicit_channel = _require_mapping(
        channels["explicit_request"], path="$schedule.channels.explicit_request"
    )
    if explicit_channel.get("live_mid_run_injection_supported") is not False:
        _fail("$schedule.channels.explicit_request", "live injection must remain false")

    cases = schedule.get("validation_cases")
    if not isinstance(cases, list) or len(cases) != 4:
        _fail("$schedule.validation_cases", "four frozen cases are required")
    by_id = {
        row.get("id"): row for row in cases if isinstance(row, Mapping) and row.get("id")
    }
    for case_id in ("origin_zero_1_to_9", "explicit_overlap", "nonzero_origin"):
        case = _require_mapping(by_id.get(case_id), path=f"$schedule.validation_cases.{case_id}")
        actual = _schedule_checkpoint_coordinates(
            origin=case["schedule_origin_generation"],
            next_generations=case["next_generations"],
            explicit=case.get("explicit_next_generations", ()),
        )
        if actual != case["checkpoint_due"]:
            _fail(f"$schedule.validation_cases.{case_id}", f"checkpoint coordinates must be {actual}")
    resume = _require_mapping(
        by_id.get("resume_preserves_phase"),
        path="$schedule.validation_cases.resume_preserves_phase",
    )
    future = range(resume["resume_generation"] + 1, resume["resume_generation"] + 9)
    actual_resume = _schedule_checkpoint_coordinates(
        origin=resume["schedule_origin_generation"], next_generations=future
    )[0]
    if actual_resume != resume["next_checkpoint_generation"]:
        _fail("$schedule.validation_cases.resume_preserves_phase", "resume phase rebased")

    gap = _require_mapping(
        schedule.get("current_implementation_gap"), path="$schedule.current_implementation_gap"
    )
    if gap.get("blocking") is not True or gap.get("s8_currently_accepted") is not False:
        _fail("$schedule.current_implementation_gap", "S8 must remain a blocking gap")
    claims = _require_mapping(schedule.get("claims"), path="$schedule.claims")
    for field in (
        "s1_canonical_runtime_qualified",
        "s8_canonical_runtime_qualified",
        "checkpoint_due_proves_persistence",
    ):
        if claims.get(field) is not False:
            _fail(f"$schedule.claims.{field}", "must remain false")
    return {"schedule_cases_validated": sorted(by_id), "s8_runtime_qualified": False}


def validate_exactness_rules(rules: Mapping[str, Any]) -> dict[str, Any]:
    if rules.get("schema_version") != "eiel_compact_summary_exactness_rules_v1":
        _fail("$exactness.schema_version", "unexpected value")
    if rules.get("wire_schema_version") != WIRE_SCHEMA_VERSION:
        _fail("$exactness.wire_schema_version", "wire schema mismatch")
    if rules.get("validator_schema_version") != VALIDATOR_SCHEMA_VERSION:
        _fail("$exactness.validator_schema_version", "validator schema mismatch")
    comparison = _require_mapping(rules.get("comparison"), path="$exactness.comparison")
    required_truth = (
        "validate_both_rows_first",
        "missing_fields_fail",
        "extra_fields_fail",
        "type_changes_fail",
        "canonical_bytes_must_match",
        "canonical_sha256_must_match",
        "first_mismatch_required_on_failure",
    )
    if any(comparison.get(field) is not True for field in required_truth):
        _fail("$exactness.comparison", "all fail-closed and canonical checks must be true")
    if comparison.get("numeric_tolerance") != 0:
        _fail("$exactness.comparison.numeric_tolerance", "must be zero")
    if comparison.get("integer_float_coercion_allowed") is not False:
        _fail("$exactness.comparison", "integer/float coercion must be false")
    if comparison.get("nan_or_infinity_allowed") is not False:
        _fail("$exactness.comparison", "non-finite values must be forbidden")
    phase_a = _require_mapping(
        rules.get("phase_a_scalar_conversion"), path="$exactness.phase_a_scalar_conversion"
    )
    if phase_a.get("source_layout") != PHASE_A_LAYOUT_VERSION:
        _fail("$exactness.phase_a_scalar_conversion.source_layout", "layout mismatch")
    if phase_a.get("maximum_exact_integer_magnitude") != (1 << 53) - 1:
        _fail("$exactness.phase_a_scalar_conversion", "float64 integer limit mismatch")
    if phase_a.get("conversion_alone_can_create_canonical_v1") is not False:
        _fail("$exactness.phase_a_scalar_conversion", "conversion capability must remain false")
    claims = _require_mapping(rules.get("claims"), path="$exactness.claims")
    for field in (
        "live_cpu_cuda_row_parity_proven",
        "canonical_native_producer_accepted",
        "canonical_capability",
    ):
        if claims.get(field) is not False:
            _fail(f"$exactness.claims.{field}", "must remain false")
    return {"numeric_tolerance": 0, "canonical_comparison": True}


def _null_paths(value: Any, *, path: str = "") -> list[str]:
    if value is None:
        return [path]
    if isinstance(value, Mapping):
        paths: list[str] = []
        for key, child in value.items():
            if not path and key == "undefined_fields":
                continue
            child_path = f"{path}.{key}" if path else key
            paths.extend(_null_paths(child, path=child_path))
        return paths
    if type(value) is list:
        paths = []
        for index, child in enumerate(value):
            paths.extend(_null_paths(child, path=f"{path}[{index}]"))
        return paths
    return []


def _sum_event_counts(counts: Mapping[str, Any]) -> int:
    return sum(
        counts[name]
        for name in ("birth_committed", "mutation", "birth_failed", "death_committed")
    )


def _validate_row_invariants(summary: Mapping[str, Any]) -> None:
    start = summary["generation_start"]
    end = summary["generation_end"]
    next_start = summary["next_generation_start"]
    deaths_by_class = summary["deaths_by_class"]
    events = summary["events"]
    generation_counts = events["generation_counts_by_type"]
    turnover_counts = events["turnover_counts_by_type"]
    identity = summary["identity"]
    terminal = summary["terminal"]
    metrics = summary["generation_metrics"]
    behavior = summary["behavior_metrics"]

    checks = (
        (start["generation"] == end["generation"], "generation markers disagree"),
        (next_start["generation"] == end["generation"] + 1, "next generation marker must increment by one"),
        (end["tick_exclusive"] == metrics["policy_loop_tick_calls"], "generation end tick must equal policy loop tick calls"),
        (summary["alive"] == metrics["capacity"] - metrics["free_slots"], "pre-turnover alive/capacity/free-slot conservation failed"),
        (summary["deaths"] == sum(deaths_by_class.values()), "death-class counts do not sum to deaths"),
        (events["authoritative_cursor"] == events["cursor_end"], "authoritative event cursor must equal cursor_end"),
        (events["cursor_pre_turnover"] - events["cursor_start"] == events["generation_event_count"], "generation event cursor span mismatch"),
        (events["cursor_end"] - events["cursor_pre_turnover"] == events["turnover_event_count"], "turnover event cursor span mismatch"),
        (events["cursor_end"] - events["cursor_start"] == events["event_row_count"], "event row count/cursor span mismatch"),
        (_sum_event_counts(generation_counts) == events["generation_event_count"], "generation event type counts do not sum to span"),
        (_sum_event_counts(turnover_counts) == events["turnover_event_count"], "turnover event type counts do not sum to span"),
        (summary["births"] == generation_counts["birth_committed"], "birth count disagrees with generation events"),
        (summary["failed_births"] == generation_counts["birth_failed"], "failed-birth count disagrees with generation events"),
        (summary["deaths"] == generation_counts["death_committed"], "death count disagrees with generation events"),
        (identity["active_agent_count"] + identity["free_slot_count"] == metrics["capacity"], "post-turnover identity/free-slot conservation failed"),
        (behavior["move_total"] == behavior["move_n"] + behavior["move_s"] + behavior["move_e"] + behavior["move_w"], "behavior move conservation failed"),
        (behavior["action_total"] == behavior["move_total"] + behavior["gather"] + behavior["rest"] + behavior["attack"], "behavior action conservation failed"),
        (behavior["cohort_survivors"] <= behavior["cohort_size"] <= behavior["row_count"], "behavior cohort counts are inconsistent"),
        (metrics["policy_cache_rows_requested"] == metrics["policy_cache_rows_hit"] + metrics["policy_cache_rows_missed"], "policy-cache request conservation failed"),
        (metrics["policy_cache_rows_missed"] == metrics["policy_cache_rows_refilled"], "policy-cache refill conservation failed"),
        ((summary["patch_metrics"] is None) == (summary["world"] == "flat"), "Patch optionality mismatch"),
    )
    for passed, message in checks:
        if not passed:
            _fail("$", message)

    active_count = identity["active_agent_count"]
    minimum = identity["active_agent_id_min"]
    maximum = identity["active_agent_id_max"]
    if active_count == 0:
        if minimum is not None or maximum is not None:
            _fail("$.identity", "empty active identity requires null min/max")
    else:
        if minimum is None or maximum is None or minimum > maximum:
            _fail("$.identity", "nonempty active identity requires ordered min/max")
        if maximum >= identity["population_next_id"]:
            _fail("$.identity.population_next_id", "must exceed every active agent ID")

    recoverable = terminal["recoverable_error_status"]
    error_code = terminal["recoverable_error_code"]
    error_digest = terminal["recoverable_error_digest_sha256"]
    if recoverable == "none" and (error_code is not None or error_digest is not None):
        _fail("$.terminal", "recoverable error fields must be null when status is none")
    if recoverable == "present" and (not error_code or error_digest is None):
        _fail("$.terminal", "recoverable error fields are required when status is present")

    failure_values = (
        terminal["failure_generation"],
        terminal["failure_tick"],
        terminal["failure_stage"],
    )
    if terminal["status"] == "failed":
        if (
            terminal["sticky_failure"] is not True
            or any(value is None for value in failure_values)
            or not terminal["failure_stage"]
        ):
            _fail("$.terminal", "failed status requires sticky failure coordinates and stage")
    elif terminal["sticky_failure"] is not False or any(value is not None for value in failure_values):
        _fail("$.terminal", "nonfailed status forbids sticky failure coordinates and stage")

    undefined = summary["undefined_fields"]
    if undefined != sorted(undefined):
        _fail("$.undefined_fields", "must use sorted deterministic order")
    expected_undefined = sorted(_null_paths(summary))
    if undefined != expected_undefined:
        _fail(
            "$.undefined_fields",
            f"must list every and only null path; expected {expected_undefined}",
        )


def validate_summary(
    summary: Mapping[str, Any],
    schema: Mapping[str, Any] | None = None,
    ownership: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate one canonical row without mutating it or any contract input."""

    schema = load_json(DEFAULT_SCHEMA_PATH) if schema is None else schema
    ownership = load_json(DEFAULT_OWNERSHIP_PATH) if ownership is None else ownership
    summary_before = _ordered_fingerprint(summary)
    schema_before = _ordered_fingerprint(schema)
    ownership_before = _ordered_fingerprint(ownership)
    validate_schema_definition(schema)
    validate_ownership(schema, ownership)
    _validate_schema_value(summary, schema, schema, path="$")
    _validate_row_invariants(summary)
    if _ordered_fingerprint(summary) != summary_before:
        raise ContractValidationError("validator mutated the summary")
    if _ordered_fingerprint(schema) != schema_before:
        raise ContractValidationError("validator mutated the schema")
    if _ordered_fingerprint(ownership) != ownership_before:
        raise ContractValidationError("validator mutated the ownership contract")
    return {
        "ok": True,
        "validator_schema_version": VALIDATOR_SCHEMA_VERSION,
        "wire_schema_version": summary["schema_version"],
        "world": summary["world"],
        "generation": summary["generation_end"]["generation"],
        "canonical_sha256": canonical_json_sha256(summary),
        "read_only": True,
    }


def _first_mismatch(expected: Any, actual: Any, *, path: str = "$") -> dict[str, Any] | None:
    if type(expected) is not type(actual):
        return {
            "path": path,
            "reason": "type_mismatch",
            "expected_type": type(expected).__name__,
            "actual_type": type(actual).__name__,
        }
    if isinstance(expected, Mapping):
        if set(expected) != set(actual):
            return {
                "path": path,
                "reason": "object_key_mismatch",
                "expected_keys": sorted(expected),
                "actual_keys": sorted(actual),
            }
        for key in expected:
            mismatch = _first_mismatch(expected[key], actual[key], path=f"{path}.{key}")
            if mismatch is not None:
                return mismatch
        return None
    if type(expected) is list:
        if len(expected) != len(actual):
            return {
                "path": path,
                "reason": "array_length_mismatch",
                "expected": len(expected),
                "actual": len(actual),
            }
        for index, value in enumerate(expected):
            mismatch = _first_mismatch(value, actual[index], path=f"{path}[{index}]")
            if mismatch is not None:
                return mismatch
        return None
    if type(expected) is float:
        equal = struct.pack(">d", expected) == struct.pack(">d", actual)
    else:
        equal = expected == actual
    if not equal:
        return {"path": path, "reason": "value_mismatch", "expected": expected, "actual": actual}
    return None


def compare_exact_rows(
    expected: Mapping[str, Any],
    actual: Mapping[str, Any],
    schema: Mapping[str, Any] | None = None,
    ownership: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate and compare two rows with exact types, values, bytes, and digest."""

    expected_receipt = validate_summary(expected, schema, ownership)
    actual_receipt = validate_summary(actual, schema, ownership)
    mismatch = _first_mismatch(expected, actual)
    expected_bytes = canonical_json_bytes(expected)
    actual_bytes = canonical_json_bytes(actual)
    exact = mismatch is None and expected_bytes == actual_bytes
    return {
        "exact": exact,
        "first_mismatch": mismatch,
        "expected_sha256": expected_receipt["canonical_sha256"],
        "actual_sha256": actual_receipt["canonical_sha256"],
        "canonical_bytes_equal": expected_bytes == actual_bytes,
        "numeric_tolerance": 0,
    }


def assert_exact_rows(
    expected: Mapping[str, Any],
    actual: Mapping[str, Any],
    schema: Mapping[str, Any] | None = None,
    ownership: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    receipt = compare_exact_rows(expected, actual, schema, ownership)
    if not receipt["exact"]:
        mismatch = receipt["first_mismatch"] or {"path": "$", "reason": "canonical_bytes_mismatch"}
        _fail(mismatch["path"], f"exact row mismatch: {mismatch['reason']}")
    return receipt


def validate_digest_preimages(
    schema: Mapping[str, Any], digest_preimages: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate that every wire digest is bound to one exact authority preimage."""

    expected_top_level = (
        "schema_version",
        "wire_schema_version",
        "status",
        "hash_algorithm",
        "digest_encoding",
        "release_endianness",
        "cross_endian_behavior",
        "canonical_json_without_lf",
        "tensor_manifest_envelope",
        "bindings",
        "claims",
    )
    _require_exact_keys(digest_preimages, expected_top_level, path="$digest_preimages")
    expected_scalars = {
        "schema_version": "eiel_compact_summary_digest_preimages_v1",
        "wire_schema_version": WIRE_SCHEMA_VERSION,
        "status": (
            "preimages_frozen_host_producers_complete_"
            "native_producers_pending"
        ),
        "hash_algorithm": "sha256",
        "digest_encoding": "lowercase_hex",
        "release_endianness": "little",
        "cross_endian_behavior": "fail_closed_before_digest",
    }
    for field, expected in expected_scalars.items():
        if digest_preimages.get(field) != expected:
            _fail(f"$digest_preimages.{field}", f"must equal {expected!r}")

    bindings = _require_mapping(
        digest_preimages.get("bindings"), path="$digest_preimages.bindings"
    )
    expected_bindings = (
        "events.active_prefix_digest_sha256",
        "identity.active_agent_ids_digest_sha256",
        "rng.authority_digest_sha256",
        "rng.policy_mutation_tape_digest_sha256",
        "terminal.recoverable_error_digest_sha256",
        "state_digests.authoritative_state_digest_sha256",
        "state_digests.canonical_checkpoint_storage_digest_sha256",
        "state_digests.next_generation_start_state_digest_sha256",
        "behavior_metrics.rows_digest_sha256",
        "lineage_metrics.records_digest_sha256",
    )
    _require_exact_keys(bindings, expected_bindings, path="$digest_preimages.bindings")

    lifecycle_fields = bindings["events.active_prefix_digest_sha256"].get("fields")
    expected_lifecycle_fields = [
        "actor_agent_id",
        "actor_slot",
        "child_agent_id",
        "event_type",
        "generation",
        "lineage_depth",
        "lineage_founder_id",
        "parent_agent_id",
        "reason_code",
        "sequence",
        "subtype_code",
        "success",
        "target_agent_id",
        "target_slot",
        "tick",
        "value",
    ]
    if lifecycle_fields != expected_lifecycle_fields:
        _fail("$digest_preimages.bindings.events", "typed lifecycle fields changed")

    behavior_definition = _resolve_ref(
        schema["properties"]["behavior_metrics"], schema
    )
    behavior_fields = bindings["behavior_metrics.rows_digest_sha256"].get("fields")
    expected_behavior_fields = [
        "generation",
        "agent_id",
        "cohort",
        "move_n",
        "move_s",
        "move_e",
        "move_w",
        "gather",
        "rest",
        "attack",
        "lifespan_steps",
        "energy_collected",
        "offspring",
        "attacks_made",
        "fitness",
        "alive_end",
        "eligible_for_selection",
        "kept_next",
        "injury_end",
        "injury_delta",
        "aggression_bias",
    ]
    if behavior_fields != expected_behavior_fields or len(set(behavior_fields)) != 21:
        _fail("$digest_preimages.bindings.behavior_metrics", "behavior row fields changed")
    required_behavior_summary = set(behavior_definition.get("required") or ())
    if "rows_digest_sha256" not in required_behavior_summary:
        _fail("$schema.$defs.behavior_metrics", "rows digest must remain required")

    rng_definition = _resolve_ref(schema["properties"]["rng"], schema)
    rng_required = tuple(rng_definition.get("required") or ())
    if "authority_digest_sha256" not in rng_required or any(
        field in rng_required
        for field in ("pcg64_state_digest_sha256", "pcg64_cursor_digest_sha256")
    ):
        _fail("$schema.$defs.rng", "must bind the composite checkpoint RNG authority")
    rng_binding = bindings["rng.authority_digest_sha256"]
    if (
        rng_binding.get("manifest_path") != "rng.digest"
        or rng_binding.get("require_complete") is not True
        or rng_binding.get("split_state_or_cursor_digests") is not False
    ):
        _fail("$digest_preimages.bindings.rng", "composite RNG authority is not exact")

    state_layers = {
        "state_digests.authoritative_state_digest_sha256": "trajectory_state",
        "state_digests.canonical_checkpoint_storage_digest_sha256": (
            "canonical_checkpoint_storage"
        ),
    }
    for field, layer in state_layers.items():
        if bindings[field].get("layer") != layer:
            _fail(f"$digest_preimages.bindings.{field}", f"layer must be {layer}")

    tape = bindings["rng.policy_mutation_tape_digest_sha256"]
    if tape.get("consumed_rows") != "half-open [0,cursor) on every tape field":
        _fail("$digest_preimages.bindings.rng.policy_mutation_tape", "cursor slice changed")
    if len(tape.get("fields") or ()) != 5 or tape.get("unconsumed_capacity_included") is not False:
        _fail("$digest_preimages.bindings.rng.policy_mutation_tape", "tape fields changed")

    claims = _require_mapping(
        digest_preimages.get("claims"), path="$digest_preimages.claims"
    )
    if claims.get("digest_preimages_frozen") is not True:
        _fail("$digest_preimages.claims", "preimages must be frozen")
    if claims.get("native_digest_plane_producer_complete") is not False:
        _fail("$digest_preimages.claims", "native producer must remain unqualified")
    return {
        "schema_version": digest_preimages["schema_version"],
        "binding_count": len(bindings),
        "digest_preimages_frozen": True,
        "native_digest_plane_producer_complete": False,
    }


def validate_contract_bundle(
    *,
    schema_path: Path = DEFAULT_SCHEMA_PATH,
    abi_path: Path = DEFAULT_ABI_PATH,
    ownership_path: Path = DEFAULT_OWNERSHIP_PATH,
    schedule_path: Path = DEFAULT_SCHEDULE_PATH,
    exactness_path: Path = DEFAULT_EXACTNESS_PATH,
    digest_preimages_path: Path = DEFAULT_DIGEST_PREIMAGES_PATH,
    document_path: Path = DEFAULT_DOCUMENT_PATH,
) -> dict[str, Any]:
    schema = load_json(schema_path)
    abi = load_json(abi_path)
    ownership = load_json(ownership_path)
    schedule = load_json(schedule_path)
    exactness = load_json(exactness_path)
    digest_preimages = load_json(digest_preimages_path)
    values = (schema, abi, ownership, schedule, exactness, digest_preimages)
    before = [_ordered_fingerprint(value) for value in values]

    validate_schema_definition(schema)
    ownership_receipt = validate_ownership(schema, ownership)
    abi_receipt = validate_native_abi(abi)
    schedule_receipt = validate_schedule(schedule)
    exactness_receipt = validate_exactness_rules(exactness)
    digest_receipt = validate_digest_preimages(schema, digest_preimages)
    try:
        document = document_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise ContractValidationError(f"cannot load contract document {document_path}: {exc}") from exc
    for marker in (
        "resident_generation_summaries_supported=false",
        "eiel_compact_generation_summary_v1_supported=false",
        "The 31 scalars are inputs to a future canonicalizer",
    ):
        if marker not in document:
            _fail("$document", f"missing capability-boundary marker {marker!r}")

    after = [_ordered_fingerprint(value) for value in values]
    if before != after:
        raise ContractValidationError("contract bundle validation mutated input data")
    return {
        "ok": True,
        "validator_schema_version": VALIDATOR_SCHEMA_VERSION,
        "wire_schema_version": WIRE_SCHEMA_VERSION,
        "wire_field_count": len(schema["properties"]),
        "phase_a": abi_receipt,
        "ownership": ownership_receipt,
        "schedule": schedule_receipt,
        "exactness": exactness_receipt,
        "digest_preimages": digest_receipt,
        "canonical_native_capability": False,
        "read_only": True,
        "simulation_runs_started": 0,
        "cuda_builds_started": 0,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA_PATH)
    parser.add_argument("--abi", type=Path, default=DEFAULT_ABI_PATH)
    parser.add_argument("--ownership", type=Path, default=DEFAULT_OWNERSHIP_PATH)
    parser.add_argument("--schedule", type=Path, default=DEFAULT_SCHEDULE_PATH)
    parser.add_argument("--exactness", type=Path, default=DEFAULT_EXACTNESS_PATH)
    parser.add_argument(
        "--digest-preimages", type=Path, default=DEFAULT_DIGEST_PREIMAGES_PATH
    )
    parser.add_argument("--document", type=Path, default=DEFAULT_DOCUMENT_PATH)
    parser.add_argument("--summary", type=Path, help="validate one canonical row")
    parser.add_argument("--expected", type=Path, help="expected canonical row")
    parser.add_argument("--actual", type=Path, help="actual canonical row")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        schema = load_json(args.schema)
        ownership = load_json(args.ownership)
        if args.summary is not None and (args.expected is not None or args.actual is not None):
            raise ContractValidationError("--summary cannot be combined with --expected/--actual")
        if (args.expected is None) != (args.actual is None):
            raise ContractValidationError("--expected and --actual must be supplied together")
        if args.summary is not None:
            result = validate_summary(load_json(args.summary), schema, ownership)
        elif args.expected is not None:
            result = compare_exact_rows(
                load_json(args.expected), load_json(args.actual), schema, ownership
            )
        else:
            result = validate_contract_bundle(
                schema_path=args.schema,
                abi_path=args.abi,
                ownership_path=args.ownership,
                schedule_path=args.schedule,
                exactness_path=args.exactness,
                digest_preimages_path=args.digest_preimages,
                document_path=args.document,
            )
    except ContractValidationError as exc:
        print(
            json.dumps(
                {
                    "ok": False,
                    "validator_schema_version": VALIDATOR_SCHEMA_VERSION,
                    "error": str(exc),
                },
                sort_keys=True,
            )
        )
        return 1
    print(json.dumps(result, sort_keys=True))
    if args.expected is not None and not result["exact"]:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
