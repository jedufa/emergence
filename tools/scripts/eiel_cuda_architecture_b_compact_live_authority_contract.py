#!/usr/bin/env python3
"""Generate the frozen Phase-0 compact live-authority contract artifacts."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final


ROOT: Final[Path] = Path(__file__).resolve().parents[2]
ENGINE_ROOT: Final[Path] = ROOT / "engine"
for _import_root in (ROOT, ENGINE_ROOT):
    if str(_import_root) not in sys.path:
        sys.path.insert(0, str(_import_root))

from engine.native_runtime import checkpoint_digests  # noqa: E402
from engine.native_runtime import compact_summary_planes as abi  # noqa: E402
from engine.native_runtime import (  # noqa: E402
    compact_summary_retained_digest_finalizer as retained_finalizer,
)
from engine.native_runtime import (  # noqa: E402
    compact_summary_checkpoint_descriptor_plan as descriptor_plan,
)


OUTPUT_FILENAMES: Final[tuple[str, ...]] = (
    "compact_live_authority_pod_v1.md",
    "compact_live_authority_pod_v1.json",
    "compact_live_authority_source_map_v1.json",
    "compact_generation_metadata_retention_v1.json",
    "compact_rng_outer_preimage_v1.json",
)

SOURCE_CLASSES: Final[tuple[str, ...]] = (
    "static_metadata",
    "pre_turnover_reporting_retention",
    "post_turnover_resident_state",
    "settled_failure_status",
    "checkpoint_tensor_digest_retention",
    "composite_rng_metadata_retention",
    "event_state",
    "identity_state",
    "behavior_producer_retention",
    "lineage_producer_retention",
    "patch_disabled_constant_undefined",
)

POD_SCHEMA_VERSION: Final[str] = (
    "eiel_cuda_architecture_b_compact_live_authority_pod_v1"
)
SOURCE_MAP_SCHEMA_VERSION: Final[str] = (
    "eiel_cuda_architecture_b_compact_live_authority_source_map_v1"
)
GENERATION_METADATA_SCHEMA_VERSION: Final[str] = (
    "eiel_cuda_architecture_b_compact_generation_metadata_retention_v1"
)
RNG_OUTER_PREIMAGE_SCHEMA_VERSION: Final[str] = (
    "eiel_cuda_architecture_b_compact_rng_outer_preimage_v1"
)

SOURCE_PROJECTION_HEADER: Final[Path] = (
    ROOT
    / "engine/perf/"
    "cuda_architecture_b_compact_summary_source_projection_v1.cuh"
)
BOUNDARY_HEADER: Final[Path] = (
    ROOT / "engine/perf/cuda_architecture_b_compact_summary_boundary_v1.cuh"
)
DESCRIPTOR_PLAN_PATH: Final[Path] = (
    ROOT / "docs/300_architecture-specifications/370_specifications/EIEL-DOC-000840--compact-summary-checkpoint-descriptor-plan-v1.json"
)

POD_BYTE_WIDTH: Final[int] = 1256
POD_ALIGNMENT: Final[int] = 8
PROJECTION_STATUS_BYTES: Final[int] = 4
ROW_ACCOUNTING_BYTES: Final[int] = POD_BYTE_WIDTH + PROJECTION_STATUS_BYTES
DESCRIPTOR_COUNT: Final[int] = 133
DESCRIPTOR_BATCH_COUNT: Final[int] = 9
HOST_CANONICAL_DIGEST_MASK: Final[int] = 0x7F

POD_MEMBERS: Final[tuple[dict[str, Any], ...]] = (
    {"name": "source_version", "offset": 0, "bytes": 4, "type": "uint32"},
    {"name": "disposition", "offset": 4, "bytes": 4, "type": "uint32"},
    {"name": "generation_index", "offset": 8, "bytes": 8, "type": "int64"},
    {"name": "cell_index", "offset": 16, "bytes": 8, "type": "int64"},
    {"name": "validity", "offset": 24, "bytes": 1, "type": "uint8"},
    {"name": "header_reserved", "offset": 25, "bytes": 7, "type": "uint8[7]"},
    {"name": "defined_mask", "offset": 32, "bytes": 8, "type": "uint64"},
    {"name": "status_codes", "offset": 40, "bytes": 48, "type": "int32[12]"},
    {"name": "integer_values", "offset": 88, "bytes": 608, "type": "int64[76]"},
    {"name": "float_values", "offset": 696, "bytes": 168, "type": "float64[21]"},
    {"name": "digest_values", "offset": 864, "bytes": 320, "type": "uint8[10][32]"},
    {"name": "text_values", "offset": 1184, "bytes": 65, "type": "uint8[1][65]"},
    {"name": "tail_reserved", "offset": 1249, "bytes": 7, "type": "uint8[7]"},
)

HOST_CANONICAL_DIGEST_SLOTS: Final[tuple[dict[str, Any], ...]] = (
    {
        "field": "rng.authority_digest_sha256",
        "boundary_enum": "kHostDigestRngAuthorityV1",
        "mask": 1 << 0,
        "source_class": "composite_rng_metadata_retention",
    },
    {
        "field": "rng.policy_mutation_tape_digest_sha256",
        "boundary_enum": "kHostDigestRngPolicyMutationTapeV1",
        "mask": 1 << 1,
        "source_class": "composite_rng_metadata_retention",
    },
    {
        "field": "state_digests.authoritative_state_digest_sha256",
        "boundary_enum": "kHostDigestAuthoritativeStateV1",
        "mask": 1 << 2,
        "source_class": "checkpoint_tensor_digest_retention",
    },
    {
        "field": "state_digests.canonical_checkpoint_storage_digest_sha256",
        "boundary_enum": "kHostDigestCanonicalCheckpointStorageV1",
        "mask": 1 << 3,
        "source_class": "checkpoint_tensor_digest_retention",
    },
    {
        "field": "state_digests.next_generation_start_state_digest_sha256",
        "boundary_enum": "kHostDigestNextGenerationStartStateV1",
        "mask": 1 << 4,
        "source_class": "checkpoint_tensor_digest_retention",
    },
    {
        "field": "behavior_metrics.rows_digest_sha256",
        "boundary_enum": "kHostDigestBehaviorRowsV1",
        "mask": 1 << 5,
        "source_class": "behavior_producer_retention",
    },
    {
        "field": "lineage_metrics.records_digest_sha256",
        "boundary_enum": "kHostDigestLineageRecordsV1",
        "mask": 1 << 6,
        "source_class": "lineage_producer_retention",
    },
)

DEVICE_RETAINED_DIGEST_FIELDS: Final[tuple[dict[str, str], ...]] = (
    {
        "field": "events.active_prefix_digest_sha256",
        "source_class": "event_state",
        "physical_source": (
            "device-retained typed lifecycle active prefix plus "
            "engine/native_runtime/compact_summary_retained_digest_finalizer.py:"
            "authority.lifecycle.active_prefix_digest_sha256"
        ),
    },
    {
        "field": "identity.active_agent_ids_digest_sha256",
        "source_class": "identity_state",
        "physical_source": (
            "device-retained checkpoint:agent_ids_by_alive_order digest plus "
            "engine/native_runtime/compact_summary_retained_digest_finalizer.py:"
            "authority.identity.tensor_sha256"
        ),
    },
)


def _required_constant(module: Any, name: str) -> Any:
    if not hasattr(module, name):
        module_name = getattr(module, "__name__", repr(module))
        raise RuntimeError(f"missing required frozen constant: {module_name}.{name}")
    return getattr(module, name)


def _require_tokens(path: Path, tokens: Sequence[str]) -> str:
    if not path.is_file():
        raise RuntimeError(f"missing required repository anchor: {path.relative_to(ROOT)}")
    text = path.read_text(encoding="utf-8")
    for token in tokens:
        if token not in text:
            raise RuntimeError(
                f"missing required frozen constant: {path.relative_to(ROOT)}:{token}"
            )
    return text


def _verify_frozen_anchors() -> None:
    source = _require_tokens(
        SOURCE_PROJECTION_HEADER,
        (
            "struct CompactSummaryGenerationCellAuthorityV1",
            "kAuthorityByteWidthV1 = 1256",
            "alignof(CompactSummaryGenerationCellAuthorityV1) == 8",
            "sizeof(CompactSummaryGenerationCellAuthorityV1) == kAuthorityByteWidthV1",
            "kAuthorityStatusValueCountV1 == 12",
            "kAuthorityIntegerValueCountV1 == 76",
            "kAuthorityFloatValueCountV1 == 21",
            "kAuthorityDigestValueByteCountV1 == 320",
            "kAuthorityTextValueByteCountV1 == 65",
            "std::uint8_t header_reserved[7]",
            "std::uint8_t tail_reserved[7]",
        ),
    )
    directly_asserted_offsets = {
        "source_version",
        "disposition",
        "generation_index",
        "cell_index",
        "validity",
        "defined_mask",
        "status_codes",
        "integer_values",
        "float_values",
        "digest_values",
        "text_values",
        "tail_reserved",
    }
    for member in POD_MEMBERS:
        if member["name"] not in directly_asserted_offsets:
            continue
        token = (
            "offsetof(CompactSummaryGenerationCellAuthorityV1, "
            f"{member['name']}) == {member['offset']}"
        )
        if token not in source:
            raise RuntimeError(
                "missing required frozen constant: "
                f"{SOURCE_PROJECTION_HEADER.relative_to(ROOT)}:{token}"
            )

    boundary = _require_tokens(
        BOUNDARY_HEADER,
        (
            "enum HostCanonicalDigestSlotV1",
            "kRequiredHostCanonicalDigestMaskV1 == 0x7FU",
            "const std::uint8_t* completed_host_digest_values",
            "std::uint32_t provided_host_canonical_digest_mask",
        ),
    )
    for slot in HOST_CANONICAL_DIGEST_SLOTS:
        pattern = rf"{slot['boundary_enum']}\s*=\s*1U\s*<<\s*{slot['mask'].bit_length() - 1}"
        if re.search(pattern, boundary) is None:
            raise RuntimeError(
                "missing required frozen constant: "
                f"{BOUNDARY_HEADER.relative_to(ROOT)}:{slot['boundary_enum']}"
            )

    expected_counts = {
        "STATUS_FIELDS": 12,
        "INTEGER_FIELDS": 76,
        "FLOAT_FIELDS": 21,
        "DIGEST_FIELDS": 10,
        "TEXT_FIELDS": 1,
        "DEFINED_FIELDS": 32,
    }
    for name, expected in expected_counts.items():
        value = _required_constant(abi, name)
        if len(value) != expected:
            raise RuntimeError(
                f"frozen ABI constant drift: compact_summary_planes.{name} "
                f"has {len(value)} entries, expected {expected}"
            )
    if _required_constant(abi, "LOGICAL_BYTES_PER_GENERATION_CELL") != 1218:
        raise RuntimeError("frozen ABI logical row width is not 1218 bytes")


def _load_descriptor_plan() -> dict[str, Any]:
    if not DESCRIPTOR_PLAN_PATH.is_file():
        raise RuntimeError(
            "missing required repository anchor: "
            f"{DESCRIPTOR_PLAN_PATH.relative_to(ROOT)}"
        )
    plan = json.loads(DESCRIPTOR_PLAN_PATH.read_text(encoding="utf-8"))
    expected_schema = _required_constant(descriptor_plan, "PLAN_SCHEMA_VERSION")
    if plan.get("schema_version") != expected_schema:
        raise RuntimeError("checkpoint descriptor plan schema constant mismatch")
    descriptors = plan.get("descriptors")
    batches = plan.get("batches")
    if not isinstance(descriptors, list) or len(descriptors) != DESCRIPTOR_COUNT:
        raise RuntimeError("checkpoint descriptor plan must contain exactly 133 descriptors")
    if not isinstance(batches, list) or len(batches) != DESCRIPTOR_BATCH_COUNT:
        raise RuntimeError("checkpoint descriptor plan must contain exactly nine batches")
    indices = [row.get("descriptor_index") for row in descriptors]
    names = [row.get("tensor_name") for row in descriptors]
    if indices != list(range(DESCRIPTOR_COUNT)):
        raise RuntimeError("checkpoint descriptor indices are not exact and contiguous")
    if names != sorted(names) or len(set(names)) != DESCRIPTOR_COUNT:
        raise RuntimeError("checkpoint descriptor names are not unique lexicographic order")
    return plan


def _status_source(field: str) -> tuple[str, str]:
    if field == "world":
        return "static_metadata", "source.pre_turnover.world (Flat code kWorldFlat)"
    if field == "checkpoint_due":
        return "static_metadata", "source.schedule -> compact_summary_cpu._checkpoint_due"
    if field == "turnover.status":
        return "post_turnover_resident_state", "source.post_turnover.turnover.status"
    if field.startswith("rng."):
        leaf = field.removeprefix("rng.")
        return "composite_rng_metadata_retention", f"source.rng.{leaf}"
    if field.startswith("terminal.") or field == "summary_status":
        leaf = field.removeprefix("terminal.")
        path = "source.settled.summary_status" if field == "summary_status" else f"source.settled.terminal.{leaf}"
        return "settled_failure_status", path
    if field == "lineage_metrics.lineage_collapse_warning":
        return "lineage_producer_retention", "source.lineage.lineage_collapse_warning"
    raise RuntimeError(f"unclassified frozen status field: {field}")


_PRE_TURNOVER_INTEGER_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "alive",
        "births",
        "failed_births",
        "deaths",
        "generation_metrics.capacity",
        "generation_metrics.free_slots",
        "generation_metrics.birth_attempts",
        "generation_metrics.native_failed_birth_event_rows",
        "generation_metrics.injury_blocked_births",
        "generation_metrics.moved",
        "generation_metrics.gathered",
        "generation_metrics.movement_attempts",
        "generation_metrics.movement_blocked_occupied",
        "generation_metrics.movement_blocked_bounds",
        "generation_metrics.policy_loop_tick_calls",
        "generation_metrics.policy_refill_calls",
        "generation_metrics.policy_refill_rows",
        "generation_metrics.policy_cache_hit_calls",
        "generation_metrics.policy_cache_rows_requested",
        "generation_metrics.policy_cache_rows_hit",
        "generation_metrics.policy_cache_rows_missed",
        "generation_metrics.policy_cache_rows_refilled",
        "generation_metrics.policy_cache_full_refill_ticks",
        "generation_metrics.policy_cache_partial_refill_ticks",
    }
)


def _integer_source(field: str) -> tuple[str, str]:
    if field.startswith("patch_metrics."):
        return "patch_disabled_constant_undefined", f"Flat source.patch=null -> {field}=0"
    if field.startswith("behavior_metrics."):
        leaf = field.removeprefix("behavior_metrics.")
        return "behavior_producer_retention", f"source.behavior.{leaf}"
    if field.startswith("lineage_metrics."):
        leaf = field.removeprefix("lineage_metrics.")
        return "lineage_producer_retention", f"source.lineage.{leaf}"
    if field.startswith("terminal."):
        leaf = field.removeprefix("terminal.")
        return "settled_failure_status", f"source.settled.terminal.{leaf}"
    if field.startswith("identity."):
        leaf = field.removeprefix("identity.")
        return (
            "identity_state",
            "engine/perf/cuda_architecture_b_compact_summary_reductions_v1.cuh:"
            f"ReductionResultV1.{leaf}",
        )
    if field.startswith("events."):
        leaf = field.removeprefix("events.")
        member = {
            "generation_counts_by_type.birth_committed": "generation_counts_by_type[0]",
            "generation_counts_by_type.mutation": "generation_counts_by_type[1]",
            "generation_counts_by_type.birth_failed": "generation_counts_by_type[2]",
            "generation_counts_by_type.death_committed": "generation_counts_by_type[3]",
            "turnover_counts_by_type.birth_committed": "turnover_counts_by_type[0]",
            "turnover_counts_by_type.mutation": "turnover_counts_by_type[1]",
            "turnover_counts_by_type.birth_failed": "turnover_counts_by_type[2]",
            "turnover_counts_by_type.death_committed": "turnover_counts_by_type[3]",
        }.get(leaf, leaf)
        return (
            "event_state",
            "engine/perf/cuda_architecture_b_compact_summary_reductions_v1.cuh:"
            f"ReductionResultV1.{member}",
        )
    if field.startswith("deaths_by_class."):
        member = {
            "deaths_by_class.energy_depleted": "deaths_energy_depleted",
            "deaths_by_class.health_depleted": "deaths_health_depleted",
        }[field]
        return (
            "event_state",
            "engine/perf/cuda_architecture_b_compact_summary_reductions_v1.cuh:"
            f"ReductionResultV1.{member}",
        )
    if field == "turnover.count" or field.startswith("next_generation_start."):
        leaf = field.removeprefix("next_generation_start.")
        path = "source.post_turnover.turnover.count" if field == "turnover.count" else f"source.post_turnover.next_generation_start.{leaf}"
        return "post_turnover_resident_state", path
    if field == "rng.policy_mutation_tape_cursor":
        return "composite_rng_metadata_retention", "source.rng.policy_mutation_tape.cursor"
    if field.startswith("generation_start.") or field.startswith("generation_end."):
        return "static_metadata", f"source.pre_turnover.{field}"
    if field in _PRE_TURNOVER_INTEGER_FIELDS:
        return "pre_turnover_reporting_retention", f"source.pre_turnover.{field}"
    raise RuntimeError(f"unclassified frozen integer field: {field}")


def _float_source(field: str) -> tuple[str, str]:
    if field.startswith("patch_metrics."):
        return "patch_disabled_constant_undefined", f"Flat source.patch=null -> {field}=+0.0"
    if field.startswith("lineage_metrics."):
        leaf = field.removeprefix("lineage_metrics.")
        return "lineage_producer_retention", f"source.lineage.{leaf}"
    if field == "generation_metrics.peak_injury":
        return "pre_turnover_reporting_retention", "source.pre_turnover.generation_metrics.peak_injury"
    raise RuntimeError(f"unclassified frozen float field: {field}")


def _digest_source(field: str) -> tuple[str, str, dict[str, Any]]:
    for slot in HOST_CANONICAL_DIGEST_SLOTS:
        if slot["field"] == field:
            return (
                str(slot["source_class"]),
                "engine/perf/cuda_architecture_b_compact_summary_boundary_v1.cuh:"
                f"RetainedHostCanonicalizationOutputsV1.completed_host_digest_values[{slot['boundary_enum']}]",
                {
                    "host_canonical": True,
                    "boundary_enum": slot["boundary_enum"],
                    "boundary_mask": slot["mask"],
                    "device_retained": False,
                },
            )
    for retained in DEVICE_RETAINED_DIGEST_FIELDS:
        if retained["field"] == field:
            return (
                retained["source_class"],
                retained["physical_source"],
                {
                    "host_canonical": False,
                    "boundary_enum": None,
                    "boundary_mask": 0,
                    "device_retained": True,
                },
            )
    if field == "terminal.recoverable_error_digest_sha256":
        return (
            "settled_failure_status",
            "engine/native_runtime/compact_summary_retained_digest_finalizer.py:authority.terminal -> recoverable_error_digest_sha256",
            {
                "host_canonical": False,
                "boundary_enum": None,
                "boundary_mask": 0,
                "device_retained": False,
            },
        )
    raise RuntimeError(f"unclassified frozen digest field: {field}")


def _text_source(field: str) -> tuple[str, str]:
    if field == "lineage_metrics.dominant_lineage_id":
        return "lineage_producer_retention", "source.lineage.dominant_lineage_id"
    raise RuntimeError(f"unclassified frozen text field: {field}")


def _field_sources() -> list[dict[str, Any]]:
    registries: tuple[tuple[str, Sequence[str]], ...] = (
        ("status_codes", abi.STATUS_FIELDS),
        ("integer_values", abi.INTEGER_FIELDS),
        ("float_values", abi.FLOAT_FIELDS),
        ("digest_values", abi.DIGEST_FIELDS),
        ("text_values", abi.TEXT_FIELDS),
    )
    result: list[dict[str, Any]] = []
    for plane, fields in registries:
        for index, field in enumerate(fields):
            extras: dict[str, Any] = {}
            if plane == "status_codes":
                source_class, physical_source = _status_source(field)
            elif plane == "integer_values":
                source_class, physical_source = _integer_source(field)
            elif plane == "float_values":
                source_class, physical_source = _float_source(field)
            elif plane == "digest_values":
                source_class, physical_source, extras = _digest_source(field)
            else:
                source_class, physical_source = _text_source(field)
            if source_class not in SOURCE_CLASSES or not physical_source:
                raise RuntimeError(f"invalid source assignment for {plane}.{field}")
            result.append(
                {
                    "plane": plane,
                    "index": index,
                    "field": field,
                    "source_class": source_class,
                    "physical_source": physical_source,
                    "abi_anchor": (
                        "engine/native_runtime/compact_summary_planes.py:"
                        f"{plane.upper().removesuffix('_VALUES').removesuffix('_CODES')}_FIELD_INDEX[{field!r}]"
                    ),
                    **extras,
                }
            )
    identities = [(row["plane"], row["field"]) for row in result]
    if len(identities) != len(set(identities)):
        raise RuntimeError("duplicate compact authority field source assignment")
    return result


def _defined_mask_source(field: str, bit: int) -> tuple[str, str]:
    if field.startswith("patch_metrics"):
        return (
            "patch_disabled_constant_undefined",
            f"Flat source.patch=null -> defined_mask bit {bit} ({field})=0",
        )
    if field.startswith("identity."):
        leaf = field.removeprefix("identity.")
        return "identity_state", f"source.post_turnover.identity.{leaf} is not null"
    if field.startswith("terminal."):
        leaf = field.removeprefix("terminal.")
        return "settled_failure_status", f"source.settled.terminal.{leaf} is not null"
    if field.startswith("lineage_metrics."):
        leaf = field.removeprefix("lineage_metrics.")
        return "lineage_producer_retention", f"source.lineage.{leaf} is not null"
    raise RuntimeError(f"unclassified frozen defined-mask field: {field}")


def _defined_mask_sources() -> list[dict[str, Any]]:
    rows = []
    for field, bit in abi.DEFINED_FIELD_BITS.items():
        source_class, physical_source = _defined_mask_source(field, bit)
        rows.append(
            {
                "bit": bit,
                "field": field,
                "source_class": source_class,
                "physical_source": physical_source,
                "abi_anchor": (
                    "engine/native_runtime/compact_summary_planes.py:"
                    f"DEFINED_FIELD_BITS[{field!r}]"
                ),
            }
        )
    if [row["bit"] for row in rows] != list(range(len(abi.DEFINED_FIELDS))):
        raise RuntimeError("defined-mask bit registry is not exact and contiguous")
    if len({row["field"] for row in rows}) != len(rows):
        raise RuntimeError("duplicate defined-mask source assignment")
    return rows


def _callback_sequence() -> list[dict[str, Any]]:
    return [
        {
            "order": 1,
            "operation": "bounded_d2h",
            "owner": "native_cpp",
            "detail": "enqueue only retained reporting rows, descriptor digests/status, identity/event digest authority, and bounded non-tensor metadata",
        },
        {
            "order": 2,
            "operation": "stream_host_callback",
            "owner": "native_cpp",
            "count": 1,
            "api": "cudaLaunchHostFunc",
            "detail": "after D2H, finalize canonical envelopes into preallocated pinned authority staging; callback invokes no CUDA API",
        },
        {
            "order": 3,
            "operation": "bounded_h2d_authority_completion",
            "owner": "native_cpp",
            "detail": "already-enqueued stream-ordered H2D copies completed authority bytes before source projection",
        },
        {
            "order": 4,
            "operation": "source_projection",
            "owner": "native_cpp_cuda",
            "detail": "project the completed generation-major POD row and write validity last",
        },
        {
            "order": 5,
            "operation": "final_synchronization",
            "owner": "native_cpp_binding",
            "count": 1,
            "detail": "the simulation call performs its one final synchronization only after projection",
        },
    ]


def _finalizer_constants() -> dict[str, Any]:
    names = (
        "_TOP_LEVEL_FIELDS",
        "_CHECKPOINT_FIELDS",
        "_IDENTITY_FIELDS",
        "_POLICY_FIELDS",
        "_REPORTING_FIELDS",
        "_LIFECYCLE_FIELDS",
        "_TERMINAL_FIELDS",
        "_SOURCE_TENSOR_DESCRIPTOR_FIELDS",
    )
    return {
        "FINALIZER_SOURCE_SCHEMA_VERSION": _required_constant(
            retained_finalizer, "FINALIZER_SOURCE_SCHEMA_VERSION"
        ),
        **{name: list(_required_constant(retained_finalizer, name)) for name in names},
    }


def _checkpoint_constants() -> dict[str, Any]:
    tuple_names = (
        "SOURCE_PROJECTION_REQUIRED_TOP_LEVEL_FIELDS",
        "SOURCE_AUTHORITY_CURSOR_METADATA_FIELDS",
        "SOURCE_RNG_REQUIRED_METADATA_FIELDS",
        "SOURCE_RNG_OPTIONAL_METADATA_FIELDS",
        "SOURCE_RNG_TENSOR_FIELDS",
    )
    scalar_names = (
        "SCHEMA_VERSION",
        "SOURCE_PROJECTION_SCHEMA_VERSION",
        "SOURCE_RNG_SCHEMA_VERSION",
        "SOURCE_AUTHORITY_SCHEMA_VERSION",
    )
    return {
        **{name: _required_constant(checkpoint_digests, name) for name in scalar_names},
        **{name: list(_required_constant(checkpoint_digests, name)) for name in tuple_names},
        "DigestLayer": [str(layer) for layer in checkpoint_digests.DigestLayer],
    }


def _descriptor_summary(plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "descriptor_index": row["descriptor_index"],
            "tensor_name": row["tensor_name"],
            "physical_source_name": row["physical_source_name"],
            "dtype": row["dtype"],
            "raw_shape_symbols": row["raw_shape_symbols"],
            "canonical_shape_symbols": row["canonical_shape_symbols"],
            "hash_mode": row["hash_mode"],
            "raw_output_index": row["raw_output_index"],
            "canonical_output_index": row["canonical_output_index"],
            "digest_inclusion": row["digest_inclusion"],
        }
        for row in plan["descriptors"]
    ]


def build_contracts() -> dict[str, dict[str, Any]]:
    """Build all machine-readable contracts after fail-closed anchor checks."""

    _verify_frozen_anchors()
    plan = _load_descriptor_plan()
    field_sources = _field_sources()
    mask_sources = _defined_mask_sources()
    finalizer_constants = _finalizer_constants()
    digest_constants = _checkpoint_constants()
    callback_sequence = _callback_sequence()

    field_counts = {
        "status": len(abi.STATUS_FIELDS),
        "integer": len(abi.INTEGER_FIELDS),
        "float": len(abi.FLOAT_FIELDS),
        "digest": len(abi.DIGEST_FIELDS),
        "text": len(abi.TEXT_FIELDS),
        "defined_mask_bits": len(abi.DEFINED_FIELDS),
        "value_fields": len(field_sources),
        "mapped_subjects_total": len(field_sources) + len(mask_sources),
    }
    registries = {
        "status_fields": list(abi.STATUS_FIELDS),
        "integer_fields": list(abi.INTEGER_FIELDS),
        "float_fields": list(abi.FLOAT_FIELDS),
        "digest_fields": list(abi.DIGEST_FIELDS),
        "text_fields": list(abi.TEXT_FIELDS),
        "defined_mask_fields": list(abi.DEFINED_FIELDS),
    }
    claims = {
        "fixed_trivially_copyable_native_structure": True,
        "standard_layout": True,
        "dynamic_allocation": False,
        "pointer_members": False,
        "all_members_initialized_before_projection": True,
        "generation_major_then_cell_major": True,
        "python_generation_ownership": False,
        "python_generation_iteration": False,
        "intermediate_host_synchronization": False,
        "final_synchronization_count": 1,
        "resident_generation_summaries_supported": False,
        "eiel_compact_generation_summary_v1_supported": False,
        "capability_promoted": False,
        "full_state_host_dependency": False,
    }
    pod = {
        "schema_version": POD_SCHEMA_VERSION,
        "wire_schema_version": abi.WIRE_SCHEMA_VERSION,
        "native_abi_version": abi.ABI_VERSION,
        "source_authority_schema_version": (
            "eiel_cuda_architecture_b_compact_summary_source_authority_v1"
        ),
        "status": "phase0_live_authority_contract_frozen_capability_false",
        "artifact_set": list(OUTPUT_FILENAMES),
        "layout": {
            "cpp_type": "CompactSummaryGenerationCellAuthorityV1",
            "byte_width": POD_BYTE_WIDTH,
            "alignment": POD_ALIGNMENT,
            "source_plane_logical_bytes": abi.LOGICAL_BYTES_PER_GENERATION_CELL,
            "members": [dict(member) for member in POD_MEMBERS],
            "native_anchor": str(SOURCE_PROJECTION_HEADER.relative_to(ROOT)),
        },
        "projection_status": {
            "cpp_type": "int32_t",
            "bytes_per_generation_cell": PROJECTION_STATUS_BYTES,
            "separate_preallocated_plane": True,
            "total_row_accounting_bytes": ROW_ACCOUNTING_BYTES,
        },
        "shape": ["generation_count", "cell_count"],
        "ordering": "generation-major then cell-major",
        "field_counts": field_counts,
        "registries": registries,
        "source_classes": list(SOURCE_CLASSES),
        "source_map": "compact_live_authority_source_map_v1.json",
        "generation_metadata_retention": "compact_generation_metadata_retention_v1.json",
        "rng_outer_preimage": "compact_rng_outer_preimage_v1.json",
        "claims": claims,
    }
    source_map = {
        "schema_version": SOURCE_MAP_SCHEMA_VERSION,
        "pod_schema_version": POD_SCHEMA_VERSION,
        "wire_schema_version": abi.WIRE_SCHEMA_VERSION,
        "status": "complete_exact_single_source_map",
        "source_classes": list(SOURCE_CLASSES),
        "field_sources": field_sources,
        "defined_mask_sources": mask_sources,
        "host_canonical_digest_fields": [dict(row) for row in HOST_CANONICAL_DIGEST_SLOTS],
        "host_canonical_digest_mask": HOST_CANONICAL_DIGEST_MASK,
        "device_retained_digest_fields": [dict(row) for row in DEVICE_RETAINED_DIGEST_FIELDS],
        "coverage": {
            **field_counts,
            "field_duplicates": 0,
            "defined_mask_duplicates": 0,
            "unmapped_fields": [],
            "unmapped_defined_mask_bits": [],
            "competing_authorities": [],
        },
        "rules": {
            "exactly_one_source_class_per_field": True,
            "exactly_one_physical_source_per_field": True,
            "pre_turnover_reconstructed_from_post_turnover": False,
            "null_replaced_by_implicit_default": False,
            "candidate_derived_cpu_authority": False,
            "full_state_host_dependency": False,
        },
    }

    descriptors = _descriptor_summary(plan)
    projection_fields = list(
        _required_constant(checkpoint_digests, "SOURCE_PROJECTION_REQUIRED_TOP_LEVEL_FIELDS")
    )
    generation_metadata = {
        "schema_version": GENERATION_METADATA_SCHEMA_VERSION,
        "status": "phase0_bounded_retention_contract_frozen",
        "source_map_schema_version": SOURCE_MAP_SCHEMA_VERSION,
        "accepted_constants": {
            "compact_summary_retained_digest_finalizer.py": finalizer_constants,
            "checkpoint_digests.py": digest_constants,
        },
        "checkpoint_tensor_descriptors": {
            "kind": "tensor_descriptor_digest_retention",
            "plan_schema_version": plan["schema_version"],
            "descriptor_schema_version": descriptor_plan.DESCRIPTOR_SCHEMA_VERSION,
            "descriptor_count": DESCRIPTOR_COUNT,
            "batch_count": DESCRIPTOR_BATCH_COUNT,
            "ordering": plan["ordering"],
            "raw_digest_bytes_per_generation_cell": DESCRIPTOR_COUNT * 32,
            "canonical_digest_bytes_per_generation_cell": DESCRIPTOR_COUNT * 32,
            "validity_entries_per_generation_cell": DESCRIPTOR_COUNT,
            "status_entries_per_generation_cell": DESCRIPTOR_COUNT,
            "descriptor_fields": list(finalizer_constants["_SOURCE_TENSOR_DESCRIPTOR_FIELDS"]),
            "descriptors": descriptors,
            "repository_anchor": str(DESCRIPTOR_PLAN_PATH.relative_to(ROOT)),
        },
        "bounded_non_tensor_metadata": {
            "kind": "bounded_non_tensor_metadata_retention",
            "finalizer_top_level_fields": finalizer_constants["_TOP_LEVEL_FIELDS"],
            "checkpoint_fields": finalizer_constants["_CHECKPOINT_FIELDS"],
            "projection_fields_excluding_tensors": [
                field for field in projection_fields if field != "tensors"
            ],
            "authority_cursor_metadata_fields": digest_constants[
                "SOURCE_AUTHORITY_CURSOR_METADATA_FIELDS"
            ],
            "required_checkpoint_metadata_fields": [
                "world",
                "alive_count",
                *digest_constants["SOURCE_AUTHORITY_CURSOR_METADATA_FIELDS"],
                *digest_constants["SOURCE_RNG_REQUIRED_METADATA_FIELDS"],
            ],
            "identity_fields": finalizer_constants["_IDENTITY_FIELDS"],
            "policy_mutation_tape_fields": finalizer_constants["_POLICY_FIELDS"],
            "reporting_fields": finalizer_constants["_REPORTING_FIELDS"],
            "lifecycle_fields": finalizer_constants["_LIFECYCLE_FIELDS"],
            "terminal_fields": finalizer_constants["_TERMINAL_FIELDS"],
            "full_tensor_payloads": False,
            "full_json_strings_inside_generation_loop": False,
            "generic_python_dict_per_generation": False,
        },
        "retention_shape": ["generation_count", "cell_count", "bounded_fields"],
        "ordering": "generation-major then cell-major",
        "host_completion_sequence": callback_sequence,
        "claims": {
            "tensor_descriptors_distinct_from_non_tensor_metadata": True,
            "bounded_d2h_only": True,
            "bounded_h2d_only": True,
            "native_cpp_stream_callback_count": 1,
            "python_callback_owner": False,
            "python_generation_ownership": False,
            "intermediate_synchronizations": 0,
            "final_synchronizations": 1,
            "capability_promoted": False,
        },
    }

    rng_metadata = {
        "schema_version": RNG_OUTER_PREIMAGE_SCHEMA_VERSION,
        "status": "phase0_composite_rng_outer_preimage_contract_frozen",
        "accepted_constants": {
            "compact_summary_retained_digest_finalizer.py": {
                "FINALIZER_SOURCE_SCHEMA_VERSION": finalizer_constants[
                    "FINALIZER_SOURCE_SCHEMA_VERSION"
                ],
                "_CHECKPOINT_FIELDS": finalizer_constants["_CHECKPOINT_FIELDS"],
                "_POLICY_FIELDS": finalizer_constants["_POLICY_FIELDS"],
                "_SOURCE_TENSOR_DESCRIPTOR_FIELDS": finalizer_constants[
                    "_SOURCE_TENSOR_DESCRIPTOR_FIELDS"
                ],
            },
            "checkpoint_digests.py": {
                "SOURCE_RNG_SCHEMA_VERSION": digest_constants["SOURCE_RNG_SCHEMA_VERSION"],
                "SOURCE_RNG_REQUIRED_METADATA_FIELDS": digest_constants[
                    "SOURCE_RNG_REQUIRED_METADATA_FIELDS"
                ],
                "SOURCE_RNG_OPTIONAL_METADATA_FIELDS": digest_constants[
                    "SOURCE_RNG_OPTIONAL_METADATA_FIELDS"
                ],
                "SOURCE_RNG_TENSOR_FIELDS": digest_constants["SOURCE_RNG_TENSOR_FIELDS"],
            },
        },
        "composite_rng_authority": {
            "producer": "engine.native_runtime.checkpoint_digests._source_projection_rng_manifest",
            "schema_version": digest_constants["SOURCE_RNG_SCHEMA_VERSION"],
            "required_non_tensor_metadata_fields": digest_constants[
                "SOURCE_RNG_REQUIRED_METADATA_FIELDS"
            ],
            "optional_non_tensor_metadata_fields": digest_constants[
                "SOURCE_RNG_OPTIONAL_METADATA_FIELDS"
            ],
            "tensor_descriptor_fields": digest_constants["SOURCE_RNG_TENSOR_FIELDS"],
            "tensor_descriptor_record_fields": finalizer_constants[
                "_SOURCE_TENSOR_DESCRIPTOR_FIELDS"
            ],
            "stream_order_and_names_preserved_inside": (
                "native_runtime_rng_manager_stream_indices"
            ),
            "serialized_state_metadata_preserved_verbatim": [
                "native_runtime_numpy_env_state",
                "native_runtime_numpy_runner_state",
                "native_runtime_numpy_policy_state",
                "native_runtime_numpy_population_state",
                "native_runtime_python_random_state",
                "native_runtime_rng_state",
                "native_runtime_numpy_birth_spawn_env_state",
            ],
            "bit_generator_names_preserved_inside_serialized_state": True,
            "separate_pcg64_state_authority": False,
            "separate_pcg64_cursor_authority": False,
            "authority_complete_required": True,
            "export_non_consuming_required": True,
        },
        "policy_mutation_tape_outer_preimage": {
            "fields": finalizer_constants["_POLICY_FIELDS"],
            "tensor_descriptor_count": int(
                plan["counts"]["policy_mutation_tape_fields"]
            ),
            "ordering": "strict lexicographic tensor_name",
            "source": "canonical consumed-prefix descriptor digests plus cursor and consumed_shapes",
        },
        "host_canonical_digest_fields": [
            dict(row)
            for row in HOST_CANONICAL_DIGEST_SLOTS
            if str(row["field"]).startswith("rng.")
        ],
        "host_completion_sequence": callback_sequence,
        "claims": {
            "composite_checkpoint_rng_authority_reused": True,
            "bounded_non_tensor_metadata": True,
            "tensor_descriptors_separate": True,
            "native_cpp_stream_callback_count": 1,
            "python_generation_ownership": False,
            "summary_induced_rng_consumption": False,
            "intermediate_synchronizations": 0,
            "final_synchronizations": 1,
            "capability_promoted": False,
        },
    }
    return {
        OUTPUT_FILENAMES[1]: pod,
        OUTPUT_FILENAMES[2]: source_map,
        OUTPUT_FILENAMES[3]: generation_metadata,
        OUTPUT_FILENAMES[4]: rng_metadata,
    }


def _markdown(pod: Mapping[str, Any], source_map: Mapping[str, Any]) -> str:
    lines = [
        "# Compact Live Authority POD v1",
        "",
        f"Schema: `{pod['schema_version']}`",
        "",
        "Status: Phase 0 contract frozen; compact capability remains false.",
        "",
        "## Native layout",
        "",
        f"- POD: `{pod['layout']['cpp_type']}`, {POD_BYTE_WIDTH} bytes, alignment {POD_ALIGNMENT}.",
        f"- Projection status: separate `int32_t`, {PROJECTION_STATUS_BYTES} bytes.",
        f"- Per generation/cell accounting: {ROW_ACCOUNTING_BYTES} bytes.",
        "- Shape/order: `[generation_count, cell_count]`, generation-major then cell-major.",
        "- Storage is preallocated; the POD has no pointers and no dynamic allocation.",
        "- Native C++ owns generation assembly and the one stream callback; Python owns no generation iteration.",
        "- `resident_generation_summaries_supported=false` and `eiel_compact_generation_summary_v1_supported=false`.",
        "",
        "| member | offset | bytes | type |",
        "|---|---:|---:|---|",
    ]
    lines.extend(
        f"| `{row['name']}` | {row['offset']} | {row['bytes']} | `{row['type']}` |"
        for row in pod["layout"]["members"]
    )
    lines.extend(
        [
            "",
            "## Digest completion",
            "",
            "Seven digest fields use the boundary's host-canonical mask `0x7f`. Active-event-prefix and active-ID authorities are retained separately on device. Recoverable-error digest authority belongs only to settled failure state.",
            "",
            "The single stream sequence is bounded D2H, one native C++ `cudaLaunchHostFunc`, pre-enqueued bounded H2D authority completion, source projection, then one final synchronization.",
            "",
            "## Exact source map",
            "",
            "Every value field and defined-mask bit below has exactly one source class and one physical source.",
            "",
            "| plane | index | field | source class | physical source |",
            "|---|---:|---|---|---|",
        ]
    )
    for row in source_map["field_sources"]:
        physical = str(row["physical_source"]).replace("|", "\\|")
        lines.append(
            f"| `{row['plane']}` | {row['index']} | `{row['field']}` | "
            f"`{row['source_class']}` | `{physical}` |"
        )
    lines.extend(
        [
            "",
            "## Defined-mask sources",
            "",
            "| bit | subject | source class | physical source |",
            "|---:|---|---|---|",
        ]
    )
    for row in source_map["defined_mask_sources"]:
        physical = str(row["physical_source"]).replace("|", "\\|")
        lines.append(
            f"| {row['bit']} | `{row['field']}` | `{row['source_class']}` | "
            f"`{physical}` |"
        )
    lines.extend(
        [
            "",
            "## Scope",
            "",
            "This is a deterministic contract artifact only. It does not promote capability, run a simulation, mutate simulation state, or claim Flat g2 row parity.",
            "",
        ]
    )
    return "\n".join(lines)


def render_artifacts() -> dict[str, str]:
    """Return the exact five filenames and deterministic UTF-8 contents."""

    contracts = build_contracts()
    rendered = {
        name: json.dumps(payload, indent=2, sort_keys=True) + "\n"
        for name, payload in contracts.items()
    }
    rendered[OUTPUT_FILENAMES[0]] = _markdown(
        contracts[OUTPUT_FILENAMES[1]], contracts[OUTPUT_FILENAMES[2]]
    )
    return {name: rendered[name] for name in OUTPUT_FILENAMES}


def generate_contract(output_dir: Path) -> tuple[Path, ...]:
    """Write the five deterministic contract artifacts into ``output_dir``."""

    output_dir.mkdir(parents=True, exist_ok=True)
    if not output_dir.is_dir():
        raise NotADirectoryError(output_dir)
    rendered = render_artifacts()
    paths = []
    for name in OUTPUT_FILENAMES:
        path = output_dir / name
        path.write_text(rendered[name], encoding="utf-8")
        paths.append(path)
    return tuple(paths)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    paths = generate_contract(args.output_dir)
    print(
        json.dumps(
            {
                "ok": True,
                "output_dir": str(args.output_dir.resolve()),
                "artifacts": [path.name for path in paths],
                "capability_promoted": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
