"""Frozen host-side contract for ``EIEL_SNAPSHOT_CAPSULE_V1``.

This module is control-plane code.  It canonicalizes and validates a snapshot
plan once before handing the compact, immutable description to the native
extension.  Snapshot payload materialization remains entirely native.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any, Final


SNAPSHOT_PLAN_SCHEMA_V1: Final[str] = "eiel_native_snapshot_plan_v1"
SNAPSHOT_CAPSULE_ABI_V1: Final[str] = "eiel_native_snapshot_capsule_abi_v1"
SNAPSHOT_EXTENSION_VERSION_V1: Final[str] = (
    "eiel_native_snapshot_extension_v1"
)
SNAPSHOT_CAPSULE_MAGIC_V1: Final[bytes] = b"EIELSNAPCAPSULE\0"
SNAPSHOT_CAPSULE_VERSION_V1: Final[int] = 1
SNAPSHOT_CAPSULE_HEADER_BYTES_V1: Final[int] = 384
SNAPSHOT_CAPSULE_DESCRIPTOR_BYTES_V1: Final[int] = 256
SNAPSHOT_CAPSULE_ALIGNMENT_V1: Final[int] = 64
SNAPSHOT_CAPSULE_ENDIAN_MARKER_V1: Final[int] = 0x01020304
SNAPSHOT_CAPSULE_DIGEST_BYTES_V1: Final[int] = 32
SNAPSHOT_CAPSULE_MAX_RANK_V1: Final[int] = 8
SNAPSHOT_PROJECTION_BACKEND_HOST_CONTIGUOUS_SYNC_V1: Final[int] = 1
SNAPSHOT_COMPACT_STAGE_PROFILE_V1: Final[str] = (
    "ACTIVE_COLUMNAR_STAGE_V1"
)
SNAPSHOT_COMPACT_STAGE_ENCODING_POLICY_V1: Final[str] = (
    "EXACT_CAPSULE_V1_EXCLUSIVE_FSYNC_CLOSE"
)
SNAPSHOT_COMPATIBILITY_STAGE_ENCODING_POLICY_V1: Final[str] = (
    "EXACT_V1_SINGLE_MEMBER_SEQUENTIAL"
)
MAX_SIGNED_INT64_V1: Final[int] = (1 << 63) - 1
MAX_UNSIGNED_INT64_V1: Final[int] = (1 << 64) - 1

SNAPSHOT_SECTION_KINDS_V1: Final[dict[str, int]] = {
    "canonical_plan_json": 1,
    "state_field": 2,
    "metrics_field": 3,
    "world_offsets": 4,
    "active_slots": 5,
    "identity": 6,
    "opaque_typed": 255,
}
SNAPSHOT_DTYPES_V1: Final[dict[str, tuple[int, int]]] = {
    "u8": (1, 1),
    "i8": (2, 1),
    "u16": (3, 2),
    "i16": (4, 2),
    "u32": (5, 4),
    "i32": (6, 4),
    "u64": (7, 8),
    "i64": (8, 8),
    "f32": (9, 4),
    "f64": (10, 8),
    "bool": (11, 1),
}

REQUIRED_NATIVE_EXPORTS_V1: Final[tuple[str, ...]] = (
    "abi_version",
    "extension_version",
    "capsule_format_manifest_v1",
    "zlib_compile_version",
    "zlib_runtime_version",
    "supports_cuda_projection_v1",
    "compile_snapshot_plan_v1",
    "inspect_snapshot_capsule_v1",
    "extract_snapshot_capsule_section_v1",
)


class SnapshotPlanContractError(ValueError):
    """A plan cannot safely define one native snapshot capsule."""


def canonical_plan_bytes_v1(value: Mapping[str, Any]) -> bytes:
    """Return the exact UTF-8 plan representation bound into the capsule."""

    if not isinstance(value, Mapping):
        raise SnapshotPlanContractError("snapshot plan must be a mapping")
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise SnapshotPlanContractError(
            f"snapshot plan is not canonical-JSON encodable: {exc}"
        ) from exc
    if not encoded:
        raise SnapshotPlanContractError("snapshot plan is empty")
    return encoded


def _positive_shape(value: object, *, label: str) -> tuple[int, ...]:
    if (
        isinstance(value, (str, bytes))
        or not isinstance(value, Sequence)
        or not 1 <= len(value) <= SNAPSHOT_CAPSULE_MAX_RANK_V1
    ):
        raise SnapshotPlanContractError(
            f"{label} must have between one and eight dimensions"
        )
    shape: list[int] = []
    elements = 1
    for raw in value:
        if type(raw) is not int or raw <= 0:
            raise SnapshotPlanContractError(
                f"{label} dimensions must be positive integers"
            )
        if elements > ((1 << 63) - 1) // raw:
            raise SnapshotPlanContractError(f"{label} element count overflows int64")
        elements *= raw
        shape.append(raw)
    return tuple(shape)


def _checked_u64_multiply(left: int, right: int, *, label: str) -> int:
    if left < 0 or right < 0 or (left and right > MAX_UNSIGNED_INT64_V1 // left):
        raise SnapshotPlanContractError(f"{label} overflows uint64")
    return left * right


def _hex_digest(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise SnapshotPlanContractError(f"{label} must be lowercase SHA-256 hex")
    return value


def _input_contract(value: object, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise SnapshotPlanContractError(f"{label} must be a mapping")
    dtype = value.get("dtype")
    if dtype not in SNAPSHOT_DTYPES_V1:
        raise SnapshotPlanContractError(f"{label} dtype is unsupported")
    shape = _positive_shape(value.get("shape"), label=f"{label}.shape")
    width = SNAPSHOT_DTYPES_V1[str(dtype)][1]
    elements = 1
    for dimension in shape:
        elements *= dimension
    byte_count = _checked_u64_multiply(
        elements, width, label=f"{label} byte extent"
    )
    return {
        "dtype": str(dtype),
        "shape": list(shape),
        "bytes": byte_count,
    }


def normalize_snapshot_plan_v1(value: Mapping[str, Any]) -> dict[str, Any]:
    """Project one already authoritative exact-V1 plan for native binding."""

    if not isinstance(value, Mapping):
        raise SnapshotPlanContractError("snapshot plan must be a mapping")
    # Keep the duplicate Python/native ABI mirror fail-closed: this helper is
    # public for build tooling, so it must not become a route around the
    # authoritative semantic validator used by the runtime wrapper.
    from engine.native_runtime.cuda_snapshot_plan_v1 import (  # noqa: PLC0415
        CudaSnapshotPlanError,
        validate_cuda_snapshot_plan_v1,
    )

    try:
        plan = validate_cuda_snapshot_plan_v1(dict(value)).to_mapping()
    except CudaSnapshotPlanError as exc:
        raise SnapshotPlanContractError(
            f"snapshot semantic plan is not authoritative V1: {exc}"
        ) from exc
    if plan.get("schema_version") != SNAPSHOT_PLAN_SCHEMA_V1:
        raise SnapshotPlanContractError("snapshot plan schema version differs")
    for key in ("snapshot_schema_id", "layout_id"):
        item = plan.get(key)
        if not isinstance(item, str) or not item or "\x00" in item:
            raise SnapshotPlanContractError(f"snapshot plan {key} is invalid")
    plan["layout_sha256"] = _hex_digest(
        plan.get("layout_sha256"), label="snapshot layout identity"
    )
    plan["native_world_control_set_sha256"] = _hex_digest(
        plan.get("native_world_control_set_sha256"),
        label="snapshot control identity",
    )
    world_count = plan.get("world_count")
    if type(world_count) is not int or world_count <= 0:
        raise SnapshotPlanContractError("snapshot world_count must be positive")
    state_input = _input_contract(plan.get("state_input"), label="state_input")
    metrics_input = _input_contract(
        plan.get("metrics_input"), label="metrics_input"
    )
    for label, contract in (
        ("state_input", state_input),
        ("metrics_input", metrics_input),
    ):
        if contract["shape"][0] != world_count:
            raise SnapshotPlanContractError(
                f"{label} leading dimension must equal world_count"
            )
        if contract["bytes"] % world_count:
            raise SnapshotPlanContractError(
                f"{label} bytes are not divisible by world_count"
            )
        contract["world_stride_bytes"] = contract["bytes"] // world_count
    raw_boundaries = plan.get("snapshot_boundaries")
    if (
        isinstance(raw_boundaries, (str, bytes))
        or not isinstance(raw_boundaries, Sequence)
        or not raw_boundaries
    ):
        raise SnapshotPlanContractError(
            "snapshot_boundaries must be a nonempty ordered array"
        )
    boundaries: list[dict[str, Any]] = []
    previous_tick = -1
    for index, raw_boundary in enumerate(raw_boundaries):
        if not isinstance(raw_boundary, Mapping):
            raise SnapshotPlanContractError(
                f"snapshot boundary {index} must be a mapping"
            )
        tick = raw_boundary.get("absolute_tick")
        raw_indices = raw_boundary.get("world_indices")
        if (
            type(tick) is not int
            or tick < 0
            or tick > MAX_SIGNED_INT64_V1
            or tick <= previous_tick
        ):
            raise SnapshotPlanContractError(
                "snapshot boundary ticks must be nonnegative and strictly ascending"
            )
        if (
            isinstance(raw_indices, (str, bytes))
            or not isinstance(raw_indices, Sequence)
            or not raw_indices
        ):
            raise SnapshotPlanContractError(
                f"snapshot boundary {tick} world_indices must be nonempty"
            )
        indices: list[int] = []
        previous_world = -1
        for raw_world in raw_indices:
            if (
                type(raw_world) is not int
                or not 0 <= raw_world < world_count
                or raw_world <= previous_world
            ):
                raise SnapshotPlanContractError(
                    f"snapshot boundary {tick} world indices must be unique, "
                    "in range, and in plan order"
                )
            indices.append(raw_world)
            previous_world = raw_world
        boundaries.append({"absolute_tick": tick, "world_indices": indices})
        previous_tick = tick
    raw_sections = plan.get("sections")
    if (
        isinstance(raw_sections, (str, bytes))
        or not isinstance(raw_sections, Sequence)
        or not raw_sections
    ):
        raise SnapshotPlanContractError("snapshot sections must be nonempty")
    sections: list[dict[str, Any]] = []
    names: set[str] = {"plan_identity", "selected_world_indices"}
    for index, raw in enumerate(raw_sections):
        if not isinstance(raw, Mapping):
            raise SnapshotPlanContractError(f"snapshot section {index} is invalid")
        name = raw.get("name")
        if (
            not isinstance(name, str)
            or not name
            or "\x00" in name
            or len(name.encode("utf-8")) > 4096
            or name in names
        ):
            raise SnapshotPlanContractError(
                f"snapshot section {index} name is invalid or duplicated"
            )
        names.add(name)
        source = raw.get("source")
        if source not in {"state", "metrics"}:
            raise SnapshotPlanContractError(
                f"snapshot section {name} source must be state or metrics"
            )
        kind = raw.get("kind")
        if isinstance(kind, str):
            kind_id = SNAPSHOT_SECTION_KINDS_V1.get(kind)
        elif type(kind) is int and kind in SNAPSHOT_SECTION_KINDS_V1.values():
            kind_id = kind
        else:
            kind_id = None
        if kind_id in {None, SNAPSHOT_SECTION_KINDS_V1["canonical_plan_json"]}:
            raise SnapshotPlanContractError(
                f"snapshot section {name} kind is unsupported"
            )
        dtype = raw.get("dtype")
        if dtype not in SNAPSHOT_DTYPES_V1:
            raise SnapshotPlanContractError(
                f"snapshot section {name} dtype is unsupported"
            )
        if raw.get("world_major", True) is not True:
            raise SnapshotPlanContractError(
                f"snapshot section {name} must be world-major in capsule V1"
            )
        raw_element_shape = raw.get("element_shape")
        if raw_element_shape is None:
            full_shape = _positive_shape(
                raw.get("shape"), label=f"section {name}.shape"
            )
            if len(full_shape) < 2 or full_shape[0] != world_count:
                raise SnapshotPlanContractError(
                    f"snapshot section {name} shape must begin with world_count"
                )
            element_shape = full_shape[1:]
        else:
            element_shape = _positive_shape(
                raw_element_shape, label=f"section {name}.element_shape"
            )
            full_shape = (world_count, *element_shape)
            supplied_shape = raw.get("shape")
            if supplied_shape is not None and _positive_shape(
                supplied_shape, label=f"section {name}.shape"
            ) != full_shape:
                raise SnapshotPlanContractError(
                    f"snapshot section {name} full and element shapes differ"
                )
        if len(element_shape) >= SNAPSHOT_CAPSULE_MAX_RANK_V1:
            raise SnapshotPlanContractError(
                f"snapshot section {name} leaves no capsule rank for world_count"
            )
        elements_per_world = 1
        for dimension in element_shape:
            elements_per_world *= dimension
        width = SNAPSHOT_DTYPES_V1[str(dtype)][1]
        bytes_per_world = _checked_u64_multiply(
            elements_per_world,
            width,
            label=f"snapshot section {name} byte extent",
        )
        offset = raw.get("source_offset_bytes", 0)
        if type(offset) is not int or offset < 0:
            raise SnapshotPlanContractError(
                f"snapshot section {name} source offset is invalid"
            )
        source_contract = (
            state_input if source == "state" else metrics_input
        )
        source_stride = raw.get(
            "source_world_stride_bytes", source_contract["world_stride_bytes"]
        )
        if source_stride != source_contract["world_stride_bytes"]:
            raise SnapshotPlanContractError(
                f"snapshot section {name} source stride differs from contiguous input"
            )
        supplied_source_shape = raw.get("source_shape")
        if supplied_source_shape is not None and _positive_shape(
            supplied_source_shape, label=f"section {name}.source_shape"
        ) != full_shape:
            raise SnapshotPlanContractError(
                f"snapshot section {name} source shape differs"
            )
        supplied_width = raw.get("element_width_bytes")
        if supplied_width is not None and supplied_width != width:
            raise SnapshotPlanContractError(
                f"snapshot section {name} element width differs"
            )
        supplied_bytes = raw.get("bytes_per_world")
        if supplied_bytes is not None and supplied_bytes != bytes_per_world:
            raise SnapshotPlanContractError(
                f"snapshot section {name} byte extent differs"
            )
        if (
            offset > source_stride
            or bytes_per_world > source_stride - offset
        ):
            raise SnapshotPlanContractError(
                f"snapshot section {name} exceeds one source world row"
            )
        sections.append(
            {
                "name": name,
                "kind": int(kind_id),
                "source": source,
                "source_offset_bytes": offset,
                "source_world_stride_bytes": source_stride,
                "world_major": True,
                "dtype": str(dtype),
                "dtype_id": SNAPSHOT_DTYPES_V1[str(dtype)][0],
                "element_width": width,
                "shape": list(full_shape),
                "element_shape": list(element_shape),
                "elements_per_world": elements_per_world,
                "payload_bytes_per_world": bytes_per_world,
            }
        )
    plan["world_count"] = world_count
    plan["state_input"] = state_input
    plan["metrics_input"] = metrics_input
    plan["snapshot_boundaries"] = boundaries
    plan["sections"] = sections
    return plan


def plan_identities_v1(
    normalized: Mapping[str, Any], canonical: bytes
) -> dict[str, str]:
    """Return the four header identities for a normalized plan."""

    return {
        "plan_sha256": hashlib.sha256(canonical).hexdigest(),
        "schema_sha256": hashlib.sha256(
            str(normalized["snapshot_schema_id"]).encode("utf-8")
        ).hexdigest(),
        "layout_sha256": str(normalized["layout_sha256"]),
        "control_sha256": str(normalized["native_world_control_set_sha256"]),
    }


def capsule_format_manifest_v1() -> dict[str, Any]:
    """Return the Python-side mirror of the fixed native format."""

    return {
        "schema_version": "eiel_snapshot_capsule_format_manifest_v1",
        "capsule_name": "EIEL_SNAPSHOT_CAPSULE_V1",
        "magic_hex": SNAPSHOT_CAPSULE_MAGIC_V1.hex(),
        "version": SNAPSHOT_CAPSULE_VERSION_V1,
        "header_bytes": SNAPSHOT_CAPSULE_HEADER_BYTES_V1,
        "descriptor_bytes": SNAPSHOT_CAPSULE_DESCRIPTOR_BYTES_V1,
        "alignment_bytes": SNAPSHOT_CAPSULE_ALIGNMENT_V1,
        "endian_marker": SNAPSHOT_CAPSULE_ENDIAN_MARKER_V1,
        "digest": "sha256",
        "digest_bytes": SNAPSHOT_CAPSULE_DIGEST_BYTES_V1,
        "max_rank": SNAPSHOT_CAPSULE_MAX_RANK_V1,
        "projection_backend": "HOST_CONTIGUOUS_SYNC_V1",
        "projection_backend_id": (
            SNAPSHOT_PROJECTION_BACKEND_HOST_CONTIGUOUS_SYNC_V1
        ),
        "supports_cuda_projection": False,
        "canonical_plan_embedded": False,
        "compact_canonical_plan_embedded": True,
        "compact_canonical_plan_header_index": (
            "ONE_BASED_SECTION_INDEX_V1"
        ),
        "plan_identity_payload_bytes": 128,
        "materialization_profile": "DENSE_SELECTED_WORLD_PARITY_ORACLE_V1",
        "active_row_compaction": False,
        "production_compaction_ready": False,
        "copy_memory_contract": (
            "PARITY_ONLY_SECTION_VECTORS_PLUS_CAPSULE_PLUS_PYTHON_BYTES_V1"
        ),
        "compact_stage_profile": SNAPSHOT_COMPACT_STAGE_PROFILE_V1,
        "compact_stage_encoding_policy": (
            SNAPSHOT_COMPACT_STAGE_ENCODING_POLICY_V1
        ),
        "compact_zero_extent_sections": True,
        "compact_stage_returns_payload_bytes": False,
        "compact_stage_exclusive_create_content_fsync_close": True,
        "compact_stage_parent_directory_fsync": False,
        "compact_stage_path_identity": "POSIX_ST_DEV_ST_INO_V1",
        "compact_active_row_compaction": True,
        "compact_production_publication_ready": False,
        "compatibility_converter": (
            "NATIVE_COMPACT_CAPSULE_TO_EXACT_CANONICAL_V1_V1"
        ),
        "compatibility_stage_encoding_policy": (
            SNAPSHOT_COMPATIBILITY_STAGE_ENCODING_POLICY_V1
        ),
        "compatibility_stage_returns_payload_bytes": False,
        "compatibility_stage_exclusive_create_content_fsync_close": True,
        "compatibility_stage_parent_directory_fsync": False,
        "compatibility_gzip_level": 9,
        "compatibility_gzip_mtime": 0,
        "section_kinds": dict(SNAPSHOT_SECTION_KINDS_V1),
        "dtypes": {
            name: {"id": identity, "width": width}
            for name, (identity, width) in SNAPSHOT_DTYPES_V1.items()
        },
    }


__all__ = [
    "REQUIRED_NATIVE_EXPORTS_V1",
    "SNAPSHOT_CAPSULE_ABI_V1",
    "SNAPSHOT_CAPSULE_ALIGNMENT_V1",
    "SNAPSHOT_CAPSULE_DESCRIPTOR_BYTES_V1",
    "SNAPSHOT_CAPSULE_ENDIAN_MARKER_V1",
    "SNAPSHOT_CAPSULE_HEADER_BYTES_V1",
    "SNAPSHOT_CAPSULE_MAGIC_V1",
    "SNAPSHOT_CAPSULE_VERSION_V1",
    "SNAPSHOT_COMPACT_STAGE_ENCODING_POLICY_V1",
    "SNAPSHOT_COMPACT_STAGE_PROFILE_V1",
    "SNAPSHOT_COMPATIBILITY_STAGE_ENCODING_POLICY_V1",
    "SNAPSHOT_DTYPES_V1",
    "SNAPSHOT_EXTENSION_VERSION_V1",
    "SNAPSHOT_PLAN_SCHEMA_V1",
    "SNAPSHOT_SECTION_KINDS_V1",
    "SnapshotPlanContractError",
    "canonical_plan_bytes_v1",
    "capsule_format_manifest_v1",
    "normalize_snapshot_plan_v1",
    "plan_identities_v1",
]
