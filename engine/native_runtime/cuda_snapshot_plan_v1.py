"""Declarative semantics and binary envelope checks for native CUDA snapshots.

This module is deliberately control-plane only.  It compiles the already
resolved Science Instrument acquisition contract into one small, canonical
plan and validates the project-owned ``EIEL_SNAPSHOT_CAPSULE_V1`` envelope.
It never materializes agent rows and it never owns simulation meaning.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import struct
from typing import Final

from emergence_contracts.science_compact_capture import (
    COMPACT_SCIENCE_AGENT_METRIC_FIELDS_V1,
    COMPACT_SCIENCE_SNAPSHOT_SCHEMA_V1,
)
from engine.campaigns.canonical import canonical_bytes, canonical_sha256
from engine.perf.cuda_full_evolution_v2.layout import (
    CUDA_NATIVE_ALIGNMENT_BYTES,
    CUDA_NATIVE_DTYPE_WIDTH_BYTES_V2,
    CUDA_NATIVE_HEADER_NAMES_V2,
    state_layout_v2,
)

NATIVE_SNAPSHOT_PLAN_SCHEMA_V1: Final[str] = "eiel_native_snapshot_plan_v1"
EIEL_SNAPSHOT_CAPSULE_SCHEMA_V1: Final[str] = "EIEL_SNAPSHOT_CAPSULE_V1"
CUDA_SNAPSHOT_SOURCE_LAYOUT_ID_V1: Final[str] = (
    "eiel_full_evolution_cuda_state_layout_v2_snapshot_source_v1"
)
NATIVE_SNAPSHOT_STATE_CAPTURE_TRANSPORT_SCHEMA_V1: Final[str] = (
    "eiel_native_snapshot_state_capture_transport_v1"
)
NATIVE_SNAPSHOT_STATE_CAPTURE_CANONICAL_PREFIX_V1: Final[str] = (
    "canonical_checkpoint_prefix_v1"
)
NATIVE_SNAPSHOT_STATE_CAPTURE_FULL_RESIDENT_V1: Final[str] = (
    "full_resident_row_v1"
)

SNAPSHOT_PLAN_FIXED_METADATA_BYTES_UPPER_BOUND_V1: Final[int] = 1024 * 1024
SNAPSHOT_PLAN_METADATA_BYTES_PER_WORLD_UPPER_BOUND_V1: Final[int] = 16 * 1024
MAX_SIGNED_INT64_V1: Final[int] = (1 << 63) - 1
MAX_UNSIGNED_INT64_V1: Final[int] = (1 << 64) - 1

SNAPSHOT_ORDERING_LAWS_V1: Final[dict[str, object]] = {
    "world_order": "native_projection_world_uid_order_v1",
    "world_block_order": ("world_state_then_active_agent_state_then_genome_catalog_v1"),
    "active_agent_order": "agent_id_numeric_ascending_v1",
    "genome_catalog_order": "genome_state_id_lexicographic_ascending_v1",
    "json_object_key_order": "unicode_codepoint_lexicographic_ascending_v1",
}
SNAPSHOT_SELECTION_LAWS_V1: Final[dict[str, object]] = {
    "time_coordinate": "absolute_tick_v1",
    "world_selection": "per_world_exact_snapshot_tick_membership_v1",
    "slot_presence": "agent_id_nonzero_v1",
    "alive_selection": "alive_bit_true_after_nonzero_identity_scan_v1",
    "duplicate_agent_identity": ("reject_across_all_nonzero_slots_including_dead_v1"),
    "slot_published": False,
    "dead_agent_rows_published": False,
}
SNAPSHOT_CATALOG_LAWS_V1: Final[dict[str, object]] = {
    "mode": "explicit_absolute_tick_or_disabled_v1",
    "membership": "active_genomes_only_v1",
    "deduplication": "genome_state_id_v1",
    "policy_order": "native_haplotype_locus_slot_projection_v1",
    "trait_order": "native_haplotype_locus_slot_projection_le_i16_v1",
}

# Capsule ABI frozen jointly with the native G2 owner.  All integers are LE.
EIEL_SNAPSHOT_CAPSULE_MAGIC_V1: Final[bytes] = b"EIELSNAPCAPSULE\0"
EIEL_SNAPSHOT_CAPSULE_VERSION_V1: Final[int] = 1
EIEL_SNAPSHOT_CAPSULE_HEADER_BYTES_V1: Final[int] = 384
EIEL_SNAPSHOT_CAPSULE_SECTION_DESCRIPTOR_BYTES_V1: Final[int] = 256
EIEL_SNAPSHOT_CAPSULE_ALIGNMENT_BYTES_V1: Final[int] = 64
EIEL_SNAPSHOT_CAPSULE_ENDIAN_MARKER_V1: Final[int] = 0x01020304
EIEL_SNAPSHOT_CAPSULE_TRAILER_BYTES_V1: Final[int] = 32
EIEL_SNAPSHOT_CAPSULE_PROJECTION_HOST_CONTIGUOUS_SYNC_V1: Final[int] = 1

EIEL_SNAPSHOT_CAPSULE_SECTION_KINDS_V1: Final[dict[str, int]] = {
    "canonical_plan_json": 1,
    "state_field": 2,
    "metrics_field": 3,
    "world_offsets": 4,
    "active_slots": 5,
    "identity": 6,
    "opaque_typed": 255,
}
EIEL_SNAPSHOT_CAPSULE_SECTION_KIND_NAMES_V1: Final[dict[int, str]] = {
    value: key for key, value in EIEL_SNAPSHOT_CAPSULE_SECTION_KINDS_V1.items()
}
EIEL_SNAPSHOT_CAPSULE_DTYPES_V1: Final[dict[str, tuple[int, int]]] = {
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
EIEL_SNAPSHOT_CAPSULE_DTYPE_NAMES_V1: Final[dict[int, tuple[str, int]]] = {
    code: (name, width)
    for name, (code, width) in EIEL_SNAPSHOT_CAPSULE_DTYPES_V1.items()
}

_CAPSULE_HEADER_PREFIX_V1 = struct.Struct("<16sHHIII" + "Q" * 13 + "32s32s32s32sIIQ")
_CAPSULE_SECTION_PREFIX_V1 = struct.Struct("<IHHII" + "Q" * 7 + "Q" * 8 + "32s")
if _CAPSULE_HEADER_PREFIX_V1.size != 280:  # pragma: no cover - import invariant
    raise RuntimeError("snapshot capsule header prefix must be exactly 280 bytes")
if _CAPSULE_SECTION_PREFIX_V1.size != 168:  # pragma: no cover - import invariant
    raise RuntimeError("snapshot capsule section prefix must be exactly 168 bytes")


class CudaSnapshotPlanError(ValueError):
    """A declarative snapshot plan is malformed or semantically ambiguous."""


class EielSnapshotCapsuleError(ValueError):
    """A native snapshot capsule does not satisfy the frozen V1 ABI."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _is_lower_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_lower_sha256(value: object, *, label: str) -> str:
    if not _is_lower_sha256(value):
        raise CudaSnapshotPlanError(f"{label} must be one lowercase SHA-256 hex")
    return str(value)


def _align(
    value: int, alignment: int = EIEL_SNAPSHOT_CAPSULE_ALIGNMENT_BYTES_V1
) -> int:
    if type(value) is not int or value < 0:
        raise ValueError("alignment input must be a nonnegative integer")
    return ((value + alignment - 1) // alignment) * alignment


def _checked_product(shape: Sequence[int], *, label: str) -> int:
    result = 1
    for dimension in shape:
        if type(dimension) is not int or dimension < 0:
            raise CudaSnapshotPlanError(
                f"{label} dimensions must be nonnegative exact integers"
            )
        result *= dimension
        if result > MAX_UNSIGNED_INT64_V1:
            raise CudaSnapshotPlanError(f"{label} element count exceeds u64")
    return result


def _canonical_fragment(value: object, *, label: str) -> str:
    try:
        return canonical_bytes(value).decode("ascii")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise CudaSnapshotPlanError(f"{label} is not canonical JSON data") from exc


def _validate_content_digest(value: object, *, label: str) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != {"algorithm", "hex"}:
        raise CudaSnapshotPlanError(f"{label} must be an exact SHA-256 digest object")
    if value.get("algorithm") != "sha256" or not _is_lower_sha256(value.get("hex")):
        raise CudaSnapshotPlanError(f"{label} must be an exact SHA-256 digest object")
    return {"algorithm": "sha256", "hex": str(value["hex"])}


def _parse_canonical_fragment(value: object, *, label: str) -> object:
    if type(value) is not str:
        raise CudaSnapshotPlanError(f"{label} must be preescaped canonical JSON")
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise CudaSnapshotPlanError(f"{label} is invalid preescaped JSON") from exc
    if _canonical_fragment(parsed, label=label) != value:
        raise CudaSnapshotPlanError(f"{label} is not canonical preescaped JSON")
    return parsed


def _snapshot_layout_descriptor_v1() -> dict[str, object]:
    """Return the one frozen CUDA V2 source layout supported by capsule V1."""

    resolved = state_layout_v2()
    try:
        fields = tuple(resolved.fields)
        state_bytes = int(resolved.state_bytes_per_world)
        canonical_state_bytes = int(resolved.canonical_checkpoint_bytes_per_world)
    except (AttributeError, TypeError, ValueError) as exc:
        raise CudaSnapshotPlanError("snapshot source layout is invalid") from exc
    field_rows: list[dict[str, object]] = []
    for field in fields:
        try:
            name = str(field.name)
            dtype = str(field.dtype)
            shape = [int(value) for value in field.shape]
            offset = int(field.offset_bytes)
            byte_count = int(field.bytes)
        except (AttributeError, TypeError, ValueError) as exc:
            raise CudaSnapshotPlanError("snapshot source field is invalid") from exc
        if dtype not in CUDA_NATIVE_DTYPE_WIDTH_BYTES_V2:
            raise CudaSnapshotPlanError(
                f"snapshot source field {name} dtype is unsupported"
            )
        if any(value <= 0 for value in shape):
            raise CudaSnapshotPlanError(
                f"snapshot source field {name} shape is invalid"
            )
        expected = _checked_product(shape, label=f"layout field {name}") * int(
            CUDA_NATIVE_DTYPE_WIDTH_BYTES_V2[dtype]
        )
        if offset < 0 or byte_count != expected:
            raise CudaSnapshotPlanError(f"snapshot source field {name} extent differs")
        field_rows.append(
            {
                "name": name,
                "dtype": dtype,
                "shape": shape,
                "offset_bytes": offset,
                "bytes": byte_count,
            }
        )
    by_name = {str(row["name"]): row for row in field_rows}
    required = {
        "scalar_header_counters",
        "resource",
        "position",
        "energy",
        "health",
        "age",
        "absolute_birth_tick",
        "alive_words",
        "agent_id",
        "parent_a_id",
        "parent_b_id",
        "policy_alleles",
        "trait_alleles",
    }
    if set(by_name) < required:
        raise CudaSnapshotPlanError("snapshot source layout lacks required fields")
    capacity = int(by_name["agent_id"]["shape"][-1])
    for name in (
        "position",
        "energy",
        "health",
        "age",
        "absolute_birth_tick",
        "agent_id",
        "parent_a_id",
        "parent_b_id",
    ):
        if by_name[name]["shape"] != [capacity]:
            raise CudaSnapshotPlanError(
                f"snapshot source field {name} capacity differs"
            )
    policy_shape = list(by_name["policy_alleles"]["shape"])
    trait_shape = list(by_name["trait_alleles"]["shape"])
    if len(policy_shape) != 3 or len(trait_shape) != 3:
        raise CudaSnapshotPlanError("snapshot allele geometry must be rank three")
    if policy_shape[-1] != capacity or trait_shape[-1] != capacity:
        raise CudaSnapshotPlanError("snapshot allele capacity differs")
    alive = by_name["alive_words"]
    alive_word_bits = int(CUDA_NATIVE_DTYPE_WIDTH_BYTES_V2[str(alive["dtype"])]) * 8
    if alive["shape"] != [capacity // alive_word_bits] or capacity % alive_word_bits:
        raise CudaSnapshotPlanError("snapshot alive-word geometry differs")
    resource_shape = list(by_name["resource"]["shape"])
    if len(resource_shape) != 1:
        raise CudaSnapshotPlanError("snapshot resource geometry must be rank one")
    if list(by_name["scalar_header_counters"]["shape"]) != [
        len(CUDA_NATIVE_HEADER_NAMES_V2)
    ]:
        raise CudaSnapshotPlanError("snapshot header registry extent differs")
    if (
        state_bytes <= 0
        or canonical_state_bytes <= 0
        or canonical_state_bytes > state_bytes
    ):
        raise CudaSnapshotPlanError("snapshot state byte extent differs")
    return {
        "layout_id": CUDA_SNAPSHOT_SOURCE_LAYOUT_ID_V1,
        "byte_order": "little",
        "alignment_bytes": int(CUDA_NATIVE_ALIGNMENT_BYTES),
        "state_bytes_per_world": state_bytes,
        "canonical_state_bytes_per_world": canonical_state_bytes,
        "header_names": list(CUDA_NATIVE_HEADER_NAMES_V2),
        "agent_capacity": capacity,
        "alive_word_bits": alive_word_bits,
        "alive_word_count": int(alive["shape"][0]),
        "resource_cell_count": int(resource_shape[0]),
        "policy_geometry": {
            "dtype": by_name["policy_alleles"]["dtype"],
            "shape": policy_shape,
            "offset_bytes": by_name["policy_alleles"]["offset_bytes"],
            "bytes": by_name["policy_alleles"]["bytes"],
        },
        "trait_geometry": {
            "dtype": by_name["trait_alleles"]["dtype"],
            "shape": trait_shape,
            "offset_bytes": by_name["trait_alleles"]["offset_bytes"],
            "bytes": by_name["trait_alleles"]["bytes"],
        },
        "fields": field_rows,
    }


def _state_capture_transport_v1(
    *, world_count: int, layout_descriptor: Mapping[str, object]
) -> dict[str, object]:
    """Bind the resident source separately from its accepted host transports."""

    state_bytes = int(layout_descriptor["state_bytes_per_world"])
    canonical_bytes = int(layout_descriptor["canonical_state_bytes_per_world"])

    def mode(name: str, row_bytes: int) -> dict[str, object]:
        total_bytes = _checked_product(
            (world_count, row_bytes),
            label=f"snapshot state capture transport {name}",
        )
        return {
            "mode": name,
            "dtype": "u8",
            "shape": [world_count, row_bytes],
            "contiguity": "C",
            "world_stride_bytes": row_bytes,
            "total_bytes": total_bytes,
        }

    return {
        "schema_version": NATIVE_SNAPSHOT_STATE_CAPTURE_TRANSPORT_SCHEMA_V1,
        "resident_source_contract": "state_input",
        "prefix_semantics": "leading_bytes_of_each_resident_world_row_v1",
        "preferred_mode": NATIVE_SNAPSHOT_STATE_CAPTURE_CANONICAL_PREFIX_V1,
        "accepted_modes": [
            mode(
                NATIVE_SNAPSHOT_STATE_CAPTURE_CANONICAL_PREFIX_V1,
                canonical_bytes,
            ),
            mode(NATIVE_SNAPSHOT_STATE_CAPTURE_FULL_RESIDENT_V1, state_bytes),
        ],
    }


_SNAPSHOT_STATE_FIELDS_V1: Final[tuple[str, ...]] = (
    "scalar_header_counters",
    "resource",
    "position",
    "energy",
    "health",
    "age",
    "absolute_birth_tick",
    "alive_words",
    "agent_id",
    "parent_a_id",
    "parent_b_id",
    "policy_alleles",
    "trait_alleles",
)


def _snapshot_sections_v1(
    *, world_count: int, layout_descriptor: Mapping[str, object]
) -> list[dict[str, object]]:
    fields = {
        str(value["name"]): value
        for value in layout_descriptor["fields"]  # type: ignore[index]
        if isinstance(value, Mapping)
    }
    state_stride = int(layout_descriptor["state_bytes_per_world"])
    sections: list[dict[str, object]] = []
    for name in _SNAPSHOT_STATE_FIELDS_V1:
        field = fields[name]
        element_shape = [int(value) for value in field["shape"]]
        source_shape = [world_count, *element_shape]
        width = int(CUDA_NATIVE_DTYPE_WIDTH_BYTES_V2[str(field["dtype"])])
        elements_per_world = _checked_product(
            element_shape, label=f"section {name} per-world shape"
        )
        sections.append(
            {
                "name": name,
                "kind": "state_field",
                "source": "state",
                "world_major": True,
                "source_offset_bytes": int(field["offset_bytes"]),
                "source_world_stride_bytes": state_stride,
                "source_shape": source_shape,
                "element_shape": element_shape,
                "dtype": str(field["dtype"]),
                "element_width_bytes": width,
                "elements_per_world": elements_per_world,
                "bytes_per_world": elements_per_world * width,
            }
        )
    capacity = int(layout_descriptor["agent_capacity"])
    metric_element_shape = [capacity, len(COMPACT_SCIENCE_AGENT_METRIC_FIELDS_V1)]
    metric_count = _checked_product(
        metric_element_shape, label="lifetime metrics per-world section"
    )
    sections.append(
        {
            "name": "lifetime_metrics",
            "kind": "metrics_field",
            "source": "metrics",
            "world_major": True,
            "source_offset_bytes": 0,
            "source_world_stride_bytes": capacity
            * len(COMPACT_SCIENCE_AGENT_METRIC_FIELDS_V1)
            * 8,
            "source_shape": [world_count, *metric_element_shape],
            "element_shape": metric_element_shape,
            "dtype": "i64",
            "element_width_bytes": 8,
            "elements_per_world": metric_count,
            "bytes_per_world": metric_count * 8,
        }
    )
    return sections


def _snapshot_ticks(value: object, *, label: str) -> list[int]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise CudaSnapshotPlanError(f"{label} must be an ordered absolute-tick list")
    ticks = list(value)
    if any(
        type(tick) is not int or tick < 0 or tick > MAX_SIGNED_INT64_V1
        for tick in ticks
    ):
        raise CudaSnapshotPlanError(f"{label} contains an invalid absolute tick")
    if ticks != sorted(set(ticks)):
        raise CudaSnapshotPlanError(
            f"{label} must be strictly increasing without duplicate ticks"
        )
    return [int(tick) for tick in ticks]


def _boundaries_from_worlds_v1(
    worlds: Sequence[Mapping[str, object]], *, catalog_absolute_tick: int | None
) -> list[dict[str, object]]:
    selected: dict[int, list[int]] = {}
    for world in worlds:
        world_index = int(world["world_index"])
        for tick in world["snapshot_ticks"]:  # type: ignore[union-attr]
            selected.setdefault(int(tick), []).append(world_index)
    world_uid_by_index = {
        int(world["world_index"]): int(world["world_uid"]) for world in worlds
    }
    return [
        {
            "absolute_tick": tick,
            "world_indices": indices,
            "world_uids": [world_uid_by_index[index] for index in indices],
            "include_genome_catalog": catalog_absolute_tick == tick,
        }
        for tick, indices in sorted(selected.items())
    ]


@dataclass(frozen=True, slots=True)
class CompiledCudaSnapshotPlanV1:
    """Immutable canonical plan bytes plus their non-circular identity."""

    canonical_plan_bytes: bytes
    plan_sha256: str

    def __post_init__(self) -> None:
        if (
            type(self.canonical_plan_bytes) is not bytes
            or not self.canonical_plan_bytes
        ):
            raise CudaSnapshotPlanError("compiled snapshot plan bytes are absent")
        if not _is_lower_sha256(self.plan_sha256):
            raise CudaSnapshotPlanError("compiled snapshot plan identity is invalid")
        if _sha256_bytes(self.canonical_plan_bytes) != self.plan_sha256:
            raise CudaSnapshotPlanError("compiled snapshot plan identity differs")

    def to_mapping(self) -> dict[str, object]:
        value = json.loads(self.canonical_plan_bytes)
        if not isinstance(value, dict):  # pragma: no cover - constructor invariant
            raise CudaSnapshotPlanError("compiled snapshot plan must be an object")
        return value


def compile_cuda_snapshot_plan_v1(
    *,
    world_uids: Sequence[int],
    world_cells: Mapping[str, Mapping[str, object]],
    native_world_control_set_sha256: str,
    execution_publication_context_sha256: str,
    catalog_absolute_tick: int | None,
    snapshot_schema_id: str = COMPACT_SCIENCE_SNAPSHOT_SCHEMA_V1,
) -> CompiledCudaSnapshotPlanV1:
    """Compile one exact, row-free native snapshot plan.

    ``world_uids`` is the authoritative native projection order.  Mapping
    insertion order is ignored.  Per-world acquisition schedules are compiled
    once into ordered boundary selections, and invariant JSON values are
    escaped once for direct native encoding.  The execution/publication
    context digest is computed by Python authority over the campaign,
    dispatch, output, and evidence identity; the capsule binds it transitively
    through the canonical plan SHA-256 without repeating that metadata.
    """

    if isinstance(world_uids, (str, bytes)) or not isinstance(world_uids, Sequence):
        raise CudaSnapshotPlanError("snapshot world order must be a sequence")
    ordered_uids = list(world_uids)
    if not ordered_uids:
        raise CudaSnapshotPlanError("snapshot plan requires at least one world")
    if any(
        type(uid) is not int or uid <= 0 or uid > MAX_UNSIGNED_INT64_V1
        for uid in ordered_uids
    ) or len(set(ordered_uids)) != len(ordered_uids):
        raise CudaSnapshotPlanError(
            "snapshot world order requires unique positive u64 identities"
        )
    if not isinstance(world_cells, Mapping):
        raise CudaSnapshotPlanError("snapshot logical worlds must be a mapping")
    if snapshot_schema_id != COMPACT_SCIENCE_SNAPSHOT_SCHEMA_V1:
        raise CudaSnapshotPlanError("snapshot compatibility schema differs from V1")
    control_sha256 = _require_lower_sha256(
        native_world_control_set_sha256,
        label="native world-control set identity",
    )
    publication_context_sha256 = _require_lower_sha256(
        execution_publication_context_sha256,
        label="execution/publication context identity",
    )
    if catalog_absolute_tick is not None and (
        type(catalog_absolute_tick) is not int
        or catalog_absolute_tick < 0
        or catalog_absolute_tick > MAX_SIGNED_INT64_V1
    ):
        raise CudaSnapshotPlanError("catalog boundary must be an absolute tick or null")

    cells_by_uid: dict[int, tuple[str, Mapping[str, object]]] = {}
    for cell_id, cell in world_cells.items():
        if type(cell_id) is not str or not cell_id:
            raise CudaSnapshotPlanError(
                "snapshot cell identities must be nonempty strings"
            )
        if not isinstance(cell, Mapping):
            raise CudaSnapshotPlanError(f"snapshot cell {cell_id} must be a mapping")
        uid = cell.get("science_logical_world_id")
        if type(uid) is not int or uid <= 0 or uid in cells_by_uid:
            raise CudaSnapshotPlanError(
                "snapshot cells require unique positive logical-world identities"
            )
        cells_by_uid[int(uid)] = (cell_id, cell)
    if set(cells_by_uid) != set(ordered_uids):
        raise CudaSnapshotPlanError(
            "snapshot cells do not exactly cover the native world order"
        )

    worlds: list[dict[str, object]] = []
    for world_index, uid in enumerate(ordered_uids):
        cell_id, cell = cells_by_uid[uid]
        acquisition = cell.get("science_acquisition")
        if not isinstance(acquisition, Mapping) or "snapshot_ticks" not in acquisition:
            raise CudaSnapshotPlanError(
                f"snapshot cell {cell_id} lacks an exact acquisition schedule"
            )
        ticks = _snapshot_ticks(
            acquisition["snapshot_ticks"], label=f"snapshot cell {cell_id} schedule"
        )
        ownership = cell.get("science_ownership")
        if not isinstance(ownership, Mapping):
            raise CudaSnapshotPlanError(
                f"snapshot cell {cell_id} ownership must be an object"
            )
        world_digest = _validate_content_digest(
            cell.get("science_world_sha256"),
            label=f"snapshot cell {cell_id} semantic-world identity",
        )
        acquisition_digest = _validate_content_digest(
            cell.get("science_acquisition_sha256"),
            label=f"snapshot cell {cell_id} acquisition identity",
        )
        invariant = {
            "cell_id": cell_id,
            "science_ownership": dict(ownership),
            "science_world_sha256": world_digest,
            "science_acquisition_sha256": acquisition_digest,
        }
        fragments = {
            f"{name}_json": _canonical_fragment(value, label=f"{cell_id} {name}")
            for name, value in invariant.items()
        }
        fragment_bytes = sum(len(value.encode("ascii")) for value in fragments.values())
        if fragment_bytes > SNAPSHOT_PLAN_METADATA_BYTES_PER_WORLD_UPPER_BOUND_V1:
            raise CudaSnapshotPlanError(
                f"snapshot cell {cell_id} invariant metadata exceeds its bound"
            )
        worlds.append(
            {
                "world_index": world_index,
                "world_uid": uid,
                "snapshot_ticks": ticks,
                **fragments,
                "invariant_metadata_sha256": canonical_sha256(invariant),
            }
        )
    boundaries = _boundaries_from_worlds_v1(
        worlds, catalog_absolute_tick=catalog_absolute_tick
    )
    if not boundaries:
        raise CudaSnapshotPlanError("snapshot plan contains no requested boundary")
    if catalog_absolute_tick is not None and all(
        boundary["absolute_tick"] != catalog_absolute_tick for boundary in boundaries
    ):
        raise CudaSnapshotPlanError(
            "catalog absolute tick is not requested by any logical world"
        )

    layout_descriptor = _snapshot_layout_descriptor_v1()
    layout_sha256 = canonical_sha256(layout_descriptor)
    metric_fields = list(COMPACT_SCIENCE_AGENT_METRIC_FIELDS_V1)
    if len(metric_fields) != 21 or len(set(metric_fields)) != len(metric_fields):
        raise CudaSnapshotPlanError(
            "snapshot V1 metric registry must contain 21 fields"
        )
    metric_sha256 = canonical_sha256(metric_fields)
    world_count = len(ordered_uids)
    state_bytes = int(layout_descriptor["state_bytes_per_world"])
    capacity = int(layout_descriptor["agent_capacity"])
    body: dict[str, object] = {
        "schema_version": NATIVE_SNAPSHOT_PLAN_SCHEMA_V1,
        "capsule_schema_id": EIEL_SNAPSHOT_CAPSULE_SCHEMA_V1,
        "snapshot_schema_id": snapshot_schema_id,
        "schema_sha256": _sha256_bytes(snapshot_schema_id.encode("utf-8")),
        "layout_id": CUDA_SNAPSHOT_SOURCE_LAYOUT_ID_V1,
        "layout_sha256": layout_sha256,
        "native_world_control_set_sha256": control_sha256,
        "execution_publication_context_sha256": publication_context_sha256,
        "metric_registry_sha256": metric_sha256,
        "metric_fields": metric_fields,
        "world_count": world_count,
        "state_input": {
            "dtype": "u8",
            "shape": [world_count, state_bytes],
            "contiguity": "C",
            "world_stride_bytes": state_bytes,
        },
        "state_capture_transport": _state_capture_transport_v1(
            world_count=world_count,
            layout_descriptor=layout_descriptor,
        ),
        "metrics_input": {
            "dtype": "i64",
            "shape": [world_count, capacity, len(metric_fields)],
            "contiguity": "C",
            "world_stride_bytes": capacity * len(metric_fields) * 8,
        },
        "layout": layout_descriptor,
        "sections": _snapshot_sections_v1(
            world_count=world_count, layout_descriptor=layout_descriptor
        ),
        "worlds": worlds,
        "snapshot_boundaries": boundaries,
        "catalog": {
            **SNAPSHOT_CATALOG_LAWS_V1,
            "absolute_tick": catalog_absolute_tick,
        },
        "ordering_laws": dict(SNAPSHOT_ORDERING_LAWS_V1),
        "selection_laws": dict(SNAPSHOT_SELECTION_LAWS_V1),
    }
    encoded = canonical_bytes(body)
    metadata_bound = (
        SNAPSHOT_PLAN_FIXED_METADATA_BYTES_UPPER_BOUND_V1
        + world_count * SNAPSHOT_PLAN_METADATA_BYTES_PER_WORLD_UPPER_BOUND_V1
        + sum(len(world["snapshot_ticks"]) for world in worlds) * 32
    )
    if len(encoded) > metadata_bound:
        raise CudaSnapshotPlanError(
            "canonical snapshot plan exceeds its bounded envelope"
        )
    compiled = CompiledCudaSnapshotPlanV1(
        canonical_plan_bytes=encoded,
        plan_sha256=_sha256_bytes(encoded),
    )
    return validate_cuda_snapshot_plan_v1(compiled)


_PLAN_TOP_LEVEL_KEYS_V1: Final[set[str]] = {
    "schema_version",
    "capsule_schema_id",
    "snapshot_schema_id",
    "schema_sha256",
    "layout_id",
    "layout_sha256",
    "native_world_control_set_sha256",
    "execution_publication_context_sha256",
    "metric_registry_sha256",
    "metric_fields",
    "world_count",
    "state_input",
    "state_capture_transport",
    "metrics_input",
    "layout",
    "sections",
    "worlds",
    "snapshot_boundaries",
    "catalog",
    "ordering_laws",
    "selection_laws",
}


def _validate_plan_mapping_v1(value: Mapping[str, object]) -> None:
    if set(value) != _PLAN_TOP_LEVEL_KEYS_V1:
        raise CudaSnapshotPlanError("snapshot plan top-level fields differ from V1")
    if (
        value.get("schema_version") != NATIVE_SNAPSHOT_PLAN_SCHEMA_V1
        or value.get("capsule_schema_id") != EIEL_SNAPSHOT_CAPSULE_SCHEMA_V1
        or value.get("snapshot_schema_id") != COMPACT_SCIENCE_SNAPSHOT_SCHEMA_V1
        or value.get("schema_sha256")
        != _sha256_bytes(COMPACT_SCIENCE_SNAPSHOT_SCHEMA_V1.encode("utf-8"))
        or value.get("layout_id") != CUDA_SNAPSHOT_SOURCE_LAYOUT_ID_V1
    ):
        raise CudaSnapshotPlanError("snapshot plan schema identity differs")
    _require_lower_sha256(
        value.get("native_world_control_set_sha256"),
        label="native world-control set identity",
    )
    _require_lower_sha256(
        value.get("execution_publication_context_sha256"),
        label="execution/publication context identity",
    )
    expected_layout = _snapshot_layout_descriptor_v1()
    if value.get("layout") != expected_layout:
        raise CudaSnapshotPlanError("snapshot plan source layout differs")
    expected_layout_sha256 = canonical_sha256(expected_layout)
    if value.get("layout_sha256") != expected_layout_sha256:
        raise CudaSnapshotPlanError("snapshot plan layout identity differs")
    metric_fields = list(COMPACT_SCIENCE_AGENT_METRIC_FIELDS_V1)
    if value.get("metric_fields") != metric_fields or value.get(
        "metric_registry_sha256"
    ) != canonical_sha256(metric_fields):
        raise CudaSnapshotPlanError("snapshot plan metric registry differs")
    world_count = value.get("world_count")
    if type(world_count) is not int or world_count <= 0:
        raise CudaSnapshotPlanError("snapshot plan world count is invalid")
    state_bytes = int(expected_layout["state_bytes_per_world"])
    capacity = int(expected_layout["agent_capacity"])
    expected_state_input = {
        "dtype": "u8",
        "shape": [world_count, state_bytes],
        "contiguity": "C",
        "world_stride_bytes": state_bytes,
    }
    expected_metrics_input = {
        "dtype": "i64",
        "shape": [world_count, capacity, len(metric_fields)],
        "contiguity": "C",
        "world_stride_bytes": capacity * len(metric_fields) * 8,
    }
    if value.get("state_input") != expected_state_input:
        raise CudaSnapshotPlanError("snapshot plan state input shape differs")
    expected_state_capture_transport = _state_capture_transport_v1(
        world_count=world_count,
        layout_descriptor=expected_layout,
    )
    if value.get("state_capture_transport") != expected_state_capture_transport:
        raise CudaSnapshotPlanError(
            "snapshot plan state capture transport differs"
        )
    if value.get("metrics_input") != expected_metrics_input:
        raise CudaSnapshotPlanError("snapshot plan metric input shape differs")
    if value.get("sections") != _snapshot_sections_v1(
        world_count=world_count, layout_descriptor=expected_layout
    ):
        raise CudaSnapshotPlanError("snapshot plan source sections differ")
    if value.get("ordering_laws") != SNAPSHOT_ORDERING_LAWS_V1:
        raise CudaSnapshotPlanError("snapshot ordering laws differ")
    if value.get("selection_laws") != SNAPSHOT_SELECTION_LAWS_V1:
        raise CudaSnapshotPlanError("snapshot selection laws differ")
    catalog = value.get("catalog")
    if not isinstance(catalog, Mapping) or set(catalog) != {
        *SNAPSHOT_CATALOG_LAWS_V1,
        "absolute_tick",
    }:
        raise CudaSnapshotPlanError("snapshot catalog law differs")
    if any(
        catalog.get(key) != child for key, child in SNAPSHOT_CATALOG_LAWS_V1.items()
    ):
        raise CudaSnapshotPlanError("snapshot catalog law differs")
    catalog_tick = catalog.get("absolute_tick")
    if catalog_tick is not None and (
        type(catalog_tick) is not int
        or catalog_tick < 0
        or catalog_tick > MAX_SIGNED_INT64_V1
    ):
        raise CudaSnapshotPlanError("snapshot catalog absolute tick is invalid")

    raw_worlds = value.get("worlds")
    if not isinstance(raw_worlds, list) or len(raw_worlds) != world_count:
        raise CudaSnapshotPlanError("snapshot plan worlds differ")
    worlds: list[Mapping[str, object]] = []
    seen_uids: set[int] = set()
    seen_cells: set[str] = set()
    for expected_index, world in enumerate(raw_worlds):
        if not isinstance(world, Mapping) or set(world) != {
            "world_index",
            "world_uid",
            "snapshot_ticks",
            "cell_id_json",
            "science_ownership_json",
            "science_world_sha256_json",
            "science_acquisition_sha256_json",
            "invariant_metadata_sha256",
        }:
            raise CudaSnapshotPlanError("snapshot plan world fields differ")
        uid = world.get("world_uid")
        if (
            world.get("world_index") != expected_index
            or type(uid) is not int
            or uid <= 0
            or uid > MAX_UNSIGNED_INT64_V1
            or uid in seen_uids
        ):
            raise CudaSnapshotPlanError("snapshot plan world order differs")
        ticks = _snapshot_ticks(
            world.get("snapshot_ticks"),
            label=f"snapshot world {expected_index} schedule",
        )
        cell_id = _parse_canonical_fragment(
            world.get("cell_id_json"), label=f"snapshot world {expected_index} cell ID"
        )
        ownership = _parse_canonical_fragment(
            world.get("science_ownership_json"),
            label=f"snapshot world {expected_index} ownership",
        )
        world_digest = _parse_canonical_fragment(
            world.get("science_world_sha256_json"),
            label=f"snapshot world {expected_index} semantic-world identity",
        )
        acquisition_digest = _parse_canonical_fragment(
            world.get("science_acquisition_sha256_json"),
            label=f"snapshot world {expected_index} acquisition identity",
        )
        if type(cell_id) is not str or not cell_id or cell_id in seen_cells:
            raise CudaSnapshotPlanError("snapshot plan cell identities differ")
        if not isinstance(ownership, dict):
            raise CudaSnapshotPlanError("snapshot plan ownership must be an object")
        world_digest = _validate_content_digest(
            world_digest, label=f"snapshot world {expected_index} semantic identity"
        )
        acquisition_digest = _validate_content_digest(
            acquisition_digest,
            label=f"snapshot world {expected_index} acquisition identity",
        )
        invariant = {
            "cell_id": cell_id,
            "science_ownership": ownership,
            "science_world_sha256": world_digest,
            "science_acquisition_sha256": acquisition_digest,
        }
        if world.get("invariant_metadata_sha256") != canonical_sha256(invariant):
            raise CudaSnapshotPlanError(
                "snapshot plan invariant metadata digest differs"
            )
        fragment_bytes = sum(
            len(str(world[key]).encode("ascii"))
            for key in (
                "cell_id_json",
                "science_ownership_json",
                "science_world_sha256_json",
                "science_acquisition_sha256_json",
            )
        )
        if fragment_bytes > SNAPSHOT_PLAN_METADATA_BYTES_PER_WORLD_UPPER_BOUND_V1:
            raise CudaSnapshotPlanError(
                "snapshot plan invariant metadata exceeds its bound"
            )
        seen_uids.add(int(uid))
        seen_cells.add(cell_id)
        worlds.append({**world, "snapshot_ticks": ticks})
    expected_boundaries = _boundaries_from_worlds_v1(
        worlds, catalog_absolute_tick=catalog_tick  # type: ignore[arg-type]
    )
    if (
        value.get("snapshot_boundaries") != expected_boundaries
        or not expected_boundaries
    ):
        raise CudaSnapshotPlanError("snapshot boundary selection differs")
    if catalog_tick is not None and all(
        boundary["absolute_tick"] != catalog_tick for boundary in expected_boundaries
    ):
        raise CudaSnapshotPlanError("snapshot catalog tick is not requested")


def validate_cuda_snapshot_plan_v1(
    value: CompiledCudaSnapshotPlanV1 | Mapping[str, object] | bytes,
) -> CompiledCudaSnapshotPlanV1:
    """Validate and return one canonical immutable plan representation."""

    if isinstance(value, CompiledCudaSnapshotPlanV1):
        raw = value.canonical_plan_bytes
    elif type(value) is bytes:
        raw = value
    elif isinstance(value, Mapping):
        try:
            raw = canonical_bytes(dict(value))
        except (TypeError, ValueError) as exc:
            raise CudaSnapshotPlanError("snapshot plan is not canonical JSON") from exc
    else:
        raise CudaSnapshotPlanError(
            "snapshot plan must be a mapping or canonical bytes"
        )
    try:
        parsed = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise CudaSnapshotPlanError("snapshot plan bytes are invalid JSON") from exc
    if not isinstance(parsed, dict):
        raise CudaSnapshotPlanError("snapshot plan must be a JSON object")
    try:
        recanonicalized = canonical_bytes(parsed)
    except (TypeError, ValueError) as exc:
        raise CudaSnapshotPlanError("snapshot plan JSON domain is invalid") from exc
    if raw != recanonicalized:
        raise CudaSnapshotPlanError("snapshot plan bytes are not exact canonical JSON")
    _validate_plan_mapping_v1(parsed)
    return CompiledCudaSnapshotPlanV1(
        canonical_plan_bytes=raw,
        plan_sha256=_sha256_bytes(raw),
    )


@dataclass(frozen=True, slots=True)
class EielSnapshotCapsuleSectionV1:
    index: int
    name: str
    kind: int
    kind_name: str
    dtype: int
    dtype_name: str
    shape: tuple[int, ...]
    element_width: int
    element_count: int
    payload_offset: int
    payload_length: int
    payload_sha256: str


@dataclass(frozen=True, slots=True)
class EielSnapshotCapsuleViewV1:
    absolute_tick: int
    world_count: int
    row_count: int
    section_count: int
    plan_sha256: str
    schema_sha256: str
    layout_sha256: str
    native_world_control_set_sha256: str
    projection_backend: int
    capsule_bytes: int
    capsule_sha256: str
    plan: CompiledCudaSnapshotPlanV1 | None
    sections: tuple[EielSnapshotCapsuleSectionV1, ...]


def _capsule_error(message: str) -> EielSnapshotCapsuleError:
    return EielSnapshotCapsuleError(message)


def _capsule_digest_hex(value: bytes) -> str:
    return value.hex()


def validate_eiel_snapshot_capsule_v1(
    value: bytes | bytearray | memoryview,
    *,
    expected_plan: (
        CompiledCudaSnapshotPlanV1 | Mapping[str, object] | bytes | None
    ) = None,
) -> EielSnapshotCapsuleViewV1:
    """Validate the complete fixed-header capsule without materializing rows."""

    if not isinstance(value, (bytes, bytearray, memoryview)):
        raise _capsule_error("snapshot capsule must be bytes-like")
    raw = bytes(value)
    if len(raw) < EIEL_SNAPSHOT_CAPSULE_HEADER_BYTES_V1 + 256 + 32:
        raise _capsule_error("snapshot capsule is truncated")
    unpacked = _CAPSULE_HEADER_PREFIX_V1.unpack_from(raw, 0)
    (
        magic,
        version,
        header_bytes,
        endian_marker,
        descriptor_bytes,
        alignment,
        flags,
        absolute_tick,
        world_count,
        row_count,
        section_count,
        directory_offset,
        directory_bytes,
        names_offset,
        names_bytes,
        payload_offset,
        payload_bytes,
        trailer_offset,
        capsule_bytes,
        plan_digest,
        schema_digest,
        layout_digest,
        control_digest,
        projection_backend,
        reserved_word,
        plan_identity_section_index,
    ) = unpacked
    if (
        magic != EIEL_SNAPSHOT_CAPSULE_MAGIC_V1
        or version != EIEL_SNAPSHOT_CAPSULE_VERSION_V1
        or header_bytes != EIEL_SNAPSHOT_CAPSULE_HEADER_BYTES_V1
        or endian_marker != EIEL_SNAPSHOT_CAPSULE_ENDIAN_MARKER_V1
        or descriptor_bytes != EIEL_SNAPSHOT_CAPSULE_SECTION_DESCRIPTOR_BYTES_V1
        or alignment != EIEL_SNAPSHOT_CAPSULE_ALIGNMENT_BYTES_V1
        or flags != 0
        or reserved_word != 0
        or plan_identity_section_index != 0
        or any(raw[280:EIEL_SNAPSHOT_CAPSULE_HEADER_BYTES_V1])
    ):
        raise _capsule_error("snapshot capsule fixed header differs from V1")
    if projection_backend != EIEL_SNAPSHOT_CAPSULE_PROJECTION_HOST_CONTIGUOUS_SYNC_V1:
        raise _capsule_error("snapshot capsule projection backend is unsupported")
    if world_count == 0 or section_count < 2:
        raise _capsule_error("snapshot capsule counts are invalid")
    if (
        absolute_tick > MAX_SIGNED_INT64_V1
        or row_count > MAX_UNSIGNED_INT64_V1
        or row_count < world_count
    ):
        raise _capsule_error("snapshot capsule tick or row count is invalid")
    if section_count > (
        MAX_UNSIGNED_INT64_V1
        // EIEL_SNAPSHOT_CAPSULE_SECTION_DESCRIPTOR_BYTES_V1
    ):
        raise _capsule_error("snapshot capsule directory extent overflows")
    directory_extent = (
        section_count * EIEL_SNAPSHOT_CAPSULE_SECTION_DESCRIPTOR_BYTES_V1
    )
    if (
        directory_offset > MAX_UNSIGNED_INT64_V1 - directory_bytes
        or names_offset > MAX_UNSIGNED_INT64_V1 - names_bytes
        or trailer_offset > MAX_UNSIGNED_INT64_V1 - EIEL_SNAPSHOT_CAPSULE_TRAILER_BYTES_V1
        or names_offset + names_bytes
        > MAX_UNSIGNED_INT64_V1 - (EIEL_SNAPSHOT_CAPSULE_ALIGNMENT_BYTES_V1 - 1)
    ):
        raise _capsule_error("snapshot capsule region extent overflows")
    if (
        directory_offset != EIEL_SNAPSHOT_CAPSULE_HEADER_BYTES_V1
        or directory_bytes
        != directory_extent
        or names_offset != directory_offset + directory_bytes
        or payload_offset != _align(names_offset + names_bytes)
        or payload_offset % EIEL_SNAPSHOT_CAPSULE_ALIGNMENT_BYTES_V1
        or trailer_offset % EIEL_SNAPSHOT_CAPSULE_ALIGNMENT_BYTES_V1
        or payload_bytes != trailer_offset - payload_offset
        or capsule_bytes != trailer_offset + EIEL_SNAPSHOT_CAPSULE_TRAILER_BYTES_V1
        or capsule_bytes != len(raw)
        or not (
            EIEL_SNAPSHOT_CAPSULE_HEADER_BYTES_V1
            <= names_offset
            <= names_offset + names_bytes
            <= payload_offset
            <= trailer_offset
            < capsule_bytes
        )
    ):
        raise _capsule_error("snapshot capsule region directory differs from V1")
    if any(raw[names_offset + names_bytes : payload_offset]):
        raise _capsule_error("snapshot capsule name-table padding is nonzero")
    expected_trailer = hashlib.sha256(raw[:trailer_offset]).digest()
    if raw[trailer_offset:capsule_bytes] != expected_trailer:
        raise _capsule_error("snapshot capsule global SHA-256 differs")

    sections: list[EielSnapshotCapsuleSectionV1] = []
    name_cursor = names_offset
    payload_cursor = payload_offset
    seen_names: set[str] = set()
    for index in range(section_count):
        descriptor_offset = directory_offset + index * descriptor_bytes
        descriptor = _CAPSULE_SECTION_PREFIX_V1.unpack_from(raw, descriptor_offset)
        (
            kind,
            dtype,
            rank,
            section_flags,
            section_reserved,
            element_width,
            element_count,
            name_offset,
            name_length,
            section_payload_offset,
            payload_length,
            logical_bytes,
            *tail,
        ) = descriptor
        shape_slots = tuple(int(child) for child in tail[:8])
        payload_digest = tail[8]
        if (
            section_flags != 0
            or section_reserved != 0
            or any(raw[descriptor_offset + 168 : descriptor_offset + descriptor_bytes])
        ):
            raise _capsule_error(
                f"snapshot capsule section {index} reserved bytes differ"
            )
        if kind not in EIEL_SNAPSHOT_CAPSULE_SECTION_KIND_NAMES_V1:
            raise _capsule_error(f"snapshot capsule section {index} kind is unknown")
        dtype_row = EIEL_SNAPSHOT_CAPSULE_DTYPE_NAMES_V1.get(dtype)
        if dtype_row is None or element_width != dtype_row[1]:
            raise _capsule_error(f"snapshot capsule section {index} dtype differs")
        if rank == 0 or rank > 8 or any(shape_slots[rank:]):
            raise _capsule_error(f"snapshot capsule section {index} rank differs")
        shape = shape_slots[:rank]
        calculated_count = 1
        for dimension in shape:
            calculated_count *= dimension
            if calculated_count > MAX_UNSIGNED_INT64_V1:
                raise _capsule_error(
                    f"snapshot capsule section {index} element count overflows"
                )
        if rank == 0:
            calculated_count = 1
        if element_count > MAX_UNSIGNED_INT64_V1 // element_width:
            raise _capsule_error(
                f"snapshot capsule section {index} payload extent overflows"
            )
        if (
            element_count != calculated_count
            or payload_length != element_count * element_width
            or logical_bytes != payload_length
        ):
            raise _capsule_error(f"snapshot capsule section {index} extent differs")
        if name_offset != name_cursor or name_length == 0:
            raise _capsule_error(
                f"snapshot capsule section {index} name extent differs"
            )
        if name_offset > MAX_UNSIGNED_INT64_V1 - name_length:
            raise _capsule_error(
                f"snapshot capsule section {index} name extent overflows"
            )
        name_end = name_offset + name_length
        if name_end > names_offset + names_bytes:
            raise _capsule_error(
                f"snapshot capsule section {index} name is out of bounds"
            )
        try:
            name = raw[name_offset:name_end].decode("utf-8")
        except UnicodeDecodeError as exc:
            raise _capsule_error(
                f"snapshot capsule section {index} name is not UTF-8"
            ) from exc
        if not name or "\0" in name or name in seen_names:
            raise _capsule_error(f"snapshot capsule section {index} name differs")
        if payload_cursor > MAX_UNSIGNED_INT64_V1 - (
            EIEL_SNAPSHOT_CAPSULE_ALIGNMENT_BYTES_V1 - 1
        ):
            raise _capsule_error(
                f"snapshot capsule section {index} placement overflows"
            )
        expected_payload_offset = _align(payload_cursor)
        if section_payload_offset > MAX_UNSIGNED_INT64_V1 - payload_length:
            raise _capsule_error(
                f"snapshot capsule section {index} payload extent overflows"
            )
        if (
            section_payload_offset != expected_payload_offset
            or section_payload_offset % EIEL_SNAPSHOT_CAPSULE_ALIGNMENT_BYTES_V1
            or section_payload_offset + payload_length > trailer_offset
            or any(raw[payload_cursor:section_payload_offset])
        ):
            raise _capsule_error(f"snapshot capsule section {index} placement differs")
        payload_end = section_payload_offset + payload_length
        if (
            hashlib.sha256(raw[section_payload_offset:payload_end]).digest()
            != payload_digest
        ):
            raise _capsule_error(f"snapshot capsule section {index} SHA-256 differs")
        sections.append(
            EielSnapshotCapsuleSectionV1(
                index=index,
                name=name,
                kind=int(kind),
                kind_name=EIEL_SNAPSHOT_CAPSULE_SECTION_KIND_NAMES_V1[int(kind)],
                dtype=int(dtype),
                dtype_name=dtype_row[0],
                shape=tuple(shape),
                element_width=int(element_width),
                element_count=int(element_count),
                payload_offset=int(section_payload_offset),
                payload_length=int(payload_length),
                payload_sha256=_capsule_digest_hex(payload_digest),
            )
        )
        seen_names.add(name)
        name_cursor = name_end
        payload_cursor = payload_end
    if name_cursor != names_offset + names_bytes:
        raise _capsule_error("snapshot capsule name table has unowned bytes")
    if any(raw[payload_cursor:trailer_offset]):
        raise _capsule_error("snapshot capsule final payload padding is nonzero")

    first = sections[0]
    identity_payload = raw[
        first.payload_offset : first.payload_offset + first.payload_length
    ]
    expected_identity_payload = (
        plan_digest + schema_digest + layout_digest + control_digest
    )
    if (
        first.name != "plan_identity"
        or first.kind != EIEL_SNAPSHOT_CAPSULE_SECTION_KINDS_V1["identity"]
        or first.dtype_name != "u8"
        or first.shape != (128,)
        or identity_payload != expected_identity_payload
    ):
        raise _capsule_error("snapshot capsule plan-identity section differs")
    selected = sections[1]
    selected_payload = raw[
        selected.payload_offset : selected.payload_offset + selected.payload_length
    ]
    if (
        selected.name != "selected_world_indices"
        or selected.kind != EIEL_SNAPSHOT_CAPSULE_SECTION_KINDS_V1["identity"]
        or selected.dtype_name != "u64"
        or selected.shape != (world_count,)
    ):
        raise _capsule_error("snapshot capsule selected-world identity differs")
    observed_indices = tuple(
        int(value[0]) for value in struct.iter_unpack("<Q", selected_payload)
    )
    if len(observed_indices) != world_count or any(
        right <= left for left, right in zip(observed_indices, observed_indices[1:])
    ):
        raise _capsule_error("snapshot capsule selected-world order differs")

    plan: CompiledCudaSnapshotPlanV1 | None = None
    plan_mapping: dict[str, object] | None = None
    if expected_plan is not None:
        plan = validate_cuda_snapshot_plan_v1(expected_plan)
        plan_mapping = plan.to_mapping()
        if (
            plan_digest.hex() != plan.plan_sha256
            or schema_digest.hex() != plan_mapping["schema_sha256"]
            or layout_digest.hex() != plan_mapping["layout_sha256"]
            or control_digest.hex() != plan_mapping["native_world_control_set_sha256"]
        ):
            raise _capsule_error("snapshot capsule differs from the expected plan")
    if plan_mapping is None:
        return EielSnapshotCapsuleViewV1(
            absolute_tick=int(absolute_tick),
            world_count=int(world_count),
            row_count=int(row_count),
            section_count=int(section_count),
            plan_sha256=plan_digest.hex(),
            schema_sha256=schema_digest.hex(),
            layout_sha256=layout_digest.hex(),
            native_world_control_set_sha256=control_digest.hex(),
            projection_backend=int(projection_backend),
            capsule_bytes=int(capsule_bytes),
            capsule_sha256=_sha256_bytes(raw),
            plan=None,
            sections=tuple(sections),
        )
    boundaries = {
        int(boundary["absolute_tick"]): boundary
        for boundary in plan_mapping["snapshot_boundaries"]
        if isinstance(boundary, Mapping)
    }
    boundary = boundaries.get(int(absolute_tick))
    if boundary is None:
        raise _capsule_error("snapshot capsule tick is not requested by its plan")
    selected_indices = tuple(int(child) for child in boundary["world_indices"])
    if world_count != len(selected_indices):
        raise _capsule_error("snapshot capsule selected-world count differs")

    declared = plan_mapping["sections"]
    if not isinstance(declared, list) or len(sections) != len(declared) + 2:
        raise _capsule_error("snapshot capsule section count differs from its plan")
    selected_bytes = b"".join(struct.pack("<Q", index) for index in selected_indices)
    if selected.shape != (len(selected_indices),) or selected_payload != selected_bytes:
        raise _capsule_error("snapshot capsule selected-world identity differs")
    for actual, expected in zip(sections[2:], declared, strict=True):
        if not isinstance(expected, Mapping):  # pragma: no cover - plan validation
            raise _capsule_error("snapshot capsule plan section is invalid")
        expected_kind = EIEL_SNAPSHOT_CAPSULE_SECTION_KINDS_V1.get(
            str(expected.get("kind"))
        )
        expected_dtype = EIEL_SNAPSHOT_CAPSULE_DTYPES_V1.get(str(expected.get("dtype")))
        expected_shape = (
            len(selected_indices),
            *tuple(int(child) for child in expected.get("element_shape", ())),
        )
        expected_payload_bytes = len(selected_indices) * int(
            expected.get("bytes_per_world", -1)
        )
        if (
            actual.name != expected.get("name")
            or actual.kind != expected_kind
            or expected_dtype is None
            or actual.dtype != expected_dtype[0]
            or actual.shape != expected_shape
            or actual.payload_length != expected_payload_bytes
        ):
            raise _capsule_error(
                f"snapshot capsule section {actual.name} differs from its plan"
            )
    return EielSnapshotCapsuleViewV1(
        absolute_tick=int(absolute_tick),
        world_count=int(world_count),
        row_count=int(row_count),
        section_count=int(section_count),
        plan_sha256=plan_digest.hex(),
        schema_sha256=schema_digest.hex(),
        layout_sha256=layout_digest.hex(),
        native_world_control_set_sha256=control_digest.hex(),
        projection_backend=int(projection_backend),
        capsule_bytes=int(capsule_bytes),
        capsule_sha256=_sha256_bytes(raw),
        plan=plan,
        sections=tuple(sections),
    )


def snapshot_capsule_section_bytes_v1(
    value: bytes | bytearray | memoryview,
    name: str,
    *,
    expected_plan: (
        CompiledCudaSnapshotPlanV1 | Mapping[str, object] | bytes | None
    ) = None,
) -> bytes:
    """Return one integrity-checked typed section for parity tooling only."""

    view = validate_eiel_snapshot_capsule_v1(value, expected_plan=expected_plan)
    matches = [section for section in view.sections if section.name == name]
    if len(matches) != 1:
        raise EielSnapshotCapsuleError("snapshot capsule section name is absent")
    raw = bytes(value)
    section = matches[0]
    return raw[section.payload_offset : section.payload_offset + section.payload_length]


# Concise aliases for callers that already name the CUDA/native context.
compile_snapshot_plan_v1 = compile_cuda_snapshot_plan_v1
validate_snapshot_plan_v1 = validate_cuda_snapshot_plan_v1
validate_snapshot_capsule_v1 = validate_eiel_snapshot_capsule_v1


__all__ = [
    "CUDA_SNAPSHOT_SOURCE_LAYOUT_ID_V1",
    "CompiledCudaSnapshotPlanV1",
    "CudaSnapshotPlanError",
    "EIEL_SNAPSHOT_CAPSULE_ALIGNMENT_BYTES_V1",
    "EIEL_SNAPSHOT_CAPSULE_DTYPE_NAMES_V1",
    "EIEL_SNAPSHOT_CAPSULE_DTYPES_V1",
    "EIEL_SNAPSHOT_CAPSULE_ENDIAN_MARKER_V1",
    "EIEL_SNAPSHOT_CAPSULE_HEADER_BYTES_V1",
    "EIEL_SNAPSHOT_CAPSULE_MAGIC_V1",
    "EIEL_SNAPSHOT_CAPSULE_PROJECTION_HOST_CONTIGUOUS_SYNC_V1",
    "EIEL_SNAPSHOT_CAPSULE_SCHEMA_V1",
    "EIEL_SNAPSHOT_CAPSULE_SECTION_DESCRIPTOR_BYTES_V1",
    "EIEL_SNAPSHOT_CAPSULE_SECTION_KIND_NAMES_V1",
    "EIEL_SNAPSHOT_CAPSULE_SECTION_KINDS_V1",
    "EIEL_SNAPSHOT_CAPSULE_TRAILER_BYTES_V1",
    "EIEL_SNAPSHOT_CAPSULE_VERSION_V1",
    "EielSnapshotCapsuleError",
    "EielSnapshotCapsuleSectionV1",
    "EielSnapshotCapsuleViewV1",
    "NATIVE_SNAPSHOT_PLAN_SCHEMA_V1",
    "NATIVE_SNAPSHOT_STATE_CAPTURE_CANONICAL_PREFIX_V1",
    "NATIVE_SNAPSHOT_STATE_CAPTURE_FULL_RESIDENT_V1",
    "NATIVE_SNAPSHOT_STATE_CAPTURE_TRANSPORT_SCHEMA_V1",
    "SNAPSHOT_CATALOG_LAWS_V1",
    "SNAPSHOT_ORDERING_LAWS_V1",
    "SNAPSHOT_SELECTION_LAWS_V1",
    "compile_cuda_snapshot_plan_v1",
    "compile_snapshot_plan_v1",
    "snapshot_capsule_section_bytes_v1",
    "validate_cuda_snapshot_plan_v1",
    "validate_eiel_snapshot_capsule_v1",
    "validate_snapshot_capsule_v1",
    "validate_snapshot_plan_v1",
]
