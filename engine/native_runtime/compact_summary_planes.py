"""Fixed host-plane utilities for compact generation-summary ABI v2.

The module freezes primitive storage and validation. It does not project wire
rows or choose semantic authorities. The sole bounded text exception is the
proven-unbounded dominant lineage identifier.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Final

import numpy as np


ABI_VERSION: Final[str] = "eiel_compact_summary_native_abi_v2"
WIRE_SCHEMA_VERSION: Final[str] = "eiel_compact_generation_summary_v1"
DEFINED_MASK_WORD_COUNT: Final[int] = 1
TEXT_PAYLOAD_BYTES: Final[int] = 64
TEXT_RECORD_BYTES: Final[int] = 65
INT64_MAX: Final[int] = np.iinfo(np.int64).max
MAX_GENERATION_START: Final[int] = INT64_MAX - 1

STATUS_FIELDS: Final[tuple[str, ...]] = (
    "summary_status",
    "world",
    "turnover.status",
    "rng.authority_complete",
    "rng.export_non_consuming",
    "terminal.status",
    "terminal.recoverable_error_status",
    "terminal.sticky_failure",
    "checkpoint_due",
    "lineage_metrics.lineage_collapse_warning",
    "terminal.recoverable_error_code",
    "terminal.failure_stage",
)

INTEGER_FIELDS: Final[tuple[str, ...]] = (
    "generation_start.generation",
    "generation_start.tick",
    "generation_end.generation",
    "generation_end.tick_exclusive",
    "next_generation_start.generation",
    "next_generation_start.tick",
    "alive",
    "births",
    "failed_births",
    "deaths",
    "deaths_by_class.energy_depleted",
    "deaths_by_class.health_depleted",
    "turnover.count",
    "events.cursor_start",
    "events.cursor_pre_turnover",
    "events.cursor_end",
    "events.authoritative_cursor",
    "events.event_row_count",
    "events.generation_event_count",
    "events.turnover_event_count",
    "events.generation_counts_by_type.birth_committed",
    "events.generation_counts_by_type.mutation",
    "events.generation_counts_by_type.birth_failed",
    "events.generation_counts_by_type.death_committed",
    "events.turnover_counts_by_type.birth_committed",
    "events.turnover_counts_by_type.mutation",
    "events.turnover_counts_by_type.birth_failed",
    "events.turnover_counts_by_type.death_committed",
    "identity.population_next_id",
    "identity.active_agent_count",
    "identity.active_agent_id_min",
    "identity.active_agent_id_max",
    "identity.free_slot_count",
    "rng.policy_mutation_tape_cursor",
    "terminal.failure_generation",
    "terminal.failure_tick",
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
    "behavior_metrics.row_count",
    "behavior_metrics.cohort_size",
    "behavior_metrics.cohort_survivors",
    "behavior_metrics.action_total",
    "behavior_metrics.move_total",
    "behavior_metrics.move_n",
    "behavior_metrics.move_s",
    "behavior_metrics.move_e",
    "behavior_metrics.move_w",
    "behavior_metrics.gather",
    "behavior_metrics.rest",
    "behavior_metrics.attack",
    "lineage_metrics.record_count",
    "lineage_metrics.active_lineages",
    "lineage_metrics.rare_persistent_lineages",
    "lineage_metrics.lineage_collapse_streak",
    "patch_metrics.patch_movement_energy_event_count",
    "patch_metrics.patch_spawn_sync_rows",
    "patch_metrics.typed_death_event_rows",
    "patch_metrics.env_rng_padding_draws",
)

FLOAT_FIELDS: Final[tuple[str, ...]] = (
    "generation_metrics.peak_injury",
    "lineage_metrics.dominant_lineage_population_share",
    "lineage_metrics.top3_lineage_population_share",
    "lineage_metrics.lineage_population_entropy",
    "lineage_metrics.lineage_reproduction_entropy",
    "lineage_metrics.dominant_lineage_reproduction_share",
    "patch_metrics.patch_food_harvest_total",
    "patch_metrics.patch_scent_emit_total",
    "patch_metrics.patch_waste_emit_total",
    "patch_metrics.patch_movement_energy_total",
    "patch_metrics.energy_collected_delta",
    "patch_metrics.patch_scent_move_queued_total",
    "patch_metrics.patch_waste_move_queued_total",
    "patch_metrics.patch_scent_gather_queued_total",
    "patch_metrics.patch_waste_gather_queued_total",
    "patch_metrics.patch_movement_energy_extra_total",
    "patch_metrics.patch_scent_emitted_total",
    "patch_metrics.patch_waste_emitted_total",
    "patch_metrics.patch_food_depleted_frac",
    "patch_metrics.patch_food_full_frac",
    "patch_metrics.patch_density_hotspot_frac",
)

DIGEST_FIELDS: Final[tuple[str, ...]] = (
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

TEXT_FIELDS: Final[tuple[str, ...]] = (
    "lineage_metrics.dominant_lineage_id",
)

NON_PATCH_DEFINED_FIELDS: Final[tuple[str, ...]] = (
    "identity.active_agent_id_min",
    "identity.active_agent_id_max",
    "terminal.recoverable_error_code",
    "terminal.recoverable_error_digest_sha256",
    "terminal.failure_generation",
    "terminal.failure_tick",
    "terminal.failure_stage",
    "lineage_metrics.active_lineages",
    "lineage_metrics.dominant_lineage_id",
    "lineage_metrics.dominant_lineage_population_share",
    "lineage_metrics.top3_lineage_population_share",
    "lineage_metrics.lineage_population_entropy",
    "lineage_metrics.lineage_reproduction_entropy",
    "lineage_metrics.dominant_lineage_reproduction_share",
    "lineage_metrics.rare_persistent_lineages",
    "lineage_metrics.lineage_collapse_warning",
    "lineage_metrics.lineage_collapse_streak",
)
PATCH_OBJECT_FIELD: Final[str] = "patch_metrics"
PATCH_DEFINED_FIELDS: Final[tuple[str, ...]] = (
    "patch_metrics.patch_spawn_sync_rows",
    "patch_metrics.typed_death_event_rows",
    "patch_metrics.env_rng_padding_draws",
    "patch_metrics.energy_collected_delta",
    "patch_metrics.patch_scent_move_queued_total",
    "patch_metrics.patch_waste_move_queued_total",
    "patch_metrics.patch_scent_gather_queued_total",
    "patch_metrics.patch_waste_gather_queued_total",
    "patch_metrics.patch_movement_energy_extra_total",
    "patch_metrics.patch_scent_emitted_total",
    "patch_metrics.patch_waste_emitted_total",
    "patch_metrics.patch_food_depleted_frac",
    "patch_metrics.patch_food_full_frac",
    "patch_metrics.patch_density_hotspot_frac",
)
DEFINED_FIELDS: Final[tuple[str, ...]] = (
    *NON_PATCH_DEFINED_FIELDS,
    PATCH_OBJECT_FIELD,
    *PATCH_DEFINED_FIELDS,
)


def _index_registry(fields: tuple[str, ...]) -> Mapping[str, int]:
    return MappingProxyType({field: index for index, field in enumerate(fields)})


STATUS_FIELD_INDEX: Final[Mapping[str, int]] = _index_registry(STATUS_FIELDS)
INTEGER_FIELD_INDEX: Final[Mapping[str, int]] = _index_registry(INTEGER_FIELDS)
FLOAT_FIELD_INDEX: Final[Mapping[str, int]] = _index_registry(FLOAT_FIELDS)
DIGEST_FIELD_INDEX: Final[Mapping[str, int]] = _index_registry(DIGEST_FIELDS)
TEXT_FIELD_INDEX: Final[Mapping[str, int]] = _index_registry(TEXT_FIELDS)
DEFINED_FIELD_BITS: Final[Mapping[str, int]] = _index_registry(DEFINED_FIELDS)


def _codebook(*values: Any) -> Mapping[Any, int]:
    return MappingProxyType({value: index + 1 for index, value in enumerate(values)})


RECOVERABLE_ERROR_CODES: Final[tuple[str, ...]] = (
    "exact_reporting_failed",
    "checkpoint_persistence_failed",
    "output_flush_failed",
)
FAILURE_STAGES: Final[tuple[str, ...]] = (
    "tick_core_and_birth_plan",
    "birth_prepare",
    "birth_transaction_and_settlement",
    "birth_event_append",
    "import_validation",
    "import_copy",
    "run_request_validation",
    "generation",
    "turnover",
    "snapshot",
    "cancellation",
    "finalization",
    "api_state",
    "typed_lifecycle_public_output",
    "behavior_public_output",
    "generation_result_validation",
)
STATUS_CODEBOOKS: Final[Mapping[str, Mapping[Any, int]]] = MappingProxyType(
    {
        "summary_status": _codebook("complete"),
        "world": _codebook("flat", "patch"),
        "turnover.status": _codebook("committed"),
        "rng.authority_complete": _codebook(False, True),
        "rng.export_non_consuming": _codebook(False, True),
        "terminal.status": _codebook(
            "not_terminal", "run_complete", "cancelled", "failed"
        ),
        "terminal.recoverable_error_status": _codebook("none", "present"),
        "terminal.sticky_failure": _codebook(False, True),
        "checkpoint_due": _codebook(False, True),
        "lineage_metrics.lineage_collapse_warning": _codebook(False, True),
        "terminal.recoverable_error_code": _codebook(*RECOVERABLE_ERROR_CODES),
        "terminal.failure_stage": _codebook(*FAILURE_STAGES),
    }
)

PATCH_INTEGER_FIELDS: Final[tuple[str, ...]] = tuple(
    field for field in INTEGER_FIELDS if field.startswith("patch_metrics.")
)
PATCH_FLOAT_FIELDS: Final[tuple[str, ...]] = tuple(
    field for field in FLOAT_FIELDS if field.startswith("patch_metrics.")
)
RATIO_FLOAT_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "lineage_metrics.dominant_lineage_population_share",
        "lineage_metrics.top3_lineage_population_share",
        "lineage_metrics.dominant_lineage_reproduction_share",
        "patch_metrics.patch_food_depleted_frac",
        "patch_metrics.patch_food_full_frac",
        "patch_metrics.patch_density_hotspot_frac",
    }
)
NONNEGATIVE_FLOAT_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "generation_metrics.peak_injury",
        "patch_metrics.patch_food_harvest_total",
        "patch_metrics.patch_scent_emit_total",
        "patch_metrics.patch_waste_emit_total",
    }
)
POSITIVE_INTEGER_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "generation_end.tick_exclusive",
        "next_generation_start.generation",
        "turnover.count",
        "identity.population_next_id",
        "generation_metrics.capacity",
    }
)

LOGICAL_BYTES_PER_GENERATION_CELL: Final[int] = (
    1 + DEFINED_MASK_WORD_COUNT * 8 + len(STATUS_FIELDS) * 4
    + len(INTEGER_FIELDS) * 8 + len(FLOAT_FIELDS) * 8
    + len(DIGEST_FIELDS) * 32 + len(TEXT_FIELDS) * TEXT_RECORD_BYTES
)


@dataclass(frozen=True)
class CompactSummaryPlanes:
    validity: np.ndarray
    defined_mask: np.ndarray
    status_codes: np.ndarray
    integer_values: np.ndarray
    float_values: np.ndarray
    digest_values: np.ndarray
    text_values: np.ndarray

    def as_mapping(self) -> Mapping[str, np.ndarray]:
        return MappingProxyType(
            {
                "validity": self.validity,
                "defined_mask": self.defined_mask,
                "status_codes": self.status_codes,
                "integer_values": self.integer_values,
                "float_values": self.float_values,
                "digest_values": self.digest_values,
                "text_values": self.text_values,
            }
        )


_PLANE_NAMES: Final[tuple[str, ...]] = (
    "validity",
    "defined_mask",
    "status_codes",
    "integer_values",
    "float_values",
    "digest_values",
    "text_values",
)
_PLANE_DTYPES: Final[Mapping[str, np.dtype[Any]]] = MappingProxyType(
    {
        "validity": np.dtype(np.uint8),
        "defined_mask": np.dtype(np.uint64),
        "status_codes": np.dtype(np.int32),
        "integer_values": np.dtype(np.int64),
        "float_values": np.dtype(np.float64),
        "digest_values": np.dtype(np.uint8),
        "text_values": np.dtype(np.uint8),
    }
)


def _positive_dimension(value: Any, *, name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an int")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    if value > np.iinfo(np.intp).max:
        raise OverflowError(f"{name} exceeds platform index range")
    return value


def allocate_compact_summary_planes(
    generation_count: int, cell_count: int
) -> CompactSummaryPlanes:
    generations = _positive_dimension(generation_count, name="generation_count")
    cells = _positive_dimension(cell_count, name="cell_count")
    return CompactSummaryPlanes(
        validity=np.zeros((generations, cells), dtype=np.uint8),
        defined_mask=np.zeros(
            (generations, cells, DEFINED_MASK_WORD_COUNT), dtype=np.uint64
        ),
        status_codes=np.zeros(
            (generations, cells, len(STATUS_FIELDS)), dtype=np.int32
        ),
        integer_values=np.zeros(
            (generations, cells, len(INTEGER_FIELDS)), dtype=np.int64
        ),
        float_values=np.zeros(
            (generations, cells, len(FLOAT_FIELDS)), dtype=np.float64
        ),
        digest_values=np.zeros(
            (generations, cells, len(DIGEST_FIELDS), 32), dtype=np.uint8
        ),
        text_values=np.zeros(
            (generations, cells, len(TEXT_FIELDS), TEXT_RECORD_BYTES),
            dtype=np.uint8,
        ),
    )


def _plane_mapping(planes: Any) -> Mapping[str, np.ndarray]:
    if isinstance(planes, CompactSummaryPlanes):
        return planes.as_mapping()
    if not isinstance(planes, Mapping):
        raise TypeError("planes must be CompactSummaryPlanes or a mapping")
    if set(planes) != set(_PLANE_NAMES):
        missing = sorted(set(_PLANE_NAMES) - set(planes))
        extra = sorted(set(planes) - set(_PLANE_NAMES), key=repr)
        raise ValueError(f"plane names mismatch: missing={missing} extra={extra}")
    return planes


def validate_compact_summary_plane_layout(
    planes: Any,
    *,
    generation_count: int | None = None,
    cell_count: int | None = None,
) -> tuple[int, int]:
    values = _plane_mapping(planes)
    validity = values["validity"]
    if not isinstance(validity, np.ndarray) or validity.ndim != 2:
        raise TypeError("validity must be a rank-2 NumPy array")
    generations, cells = validity.shape
    _positive_dimension(int(generations), name="generation_count")
    _positive_dimension(int(cells), name="cell_count")
    if generation_count is not None and generations != _positive_dimension(
        generation_count, name="generation_count"
    ):
        raise ValueError("generation_count does not match plane shape")
    if cell_count is not None and cells != _positive_dimension(
        cell_count, name="cell_count"
    ):
        raise ValueError("cell_count does not match plane shape")
    shapes = {
        "validity": (generations, cells),
        "defined_mask": (generations, cells, DEFINED_MASK_WORD_COUNT),
        "status_codes": (generations, cells, len(STATUS_FIELDS)),
        "integer_values": (generations, cells, len(INTEGER_FIELDS)),
        "float_values": (generations, cells, len(FLOAT_FIELDS)),
        "digest_values": (generations, cells, len(DIGEST_FIELDS), 32),
        "text_values": (
            generations, cells, len(TEXT_FIELDS), TEXT_RECORD_BYTES
        ),
    }
    for name in _PLANE_NAMES:
        array = values[name]
        if not isinstance(array, np.ndarray):
            raise TypeError(f"{name} must be a NumPy array")
        if array.dtype != _PLANE_DTYPES[name]:
            raise ValueError(
                f"{name} dtype must be {_PLANE_DTYPES[name]}, got {array.dtype}"
            )
        if array.shape != shapes[name]:
            raise ValueError(f"{name} shape must be {shapes[name]}, got {array.shape}")
        if not array.flags.c_contiguous:
            raise ValueError(f"{name} must be C-contiguous")
        if not array.flags.aligned:
            raise ValueError(f"{name} must be aligned")
        if not array.flags.writeable:
            raise ValueError(f"{name} must be writeable")
    return int(generations), int(cells)


def status_code(field: str, value: Any) -> int:
    try:
        codebook = STATUS_CODEBOOKS[field]
    except KeyError as exc:
        raise KeyError(f"unknown status field: {field}") from exc
    for candidate, code in codebook.items():
        if type(value) is type(candidate) and value == candidate:
            return code
    raise ValueError(f"unsupported {field} value: {value!r}")


def status_value(field: str, code: int) -> Any:
    if type(code) is not int:
        raise TypeError("status code must be an int")
    try:
        codebook = STATUS_CODEBOOKS[field]
    except KeyError as exc:
        raise KeyError(f"unknown status field: {field}") from exc
    for value, candidate in codebook.items():
        if candidate == code:
            return value
    raise ValueError(f"unknown {field} status code: {code}")


def _strict_mask_word(mask: Any) -> int:
    if isinstance(mask, np.uint64):
        return int(mask)
    if type(mask) is not int:
        raise TypeError("defined mask word must be an int or numpy.uint64")
    if mask < 0 or mask > np.iinfo(np.uint64).max:
        raise ValueError("defined mask word must fit uint64")
    return mask


def validate_defined_mask(mask: Any) -> int:
    value = _strict_mask_word(mask)
    if value >> len(DEFINED_FIELDS):
        raise ValueError("defined mask has reserved high bits set")
    patch_bit = 1 << DEFINED_FIELD_BITS[PATCH_OBJECT_FIELD]
    patch_child_bits = sum(
        1 << DEFINED_FIELD_BITS[field] for field in PATCH_DEFINED_FIELDS
    )
    if not value & patch_bit and value & patch_child_bits:
        raise ValueError("Patch child bits are reserved when patch_metrics is null")
    return value


def is_field_defined(mask: Any, field: str) -> bool:
    try:
        bit = DEFINED_FIELD_BITS[field]
    except KeyError as exc:
        raise KeyError(f"unknown nullable/optional field: {field}") from exc
    return bool(_strict_mask_word(mask) & (1 << bit))


def set_field_defined(mask: Any, field: str, defined: bool) -> int:
    if type(defined) is not bool:
        raise TypeError("defined must be a bool")
    try:
        bit = DEFINED_FIELD_BITS[field]
    except KeyError as exc:
        raise KeyError(f"unknown nullable/optional field: {field}") from exc
    value = _strict_mask_word(mask)
    candidate = value | (1 << bit) if defined else value & ~(1 << bit)
    return validate_defined_mask(candidate)


def undefined_fields_from_mask(mask: Any) -> tuple[str, ...]:
    value = validate_defined_mask(mask)
    undefined = [
        field for field in NON_PATCH_DEFINED_FIELDS
        if not is_field_defined(value, field)
    ]
    if not is_field_defined(value, PATCH_OBJECT_FIELD):
        undefined.append(PATCH_OBJECT_FIELD)
    else:
        undefined.extend(
            field for field in PATCH_DEFINED_FIELDS
            if not is_field_defined(value, field)
        )
    return tuple(sorted(undefined))


def encode_text_value(value: str) -> np.ndarray:
    if type(value) is not str:
        raise TypeError("text value must be a string")
    if not value:
        raise ValueError("dominant lineage identifier must be non-empty")
    try:
        payload = value.encode("utf-8", errors="strict")
    except UnicodeEncodeError as exc:
        raise ValueError("text value is not valid UTF-8") from exc
    if len(payload) > TEXT_PAYLOAD_BYTES:
        raise ValueError(f"text value exceeds {TEXT_PAYLOAD_BYTES} UTF-8 bytes")
    record = np.zeros((TEXT_RECORD_BYTES,), dtype=np.uint8)
    record[0] = len(payload)
    record[1 : 1 + len(payload)] = np.frombuffer(payload, dtype=np.uint8)
    return record


def decode_text_value(record: Any) -> str:
    if not isinstance(record, np.ndarray):
        raise TypeError("text value record must be a NumPy array")
    if record.dtype != np.dtype(np.uint8):
        raise ValueError("text value record dtype must be uint8")
    if record.shape != (TEXT_RECORD_BYTES,):
        raise ValueError(f"text value record shape must be ({TEXT_RECORD_BYTES},)")
    if not record.flags.c_contiguous:
        raise ValueError("text value record must be C-contiguous")
    length = int(record[0])
    if length == 0:
        raise ValueError("defined dominant lineage identifier must be non-empty")
    if length > TEXT_PAYLOAD_BYTES:
        raise ValueError("text value length exceeds payload capacity")
    if bool(np.any(record[1 + length :])):
        raise ValueError("text value tail must be zero-filled")
    try:
        return record[1 : 1 + length].tobytes().decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise ValueError("text value payload is not valid UTF-8") from exc


def float64_to_bits(value: Any) -> int:
    if type(value) not in {float, np.float64}:
        raise TypeError("float value must be float or numpy.float64")
    if not math.isfinite(float(value)):
        raise ValueError("float value must be finite")
    return int(np.asarray([value], dtype=np.float64).view(np.uint64)[0])


def float64_from_bits(bits: Any) -> float:
    if isinstance(bits, np.uint64):
        value = int(bits)
    elif type(bits) is int:
        value = bits
    else:
        raise TypeError("float bits must be an int or numpy.uint64")
    if value < 0 or value > np.iinfo(np.uint64).max:
        raise ValueError("float bits must fit uint64")
    result = np.asarray([value], dtype=np.uint64).view(np.float64)[0]
    if not math.isfinite(float(result)):
        raise ValueError("float bits encode a non-finite value")
    return float(result)


def _raw_nonzero(array: np.ndarray) -> bool:
    return bool(np.any(array.view(np.uint8)))


def _field_storage(
    values: Mapping[str, np.ndarray], generation: int, cell: int, field: str
) -> np.ndarray:
    if field in INTEGER_FIELD_INDEX:
        index = INTEGER_FIELD_INDEX[field]
        return values["integer_values"][generation, cell, index : index + 1]
    if field in FLOAT_FIELD_INDEX:
        index = FLOAT_FIELD_INDEX[field]
        return values["float_values"][generation, cell, index : index + 1]
    if field in DIGEST_FIELD_INDEX:
        return values["digest_values"][
            generation, cell, DIGEST_FIELD_INDEX[field]
        ]
    if field in TEXT_FIELD_INDEX:
        return values["text_values"][generation, cell, TEXT_FIELD_INDEX[field]]
    if field in STATUS_FIELD_INDEX:
        index = STATUS_FIELD_INDEX[field]
        return values["status_codes"][generation, cell, index : index + 1]
    raise KeyError(f"no primitive storage for defined field: {field}")


def _validate_status_row(codes: np.ndarray, mask: int) -> None:
    nullable_statuses = {
        "lineage_metrics.lineage_collapse_warning",
        "terminal.recoverable_error_code",
        "terminal.failure_stage",
    }
    for index, field in enumerate(STATUS_FIELDS):
        code = int(codes[index])
        if field in nullable_statuses and not is_field_defined(mask, field):
            if code != 0:
                raise ValueError(f"undefined {field} must use reserved code 0")
            continue
        status_value(field, code)


def _validate_terminal_row(
    values: Mapping[str, np.ndarray], generation: int, cell: int, mask: int
) -> None:
    codes = values["status_codes"][generation, cell]
    recoverable = status_value(
        "terminal.recoverable_error_status",
        int(codes[STATUS_FIELD_INDEX["terminal.recoverable_error_status"]]),
    )
    recoverable_code = is_field_defined(mask, "terminal.recoverable_error_code")
    recoverable_digest = is_field_defined(
        mask, "terminal.recoverable_error_digest_sha256"
    )
    if recoverable == "none" and (recoverable_code or recoverable_digest):
        raise ValueError("recoverable error fields must be undefined for status none")
    if recoverable == "present" and not (recoverable_code and recoverable_digest):
        raise ValueError("recoverable error code and digest are required for present")

    terminal = status_value(
        "terminal.status", int(codes[STATUS_FIELD_INDEX["terminal.status"]])
    )
    sticky = status_value(
        "terminal.sticky_failure",
        int(codes[STATUS_FIELD_INDEX["terminal.sticky_failure"]]),
    )
    failure_fields = (
        "terminal.failure_generation",
        "terminal.failure_tick",
        "terminal.failure_stage",
    )
    failure_defined = tuple(is_field_defined(mask, field) for field in failure_fields)
    if terminal == "failed":
        if sticky is not True or not all(failure_defined):
            raise ValueError("failed terminal status requires sticky failure authority")
    elif sticky is not False or any(failure_defined):
        raise ValueError("nonfailed terminal status forbids sticky failure authority")


def validate_compact_summary_planes(
    planes: Any,
    *,
    generation_count: int | None = None,
    cell_count: int | None = None,
) -> tuple[int, int]:
    generations, cells = validate_compact_summary_plane_layout(
        planes, generation_count=generation_count, cell_count=cell_count
    )
    values = _plane_mapping(planes)
    if bool(np.any(values["validity"] > 1)):
        raise ValueError("validity entries must be 0 or 1")
    for generation in range(generations):
        for cell in range(cells):
            row_arrays = tuple(
                values[name][generation, cell] for name in _PLANE_NAMES[1:]
            )
            if int(values["validity"][generation, cell]) == 0:
                if any(_raw_nonzero(array) for array in row_arrays):
                    raise ValueError(
                        f"invalid row [{generation},{cell}] must remain a frozen no-op"
                    )
                continue

            mask = validate_defined_mask(values["defined_mask"][generation, cell, 0])
            codes = values["status_codes"][generation, cell]
            _validate_status_row(codes, mask)
            world = status_value("world", int(codes[STATUS_FIELD_INDEX["world"]]))
            patch_present = is_field_defined(mask, PATCH_OBJECT_FIELD)
            if (world == "patch") != patch_present:
                raise ValueError("world and patch_metrics presence disagree")

            integers = values["integer_values"][generation, cell]
            if bool(np.any(integers < 0)):
                raise ValueError("valid integer values must be nonnegative")
            for field in POSITIVE_INTEGER_FIELDS:
                if integers[INTEGER_FIELD_INDEX[field]] <= 0:
                    raise ValueError(f"{field} must be positive")
            if integers[INTEGER_FIELD_INDEX["generation_start.generation"]] > MAX_GENERATION_START:
                raise ValueError("generation_start.generation cannot be incremented in int64")

            floats = values["float_values"][generation, cell]
            if not bool(np.all(np.isfinite(floats))):
                raise ValueError("valid float values must be finite")
            for field in NONNEGATIVE_FLOAT_FIELDS:
                if floats[FLOAT_FIELD_INDEX[field]] < 0:
                    raise ValueError(f"{field} must be nonnegative")
            for field in RATIO_FLOAT_FIELDS:
                if is_field_defined(mask, field):
                    value = floats[FLOAT_FIELD_INDEX[field]]
                    if value < 0 or value > 1:
                        raise ValueError(f"{field} must be in [0, 1]")

            for field in DEFINED_FIELDS:
                if field == PATCH_OBJECT_FIELD or is_field_defined(mask, field):
                    continue
                if _raw_nonzero(_field_storage(values, generation, cell, field)):
                    raise ValueError(f"undefined {field} must retain its no-op value")

            if not patch_present:
                for field in PATCH_INTEGER_FIELDS:
                    if integers[INTEGER_FIELD_INDEX[field]] != 0:
                        raise ValueError(f"Flat row must zero {field}")
                for field in PATCH_FLOAT_FIELDS:
                    index = FLOAT_FIELD_INDEX[field]
                    if _raw_nonzero(floats[index : index + 1]):
                        raise ValueError(f"Flat row must bit-zero {field}")

            if is_field_defined(mask, TEXT_FIELDS[0]):
                decode_text_value(values["text_values"][generation, cell, 0])
            elif _raw_nonzero(values["text_values"][generation, cell, 0]):
                raise ValueError("undefined dominant lineage ID must be a no-op")
            _validate_terminal_row(values, generation, cell, mask)
    return generations, cells


__all__ = [
    "ABI_VERSION",
    "CompactSummaryPlanes",
    "DEFINED_FIELDS",
    "DEFINED_FIELD_BITS",
    "DEFINED_MASK_WORD_COUNT",
    "DIGEST_FIELDS",
    "DIGEST_FIELD_INDEX",
    "FAILURE_STAGES",
    "FLOAT_FIELDS",
    "FLOAT_FIELD_INDEX",
    "INTEGER_FIELDS",
    "INTEGER_FIELD_INDEX",
    "INT64_MAX",
    "LOGICAL_BYTES_PER_GENERATION_CELL",
    "MAX_GENERATION_START",
    "NON_PATCH_DEFINED_FIELDS",
    "PATCH_DEFINED_FIELDS",
    "PATCH_OBJECT_FIELD",
    "RECOVERABLE_ERROR_CODES",
    "STATUS_CODEBOOKS",
    "STATUS_FIELDS",
    "STATUS_FIELD_INDEX",
    "TEXT_FIELDS",
    "TEXT_FIELD_INDEX",
    "TEXT_PAYLOAD_BYTES",
    "TEXT_RECORD_BYTES",
    "WIRE_SCHEMA_VERSION",
    "allocate_compact_summary_planes",
    "decode_text_value",
    "encode_text_value",
    "float64_from_bits",
    "float64_to_bits",
    "is_field_defined",
    "set_field_defined",
    "status_code",
    "status_value",
    "undefined_fields_from_mask",
    "validate_compact_summary_plane_layout",
    "validate_compact_summary_planes",
    "validate_defined_mask",
]
