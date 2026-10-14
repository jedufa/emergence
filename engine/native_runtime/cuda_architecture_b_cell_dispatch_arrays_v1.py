from __future__ import annotations

"""Strict metadata-only array lowering for Architecture B cell dispatch plans.

The resulting CPU NumPy arrays describe one future native controller call. This
module does not invoke C++, CUDA, a live controller, or a simulation.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from types import MappingProxyType
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

from engine.native_runtime.cuda_architecture_b_cell_schedule_v1 import (
    COMMAND_CANCEL_V1,
    COMMAND_CHECKPOINT_AND_FORK_V1,
    COMMAND_CHECKPOINT_V1,
    DISPATCH_PLAN_SCHEMA_VERSION_V1,
    CellDispatchPlanReceiptV1,
)
from engine.perf.cuda_architecture_b_checkpoint_branch import (
    canonical_branch_json_bytes,
)


CELL_DISPATCH_ARRAYS_SCHEMA_VERSION_V1: Final[str] = (
    "eiel_cuda_architecture_b_cell_dispatch_arrays_v1"
)
CELL_DISPATCH_ARRAYS_ABI_VERSION_V1: Final[int] = 1
CELL_DISPATCH_ARRAYS_DIGEST_DOMAIN_V1: Final[bytes] = (
    b"eiel.cuda_architecture_b.cell_dispatch_arrays.content.v1\x00"
)

STOP_REASON_PENDING_BOUNDARY_V1: Final[int] = 1 << 0
STOP_REASON_GENERATION_LIMIT_V1: Final[int] = 1 << 1
STOP_REASON_MAX_DISPATCH_V1: Final[int] = 1 << 2
STOP_REASON_ACCEPTED_GENERATION_CHUNK_V1: Final[int] = 1 << 3

BLOCKED_REASON_PENDING_BOUNDARY_V1: Final[int] = 1
BLOCKED_REASON_GENERATION_LIMIT_V1: Final[int] = 2

PENDING_COMMAND_NONE_V1: Final[int] = 0
PENDING_COMMAND_CHECKPOINT_V1: Final[int] = 1
PENDING_COMMAND_CHECKPOINT_AND_FORK_V1: Final[int] = 2
PENDING_COMMAND_CANCEL_V1: Final[int] = 3

ARRAY_ORDER_V1: Final[tuple[str, ...]] = (
    "cohort_offsets",
    "cohort_start_generations",
    "cohort_generation_counts",
    "cohort_stop_reason_masks",
    "cell_indices",
    "stable_cell_ids",
    "cell_stop_flags",
    "terminal_cell_indices",
    "blocked_cell_indices",
    "blocked_stable_cell_ids",
    "blocked_current_generations",
    "blocked_reason_codes",
    "blocked_pending_boundary_present",
    "blocked_pending_boundaries",
    "blocked_pending_command_kind_codes",
)

_INT64_DTYPE: Final[np.dtype[Any]] = np.dtype("<i8")
_UINT8_DTYPE: Final[np.dtype[Any]] = np.dtype("u1")
_MAX_INT64: Final[int] = (1 << 63) - 1

_PLAN_FIELDS: Final[tuple[str, ...]] = (
    "schema_version",
    "controller_id",
    "schedule_sha256",
    "max_generations_per_dispatch",
    "cohort_count",
    "cohorts",
    "terminal_cell_indices",
    "blocked_active_cells",
    "selection_policy",
    "contract",
    "plan_sha256",
)
_COHORT_FIELDS: Final[tuple[str, ...]] = (
    "cohort_index",
    "physical_batch_cell_count",
    "cell_indices",
    "stable_cell_ids",
    "start_generation",
    "generation_count",
    "stop_reason",
    "stop_cell_indices",
    "shared_generation_coordinate",
)
_BLOCKED_FIELDS: Final[tuple[str, ...]] = (
    "cell_index",
    "stable_cell_id",
    "current_generation",
    "reason",
    "pending_boundary",
    "pending_command_kind",
)
_SELECTION_POLICY: Final[Mapping[str, Any]] = MappingProxyType(
    {
        "primary": "largest_safe_same_generation_physical_batch",
        "secondary": (
            "largest_accepted_chunk_preserving_exact_boundary_decomposition"
        ),
        "cohort_order": "current_generation_ascending",
        "cell_order": "stable_cell_id_then_cell_index",
    }
)
_SOURCE_CONTRACT: Final[Mapping[str, Any]] = MappingProxyType(
    {
        "read_only": True,
        "idempotent": True,
        "metadata_only": True,
        "shared_generation_count_per_cohort": True,
        "live_controller_integration": False,
        "cuda_execution_performed": False,
        "simulation_execution_performed": False,
        "performance_claimed": False,
        "production_promotion_claimed": False,
    }
)
_STOP_REASON_BITS: Final[Mapping[str, int]] = MappingProxyType(
    {
        "pending_boundary": STOP_REASON_PENDING_BOUNDARY_V1,
        "generation_limit": STOP_REASON_GENERATION_LIMIT_V1,
        "max_generations_per_dispatch": STOP_REASON_MAX_DISPATCH_V1,
        "accepted_generation_chunk": STOP_REASON_ACCEPTED_GENERATION_CHUNK_V1,
    }
)
_BLOCKED_REASON_CODES: Final[Mapping[str, int]] = MappingProxyType(
    {
        "pending_boundary": BLOCKED_REASON_PENDING_BOUNDARY_V1,
        "generation_limit": BLOCKED_REASON_GENERATION_LIMIT_V1,
    }
)
_PENDING_COMMAND_KIND_CODES: Final[Mapping[str | None, int]] = MappingProxyType(
    {
        None: PENDING_COMMAND_NONE_V1,
        COMMAND_CHECKPOINT_V1: PENDING_COMMAND_CHECKPOINT_V1,
        COMMAND_CHECKPOINT_AND_FORK_V1: PENDING_COMMAND_CHECKPOINT_AND_FORK_V1,
        COMMAND_CANCEL_V1: PENDING_COMMAND_CANCEL_V1,
    }
)
_ARRAY_DTYPES: Final[Mapping[str, np.dtype[Any]]] = MappingProxyType(
    {
        name: (
            _UINT8_DTYPE
            if name
            in {
                "cohort_stop_reason_masks",
                "cell_stop_flags",
                "blocked_reason_codes",
                "blocked_pending_boundary_present",
                "blocked_pending_command_kind_codes",
            }
            else _INT64_DTYPE
        )
        for name in ARRAY_ORDER_V1
    }
)


class CellDispatchArraysContractError(ValueError):
    """A source dispatch plan or lowered array bundle violates the v1 contract."""


def _fail(path: str, message: str) -> None:
    raise CellDispatchArraysContractError(f"{path}: {message}")


def _exact_mapping(
    value: Any, fields: tuple[str, ...], *, path: str
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(type(key) is not str for key in value):
        _fail(path, "must be a string-keyed mapping")
    missing = sorted(set(fields) - set(value))
    extra = sorted(set(value) - set(fields))
    if missing or extra:
        _fail(path, f"field mismatch: missing={missing}, unknown={extra}")
    return value


def _exact_list(value: Any, *, path: str) -> list[Any]:
    if type(value) is not list:
        _fail(path, "must be a list")
    return value


def _exact_int(value: Any, *, path: str, minimum: int = 0) -> int:
    if type(value) is not int or not minimum <= value <= _MAX_INT64:
        _fail(path, f"must be an exact integer in [{minimum}, {_MAX_INT64}]")
    return value


def _exact_bool(value: Any, expected: bool, *, path: str) -> None:
    if type(value) is not bool or value is not expected:
        _fail(path, f"must be exactly {expected}")


def _stable_identifier(value: Any, *, path: str) -> str:
    if (
        type(value) is not str
        or not value
        or len(value) > 256
        or value.strip() != value
        or not value.isascii()
        or not value.isprintable()
        or "\x00" in value
    ):
        _fail(path, "must be a nonempty stable ASCII identifier")
    return value


def _sha256_hex(value: Any, *, path: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        _fail(path, "must be exactly 64 lowercase hexadecimal characters")
    return value


def _exact_constant_mapping(
    value: Any, expected: Mapping[str, Any], *, path: str
) -> None:
    raw = _exact_mapping(value, tuple(expected), path=path)
    for key, expected_value in expected.items():
        observed = raw[key]
        if type(observed) is not type(expected_value) or observed != expected_value:
            _fail(f"{path}.{key}", f"must be exactly {expected_value!r}")


def _strictly_increasing(values: list[int], *, path: str) -> None:
    if any(left >= right for left, right in zip(values, values[1:])):
        _fail(path, "must be strictly increasing without duplicates")


def _stop_reason_mask(value: Any, *, path: str) -> int:
    if type(value) is not str or not value:
        _fail(path, "must be a nonempty canonical stop reason")
    names = value.split("+")
    expected = [name for name in _STOP_REASON_BITS if name in names]
    if names != expected or len(set(names)) != len(names):
        _fail(path, "must use unique reasons in canonical planner order")
    try:
        return sum(_STOP_REASON_BITS[name] for name in names)
    except KeyError:
        _fail(path, "contains an unsupported stop reason")
    raise AssertionError("unreachable")


def _validated_plan_payload(
    value: CellDispatchPlanReceiptV1 | Mapping[str, Any],
) -> Mapping[str, Any]:
    if isinstance(value, CellDispatchPlanReceiptV1):
        source: Any = value.to_dict()
    else:
        source = value
    plan = _exact_mapping(source, _PLAN_FIELDS, path="dispatch_plan")
    if plan["schema_version"] != DISPATCH_PLAN_SCHEMA_VERSION_V1:
        _fail("dispatch_plan.schema_version", "is not the accepted dispatch plan v1")
    _stable_identifier(plan["controller_id"], path="dispatch_plan.controller_id")
    _sha256_hex(plan["schedule_sha256"], path="dispatch_plan.schedule_sha256")
    maximum = _exact_int(
        plan["max_generations_per_dispatch"],
        path="dispatch_plan.max_generations_per_dispatch",
        minimum=1,
    )
    cohort_count = _exact_int(plan["cohort_count"], path="dispatch_plan.cohort_count")
    cohorts = _exact_list(plan["cohorts"], path="dispatch_plan.cohorts")
    if not cohorts or len(cohorts) != cohort_count:
        _fail("dispatch_plan.cohorts", "must contain exactly cohort_count nonempty rows")

    classified_indices: list[int] = []
    known_stable_ids: set[int] = set()
    previous_generation = -1
    for cohort_index, raw_cohort in enumerate(cohorts):
        path = f"dispatch_plan.cohorts[{cohort_index}]"
        cohort = _exact_mapping(raw_cohort, _COHORT_FIELDS, path=path)
        if _exact_int(cohort["cohort_index"], path=f"{path}.cohort_index") != cohort_index:
            _fail(f"{path}.cohort_index", "must equal its zero-based row position")
        count = _exact_int(
            cohort["physical_batch_cell_count"],
            path=f"{path}.physical_batch_cell_count",
            minimum=1,
        )
        cell_indices_raw = _exact_list(cohort["cell_indices"], path=f"{path}.cell_indices")
        stable_ids_raw = _exact_list(
            cohort["stable_cell_ids"], path=f"{path}.stable_cell_ids"
        )
        if len(cell_indices_raw) != count or len(stable_ids_raw) != count:
            _fail(path, "physical batch count and per-cell list lengths disagree")
        cell_indices = [
            _exact_int(item, path=f"{path}.cell_indices[{index}]")
            for index, item in enumerate(cell_indices_raw)
        ]
        stable_ids = [
            _exact_int(item, path=f"{path}.stable_cell_ids[{index}]")
            for index, item in enumerate(stable_ids_raw)
        ]
        pairs = list(zip(stable_ids, cell_indices, strict=True))
        if any(left >= right for left, right in zip(pairs, pairs[1:])):
            _fail(path, "cells must be ordered by stable_cell_id then cell_index")
        if any(stable_id in known_stable_ids for stable_id in stable_ids):
            _fail(f"{path}.stable_cell_ids", "stable IDs must be globally unique")
        known_stable_ids.update(stable_ids)
        classified_indices.extend(cell_indices)

        start_generation = _exact_int(
            cohort["start_generation"], path=f"{path}.start_generation"
        )
        if start_generation <= previous_generation:
            _fail(
                f"{path}.start_generation",
                "cohorts must be in strictly ascending generation order",
            )
        previous_generation = start_generation
        generation_count = _exact_int(
            cohort["generation_count"], path=f"{path}.generation_count", minimum=1
        )
        if generation_count > maximum:
            _fail(f"{path}.generation_count", "exceeds max_generations_per_dispatch")
        if start_generation > _MAX_INT64 - generation_count:
            _fail(path, "generation interval exceeds signed int64")
        reason_mask = _stop_reason_mask(cohort["stop_reason"], path=f"{path}.stop_reason")
        intermediate_chunk = bool(
            reason_mask & STOP_REASON_ACCEPTED_GENERATION_CHUNK_V1
        )
        if intermediate_chunk and reason_mask != STOP_REASON_ACCEPTED_GENERATION_CHUNK_V1:
            _fail(
                f"{path}.stop_reason",
                "accepted_generation_chunk must be a standalone reason",
            )
        maximum_limited = bool(reason_mask & STOP_REASON_MAX_DISPATCH_V1)
        if maximum_limited != (generation_count == maximum):
            _fail(path, "max-dispatch reason contradicts the generation count")
        stop_raw = _exact_list(
            cohort["stop_cell_indices"], path=f"{path}.stop_cell_indices"
        )
        stop_indices = [
            _exact_int(item, path=f"{path}.stop_cell_indices[{index}]")
            for index, item in enumerate(stop_raw)
        ]
        if intermediate_chunk and stop_indices:
            _fail(
                f"{path}.stop_cell_indices",
                "intermediate accepted chunks must not identify limiting cells",
            )
        if not intermediate_chunk and not stop_indices:
            _fail(f"{path}.stop_cell_indices", "must identify at least one limiting cell")
        _strictly_increasing(stop_indices, path=f"{path}.stop_cell_indices")
        if not set(stop_indices).issubset(cell_indices):
            _fail(f"{path}.stop_cell_indices", "must be a subset of cohort cells")
        _exact_bool(
            cohort["shared_generation_coordinate"],
            True,
            path=f"{path}.shared_generation_coordinate",
        )

    terminal_raw = _exact_list(
        plan["terminal_cell_indices"], path="dispatch_plan.terminal_cell_indices"
    )
    terminal_indices = [
        _exact_int(item, path=f"dispatch_plan.terminal_cell_indices[{index}]")
        for index, item in enumerate(terminal_raw)
    ]
    if len(terminal_indices) != len(set(terminal_indices)):
        _fail("dispatch_plan.terminal_cell_indices", "must not contain duplicates")
    classified_indices.extend(terminal_indices)

    blocked_raw = _exact_list(
        plan["blocked_active_cells"], path="dispatch_plan.blocked_active_cells"
    )
    previous_blocked_key: tuple[int, int] | None = None
    for blocked_index, raw_blocked in enumerate(blocked_raw):
        path = f"dispatch_plan.blocked_active_cells[{blocked_index}]"
        blocked = _exact_mapping(raw_blocked, _BLOCKED_FIELDS, path=path)
        cell_index = _exact_int(blocked["cell_index"], path=f"{path}.cell_index")
        stable_id = _exact_int(
            blocked["stable_cell_id"], path=f"{path}.stable_cell_id"
        )
        key = (stable_id, cell_index)
        if previous_blocked_key is not None and previous_blocked_key >= key:
            _fail(path, "rows must be ordered by stable_cell_id then cell_index")
        previous_blocked_key = key
        if stable_id in known_stable_ids:
            _fail(f"{path}.stable_cell_id", "stable IDs must be globally unique")
        known_stable_ids.add(stable_id)
        classified_indices.append(cell_index)
        _exact_int(blocked["current_generation"], path=f"{path}.current_generation")
        reason = blocked["reason"]
        if type(reason) is not str or reason not in _BLOCKED_REASON_CODES:
            _fail(f"{path}.reason", "is not a supported blocked reason")
        pending_boundary = blocked["pending_boundary"]
        pending_kind = blocked["pending_command_kind"]
        if reason == "pending_boundary":
            _exact_int(pending_boundary, path=f"{path}.pending_boundary")
            if type(pending_kind) is not str or pending_kind not in _PENDING_COMMAND_KIND_CODES:
                _fail(f"{path}.pending_command_kind", "is not a supported command kind")
        elif pending_boundary is not None or pending_kind is not None:
            _fail(path, "generation-limit rows must have null pending metadata")

    if len(classified_indices) != len(set(classified_indices)):
        _fail("dispatch_plan", "active, terminal, and blocked cell classes overlap")
    if sorted(classified_indices) != list(range(len(classified_indices))):
        _fail(
            "dispatch_plan",
            "cell classifications must exactly partition contiguous indices 0..B-N-1",
        )
    _exact_constant_mapping(
        plan["selection_policy"], _SELECTION_POLICY, path="dispatch_plan.selection_policy"
    )
    _exact_constant_mapping(plan["contract"], _SOURCE_CONTRACT, path="dispatch_plan.contract")
    claimed_digest = _sha256_hex(plan["plan_sha256"], path="dispatch_plan.plan_sha256")
    digest_payload = {key: plan[key] for key in _PLAN_FIELDS if key != "plan_sha256"}
    observed_digest = sha256(canonical_branch_json_bytes(digest_payload)).hexdigest()
    if claimed_digest != observed_digest:
        _fail("dispatch_plan.plan_sha256", "does not match the canonical plan payload")
    return plan


def _readonly_array(values: Any, *, dtype: np.dtype[Any]) -> NDArray[Any]:
    array = np.array(values, dtype=dtype, order="C", copy=True)
    array.setflags(write=False)
    return array


def _array_specs(arrays: Mapping[str, NDArray[Any]]) -> Mapping[str, Any]:
    return MappingProxyType(
        {
            name: MappingProxyType(
                {
                    "dtype": arrays[name].dtype.str,
                    "shape": tuple(int(item) for item in arrays[name].shape),
                    "c_contiguous": True,
                    "readonly": True,
                }
            )
            for name in ARRAY_ORDER_V1
        }
    )


def _metadata(
    *,
    controller_id: str,
    schedule_sha256: str,
    plan_sha256: str,
    maximum: int,
    arrays: Mapping[str, NDArray[Any]],
) -> Mapping[str, Any]:
    cohort_count = len(arrays["cohort_start_generations"])
    dispatch_cell_count = len(arrays["cell_indices"])
    terminal_count = len(arrays["terminal_cell_indices"])
    blocked_count = len(arrays["blocked_cell_indices"])
    return MappingProxyType(
        {
            "schema_version": CELL_DISPATCH_ARRAYS_SCHEMA_VERSION_V1,
            "source_schema_version": DISPATCH_PLAN_SCHEMA_VERSION_V1,
            "abi_version": CELL_DISPATCH_ARRAYS_ABI_VERSION_V1,
            "controller_id": controller_id,
            "schedule_sha256": schedule_sha256,
            "plan_sha256": plan_sha256,
            "max_generations_per_dispatch": maximum,
            "cohort_count": cohort_count,
            "dispatch_cell_count": dispatch_cell_count,
            "configured_cell_count": dispatch_cell_count + terminal_count + blocked_count,
            "terminal_cell_count": terminal_count,
            "blocked_active_cell_count": blocked_count,
            "selection_policy": MappingProxyType(dict(_SELECTION_POLICY)),
            "ordering": MappingProxyType(
                {
                    "arrays": ARRAY_ORDER_V1,
                    "cohorts": "start_generation_ascending",
                    "cohort_cells": "stable_cell_id_then_cell_index",
                    "terminal_cells": "source_receipt_order",
                    "blocked_cells": "stable_cell_id_then_cell_index",
                }
            ),
            "codebooks": MappingProxyType(
                {
                    "cohort_stop_reason_bits": MappingProxyType(dict(_STOP_REASON_BITS)),
                    "blocked_reason_codes": MappingProxyType(dict(_BLOCKED_REASON_CODES)),
                    "pending_command_kind_codes": MappingProxyType(
                        {
                            "none": PENDING_COMMAND_NONE_V1,
                            COMMAND_CHECKPOINT_V1: PENDING_COMMAND_CHECKPOINT_V1,
                            COMMAND_CHECKPOINT_AND_FORK_V1: (
                                PENDING_COMMAND_CHECKPOINT_AND_FORK_V1
                            ),
                            COMMAND_CANCEL_V1: PENDING_COMMAND_CANCEL_V1,
                        }
                    ),
                }
            ),
            "arrays": _array_specs(arrays),
            "accepted_generation_chunk_semantics": MappingProxyType(
                {
                    "capability_argument_required_by_planner": True,
                    "capability_list_exposed_by_source_receipt": False,
                    "capability_list_inferred_by_lowering": False,
                    "selected_chunk_array": "cohort_generation_counts",
                    "intermediate_reason_array": "cohort_stop_reason_masks",
                }
            ),
            "nullable_encoding": MappingProxyType(
                {
                    "blocked_pending_boundary": (
                        "present=0 requires value=-1; present=1 requires value>=0"
                    ),
                    "blocked_pending_command_kind": "code=0 means explicit none",
                }
            ),
            "stop_semantics": (
                "cell_stop_flags mark limiting cells; cohort_stop_reason_masks retain "
                "the source cohort-level reason set; a standalone accepted generation "
                "chunk has no limiting cells and does not invent per-cell attribution"
            ),
            "digest_contract": MappingProxyType(
                {
                    "algorithm": "sha256",
                    "domain_hex": CELL_DISPATCH_ARRAYS_DIGEST_DOMAIN_V1.hex(),
                    "metadata_encoding": (
                        "canonical JSON UTF-8 with sorted keys and compact separators"
                    ),
                    "framing": (
                        "domain then repeated uint64_le(label_bytes) + label + "
                        "uint64_le(payload_bytes) + payload"
                    ),
                    "array_byte_order": "C",
                    "array_order": ARRAY_ORDER_V1,
                }
            ),
            "contract": MappingProxyType(
                {
                    "metadata_only": True,
                    "one_future_native_controller_call_input": True,
                    "native_cpp_call_performed": False,
                    "live_controller_integration": False,
                    "cuda_execution_performed": False,
                    "simulation_execution_performed": False,
                    "performance_claimed": False,
                    "production_promotion_claimed": False,
                }
            ),
        }
    )


@dataclass(frozen=True, slots=True)
class CellDispatchArraysV1:
    """Frozen ordered inputs for one future native cell-controller call."""

    cohort_offsets: NDArray[Any]
    cohort_start_generations: NDArray[Any]
    cohort_generation_counts: NDArray[Any]
    cohort_stop_reason_masks: NDArray[Any]
    cell_indices: NDArray[Any]
    stable_cell_ids: NDArray[Any]
    cell_stop_flags: NDArray[Any]
    terminal_cell_indices: NDArray[Any]
    blocked_cell_indices: NDArray[Any]
    blocked_stable_cell_ids: NDArray[Any]
    blocked_current_generations: NDArray[Any]
    blocked_reason_codes: NDArray[Any]
    blocked_pending_boundary_present: NDArray[Any]
    blocked_pending_boundaries: NDArray[Any]
    blocked_pending_command_kind_codes: NDArray[Any]
    metadata: Mapping[str, Any]
    content_sha256: str

    @property
    def ordered_arrays(self) -> tuple[NDArray[Any], ...]:
        return tuple(getattr(self, name) for name in ARRAY_ORDER_V1)

    def as_tuple(self) -> tuple[NDArray[Any], ...]:
        return self.ordered_arrays


def _arrays_by_name(value: CellDispatchArraysV1) -> Mapping[str, NDArray[Any]]:
    return {name: getattr(value, name) for name in ARRAY_ORDER_V1}


def _update_framed_digest(digest: Any, label: str, payload: bytes) -> None:
    label_bytes = label.encode("ascii")
    digest.update(len(label_bytes).to_bytes(8, "little"))
    digest.update(label_bytes)
    digest.update(len(payload).to_bytes(8, "little"))
    digest.update(payload)


def _content_digest(
    metadata: Mapping[str, Any], arrays: Mapping[str, NDArray[Any]]
) -> str:
    digest = sha256()
    digest.update(CELL_DISPATCH_ARRAYS_DIGEST_DOMAIN_V1)
    _update_framed_digest(digest, "metadata", canonical_branch_json_bytes(metadata))
    for name in ARRAY_ORDER_V1:
        _update_framed_digest(digest, name, arrays[name].tobytes(order="C"))
    return digest.hexdigest()


def _validate_array_bundle(value: CellDispatchArraysV1, *, verify_digest: bool) -> None:
    if not isinstance(value, CellDispatchArraysV1):
        _fail("dispatch_arrays", "must use CellDispatchArraysV1")
    arrays = _arrays_by_name(value)
    for name in ARRAY_ORDER_V1:
        array = arrays[name]
        if type(array) is not np.ndarray:
            _fail(f"dispatch_arrays.{name}", "must be an exact NumPy ndarray")
        if array.dtype != _ARRAY_DTYPES[name] or array.ndim != 1:
            _fail(f"dispatch_arrays.{name}", "has an invalid dtype or shape rank")
        if not array.flags.c_contiguous or not array.flags.aligned or array.flags.writeable:
            _fail(
                f"dispatch_arrays.{name}",
                "must be aligned, C-contiguous, and read-only",
            )

    cohort_count = len(value.cohort_start_generations)
    dispatch_count = len(value.cell_indices)
    terminal_count = len(value.terminal_cell_indices)
    blocked_count = len(value.blocked_cell_indices)
    expected_lengths = {
        "cohort_offsets": cohort_count + 1,
        "cohort_start_generations": cohort_count,
        "cohort_generation_counts": cohort_count,
        "cohort_stop_reason_masks": cohort_count,
        "cell_indices": dispatch_count,
        "stable_cell_ids": dispatch_count,
        "cell_stop_flags": dispatch_count,
        "terminal_cell_indices": terminal_count,
        "blocked_cell_indices": blocked_count,
        "blocked_stable_cell_ids": blocked_count,
        "blocked_current_generations": blocked_count,
        "blocked_reason_codes": blocked_count,
        "blocked_pending_boundary_present": blocked_count,
        "blocked_pending_boundaries": blocked_count,
        "blocked_pending_command_kind_codes": blocked_count,
    }
    for name, expected in expected_lengths.items():
        if len(arrays[name]) != expected:
            _fail(f"dispatch_arrays.{name}", f"must have shape ({expected},)")
    if cohort_count == 0:
        _fail("dispatch_arrays.cohort_offsets", "at least one cohort is required")

    offsets = value.cohort_offsets.tolist()
    if offsets[0] != 0 or offsets[-1] != dispatch_count:
        _fail("dispatch_arrays.cohort_offsets", "must span exactly all dispatch cells")
    if any(left >= right for left, right in zip(offsets, offsets[1:])):
        _fail("dispatch_arrays.cohort_offsets", "must be strictly increasing")
    starts = value.cohort_start_generations.tolist()
    if any(item < 0 for item in starts) or any(
        left >= right for left, right in zip(starts, starts[1:])
    ):
        _fail(
            "dispatch_arrays.cohort_start_generations",
            "must be nonnegative and strictly increasing",
        )

    metadata = _exact_mapping(
        value.metadata,
        (
            "schema_version",
            "source_schema_version",
            "abi_version",
            "controller_id",
            "schedule_sha256",
            "plan_sha256",
            "max_generations_per_dispatch",
            "cohort_count",
            "dispatch_cell_count",
            "configured_cell_count",
                "terminal_cell_count",
                "blocked_active_cell_count",
                "selection_policy",
                "ordering",
                "codebooks",
                "arrays",
                "accepted_generation_chunk_semantics",
                "nullable_encoding",
            "stop_semantics",
            "digest_contract",
            "contract",
        ),
        path="dispatch_arrays.metadata",
    )
    maximum = _exact_int(
        metadata["max_generations_per_dispatch"],
        path="dispatch_arrays.metadata.max_generations_per_dispatch",
        minimum=1,
    )
    counts = value.cohort_generation_counts.tolist()
    masks = value.cohort_stop_reason_masks.tolist()
    for index, (start, count, mask) in enumerate(zip(starts, counts, masks, strict=True)):
        if not 1 <= count <= maximum or start > _MAX_INT64 - count:
            _fail(f"dispatch_arrays.cohort_generation_counts[{index}]", "is out of range")
        if not 1 <= mask <= sum(_STOP_REASON_BITS.values()):
            _fail(f"dispatch_arrays.cohort_stop_reason_masks[{index}]", "has unknown bits")
        if (
            mask & STOP_REASON_ACCEPTED_GENERATION_CHUNK_V1
            and mask != STOP_REASON_ACCEPTED_GENERATION_CHUNK_V1
        ):
            _fail(
                f"dispatch_arrays.cohort_stop_reason_masks[{index}]",
                "mixes accepted_generation_chunk with a limiting reason",
            )
        if bool(mask & STOP_REASON_MAX_DISPATCH_V1) != (count == maximum):
            _fail(f"dispatch_arrays.cohort_stop_reason_masks[{index}]", "contradicts count")

    cell_indices = value.cell_indices.tolist()
    stable_ids = value.stable_cell_ids.tolist()
    stop_flags = value.cell_stop_flags.tolist()
    if any(item < 0 for item in cell_indices + stable_ids):
        _fail("dispatch_arrays.cell_indices", "cell indices and stable IDs must be nonnegative")
    if any(flag not in {0, 1} for flag in stop_flags):
        _fail("dispatch_arrays.cell_stop_flags", "must contain only 0 or 1")
    for cohort_index, (start, stop) in enumerate(
        zip(offsets[:-1], offsets[1:], strict=True)
    ):
        pairs = list(zip(stable_ids[start:stop], cell_indices[start:stop], strict=True))
        if any(left >= right for left, right in zip(pairs, pairs[1:])):
            _fail(f"dispatch_arrays.cohort[{cohort_index}]", "cell order drift")
        has_stop_cells = any(stop_flags[start:stop])
        intermediate_chunk = (
            masks[cohort_index] == STOP_REASON_ACCEPTED_GENERATION_CHUNK_V1
        )
        if intermediate_chunk and has_stop_cells:
            _fail(
                f"dispatch_arrays.cohort[{cohort_index}]",
                "intermediate accepted chunk has a limiting cell",
            )
        if not intermediate_chunk and not has_stop_cells:
            _fail(f"dispatch_arrays.cohort[{cohort_index}]", "has no limiting cell")

    terminal = value.terminal_cell_indices.tolist()
    if any(item < 0 for item in terminal) or len(terminal) != len(set(terminal)):
        _fail("dispatch_arrays.terminal_cell_indices", "must be unique nonnegative indices")
    blocked_indices = value.blocked_cell_indices.tolist()
    blocked_ids = value.blocked_stable_cell_ids.tolist()
    blocked_generations = value.blocked_current_generations.tolist()
    if any(item < 0 for item in blocked_indices + blocked_ids + blocked_generations):
        _fail("dispatch_arrays.blocked", "indices, IDs, and generations must be nonnegative")
    blocked_pairs = list(zip(blocked_ids, blocked_indices, strict=True))
    if any(left >= right for left, right in zip(blocked_pairs, blocked_pairs[1:])):
        _fail("dispatch_arrays.blocked", "row order drift")
    if len(set(stable_ids + blocked_ids)) != len(stable_ids) + len(blocked_ids):
        _fail("dispatch_arrays.stable_cell_ids", "known stable IDs must be unique")

    reasons = value.blocked_reason_codes.tolist()
    present = value.blocked_pending_boundary_present.tolist()
    boundaries = value.blocked_pending_boundaries.tolist()
    commands = value.blocked_pending_command_kind_codes.tolist()
    for index, (reason, has_boundary, boundary, command) in enumerate(
        zip(reasons, present, boundaries, commands, strict=True)
    ):
        if reason == BLOCKED_REASON_PENDING_BOUNDARY_V1:
            if has_boundary != 1 or boundary < 0 or command not in {
                PENDING_COMMAND_CHECKPOINT_V1,
                PENDING_COMMAND_CHECKPOINT_AND_FORK_V1,
                PENDING_COMMAND_CANCEL_V1,
            }:
                _fail(f"dispatch_arrays.blocked[{index}]", "invalid pending-boundary metadata")
        elif reason == BLOCKED_REASON_GENERATION_LIMIT_V1:
            if has_boundary != 0 or boundary != -1 or command != PENDING_COMMAND_NONE_V1:
                _fail(f"dispatch_arrays.blocked[{index}]", "invalid generation-limit metadata")
        else:
            _fail(f"dispatch_arrays.blocked_reason_codes[{index}]", "is unsupported")

    classified = cell_indices + terminal + blocked_indices
    if len(classified) != len(set(classified)):
        _fail("dispatch_arrays", "cell classifications overlap")
    if sorted(classified) != list(range(len(classified))):
        _fail("dispatch_arrays", "cell classifications do not form contiguous B-N")

    controller_id = _stable_identifier(
        metadata["controller_id"], path="dispatch_arrays.metadata.controller_id"
    )
    schedule_digest = _sha256_hex(
        metadata["schedule_sha256"], path="dispatch_arrays.metadata.schedule_sha256"
    )
    plan_digest = _sha256_hex(
        metadata["plan_sha256"], path="dispatch_arrays.metadata.plan_sha256"
    )
    for name, expected in {
        "cohort_count": cohort_count,
        "dispatch_cell_count": dispatch_count,
        "configured_cell_count": len(classified),
        "terminal_cell_count": terminal_count,
        "blocked_active_cell_count": blocked_count,
    }.items():
        if _exact_int(metadata[name], path=f"dispatch_arrays.metadata.{name}") != expected:
            _fail(f"dispatch_arrays.metadata.{name}", "does not match array shapes")
    expected_metadata = _metadata(
        controller_id=controller_id,
        schedule_sha256=schedule_digest,
        plan_sha256=plan_digest,
        maximum=maximum,
        arrays=arrays,
    )
    if canonical_branch_json_bytes(metadata) != canonical_branch_json_bytes(expected_metadata):
        _fail("dispatch_arrays.metadata", "does not match the exact ABI manifest")
    claimed = _sha256_hex(value.content_sha256, path="dispatch_arrays.content_sha256")
    if verify_digest and claimed != _content_digest(metadata, arrays):
        _fail("dispatch_arrays.content_sha256", "does not match metadata and array bytes")


def lower_cell_dispatch_plan_to_arrays_v1(
    value: CellDispatchPlanReceiptV1 | Mapping[str, Any],
) -> CellDispatchArraysV1:
    """Validate and lower one dispatch plan into strict contiguous typed arrays."""

    plan = _validated_plan_payload(value)
    cohort_offsets = [0]
    cell_indices: list[int] = []
    stable_ids: list[int] = []
    cell_stop_flags: list[int] = []
    starts: list[int] = []
    counts: list[int] = []
    stop_masks: list[int] = []
    for cohort in plan["cohorts"]:
        cohort_cells = list(cohort["cell_indices"])
        stop_cells = set(cohort["stop_cell_indices"])
        cell_indices.extend(cohort_cells)
        stable_ids.extend(cohort["stable_cell_ids"])
        cell_stop_flags.extend(int(index in stop_cells) for index in cohort_cells)
        cohort_offsets.append(len(cell_indices))
        starts.append(cohort["start_generation"])
        counts.append(cohort["generation_count"])
        stop_masks.append(
            sum(_STOP_REASON_BITS[name] for name in cohort["stop_reason"].split("+"))
        )

    blocked_indices: list[int] = []
    blocked_ids: list[int] = []
    blocked_generations: list[int] = []
    blocked_reasons: list[int] = []
    blocked_present: list[int] = []
    blocked_boundaries: list[int] = []
    blocked_commands: list[int] = []
    for blocked in plan["blocked_active_cells"]:
        pending_boundary = blocked["pending_boundary"]
        blocked_indices.append(blocked["cell_index"])
        blocked_ids.append(blocked["stable_cell_id"])
        blocked_generations.append(blocked["current_generation"])
        blocked_reasons.append(_BLOCKED_REASON_CODES[blocked["reason"]])
        blocked_present.append(int(pending_boundary is not None))
        blocked_boundaries.append(-1 if pending_boundary is None else pending_boundary)
        blocked_commands.append(_PENDING_COMMAND_KIND_CODES[blocked["pending_command_kind"]])

    raw_arrays = {
        "cohort_offsets": _readonly_array(cohort_offsets, dtype=_INT64_DTYPE),
        "cohort_start_generations": _readonly_array(starts, dtype=_INT64_DTYPE),
        "cohort_generation_counts": _readonly_array(counts, dtype=_INT64_DTYPE),
        "cohort_stop_reason_masks": _readonly_array(stop_masks, dtype=_UINT8_DTYPE),
        "cell_indices": _readonly_array(cell_indices, dtype=_INT64_DTYPE),
        "stable_cell_ids": _readonly_array(stable_ids, dtype=_INT64_DTYPE),
        "cell_stop_flags": _readonly_array(cell_stop_flags, dtype=_UINT8_DTYPE),
        "terminal_cell_indices": _readonly_array(
            plan["terminal_cell_indices"], dtype=_INT64_DTYPE
        ),
        "blocked_cell_indices": _readonly_array(blocked_indices, dtype=_INT64_DTYPE),
        "blocked_stable_cell_ids": _readonly_array(blocked_ids, dtype=_INT64_DTYPE),
        "blocked_current_generations": _readonly_array(
            blocked_generations, dtype=_INT64_DTYPE
        ),
        "blocked_reason_codes": _readonly_array(blocked_reasons, dtype=_UINT8_DTYPE),
        "blocked_pending_boundary_present": _readonly_array(
            blocked_present, dtype=_UINT8_DTYPE
        ),
        "blocked_pending_boundaries": _readonly_array(
            blocked_boundaries, dtype=_INT64_DTYPE
        ),
        "blocked_pending_command_kind_codes": _readonly_array(
            blocked_commands, dtype=_UINT8_DTYPE
        ),
    }
    metadata = _metadata(
        controller_id=plan["controller_id"],
        schedule_sha256=plan["schedule_sha256"],
        plan_sha256=plan["plan_sha256"],
        maximum=plan["max_generations_per_dispatch"],
        arrays=raw_arrays,
    )
    result = CellDispatchArraysV1(
        **raw_arrays,
        metadata=metadata,
        content_sha256=_content_digest(metadata, raw_arrays),
    )
    validate_cell_dispatch_arrays_v1(result)
    return result


def cell_dispatch_arrays_content_sha256_v1(value: CellDispatchArraysV1) -> str:
    """Recompute the canonical metadata-plus-array content digest."""

    _validate_array_bundle(value, verify_digest=False)
    return _content_digest(value.metadata, _arrays_by_name(value))


def validate_cell_dispatch_arrays_v1(value: CellDispatchArraysV1) -> None:
    """Fail closed on dtype, shape, order, metadata, or digest drift."""

    _validate_array_bundle(value, verify_digest=True)


__all__ = [
    "ARRAY_ORDER_V1",
    "BLOCKED_REASON_GENERATION_LIMIT_V1",
    "BLOCKED_REASON_PENDING_BOUNDARY_V1",
    "CELL_DISPATCH_ARRAYS_ABI_VERSION_V1",
    "CELL_DISPATCH_ARRAYS_SCHEMA_VERSION_V1",
    "CellDispatchArraysContractError",
    "CellDispatchArraysV1",
    "PENDING_COMMAND_CANCEL_V1",
    "PENDING_COMMAND_CHECKPOINT_AND_FORK_V1",
    "PENDING_COMMAND_CHECKPOINT_V1",
    "PENDING_COMMAND_NONE_V1",
    "STOP_REASON_GENERATION_LIMIT_V1",
    "STOP_REASON_ACCEPTED_GENERATION_CHUNK_V1",
    "STOP_REASON_MAX_DISPATCH_V1",
    "STOP_REASON_PENDING_BOUNDARY_V1",
    "cell_dispatch_arrays_content_sha256_v1",
    "lower_cell_dispatch_plan_to_arrays_v1",
    "validate_cell_dispatch_arrays_v1",
]
