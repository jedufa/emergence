"""Lossless host projection for compact generation-summary ABI v2."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any, Final

import numpy as np

from . import compact_summary_planes as abi


_SHA256_RE: Final[re.Pattern[str]] = re.compile(r"^[0-9a-f]{64}$")
_NESTED_FIELDS: Final[tuple[str, ...]] = (
    "generation_start",
    "generation_end",
    "next_generation_start",
    "deaths_by_class",
    "turnover",
    "events",
    "identity",
    "rng",
    "terminal",
    "state_digests",
    "generation_metrics",
    "behavior_metrics",
    "lineage_metrics",
)
_EVENT_COUNT_FIELDS: Final[tuple[str, ...]] = (
    "birth_committed",
    "mutation",
    "birth_failed",
    "death_committed",
)


def _get_path(value: Mapping[str, Any], path: str) -> Any:
    current: Any = value
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            raise ValueError(f"compact row is missing {path}")
        current = current[part]
    return current


def _set_path(value: dict[str, Any], path: str, item: Any) -> None:
    parts = path.split(".")
    current = value
    for part in parts[:-1]:
        child = current.get(part)
        if not isinstance(child, dict):
            raise ValueError(f"cannot project {path}: parent {part} is not an object")
        current = child
    current[parts[-1]] = item


def _mask_for_row(row: Mapping[str, Any]) -> int:
    mask = 0
    patch = row.get(abi.PATCH_OBJECT_FIELD)
    mask = abi.set_field_defined(
        mask, abi.PATCH_OBJECT_FIELD, isinstance(patch, Mapping)
    )
    for field in abi.NON_PATCH_DEFINED_FIELDS:
        mask = abi.set_field_defined(mask, field, _get_path(row, field) is not None)
    if isinstance(patch, Mapping):
        for field in abi.PATCH_DEFINED_FIELDS:
            mask = abi.set_field_defined(mask, field, _get_path(row, field) is not None)
    return mask


def _strict_integer(value: Any, *, field: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{field} must be an int")
    if value < 0 or value > abi.INT64_MAX:
        raise ValueError(f"{field} must fit nonnegative int64")
    return value


def _strict_float(value: Any, *, field: str) -> float:
    if type(value) is not float:
        raise TypeError(f"{field} must be a canonical Python float")
    if not math.isfinite(value):
        raise ValueError(f"{field} must be finite")
    return value


def _strict_digest(value: Any, *, field: str) -> bytes:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{field} must be lowercase SHA-256 hex")
    return bytes.fromhex(value)


def _require_target_noop(
    planes: abi.CompactSummaryPlanes | Mapping[str, np.ndarray],
    generation: int,
    cell: int,
) -> Mapping[str, np.ndarray]:
    generations, cells = abi.validate_compact_summary_plane_layout(planes)
    if type(generation) is not int or generation < 0 or generation >= generations:
        raise IndexError("generation index is outside compact plane bounds")
    if type(cell) is not int or cell < 0 or cell >= cells:
        raise IndexError("cell index is outside compact plane bounds")
    values = planes.as_mapping() if isinstance(planes, abi.CompactSummaryPlanes) else planes
    for name, array in values.items():
        if bool(np.any(array[generation, cell].view(np.uint8))):
            raise ValueError(f"target {name}[{generation},{cell}] is not a no-op row")
    return values


def encode_compact_summary_row(
    row: Mapping[str, Any],
    planes: abi.CompactSummaryPlanes | Mapping[str, np.ndarray],
    *,
    generation: int,
    cell: int,
) -> None:
    """Atomically encode one already-authoritative canonical row into ABI v2."""

    if not isinstance(row, Mapping):
        raise TypeError("compact row must be a mapping")
    target = _require_target_noop(planes, generation, cell)
    if row.get("schema_version") != abi.WIRE_SCHEMA_VERSION:
        raise ValueError("compact row schema_version mismatch")
    if row.get("summary_valid") is not True:
        raise ValueError("only complete valid rows can be encoded")
    if not isinstance(row.get("undefined_fields"), list):
        raise TypeError("undefined_fields must be a list")

    mask = _mask_for_row(row)
    expected_undefined = list(abi.undefined_fields_from_mask(mask))
    if row["undefined_fields"] != expected_undefined:
        raise ValueError(
            "undefined_fields disagrees with the general defined mask: "
            f"expected={expected_undefined}"
        )

    scratch = abi.allocate_compact_summary_planes(1, 1)
    scratch.defined_mask[0, 0, 0] = np.uint64(mask)
    patch_present = abi.is_field_defined(mask, abi.PATCH_OBJECT_FIELD)
    for field, index in abi.STATUS_FIELD_INDEX.items():
        value = _get_path(row, field)
        if field in abi.DEFINED_FIELD_BITS and value is None:
            continue
        scratch.status_codes[0, 0, index] = abi.status_code(field, value)
    for field, index in abi.INTEGER_FIELD_INDEX.items():
        if field.startswith("patch_metrics.") and not patch_present:
            continue
        value = _get_path(row, field)
        if field in abi.DEFINED_FIELD_BITS and value is None:
            continue
        scratch.integer_values[0, 0, index] = _strict_integer(value, field=field)
    for field, index in abi.FLOAT_FIELD_INDEX.items():
        if field.startswith("patch_metrics.") and not patch_present:
            continue
        value = _get_path(row, field)
        if field in abi.DEFINED_FIELD_BITS and value is None:
            continue
        scratch.float_values[0, 0, index] = _strict_float(value, field=field)
    for field, index in abi.DIGEST_FIELD_INDEX.items():
        value = _get_path(row, field)
        if field in abi.DEFINED_FIELD_BITS and value is None:
            continue
        scratch.digest_values[0, 0, index] = np.frombuffer(
            _strict_digest(value, field=field), dtype=np.uint8
        )
    for field, index in abi.TEXT_FIELD_INDEX.items():
        value = _get_path(row, field)
        if field in abi.DEFINED_FIELD_BITS and value is None:
            continue
        scratch.text_values[0, 0, index] = abi.encode_text_value(value)
    scratch.validity[0, 0] = 1
    abi.validate_compact_summary_planes(scratch)

    for name, source in scratch.as_mapping().items():
        if name == "validity":
            continue
        target[name][generation, cell] = source[0, 0]
    target["validity"][generation, cell] = 1


def _empty_row(*, patch_present: bool) -> dict[str, Any]:
    row: dict[str, Any] = {
        "schema_version": abi.WIRE_SCHEMA_VERSION,
        "summary_valid": True,
        "summary_status": None,
        "world": None,
        "alive": None,
        "births": None,
        "failed_births": None,
        "deaths": None,
        "checkpoint_due": None,
        "patch_metrics": {} if patch_present else None,
        "undefined_fields": [],
    }
    for field in _NESTED_FIELDS:
        row[field] = {}
    row["events"] = {
        "generation_counts_by_type": {
            name: None for name in _EVENT_COUNT_FIELDS
        },
        "turnover_counts_by_type": {
            name: None for name in _EVENT_COUNT_FIELDS
        },
    }
    return row


def project_compact_summary_row(
    planes: abi.CompactSummaryPlanes | Mapping[str, np.ndarray],
    *,
    generation: int,
    cell: int,
) -> dict[str, Any]:
    """Project one valid ABI v2 row back to the canonical wire object."""

    generations, cells = abi.validate_compact_summary_planes(planes)
    if type(generation) is not int or generation < 0 or generation >= generations:
        raise IndexError("generation index is outside compact plane bounds")
    if type(cell) is not int or cell < 0 or cell >= cells:
        raise IndexError("cell index is outside compact plane bounds")
    values = planes.as_mapping() if isinstance(planes, abi.CompactSummaryPlanes) else planes
    if int(values["validity"][generation, cell]) != 1:
        raise ValueError("invalid/no-op compact rows cannot be projected")
    mask = abi.validate_defined_mask(values["defined_mask"][generation, cell, 0])
    patch_present = abi.is_field_defined(mask, abi.PATCH_OBJECT_FIELD)
    row = _empty_row(patch_present=patch_present)

    for field, index in abi.STATUS_FIELD_INDEX.items():
        value = (
            None
            if field in abi.DEFINED_FIELD_BITS
            and not abi.is_field_defined(mask, field)
            else abi.status_value(field, int(values["status_codes"][generation, cell, index]))
        )
        _set_path(row, field, value)
    for field, index in abi.INTEGER_FIELD_INDEX.items():
        if field.startswith("patch_metrics.") and not patch_present:
            continue
        value = (
            None
            if field in abi.DEFINED_FIELD_BITS
            and not abi.is_field_defined(mask, field)
            else int(values["integer_values"][generation, cell, index])
        )
        _set_path(row, field, value)
    for field, index in abi.FLOAT_FIELD_INDEX.items():
        if field.startswith("patch_metrics.") and not patch_present:
            continue
        value = (
            None
            if field in abi.DEFINED_FIELD_BITS
            and not abi.is_field_defined(mask, field)
            else float(values["float_values"][generation, cell, index])
        )
        _set_path(row, field, value)
    for field, index in abi.DIGEST_FIELD_INDEX.items():
        value = (
            None
            if field in abi.DEFINED_FIELD_BITS
            and not abi.is_field_defined(mask, field)
            else values["digest_values"][generation, cell, index].tobytes().hex()
        )
        _set_path(row, field, value)
    for field, index in abi.TEXT_FIELD_INDEX.items():
        value = (
            None
            if field in abi.DEFINED_FIELD_BITS
            and not abi.is_field_defined(mask, field)
            else abi.decode_text_value(values["text_values"][generation, cell, index])
        )
        _set_path(row, field, value)
    row["undefined_fields"] = list(abi.undefined_fields_from_mask(mask))
    return row


def encode_compact_summary_matrix(
    rows: Sequence[Sequence[Mapping[str, Any]]],
) -> abi.CompactSummaryPlanes:
    """Atomically encode a non-empty rectangular generation/cell matrix."""

    if isinstance(rows, (str, bytes, bytearray)) or not isinstance(rows, Sequence):
        raise TypeError("rows must be a sequence of generation rows")
    if not rows:
        raise ValueError("rows must contain at least one generation")
    first = rows[0]
    if isinstance(first, (str, bytes, bytearray)) or not isinstance(first, Sequence):
        raise TypeError("each generation must be a sequence of cell rows")
    cell_count = len(first)
    if cell_count == 0:
        raise ValueError("rows must contain at least one cell")

    scratch = abi.allocate_compact_summary_planes(len(rows), cell_count)
    for generation, generation_rows in enumerate(rows):
        if (
            isinstance(generation_rows, (str, bytes, bytearray))
            or not isinstance(generation_rows, Sequence)
        ):
            raise TypeError("each generation must be a sequence of cell rows")
        if len(generation_rows) != cell_count:
            raise ValueError("compact summary matrix must be rectangular")
        for cell, row in enumerate(generation_rows):
            encode_compact_summary_row(
                row,
                scratch,
                generation=generation,
                cell=cell,
            )
    abi.validate_compact_summary_planes(scratch)
    return scratch


def project_compact_summary_matrix(
    planes: abi.CompactSummaryPlanes | Mapping[str, np.ndarray],
) -> list[list[dict[str, Any]]]:
    """Project every valid generation/cell row in deterministic index order."""

    generations, cells = abi.validate_compact_summary_planes(planes)
    return [
        [
            project_compact_summary_row(
                planes,
                generation=generation,
                cell=cell,
            )
            for cell in range(cells)
        ]
        for generation in range(generations)
    ]


def canonical_compact_summary_bytes(row: Mapping[str, Any]) -> bytes:
    """Serialize with the frozen exactness envelope, including one final LF."""

    try:
        payload = json.dumps(
            row,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"compact row is not canonical JSON: {exc}") from exc
    return (payload + "\n").encode("utf-8")


def round_trip_compact_summary_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Useful golden-fixture proof that does not mutate the source row."""

    before = deepcopy(row)
    planes = abi.allocate_compact_summary_planes(1, 1)
    encode_compact_summary_row(row, planes, generation=0, cell=0)
    projected = project_compact_summary_row(planes, generation=0, cell=0)
    if row != before:
        raise RuntimeError("compact row changed during ABI projection")
    return projected


__all__ = [
    "canonical_compact_summary_bytes",
    "encode_compact_summary_matrix",
    "encode_compact_summary_row",
    "project_compact_summary_matrix",
    "project_compact_summary_row",
    "round_trip_compact_summary_row",
]
