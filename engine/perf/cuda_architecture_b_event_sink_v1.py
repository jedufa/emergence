"""Small, append-only lifecycle-event sink used by the resident CUDA route.

The resident route owns only the current physical event window.  Historical
rows are an output stream, not simulation state, and therefore must not be
reassembled into a Python list or a checkpoint tensor.  This module contains
the CPU-side conversion/validation used when an old checkpoint still carries
an inline event prefix.  The native sink uses the same deliberately simple
format.
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
import tempfile
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import torch

from engine.perf.cuda_architecture_b_boundary import (
    EVENT_COLUMN_DTYPES,
    EVENT_COLUMN_NAMES,
)

EVENT_SINK_SCHEMA_VERSION = "eiel_resident_event_sink_v1"
EVENT_SINK_MAGIC = b"EIELEVT1"
EVENT_SINK_HEADER = struct.Struct("<8sIIII")
EVENT_SINK_RECORD = struct.Struct("<IIQ")
EVENT_SINK_ROW_BYTES = sum(
    torch.empty((), dtype=EVENT_COLUMN_DTYPES[name]).element_size()
    for name in EVENT_COLUMN_NAMES
)


class EventSinkError(RuntimeError):
    """The append-only event sink is malformed or cannot advance exactly."""


def event_history_digest_from_references_v1(
    references: Sequence[Mapping[str, Any]],
) -> str:
    """Hash authenticated event content/frontiers without run-local paths."""

    normalized: list[dict[str, Any]] = []
    for index, reference in enumerate(references):
        if not isinstance(reference, Mapping):
            raise EventSinkError(f"event reference {index} must be a mapping")
        prefix = reference.get("prefix_sha256")
        if (
            reference.get("schema_version") != EVENT_SINK_SCHEMA_VERSION
            or reference.get("prefix_identity_schema")
            != "eiel_event_sink_cell_prefix_v1"
            or type(reference.get("cell_index")) is not int
            or type(reference.get("row_count")) is not int
            or type(reference.get("row_bytes")) is not int
            or not isinstance(prefix, str)
            or len(prefix) != 64
            or any(character not in "0123456789abcdef" for character in prefix)
        ):
            raise EventSinkError(f"event reference {index} identity is invalid")
        normalized.append(
            {
                "schema_version": reference["schema_version"],
                "cell_index": reference["cell_index"],
                "row_count": reference["row_count"],
                "row_bytes": reference["row_bytes"],
                "prefix_identity_schema": reference["prefix_identity_schema"],
                "prefix_sha256": prefix,
            }
        )
    return hashlib.sha256(
        json.dumps(
            normalized, allow_nan=False, separators=(",", ":"), sort_keys=True
        ).encode("utf-8")
    ).hexdigest()


def _file_sha256_v1(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_directory_v1(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


_MANIFEST_DTYPE_NAMES = {
    torch.bool: "Bool",
    torch.uint8: "Byte",
    torch.int8: "Char",
    torch.int16: "Short",
    torch.int32: "Int",
    torch.int64: "Long",
    torch.float16: "Half",
    torch.bfloat16: "BFloat16",
    torch.float32: "Float",
    torch.float64: "Double",
}


def _checkpoint_tensor_manifest_v1(
    tensor: torch.Tensor,
    *,
    previous: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Rebuild the strict owner manifest after compacting event columns."""

    dtype_name = _MANIFEST_DTYPE_NAMES.get(tensor.dtype)
    if dtype_name is None:
        raise EventSinkError(
            f"event sink manifest dtype is unsupported: {tensor.dtype}"
        )
    value = tensor.detach().cpu().contiguous()
    raw = value.numpy().tobytes(order="C")
    row = dict(previous or {})
    row.update(
        {
            "shape": [int(dim) for dim in value.shape],
            "dtype": dtype_name,
            "device": str(value.device),
            "numel": int(value.numel()),
            "element_size": int(value.element_size()),
            "bytes": len(raw),
            "is_cuda": False,
            "is_contiguous": bool(value.is_contiguous()),
        }
    )
    # Some older projections carry a digest in each manifest row.  Preserve
    # that schema when present, but never leave the digest for the old prefix.
    if "sha256" in row:
        row["sha256"] = hashlib.sha256(raw).hexdigest()
    return row


def _checked_columns(columns: Sequence[Any]) -> tuple[torch.Tensor, ...]:
    if len(columns) != len(EVENT_COLUMN_NAMES):
        raise EventSinkError("event sink columns do not match the canonical schema")
    out: list[torch.Tensor] = []
    row_count: int | None = None
    for name, value in zip(EVENT_COLUMN_NAMES, columns, strict=True):
        if not isinstance(value, torch.Tensor) or value.ndim != 1:
            raise EventSinkError(f"event sink column is not rank-one: {name}")
        if value.dtype != EVENT_COLUMN_DTYPES[name]:
            raise EventSinkError(f"event sink column has the wrong dtype: {name}")
        if row_count is None:
            row_count = int(value.numel())
        elif int(value.numel()) != row_count:
            raise EventSinkError("event sink column lengths disagree")
        out.append(value.detach().cpu().contiguous())
    return tuple(out)


def _read_state(path: Path) -> tuple[int, list[int]]:
    if not path.exists():
        return 0, []
    try:
        with path.open("rb") as stream:
            header = stream.read(EVENT_SINK_HEADER.size)
            if len(header) != EVENT_SINK_HEADER.size:
                raise EventSinkError("event sink header is truncated")
            magic, version, cells, columns, row_bytes = EVENT_SINK_HEADER.unpack(header)
            if magic != EVENT_SINK_MAGIC or version != 1:
                raise EventSinkError("event sink header schema is unsupported")
            if columns != len(EVENT_COLUMN_NAMES) or row_bytes != EVENT_SINK_ROW_BYTES:
                raise EventSinkError("event sink header column contract is invalid")
            row_counts: list[int] = []
            while True:
                raw_record = stream.read(EVENT_SINK_RECORD.size)
                if not raw_record:
                    break
                if len(raw_record) != EVENT_SINK_RECORD.size:
                    raise EventSinkError("event sink record header is truncated")
                cell, _reserved, count = EVENT_SINK_RECORD.unpack(raw_record)
                if cells and cell >= cells:
                    raise EventSinkError("event sink record cell is outside its header")
                if not cells:
                    while len(row_counts) <= cell:
                        row_counts.append(0)
                elif len(row_counts) < cells:
                    row_counts = [0] * cells
                if cell >= len(row_counts):
                    raise EventSinkError("event sink record cell accounting is invalid")
                payload_bytes = count * EVENT_SINK_ROW_BYTES
                payload = stream.read(payload_bytes)
                if len(payload) != payload_bytes:
                    raise EventSinkError("event sink record payload is truncated")
                row_counts[cell] += count
            return cells, row_counts
    except OSError as exc:
        raise EventSinkError(f"cannot read event sink {path}: {exc}") from exc


def event_sink_prefix_identity_v1(
    path: str | Path,
    *,
    cell_index: int,
    expected_rows: int,
) -> dict[str, Any]:
    """Authenticate one committed cell prefix without materializing its rows.

    The digest covers the canonical column-major bytes of every durable record
    in the selected prefix.  Records for other cells and uncommitted suffix
    records are deliberately excluded, so a checkpoint remains independently
    verifiable when a later owner has appended to the shared batch sink.
    """

    if type(cell_index) is not int or cell_index < 0:
        raise EventSinkError("event sink cell_index must be a nonnegative integer")
    if type(expected_rows) is not int or expected_rows < 0:
        raise EventSinkError("event sink expected_rows must be nonnegative")
    identities = event_sink_prefix_identities_v1(
        path,
        expected_rows_by_cell={cell_index: expected_rows},
    )
    return identities[cell_index]


def event_sink_prefix_identities_v1(
    path: str | Path,
    *,
    expected_rows_by_cell: Mapping[int, int],
) -> dict[int, dict[str, Any]]:
    """Authenticate several cell prefixes in one bounded streaming pass."""

    if (
        not isinstance(expected_rows_by_cell, Mapping)
        or not expected_rows_by_cell
        or any(type(cell) is not int or cell < 0 for cell in expected_rows_by_cell)
        or any(
            type(rows) is not int or rows < 0 for rows in expected_rows_by_cell.values()
        )
    ):
        raise EventSinkError("event sink expected prefix map is invalid")
    target = Path(path).expanduser().resolve()
    digests: dict[int, Any] = {}
    committed_rows = {cell: 0 for cell in expected_rows_by_cell}
    for cell, rows in expected_rows_by_cell.items():
        digest = hashlib.sha256()
        digest.update(b"eiel_event_sink_cell_prefix_v1\0")
        digest.update(struct.pack("<QQ", cell, rows))
        digests[cell] = digest
    try:
        with target.open("rb") as stream:
            raw_header = stream.read(EVENT_SINK_HEADER.size)
            if len(raw_header) != EVENT_SINK_HEADER.size:
                raise EventSinkError("event sink header is truncated")
            magic, version, cells, columns, row_bytes = EVENT_SINK_HEADER.unpack(
                raw_header
            )
            if magic != EVENT_SINK_MAGIC or version != 1:
                raise EventSinkError("event sink header schema is unsupported")
            if columns != len(EVENT_COLUMN_NAMES) or row_bytes != EVENT_SINK_ROW_BYTES:
                raise EventSinkError("event sink header column contract is invalid")
            if cells and any(cell >= cells for cell in expected_rows_by_cell):
                raise EventSinkError("event sink cell_index is outside its header")
            while True:
                raw_record = stream.read(EVENT_SINK_RECORD.size)
                if not raw_record:
                    break
                if len(raw_record) != EVENT_SINK_RECORD.size:
                    if committed_rows == dict(expected_rows_by_cell):
                        break
                    raise EventSinkError("event sink record header is truncated")
                record_cell, reserved, count = EVENT_SINK_RECORD.unpack(raw_record)
                if reserved != 0 or (cells and record_cell >= cells):
                    raise EventSinkError("event sink record header is invalid")
                payload_bytes = count * EVENT_SINK_ROW_BYTES
                payload = stream.read(payload_bytes)
                if len(payload) != payload_bytes:
                    if committed_rows == dict(expected_rows_by_cell):
                        break
                    raise EventSinkError("event sink record payload is truncated")
                if record_cell not in expected_rows_by_cell:
                    continue
                expected_rows = expected_rows_by_cell[record_cell]
                if committed_rows[record_cell] >= expected_rows:
                    continue
                if committed_rows[record_cell] + count > expected_rows:
                    raise EventSinkError(
                        "checkpoint event frontier splits a durable sink record"
                    )
                digests[record_cell].update(struct.pack("<Q", count))
                digests[record_cell].update(payload)
                committed_rows[record_cell] += count
            mismatches = {
                cell: (expected_rows_by_cell[cell], committed_rows[cell])
                for cell in expected_rows_by_cell
                if committed_rows[cell] != expected_rows_by_cell[cell]
            }
            if mismatches:
                raise EventSinkError(
                    "event sink committed prefixes differ from checkpoint frontiers: "
                    f"{mismatches}"
                )
    except OSError as exc:
        raise EventSinkError(f"cannot authenticate event sink {target}: {exc}") from exc
    return {
        cell: {
            "schema_version": EVENT_SINK_SCHEMA_VERSION,
            "path": str(target),
            "cell_index": cell,
            "row_count": expected_rows_by_cell[cell],
            "row_bytes": EVENT_SINK_ROW_BYTES,
            "prefix_identity_schema": "eiel_event_sink_cell_prefix_v1",
            "prefix_sha256": digests[cell].hexdigest(),
        }
        for cell in expected_rows_by_cell
    }


def validate_event_sink_prefix_reference_v1(
    reference: Mapping[str, Any],
) -> dict[str, Any]:
    """Recompute a checkpoint-bound sink prefix and reject path corruption."""

    if not isinstance(reference, Mapping):
        raise EventSinkError("event sink checkpoint reference must be a mapping")
    if reference.get("schema_version") != EVENT_SINK_SCHEMA_VERSION:
        raise EventSinkError("event sink checkpoint reference schema is invalid")
    path = reference.get("path")
    cell_index = reference.get("cell_index")
    row_count = reference.get("row_count")
    supplied = reference.get("prefix_sha256")
    if (
        type(path) is not str
        or not path
        or type(cell_index) is not int
        or type(row_count) is not int
        or type(supplied) is not str
        or len(supplied) != 64
    ):
        raise EventSinkError("event sink checkpoint reference is incomplete")
    actual = event_sink_prefix_identity_v1(
        path,
        cell_index=cell_index,
        expected_rows=row_count,
    )
    if supplied != actual["prefix_sha256"]:
        raise EventSinkError("event sink committed prefix digest differs")
    return actual


def reconcile_event_sink_to_checkpoint_prefixes_v1(
    path: str | Path,
    *,
    references_by_cell: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    """Roll a crash-ahead sink back to authenticated checkpoint prefixes.

    The ahead image is retained beside the sink under a content-addressed
    filename before the compact committed image is atomically installed.
    This operation is intended for provider resume after all batch-cell
    checkpoint references have been collected and authenticated.
    """

    if (
        not isinstance(references_by_cell, Mapping)
        or not references_by_cell
        or any(type(cell) is not int or cell < 0 for cell in references_by_cell)
    ):
        raise EventSinkError("event sink reconciliation references are invalid")
    target = Path(path).expanduser().resolve()
    expected_rows: dict[int, int] = {}
    for cell, reference in references_by_cell.items():
        actual = validate_event_sink_prefix_reference_v1(reference)
        if actual["path"] != str(target) or actual["cell_index"] != cell:
            raise EventSinkError("event sink reconciliation reference path differs")
        expected_rows[cell] = actual["row_count"]

    try:
        original_sha256 = _file_sha256_v1(target)
        with target.open("rb") as source:
            raw_header = source.read(EVENT_SINK_HEADER.size)
            if len(raw_header) != EVENT_SINK_HEADER.size:
                raise EventSinkError("event sink header is truncated")
            magic, version, cells, columns, row_bytes = EVENT_SINK_HEADER.unpack(
                raw_header
            )
            if (
                magic != EVENT_SINK_MAGIC
                or version != 1
                or columns != len(EVENT_COLUMN_NAMES)
                or row_bytes != EVENT_SINK_ROW_BYTES
            ):
                raise EventSinkError("event sink header contract is invalid")
            if cells and set(references_by_cell) != set(range(cells)):
                raise EventSinkError(
                    "event sink reconciliation requires every declared batch cell"
                )
            retained_rows = {cell: 0 for cell in references_by_cell}
            suffix_records = 0
            fd, temp_name = tempfile.mkstemp(
                prefix=f".{target.name}.reconcile-",
                dir=target.parent,
            )
            try:
                with os.fdopen(fd, "wb") as destination:
                    destination.write(raw_header)
                    while True:
                        raw_record = source.read(EVENT_SINK_RECORD.size)
                        if not raw_record:
                            break
                        if len(raw_record) != EVENT_SINK_RECORD.size:
                            if retained_rows == expected_rows:
                                suffix_records += 1
                                break
                            raise EventSinkError(
                                "event sink record header is truncated"
                            )
                        cell, reserved, count = EVENT_SINK_RECORD.unpack(raw_record)
                        payload_bytes = count * EVENT_SINK_ROW_BYTES
                        payload = source.read(payload_bytes)
                        if len(payload) != payload_bytes:
                            if retained_rows == expected_rows:
                                suffix_records += 1
                                break
                            raise EventSinkError(
                                "event sink record payload is truncated"
                            )
                        if reserved != 0 or cell not in references_by_cell:
                            raise EventSinkError(
                                "event sink contains an unbound cell during reconciliation"
                            )
                        next_rows = retained_rows[cell] + count
                        if next_rows <= expected_rows[cell]:
                            destination.write(raw_record)
                            destination.write(payload)
                            retained_rows[cell] = next_rows
                        else:
                            suffix_records += 1
                    if retained_rows != expected_rows:
                        raise EventSinkError(
                            "event sink reconciliation could not retain exact prefixes"
                        )
                    destination.flush()
                    os.fsync(destination.fileno())
                if suffix_records:
                    backup = target.with_name(
                        f"{target.name}.uncommitted-ahead-{original_sha256[:16]}.bin"
                    )
                    if backup.exists():
                        if _file_sha256_v1(backup) != original_sha256:
                            raise EventSinkError(
                                "event sink recovery backup name collides"
                            )
                    else:
                        os.link(target, backup)
                        _fsync_directory_v1(target.parent)
                    os.replace(temp_name, target)
                    _fsync_directory_v1(target.parent)
                    temp_name = ""
                return {
                    "schema_version": "eiel_event_sink_reconciliation_v1",
                    "path": str(target),
                    "checkpoint_frontiers": dict(sorted(expected_rows.items())),
                    "suffix_records_removed": suffix_records,
                    "ahead_image_sha256": original_sha256,
                    "ahead_image_preserved": bool(suffix_records),
                }
            finally:
                if temp_name:
                    try:
                        os.unlink(temp_name)
                    except FileNotFoundError:
                        pass
    except OSError as exc:
        raise EventSinkError(f"cannot reconcile event sink {target}: {exc}") from exc


def _last_record_matches(
    path: Path,
    *,
    cell_index: int,
    columns: Sequence[torch.Tensor],
) -> bool:
    """Check a same-frontier retry against the last durable record."""

    last_payload_start: int | None = None
    last_rows = 0
    try:
        with path.open("rb") as stream:
            stream.seek(EVENT_SINK_HEADER.size)
            while True:
                raw_record = stream.read(EVENT_SINK_RECORD.size)
                if not raw_record:
                    break
                if len(raw_record) != EVENT_SINK_RECORD.size:
                    raise EventSinkError("event sink record header is truncated")
                cell, _reserved, count = EVENT_SINK_RECORD.unpack(raw_record)
                payload_start = stream.tell()
                payload_bytes = count * EVENT_SINK_ROW_BYTES
                if cell == cell_index:
                    last_payload_start = payload_start
                    last_rows = count
                stream.seek(payload_start + payload_bytes)
            if last_payload_start is None:
                return int(columns[0].numel()) == 0
            if last_rows != int(columns[0].numel()):
                return False
            column_offset = last_payload_start
            for name, column in zip(EVENT_COLUMN_NAMES, columns, strict=True):
                raw_expected = column.numpy().tobytes(order="C")
                stream.seek(column_offset)
                raw_actual = stream.read(len(raw_expected))
                if raw_actual != raw_expected:
                    return False
                column_offset += len(raw_expected)
            return True
    except OSError as exc:
        raise EventSinkError(f"cannot inspect event sink {path}: {exc}") from exc


def append_event_sink_columns_v1(
    path: str | Path,
    *,
    cell_index: int,
    columns: Sequence[Any],
    sequence_base: int,
    expected_cells: int | None = None,
) -> int:
    """Append one exact cell prefix and return its new absolute frontier.

    The operation is idempotent when the sink already contains exactly the
    requested frontier.  That matters because provider preparation may be
    retried after choosing a larger common physical event capacity.
    """

    if type(cell_index) is not int or cell_index < 0:
        raise EventSinkError("event sink cell_index must be a nonnegative integer")
    if type(sequence_base) is not int or sequence_base < 0:
        raise EventSinkError("event sink sequence_base must be nonnegative")
    normalized = _checked_columns(columns)
    row_count = int(normalized[0].numel())
    sequence = normalized[9]
    expected_sequence = torch.arange(
        sequence_base,
        sequence_base + row_count,
        dtype=torch.int64,
    )
    if not torch.equal(sequence, expected_sequence):
        raise EventSinkError("event sink sequence is not contiguous from its frontier")

    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    header_cells, row_counts = _read_state(target)
    if expected_cells is not None and (
        type(expected_cells) is not int or expected_cells < 1
    ):
        raise EventSinkError("expected event sink cell count is invalid")
    if header_cells and expected_cells is not None and header_cells != expected_cells:
        raise EventSinkError("event sink cell count changed")
    if header_cells and cell_index >= header_cells:
        raise EventSinkError("event sink cell_index is outside the existing sink")
    if expected_cells is not None and cell_index >= expected_cells:
        raise EventSinkError("event sink cell_index is outside the admitted batch")
    if len(row_counts) <= cell_index:
        row_counts.extend([0] * (cell_index + 1 - len(row_counts)))
    existing = row_counts[cell_index]
    if not target.exists():
        with target.open("wb") as stream:
            stream.write(
                EVENT_SINK_HEADER.pack(
                    EVENT_SINK_MAGIC,
                    1,
                    0 if expected_cells is None else expected_cells,
                    len(EVENT_COLUMN_NAMES),
                    EVENT_SINK_ROW_BYTES,
                )
            )
            stream.flush()
            os.fsync(stream.fileno())
    if existing == sequence_base + row_count:
        if row_count and not _last_record_matches(
            target,
            cell_index=cell_index,
            columns=normalized,
        ):
            raise EventSinkError(
                "event sink durable record disagrees with a same-frontier retry"
            )
        return existing
    if existing != sequence_base:
        raise EventSinkError(
            "event sink frontier disagrees with the source projection: "
            f"cell={cell_index} existing={existing} source={sequence_base}"
        )

    try:
        with target.open("ab") as stream:
            stream.write(EVENT_SINK_RECORD.pack(cell_index, 0, row_count))
            for column in normalized:
                stream.write(column.numpy().tobytes(order="C"))
            stream.flush()
            os.fsync(stream.fileno())
    except OSError as exc:
        raise EventSinkError(f"cannot append event sink {target}: {exc}") from exc
    return sequence_base + row_count


def iter_event_sink_columns_v1(
    path: str | Path,
    *,
    cell_index: int,
    expected_rows: int,
    start_row: int = 0,
    end_row: int | None = None,
    chunk_rows: int = 8192,
) -> Iterator[tuple[int, tuple[torch.Tensor, ...]]]:
    """Yield bounded slices from one cell without materializing its history."""

    if type(cell_index) is not int or cell_index < 0:
        raise EventSinkError("event sink cell_index must be a nonnegative integer")
    if type(expected_rows) is not int or expected_rows < 0:
        raise EventSinkError("event sink expected_rows must be nonnegative")
    if end_row is None:
        end_row = expected_rows
    if (
        type(start_row) is not int
        or type(end_row) is not int
        or start_row < 0
        or start_row > end_row
        or end_row > expected_rows
    ):
        raise EventSinkError("event sink read range is invalid")
    if type(chunk_rows) is not int or chunk_rows < 1:
        raise EventSinkError("event sink chunk_rows must be positive")

    target = Path(path).expanduser().resolve()
    try:
        with target.open("rb") as stream:
            raw_header = stream.read(EVENT_SINK_HEADER.size)
            if len(raw_header) != EVENT_SINK_HEADER.size:
                raise EventSinkError("event sink header is truncated")
            magic, version, cells, columns, row_bytes = EVENT_SINK_HEADER.unpack(
                raw_header
            )
            if magic != EVENT_SINK_MAGIC or version != 1:
                raise EventSinkError("event sink header schema is unsupported")
            if columns != len(EVENT_COLUMN_NAMES) or row_bytes != EVENT_SINK_ROW_BYTES:
                raise EventSinkError("event sink header column contract is invalid")
            if cells and cell_index >= cells:
                raise EventSinkError("event sink cell_index is outside its header")
            cell_frontier = 0
            while True:
                raw_record = stream.read(EVENT_SINK_RECORD.size)
                if not raw_record:
                    break
                if len(raw_record) != EVENT_SINK_RECORD.size:
                    raise EventSinkError("event sink record header is truncated")
                record_cell, reserved, record_count = EVENT_SINK_RECORD.unpack(
                    raw_record
                )
                if reserved != 0 or (cells and record_cell >= cells):
                    raise EventSinkError("event sink record header is invalid")
                payload_start = stream.tell()
                payload_bytes = record_count * EVENT_SINK_ROW_BYTES
                payload_end = payload_start + payload_bytes
                record_start = cell_frontier if record_cell == cell_index else 0
                record_end = (
                    record_start + record_count if record_cell == cell_index else 0
                )
                overlap_start = max(start_row, record_start)
                overlap_end = min(end_row, record_end)
                if record_cell == cell_index and overlap_start < overlap_end:
                    for row_start in range(overlap_start, overlap_end, chunk_rows):
                        row_end = min(row_start + chunk_rows, overlap_end)
                        result_columns: list[torch.Tensor] = []
                        column_offset = payload_start
                        for name in EVENT_COLUMN_NAMES:
                            dtype = EVENT_COLUMN_DTYPES[name]
                            item_bytes = torch.empty((), dtype=dtype).element_size()
                            stream.seek(
                                column_offset + (row_start - record_start) * item_bytes
                            )
                            raw_column = stream.read((row_end - row_start) * item_bytes)
                            expected_bytes = (row_end - row_start) * item_bytes
                            if len(raw_column) != expected_bytes:
                                raise EventSinkError(
                                    "event sink column payload is truncated"
                                )
                            result_columns.append(
                                torch.frombuffer(
                                    bytearray(raw_column), dtype=dtype
                                ).clone()
                            )
                            column_offset += record_count * item_bytes
                        yield row_start, tuple(result_columns)
                if record_cell == cell_index:
                    cell_frontier = record_end
                stream.seek(payload_end)
            if cell_frontier != expected_rows:
                raise EventSinkError(
                    "event sink frontier differs from the expected row count"
                )
    except OSError as exc:
        raise EventSinkError(f"cannot read event sink {target}: {exc}") from exc


def external_event_projection_v1(
    projection: Mapping[str, Any],
    *,
    sink_path: str | Path,
    cell_index: int,
    expected_cells: int | None = None,
) -> dict[str, Any]:
    """Move an inline canonical event prefix to the sink and compact the view."""

    metadata = dict(projection.get("metadata") or {})
    tensors = dict(projection.get("tensors") or {})
    columns = tuple(tensors.get(name) for name in EVENT_COLUMN_NAMES)
    if any(not isinstance(value, torch.Tensor) for value in columns):
        raise EventSinkError("inline event projection is incomplete")
    normalized = _checked_columns(columns)
    row_count = int(normalized[0].numel())
    frontier = int(
        metadata.get(
            "event_history_frontier",
            metadata.get("typed_lifecycle_event_next_sequence", row_count),
        )
        or 0
    )
    if frontier != row_count:
        raise EventSinkError(
            "inline event projection frontier must equal its stored row count"
        )
    new_frontier = append_event_sink_columns_v1(
        sink_path,
        cell_index=cell_index,
        columns=normalized,
        sequence_base=0,
        expected_cells=expected_cells,
    )
    compact_tensors = dict(tensors)
    for name in EVENT_COLUMN_NAMES:
        compact_tensors[name] = torch.empty(
            (0,), dtype=EVENT_COLUMN_DTYPES[name], device="cpu"
        )
    reference = event_sink_prefix_identity_v1(
        sink_path,
        cell_index=cell_index,
        expected_rows=new_frontier,
    )
    metadata.update(
        {
            "event_history_storage_mode": EVENT_SINK_SCHEMA_VERSION,
            "event_history_reference": reference,
            "event_history_frontier": new_frontier,
            "typed_lifecycle_event_sequence_base": new_frontier,
            "typed_lifecycle_event_next_sequence": new_frontier,
            "typed_lifecycle_event_active_count": 0,
            "event_tape_active_count": 0,
            "event_tape_cursor": 0,
            "event_tape_inactive_tail_count": 0,
            "event_cursor": 0,
            "runtime_event_cursor": max(
                int(projection.get("runtime_event_cursor", 0) or 0), new_frontier
            ),
        }
    )
    out = dict(projection)
    out["metadata"] = metadata
    out["tensors"] = compact_tensors
    if isinstance(projection.get("tensor_manifest"), Mapping):
        manifest = dict(projection["tensor_manifest"])
        for name in EVENT_COLUMN_NAMES:
            previous = manifest.get(name)
            if previous is not None and not isinstance(previous, Mapping):
                raise EventSinkError(
                    f"event sink tensor manifest row is invalid: {name}"
                )
            manifest[name] = _checkpoint_tensor_manifest_v1(
                compact_tensors[name],
                previous=previous,
            )
        out["tensor_manifest"] = manifest
    out["event_cursor"] = 0
    out["runtime_event_cursor"] = max(
        int(projection.get("runtime_event_cursor", 0) or 0), new_frontier
    )
    # Externalization changes both authoritative event tensors and metadata.
    # Keep the import-projection integrity seal bound to that compact image;
    # retaining the inline image's digest makes the first CUDA-owner import
    # fail closed even though the sink prefix itself is authentic.
    from engine.perf.cxxbackbone10_resident_state_owner import projection_digest

    metadata["import_digest"] = projection_digest(
        {"metadata": metadata, "tensors": compact_tensors}
    )
    return out


__all__ = [
    "EVENT_SINK_MAGIC",
    "EVENT_SINK_RECORD",
    "EVENT_SINK_ROW_BYTES",
    "EVENT_SINK_SCHEMA_VERSION",
    "EventSinkError",
    "append_event_sink_columns_v1",
    "event_history_digest_from_references_v1",
    "event_sink_prefix_identities_v1",
    "event_sink_prefix_identity_v1",
    "external_event_projection_v1",
    "iter_event_sink_columns_v1",
    "reconcile_event_sink_to_checkpoint_prefixes_v1",
    "validate_event_sink_prefix_reference_v1",
]
