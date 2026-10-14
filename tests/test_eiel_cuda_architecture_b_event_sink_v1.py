from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pytest
import torch

from engine.perf.cuda_architecture_b_boundary import (
    EVENT_COLUMN_DTYPES,
    EVENT_COLUMN_NAMES,
)
from engine.perf.cuda_architecture_b_event_sink_v1 import (
    EVENT_SINK_ROW_BYTES,
    EVENT_SINK_RECORD,
    EVENT_SINK_SCHEMA_VERSION,
    EventSinkError,
    _read_state,
    append_event_sink_columns_v1,
    event_history_digest_from_references_v1,
    event_sink_prefix_identity_v1,
    external_event_projection_v1,
    iter_event_sink_columns_v1,
    reconcile_event_sink_to_checkpoint_prefixes_v1,
    validate_event_sink_prefix_reference_v1,
)
from engine.campaigns.tick_native_control import (
    physical_b136_event_window_capacity_v1,
)

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


def _event_columns(rows: int, *, sequence_base: int = 0) -> tuple[torch.Tensor, ...]:
    values: list[torch.Tensor] = []
    sequence_name = "typed_lifecycle_event_sequence"
    for offset, name in enumerate(EVENT_COLUMN_NAMES):
        dtype = EVENT_COLUMN_DTYPES[name]
        if name == sequence_name:
            value = torch.arange(
                sequence_base,
                sequence_base + rows,
                dtype=dtype,
            )
        elif dtype.is_floating_point:
            value = torch.arange(rows, dtype=torch.float32).to(dtype) + offset
        else:
            value = torch.arange(rows, dtype=dtype) + offset
        values.append(value.contiguous())
    return tuple(values)


def _manifest(tensors: dict[str, torch.Tensor]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for name, tensor in tensors.items():
        raw = tensor.numpy().tobytes(order="C")
        result[name] = {
            "shape": list(tensor.shape),
            "dtype": _MANIFEST_DTYPE_NAMES[tensor.dtype],
            "device": "cpu",
            "numel": int(tensor.numel()),
            "element_size": int(tensor.element_size()),
            "bytes": len(raw),
            "is_cuda": False,
            "is_contiguous": True,
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
    return result


def test_physical_b136_event_bytes_use_exact_source_derived_capacity() -> None:
    rows = physical_b136_event_window_capacity_v1()

    assert rows == 41_804
    assert rows * EVENT_SINK_ROW_BYTES * 136 == 801_633_504


def test_event_sink_is_frontier_exact_and_idempotent(tmp_path: Path) -> None:
    path = tmp_path / "events.bin"
    first = _event_columns(3)

    assert (
        append_event_sink_columns_v1(
            path,
            cell_index=0,
            columns=first,
            sequence_base=0,
            expected_cells=2,
        )
        == 3
    )
    # Preparation retries must not duplicate an already committed prefix.
    assert (
        append_event_sink_columns_v1(
            path,
            cell_index=0,
            columns=first,
            sequence_base=0,
            expected_cells=2,
        )
        == 3
    )
    assert (
        append_event_sink_columns_v1(
            path,
            cell_index=1,
            columns=_event_columns(2),
            sequence_base=0,
            expected_cells=2,
        )
        == 2
    )
    assert _read_state(path) == (2, [3, 2])

    with pytest.raises(EventSinkError, match="frontier"):
        append_event_sink_columns_v1(
            path,
            cell_index=0,
            columns=_event_columns(1, sequence_base=4),
            sequence_base=3,
            expected_cells=2,
        )

    mismatched = list(first)
    mismatched[0] = mismatched[0].clone()
    mismatched[0][0] += 1
    with pytest.raises(EventSinkError, match="same-frontier"):
        append_event_sink_columns_v1(
            path,
            cell_index=0,
            columns=tuple(mismatched),
            sequence_base=0,
            expected_cells=2,
        )


def test_external_projection_is_compact_and_repairs_manifest(tmp_path: Path) -> None:
    columns = _event_columns(4)
    tensors = dict(zip(EVENT_COLUMN_NAMES, columns, strict=True))
    projection = {
        "schema_version": "cxxbackbone10_checkpoint_projection_v2",
        "metadata": {
            "event_history_frontier": 4,
            "typed_lifecycle_event_next_sequence": 4,
        },
        "tensors": tensors,
        "tensor_manifest": _manifest(tensors),
        "event_cursor": 4,
        "runtime_event_cursor": 4,
    }

    result = external_event_projection_v1(
        projection,
        sink_path=tmp_path / "events.bin",
        cell_index=0,
        expected_cells=1,
    )

    assert result["metadata"]["event_history_storage_mode"] == (
        EVENT_SINK_SCHEMA_VERSION
    )
    assert result["metadata"]["event_history_frontier"] == 4
    assert result["metadata"]["typed_lifecycle_event_sequence_base"] == 4
    assert result["metadata"]["typed_lifecycle_event_next_sequence"] == 4
    assert result["event_cursor"] == 0
    assert result["runtime_event_cursor"] == 4
    assert all(result["tensors"][name].numel() == 0 for name in EVENT_COLUMN_NAMES)
    assert all(
        result["tensor_manifest"][name]["shape"] == [0] for name in EVENT_COLUMN_NAMES
    )
    assert all(
        result["tensor_manifest"][name]["bytes"] == 0 for name in EVENT_COLUMN_NAMES
    )
    assert result["metadata"]["event_history_reference"]["row_count"] == 4
    assert result["metadata"]["event_history_reference"]["row_bytes"] == (
        EVENT_SINK_ROW_BYTES
    )
    assert _read_state(tmp_path / "events.bin") == (1, [4])
    from engine.perf.cxxbackbone10_resident_state_owner import projection_digest

    assert result["metadata"]["import_digest"] == projection_digest(result)
    assert all(
        torch.equal(projection["tensors"][name], columns[index])
        for index, name in enumerate(EVENT_COLUMN_NAMES)
    )


def test_event_sink_reader_is_bounded_and_preserves_absolute_rows(
    tmp_path: Path,
) -> None:
    rows = 17
    path = tmp_path / "events.bin"
    append_event_sink_columns_v1(
        path,
        cell_index=0,
        columns=_event_columns(rows),
        sequence_base=0,
        expected_cells=1,
    )

    chunks = tuple(
        iter_event_sink_columns_v1(
            path,
            cell_index=0,
            expected_rows=rows,
            start_row=3,
            end_row=15,
            chunk_rows=4,
        )
    )
    assert [start for start, _columns in chunks] == [3, 7, 11]
    assert [int(columns[0].numel()) for _start, columns in chunks] == [4, 4, 4]
    sequence_index = EVENT_COLUMN_NAMES.index("typed_lifecycle_event_sequence")
    assert torch.equal(
        torch.cat([columns[sequence_index] for _start, columns in chunks]),
        torch.arange(3, 15, dtype=torch.int64),
    )


def test_event_sink_checkpoint_prefix_rejects_same_length_corruption(
    tmp_path: Path,
) -> None:
    path = tmp_path / "events.bin"
    append_event_sink_columns_v1(
        path,
        cell_index=0,
        columns=_event_columns(3),
        sequence_base=0,
        expected_cells=1,
    )
    reference = event_sink_prefix_identity_v1(
        path,
        cell_index=0,
        expected_rows=3,
    )
    assert validate_event_sink_prefix_reference_v1(reference) == reference

    with path.open("r+b") as stream:
        stream.seek(24 + 16)
        original = stream.read(1)
        stream.seek(24 + 16)
        stream.write(bytes([original[0] ^ 1]))
    with pytest.raises(EventSinkError, match="digest differs"):
        validate_event_sink_prefix_reference_v1(reference)


def test_empty_event_sink_establishes_checkpoint_bound_authority(
    tmp_path: Path,
) -> None:
    projection = {
        "metadata": {
            "event_history_frontier": 0,
            "typed_lifecycle_event_next_sequence": 0,
        },
        "tensors": dict(zip(EVENT_COLUMN_NAMES, _event_columns(0), strict=True)),
        "event_cursor": 0,
        "runtime_event_cursor": 0,
    }
    result = external_event_projection_v1(
        projection,
        sink_path=tmp_path / "events.bin",
        cell_index=0,
        expected_cells=1,
    )
    reference = result["metadata"]["event_history_reference"]
    assert reference["row_count"] == 0
    assert validate_event_sink_prefix_reference_v1(reference) == reference


def test_event_history_digest_is_path_invariant_but_content_bound() -> None:
    base = {
        "schema_version": EVENT_SINK_SCHEMA_VERSION,
        "cell_index": 0,
        "row_count": 4,
        "row_bytes": EVENT_SINK_ROW_BYTES,
        "prefix_identity_schema": "eiel_event_sink_cell_prefix_v1",
        "prefix_sha256": "a" * 64,
    }
    left = event_history_digest_from_references_v1(
        [{**base, "path": "/run-a/events.bin"}]
    )
    right = event_history_digest_from_references_v1(
        [{**base, "path": "/run-b/events.bin"}]
    )
    changed = event_history_digest_from_references_v1(
        [{**base, "path": "/run-b/events.bin", "prefix_sha256": "b" * 64}]
    )

    assert left == right
    assert changed != left


def test_event_sink_crash_ahead_rolls_back_to_authenticated_batch_prefixes(
    tmp_path: Path,
) -> None:
    path = tmp_path / "events.bin"
    for cell in range(2):
        append_event_sink_columns_v1(
            path,
            cell_index=cell,
            columns=_event_columns(2),
            sequence_base=0,
            expected_cells=2,
        )
    references = {
        cell: event_sink_prefix_identity_v1(
            path,
            cell_index=cell,
            expected_rows=2,
        )
        for cell in range(2)
    }
    for cell in range(2):
        append_event_sink_columns_v1(
            path,
            cell_index=cell,
            columns=_event_columns(1, sequence_base=2),
            sequence_base=2,
            expected_cells=2,
        )

    receipt = reconcile_event_sink_to_checkpoint_prefixes_v1(
        path,
        references_by_cell=references,
    )
    assert receipt["suffix_records_removed"] == 2
    assert receipt["ahead_image_preserved"] is True
    assert _read_state(path) == (2, [2, 2])
    assert all(
        validate_event_sink_prefix_reference_v1(reference) == reference
        for reference in references.values()
    )
    assert len(tuple(tmp_path.glob("events.bin.uncommitted-ahead-*.bin"))) == 1


def test_event_sink_reconciliation_discards_torn_trailing_append(
    tmp_path: Path,
) -> None:
    path = tmp_path / "events.bin"
    append_event_sink_columns_v1(
        path,
        cell_index=0,
        columns=_event_columns(2),
        sequence_base=0,
        expected_cells=1,
    )
    reference = event_sink_prefix_identity_v1(path, cell_index=0, expected_rows=2)
    with path.open("ab") as stream:
        stream.write(EVENT_SINK_RECORD.pack(0, 0, 2))
        stream.write(b"torn")

    receipt = reconcile_event_sink_to_checkpoint_prefixes_v1(
        path, references_by_cell={0: reference}
    )

    assert receipt["suffix_records_removed"] == 1
    assert validate_event_sink_prefix_reference_v1(reference) == reference
    assert _read_state(path) == (1, [2])
    assert len(tuple(tmp_path.glob("events.bin.uncommitted-ahead-*.bin"))) == 1
