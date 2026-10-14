"""Focused contract tests for the standalone physical-batch packed transport."""

from __future__ import annotations

import hashlib
import io
from pathlib import Path

import pytest
import torch

import engine.campaigns.acs_physical_batch_packed_checkpoint_v1 as packed
from engine.campaigns.acs_physical_batch_packed_checkpoint_v1 import (
    MAX_PHYSICAL_PAYLOAD_BYTES,
    PREFIX,
    PhysicalBatchPackedCheckpointError,
    RaggedFieldV1,
    ZeroFillFieldV1,
    capture_physical_batch_packed_checkpoint_v1,
    finalize_physical_batch_packed_checkpoint_v1,
    load_physical_batch_packed_checkpoint_v1,
)


def _one_common_field_header(*, payload_bytes: int) -> dict[str, object]:
    return {
        "schema_version": packed.SCHEMA_VERSION,
        "format_version": packed.FORMAT_VERSION,
        "endianness": "little",
        "payload_layout": packed.PAYLOAD_LAYOUT,
        "capture_status": "captured_unsealed",
        "batch_id": "boundary",
        "cell_ids": ["cell-0"],
        "batch_cells": 1,
        "field_order": ["state"],
        "fields": [
            {
                "name": "state",
                "kind": "common",
                "storage": "payload",
                "dtype": "uint8",
                "dtype_code": packed.streamed._DTYPE_CODE["uint8"],
                "shape": [1, payload_bytes],
                "cell_shape": [payload_bytes],
                "cell_bytes": payload_bytes,
                "cell_order": "cell_ids_order_v1",
                "offset": 0,
                "bytes": payload_bytes,
                "logical_bytes": payload_bytes,
            }
        ],
        "scalar_metadata": {},
        "payload_bytes": payload_bytes,
        "operation_counts": {},
    }


def test_physical_payload_cap_accepts_exact_boundary_without_allocating_payload() -> None:
    assert MAX_PHYSICAL_PAYLOAD_BYTES == 8 * 1024**3
    header = _one_common_field_header(
        payload_bytes=MAX_PHYSICAL_PAYLOAD_BYTES
    )
    cell_ids, fields = packed._validate_container_identity(
        header, payload_bytes=MAX_PHYSICAL_PAYLOAD_BYTES
    )
    assert cell_ids == ["cell-0"]
    logical_zero_bytes, zero_fields = packed._validate_ranges(
        fields,
        batch_cells=1,
        payload_bytes=MAX_PHYSICAL_PAYLOAD_BYTES,
    )
    assert logical_zero_bytes == 0
    assert zero_fields == []


def test_physical_payload_cap_rejects_over_cap_shape_before_capture_writes(
    tmp_path: Path,
) -> None:
    target = tmp_path / "over-cap-shape.acspb"
    over_cap = torch.empty(
        (1, MAX_PHYSICAL_PAYLOAD_BYTES + 1),
        dtype=torch.uint8,
        device="meta",
    )
    with pytest.raises(PhysicalBatchPackedCheckpointError, match="physical cap"):
        capture_physical_batch_packed_checkpoint_v1(
            {"state": over_cap},
            target,
            cell_ids=("cell-0",),
            batch_id="over-cap-shape",
        )
    assert not target.exists()
    assert not tuple(tmp_path.glob(".*.capture"))


def test_physical_payload_cap_rejects_over_cap_container_and_sink_write() -> None:
    with pytest.raises(PhysicalBatchPackedCheckpointError, match="physical cap"):
        packed._validate_container_identity(
            _one_common_field_header(
                payload_bytes=MAX_PHYSICAL_PAYLOAD_BYTES + 1
            ),
            payload_bytes=MAX_PHYSICAL_PAYLOAD_BYTES + 1,
        )

    sink_stream = io.BytesIO()
    sink = packed._PayloadSink(
        sink_stream, payload_offset=MAX_PHYSICAL_PAYLOAD_BYTES
    )
    with pytest.raises(PhysicalBatchPackedCheckpointError, match="physical cap"):
        sink.write(b"x")
    assert sink_stream.getvalue() == b""


def test_zero_fill_logical_bytes_do_not_consume_physical_payload_cap(
    tmp_path: Path,
) -> None:
    target = tmp_path / "zero-fill-over-logical-cap.acspb"
    descriptor = capture_physical_batch_packed_checkpoint_v1(
        {
            "gru_pack": ZeroFillFieldV1(
                torch.uint8, (1, MAX_PHYSICAL_PAYLOAD_BYTES + 1)
            )
        },
        target,
        cell_ids=("cell-0",),
        batch_id="zero-fill-over-logical-cap",
    )
    assert descriptor["payload_bytes"] == 0
    assert descriptor["zero_fill_logical_bytes"] == MAX_PHYSICAL_PAYLOAD_BYTES + 1


def test_finalize_and_load_reject_over_cap_prefix_before_file_sizing(
    tmp_path: Path,
) -> None:
    target = tmp_path / "over-cap-prefix.acspb"
    target.write_bytes(
        PREFIX.pack(
            packed.MAGIC,
            packed.FORMAT_VERSION,
            packed.LITTLE_ENDIAN_MARKER,
            1,
            MAX_PHYSICAL_PAYLOAD_BYTES + 1,
            0,
        )
        + b"x"
    )
    descriptor = {
        "schema_version": packed.SCHEMA_VERSION,
        "descriptor_status": "captured_unsealed",
        "capture_path": str(target),
    }
    with pytest.raises(PhysicalBatchPackedCheckpointError, match="physical cap"):
        finalize_physical_batch_packed_checkpoint_v1(descriptor)
    with pytest.raises(PhysicalBatchPackedCheckpointError, match="physical cap"):
        load_physical_batch_packed_checkpoint_v1(target)


def _capture_and_publish(
    tmp_path: Path,
    fields: dict[str, object],
    *,
    cell_ids: tuple[str, ...],
    batch_id: str = "batch-test",
    chunk_bytes: int = 8,
) -> tuple[dict[str, object], Path]:
    target = tmp_path / "published.acspb"
    descriptor = capture_physical_batch_packed_checkpoint_v1(
        fields,
        target,
        cell_ids=cell_ids,
        batch_id=batch_id,
        scalar_metadata={"absolute_next_tick": 1120, "route": "test"},
        chunk_bytes=chunk_bytes,
    )
    assert descriptor["sealed"] is False
    assert Path(str(descriptor["capture_path"])).exists()
    receipt = finalize_physical_batch_packed_checkpoint_v1(
        descriptor, publication_path=target
    )
    assert receipt["sealed"] is True
    assert target.exists()
    return descriptor, target


def test_common_fields_are_one_b_wide_transfer_each_and_round_trip(tmp_path: Path) -> None:
    batch_cells = 32
    cell_ids = tuple(f"cell-{index}" for index in range(batch_cells))
    fields = {
        "weights": torch.arange(batch_cells * 3, dtype=torch.float32).reshape(
            batch_cells, 3
        ),
        "mask": torch.arange(batch_cells * 2, dtype=torch.int32).reshape(
            batch_cells, 2
        ),
        "wide": torch.ones((batch_cells, 2, 2), dtype=torch.float64),
    }

    descriptor, target = _capture_and_publish(
        tmp_path, fields, cell_ids=cell_ids, chunk_bytes=16
    )
    counts = descriptor["operation_counts"]
    assert counts["common_field_transfer_operations"] == 3
    assert counts["ragged_item_transfer_operations"] == 0
    assert counts["total_tensor_transfer_operations"] == 3
    # The operation count is independent of B; chunking is a transport detail.
    assert counts["total_tensor_transfer_operations"] < batch_cells * len(fields)

    loaded = load_physical_batch_packed_checkpoint_v1(target)
    assert loaded["cell_ids"] == list(cell_ids)
    assert loaded["field_order"] == ["mask", "weights", "wide"]
    for name, source in fields.items():
        torch.testing.assert_close(loaded["fields"][name], source)
    assert loaded["scalar_metadata"] == {
        "absolute_next_tick": 1120,
        "route": "test",
    }


def test_unsigned_rng_and_identity_fields_round_trip_bit_exactly(
    tmp_path: Path,
) -> None:
    fields = {
        "u16": torch.tensor([[0, 2**16 - 1], [17, 23]], dtype=torch.uint16),
        "u32": torch.tensor([[0, 2**32 - 1], [29, 31]], dtype=torch.uint32),
        "u64": torch.tensor([[0, 2**64 - 1], [37, 41]], dtype=torch.uint64),
    }
    _, target = _capture_and_publish(
        tmp_path,
        fields,
        cell_ids=("a", "b"),
        chunk_bytes=7,
    )

    loaded = load_physical_batch_packed_checkpoint_v1(target)
    for name, source in fields.items():
        assert loaded["fields"][name].dtype == source.dtype
        assert torch.equal(loaded["fields"][name], source)


def test_ragged_declared_sequence_and_selected_cell_compatibility(tmp_path: Path) -> None:
    cell_ids = ("a", "b", "c")
    ragged = RaggedFieldV1(
        (
            torch.tensor([1, 2], dtype=torch.int64),
            torch.tensor([[3, 4], [5, 6]], dtype=torch.int64),
            torch.tensor([], dtype=torch.int64),
        )
    )
    common = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    descriptor, target = _capture_and_publish(
        tmp_path,
        {"events": ragged, "state": common},
        cell_ids=cell_ids,
        chunk_bytes=7,
    )
    counts = descriptor["operation_counts"]
    assert counts["common_field_transfer_operations"] == 1
    assert counts["ragged_item_transfer_operations"] == 3
    assert counts["total_tensor_transfer_operations"] == 4

    whole = load_physical_batch_packed_checkpoint_v1(target)
    assert len(whole["fields"]["events"]) == 3
    torch.testing.assert_close(whole["fields"]["events"][1], ragged.values[1])
    torch.testing.assert_close(whole["fields"]["state"], common)

    selected = load_physical_batch_packed_checkpoint_v1(target, selected_cell="b")
    assert selected["selected_cell_index"] == 1
    assert selected["selected_cell_id"] == "b"
    torch.testing.assert_close(selected["fields"]["state"], common[1])
    torch.testing.assert_close(selected["fields"]["events"], ragged.values[1])


def test_dynamic_common_shapes_and_dynamic_ragged_shapes_are_declared_per_field(
    tmp_path: Path,
) -> None:
    cell_ids = ("0", "1", "2")
    matrix = torch.arange(18, dtype=torch.float32).reshape(3, 2, 3)
    vector = torch.arange(15, dtype=torch.float32).reshape(3, 5)
    ragged = RaggedFieldV1(
        (
            torch.ones((1, 2), dtype=torch.float32),
            torch.ones((4,), dtype=torch.float32),
            torch.ones((0, 3), dtype=torch.float32),
        )
    )
    _, target = _capture_and_publish(
        tmp_path,
        {"matrix": matrix, "vector": vector, "ragged": ragged},
        cell_ids=cell_ids,
        chunk_bytes=11,
    )
    loaded = load_physical_batch_packed_checkpoint_v1(target)
    torch.testing.assert_close(loaded["fields"]["matrix"], matrix)
    torch.testing.assert_close(loaded["fields"]["vector"], vector)
    assert [tuple(value.shape) for value in loaded["fields"]["ragged"]] == [
        (1, 2),
        (4,),
        (0, 3),
    ]


def test_zero_fill_capture_allocates_no_tensor_or_slab_and_hashes_logical_zeros(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    batch_cells = 136
    shape = (batch_cells, 2, 3)
    logical_bytes = batch_cells * 2 * 3 * 4
    zero_field = ZeroFillFieldV1(torch.float32, shape)

    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("zero-fill capture allocated a tensor or slab copier")

    monkeypatch.setattr(packed, "_SharedTwoSlabCopier", forbidden)
    monkeypatch.setattr(packed.torch, "empty", forbidden)
    monkeypatch.setattr(packed.torch, "zeros", forbidden)
    target = tmp_path / "zero-only.acspb"
    descriptor = capture_physical_batch_packed_checkpoint_v1(
        {"gru_pack": zero_field},
        target,
        cell_ids=tuple(f"cell-{index}" for index in range(batch_cells)),
        batch_id="b136-zero-only",
        chunk_bytes=16,
    )
    monkeypatch.undo()

    assert descriptor["payload_bytes"] == 0
    assert descriptor["zero_fill_logical_bytes"] == logical_bytes
    assert descriptor["zero_fill_fields"] == ["gru_pack"]
    assert descriptor["operation_counts"]["zero_fill_transfer_operations"] == 0
    assert descriptor["operation_counts"]["total_tensor_transfer_operations"] == 0
    assert descriptor["slab_pool"]["slab_count"] == 0

    finalize_physical_batch_packed_checkpoint_v1(
        descriptor, publication_path=target
    )
    whole = load_physical_batch_packed_checkpoint_v1(target)
    expected_digest = hashlib.sha256(bytes(logical_bytes)).hexdigest()
    assert whole["footer"]["payload_sha256"] == hashlib.sha256(b"").hexdigest()
    assert whole["footer"]["zero_fill_logical_sha256"] == {
        "gru_pack": expected_digest
    }
    assert whole["footer"]["field_sha256"]["gru_pack"] == expected_digest
    assert tuple(whole["fields"]["gru_pack"].shape) == shape
    assert whole["fields"]["gru_pack"].dtype is torch.float32
    assert torch.count_nonzero(whole["fields"]["gru_pack"]).item() == 0

    selected = load_physical_batch_packed_checkpoint_v1(
        target, selected_cell="cell-91"
    )
    assert tuple(selected["fields"]["gru_pack"].shape) == (2, 3)
    assert selected["fields"]["gru_pack"].numel() == 6
    assert torch.count_nonzero(selected["fields"]["gru_pack"]).item() == 0


def test_zero_fill_between_physical_fields_preserves_exact_range_coverage(
    tmp_path: Path,
) -> None:
    batch_cells = 4
    before = torch.arange(8, dtype=torch.int32).reshape(batch_cells, 2)
    after = torch.arange(12, dtype=torch.float32).reshape(batch_cells, 3)
    descriptor, target = _capture_and_publish(
        tmp_path,
        {
            "a_before": before,
            "gru_pack": ZeroFillFieldV1(torch.float32, (batch_cells, 5)),
            "z_after": after,
        },
        cell_ids=tuple(f"cell-{index}" for index in range(batch_cells)),
        chunk_bytes=7,
    )
    physical_bytes = before.numel() * before.element_size() + after.numel() * after.element_size()
    assert descriptor["payload_bytes"] == physical_bytes
    assert descriptor["operation_counts"]["common_field_transfer_operations"] == 2
    assert descriptor["operation_counts"]["zero_fill_transfer_operations"] == 0
    assert descriptor["operation_counts"]["total_tensor_transfer_operations"] == 2
    loaded = load_physical_batch_packed_checkpoint_v1(target)
    torch.testing.assert_close(loaded["fields"]["a_before"], before)
    torch.testing.assert_close(loaded["fields"]["z_after"], after)
    assert torch.count_nonzero(loaded["fields"]["gru_pack"]).item() == 0


def test_zero_fill_authority_and_full_batch_shape_are_strict(tmp_path: Path) -> None:
    with pytest.raises(PhysicalBatchPackedCheckpointError, match="contract"):
        ZeroFillFieldV1(
            torch.float32, (2, 3), contract="unproven_zero_contract"
        )
    with pytest.raises(PhysicalBatchPackedCheckpointError, match="outside"):
        capture_physical_batch_packed_checkpoint_v1(
            {"ordinary_state": ZeroFillFieldV1(torch.float32, (2, 3))},
            tmp_path / "outside-authority.acspb",
            cell_ids=("a", "b"),
            batch_id="bad-zero-authority",
        )
    with pytest.raises(PhysicalBatchPackedCheckpointError, match="shape"):
        capture_physical_batch_packed_checkpoint_v1(
            {"hmem": ZeroFillFieldV1(torch.float32, (3, 3))},
            tmp_path / "wrong-b.acspb",
            cell_ids=("a", "b"),
            batch_id="bad-zero-shape",
        )


def test_zero_fill_descriptor_and_footer_digest_tampering_fail_closed(
    tmp_path: Path,
) -> None:
    _, target = _capture_and_publish(
        tmp_path,
        {"hmem": ZeroFillFieldV1(torch.float32, (3, 4))},
        cell_ids=("a", "b", "c"),
    )
    original = target.read_bytes()
    descriptor_marker = b'"zero_fill":{"kind":"exact_zero","value":0}'
    assert descriptor_marker in original
    changed_marker = b'"zero_fill":{"kind":"exact_zero","value":1}'
    target.write_bytes(original.replace(descriptor_marker, changed_marker, 1))
    with pytest.raises(PhysicalBatchPackedCheckpointError, match="zero-fill|seal"):
        load_physical_batch_packed_checkpoint_v1(target)

    target.write_bytes(original)
    loaded = load_physical_batch_packed_checkpoint_v1(target)
    digest = loaded["footer"]["zero_fill_logical_sha256"]["hmem"]
    raw = bytearray(original)
    digest_offset = raw.rfind(digest.encode("ascii"))
    assert digest_offset >= 0
    raw[digest_offset] = ord("0") if raw[digest_offset] != ord("0") else ord("1")
    target.write_bytes(raw)
    with pytest.raises(PhysicalBatchPackedCheckpointError, match="zero-fill|seal|digest"):
        load_physical_batch_packed_checkpoint_v1(target)


def test_shared_copier_validates_each_tensor_own_sink_byte_delta() -> None:
    class FaultyNativePool:
        def __init__(self) -> None:
            self.calls = 0

        def copy_tensor_chunks(self, _tensor: torch.Tensor, sink: object) -> int:
            self.calls += 1
            if self.calls == 1:
                sink.write(bytes(16))
                return 2
            # Return the expected chunk count but write no bytes.  A cumulative
            # sink check would incorrectly accept this after the first tensor.
            return 1

    copier = packed._SharedTwoSlabCopier(
        chunk_bytes=8, has_cuda_fields=False
    )
    copier.native_pool = FaultyNativePool()
    copier.fallback_pool = None
    sink = packed._PayloadSink(io.BytesIO())
    copier.copy_tensor(torch.arange(4, dtype=torch.int32), sink)
    with pytest.raises(PhysicalBatchPackedCheckpointError, match="byte delta"):
        copier.copy_tensor(torch.ones(1, dtype=torch.int32), sink)


def test_required_native_pool_fails_before_creating_capture_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(packed.streamed, "_native_slab_pool_type", lambda: None)
    target = tmp_path / "native-required.acspb"

    with pytest.raises(PhysicalBatchPackedCheckpointError, match="requires the native"):
        capture_physical_batch_packed_checkpoint_v1(
            {"state": torch.zeros((2, 2), dtype=torch.float32)},
            target,
            cell_ids=("a", "b"),
            batch_id="native-required",
            require_native_pool=True,
        )

    assert not target.exists()
    assert not tuple(tmp_path.glob(".*.capture"))


def test_native_pool_telemetry_preserves_cuda_event_pipeline_counters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class NativePool:
        def __init__(self, _chunk_bytes: int, _pinned: bool) -> None:
            pass

        def telemetry(self) -> dict[str, object]:
            return {
                "pinned": True,
                "preallocated": True,
                "chunk_count": 7,
                "copy_operations": 7,
                "copied_bytes": 448,
                "field_count": 2,
                "cuda_event_wait_count": 7,
                "final_cuda_fence_count": 2,
                "per_chunk_stream_synchronizations": 0,
                "source_device": "cuda:0",
            }

    monkeypatch.setattr(
        packed.streamed, "_native_slab_pool_type", lambda: NativePool
    )
    copier = packed._SharedTwoSlabCopier(
        chunk_bytes=64, has_cuda_fields=True, require_native_pool=True
    )

    assert copier.telemetry() == {
        "schema_version": "eiel_acs_physical_batch_packed_slab_pool_v1",
        "slab_count": 2,
        "native_pool": True,
        "pinned": True,
        "preallocated": True,
        "chunk_bytes": 64,
        "chunk_count": 7,
        "copy_operations": 7,
        "copied_bytes": 448,
        "field_count": 2,
        "cuda_event_wait_count": 7,
        "final_cuda_fence_count": 2,
        "per_chunk_stream_synchronizations": 0,
        "source_device": "cuda:0",
    }


def test_producer_returns_only_scalar_file_descriptor_and_strict_load_rejects_unsealed(
    tmp_path: Path,
) -> None:
    descriptor = capture_physical_batch_packed_checkpoint_v1(
        {"state": torch.zeros((2, 2), dtype=torch.float32)},
        tmp_path / "published.acspb",
        cell_ids=("a", "b"),
        batch_id="unsealed",
    )
    capture_path = Path(str(descriptor["capture_path"]))
    assert not (tmp_path / "published.acspb").exists()
    with pytest.raises(PhysicalBatchPackedCheckpointError, match="not sealed"):
        load_physical_batch_packed_checkpoint_v1(capture_path)


def test_payload_corruption_is_detected_by_exact_field_digest(tmp_path: Path) -> None:
    _, target = _capture_and_publish(
        tmp_path,
        {"state": torch.arange(12, dtype=torch.int32).reshape(3, 4)},
        cell_ids=("a", "b", "c"),
    )
    raw = bytearray(target.read_bytes())
    # Prefix is followed immediately by the field-major payload.
    raw[PREFIX.size + 2] ^= 0x01
    target.write_bytes(raw)
    with pytest.raises(PhysicalBatchPackedCheckpointError, match="digest"):
        load_physical_batch_packed_checkpoint_v1(target)


def test_truncation_is_fail_closed(tmp_path: Path) -> None:
    _, target = _capture_and_publish(
        tmp_path,
        {"state": torch.arange(12, dtype=torch.int32).reshape(3, 4)},
        cell_ids=("a", "b", "c"),
    )
    target.write_bytes(target.read_bytes()[:-9])
    with pytest.raises(PhysicalBatchPackedCheckpointError, match="size|truncated"):
        load_physical_batch_packed_checkpoint_v1(target)


def test_torn_seal_with_footer_present_but_uncommitted_prefix_is_rejected(
    tmp_path: Path,
) -> None:
    _, target = _capture_and_publish(
        tmp_path,
        {"state": torch.arange(6, dtype=torch.int32).reshape(3, 2)},
        cell_ids=("a", "b", "c"),
    )
    raw = bytearray(target.read_bytes())
    magic, version, endian, header_len, payload_len, _footer_len = PREFIX.unpack(
        raw[: PREFIX.size]
    )
    raw[: PREFIX.size] = PREFIX.pack(
        magic, version, endian, header_len, payload_len, 0
    )
    target.write_bytes(raw)
    with pytest.raises(PhysicalBatchPackedCheckpointError, match="size|sealed"):
        load_physical_batch_packed_checkpoint_v1(target)


def test_footer_tampering_is_rejected_as_torn_or_invalid_seal(tmp_path: Path) -> None:
    _, target = _capture_and_publish(
        tmp_path,
        {"state": torch.arange(6, dtype=torch.int32).reshape(3, 2)},
        cell_ids=("a", "b", "c"),
    )
    raw = bytearray(target.read_bytes())
    raw[-1] ^= 0x01
    target.write_bytes(raw)
    with pytest.raises(PhysicalBatchPackedCheckpointError, match="footer|seal|digest"):
        load_physical_batch_packed_checkpoint_v1(target)


def test_tensor_or_callable_metadata_and_implicit_contiguous_copy_are_rejected(
    tmp_path: Path,
) -> None:
    with pytest.raises(PhysicalBatchPackedCheckpointError, match="tensor"):
        capture_physical_batch_packed_checkpoint_v1(
            {"state": torch.zeros((2, 2))},
            tmp_path / "tensor-metadata.acspb",
            cell_ids=("a", "b"),
            batch_id="bad",
            scalar_metadata={"live": torch.zeros(1)},
        )
    with pytest.raises(PhysicalBatchPackedCheckpointError, match="callable"):
        capture_physical_batch_packed_checkpoint_v1(
            {"state": torch.zeros((2, 2))},
            tmp_path / "callable-metadata.acspb",
            cell_ids=("a", "b"),
            batch_id="bad",
            scalar_metadata={"callback": lambda: None},
        )
    non_contiguous = torch.arange(8, dtype=torch.float32).reshape(2, 4)[:, ::2]
    assert not non_contiguous.is_contiguous()
    with pytest.raises(PhysicalBatchPackedCheckpointError, match="contiguous"):
        capture_physical_batch_packed_checkpoint_v1(
            {"state": non_contiguous},
            tmp_path / "non-contiguous.acspb",
            cell_ids=("a", "b"),
            batch_id="bad",
        )


def test_finalizer_performs_host_seal_and_publication_after_capture(tmp_path: Path) -> None:
    descriptor = capture_physical_batch_packed_checkpoint_v1(
        {"state": torch.ones((2, 2), dtype=torch.float32)},
        tmp_path / "final.acspb",
        cell_ids=("a", "b"),
        batch_id="finalizer",
    )
    capture_path = Path(str(descriptor["capture_path"]))
    target = tmp_path / "final.acspb"
    assert capture_path.exists()
    assert not target.exists()
    receipt = finalize_physical_batch_packed_checkpoint_v1(
        descriptor, publication_path=target
    )
    assert receipt["published_path"] == str(target.resolve())
    assert not capture_path.exists()
    loaded = load_physical_batch_packed_checkpoint_v1(target)
    assert loaded["footer"]["status"] == "complete"


def test_strict_reader_hashes_once_for_multiple_selected_cells(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, target = _capture_and_publish(
        tmp_path,
        {"state": torch.arange(12, dtype=torch.int32).reshape(3, 4)},
        cell_ids=("a", "b", "c"),
    )
    original_hash_ranges = packed._hash_ranges
    hash_calls = 0

    def counted_hash_ranges(*args: object, **kwargs: object) -> object:
        nonlocal hash_calls
        hash_calls += 1
        return original_hash_ranges(*args, **kwargs)

    monkeypatch.setattr(packed, "_hash_ranges", counted_hash_ranges)
    with packed.PhysicalBatchPackedCheckpointReaderV1(target) as reader:
        assert hash_calls == 1
        assert reader.scalar_metadata["absolute_next_tick"] == 1120
        first = reader.load_selected_cell("a")
        second = reader.load_selected_cell("c")

    assert hash_calls == 1
    torch.testing.assert_close(
        first["fields"]["state"], torch.tensor([0, 1, 2, 3], dtype=torch.int32)
    )
    torch.testing.assert_close(
        second["fields"]["state"],
        torch.tensor([8, 9, 10, 11], dtype=torch.int32),
    )


def test_strict_reader_rejects_mutation_and_is_closed_after_failure(
    tmp_path: Path,
) -> None:
    _, target = _capture_and_publish(
        tmp_path,
        {"state": torch.arange(12, dtype=torch.int32).reshape(3, 4)},
        cell_ids=("a", "b", "c"),
    )
    reader = packed.open_physical_batch_packed_checkpoint_v1(target)
    raw = bytearray(target.read_bytes())
    raw[PREFIX.size + 1] ^= 0x01
    target.write_bytes(raw)

    with pytest.raises(PhysicalBatchPackedCheckpointError, match="changed|identity"):
        reader.load_selected_cell("b")
    assert reader.closed
    with pytest.raises(PhysicalBatchPackedCheckpointError, match="closed"):
        reader.footer

    reader.close()
    close_tmp_path = tmp_path / "close-semantics"
    close_tmp_path.mkdir()
    clean_reader = packed.PhysicalBatchPackedCheckpointReaderV1(
        _capture_and_publish(
            close_tmp_path,
            {"state": torch.ones((2, 2), dtype=torch.float32)},
            cell_ids=("x", "y"),
            batch_id="close-semantics",
        )[1]
    )
    clean_reader.close()
    with pytest.raises(PhysicalBatchPackedCheckpointError, match="closed"):
        clean_reader.load_selected_cell("x")


def test_strict_reader_selected_cell_reads_only_selected_field_ranges(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, target = _capture_and_publish(
        tmp_path,
        {
            "state": torch.arange(12, dtype=torch.int32).reshape(3, 4),
            "events": RaggedFieldV1(
                (
                    torch.tensor([10, 11], dtype=torch.int16),
                    torch.tensor([20, 21, 22], dtype=torch.int16),
                    torch.tensor([30], dtype=torch.int16),
                )
            ),
        },
        cell_ids=("a", "b", "c"),
    )
    original_load_tensor_range = packed._load_tensor_range
    loaded_ranges: list[tuple[int, int]] = []

    def recorded_load_tensor_range(stream: object, **kwargs: object) -> torch.Tensor:
        loaded_ranges.append((int(kwargs["offset"]), int(kwargs["byte_count"])))
        return original_load_tensor_range(stream, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(packed, "_load_tensor_range", recorded_load_tensor_range)
    with packed.PhysicalBatchPackedCheckpointReaderV1(target) as reader:
        result = reader.load_selected_cell("b")
        common = next(
            field for field in reader.header["fields"] if field["kind"] == "common"
        )
        ragged = next(
            field for field in reader.header["fields"] if field["kind"] == "ragged"
        )

    expected_ranges = [
        (
            int(ragged["entries"][1]["offset"]),
            int(ragged["entries"][1]["bytes"]),
        ),
        (
            int(common["offset"]) + int(common["cell_bytes"]),
            int(common["cell_bytes"]),
        ),
    ]
    assert loaded_ranges == expected_ranges
    assert result["selected_cell_id"] == "b"
    assert tuple(result["fields"]["state"].tolist()) == (4, 5, 6, 7)
    assert tuple(result["fields"]["events"].tolist()) == (20, 21, 22)
