from __future__ import annotations

from pathlib import Path

import pytest
import torch

from perf.cuda_architecture_b_event_tape import (
    CUDA_EVENT_TAPE_CAPABILITIES,
    EVENT_TAPE_STATUS_CAPACITY_EXCEEDED,
    EVENT_TAPE_STATUS_OK,
    append_event_rows_cuda,
)


def _empty_tape(capacity: int) -> list[torch.Tensor]:
    return [
        torch.full((capacity,), -1, device="cuda", dtype=torch.int32),
        torch.full((capacity,), -1, device="cuda", dtype=torch.int64),
        torch.full((capacity,), -1.0, device="cuda", dtype=torch.float32),
    ]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_event_tape_appends_complete_mixed_dtype_rows_without_host_cursor(tmp_path: Path) -> None:
    tape = _empty_tape(8)
    active_count = torch.zeros(1, device="cuda", dtype=torch.int64)
    status = torch.full((1,), -1, device="cuda", dtype=torch.int32)
    first_rows = [
        torch.tensor([4, 1, 3], device="cuda", dtype=torch.int32),
        torch.tensor([100, 101, 102], device="cuda", dtype=torch.int64),
        torch.tensor([1.0, 30.0, 4.0], device="cuda", dtype=torch.float32),
    ]

    first = append_event_rows_cuda(
        tape_columns=tape,
        event_rows=first_rows,
        active_count=active_count,
        status=status,
        build_directory=tmp_path / "build",
    )
    second_rows = [column[:2].clone() for column in first_rows]
    second = append_event_rows_cuda(
        tape_columns=tape,
        event_rows=second_rows,
        active_count=active_count,
        status=status,
        build_directory=tmp_path / "build",
    )

    assert first.active_count.data_ptr() == active_count.data_ptr()
    assert second.status.data_ptr() == status.data_ptr()
    assert int(second.status.item()) == EVENT_TAPE_STATUS_OK
    assert int(second.active_count.item()) == 5
    assert tape[0][:5].tolist() == [4, 1, 3, 4, 1]
    assert tape[1][:5].tolist() == [100, 101, 102, 100, 101]
    assert tape[2][:5].tolist() == [1.0, 30.0, 4.0, 1.0, 30.0]
    assert tape[0][5:].tolist() == [-1, -1, -1]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_event_tape_capacity_failure_is_atomic(tmp_path: Path) -> None:
    tape = _empty_tape(4)
    active_count = torch.tensor([3], device="cuda", dtype=torch.int64)
    status = torch.full((1,), -1, device="cuda", dtype=torch.int32)
    before = [column.clone() for column in tape]
    rows = [
        torch.tensor([4, 1], device="cuda", dtype=torch.int32),
        torch.tensor([100, 101], device="cuda", dtype=torch.int64),
        torch.tensor([1.0, 30.0], device="cuda", dtype=torch.float32),
    ]

    result = append_event_rows_cuda(
        tape_columns=tape,
        event_rows=rows,
        active_count=active_count,
        status=status,
        build_directory=tmp_path / "build",
    )

    assert int(result.status.item()) == EVENT_TAPE_STATUS_CAPACITY_EXCEEDED
    assert int(result.active_count.item()) == 3
    assert all(torch.equal(actual, expected) for actual, expected in zip(tape, before, strict=True))


def test_event_tape_capability_does_not_claim_semantic_construction() -> None:
    assert CUDA_EVENT_TAPE_CAPABILITIES["legacy_and_typed_schema_storage"] is True
    assert CUDA_EVENT_TAPE_CAPABILITIES["semantic_event_construction"] is False
    assert CUDA_EVENT_TAPE_CAPABILITIES["promotion_qualified"] is False
