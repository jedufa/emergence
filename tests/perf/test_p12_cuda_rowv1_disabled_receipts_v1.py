from __future__ import annotations

from pathlib import Path

import pytest
import torch

from engine.evolution.p12_rowv1_cuda_entry_gate_v1 import (
    project_cuda_rowv1_v3_kernel_mirror_receipt_rowv1_v1,
)
from perf.p12_cuda_rowv1_disabled_receipts_v1 import (
    DISABLED_REASON_CODES,
    ROWV1_WIDTH,
    CUDA_DISABLED_RECEIPT_CAPABILITIES,
    write_p12_cuda_rowv1_disabled_receipts_v1,
)


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="P12 CUDA RowV1 disabled receipts require CUDA",
)

SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64


@pytest.fixture(scope="module")
def cuda_build_directory(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("p12-cuda-rowv1-disabled-receipts")


def _inputs(
    *,
    cells: int,
    attempts: int,
    capacity: int,
    reason_codes: list[list[int]],
) -> dict[str, torch.Tensor]:
    assert len(reason_codes) == cells
    assert all(len(row) == attempts for row in reason_codes)
    device = torch.device("cuda")
    base = torch.arange(cells * attempts, device=device, dtype=torch.int64).reshape(
        cells,
        attempts,
    )
    return {
        "rows": torch.full(
            (cells, capacity, ROWV1_WIDTH),
            -1,
            device=device,
            dtype=torch.int64,
        ),
        "counts": torch.full((cells,), -9, device=device, dtype=torch.int64),
        "overflow": torch.full((cells,), -9, device=device, dtype=torch.int32),
        "disabled_reason_codes": torch.tensor(
            reason_codes,
            device=device,
            dtype=torch.int32,
        ),
        "parent_agent_ids": 1000 + base,
        "parent_slots": base % 17,
        "parent_slot_tokens": 2000 + base,
        "child_agent_ids": 3000 + base,
        "child_slots": (base + 5) % 19,
        "child_slot_tokens": 4000 + base,
        "generation": torch.arange(10, 10 + cells, device=device, dtype=torch.int64),
        "tick": torch.arange(20, 20 + cells, device=device, dtype=torch.int64),
    }


def _active_rows(result: dict[str, torch.Tensor]) -> list[tuple[int, tuple[int, ...]]]:
    counts = result["counts"].detach().cpu().tolist()
    rows = result["rows"].detach().cpu()
    active: list[tuple[int, tuple[int, ...]]] = []
    for cell, count in enumerate(counts):
        for sequence in range(count):
            active.append((cell, tuple(int(value) for value in rows[cell, sequence])))
    return active


def test_cuda_disabled_receipt_capabilities_are_gate2_specific() -> None:
    assert CUDA_DISABLED_RECEIPT_CAPABILITIES["decision"] == (
        "P12-CUDA-ROWV1-DISABLED-EVENT-WRITER-A"
    )
    assert CUDA_DISABLED_RECEIPT_CAPABILITIES["device_written_raw_disabled_reason_code"]
    assert CUDA_DISABLED_RECEIPT_CAPABILITIES["device_written_overflow"]
    assert CUDA_DISABLED_RECEIPT_CAPABILITIES["successor_campaign_authorized"] is False


@pytest.mark.parametrize(
    ("capacity", "reason_codes", "expected_counts", "expected_overflow"),
    (
        (0, [[1]], [0], [1]),
        (1, [[2]], [1], [0]),
        (5, [[1, 2, 3, 4]], [4], [0]),
        (6, [[1, 2, 3, 4, 5, 6]], [6], [0]),
        (5, [[1, 2, 3, 4, 5, 6]], [5], [1]),
    ),
)
def test_cuda_disabled_receipts_freeze_capacity_semantics(
    cuda_build_directory: Path,
    capacity: int,
    reason_codes: list[list[int]],
    expected_counts: list[int],
    expected_overflow: list[int],
) -> None:
    tensors = _inputs(
        cells=1,
        attempts=len(reason_codes[0]),
        capacity=capacity,
        reason_codes=reason_codes,
    )
    result = write_p12_cuda_rowv1_disabled_receipts_v1(
        **tensors,
        build_directory=cuda_build_directory,
    )

    assert result.counts.detach().cpu().tolist() == expected_counts
    assert result.overflow.detach().cpu().tolist() == expected_overflow
    if capacity > 0:
        rows = result.rows.detach().cpu()
        for sequence, reason in enumerate(reason_codes[0][: expected_counts[0]]):
            assert int(rows[0, sequence, 2]) == sequence
            assert int(rows[0, sequence, 11]) == sequence
            assert int(rows[0, sequence, 21]) == 0
            assert int(rows[0, sequence, 22]) == reason


def test_cuda_disabled_receipt_rows_project_to_canonical_rowv1(
    cuda_build_directory: Path,
) -> None:
    tensors = _inputs(
        cells=2,
        attempts=3,
        capacity=3,
        reason_codes=[[1, 2, 3], [4, 5, 6]],
    )
    result = write_p12_cuda_rowv1_disabled_receipts_v1(
        **tensors,
        build_directory=cuda_build_directory,
    )

    assert result.counts.detach().cpu().tolist() == [3, 3]
    assert result.overflow.detach().cpu().tolist() == [0, 0]
    projected_reasons = []
    for cell, row in _active_rows(
        {"rows": result.rows, "counts": result.counts, "overflow": result.overflow}
    ):
        projected = project_cuda_rowv1_v3_kernel_mirror_receipt_rowv1_v1(
            row,
            transaction_id=cell * 100 + row[2],
            cell=cell,
            rng_stream_id="p12_cuda_rowv1_disabled_receipts_v1",
            rng_index_before=row[11] * 2,
            rng_index_after=row[11] * 2,
            allocator_before={"next_agent_id": int(row[8])},
            allocator_after={"next_agent_id": int(row[8])},
            package_binding_sha256=SHA_A,
            schema_binding_sha256=SHA_B,
            mechanics_binding_sha256=SHA_C,
        )
        assert projected["canonical_status"] == "disabled"
        assert projected["changed"] is False
        assert projected["selected_identity"] is None
        projected_reasons.append(projected["raw_disabled_reason_code"])

    assert tuple(projected_reasons) == DISABLED_REASON_CODES


def test_cuda_disabled_receipts_reject_unknown_reason_by_overflow(
    cuda_build_directory: Path,
) -> None:
    tensors = _inputs(cells=1, attempts=2, capacity=2, reason_codes=[[2, 7]])
    result = write_p12_cuda_rowv1_disabled_receipts_v1(
        **tensors,
        build_directory=cuda_build_directory,
    )

    assert result.counts.detach().cpu().tolist() == [1]
    assert result.overflow.detach().cpu().tolist() == [1]

