from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HEADER = (
    ROOT
    / "engine/perf/cuda_architecture_b_compact_summary_settlement_v1.hpp"
)


def test_settlement_sequence_has_one_callback_and_no_sync_or_allocation() -> None:
    source = HEADER.read_text(encoding="utf-8")

    d2h = source.index("cudaMemcpyDeviceToHost")
    callback = source.index("cudaLaunchHostFunc")
    h2d = source.index("cudaMemcpyHostToDevice")
    projection = source.index("launch_compact_summary_source_projection_v1")
    assert d2h < callback < h2d < projection
    assert source.count("cudaLaunchHostFunc(") == 1
    assert "cudaStreamSynchronize" not in source
    assert "cudaDeviceSynchronize" not in source
    assert "cudaMalloc" not in source
    assert "cudaFree" not in source


def test_settlement_receipt_records_bounded_operation_counts() -> None:
    source = HEADER.read_text(encoding="utf-8")

    assert "++receipt.d2h_operations" in source
    assert "receipt.host_callback_launches = 1" in source
    assert "receipt.h2d_operations = 1" in source
    assert "receipt.projection_launches = 1" in source
    assert "std::int32_t synchronizations" in source
    assert "std::int32_t allocations" in source
    assert "kSettlementProjectionLaunchFailedV1" in source
