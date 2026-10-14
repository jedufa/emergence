from __future__ import annotations

import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
HEADER = ROOT / "engine/perf/cuda_architecture_b_compact_summary_host_settlement_v1.hpp"
SOURCE = ROOT / "engine/perf/cuda_architecture_b_compact_summary_host_settlement_v1.cpp"
PROBE = (
    ROOT
    / "tests/perf/test_eiel_cuda_architecture_b_compact_summary_host_settlement_v1.cpp"
)


@pytest.fixture(scope="module")
def probe(tmp_path_factory: pytest.TempPathFactory) -> Path:
    output = tmp_path_factory.mktemp("compact_summary_host_settlement_v1") / "probe"
    completed = subprocess.run(
        [
            "g++",
            "-std=c++17",
            "-O2",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-pedantic",
            f"-I{ROOT / 'engine/perf'}",
            str(PROBE),
            str(SOURCE),
            "-o",
            str(output),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    return output


def test_multicell_state_is_isolated_and_b1_is_preserved(probe: Path) -> None:
    completed = subprocess.run(
        [str(probe)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == (
        "b2=isolated row_order=generation_major "
        "invalid_state=closed b1=unchanged\n"
    )


def test_callback_is_native_allocation_free_and_publishes_only_complete_rows() -> None:
    header = HEADER.read_text(encoding="utf-8")
    source = SOURCE.read_text(encoding="utf-8")

    assert "settle_compact_summary_rows_callback_v1" in header
    assert "canonicalize_retained_summary_v1" in source
    assert "resolve_compact_summary_host_tokens_v1" in source
    assert "render_outer_authority_digests" in source
    assert "kHostSettlementSuccessV1" in source
    assert "authority.validity = 0" in source
    assert "authority.validity = 1" in source
    for forbidden in (
        "cudaMemcpy",
        "cudaStreamSynchronize",
        "cudaDeviceSynchronize",
        "cudaMalloc",
        "cudaFree",
        "PyObject",
        "pybind11",
        "torch::",
    ):
        assert forbidden not in source


def test_callback_consumes_generation_major_retained_sources() -> None:
    source = SOURCE.read_text(encoding="utf-8")

    assert "generation_count * context->cell_count" in source
    assert "row_index / context->cell_count" in source
    assert "row_index % context->cell_count" in source
    assert "reporting_retained_row_bytes" in source
    assert "lifecycle_digest_values + row_index * 32" in source
    assert "kBehaviorDigestOffsetV1" in source
    assert "kLineageDigestOffsetV1" in source
    assert "checkpoint_due_values[row_index]" in source
    assert "header->lineage_record_count" in source
    assert "metadata_row.generation" in source
    assert "context.cell_count > 0" in source
    assert "lineage_state_values + cell" in source
    assert "lineage_state_count != context.cell_count" in source
    assert "settlement_extent_is_safe_to_clear" in source
    assert "summary_integers[kSummaryGenerationV1]" in source


def test_patch_emit_totals_preserve_cpu_per_tick_accumulation_order() -> None:
    source = SOURCE.read_text(encoding="utf-8")

    assert (
        "floats[abi::kFloatPatchMetricsPatchScentEmitTotal] =\n"
        "                patch[kPatchScentEmitTotalV1];"
    ) in source
    assert (
        "floats[abi::kFloatPatchMetricsPatchWasteEmitTotal] =\n"
        "                patch[kPatchWasteEmitTotalV1];"
    ) in source
