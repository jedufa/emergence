from __future__ import annotations

import importlib.util
import re
import shutil
import subprocess
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
import torch

from engine.native_runtime import compact_summary_cpu as cpu
from engine.native_runtime import compact_summary_planes as abi
from engine.native_runtime import compact_summary_projection as projection


ROOT = Path(__file__).resolve().parents[1]
HEADER = (
    ROOT / "engine/perf/cuda_architecture_b_compact_summary_assembler_v1.cuh"
)
CUDA_SOURCE = (
    ROOT / "engine/perf/cuda_architecture_b_compact_summary_assembler_v1.cu"
)
COMPILE_TEST = (
    ROOT
    / "tests/perf/"
    "test_eiel_cuda_architecture_b_compact_summary_assembler_v1_compile.cu"
)
PROBE = (
    ROOT
    / "tools/scripts/"
    "eiel_cuda_architecture_b_compact_summary_assembler_v1_probe.py"
)
CPU_FIXTURES = ROOT / "tests/test_eiel_compact_summary_cpu.py"


def _load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_source_and_scratch_are_device_neutral_pods_without_hotpath_frameworks() -> None:
    header = HEADER.read_text(encoding="utf-8")
    cuda = CUDA_SOURCE.read_text(encoding="utf-8")
    assert "struct CompactSummaryBoundarySourceV1" in header
    assert "struct CompactSummaryScratchRowV1" in header
    assert "std::is_standard_layout_v<CompactSummaryBoundarySourceV1>" in header
    assert "std::is_trivially_copyable_v<CompactSummaryBoundarySourceV1>" in header
    assert "*scratch.validity = 1;" in header
    assert header.index("*scratch.validity = 1;") < header.index("*scratch.ready = 1;")
    forbidden = re.compile(
        r"\b(?:ATen|Torch|torch|cudaMalloc|cudaFree|cudaDeviceSynchronize|malloc|calloc|new)\b"
    )
    assert forbidden.search(header) is None
    assert forbidden.search(cuda) is None


def test_nvcc_compile_is_sm75_zero_stack_and_zero_spill(tmp_path: Path) -> None:
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        pytest.skip("nvcc is unavailable")
    output = tmp_path / "compact_summary_assembler_v1.o"
    completed = subprocess.run(
        [
            nvcc,
            "-std=c++17",
            "-O3",
            "-lineinfo",
            "-Xptxas=-v",
            "-gencode=arch=compute_75,code=sm_75",
            "-c",
            str(COMPILE_TEST),
            "-o",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert output.is_file()
    assert "0 bytes stack frame" in completed.stderr
    assert "0 bytes spill stores, 0 bytes spill loads" in completed.stderr


@pytest.fixture(scope="module")
def cuda_probe(tmp_path_factory: pytest.TempPathFactory) -> tuple[ModuleType, ModuleType]:
    if shutil.which("nvcc") is None or not torch.cuda.is_available():
        pytest.skip("CUDA compiler and device are required")
    probe = _load_module(PROBE, "eiel_compact_summary_assembler_probe_test")
    module = probe.build_extension(
        tmp_path_factory.mktemp("compact_summary_assembler_v1")
    )
    return probe, module


def test_real_flat_patch_signed_zero_and_utf8_rows_are_exact(
    cuda_probe: tuple[ModuleType, ModuleType],
) -> None:
    probe, module = cuda_probe
    receipt = probe.run_probe(module)

    assert receipt["all_exact"] is True
    assert receipt["logical_bytes_per_generation_cell"] == 1218
    for name in ("flat", "patch", "signed_zero", "strict_utf8"):
        assert receipt["cases"][name]["exact_planes"] is True
        assert receipt["cases"][name]["exact_row"] is True
        assert receipt["cases"][name]["canonical_bytes_exact"] is True
        assert receipt["cases"][name]["assemble_status"] == 0
        assert receipt["cases"][name]["scratch_ready"] == 1
    assert receipt["cases"]["signed_zero"]["negative_zero_preserved"] is True
    assert receipt["cases"]["invalid_status"] == {
        "assemble_status": 5,
        "scratch_ready": 0,
        "bit_zero_noop": True,
    }


def test_invalid_defined_storage_is_rejected_atomically(
    cuda_probe: tuple[ModuleType, ModuleType],
) -> None:
    probe, module = cuda_probe
    fixtures = _load_module(CPU_FIXTURES, "eiel_compact_summary_cpu_fixtures")
    row = cpu.build_compact_summary_row(fixtures._source(world="flat"))
    source = abi.allocate_compact_summary_planes(1, 1)
    projection.encode_compact_summary_row(row, source, generation=0, cell=0)
    source.integer_values[
        0, 0, abi.INTEGER_FIELD_INDEX["terminal.failure_generation"]
    ] = 7
    source.validity.fill(0)

    actual, ready, status = probe.assemble_host_planes(
        module, source, prefill_scratch=True
    )

    assert status.tolist() == [10]
    assert ready.tolist() == [0]
    assert all(
        not np.any(array.view(np.uint8))
        for array in actual.as_mapping().values()
    )


def test_invalid_utf8_is_rejected_atomically(
    cuda_probe: tuple[ModuleType, ModuleType],
) -> None:
    probe, module = cuda_probe
    fixtures = _load_module(CPU_FIXTURES, "eiel_compact_summary_cpu_utf8_fixtures")
    row = cpu.build_compact_summary_row(fixtures._source(world="flat"))
    source = abi.allocate_compact_summary_planes(1, 1)
    projection.encode_compact_summary_row(row, source, generation=0, cell=0)
    record = source.text_values[0, 0, 0]
    record.fill(0)
    record[0] = 2
    record[1:3] = [0xC0, 0xAF]
    source.validity.fill(0)

    actual, ready, status = probe.assemble_host_planes(
        module, source, prefill_scratch=True
    )

    assert status.tolist() == [14]
    assert ready.tolist() == [0]
    assert all(
        not np.any(array.view(np.uint8))
        for array in actual.as_mapping().values()
    )


def test_runtime_resource_receipt_has_no_local_memory(
    cuda_probe: tuple[ModuleType, ModuleType],
) -> None:
    _, module = cuda_probe
    receipt = dict(module.resource_receipt())

    assert receipt["binary_version"] == 75
    assert receipt["registers_per_thread"] > 0
    assert receipt["local_bytes_per_thread"] == 0
    assert receipt["shared_bytes_per_block"] == 0
