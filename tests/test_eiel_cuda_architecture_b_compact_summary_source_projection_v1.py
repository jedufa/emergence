from __future__ import annotations

import importlib.util
import re
import shutil
import subprocess
from pathlib import Path
from types import ModuleType

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
HEADER = (
    ROOT
    / "engine/perf/"
    "cuda_architecture_b_compact_summary_source_projection_v1.cuh"
)
CUDA_SOURCE = (
    ROOT
    / "engine/perf/"
    "cuda_architecture_b_compact_summary_source_projection_v1.cu"
)
COMPILE_TEST = (
    ROOT
    / "tests/perf/"
    "eiel_cuda_architecture_b_compact_summary_source_projection_v1_compile.cu"
)
PROBE = (
    ROOT
    / "tools/scripts/"
    "eiel_cuda_architecture_b_compact_summary_source_projection_v1_probe.py"
)


def _load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_authority_is_complete_fixed_pod_and_component_is_standalone() -> None:
    header = HEADER.read_text(encoding="utf-8")
    cuda = CUDA_SOURCE.read_text(encoding="utf-8")
    assert "struct CompactSummaryGenerationCellAuthorityV1" in header
    assert "std::is_standard_layout_v<CompactSummaryGenerationCellAuthorityV1>" in header
    assert "std::is_trivially_copyable_v<CompactSummaryGenerationCellAuthorityV1>" in header
    assert "kAuthoritySourcePlaneLogicalBytesV1" in header
    for width in ("== 12", "== 76", "== 21", "== 320", "== 65"):
        assert width in header
    assert "*abi::row_pointer(output.validity" in cuda
    assert cuda.index("= source.validity;") < cuda.rindex(
        "producer_status[row] = projection::kProjectionSuccess;"
    )
    assert "kProjectionFailClosed" in cuda
    forbidden = re.compile(
        r"\b(?:ATen|Torch|torch|cudaMalloc|cudaFree|cudaDeviceSynchronize|malloc|calloc|new)\b"
    )
    assert forbidden.search(header) is None
    assert forbidden.search(cuda) is None
    assert "flattened_controller" not in header
    assert "flattened_controller" not in cuda


def test_nvcc_layout_and_projection_kernel_have_bounded_sm75_receipt(
    tmp_path: Path,
) -> None:
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        pytest.skip("nvcc is unavailable")

    compile_output = tmp_path / "source_projection_layout.o"
    compile_result = subprocess.run(
        [
            nvcc,
            "-std=c++17",
            "-O3",
            "-gencode=arch=compute_75,code=sm_75",
            "-c",
            str(COMPILE_TEST),
            "-o",
            str(compile_output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert compile_result.returncode == 0, compile_result.stderr
    assert compile_output.is_file()

    kernel_output = tmp_path / "source_projection_kernel.o"
    kernel_result = subprocess.run(
        [
            nvcc,
            "-std=c++17",
            "-O3",
            "-lineinfo",
            "-Xptxas=-v",
            "-gencode=arch=compute_75,code=sm_75",
            "-I",
            str(ROOT / "engine/perf"),
            "-c",
            str(CUDA_SOURCE),
            "-o",
            str(kernel_output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert kernel_result.returncode == 0, kernel_result.stderr
    assert "0 bytes stack frame" in kernel_result.stderr
    assert "0 bytes spill stores, 0 bytes spill loads" in kernel_result.stderr

    cuobjdump = shutil.which("cuobjdump")
    if cuobjdump is not None:
        sass = subprocess.run(
            [cuobjdump, "--dump-sass", str(kernel_output)],
            check=False,
            capture_output=True,
            text=True,
        )
        assert sass.returncode == 0, sass.stderr
        assert "compact_summary_source_projection_v1_kernel" in sass.stdout
        assert "STG" in sass.stdout


@pytest.fixture(scope="module")
def cuda_probe(tmp_path_factory: pytest.TempPathFactory) -> tuple[ModuleType, ModuleType]:
    if shutil.which("nvcc") is None or not torch.cuda.is_available():
        pytest.skip("CUDA compiler and device are required")
    probe = _load_module(PROBE, "eiel_compact_source_projection_probe_test")
    module = probe.build_extension(
        tmp_path_factory.mktemp("compact_summary_source_projection_v1")
    )
    return probe, module


def test_generation_major_g2_b3_planes_are_byte_exact(
    cuda_probe: tuple[ModuleType, ModuleType],
) -> None:
    probe, module = cuda_probe
    receipt = probe.run_probe(module)

    assert receipt["all_exact"] is True
    assert receipt["shape"] == [2, 3]
    assert receipt["authority_byte_width"] == 1256
    assert receipt["source_plane_logical_bytes"] == 1218
    assert all(receipt["plane_byte_exact"].values())
    assert receipt["producer_status"] == [[0, 0, 0], [11, 2, 3]]
    assert receipt["producer_status_exact"] is True
    assert receipt["null_defined_mask_exact"] is True
    assert receipt["negative_zero_preserved"] is True
    assert receipt["invalid_rows_bit_zero"] is True


def test_malformed_and_noop_rows_fail_closed_without_invented_defaults(
    cuda_probe: tuple[ModuleType, ModuleType],
) -> None:
    probe, module = cuda_probe
    validations = probe.run_probe(module)["validations"]

    assert set(validations) == {
        "failed_generation",
        "invalid_index",
        "nonfinite_float",
        "undefined_storage",
    }
    for case in validations.values():
        assert case["status_exact"] is True
        assert case["bit_zero"] is True
        assert case["producer_status"] != 0


def test_runtime_resource_receipt_has_no_local_or_shared_memory(
    cuda_probe: tuple[ModuleType, ModuleType],
) -> None:
    _, module = cuda_probe
    receipt = dict(module.resource_receipt())

    assert receipt["binary_version"] == 75
    assert receipt["registers_per_thread"] > 0
    assert receipt["local_bytes_per_thread"] == 0
    assert receipt["shared_bytes_per_block"] == 0
