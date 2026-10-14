from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
HEADER = (
    ROOT
    / "engine/perf/cuda_architecture_b_compact_summary_metadata_retention_v1.cuh"
)
CUDA_SOURCE = (
    ROOT
    / "engine/perf/cuda_architecture_b_compact_summary_metadata_retention_v1.cu"
)
FIXTURE = (
    ROOT
    / "tests/perf/"
    "test_eiel_cuda_architecture_b_compact_summary_metadata_retention_v1.cu"
)


def test_device_contract_is_fixed_width_and_forbids_runtime_ownership() -> None:
    header = HEADER.read_text(encoding="utf-8")
    source = CUDA_SOURCE.read_text(encoding="utf-8")
    input_pod = header.split("struct MetadataRetentionInputsV1 {", 1)[1].split(
        "};", 1
    )[0]
    output_pod = header.split(
        "struct alignas(8) CompactSummaryMetadataRetentionV1 {", 1
    )[1].split("};", 1)[0]

    assert "sizeof(CompactSummaryMetadataRetentionV1) == 3072" in header
    assert "kPythonMt19937WordsV1 = 624" in header
    assert "kSemanticStreamCounterCountV1 = 6" in header
    assert "kSourceVersionV1 = 2U" in header
    assert "kMutationTorchCpuDescriptorIndexV1 = 109" in header
    assert "kPatchP2MutationTorchCpuDescriptorIndexV1 = 121" in header
    assert "mutation_torch_cpu_descriptor_index_v1" in header
    assert "kRuntimeBookkeepingColumnCountV1 = 5" in header
    assert "policy_energy_tagged_sha256" in output_pod
    assert "*" not in output_pod
    assert "runtime_bookkeeping_by_tick" in input_pod
    assert "initial_runtime_event_cursor" in input_pod
    assert "initial_lifecycle_event_cursor" in input_pod
    assert "initial_generation_calls" in input_pod
    assert "initial_policy_py_semantic_draws" in input_pod
    assert "std::int64_t restore_count;" in input_pod
    assert "const std::int64_t* runtime_event_cursor;" not in input_pod
    assert "const std::int64_t* generation_calls;" not in input_pod
    assert "const std::int64_t* last_applied_actions;" not in input_pod
    assert "const std::int64_t* policy_py_semantic_draws;" not in input_pod
    assert '#error "cuda_architecture_b_compact_summary' not in header
    assert "#if defined(__CUDACC__)" in header
    assert "sha256::sha256_update_v1" in source
    assert "kPolicyEnergyTagPrefixV1" in source
    assert "cumulative_applied_actions" in source
    assert "last_bookkeeping[column] = value" in source
    assert "event_cursor - inputs.initial_lifecycle_event_cursor" in source
    assert "inputs.initial_generation_calls" in source
    assert "completed_generation_count" in source
    assert "policy_py_generation_calls" in source
    assert "kRetentionCounterOverflowV1" in source
    assert "row->validity = 1U" in source

    forbidden = re.compile(
        r"\b(?:ATen|Torch|torch|Python|PyObject|malloc|calloc|realloc|new|"
        r"cudaMalloc|cudaMemcpy|cudaDeviceSynchronize|cudaStreamSynchronize)\b"
    )
    assert forbidden.search(source) is None


def test_header_is_cpu_includable_and_preserves_output_layout(tmp_path: Path) -> None:
    compiler = shutil.which("g++") or shutil.which("c++")
    if compiler is None:
        pytest.skip("a C++ compiler is required")
    probe = tmp_path / "metadata_retention_cpu_include.cpp"
    probe.write_text(
        "#include \"engine/perf/"
        "cuda_architecture_b_compact_summary_metadata_retention_v1.cuh\"\n"
        "using namespace eiel::cuda_architecture_b::"
        "compact_summary_metadata_retention_v1;\n"
        "static_assert(sizeof(CompactSummaryMetadataRetentionV1) == 3072);\n"
        "static_assert(mutation_torch_cpu_descriptor_index_v1(132) == 109);\n"
        "static_assert(mutation_torch_cpu_descriptor_index_v1(144) == 121);\n"
        "static_assert(mutation_torch_cpu_descriptor_index_v1(133) == -1);\n"
        "static_assert(kRuntimeBookkeepingColumnCountV1 == 5);\n"
        "int main() { MetadataRetentionInputsV1 inputs{}; "
        "return inputs.restore_count == 0 ? 0 : 1; }\n",
        encoding="ascii",
    )
    result = subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-Wall",
            "-Wextra",
            "-Werror",
            f"-I{ROOT}",
            str(probe),
            "-c",
            "-o",
            str(tmp_path / "metadata_retention_cpu_include.o"),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_cuda_differential_exact_bytes_and_generation_isolation(
    tmp_path: Path,
) -> None:
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        pytest.skip("nvcc is required")
    executable = tmp_path / "compact_summary_metadata_retention_v1_fixture"
    compile_result = subprocess.run(
        [
            nvcc,
            "-std=c++17",
            "-arch=sm_75",
            "-Xcompiler=-Wall,-Wextra,-Werror",
            f"-I{ROOT / 'engine/perf'}",
            str(CUDA_SOURCE),
            str(FIXTURE),
            "-o",
            str(executable),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert compile_result.returncode == 0, compile_result.stderr

    run_result = subprocess.run(
        [str(executable)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if run_result.returncode == 77:
        pytest.skip(run_result.stderr.strip() or "CUDA device is required")
    assert run_result.returncode == 0, run_result.stderr
    assert "exact_rows=8" in run_result.stdout
    assert "checkpoint_continuation=true" in run_result.stdout
    assert "g0_g1_bookkeeping=true" in run_result.stdout
    assert "generation_isolation=true" in run_result.stdout
    assert "cell_isolation=true" in run_result.stdout
    assert "input_mutation=false" in run_result.stdout
    assert "failure_cases=10" in run_result.stdout
