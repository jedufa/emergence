from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
HEADER = (
    ROOT
    / "engine/perf/cuda_architecture_b_persistent_mate_birth_plan_v1.cuh"
)
CPU_FIXTURE = (
    ROOT
    / "tests/perf/test_cuda_architecture_b_persistent_mate_birth_plan_v1.cpp"
)
CUDA_FIXTURE = (
    ROOT
    / "tests/perf/test_cuda_architecture_b_persistent_mate_birth_plan_v1_compile.cu"
)


def test_persistent_mate_birth_plan_v1_cpu_cases(tmp_path: Path) -> None:
    compiler = shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        pytest.skip("a C++ compiler is required")
    executable = tmp_path / "persistent_mate_birth_plan_v1"
    compile_result = subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-pedantic",
            str(CPU_FIXTURE),
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
    assert run_result.returncode == 0, (
        f"fixture failed with {run_result.returncode}: "
        f"{run_result.stdout}{run_result.stderr}"
    )


def test_persistent_mate_birth_plan_v1_source_contract() -> None:
    source = HEADER.read_text(encoding="ascii")

    assert "config.max_births > config.capacity" in source
    assert "int64_t adjacent_ids[kAdjacentDirectionCount]" in source
    assert "int32_t candidate_y[kMateSpawnCandidateCount]" in source
    assert "kMateSpawnCandidateCount = 8" in source
    assert "buffers.planned_mate_slots[planned_count]" in source
    assert "config.requires_mate ? mate_slot : -1" in source
    assert "buffers.planned_energy[row] = state.energy[row]" in source
    assert "buffers.planned_energy[parent_slot]" in source
    assert "buffers.planned_energy[mate_slot]" in source
    assert "state.policy_np_semantic_draws[0] += env_semantic_draws" in source
    assert "state.metrics_np_semantic_draws[0] += runner_semantic_draws" in source

    function = source.split("void plan_births_serial_v1(", 1)[1]
    first_authoritative_write = function.index(
        "state.repro_cooldown[slot] ="
    )
    final_overflow_gate = function.index(
        "detail::counter_add_overflows(\n"
        "            state.metrics_np_semantic_draws[0]"
    )
    assert first_authoritative_write > final_overflow_gate

    parent_loop = function.split(
        "const int64_t decision = decision_count++;", 1
    )[1].split("int64_t mate_slot = -1;", 1)[0]
    cooldown_decrement = parent_loop.index(
        "--buffers.planned_repro_cooldown[parent_slot]"
    )
    for later_check in (
        "detail::slot_is_mated(",
        "buffers.planned_energy[parent_slot]",
        "config.injury_gate_active",
        "config.scar_gate_active",
        "config.chemistry_viability_active",
    ):
        assert cooldown_decrement < parent_loop.index(later_check)

    forbidden = re.compile(
        r"(?:\b(?:malloc|calloc|realloc|free|new|delete)\s*\(|std::vector|"
        r"cudaMalloc|cudaMemcpy|cudaDeviceSynchronize|"
        r"cudaStreamSynchronize|ATen|torch)\b"
    )
    assert forbidden.search(source) is None


def test_persistent_mate_birth_plan_v1_device_compile(tmp_path: Path) -> None:
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        pytest.skip("nvcc is unavailable")
    result = subprocess.run(
        [
            nvcc,
            "-std=c++17",
            "-arch=sm_75",
            "-Xcompiler=-Wall,-Wextra,-Werror",
            "-c",
            str(CUDA_FIXTURE),
            "-o",
            str(tmp_path / "persistent_mate_birth_plan_v1.o"),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
