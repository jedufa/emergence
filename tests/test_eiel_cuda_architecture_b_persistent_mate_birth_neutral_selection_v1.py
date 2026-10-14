from __future__ import annotations

from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
CPU_FIXTURE = (
    ROOT
    / "tests/perf/test_cuda_architecture_b_persistent_mate_birth_neutral_selection_v1.cpp"
)
CUDA_FIXTURE = (
    ROOT
    / "tests/perf/test_cuda_architecture_b_persistent_mate_birth_neutral_selection_v1_compile.cu"
)
MATE_HEADER = (
    ROOT / "engine/perf/cuda_architecture_b_persistent_mate_birth_plan_v1.cuh"
)
CONTROLLER = (
    ROOT / "engine/perf/cuda_architecture_b_flattened_controller_v1_kernels.cu"
)


def test_neutral_mate_selection_host_contract(tmp_path: Path) -> None:
    compiler = shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        pytest.skip("a C++ compiler is required")
    executable = tmp_path / "neutral_mate_selection"
    result = subprocess.run(
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
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    executed = subprocess.run(
        [str(executable)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert executed.returncode == 0, executed.stdout + executed.stderr


def test_neutral_mate_selection_cuda_contract(tmp_path: Path) -> None:
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        pytest.skip("nvcc unavailable")
    result = subprocess.run(
        [
            nvcc,
            "-std=c++17",
            "-arch=sm_75",
            "-Xcompiler=-Wall,-Wextra,-Werror",
            "-c",
            str(CUDA_FIXTURE),
            "-o",
            str(tmp_path / "neutral_mate_selection.o"),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_neutral_selection_and_fixed_action_source_contracts() -> None:
    mate_source = MATE_HEADER.read_text(encoding="ascii")
    controller_source = CONTROLLER.read_text(encoding="ascii")

    assert "neutral_parent_rank_v1(selection, agent_id)" in mate_source
    assert "selection.cell_seed" in mate_source
    assert "selection.generation" in mate_source
    assert "selection.absolute_tick" in mate_source
    assert "selection.trait_authority_enabled" in mate_source
    assert "selection.neutral_repro_threshold" in mate_source
    assert "genotype" not in mate_source

    assert "const bool neutral_fixed_action =" in controller_source
    assert "if (deterministic_actions || neutral_fixed_action)" in controller_source
    assert controller_source.count(
        "!deterministic_actions && !neutral_fixed_action"
    ) >= 2
    stochastic_service = controller_source.split(
        "#define EIEL_PATCH_POLICY_ARGUMENTS", 1
    )[0]
    assert (
        "!deterministic_actions\n"
        "            && !neutral_fixed_action" in stochastic_service
    )
