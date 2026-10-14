from __future__ import annotations

from pathlib import Path
import shutil
import subprocess

import pytest

from engine.perf.cuda_architecture_b_flattened_birth_scratch_layout_v2 import (
    BirthScratchSegmentV2,
    make_flattened_birth_scratch_layout_v2,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE = (
    PROJECT_ROOT
    / "tests/perf/test_cuda_architecture_b_flattened_birth_scratch_v2.cpp"
)
HOST_COMPILERS = [
    compiler
    for compiler in (shutil.which("g++"), shutil.which("clang++"))
    if compiler is not None
]


def test_runtime_event_staging_covers_decisions_and_successful_birth_mutations() -> None:
    layout = make_flattened_birth_scratch_layout_v2(
        3,
        384,
        2_048,
        32,
        5_056,
    )

    assert layout.ok()
    assert layout.segment(BirthScratchSegmentV2.EVENT_STAGING_ROWS).bytes == (
        384 + 3
    ) * 104


@pytest.mark.parametrize("compiler", HOST_COMPILERS)
def test_flattened_birth_scratch_v2_host_contract(
    tmp_path: Path,
    compiler: str,
) -> None:
    executable = tmp_path / f"flattened_birth_scratch_v2_{Path(compiler).name}"
    compile_result = subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-pedantic",
            "-I",
            str(PROJECT_ROOT),
            str(SOURCE),
            "-o",
            str(executable),
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert compile_result.returncode == 0, compile_result.stderr

    run_result = subprocess.run(
        [str(executable)],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert run_result.returncode == 0, (
        f"host contract failed with {run_result.returncode}: "
        f"{run_result.stdout}{run_result.stderr}"
    )


def test_flattened_birth_scratch_v2_nvcc_compile_only(tmp_path: Path) -> None:
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        pytest.skip("NVCC is required for the CUDA compile-only contract")

    object_file = tmp_path / "flattened_birth_scratch_v2.o"
    compile_result = subprocess.run(
        [
            nvcc,
            "-allow-unsupported-compiler",
            "-std=c++17",
            "-x",
            "cu",
            "-I",
            str(PROJECT_ROOT),
            "-c",
            str(SOURCE),
            "-o",
            str(object_file),
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert compile_result.returncode == 0, compile_result.stderr
    assert object_file.is_file()
