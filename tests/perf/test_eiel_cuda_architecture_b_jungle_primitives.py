from __future__ import annotations

from pathlib import Path
import shutil
import subprocess

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENGINE_PERF = PROJECT_ROOT / "engine" / "perf"
HOST_PROBE = Path(__file__).with_name(
    "test_cuda_architecture_b_jungle_primitives.cpp"
)
CUDA_PROBE = Path(__file__).with_name(
    "test_cuda_architecture_b_jungle_primitives_compile.cu"
)


def test_jungle_primitives_match_resident_cpu_formulas(tmp_path: Path) -> None:
    compiler = shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        pytest.skip("a C++17 host compiler is required")

    executable = tmp_path / "jungle_primitives_host_probe"
    compiled = subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-pedantic",
            "-I",
            str(ENGINE_PERF),
            str(HOST_PROBE),
            "-o",
            str(executable),
        ],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert compiled.returncode == 0, compiled.stderr

    executed = subprocess.run(
        [str(executable)],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert executed.returncode == 0, f"{executed.stdout}{executed.stderr}"
    assert executed.stdout.strip() == "jungle primitive host probe passed"


def test_jungle_primitives_compile_for_cuda_device_calls(tmp_path: Path) -> None:
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        pytest.skip("NVCC is required for the CUDA compile contract")

    object_file = tmp_path / "jungle_primitives_compile.o"
    compiled = subprocess.run(
        [
            nvcc,
            "-allow-unsupported-compiler",
            "-std=c++17",
            "--Werror",
            "all-warnings",
            "-I",
            str(ENGINE_PERF),
            "-c",
            str(CUDA_PROBE),
            "-o",
            str(object_file),
        ],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert compiled.returncode == 0, compiled.stderr
    assert object_file.is_file()
