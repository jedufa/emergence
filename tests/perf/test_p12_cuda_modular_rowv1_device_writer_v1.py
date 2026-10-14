from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "tests/perf/test_p12_cuda_modular_rowv1_device_writer_v1.cu"


def _tool(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        pytest.skip(f"{name} is not available")
    return path


def test_modular_rowv1_device_writer_probe_passes_nvcc_and_memcheck(
    tmp_path: Path,
) -> None:
    nvcc = _tool("nvcc")
    sanitizer = _tool("compute-sanitizer")
    binary = tmp_path / "test_p12_cuda_modular_rowv1_device_writer_v1"

    subprocess.run(
        [
            nvcc,
            "-std=c++17",
            "-O2",
            "-I",
            str(ROOT / "engine/perf"),
            str(SOURCE),
            "-o",
            str(binary),
        ],
        cwd=ROOT,
        check=True,
    )
    subprocess.run([str(binary)], cwd=ROOT, check=True)
    completed = subprocess.run(
        [sanitizer, "--tool", "memcheck", str(binary)],
        cwd=ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    assert "ERROR SUMMARY: 0 errors" in completed.stdout
