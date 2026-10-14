from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "engine" / "perf"
WDL = ROOT / "engine" / "native_runtime" / "wdl"
HEADER = ENGINE / "cuda_architecture_b_persistent_wdl_v1.cuh"
PROBE = ROOT / "tests" / "perf" / "test_eiel_cuda_architecture_b_persistent_wdl_v1_probe.cu"


def test_contract_is_framework_free_and_keeps_wdl_state_resident() -> None:
    source = HEADER.read_text(encoding="utf-8")
    assert "struct StateView" in source
    assert "stable_event_coordinate_exact(" in source
    assert "advance_tick_exact(" in source
    assert "cue_observation_value_exact(" in source
    assert "apply_damage_to_slot_exact(" in source
    assert "cuda_architecture_b_sha256_v1.cuh" in source
    assert re.search(r"\b(?:ATen|Torch|torch|pybind|malloc|new)\b", source) is None


def test_cuda_wdl_contract_matches_cpu_oracle_for_phase_rng_observation_and_damage(
    tmp_path: Path,
) -> None:
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        pytest.skip("nvcc is unavailable")
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        pytest.skip("a CUDA device is unavailable")
    device = subprocess.run(
        [nvidia_smi, "--query-gpu=name", "--format=csv,noheader"],
        check=False,
        capture_output=True,
        text=True,
    )
    if device.returncode != 0 or not device.stdout.strip():
        pytest.skip("a CUDA device is unavailable")

    output = tmp_path / "persistent_wdl_v1_probe"
    compiled = subprocess.run(
        [
            nvcc,
            "-std=c++17",
            "-O3",
            "--fmad=false",
            "-lineinfo",
            "-gencode=arch=compute_75,code=sm_75",
            "-I",
            str(ENGINE),
            "-I",
            str(WDL),
            str(PROBE),
            str(WDL / "wdl_native_contract.cpp"),
            "-o",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert compiled.returncode == 0, compiled.stderr

    completed = subprocess.run(
        [str(output)],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == (
        "persistent_wdl_v1_probe=ok ticks=25 coordinate=exact damage=exact\n"
    )
