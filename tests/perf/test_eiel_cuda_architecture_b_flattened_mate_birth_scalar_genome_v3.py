from __future__ import annotations

import math
from pathlib import Path
import random
import shutil
import struct
import subprocess
from typing import NamedTuple

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
HEADER = (
    PROJECT_ROOT
    / "engine/perf/cuda_architecture_b_flattened_mate_birth_scalar_genome_v3.cuh"
)
CPU_SOURCE = (
    PROJECT_ROOT
    / "tests/perf/test_cuda_architecture_b_flattened_mate_birth_scalar_genome_v3.cpp"
)
CUDA_SOURCE = (
    PROJECT_ROOT
    / "tests/perf/test_cuda_architecture_b_flattened_mate_birth_scalar_genome_v3_compile.cu"
)


def _f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def _f32_formula(base: float, step: float, index: int) -> float:
    return _f32(_f32(base) + _f32(_f32(step) * _f32(float(index))))


def _float_bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", _f32(value)))[0]


def _canon(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return round(_f32(value) * 100_000_000.0) / 100_000_000.0


def _canon_f32(value: float) -> float:
    return _f32(_canon(value))


def _clamp(value: float, lower: float, upper: float) -> float:
    return lower if value < lower else upper if value > upper else value


class _TrackedCpythonRandom:
    def __init__(self, rng: random.Random) -> None:
        self.rng = rng
        self.raw_draws = 0
        self.tape: dict[int, tuple[float, float]] = {}

    def random(self) -> float:
        self.raw_draws += 2
        return self.rng.random()

    def gauss(self, sigma: float) -> float:
        had_cached = self.rng.getstate()[2] is not None
        tape_index = self.raw_draws
        value = self.rng.gauss(0.0, sigma)
        if not had_cached:
            self.raw_draws += 4
            cached = self.rng.getstate()[2]
            assert cached is not None
            self.tape[tape_index] = (value / sigma, float(cached))
        return value

    def randbelow_two(self) -> int:
        while True:
            self.raw_draws += 1
            value = self.rng.getrandbits(2)
            if value < 2:
                return value


def _slot_float(field: int, slot: int) -> float:
    definitions = (
        (-0.8, 0.08, 17),
        (-1.5, 0.12, 19),
        (0.02, 0.01, 7),
        (10.0, 2.0, 13),
        (0.4, 0.15, 9),
        (0.3, 0.05, 10),
        (0.005, 0.01, 8),
        (0.2, 0.2, 11),
        (0.02, 0.07, 10),
        (0.1, 0.1, 8),
        (0.02, 0.08, 9),
        (0.03, 0.07, 10),
        (0.04, 0.06, 11),
        (0.1, 0.09, 9),
        (0.01, 0.04, 8),
        (0.01, 0.02, 7),
    )
    base, step, modulus = definitions[field]
    return _f32_formula(base, step, slot % modulus)


def _midpoint(field: int, parent: int, mate: int) -> float:
    return 0.5 * (
        _canon(_slot_float(field, parent)) + _canon(_slot_float(field, mate))
    )


def _mutate(
    rng: _TrackedCpythonRandom,
    value: float,
    rate: float,
    sigma: float,
    lower: float,
    upper: float,
) -> tuple[float, int]:
    draws = 1
    if rng.random() < rate:
        value += rng.gauss(sigma)
        draws += 1
    return _canon_f32(_clamp(value, lower, upper)), draws


class _ReferenceResult(NamedTuple):
    rows: list[list[int]]
    words: list[int]
    index: int
    has_gauss: int
    gauss_next: float
    raw_draws: int
    semantic_added: int
    tape: dict[int, tuple[float, float]]


def _reference(birth_count: int, seed: int) -> tuple[tuple, _ReferenceResult]:
    source_rng = random.Random(seed)
    initial_state = source_rng.getstate()
    rng = _TrackedCpythonRandom(source_rng)
    rows: list[list[int]] = []
    semantic = 0

    for birth in range(birth_count):
        parent = birth
        mate = parent if birth % 2 == 0 else 64 + birth
        hidden = (
            32 + 16 * (parent % 5)
            if rng.random() < 0.5
            else 32 + 16 * (mate % 5)
        )
        semantic += 1
        base_rate = _midpoint(2, parent, mate)
        rate = _clamp(base_rate, 0.0, 1.0)
        mutation_specs = (
            (0, 0.10, -1.0, 1.0),
            (1, 0.15, -2.0, 2.0),
            (3, 2.0, 2.0, 90.0),
            (4, 0.10, 0.25, 2.5),
            (5, 0.05, 0.0, 0.999),
            (6, 0.01, 0.0, 0.10),
            (7, 0.20, 0.0, 3.0),
            (8, 0.03, 0.0, 1.0),
        )
        mutated: list[float] = []
        for field, sigma, lower, upper in mutation_specs:
            value, draws = _mutate(
                rng, _midpoint(field, parent, mate), rate, sigma, lower, upper
            )
            mutated.append(value)
            semantic += draws

        child_mutation_rate_raw = base_rate
        semantic += 1
        if rng.random() < 0.2:
            child_mutation_rate_raw += rng.gauss(0.005)
            semantic += 1
        child_mutation_rate_raw = _clamp(child_mutation_rate_raw, 0.001, 0.20)
        child_mutation_rate = _canon_f32(child_mutation_rate_raw)

        semantic += 1
        memory_probability = _clamp(
            0.25 * child_mutation_rate_raw, 0.002, 0.05
        )
        if rng.random() < memory_probability:
            rng.randbelow_two()
            semantic += 1

        integer_fields = [
            hidden,
            0,
            ((2 + 13 * parent) & 1023) | 1,
            1 + parent % 3,
            (40000 + parent) & 32767,
            (40 + parent) & 31,
            (70 + parent) & 31,
            (90 + parent) & 63,
            0,
            0,
        ]
        midpoint_fields = [
            _canon_f32(_clamp(_midpoint(field, parent, mate), lower, upper))
            for field, lower, upper in (
                (9, 0.0, 1.0),
                (10, 0.0, 1.0),
                (11, 0.0, 1.0),
                (12, 0.0, 1.0),
                (13, 0.0, 1.0),
                (14, 0.0, 0.5),
                (15, 0.0, 0.2),
            )
        ]
        exploration, aggression, repro, temperature, inertia, noise, gain, pred = (
            mutated
        )
        float_fields = [
            exploration,
            aggression,
            child_mutation_rate,
            repro,
            temperature,
            inertia,
            noise,
            gain,
            pred,
            *midpoint_fields,
            repro,
            0.0,
            aggression,
            temperature,
            gain,
            inertia,
            noise,
            pred,
        ]
        rows.append(integer_fields + [_float_bits(value) for value in float_fields])

    final_state = source_rng.getstate()
    words_and_index = final_state[1]
    gauss_next = final_state[2]
    return initial_state, _ReferenceResult(
        rows=rows,
        words=[int(word) for word in words_and_index[:-1]],
        index=int(words_and_index[-1]),
        has_gauss=int(gauss_next is not None),
        gauss_next=0.0 if gauss_next is None else float(gauss_next),
        raw_draws=rng.raw_draws,
        semantic_added=semantic,
        tape=rng.tape,
    )


@pytest.fixture(scope="module")
def cpu_executable(tmp_path_factory: pytest.TempPathFactory) -> Path:
    compiler = shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        pytest.skip("a C++17 host compiler is required")
    build_dir = tmp_path_factory.mktemp("flattened_mate_scalar_v3")
    cuda_stub = build_dir / "cuda_runtime.h"
    cuda_stub.write_text("#pragma once\n", encoding="ascii")
    executable = build_dir / "flattened_mate_scalar_v3"
    completed = subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-pedantic",
            "-D__device__=",
            "-D__forceinline__=inline",
            "-I",
            str(build_dir),
            "-I",
            str(PROJECT_ROOT),
            str(CPU_SOURCE),
            "-o",
            str(executable),
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    return executable


@pytest.mark.parametrize("birth_count", [0, 1, 2, 3, 64])
def test_cpu_executable_matches_cpython_recombination_and_rng_order(
    cpu_executable: Path,
    birth_count: int,
) -> None:
    initial, expected = _reference(birth_count, seed=0xB17A + birth_count)
    initial_words_and_index = initial[1]
    input_lines = [
        f"{birth_count} {initial_words_and_index[-1]} 0 0 0 17",
        " ".join(str(word) for word in initial_words_and_index[:-1]),
        str(len(expected.tape)),
        *(
            f"{index} {primary:.17g} {secondary:.17g}"
            for index, (primary, secondary) in sorted(expected.tape.items())
        ),
    ]
    completed = subprocess.run(
        [str(cpu_executable)],
        cwd=PROJECT_ROOT,
        input="\n".join(input_lines) + "\n",
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    lines = completed.stdout.splitlines()
    metadata = lines[0].split()
    assert [int(value) for value in metadata[:5]] == [
        0,
        birth_count,
        expected.semantic_added,
        expected.index,
        expected.has_gauss,
    ]
    assert float(metadata[5]) == expected.gauss_next
    assert int(metadata[6]) == expected.raw_draws
    assert int(metadata[7]) == 17 + expected.semantic_added
    actual_rows = [
        [int(value) for value in line.split()]
        for line in lines[1 : 1 + birth_count]
    ]
    assert actual_rows == expected.rows
    actual_words = [int(value) for value in lines[1 + birth_count].split()]
    assert actual_words == expected.words


def test_source_contract_is_external_scratch_and_preselected_pair_only() -> None:
    source = HEADER.read_text(encoding="utf-8")
    assert "const int64_t* parent_slots;" in source
    assert "const int64_t* mate_slots;" in source
    assert "const int64_t* child_slots;" in source
    assert "mate_slot >= 0 ? mate_slot : parent_slot" in source
    assert "turnover_scalar_birth_rng_exceeds_gauss_tape" in source
    assert "cuda_architecture_b_flattened_birth_scalar_genome_v2.cuh" in source
    assert "cuda_architecture_b_persistent_turnover_scalar_genome.cuh" in source
    assert "parent_pool_slots" not in source
    assert "cudaMalloc" not in source
    assert "new " not in source
    assert "malloc(" not in source
    assert "std::vector" not in source
    assert "scratch.genome_memory_dim[birth] = 0;" in source
    assert "scratch.policy_mem_dim[birth] = 0;" in source
    assert "scratch.policy_use_gru[birth] = false;" in source


def test_nvcc_compile_only_contract(tmp_path: Path) -> None:
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        pytest.skip("NVCC is required for the CUDA compile-only contract")
    object_file = tmp_path / "flattened_mate_birth_scalar_genome_v3.o"
    completed = subprocess.run(
        [
            nvcc,
            "-allow-unsupported-compiler",
            "-std=c++17",
            "-I",
            str(PROJECT_ROOT),
            "-c",
            str(CUDA_SOURCE),
            "-o",
            str(object_file),
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert object_file.is_file()
