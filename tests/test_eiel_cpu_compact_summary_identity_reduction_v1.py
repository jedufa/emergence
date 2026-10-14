from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sysconfig
from pathlib import Path
from types import ModuleType

import numpy as np
import pybind11
import pytest

from engine.native_runtime.compact_summary_digests import (
    active_agent_ids_digest_sha256,
)


ROOT = Path(__file__).resolve().parents[1]
HEADER = (
    ROOT
    / "engine/perf/native/cpu_compact_summary_identity_reduction_v1.hpp"
)
SOURCE = HEADER.with_suffix(".cpp")
DIGEST_SOURCE = (
    ROOT / "engine/perf/native/cpu_compact_summary_digests_v1.cpp"
)
BINDING = (
    ROOT
    / "tests/perf/test_eiel_cpu_compact_summary_identity_reduction_v1_binding.cpp"
)
MODULE_NAME = "eiel_cpu_compact_summary_identity_reduction_v1_test"
INT64_MAX = np.iinfo(np.int64).max


@pytest.fixture(scope="module")
def native_identity_reduction(
    tmp_path_factory: pytest.TempPathFactory,
) -> ModuleType:
    compiler = shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        pytest.skip("a C++17 compiler is unavailable")
    build = tmp_path_factory.mktemp("cpu_compact_summary_identity_reduction_v1")
    extension = build / f"{MODULE_NAME}{sysconfig.get_config_var('EXT_SUFFIX')}"
    completed = subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O2",
            "-shared",
            "-fPIC",
            "-fvisibility=hidden",
            "-Wall",
            "-Wextra",
            "-Werror",
            f"-I{pybind11.get_include()}",
            f"-I{sysconfig.get_paths()['include']}",
            f"-I{ROOT / 'engine/perf/native'}",
            str(BINDING),
            str(SOURCE),
            str(DIGEST_SOURCE),
            "-o",
            str(extension),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    spec = importlib.util.spec_from_file_location(MODULE_NAME, extension)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _assert_cleared_failure(output: dict[str, object], status: str) -> None:
    assert output["status"] == status
    assert output["is_cleared"] is True
    assert output["active_agent_ids_digest"] == bytes(32)
    assert output["active_agent_ids_digest_hex"] == "0" * 64


def test_framework_free_strict_cxx17_compile(tmp_path: Path) -> None:
    production = HEADER.read_text(encoding="ascii") + SOURCE.read_text(
        encoding="ascii"
    )
    for forbidden in (
        "#include <cuda",
        "cuda_runtime",
        "#include <torch",
        "ATen/",
        "Python.h",
        "pybind11",
        "std::vector",
        "malloc(",
        "calloc(",
        "operator new",
        "throw ",
    ):
        assert forbidden not in production

    compiler = shutil.which("g++") or shutil.which("clang++")
    assert compiler is not None
    completed = subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O2",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-pedantic",
            "-fno-exceptions",
            "-fno-rtti",
            f"-I{ROOT / 'engine/perf/native'}",
            "-c",
            str(SOURCE),
            "-o",
            str(tmp_path / "cpu_compact_summary_identity_reduction_v1.o"),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize(
    ("capacity", "population_next_id"),
    [(0, 0), (1, 0), (17, 1), (int(INT64_MAX), int(INT64_MAX))],
)
def test_empty_identity_exact_digest_and_boundaries(
    native_identity_reduction: ModuleType,
    capacity: int,
    population_next_id: int,
) -> None:
    ids = np.array([], dtype=np.int64)
    output = native_identity_reduction.identity(
        ids, capacity, population_next_id
    )
    assert output["status"] == "success"
    assert output["active_agent_count"] == 0
    assert output["active_agent_id_min_defined"] is False
    assert output["active_agent_id_min"] == 0
    assert output["active_agent_id_max_defined"] is False
    assert output["active_agent_id_max"] == 0
    assert output["free_slot_count"] == capacity
    assert output["population_next_id"] == population_next_id
    assert output["active_agent_ids_digest_hex"] == (
        active_agent_ids_digest_sha256(ids)
    )


@pytest.mark.parametrize(
    ("ids", "capacity", "population_next_id"),
    [
        (np.array([0], dtype=np.int64), 1, 1),
        (np.array([0, 4, 17], dtype=np.int64), 5, 18),
        (
            np.array([int(INT64_MAX) - 1], dtype=np.int64),
            int(INT64_MAX),
            int(INT64_MAX),
        ),
    ],
)
def test_nonempty_identity_exact_digest_and_boundaries(
    native_identity_reduction: ModuleType,
    ids: np.ndarray,
    capacity: int,
    population_next_id: int,
) -> None:
    output = native_identity_reduction.identity(
        ids, capacity, population_next_id
    )
    assert output["status"] == "success"
    assert output["active_agent_count"] == ids.size
    assert output["active_agent_id_min_defined"] is True
    assert output["active_agent_id_min"] == int(ids[0])
    assert output["active_agent_id_max_defined"] is True
    assert output["active_agent_id_max"] == int(ids[-1])
    assert output["free_slot_count"] == capacity - ids.size
    assert output["population_next_id"] == population_next_id
    assert output["active_agent_ids_digest_hex"] == (
        active_agent_ids_digest_sha256(ids)
    )
    assert output["active_agent_ids_digest"] == bytes.fromhex(
        output["active_agent_ids_digest_hex"]
    )


@pytest.mark.parametrize(
    ("ids", "status"),
    [
        (np.array([-1, 2], dtype=np.int64), "active_agent_id_invalid"),
        (np.array([1, 1], dtype=np.int64), "active_agent_id_invalid"),
        (np.array([2, 1], dtype=np.int64), "active_agent_id_invalid"),
        (np.array([1, 2], dtype=np.int32), "dtype_mismatch"),
        (np.array([[1, 2]], dtype=np.int64), "rank_mismatch"),
    ],
)
def test_malformed_ids_fail_closed(
    native_identity_reduction: ModuleType,
    ids: np.ndarray,
    status: str,
) -> None:
    _assert_cleared_failure(
        native_identity_reduction.identity(ids, 4, 4), status
    )


def test_noncontiguous_and_storage_mismatch_fail_closed(
    native_identity_reduction: ModuleType,
) -> None:
    ids = np.arange(8, dtype=np.int64)[::2]
    assert not ids.flags.c_contiguous
    _assert_cleared_failure(
        native_identity_reduction.identity(ids, 8, 8), "non_contiguous"
    )
    contiguous = np.array([1, 3], dtype=np.int64)
    _assert_cleared_failure(
        native_identity_reduction.identity_with_storage_bytes(
            contiguous, 2, 4, contiguous.nbytes - 1
        ),
        "storage_size_mismatch",
    )


@pytest.mark.parametrize(
    ("ids", "capacity", "population_next_id", "status"),
    [
        (np.array([], dtype=np.int64), -1, 0, "capacity_out_of_range"),
        (
            np.array([], dtype=np.int64),
            2**63,
            0,
            "capacity_out_of_range",
        ),
        (
            np.array([], dtype=np.int64),
            0,
            -1,
            "population_next_id_out_of_range",
        ),
        (
            np.array([], dtype=np.int64),
            0,
            2**63,
            "population_next_id_out_of_range",
        ),
        (
            np.array([1, 2], dtype=np.int64),
            1,
            3,
            "active_count_exceeds_capacity",
        ),
        (
            np.array([1, 3], dtype=np.int64),
            2,
            3,
            "population_next_id_not_above_maximum",
        ),
        (
            np.array([1, 3], dtype=np.int64),
            2,
            2,
            "population_next_id_not_above_maximum",
        ),
        (
            np.array([int(INT64_MAX)], dtype=np.int64),
            1,
            int(INT64_MAX),
            "population_next_id_not_above_maximum",
        ),
    ],
)
def test_scalar_and_identity_ranges_fail_closed(
    native_identity_reduction: ModuleType,
    ids: np.ndarray,
    capacity: int,
    population_next_id: int,
    status: str,
) -> None:
    _assert_cleared_failure(
        native_identity_reduction.identity(
            ids, capacity, population_next_id
        ),
        status,
    )


@pytest.mark.parametrize(
    (
        "capacity",
        "active_count",
        "minimum_defined",
        "minimum",
        "maximum_defined",
        "maximum",
        "free_slots",
        "population_next_id",
        "status",
    ),
    [
        (4, 0, 1, 0, 1, 0, 4, 0, "defined_state_invalid"),
        (1, 1, 0, 0, 0, 0, 0, 1, "defined_state_invalid"),
        (1, 1, 1, 0, 0, 0, 0, 1, "defined_state_invalid"),
        (1, 1, 2, 0, 1, 0, 0, 1, "defined_flag_invalid"),
        (1, 1, 1, 0, 1, 1, 0, 2, "active_agent_id_bounds_invalid"),
        (2, 2, 1, 1, 1, 1, 0, 2, "active_agent_id_bounds_invalid"),
        (
            int(INT64_MAX),
            int(INT64_MAX),
            1,
            0,
            1,
            int(INT64_MAX) - 1,
            1,
            int(INT64_MAX),
            "slot_conservation_invalid",
        ),
    ],
)
def test_impossible_result_states_fail_closed(
    native_identity_reduction: ModuleType,
    capacity: int,
    active_count: int,
    minimum_defined: int,
    minimum: int,
    maximum_defined: int,
    maximum: int,
    free_slots: int,
    population_next_id: int,
    status: str,
) -> None:
    output = native_identity_reduction.validate_candidate(
        capacity,
        active_count,
        minimum_defined,
        minimum,
        maximum_defined,
        maximum,
        free_slots,
        population_next_id,
    )
    _assert_cleared_failure(output, status)
