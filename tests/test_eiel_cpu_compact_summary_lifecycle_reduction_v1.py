from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sysconfig
from collections.abc import Mapping
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pybind11
import pytest
import torch

from engine.native_runtime.compact_summary_authority_adapter import (
    CompactSummaryAuthorityAdapterError,
    _lifecycle_authority,
)
from engine.native_runtime.lifecycle_events import (
    LifecycleEventType,
    typed_lifecycle_event_digest,
)


ROOT = Path(__file__).resolve().parents[1]
HEADER = (
    ROOT
    / "engine/perf/native/cpu_compact_summary_lifecycle_reduction_v1.hpp"
)
SOURCE = HEADER.with_suffix(".cpp")
DIGEST_SOURCE = ROOT / "engine/perf/native/cpu_compact_summary_digests_v1.cpp"
BINDING = (
    ROOT
    / "tests/perf/test_eiel_cpu_compact_summary_lifecycle_reduction_v1_binding.cpp"
)
MODULE_NAME = "eiel_cpu_compact_summary_lifecycle_reduction_v1_test"

LIFECYCLE_DTYPES = {
    "event_type": np.dtype(np.int32),
    "generation": np.dtype(np.int64),
    "tick": np.dtype(np.int64),
    "sequence": np.dtype(np.int64),
    "actor_agent_id": np.dtype(np.int64),
    "target_agent_id": np.dtype(np.int64),
    "actor_slot": np.dtype(np.int64),
    "target_slot": np.dtype(np.int64),
    "parent_agent_id": np.dtype(np.int64),
    "child_agent_id": np.dtype(np.int64),
    "lineage_founder_id": np.dtype(np.int64),
    "lineage_depth": np.dtype(np.int32),
    "subtype_code": np.dtype(np.int32),
    "reason_code": np.dtype(np.int32),
    "success": np.dtype(np.int8),
    "value": np.dtype(np.float32),
}


@pytest.fixture(scope="module")
def native_reduction(tmp_path_factory: pytest.TempPathFactory) -> ModuleType:
    compiler = shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        pytest.skip("a C++17 compiler is unavailable")
    build = tmp_path_factory.mktemp("cpu_compact_summary_lifecycle_reduction_v1")
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


def _events(capacity: int) -> dict[str, np.ndarray]:
    events = {
        name: np.full(capacity, -1, dtype=dtype)
        for name, dtype in LIFECYCLE_DTYPES.items()
    }
    events["event_type"].fill(int(LifecycleEventType.BIRTH_COMMITTED))
    events["generation"].fill(7)
    events["tick"][:] = np.arange(capacity, dtype=np.int64) // 2
    events["sequence"][:] = np.arange(capacity, dtype=np.int64)
    events["actor_agent_id"][:] = np.arange(100, 100 + capacity)
    events["target_agent_id"][:] = np.arange(200, 200 + capacity)
    events["actor_slot"][:] = np.arange(capacity)
    events["target_slot"][:] = np.arange(capacity)
    events["parent_agent_id"][:] = np.arange(100, 100 + capacity)
    events["child_agent_id"][:] = np.arange(200, 200 + capacity)
    events["lineage_founder_id"].fill(3)
    events["lineage_depth"].fill(2)
    events["subtype_code"].fill(0)
    events["reason_code"].fill(-1)
    events["success"].fill(1)
    events["value"].fill(0.0)
    return {name: np.ascontiguousarray(value) for name, value in events.items()}


def _set_event(
    events: dict[str, np.ndarray],
    row: int,
    event_type: int,
    *,
    reason_code: int = -1,
) -> None:
    events["event_type"][row] = event_type
    events["reason_code"][row] = reason_code
    events["subtype_code"][row] = event_type
    events["value"][row] = float(reason_code if reason_code >= 0 else 0)
    if event_type == int(LifecycleEventType.BIRTH_FAILED):
        events["success"][row] = 0


def _projection(
    events: Mapping[str, np.ndarray], cursor: int
) -> dict[str, Any]:
    return {
        "metadata": {"typed_lifecycle_event_active_count": cursor},
        "tensors": {
            f"typed_lifecycle_event_{name}": torch.from_numpy(value)
            for name, value in events.items()
        },
        "event_cursor": cursor,
    }


def _python_reference(
    events: Mapping[str, np.ndarray], start: int, pre_turnover: int, end: int
) -> dict[str, Any]:
    generation_types = events["event_type"][start:pre_turnover]
    generation_reasons = events["reason_code"][start:pre_turnover]
    generation_deaths = generation_reasons[
        generation_types == int(LifecycleEventType.DEATH_COMMITTED)
    ]
    authority, generation_counts, turnover_counts, deaths = _lifecycle_authority(
        {
            "typed_event_rows": pre_turnover - start,
            "death_rows": int(generation_deaths.size),
        },
        {"turnover_result": {"typed_event_rows": end - pre_turnover}},
        _projection(events, pre_turnover),
        _projection(events, end),
    )
    prefix = {
        name: torch.from_numpy(np.ascontiguousarray(value[:end]))
        for name, value in events.items()
    }
    return {
        **authority,
        "generation_counts_by_type": generation_counts,
        "turnover_counts_by_type": turnover_counts,
        "deaths_by_class": deaths,
        "digest": typed_lifecycle_event_digest(prefix),
    }


def _assert_matches_python(
    native_reduction: ModuleType,
    events: dict[str, np.ndarray],
    start: int,
    pre_turnover: int,
    end: int,
) -> None:
    expected = _python_reference(events, start, pre_turnover, end)
    actual = native_reduction.reduce(events, start, pre_turnover, end)
    assert actual["status"] == "success"
    assert actual["event_row_count"] == expected["event_row_count"]
    assert actual["generation_event_count"] == expected["generation_event_count"]
    assert actual["turnover_event_count"] == expected["turnover_event_count"]
    count_names = ("birth_committed", "mutation", "birth_failed", "death_committed")
    assert actual["generation_counts_by_type"] == tuple(
        expected["generation_counts_by_type"][name] for name in count_names
    )
    assert actual["turnover_counts_by_type"] == tuple(
        expected["turnover_counts_by_type"][name] for name in count_names
    )
    assert actual["deaths_energy_depleted"] == expected["deaths_by_class"][
        "energy_depleted"
    ]
    assert actual["deaths_health_depleted"] == expected["deaths_by_class"][
        "health_depleted"
    ]
    assert actual["hex"] == expected["digest"]
    assert actual["digest"] == bytes.fromhex(expected["digest"])
    assert actual["is_cleared"] is False


def _assert_failure(
    output: Mapping[str, Any], expected_status: str
) -> None:
    assert output["status"] == expected_status
    assert output["is_cleared"] is True
    assert output["event_row_count"] == 0
    assert output["generation_event_count"] == 0
    assert output["turnover_event_count"] == 0
    assert output["generation_counts_by_type"] == (0, 0, 0, 0)
    assert output["turnover_counts_by_type"] == (0, 0, 0, 0)
    assert output["deaths_energy_depleted"] == 0
    assert output["deaths_health_depleted"] == 0
    assert output["digest"] == bytes(32)
    assert output["hex"] == "0" * 64


def test_framework_free_allocation_free_strict_cxx17_compile(
    tmp_path: Path,
) -> None:
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
        "std::string",
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
            str(tmp_path / "cpu_compact_summary_lifecycle_reduction_v1.o"),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_empty_prefix_matches_certified_python(
    native_reduction: ModuleType,
) -> None:
    _assert_matches_python(native_reduction, _events(0), 0, 0, 0)


def test_flat_like_generation_slice_matches_certified_python(
    native_reduction: ModuleType,
) -> None:
    events = _events(9)
    for row, (event_type, reason) in enumerate(
        ((3, 4), (1, -1), (2, -1), (3, 1), (4, 1), (4, 2))
    ):
        _set_event(events, row, event_type, reason_code=reason)
    events["value"][6:] = np.nan
    events["sequence"][6:] = -99
    _assert_matches_python(native_reduction, events, 1, 6, 6)


def test_turnover_slice_matches_certified_python(
    native_reduction: ModuleType,
) -> None:
    events = _events(9)
    specifications = (
        (3, 2),
        (1, -1),
        (1, -1),
        (4, 1),
        (2, -1),
        (4, 2),
        (3, 7),
        (2, -1),
        (4, 99),
    )
    for row, (event_type, reason) in enumerate(specifications):
        _set_event(events, row, event_type, reason_code=reason)
    _assert_matches_python(native_reduction, events, 2, 6, 9)


@pytest.mark.parametrize(
    ("start", "pre_turnover", "end", "status"),
    [
        (-1, 0, 0, "cursor_out_of_range"),
        (2, 1, 3, "cursor_order_invalid"),
        (0, 3, 2, "cursor_order_invalid"),
        (0, 2, 5, "cursor_out_of_range"),
    ],
)
def test_cursor_failures_clear_output(
    native_reduction: ModuleType,
    start: int,
    pre_turnover: int,
    end: int,
    status: str,
) -> None:
    _assert_failure(native_reduction.reduce(_events(4), start, pre_turnover, end), status)


@pytest.mark.parametrize(
    ("malformation", "status"),
    [
        ("unknown_generation_event", "unknown_event_type"),
        ("unknown_turnover_event", "unknown_event_type"),
        ("unknown_generation_death_reason", "unknown_generation_death_reason"),
    ],
)
def test_semantic_malformed_cases_match_python_failure_and_clear_output(
    native_reduction: ModuleType,
    malformation: str,
    status: str,
) -> None:
    events = _events(4)
    _set_event(events, 1, int(LifecycleEventType.DEATH_COMMITTED), reason_code=1)
    if malformation == "unknown_generation_event":
        _set_event(events, 1, 99)
    elif malformation == "unknown_turnover_event":
        _set_event(events, 3, 99)
    else:
        events["reason_code"][1] = 99

    with pytest.raises(CompactSummaryAuthorityAdapterError):
        _python_reference(events, 0, 3, 4)
    _assert_failure(native_reduction.reduce(events, 0, 3, 4), status)


@pytest.mark.parametrize(
    "malformation",
    ["short_column", "wrong_dtype", "wrong_rank", "sequence", "row_order", "nan", "noncontiguous"],
)
def test_underlying_digest_validation_failures_clear_output(
    native_reduction: ModuleType, malformation: str
) -> None:
    events = _events(4)
    if malformation == "short_column":
        events["actor_slot"] = events["actor_slot"][:-1].copy()
    elif malformation == "wrong_dtype":
        events["value"] = events["value"].astype(np.float64)
    elif malformation == "wrong_rank":
        events["event_type"] = events["event_type"].reshape(2, 2)
    elif malformation == "sequence":
        events["sequence"][0] = 9
    elif malformation == "row_order":
        events["tick"][2] = -1
    elif malformation == "nan":
        events["value"][1] = np.nan
    else:
        backing = np.zeros(8, dtype=np.int64)
        events["actor_agent_id"] = backing[::2]
        assert not events["actor_agent_id"].flags.c_contiguous
    _assert_failure(
        native_reduction.reduce(events, 1, 3, 4),
        "digest_validation_failed",
    )
