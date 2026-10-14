from __future__ import annotations

import hashlib
import importlib.util
import shutil
import subprocess
import sysconfig
from collections.abc import Mapping
from pathlib import Path
from types import ModuleType

import numpy as np
import pybind11
import pytest
import torch

from engine.native_runtime.compact_summary_digests import (
    active_agent_ids_digest_sha256,
    policy_mutation_tape_digest_sha256,
)
from engine.native_runtime.lifecycle_events import typed_lifecycle_event_digest


ROOT = Path(__file__).resolve().parents[1]
HEADER = ROOT / "engine/perf/native/cpu_compact_summary_digests_v1.hpp"
SOURCE = HEADER.with_suffix(".cpp")
BINDING = (
    ROOT / "tests/perf/test_eiel_cpu_compact_summary_digests_v1_binding.cpp"
)
MODULE_NAME = "eiel_cpu_compact_summary_digests_v1_test"

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
def native_digests(tmp_path_factory: pytest.TempPathFactory) -> ModuleType:
    compiler = shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        pytest.skip("a C++17 compiler is unavailable")
    build = tmp_path_factory.mktemp("cpu_compact_summary_digests_v1")
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


def _events(capacity: int, *, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    events: dict[str, np.ndarray] = {}
    for name, dtype in LIFECYCLE_DTYPES.items():
        if name == "event_type":
            value = rng.integers(1, 5, size=capacity, dtype=np.int32)
        elif name == "generation":
            value = np.full(capacity, seed % 7, dtype=np.int64)
        elif name == "tick":
            value = np.arange(capacity, dtype=np.int64) // 2
        elif name == "sequence":
            value = np.arange(capacity, dtype=np.int64)
        elif name == "success":
            value = rng.integers(0, 2, size=capacity, dtype=np.int8)
        elif name == "value":
            value = rng.normal(size=capacity).astype(np.float32)
        elif dtype == np.dtype(np.int32):
            value = rng.integers(-1, 9, size=capacity, dtype=np.int32)
        else:
            value = rng.integers(-1, 5000, size=capacity, dtype=np.int64)
        events[name] = np.ascontiguousarray(value)
    return events


def _python_lifecycle_digest(
    events: Mapping[str, np.ndarray], active_count: int
) -> str:
    return typed_lifecycle_event_digest(
        {
            name: torch.from_numpy(np.ascontiguousarray(value[:active_count]))
            for name, value in events.items()
        }
    )


def _policy_tape(
    capacity: int,
    cursor: int,
    *,
    width: int,
    state_width: int,
    seed: int,
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    return {
        "policy_birth_crossover_mask_tape": np.ascontiguousarray(
            rng.integers(0, 2, size=(capacity, width), dtype=np.uint8).astype(np.bool_)
        ),
        "policy_birth_delta_tape": np.ascontiguousarray(
            rng.normal(size=(capacity, width)).astype(np.float32)
        ),
        "policy_birth_mutation_mask_tape": np.ascontiguousarray(
            rng.integers(0, 2, size=(capacity, width), dtype=np.uint8).astype(np.bool_)
        ),
        "policy_birth_post_rng_state_tape": np.ascontiguousarray(
            rng.integers(0, 256, size=(capacity, state_width), dtype=np.uint8)
        ),
        "policy_birth_semantic_draws_tape": np.ascontiguousarray(
            rng.integers(0, 10000, size=capacity, dtype=np.int64)
        ),
        "policy_birth_tape_cursor": np.array([cursor], dtype=np.int64),
    }


def _assert_failure(output: Mapping[str, object], status: str) -> None:
    assert output["status"] == status
    assert output["is_zero"] is True
    assert output["digest"] == bytes(32)
    assert output["hex"] == "0" * 64


def _tensor_byte_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes(order="C")).hexdigest()


def _assert_descriptor_failure(output: Mapping[str, object], status: str) -> None:
    assert output["status"] == status
    assert output["valid"] is False
    assert output["raw_is_zero"] is True
    assert output["canonical_is_zero"] is True
    assert output["raw_digest"] == bytes(32)
    assert output["canonical_digest"] == bytes(32)
    assert output["raw_hex"] == "0" * 64
    assert output["canonical_hex"] == "0" * 64
    assert output["returned_status_code"] == output["status_code"]


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
            str(tmp_path / "cpu_compact_summary_digests_v1.o"),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_checkpoint_tensor_descriptor_modes_match_cuda_contract(
    native_digests: ModuleType,
) -> None:
    assert native_digests.mode_exact == 0
    assert native_digests.mode_dead_row_zero == 1
    assert native_digests.mode_active_prefix == 2


def test_checkpoint_tensor_exact_matches_python_bytes_for_supported_dtypes(
    native_digests: ModuleType,
) -> None:
    tensors = (
        np.array([True, False, True], dtype=np.bool_),
        np.array([-2, 0, 7], dtype=np.int8),
        np.array([0, 127, 255], dtype=np.uint8),
        np.array([[1, -2], [3, 4]], dtype=np.int32),
        np.array([[-0.0, 1.25], [2.5, -3.75]], dtype=np.float32),
        np.array([11, -22, 33], dtype=np.int64),
        np.array([[-0.0, 1.0], [2.0, -3.0]], dtype=np.float64),
    )
    for tensor in tensors:
        value = np.ascontiguousarray(tensor)
        expected = _tensor_byte_sha256(value)
        output = native_digests.checkpoint_tensor_descriptor(
            value, native_digests.mode_exact
        )
        assert output["status"] == "success"
        assert output["valid"] is True
        assert output["raw_hex"] == expected
        assert output["canonical_hex"] == expected
        assert output["raw_digest"] == bytes.fromhex(expected)
        assert output["canonical_digest"] == bytes.fromhex(expected)


def test_checkpoint_tensor_dead_rows_match_python_and_isolate_dead_payload(
    native_digests: ModuleType,
) -> None:
    alive = np.array([True, False, True, False], dtype=np.bool_)
    first = np.arange(12, dtype=np.float32).reshape(4, 3)
    second = first.copy()
    second[~alive] = np.array([[1000.0, -0.0, -1000.0], [9.5, 8.5, 7.5]])
    canonical = first.copy()
    canonical[~alive] = np.zeros_like(canonical[~alive])

    first_output = native_digests.checkpoint_tensor_descriptor(
        first, native_digests.mode_dead_row_zero, alive
    )
    second_output = native_digests.checkpoint_tensor_descriptor(
        second, native_digests.mode_dead_row_zero, alive
    )
    assert first_output["status"] == second_output["status"] == "success"
    assert first_output["valid"] is second_output["valid"] is True
    assert first_output["raw_hex"] == _tensor_byte_sha256(first)
    assert second_output["raw_hex"] == _tensor_byte_sha256(second)
    assert first_output["raw_hex"] != second_output["raw_hex"]
    assert first_output["canonical_hex"] == _tensor_byte_sha256(canonical)
    assert second_output["canonical_hex"] == first_output["canonical_hex"]


@pytest.mark.parametrize("active_length", [0, 1, 3, 5])
def test_checkpoint_tensor_active_prefix_matches_python_and_isolates_tail(
    native_digests: ModuleType,
    active_length: int,
) -> None:
    first = np.arange(15, dtype=np.int64).reshape(5, 3)
    second = first.copy()
    second[active_length:] += 100000
    expected = _tensor_byte_sha256(first[:active_length])

    first_output = native_digests.checkpoint_tensor_descriptor(
        first,
        native_digests.mode_active_prefix,
        active_length=active_length,
    )
    second_output = native_digests.checkpoint_tensor_descriptor(
        second,
        native_digests.mode_active_prefix,
        active_length=active_length,
    )
    assert first_output["status"] == second_output["status"] == "success"
    assert first_output["valid"] is second_output["valid"] is True
    assert first_output["raw_hex"] == _tensor_byte_sha256(first)
    assert second_output["raw_hex"] == _tensor_byte_sha256(second)
    assert first_output["canonical_hex"] == expected
    assert second_output["canonical_hex"] == expected
    if active_length < first.shape[0]:
        assert first_output["raw_hex"] != second_output["raw_hex"]


def test_checkpoint_tensor_descriptor_errors_fail_closed(
    native_digests: ModuleType,
) -> None:
    tensor = np.arange(6, dtype=np.int32).reshape(3, 2)
    alive = np.array([True, False, True], dtype=np.bool_)
    cases = (
        (
            native_digests.checkpoint_tensor_descriptor(
                tensor.astype(np.complex64), native_digests.mode_exact
            ),
            "dtype_mismatch",
        ),
        (
            native_digests.checkpoint_tensor_descriptor(
                tensor.reshape(1, 3, 2), native_digests.mode_exact
            ),
            "rank_mismatch",
        ),
        (
            native_digests.checkpoint_tensor_descriptor(
                tensor,
                native_digests.mode_exact,
                extent0_override=-1,
            ),
            "negative_extent",
        ),
        (
            native_digests.checkpoint_tensor_descriptor(
                tensor,
                native_digests.mode_exact,
                extent1_override=0,
                storage_bytes_override=0,
            ),
            "invalid_row_layout",
        ),
        (
            native_digests.checkpoint_tensor_descriptor(
                tensor,
                native_digests.mode_exact,
                storage_bytes_override=tensor.nbytes + 1,
            ),
            "storage_size_mismatch",
        ),
        (
            native_digests.checkpoint_tensor_descriptor(
                tensor[:, ::-1], native_digests.mode_exact
            ),
            "non_contiguous",
        ),
        (
            native_digests.checkpoint_tensor_descriptor(
                tensor, native_digests.mode_exact, null_data=True
            ),
            "null_data",
        ),
        (
            native_digests.checkpoint_tensor_descriptor(tensor, 99),
            "invalid_mode",
        ),
        (
            native_digests.checkpoint_tensor_descriptor(
                tensor, native_digests.mode_dead_row_zero
            ),
            "null_active_source",
        ),
        (
            native_digests.checkpoint_tensor_descriptor(
                tensor,
                native_digests.mode_dead_row_zero,
                alive.astype(np.uint8),
            ),
            "dtype_mismatch",
        ),
        (
            native_digests.checkpoint_tensor_descriptor(
                tensor,
                native_digests.mode_dead_row_zero,
                alive[:2],
            ),
            "invalid_row_layout",
        ),
        (
            native_digests.checkpoint_tensor_descriptor(
                tensor,
                native_digests.mode_active_prefix,
                active_length=-1,
            ),
            "active_length_out_of_range",
        ),
        (
            native_digests.checkpoint_tensor_descriptor(
                tensor,
                native_digests.mode_active_prefix,
                active_length=4,
            ),
            "active_length_out_of_range",
        ),
    )
    for output, expected_status in cases:
        _assert_descriptor_failure(output, expected_status)


def test_checkpoint_tensor_invalid_boolean_mask_fails_closed(
    native_digests: ModuleType,
) -> None:
    tensor = np.arange(6, dtype=np.int32).reshape(3, 2)
    alive = np.array([True, False, True], dtype=np.bool_)
    alive.view(np.uint8)[1] = 2
    output = native_digests.checkpoint_tensor_descriptor(
        tensor, native_digests.mode_dead_row_zero, alive
    )
    _assert_descriptor_failure(output, "boolean_value_invalid")


@pytest.mark.parametrize("count", [0, 1, 2, 17, 257])
def test_active_agent_ids_match_python_randomized(
    native_digests: ModuleType, count: int
) -> None:
    rng = np.random.default_rng(1000 + count)
    ids = np.sort(
        rng.choice(np.arange(0, 100000, dtype=np.int64), size=count, replace=False)
    )
    output = native_digests.active_agent_ids(np.ascontiguousarray(ids))
    assert output["status"] == "success"
    assert output["hex"] == active_agent_ids_digest_sha256(ids)
    assert output["digest"] == bytes.fromhex(output["hex"])


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
def test_active_agent_ids_fail_closed(
    native_digests: ModuleType, ids: np.ndarray, status: str
) -> None:
    _assert_failure(native_digests.active_agent_ids(ids), status)


def test_active_agent_ids_reject_noncontiguous_storage(
    native_digests: ModuleType,
) -> None:
    ids = np.arange(8, dtype=np.int64)[::2]
    assert not ids.flags.c_contiguous
    _assert_failure(native_digests.active_agent_ids(ids), "non_contiguous")


@pytest.mark.parametrize(
    ("capacity", "active_count", "seed"),
    [(0, 0, 1), (4, 0, 2), (4, 1, 3), (8, 5, 4), (33, 33, 5)],
)
def test_lifecycle_active_prefix_matches_python(
    native_digests: ModuleType,
    capacity: int,
    active_count: int,
    seed: int,
) -> None:
    events = _events(capacity, seed=seed)
    if capacity > active_count:
        events["value"][active_count:] = np.nan
        events["sequence"][active_count:] = -99
    output = native_digests.lifecycle(events, active_count)
    assert output["status"] == "success"
    assert output["hex"] == _python_lifecycle_digest(events, active_count)


@pytest.mark.parametrize("seed", range(10, 30))
def test_lifecycle_randomized_byte_exactness(
    native_digests: ModuleType, seed: int
) -> None:
    capacity = seed % 13 + 1
    events = _events(capacity, seed=seed)
    output = native_digests.lifecycle(events, capacity)
    assert output["status"] == "success"
    assert output["hex"] == _python_lifecycle_digest(events, capacity)


@pytest.mark.parametrize(
    ("mutation", "status"),
    [
        ("short_column", "column_length_mismatch"),
        ("wrong_dtype", "dtype_mismatch"),
        ("wrong_rank", "rank_mismatch"),
        ("sequence", "sequence_mismatch"),
        ("row_order", "row_order_invalid"),
        ("nan", "non_finite_value"),
        ("positive_inf", "non_finite_value"),
    ],
)
def test_lifecycle_malformed_authority_fails_closed(
    native_digests: ModuleType, mutation: str, status: str
) -> None:
    events = _events(4, seed=77)
    if mutation == "short_column":
        events["actor_slot"] = events["actor_slot"][:-1].copy()
    elif mutation == "wrong_dtype":
        events["value"] = events["value"].astype(np.float64)
    elif mutation == "wrong_rank":
        events["event_type"] = events["event_type"].reshape(2, 2)
    elif mutation == "sequence":
        events["sequence"][2] = 8
    elif mutation == "row_order":
        events["tick"][2] = -1
    elif mutation == "nan":
        events["value"][1] = np.nan
    elif mutation == "positive_inf":
        events["value"][1] = np.inf
    _assert_failure(native_digests.lifecycle(events, 4), status)


@pytest.mark.parametrize(("cursor", "status"), [(-1, "cursor_out_of_range"), (5, "cursor_out_of_range")])
def test_lifecycle_cursor_bounds_fail_closed(
    native_digests: ModuleType, cursor: int, status: str
) -> None:
    _assert_failure(native_digests.lifecycle(_events(4, seed=44), cursor), status)


@pytest.mark.parametrize(
    ("capacity", "cursor", "width", "state_width", "seed"),
    [
        (0, 0, 0, 0, 100),
        (4, 0, 3, 17, 101),
        (4, 1, 3, 17, 102),
        (4, 4, 3, 17, 103),
        (19, 11, 8, 64, 104),
    ],
)
def test_policy_tape_consumed_prefix_matches_python(
    native_digests: ModuleType,
    capacity: int,
    cursor: int,
    width: int,
    state_width: int,
    seed: int,
) -> None:
    tape = _policy_tape(
        capacity,
        cursor,
        width=width,
        state_width=state_width,
        seed=seed,
    )
    output = native_digests.policy_tape(tape)
    assert output["status"] == "success"
    assert output["hex"] == policy_mutation_tape_digest_sha256(tape)


@pytest.mark.parametrize("seed", range(200, 220))
def test_policy_tape_randomized_byte_exactness(
    native_digests: ModuleType, seed: int
) -> None:
    capacity = seed % 11 + 1
    cursor = seed % (capacity + 1)
    tape = _policy_tape(
        capacity,
        cursor,
        width=seed % 5 + 1,
        state_width=seed % 13 + 1,
        seed=seed,
    )
    output = native_digests.policy_tape(tape)
    assert output["status"] == "success"
    assert output["hex"] == policy_mutation_tape_digest_sha256(tape)


def test_policy_tape_ignores_inactive_tail_and_mapping_insertion_order(
    native_digests: ModuleType,
) -> None:
    first = _policy_tape(5, 2, width=4, state_width=9, seed=400)
    second = {
        key: np.copy(value) if isinstance(value, np.ndarray) else value
        for key, value in reversed(tuple(first.items()))
    }
    second["policy_birth_delta_tape"][2:] = np.nan
    second["policy_birth_crossover_mask_tape"][2:] = True
    second["policy_birth_mutation_mask_tape"][2:] = True
    second["policy_birth_post_rng_state_tape"][2:] = 255
    second["policy_birth_semantic_draws_tape"][2:] = 999999

    first_output = native_digests.policy_tape(first)
    second_output = native_digests.policy_tape(second)
    assert first_output["status"] == second_output["status"] == "success"
    assert first_output["hex"] == second_output["hex"]
    assert first_output["hex"] == policy_mutation_tape_digest_sha256(first)


@pytest.mark.parametrize(
    ("mutation", "status"),
    [
        ("negative_cursor", "cursor_out_of_range"),
        ("long_cursor", "cursor_out_of_range"),
        ("capacity", "column_length_mismatch"),
        ("mask_shape", "policy_mask_shape_mismatch"),
        ("dtype", "dtype_mismatch"),
        ("rank", "rank_mismatch"),
        ("nonfinite", "non_finite_value"),
        ("noncontiguous", "non_contiguous"),
        ("invalid_bool", "boolean_value_invalid"),
    ],
)
def test_policy_tape_malformed_authority_fails_closed(
    native_digests: ModuleType, mutation: str, status: str
) -> None:
    tape = _policy_tape(4, 3, width=3, state_width=8, seed=500)
    if mutation == "negative_cursor":
        tape["policy_birth_tape_cursor"] = -1
    elif mutation == "long_cursor":
        tape["policy_birth_tape_cursor"] = 5
    elif mutation == "capacity":
        tape["policy_birth_semantic_draws_tape"] = np.zeros(3, dtype=np.int64)
    elif mutation == "mask_shape":
        tape["policy_birth_mutation_mask_tape"] = np.zeros((4, 2), dtype=np.bool_)
    elif mutation == "dtype":
        tape["policy_birth_delta_tape"] = np.zeros((4, 3), dtype=np.float64)
    elif mutation == "rank":
        tape["policy_birth_semantic_draws_tape"] = np.zeros((4, 1), dtype=np.int64)
    elif mutation == "nonfinite":
        tape["policy_birth_delta_tape"][1, 1] = -np.inf
    elif mutation == "noncontiguous":
        backing = np.zeros((4, 6), dtype=np.float32)
        tape["policy_birth_delta_tape"] = backing[:, ::2]
    elif mutation == "invalid_bool":
        mask = tape["policy_birth_crossover_mask_tape"]
        mask.view(np.uint8)[0, 0] = 2
    _assert_failure(native_digests.policy_tape(tape), status)


def test_policy_nonfinite_inactive_tail_is_accepted(
    native_digests: ModuleType,
) -> None:
    tape = _policy_tape(4, 1, width=3, state_width=8, seed=600)
    tape["policy_birth_delta_tape"][1:] = np.nan
    output = native_digests.policy_tape(tape)
    assert output["status"] == "success"
    assert output["hex"] == policy_mutation_tape_digest_sha256(tape)
