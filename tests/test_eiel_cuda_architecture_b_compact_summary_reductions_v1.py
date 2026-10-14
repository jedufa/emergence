from __future__ import annotations

import copy
import ctypes
import os
import re
import shutil
import subprocess
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
HEADER = ROOT / "engine/perf/cuda_architecture_b_compact_summary_reductions_v1.cuh"
CUDA_SOURCE = ROOT / "engine/perf/cuda_architecture_b_compact_summary_reductions_v1.cu"
BINDING_SOURCE = (
    ROOT
    / "tests/perf/test_eiel_cuda_architecture_b_compact_summary_reductions_v1_binding.cpp"
)
COMPILE_TEST = (
    ROOT
    / "tests/perf/test_eiel_cuda_architecture_b_compact_summary_reductions_v1_compile.cu"
)

RESULT_BYTES = 456
SUMMARY_WIDTH = 31
REASON_COUNT = 10
EVENT_FIELDS = (
    "event_type",
    "generation",
    "tick",
    "sequence",
    "actor_agent_id",
    "target_agent_id",
    "actor_slot",
    "target_slot",
    "parent_agent_id",
    "child_agent_id",
    "lineage_founder_id",
    "lineage_depth",
    "subtype_code",
    "reason_code",
    "success",
    "value",
)
EVENT_DTYPES = (
    torch.int32,
    torch.int64,
    torch.int64,
    torch.int64,
    torch.int64,
    torch.int64,
    torch.int64,
    torch.int64,
    torch.int64,
    torch.int64,
    torch.int64,
    torch.int32,
    torch.int32,
    torch.int32,
    torch.int8,
    torch.float32,
)


class ReductionResultV1(ctypes.Structure):
    _fields_ = [
        ("status", ctypes.c_int32),
        ("valid", ctypes.c_uint8),
        ("active_id_bounds_defined", ctypes.c_uint8),
        ("reserved", ctypes.c_uint16),
        ("generation_event_count", ctypes.c_int64),
        ("turnover_event_count", ctypes.c_int64),
        ("generation_counts_by_type", ctypes.c_int64 * 4),
        ("turnover_counts_by_type", ctypes.c_int64 * 4),
        ("generation_failed_birth_counts_by_reason", ctypes.c_int64 * 9),
        ("turnover_failed_birth_counts_by_reason", ctypes.c_int64 * 9),
        ("birth_failure_counts_by_reason", ctypes.c_int64 * 9),
        ("native_failed_birth_event_rows", ctypes.c_int64),
        ("injury_blocked_births", ctypes.c_int64),
        ("birth_attempt_rows", ctypes.c_int64),
        ("birth_failure_rows", ctypes.c_int64),
        ("deaths_energy_depleted", ctypes.c_int64),
        ("deaths_health_depleted", ctypes.c_int64),
        ("cursor_start", ctypes.c_int64),
        ("cursor_pre_turnover", ctypes.c_int64),
        ("cursor_end", ctypes.c_int64),
        ("authoritative_cursor", ctypes.c_int64),
        ("event_row_count", ctypes.c_int64),
        ("active_event_count", ctypes.c_int64),
        ("typed_event_row_count", ctypes.c_int64),
        ("tape_event_row_count", ctypes.c_int64),
        ("population_next_id", ctypes.c_int64),
        ("active_agent_count", ctypes.c_int64),
        ("active_agent_id_min", ctypes.c_int64),
        ("active_agent_id_max", ctypes.c_int64),
        ("free_slot_count", ctypes.c_int64),
    ]


assert ctypes.sizeof(ReductionResultV1) == RESULT_BYTES


def _event(
    kind: int,
    sequence: int,
    *,
    generation: int,
    tick: int,
    reason: int | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "event_type": kind,
        "generation": generation,
        "tick": tick,
        "sequence": sequence,
        "actor_agent_id": 10,
        "target_agent_id": 20,
        "actor_slot": 0,
        "target_slot": 1,
        "parent_agent_id": 10,
        "child_agent_id": 20,
        "lineage_founder_id": 1,
        "lineage_depth": 2,
        "subtype_code": kind,
        "reason_code": -1,
        "success": 1,
        "value": 4.5,
    }
    if kind == 2:
        row.update(actor_agent_id=20, actor_slot=1, value=0.125)
    elif kind == 3:
        failure_reason = 1 if reason is None else reason
        row.update(
            target_agent_id=-1,
            target_slot=-1,
            parent_agent_id=-1,
            child_agent_id=-1,
            reason_code=failure_reason,
            success=0,
            value=float(failure_reason),
        )
    elif kind == 4:
        death_reason = 1 if reason is None else reason
        row.update(
            target_agent_id=-1,
            target_slot=-1,
            parent_agent_id=-1,
            child_agent_id=-1,
            reason_code=death_reason,
            value=float(death_reason),
        )
    return row


def _reason_tick(**counts: int) -> list[int]:
    row = [0] * REASON_COUNT
    for reason, count in counts.items():
        row[int(reason.removeprefix("r"))] = count
    return row


def _resident_states() -> list[dict[str, Any]]:
    g0_cell0 = [
        _event(1, 0, generation=6, tick=0),
        _event(1, 1, generation=7, tick=1),
        _event(4, 2, generation=7, tick=2, reason=1),
        _event(3, 3, generation=7, tick=3, reason=1),
        _event(2, 4, generation=7, tick=4),
        _event(1, 5, generation=7, tick=4),
    ]
    g0_cell1 = [
        _event(3, 0, generation=5, tick=0, reason=2),
        _event(1, 1, generation=6, tick=0),
        _event(2, 2, generation=7, tick=1),
        _event(1, 3, generation=7, tick=1),
        _event(4, 4, generation=7, tick=2, reason=2),
        _event(3, 5, generation=7, tick=3, reason=3),
    ]

    g1_cell0 = copy.deepcopy(g0_cell0)
    g1_cell0.extend(
        [
            _event(2, 6, generation=8, tick=1),
            _event(1, 7, generation=8, tick=1),
            _event(3, 8, generation=8, tick=2, reason=2),
            _event(4, 9, generation=8, tick=3, reason=1),
        ]
    )
    g1_cell1 = copy.deepcopy(g0_cell1)
    g1_cell1.extend(
        [
            _event(1, 6, generation=8, tick=1),
            _event(3, 7, generation=8, tick=2, reason=3),
            _event(1, 8, generation=8, tick=3),
            _event(4, 9, generation=8, tick=3, reason=2),
        ]
    )

    return [
        {
            "generation_index": 0,
            "generation": 7,
            "event_capacity": 10,
            "slot_capacity": 4,
            "events": [g0_cell0, g0_cell1],
            "cursor_start": [1, 2],
            "cursor_pre_turnover": [4, 5],
            "cursor_end": [6, 6],
            "summary_counts": [(1, 1, 1), (1, 1, 0)],
            "birth_decision_count": [[2, 2], [1, 2]],
            "birth_reason_counts": [
                [_reason_tick(r0=1, r1=1), _reason_tick(r5=1, r8=1)],
                [_reason_tick(r0=1), _reason_tick(r5=1, r6=1)],
            ],
            "active_agent_ids": [[2, 7, -1, -1], [4, 9, 12, -1]],
            "identity": {
                "active_count_state": [2, 3],
                "population_next_id": [8, 13],
                "slot_free_queue_count": [2, 1],
            },
        },
        {
            "generation_index": 1,
            "generation": 8,
            "event_capacity": 10,
            "slot_capacity": 4,
            "events": [g1_cell0, g1_cell1],
            "cursor_start": [6, 6],
            "cursor_pre_turnover": [9, 8],
            "cursor_end": [10, 10],
            "summary_counts": [(1, 0, 1), (1, 0, 1)],
            "birth_decision_count": [[2, 1], [2, 1]],
            "birth_reason_counts": [
                [_reason_tick(r0=1, r2=1), _reason_tick(r9=1)],
                [_reason_tick(r0=1, r3=1), _reason_tick(r7=1)],
            ],
            "active_agent_ids": [[2, 8, 13, -1], [-1, -1, -1, -1]],
            "identity": {
                "active_count_state": [3, 0],
                "population_next_id": [14, 15],
                "slot_free_queue_count": [1, 4],
            },
        },
    ]


def _summary_rows(states: list[dict[str, Any]]) -> list[list[list[float]]]:
    summary = [
        [[0.0] * SUMMARY_WIDTH for _ in state["events"]]
        for state in states
    ]
    for state in states:
        generation_index = state["generation_index"]
        for cell, (births, deaths, failed) in enumerate(state["summary_counts"]):
            row = summary[generation_index][cell]
            row[0] = float(state["generation"])
            row[1] = float(state["cursor_start"][cell])
            row[2] = float(state["cursor_pre_turnover"][cell])
            row[3] = float(state["cursor_end"][cell])
            row[4] = float(state["identity"]["active_count_state"][cell])
            row[5] = 12.5 + cell
            row[6] = float(births)
            row[7] = float(deaths)
            row[8] = float(failed)
            row[17] = 0.25 + cell
            row[28] = 100.0 + generation_index + cell
            row[29] = 25.0 + generation_index + cell
            row[30] = 10.0 + generation_index + cell
    return summary


def _event_tensors(
    rows_by_cell: list[list[dict[str, Any]]],
    capacity: int,
    device: torch.device,
) -> list[torch.Tensor]:
    tensors = [
        torch.zeros((len(rows_by_cell), capacity), dtype=dtype, device=device)
        for dtype in EVENT_DTYPES
    ]
    for cell, rows in enumerate(rows_by_cell):
        for row_index, row in enumerate(rows):
            assert set(row) == set(EVENT_FIELDS)
            for field_index, field in enumerate(EVENT_FIELDS):
                tensors[field_index][cell, row_index] = row[field]
    return tensors


def _device_arguments(
    state: dict[str, Any],
    summary: list[list[list[float]]],
    device: torch.device,
) -> tuple[Any, ...]:
    typed = _event_tensors(state["events"], state["event_capacity"], device)
    active_event_count = torch.tensor(
        state["cursor_end"], dtype=torch.int64, device=device
    )
    summary_tensor = torch.tensor(summary, dtype=torch.float64, device=device)
    decision_count = torch.tensor(
        state["birth_decision_count"], dtype=torch.int64, device=device
    )
    reason_counts = torch.tensor(
        state["birth_reason_counts"], dtype=torch.int64, device=device
    )
    active_ids = torch.tensor(
        state["active_agent_ids"], dtype=torch.int64, device=device
    )
    identity = [
        torch.tensor(state["identity"][field], dtype=torch.int64, device=device)
        for field in (
            "active_count_state",
            "population_next_id",
            "slot_free_queue_count",
        )
    ]
    return (
        typed,
        active_event_count,
        summary_tensor,
        decision_count,
        reason_counts,
        active_ids,
        identity,
        state["slot_capacity"],
        state["generation_index"],
    )


def _decode_row(raw: torch.Tensor) -> dict[str, Any]:
    row = ReductionResultV1.from_buffer_copy(
        raw.detach().to("cpu").contiguous().numpy().tobytes()
    )
    return {
        "status": int(row.status),
        "valid": bool(row.valid),
        "active_id_bounds_defined": bool(row.active_id_bounds_defined),
        "generation_event_count": int(row.generation_event_count),
        "turnover_event_count": int(row.turnover_event_count),
        "generation_counts_by_type": tuple(row.generation_counts_by_type),
        "turnover_counts_by_type": tuple(row.turnover_counts_by_type),
        "generation_failed_birth_counts_by_reason": tuple(
            row.generation_failed_birth_counts_by_reason
        ),
        "turnover_failed_birth_counts_by_reason": tuple(
            row.turnover_failed_birth_counts_by_reason
        ),
        "birth_failure_counts_by_reason": tuple(
            row.birth_failure_counts_by_reason
        ),
        "native_failed_birth_event_rows": int(row.native_failed_birth_event_rows),
        "injury_blocked_births": int(row.injury_blocked_births),
        "birth_attempt_rows": int(row.birth_attempt_rows),
        "birth_failure_rows": int(row.birth_failure_rows),
        "deaths_energy_depleted": int(row.deaths_energy_depleted),
        "deaths_health_depleted": int(row.deaths_health_depleted),
        "cursor_start": int(row.cursor_start),
        "cursor_pre_turnover": int(row.cursor_pre_turnover),
        "cursor_end": int(row.cursor_end),
        "authoritative_cursor": int(row.authoritative_cursor),
        "event_row_count": int(row.event_row_count),
        "active_event_count": int(row.active_event_count),
        "population_next_id": int(row.population_next_id),
        "active_agent_count": int(row.active_agent_count),
        "active_agent_id_min": int(row.active_agent_id_min),
        "active_agent_id_max": int(row.active_agent_id_max),
        "free_slot_count": int(row.free_slot_count),
    }


def test_framework_free_generation_major_abi_and_no_hidden_runtime_work(
    tmp_path: Path,
) -> None:
    header = HEADER.read_text(encoding="utf-8")
    cuda = CUDA_SOURCE.read_text(encoding="utf-8")
    assert "#include <cuda_runtime" not in header
    assert "struct GenerationSummaryScalarViewV1" in header
    assert "struct OrdinaryBirthMetricsSourcesV1" in header
    assert "struct ReductionOutputViewV1" in header
    assert "generation_major_row_index_v1" in cuda
    assert "sizeof(ReductionResultV1) == 456" in header
    assert "kBirthPlanUnusedReasonCodeV1 = 4" in header
    assert "result.injury_blocked_births = 0" in header
    assert "output.rows[output_row] = result;" in cuda
    assert cuda.index("output.rows[output_row] = result;") < cuda.index(
        "output.rows[output_row].valid = 1;"
    )
    assert re.search(
        r"\b(?:ATen|Torch|torch|malloc|calloc|cudaMalloc|cudaFree|new)\b",
        header + cuda,
    ) is None
    for forbidden in (
        "cudaDeviceSynchronize",
        "cudaStreamSynchronize",
        "cudaMemcpy",
    ):
        assert forbidden not in header + cuda

    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("g++ is unavailable")
    source = tmp_path / "cpu_include.cpp"
    source.write_text(
        f'#include "{HEADER}"\n'
        "static_assert(sizeof(eiel::cuda_architecture_b::"
        "compact_summary_reductions_v1::ReductionResultV1) == 456);\n"
        "int main() { return 0; }\n",
        encoding="ascii",
    )
    completed = subprocess.run(
        [compiler, "-std=c++17", "-fsyntax-only", str(source)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_nvcc_instantiates_resident_source_contract_for_sm75(tmp_path: Path) -> None:
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        pytest.skip("nvcc is unavailable")
    output = tmp_path / "compact_summary_reductions_v1.o"
    completed = subprocess.run(
        [
            nvcc,
            "-std=c++17",
            "-O2",
            "-lineinfo",
            "-Xptxas=-v",
            "-gencode=arch=compute_75,code=sm_75",
            "-c",
            str(COMPILE_TEST),
            "-o",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert output.is_file()
    assert "ptxas info" in completed.stderr


@pytest.fixture(scope="module")
def accepted_build(tmp_path_factory: pytest.TempPathFactory) -> ModuleType:
    if shutil.which("nvcc") is None or not torch.cuda.is_available():
        pytest.skip("CUDA compiler and device are required")
    from torch.utils.cpp_extension import load

    build_dir = tmp_path_factory.mktemp("compact_reductions_v1")
    previous_arch = os.environ.get("TORCH_CUDA_ARCH_LIST")
    os.environ["TORCH_CUDA_ARCH_LIST"] = "7.5"
    try:
        return load(
            name="eiel_cuda_architecture_b_compact_summary_reductions_v1_test",
            sources=[str(BINDING_SOURCE), str(CUDA_SOURCE)],
            extra_include_paths=[str(ROOT / "engine/perf")],
            extra_cflags=["-O2", "-ffp-contract=off"],
            extra_cuda_cflags=["-O2", "--fmad=false", "-lineinfo"],
            build_directory=str(build_dir),
            with_cuda=True,
            verbose=False,
        )
    finally:
        if previous_arch is None:
            os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
        else:
            os.environ["TORCH_CUDA_ARCH_LIST"] = previous_arch


def _launch(
    module: ModuleType,
    state: dict[str, Any],
    summary: list[list[list[float]]],
    output: torch.Tensor,
) -> None:
    module.reduce_into(*_device_arguments(state, summary, output.device), output)


def test_g0_g1_retention_prefix_counts_metrics_identity_and_free_slots(
    accepted_build: ModuleType,
) -> None:
    states = _resident_states()
    summary = _summary_rows(states)
    device = torch.device("cuda")
    output = torch.full(
        (2, 2, RESULT_BYTES), 0xA5, dtype=torch.uint8, device=device
    )

    _launch(accepted_build, states[0], summary, output)
    torch.cuda.synchronize(device)
    g0_bytes = output[0].detach().cpu().clone()
    assert torch.all(output[1] == 0xA5)

    _launch(accepted_build, states[1], summary, output)
    torch.cuda.synchronize(device)
    assert torch.equal(output[0].cpu(), g0_bytes)

    g0 = [_decode_row(output[0, cell]) for cell in range(2)]
    g1 = [_decode_row(output[1, cell]) for cell in range(2)]
    assert all(row["valid"] and row["status"] == 0 for row in g0 + g1)
    assert [row["cursor_start"] for row in g0 + g1] == [1, 2, 6, 6]
    assert g0[0]["generation_counts_by_type"] == (1, 0, 1, 1)
    assert g0[0]["turnover_counts_by_type"] == (1, 1, 0, 0)
    assert g0[1]["generation_counts_by_type"] == (1, 1, 0, 1)
    assert g0[1]["turnover_counts_by_type"] == (0, 0, 1, 0)
    assert g1[0]["generation_counts_by_type"] == (1, 1, 1, 0)
    assert g1[0]["turnover_counts_by_type"] == (0, 0, 0, 1)
    assert g1[1]["generation_counts_by_type"] == (1, 0, 1, 0)
    assert g1[1]["turnover_counts_by_type"] == (1, 0, 0, 1)

    assert g0[0]["native_failed_birth_event_rows"] == 1
    assert g0[0]["generation_failed_birth_counts_by_reason"][0] == 1
    assert g0[1]["turnover_failed_birth_counts_by_reason"][2] == 1
    assert g0[0]["birth_attempt_rows"] == 2
    assert g0[0]["birth_failure_rows"] == 2
    assert g0[0]["birth_failure_counts_by_reason"] == (
        1,
        0,
        0,
        0,
        1,
        0,
        0,
        1,
        0,
    )
    assert g1[1]["birth_failure_counts_by_reason"][6] == 1
    assert all(row["injury_blocked_births"] == 0 for row in g0 + g1)

    assert (g0[0]["active_agent_id_min"], g0[0]["active_agent_id_max"]) == (
        2,
        7,
    )
    assert g0[0]["free_slot_count"] == 2
    assert g1[0]["active_agent_count"] + g1[0]["free_slot_count"] == 4
    assert g1[1]["active_id_bounds_defined"] is False
    assert g1[1]["free_slot_count"] == 4


Mutation = Callable[[dict[str, Any], list[list[list[float]]]], None]


@pytest.mark.parametrize(
    ("mutation", "status"),
    [
        (lambda state, summary: summary[0][0].__setitem__(5, float("nan")), 2),
        (lambda state, summary: summary[0][0].__setitem__(1, 1.5), 2),
        (lambda state, summary: state["cursor_end"].__setitem__(0, 5), 2),
        (
            lambda state, summary: state["events"][0][2].__setitem__(
                "reason_code", 9
            ),
            3,
        ),
        (
            lambda state, summary: state["events"][0][1].__setitem__(
                "generation", 99
            ),
            3,
        ),
        (
            lambda state, summary: state["active_agent_ids"][0].__setitem__(1, 2),
            4,
        ),
        (
            lambda state, summary: state["identity"]["population_next_id"].__setitem__(
                0, 7
            ),
            4,
        ),
        (
            lambda state, summary: state["identity"][
                "slot_free_queue_count"
            ].__setitem__(0, 1),
            5,
        ),
        (
            lambda state, summary: state["birth_reason_counts"][0][0].__setitem__(
                1, -1
            ),
            6,
        ),
        (
            lambda state, summary: state["birth_decision_count"][0].__setitem__(
                0, 3
            ),
            6,
        ),
        (
            lambda state, summary: (
                state["birth_reason_counts"][0][0].__setitem__(4, 1),
                state["birth_decision_count"][0].__setitem__(0, 3),
            ),
            6,
        ),
        (
            lambda state, summary: (
                state["birth_reason_counts"][0][0].__setitem__(1, 0),
                state["birth_reason_counts"][0][0].__setitem__(2, 1),
            ),
            3,
        ),
    ],
)
def test_malformed_resident_sources_fail_closed_per_cell(
    accepted_build: ModuleType,
    mutation: Mutation,
    status: int,
) -> None:
    states = _resident_states()
    state = copy.deepcopy(states[0])
    summary = _summary_rows(states)
    mutation(state, summary)
    output = torch.zeros((2, 2, RESULT_BYTES), dtype=torch.uint8, device="cuda")
    _launch(accepted_build, state, summary, output)
    torch.cuda.synchronize()
    first = _decode_row(output[0, 0])
    second = _decode_row(output[0, 1])
    assert first["valid"] is False
    assert first["status"] == status
    assert first["generation_event_count"] == 0
    assert second["valid"] is True


def test_invalid_generation_and_tensor_shapes_fail_before_launch(
    accepted_build: ModuleType,
) -> None:
    states = _resident_states()
    summary = _summary_rows(states)
    state = states[0]
    arguments = list(_device_arguments(state, summary, torch.device("cuda")))
    output = torch.full(
        (2, 2, RESULT_BYTES), 0xA5, dtype=torch.uint8, device="cuda"
    )

    invalid_generation = arguments.copy()
    invalid_generation[-1] = -1
    with pytest.raises(RuntimeError, match="generation_index"):
        accepted_build.reduce_into(*invalid_generation, output)
    assert torch.all(output == 0xA5)

    invalid_summary = arguments.copy()
    invalid_summary[2] = invalid_summary[2][:, :, :-1].contiguous()
    with pytest.raises(RuntimeError, match="generations,cells,31"):
        accepted_build.reduce_into(*invalid_summary, output)

    invalid_reasons = arguments.copy()
    invalid_reasons[4] = invalid_reasons[4][:, :, :-1].contiguous()
    with pytest.raises(RuntimeError, match="cells,ticks,10"):
        accepted_build.reduce_into(*invalid_reasons, output)

    invalid_output = output[:1].contiguous()
    with pytest.raises(RuntimeError, match="generations,cells,456"):
        accepted_build.reduce_into(*arguments, invalid_output)


def test_mismatched_typed_column_capacity_fails_in_binding(
    accepted_build: ModuleType,
) -> None:
    states = _resident_states()
    arguments = list(
        _device_arguments(states[0], _summary_rows(states), torch.device("cuda"))
    )
    arguments[0][1] = arguments[0][1][:, :-1].contiguous()
    output = torch.zeros((2, 2, RESULT_BYTES), dtype=torch.uint8, device="cuda")
    with pytest.raises(RuntimeError, match="share exact"):
        accepted_build.reduce_into(*arguments, output)


def test_resource_receipt_is_bounded(accepted_build: ModuleType) -> None:
    assert accepted_build.contract_version.endswith("reductions_v1")
    assert accepted_build.result_bytes == RESULT_BYTES
    resources = dict(accepted_build.resource_receipt())
    assert resources["binary_version"] == 75
    assert resources["registers_per_thread"] > 0
    assert resources["local_bytes_per_thread"] >= 0
    assert resources["shared_bytes_per_block"] == 0
