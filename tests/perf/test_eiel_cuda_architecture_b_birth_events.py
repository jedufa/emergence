from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch

from perf.cuda_architecture_b_birth_events import (
    CUDA_BIRTH_EVENT_CAPABILITIES,
    EVENT_TAPE_COLUMN_NAMES,
    EVENT_TAPE_DTYPES,
    EVENT_TAPE_STATUS_CAPACITY_EXCEEDED,
    EVENT_TAPE_STATUS_OK,
    append_birth_events_cuda,
    load_cuda_birth_events_extension,
)


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


@pytest.fixture(scope="module")
def cuda_build_directory(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("archb-birth-events-build")


@pytest.fixture(scope="module")
def cuda_module(cuda_build_directory: Path) -> Any:
    return load_cuda_birth_events_extension(build_directory=cuda_build_directory)


def _empty_tape(capacity: int, *, active_count: int = 0) -> dict[str, Any]:
    columns = [
        torch.full((capacity,), -77, device="cuda", dtype=dtype)
        for dtype in EVENT_TAPE_DTYPES
    ]
    return {
        "tape_columns": columns,
        "active_count": torch.tensor(
            [active_count], device="cuda", dtype=torch.int64
        ),
        "status": torch.full((1,), -1, device="cuda", dtype=torch.int32),
    }


def _birth_inputs() -> dict[str, torch.Tensor]:
    return {
        "parent_agent_ids": torch.tensor(
            [10, 20, 30], device="cuda", dtype=torch.int64
        ),
        "child_agent_ids": torch.tensor(
            [101, 102, 103], device="cuda", dtype=torch.int64
        ),
        "parent_slots": torch.tensor([1, 4, 7], device="cuda", dtype=torch.int64),
        "child_slots": torch.tensor([8, 9, 11], device="cuda", dtype=torch.int64),
        "lineage_founder_ids": torch.tensor(
            [10, 5, 30], device="cuda", dtype=torch.int64
        ),
        "lineage_depths": torch.tensor([1, 4, 2], device="cuda", dtype=torch.int32),
        "generation": torch.tensor([6], device="cuda", dtype=torch.int64),
        "tick": torch.tensor([40, 41, 42], device="cuda", dtype=torch.int64),
        "mutation_occurred": torch.tensor(
            [True, False, True], device="cuda", dtype=torch.bool
        ),
        "mutation_delta_total": torch.tensor(
            [0.125, 99.0, -0.25], device="cuda", dtype=torch.float32
        ),
        "child_energy": torch.tensor(
            [4.5, 5.5, 6.5], device="cuda", dtype=torch.float32
        ),
    }


def _cpu_resident_owner_rows(sequence_start: int) -> dict[str, list[int | float]]:
    rows = [
        # mutation(child 101), birth(parent 10 -> child 101)
        (2, 6, 40, 101, 101, 8, 8, 10, 101, 10, 1, 2, 0.125),
        (1, 6, 40, 10, 101, 1, 8, 10, 101, 10, 1, 1, 4.5),
        # No mutation row for child 102.
        (1, 6, 41, 20, 102, 4, 9, 20, 102, 5, 4, 1, 5.5),
        # mutation(child 103), birth(parent 30 -> child 103)
        (2, 6, 42, 103, 103, 11, 11, 30, 103, 30, 2, 2, -0.25),
        (1, 6, 42, 30, 103, 7, 11, 30, 103, 30, 2, 1, 6.5),
    ]
    event_type = [row[0] for row in rows]
    generation = [row[1] for row in rows]
    tick = [row[2] for row in rows]
    actor = [row[3] for row in rows]
    target = [row[4] for row in rows]
    actor_slot = [row[5] for row in rows]
    target_slot = [row[6] for row in rows]
    parent = [row[7] for row in rows]
    child = [row[8] for row in rows]
    founder = [row[9] for row in rows]
    depth = [row[10] for row in rows]
    subtype = [row[11] for row in rows]
    value = [row[12] for row in rows]
    return {
        "event_tape_type": event_type,
        "event_tape_generation": generation,
        "event_tape_tick": tick,
        "event_tape_actor_id": actor,
        "event_tape_target_id": target,
        "event_tape_value": value,
        "typed_lifecycle_event_event_type": event_type,
        "typed_lifecycle_event_generation": generation,
        "typed_lifecycle_event_tick": tick,
        "typed_lifecycle_event_sequence": list(
            range(sequence_start, sequence_start + len(rows))
        ),
        "typed_lifecycle_event_actor_agent_id": actor,
        "typed_lifecycle_event_target_agent_id": target,
        "typed_lifecycle_event_actor_slot": actor_slot,
        "typed_lifecycle_event_target_slot": target_slot,
        "typed_lifecycle_event_parent_agent_id": parent,
        "typed_lifecycle_event_child_agent_id": child,
        "typed_lifecycle_event_lineage_founder_id": founder,
        "typed_lifecycle_event_lineage_depth": depth,
        "typed_lifecycle_event_subtype_code": subtype,
        "typed_lifecycle_event_reason_code": [-1] * len(rows),
        "typed_lifecycle_event_success": [1] * len(rows),
        "typed_lifecycle_event_value": value,
    }


def test_birth_events_match_cpu_resident_owner_semantics_exactly(
    cuda_module: Any,
    cuda_build_directory: Path,
) -> None:
    del cuda_module
    tape = _empty_tape(10, active_count=2)
    inputs = _birth_inputs()
    pointers = [column.data_ptr() for column in tape["tape_columns"]]

    result = append_birth_events_cuda(
        **tape,
        **inputs,
        build_directory=cuda_build_directory,
    )
    torch.cuda.synchronize()

    assert result.active_count.data_ptr() == tape["active_count"].data_ptr()
    assert result.status.data_ptr() == tape["status"].data_ptr()
    assert int(result.status.cpu()[0]) == EVENT_TAPE_STATUS_OK
    assert int(result.active_count.cpu()[0]) == 7
    assert pointers == [column.data_ptr() for column in tape["tape_columns"]]
    expected = _cpu_resident_owner_rows(sequence_start=2)
    for name, column in zip(
        EVENT_TAPE_COLUMN_NAMES,
        tape["tape_columns"],
        strict=True,
    ):
        assert column[2:7].cpu().tolist() == expected[name], name
        assert column[:2].cpu().tolist() == [-77, -77], name
        assert column[7:].cpu().tolist() == [-77, -77, -77], name


def test_split_calls_preserve_event_sequence_and_cursor_parity(
    cuda_module: Any,
    cuda_build_directory: Path,
) -> None:
    del cuda_module
    inputs = _birth_inputs()
    full = _empty_tape(8)
    split = _empty_tape(8)

    append_birth_events_cuda(
        **full,
        **inputs,
        build_directory=cuda_build_directory,
    )
    for start, stop in ((0, 1), (1, 3)):
        call = {
            name: tensor
            if name == "generation"
            else tensor[start:stop].contiguous()
            for name, tensor in inputs.items()
        }
        append_birth_events_cuda(
            **split,
            **call,
            build_directory=cuda_build_directory,
        )
    torch.cuda.synchronize()

    assert int(full["active_count"].cpu()[0]) == 5
    assert int(split["active_count"].cpu()[0]) == 5
    assert int(full["status"].cpu()[0]) == EVENT_TAPE_STATUS_OK
    assert int(split["status"].cpu()[0]) == EVENT_TAPE_STATUS_OK
    for full_column, split_column in zip(
        full["tape_columns"], split["tape_columns"], strict=True
    ):
        torch.testing.assert_close(full_column, split_column, rtol=0, atol=0)


def test_birth_event_capacity_failure_is_atomic_and_cursor_stable(
    cuda_module: Any,
    cuda_build_directory: Path,
) -> None:
    del cuda_module
    tape = _empty_tape(6, active_count=2)
    inputs = _birth_inputs()
    before = [column.clone() for column in tape["tape_columns"]]

    result = append_birth_events_cuda(
        **tape,
        **inputs,
        build_directory=cuda_build_directory,
    )
    torch.cuda.synchronize()

    assert int(result.status.cpu()[0]) == EVENT_TAPE_STATUS_CAPACITY_EXCEEDED
    assert int(result.active_count.cpu()[0]) == 2
    for actual, expected in zip(tape["tape_columns"], before, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_birth_event_capability_receipt_is_bounded() -> None:
    for supported in (
        "resident_semantic_inputs",
        "cpu_owner_mutation_then_birth_order",
        "mutation_event_omitted_when_absent",
        "legacy_and_16_typed_columns",
        "resident_cursor_derived_sequence",
        "single_atomic_event_tape_append",
    ):
        assert CUDA_BIRTH_EVENT_CAPABILITIES[supported] is True
    assert CUDA_BIRTH_EVENT_CAPABILITIES["capacity_failure"] == "atomic_fail_closed"
    for unsupported in (
        "generation_integration",
        "generation_turnover",
        "gru",
        "jungle",
        "wdl",
        "promotion_qualified",
    ):
        assert CUDA_BIRTH_EVENT_CAPABILITIES[unsupported] is False
