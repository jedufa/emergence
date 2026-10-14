from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import pytest
import torch

from perf.cuda_architecture_b_death_lifecycle_events import (
    CUDA_DEATH_LIFECYCLE_EVENT_CAPABILITIES,
    EVENT_TAPE_COLUMN_NAMES,
    EVENT_TAPE_DTYPES,
    EVENT_TAPE_STATUS_CAPACITY_EXCEEDED,
    EVENT_TAPE_STATUS_INVALID_SEMANTICS,
    EVENT_TAPE_STATUS_OK,
    LIFECYCLE_EVENTS_SOURCE,
    append_death_lifecycle_events_cuda,
    load_cuda_death_lifecycle_events_extension,
)


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


@pytest.fixture(scope="module")
def cuda_build_directory(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("archb-death-lifecycle-events-build")


@pytest.fixture(scope="module")
def cuda_module(cuda_build_directory: Path) -> Any:
    return load_cuda_death_lifecycle_events_extension(
        build_directory=cuda_build_directory
    )


def _empty_tape(capacity: int, *, active_count: int = 0) -> dict[str, Any]:
    return {
        "tape_columns": [
            torch.full((capacity,), -77, device="cuda", dtype=dtype)
            for dtype in EVENT_TAPE_DTYPES
        ],
        "active_count": torch.tensor(
            [active_count], device="cuda", dtype=torch.int64
        ),
        "status": torch.full((1,), -11, device="cuda", dtype=torch.int32),
    }


def _fixed_vector(
    values: list[int],
    *,
    capacity: int,
    dtype: torch.dtype,
    fill: int,
) -> torch.Tensor:
    tensor = torch.full((capacity,), fill, device="cuda", dtype=dtype)
    if values:
        tensor[: len(values)] = torch.tensor(values, device="cuda", dtype=dtype)
    return tensor


def _inputs(
    *,
    agent_ids: list[int],
    slots: list[int],
    reasons: list[int],
    capacity: int = 5,
) -> dict[str, torch.Tensor]:
    assert len(agent_ids) == len(slots) == len(reasons)
    assert capacity >= len(agent_ids)
    lineage_founder_id = torch.full((10,), -1, device="cuda", dtype=torch.int32)
    lineage_depth = torch.full((10,), -1, device="cuda", dtype=torch.int32)
    for slot, founder, depth in (
        (1, 10, 0),
        (4, 5, 3),
        (7, 30, 1),
    ):
        lineage_founder_id[slot] = founder
        lineage_depth[slot] = depth
    return {
        "death_agent_ids": _fixed_vector(
            agent_ids,
            capacity=capacity,
            dtype=torch.int64,
            fill=-901,
        ),
        "death_slots": _fixed_vector(
            slots,
            capacity=capacity,
            dtype=torch.int64,
            fill=999,
        ),
        "reason_codes": _fixed_vector(
            reasons,
            capacity=capacity,
            dtype=torch.int32,
            fill=88,
        ),
        "death_count": torch.tensor(
            [len(agent_ids)], device="cuda", dtype=torch.int64
        ),
        "generation": torch.tensor([6], device="cuda", dtype=torch.int64),
        "tick": torch.tensor([41], device="cuda", dtype=torch.int64),
        "lineage_founder_id": lineage_founder_id,
        "lineage_depth": lineage_depth,
    }


def _cpu_owner_oracle(
    inputs: dict[str, torch.Tensor],
    *,
    sequence_start: int,
) -> dict[str, list[int | float]]:
    cpu = {name: tensor.cpu().tolist() for name, tensor in inputs.items()}
    count = cpu["death_count"][0]
    agent_ids = cpu["death_agent_ids"][:count]
    slots = cpu["death_slots"][:count]
    reasons = cpu["reason_codes"][:count]
    generation = cpu["generation"][0]
    tick = cpu["tick"][0]
    sentinels = [-1] * count
    event_types = [4] * count
    values = [float(reason) for reason in reasons]
    return {
        "event_tape_type": event_types,
        "event_tape_generation": [generation] * count,
        "event_tape_tick": [tick] * count,
        "event_tape_actor_id": agent_ids,
        "event_tape_target_id": sentinels,
        "event_tape_value": values,
        "typed_lifecycle_event_event_type": event_types,
        "typed_lifecycle_event_generation": [generation] * count,
        "typed_lifecycle_event_tick": [tick] * count,
        "typed_lifecycle_event_sequence": list(
            range(sequence_start, sequence_start + count)
        ),
        "typed_lifecycle_event_actor_agent_id": agent_ids,
        "typed_lifecycle_event_target_agent_id": sentinels,
        "typed_lifecycle_event_actor_slot": slots,
        "typed_lifecycle_event_target_slot": sentinels,
        "typed_lifecycle_event_parent_agent_id": sentinels,
        "typed_lifecycle_event_child_agent_id": sentinels,
        "typed_lifecycle_event_lineage_founder_id": [
            cpu["lineage_founder_id"][slot] for slot in slots
        ],
        "typed_lifecycle_event_lineage_depth": [
            cpu["lineage_depth"][slot] for slot in slots
        ],
        "typed_lifecycle_event_subtype_code": event_types,
        "typed_lifecycle_event_reason_code": reasons,
        "typed_lifecycle_event_success": [1] * count,
        "typed_lifecycle_event_value": values,
    }


def _assert_matches_oracle(
    tape: dict[str, Any],
    inputs: dict[str, torch.Tensor],
    *,
    initial_count: int,
) -> None:
    expected = _cpu_owner_oracle(inputs, sequence_start=initial_count)
    count = len(expected["event_tape_type"])
    for name, column in zip(
        EVENT_TAPE_COLUMN_NAMES,
        tape["tape_columns"],
        strict=True,
    ):
        actual = column[initial_count : initial_count + count].cpu().tolist()
        assert actual == expected[name], name


def _assert_tape_unchanged(
    tape: dict[str, Any],
    before: list[torch.Tensor],
) -> None:
    for actual, expected in zip(tape["tape_columns"], before, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_zero_deaths_is_successful_noop(
    cuda_module: Any,
    cuda_build_directory: Path,
) -> None:
    del cuda_module
    tape = _empty_tape(6, active_count=2)
    inputs = _inputs(agent_ids=[], slots=[], reasons=[])
    before = [column.clone() for column in tape["tape_columns"]]

    result = append_death_lifecycle_events_cuda(
        **tape,
        **inputs,
        build_directory=cuda_build_directory,
    )
    torch.cuda.synchronize()

    assert int(result.status.cpu()[0]) == EVENT_TAPE_STATUS_OK
    assert int(result.active_count.cpu()[0]) == 2


@pytest.mark.parametrize(
    "prior_status",
    [EVENT_TAPE_STATUS_CAPACITY_EXCEEDED, EVENT_TAPE_STATUS_INVALID_SEMANTICS],
)
def test_prior_event_failure_is_sticky_and_prevents_publication(
    prior_status: int,
    cuda_module: Any,
    cuda_build_directory: Path,
) -> None:
    del cuda_module
    tape = _empty_tape(6, active_count=2)
    tape["status"].fill_(prior_status)
    before = [column.clone() for column in tape["tape_columns"]]
    inputs = _inputs(agent_ids=[20], slots=[4], reasons=[2], capacity=3)

    result = append_death_lifecycle_events_cuda(
        **tape,
        **inputs,
        build_directory=cuda_build_directory,
    )
    torch.cuda.synchronize()

    assert int(result.active_count.cpu()[0]) == 2
    assert int(result.status.cpu()[0]) == prior_status
    _assert_tape_unchanged(tape, before)


def test_one_death_populates_every_cpu_owner_column(
    cuda_module: Any,
    cuda_build_directory: Path,
) -> None:
    del cuda_module
    tape = _empty_tape(8, active_count=3)
    inputs = _inputs(agent_ids=[20], slots=[4], reasons=[2])
    pointers = [column.data_ptr() for column in tape["tape_columns"]]

    result = append_death_lifecycle_events_cuda(
        **tape,
        **inputs,
        build_directory=cuda_build_directory,
    )
    torch.cuda.synchronize()

    assert int(result.status.cpu()[0]) == EVENT_TAPE_STATUS_OK
    assert int(result.active_count.cpu()[0]) == 4
    assert pointers == [column.data_ptr() for column in tape["tape_columns"]]
    _assert_matches_oracle(tape, inputs, initial_count=3)


def test_multiple_deaths_preserve_dense_order_and_ignore_inactive_tails(
    cuda_module: Any,
    cuda_build_directory: Path,
) -> None:
    del cuda_module
    tape = _empty_tape(9, active_count=1)
    inputs = _inputs(
        agent_ids=[30, 10, 20],
        slots=[7, 1, 4],
        reasons=[1, 2, 1],
        capacity=6,
    )
    before_tails = [column[4:].clone() for column in tape["tape_columns"]]

    result = append_death_lifecycle_events_cuda(
        **tape,
        **inputs,
        build_directory=cuda_build_directory,
    )
    torch.cuda.synchronize()

    assert int(result.status.cpu()[0]) == EVENT_TAPE_STATUS_OK
    assert int(result.active_count.cpu()[0]) == 4
    _assert_matches_oracle(tape, inputs, initial_count=1)
    for column, expected_tail in zip(
        tape["tape_columns"],
        before_tails,
        strict=True,
    ):
        torch.testing.assert_close(column[4:], expected_tail, rtol=0, atol=0)


def test_capacity_failure_rolls_back_all_columns_and_cursor(
    cuda_module: Any,
    cuda_build_directory: Path,
) -> None:
    del cuda_module
    tape = _empty_tape(3, active_count=2)
    inputs = _inputs(agent_ids=[10, 20], slots=[1, 4], reasons=[1, 2])
    before = [column.clone() for column in tape["tape_columns"]]

    result = append_death_lifecycle_events_cuda(
        **tape,
        **inputs,
        build_directory=cuda_build_directory,
    )
    torch.cuda.synchronize()

    assert int(result.status.cpu()[0]) == EVENT_TAPE_STATUS_CAPACITY_EXCEEDED
    assert int(result.active_count.cpu()[0]) == 2
    _assert_tape_unchanged(tape, before)


InvalidMutation = Callable[[dict[str, torch.Tensor]], None]


def _invalid_count(inputs: dict[str, torch.Tensor]) -> None:
    inputs["death_count"].fill_(inputs["death_agent_ids"].numel() + 1)


def _negative_generation(inputs: dict[str, torch.Tensor]) -> None:
    inputs["generation"].fill_(-1)


def _negative_tick(inputs: dict[str, torch.Tensor]) -> None:
    inputs["tick"].fill_(-1)


def _negative_actor(inputs: dict[str, torch.Tensor]) -> None:
    inputs["death_agent_ids"][0] = -1


def _invalid_slot(inputs: dict[str, torch.Tensor]) -> None:
    inputs["death_slots"][0] = inputs["lineage_depth"].numel()


def _invalid_reason(inputs: dict[str, torch.Tensor]) -> None:
    inputs["reason_codes"][0] = 3


def _missing_founder(inputs: dict[str, torch.Tensor]) -> None:
    inputs["lineage_founder_id"][4] = -1


def _missing_depth(inputs: dict[str, torch.Tensor]) -> None:
    inputs["lineage_depth"][4] = -1


def _duplicate_actor(inputs: dict[str, torch.Tensor]) -> None:
    inputs["death_count"].fill_(2)
    inputs["death_agent_ids"][:2] = torch.tensor(
        [20, 20], device="cuda", dtype=torch.int64
    )
    inputs["death_slots"][:2] = torch.tensor(
        [4, 1], device="cuda", dtype=torch.int64
    )
    inputs["reason_codes"][:2] = torch.tensor(
        [1, 2], device="cuda", dtype=torch.int32
    )


@pytest.mark.parametrize(
    "mutate",
    [
        _invalid_count,
        _negative_generation,
        _negative_tick,
        _negative_actor,
        _invalid_slot,
        _invalid_reason,
        _missing_founder,
        _missing_depth,
        _duplicate_actor,
    ],
    ids=lambda mutate: mutate.__name__.removeprefix("_"),
)
def test_invalid_semantics_fail_before_any_tape_mutation(
    mutate: InvalidMutation,
    cuda_module: Any,
    cuda_build_directory: Path,
) -> None:
    del cuda_module
    tape = _empty_tape(7, active_count=2)
    inputs = _inputs(agent_ids=[20], slots=[4], reasons=[2])
    mutate(inputs)
    before = [column.clone() for column in tape["tape_columns"]]

    result = append_death_lifecycle_events_cuda(
        **tape,
        **inputs,
        build_directory=cuda_build_directory,
    )
    torch.cuda.synchronize()

    assert int(result.status.cpu()[0]) == EVENT_TAPE_STATUS_INVALID_SEMANTICS
    assert int(result.active_count.cpu()[0]) == 2
    _assert_tape_unchanged(tape, before)


def test_capability_receipt_is_bounded() -> None:
    for supported in (
        "dense_owner_order_preserved",
        "energy_and_health_death_reasons",
        "legacy_and_16_typed_columns",
        "resident_cursor_derived_sequence",
        "single_atomic_event_tape_append",
        "fixed_capacity_inputs",
        "device_resident_active_counts",
        "device_side_preflight",
        "composable_pybind_guard",
    ):
        assert CUDA_DEATH_LIFECYCLE_EVENT_CAPABILITIES[supported] is True
    assert (
        CUDA_DEATH_LIFECYCLE_EVENT_CAPABILITIES["capacity_failure"]
        == "atomic_fail_closed"
    )
    assert (
        CUDA_DEATH_LIFECYCLE_EVENT_CAPABILITIES["invalid_semantics"]
        == "asynchronous_atomic_fail_closed"
    )
    assert (
        CUDA_DEATH_LIFECYCLE_EVENT_CAPABILITIES[
            "host_synchronization_during_publication"
        ]
        is False
    )
    assert CUDA_DEATH_LIFECYCLE_EVENT_CAPABILITIES["data_dependent_host_shapes"] is False
    for unsupported in (
        "generation_integration",
        "birth_lifecycle",
        "transaction_integration",
        "promotion_qualified",
    ):
        assert CUDA_DEATH_LIFECYCLE_EVENT_CAPABILITIES[unsupported] is False


def test_sources_forbid_host_sync_and_dynamic_shape_primitives() -> None:
    native_source = LIFECYCLE_EVENTS_SOURCE.read_text(encoding="utf-8")
    wrapper_source = Path(
        append_death_lifecycle_events_cuda.__code__.co_filename
    ).read_text(encoding="utf-8")
    for forbidden in (
        ".item",
        "torch::equal",
        "masked_select",
        "nonzero",
        "cudaDeviceSynchronize",
        "cudaStreamSynchronize",
        "cudaMemcpy",
    ):
        assert forbidden not in native_source
    for forbidden in (
        ".item(",
        ".cpu(",
        ".tolist(",
        "torch.cuda.synchronize",
    ):
        assert forbidden not in wrapper_source
    assert "preflight_death_lifecycle_events_kernel<<<1, 1" in native_source
    assert "append_death_lifecycle_events_kernel<<<1, 1" in native_source
    assert "#ifndef EIEL_CUDA_ARCHITECTURE_B_NO_PYBIND" in native_source
    assert "PYBIND11_MODULE(TORCH_EXTENSION_NAME, module)" in native_source
