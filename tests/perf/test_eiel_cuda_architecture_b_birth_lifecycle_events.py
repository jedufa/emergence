from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch

from perf.cuda_architecture_b_birth_lifecycle_events import (
    CUDA_BIRTH_LIFECYCLE_EVENT_CAPABILITIES,
    EVENT_TAPE_COLUMN_NAMES,
    EVENT_TAPE_DTYPES,
    EVENT_TAPE_STATUS_CAPACITY_EXCEEDED,
    EVENT_TAPE_STATUS_INVALID_SEMANTICS,
    EVENT_TAPE_STATUS_OK,
    LIFECYCLE_EVENTS_SOURCE,
    append_birth_lifecycle_events_cuda,
    load_cuda_birth_lifecycle_events_extension,
)


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


@pytest.fixture(scope="module")
def cuda_build_directory(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("archb-birth-lifecycle-events-build")


@pytest.fixture(scope="module")
def cuda_module(cuda_build_directory: Path) -> Any:
    return load_cuda_birth_lifecycle_events_extension(
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
        "status": torch.full((1,), -1, device="cuda", dtype=torch.int32),
    }


def _resident_state() -> dict[str, torch.Tensor]:
    slot_to_agent_id = torch.full((12,), -1, device="cuda", dtype=torch.int64)
    lineage_founder_id = torch.full((12,), -1, device="cuda", dtype=torch.int32)
    lineage_depth = torch.full((12,), -1, device="cuda", dtype=torch.int32)
    for slot, agent_id, founder, depth in (
        (1, 10, 10, 0),
        (4, 20, 5, 3),
        (7, 30, 30, 1),
        (8, 101, 10, 1),
        (9, 102, 30, 2),
    ):
        slot_to_agent_id[slot] = agent_id
        lineage_founder_id[slot] = founder
        lineage_depth[slot] = depth
    return {
        "slot_to_agent_id": slot_to_agent_id,
        "lineage_founder_id": lineage_founder_id,
        "lineage_depth": lineage_depth,
    }


def _inputs(
    *,
    decision_parent_slots: list[int],
    decision_reason_codes: list[int],
    parent_agent_ids: list[int],
    child_agent_ids: list[int],
    parent_slots: list[int],
    child_slots: list[int],
    mutation_occurred: list[bool],
    mutation_delta_total: list[float],
    child_energy: list[float],
    decision_capacity: int = 6,
    success_capacity: int = 4,
) -> dict[str, torch.Tensor]:
    success_count = len(parent_agent_ids)
    decision_count = len(decision_parent_slots)
    assert decision_capacity >= decision_count > 0
    assert success_capacity >= success_count
    assert all(
        len(values) == success_count
        for values in (
            child_agent_ids,
            parent_slots,
            child_slots,
            mutation_occurred,
            mutation_delta_total,
            child_energy,
        )
    )
    def fixed_vector(
        values: list[int | float | bool],
        *,
        capacity: int,
        dtype: torch.dtype,
        fill: int | float | bool,
    ) -> torch.Tensor:
        tensor = torch.full((capacity,), fill, device="cuda", dtype=dtype)
        if values:
            tensor[: len(values)] = torch.tensor(values, device="cuda", dtype=dtype)
        return tensor

    return {
        **_resident_state(),
        "decision_parent_slots": fixed_vector(
            decision_parent_slots,
            capacity=decision_capacity,
            dtype=torch.int64,
            fill=-1,
        ),
        "decision_reason_codes": fixed_vector(
            decision_reason_codes,
            capacity=decision_capacity,
            dtype=torch.int32,
            fill=-1,
        ),
        "decision_count": torch.tensor(
            [decision_count], device="cuda", dtype=torch.int64
        ),
        "parent_agent_ids": fixed_vector(
            parent_agent_ids,
            capacity=success_capacity,
            dtype=torch.int64,
            fill=-1,
        ),
        "child_agent_ids": fixed_vector(
            child_agent_ids,
            capacity=success_capacity,
            dtype=torch.int64,
            fill=-1,
        ),
        "parent_slots": fixed_vector(
            parent_slots,
            capacity=success_capacity,
            dtype=torch.int64,
            fill=-1,
        ),
        "child_slots": fixed_vector(
            child_slots,
            capacity=success_capacity,
            dtype=torch.int64,
            fill=-1,
        ),
        "mutation_occurred": fixed_vector(
            mutation_occurred,
            capacity=success_capacity,
            dtype=torch.bool,
            fill=False,
        ),
        "mutation_delta_total": fixed_vector(
            mutation_delta_total,
            capacity=success_capacity,
            dtype=torch.float32,
            fill=float("nan"),
        ),
        "child_energy": fixed_vector(
            child_energy,
            capacity=success_capacity,
            dtype=torch.float32,
            fill=float("nan"),
        ),
        "successful_birth_count": torch.tensor(
            [success_count], device="cuda", dtype=torch.int64
        ),
        "generation": torch.tensor([6], device="cuda", dtype=torch.int64),
        "tick": torch.tensor([41], device="cuda", dtype=torch.int64),
    }


def _cpu_owner_oracle(
    inputs: dict[str, torch.Tensor],
    *,
    sequence_start: int,
) -> dict[str, list[int | float]]:
    cpu = {name: tensor.cpu().tolist() for name, tensor in inputs.items()}
    rows: list[dict[str, int | float]] = []
    success_ordinal = 0
    decision_count = cpu["decision_count"][0]
    for decision, reason in zip(
        cpu["decision_parent_slots"][:decision_count],
        cpu["decision_reason_codes"][:decision_count],
        strict=True,
    ):
        parent_id = cpu["slot_to_agent_id"][decision]
        if reason in (7, 8, 9):
            continue
        if reason != 0:
            founder = cpu["lineage_founder_id"][decision]
            rows.append(
                {
                    "type": 3,
                    "actor": parent_id,
                    "target": -1,
                    "actor_slot": decision,
                    "target_slot": -1,
                    "parent": -1,
                    "child": -1,
                    "founder": founder if founder >= 0 else parent_id,
                    "depth": cpu["lineage_depth"][decision],
                    "subtype": 3,
                    "reason": reason,
                    "success": 0,
                    "value": float(reason),
                }
            )
            continue

        child_id = cpu["child_agent_ids"][success_ordinal]
        child_slot = cpu["child_slots"][success_ordinal]
        common = {
            "target": child_id,
            "target_slot": child_slot,
            "parent": cpu["parent_agent_ids"][success_ordinal],
            "child": child_id,
            "founder": cpu["lineage_founder_id"][child_slot],
            "depth": cpu["lineage_depth"][child_slot],
            "reason": -1,
            "success": 1,
        }
        if cpu["mutation_occurred"][success_ordinal]:
            rows.append(
                {
                    **common,
                    "type": 2,
                    "actor": child_id,
                    "actor_slot": child_slot,
                    "subtype": 2,
                    "value": cpu["mutation_delta_total"][success_ordinal],
                }
            )
        rows.append(
            {
                **common,
                "type": 1,
                "actor": common["parent"],
                "actor_slot": cpu["parent_slots"][success_ordinal],
                "subtype": 1,
                "value": cpu["child_energy"][success_ordinal],
            }
        )
        success_ordinal += 1

    generation = cpu["generation"][0]
    tick = cpu["tick"][0]
    event_type = [row["type"] for row in rows]
    actor = [row["actor"] for row in rows]
    target = [row["target"] for row in rows]
    value = [row["value"] for row in rows]
    return {
        "event_tape_type": event_type,
        "event_tape_generation": [generation] * len(rows),
        "event_tape_tick": [tick] * len(rows),
        "event_tape_actor_id": actor,
        "event_tape_target_id": target,
        "event_tape_value": value,
        "typed_lifecycle_event_event_type": event_type,
        "typed_lifecycle_event_generation": [generation] * len(rows),
        "typed_lifecycle_event_tick": [tick] * len(rows),
        "typed_lifecycle_event_sequence": list(
            range(sequence_start, sequence_start + len(rows))
        ),
        "typed_lifecycle_event_actor_agent_id": actor,
        "typed_lifecycle_event_target_agent_id": target,
        "typed_lifecycle_event_actor_slot": [row["actor_slot"] for row in rows],
        "typed_lifecycle_event_target_slot": [row["target_slot"] for row in rows],
        "typed_lifecycle_event_parent_agent_id": [row["parent"] for row in rows],
        "typed_lifecycle_event_child_agent_id": [row["child"] for row in rows],
        "typed_lifecycle_event_lineage_founder_id": [
            row["founder"] for row in rows
        ],
        "typed_lifecycle_event_lineage_depth": [row["depth"] for row in rows],
        "typed_lifecycle_event_subtype_code": [row["subtype"] for row in rows],
        "typed_lifecycle_event_reason_code": [row["reason"] for row in rows],
        "typed_lifecycle_event_success": [row["success"] for row in rows],
        "typed_lifecycle_event_value": value,
    }


def _assert_matches_oracle(
    tape: dict[str, Any],
    inputs: dict[str, torch.Tensor],
    *,
    initial_count: int,
) -> None:
    expected = _cpu_owner_oracle(inputs, sequence_start=initial_count)
    event_count = len(expected["event_tape_type"])
    for name, column in zip(
        EVENT_TAPE_COLUMN_NAMES,
        tape["tape_columns"],
        strict=True,
    ):
        assert column[initial_count : initial_count + event_count].cpu().tolist() == expected[
            name
        ], name


def test_success_failure_success_matches_cpu_owner_interleaving(
    cuda_module: Any,
    cuda_build_directory: Path,
) -> None:
    del cuda_module
    tape = _empty_tape(10, active_count=2)
    inputs = _inputs(
        decision_parent_slots=[1, 4, 7],
        decision_reason_codes=[0, 2, 0],
        parent_agent_ids=[10, 30],
        child_agent_ids=[101, 102],
        parent_slots=[1, 7],
        child_slots=[8, 9],
        mutation_occurred=[True, False],
        mutation_delta_total=[0.125, 99.0],
        child_energy=[4.5, 6.5],
    )
    pointers = [column.data_ptr() for column in tape["tape_columns"]]

    result = append_birth_lifecycle_events_cuda(
        **tape,
        **inputs,
        build_directory=cuda_build_directory,
    )
    torch.cuda.synchronize()

    assert int(result.status.cpu()[0]) == EVENT_TAPE_STATUS_OK
    assert int(result.active_count.cpu()[0]) == 6
    assert pointers == [column.data_ptr() for column in tape["tape_columns"]]
    _assert_matches_oracle(tape, inputs, initial_count=2)


def test_failure_success_preserves_decision_order_and_skips_prefilters(
    cuda_module: Any,
    cuda_build_directory: Path,
) -> None:
    del cuda_module
    tape = _empty_tape(8)
    inputs = _inputs(
        decision_parent_slots=[4, 1, 7],
        decision_reason_codes=[1, 8, 0],
        parent_agent_ids=[30],
        child_agent_ids=[102],
        parent_slots=[7],
        child_slots=[9],
        mutation_occurred=[True],
        mutation_delta_total=[-0.25],
        child_energy=[6.5],
    )

    result = append_birth_lifecycle_events_cuda(
        **tape,
        **inputs,
        build_directory=cuda_build_directory,
    )
    torch.cuda.synchronize()

    assert int(result.status.cpu()[0]) == EVENT_TAPE_STATUS_OK
    assert int(result.active_count.cpu()[0]) == 3
    _assert_matches_oracle(tape, inputs, initial_count=0)


def test_capacity_failure_leaves_all_columns_and_cursor_unchanged(
    cuda_module: Any,
    cuda_build_directory: Path,
) -> None:
    del cuda_module
    tape = _empty_tape(5, active_count=2)
    inputs = _inputs(
        decision_parent_slots=[1, 4, 7],
        decision_reason_codes=[0, 2, 0],
        parent_agent_ids=[10, 30],
        child_agent_ids=[101, 102],
        parent_slots=[1, 7],
        child_slots=[8, 9],
        mutation_occurred=[True, False],
        mutation_delta_total=[0.125, 0.0],
        child_energy=[4.5, 6.5],
    )
    before = [column.clone() for column in tape["tape_columns"]]

    result = append_birth_lifecycle_events_cuda(
        **tape,
        **inputs,
        build_directory=cuda_build_directory,
    )
    torch.cuda.synchronize()

    assert int(result.status.cpu()[0]) == EVENT_TAPE_STATUS_CAPACITY_EXCEEDED
    assert int(result.active_count.cpu()[0]) == 2
    for actual, expected in zip(tape["tape_columns"], before, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_zero_success_capacity_and_prefilter_only_tick_is_a_noop(
    cuda_module: Any,
    cuda_build_directory: Path,
) -> None:
    del cuda_module
    tape = _empty_tape(4, active_count=1)
    inputs = _inputs(
        decision_parent_slots=[1, 4],
        decision_reason_codes=[8, 9],
        parent_agent_ids=[],
        child_agent_ids=[],
        parent_slots=[],
        child_slots=[],
        mutation_occurred=[],
        mutation_delta_total=[],
        child_energy=[],
        success_capacity=0,
    )
    before = [column.clone() for column in tape["tape_columns"]]

    result = append_birth_lifecycle_events_cuda(
        **tape,
        **inputs,
        build_directory=cuda_build_directory,
    )
    torch.cuda.synchronize()

    assert int(result.status.cpu()[0]) == EVENT_TAPE_STATUS_OK
    assert int(result.active_count.cpu()[0]) == 1
    for actual, expected in zip(tape["tape_columns"], before, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_misaligned_success_stream_fails_before_tape_mutation(
    cuda_module: Any,
    cuda_build_directory: Path,
) -> None:
    del cuda_module
    tape = _empty_tape(8, active_count=1)
    inputs = _inputs(
        decision_parent_slots=[1, 4],
        decision_reason_codes=[0, 2],
        parent_agent_ids=[10],
        child_agent_ids=[101],
        parent_slots=[4],
        child_slots=[8],
        mutation_occurred=[False],
        mutation_delta_total=[0.0],
        child_energy=[4.5],
    )
    before = [column.clone() for column in tape["tape_columns"]]

    result = append_birth_lifecycle_events_cuda(
        **tape,
        **inputs,
        build_directory=cuda_build_directory,
    )
    torch.cuda.synchronize()

    assert int(result.status.cpu()[0]) == EVENT_TAPE_STATUS_INVALID_SEMANTICS
    assert int(tape["active_count"].cpu()[0]) == 1
    for actual, expected in zip(tape["tape_columns"], before, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_capability_receipt_is_bounded() -> None:
    for supported in (
        "planner_decision_parent_order",
        "public_failure_reasons_1_to_3",
        "prefilter_reasons_7_to_9_omitted",
        "success_mutation_then_birth_order",
        "success_ordinal_alignment_validation",
        "legacy_and_16_typed_columns",
        "resident_cursor_derived_sequence",
        "single_atomic_event_tape_append",
        "fixed_capacity_inputs",
        "device_resident_active_counts",
        "device_side_preflight",
        "composable_pybind_guard",
    ):
        assert CUDA_BIRTH_LIFECYCLE_EVENT_CAPABILITIES[supported] is True
    assert (
        CUDA_BIRTH_LIFECYCLE_EVENT_CAPABILITIES["capacity_failure"]
        == "atomic_fail_closed"
    )
    assert (
        CUDA_BIRTH_LIFECYCLE_EVENT_CAPABILITIES["invalid_semantics"]
        == "asynchronous_atomic_fail_closed"
    )
    assert (
        CUDA_BIRTH_LIFECYCLE_EVENT_CAPABILITIES[
            "host_synchronization_during_publication"
        ]
        is False
    )
    assert CUDA_BIRTH_LIFECYCLE_EVENT_CAPABILITIES["dynamic_mask_sized_output"] is False
    for unsupported in (
        "patch_context_sidecars",
        "generation_turnover",
        "generation_integration",
        "gru",
        "jungle",
        "wdl",
        "promotion_qualified",
    ):
        assert CUDA_BIRTH_LIFECYCLE_EVENT_CAPABILITIES[unsupported] is False


def test_native_source_forbids_host_sync_and_dynamic_shape_primitives() -> None:
    source = LIFECYCLE_EVENTS_SOURCE.read_text(encoding="utf-8")
    for forbidden in (".item", "torch::equal", "masked_select", "nonzero"):
        assert forbidden not in source
    assert "#ifndef EIEL_CUDA_ARCHITECTURE_B_NO_PYBIND" in source
    assert "PYBIND11_MODULE(TORCH_EXTENSION_NAME, module)" in source
