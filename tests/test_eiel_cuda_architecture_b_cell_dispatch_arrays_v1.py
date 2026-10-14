from __future__ import annotations

import copy
from dataclasses import replace
from types import MappingProxyType
from typing import Any

import numpy as np
import pytest

from engine.native_runtime.cuda_architecture_b_cell_dispatch_arrays_v1 import (
    ARRAY_ORDER_V1,
    BLOCKED_REASON_GENERATION_LIMIT_V1,
    BLOCKED_REASON_PENDING_BOUNDARY_V1,
    CELL_DISPATCH_ARRAYS_SCHEMA_VERSION_V1,
    PENDING_COMMAND_CHECKPOINT_V1,
    PENDING_COMMAND_NONE_V1,
    STOP_REASON_ACCEPTED_GENERATION_CHUNK_V1,
    STOP_REASON_GENERATION_LIMIT_V1,
    STOP_REASON_PENDING_BOUNDARY_V1,
    CellDispatchArraysContractError,
    cell_dispatch_arrays_content_sha256_v1,
    lower_cell_dispatch_plan_to_arrays_v1,
    validate_cell_dispatch_arrays_v1,
)
from engine.native_runtime.cuda_architecture_b_cell_schedule_v1 import (
    COMMAND_CHECKPOINT_V1,
    CellDispatchCohortPlanV1,
    CellDispatchPlanReceiptV1,
    create_cell_schedule_v1,
)
from tests.test_eiel_cuda_architecture_b_cell_schedule_v1 import _spec as _schedule_spec


def _mixed_g2_g4_g8_receipt() -> CellDispatchPlanReceiptV1:
    return CellDispatchPlanReceiptV1(
        controller_id="mixed-g2-g4-g8",
        schedule_sha256="a" * 64,
        max_generations_per_dispatch=8,
        cohorts=(
            CellDispatchCohortPlanV1(
                cohort_index=0,
                cell_indices=(0, 1, 2),
                stable_cell_ids=(1000, 1001, 1002),
                start_generation=100,
                generation_count=2,
                stop_reason="generation_limit",
                stop_cell_indices=(0,),
            ),
        ),
        terminal_cell_indices=(),
        blocked_active_cells=(),
    )


def _classified_receipt() -> CellDispatchPlanReceiptV1:
    return CellDispatchPlanReceiptV1(
        controller_id="classified-b6",
        schedule_sha256="b" * 64,
        max_generations_per_dispatch=8,
        cohorts=(
            CellDispatchCohortPlanV1(
                cohort_index=0,
                cell_indices=(2, 4),
                stable_cell_ids=(1002, 1004),
                start_generation=100,
                generation_count=2,
                stop_reason="pending_boundary+generation_limit",
                stop_cell_indices=(2, 4),
            ),
            CellDispatchCohortPlanV1(
                cohort_index=1,
                cell_indices=(3,),
                stable_cell_ids=(1003,),
                start_generation=101,
                generation_count=4,
                stop_reason="generation_limit",
                stop_cell_indices=(3,),
            ),
        ),
        terminal_cell_indices=(0,),
        blocked_active_cells=(
            {
                "cell_index": 1,
                "stable_cell_id": 1001,
                "current_generation": 100,
                "reason": "pending_boundary",
                "pending_boundary": 0,
                "pending_command_kind": COMMAND_CHECKPOINT_V1,
            },
            {
                "cell_index": 5,
                "stable_cell_id": 1005,
                "current_generation": 108,
                "reason": "generation_limit",
                "pending_boundary": None,
                "pending_command_kind": None,
            },
        ),
    )


def _accepted_chunk_receipt(
    *,
    stop_reason: str = "accepted_generation_chunk",
    stop_cell_indices: tuple[int, ...] = (),
) -> CellDispatchPlanReceiptV1:
    return CellDispatchPlanReceiptV1(
        controller_id="accepted-chunk-b3",
        schedule_sha256="c" * 64,
        max_generations_per_dispatch=8,
        cohorts=(
            CellDispatchCohortPlanV1(
                cohort_index=0,
                cell_indices=(0, 1, 2),
                stable_cell_ids=(1000, 1001, 1002),
                start_generation=104,
                generation_count=2,
                stop_reason=stop_reason,
                stop_cell_indices=stop_cell_indices,
            ),
        ),
        terminal_cell_indices=(),
        blocked_active_cells=(),
    )


def _readonly(values: Any, dtype: Any | None = None) -> np.ndarray[Any, Any]:
    array = np.array(values, dtype=dtype, copy=True, order="C")
    array.setflags(write=False)
    return array


def test_mixed_g2_g4_g8_lowers_to_exact_contiguous_arrays() -> None:
    lowered = lower_cell_dispatch_plan_to_arrays_v1(_mixed_g2_g4_g8_receipt())

    assert lowered.cohort_offsets.tolist() == [0, 3]
    assert lowered.cohort_start_generations.tolist() == [100]
    assert lowered.cohort_generation_counts.tolist() == [2]
    assert lowered.cohort_stop_reason_masks.tolist() == [
        STOP_REASON_GENERATION_LIMIT_V1
    ]
    assert lowered.cell_indices.tolist() == [0, 1, 2]
    assert lowered.stable_cell_ids.tolist() == [1000, 1001, 1002]
    assert lowered.cell_stop_flags.tolist() == [1, 0, 0]
    assert lowered.terminal_cell_indices.shape == (0,)
    assert lowered.blocked_cell_indices.shape == (0,)
    assert lowered.metadata["configured_cell_count"] == 3
    assert lowered.metadata["contract"] == {
        "metadata_only": True,
        "one_future_native_controller_call_input": True,
        "native_cpp_call_performed": False,
        "live_controller_integration": False,
        "cuda_execution_performed": False,
        "simulation_execution_performed": False,
        "performance_claimed": False,
        "production_promotion_claimed": False,
    }
    assert lowered.metadata["schema_version"] == CELL_DISPATCH_ARRAYS_SCHEMA_VERSION_V1
    assert lowered.content_sha256 == cell_dispatch_arrays_content_sha256_v1(lowered)

    assert len(lowered.ordered_arrays) == len(ARRAY_ORDER_V1)
    for name, array in zip(ARRAY_ORDER_V1, lowered.ordered_arrays, strict=True):
        assert array is getattr(lowered, name)
        assert array.dtype != np.dtype(object)
        assert array.ndim == 1
        assert array.flags.c_contiguous
        assert array.flags.aligned
        assert not array.flags.writeable


def test_terminal_and_blocked_metadata_is_explicit_and_lossless() -> None:
    lowered = lower_cell_dispatch_plan_to_arrays_v1(_classified_receipt().to_dict())

    assert lowered.cohort_offsets.tolist() == [0, 2, 3]
    assert lowered.cohort_stop_reason_masks.tolist() == [
        STOP_REASON_PENDING_BOUNDARY_V1 | STOP_REASON_GENERATION_LIMIT_V1,
        STOP_REASON_GENERATION_LIMIT_V1,
    ]
    assert lowered.cell_indices.tolist() == [2, 4, 3]
    assert lowered.cell_stop_flags.tolist() == [1, 1, 1]
    assert lowered.terminal_cell_indices.tolist() == [0]
    assert lowered.blocked_cell_indices.tolist() == [1, 5]
    assert lowered.blocked_stable_cell_ids.tolist() == [1001, 1005]
    assert lowered.blocked_current_generations.tolist() == [100, 108]
    assert lowered.blocked_reason_codes.tolist() == [
        BLOCKED_REASON_PENDING_BOUNDARY_V1,
        BLOCKED_REASON_GENERATION_LIMIT_V1,
    ]
    assert lowered.blocked_pending_boundary_present.tolist() == [1, 0]
    assert lowered.blocked_pending_boundaries.tolist() == [0, -1]
    assert lowered.blocked_pending_command_kind_codes.tolist() == [
        PENDING_COMMAND_CHECKPOINT_V1,
        PENDING_COMMAND_NONE_V1,
    ]
    assert lowered.metadata["terminal_cell_count"] == 1
    assert lowered.metadata["blocked_active_cell_count"] == 2
    validate_cell_dispatch_arrays_v1(lowered)


def test_intermediate_accepted_generation_chunk_preserves_capability_semantics() -> None:
    lowered = lower_cell_dispatch_plan_to_arrays_v1(
        _accepted_chunk_receipt().to_dict()
    )

    assert lowered.cohort_generation_counts.tolist() == [2]
    assert lowered.cohort_stop_reason_masks.tolist() == [
        STOP_REASON_ACCEPTED_GENERATION_CHUNK_V1
    ]
    assert lowered.cell_stop_flags.tolist() == [0, 0, 0]
    assert lowered.metadata["selection_policy"]["secondary"] == (
        "largest_accepted_chunk_preserving_exact_boundary_decomposition"
    )
    assert lowered.metadata["codebooks"]["cohort_stop_reason_bits"][
        "accepted_generation_chunk"
    ] == STOP_REASON_ACCEPTED_GENERATION_CHUNK_V1
    assert lowered.metadata["accepted_generation_chunk_semantics"] == {
        "capability_argument_required_by_planner": True,
        "capability_list_exposed_by_source_receipt": False,
        "capability_list_inferred_by_lowering": False,
        "selected_chunk_array": "cohort_generation_counts",
        "intermediate_reason_array": "cohort_stop_reason_masks",
    }
    validate_cell_dispatch_arrays_v1(lowered)


def test_real_capability_aware_planner_receipt_lowers_without_semantic_loss() -> None:
    schedule = create_cell_schedule_v1(_schedule_spec(limits=(8,)))
    receipt = schedule.plan_dispatch_cohorts(
        max_generations_per_dispatch=8,
        accepted_generation_chunks=(2,),
    )
    assert receipt.to_dict()["cohorts"][0]["stop_reason"] == (
        "accepted_generation_chunk"
    )

    lowered = lower_cell_dispatch_plan_to_arrays_v1(receipt)

    assert lowered.cohort_generation_counts.tolist() == [2]
    assert lowered.cohort_stop_reason_masks.tolist() == [
        STOP_REASON_ACCEPTED_GENERATION_CHUNK_V1
    ]
    assert lowered.cell_stop_flags.tolist() == [0]
    assert lowered.metadata["plan_sha256"] == receipt.plan_sha256


@pytest.mark.parametrize(
    ("stop_reason", "stop_cell_indices", "message"),
    [
        (
            "accepted_generation_chunk",
            (0,),
            "must not identify limiting cells",
        ),
        ("generation_limit", (), "must identify at least one limiting cell"),
        (
            "generation_limit+accepted_generation_chunk",
            (),
            "must be a standalone reason",
        ),
    ],
)
def test_malformed_intermediate_chunk_semantics_fail_closed(
    stop_reason: str,
    stop_cell_indices: tuple[int, ...],
    message: str,
) -> None:
    receipt = _accepted_chunk_receipt(
        stop_reason=stop_reason,
        stop_cell_indices=stop_cell_indices,
    )

    with pytest.raises(CellDispatchArraysContractError, match=message):
        lower_cell_dispatch_plan_to_arrays_v1(receipt)


def test_receipt_and_payload_lower_deterministically() -> None:
    receipt = _mixed_g2_g4_g8_receipt()
    first = lower_cell_dispatch_plan_to_arrays_v1(receipt)
    second = lower_cell_dispatch_plan_to_arrays_v1(receipt.to_dict())
    third = lower_cell_dispatch_plan_to_arrays_v1(receipt)

    assert first.metadata == second.metadata == third.metadata
    assert first.content_sha256 == second.content_sha256 == third.content_sha256
    for first_array, second_array, third_array in zip(
        first.ordered_arrays,
        second.ordered_arrays,
        third.ordered_arrays,
        strict=True,
    ):
        np.testing.assert_array_equal(first_array, second_array)
        np.testing.assert_array_equal(first_array, third_array)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda payload: payload["cohorts"][0].update(
                physical_batch_cell_count=2
            ),
            "list lengths disagree",
        ),
        (
            lambda payload: (
                payload["cohorts"][0]["cell_indices"].reverse(),
                payload["cohorts"][0]["stable_cell_ids"].reverse(),
            ),
            "ordered by stable_cell_id",
        ),
        (
            lambda payload: payload["terminal_cell_indices"].append(0),
            "classes overlap",
        ),
        (
            lambda payload: payload.update(plan_sha256="0" * 64),
            "canonical plan payload",
        ),
    ],
)
def test_malformed_or_contradictory_source_plans_fail_closed(
    mutate: Any, message: str
) -> None:
    payload = copy.deepcopy(_mixed_g2_g4_g8_receipt().to_dict())
    mutate(payload)

    with pytest.raises(CellDispatchArraysContractError, match=message):
        lower_cell_dispatch_plan_to_arrays_v1(payload)


def test_array_shape_dtype_and_order_drift_fail_closed() -> None:
    lowered = lower_cell_dispatch_plan_to_arrays_v1(_mixed_g2_g4_g8_receipt())

    wrong_shape = replace(
        lowered,
        cohort_offsets=_readonly(lowered.cohort_offsets[:-1], dtype=np.int64),
    )
    with pytest.raises(CellDispatchArraysContractError, match=r"shape \(2,\)"):
        validate_cell_dispatch_arrays_v1(wrong_shape)

    wrong_dtype = replace(
        lowered,
        cell_indices=_readonly(lowered.cell_indices, dtype=np.int32),
    )
    with pytest.raises(CellDispatchArraysContractError, match="dtype or shape"):
        validate_cell_dispatch_arrays_v1(wrong_dtype)

    object_dtype = replace(
        lowered,
        stable_cell_ids=_readonly(lowered.stable_cell_ids, dtype=object),
    )
    with pytest.raises(CellDispatchArraysContractError, match="dtype or shape"):
        validate_cell_dispatch_arrays_v1(object_dtype)

    wrong_order = replace(
        lowered,
        cell_indices=_readonly(lowered.cell_indices[::-1], dtype=np.int64),
        stable_cell_ids=_readonly(lowered.stable_cell_ids[::-1], dtype=np.int64),
    )
    with pytest.raises(CellDispatchArraysContractError, match="cell order drift"):
        validate_cell_dispatch_arrays_v1(wrong_order)

    accepted = lower_cell_dispatch_plan_to_arrays_v1(_accepted_chunk_receipt())
    false_limiting_cell = replace(
        accepted,
        cell_stop_flags=_readonly([1, 0, 0], dtype=np.uint8),
    )
    with pytest.raises(CellDispatchArraysContractError, match="limiting cell"):
        validate_cell_dispatch_arrays_v1(false_limiting_cell)


def test_content_digest_detects_array_and_claim_drift() -> None:
    lowered = lower_cell_dispatch_plan_to_arrays_v1(_mixed_g2_g4_g8_receipt())
    changed_count = replace(
        lowered,
        cohort_generation_counts=_readonly([1], dtype=np.int64),
    )

    assert cell_dispatch_arrays_content_sha256_v1(changed_count) != lowered.content_sha256
    with pytest.raises(CellDispatchArraysContractError, match="metadata and array bytes"):
        validate_cell_dispatch_arrays_v1(changed_count)

    changed_metadata = replace(
        lowered,
        metadata=MappingProxyType(
            {**lowered.metadata, "controller_id": "different-controller"}
        ),
    )
    assert cell_dispatch_arrays_content_sha256_v1(changed_metadata) != lowered.content_sha256
    with pytest.raises(CellDispatchArraysContractError, match="metadata and array bytes"):
        validate_cell_dispatch_arrays_v1(changed_metadata)

    wrong_claim = replace(lowered, content_sha256="f" * 64)
    with pytest.raises(CellDispatchArraysContractError, match="metadata and array bytes"):
        validate_cell_dispatch_arrays_v1(wrong_claim)
