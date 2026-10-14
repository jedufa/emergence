from __future__ import annotations

import copy
import json
from typing import Any

import numpy as np
import pytest

from engine.native_runtime.cuda_architecture_b_cell_schedule_v1 import (
    CELL_STATE_ACTIVE_V1,
    CELL_STATE_CANCELLED_V1,
    CELL_STATE_COMPLETED_V1,
    CELL_STATE_FAILED_V1,
    COMMAND_CANCEL_V1,
    COMMAND_CHECKPOINT_AND_FORK_V1,
    COMMAND_CHECKPOINT_V1,
    DISPATCH_PLAN_SCHEMA_VERSION_V1,
    SCHEDULE_RECEIPT_SCHEMA_VERSION_V1,
    SCHEDULE_SPEC_SCHEMA_VERSION_V1,
    CellScheduleContractError,
    CellScheduleStateError,
    create_cell_schedule_v1,
)
from engine.perf.cuda_architecture_b_auto_batch_v1 import select_auto_batch_v1
from engine.perf.cuda_architecture_b_cell_decision_summary_v1 import (
    CELL_STATUS_UNEXECUTED_V1,
    freeze_cell_decision_summary_v1,
)
from engine.perf.cuda_architecture_b_checkpoint_branch import (
    create_checkpoint_branch,
)
from engine.perf.cuda_architecture_b_patch_p2_cell_params_v1 import (
    normalize_patch_p2_cell_params_v1,
    patch_p2_cell_params_sha256_v1,
)


_PATCH_GENERATION_CHUNKS = (2, 8, 32, 64, 128)


def _params_config() -> dict[str, object]:
    return {
        "patch_scent_decay_per_step": 0.05,
        "patch_waste_decay_per_step": 0.02,
        "patch_scent_presence_emit": 0.01,
        "patch_waste_presence_emit": 0.005,
        "patch_waste_density_emit_scale": 0.125,
        "patch_scent_max": 1.0,
        "patch_waste_max": 1.0,
        "patch_food_amount_max": 1.0,
        "patch_food_regrow_per_step": 0.02,
        "patch_food_regrow_use_fertility": True,
        "patch_food_regrow_cap_by_fertility": False,
        "patch_food_presence_threshold": 0.5,
        "patch_density_radius": 1,
        "patch_density_include_self": True,
        "patch_density_normalize": False,
        "patch_world_waste_consequence_enable": True,
        "patch_waste_regrowth_penalty_enable": True,
        "patch_waste_regrowth_penalty_scale": 0.5,
        "patch_waste_regrowth_penalty_max": 0.8,
        "patch_density_regrowth_penalty_enable": True,
        "patch_density_regrowth_penalty_scale": 0.25,
        "patch_density_regrowth_penalty_max": 0.7,
        "patch_regrowth_min_multiplier": 0.1,
        "patch_food_amount_initial_food_cell_value": 1.0,
        "patch_food_gather_amount": 1.0,
        "patch_scent_move_emit": 0.02,
        "patch_waste_move_emit": 0.03,
        "patch_scent_gather_emit": 0.05,
        "patch_waste_consume_emit": 0.05,
        "patch_movement_cost_apply_enable": True,
        "patch_movement_cost_energy_scale": 1.0,
        "patch_movement_cost_min": 0.0,
        "patch_movement_cost_max": 4.0,
    }


def _branch(cell_index: int, *, source_generation: int = 100):
    return create_checkpoint_branch(
        branch_key=f"cell-{cell_index}",
        source_checkpoint_projection_sha256=f"{cell_index + 1:064x}",
        source_generation=source_generation,
        trajectory_projection={"cell_index": cell_index, "world": "patch"},
        source_rng_state_sha256=f"{cell_index + 101:064x}",
        source_branch_id=f"parent-{cell_index}",
    )


def _auto_batch_receipt(cell_count: int) -> dict[str, Any]:
    request = {
        "world": "patch",
        "profile": "patch_p2",
        "build_identity": "architecture_b_v8",
        "extension_sha256": "a" * 64,
        "state_layout_sha256": "b" * 64,
        "active_policy_family": "fixed_mlp",
        "generations": 8,
        "ticks": 80,
        "summary_mode": "compact",
        "tick_output_storage": "rolling_single_row",
    }
    return select_auto_batch_v1(
        request=request,
        resources={
            "total_memory_bytes": 100_000,
            "free_memory_bytes": 100_000,
            "reserve_floor_bytes": 1_000,
            "reserve_fraction": 0.01,
            "shared_bytes": 1_000,
            "per_cell_bytes": 1_000,
            "sm_count": 24,
            "mandatory_kernel_blocks_per_sm": {"controller": 2},
        },
        qualified_envelopes=[
            {
                **request,
                "envelope_id": f"b{cell_count}",
                "qualified": True,
                "batch_size": cell_count,
            }
        ],
    )


def _spec(
    *,
    limits: tuple[int, ...] = (2, 4, 8),
    branches: tuple[Any, ...] | None = None,
) -> dict[str, Any]:
    cell_count = len(limits)
    source_branches = branches or tuple(_branch(index) for index in range(cell_count))
    params = normalize_patch_p2_cell_params_v1(_params_config())
    return {
        "schema_version": SCHEDULE_SPEC_SCHEMA_VERSION_V1,
        "controller_id": "test-bn-controller",
        "cell_count": cell_count,
        "cells": [
            {
                "cell_index": index,
                "stable_cell_id": 1000 + index,
                "generation_limit": limit,
                "branch": source_branches[index],
                "parameters": params,
            }
            for index, limit in enumerate(limits)
        ],
        "auto_batch_receipt": _auto_batch_receipt(cell_count),
    }


def _summary(
    schedule: Any,
    rows: tuple[tuple[int, ...], ...],
):
    maximum = schedule.to_dict()["maximum_generation_limit"]
    cell_count = schedule.cell_count
    status = np.full(
        (maximum, cell_count), CELL_STATUS_UNEXECUTED_V1, dtype=np.int32
    )
    for cell_index, values in enumerate(rows):
        status[: len(values), cell_index] = np.array(values, dtype=np.int32)
    validity = (status == 0).astype(np.uint8)
    score_basis = np.zeros((maximum, cell_count, 1), dtype=np.float64)
    stable_ids = np.array(
        [cell["stable_cell_id"] for cell in schedule.to_dict()["cells"]],
        dtype=np.int64,
    )
    return freeze_cell_decision_summary_v1(
        validity=validity,
        status=status,
        score_basis=score_basis,
        stable_cell_ids=stable_ids,
        score_policy_id="test.schedule.progress.v1",
        score_policy_digest=bytes(range(32)),
    )


def _command(
    *, cell_index: int, boundary: int, kind: str, command_id: str = "command-1"
) -> dict[str, Any]:
    return {
        "command_id": command_id,
        "cell_index": cell_index,
        "generation_boundary": boundary,
        "kind": kind,
    }


def _plan(
    schedule: Any,
    *,
    maximum: Any = 8,
    accepted_chunks: Any = _PATCH_GENERATION_CHUNKS,
):
    return schedule.plan_dispatch_cohorts(
        max_generations_per_dispatch=maximum,
        accepted_generation_chunks=accepted_chunks,
    )


def test_configurable_b3_schedule_reuses_all_accepted_identity_contracts() -> None:
    schedule = create_cell_schedule_v1(_spec())
    repeated = create_cell_schedule_v1(_spec())
    receipt = schedule.to_dict()

    assert repeated.schedule_sha256 == schedule.schedule_sha256
    assert repeated.to_dict() == receipt
    assert receipt["schema_version"] == SCHEDULE_RECEIPT_SCHEMA_VERSION_V1
    assert receipt["cell_count"] == 3
    assert receipt["maximum_generation_limit"] == 8
    assert [cell["generation_limit"] for cell in receipt["cells"]] == [2, 4, 8]
    assert receipt["auto_batch_binding"]["selected_cell_count"] == 3
    assert receipt["auto_batch_binding"]["selection_reused_without_recalculation"]
    assert len({cell["branch_id"] for cell in receipt["cells"]}) == 3
    assert [cell["parent_branch_id"] for cell in receipt["cells"]] == [
        "parent-0",
        "parent-1",
        "parent-2",
    ]
    expected_parameter_digest = patch_p2_cell_params_sha256_v1(
        normalize_patch_p2_cell_params_v1(_params_config())
    )
    assert {
        cell["parameter_digest"]["sha256"] for cell in receipt["cells"]
    } == {expected_parameter_digest}
    assert receipt["contract"]["live_controller_integration"] is False
    assert receipt["contract"]["production_promotion_claimed"] is False
    assert json.loads(json.dumps(receipt, allow_nan=False)) == receipt


def test_unknown_fields_duplicate_identities_and_batch_mismatch_fail_closed() -> None:
    unknown_top = _spec()
    unknown_top["unknown"] = True
    with pytest.raises(CellScheduleContractError, match="unknown=.*unknown"):
        create_cell_schedule_v1(unknown_top)

    unknown_cell = _spec()
    unknown_cell["cells"][0]["mystery"] = 7
    with pytest.raises(CellScheduleContractError, match="unknown=.*mystery"):
        create_cell_schedule_v1(unknown_cell)

    duplicate_branch = _branch(0)
    with pytest.raises(CellScheduleContractError, match="duplicate branch_id"):
        create_cell_schedule_v1(
            _spec(branches=(duplicate_branch, duplicate_branch, _branch(2)))
        )

    duplicate_stable = _spec()
    duplicate_stable["cells"][1]["stable_cell_id"] = 1000
    with pytest.raises(CellScheduleContractError, match="stable_cell_id.*unique"):
        create_cell_schedule_v1(duplicate_stable)

    wrong_batch = _spec()
    wrong_batch["auto_batch_receipt"] = _auto_batch_receipt(2)
    with pytest.raises(CellScheduleContractError, match="exactly match"):
        create_cell_schedule_v1(wrong_batch)


def test_dispatch_planner_steps_mixed_g2_g4_g8_without_overrunning_cells() -> None:
    schedule = create_cell_schedule_v1(_spec())

    first = _plan(schedule).to_dict()
    assert first["schema_version"] == DISPATCH_PLAN_SCHEMA_VERSION_V1
    assert first["cohorts"] == [
        {
            "cohort_index": 0,
            "physical_batch_cell_count": 3,
            "cell_indices": [0, 1, 2],
            "stable_cell_ids": [1000, 1001, 1002],
            "start_generation": 100,
            "generation_count": 2,
            "stop_reason": "generation_limit",
            "stop_cell_indices": [0],
            "shared_generation_coordinate": True,
        }
    ]

    schedule.apply_compact_summary(_summary(schedule, ((0, 0), (0, 0), (0, 0))))
    second = _plan(schedule).to_dict()
    assert second["terminal_cell_indices"] == [0]
    assert second["cohorts"][0]["cell_indices"] == [1, 2]
    assert second["cohorts"][0]["start_generation"] == 102
    assert second["cohorts"][0]["generation_count"] == 2

    schedule.apply_compact_summary(
        _summary(schedule, ((0, 0), (0, 0, 0, 0), (0, 0, 0, 0)))
    )
    third = _plan(schedule).to_dict()
    assert third["terminal_cell_indices"] == [0, 1]
    assert third["cohorts"][0]["cell_indices"] == [2]
    assert third["cohorts"][0]["start_generation"] == 104
    assert third["cohorts"][0]["generation_count"] == 2
    assert third["cohorts"][0]["stop_reason"] == "accepted_generation_chunk"
    assert third["cohorts"][0]["stop_cell_indices"] == []

    schedule.apply_compact_summary(
        _summary(schedule, ((0, 0), (0, 0, 0, 0), (0,) * 6))
    )
    fourth = _plan(schedule).to_dict()
    assert fourth["cohorts"][0]["start_generation"] == 106
    assert fourth["cohorts"][0]["generation_count"] == 2
    assert fourth["cohorts"][0]["stop_reason"] == "generation_limit"
    assert [
        plan["cohorts"][0]["generation_count"]
        for plan in (first, second, third, fourth)
    ] == [2, 2, 2, 2]


def test_dispatch_planner_uses_explicit_chunks_and_is_read_only_repeatable() -> None:
    schedule = create_cell_schedule_v1(_spec())
    before = schedule.to_dict()
    first = _plan(schedule, maximum=1, accepted_chunks=(1, 2, 8)).to_dict()
    second = _plan(schedule, maximum=1, accepted_chunks=(8, 1, 2)).to_dict()

    assert first == second
    assert first["plan_sha256"] == second["plan_sha256"]
    assert first["cohorts"][0]["physical_batch_cell_count"] == 3
    assert first["cohorts"][0]["generation_count"] == 1
    assert first["cohorts"][0]["stop_reason"] == (
        "max_generations_per_dispatch"
    )
    assert first["contract"]["live_controller_integration"] is False
    assert first["contract"]["performance_claimed"] is False
    assert schedule.to_dict() == before

    with pytest.raises(CellScheduleStateError, match="no exact decomposition"):
        _plan(schedule, maximum=1)


def test_dispatch_planner_stops_for_pending_checkpoint_and_excludes_blocked_cell() -> None:
    schedule = create_cell_schedule_v1(_spec(limits=(4, 4, 8)))
    schedule.request_boundary_command(
        _command(cell_index=2, boundary=2, kind=COMMAND_CHECKPOINT_V1)
    )
    first = _plan(schedule).to_dict()
    assert first["cohorts"][0]["cell_indices"] == [0, 1, 2]
    assert first["cohorts"][0]["generation_count"] == 2
    assert first["cohorts"][0]["stop_reason"] == "pending_boundary"
    assert first["cohorts"][0]["stop_cell_indices"] == [2]

    schedule.apply_compact_summary(_summary(schedule, ((0, 0), (0, 0), (0, 0))))
    second = _plan(schedule).to_dict()
    assert second["cohorts"][0]["cell_indices"] == [0, 1]
    assert second["cohorts"][0]["generation_count"] == 2
    assert second["blocked_active_cells"] == [
        {
            "cell_index": 2,
            "stable_cell_id": 1002,
            "current_generation": 102,
            "reason": "pending_boundary",
            "pending_boundary": 2,
            "pending_command_kind": COMMAND_CHECKPOINT_V1,
        }
    ]


def test_dispatch_planner_groups_only_matching_absolute_generations() -> None:
    branches = (_branch(0), _branch(1, source_generation=101), _branch(2))
    schedule = create_cell_schedule_v1(_spec(branches=branches))
    plan = _plan(schedule).to_dict()

    assert [(row["start_generation"], row["cell_indices"]) for row in plan["cohorts"]] == [
        (100, [0, 2]),
        (101, [1]),
    ]
    assert plan["cohorts"][0]["generation_count"] == 2
    assert plan["cohorts"][1]["generation_count"] == 2
    assert plan["cohorts"][1]["stop_reason"] == "accepted_generation_chunk"
    assert plan["cohorts"][1]["stop_cell_indices"] == []


def test_dispatch_planner_excludes_terminal_cells_and_fails_without_safe_progress() -> None:
    schedule = create_cell_schedule_v1(_spec())
    schedule.apply_compact_summary(
        _summary(schedule, ((0, 0), (0, 0, 0, 0), (0, 0)))
    )
    plan = _plan(schedule).to_dict()
    assert plan["terminal_cell_indices"] == [0, 1]
    assert plan["cohorts"][0]["cell_indices"] == [2]

    completed = create_cell_schedule_v1(_spec())
    completed.apply_compact_summary(
        _summary(completed, ((0, 0), (0, 0, 0, 0), (0,) * 8))
    )
    with pytest.raises(CellScheduleStateError, match="no safe progress"):
        _plan(completed)

    blocked = create_cell_schedule_v1(_spec())
    for cell_index in range(3):
        blocked.request_boundary_command(
            _command(
                cell_index=cell_index,
                boundary=0,
                kind=COMMAND_CANCEL_V1,
                command_id=f"cancel-{cell_index}",
            )
        )
    with pytest.raises(CellScheduleStateError, match="no safe progress"):
        _plan(blocked)


@pytest.mark.parametrize("value", [0, -1, True, 1.5, "8"])
def test_dispatch_planner_rejects_invalid_chunk_limits(value: Any) -> None:
    schedule = create_cell_schedule_v1(_spec())
    with pytest.raises(CellScheduleContractError, match=r"positive|\[1"):
        _plan(schedule, maximum=value)


@pytest.mark.parametrize(
    "value",
    [None, (), "2,8", (2, 2), (0, 2), (True, 2), (1.5, 2)],
)
def test_dispatch_planner_rejects_invalid_accepted_chunks(value: Any) -> None:
    schedule = create_cell_schedule_v1(_spec())
    with pytest.raises(CellScheduleContractError, match="accepted_generation_chunks"):
        _plan(schedule, accepted_chunks=value)


def test_dispatch_planner_fails_closed_for_unreachable_limit_or_boundary() -> None:
    unreachable_limit = create_cell_schedule_v1(_spec(limits=(2, 3, 8)))
    with pytest.raises(CellScheduleStateError, match="no exact decomposition"):
        _plan(unreachable_limit)

    unreachable_boundary = create_cell_schedule_v1(_spec())
    unreachable_boundary.request_boundary_command(
        _command(cell_index=2, boundary=1, kind=COMMAND_CHECKPOINT_V1)
    )
    with pytest.raises(CellScheduleStateError, match="no exact decomposition"):
        _plan(unreachable_boundary)


def test_dispatch_planner_honors_a_custom_exact_chunk_capability() -> None:
    schedule = create_cell_schedule_v1(_spec(limits=(4,)))
    custom = _plan(schedule, accepted_chunks=(8, 4)).to_dict()
    live_patch = _plan(schedule).to_dict()

    assert custom["cohorts"][0]["generation_count"] == 4
    assert custom["cohorts"][0]["stop_reason"] == "generation_limit"
    assert live_patch["cohorts"][0]["generation_count"] == 2
    assert live_patch["cohorts"][0]["stop_reason"] == "accepted_generation_chunk"

    noncanonical = create_cell_schedule_v1(_spec(limits=(8,)))
    exact = _plan(noncanonical, accepted_chunks=(6, 4)).to_dict()
    reordered = _plan(noncanonical, accepted_chunks=(4, 6)).to_dict()
    assert exact == reordered
    assert exact["cohorts"][0]["generation_count"] == 4


def test_heterogeneous_progress_completes_g2_g4_and_fails_one_g8_cell() -> None:
    schedule = create_cell_schedule_v1(_spec())
    first = schedule.apply_compact_summary(
        _summary(schedule, ((0, 0), (0, 0), (0,)))
    )
    assert first.to_dict()["accepted"] is True
    cells = schedule.to_dict()["cells"]
    assert [(cell["state"], cell["compact_summary_row_frontier"]) for cell in cells] == [
        (CELL_STATE_COMPLETED_V1, 2),
        (CELL_STATE_ACTIVE_V1, 2),
        (CELL_STATE_ACTIVE_V1, 1),
    ]

    schedule.apply_compact_summary(
        _summary(schedule, ((0, 0), (0, 0, 0, 0), (0, 0, -7)))
    )
    cells = schedule.to_dict()["cells"]
    assert [cell["state"] for cell in cells] == [
        CELL_STATE_COMPLETED_V1,
        CELL_STATE_COMPLETED_V1,
        CELL_STATE_FAILED_V1,
    ]
    assert cells[2]["completed_generation_count"] == 2
    assert cells[2]["compact_summary_row_frontier"] == 3
    with pytest.raises(CellScheduleStateError, match="terminal"):
        schedule.request_boundary_command(
            _command(cell_index=2, boundary=2, kind=COMMAND_CHECKPOINT_V1)
        )


def test_nonprefix_rewrite_regression_and_rows_past_limit_are_rejected() -> None:
    nonprefix = create_cell_schedule_v1(_spec())
    with pytest.raises(CellScheduleStateError, match="contiguous prefix"):
        nonprefix.apply_compact_summary(
            _summary(nonprefix, ((), (0, CELL_STATUS_UNEXECUTED_V1, 0), ()))
        )

    past_limit = create_cell_schedule_v1(_spec())
    with pytest.raises(CellScheduleStateError, match="beyond its generation limit"):
        past_limit.apply_compact_summary(_summary(past_limit, ((0, 0, 0), (), ())))

    rewrite = create_cell_schedule_v1(_spec())
    rewrite.apply_compact_summary(_summary(rewrite, ((), (0, 0), ())))
    with pytest.raises(CellScheduleStateError, match="rewrites or regresses"):
        rewrite.apply_compact_summary(_summary(rewrite, ((), (0, -9), ())))
    with pytest.raises(CellScheduleStateError, match="rewrites or regresses"):
        rewrite.apply_compact_summary(_summary(rewrite, ((), (0,), ())))


def test_first_boundary_command_wins_and_receipts_are_idempotent() -> None:
    schedule = create_cell_schedule_v1(_spec())
    first_command = _command(
        cell_index=2,
        boundary=4,
        kind=COMMAND_CHECKPOINT_AND_FORK_V1,
        command_id="fork-at-four",
    )
    first = schedule.request_boundary_command(first_command).to_dict()
    repeated = schedule.request_boundary_command(first_command).to_dict()
    competing = schedule.request_boundary_command(
        _command(
            cell_index=2,
            boundary=4,
            kind=COMMAND_CANCEL_V1,
            command_id="cancel-at-four",
        )
    ).to_dict()

    assert first["disposition"] == "accepted_first_command"
    assert repeated["disposition"] == "idempotent_first_command"
    assert competing["accepted"] is False
    assert competing["disposition"] == "rejected_first_command_wins"
    assert competing["details"]["winning_command"] == first_command


def test_fork_requires_completed_checkpoint_and_exact_existing_lineage() -> None:
    schedule = create_cell_schedule_v1(_spec())
    schedule.request_boundary_command(
        _command(
            cell_index=2,
            boundary=2,
            kind=COMMAND_CHECKPOINT_AND_FORK_V1,
            command_id="fork-at-two",
        )
    )
    parent = schedule.to_dict()["cells"][2]
    checkpoint_digest = "d" * 64
    rng_digest = "e" * 64
    child = create_checkpoint_branch(
        branch_key="fork-child",
        source_checkpoint_projection_sha256=checkpoint_digest,
        source_generation=parent["start_generation"] + 2,
        trajectory_projection={"pressure": 0.75},
        source_rng_state_sha256=rng_digest,
        source_branch_id=parent["branch_id"],
    )
    fork_value = {
        "command_id": "fork-at-two",
        "cell_index": 2,
        "generation_boundary": 2,
        "branch": child,
    }
    with pytest.raises(CellScheduleStateError, match="complete checkpoint"):
        schedule.bind_fork_branch(fork_value)
    with pytest.raises(CellScheduleStateError, match="not complete"):
        schedule.record_checkpoint_boundary(
            {
                "command_id": "fork-at-two",
                "cell_index": 2,
                "generation_boundary": 2,
                "checkpoint_projection_sha256": checkpoint_digest,
                "rng_state_sha256": rng_digest,
            }
        )

    schedule.apply_compact_summary(_summary(schedule, ((), (), (0, 0))))
    checkpoint = schedule.record_checkpoint_boundary(
        {
            "command_id": "fork-at-two",
            "cell_index": 2,
            "generation_boundary": 2,
            "checkpoint_projection_sha256": checkpoint_digest,
            "rng_state_sha256": rng_digest,
        }
    )
    fork = schedule.bind_fork_branch(fork_value)
    assert checkpoint.to_dict()["disposition"] == "checkpoint_complete"
    assert fork.to_dict()["disposition"] == "fork_branch_bound"
    assert fork.to_dict()["details"]["checkpoint_clone_plan_required"] is True
    assert fork.to_dict()["details"]["checkpoint_clone_execution_performed"] is False

    wrong_parent = create_checkpoint_branch(
        branch_key="wrong-parent",
        source_checkpoint_projection_sha256=checkpoint_digest,
        source_generation=parent["start_generation"] + 2,
        trajectory_projection={"pressure": 0.5},
        source_rng_state_sha256=rng_digest,
        source_branch_id="not-the-parent",
    )
    with pytest.raises(CellScheduleStateError, match="parent branch"):
        schedule.bind_fork_branch({**fork_value, "branch": wrong_parent})


def test_pending_boundary_cannot_be_crossed_without_settlement() -> None:
    checkpoint_schedule = create_cell_schedule_v1(_spec())
    checkpoint_schedule.request_boundary_command(
        _command(cell_index=2, boundary=2, kind=COMMAND_CHECKPOINT_V1)
    )
    with pytest.raises(CellScheduleStateError, match="incomplete checkpoint"):
        checkpoint_schedule.apply_compact_summary(
            _summary(checkpoint_schedule, ((), (), (0, 0, 0)))
        )
    with pytest.raises(CellScheduleStateError, match="incomplete checkpoint"):
        checkpoint_schedule.apply_compact_summary(
            _summary(checkpoint_schedule, ((), (), (0, 0, -9)))
        )

    cancel_schedule = create_cell_schedule_v1(_spec())
    cancel_schedule.request_boundary_command(
        _command(cell_index=2, boundary=2, kind=COMMAND_CANCEL_V1)
    )
    with pytest.raises(CellScheduleStateError, match="cancellation boundary"):
        cancel_schedule.apply_compact_summary(
            _summary(cancel_schedule, ((), (), (0, 0, 0)))
        )


def test_cancel_settles_at_exact_boundary_and_terminal_commands_fail() -> None:
    schedule = create_cell_schedule_v1(_spec())
    schedule.request_boundary_command(
        _command(
            cell_index=0,
            boundary=2,
            kind=COMMAND_CANCEL_V1,
            command_id="cancel-g2",
        )
    )
    schedule.apply_compact_summary(_summary(schedule, ((0, 0), (), ())))
    assert schedule.to_dict()["cells"][0]["state"] == CELL_STATE_ACTIVE_V1
    cancellation = schedule.settle_cancellation(
        {"command_id": "cancel-g2", "cell_index": 0, "generation_boundary": 2}
    )
    assert cancellation.to_dict()["state"] == CELL_STATE_CANCELLED_V1
    with pytest.raises(CellScheduleStateError, match="terminal"):
        schedule.request_boundary_command(
            _command(cell_index=0, boundary=2, kind=COMMAND_CHECKPOINT_V1)
        )
    with pytest.raises(CellScheduleStateError, match="terminal"):
        schedule.settle_cancellation(
            {"command_id": "cancel-g2", "cell_index": 0, "generation_boundary": 2}
        )


@pytest.mark.parametrize(
    ("method", "payload"),
    [
        (
            "request_boundary_command",
            {
                "command_id": "x",
                "cell_index": 1,
                "generation_boundary": 1,
                "kind": COMMAND_CHECKPOINT_V1,
                "unknown": 1,
            },
        ),
        (
            "record_checkpoint_boundary",
            {
                "command_id": "x",
                "cell_index": 1,
                "generation_boundary": 1,
                "checkpoint_projection_sha256": "a" * 64,
                "rng_state_sha256": "b" * 64,
                "unknown": 1,
            },
        ),
        (
            "bind_fork_branch",
            {
                "command_id": "x",
                "cell_index": 1,
                "generation_boundary": 1,
                "branch": _branch(8),
                "unknown": 1,
            },
        ),
        (
            "settle_cancellation",
            {
                "command_id": "x",
                "cell_index": 1,
                "generation_boundary": 1,
                "unknown": 1,
            },
        ),
    ],
)
def test_all_mutation_schemas_reject_unknown_fields(
    method: str, payload: dict[str, Any]
) -> None:
    schedule = create_cell_schedule_v1(_spec())
    with pytest.raises(CellScheduleContractError, match="unknown=.*unknown"):
        getattr(schedule, method)(copy.deepcopy(payload))
