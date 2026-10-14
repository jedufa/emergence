from __future__ import annotations

from copy import deepcopy

import pytest

from engine.perf.cuda_full_evolution_v2.workhorse import (
    WorkhorseDispatchPlanError,
    compile_workhorse_absolute_dispatch_plan_v2,
    validate_workhorse_absolute_dispatch_plan_v2,
)


def test_preferred_k64_lands_exactly_on_science_checkpoint_25000() -> None:
    plan = compile_workhorse_absolute_dispatch_plan_v2(
        start_next_tick=0,
        target_next_tick=50_000,
        preferred_temporal_k=64,
        admitted_temporal_k=(1, 2, 4, 8, 16, 32, 64),
        boundary_next_ticks=(25_000, 50_000),
    )

    assert validate_workhorse_absolute_dispatch_plan_v2(plan) == plan
    assert plan["segments"] == [
        {
            "start_next_tick": 0,
            "end_next_tick": 24_960,
            "temporal_k": 64,
            "dispatch_count": 390,
        },
        {
            "start_next_tick": 24_960,
            "end_next_tick": 24_992,
            "temporal_k": 32,
            "dispatch_count": 1,
        },
        {
            "start_next_tick": 24_992,
            "end_next_tick": 25_000,
            "temporal_k": 8,
            "dispatch_count": 1,
        },
        {
            "start_next_tick": 25_000,
            "end_next_tick": 49_960,
            "temporal_k": 64,
            "dispatch_count": 390,
        },
        {
            "start_next_tick": 49_960,
            "end_next_tick": 49_992,
            "temporal_k": 32,
            "dispatch_count": 1,
        },
        {
            "start_next_tick": 49_992,
            "end_next_tick": 50_000,
            "temporal_k": 8,
            "dispatch_count": 1,
        },
    ]


def test_nonzero_start_and_one_tick_tail_are_exact() -> None:
    plan = compile_workhorse_absolute_dispatch_plan_v2(
        start_next_tick=101,
        target_next_tick=118,
        preferred_temporal_k=8,
        admitted_temporal_k=(1, 2, 4, 8),
        boundary_next_ticks=(110, 118),
    )
    assert [(row["temporal_k"], row["dispatch_count"]) for row in plan["segments"]] == [
        (8, 1),
        (1, 1),
        (8, 1),
    ]
    assert plan["segments"][1]["end_next_tick"] == 110
    assert plan["segments"][-1]["end_next_tick"] == 118


def test_physical_k_changes_only_physical_plan_identity() -> None:
    common = {
        "start_next_tick": 0,
        "target_next_tick": 25,
        "admitted_temporal_k": (1, 2, 4, 8),
        "boundary_next_ticks": (13, 25),
    }
    k8 = compile_workhorse_absolute_dispatch_plan_v2(
        preferred_temporal_k=8,
        **common,
    )
    k4 = compile_workhorse_absolute_dispatch_plan_v2(
        preferred_temporal_k=4,
        **common,
    )
    assert k8["boundary_next_ticks"] == k4["boundary_next_ticks"]
    assert k8["dispatch_plan_sha256"] != k4["dispatch_plan_sha256"]
    assert k8["identity_scope"].startswith("PHYSICAL_PACKING_LOCAL")


@pytest.mark.parametrize(
    ("updates", "message"),
    (
        ({"admitted_temporal_k": (2, 4, 8)}, "include K1"),
        ({"boundary_next_ticks": (24,)}, "end at target"),
        ({"boundary_next_ticks": (13, 13, 25)}, "unique"),
        ({"boundary_next_ticks": (0, 25)}, "in-span"),
    ),
)
def test_dispatch_plan_rejects_loss_of_arbitrary_boundary_authority(
    updates: dict[str, object], message: str
) -> None:
    arguments: dict[str, object] = {
        "start_next_tick": 0,
        "target_next_tick": 25,
        "preferred_temporal_k": 8,
        "admitted_temporal_k": (1, 2, 4, 8),
        "boundary_next_ticks": (13, 25),
    }
    arguments.update(updates)
    with pytest.raises(WorkhorseDispatchPlanError, match=message):
        compile_workhorse_absolute_dispatch_plan_v2(**arguments)


def test_dispatch_plan_tampering_does_not_validate() -> None:
    plan = compile_workhorse_absolute_dispatch_plan_v2(
        start_next_tick=0,
        target_next_tick=25,
        preferred_temporal_k=8,
        admitted_temporal_k=(1, 2, 4, 8),
        boundary_next_ticks=(13, 25),
    )
    hostile = deepcopy(plan)
    hostile["segments"][0]["dispatch_count"] += 1
    with pytest.raises(WorkhorseDispatchPlanError, match="does not recompute"):
        validate_workhorse_absolute_dispatch_plan_v2(hostile)
