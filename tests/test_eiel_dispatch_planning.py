from __future__ import annotations

import pytest

from engine.dispatch_planning import (
    DispatchPlanningError,
    LogicalDispatchRequest,
    QualifiedChunkProfile,
    plan_dispatch,
)


def _profile(prefix: str = "B") -> QualifiedChunkProfile:
    return QualifiedChunkProfile(
        profile_id="bounded_chunks_v1",
        request_prefix=prefix,
        qualified_chunk_sizes=(32, 8, 2),
    )


def test_b100_composes_deterministically_from_qualified_chunks() -> None:
    plan = plan_dispatch("B100", profile=_profile())

    assert [item.units for item in plan.physical_dispatches] == [32, 32, 32, 2, 2]
    assert [item.logical_offset for item in plan.physical_dispatches] == [0, 32, 64, 96, 98]
    assert plan.qualified_units == 100
    assert plan.tail.logical_offset == 100
    assert plan.tail.units == 0
    assert plan.to_mapping()["logical_request"] == "B100"


def test_g128_uses_the_same_generic_planner() -> None:
    plan = plan_dispatch("g128", profile=_profile("g"))

    assert [item.units for item in plan.physical_dispatches] == [32, 32, 32, 32]
    assert plan.tail.units == 0


def test_tail_is_explicit_and_smaller_than_smallest_qualified_chunk() -> None:
    plan = plan_dispatch("B101", profile=_profile())

    assert [item.units for item in plan.physical_dispatches] == [32, 32, 32, 2, 2]
    assert plan.tail.logical_offset == 100
    assert plan.tail.units == 1
    assert plan.tail.units < min(_profile().qualified_chunk_sizes)


def test_plan_is_stable_across_repeated_calls() -> None:
    first = plan_dispatch("B100", profile=_profile()).to_mapping()
    second = plan_dispatch("B100", profile=_profile()).to_mapping()

    assert first == second


@pytest.mark.parametrize(
    "logical_request",
    ["", "B0", "B01", "100", "B-1", "B1.5", "B 10", 100, None],
)
def test_invalid_logical_requests_fail_closed(logical_request: object) -> None:
    with pytest.raises(DispatchPlanningError):
        LogicalDispatchRequest.parse(logical_request)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "sizes",
    [(), (2, 8), (8, 8), (8, 3), (8, 4, 3), (8, True), [8, 4, 2]],
)
def test_profile_requires_descending_divisibility_chain(sizes: object) -> None:
    with pytest.raises(DispatchPlanningError):
        QualifiedChunkProfile(
            profile_id="profile_v1",
            request_prefix="B",
            qualified_chunk_sizes=sizes,  # type: ignore[arg-type]
        )


def test_prefix_mismatch_fails_closed() -> None:
    with pytest.raises(DispatchPlanningError, match="does not match"):
        plan_dispatch("g128", profile=_profile("B"))


@pytest.mark.parametrize("limit", [0, -1, True, 1.5])
def test_dispatch_limit_must_be_an_exact_positive_integer(limit: object) -> None:
    with pytest.raises(DispatchPlanningError, match="max_physical_dispatches"):
        plan_dispatch(
            "B100",
            profile=_profile(),
            max_physical_dispatches=limit,  # type: ignore[arg-type]
        )


def test_dispatch_limit_is_checked_before_materialization() -> None:
    with pytest.raises(DispatchPlanningError, match="exceeds"):
        plan_dispatch("B100", profile=_profile(), max_physical_dispatches=4)


def test_request_and_profile_reject_boolean_integers() -> None:
    with pytest.raises(DispatchPlanningError):
        LogicalDispatchRequest(prefix="B", units=True)  # type: ignore[arg-type]
    with pytest.raises(DispatchPlanningError):
        QualifiedChunkProfile(
            profile_id="profile_v1",
            request_prefix="B",
            qualified_chunk_sizes=(8, 4, True),  # type: ignore[arg-type]
        )
