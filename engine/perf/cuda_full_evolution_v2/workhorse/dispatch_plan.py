"""Packing-local absolute-tick dispatch schedules for Workhorse V2.

Science/checkpoint boundaries are authoritative absolute ticks.  A tuned CUDA
``K`` is only a physical preference: this compiler inserts smaller admitted
dispatches so no kernel launch crosses a required boundary.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Final

from .contract import WORKHORSE_TIME_AUTHORITY_V2, canonical_sha256


WORKHORSE_ABSOLUTE_DISPATCH_PLAN_SCHEMA_V2: Final[str] = (
    "eiel_workhorse_absolute_dispatch_plan_v2"
)


class WorkhorseDispatchPlanError(RuntimeError):
    """Raised when an exact absolute-tick schedule cannot be compiled."""


def compile_workhorse_absolute_dispatch_plan_v2(
    *,
    start_next_tick: int,
    target_next_tick: int,
    preferred_temporal_k: int,
    admitted_temporal_k: Sequence[int],
    boundary_next_ticks: Sequence[int],
) -> dict[str, Any]:
    """Compile a run-length physical schedule that lands on every boundary."""

    if (
        type(start_next_tick) is not int
        or start_next_tick < 0
        or type(target_next_tick) is not int
        or target_next_tick <= start_next_tick
        or type(preferred_temporal_k) is not int
        or preferred_temporal_k <= 0
        or isinstance(admitted_temporal_k, (str, bytes, bytearray))
        or isinstance(boundary_next_ticks, (str, bytes, bytearray))
    ):
        raise WorkhorseDispatchPlanError(
            "dispatch plan requires a positive absolute span and physical K"
        )
    admitted = tuple(admitted_temporal_k)
    boundaries = tuple(boundary_next_ticks)
    if (
        not admitted
        or any(type(value) is not int or value <= 0 for value in admitted)
        or len(set(admitted)) != len(admitted)
        or tuple(sorted(admitted)) != admitted
        or preferred_temporal_k not in admitted
        or 1 not in admitted
    ):
        raise WorkhorseDispatchPlanError(
            "admitted K must be unique, increasing, include preferred K, and include K1"
        )
    if (
        not boundaries
        or any(type(value) is not int for value in boundaries)
        or tuple(sorted(set(boundaries))) != boundaries
        or boundaries[-1] != target_next_tick
        or any(not start_next_tick < value <= target_next_tick for value in boundaries)
    ):
        raise WorkhorseDispatchPlanError(
            "absolute boundaries must be unique, increasing, in-span, and end at target"
        )
    usable = tuple(
        sorted(
            (value for value in admitted if value <= preferred_temporal_k),
            reverse=True,
        )
    )
    segments: list[dict[str, int]] = []
    cursor = start_next_tick
    dispatch_count = 0
    for boundary in boundaries:
        remaining = boundary - cursor
        for temporal_k in usable:
            count, remaining = divmod(remaining, temporal_k)
            if not count:
                continue
            end = cursor + count * temporal_k
            if segments and segments[-1]["temporal_k"] == temporal_k:
                segments[-1]["dispatch_count"] += count
                segments[-1]["end_next_tick"] = end
            else:
                segments.append(
                    {
                        "start_next_tick": cursor,
                        "end_next_tick": end,
                        "temporal_k": temporal_k,
                        "dispatch_count": count,
                    }
                )
            cursor = end
            dispatch_count += count
        if remaining != 0 or cursor != boundary:
            raise WorkhorseDispatchPlanError(
                f"absolute boundary {boundary} is not representable by admitted K"
            )
    body = {
        "schema_version": WORKHORSE_ABSOLUTE_DISPATCH_PLAN_SCHEMA_V2,
        "time_authority": WORKHORSE_TIME_AUTHORITY_V2,
        "start_next_tick": start_next_tick,
        "target_next_tick": target_next_tick,
        "total_ticks": target_next_tick - start_next_tick,
        "preferred_temporal_k": preferred_temporal_k,
        "admitted_temporal_k": list(admitted),
        "boundary_next_ticks": list(boundaries),
        "segments": segments,
        "dispatch_count": dispatch_count,
        "identity_scope": "PHYSICAL_PACKING_LOCAL_EXCLUDED_FROM_SCIENCE_IDENTITY",
    }
    return {**body, "dispatch_plan_sha256": canonical_sha256(body)}


def validate_workhorse_absolute_dispatch_plan_v2(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    if type(value) is not dict or set(value) != {
        "schema_version",
        "time_authority",
        "start_next_tick",
        "target_next_tick",
        "total_ticks",
        "preferred_temporal_k",
        "admitted_temporal_k",
        "boundary_next_ticks",
        "segments",
        "dispatch_count",
        "identity_scope",
        "dispatch_plan_sha256",
    }:
        raise WorkhorseDispatchPlanError("dispatch plan fields are inexact")
    expected = compile_workhorse_absolute_dispatch_plan_v2(
        start_next_tick=value["start_next_tick"],
        target_next_tick=value["target_next_tick"],
        preferred_temporal_k=value["preferred_temporal_k"],
        admitted_temporal_k=value["admitted_temporal_k"],
        boundary_next_ticks=value["boundary_next_ticks"],
    )
    if dict(value) != expected:
        raise WorkhorseDispatchPlanError("dispatch plan does not recompute")
    return expected


__all__ = [
    "WORKHORSE_ABSOLUTE_DISPATCH_PLAN_SCHEMA_V2",
    "WorkhorseDispatchPlanError",
    "compile_workhorse_absolute_dispatch_plan_v2",
    "validate_workhorse_absolute_dispatch_plan_v2",
]
