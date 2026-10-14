"""Canonical tick-native projections over the legacy internal compiler lattice."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from engine.campaigns.canonical import canonical_sha256
from engine.campaigns.artifact_lifecycle import (
    ensure_lifecycle_contract_v1,
    rebind_lifecycle_checkpoint_schedule_v1,
)
from engine.campaigns.tick_native import (
    TICK_NATIVE_EXECUTION_ROUTE_V1,
    TickNativeExecutionRequestV1,
    assert_tick_native_plan_payload_v1,
)
from engine.campaigns.resources import planning_disk_upper_bound_absolute_ticks_v1


def _checkpoint_resume_ticks_v1(
    tick_request: TickNativeExecutionRequestV1,
) -> tuple[int, ...]:
    restore_ticks = tick_request.canonical_deliberate_restore_ticks
    if not restore_ticks:
        raise ValueError(
            "tick-native checkpoint/resume gate requires deliberate_restore_ticks"
        )
    return tuple(sorted({*restore_ticks, tick_request.total_ticks}))


def canonical_tick_native_spec_v1(
    spec: Mapping[str, Any],
    *,
    tick_request: TickNativeExecutionRequestV1,
    exercise_checkpoint_resume: bool = False,
) -> dict[str, Any]:
    """Remove generation-shaped public authority from a materialized spec."""

    value = dict(spec)
    execution = dict(value.get("execution") or {})
    execution.pop("ticks_per_generation", None)
    execution.pop("generations", None)
    execution.pop("physical_window_generations", None)
    execution.pop("max_internal_lifecycle_segments_per_dispatch", None)
    execution.pop("internal_lifecycle_schedule", None)
    execution.pop("digest_every_generation", None)
    execution["execution_route_identity"] = TICK_NATIVE_EXECUTION_ROUTE_V1
    execution.update(tick_request.payload())
    if exercise_checkpoint_resume:
        checkpoint_resume_ticks = _checkpoint_resume_ticks_v1(tick_request)
        inherited_gate = execution.get("technical_gate")
        inherited_gate = (
            dict(inherited_gate) if isinstance(inherited_gate, Mapping) else {}
        )
        execution["technical_gate"] = {
            "required": True,
            "time_authority": "absolute_tick_v1",
            "total_ticks": tick_request.total_ticks,
            "checkpoint_ticks": list(checkpoint_resume_ticks),
            "deliberate_restore_ticks": list(
                tick_request.canonical_deliberate_restore_ticks
            ),
            **(
                {"profile": inherited_gate["profile"]}
                if isinstance(inherited_gate.get("profile"), str)
                else {
                    "profile": "genome_v2_wdl_fixed_mutation_persistence_v1"
                }
            ),
        }
    else:
        execution["technical_gate"] = {"required": False}
    value["execution"] = execution
    checkpoints = dict(value.get("checkpoints") or {})
    checkpoints.pop("milestones", None)
    checkpoints["absolute_ticks"] = list(tick_request.checkpoint_ticks)
    checkpoints["interval_ticks"] = tick_request.checkpoint_every_ticks
    value["checkpoints"] = checkpoints
    comparisons = dict(value.get("comparisons") or {})
    comparisons.update(
        {
            "fresh_repeats": 0,
            "continuation_cells": 0,
            "continuation_splits": [],
        }
    )
    value["comparisons"] = comparisons
    return value


def canonical_tick_native_plan_v1(
    plan: Mapping[str, Any],
    *,
    tick_request: TickNativeExecutionRequestV1,
    exercise_checkpoint_resume: bool = False,
) -> dict[str, Any]:
    """Bind a compiled matrix to absolute ticks without a second time authority."""

    value = dict(plan)
    lifecycle = ensure_lifecycle_contract_v1(value)
    execution = dict(value.get("execution") or {})
    execution.pop("ticks_per_generation", None)
    execution.pop("generations", None)
    execution.pop("physical_window_generations", None)
    execution.pop("max_internal_lifecycle_segments_per_dispatch", None)
    execution.pop("internal_lifecycle_schedule", None)
    execution.pop("digest_every_generation", None)
    execution.pop("checkpoints", None)
    execution.pop("tick_schedule", None)
    execution["execution_route_identity"] = TICK_NATIVE_EXECUTION_ROUTE_V1
    execution.update(tick_request.payload())
    inherited_gate = execution.get("technical_gate")
    if exercise_checkpoint_resume:
        inherited_gate = (
            dict(inherited_gate) if isinstance(inherited_gate, Mapping) else {}
        )
        inherited_canary = inherited_gate.get("canary")
        inherited_canary = (
            dict(inherited_canary)
            if isinstance(inherited_canary, Mapping)
            else {}
        )
        cell_ids = inherited_gate.get("cell_ids")
        if not isinstance(cell_ids, list) or not cell_ids:
            raw_cells = value.get("cells")
            first_cell = (
                raw_cells[0].get("cell_id")
                if isinstance(raw_cells, list)
                and raw_cells
                and isinstance(raw_cells[0], Mapping)
                else None
            )
            if not isinstance(first_cell, str) or not first_cell:
                raise ValueError(
                    "tick-native checkpoint/resume gate has no materialized cell"
                )
            cell_ids = [first_cell]
        checkpoint_resume_ticks = _checkpoint_resume_ticks_v1(tick_request)
        execution["technical_gate"] = {
            "required": True,
            "execution_mode": "pre_primary_canary",
            "cell_ids": list(cell_ids),
            **(
                {"profile": inherited_gate["profile"]}
                if isinstance(inherited_gate.get("profile"), str)
                else {
                    "profile": "genome_v2_wdl_fixed_mutation_persistence_v1"
                }
            ),
            "canary": {
                "batch_id": inherited_canary.get(
                    "batch_id", "technical-gate-canary"
                ),
                "restore_batch_id": inherited_canary.get(
                    "restore_batch_id", "technical-gate-fresh-owner-restore"
                ),
                "time_authority": "absolute_tick_v1",
                "total_ticks": tick_request.total_ticks,
                "checkpoint_ticks": list(checkpoint_resume_ticks),
                "cell_ids": list(cell_ids),
            },
        }
    else:
        execution["technical_gate"] = {"required": False}
    value["execution"] = execution
    comparisons = dict(value.get("comparisons") or {})
    comparisons["branches"] = []
    value["comparisons"] = comparisons
    estimate = dict(value.get("resource_estimate") or {})
    estimate.pop("logical_generations", None)
    estimate.pop("ticks_per_generation", None)
    estimate["total_ticks"] = tick_request.total_ticks
    estimate["time_authority"] = "absolute_tick_v1"
    checkpoint_ticks = tick_request.checkpoint_ticks
    if tick_request.checkpoint_every_ticks is not None:
        checkpoint_ticks = tuple(
            range(
                tick_request.checkpoint_every_ticks,
                tick_request.total_ticks + 1,
                tick_request.checkpoint_every_ticks,
            )
        )
    checkpoint_ticks = tuple(sorted({*checkpoint_ticks, tick_request.total_ticks}))
    value["artifact_lifecycle"] = rebind_lifecycle_checkpoint_schedule_v1(
        lifecycle,
        checkpoint_ticks,
        checkpoint_authority="absolute_tick",
    )
    logical_cells = len(value.get("cells") or ())
    estimate["checkpoint_boundaries"] = len(checkpoint_ticks)
    estimate["checkpoint_ticks"] = list(checkpoint_ticks)
    estimate["resident_owner_span_ticks"] = tick_request.resident_owner_span_ticks
    estimate["native_dispatch_window_ticks"] = (
        tick_request.native_dispatch_window_ticks
    )
    estimate["resident_event_capacity"] = tick_request.event_window_capacity
    estimate["disk_bytes_upper_bound"] = planning_disk_upper_bound_absolute_ticks_v1(
        logical_trajectories=logical_cells,
        total_ticks=tick_request.total_ticks,
        checkpoint_boundaries=len(checkpoint_ticks),
    )
    value["resource_estimate"] = estimate
    from engine.campaigns.run_contract_v1 import refresh_resolved_run_contract_v1

    refreshed_contract = refresh_resolved_run_contract_v1(value)
    if refreshed_contract is not None:
        value["run_contract"] = refreshed_contract
    value.pop("plan_sha256", None)
    value["plan_sha256"] = canonical_sha256(value)
    assert_tick_native_plan_payload_v1(value)
    return value


__all__ = [
    "canonical_tick_native_plan_v1",
    "canonical_tick_native_spec_v1",
]
