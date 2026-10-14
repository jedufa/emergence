from __future__ import annotations

"""CXXCORE-17 runner/generation island preflight helpers.

Default-off diagnostic code. This module does not integrate a native path into
the live runner. It defines the summary/event-tape contract that a larger
runner/generation island would need before another native pilot is worth doing.
"""

from dataclasses import dataclass
import hashlib
import json
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from perf.cxxcore15_resident_soa_tick_island import (
    ResidentRunResult,
    ResidentSoAConfig,
    as_resident_state,
    compare_resident_to_reference,
    resident_result_payload,
    run_resident_python_reference,
)
from perf.cxxcore8_guarded_runner_bridge import state_digest


SCHEMA_VERSION = "cxxcore17_runner_generation_island_v1"
AUDIT_SCHEMA_VERSION = "cxxcore17_runner_generation_dependency_audit_v1"
SUMMARY_TAPE_SCHEMA_VERSION = "cxxcore17_runner_summary_tape_v1"
DECISION = "CXXCORE17-B-SUMMARY-TAPE-GO-NATIVE-LATER"

DEPENDENCY_CLASSES = (
    "resident_safe",
    "summary_safe",
    "sync_point_required",
    "per_tick_python_required",
    "out_of_scope_surface",
)

MODES = (
    "off",
    "python_summary_reference",
    "native_summary_shadow",
    "native_summary_authoritative_guarded",
    "native_summary_authoritative_no_compare_lower_bound",
)


@dataclass(frozen=True)
class RunnerIslandConfig:
    cxxcore17_runner_island_enable: bool = False
    cxxcore17_runner_island_mode: str = "off"
    cxxcore17_runner_island_compare_enable: bool = True
    cxxcore17_runner_island_summary_interval: int = 0
    cxxcore17_runner_island_fail_closed: bool = True


@dataclass(frozen=True)
class DependencyAuditRow:
    dependency: str
    current_owner: str
    read_write_frequency: str
    required_by_policy: bool
    required_by_metrics: bool
    can_be_summarized: bool
    can_be_deferred_to_sync: bool
    blocks_resident_island: bool
    classification: str
    notes: str


@dataclass(frozen=True)
class CandidateBoundary:
    candidate: str
    required_state_ownership: tuple[str, ...]
    python_object_materialization_frequency: str
    expected_speed_economics: str
    parity_risk: str
    rng_risk: str
    checkpoint_risk: str
    metrics_risk: str
    implementation_cost: str
    pilot_in_this_branch: bool
    assessment: str


@dataclass(frozen=True)
class RunnerTickSummaryTape:
    schema_version: str
    source: str
    ticks: int
    agents: int
    sync_interval: int
    energy_delta_total: float
    health_delta_total: float
    injury_delta_total: float
    scar_delta_total: float
    death_count: int
    food_gained_total: float
    moved_count: int
    cost_total: float
    action_counts: dict[str, int]
    sparse_summaries: list[dict[str, float | int]]
    final_state_digest: str
    digest: str


@dataclass(frozen=True)
class RunnerIslandResult:
    schema_version: str
    mode: str
    decision: str
    status: str
    parity: dict[str, Any]
    fallback_reason: str | None
    first_divergence: dict[str, Any] | None
    summary_tape: RunnerTickSummaryTape | None
    python_env_step_skipped: bool
    materializations: int
    timing: dict[str, float]
    diagnostics: dict[str, Any]


def config_from_sim_cfg(sim_cfg: Any) -> RunnerIslandConfig:
    return RunnerIslandConfig(
        cxxcore17_runner_island_enable=bool(getattr(sim_cfg, "cxxcore17_runner_island_enable", False)),
        cxxcore17_runner_island_mode=str(getattr(sim_cfg, "cxxcore17_runner_island_mode", "off") or "off"),
        cxxcore17_runner_island_compare_enable=bool(
            getattr(sim_cfg, "cxxcore17_runner_island_compare_enable", True)
        ),
        cxxcore17_runner_island_summary_interval=int(
            getattr(sim_cfg, "cxxcore17_runner_island_summary_interval", 0) or 0
        ),
        cxxcore17_runner_island_fail_closed=bool(getattr(sim_cfg, "cxxcore17_runner_island_fail_closed", True)),
    )


def dependency_audit_rows() -> list[DependencyAuditRow]:
    return [
        DependencyAuditRow(
            dependency="env scalar state and positions",
            current_owner="GridWorld arrays plus Agent scalar mirrors",
            read_write_frequency="every env tick",
            required_by_policy=True,
            required_by_metrics=True,
            can_be_summarized=True,
            can_be_deferred_to_sync=True,
            blocks_resident_island=False,
            classification="resident_safe",
            notes="Can remain resident if policy inputs/actions are precomputed and final scalar mirrors sync at a safe boundary.",
        ),
        DependencyAuditRow(
            dependency="Agent.stats last deltas and survival/collection counters",
            current_owner="Python Agent.stats in TickEnvApplyMixin",
            read_write_frequency="every env tick",
            required_by_policy=False,
            required_by_metrics=True,
            can_be_summarized=True,
            can_be_deferred_to_sync=True,
            blocks_resident_island=False,
            classification="summary_safe",
            notes="A summary tape can carry totals and final counter updates; live metrics must not read stale per-agent objects mid-window.",
        ),
        DependencyAuditRow(
            dependency="memory_action_window instrumentation",
            current_owner="Python instrumentation JSONL writers",
            read_write_frequency="pre and post env tick when enabled",
            required_by_policy=False,
            required_by_metrics=True,
            can_be_summarized=True,
            can_be_deferred_to_sync=False,
            blocks_resident_island=True,
            classification="per_tick_python_required",
            notes="Current implementation holds pending per-tick rows and reads post-env Agent health/positions.",
        ),
        DependencyAuditRow(
            dependency="prediction_demand_action_context instrumentation",
            current_owner="Python instrumentation and Patch World arrays",
            read_write_frequency="pre env, patch step, and generation summary",
            required_by_policy=False,
            required_by_metrics=True,
            can_be_summarized=True,
            can_be_deferred_to_sync=False,
            blocks_resident_island=True,
            classification="per_tick_python_required",
            notes="Could become summary-safe, but current code observes live env arrays and agent contexts each tick.",
        ),
        DependencyAuditRow(
            dependency="internal drives",
            current_owner="runner-level dense arrays plus Agent mirror compatibility",
            read_write_frequency="after every env tick",
            required_by_policy=True,
            required_by_metrics=True,
            can_be_summarized=False,
            can_be_deferred_to_sync=False,
            blocks_resident_island=True,
            classification="sync_point_required",
            notes="A larger island must own the drive arrays or sync before policy observes them.",
        ),
        DependencyAuditRow(
            dependency="chemistry state",
            current_owner="Python Agent chemistry objects",
            read_write_frequency="after every env tick and cost multiplier path",
            required_by_policy=True,
            required_by_metrics=True,
            can_be_summarized=False,
            can_be_deferred_to_sync=False,
            blocks_resident_island=True,
            classification="per_tick_python_required",
            notes="Object-based chemistry coupling currently blocks native ownership unless excluded or rehomed.",
        ),
        DependencyAuditRow(
            dependency="cost-charge expression diagnostics",
            current_owner="Python agents.cognition and diagnostics helpers",
            read_write_frequency="during env apply for cost-enabled configs",
            required_by_policy=False,
            required_by_metrics=True,
            can_be_summarized=True,
            can_be_deferred_to_sync=True,
            blocks_resident_island=False,
            classification="summary_safe",
            notes="Cost totals and hit/fallback counters are summary-safe; full phenotype expression remains Python-owned.",
        ),
        DependencyAuditRow(
            dependency="checkpoint serialization",
            current_owner="Python runner/pop/env/Agent object graph",
            read_write_frequency="safe boundaries and interruption handling",
            required_by_policy=False,
            required_by_metrics=False,
            can_be_summarized=False,
            can_be_deferred_to_sync=True,
            blocks_resident_island=False,
            classification="sync_point_required",
            notes="No checkpoint schema change; resident windows must materialize before checkpoint.",
        ),
        DependencyAuditRow(
            dependency="Patch/Jungle/Tier3/WDL/Biome optional surfaces",
            current_owner="Python runner plus GridWorld/Patch arrays",
            read_write_frequency="often every tick when enabled",
            required_by_policy=True,
            required_by_metrics=True,
            can_be_summarized=True,
            can_be_deferred_to_sync=False,
            blocks_resident_island=True,
            classification="out_of_scope_surface",
            notes="Out of CXXCORE-17 no-RNG pilot scope; must force fallback when behavior-changing features are enabled.",
        ),
    ]


def dependency_audit_payload() -> dict[str, Any]:
    rows = [row_to_dict(row) for row in dependency_audit_rows()]
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "decision": DECISION,
        "dependency_classes": list(DEPENDENCY_CLASSES),
        "rows": rows,
        "summary": {
            "resident_safe": sum(1 for row in rows if row["classification"] == "resident_safe"),
            "summary_safe": sum(1 for row in rows if row["classification"] == "summary_safe"),
            "sync_point_required": sum(1 for row in rows if row["classification"] == "sync_point_required"),
            "per_tick_python_required": sum(1 for row in rows if row["classification"] == "per_tick_python_required"),
            "out_of_scope_surface": sum(1 for row in rows if row["classification"] == "out_of_scope_surface"),
        },
        "checkpoint_schema_changed": False,
        "metrics_schema_changed": False,
        "native_rng_added": False,
    }


def candidate_boundaries() -> list[CandidateBoundary]:
    return [
        CandidateBoundary(
            candidate="A: env-only live resident window",
            required_state_ownership=("env arrays", "agent scalar deltas"),
            python_object_materialization_frequency="every tick in production",
            expected_speed_economics="failed in CXXCORE-16",
            parity_risk="low in copied/captured windows",
            rng_risk="low for no-RNG slice",
            checkpoint_risk="sync before checkpoint",
            metrics_risk="high",
            implementation_cost="already attempted",
            pilot_in_this_branch=False,
            assessment="insufficient; retained only as baseline",
        ),
        CandidateBoundary(
            candidate="B: env + agent stats + selected diagnostics summary island",
            required_state_ownership=("env arrays", "agent scalar deltas", "stats deltas", "summary tape"),
            python_object_materialization_frequency="sync boundary for supported metrics",
            expected_speed_economics="plausible if summary tape replaces per-tick object reads",
            parity_risk="medium",
            rng_risk="low for no-RNG/precomputed arrays",
            checkpoint_risk="sync before checkpoint",
            metrics_risk="medium",
            implementation_cost="medium",
            pilot_in_this_branch=True,
            assessment="best next architectural step",
        ),
        CandidateBoundary(
            candidate="C: env + drives/chemistry tick-state island",
            required_state_ownership=("env arrays", "stats", "runner drives", "chemistry state"),
            python_object_materialization_frequency="sync boundary only if drives/chemistry are rehomed",
            expected_speed_economics="possible but too invasive for this branch",
            parity_risk="high",
            rng_risk="medium",
            checkpoint_risk="medium",
            metrics_risk="high",
            implementation_cost="high",
            pilot_in_this_branch=False,
            assessment="defer until summary tape proves value",
        ),
        CandidateBoundary(
            candidate="D: action-precomputed runner window",
            required_state_ownership=("precomputed actions", "cost arrays", "env arrays", "summary tape"),
            python_object_materialization_frequency="window boundary",
            expected_speed_economics="plausible for harness/captured windows, not live policy semantics yet",
            parity_risk="medium",
            rng_risk="low if RNG arrays are Python-precomputed",
            checkpoint_risk="sync before checkpoint",
            metrics_risk="medium",
            implementation_cost="medium",
            pilot_in_this_branch=True,
            assessment="valid reference pilot; not a production live-runner path",
        ),
        CandidateBoundary(
            candidate="E: larger generation island",
            required_state_ownership=("policy/action interface", "env", "stats", "summaries", "safe checkpoints"),
            python_object_materialization_frequency="generation or checkpoint boundary",
            expected_speed_economics="highest upside but currently broad",
            parity_risk="high",
            rng_risk="high unless all RNG streams are precomputed/owned",
            checkpoint_risk="high",
            metrics_risk="high",
            implementation_cost="very high",
            pilot_in_this_branch=False,
            assessment="requires CXXCORE-18+ predesign, not immediate implementation",
        ),
    ]


def row_to_dict(row: DependencyAuditRow) -> dict[str, Any]:
    return {
        "dependency": row.dependency,
        "current_owner": row.current_owner,
        "read_write_frequency": row.read_write_frequency,
        "required_by_policy": bool(row.required_by_policy),
        "required_by_metrics": bool(row.required_by_metrics),
        "can_be_summarized": bool(row.can_be_summarized),
        "can_be_deferred_to_sync": bool(row.can_be_deferred_to_sync),
        "blocks_resident_island": bool(row.blocks_resident_island),
        "classification": row.classification,
        "notes": row.notes,
    }


def candidate_to_dict(candidate: CandidateBoundary) -> dict[str, Any]:
    return {
        "candidate": candidate.candidate,
        "required_state_ownership": list(candidate.required_state_ownership),
        "python_object_materialization_frequency": candidate.python_object_materialization_frequency,
        "expected_speed_economics": candidate.expected_speed_economics,
        "parity_risk": candidate.parity_risk,
        "rng_risk": candidate.rng_risk,
        "checkpoint_risk": candidate.checkpoint_risk,
        "metrics_risk": candidate.metrics_risk,
        "implementation_cost": candidate.implementation_cost,
        "pilot_in_this_branch": bool(candidate.pilot_in_this_branch),
        "assessment": candidate.assessment,
    }


def candidate_boundary_payload() -> dict[str, Any]:
    return {
        "schema_version": "cxxcore17_candidate_boundaries_v1",
        "decision": DECISION,
        "candidates": [candidate_to_dict(candidate) for candidate in candidate_boundaries()],
    }


def make_summary_tape(
    *,
    source: str,
    actions_by_tick: Any,
    pre_energy: Any,
    pre_health: Any,
    pre_injury: Any,
    pre_scar: Any,
    result_output: Mapping[str, Any],
    food_gained: Any | None = None,
    moved_mask: Any | None = None,
    sparse_ticks: Any | None = None,
    sparse_summaries: Any | None = None,
    sync_interval: int = 0,
) -> RunnerTickSummaryTape:
    actions = np.asarray(actions_by_tick, dtype=np.int32)
    if actions.ndim != 2:
        raise ValueError("actions_by_tick must be a 2D array")
    pre_e = np.asarray(pre_energy, dtype=np.float32)
    pre_h = np.asarray(pre_health, dtype=np.float32)
    pre_i = np.asarray(pre_injury, dtype=np.float32)
    pre_s = np.asarray(pre_scar, dtype=np.float32)
    out_e = np.asarray(result_output["new_energy"], dtype=np.float32)
    out_h = np.asarray(result_output["new_health"], dtype=np.float32)
    out_i = np.asarray(result_output["new_injury"], dtype=np.float32)
    out_s = np.asarray(result_output["new_scar"], dtype=np.float32)
    died = np.asarray(result_output.get("died_mask", np.zeros_like(out_e, dtype=bool)), dtype=bool)
    cost_total = float(np.asarray(result_output.get("cost_total", [0.0]), dtype=np.float64).reshape(-1)[0])
    food = np.asarray(food_gained if food_gained is not None else np.zeros_like(out_e), dtype=np.float32)
    moved = np.asarray(moved_mask if moved_mask is not None else np.zeros_like(out_e, dtype=bool), dtype=bool)
    sparse = _sparse_summary_rows(sparse_ticks=sparse_ticks, sparse_summaries=sparse_summaries)
    payload = {
        "schema_version": SUMMARY_TAPE_SCHEMA_VERSION,
        "source": str(source),
        "ticks": int(actions.shape[0]),
        "agents": int(actions.shape[1]),
        "sync_interval": int(sync_interval or 0),
        "energy_delta_total": float(np.sum(out_e - pre_e, dtype=np.float64)),
        "health_delta_total": float(np.sum(out_h - pre_h, dtype=np.float64)),
        "injury_delta_total": float(np.sum(out_i - pre_i, dtype=np.float64)),
        "scar_delta_total": float(np.sum(out_s - pre_s, dtype=np.float64)),
        "death_count": int(np.count_nonzero(died)),
        "food_gained_total": float(np.sum(food, dtype=np.float64)),
        "moved_count": int(np.count_nonzero(moved)),
        "cost_total": cost_total,
        "action_counts": _action_counts(actions),
        "sparse_summaries": sparse,
        "final_state_digest": state_digest(dict(result_output)),
    }
    digest_payload = dict(payload)
    digest_payload.pop("source", None)
    digest = _stable_json_digest(digest_payload)
    return RunnerTickSummaryTape(digest=digest, **payload)


def summary_tape_payload(tape: RunnerTickSummaryTape | None) -> dict[str, Any] | None:
    if tape is None:
        return None
    return {
        "schema_version": tape.schema_version,
        "source": tape.source,
        "ticks": int(tape.ticks),
        "agents": int(tape.agents),
        "sync_interval": int(tape.sync_interval),
        "energy_delta_total": float(tape.energy_delta_total),
        "health_delta_total": float(tape.health_delta_total),
        "injury_delta_total": float(tape.injury_delta_total),
        "scar_delta_total": float(tape.scar_delta_total),
        "death_count": int(tape.death_count),
        "food_gained_total": float(tape.food_gained_total),
        "moved_count": int(tape.moved_count),
        "cost_total": float(tape.cost_total),
        "action_counts": dict(tape.action_counts),
        "sparse_summaries": list(tape.sparse_summaries),
        "final_state_digest": tape.final_state_digest,
        "digest": tape.digest,
    }


def compare_summary_tapes(expected: RunnerTickSummaryTape, candidate: RunnerTickSummaryTape) -> dict[str, Any]:
    first = None
    expected_payload = summary_tape_payload(expected) or {}
    candidate_payload = summary_tape_payload(candidate) or {}
    for key in sorted(set(expected_payload) | set(candidate_payload)):
        if key == "source":
            continue
        if expected_payload.get(key) != candidate_payload.get(key):
            first = {
                "field": key,
                "python": expected_payload.get(key),
                "candidate": candidate_payload.get(key),
                "classification": "summary-tape",
                "reason": "value_mismatch",
            }
            break
    return {"ok": first is None, "first_divergence": first}


def run_python_summary_reference(
    inputs: Any,
    *,
    sparse_interval: int = 0,
    inject_divergence: bool = False,
) -> RunnerIslandResult:
    t0 = time.perf_counter()
    state = as_resident_state(inputs)
    result = run_resident_python_reference(
        state,
        config=ResidentSoAConfig(sparse_interval=int(sparse_interval or 0)),
        inject_divergence=inject_divergence,
    )
    tape = tape_from_resident_result("python_summary_island_reference", state, result, sparse_interval=sparse_interval)
    parity = compare_resident_to_reference(inputs, result)
    if inject_divergence:
        parity = dict(parity)
        parity["summary_tape"] = compare_summary_tapes(
            tape_from_resident_result(
                "python_summary_island_reference",
                state,
                run_resident_python_reference(state, config=ResidentSoAConfig(sparse_interval=int(sparse_interval or 0))),
                sparse_interval=sparse_interval,
            ),
            tape,
        )
    else:
        parity = dict(parity)
        parity["summary_tape"] = {"ok": True, "first_divergence": None}
    return RunnerIslandResult(
        schema_version=SCHEMA_VERSION,
        mode="python_summary_reference",
        decision=DECISION,
        status="ok" if bool(parity.get("ok")) and bool(parity["summary_tape"].get("ok")) else "divergence",
        parity=parity,
        fallback_reason=None if bool(parity.get("ok")) else "divergence_detected",
        first_divergence=parity.get("first_divergence") or parity["summary_tape"].get("first_divergence"),
        summary_tape=tape,
        python_env_step_skipped=True,
        materializations=1,
        timing={
            "wall_s": float(time.perf_counter() - t0),
            "policy_action_s": 0.0,
            "python_env_s": 0.0,
            "materialize_writeback_s": 0.0,
            "summary_s": 0.0,
            "native_s": 0.0,
            "compare_s": 0.0,
            "fallback_restore_s": 0.0,
        },
        diagnostics={"candidate_boundary": "B/D captured summary island"},
    )


def run_native_summary_shadow(
    inputs: Any,
    *,
    native_backend: Callable[..., Mapping[str, Any]] | None,
    sparse_interval: int = 0,
    compare: bool = True,
    inject_divergence: bool = False,
    raise_exception: bool = False,
) -> RunnerIslandResult:
    if native_backend is None:
        return _fallback_result("native_summary_shadow", "native_missing")
    state = as_resident_state(inputs)
    t0 = time.perf_counter()
    try:
        if raise_exception:
            raise RuntimeError("injected native exception")
        native_t0 = time.perf_counter()
        native = native_backend(inputs, sparse_interval=int(sparse_interval or 0), export_final=True)
        native_s = time.perf_counter() - native_t0
        native_result = _resident_result_from_native(native)
        if inject_divergence:
            native_result.output["new_energy"] = np.asarray(native_result.output["new_energy"], dtype=np.float32).copy()
            native_result.output["new_energy"][0] = np.float32(float(native_result.output["new_energy"][0]) + 1.0)
        summary_t0 = time.perf_counter()
        tape = tape_from_resident_result("native_summary_island_shadow", state, native_result, sparse_interval=sparse_interval)
        summary_s = time.perf_counter() - summary_t0
        compare_t0 = time.perf_counter()
        parity = compare_resident_to_reference(inputs, native_result) if compare else {"ok": True, "first_divergence": None}
        if compare:
            py_result = run_resident_python_reference(
                state,
                config=ResidentSoAConfig(sparse_interval=int(sparse_interval or 0)),
            )
            parity = dict(parity)
            parity["summary_tape"] = compare_summary_tapes(
                tape_from_resident_result("python_summary_island_reference", state, py_result, sparse_interval=sparse_interval),
                tape,
            )
        else:
            parity = dict(parity)
            parity["summary_tape"] = {"ok": True, "first_divergence": None}
        compare_s = time.perf_counter() - compare_t0
        ok = bool(parity.get("ok")) and bool(parity["summary_tape"].get("ok"))
        first = parity.get("first_divergence") or parity["summary_tape"].get("first_divergence")
        return RunnerIslandResult(
            schema_version=SCHEMA_VERSION,
            mode="native_summary_shadow" if compare else "native_summary_authoritative_no_compare_lower_bound",
            decision=DECISION,
            status="ok" if ok else "divergence",
            parity=parity,
            fallback_reason=None if ok else "divergence_detected",
            first_divergence=first,
            summary_tape=tape,
            python_env_step_skipped=True,
            materializations=1,
            timing={
                "wall_s": float(time.perf_counter() - t0),
                "policy_action_s": 0.0,
                "python_env_s": 0.0,
                "materialize_writeback_s": 0.0,
                "summary_s": float(summary_s),
                "native_s": float(native_s),
                "compare_s": float(compare_s),
                "fallback_restore_s": 0.0,
            },
            diagnostics={"candidate_boundary": "B/D captured native summary island"},
        )
    except Exception as exc:
        return _fallback_result("native_summary_shadow", "native_exception", exception=str(exc))


def result_payload(result: RunnerIslandResult) -> dict[str, Any]:
    return {
        "schema_version": result.schema_version,
        "mode": result.mode,
        "decision": result.decision,
        "status": result.status,
        "parity": result.parity,
        "fallback_reason": result.fallback_reason,
        "first_divergence": result.first_divergence,
        "summary_tape": summary_tape_payload(result.summary_tape),
        "python_env_step_skipped": bool(result.python_env_step_skipped),
        "materializations": int(result.materializations),
        "timing": dict(result.timing),
        "diagnostics": dict(result.diagnostics),
    }


def tape_from_resident_result(
    source: str,
    state: Any,
    result: ResidentRunResult,
    *,
    sparse_interval: int = 0,
) -> RunnerTickSummaryTape:
    return make_summary_tape(
        source=source,
        actions_by_tick=state.actions_by_tick,
        pre_energy=state.energy,
        pre_health=state.health,
        pre_injury=state.injury,
        pre_scar=state.scar,
        result_output=result.output,
        food_gained=result.food_gained,
        moved_mask=result.moved_mask,
        sparse_ticks=result.sparse_ticks,
        sparse_summaries=result.sparse_summaries,
        sync_interval=int(sparse_interval or 0),
    )


def _resident_result_from_native(native: Mapping[str, Any]) -> ResidentRunResult:
    return ResidentRunResult(
        schema_version="cxxcore15_resident_soa_tick_island_v1",
        output=dict(native["output"]),
        agent_grid=np.asarray(native["agent_grid"], dtype=np.int32),
        food_gained=np.asarray(native["food_gained"], dtype=np.float32),
        moved_mask=np.asarray(native["moved_mask"], dtype=bool),
        views_dirty=bool(native.get("views_dirty", True)),
        sparse_ticks=np.asarray(native.get("sparse_ticks", []), dtype=np.int32),
        sparse_summaries=np.asarray(native.get("sparse_summaries", []), dtype=np.float64).reshape(
            (-1, 7) if np.asarray(native.get("sparse_summaries", [])).size else (0, 7)
        ),
        final_summary=dict(native.get("final_summary", {})),
        diagnostics={"native_timing": dict(native.get("timing", {}))},
    )


def _fallback_result(mode: str, reason: str, *, exception: str | None = None) -> RunnerIslandResult:
    return RunnerIslandResult(
        schema_version=SCHEMA_VERSION,
        mode=mode,
        decision=DECISION,
        status="fallback",
        parity={"ok": False, "first_divergence": None},
        fallback_reason=reason,
        first_divergence=None,
        summary_tape=None,
        python_env_step_skipped=False,
        materializations=0,
        timing={
            "wall_s": 0.0,
            "policy_action_s": 0.0,
            "python_env_s": 0.0,
            "materialize_writeback_s": 0.0,
            "summary_s": 0.0,
            "native_s": 0.0,
            "compare_s": 0.0,
            "fallback_restore_s": 0.0,
        },
        diagnostics={"exception": exception} if exception else {},
    )


def _action_counts(actions: np.ndarray) -> dict[str, int]:
    flat = np.asarray(actions, dtype=np.int32).reshape(-1)
    out: dict[str, int] = {}
    if flat.size:
        vals, counts = np.unique(flat, return_counts=True)
        out = {str(int(v)): int(c) for v, c in zip(vals.tolist(), counts.tolist())}
    return out


def _sparse_summary_rows(*, sparse_ticks: Any | None, sparse_summaries: Any | None) -> list[dict[str, float | int]]:
    if sparse_ticks is None or sparse_summaries is None:
        return []
    ticks = np.asarray(sparse_ticks, dtype=np.int32).reshape(-1)
    summaries = np.asarray(sparse_summaries, dtype=np.float64)
    if summaries.size == 0:
        return []
    summaries = summaries.reshape((ticks.shape[0], -1))
    rows: list[dict[str, float | int]] = []
    names = ("energy_sum", "energy_min", "energy_max", "died_count", "pos_checksum", "grid_sum", "cost_total")
    for i, tick in enumerate(ticks.tolist()):
        row: dict[str, float | int] = {"tick": int(tick)}
        for j, name in enumerate(names[: summaries.shape[1]]):
            value = float(summaries[i, j])
            row[name] = int(value) if name in {"died_count", "pos_checksum", "grid_sum"} else value
        rows.append(row)
    return rows


def _stable_json_digest(payload: Mapping[str, Any]) -> str:
    data = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=_json_default)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    raise TypeError(f"unsupported JSON value: {type(value)!r}")
