from __future__ import annotations

"""CXXCORE-18 live summary/event-tape metrics helpers.

Default-off diagnostic code. Python remains authoritative for simulation
behavior. The tape is built from authoritative Python pre/post tick state and
can feed selected diagnostics without making those diagnostics read post-env
Agent objects directly.
"""

from dataclasses import dataclass
import hashlib
import json
import time
from typing import Any, Mapping, Sequence

import numpy as np

from env.gridworld_pkg.constants import (
    ACT_ATTACK,
    ACT_GATHER,
    ACT_MOVE_E,
    ACT_MOVE_N,
    ACT_MOVE_S,
    ACT_MOVE_W,
    ACT_REST,
    N_ACTIONS,
)


SCHEMA_VERSION = "cxxcore18_live_summary_tape_metrics_v1"
TAPE_SCHEMA_VERSION = "cxxcore18_live_tick_summary_tape_v1"
AUDIT_SCHEMA_VERSION = "cxxcore18_live_metrics_coupling_audit_v1"
DECISION = "CXXCORE18-A-LIVE-SUMMARY-TAPE-METRICS-GO"

CONSUMER_MEMORY_ACTION = "memory_action_window"
CONSUMER_PREDICTION_DEMAND = "prediction_demand_action_context"

MODES = (
    "off",
    "summary_tape_shadow_compare",
    "summary_tape_authoritative_diagnostics",
    "summary_tape_no_compare",
)


@dataclass(frozen=True)
class Cxxcore18SummaryTapeConfig:
    cxxcore_summary_tape_enable: bool = False
    cxxcore_summary_tape_shadow_compare_enable: bool = False
    cxxcore_summary_tape_authoritative_metrics_enable: bool = False
    cxxcore_summary_tape_consumers: tuple[str, ...] = ()
    cxxcore_summary_tape_first_divergence_limit: int = 1
    cxxcore_summary_tape_mode: str = "off"


@dataclass(frozen=True)
class LiveTickPreSnapshot:
    schema_version: str
    generation: int
    tick: int
    agent_ids: np.ndarray
    actions: np.ndarray
    pre_y: np.ndarray
    pre_x: np.ndarray
    pre_energy: np.ndarray
    pre_health: np.ndarray
    pre_injury: np.ndarray
    pre_scar: np.ndarray
    pre_alive: np.ndarray
    memory_phase: tuple[str, ...]
    prediction_contexts: tuple[dict[str, Any] | None, ...]
    lineage_id: tuple[str, ...]
    founder_id: tuple[str, ...]
    effective_memory: np.ndarray
    memory_abs: np.ndarray
    digest: str


@dataclass(frozen=True)
class LiveSummaryTape:
    schema_version: str
    generation: int
    tick: int
    agent_ids: np.ndarray
    actions: np.ndarray
    pre_y: np.ndarray
    pre_x: np.ndarray
    post_y: np.ndarray
    post_x: np.ndarray
    pre_energy: np.ndarray
    post_energy: np.ndarray
    pre_health: np.ndarray
    post_health: np.ndarray
    pre_injury: np.ndarray
    post_injury: np.ndarray
    pre_scar: np.ndarray
    post_scar: np.ndarray
    pre_alive: np.ndarray
    post_alive: np.ndarray
    memory_phase: tuple[str, ...]
    memory_post_phase: tuple[str, ...]
    prediction_contexts: tuple[dict[str, Any] | None, ...]
    lineage_id: tuple[str, ...]
    founder_id: tuple[str, ...]
    effective_memory: np.ndarray
    memory_abs: np.ndarray
    cost_scalar_total: float
    validation_random_walk_override: bool
    digest: str


@dataclass(frozen=True)
class ConsumerResult:
    schema_version: str
    consumer: str
    mode: str
    status: str
    parity: dict[str, Any]
    fallback_reason: str | None
    first_divergence: dict[str, Any] | None
    legacy_rows: list[dict[str, Any]]
    tape_rows: list[dict[str, Any]]
    timing: dict[str, float]


def config_from_sim_cfg(sim_cfg: Any) -> Cxxcore18SummaryTapeConfig:
    consumers = getattr(sim_cfg, "cxxcore_summary_tape_consumers", ()) or ()
    if isinstance(consumers, str):
        consumers_tuple = tuple(part.strip() for part in consumers.split(",") if part.strip())
    else:
        consumers_tuple = tuple(str(part) for part in consumers)
    return Cxxcore18SummaryTapeConfig(
        cxxcore_summary_tape_enable=bool(getattr(sim_cfg, "cxxcore_summary_tape_enable", False)),
        cxxcore_summary_tape_shadow_compare_enable=bool(
            getattr(sim_cfg, "cxxcore_summary_tape_shadow_compare_enable", False)
        ),
        cxxcore_summary_tape_authoritative_metrics_enable=bool(
            getattr(sim_cfg, "cxxcore_summary_tape_authoritative_metrics_enable", False)
        ),
        cxxcore_summary_tape_consumers=consumers_tuple,
        cxxcore_summary_tape_first_divergence_limit=max(
            1,
            int(getattr(sim_cfg, "cxxcore_summary_tape_first_divergence_limit", 1) or 1),
        ),
        cxxcore_summary_tape_mode=str(getattr(sim_cfg, "cxxcore_summary_tape_mode", "off") or "off"),
    )


def coupling_audit_payload() -> dict[str, Any]:
    rows = [
        _audit_row(
            "memory_action_window pre/post diagnostics",
            "MemoryActionWindowInstrumentation",
            "pre_env and post_env when enabled",
            "post_env writes JSON rows",
            "memory_action_window event and summary files",
            True,
            False,
            "summary_safe_with_sync",
            "Tape must include pre/post phase, position, health, lineage, action histogram, and memory summary inputs.",
        ),
        _audit_row(
            "prediction_demand_action_context pre-env diagnostics",
            "PredictionDemandActionContextInstrumentation",
            "pre_env when prediction-demand events exist",
            "generation summary flush",
            "prediction_demand action-context summaries",
            True,
            False,
            "summary_safe_with_sync",
            "Tape must include nearest event context and movement classification inputs captured at pre-env time.",
        ),
        _audit_row(
            "Agent.stats last deltas",
            "TickEnvApplyMixin Agent.stats writeback",
            "every env tick",
            "every env tick",
            "generation metrics and downstream drive inputs",
            True,
            True,
            "state_backend_required",
            "Can be summarized, but replacing object writes would affect drives and later tick policy state.",
        ),
        _audit_row(
            "behavior logger action rows",
            "BehaviorLogger",
            "every tick",
            "every tick",
            "behavior log artifacts",
            True,
            False,
            "summary_safe",
            "Action totals are tape-safe, but this branch leaves behavior logger unchanged.",
        ),
        _audit_row(
            "cost-charge expression diagnostics",
            "agents.cognition and EnvHotpathExpressionDiagnostics",
            "during env apply when cost charging enabled",
            "during env apply",
            "perf/debug diagnostics",
            True,
            True,
            "summary_safe_with_sync",
            "Totals can be emitted from tape after upstream Python cost computation; full phenotype expression remains Python-owned.",
        ),
        _audit_row(
            "internal drives",
            "runner dense drive arrays and Agent compatibility mirrors",
            "after every env tick",
            "after every env tick",
            "policy state and metrics",
            False,
            True,
            "drive_or_chemistry_required",
            "Not just diagnostics; changing ownership changes policy-facing state.",
        ),
        _audit_row(
            "chemistry state",
            "Agent chemistry objects",
            "after every env tick and cost multiplier path",
            "after every env tick",
            "policy/cost coupling and metrics",
            False,
            True,
            "drive_or_chemistry_required",
            "Object state ownership is out of CXXCORE-18 scope.",
        ),
        _audit_row(
            "checkpoint serialization",
            "Python runner/pop/env object graph",
            "safe boundaries",
            "checkpoint writes",
            "checkpoint files",
            False,
            True,
            "checkpoint_required",
            "No checkpoint schema change; tape is diagnostic and not checkpoint-owned.",
        ),
        _audit_row(
            "Patch/Jungle/Tier3/WDL/Biome behavior surfaces",
            "Python runner and GridWorld/Patch arrays",
            "often every tick when enabled",
            "often every tick",
            "metrics plus behavior",
            True,
            True,
            "out_of_scope",
            "CXXCORE-18 only consumes selected diagnostic summaries; behavior ownership remains Python.",
        ),
    ]
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "decision": DECISION,
        "rows": rows,
        "checkpoint_schema_changed": False,
        "metrics_schema_changed": False,
        "native_rng_added": False,
    }


def capture_pre_tick_snapshot(
    runner: Any,
    *,
    generation: int,
    tick: int,
    alive: Sequence[Any],
    agent_ids_np: Any,
    actions_np: Any,
) -> LiveTickPreSnapshot:
    agent_ids = np.asarray(agent_ids_np, dtype=np.int32).reshape(-1)
    actions = np.asarray(actions_np, dtype=np.int32).reshape(-1)
    if agent_ids.shape[0] != actions.shape[0]:
        raise ValueError("agent_ids and actions must have the same length")
    env = runner.env
    pos_y = np.asarray(getattr(env, "agent_pos_y"), dtype=np.int32)
    pos_x = np.asarray(getattr(env, "agent_pos_x"), dtype=np.int32)
    alive_arr = np.asarray(getattr(env, "agent_alive"), dtype=bool)
    alive_by_id = {int(getattr(agent, "agent_id")): agent for agent in alive}
    pre_y = pos_y[agent_ids].astype(np.int32, copy=True)
    pre_x = pos_x[agent_ids].astype(np.int32, copy=True)
    pre_alive = alive_arr[agent_ids].astype(bool, copy=True)
    pre_energy = np.asarray([float(getattr(alive_by_id.get(int(aid)), "energy", 0.0)) for aid in agent_ids], dtype=np.float32)
    pre_health = np.asarray([float(getattr(alive_by_id.get(int(aid)), "health", 0.0)) for aid in agent_ids], dtype=np.float32)
    pre_injury = np.asarray([float(getattr(alive_by_id.get(int(aid)), "injury", 0.0)) for aid in agent_ids], dtype=np.float32)
    pre_scar = np.asarray([float(getattr(alive_by_id.get(int(aid)), "scar", 0.0)) for aid in agent_ids], dtype=np.float32)
    phases = tuple(_memory_phase_at(env, int(y), int(x)) for y, x in zip(pre_y.tolist(), pre_x.tolist()))
    prediction_contexts = tuple(
        _nearest_prediction_context(runner, int(y), int(x)) for y, x in zip(pre_y.tolist(), pre_x.tolist())
    )
    lineage = tuple(str(getattr(alive_by_id.get(int(aid)), "lineage_id", "") or getattr(alive_by_id.get(int(aid)), "founder_id", "") or "") for aid in agent_ids)
    founders = tuple(str(getattr(alive_by_id.get(int(aid)), "founder_id", "") or "") for aid in agent_ids)
    effective_memory = np.asarray(
        [_effective_memory_value(runner, alive_by_id.get(int(aid))) for aid in agent_ids],
        dtype=np.float32,
    )
    memory_abs = np.asarray([_memory_abs_value(runner, alive_by_id.get(int(aid))) for aid in agent_ids], dtype=np.float32)
    payload = {
        "schema_version": "cxxcore18_live_tick_pre_snapshot_v1",
        "generation": int(generation),
        "tick": int(tick),
        "agent_ids": agent_ids,
        "actions": actions,
        "pre_y": pre_y,
        "pre_x": pre_x,
        "pre_energy": pre_energy,
        "pre_health": pre_health,
        "pre_injury": pre_injury,
        "pre_scar": pre_scar,
        "pre_alive": pre_alive,
        "memory_phase": phases,
        "prediction_contexts": prediction_contexts,
        "lineage_id": lineage,
        "founder_id": founders,
        "effective_memory": effective_memory,
        "memory_abs": memory_abs,
    }
    return LiveTickPreSnapshot(digest=_stable_json_digest(payload), **payload)


def build_live_summary_tape(
    runner: Any,
    snapshot: LiveTickPreSnapshot,
    *,
    cost_scalar_total: float = 0.0,
    validation_random_walk_override: bool = False,
) -> LiveSummaryTape:
    env = runner.env
    agent_ids = snapshot.agent_ids
    pos_y = np.asarray(getattr(env, "agent_pos_y"), dtype=np.int32)
    pos_x = np.asarray(getattr(env, "agent_pos_x"), dtype=np.int32)
    alive_arr = np.asarray(getattr(env, "agent_alive"), dtype=bool)
    agents = getattr(getattr(runner, "pop", None), "agents", {})
    post_y = pos_y[agent_ids].astype(np.int32, copy=True)
    post_x = pos_x[agent_ids].astype(np.int32, copy=True)
    post_alive = alive_arr[agent_ids].astype(bool, copy=True)
    post_energy = np.asarray([float(getattr(agents.get(int(aid)), "energy", 0.0)) for aid in agent_ids], dtype=np.float32)
    post_health = np.asarray([float(getattr(agents.get(int(aid)), "health", 0.0)) for aid in agent_ids], dtype=np.float32)
    post_injury = np.asarray([float(getattr(agents.get(int(aid)), "injury", 0.0)) for aid in agent_ids], dtype=np.float32)
    post_scar = np.asarray([float(getattr(agents.get(int(aid)), "scar", 0.0)) for aid in agent_ids], dtype=np.float32)
    post_phases = tuple(_memory_phase_at(env, int(y), int(x)) for y, x in zip(post_y.tolist(), post_x.tolist()))
    payload = {
        "schema_version": TAPE_SCHEMA_VERSION,
        "generation": int(snapshot.generation),
        "tick": int(snapshot.tick),
        "agent_ids": agent_ids.copy(),
        "actions": snapshot.actions.copy(),
        "pre_y": snapshot.pre_y.copy(),
        "pre_x": snapshot.pre_x.copy(),
        "post_y": post_y,
        "post_x": post_x,
        "pre_energy": snapshot.pre_energy.copy(),
        "post_energy": post_energy,
        "pre_health": snapshot.pre_health.copy(),
        "post_health": post_health,
        "pre_injury": snapshot.pre_injury.copy(),
        "post_injury": post_injury,
        "pre_scar": snapshot.pre_scar.copy(),
        "post_scar": post_scar,
        "pre_alive": snapshot.pre_alive.copy(),
        "post_alive": post_alive,
        "memory_phase": snapshot.memory_phase,
        "memory_post_phase": post_phases,
        "prediction_contexts": snapshot.prediction_contexts,
        "lineage_id": snapshot.lineage_id,
        "founder_id": snapshot.founder_id,
        "effective_memory": snapshot.effective_memory.copy(),
        "memory_abs": snapshot.memory_abs.copy(),
        "cost_scalar_total": float(cost_scalar_total),
        "validation_random_walk_override": bool(validation_random_walk_override),
    }
    return LiveSummaryTape(digest=_stable_json_digest(payload), **payload)


def memory_action_rows_from_tape(instrumentation: Any, runner: Any, tape: LiveSummaryTape) -> list[dict[str, Any]]:
    _validate_tape(tape)
    rows: dict[str, dict[str, Any]] = {}
    for i, aid_raw in enumerate(tape.agent_ids.tolist()):
        aid = int(aid_raw)
        phase = str(tape.memory_phase[i])
        row = rows.setdefault(
            phase,
            instrumentation._empty_row(runner, generation=int(tape.generation), tick=int(tape.tick), phase=phase),
        )
        action = int(tape.actions[i])
        row["_agent_ids"].append(aid)
        row.setdefault("_pre_positions", {})[aid] = [int(tape.pre_y[i]), int(tape.pre_x[i])]
        row["_pre_health"][aid] = float(tape.pre_health[i])
        row["agent_count"] += 1
        row["action_histogram"][str(action)] = int(row["action_histogram"].get(str(action), 0)) + 1
        if action in (ACT_MOVE_N, ACT_MOVE_S, ACT_MOVE_E, ACT_MOVE_W):
            row["movement_action_count"] += 1
        elif action == ACT_GATHER:
            row["gather_action_count"] += 1
        elif action == ACT_REST:
            row["idle_noop_count"] += 1
        elif action < 0 or action >= N_ACTIONS:
            row["unknown_other_action_count"] += 1
        if bool(getattr(instrumentation, "include_lineage", True)):
            lineage = str(tape.lineage_id[i])
            if lineage:
                row["_lineage_counts"][lineage] += 1
            founder = str(tape.founder_id[i])
            if founder:
                row["_founder_ids"].add(founder)
        if bool(getattr(instrumentation, "include_memory_state", True)):
            row["_effective_memory_values"].append(float(tape.effective_memory[i]))
            row["_memory_abs_values"].append(float(tape.memory_abs[i]))
        if float(tape.post_health[i]) < float(tape.pre_health[i]):
            row["exposure_damage_summary"]["damage_event_count"] += 1
            row["exposure_damage_summary"]["damage_total"] += float(float(tape.pre_health[i]) - float(tape.post_health[i]))
            if phase in ("cue", "hidden_gap", "active_hazard"):
                row["exposure_damage_summary"]["damage_with_prior_cue_context_count"] += 1
        if phase == "active_hazard":
            row["exposure_damage_summary"]["active_hazard_exposure_count"] += 1
        if bool(tape.post_alive[i]):
            post_phase = str(tape.memory_post_phase[i])
            if phase != "active_hazard" and post_phase == "active_hazard":
                row["exposure_damage_summary"]["hazard_entry_count"] += 1
            if phase == "active_hazard" and post_phase != "active_hazard":
                row["exposure_damage_summary"]["hazard_exit_count"] += 1
            if phase in ("cue", "hidden_gap") and post_phase == "baseline":
                row["exposure_damage_summary"]["pre_hazard_exit_count"] += 1
    out = []
    for phase in sorted(rows):
        row = rows[phase]
        row.pop("_agent_ids", None)
        row.pop("_pre_positions", None)
        row.pop("_pre_health", None)
        instrumentation._finalize_row(row)
        out.append(row)
    return out


def prediction_demand_rows_from_tape(instrumentation: Any, runner: Any, tape: LiveSummaryTape) -> list[dict[str, Any]]:
    _validate_tape(tape)
    rows: dict[tuple[int, str], dict[str, Any]] = {}
    for i, aid_raw in enumerate(tape.agent_ids.tolist()):
        ctx = tape.prediction_contexts[i]
        if not ctx:
            continue
        aid = int(aid_raw)
        action = int(tape.actions[i])
        label = _label_for_action(action)
        key = (int(ctx["event_id"]), str(ctx["window_label"]))
        row = rows.setdefault(
            key,
            instrumentation._empty_row(runner, generation=int(tape.generation), tick=int(tape.tick), context=ctx),
        )
        row["exposure_count"] += 1
        row["_agent_ids"].add(aid)
        row["action_label_counts"][label] += 1
        if label == "GATHER":
            row["gather_attempts_near_region"] += 1
        grid = np.asarray(getattr(runner.env, "grid"))
        instrumentation._record_movement(
            row,
            action=action,
            y=int(tape.pre_y[i]),
            x=int(tape.pre_x[i]),
            context=ctx,
            grid_shape=tuple(int(v) for v in grid.shape[:2]),
        )
        if bool(getattr(instrumentation, "include_lineage", True)):
            lineage = str(tape.lineage_id[i])
            if lineage:
                row["_lineage_counts"][lineage] += 1
    out = []
    for key in sorted(rows, key=lambda item: (item[0], item[1])):
        row = rows[key]
        instrumentation._finalize_row(row)
        out.append(row)
    return out


def compare_consumer_rows(
    legacy_rows: Sequence[Mapping[str, Any]],
    tape_rows: Sequence[Mapping[str, Any]],
    *,
    consumer: str,
    limit: int = 1,
) -> dict[str, Any]:
    left = [_normalize_row(row) for row in legacy_rows]
    right = [_normalize_row(row) for row in tape_rows]
    divergences: list[dict[str, Any]] = []
    if len(left) != len(right):
        divergences.append(
            {
                "consumer": consumer,
                "field": "row_count",
                "row_index": None,
                "python": len(left),
                "tape": len(right),
                "reason": "row_count_mismatch",
            }
        )
    for idx, (a, b) in enumerate(zip(left, right)):
        if a == b:
            continue
        field = _first_mismatch_field(a, b)
        divergences.append(
            {
                "consumer": consumer,
                "field": field,
                "row_index": int(idx),
                "python": a.get(field),
                "tape": b.get(field),
                "reason": "value_mismatch",
            }
        )
        if len(divergences) >= int(limit):
            break
    return {
        "ok": not divergences,
        "first_divergence": divergences[0] if divergences else None,
        "divergence_count": len(divergences),
        "legacy_row_count": len(left),
        "tape_row_count": len(right),
    }


def run_consumer_shadow_compare(
    *,
    consumer: str,
    instrumentation: Any,
    runner: Any,
    tape: LiveSummaryTape | None,
    legacy_rows: Sequence[Mapping[str, Any]],
    mode: str,
    inject_divergence: bool = False,
) -> ConsumerResult:
    timing = {"total_s": 0.0, "tape_consumer_s": 0.0, "compare_s": 0.0, "fallback_s": 0.0}
    total_t0 = time.perf_counter()
    if tape is None:
        timing["total_s"] = float(time.perf_counter() - total_t0)
        return _consumer_fallback(consumer, mode, "missing_tape", list(legacy_rows), timing)
    try:
        t0 = time.perf_counter()
        if consumer == CONSUMER_MEMORY_ACTION:
            tape_rows = memory_action_rows_from_tape(instrumentation, runner, tape)
        elif consumer == CONSUMER_PREDICTION_DEMAND:
            tape_rows = prediction_demand_rows_from_tape(instrumentation, runner, tape)
        else:
            return _consumer_fallback(consumer, mode, "unsupported_consumer", list(legacy_rows), timing)
        if inject_divergence and tape_rows:
            tape_rows = [dict(row) for row in tape_rows]
            tape_rows[0] = dict(tape_rows[0])
            tape_rows[0]["agent_count"] = int(tape_rows[0].get("agent_count", 0)) + 1
        timing["tape_consumer_s"] = float(time.perf_counter() - t0)
        cmp_t0 = time.perf_counter()
        parity = compare_consumer_rows(
            list(legacy_rows),
            tape_rows,
            consumer=consumer,
            limit=int(getattr(getattr(runner, "sim_cfg", None), "cxxcore_summary_tape_first_divergence_limit", 1) or 1),
        )
        timing["compare_s"] = float(time.perf_counter() - cmp_t0)
        timing["total_s"] = float(time.perf_counter() - total_t0)
        status = "ok" if parity["ok"] else "divergence"
        return ConsumerResult(
            schema_version=SCHEMA_VERSION,
            consumer=consumer,
            mode=mode,
            status=status,
            parity=parity,
            fallback_reason=None if parity["ok"] else "divergence_detected",
            first_divergence=parity.get("first_divergence"),
            legacy_rows=[dict(row) for row in legacy_rows],
            tape_rows=tape_rows,
            timing=timing,
        )
    except Exception as exc:
        timing["fallback_s"] = float(time.perf_counter() - total_t0)
        timing["total_s"] = float(time.perf_counter() - total_t0)
        return _consumer_fallback(consumer, mode, "malformed_tape", list(legacy_rows), timing, exception=str(exc))


def result_payload(result: ConsumerResult) -> dict[str, Any]:
    return {
        "schema_version": result.schema_version,
        "consumer": result.consumer,
        "mode": result.mode,
        "status": result.status,
        "parity": result.parity,
        "fallback_reason": result.fallback_reason,
        "first_divergence": result.first_divergence,
        "legacy_row_count": len(result.legacy_rows),
        "tape_row_count": len(result.tape_rows),
        "timing": dict(result.timing),
    }


def tape_payload(tape: LiveSummaryTape) -> dict[str, Any]:
    return {
        "schema_version": tape.schema_version,
        "generation": int(tape.generation),
        "tick": int(tape.tick),
        "rows": int(tape.agent_ids.shape[0]),
        "digest": tape.digest,
        "cost_scalar_total": float(tape.cost_scalar_total),
        "validation_random_walk_override": bool(tape.validation_random_walk_override),
    }


def _audit_row(
    dependency: str,
    owner: str,
    read_frequency: str,
    write_frequency: str,
    metric_consumer: str,
    can_be_represented: bool,
    needs_live_object: bool,
    classification: str,
    notes: str,
) -> dict[str, Any]:
    return {
        "dependency": dependency,
        "owner": owner,
        "read_frequency": read_frequency,
        "write_frequency": write_frequency,
        "metric_consumer": metric_consumer,
        "can_be_represented_in_tape": bool(can_be_represented),
        "needs_live_object": bool(needs_live_object),
        "classification": classification,
        "notes": notes,
    }


def _consumer_fallback(
    consumer: str,
    mode: str,
    reason: str,
    legacy_rows: list[Mapping[str, Any]],
    timing: Mapping[str, float],
    *,
    exception: str | None = None,
) -> ConsumerResult:
    return ConsumerResult(
        schema_version=SCHEMA_VERSION,
        consumer=consumer,
        mode=mode,
        status="fallback",
        parity={"ok": False, "first_divergence": None, "legacy_row_count": len(legacy_rows), "tape_row_count": 0},
        fallback_reason=reason,
        first_divergence=None,
        legacy_rows=[dict(row) for row in legacy_rows],
        tape_rows=[],
        timing={**dict(timing), "exception_present": 1.0 if exception else 0.0},
    )


def _validate_tape(tape: LiveSummaryTape) -> None:
    if not isinstance(tape, LiveSummaryTape):
        raise TypeError("expected LiveSummaryTape")
    n = int(tape.agent_ids.shape[0])
    fields = (
        tape.actions,
        tape.pre_y,
        tape.pre_x,
        tape.post_y,
        tape.post_x,
        tape.pre_health,
        tape.post_health,
        tape.pre_alive,
        tape.post_alive,
    )
    if any(int(field.shape[0]) != n for field in fields):
        raise ValueError("malformed tape: row arrays are not aligned")
    if len(tape.memory_phase) != n or len(tape.memory_post_phase) != n or len(tape.prediction_contexts) != n:
        raise ValueError("malformed tape: row metadata is not aligned")


def _memory_phase_at(env: Any, y: int, x: int) -> str:
    try:
        if y < 0 or x < 0:
            return "baseline"
        active = np.asarray(getattr(env, "patch_delayed_hazard_active"), dtype=np.float32)
        cue = np.asarray(getattr(env, "patch_delayed_hazard_cue"), dtype=np.float32)
        timer = np.asarray(getattr(env, "patch_delayed_hazard_timer"), dtype=np.int32)
        prior = np.asarray(getattr(env, "patch_delayed_hazard_prior_cue"), dtype=np.int8)
        if y >= active.shape[0] or x >= active.shape[1]:
            return "baseline"
        if float(active[y, x]) > 0.0:
            return "active_hazard"
        if float(cue[y, x]) > 0.0:
            return "cue"
        if int(timer[y, x]) >= 0 and int(prior[y, x]) > 0:
            return "hidden_gap"
    except Exception:
        return "baseline"
    return "baseline"


def _prediction_contexts(runner: Any) -> list[dict[str, Any]]:
    env = runner.env
    try:
        event_ids = np.asarray(getattr(env, "patch_prediction_demand_event_id"), dtype=np.int64)
    except Exception:
        return []
    if event_ids.size <= 0:
        return []
    cue = np.asarray(getattr(env, "patch_prediction_demand_cue"), dtype=np.float32)
    active = np.asarray(getattr(env, "patch_prediction_demand_active"), dtype=np.float32)
    trend = np.asarray(getattr(env, "patch_prediction_demand_trend"), dtype=np.float32)
    timer = np.asarray(getattr(env, "patch_prediction_demand_timer"), dtype=np.int32)
    age = np.asarray(getattr(env, "patch_prediction_demand_age"), dtype=np.int32)
    delays = np.asarray(getattr(env, "patch_prediction_demand_delay"), dtype=np.int32)
    prior = np.asarray(getattr(env, "patch_prediction_demand_prior_cue"), dtype=np.int8)
    signs = np.asarray(getattr(env, "patch_prediction_demand_transition_sign"), dtype=np.int8)
    out: list[dict[str, Any]] = []
    for event_id in sorted(int(v) for v in np.unique(event_ids).tolist() if int(v) > 0):
        mask = event_ids == np.int64(event_id)
        if not bool(np.any(mask)):
            continue
        ys, xs = np.nonzero(mask)
        cy = int(round(float(np.mean(ys)))) if ys.size else 0
        cx = int(round(float(np.mean(xs)))) if xs.size else 0
        active_any = bool(np.any(active[mask] != 0.0))
        cue_any = bool(np.any(cue[mask] > 0.0))
        min_timer = int(np.min(timer[mask])) if bool(np.any(timer[mask] >= 0)) else -1
        min_age = int(np.min(age[mask])) if age.size else 0
        delay = int(np.max(delays[mask])) if delays.size else 0
        trend_value = float(np.mean(trend[mask])) if trend.size else 0.0
        sign = int(np.sign(float(np.mean(signs[mask])))) if signs.size else _trend_sign(trend_value)
        out.append(
            {
                "event_id": int(event_id),
                "region_center_y": cy,
                "region_center_x": cx,
                "event_radius": _event_radius(mask, cy, cx),
                "window_label": _prediction_window_label(active_any=active_any, cue_any=cue_any, min_timer=min_timer, min_age=min_age),
                "trend_sign": int(sign),
                "configured_delay": int(delay),
                "time_to_consequence": max(0, int(min_timer)) if not active_any else 0,
                "cue_ambiguity": float(max(0.0, 1.0 - abs(float(trend_value)))) if cue_any else 0.0,
                "prior_cue": bool(np.any(prior[mask] > 0)),
            }
        )
    return out


def _nearest_prediction_context(runner: Any, y: int, x: int) -> dict[str, Any] | None:
    contexts = _prediction_contexts(runner)
    if not contexts:
        return None
    radius = max(0, int(getattr(runner.sim_cfg, "prediction_demand_action_context_window_radius", 1) or 1))
    candidates: list[tuple[int, int, dict[str, Any]]] = []
    for ctx in contexts:
        dist = max(abs(int(y) - int(ctx["region_center_y"])), abs(int(x) - int(ctx["region_center_x"])))
        if dist <= int(ctx["event_radius"]) + radius:
            candidates.append((dist, int(ctx["event_id"]), ctx))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1]))
    return candidates[0][2]


def _prediction_window_label(*, active_any: bool, cue_any: bool, min_timer: int, min_age: int) -> str:
    if active_any:
        return "active_consequence"
    if not cue_any:
        return "post_consequence_recovery"
    if int(min_age) <= 1:
        return "cue"
    if int(min_timer) <= 1:
        return "pre_consequence"
    return "waiting"


def _event_radius(mask: np.ndarray, cy: int, cx: int) -> int:
    ys, xs = np.nonzero(mask)
    if ys.size <= 0:
        return 0
    return int(max(max(abs(int(y) - int(cy)), abs(int(x) - int(cx))) for y, x in zip(ys.tolist(), xs.tolist())))


def _effective_memory_value(runner: Any, agent: Any | None) -> float:
    if agent is None:
        return 0.0
    try:
        from agents.cognition import effective_memory_dim_for_genome

        return float(effective_memory_dim_for_genome(getattr(agent, "genome", None), sim_cfg=runner.sim_cfg))
    except Exception:
        return 0.0


def _memory_abs_value(runner: Any, agent: Any | None) -> float:
    if agent is None:
        return 0.0
    try:
        hmem = getattr(runner, "_Hmem", None)
        slot = int(getattr(agent, "slot", -1))
        eff = int(_effective_memory_value(runner, agent))
        if hmem is None or slot < 0 or eff <= 0:
            return 0.0
        state = hmem[int(slot)].detach().to("cpu").numpy()
        return float(np.mean(np.abs(state[:eff])) if eff <= state.shape[0] else np.mean(np.abs(state)))
    except Exception:
        return 0.0


def _label_for_action(action: int) -> str:
    if int(action) == int(ACT_MOVE_N):
        return "MOVE_N"
    if int(action) == int(ACT_MOVE_S):
        return "MOVE_S"
    if int(action) == int(ACT_MOVE_E):
        return "MOVE_E"
    if int(action) == int(ACT_MOVE_W):
        return "MOVE_W"
    if int(action) == int(ACT_GATHER):
        return "GATHER"
    if int(action) == int(ACT_REST):
        return "REST"
    if int(action) == int(ACT_ATTACK):
        return "ATTACK"
    return "UNKNOWN"


def _trend_sign(value: float) -> int:
    if float(value) > 0.0:
        return 1
    if float(value) < 0.0:
        return -1
    return 0


def _normalize_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(row, sort_keys=True, default=_json_default))


def _first_mismatch_field(left: Mapping[str, Any], right: Mapping[str, Any]) -> str:
    for key in sorted(set(left) | set(right)):
        if left.get(key) != right.get(key):
            return str(key)
    return "<unknown>"


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
    if isinstance(value, set):
        return sorted(value)
    raise TypeError(f"unsupported JSON value: {type(value)!r}")
