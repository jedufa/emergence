from __future__ import annotations

"""CXXCORE-22 direct compact summary producer helpers.

Default-off performance engineering path. Python remains authoritative for
simulation state. This module bypasses row-compatible LiveSummaryTape
materialization for compact diagnostic consumers while preserving the CXXCORE-20
compact input/result contract.
"""

from dataclasses import dataclass
import time
from typing import Any, Mapping, Sequence

import numpy as np

from perf.cxxcore18_live_summary_tape_metrics import (
    CONSUMER_MEMORY_ACTION,
    CONSUMER_PREDICTION_DEMAND,
    LiveTickPreSnapshot,
    build_live_summary_tape,
    _effective_memory_value,
    _memory_abs_value,
    _memory_phase_at,
    _prediction_contexts,
)
from perf.cxxcore19_native_summary_tape_resident import PHASE_TO_CODE
from perf.cxxcore20_compact_summary_consumers import (
    CompactSummaryInputs,
    compare_compact_results,
    compare_compact_to_rows,
    prepare_compact_inputs_from_tape,
    run_compact_summary,
    run_python_compact_reference,
    _stable_json_digest,
    WINDOW_TO_CODE,
)


SCHEMA_VERSION = "cxxcore22_summary_tape_build_boundary_v1"
DIRECT_CONTRACT_SCHEMA_VERSION = "cxxcore22_direct_compact_summary_contract_v1"
DIRECT_PRE_SNAPSHOT_SCHEMA_VERSION = "cxxcore22_direct_compact_pre_snapshot_v1"
DECISION_LABELS = (
    "CXXCORE22-A-DIRECT-COMPACT-TAPE-E2E-GO",
    "CXXCORE22-B-COMPACT-TAPE-COMPONENT-GO-NO-E2E",
    "CXXCORE22-C-PARITY-ONLY-NO-SPEED",
    "CXXCORE22-D-BOUNDARY-TOO-WIDE",
    "CXXCORE22-E-INTEGRITY-FAIL",
    "CXXCORE22-F-CLOSE-SUMMARY-TAPE-PERF-SUBTRACK",
)


@dataclass(frozen=True)
class DirectCompactPreSnapshot:
    schema_version: str
    generation: int
    tick: int
    agent_ids: np.ndarray
    actions: np.ndarray
    pre_y: np.ndarray
    pre_x: np.ndarray
    pre_health: np.ndarray
    pre_alive: np.ndarray
    memory_phase: tuple[str, ...]
    prediction_contexts: tuple[dict[str, Any] | None, ...]
    lineage_id: tuple[str, ...]
    founder_id: tuple[str, ...]
    effective_memory: np.ndarray
    memory_abs: np.ndarray
    digest: str


def compact_contract_payload() -> dict[str, Any]:
    return {
        "schema_version": DIRECT_CONTRACT_SCHEMA_VERSION,
        "output_contract": "perf.cxxcore20_compact_summary_consumers.CompactSummaryInputs",
        "producer": "direct_compact_pre_snapshot_plus_post_python_state",
        "deterministic_order": "input agent_ids order for agent rows; sorted event_id/window groups in compact consumer",
        "consumers": [CONSUMER_MEMORY_ACTION, CONSUMER_PREDICTION_DEMAND],
        "field_groups": {
            "agent": {
                "agent_ids": "int32[n]",
                "actions": "int32[n]",
                "pre_y/pre_x/post_y/post_x": "int32[n]",
                "pre_health/post_health": "float32[n]",
                "pre_alive/post_alive": "uint8[n]",
            },
            "memory_action": {
                "phase_codes/post_phase_codes": "int32[n]",
                "effective_memory/memory_abs": "float32[n]",
                "lineage_codes/founder_codes": "int32[n]",
            },
            "prediction_demand": {
                "has_prediction": "uint8[n]",
                "prediction_event_id/window_code/center/trend/delay/time_to": "int32[n]",
                "prediction_cue_ambiguity": "float32[n]",
                "prediction_prior_cue": "uint8[n]",
            },
        },
        "row_compatibility_debug_only": True,
        "checkpoint_schema_changed": False,
        "metrics_schema_changed": False,
        "native_rng_added": False,
        "authoritative_state_mutation": False,
    }


def dataflow_audit_payload() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "summary": "CXXCORE-22 bypasses LiveSummaryTape materialization for compact consumers.",
        "current_row_path": [
            "capture_pre_tick_snapshot builds Python arrays plus phase/context tuples",
            "build_live_summary_tape materializes row-compatible pre/post tape object",
            "memory_action_window.rows_from_summary_tape builds Python dict rows",
            "prediction_demand_action_context.rows_from_summary_tape builds Python dict rows",
            "prepare_compact_inputs_from_tape translates row-compatible tape to typed compact arrays",
            "CXXCORE-20 Python/native compact aggregators consume typed arrays",
        ],
        "direct_path": [
            "capture_direct_compact_pre_snapshot builds compact pre snapshot and computes prediction contexts once per tick",
            "Python env mutation remains authoritative",
            "build_direct_compact_inputs_from_snapshot reads post state and emits CompactSummaryInputs",
            "CXXCORE-20 Python/native compact aggregators consume typed arrays directly",
            "row-compatible LiveSummaryTape is built only on guarded parity samples",
        ],
        "redundant_for_compact": [
            "LiveSummaryTape object materialization",
            "LiveSummaryTape digest on the no-compare hot path",
            "Python dict diagnostic rows outside debug/parity samples",
            "row-to-compact translation after tape construction",
        ],
        "debug_or_parity_only": [
            "row-compatible memory-action diagnostic rows",
            "row-compatible prediction-demand diagnostic rows",
            "full LiveSummaryTape payload rendering",
        ],
        "unchanged_ownership": {
            "python_authoritative_state": True,
            "checkpoint_schema_changed": False,
            "metrics_schema_changed": False,
            "native_rng_added": False,
            "behavior_config_changed": False,
            "rewards_changed": False,
        },
    }


def capture_direct_compact_pre_snapshot(
    runner: Any,
    *,
    generation: int,
    tick: int,
    alive: Sequence[Any],
    agent_ids_np: Any,
    actions_np: Any,
) -> DirectCompactPreSnapshot:
    agent_ids = np.asarray(agent_ids_np, dtype=np.int32).reshape(-1)
    actions = np.asarray(actions_np, dtype=np.int32).reshape(-1)
    if int(agent_ids.shape[0]) != int(actions.shape[0]):
        raise ValueError("agent_ids and actions must have the same length")
    env = runner.env
    pos_y = np.asarray(getattr(env, "agent_pos_y"), dtype=np.int32)
    pos_x = np.asarray(getattr(env, "agent_pos_x"), dtype=np.int32)
    alive_arr = np.asarray(getattr(env, "agent_alive"), dtype=bool)
    alive_by_id = {int(getattr(agent, "agent_id")): agent for agent in alive}
    pre_y = pos_y[agent_ids].astype(np.int32, copy=True)
    pre_x = pos_x[agent_ids].astype(np.int32, copy=True)
    pre_alive = alive_arr[agent_ids].astype(np.uint8, copy=True)
    pre_health = np.asarray([float(getattr(alive_by_id.get(int(aid)), "health", 0.0)) for aid in agent_ids], dtype=np.float32)
    phases = tuple(_memory_phase_at(env, int(y), int(x)) for y, x in zip(pre_y.tolist(), pre_x.tolist()))
    contexts = _prediction_contexts(runner)
    prediction_contexts = tuple(
        _nearest_prediction_context_from_contexts(runner, contexts, int(y), int(x))
        for y, x in zip(pre_y.tolist(), pre_x.tolist())
    )
    lineage = tuple(
        str(
            getattr(alive_by_id.get(int(aid)), "lineage_id", "")
            or getattr(alive_by_id.get(int(aid)), "founder_id", "")
            or ""
        )
        for aid in agent_ids
    )
    founders = tuple(str(getattr(alive_by_id.get(int(aid)), "founder_id", "") or "") for aid in agent_ids)
    effective_memory = np.asarray(
        [_effective_memory_value(runner, alive_by_id.get(int(aid))) for aid in agent_ids],
        dtype=np.float32,
    )
    memory_abs = np.asarray([_memory_abs_value(runner, alive_by_id.get(int(aid))) for aid in agent_ids], dtype=np.float32)
    payload = {
        "schema_version": DIRECT_PRE_SNAPSHOT_SCHEMA_VERSION,
        "generation": int(generation),
        "tick": int(tick),
        "agent_ids": agent_ids,
        "actions": actions,
        "pre_y": pre_y,
        "pre_x": pre_x,
        "pre_health": pre_health,
        "pre_alive": pre_alive,
        "memory_phase": phases,
        "prediction_contexts": prediction_contexts,
        "lineage_id": lineage,
        "founder_id": founders,
        "effective_memory": effective_memory,
        "memory_abs": memory_abs,
    }
    return DirectCompactPreSnapshot(digest=_stable_json_digest(payload), **payload)


def build_direct_compact_inputs_from_snapshot(
    runner: Any,
    snapshot: DirectCompactPreSnapshot,
    *,
    grid_shape: tuple[int, int],
) -> CompactSummaryInputs:
    env = runner.env
    agent_ids = snapshot.agent_ids
    pos_y = np.asarray(getattr(env, "agent_pos_y"), dtype=np.int32)
    pos_x = np.asarray(getattr(env, "agent_pos_x"), dtype=np.int32)
    alive_arr = np.asarray(getattr(env, "agent_alive"), dtype=bool)
    agents = getattr(getattr(runner, "pop", None), "agents", {})
    post_y = pos_y[agent_ids].astype(np.int32, copy=True)
    post_x = pos_x[agent_ids].astype(np.int32, copy=True)
    post_alive = alive_arr[agent_ids].astype(np.uint8, copy=True)
    post_health = np.asarray([float(getattr(agents.get(int(aid)), "health", 0.0)) for aid in agent_ids], dtype=np.float32)
    post_phases = tuple(_memory_phase_at(env, int(y), int(x)) for y, x in zip(post_y.tolist(), post_x.tolist()))
    lineage_labels = tuple(sorted({str(value) for value in snapshot.lineage_id if str(value)}))
    founder_labels = tuple(sorted({str(value) for value in snapshot.founder_id if str(value)}))
    lineage_map = {label: idx + 1 for idx, label in enumerate(lineage_labels)}
    founder_map = {label: idx + 1 for idx, label in enumerate(founder_labels)}
    has_prediction, event_id, window_code, center_y, center_x, trend_sign, configured_delay, time_to, cue_ambiguity, prior_cue = _encode_prediction_contexts(
        snapshot.prediction_contexts
    )
    payload = {
        "schema_version": "cxxcore20_compact_summary_inputs_v1",
        "generation": int(snapshot.generation),
        "tick": int(snapshot.tick),
        "agent_ids": np.ascontiguousarray(snapshot.agent_ids, dtype=np.int32),
        "actions": np.ascontiguousarray(snapshot.actions, dtype=np.int32),
        "pre_y": np.ascontiguousarray(snapshot.pre_y, dtype=np.int32),
        "pre_x": np.ascontiguousarray(snapshot.pre_x, dtype=np.int32),
        "post_y": post_y,
        "post_x": post_x,
        "pre_health": np.ascontiguousarray(snapshot.pre_health, dtype=np.float32),
        "post_health": post_health,
        "pre_alive": np.ascontiguousarray(snapshot.pre_alive, dtype=np.uint8),
        "post_alive": post_alive,
        "phase_codes": np.ascontiguousarray([PHASE_TO_CODE.get(str(phase), 0) for phase in snapshot.memory_phase], dtype=np.int32),
        "post_phase_codes": np.ascontiguousarray([PHASE_TO_CODE.get(str(phase), 0) for phase in post_phases], dtype=np.int32),
        "effective_memory": np.ascontiguousarray(snapshot.effective_memory, dtype=np.float32),
        "memory_abs": np.ascontiguousarray(snapshot.memory_abs, dtype=np.float32),
        "lineage_codes": np.ascontiguousarray([lineage_map.get(str(value), 0) for value in snapshot.lineage_id], dtype=np.int32),
        "founder_codes": np.ascontiguousarray([founder_map.get(str(value), 0) for value in snapshot.founder_id], dtype=np.int32),
        "has_prediction": has_prediction,
        "prediction_event_id": event_id,
        "prediction_window_code": window_code,
        "prediction_center_y": center_y,
        "prediction_center_x": center_x,
        "prediction_trend_sign": trend_sign,
        "prediction_configured_delay": configured_delay,
        "prediction_time_to_consequence": time_to,
        "prediction_cue_ambiguity": cue_ambiguity,
        "prediction_prior_cue": prior_cue,
        "grid_height": int(grid_shape[0]),
        "grid_width": int(grid_shape[1]),
        "lineage_labels": lineage_labels,
        "founder_labels": founder_labels,
    }
    return CompactSummaryInputs(digest=_stable_json_digest(payload), **payload)


def parity_against_row_semantics(
    runner: Any,
    snapshot: DirectCompactPreSnapshot,
    inputs: CompactSummaryInputs,
    *,
    memory_instrumentation: Any,
    prediction_instrumentation: Any,
    inject_divergence: bool = False,
) -> dict[str, Any]:
    live_snapshot = _live_snapshot_from_direct(snapshot)
    tape = build_live_summary_tape(runner, live_snapshot)
    direct_result = run_python_compact_reference(inputs, inject_divergence=inject_divergence)
    memory_rows = memory_instrumentation.rows_from_summary_tape(runner, tape)
    prediction_rows = prediction_instrumentation.rows_from_summary_tape(runner, tape)
    parity = compare_compact_to_rows(direct_result, memory_rows=memory_rows, prediction_rows=prediction_rows)
    parity.update(
        {
            "memory_row_count": len(memory_rows),
            "prediction_row_count": len(prediction_rows),
            "direct_digest": inputs.digest,
        }
    )
    return parity


def compare_direct_inputs_to_tape_path(
    runner: Any,
    snapshot: DirectCompactPreSnapshot,
    inputs: CompactSummaryInputs,
) -> dict[str, Any]:
    live_snapshot = _live_snapshot_from_direct(snapshot)
    tape = build_live_summary_tape(runner, live_snapshot)
    tape_inputs = prepare_compact_inputs_from_tape(tape, grid_shape=runner.env.grid.shape)
    direct = run_python_compact_reference(inputs)
    expected = run_python_compact_reference(tape_inputs)
    parity = compare_compact_results(expected, direct)
    parity.update({"direct_digest": inputs.digest, "tape_digest": tape_inputs.digest})
    return parity


def run_direct_compact_summary(
    inputs: CompactSummaryInputs,
    *,
    native_backend: Any | None,
    use_native: bool,
    fail_closed: bool = False,
) -> tuple[Any, dict[str, float], int]:
    t0 = time.perf_counter()
    fallback_count = 0
    if use_native:
        result = run_compact_summary(inputs, native_backend=native_backend, fail_closed=fail_closed)
        fallback_count = 1 if getattr(result, "fallback_reason", None) else 0
        return result, {"native_aggregation_s": float(time.perf_counter() - t0), "python_aggregation_s": 0.0}, fallback_count
    result = run_python_compact_reference(inputs)
    return result, {"native_aggregation_s": 0.0, "python_aggregation_s": float(time.perf_counter() - t0)}, fallback_count


def _nearest_prediction_context_from_contexts(
    runner: Any,
    contexts: Sequence[Mapping[str, Any]],
    y: int,
    x: int,
) -> dict[str, Any] | None:
    if not contexts:
        return None
    radius = max(0, int(getattr(runner.sim_cfg, "prediction_demand_action_context_window_radius", 1) or 1))
    candidates: list[tuple[int, int, Mapping[str, Any]]] = []
    for ctx in contexts:
        dist = max(abs(int(y) - int(ctx["region_center_y"])), abs(int(x) - int(ctx["region_center_x"])))
        if dist <= int(ctx["event_radius"]) + radius:
            candidates.append((dist, int(ctx["event_id"]), ctx))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1]))
    return dict(candidates[0][2])


def _encode_prediction_contexts(
    contexts: Sequence[Mapping[str, Any] | None],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(contexts)
    has_prediction = np.zeros((n,), dtype=np.uint8)
    event_id = np.zeros((n,), dtype=np.int32)
    window_code = np.full((n,), -1, dtype=np.int32)
    center_y = np.zeros((n,), dtype=np.int32)
    center_x = np.zeros((n,), dtype=np.int32)
    trend_sign = np.zeros((n,), dtype=np.int32)
    configured_delay = np.zeros((n,), dtype=np.int32)
    time_to = np.zeros((n,), dtype=np.int32)
    cue_ambiguity = np.zeros((n,), dtype=np.float32)
    prior_cue = np.zeros((n,), dtype=np.uint8)
    for i, ctx in enumerate(contexts):
        if not ctx:
            continue
        has_prediction[i] = 1
        event_id[i] = int(ctx.get("event_id", 0) or 0)
        window_code[i] = int(WINDOW_TO_CODE.get(str(ctx.get("window_label", "")), -1))
        center_y[i] = int(ctx.get("region_center_y", 0) or 0)
        center_x[i] = int(ctx.get("region_center_x", 0) or 0)
        trend_sign[i] = int(ctx.get("trend_sign", 0) or 0)
        configured_delay[i] = int(ctx.get("configured_delay", 0) or 0)
        time_to[i] = int(ctx.get("time_to_consequence", 0) or 0)
        cue_ambiguity[i] = np.float32(float(ctx.get("cue_ambiguity", 0.0) or 0.0))
        prior_cue[i] = 1 if bool(ctx.get("prior_cue", False)) else 0
    return has_prediction, event_id, window_code, center_y, center_x, trend_sign, configured_delay, time_to, cue_ambiguity, prior_cue


def _live_snapshot_from_direct(snapshot: DirectCompactPreSnapshot) -> Any:
    payload = {
        "schema_version": "cxxcore18_live_tick_pre_snapshot_v1",
        "generation": int(snapshot.generation),
        "tick": int(snapshot.tick),
        "agent_ids": snapshot.agent_ids.copy(),
        "actions": snapshot.actions.copy(),
        "pre_y": snapshot.pre_y.copy(),
        "pre_x": snapshot.pre_x.copy(),
        "pre_energy": np.zeros_like(snapshot.pre_health, dtype=np.float32),
        "pre_health": snapshot.pre_health.copy(),
        "pre_injury": np.zeros_like(snapshot.pre_health, dtype=np.float32),
        "pre_scar": np.zeros_like(snapshot.pre_health, dtype=np.float32),
        "pre_alive": snapshot.pre_alive.astype(bool, copy=True),
        "memory_phase": tuple(snapshot.memory_phase),
        "prediction_contexts": tuple(snapshot.prediction_contexts),
        "lineage_id": tuple(snapshot.lineage_id),
        "founder_id": tuple(snapshot.founder_id),
        "effective_memory": snapshot.effective_memory.copy(),
        "memory_abs": snapshot.memory_abs.copy(),
    }
    payload["digest"] = _stable_json_digest(payload)
    return LiveTickPreSnapshot(**payload)
