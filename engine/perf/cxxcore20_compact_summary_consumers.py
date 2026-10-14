from __future__ import annotations

"""CXXCORE-20 compact summary consumer helpers.

Default-off diagnostic code. Python remains authoritative for simulation state.
CXXCORE-20 keeps the CXXCORE-18 public row schemas unchanged and adds a compact
probe-only contract that summarizes those diagnostics without rebuilding
per-event Python row dictionaries on the hot path.
"""

from dataclasses import dataclass
import hashlib
import json
import math
import time
from typing import Any, Callable, Mapping

import numpy as np

from env.gridworld_pkg.constants import ACT_ATTACK, ACT_GATHER, ACT_MOVE_E, ACT_MOVE_N, ACT_MOVE_S, ACT_MOVE_W, ACT_REST, N_ACTIONS
from perf.cxxcore18_live_summary_tape_metrics import LiveSummaryTape
from perf.cxxcore19_native_summary_tape_resident import PHASES, PHASE_TO_CODE
from simulation.prediction_demand_action_context import ACTION_LABELS, WINDOW_LABELS


SCHEMA_VERSION = "cxxcore20_compact_summary_consumers_v1"
INPUT_SCHEMA_VERSION = "cxxcore20_compact_summary_inputs_v1"
RESULT_SCHEMA_VERSION = "cxxcore20_compact_summary_result_v1"
AUDIT_SCHEMA_VERSION = "cxxcore20_compact_consumer_boundary_audit_v1"
DECISION_LABELS = (
    "CXXCORE20-A-COMPACT-SUMMARY-CONSUMERS-GO",
    "CXXCORE20-B-COMPACT-PARITY-ONLY-NO-SPEED-CLOSE-NATIVE-SUMMARY",
    "CXXCORE20-C-METRICS-SCHEMA-BLOCKED",
    "CXXCORE20-D-PREDICTION-CONTEXT-TOO-WIDE",
    "CXXCORE20-E-RUNNER-ISLAND-REQUIRED",
    "CXXCORE20-INCONCLUSIVE",
)
DEFAULT_DECISION = "CXXCORE20-INCONCLUSIVE"

ACTION_BUCKETS = 8
MEMORY_SCALAR_FIELDS = (
    "movement_action_count",
    "gather_action_count",
    "idle_noop_count",
    "unknown_other_action_count",
    "damage_event_count",
    "damage_with_prior_cue_context_count",
    "active_hazard_exposure_count",
    "pre_hazard_exit_count",
    "hazard_entry_count",
    "hazard_exit_count",
)
PREDICTION_SCALAR_FIELDS = (
    "exposure_count",
    "unique_agent_count",
    "gather_attempts_near_region",
    "movement_toward_count",
    "movement_away_count",
    "movement_neutral_count",
    "movement_not_derivable_count",
    "lineage_count",
    "top_lineage_count",
    "active_consequence_exposure_after_prior_cue",
    "active_consequence_exposure_without_prior_cue",
)
WINDOW_TO_CODE = {label: idx for idx, label in enumerate(WINDOW_LABELS)}
CODE_TO_WINDOW = {idx: label for label, idx in WINDOW_TO_CODE.items()}
ACTION_LABEL_TO_CODE = {label: idx for idx, label in enumerate(ACTION_LABELS)}


@dataclass(frozen=True)
class CompactSummaryConfig:
    cxxcore20_compact_summary_enable: bool = False
    cxxcore20_compact_summary_mode: str = "off"
    cxxcore20_compact_summary_compare_enable: bool = True
    cxxcore20_compact_summary_fail_closed: bool = False


@dataclass(frozen=True)
class CompactSummaryInputs:
    schema_version: str
    generation: int
    tick: int
    agent_ids: np.ndarray
    actions: np.ndarray
    pre_y: np.ndarray
    pre_x: np.ndarray
    post_y: np.ndarray
    post_x: np.ndarray
    pre_health: np.ndarray
    post_health: np.ndarray
    pre_alive: np.ndarray
    post_alive: np.ndarray
    phase_codes: np.ndarray
    post_phase_codes: np.ndarray
    effective_memory: np.ndarray
    memory_abs: np.ndarray
    lineage_codes: np.ndarray
    founder_codes: np.ndarray
    has_prediction: np.ndarray
    prediction_event_id: np.ndarray
    prediction_window_code: np.ndarray
    prediction_center_y: np.ndarray
    prediction_center_x: np.ndarray
    prediction_trend_sign: np.ndarray
    prediction_configured_delay: np.ndarray
    prediction_time_to_consequence: np.ndarray
    prediction_cue_ambiguity: np.ndarray
    prediction_prior_cue: np.ndarray
    grid_height: int
    grid_width: int
    lineage_labels: tuple[str, ...]
    founder_labels: tuple[str, ...]
    digest: str


@dataclass(frozen=True)
class CompactSummaryResult:
    schema_version: str
    status: str
    fallback_reason: str | None
    memory_phase_present: np.ndarray
    memory_agent_count: np.ndarray
    memory_action_counts: np.ndarray
    memory_scalar_counts: np.ndarray
    memory_damage_total: np.ndarray
    memory_effective_memory_sum: np.ndarray
    memory_abs_sum: np.ndarray
    memory_lineage_unique_count: np.ndarray
    memory_top_lineage_code: np.ndarray
    memory_top_lineage_count: np.ndarray
    memory_founder_unique_count: np.ndarray
    prediction_group_count: int
    prediction_event_id: np.ndarray
    prediction_window_code: np.ndarray
    prediction_center_y: np.ndarray
    prediction_center_x: np.ndarray
    prediction_trend_sign: np.ndarray
    prediction_configured_delay: np.ndarray
    prediction_time_to_consequence: np.ndarray
    prediction_cue_ambiguity: np.ndarray
    prediction_action_counts: np.ndarray
    prediction_scalar_counts: np.ndarray
    prediction_top_lineage_code: np.ndarray
    timing: dict[str, float]
    digest: str


@dataclass(frozen=True)
class CompactCompareResult:
    schema_version: str
    mode: str
    status: str
    parity: dict[str, Any]
    fallback_reason: str | None
    first_divergence: dict[str, Any] | None
    compact_result: CompactSummaryResult | None
    timing: dict[str, float]
    diagnostics: dict[str, Any]


def config_from_sim_cfg(sim_cfg: Any) -> CompactSummaryConfig:
    return CompactSummaryConfig(
        cxxcore20_compact_summary_enable=bool(getattr(sim_cfg, "cxxcore20_compact_summary_enable", False)),
        cxxcore20_compact_summary_mode=str(getattr(sim_cfg, "cxxcore20_compact_summary_mode", "off") or "off"),
        cxxcore20_compact_summary_compare_enable=bool(getattr(sim_cfg, "cxxcore20_compact_summary_compare_enable", True)),
        cxxcore20_compact_summary_fail_closed=bool(getattr(sim_cfg, "cxxcore20_compact_summary_fail_closed", False)),
    )


def compact_consumer_boundary_audit_payload() -> dict[str, Any]:
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "decision_labels": list(DECISION_LABELS),
        "rows_currently_produced": {
            "memory_action_window": "one Python dict row per phase with action histogram, lineage summary, memory summary, and exposure/damage summary",
            "prediction_demand_action_context": "one Python dict row per event_id/window_label group with action, movement, lineage, and context fields",
        },
        "compact_contract": {
            "memory_action_window": "phase-indexed numeric arrays and encoded lineage/founder cardinalities",
            "prediction_demand_action_context": "event/window-keyed numeric arrays with compact context fields and movement/action counters",
            "hot_path_avoids": "per-event Python row list/dict reconstruction",
            "report_only_translation": "optional payload rendering for evidence and parity inspection",
        },
        "blocked_or_python_owned": {
            "prediction_context_lookup": "Python still captures context metadata in the summary tape; native groups encoded context arrays only.",
            "public_metrics_schema": "unchanged; compact direct consumption is probe-only.",
            "checkpoint_schema": "unchanged",
            "rng": "no native RNG",
            "live_state_mutation": "none",
        },
        "checkpoint_schema_changed": False,
        "metrics_schema_changed": False,
        "native_rng_added": False,
        "authoritative_state_mutation": False,
    }


def prepare_compact_inputs_from_tape(tape: LiveSummaryTape, *, grid_shape: tuple[int, int]) -> CompactSummaryInputs:
    n = int(tape.agent_ids.shape[0])
    if len(tape.prediction_contexts) != n or len(tape.lineage_id) != n or len(tape.founder_id) != n:
        raise ValueError("summary tape metadata is not aligned")
    lineage_labels = tuple(sorted({str(value) for value in tape.lineage_id if str(value)}))
    founder_labels = tuple(sorted({str(value) for value in tape.founder_id if str(value)}))
    lineage_map = {label: idx + 1 for idx, label in enumerate(lineage_labels)}
    founder_map = {label: idx + 1 for idx, label in enumerate(founder_labels)}
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
    for i, ctx in enumerate(tape.prediction_contexts):
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
    payload = {
        "schema_version": INPUT_SCHEMA_VERSION,
        "generation": int(tape.generation),
        "tick": int(tape.tick),
        "agent_ids": np.ascontiguousarray(tape.agent_ids, dtype=np.int32),
        "actions": np.ascontiguousarray(tape.actions, dtype=np.int32),
        "pre_y": np.ascontiguousarray(tape.pre_y, dtype=np.int32),
        "pre_x": np.ascontiguousarray(tape.pre_x, dtype=np.int32),
        "post_y": np.ascontiguousarray(tape.post_y, dtype=np.int32),
        "post_x": np.ascontiguousarray(tape.post_x, dtype=np.int32),
        "pre_health": np.ascontiguousarray(tape.pre_health, dtype=np.float32),
        "post_health": np.ascontiguousarray(tape.post_health, dtype=np.float32),
        "pre_alive": np.ascontiguousarray(tape.pre_alive, dtype=np.uint8),
        "post_alive": np.ascontiguousarray(tape.post_alive, dtype=np.uint8),
        "phase_codes": np.ascontiguousarray([PHASE_TO_CODE.get(str(phase), 0) for phase in tape.memory_phase], dtype=np.int32),
        "post_phase_codes": np.ascontiguousarray([PHASE_TO_CODE.get(str(phase), 0) for phase in tape.memory_post_phase], dtype=np.int32),
        "effective_memory": np.ascontiguousarray(tape.effective_memory, dtype=np.float32),
        "memory_abs": np.ascontiguousarray(tape.memory_abs, dtype=np.float32),
        "lineage_codes": np.ascontiguousarray([lineage_map.get(str(value), 0) for value in tape.lineage_id], dtype=np.int32),
        "founder_codes": np.ascontiguousarray([founder_map.get(str(value), 0) for value in tape.founder_id], dtype=np.int32),
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


def run_python_compact_reference(inputs: CompactSummaryInputs, *, inject_divergence: bool = False) -> CompactSummaryResult:
    t0 = time.perf_counter()
    result = _empty_result(inputs, status="ok", fallback_reason=None)
    _fill_memory(inputs, result)
    _fill_prediction(inputs, result)
    timing = dict(result.timing)
    timing["python_compact_s"] = float(time.perf_counter() - t0)
    result = _replace_result_timing(result, timing)
    if inject_divergence:
        counts = result.memory_agent_count.copy()
        counts[0] += 1
        result = _replace_result_arrays(result, memory_agent_count=counts)
    return result


def run_compact_summary(
    inputs: CompactSummaryInputs,
    *,
    native_backend: Callable[[CompactSummaryInputs], CompactSummaryResult | Mapping[str, Any]] | None,
    fail_closed: bool = False,
) -> CompactSummaryResult:
    if native_backend is None:
        if fail_closed:
            raise RuntimeError("CXXCORE-20 native compact backend unavailable")
        result = run_python_compact_reference(inputs)
        return _replace_status(result, status="fallback", fallback_reason="native_unavailable")
    try:
        out = native_backend(inputs)
        if isinstance(out, CompactSummaryResult):
            return out
        return compact_result_from_mapping(inputs, out)
    except Exception:
        if fail_closed:
            raise
        result = run_python_compact_reference(inputs)
        return _replace_status(result, status="fallback", fallback_reason="native_exception")


def run_compact_shadow_compare(
    *,
    inputs: CompactSummaryInputs,
    native_backend: Callable[[CompactSummaryInputs], CompactSummaryResult | Mapping[str, Any]] | None,
    mode: str,
    compare: bool = True,
    fail_closed: bool = False,
    inject_divergence: bool = False,
) -> CompactCompareResult:
    total_t0 = time.perf_counter()
    py_t0 = time.perf_counter()
    expected = run_python_compact_reference(inputs)
    python_s = float(time.perf_counter() - py_t0)
    native_t0 = time.perf_counter()
    candidate = run_compact_summary(inputs, native_backend=native_backend, fail_closed=fail_closed)
    native_s = float(time.perf_counter() - native_t0)
    if inject_divergence and candidate.fallback_reason is None:
        counts = candidate.memory_agent_count.copy()
        counts[0] += 1
        candidate = _replace_result_arrays(candidate, memory_agent_count=counts)
    compare_s = 0.0
    parity = {"ok": True, "first_divergence": None}
    if compare and candidate.fallback_reason is None:
        cmp_t0 = time.perf_counter()
        parity = compare_compact_results(expected, candidate)
        compare_s = float(time.perf_counter() - cmp_t0)
    elif candidate.fallback_reason:
        parity = {"ok": False, "first_divergence": None}
    status = "fallback" if candidate.fallback_reason else ("ok" if bool(parity.get("ok")) else "divergence")
    first = parity.get("first_divergence")
    return CompactCompareResult(
        schema_version=SCHEMA_VERSION,
        mode=mode,
        status=status,
        parity=parity,
        fallback_reason=candidate.fallback_reason if status == "fallback" else (None if status == "ok" else "divergence_detected"),
        first_divergence=first,
        compact_result=candidate,
        timing={
            "total_s": float(time.perf_counter() - total_t0),
            "python_compact_s": python_s,
            "native_compact_s": native_s,
            "compare_s": compare_s,
        },
        diagnostics=compact_diagnostics(candidate),
    )


def compact_result_from_mapping(inputs: CompactSummaryInputs, payload: Mapping[str, Any]) -> CompactSummaryResult:
    result = CompactSummaryResult(
        schema_version=RESULT_SCHEMA_VERSION,
        status=str(payload.get("status", "ok") or "ok"),
        fallback_reason=payload.get("fallback_reason"),
        memory_phase_present=np.ascontiguousarray(payload["memory_phase_present"], dtype=np.uint8),
        memory_agent_count=np.ascontiguousarray(payload["memory_agent_count"], dtype=np.int32),
        memory_action_counts=np.ascontiguousarray(payload["memory_action_counts"], dtype=np.int32).reshape((len(PHASES), ACTION_BUCKETS)),
        memory_scalar_counts=np.ascontiguousarray(payload["memory_scalar_counts"], dtype=np.int32).reshape((len(PHASES), len(MEMORY_SCALAR_FIELDS))),
        memory_damage_total=np.ascontiguousarray(payload["memory_damage_total"], dtype=np.float32),
        memory_effective_memory_sum=np.ascontiguousarray(payload["memory_effective_memory_sum"], dtype=np.float32),
        memory_abs_sum=np.ascontiguousarray(payload["memory_abs_sum"], dtype=np.float32),
        memory_lineage_unique_count=np.ascontiguousarray(payload["memory_lineage_unique_count"], dtype=np.int32),
        memory_top_lineage_code=np.ascontiguousarray(payload["memory_top_lineage_code"], dtype=np.int32),
        memory_top_lineage_count=np.ascontiguousarray(payload["memory_top_lineage_count"], dtype=np.int32),
        memory_founder_unique_count=np.ascontiguousarray(payload["memory_founder_unique_count"], dtype=np.int32),
        prediction_group_count=int(payload["prediction_group_count"]),
        prediction_event_id=np.ascontiguousarray(payload["prediction_event_id"], dtype=np.int32),
        prediction_window_code=np.ascontiguousarray(payload["prediction_window_code"], dtype=np.int32),
        prediction_center_y=np.ascontiguousarray(payload["prediction_center_y"], dtype=np.int32),
        prediction_center_x=np.ascontiguousarray(payload["prediction_center_x"], dtype=np.int32),
        prediction_trend_sign=np.ascontiguousarray(payload["prediction_trend_sign"], dtype=np.int32),
        prediction_configured_delay=np.ascontiguousarray(payload["prediction_configured_delay"], dtype=np.int32),
        prediction_time_to_consequence=np.ascontiguousarray(payload["prediction_time_to_consequence"], dtype=np.int32),
        prediction_cue_ambiguity=np.ascontiguousarray(payload["prediction_cue_ambiguity"], dtype=np.float32),
        prediction_action_counts=np.ascontiguousarray(payload["prediction_action_counts"], dtype=np.int32).reshape((-1, ACTION_BUCKETS)),
        prediction_scalar_counts=np.ascontiguousarray(payload["prediction_scalar_counts"], dtype=np.int32).reshape((-1, len(PREDICTION_SCALAR_FIELDS))),
        prediction_top_lineage_code=np.ascontiguousarray(payload["prediction_top_lineage_code"], dtype=np.int32),
        timing=dict(payload.get("timing", {})),
        digest="",
    )
    return _replace_result_digest(result)


def compare_compact_results(expected: CompactSummaryResult, candidate: CompactSummaryResult) -> dict[str, Any]:
    expected_payload = compact_payload(expected)
    candidate_payload = compact_payload(candidate)
    first = _first_payload_mismatch(expected_payload, candidate_payload)
    return {"ok": first is None, "first_divergence": first}


def compare_compact_to_rows(result: CompactSummaryResult, *, memory_rows: list[Mapping[str, Any]], prediction_rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    expected = summary_from_rows(memory_rows=memory_rows, prediction_rows=prediction_rows)
    candidate = compact_diagnostic_summary(result)
    first = _first_payload_mismatch(expected, candidate)
    return {"ok": first is None, "first_divergence": first}


def compact_payload(result: CompactSummaryResult) -> dict[str, Any]:
    return {
        "schema_version": result.schema_version,
        "status": result.status,
        "fallback_reason": result.fallback_reason,
        "memory_phase_present": result.memory_phase_present.astype(int).tolist(),
        "memory_agent_count": result.memory_agent_count.astype(int).tolist(),
        "memory_action_counts": result.memory_action_counts.astype(int).tolist(),
        "memory_scalar_counts": result.memory_scalar_counts.astype(int).tolist(),
        "memory_damage_total": _round_float_list(result.memory_damage_total),
        "memory_effective_memory_sum": _round_float_list(result.memory_effective_memory_sum),
        "memory_abs_sum": _round_float_list(result.memory_abs_sum),
        "memory_lineage_unique_count": result.memory_lineage_unique_count.astype(int).tolist(),
        "memory_top_lineage_code": result.memory_top_lineage_code.astype(int).tolist(),
        "memory_top_lineage_count": result.memory_top_lineage_count.astype(int).tolist(),
        "memory_founder_unique_count": result.memory_founder_unique_count.astype(int).tolist(),
        "prediction_group_count": int(result.prediction_group_count),
        "prediction_event_id": result.prediction_event_id[: result.prediction_group_count].astype(int).tolist(),
        "prediction_window_code": result.prediction_window_code[: result.prediction_group_count].astype(int).tolist(),
        "prediction_center_y": result.prediction_center_y[: result.prediction_group_count].astype(int).tolist(),
        "prediction_center_x": result.prediction_center_x[: result.prediction_group_count].astype(int).tolist(),
        "prediction_trend_sign": result.prediction_trend_sign[: result.prediction_group_count].astype(int).tolist(),
        "prediction_configured_delay": result.prediction_configured_delay[: result.prediction_group_count].astype(int).tolist(),
        "prediction_time_to_consequence": result.prediction_time_to_consequence[: result.prediction_group_count].astype(int).tolist(),
        "prediction_cue_ambiguity": _round_float_list(result.prediction_cue_ambiguity[: result.prediction_group_count]),
        "prediction_action_counts": result.prediction_action_counts[: result.prediction_group_count].astype(int).tolist(),
        "prediction_scalar_counts": result.prediction_scalar_counts[: result.prediction_group_count].astype(int).tolist(),
        "prediction_top_lineage_code": result.prediction_top_lineage_code[: result.prediction_group_count].astype(int).tolist(),
    }


def compact_diagnostic_summary(result: CompactSummaryResult) -> dict[str, Any]:
    memory = {}
    for phase_idx, phase in enumerate(PHASES):
        if int(result.memory_phase_present[phase_idx]) <= 0:
            continue
        agent_count = int(result.memory_agent_count[phase_idx])
        memory[phase] = {
            "agent_count": agent_count,
            "action_histogram": {str(i): int(result.memory_action_counts[phase_idx, i]) for i in range(N_ACTIONS)},
            "movement_action_count": int(result.memory_scalar_counts[phase_idx, MEMORY_SCALAR_FIELDS.index("movement_action_count")]),
            "gather_action_count": int(result.memory_scalar_counts[phase_idx, MEMORY_SCALAR_FIELDS.index("gather_action_count")]),
            "idle_noop_count": int(result.memory_scalar_counts[phase_idx, MEMORY_SCALAR_FIELDS.index("idle_noop_count")]),
            "unknown_other_action_count": int(result.memory_scalar_counts[phase_idx, MEMORY_SCALAR_FIELDS.index("unknown_other_action_count")]),
            "damage_event_count": int(result.memory_scalar_counts[phase_idx, MEMORY_SCALAR_FIELDS.index("damage_event_count")]),
            "damage_total": round(float(result.memory_damage_total[phase_idx]), 6),
            "damage_with_prior_cue_context_count": int(result.memory_scalar_counts[phase_idx, MEMORY_SCALAR_FIELDS.index("damage_with_prior_cue_context_count")]),
            "active_hazard_exposure_count": int(result.memory_scalar_counts[phase_idx, MEMORY_SCALAR_FIELDS.index("active_hazard_exposure_count")]),
            "pre_hazard_exit_count": int(result.memory_scalar_counts[phase_idx, MEMORY_SCALAR_FIELDS.index("pre_hazard_exit_count")]),
            "hazard_entry_count": int(result.memory_scalar_counts[phase_idx, MEMORY_SCALAR_FIELDS.index("hazard_entry_count")]),
            "hazard_exit_count": int(result.memory_scalar_counts[phase_idx, MEMORY_SCALAR_FIELDS.index("hazard_exit_count")]),
            "lineage_count": int(result.memory_lineage_unique_count[phase_idx]),
            "top_lineage_share": round(
                float(result.memory_top_lineage_count[phase_idx]) / float(agent_count) if agent_count > 0 else 0.0,
                6,
            ),
            "founder_count": int(result.memory_founder_unique_count[phase_idx]),
            "effective_memory_mean": round(
                float(result.memory_effective_memory_sum[phase_idx]) / float(agent_count) if agent_count > 0 else 0.0,
                6,
            ),
            "memory_state_abs_mean": round(
                float(result.memory_abs_sum[phase_idx]) / float(agent_count) if agent_count > 0 else 0.0,
                6,
            ),
            "memory_agent_count": agent_count,
        }
    prediction_groups = []
    for idx in range(int(result.prediction_group_count)):
        exposure = int(result.prediction_scalar_counts[idx, PREDICTION_SCALAR_FIELDS.index("exposure_count")])
        lineage_count = int(result.prediction_scalar_counts[idx, PREDICTION_SCALAR_FIELDS.index("lineage_count")])
        top_lineage_count = int(result.prediction_scalar_counts[idx, PREDICTION_SCALAR_FIELDS.index("top_lineage_count")])
        prediction_groups.append(
            {
                "event_id": int(result.prediction_event_id[idx]),
                "window_label": CODE_TO_WINDOW.get(int(result.prediction_window_code[idx]), "unknown"),
                "region_center": [int(result.prediction_center_y[idx]), int(result.prediction_center_x[idx])],
                "trend_sign": int(result.prediction_trend_sign[idx]),
                "configured_delay": int(result.prediction_configured_delay[idx]),
                "time_to_consequence": int(result.prediction_time_to_consequence[idx]),
                "cue_ambiguity": round(float(result.prediction_cue_ambiguity[idx]), 6),
                "action_label_counts": {label: int(result.prediction_action_counts[idx, ACTION_LABEL_TO_CODE[label]]) for label in ACTION_LABELS},
                "gather_attempts_near_region": int(result.prediction_scalar_counts[idx, PREDICTION_SCALAR_FIELDS.index("gather_attempts_near_region")]),
                "movement_toward_count": int(result.prediction_scalar_counts[idx, PREDICTION_SCALAR_FIELDS.index("movement_toward_count")]),
                "movement_away_count": int(result.prediction_scalar_counts[idx, PREDICTION_SCALAR_FIELDS.index("movement_away_count")]),
                "movement_neutral_count": int(result.prediction_scalar_counts[idx, PREDICTION_SCALAR_FIELDS.index("movement_neutral_count")]),
                "movement_not_derivable_count": int(result.prediction_scalar_counts[idx, PREDICTION_SCALAR_FIELDS.index("movement_not_derivable_count")]),
                "exposure_count": exposure,
                "unique_agent_count": int(result.prediction_scalar_counts[idx, PREDICTION_SCALAR_FIELDS.index("unique_agent_count")]),
                "lineage_count": lineage_count,
                "top_lineage_share": round(float(top_lineage_count) / float(exposure) if exposure > 0 else 0.0, 6),
                "active_consequence_exposure_after_prior_cue": int(result.prediction_scalar_counts[idx, PREDICTION_SCALAR_FIELDS.index("active_consequence_exposure_after_prior_cue")]),
                "active_consequence_exposure_without_prior_cue": int(result.prediction_scalar_counts[idx, PREDICTION_SCALAR_FIELDS.index("active_consequence_exposure_without_prior_cue")]),
            }
        )
    prediction_groups.sort(key=lambda row: (row["event_id"], row["window_label"]))
    return {"memory": memory, "prediction_groups": prediction_groups}


def summary_from_rows(*, memory_rows: list[Mapping[str, Any]], prediction_rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    memory = {}
    for row in sorted(memory_rows, key=lambda item: str(item.get("phase", ""))):
        exposure = dict(row.get("exposure_damage_summary", {}))
        lineage = dict(row.get("lineage_context", {}))
        memory_summary = dict(row.get("memory_summary", {}))
        memory[str(row.get("phase", ""))] = {
            "agent_count": int(row.get("agent_count", 0) or 0),
            "action_histogram": {str(i): int(dict(row.get("action_histogram", {})).get(str(i), 0) or 0) for i in range(N_ACTIONS)},
            "movement_action_count": int(row.get("movement_action_count", 0) or 0),
            "gather_action_count": int(row.get("gather_action_count", 0) or 0),
            "idle_noop_count": int(row.get("idle_noop_count", 0) or 0),
            "unknown_other_action_count": int(row.get("unknown_other_action_count", 0) or 0),
            "damage_event_count": int(exposure.get("damage_event_count", 0) or 0),
            "damage_total": round(float(exposure.get("damage_total", 0.0) or 0.0), 6),
            "damage_with_prior_cue_context_count": int(exposure.get("damage_with_prior_cue_context_count", 0) or 0),
            "active_hazard_exposure_count": int(exposure.get("active_hazard_exposure_count", 0) or 0),
            "pre_hazard_exit_count": int(exposure.get("pre_hazard_exit_count", 0) or 0),
            "hazard_entry_count": int(exposure.get("hazard_entry_count", 0) or 0),
            "hazard_exit_count": int(exposure.get("hazard_exit_count", 0) or 0),
            "lineage_count": int(lineage.get("lineage_count", 0) or 0),
            "top_lineage_share": round(float(lineage.get("top_lineage_share", 0.0) or 0.0), 6),
            "founder_count": len(list(lineage.get("founder_ids", []) or [])),
            "effective_memory_mean": round(float(memory_summary.get("effective_memory_mean", 0.0) or 0.0), 6),
            "memory_state_abs_mean": round(float(memory_summary.get("memory_state_abs_mean", 0.0) or 0.0), 6),
            "memory_agent_count": int(memory_summary.get("memory_agent_count", 0) or 0),
        }
    prediction_groups = []
    for row in prediction_rows:
        prediction_groups.append(
            {
                "event_id": int(row.get("event_id", 0) or 0),
                "window_label": str(row.get("window_label", "")),
                "region_center": [int(v) for v in list(row.get("region_center", [0, 0]))[:2]],
                "trend_sign": int(row.get("trend_sign", 0) or 0),
                "configured_delay": int(row.get("configured_delay", 0) or 0),
                "time_to_consequence": int(row.get("time_to_consequence", 0) or 0),
                "cue_ambiguity": round(float(row.get("cue_ambiguity", 0.0) or 0.0), 6),
                "action_label_counts": {label: int(dict(row.get("action_label_counts", {})).get(label, 0) or 0) for label in ACTION_LABELS},
                "gather_attempts_near_region": int(row.get("gather_attempts_near_region", 0) or 0),
                "movement_toward_count": int(row.get("movement_toward_count", 0) or 0),
                "movement_away_count": int(row.get("movement_away_count", 0) or 0),
                "movement_neutral_count": int(row.get("movement_neutral_count", 0) or 0),
                "movement_not_derivable_count": int(row.get("movement_not_derivable_count", 0) or 0),
                "exposure_count": int(row.get("exposure_count", 0) or 0),
                "unique_agent_count": int(row.get("unique_agent_count", 0) or 0),
                "lineage_count": int(row.get("lineage_count", 0) or 0),
                "top_lineage_share": round(float(row.get("top_lineage_share", 0.0) or 0.0), 6),
                "active_consequence_exposure_after_prior_cue": int(row.get("active_consequence_exposure_after_prior_cue", 0) or 0),
                "active_consequence_exposure_without_prior_cue": int(row.get("active_consequence_exposure_without_prior_cue", 0) or 0),
            }
        )
    prediction_groups.sort(key=lambda row: (row["event_id"], row["window_label"]))
    return {"memory": memory, "prediction_groups": prediction_groups}


def compact_diagnostics(result: CompactSummaryResult) -> dict[str, Any]:
    return {
        "native_rng_added": False,
        "checkpoint_schema_changed": False,
        "metrics_schema_changed": False,
        "authoritative_state_mutation": False,
        "memory_present_phases": int(np.count_nonzero(result.memory_phase_present)),
        "prediction_groups": int(result.prediction_group_count),
    }


def compare_payload(result: CompactCompareResult) -> dict[str, Any]:
    return {
        "schema_version": result.schema_version,
        "mode": result.mode,
        "status": result.status,
        "parity": result.parity,
        "fallback_reason": result.fallback_reason,
        "first_divergence": result.first_divergence,
        "timing": dict(result.timing),
        "diagnostics": dict(result.diagnostics),
    }


def _empty_result(inputs: CompactSummaryInputs, *, status: str, fallback_reason: str | None) -> CompactSummaryResult:
    max_groups = max(1, int(inputs.agent_ids.shape[0]))
    result = CompactSummaryResult(
        schema_version=RESULT_SCHEMA_VERSION,
        status=status,
        fallback_reason=fallback_reason,
        memory_phase_present=np.zeros((len(PHASES),), dtype=np.uint8),
        memory_agent_count=np.zeros((len(PHASES),), dtype=np.int32),
        memory_action_counts=np.zeros((len(PHASES), ACTION_BUCKETS), dtype=np.int32),
        memory_scalar_counts=np.zeros((len(PHASES), len(MEMORY_SCALAR_FIELDS)), dtype=np.int32),
        memory_damage_total=np.zeros((len(PHASES),), dtype=np.float32),
        memory_effective_memory_sum=np.zeros((len(PHASES),), dtype=np.float32),
        memory_abs_sum=np.zeros((len(PHASES),), dtype=np.float32),
        memory_lineage_unique_count=np.zeros((len(PHASES),), dtype=np.int32),
        memory_top_lineage_code=np.zeros((len(PHASES),), dtype=np.int32),
        memory_top_lineage_count=np.zeros((len(PHASES),), dtype=np.int32),
        memory_founder_unique_count=np.zeros((len(PHASES),), dtype=np.int32),
        prediction_group_count=0,
        prediction_event_id=np.zeros((max_groups,), dtype=np.int32),
        prediction_window_code=np.full((max_groups,), -1, dtype=np.int32),
        prediction_center_y=np.zeros((max_groups,), dtype=np.int32),
        prediction_center_x=np.zeros((max_groups,), dtype=np.int32),
        prediction_trend_sign=np.zeros((max_groups,), dtype=np.int32),
        prediction_configured_delay=np.zeros((max_groups,), dtype=np.int32),
        prediction_time_to_consequence=np.zeros((max_groups,), dtype=np.int32),
        prediction_cue_ambiguity=np.zeros((max_groups,), dtype=np.float32),
        prediction_action_counts=np.zeros((max_groups, ACTION_BUCKETS), dtype=np.int32),
        prediction_scalar_counts=np.zeros((max_groups, len(PREDICTION_SCALAR_FIELDS)), dtype=np.int32),
        prediction_top_lineage_code=np.zeros((max_groups,), dtype=np.int32),
        timing={},
        digest="",
    )
    return _replace_result_digest(result)


def _fill_memory(inputs: CompactSummaryInputs, result: CompactSummaryResult) -> None:
    for i, action_raw in enumerate(inputs.actions.tolist()):
        phase = int(inputs.phase_codes[i])
        if phase < 0 or phase >= len(PHASES):
            phase = 0
        action = int(action_raw)
        result.memory_phase_present[phase] = 1
        result.memory_agent_count[phase] += 1
        if 0 <= action < ACTION_BUCKETS:
            result.memory_action_counts[phase, action] += 1
        if action in (ACT_MOVE_N, ACT_MOVE_S, ACT_MOVE_E, ACT_MOVE_W):
            result.memory_scalar_counts[phase, MEMORY_SCALAR_FIELDS.index("movement_action_count")] += 1
        elif action == ACT_GATHER:
            result.memory_scalar_counts[phase, MEMORY_SCALAR_FIELDS.index("gather_action_count")] += 1
        elif action == ACT_REST:
            result.memory_scalar_counts[phase, MEMORY_SCALAR_FIELDS.index("idle_noop_count")] += 1
        elif action < 0 or action >= N_ACTIONS:
            result.memory_scalar_counts[phase, MEMORY_SCALAR_FIELDS.index("unknown_other_action_count")] += 1
        damage = float(inputs.pre_health[i]) - float(inputs.post_health[i])
        if damage > 0.0:
            result.memory_scalar_counts[phase, MEMORY_SCALAR_FIELDS.index("damage_event_count")] += 1
            result.memory_damage_total[phase] += np.float32(damage)
            if phase in (PHASE_TO_CODE["cue"], PHASE_TO_CODE["hidden_gap"], PHASE_TO_CODE["active_hazard"]):
                result.memory_scalar_counts[phase, MEMORY_SCALAR_FIELDS.index("damage_with_prior_cue_context_count")] += 1
        if phase == PHASE_TO_CODE["active_hazard"]:
            result.memory_scalar_counts[phase, MEMORY_SCALAR_FIELDS.index("active_hazard_exposure_count")] += 1
        post_phase = int(inputs.post_phase_codes[i])
        if bool(inputs.post_alive[i]):
            if phase != PHASE_TO_CODE["active_hazard"] and post_phase == PHASE_TO_CODE["active_hazard"]:
                result.memory_scalar_counts[phase, MEMORY_SCALAR_FIELDS.index("hazard_entry_count")] += 1
            if phase == PHASE_TO_CODE["active_hazard"] and post_phase != PHASE_TO_CODE["active_hazard"]:
                result.memory_scalar_counts[phase, MEMORY_SCALAR_FIELDS.index("hazard_exit_count")] += 1
            if phase in (PHASE_TO_CODE["cue"], PHASE_TO_CODE["hidden_gap"]) and post_phase == PHASE_TO_CODE["baseline"]:
                result.memory_scalar_counts[phase, MEMORY_SCALAR_FIELDS.index("pre_hazard_exit_count")] += 1
        result.memory_effective_memory_sum[phase] += np.float32(inputs.effective_memory[i])
        result.memory_abs_sum[phase] += np.float32(inputs.memory_abs[i])
    for phase in range(len(PHASES)):
        codes = [int(inputs.lineage_codes[i]) for i in range(int(inputs.agent_ids.shape[0])) if int(inputs.phase_codes[i]) == phase and int(inputs.lineage_codes[i]) > 0]
        result.memory_lineage_unique_count[phase], result.memory_top_lineage_code[phase], result.memory_top_lineage_count[phase] = _unique_top(codes)
        founders = {int(inputs.founder_codes[i]) for i in range(int(inputs.agent_ids.shape[0])) if int(inputs.phase_codes[i]) == phase and int(inputs.founder_codes[i]) > 0}
        result.memory_founder_unique_count[phase] = int(len(founders))


def _fill_prediction(inputs: CompactSummaryInputs, result: CompactSummaryResult) -> None:
    for i, has_ctx in enumerate(inputs.has_prediction.tolist()):
        if not int(has_ctx):
            continue
        group = _find_or_add_prediction_group(inputs, result, i)
        action_label_code = _action_label_code(int(inputs.actions[i]))
        result.prediction_action_counts[group, action_label_code] += 1
        result.prediction_scalar_counts[group, PREDICTION_SCALAR_FIELDS.index("exposure_count")] += 1
        result.prediction_scalar_counts[group, PREDICTION_SCALAR_FIELDS.index("unique_agent_count")] += 1
        if action_label_code == ACTION_LABEL_TO_CODE["GATHER"]:
            result.prediction_scalar_counts[group, PREDICTION_SCALAR_FIELDS.index("gather_attempts_near_region")] += 1
        movement_field = _movement_field(
            action=int(inputs.actions[i]),
            y=int(inputs.pre_y[i]),
            x=int(inputs.pre_x[i]),
            center_y=int(inputs.prediction_center_y[i]),
            center_x=int(inputs.prediction_center_x[i]),
            height=int(inputs.grid_height),
            width=int(inputs.grid_width),
        )
        result.prediction_scalar_counts[group, PREDICTION_SCALAR_FIELDS.index(movement_field)] += 1
        if int(inputs.prediction_window_code[i]) == WINDOW_TO_CODE["active_consequence"]:
            if int(inputs.prediction_prior_cue[i]):
                result.prediction_scalar_counts[group, PREDICTION_SCALAR_FIELDS.index("active_consequence_exposure_after_prior_cue")] += 1
            else:
                result.prediction_scalar_counts[group, PREDICTION_SCALAR_FIELDS.index("active_consequence_exposure_without_prior_cue")] += 1
    for group in range(int(result.prediction_group_count)):
        indices = [
            i
            for i in range(int(inputs.agent_ids.shape[0]))
            if int(inputs.has_prediction[i])
            and int(inputs.prediction_event_id[i]) == int(result.prediction_event_id[group])
            and int(inputs.prediction_window_code[i]) == int(result.prediction_window_code[group])
        ]
        codes = [int(inputs.lineage_codes[i]) for i in indices if int(inputs.lineage_codes[i]) > 0]
        unique, top_code, top_count = _unique_top(codes)
        result.prediction_scalar_counts[group, PREDICTION_SCALAR_FIELDS.index("lineage_count")] = unique
        result.prediction_scalar_counts[group, PREDICTION_SCALAR_FIELDS.index("top_lineage_count")] = top_count
        result.prediction_top_lineage_code[group] = top_code


def _find_or_add_prediction_group(inputs: CompactSummaryInputs, result: CompactSummaryResult, index: int) -> int:
    event_id = int(inputs.prediction_event_id[index])
    window_code = int(inputs.prediction_window_code[index])
    for group in range(int(result.prediction_group_count)):
        if int(result.prediction_event_id[group]) == event_id and int(result.prediction_window_code[group]) == window_code:
            return group
    group = int(result.prediction_group_count)
    object.__setattr__(result, "prediction_group_count", group + 1)
    result.prediction_event_id[group] = event_id
    result.prediction_window_code[group] = window_code
    result.prediction_center_y[group] = int(inputs.prediction_center_y[index])
    result.prediction_center_x[group] = int(inputs.prediction_center_x[index])
    result.prediction_trend_sign[group] = int(inputs.prediction_trend_sign[index])
    result.prediction_configured_delay[group] = int(inputs.prediction_configured_delay[index])
    result.prediction_time_to_consequence[group] = int(inputs.prediction_time_to_consequence[index])
    result.prediction_cue_ambiguity[group] = np.float32(inputs.prediction_cue_ambiguity[index])
    return group


def _replace_result_arrays(result: CompactSummaryResult, **updates: np.ndarray) -> CompactSummaryResult:
    data = result.__dict__.copy()
    data.update(updates)
    data.pop("digest", None)
    return CompactSummaryResult(digest=_stable_json_digest(data), **data)


def _replace_result_timing(result: CompactSummaryResult, timing: Mapping[str, float]) -> CompactSummaryResult:
    data = result.__dict__.copy()
    data.pop("digest", None)
    data["timing"] = dict(timing)
    return CompactSummaryResult(digest=_stable_json_digest(data), **data)


def _replace_result_digest(result: CompactSummaryResult) -> CompactSummaryResult:
    data = result.__dict__.copy()
    data.pop("digest", None)
    return CompactSummaryResult(digest=_stable_json_digest(data), **data)


def _replace_status(result: CompactSummaryResult, *, status: str, fallback_reason: str | None) -> CompactSummaryResult:
    data = result.__dict__.copy()
    data.pop("digest", None)
    data["status"] = status
    data["fallback_reason"] = fallback_reason
    return CompactSummaryResult(digest=_stable_json_digest(data), **data)


def _unique_top(codes: list[int]) -> tuple[int, int, int]:
    counts: dict[int, int] = {}
    for code in codes:
        counts[int(code)] = counts.get(int(code), 0) + 1
    if not counts:
        return 0, 0, 0
    top_code, top_count = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0]
    return int(len(counts)), int(top_code), int(top_count)


def _action_label_code(action: int) -> int:
    if int(action) == int(ACT_MOVE_N):
        return ACTION_LABEL_TO_CODE["MOVE_N"]
    if int(action) == int(ACT_MOVE_S):
        return ACTION_LABEL_TO_CODE["MOVE_S"]
    if int(action) == int(ACT_MOVE_E):
        return ACTION_LABEL_TO_CODE["MOVE_E"]
    if int(action) == int(ACT_MOVE_W):
        return ACTION_LABEL_TO_CODE["MOVE_W"]
    if int(action) == int(ACT_GATHER):
        return ACTION_LABEL_TO_CODE["GATHER"]
    if int(action) == int(ACT_REST):
        return ACTION_LABEL_TO_CODE["REST"]
    if int(action) == int(ACT_ATTACK):
        return ACTION_LABEL_TO_CODE["ATTACK"]
    return ACTION_LABEL_TO_CODE["UNKNOWN"]


def _movement_field(*, action: int, y: int, x: int, center_y: int, center_x: int, height: int, width: int) -> str:
    delta = {
        int(ACT_MOVE_N): (-1, 0),
        int(ACT_MOVE_S): (1, 0),
        int(ACT_MOVE_E): (0, 1),
        int(ACT_MOVE_W): (0, -1),
    }.get(int(action))
    if delta is None:
        return "movement_not_derivable_count"
    pre = max(abs(int(y) - int(center_y)), abs(int(x) - int(center_x)))
    dy, dx = delta
    yy = max(0, min(max(1, int(height)) - 1, int(y) + int(dy)))
    xx = max(0, min(max(1, int(width)) - 1, int(x) + int(dx)))
    post = max(abs(yy - int(center_y)), abs(xx - int(center_x)))
    if post < pre:
        return "movement_toward_count"
    if post > pre:
        return "movement_away_count"
    return "movement_neutral_count"


def _first_payload_mismatch(left: Mapping[str, Any], right: Mapping[str, Any], prefix: str = "") -> dict[str, Any] | None:
    for key in sorted(set(left) | set(right)):
        path = f"{prefix}.{key}" if prefix else str(key)
        if key not in left or key not in right:
            return {"field": path, "python": left.get(key), "candidate": right.get(key), "reason": "missing_key"}
        a = left[key]
        b = right[key]
        if isinstance(a, Mapping) and isinstance(b, Mapping):
            sub = _first_payload_mismatch(a, b, path)
            if sub is not None:
                return sub
        elif a != b:
            return {"field": path, "python": a, "candidate": b, "reason": "value_mismatch"}
    return None


def _round_float_list(values: np.ndarray) -> list[float]:
    return [round(float(value), 6) for value in np.asarray(values, dtype=np.float64).reshape(-1).tolist()]


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
    if isinstance(value, float) and math.isnan(value):
        return None
    raise TypeError(f"unsupported JSON value: {type(value)!r}")
