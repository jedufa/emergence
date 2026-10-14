from __future__ import annotations

"""CXXCORE-19 optional native summary-tape resident helpers.

Default-off diagnostic code. Python remains authoritative for simulation state.
The native boundary receives copied CXXCORE-18 tape arrays and returns resident
row buffers plus compact counters that Python can translate back into the exact
CXXCORE-18 diagnostic row contracts.
"""

from dataclasses import dataclass
import hashlib
import json
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from perf.cxxcore18_live_summary_tape_metrics import (
    CONSUMER_MEMORY_ACTION,
    CONSUMER_PREDICTION_DEMAND,
    LiveSummaryTape,
    compare_consumer_rows,
    memory_action_rows_from_tape,
    prediction_demand_rows_from_tape,
)


SCHEMA_VERSION = "cxxcore19_native_summary_tape_resident_v1"
INPUT_SCHEMA_VERSION = "cxxcore19_native_summary_tape_inputs_v1"
RESULT_SCHEMA_VERSION = "cxxcore19_native_summary_tape_result_v1"
AUDIT_SCHEMA_VERSION = "cxxcore19_native_summary_tape_contract_audit_v1"
DECISION_LABELS = (
    "CXXCORE19-A-NATIVE-SUMMARY-TAPE-RESIDENT-GO",
    "CXXCORE19-B-NATIVE-SUMMARY-TAPE-PARITY-ONLY-NO-SPEED",
    "CXXCORE19-C-DRIVES-CHEMISTRY-BLOCKER",
    "CXXCORE19-D-METRICS-CHECKPOINT-BOUNDARY-BLOCKER",
    "CXXCORE19-E-RNG-PREFLIGHT-REQUIRED",
    "CXXCORE19-F-ABANDON-NATIVE-SUMMARY-TAPE",
    "CXXCORE19-INCONCLUSIVE",
)
DEFAULT_DECISION = "CXXCORE19-INCONCLUSIVE"

PHASES = ("baseline", "cue", "hidden_gap", "active_hazard")
PHASE_TO_CODE = {phase: idx for idx, phase in enumerate(PHASES)}
CODE_TO_PHASE = {idx: phase for phase, idx in PHASE_TO_CODE.items()}
ACTION_BUCKETS = 8


@dataclass(frozen=True)
class NativeSummaryTapeConfig:
    cxxcore19_native_summary_tape_enable: bool = False
    cxxcore19_native_summary_tape_mode: str = "off"
    cxxcore19_native_summary_tape_compare_enable: bool = True
    cxxcore19_native_summary_tape_fail_closed: bool = False
    cxxcore19_native_summary_tape_first_divergence_limit: int = 1


@dataclass(frozen=True)
class NativeSummaryTapeInputs:
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
    phase_codes: np.ndarray
    post_phase_codes: np.ndarray
    prediction_contexts: tuple[dict[str, Any] | None, ...]
    lineage_id: tuple[str, ...]
    founder_id: tuple[str, ...]
    effective_memory: np.ndarray
    memory_abs: np.ndarray
    cost_scalar_total: float
    validation_random_walk_override: bool
    digest: str


@dataclass(frozen=True)
class NativeSummaryTapeResult:
    schema_version: str
    status: str
    fallback_reason: str | None
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
    phase_codes: np.ndarray
    post_phase_codes: np.ndarray
    effective_memory: np.ndarray
    memory_abs: np.ndarray
    phase_action_counts: np.ndarray
    phase_damage_total: np.ndarray
    summary_counts: np.ndarray
    cost_scalar_total: float
    validation_random_walk_override: bool
    timing: dict[str, float]
    digest: str


@dataclass(frozen=True)
class NativeConsumerCompareResult:
    schema_version: str
    mode: str
    status: str
    parity: dict[str, Any]
    fallback_reason: str | None
    first_divergence: dict[str, Any] | None
    native_result: NativeSummaryTapeResult | None
    python_rows: dict[str, list[dict[str, Any]]]
    native_rows: dict[str, list[dict[str, Any]]]
    timing: dict[str, float]
    diagnostics: dict[str, Any]


def config_from_sim_cfg(sim_cfg: Any) -> NativeSummaryTapeConfig:
    return NativeSummaryTapeConfig(
        cxxcore19_native_summary_tape_enable=bool(getattr(sim_cfg, "cxxcore19_native_summary_tape_enable", False)),
        cxxcore19_native_summary_tape_mode=str(getattr(sim_cfg, "cxxcore19_native_summary_tape_mode", "off") or "off"),
        cxxcore19_native_summary_tape_compare_enable=bool(
            getattr(sim_cfg, "cxxcore19_native_summary_tape_compare_enable", True)
        ),
        cxxcore19_native_summary_tape_fail_closed=bool(
            getattr(sim_cfg, "cxxcore19_native_summary_tape_fail_closed", False)
        ),
        cxxcore19_native_summary_tape_first_divergence_limit=max(
            1,
            int(getattr(sim_cfg, "cxxcore19_native_summary_tape_first_divergence_limit", 1) or 1),
        ),
    )


def native_contract_audit_payload() -> dict[str, Any]:
    rows = [
        {
            "surface": "CXXCORE-18 live summary tape arrays",
            "native_owner": "copied row buffers inside optional C++ call",
            "python_owner": "authoritative simulation state and object graph",
            "classification": "native_resident_diagnostic_safe",
            "rng_touched": False,
            "checkpoint_schema_changed": False,
            "metrics_schema_changed": False,
            "notes": "Native receives post-tick copied arrays and does not mutate runner/env/pop objects.",
        },
        {
            "surface": "memory_action_window rows",
            "native_owner": "phase/action/damage row-buffer counters plus copied per-agent fields",
            "python_owner": "final row translation and JSON schema",
            "classification": "summary_tape_consumer_safe",
            "rng_touched": False,
            "checkpoint_schema_changed": False,
            "metrics_schema_changed": False,
            "notes": "Translation must exactly match CXXCORE-18 row shape before any adoption decision.",
        },
        {
            "surface": "prediction_demand_action_context rows",
            "native_owner": "copied action/position fields",
            "python_owner": "prediction-context metadata and movement classification",
            "classification": "partial_native_buffer_safe",
            "rng_touched": False,
            "checkpoint_schema_changed": False,
            "metrics_schema_changed": False,
            "notes": "This pilot does not rehome event-context lookup; it only proves native row-buffer residency is viable.",
        },
        {
            "surface": "drives and chemistry",
            "native_owner": "none",
            "python_owner": "runner dense drive arrays and Agent chemistry objects",
            "classification": "out_of_scope",
            "rng_touched": False,
            "checkpoint_schema_changed": False,
            "metrics_schema_changed": False,
            "notes": "No drive/chemistry ownership is claimed by this branch.",
        },
        {
            "surface": "checkpoint and metrics schemas",
            "native_owner": "none",
            "python_owner": "existing Python checkpoint/metrics code",
            "classification": "unchanged_boundary",
            "rng_touched": False,
            "checkpoint_schema_changed": False,
            "metrics_schema_changed": False,
            "notes": "Native output is report-facing diagnostic data only.",
        },
    ]
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "decision_labels": list(DECISION_LABELS),
        "rows": rows,
        "checkpoint_schema_changed": False,
        "metrics_schema_changed": False,
        "native_rng_added": False,
        "authoritative_state_mutation": False,
    }


def prepare_native_inputs_from_tape(tape: LiveSummaryTape) -> NativeSummaryTapeInputs:
    _validate_cxx18_tape(tape)
    phase_codes = np.asarray([PHASE_TO_CODE.get(str(phase), 0) for phase in tape.memory_phase], dtype=np.int32)
    post_phase_codes = np.asarray([PHASE_TO_CODE.get(str(phase), 0) for phase in tape.memory_post_phase], dtype=np.int32)
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
        "pre_energy": np.ascontiguousarray(tape.pre_energy, dtype=np.float32),
        "post_energy": np.ascontiguousarray(tape.post_energy, dtype=np.float32),
        "pre_health": np.ascontiguousarray(tape.pre_health, dtype=np.float32),
        "post_health": np.ascontiguousarray(tape.post_health, dtype=np.float32),
        "pre_injury": np.ascontiguousarray(tape.pre_injury, dtype=np.float32),
        "post_injury": np.ascontiguousarray(tape.post_injury, dtype=np.float32),
        "pre_scar": np.ascontiguousarray(tape.pre_scar, dtype=np.float32),
        "post_scar": np.ascontiguousarray(tape.post_scar, dtype=np.float32),
        "pre_alive": np.ascontiguousarray(tape.pre_alive, dtype=np.uint8),
        "post_alive": np.ascontiguousarray(tape.post_alive, dtype=np.uint8),
        "phase_codes": np.ascontiguousarray(phase_codes, dtype=np.int32),
        "post_phase_codes": np.ascontiguousarray(post_phase_codes, dtype=np.int32),
        "prediction_contexts": tuple(None if ctx is None else dict(ctx) for ctx in tape.prediction_contexts),
        "lineage_id": tuple(str(value) for value in tape.lineage_id),
        "founder_id": tuple(str(value) for value in tape.founder_id),
        "effective_memory": np.ascontiguousarray(tape.effective_memory, dtype=np.float32),
        "memory_abs": np.ascontiguousarray(tape.memory_abs, dtype=np.float32),
        "cost_scalar_total": float(tape.cost_scalar_total),
        "validation_random_walk_override": bool(tape.validation_random_walk_override),
    }
    return NativeSummaryTapeInputs(digest=_stable_json_digest(payload), **payload)


def run_native_summary_tape_python_reference(
    inputs: NativeSummaryTapeInputs,
    *,
    inject_divergence: bool = False,
) -> NativeSummaryTapeResult:
    t0 = time.perf_counter()
    result = _result_from_arrays(
        inputs=inputs,
        status="ok",
        fallback_reason=None,
        agent_ids=inputs.agent_ids.copy(),
        actions=inputs.actions.copy(),
        pre_y=inputs.pre_y.copy(),
        pre_x=inputs.pre_x.copy(),
        post_y=inputs.post_y.copy(),
        post_x=inputs.post_x.copy(),
        pre_energy=inputs.pre_energy.copy(),
        post_energy=inputs.post_energy.copy(),
        pre_health=inputs.pre_health.copy(),
        post_health=inputs.post_health.copy(),
        pre_injury=inputs.pre_injury.copy(),
        post_injury=inputs.post_injury.copy(),
        pre_scar=inputs.pre_scar.copy(),
        post_scar=inputs.post_scar.copy(),
        pre_alive=inputs.pre_alive.copy(),
        post_alive=inputs.post_alive.copy(),
        phase_codes=inputs.phase_codes.copy(),
        post_phase_codes=inputs.post_phase_codes.copy(),
        effective_memory=inputs.effective_memory.copy(),
        memory_abs=inputs.memory_abs.copy(),
        timing={"backend_setup_s": 0.0, "native_compute_s": 0.0, "python_reference_s": 0.0},
    )
    timing = dict(result.timing)
    timing["python_reference_s"] = float(time.perf_counter() - t0)
    if inject_divergence and result.post_health.shape[0] > 0:
        post_health = result.post_health.copy()
        post_health[0] = np.float32(post_health[0] - 1.0)
        result = _replace_result_arrays(result, post_health=post_health)
    return _replace_result_timing(result, timing)


def run_native_summary_tape(
    inputs: NativeSummaryTapeInputs,
    *,
    native_backend: Callable[[NativeSummaryTapeInputs], NativeSummaryTapeResult | Mapping[str, Any]] | None,
    fail_closed: bool = False,
) -> NativeSummaryTapeResult:
    if native_backend is None:
        if fail_closed:
            raise RuntimeError("CXXCORE-19 native backend unavailable")
        return _fallback_result(inputs, "native_unavailable")
    try:
        out = native_backend(inputs)
        if isinstance(out, NativeSummaryTapeResult):
            return out
        return native_result_from_mapping(inputs, out)
    except Exception:
        if fail_closed:
            raise
        return _fallback_result(inputs, "native_exception")


def native_result_from_mapping(inputs: NativeSummaryTapeInputs, payload: Mapping[str, Any]) -> NativeSummaryTapeResult:
    return _result_from_arrays(
        inputs=inputs,
        status=str(payload.get("status", "ok") or "ok"),
        fallback_reason=payload.get("fallback_reason"),
        agent_ids=np.asarray(payload["agent_ids"], dtype=np.int32),
        actions=np.asarray(payload["actions"], dtype=np.int32),
        pre_y=np.asarray(payload["pre_y"], dtype=np.int32),
        pre_x=np.asarray(payload["pre_x"], dtype=np.int32),
        post_y=np.asarray(payload["post_y"], dtype=np.int32),
        post_x=np.asarray(payload["post_x"], dtype=np.int32),
        pre_energy=np.asarray(payload["pre_energy"], dtype=np.float32),
        post_energy=np.asarray(payload["post_energy"], dtype=np.float32),
        pre_health=np.asarray(payload["pre_health"], dtype=np.float32),
        post_health=np.asarray(payload["post_health"], dtype=np.float32),
        pre_injury=np.asarray(payload["pre_injury"], dtype=np.float32),
        post_injury=np.asarray(payload["post_injury"], dtype=np.float32),
        pre_scar=np.asarray(payload["pre_scar"], dtype=np.float32),
        post_scar=np.asarray(payload["post_scar"], dtype=np.float32),
        pre_alive=np.asarray(payload["pre_alive"], dtype=np.uint8),
        post_alive=np.asarray(payload["post_alive"], dtype=np.uint8),
        phase_codes=np.asarray(payload["phase_codes"], dtype=np.int32),
        post_phase_codes=np.asarray(payload["post_phase_codes"], dtype=np.int32),
        effective_memory=np.asarray(payload["effective_memory"], dtype=np.float32),
        memory_abs=np.asarray(payload["memory_abs"], dtype=np.float32),
        timing=dict(payload.get("timing", {})),
    )


def native_result_to_tape(inputs: NativeSummaryTapeInputs, result: NativeSummaryTapeResult) -> LiveSummaryTape:
    n = int(result.agent_ids.shape[0])
    if len(inputs.prediction_contexts) != n or len(inputs.lineage_id) != n or len(inputs.founder_id) != n:
        raise ValueError("native metadata length does not match native row buffers")
    payload = {
        "schema_version": "cxxcore18_live_tick_summary_tape_v1",
        "generation": int(result.generation),
        "tick": int(result.tick),
        "agent_ids": np.ascontiguousarray(result.agent_ids, dtype=np.int32),
        "actions": np.ascontiguousarray(result.actions, dtype=np.int32),
        "pre_y": np.ascontiguousarray(result.pre_y, dtype=np.int32),
        "pre_x": np.ascontiguousarray(result.pre_x, dtype=np.int32),
        "post_y": np.ascontiguousarray(result.post_y, dtype=np.int32),
        "post_x": np.ascontiguousarray(result.post_x, dtype=np.int32),
        "pre_energy": np.ascontiguousarray(result.pre_energy, dtype=np.float32),
        "post_energy": np.ascontiguousarray(result.post_energy, dtype=np.float32),
        "pre_health": np.ascontiguousarray(result.pre_health, dtype=np.float32),
        "post_health": np.ascontiguousarray(result.post_health, dtype=np.float32),
        "pre_injury": np.ascontiguousarray(result.pre_injury, dtype=np.float32),
        "post_injury": np.ascontiguousarray(result.post_injury, dtype=np.float32),
        "pre_scar": np.ascontiguousarray(result.pre_scar, dtype=np.float32),
        "post_scar": np.ascontiguousarray(result.post_scar, dtype=np.float32),
        "pre_alive": np.asarray(result.pre_alive, dtype=bool),
        "post_alive": np.asarray(result.post_alive, dtype=bool),
        "memory_phase": tuple(CODE_TO_PHASE.get(int(code), "baseline") for code in result.phase_codes.tolist()),
        "memory_post_phase": tuple(CODE_TO_PHASE.get(int(code), "baseline") for code in result.post_phase_codes.tolist()),
        "prediction_contexts": inputs.prediction_contexts,
        "lineage_id": inputs.lineage_id,
        "founder_id": inputs.founder_id,
        "effective_memory": np.ascontiguousarray(result.effective_memory, dtype=np.float32),
        "memory_abs": np.ascontiguousarray(result.memory_abs, dtype=np.float32),
        "cost_scalar_total": float(result.cost_scalar_total),
        "validation_random_walk_override": bool(result.validation_random_walk_override),
    }
    return LiveSummaryTape(digest=_stable_json_digest(payload), **payload)


def rows_from_native_result(
    *,
    inputs: NativeSummaryTapeInputs,
    result: NativeSummaryTapeResult,
    runner: Any,
    memory_instrumentation: Any,
    prediction_instrumentation: Any,
) -> dict[str, list[dict[str, Any]]]:
    tape = native_result_to_tape(inputs, result)
    return {
        CONSUMER_MEMORY_ACTION: memory_action_rows_from_tape(memory_instrumentation, runner, tape),
        CONSUMER_PREDICTION_DEMAND: prediction_demand_rows_from_tape(prediction_instrumentation, runner, tape),
    }


def run_native_consumer_shadow_compare(
    *,
    inputs: NativeSummaryTapeInputs,
    runner: Any,
    memory_instrumentation: Any,
    prediction_instrumentation: Any,
    python_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    native_backend: Callable[[NativeSummaryTapeInputs], NativeSummaryTapeResult | Mapping[str, Any]] | None,
    mode: str,
    compare: bool = True,
    fail_closed: bool = False,
    inject_divergence: bool = False,
) -> NativeConsumerCompareResult:
    timing = {"total_s": 0.0, "native_buffer_s": 0.0, "translation_s": 0.0, "compare_s": 0.0, "fallback_s": 0.0}
    total_t0 = time.perf_counter()
    native = run_native_summary_tape(inputs, native_backend=native_backend, fail_closed=fail_closed)
    if inject_divergence and native.status == "ok" and native.post_health.shape[0] > 0:
        post_health = native.post_health.copy()
        post_health[0] = np.float32(post_health[0] - 1.0)
        native = _replace_result_arrays(native, post_health=post_health)
    timing["native_buffer_s"] = float(native.timing.get("native_compute_s", 0.0) + native.timing.get("python_reference_s", 0.0))
    if native.fallback_reason:
        timing["fallback_s"] = float(time.perf_counter() - total_t0)
        timing["total_s"] = float(time.perf_counter() - total_t0)
        return NativeConsumerCompareResult(
            schema_version=SCHEMA_VERSION,
            mode=mode,
            status="fallback",
            parity={"ok": False, "consumers": {}, "first_divergence": None},
            fallback_reason=native.fallback_reason,
            first_divergence=None,
            native_result=native,
            python_rows={key: [dict(row) for row in value] for key, value in python_rows.items()},
            native_rows={},
            timing=timing,
            diagnostics=_diagnostics(native),
        )
    translate_t0 = time.perf_counter()
    native_rows = rows_from_native_result(
        inputs=inputs,
        result=native,
        runner=runner,
        memory_instrumentation=memory_instrumentation,
        prediction_instrumentation=prediction_instrumentation,
    )
    timing["translation_s"] = float(time.perf_counter() - translate_t0)
    consumers: dict[str, Any] = {}
    first_divergence = None
    ok = True
    if compare:
        cmp_t0 = time.perf_counter()
        for consumer in (CONSUMER_MEMORY_ACTION, CONSUMER_PREDICTION_DEMAND):
            parity = compare_consumer_rows(
                list(python_rows.get(consumer, [])),
                native_rows.get(consumer, []),
                consumer=consumer,
                limit=1,
            )
            consumers[consumer] = parity
            if not bool(parity.get("ok")):
                ok = False
                first_divergence = first_divergence or parity.get("first_divergence")
        timing["compare_s"] = float(time.perf_counter() - cmp_t0)
    else:
        consumers = {consumer: {"ok": True, "comparison_skipped": True} for consumer in (CONSUMER_MEMORY_ACTION, CONSUMER_PREDICTION_DEMAND)}
    timing["total_s"] = float(time.perf_counter() - total_t0)
    return NativeConsumerCompareResult(
        schema_version=SCHEMA_VERSION,
        mode=mode,
        status="ok" if ok else "divergence",
        parity={"ok": ok, "consumers": consumers, "first_divergence": first_divergence},
        fallback_reason=None if ok else "divergence_detected",
        first_divergence=first_divergence,
        native_result=native,
        python_rows={key: [dict(row) for row in value] for key, value in python_rows.items()},
        native_rows=native_rows,
        timing=timing,
        diagnostics=_diagnostics(native),
    )


def result_payload(result: NativeConsumerCompareResult) -> dict[str, Any]:
    return {
        "schema_version": result.schema_version,
        "mode": result.mode,
        "status": result.status,
        "parity": result.parity,
        "fallback_reason": result.fallback_reason,
        "first_divergence": result.first_divergence,
        "python_row_count": {key: len(value) for key, value in result.python_rows.items()},
        "native_row_count": {key: len(value) for key, value in result.native_rows.items()},
        "timing": dict(result.timing),
        "diagnostics": dict(result.diagnostics),
    }


def native_result_payload(result: NativeSummaryTapeResult | None) -> dict[str, Any] | None:
    if result is None:
        return None
    return {
        "schema_version": result.schema_version,
        "status": result.status,
        "fallback_reason": result.fallback_reason,
        "generation": int(result.generation),
        "tick": int(result.tick),
        "rows": int(result.agent_ids.shape[0]),
        "phase_action_counts": result.phase_action_counts.astype(int).tolist(),
        "phase_damage_total": result.phase_damage_total.astype(float).tolist(),
        "summary_counts": result.summary_counts.astype(int).tolist(),
        "cost_scalar_total": float(result.cost_scalar_total),
        "validation_random_walk_override": bool(result.validation_random_walk_override),
        "timing": dict(result.timing),
        "digest": result.digest,
    }


def _result_from_arrays(
    *,
    inputs: NativeSummaryTapeInputs,
    status: str,
    fallback_reason: str | None,
    agent_ids: np.ndarray,
    actions: np.ndarray,
    pre_y: np.ndarray,
    pre_x: np.ndarray,
    post_y: np.ndarray,
    post_x: np.ndarray,
    pre_energy: np.ndarray,
    post_energy: np.ndarray,
    pre_health: np.ndarray,
    post_health: np.ndarray,
    pre_injury: np.ndarray,
    post_injury: np.ndarray,
    pre_scar: np.ndarray,
    post_scar: np.ndarray,
    pre_alive: np.ndarray,
    post_alive: np.ndarray,
    phase_codes: np.ndarray,
    post_phase_codes: np.ndarray,
    effective_memory: np.ndarray,
    memory_abs: np.ndarray,
    timing: Mapping[str, float],
) -> NativeSummaryTapeResult:
    n = int(agent_ids.shape[0])
    arrays = (
        actions,
        pre_y,
        pre_x,
        post_y,
        post_x,
        pre_energy,
        post_energy,
        pre_health,
        post_health,
        pre_injury,
        post_injury,
        pre_scar,
        post_scar,
        pre_alive,
        post_alive,
        phase_codes,
        post_phase_codes,
        effective_memory,
        memory_abs,
    )
    if any(int(np.asarray(array).shape[0]) != n for array in arrays):
        raise ValueError("native result arrays are not aligned")
    phase_action_counts, phase_damage_total, summary_counts = _native_counters(
        actions=actions,
        pre_health=pre_health,
        post_health=post_health,
        pre_alive=pre_alive,
        post_alive=post_alive,
        phase_codes=phase_codes,
        post_phase_codes=post_phase_codes,
        pre_y=pre_y,
        pre_x=pre_x,
        post_y=post_y,
        post_x=post_x,
    )
    payload = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": str(status),
        "fallback_reason": fallback_reason,
        "generation": int(inputs.generation),
        "tick": int(inputs.tick),
        "agent_ids": np.ascontiguousarray(agent_ids, dtype=np.int32),
        "actions": np.ascontiguousarray(actions, dtype=np.int32),
        "pre_y": np.ascontiguousarray(pre_y, dtype=np.int32),
        "pre_x": np.ascontiguousarray(pre_x, dtype=np.int32),
        "post_y": np.ascontiguousarray(post_y, dtype=np.int32),
        "post_x": np.ascontiguousarray(post_x, dtype=np.int32),
        "pre_energy": np.ascontiguousarray(pre_energy, dtype=np.float32),
        "post_energy": np.ascontiguousarray(post_energy, dtype=np.float32),
        "pre_health": np.ascontiguousarray(pre_health, dtype=np.float32),
        "post_health": np.ascontiguousarray(post_health, dtype=np.float32),
        "pre_injury": np.ascontiguousarray(pre_injury, dtype=np.float32),
        "post_injury": np.ascontiguousarray(post_injury, dtype=np.float32),
        "pre_scar": np.ascontiguousarray(pre_scar, dtype=np.float32),
        "post_scar": np.ascontiguousarray(post_scar, dtype=np.float32),
        "pre_alive": np.ascontiguousarray(pre_alive, dtype=np.uint8),
        "post_alive": np.ascontiguousarray(post_alive, dtype=np.uint8),
        "phase_codes": np.ascontiguousarray(phase_codes, dtype=np.int32),
        "post_phase_codes": np.ascontiguousarray(post_phase_codes, dtype=np.int32),
        "effective_memory": np.ascontiguousarray(effective_memory, dtype=np.float32),
        "memory_abs": np.ascontiguousarray(memory_abs, dtype=np.float32),
        "phase_action_counts": phase_action_counts,
        "phase_damage_total": phase_damage_total,
        "summary_counts": summary_counts,
        "cost_scalar_total": float(inputs.cost_scalar_total),
        "validation_random_walk_override": bool(inputs.validation_random_walk_override),
        "timing": dict(timing),
    }
    return NativeSummaryTapeResult(digest=_stable_json_digest(payload), **payload)


def _fallback_result(inputs: NativeSummaryTapeInputs, reason: str) -> NativeSummaryTapeResult:
    timing = {"backend_setup_s": 0.0, "native_compute_s": 0.0, "fallback_s": 0.0}
    return _result_from_arrays(
        inputs=inputs,
        status="fallback",
        fallback_reason=reason,
        agent_ids=inputs.agent_ids.copy(),
        actions=inputs.actions.copy(),
        pre_y=inputs.pre_y.copy(),
        pre_x=inputs.pre_x.copy(),
        post_y=inputs.post_y.copy(),
        post_x=inputs.post_x.copy(),
        pre_energy=inputs.pre_energy.copy(),
        post_energy=inputs.post_energy.copy(),
        pre_health=inputs.pre_health.copy(),
        post_health=inputs.post_health.copy(),
        pre_injury=inputs.pre_injury.copy(),
        post_injury=inputs.post_injury.copy(),
        pre_scar=inputs.pre_scar.copy(),
        post_scar=inputs.post_scar.copy(),
        pre_alive=inputs.pre_alive.copy(),
        post_alive=inputs.post_alive.copy(),
        phase_codes=inputs.phase_codes.copy(),
        post_phase_codes=inputs.post_phase_codes.copy(),
        effective_memory=inputs.effective_memory.copy(),
        memory_abs=inputs.memory_abs.copy(),
        timing=timing,
    )


def _replace_result_arrays(result: NativeSummaryTapeResult, **updates: np.ndarray) -> NativeSummaryTapeResult:
    data = result.__dict__.copy()
    data.update(updates)
    data.pop("digest", None)
    phase_action_counts, phase_damage_total, summary_counts = _native_counters(
        actions=data["actions"],
        pre_health=data["pre_health"],
        post_health=data["post_health"],
        pre_alive=data["pre_alive"],
        post_alive=data["post_alive"],
        phase_codes=data["phase_codes"],
        post_phase_codes=data["post_phase_codes"],
        pre_y=data["pre_y"],
        pre_x=data["pre_x"],
        post_y=data["post_y"],
        post_x=data["post_x"],
    )
    data["phase_action_counts"] = phase_action_counts
    data["phase_damage_total"] = phase_damage_total
    data["summary_counts"] = summary_counts
    return NativeSummaryTapeResult(digest=_stable_json_digest(data), **data)


def _replace_result_timing(result: NativeSummaryTapeResult, timing: Mapping[str, float]) -> NativeSummaryTapeResult:
    data = result.__dict__.copy()
    data.pop("digest", None)
    data["timing"] = dict(timing)
    return NativeSummaryTapeResult(digest=_stable_json_digest(data), **data)


def _native_counters(
    *,
    actions: np.ndarray,
    pre_health: np.ndarray,
    post_health: np.ndarray,
    pre_alive: np.ndarray,
    post_alive: np.ndarray,
    phase_codes: np.ndarray,
    post_phase_codes: np.ndarray,
    pre_y: np.ndarray,
    pre_x: np.ndarray,
    post_y: np.ndarray,
    post_x: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    phase_action_counts = np.zeros((len(PHASES), ACTION_BUCKETS), dtype=np.int32)
    phase_damage_total = np.zeros((len(PHASES),), dtype=np.float32)
    summary_counts = np.zeros((8,), dtype=np.int32)
    for i, action_raw in enumerate(np.asarray(actions, dtype=np.int32).tolist()):
        phase = int(phase_codes[i])
        action = int(action_raw)
        if 0 <= phase < len(PHASES) and 0 <= action < ACTION_BUCKETS:
            phase_action_counts[phase, action] += 1
        if bool(pre_alive[i]) and not bool(post_alive[i]):
            summary_counts[0] += 1
        if not bool(pre_alive[i]) and bool(post_alive[i]):
            summary_counts[1] += 1
        if int(pre_y[i]) != int(post_y[i]) or int(pre_x[i]) != int(post_x[i]):
            summary_counts[2] += 1
        damage = float(pre_health[i]) - float(post_health[i])
        if damage > 0.0:
            summary_counts[3] += 1
            if 0 <= phase < len(PHASES):
                phase_damage_total[phase] += np.float32(damage)
        post_phase = int(post_phase_codes[i])
        if phase != PHASE_TO_CODE["active_hazard"] and post_phase == PHASE_TO_CODE["active_hazard"]:
            summary_counts[4] += 1
        if phase == PHASE_TO_CODE["active_hazard"] and post_phase != PHASE_TO_CODE["active_hazard"]:
            summary_counts[5] += 1
        if phase in (PHASE_TO_CODE["cue"], PHASE_TO_CODE["hidden_gap"]) and post_phase == PHASE_TO_CODE["baseline"]:
            summary_counts[6] += 1
    summary_counts[7] = int(np.asarray(actions).shape[0])
    return phase_action_counts, phase_damage_total, summary_counts


def _diagnostics(result: NativeSummaryTapeResult) -> dict[str, Any]:
    return {
        "native_rng_added": False,
        "checkpoint_schema_changed": False,
        "metrics_schema_changed": False,
        "authoritative_state_mutation": False,
        "native_rows": int(result.agent_ids.shape[0]),
        "summary_counts": result.summary_counts.astype(int).tolist(),
    }


def _validate_cxx18_tape(tape: LiveSummaryTape) -> None:
    if not isinstance(tape, LiveSummaryTape):
        raise TypeError("expected LiveSummaryTape")
    n = int(tape.agent_ids.shape[0])
    arrays = (
        tape.actions,
        tape.pre_y,
        tape.pre_x,
        tape.post_y,
        tape.post_x,
        tape.pre_energy,
        tape.post_energy,
        tape.pre_health,
        tape.post_health,
        tape.pre_injury,
        tape.post_injury,
        tape.pre_scar,
        tape.post_scar,
        tape.pre_alive,
        tape.post_alive,
        tape.effective_memory,
        tape.memory_abs,
    )
    if any(int(np.asarray(array).shape[0]) != n for array in arrays):
        raise ValueError("CXXCORE-18 tape arrays are not aligned")
    if len(tape.memory_phase) != n or len(tape.memory_post_phase) != n or len(tape.prediction_contexts) != n:
        raise ValueError("CXXCORE-18 tape metadata is not aligned")


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
