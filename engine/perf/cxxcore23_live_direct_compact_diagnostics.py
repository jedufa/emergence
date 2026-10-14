from __future__ import annotations

"""CXXCORE-23 live direct compact diagnostics adapter.

Default-off diagnostics-only integration. Python simulation state remains
authoritative; this adapter reads pre/post env state and emits the existing
diagnostic row schemas through the existing instrumentation objects.
"""

from dataclasses import dataclass
import time
from collections import Counter
from typing import Any, Mapping, Sequence

import numpy as np

from env.gridworld_pkg.constants import N_ACTIONS
from perf.cxxcore18_live_summary_tape_metrics import (
    CONSUMER_MEMORY_ACTION,
    CONSUMER_PREDICTION_DEMAND,
)
from perf.cxxcore20_compact_summary_consumers import (
    ACTION_LABEL_TO_CODE,
    ACTION_BUCKETS,
    CODE_TO_WINDOW,
    MEMORY_SCALAR_FIELDS,
    PREDICTION_SCALAR_FIELDS,
    CompactSummaryResult,
    compare_compact_results,
    compact_diagnostic_summary,
    run_python_compact_reference,
)
from perf.cxxcore22_summary_tape_build_boundary import (
    DirectCompactPreSnapshot,
    build_direct_compact_inputs_from_snapshot,
    capture_direct_compact_pre_snapshot,
    parity_against_row_semantics,
    run_direct_compact_summary,
)
from simulation.prediction_demand_action_context import ACTION_LABELS


SCHEMA_VERSION = "cxxcore23_live_direct_compact_diagnostics_v1"
DECISION_LABELS = (
    "CXXCORE23-A-LIVE-DIRECT-COMPACT-DIAGNOSTICS-GO",
    "CXXCORE23-B-PYTHON-DIRECT-GO-NATIVE-LATER",
    "CXXCORE23-C-PARITY-ONLY-NO-SPEED",
    "CXXCORE23-D-LIVE-DIRECT-REGRESSION",
    "CXXCORE23-E-LIVE-INTEGRATION-BLOCKED",
    "CXXCORE23-F-INVALID-EVIDENCE",
)
MODES = ("off", "python_direct", "native_direct")


@dataclass(frozen=True)
class Cxxcore23DirectDiagnosticsConfig:
    enable: bool = False
    mode: str = "off"
    guarded: bool = False
    compare_every_n: int = 0
    force_native_missing: bool = False
    force_native_exception: bool = False
    inject_divergence: bool = False


def config_from_sim_cfg(sim_cfg: Any) -> Cxxcore23DirectDiagnosticsConfig:
    mode = str(getattr(sim_cfg, "cxxcore_direct_compact_diagnostics_mode", "off") or "off")
    if mode not in MODES:
        mode = "off"
    return Cxxcore23DirectDiagnosticsConfig(
        enable=bool(getattr(sim_cfg, "cxxcore_direct_compact_diagnostics_enable", False)),
        mode=mode,
        guarded=bool(getattr(sim_cfg, "cxxcore_direct_compact_diagnostics_guarded", False)),
        compare_every_n=max(0, int(getattr(sim_cfg, "cxxcore_direct_compact_diagnostics_compare_every_n", 0) or 0)),
        force_native_missing=bool(getattr(sim_cfg, "cxxcore_direct_compact_diagnostics_force_native_missing", False)),
        force_native_exception=bool(getattr(sim_cfg, "cxxcore_direct_compact_diagnostics_force_native_exception", False)),
        inject_divergence=bool(getattr(sim_cfg, "cxxcore_direct_compact_diagnostics_inject_divergence", False)),
    )


def live_integration_audit_payload() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "surfaces": [
            {
                "surface": "memory-action window diagnostics",
                "current_row_path": "record_pre_env stores per-phase Python rows; record_post_env finalizes and writes rows",
                "direct_compact_replacement": "capture direct compact pre snapshot; aggregate compact result post-env; render same output rows from compact result",
                "parity_debug_adapter": "sampled LiveSummaryTape/row adapter via CXXCORE-22 parity helper",
                "risk": "row rendering must preserve schema and aggregate values",
            },
            {
                "surface": "prediction-demand action-context diagnostics",
                "current_row_path": "record_pre_env computes contexts and stores generation rows",
                "direct_compact_replacement": "direct compact input carries encoded prediction context arrays; compact result renders same generation rows",
                "parity_debug_adapter": "sampled row-compatible adapter at compare cadence",
                "risk": "sign-summary downstream relies on row fields; rendered row schema is preserved",
            },
            {
                "surface": "LiveSummaryTape construction",
                "current_row_path": "built for summary/tape paths and compact translation",
                "direct_compact_replacement": "not built on no-compare direct path",
                "parity_debug_adapter": "built only on guarded sampled parity checks",
                "risk": "guard cadence must be explicit and default off",
            },
            {
                "surface": "metrics/checkpoint/RNG",
                "current_row_path": "Python-owned runner objects and existing diagnostic files",
                "direct_compact_replacement": "read-only transient adapter; no checkpoint state and no native RNG",
                "parity_debug_adapter": "first divergence captured in adapter diagnostics",
                "risk": "must remain default off",
            },
        ],
        "checkpoint_schema_changed": False,
        "metrics_schema_changed": False,
        "native_rng_added": False,
        "authoritative_state_mutation": False,
    }


class LiveDirectCompactDiagnostics:
    def __init__(self, runner: Any, *, native_backend: Any | None = None) -> None:
        self.config = config_from_sim_cfg(getattr(runner, "sim_cfg", None))
        self.native_backend = native_backend
        self._pending: dict[tuple[int, int], DirectCompactPreSnapshot] = {}
        self._tick_index = 0
        self._parity_samples = 0
        self._failed_parity_samples = 0
        self._fallback_count = 0
        self._divergences: list[dict[str, Any]] = []
        self._full_row_adapter_build_count = 0
        self.timing = {
            "event_capture_s": 0.0,
            "direct_compact_buffer_s": 0.0,
            "python_compact_aggregation_s": 0.0,
            "native_compact_aggregation_s": 0.0,
            "row_render_s": 0.0,
            "parity_adapter_s": 0.0,
        }

    @classmethod
    def maybe_create(cls, runner: Any, *, native_backend: Any | None = None) -> "LiveDirectCompactDiagnostics | None":
        cfg = config_from_sim_cfg(getattr(runner, "sim_cfg", None))
        if not cfg.enable or cfg.mode == "off":
            return None
        return cls(runner, native_backend=native_backend)

    def record_pre_env(
        self,
        runner: Any,
        *,
        generation: int,
        tick: int,
        alive: Sequence[Any],
        agent_ids_np: Any,
        actions_np: Any,
    ) -> bool:
        if not self.config.enable or self.config.mode == "off":
            return False
        t0 = time.perf_counter()
        snapshot = capture_direct_compact_pre_snapshot(
            runner,
            generation=int(generation),
            tick=int(tick),
            alive=alive,
            agent_ids_np=agent_ids_np,
            actions_np=actions_np,
        )
        self.timing["event_capture_s"] += float(time.perf_counter() - t0)
        self._pending[(int(generation), int(tick))] = snapshot
        return True

    def record_post_env(
        self,
        runner: Any,
        *,
        generation: int,
        tick: int,
        memory_instrumentation: Any | None,
        prediction_instrumentation: Any | None,
    ) -> bool:
        snapshot = self._pending.pop((int(generation), int(tick)), None)
        if snapshot is None:
            return False
        compact_t0 = time.perf_counter()
        inputs = build_direct_compact_inputs_from_snapshot(runner, snapshot, grid_shape=runner.env.grid.shape)
        self.timing["direct_compact_buffer_s"] += float(time.perf_counter() - compact_t0)
        backend = self._native_backend()
        use_native = self.config.mode == "native_direct"
        result, agg_timing, fallbacks = run_direct_compact_summary(
            inputs,
            native_backend=backend,
            use_native=use_native,
            fail_closed=False,
        )
        self.timing["native_compact_aggregation_s"] += float(agg_timing.get("native_aggregation_s", 0.0))
        self.timing["python_compact_aggregation_s"] += float(agg_timing.get("python_aggregation_s", 0.0))
        self._fallback_count += int(fallbacks)
        if self.config.inject_divergence:
            expected = run_python_compact_reference(inputs, inject_divergence=True)
            parity = compare_compact_results(expected, result)
            self._record_divergence("injected_divergence", parity)
        if self._should_compare():
            parity_t0 = time.perf_counter()
            self._full_row_adapter_build_count += 1
            row_parity = parity_against_row_semantics(
                runner,
                snapshot,
                inputs,
                memory_instrumentation=memory_instrumentation,
                prediction_instrumentation=prediction_instrumentation,
            )
            expected = run_python_compact_reference(inputs)
            compact_parity = compare_compact_results(expected, result)
            self.timing["parity_adapter_s"] += float(time.perf_counter() - parity_t0)
            self._parity_samples += 1
            if not bool(row_parity.get("ok")) or not bool(compact_parity.get("ok")):
                self._failed_parity_samples += 1
                self._record_divergence(
                    "guarded_parity",
                    {"row_parity": row_parity, "compact_parity": compact_parity},
                )
        render_t0 = time.perf_counter()
        if memory_instrumentation is not None:
            for row in memory_rows_from_compact(runner, memory_instrumentation, inputs, result):
                memory_instrumentation._write_event(row)
        if prediction_instrumentation is not None:
            rows = prediction_rows_from_compact(runner, prediction_instrumentation, inputs, result)
            for row in rows:
                prediction_instrumentation._generation_rows.setdefault(int(generation), []).append(row)
                if prediction_instrumentation.write_events and (
                    prediction_instrumentation.max_events <= 0
                    or prediction_instrumentation._event_count < prediction_instrumentation.max_events
                ):
                    prediction_instrumentation._write_event(row)
        self.timing["row_render_s"] += float(time.perf_counter() - render_t0)
        self._tick_index += 1
        return True

    def diagnostics(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "config": self.config.__dict__,
            "timing": dict(self.timing),
            "parity_sample_count": int(self._parity_samples),
            "failed_parity_sample_count": int(self._failed_parity_samples),
            "fallback_count": int(self._fallback_count),
            "full_row_adapter_build_count": int(self._full_row_adapter_build_count),
            "divergences": list(self._divergences),
            "checkpoint_schema_changed": False,
            "metrics_schema_changed": False,
            "native_rng_added": False,
            "authoritative_state_mutation": False,
        }

    def _native_backend(self) -> Any | None:
        if self.config.force_native_missing:
            return None
        if self.config.force_native_exception:
            def _raise(_inputs: Any) -> Any:
                raise RuntimeError("forced CXXCORE-23 native exception")

            return _raise
        return self.native_backend

    def _should_compare(self) -> bool:
        if not self.config.guarded:
            return False
        if self.config.compare_every_n <= 0:
            return False
        return int(self._tick_index) % int(self.config.compare_every_n) == 0

    def _record_divergence(self, reason: str, parity: Mapping[str, Any]) -> None:
        self._divergences.append({"reason": reason, "parity": dict(parity)})


def memory_rows_from_compact(
    runner: Any,
    instrumentation: Any,
    inputs: Any,
    result: CompactSummaryResult,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    labels = list(getattr(inputs, "lineage_labels", ()))
    founders = list(getattr(inputs, "founder_labels", ()))
    summary = compact_diagnostic_summary(result)["memory"]
    for phase in sorted(summary):
        phase_idx = _phase_code_for_label(inputs, phase)
        row = instrumentation._empty_row(runner, generation=int(inputs.generation), tick=int(inputs.tick), phase=phase)
        phase_summary = summary[phase]
        row["agent_count"] = int(phase_summary["agent_count"])
        row["action_histogram"] = {str(i): int(result.memory_action_counts[phase_idx, i]) for i in range(N_ACTIONS)}
        row["movement_action_count"] = int(phase_summary["movement_action_count"])
        row["gather_action_count"] = int(phase_summary["gather_action_count"])
        row["idle_noop_count"] = int(phase_summary["idle_noop_count"])
        row["unknown_other_action_count"] = int(phase_summary["unknown_other_action_count"])
        row["lineage_context"] = {
            "lineage_count": int(result.memory_lineage_unique_count[phase_idx]),
            "effective_alive_lineages": int(result.memory_lineage_unique_count[phase_idx]),
            "top_lineage_id": _label_for_code(labels, int(result.memory_top_lineage_code[phase_idx])),
            "top_lineage_share": float(phase_summary["top_lineage_share"]),
            "founder_ids": _founders_for_phase(inputs, founders, phase_idx),
        }
        row["memory_summary"]["effective_memory_mean"] = float(phase_summary["effective_memory_mean"])
        row["memory_summary"]["memory_state_abs_mean"] = float(phase_summary["memory_state_abs_mean"])
        row["memory_summary"]["memory_agent_count"] = int(phase_summary["memory_agent_count"])
        row["exposure_damage_summary"]["damage_event_count"] = int(phase_summary["damage_event_count"])
        row["exposure_damage_summary"]["damage_total"] = float(phase_summary["damage_total"])
        row["exposure_damage_summary"]["damage_with_prior_cue_context_count"] = int(
            phase_summary["damage_with_prior_cue_context_count"]
        )
        row["exposure_damage_summary"]["active_hazard_exposure_count"] = int(phase_summary["active_hazard_exposure_count"])
        row["exposure_damage_summary"]["pre_hazard_exit_count"] = int(phase_summary["pre_hazard_exit_count"])
        row["exposure_damage_summary"]["hazard_entry_count"] = int(phase_summary["hazard_entry_count"])
        row["exposure_damage_summary"]["hazard_exit_count"] = int(phase_summary["hazard_exit_count"])
        for key in ("_agent_ids", "_pre_positions", "_pre_health", "_lineage_counts", "_founder_ids", "_effective_memory_values", "_memory_abs_values"):
            row.pop(key, None)
        rows.append(row)
    return rows


def prediction_rows_from_compact(
    runner: Any,
    instrumentation: Any,
    inputs: Any,
    result: CompactSummaryResult,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx in range(int(result.prediction_group_count)):
        context = {
            "event_id": int(result.prediction_event_id[idx]),
            "window_label": CODE_TO_WINDOW.get(int(result.prediction_window_code[idx]), "unknown"),
            "region_center_y": int(result.prediction_center_y[idx]),
            "region_center_x": int(result.prediction_center_x[idx]),
            "trend_sign": int(result.prediction_trend_sign[idx]),
            "configured_delay": int(result.prediction_configured_delay[idx]),
            "time_to_consequence": int(result.prediction_time_to_consequence[idx]),
            "cue_ambiguity": float(result.prediction_cue_ambiguity[idx]),
            "prior_cue": bool(
                int(result.prediction_scalar_counts[idx, PREDICTION_SCALAR_FIELDS.index("active_consequence_exposure_after_prior_cue")])
                > 0
            ),
        }
        row = instrumentation._empty_row(runner, generation=int(inputs.generation), tick=int(inputs.tick), context=context)
        for label in ACTION_LABELS:
            row["action_label_counts"][label] = int(result.prediction_action_counts[idx, ACTION_LABEL_TO_CODE[label]])
        row["gather_attempts_near_region"] = int(
            result.prediction_scalar_counts[idx, PREDICTION_SCALAR_FIELDS.index("gather_attempts_near_region")]
        )
        row["movement_toward_count"] = int(
            result.prediction_scalar_counts[idx, PREDICTION_SCALAR_FIELDS.index("movement_toward_count")]
        )
        row["movement_away_count"] = int(
            result.prediction_scalar_counts[idx, PREDICTION_SCALAR_FIELDS.index("movement_away_count")]
        )
        row["movement_neutral_count"] = int(
            result.prediction_scalar_counts[idx, PREDICTION_SCALAR_FIELDS.index("movement_neutral_count")]
        )
        row["movement_not_derivable_count"] = int(
            result.prediction_scalar_counts[idx, PREDICTION_SCALAR_FIELDS.index("movement_not_derivable_count")]
        )
        row["exposure_count"] = int(result.prediction_scalar_counts[idx, PREDICTION_SCALAR_FIELDS.index("exposure_count")])
        row["unique_agent_count"] = int(
            result.prediction_scalar_counts[idx, PREDICTION_SCALAR_FIELDS.index("unique_agent_count")]
        )
        row["lineage_count"] = int(result.prediction_scalar_counts[idx, PREDICTION_SCALAR_FIELDS.index("lineage_count")])
        exposure = int(row["exposure_count"])
        top_count = int(result.prediction_scalar_counts[idx, PREDICTION_SCALAR_FIELDS.index("top_lineage_count")])
        row["top_lineage_share"] = float(top_count / exposure) if exposure > 0 else 0.0
        row["active_consequence_exposure_after_prior_cue"] = int(
            result.prediction_scalar_counts[
                idx,
                PREDICTION_SCALAR_FIELDS.index("active_consequence_exposure_after_prior_cue"),
            ]
        )
        row["active_consequence_exposure_without_prior_cue"] = int(
            result.prediction_scalar_counts[
                idx,
                PREDICTION_SCALAR_FIELDS.index("active_consequence_exposure_without_prior_cue"),
            ]
        )
        row["action_label_fractions"] = _fractions(row["action_label_counts"])
        for key in ("_agent_ids", "_lineage_counts", "_prior_cue"):
            row.pop(key, None)
        rows.append(row)
    rows.sort(key=lambda item: (int(item.get("event_id", 0)), str(item.get("window_label", ""))))
    return rows


def _phase_code_for_label(inputs: Any, phase: str) -> int:
    codes = [int(code) for code in np.asarray(inputs.phase_codes, dtype=np.int32).tolist()]
    phases = {
        0: "baseline",
        1: "cue",
        2: "hidden_gap",
        3: "active_hazard",
    }
    for code, label in phases.items():
        if label == str(phase) and code in codes:
            return code
    return next((code for code, label in phases.items() if label == str(phase)), 0)


def _label_for_code(labels: Sequence[str], code: int) -> str:
    if int(code) <= 0:
        return ""
    idx = int(code) - 1
    if idx < 0 or idx >= len(labels):
        return ""
    return str(labels[idx])


def _founders_for_phase(inputs: Any, founders: Sequence[str], phase_code: int) -> list[str]:
    codes = set()
    for i, code in enumerate(np.asarray(inputs.phase_codes, dtype=np.int32).tolist()):
        if int(code) == int(phase_code):
            founder = int(inputs.founder_codes[i])
            if founder > 0:
                codes.add(founder)
    return sorted(_label_for_code(founders, code) for code in codes if _label_for_code(founders, code))


def _fractions(counts: Mapping[str, int]) -> dict[str, float]:
    total = int(sum(int(v) for v in counts.values()))
    if total <= 0:
        return {str(k): 0.0 for k in counts}
    return {str(k): float(int(v) / float(total)) for k, v in counts.items()}
