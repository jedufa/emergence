from __future__ import annotations

"""Default-off prediction-demand action-context instrumentation.

The instrumentation is reporting-only. It observes already-selected primitive
actions near prediction-demand event regions and writes deterministic aggregate
rows under the run directory. It does not mutate rewards, action choices,
selection, reproduction, checkpoint state, or prediction/memory state.
"""

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from env.gridworld_pkg.constants import (
    ACT_ATTACK,
    ACT_GATHER,
    ACT_MOVE_E,
    ACT_MOVE_N,
    ACT_MOVE_S,
    ACT_MOVE_W,
    ACT_REST,
)
from env.gridworld_pkg.patch_world import prediction_demand_enabled


PREDICTION_DEMAND_ACTION_CONTEXT_SCHEMA_VERSION = "prediction_demand_action_context_v1"
PREDICTION_DEMAND_ACTION_CONTEXT_SIGN_SUMMARY_SCHEMA_VERSION = (
    "prediction_demand_action_context_sign_summary_v1"
)
OUT_DIRNAME = "prediction_demand_action_context"
EVENTS_FILENAME = "prediction_demand_action_context_events.jsonl"
GENERATION_SUMMARY_FILENAME = "prediction_demand_action_context_generation_summary.jsonl"
RUN_SUMMARY_FILENAME = "prediction_demand_action_context_run_summary.json"
SCHEMA_FILENAME = "prediction_demand_action_context_schema.json"
SIGN_SUMMARY_FILENAME = "prediction_demand_action_context_sign_summary.json"
SIGN_WINDOW_SUMMARY_FILENAME = "prediction_demand_action_context_sign_window_summary.jsonl"
SIGN_GENERATION_SUMMARY_FILENAME = "prediction_demand_action_context_sign_generation_summary.jsonl"
SIGN_SUMMARY_CSV_FILENAME = "prediction_demand_action_context_sign_summary.csv"

WINDOW_LABELS = (
    "cue",
    "waiting",
    "pre_consequence",
    "active_consequence",
    "post_consequence_recovery",
)
ACTION_LABELS = (
    "MOVE_N",
    "MOVE_S",
    "MOVE_E",
    "MOVE_W",
    "GATHER",
    "REST",
    "ATTACK",
    "UNKNOWN",
)
SIGN_LABELS = ("boom", "bust", "neutral", "unknown")
_ACTION_LABEL_BY_ID = {
    int(ACT_MOVE_N): "MOVE_N",
    int(ACT_MOVE_S): "MOVE_S",
    int(ACT_MOVE_E): "MOVE_E",
    int(ACT_MOVE_W): "MOVE_W",
    int(ACT_GATHER): "GATHER",
    int(ACT_REST): "REST",
    int(ACT_ATTACK): "ATTACK",
}
_MOVEMENT_ACTION_LABELS = {"MOVE_N", "MOVE_S", "MOVE_E", "MOVE_W"}
_PRE_CONSEQUENCE_WINDOWS = {"cue", "waiting", "pre_consequence"}
_MOVE_DELTA = {
    int(ACT_MOVE_N): (-1, 0),
    int(ACT_MOVE_S): (1, 0),
    int(ACT_MOVE_E): (0, 1),
    int(ACT_MOVE_W): (0, -1),
}


def prediction_demand_action_context_enabled(sim_cfg: Any) -> bool:
    return bool(getattr(sim_cfg, "prediction_demand_action_context_enable", False))


def prediction_demand_action_context_schema() -> dict[str, Any]:
    return {
        "schema_version": PREDICTION_DEMAND_ACTION_CONTEXT_SCHEMA_VERSION,
        "read_only": True,
        "default_off": True,
        "checkpoint_mode": "stateless_derived_from_checkpointed_runner_env_population_state",
        "outputs": {
            "events_jsonl": EVENTS_FILENAME,
            "generation_summary_jsonl": GENERATION_SUMMARY_FILENAME,
            "run_summary_json": RUN_SUMMARY_FILENAME,
            "schema_json": SCHEMA_FILENAME,
            "sign_generation_summary_jsonl": SIGN_GENERATION_SUMMARY_FILENAME,
            "sign_summary_csv": SIGN_SUMMARY_CSV_FILENAME,
            "sign_summary_json": SIGN_SUMMARY_FILENAME,
            "sign_window_summary_jsonl": SIGN_WINDOW_SUMMARY_FILENAME,
        },
        "window_labels": list(WINDOW_LABELS),
        "action_labels": list(ACTION_LABELS),
        "sign_labels": list(SIGN_LABELS),
        "sign_summary_schema_version": PREDICTION_DEMAND_ACTION_CONTEXT_SIGN_SUMMARY_SCHEMA_VERSION,
        "required_event_fields": [
            "schema_version",
            "run_id",
            "seed",
            "generation",
            "tick",
            "window_label",
            "event_id",
            "region_center",
            "trend_sign",
            "configured_delay",
            "time_to_consequence",
            "cue_ambiguity",
            "action_label_counts",
            "action_label_fractions",
            "gather_attempts_near_region",
            "movement_toward_count",
            "movement_away_count",
            "movement_neutral_count",
            "exposure_count",
            "unique_agent_count",
            "lineage_count",
            "top_lineage_share",
            "memory_arm",
            "reward_no_intervention_audit",
        ],
    }


def _canonical_line(obj: Mapping[str, Any]) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n"


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _entropy_from_counts(counts: Mapping[str, int]) -> float:
    total = int(sum(int(v) for v in counts.values()))
    if total <= 0:
        return 0.0
    entropy = 0.0
    for value in counts.values():
        count = int(value)
        if count <= 0:
            continue
        p = float(count) / float(total)
        entropy -= p * math.log(p, 2)
    return float(entropy)


def _fractions(counts: Mapping[str, int]) -> dict[str, float]:
    total = int(sum(int(v) for v in counts.values()))
    if total <= 0:
        return {label: 0.0 for label in ACTION_LABELS}
    return {label: float(int(counts.get(label, 0))) / float(total) for label in ACTION_LABELS}


def _label_for_action(action: int) -> str:
    return _ACTION_LABEL_BY_ID.get(int(action), "UNKNOWN")


def _trend_sign(value: float) -> int:
    if float(value) > 0.0:
        return 1
    if float(value) < 0.0:
        return -1
    return 0


def _sign_label(value: Any) -> str:
    try:
        sign = int(value)
    except Exception:
        return "unknown"
    if sign > 0:
        return "boom"
    if sign < 0:
        return "bust"
    if sign == 0:
        return "neutral"
    return "unknown"


def _sign_value(label: str) -> int | None:
    if label == "boom":
        return 1
    if label == "bust":
        return -1
    if label == "neutral":
        return 0
    return None


def _mean(values: Iterable[float]) -> float:
    vals = [float(value) for value in values]
    if not vals:
        return 0.0
    return float(sum(vals) / float(len(vals)))


def _chebyshev(a_y: int, a_x: int, b_y: int, b_x: int) -> int:
    return max(abs(int(a_y) - int(b_y)), abs(int(a_x) - int(b_x)))


def _lineage_id(agent: Any | None) -> str:
    if agent is None:
        return ""
    return str(getattr(agent, "lineage_id", "") or getattr(agent, "founder_id", "") or "")


def _memory_arm(runner: Any) -> str:
    explicit = str(getattr(runner.sim_cfg, "prediction_demand_action_context_memory_arm", "") or "")
    if explicit:
        return explicit
    if bool(getattr(runner.sim_cfg, "ablate_memory", False)) or bool(
        getattr(runner.sim_cfg, "force_memory_disabled", False)
    ):
        return "memory_disabled"
    return "not_configured"


def _prediction_arm(runner: Any) -> str:
    return str(getattr(runner.sim_cfg, "prediction_demand_action_context_prediction_arm", "") or "not_configured")


def _init_arm(runner: Any) -> str:
    return str(getattr(runner.sim_cfg, "prediction_demand_action_context_init_arm", "") or "not_configured")


def _reward_no_intervention_audit() -> dict[str, Any]:
    return {
        "direct_prediction_reward_active": 0,
        "direct_error_reward_active": 0,
        "direct_cognition_reward_active": 0,
        "selection_signal_modified": False,
        "survival_modified": False,
        "reproduction_modified": False,
        "action_choice_modified": False,
        "memory_state_modified": False,
        "prediction_state_modified": False,
        "archive_intervention_active": False,
    }


class PredictionDemandActionContextInstrumentation:
    def __init__(self, runner: Any) -> None:
        self.run_dir = Path(str(getattr(runner, "run_dir", "")))
        self.out_dir = self.run_dir / OUT_DIRNAME
        self.schema_version = str(
            getattr(
                runner.sim_cfg,
                "prediction_demand_action_context_schema_version",
                PREDICTION_DEMAND_ACTION_CONTEXT_SCHEMA_VERSION,
            )
            or PREDICTION_DEMAND_ACTION_CONTEXT_SCHEMA_VERSION
        )
        self.window_radius = max(0, _safe_int(getattr(runner.sim_cfg, "prediction_demand_action_context_window_radius", 1), 1))
        self.include_lineage = bool(
            getattr(runner.sim_cfg, "prediction_demand_action_context_include_lineage", True)
        )
        self.write_events = bool(getattr(runner.sim_cfg, "prediction_demand_action_context_write_events", False))
        self.max_events = max(0, _safe_int(getattr(runner.sim_cfg, "prediction_demand_action_context_max_events_per_run", 0), 0))
        self.sample_tick_period = max(
            1,
            _safe_int(getattr(runner.sim_cfg, "prediction_demand_action_context_sample_tick_period", 1), 1),
        )
        self.sample_generation_period = max(
            1,
            _safe_int(getattr(runner.sim_cfg, "prediction_demand_action_context_sample_generation_period", 1), 1),
        )
        self.sample_first_n_generations = max(
            0,
            _safe_int(getattr(runner.sim_cfg, "prediction_demand_action_context_sample_first_n_generations", 0), 0),
        )
        self.sign_summary_enabled = bool(
            getattr(runner.sim_cfg, "prediction_demand_action_context_sign_summary_enable", False)
        )
        self.sign_summary_schema_version = str(
            getattr(
                runner.sim_cfg,
                "prediction_demand_action_context_sign_summary_schema_version",
                PREDICTION_DEMAND_ACTION_CONTEXT_SIGN_SUMMARY_SCHEMA_VERSION,
            )
            or PREDICTION_DEMAND_ACTION_CONTEXT_SIGN_SUMMARY_SCHEMA_VERSION
        )
        self.sign_summary_include_windows = bool(
            getattr(runner.sim_cfg, "prediction_demand_action_context_sign_summary_include_windows", True)
        )
        self.sign_summary_write_jsonl = bool(
            getattr(runner.sim_cfg, "prediction_demand_action_context_sign_summary_write_jsonl", True)
        )
        self.sign_summary_write_compact_csv = bool(
            getattr(runner.sim_cfg, "prediction_demand_action_context_sign_summary_write_compact_csv", False)
        )
        self.sign_summary_max_events = max(
            0,
            _safe_int(
                getattr(runner.sim_cfg, "prediction_demand_action_context_sign_summary_max_events_per_run", 0),
                0,
            ),
        )
        self._event_count = 0
        self._generation_rows: dict[int, list[dict[str, Any]]] = {}
        self._run_window_counts: Counter[str] = Counter()
        self._run_action_counts: Counter[str] = Counter({label: 0 for label in ACTION_LABELS})
        self._run_rows = 0
        self._run_exposures = 0
        self._run_context_event_ids: set[int] = set()
        self._run_active_after_prior_cue = 0
        self._run_active_without_prior_cue = 0
        self._run_sign_rows: list[dict[str, Any]] = []
        self._run_sign_generations: list[dict[str, Any]] = []
        self._run_sign_source_rows: list[Mapping[str, Any]] = []
        self._sign_summary_row_count = 0
        self.out_dir.mkdir(parents=True, exist_ok=True)
        (self.out_dir / SCHEMA_FILENAME).write_text(
            json.dumps(prediction_demand_action_context_schema(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if self.sign_summary_enabled and self.sign_summary_write_jsonl:
            (self.out_dir / SIGN_WINDOW_SUMMARY_FILENAME).touch()
            (self.out_dir / SIGN_GENERATION_SUMMARY_FILENAME).touch()

    @classmethod
    def maybe_create(cls, runner: Any) -> "PredictionDemandActionContextInstrumentation | None":
        if not prediction_demand_action_context_enabled(getattr(runner, "sim_cfg", None)):
            return None
        return cls(runner)

    def record_pre_env(
        self,
        runner: Any,
        *,
        generation: int,
        tick: int,
        alive: list[Any],
        agent_ids_np: Any,
        actions_np: Any,
    ) -> None:
        if not prediction_demand_enabled(getattr(runner, "env_cfg", None)):
            return
        if not self._sampling_allows(int(generation), int(tick)):
            return
        contexts = self._event_contexts(runner)
        if not contexts:
            return

        env = runner.env
        agent_ids = np.asarray(agent_ids_np, dtype=np.int32)
        actions = np.asarray(actions_np, dtype=np.int32)
        pos_y = np.asarray(getattr(env, "agent_pos_y"), dtype=np.int32)
        pos_x = np.asarray(getattr(env, "agent_pos_x"), dtype=np.int32)
        alive_by_id = {int(getattr(agent, "agent_id")): agent for agent in alive}
        rows: dict[tuple[int, str], dict[str, Any]] = {}
        grid_shape = tuple(int(v) for v in np.asarray(getattr(env, "grid")).shape[:2])

        for idx in range(min(int(agent_ids.size), int(actions.size))):
            aid = int(agent_ids[idx])
            if aid < 0 or aid >= pos_y.shape[0]:
                continue
            y = int(pos_y[aid])
            x = int(pos_x[aid])
            if y < 0 or x < 0:
                continue
            ctx = self._nearest_context(contexts, y, x)
            if ctx is None:
                continue

            action = int(actions[idx])
            label = _label_for_action(action)
            key = (int(ctx["event_id"]), str(ctx["window_label"]))
            row = rows.setdefault(
                key,
                self._empty_row(runner, generation=int(generation), tick=int(tick), context=ctx),
            )
            row["exposure_count"] += 1
            row["_agent_ids"].add(aid)
            row["action_label_counts"][label] += 1
            if label == "GATHER":
                row["gather_attempts_near_region"] += 1
            self._record_movement(row, action=action, y=y, x=x, context=ctx, grid_shape=grid_shape)
            if self.include_lineage:
                lineage = _lineage_id(alive_by_id.get(aid))
                if lineage:
                    row["_lineage_counts"][lineage] += 1

        for key in sorted(rows, key=lambda item: (item[0], item[1])):
            row = rows[key]
            self._finalize_row(row)
            self._generation_rows.setdefault(int(generation), []).append(row)
            if self.write_events and (self.max_events <= 0 or self._event_count < self.max_events):
                self._write_event(row)

    def rows_from_summary_tape(self, runner: Any, tape: Any) -> list[dict[str, Any]]:
        """Build diagnostic rows from a CXXCORE summary tape.

        Default-off CXXCORE-18 adapter. The existing live instrumentation remains
        authoritative unless a caller explicitly uses these diagnostic rows.
        """
        from perf.cxxcore18_live_summary_tape_metrics import prediction_demand_rows_from_tape  # noqa: WPS433

        return prediction_demand_rows_from_tape(self, runner, tape)

    def record_summary_tape(self, runner: Any, tape: Any) -> list[dict[str, Any]]:
        rows = self.rows_from_summary_tape(runner, tape)
        for row in rows:
            self._generation_rows.setdefault(int(getattr(tape, "generation")), []).append(row)
            if self.write_events and (self.max_events <= 0 or self._event_count < self.max_events):
                self._write_event(row)
        return rows

    def flush_generation_summary(self, runner: Any, *, generation: int) -> None:
        rows = list(self._generation_rows.pop(int(generation), []))
        summary = self._summarize_rows(runner, generation=int(generation), rows=rows)
        with (self.out_dir / GENERATION_SUMMARY_FILENAME).open("a", encoding="utf-8") as fh:
            fh.write(_canonical_line(summary))
        if self.sign_summary_enabled:
            self._write_sign_generation_summaries(runner, generation=int(generation), rows=rows)
        self._write_run_summary(runner)

    def _event_contexts(self, runner: Any) -> list[dict[str, Any]]:
        env = runner.env
        event_ids = np.asarray(getattr(env, "patch_prediction_demand_event_id"), dtype=np.int64)
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
        contexts: list[dict[str, Any]] = []
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
            window = self._window_label(active_any=active_any, cue_any=cue_any, min_timer=min_timer, min_age=min_age)
            trend_value = float(np.mean(trend[mask])) if trend.size else 0.0
            sign = int(np.sign(float(np.mean(signs[mask])))) if signs.size else _trend_sign(trend_value)
            event_radius = self._event_radius(mask, cy, cx)
            contexts.append(
                {
                    "event_id": int(event_id),
                    "region_center_y": cy,
                    "region_center_x": cx,
                    "event_radius": event_radius,
                    "_allowed_radius": int(event_radius) + int(self.window_radius),
                    "window_label": window,
                    "trend_sign": int(sign),
                    "configured_delay": int(delay),
                    "time_to_consequence": max(0, int(min_timer)) if not active_any else 0,
                    "cue_ambiguity": float(max(0.0, 1.0 - abs(float(trend_value)))) if cue_any else 0.0,
                    "prior_cue": bool(np.any(prior[mask] > 0)),
                }
            )
        return contexts

    def _window_label(self, *, active_any: bool, cue_any: bool, min_timer: int, min_age: int) -> str:
        if active_any:
            return "active_consequence"
        if not cue_any:
            return "post_consequence_recovery"
        if int(min_age) <= 1:
            return "cue"
        if int(min_timer) <= 1:
            return "pre_consequence"
        return "waiting"

    def _event_radius(self, mask: np.ndarray, cy: int, cx: int) -> int:
        ys, xs = np.nonzero(mask)
        if ys.size <= 0:
            return 0
        return int(max(_chebyshev(int(y), int(x), int(cy), int(cx)) for y, x in zip(ys.tolist(), xs.tolist())))

    def _nearest_context(self, contexts: list[dict[str, Any]], y: int, x: int) -> dict[str, Any] | None:
        best_distance: int | None = None
        best_event_id: int | None = None
        best_context: dict[str, Any] | None = None
        for ctx in contexts:
            distance = _chebyshev(int(y), int(x), int(ctx["region_center_y"]), int(ctx["region_center_x"]))
            allowed = int(ctx.get("_allowed_radius", int(ctx["event_radius"]) + int(self.window_radius)))
            if distance > allowed:
                continue
            event_id = int(ctx["event_id"])
            if best_distance is None or distance < best_distance or (
                distance == best_distance and (best_event_id is None or event_id < best_event_id)
            ):
                best_distance = int(distance)
                best_event_id = event_id
                best_context = ctx
        return best_context

    def _empty_row(self, runner: Any, *, generation: int, tick: int, context: Mapping[str, Any]) -> dict[str, Any]:
        event_id = int(context["event_id"])
        window = str(context["window_label"])
        return {
            "schema_version": self.schema_version,
            "run_id": self._run_id(runner),
            "seed": _safe_int(getattr(runner.sim_cfg, "seed", 0)),
            "generation": int(generation),
            "tick": int(tick),
            "window_id": f"g{int(generation):06d}:t{int(tick):06d}:e{event_id}:{window}",
            "window_label": window,
            "event_id": event_id,
            "region_center": [int(context["region_center_y"]), int(context["region_center_x"])],
            "trend_sign": int(context["trend_sign"]),
            "configured_delay": int(context["configured_delay"]),
            "time_to_consequence": int(context["time_to_consequence"]),
            "cue_ambiguity": float(context["cue_ambiguity"]),
            "action_label_counts": {label: 0 for label in ACTION_LABELS},
            "action_label_fractions": {label: 0.0 for label in ACTION_LABELS},
            "gather_attempts_near_region": 0,
            "movement_toward_count": 0,
            "movement_away_count": 0,
            "movement_neutral_count": 0,
            "movement_not_derivable_count": 0,
            "exposure_count": 0,
            "unique_agent_count": 0,
            "lineage_count": 0,
            "top_lineage_share": 0.0,
            "memory_arm": _memory_arm(runner),
            "reward_no_intervention_audit": _reward_no_intervention_audit(),
            "active_consequence_exposure_after_prior_cue": 0,
            "active_consequence_exposure_without_prior_cue": 0,
            "_agent_ids": set(),
            "_lineage_counts": Counter(),
            "_prior_cue": bool(context.get("prior_cue", False)),
        }

    def _record_movement(
        self,
        row: dict[str, Any],
        *,
        action: int,
        y: int,
        x: int,
        context: Mapping[str, Any],
        grid_shape: tuple[int, int],
    ) -> None:
        delta = _MOVE_DELTA.get(int(action))
        if delta is None:
            row["movement_not_derivable_count"] += 1
            return
        center_y = int(context["region_center_y"])
        center_x = int(context["region_center_x"])
        pre = _chebyshev(int(y), int(x), center_y, center_x)
        dy, dx = delta
        height = max(1, int(grid_shape[0]) if len(grid_shape) > 0 else 1)
        width = max(1, int(grid_shape[1]) if len(grid_shape) > 1 else height)
        yy = max(0, min(height - 1, int(y) + int(dy)))
        xx = max(0, min(width - 1, int(x) + int(dx)))
        post = _chebyshev(yy, xx, center_y, center_x)
        if post < pre:
            row["movement_toward_count"] += 1
        elif post > pre:
            row["movement_away_count"] += 1
        else:
            row["movement_neutral_count"] += 1

    def _finalize_row(self, row: dict[str, Any]) -> None:
        agent_ids = row.pop("_agent_ids", set())
        lineage_counts = row.pop("_lineage_counts", Counter())
        prior_cue = bool(row.pop("_prior_cue", False))
        row["unique_agent_count"] = int(len(agent_ids))
        row["lineage_count"] = int(len(lineage_counts))
        total_lineage = int(sum(lineage_counts.values()))
        if total_lineage > 0:
            _top_lineage, top_count = sorted(lineage_counts.items(), key=lambda kv: (-kv[1], kv[0]))[0]
            row["top_lineage_share"] = float(int(top_count) / float(total_lineage))
        row["action_label_fractions"] = _fractions(row["action_label_counts"])
        if row["window_label"] == "active_consequence":
            if prior_cue:
                row["active_consequence_exposure_after_prior_cue"] = int(row["exposure_count"])
            else:
                row["active_consequence_exposure_without_prior_cue"] = int(row["exposure_count"])

    def _write_event(self, row: Mapping[str, Any]) -> None:
        with (self.out_dir / EVENTS_FILENAME).open("a", encoding="utf-8") as fh:
            fh.write(_canonical_line(row))
        self._event_count += 1

    def _summarize_rows(self, runner: Any, *, generation: int, rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
        rows_list = list(rows)
        by_window: dict[str, dict[str, Any]] = {label: self._zero_window_summary(label) for label in WINDOW_LABELS}
        unique_agents: set[int] = set()
        active_after = 0
        active_without = 0
        for row in rows_list:
            window = str(row.get("window_label", ""))
            target = by_window.setdefault(window, self._zero_window_summary(window))
            exposure = int(row.get("exposure_count", 0) or 0)
            target["row_count"] += 1
            target["exposure_count"] += exposure
            target["unique_agent_count"] += int(row.get("unique_agent_count", 0) or 0)
            target["gather_attempts_near_region"] += int(row.get("gather_attempts_near_region", 0) or 0)
            target["movement_toward_count"] += int(row.get("movement_toward_count", 0) or 0)
            target["movement_away_count"] += int(row.get("movement_away_count", 0) or 0)
            target["movement_neutral_count"] += int(row.get("movement_neutral_count", 0) or 0)
            target["movement_not_derivable_count"] += int(row.get("movement_not_derivable_count", 0) or 0)
            for label, count in dict(row.get("action_label_counts", {})).items():
                target["action_label_counts"][str(label)] += int(count)
            active_after += int(row.get("active_consequence_exposure_after_prior_cue", 0) or 0)
            active_without += int(row.get("active_consequence_exposure_without_prior_cue", 0) or 0)
            self._run_window_counts[window] += 1
            for label, count in dict(row.get("action_label_counts", {})).items():
                self._run_action_counts[str(label)] += int(count)
            self._run_rows += 1
            self._run_exposures += exposure
            self._run_active_after_prior_cue += int(row.get("active_consequence_exposure_after_prior_cue", 0) or 0)
            self._run_active_without_prior_cue += int(row.get("active_consequence_exposure_without_prior_cue", 0) or 0)
        for window, summary in by_window.items():
            total = int(sum(summary["action_label_counts"].values()))
            summary["action_label_fractions"] = _fractions(summary["action_label_counts"])
            summary["pre_consequence_action_entropy"] = (
                _entropy_from_counts(summary["action_label_counts"]) if window == "pre_consequence" else 0.0
            )
            summary["gather_in_cued_region_rate"] = (
                float(summary["gather_attempts_near_region"]) / float(total) if total > 0 else 0.0
            )
            movement_total = int(
                summary["movement_toward_count"] + summary["movement_away_count"] + summary["movement_neutral_count"]
            )
            summary["move_toward_rate"] = (
                float(summary["movement_toward_count"]) / float(movement_total) if movement_total > 0 else 0.0
            )
            summary["move_away_rate"] = (
                float(summary["movement_away_count"]) / float(movement_total) if movement_total > 0 else 0.0
            )
        self._run_context_event_ids.update(
            int(row.get("event_id", 0)) for row in rows_list if int(row.get("event_id", 0) or 0) > 0
        )
        return {
            "schema_version": self.schema_version,
            "run_id": self._run_id(runner),
            "seed": _safe_int(getattr(runner.sim_cfg, "seed", 0)),
            "generation": int(generation),
            "prediction_demand_enable": int(bool(getattr(runner.env_cfg, "prediction_demand_enable", False))),
            "prediction_demand_action_context_enable": 1,
            "rows_emitted_total": int(len(rows_list)),
            "exposure_count_total": int(sum(int(row.get("exposure_count", 0) or 0) for row in rows_list)),
            "event_ids": sorted(int(row.get("event_id", 0) or 0) for row in rows_list),
            "windows": {k: by_window[k] for k in sorted(by_window)},
            "pre_consequence_action_entropy": by_window["pre_consequence"]["pre_consequence_action_entropy"],
            "gather_in_cued_region_rate": by_window["cue"]["gather_in_cued_region_rate"],
            "move_away_before_bust_rate": self._signed_movement_rate(rows_list, sign=-1, direction="away"),
            "move_toward_before_boom_rate": self._signed_movement_rate(rows_list, sign=1, direction="toward"),
            "active_consequence_exposure_after_prior_cue": int(active_after),
            "active_consequence_exposure_without_prior_cue": int(active_without),
            "lineage_diversity_in_prediction_demand_contexts": self._lineage_diversity(rows_list),
            "reward_no_intervention_audit": _reward_no_intervention_audit(),
            "diagnostic_sampling": self._sampling_summary(),
        }

    def _write_sign_generation_summaries(self, runner: Any, *, generation: int, rows: list[Mapping[str, Any]]) -> None:
        window_rows = self._sign_window_summaries(runner, generation=int(generation), rows=rows)
        generation_summary = self._sign_generation_summary(runner, generation=int(generation), rows=rows)
        if self.sign_summary_write_jsonl:
            with (self.out_dir / SIGN_GENERATION_SUMMARY_FILENAME).open("a", encoding="utf-8") as fh:
                fh.write(_canonical_line(generation_summary))
            if self.sign_summary_include_windows:
                with (self.out_dir / SIGN_WINDOW_SUMMARY_FILENAME).open("a", encoding="utf-8") as fh:
                    for row in window_rows:
                        if self.sign_summary_max_events > 0 and self._sign_summary_row_count >= self.sign_summary_max_events:
                            break
                        fh.write(_canonical_line(row))
                        self._sign_summary_row_count += 1
        self._run_sign_generations.append(generation_summary)
        self._run_sign_rows.extend(window_rows)
        self._run_sign_source_rows.extend(rows)

    def _sign_window_summaries(
        self,
        runner: Any,
        *,
        generation: int,
        rows: Iterable[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        rows_list = list(rows)
        summaries: list[dict[str, Any]] = []
        for sign_label in SIGN_LABELS:
            for window in WINDOW_LABELS:
                selected = [
                    row
                    for row in rows_list
                    if _sign_label(row.get("trend_sign", None)) == sign_label
                    and str(row.get("window_label", "")) == str(window)
                ]
                summary = self._sign_summary_base(
                    runner,
                    generation=int(generation),
                    sign_label=sign_label,
                    window_label=window,
                    rows=selected,
                )
                if int(summary["row_count"]) > 0:
                    summaries.append(summary)
        return summaries

    def _sign_generation_summary(
        self,
        runner: Any,
        *,
        generation: int,
        rows: Iterable[Mapping[str, Any]],
    ) -> dict[str, Any]:
        rows_list = list(rows)
        by_sign = {
            sign_label: self._sign_summary_base(
                runner,
                generation=int(generation),
                sign_label=sign_label,
                window_label="all",
                rows=[row for row in rows_list if _sign_label(row.get("trend_sign", None)) == sign_label],
            )
            for sign_label in SIGN_LABELS
        }
        return {
            "schema_version": self.sign_summary_schema_version,
            "run_id": self._run_id(runner),
            "seed": _safe_int(getattr(runner.sim_cfg, "seed", 0)),
            "generation": int(generation),
            "prediction_demand_enable": int(bool(getattr(runner.env_cfg, "prediction_demand_enable", False))),
            "prediction_arm": _prediction_arm(runner),
            "memory_arm": _memory_arm(runner),
            "init_arm": _init_arm(runner),
            "rows_emitted_total": int(len(rows_list)),
            "exposure_count_total": int(sum(int(row.get("exposure_count", 0) or 0) for row in rows_list)),
            "signs": by_sign,
            "reward_no_intervention_audit": _reward_no_intervention_audit(),
            "read_only": True,
            "default_off": True,
            "diagnostic_sampling": self._sampling_summary(),
        }

    def _sign_summary_base(
        self,
        runner: Any,
        *,
        generation: int | None,
        sign_label: str,
        window_label: str,
        rows: Iterable[Mapping[str, Any]],
    ) -> dict[str, Any]:
        rows_list = list(rows)
        action_counts: Counter[str] = Counter({label: 0 for label in ACTION_LABELS})
        event_ids: set[int] = set()
        delay_values: list[float] = []
        time_values: list[float] = []
        ambiguity_values: list[float] = []
        exposure_count = 0
        unique_agent_count = 0
        gather_attempts = 0
        movement_toward = 0
        movement_away = 0
        movement_neutral = 0
        movement_not_derivable = 0
        active_after = 0
        active_without = 0
        lineage_counts: list[float] = []
        top_lineage_shares: list[float] = []
        by_window_exposure: Counter[str] = Counter()
        for row in rows_list:
            event_id = int(row.get("event_id", 0) or 0)
            if event_id > 0:
                event_ids.add(event_id)
            exposure = int(row.get("exposure_count", 0) or 0)
            exposure_count += exposure
            unique_agent_count += int(row.get("unique_agent_count", 0) or 0)
            gather_attempts += int(row.get("gather_attempts_near_region", 0) or 0)
            movement_toward += int(row.get("movement_toward_count", 0) or 0)
            movement_away += int(row.get("movement_away_count", 0) or 0)
            movement_neutral += int(row.get("movement_neutral_count", 0) or 0)
            movement_not_derivable += int(row.get("movement_not_derivable_count", 0) or 0)
            active_after += int(row.get("active_consequence_exposure_after_prior_cue", 0) or 0)
            active_without += int(row.get("active_consequence_exposure_without_prior_cue", 0) or 0)
            delay_values.append(_safe_float(row.get("configured_delay", 0.0), 0.0))
            time_values.append(_safe_float(row.get("time_to_consequence", 0.0), 0.0))
            ambiguity_values.append(_safe_float(row.get("cue_ambiguity", 0.0), 0.0))
            lineage_counts.append(_safe_float(row.get("lineage_count", 0.0), 0.0))
            top_lineage_shares.append(_safe_float(row.get("top_lineage_share", 0.0), 0.0))
            by_window_exposure[str(row.get("window_label", ""))] += exposure
            for label, count in dict(row.get("action_label_counts", {})).items():
                action_counts[str(label)] += int(count)
        action_total = int(sum(action_counts.values()))
        movement_total = int(movement_toward + movement_away + movement_neutral)
        movement_action_count = int(sum(action_counts.get(label, 0) for label in _MOVEMENT_ACTION_LABELS))
        non_movement_count = max(0, action_total - movement_action_count)
        cue_exposure = int(by_window_exposure.get("cue", 0))
        pre_exposure = int(by_window_exposure.get("pre_consequence", 0))
        active_exposure = int(by_window_exposure.get("active_consequence", 0))
        sign_value = _sign_value(sign_label)
        result = {
            "schema_version": self.sign_summary_schema_version,
            "run_id": self._run_id(runner),
            "seed": _safe_int(getattr(runner.sim_cfg, "seed", 0)),
            "generation": int(generation) if generation is not None else None,
            "prediction_demand_enable": int(bool(getattr(runner.env_cfg, "prediction_demand_enable", False))),
            "prediction_arm": _prediction_arm(runner),
            "memory_arm": _memory_arm(runner),
            "init_arm": _init_arm(runner),
            "trend_sign_label": str(sign_label),
            "trend_sign": sign_value,
            "window_label": str(window_label),
            "row_count": int(len(rows_list)),
            "event_id_count": int(len(event_ids)),
            "event_count": int(len(event_ids)),
            "exposure_count": int(exposure_count),
            "unique_agent_count": int(unique_agent_count),
            "configured_delay_mean": _mean(delay_values),
            "mean_time_to_consequence": _mean(time_values),
            "cue_ambiguity_mean": _mean(ambiguity_values),
            "action_label_counts": {label: int(action_counts.get(label, 0)) for label in ACTION_LABELS},
            "action_label_fractions": _fractions(action_counts),
            "pre_consequence_action_entropy": (
                self._pre_consequence_entropy(rows_list) if window_label in ("pre_consequence", "all") else 0.0
            ),
            "gather_attempts_near_region": int(gather_attempts),
            "gather_in_cued_region_rate": float(gather_attempts) / float(action_total) if action_total > 0 else 0.0,
            "rest_rate": float(action_counts.get("REST", 0)) / float(action_total) if action_total > 0 else 0.0,
            "movement_action_fraction": float(movement_action_count) / float(action_total) if action_total > 0 else 0.0,
            "non_movement_fraction": float(non_movement_count) / float(action_total) if action_total > 0 else 0.0,
            "movement_toward_count": int(movement_toward),
            "movement_away_count": int(movement_away),
            "movement_neutral_count": int(movement_neutral),
            "movement_not_derivable_count": int(movement_not_derivable),
            "move_toward_before_boom_count": int(movement_toward) if sign_label == "boom" else 0,
            "move_toward_before_boom_rate": (
                float(movement_toward) / float(movement_total)
                if sign_label == "boom" and movement_total > 0
                else 0.0
            ),
            "move_away_before_boom_count": int(movement_away) if sign_label == "boom" else 0,
            "move_away_before_boom_rate": (
                float(movement_away) / float(movement_total)
                if sign_label == "boom" and movement_total > 0
                else 0.0
            ),
            "move_away_before_bust_count": int(movement_away) if sign_label == "bust" else 0,
            "move_away_before_bust_rate": (
                float(movement_away) / float(movement_total)
                if sign_label == "bust" and movement_total > 0
                else 0.0
            ),
            "move_toward_before_bust_count": int(movement_toward) if sign_label == "bust" else 0,
            "move_toward_before_bust_rate": (
                float(movement_toward) / float(movement_total)
                if sign_label == "bust" and movement_total > 0
                else 0.0
            ),
            "active_consequence_exposure_after_prior_cue": int(active_after),
            "active_consequence_exposure_without_prior_cue": int(active_without),
            "cue_window_to_active_window_exposure_change": int(active_exposure - cue_exposure),
            "pre_consequence_to_active_consequence_exposure_change": int(active_exposure - pre_exposure),
            "exposure_reduction_before_bust_proxy": (
                max(0, int(pre_exposure - active_exposure)) if sign_label == "bust" else 0
            ),
            "opportunity_gain_before_boom_proxy": (
                max(0, int(active_exposure - pre_exposure)) if sign_label == "boom" else 0
            ),
            "lineage_count_in_context_mean": _mean(lineage_counts),
            "top_lineage_share_mean": _mean(top_lineage_shares),
            "lineage_diversity_in_prediction_demand_contexts": _mean(lineage_counts),
            "reward_no_intervention_audit": _reward_no_intervention_audit(),
        }
        return result

    def _pre_consequence_entropy(self, rows: Iterable[Mapping[str, Any]]) -> float:
        counts: Counter[str] = Counter({label: 0 for label in ACTION_LABELS})
        for row in rows:
            if str(row.get("window_label", "")) != "pre_consequence":
                continue
            for label, count in dict(row.get("action_label_counts", {})).items():
                counts[str(label)] += int(count)
        return _entropy_from_counts(counts)

    def _zero_window_summary(self, window: str) -> dict[str, Any]:
        return {
            "window_label": str(window),
            "row_count": 0,
            "exposure_count": 0,
            "unique_agent_count": 0,
            "action_label_counts": {label: 0 for label in ACTION_LABELS},
            "action_label_fractions": {label: 0.0 for label in ACTION_LABELS},
            "gather_attempts_near_region": 0,
            "movement_toward_count": 0,
            "movement_away_count": 0,
            "movement_neutral_count": 0,
            "movement_not_derivable_count": 0,
            "pre_consequence_action_entropy": 0.0,
            "gather_in_cued_region_rate": 0.0,
            "move_toward_rate": 0.0,
            "move_away_rate": 0.0,
        }

    def _signed_movement_rate(self, rows: Iterable[Mapping[str, Any]], *, sign: int, direction: str) -> float:
        selected = [
            row
            for row in rows
            if int(row.get("trend_sign", 0) or 0) == int(sign)
            and str(row.get("window_label", "")) in ("cue", "waiting", "pre_consequence")
        ]
        movement_total = sum(
            int(row.get("movement_toward_count", 0) or 0)
            + int(row.get("movement_away_count", 0) or 0)
            + int(row.get("movement_neutral_count", 0) or 0)
            for row in selected
        )
        if movement_total <= 0:
            return 0.0
        key = "movement_away_count" if direction == "away" else "movement_toward_count"
        return float(sum(int(row.get(key, 0) or 0) for row in selected)) / float(movement_total)

    def _lineage_diversity(self, rows: Iterable[Mapping[str, Any]]) -> float:
        lineage_count_sum = 0.0
        row_count = 0
        for row in rows:
            lineage_count_sum += _safe_float(row.get("lineage_count", 0.0), 0.0)
            row_count += 1
        return lineage_count_sum / float(row_count) if row_count > 0 else 0.0

    def _write_run_summary(self, runner: Any) -> None:
        summary = {
            "schema_version": self.schema_version,
            "run_id": self._run_id(runner),
            "seed": _safe_int(getattr(runner.sim_cfg, "seed", 0)),
            "rows_emitted_total": int(self._run_rows),
            "exposure_count_total": int(self._run_exposures),
            "unique_context_event_count_total": int(len(self._run_context_event_ids)),
            "window_row_counts": dict(sorted(self._run_window_counts.items())),
            "action_label_counts": {label: int(self._run_action_counts.get(label, 0)) for label in ACTION_LABELS},
            "active_consequence_exposure_after_prior_cue": int(self._run_active_after_prior_cue),
            "active_consequence_exposure_without_prior_cue": int(self._run_active_without_prior_cue),
            "reward_no_intervention_audit": _reward_no_intervention_audit(),
            "read_only": True,
            "default_off": True,
            "checkpoint_mode": "stateless_derived_from_checkpointed_runner_env_population_state",
            "diagnostic_sampling": self._sampling_summary(),
        }
        (self.out_dir / RUN_SUMMARY_FILENAME).write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if self.sign_summary_enabled:
            self._write_sign_run_summary(runner)

    def _write_sign_run_summary(self, runner: Any) -> None:
        by_sign = {
            sign_label: self._sign_summary_base(
                runner,
                generation=None,
                sign_label=sign_label,
                window_label="all",
                rows=[row for row in self._run_sign_source_rows if _sign_label(row.get("trend_sign", None)) == sign_label],
            )
            for sign_label in SIGN_LABELS
        }
        by_sign_window = {
            f"{row['trend_sign_label']}|{row['window_label']}": row
            for row in self._run_sign_rows
            if int(row.get("row_count", 0) or 0) > 0
        }
        summary = {
            "schema_version": self.sign_summary_schema_version,
            "run_id": self._run_id(runner),
            "seed": _safe_int(getattr(runner.sim_cfg, "seed", 0)),
            "prediction_demand_enable": int(bool(getattr(runner.env_cfg, "prediction_demand_enable", False))),
            "prediction_arm": _prediction_arm(runner),
            "memory_arm": _memory_arm(runner),
            "init_arm": _init_arm(runner),
            "rows_emitted_total": int(sum(int(row.get("row_count", 0) or 0) for row in self._run_sign_rows)),
            "exposure_count_total": int(sum(int(row.get("exposure_count", 0) or 0) for row in self._run_sign_rows)),
            "sign_generation_summary_count": int(len(self._run_sign_generations)),
            "sign_window_summary_count": int(len(self._run_sign_rows)),
            "signs": by_sign,
            "sign_windows": by_sign_window,
            "reward_no_intervention_audit": _reward_no_intervention_audit(),
            "read_only": True,
            "default_off": True,
            "checkpoint_mode": "stateless_derived_from_checkpointed_runner_env_population_state",
            "diagnostic_sampling": self._sampling_summary(),
        }
        (self.out_dir / SIGN_SUMMARY_FILENAME).write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if self.sign_summary_write_compact_csv:
            self._write_sign_summary_csv()

    def _write_sign_summary_csv(self) -> None:
        import csv

        fields = [
            "generation",
            "trend_sign_label",
            "trend_sign",
            "window_label",
            "row_count",
            "event_id_count",
            "exposure_count",
            "unique_agent_count",
            "configured_delay_mean",
            "mean_time_to_consequence",
            "cue_ambiguity_mean",
            "gather_in_cued_region_rate",
            "rest_rate",
            "movement_action_fraction",
            "move_toward_before_boom_rate",
            "move_away_before_bust_rate",
            "active_consequence_exposure_after_prior_cue",
            "active_consequence_exposure_without_prior_cue",
            "cue_window_to_active_window_exposure_change",
            "pre_consequence_to_active_consequence_exposure_change",
        ]
        with (self.out_dir / SIGN_SUMMARY_CSV_FILENAME).open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            for row in self._run_sign_rows:
                writer.writerow({field: row.get(field, "") for field in fields})

    def _run_id(self, runner: Any) -> str:
        explicit = str(getattr(runner.sim_cfg, "prediction_demand_action_context_run_id", "") or "")
        if explicit:
            return explicit
        return Path(str(getattr(runner, "run_dir", ""))).name

    def _sampling_allows(self, generation: int, tick: int) -> bool:
        gen = int(generation)
        if self.sample_first_n_generations > 0 and gen >= int(self.sample_first_n_generations):
            return False
        if self.sample_generation_period > 1 and gen % int(self.sample_generation_period) != 0:
            return False
        if self.sample_tick_period > 1 and int(tick) % int(self.sample_tick_period) != 0:
            return False
        return True

    def _sampling_summary(self) -> dict[str, Any]:
        return {
            "sample_tick_period": int(self.sample_tick_period),
            "sample_generation_period": int(self.sample_generation_period),
            "sample_first_n_generations": int(self.sample_first_n_generations),
        }
