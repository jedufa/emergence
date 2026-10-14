from __future__ import annotations

"""Default-off delayed-hazard memory/action event-window instrumentation.

The instrumentation is read-only: it observes runner/env state after policy
selection and after delayed-hazard damage application, then writes compact
deterministic aggregate rows under the run directory.
"""

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch

from agents.cognition import effective_memory_dim_for_genome, memory_ablation_flags
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


MEMORY_ACTION_WINDOW_SCHEMA_VERSION = "memory_action_window_v1"
EVENTS_FILENAME = "memory_action_window_events.jsonl"
GENERATION_SUMMARY_FILENAME = "memory_action_window_generation_summary.jsonl"
RUN_SUMMARY_FILENAME = "memory_action_window_run_summary.json"
SCHEMA_FILENAME = "memory_action_window_schema.json"


def memory_action_window_enabled(sim_cfg: Any) -> bool:
    return bool(getattr(sim_cfg, "memory_action_window_instrumentation_enable", False))


def memory_action_window_schema() -> dict[str, Any]:
    return {
        "schema_version": MEMORY_ACTION_WINDOW_SCHEMA_VERSION,
        "read_only": True,
        "default_off": True,
        "checkpoint_mode": "stateless_derived_from_checkpointed_runner_env_population_state",
        "outputs": {
            "events_jsonl": EVENTS_FILENAME,
            "generation_summary_jsonl": GENERATION_SUMMARY_FILENAME,
            "run_summary_json": RUN_SUMMARY_FILENAME,
        },
        "required_fields": [
            "schema_version",
            "run_id",
            "seed",
            "generation",
            "tick",
            "window_id",
            "phase",
            "trait_arm",
            "world_context",
            "init_arm",
            "agent_count",
            "action_histogram",
            "movement_action_count",
            "gather_action_count",
            "reproduce_action_count",
            "idle_noop_count",
            "unknown_other_action_count",
            "lineage_context",
            "memory_summary",
            "exposure_damage_summary",
        ],
        "phase_names": ["cue", "hidden_gap", "active_hazard", "baseline"],
        "action_ids": {
            "move_n": int(ACT_MOVE_N),
            "move_s": int(ACT_MOVE_S),
            "move_e": int(ACT_MOVE_E),
            "move_w": int(ACT_MOVE_W),
            "gather": int(ACT_GATHER),
            "rest": int(ACT_REST),
            "attack": int(ACT_ATTACK),
        },
    }


def _canonical_line(obj: Mapping[str, Any]) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n"


def _mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values]
    return float(sum(vals) / len(vals)) if vals else 0.0


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _phase_at(env: Any, y: int, x: int) -> str:
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


def _phase_from_arrays(
    *,
    active: np.ndarray,
    cue: np.ndarray,
    timer: np.ndarray,
    prior: np.ndarray,
    y: int,
    x: int,
) -> str:
    if y < 0 or x < 0:
        return "baseline"
    if y >= active.shape[0] or x >= active.shape[1]:
        return "baseline"
    if float(active[y, x]) > 0.0:
        return "active_hazard"
    if float(cue[y, x]) > 0.0:
        return "cue"
    if int(timer[y, x]) >= 0 and int(prior[y, x]) > 0:
        return "hidden_gap"
    return "baseline"


class MemoryActionWindowInstrumentation:
    def __init__(self, runner: Any) -> None:
        self.run_dir = Path(str(getattr(runner, "run_dir", "")))
        self.out_dir = self.run_dir / "memory_action_window"
        self.emit_jsonl = bool(getattr(runner.sim_cfg, "memory_action_window_emit_jsonl", True))
        self.include_lineage = bool(getattr(runner.sim_cfg, "memory_action_window_include_lineage", True))
        self.include_memory_state = bool(getattr(runner.sim_cfg, "memory_action_window_include_memory_state", True))
        self.max_events = int(getattr(runner.sim_cfg, "memory_action_window_max_events_per_run", 0) or 0)
        self.sample_tick_period = max(1, _safe_int(getattr(runner.sim_cfg, "memory_action_window_sample_tick_period", 1), 1))
        self.sample_generation_period = max(
            1,
            _safe_int(getattr(runner.sim_cfg, "memory_action_window_sample_generation_period", 1), 1),
        )
        self.sample_first_n_generations = max(
            0,
            _safe_int(getattr(runner.sim_cfg, "memory_action_window_sample_first_n_generations", 0), 0),
        )
        self.schema_version = str(
            getattr(runner.sim_cfg, "memory_action_window_schema_version", MEMORY_ACTION_WINDOW_SCHEMA_VERSION)
            or MEMORY_ACTION_WINDOW_SCHEMA_VERSION
        )
        self._pending: dict[tuple[int, int], dict[str, Any]] = {}
        self._event_count = 0
        self._run_phase_counts: Counter[str] = Counter()
        self._run_action_hist: Counter[str] = Counter()
        self._run_agent_count = 0
        self._run_damage_events = 0
        self._run_rows = 0
        self.out_dir.mkdir(parents=True, exist_ok=True)
        (self.out_dir / SCHEMA_FILENAME).write_text(
            json.dumps(memory_action_window_schema(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def maybe_create(cls, runner: Any) -> "MemoryActionWindowInstrumentation | None":
        if not memory_action_window_enabled(getattr(runner, "sim_cfg", None)):
            return None
        return cls(runner)

    def _should_emit(self) -> bool:
        return bool(self.emit_jsonl and (self.max_events <= 0 or self._event_count < self.max_events))

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

    def record_pre_env(self, runner: Any, *, generation: int, tick: int, alive: list[Any], agent_ids_np: Any, actions_np: Any) -> None:
        if not self._should_emit():
            return
        if not self._sampling_allows(int(generation), int(tick)):
            return

        env = runner.env
        agent_ids = np.asarray(agent_ids_np, dtype=np.int32)
        actions = np.asarray(actions_np, dtype=np.int32)
        pos_y = np.asarray(getattr(env, "agent_pos_y"), dtype=np.int32)
        pos_x = np.asarray(getattr(env, "agent_pos_x"), dtype=np.int32)
        active = np.asarray(getattr(env, "patch_delayed_hazard_active"), dtype=np.float32)
        cue = np.asarray(getattr(env, "patch_delayed_hazard_cue"), dtype=np.float32)
        timer = np.asarray(getattr(env, "patch_delayed_hazard_timer"), dtype=np.int32)
        prior = np.asarray(getattr(env, "patch_delayed_hazard_prior_cue"), dtype=np.int8)
        hidden_gap_cell_count = float(np.sum((timer >= 0) & (prior > 0) & (cue <= 0.0) & (active <= 0.0)))
        hazard_context = {
            "cue_cell_count": float(getattr(runner.env, "wdl_delayed_hazard_cue_cell_count_last", 0.0) or 0.0),
            "active_hazard_cell_count": float(
                getattr(runner.env, "wdl_delayed_hazard_active_cell_count_last", 0.0) or 0.0
            ),
            "hidden_gap_cell_count": hidden_gap_cell_count,
            "active_hazard_without_prior_cue_count": int(
                getattr(runner.env, "wdl_delayed_hazard_active_without_prior_cue_count_last", 0) or 0
            ),
        }
        phase_rows: dict[str, dict[str, Any]] = {}
        action_by_id = {int(agent_ids[i]): int(actions[i]) for i in range(min(agent_ids.size, actions.size))}
        alive_by_id = {int(getattr(agent, "agent_id")): agent for agent in alive}
        hmem_np = None
        if self.include_memory_state:
            try:
                hmem = getattr(runner, "_Hmem", None)
                if hmem is not None:
                    with torch.no_grad():
                        hmem_np = hmem.detach().to("cpu").numpy()
            except Exception:
                hmem_np = None

        for aid in sorted(action_by_id):
            if aid < 0 or aid >= pos_y.shape[0]:
                continue
            y = int(pos_y[aid])
            x = int(pos_x[aid])
            phase = _phase_from_arrays(active=active, cue=cue, timer=timer, prior=prior, y=y, x=x)
            row = phase_rows.setdefault(
                phase,
                self._empty_row(
                    runner,
                    generation=generation,
                    tick=tick,
                    phase=phase,
                    hazard_context=hazard_context,
                ),
            )
            action = int(action_by_id[aid])
            agent = alive_by_id.get(aid)
            row["_agent_ids"].append(aid)
            row["_pre_health"][aid] = float(getattr(agent, "health", 0.0)) if agent is not None else 0.0
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
            self._add_lineage(row, agent)
            self._add_memory(row, runner, agent, hmem_np=hmem_np)

        self._pending[(int(generation), int(tick))] = {"rows": phase_rows}

    def record_post_env(self, runner: Any, *, generation: int, tick: int) -> None:
        payload = self._pending.pop((int(generation), int(tick)), None)
        if not payload:
            return
        env = runner.env
        pos_y = np.asarray(getattr(env, "agent_pos_y"), dtype=np.int32)
        pos_x = np.asarray(getattr(env, "agent_pos_x"), dtype=np.int32)
        agent_alive = np.asarray(getattr(env, "agent_alive"), dtype=bool)
        active = np.asarray(getattr(env, "patch_delayed_hazard_active"), dtype=np.float32)
        cue = np.asarray(getattr(env, "patch_delayed_hazard_cue"), dtype=np.float32)
        timer = np.asarray(getattr(env, "patch_delayed_hazard_timer"), dtype=np.int32)
        prior = np.asarray(getattr(env, "patch_delayed_hazard_prior_cue"), dtype=np.int8)
        rows = payload.get("rows", {})
        for phase in sorted(rows):
            row = rows[phase]
            for aid in list(row.pop("_agent_ids", [])):
                pre_health = float(row["_pre_health"].get(aid, 0.0))
                agent = getattr(runner.pop, "agents", {}).get(int(aid))
                post_health = float(getattr(agent, "health", pre_health)) if agent is not None else pre_health
                if post_health < pre_health:
                    row["exposure_damage_summary"]["damage_event_count"] += 1
                    row["exposure_damage_summary"]["damage_total"] += float(pre_health - post_health)
                    if phase in ("cue", "hidden_gap", "active_hazard"):
                        row["exposure_damage_summary"]["damage_with_prior_cue_context_count"] += 1
                if phase == "active_hazard":
                    row["exposure_damage_summary"]["active_hazard_exposure_count"] += 1
                if aid >= 0 and aid < pos_y.shape[0] and aid < agent_alive.shape[0] and bool(agent_alive[aid]):
                    post_phase = _phase_from_arrays(
                        active=active,
                        cue=cue,
                        timer=timer,
                        prior=prior,
                        y=int(pos_y[aid]),
                        x=int(pos_x[aid]),
                    )
                    if phase != "active_hazard" and post_phase == "active_hazard":
                        row["exposure_damage_summary"]["hazard_entry_count"] += 1
                    if phase == "active_hazard" and post_phase != "active_hazard":
                        row["exposure_damage_summary"]["hazard_exit_count"] += 1
                    if phase in ("cue", "hidden_gap") and post_phase == "baseline":
                        row["exposure_damage_summary"]["pre_hazard_exit_count"] += 1
            row.pop("_pre_health", None)
            self._finalize_row(row)
            self._write_event(row)

    def rows_from_summary_tape(self, runner: Any, tape: Any) -> list[dict[str, Any]]:
        """Build diagnostic rows from a CXXCORE summary tape.

        Default-off CXXCORE-18 adapter. It does not mutate simulation state and
        is intentionally separate from legacy `record_pre_env`/`record_post_env`.
        """
        from perf.cxxcore18_live_summary_tape_metrics import memory_action_rows_from_tape  # noqa: WPS433

        return memory_action_rows_from_tape(self, runner, tape)

    def record_summary_tape(self, runner: Any, tape: Any) -> list[dict[str, Any]]:
        rows = self.rows_from_summary_tape(runner, tape)
        for row in rows:
            self._write_event(row)
        return rows

    def flush_generation_summary(self, runner: Any, *, generation: int) -> None:
        if not self.emit_jsonl:
            return
        summary = {
            "schema_version": self.schema_version,
            "run_id": self._run_id(runner),
            "seed": _safe_int(getattr(runner.sim_cfg, "seed", 0)),
            "generation": int(generation),
            "rows_emitted_total": int(self._run_rows),
            "agent_count_total": int(self._run_agent_count),
            "damage_event_count_total": int(self._run_damage_events),
            "phase_row_counts": dict(sorted(self._run_phase_counts.items())),
            "action_histogram_total": dict(sorted(self._run_action_hist.items())),
            "diagnostic_sampling": self._sampling_summary(),
        }
        with (self.out_dir / GENERATION_SUMMARY_FILENAME).open("a", encoding="utf-8") as f:
            f.write(_canonical_line(summary))
        self._write_run_summary(runner)

    def _empty_row(
        self,
        runner: Any,
        *,
        generation: int,
        tick: int,
        phase: str,
        hazard_context: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        context = hazard_context or {}
        cue_cells = float(
            context.get("cue_cell_count", getattr(runner.env, "wdl_delayed_hazard_cue_cell_count_last", 0.0))
            or 0.0
        )
        active_cells = float(
            context.get(
                "active_hazard_cell_count",
                getattr(runner.env, "wdl_delayed_hazard_active_cell_count_last", 0.0),
            )
            or 0.0
        )
        hidden_gap_cells = float(context.get("hidden_gap_cell_count", self._hidden_gap_cell_count(runner.env)) or 0.0)
        active_without_prior_cue = int(
            context.get(
                "active_hazard_without_prior_cue_count",
                getattr(runner.env, "wdl_delayed_hazard_active_without_prior_cue_count_last", 0),
            )
            or 0
        )
        return {
            "schema_version": self.schema_version,
            "run_id": self._run_id(runner),
            "seed": _safe_int(getattr(runner.sim_cfg, "seed", 0)),
            "generation": int(generation),
            "tick": int(tick),
            "window_id": f"g{int(generation):06d}:t{int(tick):06d}:{phase}",
            "phase": str(phase),
            "trait_arm": str(getattr(runner.sim_cfg, "memory_action_window_trait_arm", "")),
            "world_context": str(getattr(runner.sim_cfg, "memory_action_window_world_context", "")),
            "init_arm": str(getattr(runner.sim_cfg, "memory_action_window_init_arm", "")),
            "agent_count": 0,
            "action_histogram": {str(i): 0 for i in range(N_ACTIONS)},
            "movement_action_count": 0,
            "gather_action_count": 0,
            "reproduce_action_count": 0,
            "idle_noop_count": 0,
            "unknown_other_action_count": 0,
            "lineage_context": {
                "lineage_count": 0,
                "effective_alive_lineages": 0,
                "top_lineage_id": "",
                "top_lineage_share": 0.0,
                "founder_ids": [],
            },
            "memory_summary": {
                "effective_memory_mean": 0.0,
                "memory_state_abs_mean": 0.0,
                "memory_state_delta_mean": 0.0,
                "memory_state_delta_available": False,
                "memory_agent_count": 0,
                "memory_ablation_active": bool(memory_ablation_flags(getattr(runner, "sim_cfg", None)).get("active", False)),
            },
            "exposure_damage_summary": {
                "cue_cell_count": cue_cells,
                "hidden_gap_cell_count": hidden_gap_cells,
                "active_hazard_cell_count": active_cells,
                "active_hazard_exposure_count": 0,
                "pre_hazard_exit_count": 0,
                "hazard_entry_count": 0,
                "hazard_exit_count": 0,
                "damage_event_count": 0,
                "damage_total": 0.0,
                "damage_with_prior_cue_context_count": 0,
                "active_hazard_without_prior_cue_count": active_without_prior_cue,
            },
            "_agent_ids": [],
            "_pre_health": {},
            "_lineage_counts": Counter(),
            "_founder_ids": set(),
            "_effective_memory_values": [],
            "_memory_abs_values": [],
        }

    def _add_lineage(self, row: dict[str, Any], agent: Any | None) -> None:
        if not self.include_lineage or agent is None:
            return
        lineage = str(getattr(agent, "lineage_id", "") or getattr(agent, "founder_id", "") or "")
        if lineage:
            row["_lineage_counts"][lineage] += 1
        founder = str(getattr(agent, "founder_id", "") or "")
        if founder:
            row["_founder_ids"].add(founder)

    def _add_memory(self, row: dict[str, Any], runner: Any, agent: Any | None, *, hmem_np: Any = None) -> None:
        if not self.include_memory_state or agent is None:
            return
        try:
            eff = int(effective_memory_dim_for_genome(getattr(agent, "genome", None), sim_cfg=runner.sim_cfg))
        except Exception:
            eff = 0
        row["_effective_memory_values"].append(float(eff))
        if eff <= 0:
            row["_memory_abs_values"].append(0.0)
            return
        try:
            slot = int(getattr(agent, "slot", -1))
            if hmem_np is None or slot < 0:
                row["_memory_abs_values"].append(0.0)
                return
            state = hmem_np[int(slot)]
            row["_memory_abs_values"].append(float(np.mean(np.abs(state[:eff])) if eff <= state.shape[0] else np.mean(np.abs(state))))
        except Exception:
            row["_memory_abs_values"].append(0.0)

    def _finalize_row(self, row: dict[str, Any]) -> None:
        lineage_counts = row.pop("_lineage_counts", Counter())
        founder_ids = sorted(row.pop("_founder_ids", set()))
        eff_values = row.pop("_effective_memory_values", [])
        mem_abs_values = row.pop("_memory_abs_values", [])
        total_lineage = int(sum(lineage_counts.values()))
        top_id = ""
        top_count = 0
        if lineage_counts:
            top_id, top_count = sorted(lineage_counts.items(), key=lambda kv: (-kv[1], kv[0]))[0]
        row["lineage_context"] = {
            "lineage_count": int(len(lineage_counts)),
            "effective_alive_lineages": int(len([v for v in lineage_counts.values() if int(v) > 0])),
            "top_lineage_id": str(top_id),
            "top_lineage_share": float(top_count / total_lineage) if total_lineage > 0 else 0.0,
            "founder_ids": founder_ids,
        }
        row["memory_summary"]["effective_memory_mean"] = _mean(eff_values)
        row["memory_summary"]["memory_state_abs_mean"] = _mean(mem_abs_values)
        row["memory_summary"]["memory_agent_count"] = int(len(mem_abs_values))
        row["exposure_damage_summary"]["damage_total"] = float(row["exposure_damage_summary"]["damage_total"])

    def _write_event(self, row: Mapping[str, Any]) -> None:
        if not self._should_emit():
            return
        with (self.out_dir / EVENTS_FILENAME).open("a", encoding="utf-8") as f:
            f.write(_canonical_line(row))
        self._event_count += 1
        self._run_rows += 1
        self._run_agent_count += int(row.get("agent_count", 0) or 0)
        self._run_damage_events += int(row.get("exposure_damage_summary", {}).get("damage_event_count", 0) or 0)
        self._run_phase_counts[str(row.get("phase", ""))] += 1
        for action, count in dict(row.get("action_histogram", {})).items():
            self._run_action_hist[str(action)] += int(count)

    def _write_run_summary(self, runner: Any) -> None:
        summary = {
            "schema_version": self.schema_version,
            "run_id": self._run_id(runner),
            "seed": _safe_int(getattr(runner.sim_cfg, "seed", 0)),
            "rows_emitted_total": int(self._run_rows),
            "agent_count_total": int(self._run_agent_count),
            "damage_event_count_total": int(self._run_damage_events),
            "phase_row_counts": dict(sorted(self._run_phase_counts.items())),
            "action_histogram_total": dict(sorted(self._run_action_hist.items())),
            "read_only": True,
            "default_off": True,
            "checkpoint_mode": "stateless_derived_from_checkpointed_runner_env_population_state",
            "diagnostic_sampling": self._sampling_summary(),
        }
        (self.out_dir / RUN_SUMMARY_FILENAME).write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def _run_id(self, runner: Any) -> str:
        explicit = str(getattr(runner.sim_cfg, "memory_action_window_run_id", "") or "")
        if explicit:
            return explicit
        return Path(str(getattr(runner, "run_dir", ""))).name

    def _hidden_gap_cell_count(self, env: Any) -> float:
        try:
            cue = np.asarray(getattr(env, "patch_delayed_hazard_cue"), dtype=np.float32)
            active = np.asarray(getattr(env, "patch_delayed_hazard_active"), dtype=np.float32)
            timer = np.asarray(getattr(env, "patch_delayed_hazard_timer"), dtype=np.int32)
            prior = np.asarray(getattr(env, "patch_delayed_hazard_prior_cue"), dtype=np.int8)
            return float(np.sum((timer >= 0) & (prior > 0) & (cue <= 0.0) & (active <= 0.0)))
        except Exception:
            return 0.0
