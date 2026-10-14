from __future__ import annotations

"""Default-off live MEM-MAX Level-4 target-alignment telemetry."""

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from agents.cognition import effective_memory_dim_for_genome, memory_ablation_flags
from env.gridworld_pkg.constants import N_ACTIONS
from env.gridworld_pkg.patch_world import mem_max_delayed_choice_enabled
from evolution.policy_lineage_telemetry import policy_hash as lineage_policy_hash
from evolution.target_alignment import action_target_delta


SCHEMA_VERSION = "mem_max_live_level4_v2"
EVENTS_FILENAME = "live_level4_event_sample.jsonl"
GENERATION_SUMMARY_FILENAME = "live_level4_generation_summary.jsonl"
RUN_SUMMARY_FILENAME = "live_level4_run_summary.json"
SCHEMA_FILENAME = "live_level4_schema.json"


def mem_max_live_level4_enabled(sim_cfg: Any) -> bool:
    return bool(getattr(sim_cfg, "mem_max_live_level4_instrumentation_enable", False))


def mem_max_live_level4_schema() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
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
            "step",
            "event_id",
            "agent_id",
            "founder_id",
            "lineage_id",
            "policy_hash",
            "cue_identity",
            "actual_runtime_cue_exposure",
            "hidden_window_phase",
            "consequence_active_phase",
            "agent_row_before",
            "agent_col_before",
            "target_a_row",
            "target_a_col",
            "target_b_row",
            "target_b_col",
            "grid_height",
            "grid_width",
            "distance_to_a_before",
            "distance_to_b_before",
            "policy_type",
            "effective_memory_dim",
            "memory_ablated",
            "cue_off",
            "hidden_state_abs_mean",
            "action_logits",
            "action_probabilities",
            "top_action",
            "second_action",
            "top_action_margin",
            "target_relevant_action_margin",
            "chosen_action",
            "executed_action",
            "action_target_class",
            "expected_delta_distance_a",
            "expected_delta_distance_b",
            "probability_mass_toward_a",
            "probability_mass_toward_b",
            "signed_cue_consistent_alignment_score",
            "level2_effective_retention",
            "level3_target_relevant_competition",
            "level4_position_aware_target_aligned",
            "agent_row_after",
            "agent_col_after",
            "actual_delta_distance_a",
            "actual_delta_distance_b",
            "target_entry",
            "choice_classification",
            "applied_consequence",
            "action_execution_success",
            "action_execution_failure_reason",
        ],
        "action_ids": {"north": 0, "south": 1, "east": 2, "west": 3, "gather": 4, "rest": 5, "attack": 6},
        "sample_rule": "deterministic first eligible events, or deterministic predeclared stratified live decision sampling when opted in",
        "level4_threshold": 0.02,
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


def _coords(mask: np.ndarray) -> tuple[int, int] | None:
    found = np.argwhere(np.asarray(mask) > 0)
    if found.size <= 0:
        return None
    y, x = [int(v) for v in found[0].tolist()]
    return y, x


def _policy_hash(agent: Any) -> str:
    for name in ("policy_hash", "policy_sha256", "policy_digest"):
        value = getattr(agent, name, None)
        if value:
            return str(value)
    try:
        policy = getattr(agent, "policy", None)
        digest = lineage_policy_hash(policy)
        if digest:
            return digest
    except Exception:
        pass
    try:
        policy = getattr(agent, "policy", None)
        h = hashlib.sha256()
        for param in policy.parameters():
            h.update(param.detach().cpu().numpy().tobytes())
        return h.hexdigest()
    except Exception:
        return ""


def _phase(cue_a: np.ndarray, cue_b: np.ndarray, active: np.ndarray, prior: np.ndarray) -> str:
    if bool(np.any(active > 0.0)):
        return "active"
    if bool(np.any((cue_a + cue_b) > 0.0)):
        return "cue"
    if bool(np.any(prior > 0)):
        return "hidden"
    return "baseline"


class MemMaxLiveLevel4Instrumentation:
    def __init__(self, runner: Any) -> None:
        self.run_dir = Path(str(getattr(runner, "run_dir", "")))
        self.out_dir = self.run_dir / "mem_max_live_level4"
        self.emit_jsonl = bool(getattr(runner.sim_cfg, "mem_max_live_level4_emit_jsonl", True))
        self.max_events = max(0, _safe_int(getattr(runner.sim_cfg, "mem_max_live_level4_max_events_per_run", 128), 128))
        self.sampling_mode = str(getattr(runner.sim_cfg, "mem_max_live_level4_sampling_mode", "first_eligible") or "first_eligible")
        self.per_stratum_limit = max(
            1,
            _safe_int(getattr(runner.sim_cfg, "mem_max_live_level4_per_stratum_limit", 4), 4),
        )
        self.sample_first_n_per_generation = max(
            0,
            _safe_int(getattr(runner.sim_cfg, "mem_max_live_level4_sample_first_n_per_generation", 64), 64),
        )
        self.sample_tick_period = max(1, _safe_int(getattr(runner.sim_cfg, "mem_max_live_level4_sample_tick_period", 1), 1))
        self.schema_version = str(getattr(runner.sim_cfg, "mem_max_live_level4_schema_version", SCHEMA_VERSION) or SCHEMA_VERSION)
        self._pending: dict[tuple[int, int], list[dict[str, Any]]] = {}
        self._run_rows = 0
        self._per_gen_sampled: Counter[int] = Counter()
        self._level_counts: Counter[str] = Counter()
        self._action_class_counts: Counter[str] = Counter()
        self._choice_counts: Counter[str] = Counter()
        self._score_sum = 0.0
        self._margin_sum = 0.0
        self._full_rows = 0
        self._full_level_counts: Counter[str] = Counter()
        self._full_action_class_counts: Counter[str] = Counter()
        self._full_action_family_counts: Counter[str] = Counter()
        self._full_choice_counts: Counter[str] = Counter()
        self._full_target_entries = 0
        self._full_score_sum = 0.0
        self._full_margin_sum = 0.0
        self._stratum_available_counts: Counter[str] = Counter()
        self._stratum_sampled_counts: Counter[str] = Counter()
        self.out_dir.mkdir(parents=True, exist_ok=True)
        (self.out_dir / SCHEMA_FILENAME).write_text(
            json.dumps(mem_max_live_level4_schema(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def maybe_create(cls, runner: Any) -> "MemMaxLiveLevel4Instrumentation | None":
        if not mem_max_live_level4_enabled(getattr(runner, "sim_cfg", None)):
            return None
        return cls(runner)

    def record_pre_env(self, runner: Any, *, generation: int, tick: int, alive: list[Any], agent_ids_np: Any, actions_np: Any) -> None:
        if self.sampling_mode != "stratified" and self.max_events > 0 and self._run_rows >= self.max_events:
            return
        if int(tick) % int(self.sample_tick_period) != 0:
            return
        if not mem_max_delayed_choice_enabled(getattr(runner, "env_cfg", None)):
            return
        env = runner.env
        target_a_arr = np.asarray(getattr(env, "mem_max_delayed_choice_target_a", np.zeros((1, 1))), dtype=np.float32)
        target_b_arr = np.asarray(getattr(env, "mem_max_delayed_choice_target_b", np.zeros_like(target_a_arr)), dtype=np.float32)
        target_a = _coords(target_a_arr)
        target_b = _coords(target_b_arr)
        if target_a is None or target_b is None:
            return
        cue_a = np.asarray(getattr(env, "mem_max_delayed_choice_cue_a", np.zeros_like(target_a_arr)), dtype=np.float32)
        cue_b = np.asarray(getattr(env, "mem_max_delayed_choice_cue_b", np.zeros_like(target_a_arr)), dtype=np.float32)
        active = np.asarray(getattr(env, "mem_max_delayed_choice_active", np.zeros_like(target_a_arr)), dtype=np.float32)
        prior = np.asarray(getattr(env, "mem_max_delayed_choice_prior_cue", np.zeros_like(target_a_arr)), dtype=np.int8)
        correct = np.asarray(getattr(env, "mem_max_delayed_choice_correct_target", np.zeros_like(target_a_arr)), dtype=np.int8)
        event_ids = np.asarray(getattr(env, "mem_max_delayed_choice_event_id", np.zeros_like(target_a_arr)), dtype=np.int64)
        event_mask = (target_a_arr + target_b_arr + cue_a + cue_b + active + (prior > 0).astype(np.float32)) > 0
        event_id = int(np.max(event_ids[event_mask])) if bool(np.any(event_mask & (event_ids > 0))) else 0
        cue_identity = 0
        if bool(np.any(cue_a > 0.0)):
            cue_identity = 1
        elif bool(np.any(cue_b > 0.0)):
            cue_identity = 2
        elif bool(np.any(prior > 0)):
            cue_identity = int(np.max(prior[prior > 0]))
        elif bool(np.any(correct > 0)):
            cue_identity = int(np.max(correct[correct > 0]))
        phase = _phase(cue_a, cue_b, active, prior)
        if phase == "baseline":
            return

        policy_cache = getattr(runner, "_mem_max_live_level4_last_policy", {}) or {}
        logits_by_id = {int(k): np.asarray(v, dtype=np.float32) for k, v in dict(policy_cache.get("logits_by_agent_id", {})).items()}
        probs_by_id = {int(k): np.asarray(v, dtype=np.float32) for k, v in dict(policy_cache.get("probs_by_agent_id", {})).items()}
        hmem = getattr(runner, "_Hmem", None)
        hmem_np = None
        if hmem is not None:
            try:
                hmem_np = hmem.detach().to("cpu").numpy()
            except Exception:
                hmem_np = None
        pos_y = np.asarray(getattr(env, "agent_pos_y"), dtype=np.int32)
        pos_x = np.asarray(getattr(env, "agent_pos_x"), dtype=np.int32)
        actions = {int(aid): int(act) for aid, act in zip(np.asarray(agent_ids_np, dtype=np.int32).tolist(), np.asarray(actions_np, dtype=np.int32).tolist())}
        alive_by_id = {int(getattr(agent, "agent_id")): agent for agent in alive}
        rows: list[dict[str, Any]] = []
        for aid in sorted(actions):
            if self.sampling_mode != "stratified" and self.max_events > 0 and self._run_rows + len(rows) >= self.max_events:
                break
            if (
                self.sampling_mode != "stratified"
                and self.sample_first_n_per_generation > 0
                and self._per_gen_sampled[int(generation)] + len(rows) >= self.sample_first_n_per_generation
            ):
                break
            if aid < 0 or aid >= pos_y.shape[0]:
                continue
            y = int(pos_y[aid])
            x = int(pos_x[aid])
            if y < 0 or x < 0:
                continue
            logits = logits_by_id.get(aid)
            probs = probs_by_id.get(aid)
            if logits is None or logits.size < N_ACTIONS:
                logits = np.zeros((N_ACTIONS,), dtype=np.float32)
            if probs is None or probs.size < N_ACTIONS:
                exp = np.exp(logits - np.max(logits))
                probs = exp / max(float(np.sum(exp)), 1e-12)
            agent = alive_by_id.get(aid)
            action = int(actions[aid])
            meta = action_target_delta(
                action,
                agent_pos=(y, x),
                target_a=target_a,
                target_b=target_b,
                height=int(target_a_arr.shape[0]),
                width=int(target_a_arr.shape[1]),
            )
            action_family = "movement" if int(action) in (0, 1, 2, 3) else "neutral"
            mass_a = float(sum(float(probs[i]) for i in range(min(N_ACTIONS, probs.size)) if action_target_delta(i, agent_pos=(y, x), target_a=target_a, target_b=target_b, height=int(target_a_arr.shape[0]), width=int(target_a_arr.shape[1]))["target_class"] in ("toward_a", "toward_both")))
            mass_b = float(sum(float(probs[i]) for i in range(min(N_ACTIONS, probs.size)) if action_target_delta(i, agent_pos=(y, x), target_a=target_a, target_b=target_b, height=int(target_a_arr.shape[0]), width=int(target_a_arr.shape[1]))["target_class"] in ("toward_b", "toward_both")))
            signed = (mass_a - mass_b) if cue_identity == 1 else ((mass_b - mass_a) if cue_identity == 2 else 0.0)
            top_order = np.argsort(-logits)
            top = int(top_order[0]) if top_order.size else 0
            second = int(top_order[1]) if top_order.size > 1 else top
            top_margin = float(logits[top] - logits[second]) if top_order.size > 1 else 0.0
            target_margin = abs(mass_a - mass_b)
            ablation_active = bool(memory_ablation_flags(getattr(runner, "sim_cfg", None)).get("active", False))
            cue_off = not bool(getattr(runner.env_cfg, "mem_max_delayed_choice_cue_enable", True))
            eff_mem = 0
            hidden_abs = 0.0
            if agent is not None:
                eff_mem = int(effective_memory_dim_for_genome(getattr(agent, "genome", None), sim_cfg=runner.sim_cfg))
                slot = int(getattr(agent, "slot", -1))
                if hmem_np is not None and slot >= 0 and slot < hmem_np.shape[0] and eff_mem > 0:
                    hidden_abs = float(np.mean(np.abs(hmem_np[slot, : min(eff_mem, hmem_np.shape[1])])))
            level2 = bool(cue_identity in (1, 2) and phase in ("hidden", "active") and eff_mem > 0 and not ablation_active and not cue_off)
            level3 = bool(level2 and target_margin > 0.02)
            level4 = bool(level3 and signed > 0.02 and meta["target_class"] in (f"toward_{'a' if cue_identity == 1 else 'b'}", "toward_both"))
            rows.append(
                {
                    "schema_version": self.schema_version,
                    "run_id": self._run_id(runner),
                    "seed": _safe_int(getattr(runner.sim_cfg, "seed", 0)),
                    "generation": int(generation),
                    "step": int(tick),
                    "event_id": int(event_id),
                    "decision_event_id": self._decision_event_id(runner, generation=generation, tick=tick, event_id=event_id, agent_id=aid),
                    "agent_id": int(aid),
                    "founder_id": str(getattr(agent, "founder_id", "") if agent is not None else ""),
                    "lineage_id": str(getattr(agent, "lineage_id", "") if agent is not None else ""),
                    "policy_hash": _policy_hash(agent) if agent is not None else "",
                    "cue_identity": int(cue_identity),
                    "actual_runtime_cue_exposure": bool(phase == "cue"),
                    "hidden_window_phase": bool(phase == "hidden"),
                    "consequence_active_phase": bool(phase == "active"),
                    "event_phase": str(phase),
                    "post_hidden_decision_opportunity": bool(phase in ("hidden", "active")),
                    "agent_row_before": int(y),
                    "agent_col_before": int(x),
                    "target_a_row": int(target_a[0]),
                    "target_a_col": int(target_a[1]),
                    "target_b_row": int(target_b[0]),
                    "target_b_col": int(target_b[1]),
                    "grid_height": int(target_a_arr.shape[0]),
                    "grid_width": int(target_a_arr.shape[1]),
                    "distance_to_a_before": abs(y - int(target_a[0])) + abs(x - int(target_a[1])),
                    "distance_to_b_before": abs(y - int(target_b[0])) + abs(x - int(target_b[1])),
                    "policy_type": type(getattr(agent, "policy", None)).__name__ if agent is not None else "",
                    "effective_memory_dim": int(eff_mem),
                    "memory_ablated": bool(ablation_active),
                    "cue_off": bool(cue_off),
                    "hidden_state_abs_mean": float(hidden_abs),
                    "action_logits": [round(float(v), 6) for v in logits[:N_ACTIONS].tolist()],
                    "action_probabilities": [round(float(v), 6) for v in probs[:N_ACTIONS].tolist()],
                    "top_action": int(top),
                    "second_action": int(second),
                    "top_action_margin": float(top_margin),
                    "target_relevant_action_margin": float(target_margin),
                    "chosen_action": int(action),
                    "executed_action": int(action),
                    "action_family": action_family,
                    "movement_opportunity": bool(phase in ("hidden", "active")),
                    "action_target_class": str(meta["target_class"]),
                    "expected_delta_distance_a": int(meta["delta_distance_a"]),
                    "expected_delta_distance_b": int(meta["delta_distance_b"]),
                    "probability_mass_toward_a": float(mass_a),
                    "probability_mass_toward_b": float(mass_b),
                    "signed_cue_consistent_alignment_score": float(signed),
                    "level2_effective_retention": bool(level2),
                    "level3_target_relevant_competition": bool(level3),
                    "level4_position_aware_target_aligned": bool(level4),
                    "_target_a": target_a,
                    "_target_b": target_b,
                    "_pre_health": _safe_float(getattr(agent, "health", 0.0) if agent is not None else 0.0),
                }
            )
        if rows:
            self._pending[(int(generation), int(tick))] = rows

    def record_post_env(self, runner: Any, *, generation: int, tick: int) -> None:
        rows = self._pending.pop((int(generation), int(tick)), [])
        if not rows:
            return
        env = runner.env
        pos_y = np.asarray(getattr(env, "agent_pos_y"), dtype=np.int32)
        pos_x = np.asarray(getattr(env, "agent_pos_x"), dtype=np.int32)
        for row in rows:
            aid = int(row["agent_id"])
            pre_a = int(row["distance_to_a_before"])
            pre_b = int(row["distance_to_b_before"])
            target_a = tuple(row.pop("_target_a"))
            target_b = tuple(row.pop("_target_b"))
            pre_health = float(row.pop("_pre_health"))
            if aid >= 0 and aid < pos_y.shape[0]:
                y2 = int(pos_y[aid])
                x2 = int(pos_x[aid])
            else:
                y2 = int(row["agent_row_before"])
                x2 = int(row["agent_col_before"])
            post_a = abs(y2 - int(target_a[0])) + abs(x2 - int(target_a[1]))
            post_b = abs(y2 - int(target_b[0])) + abs(x2 - int(target_b[1]))
            at_a = y2 == int(target_a[0]) and x2 == int(target_a[1])
            at_b = y2 == int(target_b[0]) and x2 == int(target_b[1])
            cue_identity = int(row["cue_identity"])
            if (cue_identity == 1 and at_a) or (cue_identity == 2 and at_b):
                choice = "correct"
            elif at_a or at_b:
                choice = "incorrect"
            else:
                choice = "no_choice"
            agent = getattr(runner.pop, "agents", {}).get(aid)
            post_health = _safe_float(getattr(agent, "health", pre_health) if agent is not None else pre_health)
            row["agent_row_after"] = int(y2)
            row["agent_col_after"] = int(x2)
            row["actual_delta_distance_a"] = int(pre_a - post_a)
            row["actual_delta_distance_b"] = int(pre_b - post_b)
            row["target_entry"] = bool(at_a or at_b)
            row["choice_classification"] = choice
            row["applied_consequence"] = float(max(0.0, pre_health - post_health))
            row["action_execution_success"] = bool(
                row["actual_delta_distance_a"] == int(row["expected_delta_distance_a"])
                and row["actual_delta_distance_b"] == int(row["expected_delta_distance_b"])
            )
            row["action_execution_failure_reason"] = "" if row["action_execution_success"] else "movement_blocked_or_environment_interference"
            self._record_full_stream(row)
            if self._should_sample(row):
                self._write_event(row)

    def flush_generation_summary(self, runner: Any, *, generation: int) -> None:
        if not self.emit_jsonl:
            return
        summary = self._summary(runner)
        summary["generation"] = int(generation)
        with (self.out_dir / GENERATION_SUMMARY_FILENAME).open("a", encoding="utf-8") as f:
            f.write(_canonical_line(summary))
        self._write_run_summary(runner)

    def _write_event(self, row: Mapping[str, Any]) -> None:
        if not self.emit_jsonl:
            return
        with (self.out_dir / EVENTS_FILENAME).open("a", encoding="utf-8") as f:
            f.write(_canonical_line(row))
        self._run_rows += 1
        self._per_gen_sampled[int(row.get("generation", 0))] += 1
        for key, flag in (
            ("level2", row.get("level2_effective_retention")),
            ("level3", row.get("level3_target_relevant_competition")),
            ("level4", row.get("level4_position_aware_target_aligned")),
        ):
            if bool(flag):
                self._level_counts[key] += 1
        self._action_class_counts[str(row.get("action_target_class", ""))] += 1
        self._choice_counts[str(row.get("choice_classification", ""))] += 1
        self._score_sum += float(row.get("signed_cue_consistent_alignment_score", 0.0) or 0.0)
        self._margin_sum += float(row.get("target_relevant_action_margin", 0.0) or 0.0)
        self._stratum_sampled_counts[self._stratum_key(row)] += 1

    def _record_full_stream(self, row: Mapping[str, Any]) -> None:
        self._full_rows += 1
        for key, flag in (
            ("level2", row.get("level2_effective_retention")),
            ("level3", row.get("level3_target_relevant_competition")),
            ("level4", row.get("level4_position_aware_target_aligned")),
        ):
            if bool(flag):
                self._full_level_counts[key] += 1
        self._full_action_class_counts[str(row.get("action_target_class", ""))] += 1
        self._full_action_family_counts[str(row.get("action_family", ""))] += 1
        self._full_choice_counts[str(row.get("choice_classification", ""))] += 1
        self._full_target_entries += int(bool(row.get("target_entry", False)))
        self._full_score_sum += float(row.get("signed_cue_consistent_alignment_score", 0.0) or 0.0)
        self._full_margin_sum += float(row.get("target_relevant_action_margin", 0.0) or 0.0)
        self._stratum_available_counts[self._stratum_key(row)] += 1

    def _generation_bucket(self, generation: int) -> str:
        gen = int(generation)
        if gen <= 0:
            return "g0"
        if gen == 1:
            return "early"
        if gen <= 3:
            return "middle"
        return "late"

    def _stratum_key(self, row: Mapping[str, Any]) -> str:
        phase = str(row.get("event_phase", "unknown"))
        cue = int(row.get("cue_identity", 0) or 0)
        mem = "mem_present" if int(row.get("effective_memory_dim", 0) or 0) > 0 and not bool(row.get("memory_ablated", False)) else "mem_absent"
        return "|".join(
            (
                f"cue{cue}",
                phase,
                self._generation_bucket(int(row.get("generation", 0) or 0)),
                str(row.get("policy_type", "unknown") or "unknown"),
                mem,
                str(row.get("action_family", "unknown") or "unknown"),
            )
        )

    def _should_sample(self, row: Mapping[str, Any]) -> bool:
        if self.sampling_mode != "stratified":
            return True
        if self.max_events > 0 and self._run_rows >= self.max_events:
            return False
        key = self._stratum_key(row)
        if self._stratum_sampled_counts[key] >= self.per_stratum_limit:
            return False
        return True

    def _summary(self, runner: Any) -> dict[str, Any]:
        denom = max(1, int(self._run_rows))
        full_denom = max(1, int(self._full_rows))
        return {
            "schema_version": self.schema_version,
            "run_id": self._run_id(runner),
            "seed": _safe_int(getattr(runner.sim_cfg, "seed", 0)),
            "sampling_mode": str(self.sampling_mode),
            "full_stream_rows_total": int(self._full_rows),
            "rows_emitted_total": int(self._run_rows),
            "level_counts": dict(sorted(self._level_counts.items())),
            "sampled_level_counts": dict(sorted(self._level_counts.items())),
            "full_level_counts": dict(sorted(self._full_level_counts.items())),
            "action_class_counts": dict(sorted(self._action_class_counts.items())),
            "full_action_class_counts": dict(sorted(self._full_action_class_counts.items())),
            "full_action_family_counts": dict(sorted(self._full_action_family_counts.items())),
            "choice_counts": dict(sorted(self._choice_counts.items())),
            "full_choice_counts": dict(sorted(self._full_choice_counts.items())),
            "full_target_entries": int(self._full_target_entries),
            "signed_score_mean": float(self._score_sum / float(denom)),
            "target_relevant_margin_mean": float(self._margin_sum / float(denom)),
            "full_signed_score_mean": float(self._full_score_sum / float(full_denom)),
            "full_target_relevant_margin_mean": float(self._full_margin_sum / float(full_denom)),
            "stratum_available_counts": dict(sorted(self._stratum_available_counts.items())),
            "stratum_sampled_counts": dict(sorted(self._stratum_sampled_counts.items())),
            "read_only": True,
            "default_off": True,
            "diagnostic_sampling": {
                "max_events_per_run": int(self.max_events),
                "sampling_mode": str(self.sampling_mode),
                "per_stratum_limit": int(self.per_stratum_limit),
                "sample_first_n_per_generation": int(self.sample_first_n_per_generation),
                "sample_tick_period": int(self.sample_tick_period),
                "sample_rule": (
                    "deterministic fixed first N by sorted identity within predeclared outcome-independent strata"
                    if self.sampling_mode == "stratified"
                    else "deterministic first eligible events by generation, tick, sorted agent id"
                ),
            },
        }

    def _write_run_summary(self, runner: Any) -> None:
        (self.out_dir / RUN_SUMMARY_FILENAME).write_text(
            json.dumps(self._summary(runner), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def _run_id(self, runner: Any) -> str:
        explicit = str(getattr(runner.sim_cfg, "mem_max_live_level4_run_id", "") or "")
        if explicit:
            return explicit
        return Path(str(getattr(runner, "run_dir", ""))).name

    def _decision_event_id(self, runner: Any, *, generation: int, tick: int, event_id: int, agent_id: int) -> str:
        payload = {
            "run_id": self._run_id(runner),
            "generation": int(generation),
            "tick": int(tick),
            "event_id": int(event_id),
            "agent_id": int(agent_id),
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()[:24]
