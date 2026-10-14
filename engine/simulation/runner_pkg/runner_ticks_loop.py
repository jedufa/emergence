from __future__ import annotations

"""
simulation/runner_pkg/runner_ticks_loop.py

Per-generation tick loop orchestration for the simulation runner.

This module intentionally owns ONLY orchestration:
- food spawning
- alive list build
- batched policy inference
- calling the tick subsystems in a fixed order
- optional torch.profiler integration

All "mechanics" live in the other runner_ticks_* modules.
"""

from typing import Any, Dict, Tuple
from contextlib import nullcontext
import time
from pathlib import Path
import os
import hashlib
import json

import numpy as np
from engine.simulation.forensic_v4_probes import emit_agent_attr_signature, emit_forward_order_probe
from evolution.policy_family_head_authority_runtime import charge_family_head_residual_cost
from perf.cxxrewrite4_upstream_resident import (
    flush_summary as cxxrewrite4_flush_summary,
    prepare_resident_generation_env_scalar_inputs as cxxrewrite4_prepare_resident_generation_env_scalar_inputs,
    record_action_array as cxxrewrite4_record_action_array,
    resident_scalar_sync_fail_closed as cxxrewrite4_resident_scalar_sync_fail_closed,
    sync_agents_scalar_state as cxxrewrite4_sync_agents_scalar_state,
)
from simulation.rng_manager import DeterminismViolationError
from simulation.tick_native_sampling import (
    maybe_capture_tick_native_science_sample,
    science_sampling_enabled,
)


class TickLoopMixin:
    def _record_reference_policy_cache_rows(self, *, requested: int, hit: int, missed: int) -> None:
        requested_i = int(requested)
        hit_i = int(hit)
        missed_i = int(missed)
        if min(requested_i, hit_i, missed_i) < 0 or requested_i != hit_i + missed_i:
            raise ValueError(
                "reference policy-cache counters require requested=hit+missed: "
                f"requested={requested_i} hit={hit_i} missed={missed_i}"
            )
        counts = getattr(self, "_reference_policy_cache_public_counts", None)
        if not isinstance(counts, dict) or counts.get("source") != "reference_policy_cache_service":
            counts = {
                "source": "reference_policy_cache_service",
                "policy_cache_rows_requested": 0,
                "policy_cache_rows_hit": 0,
                "policy_cache_rows_missed": 0,
                "policy_cache_rows_refilled": 0,
                "policy_cache_full_refill_ticks": 0,
                "policy_cache_partial_refill_ticks": 0,
            }
            self._reference_policy_cache_public_counts = counts
        counts["policy_cache_rows_requested"] += requested_i
        counts["policy_cache_rows_hit"] += hit_i
        counts["policy_cache_rows_missed"] += missed_i
        counts["policy_cache_rows_refilled"] += missed_i
        if missed_i > 0 and hit_i == 0:
            counts["policy_cache_full_refill_ticks"] += 1
        elif missed_i > 0 and hit_i > 0:
            counts["policy_cache_partial_refill_ticks"] += 1

    def _record_wdl_action_context_pre_env(self, alive, agent_ids_np, actions_np) -> None:
        """Record WDL cue/action context without changing actions or rewards."""
        cfg = getattr(self, "env_cfg", None)
        try:
            from env.gridworld_pkg.constants import ACT_MOVE_E, ACT_MOVE_N, ACT_MOVE_S, ACT_MOVE_W
            from env.gridworld_pkg.patch_world import wdl_delayed_hazard_enabled
        except Exception:
            return
        if not wdl_delayed_hazard_enabled(cfg):
            return
        cue = np.asarray(getattr(self.env, "patch_delayed_hazard_cue"), dtype=np.float32)
        active = np.asarray(getattr(self.env, "patch_delayed_hazard_active"), dtype=np.float32)
        pos_y = np.asarray(getattr(self.env, "agent_pos_y"), dtype=np.int32)
        pos_x = np.asarray(getattr(self.env, "agent_pos_x"), dtype=np.int32)
        h, w = tuple(cue.shape)
        if not hasattr(self, "_wdl_agents_cued_current_gen"):
            self._wdl_agents_cued_current_gen = set()
        move_delta = {
            int(ACT_MOVE_N): (-1, 0),
            int(ACT_MOVE_S): (1, 0),
            int(ACT_MOVE_E): (0, 1),
            int(ACT_MOVE_W): (0, -1),
        }
        for aid_raw, act_raw in zip(np.asarray(agent_ids_np, dtype=np.int32).tolist(), np.asarray(actions_np, dtype=np.int32).tolist()):
            aid = int(aid_raw)
            y = int(pos_y[aid])
            x = int(pos_x[aid])
            if y < 0 or x < 0:
                continue
            here_cue = float(cue[y, x])
            here_active = float(active[y, x])
            dy, dx = move_delta.get(int(act_raw), (0, 0))
            yy = max(0, min(int(h) - 1, y + int(dy)))
            xx = max(0, min(int(w) - 1, x + int(dx)))
            next_cue = float(cue[yy, xx])
            next_active = float(active[yy, xx])
            if here_cue > 0.0:
                self._wdl_agents_cued_current_gen.add(aid)
                self._wdl_cue_seen_agents_gen = int(getattr(self, "_wdl_cue_seen_agents_gen", 0)) + 1
                if next_cue < here_cue:
                    self._wdl_cue_exposed_next_move_away_count_gen = int(
                        getattr(self, "_wdl_cue_exposed_next_move_away_count_gen", 0)
                    ) + 1
                elif next_cue > here_cue:
                    self._wdl_cue_exposed_next_move_toward_count_gen = int(
                        getattr(self, "_wdl_cue_exposed_next_move_toward_count_gen", 0)
                    ) + 1
                else:
                    self._wdl_cue_exposed_stayed_count_gen = int(
                        getattr(self, "_wdl_cue_exposed_stayed_count_gen", 0)
                    ) + 1
            if here_cue > 0.0 and next_cue <= 0.0:
                self._wdl_pre_hazard_exit_count_gen = int(getattr(self, "_wdl_pre_hazard_exit_count_gen", 0)) + 1
            if here_active <= 0.0 and next_active > 0.0:
                self._wdl_hazard_enter_count_gen = int(getattr(self, "_wdl_hazard_enter_count_gen", 0)) + 1

    def _record_delayed_resource_waste_action_context_pre_env(self, alive, agent_ids_np, actions_np) -> None:
        """Record non-hazard delayed-consequence cue/action context passively."""
        cfg = getattr(self, "env_cfg", None)
        try:
            from env.gridworld_pkg.constants import ACT_GATHER, ACT_MOVE_E, ACT_MOVE_N, ACT_MOVE_S, ACT_MOVE_W
            from env.gridworld_pkg.patch_world import (
                delayed_crowding_waste_enabled,
                delayed_resource_depletion_enabled,
            )
        except Exception:
            return
        families = (
            ("delayed_resource_depletion", delayed_resource_depletion_enabled(cfg)),
            ("delayed_crowding_waste", delayed_crowding_waste_enabled(cfg)),
        )
        active_families = [(prefix, enabled) for prefix, enabled in families if enabled]
        if not active_families:
            return
        pos_y = np.asarray(getattr(self.env, "agent_pos_y"), dtype=np.int32)
        pos_x = np.asarray(getattr(self.env, "agent_pos_x"), dtype=np.int32)
        move_delta = {
            int(ACT_MOVE_N): (-1, 0),
            int(ACT_MOVE_S): (1, 0),
            int(ACT_MOVE_E): (0, 1),
            int(ACT_MOVE_W): (0, -1),
        }
        for prefix, _enabled in active_families:
            cue = np.asarray(getattr(self.env, f"{prefix}_cue"), dtype=np.float32)
            active = np.asarray(getattr(self.env, f"{prefix}_active"), dtype=np.float32)
            timer = np.asarray(getattr(self.env, f"{prefix}_timer"), dtype=np.int32)
            prior_cue = np.asarray(getattr(self.env, f"{prefix}_prior_cue"), dtype=np.int8)
            h, w = tuple(cue.shape)
            seen_attr = f"_{prefix}_agents_cued_current_gen"
            if not hasattr(self, seen_attr):
                setattr(self, seen_attr, set())
            seen = getattr(self, seen_attr)
            for aid_raw, act_raw in zip(
                np.asarray(agent_ids_np, dtype=np.int32).tolist(),
                np.asarray(actions_np, dtype=np.int32).tolist(),
            ):
                aid = int(aid_raw)
                y = int(pos_y[aid])
                x = int(pos_x[aid])
                if y < 0 or x < 0:
                    continue
                phase = "baseline"
                if float(active[y, x]) > 0.0:
                    phase = "active"
                elif float(cue[y, x]) > 0.0:
                    phase = "cue"
                elif int(prior_cue[y, x]) > 0 and int(timer[y, x]) >= 0:
                    phase = "hidden"

                if phase == "cue":
                    seen.add(aid)
                    setattr(self, f"_{prefix}_agent_cue_exposure_count_gen", int(getattr(self, f"_{prefix}_agent_cue_exposure_count_gen", 0)) + 1)
                elif phase == "hidden":
                    setattr(self, f"_{prefix}_agent_hidden_gap_exposure_count_gen", int(getattr(self, f"_{prefix}_agent_hidden_gap_exposure_count_gen", 0)) + 1)
                elif phase == "active":
                    setattr(self, f"_{prefix}_agent_active_exposure_count_gen", int(getattr(self, f"_{prefix}_agent_active_exposure_count_gen", 0)) + 1)
                    if aid in seen:
                        setattr(self, f"_{prefix}_revisit_after_cue_count_gen", int(getattr(self, f"_{prefix}_revisit_after_cue_count_gen", 0)) + 1)

                if int(act_raw) == int(ACT_GATHER) and phase in ("cue", "hidden", "active"):
                    setattr(self, f"_{prefix}_gather_{phase}_count_gen", int(getattr(self, f"_{prefix}_gather_{phase}_count_gen", 0)) + 1)

                if phase not in ("cue", "hidden", "active"):
                    continue
                dy, dx = move_delta.get(int(act_raw), (0, 0))
                yy = max(0, min(int(h) - 1, y + int(dy)))
                xx = max(0, min(int(w) - 1, x + int(dx)))
                here = float(cue[y, x] + active[y, x])
                next_value = float(cue[yy, xx] + active[yy, xx])
                if next_value < here:
                    setattr(self, f"_{prefix}_move_away_count_gen", int(getattr(self, f"_{prefix}_move_away_count_gen", 0)) + 1)
                elif next_value > here:
                    setattr(self, f"_{prefix}_move_toward_count_gen", int(getattr(self, f"_{prefix}_move_toward_count_gen", 0)) + 1)
                else:
                    setattr(self, f"_{prefix}_stay_count_gen", int(getattr(self, f"_{prefix}_stay_count_gen", 0)) + 1)

    def _apply_wdl_delayed_hazard_damage(self, deaths_this_gen: int) -> int:
        """Apply WDL delayed hazard health damage from active cells."""
        cfg = getattr(self, "env_cfg", None)
        try:
            from env.gridworld_pkg.patch_world import sample_patch_context, wdl_delayed_hazard_enabled
            from simulation.metrics_pkg.path1_drh0 import path1_drh0_damage_adjustment
        except Exception:
            return int(deaths_this_gen)
        if not wdl_delayed_hazard_enabled(cfg):
            return int(deaths_this_gen)
        if not bool(getattr(cfg, "wdl_delayed_hazard_damage_enable", True)):
            return int(deaths_this_gen)

        active = np.asarray(getattr(self.env, "patch_delayed_hazard_active"), dtype=np.float32)
        intensity = np.asarray(getattr(self.env, "patch_delayed_hazard_intensity"), dtype=np.float32)
        if active.size <= 0 or not bool(np.any(active > 0.0)):
            return int(deaths_this_gen)

        env_alive = np.asarray(getattr(self.env, "agent_alive"), dtype=bool)
        alive_ids = np.nonzero(env_alive)[0].astype(np.int32)
        pos_y = np.asarray(getattr(self.env, "agent_pos_y"), dtype=np.int32)
        pos_x = np.asarray(getattr(self.env, "agent_pos_x"), dtype=np.int32)
        damaged_ids: set[int] = set()
        mutated_agents = []
        contexts = getattr(self, "_wdl_delayed_hazard_event_contexts_gen", None)
        if contexts is None:
            contexts = []
            self._wdl_delayed_hazard_event_contexts_gen = contexts
        for name, default in (
            ("_path1_drh0_damage_in_refuge_gen", 0.0),
            ("_path1_drh0_damage_outside_refuge_gen", 0.0),
            ("_path1_drh0_damage_prevented_by_refuge_gen", 0.0),
            ("_path1_drh0_static_refuge_damage_prevented_gen", 0.0),
            ("_path1_drh0_cue_conditioned_damage_prevented_gen", 0.0),
            ("_path1_drh0_protected_exposure_count_gen", 0),
            ("_path1_drh0_refuge_exposure_count_gen", 0),
            ("_path1_drh0_no_cue_false_positive_refuge_movement_gen", 0),
        ):
            if not hasattr(self, name):
                setattr(self, name, default)

        for aid_raw in alive_ids.tolist():
            aid = int(aid_raw)
            y = int(pos_y[aid])
            x = int(pos_x[aid])
            if y < 0 or x < 0 or float(active[y, x]) <= 0.0:
                continue
            agent = getattr(self.pop, "agents", {}).get(aid)
            if agent is None or not bool(getattr(agent, "alive", False)):
                continue
            raw_damage = max(0.0, float(intensity[y, x]))
            if raw_damage <= 0.0:
                continue
            adjustment = path1_drh0_damage_adjustment(
                self.env,
                agent_id=aid,
                y=y,
                x=x,
                raw_damage=raw_damage,
                cued_agent_ids=getattr(self, "_wdl_agents_cued_current_gen", set()),
            )
            damage = float(adjustment.effective_damage)
            self._wdl_delayed_hazard_agent_active_exposure_count_gen = int(
                getattr(self, "_wdl_delayed_hazard_agent_active_exposure_count_gen", 0)
            ) + 1
            if aid in getattr(self, "_wdl_agents_cued_current_gen", set()):
                self._wdl_hazard_exposure_after_cue_count_gen = int(
                    getattr(self, "_wdl_hazard_exposure_after_cue_count_gen", 0)
                ) + 1
            self._wdl_delayed_hazard_damage_total_gen = float(
                getattr(self, "_wdl_delayed_hazard_damage_total_gen", 0.0)
            ) + float(damage)
            self._wdl_delayed_hazard_damage_max_gen = max(
                float(getattr(self, "_wdl_delayed_hazard_damage_max_gen", 0.0)),
                float(damage),
            )
            if adjustment.in_refuge:
                self._path1_drh0_damage_in_refuge_gen = float(
                    getattr(self, "_path1_drh0_damage_in_refuge_gen", 0.0)
                ) + float(damage)
                self._path1_drh0_refuge_exposure_count_gen = int(
                    getattr(self, "_path1_drh0_refuge_exposure_count_gen", 0)
                ) + 1
            else:
                self._path1_drh0_damage_outside_refuge_gen = float(
                    getattr(self, "_path1_drh0_damage_outside_refuge_gen", 0.0)
                ) + float(damage)
            self._path1_drh0_damage_prevented_by_refuge_gen = float(
                getattr(self, "_path1_drh0_damage_prevented_by_refuge_gen", 0.0)
            ) + float(adjustment.prevented_damage)
            if adjustment.static_refuge_control:
                self._path1_drh0_static_refuge_damage_prevented_gen = float(
                    getattr(self, "_path1_drh0_static_refuge_damage_prevented_gen", 0.0)
                ) + float(adjustment.prevented_damage)
            if adjustment.cue_conditioned:
                self._path1_drh0_cue_conditioned_damage_prevented_gen = float(
                    getattr(self, "_path1_drh0_cue_conditioned_damage_prevented_gen", 0.0)
                ) + float(adjustment.prevented_damage)
            if adjustment.protected:
                self._path1_drh0_protected_exposure_count_gen = int(
                    getattr(self, "_path1_drh0_protected_exposure_count_gen", 0)
                ) + 1
            if adjustment.protected and not adjustment.static_refuge_control and not adjustment.cue_observable:
                self._path1_drh0_no_cue_false_positive_refuge_movement_gen = int(
                    getattr(self, "_path1_drh0_no_cue_false_positive_refuge_movement_gen", 0)
                ) + 1
            damaged_ids.add(aid)
            try:
                ctx = sample_patch_context(self.env, y, x, agent=agent)
                ctx["wdl_damage"] = float(damage)
                ctx["path1_drh0_raw_damage"] = float(raw_damage)
                ctx["path1_drh0_prevented_damage"] = float(adjustment.prevented_damage)
                ctx["path1_drh0_in_refuge"] = int(bool(adjustment.in_refuge))
                ctx["path1_drh0_protected"] = int(bool(adjustment.protected))
                contexts.append(ctx)
            except Exception:
                pass
            h0 = float(getattr(agent, "health", 0.0))
            h1 = max(0.0, h0 - float(damage))
            agent.health = h1
            mutated_agents.append(agent)
            try:
                st = getattr(agent, "stats", None)
                if st is not None:
                    st.last_health_delta = float(getattr(st, "last_health_delta", 0.0)) - float(damage)
            except Exception:
                pass
            if h1 <= 0.0 and bool(getattr(agent, "alive", False)):
                try:
                    if not hasattr(self, "_patch_death_contexts_gen"):
                        self._patch_death_contexts_gen = []
                    self._patch_death_contexts_gen.append(sample_patch_context(self.env, y, x, agent=agent))
                except Exception:
                    pass
                agent.alive = False
                deaths_this_gen = int(deaths_this_gen) + 1
                self.env.remove_agent(aid)
                mutated_agents.append(agent)
        self._wdl_delayed_hazard_agents_damaged_gen = len(
            set(getattr(self, "_wdl_delayed_hazard_agents_damaged_ids_gen", set())) | damaged_ids
        )
        self._wdl_delayed_hazard_agents_damaged_ids_gen = set(
            getattr(self, "_wdl_delayed_hazard_agents_damaged_ids_gen", set())
        ) | damaged_ids
        try:
            cxxrewrite4_sync_agents_scalar_state(self, mutated_agents, stage="wdl_delayed_hazard")
        except Exception:
            if cxxrewrite4_resident_scalar_sync_fail_closed(self.sim_cfg):
                raise
        return int(deaths_this_gen)

    def _apply_mem_max_delayed_choice_consequence(self, deaths_this_gen: int) -> int:
        """Apply MEM-MAX delayed-choice health/energy consequence from active target windows."""
        cfg = getattr(self, "env_cfg", None)
        try:
            from env.gridworld_pkg.patch_world import mem_max_delayed_choice_enabled, sample_patch_context
        except Exception:
            return int(deaths_this_gen)
        if not mem_max_delayed_choice_enabled(cfg):
            return int(deaths_this_gen)

        env = getattr(self, "env", None)
        for metric_name in (
            "event_count",
            "cue_step_active",
            "hidden_step_active",
            "consequence_step_active",
            "target_a_cue_count",
            "target_b_cue_count",
        ):
            gen_name = f"_mem_max_delayed_choice_{metric_name}_gen"
            last_name = f"mem_max_delayed_choice_{metric_name}_last"
            setattr(self, gen_name, int(getattr(self, gen_name, 0) or 0) + int(getattr(env, last_name, 0) or 0))

        if not bool(getattr(cfg, "mem_max_delayed_choice_consequence_enable", True)):
            return int(deaths_this_gen)

        active = np.asarray(getattr(env, "mem_max_delayed_choice_active", np.zeros((1, 1))), dtype=np.float32)
        if active.size <= 0 or not bool(np.any(active > 0.0)):
            return int(deaths_this_gen)

        target_a = np.asarray(getattr(env, "mem_max_delayed_choice_target_a", np.zeros_like(active)), dtype=np.float32)
        target_b = np.asarray(getattr(env, "mem_max_delayed_choice_target_b", np.zeros_like(active)), dtype=np.float32)
        correct_map = np.asarray(getattr(env, "mem_max_delayed_choice_correct_target", np.zeros_like(active)), dtype=np.int8)
        event_ids = np.asarray(getattr(env, "mem_max_delayed_choice_event_id", np.zeros_like(active)), dtype=np.int64)
        active_mask = active > 0.0
        event_id = int(np.max(event_ids[active_mask])) if bool(np.any(event_ids[active_mask] > 0)) else 0
        correct_value = int(np.max(correct_map[active_mask])) if bool(np.any(correct_map[active_mask] > 0)) else 0
        target_a_coords = np.argwhere(target_a > 0.0)
        target_b_coords = np.argwhere(target_b > 0.0)
        if target_a_coords.size <= 0 or target_b_coords.size <= 0 or correct_value not in (1, 2):
            return int(deaths_this_gen)
        ay, ax = [int(v) for v in target_a_coords[0].tolist()]
        by, bx = [int(v) for v in target_b_coords[0].tolist()]

        env_alive = np.asarray(getattr(env, "agent_alive"), dtype=bool)
        alive_ids = np.nonzero(env_alive)[0].astype(np.int32)
        if alive_ids.size <= 0:
            return int(deaths_this_gen)
        pos_y = np.asarray(getattr(env, "agent_pos_y"), dtype=np.int32)
        pos_x = np.asarray(getattr(env, "agent_pos_x"), dtype=np.int32)

        damage_base = max(
            0.0,
            float(
                getattr(
                    cfg,
                    "mem_max_delayed_choice_health_damage",
                    getattr(cfg, "mem_max_delayed_choice_damage", 0.25),
                )
            ),
        )
        energy_base = max(0.0, float(getattr(cfg, "mem_max_delayed_choice_energy_cost", 0.0)))
        multipliers = {
            "correct": max(0.0, float(getattr(cfg, "mem_max_delayed_choice_correct_choice_damage_multiplier", 0.0))),
            "incorrect": max(0.0, float(getattr(cfg, "mem_max_delayed_choice_incorrect_choice_damage_multiplier", 1.0))),
            "no_choice": max(0.0, float(getattr(cfg, "mem_max_delayed_choice_no_choice_damage_multiplier", 1.0))),
        }
        once_per_event = bool(getattr(cfg, "mem_max_delayed_choice_consequence_once_per_event", True))
        applied_pairs = getattr(self, "_mem_max_delayed_choice_applied_event_agent_pairs_gen", None)
        if applied_pairs is None:
            applied_pairs = set()
            self._mem_max_delayed_choice_applied_event_agent_pairs_gen = applied_pairs
        move_budget = int(getattr(cfg, "mem_max_delayed_choice_cue_steps", 1) or 1) + int(
            getattr(cfg, "mem_max_delayed_choice_hidden_steps", 1) or 1
        )
        mutated_agents = []

        tick_expected = 0.0
        tick_health_damage = 0.0
        tick_energy_cost = 0.0
        tick_applied = 0
        tick_eligible = 0
        tick_duplicate = 0
        tick_correct_damage = 0.0
        tick_incorrect_damage = 0.0
        tick_no_choice_damage = 0.0
        tick_target_a_arrivals = 0
        tick_target_b_arrivals = 0
        tick_reachable = 0
        tick_reach_denom = 0

        for aid_raw in alive_ids.tolist():
            aid = int(aid_raw)
            y = int(pos_y[aid])
            x = int(pos_x[aid])
            if y < 0 or x < 0:
                continue
            agent = getattr(self.pop, "agents", {}).get(aid)
            if agent is None or not bool(getattr(agent, "alive", False)):
                continue

            tick_eligible += 1
            tick_reach_denom += 1
            if min(abs(y - ay) + abs(x - ax), abs(y - by) + abs(x - bx)) <= move_budget:
                tick_reachable += 1
            at_a = y == ay and x == ax
            at_b = y == by and x == bx
            if at_a:
                tick_target_a_arrivals += 1
            if at_b:
                tick_target_b_arrivals += 1
            if (correct_value == 1 and at_a) or (correct_value == 2 and at_b):
                choice = "correct"
            elif at_a or at_b:
                choice = "incorrect"
            else:
                choice = "no_choice"

            pair = (event_id, aid)
            if once_per_event and pair in applied_pairs:
                tick_duplicate += 1
                continue
            applied_pairs.add(pair)

            if choice == "correct":
                self._mem_max_delayed_choice_correct_arrival_count_gen = int(
                    getattr(self, "_mem_max_delayed_choice_correct_arrival_count_gen", 0)
                ) + 1
            elif choice == "incorrect":
                self._mem_max_delayed_choice_incorrect_arrival_count_gen = int(
                    getattr(self, "_mem_max_delayed_choice_incorrect_arrival_count_gen", 0)
                ) + 1
            else:
                self._mem_max_delayed_choice_no_choice_count_gen = int(
                    getattr(self, "_mem_max_delayed_choice_no_choice_count_gen", 0)
                ) + 1

            multiplier = float(multipliers[choice])
            health_damage = max(0.0, damage_base * multiplier)
            energy_cost = max(0.0, energy_base * multiplier)
            expected = float(health_damage + energy_cost)
            prevented = max(0.0, float(damage_base + energy_base) - expected)
            tick_expected += expected
            tick_health_damage += health_damage
            tick_energy_cost += energy_cost
            tick_applied += 1
            if choice == "correct":
                tick_correct_damage += health_damage
            elif choice == "incorrect":
                tick_incorrect_damage += health_damage
            else:
                tick_no_choice_damage += health_damage

            try:
                contexts = getattr(self, "_mem_max_delayed_choice_event_contexts_gen", None)
                if contexts is None:
                    contexts = []
                    self._mem_max_delayed_choice_event_contexts_gen = contexts
                ctx = sample_patch_context(env, y, x, agent=agent)
                ctx["mem_max_delayed_choice_event_id"] = int(event_id)
                ctx["mem_max_delayed_choice_choice"] = choice
                ctx["mem_max_delayed_choice_health_damage"] = float(health_damage)
                ctx["mem_max_delayed_choice_energy_cost"] = float(energy_cost)
                contexts.append(ctx)
            except Exception:
                pass

            h0 = float(getattr(agent, "health", 0.0))
            e0 = float(getattr(agent, "energy", 0.0))
            h1 = max(0.0, h0 - float(health_damage))
            e1 = max(0.0, e0 - float(energy_cost))
            agent.health = h1
            agent.energy = e1
            mutated_agents.append(agent)
            try:
                st = getattr(agent, "stats", None)
                if st is not None:
                    st.last_health_delta = float(getattr(st, "last_health_delta", 0.0)) - float(health_damage)
                    st.last_energy_delta = float(getattr(st, "last_energy_delta", 0.0)) - float(energy_cost)
                    st.mem_max_delayed_choice_last_applied_event_id = int(event_id)
            except Exception:
                pass

            self._mem_max_delayed_choice_damage_total_gen = float(
                getattr(self, "_mem_max_delayed_choice_damage_total_gen", 0.0)
            ) + float(health_damage)
            self._mem_max_delayed_choice_damage_prevented_total_gen = float(
                getattr(self, "_mem_max_delayed_choice_damage_prevented_total_gen", 0.0)
            ) + float(prevented)

            if h1 <= 0.0 and bool(getattr(agent, "alive", False)):
                try:
                    if not hasattr(self, "_patch_death_contexts_gen"):
                        self._patch_death_contexts_gen = []
                    self._patch_death_contexts_gen.append(sample_patch_context(env, y, x, agent=agent))
                except Exception:
                    pass
                agent.alive = False
                deaths_this_gen = int(deaths_this_gen) + 1
                env.remove_agent(aid)
                mutated_agents.append(agent)

        self._mem_max_delayed_choice_consequence_exposure_count_gen = int(
            getattr(self, "_mem_max_delayed_choice_consequence_exposure_count_gen", 0)
        ) + int(tick_applied)
        self._mem_max_delayed_choice_expected_consequence_total_gen = float(
            getattr(self, "_mem_max_delayed_choice_expected_consequence_total_gen", 0.0)
        ) + float(tick_expected)
        self._mem_max_delayed_choice_applied_health_damage_total_gen = float(
            getattr(self, "_mem_max_delayed_choice_applied_health_damage_total_gen", 0.0)
        ) + float(tick_health_damage)
        self._mem_max_delayed_choice_applied_energy_cost_total_gen = float(
            getattr(self, "_mem_max_delayed_choice_applied_energy_cost_total_gen", 0.0)
        ) + float(tick_energy_cost)
        self._mem_max_delayed_choice_consequence_applied_count_gen = int(
            getattr(self, "_mem_max_delayed_choice_consequence_applied_count_gen", 0)
        ) + int(tick_applied)
        self._mem_max_delayed_choice_consequence_eligible_count_gen = int(
            getattr(self, "_mem_max_delayed_choice_consequence_eligible_count_gen", 0)
        ) + int(tick_eligible)
        self._mem_max_delayed_choice_duplicate_prevented_count_gen = int(
            getattr(self, "_mem_max_delayed_choice_duplicate_prevented_count_gen", 0)
        ) + int(tick_duplicate)
        self._mem_max_delayed_choice_correct_health_damage_total_gen = float(
            getattr(self, "_mem_max_delayed_choice_correct_health_damage_total_gen", 0.0)
        ) + float(tick_correct_damage)
        self._mem_max_delayed_choice_incorrect_health_damage_total_gen = float(
            getattr(self, "_mem_max_delayed_choice_incorrect_health_damage_total_gen", 0.0)
        ) + float(tick_incorrect_damage)
        self._mem_max_delayed_choice_no_choice_health_damage_total_gen = float(
            getattr(self, "_mem_max_delayed_choice_no_choice_health_damage_total_gen", 0.0)
        ) + float(tick_no_choice_damage)
        self._mem_max_delayed_choice_target_a_arrival_count_gen = int(
            getattr(self, "_mem_max_delayed_choice_target_a_arrival_count_gen", 0)
        ) + int(tick_target_a_arrivals)
        self._mem_max_delayed_choice_target_b_arrival_count_gen = int(
            getattr(self, "_mem_max_delayed_choice_target_b_arrival_count_gen", 0)
        ) + int(tick_target_b_arrivals)
        self._mem_max_delayed_choice_reachable_agents_gen = int(
            getattr(self, "_mem_max_delayed_choice_reachable_agents_gen", 0)
        ) + int(tick_reachable)
        self._mem_max_delayed_choice_reachability_denominator_gen = int(
            getattr(self, "_mem_max_delayed_choice_reachability_denominator_gen", 0)
        ) + int(tick_reach_denom)

        env.mem_max_delayed_choice_expected_consequence_total_last = float(tick_expected)
        env.mem_max_delayed_choice_applied_health_damage_total_last = float(tick_health_damage)
        env.mem_max_delayed_choice_applied_energy_cost_total_last = float(tick_energy_cost)
        env.mem_max_delayed_choice_consequence_applied_count_last = int(tick_applied)
        env.mem_max_delayed_choice_consequence_eligible_count_last = int(tick_eligible)
        env.mem_max_delayed_choice_duplicate_prevented_count_last = int(tick_duplicate)
        env.mem_max_delayed_choice_correct_health_damage_total_last = float(tick_correct_damage)
        env.mem_max_delayed_choice_incorrect_health_damage_total_last = float(tick_incorrect_damage)
        env.mem_max_delayed_choice_no_choice_health_damage_total_last = float(tick_no_choice_damage)
        env.mem_max_delayed_choice_target_a_arrival_count_last = int(tick_target_a_arrivals)
        env.mem_max_delayed_choice_target_b_arrival_count_last = int(tick_target_b_arrivals)
        env.mem_max_delayed_choice_any_target_reachable_fraction_last = (
            float(tick_reachable) / float(tick_reach_denom) if tick_reach_denom > 0 else 0.0
        )

        if mutated_agents:
            try:
                cxxrewrite4_sync_agents_scalar_state(self, mutated_agents, stage="mem_max_delayed_choice")
            except Exception:
                if cxxrewrite4_resident_scalar_sync_fail_closed(self.sim_cfg):
                    raise
        return int(deaths_this_gen)

    def _apply_jungle_predator_pressure(self, deaths_this_gen: int) -> int:
        """Apply Jungle V0-A predator-like pressure after Patch World updates."""
        cfg = getattr(self, "env_cfg", None)
        try:
            from env.gridworld_pkg.jungle import (
                compute_predator_risk_maps,
                jungle_juvenile_damage_multiplier,
                jungle_predator_pressure_enabled,
                jungle_terrain_damage_multiplier,
            )
        except Exception:
            return int(deaths_this_gen)
        if not jungle_predator_pressure_enabled(cfg):
            return int(deaths_this_gen)

        env_alive = np.asarray(getattr(self.env, "agent_alive"), dtype=bool)
        alive_ids = np.nonzero(env_alive)[0].astype(np.int32)
        if alive_ids.size <= 0:
            return int(deaths_this_gen)

        risk = np.asarray(compute_predator_risk_maps(self.env)["risk"], dtype=np.float32)
        pos_y = np.asarray(getattr(self.env, "agent_pos_y"), dtype=np.int32)
        pos_x = np.asarray(getattr(self.env, "agent_pos_x"), dtype=np.int32)
        damage = max(0.0, float(getattr(cfg, "jungle_predator_health_damage", 0.25)))
        mutated_agents = []

        self._jungle_predator_exposure_count_gen = int(
            getattr(self, "_jungle_predator_exposure_count_gen", 0)
        ) + int(alive_ids.size)
        rng = getattr(self.env, "rng", None)
        if rng is None:
            return int(deaths_this_gen)

        for aid_raw in alive_ids.tolist():
            aid = int(aid_raw)
            agent = getattr(self.pop, "agents", {}).get(aid)
            if agent is None or not bool(getattr(agent, "alive", False)):
                continue
            y = int(pos_y[aid])
            x = int(pos_x[aid])
            if y < 0 or x < 0:
                continue
            p = float(risk[y, x])
            if p <= 0.0:
                continue
            if float(rng.random()) >= p:
                continue

            juvenile_mult, is_juvenile = jungle_juvenile_damage_multiplier(cfg, agent)
            terrain_mult, _terrain_cover, _terrain_shelter = jungle_terrain_damage_multiplier(self.env, y, x)
            raw_event_damage = float(damage) * float(juvenile_mult)
            event_damage = float(raw_event_damage) * float(terrain_mult)
            damage_prevented = max(0.0, float(raw_event_damage) - float(event_damage))
            self._jungle_predator_events_gen = int(getattr(self, "_jungle_predator_events_gen", 0)) + 1
            self._jungle_predator_damage_total_gen = float(
                getattr(self, "_jungle_predator_damage_total_gen", 0.0)
            ) + float(event_damage)
            self._jungle_predator_damage_raw_total_gen = float(
                getattr(self, "_jungle_predator_damage_raw_total_gen", 0.0)
            ) + float(raw_event_damage)
            self._jungle_predator_damage_effective_total_gen = float(
                getattr(self, "_jungle_predator_damage_effective_total_gen", 0.0)
            ) + float(event_damage)
            self._jungle_predator_damage_prevented_total_gen = float(
                getattr(self, "_jungle_predator_damage_prevented_total_gen", 0.0)
            ) + float(damage_prevented)
            if not hasattr(self, "_jungle_predator_damage_multipliers_gen"):
                self._jungle_predator_damage_multipliers_gen = []
            self._jungle_predator_damage_multipliers_gen.append(float(terrain_mult))
            if is_juvenile:
                self._jungle_juvenile_predator_events_gen = int(
                    getattr(self, "_jungle_juvenile_predator_events_gen", 0)
                ) + 1
                self._jungle_juvenile_predator_damage_total_gen = float(
                    getattr(self, "_jungle_juvenile_predator_damage_total_gen", 0.0)
                ) + float(event_damage)
            else:
                self._jungle_adult_predator_events_gen = int(
                    getattr(self, "_jungle_adult_predator_events_gen", 0)
                ) + 1
                self._jungle_adult_predator_damage_total_gen = float(
                    getattr(self, "_jungle_adult_predator_damage_total_gen", 0.0)
                ) + float(event_damage)
            try:
                from env.gridworld_pkg.patch_world import sample_patch_context  # noqa: WPS433

                if not hasattr(self, "_jungle_predator_event_contexts_gen"):
                    self._jungle_predator_event_contexts_gen = []
                ctx = sample_patch_context(self.env, y, x, agent=agent)
                ctx["damage_multiplier"] = float(terrain_mult)
                ctx["damage_prevented"] = float(damage_prevented)
                ctx["raw_damage"] = float(raw_event_damage)
                ctx["effective_damage"] = float(event_damage)
                self._jungle_predator_event_contexts_gen.append(ctx)
            except Exception:
                pass

            h0 = float(getattr(agent, "health", 0.0))
            h1 = max(0.0, h0 - float(event_damage))
            agent.health = h1
            mutated_agents.append(agent)
            try:
                st = getattr(agent, "stats", None)
                if st is not None:
                    st.last_health_delta = float(getattr(st, "last_health_delta", 0.0)) - float(event_damage)
            except Exception:
                pass
            if h1 <= 0.0 and bool(getattr(agent, "alive", False)):
                if bool(getattr(self.env, "patch_world_enable", False)):
                    try:
                        from env.gridworld_pkg.patch_world import sample_patch_context  # noqa: WPS433

                        if not hasattr(self, "_patch_death_contexts_gen"):
                            self._patch_death_contexts_gen = []
                        self._patch_death_contexts_gen.append(
                            sample_patch_context(self.env, y, x, agent=agent)
                        )
                    except Exception:
                        pass
                agent.alive = False
                self._jungle_predator_deaths_gen = int(getattr(self, "_jungle_predator_deaths_gen", 0)) + 1
                if is_juvenile:
                    self._jungle_juvenile_predator_deaths_gen = int(
                        getattr(self, "_jungle_juvenile_predator_deaths_gen", 0)
                    ) + 1
                deaths_this_gen = int(deaths_this_gen) + 1
                self.env.remove_agent(aid)
                mutated_agents.append(agent)
        try:
            cxxrewrite4_sync_agents_scalar_state(self, mutated_agents, stage="jungle_predator_pressure")
        except Exception:
            if cxxrewrite4_resident_scalar_sync_fail_closed(self.sim_cfg):
                raise
        return int(deaths_this_gen)

    def _run_spatial_droughts_step_with_budget(self, *, gen: int, tick: int, guard_call=None) -> None:
        """Run spatial drought update under RNG budget enforcement.

        Strict determinism violations must always fail fast and propagate.
        """
        try:
            abs_step = int(gen) * int(self.sim_cfg.steps_per_generation) + int(tick)
            _budget = nullcontext()
            _rm_budget = getattr(self, "rng_manager", None)
            if _rm_budget is not None:
                _expected = _rm_budget.expected_delta_for_boundary("env.update.spatial_droughts_step")
                _budget = _rm_budget.budget_scope("env.update.spatial_droughts_step", expected_delta=_expected)
            with _budget:
                if guard_call is None:
                    self.env.spatial_droughts_step(abs_step)
                else:
                    guard_call(abs_step)
        except DeterminismViolationError:
            raise
        except Exception:
            pass

    def _maybe_dump_policy_domain_components_pre(self, *, gen: int, tick: int, alive, env_alive) -> None:
        """Optionally write a diagnostic dump for policy-domain membership mismatches.

        This is a STRICT determinism investigation aid.

        Why this exists
        ---------------
        The policy domain is constructed as the intersection of:
          1) Population.alive_agents() (agent.alive)
          2) env.agent_alive[agent_id] (world presence)

        If the first executed generation after resume diverges at `policy_domain_pre`,
        we need to know which *input* to that intersection differs:
          - population alive membership
          - environment alive membership
          - or the intersection logic itself

        The dump is only written when EIEL_STATE_DUMP=1 and gen is in
        EIEL_STATE_DUMP_GENS and tick==0.
        """

        if int(tick) != 0:
            return
        if os.environ.get("EIEL_STATE_DUMP", "0").strip() != "1":
            return
        gens = os.environ.get("EIEL_STATE_DUMP_GENS", "").strip()
        if gens:
            try:
                allowed = {int(x) for x in gens.split(",") if x.strip()}
            except Exception:
                allowed = set()
            if allowed and int(gen) not in allowed:
                return

        try:
            from engine.simulation.state_dump import maybe_dump_state  # noqa: WPS433

            # Policy domain (intersection) as used by the runner.
            domain_ids_seq = [int(getattr(a, "agent_id")) for a in alive]

            # Population-side alive (sorted by population core).
            pop_alive_agents = list(self.pop.alive_agents())
            pop_ids_seq = [int(getattr(a, "agent_id")) for a in pop_alive_agents]

            # Environment-side alive as a deterministic scan over the array.
            try:
                env_ids_seq = [int(x) for x in np.where(np.asarray(env_alive, dtype=bool))[0].tolist()]
            except Exception:
                env_ids_seq = []

            pop_set = set(pop_ids_seq)
            env_set = set(env_ids_seq)
            dom_set = set(domain_ids_seq)

            # Differences that should never exist if env/pop are consistent.
            pop_not_env = sorted(list(pop_set - env_set))
            env_not_pop = sorted(list(env_set - pop_set))
            dom_not_pop = sorted(list(dom_set - pop_set))
            dom_not_env = sorted(list(dom_set - env_set))

            def _h_int_list(xs) -> str:
                b = (",".join(str(int(x)) for x in xs)).encode("utf-8")
                return hashlib.sha256(b).hexdigest()

            extra = {
                "alive_n": int(len(alive)),
                "domain_agent_ids_seq_sha256": _h_int_list(domain_ids_seq),
                "domain_agent_ids_sorted_sha256": _h_int_list(sorted(domain_ids_seq)),
                "pop_alive_ids_seq_sha256": _h_int_list(pop_ids_seq),
                "pop_alive_ids_sorted_sha256": _h_int_list(sorted(pop_ids_seq)),
                "env_alive_ids_sorted_sha256": _h_int_list(sorted(env_ids_seq)),
                "pop_not_env_n": int(len(pop_not_env)),
                "env_not_pop_n": int(len(env_not_pop)),
                "dom_not_pop_n": int(len(dom_not_pop)),
                "dom_not_env_n": int(len(dom_not_env)),
                # Deterministic previews.
                "domain_ids_seq_head": domain_ids_seq[:64],
                "pop_ids_seq_head": pop_ids_seq[:64],
                "env_ids_head": sorted(env_ids_seq)[:64],
                "pop_not_env_head": pop_not_env[:64],
                "env_not_pop_head": env_not_pop[:64],
                "dom_not_pop_head": dom_not_pop[:64],
                "dom_not_env_head": dom_not_env[:64],
            }

            maybe_dump_state(self, generation_index=int(gen), tag="policy_domain_components_pre", extra=extra)
        except Exception:
            return

    def _maybe_dump_policy_domain_pre(self, *, gen: int, tick: int, alive) -> None:
        """Optionally write a low-noise digest of the policy input domain.

        Purpose
        -------
        Phase 3 resume-determinism localization: distinguish whether divergence begins
        upstream of policy inference (alive-domain membership / ordering) or inside the
        policy cache/inference path.

        This dump is intentionally *minimal* and is only produced when:
        - EIEL_STATE_DUMP=1
        - generation is listed in EIEL_STATE_DUMP_GENS
        - tick == 0 (first tick of the generation)

        The output is a standard Phase-3 state dump with tag "policy_domain_pre" and
        includes a small `extra` payload with deterministic hashes.
        """

        # Only on the first tick; avoid spam.
        if int(tick) != 0:
            return

        # Hard-gate on the existing Phase-3 dump controls.
        if os.environ.get("EIEL_STATE_DUMP", "0").strip() != "1":
            return

        gens = os.environ.get("EIEL_STATE_DUMP_GENS", "").strip()
        if gens:
            try:
                allowed = {int(x) for x in gens.split(",") if x.strip()}
            except Exception:
                allowed = set()
            if allowed and int(gen) not in allowed:
                return

        try:
            # Capture both *sequence* (order-sensitive) and *set* (order-insensitive)
            # hashes for agent_id and slot.
            agent_ids_seq = [int(getattr(a, "agent_id")) for a in alive]
            slots_seq = [int(getattr(a, "slot", -1)) for a in alive]

            def _h_int_list(xs) -> str:
                b = (
                    ",".join(str(int(x)) for x in xs)
                    .encode("utf-8")
                )
                return hashlib.sha256(b).hexdigest()

            extra = {
                "alive_n": int(len(alive)),
                "agent_ids_seq_sha256": _h_int_list(agent_ids_seq),
                "agent_ids_sorted_sha256": _h_int_list(sorted(agent_ids_seq)),
                "slots_seq_sha256": _h_int_list(slots_seq),
                "slots_sorted_sha256": _h_int_list(sorted(slots_seq)),
                # Small previews for quick eyeballing; deterministic due to seq order.
                "agent_ids_seq_head": agent_ids_seq[:64],
                "slots_seq_head": slots_seq[:64],
            }

            from engine.simulation.state_dump import maybe_dump_state  # noqa: WPS433

            maybe_dump_state(self, generation_index=int(gen), tag="policy_domain_pre", extra=extra)
        except Exception:
            # Never interfere with the simulation.
            return

    def _maybe_dump_tick0_actions_post(self, *, gen: int, tick: int, alive, agent_ids_np, actions_np) -> None:
        """Optionally dump a deterministic digest of tick-0 actions.

        Phase-3 localization probe: determines whether divergence begins during
        action selection (policy/caching) vs during environment apply.

        Gated by EIEL_STATE_DUMP=1 and optional EIEL_STATE_DUMP_GENS, tick==0.
        """

        if int(tick) != 0:
            return
        if os.environ.get("EIEL_STATE_DUMP", "0").strip() != "1":
            return
        gens = os.environ.get("EIEL_STATE_DUMP_GENS", "").strip()
        if gens:
            try:
                allowed = {int(x) for x in gens.split(",") if x.strip()}
            except Exception:
                allowed = set()
            if allowed and int(gen) not in allowed:
                return

        try:
            from engine.simulation.state_dump import maybe_dump_state  # noqa: WPS433

            agent_ids = [int(x) for x in np.asarray(agent_ids_np, dtype=np.int64).tolist()]
            actions = [int(x) for x in np.asarray(actions_np, dtype=np.int64).tolist()]

            pairs = list(zip(agent_ids, actions))
            pairs_bytes = (";".join(f"{aid}:{act}" for aid, act in pairs)).encode("utf-8")
            pairs_sha = hashlib.sha256(pairs_bytes).hexdigest()

            hist = {}
            for act in actions:
                hist[int(act)] = int(hist.get(int(act), 0) + 1)
            hist_items = sorted(hist.items(), key=lambda kv: kv[0])

            extra = {
                "alive_n": int(len(alive)),
                "agent_action_pairs_sha256": pairs_sha,
                "agent_ids_seq_head": agent_ids[:64],
                "actions_seq_head": actions[:64],
                "action_hist": hist_items,
            }
            maybe_dump_state(self, generation_index=int(gen), tag="tick0_actions_post", extra=extra)
        except Exception:
            return

    def _maybe_dump_tick0_env_post(self, *, gen: int, tick: int, alive) -> None:
        """Optionally dump a deterministic digest after tick-0 env apply."""

        if int(tick) != 0:
            return
        if os.environ.get("EIEL_STATE_DUMP", "0").strip() != "1":
            return
        gens = os.environ.get("EIEL_STATE_DUMP_GENS", "").strip()
        if gens:
            try:
                allowed = {int(x) for x in gens.split(",") if x.strip()}
            except Exception:
                allowed = set()
            if allowed and int(gen) not in allowed:
                return

        try:
            from engine.simulation.state_dump import maybe_dump_state  # noqa: WPS433

            alive_ids_seq = [int(getattr(a, "agent_id")) for a in alive]
            b = (",".join(str(x) for x in alive_ids_seq)).encode("utf-8")
            alive_ids_seq_sha = hashlib.sha256(b).hexdigest()

            extra = {
                "alive_n": int(len(alive)),
                "alive_ids_seq_sha256": alive_ids_seq_sha,
                "alive_ids_seq_head": alive_ids_seq[:64],
            }
            maybe_dump_state(self, generation_index=int(gen), tag="tick0_env_post", extra=extra)
        except Exception:
            return

    def _batched_actions_strided_slot_cache(self, alive, buckets: Dict[str, float] | None = None) -> np.ndarray:
        """Slot-indexed policy stride cache, guarded by agent_id and slot_token.

        This AGGPERF-3 fast path keeps the current refill order but avoids
        per-agent dict lookups on stride-cache hit ticks.
        """

        stride = int(getattr(self.sim_cfg, "policy_tick_stride", 1) or 1)
        if stride <= 1:
            return self._batched_actions(alive, buckets=buckets)

        b = buckets if isinstance(buckets, dict) else None
        t_total0 = time.perf_counter()
        n = int(len(alive) if alive is not None else 0)
        actions_out = np.empty((n,), dtype=np.int32)
        if n <= 0:
            if b is not None:
                elapsed = time.perf_counter() - t_total0
                b["policy_stride_total"] = float(b.get("policy_stride_total", 0.0) + elapsed)
                b["policy_stride_slot_cache_total"] = float(
                    b.get("policy_stride_slot_cache_total", 0.0) + elapsed
                )
            return actions_out

        cap = int(getattr(getattr(self, "pop", None), "max_slots", 0) or 0)
        if cap <= 0:
            try:
                cap = int(getattr(getattr(self, "env", None), "agent_alive").shape[0])
            except Exception:
                cap = max(1, n)

        cache = getattr(self, "_policy_action_slot_cache", None)
        if (
            not isinstance(cache, dict)
            or int(cache.get("cap", 0) or 0) < cap
            or cache.get("agent_ids") is None
            or cache.get("tokens") is None
            or cache.get("cursors") is None
            or cache.get("actions") is None
        ):
            cache = {
                "cap": int(cap),
                "agent_ids": np.full((int(cap),), -1, dtype=np.int64),
                "tokens": np.full((int(cap),), -1, dtype=np.int64),
                "cursors": np.zeros((int(cap),), dtype=np.int16),
                "actions": [None for _ in range(int(cap))],
            }
            setattr(self, "_policy_action_slot_cache", cache)

        agent_ids_arr = cache["agent_ids"]
        tokens_arr = cache["tokens"]
        cursors_arr = cache["cursors"]
        action_rows = cache["actions"]
        cap = int(cache["cap"])

        refill_agents: list[Any] = []
        refill_out_idx: list[int] = []
        refill_agent_ids: list[int] = []
        refill_slots: list[int] = []
        refill_tokens: list[int] = []
        hits = 0
        misses = 0

        t_serve0 = time.perf_counter()
        for i, a in enumerate(alive):
            try:
                slot = int(getattr(a, "slot", -1))
            except Exception:
                slot = -1
            try:
                aid = int(getattr(a, "agent_id"))
            except Exception:
                aid = 0
            try:
                tok = int(getattr(a, "slot_token", 0))
            except Exception:
                tok = 0

            if 0 <= slot < cap:
                seq = action_rows[slot]
                try:
                    cur = int(cursors_arr[slot])
                    if (
                        seq is not None
                        and int(agent_ids_arr[slot]) == aid
                        and int(tokens_arr[slot]) == tok
                        and cur < int(len(seq))
                    ):
                        act = int(seq[cur])
                        actions_out[i] = np.int32(act)
                        cursors_arr[slot] = np.int16(cur + 1)
                        hits += 1
                        continue
                except Exception:
                    pass

            misses += 1
            refill_agents.append(a)
            refill_out_idx.append(int(i))
            refill_agent_ids.append(int(aid))
            refill_slots.append(int(slot))
            refill_tokens.append(int(tok))

        if b is not None:
            serve_elapsed = time.perf_counter() - t_serve0
            b["policy_stride_serve"] = float(b.get("policy_stride_serve", 0.0) + serve_elapsed)
            b["policy_stride_slot_cache_serve"] = float(
                b.get("policy_stride_slot_cache_serve", 0.0) + serve_elapsed
            )
            b["policy_stride_hit_count"] = float(b.get("policy_stride_hit_count", 0.0) + float(hits))
            b["policy_stride_miss_count"] = float(b.get("policy_stride_miss_count", 0.0) + float(misses))

        if refill_agents:
            t_refill0 = time.perf_counter()
            t_sort0 = time.perf_counter()
            order = sorted(range(len(refill_agents)), key=lambda j: int(refill_agent_ids[j]))
            refill_agents_sorted = [refill_agents[j] for j in order]
            if b is not None:
                b["policy_stride_refill_sort"] = float(
                    b.get("policy_stride_refill_sort", 0.0) + (time.perf_counter() - t_sort0)
                )

            t_call0 = time.perf_counter()
            actions_k_sorted = self._batched_actions_k(refill_agents_sorted, k=stride, buckets=buckets)
            if b is not None:
                b["policy_stride_refill_call"] = float(
                    b.get("policy_stride_refill_call", 0.0) + (time.perf_counter() - t_call0)
                )

            t_write0 = time.perf_counter()
            for row_i, j in enumerate(order):
                out_i = int(refill_out_idx[j])
                aid = int(refill_agent_ids[j])
                slot = int(refill_slots[j])
                tok = int(refill_tokens[j])
                seq = actions_k_sorted[row_i].astype(np.int32, copy=False)
                actions_out[out_i] = np.int32(int(seq[0]))
                if 0 <= slot < cap:
                    agent_ids_arr[slot] = np.int64(aid)
                    tokens_arr[slot] = np.int64(tok)
                    cursors_arr[slot] = np.int16(1)
                    action_rows[slot] = seq
            if b is not None:
                b["policy_stride_refill_write"] = float(
                    b.get("policy_stride_refill_write", 0.0) + (time.perf_counter() - t_write0)
                )
                b["policy_stride_refill_total"] = float(
                    b.get("policy_stride_refill_total", 0.0) + (time.perf_counter() - t_refill0)
                )

        if b is not None:
            elapsed = time.perf_counter() - t_total0
            b["policy_stride_total"] = float(b.get("policy_stride_total", 0.0) + elapsed)
            b["policy_stride_slot_cache_total"] = float(
                b.get("policy_stride_slot_cache_total", 0.0) + elapsed
            )

        self._record_reference_policy_cache_rows(requested=n, hit=hits, missed=misses)
        return actions_out

    def _batched_actions_strided(self, alive, buckets: Dict[str, float]):
        """Return an action for each alive agent using a stride-based cache.

        Key points
        ----------
        - Semantics match the non-strided policy path: we produce the same actions
          for the same (agent, obs, policy) state, just amortizing forward calls.
        - Determinism is preserved because policy inference is deterministic and
          we keep a canonical refill order (by agent_id) whenever we must refill.
        - Cache is keyed by agent_id (NOT slot) because slots can churn under
          heavy birth/death/compaction pressure, which would destroy hit-rate.
        """

        def _fixed_probe_actions(actions):
            try:
                from simulation.mosaic_fixed1_controller import choose_actions

                actions = choose_actions(self, alive, actions)
            except Exception:
                if bool(getattr(self.sim_cfg, "mosaic_fixed1_probe_enable", False)):
                    raise
            try:
                from simulation.predrisk_fixed1_probe import choose_actions

                actions = choose_actions(self, alive, actions)
            except Exception:
                if bool(getattr(self.sim_cfg, "predrisk_fixed1_probe_enable", False)):
                    raise
            try:
                from simulation.corridor_cp0_fixed_probe import choose_actions

                actions = choose_actions(self, alive, actions)
            except Exception:
                if bool(getattr(self.sim_cfg, "corridor_cp0_probe_enable", False)):
                    raise
            try:
                from simulation.habacclim1 import choose_actions

                actions = choose_actions(self, alive, actions)
            except Exception:
                if bool(getattr(self.sim_cfg, "habacclim1_probe_enable", False)):
                    raise
            return actions

        stride = int(getattr(self.sim_cfg, "policy_tick_stride", 1) or 1)
        if stride <= 1:
            actions = self._batched_actions(alive, buckets=buckets)
            requested = int(len(alive) if alive is not None else 0)
            self._record_reference_policy_cache_rows(requested=requested, hit=0, missed=requested)
            return _fixed_probe_actions(actions)
        if bool(getattr(self.sim_cfg, "perf_policy_stride_slot_cache_enable", False)):
            return _fixed_probe_actions(self._batched_actions_strided_slot_cache(alive, buckets=buckets))

        b = buckets if isinstance(buckets, dict) else None
        t_total0 = time.perf_counter()

        # Cache is a dict keyed by agent_id -> [token:int, actions:np.ndarray(int32), cursor:int]
        cache = getattr(self, "_policy_action_cache", None)
        if cache is None or not isinstance(cache, dict):
            cache = {}
            setattr(self, "_policy_action_cache", cache)

        n = int(len(alive) if alive is not None else 0)
        actions_out = np.empty((n,), dtype=np.int32)
        if n <= 0:
            if b is not None:
                b["policy_stride_total"] = float(b.get("policy_stride_total", 0.0) + (time.perf_counter() - t_total0))
            return _fixed_probe_actions(actions_out)

        # ------------------------------------------------------------
        # Pass 1: serve hits, collect misses for refill
        # ------------------------------------------------------------
        refill_agents = []
        refill_out_idx = []
        refill_agent_ids = []

        hits = 0
        misses = 0

        t_serve0 = time.perf_counter()
        cache_get = cache.get

        for i, a in enumerate(alive):
            try:
                aid = int(getattr(a, "agent_id"))
            except Exception:
                aid = 0
            try:
                tok = int(getattr(a, "slot_token", 0))
            except Exception:
                tok = 0

            entry = cache_get(aid)

            # Entry format: [tok, actions, cursor]
            if entry is not None and isinstance(entry, list) and len(entry) >= 3:
                try:
                    ent_tok = int(entry[0])
                    ent_actions = entry[1]
                    ent_cursor = int(entry[2])
                    ent_len = int(len(ent_actions)) if ent_actions is not None else 0

                    if ent_tok == tok and ent_cursor < ent_len:
                        act = int(ent_actions[ent_cursor])
                        actions_out[i] = np.int32(act)
                        entry[2] = int(ent_cursor + 1)
                        hits += 1
                        continue
                except Exception:
                    # fall through to miss
                    pass

            misses += 1
            refill_agents.append(a)
            refill_out_idx.append(int(i))
            refill_agent_ids.append(int(aid))

        if b is not None:
            b["policy_stride_serve"] = float(b.get("policy_stride_serve", 0.0) + (time.perf_counter() - t_serve0))
            # counts (stored as floats; these are NOT seconds)
            b["policy_stride_hit_count"] = float(b.get("policy_stride_hit_count", 0.0) + float(hits))
            b["policy_stride_miss_count"] = float(b.get("policy_stride_miss_count", 0.0) + float(misses))

        # ------------------------------------------------------------
        # Pass 2: refill misses (canonical order for GRU determinism)
        # ------------------------------------------------------------
        if refill_agents:
            t_refill0 = time.perf_counter()

            t_sort0 = time.perf_counter()
            order = sorted(range(len(refill_agents)), key=lambda j: int(refill_agent_ids[j]))
            refill_agents_sorted = [refill_agents[j] for j in order]
            if b is not None:
                b["policy_stride_refill_sort"] = float(b.get("policy_stride_refill_sort", 0.0) + (time.perf_counter() - t_sort0))

            t_call0 = time.perf_counter()
            actions_k_sorted = self._batched_actions_k(refill_agents_sorted, k=stride, buckets=buckets)  # (m, stride)
            if b is not None:
                b["policy_stride_refill_call"] = float(b.get("policy_stride_refill_call", 0.0) + (time.perf_counter() - t_call0))

            t_write0 = time.perf_counter()
            for row_i, j in enumerate(order):
                a = refill_agents[j]
                out_i = int(refill_out_idx[j])

                try:
                    aid = int(getattr(a, "agent_id"))
                except Exception:
                    aid = int(refill_agent_ids[j])
                try:
                    tok = int(getattr(a, "slot_token", 0))
                except Exception:
                    tok = 0

                seq = actions_k_sorted[row_i].astype(np.int32, copy=False)
                act0 = int(seq[0])
                actions_out[out_i] = np.int32(act0)

                # Cache remaining actions for future ticks
                try:
                    cache[aid] = [tok, seq, 1]
                except Exception:
                    pass

            if b is not None:
                b["policy_stride_refill_write"] = float(b.get("policy_stride_refill_write", 0.0) + (time.perf_counter() - t_write0))
                b["policy_stride_refill_total"] = float(b.get("policy_stride_refill_total", 0.0) + (time.perf_counter() - t_refill0))

        if b is not None:
            b["policy_stride_total"] = float(b.get("policy_stride_total", 0.0) + (time.perf_counter() - t_total0))

        self._record_reference_policy_cache_rows(requested=n, hit=hits, missed=misses)
        return _fixed_probe_actions(actions_out)

        # ------------------------------------------------------------
        # Debug: stride-token trace (only when explicitly enabled)
        # ------------------------------------------------------------
        dbg_on = os.environ.get("EIEL_DEBUG_STRIDE_TOKEN", "0").strip() == "1"
        dbg_ticks = os.environ.get("EIEL_DEBUG_STRIDE_TOKEN_TICKS", "").strip()
        dbg_slots_raw = os.environ.get("EIEL_DEBUG_STRIDE_TOKEN_SLOTS", "").strip()
        dbg_tickset = None
        dbg_slots = None
        if dbg_on:
            if dbg_ticks:
                try:
                    dbg_tickset = {int(x) for x in dbg_ticks.split(",") if x.strip()}
                except Exception:
                    dbg_tickset = None
            if dbg_slots_raw:
                try:
                    dbg_slots = {int(x) for x in dbg_slots_raw.split(",") if x.strip()}
                except Exception:
                    dbg_slots = None

        dbg_gen = int(getattr(self, "_eiel_dbg_gen", -1))
        dbg_tick = int(getattr(self, "_eiel_dbg_tick", -1))
        if dbg_on and dbg_tickset is not None and dbg_tick not in dbg_tickset:
            dbg_on = False  # outside requested ticks

        # ------------------------------------------------------------
        # Per-tick served guard (avoid consuming the cache twice when a
        # slot is queried twice within the same tick).
        # ------------------------------------------------------------
        cur_tick = dbg_tick
        cap = int(getattr(getattr(self, "pop", None), "max_slots", 0) or 0)

        served_stamp = None
        served_action = None
        if cap > 0 and cur_tick >= 0:
            try:
                served_stamp = getattr(self, "_stride_served_stamp", None)
                served_action = getattr(self, "_stride_served_action", None)
                if (
                    served_stamp is None
                    or served_action is None
                    or (not isinstance(served_stamp, np.ndarray))
                    or (not isinstance(served_action, np.ndarray))
                    or int(getattr(served_stamp, "shape", [0])[0]) != cap
                    or int(getattr(served_action, "shape", [0])[0]) != cap
                ):
                    served_stamp = np.full((cap,), -1, dtype=np.int32)
                    served_action = np.zeros((cap,), dtype=np.int32)
                    setattr(self, "_stride_served_stamp", served_stamp)
                    setattr(self, "_stride_served_action", served_action)
            except Exception:
                served_stamp = None
                served_action = None

        # ------------------------------------------------------------
        # Pass 1: serve hits, collect misses for refill
        # ------------------------------------------------------------
        refill_agents = []
        refill_out_idx = []
        refill_agent_ids = []

        t_serve0 = time.perf_counter()
        cache_get = cache.get

        for i, a in enumerate(alive):
            try:
                slot = int(getattr(a, "slot"))
            except Exception:
                slot = -1

            # Single-serve guard
            if served_stamp is not None and 0 <= slot < cap and int(served_stamp[slot]) == cur_tick:
                actions_out[i] = int(served_action[slot])
                continue

            try:
                aid = int(getattr(a, "agent_id"))
            except Exception:
                aid = 0
            try:
                tok = int(getattr(a, "slot_token", 0))
            except Exception:
                tok = 0

            entry = cache_get(slot) if slot >= 0 else None

            # ---- Fast list entry ----
            if entry is not None and isinstance(entry, list) and len(entry) >= 4:
                try:
                    ent_aid = int(entry[0])
                    ent_actions = entry[2]
                    ent_cursor = int(entry[3])
                    ent_len = int(len(ent_actions)) if ent_actions is not None else 0
                    if ent_aid == aid and ent_cursor < ent_len:
                        act = int(ent_actions[ent_cursor])
                        actions_out[i] = act
                        # Advance cursor in-place
                        entry[3] = int(ent_cursor + 1)

                        if served_stamp is not None and 0 <= slot < cap:
                            served_stamp[slot] = np.int32(cur_tick)
                            served_action[slot] = np.int32(act)
                        if dbg_on and (dbg_slots is None or slot in dbg_slots):
                            print(
                                f"[stride_token] gen={dbg_gen} tick={dbg_tick} out_i={i} agent_id={aid} "
                                f"slot={slot} tok={tok} kind=hit cursor={ent_cursor}->{ent_cursor+1}",
                                flush=True,
                            )
                        continue
                except Exception:
                    pass

            # ---- Legacy dict entry (resume compatibility) ----
            if entry is not None and isinstance(entry, dict):
                try:
                    ent_actions = entry.get("actions", None)
                    ent_cursor = int(entry.get("cursor", 0) or 0)
                    ent_len = int(len(ent_actions)) if ent_actions is not None else 0

                    ent_aid = None
                    if "agent_id" in entry:
                        try:
                            ent_aid = int(entry.get("agent_id", -1))
                        except Exception:
                            ent_aid = None
                    ent_tok = int(entry.get("token", -1) or -1)

                    match = False
                    if ent_aid is not None:
                        match = (ent_aid == aid)
                    else:
                        match = (ent_tok == tok)

                    if match and ent_cursor < ent_len:
                        act = int(ent_actions[ent_cursor])
                        actions_out[i] = act
                        entry["cursor"] = int(ent_cursor + 1)

                        if served_stamp is not None and 0 <= slot < cap:
                            served_stamp[slot] = np.int32(cur_tick)
                            served_action[slot] = np.int32(act)

                        # Opportunistically upgrade to the fast list representation (future hits)
                        try:
                            cache[slot] = [aid, tok, ent_actions, int(ent_cursor + 1)]
                        except Exception:
                            pass

                        if dbg_on and (dbg_slots is None or slot in dbg_slots):
                            print(
                                f"[stride_token] gen={dbg_gen} tick={dbg_tick} out_i={i} agent_id={aid} "
                                f"slot={slot} tok={tok} kind=hit cursor={ent_cursor}->{ent_cursor+1}",
                                flush=True,
                            )
                        continue
                except Exception:
                    pass

            # Miss: we must refill this agent's action sequence
            refill_agents.append(a)
            refill_out_idx.append(int(i))
            refill_agent_ids.append(int(aid))

            if dbg_on and (dbg_slots is None or slot in dbg_slots):
                try:
                    print(
                        f"[stride_token] gen={dbg_gen} tick={dbg_tick} out_i={i} agent_id={aid} "
                        f"slot={slot} tok={tok} kind=miss",
                        flush=True,
                    )
                except Exception:
                    pass

        if b is not None:
            b["policy_stride_serve"] = float(b.get("policy_stride_serve", 0.0) + (time.perf_counter() - t_serve0))

        # ------------------------------------------------------------
        # Pass 2: refill misses (canonical order for GRU determinism)
        # ------------------------------------------------------------
        if refill_agents:
            t_refill0 = time.perf_counter()

            # Phase 3 determinism: canonicalize GRU batch order.
            t_sort0 = time.perf_counter()
            order = sorted(range(len(refill_agents)), key=lambda j: int(refill_agent_ids[j]))
            refill_agents_sorted = [refill_agents[j] for j in order]
            if b is not None:
                b["policy_stride_refill_sort"] = float(b.get("policy_stride_refill_sort", 0.0) + (time.perf_counter() - t_sort0))

            # Model call (this is the expensive part; it already has fine-grained buckets)
            t_call0 = time.perf_counter()
            actions_k_sorted = self._batched_actions_k(refill_agents_sorted, k=stride, buckets=buckets)  # (m, stride)
            if b is not None:
                b["policy_stride_refill_call"] = float(b.get("policy_stride_refill_call", 0.0) + (time.perf_counter() - t_call0))

            # Write current action + update caches
            t_write0 = time.perf_counter()
            for row_i, j in enumerate(order):
                a = refill_agents[j]
                out_i = int(refill_out_idx[j])

                try:
                    slot = int(getattr(a, "slot"))
                except Exception:
                    slot = -1
                try:
                    tok = int(getattr(a, "slot_token", 0))
                except Exception:
                    tok = 0
                try:
                    aid = int(getattr(a, "agent_id"))
                except Exception:
                    aid = int(refill_agent_ids[j])

                seq = actions_k_sorted[row_i].astype(np.int32, copy=False)
                act0 = int(seq[0])
                actions_out[out_i] = np.int32(act0)

                if served_stamp is not None and 0 <= slot < cap:
                    served_stamp[slot] = np.int32(cur_tick)
                    served_action[slot] = np.int32(act0)

                # Cache remaining actions for future ticks
                if slot >= 0:
                    try:
                        cache[slot] = [aid, tok, seq, 1]
                    except Exception:
                        # Fallback: keep legacy dict if list update fails for some reason
                        cache[slot] = {"agent_id": aid, "token": tok, "actions": seq, "cursor": 1}

                if dbg_on and (dbg_slots is None or slot in dbg_slots):
                    try:
                        print(
                            f"[stride_token] gen={dbg_gen} tick={dbg_tick} out_i={out_i} agent_id={aid} "
                            f"slot={slot} tok={tok} kind=refill cursor=0->1",
                            flush=True,
                        )
                    except Exception:
                        pass

            if b is not None:
                b["policy_stride_refill_write"] = float(b.get("policy_stride_refill_write", 0.0) + (time.perf_counter() - t_write0))
                b["policy_stride_refill_total"] = float(b.get("policy_stride_refill_total", 0.0) + (time.perf_counter() - t_refill0))

        if b is not None:
            b["policy_stride_total"] = float(b.get("policy_stride_total", 0.0) + (time.perf_counter() - t_total0))

        return actions_out

    def _cxxrewrite_resident_generation_requested(self) -> bool:
        cfg = getattr(self, "sim_cfg", None)
        return bool(getattr(cfg, "cxxrewrite_resident_generation_runner_enable", False))

    @staticmethod
    def _cxxrewrite_resident_generation_truthy_env(name: str) -> bool:
        value = os.environ.get(str(name))
        if value is None:
            return False
        return str(value).strip().lower() not in ("", "0", "false", "no", "off")

    def _cxxrewrite_resident_generation_unsupported_reasons(self) -> list[str]:
        cfg = getattr(self, "sim_cfg", None)
        reasons: list[str] = []
        if not self._cxxrewrite_resident_generation_requested():
            return ["not_requested"]
        if bool(getattr(cfg, "enable_profiler", False)):
            reasons.append("torch_profiler_enabled")
        for name in (
            "EIEL_STATE_DUMP",
            "EIEL_DETERMINISM_PROBE",
            "EIEL_RNG_DRIFT_PROBE",
            "EIEL_BIRTH_DELTA_PROBE",
            "EIEL_DEBUG_TICK_TRACE",
            "EIEL_DEBUG_ALIVE_ORDER",
            "EIEL_DEBUG_POLICY_SLOTS",
            "EIEL_DEBUG_STRIDE_TOKEN",
            "EIEL_GUARD_ENV_ALIVE",
            "EIEL_MEMBERSHIP_PARITY",
            "EIEL_MEMBERSHIP_COMPONENTS",
        ):
            if self._cxxrewrite_resident_generation_truthy_env(name):
                reasons.append(f"debug_env:{name}")
        for flag in (
            "cxxcore_env_apply_shadow_enable",
            "cxxcore11_live_runner_hook_enable",
            "cxxrewrite_native_tick_enable",
            "cxxrewrite_native_tick_shadow_enable",
            "cxxrewrite_native_tick_authoritative_enable",
            "cxxrewrite_native_observation_enable",
            "cxxrewrite_native_policy_bridge_enable",
            "cxxrewrite_resident_frame_enable",
            "cxxrewrite_resident_frame_shadow_enable",
            "cxxrewrite_resident_frame_authoritative_enable",
            "cxxrewrite_resident_frame_native_packing_enable",
            "cxxrewrite_resident_observation_enable",
            "cxxrewrite_resident_observation_shadow_enable",
            "cxxrewrite_resident_observation_authoritative_enable",
            "cxxrewrite_resident_observation_native_pack_enable",
            "cxxrewrite_resident_policyframe_enable",
            "cxxrewrite_resident_policyframe_shadow_enable",
            "cxxrewrite_resident_policyframe_authoritative_enable",
            "cxxrewrite_resident_policyframe_native_pack_enable",
            "cxxrewrite_resident_post_env_scalar_enable",
            "cxxrewrite_resident_post_env_scalar_shadow_enable",
            "cxxrewrite_resident_post_env_scalar_authoritative_enable",
        ):
            if bool(getattr(cfg, flag, False)):
                reasons.append(f"unsupported_gate:{flag}")
        return reasons

    def _cxxrewrite_resident_generation_record_fallback(self, reasons: list[str]) -> None:
        summary = getattr(self, "_cxxrewrite_resident_generation_runner_summary", None)
        if not isinstance(summary, dict):
            summary = {
                "schema_version": "cxxrewrite_resident_generation_runner_summary_v1",
                "fast_generations": 0,
                "fallback_generations": 0,
                "fallback_reasons": {},
                "ticks": 0,
                "alive_build_source": "env_alive_intersection_pop_agents",
            }
            setattr(self, "_cxxrewrite_resident_generation_runner_summary", summary)
        summary["fallback_generations"] = int(summary.get("fallback_generations", 0)) + 1
        counts = summary.setdefault("fallback_reasons", {})
        for reason in reasons:
            counts[str(reason)] = int(counts.get(str(reason), 0)) + 1
        self._cxxrewrite_resident_generation_write_summary()

    def _cxxrewrite_resident_generation_record_fast(self, *, ticks_ran: int, elapsed_s: float) -> None:
        summary = getattr(self, "_cxxrewrite_resident_generation_runner_summary", None)
        if not isinstance(summary, dict):
            summary = {
                "schema_version": "cxxrewrite_resident_generation_runner_summary_v1",
                "fast_generations": 0,
                "fallback_generations": 0,
                "fallback_reasons": {},
                "ticks": 0,
                "alive_build_source": "env_alive_intersection_pop_agents",
            }
            setattr(self, "_cxxrewrite_resident_generation_runner_summary", summary)
        summary["fast_generations"] = int(summary.get("fast_generations", 0)) + 1
        summary["ticks"] = int(summary.get("ticks", 0)) + int(ticks_ran)
        summary["last_generation_elapsed_s"] = float(elapsed_s)
        self._cxxrewrite_resident_generation_write_summary()

    def _cxxrewrite_resident_generation_write_summary(self) -> None:
        cfg = getattr(self, "sim_cfg", None)
        if not bool(getattr(cfg, "cxxrewrite_resident_generation_runner_write_summary", True)):
            return
        run_dir = getattr(self, "run_dir", None) or getattr(self, "run_dir_path", None) or ""
        if not run_dir:
            return
        summary = getattr(self, "_cxxrewrite_resident_generation_runner_summary", None)
        if not isinstance(summary, dict):
            return
        try:
            out_dir = os.path.join(str(run_dir), "metrics")
            os.makedirs(out_dir, exist_ok=True)
            filename = str(
                getattr(
                    cfg,
                    "cxxrewrite_resident_generation_runner_summary_filename",
                    "cxxrewrite_resident_generation_runner_summary.json",
                )
                or "cxxrewrite_resident_generation_runner_summary.json"
            )
            out_path = os.path.join(out_dir, filename)
            tmp_path = out_path + ".tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, sort_keys=True)
                f.write("\n")
            os.replace(tmp_path, out_path)
        except Exception:
            if bool(getattr(cfg, "cxxrewrite_resident_generation_runner_fail_closed", False)):
                raise

    def _cxxwave1g_full_cpp_backbone_requested(self) -> bool:
        cfg = getattr(self, "sim_cfg", None)
        return bool(getattr(cfg, "cxxwave1g_full_cpp_backbone_enable", False))

    def _cxxbackbone2_live_composition_requested(self) -> bool:
        cfg = getattr(self, "sim_cfg", None)
        return bool(getattr(cfg, "cxxbackbone2_live_composition_enable", False))

    def _cxxbackbone2_live_composition_write_summary(self, payload: dict[str, Any]) -> None:
        cfg = getattr(self, "sim_cfg", None)
        if not bool(getattr(cfg, "cxxbackbone2_live_composition_write_summary", True)):
            return
        run_dir = getattr(self, "run_dir", None) or getattr(self, "run_dir_path", None) or ""
        if not run_dir:
            return
        filename = str(
            getattr(
                cfg,
                "cxxbackbone2_live_composition_summary_filename",
                "cxxbackbone2_live_composition_summary.json",
            )
            or "cxxbackbone2_live_composition_summary.json"
        )
        out_path = Path(str(run_dir)) / "metrics" / filename
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
        os.replace(str(tmp_path), str(out_path))

    def _cxxbackbone2_live_composition_attempt(self, *, gen: int, buckets: Dict[str, float]) -> dict[str, Any]:
        from perf.cxxbackbone2_live_composition import (  # noqa: WPS433
            execute_generation as cxxbackbone2_execute_generation,
            summary_payload as cxxbackbone2_summary_payload,
        )

        result = cxxbackbone2_execute_generation(
            self,
            generation=int(gen),
            buckets=buckets,
            implementation_commit="6e8dd36e6c3b0d9597f416ab9934f901470209ce",
        )
        payload = cxxbackbone2_summary_payload(result)
        setattr(self, "_cxxbackbone2_live_composition_summary", payload)
        self._cxxbackbone2_live_composition_write_summary(payload)
        return result

    def _cxxwave1g_full_cpp_backbone_write_summary(self, payload: dict[str, Any]) -> None:
        cfg = getattr(self, "sim_cfg", None)
        if not bool(getattr(cfg, "cxxwave1g_full_cpp_backbone_write_summary", True)):
            return
        run_dir = getattr(self, "run_dir", None) or getattr(self, "run_dir_path", None) or ""
        if not run_dir:
            return
        filename = str(
            getattr(
                cfg,
                "cxxwave1g_full_cpp_backbone_summary_filename",
                "cxxwave1g_full_live_cpp_backbone_summary.json",
            )
            or "cxxwave1g_full_live_cpp_backbone_summary.json"
        )
        out_path = Path(str(run_dir)) / "metrics" / filename
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
        os.replace(str(tmp_path), str(out_path))

    def _cxxwave1g_full_cpp_backbone_attempt(self, *, gen: int) -> dict[str, Any]:
        from perf.cxxwave1g_full_live_backbone import (  # noqa: WPS433
            capture_resident_runtime_state_v3,
            execute_generation,
            first_divergence_proof,
            resident_runtime_state_payload,
        )

        state = capture_resident_runtime_state_v3(
            self,
            generation=int(gen),
            tick=0,
            cell_id=str(getattr(getattr(self, "sim_cfg", None), "campaign_cell_id", "") or f"generation_{int(gen)}"),
        )
        result = execute_generation(
            state,
            {
                "run_dir": str(getattr(self, "run_dir", "") or ""),
                "profile": str(getattr(getattr(self, "sim_cfg", None), "performance_profile_name", "") or ""),
            },
            generation_index=int(gen),
            steps_per_generation=int(getattr(getattr(self, "sim_cfg", None), "steps_per_generation", 0) or 0),
        )
        payload = {
            "schema_version": "cxxwave1g_full_live_cpp_backbone_summary_v1",
            "generation": int(gen),
            "resident_state": resident_runtime_state_payload(state),
            "generation_result": result,
            "first_divergence_or_blocker": first_divergence_proof(result),
        }
        setattr(self, "_cxxwave1g_full_cpp_backbone_summary", payload)
        self._cxxwave1g_full_cpp_backbone_write_summary(payload)
        return payload

    def _cxxrewrite_resident_generation_alive_from_env(self, env_alive: Any) -> list[Any]:
        alive_mask = np.asarray(env_alive, dtype=np.bool_)
        pop_agents = getattr(getattr(self, "pop", None), "agents", {}) or {}
        alive: list[Any] = []
        for aid_raw in np.flatnonzero(alive_mask).astype(np.int64).tolist():
            agent = pop_agents.get(int(aid_raw))
            if agent is not None and bool(getattr(agent, "alive", False)):
                alive.append(agent)
        return alive

    def _cxxrewrite_resident_generation_record_patch_tick(
        self,
        *,
        gen: int,
        tick: int,
        buckets: Dict[str, float],
        aggperf3_timing: bool,
    ) -> None:
        t_aggperf3_env_context0 = time.perf_counter() if aggperf3_timing else 0.0
        if bool(getattr(self.env, "patch_world_enable", False)):
            patch_abs_step = int(gen) * int(self.sim_cfg.steps_per_generation) + int(tick)
            self.env.patch_world_step(patch_abs_step)
            if bool(getattr(self.env_cfg, "wdl_delayed_hazard_enable", False)):
                self._wdl_delayed_hazard_tick_count_gen = int(getattr(self, "_wdl_delayed_hazard_tick_count_gen", 0)) + 1
                self._wdl_delayed_hazard_event_count_gen = int(getattr(self, "_wdl_delayed_hazard_event_count_gen", 0)) + int(getattr(self.env, "wdl_delayed_hazard_event_count_last", 0))
                cue_cells = int(getattr(self.env, "wdl_delayed_hazard_cue_cell_count_last", 0))
                active_cells = int(getattr(self.env, "wdl_delayed_hazard_active_cell_count_last", 0))
                try:
                    prior = np.asarray(getattr(self.env, "patch_delayed_hazard_prior_cue"), dtype=np.int8)
                    cue = np.asarray(getattr(self.env, "patch_delayed_hazard_cue"), dtype=np.float32)
                    active = np.asarray(getattr(self.env, "patch_delayed_hazard_active"), dtype=np.float32)
                    hidden_cells = int(np.sum((prior > 0) & (cue <= 0.0) & (active <= 0.0)))
                except Exception:
                    hidden_cells = 0
                self._wdl_delayed_hazard_cue_cell_count_sum_gen = float(getattr(self, "_wdl_delayed_hazard_cue_cell_count_sum_gen", 0.0)) + float(cue_cells)
                self._wdl_delayed_hazard_active_cell_count_sum_gen = float(getattr(self, "_wdl_delayed_hazard_active_cell_count_sum_gen", 0.0)) + float(active_cells)
                if cue_cells > 0:
                    self._path1_drh0_cue_visible_steps_gen = int(getattr(self, "_path1_drh0_cue_visible_steps_gen", 0)) + 1
                if hidden_cells > 0:
                    self._path1_drh0_hidden_window_steps_gen = int(getattr(self, "_path1_drh0_hidden_window_steps_gen", 0)) + 1
                if active_cells > 0:
                    self._path1_drh0_hazard_active_steps_gen = int(getattr(self, "_path1_drh0_hazard_active_steps_gen", 0)) + 1
                self._wdl_delayed_hazard_agent_cue_exposure_count_gen = int(getattr(self, "_wdl_delayed_hazard_agent_cue_exposure_count_gen", 0)) + int(cue_cells)
                self._wdl_delayed_hazard_cue_to_active_delay_sum_gen = float(getattr(self, "_wdl_delayed_hazard_cue_to_active_delay_sum_gen", 0.0)) + float(getattr(self.env, "wdl_delayed_hazard_cue_to_active_delay_sum_last", 0.0))
                self._wdl_delayed_hazard_cue_to_active_delay_count_gen = int(getattr(self, "_wdl_delayed_hazard_cue_to_active_delay_count_gen", 0)) + int(getattr(self.env, "wdl_delayed_hazard_cue_to_active_delay_count_last", 0))
                self._wdl_delayed_hazard_cue_without_active_count_gen = int(getattr(self, "_wdl_delayed_hazard_cue_without_active_count_gen", 0)) + int(getattr(self.env, "wdl_delayed_hazard_cue_without_active_count_last", 0))
                self._wdl_delayed_hazard_active_without_prior_cue_count_gen = int(getattr(self, "_wdl_delayed_hazard_active_without_prior_cue_count_gen", 0)) + int(getattr(self.env, "wdl_delayed_hazard_active_without_prior_cue_count_last", 0))
            if bool(getattr(self.env_cfg, "prediction_demand_enable", False)):
                self._prediction_demand_tick_count_gen = int(getattr(self, "_prediction_demand_tick_count_gen", 0)) + 1
                self._prediction_demand_event_count_gen = int(getattr(self, "_prediction_demand_event_count_gen", 0)) + int(getattr(self.env, "prediction_demand_event_count_last", 0))
                self._prediction_demand_transition_count_gen = int(getattr(self, "_prediction_demand_transition_count_gen", 0)) + int(getattr(self.env, "prediction_demand_transition_count_last", 0))
                cue_cells = int(getattr(self.env, "prediction_demand_cue_cell_count_last", 0))
                active_cells = int(getattr(self.env, "prediction_demand_active_cell_count_last", 0))
                self._prediction_demand_cue_cell_count_sum_gen = float(getattr(self, "_prediction_demand_cue_cell_count_sum_gen", 0.0)) + float(cue_cells)
                self._prediction_demand_active_cell_count_sum_gen = float(getattr(self, "_prediction_demand_active_cell_count_sum_gen", 0.0)) + float(active_cells)
                self._prediction_demand_cue_ambiguity_sum_gen = float(getattr(self, "_prediction_demand_cue_ambiguity_sum_gen", 0.0)) + float(getattr(self.env, "prediction_demand_cue_ambiguity_sum_last", 0.0))
                self._prediction_demand_cue_ambiguity_count_gen = int(getattr(self, "_prediction_demand_cue_ambiguity_count_gen", 0)) + int(getattr(self.env, "prediction_demand_cue_ambiguity_count_last", 0))
                self._prediction_demand_cue_to_consequence_delay_sum_gen = float(getattr(self, "_prediction_demand_cue_to_consequence_delay_sum_gen", 0.0)) + float(getattr(self.env, "prediction_demand_cue_to_consequence_delay_sum_last", 0.0))
                self._prediction_demand_cue_to_consequence_delay_count_gen = int(getattr(self, "_prediction_demand_cue_to_consequence_delay_count_gen", 0)) + int(getattr(self.env, "prediction_demand_cue_to_consequence_delay_count_last", 0))
                self._prediction_demand_cue_without_consequence_count_gen = int(getattr(self, "_prediction_demand_cue_without_consequence_count_gen", 0)) + int(getattr(self.env, "prediction_demand_cue_without_consequence_count_last", 0))
                self._prediction_demand_consequence_without_prior_cue_count_gen = int(getattr(self, "_prediction_demand_consequence_without_prior_cue_count_gen", 0)) + int(getattr(self.env, "prediction_demand_consequence_without_prior_cue_count_last", 0))
                self._prediction_demand_resource_consequence_total_gen = float(getattr(self, "_prediction_demand_resource_consequence_total_gen", 0.0)) + float(getattr(self.env, "prediction_demand_resource_consequence_total_last", 0.0))
                try:
                    prior = np.asarray(getattr(self.env, "patch_prediction_demand_prior_cue"), dtype=np.int8)
                    active = np.asarray(getattr(self.env, "patch_prediction_demand_active"), dtype=np.float32)
                    ys = np.asarray(getattr(self.env, "agent_pos_y"), dtype=np.int32)
                    xs = np.asarray(getattr(self.env, "agent_pos_x"), dtype=np.int32)
                    pre_count = 0
                    exposure_count = 0
                    for aid in [int(getattr(a, "agent_id")) for a in self.pop.alive_agents()]:
                        y = int(ys[aid])
                        x = int(xs[aid])
                        if y < 0 or x < 0:
                            continue
                        if int(prior[y, x]) > 0 and float(active[y, x]) == 0.0:
                            pre_count += 1
                        if float(active[y, x]) != 0.0:
                            exposure_count += 1
                    self._prediction_demand_pre_consequence_agent_context_count_gen = int(getattr(self, "_prediction_demand_pre_consequence_agent_context_count_gen", 0)) + int(pre_count)
                    self._prediction_demand_consequence_agent_exposure_count_gen = int(getattr(self, "_prediction_demand_consequence_agent_exposure_count_gen", 0)) + int(exposure_count)
                except Exception:
                    pass
            for prefix, enabled_name in (
                ("delayed_resource_depletion", "delayed_resource_depletion_enable"),
                ("delayed_crowding_waste", "delayed_crowding_waste_enable"),
            ):
                if not bool(getattr(self.env_cfg, enabled_name, False)):
                    continue
                setattr(self, f"_{prefix}_tick_count_gen", int(getattr(self, f"_{prefix}_tick_count_gen", 0)) + 1)
                for field in ("event", "cue_without_active", "active_without_prior_cue", "consequence"):
                    setattr(self, f"_{prefix}_{field}_count_gen", int(getattr(self, f"_{prefix}_{field}_count_gen", 0)) + int(getattr(self.env, f"{prefix}_{field}_count_last", 0)))
                for field in ("cue", "hidden", "active"):
                    setattr(self, f"_{prefix}_{field}_cell_count_sum_gen", float(getattr(self, f"_{prefix}_{field}_cell_count_sum_gen", 0.0)) + float(getattr(self.env, f"{prefix}_{field}_cell_count_last", 0.0)))
                setattr(self, f"_{prefix}_consequence_total_gen", float(getattr(self, f"_{prefix}_consequence_total_gen", 0.0)) + float(getattr(self.env, f"{prefix}_consequence_total_last", 0.0)))
                setattr(self, f"_{prefix}_cue_to_active_delay_sum_gen", float(getattr(self, f"_{prefix}_cue_to_active_delay_sum_gen", 0.0)) + float(getattr(self.env, f"{prefix}_cue_to_active_delay_sum_last", 0.0)))
                setattr(self, f"_{prefix}_cue_to_active_delay_count_gen", int(getattr(self, f"_{prefix}_cue_to_active_delay_count_gen", 0)) + int(getattr(self.env, f"{prefix}_cue_to_active_delay_count_last", 0)))
            for attr, env_attr, default in (
                ("_patch_food_regrowth_total_gen", "patch_food_regrowth_total_last", 0.0),
                ("_patch_regrowth_base_total_gen", "patch_regrowth_base_total_last", 0.0),
                ("_patch_regrowth_effective_total_gen", "patch_regrowth_effective_total_last", 0.0),
                ("_patch_regrowth_suppressed_total_gen", "patch_regrowth_suppressed_total_last", 0.0),
                ("_patch_regrowth_multiplier_mean_sum_gen", "patch_regrowth_multiplier_mean_last", 1.0),
                ("_patch_regrowth_multiplier_p10_sum_gen", "patch_regrowth_multiplier_p10_last", 1.0),
                ("_patch_regrowth_multiplier_p90_sum_gen", "patch_regrowth_multiplier_p90_last", 1.0),
                ("_patch_waste_regrowth_penalty_mean_sum_gen", "patch_waste_regrowth_penalty_mean_last", 0.0),
                ("_patch_density_regrowth_penalty_mean_sum_gen", "patch_density_regrowth_penalty_mean_last", 0.0),
                ("_patch_high_waste_regrowth_suppression_frac_sum_gen", "patch_high_waste_regrowth_suppression_frac_last", 0.0),
                ("_patch_high_density_regrowth_suppression_frac_sum_gen", "patch_high_density_regrowth_suppression_frac_last", 0.0),
                ("_patch_recovery_positive_total_gen", "patch_recovery_positive_total_last", 0.0),
                ("_patch_recovery_negative_total_gen", "patch_recovery_negative_total_last", 0.0),
                ("_patch_recovery_decay_total_gen", "patch_recovery_decay_total_last", 0.0),
                ("_jungle_recovery_regrowth_boost_total_gen", "jungle_recovery_regrowth_boost_total_last", 0.0),
                ("_jungle_recovery_regrowth_boost_mean_sum_gen", "jungle_recovery_regrowth_boost_mean_last", 0.0),
                ("_jungle_recovery_regrowth_boost_p90_sum_gen", "jungle_recovery_regrowth_boost_p90_last", 0.0),
                ("_patch_regrowth_after_waste_total_gen", "patch_regrowth_after_waste_total_last", 0.0),
                ("_patch_regrowth_after_recovery_total_gen", "patch_regrowth_after_recovery_total_last", 0.0),
                ("_patch_recovery_added_regrowth_total_gen", "patch_recovery_added_regrowth_total_last", 0.0),
                ("_patch_scent_emitted_total_gen", "patch_scent_emitted_total_last", 0.0),
                ("_patch_scent_decayed_total_gen", "patch_scent_decayed_total_last", 0.0),
                ("_patch_waste_emitted_total_gen", "patch_waste_emitted_total_last", 0.0),
                ("_patch_waste_decayed_total_gen", "patch_waste_decayed_total_last", 0.0),
            ):
                setattr(self, attr, float(getattr(self, attr, 0.0)) + float(getattr(self.env, env_attr, default)))
            self._patch_regrowth_consequence_steps_gen = int(getattr(self, "_patch_regrowth_consequence_steps_gen", 0)) + 1
            self._patch_regrowth_multiplier_min_gen = min(
                float(getattr(self, "_patch_regrowth_multiplier_min_gen", 1.0)),
                float(getattr(self.env, "patch_regrowth_multiplier_min_last", 1.0)),
            )
            self._patch_waste_regrowth_penalty_max_gen = max(
                float(getattr(self, "_patch_waste_regrowth_penalty_max_gen", 0.0)),
                float(getattr(self.env, "patch_waste_regrowth_penalty_max_last", 0.0)),
            )
            self._patch_density_regrowth_penalty_max_gen = max(
                float(getattr(self, "_patch_density_regrowth_penalty_max_gen", 0.0)),
                float(getattr(self.env, "patch_density_regrowth_penalty_max_last", 0.0)),
            )
            self._jungle_recovery_regrowth_boost_max_gen = max(
                float(getattr(self, "_jungle_recovery_regrowth_boost_max_gen", 0.0)),
                float(getattr(self.env, "jungle_recovery_regrowth_boost_max_last", 0.0)),
            )
        if aggperf3_timing:
            buckets["aggperf3_env_context"] = float(
                buckets.get("aggperf3_env_context", 0.0) + (time.perf_counter() - t_aggperf3_env_context0)
            )

    def _run_ticks_resident_generation_fast(
        self,
        gen: int,
        buckets: Dict[str, float],
        injury_recover_mult: float,
        deaths_this_gen: int,
        successful_births: int,
        failed_births: int,
        gen_peak_injury: float,
        rest_streaks: Dict[int, int],
    ) -> Tuple[int, int, int, float, int]:
        started = time.perf_counter()
        ticks_ran = 0
        self._init_generation_tick_state(self.pop.alive_agents())
        env_alive = getattr(self.env, "agent_alive", None)
        if env_alive is None:
            raise AttributeError("env has no 'agent_alive' array; resident generation runner requires array-based env state")
        aggperf3_timing = bool(getattr(self.sim_cfg, "perf_aggperf3_timing_buckets_enable", False))
        wdl_context_enabled = bool(getattr(self.env_cfg, "patch_world_enable", False)) and bool(
            getattr(self.env_cfg, "wdl_delayed_hazard_enable", False)
        )
        delayed_context_enabled = bool(getattr(self.env_cfg, "patch_world_enable", False)) and (
            bool(getattr(self.env_cfg, "delayed_resource_depletion_enable", False))
            or bool(getattr(self.env_cfg, "delayed_crowding_waste_enable", False))
        )

        for tick in range(int(self.sim_cfg.steps_per_generation)):
            try:
                if hasattr(self, "_dev_expr_cache") and isinstance(self._dev_expr_cache, dict):
                    try:
                        diag = getattr(self, "env_hotpath_expression_diagnostics", None)
                        if diag is not None:
                            diag.record_dev_cache_clear(
                                generation=int(gen),
                                tick=int(tick),
                                entries=int(len(self._dev_expr_cache)),
                            )
                    except Exception:
                        pass
                    self._dev_expr_cache.clear()
                    if hasattr(self, "_dev_effective_scalar_cache") and isinstance(self._dev_effective_scalar_cache, dict):
                        self._dev_effective_scalar_cache.clear()
            except Exception:
                pass
            try:
                self._eiel_fdp_cur_tick = int(tick)
                self._eiel_fdp_cur_gen = int(gen)
                self._eiel_dbg_tick = int(tick)
                self._eiel_dbg_gen = int(gen)
            except Exception:
                pass
            if bool(getattr(self, "_stop_requested", False)):
                if not bool(getattr(self, "_stop_tick_notice_printed", False)):
                    try:
                        print("[interrupt] stopping ticks early to reach checkpoint boundary...", flush=True)
                    except Exception:
                        pass
                    setattr(self, "_stop_tick_notice_printed", True)
                break

            self._run_spatial_droughts_step_with_budget(gen=int(gen), tick=int(tick))
            self._cxxrewrite_resident_generation_record_patch_tick(
                gen=int(gen),
                tick=int(tick),
                buckets=buckets,
                aggperf3_timing=aggperf3_timing,
            )
            ticks_ran += 1

            t0 = time.perf_counter()
            self.env.spawn_food()
            buckets["env_spawn_food"] = buckets.get("env_spawn_food", 0.0) + (time.perf_counter() - t0)

            t_jungle0 = time.perf_counter()
            deaths_this_gen = self._apply_jungle_predator_pressure(deaths_this_gen)
            buckets["jungle_predator_pressure"] = buckets.get("jungle_predator_pressure", 0.0) + (
                time.perf_counter() - t_jungle0
            )

            t_alive0 = time.perf_counter()
            env_alive = getattr(self.env, "agent_alive", env_alive)
            alive = self._cxxrewrite_resident_generation_alive_from_env(env_alive)
            buckets["alive_list_build"] = buckets.get("alive_list_build", 0.0) + (time.perf_counter() - t_alive0)
            if not alive:
                break

            if (
                hasattr(self, "resolve_developmental_expression_for_agent")
                and not bool(getattr(self.sim_cfg, "perf_skip_dev_expression_prewarm_enable", False))
            ):
                t_aggperf3_dev0 = time.perf_counter() if aggperf3_timing else 0.0
                for agent in alive:
                    try:
                        self.resolve_developmental_expression_for_agent(agent)
                    except Exception:
                        continue
                if aggperf3_timing:
                    buckets["aggperf3_dev_expression_prewarm"] = float(
                        buckets.get("aggperf3_dev_expression_prewarm", 0.0)
                        + (time.perf_counter() - t_aggperf3_dev0)
                    )

            alive_for_births = list(alive)
            self.np_rng.shuffle(alive)

            t_aggperf3_policy0 = time.perf_counter() if aggperf3_timing else 0.0
            actions_np = self._batched_actions_strided(alive, buckets=buckets)
            agent_ids_np = np.fromiter(
                (int(getattr(agent, "agent_id")) for agent in alive),
                dtype=np.int32,
                count=len(alive),
            )
            if aggperf3_timing:
                buckets["aggperf3_policy_action_selection"] = float(
                    buckets.get("aggperf3_policy_action_selection", 0.0)
                    + (time.perf_counter() - t_aggperf3_policy0)
                )

            self._update_action_coherence(alive, actions_np)

            t_blog0 = time.perf_counter()
            self.behavior_logger.record_actions(agent_ids_np, actions_np)
            try:
                bo = getattr(self, "branch_observability", None)
                if bo is not None:
                    bo.record_niche_pre_env(
                        self,
                        generation=int(gen),
                        tick=int(tick),
                        alive=alive,
                        agent_ids_np=agent_ids_np,
                        actions_np=actions_np,
                    )
            except Exception:
                pass
            if wdl_context_enabled:
                self._record_wdl_action_context_pre_env(alive, agent_ids_np, actions_np)
            if delayed_context_enabled:
                self._record_delayed_resource_waste_action_context_pre_env(alive, agent_ids_np, actions_np)
            cxx23_handled_pre = False
            try:
                cxx23 = getattr(self, "cxxcore23_live_direct_compact_diagnostics", None)
                if cxx23 is not None:
                    cxx23_handled_pre = bool(
                        cxx23.record_pre_env(
                            self,
                            generation=int(gen),
                            tick=int(tick),
                            alive=alive,
                            agent_ids_np=agent_ids_np,
                            actions_np=actions_np,
                        )
                    )
            except Exception:
                raise
            try:
                maw = getattr(self, "memory_action_window_instrumentation", None)
                if maw is not None and not cxx23_handled_pre:
                    maw.record_pre_env(
                        self,
                        generation=int(gen),
                        tick=int(tick),
                        alive=alive,
                        agent_ids_np=agent_ids_np,
                        actions_np=actions_np,
                    )
            except Exception:
                raise
            try:
                mm4 = getattr(self, "mem_max_live_level4_instrumentation", None)
                if mm4 is not None:
                    mm4.record_pre_env(
                        self,
                        generation=int(gen),
                        tick=int(tick),
                        alive=alive,
                        agent_ids_np=agent_ids_np,
                        actions_np=actions_np,
                    )
            except Exception:
                raise
            try:
                pdac = getattr(self, "prediction_demand_action_context_instrumentation", None)
                if pdac is not None and not cxx23_handled_pre:
                    pdac.record_pre_env(
                        self,
                        generation=int(gen),
                        tick=int(tick),
                        alive=alive,
                        agent_ids_np=agent_ids_np,
                        actions_np=actions_np,
                    )
            except Exception:
                raise
            behavior_elapsed = time.perf_counter() - t_blog0
            buckets["behavior_log_tick"] = buckets.get("behavior_log_tick", 0.0) + behavior_elapsed
            if aggperf3_timing:
                buckets["aggperf3_diagnostics_tape"] = float(
                    buckets.get("aggperf3_diagnostics_tape", 0.0) + behavior_elapsed
                )

            pre_scalar_arrays = None
            if bool(getattr(self.sim_cfg, "cxxrewrite_resident_generation_scalar_env_inputs_enable", False)):
                t_pre_scalar0 = time.perf_counter()
                pre_scalar_arrays = cxxrewrite4_prepare_resident_generation_env_scalar_inputs(
                    self,
                    alive=alive,
                    agent_ids_np=agent_ids_np,
                    stage="resident_generation_pre_env",
                )
                buckets["resident_generation_pre_env_scalar_inputs"] = float(
                    buckets.get("resident_generation_pre_env_scalar_inputs", 0.0)
                    + (time.perf_counter() - t_pre_scalar0)
                )

            t_aggperf3_env_apply0 = time.perf_counter() if aggperf3_timing else 0.0
            deaths_this_gen, gen_peak_injury = self._apply_env_tick(
                alive,
                agent_ids_np,
                actions_np,
                buckets,
                deaths_this_gen,
                gen_peak_injury,
                pre_scalar_arrays=pre_scalar_arrays,
            )
            deaths_this_gen = self._apply_wdl_delayed_hazard_damage(deaths_this_gen)
            deaths_this_gen = self._apply_mem_max_delayed_choice_consequence(deaths_this_gen)
            charge_family_head_residual_cost(self, alive, generation=int(gen), tick=int(_tick))
            if aggperf3_timing:
                buckets["aggperf3_env_apply"] = float(
                    buckets.get("aggperf3_env_apply", 0.0) + (time.perf_counter() - t_aggperf3_env_apply0)
                )
            cxx23_handled_post = False
            try:
                cxx23 = getattr(self, "cxxcore23_live_direct_compact_diagnostics", None)
                if cxx23 is not None:
                    cxx23_handled_post = bool(
                        cxx23.record_post_env(
                            self,
                            generation=int(gen),
                            tick=int(tick),
                            memory_instrumentation=getattr(self, "memory_action_window_instrumentation", None),
                            prediction_instrumentation=getattr(
                                self,
                                "prediction_demand_action_context_instrumentation",
                                None,
                            ),
                        )
                    )
            except Exception:
                raise
            try:
                maw = getattr(self, "memory_action_window_instrumentation", None)
                if maw is not None and not cxx23_handled_post:
                    maw.record_post_env(self, generation=int(gen), tick=int(tick))
            except Exception:
                raise
            try:
                mm4 = getattr(self, "mem_max_live_level4_instrumentation", None)
                if mm4 is not None:
                    mm4.record_post_env(self, generation=int(gen), tick=int(tick))
            except Exception:
                raise

            t_aggperf3_internal0 = time.perf_counter() if aggperf3_timing else 0.0
            self._update_identity_drift(alive)

            t_drv0 = time.perf_counter()
            self._update_internal_drives(alive, actions_np, tick_index=int(getattr(self, "_pred_tick", 0)), buckets=buckets)
            buckets["drive_update_ms"] = buckets.get("drive_update_ms", 0.0) + (time.perf_counter() - t_drv0)

            t_chem0 = time.perf_counter()
            self._update_chemistry_state(alive, tick_index=int(getattr(self, "_pred_tick", 0)), buckets=buckets)
            buckets["chemistry_update_ms"] = buckets.get("chemistry_update_ms", 0.0) + (time.perf_counter() - t_chem0)

            t_maint0 = time.perf_counter()
            self._update_maintenance_state(
                alive,
                actions_np,
                tick_index=int(getattr(self, "_pred_tick", 0)),
                buckets=buckets,
            )
            buckets["maintenance_update_ms"] = buckets.get("maintenance_update_ms", 0.0) + (time.perf_counter() - t_maint0)

            deaths_this_gen = self._apply_rest_and_recovery(
                alive,
                actions_np,
                rest_streaks,
                injury_recover_mult,
                deaths_this_gen,
                buckets=buckets,
            )
            deaths_this_gen, gen_peak_injury = self._handle_attacks(alive, actions_np, deaths_this_gen, gen_peak_injury)
            if aggperf3_timing:
                buckets["aggperf3_internal_state_update"] = float(
                    buckets.get("aggperf3_internal_state_update", 0.0)
                    + (time.perf_counter() - t_aggperf3_internal0)
                )

            try:
                self._tick = int(tick)
                self.gen = int(gen)
            except Exception:
                pass
            try:
                _cxxrewrite4_birth_ids_before = set(int(getattr(a, "agent_id")) for a in self.pop.alive_agents())
            except Exception:
                _cxxrewrite4_birth_ids_before = set()
            _cxxrewrite4_successful_births_before = int(successful_births)
            _cxxrewrite4_failed_births_before = int(failed_births)
            successful_births, failed_births = self._handle_births(
                alive_for_births,
                successful_births,
                failed_births,
                buckets,
            )
            try:
                if int(successful_births) != int(_cxxrewrite4_successful_births_before) or int(failed_births) != int(_cxxrewrite4_failed_births_before):
                    _alive_after_births = list(self.pop.alive_agents())
                    _after_by_id = {int(getattr(a, "agent_id")): a for a in _alive_after_births}
                    _newborn_ids = sorted(set(_after_by_id.keys()) - set(_cxxrewrite4_birth_ids_before))
                    _sync_agents = [a for a in alive_for_births if bool(getattr(a, "alive", False))]
                    _sync_agents.extend(_after_by_id[int(aid)] for aid in _newborn_ids if int(aid) in _after_by_id)
                    cxxrewrite4_sync_agents_scalar_state(self, _sync_agents, stage="resident_generation_reproduction_births")
            except Exception:
                if cxxrewrite4_resident_scalar_sync_fail_closed(self.sim_cfg):
                    raise

            self._pred_tick = int(getattr(self, "_pred_tick", 0)) + 1
            maybe_capture_tick_native_science_sample(
                self,
                generation=int(gen),
                tick=int(tick),
            )

        self._cxxrewrite_resident_generation_record_fast(
            ticks_ran=int(ticks_ran),
            elapsed_s=float(time.perf_counter() - started),
        )
        try:
            cxxrewrite4_flush_summary(self)
        except Exception:
            if cxxrewrite4_resident_scalar_sync_fail_closed(self.sim_cfg):
                raise
        return deaths_this_gen, successful_births, failed_births, gen_peak_injury, ticks_ran

    def _run_ticks(
        self,
        gen: int,
        buckets: Dict[str, float],
        injury_recover_mult: float,
        deaths_this_gen: int,
        successful_births: int,
        failed_births: int,
        gen_peak_injury: float,
        rest_streaks: Dict[int, int],
    ) -> Tuple[int, int, int, float, int]:
        """
        Run the per-generation tick loop.

        Returns:
            (deaths_this_gen, successful_births, failed_births, gen_peak_injury, ticks_ran)
        """
        if self._cxxbackbone2_live_composition_requested():
            result = self._cxxbackbone2_live_composition_attempt(gen=int(gen), buckets=buckets)
            if bool(result.get("executed", False)):
                raise RuntimeError("CXXBACKBONE2 native generation execution returned without a Python sync adapter")
            if bool(getattr(self.sim_cfg, "cxxbackbone2_live_composition_fail_closed", True)):
                blockers = ",".join(str(reason) for reason in result.get("blockers", []))
                raise RuntimeError("CXXBACKBONE2 live composition unsupported for current run: " + blockers)

        if self._cxxwave1g_full_cpp_backbone_requested():
            payload = self._cxxwave1g_full_cpp_backbone_attempt(gen=int(gen))
            result = payload.get("generation_result", {})
            if bool(result.get("executed", False)):
                raise RuntimeError("CXXWAVE1G native generation execution returned without a Python sync adapter")
            if bool(getattr(self.sim_cfg, "cxxwave1g_full_cpp_backbone_fail_closed", True)):
                blockers = ",".join(str(reason) for reason in result.get("blockers", []))
                raise RuntimeError("CXXWAVE1G full C++ backbone unsupported for current run: " + blockers)

        if self._cxxrewrite_resident_generation_requested():
            reasons = self._cxxrewrite_resident_generation_unsupported_reasons()
            if not reasons:
                return self._run_ticks_resident_generation_fast(
                    gen=gen,
                    buckets=buckets,
                    injury_recover_mult=injury_recover_mult,
                    deaths_this_gen=deaths_this_gen,
                    successful_births=successful_births,
                    failed_births=failed_births,
                    gen_peak_injury=gen_peak_injury,
                    rest_streaks=rest_streaks,
                )
            self._cxxrewrite_resident_generation_record_fallback(reasons)
            if bool(getattr(self.sim_cfg, "cxxrewrite_resident_generation_runner_fail_closed", False)):
                raise RuntimeError(
                    "CXXREWRITE resident generation runner unsupported for current run: "
                    + ",".join(str(reason) for reason in reasons)
                )

        ticks_ran = 0

        # Phase 3 determinism localization: optional per-tick per-agent state detail dump.
        # Used to pinpoint first continuous-state drift tick before GRU/RNG divergence.
        def _eiel_try_get_env_array(attr_names):
            for _n in attr_names:
                try:
                    _v = getattr(self.env, _n)
                except Exception:
                    _v = None
                if _v is not None:
                    return _v
            return None

        def _eiel_quantize_scalar(x, scale=1000):
            try:
                return int(round(float(x) * float(scale)))
            except Exception:
                try:
                    return int(x)
                except Exception:
                    return 0

        def _eiel_build_agent_state_detail(alive_agents):
            cap = int(getattr(self.sim_cfg, 'population_cap', 0) or 0)
            env_energy = _eiel_try_get_env_array(['agent_energy', 'energy', 'energies'])
            env_injury = _eiel_try_get_env_array(['agent_injury', 'injury', 'injuries'])
            env_x = _eiel_try_get_env_array(['agent_x', 'x', 'xs'])
            env_y = _eiel_try_get_env_array(['agent_y', 'y', 'ys'])
            out = []
            for a in alive_agents:
                try:
                    aid = int(a.agent_id)
                except Exception:
                    continue
                if aid < 0 or (cap and aid >= cap):
                    continue
                e_q = _eiel_quantize_scalar(env_energy[aid]) if env_energy is not None else 0
                inj_q = _eiel_quantize_scalar(env_injury[aid]) if env_injury is not None else 0
                x_i = int(env_x[aid]) if env_x is not None else 0
                y_i = int(env_y[aid]) if env_y is not None else 0
                out.append((aid, e_q, inj_q, x_i, y_i))
            out.sort(key=lambda t: t[0])
            return out


        # Initialize per-generation tick state (counters + delayed prediction
        # clock). A partial-generation checkpoint continuation must preserve
        # cumulative per-generation counters from the restored checkpoint.
        resume_start_tick = int(getattr(self, "_eiel_resume_start_tick", 0) or 0)
        if resume_start_tick <= 0:
            self._init_generation_tick_state(self.pop.alive_agents())
        else:
            self._pred_tick = int(resume_start_tick)

        from native_runtime.safe_default import native_generation_requested  # noqa: WPS433

        native_safe_default_requested = bool(native_generation_requested(self))
        sampling_enabled = science_sampling_enabled(self.sim_cfg)

        if (
            bool(getattr(self.sim_cfg, "runnercore1_array_tick_authority_enable", False))
            or bool(getattr(self.sim_cfg, "runnercore1_candidate_enable", False))
            or bool(getattr(self.sim_cfg, "runnercore1_authoritative_enable", False))
            or bool(getattr(self.sim_cfg, "runnercore1_native_enable", False))
            or bool(getattr(self.sim_cfg, "runnercore2_churn_slot_lifecycle_enable", False))
            or bool(getattr(self.sim_cfg, "runnercore2_candidate_enable", False))
            or bool(getattr(self.sim_cfg, "runnercore2_authoritative_enable", False))
            or bool(getattr(self.sim_cfg, "runnercore2_native_enable", False))
            or bool(getattr(self.sim_cfg, "runnercore3_profile_enable", False))
            or bool(getattr(self.sim_cfg, "runnercore3_candidate_enable", False))
            or bool(getattr(self.sim_cfg, "runnercore3_native_enable", False))
            or bool(getattr(self.sim_cfg, "runnercore4_profile_enable", False))
            or bool(getattr(self.sim_cfg, "runnercore4_candidate_enable", False))
            or bool(getattr(self.sim_cfg, "runnercore4_native_enable", False))
            or bool(getattr(self.sim_cfg, "runnercore4_structural_payload_cache_enable", False))
            or bool(getattr(self.sim_cfg, "runnercore4_changed_parent_import_enable", False))
            or native_safe_default_requested
        ):
            if sampling_enabled:
                raise RuntimeError(
                    "TN-2 scientific sampling is not qualified for an array/native "
                    "tick authority path; use the reference runner or disable sampling"
                )
            from simulation.runnercore.tick_loop import run_array_tick_authority  # noqa: WPS433

            return run_array_tick_authority(
                self,
                gen=int(gen),
                buckets=buckets,
                injury_recover_mult=float(injury_recover_mult),
                deaths_this_gen=int(deaths_this_gen),
                successful_births=int(successful_births),
                failed_births=int(failed_births),
                gen_peak_injury=float(gen_peak_injury),
                rest_streaks=rest_streaks,
            )

        # -----------------------------
        # Optional torch.profiler
        # -----------------------------
        profiler_enabled = bool(getattr(self.sim_cfg, "enable_profiler", False))
        prof = None

        if profiler_enabled:
            import torch
            from torch.profiler import ProfilerActivity, profile, schedule

            run_dir = Path(getattr(self, "run_dir", "runs"))
            prof_dir = run_dir / "profiler"
            prof_dir.mkdir(parents=True, exist_ok=True)

            wait = int(getattr(self.sim_cfg, "profiler_wait", 2))
            warmup = int(getattr(self.sim_cfg, "profiler_warmup", 2))
            active = int(getattr(self.sim_cfg, "profiler_active", 5))
            repeat = int(getattr(self.sim_cfg, "profiler_repeat", 1))

            record_shapes = bool(getattr(self.sim_cfg, "profiler_record_shapes", True))
            profile_memory = bool(getattr(self.sim_cfg, "profiler_profile_memory", True))
            with_stack = bool(getattr(self.sim_cfg, "profiler_with_stack", True))

            prof = profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                schedule=schedule(wait=wait, warmup=warmup, active=active, repeat=repeat),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(str(prof_dir)),
                record_shapes=record_shapes,
                profile_memory=profile_memory,
                with_stack=with_stack,
            )
            prof.__enter__()

        # Cache env_alive array for fast membership checks
        env_alive = getattr(self.env, "agent_alive", None)
        if env_alive is None:
            raise AttributeError("env has no 'agent_alive' array; runner_ticks_loop requires array-based env state")
        aggperf3_timing = bool(getattr(self.sim_cfg, "perf_aggperf3_timing_buckets_enable", False))


        # ------------------------------------------------------------
        # Phase 3 determinism guard: env.spawn_food / spatial_droughts_step
        # MUST NOT mutate env.agent_alive (agent membership).
        #
        # If any bits flip here, it can "resurrect" a dead agent slot between ticks.
        # We detect and revert such mutations to keep membership stable.
        #
        # Guard behavior
        # --------------
        # - Enabled by default when running Phase 3 investigations.
        # - Compares BEFORE vs AFTER using the *current* env.agent_alive reference
        #   (handles accidental reassignment of the array inside the env).
        # - If mutation is detected, attempts to revert and emits a small debug log
        #   into: <run_dir>/debug/env_alive_guard.log
        #
        # Env:
        #   EIEL_GUARD_ENV_ALIVE=1 (default on)
        #   EIEL_GUARD_ENV_ALIVE_TICKS="60-70,..." (optional filter)
        # ------------------------------------------------------------
        _eiel_guard_alive_on = (os.getenv("EIEL_GUARD_ENV_ALIVE", "1") or "1").strip() in ("1", "true", "TRUE", "yes", "YES")

        def _eiel_tick_in_guard_filter(ti: int) -> bool:
            ticks_s = (os.getenv("EIEL_GUARD_ENV_ALIVE_TICKS", "") or "").strip()
            if not ticks_s:
                return True
            try:
                for part in str(ticks_s).split(","):
                    p = str(part).strip()
                    if not p:
                        continue
                    if "-" in p:
                        a_s, b_s = p.split("-", 1)
                        a = int(a_s.strip())
                        b = int(b_s.strip())
                        lo, hi = (a, b) if a <= b else (b, a)
                        if lo <= ti <= hi:
                            return True
                    else:
                        if int(p) == int(ti):
                            return True
            except Exception:
                # Fail-open: better to guard too much than not at all in investigations.
                return True
            return False

        def _eiel_env_alive_guard_log(line: str) -> None:
            try:
                run_dir = getattr(self, "run_dir", None) or getattr(self, "run_dir_path", None) or ""
                if not run_dir:
                    return
                out_dir = os.path.join(str(run_dir), "debug")
                os.makedirs(out_dir, exist_ok=True)
                out_path = os.path.join(out_dir, "env_alive_guard.log")
                with open(out_path, "a", encoding="utf-8") as _f:
                    _f.write(line.rstrip("\n") + "\n")
            except Exception:
                return

        def _eiel_guard_env_alive(kind: str, tick_i: int, fn):
            # NOTE: must handle env.agent_alive being reassigned inside env methods.
            nonlocal env_alive
            if not _eiel_guard_alive_on or not _eiel_tick_in_guard_filter(int(tick_i)):
                return fn()

            _env_alive_obj_before = None
            _before = None
            try:
                _env_alive_obj_before = getattr(self.env, "agent_alive", None)
                if _env_alive_obj_before is not None:
                    _before = np.asarray(_env_alive_obj_before, dtype=bool).copy()
            except Exception:
                _env_alive_obj_before = None
                _before = None

            out = fn()

            try:
                _env_alive_obj_after = getattr(self.env, "agent_alive", None)
                if _env_alive_obj_after is None or _before is None:
                    return out

                # If env swapped the array object, update cached reference used by the runner.
                if _env_alive_obj_after is not env_alive:
                    env_alive = _env_alive_obj_after
                    try:
                        _eiel_env_alive_guard_log(
                            f"[env_alive_guard] gen={int(gen)} tick={int(tick_i)} kind={kind} NOTE: env.agent_alive reassigned"
                        )
                    except Exception:
                        pass

                _after = np.asarray(_env_alive_obj_after, dtype=bool)
                if _after.shape != _before.shape:
                    _eiel_env_alive_guard_log(
                        f"[env_alive_guard] gen={int(gen)} tick={int(tick_i)} kind={kind} WARN: shape changed before={getattr(_before,'shape',None)} after={getattr(_after,'shape',None)}"
                    )
                    return out

                if np.any(_after != _before):
                    flipped = np.where(_after != _before)[0].astype(np.int64).tolist()
                    # Revert illegal mutation (best-effort)
                    try:
                        _env_alive_obj_after[:] = _before
                    except Exception:
                        try:
                            env_alive[:] = _before
                        except Exception:
                            pass

                    _eiel_env_alive_guard_log(
                        f"[env_alive_guard] gen={int(gen)} tick={int(tick_i)} kind={kind} REVERT flipped_n={len(flipped)} flipped_head={flipped[:32]}{'...' if len(flipped)>32 else ''}"
                    )
                    try:
                        from simulation.membership_trace import note_write  # noqa: WPS433
                        note_write(kind=f"env_alive_guard.revert.{kind}", agent_id=-1, detail=f"tick={int(tick_i)} flipped={flipped[:32]}{'...' if len(flipped)>32 else ''}")
                    except Exception:
                        pass
            except Exception:
                pass

            return out

        # ------------------------------------------------------------
        # Phase 3 determinism: env vs pop membership parity probe.
        #
        # Purpose
        # -------
        # We have a persistent "ghost alive slot" appearing at gen0/tick61
        # on resume only. This probe answers: is the mismatch in env.agent_alive
        # or pop.agent_alive (or both)?
        #
        # Env:
        #   EIEL_MEMBERSHIP_PARITY=1 (default on)
        #   EIEL_MEMBERSHIP_PARITY_TICKS="60-70,..." (optional filter)
        #   EIEL_MEMBERSHIP_STRICT_SYNC=0/1 (default off): if on, forces
        #       env/pop masks to their intersection when a mismatch is detected.
        # ------------------------------------------------------------
        _eiel_parity_on = (os.getenv("EIEL_MEMBERSHIP_PARITY", "1") or "1").strip() in ("1", "true", "TRUE", "yes", "YES")
        _eiel_parity_strict_sync = (os.getenv("EIEL_MEMBERSHIP_STRICT_SYNC", "0") or "0").strip() in ("1", "true", "TRUE", "yes", "YES")

        def _eiel_tick_in_parity_filter(ti: int) -> bool:
            ticks_s = (os.getenv("EIEL_MEMBERSHIP_PARITY_TICKS", "") or "").strip()
            if not ticks_s:
                return True
            try:
                for part in str(ticks_s).split(","):
                    p = str(part).strip()
                    if not p:
                        continue
                    if "-" in p:
                        a_s, b_s = p.split("-", 1)
                        a = int(a_s.strip())
                        b = int(b_s.strip())
                        lo, hi = (a, b) if a <= b else (b, a)
                        if lo <= ti <= hi:
                            return True
                    else:
                        if int(p) == int(ti):
                            return True
            except Exception:
                return True
            return False

        def _eiel_dump_membership_parity(*, tick_i: int) -> None:
            if (not _eiel_parity_on) or (not _eiel_tick_in_parity_filter(int(tick_i))):
                return
            try:
                # Gate on Phase-3 dump flag to avoid surprising overhead in normal runs.
                if os.environ.get("EIEL_STATE_DUMP", "0").strip() != "1":
                    return

                env_alive_now = np.asarray(env_alive, dtype=np.bool_)
                pop_alive_arr = getattr(self.pop, "agent_alive", None)

                if pop_alive_arr is not None:
                    pop_alive_now = np.asarray(pop_alive_arr, dtype=np.bool_)
                else:
                    cap2 = int(env_alive_now.shape[0])
                    pop_alive_now = np.zeros((cap2,), dtype=np.bool_)
                    for _a in self.pop.alive_agents():
                        try:
                            _aid = int(getattr(_a, "agent_id"))
                        except Exception:
                            continue
                        if 0 <= _aid < cap2:
                            pop_alive_now[_aid] = True

                if env_alive_now.shape != pop_alive_now.shape:
                    return

                if np.array_equal(env_alive_now, pop_alive_now):
                    return

                env_ids = np.flatnonzero(env_alive_now).astype(np.int64).tolist()
                pop_ids = np.flatnonzero(pop_alive_now).astype(np.int64).tolist()

                env_set = set(env_ids)
                pop_set = set(pop_ids)

                env_not_pop = sorted(list(env_set - pop_set))
                pop_not_env = sorted(list(pop_set - env_set))

                from engine.simulation.state_dump import maybe_dump_state  # noqa: WPS433

                maybe_dump_state(
                    self,
                    generation_index=int(gen),
                    tag=f"membership_parity_pre_env_t{int(tick_i):03d}",
                    extra={
                        "tick": int(tick_i),
                        "stage": "pre_env",
                        "env_alive_n": int(len(env_ids)),
                        "pop_alive_n": int(len(pop_ids)),
                        "env_not_pop_n": int(len(env_not_pop)),
                        "pop_not_env_n": int(len(pop_not_env)),
                        "env_not_pop_head": env_not_pop[:128],
                        "pop_not_env_head": pop_not_env[:128],
                    },
                )

                if _eiel_parity_strict_sync:
                    inter = np.logical_and(env_alive_now, pop_alive_now)
                    # Sync env
                    try:
                        env_alive[:] = inter
                    except Exception:
                        pass
                    # Sync pop array if it exists (otherwise we only sync env to avoid
                    # mutating agent objects in-place here).
                    if pop_alive_arr is not None:
                        try:
                            pop_alive_arr[:] = inter
                        except Exception:
                            pass
            except Exception:
                return
        try:
            # --------------------------------------------------------------
            # Phase 3 determinism: optional per-tick RNG probe (no RNG draws)
            # --------------------------------------------------------------
            # Use this to localize the first tick where fresh vs resume diverge.
            # Gated by EIEL_DETERMINISM_PROBE=1 and per-tick env vars:
            #   EIEL_DETERMINISM_PROBE_TICK_MIN (int, default: -1 = disabled)
            #   EIEL_DETERMINISM_PROBE_TICK_MAX (int, default: -1 = disabled)
            # If either min/max is -1, tick probing is disabled.
            __eiel_tick_probe_min = -1
            __eiel_tick_probe_max = -1
            try:
                __eiel_tick_probe_min = int(os.environ.get('EIEL_DETERMINISM_PROBE_TICK_MIN', '-1') or '-1')
                __eiel_tick_probe_max = int(os.environ.get('EIEL_DETERMINISM_PROBE_TICK_MAX', '-1') or '-1')
            except Exception:
                __eiel_tick_probe_min = -1
                __eiel_tick_probe_max = -1

            # If probe is enabled but no tick window is specified, default to the same
            # window as the tick-trace (if provided), else fail-open to full generation.
            if (os.environ.get('EIEL_DETERMINISM_PROBE', '0').strip() == '1') and (__eiel_tick_probe_min < 0 or __eiel_tick_probe_max < 0):
                _ticks_spec = (os.environ.get('EIEL_DEBUG_TICK_TRACE_TICKS', '') or '').strip()
                _lo = None
                _hi = None
                try:
                    for _part in _ticks_spec.split(','):
                        _p = _part.strip()
                        if not _p:
                            continue
                        if '-' in _p:
                            _a_s, _b_s = _p.split('-', 1)
                            _a = int(_a_s.strip())
                            _b = int(_b_s.strip())
                            _l, _h = (_a, _b) if _a <= _b else (_b, _a)
                        else:
                            _l = _h = int(_p)
                        _lo = _l if _lo is None else min(_lo, _l)
                        _hi = _h if _hi is None else max(_hi, _h)
                except Exception:
                    _lo = None
                    _hi = None
                if _lo is None or _hi is None:
                    __eiel_tick_probe_min = 0
                    __eiel_tick_probe_max = int(getattr(self.sim_cfg, 'steps_per_generation', 0) or 0) - 1
                else:
                    __eiel_tick_probe_min = int(_lo)
                    __eiel_tick_probe_max = int(_hi)

            def __eiel_maybe_tick_probe(_stage: str, _tick_i: int) -> None:
                try:
                    if os.environ.get('EIEL_DETERMINISM_PROBE', '0').strip() != '1':
                        return
                    if __eiel_tick_probe_min >= 0 and __eiel_tick_probe_max >= 0:
                        if int(_tick_i) < int(__eiel_tick_probe_min) or int(_tick_i) > int(__eiel_tick_probe_max):
                            return
                    from simulation.determinism_probe import maybe_write_probe as _maybe_write_probe  # noqa: WPS433
                    _maybe_write_probe(self, generation_index=int(gen), tag=f'tick{int(_tick_i):03d}.{str(_stage)}')
                except Exception:
                    return
            _tick_start = 0
            _tick_end = int(self.sim_cfg.steps_per_generation)
            if bool(getattr(self, "_tick_native_execution_route", False)):
                _tick_start = int(getattr(self, "_tick_native_start_next_tick", 0) or 0)
                _tick_end = int(
                    getattr(self, "_tick_native_end_next_tick", _tick_end) or _tick_end
                )
                if not 0 <= _tick_start <= _tick_end <= int(self.sim_cfg.steps_per_generation):
                    raise RuntimeError("tick-native tick span is outside total_ticks")
            for _tick in range(_tick_start, _tick_end):
                try:
                    if hasattr(self, "_dev_expr_cache") and isinstance(self._dev_expr_cache, dict):
                        try:
                            _diag = getattr(self, "env_hotpath_expression_diagnostics", None)
                            if _diag is not None:
                                _diag.record_dev_cache_clear(
                                    generation=int(gen),
                                    tick=int(_tick),
                                    entries=int(len(self._dev_expr_cache)),
                                )
                        except Exception:
                            pass
                        self._dev_expr_cache.clear()
                        if hasattr(self, "_dev_effective_scalar_cache") and isinstance(self._dev_effective_scalar_cache, dict):
                            self._dev_effective_scalar_cache.clear()
                except Exception:
                    pass

                # FDP v3+: expose current tick/gen to determinism probes (instrumentation only).
                try:
                    self._eiel_fdp_cur_tick = int(_tick)
                    self._eiel_fdp_cur_gen = int(gen)
                except Exception:
                    pass

                # FDP v3: align evolution_py trace counter at gen0/tick0.
                try:
                    if int(gen) == 0 and int(_tick) == 0:
                        _rm0 = getattr(self, 'rng_manager', None)
                        if _rm0 is not None and hasattr(_rm0, '_reset_evopy_trace'):
                            # Boundary reset has priority: do not reset counters again at gen0/tick0.
                            if getattr(_rm0, '_evopy_epoch', '') != 'boundary':
                                _rm0._reset_evopy_trace(epoch='gen0_tick0')
                            else:
                                # Keep counters intact but advance epoch label for clarity.
                                if hasattr(_rm0, '_set_evopy_epoch_only'):
                                    _rm0._set_evopy_epoch_only(epoch='gen0_tick0')
                        if _rm0 is not None and hasattr(_rm0, '_reset_evopy_ledger'):
                            if getattr(_rm0, '_evopy_epoch', '') != 'boundary':
                                _rm0._reset_evopy_ledger(epoch='gen0_tick0')
                            else:
                                if hasattr(_rm0, '_set_evopy_epoch_only'):
                                    _rm0._set_evopy_epoch_only(epoch='gen0_tick0')
                        # FDP v3: dump evolution_py state at gen0/tick0 after any reset decision.
                        try:
                            if (
                                os.environ.get("EIEL_EVOPY_LEDGER", "0").strip() == "1"
                                and _rm0 is not None
                                and hasattr(_rm0, 'dump_evopy_state_json')
                            ):
                                _run_dir0 = getattr(self, 'run_dir', '') or ''
                                if _run_dir0:
                                    _out0 = os.path.join(str(_run_dir0), 'debug', 'evopy_state__gen000000__tick000__tick0.json')
                                    _rm0.dump_evopy_state_json(path=_out0, stage='gen0_tick0')
                        except Exception:
                            pass
                except Exception:
                    pass
                # FDP v3: set RNG trace context for attribution (debug-only).
                try:
                    _rm = getattr(self, 'rng_manager', None)
                    if _rm is not None and hasattr(_rm, '_set_trace_ctx'):
                        _rm._set_trace_ctx(gen=int(gen), tick=int(_tick), stage='tick_start')
                except Exception:
                    pass

                # Phase 3 tick probe: snapshot RNG states at tick boundary (no RNG draws).
                __eiel_maybe_tick_probe('start', int(_tick))
                # FDP v3: force a last-tick boundary probe even when tick filters exclude it.
                try:
                    if os.environ.get("EIEL_DETERMINISM_PROBE", "0").strip() == "1":
                        if int(_tick) == int(self.sim_cfg.steps_per_generation) - 1:
                            from simulation.determinism_probe import maybe_write_probe as _maybe_write_probe  # noqa: WPS433
                            _maybe_write_probe(self, generation_index=int(gen), tag="tickLAST.start")
                except Exception:
                    pass
                # FDP v3: force selected tick-start probes even when tick filters exclude them.
                try:
                    if os.environ.get("EIEL_DETERMINISM_PROBE", "0").strip() == "1":
                        if int(_tick) in (110, 115, 116, 117, 118, 119, 120):
                            from simulation.determinism_probe import maybe_write_probe as _maybe_write_probe  # noqa: WPS433
                            _maybe_write_probe(self, generation_index=int(gen), tag=f"tick{int(_tick):03d}.forced_start")
                except Exception:
                    pass



                # FDP v3+: targeted bank probes to localize late-gen drift (gen3 several ticks).
                try:
                    if os.environ.get("EIEL_DETERMINISM_PROBE", "0").strip() == "1":
                        if int(gen) == 3 and (int(_tick) in (0, 50, 115, 116) or (100 <= int(_tick) <= 110)):
                            if hasattr(self, '_eiel_emit_policy_bank_probe'):
                                self._eiel_emit_policy_bank_probe(tag=f"tick{int(_tick):03d}.bank_pre_forward", generation_index=int(gen), tick=int(_tick))
                except Exception:
                    pass

                # Forensic V4 (A,C): emit bank digests + agent attr signature at tick000.start for gen1
                try:
                    if int(gen) == 1 and int(_tick) == 0:
                        if hasattr(self, '_eiel_emit_policy_bank_probe'):
                            self._eiel_emit_policy_bank_probe(tag='tick000.start', generation_index=int(gen), tick=int(_tick))
                        try:
                            _alive = list(getattr(self, 'pop').alive_agents())
                        except Exception:
                            _alive = []
                        if getattr(self, 'run_dir', None):
                            emit_agent_attr_signature(str(self.run_dir), tag='tick000.start', generation_index=int(gen), tick=int(_tick), agents=_alive)
                except Exception:
                    pass
                # Forensic ledger dump point (systematic RNG accounting).
                # Dump at a deterministic tick boundary so fresh vs resume can be diffed.
                try:
                    if (
                        os.environ.get("EIEL_EVOPY_LEDGER", "0").strip() == "1"
                        and int(gen) == 0
                        and int(_tick) in (54,)
                    ):
                        _rm = getattr(self, 'rng_manager', None)
                        if _rm is not None and hasattr(_rm, 'dump_evopy_ledger'):
                            _run_dir = getattr(self, 'run_dir', '') or ''
                            if _run_dir:
                                _out = os.path.join(str(_run_dir), 'debug', f'evopy_ledger__gen{int(gen):06d}__tick{int(_tick):03d}.json')
                                _rm.dump_evopy_ledger(path=_out)
                        # FDP v3: also dump evolution_py RNG state digest at the same tick boundary.
                        try:
                            _rm = getattr(self, 'rng_manager', None)
                            if _rm is not None and hasattr(_rm, 'dump_evopy_state_json'):
                                _run_dir = getattr(self, 'run_dir', '') or ''
                                if _run_dir:
                                    _out = os.path.join(str(_run_dir), 'debug', f'evopy_state__gen{int(gen):06d}__tick{int(_tick):03d}__tick_boundary.json')
                                    _rm.dump_evopy_state_json(path=_out, stage='tick_boundary')
                        except Exception:
                            pass
                except Exception:
                    pass
                # If Ctrl-C was requested, stop running ticks ASAP so we can
                # reach the end-of-generation "safe boundary" and write the
                # interrupt checkpoint before outer launchers (e.g. nox) decide
                # to terminate the process.
                if bool(getattr(self, "_stop_requested", False)):
                    if not bool(getattr(self, "_stop_tick_notice_printed", False)):
                        try:
                            print("[interrupt] stopping ticks early to reach checkpoint boundary...", flush=True)
                        except Exception:
                            pass
                        setattr(self, "_stop_tick_notice_printed", True)
                    break
                # Tier 2: spatial droughts (time-varying regional multipliers)
                # Absolute step index since generation 0, for reproducible scheduling.
                self._run_spatial_droughts_step_with_budget(
                    gen=int(gen),
                    tick=int(_tick),
                    guard_call=lambda abs_step: _eiel_guard_env_alive(
                        "spatial_droughts_step",
                        int(_tick),
                        lambda: self.env.spatial_droughts_step(abs_step),
                    ),
                )

                # Patch World / Patch-Jungle V0 Step 5: deterministic trace
                # decay/emission, then food regrowth and binary projection. RNG-free.
                t_aggperf3_env_context0 = time.perf_counter() if aggperf3_timing else 0.0
                if bool(getattr(self.env, "patch_world_enable", False)):
                    _patch_abs_step = int(gen) * int(self.sim_cfg.steps_per_generation) + int(_tick)
                    self.env.patch_world_step(_patch_abs_step)
                    if bool(getattr(self.env_cfg, "wdl_delayed_hazard_enable", False)):
                        self._wdl_delayed_hazard_tick_count_gen = int(
                            getattr(self, "_wdl_delayed_hazard_tick_count_gen", 0)
                        ) + 1
                        self._wdl_delayed_hazard_event_count_gen = int(
                            getattr(self, "_wdl_delayed_hazard_event_count_gen", 0)
                        ) + int(getattr(self.env, "wdl_delayed_hazard_event_count_last", 0))
                        cue_cells = int(getattr(self.env, "wdl_delayed_hazard_cue_cell_count_last", 0))
                        active_cells = int(getattr(self.env, "wdl_delayed_hazard_active_cell_count_last", 0))
                        try:
                            prior = np.asarray(getattr(self.env, "patch_delayed_hazard_prior_cue"), dtype=np.int8)
                            cue = np.asarray(getattr(self.env, "patch_delayed_hazard_cue"), dtype=np.float32)
                            active = np.asarray(getattr(self.env, "patch_delayed_hazard_active"), dtype=np.float32)
                            hidden_cells = int(np.sum((prior > 0) & (cue <= 0.0) & (active <= 0.0)))
                        except Exception:
                            hidden_cells = 0
                        self._wdl_delayed_hazard_cue_cell_count_sum_gen = float(
                            getattr(self, "_wdl_delayed_hazard_cue_cell_count_sum_gen", 0.0)
                        ) + float(cue_cells)
                        self._wdl_delayed_hazard_active_cell_count_sum_gen = float(
                            getattr(self, "_wdl_delayed_hazard_active_cell_count_sum_gen", 0.0)
                        ) + float(active_cells)
                        if cue_cells > 0:
                            self._path1_drh0_cue_visible_steps_gen = int(
                                getattr(self, "_path1_drh0_cue_visible_steps_gen", 0)
                            ) + 1
                        if hidden_cells > 0:
                            self._path1_drh0_hidden_window_steps_gen = int(
                                getattr(self, "_path1_drh0_hidden_window_steps_gen", 0)
                            ) + 1
                        if active_cells > 0:
                            self._path1_drh0_hazard_active_steps_gen = int(
                                getattr(self, "_path1_drh0_hazard_active_steps_gen", 0)
                            ) + 1
                        self._wdl_delayed_hazard_agent_cue_exposure_count_gen = int(
                            getattr(self, "_wdl_delayed_hazard_agent_cue_exposure_count_gen", 0)
                        ) + int(cue_cells)
                        self._wdl_delayed_hazard_cue_to_active_delay_sum_gen = float(
                            getattr(self, "_wdl_delayed_hazard_cue_to_active_delay_sum_gen", 0.0)
                        ) + float(getattr(self.env, "wdl_delayed_hazard_cue_to_active_delay_sum_last", 0.0))
                        self._wdl_delayed_hazard_cue_to_active_delay_count_gen = int(
                            getattr(self, "_wdl_delayed_hazard_cue_to_active_delay_count_gen", 0)
                        ) + int(getattr(self.env, "wdl_delayed_hazard_cue_to_active_delay_count_last", 0))
                        self._wdl_delayed_hazard_cue_without_active_count_gen = int(
                            getattr(self, "_wdl_delayed_hazard_cue_without_active_count_gen", 0)
                        ) + int(getattr(self.env, "wdl_delayed_hazard_cue_without_active_count_last", 0))
                        self._wdl_delayed_hazard_active_without_prior_cue_count_gen = int(
                            getattr(self, "_wdl_delayed_hazard_active_without_prior_cue_count_gen", 0)
                        ) + int(getattr(self.env, "wdl_delayed_hazard_active_without_prior_cue_count_last", 0))
                    if bool(getattr(self.env_cfg, "prediction_demand_enable", False)):
                        self._prediction_demand_tick_count_gen = int(
                            getattr(self, "_prediction_demand_tick_count_gen", 0)
                        ) + 1
                        self._prediction_demand_event_count_gen = int(
                            getattr(self, "_prediction_demand_event_count_gen", 0)
                        ) + int(getattr(self.env, "prediction_demand_event_count_last", 0))
                        self._prediction_demand_transition_count_gen = int(
                            getattr(self, "_prediction_demand_transition_count_gen", 0)
                        ) + int(getattr(self.env, "prediction_demand_transition_count_last", 0))
                        cue_cells = int(getattr(self.env, "prediction_demand_cue_cell_count_last", 0))
                        active_cells = int(getattr(self.env, "prediction_demand_active_cell_count_last", 0))
                        self._prediction_demand_cue_cell_count_sum_gen = float(
                            getattr(self, "_prediction_demand_cue_cell_count_sum_gen", 0.0)
                        ) + float(cue_cells)
                        self._prediction_demand_active_cell_count_sum_gen = float(
                            getattr(self, "_prediction_demand_active_cell_count_sum_gen", 0.0)
                        ) + float(active_cells)
                        self._prediction_demand_cue_ambiguity_sum_gen = float(
                            getattr(self, "_prediction_demand_cue_ambiguity_sum_gen", 0.0)
                        ) + float(getattr(self.env, "prediction_demand_cue_ambiguity_sum_last", 0.0))
                        self._prediction_demand_cue_ambiguity_count_gen = int(
                            getattr(self, "_prediction_demand_cue_ambiguity_count_gen", 0)
                        ) + int(getattr(self.env, "prediction_demand_cue_ambiguity_count_last", 0))
                        self._prediction_demand_cue_to_consequence_delay_sum_gen = float(
                            getattr(self, "_prediction_demand_cue_to_consequence_delay_sum_gen", 0.0)
                        ) + float(getattr(self.env, "prediction_demand_cue_to_consequence_delay_sum_last", 0.0))
                        self._prediction_demand_cue_to_consequence_delay_count_gen = int(
                            getattr(self, "_prediction_demand_cue_to_consequence_delay_count_gen", 0)
                        ) + int(getattr(self.env, "prediction_demand_cue_to_consequence_delay_count_last", 0))
                        self._prediction_demand_cue_without_consequence_count_gen = int(
                            getattr(self, "_prediction_demand_cue_without_consequence_count_gen", 0)
                        ) + int(getattr(self.env, "prediction_demand_cue_without_consequence_count_last", 0))
                        self._prediction_demand_consequence_without_prior_cue_count_gen = int(
                            getattr(self, "_prediction_demand_consequence_without_prior_cue_count_gen", 0)
                        ) + int(getattr(self.env, "prediction_demand_consequence_without_prior_cue_count_last", 0))
                        self._prediction_demand_resource_consequence_total_gen = float(
                            getattr(self, "_prediction_demand_resource_consequence_total_gen", 0.0)
                        ) + float(getattr(self.env, "prediction_demand_resource_consequence_total_last", 0.0))
                        try:
                            prior = np.asarray(getattr(self.env, "patch_prediction_demand_prior_cue"), dtype=np.int8)
                            active = np.asarray(getattr(self.env, "patch_prediction_demand_active"), dtype=np.float32)
                            ys = np.asarray(getattr(self.env, "agent_pos_y"), dtype=np.int32)
                            xs = np.asarray(getattr(self.env, "agent_pos_x"), dtype=np.int32)
                            alive_ids = [int(getattr(a, "agent_id")) for a in self.pop.alive_agents()]
                            pre_count = 0
                            exposure_count = 0
                            for aid in alive_ids:
                                y = int(ys[aid])
                                x = int(xs[aid])
                                if y < 0 or x < 0:
                                    continue
                                if int(prior[y, x]) > 0 and float(active[y, x]) == 0.0:
                                    pre_count += 1
                                if float(active[y, x]) != 0.0:
                                    exposure_count += 1
                            self._prediction_demand_pre_consequence_agent_context_count_gen = int(
                                getattr(self, "_prediction_demand_pre_consequence_agent_context_count_gen", 0)
                            ) + int(pre_count)
                            self._prediction_demand_consequence_agent_exposure_count_gen = int(
                                getattr(self, "_prediction_demand_consequence_agent_exposure_count_gen", 0)
                            ) + int(exposure_count)
                        except Exception:
                            pass
                    for _prefix, _enabled_name in (
                        ("delayed_resource_depletion", "delayed_resource_depletion_enable"),
                        ("delayed_crowding_waste", "delayed_crowding_waste_enable"),
                    ):
                        if not bool(getattr(self.env_cfg, _enabled_name, False)):
                            continue
                        setattr(
                            self,
                            f"_{_prefix}_tick_count_gen",
                            int(getattr(self, f"_{_prefix}_tick_count_gen", 0)) + 1,
                        )
                        for _field in ("event", "cue_without_active", "active_without_prior_cue", "consequence"):
                            setattr(
                                self,
                                f"_{_prefix}_{_field}_count_gen",
                                int(getattr(self, f"_{_prefix}_{_field}_count_gen", 0))
                                + int(getattr(self.env, f"{_prefix}_{_field}_count_last", 0)),
                            )
                        for _field in ("cue", "hidden", "active"):
                            setattr(
                                self,
                                f"_{_prefix}_{_field}_cell_count_sum_gen",
                                float(getattr(self, f"_{_prefix}_{_field}_cell_count_sum_gen", 0.0))
                                + float(getattr(self.env, f"{_prefix}_{_field}_cell_count_last", 0.0)),
                            )
                        setattr(
                            self,
                            f"_{_prefix}_consequence_total_gen",
                            float(getattr(self, f"_{_prefix}_consequence_total_gen", 0.0))
                            + float(getattr(self.env, f"{_prefix}_consequence_total_last", 0.0)),
                        )
                        setattr(
                            self,
                            f"_{_prefix}_cue_to_active_delay_sum_gen",
                            float(getattr(self, f"_{_prefix}_cue_to_active_delay_sum_gen", 0.0))
                            + float(getattr(self.env, f"{_prefix}_cue_to_active_delay_sum_last", 0.0)),
                        )
                        setattr(
                            self,
                            f"_{_prefix}_cue_to_active_delay_count_gen",
                            int(getattr(self, f"_{_prefix}_cue_to_active_delay_count_gen", 0))
                            + int(getattr(self.env, f"{_prefix}_cue_to_active_delay_count_last", 0)),
                        )
                    self._patch_food_regrowth_total_gen = float(
                        getattr(self, "_patch_food_regrowth_total_gen", 0.0)
                    ) + float(getattr(self.env, "patch_food_regrowth_total_last", 0.0))
                    self._patch_regrowth_base_total_gen = float(
                        getattr(self, "_patch_regrowth_base_total_gen", 0.0)
                    ) + float(getattr(self.env, "patch_regrowth_base_total_last", 0.0))
                    self._patch_regrowth_effective_total_gen = float(
                        getattr(self, "_patch_regrowth_effective_total_gen", 0.0)
                    ) + float(getattr(self.env, "patch_regrowth_effective_total_last", 0.0))
                    self._patch_regrowth_suppressed_total_gen = float(
                        getattr(self, "_patch_regrowth_suppressed_total_gen", 0.0)
                    ) + float(getattr(self.env, "patch_regrowth_suppressed_total_last", 0.0))
                    self._patch_regrowth_consequence_steps_gen = int(
                        getattr(self, "_patch_regrowth_consequence_steps_gen", 0)
                    ) + 1
                    self._patch_regrowth_multiplier_mean_sum_gen = float(
                        getattr(self, "_patch_regrowth_multiplier_mean_sum_gen", 0.0)
                    ) + float(getattr(self.env, "patch_regrowth_multiplier_mean_last", 1.0))
                    self._patch_regrowth_multiplier_min_gen = min(
                        float(getattr(self, "_patch_regrowth_multiplier_min_gen", 1.0)),
                        float(getattr(self.env, "patch_regrowth_multiplier_min_last", 1.0)),
                    )
                    self._patch_regrowth_multiplier_p10_sum_gen = float(
                        getattr(self, "_patch_regrowth_multiplier_p10_sum_gen", 0.0)
                    ) + float(getattr(self.env, "patch_regrowth_multiplier_p10_last", 1.0))
                    self._patch_regrowth_multiplier_p90_sum_gen = float(
                        getattr(self, "_patch_regrowth_multiplier_p90_sum_gen", 0.0)
                    ) + float(getattr(self.env, "patch_regrowth_multiplier_p90_last", 1.0))
                    self._patch_waste_regrowth_penalty_mean_sum_gen = float(
                        getattr(self, "_patch_waste_regrowth_penalty_mean_sum_gen", 0.0)
                    ) + float(getattr(self.env, "patch_waste_regrowth_penalty_mean_last", 0.0))
                    self._patch_waste_regrowth_penalty_max_gen = max(
                        float(getattr(self, "_patch_waste_regrowth_penalty_max_gen", 0.0)),
                        float(getattr(self.env, "patch_waste_regrowth_penalty_max_last", 0.0)),
                    )
                    self._patch_density_regrowth_penalty_mean_sum_gen = float(
                        getattr(self, "_patch_density_regrowth_penalty_mean_sum_gen", 0.0)
                    ) + float(getattr(self.env, "patch_density_regrowth_penalty_mean_last", 0.0))
                    self._patch_density_regrowth_penalty_max_gen = max(
                        float(getattr(self, "_patch_density_regrowth_penalty_max_gen", 0.0)),
                        float(getattr(self.env, "patch_density_regrowth_penalty_max_last", 0.0)),
                    )
                    self._patch_high_waste_regrowth_suppression_frac_sum_gen = float(
                        getattr(self, "_patch_high_waste_regrowth_suppression_frac_sum_gen", 0.0)
                    ) + float(getattr(self.env, "patch_high_waste_regrowth_suppression_frac_last", 0.0))
                    self._patch_high_density_regrowth_suppression_frac_sum_gen = float(
                        getattr(self, "_patch_high_density_regrowth_suppression_frac_sum_gen", 0.0)
                    ) + float(getattr(self.env, "patch_high_density_regrowth_suppression_frac_last", 0.0))
                    self._patch_recovery_positive_total_gen = float(
                        getattr(self, "_patch_recovery_positive_total_gen", 0.0)
                    ) + float(getattr(self.env, "patch_recovery_positive_total_last", 0.0))
                    self._patch_recovery_negative_total_gen = float(
                        getattr(self, "_patch_recovery_negative_total_gen", 0.0)
                    ) + float(getattr(self.env, "patch_recovery_negative_total_last", 0.0))
                    self._patch_recovery_decay_total_gen = float(
                        getattr(self, "_patch_recovery_decay_total_gen", 0.0)
                    ) + float(getattr(self.env, "patch_recovery_decay_total_last", 0.0))
                    self._jungle_recovery_regrowth_boost_total_gen = float(
                        getattr(self, "_jungle_recovery_regrowth_boost_total_gen", 0.0)
                    ) + float(getattr(self.env, "jungle_recovery_regrowth_boost_total_last", 0.0))
                    self._jungle_recovery_regrowth_boost_mean_sum_gen = float(
                        getattr(self, "_jungle_recovery_regrowth_boost_mean_sum_gen", 0.0)
                    ) + float(getattr(self.env, "jungle_recovery_regrowth_boost_mean_last", 0.0))
                    self._jungle_recovery_regrowth_boost_max_gen = max(
                        float(getattr(self, "_jungle_recovery_regrowth_boost_max_gen", 0.0)),
                        float(getattr(self.env, "jungle_recovery_regrowth_boost_max_last", 0.0)),
                    )
                    self._jungle_recovery_regrowth_boost_p90_sum_gen = float(
                        getattr(self, "_jungle_recovery_regrowth_boost_p90_sum_gen", 0.0)
                    ) + float(getattr(self.env, "jungle_recovery_regrowth_boost_p90_last", 0.0))
                    self._patch_regrowth_after_waste_total_gen = float(
                        getattr(self, "_patch_regrowth_after_waste_total_gen", 0.0)
                    ) + float(getattr(self.env, "patch_regrowth_after_waste_total_last", 0.0))
                    self._patch_regrowth_after_recovery_total_gen = float(
                        getattr(self, "_patch_regrowth_after_recovery_total_gen", 0.0)
                    ) + float(getattr(self.env, "patch_regrowth_after_recovery_total_last", 0.0))
                    self._patch_recovery_added_regrowth_total_gen = float(
                        getattr(self, "_patch_recovery_added_regrowth_total_gen", 0.0)
                    ) + float(getattr(self.env, "patch_recovery_added_regrowth_total_last", 0.0))
                    self._patch_scent_emitted_total_gen = float(
                        getattr(self, "_patch_scent_emitted_total_gen", 0.0)
                    ) + float(getattr(self.env, "patch_scent_emitted_total_last", 0.0))
                    self._patch_scent_decayed_total_gen = float(
                        getattr(self, "_patch_scent_decayed_total_gen", 0.0)
                    ) + float(getattr(self.env, "patch_scent_decayed_total_last", 0.0))
                    self._patch_waste_emitted_total_gen = float(
                        getattr(self, "_patch_waste_emitted_total_gen", 0.0)
                    ) + float(getattr(self.env, "patch_waste_emitted_total_last", 0.0))
                    self._patch_waste_decayed_total_gen = float(
                        getattr(self, "_patch_waste_decayed_total_gen", 0.0)
                    ) + float(getattr(self.env, "patch_waste_decayed_total_last", 0.0))
                if aggperf3_timing:
                    buckets["aggperf3_env_context"] = float(
                        buckets.get("aggperf3_env_context", 0.0) + (time.perf_counter() - t_aggperf3_env_context0)
                    )

                ticks_ran += 1

                t0 = time.perf_counter()
                _eiel_guard_env_alive("spawn_food", int(_tick), lambda: self.env.spawn_food())
                buckets["env_spawn_food"] = buckets.get("env_spawn_food", 0.0) + (time.perf_counter() - t0)

                t_jungle0 = time.perf_counter()
                deaths_this_gen = self._apply_jungle_predator_pressure(deaths_this_gen)
                buckets["jungle_predator_pressure"] = buckets.get("jungle_predator_pressure", 0.0) + (
                    time.perf_counter() - t_jungle0
                )

                # Phase 3 determinism: membership parity probe (env vs pop) before alive list build.
                _eiel_dump_membership_parity(tick_i=int(_tick))


                # ------------------------------------------------------------
                # Phase 3 determinism (FDP v3): membership components probe
                #
                # Purpose:
                #   Capture *raw* membership sources at tick start, BEFORE alive list build:
                #     - Population membership (self.pop.alive_agents())
                #     - Environment membership (env_alive array)
                #
                # This answers: did a "ghost" agent appear first in pop, env, or both?
                #
                # Controls:
                #   EIEL_MEMBERSHIP_COMPONENTS=1 (default on when EIEL_STATE_DUMP=1)
                #   EIEL_MEMBERSHIP_COMPONENTS_TICKS="60-70,..." (optional filter)
                # ------------------------------------------------------------
                try:
                    _mc_on = (os.getenv("EIEL_MEMBERSHIP_COMPONENTS", "1") or "1").strip().lower() in ("1", "true", "yes")
                    if _mc_on and os.environ.get("EIEL_STATE_DUMP", "0").strip() == "1":

                        def _mc_tick_ok(ti: int) -> bool:
                            ticks_s = (os.getenv("EIEL_MEMBERSHIP_COMPONENTS_TICKS", "") or "").strip()
                            if not ticks_s:
                                return True
                            try:
                                for part in str(ticks_s).split(","):
                                    p = str(part).strip()
                                    if not p:
                                        continue
                                    if "-" in p:
                                        a_s, b_s = p.split("-", 1)
                                        a = int(a_s.strip())
                                        b = int(b_s.strip())
                                        lo, hi = (a, b) if a <= b else (b, a)
                                        if lo <= ti <= hi:
                                            return True
                                    else:
                                        if int(p) == int(ti):
                                            return True
                            except Exception:
                                return True
                            return False

                        if _mc_tick_ok(int(_tick)):
                            from engine.simulation.state_dump import maybe_dump_state  # noqa: WPS433

                            # Population-side alive ids (in pop iteration order)
                            pop_alive_agents = list(self.pop.alive_agents())
                            pop_ids_seq = [int(getattr(a, "agent_id")) for a in pop_alive_agents]

                            # Environment-side alive ids (deterministic scan)
                            env_ids_sorted = [int(x) for x in np.where(np.asarray(env_alive, dtype=bool))[0].tolist()]

                            pop_set = set(pop_ids_seq)
                            env_set = set(env_ids_sorted)

                            pop_only = sorted(list(pop_set - env_set))
                            env_only = sorted(list(env_set - pop_set))
                            inter = sorted(list(pop_set & env_set))

                            def _h_int_list(xs) -> str:
                                b = (",".join(str(int(x)) for x in xs)).encode("utf-8")
                                return hashlib.sha256(b).hexdigest()

                            maybe_dump_state(
                                self,
                                generation_index=int(gen),
                                tag=f"membership_components_pre_build_t{int(_tick):03d}",
                                extra={
                                    "tick": int(_tick),
                                    "stage": "pre_alive_list_build",
                                    "pop_alive_n": int(len(pop_ids_seq)),
                                    "env_alive_n": int(len(env_ids_sorted)),
                                    "intersection_n": int(len(inter)),
                                    "pop_ids_seq_sha256": _h_int_list(pop_ids_seq),
                                    "pop_ids_sorted_sha256": _h_int_list(sorted(pop_ids_seq)),
                                    "env_ids_sorted_sha256": _h_int_list(env_ids_sorted),
                                    "pop_only_n": int(len(pop_only)),
                                    "env_only_n": int(len(env_only)),
                                    "pop_only_head": pop_only[:128],
                                    "env_only_head": env_only[:128],
                                    "intersection_head": inter[:128],
                                    "pop_ids_seq_head": pop_ids_seq[:64],
                                    "env_ids_head": env_ids_sorted[:64],
                                },
                            )
                except Exception:
                    # Never interfere with simulation execution
                    pass

                t_alive0 = time.perf_counter()

                # Array-state check instead of removed env.agent_pos dict
                alive = []
                cap = int(env_alive.shape[0])
                for a in self.pop.alive_agents():
                    aid = int(a.agent_id)
                    if 0 <= aid < cap and bool(env_alive[aid]):
                        alive.append(a)

                buckets["alive_list_build"] = buckets.get("alive_list_build", 0.0) + (time.perf_counter() - t_alive0)
                if not alive:
                    break

                # Determinism localization: snapshot policy input domain BEFORE shuffle.
                self._maybe_dump_policy_domain_pre(gen=gen, tick=_tick, alive=alive)
                # Strict debug: snapshot the *inputs* to the policy-domain intersection.
                self._maybe_dump_policy_domain_components_pre(gen=gen, tick=_tick, alive=alive, env_alive=env_alive)

                # Determinism: canonicalize alive ordering before RNG shuffle.
                # Population iteration order can differ across fresh vs resume due to reconstruction/insertion.
                alive.sort(key=lambda a: int(getattr(a, 'agent_id')))

                # D1 determinism guard:
                # Resolve developmental expression for all alive agents in stable order
                # before any subsystem can query it with alternate iteration order.
                # AGGPERF-3 may explicitly skip this conservative prewarm and let
                # policy staging populate the same tick-local cache on first use.
                if (
                    hasattr(self, "resolve_developmental_expression_for_agent")
                    and not bool(getattr(self.sim_cfg, "perf_skip_dev_expression_prewarm_enable", False))
                ):
                    t_aggperf3_dev0 = time.perf_counter() if aggperf3_timing else 0.0
                    for _a in alive:
                        try:
                            self.resolve_developmental_expression_for_agent(_a)
                        except Exception:
                            continue
                    if aggperf3_timing:
                        buckets["aggperf3_dev_expression_prewarm"] = float(
                            buckets.get("aggperf3_dev_expression_prewarm", 0.0)
                            + (time.perf_counter() - t_aggperf3_dev0)
                        )

                # RNG drift probe: capture digest before shuffling alive.
                if os.environ.get("EIEL_RNG_DRIFT_PROBE", "0") == "1":
                    try:
                        from engine.simulation.state_dump import maybe_dump_rng_drift  # noqa: WPS433

                        _alive_ids_seq = [int(getattr(a, "agent_id", 0)) for a in alive]
                        maybe_dump_rng_drift(
                            self,
                            generation_index=gen,
                            tick=_tick,
                            stage="pre_shuffle",
                            alive_ids_seq=_alive_ids_seq,
                            extra={"tick_index": tick_index},
                        )
                    except Exception:
                        pass

                if os.environ.get('EIEL_DEBUG_ALIVE_ORDER', '0').strip() == '1' and int(_tick) == 0:
                    try:
                        ids16 = [int(getattr(a, 'agent_id')) for a in alive[:16]]
                        print(f'[alive_order] gen={int(gen)} tick={int(_tick)} pre_shuffle_ids16={ids16}', flush=True)
                    except Exception:
                        pass

                self.np_rng.shuffle(alive)

                # RNG drift probe: capture digest after shuffling alive.
                if os.environ.get("EIEL_RNG_DRIFT_PROBE", "0") == "1":
                    try:
                        from engine.simulation.state_dump import maybe_dump_rng_drift  # noqa: WPS433

                        _alive_ids_seq = [int(getattr(a, "agent_id", 0)) for a in alive]
                        maybe_dump_rng_drift(
                            self,
                            generation_index=gen,
                            tick=_tick,
                            stage="post_shuffle",
                            alive_ids_seq=_alive_ids_seq,
                            extra={"tick_index": tick_index},
                        )
                    except Exception:
                        pass

                if os.environ.get('EIEL_DEBUG_ALIVE_ORDER', '0').strip() == '1' and int(_tick) == 0:
                    try:
                        ids16 = [int(getattr(a, 'agent_id')) for a in alive[:16]]
                        print(f'[alive_order] gen={int(gen)} tick={int(_tick)} post_shuffle_ids16={ids16}', flush=True)
                    except Exception:
                        pass

                # Phase 3 determinism debug: expose current (gen, tick) to stride-cache probes.
                self._eiel_dbg_gen = int(gen)
                self._eiel_dbg_tick = int(_tick)

                # Phase 3 determinism localization: prepare staged agent-state dumps (tick-filtered).
                # This must run BEFORE stage checkpoints (post_env/post_drives/...) so they can emit.
                try:
                    import os as _os  # noqa: WPS433

                    def _eiel_tick_in_filter(ti: int) -> bool:
                        """
                        Tick filter supports:
                          - CSV ints: "23,61,120"
                          - Ranges: "24-80"
                          - Mixed: "0-10,23,61-90"
                        """
                        ticks_s = (_os.getenv("EIEL_STATE_DUMP_AGENT_DETAIL_TICKS", "") or "").strip()
                        if ticks_s:
                            try:
                                allowed = set()
                                for part in str(ticks_s).split(","):
                                    p = str(part).strip()
                                    if not p:
                                        continue
                                    if "-" in p:
                                        a_s, b_s = p.split("-", 1)
                                        a = int(a_s.strip())
                                        b = int(b_s.strip())
                                        lo, hi = (a, b) if a <= b else (b, a)
                                        # cap to something sane to avoid accidental huge ranges
                                        if hi - lo > 1000000:
                                            return False
                                        allowed.update(range(lo, hi + 1))
                                    else:
                                        allowed.add(int(p))
                                return ti in allowed
                            except Exception:
                                return False
                        try:
                            tmax = int(_os.getenv("EIEL_STATE_DUMP_AGENT_DETAIL_TMAX", "30"))
                        except Exception:
                            tmax = 30
                        return ti <= tmax

                    def _eiel_stage_list() -> list:
                        stages_s = (_os.getenv("EIEL_STATE_DUMP_AGENT_DETAIL_STAGES", "") or "").strip()
                        if not stages_s:
                            return []
                        out = []
                        for s in stages_s.split(","):
                            s2 = str(s).strip()
                            if s2:
                                out.append(s2)
                        # FDP v3 helper: if post_env is requested, also emit pre_env automatically
                        # so we can prove whether divergence is introduced during env tick or earlier.
                        if ("post_env" in out) and ("pre_env" not in out):
                            out.insert(0, "pre_env")
                        return out

                    if int(gen) == 0:
                        _ti = int(_tick)
                        if _eiel_tick_in_filter(_ti):
                            _stages = _eiel_stage_list()
                            # Normalize legacy behavior: if stages not specified, treat as post_birth only.
                            if not _stages:
                                _stages = ["post_birth"]
                            # record desired stages for this tick (consumed by checkpoints below)
                            self._eiel_agent_detail_stages = _stages  # type: ignore[attr-defined]
                        else:
                            # Important: do NOT leak stage selection from a prior tick.
                            self._eiel_agent_detail_stages = []  # type: ignore[attr-defined]
                except Exception:
                    pass


                def _eiel_tick_trace_emit(stage: str, alive_now) -> None:
                    """Write compact per-tick digest trace (CSV) to run_dir/debug when enabled."""
                    try:
                        import os as __os  # noqa: WPS433
                        if (__os.getenv("EIEL_DEBUG_TICK_TRACE", "0") or "0").strip() not in ("1", "true", "TRUE", "yes", "YES"):
                            return
                        # gen filter (optional)
                        gens_s = (__os.getenv("EIEL_DEBUG_TICK_TRACE_GENS", "") or "").strip()
                        if gens_s:
                            try:
                                allowed_g = {int(x.strip()) for x in gens_s.split(",") if x.strip() != ""}
                                if int(gen) not in allowed_g:
                                    return
                            except Exception:
                                return
                        # tick filter (optional, supports ranges)
                        ticks_s = (__os.getenv("EIEL_DEBUG_TICK_TRACE_TICKS", "") or "").strip()
                        if ticks_s:
                            try:
                                allowed_t = set()
                                for part in str(ticks_s).split(","):
                                    p = str(part).strip()
                                    if not p:
                                        continue
                                    if "-" in p:
                                        a_s, b_s = p.split("-", 1)
                                        a = int(a_s.strip()); b = int(b_s.strip())
                                        lo, hi = (a, b) if a <= b else (b, a)
                                        if hi - lo > 1000000:
                                            return
                                        allowed_t.update(range(lo, hi + 1))
                                    else:
                                        allowed_t.add(int(p))
                                if int(_tick) not in allowed_t:
                                    return
                            except Exception:
                                return
                        import json as __json  # noqa: WPS433
                        import hashlib as __hashlib  # noqa: WPS433
                        # agent-state digest
                        _detail = _eiel_build_agent_state_detail(alive_now)
                        _agent_h = __hashlib.sha256(__json.dumps(_detail, separators=(",", ":"), sort_keys=False).encode("utf-8")).hexdigest()
                        # GRU hidden memory digest if available (best-effort)
                        _gru_h = ""
                        try:
                            _gru = getattr(self, "gru_hmem", None)
                            if _gru is not None:
                                # avoid huge dumps; just hash bytes if tensor/ndarray-like
                                if hasattr(_gru, "detach") and hasattr(_gru, "cpu") and hasattr(_gru, "numpy"):
                                    arr = _gru.detach().cpu().numpy()
                                    _gru_h = __hashlib.sha256(arr.tobytes()).hexdigest()
                                elif hasattr(_gru, "tobytes"):
                                    _gru_h = __hashlib.sha256(_gru.tobytes()).hexdigest()
                        except Exception:
                            _gru_h = ""
                        # cache size if present
                        _cache_n = -1
                        try:
                            _cache = getattr(self, "policy_action_cache", None)
                            if _cache is not None and hasattr(_cache, "__len__"):
                                _cache_n = int(len(_cache))
                        except Exception:
                            _cache_n = -1
                        # write line
                        run_dir = getattr(self, "run_dir", None) or getattr(self, "run_dir_path", None) or ""
                        if not run_dir:
                            return
                        out_dir = __os.path.join(str(run_dir), "debug")
                        __os.makedirs(out_dir, exist_ok=True)
                        out_path = __os.path.join(out_dir, f"tick_trace_gen{int(gen):06d}.csv")
                        header = "gen,tick,stage,alive_n,agent_state_sha256,gru_hmem_sha256,cache_n\n"
                        line = f"{int(gen)},{int(_tick)},{stage},{int(len(alive_now))},{_agent_h},{_gru_h},{_cache_n}\n"
                        if not __os.path.exists(out_path):
                            with open(out_path, "w", encoding="utf-8") as _f:
                                _f.write(header)
                                _f.write(line)
                        else:
                            with open(out_path, "a", encoding="utf-8") as _f:
                                _f.write(line)
                    except Exception:
                        return

                def _eiel_maybe_emit_agent_detail(stage: str, alive_now) -> None:
                    # Emit agent-state detail dump for this tick+stage if requested via env vars.
                    try:
                        _stages = getattr(self, "_eiel_agent_detail_stages", None)
                        if not _stages or stage not in _stages:
                            return
                        _eiel_tick_trace_emit(stage, alive_now)
                        from engine.simulation.state_dump import maybe_dump_state  # noqa: WPS433
                        import json as _json  # noqa: WPS433
                        import hashlib as _hashlib  # noqa: WPS433
                        _detail = _eiel_build_agent_state_detail(alive_now)
                        _h = _hashlib.sha256(
                            _json.dumps(_detail, separators=(",", ":"), sort_keys=False).encode("utf-8")
                        ).hexdigest()
                        maybe_dump_state(
                            self,
                            generation_index=int(gen),
                            tag=f"tick_agent_state_detail_{stage}_t{int(_tick):03d}",
                            extra={
                                "tick": int(_tick),
                                "stage": stage,
                                "agent_state_detail_sha256": _h,
                                "agent_state_detail": _detail,
                            },
                        )
                    except Exception:
                        return

                def _eiel_maybe_emit_rng_drift(stage: str, alive_now) -> None:
                    """Emit minimal RNG drift probe dumps (guarded by env vars).

                    This exists to pinpoint *where* the runner_np_rng stream diverges
                    before the observed birth delta at gen0/tick60.
                    """
                    if os.environ.get("EIEL_RNG_DRIFT_PROBE", "0") != "1":
                        return

                    try:
                        from engine.simulation.state_dump import maybe_dump_rng_drift  # noqa: WPS433

                        # alive ordering can correlate with RNG consumption; include as digest.
                        alive_ids_seq = [int(getattr(a, "agent_id")) for a in alive_now]
                        maybe_dump_rng_drift(
                            self,
                            generation_index=int(gen),
                            tick=int(_tick),
                            stage=str(stage),
                            alive_ids_seq=alive_ids_seq,
                            extra={"alive_n": int(len(alive_ids_seq))},
                        )
                    except Exception:
                        return
                if (
                    bool(getattr(self.sim_cfg, "runnercore_supported_profile_enable", False))
                    or bool(getattr(self.sim_cfg, "runnercore_shadow_contract_enable", False))
                    or bool(getattr(self.sim_cfg, "runnercore_candidate_enable", False))
                    or bool(getattr(self.sim_cfg, "runnercore_authoritative_enable", False))
                    or bool(getattr(self.sim_cfg, "runnercore_native_enable", False))
                    or bool(getattr(self.sim_cfg, "runnercore_legacy_compare_enable", False))
                ):
                    from simulation.runnercore.gate import maybe_runnercore_supported_tick  # noqa: WPS433

                    maybe_runnercore_supported_tick(
                        self,
                        alive=alive,
                        generation=int(gen),
                        tick=int(_tick),
                        buckets=buckets,
                    )

                t_aggperf3_policy0 = time.perf_counter() if aggperf3_timing else 0.0
                actions_np = self._batched_actions_strided(alive, buckets=buckets)
                agent_ids_np = np.asarray([int(a.agent_id) for a in alive], dtype=np.int32)
                try:
                    cxxrewrite4_record_action_array(
                        self,
                        alive=alive,
                        agent_ids_np=agent_ids_np,
                        actions_np=actions_np,
                        stage="tick_loop_policy_actions",
                    )
                except Exception:
                    if cxxrewrite4_resident_scalar_sync_fail_closed(self.sim_cfg):
                        raise
                if aggperf3_timing:
                    buckets["aggperf3_policy_action_selection"] = float(
                        buckets.get("aggperf3_policy_action_selection", 0.0)
                        + (time.perf_counter() - t_aggperf3_policy0)
                    )

                # FDP v3+: stash per-tick (agent_id, action) buffers so determinism_probe can hash actions.
                # This is read-only instrumentation; arrays are copied to avoid later mutation.
                try:
                    self._eiel_fdp_last_actions_tick = int(_tick)
                    self._eiel_fdp_last_actions_agent_ids = np.asarray(agent_ids_np, dtype=np.int64).copy()
                    self._eiel_fdp_last_actions_actions = np.asarray(actions_np, dtype=np.int64).copy()
                except Exception:
                    pass

                # FDP v3+: bank + action snapshot around first late-gen action mismatch (gen3 ticks 115-116).
                try:
                    if os.environ.get("EIEL_DETERMINISM_PROBE", "0").strip() == "1":
                        if int(gen) == 3 and (int(_tick) in (0, 50, 115, 116) or (100 <= int(_tick) <= 110)):
                            if hasattr(self, '_eiel_emit_policy_bank_probe'):
                                self._eiel_emit_policy_bank_probe(tag=f"tick{int(_tick):03d}.bank_post_forward", generation_index=int(gen), tick=int(_tick))
                        if int(gen) == 3 and int(_tick) in (115, 116):
                            _run_dir = getattr(self, 'run_dir', None)
                            if _run_dir:
                                _payload = {
                                    'schema_version': 'tick_actions_v1',
                                    'generation_index': int(gen),
                                    'tick': int(_tick),
                                    'agent_ids': [int(x) for x in agent_ids_np.tolist()],
                                    'actions': [int(x) for x in actions_np.tolist()],
                                }
                                _payload['sha256_agent_ids'] = hashlib.sha256(np.asarray(_payload['agent_ids'], dtype=np.int64).tobytes()).hexdigest()
                                _payload['sha256_actions'] = hashlib.sha256(np.asarray(_payload['actions'], dtype=np.int64).tobytes()).hexdigest()
                                _out = os.path.join(str(_run_dir), 'debug', f"tick_actions__gen{int(gen):06d}__tick{int(_tick):03d}.json")
                                with open(_out, 'w', encoding='utf-8') as _f:
                                    json.dump(_payload, _f, sort_keys=True)
                except Exception:
                    pass



                # Forensic V4 (B): ordering proof at gen1/tick0 (forward order + action digest)
                try:
                    if int(gen) == 1 and int(_tick) == 0:
                        import hashlib as _hashlib
                        _ad = _hashlib.sha256(actions_np.tobytes()).hexdigest()
                        emit_forward_order_probe(str(getattr(self, 'run_dir', '')), tag='gen1.tick000.forward', generation_index=int(gen), tick=int(_tick), agent_ids=agent_ids_np.tolist(), batch_slices=[], actions_digest=_ad)
                except Exception:
                    pass

                # Coherence: action switching/oscillation (cheap, before env)
                self._update_action_coherence(alive, actions_np)

                t_blog0 = time.perf_counter()
                self.behavior_logger.record_actions(agent_ids_np, actions_np)
                try:
                    _bo = getattr(self, "branch_observability", None)
                    if _bo is not None:
                        _bo.record_niche_pre_env(
                            self,
                            generation=int(gen),
                            tick=int(_tick),
                            alive=alive,
                            agent_ids_np=agent_ids_np,
                            actions_np=actions_np,
                        )
                except Exception:
                    pass
                self._record_wdl_action_context_pre_env(alive, agent_ids_np, actions_np)
                self._record_delayed_resource_waste_action_context_pre_env(alive, agent_ids_np, actions_np)
                _cxx23_handled_pre = False
                try:
                    _cxx23 = getattr(self, "cxxcore23_live_direct_compact_diagnostics", None)
                    if _cxx23 is not None:
                        _cxx23_handled_pre = bool(
                            _cxx23.record_pre_env(
                                self,
                                generation=int(gen),
                                tick=int(_tick),
                                alive=alive,
                                agent_ids_np=agent_ids_np,
                                actions_np=actions_np,
                            )
                        )
                except Exception:
                    raise
                try:
                    _maw = getattr(self, "memory_action_window_instrumentation", None)
                    if _maw is not None and not _cxx23_handled_pre:
                        _maw.record_pre_env(
                            self,
                            generation=int(gen),
                            tick=int(_tick),
                            alive=alive,
                            agent_ids_np=agent_ids_np,
                            actions_np=actions_np,
                        )
                except Exception:
                    raise
                try:
                    _mm4 = getattr(self, "mem_max_live_level4_instrumentation", None)
                    if _mm4 is not None:
                        _mm4.record_pre_env(
                            self,
                            generation=int(gen),
                            tick=int(_tick),
                            alive=alive,
                            agent_ids_np=agent_ids_np,
                            actions_np=actions_np,
                        )
                except Exception:
                    raise
                try:
                    _pdac = getattr(self, "prediction_demand_action_context_instrumentation", None)
                    if _pdac is not None and not _cxx23_handled_pre:
                        _pdac.record_pre_env(
                            self,
                            generation=int(gen),
                            tick=int(_tick),
                            alive=alive,
                            agent_ids_np=agent_ids_np,
                            actions_np=actions_np,
                        )
                except Exception:
                    raise
                behavior_elapsed = time.perf_counter() - t_blog0
                buckets["behavior_log_tick"] = buckets.get("behavior_log_tick", 0.0) + behavior_elapsed
                if aggperf3_timing:
                    buckets["aggperf3_diagnostics_tape"] = float(
                        buckets.get("aggperf3_diagnostics_tape", 0.0) + behavior_elapsed
                    )

                try:
                    _eiel_maybe_emit_rng_drift("pre_env", alive)
                    _eiel_maybe_emit_agent_detail("pre_env", alive)
                    __eiel_maybe_tick_probe("pre_env", int(_tick))
                    # FDP v3: force tick120 stage probe (ignore tick filters).
                    try:
                        if os.environ.get("EIEL_DETERMINISM_PROBE", "0").strip() == "1" and int(_tick) in (116, 117, 118, 119, 120):
                            from simulation.determinism_probe import maybe_write_probe as _maybe_write_probe  # noqa: WPS433
                            _maybe_write_probe(self, generation_index=int(gen), tag=f"tick{int(_tick):03d}.forced_pre_env")
                    except Exception:
                        pass

                except Exception:
                    pass

                t_aggperf3_env_apply0 = time.perf_counter() if aggperf3_timing else 0.0
                deaths_this_gen, gen_peak_injury = self._apply_env_tick(
                    alive, agent_ids_np, actions_np, buckets, deaths_this_gen, gen_peak_injury
                )
                deaths_this_gen = self._apply_wdl_delayed_hazard_damage(deaths_this_gen)
                deaths_this_gen = self._apply_mem_max_delayed_choice_consequence(deaths_this_gen)
                charge_family_head_residual_cost(self, alive, generation=int(gen), tick=int(_tick))
                if aggperf3_timing:
                    buckets["aggperf3_env_apply"] = float(
                        buckets.get("aggperf3_env_apply", 0.0) + (time.perf_counter() - t_aggperf3_env_apply0)
                    )
                _cxx23_handled_post = False
                try:
                    _cxx23 = getattr(self, "cxxcore23_live_direct_compact_diagnostics", None)
                    if _cxx23 is not None:
                        _cxx23_handled_post = bool(
                            _cxx23.record_post_env(
                                self,
                                generation=int(gen),
                                tick=int(_tick),
                                memory_instrumentation=getattr(self, "memory_action_window_instrumentation", None),
                                prediction_instrumentation=getattr(
                                    self,
                                    "prediction_demand_action_context_instrumentation",
                                    None,
                                ),
                            )
                        )
                except Exception:
                    raise
                try:
                    _maw = getattr(self, "memory_action_window_instrumentation", None)
                    if _maw is not None and not _cxx23_handled_post:
                        _maw.record_post_env(self, generation=int(gen), tick=int(_tick))
                except Exception:
                    raise
                try:
                    _mm4 = getattr(self, "mem_max_live_level4_instrumentation", None)
                    if _mm4 is not None:
                        _mm4.record_post_env(self, generation=int(gen), tick=int(_tick))
                except Exception:
                    raise


                try:
                    _eiel_maybe_emit_rng_drift("post_env", alive)
                    _eiel_maybe_emit_agent_detail("post_env", alive)
                    __eiel_maybe_tick_probe("post_env", int(_tick))
                    # FDP v3: force tick120 stage probe (ignore tick filters).
                    try:
                        if os.environ.get("EIEL_DETERMINISM_PROBE", "0").strip() == "1" and int(_tick) in (116, 117, 118, 119, 120):
                            from simulation.determinism_probe import maybe_write_probe as _maybe_write_probe  # noqa: WPS433
                            _maybe_write_probe(self, generation_index=int(gen), tag=f"tick{int(_tick):03d}.forced_post_env")
                    except Exception:
                        pass

                    # FDP v3: force a last-tick post_env probe (ignore tick filters).
                    try:
                        if os.environ.get("EIEL_DETERMINISM_PROBE", "0").strip() == "1":
                            if int(_tick) == int(self.sim_cfg.steps_per_generation) - 1:
                                from simulation.determinism_probe import maybe_write_probe as _maybe_write_probe  # noqa: WPS433
                                _maybe_write_probe(self, generation_index=int(gen), tag="tickLAST.post_env")
                    except Exception:
                        pass

                except Exception:
                    pass

                t_aggperf3_internal0 = time.perf_counter() if aggperf3_timing else 0.0

                # Phase 4: identity drift measurement (baseline capture + EMA drift)
                self._update_identity_drift(alive)

                # Drives update after env tick so deltas exist
                t_drv0 = time.perf_counter()
                self._update_internal_drives(alive, actions_np, tick_index=int(getattr(self, "_pred_tick", 0)), buckets=buckets)
                buckets["drive_update_ms"] = buckets.get("drive_update_ms", 0.0) + (time.perf_counter() - t_drv0)


                try:
                    _eiel_maybe_emit_rng_drift("post_drives", alive)
                    _eiel_maybe_emit_agent_detail("post_drives", alive)
                    __eiel_maybe_tick_probe("post_drives", int(_tick))
                    # FDP v3: force tick120 stage probe (ignore tick filters).
                    try:
                        if os.environ.get("EIEL_DETERMINISM_PROBE", "0").strip() == "1" and int(_tick) in (116, 117, 118, 119, 120):
                            from simulation.determinism_probe import maybe_write_probe as _maybe_write_probe  # noqa: WPS433
                            _maybe_write_probe(self, generation_index=int(gen), tag=f"tick{int(_tick):03d}.forced_post_drives")
                    except Exception:
                        pass

                except Exception:
                    pass

                t_chem0 = time.perf_counter()
                self._update_chemistry_state(alive, tick_index=int(getattr(self, "_pred_tick", 0)), buckets=buckets)
                buckets["chemistry_update_ms"] = buckets.get("chemistry_update_ms", 0.0) + (time.perf_counter() - t_chem0)

                try:
                    _eiel_maybe_emit_rng_drift("post_chemistry", alive)
                    _eiel_maybe_emit_agent_detail("post_chemistry", alive)
                    __eiel_maybe_tick_probe("post_chemistry", int(_tick))
                except Exception:
                    pass

                t_maint0 = time.perf_counter()
                self._update_maintenance_state(
                    alive,
                    actions_np,
                    tick_index=int(getattr(self, "_pred_tick", 0)),
                    buckets=buckets,
                )
                buckets["maintenance_update_ms"] = buckets.get("maintenance_update_ms", 0.0) + (time.perf_counter() - t_maint0)

                try:
                    _eiel_maybe_emit_rng_drift("post_maintenance", alive)
                    _eiel_maybe_emit_agent_detail("post_maintenance", alive)
                    __eiel_maybe_tick_probe("post_maintenance", int(_tick))
                except Exception:
                    pass

                deaths_this_gen = self._apply_rest_and_recovery(
                    alive, actions_np, rest_streaks, injury_recover_mult, deaths_this_gen, buckets=buckets
                )


                try:
                    _eiel_maybe_emit_rng_drift("post_rest", alive)
                    _eiel_maybe_emit_agent_detail("post_rest", alive)
                    __eiel_maybe_tick_probe("post_rest", int(_tick))
                    # FDP v3: force tick120 stage probe (ignore tick filters).
                    try:
                        if os.environ.get("EIEL_DETERMINISM_PROBE", "0").strip() == "1" and int(_tick) in (116, 117, 118, 119, 120):
                            from simulation.determinism_probe import maybe_write_probe as _maybe_write_probe  # noqa: WPS433
                            _maybe_write_probe(self, generation_index=int(gen), tag=f"tick{int(_tick):03d}.forced_post_rest")
                    except Exception:
                        pass

                except Exception:
                    pass

                deaths_this_gen, gen_peak_injury = self._handle_attacks(
                    alive, actions_np, deaths_this_gen, gen_peak_injury
                )
                if aggperf3_timing:
                    buckets["aggperf3_internal_state_update"] = float(
                        buckets.get("aggperf3_internal_state_update", 0.0)
                        + (time.perf_counter() - t_aggperf3_internal0)
                    )


                try:
                    _eiel_maybe_emit_rng_drift("post_attack", alive)
                    _eiel_maybe_emit_agent_detail("post_attack", alive)
                    __eiel_maybe_tick_probe("post_attack", int(_tick))
                    # FDP v3: force tick120 stage probe (ignore tick filters).
                    try:
                        if os.environ.get("EIEL_DETERMINISM_PROBE", "0").strip() == "1" and int(_tick) in (116, 117, 118, 119, 120):
                            from simulation.determinism_probe import maybe_write_probe as _maybe_write_probe  # noqa: WPS433
                            _maybe_write_probe(self, generation_index=int(gen), tag=f"tick{int(_tick):03d}.forced_post_attack")
                    except Exception:
                        pass

                except Exception:
                    pass


                # ------------------------------------------------------------
                # Phase 3 determinism localization: Birth delta probe
                #
                # Goal: determine whether a newborn (e.g., agent_id=96) is being
                # created during _handle_births at tick T in resume but not fresh.
                #
                # This probe snapshots env/pop alive membership immediately BEFORE
                # births and again AFTER births, then dumps the set delta.
                #
                # Env:
                #   EIEL_BIRTH_DELTA_PROBE=1
                #   EIEL_BIRTH_DELTA_TICKS="60-70,..." (optional; default 60-70)
                # ------------------------------------------------------------
                _eiel_birth_probe_active = False
                _eiel_birth_pop_ids_before = None
                _eiel_birth_env_ids_before = None
                _eiel_birth_succ_before = None
                _eiel_birth_fail_before = None
                if (os.environ.get("EIEL_BIRTH_DELTA_PROBE", "0").strip() == "1"):
                    try:
                        _ticks_s = (os.environ.get("EIEL_BIRTH_DELTA_TICKS", "60-70") or "").strip()
                        _ok = False
                        for _part in str(_ticks_s).split(","):
                            _p = str(_part).strip()
                            if not _p:
                                continue
                            if "-" in _p:
                                _a_s, _b_s = _p.split("-", 1)
                                _a = int(_a_s.strip()); _b = int(_b_s.strip())
                                _lo, _hi = (_a, _b) if _a <= _b else (_b, _a)
                                if _lo <= int(_tick) <= _hi:
                                    _ok = True
                                    break
                            else:
                                if int(_p) == int(_tick):
                                    _ok = True
                                    break
                        if _ok:
                            _eiel_birth_probe_active = True
                            _eiel_birth_succ_before = int(successful_births)
                            _eiel_birth_fail_before = int(failed_births)
                            _eiel_birth_pop_ids_before = [int(getattr(a, "agent_id")) for a in self.pop.alive_agents()]
                            _eiel_birth_env_ids_before = [int(x) for x in np.where(np.asarray(env_alive, dtype=bool))[0].tolist()]
                    except Exception:
                        _eiel_birth_probe_active = False
                # Birth delta probe: PRE snapshot (includes RNG digests via state_dump)
                if _eiel_birth_probe_active:
                    try:
                        from engine.simulation.state_dump import maybe_dump_state  # noqa: WPS433
                        import hashlib as _hb2  # noqa: WPS433
                        _alive_ids_seq = [int(getattr(a, "agent_id")) for a in alive]
                        _alive_ids_seq_head = _alive_ids_seq[:64]
                        _alive_ids_seq_sha = _hb2.sha256((",".join(str(int(x)) for x in _alive_ids_seq)).encode("utf-8")).hexdigest()
                        maybe_dump_state(
                            self,
                            generation_index=int(gen),
                            tag=f"birth_delta_pre_t{int(_tick):03d}",
                            extra={
                                "tick": int(_tick),
                                "stage": "pre_birth",
                                "successful_births_before": int(_eiel_birth_succ_before or 0),
                                "failed_births_before": int(_eiel_birth_fail_before or 0),
                                "alive_ids_seq_sha256": _alive_ids_seq_sha,
                                "alive_ids_seq_head": _alive_ids_seq_head,
                            },
                        )
                    except Exception:
                        pass

                # RNG drift probe: checkpoint immediately before entering _handle_births.
                try:
                    _eiel_maybe_emit_rng_drift("pre_birth", alive)
                except Exception:
                    pass

                # Provide real tick/gen context to the reproduction mixin.
                # (Instrumentation-only; safe to ignore by other mixins.)
                try:
                    self._tick = int(_tick)
                    self.gen = int(gen)
                except Exception:
                    pass

                # Determinism: births should not depend on the shuffled iteration order of the alive list.
                # Pass a canonicalized view to reproduction while keeping the main alive order intact.
                alive_for_births = list(alive)

                alive_for_births.sort(key=lambda a: int(getattr(a, 'agent_id')))
                try:
                    _cxxrewrite4_birth_ids_before = set(int(getattr(a, "agent_id")) for a in self.pop.alive_agents())
                except Exception:
                    _cxxrewrite4_birth_ids_before = set()
                _cxxrewrite4_successful_births_before = int(successful_births)
                _cxxrewrite4_failed_births_before = int(failed_births)

                # FDP v3: tighten RNG trace context around births call (debug-only).
                try:
                    _rm = getattr(self, 'rng_manager', None)
                    if _rm is not None and hasattr(_rm, '_set_trace_ctx'):
                        _rm._set_trace_ctx(gen=int(gen), tick=int(_tick), stage='births_pre_call')
                    # FDP v3: deterministic evolution_py ledger attribution token.
                    if _rm is not None and hasattr(_rm, 'evopy_ledger_set_token'):
                        _rm.evopy_ledger_set_token('births_pre')
                    # FDP v3: milestone ledger dump (pre-birth), only when explicitly enabled.
                    if _rm is not None and hasattr(_rm, 'dump_evopy_ledger') and bool(getattr(_rm, '_evopy_ledger_enabled', False)):
                            _run_dir = getattr(self, 'run_dir', '') or ''
                            if _run_dir:
                                _out = os.path.join(str(_run_dir), 'debug', f'evopy_ledger__gen{int(gen):06d}__tick{int(_tick):03d}__pre_birth.json')
                                _rm.dump_evopy_ledger(path=_out)
                except Exception:
                    pass
                successful_births, failed_births = self._handle_births(
                    alive_for_births, successful_births, failed_births, buckets
                )
                try:
                    if int(successful_births) != int(_cxxrewrite4_successful_births_before) or int(failed_births) != int(_cxxrewrite4_failed_births_before):
                        _alive_after_births = list(self.pop.alive_agents())
                        _after_by_id = {int(getattr(a, "agent_id")): a for a in _alive_after_births}
                        _newborn_ids = sorted(set(_after_by_id.keys()) - set(_cxxrewrite4_birth_ids_before))
                        _sync_agents = [a for a in alive_for_births if bool(getattr(a, "alive", False))]
                        _sync_agents.extend(_after_by_id[int(aid)] for aid in _newborn_ids if int(aid) in _after_by_id)
                        cxxrewrite4_sync_agents_scalar_state(self, _sync_agents, stage="reproduction_births")
                except Exception:
                    if cxxrewrite4_resident_scalar_sync_fail_closed(self.sim_cfg):
                        raise

                # FDP v3: milestone ledger dump (post-birth) and clear attribution token.
                try:
                    _rm = getattr(self, 'rng_manager', None)
                    # Switch token first so any immediate post-birth RNG consumption is attributed correctly.
                    if _rm is not None and hasattr(_rm, 'evopy_ledger_set_token'):
                        _rm.evopy_ledger_set_token('births_post')
                    if _rm is not None and hasattr(_rm, 'dump_evopy_ledger') and bool(getattr(_rm, '_evopy_ledger_enabled', False)):
                            _run_dir = getattr(self, 'run_dir', '') or ''
                            if _run_dir:
                                _out = os.path.join(str(_run_dir), 'debug', f'evopy_ledger__gen{int(gen):06d}__tick{int(_tick):03d}__post_birth.json')
                                _rm.dump_evopy_ledger(path=_out)
                    if _rm is not None and hasattr(_rm, 'evopy_ledger_clear_token'):
                        _rm.evopy_ledger_clear_token()
                except Exception:
                    pass


                if _eiel_birth_probe_active:
                    try:
                        from engine.simulation.state_dump import maybe_dump_state  # noqa: WPS433
                        import hashlib as _hb  # noqa: WPS433

                        _pop_after = [int(getattr(a, "agent_id")) for a in self.pop.alive_agents()]
                        _env_after = [int(x) for x in np.where(np.asarray(env_alive, dtype=bool))[0].tolist()]

                        _pop_before_set = set(_eiel_birth_pop_ids_before or [])
                        _env_before_set = set(_eiel_birth_env_ids_before or [])
                        _pop_after_set = set(_pop_after)
                        _env_after_set = set(_env_after)

                        _pop_new = sorted(list(_pop_after_set - _pop_before_set))
                        _env_new = sorted(list(_env_after_set - _env_before_set))
                        _new_intersection = sorted(list(set(_pop_new) & set(_env_new)))

                        def _h(xs) -> str:
                            b = (",".join(str(int(x)) for x in xs)).encode("utf-8")
                            return _hb.sha256(b).hexdigest()

                        maybe_dump_state(
                            self,
                            generation_index=int(gen),
                            tag=f"birth_delta_post_t{int(_tick):03d}",
                            extra={
                                "tick": int(_tick),
                                "stage": "post_birth",
                                "successful_births_delta": int(int(successful_births) - int(_eiel_birth_succ_before or 0)),
                                "failed_births_delta": int(int(failed_births) - int(_eiel_birth_fail_before or 0)),
                                "pop_alive_n_before": int(len(_eiel_birth_pop_ids_before or [])),
                                "pop_alive_n_after": int(len(_pop_after)),
                                "env_alive_n_before": int(len(_eiel_birth_env_ids_before or [])),
                                "env_alive_n_after": int(len(_env_after)),
                                "pop_new_n": int(len(_pop_new)),
                                "env_new_n": int(len(_env_new)),
                                "new_intersection_n": int(len(_new_intersection)),
                                "pop_new_head": _pop_new[:64],
                                "env_new_head": _env_new[:64],
                                "new_intersection_head": _new_intersection[:64],
                                "pop_new_sha256": _h(_pop_new),
                                "env_new_sha256": _h(_env_new),
                                "new_intersection_sha256": _h(_new_intersection),
                            },
                        )
                    except Exception:
                        pass



                try:
                    _eiel_maybe_emit_agent_detail("post_birth", alive)
                    __eiel_maybe_tick_probe("post_birth", int(_tick))
                    # FDP v3: force tick120 stage probe (ignore tick filters).
                    try:
                        if os.environ.get("EIEL_DETERMINISM_PROBE", "0").strip() == "1" and int(_tick) in (116, 117, 118, 119, 120):
                            from simulation.determinism_probe import maybe_write_probe as _maybe_write_probe  # noqa: WPS433
                            _maybe_write_probe(self, generation_index=int(gen), tag=f"tick{int(_tick):03d}.forced_post_birth")
                    except Exception:
                        pass

                    if hasattr(self, "_eiel_agent_detail_stages"):
                        delattr(self, "_eiel_agent_detail_stages")
                except Exception:
                    pass

                # advance delayed-prediction clock
                self._pred_tick = int(getattr(self, "_pred_tick", 0)) + 1
                maybe_capture_tick_native_science_sample(
                    self,
                    generation=int(gen),
                    tick=int(_tick),
                )
                if bool(getattr(self, "_tick_native_execution_route", False)):
                    from simulation.tick_native_route import maybe_capture_tick_native_report_v1

                    maybe_capture_tick_native_report_v1(self, next_tick=int(_tick) + 1)

                try:
                    __eiel_maybe_tick_probe("end", int(_tick))
                    # FDP v3: force extra end-of-tick probes to localize late-gen drift.
                    try:
                        if os.environ.get("EIEL_DETERMINISM_PROBE", "0").strip() == "1":
                            if int(_tick) == 100:
                                from simulation.determinism_probe import maybe_write_probe as _maybe_write_probe  # noqa: WPS433
                                _maybe_write_probe(self, generation_index=int(gen), tag="tick100.forced_end")
                            if int(_tick) == 110:
                                from simulation.determinism_probe import maybe_write_probe as _maybe_write_probe  # noqa: WPS433
                                _maybe_write_probe(self, generation_index=int(gen), tag="tick110.forced_end")
                            if int(_tick) == 115:
                                from simulation.determinism_probe import maybe_write_probe as _maybe_write_probe  # noqa: WPS433
                                _maybe_write_probe(self, generation_index=int(gen), tag="tick115.forced_end")
                            if int(_tick) == 116:
                                from simulation.determinism_probe import maybe_write_probe as _maybe_write_probe  # noqa: WPS433
                                _maybe_write_probe(self, generation_index=int(gen), tag="tick116.forced_end")
                            if int(_tick) == 117:
                                from simulation.determinism_probe import maybe_write_probe as _maybe_write_probe  # noqa: WPS433
                                _maybe_write_probe(self, generation_index=int(gen), tag="tick117.forced_end")
                            if int(_tick) == 118:
                                from simulation.determinism_probe import maybe_write_probe as _maybe_write_probe  # noqa: WPS433
                                _maybe_write_probe(self, generation_index=int(gen), tag="tick118.forced_end")
                            if int(_tick) == 119:
                                from simulation.determinism_probe import maybe_write_probe as _maybe_write_probe  # noqa: WPS433
                                _maybe_write_probe(self, generation_index=int(gen), tag="tick119.forced_end")
                            if int(_tick) == 120:
                                from simulation.determinism_probe import maybe_write_probe as _maybe_write_probe  # noqa: WPS433
                                _maybe_write_probe(self, generation_index=int(gen), tag="tick120.forced_end")
                            if int(_tick) == int(self.sim_cfg.steps_per_generation) - 1:
                                from simulation.determinism_probe import maybe_write_probe as _maybe_write_probe  # noqa: WPS433
                                _maybe_write_probe(self, generation_index=int(gen), tag="tickLAST.end")
                    except Exception:
                        pass

                except Exception:
                    pass

                if prof is not None:
                    prof.step()

        finally:
            if prof is not None:
                prof.__exit__(None, None, None)

        return deaths_this_gen, successful_births, failed_births, gen_peak_injury, ticks_ran
