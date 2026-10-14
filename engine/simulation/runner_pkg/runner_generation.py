from __future__ import annotations

"""
simulation/runner_pkg/generation.py

Generation lifecycle helpers.

Responsibilities:
- Reset per-generation counters
- Start generation (shock toggles, behavior logging init)
- Finish generation metrics (write one row to metrics.csv)

Important:
Metrics are logged AFTER reproduction/culling, so selected parents must be
snapshotted BEFORE culling (SelectionMixin stores _selected_parent_agents_gen).

Step 3: Developmental coupling analysis (analysis only)
------------------------------------------------------
If sim_cfg.devcoupling_enable is True, we also write a per-generation row to:
- devcoupling.csv

This does NOT affect selection or the world. It only logs cohort comparisons.
"""

from typing import Dict, Tuple, Optional, List, Set
from contextlib import nullcontext
import json
import os
import numpy as np

from agents.chemistry_coupling import classify_recovery_taxonomy, taxonomy_record
from simulation.metrics_pkg.birth_skew_mechanism import maybe_write_birth_skew_mechanism_report
from simulation.metrics_pkg.path1_drh0 import path1_drh0_generation_metrics, zero_path1_drh0_metrics


class GenerationMixin:
    def _write_a3_recovery_taxonomy(self, record: dict) -> None:
        run_dir = str(getattr(self, "run_dir", "") or getattr(self, "run_dir_path", "") or "")
        if not run_dir:
            return
        out_dir = os.path.join(run_dir, "metrics")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "recovery_taxonomy.jsonl")
        with open(out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=True))
            f.write("\n")

    def _initialize_tick_span_state(
        self,
        coordinate: int,
        *,
        shock: bool,
        injury_recover_mult: float,
        emit_legacy_generation_debug: bool,
    ) -> Tuple[
        Dict[str, float],  # buckets
        bool,              # shock
        float,             # injury_recover_mult
        int,               # deaths_this_gen
        int,               # successful_births
        List,              # alive_start
        List[int],         # alive_ids
        Set[int],          # alive_id_set
        int,               # cohort_size
        float,             # gen_peak_injury
        int,               # failed_births
        Dict[int, int],    # rest_streaks
    ]:
        # This helper initializes shared per-tick counters and observations. It
        # contains no temporal authority: callers decide whether a legacy
        # generation boundary (including shock RNG) exists.
        gen = int(coordinate)
        buckets: Dict[str, float] = {}

        deaths_this_gen = 0
        successful_births = 0
        failed_births = 0

        # Birth diagnostics (reset)

        # === Tier 3 change: reset per-generation migration pressure accumulators ===
        self._tier3_steps_gen = 0
        self._tier3_overcrowd_mean_sum_gen = 0.0
        self._tier3_overcrowd_max_gen = 0.0
        self._tier3_crossings_total_gen = 0
        self._tier3_cross_events_total_gen = 0
        self._tier3_extra_move_cost_sum_gen = 0.0
        self._tier3_extra_move_cost_max_gen = 0.0
        # === Tier 3 change: end reset per-generation migration pressure accumulators ===

        self._patch_food_regrowth_total_gen = 0.0
        self._patch_food_harvest_total_gen = 0.0
        self._patch_movement_energy_extra_total_gen = 0.0
        self._patch_movement_energy_event_count_gen = 0
        self._structural_action_opportunity_gen = {}
        self._structural_action_margin_gen = {}
        self._structural_selection_funnel_pre_turnover_gen = {}
        self._structural_tick_reproduction_funnel_gen = {}
        self._patch_scent_emitted_total_gen = 0.0
        self._patch_scent_decayed_total_gen = 0.0
        self._patch_waste_emitted_total_gen = 0.0
        self._patch_waste_decayed_total_gen = 0.0
        self._patch_regrowth_base_total_gen = 0.0
        self._patch_regrowth_effective_total_gen = 0.0
        self._patch_regrowth_suppressed_total_gen = 0.0
        self._patch_regrowth_consequence_steps_gen = 0
        self._patch_regrowth_multiplier_mean_sum_gen = 0.0
        self._patch_regrowth_multiplier_min_gen = 1.0
        self._patch_regrowth_multiplier_p10_sum_gen = 0.0
        self._patch_regrowth_multiplier_p90_sum_gen = 0.0
        self._patch_waste_regrowth_penalty_mean_sum_gen = 0.0
        self._patch_waste_regrowth_penalty_max_gen = 0.0
        self._patch_density_regrowth_penalty_mean_sum_gen = 0.0
        self._patch_density_regrowth_penalty_max_gen = 0.0
        self._patch_high_waste_regrowth_suppression_frac_sum_gen = 0.0
        self._patch_high_density_regrowth_suppression_frac_sum_gen = 0.0
        self._patch_recovery_positive_total_gen = 0.0
        self._patch_recovery_negative_total_gen = 0.0
        self._patch_recovery_decay_total_gen = 0.0
        self._jungle_recovery_regrowth_boost_total_gen = 0.0
        self._jungle_recovery_regrowth_boost_mean_sum_gen = 0.0
        self._jungle_recovery_regrowth_boost_max_gen = 0.0
        self._jungle_recovery_regrowth_boost_p90_sum_gen = 0.0
        self._patch_regrowth_after_waste_total_gen = 0.0
        self._patch_regrowth_after_recovery_total_gen = 0.0
        self._patch_recovery_added_regrowth_total_gen = 0.0
        self._patch_birth_contexts_gen = []
        self._patch_death_contexts_gen = []
        self._jungle_predator_events_gen = 0
        self._jungle_predator_exposure_count_gen = 0
        self._jungle_predator_damage_total_gen = 0.0
        self._jungle_predator_damage_raw_total_gen = 0.0
        self._jungle_predator_damage_effective_total_gen = 0.0
        self._jungle_predator_damage_prevented_total_gen = 0.0
        self._jungle_predator_damage_multipliers_gen = []
        self._jungle_predator_deaths_gen = 0
        self._jungle_predator_event_contexts_gen = []
        self._jungle_birth_scent_emitted_total_gen = 0.0
        self._jungle_birth_waste_emitted_total_gen = 0.0
        self._jungle_birth_emission_count_gen = 0
        self._jungle_birth_emission_contexts_gen = []
        self._jungle_juvenile_predator_events_gen = 0
        self._jungle_juvenile_predator_damage_total_gen = 0.0
        self._jungle_juvenile_predator_deaths_gen = 0
        self._jungle_adult_predator_events_gen = 0
        self._jungle_adult_predator_damage_total_gen = 0.0
        self._wdl_delayed_hazard_event_count_gen = 0
        self._wdl_delayed_hazard_cue_cell_count_sum_gen = 0.0
        self._wdl_delayed_hazard_active_cell_count_sum_gen = 0.0
        self._wdl_delayed_hazard_tick_count_gen = 0
        self._wdl_delayed_hazard_agent_cue_exposure_count_gen = 0
        self._wdl_delayed_hazard_agent_active_exposure_count_gen = 0
        self._wdl_delayed_hazard_damage_total_gen = 0.0
        self._wdl_delayed_hazard_damage_max_gen = 0.0
        self._wdl_delayed_hazard_agents_damaged_gen = 0
        self._wdl_delayed_hazard_agents_damaged_ids_gen = set()
        self._wdl_delayed_hazard_cue_to_active_delay_sum_gen = 0.0
        self._wdl_delayed_hazard_cue_to_active_delay_count_gen = 0
        self._wdl_delayed_hazard_cue_without_active_count_gen = 0
        self._wdl_delayed_hazard_active_without_prior_cue_count_gen = 0
        self._wdl_delayed_hazard_event_contexts_gen = []
        self._wdl_agents_cued_current_gen = set()
        self._wdl_cue_seen_agents_gen = 0
        self._wdl_cue_exposed_next_move_away_count_gen = 0
        self._wdl_cue_exposed_next_move_toward_count_gen = 0
        self._wdl_cue_exposed_stayed_count_gen = 0
        self._wdl_pre_hazard_exit_count_gen = 0
        self._wdl_hazard_enter_count_gen = 0
        self._wdl_hazard_exposure_after_cue_count_gen = 0
        self._path1_drh0_damage_in_refuge_gen = 0.0
        self._path1_drh0_damage_outside_refuge_gen = 0.0
        self._path1_drh0_damage_prevented_by_refuge_gen = 0.0
        self._path1_drh0_static_refuge_damage_prevented_gen = 0.0
        self._path1_drh0_cue_conditioned_damage_prevented_gen = 0.0
        self._path1_drh0_protected_exposure_count_gen = 0
        self._path1_drh0_refuge_exposure_count_gen = 0
        self._path1_drh0_no_cue_false_positive_refuge_movement_gen = 0
        self._path1_drh0_cue_visible_steps_gen = 0
        self._path1_drh0_hidden_window_steps_gen = 0
        self._path1_drh0_hazard_active_steps_gen = 0
        self._prediction_demand_event_count_gen = 0
        self._prediction_demand_transition_count_gen = 0
        self._prediction_demand_cue_cell_count_sum_gen = 0.0
        self._prediction_demand_active_cell_count_sum_gen = 0.0
        self._prediction_demand_tick_count_gen = 0
        self._prediction_demand_cue_ambiguity_sum_gen = 0.0
        self._prediction_demand_cue_ambiguity_count_gen = 0
        self._prediction_demand_cue_to_consequence_delay_sum_gen = 0.0
        self._prediction_demand_cue_to_consequence_delay_count_gen = 0
        self._prediction_demand_cue_without_consequence_count_gen = 0
        self._prediction_demand_consequence_without_prior_cue_count_gen = 0
        self._prediction_demand_pre_consequence_agent_context_count_gen = 0
        self._prediction_demand_consequence_agent_exposure_count_gen = 0
        self._prediction_demand_resource_consequence_total_gen = 0.0
        for _prefix in ("delayed_resource_depletion", "delayed_crowding_waste"):
            setattr(self, f"_{_prefix}_event_count_gen", 0)
            setattr(self, f"_{_prefix}_cue_cell_count_sum_gen", 0.0)
            setattr(self, f"_{_prefix}_hidden_cell_count_sum_gen", 0.0)
            setattr(self, f"_{_prefix}_active_cell_count_sum_gen", 0.0)
            setattr(self, f"_{_prefix}_tick_count_gen", 0)
            setattr(self, f"_{_prefix}_agent_cue_exposure_count_gen", 0)
            setattr(self, f"_{_prefix}_agent_hidden_gap_exposure_count_gen", 0)
            setattr(self, f"_{_prefix}_agent_active_exposure_count_gen", 0)
            setattr(self, f"_{_prefix}_consequence_count_gen", 0)
            setattr(self, f"_{_prefix}_consequence_total_gen", 0.0)
            setattr(self, f"_{_prefix}_cue_to_active_delay_sum_gen", 0.0)
            setattr(self, f"_{_prefix}_cue_to_active_delay_count_gen", 0)
            setattr(self, f"_{_prefix}_cue_without_active_count_gen", 0)
            setattr(self, f"_{_prefix}_active_without_prior_cue_count_gen", 0)
            setattr(self, f"_{_prefix}_agents_cued_current_gen", set())
            setattr(self, f"_{_prefix}_gather_cue_count_gen", 0)
            setattr(self, f"_{_prefix}_gather_hidden_count_gen", 0)
            setattr(self, f"_{_prefix}_gather_active_count_gen", 0)
            setattr(self, f"_{_prefix}_revisit_after_cue_count_gen", 0)
            setattr(self, f"_{_prefix}_move_away_count_gen", 0)
            setattr(self, f"_{_prefix}_move_toward_count_gen", 0)
            setattr(self, f"_{_prefix}_stay_count_gen", 0)
        for _name, _default in (
            ("_mem_max_delayed_choice_event_count_gen", 0),
            ("_mem_max_delayed_choice_cue_step_active_gen", 0),
            ("_mem_max_delayed_choice_hidden_step_active_gen", 0),
            ("_mem_max_delayed_choice_consequence_step_active_gen", 0),
            ("_mem_max_delayed_choice_target_a_cue_count_gen", 0),
            ("_mem_max_delayed_choice_target_b_cue_count_gen", 0),
            ("_mem_max_delayed_choice_correct_arrival_count_gen", 0),
            ("_mem_max_delayed_choice_incorrect_arrival_count_gen", 0),
            ("_mem_max_delayed_choice_no_choice_count_gen", 0),
            ("_mem_max_delayed_choice_consequence_exposure_count_gen", 0),
            ("_mem_max_delayed_choice_damage_total_gen", 0.0),
            ("_mem_max_delayed_choice_damage_prevented_total_gen", 0.0),
            ("_mem_max_delayed_choice_expected_consequence_total_gen", 0.0),
            ("_mem_max_delayed_choice_applied_health_damage_total_gen", 0.0),
            ("_mem_max_delayed_choice_applied_energy_cost_total_gen", 0.0),
            ("_mem_max_delayed_choice_consequence_applied_count_gen", 0),
            ("_mem_max_delayed_choice_consequence_eligible_count_gen", 0),
            ("_mem_max_delayed_choice_duplicate_prevented_count_gen", 0),
            ("_mem_max_delayed_choice_correct_health_damage_total_gen", 0.0),
            ("_mem_max_delayed_choice_incorrect_health_damage_total_gen", 0.0),
            ("_mem_max_delayed_choice_no_choice_health_damage_total_gen", 0.0),
            ("_mem_max_delayed_choice_target_a_arrival_count_gen", 0),
            ("_mem_max_delayed_choice_target_b_arrival_count_gen", 0),
            ("_mem_max_delayed_choice_reachable_agents_gen", 0),
            ("_mem_max_delayed_choice_reachability_denominator_gen", 0),
            ("_mem_max_delayed_choice_applied_event_agent_pairs_gen", set()),
        ):
            setattr(self, _name, _default)
        if hasattr(self, "_reset_memory_runtime_telemetry"):
            self._reset_memory_runtime_telemetry()

        # -----------------------------
        # Tier 4: reset per-generation juvenile vulnerability accumulators
        # -----------------------------
        self._tier4_steps_gen = 0
        self._tier4_juvenile_exposure_steps_gen = 0
        self._tier4_adult_exposure_steps_gen = 0
        self._tier4_juvenile_deaths_gen = 0
        self._tier4_adult_deaths_gen = 0
        self._tier4_juvenile_frac_sum_gen = 0.0
        self._tier4_deaths_age_bins_gen = None

        self._injury_blocked_births_gen = 0
        self._birth_attempts_gen = 0
        self._birth_pre_cooldown_gen = 0
        self._birth_pre_no_energy_gen = 0
        self._birth_pre_injury_gate_gen = 0
        self._birth_pre_no_space_gen = 0
        self._birth_pre_no_mate_gen = 0
        self._birth_attempt_conception_fail_gen = 0
        self._birth_gestation_fail_gen = 0
        self._birth_make_child_fail_gen = 0
        self._birth_fail_no_space_gen = 0
        self._birth_fail_no_mate_gen = 0
        self._birth_fail_buffer_gen = 0
        self._birth_fail_injury_gate_gen = 0
        self._birth_fail_scar_gate_gen = 0
        self._birth_fail_conception_gen = 0
        self._birth_fail_gestation_gen = 0
        self._dev_stage_transition_count_gen = 0
        self._dev_stage_eval_stage_counts_gen = {}
        self._dev_d2_lock_event_count_gen = 0

        # Phase A2: maintenance delayed-process counters (annotation only).
        self._maintenance_tick_count_gen = 0
        self._maintenance_action_opportunities_gen = 0
        self._maintenance_schedule_requests_gen = 0
        self._maintenance_apply_count_gen = 0
        self._maintenance_energy_cost_total_gen = 0.0
        self._maintenance_opportunity_cost_total_gen = 0.0
        self._maintenance_effect_total_gen = 0.0
        self._maintenance_domination_annotation = {
            "triggered": False,
            "tick": -1,
            "ratio": 0.0,
            "threshold": float(getattr(self.sim_cfg, "maintenance_domination_threshold", 0.75)),
            "requests": 0,
            "opportunities": 0,
        }

        # Selection snapshots (reset each gen)
        self._eligible_parent_agents_gen = []
        self._selected_parent_ids_gen = set()
        self._selected_parent_agents_gen = []
        self._lineage_reproduction_funnel_birth_contexts_gen = []
        self._lineage_reproduction_funnel_event_contexts_gen = []
        self._lineage_reproduction_funnel_scan_action_contexts_gen = []
        self._lineage_reproduction_funnel_pre_turnover_gen = {}
        self._structural_tick_reproduction_funnel_gen = {}

        alive_start_all = list(self.pop.alive_agents())

        # ---- FIX: env.agent_pos dict is removed; use env.agent_alive array instead ----
        env_alive = getattr(self.env, "agent_alive", None)
        if env_alive is None:
            raise AttributeError("env has no 'agent_alive' array; generation mixin requires array-based env state")

        cap = int(env_alive.shape[0])
        alive_start = []
        for a in alive_start_all:
            aid = int(a.agent_id)
            if 0 <= aid < cap and bool(env_alive[aid]):
                alive_start.append(a)
        # ---------------------------------------------------------------------------

        alive_ids = [int(a.agent_id) for a in alive_start]
        alive_id_set = set(alive_ids)
        cohort_size = len(alive_ids)

        if getattr(self.env_cfg, "enable_injury", False) and alive_start:
            gen_peak_injury = float(max(float(getattr(a, "injury", 0.0)) for a in alive_start))
        else:
            gen_peak_injury = 0.0

        if emit_legacy_generation_debug and gen < 5:
            vals = [float(getattr(a, "injury", 0.0)) for a in alive_start]
            if vals:
                print(f"[gen {gen}] post-cull injury mean={np.mean(vals):.3f} max={np.max(vals):.3f}")
            else:
                print(f"[gen {gen}] post-cull injury mean=0.000 max=0.000")

        rest_streaks: Dict[int, int] = {int(a.agent_id): int(getattr(a, "rest_streak", 0)) for a in alive_start}

        inj_start = {int(aid): float(getattr(self.pop.agents.get(int(aid)), "injury", 0.0)) for aid in alive_ids}
        self.behavior_logger.start_generation(gen, alive_ids, injuries=inj_start)

        return (
            buckets,
            shock,
            injury_recover_mult,
            deaths_this_gen,
            successful_births,
            alive_start,
            alive_ids,
            alive_id_set,
            cohort_size,
            gen_peak_injury,
            failed_births,
            rest_streaks,
        )

    def _start_generation(self, gen: int):
        """Start one legacy generation, retaining its historical boundary RNG."""

        _budget = nullcontext()
        rm = getattr(self, "rng_manager", None)
        if rm is not None:
            expected = rm.expected_delta_for_boundary("core_tick.generation.shock_active")
            _budget = rm.budget_scope(
                "core_tick.generation.shock_active", expected_delta=expected
            )
        with _budget:
            shock = self._shock_active(gen)
        if shock:
            self.env.set_multipliers(
                food_spawn_mult=float(self.sim_cfg.shock_food_spawn_mult),
                action_cost_mult=float(self.sim_cfg.shock_action_cost_mult),
            )
            injury_recover_mult = float(self.sim_cfg.shock_injury_recover_mult)
        else:
            self.env.reset_multipliers()
            injury_recover_mult = 1.0
        return self._initialize_tick_span_state(
            gen,
            shock=shock,
            injury_recover_mult=injury_recover_mult,
            emit_legacy_generation_debug=True,
        )

    def _start_tick_native_span(self):
        """Initialize a continuous tick span without a generation boundary."""

        self.env.reset_multipliers()
        return self._initialize_tick_span_state(
            0,
            shock=False,
            injury_recover_mult=1.0,
            emit_legacy_generation_debug=False,
        )

    def _finish_generation_metrics(
        self,
        gen: int,
        alive_after: List,
        deaths_this_gen: int,
        successful_births: int,
        failed_births: int,
        cohort_size: int,
        cohort_survivors: int,
        cohort_survival_rate: float,
        gen_peak_injury: float,
        shock: bool,
        log_gen_params: Set[str],  # kept for signature compatibility with LoopMixin
        buckets: Optional[Dict[str, float]],
    ) -> None:
        # IMPORTANT: use snapshot from SelectionMixin (pre-cull)
        eligible_parent_agents = list(getattr(self, "_eligible_parent_agents_gen", []))
        selected_parent_agents = list(getattr(self, "_selected_parent_agents_gen", []))
        self._maintenance_domination_annotation_last_gen = dict(
            getattr(self, "_maintenance_domination_annotation", {}) or {}
        )
        try:
            if hasattr(self, "_dev_flush_pending_traces"):
                self._dev_flush_pending_traces(force=True)
            if hasattr(self, "_dev_flush_trace_emit_buffer"):
                self._dev_flush_trace_emit_buffer()
            if hasattr(self, "_dev_flush_pending_lock_traces"):
                self._dev_flush_pending_lock_traces(force=True)
            if hasattr(self, "_dev_flush_lock_trace_emit_buffer"):
                self._dev_flush_lock_trace_emit_buffer()
        except Exception:
            pass

        # A3 recovery taxonomy: deterministic perturbation outcome annotation.
        if bool(getattr(self.sim_cfg, "a3_recovery_taxonomy_enable", False)):
            eff_vals = []
            drift_vals = []
            for a in alive_after:
                st = getattr(a, "stats", None)
                if st is None:
                    continue
                eff_vals.append(float(getattr(st, "effective_fitness", 0.0)))
                drift_vals.append(float(getattr(st, "identity_drift_ema", 0.0)))
            mean_eff = float(np.mean(np.asarray(eff_vals, dtype=np.float32))) if eff_vals else 0.0
            mean_drift = float(np.mean(np.asarray(drift_vals, dtype=np.float32))) if drift_vals else 0.0

            alpha = float(getattr(self.sim_cfg, "a3_recovery_baseline_ema_alpha", 0.15))
            alpha = 0.0 if alpha < 0.0 else 1.0 if alpha > 1.0 else alpha
            bfit_prev = float(getattr(self, "_a3_recovery_baseline_fitness_ema", mean_eff if mean_eff > 0.0 else 1.0))
            bdr_prev = float(getattr(self, "_a3_recovery_baseline_drift_ema", mean_drift + 1e-6))

            if not bool(shock):
                bfit_new = (1.0 - alpha) * bfit_prev + alpha * mean_eff
                bdr_new = (1.0 - alpha) * bdr_prev + alpha * mean_drift
                self._a3_recovery_baseline_fitness_ema = float(bfit_new)
                self._a3_recovery_baseline_drift_ema = float(bdr_new)
                cls = "return"
                fr = 1.0
                dr = 1.0
            else:
                self._a3_recovery_baseline_fitness_ema = float(bfit_prev)
                self._a3_recovery_baseline_drift_ema = float(bdr_prev)
                fr = float(mean_eff / max(1e-6, bfit_prev))
                dr = float((mean_drift + 1e-6) / max(1e-6, bdr_prev + 1e-6))
                cls = classify_recovery_taxonomy(
                    fitness_ratio=fr,
                    drift_ratio=dr,
                    return_fitness_ratio=float(getattr(self.sim_cfg, "a3_recovery_return_fitness_ratio", 0.92)),
                    return_drift_ratio_max=float(getattr(self.sim_cfg, "a3_recovery_return_drift_ratio_max", 1.10)),
                    shift_fitness_ratio_min=float(getattr(self.sim_cfg, "a3_recovery_shift_fitness_ratio_min", 0.70)),
                )

            rec = taxonomy_record(
                generation=int(gen),
                perturbation_active=bool(shock),
                classification=str(cls),
                fitness_ratio=float(fr),
                drift_ratio=float(dr),
                baseline_fitness_ema=float(getattr(self, "_a3_recovery_baseline_fitness_ema", 0.0)),
                baseline_drift_ema=float(getattr(self, "_a3_recovery_baseline_drift_ema", 0.0)),
                mean_effective_fitness=float(mean_eff),
                mean_identity_drift=float(mean_drift),
                thresholds={
                    "return_fitness_ratio": float(getattr(self.sim_cfg, "a3_recovery_return_fitness_ratio", 0.92)),
                    "return_drift_ratio_max": float(getattr(self.sim_cfg, "a3_recovery_return_drift_ratio_max", 1.10)),
                    "shift_fitness_ratio_min": float(getattr(self.sim_cfg, "a3_recovery_shift_fitness_ratio_min", 0.70)),
                },
            )
            self._a3_recovery_taxonomy_last_gen = dict(rec)
            try:
                self._write_a3_recovery_taxonomy(rec)
            except Exception:
                pass

        # -----------------------------
        # Finalize episode-0 per-agent means (what MetricsLogger expects)
        # -----------------------------
        k0 = int(getattr(self.sim_cfg, "episode0_k", 20))
        for a in alive_after:
            st = getattr(a, "stats", None)
            if st is None:
                continue

            # pred err mean
            c = int(getattr(st, "episode0_pred_err_count", 0))
            s = float(getattr(st, "episode0_pred_err_sum", 0.0))
            st.episode0_pred_err_mean = (s / float(c)) if c > 0 else 0.0

            # drive jitter mean (normalize by K, not by survival_steps, to keep semantics stable)
            js = float(getattr(st, "episode0_drive_jitter_sum", 0.0))
            st.episode0_drive_jitter = (js / float(k0)) if k0 > 0 else 0.0

            # selfharm (count as float, logger expects numeric)
            st.episode0_selfharm = float(getattr(st, "episode0_selfharm_events", 0))

        # -----------------------------

        # === Tier 3 change: finalize per-generation migration pressure metrics ===
        steps_gen = int(getattr(self, "_tier3_steps_gen", 0) or 0)
        if steps_gen <= 0:
            tier3_overcrowd_mean = 0.0
            tier3_overcrowd_max = 0.0
            tier3_crossings_per_step = 0.0
            tier3_cross_events_per_step = 0.0
            tier3_extra_move_cost_mean = 0.0
            tier3_extra_move_cost_max = 0.0
        else:
            tier3_overcrowd_mean = float(getattr(self, "_tier3_overcrowd_mean_sum_gen", 0.0)) / float(steps_gen)
            tier3_overcrowd_max = float(getattr(self, "_tier3_overcrowd_max_gen", 0.0))
            tier3_crossings_per_step = float(getattr(self, "_tier3_crossings_total_gen", 0)) / float(steps_gen)
            tier3_cross_events_per_step = float(getattr(self, "_tier3_cross_events_total_gen", 0)) / float(steps_gen)
            tier3_extra_move_cost_mean = float(getattr(self, "_tier3_extra_move_cost_sum_gen", 0.0)) / float(steps_gen)
            tier3_extra_move_cost_max = float(getattr(self, "_tier3_extra_move_cost_max_gen", 0.0))

        # -----------------------------
        # Tier 4: juvenile spatial vulnerability (age-only stats)
        # -----------------------------
        tier4_juvenile_exposure_steps = int(getattr(self, "_tier4_juvenile_exposure_steps_gen", 0) or 0)
        tier4_adult_exposure_steps = int(getattr(self, "_tier4_adult_exposure_steps_gen", 0) or 0)
        tier4_juvenile_deaths = int(getattr(self, "_tier4_juvenile_deaths_gen", 0) or 0)
        tier4_adult_deaths = int(getattr(self, "_tier4_adult_deaths_gen", 0) or 0)

        tier4_juvenile_death_rate_per_1k_steps = 0.0
        tier4_adult_death_rate_per_1k_steps = 0.0
        if tier4_juvenile_exposure_steps > 0:
            tier4_juvenile_death_rate_per_1k_steps = 1000.0 * float(tier4_juvenile_deaths) / float(tier4_juvenile_exposure_steps)
        if tier4_adult_exposure_steps > 0:
            tier4_adult_death_rate_per_1k_steps = 1000.0 * float(tier4_adult_deaths) / float(tier4_adult_exposure_steps)

        steps_gen_t4 = int(getattr(self, "_tier4_steps_gen", 0) or 0)
        tier4_juvenile_fraction_mean = 0.0
        if steps_gen_t4 > 0:
            tier4_juvenile_fraction_mean = float(getattr(self, "_tier4_juvenile_frac_sum_gen", 0.0)) / float(steps_gen_t4)

        # Age at first reproduction distribution among reproducers present at generation end.
        repro_ages = []
        for a in alive_after:
            try:
                st = getattr(a, "stats", None)
                fr = getattr(st, "first_repro_step", None) if st is not None else None
                if fr is not None:
                    repro_ages.append(int(fr))
            except Exception:
                continue

        tier4_fraction_reproduced_alive_end = 0.0
        tier4_first_repro_step_mean_reproducers_alive_end = 0.0
        tier4_first_repro_step_p50_reproducers_alive_end = 0.0
        tier4_first_repro_step_p90_reproducers_alive_end = 0.0

        if len(alive_after) > 0:
            tier4_fraction_reproduced_alive_end = float(len(repro_ages)) / float(len(alive_after))

        if repro_ages:
            repro_ages.sort()
            tier4_first_repro_step_mean_reproducers_alive_end = float(sum(repro_ages)) / float(len(repro_ages))
            def _pct(p: float) -> float:
                if not repro_ages:
                    return 0.0
                # nearest-rank style
                k = int(round((p / 100.0) * (len(repro_ages) - 1)))
                k = max(0, min(len(repro_ages) - 1, k))
                return float(repro_ages[k])
            tier4_first_repro_step_p50_reproducers_alive_end = _pct(50.0)
            tier4_first_repro_step_p90_reproducers_alive_end = _pct(90.0)

        # Survival curves (coarse bins) as JSON strings
        try:
            import json as _json
        except Exception:
            _json = None

        edges = list(getattr(self.env_cfg, "tier4_survival_bins_steps", [])) or []
        alive_age_bins = []
        if len(edges) >= 2:
            alive_age_bins = [0 for _ in range(len(edges) - 1)]
            for a in alive_after:
                try:
                    st = getattr(a, "stats", None)
                    age = int(getattr(st, "survival_steps", 0)) if st is not None else 0
                    for k in range(len(edges) - 1):
                        if age >= int(edges[k]) and age < int(edges[k + 1]):
                            alive_age_bins[k] += 1
                            break
                except Exception:
                    continue

        deaths_age_bins = getattr(self, "_tier4_deaths_age_bins_gen", None)
        if deaths_age_bins is None:
            deaths_age_bins = [0 for _ in range(max(0, len(edges) - 1))]

        tier4_bins_steps_json = _json.dumps(edges) if _json is not None else str(edges)
        tier4_alive_age_hist_json = _json.dumps(alive_age_bins) if _json is not None else str(alive_age_bins)
        tier4_deaths_age_hist_json = _json.dumps(deaths_age_bins) if _json is not None else str(deaths_age_bins)

        # === Tier 3 change: end finalize per-generation migration pressure metrics ===

        patch_metrics = {
            "patch_food_amount_mean": 0.0,
            "patch_food_amount_min": 0.0,
            "patch_food_amount_max": 0.0,
            "patch_food_amount_p10": 0.0,
            "patch_food_amount_p90": 0.0,
            "patch_food_depleted_frac": 0.0,
            "patch_food_full_frac": 0.0,
            "patch_food_regrowth_total": 0.0,
            "patch_food_harvest_total": 0.0,
            "patch_fertility_mean": 0.0,
            "patch_cover_mean": 0.0,
            "patch_cover_occupied_mean": 0.0,
            "patch_shelter_mean": 0.0,
            "patch_shelter_occupied_mean": 0.0,
            "patch_visibility_mean": 0.0,
            "patch_visibility_occupied_mean": 0.0,
            "patch_movement_cost_mean": 0.0,
            "patch_movement_cost_occupied_mean": 0.0,
            "patch_movement_energy_extra_total": 0.0,
            "patch_movement_energy_extra_mean": 0.0,
            "patch_scent_mean": 0.0,
            "patch_scent_max": 0.0,
            "patch_scent_p90": 0.0,
            "patch_scent_occupied_mean": 0.0,
            "patch_scent_emitted_total": 0.0,
            "patch_scent_decayed_total": 0.0,
            "patch_waste_mean": 0.0,
            "patch_waste_max": 0.0,
            "patch_waste_p90": 0.0,
            "patch_waste_occupied_mean": 0.0,
            "patch_waste_emitted_total": 0.0,
            "patch_waste_decayed_total": 0.0,
            "patch_density_mean": 0.0,
            "patch_density_max": 0.0,
            "patch_density_p90": 0.0,
            "patch_density_occupied_mean": 0.0,
            "patch_density_hotspot_frac": 0.0,
            "patch_observation_enable": 0,
            "patch_observation_schema_version": 0,
            "patch_observation_channel_count": 0,
            "patch_observation_size": 0,
            "patch_world_descriptor_schema_version": 0,
            "patch_world_diagnostics_enable": 0,
            "patch_world_lineage_descriptors_enable": 0,
            "occupied_patch_food_amount_mean": 0.0,
            "occupied_patch_fertility_mean": 0.0,
            "occupied_patch_cover_mean": 0.0,
            "occupied_patch_shelter_mean": 0.0,
            "occupied_patch_visibility_mean": 0.0,
            "occupied_patch_movement_cost_mean": 0.0,
            "occupied_patch_scent_mean": 0.0,
            "occupied_patch_waste_mean": 0.0,
            "occupied_patch_density_mean": 0.0,
            "birth_patch_count": 0.0,
            "birth_patch_food_amount_mean": 0.0,
            "birth_patch_fertility_mean": 0.0,
            "birth_patch_cover_mean": 0.0,
            "birth_patch_shelter_mean": 0.0,
            "birth_patch_visibility_mean": 0.0,
            "birth_patch_movement_cost_mean": 0.0,
            "birth_patch_scent_mean": 0.0,
            "birth_patch_waste_mean": 0.0,
            "birth_patch_density_mean": 0.0,
            "death_patch_count": 0.0,
            "death_patch_food_amount_mean": 0.0,
            "death_patch_fertility_mean": 0.0,
            "death_patch_cover_mean": 0.0,
            "death_patch_shelter_mean": 0.0,
            "death_patch_visibility_mean": 0.0,
            "death_patch_movement_cost_mean": 0.0,
            "death_patch_scent_mean": 0.0,
            "death_patch_waste_mean": 0.0,
            "death_patch_density_mean": 0.0,
            "patch_food_occupancy_bias": 0.0,
            "patch_fertility_occupancy_bias": 0.0,
            "patch_cover_occupancy_bias": 0.0,
            "patch_shelter_occupancy_bias": 0.0,
            "patch_visibility_occupancy_bias": 0.0,
            "patch_movement_cost_occupancy_bias": 0.0,
            "patch_scent_occupancy_bias": 0.0,
            "patch_waste_occupancy_bias": 0.0,
            "patch_density_occupancy_bias": 0.0,
            "lineage_habitat_descriptor_count": 0.0,
            "lineage_patch_food_amount_mean_std": 0.0,
            "lineage_patch_cover_mean_std": 0.0,
            "lineage_patch_shelter_mean_std": 0.0,
            "lineage_patch_scent_mean_std": 0.0,
            "lineage_patch_waste_mean_std": 0.0,
            "lineage_patch_density_mean_std": 0.0,
            "lineage_patch_recovery_activity_mean_std": 0.0,
            "lineage_diversity_schema_version": 0,
            "lineage_diversity_reporting_enable": 0,
            "lineage_strategy_descriptors_enable": 0,
            "lineage_branch_candidate_reporting_enable": 0,
            "lineage_active_count": 0.0,
            "lineage_extinct_count": 0.0,
            "lineage_alive_count_total": 0.0,
            "lineage_top1_alive_share": 0.0,
            "lineage_top3_alive_share": 0.0,
            "lineage_top5_alive_share": 0.0,
            "lineage_top1_birth_share": 0.0,
            "lineage_top3_birth_share": 0.0,
            "lineage_top5_birth_share": 0.0,
            "lineage_top1_death_share": 0.0,
            "lineage_top3_death_share": 0.0,
            "lineage_alive_share_entropy": 0.0,
            "lineage_birth_share_entropy": 0.0,
            "lineage_effective_alive_lineages": 0.0,
            "lineage_effective_birth_lineages": 0.0,
            "lineage_top1_dominance_warning": 0,
            "lineage_top3_dominance_warning": 0,
            "lineage_entropy_collapse_warning": 0,
            "lineage_low_active_count_warning": 0,
            "lineage_collapse_warning_any": 0,
            "lineage_strategy_descriptor_count": 0.0,
            "lineage_birth_patch_waste_mean_std": 0.0,
            "lineage_birth_patch_shelter_mean_std": 0.0,
            "lineage_birth_patch_recovery_activity_mean_std": 0.0,
            "lineage_predator_damage_mean_std": 0.0,
            "lineage_predator_damage_prevented_mean_std": 0.0,
            "lineage_reproduction_rate_std": 0.0,
            "lineage_branch_candidate_count": 0.0,
            "lineage_habitat_descriptors_json": "[]",
            "lineage_strategy_descriptors_json": "[]",
            "lineage_branch_candidates_json": "[]",
            "patch_observation_min": 0.0,
            "patch_observation_max": 0.0,
            "patch_observation_nan_count": 0.0,
            "patch_world_waste_consequence_enable": 0,
            "patch_regrowth_base_total": 0.0,
            "patch_regrowth_effective_total": 0.0,
            "patch_regrowth_suppressed_total": 0.0,
            "patch_regrowth_multiplier_mean": 1.0,
            "patch_regrowth_multiplier_min": 1.0,
            "patch_regrowth_multiplier_p10": 1.0,
            "patch_regrowth_multiplier_p90": 1.0,
            "patch_waste_regrowth_penalty_mean": 0.0,
            "patch_waste_regrowth_penalty_max": 0.0,
            "patch_density_regrowth_penalty_mean": 0.0,
            "patch_density_regrowth_penalty_max": 0.0,
            "patch_high_waste_regrowth_suppression_frac": 0.0,
            "patch_high_density_regrowth_suppression_frac": 0.0,
            "patch_fertility_damage_total": 0.0,
            "patch_fertility_recovery_total": 0.0,
            "jungle_predator_pressure_enable": 0,
            "jungle_predator_risk_mean": 0.0,
            "jungle_predator_risk_min": 0.0,
            "jungle_predator_risk_max": 0.0,
            "jungle_predator_risk_p10": 0.0,
            "jungle_predator_risk_p90": 0.0,
            "jungle_predator_occupied_risk_mean": 0.0,
            "jungle_predator_occupied_risk_p90": 0.0,
            "jungle_predator_events": 0.0,
            "jungle_predator_event_rate": 0.0,
            "jungle_predator_damage_total": 0.0,
            "jungle_predator_damage_mean": 0.0,
            "jungle_terrain_protection_enable": 0,
            "jungle_predator_damage_raw_total": 0.0,
            "jungle_predator_damage_effective_total": 0.0,
            "jungle_predator_damage_prevented_total": 0.0,
            "jungle_predator_damage_multiplier_mean": 1.0,
            "jungle_predator_damage_multiplier_min": 1.0,
            "jungle_predator_damage_multiplier_p10": 1.0,
            "jungle_predator_damage_multiplier_p90": 1.0,
            "jungle_predator_event_damage_multiplier_mean": 1.0,
            "jungle_predator_event_damage_prevented_mean": 0.0,
            "occupied_predator_damage_multiplier_mean": 1.0,
            "jungle_predator_deaths": 0.0,
            "jungle_predator_scent_component_mean": 0.0,
            "jungle_predator_waste_component_mean": 0.0,
            "jungle_predator_density_component_mean": 0.0,
            "jungle_predator_visibility_component_mean": 0.0,
            "jungle_predator_cover_reduction_mean": 0.0,
            "jungle_predator_shelter_reduction_mean": 0.0,
            "predator_event_patch_count": 0.0,
            "predator_event_patch_scent_mean": 0.0,
            "predator_event_patch_waste_mean": 0.0,
            "predator_event_patch_density_mean": 0.0,
            "predator_event_patch_cover_mean": 0.0,
            "predator_event_patch_shelter_mean": 0.0,
            "predator_event_patch_visibility_mean": 0.0,
            "jungle_reproduction_emissions_enable": 0,
            "jungle_birth_scent_emitted_total": 0.0,
            "jungle_birth_waste_emitted_total": 0.0,
            "jungle_birth_emission_count": 0.0,
            "jungle_birth_emission_patch_scent_mean": 0.0,
            "jungle_birth_emission_patch_waste_mean": 0.0,
            "jungle_birth_emission_patch_density_mean": 0.0,
            "jungle_birth_emission_patch_cover_mean": 0.0,
            "jungle_birth_emission_patch_shelter_mean": 0.0,
            "jungle_juvenile_predator_vulnerability_enable": 0,
            "jungle_juvenile_predator_events": 0.0,
            "jungle_juvenile_predator_damage_total": 0.0,
            "jungle_juvenile_predator_deaths": 0.0,
            "jungle_adult_predator_events": 0.0,
            "jungle_adult_predator_damage_total": 0.0,
            "jungle_juvenile_vs_adult_predator_event_ratio": 0.0,
            "wdl_delayed_hazard_enable": 0,
            "wdl_delayed_hazard_cue_cell_count": 0.0,
            "wdl_delayed_hazard_active_cell_count": 0.0,
            "wdl_delayed_hazard_event_count": 0.0,
            "wdl_delayed_hazard_agent_cue_exposure_count": 0.0,
            "wdl_delayed_hazard_agent_active_exposure_count": 0.0,
            "wdl_delayed_hazard_damage_total": 0.0,
            "wdl_delayed_hazard_damage_mean": 0.0,
            "wdl_delayed_hazard_damage_max": 0.0,
            "wdl_delayed_hazard_agents_damaged": 0.0,
            "wdl_delayed_hazard_cue_to_active_delay_mean": 0.0,
            "wdl_delayed_hazard_cue_without_active_count": 0.0,
            "wdl_delayed_hazard_active_without_prior_cue_count": 0.0,
            "wdl_cue_seen_agents": 0.0,
            "wdl_cue_exposed_next_move_away_count": 0.0,
            "wdl_cue_exposed_next_move_toward_count": 0.0,
            "wdl_cue_exposed_stayed_count": 0.0,
            "wdl_pre_hazard_exit_count": 0.0,
            "wdl_hazard_enter_count": 0.0,
            "wdl_hazard_avoidance_proxy": 0.0,
            "wdl_hazard_exposure_after_cue_count": 0.0,
            "wdl_hazard_damage_by_lineage_json": "{}",
            "wdl_cue_exposure_by_lineage_json": "{}",
            "wdl_hazard_exposure_by_lineage_json": "{}",
            **zero_path1_drh0_metrics(),
            "prediction_demand_enable": 0,
            "prediction_demand_observation_enable": 0,
            "prediction_demand_cue_cell_count": 0.0,
            "prediction_demand_active_cell_count": 0.0,
            "prediction_demand_event_count": 0.0,
            "prediction_demand_transition_count": 0.0,
            "prediction_demand_cue_ambiguity_mean": 0.0,
            "prediction_demand_cue_to_consequence_delay_mean": 0.0,
            "prediction_demand_cue_without_consequence_count": 0.0,
            "prediction_demand_consequence_without_prior_cue_count": 0.0,
            "prediction_demand_pre_consequence_agent_context_count": 0.0,
            "prediction_demand_consequence_agent_exposure_count": 0.0,
            "prediction_demand_resource_consequence_total": 0.0,
            "prediction_demand_direct_prediction_reward_active": 0,
            "prediction_demand_direct_error_reward_active": 0,
            "delayed_resource_depletion_enable": 0,
            "delayed_resource_depletion_cue_cell_count": 0.0,
            "delayed_resource_depletion_hidden_gap_cell_count": 0.0,
            "delayed_resource_depletion_active_cell_count": 0.0,
            "delayed_resource_depletion_event_count": 0.0,
            "delayed_resource_depletion_agent_cue_exposure_count": 0.0,
            "delayed_resource_depletion_agent_hidden_gap_exposure_count": 0.0,
            "delayed_resource_depletion_agent_active_exposure_count": 0.0,
            "delayed_resource_depletion_consequence_count": 0.0,
            "delayed_resource_depletion_consequence_total": 0.0,
            "delayed_resource_depletion_gather_cue_count": 0.0,
            "delayed_resource_depletion_gather_hidden_gap_count": 0.0,
            "delayed_resource_depletion_gather_active_count": 0.0,
            "delayed_resource_depletion_revisit_after_cue_count": 0.0,
            "delayed_resource_depletion_avoidance_proxy": 0.0,
            "delayed_crowding_waste_enable": 0,
            "delayed_crowding_waste_cue_cell_count": 0.0,
            "delayed_crowding_waste_hidden_gap_cell_count": 0.0,
            "delayed_crowding_waste_active_cell_count": 0.0,
            "delayed_crowding_waste_event_count": 0.0,
            "delayed_crowding_waste_agent_cue_exposure_count": 0.0,
            "delayed_crowding_waste_agent_hidden_gap_exposure_count": 0.0,
            "delayed_crowding_waste_agent_active_exposure_count": 0.0,
            "delayed_crowding_waste_consequence_count": 0.0,
            "delayed_crowding_waste_consequence_total": 0.0,
            "delayed_crowding_waste_gather_cue_count": 0.0,
            "delayed_crowding_waste_gather_hidden_gap_count": 0.0,
            "delayed_crowding_waste_gather_active_count": 0.0,
            "delayed_crowding_waste_revisit_after_cue_count": 0.0,
            "delayed_crowding_waste_avoidance_proxy": 0.0,
            "mem_max_delayed_choice_enable": 0,
            "mem_max_delayed_choice_observation_enable": 0,
            "mem_max_delayed_choice_event_count": 0.0,
            "mem_max_delayed_choice_target_a_cue_count": 0.0,
            "mem_max_delayed_choice_target_b_cue_count": 0.0,
            "mem_max_delayed_choice_hidden_window_active": 0.0,
            "mem_max_delayed_choice_consequence_step_active": 0.0,
            "mem_max_delayed_choice_correct_arrival_count": 0.0,
            "mem_max_delayed_choice_incorrect_arrival_count": 0.0,
            "mem_max_delayed_choice_no_choice_count": 0.0,
            "mem_max_delayed_choice_correct_choice_rate": 0.0,
            "mem_max_delayed_choice_consequence_exposure_count": 0.0,
            "mem_max_delayed_choice_damage_total": 0.0,
            "mem_max_delayed_choice_damage_prevented_total": 0.0,
            "mem_max_delayed_choice_expected_consequence_total": 0.0,
            "mem_max_delayed_choice_applied_health_damage_total": 0.0,
            "mem_max_delayed_choice_applied_energy_cost_total": 0.0,
            "mem_max_delayed_choice_consequence_applied_count": 0.0,
            "mem_max_delayed_choice_consequence_eligible_count": 0.0,
            "mem_max_delayed_choice_duplicate_prevented_count": 0.0,
            "mem_max_delayed_choice_correct_health_damage_total": 0.0,
            "mem_max_delayed_choice_incorrect_health_damage_total": 0.0,
            "mem_max_delayed_choice_no_choice_health_damage_total": 0.0,
            "mem_max_delayed_choice_target_a_arrival_count": 0.0,
            "mem_max_delayed_choice_target_b_arrival_count": 0.0,
            "mem_max_delayed_choice_any_target_reachable_fraction": 0.0,
            "mem_max_delayed_choice_shortcut_exposure": 0,
            "mem_max_delayed_choice_behavior_metric_available": 0,
            "mem_max_delayed_choice_behavior_coupling_supported": 0,
            "mem_max_delayed_choice_direct_memory_reward_active": 0,
            "mem_max_delayed_choice_direct_cognition_reward_active": 0,
            "jungle_ecological_recovery_enable": 0,
            "patch_recovery_activity_mean": 0.0,
            "patch_recovery_activity_min": 0.0,
            "patch_recovery_activity_max": 0.0,
            "patch_recovery_activity_p10": 0.0,
            "patch_recovery_activity_p90": 0.0,
            "patch_recovery_activity_occupied_mean": 0.0,
            "patch_recovery_positive_total": 0.0,
            "patch_recovery_negative_total": 0.0,
            "patch_recovery_decay_total": 0.0,
            "jungle_recovery_regrowth_boost_total": 0.0,
            "jungle_recovery_regrowth_boost_mean": 0.0,
            "jungle_recovery_regrowth_boost_max": 0.0,
            "jungle_recovery_regrowth_boost_p90": 0.0,
            "patch_regrowth_after_waste_total": 0.0,
            "patch_regrowth_after_recovery_total": 0.0,
            "patch_recovery_added_regrowth_total": 0.0,
        }
        if bool(getattr(self.env, "patch_world_enable", False)):
            try:
                from env.gridworld_pkg.patch_world import (
                    PATCH_WORLD_OBSERVATION_SCHEMA_VERSION,
                    PATCH_WORLD_DESCRIPTOR_SCHEMA_VERSION,
                    LINEAGE_DIVERSITY_SCHEMA_VERSION,
                    lineage_branch_candidate_reporting_enabled,
                    lineage_branch_candidates,
                    lineage_diversity_reporting_enabled,
                    lineage_habitat_descriptors,
                    lineage_habitat_metrics,
                    lineage_strategy_descriptors,
                    lineage_strategy_descriptors_enabled,
                    lineage_strategy_metrics,
                    delayed_crowding_waste_metrics,
                    delayed_resource_depletion_metrics,
                    mem_max_delayed_choice_metrics,
                    patch_food_metrics,
                    patch_event_contexts_summary,
                    patch_observation_validation_metrics,
                    patch_occupied_descriptor_metrics,
                    patch_recovery_metrics,
                    patch_terrain_metrics,
                    patch_trace_density_metrics,
                    patch_world_local_observation_channel_count,
                    patch_world_descriptor_enabled,
                    patch_world_observation_enabled,
                    delayed_crowding_waste_metrics,
                    prediction_demand_metrics,
                    wdl_delayed_hazard_metrics,
                )
                from env.gridworld_pkg.jungle import jungle_predator_risk_metrics

                patch_metrics.update(patch_food_metrics(self.env))
                patch_metrics.update(patch_terrain_metrics(self.env))
                patch_metrics.update(patch_trace_density_metrics(self.env))
                patch_metrics.update(patch_recovery_metrics(self.env))
                patch_metrics.update(jungle_predator_risk_metrics(self.env))
                patch_metrics.update(wdl_delayed_hazard_metrics(self.env))
                patch_metrics.update(prediction_demand_metrics(self.env))
                patch_metrics.update(delayed_resource_depletion_metrics(self.env))
                patch_metrics.update(delayed_crowding_waste_metrics(self.env))
                patch_metrics.update(mem_max_delayed_choice_metrics(self.env))
                patch_metrics["patch_world_descriptor_schema_version"] = int(PATCH_WORLD_DESCRIPTOR_SCHEMA_VERSION)
                patch_metrics["patch_world_diagnostics_enable"] = int(
                    bool(getattr(self.env_cfg, "patch_world_diagnostics_enable", True))
                )
                patch_metrics["patch_world_lineage_descriptors_enable"] = int(
                    bool(getattr(self.env_cfg, "patch_world_lineage_descriptors_enable", True))
                )
                patch_metrics["lineage_diversity_schema_version"] = int(LINEAGE_DIVERSITY_SCHEMA_VERSION)
                patch_metrics["lineage_diversity_reporting_enable"] = int(
                    lineage_diversity_reporting_enabled(self.env_cfg)
                )
                patch_metrics["lineage_strategy_descriptors_enable"] = int(
                    lineage_strategy_descriptors_enabled(self.env_cfg)
                )
                patch_metrics["lineage_branch_candidate_reporting_enable"] = int(
                    lineage_branch_candidate_reporting_enabled(self.env_cfg)
                )
                patch_metrics["patch_world_waste_consequence_enable"] = int(
                    bool(getattr(self.env_cfg, "patch_world_waste_consequence_enable", False))
                )
                patch_metrics["jungle_reproduction_emissions_enable"] = int(
                    bool(getattr(self.env_cfg, "jungle_reproduction_emissions_enable", False))
                )
                patch_metrics["jungle_juvenile_predator_vulnerability_enable"] = int(
                    bool(getattr(self.env_cfg, "jungle_juvenile_predator_vulnerability_enable", False))
                )
                patch_metrics["jungle_terrain_protection_enable"] = int(
                    bool(getattr(self.env_cfg, "jungle_terrain_protection_enable", False))
                )
                try:
                    from env.gridworld_pkg.jungle import compute_terrain_damage_multiplier_map  # noqa: WPS433

                    dmg_mult_map = compute_terrain_damage_multiplier_map(self.env)
                    occupied = np.asarray(getattr(self.env, "agent_grid"), dtype=np.int32) != -1
                    if dmg_mult_map.size and bool(np.any(occupied)):
                        patch_metrics["occupied_predator_damage_multiplier_mean"] = float(
                            np.mean(dmg_mult_map[occupied])
                        )
                except Exception:
                    pass
                if patch_world_descriptor_enabled(self.env_cfg):
                    patch_metrics.update(patch_occupied_descriptor_metrics(self.env))
                    patch_metrics.update(
                        patch_event_contexts_summary(
                            "birth",
                            list(getattr(self, "_patch_birth_contexts_gen", []) or []),
                        )
                    )
                    patch_metrics.update(
                        patch_event_contexts_summary(
                            "death",
                            list(getattr(self, "_patch_death_contexts_gen", []) or []),
                        )
                    )
                    patch_metrics.update(
                        patch_event_contexts_summary(
                            "predator_event",
                            list(getattr(self, "_jungle_predator_event_contexts_gen", []) or []),
                        )
                    )
                    patch_metrics.update(
                        patch_event_contexts_summary(
                            "jungle_birth_emission",
                            list(getattr(self, "_jungle_birth_emission_contexts_gen", []) or []),
                        )
                    )
                    if bool(getattr(self.env_cfg, "patch_world_lineage_descriptors_enable", True)):
                        birth_contexts = list(getattr(self, "_patch_birth_contexts_gen", []) or [])
                        death_contexts = list(getattr(self, "_patch_death_contexts_gen", []) or [])
                        predator_contexts = list(getattr(self, "_jungle_predator_event_contexts_gen", []) or [])
                        descriptors = lineage_habitat_descriptors(self.env, alive_after)
                        patch_metrics.update(lineage_habitat_metrics(descriptors))
                        patch_metrics["lineage_habitat_descriptors_json"] = json.dumps(
                            descriptors,
                            sort_keys=True,
                            separators=(",", ":"),
                            ensure_ascii=True,
                        )
                        if lineage_strategy_descriptors_enabled(self.env_cfg):
                            strategy_descriptors = lineage_strategy_descriptors(
                                self.env,
                                alive_after,
                                birth_contexts=birth_contexts,
                                death_contexts=death_contexts,
                                predator_event_contexts=predator_contexts,
                                cfg=self.env_cfg,
                            )
                            strategy_metrics = lineage_strategy_metrics(strategy_descriptors, cfg=self.env_cfg)
                            candidates = lineage_branch_candidates(strategy_descriptors, cfg=self.env_cfg)
                            patch_metrics.update(strategy_metrics)
                            patch_metrics["lineage_branch_candidate_count"] = float(len(candidates))
                            patch_metrics["lineage_strategy_descriptors_json"] = json.dumps(
                                strategy_descriptors,
                                sort_keys=True,
                                separators=(",", ":"),
                                ensure_ascii=True,
                            )
                            patch_metrics["lineage_branch_candidates_json"] = json.dumps(
                                candidates,
                                sort_keys=True,
                                separators=(",", ":"),
                                ensure_ascii=True,
                            )
                    patch_metrics.update(patch_observation_validation_metrics(self.env))
                patch_metrics["patch_observation_enable"] = int(
                    bool(patch_world_observation_enabled(self.env_cfg))
                )
                patch_metrics["patch_observation_schema_version"] = int(PATCH_WORLD_OBSERVATION_SCHEMA_VERSION)
                patch_metrics["patch_observation_channel_count"] = int(
                    patch_world_local_observation_channel_count(self.env_cfg)
                )
                patch_metrics["patch_observation_size"] = int(getattr(self, "_obs_size", 0) or 0)
            except Exception:
                pass
            patch_metrics["patch_food_regrowth_total"] = float(getattr(self, "_patch_food_regrowth_total_gen", 0.0))
            patch_metrics["patch_regrowth_base_total"] = float(
                getattr(self, "_patch_regrowth_base_total_gen", 0.0)
            )
            patch_metrics["patch_regrowth_effective_total"] = float(
                getattr(self, "_patch_regrowth_effective_total_gen", 0.0)
            )
            patch_metrics["patch_regrowth_suppressed_total"] = float(
                getattr(self, "_patch_regrowth_suppressed_total_gen", 0.0)
            )
            regrowth_steps = int(getattr(self, "_patch_regrowth_consequence_steps_gen", 0) or 0)
            if regrowth_steps > 0:
                patch_metrics["patch_regrowth_multiplier_mean"] = (
                    float(getattr(self, "_patch_regrowth_multiplier_mean_sum_gen", 0.0)) / float(regrowth_steps)
                )
                patch_metrics["patch_regrowth_multiplier_min"] = float(
                    getattr(self, "_patch_regrowth_multiplier_min_gen", 1.0)
                )
                patch_metrics["patch_regrowth_multiplier_p10"] = (
                    float(getattr(self, "_patch_regrowth_multiplier_p10_sum_gen", 0.0)) / float(regrowth_steps)
                )
                patch_metrics["patch_regrowth_multiplier_p90"] = (
                    float(getattr(self, "_patch_regrowth_multiplier_p90_sum_gen", 0.0)) / float(regrowth_steps)
                )
                patch_metrics["patch_waste_regrowth_penalty_mean"] = (
                    float(getattr(self, "_patch_waste_regrowth_penalty_mean_sum_gen", 0.0)) / float(regrowth_steps)
                )
                patch_metrics["patch_density_regrowth_penalty_mean"] = (
                    float(getattr(self, "_patch_density_regrowth_penalty_mean_sum_gen", 0.0)) / float(regrowth_steps)
                )
                patch_metrics["patch_high_waste_regrowth_suppression_frac"] = (
                    float(getattr(self, "_patch_high_waste_regrowth_suppression_frac_sum_gen", 0.0)) / float(regrowth_steps)
                )
                patch_metrics["patch_high_density_regrowth_suppression_frac"] = (
                    float(getattr(self, "_patch_high_density_regrowth_suppression_frac_sum_gen", 0.0)) / float(regrowth_steps)
                )
                patch_metrics["jungle_recovery_regrowth_boost_mean"] = (
                    float(getattr(self, "_jungle_recovery_regrowth_boost_mean_sum_gen", 0.0)) / float(regrowth_steps)
                )
                patch_metrics["jungle_recovery_regrowth_boost_p90"] = (
                    float(getattr(self, "_jungle_recovery_regrowth_boost_p90_sum_gen", 0.0)) / float(regrowth_steps)
                )
            patch_metrics["patch_waste_regrowth_penalty_max"] = float(
                getattr(self, "_patch_waste_regrowth_penalty_max_gen", 0.0)
            )
            patch_metrics["patch_density_regrowth_penalty_max"] = float(
                getattr(self, "_patch_density_regrowth_penalty_max_gen", 0.0)
            )
            patch_metrics["patch_recovery_positive_total"] = float(
                getattr(self, "_patch_recovery_positive_total_gen", 0.0)
            )
            patch_metrics["patch_recovery_negative_total"] = float(
                getattr(self, "_patch_recovery_negative_total_gen", 0.0)
            )
            patch_metrics["patch_recovery_decay_total"] = float(
                getattr(self, "_patch_recovery_decay_total_gen", 0.0)
            )
            patch_metrics["jungle_recovery_regrowth_boost_total"] = float(
                getattr(self, "_jungle_recovery_regrowth_boost_total_gen", 0.0)
            )
            patch_metrics["jungle_recovery_regrowth_boost_max"] = float(
                getattr(self, "_jungle_recovery_regrowth_boost_max_gen", 0.0)
            )
            patch_metrics["patch_regrowth_after_waste_total"] = float(
                getattr(self, "_patch_regrowth_after_waste_total_gen", 0.0)
            )
            patch_metrics["patch_regrowth_after_recovery_total"] = float(
                getattr(self, "_patch_regrowth_after_recovery_total_gen", 0.0)
            )
            patch_metrics["patch_recovery_added_regrowth_total"] = float(
                getattr(self, "_patch_recovery_added_regrowth_total_gen", 0.0)
            )
            patch_metrics["patch_food_harvest_total"] = float(getattr(self, "_patch_food_harvest_total_gen", 0.0))
            patch_metrics["patch_movement_energy_extra_total"] = float(
                getattr(self, "_patch_movement_energy_extra_total_gen", 0.0)
            )
            move_count = int(getattr(self, "_patch_movement_energy_event_count_gen", 0) or 0)
            if move_count > 0:
                patch_metrics["patch_movement_energy_extra_mean"] = (
                    float(getattr(self, "_patch_movement_energy_extra_total_gen", 0.0)) / float(move_count)
                )
            patch_metrics["patch_scent_emitted_total"] = float(
                getattr(self, "_patch_scent_emitted_total_gen", 0.0)
            )
            patch_metrics["patch_scent_decayed_total"] = float(
                getattr(self, "_patch_scent_decayed_total_gen", 0.0)
            )
            patch_metrics["patch_waste_emitted_total"] = float(
                getattr(self, "_patch_waste_emitted_total_gen", 0.0)
            )
            patch_metrics["patch_waste_decayed_total"] = float(
                getattr(self, "_patch_waste_decayed_total_gen", 0.0)
            )
            predator_events = int(getattr(self, "_jungle_predator_events_gen", 0) or 0)
            predator_exposures = int(getattr(self, "_jungle_predator_exposure_count_gen", 0) or 0)
            predator_damage = float(getattr(self, "_jungle_predator_damage_total_gen", 0.0))
            predator_damage_raw = float(getattr(self, "_jungle_predator_damage_raw_total_gen", predator_damage))
            predator_damage_effective = float(
                getattr(self, "_jungle_predator_damage_effective_total_gen", predator_damage)
            )
            predator_damage_prevented = float(getattr(self, "_jungle_predator_damage_prevented_total_gen", 0.0))
            patch_metrics["jungle_predator_events"] = float(predator_events)
            patch_metrics["jungle_predator_event_rate"] = (
                float(predator_events) / float(predator_exposures) if predator_exposures > 0 else 0.0
            )
            patch_metrics["jungle_predator_damage_total"] = predator_damage
            patch_metrics["jungle_predator_damage_raw_total"] = predator_damage_raw
            patch_metrics["jungle_predator_damage_effective_total"] = predator_damage_effective
            patch_metrics["jungle_predator_damage_prevented_total"] = predator_damage_prevented
            patch_metrics["jungle_predator_damage_mean"] = (
                predator_damage / float(predator_events) if predator_events > 0 else 0.0
            )
            multipliers = np.asarray(
                list(getattr(self, "_jungle_predator_damage_multipliers_gen", []) or []),
                dtype=np.float32,
            )
            if multipliers.size:
                patch_metrics["jungle_predator_damage_multiplier_mean"] = float(np.mean(multipliers))
                patch_metrics["jungle_predator_damage_multiplier_min"] = float(np.min(multipliers))
                patch_metrics["jungle_predator_damage_multiplier_p10"] = float(np.percentile(multipliers, 10))
                patch_metrics["jungle_predator_damage_multiplier_p90"] = float(np.percentile(multipliers, 90))
                patch_metrics["jungle_predator_event_damage_multiplier_mean"] = float(np.mean(multipliers))
                patch_metrics["jungle_predator_event_damage_prevented_mean"] = (
                    predator_damage_prevented / float(predator_events) if predator_events > 0 else 0.0
                )
            patch_metrics["jungle_predator_deaths"] = float(
                getattr(self, "_jungle_predator_deaths_gen", 0)
            )
            wdl_steps = int(getattr(self, "_wdl_delayed_hazard_tick_count_gen", 0) or 0)
            wdl_active_exposures = int(getattr(self, "_wdl_delayed_hazard_agent_active_exposure_count_gen", 0) or 0)
            wdl_damage_total = float(getattr(self, "_wdl_delayed_hazard_damage_total_gen", 0.0))
            wdl_delay_count = int(getattr(self, "_wdl_delayed_hazard_cue_to_active_delay_count_gen", 0) or 0)
            wdl_delay_sum = float(getattr(self, "_wdl_delayed_hazard_cue_to_active_delay_sum_gen", 0.0))
            patch_metrics["wdl_delayed_hazard_enable"] = int(
                bool(getattr(self.env_cfg, "wdl_delayed_hazard_enable", False))
            )
            patch_metrics["wdl_delayed_hazard_cue_cell_count"] = (
                float(getattr(self, "_wdl_delayed_hazard_cue_cell_count_sum_gen", 0.0)) / float(wdl_steps)
                if wdl_steps > 0
                else 0.0
            )
            patch_metrics["wdl_delayed_hazard_active_cell_count"] = (
                float(getattr(self, "_wdl_delayed_hazard_active_cell_count_sum_gen", 0.0)) / float(wdl_steps)
                if wdl_steps > 0
                else 0.0
            )
            patch_metrics["wdl_delayed_hazard_event_count"] = float(
                getattr(self, "_wdl_delayed_hazard_event_count_gen", 0)
            )
            patch_metrics["wdl_delayed_hazard_agent_cue_exposure_count"] = float(
                getattr(self, "_wdl_cue_seen_agents_gen", 0)
            )
            patch_metrics["wdl_delayed_hazard_agent_active_exposure_count"] = float(wdl_active_exposures)
            patch_metrics["wdl_delayed_hazard_damage_total"] = wdl_damage_total
            patch_metrics["wdl_delayed_hazard_damage_mean"] = (
                wdl_damage_total / float(wdl_active_exposures) if wdl_active_exposures > 0 else 0.0
            )
            patch_metrics["wdl_delayed_hazard_damage_max"] = float(
                getattr(self, "_wdl_delayed_hazard_damage_max_gen", 0.0)
            )
            patch_metrics["wdl_delayed_hazard_agents_damaged"] = float(
                getattr(self, "_wdl_delayed_hazard_agents_damaged_gen", 0)
            )
            patch_metrics["wdl_delayed_hazard_cue_to_active_delay_mean"] = (
                wdl_delay_sum / float(wdl_delay_count) if wdl_delay_count > 0 else 0.0
            )
            patch_metrics["wdl_delayed_hazard_cue_without_active_count"] = float(
                getattr(self, "_wdl_delayed_hazard_cue_without_active_count_gen", 0)
            )
            patch_metrics["wdl_delayed_hazard_active_without_prior_cue_count"] = float(
                getattr(self, "_wdl_delayed_hazard_active_without_prior_cue_count_gen", 0)
            )
            patch_metrics["wdl_cue_seen_agents"] = float(getattr(self, "_wdl_cue_seen_agents_gen", 0))
            away = int(getattr(self, "_wdl_cue_exposed_next_move_away_count_gen", 0) or 0)
            toward = int(getattr(self, "_wdl_cue_exposed_next_move_toward_count_gen", 0) or 0)
            stayed = int(getattr(self, "_wdl_cue_exposed_stayed_count_gen", 0) or 0)
            patch_metrics["wdl_cue_exposed_next_move_away_count"] = float(away)
            patch_metrics["wdl_cue_exposed_next_move_toward_count"] = float(toward)
            patch_metrics["wdl_cue_exposed_stayed_count"] = float(stayed)
            patch_metrics["wdl_pre_hazard_exit_count"] = float(getattr(self, "_wdl_pre_hazard_exit_count_gen", 0))
            patch_metrics["wdl_hazard_enter_count"] = float(getattr(self, "_wdl_hazard_enter_count_gen", 0))
            patch_metrics["wdl_hazard_avoidance_proxy"] = float(away) / float(max(1, away + toward + stayed))
            patch_metrics["wdl_hazard_exposure_after_cue_count"] = float(
                getattr(self, "_wdl_hazard_exposure_after_cue_count_gen", 0)
            )
            patch_metrics["path1_drh0_damage_in_refuge"] = float(
                getattr(self, "_path1_drh0_damage_in_refuge_gen", 0.0)
            )
            patch_metrics["path1_drh0_damage_outside_refuge"] = float(
                getattr(self, "_path1_drh0_damage_outside_refuge_gen", 0.0)
            )
            patch_metrics["path1_drh0_damage_prevented_by_refuge"] = float(
                getattr(self, "_path1_drh0_damage_prevented_by_refuge_gen", 0.0)
            )
            patch_metrics["path1_drh0_static_refuge_damage_prevented"] = float(
                getattr(self, "_path1_drh0_static_refuge_damage_prevented_gen", 0.0)
            )
            patch_metrics["path1_drh0_cue_conditioned_damage_prevented"] = float(
                getattr(self, "_path1_drh0_cue_conditioned_damage_prevented_gen", 0.0)
            )
            patch_metrics["path1_drh0_no_cue_false_positive_refuge_movement"] = float(
                getattr(self, "_path1_drh0_no_cue_false_positive_refuge_movement_gen", 0)
            )
            patch_metrics["path1_drh0_cue_visible_steps"] = float(
                getattr(self, "_path1_drh0_cue_visible_steps_gen", 0)
            )
            patch_metrics["path1_drh0_hidden_window_steps"] = float(
                getattr(self, "_path1_drh0_hidden_window_steps_gen", 0)
            )
            patch_metrics["path1_drh0_hazard_active_steps"] = float(
                getattr(self, "_path1_drh0_hazard_active_steps_gen", 0)
            )
            patch_metrics.update(path1_drh0_generation_metrics(self.env, generation_metrics=patch_metrics))
            try:
                by_lineage: dict[str, float] = {}
                for ctx in list(getattr(self, "_wdl_delayed_hazard_event_contexts_gen", []) or []):
                    lid = str(int(float(ctx.get("lineage_id", 0.0) or 0.0)))
                    by_lineage[lid] = by_lineage.get(lid, 0.0) + float(ctx.get("wdl_damage", 0.0) or 0.0)
                patch_metrics["wdl_hazard_damage_by_lineage_json"] = json.dumps(
                    by_lineage,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=True,
                )
                patch_metrics["wdl_hazard_exposure_by_lineage_json"] = json.dumps(
                    {k: 1.0 for k in sorted(by_lineage.keys())},
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=True,
                )
                patch_metrics["wdl_cue_exposure_by_lineage_json"] = "{}"
            except Exception:
                pass
            prediction_steps = int(getattr(self, "_prediction_demand_tick_count_gen", 0) or 0)
            prediction_delay_count = int(
                getattr(self, "_prediction_demand_cue_to_consequence_delay_count_gen", 0) or 0
            )
            prediction_delay_sum = float(
                getattr(self, "_prediction_demand_cue_to_consequence_delay_sum_gen", 0.0)
            )
            prediction_ambiguity_count = int(getattr(self, "_prediction_demand_cue_ambiguity_count_gen", 0) or 0)
            prediction_ambiguity_sum = float(getattr(self, "_prediction_demand_cue_ambiguity_sum_gen", 0.0))
            patch_metrics["prediction_demand_enable"] = int(
                bool(getattr(self.env_cfg, "prediction_demand_enable", False))
            )
            patch_metrics["prediction_demand_observation_enable"] = int(
                bool(getattr(self.env_cfg, "prediction_demand_observation_enable", False))
                and bool(getattr(self.env_cfg, "prediction_demand_enable", False))
            )
            patch_metrics["prediction_demand_cue_cell_count"] = (
                float(getattr(self, "_prediction_demand_cue_cell_count_sum_gen", 0.0)) / float(prediction_steps)
                if prediction_steps > 0
                else 0.0
            )
            patch_metrics["prediction_demand_active_cell_count"] = (
                float(getattr(self, "_prediction_demand_active_cell_count_sum_gen", 0.0)) / float(prediction_steps)
                if prediction_steps > 0
                else 0.0
            )
            patch_metrics["prediction_demand_event_count"] = float(
                getattr(self, "_prediction_demand_event_count_gen", 0)
            )
            patch_metrics["prediction_demand_transition_count"] = float(
                getattr(self, "_prediction_demand_transition_count_gen", 0)
            )
            patch_metrics["prediction_demand_cue_ambiguity_mean"] = (
                prediction_ambiguity_sum / float(prediction_ambiguity_count)
                if prediction_ambiguity_count > 0
                else 0.0
            )
            patch_metrics["prediction_demand_cue_to_consequence_delay_mean"] = (
                prediction_delay_sum / float(prediction_delay_count) if prediction_delay_count > 0 else 0.0
            )
            patch_metrics["prediction_demand_cue_without_consequence_count"] = float(
                getattr(self, "_prediction_demand_cue_without_consequence_count_gen", 0)
            )
            patch_metrics["prediction_demand_consequence_without_prior_cue_count"] = float(
                getattr(self, "_prediction_demand_consequence_without_prior_cue_count_gen", 0)
            )
            patch_metrics["prediction_demand_pre_consequence_agent_context_count"] = float(
                getattr(self, "_prediction_demand_pre_consequence_agent_context_count_gen", 0)
            )
            patch_metrics["prediction_demand_consequence_agent_exposure_count"] = float(
                getattr(self, "_prediction_demand_consequence_agent_exposure_count_gen", 0)
            )
            patch_metrics["prediction_demand_resource_consequence_total"] = float(
                getattr(self, "_prediction_demand_resource_consequence_total_gen", 0.0)
            )
            patch_metrics["prediction_demand_direct_prediction_reward_active"] = 0
            patch_metrics["prediction_demand_direct_error_reward_active"] = 0
            for _prefix, _enabled_name in (
                ("delayed_resource_depletion", "delayed_resource_depletion_enable"),
                ("delayed_crowding_waste", "delayed_crowding_waste_enable"),
            ):
                _steps = int(getattr(self, f"_{_prefix}_tick_count_gen", 0) or 0)
                _delay_count = int(getattr(self, f"_{_prefix}_cue_to_active_delay_count_gen", 0) or 0)
                _delay_sum = float(getattr(self, f"_{_prefix}_cue_to_active_delay_sum_gen", 0.0))
                patch_metrics[f"{_prefix}_enable"] = int(bool(getattr(self.env_cfg, _enabled_name, False)))
                patch_metrics[f"{_prefix}_cue_cell_count"] = (
                    float(getattr(self, f"_{_prefix}_cue_cell_count_sum_gen", 0.0)) / float(_steps)
                    if _steps > 0
                    else 0.0
                )
                patch_metrics[f"{_prefix}_hidden_gap_cell_count"] = (
                    float(getattr(self, f"_{_prefix}_hidden_cell_count_sum_gen", 0.0)) / float(_steps)
                    if _steps > 0
                    else 0.0
                )
                patch_metrics[f"{_prefix}_active_cell_count"] = (
                    float(getattr(self, f"_{_prefix}_active_cell_count_sum_gen", 0.0)) / float(_steps)
                    if _steps > 0
                    else 0.0
                )
                patch_metrics[f"{_prefix}_event_count"] = float(getattr(self, f"_{_prefix}_event_count_gen", 0))
                patch_metrics[f"{_prefix}_agent_cue_exposure_count"] = float(
                    getattr(self, f"_{_prefix}_agent_cue_exposure_count_gen", 0)
                )
                patch_metrics[f"{_prefix}_agent_hidden_gap_exposure_count"] = float(
                    getattr(self, f"_{_prefix}_agent_hidden_gap_exposure_count_gen", 0)
                )
                patch_metrics[f"{_prefix}_agent_active_exposure_count"] = float(
                    getattr(self, f"_{_prefix}_agent_active_exposure_count_gen", 0)
                )
                patch_metrics[f"{_prefix}_consequence_count"] = float(
                    getattr(self, f"_{_prefix}_consequence_count_gen", 0)
                )
                patch_metrics[f"{_prefix}_consequence_total"] = float(
                    getattr(self, f"_{_prefix}_consequence_total_gen", 0.0)
                )
                patch_metrics[f"{_prefix}_gather_cue_count"] = float(
                    getattr(self, f"_{_prefix}_gather_cue_count_gen", 0)
                )
                patch_metrics[f"{_prefix}_gather_hidden_gap_count"] = float(
                    getattr(self, f"_{_prefix}_gather_hidden_count_gen", 0)
                )
                patch_metrics[f"{_prefix}_gather_active_count"] = float(
                    getattr(self, f"_{_prefix}_gather_active_count_gen", 0)
                )
                patch_metrics[f"{_prefix}_revisit_after_cue_count"] = float(
                    getattr(self, f"_{_prefix}_revisit_after_cue_count_gen", 0)
                )
                _away = int(getattr(self, f"_{_prefix}_move_away_count_gen", 0) or 0)
                _toward = int(getattr(self, f"_{_prefix}_move_toward_count_gen", 0) or 0)
                _stay = int(getattr(self, f"_{_prefix}_stay_count_gen", 0) or 0)
                patch_metrics[f"{_prefix}_avoidance_proxy"] = float(_away) / float(max(1, _away + _toward + _stay))
            birth_emissions = int(getattr(self, "_jungle_birth_emission_count_gen", 0) or 0)
            patch_metrics["jungle_birth_scent_emitted_total"] = float(
                getattr(self, "_jungle_birth_scent_emitted_total_gen", 0.0)
            )
            patch_metrics["jungle_birth_waste_emitted_total"] = float(
                getattr(self, "_jungle_birth_waste_emitted_total_gen", 0.0)
            )
            patch_metrics["jungle_birth_emission_count"] = float(birth_emissions)
            juvenile_events = int(getattr(self, "_jungle_juvenile_predator_events_gen", 0) or 0)
            adult_events = int(getattr(self, "_jungle_adult_predator_events_gen", 0) or 0)
            patch_metrics["jungle_juvenile_predator_events"] = float(juvenile_events)
            patch_metrics["jungle_juvenile_predator_damage_total"] = float(
                getattr(self, "_jungle_juvenile_predator_damage_total_gen", 0.0)
            )
            patch_metrics["jungle_juvenile_predator_deaths"] = float(
                getattr(self, "_jungle_juvenile_predator_deaths_gen", 0)
            )
            patch_metrics["jungle_adult_predator_events"] = float(adult_events)
            patch_metrics["jungle_adult_predator_damage_total"] = float(
                getattr(self, "_jungle_adult_predator_damage_total_gen", 0.0)
            )
            patch_metrics["jungle_juvenile_vs_adult_predator_event_ratio"] = (
                float(juvenile_events) / float(adult_events) if adult_events > 0 else float(juvenile_events)
            )

        if bool(getattr(self.env_cfg, "mem_max_delayed_choice_enable", False)):
            correct = float(getattr(self, "_mem_max_delayed_choice_correct_arrival_count_gen", 0) or 0)
            incorrect = float(getattr(self, "_mem_max_delayed_choice_incorrect_arrival_count_gen", 0) or 0)
            no_choice = float(getattr(self, "_mem_max_delayed_choice_no_choice_count_gen", 0) or 0)
            choice_denom = correct + incorrect + no_choice
            reach_denom = float(getattr(self, "_mem_max_delayed_choice_reachability_denominator_gen", 0) or 0)
            reachable = float(getattr(self, "_mem_max_delayed_choice_reachable_agents_gen", 0) or 0)
            patch_metrics.update(
                {
                    "mem_max_delayed_choice_enable": 1,
                    "mem_max_delayed_choice_observation_enable": int(
                        bool(getattr(self.env_cfg, "mem_max_delayed_choice_observation_enable", False))
                    ),
                    "mem_max_delayed_choice_event_count": float(
                        getattr(self, "_mem_max_delayed_choice_event_count_gen", 0) or 0
                    ),
                    "mem_max_delayed_choice_target_a_cue_count": float(
                        getattr(self, "_mem_max_delayed_choice_target_a_cue_count_gen", 0) or 0
                    ),
                    "mem_max_delayed_choice_target_b_cue_count": float(
                        getattr(self, "_mem_max_delayed_choice_target_b_cue_count_gen", 0) or 0
                    ),
                    "mem_max_delayed_choice_hidden_window_active": float(
                        getattr(self, "_mem_max_delayed_choice_hidden_step_active_gen", 0) or 0
                    ),
                    "mem_max_delayed_choice_consequence_step_active": float(
                        getattr(self, "_mem_max_delayed_choice_consequence_step_active_gen", 0) or 0
                    ),
                    "mem_max_delayed_choice_correct_arrival_count": correct,
                    "mem_max_delayed_choice_incorrect_arrival_count": incorrect,
                    "mem_max_delayed_choice_no_choice_count": no_choice,
                    "mem_max_delayed_choice_correct_choice_rate": correct / choice_denom if choice_denom > 0.0 else 0.0,
                    "mem_max_delayed_choice_consequence_exposure_count": float(
                        getattr(self, "_mem_max_delayed_choice_consequence_exposure_count_gen", 0) or 0
                    ),
                    "mem_max_delayed_choice_damage_total": float(
                        getattr(self, "_mem_max_delayed_choice_damage_total_gen", 0.0) or 0.0
                    ),
                    "mem_max_delayed_choice_damage_prevented_total": float(
                        getattr(self, "_mem_max_delayed_choice_damage_prevented_total_gen", 0.0) or 0.0
                    ),
                    "mem_max_delayed_choice_expected_consequence_total": float(
                        getattr(self, "_mem_max_delayed_choice_expected_consequence_total_gen", 0.0) or 0.0
                    ),
                    "mem_max_delayed_choice_applied_health_damage_total": float(
                        getattr(self, "_mem_max_delayed_choice_applied_health_damage_total_gen", 0.0) or 0.0
                    ),
                    "mem_max_delayed_choice_applied_energy_cost_total": float(
                        getattr(self, "_mem_max_delayed_choice_applied_energy_cost_total_gen", 0.0) or 0.0
                    ),
                    "mem_max_delayed_choice_consequence_applied_count": float(
                        getattr(self, "_mem_max_delayed_choice_consequence_applied_count_gen", 0) or 0
                    ),
                    "mem_max_delayed_choice_consequence_eligible_count": float(
                        getattr(self, "_mem_max_delayed_choice_consequence_eligible_count_gen", 0) or 0
                    ),
                    "mem_max_delayed_choice_duplicate_prevented_count": float(
                        getattr(self, "_mem_max_delayed_choice_duplicate_prevented_count_gen", 0) or 0
                    ),
                    "mem_max_delayed_choice_correct_health_damage_total": float(
                        getattr(self, "_mem_max_delayed_choice_correct_health_damage_total_gen", 0.0) or 0.0
                    ),
                    "mem_max_delayed_choice_incorrect_health_damage_total": float(
                        getattr(self, "_mem_max_delayed_choice_incorrect_health_damage_total_gen", 0.0) or 0.0
                    ),
                    "mem_max_delayed_choice_no_choice_health_damage_total": float(
                        getattr(self, "_mem_max_delayed_choice_no_choice_health_damage_total_gen", 0.0) or 0.0
                    ),
                    "mem_max_delayed_choice_target_a_arrival_count": float(
                        getattr(self, "_mem_max_delayed_choice_target_a_arrival_count_gen", 0) or 0
                    ),
                    "mem_max_delayed_choice_target_b_arrival_count": float(
                        getattr(self, "_mem_max_delayed_choice_target_b_arrival_count_gen", 0) or 0
                    ),
                    "mem_max_delayed_choice_any_target_reachable_fraction": (
                        reachable / reach_denom if reach_denom > 0.0 else 0.0
                    ),
                    "mem_max_delayed_choice_shortcut_exposure": int(
                        bool(getattr(self.env_cfg, "mem_max_delayed_choice_shortcut_truth_enable", False))
                    ),
                    "mem_max_delayed_choice_behavior_metric_available": 1,
                    "mem_max_delayed_choice_behavior_coupling_supported": 1,
                    "mem_max_delayed_choice_direct_memory_reward_active": 0,
                    "mem_max_delayed_choice_direct_cognition_reward_active": 0,
                }
            )

        behavior_proxy_metrics = {}
        try:
            if hasattr(self.behavior_logger, "generation_proxy_metrics"):
                behavior_proxy_metrics = dict(self.behavior_logger.generation_proxy_metrics())
        except Exception:
            behavior_proxy_metrics = {}

        if bool(getattr(self.env_cfg, "respec_n1_zone_telemetry_enable", False)):
            from simulation.respec_n1_zone_telemetry import finish_generation

            finish_generation(
                self,
                generation=int(gen),
                alive_after=alive_after,
                eligible_parent_agents=eligible_parent_agents,
                selected_parent_agents=selected_parent_agents,
                birth_contexts=list(getattr(self, "_lineage_reproduction_funnel_birth_contexts_gen", []) or [])
                or list(getattr(self, "_patch_birth_contexts_gen", []) or []),
            )

        # Standard metrics row
        # -----------------------------
        native_public_output_counts = getattr(self, "_native_runtime_public_output_counts", None)
        policy_cache_public_counts = (
            native_public_output_counts
            if isinstance(native_public_output_counts, dict)
            and native_public_output_counts.get("source") == "native_compact_result"
            else getattr(self, "_reference_policy_cache_public_counts", None)
        )
        self.logger.log_generation(
            generation=gen,
            agents=alive_after,
            deaths_this_gen=deaths_this_gen,
            successful_births=successful_births,
            failed_births=failed_births,
            injury_blocked_births=self._injury_blocked_births_gen,
            cohort_size=cohort_size,
            cohort_survivors=cohort_survivors,
            cohort_survival_rate=cohort_survival_rate,
            shock_active=int(shock),
            idle_charge_total=getattr(self, "_idle_charge_total_gen", 0.0),
            idle_charge_deaths=getattr(self, "_idle_charge_deaths_gen", 0),
            peak_injury=gen_peak_injury,
            birth_attempts=self._birth_attempts_gen,
            birth_pre_cooldown=self._birth_pre_cooldown_gen,
            birth_pre_no_energy=self._birth_pre_no_energy_gen,
            birth_pre_injury_gate=self._birth_pre_injury_gate_gen,
            birth_pre_no_space=self._birth_pre_no_space_gen,
            birth_pre_no_mate=self._birth_pre_no_mate_gen,
            birth_attempt_conception_fail=self._birth_attempt_conception_fail_gen,
            birth_gestation_fail=self._birth_gestation_fail_gen,
            birth_make_child_fail=self._birth_make_child_fail_gen,
            birth_fail_no_space=self._birth_fail_no_space_gen,
            birth_fail_no_mate=self._birth_fail_no_mate_gen,
            birth_fail_buffer=self._birth_fail_buffer_gen,
            birth_fail_injury_gate=self._birth_fail_injury_gate_gen,
            birth_fail_scar_gate=self._birth_fail_scar_gate_gen,
            birth_fail_conception=self._birth_fail_conception_gen,
            birth_fail_gestation=self._birth_fail_gestation_gen,
            selected_parent_agents=selected_parent_agents,
            tier3_overcrowd_mean=tier3_overcrowd_mean,
            tier3_overcrowd_max=tier3_overcrowd_max,
            tier3_crossings_per_step=tier3_crossings_per_step,
            tier3_cross_events_per_step=tier3_cross_events_per_step,
            tier3_extra_move_cost_mean=tier3_extra_move_cost_mean,
            tier3_extra_move_cost_max=tier3_extra_move_cost_max,
            tier4_juvenile_exposure_steps=tier4_juvenile_exposure_steps,
            tier4_adult_exposure_steps=tier4_adult_exposure_steps,
            tier4_juvenile_deaths=tier4_juvenile_deaths,
            tier4_adult_deaths=tier4_adult_deaths,
            tier4_juvenile_death_rate_per_1k_steps=tier4_juvenile_death_rate_per_1k_steps,
            tier4_adult_death_rate_per_1k_steps=tier4_adult_death_rate_per_1k_steps,
            tier4_juvenile_fraction_mean=tier4_juvenile_fraction_mean,
            tier4_fraction_reproduced_alive_end=tier4_fraction_reproduced_alive_end,
            tier4_first_repro_step_mean_reproducers_alive_end=tier4_first_repro_step_mean_reproducers_alive_end,
            tier4_first_repro_step_p50_reproducers_alive_end=tier4_first_repro_step_p50_reproducers_alive_end,
            tier4_first_repro_step_p90_reproducers_alive_end=tier4_first_repro_step_p90_reproducers_alive_end,
            tier4_bins_steps_json=tier4_bins_steps_json,
            tier4_alive_age_hist_json=tier4_alive_age_hist_json,
            tier4_deaths_age_hist_json=tier4_deaths_age_hist_json,
            **patch_metrics,
            dev_stage_transition_count=int(getattr(self, "_dev_stage_transition_count_gen", 0) or 0),
            dev_stage_eval_stage_counts_json=json.dumps(
                dict(sorted((getattr(self, "_dev_stage_eval_stage_counts_gen", {}) or {}).items(), key=lambda kv: str(kv[0]))),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            ),
            memory_runtime_telemetry=(
                self._memory_runtime_telemetry_summary()
                if hasattr(self, "_memory_runtime_telemetry_summary")
                else {}
            ),
            behavior_proxy_metrics=behavior_proxy_metrics,
            birth_contexts=list(getattr(self, "_patch_birth_contexts_gen", []) or []),
            death_contexts=list(getattr(self, "_patch_death_contexts_gen", []) or []),
            predator_event_contexts=list(getattr(self, "_jungle_predator_event_contexts_gen", []) or []),
            native_public_output_counts=native_public_output_counts,
            native_public_metrics_overlay=getattr(self, "_native_runtime_public_metrics_overlay", None),
            policy_cache_public_counts=policy_cache_public_counts,
            sim_cfg=self.sim_cfg,
            env_cfg=self.env_cfg,
        )
        try:
            chronicle = getattr(self, "scientific_chronicle", None)
            if chronicle is not None:
                native_lifecycle_rows = []
                native_engine = getattr(self, "_native_eiel_engine", None)
                if native_engine is not None:
                    try:
                        from native_runtime.lifecycle_events import (
                            typed_lifecycle_events_from_projection,
                            typed_lifecycle_events_to_rows,
                        )

                        native_projection = native_engine.export_checkpoint_projection()
                        native_events = typed_lifecycle_events_from_projection(native_projection)
                        native_output = dict(
                            getattr(self, "_native_runtime_typed_lifecycle_public_output", {}) or {}
                        )
                        native_lifecycle_rows = typed_lifecycle_events_to_rows(
                            native_events,
                            start=int(native_output.get("start_cursor", 0) or 0),
                            stop=int(native_output.get("end_cursor", 0) or 0),
                        )
                    except Exception as exc:
                        setattr(self, "_scientific_chronicle_native_source_error", f"{type(exc).__name__}: {exc}")
                chronicle.observe_generation(
                    generation=int(gen),
                    step=max(0, int(getattr(self.sim_cfg, "steps_per_generation", 1) or 1) - 1),
                    alive_agents=alive_after,
                    deaths=int(deaths_this_gen),
                    births=int(successful_births),
                    failed_births=int(failed_births),
                    cohort_size=int(cohort_size),
                    cohort_survival_fraction=float(cohort_survival_rate),
                    patch_metrics=patch_metrics,
                    birth_contexts=list(getattr(self, "_patch_birth_contexts_gen", []) or []),
                    death_contexts=list(getattr(self, "_patch_death_contexts_gen", []) or []),
                    predator_contexts=list(getattr(self, "_jungle_predator_event_contexts_gen", []) or []),
                    wdl_contexts=list(getattr(self, "_wdl_delayed_hazard_event_contexts_gen", []) or []),
                    native_lifecycle_rows=native_lifecycle_rows,
                    native_public_output_counts=dict(
                        getattr(self, "_native_runtime_public_output_counts", {}) or {}
                    ),
                )
        except Exception as exc:
            # Chronicle is observability-only and must never interrupt mechanics.
            setattr(self, "_scientific_chronicle_last_error", f"{type(exc).__name__}: {exc}")
        try:
            gen1_f1 = getattr(self, "gen1_f1_capture", None)
            if gen1_f1 is not None:
                gen1_f1.observe_carriers(generation=int(gen), agents=alive_after)
        except Exception as exc:
            setattr(self, "_gen1_f1_capture_last_error", f"{type(exc).__name__}: {exc}")
        try:
            from simulation.mosaic_fixed1_controller import write_generation_summary

            write_generation_summary(self, generation=int(gen))
        except Exception:
            if bool(getattr(self.sim_cfg, "mosaic_fixed1_probe_enable", False)):
                raise
        try:
            from simulation.predrisk_fixed1_probe import write_generation_summary

            write_generation_summary(self, generation=int(gen))
        except Exception:
            if bool(getattr(self.sim_cfg, "predrisk_fixed1_probe_enable", False)):
                raise
        try:
            from simulation.habacclim1 import write_generation_summary

            write_generation_summary(self, generation=int(gen))
        except Exception:
            if bool(getattr(self.sim_cfg, "habacclim1_telemetry_enable", False)):
                raise
        try:
            report_result = maybe_write_birth_skew_mechanism_report(
                run_dir=getattr(self, "run_dir", None),
                generation=gen,
                sim_cfg=self.sim_cfg,
                selected_parent_agents=selected_parent_agents,
                alive_agents=alive_after,
                birth_contexts=list(getattr(self, "_patch_birth_contexts_gen", []) or []),
                state=getattr(self, "_birth_skew_mechanism_state", None),
            )
            if report_result is not None:
                self._birth_skew_mechanism_state = report_result.get("state") or {}
        except Exception as e:
            if gen < 3:
                print(f"[gen {gen}] birth-skew mechanism reporting failed: {e}")

        # -----------------------------
        # Step 3: developmental coupling analysis (optional, analysis only)
        # -----------------------------
        if bool(getattr(self.sim_cfg, "devcoupling_enable", False)):
            try:
                self.logger.log_devcoupling(
                    generation=gen,
                    agents=alive_after,
                    selected_parent_agents=selected_parent_agents,
                    tier3_overcrowd_mean=tier3_overcrowd_mean,
            tier3_overcrowd_max=tier3_overcrowd_max,
            tier3_crossings_per_step=tier3_crossings_per_step,
            tier3_cross_events_per_step=tier3_cross_events_per_step,
            tier3_extra_move_cost_mean=tier3_extra_move_cost_mean,
            tier3_extra_move_cost_max=tier3_extra_move_cost_max,
            tier4_juvenile_exposure_steps=tier4_juvenile_exposure_steps,
            tier4_adult_exposure_steps=tier4_adult_exposure_steps,
            tier4_juvenile_deaths=tier4_juvenile_deaths,
            tier4_adult_deaths=tier4_adult_deaths,
            tier4_juvenile_death_rate_per_1k_steps=tier4_juvenile_death_rate_per_1k_steps,
            tier4_adult_death_rate_per_1k_steps=tier4_adult_death_rate_per_1k_steps,
            tier4_juvenile_fraction_mean=tier4_juvenile_fraction_mean,
            tier4_fraction_reproduced_alive_end=tier4_fraction_reproduced_alive_end,
            tier4_first_repro_step_mean_reproducers_alive_end=tier4_first_repro_step_mean_reproducers_alive_end,
            tier4_first_repro_step_p50_reproducers_alive_end=tier4_first_repro_step_p50_reproducers_alive_end,
            tier4_first_repro_step_p90_reproducers_alive_end=tier4_first_repro_step_p90_reproducers_alive_end,
            tier4_bins_steps_json=tier4_bins_steps_json,
            tier4_alive_age_hist_json=tier4_alive_age_hist_json,
            tier4_deaths_age_hist_json=tier4_deaths_age_hist_json,
            sim_cfg=self.sim_cfg,
                )
            except Exception as e:
                # Never crash the sim due to optional analysis logging.
                if gen < 3:
                    print(f"[gen {gen}] devcoupling logging failed: {e}")
