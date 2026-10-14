from __future__ import annotations

"""
simulation/runner_pkg/runner_ticks_env_apply.py

Environment commit for a single tick.

Responsibilities:
- Call env.step_agents_batch for all alive agents in a vectorized batch
- Apply runner-level extra costs (e.g. GRU upkeep)
- Persist results back into Agent objects (energy/health/injury/scar)
- Cache deltas onto agent.stats so downstream modules (drives, etc.) can read them
"""

import time
import os
import numpy as np

from agents.chemistry_coupling import action_cost_multiplier_from_chem
from agents.resource_handling import RESOURCE_HANDLING_AXIS_FIELD, clamp_resource_handling_axis
from perf.cxxcore_env_apply_shadow import (
    compare_shadow_deltas,
    compute_env_apply_shadow_python,
    config_from_env,
    observed_delta_from_python,
    supported_for_first_slice,
    update_shadow_summary,
    write_shadow_summary,
)
from perf.cxxcore11_live_runner_transaction_hook import (
    config_from_sim_cfg as cxxcore11_config_from_sim_cfg,
    observed_materialized_from_live_python as cxxcore11_observed_materialized_from_live_python,
    prepare_live_tick_inputs as cxxcore11_prepare_live_tick_inputs,
    result_payload as cxxcore11_result_payload,
    run_live_runner_transaction_hook as cxxcore11_run_live_runner_transaction_hook,
    validation_rng_payload_from_actions as cxxcore11_validation_rng_payload_from_actions,
)
from perf.cxxrewrite2_native_tick import (
    native_tick_requested as cxxrewrite2_native_tick_requested,
    prepare_runner_shadow_snapshot as cxxrewrite2_prepare_runner_shadow_snapshot,
    run_runner_shadow_compare as cxxrewrite2_run_runner_shadow_compare,
    update_native_tick_summary as cxxrewrite2_update_native_tick_summary,
    write_native_tick_summary as cxxrewrite2_write_native_tick_summary,
)
from perf.cxxrewrite3_resident_frame import (
    commit_post_env_arrays as cxxrewrite3_commit_post_env_arrays,
    resident_frame_requested as cxxrewrite3_resident_frame_requested,
    stage_pre_env_native_entry as cxxrewrite3_stage_pre_env_native_entry,
)
from perf.cxxrewrite4_upstream_resident import (
    record_post_env_scalar_arrays as cxxrewrite4_record_post_env_scalar_arrays,
    resident_scalar_sync_fail_closed as cxxrewrite4_resident_scalar_sync_fail_closed,
    sync_agents_scalar_state as cxxrewrite4_sync_agents_scalar_state,
)
from .runner_action_opportunity import record_action_opportunities, record_policy_action_execution_consequences
from .runner_ticks_common import clamp01


class TickEnvApplyMixin:
    def _apply_env_tick(
        self,
        alive,
        agent_ids_np,
        actions_np,
        buckets,
        deaths_this_gen: int,
        gen_peak_injury: float,
        pre_scalar_arrays=None,
    ):
        """
        Apply the environment tick for the current alive set.

        Args:
            alive: List of alive agents.
            agent_ids_np: np.int32 array of agent ids aligned to `alive`.
            actions_np: action array aligned to `alive`.
            buckets: perf timing buckets dict.
            deaths_this_gen: current death count for this generation.
            gen_peak_injury: current peak injury observed this generation.

        Returns:
            (deaths_this_gen, gen_peak_injury)
        """

        if (
            bool(getattr(self.sim_cfg, "recomb72_audit_enable", False))
            and bool(
                getattr(
                    self.sim_cfg,
                    "recomb72_neutral_environment_enable",
                    False,
                )
            )
        ):
            # The RECOMB audit executes every real runner tick but freezes the
            # environment commit so ecology, energy, health, and audited traits
            # cannot acquire selection authority.
            for agent in alive:
                stats = getattr(agent, "stats", None)
                if stats is not None:
                    stats.last_energy_delta = 0.0
                    stats.last_health_delta = 0.0
                    stats.last_food_gained = 0.0
            return int(deaths_this_gen), float(gen_peak_injury)

        # ------------------------------------------------------------
        # Phase 3 determinism: ensure alive/actions alignment (fast path)
        # ------------------------------------------------------------
        # Notes (why this exists)
        # -----------------------
        # - `_run_ticks` may shuffle `alive` for fairness.
        # - `GridWorld.step_agents_batch(...)` already canonicalizes its *internal*
        #   apply order by `agent_id`, so we do NOT need to sort here for determinism.
        # - What *must* be true is that (`agent_ids_np`, `actions_np`) remain aligned
        #   to the `alive` list. On rare resume/debug paths we have seen misalignment.
        #
        # Fast path:
        # - If aligned, do nothing (no dict builds, no sorts).
        # Slow path (best-effort repair):
        # - Rebuild `agent_ids_np` from `alive` and re-align `actions_np` by agent_id.
        try:
            _alive_ids = np.fromiter(
                (int(getattr(a, "agent_id")) for a in alive),
                dtype=np.int32,
                count=len(alive),
            )
            _aid_in = np.asarray(agent_ids_np, dtype=np.int32) if agent_ids_np is not None else _alive_ids
            _act_in = np.asarray(actions_np, dtype=np.int32)

            # Only attempt a repair when the *shapes* match and the ordering differs.
            if _aid_in.shape[0] == _alive_ids.shape[0] and not np.array_equal(_aid_in, _alive_ids):
                if _aid_in.shape[0] == _act_in.shape[0] and _aid_in.shape[0] > 0:
                    _action_by_id = {int(_aid_in[i]): int(_act_in[i]) for i in range(_aid_in.shape[0])}
                    try:
                        actions_np = np.fromiter(
                            (_action_by_id[int(aid)] for aid in _alive_ids),
                            dtype=np.int32,
                            count=int(_alive_ids.shape[0]),
                        )
                        agent_ids_np = _alive_ids
                    except Exception:
                        # If lookup fails (missing ids), keep the original ordering.
                        pass
        except Exception:
            # Never break the simulation for an alignment repair.
            pass

        t_apply0 = time.perf_counter()
        _diag = getattr(self, "env_hotpath_expression_diagnostics", None)
        _cxxcore_shadow_enabled = bool(getattr(self.sim_cfg, "cxxcore_env_apply_shadow_enable", False))
        _cxxcore_shadow_snapshot = None
        _cxxcore_shadow_record = None
        _cxxrewrite2_native_enabled = cxxrewrite2_native_tick_requested(self.sim_cfg)
        _cxxrewrite2_native_snapshot = None
        _cxxrewrite2_native_record = None
        _cxxrewrite3_resident_enabled = cxxrewrite3_resident_frame_requested(self.sim_cfg)
        _cxxcore11_enabled = bool(getattr(self.sim_cfg, "cxxcore11_live_runner_hook_enable", False))
        _cxxcore11_inputs = None
        _cxxcore11_prepare_diag = {}
        _cxxcore11_rng_payload = None
        try:
            self.env.action_consequence_telemetry_enable = bool(
                getattr(self.sim_cfg, "structural_action_consequence_telemetry_reporting_enable", False)
                or getattr(self.sim_cfg, "structural_policy_action_family_mapping_reporting_enable", False)
            )
        except Exception:
            pass


        n = len(alive)
        if pre_scalar_arrays is not None:
            pre_energy = np.asarray(pre_scalar_arrays["energy"], dtype=np.float32).reshape(-1).copy()
            pre_health = np.asarray(pre_scalar_arrays["health"], dtype=np.float32).reshape(-1).copy()
            pre_injury = np.asarray(pre_scalar_arrays["injury"], dtype=np.float32).reshape(-1).copy()
            pre_scar = np.asarray(pre_scalar_arrays["scar"], dtype=np.float32).reshape(-1).copy()
            expected = (int(n),)
            if pre_energy.shape != expected or pre_health.shape != expected or pre_injury.shape != expected or pre_scar.shape != expected:
                raise ValueError("pre_scalar_arrays must be aligned to alive")
        else:
            pre_energy = np.fromiter((float(a.energy) for a in alive), dtype=np.float32, count=n)
            pre_health = np.fromiter((float(a.health) for a in alive), dtype=np.float32, count=n)
            pre_injury = np.fromiter((float(getattr(a, "injury", 0.0)) for a in alive), dtype=np.float32, count=n)
            pre_scar = np.fromiter((float(getattr(a, "scar", 0.0)) for a in alive), dtype=np.float32, count=n)

        energies = pre_energy.copy()
        healths = pre_health.copy()
        injuries = pre_injury.copy()
        scars = pre_scar.copy()
        try:
            _telemetry_agent_ids = np.asarray(agent_ids_np, dtype=np.int64).reshape((-1,))
            _telemetry_pre_pos_y = np.asarray(getattr(self.env, "agent_pos_y"), dtype=np.int32)[_telemetry_agent_ids].copy()
            _telemetry_pre_pos_x = np.asarray(getattr(self.env, "agent_pos_x"), dtype=np.int32)[_telemetry_agent_ids].copy()
        except Exception:
            _telemetry_agent_ids = np.asarray([], dtype=np.int64)
            _telemetry_pre_pos_y = np.asarray([], dtype=np.int32)
            _telemetry_pre_pos_x = np.asarray([], dtype=np.int32)


        # === Tier 3 change: validation random walk override ===
        # Validation-only plumbing check: force random movement so Tier 3 wiring
        # (movement cost + border crossings) is exercised deterministically.
        if bool(getattr(self.env_cfg, "migration_pressure_enable", False)) and bool(getattr(self.env_cfg, "tier3_validation_force_random_walk", False)):
            # Replace actions with random move actions (0..3). We keep dtype int32.
            # P0-A determinism enforcement: never create per-tick RNG islands.
            # Route through RNGManager policy stream.
            actions_np = self.rng_manager.policy_np.integers(0, 4, size=actions_np.shape[0], dtype=np.int32)
        # === Tier 3 change: end validation random walk override ===

                # ------------------------------------------------------------
        # Action-cost multipliers (capability coupling + life-history tradeoff)
        # ------------------------------------------------------------
        # The environment supports a per-agent multiplier applied to *all* action
        # costs (movement, gather, rest, attack). We use this for:
        # - Part B: adult predictive coherence -> higher costs when incoherent
        # - Phase 4: identity drift -> higher costs when drifting
        # - Step 6: late-life coherence efficiency -> slightly LOWER costs when coherent (post-repro only)
        #
        # All effects are deterministic and gated by life-stage. Nothing touches
        # episode-0 selection pressure directly.
        move_cost_mults = None

        disable_pred_pressure = bool(getattr(self.sim_cfg, "disable_prediction_error_selection_pressure", False))
        adult_enable = bool(getattr(self.sim_cfg, "adult_move_penalty_enable", False)) and not disable_pred_pressure
        adult_strength = float(getattr(self.sim_cfg, "adult_move_penalty_strength", 0.0))
        adult_start = int(getattr(self.sim_cfg, "adult_move_penalty_start_step", int(getattr(self.sim_cfg, "late_pred_energy_tax_start_step", int(getattr(self.sim_cfg, "episode0_k", 0))))))
        if adult_start < 0:
            adult_start = 0
        adult_scale = float(getattr(self.sim_cfg, "adult_move_penalty_scale", float(getattr(self.sim_cfg, "late_pred_energy_tax_cap", 5.0))))
        if adult_scale <= 1e-9:
            adult_scale = 5.0
        adult_cap = float(getattr(self.sim_cfg, "adult_move_penalty_cap", 3.0))
        if adult_cap < 1.0:
            adult_cap = 1.0

        id_enable = bool(getattr(self.sim_cfg, "identity_drift_enable", False))
        id_strength = float(getattr(self.sim_cfg, "identity_drift_strength", 0.0))
        id_start = int(getattr(self.sim_cfg, "identity_drift_start_step", adult_start))
        if id_start < 0:
            id_start = 0
        id_scale = float(getattr(self.sim_cfg, "identity_drift_scale", 0.10))
        if id_scale <= 1e-9:
            id_scale = 0.10
        id_cap = float(getattr(self.sim_cfg, "identity_drift_cap", 3.0))
        if id_cap < 1.0:
            id_cap = 1.0

        late_eff_enable = bool(getattr(self.sim_cfg, "late_coherence_eff_enable", False)) and not disable_pred_pressure
        late_eff_w = float(getattr(self.sim_cfg, "late_coherence_eff_weight", 0.0))
        late_eff_scale = float(getattr(self.sim_cfg, "late_coherence_eff_scale", float(getattr(self.sim_cfg, "late_pred_energy_tax_cap", 5.0))))
        if late_eff_scale <= 1e-9:
            late_eff_scale = 5.0
        late_eff_min_count = int(getattr(self.sim_cfg, "late_coherence_eff_min_count", 5))
        if late_eff_min_count < 0:
            late_eff_min_count = 0
        late_eff_floor = float(getattr(self.sim_cfg, "late_coherence_eff_floor", 0.50))
        if late_eff_floor <= 0.0:
            late_eff_floor = 0.01

        # -----------------------------
        # Tier 4: Juvenile spatial vulnerability (age-based, no hard repro bans)
        # -----------------------------
        tier4_enable = bool(getattr(self.env_cfg, "tier4_juvenile_vulnerability_enable", False))
        t4_steps = int(getattr(self.env_cfg, "tier4_juvenile_steps", 0) or 0)
        t4_grace = int(getattr(self.env_cfg, "tier4_juvenile_grace_steps", 0) or 0)
        t4_steps = max(0, t4_steps)
        t4_grace = max(0, t4_grace)

        t4_hazard_mult = float(getattr(self.env_cfg, "tier4_juvenile_hazard_multiplier", 1.0))
        t4_move_penalty = float(getattr(self.env_cfg, "tier4_juvenile_movement_penalty", 1.0))
        t4_harvest_mult = float(getattr(self.env_cfg, "tier4_juvenile_harvest_multiplier", 1.0))

        # Arrays are only required if Tier 4 is enabled, juvenile window is non-empty,
        # and at least one multiplier differs from neutral.
        need_t4_arrays = bool(tier4_enable and (t4_steps > 0) and ((t4_hazard_mult != 1.0) or (t4_harvest_mult != 1.0)))

        chemistry_action_cost_enable = bool(getattr(self.sim_cfg, "chemistry_functional_coupling_enable", False)) and bool(
            getattr(self.sim_cfg, "chemistry_action_cost_coupling_enable", False)
        )
        need_mults = (n > 0) and (
            (adult_enable and adult_strength > 0.0)
            or (id_enable and id_strength > 0.0)
            or (late_eff_enable and late_eff_w > 0.0)
            or (tier4_enable and (t4_steps > 0) and (t4_move_penalty != 1.0))
            or chemistry_action_cost_enable
        )


        # Tier 4 per-tick exposure accumulation + per-agent arrays (passed into env step).
        juvenile_hazard_mults = None
        juvenile_harvest_mults = None

        if tier4_enable and (t4_steps > 0):
            # Exposure denominators are defined as sum over steps of alive juveniles/adults at that step.
            if not hasattr(self, "_tier4_steps_gen"):
                self._tier4_steps_gen = 0
                self._tier4_juvenile_exposure_steps_gen = 0
                self._tier4_adult_exposure_steps_gen = 0
                self._tier4_juvenile_deaths_gen = 0
                self._tier4_adult_deaths_gen = 0
                self._tier4_juvenile_frac_sum_gen = 0.0
                self._tier4_deaths_age_bins_gen = None  # set lazily when bins exist

            self._tier4_steps_gen += 1

            if need_t4_arrays:
                juvenile_hazard_mults = np.ones((n,), dtype=np.float32)
                juvenile_harvest_mults = np.ones((n,), dtype=np.float32)

            # Count juveniles/adults and fill arrays (single pass).
            juveniles_this_step = 0
            for i, a in enumerate(alive):
                try:
                    st = a.stats
                except Exception:
                    continue
                age = int(getattr(st, "survival_steps", 0))
                is_juvenile = (age < t4_steps)
                if is_juvenile:
                    juveniles_this_step += 1
                    if juvenile_harvest_mults is not None:
                        juvenile_harvest_mults[i] = np.float32(t4_harvest_mult)
                    if juvenile_hazard_mults is not None:
                        # Grace: suppress hazard amplification for newborn window only.
                        if age < t4_grace:
                            juvenile_hazard_mults[i] = np.float32(1.0)
                        else:
                            juvenile_hazard_mults[i] = np.float32(t4_hazard_mult)

            adults_this_step = int(n - juveniles_this_step)
            self._tier4_juvenile_exposure_steps_gen += int(juveniles_this_step)
            self._tier4_adult_exposure_steps_gen += int(max(0, adults_this_step))
            if n > 0:
                self._tier4_juvenile_frac_sum_gen += float(juveniles_this_step) / float(n)

        if need_mults:
            mults = np.ones((n,), dtype=np.float32)
            for i, a in enumerate(alive):
                try:
                    st = a.stats
                except Exception:
                    continue

                age = int(getattr(st, "survival_steps", 0))
                m = 1.0

                # Tier 4: juvenile movement penalty (age effect). Applied multiplicatively once.
                if tier4_enable and (t4_steps > 0) and (t4_move_penalty != 1.0) and (age < t4_steps):
                    m *= float(t4_move_penalty)

                # (1) Adult pred-err penalty (>=1.0)
                if adult_enable and (adult_strength > 0.0) and age >= adult_start:
                    try:
                        ema = float(getattr(st, "adult_pred_err_ema", 0.0))
                        coh = 1.0 - clamp01(ema / adult_scale)  # 1=coherent
                        m_adult = 1.0 + adult_strength * (1.0 - coh)
                        if m_adult > adult_cap:
                            m_adult = adult_cap
                        m *= m_adult
                    except Exception:
                        pass

                # (2) Identity drift penalty (>=1.0)
                if id_enable and (id_strength > 0.0) and age >= id_start:
                    try:
                        slot = int(getattr(a, "slot", -1))
                        tok = int(getattr(a, "slot_token", 0))
                        if (
                            slot >= 0
                            and hasattr(self, "_idprobe_baseline_token")
                            and self._idprobe_baseline_token is not None
                            and hasattr(self, "_idprobe_drift_ema")
                            and self._idprobe_drift_ema is not None
                        ):
                            if int(self._idprobe_baseline_token[slot]) == tok:
                                d = float(self._idprobe_drift_ema[slot])
                                dm = 1.0 + id_strength * clamp01(d / id_scale)
                                if dm > id_cap:
                                    dm = id_cap
                                m *= dm
                                try:
                                    st.identity_drift_ema = float(d)
                                except Exception:
                                    pass
                    except Exception:
                        pass

                # (3) Late-life coherence efficiency bonus (<=1.0, post-repro only)
                # This lever creates a minimal life-history tradeoff:
                # - Early repro can ignore later coherence but pays Option C penalty (selection-time).
                # - Coherent adults gain a small cumulative efficiency edge.
                eff_mult = 1.0
                coh_score = 0.0
                if late_eff_enable and (late_eff_w > 0.0):
                    try:
                        fr = getattr(st, "first_repro_step", None)
                        if fr is not None and age > int(fr):
                            lcount = int(getattr(st, "late_pred_err_count", 0))
                            if lcount >= late_eff_min_count:
                                lsum = float(getattr(st, "late_pred_err_sum", 0.0))
                                lmean = float(lsum) / float(max(1, lcount))
                                coh_score = 1.0 - clamp01(lmean / late_eff_scale)  # 1=coherent
                                eff_mult = 1.0 - float(late_eff_w) * float(coh_score)
                                if eff_mult < late_eff_floor:
                                    eff_mult = late_eff_floor
                    except Exception:
                        eff_mult = 1.0
                        coh_score = 0.0

                m *= eff_mult

                # (4) A3 chemistry competence coupling (bounded >=1.0 action cost inflation)
                if chemistry_action_cost_enable:
                    _chem_t0 = time.perf_counter()
                    chem_mult = action_cost_multiplier_from_chem(
                        getattr(a, "chemistry", None),
                        min_mult=float(getattr(self.sim_cfg, "chemistry_action_cost_min_mult", 1.0)),
                        max_mult=float(getattr(self.sim_cfg, "chemistry_action_cost_max_mult", 1.35)),
                        strength=float(getattr(self.sim_cfg, "chemistry_action_cost_strength", 0.9)),
                        power=float(getattr(self.sim_cfg, "chemistry_action_cost_power", 1.0)),
                    )
                    if _diag is not None:
                        try:
                            _diag.record_decomposition_bucket(
                                "env_apply_chemistry_object_total",
                                time.perf_counter() - _chem_t0,
                            )
                        except Exception:
                            pass
                    m *= float(chem_mult)
                    try:
                        st.chemistry_action_cost_mult = float(chem_mult)
                    except Exception:
                        pass
                else:
                    try:
                        st.chemistry_action_cost_mult = 1.0
                    except Exception:
                        pass

                # Cache for logging/debug (best-effort)
                try:
                    st.late_coherence_eff_mult = float(eff_mult)
                    st.late_coherence_score = float(coh_score)
                except Exception:
                    pass

                mults[i] = np.float32(m)

            move_cost_mults = mults


        if _cxxcore_shadow_enabled:
            _shadow_snapshot_t0 = time.perf_counter()
            _supported, _skip_reasons = supported_for_first_slice(self.env_cfg, self.sim_cfg)
            _cxxcore_shadow_record = {
                "schema_version": "cxxcore_env_apply_shadow_runtime_record_v1",
                "generation": int(
                    self._dev_generation_index()
                    if hasattr(self, "_dev_generation_index")
                    else int(getattr(self, "_eiel_fdp_cur_gen", getattr(self, "_current_gen", -1)) or -1)
                ),
                "tick": int(
                    self._dev_tick_index()
                    if hasattr(self, "_dev_tick_index")
                    else int(getattr(self, "_eiel_fdp_cur_tick", getattr(self, "tick", -1)) or -1)
                ),
                "agents": int(n),
                "cells": int(getattr(self.env_cfg, "height", 0) or 0) * int(getattr(self.env_cfg, "width", 0) or 0),
                "backend": str(getattr(self.sim_cfg, "cxxcore_env_apply_shadow_backend", "python") or "python"),
                "skipped": not bool(_supported),
                "skip_reasons": _skip_reasons,
                "parity": "not_checked",
                "timing_s": {
                    "wall": 0.0,
                    "python_env_apply": 0.0,
                    "snapshot_copy": 0.0,
                    "shadow_compute": 0.0,
                    "materialize": 0.0,
                    "compare": 0.0,
                    "diagnostics": 0.0,
                },
            }
            if _supported:
                try:
                    _agent_ids_shadow = np.asarray(agent_ids_np, dtype=np.int32)
                    _cxxcore_shadow_snapshot = {
                        "config": config_from_env(self.env),
                        "agent_ids": _agent_ids_shadow.copy(),
                        "actions": np.asarray(actions_np, dtype=np.int32).copy(),
                        "energies": energies.copy(),
                        "healths": healths.copy(),
                        "injuries": injuries.copy(),
                        "scars": scars.copy(),
                        "pos_y_by_id": np.asarray(getattr(self.env, "agent_pos_y"), dtype=np.int32).copy(),
                        "pos_x_by_id": np.asarray(getattr(self.env, "agent_pos_x"), dtype=np.int32).copy(),
                        "pre_pos_y": np.asarray(getattr(self.env, "agent_pos_y"), dtype=np.int32)[_agent_ids_shadow].copy(),
                        "pre_pos_x": np.asarray(getattr(self.env, "agent_pos_x"), dtype=np.int32)[_agent_ids_shadow].copy(),
                        "agent_grid": np.asarray(getattr(self.env, "agent_grid"), dtype=np.int32).copy(),
                        "grid": np.asarray(getattr(self.env, "grid"), dtype=np.int8).copy(),
                        "move_cost_mults": None if move_cost_mults is None else np.asarray(move_cost_mults, dtype=np.float32).copy(),
                        "juvenile_harvest_mults": (
                            None
                            if juvenile_harvest_mults is None
                            else np.asarray(juvenile_harvest_mults, dtype=np.float32).copy()
                        ),
                    }
                except Exception as exc:
                    _cxxcore_shadow_record["skipped"] = True
                    _cxxcore_shadow_record["skip_reasons"] = {"snapshot_exception": True, "detail": str(exc)}
                    _cxxcore_shadow_snapshot = None
            _cxxcore_shadow_record["timing_s"]["snapshot_copy"] = float(time.perf_counter() - _shadow_snapshot_t0)

        if _cxxrewrite3_resident_enabled:
            try:
                cxxrewrite3_stage_pre_env_native_entry(
                    self,
                    alive=alive,
                    agent_ids_np=agent_ids_np,
                    actions_np=actions_np,
                    energies=energies,
                    healths=healths,
                    injuries=injuries,
                    scars=scars,
                )
            except Exception:
                if bool(getattr(self.sim_cfg, "cxxrewrite_resident_frame_fail_closed", False)) or bool(
                    getattr(self.sim_cfg, "cxxrewrite_resident_frame_authoritative_enable", False)
                ) or bool(getattr(self.sim_cfg, "cxxrewrite_resident_frame_native_packing_authoritative_enable", False)):
                    raise

        if _cxxrewrite2_native_enabled:
            try:
                _cxxrewrite2_native_snapshot, _cxxrewrite2_native_record = cxxrewrite2_prepare_runner_shadow_snapshot(
                    self,
                    alive=alive,
                    agent_ids_np=agent_ids_np,
                    actions_np=actions_np,
                    energies=energies,
                    healths=healths,
                    injuries=injuries,
                    scars=scars,
                    move_cost_mults=move_cost_mults,
                    juvenile_harvest_mults=juvenile_harvest_mults,
                )
            except Exception:
                if bool(getattr(self.sim_cfg, "cxxrewrite_native_tick_fail_closed", False)) or bool(
                    getattr(self.sim_cfg, "cxxrewrite_native_tick_authoritative_enable", False)
                ):
                    raise
                _cxxrewrite2_native_snapshot = None
                _cxxrewrite2_native_record = {
                    "schema_version": "cxxrewrite2_native_tick_v1",
                    "skipped": True,
                    "skip_reasons": {"snapshot_exception": True},
                    "parity": "not_checked",
                    "timing_s": {},
                }

        if _cxxcore11_enabled:
            try:
                _cxxcore11_inputs, _cxxcore11_cache, _cxxcore11_prepare_diag = cxxcore11_prepare_live_tick_inputs(
                    env=self.env,
                    env_cfg=self.env_cfg,
                    sim_cfg=self.sim_cfg,
                    alive=alive,
                    agent_ids_np=agent_ids_np,
                    actions_np=actions_np,
                    energies=energies,
                    healths=healths,
                    injuries=injuries,
                    scars=scars,
                    move_cost_mults=move_cost_mults,
                    juvenile_harvest_mults=juvenile_harvest_mults,
                    cache=getattr(self, "_cxxcore11_live_runner_hook_cache", None),
                )
                self._cxxcore11_live_runner_hook_cache = _cxxcore11_cache
                if str(getattr(self.sim_cfg, "cxxcore11_live_runner_hook_rng_policy", "none") or "none") == "validation_action_override":
                    _cxxcore11_rng_payload = cxxcore11_validation_rng_payload_from_actions(
                        np.asarray(actions_np, dtype=np.int32).reshape(1, -1)
                    )
            except Exception as exc:
                if bool(getattr(self.sim_cfg, "cxxcore11_live_runner_hook_fail_closed", True)):
                    raise
                self._cxxcore11_live_runner_hook_last = {
                    "schema_version": "cxxcore11_live_runner_transaction_hook_v1",
                    "status": "fallback",
                    "fallback_reason": "input_capture_exception",
                    "exception": str(exc),
                }
                _cxxcore11_inputs = None

        record_action_opportunities(
            self,
            alive=alive,
            agent_ids_np=agent_ids_np,
            actions_np=actions_np,
            energies=energies,
            healths=healths,
            injuries=injuries,
        )

        resource_handling_axes = None
        if bool(getattr(self.env_cfg, "respec0b_resource_specialization_enable", False)):
            resource_handling_axes = np.asarray(
                [
                    clamp_resource_handling_axis(
                        getattr(getattr(agent, "genome", None), RESOURCE_HANDLING_AXIS_FIELD, 0.0)
                    )
                    for agent in alive
                ],
                dtype=np.float32,
            )

        # FDP v3+: internal probe to isolate drift inside env.step_agents_batch (gen3/ticks 116-120 only).
        try:
            if os.environ.get("EIEL_DETERMINISM_PROBE", "0").strip() == "1":
                _g = int(getattr(self, "_eiel_fdp_cur_gen", -1))
                _t = int(getattr(self, "_eiel_fdp_cur_tick", -1))
                if _g == 3 and _t in (116, 117, 118, 119, 120):
                    from simulation.determinism_probe import maybe_write_probe as _maybe_write_probe  # noqa: WPS433
                    _maybe_write_probe(self, generation_index=_g, tag=f"tick{_t:03d}.env_batch_pre")
        except Exception:
            pass

        _env_batch_t0 = time.perf_counter()
        out = self.env.step_agents_batch(
            agent_ids=agent_ids_np,
            actions=actions_np.astype(np.int32, copy=False),
            energies=energies,
            healths=healths,
            injuries=injuries,
            scars=scars,
            move_cost_mults=move_cost_mults,
            juvenile_hazard_mults=juvenile_hazard_mults,
            juvenile_harvest_mults=juvenile_harvest_mults,
            resource_handling_axes=resource_handling_axes,
        )
        if _diag is not None:
            try:
                _diag.record_decomposition_bucket("env_apply_env_batch_total", time.perf_counter() - _env_batch_t0)
            except Exception:
                pass
        if _cxxcore_shadow_record is not None:
            _cxxcore_shadow_record["timing_s"]["python_env_apply"] = float(time.perf_counter() - _env_batch_t0)

        if isinstance(out, tuple) and len(out) == 6:
            died_mask, new_energy, new_health, food_gained, new_injury, new_scar = out
        else:
            died_mask, new_energy, new_health, food_gained = out  # type: ignore[misc]
            new_injury = injuries
            new_scar = scars

        try:
            from simulation.mosaic_fixed1_controller import record_post_env

            record_post_env(self, agent_ids=agent_ids_np, actions=actions_np, food_gained=food_gained)
        except Exception:
            if bool(getattr(self.sim_cfg, "mosaic_fixed1_probe_enable", False)):
                raise
        try:
            from simulation.predrisk_fixed1_probe import record_post_env

            record_post_env(self, agent_ids=agent_ids_np, actions=actions_np, food_gained=food_gained)
        except Exception:
            if bool(getattr(self.sim_cfg, "predrisk_fixed1_probe_enable", False)):
                raise

        if bool(getattr(self.env_cfg, "respec_n1_zone_telemetry_enable", False)):
            from simulation.respec_n1_zone_telemetry import record_tick

            record_tick(
                self,
                generation=int(self._dev_generation_index() if hasattr(self, "_dev_generation_index") else -1),
                tick=int(getattr(self, "_tick", getattr(self, "_pred_tick", 0)) or 0),
                alive=alive,
                died_mask=died_mask,
            )

        if _cxxcore11_enabled and _cxxcore11_inputs is not None:
            try:
                _cxxcore11_observed = cxxcore11_observed_materialized_from_live_python(
                    inputs=_cxxcore11_inputs,
                    env=self.env,
                    died_mask=died_mask,
                    new_energy=new_energy,
                    new_health=new_health,
                    new_injury=new_injury,
                    new_scar=new_scar,
                )
                _cxxcore11_result = cxxcore11_run_live_runner_transaction_hook(
                    _cxxcore11_inputs,
                    observed_python=_cxxcore11_observed,
                    config=cxxcore11_config_from_sim_cfg(self.sim_cfg),
                    native_backend=getattr(self, "_cxxcore11_live_hook_native_backend", None),
                    rng_payload=_cxxcore11_rng_payload,
                    inject_divergence=bool(getattr(self, "_cxxcore11_live_hook_inject_divergence", False)),
                    extra_diagnostics=_cxxcore11_prepare_diag,
                )
                self._cxxcore11_live_runner_hook_last = cxxcore11_result_payload(_cxxcore11_result)
                if _cxxcore11_result.apply_output_to_live_scalars and _cxxcore11_result.output is not None:
                    new_energy = np.asarray(_cxxcore11_result.output["new_energy"], dtype=np.float32).copy()
                    new_health = np.asarray(_cxxcore11_result.output["new_health"], dtype=np.float32).copy()
                    new_injury = np.asarray(_cxxcore11_result.output["new_injury"], dtype=np.float32).copy()
                    new_scar = np.asarray(_cxxcore11_result.output["new_scar"], dtype=np.float32).copy()
                    died_mask = np.asarray(_cxxcore11_result.output["died_mask"], dtype=bool).copy()
            except Exception as exc:
                if bool(getattr(self.sim_cfg, "cxxcore11_live_runner_hook_fail_closed", True)):
                    raise
                self._cxxcore11_live_runner_hook_last = {
                    "schema_version": "cxxcore11_live_runner_transaction_hook_v1",
                    "status": "fallback",
                    "fallback_reason": "hook_exception",
                    "exception": str(exc),
                }

        try:
            from simulation.habacclim1 import apply_post_env

            apply_post_env(
                self,
                alive=alive,
                agent_ids=agent_ids_np,
                actions=actions_np,
                new_energy=new_energy,
                food_gained=food_gained,
            )
        except Exception:
            if bool(getattr(self.env_cfg, "habacclim_enable", False)):
                raise

        # FDP v3+: internal probe to isolate drift inside env.step_agents_batch (gen3/ticks 116-120 only).
        try:
            if os.environ.get("EIEL_DETERMINISM_PROBE", "0").strip() == "1":
                _g = int(getattr(self, "_eiel_fdp_cur_gen", -1))
                _t = int(getattr(self, "_eiel_fdp_cur_tick", -1))
                if _g == 3 and _t in (116, 117, 118, 119, 120):
                    from simulation.determinism_probe import maybe_write_probe as _maybe_write_probe  # noqa: WPS433
                    _maybe_write_probe(self, generation_index=_g, tag=f"tick{_t:03d}.env_batch_post")
        except Exception:
            pass

        try:
            _telemetry_post_pos_y = np.asarray(getattr(self.env, "agent_pos_y"), dtype=np.int32)[_telemetry_agent_ids].copy()
            _telemetry_post_pos_x = np.asarray(getattr(self.env, "agent_pos_x"), dtype=np.int32)[_telemetry_agent_ids].copy()
        except Exception:
            _telemetry_post_pos_y = np.asarray([], dtype=np.int32)
            _telemetry_post_pos_x = np.asarray([], dtype=np.int32)
        record_policy_action_execution_consequences(
            self,
            alive=alive,
            agent_ids_np=agent_ids_np,
            actions_np=actions_np,
            pre_energy=pre_energy,
            pre_health=pre_health,
            pre_injury=pre_injury,
            pre_pos_y=_telemetry_pre_pos_y,
            pre_pos_x=_telemetry_pre_pos_x,
            post_pos_y=_telemetry_post_pos_y,
            post_pos_x=_telemetry_post_pos_x,
            new_energy=new_energy,
            new_health=new_health,
            food_gained=food_gained,
            new_injury=new_injury,
        )

        # === Tier 3 change: per-tick stat accumulation ===
        # The env exposes tier3_last_step_stats each tick.
        st = getattr(self.env, "tier3_last_step_stats", None) or {}
        if not hasattr(self, "_tier3_steps_gen"):
            self._tier3_steps_gen = 0
            self._tier3_overcrowd_mean_sum_gen = 0.0
            self._tier3_overcrowd_max_gen = 0.0
            self._tier3_crossings_total_gen = 0
            self._tier3_cross_events_total_gen = 0
            self._tier3_extra_move_cost_sum_gen = 0.0
            self._tier3_extra_move_cost_max_gen = 0.0

        self._tier3_steps_gen += 1
        self._tier3_overcrowd_mean_sum_gen += float(st.get("overcrowd_mean", 0.0))
        self._tier3_overcrowd_max_gen = max(float(self._tier3_overcrowd_max_gen), float(st.get("overcrowd_max", 0.0)))
        self._tier3_crossings_total_gen += int(st.get("crossings", 0))
        self._tier3_cross_events_total_gen += int(st.get("cross_events", 0))
        self._tier3_extra_move_cost_sum_gen += float(st.get("extra_move_cost_sum", 0.0))
        self._tier3_extra_move_cost_max_gen = max(float(self._tier3_extra_move_cost_max_gen), float(st.get("extra_move_cost_max", 0.0)))
        # === Tier 3 change: end per-tick stat accumulation ===

        if bool(getattr(self.env, "patch_world_enable", False)):
            self._patch_food_harvest_total_gen = float(
                getattr(self, "_patch_food_harvest_total_gen", 0.0)
            ) + float(getattr(self.env, "patch_food_harvest_total_last", 0.0))
            self._patch_movement_energy_extra_total_gen = float(
                getattr(self, "_patch_movement_energy_extra_total_gen", 0.0)
            ) + float(getattr(self.env, "patch_movement_energy_extra_last", 0.0))
            self._patch_movement_energy_event_count_gen = int(
                getattr(self, "_patch_movement_energy_event_count_gen", 0)
            ) + int(getattr(self.env, "patch_movement_energy_event_count_last", 0))

        # GRU memory upkeep cost (runner-level)
        gru_base = float(getattr(self.env_cfg, "gru_cost_base", 0.0))
        gru_per_dim = float(getattr(self.env_cfg, "gru_cost_per_dim", 0.0))
        if (gru_base > 0.0) or (gru_per_dim > 0.0):
            from agents.cognition import memory_cost_components  # noqa: WPS433

            _gru_upkeep_t0 = time.perf_counter()
            for i, agent in enumerate(alive):
                if not agent.alive:
                    continue
                g = getattr(agent, "genome", None)
                if g is None or not bool(getattr(g, "use_gru", False)):
                    continue
                parts = memory_cost_components(g, env_cfg=self.env_cfg, sim_cfg=self.sim_cfg)
                cost = float(parts.get("charged", 0.0))
                if cost > 0.0:
                    new_energy[i] = max(0.0, float(new_energy[i]) - cost)
            if _diag is not None:
                try:
                    _diag.record_decomposition_bucket("env_apply_gru_upkeep_total", time.perf_counter() - _gru_upkeep_t0)
                except Exception:
                    pass

        if bool(getattr(self.sim_cfg, "enable_experimental_perception_cost_charging", False)) or bool(
            getattr(self.sim_cfg, "enable_experimental_sensor_radius_cost_charging", False)
        ) or bool(
            getattr(self.sim_cfg, "enable_experimental_prediction_cost_charging", False)
        ) or bool(
            getattr(self.sim_cfg, "enable_experimental_developmental_cost_charging", False)
        ) or bool(
            getattr(self.sim_cfg, "enable_experimental_introspection_cost_charging", False)
        ) or bool(
            getattr(self.sim_cfg, "enable_experimental_self_regulation_cost_charging", False)
        ) or bool(
            getattr(self.sim_cfg, "enable_experimental_self_modification_cost_charging", False)
        ):
            try:
                from agents.cognition import (  # noqa: WPS433
                    NATIVE_BUDGET_AGGREGATION_SCHEMA_VERSION,
                    compute_cognitive_budget_ledger,
                    compute_cognitive_budget_ledgers_batch,
                    compute_env_apply_cognitive_cost_scalar,
                    compute_env_cost_charge_scalar,
                    compute_env_cost_charge_scalar_from_ledger,
                    compute_env_cost_charge_scalar_from_surface,
                    express_cost_charge_surface,
                    express_cognitive_phenotype,
                )
                from perf.costsurface_soa import CostSurfaceSoADiagnostic  # noqa: WPS433
                from simulation.env_hotpath_expression_diagnostics import (  # noqa: WPS433
                    reset_current_diagnostics,
                    set_current_diagnostics,
                )

                _cost_charging_t0 = time.perf_counter()
                _budget_scalar_fastpath = bool(
                    getattr(self.sim_cfg, "env_hotpath_budget_ledger_scalar_fastpath_enable", False)
                )
                _diag_token = set_current_diagnostics(_diag) if _diag is not None else None
                _native_budget_enable = bool(getattr(self.sim_cfg, "native_cognitive_budget_cost_aggregation_enable", False))
                _cost_surface_enable = bool(getattr(self.sim_cfg, "env_hotpath_cost_charge_surface_enable", False))
                _cost_surface_guard = bool(getattr(self.sim_cfg, "env_hotpath_cost_charge_surface_guard_enable", False))
                _cost_surface_guard_mode = str(
                    getattr(self.sim_cfg, "env_hotpath_cost_charge_surface_guard_mode", "sample") or "sample"
                ).strip().lower()
                _cost_surface_guard_period = max(
                    1,
                    int(getattr(self.sim_cfg, "env_hotpath_cost_charge_surface_guard_sample_period", 1000) or 1000),
                )
                _cost_surface_soa_diag_enable = bool(
                    getattr(self.sim_cfg, "env_hotpath_costsurface_soa_diagnostics_enable", False)
                )
                _cost_surface_soa_diag = None
                if bool(_cost_surface_enable and _cost_surface_soa_diag_enable):
                    _cost_surface_soa_diag = getattr(self, "_costsurface_soa_diagnostic", None)
                    if _cost_surface_soa_diag is None:
                        _cost_surface_soa_diag = CostSurfaceSoADiagnostic(
                            run_dir=str(getattr(self, "run_dir", "")),
                            enabled=True,
                        )
                        setattr(self, "_costsurface_soa_diagnostic", _cost_surface_soa_diag)
                _cost_surface_seen = 0
                _cost_surface_info = {
                    "hit_count": 0,
                    "fallback_count": 0,
                    "guard_check_count": 0,
                    "guard_mismatch_count": 0,
                    "remaining_full_phenotype_count": 0,
                    "max_abs_diff": 0.0,
                    "surface_s": 0.0,
                    "scalar_s": 0.0,
                    "guard_reference_s": 0.0,
                    "fallback_s": 0.0,
                    "skipped_reason_counts": {},
                }
                _direct_scalar_enable = bool(getattr(self.sim_cfg, "env_hotpath_direct_cost_scalar_enable", False))
                _direct_scalar_guard = bool(getattr(self.sim_cfg, "env_hotpath_direct_cost_scalar_mismatch_guard", False))
                _direct_scalar_guard_mode = str(
                    getattr(self.sim_cfg, "env_hotpath_direct_cost_scalar_guard_mode", "sample") or "sample"
                ).strip().lower()
                _direct_scalar_guard_period = max(
                    1,
                    int(getattr(self.sim_cfg, "env_hotpath_direct_cost_scalar_guard_sample_period", 1000) or 1000),
                )
                _direct_scalar_seen = 0
                _direct_scalar_info = {
                    "hit_count": 0,
                    "ledger_fallback_count": 0,
                    "guard_check_count": 0,
                    "guard_mismatch_count": 0,
                    "max_abs_diff": 0.0,
                    "direct_scalar_s": 0.0,
                    "guard_ledger_s": 0.0,
                    "fallback_ledger_s": 0.0,
                    "skipped_reason_counts": {},
                }
                _native_agents = []
                _native_indices = []
                _native_phenotypes = []
                for i, agent in enumerate(alive):
                    if not agent.alive:
                        continue
                    try:
                        _diag = getattr(self, "env_hotpath_expression_diagnostics", None)
                        if _diag is not None:
                            _st = getattr(agent, "stats", None)
                            _gen = (
                                self._dev_generation_index()
                                if hasattr(self, "_dev_generation_index")
                                else int(getattr(self, "_eiel_fdp_cur_gen", -1))
                            )
                            _tick = (
                                self._dev_tick_index()
                                if hasattr(self, "_dev_tick_index")
                                else int(getattr(self, "_eiel_fdp_cur_tick", -1))
                            )
                            _diag.record_phenotype_call(
                                agent_id=int(getattr(agent, "agent_id", 0) or 0),
                                generation=int(_gen),
                                tick=int(_tick),
                                age_steps=int(getattr(_st, "survival_steps", 0) or 0) if _st is not None else 0,
                                source="env_cost_charge_surface" if _cost_surface_enable else "env_cost_charging",
                            )
                    except Exception:
                        pass
                    if _cost_surface_enable:
                        _guard_index = int(_cost_surface_seen)
                        _cost_surface_seen += 1
                        _surface_t0 = time.perf_counter()
                        try:
                            surface = express_cost_charge_surface(
                                getattr(agent, "genome", None),
                                agent=agent,
                                env_cfg=self.env_cfg,
                                sim_cfg=self.sim_cfg,
                            )
                            _cost_surface_info["hit_count"] = int(_cost_surface_info["hit_count"]) + 1
                        except Exception:
                            _cost_surface_info["skipped_reason_counts"]["exception"] = (
                                int(_cost_surface_info["skipped_reason_counts"].get("exception", 0)) + 1
                            )
                            if _cost_surface_soa_diag is not None:
                                _cost_surface_soa_diag.record_fallback("exception")
                            _cost_surface_info["fallback_count"] = int(_cost_surface_info["fallback_count"]) + 1
                            _cost_surface_info["surface_s"] = float(_cost_surface_info["surface_s"]) + float(
                                time.perf_counter() - _surface_t0
                            )
                            _fallback_t0 = time.perf_counter()
                            _phenotype_t0 = time.perf_counter()
                            phenotype = express_cognitive_phenotype(
                                getattr(agent, "genome", None),
                                agent=agent,
                                env_cfg=self.env_cfg,
                                sim_cfg=self.sim_cfg,
                            )
                            _cost_surface_info["remaining_full_phenotype_count"] = (
                                int(_cost_surface_info["remaining_full_phenotype_count"]) + 1
                            )
                            if _diag is not None:
                                try:
                                    _diag.record_decomposition_bucket(
                                        "env_apply_phenotype_total",
                                        time.perf_counter() - _phenotype_t0,
                                    )
                                except Exception:
                                    pass
                            _ledger_t0 = time.perf_counter()
                            ledger = compute_cognitive_budget_ledger(
                                phenotype,
                                agent=agent,
                                env_cfg=self.env_cfg,
                                sim_cfg=self.sim_cfg,
                            )
                            if _diag is not None:
                                try:
                                    _diag.record_decomposition_bucket(
                                        "env_apply_budget_ledger_total",
                                        time.perf_counter() - _ledger_t0,
                                    )
                                except Exception:
                                    pass
                            cost = compute_env_cost_charge_scalar_from_ledger(ledger)
                            _cost_surface_info["fallback_s"] = float(_cost_surface_info["fallback_s"]) + float(
                                time.perf_counter() - _fallback_t0
                            )
                        else:
                            _cost_surface_info["surface_s"] = float(_cost_surface_info["surface_s"]) + float(
                                time.perf_counter() - _surface_t0
                            )
                            _surface_scalar_t0 = time.perf_counter()
                            cost = compute_env_cost_charge_scalar_from_surface(surface, sim_cfg=self.sim_cfg)
                            _surface_scalar_s = float(time.perf_counter() - _surface_scalar_t0)
                            _cost_surface_info["scalar_s"] = float(_cost_surface_info["scalar_s"]) + float(
                                _surface_scalar_s
                            )
                            if _cost_surface_soa_diag is not None:
                                _cost_surface_soa_diag.record_surface(
                                    surface,
                                    current_scalar=float(cost),
                                    current_scalar_s=float(_surface_scalar_s),
                                    sim_cfg=self.sim_cfg,
                                )
                            _should_guard = bool(
                                _cost_surface_guard
                                and (
                                    _cost_surface_guard_mode == "all"
                                    or (_cost_surface_guard_mode == "sample" and (_guard_index % _cost_surface_guard_period) == 0)
                                )
                            )
                            if _should_guard:
                                _guard_t0 = time.perf_counter()
                                _phenotype_t0 = time.perf_counter()
                                phenotype = express_cognitive_phenotype(
                                    getattr(agent, "genome", None),
                                    agent=agent,
                                    env_cfg=self.env_cfg,
                                    sim_cfg=self.sim_cfg,
                                )
                                _cost_surface_info["remaining_full_phenotype_count"] = (
                                    int(_cost_surface_info["remaining_full_phenotype_count"]) + 1
                                )
                                if _diag is not None:
                                    try:
                                        _diag.record_decomposition_bucket(
                                            "env_apply_phenotype_total",
                                            time.perf_counter() - _phenotype_t0,
                                        )
                                    except Exception:
                                        pass
                                _ledger_t0 = time.perf_counter()
                                ledger = compute_cognitive_budget_ledger(
                                    phenotype,
                                    agent=agent,
                                    env_cfg=self.env_cfg,
                                    sim_cfg=self.sim_cfg,
                                )
                                if _diag is not None:
                                    try:
                                        _diag.record_decomposition_bucket(
                                            "env_apply_budget_ledger_total",
                                            time.perf_counter() - _ledger_t0,
                                        )
                                    except Exception:
                                        pass
                                expected_cost = compute_env_cost_charge_scalar_from_ledger(ledger)
                                _cost_surface_info["guard_reference_s"] = float(
                                    _cost_surface_info["guard_reference_s"]
                                ) + float(time.perf_counter() - _guard_t0)
                                diff = abs(float(expected_cost) - float(cost))
                                _cost_surface_info["guard_check_count"] = int(_cost_surface_info["guard_check_count"]) + 1
                                _cost_surface_info["max_abs_diff"] = max(float(_cost_surface_info["max_abs_diff"]), float(diff))
                                if diff > 1.0e-12:
                                    _cost_surface_info["guard_mismatch_count"] = (
                                        int(_cost_surface_info["guard_mismatch_count"]) + 1
                                    )
                                    if _diag is not None:
                                        try:
                                            _diag.record_cost_charge_surface(_cost_surface_info)
                                        except Exception:
                                            pass
                                    raise RuntimeError(
                                        "cost charge surface mismatch: "
                                        f"expected={float(expected_cost)!r} actual={float(cost)!r} diff={float(diff)!r}"
                                    )
                        if cost > 0.0:
                            new_energy[i] = max(0.0, float(new_energy[i]) - cost)
                        continue
                    _phenotype_t0 = time.perf_counter()
                    phenotype = express_cognitive_phenotype(
                        getattr(agent, "genome", None),
                        agent=agent,
                        env_cfg=self.env_cfg,
                        sim_cfg=self.sim_cfg,
                    )
                    if _diag is not None:
                        try:
                            _diag.record_decomposition_bucket(
                                "env_apply_phenotype_total",
                                time.perf_counter() - _phenotype_t0,
                            )
                        except Exception:
                            pass
                    if _direct_scalar_enable:
                        _guard_index = int(_direct_scalar_seen)
                        _direct_scalar_seen += 1
                        _direct_t0 = time.perf_counter()
                        try:
                            cost = compute_env_cost_charge_scalar(phenotype, sim_cfg=self.sim_cfg)
                            _direct_scalar_info["hit_count"] = int(_direct_scalar_info["hit_count"]) + 1
                        except Exception:
                            _direct_scalar_info["skipped_reason_counts"]["exception"] = (
                                int(_direct_scalar_info["skipped_reason_counts"].get("exception", 0)) + 1
                            )
                            _direct_scalar_info["ledger_fallback_count"] = int(_direct_scalar_info["ledger_fallback_count"]) + 1
                            _direct_scalar_info["direct_scalar_s"] = float(_direct_scalar_info["direct_scalar_s"]) + float(
                                time.perf_counter() - _direct_t0
                            )
                            _fallback_t0 = time.perf_counter()
                            ledger = compute_cognitive_budget_ledger(
                                phenotype,
                                agent=agent,
                                env_cfg=self.env_cfg,
                                sim_cfg=self.sim_cfg,
                            )
                            _direct_scalar_info["fallback_ledger_s"] = float(_direct_scalar_info["fallback_ledger_s"]) + float(
                                time.perf_counter() - _fallback_t0
                            )
                            cost = compute_env_cost_charge_scalar_from_ledger(ledger)
                        else:
                            _direct_scalar_info["direct_scalar_s"] = float(_direct_scalar_info["direct_scalar_s"]) + float(
                                time.perf_counter() - _direct_t0
                            )
                            _should_guard = bool(
                                _direct_scalar_guard
                                and (
                                    _direct_scalar_guard_mode == "all"
                                    or (_direct_scalar_guard_mode == "sample" and (_guard_index % _direct_scalar_guard_period) == 0)
                                )
                            )
                            if _should_guard:
                                _guard_t0 = time.perf_counter()
                                ledger = compute_cognitive_budget_ledger(
                                    phenotype,
                                    agent=agent,
                                    env_cfg=self.env_cfg,
                                    sim_cfg=self.sim_cfg,
                                )
                                _direct_scalar_info["guard_ledger_s"] = float(_direct_scalar_info["guard_ledger_s"]) + float(
                                    time.perf_counter() - _guard_t0
                                )
                                expected_cost = compute_env_cost_charge_scalar_from_ledger(ledger)
                                diff = abs(float(expected_cost) - float(cost))
                                _direct_scalar_info["guard_check_count"] = int(_direct_scalar_info["guard_check_count"]) + 1
                                _direct_scalar_info["max_abs_diff"] = max(float(_direct_scalar_info["max_abs_diff"]), float(diff))
                                if diff > 1.0e-12:
                                    _direct_scalar_info["guard_mismatch_count"] = (
                                        int(_direct_scalar_info["guard_mismatch_count"]) + 1
                                    )
                                    if _diag is not None:
                                        try:
                                            _diag.record_direct_cost_scalar(_direct_scalar_info)
                                        except Exception:
                                            pass
                                    raise RuntimeError(
                                        "direct cost scalar mismatch: "
                                        f"expected={float(expected_cost)!r} actual={float(cost)!r} diff={float(diff)!r}"
                                    )
                        if cost > 0.0:
                            new_energy[i] = max(0.0, float(new_energy[i]) - cost)
                        continue
                    if _native_budget_enable:
                        _native_agents.append(agent)
                        _native_indices.append(i)
                        _native_phenotypes.append(phenotype)
                        continue
                    _ledger_t0 = time.perf_counter()
                    if _budget_scalar_fastpath:
                        cost = compute_env_apply_cognitive_cost_scalar(
                            phenotype,
                            agent=agent,
                            env_cfg=self.env_cfg,
                            sim_cfg=self.sim_cfg,
                        )
                        if _diag is not None:
                            try:
                                _diag.record_decomposition_bucket(
                                    "env_apply_budget_ledger_scalar_fastpath_total",
                                    time.perf_counter() - _ledger_t0,
                                )
                            except Exception:
                                pass
                    else:
                        ledger = compute_cognitive_budget_ledger(
                            phenotype,
                            agent=agent,
                            env_cfg=self.env_cfg,
                            sim_cfg=self.sim_cfg,
                        )
                        if _diag is not None:
                            try:
                                _diag.record_decomposition_bucket(
                                    "env_apply_budget_ledger_total",
                                    time.perf_counter() - _ledger_t0,
                                )
                            except Exception:
                                pass
                        _cost_sum_t0 = time.perf_counter()
                        cost = compute_env_cost_charge_scalar_from_ledger(ledger)
                        if _diag is not None:
                            try:
                                _diag.record_decomposition_bucket(
                                    "env_apply_cost_sum_total",
                                    time.perf_counter() - _cost_sum_t0,
                                )
                            except Exception:
                                pass
                    if cost > 0.0:
                        new_energy[i] = max(0.0, float(new_energy[i]) - cost)
                if _native_budget_enable and _native_phenotypes:
                    _ledger_t0 = time.perf_counter()
                    try:
                        _native_ledgers, _native_info = compute_cognitive_budget_ledgers_batch(
                            _native_phenotypes,
                            agents=_native_agents,
                            env_cfg=self.env_cfg,
                            sim_cfg=self.sim_cfg,
                        )
                    except Exception:
                        _native_ledgers = [
                            compute_cognitive_budget_ledger(
                                phenotype,
                                agent=agent,
                                env_cfg=self.env_cfg,
                                sim_cfg=self.sim_cfg,
                            )
                            for phenotype, agent in zip(_native_phenotypes, _native_agents)
                        ]
                        _native_info = {
                            "schema_version": NATIVE_BUDGET_AGGREGATION_SCHEMA_VERSION,
                            "backend": "python-exception-fallback",
                            "hit_count": 0,
                            "fallback_count": len(_native_ledgers),
                            "mismatch_count": 0,
                            "pack_rows": len(_native_ledgers),
                            "native_kernel_s": 0.0,
                            "input_packing_s": 0.0,
                        }
                    if _diag is not None:
                        try:
                            _diag.record_decomposition_bucket(
                                "env_apply_budget_ledger_total",
                                time.perf_counter() - _ledger_t0,
                                count=len(_native_ledgers),
                            )
                            if bool(getattr(self.sim_cfg, "native_cognitive_budget_cost_aggregation_diagnostics_enable", True)):
                                _diag.record_native_budget_cost_aggregation(_native_info)
                        except Exception:
                            pass
                    for i, ledger in zip(_native_indices, _native_ledgers):
                        _cost_sum_t0 = time.perf_counter()
                        cost = compute_env_cost_charge_scalar_from_ledger(ledger)
                        if _diag is not None:
                            try:
                                _diag.record_decomposition_bucket(
                                    "env_apply_cost_sum_total",
                                    time.perf_counter() - _cost_sum_t0,
                                )
                            except Exception:
                                pass
                        if cost > 0.0:
                            new_energy[i] = max(0.0, float(new_energy[i]) - cost)
                if _diag is not None and _direct_scalar_enable:
                    try:
                        _diag.record_direct_cost_scalar(_direct_scalar_info)
                    except Exception:
                        pass
                if _diag is not None and _cost_surface_enable:
                    try:
                        _diag.record_cost_charge_surface(_cost_surface_info)
                    except Exception:
                        pass
                if _cost_surface_soa_diag is not None:
                    try:
                        _cost_surface_soa_diag.write_summary()
                    except Exception:
                        pass
                if _diag_token is not None:
                    reset_current_diagnostics(_diag_token)
                if _diag is not None:
                    try:
                        _diag.record_decomposition_bucket(
                            "env_apply_cost_charging_total",
                            time.perf_counter() - _cost_charging_t0,
                        )
                    except Exception:
                        pass
            except Exception as exc:
                try:
                    if "_diag_token" in locals() and _diag_token is not None:
                        reset_current_diagnostics(_diag_token)
                except Exception:
                    pass
                if isinstance(exc, RuntimeError) and (
                    "cost charge surface mismatch" in str(exc) or "direct cost scalar mismatch" in str(exc)
                ):
                    raise
                pass

        if _cxxcore_shadow_record is not None:
            try:
                if _cxxcore_shadow_snapshot is not None:
                    _shadow_compute_t0 = time.perf_counter()
                    _shadow_actual = compute_env_apply_shadow_python(
                        config=_cxxcore_shadow_snapshot["config"],
                        agent_ids=_cxxcore_shadow_snapshot["agent_ids"],
                        actions=_cxxcore_shadow_snapshot["actions"],
                        energies=_cxxcore_shadow_snapshot["energies"],
                        healths=_cxxcore_shadow_snapshot["healths"],
                        injuries=_cxxcore_shadow_snapshot["injuries"],
                        scars=_cxxcore_shadow_snapshot["scars"],
                        pos_y_by_id=_cxxcore_shadow_snapshot["pos_y_by_id"],
                        pos_x_by_id=_cxxcore_shadow_snapshot["pos_x_by_id"],
                        agent_grid=_cxxcore_shadow_snapshot["agent_grid"],
                        grid=_cxxcore_shadow_snapshot["grid"],
                        move_cost_mults=_cxxcore_shadow_snapshot["move_cost_mults"],
                        juvenile_harvest_mults=_cxxcore_shadow_snapshot["juvenile_harvest_mults"],
                    )
                    _cxxcore_shadow_record["timing_s"]["shadow_compute"] = float(time.perf_counter() - _shadow_compute_t0)

                    _shadow_mat_t0 = time.perf_counter()
                    _shadow_expected = observed_delta_from_python(
                        agent_ids=_cxxcore_shadow_snapshot["agent_ids"],
                        pre_energy=_cxxcore_shadow_snapshot["energies"],
                        pre_health=_cxxcore_shadow_snapshot["healths"],
                        pre_injury=_cxxcore_shadow_snapshot["injuries"],
                        pre_scar=_cxxcore_shadow_snapshot["scars"],
                        post_energy=new_energy,
                        post_health=new_health,
                        post_injury=new_injury,
                        post_scar=new_scar,
                        food_gained=food_gained,
                        died_mask=died_mask,
                        pre_pos_y=_cxxcore_shadow_snapshot["pre_pos_y"],
                        pre_pos_x=_cxxcore_shadow_snapshot["pre_pos_x"],
                        post_pos_y_by_id=np.asarray(getattr(self.env, "agent_pos_y"), dtype=np.int32),
                        post_pos_x_by_id=np.asarray(getattr(self.env, "agent_pos_x"), dtype=np.int32),
                    )
                    _cxxcore_shadow_record["timing_s"]["materialize"] = float(time.perf_counter() - _shadow_mat_t0)

                    if bool(getattr(self.sim_cfg, "cxxcore_env_apply_shadow_compare_enable", True)):
                        _shadow_compare_t0 = time.perf_counter()
                        _shadow_parity = compare_shadow_deltas(_shadow_expected, _shadow_actual, atol=0.0)
                        _cxxcore_shadow_record["timing_s"]["compare"] = float(time.perf_counter() - _shadow_compare_t0)
                        _cxxcore_shadow_record["parity"] = _shadow_parity
                    else:
                        _cxxcore_shadow_record["parity"] = "not_checked"
                _cxxcore_shadow_record["timing_s"]["wall"] = float(
                    _cxxcore_shadow_record["timing_s"].get("snapshot_copy", 0.0)
                    + _cxxcore_shadow_record["timing_s"].get("shadow_compute", 0.0)
                    + _cxxcore_shadow_record["timing_s"].get("materialize", 0.0)
                    + _cxxcore_shadow_record["timing_s"].get("compare", 0.0)
                )
                self._cxxcore_env_apply_shadow_last = _cxxcore_shadow_record
                _shadow_diag_t0 = time.perf_counter()
                # Diagnostics overhead is summary bookkeeping and optional compact JSON write.
                _cxxcore_shadow_record["timing_s"]["diagnostics"] = float(time.perf_counter() - _shadow_diag_t0)
                _cxxcore_shadow_record["timing_s"]["wall"] = float(
                    _cxxcore_shadow_record["timing_s"].get("wall", 0.0)
                    + _cxxcore_shadow_record["timing_s"].get("diagnostics", 0.0)
                )
                self._cxxcore_env_apply_shadow_summary = update_shadow_summary(
                    getattr(self, "_cxxcore_env_apply_shadow_summary", None),
                    _cxxcore_shadow_record,
                )
                if bool(getattr(self.sim_cfg, "cxxcore_env_apply_shadow_write_summary", True)):
                    try:
                        write_shadow_summary(
                            getattr(self, "run_dir", None),
                            self._cxxcore_env_apply_shadow_summary,
                            filename=str(
                                getattr(
                                    self.sim_cfg,
                                    "cxxcore_env_apply_shadow_summary_filename",
                                    "cxxcore_env_apply_shadow_summary.json",
                                )
                            ),
                        )
                    except Exception:
                        pass
                _parity = _cxxcore_shadow_record.get("parity")
                if (
                    bool(getattr(self.sim_cfg, "cxxcore_env_apply_shadow_fail_fast", False))
                    and isinstance(_parity, dict)
                    and not bool(_parity.get("ok", False))
                ):
                    raise RuntimeError(f"CXXCORE env-apply shadow divergence: {_parity.get('first_divergence')!r}")
            except Exception as exc:
                if bool(getattr(self.sim_cfg, "cxxcore_env_apply_shadow_fail_fast", False)):
                    raise
                _cxxcore_shadow_record["skipped"] = True
                _cxxcore_shadow_record["skip_reasons"] = {"shadow_exception": True, "detail": str(exc)}
                self._cxxcore_env_apply_shadow_last = _cxxcore_shadow_record
                self._cxxcore_env_apply_shadow_summary = update_shadow_summary(
                    getattr(self, "_cxxcore_env_apply_shadow_summary", None),
                    _cxxcore_shadow_record,
                )

        if _cxxrewrite2_native_record is not None:
            try:
                _cxxrewrite2_native_record = cxxrewrite2_run_runner_shadow_compare(
                    self,
                    snapshot=_cxxrewrite2_native_snapshot,
                    record=_cxxrewrite2_native_record,
                    new_energy=new_energy,
                    new_health=new_health,
                    new_injury=new_injury,
                    new_scar=new_scar,
                    food_gained=food_gained,
                    died_mask=died_mask,
                )
                self._cxxrewrite2_native_tick_last = _cxxrewrite2_native_record
                self._cxxrewrite2_native_tick_summary = cxxrewrite2_update_native_tick_summary(
                    getattr(self, "_cxxrewrite2_native_tick_summary", None),
                    _cxxrewrite2_native_record,
                )
                if bool(getattr(self.sim_cfg, "cxxrewrite_native_tick_write_summary", True)):
                    try:
                        cxxrewrite2_write_native_tick_summary(
                            getattr(self, "run_dir", None),
                            self._cxxrewrite2_native_tick_summary,
                            filename=str(
                                getattr(
                                    self.sim_cfg,
                                    "cxxrewrite_native_tick_summary_filename",
                                    "cxxrewrite2_native_tick_summary.json",
                                )
                            ),
                        )
                    except Exception:
                        pass
            except Exception:
                if bool(getattr(self.sim_cfg, "cxxrewrite_native_tick_fail_closed", False)) or bool(
                    getattr(self.sim_cfg, "cxxrewrite_native_tick_authoritative_enable", False)
                ):
                    raise
                _cxxrewrite2_native_record["skipped"] = True
                _cxxrewrite2_native_record["skip_reasons"] = {"shadow_exception": True}
                self._cxxrewrite2_native_tick_last = _cxxrewrite2_native_record
                self._cxxrewrite2_native_tick_summary = cxxrewrite2_update_native_tick_summary(
                    getattr(self, "_cxxrewrite2_native_tick_summary", None),
                    _cxxrewrite2_native_record,
                )

        if _cxxrewrite3_resident_enabled:
            try:
                cxxrewrite3_commit_post_env_arrays(
                    self,
                    agent_ids_np=agent_ids_np,
                    new_energy=new_energy,
                    new_health=new_health,
                    new_injury=new_injury,
                    new_scar=new_scar,
                    died_mask=died_mask,
                )
            except Exception:
                if bool(getattr(self.sim_cfg, "cxxrewrite_resident_frame_fail_closed", False)) or bool(
                    getattr(self.sim_cfg, "cxxrewrite_resident_frame_authoritative_enable", False)
                ) or bool(getattr(self.sim_cfg, "cxxrewrite_resident_frame_native_packing_authoritative_enable", False)):
                    raise

        try:
            cxxrewrite4_record_post_env_scalar_arrays(
                self,
                alive=alive,
                agent_ids_np=agent_ids_np,
                new_energy=new_energy,
                new_health=new_health,
                new_injury=new_injury,
                new_scar=new_scar,
                died_mask=died_mask,
                stage="env_apply_arrays",
            )
        except Exception:
            if cxxrewrite4_resident_scalar_sync_fail_closed(self.sim_cfg):
                raise

        try:
            bo = getattr(self, "branch_observability", None)
            if bo is not None:
                bo.record_niche_post_env(
                    agent_ids_np=agent_ids_np,
                    food_gained=food_gained,
                    pre_health=pre_health,
                    new_health=new_health,
                    pre_injury=pre_injury,
                    new_injury=new_injury,
                )
        except Exception:
            pass

        # Persist env outputs back to agents + cache deltas
        for i, agent in enumerate(alive):
            if not agent.alive:
                continue

            e0 = float(pre_energy[i])
            h0 = float(pre_health[i])
            inj0 = float(pre_injury[i])

            e1 = float(new_energy[i])
            h1 = float(new_health[i])
            inj1 = float(new_injury[i])

            agent.energy = e1
            agent.health = h1
            agent.injury = inj1
            agent.scar = float(new_scar[i])

            st = getattr(agent, "stats", None)
            if st is not None:
                st.last_energy_delta = float(e1 - e0)
                st.last_health_delta = float(h1 - h0)
                st.last_injury_delta = float(inj1 - inj0)

            # Biome id (best-effort) — updated for array-based env state
            biome_id = 0
            try:
                aid = int(agent.agent_id)
                if bool(getattr(self.env, "agent_alive")[aid]):
                    by = int(getattr(self.env, "agent_pos_y")[aid])
                    bx = int(getattr(self.env, "agent_pos_x")[aid])
                    biome_grid = getattr(self.env, "biome_grid", None)
                    if biome_grid is not None:
                        biome_id = int(biome_grid[by, bx])
            except Exception:
                biome_id = 0

            try:
                agent.stats.biome_id = int(biome_id)
            except Exception:
                pass

            if agent.injury > gen_peak_injury:
                gen_peak_injury = float(agent.injury)

            # Early-life weighting marker (best-effort)
            try:
                agent.stats.early_life_weight = float(getattr(self.sim_cfg, "early_life_weight", 1.0))
            except Exception:
                pass

            st_age = int(getattr(agent.stats, "survival_steps", 0))

            k = int(getattr(self.sim_cfg, "early_life_steps", 0))
            early_window = (k > 0) and (st_age < k)


            # Tier 4: juvenile/adult death accounting uses the PRE-increment age.
            if tier4_enable and (t4_steps > 0) and bool(died_mask[i]):
                if st_age < t4_steps:
                    self._tier4_juvenile_deaths_gen += 1
                else:
                    self._tier4_adult_deaths_gen += 1

                # Coarse death-by-age bins (JSON in metrics). We lazily allocate a fixed-length counter
                # matching the configured edges.
                try:
                    edges = list(getattr(self.env_cfg, "tier4_survival_bins_steps", [])) or []
                    if len(edges) >= 2:
                        if self._tier4_deaths_age_bins_gen is None or len(self._tier4_deaths_age_bins_gen) != (len(edges) - 1):
                            self._tier4_deaths_age_bins_gen = [0 for _ in range(len(edges) - 1)]
                        # Find bin index (edges[k] <= age < edges[k+1])
                        b = 0
                        for k in range(len(edges) - 1):
                            if st_age >= int(edges[k]) and st_age < int(edges[k + 1]):
                                b = k
                                break
                            b = k
                        self._tier4_deaths_age_bins_gen[b] += 1
                except Exception:
                    pass

            agent.stats.survival_steps = st_age + 1

            fg = float(food_gained[i])
            if fg > 0.0:
                agent.stats.energy_collected = float(getattr(agent.stats, "energy_collected", 0.0)) + fg

            if early_window:
                agent.stats.early_survival_steps = int(getattr(agent.stats, "early_survival_steps", 0)) + 1
                if fg > 0.0:
                    agent.stats.early_energy_collected = float(getattr(agent.stats, "early_energy_collected", 0.0)) + fg

                dinj_pos = max(0.0, float(inj1 - inj0))
                if dinj_pos > 0.0:
                    agent.stats.early_injury_gained = float(getattr(agent.stats, "early_injury_gained", 0.0)) + dinj_pos

        dt_apply = (time.perf_counter() - t_apply0)
        buckets["env_apply"] = buckets.get("env_apply", 0.0) + dt_apply
        # Phase 0 perf: alias for the A/B report naming (values are seconds).
        buckets["action_apply_ms"] = buckets.get("action_apply_ms", 0.0) + dt_apply
        if _diag is not None:
            try:
                _diag.record_decomposition_bucket("env_apply_total", dt_apply)
            except Exception:
                pass

        _cxxrewrite4_dead_agents = []
        if bool(np.any(died_mask)):
            # ------------------------------------------------------------
            # Phase 3 determinism forensics: env delta dump at selected ticks
            # ------------------------------------------------------------
            # Purpose:
            #   When first divergence is "post_env", record exactly which agents
            #   died (or would die) on this tick and their pre/post env scalars.
            #
            # Controls:
            #   EIEL_DEBUG_ENV_DELTA=1
            #   EIEL_DEBUG_ENV_DELTA_TICKS="61" or "60-70" or "0-10,61,90-95"
            #
            # Output:
            #   state_dumps/state_XXXXXX__tick_env_delta_post_env_t061.json
            try:
                import os as _os  # noqa: WPS433

                def _eiel_tick_selected(ti: int) -> bool:
                    spec = (_os.getenv("EIEL_DEBUG_ENV_DELTA_TICKS", "") or "").strip()
                    if not spec:
                        return False
                    allowed = set()
                    for part in str(spec).split(","):
                        p = str(part).strip()
                        if not p:
                            continue
                        if "-" in p:
                            a_s, b_s = p.split("-", 1)
                            a = int(a_s.strip())
                            b = int(b_s.strip())
                            lo, hi = (a, b) if a <= b else (b, a)
                            if hi - lo > 1000000:
                                continue
                            allowed.update(range(lo, hi + 1))
                        else:
                            allowed.add(int(p))
                    return ti in allowed

                if (_os.getenv("EIEL_DEBUG_ENV_DELTA", "0") or "0").strip() in ("1", "true", "TRUE", "yes", "YES"):
                    _tick = int(getattr(self, "tick", 0) or 0)
                    if _eiel_tick_selected(_tick):
                        from simulation.state_dump import maybe_dump_state  # noqa: WPS433
                        _gen = int(getattr(self, "_current_gen", 0) or 0)

                        died_idx = np.nonzero(died_mask)[0].astype(np.int32)
                        died_ids = [int(agent_ids_np[int(i)]) for i in died_idx.tolist()]
                        died_ids.sort()

                        # Build per-death record deterministically.
                        per = []
                        for i in died_idx.tolist():
                            ii = int(i)
                            aid = int(agent_ids_np[ii])
                            rec = {
                                "agent_id": aid,
                                "pre_energy": float(pre_energy[ii]),
                                "pre_health": float(pre_health[ii]),
                                "pre_injury": float(pre_injury[ii]),
                                "post_energy": float(new_energy[ii]),
                                "post_health": float(new_health[ii]),
                                "post_injury": float(new_injury[ii]),
                            }
                            # Best-effort positions (if env arrays exist)
                            try:
                                rec["x"] = int(getattr(self.env, "agent_pos_x")[aid])
                                rec["y"] = int(getattr(self.env, "agent_pos_y")[aid])
                            except Exception:
                                pass
                            per.append(rec)

                        extra = {
                            "gen": _gen,
                            "tick": _tick,
                            "died_n": int(len(died_ids)),
                            "died_ids": died_ids,
                            "died": per,
                        }
                        maybe_dump_state(
                            self,
                            generation_index=int(_gen),
                            tag=f"tick_env_delta_post_env_t{_tick:03d}",
                            extra=extra,
                        )
            except Exception:
                pass
            for i in np.nonzero(died_mask)[0].tolist():
                agent = alive[i]
                if agent.alive:
                    if bool(getattr(self.env, "patch_world_enable", False)):
                        try:
                            from env.gridworld_pkg.patch_world import sample_patch_context  # noqa: WPS433

                            aid = int(getattr(agent, "agent_id"))
                            y = int(getattr(self.env, "agent_pos_y")[aid])
                            x = int(getattr(self.env, "agent_pos_x")[aid])
                            if not hasattr(self, "_patch_death_contexts_gen"):
                                self._patch_death_contexts_gen = []
                            self._patch_death_contexts_gen.append(
                                sample_patch_context(self.env, y, x, agent=agent)
                            )
                        except Exception:
                            pass
                    agent.alive = False
                    deaths_this_gen += 1
                    self.env.remove_agent(int(agent.agent_id))
                    _cxxrewrite4_dead_agents.append(agent)
            try:
                cxxrewrite4_sync_agents_scalar_state(self, _cxxrewrite4_dead_agents, stage="env_apply_deaths")
            except Exception:
                if cxxrewrite4_resident_scalar_sync_fail_closed(self.sim_cfg):
                    raise


        # ------------------------------------------------------------
        # Phase 3 determinism invariant: env tick must NOT create agents
        # ------------------------------------------------------------
        # Empirically, a resume run can sometimes show +1 alive immediately after post_env,
        # with an "uninitialized" agent slot (energy/injury/pos ~ 0) appearing.
        # That is a hard invariant violation: env tick may kill/mutate agents, but may not spawn.
        #
        # Enforce: alive_after_env ⊆ input agent_ids_np
        # If violated, forcibly kill/remove the unexpected agents to prevent ghost slots
        # from poisoning downstream stages and breaking determinism.
        try:
            import os as _os  # noqa: WPS433
            if (_os.getenv("EIEL_ALLOW_ENV_SPAWN", "0") or "0").strip() not in ("1", "true", "TRUE", "yes", "YES"):
                _in_ids = set(int(x) for x in np.asarray(agent_ids_np, dtype=np.int64).tolist())
                _unexpected = []
                for _a in alive:
                    try:
                        _aid = int(getattr(_a, "agent_id"))
                    except Exception:
                        continue
                    if _aid not in _in_ids:
                        _unexpected.append(_a)

                if _unexpected:
                    # Membership trace (optional)
                    try:
                        from simulation.membership_trace import note_write  # noqa: WPS433
                    except Exception:
                        note_write = None  # type: ignore[assignment]

                    for _a in _unexpected:
                        try:
                            _aid = int(getattr(_a, "agent_id"))
                        except Exception:
                            continue
                        try:
                            if note_write is not None:
                                note_write(kind="env_tick.unexpected_alive_kill", agent_id=int(_aid), detail="alive after env tick but not in input batch")
                        except Exception:
                            pass
                        try:
                            _a.alive = False
                        except Exception:
                            pass
                        try:
                            cxxrewrite4_sync_agents_scalar_state(self, [_a], stage="env_apply_unexpected_alive_kill")
                        except Exception:
                            if cxxrewrite4_resident_scalar_sync_fail_closed(self.sim_cfg):
                                raise
                        try:
                            self.env.remove_agent(int(_aid))
                        except Exception:
                            pass
                        try:
                            self.pop.agents.pop(int(_aid), None)
                        except Exception:
                            pass

                    # Filter alive list in-place (keep ordering stable)
                    alive[:] = [a for a in alive if int(getattr(a, "agent_id")) in _in_ids]
        except Exception:
            pass

        return deaths_this_gen, gen_peak_injury
