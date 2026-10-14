from __future__ import annotations

"""simulation/runner_pkg/runner_selection.py

Selection + behavior outcomes wiring.

This mixin ranks eligible agents by an *effective* fitness used only for selection.
behaviors.csv continues to record *base* fitness for comparability.

Episode-0 selection pressure supports two modes:
- Legacy combined: episode0_fitness_weight * episode0_term
- Split knobs (preferred): separate weights per episode-0 component

Constraints respected:
- No world dynamics changes.
- No new metrics.
- Backward compatible and deterministic.
- Localized to selection/fitness computation.
"""

from typing import Dict, List, Set, Tuple
import time

from evolution.selection import fitness as base_fitness


def _clamp(x: float, lo: float, hi: float) -> float:
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


class SelectionMixin:
    def _assert_no_direct_chemistry_selection_scalar(self, st) -> None:
        if not bool(getattr(self.sim_cfg, "selection_forbid_chemistry_scalar", True)):
            return
        if st is None:
            return
        forbidden_exact = (
            "chemistry_score",
            "chemistry_fitness",
            "chemistry_scalar_reward",
            "chemistry_selection_bonus",
            "locality_score",
            "locality_fitness",
            "locality_scalar_reward",
            "locality_selection_bonus",
        )
        for k in forbidden_exact:
            if hasattr(st, k):
                raise RuntimeError(
                    "[DETERMINISM_VIOLATION]\n"
                    "code=DET_FORBIDDEN_CHEMISTRY_SELECTION_SCALAR\n"
                    f"context=selection_stats_field={k}\n"
                )
        # Harden against renamed aliases (e.g. chem_bonus_score) while allowing
        # chemistry telemetry fields that are not selection inputs.
        fields = tuple(getattr(st, "__dict__", {}).keys())
        allow_chem_telemetry = {
            "chemistry_action_cost_mult",
            "chemistry_observation_fidelity",
            "chemistry_observation_bias",
            "chemistry_repro_viability",
            "structural_locality_cost_total",
            "structural_locality_mean_distance",
            "structural_locality_max_distance",
        }
        score_like_tokens = ("score", "fitness", "reward", "bonus", "selection", "scalar", "rank", "utility")
        for k in fields:
            lk = str(k).lower()
            if lk in allow_chem_telemetry:
                continue
            chem_like = ("chem" in lk) or ("damage" in lk) or ("repair_reserve" in lk) or ("regulator" in lk)
            if chem_like and any(tok in lk for tok in score_like_tokens):
                raise RuntimeError(
                    "[DETERMINISM_VIOLATION]\n"
                    "code=DET_FORBIDDEN_CHEMISTRY_SELECTION_SCALAR\n"
                    f"context=selection_stats_field={k}\n"
                )

    # -----------------------------
    # Episode-0 terms (existing metrics only)
    # -----------------------------
    def _episode0_components(self, st) -> Dict[str, float]:
        """
        Compute separate episode-0 component terms using ONLY existing episode-0 metrics.

        Returns
        -------
        components : dict[str, float]
            Each component term is NEGATIVE (penalty-like).
            More negative = worse early instability.
            Keys:
              - "pred": prediction error component term
              - "jitter": drive jitter component term
              - "thrash": switches+oscillations thrash component term
              - "selfharm": self-harm component term
              - "combined": legacy combined term (weighted sum of normalized components)
        """
        if st is None:
            return {"pred": 0.0, "jitter": 0.0, "thrash": 0.0, "selfharm": 0.0, "combined": 0.0}

        k0 = int(getattr(self.sim_cfg, "episode0_k", 20))
        k0 = max(1, k0)

        age = int(getattr(st, "survival_steps", 0))
        ep0_steps = max(1, min(k0, age))

        # 1) pred error mean in ep0
        ep0_sum = float(getattr(st, "episode0_pred_err_sum", 0.0))
        ep0_n = int(getattr(st, "episode0_pred_err_count", 0))
        ep0_pred_mean = ep0_sum / max(1.0, float(ep0_n))

        # 2) drive jitter mean in ep0
        ep0_drive_jitter_sum = float(getattr(st, "episode0_drive_jitter_sum", 0.0))
        ep0_drive_jitter_mean = ep0_drive_jitter_sum / float(ep0_steps)

        # 3) thrashing rate in ep0
        ep0_switches = float(getattr(st, "episode0_switches", 0.0))
        ep0_osc = float(getattr(st, "episode0_oscillations", 0.0))
        ep0_thrash_rate = (ep0_switches + ep0_osc) / float(ep0_steps)

        # 4) self-harm rate in ep0
        ep0_selfharm = float(getattr(st, "episode0_selfharm_events", 0.0))
        ep0_selfharm_rate = ep0_selfharm / float(ep0_steps)

        # Caps for normalization
        pred_cap = float(getattr(self.sim_cfg, "episode0_pred_err_cap", 5.0))
        jitter_cap = float(getattr(self.sim_cfg, "episode0_drive_jitter_cap", 5.0))
        thrash_cap = float(getattr(self.sim_cfg, "episode0_thrashing_rate_cap", 1.0))
        selfharm_cap = float(getattr(self.sim_cfg, "episode0_selfharm_rate_cap", 0.25))

        pred_n = _clamp(ep0_pred_mean / max(1e-9, pred_cap), 0.0, 1.0)
        if bool(getattr(self.sim_cfg, "disable_prediction_error_selection_pressure", False)):
            pred_n = 0.0
        jitter_n = _clamp(ep0_drive_jitter_mean / max(1e-9, jitter_cap), 0.0, 1.0)
        thrash_n = _clamp(ep0_thrash_rate / max(1e-9, thrash_cap), 0.0, 1.0)
        selfharm_n = _clamp(ep0_selfharm_rate / max(1e-9, selfharm_cap), 0.0, 1.0)

        # Component terms: negative penalties
        pred_term = -float(pred_n)
        jitter_term = -float(jitter_n)
        thrash_term = -float(thrash_n)
        selfharm_term = -float(selfharm_n)

        # Legacy combined term: weighted sum of normalized components (still negative)
        w_pred = float(getattr(self.sim_cfg, "episode0_pred_err_weight", 1.0))
        w_jit = float(getattr(self.sim_cfg, "episode0_drive_jitter_weight", 1.0))
        w_thr = float(getattr(self.sim_cfg, "episode0_thrashing_weight", 1.0))
        w_sh = float(getattr(self.sim_cfg, "episode0_selfharm_weight", 3.0))
        combined = -float((w_pred * pred_n) + (w_jit * jitter_n) + (w_thr * thrash_n) + (w_sh * selfharm_n))

        # Store breakdown on stats (no new logging required, but useful if you inspect objects)
        try:
            st.ep0_steps = int(ep0_steps)
            st.ep0_pred_mean = float(ep0_pred_mean)
            st.ep0_drive_jitter_mean = float(ep0_drive_jitter_mean)
            st.ep0_thrash_rate = float(ep0_thrash_rate)
            st.ep0_selfharm_rate = float(ep0_selfharm_rate)

            st.ep0_pred_term = float(pred_term)
            st.ep0_jitter_term = float(jitter_term)
            st.ep0_thrash_term = float(thrash_term)
            st.ep0_selfharm_term = float(selfharm_term)
            st.ep0_term = float(combined)
        except Exception:
            pass

        return {
            "pred": float(pred_term),
            "jitter": float(jitter_term),
            "thrash": float(thrash_term),
            "selfharm": float(selfharm_term),
            "combined": float(combined),
        }

    def _use_split_episode0(self) -> bool:
        """Split mode is active if any split knob is non-zero."""
        w_pred = float(getattr(self.sim_cfg, "episode0_pred_err_fitness_weight", 0.0))
        w_thr = float(getattr(self.sim_cfg, "episode0_thrashing_fitness_weight", 0.0))
        w_jit = float(getattr(self.sim_cfg, "episode0_drive_jitter_fitness_weight", 0.0))
        w_sh = float(getattr(self.sim_cfg, "episode0_selfharm_fitness_weight", 0.0))
        return (w_pred != 0.0) or (w_thr != 0.0) or (w_jit != 0.0) or (w_sh != 0.0)

    def _episode0_selection_adjustment(self, st) -> float:
        """
        Returns the total episode-0 adjustment to add to base fitness for selection.
        Keeps backward compatibility with legacy combined knob.
        """
        if st is None:
            return 0.0

        comps = self._episode0_components(st)

        if self._use_split_episode0():
            w_pred = float(getattr(self.sim_cfg, "episode0_pred_err_fitness_weight", 0.0))
            if bool(getattr(self.sim_cfg, "disable_prediction_error_selection_pressure", False)):
                w_pred = 0.0
            w_thr = float(getattr(self.sim_cfg, "episode0_thrashing_fitness_weight", 0.0))
            w_jit = float(getattr(self.sim_cfg, "episode0_drive_jitter_fitness_weight", 0.0))
            w_sh = float(getattr(self.sim_cfg, "episode0_selfharm_fitness_weight", 0.0))

            adj = (w_pred * comps["pred"]) + (w_thr * comps["thrash"]) + (w_jit * comps["jitter"]) + (w_sh * comps["selfharm"])
        else:
            w = float(getattr(self.sim_cfg, "episode0_fitness_weight", 0.0))
            if w == 0.0:
                return 0.0
            adj = w * comps["combined"]

        try:
            st.effective_fitness_ep0_component = float(adj)
        except Exception:
            pass

        return float(adj)

    # -----------------------------
    # Effective fitness for selection
    # -----------------------------
    def _effective_fitness(self, a) -> float:
        """Effective fitness = base fitness + (optional episode-0 selection adjustment) - legacy coherence penalties."""
        f = float(base_fitness(a))

        st = getattr(a, "stats", None)
        if st is None:
            return f
        self._assert_no_direct_chemistry_selection_scalar(st)

        # (1) Episode-0 selection pressure (split or legacy)
        f_eff = f + self._episode0_selection_adjustment(st)

        # (2) Legacy coherence penalties (unchanged)
        thrash_w = float(getattr(self.sim_cfg, "coherence_thrashing_weight", 0.15))
        drive_w = float(getattr(self.sim_cfg, "coherence_drive_jitter_weight", 0.08))
        selfharm_w = float(getattr(self.sim_cfg, "coherence_selfharm_weight", 3.0))
        pred_w = float(getattr(self.sim_cfg, "coherence_pred_err_weight", 1.25))

        reset_w = float(getattr(self.sim_cfg, "coherence_reset_shock_weight", 0.35))
        ep0_pred_w = float(getattr(self.sim_cfg, "coherence_episode0_pred_err_weight", 0.75))
        if bool(getattr(self.sim_cfg, "disable_prediction_error_selection_pressure", False)):
            pred_w = 0.0
            ep0_pred_w = 0.0

        switches = float(getattr(st, "action_switches", 0))
        osc = float(getattr(st, "action_oscillations", 0))
        thrash = 0.5 * switches + 1.0 * osc

        drive_vol = float(getattr(st, "drive_jitter", getattr(st, "drive_volatility", 0.0)))
        selfharm = float(getattr(st, "selfharm_events", 0))

        pe_sum = float(getattr(st, "pred_err_sum", 0.0))
        pe_n = float(getattr(st, "pred_err_count", 0))
        pred_err_mean = pe_sum / max(1.0, pe_n)

        reset_shock = float(getattr(st, "birth_reset_shock", 0.0))

        ep0_sum = float(getattr(st, "episode0_pred_err_sum", 0.0))
        ep0_n = float(getattr(st, "episode0_pred_err_count", 0))
        ep0_pred_mean = ep0_sum / max(1.0, ep0_n)

        legacy_penalty = (
            thrash_w * thrash
            + drive_w * drive_vol
            + selfharm_w * selfharm
            + pred_w * pred_err_mean
            + reset_w * reset_shock
            + ep0_pred_w * ep0_pred_mean
        )


        # (3) Option C: Late-life predictive coherence pressure (analysis-clean)
        # Apply a weak scalar penalty ONLY after first successful reproduction.
        late_enable = bool(getattr(self.sim_cfg, "late_pred_cost_enable", False))
        late_w = float(getattr(self.sim_cfg, "late_pred_cost_weight", 0.0))

        late_mean = 0.0
        late_cost = 0.0
        try:
            lsum = float(getattr(st, "late_pred_err_sum", 0.0))
            lcount = int(getattr(st, "late_pred_err_count", 0))
            if lcount > 0:
                late_mean = float(lsum) / float(lcount)
        except Exception:
            late_mean = 0.0

        if bool(getattr(self.sim_cfg, "disable_prediction_error_selection_pressure", False)):
            late_enable = False

        if late_enable and (late_w > 0.0) and (late_mean > 0.0):
            late_cost = float(late_w) * float(late_mean)
        else:
            late_cost = 0.0

        eff = float(f_eff - legacy_penalty - late_cost)
        if eff < 0.0:
            eff = 0.0
        try:
            st.coh_thrash = float(thrash)
            st.coh_drive_jitter = float(drive_vol)
            st.coh_selfharm = float(selfharm)
            st.coh_pred_err_mean = float(pred_err_mean)
            st.coh_reset_shock = float(reset_shock)
            st.coh_ep0_pred_err_mean = float(ep0_pred_mean)

            # Option C caches
            st.late_pred_err_mean = float(late_mean)
            st.late_pred_cost = float(late_cost)

            st.coh_penalty_total = float(legacy_penalty)
            st.effective_fitness = float(eff)
            st.effective_fitness_base = float(f)
        except Exception:
            pass

        return float(eff)

    # -----------------------------
    # Selection and behavior logging (unchanged logic)
    # -----------------------------
    def _selection_and_behaviors(
        self,
        gen: int,
        alive_after: List,
        alive_id_set: Set[int],
        alive_end_ids: Set[int],
        buckets: Dict[str, float],
    ) -> Tuple[Set[int], Set[int]]:
        t0 = time.perf_counter()

        eligible_ids: Set[int] = set(alive_id_set.intersection(alive_end_ids))
        eligible_agents = [a for a in alive_after if int(a.agent_id) in eligible_ids]

        ranked_eligible = sorted(eligible_agents, key=self._effective_fitness, reverse=True)
        n_parents = max(2, int(len(ranked_eligible) * float(self.pop_cfg.parent_fraction)))

        selected_parents = ranked_eligible[:n_parents]
        selected_parent_ids: Set[int] = {int(a.agent_id) for a in selected_parents}

        buckets["selection"] = buckets.get("selection", 0.0) + (time.perf_counter() - t0)

        self._eligible_parent_agents_gen = list(eligible_agents)
        self._selected_parent_ids_gen = set(selected_parent_ids)
        self._selected_parent_agents_gen = list(selected_parents)

        for aid in list(self.behavior_logger._data.keys()):
            aid_i = int(aid)
            a = self.pop.agents.get(aid_i)

            if a is None:
                start_inj = float(self.behavior_logger.injury_start_for(aid_i, default=0.0))
                self.behavior_logger.set_outcomes(
                    agent_id=aid_i,
                    lifespan_steps=0,
                    energy_collected=0.0,
                    offspring=0,
                    attacks_made=0,
                    fitness=0.0,
                    alive_end=0,
                    eligible_for_selection=0,
                    kept_next=0,
                    injury_end=start_inj,
                )
                continue

            alive_end = int(aid_i in alive_end_ids)
            eligible = int(aid_i in eligible_ids)

            self.behavior_logger.set_outcomes(
                agent_id=aid_i,
                agent=a,
                lifespan_steps=int(getattr(a.stats, "survival_steps", 0)),
                energy_collected=float(getattr(a.stats, "energy_collected", 0.0)),
                offspring=int(getattr(a.stats, "offspring_count", 0)),
                attacks_made=int(getattr(a.stats, "attacks", 0)),
                fitness=float(base_fitness(a)),
                alive_end=alive_end,
                eligible_for_selection=eligible,
                kept_next=int(eligible and (aid_i in selected_parent_ids)),
                injury_end=float(getattr(a, "injury", 0.0)),
            )

        return eligible_ids, selected_parent_ids
