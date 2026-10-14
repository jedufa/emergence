from __future__ import annotations

"""
simulation/runner_pkg/runner_ticks_drives.py

Internal drives update + prediction error bookkeeping.

Responsibilities:
- Update Agent.drives from cached physiology deltas
- Maintain prediction error signals (immediate or delayed horizon)
- Apply optional late-life prediction cost (unchanged behavior)

Performance notes:
This code runs every tick for every alive agent. The original refactor kept
behavior but introduced a lot of tiny NumPy allocations (per-agent arrays,
astype copies, np.clip calls). Those add up fast.

This version keeps semantics but:
- Uses scalar math for the 4 drives (no per-agent temporary arrays)
- Avoids repeated np.clip/np.asarray allocations
- Avoids importing numpy inside the hot loop
- Only uses small NumPy ops where it matters (noise draw)
"""

import numpy as np
import time
from collections import deque as _deque

from agents.cognition import effective_prediction_learning_rate_for_genome, prediction_ablation_flags
from perf.cxxrewrite4_upstream_resident import (
    resident_scalar_sync_fail_closed as cxxrewrite4_resident_scalar_sync_fail_closed,
    sync_agents_scalar_state as cxxrewrite4_sync_agents_scalar_state,
)

from .runner_ticks_common import (
    ACT_ATTACK,
    ACT_REST,
    ACT_GATHER,
    DRIVE_HUNGER,
    DRIVE_PAIN,
    DRIVE_FATIGUE,
    DRIVE_FEAR,
    N_DRIVES,
    clamp01,
)


def _clip01(x: float) -> float:
    """Fast scalar clamp to [0,1]."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    return x


def _clip(x: float, lo: float, hi: float) -> float:
    """Fast scalar clamp to [lo,hi]."""
    if x <= lo:
        return lo
    if x >= hi:
        return hi
    return x


class TickDrivesMixin:

    def _ensure_drive_scratch(self, n: int) -> None:
        """Ensure per-tick NumPy scratch buffers for drive update.

        This reduces allocator overhead in the hottest tick loop without changing any
        semantics. This helper MUST NOT draw RNG and MUST NOT mutate simulation state
        beyond (re)allocating scratch arrays on the runner instance.
        """
        try:
            n_i = int(n)
        except Exception:
            n_i = 0
        if n_i <= 0:
            return

        cur = int(getattr(self, "_drive_scratch_n", 0) or 0)
        # Fast path: already large enough.
        if cur >= n_i and hasattr(self, "_drive_slots_scratch"):
            return

        setattr(self, "_drive_scratch_n", n_i)
        # int32 slots + float32 scalar buffers (per alive agent)
        self._drive_slots_scratch = np.empty((n_i,), dtype=np.int32)
        self._drive_e_n_scratch = np.empty((n_i,), dtype=np.float32)
        self._drive_h_n_scratch = np.empty((n_i,), dtype=np.float32)
        self._drive_de_scratch = np.empty((n_i,), dtype=np.float32)
        self._drive_dh_scratch = np.empty((n_i,), dtype=np.float32)
        self._drive_dinj_scratch = np.empty((n_i,), dtype=np.float32)
        self._drive_inertia_scratch = np.empty((n_i,), dtype=np.float32)
        self._drive_noise_scratch = np.empty((n_i,), dtype=np.float32)
        # RNG noise scratch (per-agent, 4 drives). We generate float64 then cast to float32
        # to preserve the exact values from `normal(...).astype(np.float32)` without allocating.
        self._drive_nvec64_scratch = np.empty((n_i, 4), dtype=np.float64)
        self._drive_nvec32_scratch = np.empty((n_i, 4), dtype=np.float32)

    def _update_internal_drives(self, alive, actions_np, tick_index: int, buckets: Optional[Dict[str, float]] = None) -> None:
        """
        Update internal drives based on cached physiology deltas.

        Performance-critical path:
        - Drives are stored as dense runner-level arrays indexed by agent.slot.
        - We vectorize the 4-drive update across the batch.
        - Prediction error bookkeeping remains per-agent (queues + late-life tax).

        Semantics are preserved versus the scalar implementation.
        """
        n = len(alive)
        if n == 0:
            return

        # Ensure runner-level drive arrays exist (RunnerBase provides this).
        if hasattr(self, "_ensure_drive_arrays"):
            self._ensure_drive_arrays()

        # Optional perf micro-buckets (no semantic impact).
        b = buckets if isinstance(buckets, dict) else None
        drive_tokens = getattr(self, "_drive_slot_token", None)

        hunger_gain = float(getattr(self.sim_cfg, "drive_hunger_gain", 0.08))
        hunger_decay = float(getattr(self.sim_cfg, "drive_hunger_decay", 0.02))

        pain_gain = float(getattr(self.sim_cfg, "drive_pain_gain", 0.18))
        pain_decay = float(getattr(self.sim_cfg, "drive_pain_decay", 0.03))

        fatigue_gain = float(getattr(self.sim_cfg, "drive_fatigue_gain", 0.05))
        fatigue_rest_relief = float(getattr(self.sim_cfg, "drive_fatigue_rest_relief", 0.10))
        fatigue_decay = float(getattr(self.sim_cfg, "drive_fatigue_decay", 0.01))

        fear_gain = float(getattr(self.sim_cfg, "drive_fear_gain", 0.22))
        fear_decay = float(getattr(self.sim_cfg, "drive_fear_decay", 0.03))

        e_norm = float(getattr(self.env_cfg, "max_energy", 1.0)) or 1.0
        h_norm = float(getattr(self.env_cfg, "start_health", 1.0)) or 1.0

        # -----------------------------
        # Option C NOTE:
        # Late-life *selection* cost is applied at parent selection time (runner_selection.py),
        # NOT here. This tick module only maintains prediction error bookkeeping signals.
        #
        # Legacy behavior (deprecated):
        # A per-tick energy tax proportional to later-life prediction error.
        # This is a world-dynamics confound, so it is gated behind separate flags.
        # -----------------------------
        disable_pred_pressure = bool(getattr(self.sim_cfg, "disable_prediction_error_selection_pressure", False))
        late_energy_tax_enable = bool(getattr(self.sim_cfg, "late_pred_energy_tax_enable", False)) and not disable_pred_pressure
        late_energy_tax_w = float(getattr(self.sim_cfg, "late_pred_energy_tax_weight", 0.0))
        late_energy_tax_cap = float(getattr(self.sim_cfg, "late_pred_energy_tax_cap", 5.0))
        if late_energy_tax_cap <= 1e-9:
            late_energy_tax_cap = 5.0

        default_start = int(getattr(self.sim_cfg, "episode0_k", 20))
        late_energy_tax_start = int(getattr(self.sim_cfg, "late_pred_energy_tax_start_step", default_start))
        if late_energy_tax_start < 0:
            late_energy_tax_start = 0
        # Adult-coherence capability coupling (Part B): maintain per-agent EMA of pred error.
        adult_move_enable = bool(getattr(self.sim_cfg, "adult_move_penalty_enable", False))
        adult_start = int(getattr(self.sim_cfg, "adult_move_penalty_start_step", late_energy_tax_start))
        if adult_start < 0:
            adult_start = 0
        adult_alpha = float(getattr(self.sim_cfg, "adult_pred_err_ema_alpha", 0.1))
        if adult_alpha <= 0.0:
            adult_alpha = 0.0
        elif adult_alpha > 1.0:
            adult_alpha = 1.0


        # Episode-0 window length
        k0 = int(getattr(self.sim_cfg, "episode0_k", 20))

        # Pre-allocate per-agent scalar arrays (filled in one pass).
        # These are hot-loop scratch buffers, so we reuse them across ticks to avoid
        # allocator churn (no semantic changes).
        if hasattr(self, "_ensure_drive_scratch"):
            try:
                self._ensure_drive_scratch(n)
            except Exception:
                pass

        slots = getattr(self, "_drive_slots_scratch", None)
        e_n = getattr(self, "_drive_e_n_scratch", None)
        h_n = getattr(self, "_drive_h_n_scratch", None)
        de = getattr(self, "_drive_de_scratch", None)
        dh = getattr(self, "_drive_dh_scratch", None)
        dinj = getattr(self, "_drive_dinj_scratch", None)
        inertia_arr = getattr(self, "_drive_inertia_scratch", None)
        noise_arr = getattr(self, "_drive_noise_scratch", None)

        if (
            slots is None
            or e_n is None
            or h_n is None
            or de is None
            or dh is None
            or dinj is None
            or inertia_arr is None
            or noise_arr is None
        ):
            # Fallback (should be rare): allocate fresh arrays if scratch is unavailable.
            slots = np.empty((n,), dtype=np.int32)
            e_n = np.empty((n,), dtype=np.float32)
            h_n = np.empty((n,), dtype=np.float32)
            de = np.empty((n,), dtype=np.float32)
            dh = np.empty((n,), dtype=np.float32)
            dinj = np.empty((n,), dtype=np.float32)
            inertia_arr = np.empty((n,), dtype=np.float32)
            noise_arr = np.empty((n,), dtype=np.float32)
        else:
            # Use only the first n entries.
            slots = slots[:n]
            e_n = e_n[:n]
            h_n = h_n[:n]
            de = de[:n]
            dh = dh[:n]
            dinj = dinj[:n]
            inertia_arr = inertia_arr[:n]
            noise_arr = noise_arr[:n]

        # Fill defaults (matches previous np.full/np.zeros semantics).
        slots.fill(np.int32(-1))
        e_n.fill(np.float32(0.0))
        h_n.fill(np.float32(0.0))
        de.fill(np.float32(0.0))
        dh.fill(np.float32(0.0))
        dinj.fill(np.float32(0.0))
        inertia_arr.fill(np.float32(0.85))
        noise_arr.fill(np.float32(0.02))

        acts = np.asarray(actions_np, dtype=np.int32, order="C", copy=False)

        # ----------------------
        # Pass 1: gather scalars + handle prediction error / late-tax bookkeeping
        # ----------------------
        # Hoist prediction config out of the per-agent loop (this runs every tick).
        _sim_cfg = getattr(self, "sim_cfg", None)
        cxxrewrite4_late_tax_agents = []
        pred_horizon = int(getattr(_sim_cfg, "pred_horizon", 2)) if _sim_cfg is not None else 2
        if pred_horizon < 0:
            pred_horizon = 0
        elif pred_horizon > 8:
            pred_horizon = 8
        inj_scale = float(getattr(_sim_cfg, "pred_injury_scale", 1.0)) if _sim_cfg is not None else 1.0
        pred_flags = prediction_ablation_flags(_sim_cfg)
        pred_ablate = bool(pred_flags.get("ablate", False))
        pred_force_disabled = bool(pred_flags.get("force_disabled", False))
        pred_force_frozen = bool(pred_flags.get("force_frozen", False))
        pred_force_zero_lr = bool(pred_flags.get("force_zero_lr", False))
        pred_force_default = bool(pred_flags.get("force_default", False))
        pred_reset_each_tick = bool(pred_flags.get("reset_each_tick", False))
        pred_state_available_tick = not bool(pred_ablate or pred_force_disabled)
        dev_plasticity_decay_enable = bool(getattr(_sim_cfg, "dev_plasticity_decay_enable", False)) if _sim_cfg is not None else False
        dev_plasticity_half_life_steps = int(getattr(_sim_cfg, "dev_plasticity_half_life_steps", 0) or 0) if dev_plasticity_decay_enable else 0
        resolve_dev_scalars = getattr(self, "resolve_developmental_effective_scalars_for_agent", None)
        resolve_dev_expression = getattr(self, "resolve_developmental_expression_for_agent", None)
        sync_drive_slot = getattr(self, "_sync_drive_slot", None)

        _t_p1 = time.perf_counter()
        for i, a in enumerate(alive):
            if not getattr(a, "alive", False):
                continue
            slot = int(getattr(a, "slot", -1))

            # Ensure slot drive state initialized (handles slot reuse).
            #
            # Critical perf note:
            # RunnerBase._sync_drive_slot() also mirrors runner arrays back into a.drives
            # on the *token-match* fast path. That mirror is already performed later in
            # this function (Pass 2), and checkpoints are only taken at generation
            # boundaries, so doing it here every tick is wasted work.
            if slot >= 0 and drive_tokens is not None and callable(sync_drive_slot):
                try:
                    tok = int(getattr(a, "slot_token", -1))
                    if tok != int(drive_tokens[slot]):
                        sync_drive_slot(a)
                except Exception:
                    # Fallback to safe behavior if indexing/token coercion fails
                    try:
                        sync_drive_slot(a)
                    except Exception:
                        pass
            if slot < 0:
                continue
            slots[i] = np.int32(slot)

            st = getattr(a, "stats", None)

            e_n[i] = np.float32(clamp01(float(getattr(a, "energy", 0.0)) / e_norm))
            h_n[i] = np.float32(clamp01(float(getattr(a, "health", 0.0)) / h_norm))

            if st is not None:
                de[i] = np.float32(float(getattr(st, "last_energy_delta", 0.0)))
                dh[i] = np.float32(float(getattr(st, "last_health_delta", 0.0)))
                dinj[i] = np.float32(float(getattr(st, "last_injury_delta", 0.0)))

            g = getattr(a, "genome", None)
            _eff = {}
            _pred_lr_raw = None
            _scalars = resolve_dev_scalars(a) if g is not None and callable(resolve_dev_scalars) else None
            if _scalars is not None:
                inertia = float(_scalars[3])
                noise = float(_scalars[4])
                _pred_lr_raw = float(_scalars[5])
            else:
                if g is not None and callable(resolve_dev_expression):
                    try:
                        _dev = resolve_dev_expression(a)
                        _eff = _dev.get("effective", {}) if isinstance(_dev, dict) else {}
                    except Exception:
                        _eff = {}
                inertia = float(_eff.get("drive_inertia", getattr(g, "drive_inertia", 0.85))) if g is not None else 0.85
                noise = float(_eff.get("drive_noise", getattr(g, "drive_noise", 0.02))) if g is not None else 0.02
            inertia_arr[i] = np.float32(_clip(inertia, 0.0, 0.999))

            noise_arr[i] = np.float32(_clip(noise, 0.0, 0.10))
            # ---------------------------------------------------------
            # Chaos Week 2: Developmental plasticity decay (minimal v1)
            # ---------------------------------------------------------
            # Plasticity here is "how easily the agent updates internal state".
            # We implement a simple age-based exponential decay that reduces:
            #   (a) drive_noise (drive volatility)
            #   (b) pred_lr (learning rate for outcome expectations)
            #
            # decay_factor(age) = 0.5 ** (age / half_life_steps)
            # So after half_life_steps, effective plasticity is half.
            decay_factor = 1.0
            try:
                if st is not None and dev_plasticity_decay_enable:
                    hl = int(dev_plasticity_half_life_steps)
                    if hl > 0:
                        age = int(getattr(st, "survival_steps", 0))
                        decay_factor = 0.5 ** (float(age) / float(hl))
                        # Reduce drive noise as agents age (lock-in).
                        noise_arr[i] = np.float32(_clip(float(noise_arr[i]) * float(decay_factor), 0.0, 0.10))
            except Exception:
                decay_factor = 1.0


            # ----------------------
            # Prediction error logic (per-agent, preserved)
            # ----------------------
            if st is None:
                continue

            act = int(acts[i])

            if _pred_lr_raw is None:
                pred_lr = float(effective_prediction_learning_rate_for_genome(g, sim_cfg=_sim_cfg, developmental_effective=_eff))
            else:
                pred_lr = _clip(float(_pred_lr_raw), 0.0, 1.0)
                if (
                    pred_ablate
                    or pred_force_disabled
                    or pred_force_frozen
                    or pred_force_zero_lr
                ):
                    pred_lr = 0.0
                elif pred_force_default:
                    pred_lr = 0.08
            # Apply developmental plasticity decay to prediction learning rate.
            # If decay is disabled, decay_factor stays at 1.0.
            try:
                pred_lr = float(pred_lr) * float(decay_factor)
            except Exception:
                pass
            pred_lr = _clip(pred_lr, 0.0, 1.0)
            pred_state_available = pred_state_available_tick
            pred_update_enabled = bool(pred_state_available and pred_lr > 0.0 and not pred_force_frozen)
            if pred_reset_each_tick or not pred_state_available:
                try:
                    st.exp_dE[:] = 0.0
                    st.exp_dInj[:] = 0.0
                    st.prediction_reset_count = int(getattr(st, "prediction_reset_count", 0)) + 1
                except Exception:
                    pass

            pe = 0.0
            if pred_horizon <= 0:
                try:
                    exp_de = float(st.exp_dE[act])
                    exp_dinj = float(st.exp_dInj[act])
                except Exception:
                    exp_de, exp_dinj = 0.0, 0.0

                err = abs(float(de[i]) - exp_de) + inj_scale * abs(float(dinj[i]) - exp_dinj)
                pe = float(err)
                # Part B: update adult pred_err EMA (for movement penalty coupling)
                if adult_move_enable and adult_alpha > 0.0:
                    try:
                        if int(getattr(st, "survival_steps", 0)) >= adult_start:
                            prev_ema = float(getattr(st, "adult_pred_err_ema", 0.0))
                            st.adult_pred_err_ema = (1.0 - adult_alpha) * prev_ema + adult_alpha * float(err)
                    except Exception:
                        pass

                if pred_state_available:
                    st.pred_err_sum = float(getattr(st, "pred_err_sum", 0.0)) + float(err)
                    st.pred_err_count = int(getattr(st, "pred_err_count", 0)) + 1

                    try:
                        if int(getattr(st, "survival_steps", 0)) <= k0:
                            st.episode0_pred_err_sum = float(getattr(st, "episode0_pred_err_sum", 0.0)) + float(err)
                            st.episode0_pred_err_count = int(getattr(st, "episode0_pred_err_count", 0)) + 1
                    except Exception:
                        pass

                if pred_update_enabled:
                    try:
                        new_de = float(exp_de + pred_lr * (float(de[i]) - exp_de))
                        new_dinj = float(exp_dinj + pred_lr * (float(dinj[i]) - exp_dinj))
                        delta_abs = abs(new_de - exp_de) + abs(new_dinj - exp_dinj)
                        st.exp_dE[act] = new_de
                        st.exp_dInj[act] = new_dinj
                        st.prediction_update_count = int(getattr(st, "prediction_update_count", 0)) + 1
                        st.prediction_state_delta_abs_sum = float(getattr(st, "prediction_state_delta_abs_sum", 0.0)) + float(delta_abs)
                        st.prediction_state_delta_abs_max = max(float(getattr(st, "prediction_state_delta_abs_max", 0.0)), float(delta_abs))
                    except Exception:
                        pass
            else:
                try:
                    pq = st.pred_queue
                except Exception:
                    pq = []

                # Resume / checkpoint robustness: pred_queue is expected to be a deque
                # (needs .popleft()), but JSON-based checkpoints may restore it as a list.
                if isinstance(pq, list):
                    pq = _deque(pq)
                    try:
                        st.pred_queue = pq
                    except Exception:
                        pass

                while pq and int(pq[0][0]) <= int(tick_index):
                    due, a_act, a_exp_de, a_exp_dinj, e0, inj0 = pq.popleft()
                    actual_de = float(getattr(a, "energy", 0.0)) - float(e0)
                    actual_dinj = float(getattr(a, "injury", 0.0)) - float(inj0)
                    pe = abs(actual_de - float(a_exp_de)) + inj_scale * abs(actual_dinj - float(a_exp_dinj))
                    # Part B: update adult pred_err EMA (for movement penalty coupling)
                    if adult_move_enable and adult_alpha > 0.0:
                        try:
                            if int(getattr(st, "survival_steps", 0)) >= adult_start:
                                prev_ema = float(getattr(st, "adult_pred_err_ema", 0.0))
                                st.adult_pred_err_ema = (1.0 - adult_alpha) * prev_ema + adult_alpha * float(pe)
                        except Exception:
                            pass


                    if pred_state_available:
                        st.pred_err_sum = float(getattr(st, "pred_err_sum", 0.0)) + float(pe)
                        st.pred_err_count = int(getattr(st, "pred_err_count", 0)) + 1

                        try:
                            if int(getattr(st, "survival_steps", 0)) <= k0:
                                st.episode0_pred_err_sum = float(getattr(st, "episode0_pred_err_sum", 0.0)) + float(pe)
                                st.episode0_pred_err_count = int(getattr(st, "episode0_pred_err_count", 0)) + 1
                        except Exception:
                            pass

                    if pred_update_enabled:
                        try:
                            cur_de = float(st.exp_dE[int(a_act)])
                            cur_dinj = float(st.exp_dInj[int(a_act)])
                            new_de = float(cur_de + pred_lr * (actual_de - cur_de))
                            new_dinj = float(cur_dinj + pred_lr * (actual_dinj - cur_dinj))
                            delta_abs = abs(new_de - cur_de) + abs(new_dinj - cur_dinj)
                            st.exp_dE[int(a_act)] = new_de
                            st.exp_dInj[int(a_act)] = new_dinj
                            st.prediction_update_count = int(getattr(st, "prediction_update_count", 0)) + 1
                            st.prediction_state_delta_abs_sum = float(getattr(st, "prediction_state_delta_abs_sum", 0.0)) + float(delta_abs)
                            st.prediction_state_delta_abs_max = max(float(getattr(st, "prediction_state_delta_abs_max", 0.0)), float(delta_abs))
                        except Exception:
                            pass

                if pred_state_available:
                    try:
                        exp_de = float(st.exp_dE[act])
                        exp_dinj = float(st.exp_dInj[act])
                    except Exception:
                        exp_de, exp_dinj = 0.0, 0.0

                    due_tick = int(tick_index) + int(pred_horizon)
                    try:
                        pq.append(
                            (
                                due_tick,
                                int(act),
                                float(exp_de),
                                float(exp_dinj),
                                float(getattr(a, "energy", 0.0)),
                                float(getattr(a, "injury", 0.0)),
                            )
                        )
                    except Exception:
                        pass

            if act == ACT_ATTACK and float(dinj[i]) > 0.0:
                st.selfharm_events = int(getattr(st, "selfharm_events", 0)) + 1

            # -----------------------------
            # Option C: late-life prediction error window (post-first-reproduction)
            # -----------------------------
            # We accumulate a *late-life* prediction error mean that starts strictly AFTER
            # the first successful reproduction event (stats.first_repro_step).
            #
            # This is analysis-clean: we do NOT change agent energy/health here.
            try:
                fr = getattr(st, "first_repro_step", None)
                if pred_state_available and fr is not None:
                    age = int(getattr(st, "survival_steps", 0))
                    if age > int(fr):  # strictly after first reproduction tick
                        st.late_pred_err_sum = float(getattr(st, "late_pred_err_sum", 0.0)) + float(pe)
                        st.late_pred_err_count = int(getattr(st, "late_pred_err_count", 0)) + 1
            except Exception:
                pass

            # -----------------------------
            # Deprecated legacy: per-tick energy tax (confound) if explicitly enabled.
            # -----------------------------
            if late_energy_tax_enable and (late_energy_tax_w > 0.0):
                age = int(getattr(st, "survival_steps", 0))
                if age >= late_energy_tax_start:
                    total_n = int(getattr(st, "pred_err_count", 0))
                    ep0_n = int(getattr(st, "episode0_pred_err_count", 0))
                    later_n = int(total_n - ep0_n)

                    if later_n > 0:
                        total_sum = float(getattr(st, "pred_err_sum", 0.0))
                        ep0_sum = float(getattr(st, "episode0_pred_err_sum", 0.0))
                        later_sum = float(total_sum - ep0_sum)
                        later_mean = float(later_sum) / float(later_n)

                        frac = clamp01(later_mean / float(late_energy_tax_cap))
                        cost = float(late_energy_tax_w) * float(frac)

                        if cost > 0.0:
                            a.energy = max(0.0, float(getattr(a, "energy", 0.0)) - cost)
                            st.late_pred_cost_paid = float(getattr(st, "late_pred_cost_paid", 0.0)) + float(cost)
                            cxxrewrite4_late_tax_agents.append(a)

        try:
            cxxrewrite4_sync_agents_scalar_state(self, cxxrewrite4_late_tax_agents, stage="late_prediction_energy_tax")
        except Exception:
            if cxxrewrite4_resident_scalar_sync_fail_closed(self.sim_cfg):
                raise

        if b is not None:
            b["drive_p1_gather"] = b.get("drive_p1_gather", 0.0) + (time.perf_counter() - _t_p1)

        # ----------------------
        # Vectorized drive update for valid slots
        # ----------------------
        valid = slots >= 0
        if not bool(np.any(valid)):
            return

        idx = np.nonzero(valid)[0]
        m = int(idx.shape[0])
        slots_v = slots[idx]

        # Defensive: ensure runner drive arrays are large enough for the maximum slot index
        # we will touch this tick. This prevents rare out-of-bounds if drive arrays were
        # initialized via a fallback sizing path.
        try:
            need = int(np.max(slots_v)) + 1
            if hasattr(self, "_ensure_drive_arrays"):
                self._ensure_drive_arrays(required_slots=need)
        except Exception:
            pass


        _t_vec = time.perf_counter()

        # Pull previous drives (copy for delta computation).
        # Indexing already produces a copy, so avoid an extra .copy().
        prev_h0 = self._drive_hunger[slots_v].astype(np.float32, copy=False)
        prev_p0 = self._drive_pain[slots_v].astype(np.float32, copy=False)
        prev_f0 = self._drive_fatigue[slots_v].astype(np.float32, copy=False)
        prev_fr0 = self._drive_fear[slots_v].astype(np.float32, copy=False)

        e_v = e_n[idx]
        h_v = h_n[idx]
        de_v = de[idx]
        dh_v = dh[idx]
        dinj_v = dinj[idx]
        act_v = acts[idx]
        inertia_v = inertia_arr[idx]
        noise_v = noise_arr[idx]

        # Hunger target
        tgt_h = prev_h0 + np.float32(hunger_gain) * (np.float32(1.0) - e_v)
        tgt_h = tgt_h + np.float32(0.04) * (-de_v / np.float32(max(1e-6, e_norm)))
        tgt_h = tgt_h - np.float32(hunger_decay)

        gather_mask = (act_v == np.int32(ACT_GATHER)) & (de_v > np.float32(0.0))
        if bool(np.any(gather_mask)):
            tgt_h[gather_mask] = tgt_h[gather_mask] - np.float32(0.20) * (de_v[gather_mask] / np.float32(max(1e-6, e_norm)))

        np.clip(tgt_h, 0.0, 1.0, out=tgt_h)

        # Pain target
        tgt_p = prev_p0.copy()
        inj_mask = dinj_v > np.float32(0.0)
        if bool(np.any(inj_mask)):
            tgt_p[inj_mask] = tgt_p[inj_mask] + np.float32(pain_gain) * np.minimum(np.float32(1.0), dinj_v[inj_mask] / np.float32(1.0))

        dh_mask = dh_v < np.float32(0.0)
        if bool(np.any(dh_mask)):
            tgt_p[dh_mask] = tgt_p[dh_mask] + np.float32(0.10) * np.minimum(np.float32(1.0), (-dh_v[dh_mask]) / np.float32(max(1e-6, h_norm)))

        tgt_p = tgt_p - np.float32(pain_decay)
        np.clip(tgt_p, 0.0, 1.0, out=tgt_p)

        # Fatigue target
        tgt_f = prev_f0.copy()
        rest_mask = act_v == np.int32(ACT_REST)
        tgt_f[rest_mask] = tgt_f[rest_mask] - np.float32(fatigue_rest_relief)
        tgt_f[~rest_mask] = tgt_f[~rest_mask] + np.float32(fatigue_gain)
        tgt_f = tgt_f - np.float32(fatigue_decay)
        np.clip(tgt_f, 0.0, 1.0, out=tgt_f)

        # Fear target
        tgt_fr = prev_fr0.copy()
        if bool(np.any(inj_mask)):
            tgt_fr[inj_mask] = tgt_fr[inj_mask] + np.float32(fear_gain) * np.minimum(np.float32(1.0), dinj_v[inj_mask] / np.float32(1.0))
        tgt_fr = tgt_fr + np.float32(0.05) * (np.float32(1.0) - h_v)
        tgt_fr = tgt_fr - np.float32(fear_decay)
        np.clip(tgt_fr, 0.0, 1.0, out=tgt_fr)

        # Inertia interpolation
        inv = (np.float32(1.0) - inertia_v).astype(np.float32, copy=False)
        new_h = inertia_v * prev_h0 + inv * tgt_h
        new_p = inertia_v * prev_p0 + inv * tgt_p
        new_f = inertia_v * prev_f0 + inv * tgt_f
        new_fr = inertia_v * prev_fr0 + inv * tgt_fr

        if b is not None:
            b["drive_vec_update"] = b.get("drive_vec_update", 0.0) + (time.perf_counter() - _t_vec)

        _t_noise = time.perf_counter()

        # Noise
        if bool(np.any(noise_v > np.float32(0.0))):
            try:
                m = int(idx.shape[0])
                nvec_override = None
                overrides = getattr(self, "_cxxbackbone10_checkpoint_tensor_overrides", None)
                if isinstance(overrides, dict):
                    samples = overrides.get("drive_noise_samples_by_tick_slot")
                    if hasattr(samples, "detach"):
                        samples = samples.detach().cpu().numpy()
                    if samples is not None:
                        samples = np.asarray(samples, dtype=np.float32)
                        if (
                            samples.ndim == 3
                            and 0 <= int(tick_index) < int(samples.shape[0])
                            and int(samples.shape[2]) >= 4
                            and int(np.max(slots_v, initial=-1)) < int(samples.shape[1])
                        ):
                            nvec_override = np.asarray(samples[int(tick_index), slots_v, :4], dtype=np.float32)

                # Fast path: reuse scratch buffers + avoid per-tick allocations.
                n64 = getattr(self, "_drive_nvec64_scratch", None)
                n32 = getattr(self, "_drive_nvec32_scratch", None)
                if nvec_override is not None:
                    try:
                        discard64 = getattr(self, "_drive_nvec64_scratch", None)
                        if discard64 is not None and getattr(discard64, "shape", (0, 0))[0] >= m:
                            self.np_rng.standard_normal(size=(m, 4), dtype=np.float64, out=discard64[:m])
                        else:
                            self.np_rng.normal(0.0, 1.0, size=(m, 4))
                    except Exception:
                        pass
                    nv = noise_v.reshape(-1, 1).astype(np.float32, copy=False)
                    new_h = new_h + nv[:, 0] * nvec_override[:, 0]
                    new_p = new_p + nv[:, 0] * nvec_override[:, 1]
                    new_f = new_f + nv[:, 0] * nvec_override[:, 2]
                    new_fr = new_fr + nv[:, 0] * nvec_override[:, 3]
                elif (
                    n64 is not None
                    and n32 is not None
                    and getattr(n64, "shape", (0, 0))[0] >= m
                    and getattr(n32, "shape", (0, 0))[0] >= m
                ):
                    n64v = n64[:m]
                    n32v = n32[:m]

                    # IMPORTANT (determinism + semantic stability):
                    # - `Generator.normal(0,1)` and `Generator.standard_normal(dtype=float64)`
                    #   produce identical float64 streams in NumPy for the same RNG state.
                    # - We then cast float64->float32 via assignment, matching
                    #   `.astype(np.float32)` without allocating a new array.
                    self.np_rng.standard_normal(size=(m, 4), dtype=np.float64, out=n64v)
                    n32v[...] = n64v  # exact float64->float32 cast

                    nv = noise_v.astype(np.float32, copy=False)
                    new_h = new_h + nv * n32v[:, 0]
                    new_p = new_p + nv * n32v[:, 1]
                    new_f = new_f + nv * n32v[:, 2]
                    new_fr = new_fr + nv * n32v[:, 3]
                else:
                    # Fallback: original allocation-heavy path.
                    nvec = self.np_rng.normal(0.0, 1.0, size=(m, 4)).astype(np.float32, copy=False)
                    nv = noise_v.reshape(-1, 1).astype(np.float32, copy=False)
                    new_h = new_h + nv[:, 0] * nvec[:, 0]
                    new_p = new_p + nv[:, 0] * nvec[:, 1]
                    new_f = new_f + nv[:, 0] * nvec[:, 2]
                    new_fr = new_fr + nv[:, 0] * nvec[:, 3]
            except Exception:
                pass

        np.clip(new_h, 0.0, 1.0, out=new_h)
        np.clip(new_p, 0.0, 1.0, out=new_p)
        np.clip(new_f, 0.0, 1.0, out=new_f)
        np.clip(new_fr, 0.0, 1.0, out=new_fr)

        # Write back to runner arrays
        self._drive_hunger[slots_v] = new_h
        self._drive_pain[slots_v] = new_p
        self._drive_fatigue[slots_v] = new_f
        self._drive_fear[slots_v] = new_fr

        if b is not None:
            b["drive_noise"] = b.get("drive_noise", 0.0) + (time.perf_counter() - _t_noise)

        _t_p2 = time.perf_counter()

        # ----------------------
        # Pass 2: mirror to Agent.drives + drive metrics (volatility/jitter)
        # ----------------------
        for j in range(m):
            i = int(idx[j])
            a = alive[i]
            if not getattr(a, "alive", False):
                continue

            st = getattr(a, "stats", None)

            # Mirror drives back onto the Agent object (compatibility)
            d = getattr(a, "drives", None)
            if d is not None and getattr(d, "shape", None) is not None and int(d.shape[0]) >= 4:
                try:
                    d[DRIVE_HUNGER] = new_h[j]
                    d[DRIVE_PAIN] = new_p[j]
                    d[DRIVE_FATIGUE] = new_f[j]
                    d[DRIVE_FEAR] = new_fr[j]
                except Exception:
                    pass

            if st is None:
                continue

            # Volatility = sum(|delta|)
            dh0 = float(new_h[j] - prev_h0[j])
            dp0 = float(new_p[j] - prev_p0[j])
            df0 = float(new_f[j] - prev_f0[j])
            dfr0 = float(new_fr[j] - prev_fr0[j])

            vol = abs(dh0) + abs(dp0) + abs(df0) + abs(dfr0)
            st.drive_volatility = float(
                np.float32(np.float32(getattr(st, "drive_volatility", 0.0)) + np.float32(vol))
            )

            # Jitter: compare current delta vs previous delta
            jitter_inst = 0.0
            try:
                prev_delta = getattr(st, "prev_drive_delta", None)
                if prev_delta is not None and getattr(prev_delta, "shape", None) == (N_DRIVES,):
                    jitter_inst = (
                        abs(dh0 - float(prev_delta[DRIVE_HUNGER]))
                        + abs(dp0 - float(prev_delta[DRIVE_PAIN]))
                        + abs(df0 - float(prev_delta[DRIVE_FATIGUE]))
                        + abs(dfr0 - float(prev_delta[DRIVE_FEAR]))
                    )
                    st.drive_jitter = float(
                        np.float32(np.float32(getattr(st, "drive_jitter", 0.0)) + np.float32(jitter_inst))
                    )

                    try:
                        prev_delta[DRIVE_HUNGER] = np.float32(dh0)
                        prev_delta[DRIVE_PAIN] = np.float32(dp0)
                        prev_delta[DRIVE_FATIGUE] = np.float32(df0)
                        prev_delta[DRIVE_FEAR] = np.float32(dfr0)
                    except Exception:
                        st.prev_drive_delta = np.asarray([dh0, dp0, df0, dfr0], dtype=np.float32)
                else:
                    st.prev_drive_delta = np.asarray([dh0, dp0, df0, dfr0], dtype=np.float32)
            except Exception:
                pass

            # Episode-0 accumulators
            try:
                if int(getattr(st, "survival_steps", 0)) <= k0:
                    st.episode0_drive_jitter_sum = float(
                        np.float32(
                            np.float32(getattr(st, "episode0_drive_jitter_sum", 0.0))
                            + np.float32(jitter_inst)
                        )
                    )
                    if int(act_v[j]) == int(ACT_ATTACK) and float(dinj_v[j]) > 0.0:
                        st.episode0_selfharm_events = int(getattr(st, "episode0_selfharm_events", 0)) + 1
            except Exception:
                pass

        if b is not None:
            b["drive_p2_mirror"] = b.get("drive_p2_mirror", 0.0) + (time.perf_counter() - _t_p2)
