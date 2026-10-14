from __future__ import annotations

"""
simulation/runner_pkg/runner_ticks_identity_drift.py

Phase 4 — Temporal Consistency Pressure (Identity without Memory)

This module implements ONE minimal, deterministic pressure gradient:

Identity Drift Tax (capability coupling)
---------------------------------------
- Capture a fixed "probe signature" once at the end of episode-0:
    baseline_probs = softmax(policy_logits(probe_obs))

- In adulthood, periodically recompute probe_probs using the current internal state
  (for GRU agents, this depends on the current recurrent hidden state).

- Drift is measured as the mean L1 distance between current and baseline probe probs.

- Drift increases movement cost (capability penalty), applied elsewhere in
  runner_ticks_env_apply.py. This module only measures + stores drift.

Constraints respected:
- No world redesign
- No episode-0 logic changes
- No learning, no new memory system, no new recurrent state
- No randomness
- Deterministic + local + configurable + falsifiable
"""


import numpy as np


class TickIdentityDriftMixin:
    def _update_identity_drift(self, alive) -> None:
        """Update identity drift baseline and EMA drift metric.

        Hook point:
        - Call after the env tick has incremented AgentStats.survival_steps.
        - This lets us capture the baseline exactly when survival_steps == episode0_k.

        Side effects:
        - Stores baseline probe signature per slot (runner arrays).
        - Stores drift EMA per slot (runner arrays) and mirrors to agent.stats.identity_drift_ema.

        Args:
            alive: List of alive agents for this tick (shuffled order is fine).
        """
        if not bool(getattr(self.sim_cfg, "identity_drift_enable", False)):
            return
        if not alive:
            return

        # Ensure arrays exist (safe no-op if already sized)
        try:
            self._ensure_identity_arrays()
        except Exception:
            return

        k0 = int(getattr(self.sim_cfg, "episode0_k", 20))
        if k0 < 0:
            k0 = 0

        probe_every = int(getattr(self.sim_cfg, "identity_probe_every_k", 8))
        if probe_every <= 0:
            probe_every = 1

        # Tick index (we reuse the delayed-prediction clock which already advances once per tick)
        tick_index = int(getattr(self, "_pred_tick", 0))

        baseline_probs = getattr(self, "_idprobe_baseline_probs", None)
        baseline_tok = getattr(self, "_idprobe_baseline_token", None)
        drift_ema = getattr(self, "_idprobe_drift_ema", None)

        if baseline_probs is None or baseline_tok is None or drift_ema is None:
            return

        # -----------------------------
        # 1) Capture baseline signature at episode-0 end
        # -----------------------------
        baseline_need_agents = []
        baseline_need_slots = []

        for a in alive:
            try:
                st = a.stats
                if int(getattr(st, "survival_steps", 0)) != k0:
                    continue
                slot = int(getattr(a, "slot", -1))
                tok = int(getattr(a, "slot_token", 0))
                if slot < 0:
                    continue
                if int(baseline_tok[slot]) != tok:
                    baseline_need_agents.append(a)
                    baseline_need_slots.append(slot)
            except Exception:
                continue

        if baseline_need_agents:
            probs = self._probe_action_probs(baseline_need_agents)
            for i, a in enumerate(baseline_need_agents):
                try:
                    slot = int(getattr(a, "slot", -1))
                    tok = int(getattr(a, "slot_token", 0))
                    if slot < 0:
                        continue
                    baseline_probs[slot, :] = probs[i, :]
                    baseline_tok[slot] = np.int32(tok)
                    drift_ema[slot] = np.float32(0.0)
                    # Mirror for analysis convenience
                    setattr(a.stats, "identity_drift_ema", 0.0)
                except Exception:
                    continue

        # -----------------------------
        # 2) Periodic drift update in adulthood
        # -----------------------------
        if (tick_index % probe_every) != 0:
            return

        start_step = int(getattr(self.sim_cfg, "identity_drift_start_step", int(getattr(self.sim_cfg, "adult_move_penalty_start_step", 20))))
        if start_step < 0:
            start_step = 0

        alpha = float(getattr(self.sim_cfg, "identity_drift_ema_alpha", 0.25))
        if alpha < 0.0:
            alpha = 0.0
        if alpha > 1.0:
            alpha = 1.0

        scale = float(getattr(self.sim_cfg, "identity_drift_scale", 0.10))
        if scale <= 1e-9:
            scale = 0.10

        eligible_agents = []
        eligible_slots = []

        for a in alive:
            try:
                st = a.stats
                if int(getattr(st, "survival_steps", 0)) < start_step:
                    continue
                slot = int(getattr(a, "slot", -1))
                tok = int(getattr(a, "slot_token", 0))
                if slot < 0:
                    continue
                # Require a captured baseline for this exact slot token
                if int(baseline_tok[slot]) != tok:
                    continue
                eligible_agents.append(a)
                eligible_slots.append(slot)
            except Exception:
                continue

        if not eligible_agents:
            return

        probs = self._probe_action_probs(eligible_agents)

        for i, a in enumerate(eligible_agents):
            try:
                slot = int(eligible_slots[i])
                base = baseline_probs[slot, :]
                now = probs[i, :]
                drift = float(np.mean(np.abs(now - base)))
                # Normalize by scale into roughly [0,1] range (but we store raw drift for analysis)
                prev = float(drift_ema[slot])
                if alpha <= 0.0:
                    new = drift
                else:
                    new = (1.0 - alpha) * prev + alpha * drift
                drift_ema[slot] = np.float32(new)
                setattr(a.stats, "identity_drift_ema", float(new))
            except Exception:
                continue
