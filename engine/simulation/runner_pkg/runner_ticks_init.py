from __future__ import annotations

"""
simulation/runner_pkg/runner_ticks_init.py

Per-generation initialization for the tick loop.

Responsibilities:
- Reset per-generation per-agent counters used by coherence + drive bookkeeping
- Reset delayed prediction bookkeeping clock for this generation
"""

from typing import Iterable


class TickInitMixin:
    def _init_generation_tick_state(self, agents: Iterable) -> None:
        """
        Reset per-generation counters once at the start of the generation tick loop.

        Args:
            agents: Iterable of currently alive agents (Population.alive_agents()).
        """
        # Phase 3 determinism: per-slot policy action cache must not leak across generations.
        # The cache is an optimization for multi-tick batching (policy_tick_stride>1).
        # It is NOT part of the canonical simulation state and must be reset at the
        # start of each generation (and therefore also after resume).
        try:
            setattr(self, "_policy_action_cache", {})
        except Exception:
            pass
        try:
            setattr(self, "_policy_action_slot_cache", None)
        except Exception:
            pass
        self._reference_policy_cache_public_counts = {
            "source": "reference_policy_cache_service",
            "policy_cache_rows_requested": 0,
            "policy_cache_rows_hit": 0,
            "policy_cache_rows_missed": 0,
            "policy_cache_rows_refilled": 0,
            "policy_cache_full_refill_ticks": 0,
            "policy_cache_partial_refill_ticks": 0,
        }

        # Phase 3 determinism: per-tick policy index cache must not leak across generations.
        # This cache is keyed by the alive_ids + slot/token layout and can change whether
        # the policy path reuses cached tensors vs rebuilding them. If it is warm in a fresh
        # run but cleared during checkpointing/resume, gen1/tick0 can take different torch
        # execution paths and consume torch RNG differently.
        #
        # Contract: start every generation with a "cold" tick cache (same for fresh+resume).
        try:
            setattr(self, "_policy_tick_cache", None)
        except Exception:
            pass

        for a in agents:
            st = getattr(a, "stats", None)
            if st is None:
                continue

            st.action_switches = 0
            st.action_oscillations = 0
            st.drive_volatility = 0.0
            st.last_energy_delta = 0.0
            st.last_health_delta = 0.0
            st.last_injury_delta = 0.0
            st.selfharm_events = 0
            st.pred_err_sum = 0.0
            st.pred_err_count = 0
            st.prediction_update_count = 0
            st.prediction_state_delta_abs_sum = 0.0
            st.prediction_state_delta_abs_max = 0.0
            st.prediction_reset_count = 0
            # Optional: track how much "late pred cost" was paid this gen
            st.late_pred_cost_paid = 0.0
            # Life-history tradeoff: late-life coherence efficiency cache (debug/metrics)
            st.late_coherence_eff_mult = 1.0
            st.late_coherence_score = 0.0

            # Delayed prediction tickets (anticipation)
            if hasattr(st, "pred_queue"):
                try:
                    st.pred_queue.clear()
                except Exception:
                    pass

        # Global tick index within this generation (for delayed prediction scoring)
        self._pred_tick = 0
