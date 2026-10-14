from __future__ import annotations

"""
simulation/runner_pkg/runner_ticks_coherence.py

Coherence bookkeeping for actions taken over time (switching / oscillation).

This is intentionally cheap and runs before the environment commit so it can
observe the "intended action" the policy selected for the tick.
"""


class TickCoherenceMixin:
    def _update_action_coherence(self, alive, actions_np) -> None:
        """
        Update per-agent switching/oscillation counters.

        Args:
            alive: List of alive agents used this tick.
            actions_np: Numpy array of actions aligned to `alive`.
        """
        for i, a in enumerate(alive):
            if not a.alive:
                continue
            st = getattr(a, "stats", None)
            if st is None:
                continue
            act = int(actions_np[i])

            # update oscillation / switching
            if st.last_action != -1 and act != st.last_action:
                st.action_switches += 1
                # Episode-0: early-life switching
                try:
                    k0 = int(getattr(self.sim_cfg, "episode0_k", 20))
                    if int(getattr(st, "survival_steps", 0)) < k0:
                        st.episode0_switches = int(getattr(st, "episode0_switches", 0)) + 1
                except Exception:
                    pass
            if st.last_last_action != -1 and act == st.last_last_action and act != st.last_action:
                st.action_oscillations += 1
                # Episode-0: early-life oscillations
                try:
                    k0 = int(getattr(self.sim_cfg, "episode0_k", 20))
                    if int(getattr(st, "survival_steps", 0)) < k0:
                        st.episode0_oscillations = int(getattr(st, "episode0_oscillations", 0)) + 1
                except Exception:
                    pass
            st.last_last_action = st.last_action
            st.last_action = act

            try:
                ra = getattr(a, "recent_actions", None)
                if ra is not None:
                    ra.append(act)
            except Exception:
                pass
