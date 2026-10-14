from __future__ import annotations

"""
simulation/runner_pkg/runner_ticks_recovery.py

Rest charge, recovery, and related tick-level bookkeeping.

Responsibilities:
- Maintain rest streak counters
- Apply optional "rest charge" energy drain beyond a free streak length
- Convert negative energy to health loss (if configured)
- Apply injury recovery when resting (if injury is enabled)
"""

import time

from perf.cxxrewrite4_upstream_resident import (
    resident_scalar_sync_fail_closed as cxxrewrite4_resident_scalar_sync_fail_closed,
    sync_agents_scalar_state as cxxrewrite4_sync_agents_scalar_state,
)
from .runner_ticks_common import ACT_REST


class TickRecoveryMixin:
    def _apply_rest_and_recovery(self, alive, actions_np, rest_streaks, injury_recover_mult: float, deaths_this_gen: int, buckets=None):
        """
        Apply rest streak tracking and optional injury recovery.

        Args:
            alive: List of alive agents used this tick.
            actions_np: Numpy array of actions aligned to `alive`.
            rest_streaks: Dict[agent_id -> current rest streak].
            injury_recover_mult: Shock-scaled recovery multiplier.
            deaths_this_gen: current death count for this generation.
            buckets: Optional perf timing buckets dict.

        Returns:
            Updated deaths_this_gen.
        """
        t_rest0 = time.perf_counter()

        if not getattr(self.env_cfg, "enable_rest_charge", False):
            return deaths_this_gen

        free = int(getattr(self.sim_cfg, "rest_streak_free", 0))
        base = float(getattr(self.sim_cfg, "rest_charge_energy", 0.0))
        grow = float(getattr(self.sim_cfg, "rest_charge_growth", 0.0))

        deficit_to_health = float(getattr(self.sim_cfg, "rest_deficit_to_health", 1.0))
        mutated_agents = []

        for i, agent in enumerate(alive):
            if not agent.alive:
                continue
            a_id = int(agent.agent_id)

            if int(actions_np[i]) == ACT_REST:
                rest_streaks[a_id] = rest_streaks.get(a_id, 0) + 1
            else:
                rest_streaks[a_id] = 0

            agent.rest_streak = rest_streaks[a_id]

            if rest_streaks[a_id] > free:
                over = rest_streaks[a_id] - free
                charge = base + grow * float(over - 1)
                agent.energy -= charge
                mutated_agents.append(agent)
                agent.stats.idle_charge_paid = float(getattr(agent.stats, "idle_charge_paid", 0.0)) + float(charge)

                if agent.energy < 0.0:
                    agent.health -= float(-agent.energy) * deficit_to_health
                    agent.energy = 0.0
                    mutated_agents.append(agent)
                    if agent.health <= 0.0:
                        agent.alive = False
                        deaths_this_gen += 1
                        self.env.remove_agent(int(agent.agent_id))
                        mutated_agents.append(agent)

        if getattr(self.env_cfg, "enable_injury", False):
            recover = float(
                getattr(self.env_cfg, "injury_recover_per_rest_step", getattr(self.env_cfg, "injury_recover_per_rest", 0.0))
            ) * float(injury_recover_mult)
            if recover > 0.0:
                for agent in alive:
                    if not agent.alive:
                        continue
                    if int(getattr(agent, "rest_streak", 0)) > 0:
                        inj = float(getattr(agent, "injury", 0.0))
                        if inj > 0.0:
                            agent.injury = max(0.0, inj - recover)
                            mutated_agents.append(agent)

        self._idle_charge_total_gen = float(getattr(self, "_idle_charge_total_gen", 0.0)) + float(
            sum(float(getattr(a.stats, "idle_charge_paid", 0.0)) for a in alive)
        )
        self._idle_charge_deaths_gen = int(getattr(self, "_idle_charge_deaths_gen", 0))

        if isinstance(buckets, dict):
            buckets["rest_apply"] = buckets.get("rest_apply", 0.0) + (time.perf_counter() - t_rest0)

        try:
            cxxrewrite4_sync_agents_scalar_state(self, mutated_agents, stage="rest_recovery")
        except Exception:
            if cxxrewrite4_resident_scalar_sync_fail_closed(self.sim_cfg):
                raise

        return deaths_this_gen
