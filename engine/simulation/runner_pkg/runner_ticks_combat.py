from __future__ import annotations

"""
simulation/runner_pkg/runner_ticks_combat.py

Combat bookkeeping for the tick loop.

Note:
The environment resolves effects; runner-level bookkeeping counts attempted attacks.
"""

from .runner_ticks_common import ACT_ATTACK


class TickCombatMixin:
    def _handle_attacks(self, alive, actions_np, deaths_this_gen: int, gen_peak_injury: float):
        """
        Count attack actions for behavior/metrics.

        Returns:
            (deaths_this_gen, gen_peak_injury) unchanged except for counters.
        """
        if not getattr(self.env_cfg, "enable_combat", False):
            return deaths_this_gen, gen_peak_injury

        for i, agent in enumerate(alive):
            if not agent.alive:
                continue
            if int(actions_np[i]) == ACT_ATTACK:
                agent.stats.attacks += 1

        return deaths_this_gen, gen_peak_injury
