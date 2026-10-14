from __future__ import annotations

"""
simulation/runner_pkg/rest.py

Rest mechanics:
- Anti-idling rest charge
- Injury and scar recovery (gated by rest streak)

Note:
Internal drives are updated in steps.py after env stepping. Rest still affects
physiology (energy/health/injury/scar), which then shapes drives next step.
"""

from typing import Tuple


class RestMixin:
    def _apply_rest_charge(self, agent, rest_streak: int) -> Tuple[float, bool]:
        """
        Apply "rest charge" (anti-idling mechanic):
        - After rest_streak_free consecutive REST steps, we charge extra energy *cost*.
        - If energy goes below 0, convert deficit to health damage (rest_deficit_to_health * deficit).
        Returns: (charge_applied, died)
        """
        if not getattr(self.sim_cfg, "enable_rest_charge", False):
            return 0.0, False

        free = int(getattr(self.sim_cfg, "rest_streak_free", 0))
        if rest_streak <= free:
            return 0.0, False

        over = rest_streak - free  # how many steps beyond free limit

        base = float(getattr(self.sim_cfg, "rest_charge_energy", 0.0))
        growth = float(getattr(self.sim_cfg, "rest_charge_growth", 0.0))

        charge = base + growth * float(over)

        agent.energy -= charge
        died = False

        if agent.energy < 0.0:
            deficit = -agent.energy
            agent.energy = 0.0

            mult = float(getattr(self.sim_cfg, "rest_deficit_to_health", 0.0))
            agent.health -= mult * deficit

        if agent.health <= 0.0:
            died = True

        return float(charge), died

    def _maybe_apply_injury_recovery(self, agent, rest_streak: int, injury_recover_mult: float) -> None:
        """
        Injury recovery:
        - Only if env_cfg.enable_injury and agent has 'injury'
        - Only after rest streak reaches env_cfg.injury_recover_requires_streak
        - Recovery amount = injury_recover_per_rest_step * max(0, injury_recover_mult)

        Scar recovery (chronic injury):
        - Very slow
        - Requires longer rest streak (scar_recover_requires_streak)
        """
        if not getattr(self.env_cfg, "enable_injury", False):
            return
        if not hasattr(agent, "injury"):
            return

        req = int(getattr(self.env_cfg, "injury_recover_requires_streak", 0))
        if rest_streak >= req:
            per = float(getattr(self.env_cfg, "injury_recover_per_rest_step", 0.0))
            if per > 0.0:
                mult = float(max(0.0, injury_recover_mult))
                dec = per * mult
                if dec > 0.0:
                    agent_injury = float(getattr(agent, "injury", 0.0))
                    agent.injury = float(max(0.0, agent_injury - dec))

        if not getattr(self.env_cfg, "enable_scar", False):
            return
        if not hasattr(agent, "scar"):
            return

        scar_req = int(getattr(self.env_cfg, "scar_recover_requires_streak", 0))
        if rest_streak < scar_req:
            return

        scar_per = float(getattr(self.env_cfg, "scar_recover_per_rest_step", 0.0))
        if scar_per <= 0.0:
            return

        mult = float(max(0.0, injury_recover_mult))
        scar_dec = scar_per * mult
        if scar_dec <= 0.0:
            return

        s = float(getattr(agent, "scar", 0.0))
        agent.scar = float(max(0.0, s - scar_dec))
