from __future__ import annotations

from typing import Optional

class MultipliersMixin:
    def set_multipliers(
        self,
        *,
        food_spawn_mult: Optional[float] = None,
        action_cost_mult: Optional[float] = None,
        rest_cost_mult: Optional[float] = None,
    ):
        if food_spawn_mult is not None:
            self.food_spawn_mult = float(max(0.0, food_spawn_mult))
        if action_cost_mult is not None:
            self.action_cost_mult = float(max(0.0, action_cost_mult))
        if rest_cost_mult is not None:
            self.rest_cost_mult = float(max(0.0, rest_cost_mult))

    def reset_multipliers(self):
        self.food_spawn_mult = 1.0
        self.action_cost_mult = 1.0
        self.rest_cost_mult = 1.0
