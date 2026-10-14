from __future__ import annotations

from dataclasses import dataclass

@dataclass
class StepResult:
    died: bool
    energy_delta: float
    food_gained: float
    damage_taken: float
