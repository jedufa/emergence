from __future__ import annotations

"""
simulation/runner_pkg/runner_ticks_common.py

Shared helpers/constants for the per-generation tick loop modules.

Why this exists:
- Keeps small, stable utilities and constants in one place
- Avoids circular imports between the tick mixins

Notes:
- "tick" is the simulation's smallest time-advance unit (formerly called "step").
- We keep the logic identical to the old implementation; this refactor is structural.
"""

from env.gridworld_pkg.constants import ACT_ATTACK, ACT_REST, ACT_GATHER

# Drives live on Agent.drives: [hunger, pain, fatigue, fear]
DRIVE_HUNGER = 0
DRIVE_PAIN = 1
DRIVE_FATIGUE = 2
DRIVE_FEAR = 3
N_DRIVES = 4


def clamp01(x: float) -> float:
    """
    Clamp a number into [0, 1].

    Args:
        x: Input value.

    Returns:
        The clamped value.
    """
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x
