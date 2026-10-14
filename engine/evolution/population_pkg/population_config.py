from __future__ import annotations

"""evolution/population_pkg/population_config.py

Population-level configuration.

These parameters control the outer evolutionary loop:
- target population size
- parent selection fraction
- cull fraction (turnover)
- slot pool sizing (for parameter-bank execution)

The slot pool exists because the in-world reproduction system can temporarily
create more live agents than the target size within a generation. Slots provide
dense indices for banked policies even as IDs churn.
"""

from dataclasses import dataclass


@dataclass
class PopConfig:
    population_size: int = 128
    parent_fraction: float = 0.25
    cull_fraction: float = 0.50

    # In-world reproduction can temporarily exceed population_size.
    # slot_factor=6 means max_slots = population_size * 6.
    slot_factor: int = 6

    # Optional hard cap for max_slots (0 = no cap).
    max_slots_cap: int = 0
