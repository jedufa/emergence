from __future__ import annotations

"""agents/maintenance.py

Deterministic maintenance process state helpers (Phase A2).

State schema (fixed 4-slot float32 vector):
0. pending_strength      - queued repair intensity to be applied after delay
1. due_tick              - absolute tick index when queued effect becomes eligible
2. recent_load           - decays slowly; used for diminishing returns
3. last_schedule_tick    - absolute tick index of latest schedule event (or -1)
"""

from typing import Iterable

import numpy as np

MAINT_SLOT_PENDING_STRENGTH = 0
MAINT_SLOT_DUE_TICK = 1
MAINT_SLOT_RECENT_LOAD = 2
MAINT_SLOT_LAST_SCHEDULE_TICK = 3

MAINTENANCE_SIZE = 4
MAINTENANCE_SLOT_NAMES = (
    "pending_strength",
    "due_tick",
    "recent_load",
    "last_schedule_tick",
)


def clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


def zeros_maintenance_state() -> np.ndarray:
    return np.array([0.0, -1.0, 0.0, -1.0], dtype=np.float32)


def maintenance_from_values(values: Iterable[float]) -> np.ndarray:
    v = np.asarray(list(values), dtype=np.float32).reshape(-1)
    out = zeros_maintenance_state()
    if v.size <= 0:
        return out
    n = min(v.size, MAINTENANCE_SIZE)
    out[:n] = v[:n]
    # clamp continuous slots; tick slots keep exact numeric value
    out[MAINT_SLOT_PENDING_STRENGTH] = np.float32(clamp01(float(out[MAINT_SLOT_PENDING_STRENGTH])))
    out[MAINT_SLOT_RECENT_LOAD] = np.float32(clamp01(float(out[MAINT_SLOT_RECENT_LOAD])))
    return out
