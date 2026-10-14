from __future__ import annotations

"""agents/chemistry.py

Deterministic chemistry schema and helpers (Phase 1A).
"""

from typing import Iterable

import numpy as np


CHEMISTRY_SCHEMA_VERSION = "eiel_chemistry_slots_v1"


CHEM_SLOT_ENERGY_STORE = 0
CHEM_SLOT_REPAIR_RESERVE = 1
CHEM_SLOT_ACUTE_DAMAGE = 2
CHEM_SLOT_CHRONIC_DAMAGE = 3
CHEM_SLOT_REGULATOR_1 = 4
CHEM_SLOT_REGULATOR_2 = 5
CHEMISTRY_SLOT_NAMES: tuple[str, ...] = (
    "energy_store",
    "repair_reserve",
    "acute_damage",
    "chronic_damage",
    "regulator_1",
    "regulator_2",
)
CHEMISTRY_SIZE = len(CHEMISTRY_SLOT_NAMES)


def clamp01(x: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    return float(x)


def zeros_chemistry() -> np.ndarray:
    return np.zeros((CHEMISTRY_SIZE,), dtype=np.float32)


def chemistry_from_values(values: Iterable[float]) -> np.ndarray:
    if isinstance(values, np.ndarray):
        arr = np.asarray(values, dtype=np.float32).reshape(-1).copy()
    else:
        arr = np.asarray(list(values), dtype=np.float32).reshape(-1)
    if int(arr.shape[0]) != CHEMISTRY_SIZE:
        out = np.zeros((CHEMISTRY_SIZE,), dtype=np.float32)
        n = min(int(arr.shape[0]), CHEMISTRY_SIZE)
        if n > 0:
            out[:n] = arr[:n]
        arr = out
    for i in range(CHEMISTRY_SIZE):
        arr[i] = np.float32(clamp01(float(arr[i])))
    return arr


def init_chemistry_for_agent(agent: object) -> np.ndarray:
    """Deterministic chemistry initialization from physiological state."""
    e = float(getattr(agent, "energy", 0.0))
    emax = float(getattr(agent, "max_energy_norm", 1.0)) or 1.0
    h = float(getattr(agent, "health", 0.0))
    hmax = float(getattr(agent, "start_health_norm", 1.0)) or 1.0
    injury = float(getattr(agent, "injury", 0.0))

    e_n = clamp01(e / emax)
    h_n = clamp01(h / hmax)
    inj_n = clamp01(injury / hmax)

    return chemistry_from_values(
        (
            0.40 + 0.40 * e_n,
            0.35 + 0.20 * h_n,
            0.10 * inj_n,
            0.05 * inj_n,
            0.50,
            0.25,
        )
    )
