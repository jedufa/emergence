from __future__ import annotations

"""agents/chemistry_coupling.py

Deterministic, bounded chemistry -> competence coupling helpers (Phase A3).
"""

from typing import Mapping

import numpy as np

from agents.chemistry import (
    CHEM_SLOT_ACUTE_DAMAGE,
    CHEM_SLOT_CHRONIC_DAMAGE,
    CHEM_SLOT_REGULATOR_1,
    CHEM_SLOT_REGULATOR_2,
    CHEM_SLOT_REPAIR_RESERVE,
    chemistry_from_values,
)


def clamp01(x: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    return float(x)


def clamp_range(x: float, lo: float, hi: float) -> float:
    if x < lo:
        return float(lo)
    if x > hi:
        return float(hi)
    return float(x)


def _chemistry_damage_index_from_row(c: np.ndarray) -> float:
    acute = float(c[CHEM_SLOT_ACUTE_DAMAGE])
    chronic = float(c[CHEM_SLOT_CHRONIC_DAMAGE])
    repair = float(c[CHEM_SLOT_REPAIR_RESERVE])
    reg1 = float(c[CHEM_SLOT_REGULATOR_1])
    raw = 0.60 * acute + 0.40 * chronic - 0.25 * repair - 0.15 * reg1
    return clamp01(raw)


def chemistry_damage_index(chem: np.ndarray | list[float] | tuple[float, ...] | None) -> float:
    c = chemistry_from_values((0.0,) * 6) if chem is None else chemistry_from_values(chem)
    return _chemistry_damage_index_from_row(c)


def action_cost_multiplier_from_chem(
    chem: np.ndarray | list[float] | tuple[float, ...] | None,
    *,
    min_mult: float,
    max_mult: float,
    strength: float,
    power: float,
) -> float:
    d = chemistry_damage_index(chem)
    p = max(0.1, float(power))
    span = max(0.0, float(max_mult) - float(min_mult))
    s = clamp01(float(strength))
    m = float(min_mult) + span * (s * (d**p))
    return clamp_range(m, float(min_mult), float(max_mult))


def observation_fidelity_and_bias_from_chem(
    chem: np.ndarray | list[float] | tuple[float, ...] | None,
    *,
    fidelity_min: float,
    fidelity_max: float,
    noise_abs_max: float,
    fidelity_strength: float,
    noise_strength: float,
) -> tuple[float, float]:
    c = chemistry_from_values((0.0,) * 6) if chem is None else chemistry_from_values(chem)
    d = _chemistry_damage_index_from_row(c)
    reg2 = float(c[CHEM_SLOT_REGULATOR_2])

    f_span = max(0.0, float(fidelity_max) - float(fidelity_min))
    f = float(fidelity_min) + f_span * (1.0 - clamp01(float(fidelity_strength)) * d)
    fidelity = clamp_range(f, float(fidelity_min), float(fidelity_max))

    # Deterministic pseudo-noise term (no RNG): signed by regulator_2 around 0.5.
    signed = (reg2 - 0.5) * 2.0  # [-1,1]
    bias = float(noise_abs_max) * clamp01(float(noise_strength)) * d * signed
    bias = clamp_range(bias, -abs(float(noise_abs_max)), abs(float(noise_abs_max)))
    return float(fidelity), float(bias)


def reproduction_viability_scale_from_chem(
    chem: np.ndarray | list[float] | tuple[float, ...] | None,
    *,
    floor: float,
    strength: float,
) -> float:
    c = chemistry_from_values((0.0,) * 6) if chem is None else chemistry_from_values(chem)
    return reproduction_viability_scale_from_chem_row(c, floor=floor, strength=strength)


def reproduction_viability_scale_from_chem_row(
    chem: np.ndarray,
    *,
    floor: float,
    strength: float,
) -> float:
    d = _chemistry_damage_index_from_row(chem)
    floor_v = clamp_range(float(floor), 0.0, 1.0)
    s = clamp01(float(strength))
    v = 1.0 - s * d
    return clamp_range(v, floor_v, 1.0)


def classify_recovery_taxonomy(
    *,
    fitness_ratio: float,
    drift_ratio: float,
    return_fitness_ratio: float,
    return_drift_ratio_max: float,
    shift_fitness_ratio_min: float,
) -> str:
    fr = float(fitness_ratio)
    dr = float(drift_ratio)
    if fr >= float(return_fitness_ratio) and dr <= float(return_drift_ratio_max):
        return "return"
    if fr >= float(shift_fitness_ratio_min):
        return "shift"
    return "collapse"


def taxonomy_record(
    *,
    generation: int,
    perturbation_active: bool,
    classification: str,
    fitness_ratio: float,
    drift_ratio: float,
    baseline_fitness_ema: float,
    baseline_drift_ema: float,
    mean_effective_fitness: float,
    mean_identity_drift: float,
    thresholds: Mapping[str, float],
) -> dict[str, object]:
    return {
        "schema_version": "a3_recovery_taxonomy_v1",
        "generation": int(generation),
        "perturbation_active": bool(perturbation_active),
        "classification": str(classification),
        "fitness_ratio": float(fitness_ratio),
        "drift_ratio": float(drift_ratio),
        "baseline_fitness_ema": float(baseline_fitness_ema),
        "baseline_drift_ema": float(baseline_drift_ema),
        "mean_effective_fitness": float(mean_effective_fitness),
        "mean_identity_drift": float(mean_identity_drift),
        "thresholds": {
            "return_fitness_ratio": float(thresholds.get("return_fitness_ratio", 0.0)),
            "return_drift_ratio_max": float(thresholds.get("return_drift_ratio_max", 0.0)),
            "shift_fitness_ratio_min": float(thresholds.get("shift_fitness_ratio_min", 0.0)),
        },
    }
