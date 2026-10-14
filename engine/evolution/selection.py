from __future__ import annotations

from typing import List, Dict
import math


def fitness(agent) -> float:
    """
    Fitness v2: long-horizon, efficiency-aware.

    Core ideas:
    - Reproduction still wins, but with diminishing returns
    - Survival matters, but caps out
    - Energy efficiency > raw energy
    - Violence is only bad if it doesn't pay
    """

    s = agent.stats

    survival = float(s.survival_steps)         # ~0–150
    energy = float(s.energy_collected)         # ~0–150+
    offspring = float(s.offspring_count)       # usually 0–3
    attacks = float(s.attacks)                 # ~0–10

    # --- survival: valuable, but capped ---
    survival_score = min(survival, 120.0) * 0.05

    # --- reproduction: diminishing returns ---
    offspring_score = 0.0
    if offspring > 0:
        offspring_score += 180.0               # first child
    if offspring > 1:
        offspring_score += 120.0               # second
    if offspring > 2:
        offspring_score += 60.0 * (offspring - 2.0)

    # --- energy efficiency ---
    # reward energy *per step*, not just hoarding
    energy_eff = energy / max(1.0, survival)
    energy_score = energy_eff * 80.0

    # --- violence penalty (soft) ---
    attack_penalty = attacks * 2.5

    score = survival_score + offspring_score + energy_score - attack_penalty

    # -----------------------------
    # Early-life weighted component
    # -----------------------------
    # Selection pressure for "start good immediately" so inherited priors matter.
    # Computed from counters that only accumulate during the first K life steps.
    #
    # If K==0 (default), early_* counters stay 0 and this does nothing.
    # Weight comes from SimConfig, but we store it on stats (runner-side) so selection
    # stays decoupled from the runner/population wiring.
    w = float(getattr(s, "early_life_weight", 1.0))
    if not math.isfinite(w):
        w = 1.0

    early_surv = float(getattr(s, "early_survival_steps", 0.0))
    early_energy = float(getattr(s, "early_energy_collected", 0.0))
    early_inj = float(getattr(s, "early_injury_gained", 0.0))

    # Same "energy per step" idea, but restricted to early window.
    early_energy_eff = early_energy / max(1.0, early_surv)

    # Early component scale is intentionally modest so it nudges selection rather than hijacking it.
    early_component = (0.10 * early_surv) + (80.0 * early_energy_eff) - (10.0 * early_inj)

    score = score + w * early_component

    # small continuity bonus
    if getattr(agent, "alive", False):
        score += 10.0

    return float(score)


def _rank_with_cached_fitness(agents: List) -> List:
    if not agents:
        return []

    # Compute fitness ONCE per agent (sorting with a Python key recomputes constantly otherwise).
    scores: Dict[int, float] = {}
    for a in agents:
        scores[id(a)] = fitness(a)

    return sorted(agents, key=lambda a: scores[id(a)], reverse=True)


def select_top(agents: List, fraction: float) -> List:
    if not agents:
        return []
    ranked = _rank_with_cached_fitness(agents)
    k = max(1, int(len(ranked) * float(fraction)))
    return ranked[:k]


def rank_agents(agents: List) -> List:
    return _rank_with_cached_fitness(agents)
