from __future__ import annotations

"""simulation/trait_injection_pkg/cost_model.py

Phase 5B: deterministic cost model for trait injections.

Requirement
-----------
- Cost must depend on BOTH duration and apply_phase.

This cost model is intentionally simple and deterministic.
It is used for budgeting / planning and for audit visibility.

No RNG. No wall clock.
"""

from typing import Dict

from simulation.trait_injection_pkg.schema_v2 import (
    APPLY_PHASE_POST_REPRO_PRE_SELECTION,
    APPLY_PHASE_POST_SELECTION_PRE_MUTATION,
    ALLOWED_APPLY_PHASES,
)


def trait_injection_cost(duration_generations: int, apply_phase: str) -> int:
    """Compute deterministic intervention cost.

    Parameters
    ----------
    duration_generations:
        Number of generations the injection is active for (>= 1).
    apply_phase:
        One of the allowed apply phases.

    Returns
    -------
    int
        Non-negative cost.
    """
    dur = int(duration_generations)
    if dur < 0:
        dur = 0

    ap = str(apply_phase or "")
    if ap not in ALLOWED_APPLY_PHASES:
        # Fail closed: unknown phase has cost 0 (and should be rejected at validation time).
        return 0

    # Phase multiplier: earlier application is treated as more expensive.
    phase_mult = 1
    if ap == APPLY_PHASE_POST_REPRO_PRE_SELECTION:
        phase_mult = 2
    elif ap == APPLY_PHASE_POST_SELECTION_PRE_MUTATION:
        phase_mult = 1

    return int(dur * phase_mult)


def trait_injection_cost_model(duration_generations: int, apply_phase: str) -> Dict[str, int | str]:
    """Return a structured cost model object for embedding into payloads."""
    return {
        "duration_generations": int(duration_generations),
        "apply_phase": str(apply_phase or ""),
        "cost": int(trait_injection_cost(duration_generations, apply_phase)),
    }
