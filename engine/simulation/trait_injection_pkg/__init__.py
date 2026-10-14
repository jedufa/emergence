"""simulation/trait_injection_pkg

Phase 5B: Trait Injection Framework.

This package provides deterministic, ControlPlane-driven trait injection.

Key invariants
--------------
- No RNG usage (no global seeding, no stream mixing).
- Injection effects occur ONLY at SafeBoundary-approved apply phases.
- Idempotency is keyed by ControlPlane command_uuid.
- Engine physics remain untouched: selection and reproduction timing are not changed.

Public API
----------
- apply_trait_injections: apply active injections for a (generation, apply_phase).
"""

from simulation.trait_injection_pkg.injection_manager import apply_trait_injections

__all__ = [
    "apply_trait_injections",
]
