"""Optional native-style hot loop probes.

The functions here are default-off diagnostics. Normal simulation execution
must not depend on optional native/JIT tooling being present.
"""

from engine.perf.native.cognitive_cost_kernel import (
    NUMBA_AVAILABLE,
    add_arrays,
    backend_name,
    cognitive_costs_native,
    cognitive_costs_numpy,
    cognitive_costs_python,
)

__all__ = [
    "NUMBA_AVAILABLE",
    "add_arrays",
    "backend_name",
    "cognitive_costs_native",
    "cognitive_costs_numpy",
    "cognitive_costs_python",
]
