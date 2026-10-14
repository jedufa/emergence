"""simulation/autopilot_pkg/__init__.py

Autopilot package.

This package contains optional, non-core "control" logic that can steer a run
by scheduling deterministic ControlPlane commands.

Important design constraints
----------------------------
- No RNG usage (no new RNG islands).
- Decisions must be reproducible from simulation state + logged metrics.
- All world changes must flow through ControlPlane commands so they are audited.
"""

from simulation.autopilot_pkg.pressure_autopilot import PressureAutopilot

__all__ = ["PressureAutopilot"]
