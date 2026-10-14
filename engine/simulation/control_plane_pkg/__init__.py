"""simulation/control_plane_pkg

Phase 2: Control Plane Core + WAL.

This package is additive and should not affect Phase 1 semantics unless explicitly wired
at SafeBoundary step (6).
"""

from simulation.control_plane_pkg.control_plane import ControlPlane, Ack, AppliedEvent
from simulation.control_plane_pkg.commands import (
    CommandBase,
    PressureUpdateCommand,
    CheckpointRequestCommand,
    PlanAmendCommand,
    PlanCancelCommand,
    TraitInjectionCommand,
)

__all__ = [
    "ControlPlane",
    "Ack",
    "AppliedEvent",
    "CommandBase",
    "PressureUpdateCommand",
    "CheckpointRequestCommand",
    "PlanAmendCommand",
    "PlanCancelCommand",
    "TraitInjectionCommand",
]

from .rng_guard import DeterminismViolationError
