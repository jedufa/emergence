"""simulation/runner.py

Stable public entry point shim for SimulationRunner.

Keep this file logic-free; implementations live in simulation/runner_pkg/*.
"""

from __future__ import annotations

from simulation.runner_pkg.runner_base import RunnerBase
from simulation.runner_pkg.runner_bank import BankMixin
from simulation.runner_pkg.runner_policy import PolicyMixin
from simulation.runner_pkg.runner_rest import RestMixin
from simulation.runner_pkg.runner_envsync import EnvSyncMixin
from simulation.runner_pkg.runner_shock import ShockMixin
from simulation.runner_pkg.runner_generation import GenerationMixin
from simulation.runner_pkg.runner_steps import StepLoopMixin
from simulation.runner_pkg.runner_selection import SelectionMixin
from simulation.runner_pkg.runner_loop import LoopMixin

class SimulationRunner(
    RunnerBase,
    BankMixin,
    PolicyMixin,
    RestMixin,
    EnvSyncMixin,
    ShockMixin,
    GenerationMixin,
    StepLoopMixin,
    SelectionMixin,
    LoopMixin,
):
    """Composed runner: functionality split into mixins for sane file sizes."""
    pass
