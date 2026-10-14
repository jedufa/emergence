from __future__ import annotations

from env.gridworld_pkg.constants import EMPTY, FOOD
from env.gridworld_pkg.types import StepResult
from env.gridworld_pkg.multipliers import MultipliersMixin
from env.gridworld_pkg.state import StateMixin
from env.gridworld_pkg.food import FoodMixin
from env.gridworld_pkg.observations import ObservationsMixin
from env.gridworld_pkg.energy import EnergyMixin
from env.gridworld_pkg.stepping import SteppingMixin
from env.gridworld_pkg.adjacency import AdjacencyMixin

class GridWorld(
    StateMixin,
    MultipliersMixin,
    FoodMixin,
    ObservationsMixin,
    EnergyMixin,
    SteppingMixin,
    AdjacencyMixin,
):
    """Split into mixins to keep files small. Public API stays the same."""
    pass
