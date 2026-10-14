"""simulation/metrics_pkg/stress_scenarios.py

Phase 5a: Stress scenario schema and built-in defaults.

Authoritative requirement: schema_version must be exactly "stress_scenario_v1".

These scenarios are used by the EPI (Experimental Pressure Index) engine.
The schema is intentionally small and JSON-friendly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


STRESS_SCENARIO_SCHEMA_VERSION = "stress_scenario_v1"


@dataclass
class StressScenarioV1:
    """Schema: stress_scenario_v1."""

    scenario_id: str
    pressure_overrides: Dict[str, float] = field(default_factory=dict)
    duration_steps: int = 50
    seed_offset: int = 0
    evaluation_mode: str = "ticks"
    schema_version: str = STRESS_SCENARIO_SCHEMA_VERSION

    def to_dict(self) -> Dict:
        return {
            "schema_version": self.schema_version,
            "scenario_id": self.scenario_id,
            "pressure_overrides": dict(self.pressure_overrides),
            "duration_steps": int(self.duration_steps),
            "seed_offset": int(self.seed_offset),
            "evaluation_mode": str(self.evaluation_mode),
        }


def default_stress_scenarios_v1() -> List[StressScenarioV1]:
    """Built-in default stress scenarios.

    These are conservative and focus on existing runtime multipliers that are
    already supported by GridWorld.
    """

    return [
        StressScenarioV1(
            scenario_id="baseline_env",
            pressure_overrides={},
            duration_steps=50,
            seed_offset=0,
        ),
        StressScenarioV1(
            scenario_id="high_scarcity",
            pressure_overrides={"food_spawn_mult": 0.25},
            duration_steps=50,
            seed_offset=1,
        ),
        StressScenarioV1(
            scenario_id="hazard_spike",
            # Approximated via higher per-step costs.
            pressure_overrides={"action_cost_mult": 1.50},
            duration_steps=50,
            seed_offset=2,
        ),
        StressScenarioV1(
            scenario_id="resource_shift",
            pressure_overrides={
                "food_spawn_mult": 0.75,
                "rest_cost_mult": 1.25,
            },
            duration_steps=50,
            seed_offset=3,
        ),
    ]
