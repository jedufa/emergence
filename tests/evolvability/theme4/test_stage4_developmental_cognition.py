from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from agents.cognition import (
    MEMORY_LADDER,
    compute_cognitive_budget_ledger,
    developmental_expression_state,
    express_cognitive_phenotype,
    get_cognitive_trait,
)
from agents.developmental_program_v1 import canonicalize_developmental_program
from agents.genome import Genome
from agents.perception import implemented_sensor_channel_mask
from env.env_config import EnvConfig
from simulation.metrics_pkg.cognitive_status import compute_cognitive_status_metrics
from simulation.runner_pkg.runner_config import SimConfig


def _genome(*, memory_dim: int = 64, pred_lr: float = 0.4, sensor_radius: int = 3) -> Genome:
    return Genome(
        hidden_size=64,
        exploration_bias=0.1,
        aggression_bias=0.2,
        reproduction_threshold=30.0,
        mutation_rate=0.03,
        action_temperature=1.0,
        use_gru=bool(memory_dim > 0),
        memory_dim=int(memory_dim),
        sensor_channel_mask=implemented_sensor_channel_mask(),
        sensor_radius=int(sensor_radius),
        drive_inertia=0.85,
        drive_noise=0.02,
        drive_logit_gain=1.1,
        pred_lr=float(pred_lr),
        developmental_program=canonicalize_developmental_program(
            {
                "stages": [
                    {"stage_id": "juvenile", "start_step": 0, "end_step": 19},
                    {"stage_id": "adult", "start_step": 20, "end_step": -1},
                ]
            }
        ),
    )


def _agent(agent_id: int, genome: Genome, *, age: int, offspring: int = 0) -> SimpleNamespace:
    return SimpleNamespace(
        agent_id=int(agent_id),
        genome=genome,
        energy=100.0,
        health=10.0,
        alive=True,
        drives=np.zeros((4,), dtype=np.float32),
        stats=SimpleNamespace(
            survival_steps=int(age),
            pred_err_count=0,
            pred_err_sum=0.0,
            offspring_count=int(offspring),
            patch_habitat_lineage_id=int(agent_id),
        ),
    )


def _dev_cfg(*, mode: str = "step_juvenile_adult") -> SimConfig:
    sim_cfg = SimConfig(seed=1)
    sim_cfg.enable_evolvable_sensor_channel_mask = True
    sim_cfg.enable_evolvable_sensor_radius = True
    sim_cfg.enable_developmental_cognition = True
    sim_cfg.developmental_cognition_mode = str(mode)
    sim_cfg.developmental_maturity_age = 20
    sim_cfg.juvenile_cognition_scale = 0.5
    sim_cfg.adult_cognition_scale = 1.0
    sim_cfg.developmental_cognition_cost_per_unexpressed_capacity = 0.01
    sim_cfg.maturation_cost_per_tick = 0.2
    return sim_cfg


@dataclass(frozen=True)
class _DelayedMaturityFixture:
    maturity_age: int = 20

    def metadata(self) -> dict[str, object]:
        return {
            "developmental_pressure_scenario": True,
            "developmental_pressure_type": "delayed_maturity",
            "direct_development_reward": False,
        }

    def phenotypes(self, *, ablated: bool) -> tuple[dict[str, object], dict[str, object]]:
        env_cfg = EnvConfig(width=8, height=8, view_radius=3)
        sim_cfg = _dev_cfg()
        if ablated:
            sim_cfg.force_legacy_full_cognition = True
        g = _genome()
        juvenile = express_cognitive_phenotype(g, env_cfg=env_cfg, sim_cfg=sim_cfg, age_steps=5)
        adult = express_cognitive_phenotype(g, env_cfg=env_cfg, sim_cfg=sim_cfg, age_steps=25)
        return juvenile.to_dict(), adult.to_dict()


def test_stage4_developmental_traits_registered_honestly() -> None:
    for trait_id in (
        "developmental_cognition_enabled",
        "developmental_stage",
        "maturity_age",
        "juvenile_cognition_scale",
        "adult_cognition_scale",
        "cognitive_growth_rate",
        "cognitive_growth_curve",
        "age_gated_memory_expression",
        "age_gated_perception_expression",
        "age_gated_prediction_expression",
        "developmental_cognition_cost",
        "maturation_cost",
        "developmental_ablation_state",
        "developmental_expression_active",
        "developmental_pressure_test_ready",
    ):
        assert get_cognitive_trait(trait_id).category in {"development", "cognition_budget"}

    assert get_cognitive_trait("maturity_age").status == "fixed"
    assert get_cognitive_trait("juvenile_cognition_scale").directly_mutable is False
    assert get_cognitive_trait("developmental_pressure_test_ready").status == "partially_active"


def test_stage4_default_preserves_legacy_full_expression() -> None:
    env_cfg = EnvConfig(width=8, height=8, view_radius=3)
    sim_cfg = SimConfig(seed=2)
    g = _genome()
    juvenile = express_cognitive_phenotype(g, env_cfg=env_cfg, sim_cfg=sim_cfg, age_steps=0)
    adult = express_cognitive_phenotype(g, env_cfg=env_cfg, sim_cfg=sim_cfg, age_steps=100)

    assert juvenile.developmental_cognition_enabled is False
    assert juvenile.developmental_expression_factor == 1.0
    assert juvenile.effective_memory_dim == adult.effective_memory_dim == 64
    assert juvenile.active_sensor_radius == adult.active_sensor_radius == 3
    assert round(juvenile.effective_prediction_learning_rate, 6) == round(adult.effective_prediction_learning_rate, 6) == 0.4


def test_stage4_step_and_linear_expression_are_deterministic_by_age() -> None:
    env_cfg = EnvConfig(width=8, height=8, view_radius=3)
    g = _genome()

    step_cfg = _dev_cfg(mode="step_juvenile_adult")
    juvenile = express_cognitive_phenotype(g, env_cfg=env_cfg, sim_cfg=step_cfg, age_steps=5)
    adult = express_cognitive_phenotype(g, env_cfg=env_cfg, sim_cfg=step_cfg, age_steps=25)
    assert juvenile.developmental_expression_factor == 0.5
    assert adult.developmental_expression_factor == 1.0
    assert juvenile.effective_memory_dim < adult.effective_memory_dim
    assert juvenile.active_sensor_radius < adult.active_sensor_radius
    assert juvenile.active_sensor_channel_count < adult.active_sensor_channel_count
    assert juvenile.effective_prediction_learning_rate < adult.effective_prediction_learning_rate

    linear_cfg = _dev_cfg(mode="linear_growth_to_maturity")
    assert developmental_expression_state(age_steps=10, sim_cfg=linear_cfg)["factor"] == 0.75
    assert developmental_expression_state(age_steps=20, sim_cfg=linear_cfg)["factor"] == 1.0


def test_stage4_ablation_modes_override_without_mutating_genome() -> None:
    env_cfg = EnvConfig(width=8, height=8, view_radius=3)
    sim_cfg = _dev_cfg()
    g = _genome()

    sim_cfg.force_adult_cognition = True
    adult_from_birth = express_cognitive_phenotype(g, env_cfg=env_cfg, sim_cfg=sim_cfg, age_steps=0)
    assert adult_from_birth.developmental_expression_factor == 1.0
    assert adult_from_birth.force_adult_cognition_active is True

    sim_cfg.force_adult_cognition = False
    sim_cfg.force_juvenile_cognition = True
    forced_juvenile = express_cognitive_phenotype(g, env_cfg=env_cfg, sim_cfg=sim_cfg, age_steps=100)
    assert forced_juvenile.developmental_expression_factor == 0.5
    assert forced_juvenile.force_juvenile_cognition_active is True

    sim_cfg.force_juvenile_cognition = False
    sim_cfg.force_legacy_full_cognition = True
    legacy = express_cognitive_phenotype(g, env_cfg=env_cfg, sim_cfg=sim_cfg, age_steps=0)
    assert legacy.developmental_expression_factor == 1.0
    assert legacy.effective_memory_dim == g.memory_dim
    assert g.memory_dim == 64


def test_stage4_developmental_cost_is_telemetry_only_by_default_and_charged_when_enabled() -> None:
    env_cfg = EnvConfig(width=8, height=8, view_radius=3)
    sim_cfg = _dev_cfg()
    phenotype = express_cognitive_phenotype(_genome(), env_cfg=env_cfg, sim_cfg=sim_cfg, age_steps=5)
    ledger = compute_cognitive_budget_ledger(phenotype, env_cfg=env_cfg, sim_cfg=sim_cfg)

    assert ledger.developmental_cognition_cost_hypothetical > 0.0
    assert ledger.maturation_cost_hypothetical > 0.0
    assert ledger.developmental_cognition_cost_charged == 0.0
    assert ledger.maturation_cost_charged == 0.0
    assert ledger.charged_energy_delta == 0.0

    sim_cfg.enable_experimental_developmental_cost_charging = True
    charged_pheno = express_cognitive_phenotype(_genome(), env_cfg=env_cfg, sim_cfg=sim_cfg, age_steps=5)
    charged = compute_cognitive_budget_ledger(charged_pheno, env_cfg=env_cfg, sim_cfg=sim_cfg)
    assert charged.developmental_cost_charging_enabled is True
    assert charged.developmental_cognition_cost_charged > 0.0
    assert charged.maturation_cost_charged > 0.0
    assert charged.charged_energy_delta >= charged.developmental_cognition_cost_charged


def test_stage4_metrics_are_deterministic_and_include_guardrails() -> None:
    env_cfg = EnvConfig(width=8, height=8, view_radius=3)
    sim_cfg = _dev_cfg()
    sim_cfg.force_juvenile_cognition = True
    sim_cfg.developmental_pressure_scenario = True
    sim_cfg.developmental_pressure_type = "delayed_maturity"
    agents = [_agent(1, _genome(), age=5), _agent(2, _genome(), age=25, offspring=1)]

    a = compute_cognitive_status_metrics(agents, selected_parent_agents=agents[:1], env_cfg=env_cfg, sim_cfg=sim_cfg)
    b = compute_cognitive_status_metrics(agents, selected_parent_agents=agents[:1], env_cfg=env_cfg, sim_cfg=sim_cfg)

    assert a == b
    assert a["developmental_expression_factor_mean"] == 0.5
    assert a["juvenile_fraction"] == 0.5
    assert a["mature_fraction"] == 0.5
    assert a["developmental_force_juvenile_active_flag"] == 1.0
    assert a["developmental_pressure_scenario_flag"] == 1.0
    assert a["developmental_pressure_delayed_maturity_flag"] == 1.0
    assert "developmental_strategy_count_approx" in a
    assert "direct_development_reward_present_flag" in a


def test_stage4_developmental_pressure_fixture_is_deterministic_and_reward_free() -> None:
    fixture = _DelayedMaturityFixture()
    meta = fixture.metadata()
    assert meta["developmental_pressure_scenario"] is True
    assert meta["developmental_pressure_type"] == "delayed_maturity"
    assert meta["direct_development_reward"] is False

    juvenile_a, adult_a = fixture.phenotypes(ablated=False)
    juvenile_b, adult_b = fixture.phenotypes(ablated=False)
    legacy_juvenile, legacy_adult = fixture.phenotypes(ablated=True)

    assert juvenile_a == juvenile_b
    assert adult_a == adult_b
    assert juvenile_a["policy_hidden_size"] == adult_a["policy_hidden_size"] == 64
    assert juvenile_a["max_sensor_radius"] == adult_a["max_sensor_radius"]
    assert legacy_juvenile["effective_memory_dim"] == legacy_adult["effective_memory_dim"]

    root = Path(__file__).resolve().parents[3]
    selection_text = "\n".join(
        p.read_text(encoding="utf-8")
        for p in (
            root / "engine" / "simulation" / "runner_pkg" / "runner_selection.py",
            root / "engine" / "evolution" / "selection.py",
        )
    )
    assert "development_reward" not in selection_text
    assert "delayed_maturity_reward" not in selection_text
