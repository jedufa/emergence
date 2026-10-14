from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from agents.cognition import (
    compute_cognitive_budget_ledger,
    compute_env_cost_charge_scalar,
    compute_env_cost_charge_scalar_from_ledger,
    express_cognitive_phenotype,
)
from agents.developmental_program_v1 import canonicalize_developmental_program
from agents.genome import Genome
from agents.introspection import implemented_introspection_channel_mask
from agents.perception import implemented_sensor_channel_mask
from agents.self_modification import implemented_self_modification_target_mask
from agents.self_regulation import implemented_self_regulation_control_mask
from env.env_config import EnvConfig
from simulation.runner_pkg.runner_config import SimConfig


def _genome(*, pred_lr: float = 0.2, memory_dim: int = 32) -> Genome:
    return Genome(
        hidden_size=64,
        exploration_bias=0.1,
        aggression_bias=0.2,
        reproduction_threshold=30.0,
        mutation_rate=0.03,
        action_temperature=1.0,
        use_gru=True,
        memory_dim=int(memory_dim),
        sensor_channel_mask=implemented_sensor_channel_mask(),
        sensor_radius=2,
        introspection_channel_mask=implemented_introspection_channel_mask(),
        self_regulation_control_mask=implemented_self_regulation_control_mask(),
        self_modification_target_mask=implemented_self_modification_target_mask(),
        drive_inertia=0.85,
        drive_noise=0.02,
        drive_logit_gain=1.1,
        pred_lr=float(pred_lr),
        developmental_program=canonicalize_developmental_program(None),
    )


def _agent(agent_id: int = 1, *, energy: float = 80.0, pred_lr: float = 0.2, age: int = 8) -> SimpleNamespace:
    return SimpleNamespace(
        agent_id=int(agent_id),
        genome=_genome(pred_lr=pred_lr),
        energy=float(energy),
        health=9.0,
        alive=True,
        drives=np.asarray([0.8, 0.2, 0.4, 0.6], dtype=np.float32),
        stats=SimpleNamespace(
            survival_steps=int(age),
            pred_err_count=2,
            pred_err_sum=0.5,
            prediction_update_count=4,
            offspring_count=0,
            patch_habitat_lineage_id=1,
        ),
    )


def _charged_cfg() -> SimConfig:
    sim_cfg = SimConfig(seed=17)
    sim_cfg.enable_experimental_prediction_cost_charging = True
    sim_cfg.prediction_cost_base = 0.01
    sim_cfg.prediction_cost_per_lr = 0.02
    sim_cfg.prediction_state_cost = 0.003
    sim_cfg.enable_experimental_perception_cost_charging = True
    sim_cfg.enable_experimental_sensor_radius_cost_charging = True
    sim_cfg.enable_developmental_cognition = True
    sim_cfg.developmental_cognition_mode = "linear_maturation"
    sim_cfg.enable_experimental_developmental_cost_charging = True
    sim_cfg.developmental_cognition_cost_base = 0.004
    sim_cfg.maturation_cost_per_tick = 0.002
    sim_cfg.enable_introspection_channels = True
    sim_cfg.force_all_introspection_channels = True
    sim_cfg.enable_experimental_introspection_cost_charging = True
    sim_cfg.enable_self_regulation = True
    sim_cfg.force_all_self_regulation_controls = True
    sim_cfg.enable_experimental_self_regulation_cost_charging = True
    sim_cfg.enable_self_modification = True
    sim_cfg.force_all_self_modification_targets = True
    sim_cfg.enable_experimental_self_modification_cost_charging = True
    return sim_cfg


def _phenotype(agent: SimpleNamespace, sim_cfg: SimConfig):
    env_cfg = EnvConfig(width=8, height=8, view_radius=2)
    return express_cognitive_phenotype(agent.genome, agent=agent, env_cfg=env_cfg, sim_cfg=sim_cfg), env_cfg


def test_costfast1_direct_scalar_matches_ledger_env_charge() -> None:
    sim_cfg = _charged_cfg()
    agent = _agent(energy=80.0, pred_lr=0.4, age=11)
    phenotype, env_cfg = _phenotype(agent, sim_cfg)
    ledger = compute_cognitive_budget_ledger(phenotype, agent=agent, env_cfg=env_cfg, sim_cfg=sim_cfg)

    direct = compute_env_cost_charge_scalar(phenotype, sim_cfg=sim_cfg)
    reference = compute_env_cost_charge_scalar_from_ledger(ledger)

    np.testing.assert_allclose(direct, reference, rtol=0.0, atol=1e-12)
    assert direct > 0.0


def test_costfast1_direct_scalar_excludes_non_env_charge_components() -> None:
    sim_cfg = _charged_cfg()
    agent = _agent(energy=80.0, pred_lr=0.4, age=11)
    phenotype, env_cfg = _phenotype(agent, sim_cfg)
    ledger = compute_cognitive_budget_ledger(phenotype, agent=agent, env_cfg=env_cfg, sim_cfg=sim_cfg)

    direct = compute_env_cost_charge_scalar(phenotype, sim_cfg=sim_cfg)

    assert ledger.memory_upkeep_cost_charged > 0.0
    assert ledger.cognitive_upkeep_total > direct
    np.testing.assert_allclose(direct, compute_env_cost_charge_scalar_from_ledger(ledger), rtol=0.0, atol=1e-12)


def test_costfast1_direct_scalar_respects_disabled_cost_flags() -> None:
    sim_cfg = SimConfig(seed=17)
    agent = _agent(energy=80.0, pred_lr=0.4, age=11)
    phenotype, env_cfg = _phenotype(agent, sim_cfg)
    ledger = compute_cognitive_budget_ledger(phenotype, agent=agent, env_cfg=env_cfg, sim_cfg=sim_cfg)

    assert compute_env_cost_charge_scalar(phenotype, sim_cfg=sim_cfg) == 0.0
    assert compute_env_cost_charge_scalar_from_ledger(ledger) == 0.0


def test_costfast1_direct_scalar_respects_disabled_ledger_flag() -> None:
    sim_cfg = _charged_cfg()
    sim_cfg.enable_cognitive_budget_ledger = False
    agent = _agent(energy=80.0, pred_lr=0.4, age=11)
    phenotype, env_cfg = _phenotype(agent, sim_cfg)
    ledger = compute_cognitive_budget_ledger(phenotype, agent=agent, env_cfg=env_cfg, sim_cfg=sim_cfg)

    assert compute_env_cost_charge_scalar(phenotype, sim_cfg=sim_cfg) == 0.0
    assert compute_env_cost_charge_scalar_from_ledger(ledger) == 0.0


def test_costfast1_sim_config_flag_defaults_off() -> None:
    sim_cfg = SimConfig()

    assert sim_cfg.env_hotpath_direct_cost_scalar_enable is False
    assert sim_cfg.env_hotpath_direct_cost_scalar_schema_version == 1
    assert sim_cfg.env_hotpath_direct_cost_scalar_mismatch_guard is False
    assert sim_cfg.env_hotpath_direct_cost_scalar_guard_mode == "sample"
    assert sim_cfg.env_hotpath_direct_cost_scalar_guard_sample_period == 1000
