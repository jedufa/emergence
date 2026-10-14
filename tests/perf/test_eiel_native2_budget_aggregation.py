from __future__ import annotations

from dataclasses import asdict
from types import SimpleNamespace

import numpy as np

from agents.cognition import compute_cognitive_budget_ledger, compute_cognitive_budget_ledgers_batch, express_cognitive_phenotype
from agents.developmental_program_v1 import canonicalize_developmental_program
from agents.genome import Genome
from agents.introspection import implemented_introspection_channel_mask
from agents.perception import implemented_sensor_channel_mask
from agents.self_modification import implemented_self_modification_target_mask
from agents.self_regulation import implemented_self_regulation_control_mask
from env.env_config import EnvConfig
from simulation.runner_pkg.runner_config import SimConfig


def _genome(*, pred_lr: float = 0.2) -> Genome:
    return Genome(
        hidden_size=64,
        exploration_bias=0.1,
        aggression_bias=0.2,
        reproduction_threshold=30.0,
        mutation_rate=0.03,
        action_temperature=1.0,
        use_gru=True,
        memory_dim=32,
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


def _agent(agent_id: int, *, energy: float, pred_lr: float = 0.2, age: int = 8) -> SimpleNamespace:
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


def _cfg() -> SimConfig:
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


def _assert_ledgers_close(left, right) -> None:
    assert asdict(left).keys() == asdict(right).keys()
    for key, left_value in asdict(left).items():
        right_value = asdict(right)[key]
        if isinstance(left_value, float):
            np.testing.assert_allclose(left_value, right_value, rtol=0.0, atol=1e-12, err_msg=key)
        else:
            assert left_value == right_value


def test_native2_batch_budget_ledgers_match_python_reference() -> None:
    env_cfg = EnvConfig(width=8, height=8, view_radius=2)
    sim_cfg = _cfg()
    agents = [_agent(1, energy=80.0, pred_lr=0.1), _agent(2, energy=65.0, pred_lr=0.4)]
    phenotypes = [express_cognitive_phenotype(agent.genome, agent=agent, env_cfg=env_cfg, sim_cfg=sim_cfg) for agent in agents]

    expected = [
        compute_cognitive_budget_ledger(phenotype, agent=agent, env_cfg=env_cfg, sim_cfg=sim_cfg)
        for phenotype, agent in zip(phenotypes, agents)
    ]
    actual, info = compute_cognitive_budget_ledgers_batch(phenotypes, agents=agents, env_cfg=env_cfg, sim_cfg=sim_cfg)

    assert info["schema_version"] == "native_cognitive_budget_cost_aggregation_v1"
    assert info["pack_rows"] == len(agents)
    assert int(info["hit_count"]) + int(info["fallback_count"]) == len(agents)
    assert len(actual) == len(expected)
    for lhs, rhs in zip(actual, expected):
        _assert_ledgers_close(lhs, rhs)


def test_native2_batch_fallback_metadata_when_numba_unavailable(monkeypatch) -> None:
    import engine.perf.native as native_pkg

    monkeypatch.setattr(native_pkg, "NUMBA_AVAILABLE", False)
    monkeypatch.setattr(native_pkg, "backend_name", lambda: "numpy-fallback")

    env_cfg = EnvConfig(width=8, height=8, view_radius=2)
    sim_cfg = _cfg()
    agents = [_agent(1, energy=80.0)]
    phenotypes = [express_cognitive_phenotype(agents[0].genome, agent=agents[0], env_cfg=env_cfg, sim_cfg=sim_cfg)]

    ledgers, info = compute_cognitive_budget_ledgers_batch(phenotypes, agents=agents, env_cfg=env_cfg, sim_cfg=sim_cfg)

    assert len(ledgers) == 1
    assert info["backend"] == "numpy-fallback"
    assert info["hit_count"] == 0
    assert info["fallback_count"] == 1


def test_native2_mismatch_guard_falls_back_to_python_reference(monkeypatch) -> None:
    import engine.perf.native as native_pkg

    def _bad_native(components, energy):
        totals = np.zeros((components.shape[0],), dtype=np.float64)
        energy_fraction = np.zeros_like(totals)
        fractions = np.zeros_like(components)
        return totals, energy_fraction, fractions

    monkeypatch.setattr(native_pkg, "NUMBA_AVAILABLE", True)
    monkeypatch.setattr(native_pkg, "backend_name", lambda: "numba")
    monkeypatch.setattr(native_pkg, "cognitive_costs_native", _bad_native)

    env_cfg = EnvConfig(width=8, height=8, view_radius=2)
    sim_cfg = _cfg()
    sim_cfg.native_cognitive_budget_cost_aggregation_mismatch_guard = True
    agents = [_agent(1, energy=80.0)]
    phenotypes = [express_cognitive_phenotype(agents[0].genome, agent=agents[0], env_cfg=env_cfg, sim_cfg=sim_cfg)]
    expected = compute_cognitive_budget_ledger(phenotypes[0], agent=agents[0], env_cfg=env_cfg, sim_cfg=sim_cfg)

    actual, info = compute_cognitive_budget_ledgers_batch(phenotypes, agents=agents, env_cfg=env_cfg, sim_cfg=sim_cfg)

    assert info["mismatch_count"] == 1
    assert info["fallback_count"] == 1
    _assert_ledgers_close(actual[0], expected)


def test_native2_sim_config_flag_defaults_off() -> None:
    sim_cfg = SimConfig()

    assert sim_cfg.native_cognitive_budget_cost_aggregation_enable is False
    assert sim_cfg.native_cognitive_budget_cost_aggregation_schema_version == 1
    assert sim_cfg.native_cognitive_budget_cost_aggregation_fallback_mode == "numpy"
