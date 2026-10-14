from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from agents.cognition import (
    compute_cognitive_budget_ledger,
    compute_env_cost_charge_scalar_from_ledger,
    compute_env_cost_charge_scalar_from_surface,
    express_cognitive_phenotype,
    express_cost_charge_surface,
)
from agents.developmental_program_v1 import canonicalize_developmental_program
from agents.directed_inheritance import implemented_directed_inheritance_channel_mask
from agents.genome import Genome
from agents.introspection import implemented_introspection_channel_mask
from agents.perception import implemented_sensor_channel_mask
from agents.self_modification import implemented_self_modification_target_mask
from agents.self_regulation import implemented_self_regulation_control_mask
from env.env_config import EnvConfig
from simulation.env_hotpath_expression_diagnostics import EnvHotpathExpressionDiagnostics
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
        directed_inheritance_channel_mask=implemented_directed_inheritance_channel_mask(),
        drive_inertia=0.85,
        drive_noise=0.02,
        drive_logit_gain=1.1,
        pred_lr=float(pred_lr),
        developmental_program=canonicalize_developmental_program(None),
    )


def _agent(agent_id: int = 1, *, energy: float = 80.0, pred_lr: float = 0.2, age: int = 8) -> SimpleNamespace:
    genome = _genome(pred_lr=pred_lr)
    return SimpleNamespace(
        agent_id=int(agent_id),
        genome=genome,
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
        directed_inheritance_payload=None,
    )


def _charged_cfg(**overrides: object) -> SimConfig:
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
    for name, value in overrides.items():
        setattr(sim_cfg, name, value)
    return sim_cfg


def _reference(agent: SimpleNamespace, sim_cfg: SimConfig) -> float:
    env_cfg = EnvConfig(width=8, height=8, view_radius=2)
    phenotype = express_cognitive_phenotype(agent.genome, agent=agent, env_cfg=env_cfg, sim_cfg=sim_cfg)
    ledger = compute_cognitive_budget_ledger(phenotype, agent=agent, env_cfg=env_cfg, sim_cfg=sim_cfg)
    return compute_env_cost_charge_scalar_from_ledger(ledger)


def _surface(agent: SimpleNamespace, sim_cfg: SimConfig) -> float:
    env_cfg = EnvConfig(width=8, height=8, view_radius=2)
    surface = express_cost_charge_surface(agent.genome, agent=agent, env_cfg=env_cfg, sim_cfg=sim_cfg)
    return compute_env_cost_charge_scalar_from_surface(surface, sim_cfg=sim_cfg)


def test_costsurf1_surface_matches_full_phenotype_plus_ledger_scalar() -> None:
    sim_cfg = _charged_cfg()
    agent = _agent(energy=80.0, pred_lr=0.4, age=11)

    np.testing.assert_allclose(_surface(agent, sim_cfg), _reference(agent, sim_cfg), rtol=0.0, atol=1e-12)
    assert _surface(agent, sim_cfg) > 0.0


def test_costsurf1_surface_matches_with_envhot4_enabled_and_disabled() -> None:
    agent = _agent(energy=80.0, pred_lr=0.4, age=11)

    for enabled in (False, True):
        sim_cfg = _charged_cfg(env_hotpath_inactive_phenotype_defaults_enable=enabled)
        np.testing.assert_allclose(_surface(agent, sim_cfg), _reference(agent, sim_cfg), rtol=0.0, atol=1e-12)


def test_costsurf1_surface_matches_for_forced_and_ablated_states() -> None:
    variants = [
        _charged_cfg(ablate_prediction=True),
        _charged_cfg(force_prediction_zero_lr=True),
        _charged_cfg(force_all_introspection_channels=True),
        _charged_cfg(ablate_introspection=True),
        _charged_cfg(force_all_self_regulation_controls=True),
        _charged_cfg(ablate_self_regulation=True),
        _charged_cfg(force_all_self_modification_targets=True),
        _charged_cfg(ablate_self_modification=True),
        _charged_cfg(force_default_sensor_radius=True),
    ]
    agent = _agent(energy=80.0, pred_lr=0.4, age=11)

    for sim_cfg in variants:
        np.testing.assert_allclose(_surface(agent, sim_cfg), _reference(agent, sim_cfg), rtol=0.0, atol=1e-12)


def test_costsurf1_surface_preserves_directed_visibility_but_excludes_direct_env_charge() -> None:
    sim_cfg = _charged_cfg(
        enable_directed_inheritance_surface=True,
        force_all_directed_inheritance_channels=True,
        enable_experimental_directed_inheritance_cost_charging=True,
    )
    agent = _agent(energy=80.0, pred_lr=0.4, age=11)
    env_cfg = EnvConfig(width=8, height=8, view_radius=2)
    phenotype = express_cognitive_phenotype(agent.genome, agent=agent, env_cfg=env_cfg, sim_cfg=sim_cfg)
    ledger = compute_cognitive_budget_ledger(phenotype, agent=agent, env_cfg=env_cfg, sim_cfg=sim_cfg)
    surface = express_cost_charge_surface(agent.genome, agent=agent, env_cfg=env_cfg, sim_cfg=sim_cfg)

    assert phenotype.directed_inheritance_enabled is True
    assert ledger.directed_inheritance_cost_total_component > 0.0
    assert surface.directed_inheritance_visible_but_not_env_charged is True
    np.testing.assert_allclose(
        compute_env_cost_charge_scalar_from_surface(surface, sim_cfg=sim_cfg),
        compute_env_cost_charge_scalar_from_ledger(ledger),
        rtol=0.0,
        atol=1e-12,
    )


def test_costsurf1_surface_respects_disabled_cost_and_ledger_flags() -> None:
    agent = _agent(energy=80.0, pred_lr=0.4, age=11)

    assert _surface(agent, SimConfig(seed=17)) == 0.0
    assert _reference(agent, SimConfig(seed=17)) == 0.0

    sim_cfg = _charged_cfg(enable_cognitive_budget_ledger=False)
    assert _surface(agent, sim_cfg) == 0.0
    assert _reference(agent, sim_cfg) == 0.0


def test_costsurf1_sim_config_flag_defaults_off() -> None:
    sim_cfg = SimConfig()

    assert sim_cfg.env_hotpath_cost_charge_surface_enable is False
    assert sim_cfg.env_hotpath_cost_charge_surface_schema_version == 1
    assert sim_cfg.env_hotpath_cost_charge_surface_guard_enable is False
    assert sim_cfg.env_hotpath_cost_charge_surface_guard_mode == "sample"
    assert sim_cfg.env_hotpath_cost_charge_surface_guard_sample_period == 1000
    assert sim_cfg.env_hotpath_costsurface_soa_diagnostics_enable is False


def test_costsurf1_diagnostics_serializes_surface_counters(tmp_path) -> None:
    diag = EnvHotpathExpressionDiagnostics(run_dir=str(tmp_path), decomposition_enabled=True)

    diag.record_cost_charge_surface(
        {
            "hit_count": 7,
            "fallback_count": 1,
            "guard_check_count": 3,
            "guard_mismatch_count": 0,
            "remaining_full_phenotype_count": 3,
            "max_abs_diff": 0.0,
            "surface_s": 0.04,
            "scalar_s": 0.01,
            "guard_reference_s": 0.03,
            "fallback_s": 0.02,
            "skipped_reason_counts": {"exception": 1},
        }
    )

    diag.write_decomposition_summary()
    payload = __import__("json").loads(
        (tmp_path / "metrics" / "env_apply_decomposition_summary.json").read_text(encoding="utf-8")
    )

    assert payload["cost_charge_surface"]["hit_count"] == 7
    assert payload["cost_charge_surface"]["fallback_count"] == 1
    assert payload["cost_charge_surface"]["guard_check_count"] == 3
    assert payload["cost_charge_surface"]["guard_mismatch_count"] == 0
    assert payload["cost_charge_surface"]["remaining_full_phenotype_count"] == 3
    assert payload["cost_charge_surface"]["skipped_reason_counts"] == {"exception": 1}
    assert payload["seconds"]["env_apply_cost_charge_surface_total"] == 0.04
    assert payload["seconds"]["env_apply_cost_charge_surface_scalar_total"] == 0.01
    assert payload["seconds"]["env_apply_cost_charge_surface_guard_reference_total"] == 0.03
    assert payload["seconds"]["env_apply_cost_charge_surface_fallback_total"] == 0.02
