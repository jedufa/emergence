from __future__ import annotations

import copy
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np

from agents.cognition import (
    _envhot4_empty_directed_payload_default,
    _envhot4_empty_directed_payload_summary_default,
    _envhot4_inactive_directed_inheritance_default,
    _envhot4_inactive_introspection_default,
    _envhot4_inactive_self_modification_default,
    _envhot4_inactive_self_regulation_default,
    compute_cognitive_budget_ledger,
    compute_env_apply_cognitive_cost_scalar,
    express_cognitive_phenotype,
)
from agents.developmental_program_v1 import canonicalize_developmental_program
from agents.directed_inheritance import (
    DirectedInheritancePayload,
    directed_inheritance_payload_summary,
    express_directed_inheritance,
    implemented_directed_inheritance_channel_mask,
)
from agents.genome import Genome
from agents.introspection import express_introspection, implemented_introspection_channel_mask
from agents.perception import implemented_sensor_channel_mask
from agents.self_modification import express_self_modification, implemented_self_modification_target_mask
from agents.self_regulation import express_self_regulation, implemented_self_regulation_control_mask
from env.env_config import EnvConfig
from simulation.env_hotpath_expression_diagnostics import (
    EnvHotpathExpressionDiagnostics,
    reset_current_diagnostics,
    set_current_diagnostics,
)
from simulation.runner_pkg.runner_config import SimConfig


def _genome() -> Genome:
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
        directed_inheritance_channel_mask=implemented_directed_inheritance_channel_mask(),
        drive_inertia=0.85,
        drive_noise=0.02,
        drive_logit_gain=1.1,
        pred_lr=0.2,
        developmental_program=canonicalize_developmental_program(None),
    )


def _agent(genome: Genome, *, payload: object = None) -> SimpleNamespace:
    stats = SimpleNamespace(
        survival_steps=10,
        pred_err_count=2,
        pred_err_sum=0.5,
        prediction_update_count=4,
        offspring_count=0,
        patch_habitat_lineage_id=1,
    )
    return SimpleNamespace(
        agent_id=1,
        genome=genome,
        energy=80.0,
        health=9.0,
        alive=True,
        drives=np.asarray([0.8, 0.2, 0.4, 0.6], dtype=np.float32),
        stats=stats,
        directed_inheritance_payload=payload,
    )


def _cfg(**overrides: object) -> SimConfig:
    cfg = SimConfig(seed=11)
    cfg.genome_c1_enable = False
    cfg.genome_c2_enable = False
    for name, value in overrides.items():
        setattr(cfg, name, value)
    return cfg


def _optimized_copy(cfg: SimConfig) -> SimConfig:
    out = copy.deepcopy(cfg)
    out.env_hotpath_inactive_phenotype_defaults_enable = True
    return out


def _phenotype_dict(cfg: SimConfig, *, payload: object = None) -> dict[str, object]:
    genome = _genome()
    return asdict(
        express_cognitive_phenotype(
            genome,
            agent=_agent(genome, payload=payload),
            env_cfg=EnvConfig(width=8, height=8, view_radius=2),
            sim_cfg=cfg,
        )
    )


def _expressed_phenotype(cfg: SimConfig, *, payload: object = None):
    genome = _genome()
    return express_cognitive_phenotype(
        genome,
        agent=_agent(genome, payload=payload),
        env_cfg=EnvConfig(width=8, height=8, view_radius=2),
        sim_cfg=cfg,
    )


def _env_apply_ledger_cost(ledger: object) -> float:
    return float(getattr(ledger, "perception_cost_charged", 0.0)) + float(
        getattr(ledger, "prediction_update_cost_charged", 0.0)
    ) + float(
        getattr(ledger, "developmental_cognition_cost_charged", 0.0)
    ) + float(
        getattr(ledger, "maturation_cost_charged", 0.0)
    ) + float(
        getattr(ledger, "introspection_cost_total_component", 0.0)
    ) + float(
        getattr(ledger, "self_regulation_cost_total_component", 0.0)
    ) + float(
        getattr(ledger, "self_modification_cost_total_component", 0.0)
    )


def test_envhot4_plain_disabled_factories_match_current_helper_asdict() -> None:
    cfg = _cfg()
    genome = _genome()

    assert asdict(_envhot4_inactive_introspection_default()) == asdict(
        express_introspection(genome, sim_cfg=cfg)
    )
    assert asdict(_envhot4_inactive_self_regulation_default()) == asdict(
        express_self_regulation(genome, sim_cfg=cfg)
    )
    assert asdict(_envhot4_inactive_self_modification_default()) == asdict(
        express_self_modification(genome, sim_cfg=cfg)
    )
    assert asdict(_envhot4_inactive_directed_inheritance_default()) == asdict(
        express_directed_inheritance(genome, payload=DirectedInheritancePayload.from_dict(None), sim_cfg=cfg)
    )
    assert asdict(_envhot4_empty_directed_payload_default()) == asdict(
        DirectedInheritancePayload.from_dict(None)
    )
    assert _envhot4_empty_directed_payload_summary_default() == directed_inheritance_payload_summary(
        DirectedInheritancePayload.from_dict(None)
    )

    first = _envhot4_empty_directed_payload_default()
    second = _envhot4_empty_directed_payload_default()
    first.applied_influence_values["x"] = 1.0
    assert second.applied_influence_values == {}


def test_envhot4_optimized_flag_preserves_default_inactive_phenotype_asdict(tmp_path) -> None:
    cfg = _optimized_copy(_cfg())
    diag = EnvHotpathExpressionDiagnostics(run_dir=str(tmp_path), phenotype_decomposition_enabled=True)
    token = set_current_diagnostics(diag)
    try:
        optimized = _phenotype_dict(cfg)
    finally:
        reset_current_diagnostics(token)

    assert optimized == _phenotype_dict(_cfg())
    assert diag.phenotype_inactive_default_counts["factory:introspection"] == 1
    assert diag.phenotype_inactive_default_counts["factory:self_regulation"] == 1
    assert diag.phenotype_inactive_default_counts["factory:self_modification"] == 1
    assert diag.phenotype_inactive_default_counts["factory:directed_payload_empty"] == 1
    assert diag.phenotype_inactive_default_counts["factory:directed_inheritance"] == 1
    assert diag.phenotype_inactive_default_counts["factory:directed_payload_summary_empty"] == 1
    expected_buckets = {
        "phenotype_total",
        "phenotype_scalar_extraction_total",
        "phenotype_perception_total",
        "phenotype_developmental_expression_total",
        "phenotype_dataclass_assembly_total",
        "phenotype_introspection_total",
        "phenotype_self_regulation_total",
        "phenotype_self_modification_total",
        "phenotype_directed_payload_from_dict_total",
        "phenotype_directed_inheritance_total",
        "phenotype_directed_payload_summary_total",
    }
    assert expected_buckets.issubset(set(diag.phenotype_seconds))
    assert diag.phenotype_counts["phenotype_total"] == 1


def test_envhot4_falls_back_and_preserves_flag_sensitive_phenotype_asdict() -> None:
    variants = [
        _cfg(force_all_introspection_channels=True),
        _cfg(ablate_introspection=True),
        _cfg(enable_introspection_channels=True, default_introspection_channels_enabled=True),
        _cfg(enable_introspection_channels=True, default_introspection_channels_enabled=False),
        _cfg(force_all_self_regulation_controls=True),
        _cfg(ablate_self_regulation=True),
        _cfg(enable_self_regulation_surface=True, default_self_regulation_controls_enabled=True),
        _cfg(enable_self_regulation_surface=True, default_self_regulation_controls_enabled=False),
        _cfg(force_all_self_modification_targets=True),
        _cfg(ablate_self_modification=True),
        _cfg(enable_self_modification=True, default_self_modification_targets_enabled=True),
        _cfg(enable_self_modification=True, default_self_modification_targets_enabled=False),
        _cfg(force_all_directed_inheritance_channels=True),
        _cfg(ablate_directed_inheritance=True),
        _cfg(enable_directed_inheritance_surface=True, default_directed_inheritance_channels_enabled=True),
        _cfg(enable_directed_inheritance_surface=True, default_directed_inheritance_channels_enabled=False),
    ]

    for base in variants:
        assert _phenotype_dict(_optimized_copy(base)) == _phenotype_dict(base)


def test_envhot4_preserves_existing_directed_payload_object_summary() -> None:
    payload = DirectedInheritancePayload.from_dict(None)
    payload.offspring_developmental_bias_offsets["offspring_developmental_expression_bias"] = 0.125
    payload.offspring_phenotype_bias_offsets["offspring_prediction_lr_bias"] = -0.05
    payload.payload_applied = True

    assert _phenotype_dict(_optimized_copy(_cfg()), payload=payload) == _phenotype_dict(_cfg(), payload=payload)
    assert payload.offspring_developmental_bias_offsets["offspring_developmental_expression_bias"] == 0.125
    assert payload.offspring_phenotype_bias_offsets["offspring_prediction_lr_bias"] == -0.05


def test_envhot5_budget_scalar_fastpath_matches_current_env_apply_ledger_sum(tmp_path) -> None:
    variants = [
        _optimized_copy(
            _cfg(
                enable_experimental_perception_cost_charging=True,
                enable_experimental_sensor_radius_cost_charging=True,
            )
        ),
        _optimized_copy(_cfg(enable_experimental_prediction_cost_charging=True)),
        _optimized_copy(
            _cfg(
                enable_developmental_cognition=True,
                enable_experimental_developmental_cost_charging=True,
            )
        ),
        _optimized_copy(
            _cfg(
                enable_introspection_channels=True,
                default_introspection_channels_enabled=True,
                enable_experimental_introspection_cost_charging=True,
            )
        ),
        _optimized_copy(
            _cfg(
                enable_self_regulation_surface=True,
                default_self_regulation_controls_enabled=True,
                enable_experimental_self_regulation_cost_charging=True,
            )
        ),
        _optimized_copy(
            _cfg(
                enable_self_modification=True,
                default_self_modification_targets_enabled=True,
                enable_experimental_self_modification_cost_charging=True,
            )
        ),
        _optimized_copy(_cfg(enable_cognitive_budget_ledger=False)),
    ]
    diag = EnvHotpathExpressionDiagnostics(run_dir=str(tmp_path), phenotype_decomposition_enabled=True)
    token = set_current_diagnostics(diag)
    try:
        for cfg in variants:
            phenotype = _expressed_phenotype(cfg)
            ledger = compute_cognitive_budget_ledger(
                phenotype,
                agent=_agent(_genome()),
                env_cfg=EnvConfig(width=8, height=8, view_radius=2),
                sim_cfg=cfg,
            )
            assert compute_env_apply_cognitive_cost_scalar(
                phenotype,
                agent=_agent(_genome()),
                env_cfg=EnvConfig(width=8, height=8, view_radius=2),
                sim_cfg=cfg,
            ) == _env_apply_ledger_cost(ledger)
    finally:
        reset_current_diagnostics(token)

    assert diag.phenotype_counts["phenotype_budget_ledger_scalar_fastpath_total"] == len(variants)
