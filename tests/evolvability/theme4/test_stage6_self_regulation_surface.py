from __future__ import annotations

import random
from types import SimpleNamespace

import numpy as np

from agents.cognition import compute_cognitive_budget_ledger, express_cognitive_phenotype, get_cognitive_trait
from agents.developmental_program_v1 import canonicalize_developmental_program
from agents.genome import Genome
from agents.introspection import implemented_introspection_channel_mask
from agents.perception import implemented_sensor_channel_mask
from agents.self_regulation import (
    SelfRegulationRuntimeState,
    apply_self_regulation_controls,
    forbidden_self_regulation_targets,
    implemented_self_regulation_control_ids,
    implemented_self_regulation_control_mask,
    mutate_self_regulation_control_mask,
    recombine_self_regulation_control_mask,
    sanitize_self_regulation_control_mask,
    self_regulation_control_registry,
    self_regulation_control_registry_dict,
    self_regulation_runtime_summary,
)
from env.env_config import EnvConfig
from evolution.reproduction import crossover_genome
from simulation.metrics_pkg.cognitive_status import compute_cognitive_status_metrics
from simulation.runner_pkg.runner_config import SimConfig


def _genome(*, self_regulation_control_mask: int | None = None) -> Genome:
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
        self_regulation_control_mask=implemented_self_regulation_control_mask()
        if self_regulation_control_mask is None
        else int(self_regulation_control_mask),
        drive_inertia=0.85,
        drive_noise=0.02,
        drive_logit_gain=1.1,
        pred_lr=0.2,
        developmental_program=canonicalize_developmental_program(None),
    )


def _agent(genome: Genome, *, age: int = 10) -> SimpleNamespace:
    stats = SimpleNamespace(
        survival_steps=int(age),
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
    )


def _cfg(*, enabled: bool = True) -> SimConfig:
    sim_cfg = SimConfig(seed=6)
    sim_cfg.enable_self_regulation = bool(enabled)
    sim_cfg.enable_self_regulation_surface = bool(enabled)
    sim_cfg.default_self_regulation_controls_enabled = False
    sim_cfg.enable_evolvable_self_regulation_controls = True
    sim_cfg.enable_self_regulation_control_recombination = True
    sim_cfg.enable_self_regulation_control_mutation = True
    sim_cfg.genome_c1_enable = False
    sim_cfg.genome_c2_enable = False
    return sim_cfg


def test_stage6_control_registry_is_stable_safe_and_forbidden_targets_are_excluded() -> None:
    controls = self_regulation_control_registry()
    ids = [c.control_id for c in controls]
    assert ids[:5] == [
        "exploration_pressure_adjustment",
        "action_temperature_adjustment",
        "drive_gain_adjustment",
        "reproduction_threshold_adjustment",
        "prediction_lr_adjustment",
    ]
    assert tuple(implemented_self_regulation_control_ids()) == tuple(c.control_id for c in controls if c.active_now and not c.future_reserved)
    forbidden = set(forbidden_self_regulation_targets())
    for expected in ("genome", "source_code", "simulator_code", "filesystem", "rng_state", "energy_direct_set", "fitness_direct_set", "death_direct_set"):
        assert expected in forbidden
    for control in controls:
        if control.active_now:
            assert control.safe_for_agent_control is True
            assert control.can_affect_genome is False
            assert control.can_affect_host_or_simulator is False
            assert control.can_affect_source_code is False
            assert control.target_variable not in forbidden
    assert all(not c.active_now for c in controls if c.future_reserved)
    assert self_regulation_control_registry_dict()["schema_version"] == "self_regulation_control_registry_v1"


def test_stage6_self_regulation_traits_are_registered_honestly() -> None:
    mask_trait = get_cognitive_trait("self_regulation_control_mask")
    assert mask_trait.heritable is True
    assert mask_trait.directly_mutable is True
    assert mask_trait.recombinable is True
    assert mask_trait.control_ready is True
    runtime_trait = get_cognitive_trait("self_regulation_runtime_state")
    assert runtime_trait.phenotype_surface_present is True
    assert runtime_trait.heritable is False
    assert runtime_trait.control_ready is True
    assert get_cognitive_trait("future_self_regulation_channel").status == "future_reserved"


def test_stage6_default_behavior_preserves_legacy_surface() -> None:
    env_cfg = EnvConfig(width=8, height=8, view_radius=2)
    sim_cfg = _cfg(enabled=False)
    genome = _genome()
    phenotype = express_cognitive_phenotype(genome, env_cfg=env_cfg, sim_cfg=sim_cfg)
    assert phenotype.self_regulation_enabled is False
    assert phenotype.active_self_regulation_control_count == 0
    assert phenotype.self_regulation_policy_output_enabled is False
    assert phenotype.self_modification_active is False
    assert phenotype.genome_editing_active is False


def test_stage6_mask_serialization_recombination_and_mutation_are_deterministic() -> None:
    mask = implemented_self_regulation_control_mask()
    genome = _genome(self_regulation_control_mask=mask)
    restored = Genome.from_dict(genome.to_dict())
    assert restored.self_regulation_control_mask == mask

    old_payload = genome.to_dict()
    old_payload.pop("self_regulation_control_mask", None)
    assert Genome.from_dict(old_payload).self_regulation_control_mask == implemented_self_regulation_control_mask()

    low = 1
    high = implemented_self_regulation_control_mask()
    assert recombine_self_regulation_control_mask(low, high, random.Random(7)) == recombine_self_regulation_control_mask(low, high, random.Random(7))
    assert mutate_self_regulation_control_mask(low, random.Random(9), 1.0) == mutate_self_regulation_control_mask(low, random.Random(9), 1.0)
    assert sanitize_self_regulation_control_mask(1 << 30) == 0


def test_stage6_reproduction_self_regulation_mask_is_gated_and_deterministic() -> None:
    g1 = _genome(self_regulation_control_mask=1)
    g2 = _genome(self_regulation_control_mask=implemented_self_regulation_control_mask())
    disabled = _cfg(enabled=False)
    disabled.enable_evolvable_self_regulation_controls = False
    child_disabled = crossover_genome(g1, g2, random.Random(1), sim_cfg=disabled)
    assert child_disabled.self_regulation_control_mask == 1

    enabled = _cfg(enabled=True)
    a = crossover_genome(g1, g2, random.Random(11), sim_cfg=enabled)
    b = crossover_genome(g1, g2, random.Random(11), sim_cfg=enabled)
    assert a.self_regulation_control_mask == b.self_regulation_control_mask


def test_stage6_phenotype_ablation_and_budget_charging() -> None:
    env_cfg = EnvConfig(width=8, height=8, view_radius=2)
    sim_cfg = _cfg(enabled=True)
    sim_cfg.force_all_self_regulation_controls = True
    agent = _agent(_genome())
    phenotype = express_cognitive_phenotype(agent.genome, agent=agent, env_cfg=env_cfg, sim_cfg=sim_cfg)
    ledger = compute_cognitive_budget_ledger(phenotype, agent=agent, env_cfg=env_cfg, sim_cfg=sim_cfg)
    assert phenotype.active_self_regulation_control_count == len(implemented_self_regulation_control_ids())
    assert phenotype.active_self_regulation_control_ids == implemented_self_regulation_control_ids()
    assert phenotype.self_regulation_inheritable is False
    assert ledger.self_regulation_control_cost_hypothetical > 0.0
    assert ledger.self_regulation_control_cost_charged == 0.0
    assert ledger.self_regulation_cost_telemetry_only is True

    sim_cfg.enable_experimental_self_regulation_cost_charging = True
    charged_pheno = express_cognitive_phenotype(agent.genome, agent=agent, env_cfg=env_cfg, sim_cfg=sim_cfg)
    charged = compute_cognitive_budget_ledger(charged_pheno, agent=agent, env_cfg=env_cfg, sim_cfg=sim_cfg)
    assert charged.self_regulation_control_cost_charged > 0.0
    assert charged.self_regulation_cost_charging_enabled is True
    assert charged.charged_energy_delta >= charged.self_regulation_control_cost_charged

    sim_cfg.enable_experimental_self_regulation_cost_charging = False
    sim_cfg.force_all_self_regulation_controls = False
    sim_cfg.force_no_self_regulation = True
    no_reg = express_cognitive_phenotype(agent.genome, agent=agent, env_cfg=env_cfg, sim_cfg=sim_cfg)
    assert no_reg.active_self_regulation_control_count == 0
    assert no_reg.force_no_self_regulation_active is True


def test_stage6_control_application_is_bounded_deterministic_and_non_genomic() -> None:
    sim_cfg = _cfg(enabled=True)
    sim_cfg.force_all_self_regulation_controls = True
    genome = _genome()
    original = genome.to_dict()
    vector = [1.0, -1.0, 10.0, -10.0, float("nan")]

    a = apply_self_regulation_controls(SelfRegulationRuntimeState(), vector, genome=genome, sim_cfg=sim_cfg)
    b = apply_self_regulation_controls(SelfRegulationRuntimeState(), vector, genome=genome, sim_cfg=sim_cfg)
    assert a.to_dict() == b.to_dict()
    assert genome.to_dict() == original
    assert a.offsets["exploration_pressure_adjustment"] == 0.05
    assert a.offsets["action_temperature_adjustment"] == -0.05
    assert a.offsets["drive_gain_adjustment"] == 0.05
    assert a.offsets["reproduction_threshold_adjustment"] == -1.0
    assert a.offsets["prediction_lr_adjustment"] == 0.0
    assert a.effective_regulated_values["exploration_pressure_adjustment"] > genome.exploration_bias
    assert a.effective_regulated_values["action_temperature_adjustment"] < genome.action_temperature

    sim_cfg.mask_self_regulation_outputs_to_neutral = True
    neutral = apply_self_regulation_controls(SelfRegulationRuntimeState(), vector, genome=genome, sim_cfg=sim_cfg)
    assert max(abs(v) for v in neutral.offsets.values()) == 0.0

    sim_cfg.mask_self_regulation_outputs_to_neutral = False
    sim_cfg.freeze_self_regulation_state = True
    frozen = apply_self_regulation_controls(SelfRegulationRuntimeState(), vector, genome=genome, sim_cfg=sim_cfg)
    assert frozen.offsets == {}

    sim_cfg.freeze_self_regulation_state = False
    sim_cfg.reset_self_regulation_each_tick = True
    reset = apply_self_regulation_controls(a, [0.0] * len(implemented_self_regulation_control_ids()), genome=genome, sim_cfg=sim_cfg)
    assert reset.control_reset_count == 1
    assert max(abs(v) for v in reset.offsets.values()) == 0.0


def test_stage6_metrics_include_self_regulation_status_and_guardrails() -> None:
    env_cfg = EnvConfig(width=8, height=8, view_radius=2)
    sim_cfg = _cfg(enabled=True)
    sim_cfg.force_all_self_regulation_controls = True
    sim_cfg.self_regulation_pressure_scenario = True
    sim_cfg.self_regulation_pressure_type = "hunger_fear_conflict"
    agent = _agent(_genome(), age=10)
    agent.self_regulation_state = apply_self_regulation_controls(
        SelfRegulationRuntimeState(),
        [1.0] * len(implemented_self_regulation_control_ids()),
        genome=agent.genome,
        sim_cfg=sim_cfg,
    )
    agents = [agent, _agent(_genome(self_regulation_control_mask=1), age=20)]

    a = compute_cognitive_status_metrics(agents, env_cfg=env_cfg, sim_cfg=sim_cfg)
    b = compute_cognitive_status_metrics(agents, env_cfg=env_cfg, sim_cfg=sim_cfg)
    assert a == b
    assert a["self_regulation_enabled_fraction"] == 1.0
    assert a["self_regulation_control_exploration_pressure_adjustment_enabled_fraction"] == 1.0
    assert a["self_regulation_force_all_active_flag"] == 1.0
    assert a["self_regulation_pressure_scenario_flag"] == 1.0
    assert a["self_regulation_pressure_hunger_fear_conflict_flag"] == 1.0
    assert a["direct_self_regulation_reward_present_flag"] == 0.0
    assert a["self_modification_present_flag"] == 0.0
    assert a["genome_editing_present_flag"] == 0.0
    assert a["self_regulation_control_application_count_total"] == 1.0
