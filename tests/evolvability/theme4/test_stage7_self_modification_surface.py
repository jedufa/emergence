from __future__ import annotations

import random
from types import SimpleNamespace

import numpy as np

from agents.cognition import compute_cognitive_budget_ledger, express_cognitive_phenotype, get_cognitive_trait
from agents.developmental_program_v1 import canonicalize_developmental_program
from agents.genome import Genome
from agents.introspection import implemented_introspection_channel_mask
from agents.perception import implemented_sensor_channel_mask
from agents.self_regulation import implemented_self_regulation_control_mask
from agents.self_modification import (
    SelfModificationRuntimeState,
    apply_self_modification_commands,
    forbidden_self_modification_targets,
    implemented_self_modification_target_ids,
    implemented_self_modification_target_mask,
    mutate_self_modification_target_mask,
    recombine_self_modification_target_mask,
    reset_self_modification_for_reproduction,
    sanitize_self_modification_target_mask,
    self_modification_runtime_summary,
    self_modification_target_registry,
    self_modification_target_registry_dict,
)
from env.env_config import EnvConfig
from evolution.reproduction import crossover_genome
from simulation.metrics_pkg.cognitive_status import compute_cognitive_status_metrics
from simulation.runner_pkg.runner_config import SimConfig


def _genome(*, self_modification_target_mask: int | None = None) -> Genome:
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
        self_modification_target_mask=implemented_self_modification_target_mask()
        if self_modification_target_mask is None
        else int(self_modification_target_mask),
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
    sim_cfg = SimConfig(seed=7)
    sim_cfg.enable_self_modification = bool(enabled)
    sim_cfg.default_self_modification_targets_enabled = False
    sim_cfg.enable_evolvable_self_modification_targets = True
    sim_cfg.enable_self_modification_target_recombination = True
    sim_cfg.enable_self_modification_target_mutation = True
    sim_cfg.genome_c1_enable = False
    sim_cfg.genome_c2_enable = False
    return sim_cfg


def test_stage7_target_registry_is_stable_safe_and_forbidden_targets_are_excluded() -> None:
    targets = self_modification_target_registry()
    ids = [t.target_id for t in targets]
    assert ids[:5] == [
        "exploration_bias_lifetime_offset",
        "action_temperature_lifetime_offset",
        "drive_gain_lifetime_offset",
        "reproduction_threshold_lifetime_offset",
        "prediction_lr_lifetime_offset",
    ]
    assert tuple(implemented_self_modification_target_ids()) == tuple(t.target_id for t in targets if t.active_now and not t.future_reserved)
    forbidden = set(forbidden_self_modification_targets())
    for expected in (
        "genome",
        "mutation_rates",
        "source_code",
        "simulator_code",
        "filesystem",
        "network",
        "rng_state",
        "checkpoint_internals",
        "reward_direct_set",
        "fitness_direct_set",
        "energy_direct_set",
        "health_direct_set",
        "position_direct_set",
        "age_direct_set",
        "reproduction_success_direct_set",
        "death_direct_set",
    ):
        assert expected in forbidden
    for target in targets:
        if target.active_now:
            assert target.safe_for_agent_modification is True
            assert target.can_affect_genome is False
            assert target.can_affect_host_or_simulator is False
            assert target.can_affect_source_code is False
            assert target.target_variable not in forbidden
    assert all(not t.active_now for t in targets if t.future_reserved)
    assert self_modification_target_registry_dict()["schema_version"] == "self_modification_target_registry_v1"


def test_stage7_self_modification_traits_are_registered_honestly() -> None:
    mask_trait = get_cognitive_trait("self_modification_target_mask")
    assert mask_trait.heritable is True
    assert mask_trait.directly_mutable is True
    assert mask_trait.recombinable is True
    assert mask_trait.control_ready is True
    runtime_trait = get_cognitive_trait("self_modification_runtime_state")
    assert runtime_trait.phenotype_surface_present is True
    assert runtime_trait.heritable is False
    assert runtime_trait.checkpoint_required is True


def test_stage7_default_behavior_preserves_legacy_surface() -> None:
    env_cfg = EnvConfig(width=8, height=8, view_radius=2)
    sim_cfg = _cfg(enabled=False)
    phenotype = express_cognitive_phenotype(_genome(), env_cfg=env_cfg, sim_cfg=sim_cfg)
    assert phenotype.self_modification_enabled is False
    assert phenotype.active_self_modification_target_count == 0
    assert phenotype.self_modification_policy_output_enabled is False
    assert phenotype.self_modification_inheritable is False
    assert phenotype.genome_editing_active is False


def test_stage7_mask_serialization_recombination_and_mutation_are_deterministic() -> None:
    mask = implemented_self_modification_target_mask()
    genome = _genome(self_modification_target_mask=mask)
    restored = Genome.from_dict(genome.to_dict())
    assert restored.self_modification_target_mask == mask

    old_payload = genome.to_dict()
    old_payload.pop("self_modification_target_mask", None)
    assert Genome.from_dict(old_payload).self_modification_target_mask == implemented_self_modification_target_mask()

    low = 1
    high = implemented_self_modification_target_mask()
    assert recombine_self_modification_target_mask(low, high, random.Random(7)) == recombine_self_modification_target_mask(low, high, random.Random(7))
    assert mutate_self_modification_target_mask(low, random.Random(9), 1.0) == mutate_self_modification_target_mask(low, random.Random(9), 1.0)
    assert sanitize_self_modification_target_mask(1 << 30) == 0


def test_stage7_reproduction_self_modification_mask_is_gated_and_deterministic() -> None:
    g1 = _genome(self_modification_target_mask=1)
    g2 = _genome(self_modification_target_mask=implemented_self_modification_target_mask())
    disabled = _cfg(enabled=False)
    disabled.enable_evolvable_self_modification_targets = False
    child_disabled = crossover_genome(g1, g2, random.Random(1), sim_cfg=disabled)
    assert child_disabled.self_modification_target_mask == 1

    enabled = _cfg(enabled=True)
    a = crossover_genome(g1, g2, random.Random(11), sim_cfg=enabled)
    b = crossover_genome(g1, g2, random.Random(11), sim_cfg=enabled)
    assert a.self_modification_target_mask == b.self_modification_target_mask


def test_stage7_phenotype_ablation_and_budget_charging() -> None:
    env_cfg = EnvConfig(width=8, height=8, view_radius=2)
    sim_cfg = _cfg(enabled=True)
    sim_cfg.force_all_self_modification_targets = True
    agent = _agent(_genome())
    phenotype = express_cognitive_phenotype(agent.genome, agent=agent, env_cfg=env_cfg, sim_cfg=sim_cfg)
    ledger = compute_cognitive_budget_ledger(phenotype, agent=agent, env_cfg=env_cfg, sim_cfg=sim_cfg)
    assert phenotype.active_self_modification_target_count == len(implemented_self_modification_target_ids())
    assert phenotype.active_self_modification_target_ids == implemented_self_modification_target_ids()
    assert phenotype.self_modification_inheritable is False
    assert phenotype.genome_editing_active is False
    assert ledger.self_modification_command_cost_hypothetical > 0.0
    assert ledger.self_modification_command_cost_charged == 0.0
    assert ledger.self_modification_cost_telemetry_only is True

    sim_cfg.enable_experimental_self_modification_cost_charging = True
    charged_pheno = express_cognitive_phenotype(agent.genome, agent=agent, env_cfg=env_cfg, sim_cfg=sim_cfg)
    charged = compute_cognitive_budget_ledger(charged_pheno, agent=agent, env_cfg=env_cfg, sim_cfg=sim_cfg)
    assert charged.self_modification_command_cost_charged > 0.0
    assert charged.self_modification_cost_charging_enabled is True
    assert charged.charged_energy_delta >= charged.self_modification_command_cost_charged

    sim_cfg.enable_experimental_self_modification_cost_charging = False
    sim_cfg.force_all_self_modification_targets = False
    sim_cfg.force_no_self_modification = True
    no_mod = express_cognitive_phenotype(agent.genome, agent=agent, env_cfg=env_cfg, sim_cfg=sim_cfg)
    assert no_mod.active_self_modification_target_count == 0
    assert no_mod.force_no_self_modification_active is True


def test_stage7_command_application_is_bounded_persistent_and_non_genomic() -> None:
    sim_cfg = _cfg(enabled=True)
    sim_cfg.force_all_self_modification_targets = True
    genome = _genome()
    original = genome.to_dict()
    vector = [1.0, -1.0, 10.0, -10.0, float("nan")]

    a = apply_self_modification_commands(SelfModificationRuntimeState(), vector, genome=genome, sim_cfg=sim_cfg)
    b = apply_self_modification_commands(SelfModificationRuntimeState(), vector, genome=genome, sim_cfg=sim_cfg)
    assert a.to_dict() == b.to_dict()
    assert genome.to_dict() == original
    assert a.offsets["exploration_bias_lifetime_offset"] == 0.025
    assert a.offsets["action_temperature_lifetime_offset"] == -0.025
    assert a.offsets["drive_gain_lifetime_offset"] == 0.025
    assert a.offsets["reproduction_threshold_lifetime_offset"] == -0.5
    assert a.offsets["prediction_lr_lifetime_offset"] == 0.0
    assert a.effective_modified_phenotype_values["exploration_bias_lifetime_offset"] > genome.exploration_bias
    assert a.effective_modified_phenotype_values["action_temperature_lifetime_offset"] < genome.action_temperature
    assert self_modification_runtime_summary(a)["event_count"] == 1.0

    second = apply_self_modification_commands(a, [1.0, 0.0, 0.0, 0.0, 0.0], genome=genome, sim_cfg=sim_cfg)
    assert second.offsets["exploration_bias_lifetime_offset"] > 0.025

    sim_cfg.mask_self_modification_commands_to_neutral = True
    neutral = apply_self_modification_commands(SelfModificationRuntimeState(), vector, genome=genome, sim_cfg=sim_cfg)
    assert max(abs(v) for v in neutral.offsets.values()) == 0.0

    sim_cfg.mask_self_modification_commands_to_neutral = False
    sim_cfg.freeze_self_modification_state = True
    frozen = apply_self_modification_commands(SelfModificationRuntimeState(), vector, genome=genome, sim_cfg=sim_cfg)
    assert frozen.offsets == {}

    sim_cfg.freeze_self_modification_state = False
    sim_cfg.reset_self_modification_each_tick = True
    reset = apply_self_modification_commands(second, [0.0] * len(implemented_self_modification_target_ids()), genome=genome, sim_cfg=sim_cfg)
    assert reset.modification_reset_count >= 1
    assert max(abs(v) for v in reset.offsets.values()) == 0.0


def test_stage7_reproduction_reset_does_not_mutate_or_inherit_offsets() -> None:
    sim_cfg = _cfg(enabled=True)
    sim_cfg.force_all_self_modification_targets = True
    sim_cfg.reset_self_modification_on_reproduction = True
    genome = _genome()
    state = apply_self_modification_commands(
        SelfModificationRuntimeState(),
        [1.0] * len(implemented_self_modification_target_ids()),
        genome=genome,
        sim_cfg=sim_cfg,
    )
    assert max(abs(v) for v in state.offsets.values()) > 0.0
    reset = reset_self_modification_for_reproduction(state, sim_cfg=sim_cfg)
    assert reset.offsets == {}
    child = crossover_genome(genome, _genome(self_modification_target_mask=1), random.Random(3), sim_cfg=_cfg(enabled=True))
    assert not hasattr(child, "self_modification_state")


def test_stage7_metrics_include_self_modification_status_and_guardrails() -> None:
    env_cfg = EnvConfig(width=8, height=8, view_radius=2)
    sim_cfg = _cfg(enabled=True)
    sim_cfg.force_all_self_modification_targets = True
    sim_cfg.self_modification_pressure_scenario = True
    sim_cfg.self_modification_pressure_type = "persistent_risk"
    agent = _agent(_genome(), age=10)
    agent.self_modification_state = apply_self_modification_commands(
        SelfModificationRuntimeState(),
        [1.0] * len(implemented_self_modification_target_ids()),
        genome=agent.genome,
        sim_cfg=sim_cfg,
    )
    agents = [agent, _agent(_genome(self_modification_target_mask=1), age=20)]

    a = compute_cognitive_status_metrics(agents, env_cfg=env_cfg, sim_cfg=sim_cfg)
    b = compute_cognitive_status_metrics(agents, env_cfg=env_cfg, sim_cfg=sim_cfg)
    assert a == b
    assert a["self_modification_enabled_fraction"] == 1.0
    assert a["self_modification_target_exploration_bias_lifetime_offset_enabled_fraction"] == 1.0
    assert a["self_modification_force_all_active_flag"] == 1.0
    assert a["self_modification_pressure_scenario_flag"] == 1.0
    assert a["self_modification_pressure_persistent_risk_flag"] == 1.0
    assert a["direct_self_modification_reward_present_flag"] == 0.0
    assert a["genome_editing_present_flag"] == 0.0
    assert a["host_access_present_flag"] == 0.0
    assert a["source_code_modification_present_flag"] == 0.0
    assert a["self_modification_event_count_total"] == 1.0
    assert a["self_modification_strategy_count_approx"] >= 1.0
