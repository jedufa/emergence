from __future__ import annotations

import random
from types import SimpleNamespace

import numpy as np

from agents.cognition import compute_cognitive_budget_ledger, express_cognitive_phenotype, get_cognitive_trait
from agents.developmental_program_v1 import canonicalize_developmental_program
from agents.directed_inheritance import (
    DirectedInheritancePayload,
    apply_directed_inheritance_commands,
    directed_inheritance_channel_registry,
    directed_inheritance_payload_summary,
    directed_inheritance_registry_dict,
    forbidden_directed_inheritance_targets,
    implemented_directed_inheritance_channel_ids,
    implemented_directed_inheritance_channel_mask,
    mutate_directed_inheritance_channel_mask,
    recombine_directed_inheritance_channel_mask,
    reset_directed_inheritance_payload,
    sanitize_directed_inheritance_channel_mask,
)
from agents.genome import Genome
from agents.introspection import implemented_introspection_channel_mask
from agents.perception import implemented_sensor_channel_mask
from agents.self_modification import implemented_self_modification_target_mask
from agents.self_regulation import implemented_self_regulation_control_mask
from env.env_config import EnvConfig
from evolution.reproduction import crossover_genome
from simulation.metrics_pkg.cognitive_status import compute_cognitive_status_metrics
from simulation.runner_pkg.runner_config import SimConfig


def _genome(*, directed_inheritance_channel_mask: int | None = None) -> Genome:
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
        directed_inheritance_channel_mask=implemented_directed_inheritance_channel_mask()
        if directed_inheritance_channel_mask is None
        else int(directed_inheritance_channel_mask),
        drive_inertia=0.85,
        drive_noise=0.02,
        drive_logit_gain=1.1,
        pred_lr=0.2,
        developmental_program=canonicalize_developmental_program(None),
    )


def _agent(genome: Genome, *, payload: DirectedInheritancePayload | None = None) -> SimpleNamespace:
    stats = SimpleNamespace(survival_steps=4, pred_err_count=1, pred_err_sum=0.2, prediction_update_count=1, offspring_count=0, patch_habitat_lineage_id=1)
    return SimpleNamespace(agent_id=1, genome=genome, energy=80.0, health=9.0, alive=True, drives=np.asarray([0.8, 0.2, 0.4, 0.6]), stats=stats, directed_inheritance_payload=payload)


def _cfg(*, enabled: bool = True) -> SimConfig:
    cfg = SimConfig(seed=7)
    cfg.enable_directed_inheritance = bool(enabled)
    cfg.enable_directed_inheritance_surface = bool(enabled)
    cfg.enable_evolvable_directed_inheritance_channels = True
    cfg.enable_directed_inheritance_channel_recombination = True
    cfg.enable_directed_inheritance_channel_mutation = True
    cfg.genome_c1_enable = False
    cfg.genome_c2_enable = False
    return cfg


def test_stage8_registry_is_stable_safe_and_forbidden_targets_are_excluded() -> None:
    channels = directed_inheritance_channel_registry()
    ids = [c.influence_id for c in channels]
    assert ids[:6] == [
        "offspring_developmental_expression_bias",
        "offspring_juvenile_cognition_scale_bias",
        "offspring_memory_expression_bias",
        "offspring_prediction_lr_bias",
        "offspring_exploration_bias_marker",
        "offspring_reproduction_threshold_bias",
    ]
    assert implemented_directed_inheritance_channel_ids() == tuple(c.influence_id for c in channels if c.active_now and not c.future_reserved)
    forbidden = set(forbidden_directed_inheritance_targets())
    for expected in (
        "genome_direct_edit",
        "mutation_rate_edit",
        "recombination_rule_edit",
        "source_code",
        "simulator_code",
        "filesystem",
        "network",
        "rng_state",
        "checkpoint_internals",
        "direct_reward_set",
        "direct_fitness_set",
        "arbitrary_child_energy_set",
        "arbitrary_child_health_set",
        "arbitrary_child_position_set",
        "arbitrary_child_age_set",
        "direct_reproduction_success_set",
        "direct_death_state_set",
        "host_state",
    ):
        assert expected in forbidden
    for channel in channels:
        if channel.active_now:
            assert channel.safe_for_directed_inheritance is True
            assert channel.can_affect_genome is False
            assert channel.can_affect_mutation_rate is False
            assert channel.can_affect_recombination is False
            assert channel.can_affect_host_or_simulator is False
            assert channel.can_affect_source_code is False
            assert channel.target_offspring_state not in forbidden
    assert all(not c.active_now for c in channels if c.future_reserved)
    assert directed_inheritance_registry_dict()["schema_version"] == "directed_inheritance_registry_v1"


def test_stage8_traits_and_default_behavior_are_honest() -> None:
    trait = get_cognitive_trait("directed_inheritance_channel_mask")
    assert trait.heritable is True
    assert trait.directly_mutable is True
    assert trait.recombinable is True
    assert trait.control_ready is True
    payload_trait = get_cognitive_trait("non_genomic_inheritance_payload")
    assert payload_trait.phenotype_surface_present is True
    assert payload_trait.heritable is False
    phenotype = express_cognitive_phenotype(_genome(), env_cfg=EnvConfig(width=8, height=8, view_radius=2), sim_cfg=_cfg(enabled=False))
    assert phenotype.directed_inheritance_enabled is False
    assert phenotype.active_directed_inheritance_channel_count == 0
    assert phenotype.genome_editing_active is False
    assert phenotype.mutation_rate_control_active is False
    assert phenotype.recombination_control_active is False


def test_stage8_mask_serialization_recombination_and_mutation_are_deterministic() -> None:
    mask = implemented_directed_inheritance_channel_mask()
    genome = _genome(directed_inheritance_channel_mask=mask)
    restored = Genome.from_dict(genome.to_dict())
    assert restored.directed_inheritance_channel_mask == mask
    old_payload = genome.to_dict()
    old_payload.pop("directed_inheritance_channel_mask", None)
    assert Genome.from_dict(old_payload).directed_inheritance_channel_mask == implemented_directed_inheritance_channel_mask()
    assert recombine_directed_inheritance_channel_mask(1, mask, random.Random(7)) == recombine_directed_inheritance_channel_mask(1, mask, random.Random(7))
    assert mutate_directed_inheritance_channel_mask(1, random.Random(9), 1.0) == mutate_directed_inheritance_channel_mask(1, random.Random(9), 1.0)
    assert sanitize_directed_inheritance_channel_mask(1 << 30) == 0


def test_stage8_reproduction_mask_is_gated_and_deterministic() -> None:
    g1 = _genome(directed_inheritance_channel_mask=1)
    g2 = _genome(directed_inheritance_channel_mask=implemented_directed_inheritance_channel_mask())
    disabled = _cfg(enabled=False)
    disabled.enable_evolvable_directed_inheritance_channels = False
    assert crossover_genome(g1, g2, random.Random(1), sim_cfg=disabled).directed_inheritance_channel_mask == 1
    enabled = _cfg(enabled=True)
    a = crossover_genome(g1, g2, random.Random(11), sim_cfg=enabled)
    b = crossover_genome(g1, g2, random.Random(11), sim_cfg=enabled)
    assert a.directed_inheritance_channel_mask == b.directed_inheritance_channel_mask


def test_stage8_payload_application_is_bounded_non_genomic_and_ablatable() -> None:
    cfg = _cfg(enabled=True)
    cfg.default_directed_inheritance_channels_enabled = True
    cfg.force_all_directed_inheritance_channels = True
    parent = _genome()
    child = _genome()
    original_parent = parent.to_dict()
    original_child = child.to_dict()
    payload = apply_directed_inheritance_commands(None, [1.0, -1.0, 10.0, float("nan"), 0.5, -0.5], parent_genome=parent, offspring_genome=child, sim_cfg=cfg)
    assert parent.to_dict() == original_parent
    assert child.to_dict() == original_child
    assert payload.offspring_developmental_bias_offsets["offspring_developmental_expression_bias"] == 0.05
    assert payload.offspring_developmental_bias_offsets["offspring_juvenile_cognition_scale_bias"] == -0.05
    assert payload.offspring_phenotype_bias_offsets["offspring_prediction_lr_bias"] == 0.0
    assert payload.influence_event_count == 1
    assert DirectedInheritancePayload.from_dict(payload.to_dict()).to_dict() == payload.to_dict()

    cfg.mask_directed_inheritance_commands_to_neutral = True
    neutral = apply_directed_inheritance_commands(None, [1.0] * len(implemented_directed_inheritance_channel_ids()), parent_genome=parent, offspring_genome=child, sim_cfg=cfg)
    assert neutral.payload_applied is False
    cfg.mask_directed_inheritance_commands_to_neutral = False
    cfg.freeze_directed_inheritance_payload = True
    frozen = apply_directed_inheritance_commands(None, [1.0] * len(implemented_directed_inheritance_channel_ids()), parent_genome=parent, offspring_genome=child, sim_cfg=cfg)
    assert frozen.payload_frozen is True
    assert frozen.payload_applied is False
    assert reset_directed_inheritance_payload(payload).payload_applied is False


def test_stage8_phenotype_budget_metrics_and_guardrails_are_deterministic() -> None:
    cfg = _cfg(enabled=True)
    cfg.default_directed_inheritance_channels_enabled = True
    cfg.force_all_directed_inheritance_channels = True
    cfg.directed_inheritance_pressure_scenario = True
    cfg.directed_inheritance_pressure_type = "developmental_bias"
    parent = _genome()
    payload = apply_directed_inheritance_commands(
        None,
        [1.0] * len(implemented_directed_inheritance_channel_ids()),
        parent_genome=parent,
        offspring_genome=_genome(),
        sim_cfg=cfg,
    )
    agent = _agent(parent, payload=payload)
    env_cfg = EnvConfig(width=8, height=8, view_radius=2)
    phenotype = express_cognitive_phenotype(parent, agent=agent, env_cfg=env_cfg, sim_cfg=cfg)
    ledger = compute_cognitive_budget_ledger(phenotype, agent=agent, env_cfg=env_cfg, sim_cfg=cfg)
    assert phenotype.active_directed_inheritance_channel_count == len(implemented_directed_inheritance_channel_ids())
    assert phenotype.non_genomic_payload_applied is True
    assert phenotype.genome_editing_active is False
    assert phenotype.inherited_self_modification_state_active is False
    assert ledger.directed_inheritance_command_cost_hypothetical > 0.0
    assert ledger.directed_inheritance_command_cost_charged == 0.0
    cfg.enable_experimental_directed_inheritance_cost_charging = True
    charged = compute_cognitive_budget_ledger(express_cognitive_phenotype(parent, agent=agent, env_cfg=env_cfg, sim_cfg=cfg), agent=agent, env_cfg=env_cfg, sim_cfg=cfg)
    assert charged.directed_inheritance_command_cost_charged > 0.0
    a = compute_cognitive_status_metrics([agent], env_cfg=env_cfg, sim_cfg=cfg)
    b = compute_cognitive_status_metrics([agent], env_cfg=env_cfg, sim_cfg=cfg)
    assert a == b
    assert a["directed_inheritance_enabled_fraction"] == 1.0
    assert a["directed_inheritance_channel_offspring_developmental_expression_bias_enabled_fraction"] == 1.0
    assert a["directed_inheritance_force_all_active_flag"] == 1.0
    assert a["directed_inheritance_pressure_developmental_bias_flag"] == 1.0
    assert a["direct_directed_inheritance_reward_present_flag"] == 0.0
    assert a["mutation_rate_control_present_flag"] == 0.0
    assert a["recombination_control_present_flag"] == 0.0
    assert directed_inheritance_payload_summary(payload)["event_count"] == 1.0
