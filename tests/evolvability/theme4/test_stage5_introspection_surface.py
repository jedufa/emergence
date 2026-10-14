from __future__ import annotations

import random
from types import SimpleNamespace

import numpy as np

from agents.cognition import compute_cognitive_budget_ledger, express_cognitive_phenotype, get_cognitive_trait
from agents.developmental_program_v1 import canonicalize_developmental_program
from agents.genome import Genome
from agents.introspection import (
    extract_introspection_observation,
    implemented_introspection_channel_ids,
    implemented_introspection_channel_mask,
    introspection_channel_registry,
    introspection_channel_registry_dict,
    introspection_channel_value,
    mutate_introspection_channel_mask,
    recombine_introspection_channel_mask,
    sanitize_introspection_channel_mask,
)
from agents.perception import implemented_sensor_channel_mask
from env.env_config import EnvConfig
from env.gridworld_pkg.patch_world import patch_world_total_observation_size
from evolution.reproduction import crossover_genome
from simulation.metrics_pkg.cognitive_status import compute_cognitive_status_metrics
from simulation.runner_pkg.runner_config import SimConfig


def _genome(*, introspection_channel_mask: int | None = None) -> Genome:
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
        introspection_channel_mask=implemented_introspection_channel_mask()
        if introspection_channel_mask is None
        else int(introspection_channel_mask),
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
    sim_cfg = SimConfig(seed=5)
    sim_cfg.enable_introspection_channels = bool(enabled)
    sim_cfg.default_introspection_channels_enabled = False
    sim_cfg.enable_evolvable_introspection_channels = True
    sim_cfg.enable_introspection_channel_recombination = True
    sim_cfg.enable_introspection_channel_mutation = True
    sim_cfg.genome_c1_enable = False
    sim_cfg.genome_c2_enable = False
    return sim_cfg


def test_stage5_internal_channel_registry_is_stable_and_safe() -> None:
    channels = introspection_channel_registry()
    ids = [c.channel_id for c in channels]
    assert ids == sorted(ids, key=ids.index)
    assert tuple(implemented_introspection_channel_ids()) == tuple(c.channel_id for c in channels if c.active_now and not c.future_reserved)
    assert "energy" in ids
    assert "prediction_error" in ids
    assert "uncertainty" in ids
    for channel in channels:
        if channel.active_now:
            assert channel.safe_for_agent_observation is True
            assert channel.leaks_host_or_simulator_state is False
            assert "rng" not in channel.channel_id
            assert "source_code" not in channel.channel_id
            assert "file" not in channel.channel_id
    assert all(not c.active_now for c in channels if c.future_reserved)
    assert introspection_channel_registry_dict()["schema_version"] == "introspection_channel_registry_v1"


def test_stage5_introspection_traits_are_registered_honestly() -> None:
    mask_trait = get_cognitive_trait("introspection_channel_mask")
    assert mask_trait.heritable is True
    assert mask_trait.directly_mutable is True
    assert mask_trait.recombinable is True
    assert mask_trait.control_ready is False
    surface = get_cognitive_trait("introspection_observation_surface")
    assert surface.introspection_ready is True
    assert surface.control_ready is False


def test_stage5_default_behavior_preserves_legacy_observation_size() -> None:
    env_cfg = EnvConfig(width=8, height=8, view_radius=2)
    sim_cfg = _cfg(enabled=False)
    genome = _genome()
    phenotype = express_cognitive_phenotype(genome, env_cfg=env_cfg, sim_cfg=sim_cfg)

    assert phenotype.introspection_enabled is False
    assert phenotype.active_introspection_channel_count == 0
    assert phenotype.introspection_policy_input_enabled is False
    assert patch_world_total_observation_size(env_cfg) == ((2 * env_cfg.view_radius + 1) ** 2) + 2


def test_stage5_mask_serialization_recombination_and_mutation_are_deterministic() -> None:
    mask = implemented_introspection_channel_mask()
    genome = _genome(introspection_channel_mask=mask)
    restored = Genome.from_dict(genome.to_dict())
    assert restored.introspection_channel_mask == mask

    old_payload = genome.to_dict()
    old_payload.pop("introspection_channel_mask", None)
    assert Genome.from_dict(old_payload).introspection_channel_mask == implemented_introspection_channel_mask()

    low = 1
    high = implemented_introspection_channel_mask()
    assert recombine_introspection_channel_mask(low, high, random.Random(7)) == recombine_introspection_channel_mask(low, high, random.Random(7))
    assert mutate_introspection_channel_mask(low, random.Random(9), 1.0) == mutate_introspection_channel_mask(low, random.Random(9), 1.0)
    assert sanitize_introspection_channel_mask(1 << 30) == 0


def test_stage5_reproduction_introspection_mask_is_gated_and_deterministic() -> None:
    g1 = _genome(introspection_channel_mask=1)
    g2 = _genome(introspection_channel_mask=implemented_introspection_channel_mask())
    disabled = _cfg(enabled=False)
    disabled.enable_evolvable_introspection_channels = False
    child_disabled = crossover_genome(g1, g2, random.Random(1), sim_cfg=disabled)
    assert child_disabled.introspection_channel_mask == 1

    enabled = _cfg(enabled=True)
    a = crossover_genome(g1, g2, random.Random(11), sim_cfg=enabled)
    b = crossover_genome(g1, g2, random.Random(11), sim_cfg=enabled)
    assert a.introspection_channel_mask == b.introspection_channel_mask


def test_stage5_phenotype_ablation_and_fixed_shape_extraction() -> None:
    env_cfg = EnvConfig(width=8, height=8, view_radius=2)
    sim_cfg = _cfg(enabled=True)
    sim_cfg.force_all_introspection_channels = True
    agent = _agent(_genome())
    phenotype = express_cognitive_phenotype(agent.genome, agent=agent, env_cfg=env_cfg, sim_cfg=sim_cfg)
    ledger = compute_cognitive_budget_ledger(phenotype, agent=agent, env_cfg=env_cfg, sim_cfg=sim_cfg)
    vector = extract_introspection_observation(agent, env_cfg=env_cfg, sim_cfg=sim_cfg, phenotype=phenotype, ledger=ledger)

    assert phenotype.active_introspection_channel_count == len(implemented_introspection_channel_ids())
    assert phenotype.active_introspection_channel_ids == implemented_introspection_channel_ids()
    assert vector.shape == (len(implemented_introspection_channel_ids()),)
    assert np.all(vector >= 0.0)
    assert np.all(vector <= 1.0)
    assert introspection_channel_value("energy", agent=agent, env_cfg=env_cfg, sim_cfg=sim_cfg, phenotype=phenotype, ledger=ledger) > 0.0

    sim_cfg.mask_introspection_values_to_neutral = True
    neutral = extract_introspection_observation(agent, env_cfg=env_cfg, sim_cfg=sim_cfg, phenotype=phenotype, ledger=ledger)
    assert neutral.shape == vector.shape
    assert float(np.max(np.abs(neutral))) == 0.0

    sim_cfg.mask_introspection_values_to_neutral = False
    sim_cfg.force_all_introspection_channels = False
    sim_cfg.force_no_introspection_channels = True
    no_intro = express_cognitive_phenotype(agent.genome, agent=agent, env_cfg=env_cfg, sim_cfg=sim_cfg)
    assert no_intro.active_introspection_channel_count == 0
    assert no_intro.force_no_introspection_channels_active is True


def test_stage5_introspection_budget_is_telemetry_only_until_charged() -> None:
    env_cfg = EnvConfig(width=8, height=8, view_radius=2)
    sim_cfg = _cfg(enabled=True)
    sim_cfg.force_all_introspection_channels = True
    agent = _agent(_genome())
    phenotype = express_cognitive_phenotype(agent.genome, agent=agent, env_cfg=env_cfg, sim_cfg=sim_cfg)
    ledger = compute_cognitive_budget_ledger(phenotype, agent=agent, env_cfg=env_cfg, sim_cfg=sim_cfg)
    assert ledger.introspection_channel_cost_hypothetical > 0.0
    assert ledger.introspection_channel_cost_charged == 0.0
    assert ledger.introspection_cost_telemetry_only is True

    sim_cfg.enable_experimental_introspection_cost_charging = True
    charged_pheno = express_cognitive_phenotype(agent.genome, agent=agent, env_cfg=env_cfg, sim_cfg=sim_cfg)
    charged = compute_cognitive_budget_ledger(charged_pheno, agent=agent, env_cfg=env_cfg, sim_cfg=sim_cfg)
    assert charged.introspection_channel_cost_charged > 0.0
    assert charged.introspection_cost_charging_enabled is True
    assert charged.charged_energy_delta >= charged.introspection_channel_cost_charged


def test_stage5_metrics_include_introspection_status_and_guardrails() -> None:
    env_cfg = EnvConfig(width=8, height=8, view_radius=2)
    sim_cfg = _cfg(enabled=True)
    sim_cfg.force_all_introspection_channels = True
    sim_cfg.introspection_pressure_scenario = True
    sim_cfg.introspection_pressure_type = "drive_conflict"
    agents = [_agent(_genome(), age=10), _agent(_genome(introspection_channel_mask=1), age=20)]

    a = compute_cognitive_status_metrics(agents, env_cfg=env_cfg, sim_cfg=sim_cfg)
    b = compute_cognitive_status_metrics(agents, env_cfg=env_cfg, sim_cfg=sim_cfg)
    assert a == b
    assert a["introspection_enabled_fraction"] == 1.0
    assert a["introspection_channel_energy_enabled_fraction"] == 1.0
    assert a["introspection_force_all_active_flag"] == 1.0
    assert a["introspection_pressure_scenario_flag"] == 1.0
    assert a["introspection_pressure_drive_conflict_flag"] == 1.0
    assert a["direct_introspection_reward_present_flag"] == 0.0
    assert a["direct_internal_state_reward_present_flag"] == 0.0
    assert a["introspection_strategy_count_approx"] >= 1.0
