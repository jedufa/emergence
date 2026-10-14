from __future__ import annotations

import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from agents.developmental_program_v1 import canonicalize_developmental_program
from agents.genome import Genome
from agents.perception import (
    SENSOR_RADIUS_LADDER,
    apply_perception_masks,
    implemented_sensor_channel_mask,
    minimal_sensor_channel_mask,
    mutate_sensor_channel_mask,
    mutate_sensor_radius,
    perception_channel_ids,
    perception_channel_registry,
    recombine_sensor_channel_mask,
    recombine_sensor_radius,
)
from agents.cognition import (
    cognitive_trait_ids,
    compute_cognitive_budget_ledger,
    express_cognitive_phenotype,
    get_cognitive_trait,
)
from env.env_config import EnvConfig
from simulation.metrics_pkg.cognitive_status import compute_cognitive_status_metrics
from simulation.runner_pkg.runner_config import SimConfig


def _genome(
    *,
    sensor_channel_mask: int | None = None,
    sensor_radius: int = 2,
    memory_dim: int = 0,
) -> Genome:
    return Genome(
        hidden_size=64,
        exploration_bias=0.1,
        aggression_bias=0.2,
        reproduction_threshold=30.0,
        mutation_rate=0.03,
        action_temperature=1.0,
        use_gru=bool(memory_dim > 0),
        memory_dim=int(memory_dim),
        sensor_channel_mask=implemented_sensor_channel_mask() if sensor_channel_mask is None else int(sensor_channel_mask),
        sensor_radius=int(sensor_radius),
        drive_inertia=0.85,
        drive_noise=0.02,
        drive_logit_gain=1.1,
        pred_lr=0.08,
        developmental_program=canonicalize_developmental_program(None),
    )


def _agent(agent_id: int, genome: Genome) -> SimpleNamespace:
    return SimpleNamespace(
        agent_id=int(agent_id),
        genome=genome,
        energy=100.0,
        alive=True,
        drives=np.zeros((4,), dtype=np.float32),
        stats=SimpleNamespace(
            survival_steps=10,
            pred_err_count=0,
            pred_err_sum=0.0,
            offspring_count=0,
            patch_habitat_lineage_id=int(agent_id),
        ),
    )


def test_perception_channel_registry_stable_and_future_reserved_honest() -> None:
    ids = perception_channel_ids()
    assert ids[:3] == ("local_grid_state", "patch_food_amount_norm", "patch_cover")
    assert "self_energy" in ids
    assert "self_health" in ids
    future = [c for c in perception_channel_registry() if c.future_reserved]
    assert future
    assert all(c.active_now is False for c in future)


def test_sensor_traits_registered_with_honest_metadata() -> None:
    ids = cognitive_trait_ids()
    for trait_id in (
        "sensor_channel_mask",
        "active_sensor_channel_count",
        "perception_cost",
        "perception_capacity_component",
        "sensor_radius",
        "sensor_radius_cost",
        "spatial_perception_capacity",
        "perception_cell_count",
        "cell_channel_perception_capacity",
    ):
        assert trait_id in ids
    assert get_cognitive_trait("sensor_channel_mask").directly_mutable is True
    assert get_cognitive_trait("sensor_radius").allowed_ladder == SENSOR_RADIUS_LADDER
    assert get_cognitive_trait("policy_memory_enabled").directly_mutable is False
    assert get_cognitive_trait("module_count").structurally_mutable is True


def test_sensor_genes_serialize_recombine_and_mutate_deterministically() -> None:
    g = _genome(sensor_channel_mask=minimal_sensor_channel_mask(), sensor_radius=1)
    round_trip = Genome.from_dict(g.to_dict())
    assert round_trip.sensor_channel_mask == minimal_sensor_channel_mask()
    assert round_trip.sensor_radius == 1
    assert Genome.from_dict(g.to_dict()).to_dict() == round_trip.to_dict()

    rng_a = random.Random(123)
    rng_b = random.Random(123)
    parent_a = minimal_sensor_channel_mask()
    parent_b = implemented_sensor_channel_mask()
    assert recombine_sensor_channel_mask(parent_a, parent_b, rng_a) == recombine_sensor_channel_mask(parent_a, parent_b, rng_b)

    rng_c = random.Random(456)
    rng_d = random.Random(456)
    assert mutate_sensor_channel_mask(parent_a, rng_c, 1.0) == mutate_sensor_channel_mask(parent_a, rng_d, 1.0)

    rng_e = random.Random(789)
    rng_f = random.Random(789)
    assert recombine_sensor_radius(1, 3, rng_e) == recombine_sensor_radius(1, 3, rng_f)

    rng_g = random.Random(999)
    rng_h = random.Random(999)
    assert mutate_sensor_radius(2, rng_g, 1.0) == mutate_sensor_radius(2, rng_h, 1.0)
    assert mutate_sensor_radius(3, random.Random(5), 1.0) in SENSOR_RADIUS_LADDER


def test_phenotype_and_budget_reflect_channel_and_radius_capacity() -> None:
    env_cfg = EnvConfig(width=8, height=8, view_radius=3)
    sim_cfg = SimConfig(seed=1)
    sim_cfg.enable_evolvable_sensor_channel_mask = True
    sim_cfg.enable_evolvable_sensor_radius = True

    low = express_cognitive_phenotype(
        _genome(sensor_channel_mask=minimal_sensor_channel_mask(), sensor_radius=1),
        env_cfg=env_cfg,
        sim_cfg=sim_cfg,
    )
    high = express_cognitive_phenotype(
        _genome(sensor_channel_mask=implemented_sensor_channel_mask(), sensor_radius=3),
        env_cfg=env_cfg,
        sim_cfg=sim_cfg,
    )
    assert high.active_sensor_channel_count > low.active_sensor_channel_count
    assert high.active_observed_cell_count > low.active_observed_cell_count
    assert high.cell_channel_capacity > low.cell_channel_capacity

    low_ledger = compute_cognitive_budget_ledger(low, env_cfg=env_cfg, sim_cfg=sim_cfg)
    high_ledger = compute_cognitive_budget_ledger(high, env_cfg=env_cfg, sim_cfg=sim_cfg)
    assert high_ledger.perception_cost_hypothetical > low_ledger.perception_cost_hypothetical
    assert high_ledger.perception_cost_charged == 0.0
    assert high_ledger.charged_energy_delta == 0.0

    sim_cfg.enable_experimental_perception_cost_charging = True
    sim_cfg.enable_experimental_sensor_radius_cost_charging = True
    charged = compute_cognitive_budget_ledger(high, env_cfg=env_cfg, sim_cfg=sim_cfg)
    assert charged.perception_cost_charged > 0.0


def test_observation_channel_and_radius_masks_preserve_shape_and_mask_intended_values() -> None:
    env_cfg = EnvConfig(width=8, height=8, view_radius=2)
    sim_cfg = SimConfig(seed=2)
    sim_cfg.enable_evolvable_sensor_channel_mask = True
    sim_cfg.enable_evolvable_sensor_radius = True

    agent = _agent(1, _genome(sensor_channel_mask=minimal_sensor_channel_mask(), sensor_radius=1))
    local = np.arange(25, dtype=np.float32).reshape(1, 25)
    energy = np.asarray([0.75], dtype=np.float32)
    health = np.asarray([0.50], dtype=np.float32)
    before_shape = tuple(local.shape)

    apply_perception_masks(
        local=local,
        energy_norm=energy,
        health_norm=health,
        agents=[agent],
        env_cfg=env_cfg,
        sim_cfg=sim_cfg,
    )

    assert tuple(local.shape) == before_shape
    assert energy[0] == 0.0
    assert health[0] == 0.0
    assert local.reshape(5, 5)[2, 2] == 12.0
    assert local.reshape(5, 5)[0, 0] == 0.0
    assert local.reshape(5, 5)[4, 4] == 0.0


def test_perception_metrics_are_deterministic_and_include_ablation_flags() -> None:
    env_cfg = EnvConfig(width=8, height=8, view_radius=3)
    sim_cfg = SimConfig(seed=3)
    sim_cfg.enable_evolvable_sensor_channel_mask = True
    sim_cfg.enable_evolvable_sensor_radius = True
    sim_cfg.force_minimal_sensor_channels_enabled = True
    sim_cfg.force_min_sensor_radius = True

    agents = [
        _agent(1, _genome(sensor_channel_mask=implemented_sensor_channel_mask(), sensor_radius=3)),
        _agent(2, _genome(sensor_channel_mask=minimal_sensor_channel_mask(), sensor_radius=1)),
    ]
    a = compute_cognitive_status_metrics(agents, selected_parent_agents=agents[:1], env_cfg=env_cfg, sim_cfg=sim_cfg)
    b = compute_cognitive_status_metrics(agents, selected_parent_agents=agents[:1], env_cfg=env_cfg, sim_cfg=sim_cfg)
    assert a == b
    assert a["sensor_channel_active_count_mean"] == 1.0
    assert a["sensor_radius_mean"] == 1.0
    assert a["sensor_channel_force_minimal_active_flag"] == 1.0
    assert a["sensor_radius_force_min_active_flag"] == 1.0
    assert "direct_internal_selection_pressure_present_flag" in a
    assert a["direct_cognition_reward_present_flag"] == a["direct_internal_selection_pressure_present_flag"]

