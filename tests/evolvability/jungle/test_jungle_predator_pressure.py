from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from env.env_config import EnvConfig
from env.gridworld import GridWorld
from env.gridworld_pkg.jungle import (
    compute_terrain_damage_multiplier_map,
    compute_predator_risk_maps,
    jungle_reproduction_emissions_enabled,
    jungle_predator_pressure_enabled,
    jungle_predator_risk_metrics,
    jungle_terrain_protection_enabled,
    queue_jungle_birth_emission,
)
from env.gridworld_pkg.patch_world import PATCH_WORLD_OBSERVATION_CHANNEL_NAMES
from evolution.population_pkg.population_config import PopConfig
from simulation.checkpoint_bundle_v2 import (
    checkpoint_bundle_v2_apply,
    checkpoint_bundle_v2_load,
    checkpoint_bundle_v2_save,
)
from simulation.metrics_pkg.noop_loggers import NoOpMetricsLogger
from simulation.rng_manager import RNGManager
from simulation.runner import SimulationRunner
from simulation.runner_pkg.runner_config import SimConfig


def _env_cfg(*, patch_world_enable: bool = True, predator_enable: bool = False) -> EnvConfig:
    cfg = EnvConfig(width=5, height=5, view_radius=1)
    cfg.enable_biomes = False
    cfg.food_spawn_per_step = 0
    cfg.max_food = 0
    cfg.patch_world_enable = bool(patch_world_enable)
    cfg.jungle_predator_pressure_enable = bool(predator_enable)
    cfg.jungle_predator_base_risk = 0.0
    cfg.jungle_predator_scent_weight = 0.0
    cfg.jungle_predator_waste_weight = 0.0
    cfg.jungle_predator_density_weight = 0.0
    cfg.jungle_predator_visibility_weight = 0.0
    cfg.jungle_predator_cover_protection_weight = 0.0
    cfg.jungle_predator_shelter_protection_weight = 0.0
    cfg.jungle_predator_risk_min = 0.0
    cfg.jungle_predator_risk_cap = 1.0
    cfg.jungle_predator_health_damage = 1.5
    return cfg


def _sim_cfg(seed: int = 123) -> SimConfig:
    cfg = SimConfig(
        seed=int(seed),
        generations=1,
        steps_per_generation=1,
        device="cpu",
        bank_device="cpu",
        behavior_log_enable=False,
        policy_deterministic_actions=True,
        policy_tick_stride=1,
    )
    cfg.phase_e_e1_enable = False
    cfg.epi_enable = False
    cfg.checkpoint_bundle_v2_period_gen = 0
    return cfg


def _make_env(cfg: EnvConfig) -> GridWorld:
    rm = RNGManager(global_seed=11)
    return GridWorld(cfg, seed=11, run_dir=None, rng=rm.env_np)


def _make_sim(tmp_path: Path, cfg: EnvConfig) -> SimulationRunner:
    return SimulationRunner(
        env_cfg=cfg,
        pop_cfg=PopConfig(population_size=1),
        sim_cfg=_sim_cfg(),
        logger=NoOpMetricsLogger(),
        run_dir=str(tmp_path),
        perf=False,
    )


def _init_birth_counters(sim: SimulationRunner) -> None:
    for name in (
        "_birth_attempts_gen",
        "_birth_pre_cooldown_gen",
        "_birth_pre_no_energy_gen",
        "_birth_pre_no_space_gen",
        "_birth_pre_no_mate_gen",
        "_birth_attempt_conception_fail_gen",
        "_birth_gestation_fail_gen",
        "_birth_make_child_fail_gen",
        "_birth_fail_no_space_gen",
        "_birth_fail_no_mate_gen",
        "_birth_fail_buffer_gen",
        "_birth_fail_injury_gate_gen",
        "_birth_fail_scar_gate_gen",
        "_birth_fail_conception_gen",
        "_birth_fail_gestation_gen",
        "_birth_pre_injury_gate_gen",
        "_birth_precheck_no_space_gen",
        "_birth_precheck_no_mate_gen",
        "_birth_precheck_scar_gate_gen",
        "_birth_precheck_injury_gate_gen",
        "_injury_blocked_births_gen",
    ):
        setattr(sim, name, 0)


def test_jungle_predator_config_defaults_and_requires_patch_world() -> None:
    cfg = EnvConfig()
    assert cfg.jungle_predator_pressure_enable is False
    assert cfg.jungle_reproduction_emissions_enable is False
    assert cfg.jungle_juvenile_predator_vulnerability_enable is False
    assert cfg.jungle_terrain_protection_enable is False
    assert cfg.jungle_ecological_recovery_enable is False
    assert jungle_predator_pressure_enabled(cfg) is False

    invalid = _env_cfg(patch_world_enable=False, predator_enable=True)
    with pytest.raises(ValueError, match="requires patch_world_enable"):
        _make_env(invalid)

    invalid_emissions = _env_cfg(patch_world_enable=False, predator_enable=False)
    invalid_emissions.jungle_reproduction_emissions_enable = True
    with pytest.raises(ValueError, match="requires patch_world_enable"):
        _make_env(invalid_emissions)

    invalid_juvenile = _env_cfg(patch_world_enable=True, predator_enable=False)
    invalid_juvenile.jungle_juvenile_predator_vulnerability_enable = True
    with pytest.raises(ValueError, match="requires jungle_predator_pressure_enable"):
        _make_env(invalid_juvenile)

    invalid_terrain = _env_cfg(patch_world_enable=True, predator_enable=False)
    invalid_terrain.jungle_terrain_protection_enable = True
    with pytest.raises(ValueError, match="requires jungle_predator_pressure_enable"):
        _make_env(invalid_terrain)

    invalid_recovery = _env_cfg(patch_world_enable=False, predator_enable=False)
    invalid_recovery.jungle_ecological_recovery_enable = True
    with pytest.raises(ValueError, match="requires patch_world_enable"):
        _make_env(invalid_recovery)


def test_jungle_predator_risk_formula_responds_to_patch_signals() -> None:
    cfg = _env_cfg(predator_enable=True)
    env = _make_env(cfg)

    cfg.jungle_predator_scent_weight = 0.4
    cfg.jungle_predator_waste_weight = 0.3
    cfg.jungle_predator_visibility_weight = 0.2
    cfg.jungle_predator_cover_protection_weight = 0.5
    cfg.jungle_predator_shelter_protection_weight = 0.25
    env.patch_scent.fill(0.0)
    env.patch_waste.fill(0.0)
    env.patch_cover.fill(0.0)
    env.patch_shelter.fill(0.0)
    env.patch_scent[2, 2] = cfg.patch_scent_max
    env.patch_waste[2, 2] = cfg.patch_waste_max
    high_exposure = compute_predator_risk_maps(env)["risk"][2, 2]

    env.patch_cover[2, 2] = 1.0
    env.patch_shelter[2, 2] = 1.0
    protected = compute_predator_risk_maps(env)["risk"][2, 2]
    assert high_exposure > protected
    assert 0.0 <= protected <= cfg.jungle_predator_risk_cap

    density_cfg = _env_cfg(predator_enable=True)
    density_cfg.jungle_predator_density_weight = 1.0
    density_env = _make_env(density_cfg)
    density_env.add_agent(1, (2, 2))
    risk = compute_predator_risk_maps(density_env)["risk"]
    assert risk[2, 2] > risk[0, 0]
    assert np.isfinite(risk).all()


def test_jungle_predator_event_applies_health_damage_and_records_context(tmp_path: Path) -> None:
    cfg = _env_cfg(predator_enable=True)
    cfg.jungle_predator_base_risk = 1.0
    cfg.jungle_predator_visibility_weight = 1.0
    cfg.jungle_predator_risk_cap = 1.0
    cfg.jungle_predator_health_damage = 2.0
    sim = _make_sim(tmp_path, cfg)
    sim.pop.init_random()
    _init_birth_counters(sim)
    agent = sim.pop.alive_agents()[0]
    sim.env.reset()
    sim.env.add_agent(int(agent.agent_id), (2, 2))
    health_before = float(agent.health)

    deaths = sim._apply_jungle_predator_pressure(0)

    assert deaths == 0
    assert agent.alive is True
    assert agent.health == pytest.approx(health_before - 2.0)
    assert sim._jungle_predator_events_gen == 1
    assert sim._jungle_predator_damage_total_gen == pytest.approx(2.0)
    assert len(sim._jungle_predator_event_contexts_gen) == 1


def test_jungle_juvenile_predator_vulnerability_scales_damage_when_enabled(tmp_path: Path) -> None:
    cfg = _env_cfg(predator_enable=True)
    cfg.jungle_predator_base_risk = 1.0
    cfg.jungle_predator_visibility_weight = 1.0
    cfg.jungle_predator_risk_cap = 1.0
    cfg.jungle_predator_health_damage = 2.0
    cfg.jungle_juvenile_predator_vulnerability_enable = True
    cfg.jungle_juvenile_age_threshold = 5
    cfg.jungle_juvenile_predator_damage_multiplier = 2.0
    sim = _make_sim(tmp_path, cfg)
    sim.pop.init_random()
    agent = sim.pop.alive_agents()[0]
    agent.stats.survival_steps = 0
    sim.env.reset()
    sim.env.add_agent(int(agent.agent_id), (2, 2))
    health_before = float(agent.health)

    sim._apply_jungle_predator_pressure(0)

    assert agent.health == pytest.approx(health_before - 4.0)
    assert sim._jungle_juvenile_predator_events_gen == 1
    assert sim._jungle_juvenile_predator_damage_total_gen == pytest.approx(4.0)


def test_jungle_adult_predator_damage_unchanged_with_juvenile_flag(tmp_path: Path) -> None:
    cfg = _env_cfg(predator_enable=True)
    cfg.jungle_predator_visibility_weight = 1.0
    cfg.jungle_predator_risk_cap = 1.0
    cfg.jungle_predator_health_damage = 2.0
    cfg.jungle_juvenile_predator_vulnerability_enable = True
    cfg.jungle_juvenile_age_threshold = 5
    cfg.jungle_juvenile_predator_damage_multiplier = 3.0
    sim = _make_sim(tmp_path, cfg)
    sim.pop.init_random()
    agent = sim.pop.alive_agents()[0]
    agent.stats.survival_steps = 99
    sim.env.reset()
    sim.env.add_agent(int(agent.agent_id), (2, 2))
    health_before = float(agent.health)

    sim._apply_jungle_predator_pressure(0)

    assert agent.health == pytest.approx(health_before - 2.0)
    assert sim._jungle_adult_predator_events_gen == 1
    assert sim._jungle_adult_predator_damage_total_gen == pytest.approx(2.0)


def test_jungle_terrain_protection_disabled_preserves_step9_damage(tmp_path: Path) -> None:
    cfg = _env_cfg(predator_enable=True)
    cfg.jungle_predator_base_risk = 1.0
    cfg.jungle_predator_visibility_weight = 1.0
    cfg.jungle_predator_risk_cap = 1.0
    cfg.jungle_predator_health_damage = 2.0
    cfg.jungle_terrain_protection_enable = False
    sim = _make_sim(tmp_path, cfg)
    sim.pop.init_random()
    agent = sim.pop.alive_agents()[0]
    sim.env.reset()
    sim.env.add_agent(int(agent.agent_id), (2, 2))
    sim.env.patch_cover[2, 2] = np.float32(1.0)
    sim.env.patch_shelter[2, 2] = np.float32(1.0)
    health_before = float(agent.health)

    sim._apply_jungle_predator_pressure(0)

    assert jungle_terrain_protection_enabled(cfg) is False
    assert agent.health == pytest.approx(health_before - 2.0)
    assert sim._jungle_predator_damage_total_gen == pytest.approx(2.0)
    assert sim._jungle_predator_damage_prevented_total_gen == pytest.approx(0.0)
    assert sim._jungle_predator_damage_multipliers_gen == pytest.approx([1.0])


def test_jungle_terrain_protection_reduces_predator_damage_bounded(tmp_path: Path) -> None:
    cfg = _env_cfg(predator_enable=True)
    cfg.jungle_predator_base_risk = 1.0
    cfg.jungle_predator_visibility_weight = 1.0
    cfg.jungle_predator_risk_cap = 1.0
    cfg.jungle_predator_health_damage = 2.0
    cfg.jungle_terrain_protection_enable = True
    cfg.jungle_cover_damage_reduction_weight = 0.2
    cfg.jungle_shelter_damage_reduction_weight = 0.3
    cfg.jungle_terrain_damage_multiplier_min = 0.25
    sim = _make_sim(tmp_path, cfg)
    sim.pop.init_random()
    agent = sim.pop.alive_agents()[0]
    sim.env.reset()
    sim.env.add_agent(int(agent.agent_id), (2, 2))
    sim.env.patch_cover[2, 2] = np.float32(0.5)
    sim.env.patch_shelter[2, 2] = np.float32(1.0)
    health_before = float(agent.health)

    sim._apply_jungle_predator_pressure(0)

    assert jungle_terrain_protection_enabled(cfg) is True
    assert compute_terrain_damage_multiplier_map(sim.env)[2, 2] == pytest.approx(0.6)
    assert agent.health == pytest.approx(health_before - 1.2)
    assert sim._jungle_predator_damage_raw_total_gen == pytest.approx(2.0)
    assert sim._jungle_predator_damage_effective_total_gen == pytest.approx(1.2)
    assert sim._jungle_predator_damage_total_gen == pytest.approx(1.2)
    assert sim._jungle_predator_damage_prevented_total_gen == pytest.approx(0.8)
    assert sim._jungle_predator_damage_multipliers_gen == pytest.approx([0.6])
    assert sim._jungle_predator_event_contexts_gen[0]["damage_multiplier"] == pytest.approx(0.6)
    assert sim._jungle_predator_event_contexts_gen[0]["damage_prevented"] == pytest.approx(0.8)

    cfg2 = _env_cfg(predator_enable=True)
    cfg2.jungle_terrain_protection_enable = True
    cfg2.jungle_cover_damage_reduction_weight = 0.8
    cfg2.jungle_shelter_damage_reduction_weight = 0.8
    cfg2.jungle_terrain_damage_multiplier_min = 0.25
    env2 = _make_env(cfg2)
    env2.patch_cover[2, 2] = np.float32(1.0)
    env2.patch_shelter[2, 2] = np.float32(1.0)
    assert compute_terrain_damage_multiplier_map(env2)[2, 2] == pytest.approx(0.25)
    assert np.isfinite(compute_terrain_damage_multiplier_map(env2)).all()


def test_jungle_birth_emission_queue_is_gated_bounded_and_rng_free() -> None:
    cfg = _env_cfg(predator_enable=False)
    env = _make_env(cfg)
    before = repr(env.rng.bit_generator.state)
    assert jungle_reproduction_emissions_enabled(cfg) is False
    assert queue_jungle_birth_emission(env, 2, 2) == (0.0, 0.0)
    assert repr(env.rng.bit_generator.state) == before
    assert env.patch_scent_emit_pending[2, 2] == pytest.approx(0.0)
    assert env.patch_waste_emit_pending[2, 2] == pytest.approx(0.0)

    cfg.jungle_reproduction_emissions_enable = True
    cfg.jungle_birth_scent_emit = 0.5
    cfg.jungle_birth_waste_emit = 0.25
    scent, waste = queue_jungle_birth_emission(env, 2, 2)
    assert scent == pytest.approx(0.5)
    assert waste == pytest.approx(0.25)
    assert env.patch_scent_emit_pending[2, 2] == pytest.approx(0.5)
    assert env.patch_waste_emit_pending[2, 2] == pytest.approx(0.25)

    env.patch_scent[1, 1] = np.float32(0.9)
    env.patch_waste[1, 1] = np.float32(0.95)
    scent, waste = queue_jungle_birth_emission(env, 1, 1)
    assert scent == pytest.approx(0.1)
    assert waste == pytest.approx(0.05)


def test_jungle_successful_birth_queues_child_cell_emission(tmp_path: Path) -> None:
    cfg = _env_cfg(predator_enable=False)
    cfg.jungle_reproduction_emissions_enable = True
    cfg.jungle_birth_scent_emit = 0.2
    cfg.jungle_birth_waste_emit = 0.3
    cfg.reproduce_requires_mate = False
    cfg.reproduce_parent_energy_share = 0.1
    cfg.reproduce_min_energy_buffer = 0.0
    cfg.reproduce_cooldown_steps = 0
    sim = _make_sim(tmp_path, cfg)
    sim.pop.init_random()
    agent = sim.pop.alive_agents()[0]
    agent.energy = 100.0
    agent.genome.reproduction_threshold = 1.0
    sim.env.reset()
    sim.env.add_agent(int(agent.agent_id), (2, 2))
    _init_birth_counters(sim)

    successful, failed = sim._handle_births([agent], 0, 0, {})

    assert successful == 1
    assert failed == 0
    assert sim._jungle_birth_emission_count_gen == 1
    assert sim._jungle_birth_scent_emitted_total_gen == pytest.approx(0.2)
    assert sim._jungle_birth_waste_emitted_total_gen == pytest.approx(0.3)
    assert float(np.sum(sim.env.patch_scent_emit_pending)) == pytest.approx(0.2)
    assert float(np.sum(sim.env.patch_waste_emit_pending)) == pytest.approx(0.3)
    assert len(sim._jungle_birth_emission_contexts_gen) == 1


def test_jungle_predator_disabled_consumes_no_rng(tmp_path: Path) -> None:
    cfg = _env_cfg(predator_enable=False)
    sim = _make_sim(tmp_path, cfg)
    sim.pop.init_random()
    agent = sim.pop.alive_agents()[0]
    sim.env.reset()
    sim.env.add_agent(int(agent.agent_id), (2, 2))

    before = sim.rng_manager.capture_snapshot(generation_index=0)
    sim._apply_jungle_predator_pressure(0)
    after = sim.rng_manager.capture_snapshot(generation_index=0)

    assert before == after
    assert not hasattr(sim, "_jungle_predator_events_gen")


def test_jungle_predator_risk_not_exposed_in_patch_observation_channels() -> None:
    assert "predator_risk" not in PATCH_WORLD_OBSERVATION_CHANNEL_NAMES
    assert "predator_risk_map" not in PATCH_WORLD_OBSERVATION_CHANNEL_NAMES
    assert "jungle_predator_risk" not in PATCH_WORLD_OBSERVATION_CHANNEL_NAMES
    assert "birth_danger" not in PATCH_WORLD_OBSERVATION_CHANNEL_NAMES
    assert "juvenile_vulnerability" not in PATCH_WORLD_OBSERVATION_CHANNEL_NAMES
    assert "damage_multiplier" not in PATCH_WORLD_OBSERVATION_CHANNEL_NAMES
    assert "terrain_protection" not in PATCH_WORLD_OBSERVATION_CHANNEL_NAMES
    assert "patch_recovery_activity" not in PATCH_WORLD_OBSERVATION_CHANNEL_NAMES
    assert "recovery_activity" not in PATCH_WORLD_OBSERVATION_CHANNEL_NAMES


def test_jungle_predator_metrics_and_checkpoint_accumulators_roundtrip(tmp_path: Path) -> None:
    cfg = _env_cfg(predator_enable=True)
    cfg.jungle_predator_visibility_weight = 1.0
    env = _make_env(cfg)
    metrics = jungle_predator_risk_metrics(env)
    assert metrics["jungle_predator_pressure_enable"] == 1
    assert metrics["jungle_predator_risk_max"] <= cfg.jungle_predator_risk_cap
    assert metrics["jungle_predator_visibility_component_mean"] > 0.0

    sim = _make_sim(tmp_path / "src", cfg)
    sim._jungle_predator_events_gen = 3
    sim._jungle_predator_exposure_count_gen = 7
    sim._jungle_predator_damage_total_gen = 1.5
    sim._jungle_predator_damage_raw_total_gen = 2.0
    sim._jungle_predator_damage_effective_total_gen = 1.5
    sim._jungle_predator_damage_prevented_total_gen = 0.5
    sim._jungle_predator_damage_multipliers_gen = [0.75, 1.0]
    sim._jungle_predator_deaths_gen = 1
    sim._jungle_predator_event_contexts_gen = [{"scent": 0.2, "waste": 0.3, "density": 1.0}]
    sim._jungle_birth_scent_emitted_total_gen = 0.4
    sim._jungle_birth_waste_emitted_total_gen = 0.6
    sim._jungle_birth_emission_count_gen = 2
    sim._jungle_birth_emission_contexts_gen = [{"scent": 0.4, "waste": 0.6, "density": 1.0}]
    sim._jungle_juvenile_predator_events_gen = 1
    sim._jungle_juvenile_predator_damage_total_gen = 0.5
    sim._jungle_juvenile_predator_deaths_gen = 0
    sim._jungle_adult_predator_events_gen = 2
    sim._jungle_adult_predator_damage_total_gen = 1.0
    sim.env.patch_scent_emit_pending[2, 2] = np.float32(0.4)
    sim.env.patch_waste_emit_pending[2, 2] = np.float32(0.6)
    bundle = tmp_path / "bundle"
    checkpoint_bundle_v2_save(sim, str(bundle), generation_index=1, strict_replay_mode=True)
    payload = checkpoint_bundle_v2_load(str(bundle), strict=True, expected_commit_hash=None)
    assert payload["world"]["patch_world"]["pending_arrays"]["patch_scent_emit_pending"][2][2] == pytest.approx(0.4)
    assert payload["world"]["patch_world"]["pending_arrays"]["patch_waste_emit_pending"][2][2] == pytest.approx(0.6)

    restored = _make_sim(tmp_path / "dst", cfg)
    checkpoint_bundle_v2_apply(restored, payload, restore_rng=True)

    assert restored._jungle_predator_events_gen == 3
    assert restored._jungle_predator_exposure_count_gen == 7
    assert restored._jungle_predator_damage_total_gen == pytest.approx(1.5)
    assert restored._jungle_predator_damage_raw_total_gen == pytest.approx(2.0)
    assert restored._jungle_predator_damage_effective_total_gen == pytest.approx(1.5)
    assert restored._jungle_predator_damage_prevented_total_gen == pytest.approx(0.5)
    assert restored._jungle_predator_damage_multipliers_gen == [0.75, 1.0]
    assert restored._jungle_predator_deaths_gen == 1
    assert restored._jungle_predator_event_contexts_gen == [{"scent": 0.2, "waste": 0.3, "density": 1.0}]
    assert restored._jungle_birth_scent_emitted_total_gen == pytest.approx(0.4)
    assert restored._jungle_birth_waste_emitted_total_gen == pytest.approx(0.6)
    assert restored._jungle_birth_emission_count_gen == 2
    assert restored._jungle_birth_emission_contexts_gen == [{"scent": 0.4, "waste": 0.6, "density": 1.0}]
    assert restored._jungle_juvenile_predator_events_gen == 1
    assert restored._jungle_juvenile_predator_damage_total_gen == pytest.approx(0.5)
    assert restored._jungle_juvenile_predator_deaths_gen == 0
    assert restored._jungle_adult_predator_events_gen == 2
    assert restored._jungle_adult_predator_damage_total_gen == pytest.approx(1.0)
    assert restored.env.patch_scent_emit_pending[2, 2] == pytest.approx(0.4)
    assert restored.env.patch_waste_emit_pending[2, 2] == pytest.approx(0.6)
