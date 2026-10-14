from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from env.env_config import EnvConfig
from evolution.population_pkg.population_config import PopConfig
from simulation.checkpoint_bundle_v2 import checkpoint_bundle_v2_apply, checkpoint_bundle_v2_load, checkpoint_bundle_v2_save
from simulation.metrics_pkg.noop_loggers import NoOpMetricsLogger
from simulation.metrics_pkg.path1_drh0 import path1_drh0_generation_metrics
from simulation.runner import SimulationRunner
from simulation.runner_pkg.runner_config import SimConfig


def _cfg(
    *,
    mechanics: bool = True,
    observe: bool = True,
    static_control: bool = False,
    require_refuge_source: bool = True,
) -> EnvConfig:
    cfg = EnvConfig(width=5, height=5, view_radius=1)
    cfg.enable_biomes = False
    cfg.food_spawn_per_step = 0
    cfg.max_food = 0
    cfg.patch_world_enable = True
    cfg.wdl_delayed_hazard_enable = True
    cfg.wdl_delayed_hazard_damage_enable = True
    cfg.wdl_delayed_hazard_observation_enable = bool(observe)
    cfg.wdl_delayed_hazard_intensity = 0.4
    cfg.path1_drh0_reporting_enable = True
    cfg.path1_drh0_refuge_reachability_enable = True
    cfg.path1_drh0_observation_enable = bool(observe)
    cfg.path1_drh0_refuge_radius = 0
    cfg.path1_drh0_refuge_requires_shelter_or_cover = bool(require_refuge_source)
    cfg.path1_drh0_min_refuge_cells = 1
    cfg.path1_drh0_consequence_enable = bool(mechanics)
    cfg.path1_drh0_refuge_damage_reduction_enable = bool(mechanics)
    cfg.path1_drh0_static_refuge_control_enable = bool(static_control)
    cfg.path1_drh0_cue_required_for_refuge_protection = True
    cfg.path1_drh0_pre_arrival_required_for_refuge_protection = True
    cfg.path1_drh0_damage_reduction_fraction = 0.5
    cfg.path1_drh0_damage_reduction_cap = 1.0
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
    cfg.enable_cognitive_trait_registry = False
    cfg.enable_cognitive_status_metrics = False
    cfg.enable_cognitive_budget_ledger = False
    return cfg


def _make_sim(tmp_path: Path, cfg: EnvConfig, *, seed: int = 123) -> SimulationRunner:
    return SimulationRunner(
        env_cfg=cfg,
        pop_cfg=PopConfig(population_size=1),
        sim_cfg=_sim_cfg(seed=seed),
        logger=NoOpMetricsLogger(),
        run_dir=str(tmp_path),
        perf=False,
    )


def _prepare_refuge_exposure(
    tmp_path: Path,
    cfg: EnvConfig,
    *,
    cover: bool = True,
    cued: bool = True,
    seed: int = 123,
) -> tuple[SimulationRunner, object, float]:
    sim = _make_sim(tmp_path, cfg, seed=seed)
    sim.pop.init_random()
    agent = sim.pop.alive_agents()[0]
    agent_id = int(agent.agent_id)
    sim.env.reset()
    sim.env.add_agent(
        agent_id,
        (2, 2),
        slot=int(agent.slot),
        slot_token=int(agent.slot_token),
    )
    sim.env.patch_delayed_hazard_cue.fill(0.0)
    sim.env.patch_delayed_hazard_prior_cue.fill(0)
    sim.env.patch_delayed_hazard_active.fill(0.0)
    sim.env.patch_delayed_hazard_intensity.fill(0.0)
    sim.env.patch_cover.fill(0.0)
    sim.env.patch_shelter.fill(0.0)
    sim.env.patch_delayed_hazard_active[2, 2] = 1.0
    sim.env.patch_delayed_hazard_intensity[2, 2] = 0.4
    if cover:
        sim.env.patch_cover[2, 2] = 1.0
        sim.env.patch_shelter[2, 2] = 1.0
    sim._wdl_agents_cued_current_gen = {agent_id} if cued else set()
    return sim, agent, float(agent.health)


def _damage_tuple(tmp_path: Path, cfg: EnvConfig, *, cued: bool = True, cover: bool = True) -> tuple[float, ...]:
    sim, agent, before = _prepare_refuge_exposure(tmp_path, cfg, cued=cued, cover=cover)
    deaths = sim._apply_wdl_delayed_hazard_damage(0)
    return (
        float(deaths),
        round(float(before - float(agent.health)), 6),
        round(float(getattr(sim, "_wdl_delayed_hazard_damage_total_gen", 0.0)), 6),
        round(float(getattr(sim, "_path1_drh0_damage_in_refuge_gen", 0.0)), 6),
        round(float(getattr(sim, "_path1_drh0_damage_outside_refuge_gen", 0.0)), 6),
        round(float(getattr(sim, "_path1_drh0_damage_prevented_by_refuge_gen", 0.0)), 6),
        round(float(getattr(sim, "_path1_drh0_static_refuge_damage_prevented_gen", 0.0)), 6),
        round(float(getattr(sim, "_path1_drh0_cue_conditioned_damage_prevented_gen", 0.0)), 6),
    )


def test_path1_drh0_flags_off_preserve_wdl_damage(tmp_path: Path) -> None:
    sim, agent, before = _prepare_refuge_exposure(tmp_path, _cfg(mechanics=False))

    deaths = sim._apply_wdl_delayed_hazard_damage(0)

    assert deaths == 0
    assert agent.health == pytest.approx(before - 0.4)
    assert sim._wdl_delayed_hazard_damage_total_gen == pytest.approx(0.4)
    assert getattr(sim, "_path1_drh0_damage_prevented_by_refuge_gen", 0.0) == pytest.approx(0.0)


def test_path1_drh0_cue_conditioned_refuge_reduces_damage(tmp_path: Path) -> None:
    sim, agent, before = _prepare_refuge_exposure(tmp_path, _cfg(mechanics=True, observe=True), cued=True)

    deaths = sim._apply_wdl_delayed_hazard_damage(0)

    assert deaths == 0
    assert agent.health == pytest.approx(before - 0.2)
    assert sim._wdl_delayed_hazard_damage_total_gen == pytest.approx(0.2)
    assert sim._path1_drh0_damage_in_refuge_gen == pytest.approx(0.2)
    assert sim._path1_drh0_damage_prevented_by_refuge_gen == pytest.approx(0.2)
    assert sim._path1_drh0_cue_conditioned_damage_prevented_gen == pytest.approx(0.2)
    assert sim._path1_drh0_static_refuge_damage_prevented_gen == pytest.approx(0.0)


def test_path1_drh0_no_cue_or_no_refuge_source_does_not_reduce_damage(tmp_path: Path) -> None:
    no_cue = _damage_tuple(tmp_path / "no-cue", _cfg(mechanics=True, observe=True), cued=False)
    no_refuge = _damage_tuple(tmp_path / "no-refuge", _cfg(mechanics=True, observe=True), cover=False)

    assert no_cue[1:] == pytest.approx((0.4, 0.4, 0.4, 0.0, 0.0, 0.0, 0.0))
    assert no_refuge[1:] == pytest.approx((0.4, 0.4, 0.0, 0.4, 0.0, 0.0, 0.0))


def test_path1_drh0_observation_gate_blocks_cue_conditioned_reduction(tmp_path: Path) -> None:
    sim, agent, before = _prepare_refuge_exposure(tmp_path, _cfg(mechanics=True, observe=False), cued=True)

    sim._apply_wdl_delayed_hazard_damage(0)

    assert agent.health == pytest.approx(before - 0.4)
    assert sim._path1_drh0_damage_prevented_by_refuge_gen == pytest.approx(0.0)
    assert sim._path1_drh0_cue_conditioned_damage_prevented_gen == pytest.approx(0.0)


def test_path1_drh0_static_refuge_control_is_accounted_separately(tmp_path: Path) -> None:
    sim, agent, before = _prepare_refuge_exposure(
        tmp_path,
        _cfg(mechanics=True, observe=False, static_control=True),
        cued=False,
    )

    sim._apply_wdl_delayed_hazard_damage(0)

    assert agent.health == pytest.approx(before - 0.2)
    assert sim._path1_drh0_damage_prevented_by_refuge_gen == pytest.approx(0.2)
    assert sim._path1_drh0_static_refuge_damage_prevented_gen == pytest.approx(0.2)
    assert sim._path1_drh0_cue_conditioned_damage_prevented_gen == pytest.approx(0.0)


def test_path1_drh0_generation_metrics_use_mechanics_counters(tmp_path: Path) -> None:
    sim, _, _ = _prepare_refuge_exposure(tmp_path, _cfg(mechanics=True, observe=True), cued=True)
    sim._apply_wdl_delayed_hazard_damage(0)

    metrics = path1_drh0_generation_metrics(
        sim.env,
        generation_metrics={
            "path1_drh0_damage_in_refuge": sim._path1_drh0_damage_in_refuge_gen,
            "path1_drh0_damage_prevented_by_refuge": sim._path1_drh0_damage_prevented_by_refuge_gen,
            "path1_drh0_cue_conditioned_damage_prevented": sim._path1_drh0_cue_conditioned_damage_prevented_gen,
        },
    )

    assert metrics["path1_drh0_damage_in_refuge"] == pytest.approx(0.2)
    assert metrics["path1_drh0_damage_prevented_by_refuge"] == pytest.approx(0.2)
    assert metrics["path1_drh0_hazard_damage_reduction_given_pre_arrival"] == pytest.approx(0.5)
    assert metrics["path1_drh0_outcome_delta_vs_disabled_controls"] == pytest.approx(0.2)


def test_path1_drh0_reachability_failure_is_reported() -> None:
    cfg = _cfg(mechanics=False)
    cfg.path1_drh0_refuge_radius = 10
    cfg.wdl_delayed_hazard_delay_steps = 1
    shape = (5, 5)
    cue = np.zeros(shape, dtype=np.float32)
    timer = np.full(shape, -1, dtype=np.int32)
    cover = np.zeros(shape, dtype=np.float32)
    cue[0, 0] = 1.0
    timer[0, 0] = 1
    cover[4, 4] = 1.0
    env = type(
        "Env",
        (),
        {
            "cfg": cfg,
            "patch_delayed_hazard_cue": cue,
            "patch_delayed_hazard_active": np.zeros(shape, dtype=np.float32),
            "patch_delayed_hazard_prior_cue": np.zeros(shape, dtype=np.int8),
            "patch_delayed_hazard_timer": timer,
            "patch_cover": cover,
            "patch_shelter": np.zeros(shape, dtype=np.float32),
            "agent_alive": np.array([True], dtype=bool),
            "agent_pos_y": np.array([0], dtype=np.int32),
            "agent_pos_x": np.array([0], dtype=np.int32),
        },
    )()

    metrics = path1_drh0_generation_metrics(env)

    assert metrics["path1_refuge_agents_reachable_refuge_at_cue"] == pytest.approx(0.0)
    assert metrics["path1_drh0_failed_reachability_cases"] == pytest.approx(1.0)


def test_path1_drh0_damage_adjustment_is_repeatable(tmp_path: Path) -> None:
    observed = [_damage_tuple(tmp_path / f"run-{idx}", _cfg(mechanics=True, observe=True)) for idx in range(3)]
    assert observed[0] == observed[1] == observed[2]


def test_path1_drh0_checkpoint_restores_generation_counters(tmp_path: Path) -> None:
    sim, _, _ = _prepare_refuge_exposure(tmp_path / "src", _cfg(mechanics=True, observe=True), cued=True)
    sim._apply_wdl_delayed_hazard_damage(0)
    bundle = tmp_path / "checkpoint_bundles_v2" / "bundle_gen000000"
    checkpoint_bundle_v2_save(sim, str(bundle), generation_index=0, strict_replay_mode=True)

    restored = _make_sim(tmp_path / "restored", _cfg(mechanics=True, observe=True), seed=123)
    payload = checkpoint_bundle_v2_load(str(bundle), strict=True, expected_commit_hash=None)
    checkpoint_bundle_v2_apply(restored, payload, restore_rng=True)

    assert restored._path1_drh0_damage_in_refuge_gen == pytest.approx(sim._path1_drh0_damage_in_refuge_gen)
    assert restored._path1_drh0_damage_prevented_by_refuge_gen == pytest.approx(
        sim._path1_drh0_damage_prevented_by_refuge_gen
    )
    assert restored._path1_drh0_cue_conditioned_damage_prevented_gen == pytest.approx(
        sim._path1_drh0_cue_conditioned_damage_prevented_gen
    )
