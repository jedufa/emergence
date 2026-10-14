from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from env.env_config import EnvConfig
from evolution.population_pkg.population_config import PopConfig
from simulation.metrics import MetricsLogger
from simulation.performance_profiles import FAST_EXPLORATORY_PROFILE_V2, apply_performance_profile
from simulation.runner import SimulationRunner
from simulation.runner_pkg.runner_config import SimConfig
from simulation.runnercore.tick_loop import RunnerCore1UnsupportedError, classify_support


def _env_cfg(*, no_churn: bool) -> EnvConfig:
    cfg = EnvConfig(width=6, height=6, view_radius=1)
    cfg.enable_biomes = False
    cfg.food_spawn_per_step = 4
    cfg.max_food = 24
    cfg.patch_world_enable = False
    cfg.patch_world_observation_enable = False
    cfg.prediction_demand_enable = False
    cfg.prediction_demand_observation_enable = False
    if no_churn:
        cfg.reproduce_try_prob = 0.0
    return cfg


def _sim_cfg(*, runnercore1: bool, steps: int = 4) -> SimConfig:
    cfg = SimConfig(
        seed=4242,
        generations=1,
        steps_per_generation=int(steps),
        device="cpu",
        bank_device="cpu",
        behavior_log_enable=False,
        policy_deterministic_actions=True,
        policy_tick_stride=2,
    )
    apply_performance_profile(cfg, FAST_EXPLORATORY_PROFILE_V2)
    cfg.phase_e_e1_enable = False
    cfg.phase_e_e2_enable = False
    cfg.epi_enable = False
    cfg.enable_cognitive_budget_ledger = False
    cfg.chemistry_functional_coupling_enable = False
    cfg.checkpoint_bundle_v2_period_gen = 0
    cfg.maintenance_enable = False
    cfg.perf_aggperf3_timing_buckets_enable = True
    cfg.runnercore1_array_tick_authority_enable = bool(runnercore1)
    cfg.runnercore1_candidate_enable = bool(runnercore1)
    cfg.runnercore1_authoritative_enable = False
    cfg.runnercore1_native_enable = False
    return cfg


def _prepare_sim(tmp_path: Path, *, runnercore1: bool, no_churn: bool = True) -> tuple[SimulationRunner, MetricsLogger]:
    logger = MetricsLogger(str(tmp_path))
    sim = SimulationRunner(
        env_cfg=_env_cfg(no_churn=no_churn),
        pop_cfg=PopConfig(population_size=6),
        sim_cfg=_sim_cfg(runnercore1=runnercore1),
        logger=logger,
        run_dir=str(tmp_path),
        perf=True,
    )
    return sim, logger


def _fingerprint(sim: SimulationRunner) -> dict[str, object]:
    alive = sorted(sim.pop.alive_agents(), key=lambda a: int(a.agent_id))
    ids = [int(a.agent_id) for a in alive]
    hmem_obj = getattr(sim, "_Hmem", None)
    if hmem_obj is None:
        hmem = []
    else:
        hmem = np.asarray(hmem_obj.detach().cpu().numpy() if hasattr(hmem_obj, "detach") else hmem_obj).round(7).tolist()
    return {
        "alive_ids": ids,
        "energy": [round(float(a.energy), 7) for a in alive],
        "health": [round(float(a.health), 7) for a in alive],
        "injury": [round(float(getattr(a, "injury", 0.0)), 7) for a in alive],
        "scar": [round(float(getattr(a, "scar", 0.0)), 7) for a in alive],
        "pos_y": np.asarray(sim.env.agent_pos_y[ids], dtype=np.int32).tolist() if ids else [],
        "pos_x": np.asarray(sim.env.agent_pos_x[ids], dtype=np.int32).tolist() if ids else [],
        "hmem": hmem,
    }


def _run_tick_loop(sim: SimulationRunner) -> tuple[int, int, int, float]:
    if not list(sim.pop.alive_agents()):
        sim.pop.init_random()
        sim._ensure_bank()
        sim._place_agents()
        if hasattr(sim, "_mark_bank_dirty"):
            sim._mark_bank_dirty()
    for agent in sim.pop.alive_agents():
        agent.genome.reproduction_threshold = 1.0e9
    sync_fn = getattr(sim, "_phase3_pre_start_generation_sync", None)
    if callable(sync_fn):
        sync_fn(list(sim.pop.alive_agents()))
    return sim._run_ticks(
        gen=0,
        buckets={},
        injury_recover_mult=1.0,
        deaths_this_gen=0,
        successful_births=0,
        failed_births=0,
        gen_peak_injury=0.0,
        rest_streaks={},
    )


def test_runnercore1_defaults_off() -> None:
    cfg = SimConfig()

    assert cfg.runnercore1_array_tick_authority_enable is False
    assert cfg.runnercore1_candidate_enable is False
    assert cfg.runnercore1_authoritative_enable is False
    assert cfg.runnercore1_native_enable is False


def test_runnercore1_fails_closed_when_churn_or_maintenance_active(tmp_path: Path) -> None:
    sim, _logger = _prepare_sim(tmp_path, runnercore1=True, no_churn=False)
    sim.sim_cfg.maintenance_enable = True
    report = classify_support(sim)

    assert report["supported"] is False
    assert "birth_refill_churn_not_array_authoritative" in report["blockers"]
    assert "runnercore19_maintenance_arrays_required" in report["blockers"]
    with pytest.raises(RunnerCore1UnsupportedError, match="RUNNERCORE1-C/BLOCKED-BY-OWNERSHIP"):
        sim.run()
    assert (tmp_path / "metrics" / "runnercore1_array_tick_authority_summary.json").exists()


def test_runnercore1_no_churn_candidate_matches_tiny_live_fingerprint_without_legacy_actions(tmp_path: Path) -> None:
    legacy, _legacy_logger = _prepare_sim(tmp_path / "legacy", runnercore1=False)
    legacy_result = _run_tick_loop(legacy)
    legacy_fp = _fingerprint(legacy)

    candidate, _candidate_logger = _prepare_sim(tmp_path / "candidate", runnercore1=True)

    def _forbidden(*_args, **_kwargs):
        raise AssertionError("legacy action materializer called")

    candidate._batched_actions = _forbidden
    candidate._batched_actions_k = _forbidden
    candidate._batched_actions_strided = _forbidden
    candidate_result = _run_tick_loop(candidate)
    candidate_fp = _fingerprint(candidate)

    assert candidate_result == legacy_result
    assert candidate_fp == legacy_fp
    summary = candidate._runnercore1_array_tick_authority_summary
    assert summary["candidate_ran"] is True
    assert summary["fallback_counters"]["legacy_batched_actions_calls"] == 0
    assert summary["fallback_counters"]["legacy_batched_actions_k_calls"] == 0
    assert summary["fallback_counters"]["legacy_batched_actions_strided_calls"] == 0
    assert summary["fallback_counters"]["per_tick_agent_sync_rows"] == 0
    assert summary["state_summary"]["ticks_ran"] == 4
