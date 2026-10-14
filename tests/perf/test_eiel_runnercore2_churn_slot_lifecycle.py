from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from env.env_config import EnvConfig
from env.gridworld_pkg.constants import ACT_MOVE_N
from evolution.population_pkg.population_config import PopConfig
from simulation.metrics import MetricsLogger
from simulation.performance_profiles import FAST_EXPLORATORY_PROFILE_V2, apply_performance_profile
from simulation.runner import SimulationRunner
from simulation.runner_pkg.runner_config import SimConfig
from simulation.runnercore.tick_loop import RunnerCore1UnsupportedError, classify_support


_BIRTH_COUNTERS = (
    "_injury_blocked_births_gen",
    "_birth_attempts_gen",
    "_birth_pre_cooldown_gen",
    "_birth_pre_no_energy_gen",
    "_birth_pre_injury_gate_gen",
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
)


def _env_cfg(*, churn: bool, mate_required: bool = False) -> EnvConfig:
    cfg = EnvConfig(width=8, height=8, view_radius=1)
    cfg.enable_biomes = False
    cfg.food_spawn_per_step = 20
    cfg.max_food = 80
    cfg.patch_world_enable = False
    cfg.patch_world_observation_enable = False
    cfg.prediction_demand_enable = False
    cfg.prediction_demand_observation_enable = False
    cfg.reproduce_requires_mate = bool(mate_required)
    cfg.reproduce_try_prob = 1.0 if churn else 0.0
    cfg.reproduce_parent_energy_share = 0.10
    cfg.reproduce_min_energy_buffer = 0.0
    cfg.reproduce_cooldown_steps = 1
    return cfg


def _sim_cfg(*, runnercore2: bool, steps: int = 5, materialize: bool = True) -> SimConfig:
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
    cfg.runnercore2_churn_slot_lifecycle_enable = bool(runnercore2)
    cfg.runnercore2_candidate_enable = bool(runnercore2)
    cfg.runnercore2_authoritative_enable = False
    cfg.runnercore2_native_enable = False
    cfg.runnercore2_allow_birth_boundary_agent_materialization = bool(materialize)
    return cfg


def _prepare_sim(
    tmp_path: Path,
    *,
    runnercore2: bool,
    churn: bool = True,
    steps: int = 5,
    mate_required: bool = False,
    materialize: bool = True,
) -> SimulationRunner:
    sim = SimulationRunner(
        env_cfg=_env_cfg(churn=churn, mate_required=mate_required),
        pop_cfg=PopConfig(population_size=4),
        sim_cfg=_sim_cfg(runnercore2=runnercore2, steps=steps, materialize=materialize),
        logger=MetricsLogger(str(tmp_path)),
        run_dir=str(tmp_path),
        perf=True,
    )
    if not list(sim.pop.alive_agents()):
        sim.pop.init_random()
        sim._ensure_bank()
        sim._place_agents()
        if hasattr(sim, "_mark_bank_dirty"):
            sim._mark_bank_dirty()
    for agent in sim.pop.alive_agents():
        agent.genome.reproduction_threshold = 1.0
        agent.energy = 120.0
        agent.repro_cooldown = 0
    sync_fn = getattr(sim, "_phase3_pre_start_generation_sync", None)
    if callable(sync_fn):
        sync_fn(list(sim.pop.alive_agents()))
    for name in _BIRTH_COUNTERS:
        setattr(sim, name, 0)
    return sim


def _fingerprint(sim: SimulationRunner) -> dict[str, object]:
    alive = sorted(sim.pop.alive_agents(), key=lambda a: int(a.agent_id))
    ids = [int(a.agent_id) for a in alive]
    return {
        "alive_ids": ids,
        "energy": [round(float(a.energy), 6) for a in alive],
        "health": [round(float(a.health), 6) for a in alive],
        "cooldown": [int(getattr(a, "repro_cooldown", 0)) for a in alive],
        "offspring": [int(getattr(getattr(a, "stats", None), "offspring_count", 0)) for a in alive],
        "slot": [int(getattr(a, "slot", -1)) for a in alive],
        "token": [int(getattr(a, "slot_token", -1)) for a in alive],
        "pos_y": np.asarray(sim.env.agent_pos_y[ids], dtype=np.int32).tolist() if ids else [],
        "pos_x": np.asarray(sim.env.agent_pos_x[ids], dtype=np.int32).tolist() if ids else [],
        "birth_counters": {name: int(getattr(sim, name, 0)) for name in _BIRTH_COUNTERS},
    }


def _run_tick_loop(sim: SimulationRunner) -> tuple[int, int, int, float, int]:
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


def test_runnercore2_defaults_off() -> None:
    cfg = SimConfig()

    assert cfg.runnercore2_churn_slot_lifecycle_enable is False
    assert cfg.runnercore2_candidate_enable is False
    assert cfg.runnercore2_authoritative_enable is False
    assert cfg.runnercore2_native_enable is False
    assert cfg.runnercore2_allow_birth_boundary_agent_materialization is False
    assert cfg.runnercore3_profile_enable is False
    assert cfg.runnercore3_candidate_enable is False
    assert cfg.runnercore3_native_enable is False
    assert cfg.runnercore3_vectorized_birth_precheck_enable is False
    assert cfg.runnercore3_fast_birth_event_boundary_enable is False
    assert cfg.runnercore3_self_recombination_fastpath_enable is False
    assert cfg.runnercore3_vectorized_action_cache_enable is False
    assert cfg.runnercore3_allow_birth_boundary_agent_materialization is False


def test_runnercore2_fails_closed_for_mate_required_or_implicit_materialization(tmp_path: Path) -> None:
    mate = _prepare_sim(tmp_path / "mate", runnercore2=True, mate_required=True)
    mate_report = classify_support(mate)
    assert mate_report["supported"] is False
    assert "runnercore2_mate_reproduction_not_array_authoritative" in mate_report["blockers"]

    implicit = _prepare_sim(tmp_path / "implicit", runnercore2=True, materialize=False)
    implicit_report = classify_support(implicit)
    assert implicit_report["supported"] is False
    assert "runnercore2_birth_boundary_agent_materialization_must_be_explicit" in implicit_report["blockers"]
    with pytest.raises(RunnerCore1UnsupportedError, match="RUNNERCORE2-C/BLOCKED-BY-OWNERSHIP"):
        _run_tick_loop(implicit)


def test_runnercore2_churn_candidate_matches_legacy_without_legacy_actions(tmp_path: Path) -> None:
    legacy = _prepare_sim(tmp_path / "legacy", runnercore2=False)
    legacy_result = _run_tick_loop(legacy)
    legacy_fp = _fingerprint(legacy)

    candidate = _prepare_sim(tmp_path / "candidate", runnercore2=True)

    def _forbidden(*_args, **_kwargs):
        raise AssertionError("legacy action materializer called")

    candidate._batched_actions = _forbidden
    candidate._batched_actions_k = _forbidden
    candidate._batched_actions_strided = _forbidden
    candidate_result = _run_tick_loop(candidate)
    candidate_fp = _fingerprint(candidate)

    assert candidate_result == legacy_result
    assert candidate_fp == legacy_fp
    summary = candidate._runnercore2_churn_slot_lifecycle_summary
    assert summary["candidate_ran"] is True
    assert summary["fallback_counters"]["legacy_batched_actions_calls"] == 0
    assert summary["fallback_counters"]["legacy_batched_actions_k_calls"] == 0
    assert summary["fallback_counters"]["legacy_batched_actions_strided_calls"] == 0
    assert summary["fallback_counters"]["per_tick_agent_sync_rows"] == 0
    assert summary["state_summary"]["birth_events_applied"] > 0
    assert summary["state_summary"]["event_boundary_agent_create_rows"] == summary["state_summary"]["birth_events_applied"]
    assert summary["state_summary"]["policy_row_attach_events"] == summary["state_summary"]["birth_events_applied"]
    assert summary["state_summary"]["action_cache_invalidate_events"] >= summary["state_summary"]["birth_events_applied"]


def test_runnercore2_death_invalidates_slot_cache(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import simulation.runnercore.tick_loop as tick_loop

    candidate = _prepare_sim(tmp_path / "candidate", runnercore2=True, churn=False, steps=1)
    candidate.env_cfg.move_cost = 2.0
    for agent in candidate.pop.alive_agents():
        agent.energy = 0.25

    def _move_north(_runner, _state, agent_ids, *, buckets=None):
        return np.full((int(agent_ids.shape[0]),), int(ACT_MOVE_N), dtype=np.int32)

    monkeypatch.setattr(tick_loop, "runnercore1_actions_strided", _move_north)
    result = _run_tick_loop(candidate)
    summary = candidate._runnercore2_churn_slot_lifecycle_summary

    assert result[0] > 0
    assert summary["state_summary"]["death_events_applied"] == result[0]
    assert summary["state_summary"]["action_cache_invalidate_events"] >= result[0]
    assert summary["fallback_counters"]["per_tick_agent_sync_rows"] == 0
