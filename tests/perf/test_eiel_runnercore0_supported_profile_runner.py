from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from env.env_config import EnvConfig
from evolution.population_pkg.population_config import PopConfig
from simulation.metrics import MetricsLogger
from simulation.performance_profiles import FAST_EXPLORATORY_PROFILE_V2, apply_performance_profile
from simulation.runner import SimulationRunner
from simulation.runner_pkg.runner_config import SimConfig
from simulation.runnercore.gate import RunnerCoreUnsupportedError, maybe_runnercore_supported_tick
from simulation.runnercore.profile import (
    RUNNERCORE_DECISION_BLOCKED_BY_BOUNDARY,
    SUPPORTED_PROFILE,
    classify_supported_profile,
    profile_contract_payload,
)
from simulation.runnercore.state import import_generation_start, state_digest, validate_state_against_runner


def _supported_cfg(**updates: object) -> SimConfig:
    cfg = SimConfig(
        seed=123,
        generations=1,
        steps_per_generation=4,
        device="cpu",
        bank_device="cpu",
        behavior_log_enable=False,
        policy_deterministic_actions=True,
        policy_tick_stride=2,
    )
    cfg.chemistry_functional_coupling_enable = False
    cfg.checkpoint_bundle_v2_period_gen = 0
    cfg.perf_aggperf3_timing_buckets_enable = True
    cfg.runnercore_supported_profile_enable = True
    cfg.runnercore_shadow_contract_enable = True
    for key, value in updates.items():
        setattr(cfg, key, value)
    return cfg


def _agent(agent_id: int, slot: int, token: int) -> SimpleNamespace:
    genome = SimpleNamespace(
        aggression_bias=0.1 * float(agent_id),
        action_temperature=1.0,
        drive_logit_gain=0.5,
        use_gru=slot % 2 == 0,
    )
    return SimpleNamespace(
        agent_id=int(agent_id),
        slot=int(slot),
        slot_token=int(token),
        alive=True,
        energy=10.0 + float(agent_id),
        health=20.0 + float(agent_id),
        injury=0.25 * float(slot),
        age=3 + int(agent_id),
        drives=np.asarray([0.1, 0.2, 0.3, 0.4], dtype=np.float32) + float(slot),
        genome=genome,
    )


def _fake_runner(tmp_path: Path, cfg: SimConfig | None = None) -> SimpleNamespace:
    agents = [_agent(5, 0, 100), _agent(7, 2, 102)]
    env = SimpleNamespace(
        agent_pos_y=np.asarray([-1, -1, -1, -1, -1, 3, -1, 4], dtype=np.int32),
        agent_pos_x=np.asarray([-1, -1, -1, -1, -1, 1, -1, 2], dtype=np.int32),
        agent_alive=np.asarray([False, False, False, False, False, True, False, True], dtype=bool),
    )
    pop = SimpleNamespace(max_slots=4, alive_agents=lambda: agents)
    return SimpleNamespace(
        env=env,
        pop=pop,
        sim_cfg=cfg if cfg is not None else _supported_cfg(),
        run_dir=str(tmp_path),
        _Hmem=np.arange(16, dtype=np.float32).reshape(4, 4),
        _policy_action_cache={0: {"slot_token": 100, "cursor": 1, "next_action": 4}},
    )


def test_runnercore0_config_defaults_off() -> None:
    cfg = SimConfig()

    assert cfg.runnercore_supported_profile_enable is False
    assert cfg.runnercore_shadow_contract_enable is False
    assert cfg.runnercore_candidate_enable is False
    assert cfg.runnercore_native_enable is False


def test_runnercore0_supported_profile_contract_accepts_shadow_only() -> None:
    report = classify_supported_profile(_supported_cfg())

    assert report["shadow_contract_supported"] is True
    assert report["candidate_supported"] is False
    assert report["decision"] == "RUNNERCORE0-SHADOW-CONTRACT-ONLY"
    assert profile_contract_payload()["supported_profile"] == SUPPORTED_PROFILE


def test_runnercore0_candidate_fails_closed_on_boundary(tmp_path: Path) -> None:
    cfg = _supported_cfg(runnercore_candidate_enable=True)
    runner = _fake_runner(tmp_path, cfg)

    with pytest.raises(RunnerCoreUnsupportedError, match="RUNNERCORE0-C/BLOCKED-BY-BOUNDARY"):
        maybe_runnercore_supported_tick(runner, alive=runner.pop.alive_agents(), generation=0, tick=0, buckets={})

    summary = runner._runnercore0_supported_profile_summary
    assert summary["candidate_fail_closed_ticks"] == 1
    assert summary["last_support_report"]["decision"] == RUNNERCORE_DECISION_BLOCKED_BY_BOUNDARY


def test_runnercore0_state_import_preserves_slot_arrays(tmp_path: Path) -> None:
    runner = _fake_runner(tmp_path)
    state = import_generation_start(runner, alive=runner.pop.alive_agents(), generation=2, tick=3)
    digest = state_digest(state)
    validation = validate_state_against_runner(state, runner, alive=runner.pop.alive_agents())

    assert state.slot_to_agent_id.tolist() == [5, -1, 7, -1]
    assert state.slot_tokens.tolist() == [100, -1, 102, -1]
    assert state.pos_y.tolist() == [3, -1, 4, -1]
    assert state.action_cache_token.tolist() == [100, -1, -1, -1]
    assert state.hmem.shape == (4, 4)
    assert validation["ok"] is True
    assert digest["digest_sha256"]


def _smoke_env_cfg() -> EnvConfig:
    cfg = EnvConfig(width=6, height=6, view_radius=1)
    cfg.enable_biomes = False
    cfg.food_spawn_per_step = 4
    cfg.max_food = 24
    cfg.patch_world_enable = True
    cfg.patch_world_observation_enable = True
    cfg.prediction_demand_enable = True
    cfg.prediction_demand_observation_enable = True
    cfg.prediction_demand_delay_min = 4
    cfg.prediction_demand_delay_max = 6
    cfg.prediction_demand_active_duration = 2
    cfg.prediction_demand_event_period_steps = 10
    cfg.prediction_demand_radius = 1
    return cfg


def _smoke_fingerprint(tmp_path: Path, *, runnercore_shadow: bool) -> dict[str, object]:
    cfg = _supported_cfg(seed=987, generations=1, steps_per_generation=4)
    cfg.runnercore_supported_profile_enable = bool(runnercore_shadow)
    cfg.runnercore_shadow_contract_enable = bool(runnercore_shadow)
    cfg.memory_action_window_instrumentation_enable = False
    cfg.prediction_demand_action_context_enable = False
    apply_performance_profile(cfg, FAST_EXPLORATORY_PROFILE_V2)
    cfg.chemistry_functional_coupling_enable = False
    cfg.runnercore_supported_profile_enable = bool(runnercore_shadow)
    cfg.runnercore_shadow_contract_enable = bool(runnercore_shadow)
    logger = MetricsLogger(str(tmp_path))
    sim = SimulationRunner(
        env_cfg=_smoke_env_cfg(),
        pop_cfg=PopConfig(population_size=6),
        sim_cfg=cfg,
        logger=logger,
        run_dir=str(tmp_path),
        perf=True,
    )
    sim.run()
    alive = sorted(sim.pop.alive_agents(), key=lambda a: int(a.agent_id))
    ids = [int(a.agent_id) for a in alive]
    return {
        "alive_ids": ids,
        "health": [round(float(a.health), 8) for a in alive],
        "energy": [round(float(a.energy), 8) for a in alive],
        "pos_y": np.asarray(sim.env.agent_pos_y[ids], dtype=np.int32).tolist() if ids else [],
        "pos_x": np.asarray(sim.env.agent_pos_x[ids], dtype=np.int32).tolist() if ids else [],
        "final_generation": dict(getattr(logger, "last_row", {}) or {}).get("generation"),
        "runnercore_summary_exists": (tmp_path / "metrics" / "runnercore0_supported_profile_summary.json").exists(),
    }


def test_runnercore0_shadow_contract_preserves_tiny_live_fingerprint(tmp_path: Path) -> None:
    legacy = _smoke_fingerprint(tmp_path / "legacy", runnercore_shadow=False)
    shadow = _smoke_fingerprint(tmp_path / "shadow", runnercore_shadow=True)

    assert {k: v for k, v in shadow.items() if k != "runnercore_summary_exists"} == {
        k: v for k, v in legacy.items() if k != "runnercore_summary_exists"
    }
    assert legacy["runnercore_summary_exists"] is False
    assert shadow["runnercore_summary_exists"] is True
