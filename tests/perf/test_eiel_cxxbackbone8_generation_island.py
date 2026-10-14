from __future__ import annotations

from pathlib import Path
import random

import numpy as np
import pytest
import torch

from env.env_config import EnvConfig
from evolution.population_pkg.population_config import PopConfig
from perf.cxxbackbone8_generation_island import (
    DECISION_INVALID,
    capture_resident_runtime_state_v8,
    execute_generation,
    implementation_attempts_payload,
    resident_runtime_state_v8_payload,
)
from simulation.metrics import MetricsLogger
from simulation.performance_profiles import FAST_EXPLORATORY_PROFILE_V2, apply_performance_profile
from simulation.runner import SimulationRunner
from simulation.runner_pkg.runner_config import SimConfig
from simulation.runnercore.tick_loop import RunnerCore1UnsupportedError
from simulation.runnercore.tick_state import import_generation_state


def _sim(tmp_path: Path, *, kind: str, cxxbackbone8: bool = False) -> SimulationRunner:
    random.seed(12345)
    np.random.seed(12345)
    torch.manual_seed(12345)
    env_cfg = EnvConfig(width=8, height=8, view_radius=1)
    env_cfg.enable_biomes = False
    env_cfg.food_spawn_per_step = 20
    env_cfg.max_food = 80
    env_cfg.patch_world_enable = kind == "patch"
    env_cfg.patch_world_observation_enable = kind == "patch"
    env_cfg.prediction_demand_enable = False
    env_cfg.prediction_demand_observation_enable = False
    env_cfg.reproduce_try_prob = 0.0
    env_cfg.reproduce_requires_mate = False
    cfg = SimConfig(
        seed=4242,
        generations=1,
        steps_per_generation=4,
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
    cfg.runnercore1_array_tick_authority_enable = True
    cfg.runnercore1_candidate_enable = True
    cfg.runnercore1_fail_closed_on_unsupported = True
    cfg.cxxbackbone3_policy_aten_gru_authority_enable = True
    cfg.cxxbackbone4_policy_aten_mlp_authority_enable = True
    cfg.cxxbackbone3_policy_aten_build_dir = str(tmp_path / "aten_build")
    cfg.cxxbackbone7_resident_patch_policy_action_loop_enable = True
    cfg.cxxbackbone8_resident_flat_patch_generation_island_enable = bool(cxxbackbone8)
    cfg.cxxbackbone8_resident_flat_patch_generation_island_fail_closed = True
    return SimulationRunner(
        env_cfg=env_cfg,
        pop_cfg=PopConfig(population_size=12),
        sim_cfg=cfg,
        logger=MetricsLogger(str(tmp_path / kind)),
        run_dir=str(tmp_path / kind),
        perf=True,
    )


def _initialized_state(sim: SimulationRunner):
    sim.pop.init_random()
    sim._ensure_bank()
    sim._place_agents()
    sim._mark_bank_dirty()
    sync_fn = getattr(sim, "_phase3_pre_start_generation_sync", None)
    if callable(sync_fn):
        sync_fn(list(sim.pop.alive_agents()))
    return import_generation_state(sim, generation=0, profile="supported_fast_runner_v1")


def test_cxxbackbone8_flags_default_off() -> None:
    cfg = SimConfig()

    assert cfg.cxxbackbone8_resident_flat_patch_generation_island_enable is False
    assert cfg.cxxbackbone8_resident_flat_patch_generation_island_fail_closed is True
    assert cfg.cxxbackbone8_resident_flat_patch_generation_island_write_summary is True


def test_resident_runtime_state_v8_captures_patch_state_and_blockers(tmp_path: Path) -> None:
    sim = _sim(tmp_path, kind="patch")
    state = _initialized_state(sim)
    snapshot = capture_resident_runtime_state_v8(sim, state, generation=0, buckets={})
    payload = resident_runtime_state_v8_payload(snapshot)

    assert payload["schema_version"] == "resident_runtime_state_v8"
    assert payload["alive_agent_ids"]["shape"] == [12]
    assert "patch_food_amount" in payload["patch_arrays"]
    assert "native_action_application_missing" in payload["blockers"]
    assert payload["capability_mask"]["flat_observation_exact_from_cxxbackbone7"] is True


def test_cxxbackbone8_execute_generation_fails_closed_without_mutation(tmp_path: Path) -> None:
    sim = _sim(tmp_path, kind="flat")
    state = _initialized_state(sim)
    before = capture_resident_runtime_state_v8(sim, state, generation=0, buckets={})
    result = execute_generation(sim, state, generation_index=0, steps_per_generation=4, buckets={})
    after = capture_resident_runtime_state_v8(sim, state, generation=0, buckets={})

    assert result["decision"] == DECISION_INVALID
    assert result["executed"] is False
    assert result["hidden_fallback"] is False
    assert result["science_state_mutated"] is False
    assert result["resident_state_after_digest"] == before.state_digest
    assert after.state_digest == before.state_digest
    assert len(implementation_attempts_payload()) == 2


def test_cxxbackbone8_production_gate_writes_summary_and_fails_closed(tmp_path: Path) -> None:
    sim = _sim(tmp_path, kind="patch", cxxbackbone8=True)

    with pytest.raises(RunnerCore1UnsupportedError, match="CXXBACKBONE8-G"):
        sim.run()

    summary = tmp_path / "patch" / "metrics" / "cxxbackbone8_generation_island_summary.json"
    assert summary.exists()
    text = summary.read_text(encoding="utf-8")
    assert "native_grouping_and_row_maps_missing" in text
    assert "extend_cxxbackbone7_service_to_generation_boundary" in text
