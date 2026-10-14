from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from env.env_config import EnvConfig
from evolution.population_pkg.population_config import PopConfig
from perf.cxxbackbone6_policy_service import (
    capture_resident_runtime_state_v6,
    execute_tick_flat_policy_service_v1,
    resident_runtime_state_v6_payload,
)
from simulation.metrics import MetricsLogger
from simulation.performance_profiles import FAST_EXPLORATORY_PROFILE_V2, apply_performance_profile
from simulation.runner import SimulationRunner
from simulation.runner_pkg.runner_config import SimConfig
from simulation.runnercore.action_cache import runnercore1_actions_strided
from simulation.runnercore.tick_state import import_generation_state


def _sim(tmp_path: Path, label: str, *, cxxbackbone6: bool) -> SimulationRunner:
    torch.manual_seed(12345)
    env_cfg = EnvConfig(width=8, height=8, view_radius=1)
    env_cfg.enable_biomes = False
    env_cfg.food_spawn_per_step = 20
    env_cfg.max_food = 80
    env_cfg.patch_world_enable = False
    env_cfg.patch_world_observation_enable = False
    env_cfg.prediction_demand_enable = False
    env_cfg.prediction_demand_observation_enable = False
    env_cfg.reproduce_try_prob = 0.0
    env_cfg.reproduce_requires_mate = False
    cfg = SimConfig(
        seed=4242,
        generations=1,
        steps_per_generation=1,
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
    cfg.runnercore3_vectorized_action_cache_enable = False
    cfg.cxxbackbone3_policy_aten_gru_authority_enable = True
    cfg.cxxbackbone4_policy_aten_mlp_authority_enable = True
    cfg.cxxbackbone3_policy_aten_build_dir = str(tmp_path / "aten_build")
    cfg.cxxbackbone5_policy_envelope_profile_enable = True
    cfg.cxxbackbone6_resident_policy_vertical_slice_enable = bool(cxxbackbone6)
    sim = SimulationRunner(
        env_cfg=env_cfg,
        pop_cfg=PopConfig(population_size=8),
        sim_cfg=cfg,
        logger=MetricsLogger(str(tmp_path / label)),
        run_dir=str(tmp_path / label),
        perf=True,
    )
    sim.pop.init_random()
    sim._ensure_bank()
    sim._place_agents()
    sim._mark_bank_dirty()
    sync_fn = getattr(sim, "_phase3_pre_start_generation_sync", None)
    if callable(sync_fn):
        sync_fn(list(sim.pop.alive_agents()))
    return sim


def _ids(sim: SimulationRunner) -> np.ndarray:
    return np.asarray([int(agent.agent_id) for agent in sorted(sim.pop.alive_agents(), key=lambda a: int(a.agent_id))], dtype=np.int32)


def test_cxxbackbone6_flat_policy_service_matches_runnercore_actions(tmp_path: Path) -> None:
    reference = _sim(tmp_path, "reference", cxxbackbone6=False)
    candidate = _sim(tmp_path, "candidate", cxxbackbone6=True)
    ids = _ids(reference)
    ref_state = import_generation_state(reference, generation=0, profile="supported_fast_runner_v1")
    cand_state = import_generation_state(candidate, generation=0, profile="supported_fast_runner_v1")
    ref_buckets: dict[str, float] = {}
    cand_buckets: dict[str, float] = {}

    ref_actions = runnercore1_actions_strided(reference, ref_state, ids, buckets=ref_buckets)
    cand_actions = runnercore1_actions_strided(candidate, cand_state, ids, buckets=cand_buckets)

    np.testing.assert_array_equal(cand_actions, ref_actions)
    np.testing.assert_array_equal(candidate._Hmem.detach().cpu().numpy(), reference._Hmem.detach().cpu().numpy())
    np.testing.assert_array_equal(cand_state.cache_actions, ref_state.cache_actions)
    np.testing.assert_array_equal(cand_state.cache_cursor, ref_state.cache_cursor)
    assert cand_buckets["cxxbackbone6_service_execute_tick_calls"] == 1.0
    assert cand_buckets["cxxbackbone6_slot_lookups_inside_lane"] == 0.0
    assert cand_buckets["cxxbackbone6_observation_callbacks_inside_lane"] == 0.0


def test_cxxbackbone6_resident_state_v6_payload_is_stable(tmp_path: Path) -> None:
    sim = _sim(tmp_path, "state", cxxbackbone6=True)
    ids = _ids(sim)
    state = import_generation_state(sim, generation=0, profile="supported_fast_runner_v1")
    snapshot = capture_resident_runtime_state_v6(sim, state, ids)
    payload = resident_runtime_state_v6_payload(snapshot)

    assert payload["schema_version"] == "cxxbackbone6_policy_service_v1"
    assert payload["alive_agent_ids"]["shape"] == [8]
    assert payload["flat_local_observations"]["shape"] == [8, 9]
