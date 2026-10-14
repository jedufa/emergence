from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from env.env_config import EnvConfig
from evolution.population_pkg.population_config import PopConfig
from simulation.metrics import MetricsLogger
from simulation.performance_profiles import FAST_EXPLORATORY_PROFILE_V2, apply_performance_profile
from simulation.runner import SimulationRunner
from simulation.runner_pkg.runner_config import SimConfig


def _env_cfg() -> EnvConfig:
    cfg = EnvConfig(width=8, height=8, view_radius=1)
    cfg.enable_biomes = False
    cfg.food_spawn_per_step = 20
    cfg.max_food = 80
    cfg.patch_world_enable = False
    cfg.patch_world_observation_enable = False
    cfg.prediction_demand_enable = False
    cfg.prediction_demand_observation_enable = False
    cfg.reproduce_requires_mate = False
    cfg.reproduce_try_prob = 1.0
    cfg.reproduce_parent_energy_share = 0.10
    cfg.reproduce_min_energy_buffer = 0.0
    cfg.reproduce_cooldown_steps = 1
    return cfg


def _sim_cfg(*, runnercore4: bool, steps: int = 5) -> SimConfig:
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
    if runnercore4:
        cfg.runnercore4_profile_enable = True
        cfg.runnercore4_candidate_enable = True
        cfg.runnercore4_allow_birth_boundary_agent_materialization = True
        cfg.runnercore4_structural_payload_cache_enable = True
        cfg.runnercore4_changed_parent_import_enable = True
    else:
        cfg.runnercore3_profile_enable = True
        cfg.runnercore3_candidate_enable = True
        cfg.runnercore3_allow_birth_boundary_agent_materialization = True
        cfg.runnercore3_fast_birth_event_boundary_enable = True
        cfg.runnercore3_self_recombination_fastpath_enable = True
    return cfg


def _prepare_sim(tmp_path: Path, *, runnercore4: bool, steps: int = 5) -> SimulationRunner:
    torch.manual_seed(12345)
    sim = SimulationRunner(
        env_cfg=_env_cfg(),
        pop_cfg=PopConfig(population_size=4),
        sim_cfg=_sim_cfg(runnercore4=runnercore4, steps=steps),
        logger=MetricsLogger(str(tmp_path)),
        run_dir=str(tmp_path),
        perf=True,
    )
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
    return sim


def _sha_json(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _policy_hash(policy: Any) -> str:
    digest = hashlib.sha256()
    for param in getattr(policy, "parameters", lambda: [])():
        arr = param.detach().cpu().contiguous().numpy()
        digest.update(str(tuple(arr.shape)).encode("ascii"))
        digest.update(arr.tobytes())
    return digest.hexdigest()


def _extended_fingerprint(sim: SimulationRunner) -> dict[str, Any]:
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
        "genome_hashes": [a.genome.canonical_hash() for a in alive],
        "policy_hashes": [_policy_hash(a.policy) for a in alive],
    }


def test_runnercore4_defaults_off() -> None:
    cfg = SimConfig()

    assert cfg.runnercore4_profile_enable is False
    assert cfg.runnercore4_candidate_enable is False
    assert cfg.runnercore4_native_enable is False
    assert cfg.runnercore4_allow_birth_boundary_agent_materialization is False
    assert cfg.runnercore4_structural_payload_cache_enable is False
    assert cfg.runnercore4_changed_parent_import_enable is False


def test_runnercore4_direct_self_child_matches_runnercore3_bytes_policy_and_rng(tmp_path: Path) -> None:
    r3 = _prepare_sim(tmp_path / "r3", runnercore4=False, steps=1)
    r4 = _prepare_sim(tmp_path / "r4", runnercore4=True, steps=1)
    p3 = sorted(r3.pop.alive_agents(), key=lambda a: int(a.agent_id))[0]
    p4 = sorted(r4.pop.alive_agents(), key=lambda a: int(a.agent_id))[0]

    children3 = [r3.pop.make_child(p3, p3, 12.5, buckets={}) for _ in range(2)]
    children4 = [r4.pop.make_child(p4, p4, 12.5, buckets={}) for _ in range(2)]

    assert [c is not None for c in children4] == [True, True]
    assert [c.genome.canonical_hash() for c in children4 if c is not None] == [
        c.genome.canonical_hash() for c in children3 if c is not None
    ]
    assert [_policy_hash(c.policy) for c in children4 if c is not None] == [
        _policy_hash(c.policy) for c in children3 if c is not None
    ]
    assert _sha_json(r4.rng_manager.indices_snapshot()) == _sha_json(r3.rng_manager.indices_snapshot())

    stats = getattr(r4.sim_cfg, "_runnercore4_structural_payload_stats", {})
    assert stats["child_construction_calls"] == 2
    assert stats["intermediate_child_genome_construction_avoided"] == 2
    assert stats["child_genome_to_dict_avoided"] == 2
    assert stats["structural_payload_cache_misses"] == 1
    assert stats["structural_payload_cache_hits"] == 1
    assert stats["c1_rng_budget_checks"] == 2
    assert stats["c2_rng_budget_checks"] == 2


def test_runnercore4_live_churn_matches_runnercore3_without_hidden_fallback(tmp_path: Path) -> None:
    r3 = _prepare_sim(tmp_path / "r3_live", runnercore4=False)
    r4 = _prepare_sim(tmp_path / "r4_live", runnercore4=True)

    def _forbidden(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("legacy action materializer called")

    for sim in (r3, r4):
        sim._batched_actions = _forbidden
        sim._batched_actions_k = _forbidden
        sim._batched_actions_strided = _forbidden

    result3 = r3._run_ticks(
        gen=0,
        buckets={},
        injury_recover_mult=1.0,
        deaths_this_gen=0,
        successful_births=0,
        failed_births=0,
        gen_peak_injury=0.0,
        rest_streaks={},
    )
    result4 = r4._run_ticks(
        gen=0,
        buckets={},
        injury_recover_mult=1.0,
        deaths_this_gen=0,
        successful_births=0,
        failed_births=0,
        gen_peak_injury=0.0,
        rest_streaks={},
    )

    assert result4 == result3
    assert _extended_fingerprint(r4) == _extended_fingerprint(r3)
    summary = r4._runnercore4_structural_payload_fastpath_summary
    assert summary["candidate_ran"] is True
    assert summary["fallback_counters"]["legacy_batched_actions_calls"] == 0
    assert summary["fallback_counters"]["legacy_batched_actions_k_calls"] == 0
    assert summary["fallback_counters"]["legacy_batched_actions_strided_calls"] == 0
    assert summary["fallback_counters"]["legacy_action_cache_refill_calls"] == 0
    assert summary["fallback_counters"]["per_tick_agent_sync_rows"] == 0
    assert summary["fallback_counters"]["unsupported_fallback_count"] == 0
    assert summary["runnercore4_structural_payload"]["intermediate_child_genome_construction_avoided"] > 0
    assert summary["runnercore4_structural_payload"]["event_boundary_parent_import_rows_avoided"] > 0
