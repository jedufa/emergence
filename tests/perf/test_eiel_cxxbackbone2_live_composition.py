from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from env.env_config import EnvConfig
from evolution.population_pkg.population_config import PopConfig
from perf.cxxbackbone2_live_composition import (
    build_live_composition_map,
    capture_resident_runtime_state_v5,
    composition_group_status,
    composition_map_payload,
    execute_generation,
    summary_payload,
    write_composition_map,
)
from simulation.metrics import MetricsLogger
from simulation.performance_profiles import FAST_EXPLORATORY_PROFILE_V2, apply_performance_profile
from simulation.runner import SimulationRunner
from simulation.runner_pkg.runner_config import SimConfig
from simulation.checkpoint_bundle_v2 import checkpoint_bundle_v2_apply, checkpoint_bundle_v2_load, checkpoint_bundle_v2_save
from simulation.runnercore.action_cache import _compute_actions_k
from simulation.runnercore.tick_state import import_generation_state


class _Pop:
    def __init__(self) -> None:
        self._agents = [
            SimpleNamespace(agent_id=10, slot=0, alive=True),
            SimpleNamespace(agent_id=11, slot=1, alive=True),
        ]

    def alive_agents(self):
        return list(self._agents)


def _runner(tmp_path: Path):
    return SimpleNamespace(
        run_dir=str(tmp_path),
        sim_cfg=SimpleNamespace(
            steps_per_generation=3,
            population_cap=4,
            performance_profile_name="test_profile",
            campaign_cell_id="cell_test",
        ),
        env=SimpleNamespace(
            agent_alive=np.asarray([False, False, False, False, False, False, False, False, False, False, True, True]),
            agent_pos_y=np.asarray([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2], dtype=np.int32),
            agent_pos_x=np.asarray([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 4], dtype=np.int32),
            agent_energy=np.asarray([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7.0, 8.0], dtype=np.float32),
            agent_health=np.asarray([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0.9], dtype=np.float32),
            agent_grid=np.asarray([[10, -1], [-1, 11]], dtype=np.int32),
        ),
        pop=_Pop(),
    )


def test_cxxbackbone2_live_composition_map_has_all_19_surfaces(tmp_path: Path) -> None:
    rows = build_live_composition_map(implementation_commit="abc123")
    payload = composition_map_payload(rows)
    out = write_composition_map(tmp_path / "map.json", implementation_commit="abc123")

    assert out.exists()
    assert payload["surface_count"] == 19
    assert payload["live_integrated_count"] == 3
    assert payload["blocked_count"] == 16
    assert all(row.contract_source for row in rows)


def test_cxxbackbone2_state_v5_captures_live_runner_arrays(tmp_path: Path) -> None:
    state = capture_resident_runtime_state_v5(_runner(tmp_path), generation=2, implementation_commit="abc123")
    payload = state.__dict__

    assert state.schema_version == "resident_runtime_state_v5"
    assert payload["generation"] == 2
    assert state.arrays["alive_agent_ids"].tolist() == [10, 11]
    assert "execution_cell_metadata" in state.live_integrated_surfaces
    assert "action_cache" in state.blocked_surfaces


def test_cxxbackbone2_execute_generation_fails_closed_before_mutation(tmp_path: Path) -> None:
    buckets: dict[str, float] = {}
    result = execute_generation(_runner(tmp_path), generation=3, buckets=buckets, implementation_commit="abc123")
    summary = summary_payload(result)

    assert result["executed"] is False
    assert result["hidden_fallback"] is False
    assert result["native_call_count"] == 0
    assert summary["decision"] == "CXXBACKBONE2-D/MULTIPLE-LIVE-GROUPS-COMPOSITION-BLOCKED"
    assert summary["composition_group_status"]["group_a_policy_core"]["supported"] is False
    assert "production_observation_builder_still_python_owned" in summary["composition_group_status"]["group_a_policy_core"]["blockers"]
    assert "cxxbackbone2_live_composition_gate" in buckets


def test_cxxbackbone2_group_status_localizes_policy_core_live_blockers(tmp_path: Path) -> None:
    rows = build_live_composition_map(implementation_commit="abc123")
    status = composition_group_status(_runner(tmp_path), rows=rows)

    policy = status["group_a_policy_core"]
    assert policy["supported"] is False
    assert "observations_staging" in policy["live_integrated_surfaces"]
    assert "action_cache" in policy["blocked_surfaces"]
    assert "mlp_pack_missing_or_not_synced" in policy["blockers"]
    assert "hmem_writeback_still_torch_python_orchestrated" in policy["blockers"]


def test_cxxbackbone2_gate_reaches_real_runner_and_writes_summary(tmp_path: Path) -> None:
    env_cfg = EnvConfig(width=6, height=6, view_radius=1)
    env_cfg.enable_biomes = False
    env_cfg.patch_world_enable = False
    env_cfg.patch_world_observation_enable = False
    cfg = SimConfig(
        seed=5151,
        generations=1,
        steps_per_generation=2,
        device="cpu",
        bank_device="cpu",
        behavior_log_enable=False,
        policy_deterministic_actions=True,
    )
    apply_performance_profile(cfg, FAST_EXPLORATORY_PROFILE_V2)
    cfg.phase_e_e1_enable = False
    cfg.phase_e_e2_enable = False
    cfg.epi_enable = False
    cfg.enable_cognitive_budget_ledger = False
    cfg.chemistry_functional_coupling_enable = False
    cfg.checkpoint_bundle_v2_period_gen = 0
    cfg.maintenance_enable = False
    cfg.cxxbackbone2_live_composition_enable = True
    cfg.cxxbackbone2_live_composition_fail_closed = True
    sim = SimulationRunner(
        env_cfg=env_cfg,
        pop_cfg=PopConfig(population_size=4),
        sim_cfg=cfg,
        logger=MetricsLogger(str(tmp_path)),
        run_dir=str(tmp_path),
        perf=True,
    )

    with pytest.raises(RuntimeError, match="CXXBACKBONE2 live composition unsupported"):
        sim.run()

    summary = tmp_path / "metrics" / "cxxbackbone2_live_composition_summary.json"
    assert summary.exists()
    text = summary.read_text(encoding="utf-8")
    assert "CXXBACKBONE2-D/MULTIPLE-LIVE-GROUPS-COMPOSITION-BLOCKED" in text
    assert "action_cache:exact_contract_exists_but_live_runner_still_python_authoritative" in text


def test_cxxbackbone2_policy_core_probe_reaches_real_runner_refill_boundary(tmp_path: Path) -> None:
    torch.manual_seed(12345)
    env_cfg = EnvConfig(width=8, height=8, view_radius=1)
    env_cfg.enable_biomes = False
    env_cfg.food_spawn_per_step = 20
    env_cfg.max_food = 80
    env_cfg.patch_world_enable = False
    env_cfg.patch_world_observation_enable = False
    env_cfg.prediction_demand_enable = False
    env_cfg.prediction_demand_observation_enable = False
    cfg = SimConfig(
        seed=4242,
        generations=1,
        steps_per_generation=1,
        device="cpu",
        bank_device="cpu",
        behavior_log_enable=False,
        policy_deterministic_actions=True,
        policy_tick_stride=1,
    )
    apply_performance_profile(cfg, FAST_EXPLORATORY_PROFILE_V2)
    cfg.phase_e_e1_enable = False
    cfg.phase_e_e2_enable = False
    cfg.epi_enable = False
    cfg.enable_cognitive_budget_ledger = False
    cfg.chemistry_functional_coupling_enable = False
    cfg.checkpoint_bundle_v2_period_gen = 0
    cfg.maintenance_enable = False
    cfg.cxxbackbone2_policy_core_live_probe_enable = True
    cfg.cxxbackbone2_policy_core_live_probe_build_dir = str(tmp_path / "policy_core_build")
    sim = SimulationRunner(
        env_cfg=env_cfg,
        pop_cfg=PopConfig(population_size=4),
        sim_cfg=cfg,
        logger=MetricsLogger(str(tmp_path)),
        run_dir=str(tmp_path),
        perf=True,
    )
    sim.pop.init_random()
    sim._ensure_bank()
    sim._place_agents()
    sim._mark_bank_dirty()
    sync_fn = getattr(sim, "_phase3_pre_start_generation_sync", None)
    if callable(sync_fn):
        sync_fn(list(sim.pop.alive_agents()))

    alive = sorted(sim.pop.alive_agents(), key=lambda agent: int(agent.agent_id))
    ids = np.asarray([int(agent.agent_id) for agent in alive], dtype=np.int32)
    state = import_generation_state(sim, generation=0, profile="supported_fast_runner_v2")
    actions = _compute_actions_k(sim, state, ids, k=1, buckets={})
    summary = sim._cxxbackbone2_policy_core_live_probe_summary

    assert actions.shape == (4, 1)
    assert summary["executed_native"] is True
    assert summary["hidden_fallback"] is False
    assert summary["exact_action_parity"] is True
    assert summary["tolerant_hmem_parity"] is True
    assert summary["supported"] is False
    assert "native_policy_hmem_not_bit_exact_with_production" in summary["blockers"]
    assert (tmp_path / "metrics" / "cxxbackbone2_policy_core_live_probe_summary.json").exists()


def test_cxxbackbone3_aten_gru_authority_matches_production_hmem_bits(tmp_path: Path) -> None:
    def _run(label: str, *, gru_authority: bool, mlp_authority: bool = False):
        torch.manual_seed(12345)
        env_cfg = EnvConfig(width=8, height=8, view_radius=1)
        env_cfg.enable_biomes = False
        env_cfg.food_spawn_per_step = 20
        env_cfg.max_food = 80
        env_cfg.patch_world_enable = False
        env_cfg.patch_world_observation_enable = False
        env_cfg.prediction_demand_enable = False
        env_cfg.prediction_demand_observation_enable = False
        cfg = SimConfig(
            seed=4242,
            generations=1,
            steps_per_generation=1,
            device="cpu",
            bank_device="cpu",
            behavior_log_enable=False,
            policy_deterministic_actions=True,
            policy_tick_stride=1,
        )
        apply_performance_profile(cfg, FAST_EXPLORATORY_PROFILE_V2)
        cfg.phase_e_e1_enable = False
        cfg.phase_e_e2_enable = False
        cfg.epi_enable = False
        cfg.enable_cognitive_budget_ledger = False
        cfg.chemistry_functional_coupling_enable = False
        cfg.checkpoint_bundle_v2_period_gen = 0
        cfg.maintenance_enable = False
        cfg.cxxbackbone3_policy_aten_gru_authority_enable = bool(gru_authority)
        cfg.cxxbackbone4_policy_aten_mlp_authority_enable = bool(mlp_authority)
        cfg.cxxbackbone3_policy_aten_build_dir = str(tmp_path / "aten_build")
        sim = SimulationRunner(
            env_cfg=env_cfg,
            pop_cfg=PopConfig(population_size=4),
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
        alive = sorted(sim.pop.alive_agents(), key=lambda agent: int(agent.agent_id))
        ids = np.asarray([int(agent.agent_id) for agent in alive], dtype=np.int32)
        state = import_generation_state(sim, generation=0, profile="supported_fast_runner_v2")
        buckets: dict[str, float] = {}
        actions = _compute_actions_k(sim, state, ids, k=1, buckets=buckets)
        hmem = sim._Hmem.detach().cpu().numpy().copy()
        return actions, hmem, buckets

    reference_actions, reference_hmem, _reference_buckets = _run("reference", gru_authority=False)
    aten_actions, aten_hmem, aten_buckets = _run("aten", gru_authority=True)

    np.testing.assert_array_equal(aten_actions, reference_actions)
    np.testing.assert_array_equal(aten_hmem, reference_hmem)
    assert aten_buckets["cxxbackbone3_policy_aten_gru_rows"] > 0


def test_cxxbackbone4_aten_mlp_and_gru_authority_match_production_bits(tmp_path: Path) -> None:
    def _run(label: str, *, authority: bool):
        torch.manual_seed(12345)
        env_cfg = EnvConfig(width=8, height=8, view_radius=1)
        env_cfg.enable_biomes = False
        env_cfg.food_spawn_per_step = 20
        env_cfg.max_food = 80
        env_cfg.patch_world_enable = False
        env_cfg.patch_world_observation_enable = False
        env_cfg.prediction_demand_enable = False
        env_cfg.prediction_demand_observation_enable = False
        cfg = SimConfig(
            seed=4242,
            generations=1,
            steps_per_generation=1,
            device="cpu",
            bank_device="cpu",
            behavior_log_enable=False,
            policy_deterministic_actions=True,
            policy_tick_stride=1,
        )
        apply_performance_profile(cfg, FAST_EXPLORATORY_PROFILE_V2)
        cfg.phase_e_e1_enable = False
        cfg.phase_e_e2_enable = False
        cfg.epi_enable = False
        cfg.enable_cognitive_budget_ledger = False
        cfg.chemistry_functional_coupling_enable = False
        cfg.checkpoint_bundle_v2_period_gen = 0
        cfg.maintenance_enable = False
        cfg.cxxbackbone3_policy_aten_gru_authority_enable = bool(authority)
        cfg.cxxbackbone4_policy_aten_mlp_authority_enable = bool(authority)
        cfg.cxxbackbone3_policy_aten_build_dir = str(tmp_path / "aten_build")
        sim = SimulationRunner(
            env_cfg=env_cfg,
            pop_cfg=PopConfig(population_size=4),
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
        alive = sorted(sim.pop.alive_agents(), key=lambda agent: int(agent.agent_id))
        ids = np.asarray([int(agent.agent_id) for agent in alive], dtype=np.int32)
        state = import_generation_state(sim, generation=0, profile="supported_fast_runner_v2")
        buckets: dict[str, float] = {}
        actions = _compute_actions_k(sim, state, ids, k=1, buckets=buckets)
        hmem = sim._Hmem.detach().cpu().numpy().copy()
        return actions, hmem, buckets

    reference_actions, reference_hmem, _reference_buckets = _run("reference_mlp_gru", authority=False)
    aten_actions, aten_hmem, aten_buckets = _run("aten_mlp_gru", authority=True)

    np.testing.assert_array_equal(aten_actions, reference_actions)
    np.testing.assert_array_equal(aten_hmem, reference_hmem)
    assert aten_buckets["cxxbackbone3_policy_aten_gru_rows"] > 0
    assert aten_buckets["cxxbackbone4_policy_aten_mlp_rows"] > 0


def test_cxxbackbone4_policy_authority_checkpoint_continuation(tmp_path: Path) -> None:
    def _sim(label: str) -> SimulationRunner:
        torch.manual_seed(12345)
        env_cfg = EnvConfig(width=8, height=8, view_radius=1)
        env_cfg.enable_biomes = False
        env_cfg.food_spawn_per_step = 20
        env_cfg.max_food = 80
        env_cfg.patch_world_enable = False
        env_cfg.patch_world_observation_enable = False
        env_cfg.prediction_demand_enable = False
        env_cfg.prediction_demand_observation_enable = False
        cfg = SimConfig(
            seed=4242,
            generations=1,
            steps_per_generation=1,
            device="cpu",
            bank_device="cpu",
            behavior_log_enable=False,
            policy_deterministic_actions=True,
            policy_tick_stride=1,
        )
        apply_performance_profile(cfg, FAST_EXPLORATORY_PROFILE_V2)
        cfg.phase_e_e1_enable = False
        cfg.phase_e_e2_enable = False
        cfg.epi_enable = False
        cfg.enable_cognitive_budget_ledger = False
        cfg.chemistry_functional_coupling_enable = False
        cfg.checkpoint_bundle_v2_period_gen = 0
        cfg.maintenance_enable = False
        cfg.cxxbackbone3_policy_aten_gru_authority_enable = True
        cfg.cxxbackbone4_policy_aten_mlp_authority_enable = True
        cfg.cxxbackbone3_policy_aten_build_dir = str(tmp_path / "aten_build")
        sim = SimulationRunner(
            env_cfg=env_cfg,
            pop_cfg=PopConfig(population_size=4),
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

    def _policy_step(sim: SimulationRunner, generation: int) -> tuple[np.ndarray, np.ndarray]:
        alive = sorted(sim.pop.alive_agents(), key=lambda agent: int(agent.agent_id))
        ids = np.asarray([int(agent.agent_id) for agent in alive], dtype=np.int32)
        state = import_generation_state(sim, generation=generation, profile="supported_fast_runner_v2")
        actions = _compute_actions_k(sim, state, ids, k=1, buckets={})
        hmem = sim._Hmem.detach().cpu().numpy().copy()
        return actions, hmem

    uninterrupted = _sim("uninterrupted")
    first_actions, first_hmem = _policy_step(uninterrupted, generation=0)
    bundle = tmp_path / "checkpoint_bundles_v2" / "bundle_gen000001"
    checkpoint_bundle_v2_save(uninterrupted, str(bundle), generation_index=1, strict_replay_mode=True)
    uninterrupted_actions, uninterrupted_hmem = _policy_step(uninterrupted, generation=1)

    restored = _sim("restored")
    payload = checkpoint_bundle_v2_load(str(bundle), strict=True)
    checkpoint_bundle_v2_apply(restored, payload, restore_rng=True)
    restored_actions, restored_hmem = _policy_step(restored, generation=1)

    assert first_actions.shape == uninterrupted_actions.shape
    assert first_hmem.shape == uninterrupted_hmem.shape
    np.testing.assert_array_equal(restored_actions, uninterrupted_actions)
    np.testing.assert_array_equal(restored_hmem, uninterrupted_hmem)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA policy authority probe requires CUDA-capable Torch")
def test_cxxbackbone4_aten_policy_authority_cuda_matches_production_bits(tmp_path: Path) -> None:
    def _run(label: str, *, authority: bool):
        torch.manual_seed(12345)
        env_cfg = EnvConfig(width=8, height=8, view_radius=1)
        env_cfg.enable_biomes = False
        env_cfg.food_spawn_per_step = 20
        env_cfg.max_food = 80
        env_cfg.patch_world_enable = False
        env_cfg.patch_world_observation_enable = False
        env_cfg.prediction_demand_enable = False
        env_cfg.prediction_demand_observation_enable = False
        cfg = SimConfig(
            seed=4242,
            generations=1,
            steps_per_generation=1,
            device="cuda",
            bank_device="cuda",
            behavior_log_enable=False,
            policy_deterministic_actions=True,
            policy_tick_stride=1,
        )
        apply_performance_profile(cfg, FAST_EXPLORATORY_PROFILE_V2)
        cfg.phase_e_e1_enable = False
        cfg.phase_e_e2_enable = False
        cfg.epi_enable = False
        cfg.enable_cognitive_budget_ledger = False
        cfg.chemistry_functional_coupling_enable = False
        cfg.checkpoint_bundle_v2_period_gen = 0
        cfg.maintenance_enable = False
        cfg.cxxbackbone3_policy_aten_gru_authority_enable = bool(authority)
        cfg.cxxbackbone4_policy_aten_mlp_authority_enable = bool(authority)
        cfg.cxxbackbone3_policy_aten_build_dir = str(tmp_path / "aten_cuda_build")
        sim = SimulationRunner(
            env_cfg=env_cfg,
            pop_cfg=PopConfig(population_size=4),
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
        alive = sorted(sim.pop.alive_agents(), key=lambda agent: int(agent.agent_id))
        ids = np.asarray([int(agent.agent_id) for agent in alive], dtype=np.int32)
        state = import_generation_state(sim, generation=0, profile="supported_fast_runner_v2")
        buckets: dict[str, float] = {}
        actions = _compute_actions_k(sim, state, ids, k=1, buckets=buckets)
        hmem = sim._Hmem.detach().cpu().numpy().copy()
        return actions, hmem, buckets

    reference_actions, reference_hmem, _reference_buckets = _run("cuda_reference", authority=False)
    aten_actions, aten_hmem, aten_buckets = _run("cuda_aten", authority=True)

    np.testing.assert_array_equal(aten_actions, reference_actions)
    np.testing.assert_array_equal(aten_hmem, reference_hmem)
    assert aten_buckets["cxxbackbone3_policy_aten_gru_rows"] > 0
    assert aten_buckets["cxxbackbone4_policy_aten_mlp_rows"] > 0
