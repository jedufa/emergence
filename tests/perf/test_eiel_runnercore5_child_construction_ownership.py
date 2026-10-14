from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from env.env_config import EnvConfig
from env.gridworld_pkg.world import GridWorld
from agents.genome import Genome
from agents.modular_genome_v1 import canonical_payload_from_genome_dict
from agents.perception import minimal_sensor_channel_mask
from agents.cognition import compute_env_cost_charge_scalar_from_surface, express_cost_charge_surface
from evolution.operators_c1 import apply_operator_pipeline
from evolution.population_pkg.population_config import PopConfig
from simulation.metrics import MetricsLogger
from simulation.performance_profiles import FAST_EXPLORATORY_PROFILE_V2, apply_performance_profile
from simulation.runner import SimulationRunner
from simulation.runner_pkg.runner_config import SimConfig
import simulation.runnercore.action_cache as action_cache_mod
from simulation.runnercore.action_cache import _compute_actions_k
from simulation.runnercore.tick_loop import classify_support
from simulation.runnercore.tick_state import import_generation_state


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


def _env_cfg(*, patch_world: bool = False, reproduce_requires_mate: bool = False) -> EnvConfig:
    cfg = EnvConfig(width=8, height=8, view_radius=1)
    cfg.enable_biomes = False
    cfg.food_spawn_per_step = 20
    cfg.max_food = 80
    cfg.patch_world_enable = bool(patch_world)
    cfg.patch_world_observation_enable = bool(patch_world)
    cfg.prediction_demand_enable = False
    cfg.prediction_demand_observation_enable = False
    cfg.reproduce_requires_mate = bool(reproduce_requires_mate)
    cfg.reproduce_try_prob = 1.0
    cfg.reproduce_parent_energy_share = 0.10
    cfg.reproduce_min_energy_buffer = 0.0
    cfg.reproduce_cooldown_steps = 1
    return cfg


def _sim_cfg(
    *,
    runnercore4: bool = True,
    runnercore5: bool,
    runnercore6: bool = False,
    runnercore7: bool = False,
    runnercore9: bool = False,
    runnercore10: bool = False,
    runnercore11: bool = False,
    runnercore12: bool = False,
    runnercore13: bool = False,
    runnercore15: bool = False,
    runnercore16: bool = False,
    runnercore17: bool = False,
    runnercore18: bool = False,
    runnercore19: bool = False,
    runnercore20: bool = False,
    runnercore21: bool = False,
    runnercore22: bool = False,
    runnercore23: bool = False,
    runnercore24: bool = False,
    runnercore25: bool = False,
    runnercore26: bool = False,
    runnercore27: bool = False,
    runnercore30: bool = False,
    runnercore31: bool = False,
    runnercore32: bool = False,
    runnercore33: bool = False,
    runnercore34: bool = False,
    runnercore42: bool = False,
    runnercore43: bool = False,
    runnercore45: bool = False,
    runnercore47: bool = False,
    runnercore48: bool = False,
    runnercore49: bool = False,
    runnercore50: bool = False,
    steps: int = 5,
    policy_tick_stride: int = 2,
    maintenance_enable: bool = False,
) -> SimConfig:
    runnercore4 = bool(runnercore4 or runnercore5 or runnercore6 or runnercore7 or runnercore9 or runnercore10 or runnercore11 or runnercore18)
    runnercore17 = bool(runnercore17 or runnercore18)
    runnercore16 = bool(runnercore16 or runnercore17)
    runnercore15 = bool(runnercore15 or runnercore16)
    runnercore13 = bool(runnercore13 or runnercore15)
    runnercore12 = bool(runnercore12 or runnercore13)
    runnercore11 = bool(runnercore11 or runnercore12)
    runnercore6 = bool(runnercore6 or runnercore9 or runnercore10 or runnercore11)
    runnercore7 = bool(runnercore7 or runnercore9 or runnercore10 or runnercore11)
    cfg = SimConfig(
        seed=4242,
        generations=1,
        steps_per_generation=int(steps),
        device="cpu",
        bank_device="cpu",
        behavior_log_enable=False,
        policy_deterministic_actions=True,
        policy_tick_stride=int(policy_tick_stride),
    )
    apply_performance_profile(cfg, FAST_EXPLORATORY_PROFILE_V2)
    cfg.phase_e_e1_enable = False
    cfg.phase_e_e2_enable = False
    cfg.epi_enable = False
    cfg.enable_cognitive_budget_ledger = False
    cfg.chemistry_functional_coupling_enable = False
    cfg.checkpoint_bundle_v2_period_gen = 0
    cfg.maintenance_enable = bool(maintenance_enable)
    cfg.maintenance_delay_ticks = 0
    cfg.perf_aggperf3_timing_buckets_enable = True
    if runnercore4:
        cfg.runnercore4_profile_enable = True
        cfg.runnercore4_candidate_enable = True
        cfg.runnercore4_allow_birth_boundary_agent_materialization = True
        cfg.runnercore4_structural_payload_cache_enable = True
        cfg.runnercore4_changed_parent_import_enable = True
    if runnercore5:
        cfg.runnercore5_profile_enable = True
        cfg.runnercore5_candidate_enable = True
        cfg.runnercore5_fail_closed_on_unsupported = True
        cfg.runnercore5_allow_birth_boundary_agent_materialization = True
        cfg.runnercore5_flat_self_policy_rng_enable = True
        cfg.runnercore5_direct_newborn_mlp_bank_attach_enable = True
        cfg.runnercore5_c1_canonical_boundary_fastpath_enable = True
    if runnercore6:
        cfg.runnercore4_profile_enable = True
        cfg.runnercore4_candidate_enable = True
        cfg.runnercore4_fail_closed_on_unsupported = True
        cfg.runnercore4_allow_birth_boundary_agent_materialization = True
        cfg.runnercore4_structural_payload_cache_enable = True
        cfg.runnercore4_changed_parent_import_enable = True
        cfg.runnercore5_profile_enable = True
        cfg.runnercore5_candidate_enable = True
        cfg.runnercore5_fail_closed_on_unsupported = True
        cfg.runnercore5_allow_birth_boundary_agent_materialization = True
        cfg.runnercore5_direct_newborn_mlp_bank_attach_enable = True
        cfg.runnercore6_profile_enable = True
        cfg.runnercore6_candidate_enable = True
        cfg.runnercore6_fail_closed_on_unsupported = True
        cfg.runnercore6_allow_birth_boundary_agent_materialization = True
        cfg.runnercore6_resident_newborn_row_attach_enable = True
        cfg.runnercore5_flat_self_policy_rng_enable = bool(runnercore7)
        cfg.runnercore5_c1_canonical_boundary_fastpath_enable = bool(runnercore7)
        cfg.runnercore6_trusted_canonical_genome_materialization_enable = bool(runnercore7)
        cfg.runnercore7_self_mlp_policy_source_alias_enable = bool(runnercore7)
        cfg.runnercore7_self_gru_policy_source_alias_enable = bool(runnercore7)
        cfg.runnercore7_direct_newborn_gru_bank_attach_enable = bool(runnercore7)
        cfg.runnercore7_static_reproduction_threshold_fastpath_enable = bool(runnercore7)
    if runnercore9:
        cfg.runnercore9_profile_enable = True
        cfg.runnercore9_candidate_enable = True
        cfg.runnercore9_fail_closed_on_unsupported = True
        cfg.runnercore9_c1_trace_enable = True
        cfg.runnercore9_c1_scalar_op_normalize_skip_enable = False
        cfg.runnercore9_lazy_newborn_materialization_enable = True
    if runnercore10:
        cfg.runnercore10_profile_enable = True
        cfg.runnercore10_candidate_enable = True
        cfg.runnercore10_fail_closed_on_unsupported = True
        cfg.runnercore10_typed_child_contract_enable = True
        cfg.runnercore10_defer_direct_attached_compat_enable = True
    if runnercore11:
        cfg.runnercore11_profile_enable = True
        cfg.runnercore11_candidate_enable = True
        cfg.runnercore11_fail_closed_on_unsupported = True
        cfg.runnercore11_c1_typed_rows_enable = True
        cfg.runnercore11_fast_weighted_food_spawn_enable = True
        cfg.runnercore11_generation_end_proxy_agents_enable = True
    if runnercore12:
        cfg.runnercore12_c1_payload_digest_reuse_enable = True
    if runnercore13:
        cfg.runnercore13_c1_projection_digest_sample_only_enable = True
    if runnercore15:
        cfg.runnercore15_c1_noop_normalize_skip_enable = True
    if runnercore16:
        cfg.runnercore16_c1_empty_structure_fastpath_enable = True
        cfg.runnercore16_c1_empty_structure_fastpath_fail_closed = True
    if runnercore17:
        cfg.runnercore17_resident_birth_parent_enable = True
        cfg.runnercore17_resident_birth_parent_fail_closed = True
    if runnercore18:
        cfg.runnercore18_resident_parent_minimal_refresh_enable = True
    if runnercore19:
        cfg.runnercore19_maintenance_arrays_enable = True
    if runnercore20:
        cfg.runnercore20_policy_observation_arrays_enable = True
    if runnercore21:
        cfg.runnercore21_cost_charge_surface_arrays_enable = True
    if runnercore22:
        cfg.runnercore22_full_birth_boundary_bridge_enable = True
    if runnercore23:
        cfg.runnercore23_jungle_predator_bridge_enable = True
    if runnercore24:
        cfg.runnercore24_native_jungle_predator_arrays_enable = True
    if runnercore25:
        cfg.runnercore25_full_birth_resident_newborn_attach_enable = True
    if runnercore26:
        cfg.runnercore26_vectorized_jungle_predator_sampling_enable = True
    if runnercore27:
        cfg.runnercore27_full_birth_cooldown_prefilter_enable = True
    if runnercore30:
        cfg.runnercore30_full_birth_branch_prefilter_enable = True
    if runnercore31:
        cfg.runnercore31_full_birth_static_dev_threshold_enable = True
    if runnercore32:
        cfg.runnercore32_full_birth_inline_grid_scan_enable = True
    if runnercore33:
        cfg.runnercore33_stride1_direct_action_enable = True
    if runnercore34:
        cfg.runnercore34_full_birth_resident_decision_enable = True
    if runnercore42:
        cfg.runnercore42_full_birth_patch_context_cache_enable = True
    if runnercore43:
        cfg.runnercore43_full_birth_threshold_tick_cache_enable = True
    if runnercore45:
        cfg.runnercore45_jungle_predator_context_cache_enable = True
    if runnercore47:
        cfg.runnercore47_developmental_trace_buffered_emit_enable = True
    if runnercore48:
        cfg.runnercore48_vectorized_chemistry_observation_enable = True
    if runnercore49:
        cfg.runnercore49_repro_viability_tick_cache_enable = True
    if runnercore50:
        cfg.runnercore50_spawn_first_valid_enable = True
    return cfg


def _prepare_sim(
    tmp_path: Path,
    *,
    runnercore4: bool = True,
    runnercore5: bool,
    runnercore6: bool = False,
    runnercore7: bool = False,
    runnercore9: bool = False,
    runnercore10: bool = False,
    runnercore11: bool = False,
    runnercore12: bool = False,
    runnercore13: bool = False,
    runnercore15: bool = False,
    runnercore16: bool = False,
    runnercore17: bool = False,
    runnercore18: bool = False,
    runnercore19: bool = False,
    runnercore20: bool = False,
    runnercore21: bool = False,
    runnercore22: bool = False,
    runnercore23: bool = False,
    runnercore24: bool = False,
    runnercore25: bool = False,
    runnercore26: bool = False,
    runnercore27: bool = False,
    runnercore30: bool = False,
    runnercore31: bool = False,
    runnercore32: bool = False,
    runnercore33: bool = False,
    runnercore34: bool = False,
    runnercore42: bool = False,
    runnercore43: bool = False,
    runnercore45: bool = False,
    runnercore47: bool = False,
    runnercore48: bool = False,
    runnercore49: bool = False,
    runnercore50: bool = False,
    steps: int = 5,
    policy_tick_stride: int = 2,
    maintenance_enable: bool = False,
    patch_world: bool = False,
    reproduce_requires_mate: bool = False,
) -> SimulationRunner:
    torch.manual_seed(12345)
    sim = SimulationRunner(
        env_cfg=_env_cfg(patch_world=patch_world, reproduce_requires_mate=reproduce_requires_mate),
        pop_cfg=PopConfig(population_size=4),
        sim_cfg=_sim_cfg(
            runnercore4=runnercore4,
            runnercore5=runnercore5,
            runnercore6=runnercore6,
            runnercore7=runnercore7,
            runnercore9=runnercore9,
            runnercore10=runnercore10,
            runnercore11=runnercore11,
            runnercore12=runnercore12,
            runnercore13=runnercore13,
            runnercore15=runnercore15,
            runnercore16=runnercore16,
            runnercore17=runnercore17,
            runnercore18=runnercore18,
            runnercore19=runnercore19,
            runnercore20=runnercore20,
            runnercore21=runnercore21,
            runnercore22=runnercore22,
            runnercore23=runnercore23,
            runnercore24=runnercore24,
            runnercore25=runnercore25,
            runnercore26=runnercore26,
            runnercore27=runnercore27,
            runnercore30=runnercore30,
            runnercore31=runnercore31,
            runnercore32=runnercore32,
            runnercore33=runnercore33,
            runnercore34=runnercore34,
            runnercore42=runnercore42,
            runnercore43=runnercore43,
            runnercore45=runnercore45,
            runnercore47=runnercore47,
            runnercore48=runnercore48,
            runnercore49=runnercore49,
            runnercore50=runnercore50,
            steps=steps,
            policy_tick_stride=policy_tick_stride,
            maintenance_enable=maintenance_enable,
        ),
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
    for name in _BIRTH_COUNTERS:
        setattr(sim, name, 0)
    return sim


def _sha_json(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


class _SeqRng:
    def __init__(self, values: list[float]) -> None:
        self._values = list(values)
        self._idx = 0

    def random(self) -> float:
        if self._idx >= len(self._values):
            return 0.5
        value = float(self._values[self._idx])
        self._idx += 1
        return value


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


def test_make_child_materialized_agent_does_not_keep_row_spec_sentinel(tmp_path: Path) -> None:
    sim = _prepare_sim(
        tmp_path / "materialized_child_sentinel",
        runnercore5=False,
        runnercore6=True,
        runnercore7=True,
        steps=1,
        policy_tick_stride=1,
    )
    parents = sorted(sim.pop.alive_agents(), key=lambda a: int(a.agent_id))[:2]
    buckets: dict[str, float] = {}
    child = sim.pop.make_child(parents[0], parents[1], 10.0, buckets=buckets)

    assert child is not None
    assert str(getattr(child, "typed_materialization_state", "")) == "compatibility_agent_materialized"


def _maintenance_fingerprint(sim: SimulationRunner) -> dict[str, Any]:
    alive = sorted(sim.pop.alive_agents(), key=lambda a: int(a.agent_id))
    stats = [getattr(a, "stats", None) for a in alive]
    return {
        "chemistry": [
            [round(float(x), 6) for x in np.asarray(getattr(a, "chemistry"), dtype=np.float32).reshape(-1).tolist()]
            for a in alive
        ],
        "maintenance": [
            [round(float(x), 6) for x in np.asarray(getattr(a, "maintenance"), dtype=np.float32).reshape(-1).tolist()]
            for a in alive
        ],
        "maintenance_requests": [int(getattr(st, "maintenance_requests", 0)) for st in stats],
        "maintenance_applied": [int(getattr(st, "maintenance_applied", 0)) for st in stats],
        "maintenance_energy_spent_total": [
            round(float(getattr(st, "maintenance_energy_spent_total", 0.0)), 6) for st in stats
        ],
        "maintenance_opportunity_cost_total": [
            round(float(getattr(st, "maintenance_opportunity_cost_total", 0.0)), 6) for st in stats
        ],
        "maintenance_applied_effect_total": [
            round(float(getattr(st, "maintenance_applied_effect_total", 0.0)), 6) for st in stats
        ],
        "generation_counters": {
            "ticks": int(getattr(sim, "_maintenance_tick_count_gen", 0)),
            "opportunities": int(getattr(sim, "_maintenance_action_opportunities_gen", 0)),
            "requests": int(getattr(sim, "_maintenance_schedule_requests_gen", 0)),
            "applied": int(getattr(sim, "_maintenance_apply_count_gen", 0)),
            "energy_cost": round(float(getattr(sim, "_maintenance_energy_cost_total_gen", 0.0)), 6),
            "opportunity_cost": round(float(getattr(sim, "_maintenance_opportunity_cost_total_gen", 0.0)), 6),
            "effect": round(float(getattr(sim, "_maintenance_effect_total_gen", 0.0)), 6),
            "annotation": dict(getattr(sim, "_maintenance_domination_annotation", {}) or {}),
        },
    }


def test_runnercore5_defaults_off() -> None:
    cfg = SimConfig()

    assert cfg.runnercore5_profile_enable is False
    assert cfg.runnercore5_candidate_enable is False
    assert cfg.runnercore5_native_enable is False
    assert cfg.runnercore5_allow_birth_boundary_agent_materialization is False
    assert cfg.runnercore5_flat_self_policy_rng_enable is False
    assert cfg.runnercore5_direct_newborn_mlp_bank_attach_enable is False
    assert cfg.runnercore5_c1_canonical_boundary_fastpath_enable is False
    assert cfg.runnercore6_profile_enable is False
    assert cfg.runnercore6_candidate_enable is False
    assert cfg.runnercore6_native_enable is False
    assert cfg.runnercore6_allow_birth_boundary_agent_materialization is False
    assert cfg.runnercore6_resident_newborn_row_attach_enable is False
    assert cfg.runnercore6_trusted_canonical_genome_materialization_enable is False
    assert cfg.runnercore7_self_mlp_policy_source_alias_enable is False
    assert cfg.runnercore7_self_gru_policy_source_alias_enable is False
    assert cfg.runnercore7_direct_newborn_gru_bank_attach_enable is False
    assert cfg.runnercore7_static_reproduction_threshold_fastpath_enable is False
    assert cfg.runnercore9_profile_enable is False
    assert cfg.runnercore9_candidate_enable is False
    assert cfg.runnercore9_native_enable is False
    assert cfg.runnercore9_c1_trace_enable is False
    assert cfg.runnercore9_c1_scalar_op_normalize_skip_enable is False
    assert cfg.runnercore9_lazy_newborn_materialization_enable is False
    assert cfg.runnercore10_profile_enable is False
    assert cfg.runnercore10_candidate_enable is False
    assert cfg.runnercore10_native_enable is False
    assert cfg.runnercore10_typed_child_contract_enable is False
    assert cfg.runnercore10_defer_direct_attached_compat_enable is False
    assert cfg.runnercore11_profile_enable is False
    assert cfg.runnercore11_candidate_enable is False
    assert cfg.runnercore11_native_enable is False
    assert cfg.runnercore11_c1_typed_rows_enable is False
    assert cfg.runnercore11_fast_weighted_food_spawn_enable is False
    assert cfg.runnercore11_generation_end_proxy_agents_enable is False
    assert cfg.runnercore12_c1_payload_digest_reuse_enable is False
    assert cfg.runnercore13_c1_projection_digest_sample_only_enable is False
    assert cfg.runnercore14_native_c1_projection_enable is False
    assert cfg.runnercore14_native_c1_projection_shadow_enable is False
    assert cfg.runnercore14_native_c1_projection_fail_closed is False
    assert cfg.runnercore14_native_c1_projection_library_path == ""
    assert cfg.runnercore15_c1_noop_normalize_skip_enable is False
    assert cfg.runnercore16_c1_empty_structure_fastpath_enable is False
    assert cfg.runnercore16_c1_empty_structure_fastpath_fail_closed is False
    assert cfg.runnercore17_resident_birth_parent_enable is False
    assert cfg.runnercore17_resident_birth_parent_fail_closed is False
    assert cfg.runnercore18_resident_parent_minimal_refresh_enable is False
    assert cfg.runnercore19_maintenance_arrays_enable is False
    assert cfg.runnercore20_policy_observation_arrays_enable is False
    assert cfg.runnercore21_cost_charge_surface_arrays_enable is False
    assert cfg.runnercore22_full_birth_boundary_bridge_enable is False
    assert cfg.runnercore23_jungle_predator_bridge_enable is False
    assert cfg.runnercore24_native_jungle_predator_arrays_enable is False
    assert cfg.runnercore25_full_birth_resident_newborn_attach_enable is False
    assert cfg.runnercore26_vectorized_jungle_predator_sampling_enable is False
    assert cfg.runnercore27_full_birth_cooldown_prefilter_enable is False
    assert cfg.runnercore30_full_birth_branch_prefilter_enable is False
    assert cfg.runnercore31_full_birth_static_dev_threshold_enable is False
    assert cfg.runnercore32_full_birth_inline_grid_scan_enable is False
    assert cfg.runnercore33_stride1_direct_action_enable is False
    assert cfg.runnercore34_full_birth_resident_decision_enable is False
    assert cfg.runnercore42_full_birth_patch_context_cache_enable is False
    assert cfg.runnercore43_full_birth_threshold_tick_cache_enable is False
    assert cfg.runnercore45_jungle_predator_context_cache_enable is False
    assert cfg.runnercore47_developmental_trace_buffered_emit_enable is False
    assert cfg.runnercore48_vectorized_chemistry_observation_enable is False
    assert cfg.runnercore49_repro_viability_tick_cache_enable is False
    assert cfg.runnercore50_spawn_first_valid_enable is False


def test_runnercore11_fast_weighted_food_spawn_preserves_grid_and_rng() -> None:
    cfg = EnvConfig(width=16, height=16, view_radius=1)
    cfg.enable_biomes = False
    cfg.food_spawn_per_step = 32
    cfg.max_food = 256
    cfg.patch_world_enable = False
    cfg.spatial_ecology_enable = False
    cfg.spatial_droughts_enable = False

    legacy = GridWorld(cfg, seed=123, run_dir=None, rng=np.random.default_rng(123))
    fast = GridWorld(cfg, seed=123, run_dir=None, rng=np.random.default_rng(123))
    legacy.grid[0, :8] = 1
    fast.grid[0, :8] = 1
    setattr(fast, "_runnercore11_fast_weighted_food_spawn_enable", True)

    for _ in range(12):
        legacy.spawn_food()
        fast.spawn_food()

    assert np.array_equal(legacy.grid, fast.grid)
    assert legacy.rng.bit_generator.state == fast.rng.bit_generator.state
    assert getattr(fast, "_runnercore11_fast_weighted_food_spawn_calls", 0) > 0
    assert getattr(fast, "_runnercore11_fast_weighted_food_spawn_draws", 0) > 0


def test_runnercore9_c1_trace_keeps_scalar_normalize_skip_blocked(tmp_path: Path) -> None:
    raw_payload = {
        "genes": {},
        "modules": [
            {
                "module_id": "m000",
                "module_type": "generic",
                "params": {"expression_scale": 1.0},
                "io": {
                    "inputs": [{"slot_id": "in", "value_type": "scalar"}],
                    "outputs": [{"slot_id": "out", "value_type": "scalar"}],
                    "caps": {"max_fanin": 6, "max_fanout": 6},
                },
                "latent_pos": [0.0, 0.0],
            },
            {
                "module_id": "m001",
                "module_type": "generic",
                "params": {"expression_scale": 1.0},
                "io": {
                    "inputs": [{"slot_id": "in", "value_type": "scalar"}],
                    "outputs": [{"slot_id": "out", "value_type": "scalar"}],
                    "caps": {"max_fanin": 6, "max_fanout": 6},
                },
                "latent_pos": [1.0, 0.0],
            },
        ],
        "connections": [
            {
                "connection_id": "c000",
                "src_module_id": "m000",
                "dst_module_id": "m001",
                "src_slot": "out",
                "dst_slot": "in",
                "value_type": "gate",
                "channel": "test",
                "weight": 1.0,
            }
        ],
    }
    payload = canonical_payload_from_genome_dict(raw_payload)
    draws = [0.65, 0.99, 0.20, 0.80, 0.40, 0.60]

    legacy, legacy_telemetry = apply_operator_pipeline(
        payload=payload,
        rng=_SeqRng(draws),
        rng_manager=None,
        max_ops=2,
        payload_is_canonical=True,
        skip_final_normalize=True,
    )
    trace: list[dict[str, Any]] = []
    candidate, candidate_telemetry = apply_operator_pipeline(
        payload=payload,
        rng=_SeqRng(draws),
        rng_manager=None,
        max_ops=2,
        payload_is_canonical=True,
        skip_final_normalize=True,
        skip_scalar_op_normalize=True,
        trace_records=trace,
    )

    assert _sha_json(candidate) == _sha_json(legacy)
    assert candidate_telemetry["scalar_normalize_skipped"] == 0
    assert legacy_telemetry["scalar_normalize_skipped"] == 0
    assert [row["operator_id"] for row in trace] == ["gate_drift", "promoter_expression_mutation"]
    assert not any(row["normalization_skipped"] for row in trace)


def test_runnercore16_empty_structure_fastpath_matches_runnercore15_payload_and_rng() -> None:
    payload = Genome.random(random.Random(777)).to_raw_dict()
    payload["modules"] = []
    payload["connections"] = []
    draws = [0.01, 0.99, 0.11, 0.22, 0.33, 0.44, 0.55, 0.66]
    legacy_rng = _SeqRng(draws)
    fast_rng = _SeqRng(draws)

    legacy, legacy_telemetry = apply_operator_pipeline(
        payload=payload,
        rng=legacy_rng,
        rng_manager=None,
        max_ops=2,
        payload_is_canonical=True,
        skip_final_normalize=True,
        skip_noop_normalize=True,
        emit_final_payload_digest=True,
    )
    candidate, candidate_telemetry = apply_operator_pipeline(
        payload=payload,
        rng=fast_rng,
        rng_manager=None,
        max_ops=2,
        payload_is_canonical=True,
        skip_final_normalize=True,
        skip_noop_normalize=True,
        empty_structure_fastpath=True,
        empty_structure_fastpath_fail_closed=True,
        emit_final_payload_digest=True,
    )

    assert _sha_json(candidate) == _sha_json(legacy)
    assert candidate_telemetry["final_payload_digest"] == legacy_telemetry["final_payload_digest"]
    assert candidate_telemetry["schedule_ids"] == legacy_telemetry["schedule_ids"]
    assert candidate_telemetry["applied_ids"] == legacy_telemetry["applied_ids"]
    assert candidate_telemetry["usage_counts"] == legacy_telemetry["usage_counts"]
    assert candidate_telemetry["noop_count"] == legacy_telemetry["noop_count"]
    assert legacy_telemetry["normalize_calls"] == 1
    assert candidate_telemetry["normalize_calls"] == 0
    assert candidate_telemetry["empty_structure_fastpath_hits"] == 1
    assert candidate_telemetry["empty_structure_fastpath_misses"] == 0
    assert fast_rng._idx == legacy_rng._idx


def test_runnercore5_flat_self_policy_matches_runnercore4_bytes_and_rng(tmp_path: Path) -> None:
    r4 = _prepare_sim(tmp_path / "r4", runnercore5=False, steps=1)
    r5 = _prepare_sim(tmp_path / "r5", runnercore5=True, steps=1)
    p4 = next(a for a in sorted(r4.pop.alive_agents(), key=lambda a: int(a.agent_id)) if hasattr(a.policy, "net"))
    p5 = next(a for a in sorted(r5.pop.alive_agents(), key=lambda a: int(a.agent_id)) if hasattr(a.policy, "net"))

    children4 = [r4.pop.make_child(p4, p4, 12.5, buckets={}) for _ in range(2)]
    children5 = [r5.pop.make_child(p5, p5, 12.5, buckets={}) for _ in range(2)]

    assert [c is not None for c in children5] == [True, True]
    assert [c.genome.canonical_hash() for c in children5 if c is not None] == [
        c.genome.canonical_hash() for c in children4 if c is not None
    ]
    assert [_policy_hash(c.policy) for c in children5 if c is not None] == [
        _policy_hash(c.policy) for c in children4 if c is not None
    ]
    assert _sha_json(r5.rng_manager.indices_snapshot()) == _sha_json(r4.rng_manager.indices_snapshot())

    stats = getattr(r5.sim_cfg, "_runnercore5_child_construction_stats", {})
    assert stats["self_policy_flat_rng_calls"] == 2
    assert stats["self_policy_flat_rng_elements"] > 0
    assert stats["c1_canonical_boundary_fastpath_calls"] == 2


def test_runnercore6_trusted_canonical_genome_materialization_matches_from_dict(tmp_path: Path) -> None:
    sim = _prepare_sim(tmp_path / "trusted_genome", runnercore5=True, steps=1)
    parent = next(a for a in sorted(sim.pop.alive_agents(), key=lambda a: int(a.agent_id)))
    child = sim.pop.make_child(parent, parent, 12.5, buckets={})
    assert child is not None
    payload = child.genome.canonical_payload()

    regular = Genome.from_dict(payload)
    trusted = Genome.from_trusted_canonical_payload(payload)

    assert trusted.canonical_hash() == regular.canonical_hash()
    assert trusted.to_dict() == regular.to_dict()


def test_runnercore5_live_churn_matches_runnercore4_with_direct_bank_attach(tmp_path: Path) -> None:
    r4 = _prepare_sim(tmp_path / "r4_live", runnercore5=False)
    r5 = _prepare_sim(tmp_path / "r5_live", runnercore5=True)

    def _forbidden(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("legacy action materializer called")

    for sim in (r4, r5):
        sim._batched_actions = _forbidden
        sim._batched_actions_k = _forbidden
        sim._batched_actions_strided = _forbidden

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
    result5 = r5._run_ticks(
        gen=0,
        buckets={},
        injury_recover_mult=1.0,
        deaths_this_gen=0,
        successful_births=0,
        failed_births=0,
        gen_peak_injury=0.0,
        rest_streaks={},
    )

    assert result5 == result4
    assert _extended_fingerprint(r5) == _extended_fingerprint(r4)
    summary = r5._runnercore5_child_construction_ownership_summary
    fallback = summary["fallback_counters"]
    assert fallback["legacy_batched_actions_calls"] == 0
    assert fallback["legacy_batched_actions_k_calls"] == 0
    assert fallback["legacy_batched_actions_strided_calls"] == 0
    assert fallback["legacy_action_cache_refill_calls"] == 0
    assert fallback["per_tick_agent_sync_rows"] == 0
    assert fallback["unsupported_fallback_count"] == 0

    stats = summary["runnercore5_child_construction"]
    assert stats["direct_mlp_bank_attach_rows"] > 0
    assert stats.get("direct_mlp_bank_attach_unsupported_rows", 0) == 0
    assert stats["direct_mlp_bank_attach_sync_rows_avoided"] == stats["direct_mlp_bank_attach_rows"]


def test_runnercore6_live_churn_matches_runnercore4_with_resident_newborn_attach(tmp_path: Path) -> None:
    r4 = _prepare_sim(tmp_path / "r4_live", runnercore5=False)
    r6 = _prepare_sim(tmp_path / "r6_live", runnercore5=False, runnercore6=True)

    def _forbidden(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("legacy action materializer called")

    for sim in (r4, r6):
        sim._batched_actions = _forbidden
        sim._batched_actions_k = _forbidden
        sim._batched_actions_strided = _forbidden

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
    result6 = r6._run_ticks(
        gen=0,
        buckets={},
        injury_recover_mult=1.0,
        deaths_this_gen=0,
        successful_births=0,
        failed_births=0,
        gen_peak_injury=0.0,
        rest_streaks={},
    )

    assert result6 == result4
    assert _extended_fingerprint(r6) == _extended_fingerprint(r4)
    summary = r6._runnercore6_resident_child_row_summary
    fallback = summary["fallback_counters"]
    assert fallback["legacy_batched_actions_calls"] == 0
    assert fallback["legacy_batched_actions_k_calls"] == 0
    assert fallback["legacy_batched_actions_strided_calls"] == 0
    assert fallback["legacy_action_cache_refill_calls"] == 0
    assert fallback["per_tick_agent_sync_rows"] == 0
    assert fallback["unsupported_fallback_count"] == 0

    stats = summary["runnercore6_resident_child_row"]
    assert stats["resident_newborn_attach_rows"] > 0
    assert stats.get("resident_newborn_attach_unsupported_rows", 0) == 0
    assert stats["generic_newborn_import_rows_avoided"] == stats["resident_newborn_attach_rows"]
    assert stats["runnercore6_child_row_specs_created"] >= stats["resident_newborn_attach_rows"]
    assert stats["runnercore6_resident_first_attaches"] == stats["resident_newborn_attach_rows"]
    assert stats["runnercore6_pre_attach_agent_materializations"] == 0
    assert stats["runnercore6_agent_materializations_deferred"] == stats["runnercore6_child_row_specs_created"]
    assert stats["runnercore6_post_attach_export_materializations"] == stats["resident_newborn_attach_rows"]
    assert stats["runnercore6_event_import_rows_avoided"] == stats["resident_newborn_attach_rows"]


def test_runnercore7_live_churn_matches_runnercore4_with_policy_source_alias(tmp_path: Path) -> None:
    r4 = _prepare_sim(tmp_path / "r4_live", runnercore5=False)
    r7 = _prepare_sim(tmp_path / "r7_live", runnercore5=False, runnercore6=True, runnercore7=True)

    def _forbidden(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("legacy action materializer called")

    for sim in (r4, r7):
        sim._batched_actions = _forbidden
        sim._batched_actions_k = _forbidden
        sim._batched_actions_strided = _forbidden

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
    result7 = r7._run_ticks(
        gen=0,
        buckets={},
        injury_recover_mult=1.0,
        deaths_this_gen=0,
        successful_births=0,
        failed_births=0,
        gen_peak_injury=0.0,
        rest_streaks={},
    )

    assert result7 == result4
    assert _extended_fingerprint(r7) == _extended_fingerprint(r4)
    assert _sha_json(r7.rng_manager.indices_snapshot()) == _sha_json(r4.rng_manager.indices_snapshot())

    summary = r7._runnercore6_resident_child_row_summary
    fallback = summary["fallback_counters"]
    assert fallback["legacy_batched_actions_calls"] == 0
    assert fallback["legacy_batched_actions_k_calls"] == 0
    assert fallback["legacy_batched_actions_strided_calls"] == 0
    assert fallback["legacy_action_cache_refill_calls"] == 0
    assert fallback["per_tick_agent_sync_rows"] == 0
    assert fallback["unsupported_fallback_count"] == 0

    stats = summary["runnercore6_resident_child_row"]
    assert stats["resident_newborn_attach_rows"] > 0
    assert stats.get("resident_newborn_attach_unsupported_rows", 0) == 0

    construction = summary["runnercore5_child_construction"]
    assert construction["runnercore7_self_policy_source_alias_calls"] > 0
    assert construction["runnercore7_self_policy_source_alias_rng_elements"] > 0
    assert construction["runnercore7_gru_policy_source_alias_calls"] > 0
    assert construction["runnercore7_gru_policy_init_rng_replay_elements"] > 0
    assert construction["direct_gru_bank_attach_rows"] > 0
    assert construction["direct_gru_bank_attach_sync_rows_avoided"] == construction["direct_gru_bank_attach_rows"]
    assert construction["runnercore7_static_reproduction_threshold_fastpath_calls"] > 0


def test_runnercore9_c1_scalar_normalize_skip_matches_runnercore7(tmp_path: Path) -> None:
    r7 = _prepare_sim(tmp_path / "r7_live", runnercore5=False, runnercore6=True, runnercore7=True)
    r9 = _prepare_sim(tmp_path / "r9_live", runnercore5=False, runnercore9=True)

    def _forbidden(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("legacy action materializer called")

    for sim in (r7, r9):
        sim._batched_actions = _forbidden
        sim._batched_actions_k = _forbidden
        sim._batched_actions_strided = _forbidden

    result7 = r7._run_ticks(
        gen=0,
        buckets={},
        injury_recover_mult=1.0,
        deaths_this_gen=0,
        successful_births=0,
        failed_births=0,
        gen_peak_injury=0.0,
        rest_streaks={},
    )
    result9 = r9._run_ticks(
        gen=0,
        buckets={},
        injury_recover_mult=1.0,
        deaths_this_gen=0,
        successful_births=0,
        failed_births=0,
        gen_peak_injury=0.0,
        rest_streaks={},
    )

    assert result9 == result7
    assert _extended_fingerprint(r9) == _extended_fingerprint(r7)
    assert _sha_json(r9.rng_manager.indices_snapshot()) == _sha_json(r7.rng_manager.indices_snapshot())

    summary = r9._runnercore9_c1_payload_or_lazy_newborn_summary
    fallback = summary["fallback_counters"]
    assert fallback["legacy_batched_actions_calls"] == 0
    assert fallback["legacy_batched_actions_k_calls"] == 0
    assert fallback["legacy_batched_actions_strided_calls"] == 0
    assert fallback["legacy_action_cache_refill_calls"] == 0
    assert fallback["per_tick_agent_sync_rows"] == 0
    assert fallback["unsupported_fallback_count"] == 0

    stats = summary["runnercore9_c1_payload_or_lazy_newborn"]
    assert stats["c1_operator_pipeline_calls"] > 0
    assert stats["c1_scalar_normalize_skipped"] >= 0
    assert stats["c1_trace_rows"] >= stats["c1_operator_pipeline_calls"]
    assert stats["lazy_newborn_contract"] == "resident_rows_defer_compat_agent_until_export"
    assert stats["newborn_agent_materializations_deferred"] > 0
    assert stats["resident_rows_without_agent_objects"] == stats["newborn_agent_materializations_deferred"]
    assert stats["generation_end_materializations"] + stats["newborn_agent_materializations_on_demand"] > 0
    sample = stats["c1_trace_sample"]
    assert isinstance(sample, list)
    assert sample
    assert all("normalized_payload_digest" in row for row in sample)


def test_runnercore9_lazy_materializes_policy_bank_sync_subset(tmp_path: Path) -> None:
    r7 = _prepare_sim(tmp_path / "r7_mixed_policy", runnercore5=False, runnercore6=True, runnercore7=True)
    r9 = _prepare_sim(tmp_path / "r9_mixed_policy", runnercore5=False, runnercore9=True)

    for sim in (r7, r9):
        for agent in sim.pop.alive_agents():
            agent.genome.memory_dim = 8
            agent.genome.use_gru = False
            agent.energy = 120.0
            agent.repro_cooldown = 0

    def _forbidden(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("legacy action materializer called")

    for sim in (r7, r9):
        sim._batched_actions = _forbidden
        sim._batched_actions_k = _forbidden
        sim._batched_actions_strided = _forbidden

    torch.manual_seed(20260708)
    result7 = r7._run_ticks(
        gen=0,
        buckets={},
        injury_recover_mult=1.0,
        deaths_this_gen=0,
        successful_births=0,
        failed_births=0,
        gen_peak_injury=0.0,
        rest_streaks={},
    )
    torch.manual_seed(20260708)
    result9 = r9._run_ticks(
        gen=0,
        buckets={},
        injury_recover_mult=1.0,
        deaths_this_gen=0,
        successful_births=0,
        failed_births=0,
        gen_peak_injury=0.0,
        rest_streaks={},
    )

    assert result9 == result7
    assert _extended_fingerprint(r9) == _extended_fingerprint(r7)
    assert _sha_json(r9.rng_manager.indices_snapshot()) == _sha_json(r7.rng_manager.indices_snapshot())

    summary = r9._runnercore9_c1_payload_or_lazy_newborn_summary
    fallback = summary["fallback_counters"]
    assert fallback["legacy_batched_actions_calls"] == 0
    assert fallback["legacy_action_cache_refill_calls"] == 0
    assert fallback["per_tick_agent_sync_rows"] == 0
    assert fallback["unsupported_fallback_count"] == 0

    stats = summary["runnercore9_c1_payload_or_lazy_newborn"]
    assert stats["policy_bank_sync_forced_materializations"] > 0
    assert stats["newborn_agent_materializations_deferred"] >= 0


def test_runnercore10_typed_child_contract_matches_runnercore7_and_defers_direct_rows(tmp_path: Path) -> None:
    r7 = _prepare_sim(tmp_path / "r7_live", runnercore5=False, runnercore6=True, runnercore7=True)
    r10 = _prepare_sim(tmp_path / "r10_live", runnercore5=False, runnercore10=True)

    def _forbidden(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("legacy action materializer called")

    for sim in (r7, r10):
        sim._batched_actions = _forbidden
        sim._batched_actions_k = _forbidden
        sim._batched_actions_strided = _forbidden

    result7 = r7._run_ticks(
        gen=0,
        buckets={},
        injury_recover_mult=1.0,
        deaths_this_gen=0,
        successful_births=0,
        failed_births=0,
        gen_peak_injury=0.0,
        rest_streaks={},
    )
    result10 = r10._run_ticks(
        gen=0,
        buckets={},
        injury_recover_mult=1.0,
        deaths_this_gen=0,
        successful_births=0,
        failed_births=0,
        gen_peak_injury=0.0,
        rest_streaks={},
    )

    assert result10 == result7
    assert _extended_fingerprint(r10) == _extended_fingerprint(r7)
    assert _sha_json(r10.rng_manager.indices_snapshot()) == _sha_json(r7.rng_manager.indices_snapshot())

    summary = r10._runnercore10_typed_child_genome_policy_row_summary
    fallback = summary["fallback_counters"]
    assert fallback["legacy_batched_actions_calls"] == 0
    assert fallback["legacy_batched_actions_k_calls"] == 0
    assert fallback["legacy_batched_actions_strided_calls"] == 0
    assert fallback["legacy_action_cache_refill_calls"] == 0
    assert fallback["per_tick_agent_sync_rows"] == 0
    assert fallback["unsupported_fallback_count"] == 0

    stats = summary["runnercore10_typed_child"]
    assert stats["typed_child_specs_created"] > 0
    assert stats["typed_child_specs_attached"] > 0
    assert stats["typed_child_policy_rows_attached"] == stats["typed_child_specs_attached"]
    assert stats["typed_child_direct_mlp_rows"] > 0
    assert stats["typed_child_direct_gru_rows"] > 0
    assert stats["agent_materialization_avoided"] == (
        stats["typed_child_direct_mlp_rows"] + stats["typed_child_direct_gru_rows"]
    )
    assert stats["resident_rows_without_agent_objects"] == stats["agent_materialization_avoided"]
    assert stats["unsupported_typed_child_fallback_count"] == 0
    assert stats["hidden_agent_sync_rows"] == 0


def test_runnercore11_c1_typed_row_matches_runnercore7_and_defers_genome_materialization(tmp_path: Path) -> None:
    r7 = _prepare_sim(tmp_path / "r7_live", runnercore5=False, runnercore6=True, runnercore7=True)
    r11 = _prepare_sim(tmp_path / "r11_live", runnercore5=False, runnercore11=True)

    def _forbidden(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("legacy action materializer called")

    for sim in (r7, r11):
        sim._batched_actions = _forbidden
        sim._batched_actions_k = _forbidden
        sim._batched_actions_strided = _forbidden

    result7 = r7._run_ticks(
        gen=0,
        buckets={},
        injury_recover_mult=1.0,
        deaths_this_gen=0,
        successful_births=0,
        failed_births=0,
        gen_peak_injury=0.0,
        rest_streaks={},
    )
    result11 = r11._run_ticks(
        gen=0,
        buckets={},
        injury_recover_mult=1.0,
        deaths_this_gen=0,
        successful_births=0,
        failed_births=0,
        gen_peak_injury=0.0,
        rest_streaks={},
    )

    assert result11 == result7
    assert _extended_fingerprint(r11) == _extended_fingerprint(r7)
    assert _sha_json(r11.rng_manager.indices_snapshot()) == _sha_json(r7.rng_manager.indices_snapshot())

    summary = r11._runnercore11_c1_typed_genome_row_summary
    fallback = summary["fallback_counters"]
    assert fallback["legacy_batched_actions_calls"] == 0
    assert fallback["legacy_batched_actions_k_calls"] == 0
    assert fallback["legacy_batched_actions_strided_calls"] == 0
    assert fallback["legacy_action_cache_refill_calls"] == 0
    assert fallback["per_tick_agent_sync_rows"] == 0
    assert fallback["unsupported_fallback_count"] == 0

    stats = summary["runnercore11_c1_typed_genome_row"]
    assert stats["c1_operator_pipeline_calls"] > 0
    assert stats["c1_typed_rows_created"] > 0
    assert stats["c1_typed_rows_attached"] > 0
    assert stats["policy_rows_attached"] == stats["c1_typed_rows_attached"]
    assert stats["c1_projection_trace_rows"] >= stats["c1_typed_rows_created"]
    assert stats["genome_from_dict_avoided"] == stats["c1_typed_rows_created"]
    assert stats["genome_materialization_avoided"] == stats["c1_typed_rows_created"]
    assert stats["genome_materialization_forced"] >= 0
    assert stats["generation_end_proxy_agents_created"] >= 0
    assert stats["generation_end_proxy_genome_materialization_avoided"] >= 0
    assert stats["c1_typed_rows_missing_field_fail_closed"] == 0
    assert stats["c1_typed_rows_unsupported_kind"] == 0
    assert stats["hidden_agent_sync_rows"] == 0
    assert stats["per_tick_agent_sync_rows"] == 0
    assert stats["c1_projection_trace_sample"]
    assert "c1_payload_hash" in stats["c1_projection_trace_sample"][0]


def test_runnercore12_c1_payload_digest_reuse_matches_runnercore11(tmp_path: Path) -> None:
    r11 = _prepare_sim(tmp_path / "r11_live", runnercore5=False, runnercore11=True)
    r12 = _prepare_sim(tmp_path / "r12_live", runnercore5=False, runnercore12=True)

    def _forbidden(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("legacy action materializer called")

    for sim in (r11, r12):
        sim._batched_actions = _forbidden
        sim._batched_actions_k = _forbidden
        sim._batched_actions_strided = _forbidden

    result11 = r11._run_ticks(
        gen=0,
        buckets={},
        injury_recover_mult=1.0,
        deaths_this_gen=0,
        successful_births=0,
        failed_births=0,
        gen_peak_injury=0.0,
        rest_streaks={},
    )
    result12 = r12._run_ticks(
        gen=0,
        buckets={},
        injury_recover_mult=1.0,
        deaths_this_gen=0,
        successful_births=0,
        failed_births=0,
        gen_peak_injury=0.0,
        rest_streaks={},
    )

    assert result12 == result11
    assert _extended_fingerprint(r12) == _extended_fingerprint(r11)
    assert _sha_json(r12.rng_manager.indices_snapshot()) == _sha_json(r11.rng_manager.indices_snapshot())

    summary = r12._runnercore11_c1_typed_genome_row_summary
    fallback = summary["fallback_counters"]
    assert fallback["legacy_batched_actions_calls"] == 0
    assert fallback["legacy_batched_actions_k_calls"] == 0
    assert fallback["legacy_batched_actions_strided_calls"] == 0
    assert fallback["legacy_action_cache_refill_calls"] == 0
    assert fallback["per_tick_agent_sync_rows"] == 0
    assert fallback["unsupported_fallback_count"] == 0

    stats = summary["runnercore11_c1_typed_genome_row"]
    assert stats["runnercore12_c1_payload_digest_reuse_enabled"] > 0
    assert stats["runnercore12_c1_payload_digest_reuse_rows"] == stats["c1_typed_rows_created"]
    assert stats["runnercore12_c1_payload_digest_reuse_fallbacks"] == 0


def test_runnercore13_c1_projection_digest_sample_only_matches_runnercore12(tmp_path: Path) -> None:
    r12 = _prepare_sim(tmp_path / "r12_live", runnercore5=False, runnercore12=True, steps=8)
    r13 = _prepare_sim(tmp_path / "r13_live", runnercore5=False, runnercore13=True, steps=8)

    def _forbidden(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("legacy action materializer called")

    for sim in (r12, r13):
        sim._batched_actions = _forbidden
        sim._batched_actions_k = _forbidden
        sim._batched_actions_strided = _forbidden

    result12 = r12._run_ticks(
        gen=0,
        buckets={},
        injury_recover_mult=1.0,
        deaths_this_gen=0,
        successful_births=0,
        failed_births=0,
        gen_peak_injury=0.0,
        rest_streaks={},
    )
    result13 = r13._run_ticks(
        gen=0,
        buckets={},
        injury_recover_mult=1.0,
        deaths_this_gen=0,
        successful_births=0,
        failed_births=0,
        gen_peak_injury=0.0,
        rest_streaks={},
    )

    assert result13 == result12
    assert _extended_fingerprint(r13) == _extended_fingerprint(r12)
    assert _sha_json(r13.rng_manager.indices_snapshot()) == _sha_json(r12.rng_manager.indices_snapshot())

    summary = r13._runnercore11_c1_typed_genome_row_summary
    fallback = summary["fallback_counters"]
    assert fallback["legacy_batched_actions_calls"] == 0
    assert fallback["legacy_batched_actions_k_calls"] == 0
    assert fallback["legacy_batched_actions_strided_calls"] == 0
    assert fallback["legacy_action_cache_refill_calls"] == 0
    assert fallback["per_tick_agent_sync_rows"] == 0
    assert fallback["unsupported_fallback_count"] == 0

    stats = summary["runnercore11_c1_typed_genome_row"]
    assert stats["runnercore12_c1_payload_digest_reuse_rows"] == stats["c1_typed_rows_created"]
    assert stats["runnercore12_c1_payload_digest_reuse_fallbacks"] == 0
    assert stats["runnercore13_c1_projection_digest_sample_only_enabled"] == stats["c1_typed_rows_created"]
    assert stats["runnercore13_c1_projection_digest_full_rows"] == len(stats["c1_projection_trace_sample"])
    assert stats["runnercore13_c1_projection_digest_skipped_rows"] >= 0
    assert (
        stats["runnercore13_c1_projection_digest_full_rows"]
        + stats["runnercore13_c1_projection_digest_skipped_rows"]
        == stats["c1_typed_rows_created"]
    )
    assert stats["c1_projection_trace_sample"]
    assert stats["c1_projection_trace_sample"][0]["projection_digest"]


def test_runnercore15_c1_noop_normalize_skip_matches_runnercore13(tmp_path: Path) -> None:
    r13 = _prepare_sim(tmp_path / "r13_live", runnercore5=False, runnercore13=True, steps=8)
    r15 = _prepare_sim(tmp_path / "r15_live", runnercore5=False, runnercore15=True, steps=8)

    def _forbidden(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("legacy action materializer called")

    for sim in (r13, r15):
        sim._batched_actions = _forbidden
        sim._batched_actions_k = _forbidden
        sim._batched_actions_strided = _forbidden

    result13 = r13._run_ticks(
        gen=0,
        buckets={},
        injury_recover_mult=1.0,
        deaths_this_gen=0,
        successful_births=0,
        failed_births=0,
        gen_peak_injury=0.0,
        rest_streaks={},
    )
    result15 = r15._run_ticks(
        gen=0,
        buckets={},
        injury_recover_mult=1.0,
        deaths_this_gen=0,
        successful_births=0,
        failed_births=0,
        gen_peak_injury=0.0,
        rest_streaks={},
    )

    assert result15 == result13
    assert _extended_fingerprint(r15) == _extended_fingerprint(r13)
    assert _sha_json(r15.rng_manager.indices_snapshot()) == _sha_json(r13.rng_manager.indices_snapshot())

    summary = r15._runnercore11_c1_typed_genome_row_summary
    fallback = summary["fallback_counters"]
    assert fallback["legacy_batched_actions_calls"] == 0
    assert fallback["legacy_batched_actions_k_calls"] == 0
    assert fallback["legacy_batched_actions_strided_calls"] == 0
    assert fallback["legacy_action_cache_refill_calls"] == 0
    assert fallback["per_tick_agent_sync_rows"] == 0
    assert fallback["unsupported_fallback_count"] == 0

    stats = summary["runnercore11_c1_typed_genome_row"]
    assert stats["runnercore12_c1_payload_digest_reuse_rows"] == stats["c1_typed_rows_created"]
    assert stats["runnercore12_c1_payload_digest_reuse_fallbacks"] == 0
    assert stats["runnercore13_c1_projection_digest_sample_only_enabled"] == stats["c1_typed_rows_created"]
    assert stats["runnercore15_c1_noop_normalize_skip_enabled"] == stats["c1_typed_rows_created"]
    assert stats["runnercore15_c1_noop_normalize_skipped"] > 0


def test_runnercore16_empty_structure_fastpath_matches_runnercore15_live(tmp_path: Path) -> None:
    r15 = _prepare_sim(tmp_path / "r15_live", runnercore5=False, runnercore15=True, steps=8)
    r16 = _prepare_sim(tmp_path / "r16_live", runnercore5=False, runnercore16=True, steps=8)

    def _forbidden(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("legacy action materializer called")

    for sim in (r15, r16):
        sim._batched_actions = _forbidden
        sim._batched_actions_k = _forbidden
        sim._batched_actions_strided = _forbidden

    result15 = r15._run_ticks(
        gen=0,
        buckets={},
        injury_recover_mult=1.0,
        deaths_this_gen=0,
        successful_births=0,
        failed_births=0,
        gen_peak_injury=0.0,
        rest_streaks={},
    )
    result16 = r16._run_ticks(
        gen=0,
        buckets={},
        injury_recover_mult=1.0,
        deaths_this_gen=0,
        successful_births=0,
        failed_births=0,
        gen_peak_injury=0.0,
        rest_streaks={},
    )

    assert result16 == result15
    assert _extended_fingerprint(r16) == _extended_fingerprint(r15)
    assert _sha_json(r16.rng_manager.indices_snapshot()) == _sha_json(r15.rng_manager.indices_snapshot())

    summary = r16._runnercore11_c1_typed_genome_row_summary
    fallback = summary["fallback_counters"]
    assert fallback["legacy_batched_actions_calls"] == 0
    assert fallback["legacy_batched_actions_k_calls"] == 0
    assert fallback["legacy_batched_actions_strided_calls"] == 0
    assert fallback["legacy_action_cache_refill_calls"] == 0
    assert fallback["per_tick_agent_sync_rows"] == 0
    assert fallback["unsupported_fallback_count"] == 0

    stats = summary["runnercore11_c1_typed_genome_row"]
    assert stats["runnercore16_c1_empty_structure_fastpath_enabled"] == stats["c1_typed_rows_created"]
    assert stats["runnercore16_c1_empty_structure_fastpath_hits"] == stats["c1_typed_rows_created"]
    assert stats["runnercore16_c1_empty_structure_fastpath_misses"] == 0
    assert stats["runnercore16_c1_empty_structure_fastpath_noop_draws"] > 0


def test_runnercore17_resident_birth_parent_matches_runnercore16_live(tmp_path: Path) -> None:
    r16 = _prepare_sim(tmp_path / "r16_live", runnercore5=False, runnercore16=True, steps=8)
    r17 = _prepare_sim(tmp_path / "r17_live", runnercore5=False, runnercore17=True, steps=8)

    def _forbidden(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("legacy action materializer called")

    for sim in (r16, r17):
        sim._batched_actions = _forbidden
        sim._batched_actions_k = _forbidden
        sim._batched_actions_strided = _forbidden

    result16 = r16._run_ticks(
        gen=0,
        buckets={},
        injury_recover_mult=1.0,
        deaths_this_gen=0,
        successful_births=0,
        failed_births=0,
        gen_peak_injury=0.0,
        rest_streaks={},
    )
    result17 = r17._run_ticks(
        gen=0,
        buckets={},
        injury_recover_mult=1.0,
        deaths_this_gen=0,
        successful_births=0,
        failed_births=0,
        gen_peak_injury=0.0,
        rest_streaks={},
    )

    assert result17 == result16
    assert _extended_fingerprint(r17) == _extended_fingerprint(r16)
    assert _sha_json(r17.rng_manager.indices_snapshot()) == _sha_json(r16.rng_manager.indices_snapshot())

    summary = r17._runnercore11_c1_typed_genome_row_summary
    fallback = summary["fallback_counters"]
    assert fallback["legacy_batched_actions_calls"] == 0
    assert fallback["legacy_batched_actions_k_calls"] == 0
    assert fallback["legacy_batched_actions_strided_calls"] == 0
    assert fallback["legacy_action_cache_refill_calls"] == 0
    assert fallback["per_tick_agent_sync_rows"] == 0
    assert fallback["unsupported_fallback_count"] == 0

    stats = summary["runnercore11_c1_typed_genome_row"]
    assert stats["runnercore17_resident_birth_parent_candidates"] > 0
    assert stats["runnercore17_resident_birth_parent_specs_used"] > 0
    assert stats["runnercore17_resident_birth_parent_state_refreshes"] == stats[
        "runnercore17_resident_birth_parent_specs_used"
    ]
    assert stats["runnercore17_resident_birth_parent_materializations_avoided"] == stats[
        "runnercore17_resident_birth_parent_specs_used"
    ]
    assert stats["runnercore17_resident_birth_parent_parent_import_syncs"] > 0
    assert stats["runnercore17_resident_birth_parent_missing_specs"] == 0
    assert stats["runnercore17_resident_birth_parent_unsupported_fallback_count"] == 0


def test_runnercore18_minimal_resident_parent_refresh_matches_runnercore17_live(tmp_path: Path) -> None:
    r17 = _prepare_sim(tmp_path / "r17_live", runnercore5=False, runnercore17=True, steps=8)
    r18 = _prepare_sim(tmp_path / "r18_live", runnercore5=False, runnercore18=True, steps=8)

    def _forbidden(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("legacy action materializer called")

    for sim in (r17, r18):
        sim._batched_actions = _forbidden
        sim._batched_actions_k = _forbidden
        sim._batched_actions_strided = _forbidden

    result17 = r17._run_ticks(
        gen=0,
        buckets={},
        injury_recover_mult=1.0,
        deaths_this_gen=0,
        successful_births=0,
        failed_births=0,
        gen_peak_injury=0.0,
        rest_streaks={},
    )
    result18 = r18._run_ticks(
        gen=0,
        buckets={},
        injury_recover_mult=1.0,
        deaths_this_gen=0,
        successful_births=0,
        failed_births=0,
        gen_peak_injury=0.0,
        rest_streaks={},
    )

    assert result18 == result17
    assert _extended_fingerprint(r18) == _extended_fingerprint(r17)
    assert _sha_json(r18.rng_manager.indices_snapshot()) == _sha_json(r17.rng_manager.indices_snapshot())

    summary = r18._runnercore11_c1_typed_genome_row_summary
    fallback = summary["fallback_counters"]
    assert fallback["legacy_batched_actions_calls"] == 0
    assert fallback["legacy_batched_actions_k_calls"] == 0
    assert fallback["legacy_batched_actions_strided_calls"] == 0
    assert fallback["legacy_action_cache_refill_calls"] == 0
    assert fallback["per_tick_agent_sync_rows"] == 0
    assert fallback["unsupported_fallback_count"] == 0

    stats = summary["runnercore11_c1_typed_genome_row"]
    assert stats["runnercore17_resident_birth_parent_specs_used"] > 0
    assert stats["runnercore18_resident_parent_minimal_refresh_enabled"] > 0
    assert stats["runnercore18_resident_parent_minimal_refreshes"] == stats[
        "runnercore17_resident_birth_parent_specs_used"
    ]
    assert stats["runnercore18_resident_parent_full_refreshes"] > 0
    assert stats["runnercore18_resident_parent_full_refreshes"] <= stats[
        "runnercore18_resident_parent_minimal_refreshes"
    ]
    assert stats["runnercore18_resident_parent_unsupported_fallback_count"] == 0


def test_runnercore18_stride1_matches_runnercore4_live(tmp_path: Path) -> None:
    r4 = _prepare_sim(tmp_path / "r4_stride1_live", runnercore5=False, steps=8, policy_tick_stride=1)
    r18 = _prepare_sim(tmp_path / "r18_stride1_live", runnercore5=False, runnercore18=True, steps=8, policy_tick_stride=1)

    def _forbidden(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("legacy action materializer called")

    for sim in (r4, r18):
        sim._batched_actions = _forbidden
        sim._batched_actions_k = _forbidden
        sim._batched_actions_strided = _forbidden

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
    result18 = r18._run_ticks(
        gen=0,
        buckets={},
        injury_recover_mult=1.0,
        deaths_this_gen=0,
        successful_births=0,
        failed_births=0,
        gen_peak_injury=0.0,
        rest_streaks={},
    )

    assert result18 == result4
    assert _extended_fingerprint(r18) == _extended_fingerprint(r4)
    assert _sha_json(r18.rng_manager.indices_snapshot()) == _sha_json(r4.rng_manager.indices_snapshot())

    summary = r18._runnercore11_c1_typed_genome_row_summary
    fallback = summary["fallback_counters"]
    assert fallback["legacy_batched_actions_calls"] == 0
    assert fallback["legacy_batched_actions_k_calls"] == 0
    assert fallback["legacy_batched_actions_strided_calls"] == 0
    assert fallback["legacy_action_cache_refill_calls"] == 0
    assert fallback["per_tick_agent_sync_rows"] == 0
    assert fallback["unsupported_fallback_count"] == 0

    stats = summary["runnercore11_c1_typed_genome_row"]
    assert stats["runnercore18_resident_parent_minimal_refresh_enabled"] > 0
    assert stats["runnercore18_resident_parent_unsupported_fallback_count"] == 0


def test_runnercore19_maintenance_arrays_match_legacy_live(tmp_path: Path) -> None:
    legacy = _prepare_sim(
        tmp_path / "legacy_maintenance_live",
        runnercore4=False,
        runnercore5=False,
        maintenance_enable=True,
        steps=8,
        policy_tick_stride=1,
    )
    candidate = _prepare_sim(
        tmp_path / "runnercore19_maintenance_live",
        runnercore5=False,
        runnercore18=True,
        runnercore19=True,
        maintenance_enable=True,
        steps=8,
        policy_tick_stride=1,
    )

    def _forbidden(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("legacy action materializer called")

    candidate._batched_actions = _forbidden
    candidate._batched_actions_k = _forbidden
    candidate._batched_actions_strided = _forbidden

    result_legacy = legacy._run_ticks(
        gen=0,
        buckets={},
        injury_recover_mult=1.0,
        deaths_this_gen=0,
        successful_births=0,
        failed_births=0,
        gen_peak_injury=0.0,
        rest_streaks={},
    )
    result_candidate = candidate._run_ticks(
        gen=0,
        buckets={},
        injury_recover_mult=1.0,
        deaths_this_gen=0,
        successful_births=0,
        failed_births=0,
        gen_peak_injury=0.0,
        rest_streaks={},
    )

    assert result_candidate == result_legacy
    assert _extended_fingerprint(candidate) == _extended_fingerprint(legacy)
    assert _maintenance_fingerprint(candidate) == _maintenance_fingerprint(legacy)
    assert _sha_json(candidate.rng_manager.indices_snapshot()) == _sha_json(legacy.rng_manager.indices_snapshot())
    assert _maintenance_fingerprint(candidate)["generation_counters"]["requests"] > 0

    summary = candidate._runnercore11_c1_typed_genome_row_summary
    fallback = summary["fallback_counters"]
    assert fallback["legacy_batched_actions_calls"] == 0
    assert fallback["legacy_batched_actions_k_calls"] == 0
    assert fallback["legacy_batched_actions_strided_calls"] == 0
    assert fallback["legacy_action_cache_refill_calls"] == 0
    assert fallback["per_tick_agent_sync_rows"] == 0
    assert fallback["unsupported_fallback_count"] == 0


def test_runnercore20_policy_observation_arrays_match_legacy_actions(tmp_path: Path) -> None:
    legacy = _prepare_sim(
        tmp_path / "legacy_policy_observation",
        runnercore4=False,
        runnercore5=False,
        steps=1,
        policy_tick_stride=1,
    )
    candidate = _prepare_sim(
        tmp_path / "runnercore20_policy_observation",
        runnercore5=False,
        runnercore18=True,
        runnercore20=True,
        steps=1,
        policy_tick_stride=1,
    )
    for sim in (legacy, candidate):
        sim.sim_cfg.chemistry_functional_coupling_enable = True
        sim.sim_cfg.enable_evolvable_sensor_channel_mask = True
        sim.sim_cfg.force_default_sensor_radius = True
        for idx, agent in enumerate(sorted(sim.pop.alive_agents(), key=lambda a: int(a.agent_id))):
            agent.genome.sensor_channel_mask = int(minimal_sensor_channel_mask())
            agent.chemistry = np.asarray(
                (0.15 + 0.05 * idx, 0.30, 0.40, 0.65, 0.10, 0.95),
                dtype=np.float32,
            )
        sync_fn = getattr(sim, "_phase3_pre_start_generation_sync", None)
        if callable(sync_fn):
            sync_fn(list(sim.pop.alive_agents()))

    alive = sorted(legacy.pop.alive_agents(), key=lambda a: int(a.agent_id))
    ids = np.asarray([int(a.agent_id) for a in alive], dtype=np.int32)

    legacy_actions = legacy._batched_actions(alive, buckets={}).astype(np.int32, copy=False)
    state = import_generation_state(candidate, generation=0, profile="supported_fast_runner_v2")
    resident_actions = _compute_actions_k(candidate, state, ids, k=1, buckets={})[:, 0].astype(np.int32, copy=False)

    assert resident_actions.tolist() == legacy_actions.tolist()
    for agent in alive:
        slot = int(state.slot_for_agent_id(int(agent.agent_id)))
        assert slot >= 0
        assert round(float(state.chemistry_observation_fidelity[slot]), 6) == round(
            float(agent.stats.chemistry_observation_fidelity),
            6,
        )
        assert round(float(state.chemistry_observation_bias[slot]), 6) == round(
            float(agent.stats.chemistry_observation_bias),
            6,
        )


def test_runnercore48_vectorized_chemistry_observation_matches_scalar(tmp_path: Path, monkeypatch: Any) -> None:
    scalar = _prepare_sim(
        tmp_path / "runnercore48_scalar_chemistry_observation",
        runnercore5=False,
        runnercore18=True,
        runnercore20=True,
        steps=1,
        policy_tick_stride=1,
    )
    candidate = _prepare_sim(
        tmp_path / "runnercore48_vectorized_chemistry_observation",
        runnercore5=False,
        runnercore18=True,
        runnercore20=True,
        runnercore48=True,
        steps=1,
        policy_tick_stride=1,
    )
    for sim in (scalar, candidate):
        sim.sim_cfg.chemistry_functional_coupling_enable = True
        sim.sim_cfg.runnercore3_profile_enable = True
        for idx, agent in enumerate(sorted(sim.pop.alive_agents(), key=lambda a: int(a.agent_id))):
            agent.chemistry = np.asarray(
                (
                    0.10 + 0.07 * idx,
                    0.20 + 0.03 * idx,
                    0.35 + 0.05 * idx,
                    0.65 - 0.04 * idx,
                    0.15 + 0.02 * idx,
                    0.95 - 0.10 * idx,
                ),
                dtype=np.float32,
            )
        sync_fn = getattr(sim, "_phase3_pre_start_generation_sync", None)
        if callable(sync_fn):
            sync_fn(list(sim.pop.alive_agents()))

    alive = sorted(scalar.pop.alive_agents(), key=lambda a: int(a.agent_id))
    ids = np.asarray([int(a.agent_id) for a in alive], dtype=np.int32)

    scalar_state = import_generation_state(scalar, generation=0, profile="supported_fast_runner_v2")
    scalar_actions = _compute_actions_k(scalar, scalar_state, ids, k=1, buckets={}).astype(np.int32, copy=False)
    scalar_slots = np.asarray([scalar_state.slot_for_agent_id(int(aid)) for aid in ids.tolist()], dtype=np.int32)
    scalar_fidelity = np.asarray(scalar_state.chemistry_observation_fidelity[scalar_slots], dtype=np.float32).copy()
    scalar_bias = np.asarray(scalar_state.chemistry_observation_bias[scalar_slots], dtype=np.float32).copy()

    def _forbidden_scalar_helper(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("scalar chemistry observation helper called")

    monkeypatch.setattr(action_cache_mod, "observation_fidelity_and_bias_from_chem", _forbidden_scalar_helper)

    candidate_state = import_generation_state(candidate, generation=0, profile="supported_fast_runner_v2")
    candidate_actions = _compute_actions_k(candidate, candidate_state, ids, k=1, buckets={}).astype(np.int32, copy=False)
    candidate_slots = np.asarray([candidate_state.slot_for_agent_id(int(aid)) for aid in ids.tolist()], dtype=np.int32)

    assert candidate_actions.tolist() == scalar_actions.tolist()
    np.testing.assert_array_equal(candidate_state.chemistry_observation_fidelity[candidate_slots], scalar_fidelity)
    np.testing.assert_array_equal(candidate_state.chemistry_observation_bias[candidate_slots], scalar_bias)
    assert (
        candidate_state.runnercore3_profile_counts["runnercore48_chemistry_observation_vectorized_rows"]
        == int(ids.shape[0])
    )


def test_runnercore21_cost_charge_surface_arrays_match_accepted_scalar(tmp_path: Path) -> None:
    sim = _prepare_sim(
        tmp_path / "runnercore21_cost_charge_surface",
        runnercore5=False,
        runnercore18=True,
        runnercore20=True,
        runnercore21=True,
        steps=1,
        policy_tick_stride=1,
    )
    sim.sim_cfg.enable_cognitive_budget_ledger = True
    sim.sim_cfg.env_hotpath_cost_charge_surface_enable = True
    sim.sim_cfg.enable_experimental_sensor_radius_cost_charging = True
    sim.sim_cfg.enable_experimental_prediction_cost_charging = True
    sim.sim_cfg.enable_experimental_developmental_cost_charging = True
    sim.sim_cfg.force_default_sensor_radius = True
    sim.sim_cfg.prediction_cost_base = 0.01
    sim.sim_cfg.prediction_cost_per_lr = 0.02
    sim.sim_cfg.sensor_radius_cost_base = 0.001
    sim.sim_cfg.sensor_radius_cost_per_cell = 0.001
    sim.sim_cfg.sensor_radius_cost_per_cell_channel = 0.001
    sim.sim_cfg.developmental_cognition_cost_base = 0.0005
    sim.sim_cfg.developmental_cognition_cost_per_unexpressed_capacity = 0.001
    sim.sim_cfg.maturation_cost_per_tick = 0.0005
    sim.sim_cfg.developmental_maturity_age = 3

    expected: dict[int, tuple[float, float]] = {}
    for idx, agent in enumerate(sorted(sim.pop.alive_agents(), key=lambda a: int(a.agent_id))):
        agent.genome.sensor_radius = 1 + (idx % 3)
        agent.genome.pred_lr = 0.05 + 0.01 * idx
        agent.stats.survival_steps = idx
        surface = express_cost_charge_surface(
            agent.genome,
            agent=agent,
            env_cfg=sim.env_cfg,
            sim_cfg=sim.sim_cfg,
        )
        total = compute_env_cost_charge_scalar_from_surface(surface, sim_cfg=sim.sim_cfg)
        expected[int(agent.agent_id)] = (
            round(float(total) - float(surface.maturation_cost_charged), 9),
            round(float(surface.maturation_cost_charged), 9),
        )

    state = import_generation_state(sim, generation=0, profile="supported_fast_runner_v2")
    for agent in sorted(sim.pop.alive_agents(), key=lambda a: int(a.agent_id)):
        slot = int(state.slot_for_agent_id(int(agent.agent_id)))
        assert slot >= 0
        base, maturation = expected[int(agent.agent_id)]
        assert np.isclose(float(state.cost_charge_scalar[slot]), base, atol=1e-7)
        assert np.isclose(float(state.cost_charge_maturation_scalar[slot]), maturation, atol=1e-7)
    assert any(base > 0.0 or maturation > 0.0 for base, maturation in expected.values())


def test_runnercore21_cost_charge_surface_arrays_match_legacy_live(tmp_path: Path) -> None:
    legacy = _prepare_sim(
        tmp_path / "legacy_cost_charge_surface_live",
        runnercore4=False,
        runnercore5=False,
        steps=5,
        policy_tick_stride=1,
    )
    candidate = _prepare_sim(
        tmp_path / "runnercore21_cost_charge_surface_live",
        runnercore5=False,
        runnercore18=True,
        runnercore20=True,
        runnercore21=True,
        steps=5,
        policy_tick_stride=1,
    )

    for sim in (legacy, candidate):
        sim.env_cfg.reproduce_try_prob = 0.0
        sim.sim_cfg.enable_cognitive_budget_ledger = True
        sim.sim_cfg.env_hotpath_cost_charge_surface_enable = True
        sim.sim_cfg.enable_experimental_sensor_radius_cost_charging = True
        sim.sim_cfg.enable_experimental_prediction_cost_charging = True
        sim.sim_cfg.enable_experimental_developmental_cost_charging = True
        sim.sim_cfg.force_default_sensor_radius = True
        sim.sim_cfg.prediction_cost_base = 0.01
        sim.sim_cfg.prediction_cost_per_lr = 0.02
        sim.sim_cfg.sensor_radius_cost_base = 0.001
        sim.sim_cfg.sensor_radius_cost_per_cell = 0.001
        sim.sim_cfg.sensor_radius_cost_per_cell_channel = 0.001
        sim.sim_cfg.developmental_cognition_cost_base = 0.0005
        sim.sim_cfg.developmental_cognition_cost_per_unexpressed_capacity = 0.001
        sim.sim_cfg.maturation_cost_per_tick = 0.0005
        sim.sim_cfg.developmental_maturity_age = 3
        for idx, agent in enumerate(sorted(sim.pop.alive_agents(), key=lambda a: int(a.agent_id))):
            agent.energy = 60.0 - float(idx)
            agent.genome.sensor_radius = 1 + (idx % 3)
            agent.genome.pred_lr = 0.05 + 0.01 * idx
            agent.genome.reproduction_threshold = 1_000.0
            agent.stats.survival_steps = idx
            agent.stats.early_survival_steps = idx
        sync_fn = getattr(sim, "_phase3_pre_start_generation_sync", None)
        if callable(sync_fn):
            sync_fn(list(sim.pop.alive_agents()))

    def _forbidden(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("legacy action materializer called")

    candidate._batched_actions = _forbidden
    candidate._batched_actions_k = _forbidden
    candidate._batched_actions_strided = _forbidden

    legacy_buckets: dict[str, float] = {}
    candidate_buckets: dict[str, float] = {}
    result_legacy = legacy._run_ticks(
        gen=0,
        buckets=legacy_buckets,
        injury_recover_mult=1.0,
        deaths_this_gen=0,
        successful_births=0,
        failed_births=0,
        gen_peak_injury=0.0,
        rest_streaks={},
    )
    result_candidate = candidate._run_ticks(
        gen=0,
        buckets=candidate_buckets,
        injury_recover_mult=1.0,
        deaths_this_gen=0,
        successful_births=0,
        failed_births=0,
        gen_peak_injury=0.0,
        rest_streaks={},
    )

    assert result_candidate == result_legacy
    candidate_fp = _extended_fingerprint(candidate)
    legacy_fp = _extended_fingerprint(legacy)
    assert np.allclose(candidate_fp.pop("energy"), legacy_fp.pop("energy"), atol=5e-6)
    assert np.allclose(candidate_fp.pop("health"), legacy_fp.pop("health"), atol=5e-6)
    assert candidate_fp == legacy_fp
    assert _sha_json(candidate.rng_manager.indices_snapshot()) == _sha_json(legacy.rng_manager.indices_snapshot())
    assert float(candidate_buckets.get("runnercore21_cost_charge_total", 0.0)) > 0.0

    summary = candidate._runnercore11_c1_typed_genome_row_summary
    fallback = summary["fallback_counters"]
    assert fallback["legacy_batched_actions_calls"] == 0
    assert fallback["legacy_batched_actions_k_calls"] == 0
    assert fallback["legacy_batched_actions_strided_calls"] == 0
    assert fallback["legacy_action_cache_refill_calls"] == 0
    assert fallback["per_tick_agent_sync_rows"] == 0
    assert fallback["unsupported_fallback_count"] == 0


def test_runnercore22_full_birth_boundary_bridge_matches_legacy_mate_patch_live(tmp_path: Path) -> None:
    legacy = _prepare_sim(
        tmp_path / "legacy_mate_patch_birth_live",
        runnercore4=False,
        runnercore5=False,
        steps=2,
        policy_tick_stride=1,
        patch_world=True,
        reproduce_requires_mate=True,
    )
    candidate = _prepare_sim(
        tmp_path / "runnercore22_mate_patch_birth_live",
        runnercore5=False,
        runnercore18=True,
        runnercore20=True,
        runnercore22=True,
        steps=2,
        policy_tick_stride=1,
        patch_world=True,
        reproduce_requires_mate=True,
    )

    forced_positions = [(2, 2), (2, 3), (5, 5), (5, 4)]
    for sim in (legacy, candidate):
        sim.sim_cfg.chemistry_functional_coupling_enable = True
        sim.sim_cfg.chemistry_repro_viability_enable = True
        sim.env_cfg.reproduce_try_prob = 1.0
        sim.env_cfg.reproduce_parent_energy_share = 0.10
        sim.env_cfg.reproduce_cooldown_steps = 2
        sim.env_cfg.jungle_predator_pressure_enable = False
        alive = sorted(sim.pop.alive_agents(), key=lambda a: int(a.agent_id))
        for agent in alive:
            sim.env.remove_agent(int(agent.agent_id))
        for idx, (agent, pos) in enumerate(zip(alive, forced_positions)):
            agent.energy = 120.0 - float(idx)
            agent.repro_cooldown = 0
            agent.genome.reproduction_threshold = 1.0
            agent.chemistry = np.asarray((1.0, 1.0, 0.0, 0.0, 0.5, 0.5), dtype=np.float32)
            agent.stats.survival_steps = idx
            sim.env.add_agent(int(agent.agent_id), pos=pos)
        sync_fn = getattr(sim, "_phase3_pre_start_generation_sync", None)
        if callable(sync_fn):
            sync_fn(list(sim.pop.alive_agents()))

    def _forbidden(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("legacy action materializer called")

    candidate._batched_actions = _forbidden
    candidate._batched_actions_k = _forbidden
    candidate._batched_actions_strided = _forbidden

    legacy_buckets: dict[str, float] = {}
    candidate_buckets: dict[str, float] = {}
    result_legacy = legacy._run_ticks(
        gen=0,
        buckets=legacy_buckets,
        injury_recover_mult=1.0,
        deaths_this_gen=0,
        successful_births=0,
        failed_births=0,
        gen_peak_injury=0.0,
        rest_streaks={},
    )
    result_candidate = candidate._run_ticks(
        gen=0,
        buckets=candidate_buckets,
        injury_recover_mult=1.0,
        deaths_this_gen=0,
        successful_births=0,
        failed_births=0,
        gen_peak_injury=0.0,
        rest_streaks={},
    )

    assert result_candidate == result_legacy
    candidate_fp = _extended_fingerprint(candidate)
    legacy_fp = _extended_fingerprint(legacy)
    assert np.allclose(candidate_fp.pop("energy"), legacy_fp.pop("energy"), atol=5e-6)
    assert np.allclose(candidate_fp.pop("health"), legacy_fp.pop("health"), atol=5e-6)
    assert candidate_fp == legacy_fp
    assert _sha_json(candidate.rng_manager.indices_snapshot()) == _sha_json(legacy.rng_manager.indices_snapshot())
    assert len(candidate.pop.alive_agents()) > len(forced_positions)
    assert len(getattr(candidate, "_patch_birth_contexts_gen", []) or []) == len(
        getattr(legacy, "_patch_birth_contexts_gen", []) or []
    )

    summary = candidate._runnercore11_c1_typed_genome_row_summary
    fallback = summary["fallback_counters"]
    assert fallback["legacy_batched_actions_calls"] == 0
    assert fallback["legacy_batched_actions_k_calls"] == 0
    assert fallback["legacy_batched_actions_strided_calls"] == 0
    assert fallback["legacy_action_cache_refill_calls"] == 0
    assert fallback["unsupported_fallback_count"] == 0
    profile_counts = summary["runnercore3_profile"]["counts"]
    assert profile_counts["runnercore22_full_birth_boundary_bridge_export_rows"] > 0
    assert profile_counts["runnercore22_full_birth_boundary_bridge_import_rows"] > 0


def test_runnercore25_full_birth_row_spec_bridge_matches_legacy_mate_patch_live(tmp_path: Path) -> None:
    legacy = _prepare_sim(
        tmp_path / "legacy_mate_patch_birth_r25_live",
        runnercore4=False,
        runnercore5=False,
        steps=2,
        policy_tick_stride=1,
        patch_world=True,
        reproduce_requires_mate=True,
    )
    candidate = _prepare_sim(
        tmp_path / "runnercore25_mate_patch_birth_live",
        runnercore5=False,
        runnercore18=True,
        runnercore20=True,
        runnercore22=True,
        runnercore25=True,
        steps=2,
        policy_tick_stride=1,
        patch_world=True,
        reproduce_requires_mate=True,
    )

    forced_positions = [(2, 2), (2, 3), (5, 5), (5, 4)]
    for sim in (legacy, candidate):
        sim.sim_cfg.chemistry_functional_coupling_enable = True
        sim.sim_cfg.chemistry_repro_viability_enable = True
        sim.env_cfg.reproduce_try_prob = 1.0
        sim.env_cfg.reproduce_parent_energy_share = 0.10
        sim.env_cfg.reproduce_cooldown_steps = 2
        sim.env_cfg.jungle_predator_pressure_enable = False
        alive = sorted(sim.pop.alive_agents(), key=lambda a: int(a.agent_id))
        for agent in alive:
            sim.env.remove_agent(int(agent.agent_id))
        for idx, (agent, pos) in enumerate(zip(alive, forced_positions)):
            agent.energy = 120.0 - float(idx)
            agent.repro_cooldown = 0
            agent.genome.reproduction_threshold = 1.0
            agent.chemistry = np.asarray((1.0, 1.0, 0.0, 0.0, 0.5, 0.5), dtype=np.float32)
            agent.stats.survival_steps = idx
            sim.env.add_agent(int(agent.agent_id), pos=pos)
        sync_fn = getattr(sim, "_phase3_pre_start_generation_sync", None)
        if callable(sync_fn):
            sync_fn(list(sim.pop.alive_agents()))

    def _forbidden(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("legacy action materializer called")

    candidate._batched_actions = _forbidden
    candidate._batched_actions_k = _forbidden
    candidate._batched_actions_strided = _forbidden

    legacy_buckets: dict[str, float] = {}
    candidate_buckets: dict[str, float] = {}
    result_legacy = legacy._run_ticks(
        gen=0,
        buckets=legacy_buckets,
        injury_recover_mult=1.0,
        deaths_this_gen=0,
        successful_births=0,
        failed_births=0,
        gen_peak_injury=0.0,
        rest_streaks={},
    )
    result_candidate = candidate._run_ticks(
        gen=0,
        buckets=candidate_buckets,
        injury_recover_mult=1.0,
        deaths_this_gen=0,
        successful_births=0,
        failed_births=0,
        gen_peak_injury=0.0,
        rest_streaks={},
    )

    assert result_candidate == result_legacy
    candidate_fp = _extended_fingerprint(candidate)
    legacy_fp = _extended_fingerprint(legacy)
    assert np.allclose(candidate_fp.pop("energy"), legacy_fp.pop("energy"), atol=5e-6)
    assert np.allclose(candidate_fp.pop("health"), legacy_fp.pop("health"), atol=5e-6)
    assert candidate_fp == legacy_fp
    assert _sha_json(candidate.rng_manager.indices_snapshot()) == _sha_json(legacy.rng_manager.indices_snapshot())
    assert len(candidate.pop.alive_agents()) > len(forced_positions)
    assert len(getattr(candidate, "_patch_birth_contexts_gen", []) or []) == len(
        getattr(legacy, "_patch_birth_contexts_gen", []) or []
    )

    summary = candidate._runnercore11_c1_typed_genome_row_summary
    fallback = summary["fallback_counters"]
    assert fallback["legacy_batched_actions_calls"] == 0
    assert fallback["legacy_batched_actions_k_calls"] == 0
    assert fallback["legacy_batched_actions_strided_calls"] == 0
    assert fallback["legacy_action_cache_refill_calls"] == 0
    assert fallback["unsupported_fallback_count"] == 0
    profile_counts = summary["runnercore3_profile"]["counts"]
    assert profile_counts["runnercore22_full_birth_boundary_bridge_export_rows"] > 0
    assert profile_counts["runnercore22_full_birth_boundary_bridge_import_rows"] > 0
    assert profile_counts["runnercore25_full_birth_row_specs_created"] > 0
    assert profile_counts["runnercore25_full_birth_row_specs_applied"] > 0
    assert profile_counts["runnercore25_full_birth_newborn_attach_rows"] > 0


def test_runnercore27_full_birth_cooldown_prefilter_matches_runnercore22_live(tmp_path: Path) -> None:
    baseline = _prepare_sim(
        tmp_path / "runnercore22_mate_patch_birth_cooldown_baseline_live",
        runnercore5=False,
        runnercore18=True,
        runnercore20=True,
        runnercore22=True,
        steps=3,
        policy_tick_stride=1,
        patch_world=True,
        reproduce_requires_mate=True,
    )
    candidate = _prepare_sim(
        tmp_path / "runnercore27_mate_patch_birth_cooldown_prefilter_live",
        runnercore5=False,
        runnercore18=True,
        runnercore20=True,
        runnercore22=True,
        runnercore27=True,
        steps=3,
        policy_tick_stride=1,
        patch_world=True,
        reproduce_requires_mate=True,
    )

    forced_positions = [(2, 2), (2, 3), (5, 5), (5, 4)]
    cooldowns = [3, 0, 3, 0]
    for sim in (baseline, candidate):
        sim.sim_cfg.chemistry_functional_coupling_enable = True
        sim.sim_cfg.chemistry_repro_viability_enable = True
        sim.sim_cfg.lineage_reproduction_funnel_write_events = False
        sim.env_cfg.reproduce_try_prob = 1.0
        sim.env_cfg.reproduce_parent_energy_share = 0.10
        sim.env_cfg.reproduce_cooldown_steps = 2
        sim.env_cfg.jungle_predator_pressure_enable = False
        alive = sorted(sim.pop.alive_agents(), key=lambda a: int(a.agent_id))
        for agent in alive:
            sim.env.remove_agent(int(agent.agent_id))
        for idx, (agent, pos) in enumerate(zip(alive, forced_positions)):
            agent.energy = 120.0 - float(idx)
            agent.repro_cooldown = int(cooldowns[idx])
            agent.genome.reproduction_threshold = 1.0
            agent.chemistry = np.asarray((1.0, 1.0, 0.0, 0.0, 0.5, 0.5), dtype=np.float32)
            agent.stats.survival_steps = idx
            sim.env.add_agent(int(agent.agent_id), pos=pos)
        sync_fn = getattr(sim, "_phase3_pre_start_generation_sync", None)
        if callable(sync_fn):
            sync_fn(list(sim.pop.alive_agents()))

    def _forbidden(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("legacy action materializer called")

    candidate._batched_actions = _forbidden
    candidate._batched_actions_k = _forbidden
    candidate._batched_actions_strided = _forbidden

    baseline_buckets: dict[str, float] = {}
    candidate_buckets: dict[str, float] = {}
    result_baseline = baseline._run_ticks(
        gen=0,
        buckets=baseline_buckets,
        injury_recover_mult=1.0,
        deaths_this_gen=0,
        successful_births=0,
        failed_births=0,
        gen_peak_injury=0.0,
        rest_streaks={},
    )
    result_candidate = candidate._run_ticks(
        gen=0,
        buckets=candidate_buckets,
        injury_recover_mult=1.0,
        deaths_this_gen=0,
        successful_births=0,
        failed_births=0,
        gen_peak_injury=0.0,
        rest_streaks={},
    )

    assert result_candidate == result_baseline
    candidate_fp = _extended_fingerprint(candidate)
    baseline_fp = _extended_fingerprint(baseline)
    assert np.allclose(candidate_fp.pop("energy"), baseline_fp.pop("energy"), atol=5e-6)
    assert np.allclose(candidate_fp.pop("health"), baseline_fp.pop("health"), atol=5e-6)
    assert candidate_fp == baseline_fp
    assert _sha_json(candidate.rng_manager.indices_snapshot()) == _sha_json(baseline.rng_manager.indices_snapshot())
    assert np.allclose(candidate.env.rng.random(16), baseline.env.rng.random(16), atol=0.0)
    assert len(candidate.pop.alive_agents()) == len(baseline.pop.alive_agents())
    assert len(getattr(candidate, "_patch_birth_contexts_gen", []) or []) == len(
        getattr(baseline, "_patch_birth_contexts_gen", []) or []
    )

    summary = candidate._runnercore11_c1_typed_genome_row_summary
    fallback = summary["fallback_counters"]
    assert fallback["legacy_batched_actions_calls"] == 0
    assert fallback["legacy_batched_actions_k_calls"] == 0
    assert fallback["legacy_batched_actions_strided_calls"] == 0
    assert fallback["legacy_action_cache_refill_calls"] == 0
    assert fallback["unsupported_fallback_count"] == 0
    profile_counts = summary["runnercore3_profile"]["counts"]
    assert profile_counts["runnercore22_full_birth_boundary_bridge_export_rows"] > 0
    assert profile_counts["runnercore22_full_birth_boundary_bridge_import_rows"] > 0
    assert profile_counts["runnercore27_full_birth_cooldown_prefilter_considered_rows"] > 0
    assert profile_counts["runnercore27_full_birth_cooldown_prefilter_rows"] > 0


def test_runnercore30_full_birth_isolated_branch_prefilter_matches_runnercore27_live(tmp_path: Path) -> None:
    baseline = _prepare_sim(
        tmp_path / "runnercore27_mate_patch_birth_isolated_baseline_live",
        runnercore5=False,
        runnercore18=True,
        runnercore20=True,
        runnercore22=True,
        runnercore27=True,
        steps=1,
        policy_tick_stride=1,
        patch_world=True,
        reproduce_requires_mate=True,
    )
    candidate = _prepare_sim(
        tmp_path / "runnercore30_mate_patch_birth_isolated_branch_prefilter_live",
        runnercore5=False,
        runnercore18=True,
        runnercore20=True,
        runnercore22=True,
        runnercore27=True,
        runnercore30=True,
        steps=1,
        policy_tick_stride=1,
        patch_world=True,
        reproduce_requires_mate=True,
    )

    forced_positions = [(0, 0), (0, 7), (7, 0), (7, 7)]
    for sim in (baseline, candidate):
        sim.sim_cfg.chemistry_functional_coupling_enable = True
        sim.sim_cfg.chemistry_repro_viability_enable = True
        sim.sim_cfg.lineage_reproduction_funnel_write_events = False
        sim.env_cfg.reproduce_try_prob = 1.0
        sim.env_cfg.reproduce_parent_energy_share = 0.10
        sim.env_cfg.reproduce_cooldown_steps = 2
        sim.env_cfg.jungle_predator_pressure_enable = False
        alive = sorted(sim.pop.alive_agents(), key=lambda a: int(a.agent_id))
        for agent in alive:
            sim.env.remove_agent(int(agent.agent_id))
        for idx, (agent, pos) in enumerate(zip(alive, forced_positions)):
            agent.energy = 120.0 - float(idx)
            agent.repro_cooldown = 0
            agent.genome.reproduction_threshold = 1.0
            agent.chemistry = np.asarray((1.0, 1.0, 0.0, 0.0, 0.5, 0.5), dtype=np.float32)
            agent.stats.survival_steps = idx
            sim.env.add_agent(int(agent.agent_id), pos=pos)
        sync_fn = getattr(sim, "_phase3_pre_start_generation_sync", None)
        if callable(sync_fn):
            sync_fn(list(sim.pop.alive_agents()))

    def _forbidden(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("legacy action materializer called")

    candidate._batched_actions = _forbidden
    candidate._batched_actions_k = _forbidden
    candidate._batched_actions_strided = _forbidden

    baseline_buckets: dict[str, float] = {}
    candidate_buckets: dict[str, float] = {}
    result_baseline = baseline._run_ticks(
        gen=0,
        buckets=baseline_buckets,
        injury_recover_mult=1.0,
        deaths_this_gen=0,
        successful_births=0,
        failed_births=0,
        gen_peak_injury=0.0,
        rest_streaks={},
    )
    result_candidate = candidate._run_ticks(
        gen=0,
        buckets=candidate_buckets,
        injury_recover_mult=1.0,
        deaths_this_gen=0,
        successful_births=0,
        failed_births=0,
        gen_peak_injury=0.0,
        rest_streaks={},
    )

    assert result_candidate == result_baseline
    candidate_fp = _extended_fingerprint(candidate)
    baseline_fp = _extended_fingerprint(baseline)
    assert np.allclose(candidate_fp.pop("energy"), baseline_fp.pop("energy"), atol=5e-6)
    assert np.allclose(candidate_fp.pop("health"), baseline_fp.pop("health"), atol=5e-6)
    assert candidate_fp == baseline_fp
    assert _sha_json(candidate.rng_manager.indices_snapshot()) == _sha_json(baseline.rng_manager.indices_snapshot())
    assert np.allclose(candidate.env.rng.random(16), baseline.env.rng.random(16), atol=0.0)
    assert np.allclose(candidate.np_rng.random(16), baseline.np_rng.random(16), atol=0.0)
    assert getattr(candidate, "_dev_stage_transition_count_gen", 0) == getattr(
        baseline,
        "_dev_stage_transition_count_gen",
        0,
    )
    assert getattr(candidate, "_dev_stage_eval_stage_counts_gen", {}) == getattr(
        baseline,
        "_dev_stage_eval_stage_counts_gen",
        {},
    )
    assert len(getattr(candidate, "_patch_birth_contexts_gen", []) or []) == len(
        getattr(baseline, "_patch_birth_contexts_gen", []) or []
    )

    baseline_profile = baseline._runnercore11_c1_typed_genome_row_summary["runnercore3_profile"]["counts"]
    candidate_profile = candidate._runnercore11_c1_typed_genome_row_summary["runnercore3_profile"]["counts"]
    assert candidate_profile["runnercore30_full_birth_branch_prefilter_rows"] > 0
    assert candidate_profile["runnercore30_full_birth_branch_prefilter_isolated_no_mate_rows"] > 0
    assert candidate_profile["runnercore22_full_birth_boundary_bridge_export_rows"] < baseline_profile[
        "runnercore22_full_birth_boundary_bridge_export_rows"
    ]
    fallback = candidate._runnercore11_c1_typed_genome_row_summary["fallback_counters"]
    assert fallback["legacy_batched_actions_calls"] == 0
    assert fallback["legacy_batched_actions_k_calls"] == 0
    assert fallback["legacy_batched_actions_strided_calls"] == 0
    assert fallback["legacy_action_cache_refill_calls"] == 0
    assert fallback["unsupported_fallback_count"] == 0


def test_runnercore31_full_birth_static_dev_threshold_matches_runnercore27_live(tmp_path: Path) -> None:
    baseline = _prepare_sim(
        tmp_path / "runnercore27_mate_patch_birth_static_dev_baseline_live",
        runnercore5=False,
        runnercore18=True,
        runnercore20=True,
        runnercore22=True,
        runnercore27=True,
        steps=3,
        policy_tick_stride=1,
        patch_world=True,
        reproduce_requires_mate=True,
    )
    candidate = _prepare_sim(
        tmp_path / "runnercore31_mate_patch_birth_static_dev_threshold_live",
        runnercore5=False,
        runnercore18=True,
        runnercore20=True,
        runnercore22=True,
        runnercore27=True,
        runnercore31=True,
        steps=3,
        policy_tick_stride=1,
        patch_world=True,
        reproduce_requires_mate=True,
    )

    forced_positions = [(2, 2), (2, 3), (5, 5), (5, 4)]
    cooldowns = [3, 0, 3, 0]
    for sim in (baseline, candidate):
        sim.sim_cfg.chemistry_functional_coupling_enable = True
        sim.sim_cfg.chemistry_repro_viability_enable = True
        sim.sim_cfg.lineage_reproduction_funnel_write_events = False
        sim.env_cfg.reproduce_try_prob = 1.0
        sim.env_cfg.reproduce_parent_energy_share = 0.10
        sim.env_cfg.reproduce_cooldown_steps = 2
        sim.env_cfg.jungle_predator_pressure_enable = False
        alive = sorted(sim.pop.alive_agents(), key=lambda a: int(a.agent_id))
        for agent in alive:
            sim.env.remove_agent(int(agent.agent_id))
        for idx, (agent, pos) in enumerate(zip(alive, forced_positions)):
            agent.energy = 120.0 - float(idx)
            agent.repro_cooldown = int(cooldowns[idx])
            agent.genome.reproduction_threshold = 1.0
            agent.chemistry = np.asarray((1.0, 1.0, 0.0, 0.0, 0.5, 0.5), dtype=np.float32)
            agent.stats.survival_steps = idx
            sim.env.add_agent(int(agent.agent_id), pos=pos)
        sync_fn = getattr(sim, "_phase3_pre_start_generation_sync", None)
        if callable(sync_fn):
            sync_fn(list(sim.pop.alive_agents()))

    def _forbidden(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("legacy action materializer called")

    candidate._batched_actions = _forbidden
    candidate._batched_actions_k = _forbidden
    candidate._batched_actions_strided = _forbidden

    baseline_buckets: dict[str, float] = {}
    candidate_buckets: dict[str, float] = {}
    result_baseline = baseline._run_ticks(
        gen=0,
        buckets=baseline_buckets,
        injury_recover_mult=1.0,
        deaths_this_gen=0,
        successful_births=0,
        failed_births=0,
        gen_peak_injury=0.0,
        rest_streaks={},
    )
    result_candidate = candidate._run_ticks(
        gen=0,
        buckets=candidate_buckets,
        injury_recover_mult=1.0,
        deaths_this_gen=0,
        successful_births=0,
        failed_births=0,
        gen_peak_injury=0.0,
        rest_streaks={},
    )

    assert result_candidate == result_baseline
    candidate_fp = _extended_fingerprint(candidate)
    baseline_fp = _extended_fingerprint(baseline)
    assert np.allclose(candidate_fp.pop("energy"), baseline_fp.pop("energy"), atol=5e-6)
    assert np.allclose(candidate_fp.pop("health"), baseline_fp.pop("health"), atol=5e-6)
    assert candidate_fp == baseline_fp
    assert _sha_json(candidate.rng_manager.indices_snapshot()) == _sha_json(baseline.rng_manager.indices_snapshot())
    assert np.allclose(candidate.env.rng.random(16), baseline.env.rng.random(16), atol=0.0)
    assert np.allclose(candidate.np_rng.random(16), baseline.np_rng.random(16), atol=0.0)
    assert getattr(candidate, "_dev_stage_transition_count_gen", 0) == getattr(
        baseline,
        "_dev_stage_transition_count_gen",
        0,
    )
    assert getattr(candidate, "_dev_stage_eval_stage_counts_gen", {}) == getattr(
        baseline,
        "_dev_stage_eval_stage_counts_gen",
        {},
    )
    assert len(getattr(candidate, "_patch_birth_contexts_gen", []) or []) == len(
        getattr(baseline, "_patch_birth_contexts_gen", []) or []
    )
    stats = getattr(candidate.sim_cfg, "_runnercore6_resident_child_row_stats", {}) or {}
    assert int(stats.get("runnercore31_full_birth_static_dev_threshold_hits", 0)) > 0
    assert int(stats.get("runnercore31_full_birth_static_dev_threshold_synthetic_lookups", 0)) > 0


def test_runnercore33_stride1_direct_action_matches_runnercore31_live(tmp_path: Path) -> None:
    baseline = _prepare_sim(
        tmp_path / "runnercore31_stride1_action_baseline_live",
        runnercore5=False,
        runnercore18=True,
        runnercore20=True,
        runnercore22=True,
        runnercore27=True,
        runnercore31=True,
        steps=3,
        policy_tick_stride=1,
        patch_world=True,
        reproduce_requires_mate=True,
    )
    candidate = _prepare_sim(
        tmp_path / "runnercore33_stride1_direct_action_live",
        runnercore5=False,
        runnercore18=True,
        runnercore20=True,
        runnercore22=True,
        runnercore27=True,
        runnercore31=True,
        runnercore33=True,
        steps=3,
        policy_tick_stride=1,
        patch_world=True,
        reproduce_requires_mate=True,
    )

    for sim in (baseline, candidate):
        sim.sim_cfg.chemistry_functional_coupling_enable = True
        sim.sim_cfg.chemistry_repro_viability_enable = True
        sim.sim_cfg.lineage_reproduction_funnel_write_events = False
        sim.env_cfg.reproduce_try_prob = 1.0
        sim.env_cfg.reproduce_parent_energy_share = 0.10
        sim.env_cfg.reproduce_cooldown_steps = 2
        sim.env_cfg.jungle_predator_pressure_enable = False
        sync_fn = getattr(sim, "_phase3_pre_start_generation_sync", None)
        if callable(sync_fn):
            sync_fn(list(sim.pop.alive_agents()))

    def _forbidden(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("legacy action materializer called")

    candidate._batched_actions = _forbidden
    candidate._batched_actions_k = _forbidden
    candidate._batched_actions_strided = _forbidden

    baseline_buckets: dict[str, float] = {}
    candidate_buckets: dict[str, float] = {}
    result_baseline = baseline._run_ticks(
        gen=0,
        buckets=baseline_buckets,
        injury_recover_mult=1.0,
        deaths_this_gen=0,
        successful_births=0,
        failed_births=0,
        gen_peak_injury=0.0,
        rest_streaks={},
    )
    result_candidate = candidate._run_ticks(
        gen=0,
        buckets=candidate_buckets,
        injury_recover_mult=1.0,
        deaths_this_gen=0,
        successful_births=0,
        failed_births=0,
        gen_peak_injury=0.0,
        rest_streaks={},
    )

    assert result_candidate == result_baseline
    candidate_fp = _extended_fingerprint(candidate)
    baseline_fp = _extended_fingerprint(baseline)
    assert np.allclose(candidate_fp.pop("energy"), baseline_fp.pop("energy"), atol=5e-6)
    assert np.allclose(candidate_fp.pop("health"), baseline_fp.pop("health"), atol=5e-6)
    assert candidate_fp == baseline_fp
    assert _sha_json(candidate.rng_manager.indices_snapshot()) == _sha_json(baseline.rng_manager.indices_snapshot())
    assert np.allclose(candidate.env.rng.random(16), baseline.env.rng.random(16), atol=0.0)
    assert np.allclose(candidate.np_rng.random(16), baseline.np_rng.random(16), atol=0.0)
    assert float(candidate_buckets.get("runnercore33_stride1_direct_action_rows", 0.0)) > 0.0
    fallback = candidate._runnercore11_c1_typed_genome_row_summary["fallback_counters"]
    assert fallback["legacy_batched_actions_calls"] == 0
    assert fallback["legacy_batched_actions_k_calls"] == 0
    assert fallback["legacy_batched_actions_strided_calls"] == 0
    assert fallback["legacy_action_cache_refill_calls"] == 0
    assert fallback["unsupported_fallback_count"] == 0


def test_runnercore34_full_birth_resident_decision_matches_runnercore31_live(tmp_path: Path) -> None:
    baseline = _prepare_sim(
        tmp_path / "runnercore31_mate_patch_birth_resident_decision_baseline_live",
        runnercore5=False,
        runnercore18=True,
        runnercore20=True,
        runnercore22=True,
        runnercore27=True,
        runnercore31=True,
        steps=3,
        policy_tick_stride=1,
        patch_world=True,
        reproduce_requires_mate=True,
    )
    candidate = _prepare_sim(
        tmp_path / "runnercore34_mate_patch_birth_resident_decision_live",
        runnercore5=False,
        runnercore18=True,
        runnercore20=True,
        runnercore22=True,
        runnercore27=True,
        runnercore31=True,
        runnercore34=True,
        steps=3,
        policy_tick_stride=1,
        patch_world=True,
        reproduce_requires_mate=True,
    )

    forced_positions = [(2, 2), (2, 3), (5, 5), (5, 4)]
    for sim in (baseline, candidate):
        sim.sim_cfg.chemistry_functional_coupling_enable = True
        sim.sim_cfg.chemistry_repro_viability_enable = True
        sim.sim_cfg.lineage_reproduction_funnel_reporting_enable = False
        sim.sim_cfg.lineage_reproduction_funnel_write_events = False
        sim.env_cfg.reproduce_try_prob = 1.0
        sim.env_cfg.reproduce_parent_energy_share = 0.10
        sim.env_cfg.reproduce_cooldown_steps = 2
        sim.env_cfg.jungle_predator_pressure_enable = False
        alive = sorted(sim.pop.alive_agents(), key=lambda a: int(a.agent_id))
        for agent in alive:
            sim.env.remove_agent(int(agent.agent_id))
        for idx, (agent, pos) in enumerate(zip(alive, forced_positions)):
            agent.energy = 120.0 - float(idx)
            agent.repro_cooldown = 0
            agent.genome.reproduction_threshold = 1.0
            agent.chemistry = np.asarray((1.0, 1.0, 0.0, 0.0, 0.5, 0.5), dtype=np.float32)
            agent.stats.survival_steps = idx
            sim.env.add_agent(int(agent.agent_id), pos=pos)
        sync_fn = getattr(sim, "_phase3_pre_start_generation_sync", None)
        if callable(sync_fn):
            sync_fn(list(sim.pop.alive_agents()))

    def _forbidden(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("legacy action materializer called")

    candidate._batched_actions = _forbidden
    candidate._batched_actions_k = _forbidden
    candidate._batched_actions_strided = _forbidden

    baseline_buckets: dict[str, float] = {}
    candidate_buckets: dict[str, float] = {}
    result_baseline = baseline._run_ticks(
        gen=0,
        buckets=baseline_buckets,
        injury_recover_mult=1.0,
        deaths_this_gen=0,
        successful_births=0,
        failed_births=0,
        gen_peak_injury=0.0,
        rest_streaks={},
    )
    result_candidate = candidate._run_ticks(
        gen=0,
        buckets=candidate_buckets,
        injury_recover_mult=1.0,
        deaths_this_gen=0,
        successful_births=0,
        failed_births=0,
        gen_peak_injury=0.0,
        rest_streaks={},
    )

    assert result_candidate == result_baseline
    candidate_fp = _extended_fingerprint(candidate)
    baseline_fp = _extended_fingerprint(baseline)
    assert np.allclose(candidate_fp.pop("energy"), baseline_fp.pop("energy"), atol=5e-6)
    assert np.allclose(candidate_fp.pop("health"), baseline_fp.pop("health"), atol=5e-6)
    assert candidate_fp == baseline_fp
    assert _sha_json(candidate.rng_manager.indices_snapshot()) == _sha_json(baseline.rng_manager.indices_snapshot())
    assert np.allclose(candidate.env.rng.random(16), baseline.env.rng.random(16), atol=0.0)
    assert np.allclose(candidate.np_rng.random(16), baseline.np_rng.random(16), atol=0.0)
    assert len(getattr(candidate, "_patch_birth_contexts_gen", []) or []) == len(
        getattr(baseline, "_patch_birth_contexts_gen", []) or []
    )

    baseline_profile = baseline._runnercore11_c1_typed_genome_row_summary["runnercore3_profile"]["counts"]
    candidate_profile = candidate._runnercore11_c1_typed_genome_row_summary["runnercore3_profile"]["counts"]
    assert candidate_profile["runnercore34_full_birth_resident_decision_rows"] > 0
    assert candidate_profile["runnercore34_full_birth_parent_export_rows"] > 0
    assert candidate_profile["runnercore34_full_birth_candidate_exports_avoided"] > 0
    assert candidate_profile["runnercore22_full_birth_boundary_bridge_export_rows"] < baseline_profile[
        "runnercore22_full_birth_boundary_bridge_export_rows"
    ]
    fallback = candidate._runnercore11_c1_typed_genome_row_summary["fallback_counters"]
    assert fallback["legacy_batched_actions_calls"] == 0
    assert fallback["legacy_batched_actions_k_calls"] == 0
    assert fallback["legacy_batched_actions_strided_calls"] == 0
    assert fallback["legacy_action_cache_refill_calls"] == 0
    assert fallback["unsupported_fallback_count"] == 0


def test_runnercore34_checkpoint_period_remains_support_blocker_until_continuation_is_owned(tmp_path: Path) -> None:
    sim = _prepare_sim(
        tmp_path / "runnercore34_checkpoint_support",
        runnercore5=False,
        runnercore18=True,
        runnercore20=True,
        runnercore22=True,
        runnercore27=True,
        runnercore31=True,
        runnercore34=True,
        steps=2,
        policy_tick_stride=1,
        patch_world=True,
        reproduce_requires_mate=True,
    )
    sim.sim_cfg.chemistry_functional_coupling_enable = True
    sim.sim_cfg.chemistry_repro_viability_enable = True
    sim.sim_cfg.checkpoint_bundle_v2_period_gen = 1
    sim.env_cfg.reproduce_try_prob = 1.0
    sim.env_cfg.reproduce_parent_energy_share = 0.10
    sim.env_cfg.reproduce_cooldown_steps = 2
    sim.env_cfg.jungle_predator_pressure_enable = False

    report = classify_support(sim)

    assert report["supported"] is False
    assert "checkpoint_bundle_v2_period_gen_must_be_0" in report["blockers"]


def test_runnercore42_patch_birth_context_cache_matches_runnercore34_live(tmp_path: Path) -> None:
    baseline = _prepare_sim(
        tmp_path / "runnercore34_patch_birth_context_baseline_live",
        runnercore5=False,
        runnercore18=True,
        runnercore20=True,
        runnercore22=True,
        runnercore27=True,
        runnercore31=True,
        runnercore34=True,
        steps=3,
        policy_tick_stride=1,
        patch_world=True,
        reproduce_requires_mate=True,
    )
    candidate = _prepare_sim(
        tmp_path / "runnercore42_patch_birth_context_cache_live",
        runnercore5=False,
        runnercore18=True,
        runnercore20=True,
        runnercore22=True,
        runnercore27=True,
        runnercore31=True,
        runnercore34=True,
        runnercore42=True,
        steps=3,
        policy_tick_stride=1,
        patch_world=True,
        reproduce_requires_mate=True,
    )

    forced_positions = [(2, 2), (2, 3), (5, 5), (5, 4)]
    for sim in (baseline, candidate):
        sim.sim_cfg.chemistry_functional_coupling_enable = True
        sim.sim_cfg.chemistry_repro_viability_enable = True
        sim.sim_cfg.lineage_reproduction_funnel_reporting_enable = False
        sim.sim_cfg.lineage_reproduction_funnel_write_events = False
        sim.env_cfg.reproduce_try_prob = 1.0
        sim.env_cfg.reproduce_parent_energy_share = 0.10
        sim.env_cfg.reproduce_cooldown_steps = 2
        sim.env_cfg.jungle_predator_pressure_enable = False
        sim.env_cfg.jungle_reproduction_emissions_enable = True
        sim.env_cfg.jungle_birth_scent_emit = 0.2
        sim.env_cfg.jungle_birth_waste_emit = 0.3
        alive = sorted(sim.pop.alive_agents(), key=lambda a: int(a.agent_id))
        for agent in alive:
            sim.env.remove_agent(int(agent.agent_id))
        for idx, (agent, pos) in enumerate(zip(alive, forced_positions)):
            agent.energy = 120.0 - float(idx)
            agent.repro_cooldown = 0
            agent.genome.reproduction_threshold = 1.0
            agent.chemistry = np.asarray((1.0, 1.0, 0.0, 0.0, 0.5, 0.5), dtype=np.float32)
            agent.stats.survival_steps = idx
            sim.env.add_agent(int(agent.agent_id), pos=pos)
        sync_fn = getattr(sim, "_phase3_pre_start_generation_sync", None)
        if callable(sync_fn):
            sync_fn(list(sim.pop.alive_agents()))

    baseline_buckets: dict[str, float] = {}
    candidate_buckets: dict[str, float] = {}
    result_baseline = baseline._run_ticks(
        gen=0,
        buckets=baseline_buckets,
        injury_recover_mult=1.0,
        deaths_this_gen=0,
        successful_births=0,
        failed_births=0,
        gen_peak_injury=0.0,
        rest_streaks={},
    )
    result_candidate = candidate._run_ticks(
        gen=0,
        buckets=candidate_buckets,
        injury_recover_mult=1.0,
        deaths_this_gen=0,
        successful_births=0,
        failed_births=0,
        gen_peak_injury=0.0,
        rest_streaks={},
    )

    assert result_candidate == result_baseline
    candidate_fp = _extended_fingerprint(candidate)
    baseline_fp = _extended_fingerprint(baseline)
    assert np.allclose(candidate_fp.pop("energy"), baseline_fp.pop("energy"), atol=5e-6)
    assert np.allclose(candidate_fp.pop("health"), baseline_fp.pop("health"), atol=5e-6)
    assert candidate_fp == baseline_fp
    assert _sha_json(candidate.rng_manager.indices_snapshot()) == _sha_json(baseline.rng_manager.indices_snapshot())
    assert np.allclose(candidate.env.rng.random(16), baseline.env.rng.random(16), atol=0.0)
    assert np.allclose(candidate.np_rng.random(16), baseline.np_rng.random(16), atol=0.0)
    assert getattr(candidate, "_patch_birth_contexts_gen", []) == getattr(baseline, "_patch_birth_contexts_gen", [])
    assert getattr(candidate, "_jungle_birth_emission_contexts_gen", []) == getattr(
        baseline,
        "_jungle_birth_emission_contexts_gen",
        [],
    )
    assert np.array_equal(candidate.env.local_density_map, baseline.env.local_density_map)

    stats = getattr(candidate.sim_cfg, "_runnercore6_resident_child_row_stats", {}) or {}
    assert int(stats.get("runnercore42_patch_birth_context_cache_rows", 0)) > 0
    assert int(stats.get("runnercore42_jungle_birth_context_cache_rows", 0)) > 0
    fallback = candidate._runnercore11_c1_typed_genome_row_summary["fallback_counters"]
    assert fallback["legacy_batched_actions_calls"] == 0
    assert fallback["legacy_batched_actions_k_calls"] == 0
    assert fallback["legacy_batched_actions_strided_calls"] == 0
    assert fallback["legacy_action_cache_refill_calls"] == 0
    assert fallback["unsupported_fallback_count"] == 0


def test_runnercore43_threshold_tick_cache_matches_runnercore34_live(tmp_path: Path) -> None:
    baseline = _prepare_sim(
        tmp_path / "runnercore34_threshold_cache_baseline_live",
        runnercore5=False,
        runnercore18=True,
        runnercore20=True,
        runnercore22=True,
        runnercore27=True,
        runnercore31=True,
        runnercore34=True,
        steps=3,
        policy_tick_stride=1,
        patch_world=True,
        reproduce_requires_mate=True,
    )
    candidate = _prepare_sim(
        tmp_path / "runnercore43_threshold_cache_live",
        runnercore5=False,
        runnercore18=True,
        runnercore20=True,
        runnercore22=True,
        runnercore27=True,
        runnercore31=True,
        runnercore34=True,
        runnercore43=True,
        steps=3,
        policy_tick_stride=1,
        patch_world=True,
        reproduce_requires_mate=True,
    )

    forced_positions = [(2, 2), (2, 3), (5, 5), (5, 4)]
    for sim in (baseline, candidate):
        sim.sim_cfg.chemistry_functional_coupling_enable = True
        sim.sim_cfg.chemistry_repro_viability_enable = True
        sim.sim_cfg.lineage_reproduction_funnel_reporting_enable = False
        sim.sim_cfg.lineage_reproduction_funnel_write_events = False
        sim.env_cfg.reproduce_try_prob = 1.0
        sim.env_cfg.reproduce_parent_energy_share = 0.10
        sim.env_cfg.reproduce_cooldown_steps = 2
        sim.env_cfg.jungle_predator_pressure_enable = False
        alive = sorted(sim.pop.alive_agents(), key=lambda a: int(a.agent_id))
        for agent in alive:
            sim.env.remove_agent(int(agent.agent_id))
        for idx, (agent, pos) in enumerate(zip(alive, forced_positions)):
            agent.energy = 120.0 - float(idx)
            agent.repro_cooldown = 0
            agent.genome.reproduction_threshold = 1.0
            agent.chemistry = np.asarray((1.0, 1.0, 0.0, 0.0, 0.5, 0.5), dtype=np.float32)
            agent.stats.survival_steps = idx
            sim.env.add_agent(int(agent.agent_id), pos=pos)
        sync_fn = getattr(sim, "_phase3_pre_start_generation_sync", None)
        if callable(sync_fn):
            sync_fn(list(sim.pop.alive_agents()))

    baseline_buckets: dict[str, float] = {}
    candidate_buckets: dict[str, float] = {}
    result_baseline = baseline._run_ticks(
        gen=0,
        buckets=baseline_buckets,
        injury_recover_mult=1.0,
        deaths_this_gen=0,
        successful_births=0,
        failed_births=0,
        gen_peak_injury=0.0,
        rest_streaks={},
    )
    result_candidate = candidate._run_ticks(
        gen=0,
        buckets=candidate_buckets,
        injury_recover_mult=1.0,
        deaths_this_gen=0,
        successful_births=0,
        failed_births=0,
        gen_peak_injury=0.0,
        rest_streaks={},
    )

    assert result_candidate == result_baseline
    candidate_fp = _extended_fingerprint(candidate)
    baseline_fp = _extended_fingerprint(baseline)
    assert np.allclose(candidate_fp.pop("energy"), baseline_fp.pop("energy"), atol=5e-6)
    assert np.allclose(candidate_fp.pop("health"), baseline_fp.pop("health"), atol=5e-6)
    assert candidate_fp == baseline_fp
    assert _sha_json(candidate.rng_manager.indices_snapshot()) == _sha_json(baseline.rng_manager.indices_snapshot())
    assert np.allclose(candidate.env.rng.random(16), baseline.env.rng.random(16), atol=0.0)
    assert np.allclose(candidate.np_rng.random(16), baseline.np_rng.random(16), atol=0.0)
    assert getattr(candidate, "_patch_birth_contexts_gen", []) == getattr(baseline, "_patch_birth_contexts_gen", [])

    stats = getattr(candidate.sim_cfg, "_runnercore6_resident_child_row_stats", {}) or {}
    assert int(stats.get("runnercore43_full_birth_threshold_tick_cache_misses", 0)) > 0
    fallback = candidate._runnercore11_c1_typed_genome_row_summary["fallback_counters"]
    assert fallback["legacy_batched_actions_calls"] == 0
    assert fallback["legacy_batched_actions_k_calls"] == 0
    assert fallback["legacy_batched_actions_strided_calls"] == 0
    assert fallback["legacy_action_cache_refill_calls"] == 0
    assert fallback["unsupported_fallback_count"] == 0


def test_runnercore49_repro_viability_tick_cache_matches_runnercore34_live(tmp_path: Path) -> None:
    baseline = _prepare_sim(
        tmp_path / "runnercore49_repro_viability_cache_baseline_live",
        runnercore5=False,
        runnercore18=True,
        runnercore20=True,
        runnercore22=True,
        runnercore27=True,
        runnercore31=True,
        runnercore34=True,
        runnercore43=True,
        steps=1,
        policy_tick_stride=1,
        patch_world=False,
        reproduce_requires_mate=True,
    )
    candidate = _prepare_sim(
        tmp_path / "runnercore49_repro_viability_cache_live",
        runnercore5=False,
        runnercore18=True,
        runnercore20=True,
        runnercore22=True,
        runnercore27=True,
        runnercore31=True,
        runnercore34=True,
        runnercore43=True,
        runnercore49=True,
        steps=1,
        policy_tick_stride=1,
        patch_world=False,
        reproduce_requires_mate=True,
    )

    forced_positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
    for sim in (baseline, candidate):
        sim.env_cfg.width = 2
        sim.env_cfg.height = 2
        sim.env.cfg.width = 2
        sim.env.cfg.height = 2
        sim.sim_cfg.chemistry_functional_coupling_enable = True
        sim.sim_cfg.chemistry_repro_viability_enable = True
        sim.sim_cfg.lineage_reproduction_funnel_reporting_enable = False
        sim.sim_cfg.lineage_reproduction_funnel_write_events = False
        sim.env_cfg.reproduce_try_prob = 1.0
        sim.env_cfg.reproduce_parent_energy_share = 0.10
        sim.env_cfg.reproduce_cooldown_steps = 1
        sim.env_cfg.jungle_predator_pressure_enable = False
        sim.env_cfg.food_spawn_per_step = 0
        alive = sorted(sim.pop.alive_agents(), key=lambda a: int(a.agent_id))
        for agent in alive:
            sim.env.remove_agent(int(agent.agent_id))
        for idx, (agent, pos) in enumerate(zip(alive, forced_positions)):
            agent.energy = 120.0 - float(idx)
            agent.repro_cooldown = 0
            agent.genome.reproduction_threshold = 1.0
            agent.chemistry = np.asarray(
                (1.0, 0.85 - 0.05 * idx, 0.05 * idx, 0.10 + 0.03 * idx, 0.5, 0.5),
                dtype=np.float32,
            )
            agent.stats.survival_steps = idx
            sim.env.add_agent(int(agent.agent_id), pos=pos)
        sync_fn = getattr(sim, "_phase3_pre_start_generation_sync", None)
        if callable(sync_fn):
            sync_fn(list(sim.pop.alive_agents()))

    baseline_buckets: dict[str, float] = {}
    candidate_buckets: dict[str, float] = {}
    result_baseline = baseline._run_ticks(
        gen=0,
        buckets=baseline_buckets,
        injury_recover_mult=1.0,
        deaths_this_gen=0,
        successful_births=0,
        failed_births=0,
        gen_peak_injury=0.0,
        rest_streaks={},
    )
    result_candidate = candidate._run_ticks(
        gen=0,
        buckets=candidate_buckets,
        injury_recover_mult=1.0,
        deaths_this_gen=0,
        successful_births=0,
        failed_births=0,
        gen_peak_injury=0.0,
        rest_streaks={},
    )

    assert result_candidate == result_baseline
    candidate_fp = _extended_fingerprint(candidate)
    baseline_fp = _extended_fingerprint(baseline)
    assert np.allclose(candidate_fp.pop("energy"), baseline_fp.pop("energy"), atol=5e-6)
    assert np.allclose(candidate_fp.pop("health"), baseline_fp.pop("health"), atol=5e-6)
    assert candidate_fp == baseline_fp
    assert _sha_json(candidate.rng_manager.indices_snapshot()) == _sha_json(baseline.rng_manager.indices_snapshot())
    assert np.allclose(candidate.env.rng.random(16), baseline.env.rng.random(16), atol=0.0)
    assert np.allclose(candidate.np_rng.random(16), baseline.np_rng.random(16), atol=0.0)

    profile_counts = candidate._runnercore11_c1_typed_genome_row_summary["runnercore3_profile"]["counts"]
    assert profile_counts["runnercore49_repro_viability_tick_cache_hits"] > 0
    assert profile_counts["runnercore49_repro_viability_tick_cache_misses"] > 0
    fallback = candidate._runnercore11_c1_typed_genome_row_summary["fallback_counters"]
    assert fallback["legacy_batched_actions_calls"] == 0
    assert fallback["legacy_batched_actions_k_calls"] == 0
    assert fallback["legacy_batched_actions_strided_calls"] == 0
    assert fallback["legacy_action_cache_refill_calls"] == 0
    assert fallback["unsupported_fallback_count"] == 0


def test_runnercore50_spawn_first_valid_matches_runnercore34_live(tmp_path: Path) -> None:
    baseline = _prepare_sim(
        tmp_path / "runnercore50_spawn_first_valid_baseline_live",
        runnercore5=False,
        runnercore18=True,
        runnercore20=True,
        runnercore22=True,
        runnercore27=True,
        runnercore31=True,
        runnercore34=True,
        runnercore43=True,
        runnercore49=True,
        steps=3,
        policy_tick_stride=1,
        patch_world=True,
        reproduce_requires_mate=True,
    )
    candidate = _prepare_sim(
        tmp_path / "runnercore50_spawn_first_valid_live",
        runnercore5=False,
        runnercore18=True,
        runnercore20=True,
        runnercore22=True,
        runnercore27=True,
        runnercore31=True,
        runnercore34=True,
        runnercore43=True,
        runnercore49=True,
        runnercore50=True,
        steps=3,
        policy_tick_stride=1,
        patch_world=True,
        reproduce_requires_mate=True,
    )

    forced_positions = [(2, 2), (2, 3), (5, 5), (5, 4)]
    for sim in (baseline, candidate):
        sim.sim_cfg.chemistry_functional_coupling_enable = True
        sim.sim_cfg.chemistry_repro_viability_enable = True
        sim.sim_cfg.lineage_reproduction_funnel_reporting_enable = False
        sim.sim_cfg.lineage_reproduction_funnel_write_events = False
        sim.env_cfg.reproduce_try_prob = 1.0
        sim.env_cfg.reproduce_parent_energy_share = 0.10
        sim.env_cfg.reproduce_cooldown_steps = 2
        sim.env_cfg.jungle_predator_pressure_enable = False
        alive = sorted(sim.pop.alive_agents(), key=lambda a: int(a.agent_id))
        for agent in alive:
            sim.env.remove_agent(int(agent.agent_id))
        for idx, (agent, pos) in enumerate(zip(alive, forced_positions)):
            agent.energy = 120.0 - float(idx)
            agent.repro_cooldown = 0
            agent.genome.reproduction_threshold = 1.0
            agent.chemistry = np.asarray((1.0, 1.0, 0.0, 0.0, 0.5, 0.5), dtype=np.float32)
            agent.stats.survival_steps = idx
            sim.env.add_agent(int(agent.agent_id), pos=pos)
        sync_fn = getattr(sim, "_phase3_pre_start_generation_sync", None)
        if callable(sync_fn):
            sync_fn(list(sim.pop.alive_agents()))

    baseline_buckets: dict[str, float] = {}
    candidate_buckets: dict[str, float] = {}
    result_baseline = baseline._run_ticks(
        gen=0,
        buckets=baseline_buckets,
        injury_recover_mult=1.0,
        deaths_this_gen=0,
        successful_births=0,
        failed_births=0,
        gen_peak_injury=0.0,
        rest_streaks={},
    )
    result_candidate = candidate._run_ticks(
        gen=0,
        buckets=candidate_buckets,
        injury_recover_mult=1.0,
        deaths_this_gen=0,
        successful_births=0,
        failed_births=0,
        gen_peak_injury=0.0,
        rest_streaks={},
    )

    assert result_candidate == result_baseline
    candidate_fp = _extended_fingerprint(candidate)
    baseline_fp = _extended_fingerprint(baseline)
    assert np.allclose(candidate_fp.pop("energy"), baseline_fp.pop("energy"), atol=5e-6)
    assert np.allclose(candidate_fp.pop("health"), baseline_fp.pop("health"), atol=5e-6)
    assert candidate_fp == baseline_fp
    assert _sha_json(candidate.rng_manager.indices_snapshot()) == _sha_json(baseline.rng_manager.indices_snapshot())
    assert np.allclose(candidate.env.rng.random(16), baseline.env.rng.random(16), atol=0.0)
    assert np.allclose(candidate.np_rng.random(16), baseline.np_rng.random(16), atol=0.0)
    assert getattr(candidate, "_patch_birth_contexts_gen", []) == getattr(baseline, "_patch_birth_contexts_gen", [])

    profile_counts = candidate._runnercore11_c1_typed_genome_row_summary["runnercore3_profile"]["counts"]
    assert profile_counts["runnercore50_spawn_first_valid_scan_rows"] > 0
    assert profile_counts["runnercore50_spawn_first_valid_selected_rows"] > 0
    fallback = candidate._runnercore11_c1_typed_genome_row_summary["fallback_counters"]
    assert fallback["legacy_batched_actions_calls"] == 0
    assert fallback["legacy_batched_actions_k_calls"] == 0
    assert fallback["legacy_batched_actions_strided_calls"] == 0
    assert fallback["legacy_action_cache_refill_calls"] == 0
    assert fallback["unsupported_fallback_count"] == 0


def test_runnercore47_buffered_developmental_trace_matches_legacy_trace(tmp_path: Path) -> None:
    baseline = _prepare_sim(
        tmp_path / "runnercore43_dev_trace_legacy_live",
        runnercore5=False,
        runnercore18=True,
        runnercore20=True,
        runnercore22=True,
        runnercore27=True,
        runnercore31=True,
        runnercore34=True,
        runnercore43=True,
        steps=4,
        policy_tick_stride=1,
        patch_world=True,
        reproduce_requires_mate=True,
    )
    candidate = _prepare_sim(
        tmp_path / "runnercore47_dev_trace_buffered_live",
        runnercore5=False,
        runnercore18=True,
        runnercore20=True,
        runnercore22=True,
        runnercore27=True,
        runnercore31=True,
        runnercore34=True,
        runnercore43=True,
        runnercore47=True,
        steps=4,
        policy_tick_stride=1,
        patch_world=True,
        reproduce_requires_mate=True,
    )

    isolated_positions = [(1, 1), (1, 5), (5, 1), (5, 5)]
    for sim in (baseline, candidate):
        sim.sim_cfg.chemistry_functional_coupling_enable = False
        sim.sim_cfg.chemistry_repro_viability_enable = False
        sim.sim_cfg.lineage_reproduction_funnel_reporting_enable = False
        sim.sim_cfg.lineage_reproduction_funnel_write_events = False
        sim.env_cfg.reproduce_try_prob = 1.0
        sim.env_cfg.reproduce_parent_energy_share = 0.10
        sim.env_cfg.reproduce_cooldown_steps = 0
        sim.env_cfg.jungle_predator_pressure_enable = False
        alive = sorted(sim.pop.alive_agents(), key=lambda a: int(a.agent_id))
        for agent in alive:
            sim.env.remove_agent(int(agent.agent_id))
        for idx, (agent, pos) in enumerate(zip(alive, isolated_positions)):
            agent.energy = 120.0 - float(idx)
            agent.repro_cooldown = 0
            agent.genome.reproduction_threshold = 1.0
            agent.stats.survival_steps = idx
            sim.env.add_agent(int(agent.agent_id), pos=pos)
        sync_fn = getattr(sim, "_phase3_pre_start_generation_sync", None)
        if callable(sync_fn):
            sync_fn(list(sim.pop.alive_agents()))

    baseline_buckets: dict[str, float] = {}
    candidate_buckets: dict[str, float] = {}
    result_baseline = baseline._run_ticks(
        gen=0,
        buckets=baseline_buckets,
        injury_recover_mult=1.0,
        deaths_this_gen=0,
        successful_births=0,
        failed_births=0,
        gen_peak_injury=0.0,
        rest_streaks={},
    )
    result_candidate = candidate._run_ticks(
        gen=0,
        buckets=candidate_buckets,
        injury_recover_mult=1.0,
        deaths_this_gen=0,
        successful_births=0,
        failed_births=0,
        gen_peak_injury=0.0,
        rest_streaks={},
    )
    for sim in (baseline, candidate):
        sim._dev_flush_pending_traces(force=True)
        sim._dev_flush_trace_emit_buffer()

    assert result_candidate == result_baseline
    candidate_fp = _extended_fingerprint(candidate)
    baseline_fp = _extended_fingerprint(baseline)
    assert np.allclose(candidate_fp.pop("energy"), baseline_fp.pop("energy"), atol=5e-6)
    assert np.allclose(candidate_fp.pop("health"), baseline_fp.pop("health"), atol=5e-6)
    assert candidate_fp == baseline_fp
    assert _sha_json(candidate.rng_manager.indices_snapshot()) == _sha_json(baseline.rng_manager.indices_snapshot())
    assert np.allclose(candidate.env.rng.random(16), baseline.env.rng.random(16), atol=0.0)
    assert np.allclose(candidate.np_rng.random(16), baseline.np_rng.random(16), atol=0.0)
    assert getattr(candidate, "_dev_stage_transition_count_gen", 0) == getattr(
        baseline,
        "_dev_stage_transition_count_gen",
        0,
    )
    assert getattr(candidate, "_dev_stage_eval_stage_counts_gen", {}) == getattr(
        baseline,
        "_dev_stage_eval_stage_counts_gen",
        {},
    )
    baseline_trace = Path(baseline.run_dir) / "metrics" / "developmental_stage_trace.jsonl"
    candidate_trace = Path(candidate.run_dir) / "metrics" / "developmental_stage_trace.jsonl"
    assert candidate_trace.read_text(encoding="utf-8") == baseline_trace.read_text(encoding="utf-8")
    assert getattr(candidate, "_dev_trace_emit_buffer", []) == []
    counters = getattr(candidate, "_dev_trace_emit_counters", {})
    assert int(counters.get("records_queued", 0)) > 0
    assert int(counters.get("records_written", 0)) == int(counters.get("records_queued", 0))
    assert int(counters.get("flush_count", 0)) >= 1
    assert int(counters.get("file_opens", 0)) == int(counters.get("flush_count", 0))
    assert int(counters.get("makedirs_calls", 0)) <= 1
    assert int(counters.get("write_errors", 0)) == 0

    fallback = candidate._runnercore11_c1_typed_genome_row_summary["fallback_counters"]
    assert fallback["legacy_batched_actions_calls"] == 0
    assert fallback["legacy_batched_actions_k_calls"] == 0
    assert fallback["legacy_batched_actions_strided_calls"] == 0
    assert fallback["legacy_action_cache_refill_calls"] == 0
    assert fallback["unsupported_fallback_count"] == 0


def test_runnercore34_row_spec_child_bridge_matches_r34_live(tmp_path: Path) -> None:
    baseline = _prepare_sim(
        tmp_path / "runnercore34_row_spec_baseline_live",
        runnercore5=False,
        runnercore6=True,
        runnercore7=True,
        runnercore11=True,
        runnercore12=True,
        runnercore13=True,
        runnercore15=True,
        runnercore16=True,
        runnercore18=True,
        runnercore20=True,
        runnercore22=True,
        runnercore27=True,
        runnercore31=True,
        runnercore34=True,
        steps=3,
        policy_tick_stride=1,
        patch_world=True,
        reproduce_requires_mate=True,
    )
    candidate = _prepare_sim(
        tmp_path / "runnercore34_row_spec_candidate_live",
        runnercore5=False,
        runnercore6=True,
        runnercore7=True,
        runnercore11=True,
        runnercore12=True,
        runnercore13=True,
        runnercore15=True,
        runnercore16=True,
        runnercore18=True,
        runnercore20=True,
        runnercore22=True,
        runnercore25=True,
        runnercore27=True,
        runnercore31=True,
        runnercore34=True,
        steps=3,
        policy_tick_stride=1,
        patch_world=True,
        reproduce_requires_mate=True,
    )

    forced_positions = [(2, 2), (2, 3), (5, 5), (5, 4)]
    for sim in (baseline, candidate):
        sim.sim_cfg.chemistry_functional_coupling_enable = True
        sim.sim_cfg.chemistry_repro_viability_enable = True
        sim.sim_cfg.lineage_reproduction_funnel_reporting_enable = False
        sim.sim_cfg.lineage_reproduction_funnel_write_events = False
        sim.env_cfg.reproduce_try_prob = 1.0
        sim.env_cfg.reproduce_parent_energy_share = 0.10
        sim.env_cfg.reproduce_cooldown_steps = 2
        sim.env_cfg.jungle_predator_pressure_enable = False
        alive = sorted(sim.pop.alive_agents(), key=lambda a: int(a.agent_id))
        for agent in alive:
            sim.env.remove_agent(int(agent.agent_id))
        for idx, (agent, pos) in enumerate(zip(alive, forced_positions)):
            agent.energy = 120.0 - float(idx)
            agent.repro_cooldown = 0
            agent.genome.reproduction_threshold = 1.0
            agent.chemistry = np.asarray((1.0, 1.0, 0.0, 0.0, 0.5, 0.5), dtype=np.float32)
            agent.stats.survival_steps = idx
            sim.env.add_agent(int(agent.agent_id), pos=pos)
        sync_fn = getattr(sim, "_phase3_pre_start_generation_sync", None)
        if callable(sync_fn):
            sync_fn(list(sim.pop.alive_agents()))

    def _forbidden(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("legacy action materializer called")

    candidate._batched_actions = _forbidden
    candidate._batched_actions_k = _forbidden
    candidate._batched_actions_strided = _forbidden

    baseline_buckets: dict[str, float] = {}
    candidate_buckets: dict[str, float] = {}
    result_baseline = baseline._run_ticks(
        gen=0,
        buckets=baseline_buckets,
        injury_recover_mult=1.0,
        deaths_this_gen=0,
        successful_births=0,
        failed_births=0,
        gen_peak_injury=0.0,
        rest_streaks={},
    )
    result_candidate = candidate._run_ticks(
        gen=0,
        buckets=candidate_buckets,
        injury_recover_mult=1.0,
        deaths_this_gen=0,
        successful_births=0,
        failed_births=0,
        gen_peak_injury=0.0,
        rest_streaks={},
    )

    assert result_candidate == result_baseline
    candidate_fp = _extended_fingerprint(candidate)
    baseline_fp = _extended_fingerprint(baseline)
    assert np.allclose(candidate_fp.pop("energy"), baseline_fp.pop("energy"), atol=5e-6)
    assert np.allclose(candidate_fp.pop("health"), baseline_fp.pop("health"), atol=5e-6)
    assert candidate_fp == baseline_fp
    assert _sha_json(candidate.rng_manager.indices_snapshot()) == _sha_json(baseline.rng_manager.indices_snapshot())
    assert np.allclose(candidate.env.rng.random(16), baseline.env.rng.random(16), atol=0.0)
    assert np.allclose(candidate.np_rng.random(16), baseline.np_rng.random(16), atol=0.0)
    assert len(getattr(candidate, "_patch_birth_contexts_gen", []) or []) == len(
        getattr(baseline, "_patch_birth_contexts_gen", []) or []
    )

    summary = candidate._runnercore11_c1_typed_genome_row_summary
    fallback = summary["fallback_counters"]
    assert fallback["legacy_batched_actions_calls"] == 0
    assert fallback["legacy_batched_actions_k_calls"] == 0
    assert fallback["legacy_batched_actions_strided_calls"] == 0
    assert fallback["legacy_action_cache_refill_calls"] == 0
    assert fallback["unsupported_fallback_count"] == 0
    profile_counts = summary["runnercore3_profile"]["counts"]
    assert profile_counts["runnercore34_full_birth_resident_decision_rows"] > 0
    assert profile_counts["runnercore25_full_birth_row_specs_created"] > 0
    assert profile_counts["runnercore25_full_birth_row_specs_applied"] > 0
    assert profile_counts["runnercore25_full_birth_newborn_attach_rows"] > 0
    r6 = summary["runnercore6_resident_child_row"]
    assert r6["runnercore25_full_birth_row_specs_created"] > 0
    assert r6["runnercore25_full_birth_row_specs_applied"] > 0
    assert r6["runnercore25_full_birth_newborn_attach_rows"] > 0


def test_runnercore23_jungle_predator_bridge_matches_legacy_live(tmp_path: Path) -> None:
    legacy = _prepare_sim(
        tmp_path / "legacy_jungle_predator_live",
        runnercore4=False,
        runnercore5=False,
        steps=2,
        policy_tick_stride=1,
        patch_world=True,
    )
    candidate = _prepare_sim(
        tmp_path / "runnercore23_jungle_predator_live",
        runnercore5=False,
        runnercore18=True,
        runnercore22=True,
        runnercore23=True,
        steps=2,
        policy_tick_stride=1,
        patch_world=True,
    )

    forced_positions = [(2, 2), (2, 3), (5, 5), (5, 4)]
    for sim in (legacy, candidate):
        sim.env_cfg.reproduce_try_prob = 0.0
        sim.env_cfg.jungle_predator_pressure_enable = True
        sim.env_cfg.jungle_predator_base_risk = 1.0
        sim.env_cfg.jungle_predator_risk_cap = 1.0
        sim.env_cfg.jungle_predator_health_damage = 0.10
        sim.env_cfg.jungle_terrain_protection_enable = False
        alive = sorted(sim.pop.alive_agents(), key=lambda a: int(a.agent_id))
        for agent in alive:
            sim.env.remove_agent(int(agent.agent_id))
        for idx, (agent, pos) in enumerate(zip(alive, forced_positions)):
            agent.energy = 80.0 - float(idx)
            agent.health = 1.0
            agent.repro_cooldown = 0
            agent.stats.survival_steps = idx
            sim.env.add_agent(int(agent.agent_id), pos=pos)
        sync_fn = getattr(sim, "_phase3_pre_start_generation_sync", None)
        if callable(sync_fn):
            sync_fn(list(sim.pop.alive_agents()))

    def _forbidden(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("legacy action materializer called")

    candidate._batched_actions = _forbidden
    candidate._batched_actions_k = _forbidden
    candidate._batched_actions_strided = _forbidden

    legacy_buckets: dict[str, float] = {}
    candidate_buckets: dict[str, float] = {}
    result_legacy = legacy._run_ticks(
        gen=0,
        buckets=legacy_buckets,
        injury_recover_mult=1.0,
        deaths_this_gen=0,
        successful_births=0,
        failed_births=0,
        gen_peak_injury=0.0,
        rest_streaks={},
    )
    result_candidate = candidate._run_ticks(
        gen=0,
        buckets=candidate_buckets,
        injury_recover_mult=1.0,
        deaths_this_gen=0,
        successful_births=0,
        failed_births=0,
        gen_peak_injury=0.0,
        rest_streaks={},
    )

    assert result_candidate == result_legacy
    candidate_fp = _extended_fingerprint(candidate)
    legacy_fp = _extended_fingerprint(legacy)
    assert np.allclose(candidate_fp.pop("energy"), legacy_fp.pop("energy"), atol=5e-6)
    assert np.allclose(candidate_fp.pop("health"), legacy_fp.pop("health"), atol=5e-6)
    assert candidate_fp == legacy_fp
    candidate_rng = candidate.rng_manager.indices_snapshot()
    legacy_rng = legacy.rng_manager.indices_snapshot()
    for stream in ("mutation.py", "mutation.np"):
        assert candidate_rng.get(stream) == legacy_rng.get(stream)
    assert int(getattr(candidate, "_jungle_predator_events_gen", 0)) > 0
    assert len(getattr(candidate, "_jungle_predator_event_contexts_gen", []) or []) == len(
        getattr(legacy, "_jungle_predator_event_contexts_gen", []) or []
    )

    summary = candidate._runnercore11_c1_typed_genome_row_summary
    fallback = summary["fallback_counters"]
    assert fallback["legacy_batched_actions_calls"] == 0
    assert fallback["legacy_batched_actions_k_calls"] == 0
    assert fallback["legacy_batched_actions_strided_calls"] == 0
    assert fallback["legacy_action_cache_refill_calls"] == 0
    assert fallback["unsupported_fallback_count"] == 0
    profile_counts = summary["runnercore3_profile"]["counts"]
    assert profile_counts["runnercore23_jungle_predator_bridge_export_rows"] > 0
    assert profile_counts["runnercore23_jungle_predator_bridge_import_rows"] > 0


def test_runnercore24_native_jungle_predator_arrays_match_legacy_live(tmp_path: Path) -> None:
    legacy = _prepare_sim(
        tmp_path / "legacy_jungle_predator_native_live",
        runnercore4=False,
        runnercore5=False,
        steps=2,
        policy_tick_stride=1,
        patch_world=True,
    )
    candidate = _prepare_sim(
        tmp_path / "runnercore24_jungle_predator_native_live",
        runnercore5=False,
        runnercore18=True,
        runnercore22=True,
        runnercore24=True,
        steps=2,
        policy_tick_stride=1,
        patch_world=True,
    )

    forced_positions = [(2, 2), (2, 3), (5, 5), (5, 4)]
    for sim in (legacy, candidate):
        sim.env_cfg.reproduce_try_prob = 0.0
        sim.env_cfg.jungle_predator_pressure_enable = True
        sim.env_cfg.jungle_predator_base_risk = 1.0
        sim.env_cfg.jungle_predator_risk_cap = 1.0
        sim.env_cfg.jungle_predator_health_damage = 0.10
        sim.env_cfg.jungle_terrain_protection_enable = False
        alive = sorted(sim.pop.alive_agents(), key=lambda a: int(a.agent_id))
        for agent in alive:
            sim.env.remove_agent(int(agent.agent_id))
        for idx, (agent, pos) in enumerate(zip(alive, forced_positions)):
            agent.energy = 80.0 - float(idx)
            agent.health = 1.0
            agent.repro_cooldown = 0
            agent.stats.survival_steps = idx
            sim.env.add_agent(int(agent.agent_id), pos=pos)
        sync_fn = getattr(sim, "_phase3_pre_start_generation_sync", None)
        if callable(sync_fn):
            sync_fn(list(sim.pop.alive_agents()))

    def _forbidden(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("legacy action materializer called")

    candidate._batched_actions = _forbidden
    candidate._batched_actions_k = _forbidden
    candidate._batched_actions_strided = _forbidden

    legacy_buckets: dict[str, float] = {}
    candidate_buckets: dict[str, float] = {}
    result_legacy = legacy._run_ticks(
        gen=0,
        buckets=legacy_buckets,
        injury_recover_mult=1.0,
        deaths_this_gen=0,
        successful_births=0,
        failed_births=0,
        gen_peak_injury=0.0,
        rest_streaks={},
    )
    result_candidate = candidate._run_ticks(
        gen=0,
        buckets=candidate_buckets,
        injury_recover_mult=1.0,
        deaths_this_gen=0,
        successful_births=0,
        failed_births=0,
        gen_peak_injury=0.0,
        rest_streaks={},
    )

    assert result_candidate == result_legacy
    candidate_fp = _extended_fingerprint(candidate)
    legacy_fp = _extended_fingerprint(legacy)
    assert np.allclose(candidate_fp.pop("energy"), legacy_fp.pop("energy"), atol=5e-6)
    assert np.allclose(candidate_fp.pop("health"), legacy_fp.pop("health"), atol=5e-6)
    assert candidate_fp == legacy_fp
    candidate_rng = candidate.rng_manager.indices_snapshot()
    legacy_rng = legacy.rng_manager.indices_snapshot()
    for stream in ("mutation.py", "mutation.np"):
        assert candidate_rng.get(stream) == legacy_rng.get(stream)
    assert int(getattr(candidate, "_jungle_predator_events_gen", 0)) > 0
    assert len(getattr(candidate, "_jungle_predator_event_contexts_gen", []) or []) == len(
        getattr(legacy, "_jungle_predator_event_contexts_gen", []) or []
    )

    summary = candidate._runnercore11_c1_typed_genome_row_summary
    fallback = summary["fallback_counters"]
    assert fallback["legacy_batched_actions_calls"] == 0
    assert fallback["legacy_batched_actions_k_calls"] == 0
    assert fallback["legacy_batched_actions_strided_calls"] == 0
    assert fallback["legacy_action_cache_refill_calls"] == 0
    assert fallback["unsupported_fallback_count"] == 0
    profile_counts = summary["runnercore3_profile"]["counts"]
    assert profile_counts["runnercore23_jungle_predator_bridge_export_rows"] == 0
    assert profile_counts["runnercore23_jungle_predator_bridge_import_rows"] == 0
    assert profile_counts["runnercore24_jungle_predator_native_considered_rows"] > 0
    assert profile_counts["runnercore24_jungle_predator_native_event_rows"] > 0
    assert profile_counts["runnercore24_jungle_predator_native_mirror_sync_rows"] > 0


def test_runnercore45_jungle_predator_context_cache_matches_runnercore24_live(tmp_path: Path) -> None:
    baseline = _prepare_sim(
        tmp_path / "runnercore24_jungle_context_cache_baseline_live",
        runnercore5=False,
        runnercore18=True,
        runnercore22=True,
        runnercore24=True,
        steps=2,
        policy_tick_stride=1,
        patch_world=True,
    )
    candidate = _prepare_sim(
        tmp_path / "runnercore45_jungle_context_cache_live",
        runnercore5=False,
        runnercore18=True,
        runnercore22=True,
        runnercore24=True,
        runnercore45=True,
        steps=2,
        policy_tick_stride=1,
        patch_world=True,
    )

    forced_positions = [(2, 2), (2, 3), (5, 5), (5, 4)]
    for sim in (baseline, candidate):
        sim.env_cfg.reproduce_try_prob = 0.0
        sim.env_cfg.jungle_predator_pressure_enable = True
        sim.env_cfg.jungle_predator_base_risk = 1.0
        sim.env_cfg.jungle_predator_risk_cap = 1.0
        sim.env_cfg.jungle_predator_health_damage = 0.10
        sim.env_cfg.jungle_terrain_protection_enable = False
        alive = sorted(sim.pop.alive_agents(), key=lambda a: int(a.agent_id))
        for agent in alive:
            sim.env.remove_agent(int(agent.agent_id))
        for idx, (agent, pos) in enumerate(zip(alive, forced_positions)):
            agent.energy = 80.0 - float(idx)
            agent.health = 0.05
            agent.repro_cooldown = 0
            agent.stats.survival_steps = idx
            sim.env.add_agent(int(agent.agent_id), pos=pos)
        sync_fn = getattr(sim, "_phase3_pre_start_generation_sync", None)
        if callable(sync_fn):
            sync_fn(list(sim.pop.alive_agents()))

    def _forbidden(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("legacy action materializer called")

    candidate._batched_actions = _forbidden
    candidate._batched_actions_k = _forbidden
    candidate._batched_actions_strided = _forbidden

    baseline_buckets: dict[str, float] = {}
    candidate_buckets: dict[str, float] = {}
    result_baseline = baseline._run_ticks(
        gen=0,
        buckets=baseline_buckets,
        injury_recover_mult=1.0,
        deaths_this_gen=0,
        successful_births=0,
        failed_births=0,
        gen_peak_injury=0.0,
        rest_streaks={},
    )
    result_candidate = candidate._run_ticks(
        gen=0,
        buckets=candidate_buckets,
        injury_recover_mult=1.0,
        deaths_this_gen=0,
        successful_births=0,
        failed_births=0,
        gen_peak_injury=0.0,
        rest_streaks={},
    )

    assert result_candidate == result_baseline
    candidate_fp = _extended_fingerprint(candidate)
    baseline_fp = _extended_fingerprint(baseline)
    assert np.allclose(candidate_fp.pop("energy"), baseline_fp.pop("energy"), atol=5e-6)
    assert np.allclose(candidate_fp.pop("health"), baseline_fp.pop("health"), atol=5e-6)
    assert candidate_fp == baseline_fp
    assert np.allclose(candidate.env.rng.random(16), baseline.env.rng.random(16), atol=0.0)
    candidate_rng = candidate.rng_manager.indices_snapshot()
    baseline_rng = baseline.rng_manager.indices_snapshot()
    for stream in ("mutation.py", "mutation.np"):
        assert candidate_rng.get(stream) == baseline_rng.get(stream)
    assert getattr(candidate, "_jungle_predator_event_contexts_gen", []) == getattr(
        baseline,
        "_jungle_predator_event_contexts_gen",
        [],
    )
    assert getattr(candidate, "_patch_death_contexts_gen", []) == getattr(
        baseline,
        "_patch_death_contexts_gen",
        [],
    )

    summary = candidate._runnercore11_c1_typed_genome_row_summary
    fallback = summary["fallback_counters"]
    assert fallback["legacy_batched_actions_calls"] == 0
    assert fallback["legacy_batched_actions_k_calls"] == 0
    assert fallback["legacy_batched_actions_strided_calls"] == 0
    assert fallback["legacy_action_cache_refill_calls"] == 0
    assert fallback["unsupported_fallback_count"] == 0
    profile_counts = summary["runnercore3_profile"]["counts"]
    assert profile_counts["runnercore24_jungle_predator_native_event_rows"] > 0
    assert profile_counts["runnercore24_jungle_predator_native_death_rows"] > 0
    assert profile_counts["runnercore45_jungle_predator_context_cache_rows"] > 0


def test_runnercore26_vectorized_jungle_predator_sampling_matches_runnercore24_live(tmp_path: Path) -> None:
    baseline = _prepare_sim(
        tmp_path / "runnercore24_jungle_predator_native_baseline_live",
        runnercore5=False,
        runnercore18=True,
        runnercore22=True,
        runnercore24=True,
        steps=2,
        policy_tick_stride=1,
        patch_world=True,
    )
    candidate = _prepare_sim(
        tmp_path / "runnercore26_jungle_predator_vectorized_live",
        runnercore5=False,
        runnercore18=True,
        runnercore22=True,
        runnercore24=True,
        runnercore26=True,
        steps=2,
        policy_tick_stride=1,
        patch_world=True,
    )

    forced_positions = [(2, 2), (2, 3), (5, 5), (5, 4)]
    for sim in (baseline, candidate):
        sim.env_cfg.reproduce_try_prob = 0.0
        sim.env_cfg.jungle_predator_pressure_enable = True
        sim.env_cfg.jungle_predator_base_risk = 1.0
        sim.env_cfg.jungle_predator_risk_cap = 1.0
        sim.env_cfg.jungle_predator_health_damage = 0.10
        sim.env_cfg.jungle_terrain_protection_enable = False
        alive = sorted(sim.pop.alive_agents(), key=lambda a: int(a.agent_id))
        for agent in alive:
            sim.env.remove_agent(int(agent.agent_id))
        for idx, (agent, pos) in enumerate(zip(alive, forced_positions)):
            agent.energy = 80.0 - float(idx)
            agent.health = 1.0
            agent.repro_cooldown = 0
            agent.stats.survival_steps = idx
            sim.env.add_agent(int(agent.agent_id), pos=pos)
        sync_fn = getattr(sim, "_phase3_pre_start_generation_sync", None)
        if callable(sync_fn):
            sync_fn(list(sim.pop.alive_agents()))

    def _forbidden(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("legacy action materializer called")

    candidate._batched_actions = _forbidden
    candidate._batched_actions_k = _forbidden
    candidate._batched_actions_strided = _forbidden

    baseline_buckets: dict[str, float] = {}
    candidate_buckets: dict[str, float] = {}
    result_baseline = baseline._run_ticks(
        gen=0,
        buckets=baseline_buckets,
        injury_recover_mult=1.0,
        deaths_this_gen=0,
        successful_births=0,
        failed_births=0,
        gen_peak_injury=0.0,
        rest_streaks={},
    )
    result_candidate = candidate._run_ticks(
        gen=0,
        buckets=candidate_buckets,
        injury_recover_mult=1.0,
        deaths_this_gen=0,
        successful_births=0,
        failed_births=0,
        gen_peak_injury=0.0,
        rest_streaks={},
    )

    assert result_candidate == result_baseline
    candidate_fp = _extended_fingerprint(candidate)
    baseline_fp = _extended_fingerprint(baseline)
    assert np.allclose(candidate_fp.pop("energy"), baseline_fp.pop("energy"), atol=5e-6)
    assert np.allclose(candidate_fp.pop("health"), baseline_fp.pop("health"), atol=5e-6)
    assert candidate_fp == baseline_fp
    assert np.allclose(candidate.env.rng.random(16), baseline.env.rng.random(16), atol=0.0)
    candidate_rng = candidate.rng_manager.indices_snapshot()
    baseline_rng = baseline.rng_manager.indices_snapshot()
    for stream in ("mutation.py", "mutation.np"):
        assert candidate_rng.get(stream) == baseline_rng.get(stream)
    assert int(getattr(candidate, "_jungle_predator_events_gen", 0)) > 0
    assert len(getattr(candidate, "_jungle_predator_event_contexts_gen", []) or []) == len(
        getattr(baseline, "_jungle_predator_event_contexts_gen", []) or []
    )

    summary = candidate._runnercore11_c1_typed_genome_row_summary
    fallback = summary["fallback_counters"]
    assert fallback["legacy_batched_actions_calls"] == 0
    assert fallback["legacy_batched_actions_k_calls"] == 0
    assert fallback["legacy_batched_actions_strided_calls"] == 0
    assert fallback["legacy_action_cache_refill_calls"] == 0
    assert fallback["unsupported_fallback_count"] == 0
    profile_counts = summary["runnercore3_profile"]["counts"]
    assert profile_counts["runnercore24_jungle_predator_native_considered_rows"] > 0
    assert profile_counts["runnercore24_jungle_predator_native_event_rows"] > 0
    assert profile_counts["runnercore26_jungle_predator_vectorized_sample_rows"] > 0
    assert profile_counts["runnercore26_jungle_predator_vectorized_event_rows"] > 0
