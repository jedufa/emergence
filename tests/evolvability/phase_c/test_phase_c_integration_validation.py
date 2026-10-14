from __future__ import annotations

from pathlib import Path

import pytest

from agents.genome import Genome
from env.env_config import EnvConfig
from evolution.operators_c1 import (
    C1_OPERATOR_CATALOG_VERSION,
    apply_operator_pipeline,
)
from evolution.population_pkg.population_config import PopConfig
from evolution.recombination_c2 import (
    C2_RECOMBINATION_VERSION,
    recombine_lineage_payloads,
)
from simulation.checkpoint_bundle_v2 import (
    checkpoint_bundle_v2_apply,
    checkpoint_bundle_v2_load,
    checkpoint_bundle_v2_save,
)
from simulation.metrics_pkg.noop_loggers import NoOpMetricsLogger
from simulation.rng_manager import RNGManager
from simulation.runner import SimulationRunner
from simulation.runner_pkg.runner_config import SimConfig


def _parent_payload_a() -> dict:
    return {
        "schema_version": "modular_genome_v1",
        "hidden_size": 64,
        "exploration_bias": 0.1,
        "aggression_bias": 0.2,
        "reproduction_threshold": 30.0,
        "mutation_rate": 0.05,
        "action_temperature": 1.0,
        "use_gru": False,
        "memory_dim": 0,
        "drive_inertia": 0.8,
        "drive_noise": 0.02,
        "drive_logit_gain": 1.0,
        "pred_lr": 0.08,
        "modules": [
            {"module_id": "m000", "module_type": "generic", "params": {"p": 1.0}},
            {"module_id": "m010", "module_type": "generic", "params": {"p": 2.0}},
            {"module_id": "m020", "module_type": "generic", "params": {"p": 3.0}},
        ],
        "connections": [
            {"connection_id": "c000", "src_module_id": "m000", "dst_module_id": "m010", "src_slot": "out", "dst_slot": "in", "value_type": "scalar", "weight": 1.0},
            {"connection_id": "c010", "src_module_id": "m010", "dst_module_id": "m020", "src_slot": "out", "dst_slot": "in", "value_type": "scalar", "weight": 1.0},
        ],
    }


def _parent_payload_b() -> dict:
    return {
        "schema_version": "modular_genome_v1",
        "hidden_size": 64,
        "exploration_bias": -0.1,
        "aggression_bias": 0.1,
        "reproduction_threshold": 32.0,
        "mutation_rate": 0.04,
        "action_temperature": 1.1,
        "use_gru": False,
        "memory_dim": 0,
        "drive_inertia": 0.85,
        "drive_noise": 0.01,
        "drive_logit_gain": 1.1,
        "pred_lr": 0.09,
        "modules": [
            {"module_id": "m000", "module_type": "generic", "params": {"p": 4.0}},
            {"module_id": "m010", "module_type": "generic", "params": {"p": 5.0}},
            {"module_id": "m030", "module_type": "generic", "params": {"p": 6.0}},
        ],
        "connections": [
            {"connection_id": "c000", "src_module_id": "m000", "dst_module_id": "m010", "src_slot": "out", "dst_slot": "in", "value_type": "scalar", "weight": 1.0},
            {"connection_id": "c020", "src_module_id": "m010", "dst_module_id": "m030", "src_slot": "out", "dst_slot": "in", "value_type": "scalar", "weight": 1.0},
        ],
    }


def _run_c1_c2_once(seed: int) -> tuple[str, int]:
    rm = RNGManager(global_seed=int(seed))
    pre = rm.indices_snapshot()
    pa, _ = apply_operator_pipeline(payload=_parent_payload_a(), rng=rm.mutation_py, rng_manager=rm, max_ops=2)
    pb, _ = apply_operator_pipeline(payload=_parent_payload_b(), rng=rm.mutation_py, rng_manager=rm, max_ops=2)
    child_payload, summary = recombine_lineage_payloads(parent_a_payload=pa, parent_b_payload=pb, rng=rm.mutation_py, rng_manager=rm)
    post = rm.indices_snapshot()
    g = Genome.from_dict(child_payload)
    assert str(g.to_dict().get("c1_catalog_version", "")) == C1_OPERATOR_CATALOG_VERSION
    assert str(g.to_dict().get("c2_recombination_version", "")) == C2_RECOMBINATION_VERSION
    assert int(summary.budget_draws) >= 0
    return str(g.canonical_hash()), int(post.get("mutation.py", 0)) - int(pre.get("mutation.py", 0))


def _make_sim(tmp_path: Path, *, seed: int, generations: int, steps: int, pop_size: int = 8) -> SimulationRunner:
    env_cfg = EnvConfig(width=10, height=10)
    pop_cfg = PopConfig(population_size=int(pop_size))
    sim_cfg = SimConfig(seed=int(seed), generations=int(generations), steps_per_generation=int(steps), device="cpu", bank_device="cpu", behavior_log_enable=False)
    sim_cfg.policy_deterministic_actions = True
    sim_cfg.checkpoint_bundle_v2_period_gen = 1
    sim_cfg.genome_c1_enable = True
    sim_cfg.genome_c1_catalog_version = C1_OPERATOR_CATALOG_VERSION
    sim_cfg.genome_c1_max_ops_per_offspring = 2
    sim_cfg.genome_c1_invalid_operator_policy = "reject"
    sim_cfg.genome_c2_enable = True
    sim_cfg.genome_c2_recombination_version = C2_RECOMBINATION_VERSION
    sim_cfg.genome_c2_alignment_key = "homology_id"
    sim_cfg.genome_c2_unmatched_policy = "keep_all_sorted"
    return SimulationRunner(env_cfg, pop_cfg, sim_cfg, NoOpMetricsLogger(), str(tmp_path), perf=False)


def _genome_hashes(sim: SimulationRunner) -> dict[int, str]:
    out: dict[int, str] = {}
    for aid, a in sim.pop.agents.items():
        out[int(aid)] = str(a.genome.canonical_hash())
    return out


def test_phase_c_integration_c1_c2_pipeline_deterministic_draws_and_hash() -> None:
    h1, d1 = _run_c1_c2_once(seed=31001)
    h2, d2 = _run_c1_c2_once(seed=31001)
    assert h1 == h2
    assert d1 == d2
    assert d1 > 0


def test_phase_c_integration_c1_c2_strict_budget_mismatch_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EIEL_STRICT_DETERMINISM", "1")
    rm = RNGManager(global_seed=31002)
    orig = rm.expected_delta_for_boundary

    def _patched(label: str, *, ctx=None):
        out = dict(orig(label, ctx=ctx))
        if str(label) == "mutation.c2.recombination_lineage":
            out["mutation.py"] = int(out.get("mutation.py", 0)) + 1
        return out

    monkeypatch.setattr(rm, "expected_delta_for_boundary", _patched)
    pa, _ = apply_operator_pipeline(payload=_parent_payload_a(), rng=rm.mutation_py, rng_manager=rm, max_ops=2)
    pb, _ = apply_operator_pipeline(payload=_parent_payload_b(), rng=rm.mutation_py, rng_manager=rm, max_ops=2)
    with pytest.raises(RuntimeError, match="DET_RNG_BUDGET_MISMATCH"):
        _ = recombine_lineage_payloads(parent_a_payload=pa, parent_b_payload=pb, rng=rm.mutation_py, rng_manager=rm)


def test_phase_c_integration_fresh_vs_resume_parity_with_c1_c2_enabled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EIEL_STRICT_DETERMINISM", "1")
    run_a = tmp_path / "phase_c_pair_a"
    sim_a = _make_sim(run_a, seed=31003, generations=3, steps=4, pop_size=8)
    sim_a.run(start_gen=0, skip_init=False)
    bundle = run_a / "checkpoint_bundles_v2" / "bundle_gen000003"
    checkpoint_bundle_v2_save(sim_a, str(bundle), generation_index=3, strict_replay_mode=True)
    payload = checkpoint_bundle_v2_load(bundle, strict=True)

    sim_a.sim_cfg.generations = 2
    sim_a.run(start_gen=3, skip_init=True)
    h_a = _genome_hashes(sim_a)
    r_a = sim_a.rng_manager.capture_snapshot(generation_index=999)

    run_b = tmp_path / "phase_c_pair_b"
    sim_b = _make_sim(run_b, seed=31003, generations=3, steps=4, pop_size=8)
    checkpoint_bundle_v2_apply(sim_b, payload, restore_rng=True)
    sim_b.sim_cfg.generations = 2
    sim_b.run(start_gen=3, skip_init=True)
    h_b = _genome_hashes(sim_b)
    r_b = sim_b.rng_manager.capture_snapshot(generation_index=999)

    assert h_a == h_b
    assert r_a == r_b
