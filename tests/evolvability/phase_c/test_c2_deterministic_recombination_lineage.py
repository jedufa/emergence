from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from agents.genome import Genome
from agents.structural_identity_v1 import new_native_identity
from env.env_config import EnvConfig
from evolution.recombination_c2 import (
    C2_FROZEN_ALIGNMENT_KEY,
    C2_FROZEN_UNMATCHED_POLICY,
    C2_RECOMBINATION_VERSION,
    recombine_lineage_payloads,
)
from evolution.population_pkg.population_config import PopConfig
from simulation.checkpoint_bundle_v2 import checkpoint_bundle_v2_apply, checkpoint_bundle_v2_load, checkpoint_bundle_v2_save
from simulation.metrics_pkg.noop_loggers import NoOpMetricsLogger
from simulation.rng_manager import RNGManager
from simulation.runner import SimulationRunner
from simulation.runner_pkg.runner_config import SimConfig


def _identity(role: str, *, child: int, position: list[float]) -> dict:
    identity = new_native_identity(
        context={
            "campaign_id": "phase-c2-fixture",
            "branch_id": "test",
            "generation": 1,
            "child_agent_id": int(child),
            "parent_agent_ids": [1, 2],
        },
        operator_id="fixture",
        attempt_index=0,
        operator_role=role,
    )
    identity["topology_pos"] = list(position)
    return identity


def _parent_a() -> dict:
    return {
        "schema_version": "modular_genome_v1",
        "hidden_size": 64,
        "exploration_bias": 0.1,
        "aggression_bias": 0.1,
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
            {"module_id": "m000", "module_type": "generic", "params": {"p": 1.0}, **_identity("shared-0", child=900, position=[0.40, 0.40])},
            {"module_id": "m010", "module_type": "generic", "params": {"p": 2.0}, **_identity("shared-1", child=900, position=[0.45, 0.40])},
            {"module_id": "m020", "module_type": "generic", "params": {"p": 3.0}, **_identity("only-a", child=901, position=[0.50, 0.40])},
        ],
        "connections": [
            {"connection_id": "c000", "src_module_id": "m000", "dst_module_id": "m010", "src_slot": "out", "dst_slot": "in", "value_type": "scalar", "weight": 1.0},
            {"connection_id": "c010", "src_module_id": "m010", "dst_module_id": "m020", "src_slot": "out", "dst_slot": "in", "value_type": "scalar", "weight": 1.0},
        ],
    }


def _parent_b() -> dict:
    return {
        "schema_version": "modular_genome_v1",
        "hidden_size": 64,
        "exploration_bias": -0.1,
        "aggression_bias": 0.2,
        "reproduction_threshold": 35.0,
        "mutation_rate": 0.04,
        "action_temperature": 1.1,
        "use_gru": False,
        "memory_dim": 0,
        "drive_inertia": 0.85,
        "drive_noise": 0.01,
        "drive_logit_gain": 1.1,
        "pred_lr": 0.09,
        "modules": [
            {"module_id": "m000", "module_type": "generic", "params": {"p": 4.0}, **_identity("shared-0", child=900, position=[0.40, 0.40])},
            {"module_id": "m010", "module_type": "generic", "params": {"p": 5.0}, **_identity("shared-1", child=900, position=[0.45, 0.40])},
            {"module_id": "m030", "module_type": "generic", "params": {"p": 6.0}, **_identity("only-b", child=902, position=[0.50, 0.40])},
        ],
        "connections": [
            {"connection_id": "c000", "src_module_id": "m000", "dst_module_id": "m010", "src_slot": "out", "dst_slot": "in", "value_type": "scalar", "weight": 1.0},
            {"connection_id": "c020", "src_module_id": "m010", "dst_module_id": "m030", "src_slot": "out", "dst_slot": "in", "value_type": "scalar", "weight": 1.0},
        ],
    }


def _make_sim(tmp_path: Path, *, seed: int = 505, generations: int = 2, steps: int = 4, pop_size: int = 10) -> SimulationRunner:
    env_cfg = EnvConfig(width=10, height=10)
    pop_cfg = PopConfig(population_size=pop_size)
    sim_cfg = SimConfig(seed=int(seed), generations=int(generations), steps_per_generation=int(steps), device="cpu", bank_device="cpu", behavior_log_enable=False)
    sim_cfg.policy_deterministic_actions = True
    sim_cfg.checkpoint_bundle_v2_period_gen = 1
    return SimulationRunner(env_cfg, pop_cfg, sim_cfg, NoOpMetricsLogger(), str(tmp_path), perf=False)


def _genome_hashes(sim: SimulationRunner) -> dict[int, str]:
    return {int(aid): str(a.genome.canonical_hash()) for aid, a in sim.pop.agents.items()}


def test_c2_unit_alignment_deterministic_under_parent_permutations() -> None:
    pa = _parent_a()
    pb = _parent_b()
    pa2 = deepcopy(pa)
    pb2 = deepcopy(pb)
    pa2["modules"] = list(reversed(pa2["modules"]))
    pb2["modules"] = list(reversed(pb2["modules"]))
    rm1 = RNGManager(global_seed=21001)
    rm2 = RNGManager(global_seed=21001)
    out1, s1 = recombine_lineage_payloads(parent_a_payload=pa, parent_b_payload=pb, rng=rm1.mutation_py, rng_manager=rm1)
    out2, s2 = recombine_lineage_payloads(parent_a_payload=pa2, parent_b_payload=pb2, rng=rm2.mutation_py, rng_manager=rm2)
    assert Genome.from_dict(out1).canonical_bytes() == Genome.from_dict(out2).canonical_bytes()
    assert s1 == s2


def test_c2_unit_unmatched_resolution_keep_all_sorted() -> None:
    rm = RNGManager(global_seed=21002)
    out, summary = recombine_lineage_payloads(parent_a_payload=_parent_a(), parent_b_payload=_parent_b(), rng=rm.mutation_py, rng_manager=rm)
    mids = [str(m["module_id"]) for m in out["modules"]]
    assert mids == sorted(mids)
    assert len(mids) == 4
    assert summary.unmatched_policy == C2_FROZEN_UNMATCHED_POLICY
    assert summary.alignment_key == C2_FROZEN_ALIGNMENT_KEY
    assert int(summary.unmatched_modules_parent_a) == 1
    assert int(summary.unmatched_modules_parent_b) == 1


def test_c2_unit_budget_mismatch_strict_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EIEL_STRICT_DETERMINISM", "1")
    rm = RNGManager(global_seed=21003)
    orig = rm.expected_delta_for_boundary

    def _patched(label: str, *, ctx=None):
        out = dict(orig(label, ctx=ctx))
        if str(label) == "mutation.c2.recombination_lineage":
            out["mutation.py"] = int(out.get("mutation.py", 0)) + 1
        return out

    monkeypatch.setattr(rm, "expected_delta_for_boundary", _patched)
    with pytest.raises(RuntimeError, match="DET_RNG_BUDGET_MISMATCH"):
        _ = recombine_lineage_payloads(parent_a_payload=_parent_a(), parent_b_payload=_parent_b(), rng=rm.mutation_py, rng_manager=rm)


def test_c2_integration_canonical_hash_stability_after_recombination() -> None:
    rm1 = RNGManager(global_seed=21004)
    rm2 = RNGManager(global_seed=21004)
    out1, _ = recombine_lineage_payloads(parent_a_payload=_parent_a(), parent_b_payload=_parent_b(), rng=rm1.mutation_py, rng_manager=rm1)
    out2, _ = recombine_lineage_payloads(parent_a_payload=_parent_a(), parent_b_payload=_parent_b(), rng=rm2.mutation_py, rng_manager=rm2)
    g1 = Genome.from_dict(out1)
    g2 = Genome.from_dict(out2)
    assert g1.canonical_hash() == g2.canonical_hash()
    assert str(g1.to_dict().get("c2_recombination_version", "")) == C2_RECOMBINATION_VERSION


def test_c2_integration_checkpoint_roundtrip_recombination_derived_offspring(tmp_path: Path) -> None:
    sim_a = _make_sim(tmp_path / "c2_ckpt_a", seed=22001, generations=2, steps=4, pop_size=8)
    sim_a.run(start_gen=0, skip_init=False)
    for a in sim_a.pop.agents.values():
        rm = RNGManager(global_seed=22010 + int(a.agent_id))
        out, _ = recombine_lineage_payloads(parent_a_payload=_parent_a(), parent_b_payload=_parent_b(), rng=rm.mutation_py, rng_manager=rm)
        a.genome = Genome.from_dict(out)
    bundle = tmp_path / "c2_ckpt_a" / "checkpoint_bundles_v2" / "bundle_gen000002"
    checkpoint_bundle_v2_save(sim_a, str(bundle), generation_index=2, strict_replay_mode=True)
    payload = checkpoint_bundle_v2_load(bundle, strict=True)

    sim_b = _make_sim(tmp_path / "c2_ckpt_b", seed=22001, generations=2, steps=4, pop_size=8)
    checkpoint_bundle_v2_apply(sim_b, payload, restore_rng=True)
    assert _genome_hashes(sim_a) == _genome_hashes(sim_b)


def test_c2_functional_fresh_vs_resume_parity_with_c2_enabled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EIEL_STRICT_DETERMINISM", "1")
    run_a = tmp_path / "c2_parity_a"
    sim_a = _make_sim(run_a, seed=23001, generations=3, steps=4, pop_size=8)
    sim_a.run(start_gen=0, skip_init=False)
    bundle = run_a / "checkpoint_bundles_v2" / "bundle_gen000003"
    checkpoint_bundle_v2_save(sim_a, str(bundle), generation_index=3, strict_replay_mode=True)
    payload = checkpoint_bundle_v2_load(bundle, strict=True)

    sim_a.sim_cfg.generations = 2
    sim_a.run(start_gen=3, skip_init=True)
    h_a = _genome_hashes(sim_a)
    r_a = sim_a.rng_manager.capture_snapshot(generation_index=999)

    run_b = tmp_path / "c2_parity_b"
    sim_b = _make_sim(run_b, seed=23001, generations=3, steps=4, pop_size=8)
    checkpoint_bundle_v2_apply(sim_b, payload, restore_rng=True)
    sim_b.sim_cfg.generations = 2
    sim_b.run(start_gen=3, skip_init=True)
    h_b = _genome_hashes(sim_b)
    r_b = sim_b.rng_manager.capture_snapshot(generation_index=999)
    assert h_a == h_b
    assert r_a == r_b


def test_c2_functional_path_coverage_multiple_parent_topologies() -> None:
    rm = RNGManager(global_seed=24001)
    pa = _parent_a()
    pb = _parent_b()
    out1, s1 = recombine_lineage_payloads(parent_a_payload=pa, parent_b_payload=pb, rng=rm.mutation_py, rng_manager=rm)
    assert s1.matched_modules >= 1
    assert s1.unmatched_modules_parent_a >= 0
    assert s1.unmatched_modules_parent_b >= 0

    # Local-slot renaming does not erase biological homology.
    p0 = _parent_a()
    p1 = _parent_b()
    for m in p0["modules"]:
        m["module_id"] = f"a_{m['module_id']}"
    for m in p1["modules"]:
        m["module_id"] = f"b_{m['module_id']}"
    for c in p0["connections"]:
        c["src_module_id"] = f"a_{c['src_module_id']}"
        c["dst_module_id"] = f"a_{c['dst_module_id']}"
    for c in p1["connections"]:
        c["src_module_id"] = f"b_{c['src_module_id']}"
        c["dst_module_id"] = f"b_{c['dst_module_id']}"
    out2, s2 = recombine_lineage_payloads(parent_a_payload=p0, parent_b_payload=p1, rng=rm.mutation_py, rng_manager=rm)
    assert s2.matched_modules == 2
    assert s2.unmatched_modules_parent_a == 1
    assert s2.unmatched_modules_parent_b == 1
    assert len(out1["modules"]) >= 1
    assert len(out2["modules"]) >= 1
