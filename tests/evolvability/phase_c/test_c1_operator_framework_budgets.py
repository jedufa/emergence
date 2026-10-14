from __future__ import annotations

from pathlib import Path

import pytest

from agents.genome import Genome
from env.env_config import EnvConfig
from evolution.operators_c1 import (
    C1_FROZEN_OPERATOR_MIX,
    C1_OPERATOR_CATALOG_VERSION,
    apply_operator_pipeline,
    operator_catalog,
    schedule_operator_ids,
)
from evolution.population_pkg.population_config import PopConfig
from simulation.checkpoint_bundle_v2 import checkpoint_bundle_v2_apply, checkpoint_bundle_v2_load, checkpoint_bundle_v2_save
from simulation.metrics_pkg.noop_loggers import NoOpMetricsLogger
from simulation.rng_manager import RNGManager
from simulation.runner import SimulationRunner
from simulation.runner_pkg.runner_config import SimConfig


def _payload() -> dict:
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
        "drive_noise": 0.01,
        "drive_logit_gain": 1.0,
        "pred_lr": 0.08,
        "modules": [
            {
                "module_id": "m000",
                "module_type": "generic",
                "io": {
                    "inputs": [{"slot_id": "in", "value_type": "scalar"}],
                    "outputs": [{"slot_id": "out", "value_type": "scalar"}],
                    "caps": {"max_fanin": 6, "max_fanout": 6},
                },
            },
            {
                "module_id": "m001",
                "module_type": "generic",
                "io": {
                    "inputs": [{"slot_id": "in", "value_type": "scalar"}],
                    "outputs": [{"slot_id": "out", "value_type": "scalar"}],
                    "caps": {"max_fanin": 6, "max_fanout": 6},
                },
            },
            {
                "module_id": "m002",
                "module_type": "generic",
                "io": {
                    "inputs": [{"slot_id": "in", "value_type": "scalar"}],
                    "outputs": [{"slot_id": "out", "value_type": "scalar"}],
                    "caps": {"max_fanin": 6, "max_fanout": 6},
                },
            },
        ],
        "connections": [
            {
                "connection_id": "c000",
                "src_module_id": "m000",
                "dst_module_id": "m001",
                "src_slot": "out",
                "dst_slot": "in",
                "value_type": "scalar",
                "weight": 1.0,
            },
            {
                "connection_id": "c001",
                "src_module_id": "m001",
                "dst_module_id": "m002",
                "src_slot": "out",
                "dst_slot": "in",
                "value_type": "scalar",
                "weight": 1.0,
            },
        ],
    }


def _payload_noop() -> dict:
    p = _payload()
    p["modules"] = []
    p["connections"] = []
    return p


def _make_sim(tmp_path: Path, *, seed: int = 707, generations: int = 2, steps: int = 4, pop_size: int = 10) -> SimulationRunner:
    env_cfg = EnvConfig(width=10, height=10)
    pop_cfg = PopConfig(population_size=pop_size)
    sim_cfg = SimConfig(seed=int(seed), generations=int(generations), steps_per_generation=int(steps), device="cpu", bank_device="cpu", behavior_log_enable=False)
    sim_cfg.policy_deterministic_actions = True
    sim_cfg.checkpoint_bundle_v2_period_gen = 1
    return SimulationRunner(env_cfg, pop_cfg, sim_cfg, NoOpMetricsLogger(), str(tmp_path), perf=False)


def _genome_hashes(sim: SimulationRunner) -> dict[int, str]:
    out: dict[int, str] = {}
    for aid, a in sim.pop.agents.items():
        out[int(aid)] = str(a.genome.canonical_hash())
    return out


def test_c1_unit_schedule_determinism_from_fixed_snapshot() -> None:
    rm = RNGManager(global_seed=17001)
    snap = rm.capture_snapshot(generation_index=0)
    s1 = schedule_operator_ids(rng=rm.mutation_py, max_ops=2, rng_manager=rm)
    rm.apply_snapshot_dict(snap)
    s2 = schedule_operator_ids(rng=rm.mutation_py, max_ops=2, rng_manager=rm)
    assert s1 == s2


def test_c1_unit_operator_pipeline_determinism_for_fixed_seed() -> None:
    rm = RNGManager(global_seed=17002)
    snap = rm.capture_snapshot(generation_index=0)
    p1, t1 = apply_operator_pipeline(payload=_payload(), rng=rm.mutation_py, rng_manager=rm, max_ops=2)
    rm.apply_snapshot_dict(snap)
    p2, t2 = apply_operator_pipeline(payload=_payload(), rng=rm.mutation_py, rng_manager=rm, max_ops=2)
    assert Genome.from_dict(p1).canonical_bytes() == Genome.from_dict(p2).canonical_bytes()
    assert t1["schedule_ids"] == t2["schedule_ids"]


def test_c1_unit_noop_normalize_skip_preserves_payload_and_rng() -> None:
    canonical = Genome.from_dict(_payload_noop()).canonical_payload()
    rm_legacy = RNGManager(global_seed=17012)
    snap = rm_legacy.capture_snapshot(generation_index=0)
    legacy, legacy_telemetry = apply_operator_pipeline(
        payload=canonical,
        rng=rm_legacy.mutation_py,
        rng_manager=rm_legacy,
        max_ops=2,
        payload_is_canonical=True,
        skip_final_normalize=True,
        skip_noop_normalize=False,
    )
    legacy_indices = rm_legacy.indices_snapshot()

    rm_candidate = RNGManager(global_seed=17012)
    rm_candidate.apply_snapshot_dict(snap)
    candidate, candidate_telemetry = apply_operator_pipeline(
        payload=canonical,
        rng=rm_candidate.mutation_py,
        rng_manager=rm_candidate,
        max_ops=2,
        payload_is_canonical=True,
        skip_final_normalize=True,
        skip_noop_normalize=True,
    )

    assert Genome.from_dict(candidate).canonical_bytes() == Genome.from_dict(legacy).canonical_bytes()
    assert candidate_telemetry["schedule_ids"] == legacy_telemetry["schedule_ids"]
    assert candidate_telemetry["applied_ids"] == legacy_telemetry["applied_ids"]
    assert candidate_telemetry["noop_count"] == legacy_telemetry["noop_count"]
    assert 0 < candidate_telemetry["noop_normalize_skipped"] <= candidate_telemetry["noop_count"]
    assert candidate_telemetry["normalize_calls"] < legacy_telemetry["normalize_calls"]
    assert legacy_telemetry["normalize_calls"] == legacy_telemetry["noop_count"]
    assert int(rm_candidate.indices_snapshot().get("mutation.py", 0)) == int(legacy_indices.get("mutation.py", 0))


def test_c1_unit_per_operator_draw_count_invariant_active_vs_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EIEL_STRICT_DETERMINISM", "1")
    original = dict(C1_FROZEN_OPERATOR_MIX)
    try:
        for idx, spec in enumerate(operator_catalog()):
            C1_FROZEN_OPERATOR_MIX.clear()
            for k in list(original.keys()):
                C1_FROZEN_OPERATOR_MIX[k] = 0.0
            C1_FROZEN_OPERATOR_MIX[spec.op_id] = 1.0

            rm_a = RNGManager(global_seed=18000 + int(idx))
            pre_a = rm_a.indices_snapshot()
            _ = apply_operator_pipeline(payload=_payload(), rng=rm_a.mutation_py, rng_manager=rm_a, max_ops=1)
            post_a = rm_a.indices_snapshot()
            delta_a = int(post_a.get("mutation.py", 0)) - int(pre_a.get("mutation.py", 0))

            rm_b = RNGManager(global_seed=19000 + int(idx))
            pre_b = rm_b.indices_snapshot()
            _ = apply_operator_pipeline(payload=_payload_noop(), rng=rm_b.mutation_py, rng_manager=rm_b, max_ops=1)
            post_b = rm_b.indices_snapshot()
            delta_b = int(post_b.get("mutation.py", 0)) - int(pre_b.get("mutation.py", 0))

            assert delta_a == delta_b
            assert delta_a >= int(spec.py_draws)
    finally:
        C1_FROZEN_OPERATOR_MIX.clear()
        C1_FROZEN_OPERATOR_MIX.update(original)


def test_c1_unit_strict_budget_mismatch_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EIEL_STRICT_DETERMINISM", "1")
    rm = RNGManager(global_seed=17003)
    orig = rm.expected_delta_for_boundary

    def _patched(label: str, *, ctx=None):
        out = dict(orig(label, ctx=ctx))
        if str(label).startswith("mutation.c1."):
            out["mutation.py"] = int(out.get("mutation.py", 0)) + 1
        return out

    monkeypatch.setattr(rm, "expected_delta_for_boundary", _patched)
    with pytest.raises(RuntimeError, match="DET_RNG_BUDGET_MISMATCH"):
        _ = apply_operator_pipeline(payload=_payload(), rng=rm.mutation_py, rng_manager=rm, max_ops=1)


def test_c1_unit_strict_unknown_operator_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EIEL_STRICT_DETERMINISM", "1")
    import evolution.operators_c1 as c1_mod

    rm = RNGManager(global_seed=17004)
    monkeypatch.setattr(c1_mod, "schedule_operator_ids", lambda **_: ["unknown_op_id"])
    with pytest.raises(RuntimeError, match="DET_C1_UNKNOWN_OPERATOR"):
        _ = apply_operator_pipeline(payload=_payload(), rng=rm.mutation_py, rng_manager=rm, max_ops=1, invalid_operator_policy="reject")


def test_c1_integration_execution_respects_b_constraints() -> None:
    rm = RNGManager(global_seed=17005)
    p, telemetry = apply_operator_pipeline(payload=_payload(), rng=rm.mutation_py, rng_manager=rm, max_ops=2)
    g = Genome.from_dict(p).to_dict()
    assert len(g["modules"]) <= 12
    assert len(g["connections"]) <= 32
    assert str(p.get("genes", {}).get("c1_catalog_version", "")) == C1_OPERATOR_CATALOG_VERSION
    assert str(telemetry.get("budget_status", "")) == "ok"


def test_c1_integration_runner_freeze_non_strict_normalizes_knobs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("EIEL_STRICT_DETERMINISM", raising=False)
    monkeypatch.delenv("DEBUG_DETERMINISM", raising=False)
    env_cfg = EnvConfig(width=8, height=8)
    pop_cfg = PopConfig(population_size=4)
    sim_cfg = SimConfig(seed=177, generations=1, steps_per_generation=1, device="cpu", bank_device="cpu", behavior_log_enable=False)
    sim_cfg.genome_c1_enable = False
    sim_cfg.genome_c1_catalog_version = "other_version"
    sim_cfg.genome_c1_max_ops_per_offspring = 99
    sim_cfg.genome_c1_invalid_operator_policy = "clip"
    sim = SimulationRunner(env_cfg, pop_cfg, sim_cfg, NoOpMetricsLogger(), str(tmp_path / "c1_freeze_non_strict"), perf=False)
    assert bool(sim.sim_cfg.genome_c1_enable) is True
    assert str(sim.sim_cfg.genome_c1_catalog_version) == C1_OPERATOR_CATALOG_VERSION
    assert int(sim.sim_cfg.genome_c1_max_ops_per_offspring) == 2
    assert str(sim.sim_cfg.genome_c1_invalid_operator_policy) == "reject"


def test_c1_integration_runner_freeze_strict_mismatch_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EIEL_STRICT_DETERMINISM", "1")
    env_cfg = EnvConfig(width=8, height=8)
    pop_cfg = PopConfig(population_size=4)
    sim_cfg = SimConfig(seed=178, generations=1, steps_per_generation=1, device="cpu", bank_device="cpu", behavior_log_enable=False)
    sim_cfg.genome_c1_max_ops_per_offspring = 9
    with pytest.raises(RuntimeError, match="DET_GENOME_C1_FREEZE_MISMATCH"):
        SimulationRunner(env_cfg, pop_cfg, sim_cfg, NoOpMetricsLogger(), str(tmp_path / "c1_freeze_strict"), perf=False)


def test_c1_integration_checkpoint_resume_parity_operator_pipeline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EIEL_STRICT_DETERMINISM", "1")
    run_a = tmp_path / "c1_parity_a"
    sim_a = _make_sim(run_a, seed=20021, generations=3, steps=4, pop_size=8)
    sim_a.run(start_gen=0, skip_init=False)
    bundle = run_a / "checkpoint_bundles_v2" / "bundle_gen000003"
    checkpoint_bundle_v2_save(sim_a, str(bundle), generation_index=3, strict_replay_mode=True)
    payload = checkpoint_bundle_v2_load(bundle, strict=True)

    sim_a.sim_cfg.generations = 2
    sim_a.run(start_gen=3, skip_init=True)
    h_a = _genome_hashes(sim_a)
    r_a = sim_a.rng_manager.capture_snapshot(generation_index=999)

    run_b = tmp_path / "c1_parity_b"
    sim_b = _make_sim(run_b, seed=20021, generations=3, steps=4, pop_size=8)
    checkpoint_bundle_v2_apply(sim_b, payload, restore_rng=True)
    sim_b.sim_cfg.generations = 2
    sim_b.run(start_gen=3, skip_init=True)
    h_b = _genome_hashes(sim_b)
    r_b = sim_b.rng_manager.capture_snapshot(generation_index=999)

    assert h_a == h_b
    assert r_a == r_b
