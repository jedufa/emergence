from __future__ import annotations

import json
from pathlib import Path

import pytest

from agents.genome import Genome
from agents.modular_genome_v1 import B3_FROZEN_LOCALITY, MODULAR_GENOME_SCHEMA_VERSION, canonical_payload_from_genome_dict
from env.env_config import EnvConfig
from evolution.population_pkg.population_config import PopConfig
from simulation.checkpoint_bundle_v2 import checkpoint_bundle_v2_apply, checkpoint_bundle_v2_load, checkpoint_bundle_v2_save
from simulation.metrics_pkg.noop_loggers import NoOpMetricsLogger
from simulation.runner import SimulationRunner


def _make_sim(tmp_path: Path, *, seed: int = 33, generations: int = 2, steps: int = 4, pop_size: int = 8) -> SimulationRunner:
    env_cfg = EnvConfig(width=10, height=10)
    pop_cfg = PopConfig(population_size=pop_size)
    from simulation.runner_pkg.runner_config import SimConfig

    sim_cfg = SimConfig(seed=int(seed), generations=int(generations), steps_per_generation=int(steps), device="cpu", bank_device="cpu", behavior_log_enable=False)
    sim_cfg.policy_deterministic_actions = True
    sim_cfg.checkpoint_bundle_v2_period_gen = 1
    return SimulationRunner(env_cfg, pop_cfg, sim_cfg, NoOpMetricsLogger(), str(tmp_path), perf=False)


def _payload_locality() -> dict:
    return {
        "schema_version": MODULAR_GENOME_SCHEMA_VERSION,
        "hidden_size": 64,
        "exploration_bias": 0.1,
        "aggression_bias": 0.1,
        "reproduction_threshold": 30.0,
        "mutation_rate": 0.05,
        "action_temperature": 1.0,
        "use_gru": False,
        "memory_dim": 0,
        "drive_inertia": 0.8,
        "drive_noise": 0.01,
        "drive_logit_gain": 1.0,
        "pred_lr": 0.08,
        "b2_invalid_policy": "clip",
        "modules": [
            {"module_id": "m0", "module_type": "generic", "io": {"inputs": [{"slot_id": "in", "value_type": "scalar"}], "outputs": [{"slot_id": "out", "value_type": "scalar"}], "caps": {"max_fanin": 8, "max_fanout": 8}}},
            {"module_id": "m1", "module_type": "generic", "io": {"inputs": [{"slot_id": "in", "value_type": "scalar"}], "outputs": [{"slot_id": "out", "value_type": "scalar"}], "caps": {"max_fanin": 8, "max_fanout": 8}}},
            {"module_id": "m2", "module_type": "generic", "io": {"inputs": [{"slot_id": "in", "value_type": "scalar"}], "outputs": [{"slot_id": "out", "value_type": "scalar"}], "caps": {"max_fanin": 8, "max_fanout": 8}}},
            {"module_id": "m3", "module_type": "generic", "io": {"inputs": [{"slot_id": "in", "value_type": "scalar"}], "outputs": [{"slot_id": "out", "value_type": "scalar"}], "caps": {"max_fanin": 8, "max_fanout": 8}}},
        ],
        "connections": [
            {"connection_id": "c0", "src_module_id": "m0", "dst_module_id": "m1", "src_slot": "out", "dst_slot": "in", "value_type": "scalar", "weight": 1.0},
            {"connection_id": "c1", "src_module_id": "m0", "dst_module_id": "m2", "src_slot": "out", "dst_slot": "in", "value_type": "scalar", "weight": 1.0},
            {"connection_id": "c2", "src_module_id": "m0", "dst_module_id": "m3", "src_slot": "out", "dst_slot": "in", "value_type": "scalar", "weight": 1.0},
            {"connection_id": "c3", "src_module_id": "m2", "dst_module_id": "m3", "src_slot": "out", "dst_slot": "in", "value_type": "scalar", "weight": 1.0},
        ],
    }


def _genome_hashes(sim: SimulationRunner) -> dict[int, str]:
    out: dict[int, str] = {}
    for aid, a in sim.pop.agents.items():
        g = getattr(a, "genome", None)
        if g is None:
            continue
        out[int(aid)] = str(g.canonical_hash())
    return out


def test_b3_positions_deterministic_same_input_same_positions() -> None:
    p = _payload_locality()
    g1 = Genome.from_dict(p).to_dict()
    g2 = Genome.from_dict(p).to_dict()
    pos1 = {m["module_id"]: tuple(m["latent_pos"]) for m in g1["modules"]}
    pos2 = {m["module_id"]: tuple(m["latent_pos"]) for m in g2["modules"]}
    assert pos1 == pos2


def test_b3_positions_permutation_invariant() -> None:
    p = _payload_locality()
    p2 = dict(p)
    p2["modules"] = list(reversed(list(p["modules"])))
    g1 = Genome.from_dict(p).to_dict()
    g2 = Genome.from_dict(p2).to_dict()
    pos1 = {m["module_id"]: tuple(m["latent_pos"]) for m in g1["modules"]}
    pos2 = {m["module_id"]: tuple(m["latent_pos"]) for m in g2["modules"]}
    assert pos1 == pos2


def test_b3_distance_cost_is_deterministic_and_bounded() -> None:
    g = Genome.from_dict(_payload_locality()).to_dict()
    for c in g["connections"]:
        d = float(c.get("locality_distance", 0.0))
        k = float(c.get("locality_cost", 0.0))
        assert 0.0 <= d <= 1.0
        assert 0.0 <= k <= 1.0


def test_b3_numeric_canonicalization_stable() -> None:
    g = Genome.from_dict(_payload_locality()).to_dict()
    raw = float(g["b3_locality_cost_total"])
    g2 = Genome.from_dict(g).to_dict()
    assert float(g2["b3_locality_cost_total"]) == raw


def test_b3_b1b2b3_roundtrip_canonical_stability() -> None:
    g1 = Genome.from_dict(_payload_locality())
    g2 = Genome.from_dict(g1.to_dict())
    assert g1.canonical_bytes() == g2.canonical_bytes()


def test_b3_checkpoint_roundtrip_restores_state(tmp_path: Path) -> None:
    sim_a = _make_sim(tmp_path / "b3_ckpt_a", seed=44, generations=1, steps=3, pop_size=6)
    sim_a.run(start_gen=0, skip_init=False)
    for a in sim_a.pop.agents.values():
        a.genome = Genome.from_dict(_payload_locality())
        a.stats.structural_locality_cost_total = float(a.genome.b3_locality_cost_total)
    bundle = tmp_path / "b3_ckpt_a" / "checkpoint_bundles_v2" / "bundle_gen000001"
    checkpoint_bundle_v2_save(sim_a, str(bundle), generation_index=1, strict_replay_mode=True)
    payload = checkpoint_bundle_v2_load(bundle, strict=True)

    sim_b = _make_sim(tmp_path / "b3_ckpt_b", seed=44, generations=1, steps=3, pop_size=6)
    checkpoint_bundle_v2_apply(sim_b, payload, restore_rng=True)
    assert _genome_hashes(sim_a) == _genome_hashes(sim_b)


def test_b3_accounting_signal_present_not_selection_scalar(tmp_path: Path) -> None:
    sim = _make_sim(tmp_path / "b3_accounting", seed=45, generations=1, steps=2, pop_size=4)
    sim.run(start_gen=0, skip_init=False)
    a = next(iter(sim.pop.agents.values()))
    a.genome = Genome.from_dict(_payload_locality())
    a.stats.structural_locality_cost_total = float(a.genome.b3_locality_cost_total)
    # Must not fail: telemetry field is allowed (accounting only).
    _ = sim._effective_fitness(a)
    # Should fail if an explicit locality selection scalar appears.
    a.stats.locality_score = 1.0
    with pytest.raises(RuntimeError, match="DET_FORBIDDEN_CHEMISTRY_SELECTION_SCALAR"):
        sim._effective_fitness(a)


def test_b3_fresh_vs_resume_parity_with_b3_enabled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EIEL_STRICT_DETERMINISM", "1")
    run_a = tmp_path / "b3_parity_a"
    sim_a = _make_sim(run_a, seed=46, generations=3, steps=4, pop_size=8)
    sim_a.run(start_gen=0, skip_init=False)
    bundle = run_a / "checkpoint_bundles_v2" / "bundle_gen000003"
    checkpoint_bundle_v2_save(sim_a, str(bundle), generation_index=3, strict_replay_mode=True)
    payload = checkpoint_bundle_v2_load(bundle, strict=True)

    sim_a.sim_cfg.generations = 2
    sim_a.run(start_gen=3, skip_init=True)
    ha = _genome_hashes(sim_a)
    ra = sim_a.rng_manager.capture_snapshot(generation_index=999)

    run_b = tmp_path / "b3_parity_b"
    sim_b = _make_sim(run_b, seed=46, generations=3, steps=4, pop_size=8)
    checkpoint_bundle_v2_apply(sim_b, payload, restore_rng=True)
    sim_b.sim_cfg.generations = 2
    sim_b.run(start_gen=3, skip_init=True)
    hb = _genome_hashes(sim_b)
    rb = sim_b.rng_manager.capture_snapshot(generation_index=999)
    assert ha == hb
    assert ra == rb


def test_b3_locality_bias_filters_nonlocal_connections(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(B3_FROZEN_LOCALITY, "max_connection_distance", 0.05)
    g = canonical_payload_from_genome_dict(_payload_locality())
    total = int(len(_payload_locality()["connections"]))
    kept = int(len(g["connections"]))
    dropped = int(g.get("b3_locality_dropped_connections", 0))
    assert kept + dropped == total
    assert dropped >= 1


def test_b3_strict_conflict_caps_vs_locality_deterministic_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EIEL_STRICT_DETERMINISM", "1")
    monkeypatch.setitem(B3_FROZEN_LOCALITY, "max_connection_distance", 0.0)
    p = _payload_locality()
    p["b2_invalid_policy"] = "reject"
    with pytest.raises(RuntimeError, match="DET_GENOME_B3_LOCALITY_VIOLATION"):
        Genome.from_dict(p)


def test_b3_probe_surface_carries_locality_state_hashes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EIEL_DETERMINISM_PROBE", "1")
    sim = _make_sim(tmp_path / "b3_probe", seed=47, generations=1, steps=2, pop_size=6)
    sim.run(start_gen=0, skip_init=False)
    out = tmp_path / "b3_probe" / "debug" / "determinism_probe.jsonl"
    assert out.exists()
    lines = [ln.strip() for ln in out.read_text(encoding="utf-8").splitlines() if ln.strip()]
    rec = json.loads(lines[-1])
    hashes = rec.get("hashes", {})
    assert "pop_alive_stats_structural_locality_cost_total" in hashes


def test_b3_runner_freeze_non_strict_normalizes_knobs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("EIEL_STRICT_DETERMINISM", raising=False)
    from simulation.runner_pkg.runner_config import SimConfig

    env_cfg = EnvConfig(width=8, height=8)
    pop_cfg = PopConfig(population_size=4)
    sim_cfg = SimConfig(seed=9, generations=1, steps_per_generation=1, device="cpu", bank_device="cpu", behavior_log_enable=False)
    sim_cfg.genome_b3_locality_enable = False
    sim_cfg.genome_b3_max_connection_distance = 9.0
    sim_cfg.genome_b3_distance_cost_scale = 9.0
    sim_cfg.genome_b3_distance_cost_max = 9.0
    sim = SimulationRunner(env_cfg, pop_cfg, sim_cfg, NoOpMetricsLogger(), str(tmp_path / "b3_freeze_non_strict"), perf=False)
    assert bool(sim.sim_cfg.genome_b3_locality_enable) is True
    assert float(sim.sim_cfg.genome_b3_max_connection_distance) == float(B3_FROZEN_LOCALITY["max_connection_distance"])
    assert float(sim.sim_cfg.genome_b3_distance_cost_scale) == float(B3_FROZEN_LOCALITY["distance_cost_scale"])
    assert float(sim.sim_cfg.genome_b3_distance_cost_max) == float(B3_FROZEN_LOCALITY["distance_cost_max"])


def test_b3_runner_freeze_strict_mismatch_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EIEL_STRICT_DETERMINISM", "1")
    from simulation.runner_pkg.runner_config import SimConfig

    env_cfg = EnvConfig(width=8, height=8)
    pop_cfg = PopConfig(population_size=4)
    sim_cfg = SimConfig(seed=10, generations=1, steps_per_generation=1, device="cpu", bank_device="cpu", behavior_log_enable=False)
    sim_cfg.genome_b3_max_connection_distance = 9.0
    with pytest.raises(RuntimeError, match="DET_GENOME_B3_FREEZE_MISMATCH"):
        SimulationRunner(env_cfg, pop_cfg, sim_cfg, NoOpMetricsLogger(), str(tmp_path / "b3_freeze_strict"), perf=False)
