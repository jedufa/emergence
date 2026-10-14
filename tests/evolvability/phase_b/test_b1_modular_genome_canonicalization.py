from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from agents.genome import Genome
from agents.modular_genome_v1 import MODULAR_GENOME_SCHEMA_VERSION, canon_float
from env.env_config import EnvConfig
from evolution.population_pkg.population_config import PopConfig
from simulation.checkpoint_bundle_v2 import checkpoint_bundle_v2_apply, checkpoint_bundle_v2_load, checkpoint_bundle_v2_save
from simulation.metrics_pkg.noop_loggers import NoOpMetricsLogger
from simulation.rng_manager import RNGManager
from simulation.runner import SimulationRunner
from simulation.runner_pkg.runner_config import SimConfig


def _make_sim(tmp_path: Path, *, seed: int = 123, generations: int = 2, steps: int = 4, pop_size: int = 10) -> SimulationRunner:
    env_cfg = EnvConfig(width=10, height=10)
    pop_cfg = PopConfig(population_size=pop_size)
    sim_cfg = SimConfig(
        seed=int(seed),
        generations=int(generations),
        steps_per_generation=int(steps),
        device="cpu",
        bank_device="cpu",
        behavior_log_enable=False,
    )
    sim_cfg.policy_deterministic_actions = True
    sim_cfg.checkpoint_bundle_v2_period_gen = 1
    return SimulationRunner(
        env_cfg=env_cfg,
        pop_cfg=pop_cfg,
        sim_cfg=sim_cfg,
        logger=NoOpMetricsLogger(),
        run_dir=str(tmp_path),
        perf=False,
    )


def _genome_hashes(sim: SimulationRunner) -> dict[int, str]:
    out: dict[int, str] = {}
    for aid, a in sim.pop.agents.items():
        g = getattr(a, "genome", None)
        if g is None:
            continue
        out[int(aid)] = str(g.canonical_hash())
    return out


def _stream_index_view(snapshot: dict) -> dict[str, dict[str, object]]:
    out: dict[str, dict[str, object]] = {}
    for sid, s in (snapshot.get("streams", {}) or {}).items():
        idx = (s.get("indices", {}) or {}) if isinstance(s, dict) else {}
        out[str(sid)] = {
            "py": int(idx.get("py", 0) or 0),
            "np": int(idx.get("np", 0) or 0),
            "torch_cpu": int(idx.get("torch_cpu", 0) or 0),
            "torch_cuda": dict(idx.get("torch_cuda", {}) or {}),
        }
    return out


def test_b1_canonical_serialization_idempotence() -> None:
    g = Genome.random(random.Random(11))
    b1 = g.canonical_bytes()
    b2 = g.canonical_bytes()
    assert b1 == b2
    assert g.canonical_hash() == g.canonical_hash()


def test_b1_order_independent_canonical_bytes() -> None:
    base = Genome.random(random.Random(12)).to_dict()
    p1 = dict(base)
    p2 = dict(base)
    p1["modules"] = list(reversed(list(base.get("modules", []))))
    p2["connections"] = list(reversed(list(base.get("connections", []))))
    g1 = Genome.from_dict(p1)
    g2 = Genome.from_dict(p2)
    assert g1.canonical_bytes() == g2.canonical_bytes()


def test_b1_float_canonicalization_stability() -> None:
    payload = {
        "hidden_size": 64,
        "exploration_bias": 0.123456789123,
        "aggression_bias": 0.234567891234,
        "reproduction_threshold": 30.0,
        "mutation_rate": 0.05,
        "action_temperature": 1.00000000009,
        "use_gru": False,
        "memory_dim": 0,
        "drive_inertia": 0.876543219,
        "drive_noise": 0.0123456789,
        "drive_logit_gain": 1.23456789,
        "pred_lr": 0.087654321,
    }
    g = Genome.from_dict(payload)
    d = g.to_dict()
    assert float(d["exploration_bias"]) == canon_float(payload["exploration_bias"], field="exploration_bias")
    assert float(d["action_temperature"]) == canon_float(payload["action_temperature"], field="action_temperature")


def test_b1_deterministic_ids_for_missing_ids() -> None:
    payload = {
        "schema_version": MODULAR_GENOME_SCHEMA_VERSION,
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
            {"module_type": "zmod", "params": {"x": 1.0}},
            {"module_type": "amod", "params": {"x": 2.0}},
        ],
        "connections": [
            {"src_module_id": "b", "dst_module_id": "a", "channel": "c", "weight": 1.0},
            {"src_module_id": "a", "dst_module_id": "b", "channel": "c", "weight": 2.0},
        ],
    }
    g1 = Genome.from_dict(payload)
    payload["modules"] = list(reversed(payload["modules"]))
    payload["connections"] = list(reversed(payload["connections"]))
    g2 = Genome.from_dict(payload)
    ids1 = [m["module_id"] for m in g1.to_dict()["modules"]]
    ids2 = [m["module_id"] for m in g2.to_dict()["modules"]]
    cids1 = [c["connection_id"] for c in g1.to_dict()["connections"]]
    cids2 = [c["connection_id"] for c in g2.to_dict()["connections"]]
    assert ids1 == ids2
    assert cids1 == cids2


def test_b1_legacy_to_v1_migration_is_deterministic_and_idempotent() -> None:
    legacy = {
        "hidden_size": 64,
        "exploration_bias": -0.12,
        "aggression_bias": 0.34,
        "reproduction_threshold": 33.0,
        "mutation_rate": 0.04,
        "action_temperature": 1.1,
        "use_gru": True,
        "memory_dim": 16,
        "drive_inertia": 0.88,
        "drive_noise": 0.02,
        "drive_logit_gain": 1.2,
        "pred_lr": 0.09,
    }
    g1 = Genome.from_dict(legacy)
    g2 = Genome.from_dict(g1.to_dict())
    assert g1.to_dict()["schema_version"] == MODULAR_GENOME_SCHEMA_VERSION
    assert g1.canonical_bytes() == g2.canonical_bytes()


def test_b1_strict_unsupported_schema_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EIEL_STRICT_DETERMINISM", "1")
    with pytest.raises(RuntimeError, match="DET_GENOME_SCHEMA_UNSUPPORTED"):
        Genome.from_dict({"schema_version": "modular_genome_v999", "hidden_size": 64})


def test_b1_strict_malformed_schema_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EIEL_STRICT_DETERMINISM", "1")
    with pytest.raises(RuntimeError, match="DET_GENOME_SCHEMA_MALFORMED"):
        Genome.from_dict(
            {
                "schema_version": MODULAR_GENOME_SCHEMA_VERSION,
                "hidden_size": 64,
                "modules": {"bad": "type"},
            }
        )


def test_b1_bool_string_parsing_is_deterministic() -> None:
    g_false = Genome.from_dict({"hidden_size": 64, "use_gru": "false", "memory_dim": 0})
    g_true = Genome.from_dict({"hidden_size": 64, "use_gru": "true", "memory_dim": 0})
    assert bool(g_false.use_gru) is False
    assert bool(g_true.use_gru) is True


def test_b1_strict_invalid_bool_string_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EIEL_STRICT_DETERMINISM", "1")
    with pytest.raises(RuntimeError, match="DET_GENOME_SCHEMA_MALFORMED"):
        Genome.from_dict({"hidden_size": 64, "use_gru": "not_a_bool"})


def test_b1_loader_migration_path_is_rng_index_neutral() -> None:
    rm = RNGManager(global_seed=77)
    pre = rm.capture_snapshot(generation_index=0)
    _ = Genome.from_dict({"hidden_size": 64, "exploration_bias": 0.1})
    _ = Genome.from_dict({"schema_version": MODULAR_GENOME_SCHEMA_VERSION, "hidden_size": 64, "exploration_bias": 0.1})
    post = rm.capture_snapshot(generation_index=0)
    assert pre == post


def test_b1_mixed_legacy_v1_payloads_have_stable_canonical_hashes() -> None:
    legacy = Genome.from_dict({"hidden_size": 64, "exploration_bias": 0.2, "aggression_bias": 0.1})
    v1 = Genome.from_dict(
        {
            "schema_version": MODULAR_GENOME_SCHEMA_VERSION,
            "hidden_size": 64,
            "exploration_bias": 0.2,
            "aggression_bias": 0.1,
            "modules": legacy.to_dict().get("modules", []),
            "connections": legacy.to_dict().get("connections", []),
        }
    )
    assert legacy.canonical_hash() == v1.canonical_hash()


def test_b1_functional_fresh_vs_resume_parity_with_v1_schema(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EIEL_STRICT_DETERMINISM", "1")
    run_a = tmp_path / "b1_run_a"
    sim_a = _make_sim(run_a, seed=9103, generations=3, steps=6, pop_size=10)
    sim_a.run(start_gen=0, skip_init=False)

    bundle = run_a / "checkpoint_bundles_v2" / "bundle_gen000003"
    checkpoint_bundle_v2_save(sim_a, str(bundle), generation_index=3, strict_replay_mode=True)
    payload = checkpoint_bundle_v2_load(bundle, strict=True)

    sim_a.sim_cfg.generations = 2
    sim_a.run(start_gen=3, skip_init=True)
    hash_a = _genome_hashes(sim_a)
    rng_a = sim_a.rng_manager.capture_snapshot(generation_index=999)

    run_b = tmp_path / "b1_run_b"
    sim_b = _make_sim(run_b, seed=9103, generations=3, steps=6, pop_size=10)
    checkpoint_bundle_v2_apply(sim_b, payload, restore_rng=True)
    sim_b.sim_cfg.generations = 2
    sim_b.run(start_gen=3, skip_init=True)
    hash_b = _genome_hashes(sim_b)
    rng_b = sim_b.rng_manager.capture_snapshot(generation_index=999)

    assert hash_a == hash_b
    assert rng_a == rng_b


def test_b1_integration_legacy_checkpoint_genome_migrates_rng_neutral(tmp_path: Path) -> None:
    run_a = tmp_path / "b1_legacy_load"
    sim_a = _make_sim(run_a, seed=9104, generations=1, steps=3, pop_size=8)
    sim_a.run(start_gen=0, skip_init=False)
    bundle = run_a / "checkpoint_bundles_v2" / "bundle_gen000001"
    checkpoint_bundle_v2_save(sim_a, str(bundle), generation_index=1, strict_replay_mode=True)

    pop_path = bundle / "population.json"
    pop = json.loads(pop_path.read_text(encoding="utf-8"))
    for ad in pop.get("agents", []):
        gd = ad.get("genome") or {}
        if isinstance(gd, dict):
            gd.pop("schema_version", None)
            gd.pop("modules", None)
            gd.pop("connections", None)
            ad["genome"] = gd
    pop_path.write_text(json.dumps(pop, sort_keys=True, separators=(",", ":")), encoding="utf-8")

    payload = checkpoint_bundle_v2_load(bundle, strict=True)
    run_b = tmp_path / "b1_legacy_load_b"
    sim_b = _make_sim(run_b, seed=9104, generations=1, steps=3, pop_size=8)
    pre = sim_b.rng_manager.capture_snapshot(generation_index=10)
    checkpoint_bundle_v2_apply(sim_b, payload, restore_rng=False)
    post = sim_b.rng_manager.capture_snapshot(generation_index=10)
    assert _stream_index_view(pre) == _stream_index_view(post)
    for a in sim_b.pop.agents.values():
        assert str(getattr(getattr(a, "genome", None), "schema_version", "")) == MODULAR_GENOME_SCHEMA_VERSION


def test_b1_non_strict_bundle_apply_malformed_schema_never_silently_defaults(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Explicitly non-strict for this regression: must still fail deterministically.
    monkeypatch.delenv("EIEL_STRICT_DETERMINISM", raising=False)
    monkeypatch.delenv("DEBUG_DETERMINISM", raising=False)

    run_a = tmp_path / "b1_bad_schema_bundle"
    sim_a = _make_sim(run_a, seed=9105, generations=1, steps=3, pop_size=6)
    sim_a.run(start_gen=0, skip_init=False)
    bundle = run_a / "checkpoint_bundles_v2" / "bundle_gen000001"
    checkpoint_bundle_v2_save(sim_a, str(bundle), generation_index=1, strict_replay_mode=True)

    pop_path = bundle / "population.json"
    pop = json.loads(pop_path.read_text(encoding="utf-8"))
    if pop.get("agents"):
        pop["agents"][0]["genome"] = {"schema_version": "modular_genome_v999"}
    pop_path.write_text(json.dumps(pop, sort_keys=True, separators=(",", ":")), encoding="utf-8")

    payload = checkpoint_bundle_v2_load(bundle, strict=True)
    sim_b = _make_sim(tmp_path / "b1_bad_schema_bundle_b", seed=9105, generations=1, steps=3, pop_size=6)
    with pytest.raises(RuntimeError, match="DET_GENOME_MIGRATION_ERROR"):
        checkpoint_bundle_v2_apply(sim_b, payload, restore_rng=False)
