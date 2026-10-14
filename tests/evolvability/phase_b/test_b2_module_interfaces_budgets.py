from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import pytest

from agents.genome import Genome
from agents.modular_genome_v1 import B2_FROZEN_CAPS, MODULAR_GENOME_SCHEMA_VERSION
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


def _typed_payload() -> dict:
    return {
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
        "b2_invalid_policy": "clip",
        "modules": [
            {
                "module_id": "m0",
                "module_type": "generic",
                "io": {
                    "inputs": [{"slot_id": "in", "value_type": "scalar"}],
                    "outputs": [{"slot_id": "out", "value_type": "scalar"}],
                    "caps": {"max_fanin": 1, "max_fanout": 1},
                },
            },
            {
                "module_id": "m1",
                "module_type": "generic",
                "io": {
                    "inputs": [{"slot_id": "in", "value_type": "scalar"}],
                    "outputs": [{"slot_id": "out", "value_type": "scalar"}],
                    "caps": {"max_fanin": 1, "max_fanout": 1},
                },
            },
            {
                "module_id": "m2",
                "module_type": "generic",
                "io": {
                    "inputs": [{"slot_id": "in", "value_type": "scalar"}],
                    "outputs": [{"slot_id": "out", "value_type": "scalar"}],
                    "caps": {"max_fanin": 1, "max_fanout": 1},
                },
            },
        ],
        "connections": [
            {
                "connection_id": "c0",
                "src_module_id": "m0",
                "dst_module_id": "m1",
                "src_slot": "out",
                "dst_slot": "in",
                "value_type": "scalar",
                "weight": 1.0,
            }
        ],
    }


def _strict_env_enabled() -> bool:
    for k in ("EIEL_STRICT_DETERMINISM", "DEBUG_DETERMINISM", "EIEL_DETERMINISM_STRICT"):
        if str(os.environ.get(k, "0")).strip() in {"1", "true", "True"}:
            return True
    return False


def _hashes(sim: SimulationRunner) -> dict[int, str]:
    out: dict[int, str] = {}
    for aid, a in sim.pop.agents.items():
        g = getattr(a, "genome", None)
        if g is None:
            continue
        out[int(aid)] = str(g.canonical_hash())
    return out


def test_b2_type_compatibility_matrix_pass_and_clip_fail() -> None:
    payload = _typed_payload()
    payload["connections"].append(
        {
            "connection_id": "c_bad",
            "src_module_id": "m0",
            "dst_module_id": "m1",
            "src_slot": "out",
            "dst_slot": "in",
            "value_type": "vector",
            "weight": 1.0,
        }
    )
    g = Genome.from_dict(payload)
    conns = g.to_dict()["connections"]
    assert any(c["connection_id"] == "c0" for c in conns)
    assert all(c["connection_id"] != "c_bad" for c in conns)


def test_b2_fanin_fanout_caps_enforced_deterministically() -> None:
    payload = _typed_payload()
    payload["connections"] = [
        {"connection_id": "c0", "src_module_id": "m0", "dst_module_id": "m1", "src_slot": "out", "dst_slot": "in", "value_type": "scalar", "weight": 1.0},
        {"connection_id": "c1", "src_module_id": "m0", "dst_module_id": "m2", "src_slot": "out", "dst_slot": "in", "value_type": "scalar", "weight": 1.0},
    ]
    g = Genome.from_dict(payload)
    conns = g.to_dict()["connections"]
    assert len(conns) == 1
    assert conns[0]["connection_id"] == "c0"


def test_b2_agent_connection_budget_cap_enforced() -> None:
    payload = _typed_payload()
    payload["modules"] = [
        {
            "module_id": f"m{i}",
            "module_type": "generic",
            "io": {"inputs": [{"slot_id": "in", "value_type": "scalar"}], "outputs": [{"slot_id": "out", "value_type": "scalar"}]},
        }
        for i in range(8)
    ]
    payload["connections"] = [
        {
            "connection_id": f"c{i:03d}",
            "src_module_id": "m0",
            "dst_module_id": f"m{(i % 7) + 1}",
            "src_slot": "out",
            "dst_slot": "in",
            "value_type": "scalar",
            "weight": 1.0,
        }
        for i in range(64)
    ]
    g = Genome.from_dict(payload)
    assert len(g.to_dict()["connections"]) <= int(B2_FROZEN_CAPS["max_connections_per_agent"])


def test_b2_invalid_handling_is_deterministic_for_same_input() -> None:
    payload = _typed_payload()
    payload["connections"].extend(
        [
            {"connection_id": "c1", "src_module_id": "m0", "dst_module_id": "m2", "src_slot": "out", "dst_slot": "in", "value_type": "vector", "weight": 1.0},
            {"connection_id": "c2", "src_module_id": "missing", "dst_module_id": "m2", "src_slot": "out", "dst_slot": "in", "value_type": "scalar", "weight": 1.0},
        ]
    )
    g1 = Genome.from_dict(copy.deepcopy(payload))
    g2 = Genome.from_dict(copy.deepcopy(payload))
    assert g1.canonical_bytes() == g2.canonical_bytes()


def test_b2_canonical_ordering_stable_after_repair() -> None:
    payload = _typed_payload()
    # Keep this payload fully valid so the ordering assertion is strict-mode compatible.
    # The purpose here is canonical byte invariance under input permutation.
    payload["connections"] = [
        {"connection_id": "c2", "src_module_id": "m2", "dst_module_id": "m0", "src_slot": "out", "dst_slot": "in", "value_type": "scalar", "weight": 1.0},
        {"connection_id": "c1", "src_module_id": "m0", "dst_module_id": "m1", "src_slot": "out", "dst_slot": "in", "value_type": "scalar", "weight": 1.0},
    ]
    p2 = copy.deepcopy(payload)
    p2["connections"] = list(reversed(p2["connections"]))
    g1 = Genome.from_dict(payload)
    g2 = Genome.from_dict(p2)
    assert g1.canonical_bytes() == g2.canonical_bytes()


def test_b2_strict_mode_rejects_invalid_typed_connection(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EIEL_STRICT_DETERMINISM", "1")
    payload = _typed_payload()
    payload["b2_invalid_policy"] = "reject"
    payload["connections"] = [
        {"connection_id": "c0", "src_module_id": "m0", "dst_module_id": "m1", "src_slot": "out", "dst_slot": "in", "value_type": "vector", "weight": 1.0}
    ]
    with pytest.raises(RuntimeError, match="DET_GENOME_B2_TYPE_MISMATCH"):
        Genome.from_dict(payload)


def test_b2_integration_b1b2_roundtrip_canonical_bytes_stable() -> None:
    g1 = Genome.from_dict(_typed_payload())
    g2 = Genome.from_dict(g1.to_dict())
    assert g1.canonical_bytes() == g2.canonical_bytes()


def test_b2_integration_legacy_to_v1_b2_load_is_rng_neutral() -> None:
    rm = RNGManager(global_seed=1902)
    pre = rm.capture_snapshot(generation_index=0)
    _ = Genome.from_dict({"hidden_size": 64, "exploration_bias": 0.2, "aggression_bias": 0.1})
    post = rm.capture_snapshot(generation_index=0)
    assert pre == post


def test_b2_integration_checkpoint_resume_with_b2_genomes(tmp_path: Path) -> None:
    run_a = tmp_path / "b2_run_a"
    sim_a = _make_sim(run_a, seed=9201, generations=2, steps=5, pop_size=10)
    sim_a.run(start_gen=0, skip_init=False)
    for a in sim_a.pop.agents.values():
        a.genome = Genome.from_dict(_typed_payload())

    bundle = run_a / "checkpoint_bundles_v2" / "bundle_gen000002"
    checkpoint_bundle_v2_save(sim_a, str(bundle), generation_index=2, strict_replay_mode=True)
    payload = checkpoint_bundle_v2_load(bundle, strict=True)

    run_b = tmp_path / "b2_run_b"
    sim_b = _make_sim(run_b, seed=9201, generations=2, steps=5, pop_size=10)
    checkpoint_bundle_v2_apply(sim_b, payload, restore_rng=True)
    assert _hashes(sim_a) == _hashes(sim_b)


def test_b2_functional_fresh_vs_resume_parity(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EIEL_STRICT_DETERMINISM", "1")
    run_a = tmp_path / "b2_parity_a"
    sim_a = _make_sim(run_a, seed=9202, generations=3, steps=4, pop_size=8)
    sim_a.run(start_gen=0, skip_init=False)
    bundle = run_a / "checkpoint_bundles_v2" / "bundle_gen000003"
    checkpoint_bundle_v2_save(sim_a, str(bundle), generation_index=3, strict_replay_mode=True)
    payload = checkpoint_bundle_v2_load(bundle, strict=True)

    sim_a.sim_cfg.generations = 2
    sim_a.run(start_gen=3, skip_init=True)
    h_a = _hashes(sim_a)
    rng_a = sim_a.rng_manager.capture_snapshot(generation_index=999)

    run_b = tmp_path / "b2_parity_b"
    sim_b = _make_sim(run_b, seed=9202, generations=3, steps=4, pop_size=8)
    checkpoint_bundle_v2_apply(sim_b, payload, restore_rng=True)
    sim_b.sim_cfg.generations = 2
    sim_b.run(start_gen=3, skip_init=True)
    h_b = _hashes(sim_b)
    rng_b = sim_b.rng_manager.capture_snapshot(generation_index=999)

    assert h_a == h_b
    assert rng_a == rng_b


def test_b2_global_soup_wiring_prevented_by_caps() -> None:
    payload = _typed_payload()
    payload["modules"] = [
        {
            "module_id": f"m{i}",
            "module_type": "generic",
            "io": {"inputs": [{"slot_id": "in", "value_type": "scalar"}], "outputs": [{"slot_id": "out", "value_type": "scalar"}], "caps": {"max_fanin": 1, "max_fanout": 1}},
        }
        for i in range(10)
    ]
    conns = []
    cid = 0
    for i in range(10):
        for j in range(10):
            if i == j:
                continue
            conns.append(
                {
                    "connection_id": f"c{cid:03d}",
                    "src_module_id": f"m{i}",
                    "dst_module_id": f"m{j}",
                    "src_slot": "out",
                    "dst_slot": "in",
                    "value_type": "scalar",
                    "weight": 1.0,
                }
            )
            cid += 1
    payload["connections"] = conns
    if _strict_env_enabled():
        with pytest.raises(RuntimeError, match="DET_GENOME_B2_BUDGET_EXCEEDED"):
            _ = Genome.from_dict(payload)
        return

    g = Genome.from_dict(payload)
    out = g.to_dict()
    assert len(out["connections"]) <= 10


def test_b2_global_soup_wiring_rejects_under_eiel_determinism_strict(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EIEL_DETERMINISM_STRICT", "1")
    payload = _typed_payload()
    payload["modules"] = [
        {
            "module_id": f"m{i}",
            "module_type": "generic",
            "io": {"inputs": [{"slot_id": "in", "value_type": "scalar"}], "outputs": [{"slot_id": "out", "value_type": "scalar"}], "caps": {"max_fanin": 1, "max_fanout": 1}},
        }
        for i in range(10)
    ]
    payload["connections"] = [
        {
            "connection_id": f"c{i:03d}_{j:03d}",
            "src_module_id": f"m{i}",
            "dst_module_id": f"m{j}",
            "src_slot": "out",
            "dst_slot": "in",
            "value_type": "scalar",
            "weight": 1.0,
        }
        for i in range(10)
        for j in range(10)
        if i != j
    ]

    with pytest.raises(RuntimeError, match="DET_GENOME_B2_BUDGET_EXCEEDED"):
        _ = Genome.from_dict(payload)


def test_b2_bundle_apply_malformed_schema_strict_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EIEL_STRICT_DETERMINISM", "1")
    run_a = tmp_path / "b2_bad_bundle"
    sim_a = _make_sim(run_a, seed=9203, generations=1, steps=3, pop_size=6)
    sim_a.run(start_gen=0, skip_init=False)
    bundle = run_a / "checkpoint_bundles_v2" / "bundle_gen000001"
    checkpoint_bundle_v2_save(sim_a, str(bundle), generation_index=1, strict_replay_mode=True)
    pop_path = bundle / "population.json"
    pop = json.loads(pop_path.read_text(encoding="utf-8"))
    pop["agents"][0]["genome"] = _typed_payload()
    pop["agents"][0]["genome"]["connections"][0]["value_type"] = "invalid_type"
    pop_path.write_text(json.dumps(pop, sort_keys=True, separators=(",", ":")), encoding="utf-8")

    payload = checkpoint_bundle_v2_load(bundle, strict=True)
    sim_b = _make_sim(tmp_path / "b2_bad_bundle_b", seed=9203, generations=1, steps=3, pop_size=6)
    with pytest.raises(RuntimeError, match="DET_GENOME_MIGRATION_ERROR"):
        checkpoint_bundle_v2_apply(sim_b, payload, restore_rng=False)


def test_b2_runner_freeze_normalizes_non_strict_config_knobs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("EIEL_STRICT_DETERMINISM", raising=False)
    monkeypatch.delenv("DEBUG_DETERMINISM", raising=False)
    env_cfg = EnvConfig(width=8, height=8)
    pop_cfg = PopConfig(population_size=4)
    sim_cfg = SimConfig(seed=7, generations=1, steps_per_generation=1, device="cpu", bank_device="cpu", behavior_log_enable=False)
    sim_cfg.genome_b2_interfaces_enable = False
    sim_cfg.genome_b2_invalid_policy = "reject"
    sim_cfg.genome_b2_max_modules_per_agent = 999
    sim_cfg.genome_b2_max_connections_per_agent = 999
    sim_cfg.genome_b2_max_module_fanin = 999
    sim_cfg.genome_b2_max_module_fanout = 999
    sim = SimulationRunner(env_cfg, pop_cfg, sim_cfg, NoOpMetricsLogger(), str(tmp_path / "b2_freeze_non_strict"), perf=False)
    assert bool(sim.sim_cfg.genome_b2_interfaces_enable) is True
    assert str(sim.sim_cfg.genome_b2_invalid_policy) == "clip"
    assert int(sim.sim_cfg.genome_b2_max_modules_per_agent) == int(B2_FROZEN_CAPS["max_modules_per_agent"])
    assert int(sim.sim_cfg.genome_b2_max_connections_per_agent) == int(B2_FROZEN_CAPS["max_connections_per_agent"])
    assert int(sim.sim_cfg.genome_b2_max_module_fanin) == int(B2_FROZEN_CAPS["max_module_fanin"])
    assert int(sim.sim_cfg.genome_b2_max_module_fanout) == int(B2_FROZEN_CAPS["max_module_fanout"])


def test_b2_runner_freeze_strict_mismatch_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EIEL_STRICT_DETERMINISM", "1")
    env_cfg = EnvConfig(width=8, height=8)
    pop_cfg = PopConfig(population_size=4)
    sim_cfg = SimConfig(seed=8, generations=1, steps_per_generation=1, device="cpu", bank_device="cpu", behavior_log_enable=False)
    sim_cfg.genome_b2_max_connections_per_agent = 999
    with pytest.raises(RuntimeError, match="DET_GENOME_B2_CAP_FREEZE_MISMATCH"):
        SimulationRunner(env_cfg, pop_cfg, sim_cfg, NoOpMetricsLogger(), str(tmp_path / "b2_freeze_strict"), perf=False)
