from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from env.env_config import EnvConfig
from evolution.population_pkg.population_config import PopConfig
from simulation.checkpoint_bundle_v2 import (
    _apply_metrics_state,
    checkpoint_bundle_v2_apply,
    checkpoint_bundle_v2_load,
    checkpoint_bundle_v2_save,
)
from simulation.metrics_pkg.noop_loggers import NoOpMetricsLogger
from simulation.metrics_pkg.phase_e_metrics_engine import E1_SCHEMA_VERSION, E1_SENTINEL_SCHEMA_VERSION, PhaseEMetricsEngine
from simulation.rng_manager import RNGManager
from simulation.runner import SimulationRunner
from simulation.runner_pkg.runner_config import SimConfig


class _DummyGenome:
    def __init__(self, payload: dict, h: str) -> None:
        self._payload = dict(payload)
        self._h = str(h)
        self.hash_calls = 0
        self.c1_operator_noop_count = int(payload.get("c1_operator_noop_count", 0) or 0)
        for key in (
            "c1_op_count__bounded_rewire",
            "c1_op_count__duplicate_with_decay",
            "c1_op_count__edge_split_insertion",
            "c1_op_count__fusion_merge_compression",
            "c1_op_count__gate_drift",
            "c1_op_count__motif_duplication",
            "c1_op_count__promoter_expression_mutation",
        ):
            setattr(self, key, int(payload.get(key, 0) or 0))

    def to_dict(self) -> dict:
        return dict(self._payload)

    def canonical_hash(self) -> str:
        self.hash_calls += 1
        return self._h


class _DummyAgent:
    def __init__(self, aid: int, payload: dict, h: str) -> None:
        self.agent_id = int(aid)
        self.alive = True
        self.genome = _DummyGenome(payload, h)


class _DummyPop:
    def __init__(self, agents: list[_DummyAgent]) -> None:
        self._agents = list(agents)

    def alive_agents(self) -> list[_DummyAgent]:
        return list(self._agents)


def _make_dummy_sim(tmp_path: Path) -> SimpleNamespace:
    payload_a = {
        "modules": [
            {"module_id": "m0", "module_type": "policy_core", "params": {}},
            {"module_id": "m1", "module_type": "behavior_bias", "params": {}},
        ],
        "connections": [
            {
                "connection_id": "c0",
                "src_module_id": "m0",
                "dst_module_id": "m1",
            }
        ],
        "b3_locality_cost_total": 0.1,
        "c1_operator_noop_count": 1,
        "c1_op_count__bounded_rewire": 1,
    }
    payload_b = {
        "modules": [
            {"module_id": "m0", "module_type": "policy_core", "params": {}},
            {"module_id": "m2", "module_type": "identity_regulation", "params": {}},
        ],
        "connections": [],
        "b3_locality_cost_total": 0.0,
        "c1_operator_noop_count": 2,
        "c1_op_count__bounded_rewire": 2,
    }
    agents = [_DummyAgent(1, payload_a, "h_a"), _DummyAgent(2, payload_b, "h_b")]
    sim_cfg = SimpleNamespace(
        phase_e_e1_enable=True,
        phase_e_e1_cadence_gen=1,
        phase_e_e1_knockout_every_gen=1,
        phase_e_e1_knockout_max_agents=8,
        phase_e_e1_knockout_max_modules_per_agent=4,
    )
    sim = SimpleNamespace(
        sim_cfg=sim_cfg,
        pop=_DummyPop(agents),
        rng_manager=RNGManager(global_seed=7),
        run_dir=str(tmp_path),
        _a3_recovery_taxonomy_last_gen={"classification": "return", "generation": 1},
    )
    return sim


def _make_sim(tmp_path: Path, *, seed: int, generations: int = 3, steps: int = 4, pop_size: int = 8) -> SimulationRunner:
    env_cfg = EnvConfig(width=10, height=10)
    pop_cfg = PopConfig(population_size=int(pop_size))
    sim_cfg = SimConfig(
        seed=int(seed),
        generations=int(generations),
        steps_per_generation=int(steps),
        device="cpu",
        bank_device="cpu",
        behavior_log_enable=False,
    )
    sim_cfg.policy_deterministic_actions = True
    sim_cfg.phase_e_e1_enable = True
    sim_cfg.phase_e_e1_cadence_gen = 1
    sim_cfg.phase_e_e1_knockout_every_gen = 1
    return SimulationRunner(env_cfg=env_cfg, pop_cfg=pop_cfg, sim_cfg=sim_cfg, logger=NoOpMetricsLogger(), run_dir=str(tmp_path), perf=False)


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out: list[dict] = []
    for ln in path.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if ln:
            out.append(json.loads(ln))
    return out


def _capture_main_fp(sim: SimulationRunner) -> dict:
    env = sim.env
    pop = sim.pop
    return {
        "env": {
            "grid": env.grid.copy(),
            "agent_grid": env.agent_grid.copy(),
            "agent_alive": env.agent_alive.copy(),
            "agent_pos_x": env.agent_pos_x.copy(),
            "agent_pos_y": env.agent_pos_y.copy(),
        },
        "rng": sim.rng_manager.capture_snapshot(),
        "agents": {
            int(aid): {
                "alive": bool(a.alive),
                "energy": float(a.energy),
                "health": float(a.health),
                "injury": float(a.injury),
                "hash": str(a.genome.canonical_hash()),
            }
            for aid, a in sorted(pop.agents.items(), key=lambda kv: int(kv[0]))
        },
    }


def test_e1_unit_deterministic_outputs_fixed_input(tmp_path: Path) -> None:
    sim = _make_dummy_sim(tmp_path)
    eng_a = PhaseEMetricsEngine(run_dir=str(tmp_path), retention_k=2, neutrality_window=4)
    eng_b = PhaseEMetricsEngine(run_dir=str(tmp_path), retention_k=2, neutrality_window=4)

    eng_a.compute_post_boundary(sim=sim, generation_index=2)
    rec_a = dict(eng_a.last_record or {})
    sent_a = dict(eng_a.last_sentinel or {})

    eng_b.compute_post_boundary(sim=sim, generation_index=2)
    rec_b = dict(eng_b.last_record or {})
    sent_b = dict(eng_b.last_sentinel or {})

    assert rec_a == rec_b
    assert sent_a == sent_b
    assert rec_a["schema_version"] == E1_SCHEMA_VERSION
    assert sent_a["schema_version"] == E1_SENTINEL_SCHEMA_VERSION


def test_e1_reuses_structure_hashes_within_before_side_metrics(tmp_path: Path) -> None:
    sim = _make_dummy_sim(tmp_path)
    eng = PhaseEMetricsEngine(run_dir=str(tmp_path), retention_k=2, neutrality_window=4)

    eng.compute_post_boundary(sim=sim, generation_index=2)

    agents = list(sim.pop.alive_agents())
    calls_by_agent = {int(a.agent_id): int(a.genome.hash_calls) for a in agents}
    assert calls_by_agent == {1: 2, 2: 2}
    assert bool((eng.last_sentinel or {}).get("population_hash_equal", False)) is True


def test_e1_changed_agent_hash_reuse_preserves_outputs_and_avoids_after_hash_calls(tmp_path: Path) -> None:
    default_sim = _make_dummy_sim(tmp_path / "default")
    default_eng = PhaseEMetricsEngine(run_dir=str(tmp_path / "default"), retention_k=2, neutrality_window=4)
    default_eng.compute_post_boundary(sim=default_sim, generation_index=2)

    reuse_sim = _make_dummy_sim(tmp_path / "reuse")
    reuse_sim.sim_cfg.phase_e_changed_agent_hash_reuse_enable = True
    reuse_eng = PhaseEMetricsEngine(run_dir=str(tmp_path / "reuse"), retention_k=2, neutrality_window=4)
    reuse_eng.compute_post_boundary(sim=reuse_sim, generation_index=2)

    assert reuse_eng.last_record == default_eng.last_record
    assert reuse_eng.last_sentinel == default_eng.last_sentinel

    agents = list(reuse_sim.pop.alive_agents())
    calls_by_agent = {int(a.agent_id): int(a.genome.hash_calls) for a in agents}
    assert calls_by_agent == {1: 1, 2: 1}

    stats = getattr(reuse_sim.sim_cfg, "_phase_e_changed_agent_hash_reuse_stats")
    assert stats["phasee_after_rows_total"] == 2
    assert stats["phasee_after_hash_reused"] == 2
    assert stats["phasee_after_hash_recomputed"] == 0


def test_e1_integration_artifacts_and_cadence(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_cadence"
    sim = _make_sim(run_dir, seed=101, generations=4)
    sim.sim_cfg.phase_e_e1_cadence_gen = 2
    sim.sim_cfg.phase_e_e1_knockout_every_gen = 2
    sim.run(start_gen=0, skip_init=False)

    metrics_rows = _read_jsonl(run_dir / "metrics" / "phase_e_e1_metrics.jsonl")
    sentinel_rows = _read_jsonl(run_dir / "metrics" / "phase_e_e1_sentinel.jsonl")

    assert [int(r.get("generation", -1)) for r in metrics_rows] == [2, 4]
    assert [int(r.get("generation", -1)) for r in sentinel_rows] == [2, 4]

    latest = json.loads((run_dir / "phase_e_e1_latest.json").read_text(encoding="utf-8"))
    assert int(latest["generation"]) == 4
    assert latest["schema_version"] == E1_SCHEMA_VERSION


def test_e1_probe_isolation_no_main_state_mutation(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_isolation"
    sim = _make_sim(run_dir, seed=202, generations=1)
    sim._place_agents()

    eng = PhaseEMetricsEngine(run_dir=str(run_dir), retention_k=2, neutrality_window=4)
    before = _capture_main_fp(sim)
    eng.compute_post_boundary(sim=sim, generation_index=1)
    after = _capture_main_fp(sim)

    for k in ("grid", "agent_grid", "agent_alive", "agent_pos_x", "agent_pos_y"):
        assert np.array_equal(before["env"][k], after["env"][k])
    assert before["rng"] == after["rng"]
    assert before["agents"] == after["agents"]


def test_e1_budget_mismatch_detects_rng_drift(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("EIEL_STRICT_DETERMINISM", "1")

    sim = _make_dummy_sim(tmp_path)
    eng = PhaseEMetricsEngine(run_dir=str(tmp_path), retention_k=2, neutrality_window=4)

    orig = eng._compute_neutrality

    def _drifting_neutrality(agents):  # type: ignore[no-untyped-def]
        sim.rng_manager.metrics_np.random()
        return orig(agents)

    monkeypatch.setattr(eng, "_compute_neutrality", _drifting_neutrality)

    with pytest.raises(RuntimeError, match="DET_RNG_BUDGET_MISMATCH"):
        eng.compute_post_boundary(sim=sim, generation_index=1)


def test_e1_checkpoint_resume_state_parity(tmp_path: Path) -> None:
    run_fresh = tmp_path / "fresh"
    sim_f = _make_sim(run_fresh, seed=303, generations=3)
    sim_f.sim_cfg.generations = 2
    sim_f.run(start_gen=0, skip_init=False)

    bundle = run_fresh / "checkpoint_bundles_v2" / "bundle_gen000002"
    checkpoint_bundle_v2_save(sim_f, str(bundle), generation_index=2, strict_replay_mode=True)
    payload = checkpoint_bundle_v2_load(str(bundle), strict=True)

    sim_f.sim_cfg.generations = 3
    sim_f.run(start_gen=2, skip_init=True)

    run_resume = tmp_path / "resume"
    sim_r = _make_sim(run_resume, seed=303, generations=3)
    checkpoint_bundle_v2_apply(sim_r, payload, restore_rng=True)
    sim_r.sim_cfg.generations = 3
    sim_r.run(start_gen=2, skip_init=True)

    latest_f = json.loads((run_fresh / "phase_e_e1_latest.json").read_text(encoding="utf-8"))
    latest_r = json.loads((run_resume / "phase_e_e1_latest.json").read_text(encoding="utf-8"))
    assert latest_f == latest_r

    rows_f = _read_jsonl(run_fresh / "metrics" / "phase_e_e1_metrics.jsonl")
    rows_r = _read_jsonl(run_resume / "metrics" / "phase_e_e1_metrics.jsonl")
    sent_f = _read_jsonl(run_fresh / "metrics" / "phase_e_e1_sentinel.jsonl")
    sent_r = _read_jsonl(run_resume / "metrics" / "phase_e_e1_sentinel.jsonl")

    # Compare post-resume generation only.
    f_last = [r for r in rows_f if int(r.get("generation", -1)) == 3]
    r_last = [r for r in rows_r if int(r.get("generation", -1)) == 3]
    fs_last = [r for r in sent_f if int(r.get("generation", -1)) == 3]
    rs_last = [r for r in sent_r if int(r.get("generation", -1)) == 3]

    assert f_last == r_last
    assert len(fs_last) == 1 and len(rs_last) == 1
    assert bool(fs_last[0].get("population_hash_equal", False)) is True
    assert bool(rs_last[0].get("population_hash_equal", False)) is True
    assert fs_last[0].get("population_hash_after", "") == rs_last[0].get("population_hash_after", "")
    assert all(int(v) == 0 for v in fs_last[0].get("rng_indices_delta", {}).values())
    assert all(int(v) == 0 for v in rs_last[0].get("rng_indices_delta", {}).values())


def test_e1_checkpoint_bundle_persists_phase_e_state(tmp_path: Path) -> None:
    run_dir = tmp_path / "bundle_state"
    sim = _make_sim(run_dir, seed=404, generations=2)
    sim.sim_cfg.checkpoint_bundle_v2_period_gen = 1
    sim.run(start_gen=0, skip_init=False)

    bundle = run_dir / "checkpoint_bundles_v2" / "bundle_gen000002"
    assert bundle.exists()
    metrics_state = json.loads((bundle / "metrics_state.json").read_text(encoding="utf-8"))
    assert metrics_state.get("schema_version") == "metrics_state_v1"
    assert "phase_e_e1_state" in metrics_state
    e1_state = dict(metrics_state.get("phase_e_e1_state", {}) or {})
    assert e1_state.get("schema_version") == "phase_e_e1_state_v1"
    assert isinstance(e1_state.get("seen_first_gen", {}), dict)


def test_e1_apply_metrics_state_restores_e1_even_when_logger_has_restore_state(tmp_path: Path) -> None:
    class _LoggerWithRestore:
        def __init__(self) -> None:
            self.calls = 0

        def restore_state(self, state):  # type: ignore[no-untyped-def]
            self.calls += 1

    sim = SimpleNamespace(
        run_dir=str(tmp_path / "sim"),
        sim_cfg=SimpleNamespace(phase_e_e1_retention_k_gen=3, phase_e_e1_neutrality_window=8),
        logger=_LoggerWithRestore(),
    )
    state = {
        "schema_version": "metrics_state_v1",
        "collapse_streak": 2,
        "phase_e_e1_state": {
            "schema_version": "phase_e_e1_state_v1",
            "retention_k": 3,
            "neutrality_window": 8,
            "seen_first_gen": {"h1": 1, "h2": 2},
            "neutrality_history": [0.1, 0.2],
            "taxonomy_counts": {"return": 5, "shift": 0, "collapse": 1},
        },
    }

    _apply_metrics_state(sim, state)
    assert sim.logger.calls == 1
    e1 = getattr(sim, "_phase_e_metrics_engine", None)
    assert e1 is not None
    st = e1.export_state()
    assert st.get("seen_first_gen", {}) == {"h1": 1, "h2": 2}
    assert st.get("taxonomy_counts", {}).get("return", -1) == 5


def test_e1_causality_contrast_functional_density_tracks_structure_complexity(tmp_path: Path) -> None:
    def _agent_payload(modules_n: int) -> dict:
        modules = [
            {"module_id": f"m{idx}", "module_type": "policy_core", "params": {}}
            for idx in range(int(modules_n))
        ]
        connections = [
            {"connection_id": f"c{idx}", "src_module_id": f"m{idx}", "dst_module_id": f"m{idx + 1}"}
            for idx in range(max(0, int(modules_n) - 1))
        ]
        return {"modules": modules, "connections": connections, "b3_locality_cost_total": 0.0}

    def _build_sim(run_dir: Path, *, modules_n: int) -> SimpleNamespace:
        agents = [
            _DummyAgent(1, _agent_payload(modules_n), f"h1_{modules_n}"),
            _DummyAgent(2, _agent_payload(modules_n), f"h2_{modules_n}"),
        ]
        sim_cfg = SimpleNamespace(
            phase_e_e1_enable=True,
            phase_e_e1_cadence_gen=1,
            phase_e_e1_knockout_every_gen=1,
            phase_e_e1_knockout_max_agents=8,
            phase_e_e1_knockout_max_modules_per_agent=4,
        )
        return SimpleNamespace(
            sim_cfg=sim_cfg,
            pop=_DummyPop(agents),
            rng_manager=RNGManager(global_seed=7),
            run_dir=str(run_dir),
            _a3_recovery_taxonomy_last_gen={"classification": "return", "generation": 1},
        )

    eng_simple = PhaseEMetricsEngine(run_dir=str(tmp_path / "simple"), retention_k=2, neutrality_window=4)
    eng_complex = PhaseEMetricsEngine(run_dir=str(tmp_path / "complex"), retention_k=2, neutrality_window=4)

    eng_simple.compute_post_boundary(sim=_build_sim(tmp_path / "simple", modules_n=1), generation_index=2)
    eng_complex.compute_post_boundary(sim=_build_sim(tmp_path / "complex", modules_n=4), generation_index=2)

    rec_simple = dict(eng_simple.last_record or {})
    rec_complex = dict(eng_complex.last_record or {})

    # Causal contrast: with deterministic scoring, denser simple structures have larger relative knockout impact.
    assert float(rec_simple.get("functional_density", -1.0)) > float(rec_complex.get("functional_density", -1.0))
