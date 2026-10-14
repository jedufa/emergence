from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from env.env_config import EnvConfig
from evolution.population_pkg.population_config import PopConfig
from simulation.checkpoint_bundle_v2 import checkpoint_bundle_v2_apply, checkpoint_bundle_v2_load, checkpoint_bundle_v2_save
from simulation.metrics_pkg.noop_loggers import NoOpMetricsLogger
from simulation.metrics_pkg.phase_e_metrics_engine import E2_SCHEMA_VERSION, PhaseEFailureTaxonomyEngine
from simulation.rng_manager import RNGManager
from simulation.runner import SimulationRunner
from simulation.runner_pkg.runner_config import SimConfig


class _DummyGenome:
    def __init__(self, modules: int) -> None:
        self._mods = int(modules)

    def to_dict(self) -> dict:
        return {
            "modules": [
                {"module_id": f"m{i:03d}", "module_type": "generic", "params": {}}
                for i in range(self._mods)
            ],
            "connections": [],
        }


class _DummyAgent:
    def __init__(self, aid: int, modules: int) -> None:
        self.agent_id = int(aid)
        self.alive = True
        self.genome = _DummyGenome(modules)


class _DummyPop:
    def __init__(self, agents: list[_DummyAgent]) -> None:
        self._agents = list(agents)

    def alive_agents(self) -> list[_DummyAgent]:
        return list(self._agents)


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out: list[dict] = []
    for ln in path.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if ln:
            out.append(json.loads(ln))
    return out


def _make_dummy_sim(tmp_path: Path) -> SimpleNamespace:
    sim_cfg = SimpleNamespace(
        phase_e_e2_enable=True,
        phase_e_e2_cadence_gen=1,
        phase_e_e2_drift_collapse_min_streak=2,
        phase_e_e2_drift_collapse_min_ratio=0.5,
        phase_e_e2_structural_stagnation_min_streak=3,
        phase_e_e2_bloat_cap_ratio=0.95,
        phase_e_e2_mutation_explosion_ratio=0.75,
        phase_e_e2_identity_mimicry_max_ratio=0.25,
        genome_b2_max_modules_per_agent=12,
        selection_forbid_chemistry_scalar=False,
    )
    agents = [_DummyAgent(i + 1, modules=12) for i in range(8)]
    e1 = SimpleNamespace(
        last_record={
            "schema_version": "phase_e_e1_metrics_v1",
            "generation": 3,
            "population_alive": 8,
            "unique_structures": 1,
            "introduced_this_gen": 7,
            "recovery_taxonomy": {
                "last": "collapse",
                "counts": {"return": 0, "shift": 0, "collapse": 3},
                "ratios": {"return": 0.0, "shift": 0.0, "collapse": 1.0},
            },
        },
        last_sentinel={
            "rng_delta_nonzero_keys": [],
            "population_hash_equal": True,
        },
    )
    sim = SimpleNamespace(
        sim_cfg=sim_cfg,
        pop=_DummyPop(agents),
        rng_manager=RNGManager(global_seed=123),
        run_dir=str(tmp_path),
        _phase_e_metrics_engine=e1,
        _maintenance_domination_annotation_last_gen={
            "triggered": True,
            "ratio": 0.9,
            "threshold": 0.75,
            "requests": 9,
            "opportunities": 10,
        },
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
    sim_cfg.phase_e_e2_enable = True
    sim_cfg.phase_e_e1_cadence_gen = 1
    sim_cfg.phase_e_e2_cadence_gen = 1
    sim_cfg.phase_e_e1_knockout_every_gen = 1
    sim_cfg.checkpoint_bundle_v2_period_gen = 1
    return SimulationRunner(env_cfg=env_cfg, pop_cfg=pop_cfg, sim_cfg=sim_cfg, logger=NoOpMetricsLogger(), run_dir=str(tmp_path), perf=False)


def test_e2_unit_deterministic_categories_fixed_input(tmp_path: Path) -> None:
    sim = _make_dummy_sim(tmp_path)
    e2a = PhaseEFailureTaxonomyEngine(run_dir=str(tmp_path / "a"))
    e2b = PhaseEFailureTaxonomyEngine(run_dir=str(tmp_path / "b"))

    e2a.compute_post_boundary(sim=sim, generation_index=3)
    e2b.compute_post_boundary(sim=sim, generation_index=3)

    assert e2a.last_record == e2b.last_record
    rec = dict(e2a.last_record or {})
    assert rec.get("schema_version") == E2_SCHEMA_VERSION
    cats = set(rec.get("categories", []))
    assert "maintenance_domination" in cats
    assert "mutation_explosion" in cats
    assert "identity_mimicry" in cats
    assert "bloat_ceiling_hit" in cats
    assert "hidden_scalar_reward_channel" in cats


def test_e2_distinct_annotation_artifacts(tmp_path: Path) -> None:
    sim = _make_dummy_sim(tmp_path)
    e2 = PhaseEFailureTaxonomyEngine(run_dir=str(tmp_path))
    e2.compute_post_boundary(sim=sim, generation_index=3)

    tax_rows = _read_jsonl(tmp_path / "metrics" / "phase_e_e2_taxonomy.jsonl")
    ann_rows = _read_jsonl(tmp_path / "metrics" / "phase_e_e2_annotations.jsonl")

    assert len(tax_rows) == 1
    cats = list(tax_rows[0].get("categories", []))
    assert len(cats) >= 3
    ann_cats = sorted(str(r.get("category", "")) for r in ann_rows)
    assert ann_cats == sorted(cats)


def test_e2_unit_structural_stagnation_trigger(tmp_path: Path) -> None:
    sim = _make_dummy_sim(tmp_path)
    sim.sim_cfg.phase_e_e2_structural_stagnation_min_streak = 2
    sim._phase_e_metrics_engine.last_record["introduced_this_gen"] = 0
    sim._phase_e_metrics_engine.last_record["recovery_taxonomy"]["last"] = "return"
    sim._phase_e_metrics_engine.last_record["recovery_taxonomy"]["ratios"]["collapse"] = 0.0

    eng = PhaseEFailureTaxonomyEngine(run_dir=str(tmp_path))
    eng.compute_post_boundary(sim=sim, generation_index=1)
    eng.compute_post_boundary(sim=sim, generation_index=2)
    cats = set((eng.last_record or {}).get("categories", []))
    assert "structural_stagnation" in cats


def test_e2_unit_rng_desync_trigger(tmp_path: Path) -> None:
    sim = _make_dummy_sim(tmp_path)
    sim._phase_e_metrics_engine.last_sentinel = {
        "rng_delta_nonzero_keys": ["metrics.np"],
        "population_hash_equal": False,
    }
    eng = PhaseEFailureTaxonomyEngine(run_dir=str(tmp_path))
    eng.compute_post_boundary(sim=sim, generation_index=3)
    rec = dict(eng.last_record or {})
    cats = set(rec.get("categories", []))
    assert "rng_desync" in cats
    det = dict((rec.get("details", {}) or {}).get("rng_desync", {}) or {})
    assert det.get("population_hash_equal") is False
    assert det.get("rng_delta_nonzero_keys") == ["metrics.np"]


def test_e2_budget_mismatch_detects_rng_drift(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("EIEL_STRICT_DETERMINISM", "1")

    sim = _make_dummy_sim(tmp_path)
    eng = PhaseEFailureTaxonomyEngine(run_dir=str(tmp_path))
    original = eng._detect_categories

    def _drift(*args, **kwargs):  # type: ignore[no-untyped-def]
        sim.rng_manager.metrics_np.random()
        return original(*args, **kwargs)

    monkeypatch.setattr(eng, "_detect_categories", _drift)
    with pytest.raises(RuntimeError, match="DET_RNG_BUDGET_MISMATCH"):
        eng.compute_post_boundary(sim=sim, generation_index=3)


def test_e2_integration_artifacts_and_schema(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_e2"
    sim = _make_sim(run_dir, seed=505, generations=2)
    sim.run(start_gen=0, skip_init=False)

    latest = json.loads((run_dir / "phase_e_e2_latest.json").read_text(encoding="utf-8"))
    assert latest.get("schema_version") == E2_SCHEMA_VERSION
    assert isinstance(latest.get("categories", []), list)

    tax_rows = _read_jsonl(run_dir / "metrics" / "phase_e_e2_taxonomy.jsonl")
    assert len(tax_rows) >= 2


def test_e2_checkpoint_resume_state_and_parity(tmp_path: Path) -> None:
    run_f = tmp_path / "fresh"
    sim_f = _make_sim(run_f, seed=606, generations=3)
    sim_f.sim_cfg.generations = 2
    sim_f.run(start_gen=0, skip_init=False)

    bundle = run_f / "checkpoint_bundles_v2" / "bundle_gen000002"
    checkpoint_bundle_v2_save(sim_f, str(bundle), generation_index=2, strict_replay_mode=True)

    ms = json.loads((bundle / "metrics_state.json").read_text(encoding="utf-8"))
    assert "phase_e_e2_state" in ms

    payload = checkpoint_bundle_v2_load(str(bundle), strict=True)

    sim_f.sim_cfg.generations = 3
    sim_f.run(start_gen=2, skip_init=True)

    run_r = tmp_path / "resume"
    sim_r = _make_sim(run_r, seed=606, generations=3)
    checkpoint_bundle_v2_apply(sim_r, payload, restore_rng=True)
    sim_r.sim_cfg.generations = 3
    sim_r.run(start_gen=2, skip_init=True)

    latest_f = json.loads((run_f / "phase_e_e2_latest.json").read_text(encoding="utf-8"))
    latest_r = json.loads((run_r / "phase_e_e2_latest.json").read_text(encoding="utf-8"))
    assert latest_f == latest_r

    rows_f = [r for r in _read_jsonl(run_f / "metrics" / "phase_e_e2_taxonomy.jsonl") if int(r.get("generation", -1)) == 3]
    rows_r = [r for r in _read_jsonl(run_r / "metrics" / "phase_e_e2_taxonomy.jsonl") if int(r.get("generation", -1)) == 3]
    assert rows_f == rows_r
