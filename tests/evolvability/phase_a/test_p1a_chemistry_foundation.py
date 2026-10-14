from __future__ import annotations

import inspect
from pathlib import Path
from types import MethodType, SimpleNamespace

import numpy as np
import pytest

from agents.agent import Agent
from agents.chemistry import CHEMISTRY_SIZE, CHEMISTRY_SLOT_NAMES, chemistry_from_values
from env.env_config import EnvConfig
from evolution.population_pkg.population_config import PopConfig
from simulation.checkpoint_bundle_v2 import checkpoint_bundle_v2_apply, checkpoint_bundle_v2_load, checkpoint_bundle_v2_save
from simulation.metrics_pkg.noop_loggers import NoOpMetricsLogger
from simulation.runner import SimulationRunner
from simulation.runner_pkg.runner_config import SimConfig
from simulation.runner_pkg.runner_ticks_chemistry import TickChemistryMixin
from simulation.runner_pkg.runner_ticks_loop import TickLoopMixin
from simulation.rng_manager import RNGManager
from simulation.safeboundary_hash_log import compute_safeboundary_hashes


class _DummyChemRunner(TickChemistryMixin):
    def __init__(self, rm: RNGManager) -> None:
        self.rng_manager = rm


def _make_agent(*, aid: int, energy: float = 8.0, health: float = 6.0, injury: float = 0.2) -> SimpleNamespace:
    return SimpleNamespace(
        alive=True,
        agent_id=int(aid),
        energy=float(energy),
        max_energy_norm=10.0,
        health=float(health),
        start_health_norm=10.0,
        injury=float(injury),
        drives=np.array([0.3, 0.2, 0.1, 0.4], dtype=np.float32),
        chemistry=np.array([0.45, 0.35, 0.05, 0.02, 0.5, 0.25], dtype=np.float32),
    )


def _make_sim(tmp_path: Path, *, seed: int = 123, generations: int = 2, steps: int = 3, pop_size: int = 8) -> SimulationRunner:
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
    # Keep chemistry-foundation tests isolated from unrelated Phase-E metrics defaults.
    sim_cfg.phase_e_e1_enable = False
    sim_cfg.checkpoint_bundle_v2_period_gen = 1
    return SimulationRunner(
        env_cfg=env_cfg,
        pop_cfg=pop_cfg,
        sim_cfg=sim_cfg,
        logger=NoOpMetricsLogger(),
        run_dir=str(tmp_path),
        perf=False,
    )


def _chem_snapshot(sim: SimulationRunner) -> dict[int, tuple[float, ...]]:
    out: dict[int, tuple[float, ...]] = {}
    for aid in sorted(sim.pop.agents.keys()):
        a = sim.pop.agents[aid]
        out[int(aid)] = tuple(float(x) for x in np.asarray(getattr(a, "chemistry", np.zeros((6,), dtype=np.float32))).reshape(-1)[:6])
    return out


def test_p1a_chemistry_schema_exactness_and_order() -> None:
    assert CHEMISTRY_SIZE == 6
    assert CHEMISTRY_SLOT_NAMES == (
        "energy_store",
        "repair_reserve",
        "acute_damage",
        "chronic_damage",
        "regulator_1",
        "regulator_2",
    )
    a = Agent(agent_id=1, genome=SimpleNamespace(), policy=None, energy=1.0, health=1.0)
    assert isinstance(a.chemistry, np.ndarray)
    assert a.chemistry.shape == (6,)
    assert str(a.chemistry.dtype) == "float32"


def test_p1a_chemistry_clamp_determinism() -> None:
    c = chemistry_from_values([2.0, -1.0, 0.4, 0.6, 99.0, -5.0, 123.0])
    assert np.allclose(c, np.array([1.0, 0.0, 0.4, 0.6, 1.0, 0.0], dtype=np.float32))


def test_p1a_chemistry_from_ndarray_copies_and_clamps() -> None:
    raw = np.array([2.0, -1.0, 0.4, 0.6, 99.0, -5.0], dtype=np.float32)
    c = chemistry_from_values(raw)
    assert np.allclose(c, np.array([1.0, 0.0, 0.4, 0.6, 1.0, 0.0], dtype=np.float32))
    assert np.allclose(raw, np.array([2.0, -1.0, 0.4, 0.6, 99.0, -5.0], dtype=np.float32))


def test_p1a_chemistry_from_values_shape_copy_and_nan_semantics() -> None:
    short = chemistry_from_values((0.25, 2.0))
    assert np.array_equal(short, np.array([0.25, 1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32))

    long = chemistry_from_values((0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7))
    assert np.array_equal(long, np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=np.float32))

    raw = np.array([np.nan, -1.0, 2.0, 0.5, 0.6, 0.7], dtype=np.float32)
    out = chemistry_from_values(raw)
    assert np.isnan(out[0])
    assert np.array_equal(out[1:], np.array([0.0, 1.0, 0.5, 0.6, 0.7], dtype=np.float32))
    raw[1] = 0.5
    assert out[1] == np.float32(0.0)


def test_p1a_chemistry_tick_deterministic_for_fixed_snapshot() -> None:
    rm = RNGManager(global_seed=1001)
    runner = _DummyChemRunner(rm)
    alive = [_make_agent(aid=1), _make_agent(aid=2, injury=0.5)]

    snap0 = rm.capture_snapshot(generation_index=0)
    runner._update_chemistry_state(alive, tick_index=0)
    out1 = [tuple(float(x) for x in a.chemistry.tolist()) for a in alive]
    idx1 = rm.indices_snapshot()

    rm.apply_snapshot_dict(snap0)
    alive2 = [_make_agent(aid=1), _make_agent(aid=2, injury=0.5)]
    runner._update_chemistry_state(alive2, tick_index=0)
    out2 = [tuple(float(x) for x in a.chemistry.tolist()) for a in alive2]
    idx2 = rm.indices_snapshot()

    assert out1 == out2
    assert int(idx1.get("policy.np", 0)) == int(idx2.get("policy.np", 0))


def test_p1a_chemistry_acute_vs_chronic_timescale_sanity() -> None:
    rm = RNGManager(global_seed=1002)
    runner = _DummyChemRunner(rm)
    a = _make_agent(aid=7, injury=2.0, health=2.0)
    c0 = np.asarray(a.chemistry, dtype=np.float32).copy()
    runner._update_chemistry_state([a], tick_index=0)
    c1 = np.asarray(a.chemistry, dtype=np.float32).copy()
    for _ in range(15):
        runner._update_chemistry_state([a], tick_index=0)
    assert abs(float(c1[2] - c0[2])) >= abs(float(c1[3] - c0[3]))


def test_p1a_chemistry_budget_expected_and_mismatch_strict(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EIEL_STRICT_DETERMINISM", "1")
    rm = RNGManager(global_seed=1003)
    runner = _DummyChemRunner(rm)
    alive = [_make_agent(aid=1), _make_agent(aid=2), _make_agent(aid=3)]
    runner._update_chemistry_state(alive, tick_index=0)

    original = rm.expected_delta_for_boundary

    def _bad(self, label: str, *, ctx=None):
        out = original(label, ctx=ctx)
        if label == "chemistry.tick.update":
            out["policy.np"] = int(out.get("policy.np", 0)) + 1
        return out

    rm.expected_delta_for_boundary = MethodType(_bad, rm)  # type: ignore[assignment]
    with pytest.raises(RuntimeError, match="DET_RNG_BUDGET_MISMATCH"):
        runner._update_chemistry_state(alive, tick_index=1)


def test_p1a_chemistry_placeholder_draw_invariance() -> None:
    rm = RNGManager(global_seed=1004)
    runner = _DummyChemRunner(rm)
    low = [_make_agent(aid=1, injury=0.0), _make_agent(aid=2, injury=0.0), _make_agent(aid=3, injury=0.0)]
    high = [_make_agent(aid=1, injury=2.0), _make_agent(aid=2, injury=2.0), _make_agent(aid=3, injury=2.0)]

    i0 = rm.indices_snapshot()
    runner._update_chemistry_state(low, tick_index=0)
    i1 = rm.indices_snapshot()
    runner._update_chemistry_state(high, tick_index=1)
    i2 = rm.indices_snapshot()

    d1 = int(i1.get("policy.np", 0)) - int(i0.get("policy.np", 0))
    d2 = int(i2.get("policy.np", 0)) - int(i1.get("policy.np", 0))
    assert d1 == 6
    assert d2 == 6


def test_p1a_integration_spawn_initializes_chemistry(tmp_path: Path) -> None:
    sim = _make_sim(tmp_path / "spawn_init", generations=1, steps=1, pop_size=6)
    sim.pop.init_random()
    for a in sim.pop.agents.values():
        c = np.asarray(getattr(a, "chemistry", np.zeros((0,), dtype=np.float32)), dtype=np.float32).reshape(-1)
        assert c.shape == (6,)
        assert bool(np.all(c >= 0.0))
        assert bool(np.all(c <= 1.0))


def test_p1a_integration_tick_stage_order_includes_chemistry_between_drives_and_rest() -> None:
    src = inspect.getsource(TickLoopMixin._run_ticks)
    i_drives = src.index("self._update_internal_drives")
    i_chem = src.index("self._update_chemistry_state")
    i_rest = src.index("self._apply_rest_and_recovery")
    assert i_drives < i_chem < i_rest


def test_p1a_integration_rng_index_delta_for_chemistry_boundary() -> None:
    rm = RNGManager(global_seed=1005)
    runner = _DummyChemRunner(rm)
    alive = [_make_agent(aid=1), _make_agent(aid=2), _make_agent(aid=3), _make_agent(aid=4)]
    i0 = rm.indices_snapshot()
    runner._update_chemistry_state(alive, tick_index=0)
    i1 = rm.indices_snapshot()
    assert int(i1.get("policy.np", 0)) - int(i0.get("policy.np", 0)) == 8


def test_p1a_integration_checkpoint_roundtrip_restores_chemistry(tmp_path: Path) -> None:
    run_a = tmp_path / "run_a_ckpt"
    sim_a = _make_sim(run_a, generations=1, steps=1, pop_size=6)
    sim_a.pop.init_random()
    sim_a._ensure_bank()
    sim_a._place_agents()
    sim_a._update_chemistry_state(sim_a.pop.alive_agents(), tick_index=0)
    before = _chem_snapshot(sim_a)

    bundle_dir = checkpoint_bundle_v2_save(
        sim_a,
        str(run_a / "checkpoint_bundles_v2" / "bundle_gen000000"),
        generation_index=0,
        strict_replay_mode=True,
    )
    payload = checkpoint_bundle_v2_load(bundle_dir, strict=True)

    run_b = tmp_path / "run_b_ckpt"
    sim_b = _make_sim(run_b, generations=1, steps=1, pop_size=6)
    checkpoint_bundle_v2_apply(sim_b, payload, restore_rng=True)
    after = _chem_snapshot(sim_b)
    assert before == after


def test_p1a_integration_chemistry_changes_safeboundary_population_hash(tmp_path: Path) -> None:
    sim = _make_sim(tmp_path / "hash_surface", generations=1, steps=1, pop_size=6)
    sim.pop.init_random()
    sim._ensure_bank()
    sim._place_agents()
    h0 = compute_safeboundary_hashes(sim)["population_hash"]
    aid = sorted(sim.pop.agents.keys())[0]
    sim.pop.agents[aid].chemistry[2] = np.float32(0.99)
    h1 = compute_safeboundary_hashes(sim)["population_hash"]
    assert h0 != h1


def test_p1a_functional_fresh_vs_resume_chemistry_and_rng_parity(tmp_path: Path) -> None:
    run_a = tmp_path / "run_a"
    sim_a = _make_sim(run_a, seed=2026, generations=2, steps=4, pop_size=10)
    sim_a.run(start_gen=0, skip_init=False)

    bundle = run_a / "checkpoint_bundles_v2" / "bundle_gen000002"
    checkpoint_bundle_v2_save(
        sim_a,
        str(bundle),
        generation_index=2,
        strict_replay_mode=True,
    )
    payload = checkpoint_bundle_v2_load(bundle, strict=True)

    # Fresh continuation from boundary
    sim_a.sim_cfg.generations = 1
    sim_a.run(start_gen=2, skip_init=True)
    final_a = _chem_snapshot(sim_a)
    rng_a = sim_a.rng_manager.capture_snapshot(generation_index=999)

    # Resume continuation from boundary
    run_b = tmp_path / "run_b"
    sim_b = _make_sim(run_b, seed=2026, generations=2, steps=4, pop_size=10)
    checkpoint_bundle_v2_apply(sim_b, payload, restore_rng=True)
    sim_b.sim_cfg.generations = 1
    sim_b.run(start_gen=2, skip_init=True)
    final_b = _chem_snapshot(sim_b)
    rng_b = sim_b.rng_manager.capture_snapshot(generation_index=999)

    assert final_a == final_b
    assert rng_a == rng_b


def test_p1a_functional_strict_budget_drift_fails_fast(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("EIEL_STRICT_DETERMINISM", "1")
    sim = _make_sim(tmp_path / "strict_drift", seed=3030, generations=1, steps=2, pop_size=6)

    original = sim.rng_manager.expected_delta_for_boundary

    def _bad(self, label: str, *, ctx=None):
        out = original(label, ctx=ctx)
        if label == "chemistry.tick.update":
            out["policy.np"] = int(out.get("policy.np", 0)) + 1
        return out

    sim.rng_manager.expected_delta_for_boundary = MethodType(_bad, sim.rng_manager)  # type: ignore[assignment]
    with pytest.raises(RuntimeError, match="DET_RNG_BUDGET_MISMATCH"):
        sim.run(start_gen=0, skip_init=False)
