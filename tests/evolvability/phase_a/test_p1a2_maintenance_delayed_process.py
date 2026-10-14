from __future__ import annotations

import inspect
from pathlib import Path
from types import MethodType, SimpleNamespace

import numpy as np
import pytest

from agents.chemistry import chemistry_from_values
from agents.maintenance import MAINTENANCE_SIZE, MAINTENANCE_SLOT_NAMES, maintenance_from_values
from env.env_config import EnvConfig
from evolution.population_pkg.population_config import PopConfig
from simulation.checkpoint_bundle_v2 import checkpoint_bundle_v2_apply, checkpoint_bundle_v2_load, checkpoint_bundle_v2_save
from simulation.metrics_pkg.noop_loggers import NoOpMetricsLogger
from simulation.runner import SimulationRunner
from simulation.runner_pkg.runner_config import SimConfig
from simulation.runner_pkg.runner_ticks_loop import TickLoopMixin
from simulation.runner_pkg.runner_ticks_maintenance import TickMaintenanceMixin
from simulation.rng_manager import RNGManager
from simulation.safeboundary_hash_log import compute_safeboundary_hashes
from simulation.determinism_probe import _hash_pop_chemistry, _hash_pop_maintenance


class _DummyMaintRunner(TickMaintenanceMixin):
    def __init__(self, rm: RNGManager, *, maintenance_enable: bool = True) -> None:
        self.rng_manager = rm
        self.sim_cfg = SimpleNamespace(
            maintenance_enable=maintenance_enable,
            maintenance_delay_ticks=3,
            maintenance_base_effect=0.20,
            maintenance_diminishing_factor=0.75,
            maintenance_recent_decay=0.9,
            maintenance_energy_cost=0.1,
            maintenance_opportunity_cost_coeff=0.5,
            maintenance_domination_threshold=0.75,
            maintenance_domination_min_ticks=2,
        )
        # deterministic local-food opportunity grid (single food at center)
        grid = np.zeros((5, 5), dtype=np.int8)
        grid[2, 2] = 1
        self.env = SimpleNamespace(
            grid=grid,
            agent_pos_y=np.array([2, 2, 1, 1, 3, 3, 0, 0, 4, 4], dtype=np.int32),
            agent_pos_x=np.array([2, 1, 2, 1, 2, 3, 0, 1, 3, 4], dtype=np.int32),
        )


def _make_agent(*, aid: int, energy: float = 4.0, acute: float = 0.6, chronic: float = 0.2) -> SimpleNamespace:
    return SimpleNamespace(
        alive=True,
        agent_id=int(aid),
        energy=float(energy),
        chemistry=chemistry_from_values((0.4, 0.25, acute, chronic, 0.1, 0.2)),
        maintenance=maintenance_from_values((0.0, -1.0, 0.0, -1.0)),
        stats=SimpleNamespace(
            maintenance_requests=0,
            maintenance_applied=0,
            maintenance_energy_spent_total=0.0,
            maintenance_opportunity_cost_total=0.0,
            maintenance_applied_effect_total=0.0,
        ),
    )


def test_p1a2_maintenance_from_values_shape_copy_and_slot_clamps() -> None:
    short = maintenance_from_values((2.0,))
    assert np.array_equal(short, np.array([1.0, -1.0, 0.0, -1.0], dtype=np.float32))

    long = maintenance_from_values((1.5, 7.0, -0.5, 9.0, 99.0))
    assert np.array_equal(long, np.array([1.0, 7.0, 0.0, 9.0], dtype=np.float32))

    raw = np.array([np.nan, 3.0, 2.0, -4.0], dtype=np.float32)
    out = maintenance_from_values(raw)
    assert np.isnan(out[0])
    assert np.array_equal(out[1:], np.array([3.0, 1.0, -4.0], dtype=np.float32))
    raw[2] = 0.25
    assert out[2] == np.float32(1.0)


def _make_sim(
    tmp_path: Path,
    *,
    seed: int = 123,
    generations: int = 3,
    steps: int = 6,
    pop_size: int = 8,
    maintenance_enable: bool = True,
) -> SimulationRunner:
    env_cfg = EnvConfig(width=10, height=10)
    pop_cfg = PopConfig(population_size=pop_size)
    sim_cfg = SimConfig(
        seed=int(seed),
        generations=int(generations),
        steps_per_generation=int(steps),
        device="cpu",
        bank_device="cpu",
        behavior_log_enable=False,
        maintenance_enable=bool(maintenance_enable),
    )
    # Keep maintenance tests isolated from unrelated Phase-E metrics defaults.
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


def _maint_snapshot(sim: SimulationRunner) -> dict[int, tuple[float, ...]]:
    out: dict[int, tuple[float, ...]] = {}
    for aid in sorted(sim.pop.agents.keys()):
        a = sim.pop.agents[aid]
        out[int(aid)] = tuple(float(x) for x in np.asarray(getattr(a, "maintenance", np.zeros((4,), dtype=np.float32))).reshape(-1)[:4])
    return out


def _chem_snapshot(sim: SimulationRunner) -> dict[int, tuple[float, ...]]:
    out: dict[int, tuple[float, ...]] = {}
    for aid in sorted(sim.pop.agents.keys()):
        a = sim.pop.agents[aid]
        out[int(aid)] = tuple(float(x) for x in np.asarray(getattr(a, "chemistry", np.zeros((6,), dtype=np.float32))).reshape(-1)[:6])
    return out


def test_p1a2_maintenance_schema_exactness_and_order() -> None:
    assert MAINTENANCE_SIZE == 4
    assert MAINTENANCE_SLOT_NAMES == (
        "pending_strength",
        "due_tick",
        "recent_load",
        "last_schedule_tick",
    )


def test_p1a2_delayed_effect_applies_only_after_delay() -> None:
    rm = RNGManager(global_seed=7001)
    runner = _DummyMaintRunner(rm)
    a = _make_agent(aid=0)
    rest = np.array([5], dtype=np.int32)

    c0 = float(a.chemistry[2])
    runner._update_maintenance_state([a], rest, tick_index=0)
    c1 = float(a.chemistry[2])
    runner._update_maintenance_state([a], np.array([0], dtype=np.int32), tick_index=1)
    c2 = float(a.chemistry[2])
    runner._update_maintenance_state([a], np.array([0], dtype=np.int32), tick_index=2)
    c3 = float(a.chemistry[2])
    runner._update_maintenance_state([a], np.array([0], dtype=np.int32), tick_index=3)
    c4 = float(a.chemistry[2])

    assert c1 == c0
    assert c2 == c1
    assert c3 == c2
    assert c4 < c3


def test_p1a2_diminishing_returns_monotonicity() -> None:
    rm = RNGManager(global_seed=7002)
    runner = _DummyMaintRunner(rm)
    a = _make_agent(aid=0)

    deltas = []
    for t in (0, 4, 8):
        acute_before = float(a.chemistry[2])
        runner._update_maintenance_state([a], np.array([5], dtype=np.int32), tick_index=t)
        for k in (t + 1, t + 2, t + 3):
            runner._update_maintenance_state([a], np.array([0], dtype=np.int32), tick_index=k)
        acute_after = float(a.chemistry[2])
        deltas.append(max(0.0, acute_before - acute_after))

    assert deltas[0] >= deltas[1] >= deltas[2]


def test_p1a2_energy_and_opportunity_cost_accounting() -> None:
    rm = RNGManager(global_seed=7003)
    runner = _DummyMaintRunner(rm)
    a = _make_agent(aid=0, energy=4.0)
    # agent_id=0 at (2,2): local patch contains food => non-zero opportunity cost
    runner._update_maintenance_state([a], np.array([5], dtype=np.int32), tick_index=0)
    assert float(a.energy) < 3.9  # base cost + opportunity cost
    assert float(a.stats.maintenance_energy_spent_total) > 0.0
    assert float(a.stats.maintenance_opportunity_cost_total) > 0.0


def test_p1a2_budget_expected_and_mismatch_strict(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EIEL_STRICT_DETERMINISM", "1")
    rm = RNGManager(global_seed=7004)
    runner = _DummyMaintRunner(rm)
    alive = [_make_agent(aid=0), _make_agent(aid=1)]

    runner._update_maintenance_state(alive, np.array([5, 0], dtype=np.int32), tick_index=0)

    original = rm.expected_delta_for_boundary

    def _bad(self, label: str, *, ctx=None):
        out = original(label, ctx=ctx)
        if label == "maintenance.tick.schedule":
            out["policy.np"] = int(out.get("policy.np", 0)) + 1
        return out

    rm.expected_delta_for_boundary = MethodType(_bad, rm)  # type: ignore[assignment]
    with pytest.raises(RuntimeError, match="DET_RNG_BUDGET_MISMATCH"):
        runner._update_maintenance_state(alive, np.array([5, 5], dtype=np.int32), tick_index=1)


def test_p1a2_placeholder_draw_invariance_across_branches() -> None:
    rm = RNGManager(global_seed=7005)
    runner = _DummyMaintRunner(rm)
    alive = [_make_agent(aid=0), _make_agent(aid=1), _make_agent(aid=2)]

    i0 = rm.indices_snapshot()
    runner._update_maintenance_state(alive, np.array([5, 5, 5], dtype=np.int32), tick_index=0)
    i1 = rm.indices_snapshot()
    runner._update_maintenance_state(alive, np.array([0, 0, 0], dtype=np.int32), tick_index=1)
    i2 = rm.indices_snapshot()

    assert int(i1.get("policy.np", 0)) - int(i0.get("policy.np", 0)) == 0
    assert int(i2.get("policy.np", 0)) - int(i1.get("policy.np", 0)) == 0


def test_p1a2_integration_stage_order_maintenance_between_chemistry_and_rest_runtime_probe(tmp_path: Path) -> None:
    sim = _make_sim(tmp_path / "stage_order", generations=1, steps=1, pop_size=6)
    events: list[str] = []

    orig_chem = sim._update_chemistry_state
    orig_maint = sim._update_maintenance_state
    orig_rest = sim._apply_rest_and_recovery

    def _chem(*args, **kwargs):
        events.append("chemistry")
        return orig_chem(*args, **kwargs)

    def _maint(*args, **kwargs):
        events.append("maintenance")
        return orig_maint(*args, **kwargs)

    def _rest(*args, **kwargs):
        events.append("rest")
        return orig_rest(*args, **kwargs)

    sim._update_chemistry_state = MethodType(lambda self, *a, **k: _chem(*a, **k), sim)  # type: ignore[assignment]
    sim._update_maintenance_state = MethodType(lambda self, *a, **k: _maint(*a, **k), sim)  # type: ignore[assignment]
    sim._apply_rest_and_recovery = MethodType(lambda self, *a, **k: _rest(*a, **k), sim)  # type: ignore[assignment]

    sim.run(start_gen=0, skip_init=False)

    assert "chemistry" in events and "maintenance" in events and "rest" in events
    assert events.index("chemistry") < events.index("maintenance") < events.index("rest")


def test_p1a2_integration_source_stage_order_contains_maintenance() -> None:
    src = inspect.getsource(TickLoopMixin._run_ticks)
    i_chem = src.index("self._update_chemistry_state")
    i_maint = src.index("self._update_maintenance_state")
    i_rest = src.index("self._apply_rest_and_recovery")
    assert i_chem < i_maint < i_rest


def test_p1a2_integration_rng_index_delta_matches_budget_contract() -> None:
    rm = RNGManager(global_seed=7006)
    runner = _DummyMaintRunner(rm)
    alive = [_make_agent(aid=0), _make_agent(aid=1), _make_agent(aid=2)]
    i0 = rm.indices_snapshot()
    runner._update_maintenance_state(alive, np.array([5, 0, 5], dtype=np.int32), tick_index=0)
    i1 = rm.indices_snapshot()
    assert int(i1.get("policy.np", 0)) - int(i0.get("policy.np", 0)) == 0


def test_p1a2_integration_checkpoint_roundtrip_restores_maintenance(tmp_path: Path) -> None:
    run_a = tmp_path / "run_a_ckpt"
    sim_a = _make_sim(run_a, generations=1, steps=4, pop_size=8)
    sim_a.run(start_gen=0, skip_init=False)
    before = _maint_snapshot(sim_a)

    bundle_dir = checkpoint_bundle_v2_save(
        sim_a,
        str(run_a / "checkpoint_bundles_v2" / "bundle_gen000001"),
        generation_index=1,
        strict_replay_mode=True,
    )
    payload = checkpoint_bundle_v2_load(bundle_dir, strict=True)

    run_b = tmp_path / "run_b_ckpt"
    sim_b = _make_sim(run_b, generations=1, steps=4, pop_size=8)
    checkpoint_bundle_v2_apply(sim_b, payload, restore_rng=True)
    after = _maint_snapshot(sim_b)
    assert before == after


def test_p1a2_integration_maintenance_changes_safeboundary_population_hash(tmp_path: Path) -> None:
    sim = _make_sim(tmp_path / "hash_surface", generations=1, steps=1, pop_size=6)
    sim.pop.init_random()
    sim._ensure_bank()
    sim._place_agents()
    h0 = compute_safeboundary_hashes(sim)["population_hash"]
    aid = sorted(sim.pop.agents.keys())[0]
    sim.pop.agents[aid].maintenance = maintenance_from_values((0.8, 5.0, 0.4, 1.0))
    h1 = compute_safeboundary_hashes(sim)["population_hash"]
    assert h0 != h1


def test_p1a2_functional_fresh_vs_resume_maintenance_and_rng_parity(tmp_path: Path) -> None:
    run_a = tmp_path / "run_a"
    sim_a = _make_sim(run_a, seed=8010, generations=3, steps=6, pop_size=10)
    sim_a.run(start_gen=0, skip_init=False)

    bundle = run_a / "checkpoint_bundles_v2" / "bundle_gen000003"
    checkpoint_bundle_v2_save(sim_a, str(bundle), generation_index=3, strict_replay_mode=True)
    payload = checkpoint_bundle_v2_load(bundle, strict=True)

    sim_a.sim_cfg.generations = 2
    sim_a.run(start_gen=3, skip_init=True)
    maint_a = _maint_snapshot(sim_a)
    chem_a = _chem_snapshot(sim_a)
    rng_a = sim_a.rng_manager.capture_snapshot(generation_index=999)

    run_b = tmp_path / "run_b"
    sim_b = _make_sim(run_b, seed=8010, generations=3, steps=6, pop_size=10)
    checkpoint_bundle_v2_apply(sim_b, payload, restore_rng=True)
    sim_b.sim_cfg.generations = 2
    sim_b.run(start_gen=3, skip_init=True)
    maint_b = _maint_snapshot(sim_b)
    chem_b = _chem_snapshot(sim_b)
    rng_b = sim_b.rng_manager.capture_snapshot(generation_index=999)

    assert maint_a == maint_b
    assert chem_a == chem_b
    assert rng_a == rng_b


def test_p1a2_functional_strict_budget_drift_fails_fast(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("EIEL_STRICT_DETERMINISM", "1")
    sim = _make_sim(tmp_path / "strict_drift", seed=8011, generations=1, steps=3, pop_size=8)

    original = sim.rng_manager.expected_delta_for_boundary

    def _bad(self, label: str, *, ctx=None):
        out = original(label, ctx=ctx)
        if label == "maintenance.tick.apply_delayed":
            out["policy.np"] = int(out.get("policy.np", 0)) + 1
        return out

    sim.rng_manager.expected_delta_for_boundary = MethodType(_bad, sim.rng_manager)  # type: ignore[assignment]
    with pytest.raises(RuntimeError, match="DET_RNG_BUDGET_MISMATCH"):
        sim.run(start_gen=0, skip_init=False)


def test_p1a2_scenario_domination_detector_triggers_under_forced_pathology(tmp_path: Path) -> None:
    sim = _make_sim(tmp_path / "domination", seed=8012, generations=1, steps=10, pop_size=8)

    def _all_rest(self, alive, buckets=None):
        return np.full((len(alive),), 5, dtype=np.int32)

    sim._batched_actions_strided = MethodType(_all_rest, sim)  # type: ignore[assignment]
    sim.run(start_gen=0, skip_init=False)

    ann = dict(getattr(sim, "_maintenance_domination_annotation_last_gen", {}))
    assert bool(ann.get("triggered", False)) is True
    assert float(ann.get("ratio", 0.0)) >= float(ann.get("threshold", 1.0))


def test_p1a2_scenario_baseline_non_dominance_default_knobs(tmp_path: Path) -> None:
    sim = _make_sim(tmp_path / "baseline_non_dom", seed=8013, generations=1, steps=10, pop_size=8)
    sim.run(start_gen=0, skip_init=False)

    ann = dict(getattr(sim, "_maintenance_domination_annotation_last_gen", {}))
    assert int(ann.get("opportunities", 0)) > 0
    assert float(ann.get("ratio", 0.0)) < float(ann.get("threshold", 1.0))
    assert bool(ann.get("triggered", False)) is False


def test_p1a2_scenario_maintenance_not_irrelevant_delayed_effect_visible() -> None:
    rm_on = RNGManager(global_seed=9001)
    rm_off = RNGManager(global_seed=9001)
    runner_on = _DummyMaintRunner(rm_on, maintenance_enable=True)
    runner_off = _DummyMaintRunner(rm_off, maintenance_enable=False)
    a_on = _make_agent(aid=0, acute=0.7, chronic=0.4)
    a_off = _make_agent(aid=0, acute=0.7, chronic=0.4)

    for t in range(6):
        act = np.array([5 if t == 0 else 0], dtype=np.int32)
        runner_on._update_maintenance_state([a_on], act, tick_index=t)
        runner_off._update_maintenance_state([a_off], act, tick_index=t)

    assert float(a_on.chemistry[2]) < float(a_off.chemistry[2])
    assert float(a_on.chemistry[1]) > float(a_off.chemistry[1])


def test_p1a2_probe_hash_chemistry_changes_on_single_agent_mutation() -> None:
    a0 = SimpleNamespace(agent_id=0, alive=True, chemistry=np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=np.float32))
    a1 = SimpleNamespace(agent_id=1, alive=True, chemistry=np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.2], dtype=np.float32))
    sim = SimpleNamespace(pop=SimpleNamespace(agents={0: a0, 1: a1}))

    h0 = _hash_pop_chemistry(sim, 2)
    a1.chemistry[2] = np.float32(0.95)
    h1 = _hash_pop_chemistry(sim, 2)

    assert h0 is not None and h1 is not None
    assert h0 != h1


def test_p1a2_probe_hash_maintenance_changes_on_single_agent_mutation() -> None:
    a0 = SimpleNamespace(agent_id=0, alive=True, maintenance=np.array([0.1, 4.0, 0.2, 2.0], dtype=np.float32))
    a1 = SimpleNamespace(agent_id=1, alive=True, maintenance=np.array([0.2, 5.0, 0.1, 3.0], dtype=np.float32))
    sim = SimpleNamespace(pop=SimpleNamespace(agents={0: a0, 1: a1}))

    h0 = _hash_pop_maintenance(sim, 0)
    a0.maintenance[0] = np.float32(0.85)
    h1 = _hash_pop_maintenance(sim, 0)

    assert h0 is not None and h1 is not None
    assert h0 != h1


def test_p1a2_long_horizon_replay_gate(tmp_path: Path) -> None:
    run_a = tmp_path / "long_a"
    sim_a = _make_sim(run_a, seed=9010, generations=6, steps=12, pop_size=10)
    sim_a.run(start_gen=0, skip_init=False)

    bundle = run_a / "checkpoint_bundles_v2" / "bundle_gen000006"
    checkpoint_bundle_v2_save(sim_a, str(bundle), generation_index=6, strict_replay_mode=True)
    payload = checkpoint_bundle_v2_load(bundle, strict=True)

    sim_a.sim_cfg.generations = 2
    sim_a.run(start_gen=6, skip_init=True)
    final_a = _maint_snapshot(sim_a)
    rng_a = sim_a.rng_manager.capture_snapshot(generation_index=1000)

    run_b = tmp_path / "long_b"
    sim_b = _make_sim(run_b, seed=9010, generations=6, steps=12, pop_size=10)
    checkpoint_bundle_v2_apply(sim_b, payload, restore_rng=True)
    sim_b.sim_cfg.generations = 2
    sim_b.run(start_gen=6, skip_init=True)
    final_b = _maint_snapshot(sim_b)
    rng_b = sim_b.rng_manager.capture_snapshot(generation_index=1000)

    assert final_a == final_b
    assert rng_a == rng_b
