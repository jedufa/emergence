from __future__ import annotations

import csv
from pathlib import Path
from types import SimpleNamespace

from env.env_config import EnvConfig
from evolution.population_pkg.population_config import PopConfig
from simulation.checkpoint_bundle_v2 import checkpoint_bundle_v2_apply, checkpoint_bundle_v2_load, checkpoint_bundle_v2_save
from simulation.metrics_pkg.metrics_logger import MetricsLogger
from simulation.metrics_pkg.rows import MetricsRow
from simulation.metrics_pkg.stats import percentile
from simulation.runner import SimulationRunner
from simulation.runner_pkg.runner_config import SimConfig


def _mk_agent(aid: int, *, chem: tuple[float, float, float, float, float, float], alive: bool = True) -> SimpleNamespace:
    stats = SimpleNamespace(
        survival_steps=10,
        energy_collected=2.0,
        offspring_count=1,
        attacks=0,
        idle_charge_paid=0.0,
        action_switches=0,
        action_oscillations=0,
        drive_volatility=0.0,
        drive_jitter=0.0,
        identity_drift_ema=0.0,
        chemistry_action_cost_mult=1.0,
        chemistry_observation_fidelity=1.0,
        chemistry_observation_bias=0.0,
        chemistry_repro_viability=1.0,
        late_pred_err_count=0,
        late_pred_err_sum=0.0,
        first_repro_step=None,
    )
    genome = SimpleNamespace(
        aggression_bias=0.1,
        exploration_bias=0.0,
        reproduction_threshold=30.0,
        mutation_rate=0.05,
        hidden_size=64,
    )
    return SimpleNamespace(
        agent_id=int(aid),
        alive=bool(alive),
        stats=stats,
        genome=genome,
        injury=0.0,
        scar=0.0,
        drives=[0.1, 0.2, 0.3, 0.4],
        chemistry=list(chem),
    )


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _make_sim(tmp_path: Path, *, seed: int = 123, generations: int = 3, steps: int = 4, pop_size: int = 10) -> SimulationRunner:
    env_cfg = EnvConfig(width=10, height=10)
    pop_cfg = PopConfig(population_size=pop_size)
    sim_cfg = SimConfig(seed=int(seed), generations=int(generations), steps_per_generation=int(steps), device="cpu", bank_device="cpu", behavior_log_enable=False)
    sim_cfg.policy_deterministic_actions = True
    sim_cfg.checkpoint_bundle_v2_period_gen = 1
    return SimulationRunner(env_cfg, pop_cfg, sim_cfg, MetricsLogger(str(tmp_path)), str(tmp_path), perf=False)


def test_chem_metrics_schema_columns_present_and_ordered() -> None:
    keys = list(MetricsRow.__annotations__.keys())
    expected = [
        "chem_energy_store_mean",
        "chem_repair_reserve_mean",
        "chem_acute_damage_mean",
        "chem_chronic_damage_mean",
        "chem_regulator_1_mean",
        "chem_regulator_2_mean",
        "chem_acute_damage_p90",
        "chem_chronic_damage_p90",
        "chem_acute_high_frac",
        "chem_chronic_high_frac",
        "chem_chronic_damage_mean_selected_parents",
        "chem_repair_reserve_mean_selected_parents",
    ]
    for col in expected:
        assert col in keys
    i0 = keys.index("chem_energy_store_mean")
    i1 = keys.index("chem_repair_reserve_mean_selected_parents")
    assert i0 < i1


def test_chem_metrics_aggregate_values_and_threshold_fractions(tmp_path: Path) -> None:
    ml = MetricsLogger(str(tmp_path))
    sim_cfg = SimConfig(seed=1, generations=1, steps_per_generation=1)
    sim_cfg.chem_acute_high_threshold = 0.7
    sim_cfg.chem_chronic_high_threshold = 0.7
    agents = [
        _mk_agent(1, chem=(1.0, 0.1, 0.2, 0.1, 0.3, 0.4)),
        _mk_agent(2, chem=(0.8, 0.2, 0.8, 0.9, 0.5, 0.6)),
        _mk_agent(3, chem=(0.6, 0.3, 0.9, 0.2, 0.7, 0.8)),
    ]
    ml.log_generation(generation=0, agents=agents, deaths_this_gen=0, selected_parent_agents=[agents[1], agents[2]], sim_cfg=sim_cfg)
    row = _read_rows(Path(tmp_path) / "metrics.csv")[-1]
    acute = [0.2, 0.8, 0.9]
    chronic = [0.1, 0.9, 0.2]
    assert abs(float(row["chem_acute_damage_mean"]) - (sum(acute) / 3.0)) < 1e-9
    assert abs(float(row["chem_chronic_damage_mean"]) - (sum(chronic) / 3.0)) < 1e-9
    assert abs(float(row["chem_acute_damage_p90"]) - float(percentile(acute, 90))) < 1e-9
    assert abs(float(row["chem_chronic_damage_p90"]) - float(percentile(chronic, 90))) < 1e-9
    assert abs(float(row["chem_acute_high_frac"]) - (2.0 / 3.0)) < 1e-9
    assert abs(float(row["chem_chronic_high_frac"]) - (1.0 / 3.0)) < 1e-9
    assert abs(float(row["chem_chronic_damage_mean_selected_parents"]) - ((0.9 + 0.2) / 2.0)) < 1e-9
    assert abs(float(row["chem_repair_reserve_mean_selected_parents"]) - ((0.2 + 0.3) / 2.0)) < 1e-9


def test_chem_metrics_resume_logger_schema_stability(tmp_path: Path) -> None:
    ml_a = MetricsLogger(str(tmp_path))
    agents = [_mk_agent(1, chem=(0.2, 0.1, 0.2, 0.1, 0.2, 0.1))]
    ml_a.log_generation(generation=0, agents=agents, deaths_this_gen=0, selected_parent_agents=agents, sim_cfg=SimConfig())
    # Simulate resume path with a fresh logger object reading existing metrics.csv.
    ml_b = MetricsLogger(str(tmp_path))
    ml_b.log_generation(generation=1, agents=agents, deaths_this_gen=0, selected_parent_agents=agents, sim_cfg=SimConfig())
    rows = _read_rows(Path(tmp_path) / "metrics.csv")
    assert len(rows) == 2
    assert "chem_energy_store_mean" in rows[0]
    assert "chem_repair_reserve_mean_selected_parents" in rows[1]


def test_chem_metrics_rng_indices_unchanged_by_logging(tmp_path: Path) -> None:
    sim = _make_sim(tmp_path / "rng_neutral", seed=2201, generations=1, steps=2, pop_size=6)
    sim.run(start_gen=0, skip_init=False)
    alive = list(sim.pop.alive_agents())
    pre = sim.rng_manager.capture_snapshot(generation_index=2)
    sim.logger.log_generation(
        generation=999,
        agents=alive,
        deaths_this_gen=0,
        selected_parent_agents=alive,
        sim_cfg=sim.sim_cfg,
    )
    post = sim.rng_manager.capture_snapshot(generation_index=2)
    assert pre == post


def test_chem_metrics_permutation_invariance_for_same_agent_set(tmp_path: Path) -> None:
    sim_cfg = SimConfig(seed=1, generations=1, steps_per_generation=1)
    sim_cfg.chem_acute_high_threshold = 0.7
    sim_cfg.chem_chronic_high_threshold = 0.7

    agents = [
        _mk_agent(
            aid=i,
            chem=(
                1.0 - 0.01 * i,
                0.10 + 0.01 * i,
                (i % 9) / 10.0,
                ((i * 3) % 9) / 10.0,
                0.20 + 0.005 * i,
                0.30 + 0.004 * i,
            ),
        )
        for i in range(1, 33)
    ]
    parents = [a for a in agents if (a.agent_id % 2 == 0)]

    run_a = tmp_path / "perm_a"
    run_b = tmp_path / "perm_b"
    ml_a = MetricsLogger(str(run_a))
    ml_b = MetricsLogger(str(run_b))

    agents_a = list(agents)
    agents_b = list(reversed(agents))
    parents_a = list(parents)
    parents_b = list(reversed(parents))

    ml_a.log_generation(generation=0, agents=agents_a, deaths_this_gen=0, selected_parent_agents=parents_a, sim_cfg=sim_cfg)
    ml_b.log_generation(generation=0, agents=agents_b, deaths_this_gen=0, selected_parent_agents=parents_b, sim_cfg=sim_cfg)

    row_a = _read_rows(run_a / "metrics.csv")[-1]
    row_b = _read_rows(run_b / "metrics.csv")[-1]
    chem_cols = [
        "chem_energy_store_mean",
        "chem_repair_reserve_mean",
        "chem_acute_damage_mean",
        "chem_chronic_damage_mean",
        "chem_regulator_1_mean",
        "chem_regulator_2_mean",
        "chem_acute_damage_p90",
        "chem_chronic_damage_p90",
        "chem_acute_high_frac",
        "chem_chronic_high_frac",
        "chem_chronic_damage_mean_selected_parents",
        "chem_repair_reserve_mean_selected_parents",
    ]
    for col in chem_cols:
        assert row_a[col] == row_b[col]


def test_chem_metrics_fresh_vs_resume_parity(tmp_path: Path) -> None:
    run_a = tmp_path / "chem_obs_a"
    sim_a = _make_sim(run_a, seed=2001, generations=3, steps=4, pop_size=10)
    sim_a.run(start_gen=0, skip_init=False)
    bundle = run_a / "checkpoint_bundles_v2" / "bundle_gen000003"
    checkpoint_bundle_v2_save(sim_a, str(bundle), generation_index=3, strict_replay_mode=True)
    payload = checkpoint_bundle_v2_load(bundle, strict=True)

    sim_a.sim_cfg.generations = 2
    sim_a.run(start_gen=3, skip_init=True)
    rows_a = _read_rows(run_a / "metrics.csv")
    tail_a = [r for r in rows_a if int(r["generation"]) >= 3]

    run_b = tmp_path / "chem_obs_b"
    sim_b = _make_sim(run_b, seed=2001, generations=3, steps=4, pop_size=10)
    checkpoint_bundle_v2_apply(sim_b, payload, restore_rng=True)
    sim_b.sim_cfg.generations = 2
    sim_b.run(start_gen=3, skip_init=True)
    rows_b = _read_rows(run_b / "metrics.csv")

    chem_cols = [
        "chem_energy_store_mean",
        "chem_repair_reserve_mean",
        "chem_acute_damage_mean",
        "chem_chronic_damage_mean",
        "chem_regulator_1_mean",
        "chem_regulator_2_mean",
        "chem_acute_damage_p90",
        "chem_chronic_damage_p90",
        "chem_acute_high_frac",
        "chem_chronic_high_frac",
        "chem_chronic_damage_mean_selected_parents",
        "chem_repair_reserve_mean_selected_parents",
    ]
    assert len(tail_a) == len(rows_b)
    for ra, rb in zip(tail_a, rows_b):
        for c in chem_cols:
            assert ra[c] == rb[c]


def test_chem_metrics_longer_run_observable_nontrivial_trajectory(tmp_path: Path) -> None:
    run_dir = tmp_path / "chem_obs_long"
    sim = _make_sim(run_dir, seed=2002, generations=6, steps=8, pop_size=16)
    sim.run(start_gen=0, skip_init=False)
    rows = _read_rows(run_dir / "metrics.csv")
    es = [float(r["chem_energy_store_mean"]) for r in rows]
    assert len(es) >= 2
    assert max(es) - min(es) > 1e-6
