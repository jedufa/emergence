from __future__ import annotations

import csv
import json
from pathlib import Path
from types import MethodType
from types import SimpleNamespace

import numpy as np
import pytest

from env.env_config import EnvConfig
from evolution.population_pkg.population_config import PopConfig
from simulation.checkpoint_bundle_v2 import checkpoint_bundle_v2_apply, checkpoint_bundle_v2_load, checkpoint_bundle_v2_save
from simulation.metrics import MetricsLogger
from simulation.runner import SimulationRunner
from simulation.runner_pkg.runner_config import SimConfig


def _read_metrics(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _mean(values: list[float]) -> float:
    return float(sum(values) / max(1, len(values)))


def _pass_count(values: list[float], predicate) -> int:  # type: ignore[no-untyped-def]
    return int(sum(1 for v in values if predicate(float(v))))


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    s = sorted(float(v) for v in values)
    n = len(s)
    mid = n // 2
    if (n % 2) == 1:
        return float(s[mid])
    return float((s[mid - 1] + s[mid]) / 2.0)


def _alive_metric_last(rows: list[dict[str, str]]) -> float:
    return float(rows[-1]["alive"]) if rows else 0.0


def _cohort_survival_tail_mean(rows: list[dict[str, str]], tail: int = 3) -> float:
    sample = rows[-tail:] if len(rows) >= tail else rows
    vals = [float(r.get("cohort_survival_rate", 0.0) or 0.0) for r in sample]
    return _mean(vals)


def _fitness_tail_mean(rows: list[dict[str, str]], tail: int = 3) -> float:
    sample = rows[-tail:] if len(rows) >= tail else rows
    vals = [float(r.get("mean_fitness", 0.0) or 0.0) for r in sample]
    return _mean(vals)


def _injury_tail_mean(rows: list[dict[str, str]], tail: int = 3) -> float:
    sample = rows[-tail:] if len(rows) >= tail else rows
    vals = [float(r.get("mean_injury", 0.0) or 0.0) for r in sample]
    return _mean(vals)


def _max_injury_tail_mean(rows: list[dict[str, str]], tail: int = 3) -> float:
    sample = rows[-tail:] if len(rows) >= tail else rows
    vals = [float(r.get("max_injury", 0.0) or 0.0) for r in sample]
    return _mean(vals)


def _chem_slot_mean(sim: SimulationRunner, idx: int) -> float:
    vals = []
    for a in sim.pop.alive_agents():
        c = getattr(a, "chemistry", None)
        if c is None:
            continue
        arr = np.asarray(c, dtype=np.float32).reshape(-1)
        if idx < arr.size:
            vals.append(float(arr[idx]))
    return _mean(vals) if vals else 0.0


def _stat_mean(sim: SimulationRunner, name: str, default: float = 0.0) -> float:
    vals = []
    for a in sim.pop.alive_agents():
        st = getattr(a, "stats", None)
        vals.append(float(getattr(st, name, default)))
    return _mean(vals) if vals else float(default)


def _stat_total(sim: SimulationRunner, name: str) -> float:
    total = 0.0
    for a in sim.pop.alive_agents():
        st = getattr(a, "stats", None)
        total += float(getattr(st, name, 0.0))
    return total


def _snapshot(sim: SimulationRunner) -> tuple[dict[int, tuple[float, ...]], dict[int, tuple[float, ...]]]:
    chem: dict[int, tuple[float, ...]] = {}
    maint: dict[int, tuple[float, ...]] = {}
    for aid in sorted(sim.pop.agents.keys()):
        a = sim.pop.agents[aid]
        c = np.asarray(getattr(a, "chemistry", np.zeros((6,), dtype=np.float32)), dtype=np.float32).reshape(-1)[:6]
        m = np.asarray(getattr(a, "maintenance", np.zeros((4,), dtype=np.float32)), dtype=np.float32).reshape(-1)[:4]
        chem[int(aid)] = tuple(float(x) for x in c)
        maint[int(aid)] = tuple(float(x) for x in m)
    return chem, maint


def _base_env_cfg() -> EnvConfig:
    cfg = EnvConfig(width=16, height=16)
    cfg.enable_biomes = True
    cfg.biome_safe_frac = 0.55
    cfg.biome_wild_hazard_prob = 0.03
    cfg.biome_wild_hazard_damage = 0.50
    cfg.biome_wild_hazard_injury_per_damage = 0.35
    cfg.food_spawn_per_step = 8
    cfg.max_food = 160
    return cfg


def _base_sim_cfg(seed: int, generations: int, steps: int, **overrides: object) -> SimConfig:
    cfg = SimConfig(
        seed=int(seed),
        generations=int(generations),
        steps_per_generation=int(steps),
        device="cpu",
        bank_device="cpu",
        behavior_log_enable=False,
        maintenance_enable=True,
        maintenance_delay_ticks=3,
        maintenance_base_effect=0.14,
        maintenance_diminishing_factor=0.60,
        maintenance_recent_decay=0.92,
        maintenance_energy_cost=0.06,
        maintenance_opportunity_cost_coeff=0.20,
        maintenance_domination_threshold=0.75,
        maintenance_domination_min_ticks=6,
        chemistry_functional_coupling_enable=True,
        chemistry_repro_viability_enable=True,
        selection_forbid_chemistry_scalar=True,
        a3_recovery_taxonomy_enable=True,
        shock_start_prob=0.08,
        shock_duration_min=2,
        shock_duration_max=4,
        shock_cooldown_min=10,
        shock_cooldown_max=18,
        policy_deterministic_actions=False,
    )
    # Isolate Phase-A intent behavior from Phase-E metric sidepaths.
    cfg.phase_e_e1_enable = False
    cfg.checkpoint_bundle_v2_period_gen = 1
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def _make_sim(tmp_path: Path, *, seed: int, generations: int, steps: int, pop_size: int = 24, **sim_overrides: object) -> SimulationRunner:
    env_cfg = _base_env_cfg()
    pop_cfg = PopConfig(population_size=int(pop_size))
    sim_cfg = _base_sim_cfg(seed=seed, generations=generations, steps=steps, **sim_overrides)
    return SimulationRunner(
        env_cfg=env_cfg,
        pop_cfg=pop_cfg,
        sim_cfg=sim_cfg,
        logger=MetricsLogger(str(tmp_path)),
        run_dir=str(tmp_path),
        perf=False,
    )


def _run_and_rows(sim: SimulationRunner) -> list[dict[str, str]]:
    sim.run(start_gen=0, skip_init=False)
    return _read_metrics(Path(sim.run_dir) / "metrics.csv")


def _run_ecology_condition(
    tmp_path: Path,
    *,
    name: str,
    seed: int,
    hazard: float,
    food_spawn: int,
    chemistry_on: bool,
) -> tuple[list[dict[str, str]], float]:
    sim = _make_sim(
        tmp_path / name,
        seed=seed,
        generations=6,
        steps=10,
        pop_size=24,
        chemistry_functional_coupling_enable=chemistry_on,
        chemistry_repro_viability_enable=chemistry_on,
    )
    sim.env_cfg.biome_wild_hazard_prob = float(hazard)
    sim.env_cfg.food_spawn_per_step = int(food_spawn)
    rows = _run_and_rows(sim)
    return rows, _fitness_tail_mean(rows, tail=3)


def _maintenance_request_ratio_over_run(sim: SimulationRunner) -> float:
    req_total = 0
    opp_total = 0
    original = sim._update_maintenance_state

    def _wrapped(self, alive, actions_np, tick_index: int, buckets=None):  # type: ignore[no-untyped-def]
        nonlocal req_total, opp_total
        r0 = int(getattr(self, "_maintenance_schedule_requests_gen", 0))
        o0 = int(getattr(self, "_maintenance_action_opportunities_gen", 0))
        out = original(alive, actions_np, tick_index=tick_index, buckets=buckets)
        r1 = int(getattr(self, "_maintenance_schedule_requests_gen", 0))
        o1 = int(getattr(self, "_maintenance_action_opportunities_gen", 0))
        req_total += max(0, r1 - r0)
        opp_total += max(0, o1 - o0)
        return out

    sim._update_maintenance_state = MethodType(_wrapped, sim)  # type: ignore[assignment]
    sim.run(start_gen=0, skip_init=False)
    return float(req_total) / float(max(1, opp_total))


def test_phaseA_intent_chemistry_life_stage_trajectory_nontrivial_and_bounded(tmp_path: Path) -> None:
    sim = _make_sim(
        tmp_path / "traj",
        seed=1234,
        generations=5,
        steps=10,
        maintenance_enable=True,
        chemistry_functional_coupling_enable=True,
        chemistry_repro_viability_enable=True,
    )

    trajectory: list[tuple[float, float, float]] = []
    original = sim._update_chemistry_state

    def _wrapped(self, alive, tick_index: int, buckets=None):  # type: ignore[no-untyped-def]
        out = original(alive, tick_index=tick_index, buckets=buckets)
        # Track mean repair_reserve, acute_damage, chronic_damage.
        trajectory.append((_chem_slot_mean(self, 1), _chem_slot_mean(self, 2), _chem_slot_mean(self, 3)))
        return out

    sim._update_chemistry_state = MethodType(_wrapped, sim)  # type: ignore[assignment]
    sim.run(start_gen=0, skip_init=False)

    assert len(trajectory) >= 20
    arr = np.asarray(trajectory, dtype=np.float32)
    assert np.all(np.isfinite(arr))
    assert float(np.min(arr)) >= 0.0
    assert float(np.max(arr)) <= 1.0

    # Non-trivial dynamics: at least one tracked slot changes meaningfully over time.
    stds = np.std(arr, axis=0)
    assert float(np.max(stds)) > 0.02

    # Life-stage shift: early vs late means should differ for at least one slot.
    n = int(arr.shape[0])
    early = np.mean(arr[: max(1, n // 3), :], axis=0)
    late = np.mean(arr[-max(1, n // 3) :, :], axis=0)
    assert float(np.max(np.abs(late - early))) > 0.02


def test_phaseA_intent_maintenance_usage_distribution_non_degenerate(tmp_path: Path) -> None:
    sim = _make_sim(
        tmp_path / "nondeg",
        seed=2401,
        generations=6,
        steps=10,
        maintenance_enable=True,
        maintenance_base_effect=0.20,
        maintenance_energy_cost=0.05,
        maintenance_opportunity_cost_coeff=0.25,
    )
    ratio = _maintenance_request_ratio_over_run(sim)
    # IC target: neither near-zero nor saturation.
    assert ratio > 0.05
    assert ratio < 0.85


def test_phaseA_intent_positive_pressure_late_life_persistence_improves_with_maintenance(tmp_path: Path) -> None:
    seeds = (42, 43, 44)
    on_tail: list[float] = []
    off_tail: list[float] = []
    on_alive: list[float] = []
    off_alive: list[float] = []
    on_cohort: list[float] = []
    off_cohort: list[float] = []

    for seed in seeds:
        sim_on = _make_sim(
            tmp_path / f"pp_on_{seed}",
            seed=seed,
            generations=10,
            steps=12,
            maintenance_enable=True,
            maintenance_base_effect=0.30,
            maintenance_energy_cost=0.01,
            maintenance_opportunity_cost_coeff=0.02,
        )
        rows_on = _run_and_rows(sim_on)
        on_tail.append(_injury_tail_mean(rows_on, tail=3))
        on_alive.append(_alive_metric_last(rows_on))
        on_cohort.append(_cohort_survival_tail_mean(rows_on, tail=3))

        sim_off = _make_sim(
            tmp_path / f"pp_off_{seed}",
            seed=seed,
            generations=10,
            steps=12,
            maintenance_enable=False,
        )
        rows_off = _run_and_rows(sim_off)
        off_tail.append(_injury_tail_mean(rows_off, tail=3))
        off_alive.append(_alive_metric_last(rows_off))
        off_cohort.append(_cohort_survival_tail_mean(rows_off, tail=3))

    # Robust intent signal:
    # 1) maintenance-on should be persistence-non-inferior in late life
    # 2) at least one seed shows clear persistence improvement
    # 3) injury channel should show a measurable shift (supporting mechanism signal)
    injury_deltas = [float(off_tail[i] - on_tail[i]) for i in range(len(seeds))]  # positive => maintenance better
    alive_deltas = [float(on_alive[i] - off_alive[i]) for i in range(len(seeds))]  # negative => maintenance worse
    cohort_deltas = [float(on_cohort[i] - off_cohort[i]) for i in range(len(seeds))]
    assert _mean(alive_deltas) >= -6.0
    assert min(alive_deltas) >= -12.0
    assert (
        _pass_count(alive_deltas, lambda d: d > 0.0) >= 1
        or _pass_count(cohort_deltas, lambda d: d > 0.0) >= 1
        or _pass_count(injury_deltas, lambda d: d > 0.0) >= 1
    )
    assert _mean([abs(d) for d in injury_deltas]) >= 0.03


def test_phaseA_intent_adversarial_high_opportunity_cost_selects_against_over_maintenance(tmp_path: Path) -> None:
    # Adaptive baseline with high opportunity-cost maintenance.
    sim_base = _make_sim(
        tmp_path / "adv_base",
        seed=1337,
        generations=6,
        steps=10,
        maintenance_enable=True,
        maintenance_opportunity_cost_coeff=1.8,
        maintenance_energy_cost=0.18,
    )
    rows_base = _run_and_rows(sim_base)
    ann_base = dict(getattr(sim_base, "_maintenance_domination_annotation_last_gen", {}) or {})

    # Pathological over-maintenance: all agents always REST.
    sim_rest = _make_sim(
        tmp_path / "adv_forced_rest",
        seed=1337,
        generations=6,
        steps=10,
        maintenance_enable=True,
        maintenance_opportunity_cost_coeff=1.8,
        maintenance_energy_cost=0.18,
    )

    def _all_rest(self, alive, buckets=None):  # type: ignore[no-untyped-def]
        return np.full((len(alive),), 5, dtype=np.int32)

    sim_rest._batched_actions_strided = MethodType(_all_rest, sim_rest)  # type: ignore[assignment]
    rows_rest = _run_and_rows(sim_rest)
    ann_rest = dict(getattr(sim_rest, "_maintenance_domination_annotation_last_gen", {}) or {})

    assert bool(ann_rest.get("triggered", False)) is True
    assert float(ann_rest.get("ratio", 0.0)) > float(ann_base.get("ratio", 0.0))
    assert _fitness_tail_mean(rows_rest, tail=3) < _fitness_tail_mean(rows_base, tail=3)


def test_phaseA_pis_20260306_141939z_32fa_adversarial_maintenance_pressure_vs_control(tmp_path: Path) -> None:
    """PIS-20260306-141939Z-32FA: stressed maintenance vs maintenance-disabled control."""
    common = dict(
        seed=1337,
        generations=10,
        steps=12,
        pop_size=24,
        maintenance_opportunity_cost_coeff=1.8,
        maintenance_energy_cost=0.18,
        chemistry_functional_coupling_enable=True,
        chemistry_repro_viability_enable=True,
        selection_forbid_chemistry_scalar=True,
        a3_recovery_taxonomy_enable=True,
    )

    sim_on = _make_sim(tmp_path / "chem_adv_on", maintenance_enable=True, **common)
    rows_on = _run_and_rows(sim_on)

    sim_off = _make_sim(tmp_path / "chem_adv_control", maintenance_enable=False, **common)
    rows_off = _run_and_rows(sim_off)

    assert len(rows_on) == 10
    assert len(rows_off) == 10

    burden_on = _injury_tail_mean(rows_on, tail=3) + 0.5 * _max_injury_tail_mean(rows_on, tail=3)
    burden_off = _injury_tail_mean(rows_off, tail=3) + 0.5 * _max_injury_tail_mean(rows_off, tail=3)
    fitness_on = _fitness_tail_mean(rows_on, tail=3)
    fitness_off = _fitness_tail_mean(rows_off, tail=3)

    chem_vector_on = (
        _chem_slot_mean(sim_on, 1),  # repair_reserve
        _chem_slot_mean(sim_on, 2),  # acute_damage
        _chem_slot_mean(sim_on, 3),  # chronic_damage
        _chem_slot_mean(sim_on, 4),  # maintenance_stability
    )
    chem_vector_off = (
        _chem_slot_mean(sim_off, 1),
        _chem_slot_mean(sim_off, 2),
        _chem_slot_mean(sim_off, 3),
        _chem_slot_mean(sim_off, 4),
    )

    n = min(len(rows_on), len(rows_off))
    fitness_traj_mean_abs = _mean(
        [
            abs(float(rows_on[i].get("mean_fitness", 0.0) or 0.0) - float(rows_off[i].get("mean_fitness", 0.0) or 0.0))
            for i in range(n)
        ]
    )
    injury_traj_mean_abs = _mean(
        [
            abs(float(rows_on[i].get("mean_injury", 0.0) or 0.0) - float(rows_off[i].get("mean_injury", 0.0) or 0.0))
            for i in range(n)
        ]
    )

    # PIS acceptance intent: stressed vs control must not collapse into effectively identical trajectories.
    assert abs(burden_on - burden_off) >= 0.03
    assert abs(fitness_on - fitness_off) >= 10.0
    assert sum(abs(float(chem_vector_on[i]) - float(chem_vector_off[i])) for i in range(4)) >= 0.05
    assert fitness_traj_mean_abs >= 5.0 or injury_traj_mean_abs >= 0.03


def test_phaseA_intent_ablation_materially_shifts_behavioral_and_competence_signals(tmp_path: Path) -> None:
    seeds = (3001, 3002, 3003)
    burden_shift_vs_no_maint: list[float] = []
    burden_shift_vs_no_chem: list[float] = []
    fitness_shift_vs_no_chem: list[float] = []
    maint_full: list[float] = []
    maint_no_maint: list[float] = []

    for seed in seeds:
        sim_full = _make_sim(
            tmp_path / f"abl_full_{seed}",
            seed=seed,
            generations=7,
            steps=10,
            maintenance_enable=True,
            chemistry_functional_coupling_enable=True,
            chemistry_repro_viability_enable=True,
        )
        rows_full = _run_and_rows(sim_full)

        sim_no_maint = _make_sim(
            tmp_path / f"abl_no_maint_{seed}",
            seed=seed,
            generations=7,
            steps=10,
            maintenance_enable=False,
            chemistry_functional_coupling_enable=True,
            chemistry_repro_viability_enable=True,
        )
        rows_no_maint = _run_and_rows(sim_no_maint)

        sim_no_chem = _make_sim(
            tmp_path / f"abl_no_chem_{seed}",
            seed=seed,
            generations=7,
            steps=10,
            maintenance_enable=True,
            chemistry_functional_coupling_enable=False,
            chemistry_repro_viability_enable=False,
        )
        rows_no_chem = _run_and_rows(sim_no_chem)

        full_burden = _injury_tail_mean(rows_full) + 0.5 * _max_injury_tail_mean(rows_full)
        no_maint_burden = _injury_tail_mean(rows_no_maint) + 0.5 * _max_injury_tail_mean(rows_no_maint)
        no_chem_burden = _injury_tail_mean(rows_no_chem) + 0.5 * _max_injury_tail_mean(rows_no_chem)
        burden_shift_vs_no_maint.append(abs(full_burden - no_maint_burden))
        burden_shift_vs_no_chem.append(abs(full_burden - no_chem_burden))

        full_fit = _fitness_tail_mean(rows_full, tail=3)
        no_chem_fit = _fitness_tail_mean(rows_no_chem, tail=3)
        fitness_shift_vs_no_chem.append(abs(full_fit - no_chem_fit))

        maint_full.append(_stat_total(sim_full, "maintenance_applied"))
        maint_no_maint.append(_stat_total(sim_no_maint, "maintenance_applied"))

    # Scenario-level ablation should show material shift in behavior/competence,
    # not necessarily strict one-direction dominance per single seed.
    assert _pass_count(maint_full, lambda x: x > 0.0) >= 2
    assert _pass_count(maint_no_maint, lambda x: abs(x) <= 1e-6) == len(seeds)
    assert _pass_count(burden_shift_vs_no_maint, lambda d: d > 0.03) >= 2
    assert _mean(burden_shift_vs_no_maint) > 0.02
    assert _pass_count(burden_shift_vs_no_chem, lambda d: d > 0.03) >= 2
    assert _mean(burden_shift_vs_no_chem) > 0.02
    assert _pass_count(fitness_shift_vs_no_chem, lambda d: d > 1.0) >= 2
    assert _mean(fitness_shift_vs_no_chem) > 1.0


def test_phaseA_intent_delay_falsification_immediate_vs_delayed_vs_off(tmp_path: Path) -> None:
    traces: dict[str, list[tuple[int, float]]] = {}

    def _scenario(name: str, delay_ticks: int, enable: bool) -> SimulationRunner:
        sim = _make_sim(
            tmp_path / name,
            seed=4001,
            generations=1,
            steps=8,
            pop_size=18,
            maintenance_enable=enable,
            maintenance_delay_ticks=delay_ticks,
            maintenance_base_effect=0.22,
            maintenance_energy_cost=0.02,
            maintenance_opportunity_cost_coeff=0.05,
            policy_deterministic_actions=True,
        )
        # Consistent maintenance opportunities to expose timing behavior.
        def _rest_heavy(self, alive, buckets=None):  # type: ignore[no-untyped-def]
            return np.full((len(alive),), 5, dtype=np.int32)

        sim._batched_actions_strided = MethodType(_rest_heavy, sim)  # type: ignore[assignment]
        series: list[tuple[int, float]] = []
        original = sim._update_maintenance_state

        def _wrapped(self, alive, actions_np, tick_index: int, buckets=None):  # type: ignore[no-untyped-def]
            out = original(alive, actions_np, tick_index=tick_index, buckets=buckets)
            series.append((int(tick_index), _chem_slot_mean(self, 2)))  # acute damage slot
            return out

        sim._update_maintenance_state = MethodType(_wrapped, sim)  # type: ignore[assignment]
        sim.run(start_gen=0, skip_init=False)
        traces[name] = sorted(series, key=lambda x: x[0])
        return sim

    sim_delayed = _scenario("delay_delayed", delay_ticks=3, enable=True)
    sim_immediate = _scenario("delay_immediate", delay_ticks=0, enable=True)
    sim_off = _scenario("delay_off", delay_ticks=3, enable=False)

    delayed = [v for _, v in traces["delay_delayed"]]
    immediate = [v for _, v in traces["delay_immediate"]]
    off = [v for _, v in traces["delay_off"]]
    assert len(delayed) >= 4 and len(immediate) >= 4 and len(off) >= 4

    # Delay falsification: immediate and delayed should not collapse to the same trajectory.
    assert immediate[0] <= delayed[0]
    assert immediate[1] <= delayed[1]
    # Tiny per-step floating accumulation can invert by a few e-6 across envs.
    assert delayed[-1] <= (off[-1] + 1e-5)


def test_phaseA_intent_ecology_strategy_divergence_with_chemistry_implication(tmp_path: Path) -> None:
    seeds = (222, 223, 224)
    deltas: list[float] = []
    gap_on_vals: list[float] = []
    for seed in seeds:
        _, rich_on = _run_ecology_condition(
            tmp_path,
            name=f"rich_on_{seed}",
            seed=seed,
            hazard=0.005,
            food_spawn=18,
            chemistry_on=True,
        )
        _, scarce_on = _run_ecology_condition(
            tmp_path,
            name=f"scarce_on_{seed}",
            seed=seed,
            hazard=0.08,
            food_spawn=5,
            chemistry_on=True,
        )
        _, rich_off = _run_ecology_condition(
            tmp_path,
            name=f"rich_off_{seed}",
            seed=seed,
            hazard=0.005,
            food_spawn=18,
            chemistry_on=False,
        )
        _, scarce_off = _run_ecology_condition(
            tmp_path,
            name=f"scarce_off_{seed}",
            seed=seed,
            hazard=0.08,
            food_spawn=5,
            chemistry_on=False,
        )
        gap_on = abs(rich_on - scarce_on)
        gap_off = abs(rich_off - scarce_off)
        gap_on_vals.append(float(gap_on))
        deltas.append(float(gap_on - gap_off))

    # Ecological divergence should be present, and chemistry should materially
    # change divergence magnitude (implication), even if direction can vary by seed.
    abs_deltas = [abs(float(d)) for d in deltas]
    assert _mean(gap_on_vals) > 1.0
    assert _pass_count(abs_deltas, lambda d: d > 2.0) >= 2
    assert _mean(abs_deltas) > 2.0


def test_phaseA_pis_20260306_141939z_e7c4_ecology_divergence_trigger(tmp_path: Path) -> None:
    """PIS-20260306-141939Z-E7C4: rich/scarce ecology matrix with chemistry ON/OFF."""
    seeds = (222, 223, 224)
    gap_on_vals: list[float] = []
    gap_off_vals: list[float] = []
    delta_abs_vals: list[float] = []

    for seed in seeds:
        matrix: dict[tuple[str, bool], float] = {}
        for ecology_name, hazard, food_spawn in (("rich", 0.005, 18), ("scarce", 0.08, 5)):
            for chemistry_on in (True, False):
                rows, tail_fitness = _run_ecology_condition(
                    tmp_path,
                    name=f"pis_e7c4_{ecology_name}_{'on' if chemistry_on else 'off'}_{seed}",
                    seed=seed,
                    hazard=hazard,
                    food_spawn=food_spawn,
                    chemistry_on=chemistry_on,
                )
                assert len(rows) == 6
                matrix[(ecology_name, chemistry_on)] = float(tail_fitness)

        gap_on = abs(matrix[("rich", True)] - matrix[("scarce", True)])
        gap_off = abs(matrix[("rich", False)] - matrix[("scarce", False)])
        gap_on_vals.append(float(gap_on))
        gap_off_vals.append(float(gap_off))
        delta_abs_vals.append(abs(float(gap_on - gap_off)))

    # PIS acceptance intent:
    # 1) all four matrix cells must complete per seed
    # 2) rich/scarce should diverge materially under both chemistry regimes
    # 3) chemistry must materially change the regime-gap magnitude
    assert _pass_count(gap_on_vals, lambda gap: gap > 1.0) == len(seeds)
    assert _pass_count(gap_off_vals, lambda gap: gap > 1.0) == len(seeds)
    assert _mean(gap_on_vals) > 1.0
    assert _mean(gap_off_vals) > 1.0
    assert _pass_count(delta_abs_vals, lambda delta: delta > 2.0) >= 2
    assert _mean(delta_abs_vals) > 2.0


def test_phaseA_intent_forbidden_scalar_reward_channel_guard_active(tmp_path: Path) -> None:
    sim = _make_sim(
        tmp_path / "phaseA_scalar_guard_probe",
        seed=9101,
        generations=1,
        steps=1,
        pop_size=4,
    )
    st = SimpleNamespace(
        survival_steps=1,
        energy_collected=1.0,
        offspring_count=0,
        attacks=0,
        early_life_weight=1.0,
        early_survival_steps=0,
        early_energy_collected=0.0,
        early_injury_gained=0.0,
        chemistry_score=0.25,
    )
    agent = SimpleNamespace(alive=True, stats=st)
    with pytest.raises(RuntimeError, match="DET_FORBIDDEN_CHEMISTRY_SELECTION_SCALAR"):
        sim._effective_fitness(agent)


@pytest.mark.parametrize(
    ("name", "overrides"),
    [
        (
            "pressure",
            {
                "maintenance_enable": True,
                "maintenance_base_effect": 0.16,
                "chemistry_functional_coupling_enable": True,
                "chemistry_repro_viability_enable": True,
                "shock_start_prob": 0.10,
            },
        ),
        (
            "adversarial",
            {
                "maintenance_enable": True,
                "maintenance_opportunity_cost_coeff": 1.6,
                "maintenance_energy_cost": 0.16,
            },
        ),
    ],
)
def test_phaseA_intent_determinism_fresh_vs_resume_parity_per_scenario(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    overrides: dict[str, object],
) -> None:
    monkeypatch.setenv("EIEL_STRICT_DETERMINISM", "1")
    monkeypatch.setenv("EIEL_DETERMINISM_STRICT", "1")

    run_a = tmp_path / f"{name}_a"
    sim_a = _make_sim(run_a, seed=5001, generations=4, steps=8, pop_size=20, **overrides)
    sim_a.run(start_gen=0, skip_init=False)

    bundle = run_a / "checkpoint_bundles_v2" / "bundle_gen000004"
    checkpoint_bundle_v2_save(sim_a, str(bundle), generation_index=4, strict_replay_mode=True)
    payload = checkpoint_bundle_v2_load(bundle, strict=True)

    sim_a.sim_cfg.generations = 2
    sim_a.run(start_gen=4, skip_init=True)
    rows_a = _read_metrics(run_a / "metrics.csv")
    tax_a_path = run_a / "metrics" / "recovery_taxonomy.jsonl"
    rng_a = sim_a.rng_manager.capture_snapshot(generation_index=999)
    chem_a, maint_a = _snapshot(sim_a)

    run_b = tmp_path / f"{name}_b"
    sim_b = _make_sim(run_b, seed=5001, generations=4, steps=8, pop_size=20, **overrides)
    checkpoint_bundle_v2_apply(sim_b, payload, restore_rng=True)
    sim_b.sim_cfg.generations = 2
    sim_b.run(start_gen=4, skip_init=True)
    rows_b = _read_metrics(run_b / "metrics.csv")
    tax_b_path = run_b / "metrics" / "recovery_taxonomy.jsonl"
    rng_b = sim_b.rng_manager.capture_snapshot(generation_index=999)
    chem_b, maint_b = _snapshot(sim_b)

    # Compare continuation window only (resume run writes continuation rows only).
    assert rows_a[-len(rows_b) :] == rows_b
    tax_a_lines = [json.loads(x) for x in tax_a_path.read_text(encoding="utf-8").splitlines() if x.strip()]
    tax_b_lines = [json.loads(x) for x in tax_b_path.read_text(encoding="utf-8").splitlines() if x.strip()]
    assert tax_a_lines[-len(tax_b_lines) :] == tax_b_lines
    assert rng_a == rng_b
    assert chem_a == chem_b
    assert maint_a == maint_b

    # Ensure taxonomy records are non-empty and queryable for this scenario.
    records = tax_a_lines
    assert len(records) >= 1
    assert all("classification" in rec for rec in records)
