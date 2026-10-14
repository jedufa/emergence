from __future__ import annotations

import json
import inspect
from pathlib import Path
from types import MethodType, SimpleNamespace

import numpy as np
import pytest
import torch

from agents.chemistry import chemistry_from_values
from agents.chemistry_coupling import (
    action_cost_multiplier_from_chem,
    classify_recovery_taxonomy,
    observation_fidelity_and_bias_from_chem,
    reproduction_viability_scale_from_chem,
    reproduction_viability_scale_from_chem_row,
)
from env.env_config import EnvConfig
from evolution.population_pkg.population_config import PopConfig
from simulation.checkpoint_bundle_v2 import checkpoint_bundle_v2_apply, checkpoint_bundle_v2_load, checkpoint_bundle_v2_save
from simulation.determinism_probe import _hash_pop_stats_scalar
from simulation.metrics_pkg.noop_loggers import NoOpMetricsLogger
from simulation.runner import SimulationRunner
from simulation.runner_pkg.runner_config import SimConfig
from simulation.runner_pkg.runner_selection import SelectionMixin


class _DummySelection(SelectionMixin):
    def __init__(self) -> None:
        self.sim_cfg = SimpleNamespace(
            selection_forbid_chemistry_scalar=True,
            episode0_pred_err_fitness_weight=0.0,
            episode0_thrashing_fitness_weight=0.0,
            episode0_drive_jitter_fitness_weight=0.0,
            episode0_selfharm_fitness_weight=0.0,
            episode0_fitness_weight=0.0,
            coherence_thrashing_weight=0.0,
            coherence_drive_jitter_weight=0.0,
            coherence_selfharm_weight=0.0,
            coherence_pred_err_weight=0.0,
            coherence_reset_shock_weight=0.0,
            coherence_episode0_pred_err_weight=0.0,
            late_pred_cost_enable=False,
            late_pred_cost_weight=0.0,
        )


def _make_sim(
    tmp_path: Path,
    *,
    seed: int = 123,
    generations: int = 2,
    steps: int = 4,
    pop_size: int = 8,
    coupling_enable: bool = True,
    action_cost_coupling_enable: bool = False,
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
        chemistry_functional_coupling_enable=bool(coupling_enable),
        chemistry_action_cost_coupling_enable=bool(action_cost_coupling_enable),
        chemistry_repro_viability_enable=bool(coupling_enable),
        a3_recovery_taxonomy_enable=True,
    )
    # Keep coupling tests isolated from unrelated Phase-E metrics defaults.
    sim_cfg.phase_e_e1_enable = False
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


def _maint_stats_snapshot(sim: SimulationRunner) -> dict[int, tuple[float, ...]]:
    out: dict[int, tuple[float, ...]] = {}
    for aid in sorted(sim.pop.agents.keys()):
        a = sim.pop.agents[aid]
        st = a.stats
        out[int(aid)] = (
            float(getattr(st, "chemistry_action_cost_mult", 1.0)),
            float(getattr(st, "chemistry_observation_fidelity", 1.0)),
            float(getattr(st, "chemistry_repro_viability", 1.0)),
        )
    return out


def test_p1a3_unit_action_cost_coupling_bounded_and_monotonic() -> None:
    low = chemistry_from_values((0.5, 0.8, 0.05, 0.05, 0.8, 0.5))
    high = chemistry_from_values((0.5, 0.0, 0.95, 0.95, 0.0, 0.5))
    m_low = action_cost_multiplier_from_chem(low, min_mult=1.0, max_mult=1.4, strength=1.0, power=1.0)
    m_high = action_cost_multiplier_from_chem(high, min_mult=1.0, max_mult=1.4, strength=1.0, power=1.0)
    assert 1.0 <= m_low <= 1.4
    assert 1.0 <= m_high <= 1.4
    assert m_high >= m_low


def test_p1a3_unit_observation_coupling_bounded_and_deterministic() -> None:
    c = chemistry_from_values((0.5, 0.2, 0.7, 0.6, 0.3, 0.9))
    f1, b1 = observation_fidelity_and_bias_from_chem(
        c,
        fidelity_min=0.7,
        fidelity_max=1.0,
        noise_abs_max=0.15,
        fidelity_strength=0.9,
        noise_strength=0.8,
    )
    f2, b2 = observation_fidelity_and_bias_from_chem(
        c,
        fidelity_min=0.7,
        fidelity_max=1.0,
        noise_abs_max=0.15,
        fidelity_strength=0.9,
        noise_strength=0.8,
    )
    assert 0.7 <= f1 <= 1.0
    assert abs(b1) <= 0.15
    assert f1 == f2 and b1 == b2


def test_p1a3_unit_repro_viability_bounded_and_monotonic() -> None:
    c_ok = chemistry_from_values((0.4, 0.9, 0.1, 0.1, 0.7, 0.5))
    c_bad = chemistry_from_values((0.4, 0.1, 0.9, 0.9, 0.1, 0.5))
    v_ok = reproduction_viability_scale_from_chem(c_ok, floor=0.2, strength=0.8)
    v_bad = reproduction_viability_scale_from_chem(c_bad, floor=0.2, strength=0.8)
    assert reproduction_viability_scale_from_chem_row(c_ok, floor=0.2, strength=0.8) == v_ok
    assert reproduction_viability_scale_from_chem_row(c_bad, floor=0.2, strength=0.8) == v_bad
    assert 0.2 <= v_ok <= 1.0
    assert 0.2 <= v_bad <= 1.0
    assert v_bad <= v_ok


def test_p1a3_unit_selection_scalar_ban_guard() -> None:
    sel = _DummySelection()
    st = SimpleNamespace(
        survival_steps=1,
        energy_collected=1.0,
        offspring_count=0,
        attacks=0,
        early_life_weight=1.0,
        early_survival_steps=0,
        early_energy_collected=0.0,
        early_injury_gained=0.0,
        chemistry_score=0.7,
    )
    agent = SimpleNamespace(alive=True, stats=st)
    with pytest.raises(RuntimeError, match="DET_FORBIDDEN_CHEMISTRY_SELECTION_SCALAR"):
        sel._effective_fitness(agent)


def test_p1a3_unit_selection_scalar_ban_guard_alias_field() -> None:
    sel = _DummySelection()
    st = SimpleNamespace(
        survival_steps=1,
        energy_collected=1.0,
        offspring_count=0,
        attacks=0,
        early_life_weight=1.0,
        early_survival_steps=0,
        early_energy_collected=0.0,
        early_injury_gained=0.0,
        chem_bonus_score=0.8,
    )
    agent = SimpleNamespace(alive=True, stats=st)
    with pytest.raises(RuntimeError, match="DET_FORBIDDEN_CHEMISTRY_SELECTION_SCALAR"):
        sel._effective_fitness(agent)


def test_p1a3_unit_selection_effective_fitness_has_no_chemistry_stat_reads() -> None:
    src = inspect.getsource(SelectionMixin._effective_fitness)
    assert "getattr(st, \"chemistry" not in src
    assert "st.chemistry" not in src


def test_p1a3_unit_taxonomy_classifier_return_shift_collapse() -> None:
    assert classify_recovery_taxonomy(
        fitness_ratio=0.95,
        drift_ratio=1.00,
        return_fitness_ratio=0.92,
        return_drift_ratio_max=1.10,
        shift_fitness_ratio_min=0.70,
    ) == "return"
    assert classify_recovery_taxonomy(
        fitness_ratio=0.80,
        drift_ratio=1.40,
        return_fitness_ratio=0.92,
        return_drift_ratio_max=1.10,
        shift_fitness_ratio_min=0.70,
    ) == "shift"
    assert classify_recovery_taxonomy(
        fitness_ratio=0.50,
        drift_ratio=2.00,
        return_fitness_ratio=0.92,
        return_drift_ratio_max=1.10,
        shift_fitness_ratio_min=0.70,
    ) == "collapse"


def test_p1a3_integration_action_cost_coupling_affects_env_boundary(tmp_path: Path) -> None:
    sim = _make_sim(
        tmp_path / "action_cost",
        generations=1,
        steps=1,
        pop_size=6,
        coupling_enable=True,
        action_cost_coupling_enable=True,
    )
    sim.pop.init_random()
    sim._ensure_bank()
    sim._place_agents()
    alive = sim.pop.alive_agents()[:3]
    alive[0].chemistry = chemistry_from_values((0.2, 0.0, 1.0, 1.0, 0.0, 0.5))
    alive[1].chemistry = chemistry_from_values((0.8, 0.8, 0.0, 0.0, 0.8, 0.5))
    alive[2].chemistry = chemistry_from_values((0.8, 0.8, 0.0, 0.0, 0.8, 0.5))

    captured = {}

    def _fake_step(self, **kwargs):
        captured["move_cost_mults"] = np.asarray(kwargs.get("move_cost_mults"), dtype=np.float32).copy()
        n = int(np.asarray(kwargs["agent_ids"]).shape[0])
        return (
            np.zeros((n,), dtype=np.bool_),
            np.asarray(kwargs["energies"], dtype=np.float32).copy(),
            np.asarray(kwargs["healths"], dtype=np.float32).copy(),
            np.zeros((n,), dtype=np.float32),
            np.asarray(kwargs["injuries"], dtype=np.float32).copy(),
            np.asarray(kwargs["scars"], dtype=np.float32).copy(),
        )

    sim.env.step_agents_batch = MethodType(_fake_step, sim.env)  # type: ignore[assignment]
    ids = np.asarray([int(a.agent_id) for a in alive], dtype=np.int32)
    acts = np.zeros((len(alive),), dtype=np.int32)
    sim._apply_env_tick(alive, ids, acts, {}, 0, 0.0)

    m = captured["move_cost_mults"]
    assert float(m[0]) > float(m[1])
    assert float(m[0]) >= 1.0


def test_p1a3_integration_action_cost_coupling_default_off_at_env_boundary(tmp_path: Path) -> None:
    sim = _make_sim(tmp_path / "action_cost_default_off", generations=1, steps=1, pop_size=6, coupling_enable=True)
    sim.pop.init_random()
    sim._ensure_bank()
    sim._place_agents()
    alive = sim.pop.alive_agents()[:3]
    alive[0].chemistry = chemistry_from_values((0.2, 0.0, 1.0, 1.0, 0.0, 0.5))

    captured = {}

    def _fake_step(self, **kwargs):
        captured["move_cost_mults"] = kwargs.get("move_cost_mults")
        n = int(np.asarray(kwargs["agent_ids"]).shape[0])
        return (
            np.zeros((n,), dtype=np.bool_),
            np.asarray(kwargs["energies"], dtype=np.float32).copy(),
            np.asarray(kwargs["healths"], dtype=np.float32).copy(),
            np.zeros((n,), dtype=np.float32),
            np.asarray(kwargs["injuries"], dtype=np.float32).copy(),
            np.asarray(kwargs["scars"], dtype=np.float32).copy(),
        )

    sim.env.step_agents_batch = MethodType(_fake_step, sim.env)  # type: ignore[assignment]
    ids = np.asarray([int(a.agent_id) for a in alive], dtype=np.int32)
    acts = np.zeros((len(alive),), dtype=np.int32)
    sim._apply_env_tick(alive, ids, acts, {}, 0, 0.0)

    assert captured["move_cost_mults"] is None


def test_p1a3_integration_observation_fidelity_changes_policy_input_deterministically(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sim = _make_sim(tmp_path / "obs", generations=1, steps=1, pop_size=6, coupling_enable=True)
    sim.pop.init_random()
    for a in sim.pop.agents.values():
        a.genome.use_gru = False
    sim._ensure_bank()
    sim._place_agents()
    alive = sim.pop.alive_agents()[:4]
    alive[0].chemistry = chemistry_from_values((0.2, 0.0, 1.0, 1.0, 0.0, 1.0))

    captured: dict[str, np.ndarray] = {}

    def _fake_forward(x: torch.Tensor, *_args, **_kwargs):
        captured["x"] = x.detach().cpu().numpy().copy()
        n = int(x.shape[0])
        return torch.zeros((n, 7), device=x.device, dtype=x.dtype)

    monkeypatch.setattr("simulation.runner_pkg.runner_policy.forward_batched_mlp", _fake_forward)
    sim._batched_actions(alive)
    x_on = captured["x"].copy()

    sim.sim_cfg.chemistry_functional_coupling_enable = False
    sim._batched_actions(alive)
    x_off = captured["x"].copy()

    assert not np.allclose(x_on, x_off)


def test_p1a3_integration_repro_viability_penalty_blocks_high_chronic(tmp_path: Path) -> None:
    sim = _make_sim(tmp_path / "repro", generations=1, steps=1, pop_size=6, coupling_enable=True)
    sim.pop.init_random()
    sim._ensure_bank()
    sim._place_agents()

    sim.env_cfg.reproduce_requires_mate = False
    sim.env_cfg.reproduce_try_prob = 1.0
    sim.env_cfg.injury_birth_fail_scar = 0.0
    sim.env_cfg.injury_birth_fail_injury = 0.0

    alive = sim.pop.alive_agents()[:1]
    a = alive[0]
    a.energy = max(10.0, float(a.genome.reproduction_threshold) + 10.0)
    a.repro_cooldown = 0
    a.chemistry = chemistry_from_values((0.5, 0.0, 0.9, 1.0, 0.0, 0.5))
    sim.sim_cfg.chemistry_repro_chronic_block_threshold = 0.9

    b = {}
    succ, fail = sim._handle_births(alive, 0, 0, b)
    assert succ == 0
    assert int(getattr(sim, "_birth_fail_injury_gate_gen", 0)) >= 1


def test_p1a3_integration_checkpoint_roundtrip_restores_a3_metadata(tmp_path: Path) -> None:
    run_a = tmp_path / "run_a_ckpt"
    sim_a = _make_sim(run_a, generations=1, steps=2, pop_size=8, coupling_enable=True)
    sim_a.run(start_gen=0, skip_init=False)
    sim_a._a3_recovery_baseline_fitness_ema = 1.23
    sim_a._a3_recovery_baseline_drift_ema = 0.45
    sim_a._a3_recovery_taxonomy_last_gen = {"classification": "shift", "generation": 1}

    before_stats = _maint_stats_snapshot(sim_a)

    bundle_dir = checkpoint_bundle_v2_save(
        sim_a,
        str(run_a / "checkpoint_bundles_v2" / "bundle_gen000001"),
        generation_index=1,
        strict_replay_mode=True,
    )
    payload = checkpoint_bundle_v2_load(bundle_dir, strict=True)

    run_b = tmp_path / "run_b_ckpt"
    sim_b = _make_sim(run_b, generations=1, steps=2, pop_size=8, coupling_enable=True)
    checkpoint_bundle_v2_apply(sim_b, payload, restore_rng=True)

    after_stats = _maint_stats_snapshot(sim_b)
    assert before_stats == after_stats
    assert float(getattr(sim_b, "_a3_recovery_baseline_fitness_ema", 0.0)) == pytest.approx(1.23)
    assert float(getattr(sim_b, "_a3_recovery_baseline_drift_ema", 0.0)) == pytest.approx(0.45)


def test_p1a3_integration_probe_hash_includes_a3_stats_signal() -> None:
    a0 = SimpleNamespace(agent_id=0, alive=True, stats=SimpleNamespace(chemistry_action_cost_mult=1.0))
    a1 = SimpleNamespace(agent_id=1, alive=True, stats=SimpleNamespace(chemistry_action_cost_mult=1.0))
    sim = SimpleNamespace(pop=SimpleNamespace(agents={0: a0, 1: a1}))
    h0 = _hash_pop_stats_scalar(sim, "chemistry_action_cost_mult")
    a1.stats.chemistry_action_cost_mult = 1.3
    h1 = _hash_pop_stats_scalar(sim, "chemistry_action_cost_mult")
    assert h0 is not None and h1 is not None
    assert h0 != h1


def test_p1a3_integration_recovery_taxonomy_artifact_queryable(tmp_path: Path) -> None:
    sim = _make_sim(tmp_path / "taxonomy", generations=2, steps=3, pop_size=8, coupling_enable=True)
    sim.run(start_gen=0, skip_init=False)
    p = Path(sim.run_dir) / "metrics" / "recovery_taxonomy.jsonl"
    assert p.exists()
    lines = [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
    assert len(lines) >= 1
    assert all("classification" in rec for rec in lines)


def test_p1a3_functional_fresh_vs_resume_parity_strict_with_a3(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EIEL_STRICT_DETERMINISM", "1")
    run_a = tmp_path / "run_a"
    sim_a = _make_sim(run_a, seed=9201, generations=3, steps=6, pop_size=10, coupling_enable=True)
    sim_a.run(start_gen=0, skip_init=False)

    bundle = run_a / "checkpoint_bundles_v2" / "bundle_gen000003"
    checkpoint_bundle_v2_save(sim_a, str(bundle), generation_index=3, strict_replay_mode=True)
    payload = checkpoint_bundle_v2_load(bundle, strict=True)

    sim_a.sim_cfg.generations = 2
    sim_a.run(start_gen=3, skip_init=True)
    out_a = _maint_stats_snapshot(sim_a)
    rng_a = sim_a.rng_manager.capture_snapshot(generation_index=999)

    run_b = tmp_path / "run_b"
    sim_b = _make_sim(run_b, seed=9201, generations=3, steps=6, pop_size=10, coupling_enable=True)
    checkpoint_bundle_v2_apply(sim_b, payload, restore_rng=True)
    sim_b.sim_cfg.generations = 2
    sim_b.run(start_gen=3, skip_init=True)
    out_b = _maint_stats_snapshot(sim_b)
    rng_b = sim_b.rng_manager.capture_snapshot(generation_index=999)

    assert out_a == out_b
    assert rng_a == rng_b


def test_p1a3_functional_chemistry_ablation_weakens_competence_signal() -> None:
    c = chemistry_from_values((0.3, 0.0, 1.0, 1.0, 0.0, 1.0))
    on = action_cost_multiplier_from_chem(c, min_mult=1.0, max_mult=1.35, strength=0.9, power=1.0)
    off = action_cost_multiplier_from_chem(c, min_mult=1.0, max_mult=1.0, strength=0.0, power=1.0)
    assert on > 1.0
    assert off == pytest.approx(1.0)


def test_p1a3_long_horizon_replay_gate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EIEL_STRICT_DETERMINISM", "1")
    run_a = tmp_path / "long_a"
    sim_a = _make_sim(run_a, seed=9202, generations=6, steps=12, pop_size=10, coupling_enable=True)
    sim_a.run(start_gen=0, skip_init=False)

    bundle = run_a / "checkpoint_bundles_v2" / "bundle_gen000006"
    checkpoint_bundle_v2_save(sim_a, str(bundle), generation_index=6, strict_replay_mode=True)
    payload = checkpoint_bundle_v2_load(bundle, strict=True)

    sim_a.sim_cfg.generations = 2
    sim_a.run(start_gen=6, skip_init=True)
    final_a = _maint_stats_snapshot(sim_a)
    rng_a = sim_a.rng_manager.capture_snapshot(generation_index=1000)

    run_b = tmp_path / "long_b"
    sim_b = _make_sim(run_b, seed=9202, generations=6, steps=12, pop_size=10, coupling_enable=True)
    checkpoint_bundle_v2_apply(sim_b, payload, restore_rng=True)
    sim_b.sim_cfg.generations = 2
    sim_b.run(start_gen=6, skip_init=True)
    final_b = _maint_stats_snapshot(sim_b)
    rng_b = sim_b.rng_manager.capture_snapshot(generation_index=1000)

    assert final_a == final_b
    assert rng_a == rng_b
