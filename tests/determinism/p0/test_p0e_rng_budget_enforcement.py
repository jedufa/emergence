from __future__ import annotations

from types import MethodType, SimpleNamespace

import numpy as np
import pytest
import torch

from agents.agent_policy import MLPPolicy
from evolution.population_pkg import spawn as population_spawn
from evolution.reproduction import crossover_policy
from simulation.rng_manager import DeterminismViolationError, RNGManager
from simulation.runner_pkg.runner_ticks_loop import TickLoopMixin
from simulation.runner_pkg.runner_shock import ShockMixin


class _DummyShock(ShockMixin):
    def __init__(self, rm: RNGManager):
        self.rng_manager = rm
        self.sim_cfg = SimpleNamespace(
            shock_pulsed=False,
            shock_start_prob=0.0,
            shock_period=0,
            shock_duration=0,
        )


class _DummyTickLoop(TickLoopMixin):
    def __init__(self, rm: RNGManager, *, spatial_step_fn) -> None:
        self.rng_manager = rm
        self.sim_cfg = SimpleNamespace(steps_per_generation=80)
        self.env = SimpleNamespace(spatial_droughts_step=spatial_step_fn)


def _spawn_host_and_agent(rm: RNGManager, *, birth_noise: float) -> tuple[SimpleNamespace, SimpleNamespace]:
    host = SimpleNamespace(np_rng=rm.mutation_np, rng_manager=rm)
    stats = SimpleNamespace(
        exp_dE=np.zeros((7,), dtype=np.float32),
        exp_dInj=np.zeros((7,), dtype=np.float32),
        pred_queue=[],
        prev_drive_delta=np.zeros((4,), dtype=np.float32),
    )
    genome = SimpleNamespace(
        drive_sp_hunger=0.55,
        drive_sp_pain=0.05,
        drive_sp_fatigue=0.15,
        drive_sp_fear=0.10,
        drive_birth_noise=float(birth_noise),
        exp_prior_strength=0.35,
        exp_prior_noise=0.05,
    )
    agent = SimpleNamespace(
        energy=5.0,
        max_energy_norm=10.0,
        genome=genome,
        stats=stats,
    )
    return host, agent


def test_p0e_boundary_expected_per_domain() -> None:
    rm = RNGManager(global_seed=111)

    # Python domain: fixed shock budget (policy.py += 8)
    shock = _DummyShock(rm)
    expected_shock = rm.expected_delta_for_boundary("core_tick.generation.shock_active")
    with rm.budget_scope("core_tick.generation.shock_active", expected_delta=expected_shock):
        _ = shock._shock_active(gen=0)

    # NumPy domain: spawn birth-noise boundary (mutation.np += 4 when noise enabled)
    host, agent = _spawn_host_and_agent(rm, birth_noise=0.03)
    population_spawn._init_drives_for_agent(host, agent, None, None)

    # Torch domain: mutation crossover policy budget (mutation.torch_cpu += sum(numel(params)))
    p1 = MLPPolicy(obs_size=10, hidden_size=16, n_actions=7)
    p2 = MLPPolicy(obs_size=10, hidden_size=16, n_actions=7)
    _ = crossover_policy(p1, p2, obs_size=10, hidden_size=16, n_actions=7, rng=rm.mutation_py, rng_manager=rm)


def test_p0e_strict_failure_on_spawn_budget_drift(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EIEL_STRICT_DETERMINISM", "1")
    rm = RNGManager(global_seed=222)
    host, agent = _spawn_host_and_agent(rm, birth_noise=0.03)

    original = rm.expected_delta_for_boundary

    def _bad(self, label: str, *, ctx=None):
        out = original(label, ctx=ctx)
        if label == "spawn.population.init_drives.birth_noise":
            out["mutation.np"] = int(out.get("mutation.np", 0)) + 1
        return out

    rm.expected_delta_for_boundary = MethodType(_bad, rm)  # type: ignore[assignment]

    with pytest.raises(RuntimeError, match="DET_RNG_BUDGET_MISMATCH"):
        population_spawn._init_drives_for_agent(host, agent, None, None)


def test_p0e_multi_step_drift_localizes_first_violating_boundary(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EIEL_STRICT_DETERMINISM", "1")
    rm = RNGManager(global_seed=333)
    shock = _DummyShock(rm)

    expected_ok = rm.expected_delta_for_boundary("core_tick.generation.shock_active")
    with rm.budget_scope("core_tick.generation.shock_active", expected_delta=expected_ok):
        _ = shock._shock_active(gen=0)

    with pytest.raises(RuntimeError, match="DET_RNG_BUDGET_MISMATCH"):
        with rm.budget_scope("boundary.second", expected_delta={"policy.py": 7}):
            _ = shock._shock_active(gen=1)

    try:
        with rm.budget_scope("boundary.second", expected_delta={"policy.py": 7}):
            _ = shock._shock_active(gen=2)
    except RuntimeError as e:
        s = str(e)
        assert "label=boundary.second" in s
        return
    raise AssertionError("Expected DET_RNG_BUDGET_MISMATCH with boundary.second label")


def test_p0e_replay_parity_for_budget_deltas() -> None:
    rm = RNGManager(global_seed=444)
    snap0 = rm.capture_snapshot(generation_index=0)
    shock = _DummyShock(rm)
    p1 = MLPPolicy(obs_size=10, hidden_size=16, n_actions=7)
    p2 = MLPPolicy(obs_size=10, hidden_size=16, n_actions=7)

    def _run_sequence() -> tuple[float, float, tuple[int, int, int]]:
        exp_shock = rm.expected_delta_for_boundary("core_tick.generation.shock_active")
        with rm.budget_scope("core_tick.generation.shock_active", expected_delta=exp_shock):
            active = bool(shock._shock_active(gen=0))

        exp_spawn = rm.expected_delta_for_boundary(
            "spawn.population.init_drives.birth_noise",
            ctx={"birth_noise_draws": 4},
        )
        with rm.budget_scope("spawn.population.init_drives.birth_noise", expected_delta=exp_spawn):
            v = rm.mutation_np.normal(0.0, 1.0, size=(4,)).astype(np.float32)

        child = crossover_policy(
            p1,
            p2,
            obs_size=10,
            hidden_size=16,
            n_actions=7,
            rng=rm.mutation_py,
            rng_manager=rm,
        )
        idx = rm.indices_snapshot()
        return float(v.sum()), float(next(child.parameters()).detach().sum()), (
            int(idx.get("policy.py", 0)),
            int(idx.get("mutation.np", 0)),
            int(idx.get("mutation.torch_cpu", 0)),
        )

    out1 = _run_sequence()
    rm.apply_snapshot_dict(snap0)
    out2 = _run_sequence()
    assert out1 == out2


def test_p0e_strict_failure_on_torch_budget_drift(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EIEL_STRICT_DETERMINISM", "1")
    rm = RNGManager(global_seed=555)
    p1 = MLPPolicy(obs_size=10, hidden_size=16, n_actions=7)
    p2 = MLPPolicy(obs_size=10, hidden_size=16, n_actions=7)

    original = rm.expected_delta_for_boundary

    def _bad(self, label: str, *, ctx=None):
        out = original(label, ctx=ctx)
        if label == "mutation.reproduction.crossover_policy":
            out["mutation.torch_cpu"] = int(out.get("mutation.torch_cpu", 0)) - 1
        return out

    rm.expected_delta_for_boundary = MethodType(_bad, rm)  # type: ignore[assignment]
    with pytest.raises(RuntimeError, match="DET_RNG_BUDGET_MISMATCH"):
        _ = crossover_policy(p1, p2, obs_size=10, hidden_size=16, n_actions=7, rng=rm.mutation_py, rng_manager=rm)


def test_p0e_env_boundary_propagates_non_mismatch_determinism_violation() -> None:
    rm = RNGManager(global_seed=666)

    def _boom(_abs_step: int) -> None:
        raise DeterminismViolationError("DET_RNG_BUDGET_UNKNOWN_BOUNDARY label=env.update.spatial_droughts_step")

    loop = _DummyTickLoop(rm, spatial_step_fn=_boom)
    with pytest.raises(DeterminismViolationError, match="DET_RNG_BUDGET_UNKNOWN_BOUNDARY"):
        loop._run_spatial_droughts_step_with_budget(gen=0, tick=0)
