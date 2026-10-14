"""tests/integration/metrics/test_epi_engine.py

Phase 5a validation tests for EPI (Experimental Pressure Index).
"""

from __future__ import annotations

import copy
import tempfile

import numpy as np
import pytest
import torch

from env.env_config import EnvConfig
from evolution.population_pkg.population_config import PopConfig
from simulation.metrics_pkg.epi_engine import EPIEngine, derive_stress_seed
from simulation.metrics_pkg.noop_loggers import NoOpMetricsLogger
from simulation.metrics_pkg.stress_scenarios import StressScenarioV1
from simulation.runner import SimulationRunner
from simulation.runner_pkg.runner_config import SimConfig


def _capture_sim_fingerprint(sim: SimulationRunner) -> dict:
    """Capture determinism-relevant state from the main sim for mutation checks."""

    # Env arrays + multipliers
    env = sim.env
    pop = sim.pop

    fp = {
        "env": {
            "grid": env.grid.copy(),
            "agent_grid": env.agent_grid.copy(),
            "agent_alive": env.agent_alive.copy(),
            "agent_pos_x": env.agent_pos_x.copy(),
            "agent_pos_y": env.agent_pos_y.copy(),
            "food_spawn_mult": float(env.food_spawn_mult),
            "action_cost_mult": float(env.action_cost_mult),
            "rest_cost_mult": float(env.rest_cost_mult),
        },
        "agents": {},
        "rng": {
            "env_np": copy.deepcopy(env.rng.bit_generator.state),
            "pop_np": copy.deepcopy(pop.np_rng.bit_generator.state),
            "runner_np": copy.deepcopy(sim.np_rng.bit_generator.state),
            "pop_py": copy.deepcopy(pop.rng.getstate()),
            "rng_manager_snapshot": copy.deepcopy(sim.rng_manager.capture_snapshot()),
        },
        "torch": {
            "cpu": torch.get_rng_state().clone(),
            "cuda": [t.clone() for t in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else [],
        },
    }

    # Agent state (small pop in tests).
    for aid in sorted(pop.agents.keys()):
        a = pop.agents[aid]
        fp["agents"][aid] = {
            "alive": bool(a.alive),
            "energy": float(a.energy),
            "health": float(a.health),
            "injury": float(a.injury),
            "offspring_count": int(a.stats.offspring_count),
            "genome": dict(a.genome),
            "policy_sd": {
                k: v.detach().cpu().clone() for k, v in a.policy.state_dict().items()
            },
        }

    return fp


def _assert_fingerprint_equal(a: dict, b: dict) -> None:
    # Env arrays
    for k in ["grid", "agent_grid", "agent_alive", "agent_pos_x", "agent_pos_y"]:
        assert np.array_equal(a["env"][k], b["env"][k])
    for k in ["food_spawn_mult", "action_cost_mult", "rest_cost_mult"]:
        assert a["env"][k] == b["env"][k]

    # RNG
    assert a["rng"]["env_np"] == b["rng"]["env_np"]
    assert a["rng"]["pop_np"] == b["rng"]["pop_np"]
    assert a["rng"]["runner_np"] == b["rng"]["runner_np"]
    assert a["rng"]["pop_py"] == b["rng"]["pop_py"]
    assert a["rng"]["rng_manager_snapshot"] == b["rng"]["rng_manager_snapshot"]

    # Torch
    assert torch.equal(a["torch"]["cpu"], b["torch"]["cpu"])
    assert len(a["torch"]["cuda"]) == len(b["torch"]["cuda"])
    for s0, s1 in zip(a["torch"]["cuda"], b["torch"]["cuda"]):
        assert torch.equal(s0, s1)

    # Agents
    assert set(a["agents"].keys()) == set(b["agents"].keys())
    for aid in a["agents"].keys():
        aa = a["agents"][aid]
        bb = b["agents"][aid]
        assert aa["alive"] == bb["alive"]
        assert aa["energy"] == bb["energy"]
        assert aa["health"] == bb["health"]
        assert aa["injury"] == bb["injury"]
        assert aa["offspring_count"] == bb["offspring_count"]
        assert aa["genome"] == bb["genome"]
        assert set(aa["policy_sd"].keys()) == set(bb["policy_sd"].keys())
        for k in aa["policy_sd"].keys():
            assert torch.equal(aa["policy_sd"][k], bb["policy_sd"][k])


def test_epi_seed_derivation_stable() -> None:
    seed = derive_stress_seed(
        global_seed=123,
        scenario_id="high_scarcity",
        generation_index=42,
        seed_offset=7,
    )
    assert seed == 5064600856925656499


def test_epi_shadow_clone_no_state_mutation() -> None:
    env_cfg = EnvConfig(width=10, height=10)
    pop_cfg = PopConfig(population_size=8)
    sim_cfg = SimConfig(
        seed=123,
        generations=1,
        steps_per_generation=4,
        device="cpu",
        bank_device="cpu",
        behavior_log_enable=False,
    )

    with tempfile.TemporaryDirectory() as td:
        sim = SimulationRunner(
            env_cfg=env_cfg,
            pop_cfg=pop_cfg,
            sim_cfg=sim_cfg,
            logger=NoOpMetricsLogger(),
            run_dir=td,
            perf=False,
        )

        # Ensure env has agents placed.
        sim._place_agents()

        before = _capture_sim_fingerprint(sim)

        epi = EPIEngine()
        scenarios = [
            StressScenarioV1(
                scenario_id="baseline_env",
                pressure_overrides={},
                duration_steps=2,
                seed_offset=0,
            )
        ]
        _ = epi.compute(sim=sim, generation_index=0, scenarios=scenarios)

        after = _capture_sim_fingerprint(sim)
        _assert_fingerprint_equal(before, after)


def test_epi_shadow_binds_expected_rng_domains(monkeypatch: pytest.MonkeyPatch) -> None:
    """Validate shadow bindings: env/runner on stress, population on mutation."""

    env_cfg = EnvConfig(width=10, height=10)
    pop_cfg = PopConfig(population_size=6)
    sim_cfg = SimConfig(
        seed=111,
        generations=1,
        steps_per_generation=3,
        device="cpu",
        bank_device="cpu",
        behavior_log_enable=False,
    )

    with tempfile.TemporaryDirectory() as td:
        sim = SimulationRunner(
            env_cfg=env_cfg,
            pop_cfg=pop_cfg,
            sim_cfg=sim_cfg,
            logger=NoOpMetricsLogger(),
            run_dir=td,
            perf=False,
        )
        sim._place_agents()

        # Patch the tick loop to assert RNG injection, then no-op.
        original = SimulationRunner._run_ticks

        def _assert_rng_and_noop(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            assert self.np_rng is self.rng_manager.stress_np
            assert self.env.rng is self.rng_manager.stress_np
            assert self.pop.rng_manager is self.rng_manager
            assert self.pop.rng is self.rng_manager.mutation_py
            assert self.pop.np_rng is self.rng_manager.mutation_np
            return (0, 0, 0, 0.0, 0)

        monkeypatch.setattr(SimulationRunner, "_run_ticks", _assert_rng_and_noop)
        try:
            epi = EPIEngine()
            scenarios = [
                StressScenarioV1(
                    scenario_id="baseline_env",
                    pressure_overrides={},
                    duration_steps=1,
                    seed_offset=0,
                )
            ]
            _ = epi.compute(sim=sim, generation_index=0, scenarios=scenarios)
        finally:
            monkeypatch.setattr(SimulationRunner, "_run_ticks", original)


def test_epi_compute_strict_deepcopy_compat(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strict mode must allow EPI compute without forbidden deepcopy ops."""
    monkeypatch.setenv("EIEL_STRICT_DETERMINISM", "1")
    monkeypatch.setenv("DEBUG_DETERMINISM", "1")

    env_cfg = EnvConfig(width=10, height=10)
    pop_cfg = PopConfig(population_size=6)
    sim_cfg = SimConfig(
        seed=222,
        generations=1,
        steps_per_generation=3,
        device="cpu",
        bank_device="cpu",
        behavior_log_enable=False,
    )

    with tempfile.TemporaryDirectory() as td:
        sim = SimulationRunner(
            env_cfg=env_cfg,
            pop_cfg=pop_cfg,
            sim_cfg=sim_cfg,
            logger=NoOpMetricsLogger(),
            run_dir=td,
            perf=False,
        )
        sim._place_agents()

        epi = EPIEngine()
        scenarios = [
            StressScenarioV1(
                scenario_id="baseline_env",
                pressure_overrides={},
                duration_steps=1,
                seed_offset=0,
            )
        ]
        rec = epi.compute(sim=sim, generation_index=0, scenarios=scenarios)
        assert rec["schema_version"] == "epi_v2"
