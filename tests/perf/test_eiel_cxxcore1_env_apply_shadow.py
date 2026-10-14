from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import shutil

import numpy as np
import pytest

from env.env_config import EnvConfig
from env.gridworld_pkg.world import GridWorld
from perf.cxxcore_env_apply_shadow import (
    compare_shadow_deltas,
    compute_env_apply_shadow_python,
    config_from_env,
    feature_skip_reasons,
    observed_delta_from_python,
)
from simulation.rng_manager import RNGManager
from simulation.runner_pkg.runner_config import SimConfig
from simulation.runner_pkg.runner_ticks_env_apply import TickEnvApplyMixin
from tools.scripts.eiel_cxxcore1_env_apply_shadow_native_smoke import (
    compile_cpp_kernel,
    load_env_apply_no_rng,
)


class _Stats:
    survival_steps = 0
    energy_collected = 0.0
    early_survival_steps = 0
    early_energy_collected = 0.0
    early_injury_gained = 0.0


class _Agent:
    def __init__(self, agent_id: int, *, energy: float, health: float = 10.0) -> None:
        self.agent_id = int(agent_id)
        self.energy = float(energy)
        self.health = float(health)
        self.injury = 0.0
        self.scar = 0.0
        self.alive = True
        self.stats = _Stats()


class _Runner(TickEnvApplyMixin):
    pass


def _no_rng_env_cfg() -> EnvConfig:
    return EnvConfig(
        width=5,
        height=4,
        food_spawn_per_step=0,
        max_food=0,
        enable_biomes=False,
        enable_combat=False,
        enable_injury=False,
        enable_scar=False,
        migration_pressure_enable=False,
        tier3_validation_force_random_walk=False,
        gru_cost_base=0.0,
        gru_cost_per_dim=0.0,
    )


def _build_world() -> GridWorld:
    cfg = _no_rng_env_cfg()
    rng_manager = RNGManager(global_seed=101)
    env = GridWorld(cfg, seed=101, rng=rng_manager.env_np)
    env.add_agent(7, pos=(1, 1))
    env.add_agent(2, pos=(1, 2))
    env.add_agent(5, pos=(2, 0))
    env.grid[:, :] = 0
    env.grid[1, 1] = 1
    return env


def _fixture_arrays(env: GridWorld) -> dict[str, np.ndarray]:
    agent_ids = np.asarray([7, 2, 5], dtype=np.int32)
    return {
        "agent_ids": agent_ids,
        "actions": np.asarray([4, 2, 0], dtype=np.int32),
        "energies": np.asarray([5.0, 8.0, 0.75], dtype=np.float32),
        "healths": np.asarray([9.0, 7.0, 4.0], dtype=np.float32),
        "injuries": np.zeros((3,), dtype=np.float32),
        "scars": np.zeros((3,), dtype=np.float32),
        "pre_pos_y": env.agent_pos_y[agent_ids].copy(),
        "pre_pos_x": env.agent_pos_x[agent_ids].copy(),
        "pos_y_by_id": env.agent_pos_y.copy(),
        "pos_x_by_id": env.agent_pos_x.copy(),
        "agent_grid": env.agent_grid.copy(),
        "grid": env.grid.copy(),
        "move_cost_mults": np.ones((3,), dtype=np.float32),
        "juvenile_harvest_mults": np.ones((3,), dtype=np.float32),
    }


def _shadow_kwargs(data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {
        key: value
        for key, value in data.items()
        if key not in {"pre_pos_y", "pre_pos_x"}
    }


def test_cxxcore1_helper_matches_gridworld_no_rng_delta() -> None:
    env = _build_world()
    data = _fixture_arrays(env)
    shadow = compute_env_apply_shadow_python(config=config_from_env(env), **_shadow_kwargs(data))
    died, new_energy, new_health, food_gained, new_injury, new_scar = env.step_agents_batch(
        agent_ids=data["agent_ids"],
        actions=data["actions"],
        energies=data["energies"].copy(),
        healths=data["healths"].copy(),
        injuries=data["injuries"].copy(),
        scars=data["scars"].copy(),
        move_cost_mults=data["move_cost_mults"],
        juvenile_harvest_mults=data["juvenile_harvest_mults"],
    )
    observed = observed_delta_from_python(
        agent_ids=data["agent_ids"],
        pre_energy=data["energies"],
        pre_health=data["healths"],
        pre_injury=data["injuries"],
        pre_scar=data["scars"],
        post_energy=new_energy,
        post_health=new_health,
        post_injury=new_injury,
        post_scar=new_scar,
        food_gained=food_gained,
        died_mask=died,
        pre_pos_y=data["pre_pos_y"],
        pre_pos_x=data["pre_pos_x"],
        post_pos_y_by_id=env.agent_pos_y,
        post_pos_x_by_id=env.agent_pos_x,
    )

    assert compare_shadow_deltas(observed, shadow)["ok"]


def test_cxxcore1_skip_reasons_reject_default_rng_surfaces() -> None:
    reasons = feature_skip_reasons(EnvConfig(), SimConfig())

    assert reasons["biome_rng"]
    assert reasons["combat"]
    assert reasons["injury_cost_surface"]
    assert reasons["scar_cost_surface"]


def test_cxxcore1_comparator_reports_first_divergence() -> None:
    env = _build_world()
    data = _fixture_arrays(env)
    expected = compute_env_apply_shadow_python(config=config_from_env(env), **_shadow_kwargs(data))
    actual = {k: np.copy(v) for k, v in expected.items()}
    actual["new_energy"][1] = np.float32(actual["new_energy"][1] + 1.0)

    report = compare_shadow_deltas(expected, actual)

    assert not report["ok"]
    assert report["first_divergence"]["field"] == "new_energy"
    assert report["first_divergence"]["index"] == 1


def _build_runner(tmp_path: Path, *, shadow: bool) -> tuple[_Runner, list[_Agent], np.ndarray, np.ndarray]:
    env = _build_world()
    sim_cfg = SimConfig()
    sim_cfg.cxxcore_env_apply_shadow_enable = bool(shadow)
    sim_cfg.cxxcore_env_apply_shadow_fail_fast = True
    sim_cfg.cxxcore_env_apply_shadow_write_summary = bool(shadow)
    runner = _Runner()
    runner.env = env
    runner.env_cfg = env.cfg
    runner.sim_cfg = sim_cfg
    runner.rng_manager = SimpleNamespace(policy_np=np.random.default_rng(202))
    runner.run_dir = str(tmp_path)
    runner.tick = 0
    runner._current_gen = 0
    alive = [
        _Agent(7, energy=5.0, health=9.0),
        _Agent(2, energy=8.0, health=7.0),
        _Agent(5, energy=0.75, health=4.0),
    ]
    return runner, alive, np.asarray([7, 2, 5], dtype=np.int32), np.asarray([4, 2, 0], dtype=np.int32)


def test_cxxcore1_runner_shadow_keeps_authoritative_rows_unchanged(tmp_path: Path) -> None:
    base_runner, base_alive, base_ids, base_actions = _build_runner(tmp_path / "base", shadow=False)
    shadow_runner, shadow_alive, shadow_ids, shadow_actions = _build_runner(tmp_path / "shadow", shadow=True)

    base_runner._apply_env_tick(base_alive, base_ids, base_actions, {}, 0, 0.0)
    shadow_runner._apply_env_tick(shadow_alive, shadow_ids, shadow_actions, {}, 0, 0.0)

    np.testing.assert_allclose(
        [a.energy for a in shadow_alive],
        [a.energy for a in base_alive],
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_array_equal(shadow_runner.env.agent_pos_y, base_runner.env.agent_pos_y)
    np.testing.assert_array_equal(shadow_runner.env.agent_pos_x, base_runner.env.agent_pos_x)
    assert shadow_runner._cxxcore_env_apply_shadow_last["parity"]["ok"]
    assert (tmp_path / "shadow" / "metrics" / "cxxcore_env_apply_shadow_summary.json").exists()


def test_cxxcore1_native_kernel_matches_python_reference(tmp_path: Path) -> None:
    if shutil.which("g++") is None:
        pytest.skip("g++ unavailable")
    compile_info = compile_cpp_kernel(tmp_path)
    assert compile_info["ok"], compile_info
    env = _build_world()
    data = _fixture_arrays(env)
    expected = compute_env_apply_shadow_python(config=config_from_env(env), **_shadow_kwargs(data))
    native = load_env_apply_no_rng(tmp_path / "libcxxcore1_env_apply_shadow.so")
    actual = native(config=config_from_env(env), **_shadow_kwargs(data))

    assert compare_shadow_deltas(expected, actual)["ok"]
