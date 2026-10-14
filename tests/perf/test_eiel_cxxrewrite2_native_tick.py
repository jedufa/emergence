from __future__ import annotations

from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np
import pytest

from perf.cxxcore2_state_backend import as_backend_inputs, compare_backend_states, run_backend_python_reference
from perf.cxxcore_env_apply_shadow import EnvApplyShadowConfig
from perf.cxxrewrite2_native_tick import (
    assert_native_authoritative_not_supported,
    compile_native_backend,
    load_native_backend,
    native_api_schema,
    native_backend_status,
    native_tick_requested,
    prepare_runner_shadow_snapshot,
    run_runner_shadow_compare,
    selected_slice_audit,
)
from simulation.runner_pkg.runner_config import SimConfig


def _inputs():
    config = EnvApplyShadowConfig(
        height=4,
        width=5,
        move_cost=1.0,
        gather_cost=0.5,
        rest_cost=0.1,
        attack_cost=2.0,
        action_cost_mult=1.0,
        rest_cost_mult=1.0,
        food_energy=3.0,
        max_energy=20.0,
    )
    agent_ids = np.asarray([7, 2, 5], dtype=np.int32)
    pos_y_by_id = np.full((8,), -1, dtype=np.int32)
    pos_x_by_id = np.full((8,), -1, dtype=np.int32)
    pos_y_by_id[[7, 2, 5]] = np.asarray([1, 1, 2], dtype=np.int32)
    pos_x_by_id[[7, 2, 5]] = np.asarray([1, 2, 0], dtype=np.int32)
    agent_grid = np.full((4, 5), -1, dtype=np.int32)
    agent_grid[1, 1] = 7
    agent_grid[1, 2] = 2
    agent_grid[2, 0] = 5
    grid = np.zeros((4, 5), dtype=np.int8)
    grid[1, 1] = 1
    actions = np.asarray([[4, 2, 0], [5, 5, 5], [3, 4, 1]], dtype=np.int32)
    return as_backend_inputs(
        config=config,
        agent_ids=agent_ids,
        initial_energy=np.asarray([10.0, 9.0, 8.0], dtype=np.float32),
        initial_health=np.asarray([9.0, 7.0, 4.0], dtype=np.float32),
        initial_injury=np.zeros((3,), dtype=np.float32),
        initial_scar=np.zeros((3,), dtype=np.float32),
        pos_y_by_id=pos_y_by_id,
        pos_x_by_id=pos_x_by_id,
        agent_grid=agent_grid,
        grid=grid,
        actions_by_tick=actions,
    )


def test_cxxrewrite2_default_flags_off_and_contracts_present() -> None:
    cfg = SimConfig()

    assert native_tick_requested(cfg) is False
    assert cfg.cxxrewrite_native_tick_enable is False
    assert cfg.cxxrewrite_native_tick_shadow_enable is False
    assert cfg.cxxrewrite_native_tick_authoritative_enable is False
    assert "movement" in native_api_schema()["included_semantics"]
    assert selected_slice_audit()["native_authoritative_supported"] is False
    assert native_backend_status()["normal_import_requires_native"] is False


def test_cxxrewrite2_authoritative_fails_closed() -> None:
    cfg = SimConfig(cxxrewrite_native_tick_enable=True, cxxrewrite_native_tick_authoritative_enable=True)

    with pytest.raises(RuntimeError, match="not supported"):
        assert_native_authoritative_not_supported(cfg)


def test_cxxrewrite2_native_backend_matches_python_reference(tmp_path: Path) -> None:
    if shutil.which("g++") is None:
        pytest.skip("g++ unavailable")
    build = compile_native_backend(tmp_path)
    assert build["ok"], build
    inputs = _inputs()
    expected = run_backend_python_reference(inputs, checkpoint_interval=2)
    actual = load_native_backend(build["library_path"])(inputs, checkpoint_interval=2)

    assert compare_backend_states(expected, actual)["ok"]
    assert actual["checkpoint_count"] == 2


def test_cxxrewrite2_runner_shadow_missing_native_records_skip(tmp_path: Path) -> None:
    env_cfg = SimpleNamespace(
        enable_biomes=False,
        patch_world_enable=False,
        migration_pressure_enable=False,
        enable_combat=False,
        enable_injury=False,
        enable_scar=False,
        tier3_validation_force_random_walk=False,
        gru_cost_base=0.0,
        gru_cost_per_dim=0.0,
        height=4,
        width=5,
        move_cost=1.0,
        gather_cost=0.5,
        rest_cost=0.1,
        attack_cost=2.0,
        food_energy=3.0,
        max_energy=20.0,
    )
    sim_cfg = SimConfig(
        cxxrewrite_native_tick_enable=True,
        cxxrewrite_native_tick_shadow_enable=True,
        cxxrewrite_native_tick_validate_shadow_enable=True,
        cxxrewrite_native_tick_library_path=str(tmp_path / "missing.so"),
    )
    env = SimpleNamespace(
        cfg=env_cfg,
        action_cost_mult=1.0,
        rest_cost_mult=1.0,
        agent_pos_y=np.asarray([1, 1], dtype=np.int32),
        agent_pos_x=np.asarray([1, 2], dtype=np.int32),
        agent_alive=np.asarray([True, True], dtype=bool),
        agent_grid=np.asarray([[-1, -1, -1], [-1, 0, 1], [-1, -1, -1]], dtype=np.int32),
        grid=np.zeros((3, 3), dtype=np.int8),
    )
    runner = SimpleNamespace(env=env, env_cfg=env_cfg, sim_cfg=sim_cfg, run_dir=str(tmp_path))
    alive = [
        SimpleNamespace(agent_id=0, alive=True, energy=10.0, health=8.0, injury=0.0, scar=0.0),
        SimpleNamespace(agent_id=1, alive=True, energy=9.0, health=8.0, injury=0.0, scar=0.0),
    ]
    snapshot, record = prepare_runner_shadow_snapshot(
        runner,
        alive=alive,
        agent_ids_np=np.asarray([0, 1], dtype=np.int32),
        actions_np=np.asarray([5, 5], dtype=np.int32),
        energies=np.asarray([10.0, 9.0], dtype=np.float32),
        healths=np.asarray([8.0, 8.0], dtype=np.float32),
        injuries=np.zeros((2,), dtype=np.float32),
        scars=np.zeros((2,), dtype=np.float32),
        move_cost_mults=None,
        juvenile_harvest_mults=None,
    )

    out = run_runner_shadow_compare(
        runner,
        snapshot=snapshot,
        record=record,
        new_energy=np.asarray([9.9, 8.9], dtype=np.float32),
        new_health=np.asarray([8.0, 8.0], dtype=np.float32),
        new_injury=np.zeros((2,), dtype=np.float32),
        new_scar=np.zeros((2,), dtype=np.float32),
        food_gained=np.zeros((2,), dtype=np.float32),
        died_mask=np.zeros((2,), dtype=bool),
    )

    assert out["skipped"] is True
    assert out["skip_reasons"]["native_library_missing"] is True
