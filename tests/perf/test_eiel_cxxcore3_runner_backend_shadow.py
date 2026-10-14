from __future__ import annotations

from pathlib import Path
import shutil

import numpy as np
import pytest

from perf.cxxcore3_runner_backend_shadow import (
    as_runner_backend_shadow_inputs,
    compare_runner_backend_states,
    compare_summaries,
    deterministic_cost_scalar_matrix,
    run_runner_backend_python_reference,
    state_summary,
)
from perf.cxxcore_env_apply_shadow import EnvApplyShadowConfig
from tools.scripts.eiel_cxxcore3_runner_backend_shadow_probe import compile_cpp_kernel, load_runner_backend


def _inputs(*, cost_enabled: bool = False):
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
    costs = deterministic_cost_scalar_matrix(agent_ids, actions, enabled=cost_enabled)
    return as_runner_backend_shadow_inputs(
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
        cost_scalars_by_tick=costs,
    )


def test_cxxcore3_python_runner_shadow_no_cost_parity() -> None:
    inputs = _inputs(cost_enabled=False)
    first = run_runner_backend_python_reference(inputs)
    second = run_runner_backend_python_reference(inputs)

    assert compare_runner_backend_states(first, second)["ok"]


def test_cxxcore3_sparse_summary_parity() -> None:
    inputs = _inputs(cost_enabled=False)
    state = run_runner_backend_python_reference(inputs)

    report = compare_summaries(state_summary(state), state_summary(state))

    assert report["ok"]
    assert "energy_sum" in report["fields_compared"]


def test_cxxcore3_cost_scalar_fallback_parity() -> None:
    inputs = _inputs(cost_enabled=True)
    expected = run_runner_backend_python_reference(inputs)
    actual = run_runner_backend_python_reference(inputs)

    report = compare_runner_backend_states(expected, actual)

    assert report["ok"]
    assert float(expected["cost_total"][0]) > 0.0


def test_cxxcore3_rejects_bad_cost_shape() -> None:
    inputs = _inputs(cost_enabled=False)
    with pytest.raises(ValueError, match="cost_scalars_by_tick"):
        as_runner_backend_shadow_inputs(
            config=inputs.state.config,
            agent_ids=inputs.state.agent_ids,
            initial_energy=inputs.state.initial_energy,
            initial_health=inputs.state.initial_health,
            initial_injury=inputs.state.initial_injury,
            initial_scar=inputs.state.initial_scar,
            pos_y_by_id=inputs.state.pos_y_by_id,
            pos_x_by_id=inputs.state.pos_x_by_id,
            agent_grid=inputs.state.agent_grid,
            grid=inputs.state.grid,
            actions_by_tick=inputs.state.actions_by_tick,
            cost_scalars_by_tick=np.zeros((1, 1), dtype=np.float32),
        )


def test_cxxcore3_native_runner_backend_matches_python_reference(tmp_path: Path) -> None:
    if shutil.which("g++") is None:
        pytest.skip("g++ unavailable")
    compile_info = compile_cpp_kernel(tmp_path)
    assert compile_info["ok"], compile_info
    backend = load_runner_backend(tmp_path / "libcxxcore3_runner_backend.so")
    inputs = _inputs(cost_enabled=True)
    expected = run_runner_backend_python_reference(inputs)
    actual = backend(inputs, export_final=True)

    assert compare_runner_backend_states(expected, actual)["ok"]
    assert compare_summaries(state_summary(expected), actual["native_summary"])["ok"]
