from __future__ import annotations

from pathlib import Path
import shutil

import numpy as np
import pytest

from perf.cxxcore_env_apply_shadow import EnvApplyShadowConfig
from perf.cxxcore2_state_backend import (
    as_backend_inputs,
    compare_backend_states,
    run_backend_python_reference,
)
from tools.scripts.eiel_cxxcore2_state_backend_native_smoke import compile_cpp_kernel, load_backend


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


def test_cxxcore2_python_backend_reference_is_deterministic() -> None:
    inputs = _inputs()
    first = run_backend_python_reference(inputs, checkpoint_interval=2)
    second = run_backend_python_reference(inputs, checkpoint_interval=2)

    assert compare_backend_states(first, second)["ok"]
    assert first["checkpoint_energy"].shape == (2, 3)


def test_cxxcore2_comparator_localizes_grid_divergence() -> None:
    inputs = _inputs()
    expected = run_backend_python_reference(inputs)
    actual = {key: np.copy(value) for key, value in expected.items()}
    actual["grid"][1, 1] = np.int8(1)

    report = compare_backend_states(expected, actual)

    assert not report["ok"]
    assert report["first_divergence"]["field"] == "grid"
    assert report["first_divergence"]["classification"] == "gather"


def test_cxxcore2_rejects_wrong_action_shape() -> None:
    inputs = _inputs()
    with pytest.raises(ValueError, match="columns must match"):
        as_backend_inputs(
            config=inputs.config,
            agent_ids=inputs.agent_ids,
            initial_energy=inputs.initial_energy,
            initial_health=inputs.initial_health,
            initial_injury=inputs.initial_injury,
            initial_scar=inputs.initial_scar,
            pos_y_by_id=inputs.pos_y_by_id,
            pos_x_by_id=inputs.pos_x_by_id,
            agent_grid=inputs.agent_grid,
            grid=inputs.grid,
            actions_by_tick=np.zeros((2, 2), dtype=np.int32),
        )


def test_cxxcore2_native_backend_matches_python_reference(tmp_path: Path) -> None:
    if shutil.which("g++") is None:
        pytest.skip("g++ unavailable")
    compile_info = compile_cpp_kernel(tmp_path)
    assert compile_info["ok"], compile_info
    inputs = _inputs()
    expected = run_backend_python_reference(inputs, checkpoint_interval=2)
    actual = load_backend(tmp_path / "libcxxcore2_state_backend.so")(inputs, checkpoint_interval=2)

    assert compare_backend_states(expected, actual)["ok"]
    assert actual["checkpoint_count"] == 2
