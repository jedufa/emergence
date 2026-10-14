from __future__ import annotations

from pathlib import Path
import shutil

import numpy as np
import pytest

from agents.cognition import compute_env_cost_charge_scalar_from_surface, express_cost_charge_surface
from env.env_config import EnvConfig
from perf.cxxcore4_cost_layout_backend import (
    as_cost_layout,
    as_cost_layout_backend_inputs,
    compare_cost_vectors,
    compare_runner_backend_states,
    compare_summaries,
    cost_layout_audit_payload,
    cost_matrix_from_layout,
    evaluate_cost_layout_python,
    layout_summary,
    run_cost_layout_backend_python_reference,
    state_summary,
)
from perf.cxxcore_env_apply_shadow import EnvApplyShadowConfig
from perf.native.cost_charge_surface_kernel import COST_SURFACE_COLUMN_NAMES, pack_cost_charge_surfaces
from tests.perf.test_eiel_costsurf1_cost_charge_surface import _agent, _charged_cfg
from tools.scripts.eiel_cxxcore4_cost_layout_backend_probe import compile_cpp_kernel, load_cost_layout_backend


def _production_layout(agent_count: int = 3):
    sim_cfg = _charged_cfg()
    env_cfg = EnvConfig(width=8, height=8, view_radius=2)
    agents = [_agent(i, pred_lr=0.1 + 0.05 * i, age=5 + i) for i in range(agent_count)]
    surfaces = [express_cost_charge_surface(agent.genome, agent=agent, env_cfg=env_cfg, sim_cfg=sim_cfg) for agent in agents]
    production = np.asarray(
        [compute_env_cost_charge_scalar_from_surface(surface, sim_cfg=sim_cfg) for surface in surfaces],
        dtype=np.float64,
    )
    layout = as_cost_layout(
        agent_ids=np.arange(agent_count, dtype=np.int32),
        rows=pack_cost_charge_surfaces(surfaces),
        charge_channels=sim_cfg.enable_experimental_perception_cost_charging,
        charge_radius=sim_cfg.enable_experimental_sensor_radius_cost_charging,
        ledger_enabled=sim_cfg.enable_cognitive_budget_ledger,
    )
    return layout, production


def _backend_inputs():
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
    agent_ids = np.asarray([0, 1, 2], dtype=np.int32)
    rows = np.asarray(
        [
            [0.01, 0.02, 0.03, 0.00, 0.01, 0.02, 0.03, 0.04],
            [0.02, 0.01, 0.04, 0.01, 0.01, 0.02, 0.01, 0.02],
            [0.00, 0.03, 0.01, 0.02, 0.00, 0.01, 0.03, 0.01],
        ],
        dtype=np.float64,
    )
    layout = as_cost_layout(
        agent_ids=agent_ids,
        rows=rows,
        charge_channels=True,
        charge_radius=True,
        ledger_enabled=True,
    )
    pos_y_by_id = np.asarray([1, 1, 2], dtype=np.int32)
    pos_x_by_id = np.asarray([1, 2, 0], dtype=np.int32)
    agent_grid = np.full((4, 5), -1, dtype=np.int32)
    agent_grid[1, 1] = 0
    agent_grid[1, 2] = 1
    agent_grid[2, 0] = 2
    grid = np.zeros((4, 5), dtype=np.int8)
    grid[1, 1] = 1
    actions = np.asarray([[4, 2, 0], [5, 5, 5], [3, 4, 1]], dtype=np.int32)
    return as_cost_layout_backend_inputs(
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
        layout=layout,
    )


def test_cxxcore4_python_layout_matches_production_surface_scalar() -> None:
    layout, production = _production_layout()

    actual = evaluate_cost_layout_python(layout)
    report = compare_cost_vectors(agent_ids=layout.agent_ids, production=production, layout=actual)

    assert report["ok"]
    assert report["max_abs_diff"] == 0.0
    assert float(np.sum(actual)) > 0.0


def test_cxxcore4_layout_digest_stable_and_summary_explicit() -> None:
    layout, _ = _production_layout()
    same = as_cost_layout(
        agent_ids=layout.agent_ids.copy(),
        rows=layout.rows.copy(),
        charge_channels=layout.charge_channels,
        charge_radius=layout.charge_radius,
        ledger_enabled=layout.ledger_enabled,
    )

    assert same.digest == layout.digest
    summary = layout_summary(layout)
    assert summary["schema_version"] == "cxxcore_cost_layout_v1"
    assert summary["field_names"] == list(COST_SURFACE_COLUMN_NAMES)
    assert summary["omitted_fields"] == []


def test_cxxcore4_action_edge_fixture_uses_stable_scalar_matrix() -> None:
    layout, _ = _production_layout()
    actions = np.asarray([[0, 4, 5], [1, 6, 3]], dtype=np.int32)

    matrix = cost_matrix_from_layout(layout, actions)

    assert matrix.shape == actions.shape
    np.testing.assert_allclose(matrix[0], matrix[1], rtol=0.0, atol=0.0)


def test_cxxcore4_rejects_bad_layout_shape() -> None:
    with pytest.raises(ValueError, match="exactly 8 columns"):
        as_cost_layout(
            agent_ids=np.asarray([1, 2], dtype=np.int32),
            rows=np.zeros((2, 7), dtype=np.float64),
            charge_channels=True,
            charge_radius=True,
        )


def test_cxxcore4_audit_classifies_layout_feasible_with_omissions() -> None:
    payload = cost_layout_audit_payload()

    assert payload["preliminary_finding"] == "layout_contract_feasible_with_limited_omissions"
    assert len(payload["represented_fields"]) == 8
    assert payload["rng_boundary"].startswith("no RNG")


def test_cxxcore4_native_scalar_and_backend_match_python_reference(tmp_path: Path) -> None:
    if shutil.which("g++") is None:
        pytest.skip("g++ unavailable")
    compile_info = compile_cpp_kernel(tmp_path)
    assert compile_info["ok"], compile_info
    native_scalars, native_backend = load_cost_layout_backend(tmp_path / "libcxxcore4_cost_layout_backend.so")
    inputs = _backend_inputs()

    np.testing.assert_allclose(
        native_scalars(inputs.layout),
        evaluate_cost_layout_python(inputs.layout),
        rtol=0.0,
        atol=1e-12,
    )

    expected = run_cost_layout_backend_python_reference(inputs)
    actual = native_backend(inputs, export_final=True)
    assert compare_runner_backend_states(expected, actual, atol=1.0e-6)["ok"]
    assert compare_summaries(state_summary(expected), actual["native_summary"], atol=1.0e-6)["ok"]
