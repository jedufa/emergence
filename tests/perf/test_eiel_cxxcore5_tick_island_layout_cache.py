from __future__ import annotations

from pathlib import Path
import shutil

import numpy as np
import pytest

from perf.cxxcore4_cost_layout_backend import compare_cost_vectors, evaluate_cost_layout_python
from perf.cxxcore5_tick_island_layout_cache import (
    CostLayoutCache,
    as_tick_island_inputs,
    audit_payload,
    cache_summary,
    checkpoint_metrics_boundary_classification,
    compare_runner_backend_states,
    compare_summaries,
    run_tick_island_python_reference,
    state_summary,
)
from simulation.runner_pkg.runner_config import SimConfig
from tools.scripts.eiel_cxxcore5_tick_island_layout_cache_probe import (
    _capture_fixture,
    compile_cpp_kernel,
    load_tick_island_backend,
)


def test_cxxcore5_layout_cache_full_rebuild_parity() -> None:
    inputs, _, full_snapshot, full_info, _, _, _, _ = _capture_fixture(n=8, width=8, height=8, ticks=4, seed=11)

    report = compare_cost_vectors(
        agent_ids=full_snapshot.layout.agent_ids,
        production=full_info["production_scalars"],
        layout=evaluate_cost_layout_python(full_snapshot.layout),
        actions_by_tick=inputs.backend.runner.state.actions_by_tick,
    )

    assert report["ok"]
    summary = cache_summary(full_snapshot)
    assert summary["cache_misses"] == 1
    assert summary["full_rebuilds"] == 1
    assert summary["rows_rebuilt"] == 8


def test_cxxcore5_layout_cache_hit_path_reuses_digest() -> None:
    _, _, full_snapshot, _, hit_snapshot, hit_info, _, _ = _capture_fixture(n=8, width=8, height=8, ticks=4, seed=12)

    assert hit_info["cache_mode"] == "hit"
    assert hit_snapshot.layout.digest == full_snapshot.layout.digest
    assert hit_snapshot.cache_hits == 1
    assert hit_snapshot.cache_misses == 1


def test_cxxcore5_conservative_row_update_parity() -> None:
    _, update_inputs, _, _, _, _, update_snapshot, update_info = _capture_fixture(n=8, width=8, height=8, ticks=4, seed=13)

    assert update_info["cache_mode"] == "row_update"
    assert update_snapshot.row_updates >= 1
    report = compare_cost_vectors(
        agent_ids=update_snapshot.layout.agent_ids,
        production=update_info["production_scalars"],
        layout=evaluate_cost_layout_python(update_snapshot.layout),
        actions_by_tick=update_inputs.backend.runner.state.actions_by_tick,
    )
    assert report["ok"]


def test_cxxcore5_tick_island_python_reference_parity() -> None:
    inputs, _, _, _, _, _, _, _ = _capture_fixture(n=8, width=8, height=8, ticks=5, seed=14)
    first = run_tick_island_python_reference(inputs)
    second = run_tick_island_python_reference(inputs)

    assert compare_runner_backend_states(first, second)["ok"]
    assert compare_summaries(state_summary(first), state_summary(second))["ok"]


def test_cxxcore5_native_tick_island_final_and_sparse_parity(tmp_path: Path) -> None:
    if shutil.which("g++") is None:
        pytest.skip("g++ unavailable")
    compile_info = compile_cpp_kernel(tmp_path)
    assert compile_info["ok"], compile_info
    native = load_tick_island_backend(tmp_path / "libcxxcore5_tick_island.so")
    inputs, _, _, _, _, _, _, _ = _capture_fixture(n=8, width=8, height=8, ticks=5, seed=15)

    expected = run_tick_island_python_reference(inputs)
    actual = native(inputs, export_final=True)
    sparse = native(inputs, export_final=False)

    assert compare_runner_backend_states(expected, actual, atol=1.0e-6)["ok"]
    assert compare_summaries(state_summary(expected), sparse["native_summary"], atol=1.0e-6)["ok"]


def test_cxxcore5_no_native_import_hard_failure_and_flags_off_behavior() -> None:
    payload = audit_payload()
    sim_cfg = SimConfig()
    classifications = checkpoint_metrics_boundary_classification()

    assert payload["finding"] == "cache_or_incremental_layout_feasible_for_shadow_no_rng_slice"
    assert sim_cfg.env_hotpath_cost_charge_surface_enable is False
    assert any(item["classification"] == "rebuildable shadow/cache state" for item in classifications)


def test_cxxcore5_cache_order_change_forces_full_rebuild() -> None:
    inputs, _, _, _, hit_snapshot, _, _, _ = _capture_fixture(n=8, width=8, height=8, ticks=4, seed=16)
    env_cfg = __import__("tools.scripts.eiel_cxxcore5_tick_island_layout_cache_probe", fromlist=["_env_cfg"])._env_cfg(8, 8)
    sim_cfg = __import__("tools.scripts.eiel_cxxcore5_tick_island_layout_cache_probe", fromlist=["_charged_cfg"])._charged_cfg()
    agent_factory = __import__("tools.scripts.eiel_cxxcore5_tick_island_layout_cache_probe", fromlist=["_agent"])._agent
    agents = [agent_factory(int(aid)) for aid in inputs.backend.runner.state.agent_ids.tolist()]
    cache = CostLayoutCache(env_cfg=env_cfg, sim_cfg=sim_cfg)
    cache.build_or_update(agents, force_full_rebuild=True)
    agents.reverse()
    snapshot, info = cache.build_or_update(agents)

    assert snapshot.layout.digest != hit_snapshot.layout.digest
    assert info["cache_mode"] == "full_rebuild"
    assert snapshot.invalidation_reasons["agent_order_changed"] == 1
