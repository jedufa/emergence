from __future__ import annotations

from pathlib import Path
import shutil
from typing import Any

import pytest

from perf.cxxcore6_authoritative_pilot import (
    Cxxcore6PilotConfig,
    audit_payload,
    boundary_classification,
    config_from_sim_cfg,
    rng_preflight_audit,
    run_authoritative_pilot,
)
from perf.cxxcore5_tick_island_layout_cache import (
    compare_runner_backend_states,
    compare_summaries,
    run_tick_island_python_reference,
    state_summary,
)
from simulation.runner_pkg.runner_config import SimConfig
from tools.scripts.eiel_cxxcore6_authoritative_pilot_probe import (
    _capture_fixture,
    compile_cpp_kernel,
    load_authoritative_backend,
)


def _native_backend(tmp_path: Path) -> Any:
    if shutil.which("g++") is None:
        pytest.skip("g++ unavailable")
    compile_info = compile_cpp_kernel(tmp_path)
    assert compile_info["ok"], compile_info
    return load_authoritative_backend(Path(str(compile_info["library_path"])))


def test_cxxcore6_default_off_config_and_disabled_status() -> None:
    sim_cfg = SimConfig()
    config = config_from_sim_cfg(sim_cfg)
    inputs, _, _, _, _, _, _, _ = _capture_fixture(n=8, width=8, height=8, ticks=4, seed=6101)

    result = run_authoritative_pilot(inputs, config=config, native_backend=None)

    assert sim_cfg.cxxcore6_authoritative_pilot_enable is False
    assert config.cxxcore6_authoritative_pilot_enable is False
    assert result.status == "disabled"
    assert result.fallback_occurred is False
    assert result.output is None


def test_cxxcore6_native_unavailable_falls_back_without_hard_dependency() -> None:
    inputs, _, _, _, _, _, _, _ = _capture_fixture(n=8, width=8, height=8, ticks=4, seed=6102)
    expected = run_tick_island_python_reference(inputs)
    config = Cxxcore6PilotConfig(cxxcore6_authoritative_pilot_enable=True)

    result = run_authoritative_pilot(inputs, config=config, native_backend=None, python_reference=expected)

    assert result.status == "fallback"
    assert result.fallback_occurred is True
    assert result.fallback_reason == "native_unavailable"
    assert result.python_state_restored is True
    assert compare_runner_backend_states(expected, result.output)["ok"]


def test_cxxcore6_force_python_baseline_kill_switch() -> None:
    inputs, _, _, _, _, _, _, _ = _capture_fixture(n=8, width=8, height=8, ticks=4, seed=6103)
    expected = run_tick_island_python_reference(inputs)
    config = Cxxcore6PilotConfig(cxxcore6_authoritative_pilot_enable=True, cxxcore6_force_python_baseline=True)

    result = run_authoritative_pilot(inputs, config=config, native_backend=None, python_reference=expected)

    assert result.status == "fallback"
    assert result.fallback_reason == "force_python_baseline"
    assert compare_runner_backend_states(expected, result.output)["ok"]


def test_cxxcore6_no_rng_authoritative_final_state_parity(tmp_path: Path) -> None:
    native = _native_backend(tmp_path)
    inputs, _, _, _, _, _, _, _ = _capture_fixture(n=8, width=8, height=8, ticks=5, seed=6104)
    expected = run_tick_island_python_reference(inputs)
    config = Cxxcore6PilotConfig(
        cxxcore6_authoritative_pilot_enable=True,
        cxxcore6_authoritative_compare_mode="final_full",
    )

    result = run_authoritative_pilot(inputs, config=config, native_backend=native, python_reference=expected)

    assert result.status == "authoritative"
    assert result.fallback_occurred is False
    assert result.cxx_result_retained is True
    assert result.parity is not None
    assert result.parity["ok"]
    assert compare_runner_backend_states(expected, result.output, atol=1.0e-6)["ok"]


def test_cxxcore6_sparse_summary_parity(tmp_path: Path) -> None:
    native = _native_backend(tmp_path)
    inputs, _, _, _, _, _, _, _ = _capture_fixture(n=8, width=8, height=8, ticks=5, seed=6105)
    expected = run_tick_island_python_reference(inputs)
    config = Cxxcore6PilotConfig(
        cxxcore6_authoritative_pilot_enable=True,
        cxxcore6_authoritative_compare_mode="sparse",
    )

    result = run_authoritative_pilot(inputs, config=config, native_backend=native, python_reference=expected)

    assert result.status == "authoritative"
    assert result.parity is not None
    assert result.parity["ok"]
    assert compare_summaries(state_summary(expected), state_summary(result.output), atol=1.0e-6)["ok"]


def test_cxxcore6_injected_divergence_falls_back_to_python_reference(tmp_path: Path) -> None:
    native = _native_backend(tmp_path)
    inputs, _, _, _, _, _, _, _ = _capture_fixture(n=8, width=8, height=8, ticks=5, seed=6106)
    expected = run_tick_island_python_reference(inputs)
    config = Cxxcore6PilotConfig(
        cxxcore6_authoritative_pilot_enable=True,
        cxxcore6_authoritative_compare_mode="final_full",
        cxxcore6_authoritative_fallback_on_divergence=True,
    )

    result = run_authoritative_pilot(
        inputs,
        config=config,
        native_backend=native,
        python_reference=expected,
        inject_divergence=True,
    )

    assert result.status == "fallback"
    assert result.fallback_reason == "divergence"
    assert result.python_state_restored is True
    assert result.cxx_result_retained is False
    assert result.parity is not None
    assert result.parity["ok"] is False
    assert compare_runner_backend_states(expected, result.output, atol=1.0e-6)["ok"]


def test_cxxcore6_boundary_audits_do_not_add_checkpoint_or_metrics_state() -> None:
    audit = audit_payload()
    boundary = boundary_classification()
    rng = rng_preflight_audit()

    assert audit["checkpoint_boundary"].startswith("no checkpoint-required state added")
    assert audit["metrics_boundary"].startswith("no metrics definitions changed")
    assert audit["rng_boundary"].startswith("no independent C++ RNG")
    assert any(item["classification"] == "guarded authoritative pilot writeback state" for item in boundary)
    assert rng["precomputed_rng_contract"]["status"] == "deferred"
    assert "validation random-walk action overrides" in rng["precomputed_rng_contract"]["required_arrays"]
