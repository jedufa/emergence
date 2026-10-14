from __future__ import annotations

from pathlib import Path
import shutil

import numpy as np
import pytest

from perf.cxxcore5_tick_island_layout_cache import compare_runner_backend_states, run_tick_island_python_reference
from perf.cxxcore7_rng_precompute import (
    apply_validation_action_override,
    build_rng_payload,
    precompute_validation_action_override_ints,
)
from perf.cxxcore8_guarded_runner_bridge import (
    Cxxcore8BridgeConfig,
    bridge_audit,
    config_from_sim_cfg,
    run_guarded_runner_bridge,
)
from simulation.runner_pkg.runner_config import SimConfig
from tools.scripts.eiel_cxxcore5_tick_island_layout_cache_probe import _capture_fixture
from tools.scripts.eiel_cxxcore7_rng_precompute_runner_bridge_probe import (
    compile_cpp_kernel,
    load_rng_precompute_backend,
)


def _payload(*, ticks: int, agents: int, seed: int = 8108):
    return build_rng_payload(
        validation_action_override_ints=precompute_validation_action_override_ints(
            rng=np.random.default_rng(seed),
            ticks=ticks,
            agents=agents,
        )
    )


def _native_backend(tmp_path: Path):
    if shutil.which("g++") is None:
        pytest.skip("g++ unavailable")
    compile_info = compile_cpp_kernel(tmp_path)
    assert compile_info["ok"], compile_info
    return load_rng_precompute_backend(Path(str(compile_info["library_path"])))


def test_cxxcore8_default_off_flags_and_disabled_status() -> None:
    sim_cfg = SimConfig()
    config = config_from_sim_cfg(sim_cfg)
    inputs, _, _, _, _, _, _, _ = _capture_fixture(n=8, width=8, height=8, ticks=4, seed=8101)

    result = run_guarded_runner_bridge(inputs, config=config, native_backend=None)

    assert config.cxxcore8_guarded_runner_bridge_enable is False
    assert sim_cfg.cxxcore8_guarded_runner_bridge_enable is False
    assert result.status == "disabled"
    assert result.writeback_applied is False


def test_cxxcore8_shadow_no_rng_parity(tmp_path: Path) -> None:
    native = _native_backend(tmp_path)
    inputs, _, _, _, _, _, _, _ = _capture_fixture(n=8, width=8, height=8, ticks=5, seed=8102)
    config = Cxxcore8BridgeConfig(
        cxxcore8_guarded_runner_bridge_enable=True,
        cxxcore8_guarded_runner_bridge_mode="shadow",
    )

    result = run_guarded_runner_bridge(inputs, config=config, native_backend=native)

    assert result.status == "shadow"
    assert result.writeback_applied is False
    assert result.cxx_result_retained is False
    assert result.parity is not None and result.parity["ok"]
    assert result.sparse_parity is not None and result.sparse_parity["ok"]


def test_cxxcore8_authoritative_no_rng_parity_and_writeback(tmp_path: Path) -> None:
    native = _native_backend(tmp_path)
    inputs, _, _, _, _, _, _, _ = _capture_fixture(n=8, width=8, height=8, ticks=5, seed=8103)
    expected = run_tick_island_python_reference(inputs)
    config = Cxxcore8BridgeConfig(
        cxxcore8_guarded_runner_bridge_enable=True,
        cxxcore8_guarded_runner_bridge_mode="authoritative_no_rng",
    )

    result = run_guarded_runner_bridge(inputs, config=config, native_backend=native)

    assert result.status == "authoritative"
    assert result.writeback_applied is True
    assert result.cxx_result_retained is True
    assert result.parity is not None and result.parity["ok"]
    assert compare_runner_backend_states(expected, result.output, atol=1.0e-6)["ok"]


def test_cxxcore8_authoritative_precomputed_rng_parity(tmp_path: Path) -> None:
    native = _native_backend(tmp_path)
    inputs, _, _, _, _, _, _, _ = _capture_fixture(n=8, width=8, height=8, ticks=5, seed=8104)
    payload = _payload(ticks=5, agents=8)
    expected = run_tick_island_python_reference(apply_validation_action_override(inputs, payload))
    config = Cxxcore8BridgeConfig(
        cxxcore8_guarded_runner_bridge_enable=True,
        cxxcore8_guarded_runner_bridge_mode="authoritative_precomputed_rng",
        cxxcore8_runner_bridge_allow_validation_rng_precompute=True,
    )

    result = run_guarded_runner_bridge(inputs, config=config, native_backend=native, rng_payload=payload)

    assert result.status == "authoritative"
    assert result.writeback_applied is True
    assert result.parity is not None and result.parity["ok"]
    assert compare_runner_backend_states(expected, result.output, atol=1.0e-6)["ok"]


def test_cxxcore8_missing_native_fallback() -> None:
    inputs, _, _, _, _, _, _, _ = _capture_fixture(n=8, width=8, height=8, ticks=5, seed=8105)
    config = Cxxcore8BridgeConfig(
        cxxcore8_guarded_runner_bridge_enable=True,
        cxxcore8_guarded_runner_bridge_mode="authoritative_no_rng",
    )

    result = run_guarded_runner_bridge(inputs, config=config, native_backend=None)

    assert result.status == "fallback"
    assert result.fallback_reason == "native_unavailable"
    assert result.python_state_restored is True
    assert result.cxx_result_retained is False


def test_cxxcore8_missing_rng_payload_fallback() -> None:
    inputs, _, _, _, _, _, _, _ = _capture_fixture(n=8, width=8, height=8, ticks=5, seed=8106)
    config = Cxxcore8BridgeConfig(
        cxxcore8_guarded_runner_bridge_enable=True,
        cxxcore8_guarded_runner_bridge_mode="authoritative_precomputed_rng",
        cxxcore8_runner_bridge_allow_validation_rng_precompute=True,
    )

    result = run_guarded_runner_bridge(inputs, config=config, native_backend=lambda *_args, **_kwargs: {})

    assert result.status == "fallback"
    assert result.fallback_reason == "rng_payload_missing"
    assert result.writeback_applied is False


def test_cxxcore8_injected_divergence_fallback(tmp_path: Path) -> None:
    native = _native_backend(tmp_path)
    inputs, _, _, _, _, _, _, _ = _capture_fixture(n=8, width=8, height=8, ticks=5, seed=8107)
    payload = _payload(ticks=5, agents=8)
    expected = run_tick_island_python_reference(apply_validation_action_override(inputs, payload))
    config = Cxxcore8BridgeConfig(
        cxxcore8_guarded_runner_bridge_enable=True,
        cxxcore8_guarded_runner_bridge_mode="authoritative_precomputed_rng",
        cxxcore8_runner_bridge_allow_validation_rng_precompute=True,
    )

    result = run_guarded_runner_bridge(
        inputs,
        config=config,
        native_backend=native,
        rng_payload=payload,
        inject_divergence=True,
    )

    assert result.status == "fallback"
    assert result.fallback_reason == "divergence"
    assert result.parity is not None and result.parity["ok"] is False
    assert result.writeback_applied is False
    assert compare_runner_backend_states(expected, result.output, atol=1.0e-6)["ok"]


def test_cxxcore8_bridge_audit_boundaries() -> None:
    audit = bridge_audit()

    assert audit["checkpoint_boundary"] == "no checkpoint state changes required"
    assert audit["metrics_boundary"] == "no metrics definition changes required"
    assert audit["rng_boundary"].startswith("no independent C++ RNG")
    assert "TickEnvApplyMixin._apply_env_tick" in audit["insertion_point"]
    assert audit["biome_hazard_expansion"].startswith("deferred")
