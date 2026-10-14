from __future__ import annotations

from dataclasses import replace
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
from perf.cxxcore9_runner_adapter import (
    Cxxcore9AdapterConfig,
    adapter_boundary_map,
    assert_output_matches_python_reference,
    config_from_sim_cfg,
    result_payload,
    run_runner_adapter,
)
from simulation.runner_pkg.runner_config import SimConfig
from tools.scripts.eiel_cxxcore5_tick_island_layout_cache_probe import _capture_fixture
from tools.scripts.eiel_cxxcore7_rng_precompute_runner_bridge_probe import (
    compile_cpp_kernel,
    load_rng_precompute_backend,
)


def _payload(*, ticks: int, agents: int, seed: int = 9109):
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


def test_cxxcore9_default_off_flags_and_disabled_status() -> None:
    sim_cfg = SimConfig()
    config = config_from_sim_cfg(sim_cfg)
    inputs, _, _, _, _, _, _, _ = _capture_fixture(n=8, width=8, height=8, ticks=4, seed=9101)

    result = run_runner_adapter(inputs, config=config, native_backend=None)

    assert sim_cfg.cxxcore9_runner_adapter_enable is False
    assert config.cxxcore9_runner_adapter_mode == "off"
    assert result.fallback_reason == "adapter_disabled"
    assert result.attempted_native is False
    assert result.writeback_applied is False
    assert result.output is None


def test_cxxcore9_shadow_no_rng_parity(tmp_path: Path) -> None:
    native = _native_backend(tmp_path)
    inputs, _, _, _, _, _, _, _ = _capture_fixture(n=8, width=8, height=8, ticks=5, seed=9102)
    config = Cxxcore9AdapterConfig(
        cxxcore9_runner_adapter_enable=True,
        cxxcore9_runner_adapter_mode="shadow",
    )

    result = run_runner_adapter(inputs, config=config, native_backend=native)

    assert result.attempted_native is True
    assert result.retained_native_output is False
    assert result.parity_checked is True
    assert result.parity_passed is True
    assert result.writeback_applied is False


def test_cxxcore9_guarded_authoritative_no_rng_parity_and_retention(tmp_path: Path) -> None:
    native = _native_backend(tmp_path)
    inputs, _, _, _, _, _, _, _ = _capture_fixture(n=8, width=8, height=8, ticks=5, seed=9103)
    config = Cxxcore9AdapterConfig(
        cxxcore9_runner_adapter_enable=True,
        cxxcore9_runner_adapter_mode="guarded_authoritative_no_rng",
        cxxcore9_runner_adapter_parity_policy="interval",
        cxxcore9_runner_adapter_parity_interval=2,
    )

    result = run_runner_adapter(inputs, config=config, native_backend=native)

    assert result.retained_native_output is True
    assert result.parity_passed is True
    assert result.writeback_applied is True
    assert assert_output_matches_python_reference(inputs, result)["ok"]


def test_cxxcore9_validation_rng_precompute_parity(tmp_path: Path) -> None:
    native = _native_backend(tmp_path)
    inputs, _, _, _, _, _, _, _ = _capture_fixture(n=8, width=8, height=8, ticks=5, seed=9104)
    payload = _payload(ticks=5, agents=8)
    expected = run_tick_island_python_reference(apply_validation_action_override(inputs, payload))
    config = Cxxcore9AdapterConfig(
        cxxcore9_runner_adapter_enable=True,
        cxxcore9_runner_adapter_mode="guarded_authoritative_validation_rng",
        cxxcore9_runner_adapter_parity_policy="interval",
        cxxcore9_runner_adapter_parity_interval=2,
    )

    result = run_runner_adapter(inputs, config=config, native_backend=native, rng_payload=payload)

    assert result.retained_native_output is True
    assert result.parity_passed is True
    assert compare_runner_backend_states(expected, result.output, atol=1.0e-6)["ok"]


def test_cxxcore9_native_unavailable_fallback_restores_python_reference() -> None:
    inputs, _, _, _, _, _, _, _ = _capture_fixture(n=8, width=8, height=8, ticks=5, seed=9105)
    config = Cxxcore9AdapterConfig(
        cxxcore9_runner_adapter_enable=True,
        cxxcore9_runner_adapter_mode="guarded_authoritative_no_rng",
    )

    result = run_runner_adapter(inputs, config=config, native_backend=None)

    assert result.fallback_occurred is True
    assert result.fallback_reason == "native_unavailable"
    assert result.python_state_restored is True
    assert result.writeback_applied is False
    assert assert_output_matches_python_reference(inputs, result)["ok"]


def test_cxxcore9_injected_divergence_fallback_materialized_state(tmp_path: Path) -> None:
    native = _native_backend(tmp_path)
    inputs, _, _, _, _, _, _, _ = _capture_fixture(n=8, width=8, height=8, ticks=5, seed=9106)
    payload = _payload(ticks=5, agents=8)
    expected = run_tick_island_python_reference(apply_validation_action_override(inputs, payload))
    config = Cxxcore9AdapterConfig(
        cxxcore9_runner_adapter_enable=True,
        cxxcore9_runner_adapter_mode="guarded_authoritative_validation_rng",
    )

    result = run_runner_adapter(
        inputs,
        config=config,
        native_backend=native,
        rng_payload=payload,
        inject_divergence=True,
    )

    assert result.fallback_occurred is True
    assert result.fallback_reason == "divergence"
    assert result.parity_checked is True
    assert result.parity_passed is False
    assert result.writeback_applied is False
    assert compare_runner_backend_states(expected, result.output, atol=1.0e-6)["ok"]


def test_cxxcore9_malformed_rng_payload_fallback() -> None:
    inputs, _, _, _, _, _, _, _ = _capture_fixture(n=8, width=8, height=8, ticks=5, seed=9107)
    payload = _payload(ticks=5, agents=8)
    bad_array = replace(payload.validation_action_override_ints, values=np.zeros((1, 1), dtype=np.int32))
    bad_payload = replace(payload, validation_action_override_ints=bad_array)
    config = Cxxcore9AdapterConfig(
        cxxcore9_runner_adapter_enable=True,
        cxxcore9_runner_adapter_mode="guarded_authoritative_validation_rng",
    )

    result = run_runner_adapter(inputs, config=config, native_backend=lambda *_args, **_kwargs: {}, rng_payload=bad_payload)

    assert result.fallback_occurred is True
    assert result.fallback_reason == "rng_payload_invalid"
    assert result.writeback_applied is False
    assert assert_output_matches_python_reference(inputs, result)["ok"]


def test_cxxcore9_exception_fallback_and_repeated_ticks_deterministic() -> None:
    inputs, _, _, _, _, _, _, _ = _capture_fixture(n=8, width=8, height=8, ticks=5, seed=9108)
    config = Cxxcore9AdapterConfig(
        cxxcore9_runner_adapter_enable=True,
        cxxcore9_runner_adapter_mode="guarded_authoritative_no_rng",
        cxxcore9_runner_adapter_fail_closed=False,
    )

    def _raising_native(*_args, **_kwargs):
        raise RuntimeError("injected native failure")

    first = run_runner_adapter(inputs, config=config, native_backend=_raising_native)
    second = run_runner_adapter(inputs, config=config, native_backend=_raising_native)

    assert first.fallback_reason == "exception_fallback"
    assert first.python_state_restored is True
    assert compare_runner_backend_states(first.output, second.output, atol=0.0)["ok"]
    assert assert_output_matches_python_reference(inputs, first)["ok"]


def test_cxxcore9_boundary_map_and_payload_contract() -> None:
    inputs, _, _, _, _, _, _, _ = _capture_fixture(n=4, width=4, height=4, ticks=3, seed=9110)
    config = Cxxcore9AdapterConfig()
    result = run_runner_adapter(inputs, config=config, native_backend=None)
    payload = result_payload(result)
    boundary = adapter_boundary_map()

    assert payload["schema_version"] == "cxxcore9_runner_adapter_v1"
    assert payload["restore_token"]["pre_state_digest"]
    assert boundary["restore_fields"]
    assert boundary["phases"][-1]["phase"] == "death_stat_side_effect_update"
