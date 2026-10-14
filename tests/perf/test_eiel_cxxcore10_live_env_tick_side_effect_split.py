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
from perf.cxxcore10_live_env_tick_side_effect_split import (
    Cxxcore10SplitConfig,
    as_env_tick_inputs,
    config_from_sim_cfg,
    materialize_python_reference,
    result_payload,
    run_env_tick_split_transaction,
    side_effect_taxonomy_payload,
)
from simulation.runner_pkg.runner_config import SimConfig
from tools.scripts.eiel_cxxcore5_tick_island_layout_cache_probe import _capture_fixture
from tools.scripts.eiel_cxxcore7_rng_precompute_runner_bridge_probe import (
    compile_cpp_kernel,
    load_rng_precompute_backend,
)


def _payload(*, ticks: int, agents: int, seed: int = 10109):
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


def test_cxxcore10_side_effect_taxonomy_stable_and_complete() -> None:
    first = side_effect_taxonomy_payload()
    second = side_effect_taxonomy_payload()
    names = {row["name"] for row in first["rows"]}
    classifications = {row["classification"] for row in first["rows"]}

    assert first == second
    assert "Agent.energy_health_injury_scar_writes" in names
    assert "CostChargeSurface_layout_cache" in names
    assert "checkpoint_owned_python_state" in classifications
    assert first["metrics_schema_changed"] is False
    assert first["checkpoint_schema_changed"] is False


def test_cxxcore10_default_off_flags_and_noop_status() -> None:
    sim_cfg = SimConfig()
    config = config_from_sim_cfg(sim_cfg)
    inputs, _, _, _, _, _, _, _ = _capture_fixture(n=8, width=8, height=8, ticks=4, seed=10101)

    result = run_env_tick_split_transaction(inputs, config=config, native_backend=None)

    assert sim_cfg.cxxcore10_env_tick_split_enable is False
    assert result.status == "disabled"
    assert result.fallback_reason == "split_disabled"
    assert result.output is None
    assert result.writeback_applied is False


def test_cxxcore10_python_split_reference_parity_vs_direct_reference() -> None:
    inputs, _, _, _, _, _, _, _ = _capture_fixture(n=8, width=8, height=8, ticks=5, seed=10102)
    config = Cxxcore10SplitConfig(
        cxxcore10_env_tick_split_enable=True,
        cxxcore10_env_tick_split_mode="python_split_reference",
    )
    expected = run_tick_island_python_reference(inputs)

    result = run_env_tick_split_transaction(inputs, config=config, native_backend=None)

    assert result.status == "python_split_reference"
    assert result.writeback_applied is True
    assert result.parity is not None and result.parity["ok"]
    assert compare_runner_backend_states(expected, result.output, atol=0.0)["ok"]
    payload = result_payload(result)
    assert payload["diagnostics"]["writeback"]["live_python_objects_mutated"] is False


def test_cxxcore10_materialized_reference_is_pure_and_digest_stable() -> None:
    inputs, _, _, _, _, _, _, _ = _capture_fixture(n=8, width=8, height=8, ticks=5, seed=10103)
    env_inputs = as_env_tick_inputs(inputs)

    first = materialize_python_reference(env_inputs)
    second = materialize_python_reference(env_inputs)

    assert first.digest == second.digest
    assert compare_runner_backend_states(first.fields, second.fields, atol=0.0)["ok"]


def test_cxxcore10_native_shadow_transaction_no_rng_parity(tmp_path: Path) -> None:
    native = _native_backend(tmp_path)
    inputs, _, _, _, _, _, _, _ = _capture_fixture(n=8, width=8, height=8, ticks=5, seed=10104)
    config = Cxxcore10SplitConfig(
        cxxcore10_env_tick_split_enable=True,
        cxxcore10_env_tick_split_mode="native_shadow_transaction",
    )
    expected = run_tick_island_python_reference(inputs)

    result = run_env_tick_split_transaction(inputs, config=config, native_backend=native)

    assert result.status == "shadow"
    assert result.retained_native_output is False
    assert result.writeback_applied is False
    assert result.parity is not None and result.parity["ok"]
    assert compare_runner_backend_states(expected, result.output, atol=0.0)["ok"]


def test_cxxcore10_native_guarded_transaction_no_rng_parity(tmp_path: Path) -> None:
    native = _native_backend(tmp_path)
    inputs, _, _, _, _, _, _, _ = _capture_fixture(n=8, width=8, height=8, ticks=5, seed=10105)
    config = Cxxcore10SplitConfig(
        cxxcore10_env_tick_split_enable=True,
        cxxcore10_env_tick_split_mode="native_guarded_transaction_no_rng",
        cxxcore10_env_tick_split_parity_interval=2,
        cxxcore10_env_tick_split_parity_policy="interval",
    )
    expected = run_tick_island_python_reference(inputs)

    result = run_env_tick_split_transaction(inputs, config=config, native_backend=native)

    assert result.status == "guarded_transaction"
    assert result.retained_native_output is True
    assert result.writeback_applied is True
    assert result.parity is not None and result.parity["ok"]
    assert compare_runner_backend_states(expected, result.output, atol=1.0e-6)["ok"]


def test_cxxcore10_native_guarded_validation_rng_parity(tmp_path: Path) -> None:
    native = _native_backend(tmp_path)
    inputs, _, _, _, _, _, _, _ = _capture_fixture(n=8, width=8, height=8, ticks=5, seed=10106)
    payload = _payload(ticks=5, agents=8)
    config = Cxxcore10SplitConfig(
        cxxcore10_env_tick_split_enable=True,
        cxxcore10_env_tick_split_mode="native_guarded_transaction_validation_rng",
        cxxcore10_env_tick_split_parity_interval=2,
        cxxcore10_env_tick_split_parity_policy="interval",
    )
    expected = run_tick_island_python_reference(apply_validation_action_override(inputs, payload))

    result = run_env_tick_split_transaction(inputs, config=config, native_backend=native, rng_payload=payload)

    assert result.status == "guarded_transaction"
    assert result.retained_native_output is True
    assert result.parity is not None and result.parity["ok"]
    assert compare_runner_backend_states(expected, result.output, atol=1.0e-6)["ok"]


def test_cxxcore10_malformed_rng_payload_fallback() -> None:
    inputs, _, _, _, _, _, _, _ = _capture_fixture(n=8, width=8, height=8, ticks=5, seed=10107)
    payload = _payload(ticks=5, agents=8)
    bad_array = replace(payload.validation_action_override_ints, values=np.zeros((1, 1), dtype=np.int32))
    bad_payload = replace(payload, validation_action_override_ints=bad_array)
    config = Cxxcore10SplitConfig(
        cxxcore10_env_tick_split_enable=True,
        cxxcore10_env_tick_split_mode="native_guarded_transaction_validation_rng",
    )
    expected = run_tick_island_python_reference(inputs)

    result = run_env_tick_split_transaction(inputs, config=config, native_backend=lambda *_a, **_k: {}, rng_payload=bad_payload)

    assert result.status == "fallback"
    assert result.fallback_reason == "rng_payload_invalid"
    assert result.python_state_restored is True
    assert compare_runner_backend_states(expected, result.output, atol=0.0)["ok"]


def test_cxxcore10_missing_native_fallback() -> None:
    inputs, _, _, _, _, _, _, _ = _capture_fixture(n=8, width=8, height=8, ticks=5, seed=10108)
    config = Cxxcore10SplitConfig(
        cxxcore10_env_tick_split_enable=True,
        cxxcore10_env_tick_split_mode="native_guarded_transaction_no_rng",
    )
    expected = run_tick_island_python_reference(inputs)

    result = run_env_tick_split_transaction(inputs, config=config, native_backend=None)

    assert result.status == "fallback"
    assert result.fallback_reason == "native_unavailable"
    assert result.retained_native_output is False
    assert result.writeback_applied is False
    assert compare_runner_backend_states(expected, result.output, atol=0.0)["ok"]


def test_cxxcore10_injected_divergence_fallback_materializes_python_reference(tmp_path: Path) -> None:
    native = _native_backend(tmp_path)
    inputs, _, _, _, _, _, _, _ = _capture_fixture(n=8, width=8, height=8, ticks=5, seed=10109)
    config = Cxxcore10SplitConfig(
        cxxcore10_env_tick_split_enable=True,
        cxxcore10_env_tick_split_mode="native_guarded_transaction_no_rng",
    )
    expected = run_tick_island_python_reference(inputs)

    result = run_env_tick_split_transaction(inputs, config=config, native_backend=native, inject_divergence=True)

    assert result.status == "fallback"
    assert result.fallback_reason == "divergence"
    assert result.parity is not None and result.parity["ok"] is False
    assert result.first_divergence is not None
    assert result.python_state_restored is True
    assert result.writeback_applied is False
    assert compare_runner_backend_states(expected, result.output, atol=0.0)["ok"]


def test_cxxcore10_exception_fallback_when_not_fail_closed() -> None:
    inputs, _, _, _, _, _, _, _ = _capture_fixture(n=8, width=8, height=8, ticks=5, seed=10110)
    config = Cxxcore10SplitConfig(
        cxxcore10_env_tick_split_enable=True,
        cxxcore10_env_tick_split_mode="native_guarded_transaction_no_rng",
        cxxcore10_env_tick_split_fail_closed=False,
    )
    expected = run_tick_island_python_reference(inputs)

    def _raising_native(*_args, **_kwargs):
        raise RuntimeError("injected native failure")

    result = run_env_tick_split_transaction(inputs, config=config, native_backend=_raising_native)

    assert result.status == "fallback"
    assert result.fallback_reason == "exception_fallback"
    assert compare_runner_backend_states(expected, result.output, atol=0.0)["ok"]


def test_cxxcore10_no_checkpoint_or_metrics_schema_fields_added() -> None:
    inputs, _, _, _, _, _, _, _ = _capture_fixture(n=4, width=4, height=4, ticks=3, seed=10111)
    config = Cxxcore10SplitConfig(
        cxxcore10_env_tick_split_enable=True,
        cxxcore10_env_tick_split_mode="python_split_reference",
    )

    result = run_env_tick_split_transaction(inputs, config=config, native_backend=None)
    payload = result_payload(result)

    assert payload["diagnostics"]["metrics_schema_changed"] is False
    assert payload["diagnostics"]["checkpoint_schema_changed"] is False
    assert payload["diagnostics"]["side_effect_plan"]["metrics_schema_changed"] is False
    assert payload["diagnostics"]["side_effect_plan"]["checkpoint_schema_changed"] is False
