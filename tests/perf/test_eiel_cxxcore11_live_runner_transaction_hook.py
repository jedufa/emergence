from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import shutil

import numpy as np
import pytest

from perf.cxxcore5_tick_island_layout_cache import compare_runner_backend_states, run_tick_island_python_reference
from perf.cxxcore11_live_runner_transaction_hook import (
    Cxxcore11LiveHookConfig,
    config_from_sim_cfg,
    live_hook_preflight_audit,
    result_payload,
    run_live_runner_transaction_hook,
    validation_rng_payload_from_actions,
)
from perf.cxxcore7_rng_precompute import (
    apply_validation_action_override,
    build_rng_payload,
    precompute_validation_action_override_ints,
)
from simulation.runner_pkg.runner_config import SimConfig
from tools.scripts.eiel_cxxcore5_tick_island_layout_cache_probe import _capture_fixture
from tools.scripts.eiel_cxxcore7_rng_precompute_runner_bridge_probe import (
    compile_cpp_kernel,
    load_rng_precompute_backend,
)


def _native_backend(tmp_path: Path):
    if shutil.which("g++") is None:
        pytest.skip("g++ unavailable")
    compile_info = compile_cpp_kernel(tmp_path)
    assert compile_info["ok"], compile_info
    return load_rng_precompute_backend(Path(str(compile_info["library_path"])))


def _observed(inputs):
    return run_tick_island_python_reference(inputs)


def test_cxxcore11_preflight_audit_keeps_python_env_step_authoritative() -> None:
    audit = live_hook_preflight_audit()
    names = {row["name"] for row in audit["rows"]}
    classifications = {row["classification"] for row in audit["rows"]}

    assert audit["stop_gate_present"] is False
    assert audit["direct_python_env_step_skip_supported"] is False
    assert "live_GridWorld_arrays" in names
    assert "checkpoint_payload_state" in names
    assert "python_side_effect_writeback" in classifications
    assert "checkpoint_owned_python_state" in classifications


def test_cxxcore11_default_off_flags_and_noop_status() -> None:
    sim_cfg = SimConfig()
    config = config_from_sim_cfg(sim_cfg)
    inputs, _, _, _, _, _, _, _ = _capture_fixture(n=8, width=8, height=8, ticks=4, seed=11001)

    result = run_live_runner_transaction_hook(inputs, observed_python=_observed(inputs), config=config, native_backend=None)

    assert sim_cfg.cxxcore11_live_runner_hook_enable is False
    assert result.status == "disabled"
    assert result.fallback_reason == "hook_disabled"
    assert result.output is None
    assert result.apply_output_to_live_scalars is False


def test_cxxcore11_invalid_config_rejected() -> None:
    inputs, _, _, _, _, _, _, _ = _capture_fixture(n=4, width=4, height=4, ticks=3, seed=11002)
    with pytest.raises(ValueError, match="unsupported CXXCORE-11 live hook mode"):
        run_live_runner_transaction_hook(
            inputs,
            observed_python=_observed(inputs),
            config=Cxxcore11LiveHookConfig(
                cxxcore11_live_runner_hook_enable=True,
                cxxcore11_live_runner_hook_mode="bad",
            ),
        )
    with pytest.raises(ValueError, match="parity_interval must be >= 1"):
        run_live_runner_transaction_hook(
            inputs,
            observed_python=_observed(inputs),
            config=Cxxcore11LiveHookConfig(
                cxxcore11_live_runner_hook_enable=True,
                cxxcore11_live_runner_hook_mode="python_split_reference_live_hook",
                cxxcore11_live_runner_hook_parity_interval=0,
            ),
        )


def test_cxxcore11_strict_native_required_missing_raises() -> None:
    inputs, _, _, _, _, _, _, _ = _capture_fixture(n=4, width=4, height=4, ticks=3, seed=11003)
    with pytest.raises(RuntimeError, match="native backend required"):
        run_live_runner_transaction_hook(
            inputs,
            observed_python=_observed(inputs),
            config=Cxxcore11LiveHookConfig(
                cxxcore11_live_runner_hook_enable=True,
                cxxcore11_live_runner_hook_mode="native_guarded_no_rng_live_hook",
                cxxcore11_live_runner_hook_strict_native_required=True,
            ),
            native_backend=None,
        )


def test_cxxcore11_python_split_live_hook_parity() -> None:
    inputs, _, _, _, _, _, _, _ = _capture_fixture(n=8, width=8, height=8, ticks=5, seed=11004)
    expected = _observed(inputs)

    result = run_live_runner_transaction_hook(
        inputs,
        observed_python=expected,
        config=Cxxcore11LiveHookConfig(
            cxxcore11_live_runner_hook_enable=True,
            cxxcore11_live_runner_hook_mode="python_split_reference_live_hook",
        ),
    )

    assert result.status == "live_hook"
    assert result.retained_native_output is False
    assert result.apply_output_to_live_scalars is False
    assert result.live_observed_parity is not None and result.live_observed_parity["ok"]
    assert compare_runner_backend_states(expected, result.output, atol=0.0)["ok"]


def test_cxxcore11_native_shadow_live_hook_parity(tmp_path: Path) -> None:
    native = _native_backend(tmp_path)
    inputs, _, _, _, _, _, _, _ = _capture_fixture(n=8, width=8, height=8, ticks=5, seed=11005)
    expected = _observed(inputs)

    result = run_live_runner_transaction_hook(
        inputs,
        observed_python=expected,
        config=Cxxcore11LiveHookConfig(
            cxxcore11_live_runner_hook_enable=True,
            cxxcore11_live_runner_hook_mode="native_shadow_live_hook",
        ),
        native_backend=native,
    )

    assert result.status == "live_hook"
    assert result.retained_native_output is False
    assert result.live_observed_parity is not None and result.live_observed_parity["ok"]
    assert compare_runner_backend_states(expected, result.output, atol=1.0e-6)["ok"]


def test_cxxcore11_native_guarded_no_rng_retains_scalar_output(tmp_path: Path) -> None:
    native = _native_backend(tmp_path)
    inputs, _, _, _, _, _, _, _ = _capture_fixture(n=8, width=8, height=8, ticks=5, seed=11006)
    expected = _observed(inputs)

    result = run_live_runner_transaction_hook(
        inputs,
        observed_python=expected,
        config=Cxxcore11LiveHookConfig(
            cxxcore11_live_runner_hook_enable=True,
            cxxcore11_live_runner_hook_mode="native_guarded_no_rng_live_hook",
            cxxcore11_live_runner_hook_parity_interval=2,
            cxxcore11_live_runner_hook_parity_policy="interval",
        ),
        native_backend=native,
    )

    assert result.status == "live_hook"
    assert result.retained_native_output is True
    assert result.apply_output_to_live_scalars is True
    assert result.live_observed_parity is not None and result.live_observed_parity["ok"]
    assert compare_runner_backend_states(expected, result.output, atol=1.0e-6)["ok"]


def test_cxxcore11_native_guarded_validation_rng_parity(tmp_path: Path) -> None:
    native = _native_backend(tmp_path)
    inputs, _, _, _, _, _, _, _ = _capture_fixture(n=8, width=8, height=8, ticks=5, seed=11007)
    payload = build_rng_payload(
        validation_action_override_ints=precompute_validation_action_override_ints(
            rng=np.random.default_rng(11007),
            ticks=5,
            agents=8,
        )
    )
    observed = run_tick_island_python_reference(apply_validation_action_override(inputs, payload))

    result = run_live_runner_transaction_hook(
        inputs,
        observed_python=observed,
        config=Cxxcore11LiveHookConfig(
            cxxcore11_live_runner_hook_enable=True,
            cxxcore11_live_runner_hook_mode="native_guarded_validation_rng_live_hook",
            cxxcore11_live_runner_hook_rng_policy="validation_action_override",
            cxxcore11_live_runner_hook_parity_policy="interval",
            cxxcore11_live_runner_hook_parity_interval=2,
        ),
        native_backend=native,
        rng_payload=validation_rng_payload_from_actions(payload.validation_action_override_ints.values),
    )

    assert result.status == "live_hook"
    assert result.retained_native_output is True
    assert result.live_observed_parity is not None and result.live_observed_parity["ok"]


def test_cxxcore11_missing_native_fallback() -> None:
    inputs, _, _, _, _, _, _, _ = _capture_fixture(n=8, width=8, height=8, ticks=5, seed=11008)
    expected = _observed(inputs)

    result = run_live_runner_transaction_hook(
        inputs,
        observed_python=expected,
        config=Cxxcore11LiveHookConfig(
            cxxcore11_live_runner_hook_enable=True,
            cxxcore11_live_runner_hook_mode="native_guarded_no_rng_live_hook",
        ),
        native_backend=None,
    )

    assert result.status == "fallback"
    assert result.fallback_reason == "native_unavailable"
    assert result.retained_native_output is False
    assert result.python_state_restored is True
    assert compare_runner_backend_states(expected, result.output, atol=0.0)["ok"]


def test_cxxcore11_malformed_rng_payload_fallback() -> None:
    inputs, _, _, _, _, _, _, _ = _capture_fixture(n=8, width=8, height=8, ticks=5, seed=11009)
    payload = validation_rng_payload_from_actions(np.zeros((5, 8), dtype=np.int32))
    bad_array = replace(payload.validation_action_override_ints, values=np.zeros((1, 1), dtype=np.int32))
    bad_payload = replace(payload, validation_action_override_ints=bad_array)
    expected = _observed(inputs)

    result = run_live_runner_transaction_hook(
        inputs,
        observed_python=expected,
        config=Cxxcore11LiveHookConfig(
            cxxcore11_live_runner_hook_enable=True,
            cxxcore11_live_runner_hook_mode="native_guarded_validation_rng_live_hook",
            cxxcore11_live_runner_hook_fail_closed=False,
            cxxcore11_live_runner_hook_rng_policy="validation_action_override",
        ),
        native_backend=lambda *_a, **_k: {},
        rng_payload=bad_payload,
    )

    assert result.status == "fallback"
    assert result.fallback_reason == "rng_payload_invalid"
    assert result.python_state_restored is True
    assert compare_runner_backend_states(expected, result.output, atol=0.0)["ok"]


def test_cxxcore11_injected_divergence_fallback(tmp_path: Path) -> None:
    native = _native_backend(tmp_path)
    inputs, _, _, _, _, _, _, _ = _capture_fixture(n=8, width=8, height=8, ticks=5, seed=11010)
    expected = _observed(inputs)

    result = run_live_runner_transaction_hook(
        inputs,
        observed_python=expected,
        config=Cxxcore11LiveHookConfig(
            cxxcore11_live_runner_hook_enable=True,
            cxxcore11_live_runner_hook_mode="native_guarded_no_rng_live_hook",
        ),
        native_backend=native,
        inject_divergence=True,
    )

    assert result.status == "fallback"
    assert result.fallback_reason in {"divergence", "live_observed_divergence"}
    assert result.first_divergence is not None
    assert result.python_state_restored is True
    assert compare_runner_backend_states(expected, result.output, atol=0.0)["ok"]


def test_cxxcore11_exception_fallback_when_not_fail_closed() -> None:
    inputs, _, _, _, _, _, _, _ = _capture_fixture(n=8, width=8, height=8, ticks=5, seed=11011)
    expected = _observed(inputs)

    def _raising_native(*_args, **_kwargs):
        raise RuntimeError("injected native failure")

    result = run_live_runner_transaction_hook(
        inputs,
        observed_python=expected,
        config=Cxxcore11LiveHookConfig(
            cxxcore11_live_runner_hook_enable=True,
            cxxcore11_live_runner_hook_mode="native_guarded_no_rng_live_hook",
            cxxcore11_live_runner_hook_fail_closed=False,
        ),
        native_backend=_raising_native,
    )

    assert result.status == "fallback"
    assert result.fallback_reason == "exception_fallback"
    assert compare_runner_backend_states(expected, result.output, atol=0.0)["ok"]


def test_cxxcore11_no_checkpoint_or_metrics_schema_fields_added() -> None:
    inputs, _, _, _, _, _, _, _ = _capture_fixture(n=4, width=4, height=4, ticks=3, seed=11012)

    result = run_live_runner_transaction_hook(
        inputs,
        observed_python=_observed(inputs),
        config=Cxxcore11LiveHookConfig(
            cxxcore11_live_runner_hook_enable=True,
            cxxcore11_live_runner_hook_mode="python_split_reference_live_hook",
        ),
    )
    payload = result_payload(result)

    assert payload["diagnostics"]["metrics_schema_changed"] is False
    assert payload["diagnostics"]["checkpoint_schema_changed"] is False
    assert payload["diagnostics"]["python_env_step_authoritative"] is True
    assert payload["diagnostics"]["native_skips_python_env_step"] is False
