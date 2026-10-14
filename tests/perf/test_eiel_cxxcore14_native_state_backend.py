from __future__ import annotations

import numpy as np

from perf.cxxcore13_python_state_backend import (
    Cxxcore13StateBackendConfig,
    as_backend_input,
    compare_transaction_to_reference,
    live_state_digest,
    live_state_from_backend_input,
    native_output_to_transaction,
    result_payload,
    run_state_backend,
)
from perf.cxxcore5_tick_island_layout_cache import run_tick_island_python_reference
from simulation.runner_pkg.runner_config import SimConfig
from tools.scripts.eiel_cxxcore5_tick_island_layout_cache_probe import _capture_fixture


def _inputs(seed: int = 14001, *, n: int = 8, width: int = 8, height: int = 8, ticks: int = 5):
    inputs, _, _, _, _, _, _, _ = _capture_fixture(n=n, width=width, height=height, ticks=ticks, seed=seed)
    return inputs


def _config(mode: str, **kwargs):
    return Cxxcore13StateBackendConfig(
        cxxcore13_state_backend_enable=True,
        cxxcore13_state_backend_mode=mode,
        **kwargs,
    )


def _native_reference(inputs):
    return run_tick_island_python_reference(inputs)


def _native_malformed(_inputs):
    return {"new_energy": np.zeros((1,), dtype=np.float32)}


def test_cxxcore14_native_transaction_adapter_parity_on_tiny_fixture() -> None:
    inputs = _inputs(14002)
    backend_input = as_backend_input(inputs)

    tx = native_output_to_transaction(backend_input, _native_reference(inputs))
    parity = compare_transaction_to_reference(backend_input, tx)

    assert parity["ok"]
    assert tx.diagnostics["native_output_adapter"] is True
    assert tx.food_gained.shape == backend_input.initial_energy.shape
    assert tx.agent_grid.shape == backend_input.agent_grid.shape


def test_cxxcore14_native_authoritative_no_rng_materializes_and_skips_env_step() -> None:
    backend_input = as_backend_input(_inputs(14003))
    live_state = live_state_from_backend_input(backend_input)

    result = run_state_backend(
        backend_input.tick_inputs,
        config=_config("native_backend_authoritative_no_rng"),
        live_state=live_state,
        native_backend=_native_reference,
    )
    payload = result_payload(result)

    assert result.status == "authoritative_parity"
    assert result.parity is not None and result.parity["ok"]
    assert result.python_env_step_skipped is True
    assert result.native_available is True
    assert payload["diagnostics"]["checkpoint_schema_changed"] is False
    assert payload["diagnostics"]["metrics_schema_changed"] is False
    assert payload["diagnostics"]["native_rng_added"] is False
    assert payload["diagnostics"]["native_direct_live_object_mutation"] is False


def test_cxxcore14_native_no_compare_authoritative_lower_bound_mode() -> None:
    backend_input = as_backend_input(_inputs(14004))
    live_state = live_state_from_backend_input(backend_input)

    result = run_state_backend(
        backend_input.tick_inputs,
        config=_config("native_backend_authoritative_no_rng_no_compare"),
        live_state=live_state,
        native_backend=_native_reference,
    )

    assert result.status == "authoritative_no_compare"
    assert result.python_env_step_skipped is True
    assert result.parity is not None and result.parity["comparison_skipped"] is True


def test_cxxcore14_shadow_native_calls_native_but_does_not_skip_env_step() -> None:
    result = run_state_backend(_inputs(14005), config=_config("native_backend_shadow_no_rng"), native_backend=_native_reference)

    assert result.status == "reference_parity"
    assert result.parity is not None and result.parity["ok"]
    assert result.python_env_step_skipped is False
    assert result.native_available is True


def test_cxxcore14_flags_off_unchanged() -> None:
    sim_cfg = SimConfig()
    assert sim_cfg.cxxcore13_state_backend_enable is False
    result = run_state_backend(_inputs(14006), config=Cxxcore13StateBackendConfig())
    assert result.status == "disabled"
    assert result.fallback_reason == "pilot_disabled"


def test_cxxcore14_native_unavailable_fallback_safe() -> None:
    result = run_state_backend(_inputs(14007), config=_config("native_backend_authoritative_no_rng"), native_backend=None)

    assert result.status == "fallback"
    assert result.fallback_reason == "native_unavailable"
    assert result.python_env_step_skipped is False


def test_cxxcore14_unsupported_feature_fallback_before_mutation() -> None:
    backend_input = as_backend_input(_inputs(14008))
    live_state = live_state_from_backend_input(backend_input)
    before = live_state_digest(live_state)

    result = run_state_backend(
        backend_input.tick_inputs,
        config=_config("native_backend_authoritative_no_rng"),
        live_state=live_state,
        native_backend=_native_reference,
        unsupported_feature_reasons=["tier3_transition_rng"],
    )

    assert result.status == "fallback"
    assert result.fallback_reason == "unsupported_features"
    assert result.python_state_restored is True
    assert live_state_digest(live_state) == before


def test_cxxcore14_injected_native_divergence_localizes_and_restores() -> None:
    backend_input = as_backend_input(_inputs(14009))
    live_state = live_state_from_backend_input(backend_input)
    before = live_state_digest(live_state)

    result = run_state_backend(
        backend_input.tick_inputs,
        config=_config("native_backend_authoritative_no_rng"),
        live_state=live_state,
        native_backend=_native_reference,
        inject_divergence=True,
    )

    assert result.status == "fallback"
    assert result.fallback_reason == "parity_divergence"
    assert result.first_divergence is not None
    assert result.first_divergence["field"] == "new_energy"
    assert result.python_state_restored is True
    assert live_state_digest(live_state) == before


def test_cxxcore14_native_exception_restores_when_not_fail_closed() -> None:
    backend_input = as_backend_input(_inputs(14010))
    live_state = live_state_from_backend_input(backend_input)
    before = live_state_digest(live_state)

    result = run_state_backend(
        backend_input.tick_inputs,
        config=_config("native_backend_authoritative_no_rng", cxxcore13_state_backend_fail_closed=False),
        live_state=live_state,
        native_backend=_native_reference,
        raise_exception=True,
    )

    assert result.status == "fallback"
    assert result.fallback_reason == "exception_fallback"
    assert result.python_state_restored is True
    assert live_state_digest(live_state) == before


def test_cxxcore14_malformed_native_output_restores_when_not_fail_closed() -> None:
    backend_input = as_backend_input(_inputs(14011))
    live_state = live_state_from_backend_input(backend_input)
    before = live_state_digest(live_state)

    result = run_state_backend(
        backend_input.tick_inputs,
        config=_config("native_backend_authoritative_no_rng", cxxcore13_state_backend_fail_closed=False),
        live_state=live_state,
        native_backend=_native_malformed,
    )

    assert result.status == "fallback"
    assert result.fallback_reason == "exception_fallback"
    assert result.python_state_restored is True
    assert live_state_digest(live_state) == before


def test_cxxcore14_checkpoint_metrics_and_rng_schema_unchanged() -> None:
    result = run_state_backend(_inputs(14012), config=_config("native_backend_authoritative_no_rng"), native_backend=_native_reference)
    payload = result_payload(result)

    assert payload["diagnostics"]["checkpoint_schema_changed"] is False
    assert payload["diagnostics"]["metrics_schema_changed"] is False
    assert payload["diagnostics"]["native_rng_added"] is False
    assert payload["diagnostics"]["native_direct_live_object_mutation"] is False
