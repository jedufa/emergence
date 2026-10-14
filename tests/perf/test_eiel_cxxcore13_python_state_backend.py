from __future__ import annotations

from perf.cxxcore13_python_state_backend import (
    DECISION,
    Cxxcore13StateBackendConfig,
    as_backend_input,
    capture_restore_token,
    compare_transaction_to_reference,
    config_from_sim_cfg,
    live_state_digest,
    live_state_from_backend_input,
    materialize_transaction_to_live_state,
    restore_from_token,
    result_payload,
    run_python_backend_transaction,
    run_state_backend,
    state_backend_audit_payload,
)
from simulation.runner_pkg.runner_config import SimConfig
from tools.scripts.eiel_cxxcore5_tick_island_layout_cache_probe import _capture_fixture


def _inputs(seed: int = 13001):
    inputs, _, _, _, _, _, _, _ = _capture_fixture(n=8, width=8, height=8, ticks=5, seed=seed)
    return inputs


def _config(mode: str, **kwargs):
    return Cxxcore13StateBackendConfig(
        cxxcore13_state_backend_enable=True,
        cxxcore13_state_backend_mode=mode,
        **kwargs,
    )


def test_cxxcore13_default_flags_are_off_and_import_without_native() -> None:
    sim_cfg = SimConfig()
    cfg = config_from_sim_cfg(sim_cfg)

    assert sim_cfg.cxxcore13_state_backend_enable is False
    assert cfg.cxxcore13_state_backend_enable is False
    assert cfg.cxxcore13_state_backend_mode == "off"
    assert cfg.cxxcore13_state_backend_rng_policy == "none"


def test_cxxcore13_audit_maps_required_state_backend_surfaces() -> None:
    audit = state_backend_audit_payload()
    names = {row["name"] for row in audit["rows"]}
    classes = set(audit["classifications"])

    assert audit["decision"] == DECISION
    assert audit["minimal_authoritative_skip_supported"] is True
    assert "food_gained" in names
    assert "agent_grid_occupancy" in names
    assert "views_dirty_flag" in names
    assert "CostChargeSurface_layout_rows" in names
    assert "phenotype_expression_path" in names
    assert "perception_expression_path" in names
    assert "env_hotpath_diagnostics_config" in names
    assert "owned_in_minimal_backend" in classes
    assert "deferred_feature_gated" in classes
    assert "requires_rng_precompute" in classes
    assert "requires_checkpoint_schema_preflight" in classes
    assert "requires_metrics_schema_preflight" in classes
    assert "out_of_scope_for_cxxcore13" in classes


def test_cxxcore13_flags_off_probe_is_disabled() -> None:
    result = run_state_backend(_inputs(13002), config=Cxxcore13StateBackendConfig())

    assert result.status == "disabled"
    assert result.fallback_reason == "pilot_disabled"
    assert result.python_env_step_skipped is False


def test_cxxcore13_restore_token_roundtrips_live_state() -> None:
    backend_input = as_backend_input(_inputs(13003))
    state = live_state_from_backend_input(backend_input)
    before = live_state_digest(state)
    token = capture_restore_token(state)

    state.energy += 7.0
    state.resource_grid[:, :] = 1
    state.views_dirty = True
    assert live_state_digest(state) != before

    restore_from_token(state, token)
    assert live_state_digest(state) == before
    assert token.state_digest == before


def test_cxxcore13_python_transaction_has_complete_writeback_contract_and_parity() -> None:
    backend_input = as_backend_input(_inputs(13004))

    transaction = run_python_backend_transaction(backend_input)
    parity = compare_transaction_to_reference(backend_input, transaction)

    assert parity["ok"]
    assert transaction.food_gained.shape == backend_input.initial_energy.shape
    assert transaction.moved_mask.shape == backend_input.initial_energy.shape
    assert transaction.alive_mask.shape == backend_input.initial_energy.shape
    assert transaction.agent_grid.shape == backend_input.agent_grid.shape
    assert transaction.resource_grid.shape == backend_input.resource_grid.shape
    assert transaction.cost_total_by_agent.shape == backend_input.initial_energy.shape


def test_cxxcore13_shadow_reference_does_not_skip_python_env_step() -> None:
    result = run_state_backend(_inputs(13005), config=_config("shadow_reference"))

    assert result.status == "reference_parity"
    assert result.parity is not None and result.parity["ok"]
    assert result.python_env_step_skipped is False
    assert result.python_state_restored is True


def test_cxxcore13_authoritative_no_rng_skips_env_step_and_writes_live_state() -> None:
    backend_input = as_backend_input(_inputs(13006))
    live_state = live_state_from_backend_input(backend_input)

    result = run_state_backend(
        backend_input.tick_inputs,
        config=_config("python_backend_authoritative_no_rng"),
        live_state=live_state,
    )
    payload = result_payload(result)

    assert result.status == "authoritative_parity"
    assert result.fallback_reason is None
    assert result.parity is not None and result.parity["ok"]
    assert result.python_env_step_skipped is True
    assert payload["diagnostics"]["checkpoint_schema_changed"] is False
    assert payload["diagnostics"]["metrics_schema_changed"] is False
    assert payload["diagnostics"]["native_rng_added"] is False
    assert payload["transaction"]["views_dirty"] is True


def test_cxxcore13_materialized_state_digest_matches_transaction_output() -> None:
    backend_input = as_backend_input(_inputs(13007))
    live_state = live_state_from_backend_input(backend_input)
    tx = run_python_backend_transaction(backend_input)

    materialize_transaction_to_live_state(live_state, backend_input, tx)

    assert live_state.energy.tolist() == tx.output["new_energy"].tolist()
    assert live_state.resource_grid.tolist() == tx.resource_grid.tolist()
    assert live_state.agent_grid.tolist() == tx.agent_grid.tolist()
    assert int(live_state.stats_survival_steps[0]) == int(backend_input.actions_by_tick.shape[0])


def test_cxxcore13_unsupported_features_fallback_before_mutation() -> None:
    backend_input = as_backend_input(_inputs(13008))
    live_state = live_state_from_backend_input(backend_input)
    before = live_state_digest(live_state)

    result = run_state_backend(
        backend_input.tick_inputs,
        config=_config("python_backend_authoritative_no_rng"),
        live_state=live_state,
        unsupported_feature_reasons=["patch_world_enable"],
    )

    assert result.status == "fallback"
    assert result.fallback_reason == "unsupported_features"
    assert result.python_env_step_skipped is False
    assert result.python_state_restored is True
    assert live_state_digest(live_state) == before


def test_cxxcore13_injected_divergence_localizes_and_restores_state() -> None:
    backend_input = as_backend_input(_inputs(13009))
    live_state = live_state_from_backend_input(backend_input)
    before = live_state_digest(live_state)

    result = run_state_backend(
        backend_input.tick_inputs,
        config=_config("python_backend_authoritative_no_rng"),
        live_state=live_state,
        inject_divergence=True,
    )

    assert result.status == "fallback"
    assert result.fallback_reason == "parity_divergence"
    assert result.first_divergence is not None
    assert result.first_divergence["field"] == "new_energy"
    assert result.python_state_restored is True
    assert live_state_digest(live_state) == before


def test_cxxcore13_exception_fallback_restores_when_not_fail_closed() -> None:
    backend_input = as_backend_input(_inputs(13010))
    live_state = live_state_from_backend_input(backend_input)
    before = live_state_digest(live_state)

    result = run_state_backend(
        backend_input.tick_inputs,
        config=_config("python_backend_authoritative_no_rng", cxxcore13_state_backend_fail_closed=False),
        live_state=live_state,
        raise_exception=True,
    )

    assert result.status == "fallback"
    assert result.fallback_reason == "exception_fallback"
    assert result.python_state_restored is True
    assert live_state_digest(live_state) == before


def test_cxxcore13_native_missing_fallback_and_validation_rng_deferred() -> None:
    native_result = run_state_backend(_inputs(13011), config=_config("native_backend_shadow_no_rng"), native_backend=None)
    rng_result = run_state_backend(
        _inputs(13012),
        config=_config(
            "python_backend_authoritative_validation_rng",
            cxxcore13_state_backend_rng_policy="validation_action_override",
        ),
    )

    assert native_result.status == "fallback"
    assert native_result.fallback_reason == "native_unavailable"
    assert rng_result.status == "fallback"
    assert rng_result.fallback_reason == "validation_rng_writeback_deferred"
