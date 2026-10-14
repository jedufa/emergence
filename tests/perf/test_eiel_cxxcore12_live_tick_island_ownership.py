from __future__ import annotations

from pathlib import Path
import shutil

import pytest

from perf.cxxcore12_live_tick_island_ownership import (
    DECISION,
    Cxxcore12LiveTickIslandConfig,
    config_from_sim_cfg,
    feasibility_payload,
    ownership_audit_payload,
    result_payload,
    run_cxxcore12_probe,
    state_backend_requirement_payload,
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


def test_cxxcore12_default_flags_are_off_and_import_without_native() -> None:
    sim_cfg = SimConfig()
    cfg = config_from_sim_cfg(sim_cfg)

    assert sim_cfg.cxxcore12_live_tick_island_enable is False
    assert cfg.cxxcore12_live_tick_island_enable is False
    assert cfg.cxxcore12_live_tick_island_mode == "off"
    assert cfg.cxxcore12_live_tick_island_rng_policy == "none"


def test_cxxcore12_ownership_audit_has_required_surfaces_and_classifications() -> None:
    audit = ownership_audit_payload()
    names = {row["name"] for row in audit["rows"]}
    classes = {row["classification"] for row in audit["rows"]}

    assert audit["decision"] == DECISION
    assert audit["python_env_step_skip_safe_now"] is False
    assert audit["native_authoritative_live_tick_island_supported_now"] is False
    assert "food_gained" in names
    assert "agent_grid_occupancy" in names
    assert "tier3_last_step_stats" in names
    assert "checkpoint_payload_state" in names
    assert "native_transaction_owned_now" in classes
    assert "python_writeback_derived_now" in classes
    assert "python_authoritative_retained" in classes
    assert "requires_state_backend_before_live_ownership" in classes
    assert "requires_rng_preflight_before_live_ownership" in classes
    assert "requires_checkpoint_metrics_preflight_before_live_ownership" in classes
    assert "out_of_scope_for_cxxcore12" in classes


def test_cxxcore12_feasibility_decision_is_state_backend_required() -> None:
    feasibility = feasibility_payload()
    requirement = state_backend_requirement_payload()

    assert feasibility["decision"] == "CXXCORE12-B-STATE-BACKEND-REQUIRED"
    assert feasibility["path_b_required"] is True
    assert feasibility["path_a_conditions"]["python_gridworld_step_can_be_skipped"] is False
    assert "food_gained" in requirement["minimum_fields"]
    assert "agent_grid" in requirement["minimum_fields"]
    assert "patch_food_amount" in requirement["optional_but_required_when_feature_enabled"]
    assert "tier3_last_step_stats" in requirement["optional_but_required_when_feature_enabled"]


def test_cxxcore12_flags_off_probe_is_disabled() -> None:
    inputs, _, _, _, _, _, _, _ = _capture_fixture(n=8, width=8, height=8, ticks=4, seed=12001)

    result = run_cxxcore12_probe(inputs, config=Cxxcore12LiveTickIslandConfig(), native_backend=None)

    assert result.status == "disabled"
    assert result.fallback_reason == "pilot_disabled"
    assert result.python_env_step_skipped is False
    assert result.native_output_retained is False


def test_cxxcore12_python_split_reference_parity_passes() -> None:
    inputs, _, _, _, _, _, _, _ = _capture_fixture(n=8, width=8, height=8, ticks=5, seed=12002)

    result = run_cxxcore12_probe(
        inputs,
        config=Cxxcore12LiveTickIslandConfig(
            cxxcore12_live_tick_island_enable=True,
            cxxcore12_live_tick_island_mode="python_split_reference",
        ),
        native_backend=None,
    )

    assert result.status == "reference_parity"
    assert result.parity is not None and result.parity["ok"]
    assert result.python_env_step_skipped is False
    assert result.native_output_retained is False


def test_cxxcore12_native_shadow_reference_parity_passes(tmp_path: Path) -> None:
    native = _native_backend(tmp_path)
    inputs, _, _, _, _, _, _, _ = _capture_fixture(n=8, width=8, height=8, ticks=5, seed=12003)

    result = run_cxxcore12_probe(
        inputs,
        config=Cxxcore12LiveTickIslandConfig(
            cxxcore12_live_tick_island_enable=True,
            cxxcore12_live_tick_island_mode="native_shadow_reference",
        ),
        native_backend=native,
    )

    assert result.status == "reference_parity"
    assert result.parity is not None and result.parity["ok"]
    assert result.native_available is True
    assert result.native_output_retained is False
    assert result.python_env_step_skipped is False


def test_cxxcore12_missing_native_fallback_for_shadow_reference() -> None:
    inputs, _, _, _, _, _, _, _ = _capture_fixture(n=8, width=8, height=8, ticks=5, seed=12004)

    result = run_cxxcore12_probe(
        inputs,
        config=Cxxcore12LiveTickIslandConfig(
            cxxcore12_live_tick_island_enable=True,
            cxxcore12_live_tick_island_mode="native_shadow_reference",
        ),
        native_backend=None,
    )

    assert result.status == "fallback"
    assert result.fallback_reason == "native_unavailable"
    assert result.python_state_restored is True
    assert result.python_env_step_skipped is False


def test_cxxcore12_authoritative_modes_block_before_fake_timing() -> None:
    inputs, _, _, _, _, _, _, _ = _capture_fixture(n=8, width=8, height=8, ticks=5, seed=12005)

    result = run_cxxcore12_probe(
        inputs,
        config=Cxxcore12LiveTickIslandConfig(
            cxxcore12_live_tick_island_enable=True,
            cxxcore12_live_tick_island_mode="native_authoritative_no_rng_live_tick_island",
        ),
        native_backend=None,
    )
    payload = result_payload(result)

    assert result.status == "blocked"
    assert result.fallback_reason == "state_backend_required"
    assert result.python_env_step_skipped is False
    assert result.native_output_retained is False
    assert payload["diagnostics"]["authoritative_arm_not_timed"] is True


def test_cxxcore12_validation_rng_authoritative_mode_blocks_without_cpp_rng() -> None:
    inputs, _, _, _, _, _, _, _ = _capture_fixture(n=8, width=8, height=8, ticks=5, seed=12006)

    result = run_cxxcore12_probe(
        inputs,
        config=Cxxcore12LiveTickIslandConfig(
            cxxcore12_live_tick_island_enable=True,
            cxxcore12_live_tick_island_mode="native_authoritative_validation_rng_live_tick_island",
            cxxcore12_live_tick_island_rng_policy="validation_action_override",
        ),
        native_backend=lambda *_a, **_k: {},
    )

    assert result.status == "blocked"
    assert result.fallback_reason == "state_backend_required"
    assert result.diagnostics["native_rng_added"] is False
    assert result.python_env_step_skipped is False


def test_cxxcore12_injected_divergence_localizes_first_divergence(tmp_path: Path) -> None:
    native = _native_backend(tmp_path)
    inputs, _, _, _, _, _, _, _ = _capture_fixture(n=8, width=8, height=8, ticks=5, seed=12007)

    result = run_cxxcore12_probe(
        inputs,
        config=Cxxcore12LiveTickIslandConfig(
            cxxcore12_live_tick_island_enable=True,
            cxxcore12_live_tick_island_mode="native_shadow_reference",
        ),
        native_backend=native,
        inject_divergence=True,
    )

    assert result.status == "fallback"
    assert result.first_divergence is not None
    assert result.first_divergence["field"] == "new_energy"
    assert result.python_env_step_skipped is False


def test_cxxcore12_native_exception_fallback_when_not_fail_closed() -> None:
    inputs, _, _, _, _, _, _, _ = _capture_fixture(n=8, width=8, height=8, ticks=5, seed=12008)

    def _raising_native(*_args, **_kwargs):
        raise RuntimeError("injected native failure")

    result = run_cxxcore12_probe(
        inputs,
        config=Cxxcore12LiveTickIslandConfig(
            cxxcore12_live_tick_island_enable=True,
            cxxcore12_live_tick_island_mode="native_shadow_reference",
            cxxcore12_live_tick_island_fail_closed=False,
        ),
        native_backend=_raising_native,
    )

    assert result.status == "fallback"
    assert result.fallback_reason == "exception_fallback"
    assert result.python_state_restored is True


def test_cxxcore12_checkpoint_metrics_schema_unchanged() -> None:
    inputs, _, _, _, _, _, _, _ = _capture_fixture(n=4, width=4, height=4, ticks=3, seed=12009)

    result = run_cxxcore12_probe(
        inputs,
        config=Cxxcore12LiveTickIslandConfig(
            cxxcore12_live_tick_island_enable=True,
            cxxcore12_live_tick_island_mode="python_split_reference",
        ),
        native_backend=None,
    )
    payload = result_payload(result)

    assert payload["diagnostics"]["metrics_schema_changed"] is False
    assert payload["diagnostics"]["checkpoint_schema_changed"] is False
    assert payload["diagnostics"]["native_rng_added"] is False
    assert payload["diagnostics"]["state_backend_required"] is True
