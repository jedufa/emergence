from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import shutil

import numpy as np
import pytest

from perf.cxxcore5_tick_island_layout_cache import (
    compare_runner_backend_states,
    compare_summaries,
    run_tick_island_python_reference,
    state_summary,
)
from perf.cxxcore7_rng_precompute import (
    Cxxcore7PilotConfig,
    apply_validation_action_override,
    build_rng_payload,
    config_from_sim_cfg,
    precompute_validation_action_override_ints,
    rng_surface_audit,
    run_authoritative_precomputed_rng_pilot,
    validate_rng_payload,
)
from simulation.runner_pkg.runner_config import SimConfig
from tools.scripts.eiel_cxxcore7_rng_precompute_runner_bridge_probe import (
    compile_cpp_kernel,
    load_rng_precompute_backend,
)
from tools.scripts.eiel_cxxcore5_tick_island_layout_cache_probe import _capture_fixture


def _payload(*, ticks: int, agents: int, seed: int = 7107):
    override = precompute_validation_action_override_ints(
        rng=np.random.default_rng(seed),
        ticks=ticks,
        agents=agents,
    )
    return build_rng_payload(validation_action_override_ints=override)


def _native_backend(tmp_path: Path):
    if shutil.which("g++") is None:
        pytest.skip("g++ unavailable")
    compile_info = compile_cpp_kernel(tmp_path)
    assert compile_info["ok"], compile_info
    return load_rng_precompute_backend(Path(str(compile_info["library_path"])))


def test_cxxcore7_default_off_config_and_schema_validation() -> None:
    sim_cfg = SimConfig()
    config = config_from_sim_cfg(sim_cfg)
    payload = _payload(ticks=4, agents=8)
    report = validate_rng_payload(payload, ticks=4, agents=8)

    assert config.cxxcore7_rng_precompute_enable is False
    assert config.cxxcore7_validation_action_override_enable is False
    assert report["ok"]
    assert payload.validation_action_override_ints is not None
    assert payload.validation_action_override_ints.draw_schedule.startswith("one int32 draw")
    assert payload.validation_action_override_ints.consumed_conditionally is False


def test_cxxcore7_payload_shape_mismatch_fallback() -> None:
    inputs, _, _, _, _, _, _, _ = _capture_fixture(n=8, width=8, height=8, ticks=5, seed=7201)
    payload = _payload(ticks=4, agents=8)
    config = Cxxcore7PilotConfig(
        cxxcore7_rng_precompute_enable=True,
        cxxcore7_validation_action_override_enable=True,
        cxxcore6_authoritative_pilot_enable=True,
        cxxcore6_authoritative_compare_mode="final_full",
    )

    result = run_authoritative_precomputed_rng_pilot(inputs, config=config, rng_payload=payload, native_backend=None)

    assert result.status == "fallback"
    assert result.fallback_reason == "rng_payload_invalid"
    assert result.python_state_restored is True
    assert result.cxx_result_retained is False
    assert result.diagnostics["rng_validation"]["ok"] is False


def test_cxxcore7_missing_rng_payload_fallback() -> None:
    inputs, _, _, _, _, _, _, _ = _capture_fixture(n=8, width=8, height=8, ticks=5, seed=7202)
    config = Cxxcore7PilotConfig(
        cxxcore7_rng_precompute_enable=True,
        cxxcore7_validation_action_override_enable=True,
        cxxcore6_authoritative_pilot_enable=True,
    )

    result = run_authoritative_precomputed_rng_pilot(inputs, config=config, rng_payload=None, native_backend=None)

    assert result.status == "fallback"
    assert result.fallback_reason == "rng_payload_missing"
    assert result.python_state_restored is True


def test_cxxcore7_apply_validation_action_override_changes_action_stream_only() -> None:
    inputs, _, _, _, _, _, _, _ = _capture_fixture(n=8, width=8, height=8, ticks=5, seed=7203)
    payload = _payload(ticks=5, agents=8)

    overridden = apply_validation_action_override(inputs, payload)

    assert not np.array_equal(
        inputs.backend.runner.state.actions_by_tick,
        overridden.backend.runner.state.actions_by_tick,
    )
    assert np.array_equal(inputs.backend.runner.state.initial_energy, overridden.backend.runner.state.initial_energy)
    assert np.array_equal(inputs.backend.runner.state.agent_ids, overridden.backend.runner.state.agent_ids)


def test_cxxcore7_no_independent_cpp_rng_guard() -> None:
    source = Path("engine/perf/native/cxxcore7_rng_precompute_kernel.cpp").read_text(encoding="utf-8")
    forbidden = ("std::random", "mt19937", "rand(", "srand(", "random_device")

    assert all(token not in source for token in forbidden)
    assert "validation_action_override_ints" in source


def test_cxxcore7_validation_action_override_native_final_and_sparse_parity(tmp_path: Path) -> None:
    native = _native_backend(tmp_path)
    inputs, _, _, _, _, _, _, _ = _capture_fixture(n=8, width=8, height=8, ticks=6, seed=7204)
    payload = _payload(ticks=6, agents=8)
    ref_inputs = apply_validation_action_override(inputs, payload)
    expected = run_tick_island_python_reference(ref_inputs)
    config = Cxxcore7PilotConfig(
        cxxcore7_rng_precompute_enable=True,
        cxxcore7_validation_action_override_enable=True,
        cxxcore6_authoritative_pilot_enable=True,
        cxxcore6_authoritative_compare_mode="final_full",
    )

    final = run_authoritative_precomputed_rng_pilot(
        inputs,
        config=config,
        rng_payload=payload,
        native_backend=native,
        python_reference=expected,
        compare_mode="final_full",
    )
    sparse = run_authoritative_precomputed_rng_pilot(
        inputs,
        config=replace(config, cxxcore6_authoritative_compare_mode="sparse"),
        rng_payload=payload,
        native_backend=native,
        python_reference=expected,
        compare_mode="sparse",
    )

    assert final.status == "authoritative"
    assert final.parity is not None and final.parity["ok"]
    assert compare_runner_backend_states(expected, final.output, atol=1.0e-6)["ok"]
    assert sparse.status == "authoritative"
    assert sparse.parity is not None and sparse.parity["ok"]
    assert compare_summaries(state_summary(expected), state_summary(sparse.output), atol=1.0e-6)["ok"]


def test_cxxcore7_native_unavailable_fallback_uses_overridden_python_reference() -> None:
    inputs, _, _, _, _, _, _, _ = _capture_fixture(n=8, width=8, height=8, ticks=5, seed=7205)
    payload = _payload(ticks=5, agents=8)
    expected = run_tick_island_python_reference(apply_validation_action_override(inputs, payload))
    config = Cxxcore7PilotConfig(
        cxxcore7_rng_precompute_enable=True,
        cxxcore7_validation_action_override_enable=True,
        cxxcore6_authoritative_pilot_enable=True,
    )

    result = run_authoritative_precomputed_rng_pilot(
        inputs,
        config=config,
        rng_payload=payload,
        native_backend=None,
        python_reference=expected,
    )

    assert result.status == "fallback"
    assert result.fallback_reason == "native_unavailable"
    assert compare_runner_backend_states(expected, result.output, atol=1.0e-6)["ok"]


def test_cxxcore7_injected_divergence_falls_back_to_python_reference(tmp_path: Path) -> None:
    native = _native_backend(tmp_path)
    inputs, _, _, _, _, _, _, _ = _capture_fixture(n=8, width=8, height=8, ticks=5, seed=7206)
    payload = _payload(ticks=5, agents=8)
    expected = run_tick_island_python_reference(apply_validation_action_override(inputs, payload))
    config = Cxxcore7PilotConfig(
        cxxcore7_rng_precompute_enable=True,
        cxxcore7_validation_action_override_enable=True,
        cxxcore6_authoritative_pilot_enable=True,
        cxxcore6_authoritative_compare_mode="final_full",
        cxxcore6_authoritative_fallback_on_divergence=True,
    )

    result = run_authoritative_precomputed_rng_pilot(
        inputs,
        config=config,
        rng_payload=payload,
        native_backend=native,
        python_reference=expected,
        compare_mode="final_full",
        inject_divergence=True,
    )

    assert result.status == "fallback"
    assert result.fallback_reason == "divergence"
    assert result.parity is not None and result.parity["ok"] is False
    assert compare_runner_backend_states(expected, result.output, atol=1.0e-6)["ok"]


def test_cxxcore7_rng_audit_boundaries() -> None:
    audit = rng_surface_audit(included_surfaces=("validation_action_override_ints",))
    rows = {row["surface"]: row for row in audit["rows"]}

    assert rows["validation random-walk action override"]["included_in_cxxcore7"] is True
    assert rows["biome ambient hazard"]["included_in_cxxcore7"] is False
    assert rows["Tier 3 transition/crossing risk"]["branch_dependent"] is True
    assert audit["checkpoint_boundary"].startswith("no checkpoint-required state added")
    assert audit["cpp_rng_boundary"].startswith("no independent C++ RNG")
