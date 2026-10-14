from __future__ import annotations

from pathlib import Path

from perf.cxxwave1h_capability_live_backbone import (
    capability_graph_payload,
    compare_flat_mlp_results,
    compile_flat_mlp_kernel,
    execute_t0_flat_mlp_generation,
    input_payload,
    load_flat_mlp_native_backend,
    make_flat_mlp_fixture,
    resolve_capabilities_v2,
    result_digest,
    run_flat_mlp_python_reference,
)


def test_cxxwave1h_capability_resolution_ignores_inactive_later_tiers() -> None:
    resolution = resolve_capabilities_v2(
        requested_tier="T0",
        active_features=("flat_world", "mlp_only", "deterministic_actions", "no_hmem", "no_checkpoint", "no_public_report"),
        native_state=(
            "execution_cell_metadata",
            "population_slots_stable_ids",
            "mlp_policy_banks",
            "policy_row_mappings",
            "flat_observations_staging",
            "action_selection",
            "action_cache_cursor",
            "flat_environment_state",
            "chemistry_scalar_state",
            "minimal_summary",
        ),
        native_kernels=("cxxwave1h_flat_mlp_generation_v1",),
    )
    payload = capability_graph_payload(resolution)

    assert resolution.supported is True
    assert resolution.active_capabilities == ("flat_mlp_generation_native",)
    assert "patch_low_churn_checkpoint_native" in resolution.inactive_capabilities
    assert payload["schema_version"] == "cpp_backbone_capability_graph_v2"


def test_cxxwave1h_capability_resolution_fails_closed_for_missing_t0_kernel() -> None:
    resolution = resolve_capabilities_v2(
        requested_tier="T0",
        active_features=("flat_world", "mlp_only", "deterministic_actions", "no_hmem", "no_checkpoint", "no_public_report"),
        native_state=(
            "execution_cell_metadata",
            "population_slots_stable_ids",
            "mlp_policy_banks",
            "policy_row_mappings",
            "flat_observations_staging",
            "action_selection",
            "action_cache_cursor",
            "flat_environment_state",
            "chemistry_scalar_state",
            "minimal_summary",
        ),
        native_kernels=(),
    )

    assert resolution.supported is False
    assert "flat_mlp_generation_native:missing_kernel:cxxwave1h_flat_mlp_generation_v1" in resolution.blockers


def test_cxxwave1h_python_reference_is_deterministic() -> None:
    inputs = make_flat_mlp_fixture(population=8, grid=8, ticks=5, seed=181)
    a = run_flat_mlp_python_reference(inputs)
    b = run_flat_mlp_python_reference(inputs)

    assert result_digest(a) == result_digest(b)
    assert compare_flat_mlp_results(a, b)["ok"]
    assert input_payload(inputs)["population"] == 8


def test_cxxwave1h_t0_native_flat_mlp_generation_parity(tmp_path: Path) -> None:
    inputs = make_flat_mlp_fixture(population=12, grid=8, ticks=6, seed=182, hidden=8)
    compile_info = compile_flat_mlp_kernel(tmp_path / "build")

    assert compile_info["ok"], compile_info

    backend = load_flat_mlp_native_backend(Path(str(compile_info["library_path"])))
    payload = execute_t0_flat_mlp_generation(inputs, native_backend=backend)

    assert payload["executed_native"] is True
    assert payload["hidden_fallback"] is False
    assert payload["decision"] == "CXXWAVE1H-T0-A/FLAT-MLP-LIVE-NATIVE-PARITY"
    assert payload["parity"]["ok"] is True
    assert payload["python_digest"] == payload["native_digest"]
