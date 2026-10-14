from __future__ import annotations

import pytest

from perf.cxxcore15_resident_soa_tick_island import (
    ResidentSoAConfig,
    as_resident_state,
    compare_resident_to_reference,
    reset_resident_state,
    resident_ownership_audit_payload,
    resident_result_payload,
    resident_state_digest,
    run_resident_python_reference,
)
from tools.scripts.eiel_cxxcore5_tick_island_layout_cache_probe import _capture_fixture


def _inputs(seed: int = 15001, *, n: int = 8, width: int = 8, height: int = 8, ticks: int = 5):
    inputs, _, _, _, _, _, _, _ = _capture_fixture(n=n, width=width, height=height, ticks=ticks, seed=seed)
    return inputs


def test_cxxcore15_resident_state_initialization_is_deterministic() -> None:
    inputs = _inputs(15002)
    a = as_resident_state(inputs)
    b = as_resident_state(inputs)

    assert resident_state_digest(a) == resident_state_digest(b)
    assert a.agent_ids.shape == b.agent_ids.shape
    assert a.resource_grid.shape == b.resource_grid.shape


def test_cxxcore15_final_export_parity_against_reference() -> None:
    inputs = _inputs(15003, n=16, width=8, height=8, ticks=10)
    state = as_resident_state(inputs)

    result = run_resident_python_reference(state, config=ResidentSoAConfig(sparse_interval=5))
    parity = compare_resident_to_reference(inputs, result)

    assert parity["ok"]
    assert result.output["new_energy"].shape == state.energy.shape
    assert result.agent_grid.shape == state.agent_grid.shape
    assert result.food_gained.shape == state.energy.shape
    assert result.moved_mask.shape == state.energy.shape


def test_cxxcore15_sparse_digest_parity_and_final_summary() -> None:
    inputs = _inputs(15004, n=16, width=8, height=8, ticks=11)
    result = run_resident_python_reference(as_resident_state(inputs), config=ResidentSoAConfig(sparse_interval=4))
    payload = resident_result_payload(result)

    assert result.sparse_ticks.tolist() == [4, 8, 11]
    assert result.sparse_summaries.shape == (3, 7)
    assert payload["final_summary"]["cost_total"] >= 0.0
    assert payload["views_dirty"] is True


def test_cxxcore15_first_divergence_localizes_against_reference() -> None:
    inputs = _inputs(15005)
    result = run_resident_python_reference(as_resident_state(inputs), inject_divergence=True)
    parity = compare_resident_to_reference(inputs, result)

    assert parity["ok"] is False
    assert parity["first_divergence"] is not None
    assert parity["first_divergence"]["field"] == "new_energy"


def test_cxxcore15_fallback_before_mutation_for_unsupported_features() -> None:
    inputs = _inputs(15006)
    state = as_resident_state(inputs)
    before = resident_state_digest(state)

    with pytest.raises(ValueError, match="unsupported resident features"):
        run_resident_python_reference(
            state,
            config=ResidentSoAConfig(unsupported_feature_reasons=("patch_world_enable",)),
        )

    assert resident_state_digest(state) == before


def test_cxxcore15_reset_rebuild_correctness() -> None:
    state = as_resident_state(_inputs(15007))
    rebuilt = reset_resident_state(state)

    assert resident_state_digest(state) == resident_state_digest(rebuilt)


def test_cxxcore15_audit_checkpoint_metrics_rng_unchanged() -> None:
    audit = resident_ownership_audit_payload()

    assert audit["checkpoint_schema_changed"] is False
    assert audit["metrics_schema_changed"] is False
    assert audit["native_rng_added"] is False
    assert "energy" in audit["resident_arrays"]
    assert "agent_grid" in audit["resident_arrays"]
    assert audit["export_every_tick"] == []
