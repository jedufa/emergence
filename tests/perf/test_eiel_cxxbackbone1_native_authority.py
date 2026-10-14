from __future__ import annotations

import numpy as np

from perf.cxxbackbone1_native_authority import (
    RESIDENT_STATE_SCHEMA_VERSION,
    SURFACE_NAMES,
    build_surface_ledger,
    cxx_backbone_capability_graph_v3_payload,
    resident_runtime_state_v4,
    resident_runtime_state_v4_payload,
)


def test_cxxbackbone1_surface_ledger_lists_all_19_surfaces() -> None:
    ledger = build_surface_ledger(active_features=("full_scientific", "public_report", "multi_cell"))

    assert len(ledger) == 19
    assert tuple((row.surface_id, row.surface_name) for row in ledger) == SURFACE_NAMES
    assert all(row.active for row in ledger)
    assert all(not row.native_authoritative for row in ledger)
    assert ledger[0].blocker == "action_cache:native_authority_missing"


def test_cxxbackbone1_t0_flat_implementation_marks_only_actual_surfaces_native() -> None:
    implemented = {
        "execution_cell_metadata": {"native_authoritative": True, "implemented_contract": "cxxwave1h_t0", "generation_parity": "exact"},
        "capability_support_mask": {"native_authoritative": True, "implemented_contract": "cxxwave1h_t0", "unit_parity": "exact"},
        "action_cache": {"native_authoritative": True, "implemented_contract": "cxxwave1h_t0_cursor", "generation_parity": "exact"},
        "mlp_gru_policy_banks": {"native_authoritative": True, "implemented_contract": "cxxwave1h_t0_mlp_only", "generation_parity": "exact"},
        "observations_staging": {"native_authoritative": True, "implemented_contract": "cxxwave1h_t0_flat_obs", "generation_parity": "exact"},
        "policy_row_mappings": {"native_authoritative": True, "implemented_contract": "cxxwave1h_t0_row_order", "generation_parity": "exact"},
        "population_slots_stable_ids": {"native_authoritative": True, "implemented_contract": "cxxwave1h_t0_slots", "generation_parity": "exact"},
    }
    payload = cxx_backbone_capability_graph_v3_payload(active_features=("flat_world", "mlp_only"), implemented_surfaces=implemented)

    assert payload["schema_version"] == "cpp_backbone_capability_graph_v3"
    assert payload["native_authoritative_count"] == 7
    assert payload["active_surface_count"] == 14
    assert payload["blocking_surface_count"] == 7
    assert payload["all_active_surfaces_native"] is False


def test_resident_runtime_state_v4_payload_is_deterministic() -> None:
    arrays = {
        "agent_ids": np.asarray([2, 5], dtype=np.int32),
        "slot_tokens": np.asarray([10, 11], dtype=np.int64),
        "mlp_pack": np.zeros((2, 4), dtype=np.float32),
        "action_cache_cursor": np.zeros((2,), dtype=np.int32),
        "obs_flat": np.zeros((2, 9), dtype=np.float32),
    }
    state_a = resident_runtime_state_v4(
        generation=1,
        tick=0,
        cell_id="flat_t0",
        active_features=("flat_world", "mlp_only"),
        arrays=arrays,
    )
    state_b = resident_runtime_state_v4(
        generation=1,
        tick=0,
        cell_id="flat_t0",
        active_features=("flat_world", "mlp_only"),
        arrays=arrays,
    )
    payload = resident_runtime_state_v4_payload(state_a)

    assert state_a.state_digest == state_b.state_digest
    assert payload["schema_version"] == RESIDENT_STATE_SCHEMA_VERSION
    assert payload["cell_id"] == "flat_t0"
    assert any(row["field"] == "action_cache_cursor" for row in payload["field_authority"])
