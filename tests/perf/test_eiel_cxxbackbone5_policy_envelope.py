from __future__ import annotations

from perf.cxxbackbone5_policy_envelope import (
    live_policy_boundary_map_payload,
    master_surface_ledger_payload,
    phase_attempt_ledger_payload,
)


def test_cxxbackbone5_boundary_map_names_live_python_policy_orchestration() -> None:
    payload = live_policy_boundary_map_payload(head="test-head")

    assert payload["schema_version"] == "cxxbackbone5_live_policy_boundary_map_v1"
    assert payload["full_native_policy_envelope_supported"] is False
    steps = {row["step"]: row for row in payload["steps"]}
    assert "cache_hit_refill_decision" in steps
    assert "observation_construction" in steps
    assert "policy_family_grouping" in steps
    assert "row_map_creation" in steps
    assert "action_application" in steps
    assert "Python" in steps["observation_construction"]["state_owner"]


def test_cxxbackbone5_master_ledger_preserves_historical_all19_exactness() -> None:
    payload = master_surface_ledger_payload(head="test-head")

    assert payload["schema_version"] == "cxxbackbone5_master_surface_ledger_v1"
    assert payload["surface_count"] == 19
    assert payload["historical_contract_exact_count"] == 19
    assert payload["full_native_policy_envelope_live"] is False
    rows = {row["surface_name"]: row for row in payload["rows"]}
    assert rows["mlp_gru_policy_banks"]["cuda_parity"] == "exact_for_cxxbackbone4_policy_math"
    assert rows["action_cache"]["current_live_authority"] == "not_live_native"


def test_cxxbackbone5_phase_ledger_attempts_later_groups_fail_closed() -> None:
    payload = phase_attempt_ledger_payload(head="test-head")

    attempts = {row["phase"]: row for row in payload["attempts"]}
    assert attempts["phase0_live_boundary_and_cost_map"]["status"] == "completed"
    assert attempts["phase1_unified_native_policy_service"]["status"] == "blocked"
    assert attempts["phase6_patch_wdl_jungle_environment_composition"]["status"] == "blocked"
    assert payload["decision"] == "CXXBACKBONE5-I/PARTIAL-AUTONOMOUS-CLOSEOUT"
