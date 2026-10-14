from __future__ import annotations

from perf.cxxbackbone9_generation_core import (
    DECISION_POLICY_INTEGRATION_BLOCKED,
    MANDATORY_BORDER_COUNTERS,
    execute_generation,
    generation_call_graph_payload,
    resident_runtime_state_v9_audit_payload,
)


def test_cxxbackbone9_execute_generation_fails_closed_without_resident_state_v9() -> None:
    result = execute_generation(
        resident_state={},
        capability_mask={"flat": True, "patch": True, "churn": False},
        generation_index=0,
        steps_per_generation=80,
    )

    assert result["executed"] is False
    assert result["hidden_fallback"] is False
    assert result["decision"] == DECISION_POLICY_INTEGRATION_BLOCKED
    assert result["native_tick_loop_count"] == 0
    assert result["monolithic_cpp_loop"] is False
    assert result["cuda_resident_generation_loop"] is False
    assert result["border_counters"]["python_generation_calls"] == 1.0
    for name in MANDATORY_BORDER_COUNTERS:
        assert name in result["border_counters"]


def test_cxxbackbone9_call_graph_records_current_python_boundaries() -> None:
    payload = generation_call_graph_payload()
    operations = {row["operation"]: row for row in payload["rows"]}

    assert payload["schema_version"] == "cxxbackbone9_generation_call_graph_v1"
    assert payload["summary"]["native_owned_operations"] == 0
    assert operations["tick loop"]["owner"] == "python"
    assert operations["MLP/GRU/Hmem policy"]["owner"] == "mixed_python_grouping_native_aten_policy"
    assert operations["action application and Patch state"]["native_function"].startswith("cxxbackbone8r_")


def test_resident_runtime_state_v9_audit_is_schema_not_authority() -> None:
    payload = resident_runtime_state_v9_audit_payload()

    assert payload["schema_version"] == "resident_runtime_state_v9_audit_v1"
    assert payload["authoritative_owner"] == "blocked_not_yet_native_authoritative"
    assert "policy_banks" not in payload
    assert payload["current_blockers"]
