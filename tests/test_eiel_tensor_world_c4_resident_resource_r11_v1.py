"""CPU/static inventory for the current R11 graph-owner resource envelope."""

from engine.tensor_world.c4_resident_resource_r11 import (
    C4_RESIDENT_RESOURCE_R11_DECISION_V1,
    c4_resident_composed_r11_budget_rows_v1,
    c4_resident_composed_r11_resource_contract_v1,
)


def test_r11_resource_inventory_includes_typed_staging_and_run_control() -> None:
    rows = c4_resident_composed_r11_budget_rows_v1()
    contract = c4_resident_composed_r11_resource_contract_v1()
    assert len(rows) == 112
    assert len({row.path for row in rows}) == 112
    assert contract["decision"] == C4_RESIDENT_RESOURCE_R11_DECISION_V1
    assert contract["streams"] == 5
    assert contract["graph_window_ticks"] == 5
    assert contract["capacity"] == 512
    assert contract["run_control_tensor_count"] == 16
    assert contract["identity_staging_tensor_count"] == 6
    assert contract["full_policy_shadow_payload_bytes"] == 0
    assert contract["guarded_shared_policy_payload_bytes"] == 1_907_687_424
    assert contract["required_steady_state_h2d_bytes"] == 0
    assert contract["required_steady_state_d2h_bytes"] == 0
    assert contract["graph_node_count"] is None
    assert contract["gate_claim"] is False
