"""Contract tests for the repaired R11 authority map."""

from engine.tensor_world.c4_resident_authority_r11 import (
    C4_RESIDENT_AUTHORITY_R11_SCHEMA_VERSION_V1,
    c4_resident_authority_fields_r11_v1,
    c4_resident_authority_map_receipt_r11_v1,
    validate_c4_resident_authority_map_r11_v1,
)


def test_r11_authority_map_has_one_owner_and_ordered_device_projections() -> None:
    validate_c4_resident_authority_map_r11_v1()
    fields = c4_resident_authority_fields_r11_v1()
    assert len(fields) >= 14
    assert [field.commit_order for field in fields] == sorted(
        field.commit_order for field in fields
    )
    assert len({field.canonical_owner for field in fields}) == len(fields)
    assert all(field.derived for field in fields)
    positions = next(field for field in fields if field.name == "positions")
    assert positions.dtype == "int64"
    assert "pos_y_i32" in positions.projection


def test_r11_authority_receipt_is_fail_closed_and_has_no_host_authority() -> None:
    receipt = c4_resident_authority_map_receipt_r11_v1()
    assert receipt["schema_version"] == C4_RESIDENT_AUTHORITY_R11_SCHEMA_VERSION_V1
    assert receipt["host_tick_authority"] is False
    assert receipt["host_snapshot_rollback"] is False
    assert receipt["gate_claim"] is False
    assert receipt["authority_count"] == receipt["canonical_mutable_owner_count"]
    assert receipt["commit_order"][:3] == ["positions", "alive_mask", "agent_identity"]
