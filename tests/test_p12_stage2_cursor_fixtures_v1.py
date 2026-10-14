from __future__ import annotations

import pytest

from engine.evolution.p12_stage2_contracts_v1 import (
    EED_CURSOR_CONTRACT_ID,
    RESPEC_CURSOR_CONTRACT_ID,
    P12Stage2ContractError,
)
from engine.evolution.p12_stage2_cursor_fixtures_v1 import (
    clear_cursor_row_v1,
    create_cursor_row_v1,
    rebind_reused_slot_cursor_v1,
    write_cursor_row_v1,
)


@pytest.mark.parametrize("contract_id", [EED_CURSOR_CONTRACT_ID, RESPEC_CURSOR_CONTRACT_ID])
def test_cursor_write_requires_exact_slot_token_owner(contract_id: str) -> None:
    fixture = create_cursor_row_v1(
        contract_id=contract_id,
        slot=3,
        slot_token=7,
        value=11,
    )

    updated = write_cursor_row_v1(fixture, slot=3, slot_token=7, value=12)

    assert updated["row"]["value"] == 12
    with pytest.raises(P12Stage2ContractError, match="stale slot-token"):
        write_cursor_row_v1(fixture, slot=3, slot_token=6, value=13)
    with pytest.raises(P12Stage2ContractError, match="stale slot-token"):
        write_cursor_row_v1(fixture, slot=4, slot_token=7, value=13)


@pytest.mark.parametrize("reason", ["death", "rollback_rejection", "turnover_reset"])
def test_cursor_clear_blocks_later_writes(reason: str) -> None:
    fixture = create_cursor_row_v1(
        contract_id=EED_CURSOR_CONTRACT_ID,
        slot=2,
        slot_token=4,
        value=99,
    )

    cleared = clear_cursor_row_v1(fixture, reason=reason)

    assert cleared["row"]["value"] == 0
    assert cleared["row"]["live"] is False
    with pytest.raises(P12Stage2ContractError, match="dead cursor"):
        write_cursor_row_v1(cleared, slot=2, slot_token=4, value=100)


def test_reused_slot_starts_empty_and_rejects_non_increasing_token() -> None:
    old = create_cursor_row_v1(
        contract_id=RESPEC_CURSOR_CONTRACT_ID,
        slot=5,
        slot_token=8,
        value=42,
    )
    cleared = clear_cursor_row_v1(old, reason="death")

    with pytest.raises(P12Stage2ContractError, match="newer cursor token"):
        rebind_reused_slot_cursor_v1(cleared, slot=5, new_slot_token=8)

    fresh = rebind_reused_slot_cursor_v1(cleared, slot=5, new_slot_token=9)

    assert fresh["row"] == {
        "slot": 5,
        "slot_token": 9,
        "value": 0,
        "live": True,
    }
