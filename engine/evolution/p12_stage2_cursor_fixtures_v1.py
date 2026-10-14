"""P1.2 slot-token cursor ownership fixtures for EED and RESPEC."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final

from .p12_stage2_contracts_v1 import (
    EED_CURSOR_CONTRACT_ID,
    RESPEC_CURSOR_CONTRACT_ID,
    P12Stage2ContractError,
)


SCHEMA_VERSION: Final[str] = "p12_stage2_slot_token_cursor_fixture_v1"
SUPPORTED_CURSOR_CONTRACTS: Final[frozenset[str]] = frozenset(
    {EED_CURSOR_CONTRACT_ID, RESPEC_CURSOR_CONTRACT_ID}
)


@dataclass(frozen=True, slots=True)
class CursorRowV1:
    slot: int
    slot_token: int
    value: int
    live: bool

    def to_dict(self) -> dict[str, int | bool]:
        return {
            "slot": int(self.slot),
            "slot_token": int(self.slot_token),
            "value": int(self.value),
            "live": bool(self.live),
        }


def _validate_contract(contract_id: str) -> str:
    if contract_id not in SUPPORTED_CURSOR_CONTRACTS:
        raise P12Stage2ContractError(f"unsupported cursor contract: {contract_id}")
    return str(contract_id)


def _validate_owner(*, slot: int, slot_token: int) -> tuple[int, int]:
    slot = int(slot)
    slot_token = int(slot_token)
    if slot < 0:
        raise P12Stage2ContractError("slot must be nonnegative")
    if slot_token <= 0:
        raise P12Stage2ContractError("slot_token must be positive")
    return slot, slot_token


def create_cursor_row_v1(
    *,
    contract_id: str,
    slot: int,
    slot_token: int,
    value: int = 0,
) -> dict[str, Any]:
    """Create a live cursor row owned by the current slot token."""

    contract_id = _validate_contract(contract_id)
    slot, slot_token = _validate_owner(slot=slot, slot_token=slot_token)
    row = CursorRowV1(slot=slot, slot_token=slot_token, value=int(value), live=True)
    return {
        "schema_version": SCHEMA_VERSION,
        "contract_id": contract_id,
        "row": row.to_dict(),
    }


def write_cursor_row_v1(
    fixture: dict[str, Any],
    *,
    slot: int,
    slot_token: int,
    value: int,
) -> dict[str, Any]:
    """Write cursor state only for the exact current slot-token owner."""

    contract_id = _validate_contract(str(fixture.get("contract_id", "")))
    slot, slot_token = _validate_owner(slot=slot, slot_token=slot_token)
    row = dict(fixture.get("row", {}))
    if int(row.get("slot", -1)) != slot or int(row.get("slot_token", 0)) != slot_token:
        raise P12Stage2ContractError("stale slot-token cursor write rejected")
    if not bool(row.get("live", False)):
        raise P12Stage2ContractError("dead cursor row cannot be written")
    return {
        "schema_version": SCHEMA_VERSION,
        "contract_id": contract_id,
        "row": CursorRowV1(
            slot=slot,
            slot_token=slot_token,
            value=int(value),
            live=True,
        ).to_dict(),
    }


def clear_cursor_row_v1(
    fixture: dict[str, Any],
    *,
    reason: str,
) -> dict[str, Any]:
    """Clear cursor state on death, rollback rejection, or turnover reset."""

    contract_id = _validate_contract(str(fixture.get("contract_id", "")))
    if reason not in {"death", "rollback_rejection", "turnover_reset"}:
        raise P12Stage2ContractError("unsupported cursor clear reason")
    row = dict(fixture.get("row", {}))
    slot = int(row.get("slot", -1))
    slot_token = int(row.get("slot_token", 0))
    _validate_owner(slot=slot, slot_token=slot_token)
    return {
        "schema_version": SCHEMA_VERSION,
        "contract_id": contract_id,
        "clear_reason": reason,
        "row": CursorRowV1(
            slot=slot,
            slot_token=slot_token,
            value=0,
            live=False,
        ).to_dict(),
    }


def rebind_reused_slot_cursor_v1(
    fixture: dict[str, Any],
    *,
    slot: int,
    new_slot_token: int,
) -> dict[str, Any]:
    """Create a fresh cursor for a reused slot, rejecting non-increasing tokens."""

    contract_id = _validate_contract(str(fixture.get("contract_id", "")))
    row = dict(fixture.get("row", {}))
    slot, new_slot_token = _validate_owner(slot=slot, slot_token=new_slot_token)
    if int(row.get("slot", -1)) != slot:
        raise P12Stage2ContractError("cursor slot reuse mismatch")
    if new_slot_token <= int(row.get("slot_token", 0)):
        raise P12Stage2ContractError("slot reuse requires a newer cursor token")
    return create_cursor_row_v1(
        contract_id=contract_id,
        slot=slot,
        slot_token=new_slot_token,
        value=0,
    )


__all__ = [
    "SCHEMA_VERSION",
    "create_cursor_row_v1",
    "write_cursor_row_v1",
    "clear_cursor_row_v1",
    "rebind_reused_slot_cursor_v1",
]
