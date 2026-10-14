from __future__ import annotations

from pathlib import Path

import pytest

from engine.campaigns.tick_native import TICK_NATIVE_EXECUTION_ROUTE_V1
from engine.campaigns.tick_native_control import (
    TickNativeControlError,
    TickNativeControlV1,
    assert_tick_native_plan_payload_v1,
)
from tools.verification.tick_native_generation_firewall_v1 import run_firewall


REPO_ROOT = Path(__file__).resolve().parents[1]


def _plan_payload() -> dict[str, object]:
    control = TickNativeControlV1(total_ticks=5_000)
    return {
        "execution": {
            "execution_route_identity": TICK_NATIVE_EXECUTION_ROUTE_V1,
            "total_ticks": control.total_ticks,
            "tick_native_control": control.payload(),
        },
        "resource_estimate": {
            "resident_event_capacity": control.event_window_capacity,
            "resident_owner_span_ticks": control.resident_owner_span_ticks,
        },
    }


@pytest.mark.parametrize(
    "target",
    (
        ("execution", "nested", "logical_generation_count"),
        ("execution", "rows", 0, "next_generation"),
        ("resource_estimate", "nested", "physical_window_generations"),
        ("execution", "nested", "internal_lifecycle_schedule"),
        ("execution", "nested", "internal_lifecycle_interval_ticks"),
        ("execution", "nested", "refill_chunk_ticks"),
        ("execution", "nested", "legacy_turnover_interval_ticks"),
    ),
)
def test_recursive_plan_firewall_rejects_nested_generation_authority(
    target: tuple[object, ...],
) -> None:
    payload = _plan_payload()
    cursor: object = payload
    for offset, part in enumerate(target[:-1]):
        if isinstance(part, int):
            assert isinstance(cursor, list)
            while len(cursor) <= part:
                cursor.append({})
            cursor = cursor[part]
        else:
            assert isinstance(cursor, dict)
            next_part = target[offset + 1]
            cursor = cursor.setdefault(part, [] if isinstance(next_part, int) else {})
    assert isinstance(cursor, dict)
    cursor[target[-1]] = 7
    with pytest.raises(TickNativeControlError, match="forbidden generation authority"):
        assert_tick_native_plan_payload_v1(payload)


def test_path_symbol_firewall_reports_no_unknown_or_forbidden_active_seam() -> None:
    receipt = run_firewall(REPO_ROOT)
    assert receipt["path_symbol_aware"] is True
    assert receipt["status"] == "PASS", receipt["findings"]
