from __future__ import annotations

import math

import pytest

from engine.device_selection import (
    AmbiguousDeviceSelectionError,
    DeviceSelectionError,
    UnsatisfiedDeviceSelectionError,
    plan_device_selection,
)


VERSION = "eiel_device_selection_policy_v1"


def _device(
    index: int,
    uuid: str,
    *,
    headroom: int,
    latency: float,
    throughput: float,
) -> dict[str, object]:
    return {
        "index": index,
        "uuid": uuid,
        "memory_headroom_bytes": headroom,
        "estimated_latency_ms": latency,
        "estimated_throughput": throughput,
    }


def _policy(**updates: object) -> dict[str, object]:
    value: dict[str, object] = {
        "schema_version": VERSION,
        "device_index": None,
        "device_uuid": None,
        "minimum_memory_headroom_bytes": 0,
        "objective": "latency",
    }
    value.update(updates)
    return value


INVENTORY = [
    _device(0, "GPU-alpha", headroom=8_000, latency=4.0, throughput=100.0),
    _device(1, "GPU-beta", headroom=12_000, latency=6.0, throughput=160.0),
]


def test_latency_objective_selects_lowest_latency() -> None:
    plan = plan_device_selection(INVENTORY, _policy(objective="latency"))

    assert plan.selected.uuid == "GPU-alpha"
    assert plan.selection_basis == "latency"
    assert plan.eligible_device_count == 2


def test_throughput_objective_selects_highest_throughput() -> None:
    plan = plan_device_selection(INVENTORY, _policy(objective="throughput"))

    assert plan.selected.uuid == "GPU-beta"


def test_minimum_memory_headroom_filters_faster_device() -> None:
    plan = plan_device_selection(
        INVENTORY,
        _policy(objective="latency", minimum_memory_headroom_bytes=10_000),
    )

    assert plan.selected.uuid == "GPU-beta"
    assert plan.eligible_device_count == 1


@pytest.mark.parametrize(
    ("selector", "expected"),
    [({"device_index": 1}, "GPU-beta"), ({"device_uuid": "GPU-alpha"}, "GPU-alpha")],
)
def test_explicit_selector_is_honored(
    selector: dict[str, object], expected: str
) -> None:
    plan = plan_device_selection(INVENTORY, _policy(**selector))

    assert plan.selected.uuid == expected
    assert plan.selection_basis == "explicit_device"


def test_matching_index_and_uuid_are_accepted() -> None:
    plan = plan_device_selection(
        INVENTORY, _policy(device_index=1, device_uuid="GPU-beta")
    )

    assert plan.selected.index == 1


def test_conflicting_index_and_uuid_are_explicitly_ambiguous() -> None:
    with pytest.raises(AmbiguousDeviceSelectionError, match="different devices"):
        plan_device_selection(
            INVENTORY, _policy(device_index=0, device_uuid="GPU-beta")
        )


@pytest.mark.parametrize(
    "selector",
    [{"device_index": 7}, {"device_uuid": "GPU-missing"}],
)
def test_missing_explicit_selector_is_unsatisfied(
    selector: dict[str, object],
) -> None:
    with pytest.raises(UnsatisfiedDeviceSelectionError, match="unavailable"):
        plan_device_selection(INVENTORY, _policy(**selector))


def test_explicit_device_must_meet_headroom() -> None:
    with pytest.raises(UnsatisfiedDeviceSelectionError, match="headroom"):
        plan_device_selection(
            INVENTORY,
            _policy(device_index=0, minimum_memory_headroom_bytes=9_000),
        )


def test_empty_inventory_is_unsatisfied() -> None:
    with pytest.raises(UnsatisfiedDeviceSelectionError, match="headroom"):
        plan_device_selection([], _policy())


def test_tie_breaking_is_deterministic_and_input_order_independent() -> None:
    first = _device(3, "GPU-zeta", headroom=100, latency=2.0, throughput=8.0)
    second = _device(2, "GPU-eta", headroom=100, latency=2.0, throughput=8.0)

    forward = plan_device_selection([first, second], _policy()).selected
    reverse = plan_device_selection([second, first], _policy()).selected

    assert forward == reverse
    assert forward.index == 2


def test_latency_tie_prefers_throughput_then_headroom() -> None:
    rows = [
        _device(0, "GPU-a", headroom=500, latency=2.0, throughput=8.0),
        _device(1, "GPU-b", headroom=100, latency=2.0, throughput=9.0),
    ]

    assert plan_device_selection(rows, _policy()).selected.uuid == "GPU-b"


def test_throughput_tie_prefers_latency_then_headroom() -> None:
    rows = [
        _device(0, "GPU-a", headroom=500, latency=3.0, throughput=9.0),
        _device(1, "GPU-b", headroom=100, latency=2.0, throughput=9.0),
    ]

    selected = plan_device_selection(rows, _policy(objective="throughput")).selected
    assert selected.uuid == "GPU-b"


@pytest.mark.parametrize(
    "inventory",
    [
        "not-an-array",
        [_device(0, "GPU-a", headroom=1, latency=1, throughput=1), {"index": 1}],
    ],
)
def test_inventory_shape_is_strict(inventory: object) -> None:
    with pytest.raises(DeviceSelectionError):
        plan_device_selection(inventory, _policy())


@pytest.mark.parametrize("field", ["index", "uuid"])
def test_inventory_identities_must_be_unique(field: str) -> None:
    first = _device(0, "GPU-a", headroom=1, latency=1, throughput=1)
    second = _device(1, "GPU-b", headroom=1, latency=1, throughput=1)
    second[field] = first[field]

    with pytest.raises(DeviceSelectionError, match="unique"):
        plan_device_selection([first, second], _policy())


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("index", True),
        ("index", -1),
        ("uuid", "bad uuid"),
        ("memory_headroom_bytes", -1),
        ("estimated_latency_ms", 0),
        ("estimated_latency_ms", math.inf),
        ("estimated_throughput", math.nan),
    ],
)
def test_inventory_values_are_strict(field: str, value: object) -> None:
    row = _device(0, "GPU-a", headroom=1, latency=1, throughput=1)
    row[field] = value

    with pytest.raises(DeviceSelectionError):
        plan_device_selection([row], _policy())


@pytest.mark.parametrize(
    "updates",
    [
        {"schema_version": "eiel_device_selection_policy_v2"},
        {"objective": "balanced"},
        {"device_index": True},
        {"device_uuid": "bad uuid"},
        {"minimum_memory_headroom_bytes": -1},
    ],
)
def test_policy_values_are_strict(updates: dict[str, object]) -> None:
    with pytest.raises(DeviceSelectionError):
        plan_device_selection(INVENTORY, _policy(**updates))


def test_policy_fields_are_exact() -> None:
    policy = _policy()
    policy["unexpected"] = True

    with pytest.raises(DeviceSelectionError, match="extra"):
        plan_device_selection(INVENTORY, policy)


def test_planner_has_no_inventory_side_effects() -> None:
    inventory = [dict(row) for row in INVENTORY]
    before = [dict(row) for row in inventory]

    plan_device_selection(inventory, _policy())

    assert inventory == before
