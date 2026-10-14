from __future__ import annotations

import json

import pytest

from engine.campaigns.route_sentinels import (
    ROUTE_TRANSITION_SENTINEL_SCHEMA_V1,
    RouteTransitionSentinelV1,
    first_route_transition_difference,
)


def _sentinel(**changes: object) -> RouteTransitionSentinelV1:
    values: dict[str, object] = {
        "transition_name": "post_boundary_cache_refill",
        "absolute_generation": 2,
        "absolute_tick": 1,
        "physical_window_position": 0,
        "rng_cursor": 48,
        "tape_cursor": 96,
        "cache_epoch": 2,
        "refill_calls": 1,
        "refill_row_digest": "refill-row-a",
        "action_digest": "actions-a",
        "alive_identity_digest": "alive-a",
        "runtime_event_cursor": 14,
        "world_digest": "world-a",
        "policy_digest": "policy-a",
    }
    values.update(changes)
    return RouteTransitionSentinelV1(**values)


def test_compact_payload_omits_unsupplied_optional_values() -> None:
    sentinel = RouteTransitionSentinelV1(
        transition_name="generation_boundary",
        absolute_generation=2,
        absolute_tick=0,
        physical_window_position=1,
    )

    assert sentinel.payload() == {
        "schema_version": ROUTE_TRANSITION_SENTINEL_SCHEMA_V1,
        "transition_name": "generation_boundary",
        "absolute_generation": 2,
        "absolute_tick": 0,
        "physical_window_position": 1,
    }
    assert "profile_fields" not in sentinel.payload()


def test_canonical_serialization_and_digest_are_order_independent() -> None:
    first = _sentinel(
        profile_fields={
            "fixed_mlp": {"lane": 1, "owners": ["controller", "cache"]},
            "boundary_kind": "turnover",
        }
    )
    second = _sentinel(
        profile_fields={
            "boundary_kind": "turnover",
            "fixed_mlp": {"owners": ["controller", "cache"], "lane": 1},
        }
    )

    assert first.canonical_bytes() == second.canonical_bytes()
    assert first.digest_sha256 == second.digest_sha256
    assert json.loads(first.canonical_bytes()) == first.payload()
    assert first.digest_sha256 == (
        "07f7961dce7628f460c28040fd391436df07a6231cd68034704cc6d7af1ea285"
    )


def test_mapping_round_trip_preserves_canonical_projection() -> None:
    original = _sentinel(profile_fields={"policy_cache_rows": 2})

    restored = RouteTransitionSentinelV1.from_mapping(original.payload())

    assert restored == original
    assert restored.digest_sha256 == original.digest_sha256


def test_profile_fields_are_detached_and_must_be_canonical_json() -> None:
    supplied = {"nested": {"rows": [0, 1]}}
    sentinel = _sentinel(profile_fields=supplied)
    supplied["nested"]["rows"].append(2)

    assert sentinel.payload()["profile_fields"] == {"nested": {"rows": [0, 1]}}
    with pytest.raises(ValueError, match="canonical JSON"):
        _sentinel(profile_fields={"bad": {1, 2}})
    with pytest.raises(ValueError, match="conflicts"):
        _sentinel(profile_fields={"action_digest": "duplicate"})


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"absolute_generation": -1}, "absolute_generation"),
        ({"absolute_tick": True}, "absolute_tick"),
        ({"physical_window_position": -1}, "physical_window_position"),
        ({"rng_cursor": -1}, "rng_cursor"),
        ({"transition_name": ""}, "transition_name"),
        ({"action_digest": ""}, "action_digest"),
    ],
)
def test_invalid_common_fields_fail_closed(
    changes: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _sentinel(**changes)


def test_first_difference_localizes_earliest_transition_field() -> None:
    expected = [
        _sentinel(transition_name="generation_boundary", absolute_tick=0),
        _sentinel(action_digest="actions-expected"),
    ]
    actual = [
        _sentinel(transition_name="generation_boundary", absolute_tick=0),
        _sentinel(action_digest="actions-actual", world_digest="world-actual"),
    ]

    difference = first_route_transition_difference(expected, actual)

    assert difference == {
        "schema_version": "eiel_route_transition_first_difference_v1",
        "sentinel_index": 1,
        "transition_name": "post_boundary_cache_refill",
        "path": "sentinels[1].action_digest",
        "expected_present": True,
        "actual_present": True,
        "expected": "actions-expected",
        "actual": "actions-actual",
    }


def test_first_difference_reports_absent_optional_field() -> None:
    expected = [_sentinel(cache_epoch=None)]
    actual = [_sentinel(cache_epoch=2)]

    difference = first_route_transition_difference(expected, actual)

    assert difference is not None
    assert difference["path"] == "sentinels[0].cache_epoch"
    assert difference["expected_present"] is False
    assert difference["actual_present"] is True
    assert "expected" not in difference
    assert difference["actual"] == 2


def test_first_difference_reports_missing_sentinel_and_exact_parity() -> None:
    sentinel = _sentinel()

    assert first_route_transition_difference([sentinel], [sentinel.payload()]) is None
    difference = first_route_transition_difference([sentinel], [])
    assert difference is not None
    assert difference["path"] == "sentinels[0]"
    assert difference["expected_present"] is True
    assert difference["actual_present"] is False
    assert "actual" not in difference


def test_mapping_loader_rejects_unknown_or_wrong_schema() -> None:
    raw = _sentinel().payload()
    with pytest.raises(ValueError, match="unknown fields"):
        RouteTransitionSentinelV1.from_mapping({**raw, "checkpoint": {}})
    with pytest.raises(ValueError, match="schema_version"):
        RouteTransitionSentinelV1.from_mapping(
            {**raw, "schema_version": "eiel_route_transition_sentinel_v2"}
        )
