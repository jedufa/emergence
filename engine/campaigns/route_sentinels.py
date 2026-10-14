"""Compact transition sentinels for execution-route qualification.

The sentinel deliberately captures only localization data.  It is not a
checkpoint projection and must not become an alternate source of runtime
authority.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from .canonical import canonical_bytes as _canonical_bytes
from .canonical import canonical_sha256


ROUTE_TRANSITION_SENTINEL_SCHEMA_V1 = "eiel_route_transition_sentinel_v1"
ROUTE_TRANSITION_FIRST_DIFFERENCE_SCHEMA_V1 = (
    "eiel_route_transition_first_difference_v1"
)

_OPTIONAL_FIELDS = (
    "rng_cursor",
    "tape_cursor",
    "cache_epoch",
    "refill_calls",
    "refill_row_digest",
    "action_digest",
    "alive_identity_digest",
    "runtime_event_cursor",
    "world_digest",
    "policy_digest",
)
_FIELD_ORDER = (
    "schema_version",
    "transition_name",
    "absolute_generation",
    "absolute_tick",
    "physical_window_position",
    *_OPTIONAL_FIELDS,
    "profile_fields",
)
_KNOWN_FIELDS = frozenset(_FIELD_ORDER)
_MISSING = object()


def _require_nonnegative_integer(name: str, value: int | None) -> None:
    if value is not None and (type(value) is not int or value < 0):
        raise ValueError(f"{name} must be a non-negative integer")


def _require_optional_text(name: str, value: str | None) -> None:
    if value is not None and (not isinstance(value, str) or not value):
        raise ValueError(f"{name} must be a non-empty string when supplied")


def _canonical_copy(value: Any) -> Any:
    """Detach and normalize a value through the canonical JSON boundary."""

    return json.loads(_canonical_bytes(value).decode("ascii"))


@dataclass(frozen=True, slots=True)
class RouteTransitionSentinelV1:
    """A compact observation at one execution-route transition."""

    transition_name: str
    absolute_generation: int
    absolute_tick: int
    physical_window_position: int
    rng_cursor: int | None = None
    tape_cursor: int | None = None
    cache_epoch: int | None = None
    refill_calls: int | None = None
    refill_row_digest: str | None = None
    action_digest: str | None = None
    alive_identity_digest: str | None = None
    runtime_event_cursor: int | None = None
    world_digest: str | None = None
    policy_digest: str | None = None
    profile_fields: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.transition_name, str) or not self.transition_name:
            raise ValueError("transition_name must be a non-empty string")
        for name in (
            "absolute_generation",
            "absolute_tick",
            "physical_window_position",
            "rng_cursor",
            "tape_cursor",
            "cache_epoch",
            "refill_calls",
            "runtime_event_cursor",
        ):
            _require_nonnegative_integer(name, getattr(self, name))
        for name in (
            "refill_row_digest",
            "action_digest",
            "alive_identity_digest",
            "world_digest",
            "policy_digest",
        ):
            _require_optional_text(name, getattr(self, name))
        if not isinstance(self.profile_fields, Mapping):
            raise ValueError("profile_fields must be a mapping")
        for name in self.profile_fields:
            if not isinstance(name, str) or not name:
                raise ValueError("profile_fields keys must be non-empty strings")
            if name in _KNOWN_FIELDS or name == "digest_sha256":
                raise ValueError(
                    f"profile_fields key {name!r} conflicts with a sentinel field"
                )
        try:
            normalized_profile_fields = _canonical_copy(dict(self.profile_fields))
        except ValueError as exc:
            raise ValueError("profile_fields must contain canonical JSON data") from exc
        object.__setattr__(self, "profile_fields", normalized_profile_fields)

    def payload(self) -> dict[str, Any]:
        """Return the canonical public projection, omitting absent values."""

        value: dict[str, Any] = {
            "schema_version": ROUTE_TRANSITION_SENTINEL_SCHEMA_V1,
            "transition_name": self.transition_name,
            "absolute_generation": self.absolute_generation,
            "absolute_tick": self.absolute_tick,
            "physical_window_position": self.physical_window_position,
        }
        for name in _OPTIONAL_FIELDS:
            item = getattr(self, name)
            if item is not None:
                value[name] = item
        if self.profile_fields:
            value["profile_fields"] = _canonical_copy(self.profile_fields)
        return value

    def canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.payload())

    @property
    def digest_sha256(self) -> str:
        return canonical_sha256(self.payload())

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> RouteTransitionSentinelV1:
        """Load a strict v1 sentinel mapping."""

        if not isinstance(raw, Mapping):
            raise ValueError("route transition sentinel must be a mapping")
        unknown = set(raw).difference(_KNOWN_FIELDS)
        if unknown:
            raise ValueError(
                "route transition sentinel has unknown fields: "
                + ", ".join(sorted(str(name) for name in unknown))
            )
        if raw.get("schema_version") != ROUTE_TRANSITION_SENTINEL_SCHEMA_V1:
            raise ValueError("route transition sentinel schema_version is not v1")
        required = (
            "transition_name",
            "absolute_generation",
            "absolute_tick",
            "physical_window_position",
        )
        missing = [name for name in required if name not in raw]
        if missing:
            raise ValueError(
                "route transition sentinel is missing fields: " + ", ".join(missing)
            )
        kwargs = {name: raw[name] for name in required}
        kwargs.update({name: raw[name] for name in _OPTIONAL_FIELDS if name in raw})
        kwargs["profile_fields"] = raw.get("profile_fields", {})
        return cls(**kwargs)


def _sentinel_payload(
    value: RouteTransitionSentinelV1 | Mapping[str, Any],
) -> dict[str, Any]:
    if isinstance(value, RouteTransitionSentinelV1):
        return value.payload()
    return RouteTransitionSentinelV1.from_mapping(value).payload()


def _ordered_keys(left: Mapping[str, Any], right: Mapping[str, Any]) -> list[str]:
    present = set(left).union(right)
    ordered = [name for name in _FIELD_ORDER if name in present]
    ordered.extend(sorted(present.difference(ordered)))
    return ordered


def _first_value_difference(
    left: Any,
    right: Any,
    *,
    path: str,
) -> tuple[str, Any, Any] | None:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        for key in _ordered_keys(left, right):
            left_value = left.get(key, _MISSING)
            right_value = right.get(key, _MISSING)
            child_path = f"{path}.{key}"
            if left_value is _MISSING or right_value is _MISSING:
                return child_path, left_value, right_value
            difference = _first_value_difference(
                left_value,
                right_value,
                path=child_path,
            )
            if difference is not None:
                return difference
        return None
    if isinstance(left, list) and isinstance(right, list):
        for index in range(max(len(left), len(right))):
            left_value = left[index] if index < len(left) else _MISSING
            right_value = right[index] if index < len(right) else _MISSING
            child_path = f"{path}[{index}]"
            if left_value is _MISSING or right_value is _MISSING:
                return child_path, left_value, right_value
            difference = _first_value_difference(
                left_value,
                right_value,
                path=child_path,
            )
            if difference is not None:
                return difference
        return None
    if type(left) is not type(right) or left != right:
        return path, left, right
    return None


def first_route_transition_difference(
    expected: Sequence[RouteTransitionSentinelV1 | Mapping[str, Any]],
    actual: Sequence[RouteTransitionSentinelV1 | Mapping[str, Any]],
) -> dict[str, Any] | None:
    """Return the earliest sentinel/field mismatch, or ``None`` for parity."""

    for index in range(max(len(expected), len(actual))):
        expected_value = (
            _sentinel_payload(expected[index]) if index < len(expected) else _MISSING
        )
        actual_value = (
            _sentinel_payload(actual[index]) if index < len(actual) else _MISSING
        )
        if expected_value is _MISSING or actual_value is _MISSING:
            difference = (f"sentinels[{index}]", expected_value, actual_value)
        else:
            difference = _first_value_difference(
                expected_value,
                actual_value,
                path=f"sentinels[{index}]",
            )
        if difference is None:
            continue
        path, expected_item, actual_item = difference
        transition_name = None
        for payload in (expected_value, actual_value):
            if isinstance(payload, Mapping):
                transition_name = payload.get("transition_name")
                if transition_name is not None:
                    break
        return {
            "schema_version": ROUTE_TRANSITION_FIRST_DIFFERENCE_SCHEMA_V1,
            "sentinel_index": index,
            "transition_name": transition_name,
            "path": path,
            "expected_present": expected_item is not _MISSING,
            "actual_present": actual_item is not _MISSING,
            **(
                {"expected": expected_item}
                if expected_item is not _MISSING
                else {}
            ),
            **({"actual": actual_item} if actual_item is not _MISSING else {}),
        }
    return None


__all__ = [
    "ROUTE_TRANSITION_FIRST_DIFFERENCE_SCHEMA_V1",
    "ROUTE_TRANSITION_SENTINEL_SCHEMA_V1",
    "RouteTransitionSentinelV1",
    "first_route_transition_difference",
]
