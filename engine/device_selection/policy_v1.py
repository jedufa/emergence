"""Side-effect-free device selection from caller-supplied inventory data.

This module does not enumerate devices, import accelerator libraries, or mutate
process state. Callers must supply qualified inventory measurements.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Any, Final, Mapping, Sequence


DEVICE_SELECTION_POLICY_VERSION: Final[str] = "eiel_device_selection_policy_v1"
SELECTION_OBJECTIVES: Final[frozenset[str]] = frozenset({"latency", "throughput"})
_DEVICE_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "index",
        "uuid",
        "memory_headroom_bytes",
        "estimated_latency_ms",
        "estimated_throughput",
    }
)
_POLICY_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "device_index",
        "device_uuid",
        "minimum_memory_headroom_bytes",
        "objective",
    }
)
_IDENTIFIER_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:+-]{0,127}\Z")


class DeviceSelectionError(ValueError):
    """Inventory or policy input is malformed."""


class UnsatisfiedDeviceSelectionError(DeviceSelectionError):
    """No supplied device satisfies the validated policy."""


class AmbiguousDeviceSelectionError(DeviceSelectionError):
    """Validated selectors identify different supplied devices."""


@dataclass(frozen=True, slots=True)
class DeviceCandidate:
    index: int
    uuid: str
    memory_headroom_bytes: int
    estimated_latency_ms: float
    estimated_throughput: float


@dataclass(frozen=True, slots=True)
class DeviceSelectionPolicy:
    schema_version: str
    device_index: int | None
    device_uuid: str | None
    minimum_memory_headroom_bytes: int
    objective: str


@dataclass(frozen=True, slots=True)
class DeviceSelectionPlan:
    schema_version: str
    policy: DeviceSelectionPolicy
    selected: DeviceCandidate
    eligible_device_count: int
    selection_basis: str


def _exact_object(
    value: Any, *, fields: frozenset[str], name: str
) -> Mapping[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise DeviceSelectionError(f"{name} must be an object with string keys")
    missing = sorted(fields - set(value))
    extra = sorted(set(value) - fields)
    if missing or extra:
        raise DeviceSelectionError(
            f"{name} fields are not exact; missing={missing!r} extra={extra!r}"
        )
    return value


def _nonnegative_integer(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise DeviceSelectionError(f"{name} must be a nonnegative integer")
    return value


def _identifier(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER_RE.fullmatch(value) is None:
        raise DeviceSelectionError(f"{name} must be a stable device identifier")
    return value


def _positive_finite(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DeviceSelectionError(f"{name} must be a positive finite number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise DeviceSelectionError(f"{name} must be a positive finite number")
    return result


def _candidate(value: Any, *, position: int) -> DeviceCandidate:
    row = _exact_object(value, fields=_DEVICE_FIELDS, name=f"inventory[{position}]")
    return DeviceCandidate(
        index=_nonnegative_integer(row["index"], name=f"inventory[{position}].index"),
        uuid=_identifier(row["uuid"], name=f"inventory[{position}].uuid"),
        memory_headroom_bytes=_nonnegative_integer(
            row["memory_headroom_bytes"],
            name=f"inventory[{position}].memory_headroom_bytes",
        ),
        estimated_latency_ms=_positive_finite(
            row["estimated_latency_ms"],
            name=f"inventory[{position}].estimated_latency_ms",
        ),
        estimated_throughput=_positive_finite(
            row["estimated_throughput"],
            name=f"inventory[{position}].estimated_throughput",
        ),
    )


def _inventory(value: Any) -> tuple[DeviceCandidate, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise DeviceSelectionError("inventory must be an array of device objects")
    candidates = tuple(_candidate(row, position=index) for index, row in enumerate(value))
    indexes = [candidate.index for candidate in candidates]
    uuids = [candidate.uuid for candidate in candidates]
    if len(indexes) != len(set(indexes)):
        raise DeviceSelectionError("inventory device indexes must be unique")
    if len(uuids) != len(set(uuids)):
        raise DeviceSelectionError("inventory device UUIDs must be unique")
    return candidates


def _policy(value: Any) -> DeviceSelectionPolicy:
    row = _exact_object(value, fields=_POLICY_FIELDS, name="policy")
    if row["schema_version"] != DEVICE_SELECTION_POLICY_VERSION:
        raise DeviceSelectionError(
            f"policy.schema_version must be {DEVICE_SELECTION_POLICY_VERSION!r}"
        )
    raw_index = row["device_index"]
    device_index = (
        None
        if raw_index is None
        else _nonnegative_integer(raw_index, name="policy.device_index")
    )
    raw_uuid = row["device_uuid"]
    device_uuid = (
        None if raw_uuid is None else _identifier(raw_uuid, name="policy.device_uuid")
    )
    objective = row["objective"]
    if not isinstance(objective, str) or objective not in SELECTION_OBJECTIVES:
        raise DeviceSelectionError(
            f"policy.objective must be one of {sorted(SELECTION_OBJECTIVES)!r}"
        )
    return DeviceSelectionPolicy(
        schema_version=DEVICE_SELECTION_POLICY_VERSION,
        device_index=device_index,
        device_uuid=device_uuid,
        minimum_memory_headroom_bytes=_nonnegative_integer(
            row["minimum_memory_headroom_bytes"],
            name="policy.minimum_memory_headroom_bytes",
        ),
        objective=objective,
    )


def _objective_key(candidate: DeviceCandidate, objective: str) -> tuple[Any, ...]:
    if objective == "latency":
        return (
            candidate.estimated_latency_ms,
            -candidate.estimated_throughput,
            -candidate.memory_headroom_bytes,
            candidate.index,
            candidate.uuid,
        )
    return (
        -candidate.estimated_throughput,
        candidate.estimated_latency_ms,
        -candidate.memory_headroom_bytes,
        candidate.index,
        candidate.uuid,
    )


def plan_device_selection(
    inventory: Any, policy: Any
) -> DeviceSelectionPlan:
    """Validate inputs and select one supplied device deterministically."""

    candidates = _inventory(inventory)
    checked_policy = _policy(policy)
    by_index = (
        next(
            (row for row in candidates if row.index == checked_policy.device_index),
            None,
        )
        if checked_policy.device_index is not None
        else None
    )
    by_uuid = (
        next(
            (row for row in candidates if row.uuid == checked_policy.device_uuid),
            None,
        )
        if checked_policy.device_uuid is not None
        else None
    )
    if checked_policy.device_index is not None and by_index is None:
        raise UnsatisfiedDeviceSelectionError(
            f"requested device index {checked_policy.device_index} is unavailable"
        )
    if checked_policy.device_uuid is not None and by_uuid is None:
        raise UnsatisfiedDeviceSelectionError(
            f"requested device UUID {checked_policy.device_uuid!r} is unavailable"
        )
    if by_index is not None and by_uuid is not None and by_index != by_uuid:
        raise AmbiguousDeviceSelectionError(
            "device_index and device_uuid identify different devices"
        )

    requested = by_index or by_uuid
    eligible = tuple(
        row
        for row in candidates
        if row.memory_headroom_bytes
        >= checked_policy.minimum_memory_headroom_bytes
        and (requested is None or row == requested)
    )
    if not eligible:
        detail = "requested device" if requested is not None else "supplied inventory"
        raise UnsatisfiedDeviceSelectionError(
            f"{detail} does not satisfy minimum memory headroom "
            f"{checked_policy.minimum_memory_headroom_bytes}"
        )

    selected = min(
        eligible,
        key=lambda candidate: _objective_key(candidate, checked_policy.objective),
    )
    basis = "explicit_device" if requested is not None else checked_policy.objective
    return DeviceSelectionPlan(
        schema_version=DEVICE_SELECTION_POLICY_VERSION,
        policy=checked_policy,
        selected=selected,
        eligible_device_count=len(eligible),
        selection_basis=basis,
    )


__all__ = [
    "AmbiguousDeviceSelectionError",
    "DEVICE_SELECTION_POLICY_VERSION",
    "DeviceCandidate",
    "DeviceSelectionError",
    "DeviceSelectionPlan",
    "DeviceSelectionPolicy",
    "SELECTION_OBJECTIVES",
    "UnsatisfiedDeviceSelectionError",
    "plan_device_selection",
]
