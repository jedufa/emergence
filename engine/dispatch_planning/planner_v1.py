"""Pure planning for composable qualified physical dispatches.

The planner has no CUDA, artifact-loading, source-allocation, or production
routing authority. A profile declares a divisibility chain of independently
qualified physical sizes. Largest-first composition is therefore deterministic
and leaves a tail smaller than the smallest qualified size.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Final


DISPATCH_PLAN_SCHEMA_VERSION_V1: Final[str] = "eiel_dispatch_plan_v1"
DEFAULT_MAX_PHYSICAL_DISPATCHES: Final[int] = 100_000
MAX_EXACT_INTEGER: Final[int] = (1 << 63) - 1
_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")
_PREFIX = re.compile(r"^[A-Za-z]{1,16}$")
_REQUEST = re.compile(r"^(?P<prefix>[A-Za-z]{1,16})(?P<units>[1-9][0-9]*)$")


class DispatchPlanningError(ValueError):
    """Raised when a profile, request, or resulting plan is invalid."""


def _exact_positive_integer(value: object, *, name: str) -> int:
    if type(value) is not int or not 1 <= value <= MAX_EXACT_INTEGER:
        raise DispatchPlanningError(
            f"{name} must be an exact integer in [1, {MAX_EXACT_INTEGER}]"
        )
    return value


@dataclass(frozen=True)
class LogicalDispatchRequest:
    """One logical request, such as ``B100`` or ``g128``."""

    prefix: str
    units: int

    def __post_init__(self) -> None:
        if not isinstance(self.prefix, str) or _PREFIX.fullmatch(self.prefix) is None:
            raise DispatchPlanningError("request prefix is invalid")
        _exact_positive_integer(self.units, name="request units")

    @classmethod
    def parse(cls, value: str) -> LogicalDispatchRequest:
        if not isinstance(value, str):
            raise DispatchPlanningError("logical request must be a string")
        match = _REQUEST.fullmatch(value)
        if match is None:
            raise DispatchPlanningError(
                "logical request must contain a prefix followed by a positive integer"
            )
        units = int(match.group("units"))
        _exact_positive_integer(units, name="request units")
        return cls(prefix=match.group("prefix"), units=units)

    @property
    def logical_id(self) -> str:
        return f"{self.prefix}{self.units}"


@dataclass(frozen=True)
class QualifiedChunkProfile:
    """Qualified sizes for one logical request prefix."""

    profile_id: str
    request_prefix: str
    qualified_chunk_sizes: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.profile_id, str) or _IDENTIFIER.fullmatch(self.profile_id) is None:
            raise DispatchPlanningError("profile_id is invalid")
        if (
            not isinstance(self.request_prefix, str)
            or _PREFIX.fullmatch(self.request_prefix) is None
        ):
            raise DispatchPlanningError("request_prefix is invalid")
        if not isinstance(self.qualified_chunk_sizes, tuple) or not self.qualified_chunk_sizes:
            raise DispatchPlanningError(
                "qualified_chunk_sizes must be a nonempty tuple"
            )
        sizes = self.qualified_chunk_sizes
        for index, size in enumerate(sizes):
            _exact_positive_integer(size, name=f"qualified_chunk_sizes[{index}]")
        if tuple(sorted(set(sizes), reverse=True)) != sizes:
            raise DispatchPlanningError(
                "qualified_chunk_sizes must be unique and strictly descending"
            )
        for larger, smaller in zip(sizes, sizes[1:]):
            if larger % smaller != 0:
                raise DispatchPlanningError(
                    "qualified_chunk_sizes must form a divisibility chain"
                )


@dataclass(frozen=True)
class PhysicalDispatch:
    ordinal: int
    logical_offset: int
    units: int


@dataclass(frozen=True)
class TailDispatch:
    logical_offset: int
    units: int


@dataclass(frozen=True)
class DispatchPlan:
    schema_version: str
    profile_id: str
    request: LogicalDispatchRequest
    physical_dispatches: tuple[PhysicalDispatch, ...]
    tail: TailDispatch

    @property
    def qualified_units(self) -> int:
        return sum(dispatch.units for dispatch in self.physical_dispatches)

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "profile_id": self.profile_id,
            "logical_request": self.request.logical_id,
            "qualified_units": self.qualified_units,
            "physical_dispatches": [
                {
                    "ordinal": dispatch.ordinal,
                    "logical_offset": dispatch.logical_offset,
                    "units": dispatch.units,
                }
                for dispatch in self.physical_dispatches
            ],
            "tail": {
                "logical_offset": self.tail.logical_offset,
                "units": self.tail.units,
            },
        }


def plan_dispatch(
    request: LogicalDispatchRequest | str,
    *,
    profile: QualifiedChunkProfile,
    max_physical_dispatches: int = DEFAULT_MAX_PHYSICAL_DISPATCHES,
) -> DispatchPlan:
    """Compose a logical request from qualified chunks and a bounded tail."""

    if isinstance(request, str):
        request = LogicalDispatchRequest.parse(request)
    if not isinstance(request, LogicalDispatchRequest):
        raise DispatchPlanningError(
            "request must be a LogicalDispatchRequest or logical request string"
        )
    if not isinstance(profile, QualifiedChunkProfile):
        raise DispatchPlanningError("profile must be a QualifiedChunkProfile")
    max_dispatches = _exact_positive_integer(
        max_physical_dispatches, name="max_physical_dispatches"
    )
    if request.prefix != profile.request_prefix:
        raise DispatchPlanningError(
            f"request prefix {request.prefix!r} does not match profile prefix "
            f"{profile.request_prefix!r}"
        )

    remaining = request.units
    dispatch_count = 0
    counts: list[tuple[int, int]] = []
    for size in profile.qualified_chunk_sizes:
        count, remaining = divmod(remaining, size)
        dispatch_count += count
        if dispatch_count > max_dispatches:
            raise DispatchPlanningError(
                "planned physical dispatch count exceeds max_physical_dispatches"
            )
        counts.append((size, count))

    physical: list[PhysicalDispatch] = []
    offset = 0
    for size, count in counts:
        for _ in range(count):
            physical.append(
                PhysicalDispatch(
                    ordinal=len(physical), logical_offset=offset, units=size
                )
            )
            offset += size

    tail = TailDispatch(logical_offset=offset, units=remaining)
    if offset + remaining != request.units:
        raise AssertionError("dispatch planner produced an inconsistent plan")
    return DispatchPlan(
        schema_version=DISPATCH_PLAN_SCHEMA_VERSION_V1,
        profile_id=profile.profile_id,
        request=request,
        physical_dispatches=tuple(physical),
        tail=tail,
    )
