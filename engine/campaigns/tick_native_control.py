"""Active tick-native control-plane contracts.

The production control plane carries absolute ticks all the way to its owner,
dispatch, checkpoint, reporting, and resource contracts.  It deliberately has
no refill cadence or generation-shaped compatibility authority.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final, Mapping

from .tick_native import MAX_SIGNED_INT64, TICK_NATIVE_EXECUTION_ROUTE_V1


TICK_NATIVE_CONTROL_SCHEMA_VERSION = "eiel-tick-native-control-v2"
DEFAULT_RESIDENT_OWNER_SPAN_TICKS_V1 = 1_120
DEFAULT_NATIVE_DISPATCH_WINDOW_TICKS_V1 = 1_120
DEFAULT_SOURCE_WINDOW_TICKS_V1 = 160
DEFAULT_EVENT_WINDOW_CAPACITY_V1 = 65_536

# These are source-derived bounds for the amended physical-B136 profile.  The
# repository-wide default above remains a compatibility default for routes
# whose exact profile has not been admitted by this contract.
SOURCE_DERIVED_BASE_EVENT_ROWS_PER_SOURCE_WINDOW_V1: Final[int] = 41_164
SOURCE_DERIVED_MAX_EVENT_ROWS_PER_ORDINARY_BIRTH_V1: Final[int] = 2
EVENT_ROW_STORAGE_BYTES_V1: Final[int] = 141
PHYSICAL_B136_BATCH_CELLS_V1: Final[int] = 136
PHYSICAL_B136_POPULATION_V1: Final[int] = 256
PHYSICAL_B136_GRID_V1: Final[tuple[int, int]] = (32, 32)
PHYSICAL_B136_RESIDENT_CAPACITY_V1: Final[int] = 512
PHYSICAL_B136_MAX_BIRTHS_PER_TICK_V1: Final[int] = 3
PHYSICAL_B136_SOURCE_WINDOW_TICKS_V1: Final[int] = 160
FORBIDDEN_ACTIVE_GENERATION_FIELDS_V1 = frozenset(
    {
        "generations",
        "ticks_per_generation",
        "logical_generation",
        "logical_generation_count",
        "physical_generation",
        "physical_window_generations",
        "generation_count",
        "generation_calls",
        "steps_per_generation",
        "digest_every_generation",
        "checkpoint_milestones",
        "generation_boundary",
        "generation_close",
        "start_generation",
        "next_generation",
        "turnover_next_generation",
        "max_internal_lifecycle_segments_per_dispatch",
        "internal_lifecycle_interval_ticks",
        "internal_lifecycle_schedule",
        "refill_chunk_ticks",
        "legacy_turnover_interval_ticks",
    }
)


class TickNativeControlError(ValueError):
    """A tick-native control contract is malformed or ambiguous."""


def _positive(value: Any, *, name: str, maximum: int = MAX_SIGNED_INT64) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise TickNativeControlError(f"{name} must be a positive integer")
    if value > maximum:
        raise TickNativeControlError(f"{name} exceeds signed int64")
    return int(value)


def source_derived_event_window_capacity_v1(
    *,
    max_births_per_tick: int,
    source_window_ticks: int,
    base_event_rows_per_source_window: int = (
        SOURCE_DERIVED_BASE_EVENT_ROWS_PER_SOURCE_WINDOW_V1
    ),
    max_event_rows_per_ordinary_birth: int = (
        SOURCE_DERIVED_MAX_EVENT_ROWS_PER_ORDINARY_BIRTH_V1
    ),
) -> int:
    """Return the bounded current-window event-row requirement.

    The source-derived base is one bounded 160-tick source profile.  Additional
    ordinary births are charged only to the same bounded window; no horizon,
    cursor, sink frontier, or cumulative history value participates.
    """

    births = _positive(max_births_per_tick, name="max_births_per_tick")
    ticks = _positive(source_window_ticks, name="source_window_ticks")
    base_rows = _positive(
        base_event_rows_per_source_window,
        name="base_event_rows_per_source_window",
    )
    rows_per_birth = _positive(
        max_event_rows_per_ordinary_birth,
        name="max_event_rows_per_ordinary_birth",
    )
    return base_rows + (births - 1) * ticks * rows_per_birth


def physical_b136_event_window_capacity_v1() -> int:
    """Return the frozen exact source-derived B136 event-window capacity."""

    return source_derived_event_window_capacity_v1(
        max_births_per_tick=PHYSICAL_B136_MAX_BIRTHS_PER_TICK_V1,
        source_window_ticks=PHYSICAL_B136_SOURCE_WINDOW_TICKS_V1,
    )


def _cursor(value: Any, *, name: str, total_ticks: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TickNativeControlError(f"{name} must be an integer")
    if value < 0 or value > total_ticks:
        raise TickNativeControlError(f"{name} must be within total_ticks")
    return int(value)


def _schedule(
    values: Any,
    *,
    name: str,
    total_ticks: int,
    allow_empty: bool = True,
) -> tuple[int, ...]:
    if values is None:
        return ()
    if not isinstance(values, (list, tuple)):
        raise TickNativeControlError(f"{name} must be an integer array")
    result = tuple(values)
    if not allow_empty and not result:
        raise TickNativeControlError(f"{name} must not be empty")
    previous = 0
    for value in result:
        _positive(value, name=f"{name} entry")
        if value <= previous or value > total_ticks:
            raise TickNativeControlError(
                f"{name} must be strictly increasing and inside total_ticks"
            )
        previous = value
    return result


@dataclass(frozen=True, slots=True)
class TickSpanV1:
    """A half-open active owner or dispatch span ``[start_tick, end_tick)``."""

    start_tick: int
    end_tick: int

    def __post_init__(self) -> None:
        if isinstance(self.start_tick, bool) or not isinstance(self.start_tick, int):
            raise TickNativeControlError("span.start_tick must be an integer")
        if isinstance(self.end_tick, bool) or not isinstance(self.end_tick, int):
            raise TickNativeControlError("span.end_tick must be an integer")
        if self.start_tick < 0 or self.end_tick <= self.start_tick:
            raise TickNativeControlError("tick spans must be non-empty and ordered")
        if self.end_tick > MAX_SIGNED_INT64:
            raise TickNativeControlError("span.end_tick exceeds signed int64")

    @property
    def tick_count(self) -> int:
        return self.end_tick - self.start_tick

    def payload(self) -> dict[str, int]:
        return {
            "start_tick": self.start_tick,
            "end_tick": self.end_tick,
            "tick_count": self.tick_count,
        }


@dataclass(frozen=True, slots=True)
class TickNativeControlV1:
    """Canonical active control coordinates for one tick-native trajectory."""

    total_ticks: int
    start_next_tick: int = 0
    resident_owner_span_ticks: int = DEFAULT_RESIDENT_OWNER_SPAN_TICKS_V1
    native_dispatch_window_ticks: int = DEFAULT_NATIVE_DISPATCH_WINDOW_TICKS_V1
    report_every_ticks: int | None = None
    science_sample_interval_ticks: int | None = None
    reducer_every_ticks: int | None = None
    archive_every_ticks: int | None = None
    checkpoint_ticks: tuple[int, ...] = ()
    checkpoint_every_ticks: int | None = None
    event_window_capacity: int = DEFAULT_EVENT_WINDOW_CAPACITY_V1
    source_window_ticks: int = DEFAULT_SOURCE_WINDOW_TICKS_V1
    schema_version: str = TICK_NATIVE_CONTROL_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != TICK_NATIVE_CONTROL_SCHEMA_VERSION:
            raise TickNativeControlError(
                f"schema_version must be {TICK_NATIVE_CONTROL_SCHEMA_VERSION!r}"
            )
        total = _positive(self.total_ticks, name="total_ticks")
        _cursor(self.start_next_tick, name="start_next_tick", total_ticks=total)
        owner = _positive(
            self.resident_owner_span_ticks, name="resident_owner_span_ticks"
        )
        dispatch = _positive(
            self.native_dispatch_window_ticks,
            name="native_dispatch_window_ticks",
        )
        source = _positive(self.source_window_ticks, name="source_window_ticks")
        if dispatch > owner:
            raise TickNativeControlError(
                "native_dispatch_window_ticks cannot exceed resident_owner_span_ticks"
            )
        if source > owner:
            raise TickNativeControlError(
                "source_window_ticks cannot exceed resident_owner_span_ticks"
            )
        for name in (
            "report_every_ticks",
            "science_sample_interval_ticks",
            "reducer_every_ticks",
            "archive_every_ticks",
            "checkpoint_every_ticks",
        ):
            value = getattr(self, name)
            if value is not None:
                _positive(value, name=name)
        _positive(self.event_window_capacity, name="event_window_capacity")
        _schedule(
            self.checkpoint_ticks,
            name="checkpoint_ticks",
            total_ticks=total,
        )
        if self.checkpoint_every_ticks is not None and self.checkpoint_ticks:
            raise TickNativeControlError(
                "checkpoint_every_ticks and checkpoint_ticks are alternative authorities"
            )

    @classmethod
    def from_execution(cls, execution: Mapping[str, Any]) -> "TickNativeControlV1":
        if not isinstance(execution, Mapping):
            raise TickNativeControlError("tick-native execution must be an object")
        nested = execution.get("tick_native_control", {})
        if nested is None:
            nested = {}
        if not isinstance(nested, Mapping):
            raise TickNativeControlError("execution.tick_native_control must be an object")
        allowed = {
            "schema_version",
            "time_authority",
            "total_ticks",
            "next_absolute_tick",
            "schedules",
            "resident_owner_span_ticks",
            "native_dispatch_window_ticks",
            "report_every_ticks",
            "science_sample_interval_ticks",
            "reducer_every_ticks",
            "archive_every_ticks",
            "checkpoint_ticks",
            "checkpoint_every_ticks",
            "event_window_capacity",
            "source_window_ticks",
        }
        unknown = sorted(set(nested) - allowed)
        if unknown:
            raise TickNativeControlError(
                f"execution.tick_native_control contains unsupported fields: {unknown}"
            )

        def pick(name: str, default: Any = None) -> Any:
            if name in nested:
                return nested[name]
            return execution.get(name, default)

        total = nested.get("total_ticks", execution.get("total_ticks"))
        start_next_tick = nested.get(
            "next_absolute_tick", execution.get("start_next_tick", 0)
        )
        schedules = nested.get("schedules", {})
        if schedules is not None and not isinstance(schedules, Mapping):
            raise TickNativeControlError("tick_native_control.schedules must be an object")
        if isinstance(schedules, Mapping):
            schedule_values = {
                "checkpoint_ticks": schedules.get("checkpoint_ticks"),
                "report_every_ticks": schedules.get("report_ticks"),
                "science_sample_interval_ticks": schedules.get(
                    "science_sample_ticks"
                ),
                "reducer_every_ticks": schedules.get("reducer_ticks"),
                "archive_every_ticks": schedules.get("archive_ticks"),
            }
        else:
            schedule_values = {}
        def pick_schedule(name: str, default: Any = None) -> Any:
            if name in nested:
                return nested[name]
            if name in execution:
                return execution[name]
            values = schedule_values.get(name)
            if values in (None, []):
                return default
            # The canonical payload also carries the direct interval fields;
            # this fallback is only for hand-authored control payloads.
            return values

        if pick("checkpoint_ticks", None) is None:
            checkpoint_ticks = schedule_values.get("checkpoint_ticks")
        else:
            checkpoint_ticks = pick("checkpoint_ticks", ())
        checkpoint_every = pick("checkpoint_every_ticks")
        if checkpoint_ticks in (None, ()) and checkpoint_every is None:
            checkpoint_ticks = (total,)
        return cls(
            total_ticks=total,
            start_next_tick=start_next_tick,
            resident_owner_span_ticks=pick(
                "resident_owner_span_ticks", DEFAULT_RESIDENT_OWNER_SPAN_TICKS_V1
            ),
            native_dispatch_window_ticks=pick(
                "native_dispatch_window_ticks", DEFAULT_NATIVE_DISPATCH_WINDOW_TICKS_V1
            ),
            report_every_ticks=pick_schedule("report_every_ticks"),
            science_sample_interval_ticks=pick_schedule(
                "science_sample_interval_ticks"
            ),
            reducer_every_ticks=pick_schedule("reducer_every_ticks"),
            archive_every_ticks=pick_schedule("archive_every_ticks"),
            checkpoint_ticks=tuple(checkpoint_ticks or ()),
            checkpoint_every_ticks=checkpoint_every,
            event_window_capacity=pick(
                "event_window_capacity", DEFAULT_EVENT_WINDOW_CAPACITY_V1
            ),
            source_window_ticks=pick(
                "source_window_ticks", DEFAULT_SOURCE_WINDOW_TICKS_V1
            ),
            schema_version=nested.get(
                "schema_version", TICK_NATIVE_CONTROL_SCHEMA_VERSION
            ),
        )

    @property
    def resolved_checkpoint_ticks(self) -> tuple[int, ...]:
        if self.checkpoint_every_ticks is not None:
            values = tuple(
                range(
                    self.checkpoint_every_ticks,
                    self.total_ticks + 1,
                    self.checkpoint_every_ticks,
                )
            )
        else:
            values = self.checkpoint_ticks
        return tuple(sorted({*values, self.total_ticks}))

    def _spans(self, *, start_next_tick: int, window_ticks: int) -> tuple[TickSpanV1, ...]:
        cursor = _cursor(
            start_next_tick, name="start_next_tick", total_ticks=self.total_ticks
        )
        result: list[TickSpanV1] = []
        while cursor < self.total_ticks:
            end = min(self.total_ticks, cursor + window_ticks)
            result.append(TickSpanV1(cursor, end))
            cursor = end
        return tuple(result)

    def owner_spans(self, *, start_next_tick: int | None = None) -> tuple[TickSpanV1, ...]:
        return self._spans(
            start_next_tick=self.start_next_tick
            if start_next_tick is None
            else start_next_tick,
            window_ticks=self.resident_owner_span_ticks,
        )

    def dispatch_spans(self, *, start_next_tick: int | None = None) -> tuple[TickSpanV1, ...]:
        return self._spans(
            start_next_tick=self.start_next_tick
            if start_next_tick is None
            else start_next_tick,
            window_ticks=self.native_dispatch_window_ticks,
        )

    def payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "time_authority": "absolute_tick_v1",
            "total_ticks": self.total_ticks,
            "next_absolute_tick": self.start_next_tick,
            "resident_owner_span_ticks": self.resident_owner_span_ticks,
            "native_dispatch_window_ticks": self.native_dispatch_window_ticks,
            "source_window_ticks": self.source_window_ticks,
            "event_window_capacity": self.event_window_capacity,
            "schedules": {
                "checkpoint_ticks": list(self.resolved_checkpoint_ticks),
                "report_ticks": _periodic_schedule(
                    self.report_every_ticks, total_ticks=self.total_ticks
                ),
                "science_sample_ticks": _periodic_schedule(
                    self.science_sample_interval_ticks, total_ticks=self.total_ticks
                ),
                "reducer_ticks": _periodic_schedule(
                    self.reducer_every_ticks, total_ticks=self.total_ticks
                ),
                "archive_ticks": _periodic_schedule(
                    self.archive_every_ticks, total_ticks=self.total_ticks
                ),
            },
            # Direct fields are retained alongside the expanded schedules so
            # the contract is lossless and can be revalidated after JSON
            # round-tripping.
            "report_every_ticks": self.report_every_ticks,
            "science_sample_interval_ticks": self.science_sample_interval_ticks,
            "reducer_every_ticks": self.reducer_every_ticks,
            "archive_every_ticks": self.archive_every_ticks,
            # A fixed schedule is canonicalized with its terminal frontier;
            # periodic schedules retain only the interval as their authority.
            "checkpoint_ticks": (
                []
                if self.checkpoint_every_ticks is not None
                else list(self.resolved_checkpoint_ticks)
            ),
            "checkpoint_every_ticks": self.checkpoint_every_ticks,
        }


def _periodic_schedule(interval: int | None, *, total_ticks: int) -> list[int]:
    if interval is None:
        return [total_ticks]
    return list(range(interval, total_ticks + 1, interval)) + (
        [] if total_ticks % interval == 0 else [total_ticks]
    )


def tick_native_control_from_execution(
    execution: Mapping[str, Any],
) -> TickNativeControlV1:
    """Parse the active control contract, including legacy input aliases only."""

    try:
        return TickNativeControlV1.from_execution(execution)
    except (TypeError, TickNativeControlError) as exc:
        if isinstance(exc, TickNativeControlError):
            raise
        raise TickNativeControlError(str(exc)) from exc


def assert_tick_native_plan_payload_v1(payload: Mapping[str, Any]) -> None:
    """Fail closed if active plan fields regain generation-shaped authority."""

    def walk_active_payload(
        value: Any,
        *,
        path: tuple[str, ...],
    ) -> list[str]:
        found: list[str] = []
        if isinstance(value, Mapping):
            for key, child in value.items():
                name = str(key)
                child_path = (*path, name)
                if name in FORBIDDEN_ACTIVE_GENERATION_FIELDS_V1:
                    found.append(".".join(child_path))
                found.extend(walk_active_payload(child, path=child_path))
        elif isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                found.extend(
                    walk_active_payload(child, path=(*path, f"[{index}]"))
                )
        return found

    execution = payload.get("execution")
    if not isinstance(execution, Mapping):
        raise TickNativeControlError("tick-native plan is missing execution")
    if execution.get("execution_route_identity") != TICK_NATIVE_EXECUTION_ROUTE_V1:
        raise TickNativeControlError("plan is not tick-native")
    forbidden = sorted(walk_active_payload(execution, path=("execution",)))
    if forbidden:
        raise TickNativeControlError(
            f"tick-native active execution contains forbidden generation authority: {forbidden}"
        )
    control = execution.get("tick_native_control")
    if not isinstance(control, Mapping):
        raise TickNativeControlError("tick-native plan is missing tick_native_control")
    parsed = TickNativeControlV1.from_execution(execution)
    if control != parsed.payload():
        raise TickNativeControlError("tick_native_control is not canonical")
    estimate = payload.get("resource_estimate")
    if isinstance(estimate, Mapping):
        forbidden_estimate = sorted(
            walk_active_payload(estimate, path=("resource_estimate",))
        )
        if forbidden_estimate:
            raise TickNativeControlError(
                "tick-native resource estimate contains forbidden generation authority: "
                f"{forbidden_estimate}"
            )


__all__ = [
    "DEFAULT_EVENT_WINDOW_CAPACITY_V1",
    "DEFAULT_NATIVE_DISPATCH_WINDOW_TICKS_V1",
    "DEFAULT_RESIDENT_OWNER_SPAN_TICKS_V1",
    "DEFAULT_SOURCE_WINDOW_TICKS_V1",
    "EVENT_ROW_STORAGE_BYTES_V1",
    "FORBIDDEN_ACTIVE_GENERATION_FIELDS_V1",
    "PHYSICAL_B136_BATCH_CELLS_V1",
    "PHYSICAL_B136_GRID_V1",
    "PHYSICAL_B136_MAX_BIRTHS_PER_TICK_V1",
    "PHYSICAL_B136_POPULATION_V1",
    "PHYSICAL_B136_RESIDENT_CAPACITY_V1",
    "PHYSICAL_B136_SOURCE_WINDOW_TICKS_V1",
    "SOURCE_DERIVED_MAX_EVENT_ROWS_PER_ORDINARY_BIRTH_V1",
    "SOURCE_DERIVED_BASE_EVENT_ROWS_PER_SOURCE_WINDOW_V1",
    "TICK_NATIVE_CONTROL_SCHEMA_VERSION",
    "TickNativeControlError",
    "TickNativeControlV1",
    "TickSpanV1",
    "assert_tick_native_plan_payload_v1",
    "physical_b136_event_window_capacity_v1",
    "source_derived_event_window_capacity_v1",
    "tick_native_control_from_execution",
]
