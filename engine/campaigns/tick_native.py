"""Canonical absolute-tick contracts plus explicit legacy import helpers."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from .canonical import canonical_bytes as campaign_canonical_bytes


TICK_SCHEDULE_SCHEMA_VERSION = "eiel-tick-schedule-v1"
ABSOLUTE_TICK_SCHEDULE_SCHEMA_VERSION = "eiel-absolute-tick-schedule-v2"
TICK_CURSOR_SCHEMA_VERSION = "eiel-tick-cursor-v1"
TICK_NATIVE_EXECUTION_REQUEST_SCHEMA_VERSION = "eiel-tick-native-execution-request-v2"
TICK_NATIVE_EXECUTION_ROUTE_V1 = "tick_native_execution_route_v1"
MAX_SIGNED_INT64 = (1 << 63) - 1


class TickNativeError(ValueError):
    """A tick-native compatibility input is invalid or ambiguous."""


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise TickNativeError(f"{name} must be a positive integer")
    return value


def _positive_int64(value: object, name: str) -> int:
    value = _positive_int(value, name)
    if value > MAX_SIGNED_INT64:
        raise TickNativeError(f"{name} exceeds signed int64")
    return value


def _checked_total_ticks(generations: object, ticks_per_generation: object) -> int:
    generations = _positive_int(generations, "generations")
    ticks_per_generation = _positive_int(ticks_per_generation, "ticks_per_generation")
    if generations > MAX_SIGNED_INT64 // ticks_per_generation:
        raise TickNativeError("generations * ticks_per_generation exceeds signed int64")
    return generations * ticks_per_generation


@dataclass(frozen=True, slots=True)
class TickScheduleV1:
    """Canonical logical schedule projected from a legacy G/T horizon."""

    total_ticks: int
    generations: int | None = None
    ticks_per_generation: int | None = None
    science_sample_interval_ticks: int | None = None
    legacy_turnover_interval_ticks: int | None = None
    schema_version: str = TICK_SCHEDULE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != TICK_SCHEDULE_SCHEMA_VERSION:
            raise TickNativeError(
                f"schema_version must be {TICK_SCHEDULE_SCHEMA_VERSION!r}"
            )
        _positive_int64(self.total_ticks, "total_ticks")
        if (self.generations is None) != (self.ticks_per_generation is None):
            raise TickNativeError(
                "generations and ticks_per_generation must be supplied together"
            )
        if self.generations is not None:
            projected = _checked_total_ticks(self.generations, self.ticks_per_generation)
            if projected != self.total_ticks:
                raise TickNativeError("total_ticks disagrees with legacy G/T projection")
        if self.science_sample_interval_ticks is not None:
            _positive_int64(
                self.science_sample_interval_ticks,
                "science_sample_interval_ticks",
            )
        if self.legacy_turnover_interval_ticks is not None:
            interval = _positive_int(
                self.legacy_turnover_interval_ticks,
                "legacy_turnover_interval_ticks",
            )
            if interval > self.total_ticks or self.total_ticks % interval:
                raise TickNativeError(
                    "legacy_turnover_interval_ticks must divide total_ticks"
                )

    @classmethod
    def from_legacy(
        cls,
        generations: int,
        ticks_per_generation: int,
        *,
        science_sample_interval_ticks: int | None = None,
        legacy_turnover_interval_ticks: int | None = None,
    ) -> "TickScheduleV1":
        return cls(
            total_ticks=_checked_total_ticks(generations, ticks_per_generation),
            generations=generations,
            ticks_per_generation=ticks_per_generation,
            science_sample_interval_ticks=science_sample_interval_ticks,
            legacy_turnover_interval_ticks=legacy_turnover_interval_ticks,
        )

    def payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "total_ticks": self.total_ticks,
            "legacy_projection": (
                {
                    "generations": self.generations,
                    "ticks_per_generation": self.ticks_per_generation,
                }
                if self.generations is not None
                else None
            ),
            "science_sample_interval_ticks": self.science_sample_interval_ticks,
            "legacy_turnover_interval_ticks": self.legacy_turnover_interval_ticks,
        }

    def canonical_bytes(self) -> bytes:
        return campaign_canonical_bytes(self.payload())


@dataclass(frozen=True, slots=True)
class AbsoluteTickScheduleV2:
    """Canonical schedule for routes whose only time authority is absolute ticks."""

    total_ticks: int
    science_sample_interval_ticks: int | None = None
    schema_version: str = ABSOLUTE_TICK_SCHEDULE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ABSOLUTE_TICK_SCHEDULE_SCHEMA_VERSION:
            raise TickNativeError(
                f"schema_version must be {ABSOLUTE_TICK_SCHEDULE_SCHEMA_VERSION!r}"
            )
        _positive_int64(self.total_ticks, "total_ticks")
        if self.science_sample_interval_ticks is not None:
            _positive_int64(
                self.science_sample_interval_ticks,
                "science_sample_interval_ticks",
            )

    def payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "total_ticks": self.total_ticks,
            "science_sample_interval_ticks": self.science_sample_interval_ticks,
        }

    def canonical_bytes(self) -> bytes:
        return campaign_canonical_bytes(self.payload())


@dataclass(frozen=True, slots=True)
class TickCursorV1:
    """Half-open committed trajectory cursor: ``[0, next_tick)``."""

    total_ticks: int
    next_tick: int = 0
    schema_version: str = TICK_CURSOR_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != TICK_CURSOR_SCHEMA_VERSION:
            raise TickNativeError(
                f"schema_version must be {TICK_CURSOR_SCHEMA_VERSION!r}"
            )
        _positive_int64(self.total_ticks, "total_ticks")
        if isinstance(self.next_tick, bool) or not isinstance(self.next_tick, int):
            raise TickNativeError("next_tick must be an integer")
        if not 0 <= self.next_tick <= self.total_ticks:
            raise TickNativeError("next_tick must be within the closed cursor range")

    @property
    def absolute_tick(self) -> int | None:
        """The tick to execute now, or ``None`` once the horizon is committed."""

        return None if self.next_tick == self.total_ticks else self.next_tick

    @property
    def committed_range(self) -> range:
        return range(self.next_tick)

    def advance(self) -> "TickCursorV1":
        if self.absolute_tick is None:
            raise TickNativeError("cannot advance a completed tick cursor")
        return TickCursorV1(total_ticks=self.total_ticks, next_tick=self.next_tick + 1)


@dataclass(frozen=True, slots=True)
class TickNativeExecutionRequestV1:
    """Provider-facing direct-tick request with no G/T horizon authority."""

    total_ticks: int
    start_next_tick: int = 0
    runtime_partition_ticks: int | None = None
    science_sample_interval_ticks: int | None = None
    report_every_ticks: int | None = None
    checkpoint_every_ticks: int | None = None
    checkpoint_ticks: tuple[int, ...] = ()
    deliberate_restore_tick: int | None = None
    deliberate_restore_ticks: tuple[int, ...] = ()
    schema_version: str = TICK_NATIVE_EXECUTION_REQUEST_SCHEMA_VERSION
    resident_owner_span_ticks: int = 1_120
    native_dispatch_window_ticks: int = 1_120
    reducer_every_ticks: int | None = None
    archive_every_ticks: int | None = None
    event_window_capacity: int = 65_536
    source_window_ticks: int = 160

    def __post_init__(self) -> None:
        if self.schema_version != TICK_NATIVE_EXECUTION_REQUEST_SCHEMA_VERSION:
            raise TickNativeError(
                "schema_version must be "
                f"{TICK_NATIVE_EXECUTION_REQUEST_SCHEMA_VERSION!r}"
            )
        _positive_int64(self.total_ticks, "total_ticks")
        if isinstance(self.start_next_tick, bool) or not isinstance(
            self.start_next_tick, int
        ):
            raise TickNativeError("start_next_tick must be an integer")
        if not 0 <= self.start_next_tick <= self.total_ticks:
            raise TickNativeError("start_next_tick must be within the cursor range")
        for name in (
            "runtime_partition_ticks",
            "science_sample_interval_ticks",
            "report_every_ticks",
            "checkpoint_every_ticks",
            "resident_owner_span_ticks",
            "native_dispatch_window_ticks",
            "reducer_every_ticks",
            "archive_every_ticks",
            "event_window_capacity",
            "source_window_ticks",
        ):
            value = getattr(self, name)
            if value is not None:
                _positive_int64(value, name)
        if self.native_dispatch_window_ticks > self.resident_owner_span_ticks:
            raise TickNativeError(
                "native_dispatch_window_ticks cannot exceed resident_owner_span_ticks"
            )
        if self.source_window_ticks > self.resident_owner_span_ticks:
            raise TickNativeError(
                "source_window_ticks cannot exceed resident_owner_span_ticks"
            )
        previous = 0
        for value in self.checkpoint_ticks:
            _positive_int64(value, "checkpoint_ticks entry")
            if value <= previous or value > self.total_ticks:
                raise TickNativeError(
                    "checkpoint_ticks must be strictly increasing and inside total_ticks"
                )
            previous = value
        if self.deliberate_restore_tick is not None and self.deliberate_restore_ticks:
            if self.deliberate_restore_ticks != (self.deliberate_restore_tick,):
                raise TickNativeError(
                    "legacy deliberate_restore_tick conflicts with deliberate_restore_ticks"
                )
        restore_ticks = self.canonical_deliberate_restore_ticks
        previous_restore = 0
        for restore_tick in restore_ticks:
            _positive_int64(restore_tick, "deliberate_restore_ticks entry")
            if restore_tick <= previous_restore or restore_tick >= self.total_ticks:
                raise TickNativeError(
                    "deliberate_restore_ticks must be strictly increasing before total_ticks"
                )
            restore_is_scheduled = (
                restore_tick in self.checkpoint_ticks
                or (
                    self.checkpoint_every_ticks is not None
                    and restore_tick % self.checkpoint_every_ticks == 0
                )
            )
            if not restore_is_scheduled:
                raise TickNativeError(
                    "deliberate_restore_ticks must name exact checkpoint ticks"
                )
            previous_restore = restore_tick
        if self.checkpoint_every_ticks is not None and self.checkpoint_ticks:
            raise TickNativeError(
                "checkpoint_every_ticks and checkpoint_ticks are alternative authorities"
            )

    @classmethod
    def from_execution(cls, execution: Any) -> "TickNativeExecutionRequestV1":
        if not isinstance(execution, dict):
            raise TickNativeError("tick-native execution request must be an object")
        if execution.get("execution_route_identity") != TICK_NATIVE_EXECUTION_ROUTE_V1:
            raise TickNativeError("execution_route_identity is not tick-native")
        for legacy_name in (
            "generations",
            "ticks_per_generation",
            "logical_generation_count",
            "physical_window_generations",
            "generation_count",
            "max_internal_lifecycle_segments_per_dispatch",
            "internal_lifecycle_schedule",
            "internal_lifecycle_interval_ticks",
            "refill_chunk_ticks",
            "legacy_turnover_interval_ticks",
        ):
            if execution.get(legacy_name) is not None:
                raise TickNativeError(
                    f"tick-native execution request must not use {legacy_name} authority"
                )

        def optional_positive(name: str) -> int | None:
            value = execution.get(name)
            if value in (None, 0):
                return None
            return value

        raw_checkpoint_ticks = execution.get("checkpoint_ticks", ())
        if not isinstance(raw_checkpoint_ticks, (list, tuple)) or any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in raw_checkpoint_ticks
        ):
            raise TickNativeError("checkpoint_ticks must be an integer array")
        raw_restore_ticks = execution.get("deliberate_restore_ticks", ())
        if not isinstance(raw_restore_ticks, (list, tuple)) or any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in raw_restore_ticks
        ):
            raise TickNativeError("deliberate_restore_ticks must be an integer array")

        control = execution.get("tick_native_control", {})
        if control is None:
            control = {}
        if not isinstance(control, dict):
            raise TickNativeError("tick_native_control must be an object")

        def control_value(name: str, default: Any = None) -> Any:
            return control.get(name, execution.get(name, default))

        return cls(
            total_ticks=execution.get("total_ticks"),
            start_next_tick=execution.get("start_next_tick", 0),
            runtime_partition_ticks=optional_positive("runtime_partition_ticks"),
            science_sample_interval_ticks=optional_positive(
                "science_sample_interval_ticks"
            ),
            report_every_ticks=optional_positive("report_every_ticks"),
            checkpoint_every_ticks=optional_positive("checkpoint_every_ticks"),
            checkpoint_ticks=tuple(raw_checkpoint_ticks),
            deliberate_restore_tick=optional_positive("deliberate_restore_tick"),
            deliberate_restore_ticks=tuple(raw_restore_ticks),
            resident_owner_span_ticks=control_value("resident_owner_span_ticks", 1_120),
            native_dispatch_window_ticks=control_value(
                "native_dispatch_window_ticks", 1_120
            ),
            reducer_every_ticks=control_value("reducer_every_ticks"),
            archive_every_ticks=control_value("archive_every_ticks"),
            event_window_capacity=control_value("event_window_capacity", 65_536),
            source_window_ticks=control_value("source_window_ticks", 160),
        )

    @property
    def canonical_deliberate_restore_ticks(self) -> tuple[int, ...]:
        if self.deliberate_restore_ticks:
            return self.deliberate_restore_ticks
        return (
            ()
            if self.deliberate_restore_tick is None
            else (self.deliberate_restore_tick,)
        )

    def control_payload(self) -> dict[str, Any]:
        # Keep one lossless control object on every active tick-native request.
        # The local import avoids making the legacy compatibility module depend
        # on this contract at import time.
        from .tick_native_control import TickNativeControlV1

        return TickNativeControlV1(
            total_ticks=self.total_ticks,
            start_next_tick=self.start_next_tick,
            resident_owner_span_ticks=self.resident_owner_span_ticks,
            native_dispatch_window_ticks=self.native_dispatch_window_ticks,
            report_every_ticks=self.report_every_ticks,
            science_sample_interval_ticks=self.science_sample_interval_ticks,
            reducer_every_ticks=self.reducer_every_ticks,
            archive_every_ticks=self.archive_every_ticks,
            checkpoint_ticks=self.checkpoint_ticks,
            checkpoint_every_ticks=self.checkpoint_every_ticks,
            event_window_capacity=self.event_window_capacity,
            source_window_ticks=self.source_window_ticks,
        ).payload()

    def payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "execution_route_identity": TICK_NATIVE_EXECUTION_ROUTE_V1,
            "time_authority": "absolute_tick_v1",
            "total_ticks": self.total_ticks,
            "start_next_tick": self.start_next_tick,
            "runtime_partition_ticks": self.runtime_partition_ticks,
            "science_sample_interval_ticks": self.science_sample_interval_ticks,
            "report_every_ticks": self.report_every_ticks,
            "checkpoint_every_ticks": self.checkpoint_every_ticks,
            "checkpoint_ticks": list(self.checkpoint_ticks),
            "deliberate_restore_tick": self.deliberate_restore_tick,
            "deliberate_restore_ticks": list(
                self.canonical_deliberate_restore_ticks
            ),
            "resident_owner_span_ticks": self.resident_owner_span_ticks,
            "native_dispatch_window_ticks": self.native_dispatch_window_ticks,
            "reducer_every_ticks": self.reducer_every_ticks,
            "archive_every_ticks": self.archive_every_ticks,
            "event_window_capacity": self.event_window_capacity,
            "source_window_ticks": self.source_window_ticks,
            "tick_native_control": self.control_payload(),
        }


@dataclass(frozen=True, slots=True)
class RunNameV1:
    """Strict parser/renderer for legacy G/T and tick-native campaign names."""

    batch_cells: int
    world_side: int
    initial_population: int
    resident_capacity: int
    device: str
    total_ticks: int | None = None
    generations: int | None = None
    ticks_per_generation: int | None = None
    science_sample_interval_ticks: int | None = None
    checkpoint_interval_ticks: int | None = None
    report_interval_ticks: int | None = None
    repeats: int | None = None
    stable_suffix: str = ""

    def __post_init__(self) -> None:
        for name in (
            "batch_cells",
            "world_side",
            "initial_population",
            "resident_capacity",
        ):
            _positive_int(getattr(self, name), name)
        if self.device not in {"cpu", "gpu"}:
            raise TickNativeError("device must be cpu or gpu")
        legacy = self.generations is not None or self.ticks_per_generation is not None
        native = self.total_ticks is not None
        if legacy == native:
            raise TickNativeError("run name must use exactly one legacy or tick-native horizon")
        if legacy:
            _checked_total_ticks(self.generations, self.ticks_per_generation)
        else:
            _positive_int64(self.total_ticks, "total_ticks")
        for name in (
            "science_sample_interval_ticks",
            "checkpoint_interval_ticks",
            "report_interval_ticks",
        ):
            value = getattr(self, name)
            if value is not None:
                _positive_int64(value, name)
        if self.repeats is not None:
            _positive_int(self.repeats, "repeats")
        if not isinstance(self.stable_suffix, str) or (
            self.stable_suffix
            and not re.fullmatch(
                r"-[A-Za-z0-9][A-Za-z0-9._-]*", self.stable_suffix
            )
        ):
            raise TickNativeError("stable_suffix must be a hyphen-prefixed stable token")

    @property
    def is_legacy(self) -> bool:
        return self.generations is not None

    @property
    def resolved_total_ticks(self) -> int:
        if self.is_legacy:
            return _checked_total_ticks(self.generations, self.ticks_per_generation)
        assert self.total_ticks is not None
        return self.total_ticks

    def render(self) -> str:
        value = (
            f"b{self.batch_cells}w{self.world_side}p{self.initial_population}"
            f"c{self.resident_capacity}"
        )
        if self.is_legacy:
            value += f"g{self.generations}t{self.ticks_per_generation}"
        else:
            value += f"t{self.total_ticks}"
        for prefix, interval in (
            ("sci", self.science_sample_interval_ticks),
            ("ckp", self.checkpoint_interval_ticks),
            ("report", self.report_interval_ticks),
            ("x", self.repeats),
        ):
            if interval is not None:
                value += f"{prefix}{interval}"
        return f"{value}{self.device}{self.stable_suffix}"


_RUN_PREFIX = re.compile(r"^b(?P<b>[1-9][0-9]*)w(?P<w>[1-9][0-9]*)p(?P<p>[1-9][0-9]*)c(?P<c>[1-9][0-9]*)(?P<tail>.+)$")
_TIME_TOKEN = re.compile(r"^(?:g(?P<g>[1-9][0-9]*)t(?P<tpg>[1-9][0-9]*)|t(?P<total>[1-9][0-9]*))(?P<rest>.*)$")
_OPTIONAL_TOKEN = re.compile(r"^(?P<kind>sci|ckp|report|x)(?P<value>[1-9][0-9]*)(?P<rest>.*)$")


def parse_run_name_v1(value: str) -> RunNameV1:
    """Parse a supported name exactly; malformed mixed horizons fail closed."""

    if not isinstance(value, str):
        raise TickNativeError("run name must be a string")
    prefix = _RUN_PREFIX.fullmatch(value)
    if prefix is None:
        raise TickNativeError("run name does not have the required b/w/p/c prefix")
    horizon = _TIME_TOKEN.fullmatch(prefix.group("tail"))
    if horizon is None:
        raise TickNativeError("run name lacks an unambiguous g/t or total-t horizon")
    rest = horizon.group("rest")
    options: dict[str, int] = {}
    while rest and not rest.startswith(("cpu", "gpu", "-")):
        match = _OPTIONAL_TOKEN.fullmatch(rest)
        if match is None or match.group("kind") in options:
            raise TickNativeError("run name has malformed or duplicate schedule tokens")
        options[match.group("kind")] = int(match.group("value"))
        rest = match.group("rest")
    device_match = re.match(r"^(cpu|gpu)(.*)$", rest)
    if device_match is None:
        raise TickNativeError("run name must end its schedule with cpu or gpu")
    stable_suffix = device_match.group(2)
    parsed = RunNameV1(
        batch_cells=int(prefix.group("b")),
        world_side=int(prefix.group("w")),
        initial_population=int(prefix.group("p")),
        resident_capacity=int(prefix.group("c")),
        device=device_match.group(1),
        total_ticks=(int(horizon.group("total")) if horizon.group("total") else None),
        generations=(int(horizon.group("g")) if horizon.group("g") else None),
        ticks_per_generation=(int(horizon.group("tpg")) if horizon.group("tpg") else None),
        science_sample_interval_ticks=options.get("sci"),
        checkpoint_interval_ticks=options.get("ckp"),
        report_interval_ticks=options.get("report"),
        repeats=options.get("x"),
        stable_suffix=stable_suffix,
    )
    if parsed.render() != value:
        raise TickNativeError("run name schedule tokens are not in canonical order")
    return parsed


def assert_tick_native_plan_payload_v1(payload: Any) -> None:
    """Apply the active-plan firewall through the canonical tick-native API."""

    from .tick_native_control import (
        assert_tick_native_plan_payload_v1 as assert_control_payload,
    )

    assert_control_payload(payload)


__all__ = [
    "ABSOLUTE_TICK_SCHEDULE_SCHEMA_VERSION",
    "AbsoluteTickScheduleV2",
    "MAX_SIGNED_INT64",
    "RunNameV1",
    "TICK_CURSOR_SCHEMA_VERSION",
    "TICK_NATIVE_EXECUTION_REQUEST_SCHEMA_VERSION",
    "TICK_NATIVE_EXECUTION_ROUTE_V1",
    "TICK_SCHEDULE_SCHEMA_VERSION",
    "TickCursorV1",
    "TickNativeExecutionRequestV1",
    "TickNativeError",
    "TickScheduleV1",
    "assert_tick_native_plan_payload_v1",
    "parse_run_name_v1",
]
