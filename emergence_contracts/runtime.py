"""C-030 execution schedule and first-uncommitted run cursor contract."""

from __future__ import annotations

from dataclasses import dataclass

from ._validation import require_instance, require_int, require_optional_positive_int
from .errors import contract_invalid
from .ids import SchemaVersion


CONTRACT_ID = "C-030"
CONTRACT_OWNER = "Contracts; authority implemented by Runtime"
API_VERSION = SchemaVersion("c030_v1")
SCHEDULE_SCHEMA_VERSION = SchemaVersion("execution_schedule_v1")
CURSOR_SCHEMA_VERSION = SchemaVersion("run_cursor_v1")


@dataclass(frozen=True, slots=True)
class LegacyGenerationProjectionV1:
    generations: int
    ticks_per_generation: int

    def __post_init__(self) -> None:
        require_int(self.generations, field="generations", minimum=1)
        require_int(
            self.ticks_per_generation, field="ticks_per_generation", minimum=1
        )

    @property
    def total_ticks(self) -> int:
        value = self.generations * self.ticks_per_generation
        return require_int(value, field="legacy_projection.total_ticks", minimum=1)

    def to_dict(self) -> dict[str, int]:
        return {
            "generations": self.generations,
            "ticks_per_generation": self.ticks_per_generation,
        }


@dataclass(frozen=True, slots=True)
class TurnoverScheduleV1:
    turnover_ticks: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.turnover_ticks, tuple):
            raise contract_invalid(
                "turnover ticks must be a tuple", field="turnover_ticks"
            )
        for tick in self.turnover_ticks:
            require_int(tick, field="turnover_ticks", minimum=0)
        if tuple(sorted(set(self.turnover_ticks))) != self.turnover_ticks:
            raise contract_invalid(
                "turnover ticks must be sorted and unique", field="turnover_ticks"
            )

    def to_dict(self) -> dict[str, list[int]]:
        return {"turnover_ticks": list(self.turnover_ticks)}


@dataclass(frozen=True, slots=True)
class ExecutionScheduleV1:
    total_ticks: int
    start_next_tick: int = 0
    runtime_partition_ticks: int | None = None
    science_sample_interval_ticks: int | None = None
    report_every_ticks: int | None = None
    checkpoint_every_ticks: int | None = None
    turnover_schedule: TurnoverScheduleV1 | None = None
    legacy_projection: LegacyGenerationProjectionV1 | None = None
    schema_version: SchemaVersion = SCHEDULE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        total = require_int(self.total_ticks, field="total_ticks", minimum=1)
        require_int(self.start_next_tick, field="start_next_tick", maximum=total)
        for field_name in (
            "runtime_partition_ticks",
            "science_sample_interval_ticks",
            "report_every_ticks",
            "checkpoint_every_ticks",
        ):
            require_optional_positive_int(getattr(self, field_name), field=field_name)
        if self.turnover_schedule is not None:
            require_instance(
                self.turnover_schedule, TurnoverScheduleV1, field="turnover_schedule"
            )
            if any(tick >= total for tick in self.turnover_schedule.turnover_ticks):
                raise contract_invalid(
                    "turnover tick must be inside the run interval",
                    field="turnover_schedule",
                )
        if self.legacy_projection is not None:
            require_instance(
                self.legacy_projection,
                LegacyGenerationProjectionV1,
                field="legacy_projection",
            )
            if self.legacy_projection.total_ticks != total:
                raise contract_invalid(
                    "legacy projection total does not match total_ticks",
                    field="legacy_projection",
                )
        if self.schema_version != SCHEDULE_SCHEMA_VERSION:
            raise contract_invalid(
                "unsupported execution schedule schema", field="schema_version"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": str(self.schema_version),
            "total_ticks": self.total_ticks,
            "start_next_tick": self.start_next_tick,
            "runtime_partition_ticks": self.runtime_partition_ticks,
            "science_sample_interval_ticks": self.science_sample_interval_ticks,
            "report_every_ticks": self.report_every_ticks,
            "checkpoint_every_ticks": self.checkpoint_every_ticks,
            "turnover_schedule": (
                None if self.turnover_schedule is None else self.turnover_schedule.to_dict()
            ),
            "legacy_projection": (
                None if self.legacy_projection is None else self.legacy_projection.to_dict()
            ),
        }

    @classmethod
    def from_dict(cls, value: object) -> "ExecutionScheduleV1":
        if not isinstance(value, dict):
            raise contract_invalid("schedule must be an object", field="schedule")
        expected = {
            "schema_version",
            "total_ticks",
            "start_next_tick",
            "runtime_partition_ticks",
            "science_sample_interval_ticks",
            "report_every_ticks",
            "checkpoint_every_ticks",
            "turnover_schedule",
            "legacy_projection",
        }
        if set(value) != expected:
            raise contract_invalid("schedule has an invalid shape", field="schedule")
        turnover = value["turnover_schedule"]
        legacy = value["legacy_projection"]
        return cls(
            total_ticks=value["total_ticks"],
            start_next_tick=value["start_next_tick"],
            runtime_partition_ticks=value["runtime_partition_ticks"],
            science_sample_interval_ticks=value["science_sample_interval_ticks"],
            report_every_ticks=value["report_every_ticks"],
            checkpoint_every_ticks=value["checkpoint_every_ticks"],
            turnover_schedule=(
                None
                if turnover is None
                else TurnoverScheduleV1(tuple(turnover["turnover_ticks"]))
            ),
            legacy_projection=(
                None
                if legacy is None
                else LegacyGenerationProjectionV1(
                    generations=legacy["generations"],
                    ticks_per_generation=legacy["ticks_per_generation"],
                )
            ),
            schema_version=SchemaVersion(value["schema_version"]),
        )


@dataclass(frozen=True, slots=True)
class RunCursorV1:
    total_ticks: int
    next_tick: int
    schema_version: SchemaVersion = CURSOR_SCHEMA_VERSION

    def __post_init__(self) -> None:
        total = require_int(self.total_ticks, field="total_ticks", minimum=1)
        require_int(self.next_tick, field="next_tick", maximum=total)
        if self.schema_version != CURSOR_SCHEMA_VERSION:
            raise contract_invalid("unsupported cursor schema", field="schema_version")

    @property
    def complete(self) -> bool:
        return self.next_tick == self.total_ticks

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": str(self.schema_version),
            "total_ticks": self.total_ticks,
            "next_tick": self.next_tick,
        }

    @classmethod
    def from_dict(cls, value: object) -> "RunCursorV1":
        if not isinstance(value, dict) or set(value) != {
            "schema_version",
            "total_ticks",
            "next_tick",
        }:
            raise contract_invalid("cursor has an invalid shape", field="cursor")
        return cls(
            total_ticks=value["total_ticks"],
            next_tick=value["next_tick"],
            schema_version=SchemaVersion(value["schema_version"]),
        )
