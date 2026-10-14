"""Typed, source-neutral observations consumed by the WRC reducer."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Any, Mapping

from .canonical import canonical_sha256


JsonScalar = str | int | float | bool | None


class ReducerInputError(ValueError):
    """An observation or identity violates the deterministic reducer contract."""


@dataclass(frozen=True, slots=True)
class SourceLocator:
    """Pointer back to an authoritative or retained source record."""

    kind: str
    uri: str
    row_number: int | None = None
    event_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.kind, str) or not self.kind:
            raise ReducerInputError("source kind must be nonempty")
        if not isinstance(self.uri, str) or not self.uri:
            raise ReducerInputError("source URI must be nonempty")
        if self.row_number is not None and self.row_number < 1:
            raise ReducerInputError("source row number must be positive")
        if self.event_id is not None and not self.event_id:
            raise ReducerInputError("source event ID must be nonempty when present")

    def payload(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class Observation:
    """One normalized measurement boundary.

    `sequence` is the strictly increasing source order.  `tick` is the logical
    simulation tick represented by the boundary and may equal sequence for
    sources that have no stronger time coordinate.
    """

    sequence: int
    tick: int
    metrics: Mapping[str, float]
    source: SourceLocator

    def __post_init__(self) -> None:
        if isinstance(self.sequence, bool) or int(self.sequence) < 0:
            raise ReducerInputError("observation sequence must be a nonnegative int")
        if isinstance(self.tick, bool) or int(self.tick) < 0:
            raise ReducerInputError("observation tick must be a nonnegative int")
        if not isinstance(self.metrics, Mapping):
            raise ReducerInputError("observation metrics must be a mapping")
        for name, value in self.metrics.items():
            if not isinstance(name, str) or not name:
                raise ReducerInputError("metric names must be nonempty strings")
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ReducerInputError(f"metric {name!r} is not numeric")
            if not math.isfinite(float(value)):
                raise ReducerInputError(f"metric {name!r} is nonfinite")

    def payload(self) -> dict[str, Any]:
        return {
            "sequence": int(self.sequence),
            "tick": int(self.tick),
            "metrics": {name: float(value) for name, value in sorted(self.metrics.items())},
            "source": self.source.payload(),
        }


@dataclass(frozen=True, slots=True)
class WorldIdentity:
    """Identity shared by all observations in one simulated world."""

    world_id: str
    parameter_id: str
    parameter_value: JsonScalar
    seed: int
    ticks_per_observation: int = 1
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.world_id:
            raise ReducerInputError("world_id must be nonempty")
        if not self.parameter_id:
            raise ReducerInputError("parameter_id must be nonempty")
        if isinstance(self.seed, bool):
            raise ReducerInputError("seed must be an integer")
        if int(self.ticks_per_observation) < 1:
            raise ReducerInputError("ticks_per_observation must be positive")
        if isinstance(self.parameter_value, float) and not math.isfinite(
            self.parameter_value
        ):
            raise ReducerInputError("parameter value must be finite")
        if not isinstance(self.metadata, Mapping):
            raise ReducerInputError("world metadata must be a mapping")

    def payload(self) -> dict[str, Any]:
        return {
            "world_id": self.world_id,
            "parameter_id": self.parameter_id,
            "parameter_value": self.parameter_value,
            "seed": int(self.seed),
            "ticks_per_observation": int(self.ticks_per_observation),
            "metadata": dict(sorted(self.metadata.items())),
        }

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.payload())


@dataclass(frozen=True, slots=True)
class CandidatePoint:
    sequence: int
    tick: int
    value: float
    score: float
    reasons: tuple[str, ...]
    source: SourceLocator

    def payload(self) -> dict[str, Any]:
        return {
            "sequence": int(self.sequence),
            "tick": int(self.tick),
            "value": float(self.value),
            "score": float(self.score),
            "reasons": list(self.reasons),
            "source": self.source.payload(),
        }


__all__ = [
    "CandidatePoint",
    "JsonScalar",
    "Observation",
    "ReducerInputError",
    "SourceLocator",
    "WorldIdentity",
]
