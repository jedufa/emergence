"""Observation providers for live/replay WRC reduction."""

from __future__ import annotations

from contextlib import AbstractContextManager, contextmanager
import csv
import io
import json
import math
from pathlib import Path
import re
import sys
from typing import Iterable, Iterator, Mapping, Protocol, TextIO
import zipfile

from .config import ReducerConfig
from .model import Observation, SourceLocator


class ObservationProviderError(ValueError):
    """A persisted observation source is malformed or incomplete."""


class TextResource(Protocol):
    @property
    def uri(self) -> str: ...

    def open_text(self) -> AbstractContextManager[TextIO]: ...


class PathTextResource:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    @property
    def uri(self) -> str:
        return str(self.path.resolve())

    @contextmanager
    def open_text(self) -> Iterator[TextIO]:
        try:
            with self.path.open(newline="", encoding="utf-8", errors="replace") as handle:
                yield handle
        except OSError as exc:
            raise ObservationProviderError(f"cannot read {self.path}: {exc}") from exc


class ZipTextResource:
    def __init__(self, archive: str | Path, member: str) -> None:
        self.archive = Path(archive)
        self.member = str(member)

    @property
    def uri(self) -> str:
        return f"zip://{self.archive.resolve()}!/{self.member}"

    @contextmanager
    def open_text(self) -> Iterator[TextIO]:
        try:
            with zipfile.ZipFile(self.archive) as bundle:
                raw = bundle.open(self.member, "r")
                wrapper = io.TextIOWrapper(raw, encoding="utf-8", errors="replace", newline="")
                try:
                    yield wrapper
                finally:
                    wrapper.close()
        except (OSError, KeyError, zipfile.BadZipFile) as exc:
            raise ObservationProviderError(
                f"cannot read {self.archive}!/{self.member}: {exc}"
            ) from exc


class MetricSelector:
    """Deterministic include/exclude filter for wide metrics sources."""

    def __init__(
        self,
        config: ReducerConfig,
        *,
        required_metrics: Iterable[str] = (),
    ) -> None:
        self.required_metrics = {str(name) for name in required_metrics if str(name)}
        self.includes = tuple(re.compile(pattern) for pattern in config.metric_include_patterns)
        self.excludes = tuple(re.compile(pattern) for pattern in config.metric_exclude_patterns)

    def accepts(self, name: str) -> bool:
        if name in self.required_metrics:
            return True
        if name in {"generation", "tick", "step", "sequence"}:
            return False
        if self.includes and not any(pattern.search(name) for pattern in self.includes):
            return False
        if any(pattern.search(name) for pattern in self.excludes):
            return False
        return True


def _raise_csv_field_limit() -> None:
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


def _numeric(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        result = float(value)
        return result if math.isfinite(result) else None
    if not isinstance(value, str):
        return None
    raw = value.strip()
    if not raw or raw.lower() in {"true", "false", "none", "null", "nan", "inf", "-inf"}:
        return None
    if raw[:1] in {"{", "[", '"'}:
        return None
    try:
        result = float(raw)
    except ValueError:
        return None
    return result if math.isfinite(result) else None


class MetricsCsvObservationProvider:
    """Replay generation/tick metrics through the source-neutral reducer core."""

    def __init__(
        self,
        resource: TextResource,
        *,
        ticks_per_observation: int = 1,
        config: ReducerConfig | None = None,
        required_metrics: Iterable[str] = (),
        allow_partial_last_row: bool = True,
    ) -> None:
        if int(ticks_per_observation) < 1:
            raise ObservationProviderError("ticks_per_observation must be positive")
        self.resource = resource
        self.ticks_per_observation = int(ticks_per_observation)
        self.config = config or ReducerConfig()
        self.selector = MetricSelector(
            self.config, required_metrics=required_metrics
        )
        self.allow_partial_last_row = bool(allow_partial_last_row)

    def _parse_row(
        self,
        *,
        data_index: int,
        row: Mapping[str | None, str | None],
    ) -> Observation:
        row_number = data_index + 2
        try:
            generation_raw = row.get("generation")
            sequence = (
                int(float(generation_raw))
                if generation_raw not in (None, "")
                else data_index
            )
            explicit_tick = _numeric(row.get("tick"))
            tick = (
                int(explicit_tick)
                if explicit_tick is not None
                else (sequence + 1) * self.ticks_per_observation
            )
            metrics: dict[str, float] = {}
            for name, raw in row.items():
                if name is None or not self.selector.accepts(name):
                    continue
                value = _numeric(raw)
                if value is not None:
                    metrics[name] = value
            if not metrics:
                raise ObservationProviderError(
                    f"metrics row has no numeric values: "
                    f"{self.resource.uri}:{row_number}"
                )
            return Observation(
                sequence=sequence,
                tick=tick,
                metrics=metrics,
                source=SourceLocator(
                    kind="metrics_csv",
                    uri=self.resource.uri,
                    row_number=row_number,
                ),
            )
        except (TypeError, ValueError) as exc:
            raise ObservationProviderError(
                f"invalid metrics row {self.resource.uri}:{row_number}: {exc}"
            ) from exc

    def __iter__(self) -> Iterator[Observation]:
        _raise_csv_field_limit()
        with self.resource.open_text() as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                raise ObservationProviderError(
                    f"metrics source has no header: {self.resource.uri}"
                )
            observed = 0
            pending: tuple[int, Mapping[str | None, str | None]] | None = None
            try:
                for data_index, row in enumerate(reader):
                    if pending is not None:
                        observation = self._parse_row(
                            data_index=pending[0],
                            row=pending[1],
                        )
                        observed += 1
                        yield observation
                    pending = (data_index, row)
            except csv.Error as exc:
                # A truncated process may leave only its final physical CSV row
                # malformed. Earlier malformed material remains fail-closed.
                if not self.allow_partial_last_row or observed == 0:
                    raise ObservationProviderError(
                        f"malformed metrics CSV {self.resource.uri}: {exc}"
                    ) from exc
                pending = None
            if pending is not None:
                try:
                    observation = self._parse_row(
                        data_index=pending[0],
                        row=pending[1],
                    )
                except ObservationProviderError:
                    if not self.allow_partial_last_row:
                        raise
                else:
                    observed += 1
                    yield observation
            if observed == 0:
                raise ObservationProviderError(
                    f"metrics source is header-only or nonnumeric: {self.resource.uri}"
                )



class ChronicleObservationProvider:
    """Replay compact Chronicle generation summaries through the same core.

    Chronicle generation summaries are deliberately compact.  Their output is
    comparable with live or metrics replay for overlapping measurement names,
    but it must not be mistaken for full metrics coverage.
    """

    def __init__(
        self,
        resource: TextResource,
        *,
        config: ReducerConfig | None = None,
        required_metrics: Iterable[str] = (),
    ) -> None:
        self.resource = resource
        self.config = config or ReducerConfig()
        self.selector = MetricSelector(
            self.config, required_metrics=required_metrics
        )

    def __iter__(self) -> Iterator[Observation]:
        observed = 0
        with self.resource.open_text() as handle:
            for line_number, raw in enumerate(handle, start=1):
                if not raw.strip():
                    continue
                try:
                    event = json.loads(raw)
                except json.JSONDecodeError as exc:
                    raise ObservationProviderError(
                        f"malformed Chronicle event {self.resource.uri}:{line_number}: {exc}"
                    ) from exc
                if not isinstance(event, Mapping):
                    raise ObservationProviderError(
                        f"Chronicle event is not an object: {self.resource.uri}:{line_number}"
                    )
                if event.get("event_type") != "generation_summary":
                    continue
                measurements = event.get("measurements")
                if not isinstance(measurements, Mapping):
                    continue
                metrics = {
                    str(name): value
                    for name, raw_value in measurements.items()
                    if self.selector.accepts(str(name))
                    and (value := _numeric(raw_value)) is not None
                }
                if not metrics:
                    continue
                generation = int(event.get("generation", observed))
                tick = int(event.get("step", generation))
                observed += 1
                yield Observation(
                    sequence=generation,
                    tick=tick,
                    metrics=metrics,
                    source=SourceLocator(
                        kind="chronicle_generation_summary",
                        uri=self.resource.uri,
                        row_number=line_number,
                        event_id=str(event.get("event_id") or "") or None,
                    ),
                )
        if observed == 0:
            raise ObservationProviderError(
                f"Chronicle has no generation summaries: {self.resource.uri}"
            )


__all__ = [
    "ChronicleObservationProvider",
    "MetricSelector",
    "MetricsCsvObservationProvider",
    "ObservationProviderError",
    "PathTextResource",
    "TextResource",
    "ZipTextResource",
]
