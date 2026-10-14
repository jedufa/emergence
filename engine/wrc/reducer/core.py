"""Streaming four-stage trajectory reduction.

The implementation stores bounded candidates and online statistics per metric.
It never needs to retain the full raw trajectory.  Post-run providers replay
saved observations through this exact same code path, which makes live and
retrospective results directly comparable.
"""

from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

from .canonical import canonical_sha256, pretty_json

from .config import (
    REDUCER_VERSION,
    STATE_SCHEMA_VERSION,
    WORLD_RESULT_SCHEMA_VERSION,
    ReducerConfig,
)
from .model import CandidatePoint, Observation, ReducerInputError, SourceLocator, WorldIdentity


@dataclass(slots=True)
class _Point:
    sequence: int
    tick: int
    value: float
    source: SourceLocator

    def payload(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence,
            "tick": self.tick,
            "value": self.value,
            "source": self.source.payload(),
        }

    @classmethod
    def from_payload(cls, value: Mapping[str, Any]) -> "_Point":
        source = value.get("source")
        if not isinstance(source, Mapping):
            raise ReducerInputError("point source is missing")
        return cls(
            sequence=int(value["sequence"]),
            tick=int(value["tick"]),
            value=float(value["value"]),
            source=SourceLocator(
                kind=str(source["kind"]),
                uri=str(source["uri"]),
                row_number=(
                    int(source["row_number"])
                    if source.get("row_number") is not None
                    else None
                ),
                event_id=(
                    str(source["event_id"])
                    if source.get("event_id") is not None
                    else None
                ),
            ),
        )


@dataclass(slots=True)
class _CandidateRecord:
    point: _Point
    score: float
    reasons: set[str]

    def payload(self) -> dict[str, Any]:
        return {
            **self.point.payload(),
            "score": self.score,
            "reasons": sorted(self.reasons),
        }

    @classmethod
    def from_payload(cls, value: Mapping[str, Any]) -> "_CandidateRecord":
        return cls(
            point=_Point.from_payload(value),
            score=float(value["score"]),
            reasons={str(item) for item in value.get("reasons", [])},
        )


class _TopCandidates:
    def __init__(self, budget: int) -> None:
        self.budget = int(budget)
        self._items: dict[int, _CandidateRecord] = {}

    def add(self, point: _Point, *, score: float, reason: str) -> None:
        if not math.isfinite(score):
            return
        existing = self._items.get(point.sequence)
        if existing is not None:
            existing.score = max(existing.score, float(score))
            existing.reasons.add(reason)
        else:
            self._items[point.sequence] = _CandidateRecord(
                point=point,
                score=float(score),
                reasons={reason},
            )
        while len(self._items) > self.budget:
            worst_sequence = min(
                self._items,
                key=lambda sequence: (
                    self._items[sequence].score,
                    -self._items[sequence].point.sequence,
                ),
            )
            del self._items[worst_sequence]

    def values(self) -> list[_CandidateRecord]:
        return [self._items[key] for key in sorted(self._items)]

    def payload(self) -> list[dict[str, Any]]:
        return [item.payload() for item in self.values()]

    @classmethod
    def from_payload(
        cls, budget: int, value: Iterable[Mapping[str, Any]]
    ) -> "_TopCandidates":
        result = cls(budget)
        for row in value:
            item = _CandidateRecord.from_payload(row)
            result._items[item.point.sequence] = item
        return result


@dataclass(slots=True)
class _StableSpan:
    start_sequence: int
    start_tick: int
    end_sequence: int
    end_tick: int
    observation_count: int
    maximum_step_score: float

    def payload(self) -> dict[str, Any]:
        return {
            "start_sequence": self.start_sequence,
            "start_tick": self.start_tick,
            "end_sequence": self.end_sequence,
            "end_tick": self.end_tick,
            "observation_count": self.observation_count,
            "maximum_step_score": self.maximum_step_score,
        }

    @classmethod
    def from_payload(cls, value: Mapping[str, Any]) -> "_StableSpan":
        return cls(
            start_sequence=int(value["start_sequence"]),
            start_tick=int(value["start_tick"]),
            end_sequence=int(value["end_sequence"]),
            end_tick=int(value["end_tick"]),
            observation_count=int(value["observation_count"]),
            maximum_step_score=float(value["maximum_step_score"]),
        )


class _TopStableSpans:
    def __init__(self, budget: int) -> None:
        self.budget = int(budget)
        self._items: list[_StableSpan] = []

    def add(self, span: _StableSpan) -> None:
        if span.observation_count < 2:
            return
        self._items.append(span)
        self._items.sort(
            key=lambda item: (
                -item.observation_count,
                item.start_sequence,
                item.end_sequence,
            )
        )
        del self._items[self.budget :]

    def values(self) -> list[_StableSpan]:
        return sorted(self._items, key=lambda item: item.start_sequence)

    def payload(self) -> list[dict[str, Any]]:
        return [item.payload() for item in self.values()]

    @classmethod
    def from_payload(
        cls, budget: int, value: Iterable[Mapping[str, Any]]
    ) -> "_TopStableSpans":
        result = cls(budget)
        result._items = [_StableSpan.from_payload(row) for row in value]
        return result


def _change_score(
    reference: float,
    current: float,
    *,
    relative_threshold: float,
    absolute_threshold: float,
    scale_floor: float,
    running_scale: float = 0.0,
) -> tuple[bool, float, float, float]:
    absolute = abs(current - reference)
    scale = max(abs(reference), abs(current), abs(running_scale), scale_floor)
    relative = absolute / scale
    relative_ratio = (
        relative / relative_threshold if relative_threshold > 0.0 else 0.0
    )
    absolute_ratio = (
        absolute / absolute_threshold if absolute_threshold > 0.0 else 0.0
    )
    crossed = (
        (relative_threshold > 0.0 and relative >= relative_threshold)
        or (absolute_threshold > 0.0 and absolute >= absolute_threshold)
    )
    return crossed, max(relative_ratio, absolute_ratio), absolute, relative


class _MetricState:
    def __init__(self, config: ReducerConfig) -> None:
        self.config = config
        self.count = 0
        self.first: _Point | None = None
        self.last: _Point | None = None
        self.previous: _Point | None = None
        self.minimum: _Point | None = None
        self.maximum: _Point | None = None
        self.drift_anchor: _Point | None = None
        self.previous_delta: float | None = None
        self.mean = 0.0
        self.m2 = 0.0
        self.mean_abs = 0.0
        self.sum_x = 0.0
        self.sum_y = 0.0
        self.sum_xx = 0.0
        self.sum_xy = 0.0
        self.area = 0.0
        self.total_variation = 0.0
        self.maximum_step_change = 0.0
        self.first_nonzero_sequence: int | None = None
        self.first_nonzero_tick: int | None = None
        self.first_significant_sequence: int | None = None
        self.first_significant_tick: int | None = None
        self.local_candidates = _TopCandidates(config.local_candidate_budget)
        self.turning_candidates = _TopCandidates(config.turning_candidate_budget)
        self.cumulative_candidates = _TopCandidates(
            config.cumulative_candidate_budget
        )
        self.baseline_candidates = _TopCandidates(config.baseline_candidate_budget)
        self.outlier_candidates = _TopCandidates(config.outlier_candidate_budget)
        self.stable_spans = _TopStableSpans(config.stable_span_budget)
        self.stable_span_count = 0
        self._stable_start: _Point | None = None
        self._stable_observation_count = 0
        self._stable_maximum_step_score = 0.0

    @property
    def standard_deviation(self) -> float:
        return math.sqrt(self.m2 / (self.count - 1)) if self.count > 1 else 0.0

    def _close_stable_span(self) -> None:
        if (
            self._stable_start is not None
            and self.previous is not None
            and self._stable_observation_count >= 2
        ):
            self.stable_span_count += 1
            self.stable_spans.add(
                _StableSpan(
                    start_sequence=self._stable_start.sequence,
                    start_tick=self._stable_start.tick,
                    end_sequence=self.previous.sequence,
                    end_tick=self.previous.tick,
                    observation_count=self._stable_observation_count,
                    maximum_step_score=self._stable_maximum_step_score,
                )
            )
        self._stable_start = None
        self._stable_observation_count = 0
        self._stable_maximum_step_score = 0.0

    def ingest(self, observation: Observation, value: float) -> None:
        point = _Point(
            sequence=observation.sequence,
            tick=observation.tick,
            value=float(value),
            source=observation.source,
        )
        if self.count == 0:
            self.count = 1
            self.first = point
            self.last = point
            self.previous = point
            self.minimum = point
            self.maximum = point
            self.drift_anchor = point
            self.mean = value
            self.mean_abs = abs(value)
            x = float(point.tick)
            self.sum_x = x
            self.sum_y = value
            self.sum_xx = x * x
            self.sum_xy = x * value
            if abs(value) > self.config.zero_epsilon:
                self.first_nonzero_sequence = point.sequence
                self.first_nonzero_tick = point.tick
            return

        assert self.first is not None
        assert self.previous is not None
        assert self.drift_anchor is not None
        assert self.minimum is not None
        assert self.maximum is not None

        previous = self.previous
        running_scale = max(self.mean_abs, self.config.scale_floor)
        local_crossed, local_score, absolute_delta, _ = _change_score(
            previous.value,
            point.value,
            relative_threshold=self.config.local_relative_threshold,
            absolute_threshold=self.config.local_absolute_threshold,
            scale_floor=self.config.scale_floor,
            running_scale=running_scale,
        )
        stable_crossed, stable_score, _, _ = _change_score(
            previous.value,
            point.value,
            relative_threshold=self.config.stable_relative_threshold,
            absolute_threshold=self.config.stable_absolute_threshold,
            scale_floor=self.config.scale_floor,
            running_scale=running_scale,
        )
        is_stable = not stable_crossed
        if is_stable:
            if self._stable_start is None:
                self._stable_start = previous
                self._stable_observation_count = 2
                self._stable_maximum_step_score = stable_score
            else:
                self._stable_observation_count += 1
                self._stable_maximum_step_score = max(
                    self._stable_maximum_step_score, stable_score
                )
        else:
            self._close_stable_span()

        if local_crossed:
            self.local_candidates.add(
                point, score=local_score, reason="stage2:local_change"
            )
            if self.first_significant_sequence is None:
                self.first_significant_sequence = point.sequence
                self.first_significant_tick = point.tick

        delta = point.value - previous.value
        if (
            self.previous_delta is not None
            and self.previous_delta != 0.0
            and delta != 0.0
            and (self.previous_delta > 0.0) != (delta > 0.0)
        ):
            curvature = min(abs(self.previous_delta), abs(delta))
            scale = max(
                abs(previous.value),
                abs(point.value),
                running_scale,
                self.config.scale_floor,
            )
            turning_score = curvature / scale
            if turning_score >= self.config.stable_relative_threshold:
                self.turning_candidates.add(
                    previous,
                    score=(
                        turning_score
                        / max(
                            self.config.stable_relative_threshold,
                            self.config.scale_floor,
                        )
                    ),
                    reason="stage2:local_extremum",
                )

        cumulative_crossed, cumulative_score, _, _ = _change_score(
            self.drift_anchor.value,
            point.value,
            relative_threshold=self.config.cumulative_relative_threshold,
            absolute_threshold=self.config.cumulative_absolute_threshold,
            scale_floor=self.config.scale_floor,
            running_scale=running_scale,
        )
        if cumulative_crossed:
            self.cumulative_candidates.add(
                point,
                score=cumulative_score,
                reason="stage3:cumulative_drift",
            )
            self.drift_anchor = point
            if self.first_significant_sequence is None:
                self.first_significant_sequence = point.sequence
                self.first_significant_tick = point.tick

        baseline_crossed, baseline_score, _, _ = _change_score(
            self.first.value,
            point.value,
            relative_threshold=self.config.baseline_relative_threshold,
            absolute_threshold=self.config.baseline_absolute_threshold,
            scale_floor=self.config.scale_floor,
            running_scale=running_scale,
        )
        if baseline_crossed:
            self.baseline_candidates.add(
                point,
                score=baseline_score,
                reason="stage3:baseline_drift",
            )

        if self.count >= self.config.outlier_min_samples:
            standard_deviation = self.standard_deviation
            if standard_deviation > self.config.scale_floor:
                z_score = abs(point.value - self.mean) / standard_deviation
                if z_score >= self.config.outlier_z_threshold:
                    self.outlier_candidates.add(
                        point,
                        score=z_score / self.config.outlier_z_threshold,
                        reason="stage4:online_outlier",
                    )

        if point.value < self.minimum.value:
            self.minimum = point
        if point.value > self.maximum.value:
            self.maximum = point

        dx = float(point.tick - previous.tick)
        if dx < 0:
            raise ReducerInputError("metric tick order regressed")
        self.area += 0.5 * (previous.value + point.value) * dx
        self.total_variation += absolute_delta
        self.maximum_step_change = max(self.maximum_step_change, absolute_delta)

        self.count += 1
        delta_mean = point.value - self.mean
        self.mean += delta_mean / self.count
        self.m2 += delta_mean * (point.value - self.mean)
        self.mean_abs += (abs(point.value) - self.mean_abs) / self.count
        x = float(point.tick)
        self.sum_x += x
        self.sum_y += point.value
        self.sum_xx += x * x
        self.sum_xy += x * point.value

        if (
            self.first_nonzero_sequence is None
            and abs(point.value) > self.config.zero_epsilon
        ):
            self.first_nonzero_sequence = point.sequence
            self.first_nonzero_tick = point.tick

        self.previous_delta = delta
        self.previous = point
        self.last = point

    def _slope(self) -> float:
        denominator = self.count * self.sum_xx - self.sum_x * self.sum_x
        if self.count < 2 or abs(denominator) <= self.config.scale_floor:
            return 0.0
        return (self.count * self.sum_xy - self.sum_x * self.sum_y) / denominator

    def _relative(self, value: float, reference: float) -> float | None:
        denominator = abs(reference)
        if denominator <= self.config.scale_floor:
            return None
        return value / denominator

    def _merged_candidates(self) -> list[CandidatePoint]:
        assert self.first is not None
        assert self.last is not None
        assert self.minimum is not None
        assert self.maximum is not None
        merged: dict[int, _CandidateRecord] = {}

        def add(point: _Point, score: float, reason: str) -> None:
            current = merged.get(point.sequence)
            if current is None:
                merged[point.sequence] = _CandidateRecord(
                    point=point, score=score, reasons={reason}
                )
            else:
                current.score = max(current.score, score)
                current.reasons.add(reason)

        add(self.first, float("inf"), "stage1:anchor_first")
        add(self.last, float("inf"), "stage1:anchor_last")
        add(self.minimum, float("inf"), "stage4:minimum")
        add(self.maximum, float("inf"), "stage4:maximum")
        for bucket in (
            self.local_candidates,
            self.turning_candidates,
            self.cumulative_candidates,
            self.baseline_candidates,
            self.outlier_candidates,
        ):
            for item in bucket.values():
                for reason in item.reasons:
                    add(item.point, item.score, reason)

        mandatory = {
            self.first.sequence,
            self.last.sequence,
            self.minimum.sequence,
            self.maximum.sequence,
        }
        if len(merged) > self.config.max_selected_points_per_metric:
            optional = [
                item for sequence, item in merged.items() if sequence not in mandatory
            ]
            optional.sort(key=lambda item: (-item.score, item.point.sequence))
            allowed_optional = max(
                0, self.config.max_selected_points_per_metric - len(mandatory)
            )
            keep = mandatory | {
                item.point.sequence for item in optional[:allowed_optional]
            }
            merged = {sequence: item for sequence, item in merged.items() if sequence in keep}

        output: list[CandidatePoint] = []
        for sequence in sorted(merged):
            item = merged[sequence]
            score = item.score if math.isfinite(item.score) else 1.0e308
            output.append(
                CandidatePoint(
                    sequence=item.point.sequence,
                    tick=item.point.tick,
                    value=item.point.value,
                    score=score,
                    reasons=tuple(sorted(item.reasons)),
                    source=item.point.source,
                )
            )
        return output

    def finalize(self) -> dict[str, Any]:
        if self.count < 1 or self.first is None or self.last is None:
            raise ReducerInputError("cannot finalize an empty metric")
        self._close_stable_span()
        assert self.minimum is not None
        assert self.maximum is not None
        duration = self.last.tick - self.first.tick
        net_change = self.last.value - self.first.value
        time_weighted_mean = self.area / duration if duration > 0 else self.mean
        range_value = self.maximum.value - self.minimum.value
        scale = max(
            abs(self.mean),
            abs(self.first.value),
            abs(self.last.value),
            self.config.scale_floor,
        )
        relative_range = range_value / scale
        relative_variation = self.total_variation / scale
        path_efficiency = (
            abs(net_change) / self.total_variation
            if self.total_variation > self.config.scale_floor
            else 1.0
        )
        selected = self._merged_candidates()
        maximum_outlier_score = max(
            (item.score for item in self.outlier_candidates.values()), default=0.0
        )
        information_score = (
            math.log1p(relative_range)
            + math.log1p(relative_variation)
            + min(maximum_outlier_score, 20.0) / 10.0
            + min(len(selected), 16) / 32.0
        )
        return {
            "count": self.count,
            "first": self.first.payload(),
            "last": self.last.payload(),
            "minimum": self.minimum.payload(),
            "maximum": self.maximum.payload(),
            "mean": self.mean,
            "standard_deviation": self.standard_deviation,
            "time_weighted_mean": time_weighted_mean,
            "area": self.area,
            "duration_ticks": duration,
            "net_change": net_change,
            "relative_net_change": self._relative(net_change, self.first.value),
            "range": range_value,
            "relative_range": relative_range,
            "total_variation": self.total_variation,
            "relative_total_variation": relative_variation,
            "maximum_step_change": self.maximum_step_change,
            "path_efficiency": path_efficiency,
            "slope_per_tick": self._slope(),
            "first_nonzero_sequence": self.first_nonzero_sequence,
            "first_nonzero_tick": self.first_nonzero_tick,
            "first_significant_sequence": self.first_significant_sequence,
            "first_significant_tick": self.first_significant_tick,
            "stable_span_count": self.stable_span_count,
            "longest_stable_observation_count": max(
                (span.observation_count for span in self.stable_spans.values()),
                default=0,
            ),
            "stable_spans": self.stable_spans.payload(),
            "selected_points": [point.payload() for point in selected],
            "information_score": information_score,
        }

    def snapshot(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "first": self.first.payload() if self.first else None,
            "last": self.last.payload() if self.last else None,
            "previous": self.previous.payload() if self.previous else None,
            "minimum": self.minimum.payload() if self.minimum else None,
            "maximum": self.maximum.payload() if self.maximum else None,
            "drift_anchor": self.drift_anchor.payload() if self.drift_anchor else None,
            "previous_delta": self.previous_delta,
            "mean": self.mean,
            "m2": self.m2,
            "mean_abs": self.mean_abs,
            "sum_x": self.sum_x,
            "sum_y": self.sum_y,
            "sum_xx": self.sum_xx,
            "sum_xy": self.sum_xy,
            "area": self.area,
            "total_variation": self.total_variation,
            "maximum_step_change": self.maximum_step_change,
            "first_nonzero_sequence": self.first_nonzero_sequence,
            "first_nonzero_tick": self.first_nonzero_tick,
            "first_significant_sequence": self.first_significant_sequence,
            "first_significant_tick": self.first_significant_tick,
            "local_candidates": self.local_candidates.payload(),
            "turning_candidates": self.turning_candidates.payload(),
            "cumulative_candidates": self.cumulative_candidates.payload(),
            "baseline_candidates": self.baseline_candidates.payload(),
            "outlier_candidates": self.outlier_candidates.payload(),
            "stable_spans": self.stable_spans.payload(),
            "stable_span_count": self.stable_span_count,
            "stable_start": self._stable_start.payload() if self._stable_start else None,
            "stable_observation_count": self._stable_observation_count,
            "stable_maximum_step_score": self._stable_maximum_step_score,
        }

    @classmethod
    def from_snapshot(
        cls, config: ReducerConfig, value: Mapping[str, Any]
    ) -> "_MetricState":
        result = cls(config)
        result.count = int(value.get("count", 0))
        for name in (
            "first",
            "last",
            "previous",
            "minimum",
            "maximum",
            "drift_anchor",
        ):
            raw = value.get(name)
            setattr(result, name, _Point.from_payload(raw) if isinstance(raw, Mapping) else None)
        result.previous_delta = (
            float(value["previous_delta"])
            if value.get("previous_delta") is not None
            else None
        )
        for name in (
            "mean",
            "m2",
            "mean_abs",
            "sum_x",
            "sum_y",
            "sum_xx",
            "sum_xy",
            "area",
            "total_variation",
            "maximum_step_change",
        ):
            setattr(result, name, float(value.get(name, 0.0)))
        for name in (
            "first_nonzero_sequence",
            "first_nonzero_tick",
            "first_significant_sequence",
            "first_significant_tick",
        ):
            raw = value.get(name)
            setattr(result, name, int(raw) if raw is not None else None)
        result.local_candidates = _TopCandidates.from_payload(
            config.local_candidate_budget, value.get("local_candidates", [])
        )
        result.turning_candidates = _TopCandidates.from_payload(
            config.turning_candidate_budget, value.get("turning_candidates", [])
        )
        result.cumulative_candidates = _TopCandidates.from_payload(
            config.cumulative_candidate_budget,
            value.get("cumulative_candidates", []),
        )
        result.baseline_candidates = _TopCandidates.from_payload(
            config.baseline_candidate_budget, value.get("baseline_candidates", [])
        )
        result.outlier_candidates = _TopCandidates.from_payload(
            config.outlier_candidate_budget, value.get("outlier_candidates", [])
        )
        result.stable_spans = _TopStableSpans.from_payload(
            config.stable_span_budget, value.get("stable_spans", [])
        )
        result.stable_span_count = int(value.get("stable_span_count", 0))
        raw_start = value.get("stable_start")
        result._stable_start = (
            _Point.from_payload(raw_start) if isinstance(raw_start, Mapping) else None
        )
        result._stable_observation_count = int(
            value.get("stable_observation_count", 0)
        )
        result._stable_maximum_step_score = float(
            value.get("stable_maximum_step_score", 0.0)
        )
        return result


class WorldReducer:
    """Incremental reducer for one world's ordered observation stream."""

    def __init__(
        self,
        identity: WorldIdentity,
        *,
        config: ReducerConfig | None = None,
    ) -> None:
        self.identity = identity
        self.config = config or ReducerConfig()
        self._metrics: dict[str, _MetricState] = {}
        self.observation_count = 0
        self.last_sequence: int | None = None
        self.last_tick: int | None = None
        self.source_kinds: set[str] = set()
        self._finalized_result: dict[str, Any] | None = None

    @property
    def finalized(self) -> bool:
        return self._finalized_result is not None

    @property
    def finalized_result(self) -> dict[str, Any] | None:
        return deepcopy(self._finalized_result)

    def ingest(self, observation: Observation) -> None:
        if self.finalized:
            raise ReducerInputError("cannot ingest after world reducer finalization")
        if self.last_sequence is not None and observation.sequence <= self.last_sequence:
            raise ReducerInputError(
                "observation sequence must be strictly increasing within a world"
            )
        if self.last_tick is not None and observation.tick < self.last_tick:
            raise ReducerInputError("observation tick must not regress")
        for name, value in sorted(observation.metrics.items()):
            state = self._metrics.setdefault(name, _MetricState(self.config))
            state.ingest(observation, float(value))
        self.observation_count += 1
        self.last_sequence = observation.sequence
        self.last_tick = observation.tick
        self.source_kinds.add(observation.source.kind)

    def ingest_many(self, observations: Iterable[Observation]) -> None:
        for observation in observations:
            self.ingest(observation)

    def snapshot(self) -> dict[str, Any]:
        payload = {
            "schema_version": STATE_SCHEMA_VERSION,
            "reducer_version": REDUCER_VERSION,
            "config": self.config.payload(),
            "config_sha256": self.config.sha256,
            "identity": self.identity.payload(),
            "identity_sha256": self.identity.sha256,
            "observation_count": self.observation_count,
            "last_sequence": self.last_sequence,
            "last_tick": self.last_tick,
            "source_kinds": sorted(self.source_kinds),
            "finalized": self.finalized,
            "finalized_result": self._finalized_result,
            "metrics": {
                name: state.snapshot() for name, state in sorted(self._metrics.items())
            },
        }
        payload["state_sha256"] = canonical_sha256(payload)
        return payload

    @classmethod
    def from_snapshot(cls, value: Mapping[str, Any]) -> "WorldReducer":
        if value.get("schema_version") != STATE_SCHEMA_VERSION:
            raise ReducerInputError("unsupported reducer state schema")
        if value.get("reducer_version") != REDUCER_VERSION:
            raise ReducerInputError("reducer state version differs")
        expected_state_hash = value.get("state_sha256")
        material = dict(value)
        material.pop("state_sha256", None)
        if expected_state_hash != canonical_sha256(material):
            raise ReducerInputError("reducer state hash differs")
        identity_raw = value.get("identity")
        config_raw = value.get("config")
        if not isinstance(identity_raw, Mapping) or not isinstance(config_raw, Mapping):
            raise ReducerInputError("reducer state lacks identity or config")
        identity = WorldIdentity(
            world_id=str(identity_raw["world_id"]),
            parameter_id=str(identity_raw["parameter_id"]),
            parameter_value=identity_raw.get("parameter_value"),
            seed=int(identity_raw["seed"]),
            ticks_per_observation=int(identity_raw.get("ticks_per_observation", 1)),
            metadata=(
                dict(identity_raw.get("metadata", {}))
                if isinstance(identity_raw.get("metadata", {}), Mapping)
                else {}
            ),
        )
        config = ReducerConfig.from_mapping(config_raw)
        if value.get("config_sha256") != config.sha256:
            raise ReducerInputError("reducer state configuration hash differs")
        if value.get("identity_sha256") != identity.sha256:
            raise ReducerInputError("reducer state identity hash differs")
        result = cls(identity, config=config)
        result.observation_count = int(value.get("observation_count", 0))
        result.last_sequence = (
            int(value["last_sequence"])
            if value.get("last_sequence") is not None
            else None
        )
        result.last_tick = (
            int(value["last_tick"]) if value.get("last_tick") is not None else None
        )
        result.source_kinds = {str(item) for item in value.get("source_kinds", [])}
        finalized = value.get("finalized", False)
        if not isinstance(finalized, bool):
            raise ReducerInputError("reducer state finalized flag is invalid")
        finalized_result = value.get("finalized_result")
        if finalized and not isinstance(finalized_result, Mapping):
            raise ReducerInputError("finalized reducer state lacks result")
        if not finalized and finalized_result is not None:
            raise ReducerInputError("unfinalized reducer state contains result")
        metrics = value.get("metrics")
        if not isinstance(metrics, Mapping):
            raise ReducerInputError("reducer state metrics are missing")
        result._metrics = {
            str(name): _MetricState.from_snapshot(config, state)
            for name, state in metrics.items()
            if isinstance(state, Mapping)
        }
        if finalized:
            result.restore_finalized(finalized_result)
        return result

    def restore_finalized(self, result: Mapping[str, Any] | None) -> None:
        if not isinstance(result, Mapping):
            raise ReducerInputError("finalized reducer result must be an object")
        value = dict(result)
        if value.get("identity_sha256") != self.identity.sha256:
            raise ReducerInputError("finalized reducer result identity differs")
        expected_hash = value.get("result_sha256")
        material = dict(value)
        material.pop("result_sha256", None)
        if expected_hash != canonical_sha256(material):
            raise ReducerInputError("finalized reducer result hash differs")
        self._finalized_result = deepcopy(value)

    def finalize(self) -> dict[str, Any]:
        if self._finalized_result is not None:
            return deepcopy(self._finalized_result)
        if self.observation_count < 1:
            raise ReducerInputError("cannot finalize an empty world reducer")
        metric_summaries = {
            name: state.finalize() for name, state in sorted(self._metrics.items())
        }
        row_index: dict[tuple[str, int | None, str | None], dict[str, Any]] = {}
        for metric_name, summary in metric_summaries.items():
            for point in summary["selected_points"]:
                source = point["source"]
                key = (
                    str(source["uri"]),
                    (
                        int(source["row_number"])
                        if source.get("row_number") is not None
                        else None
                    ),
                    (
                        str(source["event_id"])
                        if source.get("event_id") is not None
                        else None
                    ),
                )
                row = row_index.setdefault(
                    key,
                    {
                        "sequence": int(point["sequence"]),
                        "tick": int(point["tick"]),
                        "source": dict(source),
                        "metrics": {},
                    },
                )
                row["metrics"][metric_name] = list(point["reasons"])
        rows = sorted(
            row_index.values(),
            key=lambda row: (
                row["sequence"],
                row["source"]["uri"],
                row["source"].get("row_number") or 0,
                row["source"].get("event_id") or "",
            ),
        )
        result = {
            "schema_version": WORLD_RESULT_SCHEMA_VERSION,
            "reducer_version": REDUCER_VERSION,
            "config_sha256": self.config.sha256,
            "identity": self.identity.payload(),
            "identity_sha256": self.identity.sha256,
            "observation_count": self.observation_count,
            "first_sequence": min(
                summary["first"]["sequence"] for summary in metric_summaries.values()
            ),
            "last_sequence": self.last_sequence,
            "last_tick": self.last_tick,
            "source_kinds": sorted(self.source_kinds),
            "metric_count": len(metric_summaries),
            "metrics": metric_summaries,
            "raw_row_index": rows,
            "raw_row_index_count": len(rows),
            "limitations": [
                "Reduced summaries do not replace the authoritative raw source.",
                "Selected row locators are a drill-back index, not a complete trajectory.",
            ],
        }
        result["result_sha256"] = canonical_sha256(result)
        self._finalized_result = deepcopy(result)
        return deepcopy(result)


class LiveReducerSink:
    """Small adapter for live simulation callbacks.

    This source-neutral primitive is called at a native reporting boundary by
    the WRC-scoped default adapter in ``live.py``.  A post-run provider replays
    the same observations into ``WorldReducer``; this class itself remains
    unaware of simulator configuration and retention policy.
    """

    def __init__(
        self,
        identity: WorldIdentity,
        *,
        config: ReducerConfig | None = None,
        source_uri: str | None = None,
    ) -> None:
        self.reducer = WorldReducer(identity, config=config)
        self.source_uri = source_uri or f"live://{identity.world_id}"

    def ingest(
        self,
        *,
        sequence: int,
        metrics: Mapping[str, int | float],
        tick: int | None = None,
        event_id: str | None = None,
    ) -> None:
        resolved_tick = (
            int(tick)
            if tick is not None
            else (int(sequence) + 1) * self.reducer.identity.ticks_per_observation
        )
        self.reducer.ingest(
            Observation(
                sequence=int(sequence),
                tick=resolved_tick,
                metrics={name: float(value) for name, value in metrics.items()},
                source=SourceLocator(
                    kind="live",
                    uri=self.source_uri,
                    event_id=event_id,
                ),
            )
        )

    def snapshot(self) -> dict[str, Any]:
        return self.reducer.snapshot()

    @classmethod
    def from_snapshot(
        cls,
        value: Mapping[str, Any],
        *,
        source_uri: str | None = None,
    ) -> "LiveReducerSink":
        result = cls.__new__(cls)
        result.reducer = WorldReducer.from_snapshot(value)
        result.source_uri = source_uri or f"live://{result.reducer.identity.world_id}"
        return result

    @classmethod
    def from_snapshot_file(
        cls,
        path: str | Path,
        *,
        source_uri: str | None = None,
    ) -> "LiveReducerSink":
        source = Path(path)
        try:
            value = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ReducerInputError(
                f"cannot restore reducer snapshot {source}: {exc}"
            ) from exc
        if not isinstance(value, Mapping):
            raise ReducerInputError("reducer snapshot root must be an object")
        return cls.from_snapshot(value, source_uri=source_uri)

    def write_snapshot(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(pretty_json(self.snapshot()), encoding="utf-8")

    def finalize(self) -> dict[str, Any]:
        return self.reducer.finalize()


__all__ = ["LiveReducerSink", "WorldReducer"]
