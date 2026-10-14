from __future__ import annotations

import csv
from pathlib import Path

from engine.wrc.reducer import (
    Observation,
    ReducerConfig,
    SourceLocator,
    WorldIdentity,
    WorldReducer,
)
from engine.wrc.reducer.providers import MetricsCsvObservationProvider, PathTextResource


def _identity(*, path: Path | None = None) -> WorldIdentity:
    return WorldIdentity(
        world_id="run-1",
        parameter_id="env_cfg.test",
        parameter_value=1.0,
        seed=101,
        ticks_per_observation=10,
        metadata={
            "source_kind": "metrics_csv" if path else "live",
            "source_path": str(path) if path else "",
        },
    )


def _observation(index: int, metrics: dict[str, float]) -> Observation:
    return Observation(
        sequence=index,
        tick=index,
        metrics=metrics,
        source=SourceLocator(
            kind="live",
            uri="live://run-1",
            event_id=f"event-{index}",
        ),
    )


def test_static_phase_keeps_only_start_and_end_anchors() -> None:
    reducer = WorldReducer(_identity())
    reducer.ingest_many(_observation(index, {"stable": 7.0}) for index in range(500))

    summary = reducer.finalize()["metrics"]["stable"]

    assert summary["count"] == 500
    assert summary["net_change"] == 0.0
    assert summary["range"] == 0.0
    assert [point["sequence"] for point in summary["selected_points"]] == [0, 499]


def test_cumulative_drift_detects_repeated_subthreshold_changes() -> None:
    config = ReducerConfig(
        local_relative_threshold=0.05,
        cumulative_relative_threshold=0.005,
    )
    reducer = WorldReducer(_identity(), config=config)
    value = 100.0
    for index in range(101):
        reducer.ingest(_observation(index, {"slow": value}))
        value *= 1.0001

    points = reducer.finalize()["metrics"]["slow"]["selected_points"]

    assert any("stage3:cumulative_drift" in point["reasons"] for point in points)
    assert len(points) > 2
    assert all(
        "stage2:local_change" not in point["reasons"] for point in points[1:-1]
    )


def test_snapshot_restore_matches_uninterrupted_live_reduction() -> None:
    identity = _identity()
    observations = [
        _observation(index, {"metric": float(index * index), "other": float(index % 3)})
        for index in range(12)
    ]

    uninterrupted = WorldReducer(identity)
    uninterrupted.ingest_many(observations)

    interrupted = WorldReducer(identity)
    interrupted.ingest_many(observations[:6])
    resumed = WorldReducer.from_snapshot(interrupted.snapshot())
    resumed.ingest_many(observations[6:])

    assert resumed.finalize() == uninterrupted.finalize()


def test_metrics_csv_replay_uses_same_core_as_live_ingestion(tmp_path: Path) -> None:
    path = tmp_path / "metrics.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["generation", "a", "b", "json"])
        writer.writeheader()
        for generation in range(5):
            writer.writerow(
                {
                    "generation": generation,
                    "a": generation * 2,
                    "b": 10 - generation,
                    "json": '{"ignored":true}',
                }
            )

    identity = _identity(path=path)
    provider = MetricsCsvObservationProvider(
        PathTextResource(path),
        ticks_per_observation=identity.ticks_per_observation,
    )
    replay = WorldReducer(identity)
    replay.ingest_many(provider)

    live = WorldReducer(identity)
    for observation in MetricsCsvObservationProvider(
        PathTextResource(path),
        ticks_per_observation=identity.ticks_per_observation,
    ):
        live.ingest(observation)

    result = replay.finalize()
    assert live.finalize() == result
    assert result["last_tick"] == 50
    assert set(result["metrics"]) == {"a", "b"}


def test_retained_trajectory_is_bounded_without_losing_anchors() -> None:
    config = ReducerConfig(max_selected_points_per_metric=8)
    reducer = WorldReducer(_identity(), config=config)
    for index in range(200):
        value = float((index % 7) - 3) * (1.0 + index / 100.0)
        reducer.ingest(_observation(index, {"oscillating": value}))

    points = reducer.finalize()["metrics"]["oscillating"]["selected_points"]

    assert len(points) <= 8
    assert points[0]["sequence"] == 0
    assert points[-1]["sequence"] == 199
    assert "stage1:anchor_first" in points[0]["reasons"]
    assert "stage1:anchor_last" in points[-1]["reasons"]


def test_raw_row_index_is_resume_stable() -> None:
    observations = [
        Observation(
            sequence=index,
            tick=index,
            metrics={"a": float(index)},
            source=SourceLocator(
                kind="metrics_csv",
                uri="file:///tmp/metrics.csv",
                row_number=index + 2,
            ),
        )
        for index in range(3)
    ]

    interrupted = WorldReducer(_identity())
    interrupted.ingest_many(observations[:2])
    resumed = WorldReducer.from_snapshot(interrupted.snapshot())
    resumed.ingest_many(observations[2:])

    uninterrupted = WorldReducer(_identity())
    uninterrupted.ingest_many(observations)

    assert resumed.finalize() == uninterrupted.finalize()
    rows = resumed.finalize()["raw_row_index"]
    assert rows[-1]["source"]["kind"] == "metrics_csv"
    assert rows[-1]["source"]["row_number"] == 4
