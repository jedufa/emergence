from __future__ import annotations

from copy import deepcopy

from engine.wrc.reducer import (
    LiveReducerSink,
    Observation,
    ReducerConfig,
    SourceLocator,
    WorldIdentity,
    WorldReducer,
)


def _identity() -> WorldIdentity:
    return WorldIdentity(
        world_id="world-1",
        parameter_id="env_cfg.example",
        parameter_value=1.0,
        seed=101,
        ticks_per_observation=1,
    )


def _observation(sequence: int, value: float) -> Observation:
    return Observation(
        sequence=sequence,
        tick=sequence,
        metrics={"signal": value},
        source=SourceLocator(
            kind="live",
            uri="live://world-1",
            event_id=f"event-{sequence}",
        ),
    )


def test_long_static_phase_is_compressed_to_anchors_and_span_summary() -> None:
    reducer = WorldReducer(_identity())
    reducer.ingest_many(_observation(index, 3.0) for index in range(1000))

    result = reducer.finalize()
    metric = result["metrics"]["signal"]

    assert [point["sequence"] for point in metric["selected_points"]] == [0, 999]
    assert metric["stable_span_count"] == 1
    assert metric["longest_stable_observation_count"] == 1000
    assert metric["net_change"] == 0.0


def test_cumulative_drift_catches_repeated_subthreshold_steps() -> None:
    config = ReducerConfig(
        local_relative_threshold=0.02,
        stable_relative_threshold=0.001,
        cumulative_relative_threshold=0.01,
        baseline_relative_threshold=0.02,
    )
    reducer = WorldReducer(_identity(), config=config)
    reducer.ingest_many(
        _observation(index, 1.0 + index * 0.0001) for index in range(250)
    )

    metric = reducer.finalize()["metrics"]["signal"]
    reasons = {
        reason
        for point in metric["selected_points"]
        for reason in point["reasons"]
    }

    assert "stage2:local_change" not in reasons
    assert "stage3:cumulative_drift" in reasons
    assert "stage3:baseline_drift" in reasons
    assert metric["last"]["value"] > 1.02


def test_outlier_and_turning_points_are_retained() -> None:
    config = ReducerConfig(
        local_relative_threshold=0.5,
        stable_relative_threshold=0.01,
        outlier_z_threshold=2.5,
        outlier_min_samples=5,
    )
    values = [0.0, 1.0, -1.0, 0.5, -0.5, 0.25, -0.25, 0.0, 0.2, -0.2, 10.0, 0.0]
    reducer = WorldReducer(_identity(), config=config)
    reducer.ingest_many(_observation(index, value) for index, value in enumerate(values))

    metric = reducer.finalize()["metrics"]["signal"]
    reasons = {
        reason
        for point in metric["selected_points"]
        for reason in point["reasons"]
    }

    assert "stage4:online_outlier" in reasons
    assert "stage2:local_extremum" in reasons
    assert metric["maximum"]["value"] == 10.0


def test_snapshot_restore_matches_uninterrupted_reduction() -> None:
    observations = [_observation(index, float(index * index % 17)) for index in range(50)]

    uninterrupted = WorldReducer(_identity())
    uninterrupted.ingest_many(observations)

    interrupted = WorldReducer(_identity())
    interrupted.ingest_many(observations[:23])
    restored = WorldReducer.from_snapshot(deepcopy(interrupted.snapshot()))
    restored.ingest_many(observations[23:])

    assert restored.finalize() == uninterrupted.finalize()


def _scientific_projection(result: dict) -> dict:
    def strip_provenance(value: object) -> object:
        if isinstance(value, dict):
            return {
                key: strip_provenance(item)
                for key, item in value.items()
                if key != "source"
            }
        if isinstance(value, list):
            return [strip_provenance(item) for item in value]
        return value

    return strip_provenance(result["metrics"])


def test_live_sink_uses_same_core_semantics() -> None:
    values = [1.0, 1.001, 1.002, 1.1, 1.1, 0.9]
    sink = LiveReducerSink(_identity(), source_uri="live://world-1")
    replay = WorldReducer(_identity())
    for sequence, value in enumerate(values):
        sink.ingest(sequence=sequence, tick=sequence, metrics={"signal": value})
        replay.ingest(_observation(sequence, value))

    assert _scientific_projection(sink.finalize()) == _scientific_projection(
        replay.finalize()
    )


def test_live_sink_snapshot_file_resumes_without_semantic_drift(tmp_path) -> None:
    values = [1.0, 1.1, 1.2, 1.0, 0.8]
    sink = LiveReducerSink(_identity(), source_uri="live://before")
    for sequence, value in enumerate(values[:3]):
        sink.ingest(sequence=sequence, tick=sequence, metrics={"signal": value})
    snapshot = tmp_path / "reducer-state.json"
    sink.write_snapshot(snapshot)

    resumed = LiveReducerSink.from_snapshot_file(
        snapshot,
        source_uri="live://after",
    )
    for sequence, value in enumerate(values[3:], start=3):
        resumed.ingest(sequence=sequence, tick=sequence, metrics={"signal": value})

    uninterrupted = LiveReducerSink(_identity(), source_uri="live://full")
    for sequence, value in enumerate(values):
        uninterrupted.ingest(
            sequence=sequence,
            tick=sequence,
            metrics={"signal": value},
        )

    assert _scientific_projection(resumed.finalize()) == _scientific_projection(
        uninterrupted.finalize()
    )
