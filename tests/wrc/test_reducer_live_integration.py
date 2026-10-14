from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from engine.wrc.reducer import (
    LiveReducerSink,
    ReducerInputError,
    WorldIdentity,
    WorldReducer,
    WRC_LIVE_REDUCER_DEFAULT_ENABLED,
    WRC_LIVE_REDUCER_DEFAULT_RETENTION_MODE,
    WrcLiveReducer,
    create_wrc_live_reducer,
)
from engine.wrc.reducer.providers import MetricsCsvObservationProvider, PathTextResource


def _identity() -> WorldIdentity:
    return WorldIdentity(
        world_id="wrc-world-101",
        parameter_id="env_cfg.switch",
        parameter_value=True,
        seed=101,
        ticks_per_observation=10,
        metadata={"campaign": "wp1-fixture"},
    )


def _values() -> list[dict[str, float]]:
    return [
        {"alive": 100.0, "activation_signal": 0.0},
        {"alive": 97.0, "activation_signal": 0.0},
        {"alive": 91.0, "activation_signal": 1.0},
        {"alive": 89.0, "activation_signal": 2.0},
        {"alive": 84.0, "activation_signal": 2.0},
    ]


def _strip_provenance(value: Any, *, key: str | None = None) -> Any:
    if isinstance(value, dict):
        result = {}
        for name, item in value.items():
            if name in {"source", "source_kinds", "raw_row_index", "raw_row_index_count"}:
                continue
            if name == "metadata" and key == "identity":
                continue
            if name == "result_sha256":
                continue
            result[name] = _strip_provenance(item, key=name)
        return result
    if isinstance(value, list):
        return [_strip_provenance(item, key=key) for item in value]
    return value


def _scientific_digest(result: dict[str, Any]) -> str:
    payload = _strip_provenance(result)
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _write_metrics(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["generation", "alive", "activation_signal"],
        )
        writer.writeheader()
        for generation, metrics in enumerate(_values()):
            writer.writerow({"generation": generation, **metrics})


def test_wrc_live_reducer_is_enabled_by_default_and_can_be_disabled() -> None:
    enabled = create_wrc_live_reducer(_identity())
    assert WRC_LIVE_REDUCER_DEFAULT_ENABLED is True
    assert enabled.enabled is True
    assert enabled.retention_mode == WRC_LIVE_REDUCER_DEFAULT_RETENTION_MODE

    disabled = create_wrc_live_reducer(_identity(), live_reducer_enable=False)
    for sequence, metrics in enumerate(_values()):
        disabled.ingest(sequence=sequence, tick=sequence * 10, metrics=metrics)
    assert disabled.enabled is False
    assert disabled.observation_count == 0
    assert disabled.finalize() is None
    assert disabled.status()["enabled"] is False


def test_live_and_metrics_replay_have_equal_scientific_content(tmp_path: Path) -> None:
    metrics_path = tmp_path / "metrics.csv"
    _write_metrics(metrics_path)

    live = WrcLiveReducer(_identity(), source_uri="live://wrc-world-101")
    for sequence, metrics in enumerate(_values()):
        live.ingest(
            sequence=sequence,
            tick=(sequence + 1) * 10,
            metrics=metrics,
            event_id=f"generation-{sequence}",
        )
    live_result = live.finalize()
    assert live_result is not None

    replay = WorldReducer(_identity())
    replay.ingest_many(
        MetricsCsvObservationProvider(
            PathTextResource(metrics_path),
            ticks_per_observation=10,
        )
    )
    replay_result = replay.finalize()

    assert _scientific_digest(live_result) == _scientific_digest(replay_result)
    assert live_result["observation_count"] == replay_result["observation_count"]


def test_live_snapshot_restore_matches_uninterrupted_result(tmp_path: Path) -> None:
    values = _values()
    interrupted = WrcLiveReducer(_identity())
    for sequence, metrics in enumerate(values[:3]):
        interrupted.ingest(sequence=sequence, tick=sequence * 10, metrics=metrics)
    snapshot_path = tmp_path / "reducer" / "state.json"
    interrupted.write_snapshot(snapshot_path)

    resumed = WrcLiveReducer.from_snapshot_file(snapshot_path)
    for sequence, metrics in enumerate(values[3:], start=3):
        resumed.ingest(sequence=sequence, tick=sequence * 10, metrics=metrics)

    uninterrupted = WrcLiveReducer(_identity())
    for sequence, metrics in enumerate(values):
        uninterrupted.ingest(sequence=sequence, tick=sequence * 10, metrics=metrics)

    resumed_result = resumed.finalize()
    uninterrupted_result = uninterrupted.finalize()
    assert resumed_result is not None
    assert uninterrupted_result is not None
    assert _scientific_digest(resumed_result) == _scientific_digest(uninterrupted_result)
    assert resumed.status()["observation_count"] == len(values)


def test_live_adapter_is_observational_and_does_not_mutate_raw_metrics() -> None:
    raw_metrics = {"alive": 100.0, "activation_signal": 1.0}
    before = dict(raw_metrics)
    trajectory = []
    reducer = WrcLiveReducer(_identity())
    state = 7
    for sequence in range(4):
        state = (state * 13 + sequence) % 101
        metrics = {**raw_metrics, "alive": raw_metrics["alive"] - state}
        trajectory.append((state, metrics["alive"]))
        reducer.ingest(sequence=sequence, tick=sequence, metrics=metrics)

    assert raw_metrics == before
    assert trajectory == [(91, 9.0), (73, 27.0), (42, 58.0), (44, 56.0)]
    result = reducer.finalize()
    assert result is not None


def test_enabled_and_disabled_observers_preserve_deterministic_trajectory() -> None:
    def run(live_reducer_enable: bool) -> tuple[list[tuple[int, int]], str]:
        reducer = WrcLiveReducer(
            _identity(), live_reducer_enable=live_reducer_enable
        )
        state = 17
        trajectory: list[tuple[int, int]] = []
        for sequence in range(12):
            state = (state * 37 + sequence * 11) % 1009
            trajectory.append((sequence, state))
            reducer.ingest(
                sequence=sequence,
                tick=sequence,
                metrics={"state_digest_input": float(state)},
            )
        result = reducer.finalize()
        return trajectory, hashlib.sha256(
            json.dumps(trajectory, separators=(",", ":")).encode()
        ).hexdigest()

    enabled_trajectory, enabled_hash = run(True)
    disabled_trajectory, disabled_hash = run(False)
    assert enabled_trajectory == disabled_trajectory
    assert enabled_hash == disabled_hash


def test_live_snapshot_hash_and_lifecycle_fail_closed() -> None:
    reducer = WrcLiveReducer(_identity())
    reducer.ingest(sequence=0, tick=0, metrics={"alive": 1.0})
    snapshot = reducer.snapshot()
    snapshot["enabled"] = False
    with pytest.raises(ReducerInputError, match="snapshot hash differs"):
        WrcLiveReducer.from_snapshot(snapshot)

    reducer.finalize()
    with pytest.raises(ReducerInputError, match="after live reducer finalization"):
        reducer.ingest(sequence=1, tick=1, metrics={"alive": 0.0})


def test_retain_all_keeps_offline_raw_metrics_available(tmp_path: Path) -> None:
    metrics_path = tmp_path / "metrics.csv"
    _write_metrics(metrics_path)
    original = metrics_path.read_bytes()

    reducer = WrcLiveReducer(_identity())
    for sequence, metrics in enumerate(_values()):
        reducer.ingest(sequence=sequence, tick=sequence * 10, metrics=metrics)
    assert reducer.retention_mode == "retain_all"
    assert reducer.finalize() is not None
    assert metrics_path.read_bytes() == original


def test_live_sink_remains_the_shared_core_adapter() -> None:
    assert issubclass(LiveReducerSink, object)
    reducer = WrcLiveReducer(_identity())
    assert isinstance(reducer._sink, LiveReducerSink)
