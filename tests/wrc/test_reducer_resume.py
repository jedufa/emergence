from __future__ import annotations

from copy import deepcopy
import json

import pytest

from engine.wrc.reducer import (
    FinalizedWorldError,
    Observation,
    ReducerStateError,
    ReducerStateStore,
    SourceLocator,
    WorldIdentity,
    WorldReducer,
    load_checkpoint,
    save_checkpoint,
)


def _identity() -> WorldIdentity:
    return WorldIdentity("resume-world", "parameter", True, 101)


def _observation(sequence: int, value: float) -> Observation:
    return Observation(
        sequence=sequence,
        tick=sequence * 10,
        metrics={"signal": value, "alive": 100.0 - value},
        source=SourceLocator("fixture", "fixture://resume", row_number=sequence + 1),
    )


def test_state_store_resume_matches_uninterrupted_and_is_reducer_only(tmp_path) -> None:
    observations = [_observation(index, float(index * index % 11)) for index in range(30)]
    uninterrupted = WorldReducer(_identity())
    uninterrupted.ingest_many(observations)

    store = ReducerStateStore(tmp_path / "reducer-state")
    partial = WorldReducer(_identity())
    partial.ingest_many(observations[:13])
    store.save(partial)
    resumed = store.restore(_identity())
    assert resumed is not None
    resumed.ingest_many(observations[13:])

    assert resumed.finalize() == uninterrupted.finalize()
    payload = json.loads(store.path_for(_identity().world_id).read_text())
    assert "simulator" not in payload
    assert payload["reducer_state"]["schema_version"]


def test_checkpoint_integrity_is_fail_closed(tmp_path) -> None:
    reducer = WorldReducer(_identity())
    reducer.ingest(_observation(0, 1.0))
    path = tmp_path / "checkpoint.json"
    save_checkpoint(path, reducer)
    value = json.loads(path.read_text())
    value["reducer_state"]["observation_count"] = 99
    path.write_text(json.dumps(value))
    with pytest.raises(ReducerStateError):
        load_checkpoint(path)


def test_finalized_checkpoint_requires_a_durable_result(tmp_path) -> None:
    reducer = WorldReducer(_identity())
    reducer.ingest(_observation(0, 1.0))
    path = tmp_path / "invalid-finalized-checkpoint.json"
    with pytest.raises(ReducerStateError, match="requires finalized result"):
        save_checkpoint(path, reducer, finalized=True)
    assert not path.exists()


def test_finalized_world_is_idempotent_and_cannot_be_reopened(tmp_path) -> None:
    reducer = WorldReducer(_identity())
    reducer.ingest(_observation(0, 1.0))
    store = ReducerStateStore(tmp_path / "state")
    result = store.finalize(reducer)
    assert store.finalize(reducer) == result
    restored = store.restore(_identity())
    assert restored is not None
    assert restored.finalize() == result
    with pytest.raises(Exception):
        restored.ingest(_observation(1, 2.0))
    with pytest.raises(FinalizedWorldError):
        store.save(WorldReducer(_identity()))


def test_core_finalization_is_idempotent_and_snapshot_preserves_terminal_state() -> None:
    reducer = WorldReducer(_identity())
    reducer.ingest(_observation(0, 1.0))
    result = reducer.finalize()
    assert reducer.finalize() == result
    restored = WorldReducer.from_snapshot(deepcopy(reducer.snapshot()))
    assert restored.finalized
    assert restored.finalize() == result
