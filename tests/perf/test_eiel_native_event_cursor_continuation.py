from __future__ import annotations

import copy
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from perf.cxxbackbone10_resident_state_owner import (
    _typed_lifecycle_event_tensors_for_runner as _checkpoint_lifecycle_tape_for_runner,
)
from native_runtime.lifecycle_events import (
    FIELD_DTYPES,
    empty_typed_lifecycle_event_tensors,
    typed_lifecycle_events_from_reference_contexts,
    write_typed_lifecycle_public_outputs,
)
from native_runtime.runner_integration import (
    _reference_typed_lifecycle_completion_required,
    emit_route_neutral_typed_lifecycle_public_outputs,
)


class _TensorOnlyRunner:
    def __init__(self, run_dir: Path, checkpoint_events: dict, *, checkpoint_cursor: int | None = None) -> None:
        self.run_dir = str(run_dir)
        self._cxxbackbone10_checkpoint_tensor_overrides = {
            **{f"typed_lifecycle_event_{name}": tensor.clone() for name, tensor in checkpoint_events.items()},
            "event_tape_type": checkpoint_events["event_type"].clone(),
            "event_tape_generation": checkpoint_events["generation"].clone(),
            "event_tape_tick": checkpoint_events["tick"].clone(),
            "event_tape_actor_id": checkpoint_events["actor_agent_id"].clone(),
            "event_tape_target_id": checkpoint_events["target_agent_id"].clone(),
            "event_tape_value": checkpoint_events["value"].clone(),
        }
        self._cxxbackbone10_checkpoint_metadata_overrides = {}
        if checkpoint_cursor is not None:
            self._cxxbackbone10_checkpoint_metadata_overrides["event_tape_cursor"] = checkpoint_cursor

    @property
    def pop(self) -> object:
        raise AssertionError("typed lifecycle continuation must not materialize Python Agents")


class _ProjectionEngine:
    def __init__(self, projection: dict) -> None:
        self.projection = projection

    def export_checkpoint_projection(self) -> dict:
        return self.projection


def _event(*, generation: int, tick: int, parent_id: int, child_id: int) -> dict:
    return typed_lifecycle_events_from_reference_contexts(
        [
            {
                "generation": generation,
                "tick": tick,
                "event_type": "birth_success",
                "parent_id": parent_id,
                "child_id": child_id,
            }
        ]
    )


def _prefixed(events: dict) -> dict:
    return {f"typed_lifecycle_event_{name}": events[name] for name in FIELD_DTYPES}


def _projection(events: dict) -> dict:
    event_rows = int(events["event_type"].shape[0])
    return {
        "metadata": {
            "typed_lifecycle_event_active_count": event_rows,
            "event_tape_active_count": event_rows,
            "event_tape_cursor": event_rows,
        },
        "tensors": {
            **{name: tensor.clone() for name, tensor in _prefixed(events).items()},
            "event_tape_type": events["event_type"].clone(),
            "event_tape_generation": events["generation"].clone(),
            "event_tape_tick": events["tick"].clone(),
            "event_tape_actor_id": events["actor_agent_id"].clone(),
            "event_tape_target_id": events["target_agent_id"].clone(),
            "event_tape_value": events["value"].clone(),
        },
    }


def _clone_projection(projection: dict) -> dict:
    return {
        "metadata": copy.deepcopy(projection["metadata"]),
        "tensors": {name: tensor.clone() for name, tensor in projection["tensors"].items()},
    }


def _public_paths(runner: _TensorOnlyRunner) -> tuple[Path, Path]:
    metrics = Path(runner.run_dir) / "metrics"
    return metrics / "typed_lifecycle_events.jsonl", metrics / "typed_lifecycle_events.csv"


def _install_reference_accumulator(monkeypatch: pytest.MonkeyPatch, events: dict) -> None:
    from perf import cxxbackbone10_resident_state_owner as resident_owner

    monkeypatch.setattr(
        resident_owner,
        "_typed_lifecycle_event_tensors_for_runner",
        lambda runner: getattr(
            runner,
            "_cxxbackbone10_reference_lifecycle_event_accumulator",
            _prefixed(events),
        ),
    )


def _canonical_events(runner: _TensorOnlyRunner) -> dict:
    prefixed = runner._cxxbackbone10_reference_lifecycle_event_accumulator
    return {name: prefixed[f"typed_lifecycle_event_{name}"] for name in FIELD_DTYPES}


def _read_public_rows(runner: _TensorOnlyRunner) -> tuple[list[dict], list[dict]]:
    jsonl_path, csv_path = _public_paths(runner)
    json_rows = [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines()]
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    return json_rows, csv_rows


def test_reference_event_completion_follows_scope_lock_not_public_output_toggle() -> None:
    runner = SimpleNamespace(
        sim_cfg=SimpleNamespace(
            native_runtime_cxx26_scope_lock_enable=True,
            typed_lifecycle_public_output_route_neutral_enable=False,
        )
    )

    assert _reference_typed_lifecycle_completion_required(runner) is True


def test_reference_event_completion_skips_observed_persistent_native_generation() -> None:
    runner = SimpleNamespace(
        sim_cfg=SimpleNamespace(
            native_runtime_cxx26_scope_lock_enable=True,
            typed_lifecycle_public_output_route_neutral_enable=True,
        ),
        _native_runtime_summary={
            "decision": "EIEL-NATIVE-RUNTIME-PERSISTENT-GENERATION",
            "executed": True,
        },
    )

    assert _reference_typed_lifecycle_completion_required(runner) is False


def test_reference_owner_completes_rows_and_emits_route_neutral_public_tape_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sparse = typed_lifecycle_events_from_reference_contexts(
        [
            {
                "generation": 0,
                "tick": 0,
                "event_type": "birth_success",
                "parent_id": 10,
                "child_id": 11,
            },
            {
                "generation": 0,
                "tick": 1,
                "event_type": "death_committed",
                "agent_id": 12,
                "reason_code": 2,
                "value": 2.0,
            },
        ]
    )
    runner = SimpleNamespace(
        run_dir=str(tmp_path / "reference"),
        sim_cfg=SimpleNamespace(
            native_runtime_cxx26_scope_lock_enable=True,
            typed_lifecycle_public_output_route_neutral_enable=True,
        ),
        pop=SimpleNamespace(
            agents={
                10: SimpleNamespace(
                    agent_id=10,
                    slot=3,
                    stats=SimpleNamespace(founder_id=2, lineage_depth=1, parent_ids=()),
                ),
                12: SimpleNamespace(
                    agent_id=12,
                    slot=4,
                    stats=SimpleNamespace(founder_id=5, lineage_depth=3, parent_ids=()),
                ),
            }
        ),
        _cxxbackbone10_checkpoint_tensor_overrides={},
        _cxxbackbone10_checkpoint_metadata_overrides={},
        _reference_typed_lifecycle_identity_registry={
            11: {
                "slot": 7,
                "lineage_founder_id": 2,
                "lineage_depth": 2,
                "parent_agent_id": 10,
            }
        },
    )
    _install_reference_accumulator(monkeypatch, sparse)

    first = emit_route_neutral_typed_lifecycle_public_outputs(runner)
    paths = _public_paths(runner)
    first_bytes = tuple(path.read_bytes() for path in paths)
    second = emit_route_neutral_typed_lifecycle_public_outputs(runner)
    rows, csv_rows = _read_public_rows(runner)

    assert first["schema_version"] == "eiel_native_typed_lifecycle_public_output_v1"
    assert first["jsonl_path"].endswith("/metrics/typed_lifecycle_events.jsonl")
    assert first["csv_path"].endswith("/metrics/typed_lifecycle_events.csv")
    assert first["rows_written"] == 2
    assert second["rows_written"] == 0
    assert tuple(path.read_bytes() for path in paths) == first_bytes
    assert len(rows) == len(csv_rows) == 2
    assert [row["sequence"] for row in rows] == [0, 1]
    assert rows[0]["actor_slot"] == 3
    assert rows[0]["target_slot"] == 7
    assert rows[0]["lineage_founder_id"] == 2
    assert rows[0]["lineage_depth"] == 2
    assert rows[1]["actor_slot"] == 4
    assert rows[1]["lineage_founder_id"] == 5
    assert rows[1]["lineage_depth"] == 3
    assert rows[1]["reason_code"] == 2


def test_existing_checkpoint_row_plus_one_reference_event_writes_new_row_exactly_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = _event(generation=0, tick=0, parent_id=10, child_id=11)
    reference = _event(generation=1, tick=0, parent_id=20, child_id=21)
    runner = _TensorOnlyRunner(tmp_path / "resume", checkpoint)
    jsonl_path, csv_path = _public_paths(runner)
    write_typed_lifecycle_public_outputs(jsonl_path, csv_path, checkpoint)
    _install_reference_accumulator(monkeypatch, reference)

    summary = emit_route_neutral_typed_lifecycle_public_outputs(runner)

    json_rows, csv_rows = _read_public_rows(runner)
    checkpoint_cursor = int(checkpoint["event_type"].shape[0])
    continued = _canonical_events(runner)
    checkpoint_view = _checkpoint_lifecycle_tape_for_runner(runner)
    assert summary["source"] == "native_checkpoint_boundary_plus_reference_cumulative_lifecycle_event_tensors"
    assert summary["start_cursor"] == checkpoint_cursor
    assert summary["end_cursor"] == checkpoint_cursor + 1
    assert summary["rows_written"] == 1
    assert runner._typed_lifecycle_event_output_accumulator_cursor == checkpoint_cursor + 1
    assert runner._typed_lifecycle_event_public_cursor == checkpoint_cursor + 1
    assert not any(
        name.startswith("typed_lifecycle_event_") or name.startswith("event_tape_")
        for name in runner._cxxbackbone10_checkpoint_tensor_overrides
    )
    assert "event_tape_cursor" not in runner._cxxbackbone10_checkpoint_metadata_overrides
    assert continued["sequence"].tolist() == [0, checkpoint_cursor]
    assert int(continued["child_agent_id"][checkpoint_cursor].item()) == 21
    assert [int(row["sequence"]) for row in json_rows] == [0, 1]
    assert int(json_rows[checkpoint_cursor]["child_agent_id"]) == 21
    assert [int(row["sequence"]) for row in csv_rows] == [0, 1]
    assert int(csv_rows[checkpoint_cursor]["child_agent_id"]) == 21
    for name in FIELD_DTYPES:
        assert torch.equal(
            checkpoint_view[f"typed_lifecycle_event_{name}"],
            continued[name],
        ), name


def test_repeated_reference_emission_is_idempotent_after_checkpoint_continuation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = _event(generation=0, tick=0, parent_id=10, child_id=11)
    reference = _event(generation=1, tick=0, parent_id=20, child_id=21)
    runner = _TensorOnlyRunner(tmp_path / "repeat", checkpoint)
    jsonl_path, csv_path = _public_paths(runner)
    write_typed_lifecycle_public_outputs(jsonl_path, csv_path, checkpoint)
    _install_reference_accumulator(monkeypatch, reference)

    first = emit_route_neutral_typed_lifecycle_public_outputs(runner)
    first_bytes = (jsonl_path.read_bytes(), csv_path.read_bytes())
    second = emit_route_neutral_typed_lifecycle_public_outputs(runner)

    assert first["rows_written"] == 1
    assert second["start_cursor"] == 2
    assert second["end_cursor"] == 2
    assert second["rows_written"] == 0
    assert (jsonl_path.read_bytes(), csv_path.read_bytes()) == first_bytes
    assert len(_read_public_rows(runner)[0]) == 2
    assert runner._typed_lifecycle_event_output_accumulator_cursor == 2


def test_native_reexport_preserves_continued_accumulator_and_cursor_exactly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = _event(generation=0, tick=0, parent_id=10, child_id=11)
    reference = _event(generation=1, tick=0, parent_id=20, child_id=21)
    restored = _TensorOnlyRunner(tmp_path / "restored_reference", checkpoint)
    restored_jsonl, restored_csv = _public_paths(restored)
    write_typed_lifecycle_public_outputs(restored_jsonl, restored_csv, checkpoint)
    _install_reference_accumulator(monkeypatch, reference)
    restored_summary = emit_route_neutral_typed_lifecycle_public_outputs(restored)
    continued = _canonical_events(restored)
    projection = _projection(continued)
    engine = _ProjectionEngine(projection)
    runner = _TensorOnlyRunner(tmp_path / "native_reexport", checkpoint)
    runner._native_eiel_engine = engine
    before = _clone_projection(engine.export_checkpoint_projection())

    first = emit_route_neutral_typed_lifecycle_public_outputs(runner)
    second = emit_route_neutral_typed_lifecycle_public_outputs(runner)
    reexported = engine.export_checkpoint_projection()

    assert restored_summary["start_cursor"] == 1
    assert restored_summary["end_cursor"] == 2
    assert restored_summary["rows_written"] == 1
    assert restored._typed_lifecycle_event_output_accumulator_cursor == 2
    assert first["end_cursor"] == 2
    assert first["rows_written"] == 2
    assert second["start_cursor"] == 2
    assert second["rows_written"] == 0
    assert runner._typed_lifecycle_event_output_accumulator_cursor == 2
    assert runner._typed_lifecycle_event_public_cursor == 2
    assert reexported["metadata"]["event_tape_cursor"] == 2
    assert reexported["metadata"]["typed_lifecycle_event_active_count"] == 2
    assert reexported["metadata"] == before["metadata"]
    for name, expected in before["tensors"].items():
        assert torch.equal(reexported["tensors"][name], expected), name
    for name in FIELD_DTYPES:
        assert torch.equal(
            _canonical_events(runner)[name],
            before["tensors"][f"typed_lifecycle_event_{name}"],
        ), name


def test_restored_reference_tape_reuses_same_output_directory_without_duplication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = _event(generation=0, tick=0, parent_id=10, child_id=11)
    run_dir = tmp_path / "same_output"
    first = _TensorOnlyRunner(run_dir, checkpoint)
    _install_reference_accumulator(monkeypatch, empty_typed_lifecycle_event_tensors())

    initial = emit_route_neutral_typed_lifecycle_public_outputs(first)
    before = tuple(path.read_bytes() for path in _public_paths(first))
    restored = _TensorOnlyRunner(run_dir, checkpoint)
    resumed = emit_route_neutral_typed_lifecycle_public_outputs(restored)

    assert initial["start_cursor"] == 0
    assert initial["rows_written"] == 1
    assert resumed["cursor_source"] == "existing_file_row_count"
    assert resumed["start_cursor"] == 1
    assert resumed["end_cursor"] == 1
    assert resumed["rows_written"] == 0
    assert tuple(path.read_bytes() for path in _public_paths(restored)) == before
    assert _canonical_events(restored)["sequence"].tolist() == [0]


def test_canonical_reference_tape_replays_fully_into_new_output_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = _event(generation=0, tick=0, parent_id=10, child_id=11)
    reference = _event(generation=1, tick=0, parent_id=20, child_id=21)
    runner = _TensorOnlyRunner(tmp_path / "first_output", checkpoint)
    _install_reference_accumulator(monkeypatch, reference)

    first = emit_route_neutral_typed_lifecycle_public_outputs(runner)
    first_bytes = tuple(path.read_bytes() for path in _public_paths(runner))
    runner.run_dir = str(tmp_path / "new_output")
    replay = emit_route_neutral_typed_lifecycle_public_outputs(runner)

    assert first["end_cursor"] == 2
    assert replay["cursor_source"] == "new_output"
    assert replay["start_cursor"] == 0
    assert replay["end_cursor"] == 2
    assert replay["rows_written"] == 2
    assert tuple(path.read_bytes() for path in _public_paths(runner)) == first_bytes
    assert _canonical_events(runner)["sequence"].tolist() == [0, 1]


def test_native_reexport_cursor_contradiction_fails_before_output(
    tmp_path: Path,
) -> None:
    checkpoint = _event(generation=0, tick=0, parent_id=10, child_id=11)
    projection = _projection(checkpoint)
    projection["metadata"]["event_tape_cursor"] = 0
    runner = _TensorOnlyRunner(tmp_path / "native_cursor_contradiction", checkpoint)
    runner._native_eiel_engine = _ProjectionEngine(projection)

    with pytest.raises(RuntimeError, match="typed lifecycle projection cursor contradiction"):
        emit_route_neutral_typed_lifecycle_public_outputs(runner)

    assert not hasattr(runner, "_cxxbackbone10_reference_lifecycle_event_accumulator")
    assert not hasattr(runner, "_typed_lifecycle_event_output_accumulator_cursor")
    assert not (Path(runner.run_dir) / "metrics").exists()


def test_checkpoint_cursor_contradiction_fails_closed_before_public_output_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = _event(generation=0, tick=0, parent_id=10, child_id=11)
    reference = _event(generation=1, tick=0, parent_id=20, child_id=21)
    runner = _TensorOnlyRunner(tmp_path / "contradiction", checkpoint, checkpoint_cursor=0)
    jsonl_path, csv_path = _public_paths(runner)
    write_typed_lifecycle_public_outputs(jsonl_path, csv_path, checkpoint)
    before = (jsonl_path.read_bytes(), csv_path.read_bytes())
    _install_reference_accumulator(monkeypatch, reference)

    with pytest.raises(RuntimeError, match="typed lifecycle checkpoint cursor contradiction"):
        emit_route_neutral_typed_lifecycle_public_outputs(runner)

    assert (jsonl_path.read_bytes(), csv_path.read_bytes()) == before
    assert not hasattr(runner, "_cxxbackbone10_reference_lifecycle_event_accumulator")
    assert not hasattr(runner, "_typed_lifecycle_event_output_accumulator_cursor")


def test_public_prefix_conflict_does_not_consume_checkpoint_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = _event(generation=0, tick=0, parent_id=10, child_id=11)
    reference = _event(generation=1, tick=0, parent_id=20, child_id=21)
    runner = _TensorOnlyRunner(tmp_path / "public_conflict", checkpoint, checkpoint_cursor=1)
    jsonl_path, csv_path = _public_paths(runner)
    write_typed_lifecycle_public_outputs(jsonl_path, csv_path, checkpoint)
    row = json.loads(jsonl_path.read_text(encoding="utf-8"))
    row["child_agent_id"] = 99
    jsonl_path.write_text(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    before_tensors = set(runner._cxxbackbone10_checkpoint_tensor_overrides)
    before_metadata = dict(runner._cxxbackbone10_checkpoint_metadata_overrides)
    before_files = (jsonl_path.read_bytes(), csv_path.read_bytes())
    _install_reference_accumulator(monkeypatch, reference)

    with pytest.raises(RuntimeError, match="jsonl prefix conflicts with canonical tape"):
        emit_route_neutral_typed_lifecycle_public_outputs(runner)

    assert set(runner._cxxbackbone10_checkpoint_tensor_overrides) == before_tensors
    assert runner._cxxbackbone10_checkpoint_metadata_overrides == before_metadata
    assert (jsonl_path.read_bytes(), csv_path.read_bytes()) == before_files
    assert not hasattr(runner, "_cxxbackbone10_reference_lifecycle_event_accumulator")
    assert not hasattr(runner, "_typed_lifecycle_event_output_accumulator_cursor")
