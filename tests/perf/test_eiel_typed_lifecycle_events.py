from __future__ import annotations

import csv
import json

import pytest
import torch

from engine.campaigns.artifact_lifecycle import (
    CHECKPOINT_TRANSIT,
    DEBUG_FORENSIC,
    lifecycle_profile_contract_v1,
)
from native_runtime.event_matrix import (
    compare_typed_lifecycle_artifacts,
    checkpoint_sidecar_semantics_from_projection,
    final_lifecycle_summary_from_artifact,
    final_lineage_summary_semantics_from_projection,
    final_summary_json_semantics_from_projection,
    lineage_funnel_semantic_rows_from_artifact,
    lifecycle_counts_from_artifact,
    lifecycle_semantic_rows_from_artifact,
    typed_lifecycle_artifact_from_projection,
)
from native_runtime.lifecycle_events import (
    FIELD_DTYPES,
    LifecycleEventType,
    complete_reference_typed_lifecycle_events,
    empty_typed_lifecycle_event_tensors,
    schema_manifest,
    typed_lifecycle_event_digest,
    typed_lifecycle_events_from_legacy_tape,
    typed_lifecycle_events_from_projection,
    typed_lifecycle_events_from_reference_contexts,
    typed_lifecycle_events_to_rows,
    write_typed_lifecycle_events_csv,
    write_typed_lifecycle_events_jsonl,
    write_typed_lifecycle_public_outputs,
)


def _typed_storage_projection(*, capacity: int = 3, active_count: int = 2) -> dict:
    contexts = [
        {
            "generation": 1,
            "tick": index,
            "event_type": "birth_success",
            "parent_id": 10 + index,
            "child_id": 20 + index,
        }
        for index in range(capacity)
    ]
    events = typed_lifecycle_events_from_reference_contexts(contexts)
    return {
        "metadata": {
            "typed_lifecycle_event_active_count": active_count,
            "typed_lifecycle_event_next_sequence": active_count,
            "event_tape_active_count": active_count,
            "event_tape_cursor": active_count,
            "event_tape_inactive_tail_count": capacity - active_count,
        },
        "tensors": {
            f"typed_lifecycle_event_{name}": tensor.clone()
            for name, tensor in events.items()
        },
    }


def test_typed_lifecycle_event_schema_manifest_is_stable() -> None:
    manifest = schema_manifest()

    assert manifest["schema_version"] == "eiel_typed_lifecycle_event_v1"
    assert manifest["event_types"]["birth_committed"] == 1
    assert manifest["event_types"]["mutation"] == 2
    assert manifest["event_types"]["birth_failed"] == 3
    assert manifest["event_types"]["death_committed"] == 4
    assert manifest["ordering"] == ["generation", "tick", "sequence"]


def test_empty_typed_lifecycle_event_tensors_are_deterministic() -> None:
    events = empty_typed_lifecycle_event_tensors()

    assert sorted(events) == sorted(schema_manifest()["fields"])
    assert all(int(tensor.numel()) == 0 for tensor in events.values())
    assert typed_lifecycle_event_digest(events) == typed_lifecycle_event_digest(empty_typed_lifecycle_event_tensors())


def test_typed_projection_accepts_zero_rows_with_complete_columns() -> None:
    projection = {
        "metadata": {
            "typed_lifecycle_event_active_count": 0,
            "event_tape_active_count": 0,
            "event_tape_cursor": 0,
            "event_tape_inactive_tail_count": 0,
        },
        "tensors": {
            f"typed_lifecycle_event_{name}": tensor
            for name, tensor in empty_typed_lifecycle_event_tensors().items()
        },
    }

    events = typed_lifecycle_events_from_projection(projection)

    assert set(events) == set(FIELD_DTYPES)
    assert all(tensor.numel() == 0 for tensor in events.values())


def test_typed_projection_keeps_only_validated_active_prefix_when_capacity_grows() -> None:
    projection = _typed_storage_projection(capacity=5, active_count=2)

    events = typed_lifecycle_events_from_projection(projection)

    assert {int(tensor.shape[0]) for tensor in events.values()} == {2}
    assert events["sequence"].tolist() == [0, 1]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("event_tape_active_count", 1),
        ("event_tape_cursor", 1),
        ("typed_lifecycle_event_next_sequence", 1),
    ],
)
def test_typed_projection_rejects_count_and_cursor_disagreement(field: str, value: int) -> None:
    projection = _typed_storage_projection()
    projection["metadata"][field] = value

    with pytest.raises(RuntimeError, match="typed lifecycle projection cursor contradiction"):
        typed_lifecycle_events_from_projection(projection)


@pytest.mark.parametrize("delta", [-1, 1])
def test_typed_projection_rejects_short_or_long_column(delta: int) -> None:
    projection = _typed_storage_projection()
    name = "typed_lifecycle_event_value"
    value = projection["tensors"][name]
    projection["tensors"][name] = (
        value[:-1].clone()
        if delta < 0
        else torch.cat((value, torch.zeros((1,), dtype=value.dtype)))
    )

    with pytest.raises(RuntimeError, match="columns have inconsistent lengths"):
        typed_lifecycle_events_from_projection(projection)


def test_typed_projection_rejects_missing_mandatory_column_before_legacy_fallback() -> None:
    projection = _typed_storage_projection()
    del projection["tensors"]["typed_lifecycle_event_child_agent_id"]
    projection["tensors"].update(
        {
            "event_tape_type": torch.zeros((3,), dtype=torch.int32),
            "event_tape_actor_id": torch.zeros((3,), dtype=torch.int64),
            "event_tape_target_id": torch.zeros((3,), dtype=torch.int64),
            "event_tape_value": torch.zeros((3,), dtype=torch.float32),
        }
    )

    with pytest.raises(RuntimeError, match="missing mandatory columns"):
        typed_lifecycle_events_from_projection(projection)


def test_typed_projection_rejects_active_count_beyond_capacity() -> None:
    projection = _typed_storage_projection(capacity=3, active_count=2)
    for field in (
        "typed_lifecycle_event_active_count",
        "typed_lifecycle_event_next_sequence",
        "event_tape_active_count",
        "event_tape_cursor",
    ):
        projection["metadata"][field] = 4
    projection["metadata"]["event_tape_inactive_tail_count"] = 0

    with pytest.raises(RuntimeError, match="active count exceeds column capacity"):
        typed_lifecycle_events_from_projection(projection)


def test_typed_projection_rejects_inactive_tail_contradiction() -> None:
    projection = _typed_storage_projection(capacity=3, active_count=2)
    projection["metadata"]["event_tape_inactive_tail_count"] = 0

    with pytest.raises(RuntimeError, match="inactive-tail contradiction"):
        typed_lifecycle_events_from_projection(projection)


def test_direct_event_serialization_rejects_inconsistent_column_lengths() -> None:
    events = typed_lifecycle_events_from_reference_contexts(
        [{"generation": 1, "tick": 0, "event_type": "birth_success", "parent_id": 10, "child_id": 11}]
    )
    events["value"] = torch.empty((0,), dtype=torch.float32)

    with pytest.raises(RuntimeError, match="columns have inconsistent lengths"):
        typed_lifecycle_events_to_rows(events)


def test_legacy_tape_maps_birth_mutation_failure_and_death() -> None:
    projection = {
        "metadata": {"generation": 5, "tick": 7, "event_tape_active_count": 4},
        "tensors": {
            "event_tape_type": torch.tensor(
                [
                    LifecycleEventType.MUTATION,
                    LifecycleEventType.BIRTH_COMMITTED,
                    LifecycleEventType.BIRTH_FAILED,
                    LifecycleEventType.DEATH_COMMITTED,
                ],
                dtype=torch.int32,
            ),
            "event_tape_actor_id": torch.tensor([22, 11, 33, 44], dtype=torch.int64),
            "event_tape_target_id": torch.tensor([22, 22, -1, -1], dtype=torch.int64),
            "event_tape_value": torch.tensor([0.0, 0.0, 4.0, 2.0], dtype=torch.float32),
        },
    }

    events = typed_lifecycle_events_from_legacy_tape(projection)

    assert events["generation"].tolist() == [5, 5, 5, 5]
    assert events["tick"].tolist() == [7, 7, 7, 7]
    assert events["sequence"].tolist() == [0, 1, 2, 3]
    assert events["parent_agent_id"].tolist() == [-1, 11, -1, -1]
    assert events["child_agent_id"].tolist() == [22, 22, -1, -1]
    assert events["reason_code"].tolist() == [-1, -1, 4, -1]
    assert events["success"].tolist() == [1, 1, 0, 1]


def test_legacy_tape_preserves_optional_per_row_generation_and_tick() -> None:
    projection = {
        "metadata": {"generation": 5, "tick": 99, "event_tape_active_count": 3},
        "tensors": {
            "event_tape_type": torch.tensor(
                [
                    LifecycleEventType.BIRTH_COMMITTED,
                    LifecycleEventType.BIRTH_FAILED,
                    LifecycleEventType.DEATH_COMMITTED,
                ],
                dtype=torch.int32,
            ),
            "event_tape_generation": torch.tensor([2, 2, 2], dtype=torch.int64),
            "event_tape_tick": torch.tensor([7, 8, 9], dtype=torch.int64),
            "event_tape_actor_id": torch.tensor([11, 12, 13], dtype=torch.int64),
            "event_tape_target_id": torch.tensor([21, -1, -1], dtype=torch.int64),
            "event_tape_value": torch.tensor([0.0, 1.0, 2.0], dtype=torch.float32),
        },
    }

    events = typed_lifecycle_events_from_legacy_tape(projection)

    assert events["generation"].tolist() == [2, 2, 2]
    assert events["tick"].tolist() == [7, 8, 9]
    assert events["sequence"].tolist() == [0, 1, 2]


def test_reference_contexts_map_birth_success_and_failure() -> None:
    events = typed_lifecycle_events_from_reference_contexts(
        [
            {
                "generation": 2,
                "tick": 3,
                "event_type": "birth_success",
                "parent_id": 10,
                "child_id": 11,
                "lineage_id": 10,
                "lineage_depth": 1,
                "birth_success": 1,
            },
            {
                "generation": 2,
                "tick": 4,
                "event_type": "birth_fail_no_space",
                "failure_reason": "birth_fail_no_space",
                "parent_id": 12,
                "child_id": -1,
                "birth_failed": 1,
            },
        ]
    )

    assert events["event_type"].tolist() == [
        LifecycleEventType.BIRTH_COMMITTED,
        LifecycleEventType.BIRTH_FAILED,
    ]
    assert events["generation"].tolist() == [2, 2]
    assert events["tick"].tolist() == [3, 4]
    assert events["parent_agent_id"].tolist() == [10, -1]
    assert events["child_agent_id"].tolist() == [11, -1]
    assert events["reason_code"].tolist() == [-1, 1]
    assert events["success"].tolist() == [1, 0]


def test_reference_contexts_preserve_complete_birth_failure_mutation_and_death_fields() -> None:
    events = typed_lifecycle_events_from_reference_contexts(
        [
            {
                "generation": 2,
                "tick": 0,
                "event_type": "birth_success",
                "parent_id": 10,
                "child_id": 11,
                "parent_slot": 4,
                "child_slot": 9,
                "lineage_id": "L00000003",
                "lineage_depth": 2,
            },
            {
                "generation": 2,
                "tick": 1,
                "event_type": "mutation",
                "parent_id": 10,
                "child_id": 11,
                "child_slot": 9,
                "lineage_founder_id": 3,
                "lineage_depth": 2,
                "mutation_delta": 0.25,
            },
            {
                "generation": 2,
                "tick": 2,
                "event_type": "birth_fail_no_space",
                "parent_id": 12,
                "parent_slot": 7,
                "lineage_founder_id": 4,
                "lineage_depth": 1,
            },
            {
                "generation": 2,
                "tick": 3,
                "event_type": "death_committed",
                "agent_id": 13,
                "slot": 8,
                "lineage_founder_id": 5,
                "lineage_depth": 3,
                "reason_code": 2,
                "value": 2.0,
            },
        ]
    )

    assert events["event_type"].tolist() == [1, 2, 3, 4]
    assert events["sequence"].tolist() == [0, 1, 2, 3]
    assert events["actor_agent_id"].tolist() == [10, 11, 12, 13]
    assert events["target_agent_id"].tolist() == [11, 11, -1, -1]
    assert events["actor_slot"].tolist() == [4, 9, 7, 8]
    assert events["target_slot"].tolist() == [9, 9, -1, -1]
    assert events["parent_agent_id"].tolist() == [10, 10, -1, -1]
    assert events["child_agent_id"].tolist() == [11, 11, -1, -1]
    assert events["lineage_founder_id"].tolist() == [3, 3, 4, 5]
    assert events["lineage_depth"].tolist() == [2, 2, 1, 3]
    assert events["reason_code"].tolist() == [-1, -1, 1, 2]


def test_reference_completion_restores_legacy_stripped_action_death_fields() -> None:
    events = typed_lifecycle_events_from_legacy_tape(
        {
            "metadata": {"event_tape_active_count": 1},
            "tensors": {
                "event_tape_type": torch.tensor([4], dtype=torch.int32),
                "event_tape_generation": torch.tensor([3], dtype=torch.int64),
                "event_tape_tick": torch.tensor([7], dtype=torch.int64),
                "event_tape_actor_id": torch.tensor([12], dtype=torch.int64),
                "event_tape_target_id": torch.tensor([-1], dtype=torch.int64),
                "event_tape_value": torch.tensor([2.0], dtype=torch.float32),
            },
        }
    )

    completed = complete_reference_typed_lifecycle_events(
        events,
        {12: {"slot": 6, "lineage_founder_id": 4, "lineage_depth": 2, "parent_agent_id": 10}},
        require_complete=True,
    )

    assert completed["actor_slot"].tolist() == [6]
    assert completed["lineage_founder_id"].tolist() == [4]
    assert completed["lineage_depth"].tolist() == [2]
    assert completed["reason_code"].tolist() == [2]


def test_typed_lifecycle_rows_and_writers_resume_from_cursor(tmp_path) -> None:
    events = typed_lifecycle_events_from_reference_contexts(
        [
            {"generation": 1, "tick": 0, "event_type": "birth_success", "parent_id": 10, "child_id": 11},
            {
                "generation": 1,
                "tick": 1,
                "event_type": "birth_fail_no_space",
                "failure_reason": "birth_fail_no_space",
                "parent_id": 12,
            },
        ]
    )

    rows = typed_lifecycle_events_to_rows(events)
    assert [row["sequence"] for row in rows] == [0, 1]

    jsonl = tmp_path / "events.jsonl"
    csv_path = tmp_path / "events.csv"
    cursor = write_typed_lifecycle_events_jsonl(jsonl, events, start=0, stop=1)
    cursor = write_typed_lifecycle_events_jsonl(jsonl, events, start=cursor)
    csv_cursor = write_typed_lifecycle_events_csv(csv_path, events, start=0, stop=1)
    csv_cursor = write_typed_lifecycle_events_csv(csv_path, events, start=csv_cursor)

    json_rows = [json.loads(line) for line in jsonl.read_text(encoding="utf-8").splitlines()]
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        csv_rows = list(csv.DictReader(handle))

    assert cursor == 2
    assert csv_cursor == 2
    assert [row["sequence"] for row in json_rows] == [0, 1]
    assert [int(row["sequence"]) for row in csv_rows] == [0, 1]
    assert len(json_rows) == 2
    assert len(csv_rows) == 2


def test_typed_lifecycle_events_from_projection_strips_prefix_and_active_count() -> None:
    projection = {
        "metadata": {"typed_lifecycle_event_active_count": 1},
        "tensors": {
            "typed_lifecycle_event_event_type": torch.tensor([1, 3], dtype=torch.int32),
            "typed_lifecycle_event_generation": torch.tensor([2, 2], dtype=torch.int64),
            "typed_lifecycle_event_tick": torch.tensor([0, 1], dtype=torch.int64),
            "typed_lifecycle_event_sequence": torch.tensor([0, 1], dtype=torch.int64),
            "typed_lifecycle_event_actor_agent_id": torch.tensor([10, 12], dtype=torch.int64),
            "typed_lifecycle_event_target_agent_id": torch.tensor([11, -1], dtype=torch.int64),
            "typed_lifecycle_event_actor_slot": torch.tensor([-1, -1], dtype=torch.int64),
            "typed_lifecycle_event_target_slot": torch.tensor([-1, -1], dtype=torch.int64),
            "typed_lifecycle_event_parent_agent_id": torch.tensor([10, -1], dtype=torch.int64),
            "typed_lifecycle_event_child_agent_id": torch.tensor([11, -1], dtype=torch.int64),
            "typed_lifecycle_event_lineage_founder_id": torch.tensor([-1, -1], dtype=torch.int64),
            "typed_lifecycle_event_lineage_depth": torch.tensor([-1, -1], dtype=torch.int32),
            "typed_lifecycle_event_subtype_code": torch.tensor([1, 3], dtype=torch.int32),
            "typed_lifecycle_event_reason_code": torch.tensor([-1, 1], dtype=torch.int32),
            "typed_lifecycle_event_success": torch.tensor([1, 0], dtype=torch.int8),
            "typed_lifecycle_event_value": torch.tensor([0.0, 1.0], dtype=torch.float32),
        },
    }

    events = typed_lifecycle_events_from_projection(projection)

    assert events["event_type"].tolist() == [1]
    assert events["sequence"].tolist() == [0]


def test_typed_lifecycle_events_from_projection_falls_back_to_legacy_tape() -> None:
    projection = {
        "metadata": {"event_tape_active_count": 1, "generation": 3, "tick": 4},
        "tensors": {
            "event_tape_type": torch.tensor([3], dtype=torch.int32),
            "event_tape_actor_id": torch.tensor([10], dtype=torch.int64),
            "event_tape_target_id": torch.tensor([-1], dtype=torch.int64),
            "event_tape_value": torch.tensor([1.0], dtype=torch.float32),
        },
    }

    events = typed_lifecycle_events_from_projection(projection)

    assert events["event_type"].tolist() == [3]
    assert events["generation"].tolist() == [3]
    assert events["reason_code"].tolist() == [1]


def test_typed_lifecycle_public_outputs_reconstruct_missing_counterpart(tmp_path) -> None:
    events = typed_lifecycle_events_from_reference_contexts(
        [
            {"generation": 1, "tick": 0, "event_type": "birth_success", "parent_id": 10, "child_id": 11},
            {"generation": 1, "tick": 1, "event_type": "birth_fail_no_space", "parent_id": 12},
        ]
    )
    jsonl = tmp_path / "events.jsonl"
    csv_path = tmp_path / "events.csv"

    first = write_typed_lifecycle_public_outputs(jsonl, csv_path, events)
    csv_path.unlink()
    second = write_typed_lifecycle_public_outputs(jsonl, csv_path, events)
    jsonl.unlink()
    third = write_typed_lifecycle_public_outputs(jsonl, csv_path, events)

    assert first["end_cursor"] == 2
    assert second["cursor_source"] == "existing_file_row_count"
    assert second["jsonl_existed"] is True
    assert second["csv_existed"] is False
    assert third["jsonl_existed"] is False
    assert third["csv_existed"] is True
    assert len(jsonl.read_text(encoding="utf-8").splitlines()) == 2
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        assert len(list(csv.DictReader(handle))) == 2


def test_typed_lifecycle_public_outputs_obey_checkpoint_transit_trace_policy(tmp_path) -> None:
    contract = lifecycle_profile_contract_v1(
        CHECKPOINT_TRANSIT,
        total_max_bytes=10_000,
        checkpoint_schedule=[1],
    )
    (tmp_path / "artifact_lifecycle_contract.json").write_text(
        json.dumps(contract, sort_keys=True), encoding="utf-8"
    )
    events = typed_lifecycle_events_from_reference_contexts(
        [{"generation": 1, "tick": 0, "event_type": "birth_success", "parent_id": 10, "child_id": 11}]
    )

    with pytest.raises(RuntimeError, match="forbids"):
        write_typed_lifecycle_public_outputs(
            tmp_path / "metrics" / "typed_lifecycle_events.jsonl",
            tmp_path / "metrics" / "typed_lifecycle_events.csv",
            events,
        )

    violation = json.loads(
        (tmp_path / "lifecycle_storage_violation_receipt.json").read_text(
            encoding="utf-8"
        )
    )
    assert violation["artifact_family"] == "native_runtime.typed_lifecycle_events"
    assert violation["owner_writer"] == "write_typed_lifecycle_public_outputs"
    assert violation["automatic_prune_allowed"] is False

    debug_root = tmp_path / "debug"
    debug_contract = lifecycle_profile_contract_v1(
        DEBUG_FORENSIC,
        total_max_bytes=10_000_000,
        checkpoint_schedule=[1],
    )
    (debug_root / "artifact_lifecycle_contract.json").parent.mkdir(parents=True)
    (debug_root / "artifact_lifecycle_contract.json").write_text(
        json.dumps(debug_contract, sort_keys=True), encoding="utf-8"
    )
    output = write_typed_lifecycle_public_outputs(
        debug_root / "metrics" / "typed_lifecycle_events.jsonl",
        debug_root / "metrics" / "typed_lifecycle_events.csv",
        events,
    )
    assert output["end_cursor"] == 1


def test_typed_lifecycle_public_outputs_fail_closed_on_conflicting_rows(tmp_path) -> None:
    events = typed_lifecycle_events_from_reference_contexts(
        [{"generation": 1, "tick": 0, "event_type": "birth_success", "parent_id": 10, "child_id": 11}]
    )
    jsonl = tmp_path / "events.jsonl"
    csv_path = tmp_path / "events.csv"
    write_typed_lifecycle_events_jsonl(jsonl, events)
    csv_path.write_text("schema_version,event_type\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="row count mismatch"):
        write_typed_lifecycle_public_outputs(jsonl, csv_path, events)


def test_typed_lifecycle_public_outputs_fail_closed_on_cursor_conflict(tmp_path) -> None:
    events = typed_lifecycle_events_from_reference_contexts(
        [{"generation": 1, "tick": 0, "event_type": "birth_success", "parent_id": 10, "child_id": 11}]
    )
    jsonl = tmp_path / "events.jsonl"
    csv_path = tmp_path / "events.csv"
    write_typed_lifecycle_public_outputs(jsonl, csv_path, events)

    with pytest.raises(RuntimeError, match="conflicts with cursor"):
        write_typed_lifecycle_public_outputs(jsonl, csv_path, events, cursor=0)


def test_typed_lifecycle_public_outputs_fail_closed_on_canonical_prefix_conflict(tmp_path) -> None:
    events = typed_lifecycle_events_from_reference_contexts(
        [{"generation": 1, "tick": 0, "event_type": "birth_success", "parent_id": 10, "child_id": 11}]
    )
    jsonl = tmp_path / "events.jsonl"
    csv_path = tmp_path / "events.csv"
    write_typed_lifecycle_public_outputs(jsonl, csv_path, events)
    row = json.loads(jsonl.read_text(encoding="utf-8"))
    row["child_agent_id"] = 99
    jsonl.write_text(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    before = (jsonl.read_bytes(), csv_path.read_bytes())

    with pytest.raises(RuntimeError, match="jsonl prefix conflicts with canonical tape"):
        write_typed_lifecycle_public_outputs(jsonl, csv_path, events)

    assert (jsonl.read_bytes(), csv_path.read_bytes()) == before


def test_typed_lifecycle_public_outputs_handle_empty_and_repeated_resume(tmp_path) -> None:
    events = empty_typed_lifecycle_event_tensors()
    jsonl = tmp_path / "events.jsonl"
    csv_path = tmp_path / "events.csv"

    first = write_typed_lifecycle_public_outputs(jsonl, csv_path, events)
    second = write_typed_lifecycle_public_outputs(jsonl, csv_path, events)

    assert first["start_cursor"] == 0
    assert first["end_cursor"] == 0
    assert second["cursor_source"] == "existing_file_row_count"
    assert second["rows_written"] == 0
    assert jsonl.read_text(encoding="utf-8") == ""
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        assert list(csv.DictReader(handle)) == []


def test_typed_lifecycle_artifact_harness_compares_projection_and_public_rows(tmp_path) -> None:
    events = typed_lifecycle_events_from_reference_contexts(
        [{"generation": 1, "tick": 0, "event_type": "birth_success", "parent_id": 10, "child_id": 11}]
    )
    projection = {
        "metadata": {"typed_lifecycle_event_active_count": 1},
        "tensors": {f"typed_lifecycle_event_{name}": tensor for name, tensor in events.items()},
    }
    jsonl = tmp_path / "events.jsonl"
    csv_path = tmp_path / "events.csv"
    write_typed_lifecycle_public_outputs(jsonl, csv_path, events)

    left = typed_lifecycle_artifact_from_projection(projection, source="left", jsonl_path=jsonl, csv_path=csv_path)
    right = typed_lifecycle_artifact_from_projection(projection, source="right", jsonl_path=jsonl, csv_path=csv_path)
    comparison = compare_typed_lifecycle_artifacts(left, right, require_public_rows=True)

    assert comparison["ok"] is True
    assert comparison["expected"]["active_count"] == 1


def test_typed_lifecycle_artifact_derives_counts_and_summary() -> None:
    events = typed_lifecycle_events_from_legacy_tape(
        {
            "metadata": {"generation": 5, "tick": 7, "event_tape_active_count": 4},
            "tensors": {
                "event_tape_type": torch.tensor([2, 1, 3, 4], dtype=torch.int32),
                "event_tape_actor_id": torch.tensor([22, 11, 33, 44], dtype=torch.int64),
                "event_tape_target_id": torch.tensor([22, 22, -1, -1], dtype=torch.int64),
                "event_tape_value": torch.tensor([0.0, 0.0, 1.0, 2.0], dtype=torch.float32),
            },
        }
    )
    projection = {
        "metadata": {"typed_lifecycle_event_active_count": 4},
        "tensors": {f"typed_lifecycle_event_{name}": tensor for name, tensor in events.items()},
    }

    artifact = typed_lifecycle_artifact_from_projection(projection, source="typed")
    semantic_rows = lifecycle_semantic_rows_from_artifact(artifact)
    counts = lifecycle_counts_from_artifact(artifact)
    summary = final_lifecycle_summary_from_artifact(artifact, final_alive_count=41)

    assert [row["event_type"] for row in semantic_rows] == [2, 1, 3, 4]
    assert counts["mutations"] == 1
    assert counts["successful_births"] == 1
    assert counts["failed_births"] == 1
    assert counts["deaths"] == 1
    assert summary["final_alive_count"] == 41
    assert summary["event_cursor"] == 4


def test_typed_lifecycle_artifact_derives_lineage_funnel_semantic_rows() -> None:
    events = typed_lifecycle_events_from_reference_contexts(
        [
            {
                "generation": 2,
                "tick": 3,
                "event_type": "birth_success",
                "parent_id": 10,
                "child_id": 11,
                "lineage_id": 10,
                "lineage_depth": 1,
            },
            {
                "generation": 2,
                "tick": 4,
                "event_type": "birth_fail_no_space",
                "failure_reason": "birth_fail_no_space",
                "parent_id": 12,
            },
        ]
    )
    projection = {
        "metadata": {"typed_lifecycle_event_active_count": 2},
        "tensors": {f"typed_lifecycle_event_{name}": tensor for name, tensor in events.items()},
    }

    artifact = typed_lifecycle_artifact_from_projection(projection, source="typed")
    rows = lineage_funnel_semantic_rows_from_artifact(artifact)

    assert [row["event_type"] for row in rows] == ["birth_success", "birth_failed"]
    assert [row["parent_id"] for row in rows] == [10, 12]
    assert [row["child_id"] for row in rows] == [11, -1]
    assert [row["birth_success"] for row in rows] == [1, 0]
    assert [row["failure_reason_code"] for row in rows] == [-1, 1]


def test_checkpoint_sidecar_semantics_extracts_authoritative_event_fields() -> None:
    events = typed_lifecycle_events_from_reference_contexts(
        [{"generation": 1, "tick": 0, "event_type": "birth_success", "parent_id": 10, "child_id": 11}]
    )
    projection = {
        "metadata": {
            "typed_lifecycle_event_schema_version": "eiel_typed_lifecycle_event_v1",
            "typed_lifecycle_event_active_count": 1,
            "event_tape_active_count": 1,
            "event_tape_cursor": 1,
            "agent_grid_phase_contract_schema_version": "eiel_agent_grid_phase_contract_v1",
            "agent_grid_authority_epoch": "generation:1:tick:0",
        },
        "tensors": {f"typed_lifecycle_event_{name}": tensor for name, tensor in events.items()},
    }

    semantics = checkpoint_sidecar_semantics_from_projection(projection)

    assert semantics["event_schema_version"] == "eiel_typed_lifecycle_event_v1"
    assert semantics["active_event_count"] == 1
    assert semantics["event_cursor"] == 1
    assert semantics["agent_grid_authority_epoch"] == "generation:1:tick:0"


def test_final_lineage_and_summary_json_semantics_are_implementation_neutral() -> None:
    projection = {
        "schema_version": "native_checkpoint_projection",
        "metadata": {
            "world": "flat",
            "generation": 2,
            "tick": 3,
            "typed_lifecycle_event_active_count": 1,
            "event_tape_cursor": 1,
            "agent_grid_phase_contract_schema_version": "eiel_agent_grid_phase_contract_v1",
            "agent_grid_authority_epoch": "generation:2:tick:3",
        },
        "tensors": {
            "alive_mask": torch.tensor([True, False, True]),
            "slot_to_agent_id": torch.tensor([10, -1, 12], dtype=torch.int64),
            "lineage_founder_id": torch.tensor([10, -1, 10], dtype=torch.int32),
            "lineage_depth": torch.tensor([0, 0, 1], dtype=torch.int32),
            "offspring_count": torch.tensor([1, 0, 0], dtype=torch.int32),
            "typed_lifecycle_event_event_type": torch.tensor([1], dtype=torch.int32),
            "typed_lifecycle_event_generation": torch.tensor([2], dtype=torch.int64),
            "typed_lifecycle_event_tick": torch.tensor([3], dtype=torch.int64),
            "typed_lifecycle_event_sequence": torch.tensor([0], dtype=torch.int64),
            "typed_lifecycle_event_actor_agent_id": torch.tensor([10], dtype=torch.int64),
            "typed_lifecycle_event_target_agent_id": torch.tensor([12], dtype=torch.int64),
            "typed_lifecycle_event_actor_slot": torch.tensor([0], dtype=torch.int64),
            "typed_lifecycle_event_target_slot": torch.tensor([2], dtype=torch.int64),
            "typed_lifecycle_event_parent_agent_id": torch.tensor([10], dtype=torch.int64),
            "typed_lifecycle_event_child_agent_id": torch.tensor([12], dtype=torch.int64),
            "typed_lifecycle_event_lineage_founder_id": torch.tensor([10], dtype=torch.int64),
            "typed_lifecycle_event_lineage_depth": torch.tensor([1], dtype=torch.int32),
            "typed_lifecycle_event_subtype_code": torch.tensor([1], dtype=torch.int32),
            "typed_lifecycle_event_reason_code": torch.tensor([-1], dtype=torch.int32),
            "typed_lifecycle_event_success": torch.tensor([1], dtype=torch.int8),
            "typed_lifecycle_event_value": torch.tensor([0.0], dtype=torch.float32),
        },
    }

    lineage = final_lineage_summary_semantics_from_projection(projection)
    summary = final_summary_json_semantics_from_projection(projection)

    assert lineage["surviving_lineage_ids"] == [10]
    assert lineage["lineages"][0]["member_ids"] == [10, 12]
    assert lineage["lineages"][0]["cumulative_births"] == 1
    authoritative = summary["authoritative_semantic_fields"]
    assert authoritative["typed_event_active_count"] == 1
    assert authoritative["final_lineage_summary"] == lineage
