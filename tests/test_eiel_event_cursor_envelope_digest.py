from __future__ import annotations

from copy import deepcopy

import pytest
import torch

from native_runtime.lifecycle_events import (
    authoritative_event_cursor_envelope_digest_sha256,
    typed_lifecycle_events_from_reference_contexts,
)


def _projection(*, capacity: int = 3, active_count: int = 2) -> dict:
    events = typed_lifecycle_events_from_reference_contexts(
        [
            {
                "generation": 4,
                "tick": index,
                "event_type": "birth_success",
                "parent_id": 10 + index,
                "child_id": 20 + index,
            }
            for index in range(capacity)
        ]
    )
    return {
        "event_cursor": active_count,
        "runtime_event_cursor": 17,
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


def _set_active_count(projection: dict, active_count: int) -> None:
    capacity = len(projection["tensors"]["typed_lifecycle_event_event_type"])
    projection["event_cursor"] = active_count
    projection["metadata"].update(
        {
            "typed_lifecycle_event_active_count": active_count,
            "typed_lifecycle_event_next_sequence": active_count,
            "event_tape_active_count": active_count,
            "event_tape_cursor": active_count,
            "event_tape_inactive_tail_count": capacity - active_count,
        }
    )


def _assert_projection_equal(actual: dict, expected: dict) -> None:
    assert actual.keys() == expected.keys()
    assert actual["event_cursor"] == expected["event_cursor"]
    assert actual["runtime_event_cursor"] == expected["runtime_event_cursor"]
    assert actual["metadata"] == expected["metadata"]
    assert actual["tensors"].keys() == expected["tensors"].keys()
    for name, tensor in actual["tensors"].items():
        assert torch.equal(tensor, expected["tensors"][name]), name


def test_event_cursor_envelope_digest_is_deterministic_for_equivalent_projections() -> (
    None
):
    first = _projection()
    second = deepcopy(first)
    second["metadata"] = dict(reversed(tuple(second["metadata"].items())))
    second["tensors"] = dict(reversed(tuple(second["tensors"].items())))

    first_digest = authoritative_event_cursor_envelope_digest_sha256(first)
    second_digest = authoritative_event_cursor_envelope_digest_sha256(second)

    assert first_digest == second_digest
    assert len(first_digest) == 64


def test_event_cursor_envelope_digest_is_cursor_sequence_and_active_row_sensitive() -> (
    None
):
    baseline = _projection()
    baseline_digest = authoritative_event_cursor_envelope_digest_sha256(baseline)

    active_cursor_changed = deepcopy(baseline)
    _set_active_count(active_cursor_changed, 1)
    runtime_cursor_changed = deepcopy(baseline)
    runtime_cursor_changed["runtime_event_cursor"] += 1
    sequence_changed = deepcopy(baseline)
    sequence_changed["tensors"]["typed_lifecycle_event_sequence"][1] = 99
    active_row_changed = deepcopy(baseline)
    active_row_changed["tensors"]["typed_lifecycle_event_child_agent_id"][1] += 1

    assert (
        authoritative_event_cursor_envelope_digest_sha256(active_cursor_changed)
        != baseline_digest
    )
    assert (
        authoritative_event_cursor_envelope_digest_sha256(runtime_cursor_changed)
        != baseline_digest
    )
    assert (
        authoritative_event_cursor_envelope_digest_sha256(sequence_changed)
        != baseline_digest
    )
    assert (
        authoritative_event_cursor_envelope_digest_sha256(active_row_changed)
        != baseline_digest
    )


@pytest.mark.parametrize(
    ("path", "value", "error"),
    [
        (("event_cursor",), True, TypeError),
        (("runtime_event_cursor",), 1.0, TypeError),
        (("runtime_event_cursor",), -1, RuntimeError),
        (("metadata", "typed_lifecycle_event_next_sequence"), False, TypeError),
    ],
)
def test_event_cursor_envelope_digest_rejects_non_exact_or_negative_counts(
    path: tuple[str, ...],
    value: object,
    error: type[Exception],
) -> None:
    projection = _projection()
    owner = projection if len(path) == 1 else projection[path[0]]
    owner[path[-1]] = value

    with pytest.raises(error, match="integer|nonnegative"):
        authoritative_event_cursor_envelope_digest_sha256(projection)


def test_event_cursor_envelope_digest_rejects_cursor_contradiction() -> None:
    projection = _projection()
    projection["event_cursor"] += 1

    with pytest.raises(RuntimeError, match="authoritative event cursor contradiction"):
        authoritative_event_cursor_envelope_digest_sha256(projection)


@pytest.mark.parametrize("malformation", ("missing", "uneven"))
def test_event_cursor_envelope_digest_rejects_malformed_columns(
    malformation: str,
) -> None:
    projection = _projection()
    column = "typed_lifecycle_event_child_agent_id"
    if malformation == "missing":
        del projection["tensors"][column]
    else:
        projection["tensors"][column] = projection["tensors"][column][:-1]

    with pytest.raises(RuntimeError, match="mandatory columns|inconsistent lengths"):
        authoritative_event_cursor_envelope_digest_sha256(projection)


def test_event_cursor_envelope_digest_does_not_mutate_projection() -> None:
    projection = _projection()
    expected = deepcopy(projection)

    authoritative_event_cursor_envelope_digest_sha256(projection)

    _assert_projection_equal(projection, expected)
