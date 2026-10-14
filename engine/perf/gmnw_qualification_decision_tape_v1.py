"""Strict CPU analyzer for the bounded GMNW qualification decision tape v1."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from typing import Any, Final

import torch


GMNW_QUALIFICATION_DECISION_TAPE_SCHEMA_VERSION_V1: Final[str] = (
    "eiel_gmnw_qualification_decision_tape_v1"
)
GMNW_QUALIFICATION_DECISION_RECEIPT_SCHEMA_VERSION_V1: Final[str] = (
    "eiel_gmnw_qualification_decision_receipt_v1"
)
GMNW_QUALIFICATION_DECISION_ROW_RECEIPT_SCHEMA_VERSION_V1: Final[str] = (
    "eiel_gmnw_qualification_decision_row_receipt_v1"
)
GMNW_QUALIFICATION_INITIAL_STREAM_COUNT_V1: Final[int] = 64
GMNW_QUALIFICATION_EPISODES_PER_STREAM_V1: Final[int] = 48
GMNW_QUALIFICATION_DECISION_ROWS_V1: Final[int] = (
    GMNW_QUALIFICATION_INITIAL_STREAM_COUNT_V1
    * GMNW_QUALIFICATION_EPISODES_PER_STREAM_V1
)
GMNW_QUALIFICATION_DECISION_COLUMNS_V1: Final[tuple[str, ...]] = (
    "stream_id",
    "episode_index",
    "cue_identity",
    "selected_private_action",
    "correct_action",
    "correctness",
)
GMNW_QUALIFICATION_DECISION_TENSOR_NAMES_V1: Final[tuple[str, ...]] = (
    "gmnw_qualification_decision_present",
    "gmnw_qualification_decision_rows",
    "gmnw_qualification_decision_status",
)
GMNW_QUALIFICATION_CUE_TO_ACTION_V1: Final[tuple[int, ...]] = (0, 2, 1, 3)
GMNW_QUALIFICATION_PRIVATE_ACTION_COUNT_V1: Final[int] = 7
GMNW_QUALIFICATION_ABSENT_SENTINEL_V1: Final[int] = -1
GMNW_QUALIFICATION_STATUS_SUCCESS_V1: Final[int] = 0
GMNW_QUALIFICATION_DECISION_ROW_RECEIPT_FIELDS_V1: Final[tuple[str, ...]] = (
    "cell_index",
    "row_index",
    *GMNW_QUALIFICATION_DECISION_COLUMNS_V1,
)
GMNW_QUALIFICATION_DECISION_ROW_RECEIPT_ENCODING_V1: Final[str] = (
    "ascii_canonical_json_lines_header_then_present_rows"
)


class GmnwQualificationDecisionTapeV1Error(ValueError):
    """The supplied qualification decision tape violates the v1 contract."""


def qualification_decision_receipt_schema_v1() -> dict[str, Any]:
    """Return the concise, JSON-safe analyzer receipt schema."""

    return {
        "schema_version": GMNW_QUALIFICATION_DECISION_RECEIPT_SCHEMA_VERSION_V1,
        "top_level_fields": [
            "schema_version",
            "tape_schema_version",
            "row_receipt_schema_version",
            "row_receipt_sha256",
            "cell_count",
            "expected_decision_count",
            "D",
            "C",
            "missing_count",
            "coverage",
            "cells",
        ],
        "cell_fields": [
            "cell_index",
            "cue_alphabet",
            "D",
            "C",
            "D_c",
            "C_c",
            "cue_accuracy",
            "accuracy",
            "cue_balanced_accuracy",
            "observed_cue_count",
            "missing_count",
            "coverage",
        ],
        "row_receipt_fields": list(
            GMNW_QUALIFICATION_DECISION_ROW_RECEIPT_FIELDS_V1
        ),
        "row_receipt_encoding": (
            GMNW_QUALIFICATION_DECISION_ROW_RECEIPT_ENCODING_V1
        ),
        "absent_cue_accuracy": None,
        "cue_balanced_accuracy": "mean_over_observed_cues",
    }


def _fail(path: str, message: str) -> GmnwQualificationDecisionTapeV1Error:
    return GmnwQualificationDecisionTapeV1Error(f"{path}: {message}")


def _canonical_json_line(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
        + b"\n"
    )


def _require_tensor(
    value: object,
    *,
    path: str,
    dtypes: tuple[torch.dtype, ...],
    shape: tuple[int, ...] | None = None,
    rank: int | None = None,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise _fail(path, "must be a torch tensor")
    if value.dtype not in dtypes:
        expected = " or ".join(str(dtype).removeprefix("torch.") for dtype in dtypes)
        raise _fail(path, f"must have exact dtype {expected}")
    if value.layout != torch.strided:
        raise _fail(path, "must use strided tensor layout")
    if rank is not None and value.ndim != rank:
        raise _fail(path, f"must have rank {rank}")
    if shape is not None and tuple(value.shape) != shape:
        raise _fail(path, f"must have exact shape {list(shape)}")
    return value.detach().to(device="cpu").contiguous()


def _cue_alphabets(
    value: object,
    *,
    cell_count: int,
) -> tuple[int, ...]:
    if isinstance(value, torch.Tensor):
        tensor = _require_tensor(
            value,
            path="cue_alphabet_by_cell",
            dtypes=(torch.int32,),
            shape=(cell_count,),
        )
        values = tuple(int(item) for item in tensor.tolist())
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if len(value) != cell_count:
            raise _fail(
                "cue_alphabet_by_cell",
                f"must contain exactly {cell_count} values",
            )
        if any(type(item) is not int for item in value):
            raise _fail(
                "cue_alphabet_by_cell", "values must be exact integers"
            )
        values = tuple(value)
    else:
        raise _fail(
            "cue_alphabet_by_cell", "must be an int32 tensor or integer sequence"
        )
    if any(item not in (2, 4) for item in values):
        raise _fail("cue_alphabet_by_cell", "values must be exactly 2 or 4")
    return values


def _validated_tape(
    tape: Mapping[str, object],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    if not isinstance(tape, Mapping):
        raise _fail("tape", "must be a mapping")
    if any(type(key) is not str for key in tape):
        raise _fail("tape", "keys must be exact strings")
    expected = set(GMNW_QUALIFICATION_DECISION_TENSOR_NAMES_V1)
    actual = set(tape)
    if actual != expected:
        raise _fail(
            "tape",
            f"field set mismatch; missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}",
        )

    present = _require_tensor(
        tape[GMNW_QUALIFICATION_DECISION_TENSOR_NAMES_V1[0]],
        path=GMNW_QUALIFICATION_DECISION_TENSOR_NAMES_V1[0],
        dtypes=(torch.int8, torch.bool),
        rank=2,
    )
    if present.shape[0] <= 0:
        raise _fail(
            GMNW_QUALIFICATION_DECISION_TENSOR_NAMES_V1[0],
            "cell dimension must be positive",
        )
    cell_count = int(present.shape[0])
    expected_present_shape = (cell_count, GMNW_QUALIFICATION_DECISION_ROWS_V1)
    if tuple(present.shape) != expected_present_shape:
        raise _fail(
            GMNW_QUALIFICATION_DECISION_TENSOR_NAMES_V1[0],
            f"must have exact shape {list(expected_present_shape)}",
        )
    rows = _require_tensor(
        tape[GMNW_QUALIFICATION_DECISION_TENSOR_NAMES_V1[1]],
        path=GMNW_QUALIFICATION_DECISION_TENSOR_NAMES_V1[1],
        dtypes=(torch.int64,),
        shape=(
            cell_count,
            GMNW_QUALIFICATION_DECISION_ROWS_V1,
            len(GMNW_QUALIFICATION_DECISION_COLUMNS_V1),
        ),
    )
    status = _require_tensor(
        tape[GMNW_QUALIFICATION_DECISION_TENSOR_NAMES_V1[2]],
        path=GMNW_QUALIFICATION_DECISION_TENSOR_NAMES_V1[2],
        dtypes=(torch.int32,),
        shape=(cell_count,),
    )
    if bool(torch.count_nonzero(status).item()):
        first = int(torch.nonzero(status, as_tuple=False)[0, 0].item())
        raise _fail(
            f"{GMNW_QUALIFICATION_DECISION_TENSOR_NAMES_V1[2]}[{first}]",
            f"must equal {GMNW_QUALIFICATION_STATUS_SUCCESS_V1}",
        )
    if present.dtype == torch.int8:
        invalid = (present != 0) & (present != 1)
        if bool(torch.any(invalid).item()):
            cell, row = (int(item) for item in torch.nonzero(invalid)[0].tolist())
            raise _fail(
                f"{GMNW_QUALIFICATION_DECISION_TENSOR_NAMES_V1[0]}"
                f"[{cell},{row}]",
                "must equal 0 or 1",
            )
    return present.to(dtype=torch.bool), rows, status, cell_count


def analyze_gmnw_qualification_decision_tape_v1(
    tape: Mapping[str, object],
    *,
    cue_alphabet_by_cell: object,
    require_complete: bool = True,
) -> dict[str, Any]:
    """Validate the tape and return deterministic per-cell decision metrics."""

    if type(require_complete) is not bool:
        raise _fail("require_complete", "must be an exact boolean")
    present, rows, _status, cell_count = _validated_tape(tape)
    cue_alphabets = _cue_alphabets(
        cue_alphabet_by_cell,
        cell_count=cell_count,
    )
    digest = hashlib.sha256()
    digest.update(
        _canonical_json_line(
            {
                "schema_version": (
                    GMNW_QUALIFICATION_DECISION_ROW_RECEIPT_SCHEMA_VERSION_V1
                ),
                "encoding": GMNW_QUALIFICATION_DECISION_ROW_RECEIPT_ENCODING_V1,
                "fields": list(
                    GMNW_QUALIFICATION_DECISION_ROW_RECEIPT_FIELDS_V1
                ),
                "cell_count": cell_count,
                "rows_per_cell": GMNW_QUALIFICATION_DECISION_ROWS_V1,
                "cue_alphabet_by_cell": list(cue_alphabets),
            }
        )
    )

    cell_receipts: list[dict[str, Any]] = []
    total_decisions = 0
    total_correct = 0
    sentinel = GMNW_QUALIFICATION_ABSENT_SENTINEL_V1
    for cell_index, cue_alphabet in enumerate(cue_alphabets):
        decision_count = 0
        correct_count = 0
        decisions_by_cue = [0] * cue_alphabet
        correct_by_cue = [0] * cue_alphabet
        for row_index in range(GMNW_QUALIFICATION_DECISION_ROWS_V1):
            values = tuple(int(value) for value in rows[cell_index, row_index].tolist())
            row_path = (
                f"{GMNW_QUALIFICATION_DECISION_TENSOR_NAMES_V1[1]}"
                f"[{cell_index},{row_index}]"
            )
            if not bool(present[cell_index, row_index].item()):
                if values != (sentinel,) * len(GMNW_QUALIFICATION_DECISION_COLUMNS_V1):
                    raise _fail(row_path, "absent row must contain only -1 sentinels")
                continue

            stream_id, episode_index, cue, selected, correct, correctness = values
            expected_stream = row_index // GMNW_QUALIFICATION_EPISODES_PER_STREAM_V1
            expected_episode = row_index % GMNW_QUALIFICATION_EPISODES_PER_STREAM_V1
            if not 0 <= stream_id < GMNW_QUALIFICATION_INITIAL_STREAM_COUNT_V1:
                raise _fail(f"{row_path}.stream_id", "out of range [0,63]")
            if stream_id != expected_stream:
                raise _fail(
                    f"{row_path}.stream_id",
                    f"must equal index-derived stream_id {expected_stream}",
                )
            if not 0 <= episode_index < GMNW_QUALIFICATION_EPISODES_PER_STREAM_V1:
                raise _fail(f"{row_path}.episode_index", "out of range [0,47]")
            if episode_index != expected_episode:
                raise _fail(
                    f"{row_path}.episode_index",
                    f"must equal index-derived episode_index {expected_episode}",
                )
            if not 0 <= cue < cue_alphabet:
                raise _fail(
                    f"{row_path}.cue_identity",
                    f"out of range [0,{cue_alphabet - 1}]",
                )
            if not 0 <= selected < GMNW_QUALIFICATION_PRIVATE_ACTION_COUNT_V1:
                raise _fail(
                    f"{row_path}.selected_private_action", "out of range [0,6]"
                )
            expected_correct = GMNW_QUALIFICATION_CUE_TO_ACTION_V1[cue]
            if correct != expected_correct:
                raise _fail(
                    f"{row_path}.correct_action",
                    f"must equal cue mapping value {expected_correct}",
                )
            if correctness not in (0, 1):
                raise _fail(f"{row_path}.correctness", "must equal 0 or 1")
            recomputed = int(selected == expected_correct)
            if correctness != recomputed:
                raise _fail(
                    f"{row_path}.correctness",
                    f"must equal recomputed correctness {recomputed}",
                )

            digest.update(
                _canonical_json_line(
                    [cell_index, row_index, *values]
                )
            )
            decision_count += 1
            correct_count += correctness
            decisions_by_cue[cue] += 1
            correct_by_cue[cue] += correctness

        if decision_count == 0:
            raise _fail(f"cells[{cell_index}].D", "must contain at least one decision")
        cue_accuracy = [
            (correct / decisions if decisions else None)
            for decisions, correct in zip(
                decisions_by_cue, correct_by_cue, strict=True
            )
        ]
        observed_accuracies = [value for value in cue_accuracy if value is not None]
        missing_count = GMNW_QUALIFICATION_DECISION_ROWS_V1 - decision_count
        if require_complete and missing_count:
            raise _fail(
                f"cells[{cell_index}].coverage",
                f"qualification requires all rows; missing {missing_count}",
            )
        if require_complete and len(observed_accuracies) != cue_alphabet:
            raise _fail(
                f"cells[{cell_index}].observed_cue_count",
                "qualification requires every cue",
            )
        cell_receipts.append(
            {
                "cell_index": cell_index,
                "cue_alphabet": cue_alphabet,
                "D": decision_count,
                "C": correct_count,
                "D_c": decisions_by_cue,
                "C_c": correct_by_cue,
                "cue_accuracy": cue_accuracy,
                "accuracy": correct_count / decision_count,
                "cue_balanced_accuracy": sum(observed_accuracies)
                / len(observed_accuracies),
                "observed_cue_count": len(observed_accuracies),
                "missing_count": missing_count,
                "coverage": decision_count
                / GMNW_QUALIFICATION_DECISION_ROWS_V1,
            }
        )
        total_decisions += decision_count
        total_correct += correct_count

    expected_decisions = cell_count * GMNW_QUALIFICATION_DECISION_ROWS_V1
    return {
        "schema_version": GMNW_QUALIFICATION_DECISION_RECEIPT_SCHEMA_VERSION_V1,
        "tape_schema_version": (
            GMNW_QUALIFICATION_DECISION_TAPE_SCHEMA_VERSION_V1
        ),
        "row_receipt_schema_version": (
            GMNW_QUALIFICATION_DECISION_ROW_RECEIPT_SCHEMA_VERSION_V1
        ),
        "row_receipt_sha256": digest.hexdigest(),
        "cell_count": cell_count,
        "expected_decision_count": expected_decisions,
        "D": total_decisions,
        "C": total_correct,
        "missing_count": expected_decisions - total_decisions,
        "coverage": total_decisions / expected_decisions,
        "cells": cell_receipts,
    }


__all__ = [
    "GMNW_QUALIFICATION_ABSENT_SENTINEL_V1",
    "GMNW_QUALIFICATION_CUE_TO_ACTION_V1",
    "GMNW_QUALIFICATION_DECISION_COLUMNS_V1",
    "GMNW_QUALIFICATION_DECISION_RECEIPT_SCHEMA_VERSION_V1",
    "GMNW_QUALIFICATION_DECISION_ROWS_V1",
    "GMNW_QUALIFICATION_DECISION_ROW_RECEIPT_ENCODING_V1",
    "GMNW_QUALIFICATION_DECISION_ROW_RECEIPT_FIELDS_V1",
    "GMNW_QUALIFICATION_DECISION_ROW_RECEIPT_SCHEMA_VERSION_V1",
    "GMNW_QUALIFICATION_DECISION_TAPE_SCHEMA_VERSION_V1",
    "GMNW_QUALIFICATION_DECISION_TENSOR_NAMES_V1",
    "GMNW_QUALIFICATION_EPISODES_PER_STREAM_V1",
    "GMNW_QUALIFICATION_INITIAL_STREAM_COUNT_V1",
    "GMNW_QUALIFICATION_PRIVATE_ACTION_COUNT_V1",
    "GMNW_QUALIFICATION_STATUS_SUCCESS_V1",
    "GmnwQualificationDecisionTapeV1Error",
    "analyze_gmnw_qualification_decision_tape_v1",
    "qualification_decision_receipt_schema_v1",
]
