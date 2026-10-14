from __future__ import annotations

import copy
from typing import Any

import pytest
import torch

from engine.perf import gmnw_qualification_decision_tape_v1 as tape_v1


def _tape(
    cue_alphabets: tuple[int, ...] = (2, 4),
    *,
    present_dtype: torch.dtype = torch.int8,
) -> dict[str, torch.Tensor]:
    cells = len(cue_alphabets)
    present = torch.ones(
        (cells, tape_v1.GMNW_QUALIFICATION_DECISION_ROWS_V1),
        dtype=present_dtype,
    )
    rows = torch.empty(
        (cells, tape_v1.GMNW_QUALIFICATION_DECISION_ROWS_V1, 6),
        dtype=torch.int64,
    )
    for cell, cue_alphabet in enumerate(cue_alphabets):
        for row_index in range(tape_v1.GMNW_QUALIFICATION_DECISION_ROWS_V1):
            stream_id, episode_index = divmod(
                row_index, tape_v1.GMNW_QUALIFICATION_EPISODES_PER_STREAM_V1
            )
            cue = row_index % cue_alphabet
            correct_action = tape_v1.GMNW_QUALIFICATION_CUE_TO_ACTION_V1[cue]
            selected_action = correct_action if row_index % 5 else 6
            rows[cell, row_index] = torch.tensor(
                (
                    stream_id,
                    episode_index,
                    cue,
                    selected_action,
                    correct_action,
                    int(selected_action == correct_action),
                ),
                dtype=torch.int64,
            )
    return {
        "gmnw_qualification_decision_present": present,
        "gmnw_qualification_decision_rows": rows,
        "gmnw_qualification_decision_status": torch.zeros(
            (cells,), dtype=torch.int32
        ),
    }


def _analyze(
    tape: dict[str, torch.Tensor],
    cue_alphabets: object = (2, 4),
    *,
    require_complete: bool = True,
) -> dict[str, Any]:
    return tape_v1.analyze_gmnw_qualification_decision_tape_v1(
        tape,
        cue_alphabet_by_cell=cue_alphabets,
        require_complete=require_complete,
    )


def test_full_cpu_tape_reports_exact_counts_balances_and_schema() -> None:
    receipt = _analyze(
        _tape(), torch.tensor([2, 4], dtype=torch.int32)
    )

    assert receipt["schema_version"] == (
        tape_v1.GMNW_QUALIFICATION_DECISION_RECEIPT_SCHEMA_VERSION_V1
    )
    assert receipt["cell_count"] == 2
    assert receipt["expected_decision_count"] == 6144
    assert receipt["D"] == 6144
    assert receipt["C"] == 4914
    assert receipt["missing_count"] == 0
    assert receipt["coverage"] == 1.0
    assert receipt["row_receipt_sha256"] == (
        "f7c06cd19a0e8dbcd46472fa089bbd7720403cfa59c24f6c04f71138c8ed10df"
    )
    assert receipt["cells"][0]["D_c"] == [1536, 1536]
    assert receipt["cells"][0]["C_c"] == [1228, 1229]
    assert receipt["cells"][1]["D_c"] == [768, 768, 768, 768]
    assert receipt["cells"][1]["C_c"] == [614, 614, 614, 615]
    for cell in receipt["cells"]:
        assert cell["D"] == 3072
        assert cell["C"] == 2457
        assert sum(cell["D_c"]) == cell["D"]
        assert sum(cell["C_c"]) == cell["C"]
        assert cell["accuracy"] == pytest.approx(2457 / 3072)
        assert cell["cue_balanced_accuracy"] == pytest.approx(0.7998046875)
        assert cell["observed_cue_count"] == cell["cue_alphabet"]
        assert cell["missing_count"] == 0
        assert cell["coverage"] == 1.0

    schema = tape_v1.qualification_decision_receipt_schema_v1()
    assert schema["row_receipt_fields"] == [
        "cell_index",
        "row_index",
        "stream_id",
        "episode_index",
        "cue_identity",
        "selected_private_action",
        "correct_action",
        "correctness",
    ]
    assert schema["cue_balanced_accuracy"] == "mean_over_observed_cues"


def test_missing_rows_are_allowed_and_counted_exactly() -> None:
    tape = _tape((4,))
    missing = (0, 1, 2, 101, 3071)
    for row_index in missing:
        tape["gmnw_qualification_decision_present"][0, row_index] = 0
        tape["gmnw_qualification_decision_rows"][0, row_index].fill_(-1)

    receipt = _analyze(tape, (4,), require_complete=False)
    cell = receipt["cells"][0]

    assert receipt["D"] == 3072 - len(missing)
    assert receipt["missing_count"] == len(missing)
    assert receipt["coverage"] == pytest.approx((3072 - len(missing)) / 3072)
    assert cell["D"] == 3072 - len(missing)
    assert cell["missing_count"] == len(missing)
    assert sum(cell["D_c"]) == cell["D"]


def test_sparse_tape_balances_only_over_observed_cues() -> None:
    tape = _tape((4,))
    tape["gmnw_qualification_decision_present"].zero_()
    tape["gmnw_qualification_decision_rows"].fill_(-1)
    tape["gmnw_qualification_decision_present"][0, 0] = 1
    tape["gmnw_qualification_decision_rows"][0, 0] = torch.tensor(
        [0, 0, 0, 0, 0, 1], dtype=torch.int64
    )

    cell = _analyze(tape, (4,), require_complete=False)["cells"][0]

    assert cell["D_c"] == [1, 0, 0, 0]
    assert cell["C_c"] == [1, 0, 0, 0]
    assert cell["cue_accuracy"] == [1.0, None, None, None]
    assert cell["cue_balanced_accuracy"] == 1.0
    assert cell["observed_cue_count"] == 1
    assert cell["missing_count"] == 3071


def test_semantically_equal_bool_and_int8_tapes_have_same_receipt_hash() -> None:
    int8_receipt = _analyze(_tape(present_dtype=torch.int8))
    bool_receipt = _analyze(_tape(present_dtype=torch.bool))

    assert int8_receipt["row_receipt_sha256"] == bool_receipt["row_receipt_sha256"]
    assert int8_receipt == bool_receipt


def test_hash_is_deterministic_and_binds_rows_and_missingness() -> None:
    original = _tape((2,))
    repeated = copy.deepcopy(original)
    changed = copy.deepcopy(original)
    changed["gmnw_qualification_decision_rows"][0, 1, 3] = 6
    changed["gmnw_qualification_decision_rows"][0, 1, 5] = 0
    truncated = copy.deepcopy(original)
    truncated["gmnw_qualification_decision_present"][0, 1] = 0
    truncated["gmnw_qualification_decision_rows"][0, 1].fill_(-1)

    original_hash = _analyze(original, (2,))["row_receipt_sha256"]

    assert _analyze(repeated, (2,))["row_receipt_sha256"] == original_hash
    assert _analyze(changed, (2,))["row_receipt_sha256"] != original_hash
    assert (
        _analyze(truncated, (2,), require_complete=False)["row_receipt_sha256"]
        != original_hash
    )


def test_incomplete_tape_fails_closed_for_qualification_by_default() -> None:
    tape = _tape((4,))
    tape["gmnw_qualification_decision_present"][0, 1] = 0
    tape["gmnw_qualification_decision_rows"][0, 1].fill_(-1)

    with pytest.raises(
        tape_v1.GmnwQualificationDecisionTapeV1Error,
        match="qualification requires all rows",
    ):
        _analyze(tape, (4,))


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda tape: tape.pop("gmnw_qualification_decision_status"), "missing"),
        (lambda tape: tape.update(extra=torch.tensor(0)), "extra"),
        (
            lambda tape: tape.__setitem__(
                "gmnw_qualification_decision_present",
                tape["gmnw_qualification_decision_present"].to(torch.uint8),
            ),
            "dtype int8 or bool",
        ),
        (
            lambda tape: tape.__setitem__(
                "gmnw_qualification_decision_rows",
                tape["gmnw_qualification_decision_rows"].to(torch.int32),
            ),
            "dtype int64",
        ),
        (
            lambda tape: tape.__setitem__(
                "gmnw_qualification_decision_status",
                tape["gmnw_qualification_decision_status"].to(torch.int64),
            ),
            "dtype int32",
        ),
        (
            lambda tape: tape.__setitem__(
                "gmnw_qualification_decision_present",
                tape["gmnw_qualification_decision_present"][:, :-1],
            ),
            "shape",
        ),
        (
            lambda tape: tape.__setitem__(
                "gmnw_qualification_decision_rows",
                tape["gmnw_qualification_decision_rows"][:, :, :-1],
            ),
            "shape",
        ),
        (
            lambda tape: tape.__setitem__(
                "gmnw_qualification_decision_status",
                tape["gmnw_qualification_decision_status"][:, None],
            ),
            "shape",
        ),
        (
            lambda tape: tape["gmnw_qualification_decision_status"].fill_(9),
            "must equal 0",
        ),
        (
            lambda tape: tape["gmnw_qualification_decision_present"].fill_(2),
            "must equal 0 or 1",
        ),
    ],
)
def test_malformed_mapping_dtype_shape_status_or_presence_fails_closed(
    mutation: Any,
    match: str,
) -> None:
    tape = _tape((2,))
    mutation(tape)

    with pytest.raises(tape_v1.GmnwQualificationDecisionTapeV1Error, match=match):
        _analyze(tape, (2,))


@pytest.mark.parametrize(
    ("column", "value", "match"),
    [
        (0, 1, "index-derived stream_id"),
        (1, 1, "index-derived episode_index"),
        (2, 2, "cue_identity"),
        (3, -1, "selected_private_action"),
        (3, 7, "selected_private_action"),
        (4, 2, "correct_action"),
        (5, -1, "correctness"),
        (5, 2, "correctness"),
    ],
)
def test_present_row_field_tampering_fails_closed(
    column: int,
    value: int,
    match: str,
) -> None:
    tape = _tape((2,))
    tape["gmnw_qualification_decision_rows"][0, 0, column] = value

    with pytest.raises(tape_v1.GmnwQualificationDecisionTapeV1Error, match=match):
        _analyze(tape, (2,))


def test_correctness_recomputation_fails_closed() -> None:
    tape = _tape((2,))
    tape["gmnw_qualification_decision_rows"][0, 1, 5] = 0

    with pytest.raises(
        tape_v1.GmnwQualificationDecisionTapeV1Error,
        match="recomputed correctness",
    ):
        _analyze(tape, (2,))


def test_absent_row_requires_all_sentinels() -> None:
    tape = _tape((2,))
    tape["gmnw_qualification_decision_present"][0, 10] = 0

    with pytest.raises(
        tape_v1.GmnwQualificationDecisionTapeV1Error,
        match="absent row must contain only -1 sentinels",
    ):
        _analyze(tape, (2,))


@pytest.mark.parametrize(
    ("cue_alphabets", "match"),
    [
        ((2,), "exactly 2 values"),
        ((2, 3), "exactly 2 or 4"),
        ((2, True), "exact integers"),
        (torch.tensor([2, 4], dtype=torch.int64), "dtype int32"),
    ],
)
def test_cue_alphabet_contract_fails_closed(
    cue_alphabets: object,
    match: str,
) -> None:
    with pytest.raises(tape_v1.GmnwQualificationDecisionTapeV1Error, match=match):
        _analyze(_tape(), cue_alphabets)


def test_zero_decisions_per_cell_is_rejected() -> None:
    tape = _tape((2, 4))
    tape["gmnw_qualification_decision_present"][1].zero_()
    tape["gmnw_qualification_decision_rows"][1].fill_(-1)

    with pytest.raises(
        tape_v1.GmnwQualificationDecisionTapeV1Error,
        match=r"cells\[1\]\.D: must contain at least one decision",
    ):
        _analyze(tape)
