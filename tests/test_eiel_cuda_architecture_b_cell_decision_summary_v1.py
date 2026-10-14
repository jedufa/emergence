from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest

from engine.perf.cuda_architecture_b_cell_decision_summary_v1 import (
    CELL_STATUS_UNEXECUTED_V1,
    COMPOSITE_SCIENTIFIC_SCORE_DEFINED_V1,
    CONTAINS_FULL_SIMULATION_STATE_V1,
    CONTRACT_SCHEMA_VERSION_V1,
    CONTRACT_VERSION_V1,
    FULL_STATE_TRANSFER_CLAIMED_V1,
    CellDecisionSummaryContractError,
    cell_decision_summary_layout_v1,
    freeze_cell_decision_summary_v1,
    rank_cell_decision_summary_cpu_oracle_v1,
    validate_cell_decision_summary_v1,
)


ROOT = Path(__file__).resolve().parents[1]
HEADER = ROOT / "engine/perf/cuda_architecture_b_cell_decision_summary_v1.cuh"
POLICY_ID = "test/caller-frozen-lexicographic-basis/v1"
POLICY_DIGEST = bytes(range(32))


def _summary():
    validity = np.array([[1, 1, 0, 0], [1, 1, 1, 1]], dtype=np.uint8)
    status = np.array(
        [[0, 0, 17, CELL_STATUS_UNEXECUTED_V1], [0, 0, 0, 0]],
        dtype=np.int32,
    )
    score_basis = np.array(
        [
            [[5.0, 1.0], [5.0, 2.0], [0.0, 0.0], [0.0, 0.0]],
            [[7.0, 1.0], [7.0, 1.0], [8.0, 0.0], [7.0, 1.0]],
        ],
        dtype=np.float64,
    )
    return freeze_cell_decision_summary_v1(
        validity=validity,
        status=status,
        score_basis=score_basis,
        stable_cell_ids=np.array([40, 10, 30, 20], dtype=np.int64),
        score_policy_id=POLICY_ID,
        score_policy_digest=POLICY_DIGEST,
    )


def test_layout_is_exact_c_contiguous_generation_cell_basis() -> None:
    layout = cell_decision_summary_layout_v1(2, 4, 3)

    assert CONTRACT_SCHEMA_VERSION_V1 == (
        "eiel_cuda_architecture_b_cell_decision_summary_v1"
    )
    assert CONTRACT_VERSION_V1 == 1
    assert layout.validity_shape == (2, 4)
    assert layout.validity_strides_bytes == (4, 1)
    assert layout.status_shape == (2, 4)
    assert layout.status_strides_bytes == (16, 4)
    assert layout.score_basis_shape == (2, 4, 3)
    assert layout.score_basis_strides_bytes == (96, 24, 8)
    assert layout.stable_cell_ids_shape == (4,)
    assert layout.stable_cell_ids_strides_bytes == (8,)
    assert layout.logical_plane_bytes == 2 * 4 * (1 + 4 + 3 * 8) + 4 * 8


def test_freeze_validates_dtypes_shapes_and_makes_independent_snapshots() -> None:
    validity = np.ones((1, 2), dtype=np.uint8)
    status = np.zeros((1, 2), dtype=np.int32)
    basis = np.array([[[2.0], [1.0]]], dtype=np.float64)
    ids = np.array([9, 3], dtype=np.int64)
    summary = freeze_cell_decision_summary_v1(
        validity=validity,
        status=status,
        score_basis=basis,
        stable_cell_ids=ids,
        score_policy_id=POLICY_ID,
        score_policy_digest=POLICY_DIGEST.hex(),
    )

    validity[0, 0] = 0
    basis[0, 0, 0] = 99.0
    assert summary.validity.tolist() == [[1, 1]]
    assert summary.score_basis.tolist() == [[[2.0], [1.0]]]
    assert summary.score_policy_digest == POLICY_DIGEST
    for plane in (
        summary.validity,
        summary.status,
        summary.score_basis,
        summary.stable_cell_ids,
    ):
        assert plane.flags.c_contiguous
        assert not plane.flags.writeable
    validate_cell_decision_summary_v1(summary)


def test_cpu_oracle_is_lexicographic_and_stable_id_breaks_only_exact_ties() -> None:
    rankings = rank_cell_decision_summary_cpu_oracle_v1(
        _summary(),
        expected_policy_id=POLICY_ID,
        expected_policy_digest=POLICY_DIGEST,
    )

    assert [row.stable_cell_id for row in rankings[0]] == [10, 40]
    assert [row.score_basis for row in rankings[0]] == [(5.0, 2.0), (5.0, 1.0)]
    assert [row.stable_cell_id for row in rankings[1]] == [30, 10, 20, 40]
    assert [row.rank_index for row in rankings[1]] == [0, 1, 2, 3]
    assert all(not hasattr(row, "score") for generation in rankings for row in generation)


def test_stable_id_tie_break_is_invariant_to_cell_axis_order() -> None:
    summary = _summary()
    permutation = np.array([3, 2, 0, 1])
    reordered = freeze_cell_decision_summary_v1(
        validity=np.ascontiguousarray(summary.validity[:, permutation]),
        status=np.ascontiguousarray(summary.status[:, permutation]),
        score_basis=np.ascontiguousarray(summary.score_basis[:, permutation, :]),
        stable_cell_ids=np.ascontiguousarray(summary.stable_cell_ids[permutation]),
        score_policy_id=POLICY_ID,
        score_policy_digest=POLICY_DIGEST,
    )

    original_ids = [
        [row.stable_cell_id for row in generation]
        for generation in rank_cell_decision_summary_cpu_oracle_v1(summary)
    ]
    reordered_ids = [
        [row.stable_cell_id for row in generation]
        for generation in rank_cell_decision_summary_cpu_oracle_v1(reordered)
    ]
    assert reordered_ids == original_ids


def test_failed_and_unexecuted_rows_are_excluded_and_canonical_zero() -> None:
    summary = _summary()
    first_generation = rank_cell_decision_summary_cpu_oracle_v1(summary)[0]
    assert {row.stable_cell_id for row in first_generation} == {10, 40}

    invalid_basis = np.array([[[1.0], [-0.0]]], dtype=np.float64)
    with pytest.raises(CellDecisionSummaryContractError, match="positive zero"):
        freeze_cell_decision_summary_v1(
            validity=np.array([[1, 0]], dtype=np.uint8),
            status=np.array([[0, CELL_STATUS_UNEXECUTED_V1]], dtype=np.int32),
            score_basis=invalid_basis,
            stable_cell_ids=np.array([0, 1], dtype=np.int64),
            score_policy_id=POLICY_ID,
            score_policy_digest=POLICY_DIGEST,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("validity", np.ones((1, 1), dtype=np.int32), "dtype uint8"),
        ("status", np.zeros((1, 1), dtype=np.int64), "dtype int32"),
        ("score_basis", np.ones((1, 1, 1), dtype=np.float32), "dtype float64"),
        ("stable_cell_ids", np.array([0], dtype=np.uint64), "dtype int64"),
    ],
)
def test_exact_native_dtypes_are_required(
    field: str, value: np.ndarray, message: str
) -> None:
    kwargs = {
        "validity": np.ones((1, 1), dtype=np.uint8),
        "status": np.zeros((1, 1), dtype=np.int32),
        "score_basis": np.ones((1, 1, 1), dtype=np.float64),
        "stable_cell_ids": np.array([0], dtype=np.int64),
        "score_policy_id": POLICY_ID,
        "score_policy_digest": POLICY_DIGEST,
    }
    kwargs[field] = value
    with pytest.raises(CellDecisionSummaryContractError, match=message):
        freeze_cell_decision_summary_v1(**kwargs)


def test_malformed_identity_status_ids_and_nonfinite_basis_fail_closed() -> None:
    base = {
        "validity": np.ones((1, 2), dtype=np.uint8),
        "status": np.zeros((1, 2), dtype=np.int32),
        "score_basis": np.ones((1, 2, 1), dtype=np.float64),
        "stable_cell_ids": np.array([0, 1], dtype=np.int64),
        "score_policy_id": POLICY_ID,
        "score_policy_digest": POLICY_DIGEST,
    }
    cases = (
        ({"score_policy_id": ""}, "nonempty"),
        ({"score_policy_id": "x" * 257}, "at most 256"),
        ({"score_policy_digest": bytes(32)}, "all-zero"),
        ({"stable_cell_ids": np.array([1, 1], dtype=np.int64)}, "unique"),
        (
            {
                "validity": np.array([[0, 1]], dtype=np.uint8),
                "status": np.zeros((1, 2), dtype=np.int32),
            },
            "validity must be 1 exactly",
        ),
        (
            {"score_basis": np.array([[[np.nan], [1.0]]], dtype=np.float64)},
            "must be finite",
        ),
    )
    for changes, message in cases:
        with pytest.raises(CellDecisionSummaryContractError, match=message):
            freeze_cell_decision_summary_v1(**(base | changes))


def test_policy_identity_mismatch_fails_before_ranking() -> None:
    summary = _summary()
    with pytest.raises(CellDecisionSummaryContractError, match="policy_id mismatch"):
        rank_cell_decision_summary_cpu_oracle_v1(
            summary,
            expected_policy_id="different-policy",
            expected_policy_digest=POLICY_DIGEST,
        )
    with pytest.raises(CellDecisionSummaryContractError, match="digest mismatch"):
        rank_cell_decision_summary_cpu_oracle_v1(
            summary,
            expected_policy_id=POLICY_ID,
            expected_policy_digest=b"x" * 32,
        )


def test_contract_is_summary_only_and_defines_no_scientific_score() -> None:
    summary = _summary()
    assert CONTAINS_FULL_SIMULATION_STATE_V1 is False
    assert FULL_STATE_TRANSFER_CLAIMED_V1 is False
    assert COMPOSITE_SCIENTIFIC_SCORE_DEFINED_V1 is False
    assert summary.contains_full_simulation_state is False
    assert summary.full_state_transfer_claimed is False
    assert summary.composite_scientific_score_defined is False

    with pytest.raises(CellDecisionSummaryContractError, match="summary-only"):
        validate_cell_decision_summary_v1(
            replace(summary, full_state_transfer_claimed=True)
        )


def test_header_is_device_neutral_pod_with_exact_offsets_and_cpu_helpers(
    tmp_path: Path,
) -> None:
    source = HEADER.read_text(encoding="utf-8")
    for forbidden in (
        "#include <torch/",
        "#include <ATen/",
        "cudaMalloc",
        "cudaMemcpy",
        "<<<",
        "__global__",
    ):
        assert forbidden not in source

    compiler = shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        pytest.skip("a host C++ compiler is unavailable")
    probe = tmp_path / "cell_decision_summary_v1_probe.cpp"
    probe.write_text(
        """
#include "cuda_architecture_b_cell_decision_summary_v1.cuh"

namespace cds =
    eiel::cuda_architecture_b::cell_decision_summary_v1;

int main() {
    static_assert(cds::kContractVersionV1 == 1U);
    static_assert(cds::kPolicyDigestBytesV1 == 32);
    static_assert(cds::valid_dimensions_v1(2, 4, 3));
    static_assert(cds::validity_offset_v1(1, 2, 4) == 6);
    static_assert(cds::status_offset_v1(1, 2, 4) == 6);
    static_assert(cds::score_basis_offset_v1(1, 2, 1, 4, 3) == 19);
    static_assert(cds::stable_cell_id_offset_v1(2) == 2);
    static_assert(cds::cell_is_rankable_v1(1U, 0));
    static_assert(!cds::cell_is_rankable_v1(0U, 9));
    static_assert(cds::cell_status_is_failed_v1(9));
    static_assert(cds::cell_status_is_unexecuted_v1(
        cds::kCellStatusUnexecutedV1
    ));
    static_assert(cds::logical_plane_bytes_v1(2, 4, 3)
        == 2U * 4U * (1U + 4U + 3U * 8U) + 4U * 8U);
    static_assert(cds::generation_cell_offset_v1(-1, 0, 4) == -1);
    static_assert(cds::score_basis_offset_v1(0, 0, 3, 4, 3) == -1);
    static_assert(cds::logical_plane_bytes_v1(
        std::numeric_limits<std::size_t>::max(), 2, 1
    ) == 0U);
    static_assert(!cds::kContainsFullSimulationStateV1);
    static_assert(!cds::kFullStateTransferClaimedV1);
    static_assert(!cds::kCompositeScientificScoreDefinedV1);
    const std::uint8_t policy_id[] = {'p', '1'};
    std::uint8_t policy_digest[32] = {1};
    std::uint8_t validity[8] = {1, 1, 1, 1, 1, 1, 1, 1};
    std::int32_t status[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    double score_basis[24] = {};
    std::int64_t stable_cell_ids[4] = {10, 11, 12, 13};
    cds::CellDecisionSummaryViewV1 view{
        1U,
        0U,
        2,
        4,
        3,
        {policy_id, 2, policy_digest, 32},
        validity,
        status,
        score_basis,
        stable_cell_ids,
    };
    if (!cds::valid_view_metadata_v1(view)
        || !cds::valid_view_contents_serial_v1(view)) {
        return 1;
    }
    view.reserved_zero = 1U;
    if (cds::valid_view_metadata_v1(view)) {
        return 2;
    }
    return 0;
}
""",
        encoding="utf-8",
    )
    completed = subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-I",
            str(ROOT / "engine/perf"),
            str(probe),
            "-o",
            str(tmp_path / "probe"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    executed = subprocess.run(
        [str(tmp_path / "probe")],
        check=False,
        capture_output=True,
        text=True,
    )
    assert executed.returncode == 0, executed.stderr


def test_layout_rejects_native_index_overflow() -> None:
    with pytest.raises(
        CellDecisionSummaryContractError,
        match="native int64 index space",
    ):
        cell_decision_summary_layout_v1(1 << 62, 4, 3)
