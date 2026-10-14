from __future__ import annotations

"""CPU oracle for the compact native cell-decision summary v1 contract.

The caller defines the ordered, higher-first score basis and identifies that
policy with an ID and SHA-256 digest. This module validates and ranks that basis
without deriving, weighting, normalizing, or interpreting a scientific score.
"""

from dataclasses import dataclass
from typing import Final

import numpy as np


CONTRACT_SCHEMA_VERSION_V1: Final[str] = (
    "eiel_cuda_architecture_b_cell_decision_summary_v1"
)
CONTRACT_VERSION_V1: Final[int] = 1
POLICY_DIGEST_BYTES_V1: Final[int] = 32
MAX_INDEX_V1: Final[int] = (1 << 63) - 1
CELL_STATUS_SUCCESS_V1: Final[int] = 0
CELL_STATUS_UNEXECUTED_V1: Final[int] = -(1 << 31)
CELL_VALIDITY_EXCLUDED_V1: Final[int] = 0
CELL_VALIDITY_RANKABLE_V1: Final[int] = 1
CONTAINS_FULL_SIMULATION_STATE_V1: Final[bool] = False
FULL_STATE_TRANSFER_CLAIMED_V1: Final[bool] = False
COMPOSITE_SCIENTIFIC_SCORE_DEFINED_V1: Final[bool] = False


class CellDecisionSummaryContractError(ValueError):
    """The supplied summary violates the frozen v1 contract."""


@dataclass(frozen=True)
class CellDecisionSummaryLayoutV1:
    generation_count: int
    cell_count: int
    score_basis_count: int
    validity_shape: tuple[int, int]
    validity_strides_bytes: tuple[int, int]
    status_shape: tuple[int, int]
    status_strides_bytes: tuple[int, int]
    score_basis_shape: tuple[int, int, int]
    score_basis_strides_bytes: tuple[int, int, int]
    stable_cell_ids_shape: tuple[int]
    stable_cell_ids_strides_bytes: tuple[int]
    logical_plane_bytes: int


@dataclass(frozen=True)
class CellDecisionSummaryV1:
    contract_version: int
    score_policy_id: str
    score_policy_digest: bytes
    validity: np.ndarray
    status: np.ndarray
    score_basis: np.ndarray
    stable_cell_ids: np.ndarray
    contains_full_simulation_state: bool = CONTAINS_FULL_SIMULATION_STATE_V1
    full_state_transfer_claimed: bool = FULL_STATE_TRANSFER_CLAIMED_V1
    composite_scientific_score_defined: bool = (
        COMPOSITE_SCIENTIFIC_SCORE_DEFINED_V1
    )

    @property
    def generation_count(self) -> int:
        return int(self.validity.shape[0])

    @property
    def cell_count(self) -> int:
        return int(self.validity.shape[1])

    @property
    def score_basis_count(self) -> int:
        return int(self.score_basis.shape[2])


@dataclass(frozen=True)
class RankedCellDecisionV1:
    generation_index: int
    rank_index: int
    cell_index: int
    stable_cell_id: int
    # This is the caller's basis verbatim, not a composite score.
    score_basis: tuple[float, ...]


def cell_decision_summary_layout_v1(
    generation_count: int,
    cell_count: int,
    score_basis_count: int,
) -> CellDecisionSummaryLayoutV1:
    for name, value in (
        ("generation_count", generation_count),
        ("cell_count", cell_count),
        ("score_basis_count", score_basis_count),
    ):
        if type(value) is not int or value <= 0:
            raise CellDecisionSummaryContractError(
                f"{name} must be a positive exact integer"
            )

    row_count = generation_count * cell_count
    if row_count > MAX_INDEX_V1 or row_count > MAX_INDEX_V1 // score_basis_count:
        raise CellDecisionSummaryContractError(
            "summary dimensions exceed the native int64 index space"
        )
    row_bytes = 1 + 4 + score_basis_count * 8
    logical_plane_bytes = row_count * row_bytes + cell_count * 8
    if logical_plane_bytes > MAX_INDEX_V1:
        raise CellDecisionSummaryContractError(
            "summary planes exceed the native size bound"
        )
    return CellDecisionSummaryLayoutV1(
        generation_count=generation_count,
        cell_count=cell_count,
        score_basis_count=score_basis_count,
        validity_shape=(generation_count, cell_count),
        validity_strides_bytes=(cell_count, 1),
        status_shape=(generation_count, cell_count),
        status_strides_bytes=(cell_count * 4, 4),
        score_basis_shape=(generation_count, cell_count, score_basis_count),
        score_basis_strides_bytes=(
            cell_count * score_basis_count * 8,
            score_basis_count * 8,
            8,
        ),
        stable_cell_ids_shape=(cell_count,),
        stable_cell_ids_strides_bytes=(8,),
        logical_plane_bytes=logical_plane_bytes,
    )


def _require_array(
    value: object,
    *,
    name: str,
    dtype: np.dtype,
    ndim: int,
) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise CellDecisionSummaryContractError(f"{name} must be a numpy array")
    if value.dtype != dtype:
        raise CellDecisionSummaryContractError(
            f"{name} must have exact dtype {dtype.name}"
        )
    if value.ndim != ndim:
        raise CellDecisionSummaryContractError(f"{name} must have rank {ndim}")
    if not value.flags.c_contiguous:
        raise CellDecisionSummaryContractError(f"{name} must be C-contiguous")
    return value


def _normalize_policy_id(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise CellDecisionSummaryContractError(
            "score_policy_id must be a nonempty string"
        )
    if "\x00" in value:
        raise CellDecisionSummaryContractError(
            "score_policy_id must not contain NUL bytes"
        )
    try:
        encoded = value.encode("ascii")
    except UnicodeEncodeError as exc:
        raise CellDecisionSummaryContractError(
            "score_policy_id must be ASCII"
        ) from exc
    if len(encoded) > 256:
        raise CellDecisionSummaryContractError(
            "score_policy_id must contain at most 256 ASCII bytes"
        )
    return value


def _normalize_policy_digest(value: object) -> bytes:
    if isinstance(value, str):
        if len(value) != POLICY_DIGEST_BYTES_V1 * 2:
            raise CellDecisionSummaryContractError(
                "score_policy_digest hex must contain exactly 64 characters"
            )
        try:
            digest = bytes.fromhex(value)
        except ValueError as exc:
            raise CellDecisionSummaryContractError(
                "score_policy_digest must be valid hexadecimal"
            ) from exc
    elif isinstance(value, bytes):
        digest = value
    else:
        raise CellDecisionSummaryContractError(
            "score_policy_digest must be 32 bytes or 64 hexadecimal characters"
        )
    if len(digest) != POLICY_DIGEST_BYTES_V1:
        raise CellDecisionSummaryContractError(
            "score_policy_digest must contain exactly 32 bytes"
        )
    if not any(digest):
        raise CellDecisionSummaryContractError(
            "score_policy_digest must not be the all-zero absent marker"
        )
    return digest


def _validate_planes(
    *,
    validity: object,
    status: object,
    score_basis: object,
    stable_cell_ids: object,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    validity_array = _require_array(
        validity, name="validity", dtype=np.dtype(np.uint8), ndim=2
    )
    status_array = _require_array(
        status, name="status", dtype=np.dtype(np.int32), ndim=2
    )
    score_basis_array = _require_array(
        score_basis, name="score_basis", dtype=np.dtype(np.float64), ndim=3
    )
    cell_ids_array = _require_array(
        stable_cell_ids,
        name="stable_cell_ids",
        dtype=np.dtype(np.int64),
        ndim=1,
    )

    generation_count, cell_count = validity_array.shape
    if generation_count <= 0 or cell_count <= 0:
        raise CellDecisionSummaryContractError(
            "validity must have positive [G,C] dimensions"
        )
    if status_array.shape != validity_array.shape:
        raise CellDecisionSummaryContractError(
            "status shape must exactly match validity [G,C]"
        )
    if (
        score_basis_array.shape[:2] != validity_array.shape
        or score_basis_array.shape[2] <= 0
    ):
        raise CellDecisionSummaryContractError(
            "score_basis shape must be [G,C,K] with positive K"
        )
    if cell_ids_array.shape != (cell_count,):
        raise CellDecisionSummaryContractError(
            "stable_cell_ids shape must be [C]"
        )
    if np.any(cell_ids_array < 0):
        raise CellDecisionSummaryContractError(
            "stable_cell_ids must be nonnegative"
        )
    if np.unique(cell_ids_array).size != cell_count:
        raise CellDecisionSummaryContractError(
            "stable_cell_ids must be unique"
        )

    known_validity = (validity_array == CELL_VALIDITY_EXCLUDED_V1) | (
        validity_array == CELL_VALIDITY_RANKABLE_V1
    )
    if not bool(np.all(known_validity)):
        raise CellDecisionSummaryContractError("validity values must be 0 or 1")

    successful = status_array == CELL_STATUS_SUCCESS_V1
    rankable = validity_array == CELL_VALIDITY_RANKABLE_V1
    if not bool(np.array_equal(successful, rankable)):
        raise CellDecisionSummaryContractError(
            "validity must be 1 exactly for status 0; failed and unexecuted "
            "rows must have validity 0"
        )

    if not bool(np.all(np.isfinite(score_basis_array[rankable]))):
        raise CellDecisionSummaryContractError(
            "rankable score_basis rows must be finite"
        )
    excluded_bits = score_basis_array[~rankable].view(np.uint64)
    if not bool(np.all(excluded_bits == 0)):
        raise CellDecisionSummaryContractError(
            "failed and unexecuted score_basis rows must be canonical positive zero"
        )
    return validity_array, status_array, score_basis_array, cell_ids_array


def freeze_cell_decision_summary_v1(
    *,
    validity: object,
    status: object,
    score_basis: object,
    stable_cell_ids: object,
    score_policy_id: object,
    score_policy_digest: object,
) -> CellDecisionSummaryV1:
    """Validate caller planes and return immutable, C-contiguous snapshots."""

    policy_id = _normalize_policy_id(score_policy_id)
    policy_digest = _normalize_policy_digest(score_policy_digest)
    planes = _validate_planes(
        validity=validity,
        status=status,
        score_basis=score_basis,
        stable_cell_ids=stable_cell_ids,
    )
    frozen_planes: list[np.ndarray] = []
    for plane in planes:
        frozen = np.array(plane, dtype=plane.dtype, order="C", copy=True)
        frozen.flags.writeable = False
        frozen_planes.append(frozen)
    return CellDecisionSummaryV1(
        contract_version=CONTRACT_VERSION_V1,
        score_policy_id=policy_id,
        score_policy_digest=policy_digest,
        validity=frozen_planes[0],
        status=frozen_planes[1],
        score_basis=frozen_planes[2],
        stable_cell_ids=frozen_planes[3],
    )


def validate_cell_decision_summary_v1(
    summary: CellDecisionSummaryV1,
    *,
    expected_policy_id: object | None = None,
    expected_policy_digest: object | None = None,
) -> None:
    if not isinstance(summary, CellDecisionSummaryV1):
        raise CellDecisionSummaryContractError(
            "summary must be CellDecisionSummaryV1"
        )
    if summary.contract_version != CONTRACT_VERSION_V1:
        raise CellDecisionSummaryContractError("unsupported contract_version")
    policy_id = _normalize_policy_id(summary.score_policy_id)
    policy_digest = _normalize_policy_digest(summary.score_policy_digest)
    if expected_policy_id is not None:
        expected_id = _normalize_policy_id(expected_policy_id)
        if policy_id != expected_id:
            raise CellDecisionSummaryContractError("score_policy_id mismatch")
    if expected_policy_digest is not None:
        expected_digest = _normalize_policy_digest(expected_policy_digest)
        if policy_digest != expected_digest:
            raise CellDecisionSummaryContractError("score_policy_digest mismatch")
    if (
        summary.contains_full_simulation_state
        or summary.full_state_transfer_claimed
        or summary.composite_scientific_score_defined
    ):
        raise CellDecisionSummaryContractError(
            "v1 is summary-only and defines no composite scientific score"
        )
    planes = _validate_planes(
        validity=summary.validity,
        status=summary.status,
        score_basis=summary.score_basis,
        stable_cell_ids=summary.stable_cell_ids,
    )
    if any(plane.flags.writeable for plane in planes):
        raise CellDecisionSummaryContractError(
            "summary planes must be frozen read-only snapshots"
        )


def rank_cell_decision_summary_cpu_oracle_v1(
    summary: CellDecisionSummaryV1,
    *,
    expected_policy_id: object | None = None,
    expected_policy_digest: object | None = None,
) -> tuple[tuple[RankedCellDecisionV1, ...], ...]:
    """Rank each generation by caller basis, then stable cell ID ascending."""

    validate_cell_decision_summary_v1(
        summary,
        expected_policy_id=expected_policy_id,
        expected_policy_digest=expected_policy_digest,
    )
    rankings: list[tuple[RankedCellDecisionV1, ...]] = []
    for generation in range(summary.generation_count):
        eligible = [
            cell
            for cell in range(summary.cell_count)
            if summary.validity[generation, cell] == CELL_VALIDITY_RANKABLE_V1
            and summary.status[generation, cell] == CELL_STATUS_SUCCESS_V1
        ]
        eligible.sort(
            key=lambda cell: (
                *(
                    -float(value)
                    for value in summary.score_basis[generation, cell]
                ),
                int(summary.stable_cell_ids[cell]),
            )
        )
        rankings.append(
            tuple(
                RankedCellDecisionV1(
                    generation_index=generation,
                    rank_index=rank_index,
                    cell_index=cell,
                    stable_cell_id=int(summary.stable_cell_ids[cell]),
                    score_basis=tuple(
                        float(value)
                        for value in summary.score_basis[generation, cell]
                    ),
                )
                for rank_index, cell in enumerate(eligible)
            )
        )
    return tuple(rankings)
