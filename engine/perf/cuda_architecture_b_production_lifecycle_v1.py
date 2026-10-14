from __future__ import annotations

"""Deterministic CUDA production-lifecycle settlement contract, version 1.

This module is metadata-only. It does not poll ticks, launch CUDA, execute a
simulation, or establish trajectory ownership. Cancellation is accepted only
as a completed-generation boundary. The first accepted request wins.

If a device failure is observed at or before that boundary, failure wins. An
exact tie therefore settles as ``failed``. A failure record after the accepted
cancellation boundary is contradictory evidence of execution beyond the
cancellation frontier and is rejected.

Live CUDA owner integration remains pending. A validated receipt from this
module does not promote any production capability.
"""

from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Any, Final


CONTRACT_SCHEMA_VERSION_V1: Final[str] = (
    "eiel_cuda_architecture_b_production_lifecycle_v1"
)
CANCELLATION_RECEIPT_SCHEMA_VERSION_V1: Final[str] = (
    "eiel_cuda_architecture_b_production_cancellation_receipt_v1"
)
TERMINAL_RECEIPT_SCHEMA_VERSION_V1: Final[str] = (
    "eiel_cuda_architecture_b_production_terminal_receipt_v1"
)

TERMINAL_COMPLETED_V1: Final[str] = "completed"
TERMINAL_CANCELLED_V1: Final[str] = "cancelled"
TERMINAL_FAILED_V1: Final[str] = "failed"

CANCEL_ACCEPTED_V1: Final[str] = "accepted"
CANCEL_ALREADY_REQUESTED_V1: Final[str] = "already_requested"

COMPACT_ROW_SUCCESS_STATUS_V1: Final[int] = 0
COMPACT_ROW_UNPUBLISHED_STATUS_V1: Final[int] = -1

LIVE_CUDA_OWNER_INTEGRATION_STATUS_V1: Final[str] = "pending"
PRODUCTION_CAPABILITY_PROMOTED_V1: Final[bool] = False

_SHA256_HEX_LENGTH: Final[int] = 64
_MAX_COUNT: Final[int] = (1 << 63) - 1


class ProductionLifecycleContractError(ValueError):
    """Raised when lifecycle evidence is malformed or contradictory."""


class ProductionLifecycleMutationError(RuntimeError):
    """Raised when a caller tries to mutate a terminal lifecycle."""


def _fail(path: str, message: str) -> None:
    raise ProductionLifecycleContractError(f"{path}: {message}")


def _stable_ascii_identifier(value: Any, *, path: str) -> str:
    if (
        type(value) is not str
        or not value
        or len(value) > 256
        or value.strip() != value
        or not value.isascii()
        or not value.isprintable()
        or "\x00" in value
    ):
        _fail(path, "must be a nonempty stable ASCII identifier")
    return value


def _exact_int(
    value: Any,
    *,
    path: str,
    minimum: int = 0,
    maximum: int = _MAX_COUNT,
) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        _fail(path, f"must be an exact integer in [{minimum}, {maximum}]")
    return value


def _exact_bool(value: Any, *, path: str) -> bool:
    if type(value) is not bool:
        _fail(path, "must be an exact boolean")
    return value


def _sha256_hex(value: Any, *, path: str) -> str:
    if (
        type(value) is not str
        or len(value) != _SHA256_HEX_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        _fail(path, "must be exactly 64 lowercase hexadecimal characters")
    return value


def _canonical_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class CheckpointPolicyV1:
    """Explicit checkpoint requirements for abnormal terminal outcomes."""

    checkpoint_on_cancel_required: bool
    checkpoint_on_failure_required: bool


@dataclass(frozen=True, slots=True)
class CancellationRequestV1:
    request_id: str
    controller_id: str
    run_id: str
    requested_boundary: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "controller_id": self.controller_id,
            "run_id": self.run_id,
            "requested_boundary": self.requested_boundary,
        }


@dataclass(frozen=True, slots=True)
class CancellationRequestReceiptV1:
    schema_version: str
    request_id: str
    controller_id: str
    run_id: str
    disposition: str
    first_request_id: str
    requested_boundary: int
    state_revision: int
    generation_boundary_only: bool = True
    tick_polling: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "request_id": self.request_id,
            "controller_id": self.controller_id,
            "run_id": self.run_id,
            "disposition": self.disposition,
            "first_request_id": self.first_request_id,
            "requested_boundary": self.requested_boundary,
            "state_revision": self.state_revision,
            "generation_boundary_only": self.generation_boundary_only,
            "tick_polling": self.tick_polling,
        }


@dataclass(frozen=True, slots=True)
class FirstDeviceFailureRecordV1:
    """The authoritative first device failure at a generation boundary.

    ``generation_boundary`` is the count of generations completed by every
    cell before the failing generation began. ``tick_index`` identifies the
    device failure but is never used as a cancellation polling coordinate.
    """

    cell_index: int
    generation_boundary: int
    tick_index: int
    stage: str
    status: int
    failure_code: str
    sequence: int = 1
    authoritative: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "cell_index": self.cell_index,
            "generation_boundary": self.generation_boundary,
            "tick_index": self.tick_index,
            "stage": self.stage,
            "status": self.status,
            "failure_code": self.failure_code,
            "sequence": self.sequence,
            "authoritative": self.authoritative,
        }


@dataclass(frozen=True, slots=True)
class ContinuationDigestReceiptV1:
    """Pairwise authoritative/checkpoint digests required for continuation."""

    authoritative_state_sha256: str
    checkpoint_state_sha256: str
    authoritative_canonical_sha256: str
    checkpoint_canonical_sha256: str
    authoritative_rng_sha256: str
    checkpoint_rng_sha256: str
    authoritative_event_cursor_sha256: str
    checkpoint_event_cursor_sha256: str

    @property
    def exact(self) -> bool:
        return (
            self.authoritative_state_sha256 == self.checkpoint_state_sha256
            and self.authoritative_canonical_sha256 == self.checkpoint_canonical_sha256
            and self.authoritative_rng_sha256 == self.checkpoint_rng_sha256
            and self.authoritative_event_cursor_sha256
            == self.checkpoint_event_cursor_sha256
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "authoritative_state_sha256": self.authoritative_state_sha256,
            "checkpoint_state_sha256": self.checkpoint_state_sha256,
            "authoritative_canonical_sha256": (self.authoritative_canonical_sha256),
            "checkpoint_canonical_sha256": self.checkpoint_canonical_sha256,
            "authoritative_rng_sha256": self.authoritative_rng_sha256,
            "checkpoint_rng_sha256": self.checkpoint_rng_sha256,
            "authoritative_event_cursor_sha256": (
                self.authoritative_event_cursor_sha256
            ),
            "checkpoint_event_cursor_sha256": (self.checkpoint_event_cursor_sha256),
            "exact": self.exact,
        }


@dataclass(frozen=True, slots=True)
class ExecutionOwnershipEvidenceV1:
    """Caller-provided proof flags; every unsupported claim defaults false."""

    no_hidden_fallback: bool = False
    no_runtime_jit: bool = False
    no_aten_tick_or_generation_ownership: bool = False
    no_python_tick_ownership: bool = False
    no_python_generation_ownership: bool = False

    def to_dict(self) -> dict[str, bool]:
        return {
            "no_hidden_fallback": self.no_hidden_fallback,
            "no_runtime_jit": self.no_runtime_jit,
            "no_aten_tick_or_generation_ownership": (
                self.no_aten_tick_or_generation_ownership
            ),
            "no_python_tick_ownership": self.no_python_tick_ownership,
            "no_python_generation_ownership": (self.no_python_generation_ownership),
        }


@dataclass(frozen=True, slots=True)
class ProductionSettlementInputV1:
    controller_id: str
    run_id: str
    completed_generations_by_cell: tuple[int, ...]
    compact_row_validity_by_cell: tuple[tuple[bool, ...], ...]
    compact_row_status_by_cell: tuple[tuple[int, ...], ...]
    first_device_failure: FirstDeviceFailureRecordV1 | None
    checkpoint_requested: bool
    checkpoint_produced: bool
    continuation_digests: ContinuationDigestReceiptV1 | None = None
    ownership_evidence: ExecutionOwnershipEvidenceV1 = ExecutionOwnershipEvidenceV1()


@dataclass(frozen=True, slots=True)
class ProductionTerminalReceiptV1:
    schema_version: str
    controller_id: str
    run_id: str
    terminal_status: str
    terminal: bool
    state_revision: int
    settlement_sequence: int
    requested_generations: int
    completed_generations_by_cell: tuple[int, ...]
    cancellation_request: CancellationRequestV1 | None
    first_device_failure: FirstDeviceFailureRecordV1 | None
    compact_row_validity_by_cell: tuple[tuple[bool, ...], ...]
    compact_row_status_by_cell: tuple[tuple[int, ...], ...]
    checkpoint_required: bool
    checkpoint_requested: bool
    checkpoint_produced: bool
    continuation_digest_exact: bool
    continuation_allowed: bool
    ownership_claims: ExecutionOwnershipEvidenceV1
    live_cuda_owner_integration_status: str
    production_capability_promoted: bool

    def _payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "controller_id": self.controller_id,
            "run_id": self.run_id,
            "terminal_status": self.terminal_status,
            "terminal": self.terminal,
            "state_revision": self.state_revision,
            "settlement_sequence": self.settlement_sequence,
            "requested_generations": self.requested_generations,
            "completed_generations_by_cell": list(self.completed_generations_by_cell),
            "cancellation_request": (
                None
                if self.cancellation_request is None
                else self.cancellation_request.to_dict()
            ),
            "first_device_failure": (
                None
                if self.first_device_failure is None
                else self.first_device_failure.to_dict()
            ),
            "compact_row_validity_by_cell": [
                list(row) for row in self.compact_row_validity_by_cell
            ],
            "compact_row_status_by_cell": [
                list(row) for row in self.compact_row_status_by_cell
            ],
            "checkpoint_required": self.checkpoint_required,
            "checkpoint_requested": self.checkpoint_requested,
            "checkpoint_produced": self.checkpoint_produced,
            "continuation_digest_exact": self.continuation_digest_exact,
            "continuation_allowed": self.continuation_allowed,
            "ownership_claims": self.ownership_claims.to_dict(),
            "live_cuda_owner_integration_status": (
                self.live_cuda_owner_integration_status
            ),
            "production_capability_promoted": (self.production_capability_promoted),
        }

    @property
    def receipt_sha256(self) -> str:
        return _canonical_sha256(self._payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self._payload()
        payload["receipt_sha256"] = self.receipt_sha256
        return payload


def _validate_checkpoint_policy(policy: CheckpointPolicyV1) -> None:
    if type(policy) is not CheckpointPolicyV1:
        _fail("checkpoint_policy", "must be CheckpointPolicyV1")
    _exact_bool(
        policy.checkpoint_on_cancel_required,
        path="checkpoint_policy.checkpoint_on_cancel_required",
    )
    _exact_bool(
        policy.checkpoint_on_failure_required,
        path="checkpoint_policy.checkpoint_on_failure_required",
    )


def _validate_cancel_request(
    request: CancellationRequestV1,
    *,
    controller_id: str,
    run_id: str,
    requested_generations: int,
) -> CancellationRequestV1:
    if type(request) is not CancellationRequestV1:
        _fail("cancellation_request", "must be CancellationRequestV1")
    request_id = _stable_ascii_identifier(
        request.request_id, path="cancellation_request.request_id"
    )
    request_controller_id = _stable_ascii_identifier(
        request.controller_id, path="cancellation_request.controller_id"
    )
    request_run_id = _stable_ascii_identifier(
        request.run_id, path="cancellation_request.run_id"
    )
    if request_controller_id != controller_id:
        _fail("cancellation_request.controller_id", "identity mismatch")
    if request_run_id != run_id:
        _fail("cancellation_request.run_id", "identity mismatch")
    requested_boundary = _exact_int(
        request.requested_boundary,
        path="cancellation_request.requested_boundary",
        maximum=requested_generations,
    )
    return CancellationRequestV1(
        request_id=request_id,
        controller_id=request_controller_id,
        run_id=request_run_id,
        requested_boundary=requested_boundary,
    )


def _validate_failure_record(
    record: FirstDeviceFailureRecordV1,
    *,
    cell_count: int,
    requested_generations: int,
) -> FirstDeviceFailureRecordV1:
    if type(record) is not FirstDeviceFailureRecordV1:
        _fail(
            "settlement.first_device_failure",
            "must be FirstDeviceFailureRecordV1",
        )
    cell_index = _exact_int(
        record.cell_index,
        path="settlement.first_device_failure.cell_index",
        maximum=cell_count - 1,
    )
    boundary = _exact_int(
        record.generation_boundary,
        path="settlement.first_device_failure.generation_boundary",
        maximum=requested_generations - 1,
    )
    tick_index = _exact_int(
        record.tick_index,
        path="settlement.first_device_failure.tick_index",
    )
    stage = _stable_ascii_identifier(
        record.stage, path="settlement.first_device_failure.stage"
    )
    status = _exact_int(
        record.status,
        path="settlement.first_device_failure.status",
        minimum=1,
    )
    failure_code = _stable_ascii_identifier(
        record.failure_code,
        path="settlement.first_device_failure.failure_code",
    )
    if record.sequence != 1:
        _fail(
            "settlement.first_device_failure.sequence",
            "must be exactly 1 for the first failure",
        )
    if record.authoritative is not True:
        _fail(
            "settlement.first_device_failure.authoritative",
            "must be exactly true",
        )
    return FirstDeviceFailureRecordV1(
        cell_index=cell_index,
        generation_boundary=boundary,
        tick_index=tick_index,
        stage=stage,
        status=status,
        failure_code=failure_code,
    )


def _validate_continuation_digests(
    receipt: ContinuationDigestReceiptV1 | None,
) -> ContinuationDigestReceiptV1 | None:
    if receipt is None:
        return None
    if type(receipt) is not ContinuationDigestReceiptV1:
        _fail(
            "settlement.continuation_digests",
            "must be ContinuationDigestReceiptV1 or None",
        )
    values: dict[str, str] = {}
    for field in (
        "authoritative_state_sha256",
        "checkpoint_state_sha256",
        "authoritative_canonical_sha256",
        "checkpoint_canonical_sha256",
        "authoritative_rng_sha256",
        "checkpoint_rng_sha256",
        "authoritative_event_cursor_sha256",
        "checkpoint_event_cursor_sha256",
    ):
        values[field] = _sha256_hex(
            getattr(receipt, field),
            path=f"settlement.continuation_digests.{field}",
        )
    return ContinuationDigestReceiptV1(**values)


def _validate_ownership_evidence(
    evidence: ExecutionOwnershipEvidenceV1,
) -> ExecutionOwnershipEvidenceV1:
    if type(evidence) is not ExecutionOwnershipEvidenceV1:
        _fail(
            "settlement.ownership_evidence",
            "must be ExecutionOwnershipEvidenceV1",
        )
    values: dict[str, bool] = {}
    for field in (
        "no_hidden_fallback",
        "no_runtime_jit",
        "no_aten_tick_or_generation_ownership",
        "no_python_tick_ownership",
        "no_python_generation_ownership",
    ):
        values[field] = _exact_bool(
            getattr(evidence, field),
            path=f"settlement.ownership_evidence.{field}",
        )
    return ExecutionOwnershipEvidenceV1(**values)


class CudaArchitectureBProductionLifecycleV1:
    """One-run first-request-wins lifecycle settlement state machine."""

    __slots__ = (
        "_cell_count",
        "_checkpoint_policy",
        "_controller_id",
        "_first_cancellation_request",
        "_requested_generations",
        "_run_id",
        "_settlement_input",
        "_state_revision",
        "_terminal_receipt",
    )

    def __init__(
        self,
        *,
        controller_id: str,
        run_id: str,
        cell_count: int,
        requested_generations: int,
        checkpoint_policy: CheckpointPolicyV1,
    ) -> None:
        self._controller_id = _stable_ascii_identifier(
            controller_id, path="controller_id"
        )
        self._run_id = _stable_ascii_identifier(run_id, path="run_id")
        self._cell_count = _exact_int(cell_count, path="cell_count", minimum=1)
        self._requested_generations = _exact_int(
            requested_generations,
            path="requested_generations",
            minimum=1,
        )
        _validate_checkpoint_policy(checkpoint_policy)
        self._checkpoint_policy = checkpoint_policy
        self._first_cancellation_request: CancellationRequestV1 | None = None
        self._settlement_input: ProductionSettlementInputV1 | None = None
        self._terminal_receipt: ProductionTerminalReceiptV1 | None = None
        self._state_revision = 0

    @property
    def state_revision(self) -> int:
        return self._state_revision

    @property
    def terminal_receipt(self) -> ProductionTerminalReceiptV1 | None:
        return self._terminal_receipt

    def request_cancellation(
        self, request: CancellationRequestV1
    ) -> CancellationRequestReceiptV1:
        if self._terminal_receipt is not None:
            raise ProductionLifecycleMutationError(
                "terminal lifecycle rejects cancellation mutation"
            )
        validated = _validate_cancel_request(
            request,
            controller_id=self._controller_id,
            run_id=self._run_id,
            requested_generations=self._requested_generations,
        )
        if self._first_cancellation_request is None:
            self._first_cancellation_request = validated
            self._state_revision += 1
            disposition = CANCEL_ACCEPTED_V1
        else:
            disposition = CANCEL_ALREADY_REQUESTED_V1

        first = self._first_cancellation_request
        assert first is not None
        return CancellationRequestReceiptV1(
            schema_version=CANCELLATION_RECEIPT_SCHEMA_VERSION_V1,
            request_id=validated.request_id,
            controller_id=self._controller_id,
            run_id=self._run_id,
            disposition=disposition,
            first_request_id=first.request_id,
            requested_boundary=first.requested_boundary,
            state_revision=self._state_revision,
        )

    def cancellation_due_at_boundary(
        self, completed_generation_boundary: int
    ) -> CancellationRequestV1 | None:
        boundary = _exact_int(
            completed_generation_boundary,
            path="completed_generation_boundary",
            maximum=self._requested_generations,
        )
        cancellation = self._first_cancellation_request
        if cancellation is None or boundary < cancellation.requested_boundary:
            return None
        if boundary == cancellation.requested_boundary:
            return cancellation
        _fail(
            "completed_generation_boundary",
            "cannot be after the first accepted cancellation boundary",
        )

    def settle(
        self, settlement: ProductionSettlementInputV1
    ) -> ProductionTerminalReceiptV1:
        if self._terminal_receipt is not None:
            if settlement == self._settlement_input:
                return self._terminal_receipt
            raise ProductionLifecycleMutationError(
                "terminal lifecycle rejects non-identical settlement mutation"
            )
        validated = self._validate_settlement(settlement)
        receipt = self._build_terminal_receipt(validated)
        self._settlement_input = validated
        self._terminal_receipt = receipt
        self._state_revision = receipt.state_revision
        return receipt

    def _validate_settlement(
        self, settlement: ProductionSettlementInputV1
    ) -> ProductionSettlementInputV1:
        if type(settlement) is not ProductionSettlementInputV1:
            _fail("settlement", "must be ProductionSettlementInputV1")
        controller_id = _stable_ascii_identifier(
            settlement.controller_id, path="settlement.controller_id"
        )
        run_id = _stable_ascii_identifier(settlement.run_id, path="settlement.run_id")
        if controller_id != self._controller_id:
            _fail("settlement.controller_id", "identity mismatch")
        if run_id != self._run_id:
            _fail("settlement.run_id", "identity mismatch")

        completed = self._validate_completed_counts(
            settlement.completed_generations_by_cell
        )
        failure = (
            None
            if settlement.first_device_failure is None
            else _validate_failure_record(
                settlement.first_device_failure,
                cell_count=self._cell_count,
                requested_generations=self._requested_generations,
            )
        )
        validity, statuses = self._validate_compact_rows(
            settlement.compact_row_validity_by_cell,
            settlement.compact_row_status_by_cell,
            completed=completed,
            failure=failure,
        )
        checkpoint_requested = _exact_bool(
            settlement.checkpoint_requested,
            path="settlement.checkpoint_requested",
        )
        checkpoint_produced = _exact_bool(
            settlement.checkpoint_produced,
            path="settlement.checkpoint_produced",
        )
        digests = _validate_continuation_digests(settlement.continuation_digests)
        evidence = _validate_ownership_evidence(settlement.ownership_evidence)

        return ProductionSettlementInputV1(
            controller_id=controller_id,
            run_id=run_id,
            completed_generations_by_cell=completed,
            compact_row_validity_by_cell=validity,
            compact_row_status_by_cell=statuses,
            first_device_failure=failure,
            checkpoint_requested=checkpoint_requested,
            checkpoint_produced=checkpoint_produced,
            continuation_digests=digests,
            ownership_evidence=evidence,
        )

    def _validate_completed_counts(self, value: Any) -> tuple[int, ...]:
        if type(value) is not tuple or len(value) != self._cell_count:
            _fail(
                "settlement.completed_generations_by_cell",
                f"must be an exact tuple of {self._cell_count} cells",
            )
        return tuple(
            _exact_int(
                count,
                path=f"settlement.completed_generations_by_cell[{index}]",
                maximum=self._requested_generations,
            )
            for index, count in enumerate(value)
        )

    def _validate_compact_rows(
        self,
        validity_value: Any,
        status_value: Any,
        *,
        completed: tuple[int, ...],
        failure: FirstDeviceFailureRecordV1 | None,
    ) -> tuple[tuple[tuple[bool, ...], ...], tuple[tuple[int, ...], ...]]:
        validity = self._validate_matrix_shape(
            validity_value, path="settlement.compact_row_validity_by_cell"
        )
        statuses = self._validate_matrix_shape(
            status_value, path="settlement.compact_row_status_by_cell"
        )
        normalized_validity: list[tuple[bool, ...]] = []
        normalized_statuses: list[tuple[int, ...]] = []

        for cell_index, (validity_row, status_row, completed_count) in enumerate(
            zip(validity, statuses, completed, strict=True)
        ):
            cell_validity: list[bool] = []
            cell_statuses: list[int] = []
            for generation, (valid, status) in enumerate(
                zip(validity_row, status_row, strict=True)
            ):
                valid = _exact_bool(
                    valid,
                    path=(
                        "settlement.compact_row_validity_by_cell"
                        f"[{cell_index}][{generation}]"
                    ),
                )
                status = _exact_int(
                    status,
                    path=(
                        "settlement.compact_row_status_by_cell"
                        f"[{cell_index}][{generation}]"
                    ),
                    minimum=COMPACT_ROW_UNPUBLISHED_STATUS_V1,
                )
                expected_valid = generation < completed_count
                if valid is not expected_valid:
                    _fail(
                        "settlement.compact_row_validity_by_cell"
                        f"[{cell_index}][{generation}]",
                        "invalid tail or non-prefix validity contradicts "
                        "completed-generation count",
                    )
                failure_row = (
                    failure is not None
                    and failure.cell_index == cell_index
                    and failure.generation_boundary == generation
                )
                expected_status = (
                    COMPACT_ROW_SUCCESS_STATUS_V1
                    if valid
                    else (
                        failure.status
                        if failure_row
                        else COMPACT_ROW_UNPUBLISHED_STATUS_V1
                    )
                )
                if status != expected_status:
                    _fail(
                        "settlement.compact_row_status_by_cell"
                        f"[{cell_index}][{generation}]",
                        f"must be exactly {expected_status}",
                    )
                cell_validity.append(valid)
                cell_statuses.append(status)
            normalized_validity.append(tuple(cell_validity))
            normalized_statuses.append(tuple(cell_statuses))
        return tuple(normalized_validity), tuple(normalized_statuses)

    def _validate_matrix_shape(
        self, value: Any, *, path: str
    ) -> tuple[tuple[Any, ...], ...]:
        if type(value) is not tuple or len(value) != self._cell_count:
            _fail(path, f"must be an exact tuple of {self._cell_count} cells")
        rows: list[tuple[Any, ...]] = []
        for index, row in enumerate(value):
            if type(row) is not tuple or len(row) != self._requested_generations:
                _fail(
                    f"{path}[{index}]",
                    "must be an exact tuple with one row per requested "
                    f"generation ({self._requested_generations})",
                )
            rows.append(row)
        return tuple(rows)

    def _build_terminal_receipt(
        self, settlement: ProductionSettlementInputV1
    ) -> ProductionTerminalReceiptV1:
        cancellation = self._first_cancellation_request
        failure = settlement.first_device_failure
        if cancellation is not None and failure is not None:
            if failure.generation_boundary > cancellation.requested_boundary:
                _fail(
                    "settlement.first_device_failure.generation_boundary",
                    "cannot be after the first accepted cancellation boundary",
                )
            terminal_status = TERMINAL_FAILED_V1
        elif failure is not None:
            terminal_status = TERMINAL_FAILED_V1
        elif cancellation is not None:
            terminal_status = TERMINAL_CANCELLED_V1
        else:
            terminal_status = TERMINAL_COMPLETED_V1

        self._validate_terminal_progress(
            terminal_status,
            settlement.completed_generations_by_cell,
            cancellation=cancellation,
            failure=failure,
        )
        checkpoint_required = (
            terminal_status == TERMINAL_CANCELLED_V1
            and self._checkpoint_policy.checkpoint_on_cancel_required
        ) or (
            terminal_status == TERMINAL_FAILED_V1
            and self._checkpoint_policy.checkpoint_on_failure_required
        )
        self._validate_checkpoint_settlement(
            checkpoint_required=checkpoint_required,
            checkpoint_requested=settlement.checkpoint_requested,
            checkpoint_produced=settlement.checkpoint_produced,
            continuation_digests=settlement.continuation_digests,
        )
        continuation_exact = (
            settlement.continuation_digests is not None
            and settlement.continuation_digests.exact
        )
        continuation_allowed = (
            settlement.checkpoint_requested
            and settlement.checkpoint_produced
            and continuation_exact
        )
        return ProductionTerminalReceiptV1(
            schema_version=TERMINAL_RECEIPT_SCHEMA_VERSION_V1,
            controller_id=self._controller_id,
            run_id=self._run_id,
            terminal_status=terminal_status,
            terminal=True,
            state_revision=self._state_revision + 1,
            settlement_sequence=1,
            requested_generations=self._requested_generations,
            completed_generations_by_cell=(settlement.completed_generations_by_cell),
            cancellation_request=cancellation,
            first_device_failure=failure,
            compact_row_validity_by_cell=(settlement.compact_row_validity_by_cell),
            compact_row_status_by_cell=settlement.compact_row_status_by_cell,
            checkpoint_required=checkpoint_required,
            checkpoint_requested=settlement.checkpoint_requested,
            checkpoint_produced=settlement.checkpoint_produced,
            continuation_digest_exact=continuation_exact,
            continuation_allowed=continuation_allowed,
            ownership_claims=settlement.ownership_evidence,
            live_cuda_owner_integration_status=(LIVE_CUDA_OWNER_INTEGRATION_STATUS_V1),
            production_capability_promoted=PRODUCTION_CAPABILITY_PROMOTED_V1,
        )

    def _validate_terminal_progress(
        self,
        terminal_status: str,
        completed: tuple[int, ...],
        *,
        cancellation: CancellationRequestV1 | None,
        failure: FirstDeviceFailureRecordV1 | None,
    ) -> None:
        if terminal_status == TERMINAL_COMPLETED_V1:
            if any(count != self._requested_generations for count in completed):
                _fail(
                    "settlement.completed_generations_by_cell",
                    "completed settlement requires every requested generation",
                )
            return
        if terminal_status == TERMINAL_CANCELLED_V1:
            assert cancellation is not None
            if any(count != cancellation.requested_boundary for count in completed):
                _fail(
                    "settlement.completed_generations_by_cell",
                    "cancelled settlement must stop every cell at the first "
                    "requested generation boundary",
                )
            return

        assert failure is not None
        boundary = failure.generation_boundary
        if completed[failure.cell_index] != boundary:
            _fail(
                f"settlement.completed_generations_by_cell[{failure.cell_index}]",
                "failing cell must stop at the recorded failure boundary",
            )
        upper = min(boundary + 1, self._requested_generations)
        if cancellation is not None:
            upper = min(upper, cancellation.requested_boundary)
        if any(count < boundary or count > upper for count in completed):
            _fail(
                "settlement.completed_generations_by_cell",
                "failed settlement may contain only the common completed "
                "prefix and at most one in-flight completed row",
            )

    @staticmethod
    def _validate_checkpoint_settlement(
        *,
        checkpoint_required: bool,
        checkpoint_requested: bool,
        checkpoint_produced: bool,
        continuation_digests: ContinuationDigestReceiptV1 | None,
    ) -> None:
        if checkpoint_required and not checkpoint_requested:
            _fail(
                "settlement.checkpoint_requested",
                "terminal policy requires an explicit checkpoint request",
            )
        if checkpoint_requested and not checkpoint_produced:
            _fail(
                "settlement.checkpoint_produced",
                "requested checkpoint is absent",
            )
        if checkpoint_produced and not checkpoint_requested:
            _fail(
                "settlement.checkpoint_produced",
                "checkpoint cannot be produced without an explicit request",
            )
        if continuation_digests is not None and not checkpoint_produced:
            _fail(
                "settlement.continuation_digests",
                "digest receipt requires a produced checkpoint",
            )
