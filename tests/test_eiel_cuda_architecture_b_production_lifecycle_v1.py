from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import pytest

from engine.perf.cuda_architecture_b_production_lifecycle_v1 import (
    CANCEL_ACCEPTED_V1,
    CANCEL_ALREADY_REQUESTED_V1,
    TERMINAL_CANCELLED_V1,
    TERMINAL_COMPLETED_V1,
    TERMINAL_FAILED_V1,
    CancellationRequestV1,
    CheckpointPolicyV1,
    ContinuationDigestReceiptV1,
    CudaArchitectureBProductionLifecycleV1,
    ExecutionOwnershipEvidenceV1,
    FirstDeviceFailureRecordV1,
    ProductionLifecycleContractError,
    ProductionLifecycleMutationError,
    ProductionSettlementInputV1,
)


CONTROLLER_ID = "cuda-controller-7"
RUN_ID = "patch-g64-run-19"
CELLS = 2
GENERATIONS = 3


def _lifecycle() -> CudaArchitectureBProductionLifecycleV1:
    return CudaArchitectureBProductionLifecycleV1(
        controller_id=CONTROLLER_ID,
        run_id=RUN_ID,
        cell_count=CELLS,
        requested_generations=GENERATIONS,
        checkpoint_policy=CheckpointPolicyV1(
            checkpoint_on_cancel_required=True,
            checkpoint_on_failure_required=True,
        ),
    )


def _digests() -> ContinuationDigestReceiptV1:
    return ContinuationDigestReceiptV1(
        authoritative_state_sha256="1" * 64,
        checkpoint_state_sha256="1" * 64,
        authoritative_canonical_sha256="2" * 64,
        checkpoint_canonical_sha256="2" * 64,
        authoritative_rng_sha256="3" * 64,
        checkpoint_rng_sha256="3" * 64,
        authoritative_event_cursor_sha256="4" * 64,
        checkpoint_event_cursor_sha256="4" * 64,
    )


def _failure(*, cell_index: int = 0, boundary: int = 1) -> FirstDeviceFailureRecordV1:
    return FirstDeviceFailureRecordV1(
        cell_index=cell_index,
        generation_boundary=boundary,
        tick_index=17,
        stage="birth_event_append",
        status=91,
        failure_code="event_capacity_exceeded",
    )


def _rows(
    completed: tuple[int, ...],
    *,
    failure: FirstDeviceFailureRecordV1 | None = None,
) -> tuple[tuple[tuple[bool, ...], ...], tuple[tuple[int, ...], ...]]:
    validity: list[tuple[bool, ...]] = []
    statuses: list[tuple[int, ...]] = []
    for cell_index, completed_count in enumerate(completed):
        validity.append(
            tuple(generation < completed_count for generation in range(GENERATIONS))
        )
        statuses.append(
            tuple(
                0
                if generation < completed_count
                else (
                    failure.status
                    if failure is not None
                    and failure.cell_index == cell_index
                    and failure.generation_boundary == generation
                    else -1
                )
                for generation in range(GENERATIONS)
            )
        )
    return tuple(validity), tuple(statuses)


def _settlement(
    completed: tuple[int, ...],
    *,
    failure: FirstDeviceFailureRecordV1 | None = None,
    checkpoint_requested: bool = False,
    checkpoint_produced: bool = False,
    continuation_digests: ContinuationDigestReceiptV1 | None = None,
    ownership_evidence: ExecutionOwnershipEvidenceV1 = (ExecutionOwnershipEvidenceV1()),
) -> ProductionSettlementInputV1:
    validity, statuses = _rows(completed, failure=failure)
    return ProductionSettlementInputV1(
        controller_id=CONTROLLER_ID,
        run_id=RUN_ID,
        completed_generations_by_cell=completed,
        compact_row_validity_by_cell=validity,
        compact_row_status_by_cell=statuses,
        first_device_failure=failure,
        checkpoint_requested=checkpoint_requested,
        checkpoint_produced=checkpoint_produced,
        continuation_digests=continuation_digests,
        ownership_evidence=ownership_evidence,
    )


def _request(
    request_id: str = "cancel-first", *, boundary: int = 1
) -> CancellationRequestV1:
    return CancellationRequestV1(
        request_id=request_id,
        controller_id=CONTROLLER_ID,
        run_id=RUN_ID,
        requested_boundary=boundary,
    )


def test_normal_completion_has_conservative_claims_and_no_promotion() -> None:
    lifecycle = _lifecycle()
    receipt = lifecycle.settle(_settlement((3, 3)))

    assert receipt.terminal_status == TERMINAL_COMPLETED_V1
    assert receipt.completed_generations_by_cell == (3, 3)
    assert receipt.ownership_claims == ExecutionOwnershipEvidenceV1()
    assert receipt.ownership_claims.to_dict() == {
        "no_hidden_fallback": False,
        "no_runtime_jit": False,
        "no_aten_tick_or_generation_ownership": False,
        "no_python_tick_ownership": False,
        "no_python_generation_ownership": False,
    }
    assert receipt.live_cuda_owner_integration_status == "pending"
    assert receipt.production_capability_promoted is False
    assert len(receipt.receipt_sha256) == 64


def test_first_cancellation_request_wins_at_a_generation_boundary() -> None:
    lifecycle = _lifecycle()

    first = lifecycle.request_cancellation(_request(boundary=1))
    second = lifecycle.request_cancellation(_request("cancel-second", boundary=2))
    receipt = lifecycle.settle(
        _settlement(
            (1, 1),
            checkpoint_requested=True,
            checkpoint_produced=True,
            continuation_digests=_digests(),
        )
    )

    assert first.disposition == CANCEL_ACCEPTED_V1
    assert first.generation_boundary_only is True
    assert first.tick_polling is False
    assert second.disposition == CANCEL_ALREADY_REQUESTED_V1
    assert second.first_request_id == "cancel-first"
    assert second.requested_boundary == 1
    assert lifecycle.state_revision == 2
    assert receipt.terminal_status == TERMINAL_CANCELLED_V1
    assert receipt.cancellation_request == _request(boundary=1)
    assert receipt.checkpoint_required is True
    assert receipt.continuation_allowed is True


def test_cancellation_boundary_query_returns_only_at_accepted_boundary() -> None:
    lifecycle = _lifecycle()

    assert lifecycle.cancellation_due_at_boundary(3) is None
    lifecycle.request_cancellation(_request(boundary=2))
    revision = lifecycle.state_revision

    assert lifecycle.cancellation_due_at_boundary(0) is None
    assert lifecycle.cancellation_due_at_boundary(1) is None
    assert lifecycle.cancellation_due_at_boundary(2) == _request(boundary=2)
    assert lifecycle.state_revision == revision
    assert lifecycle.terminal_receipt is None


@pytest.mark.parametrize("boundary", [True, 1.0, -1, GENERATIONS + 1])
def test_cancellation_boundary_query_rejects_invalid_boundaries(
    boundary: object,
) -> None:
    lifecycle = _lifecycle()

    with pytest.raises(
        ProductionLifecycleContractError,
        match="completed_generation_boundary: must be an exact integer",
    ):
        lifecycle.cancellation_due_at_boundary(boundary)  # type: ignore[arg-type]

    assert lifecycle.state_revision == 0
    assert lifecycle.terminal_receipt is None


def test_cancellation_boundary_query_fails_closed_after_accepted_boundary() -> None:
    lifecycle = _lifecycle()
    lifecycle.request_cancellation(_request(boundary=1))
    revision = lifecycle.state_revision

    with pytest.raises(
        ProductionLifecycleContractError,
        match="cannot be after the first accepted cancellation boundary",
    ):
        lifecycle.cancellation_due_at_boundary(2)

    assert lifecycle.cancellation_due_at_boundary(1) == _request(boundary=1)
    assert lifecycle.state_revision == revision
    assert lifecycle.terminal_receipt is None


def test_authoritative_device_failure_settles_failed() -> None:
    lifecycle = _lifecycle()
    failure = _failure()
    receipt = lifecycle.settle(
        _settlement(
            (1, 2),
            failure=failure,
            checkpoint_requested=True,
            checkpoint_produced=True,
            continuation_digests=_digests(),
            ownership_evidence=ExecutionOwnershipEvidenceV1(no_hidden_fallback=True),
        )
    )

    assert receipt.terminal_status == TERMINAL_FAILED_V1
    assert receipt.first_device_failure == failure
    assert receipt.compact_row_status_by_cell[0][1] == failure.status
    assert receipt.checkpoint_required is True
    assert receipt.continuation_allowed is True
    assert receipt.ownership_claims.no_hidden_fallback is True
    assert receipt.ownership_claims.no_runtime_jit is False


def test_failure_wins_an_exact_cancellation_boundary_tie() -> None:
    lifecycle = _lifecycle()
    lifecycle.request_cancellation(_request(boundary=1))
    failure = _failure(boundary=1)

    receipt = lifecycle.settle(
        _settlement(
            (1, 1),
            failure=failure,
            checkpoint_requested=True,
            checkpoint_produced=True,
            continuation_digests=_digests(),
        )
    )

    assert receipt.terminal_status == TERMINAL_FAILED_V1
    assert receipt.cancellation_request is not None
    assert (
        receipt.first_device_failure.generation_boundary
        == receipt.cancellation_request.requested_boundary
    )


def test_invalid_compact_tail_fails_closed() -> None:
    lifecycle = _lifecycle()
    settlement = _settlement((2, 2))
    invalid_tail = ((True, False, False), settlement.compact_row_validity_by_cell[1])

    with pytest.raises(
        ProductionLifecycleContractError, match="invalid tail|non-prefix"
    ):
        lifecycle.settle(replace(settlement, compact_row_validity_by_cell=invalid_tail))

    assert lifecycle.terminal_receipt is None
    assert lifecycle.state_revision == 0


@pytest.mark.parametrize(
    ("checkpoint_requested", "checkpoint_produced", "message"),
    [
        (False, False, "requires an explicit checkpoint request"),
        (True, False, "requested checkpoint is absent"),
    ],
)
def test_cancel_checkpoint_requirement_fails_closed_when_absent(
    checkpoint_requested: bool,
    checkpoint_produced: bool,
    message: str,
) -> None:
    lifecycle = _lifecycle()
    lifecycle.request_cancellation(_request(boundary=1))

    with pytest.raises(ProductionLifecycleContractError, match=message):
        lifecycle.settle(
            _settlement(
                (1, 1),
                checkpoint_requested=checkpoint_requested,
                checkpoint_produced=checkpoint_produced,
            )
        )

    assert lifecycle.terminal_receipt is None


def test_mismatched_checkpoint_digests_disable_continuation() -> None:
    lifecycle = _lifecycle()
    lifecycle.request_cancellation(_request(boundary=1))
    mismatched = replace(_digests(), checkpoint_rng_sha256="9" * 64)

    receipt = lifecycle.settle(
        _settlement(
            (1, 1),
            checkpoint_requested=True,
            checkpoint_produced=True,
            continuation_digests=mismatched,
        )
    )

    assert receipt.terminal_status == TERMINAL_CANCELLED_V1
    assert receipt.continuation_digest_exact is False
    assert receipt.continuation_allowed is False


def test_repeated_identical_settlement_is_idempotent() -> None:
    lifecycle = _lifecycle()
    settlement = _settlement((3, 3))

    first = lifecycle.settle(settlement)
    revision = lifecycle.state_revision
    second = lifecycle.settle(settlement)

    assert second is first
    assert lifecycle.state_revision == revision == 1
    assert second.receipt_sha256 == first.receipt_sha256


def test_terminal_receipt_is_frozen_and_all_mutations_are_rejected() -> None:
    lifecycle = _lifecycle()
    settlement = _settlement((3, 3))
    receipt = lifecycle.settle(settlement)

    with pytest.raises(FrozenInstanceError):
        receipt.terminal_status = TERMINAL_FAILED_V1  # type: ignore[misc]
    with pytest.raises(ProductionLifecycleMutationError, match="non-identical"):
        lifecycle.settle(replace(settlement, checkpoint_requested=True))
    with pytest.raises(ProductionLifecycleMutationError, match="terminal"):
        lifecycle.request_cancellation(_request())

    assert lifecycle.terminal_receipt is receipt
    assert lifecycle.state_revision == 1


def test_controller_and_run_identity_mismatches_do_not_mutate_state() -> None:
    lifecycle = _lifecycle()
    wrong_request = replace(_request(), controller_id="other-controller")

    with pytest.raises(ProductionLifecycleContractError, match="identity mismatch"):
        lifecycle.request_cancellation(wrong_request)
    assert lifecycle.state_revision == 0

    settlement = _settlement((3, 3))
    with pytest.raises(ProductionLifecycleContractError, match="identity mismatch"):
        lifecycle.settle(replace(settlement, run_id="other-run"))
    assert lifecycle.terminal_receipt is None

    receipt = lifecycle.settle(settlement)
    assert receipt.controller_id == CONTROLLER_ID
    assert receipt.run_id == RUN_ID
