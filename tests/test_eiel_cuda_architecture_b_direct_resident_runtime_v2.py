from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest

from engine.perf.cuda_architecture_b_direct_resident_runtime_v2 import (
    CONTINUATION_FORBIDDEN,
    CONTINUATION_NEW_NATIVE_OWNER_FROM_DURABLE_CHECKPOINT_ONLY,
    CONTINUATION_NOT_APPLICABLE,
    NATIVE_OWNER_KIND,
    OWNER_INTEGRITY_INTACT,
    OWNER_INTEGRITY_SUSPECT_AND_SEALED,
    RESULT_SCHEMA_VERSION,
    TERMINAL_CANCELLED,
    TERMINAL_COMPLETED,
    TERMINAL_FAILED,
    CudaArchitectureBDirectRuntimeV2ContractError,
    checkpoint_digests_are_exact_v2,
    validate_owner_identity_v2,
    validate_terminal_result_v2,
)


def _owner(**overrides: object) -> dict[str, object]:
    value: dict[str, object] = {
        "kind": NATIVE_OWNER_KIND,
        "run_id": "patch-g64-run-19",
        "controller_id": "cuda-controller-7",
        "owner_epoch": 3,
    }
    value.update(overrides)
    return value


def _digests(*, exact: bool = True) -> dict[str, str]:
    return {
        "authoritative_state_sha256": "1" * 64,
        "checkpoint_state_sha256": ("1" if exact else "9") * 64,
        "authoritative_canonical_sha256": "2" * 64,
        "checkpoint_canonical_sha256": "2" * 64,
        "authoritative_rng_sha256": "3" * 64,
        "checkpoint_rng_sha256": "3" * 64,
        "authoritative_event_cursor_sha256": "4" * 64,
        "checkpoint_event_cursor_sha256": "4" * 64,
    }


def _result(
    terminal_status: str = TERMINAL_COMPLETED,
    *,
    start_generation: int = 10,
    requested_generations: int = 4,
    completed_generations: int | None = None,
    completed_by_cell: list[int] | None = None,
    checkpoint_produced: bool | None = None,
    checkpoint_durable: bool | None = None,
    digest_exact: bool = True,
    **overrides: object,
) -> dict[str, Any]:
    if completed_generations is None:
        completed_generations = {
            TERMINAL_COMPLETED: requested_generations,
            TERMINAL_CANCELLED: 2,
            TERMINAL_FAILED: 1,
        }[terminal_status]
    if completed_by_cell is None:
        completed_by_cell = (
            [completed_generations, completed_generations + 1]
            if terminal_status == TERMINAL_FAILED
            else [completed_generations, completed_generations]
        )
    if checkpoint_produced is None:
        checkpoint_produced = terminal_status == TERMINAL_CANCELLED
    if checkpoint_durable is None:
        checkpoint_durable = checkpoint_produced

    next_generation = start_generation + completed_generations
    last_window = (
        None
        if completed_generations == 0
        else {
            "start_generation": next_generation - min(2, completed_generations),
            "completed_generations": min(2, completed_generations),
            "next_generation": next_generation,
        }
    )
    checkpoint_usable = checkpoint_produced and checkpoint_durable and digest_exact
    cancellation = (
        {
            "request_id": "cancel-request-7",
            "reason": "operator_requested",
            "cancel_sequence": 1,
            "accepted_generation": next_generation - 1,
            "completed_tick": 0,
            "next_tick": 1,
            "consumed_next_generation": next_generation,
        }
        if terminal_status == TERMINAL_CANCELLED
        else None
    )
    failure = (
        {
            "code": "CUDA_CONTROLLER_FAILED",
            "stage": "generation",
            "message": "first native failure",
            "generation": min(
                start_generation + completed_generations,
                start_generation + requested_generations - 1,
            ),
            "tick": 0,
        }
        if terminal_status == TERMINAL_FAILED
        else None
    )
    result: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "owner_identity": _owner(),
        "terminal_status": terminal_status,
        "start_generation": start_generation,
        "requested_generations": requested_generations,
        "completed_generations": completed_generations,
        "next_generation": next_generation,
        "completed_generations_by_cell": completed_by_cell,
        "last_settled_window": last_window,
        "cancelled": terminal_status == TERMINAL_CANCELLED,
        "failed": terminal_status == TERMINAL_FAILED,
        "cancellation": cancellation,
        "failure": failure,
        "checkpoint_required": terminal_status != TERMINAL_COMPLETED,
        "checkpoint_produced": checkpoint_produced,
        "checkpoint_boundary": next_generation if checkpoint_produced else None,
        "checkpoint_digests": (
            _digests(exact=digest_exact) if checkpoint_produced else None
        ),
        "checkpoint_durable": checkpoint_durable,
        "continuation_policy": (
            CONTINUATION_NOT_APPLICABLE
            if terminal_status == TERMINAL_COMPLETED
            else (
                CONTINUATION_NEW_NATIVE_OWNER_FROM_DURABLE_CHECKPOINT_ONLY
                if checkpoint_usable
                else CONTINUATION_FORBIDDEN
            )
        ),
        "owner_integrity": (
            OWNER_INTEGRITY_SUSPECT_AND_SEALED
            if terminal_status == TERMINAL_FAILED
            else OWNER_INTEGRITY_INTACT
        ),
        "hidden_fallback": False,
        "runtime_jit": False,
        "python_window_ownership": False,
    }
    result.update(overrides)
    return result


@pytest.mark.parametrize(
    "result",
    (
        _result(),
        _result(TERMINAL_CANCELLED),
        _result(
            TERMINAL_FAILED,
            checkpoint_produced=True,
            checkpoint_durable=True,
        ),
        _result(
            TERMINAL_FAILED,
            checkpoint_produced=False,
            checkpoint_durable=False,
        ),
        _result(
            TERMINAL_FAILED,
            completed_generations=0,
            completed_by_cell=[0, 0],
        ),
    ),
)
def test_valid_terminal_results_are_copied_without_mutating_input(
    result: dict[str, Any],
) -> None:
    original = deepcopy(result)

    validated = validate_terminal_result_v2(
        result,
        expected_owner_identity=_owner(),
        expected_start_generation=10,
        expected_requested_generations=4,
        expected_cell_count=2,
    )

    assert validated == result
    assert validated is not result
    assert validated["owner_identity"] is not result["owner_identity"]
    assert result == original


def test_owner_identity_helper_is_strict_and_copies_extras() -> None:
    owner = _owner(native_control_token="resident-control-11")

    validated = validate_owner_identity_v2(owner)

    assert validated == owner
    assert validated is not owner


@pytest.mark.parametrize(
    "owner",
    (
        None,
        _owner(kind="python"),
        _owner(run_id=""),
        _owner(controller_id=" controller"),
        _owner(owner_epoch=True),
        _owner(owner_epoch=0),
    ),
)
def test_invalid_owner_identity_rejects(owner: object) -> None:
    with pytest.raises(CudaArchitectureBDirectRuntimeV2ContractError):
        validate_owner_identity_v2(owner)


def test_expected_owner_identity_mismatch_rejects() -> None:
    with pytest.raises(
        CudaArchitectureBDirectRuntimeV2ContractError,
        match="owner_identity.run_id",
    ):
        validate_terminal_result_v2(
            _result(), expected_owner_identity=_owner(run_id="different-run")
        )


@pytest.mark.parametrize(
    ("overrides", "message"),
    (
        ({"start_generation": True}, "start_generation"),
        ({"requested_generations": 0}, "requested_generations"),
        ({"completed_generations": 5}, "completed_generations"),
        ({"next_generation": 13}, "next_generation"),
        ({"completed_generations_by_cell": []}, "nonempty sequence"),
        ({"completed_generations_by_cell": [4, True]}, "exact integer"),
        ({"completed_generations_by_cell": [3, 4]}, "minimum per-cell"),
    ),
)
def test_generation_contract_rejects_contradictions(
    overrides: dict[str, object], message: str
) -> None:
    with pytest.raises(CudaArchitectureBDirectRuntimeV2ContractError, match=message):
        validate_terminal_result_v2(_result(**overrides))


@pytest.mark.parametrize(
    "result",
    (
        _result(completed_generations=3, completed_by_cell=[3, 3]),
        _result(TERMINAL_CANCELLED, completed_by_cell=[2, 3]),
        _result(
            TERMINAL_FAILED,
            completed_generations=1,
            completed_by_cell=[1, 3],
        ),
        _result(
            TERMINAL_FAILED,
            completed_generations=4,
            completed_by_cell=[4, 4],
        ),
    ),
)
def test_terminal_specific_completion_rules_reject(result: dict[str, Any]) -> None:
    with pytest.raises(CudaArchitectureBDirectRuntimeV2ContractError):
        validate_terminal_result_v2(result)


def test_unhashable_terminal_status_raises_contract_error() -> None:
    result = _result()
    result["terminal_status"] = []

    with pytest.raises(
        CudaArchitectureBDirectRuntimeV2ContractError,
        match="terminal_status",
    ):
        validate_terminal_result_v2(result)


@pytest.mark.parametrize(
    "overrides",
    (
        {"cancelled": True},
        {"failed": True},
        {"cancelled": True, "failed": True},
        {"cancelled": 1},
    ),
)
def test_cancellation_and_failure_must_be_exclusive(
    overrides: dict[str, object],
) -> None:
    with pytest.raises(
        CudaArchitectureBDirectRuntimeV2ContractError,
        match="cancelled|failed|boolean",
    ):
        validate_terminal_result_v2(_result(**overrides))


@pytest.mark.parametrize(
    ("terminal_status", "overrides", "message"),
    (
        (TERMINAL_COMPLETED, {"cancellation": {}}, "cancellation"),
        (TERMINAL_COMPLETED, {"failure": {}}, "failure"),
        (TERMINAL_CANCELLED, {"cancellation": None}, "cancellation"),
        (
            TERMINAL_CANCELLED,
            {"cancellation": {"request_id": "incomplete"}},
            "field set",
        ),
        (
            TERMINAL_FAILED,
            {"failure": None},
            "failure",
        ),
    ),
)
def test_terminal_identity_receipts_are_required_and_exclusive(
    terminal_status: str,
    overrides: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(CudaArchitectureBDirectRuntimeV2ContractError, match=message):
        validate_terminal_result_v2(_result(terminal_status, **overrides))


@pytest.mark.parametrize(
    "cancellation_overrides",
    (
        {"reason": "unsupported"},
        {"cancel_sequence": 0},
        {"next_tick": 2},
        {"consumed_next_generation": 13},
    ),
)
def test_cancellation_identity_and_coordinate_must_be_exact(
    cancellation_overrides: dict[str, object],
) -> None:
    result = _result(TERMINAL_CANCELLED)
    result["cancellation"].update(cancellation_overrides)
    with pytest.raises(CudaArchitectureBDirectRuntimeV2ContractError):
        validate_terminal_result_v2(result)


@pytest.mark.parametrize(
    "failure_overrides",
    (
        {"stage": "cuda"},
        {"code": ""},
        {"message": ""},
        {"generation": 99},
        {"tick": True},
    ),
)
def test_first_failure_identity_must_be_exact(
    failure_overrides: dict[str, object],
) -> None:
    result = _result(TERMINAL_FAILED)
    result["failure"].update(failure_overrides)
    with pytest.raises(CudaArchitectureBDirectRuntimeV2ContractError):
        validate_terminal_result_v2(result)


@pytest.mark.parametrize(
    "last_window",
    (
        None,
        {
            "start_generation": 12,
            "completed_generations": 1,
            "next_generation": 14,
        },
        {
            "start_generation": 11,
            "completed_generations": 2,
            "next_generation": 13,
        },
        {
            "start_generation": 9,
            "completed_generations": 5,
            "next_generation": 14,
        },
    ),
)
def test_last_settled_window_must_end_at_common_frontier(
    last_window: object,
) -> None:
    with pytest.raises(
        CudaArchitectureBDirectRuntimeV2ContractError,
        match="last_settled_window",
    ):
        validate_terminal_result_v2(_result(last_settled_window=last_window))


def test_zero_completion_forbids_a_last_settled_window() -> None:
    result = _result(
        TERMINAL_FAILED,
        completed_generations=0,
        completed_by_cell=[0, 0],
    )
    result["last_settled_window"] = {
        "start_generation": 10,
        "completed_generations": 1,
        "next_generation": 11,
    }

    with pytest.raises(
        CudaArchitectureBDirectRuntimeV2ContractError,
        match="no generation has settled",
    ):
        validate_terminal_result_v2(result)


@pytest.mark.parametrize(
    ("overrides", "message"),
    (
        ({"checkpoint_required": True}, "checkpoint_required"),
        (
            {
                "checkpoint_produced": False,
                "checkpoint_boundary": 14,
                "checkpoint_digests": None,
                "checkpoint_durable": False,
            },
            "checkpoint_boundary",
        ),
        (
            {
                "checkpoint_produced": False,
                "checkpoint_boundary": None,
                "checkpoint_digests": _digests(),
                "checkpoint_durable": False,
            },
            "checkpoint_digests",
        ),
        (
            {
                "checkpoint_produced": False,
                "checkpoint_boundary": None,
                "checkpoint_digests": None,
                "checkpoint_durable": True,
            },
            "checkpoint_durable",
        ),
    ),
)
def test_checkpoint_presence_contract_rejects(
    overrides: dict[str, object], message: str
) -> None:
    with pytest.raises(CudaArchitectureBDirectRuntimeV2ContractError, match=message):
        validate_terminal_result_v2(_result(**overrides))


@pytest.mark.parametrize(
    "result",
    (
        _result(
            TERMINAL_CANCELLED,
            checkpoint_produced=False,
            checkpoint_durable=False,
        ),
        _result(
            TERMINAL_CANCELLED,
            checkpoint_produced=True,
            checkpoint_durable=False,
        ),
        _result(
            TERMINAL_CANCELLED,
            checkpoint_produced=True,
            checkpoint_durable=True,
            digest_exact=False,
        ),
    ),
)
def test_cancelled_terminal_requires_exact_durable_checkpoint(
    result: dict[str, Any],
) -> None:
    with pytest.raises(
        CudaArchitectureBDirectRuntimeV2ContractError,
        match="exact durable checkpoint",
    ):
        validate_terminal_result_v2(result)


def test_failed_nonexact_checkpoint_is_sealed_and_nonresumable() -> None:
    result = _result(
        TERMINAL_FAILED,
        checkpoint_produced=True,
        checkpoint_durable=True,
        digest_exact=False,
    )

    validated = validate_terminal_result_v2(result)

    assert validated["continuation_policy"] == CONTINUATION_FORBIDDEN
    assert checkpoint_digests_are_exact_v2(validated["checkpoint_digests"]) is False


@pytest.mark.parametrize(
    "overrides",
    (
        {"continuation_policy": CONTINUATION_FORBIDDEN},
        {"owner_integrity": OWNER_INTEGRITY_SUSPECT_AND_SEALED},
        {"hidden_fallback": True},
        {"runtime_jit": True},
        {"python_window_ownership": True},
    ),
)
def test_ownership_and_continuation_claims_fail_closed(
    overrides: dict[str, object],
) -> None:
    with pytest.raises(CudaArchitectureBDirectRuntimeV2ContractError):
        validate_terminal_result_v2(_result(**overrides))


def test_missing_required_result_field_rejects() -> None:
    result = _result()
    del result["checkpoint_durable"]

    with pytest.raises(
        CudaArchitectureBDirectRuntimeV2ContractError,
        match="checkpoint_durable",
    ):
        validate_terminal_result_v2(result)


def test_result_must_be_mapping_and_metadata_must_be_json_safe() -> None:
    with pytest.raises(CudaArchitectureBDirectRuntimeV2ContractError):
        validate_terminal_result_v2([])

    with pytest.raises(
        CudaArchitectureBDirectRuntimeV2ContractError,
        match="JSON-safe",
    ):
        validate_terminal_result_v2(_result(native_handle=object()))
