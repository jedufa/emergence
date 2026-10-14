from __future__ import annotations

import copy
import threading
from types import SimpleNamespace
from typing import Any

import pytest

from native_runtime import run_controller as run_controller_module
from native_runtime.run_controller import (
    NATIVE_CHECKPOINT_ON_CANCEL_CAPABILITY,
    NATIVE_CONTROLLER_PROTOCOL_SCHEMA_VERSION,
    NATIVE_CONTROLLER_PROTOCOL_VERSION,
    NATIVE_PHASE4_REQUIRED_CAPABILITIES,
    NATIVE_RESULT_SCHEMA_VERSION,
    NativeRunController,
    NativeRunControllerError,
)


RUN_ID = "cancellation-run"
CONTROLLER_ID = "cancellation-controller"
OWNER_EPOCH = 23


class _FakeNativeOwnerBase:
    def __init__(self) -> None:
        self.status = "initialized"
        self.sealed = False
        self.final: dict[str, Any] | None = None

    def controller_manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "eiel_native_architecture_b_controller_manifest_v1",
            "run_id": RUN_ID,
            "controller_id": CONTROLLER_ID,
            "owner_epoch": OWNER_EPOCH,
            "status": self.status,
            "run_started": self.status != "initialized",
            "sealed": self.sealed,
            "native_owner_private": True,
            "python_callback_entry_points": 0,
        }

    def final_result(self) -> dict[str, Any]:
        if self.final is None:
            return {"terminal": False, "status": self.status}
        return copy.deepcopy(self.final)

    def start(self) -> None:
        self.status = "running"

    def terminalize(self, status: str) -> None:
        self.status = status
        self.sealed = True
        self.final = {
            "schema_version": "eiel_native_architecture_b_run_result_v1",
            "run_id": RUN_ID,
            "controller_id": CONTROLLER_ID,
            "owner_epoch": OWNER_EPOCH,
            "terminal_status": status,
            "terminal_marker": f"stable-{status}",
        }


class _FakeNativeOwner(_FakeNativeOwnerBase):
    def __init__(self) -> None:
        super().__init__()
        self.cancel_requests: list[dict[str, Any]] = []
        self.first_cancel_request_id: str | None = None
        self.first_cancel_reason: str | None = None

    def request_cancel(self, request: dict[str, Any]) -> dict[str, Any]:
        self.cancel_requests.append(copy.deepcopy(request))
        if self.status in {"run_complete", "cancelled", "failed"}:
            disposition = "already_terminal"
        elif self.status != "running":
            disposition = "not_running"
        elif self.first_cancel_request_id is None:
            self.first_cancel_request_id = str(request["request_id"])
            self.first_cancel_reason = str(request["reason"])
            disposition = "accepted"
        else:
            disposition = "already_requested"

        return {
            "request_id": request["request_id"],
            "controller_id": CONTROLLER_ID,
            "owner_epoch": OWNER_EPOCH,
            "disposition": disposition,
            "first_cancel_request_id": self.first_cancel_request_id,
            "cancel_sequence": (
                1 if self.first_cancel_request_id is not None else None
            ),
            "state_observed": self.status,
            "terminal_status_changed": False,
        }


class _Phase4CancellationOwner(_FakeNativeOwnerBase):
    """Focused owner fake for the native active-run cancellation path."""

    def __init__(
        self,
        *,
        receipt_mutator: Any | None = None,
    ) -> None:
        super().__init__()
        self.receipt_mutator = receipt_mutator
        self.dispatch_started = threading.Event()
        self.cancel_latched = threading.Event()
        self.release_dispatch = threading.Event()
        self.cancel_requests: list[dict[str, Any]] = []
        self.next_generation_started = False
        self.first_cancel_request_id: str | None = None
        self.first_cancel_reason: str | None = None

    def controller_manifest(self) -> dict[str, Any]:
        manifest = super().controller_manifest()
        manifest.update(
            {
                "controller_protocol_version": NATIVE_CONTROLLER_PROTOCOL_VERSION,
                "controller_protocol_schema_version": (
                    NATIVE_CONTROLLER_PROTOCOL_SCHEMA_VERSION
                ),
                "controller_protocol_capabilities": {
                    capability: True
                    for capability in NATIVE_PHASE4_REQUIRED_CAPABILITIES
                },
                "persistence_mailbox_required": True,
            }
        )
        return manifest

    def request_cancel(self, request: dict[str, Any]) -> dict[str, Any]:
        self.cancel_requests.append(copy.deepcopy(request))
        if self.first_cancel_request_id is None:
            assert self.status == "tick_running"
            self.first_cancel_request_id = str(request["request_id"])
            self.first_cancel_reason = str(request["reason"])
            self.cancel_latched.set()
            disposition = "accepted"
        else:
            disposition = "already_requested"
        return {
            "request_id": request["request_id"],
            "controller_id": CONTROLLER_ID,
            "owner_epoch": OWNER_EPOCH,
            "disposition": disposition,
            "first_cancel_request_id": self.first_cancel_request_id,
            "cancel_sequence": 1,
            "state_observed": "running",
            "terminal_status_changed": False,
        }

    def run_generations(
        self,
        start_generation: int,
        generation_count: int,
        steps_per_generation: int,
        max_births: int,
        schedule: dict[str, Any],
    ) -> dict[str, Any]:
        assert (start_generation, generation_count, steps_per_generation, max_births) == (
            4,
            2,
            8,
            2,
        )
        assert schedule["checkpoint_on_cancel"] is True
        self.status = "tick_running"
        self.dispatch_started.set()
        assert self.release_dispatch.wait(1)
        assert self.first_cancel_request_id is not None
        assert self.first_cancel_reason == "operator_requested"

        snapshot_sequence = 10
        snapshot_id = f"{CONTROLLER_ID}:owner:{OWNER_EPOCH}:boundary:{snapshot_sequence}"
        cursor = "generation:5:tick:0"
        digest = "d" * 64
        request_receipt = {
            "request_id": self.first_cancel_request_id,
            "controller_id": CONTROLLER_ID,
            "owner_epoch": OWNER_EPOCH,
            "disposition": "accepted",
            "first_cancel_request_id": self.first_cancel_request_id,
            "cancel_sequence": 1,
            "state_observed": "running",
            "terminal_status_changed": False,
        }
        observation_receipt = {
            "schema_version": (
                "eiel_native_architecture_b_cancellation_observation_receipt_v1"
            ),
            "request_id": self.first_cancel_request_id,
            "reason": self.first_cancel_reason,
            "controller_id": CONTROLLER_ID,
            "owner_epoch": OWNER_EPOCH,
            "cancel_sequence": 1,
            "observation_cursor": cursor,
            "observed_after_completed_generation": 4,
            "boundary_ordinal": snapshot_sequence,
            "snapshot_sequence": snapshot_sequence,
            "snapshot_id": snapshot_id,
            "snapshot_digest_sha256": digest,
            "checkpoint_requested": True,
            "checkpoint_persisted": True,
        }
        terminal_receipt = {
            "schema_version": (
                "eiel_native_architecture_b_cancellation_terminal_receipt_v1"
            ),
            "request_id": self.first_cancel_request_id,
            "controller_id": CONTROLLER_ID,
            "owner_epoch": OWNER_EPOCH,
            "terminal_status": "cancelled",
            "request_observation_cursor": cursor,
            "final_safe_cursor": cursor,
            "trajectory_commit_frontier": 4,
            "snapshot_frontier": 4,
            "output_ack_frontier": 4,
            "snapshot_sequence": snapshot_sequence,
            "snapshot_id": snapshot_id,
            "snapshot_digest_sha256": digest,
            "cancellation_checkpoint_requested": True,
            "cancellation_checkpoint_persisted": True,
            "success_marker_allowed": False,
            "promotion_result_allowed": False,
        }
        self.status = "cancelled"
        self.sealed = True
        self.final = {
            "schema_version": NATIVE_RESULT_SCHEMA_VERSION,
            "decision": "ARCHB-CPU-VERTICAL-SLICE-NATIVE-RUN-MANY",
            "run_id": RUN_ID,
            "controller_id": CONTROLLER_ID,
            "owner_epoch": OWNER_EPOCH,
            "controller_protocol_version": NATIVE_CONTROLLER_PROTOCOL_VERSION,
            "executed": True,
            "native_owner": True,
            "hidden_fallback": False,
            "runtime_jit": False,
            "terminal_status": "cancelled",
            "start_generation": 4,
            "requested_generations": 2,
            "completed_generations": 1,
            "trajectory_commit_frontier": 4,
            "snapshot_frontier": 4,
            "next_generation": 5,
            "next_tick": 0,
            "controller_invocations": 1,
            "python_generation_dispatches": 0,
            "native_generation_transactions": 1,
            "turnover_commits": 1,
            "boundary_snapshot_count": 1,
            "projection_export_count": 1,
            "observer_live_owner_exports": 0,
            "persistence_acknowledgement_count": 1,
            "projection_tensor_bytes": 0,
            "boundary_wall_time_ns": 1,
            "boundaries": [{}],
            "final_snapshot_sequence": snapshot_sequence,
            "success_marker_allowed": False,
            "promotion_result_allowed": False,
            "cancel_request_id": self.first_cancel_request_id,
            "cancel_request_observation_cursor": cursor,
            "final_safe_cursor": cursor,
            "output_ack_frontier": 4,
            "cancellation_checkpoint_requested": True,
            "cancellation_checkpoint_persisted": True,
            "cancelled_after_completed_generation": 4,
            "cancelled_at_generation_boundary": 5,
            "cancelled_snapshot_sequence": snapshot_sequence,
            "cancelled_snapshot_id": snapshot_id,
            "cancelled_snapshot_completion_disposition": (
                "cancelled_no_success_or_promotion"
            ),
            "cancelled_snapshot_durable": True,
            "cancel_request_receipt": request_receipt,
            "cancel_observation_receipt": observation_receipt,
            "cancel_terminal_receipt": terminal_receipt,
        }
        if self.receipt_mutator is not None:
            self.receipt_mutator(self.final)
        return copy.deepcopy(self.final)


def _controller(native: _FakeNativeOwnerBase) -> NativeRunController:
    return NativeRunController(
        native,
        start_generation=4,
        default_steps_per_generation=8,
        default_max_births=2,
        safe_default_decision={"supported": True},
        extension_path="/not-loaded-by-cancellation-tests.so",
        run_id=RUN_ID,
    )


def _phase4_controller(native: _Phase4CancellationOwner) -> NativeRunController:
    return NativeRunController(
        native,
        start_generation=4,
        default_steps_per_generation=8,
        default_max_births=2,
        safe_default_decision={"supported": True},
        extension_path="/not-loaded-by-cancellation-tests.so",
        run_id=RUN_ID,
        persistence_mailbox_required=True,
        initial_boundary_ordinal=9,
        schedule_origin_generation=4,
        phase4_capabilities=NATIVE_PHASE4_REQUIRED_CAPABILITIES
        | frozenset({NATIVE_CHECKPOINT_ON_CANCEL_CAPABILITY}),
    )


def _run_phase4_cancellation(
    monkeypatch: pytest.MonkeyPatch,
    *,
    receipt_mutator: Any | None = None,
) -> tuple[dict[str, Any], dict[str, Any], _Phase4CancellationOwner]:
    native = _Phase4CancellationOwner(receipt_mutator=receipt_mutator)
    controller = _phase4_controller(native)
    monkeypatch.setattr(
        controller,
        "_normalize_scheduled_boundary",
        lambda _raw, **_kwargs: {"tensor_manifest_boundary_bytes": {"total": 0}},
    )

    class _PersistenceLoop:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            pass

        def run(self) -> dict[str, Any]:
            assert native.dispatch_started.wait(1)
            assert native.cancel_latched.wait(1)
            native.release_dispatch.set()
            return {
                "schema_version": "eiel_architecture_b_persistence_loop_result_v1",
                "stop_reason": "terminal_state_seen",
                "terminal_state_seen": "cancelled",
                "snapshots_seen": 1,
                "progress_only_snapshots": 0,
                "skipped_snapshot_sequences": 0,
                "persistence_acknowledgements": 1,
                "identical_acknowledgement_retries": 0,
                "checkpoints_persisted": 1,
                "architecture_b_checkpoint_bundle_paths": ["runtime/cancel.bundle"],
                "last_snapshot_sequence": 10,
                "last_progress": None,
            }

    monkeypatch.setattr(
        run_controller_module,
        "ArchitectureBPersistenceLoop",
        _PersistenceLoop,
    )
    result_box: dict[str, dict[str, Any]] = {}
    error_box: dict[str, BaseException] = {}

    def _run() -> None:
        try:
            result_box["result"] = controller.run_with_persistence(
                SimpleNamespace(run_dir="/fake"),
                2,
                snapshot_schedule={
                    "schema_version": "eiel_native_snapshot_schedule_v1",
                    "progress_every_generations": 0,
                    "report_every_generations": 0,
                    "checkpoint_every_generations": 0,
                    "explicit_next_generations": [],
                    "emit_final_snapshot": True,
                    "typed_lifecycle_mode": "cursor_bounded_delta",
                    "mailbox_capacity": 1,
                    "checkpoint_on_cancel": True,
                },
                join_timeout_seconds=1,
            )
        except BaseException as exc:
            error_box["error"] = exc

    worker = threading.Thread(target=_run)
    worker.start()
    assert native.dispatch_started.wait(1)
    accepted = controller.request_cancel("cancel-at-safe-boundary")
    worker.join(2)
    assert not worker.is_alive()
    if "error" in error_box:
        raise error_box["error"]
    return accepted, result_box["result"], native


def test_request_cancel_is_an_explicit_idempotent_native_owner_operation() -> None:
    native = _FakeNativeOwner()
    controller = _controller(native)
    native.start()

    accepted = controller.request_cancel(
        "cancel-first",
        reason="operator_requested",
    )
    repeated = controller.request_cancel(
        "cancel-second",
        reason="supervisor_requested",
    )

    assert native.cancel_requests == [
        {
            "api_version": 1,
            "request_id": "cancel-first",
            "expected_controller_id": CONTROLLER_ID,
            "expected_owner_epoch": OWNER_EPOCH,
            "reason": "operator_requested",
        },
        {
            "api_version": 1,
            "request_id": "cancel-second",
            "expected_controller_id": CONTROLLER_ID,
            "expected_owner_epoch": OWNER_EPOCH,
            "reason": "supervisor_requested",
        },
    ]
    assert accepted["disposition"] == "accepted"
    assert repeated["disposition"] == "already_requested"
    assert repeated["first_cancel_request_id"] == "cancel-first"
    assert repeated["cancel_sequence"] == accepted["cancel_sequence"] == 1
    assert native.first_cancel_reason == "operator_requested"
    assert native.status == "running"
    assert accepted["terminal_status_changed"] is False
    assert controller.final_result() == {"terminal": False, "status": "running"}


def test_request_cancel_fails_closed_when_native_owner_has_no_operation() -> None:
    native = _FakeNativeOwnerBase()
    controller = _controller(native)
    native.start()

    with pytest.raises(
        NativeRunControllerError,
        match="native owner does not support complete-generation-boundary cancellation",
    ):
        controller.request_cancel("unsupported-owner")

    assert native.status == "running"
    assert controller.final_result() == {"terminal": False, "status": "running"}


def test_request_cancel_uses_native_owner_for_not_running_disposition() -> None:
    native = _FakeNativeOwner()
    controller = _controller(native)

    receipt = controller.request_cancel("before-run")

    assert receipt["disposition"] == "not_running"
    assert receipt["state_observed"] == "initialized"
    assert receipt["first_cancel_request_id"] is None
    assert native.cancel_requests[0]["request_id"] == "before-run"
    assert controller.final_result() == {"terminal": False, "status": "initialized"}


def test_request_cancel_preserves_active_run_when_control_point_is_unavailable() -> None:
    class _UnavailableControlPointOwner(_FakeNativeOwner):
        def request_cancel(self, request: dict[str, Any]) -> dict[str, Any]:
            self.cancel_requests.append(copy.deepcopy(request))
            return {
                "request_id": request["request_id"],
                "controller_id": CONTROLLER_ID,
                "owner_epoch": OWNER_EPOCH,
                "disposition": "control_point_unavailable",
                "first_cancel_request_id": None,
                "cancel_sequence": None,
                "state_observed": "running",
                "terminal_status_changed": False,
            }

    native = _UnavailableControlPointOwner()
    controller = _controller(native)
    native.start()

    receipt = controller.request_cancel("outside-control-point")

    assert receipt["disposition"] == "control_point_unavailable"
    assert receipt["first_cancel_request_id"] is None
    assert receipt["cancel_sequence"] is None
    assert native.first_cancel_request_id is None
    assert native.status == "running"


def test_unavailable_control_point_allows_independent_terminal_progress() -> None:
    class _ProgressingUnavailableControlPointOwner(_FakeNativeOwner):
        def request_cancel(self, request: dict[str, Any]) -> dict[str, Any]:
            self.cancel_requests.append(copy.deepcopy(request))
            receipt = {
                "request_id": request["request_id"],
                "controller_id": CONTROLLER_ID,
                "owner_epoch": OWNER_EPOCH,
                "disposition": "control_point_unavailable",
                "first_cancel_request_id": None,
                "cancel_sequence": None,
                "state_observed": "running",
                "terminal_status_changed": False,
            }
            self.terminalize("run_complete")
            return receipt

    native = _ProgressingUnavailableControlPointOwner()
    controller = _controller(native)
    native.start()

    receipt = controller.request_cancel("outside-control-point-during-progress")

    assert receipt["disposition"] == "control_point_unavailable"
    assert native.first_cancel_request_id is None
    assert native.status == "run_complete"


def test_request_cancel_rejects_mid_generation_semantics_before_native_call() -> None:
    native = _FakeNativeOwner()
    controller = _controller(native)
    native.start()

    with pytest.raises(
        NativeRunControllerError,
        match="available only at a complete-generation boundary",
    ):
        controller.request_cancel("cancel-now", reason="immediate")

    assert native.cancel_requests == []
    assert native.status == "running"


@pytest.mark.parametrize("terminal_status", ["run_complete", "cancelled", "failed"])
def test_request_cancel_preserves_existing_terminal_status_and_result(
    terminal_status: str,
) -> None:
    native = _FakeNativeOwner()
    controller = _controller(native)
    if terminal_status == "cancelled":
        native.first_cancel_request_id = "earlier-cancel"
        native.first_cancel_reason = "signal_received"
    native.terminalize(terminal_status)
    before = controller.final_result()

    receipt = controller.request_cancel("after-terminal")
    after = controller.final_result()

    assert receipt["disposition"] == "already_terminal"
    assert receipt["state_observed"] == terminal_status
    assert receipt["terminal_status_changed"] is False
    assert after == before
    assert after["terminal_status"] == terminal_status


def test_request_cancel_rejects_receipt_that_claims_terminal_transition() -> None:
    class _InvalidReceiptOwner(_FakeNativeOwner):
        def request_cancel(self, request: dict[str, Any]) -> dict[str, Any]:
            receipt = super().request_cancel(request)
            receipt["terminal_status_changed"] = True
            return receipt

    native = _InvalidReceiptOwner()
    controller = _controller(native)
    native.start()

    with pytest.raises(
        NativeRunControllerError,
        match="terminal_status_changed must be False",
    ):
        controller.request_cancel("invalid-transition")

    assert native.status == "running"


def test_phase4_active_run_cancellation_closes_receipts_at_next_safe_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    accepted, result, native = _run_phase4_cancellation(monkeypatch)

    assert accepted == {
        "request_id": "cancel-at-safe-boundary",
        "controller_id": CONTROLLER_ID,
        "owner_epoch": OWNER_EPOCH,
        "disposition": "accepted",
        "first_cancel_request_id": "cancel-at-safe-boundary",
        "cancel_sequence": 1,
        "state_observed": "running",
        "terminal_status_changed": False,
    }
    assert native.cancel_requests == [
        {
            "api_version": 1,
            "request_id": "cancel-at-safe-boundary",
            "expected_controller_id": CONTROLLER_ID,
            "expected_owner_epoch": OWNER_EPOCH,
            "reason": "operator_requested",
        }
    ]
    assert native.next_generation_started is False
    assert result["terminal_status"] == "cancelled"
    assert result["cancel_request_id"] == "cancel-at-safe-boundary"
    assert result["cancel_request_observation_cursor"] == "generation:5:tick:0"
    assert result["final_safe_cursor"] == "generation:5:tick:0"
    assert result["trajectory_commit_frontier"] == 4
    assert result["snapshot_frontier"] == result["output_ack_frontier"] == 4
    assert result["cancellation_checkpoint_requested"] is True
    assert result["cancellation_checkpoint_persisted"] is True
    assert result["cancelled_snapshot_durable"] is True
    assert result["success_marker_allowed"] is False
    assert result["promotion_result_allowed"] is False

    observation = result["cancel_observation_receipt"]
    terminal = result["cancel_terminal_receipt"]
    assert observation["request_id"] == terminal["request_id"] == result[
        "cancel_request_id"
    ]
    assert observation["controller_id"] == terminal["controller_id"] == CONTROLLER_ID
    assert observation["owner_epoch"] == terminal["owner_epoch"] == OWNER_EPOCH
    assert observation["observation_cursor"] == terminal[
        "request_observation_cursor"
    ] == terminal["final_safe_cursor"] == result["final_safe_cursor"]
    assert observation["observed_after_completed_generation"] == terminal[
        "trajectory_commit_frontier"
    ]
    assert observation["checkpoint_requested"] is True
    assert observation["checkpoint_persisted"] is True
    assert terminal["cancellation_checkpoint_requested"] is True
    assert terminal["cancellation_checkpoint_persisted"] is True
    assert terminal["success_marker_allowed"] is False
    assert terminal["promotion_result_allowed"] is False
    assert "snapshot_validation_receipt" not in result
    assert "snapshot_validation_prepare_receipt" not in result


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda result: result["cancel_observation_receipt"].update(
                {"unexpected": True}
            ),
            "cancellation observation receipt fields mismatch",
        ),
        (
            lambda result: result["cancel_terminal_receipt"].update(
                {"snapshot_frontier": 3}
            ),
            "cancellation terminal receipt snapshot frontier mismatch",
        ),
        (
            lambda result: result["cancel_terminal_receipt"].update(
                {"cancellation_checkpoint_persisted": False}
            ),
            "cancellation terminal receipt checkpoint persistence mismatch",
        ),
    ],
)
def test_phase4_cancelled_result_rejects_non_authoritative_receipts(
    monkeypatch: pytest.MonkeyPatch,
    mutate: Any,
    message: str,
) -> None:
    with pytest.raises(NativeRunControllerError, match=message):
        _run_phase4_cancellation(monkeypatch, receipt_mutator=mutate)
