from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

import native_runtime.run_controller as run_controller_module
from native_runtime.run_controller import (
    NATIVE_CHECKPOINT_ON_CANCEL_CAPABILITY,
    NATIVE_CONTROLLER_VERSION,
    NATIVE_CONTROLLER_PROTOCOL_SCHEMA_VERSION,
    NATIVE_CONTROLLER_PROTOCOL_VERSION,
    NATIVE_PHASE4_REQUIRED_CAPABILITIES,
    NATIVE_RESULT_SCHEMA_VERSION,
    NATIVE_SNAPSHOT_VALIDATION_RECEIPT_SCHEMA_VERSION,
    TIER4_JUVENILE_VULNERABILITY_UNSUPPORTED_MESSAGE,
    TIER4_JUVENILE_VULNERABILITY_UNSUPPORTED_REASON,
    NativeRunController,
    NativeRunControllerError,
)


RUN_ID = "run-probe"
CONTROLLER_ID = "controller-probe"
OWNER_EPOCH = 17
SOURCE_PROJECTION_DIGEST = "a" * 64
OTHER_SOURCE_PROJECTION_DIGEST = "b" * 64


def test_exact_value_comparison_accepts_only_matching_tensor_nan_positions() -> None:
    baseline = torch.tensor([1.0, float("nan"), -2.0], dtype=torch.float64)

    assert run_controller_module._exact_values_equal(baseline, baseline.clone())
    assert not run_controller_module._exact_values_equal(
        baseline,
        torch.tensor([1.0, -2.0, float("nan")], dtype=torch.float64),
    )
    assert not run_controller_module._exact_values_equal(
        baseline,
        torch.tensor([1.0, float("nan"), -3.0], dtype=torch.float64),
    )


def _snapshot_validation_receipt(**updates: Any) -> dict[str, Any]:
    receipt = {
        "schema_version": NATIVE_SNAPSHOT_VALIDATION_RECEIPT_SCHEMA_VERSION,
        "run_id": RUN_ID,
        "controller_id": CONTROLLER_ID,
        "owner_epoch": OWNER_EPOCH,
        "acknowledged": True,
        "disposition": "accepted",
        "terminal_status": "run_complete",
        "trajectory_commit_frontier": 7,
        "snapshot_frontier": 7,
        "final_snapshot_sequence": 1,
        "validated_source_projection_digest": SOURCE_PROJECTION_DIGEST,
        "success_marker_allowed": True,
    }
    receipt.update(updates)
    return receipt


class _NativeControllerProbe:
    def __init__(
        self,
        result: dict[str, Any] | None = None,
        *,
        snapshot_failure_result: dict[str, Any] | None = None,
        run_id: str = RUN_ID,
        controller_id: str = CONTROLLER_ID,
        owner_epoch: int = OWNER_EPOCH,
        receipt_updates: dict[str, Any] | None = None,
    ) -> None:
        self.calls: list[tuple[int, int, int, int]] = []
        self.result = result
        self.snapshot_failure_result = snapshot_failure_result
        self.snapshot_validation_acknowledgements = 0
        self.snapshot_validation_digests: list[str] = []
        self.snapshot_validation_failures: list[str] = []
        self.run_id = run_id
        self.controller_id = controller_id
        self.owner_epoch = owner_epoch
        self.receipt_updates = dict(receipt_updates or {})
        self.status = "initialized"
        self.run_started = False
        self.sealed = False

    def controller_manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "eiel_native_architecture_b_controller_manifest_v1",
            "run_id": self.run_id,
            "controller_id": self.controller_id,
            "owner_epoch": self.owner_epoch,
            "status": self.status,
            "run_started": self.run_started,
            "sealed": self.sealed,
            "native_owner_private": True,
            "python_callback_entry_points": 0,
        }

    def run_generations(
        self,
        start_generation: int,
        generation_count: int,
        steps_per_generation: int,
        max_births: int,
    ) -> dict[str, Any]:
        self.calls.append(
            (
                start_generation,
                generation_count,
                steps_per_generation,
                max_births,
            )
        )
        self.run_started = True
        if self.result is None:
            raise AssertionError("native trajectory dispatch was not expected")
        self.status = str(self.result["terminal_status"])
        self.sealed = self.status in {"run_complete", "failed", "cancelled"}
        return self.result

    def acknowledge_snapshot_validation(
        self, source_projection_digest: str
    ) -> dict[str, Any]:
        assert self.status == "awaiting_snapshot_validation"
        assert self.result is not None
        self.snapshot_validation_acknowledgements += 1
        self.snapshot_validation_digests.append(source_projection_digest)
        self.result["terminal_status"] = "run_complete"
        self.result["snapshot_frontier"] = self.result["trajectory_commit_frontier"]
        self.result["success_marker_allowed"] = True
        self.status = "run_complete"
        self.sealed = True
        return _snapshot_validation_receipt(
            **{
                "run_id": self.run_id,
                "controller_id": self.controller_id,
                "owner_epoch": self.owner_epoch,
                "trajectory_commit_frontier": self.result[
                    "trajectory_commit_frontier"
                ],
                "snapshot_frontier": self.result["snapshot_frontier"],
                "final_snapshot_sequence": self.result["final_snapshot_sequence"],
                "validated_source_projection_digest": source_projection_digest,
                **self.receipt_updates,
            }
        )

    def fail_snapshot_validation(self, message: str) -> dict[str, Any]:
        assert self.status == "awaiting_snapshot_validation"
        assert self.snapshot_failure_result is not None
        self.snapshot_validation_failures.append(message)
        self.result = self.snapshot_failure_result
        self.status = "failed"
        self.sealed = True
        return self.result

    def final_result(self) -> dict[str, Any]:
        if not self.sealed:
            return {"terminal": False, "status": self.status}
        assert self.result is not None
        return self.result


class _PersistenceFailureProbe(_NativeControllerProbe):
    """Native-owner fake for concurrent failure ordering in mailbox mode."""

    def __init__(self, result: dict[str, Any]) -> None:
        super().__init__(result)
        self.persistence_calls: list[tuple[int, int, int, int, dict[str, Any]]] = []

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

    def run_generations(
        self,
        start_generation: int,
        generation_count: int,
        steps_per_generation: int,
        max_births: int,
        schedule: dict[str, Any],
    ) -> dict[str, Any]:
        self.persistence_calls.append(
            (
                start_generation,
                generation_count,
                steps_per_generation,
                max_births,
                dict(schedule),
            )
        )
        self.run_started = True
        self.status = "failed"
        self.sealed = True
        assert self.result is not None
        return self.result


def _controller(
    native: _NativeControllerProbe,
    *,
    steps_per_generation: int = 4,
    persistence_mailbox_required: bool = False,
) -> NativeRunController:
    return NativeRunController(
        native,
        start_generation=7,
        default_steps_per_generation=steps_per_generation,
        default_max_births=0,
        safe_default_decision={"supported": True},
        extension_path="/not-loaded-in-this-test.so",
        run_id=RUN_ID,
        persistence_mailbox_required=persistence_mailbox_required,
        phase4_capabilities=(
            NATIVE_PHASE4_REQUIRED_CAPABILITIES
            | frozenset({NATIVE_CHECKPOINT_ON_CANCEL_CAPABILITY})
            if persistence_mailbox_required
            else frozenset()
        ),
    )


def _failed_native_result(
    *,
    error_code: str,
    message: str,
    completed_generations: int = 0,
    trajectory_commit_frontier: int = 6,
    snapshot_frontier: int = 6,
) -> dict[str, Any]:
    return {
        "schema_version": NATIVE_RESULT_SCHEMA_VERSION,
        "decision": "ARCHB-CPU-D/FIRST-AUTHORITATIVE-OWNERSHIP-BLOCKER",
        "run_id": RUN_ID,
        "controller_id": CONTROLLER_ID,
        "owner_epoch": OWNER_EPOCH,
        "executed": False,
        "native_owner": True,
        "hidden_fallback": False,
        "runtime_jit": False,
        "terminal_status": "failed",
        "start_generation": 7,
        "requested_generations": 1,
        "completed_generations": completed_generations,
        "trajectory_commit_frontier": trajectory_commit_frontier,
        "snapshot_frontier": snapshot_frontier,
        "active_generation": 7 + completed_generations,
        "controller_invocations": 1,
        "trajectory_dispatch_count": 1,
        "python_generation_dispatches": 0,
        "boundaries": [],
        "root_failure": {
            "error_sequence": 1,
            "error_code": error_code,
            "origin": "FlatCpuArchitectureBControllerV1",
            "run_id": RUN_ID,
            "controller_id": CONTROLLER_ID,
            "owner_epoch": OWNER_EPOCH,
            "active_generation": 7 + completed_generations,
            "message": message,
            "owner_integrity": "suspect_and_sealed",
            "retry_disposition": (
                "new_owner_from_last_durable_complete_boundary_only"
            ),
        },
        "success_marker_allowed": False,
    }


def _assert_failure_payload_survives(
    actual: dict[str, Any],
    expected: dict[str, Any],
) -> None:
    assert actual == expected
    assert actual["terminal_status"] == "failed"
    assert actual["root_failure"] == expected["root_failure"]
    assert actual["trajectory_commit_frontier"] == expected[
        "trajectory_commit_frontier"
    ]
    assert actual["snapshot_frontier"] == expected["snapshot_frontier"]
    assert actual["success_marker_allowed"] is False


@pytest.mark.parametrize(
    ("generation_count", "kwargs", "message"),
    [
        (0, {}, "generation_count must be at least 1"),
        (-1, {}, "generation_count must be at least 1"),
        (True, {}, "generation_count must be an integer"),
        (1.0, {}, "generation_count must be an integer"),
        (1, {"steps_per_generation": 0}, "steps_per_generation must be at least 1"),
        (1, {"max_births": -1}, "max_births must be at least 0"),
    ],
)
def test_run_many_rejects_invalid_arguments_before_dispatch(
    generation_count: Any,
    kwargs: dict[str, Any],
    message: str,
) -> None:
    native = _NativeControllerProbe()
    controller = _controller(native)

    with pytest.raises(NativeRunControllerError) as exc_info:
        controller.run_many(generation_count, **kwargs)

    assert message in str(exc_info.value)
    assert native.calls == []
    assert controller.manifest()["trajectory_dispatch_count"] == 0
    assert controller.final_result() == {"terminal": False, "status": "initialized"}


def test_native_failure_after_trajectory_entry_is_returned_and_retained() -> None:
    root_message = (
        "Architecture B steps_per_generation does not match "
        "the imported run configuration"
    )
    failed_result = _failed_native_result(
        error_code="native_generation_transaction_failed",
        message=root_message,
    )
    native = _NativeControllerProbe(failed_result)
    controller = _controller(native, steps_per_generation=4)

    returned = controller.run_many(1, steps_per_generation=5)

    _assert_failure_payload_survives(returned, failed_result)
    assert native.calls == [(7, 1, 5, 0)]
    manifest = controller.manifest()
    assert manifest["status"] == "failed"
    assert manifest["trajectory_dispatch_count"] == 1
    assert manifest["native_controller"]["status"] == "failed"
    assert manifest["native_controller"]["sealed"] is True

    retained = controller.final_result()
    _assert_failure_payload_survives(retained, failed_result)
    assert retained == returned
    assert native.final_result() == retained

    with pytest.raises(NativeRunControllerError, match="single-use"):
        controller.run_many(1, steps_per_generation=4)

    assert native.calls == [(7, 1, 5, 0)]


def test_failed_result_requires_native_first_error_wins_sequence() -> None:
    failed_result = _failed_native_result(
        error_code="persistence_failed",
        message="the original persistence failure",
    )
    failed_result["root_failure"]["error_sequence"] = 2
    controller = _controller(_NativeControllerProbe())

    with pytest.raises(NativeRunControllerError, match="not first-error-wins"):
        controller._normalize_failed_result(failed_result)


def test_persistence_cancellation_failure_does_not_mask_native_root_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root_message = "first durable persistence failure"
    failed_result = _failed_native_result(
        error_code="snapshot_persistence_failed",
        message=root_message,
        completed_generations=1,
        trajectory_commit_frontier=7,
        snapshot_frontier=6,
    )
    native = _PersistenceFailureProbe(failed_result)
    controller = _controller(native, persistence_mailbox_required=True)

    class _CancellationFailureLoop:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            pass

        def run(self) -> dict[str, Any]:
            raise RuntimeError("cancellation-side persistence teardown failed")

    monkeypatch.setattr(
        run_controller_module,
        "ArchitectureBPersistenceLoop",
        _CancellationFailureLoop,
    )

    with pytest.raises(NativeRunControllerError, match=root_message):
        controller.run_with_persistence(
            SimpleNamespace(run_dir="/fake"),
            1,
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

    final = controller.final_result()
    _assert_failure_payload_survives(final, failed_result)
    assert native.persistence_calls[0][:4] == (7, 1, 4, 0)
    assert "snapshot_validation_receipt" not in final
    assert "public_output_finalization" not in final


@pytest.mark.parametrize(
    ("stage", "terminal_status", "success_only_field"),
    [
        ("persistence", "failed", "snapshot_validation_receipt"),
        ("output", "failed", "public_output_finalized"),
        ("finalization", "cancelled", "public_output_finalization"),
    ],
)
def test_failure_terminals_reject_success_markers(
    stage: str,
    terminal_status: str,
    success_only_field: str,
) -> None:
    native = _NativeControllerProbe()
    controller = _controller(native)
    terminal = {
        "terminal_status": terminal_status,
        "success_marker_allowed": False,
        success_only_field: {"stage": stage},
    }
    native.result = terminal
    native.status = terminal_status
    native.sealed = True
    controller._final = terminal

    with pytest.raises(
        NativeRunControllerError,
        match=f"{terminal_status} result contains success-only field",
    ):
        controller.final_result()


@pytest.mark.parametrize(
    ("native_kwargs", "message"),
    [
        ({"run_id": "wrong-run"}, "run identity mismatch"),
        ({"controller_id": ""}, "controller_id must be a non-empty identifier"),
        ({"owner_epoch": 0}, "owner_epoch must be at least 1"),
    ],
)
def test_controller_initialization_rejects_invalid_native_identity(
    native_kwargs: dict[str, Any],
    message: str,
) -> None:
    native = _NativeControllerProbe(**native_kwargs)

    with pytest.raises(NativeRunControllerError, match=message):
        _controller(native)


def test_post_run_snapshot_validation_failure_agrees_with_native_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot_message = "post-run snapshot digest validation failed"
    awaiting_result = {
        "schema_version": NATIVE_RESULT_SCHEMA_VERSION,
        "terminal_status": "awaiting_snapshot_validation",
        "trajectory_commit_frontier": 7,
        "snapshot_frontier": 6,
        "final_snapshot_sequence": 1,
        "success_marker_allowed": False,
    }
    failed_result = _failed_native_result(
        error_code="snapshot_validation_failed",
        message=snapshot_message,
        completed_generations=1,
        trajectory_commit_frontier=7,
        snapshot_frontier=6,
    )
    native = _NativeControllerProbe(
        awaiting_result,
        snapshot_failure_result=failed_result,
    )
    controller = _controller(native)

    def fail_post_run_validation(
        raw: Any,
        *,
        requested: int,
        expected_terminal_status: str,
    ) -> dict[str, Any]:
        assert raw == awaiting_result
        assert requested == 1
        assert expected_terminal_status == "awaiting_snapshot_validation"
        raise NativeRunControllerError(snapshot_message)

    monkeypatch.setattr(
        controller,
        "_normalize_and_validate_result",
        fail_post_run_validation,
    )

    facade_result = controller.run_many(1)
    native_result = native.final_result()

    assert len(native.snapshot_validation_failures) == 1
    assert snapshot_message in native.snapshot_validation_failures[0]
    assert native.snapshot_validation_acknowledgements == 0
    _assert_failure_payload_survives(facade_result, failed_result)
    _assert_failure_payload_survives(native_result, failed_result)
    assert controller.final_result() == native_result == facade_result
    assert controller.manifest()["status"] == "failed"


def test_successful_controller_rejects_a_second_run_entry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    awaiting_result = {
        "schema_version": NATIVE_RESULT_SCHEMA_VERSION,
        "terminal_status": "awaiting_snapshot_validation",
        "trajectory_commit_frontier": 7,
        "snapshot_frontier": 6,
        "success_marker_allowed": False,
        "final_snapshot_sequence": 1,
        "final_checkpoint_projection_digest_manifests": {
            "source_projection": {"digest": SOURCE_PROJECTION_DIGEST},
        },
        "final_checkpoint_projection": {"projection": "stable-probe"},
    }
    completed_result = {
        **awaiting_result,
        "terminal_status": "run_complete",
        "snapshot_frontier": 7,
        "success_marker_allowed": True,
        "snapshot_validation_receipt": _snapshot_validation_receipt(),
    }
    native = _NativeControllerProbe(awaiting_result)
    controller = _controller(native)
    monkeypatch.setattr(
        controller,
        "_normalize_and_validate_result",
        lambda raw, *, requested, expected_terminal_status: dict(raw),
    )
    monkeypatch.setattr(
        run_controller_module,
        "_projection_digest_manifests",
        lambda projection: {
            "source_projection": {"digest": SOURCE_PROJECTION_DIGEST}
        },
    )

    assert controller.run_many(1) == completed_result
    assert native.snapshot_validation_acknowledgements == 1
    assert native.snapshot_validation_digests == [SOURCE_PROJECTION_DIGEST]
    assert native.snapshot_validation_failures == []
    assert "boundaries" not in completed_result["snapshot_validation_receipt"]
    with pytest.raises(NativeRunControllerError, match="single-use"):
        controller.run_many(1)

    assert native.calls == [(7, 1, 4, 0)]
    assert controller.final_result() == completed_result
    assert controller.manifest()["status"] == "run_complete"


@pytest.mark.parametrize(
    ("receipt_updates", "message"),
    [
        ({"run_id": "wrong-run"}, "run identity mismatch"),
        ({"controller_id": "wrong-controller"}, "controller identity mismatch"),
        ({"owner_epoch": OWNER_EPOCH + 1}, "owner epoch mismatch"),
        ({"terminal_status": "cancelled"}, "not terminal run_complete"),
        ({"success_marker_allowed": False}, "success_marker_allowed must be True"),
        ({"unexpected_receipt_field": True}, "fields mismatch"),
        ({"snapshot_frontier": 6}, "frontier mismatch"),
        ({"trajectory_commit_frontier": 6}, "frontier mismatch"),
        ({"final_snapshot_sequence": 2}, "sequence mismatch"),
        (
            {"validated_source_projection_digest": "A" * 64},
            "must be exactly 64 lowercase hexadecimal characters",
        ),
        (
            {"validated_source_projection_digest": OTHER_SOURCE_PROJECTION_DIGEST},
            "source-projection digest mismatch",
        ),
    ],
)
def test_snapshot_validation_receipt_enforces_identity_and_frontiers(
    receipt_updates: dict[str, Any],
    message: str,
) -> None:
    controller = _controller(_NativeControllerProbe())
    provisional = {
        "trajectory_commit_frontier": 7,
        "final_snapshot_sequence": 1,
        "final_checkpoint_projection_digest_manifests": {
            "source_projection": {"digest": SOURCE_PROJECTION_DIGEST}
        },
    }

    with pytest.raises(NativeRunControllerError, match=message):
        controller._normalize_snapshot_validation_receipt(
            _snapshot_validation_receipt(**receipt_updates),
            provisional=provisional,
        )


def test_provisional_result_rejects_final_authority_digest_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _controller(_NativeControllerProbe())
    raw_result = {
        "schema_version": NATIVE_RESULT_SCHEMA_VERSION,
        "run_id": RUN_ID,
        "controller_id": CONTROLLER_ID,
        "owner_epoch": OWNER_EPOCH,
        "terminal_status": "awaiting_snapshot_validation",
        "executed": True,
        "native_owner": True,
        "hidden_fallback": False,
        "runtime_jit": False,
        "completed_generations": 1,
        "start_generation": 7,
        "requested_generations": 1,
        "trajectory_commit_frontier": 7,
        "snapshot_frontier": 6,
        "next_generation": 8,
        "next_tick": 0,
        "controller_invocations": 1,
        "python_generation_dispatches": 0,
        "native_generation_transactions": 1,
        "turnover_commits": 1,
        "boundary_snapshot_count": 1,
        "projection_export_count": 2,
        "projection_tensor_bytes": 0,
        "final_snapshot_sequence": 1,
        "success_marker_allowed": False,
        "boundaries": [{}],
    }
    monkeypatch.setattr(
        controller,
        "_normalize_boundary",
        lambda raw, *, offset: {
            "post_turnover_checkpoint_projection": {"projection": "final"},
            "checkpoint_projection_digest_manifests": {
                "source_projection": {"digest": "boundary-digest"}
            },
            "tensor_manifest_boundary_bytes": {"total": 0},
        },
    )
    monkeypatch.setattr(
        run_controller_module,
        "_projection_digest_manifests",
        lambda projection: {
            "source_projection": {"digest": "different-final-digest"}
        },
    )

    with pytest.raises(
        NativeRunControllerError,
        match="final checkpoint projection does not match the last boundary",
    ):
        controller._normalize_and_validate_result(
            raw_result,
            requested=1,
            expected_terminal_status="awaiting_snapshot_validation",
        )


def _patch_factory_inputs(
    monkeypatch: pytest.MonkeyPatch,
    extension_path: Path,
) -> SimpleNamespace:
    decision = SimpleNamespace(
        extension_loaded=True,
        extension_runtime_jit=False,
        runtime_jit_prohibited=True,
        extension_path=str(extension_path),
        to_dict=lambda: {"supported": True},
    )
    projection = SimpleNamespace(
        tensors={},
        metadata={},
        pending_generation_free_agent_ids=[],
        pending_generation_free_slots=[],
    )
    monkeypatch.setattr(
        run_controller_module,
        "validate_native_runtime_supported",
        lambda runner: decision,
    )

    import perf.cxxbackbone10_resident_state_owner as resident_owner_module

    monkeypatch.setattr(
        resident_owner_module,
        "build_import_projection",
        lambda runner, scope, *, generation, tick: projection,
    )
    return SimpleNamespace(
        run_dir=extension_path.parent / "fallback-run",
        sim_cfg=SimpleNamespace(
            chronicle_run_id="",
            steps_per_generation=4,
            native_runtime_max_births_per_generation=0,
        )
    )


@pytest.mark.parametrize(
    ("configured_run_id", "expected_run_id"),
    [
        ("chronicle-configured-run", "chronicle-configured-run"),
        ("", "fallback-run"),
    ],
)
def test_from_runner_passes_logical_run_id_and_captures_native_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    configured_run_id: str,
    expected_run_id: str,
) -> None:
    extension_path = tmp_path / "architecture_b_candidate.so"
    extension_bytes = b"not a live extension"
    extension_path.write_bytes(extension_bytes)
    runner = _patch_factory_inputs(monkeypatch, extension_path)
    runner.sim_cfg.chronicle_run_id = configured_run_id
    constructed: list[tuple[dict[str, Any], dict[str, Any], str]] = []

    def native_controller_type(
        tensors: dict[str, Any],
        metadata: dict[str, Any],
        run_id: str,
    ) -> _NativeControllerProbe:
        constructed.append((tensors, metadata, run_id))
        return _NativeControllerProbe(run_id=run_id)

    monkeypatch.setattr(
        run_controller_module,
        "load_prebuilt_extension",
        lambda path: SimpleNamespace(
            EIEL_NATIVE_ARCHITECTURE_B_CONTROLLER_VERSION=NATIVE_CONTROLLER_VERSION,
            FlatCpuArchitectureBControllerV1=native_controller_type,
        ),
    )

    controller = NativeRunController.from_runner(
        runner,
        extension_path=extension_path,
        expected_extension_sha256=hashlib.sha256(extension_bytes).hexdigest(),
    )

    assert constructed == [
        (
            {},
            {"native_runtime_architecture_b_compact_reporting_only_v1": False},
            expected_run_id,
        )
    ]
    manifest = controller.manifest()
    assert manifest["run_id"] == expected_run_id
    assert manifest["controller_id"] == CONTROLLER_ID
    assert manifest["owner_epoch"] == OWNER_EPOCH


@pytest.mark.parametrize(
    ("tier4_enabled", "tier4_juvenile_steps"),
    [
        (True, 0),
        (False, 1),
    ],
)
def test_from_runner_rejects_tier4_before_native_owner_construction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tier4_enabled: bool,
    tier4_juvenile_steps: int,
) -> None:
    extension_path = tmp_path / "architecture_b_candidate.so"
    extension_bytes = b"not a live extension"
    extension_path.write_bytes(extension_bytes)
    runner = _patch_factory_inputs(monkeypatch, extension_path)
    runner.env_cfg = SimpleNamespace(
        tier4_juvenile_vulnerability_enable=tier4_enabled,
        tier4_juvenile_steps=tier4_juvenile_steps,
    )
    owner_constructions: list[str] = []

    def native_controller_type(
        tensors: dict[str, Any],
        metadata: dict[str, Any],
        run_id: str,
    ) -> _NativeControllerProbe:
        owner_constructions.append(run_id)
        return _NativeControllerProbe(run_id=run_id)

    monkeypatch.setattr(
        run_controller_module,
        "load_prebuilt_extension",
        lambda path: SimpleNamespace(
            EIEL_NATIVE_ARCHITECTURE_B_CONTROLLER_VERSION=NATIVE_CONTROLLER_VERSION,
            FlatCpuArchitectureBControllerV1=native_controller_type,
        ),
    )

    with pytest.raises(NativeRunControllerError) as exc_info:
        NativeRunController.from_runner(
            runner,
            extension_path=extension_path,
            expected_extension_sha256=hashlib.sha256(extension_bytes).hexdigest(),
        )

    assert str(exc_info.value) == (
        "EIEL-NATIVE-RUN-CONTROLLER-FAILED: "
        f"{TIER4_JUVENILE_VULNERABILITY_UNSUPPORTED_REASON}: "
        f"{TIER4_JUVENILE_VULNERABILITY_UNSUPPORTED_MESSAGE}"
    )
    assert owner_constructions == []


def test_from_runner_accepts_disabled_default_tier4(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extension_path = tmp_path / "architecture_b_candidate.so"
    extension_bytes = b"not a live extension"
    extension_path.write_bytes(extension_bytes)
    runner = _patch_factory_inputs(monkeypatch, extension_path)
    runner.env_cfg = SimpleNamespace(
        tier4_juvenile_vulnerability_enable=False,
        tier4_juvenile_steps=0,
    )
    owner_constructions: list[str] = []

    def native_controller_type(
        tensors: dict[str, Any],
        metadata: dict[str, Any],
        run_id: str,
    ) -> _NativeControllerProbe:
        owner_constructions.append(run_id)
        return _NativeControllerProbe(run_id=run_id)

    monkeypatch.setattr(
        run_controller_module,
        "load_prebuilt_extension",
        lambda path: SimpleNamespace(
            EIEL_NATIVE_ARCHITECTURE_B_CONTROLLER_VERSION=NATIVE_CONTROLLER_VERSION,
            FlatCpuArchitectureBControllerV1=native_controller_type,
        ),
    )

    controller = NativeRunController.from_runner(
        runner,
        extension_path=extension_path,
        expected_extension_sha256=hashlib.sha256(extension_bytes).hexdigest(),
    )

    assert owner_constructions == ["fallback-run"]
    assert controller.manifest()["status"] == "initialized"


def test_from_runner_rejects_extension_sha_mismatch_before_loading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extension_path = tmp_path / "architecture_b_candidate.so"
    extension_path.write_bytes(b"not a live extension")
    runner = _patch_factory_inputs(monkeypatch, extension_path)

    def unexpected_load(path: Path) -> Any:
        pytest.fail(f"extension loader reached for rejected digest: {path}")

    monkeypatch.setattr(
        run_controller_module,
        "load_prebuilt_extension",
        unexpected_load,
    )

    with pytest.raises(NativeRunControllerError, match="extension SHA-256 mismatch"):
        NativeRunController.from_runner(
            runner,
            extension_path=extension_path,
            expected_extension_sha256="0" * 64,
        )


def test_from_runner_rejects_extension_missing_native_controller(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extension_path = tmp_path / "architecture_b_candidate.so"
    extension_bytes = b"not a live extension"
    extension_path.write_bytes(extension_bytes)
    runner = _patch_factory_inputs(monkeypatch, extension_path)
    monkeypatch.setattr(
        run_controller_module,
        "load_prebuilt_extension",
        lambda path: SimpleNamespace(),
    )

    with pytest.raises(
        NativeRunControllerError,
        match="prebuilt extension is missing FlatCpuArchitectureBControllerV1",
    ):
        NativeRunController.from_runner(
            runner,
            extension_path=extension_path,
            expected_extension_sha256=hashlib.sha256(extension_bytes).hexdigest(),
        )
