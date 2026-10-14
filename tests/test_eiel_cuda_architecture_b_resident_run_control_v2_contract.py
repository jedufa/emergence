from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HEADER = ROOT / "engine" / "perf" / "cuda_architecture_b_resident_run_control_v2.hpp"


def _braced_body(source: str, opening_brace: int) -> str:
    depth = 1
    for index in range(opening_brace + 1, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[opening_brace + 1 : index]
    raise AssertionError("unbalanced C++ braces")


def _definition_body(source: str, name: str) -> str:
    match = re.search(
        rf"\b{re.escape(name)}\s*\([^;{{}}]*\)\s*(?:const\s*)?\{{",
        source,
    )
    assert match is not None, f"missing C++ definition {name}"
    return _braced_body(source, match.end() - 1)


def _named_initializer_literals(source: str, name: str) -> tuple[str, ...]:
    match = re.search(rf"\b{re.escape(name)}\b[^;{{}}]*(?:=\s*)?\{{", source)
    assert match is not None, f"missing initializer {name}"
    body = _braced_body(source, match.end() - 1)
    return tuple(re.findall(r'"([^"\\]*(?:\\.[^"\\]*)*)"', body))


def test_header_is_standard_library_only_and_has_no_callback_api() -> None:
    source = HEADER.read_text(encoding="utf-8")
    lowered = source.lower()

    assert source.startswith("#pragma once\n")
    assert not re.search(r'^#include\s+"', source, flags=re.MULTILINE)
    for prohibited in (
        "<torch/",
        "torch::",
        "at::tensor",
        "pybind",
        "py::",
        "python.h",
        "pyobject",
        "std::function",
        "function<",
    ):
        assert prohibited not in lowered

    assert "python_callback_entry_points = 0" in source
    assert "class RunControlV2 final" in source


def test_cancel_request_has_exact_fields_and_closed_reasons() -> None:
    source = HEADER.read_text(encoding="utf-8")

    assert _named_initializer_literals(source, "kCancelRequestFieldsV2") == (
        "api_version",
        "request_id",
        "expected_controller_id",
        "expected_owner_epoch",
        "reason",
    )
    assert _named_initializer_literals(source, "kCancelReasonsV2") == (
        "operator_requested",
        "signal_received",
        "supervisor_requested",
    )
    assert "fields.size() != kCancelRequestFieldsV2.size()" in source
    assert "std::count(fields.begin(), fields.end(), required) == 1" in source
    assert "request.api_version != kCancelRequestApiVersionV2" in source
    assert "cancel_reason_is_supported_v2(request.reason)" in source

    request_type = re.search(
        r"struct CancelRequestV2\s*\{(?P<body>.*?)\};",
        source,
        flags=re.DOTALL,
    )
    assert request_type is not None
    declared_fields = re.findall(
        r"\b(?:std::int64_t|std::string)\s+([a-z_]+)(?:\s*=.*)?;",
        request_type.group("body"),
    )
    expected_fields = _named_initializer_literals(source, "kCancelRequestFieldsV2")
    assert declared_fields == list(expected_fields)


def test_identity_is_unique_and_start_is_single_use() -> None:
    source = HEADER.read_text(encoding="utf-8")
    start = _definition_body(source, "start_run")

    assert "inline static std::atomic<std::uint64_t> identity_sequence_v2_{1}" in source
    assert "make_unique_run_id_v2()" in source
    assert "next_identity_sequence_v2()" in source
    assert '"archb-cuda-controller-v2-"' in source
    assert "RunControlV2(const RunControlV2&) = delete" in source
    assert "RunControlV2(RunControlV2&&) = delete" in source

    assert "std::lock_guard<std::mutex> lock(mutex_)" in start
    assert "if (run_started_)" in start
    assert "run_started_ = true" in start
    assert "generation_transaction_open_ = true" in start
    assert "status_ = RunStatusV2::kGenerationStart" in start
    assert "requested_generations < 1" in start
    assert "target_next_generation_ = start_generation + requested_generations" in start
    assert source.count("run_started_ = false") == 1


def test_control_window_is_tick_zero_only_and_bounded_to_10ms() -> None:
    source = HEADER.read_text(encoding="utf-8")
    window = _definition_body(source, "inter_tick_control_point")

    assert "kInterTickControlWaitTimeoutMsV2 = 10" in source
    assert "completed_tick != 0" in window
    assert "next_tick != 1" in window
    assert "!generation_transaction_open_" in window
    assert "last_control_window_generation_ == generation" in window
    assert window.index("inter_tick_control_window_open_ = true") < window.index(
        "condition_.wait_for"
    )
    assert "std::chrono::milliseconds(kInterTickControlWaitTimeoutMsV2)" in window
    assert "cancel_requested_ && first_cancel_request_receipt_ready_" in window
    assert window.index("condition_.wait_for") < window.index(
        "inter_tick_control_window_open_ = false"
    )
    assert "status_ = RunStatusV2::kTickRunning" in window


def test_request_cancel_is_thread_safe_idempotent_and_metadata_only() -> None:
    source = HEADER.read_text(encoding="utf-8")
    request = _definition_body(source, "request_cancel")

    assert request.index("validate_cancel_request_v2(request)") < request.index(
        "std::lock_guard<std::mutex> lock(mutex_)"
    )
    dispositions = (
        '"identity_mismatch"',
        '"already_terminal"',
        '"already_requested"',
        '"not_running"',
        '"control_point_unavailable"',
        '"accepted"',
    )
    positions = [request.index(disposition) for disposition in dispositions]
    assert positions == sorted(positions)

    assert "!generation_transaction_open_" in request
    assert "status_ != RunStatusV2::kInterTickControl" not in request
    assert "!inter_tick_control_window_open_" not in request
    assert "inter_tick_completed_tick_ != 0" not in request
    assert "inter_tick_next_tick_ != 1" not in request
    assert "first_cancel_request_id_ = request.request_id" in request
    assert "first_cancel_reason_ = request.reason" in request
    assert "cancel_sequence_ += 1" in request
    assert "cancellation_arrival_coordinate_locked_v2()" in request
    assert "first_cancel_request_receipt_ready_ = true" in request

    lowered = request.lower()
    for prohibited in (
        "tensor",
        "trajectory",
        "generation_transaction_open_ =",
        "active_generation_ =",
        "status_ =",
        "sealed_ =",
    ):
        assert prohibited not in lowered


def test_failure_precedes_boundary_cancellation_and_seals_terminal_state() -> None:
    source = HEADER.read_text(encoding="utf-8")
    boundary = _definition_body(source, "consume_cancellation_at_boundary")
    seal_cancelled = _definition_body(source, "seal_cancelled_boundary_locked_v2")
    failure = _definition_body(source, "publish_failure")
    seal_failure = _definition_body(source, "seal_failure_locked_v2")

    assert boundary.index("failure_latched_") < boundary.index("sealed_")
    assert boundary.index("failure_latched_") < boundary.index("cancel_requested_")
    assert "status_ = RunStatusV2::kCancelled" in seal_cancelled
    assert "sealed_ = true" in seal_cancelled
    assert "cancel_consumed_ = true" in seal_cancelled

    assert "seal_failure_locked_v2(failure)" in failure
    assert "failure_latched_ = true" in seal_failure
    assert "inter_tick_control_window_open_ = false" in seal_failure
    assert "generation_transaction_open_ = false" in seal_failure
    assert "status_ = RunStatusV2::kFailed" in seal_failure
    assert "sealed_ = true" in seal_failure
    assert "cancel_consumed_ = true" not in seal_failure
    assert "terminal_status_changed_to_failed = true" in failure


def test_boundary_consumption_records_safe_coordinate_and_receipts() -> None:
    source = HEADER.read_text(encoding="utf-8")
    boundary = _definition_body(source, "consume_cancellation_at_boundary")
    seal_cancelled = _definition_body(source, "seal_cancelled_boundary_locked_v2")

    for field in (
        "completed_generation",
        "next_generation",
        "boundary_ordinal",
        "trajectory_commit_frontier",
        "snapshot_frontier",
        "output_ack_frontier",
        "snapshot_sequence",
        "snapshot_id",
        "snapshot_digest_sha256",
        "cancellation_checkpoint_requested",
        "cancellation_checkpoint_persisted",
    ):
        assert field in source

    assert "validate_boundary_locked_v2(boundary)" in boundary
    assert '"awaiting_cancellation_checkpoint"' in boundary
    assert "validate_durable_boundary_settlement_locked_v2(boundary)" in boundary
    assert "seal_cancelled_boundary_locked_v2(boundary)" in boundary
    assert "observation.request_control_coordinate" in seal_cancelled
    assert "observation.boundary = boundary" in seal_cancelled
    assert "boundary_cursor_v2(boundary.next_generation)" in seal_cancelled
    assert "observation.final_safe_cursor" in seal_cancelled
    assert "terminal.final_boundary = boundary" in seal_cancelled
    assert "terminal.cancellation = observation" in seal_cancelled
    assert "final_result_ = terminal" in seal_cancelled


def test_cancelled_terminal_waits_for_exact_durable_checkpoint() -> None:
    source = HEADER.read_text(encoding="utf-8")
    consume = _definition_body(source, "consume_cancellation_at_boundary")
    acknowledge = _definition_body(
        source, "validate_cancellation_checkpoint_ack_locked_v2"
    )
    durable = _definition_body(source, "validate_durable_boundary_settlement_locked_v2")

    assert consume.index("awaiting_cancellation_checkpoint_ = true") < consume.index(
        '"awaiting_cancellation_checkpoint"'
    )
    assert "!boundary.cancellation_checkpoint_requested" in consume
    assert "!boundary.cancellation_checkpoint_persisted" in consume
    assert "generation_transaction_open_ || !last_boundary_" in acknowledge
    assert "changed boundary identity" in acknowledge
    assert (
        "boundary.snapshot_frontier != boundary.trajectory_commit_frontier" in durable
    )
    assert (
        "boundary.output_ack_frontier != boundary.trajectory_commit_frontier" in durable
    )
    assert "boundary.snapshot_sequence < 0" in durable
    assert "boundary.snapshot_id.empty()" in durable
    assert "boundary.snapshot_digest_sha256.empty()" in durable

    public_section, private_section = source.split("  private:", maxsplit=1)
    definition = "BoundaryResolutionReceiptV2 seal_cancelled_boundary_locked_v2"
    assert definition not in public_section
    assert definition in private_section


def test_cancellation_ack_is_identity_bound_and_fails_closed() -> None:
    source = HEADER.read_text(encoding="utf-8")
    acknowledge = _definition_body(source, "acknowledge_cancellation_checkpoint")

    assert "expected_controller_id != identity_.controller_id" in acknowledge
    assert "expected_owner_epoch != identity_.owner_epoch" in acknowledge
    assert '"CANCELLATION_CHECKPOINT_IDENTITY_MISMATCH"' in acknowledge
    assert '"CANCELLATION_CHECKPOINT_ACK_FAILED"' in acknowledge
    assert "seal_failure_locked_v2" in acknowledge
    assert "validate_cancellation_checkpoint_ack_locked_v2(boundary)" in acknowledge


def test_final_completion_waits_for_target_bound_output_ack() -> None:
    source = HEADER.read_text(encoding="utf-8")
    consume = _definition_body(source, "consume_cancellation_at_boundary")
    complete = _definition_body(source, "acknowledge_run_complete")
    validate = _definition_body(source, "validate_final_output_ack_locked_v2")

    assert '"awaiting_final_output_ack"' in consume
    assert "boundary.next_generation == target_next_generation_" in consume
    assert "!awaiting_final_output_ack_" in complete
    assert "expected_controller_id != identity_.controller_id" in complete
    assert "validate_final_output_ack_locked_v2(acknowledged_boundary)" in complete
    assert '"RUN_COMPLETE_ACK_FAILED"' in complete
    assert "seal_failure_locked_v2" in complete
    assert "return *final_result_" in complete
    assert "boundary.next_generation != target_next_generation_" in validate
    assert "validate_durable_boundary_settlement_locked_v2(boundary)" in validate


def test_loop_can_observe_only_accepted_same_generation_cancellation() -> None:
    source = HEADER.read_text(encoding="utf-8")
    pending = _definition_body(source, "cancellation_pending_for_generation")

    for required in (
        "run_started_",
        "!sealed_",
        "!failure_latched_",
        "generation_transaction_open_",
        "active_generation_ == generation",
        "cancel_requested_",
        "!cancel_consumed_",
        "accepted_cancel_coordinate_.has_value()",
        "accepted_cancel_coordinate_->generation == generation",
    ):
        assert required in pending


def test_manifest_and_final_result_are_locked_complete_snapshots() -> None:
    source = HEADER.read_text(encoding="utf-8")
    manifest = _definition_body(source, "manifest")
    final_result = _definition_body(source, "final_result")

    for receipt_type in (
        "StartReceiptV2",
        "GenerationStartReceiptV2",
        "ControlWindowReceiptV2",
        "CancelRequestReceiptV2",
        "CancellationObservationReceiptV2",
        "FailureReceiptV2",
        "TerminalReceiptV2",
        "BoundaryResolutionReceiptV2",
        "RunControlManifestV2",
    ):
        assert f"struct {receipt_type}" in source

    assert "std::lock_guard<std::mutex> lock(mutex_)" in manifest
    assert "out.inter_tick_control_window_open" in manifest
    assert "out.inter_tick_control_coordinate" in manifest
    assert "out.cancel_requested" in manifest
    assert "out.cancel_consumed" in manifest
    assert "out.awaiting_cancellation_checkpoint" in manifest
    assert "out.failure_latched" in manifest
    assert "out.last_boundary" in manifest
    assert "out.final_result" in manifest
    assert "std::lock_guard<std::mutex> lock(mutex_)" in final_result
    assert "return final_result_" in final_result
