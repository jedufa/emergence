#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <limits>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace eiel::cuda_architecture_b::resident_run_control_v2 {

inline constexpr std::int64_t kRunControlProtocolVersionV2 = 2;
inline constexpr std::int64_t kCancelRequestApiVersionV2 = 1;
inline constexpr std::int64_t kInterTickControlWaitTimeoutMsV2 = 10;

inline constexpr std::array<std::string_view, 5> kCancelRequestFieldsV2{
    "api_version",
    "request_id",
    "expected_controller_id",
    "expected_owner_epoch",
    "reason",
};

inline constexpr std::array<std::string_view, 3> kCancelReasonsV2{
    "operator_requested",
    "signal_received",
    "supervisor_requested",
};

inline bool cancel_request_fields_are_exact_v2(
    const std::vector<std::string>& fields
) {
    if (fields.size() != kCancelRequestFieldsV2.size()) {
        return false;
    }
    return std::all_of(
        kCancelRequestFieldsV2.begin(),
        kCancelRequestFieldsV2.end(),
        [&fields](const std::string_view required) {
            return std::count(fields.begin(), fields.end(), required) == 1;
        }
    );
}

inline bool cancel_reason_is_supported_v2(const std::string_view reason) {
    return std::find(
               kCancelReasonsV2.begin(),
               kCancelReasonsV2.end(),
               reason
           ) != kCancelReasonsV2.end();
}

enum class RunStatusV2 {
    kInitialized,
    kGenerationStart,
    kInterTickControl,
    kTickRunning,
    kGenerationComplete,
    kCancelled,
    kFailed,
    kRunComplete,
};

inline std::string status_name_v2(const RunStatusV2 status) {
    switch (status) {
        case RunStatusV2::kInitialized:
            return "initialized";
        case RunStatusV2::kGenerationStart:
            return "generation_start";
        case RunStatusV2::kInterTickControl:
            return "inter_tick_control";
        case RunStatusV2::kTickRunning:
            return "tick_running";
        case RunStatusV2::kGenerationComplete:
            return "generation_complete";
        case RunStatusV2::kCancelled:
            return "cancelled";
        case RunStatusV2::kFailed:
            return "failed";
        case RunStatusV2::kRunComplete:
            return "run_complete";
    }
    throw std::logic_error("unknown CUDA Architecture B run-control status");
}

struct RunIdentityV2 {
    std::string run_id;
    std::string controller_id;
    std::int64_t owner_epoch = 1;
};

struct CancelRequestV2 {
    std::int64_t api_version = kCancelRequestApiVersionV2;
    std::string request_id;
    std::string expected_controller_id;
    std::int64_t expected_owner_epoch = 1;
    std::string reason;
};

struct ControlCoordinateV2 {
    std::int64_t generation = -1;
    std::int64_t completed_tick = -1;
    std::int64_t next_tick = -1;
    std::uint64_t window_count = 0;
    bool generation_transaction_open = false;
};

struct BoundaryMetadataV2 {
    std::int64_t completed_generation = -1;
    std::int64_t next_generation = -1;
    std::int64_t boundary_ordinal = 0;
    std::int64_t trajectory_commit_frontier = -1;
    std::int64_t snapshot_frontier = -1;
    std::int64_t output_ack_frontier = -1;
    std::int64_t snapshot_sequence = -1;
    std::string snapshot_id;
    std::string snapshot_digest_sha256;
    bool cancellation_checkpoint_requested = false;
    bool cancellation_checkpoint_persisted = false;
};

struct FailureMetadataV2 {
    std::string error_code;
    std::string message;
    std::int64_t active_generation = -1;
    std::int64_t trajectory_commit_frontier = -1;
};

struct StartReceiptV2 {
    std::string schema_version =
        "eiel_cuda_architecture_b_run_control_start_receipt_v2";
    RunIdentityV2 identity;
    std::int64_t start_generation = -1;
    bool single_use = true;
    std::string status;
};

struct GenerationStartReceiptV2 {
    std::string schema_version =
        "eiel_cuda_architecture_b_generation_start_receipt_v2";
    RunIdentityV2 identity;
    std::int64_t generation = -1;
    bool generation_transaction_open = false;
    std::string status;
};

struct ControlWindowReceiptV2 {
    std::string schema_version =
        "eiel_cuda_architecture_b_inter_tick_control_window_receipt_v2";
    RunIdentityV2 identity;
    ControlCoordinateV2 coordinate;
    std::int64_t wait_timeout_ms = kInterTickControlWaitTimeoutMsV2;
    std::string close_disposition;
    bool cancel_requested = false;
    std::optional<std::string> first_cancel_request_id;
    std::optional<std::int64_t> cancel_sequence;
    std::string status_after_close;
};

struct CancelRequestReceiptV2 {
    std::string schema_version =
        "eiel_cuda_architecture_b_cancel_request_receipt_v2";
    std::string request_id;
    RunIdentityV2 identity;
    std::string disposition;
    std::optional<std::string> first_cancel_request_id;
    std::optional<std::int64_t> cancel_sequence;
    std::optional<ControlCoordinateV2> accepted_control_coordinate;
    std::string state_observed;
    bool terminal_status_changed = false;
};

struct CancellationObservationReceiptV2 {
    std::string schema_version =
        "eiel_cuda_architecture_b_cancellation_observation_receipt_v2";
    std::string request_id;
    std::string reason;
    RunIdentityV2 identity;
    std::int64_t cancel_sequence = 0;
    ControlCoordinateV2 request_control_coordinate;
    BoundaryMetadataV2 boundary;
    std::string request_observation_cursor;
    std::string final_safe_cursor;
    std::string terminal_status = "cancelled";
    bool success_marker_allowed = false;
    bool promotion_result_allowed = false;
};

struct FailureReceiptV2 {
    std::string schema_version =
        "eiel_cuda_architecture_b_failure_receipt_v2";
    RunIdentityV2 identity;
    std::string disposition;
    FailureMetadataV2 failure;
    bool accepted = false;
    bool terminal_status_changed_to_failed = false;
    std::string terminal_status;
};

struct TerminalReceiptV2 {
    std::string schema_version =
        "eiel_cuda_architecture_b_terminal_receipt_v2";
    RunIdentityV2 identity;
    std::string terminal_status;
    bool sealed = false;
    std::optional<BoundaryMetadataV2> final_boundary;
    std::optional<CancellationObservationReceiptV2> cancellation;
    std::optional<FailureMetadataV2> failure;
    bool success_marker_allowed = false;
    bool promotion_result_allowed = false;
};

struct BoundaryResolutionReceiptV2 {
    std::string schema_version =
        "eiel_cuda_architecture_b_boundary_resolution_receipt_v2";
    RunIdentityV2 identity;
    std::string disposition;
    std::optional<BoundaryMetadataV2> boundary;
    std::optional<CancellationObservationReceiptV2> cancellation;
    bool terminal = false;
    std::string terminal_status;
};

struct RunControlManifestV2 {
    std::string schema_version =
        "eiel_cuda_architecture_b_resident_run_control_manifest_v2";
    RunIdentityV2 identity;
    std::int64_t controller_protocol_version = kRunControlProtocolVersionV2;
    std::string controller_protocol_schema_version =
        "eiel_cuda_architecture_b_resident_run_control_protocol_v2";
    std::string status;
    bool run_started = false;
    bool sealed = false;
    bool native_owner_private = true;
    std::int64_t python_callback_entry_points = 0;
    std::int64_t inter_tick_control_wait_timeout_ms =
        kInterTickControlWaitTimeoutMsV2;
    bool generation_transaction_open = false;
    std::int64_t active_generation = -1;
    std::int64_t start_generation = -1;
    std::int64_t requested_generations = 0;
    std::int64_t target_next_generation = -1;
    bool checkpoint_on_cancel = true;
    bool final_output_ack_required = true;
    bool inter_tick_control_window_open = false;
    ControlCoordinateV2 inter_tick_control_coordinate;
    std::uint64_t inter_tick_control_window_count = 0;
    bool cancel_requested = false;
    bool cancel_consumed = false;
    bool awaiting_cancellation_checkpoint = false;
    bool awaiting_final_output_ack = false;
    std::optional<std::string> first_cancel_request_id;
    std::optional<std::string> first_cancel_reason;
    std::optional<std::int64_t> cancel_sequence;
    bool failure_latched = false;
    std::optional<BoundaryMetadataV2> last_boundary;
    std::optional<TerminalReceiptV2> final_result;
};

class RunControlV2 final {
  public:
    RunControlV2()
        : RunControlV2(make_unique_run_id_v2(), 1) {}

    explicit RunControlV2(
        std::string run_id,
        const std::int64_t owner_epoch = 1
    )
        : identity_(make_identity_v2(std::move(run_id), owner_epoch)) {}

    RunControlV2(const RunControlV2&) = delete;
    RunControlV2& operator=(const RunControlV2&) = delete;
    RunControlV2(RunControlV2&&) = delete;
    RunControlV2& operator=(RunControlV2&&) = delete;

    StartReceiptV2 start_run(
        const std::int64_t start_generation,
        const std::int64_t requested_generations
    ) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (run_started_) {
            throw std::logic_error("CUDA Architecture B RunControlV2 is single-use");
        }
        if (sealed_) {
            throw std::logic_error("cannot start a sealed CUDA run controller");
        }
        if (
            start_generation < 0
            || requested_generations < 1
            || start_generation > std::numeric_limits<std::int64_t>::max()
                - requested_generations
        ) {
            throw std::invalid_argument(
                "the run target must be one positive non-overflowing generation range"
            );
        }
        start_generation_ = start_generation;
        requested_generations_ = requested_generations;
        target_next_generation_ = start_generation + requested_generations;
        run_started_ = true;
        generation_transaction_open_ = true;
        active_generation_ = start_generation;
        status_ = RunStatusV2::kGenerationStart;
        return StartReceiptV2{
            "eiel_cuda_architecture_b_run_control_start_receipt_v2",
            identity_,
            start_generation,
            true,
            status_name_v2(status_),
        };
    }

    GenerationStartReceiptV2 begin_generation(
        const std::int64_t generation
    ) {
        std::lock_guard<std::mutex> lock(mutex_);
        require_live_started_locked_v2("begin_generation");
        if (failure_latched_) {
            throw std::logic_error("failure takes precedence over generation start");
        }
        if (generation_transaction_open_) {
            throw std::logic_error("a generation transaction is already open");
        }
        if (status_ != RunStatusV2::kGenerationComplete || !last_boundary_) {
            throw std::logic_error("the prior generation boundary is not complete");
        }
        if (generation != last_boundary_->next_generation) {
            throw std::invalid_argument(
                "generation does not match the prior boundary next_generation"
            );
        }
        if (generation >= target_next_generation_) {
            throw std::logic_error("the frozen run target has already been reached");
        }
        if (cancel_requested_ && !cancel_consumed_) {
            throw std::logic_error(
                "pending cancellation must be consumed at the completed boundary"
            );
        }
        generation_transaction_open_ = true;
        active_generation_ = generation;
        status_ = RunStatusV2::kGenerationStart;
        return GenerationStartReceiptV2{
            "eiel_cuda_architecture_b_generation_start_receipt_v2",
            identity_,
            generation,
            true,
            status_name_v2(status_),
        };
    }

    ControlWindowReceiptV2 inter_tick_control_point(
        const std::int64_t generation,
        const std::int64_t completed_tick,
        const std::int64_t next_tick
    ) {
        std::unique_lock<std::mutex> lock(mutex_);
        require_live_started_locked_v2("inter_tick_control_point");
        if (
            !generation_transaction_open_
            || generation != active_generation_
            || completed_tick != 0
            || next_tick != 1
        ) {
            throw std::invalid_argument(
                "the CUDA cancellation window must follow completed tick 0 "
                "inside the active generation transaction"
            );
        }
        if (
            status_ != RunStatusV2::kGenerationStart
            && status_ != RunStatusV2::kTickRunning
        ) {
            throw std::logic_error("the inter-tick control point is unavailable");
        }
        if (last_control_window_generation_ == generation) {
            throw std::logic_error(
                "the post-tick-0 control window may open only once per generation"
            );
        }

        inter_tick_control_window_open_ = true;
        inter_tick_control_generation_ = generation;
        inter_tick_completed_tick_ = completed_tick;
        inter_tick_next_tick_ = next_tick;
        inter_tick_control_window_count_ += 1;
        last_control_window_generation_ = generation;
        status_ = RunStatusV2::kInterTickControl;
        condition_.notify_all();

        condition_.wait_for(
            lock,
            std::chrono::milliseconds(kInterTickControlWaitTimeoutMsV2),
            [this]() {
                return failure_latched_
                    || sealed_
                    || (cancel_requested_ && first_cancel_request_receipt_ready_);
            }
        );

        inter_tick_control_window_open_ = false;
        std::string close_disposition = "timeout";
        if (failure_latched_) {
            close_disposition = "failed";
        } else if (sealed_) {
            close_disposition = "sealed";
        } else if (cancel_requested_ && first_cancel_request_receipt_ready_) {
            close_disposition = "cancel_requested";
        }
        if (!sealed_ && !failure_latched_) {
            status_ = RunStatusV2::kTickRunning;
        }
        condition_.notify_all();

        ControlWindowReceiptV2 receipt;
        receipt.identity = identity_;
        receipt.coordinate = current_control_coordinate_locked_v2();
        receipt.close_disposition = close_disposition;
        receipt.cancel_requested = cancel_requested_;
        if (cancel_requested_) {
            receipt.first_cancel_request_id = first_cancel_request_id_;
            receipt.cancel_sequence = cancel_sequence_;
        }
        receipt.status_after_close = status_name_v2(status_);
        return receipt;
    }

    CancelRequestReceiptV2 request_cancel(const CancelRequestV2& request) {
        validate_cancel_request_v2(request);

        std::lock_guard<std::mutex> lock(mutex_);
        CancelRequestReceiptV2 receipt;
        receipt.request_id = request.request_id;
        receipt.identity = identity_;
        receipt.terminal_status_changed = false;

        if (
            request.expected_controller_id != identity_.controller_id
            || request.expected_owner_epoch != identity_.owner_epoch
        ) {
            receipt.disposition = "identity_mismatch";
            receipt.state_observed = run_started_ ? "running" : "initialized";
        } else if (sealed_ || is_terminal_status_v2(status_)) {
            receipt.disposition = "already_terminal";
            receipt.state_observed = status_name_v2(status_);
        } else if (cancel_requested_) {
            receipt.disposition = "already_requested";
            receipt.state_observed = "running";
        } else if (!run_started_) {
            receipt.disposition = "not_running";
            receipt.state_observed = "initialized";
        } else if (!generation_transaction_open_) {
            receipt.disposition = "control_point_unavailable";
            receipt.state_observed = status_name_v2(status_);
        } else {
            first_cancel_request_id_ = request.request_id;
            first_cancel_reason_ = request.reason;
            cancel_sequence_ += 1;
            cancel_requested_ = true;
            accepted_cancel_coordinate_ =
                cancellation_arrival_coordinate_locked_v2();
            receipt.disposition = "accepted";
            receipt.state_observed = status_name_v2(status_);
            receipt.accepted_control_coordinate = accepted_cancel_coordinate_;
        }

        if (cancel_requested_) {
            receipt.first_cancel_request_id = first_cancel_request_id_;
            receipt.cancel_sequence = cancel_sequence_;
        }
        if (receipt.disposition == "accepted") {
            first_cancel_request_receipt_ready_ = true;
            condition_.notify_all();
        }
        return receipt;
    }

    BoundaryResolutionReceiptV2 consume_cancellation_at_boundary(
        const BoundaryMetadataV2& boundary
    ) {
        std::lock_guard<std::mutex> lock(mutex_);
        BoundaryResolutionReceiptV2 receipt;
        receipt.identity = identity_;

        // Failure is resolved before pending cancellation at the same boundary.
        if (failure_latched_ || status_ == RunStatusV2::kFailed) {
            receipt.disposition = "failed";
            receipt.terminal = true;
            receipt.terminal_status = "failed";
            return receipt;
        }
        if (sealed_) {
            receipt.disposition = status_ == RunStatusV2::kCancelled
                ? "already_cancelled"
                : "already_terminal";
            receipt.terminal = true;
            receipt.terminal_status = status_name_v2(status_);
            if (final_result_ && final_result_->final_boundary) {
                receipt.boundary = final_result_->final_boundary;
                receipt.cancellation = final_result_->cancellation;
            }
            return receipt;
        }

        require_live_started_locked_v2("consume_cancellation_at_boundary");
        if (inter_tick_control_window_open_) {
            throw std::logic_error(
                "a generation boundary cannot resolve inside the control window"
            );
        }
        validate_boundary_locked_v2(boundary);
        if (
            cancel_requested_
            && !cancel_consumed_
            && !boundary.cancellation_checkpoint_requested
        ) {
            throw std::invalid_argument(
                "the frozen checkpoint-on-cancel route requires a boundary checkpoint"
            );
        }
        generation_transaction_open_ = false;
        last_boundary_ = boundary;
        status_ = RunStatusV2::kGenerationComplete;
        receipt.boundary = boundary;

        if (!cancel_requested_ || cancel_consumed_) {
            if (boundary.next_generation == target_next_generation_) {
                awaiting_final_output_ack_ = true;
                receipt.disposition = "awaiting_final_output_ack";
            } else {
                receipt.disposition = "continue";
            }
            receipt.terminal = false;
            receipt.terminal_status = status_name_v2(status_);
            condition_.notify_all();
            return receipt;
        }
        if (!accepted_cancel_coordinate_) {
            throw std::logic_error(
                "pending cancellation has no accepted control coordinate"
            );
        }
        if (!boundary.cancellation_checkpoint_persisted) {
            awaiting_cancellation_checkpoint_ = true;
            receipt.disposition = "awaiting_cancellation_checkpoint";
            receipt.terminal = false;
            receipt.terminal_status = status_name_v2(status_);
            condition_.notify_all();
            return receipt;
        }
        validate_durable_boundary_settlement_locked_v2(boundary);
        return seal_cancelled_boundary_locked_v2(boundary);
    }

    BoundaryResolutionReceiptV2 acknowledge_cancellation_checkpoint(
        const std::string& expected_controller_id,
        const std::int64_t expected_owner_epoch,
        const BoundaryMetadataV2& boundary
    ) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (failure_latched_ || status_ == RunStatusV2::kFailed) {
            return failed_boundary_resolution_locked_v2();
        }
        if (
            expected_controller_id != identity_.controller_id
            || expected_owner_epoch != identity_.owner_epoch
        ) {
            seal_failure_locked_v2(FailureMetadataV2{
                "CANCELLATION_CHECKPOINT_IDENTITY_MISMATCH",
                "cancellation checkpoint acknowledgement identity mismatch",
                active_generation_,
                last_boundary_
                    ? last_boundary_->trajectory_commit_frontier
                    : -1,
            });
            return failed_boundary_resolution_locked_v2();
        }
        try {
            validate_cancellation_checkpoint_ack_locked_v2(boundary);
        } catch (const std::exception& error) {
            seal_failure_locked_v2(FailureMetadataV2{
                "CANCELLATION_CHECKPOINT_ACK_FAILED",
                error.what(),
                active_generation_,
                last_boundary_
                    ? last_boundary_->trajectory_commit_frontier
                    : -1,
            });
            return failed_boundary_resolution_locked_v2();
        }
        last_boundary_ = boundary;
        return seal_cancelled_boundary_locked_v2(boundary);
    }

    bool cancellation_pending_for_generation(
        const std::int64_t generation
    ) const {
        std::lock_guard<std::mutex> lock(mutex_);
        return run_started_
            && !sealed_
            && !failure_latched_
            && generation_transaction_open_
            && active_generation_ == generation
            && cancel_requested_
            && !cancel_consumed_
            && accepted_cancel_coordinate_.has_value()
            && accepted_cancel_coordinate_->generation == generation;
    }

    FailureReceiptV2 publish_failure(FailureMetadataV2 failure) {
        if (failure.error_code.empty()) {
            throw std::invalid_argument("failure error_code must not be empty");
        }
        if (failure.message.empty()) {
            throw std::invalid_argument("failure message must not be empty");
        }

        std::lock_guard<std::mutex> lock(mutex_);
        FailureReceiptV2 receipt;
        receipt.identity = identity_;
        if (failure_latched_) {
            receipt.disposition = "already_failed";
            receipt.failure = *failure_;
            receipt.terminal_status = "failed";
            return receipt;
        }
        if (sealed_) {
            receipt.disposition = "already_terminal";
            receipt.failure = std::move(failure);
            receipt.terminal_status = status_name_v2(status_);
            return receipt;
        }

        if (failure.active_generation < 0) {
            failure.active_generation = active_generation_;
        }
        seal_failure_locked_v2(failure);

        receipt.disposition = "accepted";
        receipt.failure = std::move(failure);
        receipt.accepted = true;
        receipt.terminal_status_changed_to_failed = true;
        receipt.terminal_status = "failed";
        condition_.notify_all();
        return receipt;
    }

    TerminalReceiptV2 acknowledge_run_complete(
        const std::string& expected_controller_id,
        const std::int64_t expected_owner_epoch,
        const BoundaryMetadataV2& acknowledged_boundary
    ) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (failure_latched_ || status_ == RunStatusV2::kFailed) {
            if (!final_result_) {
                throw std::logic_error("failed run has no terminal receipt");
            }
            return *final_result_;
        }
        if (sealed_) {
            if (!final_result_) {
                throw std::logic_error("sealed run has no terminal receipt");
            }
            return *final_result_;
        }
        require_live_started_locked_v2("publish_run_complete");
        if (
            generation_transaction_open_
            || status_ != RunStatusV2::kGenerationComplete
            || !last_boundary_
            || !awaiting_final_output_ack_
        ) {
            throw std::logic_error(
                "run completion requires a resolved complete-generation boundary"
            );
        }
        if (cancel_requested_ && !cancel_consumed_) {
            throw std::logic_error(
                "pending cancellation takes precedence over run completion"
            );
        }
        try {
            if (
                expected_controller_id != identity_.controller_id
                || expected_owner_epoch != identity_.owner_epoch
            ) {
                throw std::invalid_argument(
                    "run-complete acknowledgement identity mismatch"
                );
            }
            validate_final_output_ack_locked_v2(acknowledged_boundary);
        } catch (const std::exception& error) {
            seal_failure_locked_v2(FailureMetadataV2{
                "RUN_COMPLETE_ACK_FAILED",
                error.what(),
                active_generation_,
                last_boundary_
                    ? last_boundary_->trajectory_commit_frontier
                    : -1,
            });
            return *final_result_;
        }
        last_boundary_ = acknowledged_boundary;
        awaiting_final_output_ack_ = false;

        status_ = RunStatusV2::kRunComplete;
        sealed_ = true;
        TerminalReceiptV2 terminal;
        terminal.identity = identity_;
        terminal.terminal_status = "run_complete";
        terminal.sealed = true;
        terminal.final_boundary = last_boundary_;
        final_result_ = terminal;
        condition_.notify_all();
        return terminal;
    }

    RunControlManifestV2 manifest() const {
        std::lock_guard<std::mutex> lock(mutex_);
        RunControlManifestV2 out;
        out.identity = identity_;
        out.status = status_name_v2(status_);
        out.run_started = run_started_;
        out.sealed = sealed_;
        out.generation_transaction_open = generation_transaction_open_;
        out.active_generation = active_generation_;
        out.start_generation = start_generation_;
        out.requested_generations = requested_generations_;
        out.target_next_generation = target_next_generation_;
        out.inter_tick_control_window_open = inter_tick_control_window_open_;
        out.inter_tick_control_coordinate = current_control_coordinate_locked_v2();
        out.inter_tick_control_window_count = inter_tick_control_window_count_;
        out.cancel_requested = cancel_requested_;
        out.cancel_consumed = cancel_consumed_;
        out.awaiting_cancellation_checkpoint = awaiting_cancellation_checkpoint_;
        out.awaiting_final_output_ack = awaiting_final_output_ack_;
        if (cancel_requested_) {
            out.first_cancel_request_id = first_cancel_request_id_;
            out.first_cancel_reason = first_cancel_reason_;
            out.cancel_sequence = cancel_sequence_;
        }
        out.failure_latched = failure_latched_;
        out.last_boundary = last_boundary_;
        out.final_result = final_result_;
        return out;
    }

    std::optional<TerminalReceiptV2> final_result() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return final_result_;
    }

  private:
    BoundaryResolutionReceiptV2 failed_boundary_resolution_locked_v2() const {
        BoundaryResolutionReceiptV2 receipt;
        receipt.identity = identity_;
        receipt.disposition = "failed";
        receipt.terminal = true;
        receipt.terminal_status = "failed";
        return receipt;
    }

    void seal_failure_locked_v2(FailureMetadataV2 failure) {
        failure_latched_ = true;
        failure_ = failure;
        inter_tick_control_window_open_ = false;
        generation_transaction_open_ = false;
        awaiting_cancellation_checkpoint_ = false;
        awaiting_final_output_ack_ = false;
        status_ = RunStatusV2::kFailed;
        sealed_ = true;

        TerminalReceiptV2 terminal;
        terminal.identity = identity_;
        terminal.terminal_status = "failed";
        terminal.sealed = true;
        terminal.final_boundary = last_boundary_;
        terminal.failure = std::move(failure);
        final_result_ = std::move(terminal);
        condition_.notify_all();
    }

    BoundaryResolutionReceiptV2 seal_cancelled_boundary_locked_v2(
        const BoundaryMetadataV2& boundary
    ) {
        if (!accepted_cancel_coordinate_) {
            throw std::logic_error(
                "cannot seal cancellation without its accepted control coordinate"
            );
        }
        BoundaryResolutionReceiptV2 receipt;
        receipt.identity = identity_;
        receipt.boundary = boundary;
        cancel_consumed_ = true;
        awaiting_cancellation_checkpoint_ = false;
        CancellationObservationReceiptV2 observation;
        observation.request_id = first_cancel_request_id_;
        observation.reason = first_cancel_reason_;
        observation.identity = identity_;
        observation.cancel_sequence = cancel_sequence_;
        observation.request_control_coordinate = *accepted_cancel_coordinate_;
        observation.boundary = boundary;
        observation.request_observation_cursor =
            boundary_cursor_v2(boundary.next_generation);
        observation.final_safe_cursor = observation.request_observation_cursor;

        status_ = RunStatusV2::kCancelled;
        sealed_ = true;
        TerminalReceiptV2 terminal;
        terminal.identity = identity_;
        terminal.terminal_status = "cancelled";
        terminal.sealed = true;
        terminal.final_boundary = boundary;
        terminal.cancellation = observation;
        final_result_ = terminal;

        receipt.disposition = "cancelled";
        receipt.cancellation = observation;
        receipt.terminal = true;
        receipt.terminal_status = "cancelled";
        condition_.notify_all();
        return receipt;
    }

    static std::uint64_t next_identity_sequence_v2() {
        const std::uint64_t sequence = identity_sequence_v2_.fetch_add(
            1,
            std::memory_order_relaxed
        );
        if (sequence == 0 || sequence == std::numeric_limits<std::uint64_t>::max()) {
            throw std::overflow_error("CUDA run-control identity sequence exhausted");
        }
        return sequence;
    }

    static std::string make_unique_run_id_v2() {
        const auto now = std::chrono::system_clock::now().time_since_epoch();
        const auto nanoseconds =
            std::chrono::duration_cast<std::chrono::nanoseconds>(now).count();
        return "archb-cuda-run-v2-" + std::to_string(nanoseconds) + "-"
            + std::to_string(next_identity_sequence_v2());
    }

    static RunIdentityV2 make_identity_v2(
        std::string run_id,
        const std::int64_t owner_epoch
    ) {
        if (run_id.empty()) {
            throw std::invalid_argument("run_id must not be empty");
        }
        if (owner_epoch < 1) {
            throw std::invalid_argument("owner_epoch must be positive");
        }
        const std::string controller_id =
            "archb-cuda-controller-v2-" + std::to_string(next_identity_sequence_v2())
            + ":owner:" + std::to_string(owner_epoch) + ":run:" + run_id;
        return RunIdentityV2{
            std::move(run_id),
            controller_id,
            owner_epoch,
        };
    }

    static bool is_terminal_status_v2(const RunStatusV2 status) {
        return status == RunStatusV2::kCancelled
            || status == RunStatusV2::kFailed
            || status == RunStatusV2::kRunComplete;
    }

    static void validate_cancel_request_v2(const CancelRequestV2& request) {
        if (request.api_version != kCancelRequestApiVersionV2) {
            throw std::invalid_argument("cancel request api_version must equal 1");
        }
        if (request.request_id.empty()) {
            throw std::invalid_argument("cancel request_id must not be empty");
        }
        if (!cancel_reason_is_supported_v2(request.reason)) {
            throw std::invalid_argument("cancel reason is unsupported");
        }
    }

    static std::string boundary_cursor_v2(const std::int64_t generation) {
        return "generation:" + std::to_string(generation) + ":tick:0";
    }

    void require_live_started_locked_v2(const char* operation) const {
        if (!run_started_) {
            throw std::logic_error(
                std::string(operation) + " requires a started run"
            );
        }
        if (sealed_) {
            throw std::logic_error(
                std::string(operation) + " is unavailable after terminal sealing"
            );
        }
    }

    void validate_boundary_locked_v2(const BoundaryMetadataV2& boundary) const {
        if (!generation_transaction_open_) {
            throw std::logic_error("no generation transaction is open");
        }
        validate_boundary_values_locked_v2(boundary);
    }

    void validate_boundary_values_locked_v2(
        const BoundaryMetadataV2& boundary
    ) const {
        if (boundary.completed_generation != active_generation_) {
            throw std::invalid_argument(
                "completed_generation does not match the active generation"
            );
        }
        if (
            boundary.completed_generation == std::numeric_limits<std::int64_t>::max()
            || boundary.next_generation != boundary.completed_generation + 1
        ) {
            throw std::invalid_argument(
                "next_generation must exactly follow completed_generation"
            );
        }
        if (boundary.boundary_ordinal < 1) {
            throw std::invalid_argument("boundary_ordinal must be positive");
        }
        if (boundary.trajectory_commit_frontier != boundary.completed_generation) {
            throw std::invalid_argument(
                "trajectory_commit_frontier must equal completed_generation"
            );
        }
        if (
            boundary.snapshot_frontier > boundary.trajectory_commit_frontier
            || boundary.output_ack_frontier > boundary.snapshot_frontier
        ) {
            throw std::invalid_argument("boundary frontiers are contradictory");
        }
        if (
            boundary.cancellation_checkpoint_persisted
            && !boundary.cancellation_checkpoint_requested
        ) {
            throw std::invalid_argument(
                "a persisted cancellation checkpoint must have been requested"
            );
        }
        const bool has_snapshot_sequence = boundary.snapshot_sequence >= 0;
        if (has_snapshot_sequence != !boundary.snapshot_id.empty()) {
            throw std::invalid_argument(
                "snapshot_sequence and snapshot_id must be present together"
            );
        }
        if (!boundary.snapshot_digest_sha256.empty()) {
            if (!has_snapshot_sequence || boundary.snapshot_digest_sha256.size() != 64) {
                throw std::invalid_argument(
                    "snapshot digest requires a snapshot and 64 hexadecimal characters"
                );
            }
            if (!std::all_of(
                    boundary.snapshot_digest_sha256.begin(),
                    boundary.snapshot_digest_sha256.end(),
                    [](const char value) {
                        return (value >= '0' && value <= '9')
                            || (value >= 'a' && value <= 'f');
                    }
                )) {
                throw std::invalid_argument(
                    "snapshot digest must be lowercase hexadecimal"
                );
            }
        }
    }

    void validate_durable_boundary_settlement_locked_v2(
        const BoundaryMetadataV2& boundary
    ) const {
        if (
            boundary.snapshot_frontier != boundary.trajectory_commit_frontier
            || boundary.output_ack_frontier != boundary.trajectory_commit_frontier
            || boundary.snapshot_sequence < 0
            || boundary.snapshot_id.empty()
            || boundary.snapshot_digest_sha256.empty()
        ) {
            throw std::invalid_argument(
                "cancelled publication requires one exact durable checkpoint"
            );
        }
    }

    void validate_cancellation_checkpoint_ack_locked_v2(
        const BoundaryMetadataV2& boundary
    ) const {
        if (generation_transaction_open_ || !last_boundary_) {
            throw std::logic_error(
                "cancellation checkpoint acknowledgement has no closed boundary"
            );
        }
        validate_boundary_values_locked_v2(boundary);
        if (
            !boundary.cancellation_checkpoint_requested
            || !boundary.cancellation_checkpoint_persisted
        ) {
            throw std::invalid_argument(
                "cancellation acknowledgement omitted its persisted checkpoint"
            );
        }
        if (
            boundary.completed_generation != last_boundary_->completed_generation
            || boundary.next_generation != last_boundary_->next_generation
            || boundary.boundary_ordinal != last_boundary_->boundary_ordinal
            || boundary.trajectory_commit_frontier
                != last_boundary_->trajectory_commit_frontier
        ) {
            throw std::invalid_argument(
                "cancellation checkpoint acknowledgement changed boundary identity"
            );
        }
        validate_durable_boundary_settlement_locked_v2(boundary);
    }

    void validate_final_output_ack_locked_v2(
        const BoundaryMetadataV2& boundary
    ) const {
        if (!last_boundary_) {
            throw std::logic_error("run-complete acknowledgement has no boundary");
        }
        validate_boundary_values_locked_v2(boundary);
        if (
            boundary.completed_generation != last_boundary_->completed_generation
            || boundary.next_generation != target_next_generation_
            || boundary.boundary_ordinal != last_boundary_->boundary_ordinal
            || boundary.trajectory_commit_frontier
                != last_boundary_->trajectory_commit_frontier
        ) {
            throw std::invalid_argument(
                "run-complete acknowledgement changed the final boundary"
            );
        }
        validate_durable_boundary_settlement_locked_v2(boundary);
    }

    ControlCoordinateV2 current_control_coordinate_locked_v2() const {
        return ControlCoordinateV2{
            inter_tick_control_generation_,
            inter_tick_completed_tick_,
            inter_tick_next_tick_,
            inter_tick_control_window_count_,
            generation_transaction_open_,
        };
    }

    ControlCoordinateV2 cancellation_arrival_coordinate_locked_v2() const {
        if (inter_tick_control_generation_ == active_generation_) {
            return current_control_coordinate_locked_v2();
        }
        return ControlCoordinateV2{
            active_generation_,
            -1,
            0,
            inter_tick_control_window_count_,
            generation_transaction_open_,
        };
    }

    inline static std::atomic<std::uint64_t> identity_sequence_v2_{1};

    RunIdentityV2 identity_;
    mutable std::mutex mutex_;
    std::condition_variable condition_;
    RunStatusV2 status_ = RunStatusV2::kInitialized;
    bool run_started_ = false;
    bool sealed_ = false;
    bool generation_transaction_open_ = false;
    std::int64_t active_generation_ = -1;
    std::int64_t start_generation_ = -1;
    std::int64_t requested_generations_ = 0;
    std::int64_t target_next_generation_ = -1;
    bool inter_tick_control_window_open_ = false;
    std::int64_t inter_tick_control_generation_ = -1;
    std::int64_t inter_tick_completed_tick_ = -1;
    std::int64_t inter_tick_next_tick_ = -1;
    std::uint64_t inter_tick_control_window_count_ = 0;
    std::int64_t last_control_window_generation_ = -1;
    bool cancel_requested_ = false;
    bool cancel_consumed_ = false;
    bool awaiting_cancellation_checkpoint_ = false;
    bool awaiting_final_output_ack_ = false;
    bool first_cancel_request_receipt_ready_ = false;
    std::string first_cancel_request_id_;
    std::string first_cancel_reason_;
    std::int64_t cancel_sequence_ = 0;
    std::optional<ControlCoordinateV2> accepted_cancel_coordinate_;
    bool failure_latched_ = false;
    std::optional<FailureMetadataV2> failure_;
    std::optional<BoundaryMetadataV2> last_boundary_;
    std::optional<TerminalReceiptV2> final_result_;
};

}  // namespace eiel::cuda_architecture_b::resident_run_control_v2
