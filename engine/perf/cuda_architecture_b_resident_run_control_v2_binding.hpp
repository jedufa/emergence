#pragma once

#include "cuda_architecture_b_resident_run_control_v2.hpp"

#include <pybind11/pybind11.h>

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>

namespace eiel::cuda_architecture_b::resident_run_control_v2_binding {

namespace py = pybind11;
namespace run_control =
    eiel::cuda_architecture_b::resident_run_control_v2;

inline constexpr std::array<std::string_view, 8> kCancelReceiptFieldsV2{
    "request_id",
    "controller_id",
    "owner_epoch",
    "disposition",
    "first_cancel_request_id",
    "cancel_sequence",
    "state_observed",
    "terminal_status_changed",
};

inline void add_identity_fields_v2(
    py::dict& out,
    const run_control::RunIdentityV2& identity
) {
    out["run_id"] = identity.run_id;
    out["controller_id"] = identity.controller_id;
    out["owner_epoch"] = identity.owner_epoch;
}

inline py::dict identity_to_dict_v2(
    const run_control::RunIdentityV2& identity
) {
    py::dict out;
    add_identity_fields_v2(out, identity);
    return out;
}

inline py::dict control_coordinate_to_dict_v2(
    const run_control::ControlCoordinateV2& coordinate
) {
    py::dict out;
    out["generation"] = coordinate.generation;
    out["completed_tick"] = coordinate.completed_tick;
    out["next_tick"] = coordinate.next_tick;
    out["window_count"] = coordinate.window_count;
    out["generation_transaction_open"] =
        coordinate.generation_transaction_open;
    return out;
}

inline py::dict boundary_metadata_to_dict_v2(
    const run_control::BoundaryMetadataV2& boundary
) {
    py::dict out;
    out["completed_generation"] = boundary.completed_generation;
    out["next_generation"] = boundary.next_generation;
    out["boundary_ordinal"] = boundary.boundary_ordinal;
    out["trajectory_commit_frontier"] = boundary.trajectory_commit_frontier;
    out["snapshot_frontier"] = boundary.snapshot_frontier;
    out["output_ack_frontier"] = boundary.output_ack_frontier;
    out["snapshot_sequence"] = boundary.snapshot_sequence;
    out["snapshot_id"] = boundary.snapshot_id;
    out["snapshot_digest_sha256"] = boundary.snapshot_digest_sha256;
    out["cancellation_checkpoint_requested"] =
        boundary.cancellation_checkpoint_requested;
    out["cancellation_checkpoint_persisted"] =
        boundary.cancellation_checkpoint_persisted;
    return out;
}

inline run_control::BoundaryMetadataV2 boundary_metadata_from_dict_v2(
    const py::dict& boundary
) {
    constexpr std::array<std::string_view, 11> fields{
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
    };
    if (py::len(boundary) != static_cast<py::size_t>(fields.size())) {
        throw py::value_error(
            "CUDA V2 boundary acknowledgement must contain exactly 11 fields"
        );
    }
    for (const std::string_view field : fields) {
        if (!boundary.contains(py::str(field.data(), field.size()))) {
            throw py::value_error(
                "CUDA V2 boundary acknowledgement is missing "
                + std::string(field)
            );
        }
    }
    run_control::BoundaryMetadataV2 out;
    out.completed_generation = py::cast<std::int64_t>(boundary["completed_generation"]);
    out.next_generation = py::cast<std::int64_t>(boundary["next_generation"]);
    out.boundary_ordinal = py::cast<std::int64_t>(boundary["boundary_ordinal"]);
    out.trajectory_commit_frontier =
        py::cast<std::int64_t>(boundary["trajectory_commit_frontier"]);
    out.snapshot_frontier = py::cast<std::int64_t>(boundary["snapshot_frontier"]);
    out.output_ack_frontier =
        py::cast<std::int64_t>(boundary["output_ack_frontier"]);
    out.snapshot_sequence = py::cast<std::int64_t>(boundary["snapshot_sequence"]);
    out.snapshot_id = py::cast<std::string>(boundary["snapshot_id"]);
    out.snapshot_digest_sha256 =
        py::cast<std::string>(boundary["snapshot_digest_sha256"]);
    out.cancellation_checkpoint_requested =
        py::cast<bool>(boundary["cancellation_checkpoint_requested"]);
    out.cancellation_checkpoint_persisted =
        py::cast<bool>(boundary["cancellation_checkpoint_persisted"]);
    return out;
}

inline py::dict failure_metadata_to_dict_v2(
    const run_control::FailureMetadataV2& failure
) {
    py::dict out;
    out["error_code"] = failure.error_code;
    out["message"] = failure.message;
    out["active_generation"] = failure.active_generation;
    out["trajectory_commit_frontier"] = failure.trajectory_commit_frontier;
    return out;
}

inline py::dict receipt_to_dict_v2(
    const run_control::StartReceiptV2& receipt
) {
    py::dict out;
    out["schema_version"] = receipt.schema_version;
    add_identity_fields_v2(out, receipt.identity);
    out["start_generation"] = receipt.start_generation;
    out["single_use"] = receipt.single_use;
    out["status"] = receipt.status;
    return out;
}

inline py::dict receipt_to_dict_v2(
    const run_control::GenerationStartReceiptV2& receipt
) {
    py::dict out;
    out["schema_version"] = receipt.schema_version;
    add_identity_fields_v2(out, receipt.identity);
    out["generation"] = receipt.generation;
    out["generation_transaction_open"] =
        receipt.generation_transaction_open;
    out["status"] = receipt.status;
    return out;
}

inline py::dict receipt_to_dict_v2(
    const run_control::ControlWindowReceiptV2& receipt
) {
    py::dict out;
    out["schema_version"] = receipt.schema_version;
    add_identity_fields_v2(out, receipt.identity);
    out["coordinate"] = control_coordinate_to_dict_v2(receipt.coordinate);
    out["wait_timeout_ms"] = receipt.wait_timeout_ms;
    out["close_disposition"] = receipt.close_disposition;
    out["cancel_requested"] = receipt.cancel_requested;
    out["first_cancel_request_id"] = receipt.first_cancel_request_id
        ? py::cast(*receipt.first_cancel_request_id)
        : py::none();
    out["cancel_sequence"] = receipt.cancel_sequence
        ? py::cast(*receipt.cancel_sequence)
        : py::none();
    out["status_after_close"] = receipt.status_after_close;
    return out;
}

inline py::dict receipt_to_dict_v2(
    const run_control::CancelRequestReceiptV2& receipt
) {
    py::dict out;
    out["request_id"] = receipt.request_id;
    out["controller_id"] = receipt.identity.controller_id;
    out["owner_epoch"] = receipt.identity.owner_epoch;
    out["disposition"] = receipt.disposition;
    out["first_cancel_request_id"] = receipt.first_cancel_request_id
        ? py::cast(*receipt.first_cancel_request_id)
        : py::none();
    out["cancel_sequence"] = receipt.cancel_sequence
        ? py::cast(*receipt.cancel_sequence)
        : py::none();
    out["state_observed"] = receipt.state_observed;
    out["terminal_status_changed"] = receipt.terminal_status_changed;
    return out;
}

inline py::dict receipt_to_dict_v2(
    const run_control::CancellationObservationReceiptV2& receipt
) {
    py::dict out;
    out["schema_version"] = receipt.schema_version;
    out["request_id"] = receipt.request_id;
    out["reason"] = receipt.reason;
    add_identity_fields_v2(out, receipt.identity);
    out["cancel_sequence"] = receipt.cancel_sequence;
    out["request_control_coordinate"] =
        control_coordinate_to_dict_v2(receipt.request_control_coordinate);
    out["boundary"] = boundary_metadata_to_dict_v2(receipt.boundary);
    out["request_observation_cursor"] = receipt.request_observation_cursor;
    out["final_safe_cursor"] = receipt.final_safe_cursor;
    out["terminal_status"] = receipt.terminal_status;
    out["success_marker_allowed"] = receipt.success_marker_allowed;
    out["promotion_result_allowed"] = receipt.promotion_result_allowed;
    return out;
}

inline py::dict receipt_to_dict_v2(
    const run_control::FailureReceiptV2& receipt
) {
    py::dict out;
    out["schema_version"] = receipt.schema_version;
    add_identity_fields_v2(out, receipt.identity);
    out["disposition"] = receipt.disposition;
    out["failure"] = failure_metadata_to_dict_v2(receipt.failure);
    out["accepted"] = receipt.accepted;
    out["terminal_status_changed_to_failed"] =
        receipt.terminal_status_changed_to_failed;
    out["terminal_status"] = receipt.terminal_status;
    return out;
}

inline py::dict receipt_to_dict_v2(
    const run_control::TerminalReceiptV2& receipt
) {
    py::dict out;
    out["schema_version"] = receipt.schema_version;
    add_identity_fields_v2(out, receipt.identity);
    out["terminal_status"] = receipt.terminal_status;
    out["sealed"] = receipt.sealed;
    if (receipt.final_boundary) {
        out["final_boundary"] =
            boundary_metadata_to_dict_v2(*receipt.final_boundary);
    } else {
        out["final_boundary"] = py::none();
    }
    if (receipt.cancellation) {
        out["cancellation"] = receipt_to_dict_v2(*receipt.cancellation);
    } else {
        out["cancellation"] = py::none();
    }
    if (receipt.failure) {
        out["failure"] = failure_metadata_to_dict_v2(*receipt.failure);
    } else {
        out["failure"] = py::none();
    }
    out["success_marker_allowed"] = receipt.success_marker_allowed;
    out["promotion_result_allowed"] = receipt.promotion_result_allowed;
    return out;
}

inline py::dict receipt_to_dict_v2(
    const run_control::BoundaryResolutionReceiptV2& receipt
) {
    py::dict out;
    out["schema_version"] = receipt.schema_version;
    add_identity_fields_v2(out, receipt.identity);
    out["disposition"] = receipt.disposition;
    if (receipt.boundary) {
        out["boundary"] = boundary_metadata_to_dict_v2(*receipt.boundary);
    } else {
        out["boundary"] = py::none();
    }
    if (receipt.cancellation) {
        out["cancellation"] = receipt_to_dict_v2(*receipt.cancellation);
    } else {
        out["cancellation"] = py::none();
    }
    out["terminal"] = receipt.terminal;
    out["terminal_status"] = receipt.terminal_status;
    return out;
}

inline py::dict manifest_to_dict_v2(
    const run_control::RunControlManifestV2& manifest
) {
    py::dict out;
    out["schema_version"] = manifest.schema_version;
    add_identity_fields_v2(out, manifest.identity);
    out["controller_protocol_version"] = manifest.controller_protocol_version;
    out["controller_protocol_schema_version"] =
        manifest.controller_protocol_schema_version;
    out["status"] = manifest.status;
    out["run_started"] = manifest.run_started;
    out["sealed"] = manifest.sealed;
    out["native_owner_private"] = manifest.native_owner_private;
    out["python_callback_entry_points"] = manifest.python_callback_entry_points;
    out["inter_tick_control_wait_timeout_ms"] =
        manifest.inter_tick_control_wait_timeout_ms;
    out["generation_transaction_open"] =
        manifest.generation_transaction_open;
    out["active_generation"] = manifest.active_generation;
    out["start_generation"] = manifest.start_generation;
    out["requested_generations"] = manifest.requested_generations;
    out["target_next_generation"] = manifest.target_next_generation;
    out["checkpoint_on_cancel"] = manifest.checkpoint_on_cancel;
    out["final_output_ack_required"] = manifest.final_output_ack_required;
    out["inter_tick_control_window_open"] =
        manifest.inter_tick_control_window_open;
    out["inter_tick_control_coordinate"] =
        control_coordinate_to_dict_v2(manifest.inter_tick_control_coordinate);
    out["inter_tick_control_window_count"] =
        manifest.inter_tick_control_window_count;
    out["cancel_requested"] = manifest.cancel_requested;
    out["cancel_consumed"] = manifest.cancel_consumed;
    out["awaiting_cancellation_checkpoint"] =
        manifest.awaiting_cancellation_checkpoint;
    out["awaiting_final_output_ack"] = manifest.awaiting_final_output_ack;
    out["first_cancel_request_id"] = manifest.first_cancel_request_id
        ? py::cast(*manifest.first_cancel_request_id)
        : py::none();
    out["first_cancel_reason"] = manifest.first_cancel_reason
        ? py::cast(*manifest.first_cancel_reason)
        : py::none();
    out["cancel_sequence"] = manifest.cancel_sequence
        ? py::cast(*manifest.cancel_sequence)
        : py::none();
    out["failure_latched"] = manifest.failure_latched;
    if (manifest.last_boundary) {
        out["last_boundary"] =
            boundary_metadata_to_dict_v2(*manifest.last_boundary);
    } else {
        out["last_boundary"] = py::none();
    }
    if (manifest.final_result) {
        out["final_result"] = receipt_to_dict_v2(*manifest.final_result);
    } else {
        out["final_result"] = py::none();
    }
    return out;
}

inline run_control::CancelRequestV2 cancel_request_from_dict_v2(
    const py::dict& request
) {
    if (
        py::len(request)
        != static_cast<py::size_t>(run_control::kCancelRequestFieldsV2.size())
    ) {
        throw py::value_error(
            "CUDA Architecture B cancel request must contain exactly five fields"
        );
    }
    for (const std::string_view field : run_control::kCancelRequestFieldsV2) {
        if (!request.contains(py::str(field.data(), field.size()))) {
            throw py::value_error(
                "CUDA Architecture B cancel request is missing "
                + std::string(field)
            );
        }
    }

    run_control::CancelRequestV2 out;
    out.api_version = py::cast<std::int64_t>(request["api_version"]);
    out.request_id = py::cast<std::string>(request["request_id"]);
    out.expected_controller_id =
        py::cast<std::string>(request["expected_controller_id"]);
    out.expected_owner_epoch =
        py::cast<std::int64_t>(request["expected_owner_epoch"]);
    out.reason = py::cast<std::string>(request["reason"]);
    return out;
}

inline py::dict request_cancel_v2(
    run_control::RunControlV2& controller,
    const py::dict& request
) {
    const run_control::CancelRequestV2 parsed =
        cancel_request_from_dict_v2(request);
    run_control::CancelRequestReceiptV2 receipt;
    {
        py::gil_scoped_release release;
        receipt = controller.request_cancel(parsed);
    }
    return receipt_to_dict_v2(receipt);
}

inline py::dict manifest_v2(run_control::RunControlV2& controller) {
    run_control::RunControlManifestV2 manifest;
    {
        py::gil_scoped_release release;
        manifest = controller.manifest();
    }
    return manifest_to_dict_v2(manifest);
}

inline py::object final_result_v2(run_control::RunControlV2& controller) {
    std::optional<run_control::TerminalReceiptV2> result;
    {
        py::gil_scoped_release release;
        result = controller.final_result();
    }
    if (!result) {
        return py::none();
    }
    return receipt_to_dict_v2(*result);
}

inline py::dict acknowledge_cancellation_checkpoint_v2(
    run_control::RunControlV2& controller,
    const std::string& expected_controller_id,
    const std::int64_t expected_owner_epoch,
    const py::dict& boundary
) {
    const auto parsed = boundary_metadata_from_dict_v2(boundary);
    run_control::BoundaryResolutionReceiptV2 receipt;
    {
        py::gil_scoped_release release;
        receipt = controller.acknowledge_cancellation_checkpoint(
            expected_controller_id,
            expected_owner_epoch,
            parsed
        );
    }
    return receipt_to_dict_v2(receipt);
}

inline py::dict acknowledge_run_complete_v2(
    run_control::RunControlV2& controller,
    const std::string& expected_controller_id,
    const std::int64_t expected_owner_epoch,
    const py::dict& boundary
) {
    const auto parsed = boundary_metadata_from_dict_v2(boundary);
    run_control::TerminalReceiptV2 receipt;
    {
        py::gil_scoped_release release;
        receipt = controller.acknowledge_run_complete(
            expected_controller_id,
            expected_owner_epoch,
            parsed
        );
    }
    return receipt_to_dict_v2(receipt);
}

inline py::dict publish_failure_v2(
    run_control::RunControlV2& controller,
    const std::string& error_code,
    const std::string& message,
    const std::int64_t active_generation,
    const std::int64_t trajectory_commit_frontier
) {
    run_control::FailureReceiptV2 receipt;
    {
        py::gil_scoped_release release;
        receipt = controller.publish_failure(run_control::FailureMetadataV2{
            error_code,
            message,
            active_generation,
            trajectory_commit_frontier,
        });
    }
    return receipt_to_dict_v2(receipt);
}

inline void bind_run_control_v2(py::module_& module) {
    py::class_<
        run_control::RunControlV2,
        std::shared_ptr<run_control::RunControlV2>
    >(
        module,
        "RunControlV2",
        py::module_local()
    )
        .def(py::init<>())
        .def(
            py::init<std::string, std::int64_t>(),
            py::arg("run_id"),
            py::arg("owner_epoch") = 1
        )
        .def("request_cancel", &request_cancel_v2, py::arg("request"))
        .def(
            "acknowledge_cancellation_checkpoint",
            &acknowledge_cancellation_checkpoint_v2,
            py::arg("expected_controller_id"),
            py::arg("expected_owner_epoch"),
            py::arg("boundary")
        )
        .def(
            "acknowledge_run_complete",
            &acknowledge_run_complete_v2,
            py::arg("expected_controller_id"),
            py::arg("expected_owner_epoch"),
            py::arg("boundary")
        )
        .def(
            "publish_failure",
            &publish_failure_v2,
            py::arg("error_code"),
            py::arg("message"),
            py::arg("active_generation"),
            py::arg("trajectory_commit_frontier")
        )
        .def("manifest", &manifest_v2)
        .def("final_result", &final_result_v2);
}

}  // namespace eiel::cuda_architecture_b::resident_run_control_v2_binding
