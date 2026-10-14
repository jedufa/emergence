#pragma once

#include "cuda_architecture_b_flattened_controller_v1.hpp"
#include "cuda_architecture_b_resident_event_archive_v1.hpp"
#include "cuda_architecture_b_resident_window_refill_v1.hpp"

#include <torch/extension.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace eiel::cuda_architecture_b::resident_window_controller_v1 {

namespace py = pybind11;
namespace event_archive =
    eiel::cuda_architecture_b::resident_event_archive_v1;
namespace window_refill =
    eiel::cuda_architecture_b::resident_window_refill_v1;

constexpr int64_t kControllerStatusArgumentV1 = 59;
constexpr int64_t kFullTickOutputsArgumentV1 = 61;
constexpr int64_t kEventStateArgumentV1 = 67;
constexpr int64_t kOrdinaryBirthStateArgumentV1 = 68;
constexpr int64_t kTurnoverOutputsArgumentV1 = 74;
constexpr int64_t kPatchStatsArgumentV1 = 78;
constexpr int64_t kGenerationArgumentV1 = 120;
constexpr int64_t kGenerationCountArgumentV1 = 143;
constexpr int64_t kNativeArgumentCountV1 = 160;
constexpr int64_t kPhysicalWindowGenerationsV1 = 2;
constexpr int64_t kMinimumPhysicalWindowGenerationsV1 = 1;
constexpr int64_t kMaximumPhysicalWindowGenerationsV1 = 2;
constexpr int64_t kMaximumLogicalGenerationsV1 = 128;

constexpr int64_t kLifecycleStatusArgumentV1 = 58;
constexpr int64_t kGenerationSummaryOutputV1 = 19;
constexpr int64_t kTurnoverCountsOutputV1 = 7;
constexpr int64_t kTurnoverStatusOutputV1 = 8;
constexpr int64_t kEventActiveCountV1 = 22;
constexpr int64_t kEventStatusV1 = 23;
constexpr int64_t kEventSequenceBaseV1 = 30;

constexpr int64_t kPythonRandomWordsV1 = 35;
constexpr int64_t kPythonRandomIndexV1 = 36;
constexpr int64_t kPythonRandomHasGaussV1 = 37;
constexpr int64_t kPythonRandomGaussNextV1 = 38;
constexpr int64_t kPythonRandomPrimaryTapeV1 = 39;
constexpr int64_t kPythonRandomSecondaryTapeV1 = 40;
constexpr int64_t kPythonRandomRawDrawsV1 = 41;
constexpr int64_t kMutationPySemanticDrawsV1 = 42;

constexpr int64_t kPolicyRngStateV1 = 85;
constexpr int64_t kPolicyDeltaTapeV1 = 86;
constexpr int64_t kPolicyMutationMaskTapeV1 = 87;
constexpr int64_t kPolicyCrossoverMaskPackedTapeV1 = 88;
constexpr int64_t kPolicyPackedPostStateTapeV1 = 89;
constexpr int64_t kPolicySemanticTapeV1 = 90;
constexpr int64_t kPolicyAbsoluteCursorV1 = 91;
constexpr int64_t kMutationTorchSemanticDrawsV1 = 92;
constexpr int64_t kPolicyWindowBaseV1 = 111;
constexpr int64_t kPythonRandomWindowBaseV1 = 112;

constexpr int64_t kFullTickDriveNoiseV1 = 5;
constexpr int64_t kFullTickRunnerPcg64StateWordsV1 = 16;
constexpr int64_t kFullTickRunnerPcg64BufferWordsV1 = 17;
constexpr int64_t kFullTickDriveNoiseSamplesV1 = 20;
constexpr int64_t kFullTickDriveNoiseSamplesValidV1 = 21;

inline torch::Tensor immutable_host_copy_v1(const torch::Tensor& value) {
    return value.detach().cpu().clone().contiguous();
}

inline std::string first_failure_record_suffix_v1(
    const py::dict& receipt,
    const int64_t cell_index
) {
    if (!receipt.contains("first_failure_by_cell")) {
        return {};
    }
    const py::handle value = receipt["first_failure_by_cell"];
    if (!PyList_CheckExact(value.ptr())) {
        return {};
    }
    const py::list records = py::reinterpret_borrow<py::list>(value);
    for (const py::handle item : records) {
        if (!PyDict_CheckExact(item.ptr())) {
            continue;
        }
        const py::dict record = py::reinterpret_borrow<py::dict>(item);
        if (!record.contains("cell")) {
            continue;
        }
        const py::handle record_cell = record["cell"];
        if (
            !PyLong_CheckExact(record_cell.ptr())
            || py::cast<int64_t>(record_cell) != cell_index
        ) {
            continue;
        }
        return " first_failure_by_cell="
            + py::cast<std::string>(py::repr(record));
    }
    return {};
}

inline void require_zero_int32_status_v1(
    const torch::Tensor& value,
    const char* name,
    const py::dict* diagnostic_receipt = nullptr
) {
    TORCH_CHECK(value.defined(), name, " must be defined");
    TORCH_CHECK(value.scalar_type() == torch::kInt32, name, " must be int32");
    TORCH_CHECK(value.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(!value.requires_grad(), name, " must not require gradients");
    const auto host = value.detach().cpu().contiguous();
    const auto* rows = host.data_ptr<int32_t>();
    for (int64_t index = 0; index < host.numel(); ++index) {
        if (rows[index] == 0) {
            continue;
        }
        std::string message =
            std::string(name)
            + " must be zero at an accepted window boundary: cell_index="
            + std::to_string(index)
            + " status="
            + std::to_string(rows[index]);
        if (diagnostic_receipt != nullptr) {
            message += first_failure_record_suffix_v1(
                *diagnostic_receipt,
                index
            );
        }
        TORCH_CHECK(false, message);
    }
}

inline const char* turnover_failure_phase_name_v1(const int32_t phase) {
    switch (phase) {
        case 1:
            return "plan";
        case 2:
            return "cull";
        case 3:
            return "refill";
        case 4:
            return "identity";
        case 5:
            return "scalar";
        case 6:
            return "drive";
        case 7:
            return "chemistry";
        case 8:
            return "policy";
        case 10:
            return "aggregate_only";
        default:
            return "legacy_or_unknown";
    }
}

inline void require_zero_turnover_status_v1(const torch::Tensor& value) {
    constexpr int32_t diagnostic_stride = 1000;
    TORCH_CHECK(value.defined(), "turnover status must be defined");
    TORCH_CHECK(
        value.scalar_type() == torch::kInt32,
        "turnover status must be int32"
    );
    TORCH_CHECK(value.is_contiguous(), "turnover status must be contiguous");
    TORCH_CHECK(
        !value.requires_grad(),
        "turnover status must not require gradients"
    );
    const auto host = value.detach().cpu().contiguous();
    const auto* rows = host.data_ptr<int32_t>();
    for (int64_t index = 0; index < host.numel(); ++index) {
        const int32_t status = rows[index];
        const int32_t phase = status >= diagnostic_stride
            ? status / diagnostic_stride
            : 0;
        const int32_t substatus = status >= diagnostic_stride
            ? status % diagnostic_stride
            : status;
        TORCH_CHECK(
            status == 0,
            "turnover status must be zero at an accepted window boundary: "
            "cell_index=",
            index,
            " status=",
            status,
            " diagnostic_phase=",
            turnover_failure_phase_name_v1(phase),
            " diagnostic_substatus=",
            substatus
        );
    }
}

inline py::dict event_drain_receipt_to_dict_v1(
    const event_archive::EventDrainReceiptV1& receipt
) {
    py::dict out;
    out["active_count_before"] = receipt.active_count_before;
    out["event_sequence_base_before"] = receipt.sequence_base_before;
    out["event_sequence_base_after"] = receipt.sequence_base_after;
    out["host_archive_bytes_appended"] = receipt.copied_bytes;
    out["host_archive_cumulative_bytes"] = receipt.cumulative_archive_bytes;
    out["host_archive_remaining_budget_bytes"] = receipt.remaining_budget_bytes;
    return out;
}

inline py::list event_histories_to_list_v1(
    const std::vector<event_archive::EventChunkV1>& histories
) {
    py::list cells;
    for (const auto& history : histories) {
        py::list columns;
        for (const auto& column : history) {
            columns.append(column);
        }
        cells.append(std::move(columns));
    }
    return cells;
}

enum OptionDM0AdditiveFieldV1 : std::size_t {
    kM0PolicyRefillCallsV1 = 0,
    kM0PolicyD2hOperationsV1,
    kM0PolicyH2dOperationsV1,
    kM0PolicyHostWallTimeV1,
    kM0PolicyD2hTimeV1,
    kM0PolicyH2dTimeV1,
    kM0PolicyTorchRandomTimeV1,
    kM0PolicyComparePackTimeV1,
    kM0PolicyAllocationCopyTimeV1,
    kM0GaussianRefillCallsV1,
    kM0GaussianD2hOperationsV1,
    kM0GaussianH2dOperationsV1,
    kM0GaussianHostWallTimeV1,
    kM0GaussianD2hTimeV1,
    kM0GaussianH2dTimeV1,
    kM0GaussianRawGenerationTimeV1,
    kM0GaussianTranscendentalTimeV1,
    kM0GaussianAllocationCopyTimeV1,
    kM0HostSynchronizationsInsideLoopV1,
    kM0FinalHostSynchronizationsV1,
    kM0IntermediateD2hOperationsV1,
    kM0FinalD2hOperationsV1,
    kM0IntermediateH2dOperationsV1,
    kM0FinalH2dOperationsV1,
    kM0IntermediateHostSynchronizationsV1,
    kM0AtenOperationsInsideTickLoopV1,
    kM0ExactCpuPolicyServiceCallsV1,
    kM0AdditiveFieldCountV1,
};

inline constexpr std::array<const char*, kM0AdditiveFieldCountV1>
    kOptionDM0AdditiveFieldsV1{
        "policy_birth_segment_refill_calls_inside_tick_loop",
        "policy_birth_segment_device_to_host_operations_lower_bound",
        "policy_birth_segment_host_to_device_operations_lower_bound",
        "policy_birth_segment_host_wall_time_ns",
        "policy_birth_segment_device_to_host_time_ns",
        "policy_birth_segment_host_to_device_time_ns",
        "policy_birth_segment_torch_random_generation_time_ns",
        "policy_birth_segment_torch_compare_pack_time_ns",
        "policy_birth_segment_allocation_stack_copy_time_ns",
        "gaussian_segment_refill_calls_inside_tick_loop",
        "gaussian_segment_device_to_host_operations_lower_bound",
        "gaussian_segment_host_to_device_operations_lower_bound",
        "gaussian_segment_host_wall_time_ns",
        "gaussian_segment_device_to_host_time_ns",
        "gaussian_segment_host_to_device_time_ns",
        "gaussian_segment_raw_generation_time_ns",
        "gaussian_segment_transcendental_conversion_time_ns",
        "gaussian_segment_allocation_copy_time_ns",
        "host_synchronizations_inside_loop",
        "final_host_synchronizations",
        "intermediate_device_to_host_operations",
        "final_device_to_host_operations",
        "intermediate_host_to_device_operations",
        "final_host_to_device_operations",
        "intermediate_host_synchronizations",
        "aten_operations_inside_tick_loop",
        "exact_cpu_policy_service_calls_inside_tick_loop",
    };

enum OptionDM0RawPolicyFieldV1 : std::size_t {
    kM0RawCallsV1 = 0,
    kM0RawHostWallTimeV1,
    kM0RawD2hTimeV1,
    kM0RawH2dTimeV1,
    kM0RawCpuMlpTimeV1,
    kM0RawCpuSamplingTimeV1,
    kM0RawFinalSyncTimeV1,
    kM0RawD2hBytesV1,
    kM0RawH2dBytesV1,
    kM0RawD2hOperationsV1,
    kM0RawH2dOperationsV1,
    kM0RawExplicitFinalSyncsV1,
    kM0RawImplicitD2hWaitsV1,
    kM0RawHostAllocationCountV1,
    kM0RawHostAllocationBytesV1,
    kM0RawPolicyFieldCountV1,
};

inline constexpr std::array<const char*, kM0RawPolicyFieldCountV1>
    kOptionDM0RawPolicyFieldsV1{
        "calls_inside_tick_loop",
        "host_wall_time_ns",
        "device_to_host_time_ns",
        "host_to_device_time_ns",
        "cpu_mlp_time_ns",
        "cpu_sampling_time_ns",
        "final_stream_synchronization_time_ns",
        "device_to_host_bytes_lower_bound",
        "host_to_device_bytes_lower_bound",
        "device_to_host_operations",
        "host_to_device_operations",
        "explicit_final_stream_synchronizations",
        "implicit_blocking_device_to_host_transfer_waits_lower_bound",
        "host_allocation_count_lower_bound",
        "host_allocation_bytes_lower_bound",
    };

inline constexpr std::array<const char*, 3> kOptionDM0RawPolicyScopeFieldsV1{
    "host_allocation_measurement_scope",
    "transfer_operation_measurement_scope",
    "synchronization_measurement_scope",
};

inline constexpr std::array<const char*, 3> kOptionDM0RawPolicyScopesV1{
    "lower_bound_of_nonempty_explicit_service_owned_host_tensor_and_vector_outputs",
    "nonempty_tensor_copy_operations_only",
    "explicit_final_cuda_stream_synchronizations_and_lower_bound_blocking_"
    "device_to_host_tensor_transfers",
};

inline int64_t checked_nonnegative_sum_v1(
    const int64_t total,
    const int64_t increment,
    const char* field
) {
    TORCH_CHECK(increment >= 0, field, " must be nonnegative");
    TORCH_CHECK(
        increment <= std::numeric_limits<int64_t>::max() - total,
        field,
        " aggregate overflow"
    );
    return total + increment;
}

inline int64_t exact_nonnegative_receipt_integer_v1(
    const py::dict& receipt,
    const char* field
) {
    TORCH_CHECK(receipt.contains(field), "M0 window receipt lacks ", field);
    const py::handle value = receipt[field];
    TORCH_CHECK(
        PyLong_CheckExact(value.ptr()),
        "M0 window receipt field must be an exact integer: ",
        field
    );
    const int64_t converted = py::cast<int64_t>(value);
    TORCH_CHECK(
        converted >= 0,
        "M0 window receipt field must be nonnegative: ",
        field
    );
    return converted;
}

inline int64_t exact_nonnegative_native_integer_v1(
    const py::dict& receipt,
    const char* field
) {
    TORCH_CHECK(receipt.contains(field), "native window receipt lacks ", field);
    const py::handle value = receipt[field];
    TORCH_CHECK(
        PyLong_CheckExact(value.ptr()),
        "native window receipt field must be an exact integer: ",
        field
    );
    const int64_t converted = py::cast<int64_t>(value);
    TORCH_CHECK(
        converted >= 0,
        "native window receipt field must be nonnegative: ",
        field
    );
    return converted;
}

inline bool exact_receipt_boolean_v1(
    const py::dict& receipt,
    const char* field
) {
    TORCH_CHECK(receipt.contains(field), "M0 window receipt lacks ", field);
    const py::handle value = receipt[field];
    TORCH_CHECK(
        PyBool_Check(value.ptr()),
        "M0 window receipt field must be an exact boolean: ",
        field
    );
    return py::cast<bool>(value);
}

inline std::string exact_receipt_string_v1(
    const py::dict& receipt,
    const char* field
) {
    TORCH_CHECK(receipt.contains(field), "M0 window receipt lacks ", field);
    const py::handle value = receipt[field];
    TORCH_CHECK(
        PyUnicode_CheckExact(value.ptr()),
        "M0 window receipt field must be an exact string: ",
        field
    );
    return py::cast<std::string>(value);
}

inline py::list exact_receipt_list_v1(
    const py::dict& receipt,
    const char* field
) {
    TORCH_CHECK(receipt.contains(field), "M0 window receipt lacks ", field);
    const py::handle value = receipt[field];
    TORCH_CHECK(
        PyList_CheckExact(value.ptr()),
        "M0 window receipt field must be an exact list: ",
        field
    );
    return py::reinterpret_borrow<py::list>(value);
}

inline py::dict exact_receipt_dict_v1(
    const py::dict& receipt,
    const char* field
) {
    TORCH_CHECK(receipt.contains(field), "M0 window receipt lacks ", field);
    const py::handle value = receipt[field];
    TORCH_CHECK(
        PyDict_CheckExact(value.ptr()),
        "M0 window receipt field must be an exact dict: ",
        field
    );
    return py::reinterpret_borrow<py::dict>(value);
}

inline py::dict exact_native_receipt_dict_v1(
    const py::dict& receipt,
    const char* field
) {
    TORCH_CHECK(receipt.contains(field), "native window receipt lacks ", field);
    const py::handle value = receipt[field];
    TORCH_CHECK(
        PyDict_CheckExact(value.ptr()),
        "native window receipt field must be an exact dict: ",
        field
    );
    return py::reinterpret_borrow<py::dict>(value);
}

struct OptionDM0WindowAggregateV1 {
    std::array<int64_t, kM0AdditiveFieldCountV1> additive{};
    std::array<int64_t, kM0RawPolicyFieldCountV1> raw_policy{};
    py::list policy_segment_receipts;
    py::list gaussian_segment_receipts;
    py::list gaussian_segment_boundary_receipts;
    int64_t gaussian_segment_reuse_calls = 0;
    int64_t window_count = 0;
    int64_t terminal_window_count = 0;

    void add_window(const py::dict& receipt, const bool terminal_window) {
        TORCH_CHECK(
            exact_receipt_boolean_v1(
                receipt,
                "option_d_m0_telemetry_enabled"
            ),
            "M0 child window telemetry must be enabled"
        );
        std::array<int64_t, kM0AdditiveFieldCountV1> window_values{};
        for (std::size_t index = 0; index < additive.size(); ++index) {
            window_values[index] = exact_nonnegative_receipt_integer_v1(
                receipt,
                kOptionDM0AdditiveFieldsV1[index]
            );
            std::size_t aggregate_index = index;
            if (!terminal_window) {
                if (index == kM0FinalHostSynchronizationsV1) {
                    aggregate_index = kM0IntermediateHostSynchronizationsV1;
                } else if (index == kM0FinalD2hOperationsV1) {
                    aggregate_index = kM0IntermediateD2hOperationsV1;
                } else if (index == kM0FinalH2dOperationsV1) {
                    aggregate_index = kM0IntermediateH2dOperationsV1;
                }
            }
            additive[aggregate_index] = checked_nonnegative_sum_v1(
                additive[aggregate_index],
                window_values[index],
                kOptionDM0AdditiveFieldsV1[aggregate_index]
            );
        }

        const auto append_segment_receipts = [&] (
            const char* field,
            const int64_t expected_count,
            const char* operation,
            const char* segment_schema,
            const char* segment_boundary,
            const std::initializer_list<
                std::pair<const char*, std::size_t>
            > counter_fields,
            py::list& destination
        ) {
            const py::list source = exact_receipt_list_v1(receipt, field);
            TORCH_CHECK(
                py::len(source) == expected_count,
                field,
                " length differs from its call counter"
            );
            std::vector<int64_t> nested_totals(counter_fields.size(), 0);
            for (const py::handle item : source) {
                TORCH_CHECK(
                    PyDict_CheckExact(item.ptr()),
                    field,
                    " entries must be exact dicts"
                );
                const auto item_receipt =
                    py::reinterpret_borrow<py::dict>(item);
                const std::string observed_operation = exact_receipt_string_v1(
                    item_receipt,
                    "operation"
                );
                const bool gaussian_boundary_reuse =
                    std::string(field) == "gaussian_segment_boundary_receipts"
                    && observed_operation
                        == "reuse_cpython_gaussian_window_v1";
                TORCH_CHECK(
                    exact_receipt_boolean_v1(
                        item_receipt,
                        "option_d_m0_telemetry_enabled"
                    ),
                    field,
                    " entry telemetry must be enabled"
                );
                TORCH_CHECK(
                    exact_receipt_string_v1(item_receipt, "schema_version")
                        == "eiel_cuda_architecture_b_native_resident_window_"
                           "refill_v1"
                    && (observed_operation == operation || gaussian_boundary_reuse)
                    && exact_receipt_string_v1(
                        item_receipt,
                        "segment_schema"
                    ) == segment_schema
                    && exact_receipt_string_v1(
                        item_receipt,
                        "segment_boundary"
                    ) == segment_boundary,
                    field,
                    " entry schema or operation is invalid"
                );
                if (
                    std::string(field) == "gaussian_segment_boundary_receipts"
                ) {
                    TORCH_CHECK(
                        exact_receipt_boolean_v1(item_receipt, "refill_performed")
                            == !gaussian_boundary_reuse,
                        field,
                        " refill marker differs from its operation"
                    );
                }
                (void) exact_nonnegative_receipt_integer_v1(
                    item_receipt,
                    "generation"
                );
                (void) exact_nonnegative_receipt_integer_v1(
                    item_receipt,
                    "completed_tick"
                );
                (void) exact_nonnegative_receipt_integer_v1(
                    item_receipt,
                    "next_tick"
                );
                std::size_t counter_index = 0;
                for (const auto& [nested_field, aggregate_index] : counter_fields) {
                    (void) aggregate_index;
                    nested_totals[counter_index] = checked_nonnegative_sum_v1(
                        nested_totals[counter_index],
                        exact_nonnegative_receipt_integer_v1(
                            item_receipt,
                            nested_field
                        ),
                        nested_field
                    );
                    ++counter_index;
                }
                destination.append(item);
            }
            std::size_t counter_index = 0;
            for (const auto& [nested_field, aggregate_index] : counter_fields) {
                TORCH_CHECK(
                    nested_totals[counter_index] == window_values[aggregate_index],
                    field,
                    " nested total differs for ",
                    nested_field
                );
                ++counter_index;
            }
        };
        append_segment_receipts(
            "policy_birth_segment_refill_receipts",
            window_values[kM0PolicyRefillCallsV1],
            "refill_fixed_mlp_policy_window_v1",
            "eiel_cuda_architecture_b_policy_birth_segment_refill_v1",
            "native_birth_boundary_v1",
            {
                {"device_to_host_operations_lower_bound", kM0PolicyD2hOperationsV1},
                {"host_to_device_operations_lower_bound", kM0PolicyH2dOperationsV1},
                {"host_wall_time_ns", kM0PolicyHostWallTimeV1},
                {"device_to_host_time_ns", kM0PolicyD2hTimeV1},
                {"host_to_device_time_ns", kM0PolicyH2dTimeV1},
                {"torch_random_generation_time_ns", kM0PolicyTorchRandomTimeV1},
                {"torch_compare_pack_time_ns", kM0PolicyComparePackTimeV1},
                {"policy_allocation_stack_copy_time_ns", kM0PolicyAllocationCopyTimeV1},
            },
            policy_segment_receipts
        );
        const int64_t gaussian_reuse_calls = exact_nonnegative_receipt_integer_v1(
            receipt,
            "gaussian_segment_reuse_calls_inside_tick_loop"
        );
        gaussian_segment_reuse_calls = checked_nonnegative_sum_v1(
            gaussian_segment_reuse_calls,
            gaussian_reuse_calls,
            "gaussian segment reuse calls"
        );
        append_segment_receipts(
            "gaussian_segment_boundary_receipts",
            checked_nonnegative_sum_v1(
                window_values[kM0GaussianRefillCallsV1],
                gaussian_reuse_calls,
                "gaussian segment boundary calls"
            ),
            "refill_cpython_gaussian_window_v1",
            "host_prepared_exact_gaussian_segment_v1",
            "native_tick_boundary_v1",
            {
                {"device_to_host_operations_lower_bound", kM0GaussianD2hOperationsV1},
                {"host_to_device_operations_lower_bound", kM0GaussianH2dOperationsV1},
                {"host_wall_time_ns", kM0GaussianHostWallTimeV1},
                {"device_to_host_time_ns", kM0GaussianD2hTimeV1},
                {"host_to_device_time_ns", kM0GaussianH2dTimeV1},
                {"gaussian_raw_generation_time_ns", kM0GaussianRawGenerationTimeV1},
                {"gaussian_transcendental_conversion_time_ns", kM0GaussianTranscendentalTimeV1},
                {"gaussian_allocation_copy_time_ns", kM0GaussianAllocationCopyTimeV1},
            },
            gaussian_segment_boundary_receipts
        );
        append_segment_receipts(
            "gaussian_segment_refill_receipts",
            window_values[kM0GaussianRefillCallsV1],
            "refill_cpython_gaussian_window_v1",
            "host_prepared_exact_gaussian_segment_v1",
            "native_tick_boundary_v1",
            {},
            gaussian_segment_receipts
        );
        TORCH_CHECK(
            exact_receipt_boolean_v1(
                receipt,
                "native_host_policy_birth_segment_refill_used"
            ) == (window_values[kM0PolicyRefillCallsV1] > 0),
            "M0 policy segment-refill marker differs from its call counter"
        );
        TORCH_CHECK(
            exact_receipt_boolean_v1(
                receipt,
                "native_host_gaussian_segment_refill_used"
            ) == (window_values[kM0GaussianRefillCallsV1] > 0),
            "M0 Gaussian segment-refill marker differs from its call counter"
        );
        const int64_t service_calls =
            window_values[kM0ExactCpuPolicyServiceCallsV1];
        TORCH_CHECK(
            exact_receipt_boolean_v1(
                receipt,
                "native_host_policy_service_used"
            ) == (service_calls > 0),
            "M0 CPU policy-service marker differs from its call counter"
        );
        const std::string expected_aten_semantics = service_calls > 0
            ? "lower_bound_from_explicit_native_host_transfers"
            : "exact_zero";
        TORCH_CHECK(
            exact_receipt_string_v1(
                receipt,
                "aten_operations_inside_tick_loop_count_semantics"
            ) == expected_aten_semantics,
            "M0 ATen operation count semantics differ from policy-service use"
        );

        const py::dict raw = exact_receipt_dict_v1(
            receipt,
            "option_d_m0_exact_cpu_policy_service_raw_metrics"
        );
        TORCH_CHECK(
            py::len(raw)
                == static_cast<Py_ssize_t>(
                    kM0RawPolicyFieldCountV1
                    + kOptionDM0RawPolicyScopeFieldsV1.size()
                ),
            "M0 CPU policy raw metric field set differs from its contract"
        );
        std::array<int64_t, kM0RawPolicyFieldCountV1> window_raw{};
        for (std::size_t index = 0; index < raw_policy.size(); ++index) {
            window_raw[index] = exact_nonnegative_receipt_integer_v1(
                raw,
                kOptionDM0RawPolicyFieldsV1[index]
            );
            raw_policy[index] = checked_nonnegative_sum_v1(
                raw_policy[index],
                window_raw[index],
                kOptionDM0RawPolicyFieldsV1[index]
            );
        }
        for (
            std::size_t index = 0;
            index < kOptionDM0RawPolicyScopeFieldsV1.size();
            ++index
        ) {
            TORCH_CHECK(
                exact_receipt_string_v1(
                    raw,
                    kOptionDM0RawPolicyScopeFieldsV1[index]
                ) == kOptionDM0RawPolicyScopesV1[index],
                "M0 CPU policy raw metric scope differs from its contract"
            );
        }
        TORCH_CHECK(
            window_raw[kM0RawCallsV1] == service_calls,
            "M0 CPU policy raw call count differs from native call count"
        );
        TORCH_CHECK(
            window_values[kM0AtenOperationsInsideTickLoopV1]
                == checked_nonnegative_sum_v1(
                    window_raw[kM0RawD2hOperationsV1],
                    window_raw[kM0RawH2dOperationsV1],
                    "M0 CPU policy raw transfer operations"
                ),
            "M0 ATen lower bound differs from CPU policy transfer operations"
        );
        TORCH_CHECK(
            window_raw[kM0RawExplicitFinalSyncsV1] == service_calls,
            "M0 CPU policy explicit synchronization count differs from calls"
        );
        TORCH_CHECK(
            window_raw[kM0RawImplicitD2hWaitsV1]
                == window_raw[kM0RawD2hOperationsV1],
            "M0 CPU policy implicit wait count differs from D2H operations"
        );
        window_count = checked_nonnegative_sum_v1(
            window_count,
            1,
            "M0 settled window count"
        );
        if (terminal_window) {
            terminal_window_count = checked_nonnegative_sum_v1(
                terminal_window_count,
                1,
                "M0 terminal window count"
            );
        }
    }

    void add_outer_intermediate_transfers(
        const int64_t device_to_host_operations,
        const int64_t host_to_device_operations
    ) {
        additive[kM0IntermediateD2hOperationsV1] = checked_nonnegative_sum_v1(
            additive[kM0IntermediateD2hOperationsV1],
            device_to_host_operations,
            "M0 intermediate device-to-host operations"
        );
        additive[kM0IntermediateH2dOperationsV1] = checked_nonnegative_sum_v1(
            additive[kM0IntermediateH2dOperationsV1],
            host_to_device_operations,
            "M0 intermediate host-to-device operations"
        );
    }

    void add_outer_final_device_to_host_operations(const int64_t operations) {
        additive[kM0FinalD2hOperationsV1] = checked_nonnegative_sum_v1(
            additive[kM0FinalD2hOperationsV1],
            operations,
            "M0 final device-to-host operations"
        );
    }

    void publish(py::dict& out) const {
        TORCH_CHECK(window_count > 0, "M0 telemetry has no settled child windows");
        TORCH_CHECK(
            terminal_window_count == 1,
            "M0 telemetry must contain exactly one route-terminal child window"
        );
        out["option_d_m0_telemetry_enabled"] = true;
        for (std::size_t index = 0; index < additive.size(); ++index) {
            out[kOptionDM0AdditiveFieldsV1[index]] = additive[index];
        }
        out["policy_birth_segment_refill_receipts"] =
            policy_segment_receipts;
        out["gaussian_segment_refill_receipts"] =
            gaussian_segment_receipts;
        out["gaussian_segment_boundary_receipts"] =
            gaussian_segment_boundary_receipts;
        out["gaussian_segment_reuse_calls_inside_tick_loop"] =
            gaussian_segment_reuse_calls;
        out["native_host_policy_birth_segment_refill_used"] =
            additive[kM0PolicyRefillCallsV1] > 0;
        out["native_host_gaussian_segment_refill_used"] =
            additive[kM0GaussianRefillCallsV1] > 0;
        out["native_host_policy_service_used"] =
            additive[kM0ExactCpuPolicyServiceCallsV1] > 0;
        out["aten_operations_inside_tick_loop_count_semantics"] =
            additive[kM0ExactCpuPolicyServiceCallsV1] > 0
                ? "lower_bound_from_explicit_native_host_transfers"
                : "exact_zero";
        py::dict raw;
        for (std::size_t index = 0; index < raw_policy.size(); ++index) {
            raw[kOptionDM0RawPolicyFieldsV1[index]] = raw_policy[index];
        }
        for (
            std::size_t index = 0;
            index < kOptionDM0RawPolicyScopeFieldsV1.size();
            ++index
        ) {
            raw[kOptionDM0RawPolicyScopeFieldsV1[index]] =
                kOptionDM0RawPolicyScopesV1[index];
        }
        out["option_d_m0_exact_cpu_policy_service_raw_metrics"] =
            std::move(raw);
    }
};

template <typename SubmitWindowV1>
inline py::dict execute_resident_windows_core_v1(
    SubmitWindowV1&& submit_window,
    const int64_t initial_generation,
    const int64_t physical_window_generations,
    const int64_t logical_generation_count,
    const torch::Tensor& generation_summary_scalar,
    const torch::Tensor& runtime_bookkeeping_by_tick,
    const torch::Tensor& patch_stats,
    const std::vector<torch::Tensor>& event_state_tensors,
    const std::vector<torch::Tensor>& ordinary_birth_state_tensors,
    const torch::Tensor& controller_status,
    const torch::Tensor& lifecycle_status,
    const torch::Tensor& turnover_counts,
    const torch::Tensor& turnover_status,
    const int64_t policy_row_width,
    const bool policy_weight_mutation_enable,
    const std::string& policy_weight_mutation_mode,
    const double policy_weight_mutation_rate,
    const double policy_weight_mutation_std,
    const bool internal_pybind_redispatch,
    const bool direct_cpp_window_core_dispatch,
    const bool production_native_chunk_loop,
    const bool allow_partial_window_completion = false,
    window_refill::PolicyWindowIdentityStateV1*
        policy_window_identity_state = nullptr,
    const std::string& event_history_sink_path = ""
) {
    TORCH_CHECK(
        physical_window_generations >= kMinimumPhysicalWindowGenerationsV1
            && physical_window_generations <= kMaximumPhysicalWindowGenerationsV1,
        "bounded native resident dispatch requires physical g1 or g2 windows"
    );
    TORCH_CHECK(
        policy_row_width > 0,
        "resident policy row width must be positive"
    );
    TORCH_CHECK(
        logical_generation_count >= 1
            && logical_generation_count <= kMaximumLogicalGenerationsV1,
        "logical generations must be in [1,128]"
    );
    TORCH_CHECK(initial_generation >= 0, "initial generation must be nonnegative");
    TORCH_CHECK(
        initial_generation
            <= std::numeric_limits<int64_t>::max() - logical_generation_count,
        "terminal generation would overflow int64"
    );
    // Keep physical windows aligned to the absolute-generation lattice.  A
    // continuation at an odd generation must first consume the one-generation
    // suffix of the resident g2 window before returning to complete g2
    // windows.  Chunking relative to the resumed prefix changes which half of
    // the precomputed native tapes a generation consumes.
    const int64_t first_window_generations = std::min(
        physical_window_generations
            - initial_generation % physical_window_generations,
        logical_generation_count
    );
    const int64_t remaining_after_first =
        logical_generation_count - first_window_generations;
    const int64_t window_count = 1
        + (remaining_after_first + physical_window_generations - 1)
            / physical_window_generations;
    TORCH_CHECK(
        event_state_tensors.size() == 31,
        "resident event state must contain the exact 31-tensor ABI"
    );
    TORCH_CHECK(
        ordinary_birth_state_tensors.size() == 113
            || ordinary_birth_state_tensors.size() == 122
            || ordinary_birth_state_tensors.size() == 139,
        "resident ordinary-birth state must contain the exact 113 legacy, "
        "122 diploid, or 139 Genome V2 tensor ABI"
    );
    TORCH_CHECK(
        std::isfinite(policy_weight_mutation_rate)
            && std::isfinite(policy_weight_mutation_std),
        "resident policy refill settings must be finite"
    );
    TORCH_CHECK(
        generation_summary_scalar.scalar_type() == torch::kFloat64
            && generation_summary_scalar.dim() == 3
            && generation_summary_scalar.size(0) == physical_window_generations
            && generation_summary_scalar.size(2) == flattened_controller_v1::kGenerationSummaryScalarWidthV1,
        "resident scalar summary must be float64[physical_window_generations,cells,31]"
    );
    const int64_t cells = generation_summary_scalar.size(1);
    TORCH_CHECK(
        runtime_bookkeeping_by_tick.dim() == 3
            && runtime_bookkeeping_by_tick.size(1) % physical_window_generations == 0,
        "resident runtime bookkeeping must contain complete per-generation ticks"
    );
    const int64_t ticks_per_generation =
        runtime_bookkeeping_by_tick.size(1) / physical_window_generations;
    const bool exact_partial_tick_window =
        allow_partial_window_completion
        && physical_window_generations == 1
        && ticks_per_generation > 0
        && ticks_per_generation < 160;
    TORCH_CHECK(
        ticks_per_generation == 80 || ticks_per_generation == 160
            || exact_partial_tick_window,
        "resident physical window ticks must be a full 80/160 interval or "
        "one bounded partial interval"
    );
    TORCH_CHECK(
        runtime_bookkeeping_by_tick.scalar_type() == torch::kInt64
            && runtime_bookkeeping_by_tick.sizes()
                == torch::IntArrayRef({
                    cells,
                    physical_window_generations * ticks_per_generation,
                    5,
                }),
        "resident runtime bookkeeping must be int64[cells,physical_ticks,5]"
    );
    TORCH_CHECK(
        patch_stats.scalar_type() == torch::kFloat64
            && patch_stats.sizes() == torch::IntArrayRef({
                cells,
                physical_window_generations * ticks_per_generation,
                25,
            }),
        "resident Patch stats must be float64[cells,physical_ticks,25]"
    );
    TORCH_CHECK(
        turnover_counts.scalar_type() == torch::kInt64
            && turnover_counts.dim() == 2
            && turnover_counts.size(0) == cells
            && turnover_counts.size(1) == 8,
        "resident turnover counts output must be int64[cells,8]"
    );
    const auto event_status = event_state_tensors[kEventStatusV1];

    std::vector<torch::Tensor> event_columns(
        event_state_tensors.begin(),
        event_state_tensors.begin() + event_archive::kEventColumnCountV1
    );
    auto event_active_count = event_state_tensors[kEventActiveCountV1];
    auto event_sequence_base = event_state_tensors[kEventSequenceBaseV1];
    event_archive::EventArchiveV1 archived_events;
    std::optional<event_archive::EventSinkV1> external_event_sink;
    if (!event_history_sink_path.empty()) {
        external_event_sink.emplace(event_history_sink_path, cells);
    }
    const auto cell_scalar = [cells](
                                 const torch::Tensor& value,
                                 const char* name
                             ) {
        TORCH_CHECK(
            value.defined() && value.numel() == cells,
            name,
            " must contain exactly one scalar per cell"
        );
        return value.reshape({cells});
    };

    std::vector<torch::Tensor> summary_chunks;
    std::vector<torch::Tensor> bookkeeping_chunks;
    std::vector<torch::Tensor> patch_chunks;
    std::vector<torch::Tensor> turnover_count_chunks;
    summary_chunks.reserve(static_cast<std::size_t>(window_count));
    bookkeeping_chunks.reserve(static_cast<std::size_t>(window_count));
    patch_chunks.reserve(static_cast<std::size_t>(window_count));
    turnover_count_chunks.reserve(static_cast<std::size_t>(window_count));
    py::list window_receipts;
    py::list turnover_count_terminal_generations;
    py::list event_drain_receipts;
    py::list gaussian_refill_receipts;
    py::list gaussian_segment_refill_receipts;
    py::list gaussian_segment_boundary_receipts;
    py::list policy_birth_segment_refill_receipts;
    py::list policy_refill_receipts;
    py::list policy_birth_identity_entry_ensure_receipts;
    py::dict terminal_window_receipt;
    bool terminal_window_receipt_observed = false;
    py::dict resume_gaussian_refill_receipt;
    py::dict resume_policy_refill_receipt;
    bool first_policy_identity_entry_physical_refill = false;
    int64_t gaussian_segment_refill_calls_inside_tick_loop = 0;
    int64_t gaussian_segment_reuse_calls_inside_tick_loop = 0;
    int64_t gaussian_segment_device_to_host_operations_lower_bound = 0;
    int64_t gaussian_segment_host_to_device_operations_lower_bound = 0;
    int64_t gaussian_segment_host_wall_time_ns = 0;
    int64_t gaussian_segment_device_to_host_time_ns = 0;
    int64_t gaussian_segment_host_to_device_time_ns = 0;
    int64_t gaussian_segment_raw_generation_time_ns = 0;
    int64_t gaussian_segment_transcendental_conversion_time_ns = 0;
    int64_t gaussian_segment_allocation_copy_time_ns = 0;
    int64_t policy_birth_segment_host_wall_time_ns = 0;
    int64_t policy_birth_segment_device_to_host_time_ns = 0;
    int64_t policy_birth_segment_host_to_device_time_ns = 0;
    int64_t policy_birth_segment_torch_random_generation_time_ns = 0;
    int64_t policy_birth_segment_torch_compare_pack_time_ns = 0;
    int64_t policy_birth_segment_allocation_stack_copy_time_ns = 0;
    int64_t policy_birth_identity_ensure_calls_total = 0;
    int64_t policy_birth_identity_materialization_calls_total = 0;
    int64_t policy_birth_identity_physical_refill_calls_total = 0;
    int64_t policy_birth_identity_resident_suffix_reuse_calls_total = 0;
    int64_t policy_birth_identity_device_to_host_operations_total = 0;
    int64_t policy_birth_identity_host_to_device_operations_total = 0;
    int64_t policy_birth_identity_device_to_host_bytes_total = 0;
    int64_t policy_birth_identity_host_to_device_bytes_total = 0;
    int64_t policy_birth_identity_host_wall_time_ns_total = 0;
    int64_t policy_birth_identity_device_to_host_time_ns_total = 0;
    int64_t policy_birth_identity_host_to_device_time_ns_total = 0;
    int64_t intermediate_device_to_host_operations = 0;
    int64_t intermediate_host_to_device_operations = 0;
    std::optional<OptionDM0WindowAggregateV1> option_d_m0_aggregate;
    int64_t option_d_m0_outer_boundary_d2h_operations = 0;
    int64_t option_d_m0_outer_event_d2h_operations = 0;
    int64_t option_d_m0_outer_refill_d2h_operations = 0;
    int64_t option_d_m0_outer_refill_h2d_operations = 0;
    if constexpr (window_refill::kOptionDM0TelemetryEnabledV1) {
        option_d_m0_aggregate.emplace();
    }

    const auto record_option_d_m0_outer_refill = [&] (
        const py::dict& receipt,
        const char* expected_operation
    ) {
        if constexpr (window_refill::kOptionDM0TelemetryEnabledV1) {
            TORCH_CHECK(
                exact_receipt_boolean_v1(
                    receipt,
                    "option_d_m0_telemetry_enabled"
                ),
                "M0 outer refill telemetry must be enabled"
            );
            TORCH_CHECK(
                exact_receipt_string_v1(receipt, "schema_version")
                    == "eiel_cuda_architecture_b_native_resident_window_"
                       "refill_v1"
                && exact_receipt_string_v1(receipt, "operation")
                    == expected_operation,
                "M0 outer refill schema or operation is invalid"
            );
            const int64_t d2h = exact_nonnegative_receipt_integer_v1(
                receipt,
                "device_to_host_operations_lower_bound"
            );
            const int64_t h2d = exact_nonnegative_receipt_integer_v1(
                receipt,
                "host_to_device_operations_lower_bound"
            );
            option_d_m0_outer_refill_d2h_operations =
                checked_nonnegative_sum_v1(
                    option_d_m0_outer_refill_d2h_operations,
                    d2h,
                    "M0 outer refill D2H operations"
                );
            option_d_m0_outer_refill_h2d_operations =
                checked_nonnegative_sum_v1(
                    option_d_m0_outer_refill_h2d_operations,
                    h2d,
                    "M0 outer refill H2D operations"
                );
            option_d_m0_aggregate->add_outer_intermediate_transfers(
                d2h,
                h2d
            );
        }
    };

    const auto accumulate_policy_identity_receipt = [&] (
        const py::dict& receipt,
        const bool is_materialization
    ) {
        if (is_materialization) {
            ++policy_birth_identity_materialization_calls_total;
            ++policy_birth_identity_physical_refill_calls_total;
        }
        const auto add_if_present = [&] (
            const char* field,
            int64_t* const total
        ) {
            if (receipt.contains(field)) {
                *total += py::cast<int64_t>(receipt[field]);
            }
        };
        add_if_present(
            "device_to_host_operations_lower_bound",
            &policy_birth_identity_device_to_host_operations_total
        );
        add_if_present(
            "host_to_device_operations_lower_bound",
            &policy_birth_identity_host_to_device_operations_total
        );
        add_if_present(
            "device_to_host_bytes_lower_bound",
            &policy_birth_identity_device_to_host_bytes_total
        );
        add_if_present(
            "host_to_device_bytes_lower_bound",
            &policy_birth_identity_host_to_device_bytes_total
        );
        add_if_present(
            "host_wall_time_ns",
            &policy_birth_identity_host_wall_time_ns_total
        );
        add_if_present(
            "device_to_host_time_ns",
            &policy_birth_identity_device_to_host_time_ns_total
        );
        add_if_present(
            "host_to_device_time_ns",
            &policy_birth_identity_host_to_device_time_ns_total
        );
    };

    const auto refill_gaussian_window = [&]() {
        const auto& primary_tape =
            ordinary_birth_state_tensors[kPythonRandomPrimaryTapeV1];
        if (
            !primary_tape.defined()
            || primary_tape.dim() != 2
            || primary_tape.size(1) == 0
        ) {
            // V2 tape-free births keep Gauss capacity 0. Do not host-refill.
            py::dict skipped;
            skipped["schema_version"] =
                "eiel_cuda_architecture_b_native_resident_window_refill_v1";
            skipped["operation"] = "refill_cpython_gaussian_window_v1";
            skipped["skipped"] = true;
            skipped["reason"] = "v2_tape_free_zero_gauss_capacity";
            skipped["physical_capacity"] = 0;
            return skipped;
        }
        return window_refill::refill_cpython_gaussian_window_v1(
            ordinary_birth_state_tensors[kPythonRandomWordsV1],
            cell_scalar(
                ordinary_birth_state_tensors[kPythonRandomIndexV1],
                "python_random_index"
            ),
            cell_scalar(
                ordinary_birth_state_tensors[kPythonRandomHasGaussV1],
                "python_random_has_gauss"
            ),
            cell_scalar(
                ordinary_birth_state_tensors[kPythonRandomGaussNextV1],
                "python_random_gauss_next"
            ),
            ordinary_birth_state_tensors[kPythonRandomPrimaryTapeV1],
            ordinary_birth_state_tensors[kPythonRandomSecondaryTapeV1],
            cell_scalar(
                ordinary_birth_state_tensors[kPythonRandomRawDrawsV1],
                "python_random_raw_draws"
            ),
            cell_scalar(
                ordinary_birth_state_tensors[kMutationPySemanticDrawsV1],
                "mutation_py_semantic_draws"
            ),
            cell_scalar(
                ordinary_birth_state_tensors[kPythonRandomWindowBaseV1],
                "python_random_window_base"
            )
        );
    };
    const auto refill_policy_window = [&]() {
        return window_refill::refill_fixed_mlp_policy_window_with_identity_v1(
            ordinary_birth_state_tensors[kPolicyRngStateV1],
            ordinary_birth_state_tensors[kPolicyDeltaTapeV1],
            ordinary_birth_state_tensors[kPolicyMutationMaskTapeV1],
            ordinary_birth_state_tensors[kPolicyCrossoverMaskPackedTapeV1],
            ordinary_birth_state_tensors[kPolicyPackedPostStateTapeV1],
            ordinary_birth_state_tensors[kPolicySemanticTapeV1],
            cell_scalar(
                ordinary_birth_state_tensors[kPolicyAbsoluteCursorV1],
                "policy_birth_tape_cursor"
            ),
            cell_scalar(
                ordinary_birth_state_tensors[kMutationTorchSemanticDrawsV1],
                "mutation_torch_cpu_semantic_draws"
            ),
            cell_scalar(
                ordinary_birth_state_tensors[kPolicyWindowBaseV1],
                "policy_birth_window_base"
            ),
            policy_row_width,
            policy_weight_mutation_enable,
            policy_weight_mutation_mode,
            policy_weight_mutation_rate,
            policy_weight_mutation_std,
            window_refill::kFragIdentityProducerSchemaSha256V1,
            policy_window_identity_state
        );
    };
    // An aligned fresh owner starts after a fully consumed physical window and
    // must refill.  For an odd fresh owner, source preparation has regenerated
    // the exact remaining Gaussian and policy sequence from the checkpoint RNG
    // state and anchored it at the checkpoint's absolute counters. Refilling
    // here would skip one absolute generation of RNG state.
    const bool reconstructed_window_suffix =
        initial_generation > 0
        && initial_generation % physical_window_generations != 0;
    const bool resume_boundary_refill =
        initial_generation > 0 && !reconstructed_window_suffix;
    if (resume_boundary_refill) {
        resume_gaussian_refill_receipt = refill_gaussian_window();
        resume_policy_refill_receipt = refill_policy_window();
        record_option_d_m0_outer_refill(
            resume_gaussian_refill_receipt,
            "refill_cpython_gaussian_window_v1"
        );
        accumulate_policy_identity_receipt(
            resume_policy_refill_receipt, true
        );
        record_option_d_m0_outer_refill(
            resume_policy_refill_receipt,
            "refill_fixed_mlp_policy_window_v1"
        );
    }

    int64_t completed_generations = 0;
    for (int64_t window_index = 0; window_index < window_count; ++window_index) {
        const int64_t generation_base = initial_generation + completed_generations;
        const int64_t requested_window_generations = std::min(
            completed_generations == 0
                ? first_window_generations
                : physical_window_generations,
            logical_generation_count - completed_generations
        );
        const auto receipt = submit_window(
            generation_base,
            requested_window_generations
        );
        if (
            window_index == 0
            && receipt.contains(
                "policy_birth_identity_entry_physical_refill_performed"
            )
        ) {
            const py::handle value = receipt[
                "policy_birth_identity_entry_physical_refill_performed"
            ];
            TORCH_CHECK(
                PyBool_Check(value.ptr()),
                "policy identity entry-refill marker must be an exact boolean"
            );
            first_policy_identity_entry_physical_refill =
                py::cast<bool>(value);
        }
        if (receipt.contains("policy_birth_identity_entry_ensure_receipt")) {
            const auto entry_receipt = py::cast<py::dict>(
                receipt["policy_birth_identity_entry_ensure_receipt"]
            );
            if (py::len(entry_receipt) > 0) {
                policy_birth_identity_entry_ensure_receipts.append(
                    entry_receipt
                );
            }
        }
        const auto add_child_identity_total = [&] (
            const char* field,
            int64_t* const total
        ) {
            if (receipt.contains(field)) {
                *total += py::cast<int64_t>(receipt[field]);
            }
        };
        add_child_identity_total(
            "policy_birth_identity_ensure_calls_total",
            &policy_birth_identity_ensure_calls_total
        );
        add_child_identity_total(
            "policy_birth_identity_physical_refill_calls_total",
            &policy_birth_identity_physical_refill_calls_total
        );
        add_child_identity_total(
            "policy_birth_identity_resident_suffix_reuse_calls_total",
            &policy_birth_identity_resident_suffix_reuse_calls_total
        );
        add_child_identity_total(
            "policy_birth_identity_device_to_host_operations_lower_bound",
            &policy_birth_identity_device_to_host_operations_total
        );
        add_child_identity_total(
            "policy_birth_identity_host_to_device_operations_lower_bound",
            &policy_birth_identity_host_to_device_operations_total
        );
        add_child_identity_total(
            "policy_birth_identity_device_to_host_bytes_lower_bound",
            &policy_birth_identity_device_to_host_bytes_total
        );
        add_child_identity_total(
            "policy_birth_identity_host_to_device_bytes_lower_bound",
            &policy_birth_identity_host_to_device_bytes_total
        );
        add_child_identity_total(
            "policy_birth_identity_host_wall_time_ns",
            &policy_birth_identity_host_wall_time_ns_total
        );
        add_child_identity_total(
            "policy_birth_identity_device_to_host_time_ns",
            &policy_birth_identity_device_to_host_time_ns_total
        );
        add_child_identity_total(
            "policy_birth_identity_host_to_device_time_ns",
            &policy_birth_identity_host_to_device_time_ns_total
        );
        const int64_t receipt_requested_window_generations =
            exact_nonnegative_native_integer_v1(
                receipt,
                "requested_generations"
            );
        int64_t completed_window_generations =
            exact_nonnegative_native_integer_v1(
                receipt,
                "completed_generations"
            );
        if (receipt.contains("native_run_control_v2_next_generation")) {
            const int64_t run_control_next_generation =
                exact_nonnegative_native_integer_v1(
                    receipt,
                    "native_run_control_v2_next_generation"
                );
            TORCH_CHECK(
                allow_partial_window_completion
                    && run_control_next_generation >= generation_base
                    && run_control_next_generation
                        <= generation_base + requested_window_generations,
                "native run-control next_generation is outside the requested window"
            );
            completed_window_generations =
                run_control_next_generation - generation_base;
        }
        bool stop_after_completed_prefix = false;
        if (receipt.contains(
                "native_run_control_v2_stop_after_completed_prefix"
            )) {
            const py::handle stop_value =
                receipt["native_run_control_v2_stop_after_completed_prefix"];
            TORCH_CHECK(
                PyBool_Check(stop_value.ptr()),
                "native run-control stop marker must be an exact boolean"
            );
            stop_after_completed_prefix = py::cast<bool>(stop_value);
        }
        TORCH_CHECK(
            receipt_requested_window_generations
                == requested_window_generations,
            "native physical window changed its requested generation count"
        );
        require_zero_int32_status_v1(
            controller_status,
            "controller status",
            &receipt
        );
        require_zero_int32_status_v1(event_status, "event status");
        require_zero_int32_status_v1(lifecycle_status, "lifecycle status");
        require_zero_turnover_status_v1(turnover_status);
        TORCH_CHECK(
            completed_window_generations == requested_window_generations
                || (
                    allow_partial_window_completion
                    && completed_window_generations > 0
                    && completed_window_generations < requested_window_generations
            ),
            "native physical window did not return an accepted complete prefix"
        );
        const bool route_terminal_window =
            completed_window_generations < requested_window_generations
            || stop_after_completed_prefix
            || window_index + 1 >= window_count;
        if (route_terminal_window) {
            TORCH_CHECK(
                !terminal_window_receipt_observed,
                "resident route observed more than one terminal window receipt"
            );
            terminal_window_receipt = py::dict(receipt);
            terminal_window_receipt_observed = true;
        }

        summary_chunks.push_back(immutable_host_copy_v1(
            generation_summary_scalar.narrow(
                0,
                0,
                completed_window_generations
            )
        ));
        bookkeeping_chunks.push_back(
            immutable_host_copy_v1(runtime_bookkeeping_by_tick.narrow(
                1,
                0,
                completed_window_generations * ticks_per_generation
            ))
        );
        patch_chunks.push_back(immutable_host_copy_v1(patch_stats.narrow(
            1,
            0,
            completed_window_generations * ticks_per_generation
        )));
        turnover_count_chunks.push_back(immutable_host_copy_v1(turnover_counts));
        turnover_count_terminal_generations.append(
            py::int_(generation_base + completed_window_generations - 1)
        );
        if constexpr (window_refill::kOptionDM0TelemetryEnabledV1) {
            int64_t boundary_d2h_operations = 0;
            for (const torch::Tensor* value : {
                     &controller_status,
                     &event_status,
                     &lifecycle_status,
                     &turnover_status,
                     &generation_summary_scalar,
                     &runtime_bookkeeping_by_tick,
                     &patch_stats,
                 }) {
                if (value->is_cuda() && value->numel() > 0) {
                    boundary_d2h_operations = checked_nonnegative_sum_v1(
                        boundary_d2h_operations,
                        1,
                        "M0 outer boundary D2H operations"
                    );
                }
            }
            option_d_m0_outer_boundary_d2h_operations =
                checked_nonnegative_sum_v1(
                    option_d_m0_outer_boundary_d2h_operations,
                    boundary_d2h_operations,
                    "M0 outer boundary D2H operations"
                );
            if (route_terminal_window) {
                option_d_m0_aggregate
                    ->add_outer_final_device_to_host_operations(
                        boundary_d2h_operations
                    );
            } else {
                option_d_m0_aggregate->add_outer_intermediate_transfers(
                    boundary_d2h_operations,
                    0
                );
            }
            option_d_m0_aggregate->add_window(
                receipt,
                route_terminal_window
            );
        }
        if (receipt.contains("gaussian_segment_refill_calls_inside_tick_loop")) {
            gaussian_segment_refill_calls_inside_tick_loop += py::cast<int64_t>(
                receipt["gaussian_segment_refill_calls_inside_tick_loop"]
            );
        }
        if (receipt.contains("gaussian_segment_refill_receipts")) {
            for (
                const py::handle item :
                py::cast<py::list>(receipt["gaussian_segment_refill_receipts"])
            ) {
                gaussian_segment_refill_receipts.append(item);
            }
        }
        if (receipt.contains("gaussian_segment_boundary_receipts")) {
            for (
                const py::handle item :
                py::cast<py::list>(receipt["gaussian_segment_boundary_receipts"])
            ) {
                gaussian_segment_boundary_receipts.append(item);
            }
        }
        if (receipt.contains("policy_birth_segment_refill_receipts")) {
            for (
                const py::handle item :
                py::cast<py::list>(
                    receipt["policy_birth_segment_refill_receipts"]
                )
            ) {
                policy_birth_segment_refill_receipts.append(item);
            }
        }
        if (receipt.contains("gaussian_segment_reuse_calls_inside_tick_loop")) {
            gaussian_segment_reuse_calls_inside_tick_loop += py::cast<int64_t>(
                receipt["gaussian_segment_reuse_calls_inside_tick_loop"]
            );
        }
        if (receipt.contains("gaussian_segment_device_to_host_operations_lower_bound")) {
            gaussian_segment_device_to_host_operations_lower_bound += py::cast<int64_t>(
                receipt["gaussian_segment_device_to_host_operations_lower_bound"]
            );
        }
        if (receipt.contains("gaussian_segment_host_to_device_operations_lower_bound")) {
            gaussian_segment_host_to_device_operations_lower_bound += py::cast<int64_t>(
                receipt["gaussian_segment_host_to_device_operations_lower_bound"]
            );
        }
        if (receipt.contains("gaussian_segment_host_wall_time_ns")) {
            gaussian_segment_host_wall_time_ns += py::cast<int64_t>(
                receipt["gaussian_segment_host_wall_time_ns"]
            );
        }
        if (receipt.contains("gaussian_segment_device_to_host_time_ns")) {
            gaussian_segment_device_to_host_time_ns += py::cast<int64_t>(
                receipt["gaussian_segment_device_to_host_time_ns"]
            );
        }
        if (receipt.contains("gaussian_segment_host_to_device_time_ns")) {
            gaussian_segment_host_to_device_time_ns += py::cast<int64_t>(
                receipt["gaussian_segment_host_to_device_time_ns"]
            );
        }
        if (receipt.contains("gaussian_segment_raw_generation_time_ns")) {
            gaussian_segment_raw_generation_time_ns += py::cast<int64_t>(
                receipt["gaussian_segment_raw_generation_time_ns"]
            );
        }
        if (receipt.contains("gaussian_segment_transcendental_conversion_time_ns")) {
            gaussian_segment_transcendental_conversion_time_ns += py::cast<int64_t>(
                receipt["gaussian_segment_transcendental_conversion_time_ns"]
            );
        }
        if (receipt.contains("gaussian_segment_allocation_copy_time_ns")) {
            gaussian_segment_allocation_copy_time_ns += py::cast<int64_t>(
                receipt["gaussian_segment_allocation_copy_time_ns"]
            );
        }
        if (receipt.contains("policy_birth_segment_host_wall_time_ns")) {
            policy_birth_segment_host_wall_time_ns += py::cast<int64_t>(
                receipt["policy_birth_segment_host_wall_time_ns"]
            );
        }
        if (receipt.contains("policy_birth_segment_device_to_host_time_ns")) {
            policy_birth_segment_device_to_host_time_ns += py::cast<int64_t>(
                receipt["policy_birth_segment_device_to_host_time_ns"]
            );
        }
        if (receipt.contains("policy_birth_segment_host_to_device_time_ns")) {
            policy_birth_segment_host_to_device_time_ns += py::cast<int64_t>(
                receipt["policy_birth_segment_host_to_device_time_ns"]
            );
        }
        if (receipt.contains("policy_birth_segment_torch_random_generation_time_ns")) {
            policy_birth_segment_torch_random_generation_time_ns += py::cast<int64_t>(
                receipt["policy_birth_segment_torch_random_generation_time_ns"]
            );
        }
        if (receipt.contains("policy_birth_segment_torch_compare_pack_time_ns")) {
            policy_birth_segment_torch_compare_pack_time_ns += py::cast<int64_t>(
                receipt["policy_birth_segment_torch_compare_pack_time_ns"]
            );
        }
        if (receipt.contains("policy_birth_segment_allocation_stack_copy_time_ns")) {
            policy_birth_segment_allocation_stack_copy_time_ns += py::cast<int64_t>(
                receipt["policy_birth_segment_allocation_stack_copy_time_ns"]
            );
        }
        if (receipt.contains("intermediate_device_to_host_operations")) {
            intermediate_device_to_host_operations += py::cast<int64_t>(
                receipt["intermediate_device_to_host_operations"]
            );
        }
        if (receipt.contains("intermediate_host_to_device_operations")) {
            intermediate_host_to_device_operations += py::cast<int64_t>(
                receipt["intermediate_host_to_device_operations"]
            );
        }
        window_receipts.append(receipt);
        completed_generations += completed_window_generations;

        if (
            completed_window_generations < requested_window_generations
            || stop_after_completed_prefix
        ) {
            break;
        }

        if (window_index + 1 < window_count) {
            const auto drain_receipt = external_event_sink.has_value()
                ? event_archive::drain_event_window_to_sink_v1(
                    event_columns,
                    event_active_count,
                    event_sequence_base,
                    event_status,
                    *external_event_sink
                )
                : event_archive::drain_event_window_v1(
                    event_columns,
                    event_active_count,
                    event_sequence_base,
                    event_status,
                    archived_events
                );
            if constexpr (window_refill::kOptionDM0TelemetryEnabledV1) {
                int64_t event_d2h_operations = 0;
                for (
                    const torch::Tensor* value :
                    std::array<const torch::Tensor*, 3>{
                        &event_active_count,
                        &event_sequence_base,
                        &event_status,
                    }
                ) {
                    if (value->is_cuda() && value->numel() > 0) {
                        event_d2h_operations = checked_nonnegative_sum_v1(
                            event_d2h_operations,
                            1,
                            "M0 event-drain D2H operations"
                        );
                    }
                }
                if (event_columns.front().is_cuda()) {
                    const bool any_active = std::any_of(
                        drain_receipt.active_count_before.begin(),
                        drain_receipt.active_count_before.end(),
                        [](const int64_t count) { return count > 0; }
                    );
                    if (any_active) {
                        event_d2h_operations = checked_nonnegative_sum_v1(
                            event_d2h_operations,
                            static_cast<int64_t>(
                                event_archive::kEventColumnCountV1
                            ),
                            "M0 event-drain D2H operations"
                        );
                    }
                }
                option_d_m0_outer_event_d2h_operations =
                    checked_nonnegative_sum_v1(
                        option_d_m0_outer_event_d2h_operations,
                        event_d2h_operations,
                        "M0 outer event D2H operations"
                    );
                option_d_m0_aggregate->add_outer_intermediate_transfers(
                    event_d2h_operations,
                    0
                );
            }
            event_drain_receipts.append(
                event_drain_receipt_to_dict_v1(drain_receipt)
            );
            const py::dict gaussian_refill_receipt = refill_gaussian_window();
            record_option_d_m0_outer_refill(
                gaussian_refill_receipt,
                "refill_cpython_gaussian_window_v1"
            );
            gaussian_refill_receipts.append(gaussian_refill_receipt);
            const py::dict policy_refill_receipt = refill_policy_window();
            record_option_d_m0_outer_refill(
                policy_refill_receipt,
                "refill_fixed_mlp_policy_window_v1"
            );
            policy_refill_receipts.append(policy_refill_receipt);
            accumulate_policy_identity_receipt(
                policy_refill_receipt, true
            );
            generation_summary_scalar.fill_(
                std::numeric_limits<double>::quiet_NaN()
            );
            runtime_bookkeeping_by_tick.zero_();
            patch_stats.fill_(std::numeric_limits<double>::quiet_NaN());
        }
    }

    std::vector<event_archive::EventChunkV1> event_histories;
    std::vector<int64_t> final_event_counts;
    if (external_event_sink.has_value()) {
        const auto final_drain_receipt = event_archive::drain_event_window_to_sink_v1(
            event_columns,
            event_active_count,
            event_sequence_base,
            event_status,
            *external_event_sink
        );
        final_event_counts = final_drain_receipt.active_count_before;
        event_drain_receipts.append(
            event_drain_receipt_to_dict_v1(final_drain_receipt)
        );
    } else {
        if (archived_events.cells.empty()) {
            archived_events.cells.resize(static_cast<std::size_t>(cells));
            archived_events.archived_rows.assign(
                static_cast<std::size_t>(cells),
                0
            );
        }
        event_histories = event_archive::materialize_event_history_v1(
            event_columns,
            event_active_count,
            event_sequence_base,
            event_status,
            archived_events
        );
    }
    if constexpr (window_refill::kOptionDM0TelemetryEnabledV1) {
        int64_t final_event_d2h_operations = 0;
        for (
            const torch::Tensor* value :
            std::array<const torch::Tensor*, 3>{
                &event_active_count,
                &event_sequence_base,
                &event_status,
            }
        ) {
            if (value->is_cuda() && value->numel() > 0) {
                final_event_d2h_operations = checked_nonnegative_sum_v1(
                    final_event_d2h_operations,
                    1,
                    "M0 final event D2H operations"
                );
            }
        }
        if (event_columns.front().is_cuda()) {
            if (external_event_sink.has_value()) {
                const bool any_active = std::any_of(
                    final_event_counts.begin(),
                    final_event_counts.end(),
                    [](const int64_t count) { return count > 0; }
                );
                if (any_active) {
                    final_event_d2h_operations = checked_nonnegative_sum_v1(
                        final_event_d2h_operations,
                        static_cast<int64_t>(event_archive::kEventColumnCountV1),
                        "M0 final event D2H operations"
                    );
                }
            } else {
                for (std::size_t cell = 0; cell < event_histories.size(); ++cell) {
                    const int64_t total_rows = event_histories[cell].front().numel();
                    const int64_t archived_rows = archived_events.archived_rows[cell];
                    TORCH_CHECK(
                        total_rows >= archived_rows,
                        "M0 final event rows precede archived rows"
                    );
                    if (total_rows > archived_rows) {
                        final_event_d2h_operations = checked_nonnegative_sum_v1(
                            final_event_d2h_operations,
                            static_cast<int64_t>(event_archive::kEventColumnCountV1),
                            "M0 final event D2H operations"
                        );
                    }
                }
            }
        }
        option_d_m0_outer_event_d2h_operations = checked_nonnegative_sum_v1(
            option_d_m0_outer_event_d2h_operations,
            final_event_d2h_operations,
            "M0 outer event D2H operations"
        );
        option_d_m0_aggregate->add_outer_final_device_to_host_operations(
            final_event_d2h_operations
        );
    }
    py::list event_rows_by_cell;
    int64_t materialized_event_rows = 0;
    if (external_event_sink.has_value()) {
        for (const int64_t rows : external_event_sink->archived_rows) {
            TORCH_CHECK(
                rows <= std::numeric_limits<int64_t>::max() - materialized_event_rows,
                "external event row count overflow"
            );
            materialized_event_rows += rows;
            event_rows_by_cell.append(rows);
        }
    } else {
      for (const auto& history : event_histories) {
        const int64_t rows = history.front().numel();
        TORCH_CHECK(
            rows <= std::numeric_limits<int64_t>::max() - materialized_event_rows,
            "materialized event row count overflow"
        );
        materialized_event_rows += rows;
        event_rows_by_cell.append(rows);
      }
    }
    if constexpr (window_refill::kOptionDM0TelemetryEnabledV1) {
        TORCH_CHECK(
            option_d_m0_aggregate->window_count
                == static_cast<int64_t>(py::len(window_receipts)),
            "M0 aggregate window count differs from settled window receipts"
        );
    }
    TORCH_CHECK(
        terminal_window_receipt_observed,
        "resident route lacks its terminal window receipt"
    );
    const int64_t terminal_final_stream_synchronization_time_ns =
        exact_nonnegative_native_integer_v1(
            terminal_window_receipt,
            "final_stream_synchronization_time_ns"
        );
    const int64_t terminal_last_useful_kernel_completion_observed_host_time_ns =
        exact_nonnegative_native_integer_v1(
            terminal_window_receipt,
            "last_useful_kernel_completion_observed_host_time_ns"
        );
    const int64_t terminal_compact_settlement_d2h_operations =
        exact_nonnegative_native_integer_v1(
            terminal_window_receipt,
            "compact_settlement_d2h_operations"
        );
    const int64_t terminal_final_host_synchronizations =
        exact_nonnegative_native_integer_v1(
            terminal_window_receipt,
            "final_host_synchronizations"
        );
    const int64_t terminal_final_device_to_host_operations =
        exact_nonnegative_native_integer_v1(
            terminal_window_receipt,
            "final_device_to_host_operations"
        );
    const py::dict terminal_compact_summary_reporting_storage =
        exact_native_receipt_dict_v1(
            terminal_window_receipt,
            "compact_summary_reporting_storage"
        );

    py::dict out;
    out["schema_version"] =
        "eiel_cuda_architecture_b_native_resident_window_controller_v1";
    out["owner"] = "native_extension";
    out["physical_window_generations"] = physical_window_generations;
    out["logical_generation_count"] = logical_generation_count;
    out["window_count"] = window_count;
    out["physical_window_alignment"] =
        "absolute_generation_lattice_v1";
    out["first_window_generations"] = first_window_generations;
    out["settled_window_count"] = py::len(window_receipts);
    out["completed_generations"] = completed_generations;
    out["final_stream_synchronization_time_ns"] =
        terminal_final_stream_synchronization_time_ns;
    out["last_useful_kernel_completion_observed_host_time_ns"] =
        terminal_last_useful_kernel_completion_observed_host_time_ns;
    out["compact_settlement_d2h_operations"] =
        terminal_compact_settlement_d2h_operations;
    // M0 route aggregation, when compiled in, deliberately overwrites these
    // terminal-window fallbacks with route-level final-transfer counters.
    out["final_host_synchronizations"] =
        terminal_final_host_synchronizations;
    out["final_device_to_host_operations"] =
        terminal_final_device_to_host_operations;
    out["compact_summary_reporting_storage"] =
        py::dict(terminal_compact_summary_reporting_storage);
    out["window_receipts"] = std::move(window_receipts);
    out["generation_summary_scalar"] = torch::cat(summary_chunks, 0);
    out["runtime_bookkeeping_by_tick"] = torch::cat(bookkeeping_chunks, 1);
    out["patch_stats"] = torch::cat(patch_chunks, 1);
    out["turnover_counts_by_window"] = torch::stack(turnover_count_chunks, 0);
    out["turnover_count_terminal_generations"] =
        std::move(turnover_count_terminal_generations);
    if (!external_event_sink.has_value()) {
        out["event_histories"] = event_histories_to_list_v1(event_histories);
    }
    out["event_rows_by_cell"] = std::move(event_rows_by_cell);
    out["event_drain_receipts"] = std::move(event_drain_receipts);
    if (external_event_sink.has_value()) {
        py::list frontier_by_cell;
        py::list pending_rows_by_cell;
        for (const int64_t frontier : external_event_sink->archived_rows) {
            frontier_by_cell.append(frontier);
            pending_rows_by_cell.append(0);
        }
        py::dict event_reference;
        event_reference["schema_version"] = "eiel_resident_event_sink_v1";
        event_reference["path"] = external_event_sink->path;
        event_reference["row_bytes"] = event_archive::kEventRowBytesV1;
        event_reference["row_count_by_cell"] = frontier_by_cell;
        event_reference["cell_count"] = cells;
        out["event_history_storage_mode"] = "eiel_resident_event_sink_v1";
        out["event_history_reference"] = std::move(event_reference);
        out["event_history_frontier_by_cell"] = frontier_by_cell;
        out["event_pending_rows_by_cell"] = pending_rows_by_cell;
        out["event_sequence_base"] = event_sequence_base.detach().cpu().contiguous();
        if (cells == 1) {
            out["event_history_frontier"] = py::int_(
                external_event_sink->archived_rows.front()
            );
        } else {
            out["event_history_frontier"] = frontier_by_cell;
        }
        out["host_event_archive_strategy"] = "external_append_only_sink_v1";
        out["host_event_archive_budget_bytes"] = py::none();
        out["host_event_archive_bytes_appended"] = external_event_sink->archived_bytes;
        out["host_event_materialized_bytes"] = 0;
        out["host_event_archive_budget_enforced"] = false;
    } else {
        out["host_event_archive_strategy"] = "bounded_host_archive_v1";
        out["host_event_archive_budget_bytes"] =
            event_archive::kHostEventArchiveBudgetBytesV1;
        out["host_event_archive_bytes_appended"] = archived_events.archived_bytes;
        TORCH_CHECK(
            materialized_event_rows
                <= std::numeric_limits<int64_t>::max()
                    / event_archive::kEventRowBytesV1,
            "materialized event byte count overflow"
        );
        out["host_event_materialized_bytes"] =
            materialized_event_rows * event_archive::kEventRowBytesV1;
        out["host_event_archive_budget_enforced"] = true;
    }
    out["gaussian_refill_receipts"] = std::move(gaussian_refill_receipts);
    out["native_host_gaussian_segment_refill_used"] =
        gaussian_segment_refill_calls_inside_tick_loop > 0;
    out["gaussian_segment_refill_calls_inside_tick_loop"] =
        gaussian_segment_refill_calls_inside_tick_loop;
    out["gaussian_segment_refill_receipts"] =
        std::move(gaussian_segment_refill_receipts);
    out["gaussian_segment_boundary_receipts"] =
        std::move(gaussian_segment_boundary_receipts);
    out["gaussian_segment_reuse_calls_inside_tick_loop"] =
        gaussian_segment_reuse_calls_inside_tick_loop;
    out["gaussian_segment_device_to_host_operations_lower_bound"] =
        gaussian_segment_device_to_host_operations_lower_bound;
    out["gaussian_segment_host_to_device_operations_lower_bound"] =
        gaussian_segment_host_to_device_operations_lower_bound;
    out["gaussian_segment_host_wall_time_ns"] =
        gaussian_segment_host_wall_time_ns;
    out["gaussian_segment_device_to_host_time_ns"] =
        gaussian_segment_device_to_host_time_ns;
    out["gaussian_segment_host_to_device_time_ns"] =
        gaussian_segment_host_to_device_time_ns;
    out["gaussian_segment_raw_generation_time_ns"] =
        gaussian_segment_raw_generation_time_ns;
    out["gaussian_segment_transcendental_conversion_time_ns"] =
        gaussian_segment_transcendental_conversion_time_ns;
    out["gaussian_segment_allocation_copy_time_ns"] =
        gaussian_segment_allocation_copy_time_ns;
    out["policy_birth_segment_host_wall_time_ns"] =
        policy_birth_segment_host_wall_time_ns;
    out["policy_birth_segment_device_to_host_time_ns"] =
        policy_birth_segment_device_to_host_time_ns;
    out["policy_birth_segment_host_to_device_time_ns"] =
        policy_birth_segment_host_to_device_time_ns;
    out["policy_birth_segment_torch_random_generation_time_ns"] =
        policy_birth_segment_torch_random_generation_time_ns;
    out["policy_birth_segment_torch_compare_pack_time_ns"] =
        policy_birth_segment_torch_compare_pack_time_ns;
    out["policy_birth_segment_allocation_stack_copy_time_ns"] =
        policy_birth_segment_allocation_stack_copy_time_ns;
    out["policy_birth_segment_refill_receipts"] =
        std::move(policy_birth_segment_refill_receipts);
    out["intermediate_device_to_host_operations"] =
        intermediate_device_to_host_operations;
    out["intermediate_host_to_device_operations"] =
        intermediate_host_to_device_operations;
    out["policy_refill_receipts"] = std::move(policy_refill_receipts);
    out["policy_birth_identity_entry_ensure_receipts"] =
        std::move(policy_birth_identity_entry_ensure_receipts);
    out["policy_birth_identity_ensure_calls_total"] =
        policy_birth_identity_ensure_calls_total;
    out["policy_birth_identity_materialization_calls_total"] =
        policy_birth_identity_materialization_calls_total;
    out["policy_birth_identity_physical_refill_calls_total"] =
        policy_birth_identity_physical_refill_calls_total;
    out["policy_birth_identity_resident_suffix_reuse_calls_total"] =
        policy_birth_identity_resident_suffix_reuse_calls_total;
    out["policy_birth_identity_device_to_host_operations_lower_bound"] =
        policy_birth_identity_device_to_host_operations_total;
    out["policy_birth_identity_host_to_device_operations_lower_bound"] =
        policy_birth_identity_host_to_device_operations_total;
    out["policy_birth_identity_device_to_host_bytes_lower_bound"] =
        policy_birth_identity_device_to_host_bytes_total;
    out["policy_birth_identity_host_to_device_bytes_lower_bound"] =
        policy_birth_identity_host_to_device_bytes_total;
    out["policy_birth_identity_host_wall_time_ns"] =
        policy_birth_identity_host_wall_time_ns_total;
    out["policy_birth_identity_device_to_host_time_ns"] =
        policy_birth_identity_device_to_host_time_ns_total;
    out["policy_birth_identity_host_to_device_time_ns"] =
        policy_birth_identity_host_to_device_time_ns_total;
    out["resume_boundary_refill"] = resume_boundary_refill;
    out["checkpointed_gaussian_suffix_preserved"] =
        reconstructed_window_suffix;
    out["checkpointed_policy_suffix_preserved"] =
        reconstructed_window_suffix
        && !first_policy_identity_entry_physical_refill;
    out["checkpointed_gaussian_policy_suffix_preserved"] =
        reconstructed_window_suffix
        && !first_policy_identity_entry_physical_refill;
    out["checkpointed_policy_suffix_replaced_by_entry_hydration"] =
        reconstructed_window_suffix
        && first_policy_identity_entry_physical_refill;
    out["resume_gaussian_refill_receipt"] =
        std::move(resume_gaussian_refill_receipt);
    out["resume_policy_refill_receipt"] =
        std::move(resume_policy_refill_receipt);
    out["external_python_window_boundaries"] = 0;
    out["native_window_loop_owner"] = "native_extension";
    out["internal_pybind_redispatch"] = internal_pybind_redispatch;
    out["direct_cpp_window_core_dispatch"] = direct_cpp_window_core_dispatch;
    out["production_native_chunk_loop"] = production_native_chunk_loop;
    out["hidden_fallback"] = false;
    out["runtime_jit"] = false;
    if constexpr (window_refill::kOptionDM0TelemetryEnabledV1) {
        option_d_m0_aggregate->publish(out);
        out["resident_window_outer_boundary_device_to_host_operations_lower_bound"] =
            option_d_m0_outer_boundary_d2h_operations;
        out["resident_window_outer_event_device_to_host_operations_lower_bound"] =
            option_d_m0_outer_event_d2h_operations;
        out["resident_window_outer_refill_device_to_host_operations_lower_bound"] =
            option_d_m0_outer_refill_d2h_operations;
        out["resident_window_outer_refill_host_to_device_operations_lower_bound"] =
            option_d_m0_outer_refill_h2d_operations;
        out["resident_window_outer_transfer_operation_measurement_scope"] =
            "nonempty_cuda_tensor_copy_operations_in_native_resident_window_"
            "boundary_status_summary_event_archive_and_rng_refill_v1";
        out["resident_window_route_counter_semantics"] =
            "child_final_counters_reclassified_as_intermediate_except_for_"
            "the_single_route_terminal_window_v1";
    }
    return out;
}

inline py::dict execute_resident_windows_v1(
    const py::function& window_controller,
    const py::list& window_arguments,
    const int64_t logical_generation_count,
    const int64_t generation_argument_index,
    const int64_t generation_count_argument_index,
    const torch::Tensor& generation_summary_scalar,
    const torch::Tensor& runtime_bookkeeping_by_tick,
    const torch::Tensor& patch_stats,
    const std::vector<torch::Tensor>& event_state_tensors,
    const std::vector<torch::Tensor>& ordinary_birth_state_tensors,
    const bool policy_weight_mutation_enable,
    const std::string& policy_weight_mutation_mode,
    const double policy_weight_mutation_rate,
    const double policy_weight_mutation_std
) {
    TORCH_CHECK(
        static_cast<int64_t>(window_arguments.size()) == kNativeArgumentCountV1,
        "resident window controller requires the exact 160-argument native ABI"
    );
    TORCH_CHECK(
        generation_argument_index == kGenerationArgumentV1
            && generation_count_argument_index == kGenerationCountArgumentV1,
        "resident window controller argument indices disagree with the native ABI"
    );
    const int64_t physical_window_generations = py::cast<int64_t>(
        window_arguments[generation_count_argument_index]
    );
    const int64_t initial_generation = py::cast<int64_t>(
        window_arguments[generation_argument_index]
    );
    TORCH_CHECK(
        py::cast<std::vector<torch::Tensor>>(
            window_arguments[kEventStateArgumentV1]
        ).size() == event_state_tensors.size(),
        "resident event-state argument changed"
    );
    TORCH_CHECK(
        py::cast<std::vector<torch::Tensor>>(
            window_arguments[kOrdinaryBirthStateArgumentV1]
        ).size() == ordinary_birth_state_tensors.size(),
        "resident ordinary-birth argument changed"
    );
    const auto full_tick_outputs = py::cast<std::vector<torch::Tensor>>(
        window_arguments[kFullTickOutputsArgumentV1]
    );
    TORCH_CHECK(
        full_tick_outputs.size() == 19
            && full_tick_outputs[kGenerationSummaryOutputV1].is_same(
                generation_summary_scalar
            ),
        "resident native dispatch requires scalar summaries with compact planes disabled"
    );
    const auto patch_argument = py::cast<std::vector<torch::Tensor>>(
        window_arguments[kPatchStatsArgumentV1]
    );
    TORCH_CHECK(
        !patch_argument.empty() && patch_argument.front().is_same(patch_stats),
        "resident Patch stats argument changed"
    );

    py::list call_arguments;
    for (const auto& argument : window_arguments) {
        call_arguments.append(argument);
    }
    const auto controller_status = py::cast<torch::Tensor>(
        call_arguments[kControllerStatusArgumentV1]
    );
    const auto lifecycle_status = py::cast<torch::Tensor>(
        call_arguments[kLifecycleStatusArgumentV1]
    );
    const auto turnover_outputs = py::cast<std::vector<torch::Tensor>>(
        call_arguments[kTurnoverOutputsArgumentV1]
    );
    TORCH_CHECK(
        turnover_outputs.size() > kTurnoverStatusOutputV1,
        "resident dispatch requires generation turnover outputs"
    );
    const int64_t policy_row_width =
        ordinary_birth_state_tensors[kPolicyDeltaTapeV1].size(2);
    TORCH_CHECK(
        policy_row_width > 1
            || policy_weight_mutation_enable
            || ordinary_birth_state_tensors[kPolicyCrossoverMaskPackedTapeV1].size(2) == 1,
        "native pybind resident dispatch cannot infer full policy width from compact policy mutation tape"
    );
    const auto submit_window = [&](
        const int64_t generation_base,
        const int64_t requested_window_generations
    ) {
        call_arguments[generation_argument_index] = py::int_(generation_base);
        call_arguments[generation_count_argument_index] =
            py::int_(requested_window_generations);
        const py::tuple positional_arguments(call_arguments);
        PyObject* raw_result = PyObject_CallObject(
            window_controller.ptr(),
            positional_arguments.ptr()
        );
        if (raw_result == nullptr) {
            throw py::error_already_set();
        }
        const auto result = py::reinterpret_steal<py::object>(raw_result);
        TORCH_CHECK(
            py::isinstance<py::dict>(result),
            "native window receipt must be a dict"
        );
        return py::cast<py::dict>(result);
    };
    return execute_resident_windows_core_v1(
        submit_window,
        initial_generation,
        physical_window_generations,
        logical_generation_count,
        generation_summary_scalar,
        runtime_bookkeeping_by_tick,
        patch_stats,
        event_state_tensors,
        ordinary_birth_state_tensors,
        controller_status,
        lifecycle_status,
        turnover_outputs[kTurnoverCountsOutputV1],
        turnover_outputs[kTurnoverStatusOutputV1],
        policy_row_width,
        policy_weight_mutation_enable,
        policy_weight_mutation_mode,
        policy_weight_mutation_rate,
        policy_weight_mutation_std,
        true,
        false,
        false
    );
}

}  // namespace eiel::cuda_architecture_b::resident_window_controller_v1
