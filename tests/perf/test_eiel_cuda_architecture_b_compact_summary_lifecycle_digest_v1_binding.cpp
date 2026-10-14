#include "cuda_architecture_b_compact_summary_lifecycle_digest_v1.cuh"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include <array>
#include <cstdint>
#include <vector>

namespace lifecycle_digest =
    eiel::cuda_architecture_b::compact_summary_lifecycle_digest_v1;

namespace {

constexpr std::int64_t kPhysicalCapacityV1 = -2;

void require_cuda_contiguous(
    const torch::Tensor& tensor,
    const torch::Device& device,
    const char* const name
) {
    TORCH_CHECK(tensor.defined(), name, " must be defined");
    TORCH_CHECK(tensor.is_cuda(), name, " must be CUDA resident");
    TORCH_CHECK(tensor.device() == device, name, " device mismatch");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(!tensor.requires_grad(), name, " must not require gradients");
}

void null_field(
    lifecycle_digest::FlatTypedLifecycleColumnsV1* const columns,
    const std::int64_t field
) {
    switch (field) {
        case 0: columns->typed_actor_id = nullptr; break;
        case 1: columns->typed_actor_slot = nullptr; break;
        case 2: columns->typed_child_id = nullptr; break;
        case 3: columns->typed_event_type = nullptr; break;
        case 4: columns->typed_generation = nullptr; break;
        case 5: columns->typed_lineage_depth = nullptr; break;
        case 6: columns->typed_founder_id = nullptr; break;
        case 7: columns->typed_parent_id = nullptr; break;
        case 8: columns->typed_reason = nullptr; break;
        case 9: columns->typed_sequence = nullptr; break;
        case 10: columns->typed_subtype = nullptr; break;
        case 11: columns->typed_success = nullptr; break;
        case 12: columns->typed_target_id = nullptr; break;
        case 13: columns->typed_target_slot = nullptr; break;
        case 14: columns->typed_tick = nullptr; break;
        case 15: columns->typed_value = nullptr; break;
        default: break;
    }
}

void digest_into(
    const std::vector<torch::Tensor>& fields,
    const torch::Tensor& event_cursor,
    const std::int64_t generation_index,
    torch::Tensor digests,
    torch::Tensor status,
    torch::Tensor validity,
    const std::int64_t capacity_override,
    const std::int64_t null_field_index
) {
    TORCH_CHECK(
        fields.size() == static_cast<std::size_t>(lifecycle_digest::kFieldCountV1),
        "fields must contain the 16 lexicographically ordered lifecycle columns"
    );
    TORCH_CHECK(digests.defined() && digests.is_cuda(), "digests must be CUDA");
    const torch::Device device = digests.device();
    require_cuda_contiguous(digests, device, "digests");
    require_cuda_contiguous(status, device, "status");
    require_cuda_contiguous(validity, device, "validity");
    require_cuda_contiguous(event_cursor, device, "event_cursor");
    TORCH_CHECK(digests.scalar_type() == c10::ScalarType::Byte, "digests must be uint8");
    TORCH_CHECK(status.scalar_type() == c10::ScalarType::Int, "status must be int32");
    TORCH_CHECK(validity.scalar_type() == c10::ScalarType::Byte, "validity must be uint8");
    TORCH_CHECK(event_cursor.scalar_type() == c10::ScalarType::Long, "event_cursor must be int64");
    TORCH_CHECK(
        digests.dim() == 3 && digests.size(2) == lifecycle_digest::kDigestBytesV1,
        "digests must be [generations,cells,32]"
    );
    const std::int64_t generations = digests.size(0);
    const std::int64_t cells = digests.size(1);
    TORCH_CHECK(generations > 0 && cells > 0, "output generations and cells must be positive");
    TORCH_CHECK(
        generation_index >= 0 && generation_index < generations,
        "generation_index is out of range"
    );
    TORCH_CHECK(
        status.dim() == 2 && status.size(0) == generations && status.size(1) == cells,
        "status must be [generations,cells]"
    );
    TORCH_CHECK(validity.sizes() == status.sizes(), "validity/status shapes disagree");
    TORCH_CHECK(
        event_cursor.dim() == 1 && event_cursor.numel() == cells,
        "event_cursor must be [cells]"
    );

    const std::array<c10::ScalarType, lifecycle_digest::kFieldCountV1> dtypes = {
        c10::ScalarType::Long,
        c10::ScalarType::Long,
        c10::ScalarType::Long,
        c10::ScalarType::Int,
        c10::ScalarType::Long,
        c10::ScalarType::Int,
        c10::ScalarType::Long,
        c10::ScalarType::Long,
        c10::ScalarType::Int,
        c10::ScalarType::Long,
        c10::ScalarType::Int,
        c10::ScalarType::Char,
        c10::ScalarType::Long,
        c10::ScalarType::Long,
        c10::ScalarType::Long,
        c10::ScalarType::Float,
    };
    const std::int64_t physical_capacity = fields[0].size(1);
    for (std::size_t index = 0; index < fields.size(); ++index) {
        require_cuda_contiguous(fields[index], device, "lifecycle field");
        TORCH_CHECK(fields[index].scalar_type() == dtypes[index], "lifecycle field dtype mismatch");
        TORCH_CHECK(
            fields[index].dim() == 2
                && fields[index].size(0) == cells
                && fields[index].size(1) == physical_capacity,
            "lifecycle fields must share [cells,capacity]"
        );
    }
    TORCH_CHECK(
        capacity_override == kPhysicalCapacityV1
            || capacity_override == -1
            || capacity_override == physical_capacity,
        "capacity_override may select physical capacity or the -1 fault only"
    );
    TORCH_CHECK(
        null_field_index >= -1
            && null_field_index < lifecycle_digest::kFieldCountV1,
        "null_field_index is out of range"
    );

    lifecycle_digest::FlatTypedLifecycleColumnsV1 columns{
        fields[3].data_ptr<std::int32_t>(),
        fields[4].data_ptr<std::int64_t>(),
        fields[14].data_ptr<std::int64_t>(),
        fields[9].data_ptr<std::int64_t>(),
        fields[0].data_ptr<std::int64_t>(),
        fields[12].data_ptr<std::int64_t>(),
        fields[1].data_ptr<std::int64_t>(),
        fields[13].data_ptr<std::int64_t>(),
        fields[7].data_ptr<std::int64_t>(),
        fields[2].data_ptr<std::int64_t>(),
        fields[6].data_ptr<std::int64_t>(),
        fields[5].data_ptr<std::int32_t>(),
        fields[10].data_ptr<std::int32_t>(),
        fields[8].data_ptr<std::int32_t>(),
        fields[11].data_ptr<std::int8_t>(),
        fields[15].data_ptr<float>(),
    };
    if (null_field_index >= 0) {
        null_field(&columns, null_field_index);
    }
    const lifecycle_digest::FlatLifecycleDigestInputsV1 inputs{
        columns,
        event_cursor.data_ptr<std::int64_t>(),
        capacity_override == kPhysicalCapacityV1
            ? physical_capacity
            : capacity_override,
    };
    const lifecycle_digest::LifecycleDigestOutputsV1 outputs{
        digests.data_ptr<std::uint8_t>(),
        status.data_ptr<std::int32_t>(),
        validity.data_ptr<std::uint8_t>(),
        generations,
        cells,
        generation_index,
    };

    c10::cuda::CUDAGuard guard(device);
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream(device.index());
    lifecycle_digest::launch_compact_summary_lifecycle_digest_v1(
        inputs,
        outputs,
        stream
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

pybind11::dict resource_receipt() {
    const lifecycle_digest::KernelResourceReceiptV1 receipt =
        lifecycle_digest::query_compact_summary_lifecycle_digest_v1_resources();
    pybind11::dict output;
    output["binary_version"] = receipt.binary_version;
    output["ptx_version"] = receipt.ptx_version;
    output["max_threads_per_block"] = receipt.max_threads_per_block;
    output["registers_per_thread"] = receipt.registers_per_thread;
    output["occupancy_blocks_per_multiprocessor"] =
        receipt.occupancy_blocks_per_multiprocessor;
    output["constant_bytes"] = receipt.constant_bytes;
    output["local_bytes_per_thread"] = receipt.local_bytes_per_thread;
    output["shared_bytes_per_block"] = receipt.shared_bytes_per_block;
    return output;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def(
        "digest_into",
        &digest_into,
        pybind11::arg("fields"),
        pybind11::arg("event_cursor"),
        pybind11::arg("generation_index"),
        pybind11::arg("digests"),
        pybind11::arg("status"),
        pybind11::arg("validity"),
        pybind11::arg("capacity_override") = kPhysicalCapacityV1,
        pybind11::arg("null_field_index") = -1
    );
    module.def("resource_receipt", &resource_receipt);
    module.attr("contract_version") = lifecycle_digest::kContractVersionV1;
    module.attr("status_success") = static_cast<std::int32_t>(
        lifecycle_digest::kLifecycleDigestSuccessV1
    );
    module.attr("status_invalid_capacity") = static_cast<std::int32_t>(
        lifecycle_digest::kLifecycleDigestInvalidCapacityV1
    );
    module.attr("status_count_out_of_range") = static_cast<std::int32_t>(
        lifecycle_digest::kLifecycleDigestCountOutOfRangeV1
    );
    module.attr("status_null_column") = static_cast<std::int32_t>(
        lifecycle_digest::kLifecycleDigestNullColumnV1
    );
    module.attr("status_sequence_mismatch") = static_cast<std::int32_t>(
        lifecycle_digest::kLifecycleDigestSequenceMismatchV1
    );
    module.attr("status_nonfinite_value") = static_cast<std::int32_t>(
        lifecycle_digest::kLifecycleDigestNonfiniteValueV1
    );
    module.attr("status_row_order_invalid") = static_cast<std::int32_t>(
        lifecycle_digest::kLifecycleDigestRowOrderInvalidV1
    );
}
