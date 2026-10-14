#include "../../engine/perf/cuda_architecture_b_compact_summary_reductions_v1.cuh"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace reductions =
    eiel::cuda_architecture_b::compact_summary_reductions_v1;

namespace {

constexpr std::int64_t kTypedColumnCount = 16;
constexpr std::int64_t kIdentityScalarCount = 3;

constexpr std::array<c10::ScalarType, kTypedColumnCount> kTypedDtypes = {
    c10::ScalarType::Int,
    c10::ScalarType::Long,
    c10::ScalarType::Long,
    c10::ScalarType::Long,
    c10::ScalarType::Long,
    c10::ScalarType::Long,
    c10::ScalarType::Long,
    c10::ScalarType::Long,
    c10::ScalarType::Long,
    c10::ScalarType::Long,
    c10::ScalarType::Long,
    c10::ScalarType::Int,
    c10::ScalarType::Int,
    c10::ScalarType::Int,
    c10::ScalarType::Char,
    c10::ScalarType::Float,
};

void check_cuda_contiguous(
    const torch::Tensor& tensor,
    const torch::Device& device,
    const c10::ScalarType dtype,
    const std::string& name
) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be CUDA resident");
    TORCH_CHECK(tensor.device() == device, name, " must share one CUDA device");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be C-contiguous");
    TORCH_CHECK(tensor.scalar_type() == dtype, name, " has the wrong dtype");
}

void check_vector_size(
    const std::vector<torch::Tensor>& values,
    const std::int64_t expected,
    const char* const name
) {
    TORCH_CHECK(
        static_cast<std::int64_t>(values.size()) == expected,
        name,
        " must contain ",
        expected,
        " tensors"
    );
}

void reduce_compact_summary_sources_v1(
    const std::vector<torch::Tensor>& typed_columns,
    torch::Tensor active_event_count,
    torch::Tensor generation_summary_scalar,
    torch::Tensor birth_decision_count,
    torch::Tensor birth_reason_counts,
    torch::Tensor active_agent_ids,
    const std::vector<torch::Tensor>& identity_scalars,
    const std::int64_t slot_capacity,
    const std::int64_t generation_index,
    torch::Tensor result_bytes
) {
    check_vector_size(typed_columns, kTypedColumnCount, "typed_columns");
    check_vector_size(identity_scalars, kIdentityScalarCount, "identity_scalars");
    TORCH_CHECK(
        typed_columns[0].dim() == 2,
        "typed lifecycle columns must have shape [cells, capacity]"
    );
    const std::int64_t cells = typed_columns[0].size(0);
    const std::int64_t event_capacity = typed_columns[0].size(1);
    TORCH_CHECK(cells > 0, "at least one cell is required");
    TORCH_CHECK(event_capacity >= 0, "event capacity must be nonnegative");
    TORCH_CHECK(slot_capacity > 0, "slot_capacity must be positive");
    const torch::Device device = typed_columns[0].device();

    for (std::int64_t index = 0; index < kTypedColumnCount; ++index) {
        const std::string name = "typed_columns[" + std::to_string(index) + "]";
        check_cuda_contiguous(
            typed_columns[index], device, kTypedDtypes[index], name
        );
        TORCH_CHECK(
            typed_columns[index].dim() == 2
                && typed_columns[index].size(0) == cells
                && typed_columns[index].size(1) == event_capacity,
            name,
            " must share exact [cells, event_capacity] shape"
        );
    }
    check_cuda_contiguous(
        active_event_count,
        device,
        c10::ScalarType::Long,
        "active_event_count"
    );
    TORCH_CHECK(
        active_event_count.dim() == 1 && active_event_count.numel() == cells,
        "active_event_count must have shape [cells]"
    );
    check_cuda_contiguous(
        generation_summary_scalar,
        device,
        c10::ScalarType::Double,
        "generation_summary_scalar"
    );
    TORCH_CHECK(
        generation_summary_scalar.dim() == 3
            && generation_summary_scalar.size(1) == cells
            && generation_summary_scalar.size(2)
                == reductions::kGenerationSummaryScalarWidthV1,
        "generation_summary_scalar must have exact shape [generations,cells,31]"
    );
    const std::int64_t generations = generation_summary_scalar.size(0);
    TORCH_CHECK(
        generation_index >= 0 && generation_index < generations,
        "generation_index must address generation_summary_scalar"
    );
    check_cuda_contiguous(
        birth_decision_count,
        device,
        c10::ScalarType::Long,
        "birth_decision_count"
    );
    TORCH_CHECK(
        birth_decision_count.dim() == 2
            && birth_decision_count.size(0) == cells
            && birth_decision_count.size(1) > 0,
        "birth_decision_count must have exact shape [cells,ticks]"
    );
    const std::int64_t ticks = birth_decision_count.size(1);
    check_cuda_contiguous(
        birth_reason_counts,
        device,
        c10::ScalarType::Long,
        "birth_reason_counts"
    );
    TORCH_CHECK(
        birth_reason_counts.dim() == 3
            && birth_reason_counts.size(0) == cells
            && birth_reason_counts.size(1) == ticks
            && birth_reason_counts.size(2)
                == reductions::kBirthPlanReasonCountV1,
        "birth_reason_counts must have exact shape [cells,ticks,10]"
    );

    check_cuda_contiguous(
        active_agent_ids,
        device,
        c10::ScalarType::Long,
        "active_agent_ids"
    );
    TORCH_CHECK(
        active_agent_ids.dim() == 2
            && active_agent_ids.size(0) == cells
            && active_agent_ids.size(1) == slot_capacity,
        "active_agent_ids must have exact shape [cells,slot_capacity]"
    );
    for (std::int64_t index = 0; index < kIdentityScalarCount; ++index) {
        const std::string name =
            "identity_scalars[" + std::to_string(index) + "]";
        check_cuda_contiguous(
            identity_scalars[index], device, c10::ScalarType::Long, name
        );
        TORCH_CHECK(
            identity_scalars[index].dim() == 1
                && identity_scalars[index].numel() == cells,
            name,
            " must have shape [cells]"
        );
    }
    check_cuda_contiguous(
        result_bytes, device, c10::ScalarType::Byte, "result_bytes"
    );
    TORCH_CHECK(
        result_bytes.dim() == 3
            && result_bytes.size(0) == generations
            && result_bytes.size(1) == cells
            && result_bytes.size(2)
                == static_cast<std::int64_t>(sizeof(reductions::ReductionResultV1)),
        "result_bytes must have exact shape [generations,cells,456]"
    );

    reductions::ReductionInputsV1 inputs{
        {
            {
                typed_columns[0].data_ptr<std::int32_t>(),
                typed_columns[1].data_ptr<std::int64_t>(),
                typed_columns[2].data_ptr<std::int64_t>(),
                typed_columns[3].data_ptr<std::int64_t>(),
                typed_columns[4].data_ptr<std::int64_t>(),
                typed_columns[5].data_ptr<std::int64_t>(),
                typed_columns[6].data_ptr<std::int64_t>(),
                typed_columns[7].data_ptr<std::int64_t>(),
                typed_columns[8].data_ptr<std::int64_t>(),
                typed_columns[9].data_ptr<std::int64_t>(),
                typed_columns[10].data_ptr<std::int64_t>(),
                typed_columns[11].data_ptr<std::int32_t>(),
                typed_columns[12].data_ptr<std::int32_t>(),
                typed_columns[13].data_ptr<std::int32_t>(),
                typed_columns[14].data_ptr<std::int8_t>(),
                typed_columns[15].data_ptr<float>(),
                event_capacity,
                event_capacity,
            },
            active_event_count.data_ptr<std::int64_t>(),
        },
        {
            generation_summary_scalar.data_ptr<double>(),
            generations,
            cells,
            generation_summary_scalar.size(2),
        },
        {
            active_agent_ids.data_ptr<std::int64_t>(),
            identity_scalars[0].data_ptr<std::int64_t>(),
            identity_scalars[1].data_ptr<std::int64_t>(),
            identity_scalars[2].data_ptr<std::int64_t>(),
            slot_capacity,
            slot_capacity,
        },
        {
            birth_decision_count.data_ptr<std::int64_t>(),
            birth_reason_counts.data_ptr<std::int64_t>(),
            ticks,
            ticks,
            ticks * reductions::kBirthPlanReasonCountV1,
            reductions::kBirthPlanReasonCountV1,
            reductions::kBirthPlanReasonCountV1,
        },
        generation_index,
        cells,
    };

    c10::cuda::CUDAGuard guard(device);
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream(device.index());
    const reductions::ReductionLaunchStatusV1 launch_status =
        reductions::launch_compact_summary_reductions_v1(
        inputs,
        {
            reinterpret_cast<reductions::ReductionResultV1*>(
                result_bytes.data_ptr<std::uint8_t>()
            ),
            generations,
            cells,
        },
        stream
    );
    TORCH_CHECK(
        launch_status == reductions::kReductionLaunchSuccessV1,
        "compact reduction rejected generation-major request/output shape: ",
        static_cast<std::int32_t>(launch_status)
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

pybind11::dict resource_receipt_v1() {
    const reductions::KernelResourceReceiptV1 receipt =
        reductions::query_compact_summary_reductions_v1_resources();
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
        "reduce_into",
        &reduce_compact_summary_sources_v1,
        "Allocation-free compact lifecycle and identity reductions"
    );
    module.def(
        "resource_receipt",
        &resource_receipt_v1,
        "Compact reduction CUDA kernel resource receipt"
    );
    module.attr("contract_version") = reductions::kContractVersionV1;
    module.attr("result_bytes") = sizeof(reductions::ReductionResultV1);
}
