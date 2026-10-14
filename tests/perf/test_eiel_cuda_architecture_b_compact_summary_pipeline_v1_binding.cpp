#include "cuda_architecture_b_compact_summary_assembler_v1.cuh"
#include "cuda_architecture_b_compact_summary_finalizer_v2.cuh"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace summary_abi = eiel::cuda_architecture_b::compact_summary_v2;
namespace finalizer =
    eiel::cuda_architecture_b::compact_summary_finalizer_v2;

void eiel_cuda_architecture_b_compact_summary_pipeline_v1_launch(
    const summary_abi::DefinedMaskV1* source_defined_masks,
    const std::int32_t* source_status_codes,
    const std::int64_t* source_integer_values,
    const double* source_float_values,
    const std::uint8_t* source_digest_values,
    const std::uint8_t* source_text_values,
    std::uint8_t* scratch_ready,
    std::uint8_t* scratch_validity,
    summary_abi::DefinedMaskV1* scratch_defined_masks,
    std::int32_t* scratch_status_codes,
    std::int64_t* scratch_integer_values,
    double* scratch_float_values,
    std::uint8_t* scratch_digest_values,
    std::uint8_t* scratch_text_values,
    summary_abi::CompactSummaryViewV2 scratch_view,
    summary_abi::CompactSummaryViewV2 output_view,
    std::int32_t* assemble_status,
    std::int32_t* producer_status,
    std::int64_t row_count,
    std::int64_t generation,
    cudaStream_t stream
);

cudaError_t
eiel_cuda_architecture_b_compact_summary_pipeline_v1_gate_attributes(
    cudaFuncAttributes* attributes
);

cudaError_t eiel_cuda_architecture_b_compact_summary_assembler_v1_attributes(
    cudaFuncAttributes* attributes
);

namespace {

constexpr std::array<c10::ScalarType, 6> kSourceDtypes = {
    c10::ScalarType::UInt64,
    c10::ScalarType::Int,
    c10::ScalarType::Long,
    c10::ScalarType::Double,
    c10::ScalarType::Byte,
    c10::ScalarType::Byte,
};

constexpr std::array<c10::ScalarType, summary_abi::kPlaneCountV2> kPlaneDtypes = {
    c10::ScalarType::Byte,
    c10::ScalarType::UInt64,
    c10::ScalarType::Int,
    c10::ScalarType::Long,
    c10::ScalarType::Double,
    c10::ScalarType::Byte,
    c10::ScalarType::Byte,
};

void check_tensor(
    const torch::Tensor& tensor,
    const torch::Device& device,
    const c10::ScalarType dtype,
    const std::vector<std::int64_t>& shape,
    const std::string& name
) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be CUDA resident");
    TORCH_CHECK(tensor.device() == device, name, " device mismatch");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(!tensor.requires_grad(), name, " must not require gradients");
    TORCH_CHECK(tensor.scalar_type() == dtype, name, " dtype mismatch");
    TORCH_CHECK(
        tensor.sizes().vec() == shape,
        name,
        " must have shape ",
        shape,
        "; received ",
        tensor.sizes()
    );
}

summary_abi::CompactSummaryViewV2 make_scratch_view(
    const std::vector<torch::Tensor>& planes,
    const std::int64_t cells
) {
    return {
        1,
        cells,
        {planes[summary_abi::kPlaneValidity].data_ptr<std::uint8_t>(), 0,
         planes[summary_abi::kPlaneValidity].stride(0)},
        {reinterpret_cast<summary_abi::DefinedMaskV1*>(
             planes[summary_abi::kPlaneDefinedMask].data_ptr()),
         0,
         planes[summary_abi::kPlaneDefinedMask].stride(0)},
        {planes[summary_abi::kPlaneStatusCodes].data_ptr<std::int32_t>(), 0,
         planes[summary_abi::kPlaneStatusCodes].stride(0)},
        {planes[summary_abi::kPlaneIntegerValues].data_ptr<std::int64_t>(), 0,
         planes[summary_abi::kPlaneIntegerValues].stride(0)},
        {planes[summary_abi::kPlaneFloatValues].data_ptr<double>(), 0,
         planes[summary_abi::kPlaneFloatValues].stride(0)},
        {planes[summary_abi::kPlaneDigestValues].data_ptr<std::uint8_t>(), 0,
         planes[summary_abi::kPlaneDigestValues].stride(0)},
        {planes[summary_abi::kPlaneTextValues].data_ptr<std::uint8_t>(), 0,
         planes[summary_abi::kPlaneTextValues].stride(0)},
    };
}

summary_abi::CompactSummaryViewV2 make_output_view(
    const std::vector<torch::Tensor>& planes,
    const std::int64_t generations,
    const std::int64_t cells
) {
    return {
        generations,
        cells,
        {planes[summary_abi::kPlaneValidity].data_ptr<std::uint8_t>(),
         planes[summary_abi::kPlaneValidity].stride(0),
         planes[summary_abi::kPlaneValidity].stride(1)},
        {reinterpret_cast<summary_abi::DefinedMaskV1*>(
             planes[summary_abi::kPlaneDefinedMask].data_ptr()),
         planes[summary_abi::kPlaneDefinedMask].stride(0),
         planes[summary_abi::kPlaneDefinedMask].stride(1)},
        {planes[summary_abi::kPlaneStatusCodes].data_ptr<std::int32_t>(),
         planes[summary_abi::kPlaneStatusCodes].stride(0),
         planes[summary_abi::kPlaneStatusCodes].stride(1)},
        {planes[summary_abi::kPlaneIntegerValues].data_ptr<std::int64_t>(),
         planes[summary_abi::kPlaneIntegerValues].stride(0),
         planes[summary_abi::kPlaneIntegerValues].stride(1)},
        {planes[summary_abi::kPlaneFloatValues].data_ptr<double>(),
         planes[summary_abi::kPlaneFloatValues].stride(0),
         planes[summary_abi::kPlaneFloatValues].stride(1)},
        {planes[summary_abi::kPlaneDigestValues].data_ptr<std::uint8_t>(),
         planes[summary_abi::kPlaneDigestValues].stride(0),
         planes[summary_abi::kPlaneDigestValues].stride(1)},
        {planes[summary_abi::kPlaneTextValues].data_ptr<std::uint8_t>(),
         planes[summary_abi::kPlaneTextValues].stride(0),
         planes[summary_abi::kPlaneTextValues].stride(1)},
    };
}

void run_into(
    const std::vector<torch::Tensor>& source_planes,
    const std::vector<torch::Tensor>& scratch_planes,
    const torch::Tensor& scratch_ready,
    const std::vector<torch::Tensor>& output_planes,
    const torch::Tensor& assemble_status,
    const torch::Tensor& producer_status,
    const std::int64_t generation
) {
    TORCH_CHECK(source_planes.size() == 6, "source_planes must contain six planes");
    TORCH_CHECK(
        scratch_planes.size() == summary_abi::kPlaneCountV2,
        "scratch_planes must contain seven ABI-v2 planes"
    );
    TORCH_CHECK(
        output_planes.size() == summary_abi::kPlaneCountV2,
        "output_planes must contain seven ABI-v2 planes"
    );
    const torch::Tensor& source_mask = source_planes[0];
    TORCH_CHECK(source_mask.dim() == 2, "source mask must be [cells,1]");
    const std::int64_t cells = source_mask.size(0);
    TORCH_CHECK(cells > 0, "at least one source cell is required");
    const torch::Device device = source_mask.device();

    const std::vector<std::vector<std::int64_t>> source_shapes = {
        {cells, 1},
        {cells, summary_abi::kStatusFieldCountV2},
        {cells, summary_abi::kIntegerFieldCountV2},
        {cells, summary_abi::kFloatFieldCountV2},
        {cells, summary_abi::kDigestFieldCountV2, summary_abi::kDigestBytesV2},
        {cells, summary_abi::kTextFieldCountV2, summary_abi::kTextRecordBytesV2},
    };
    for (std::int64_t index = 0; index < 6; ++index) {
        check_tensor(
            source_planes[index], device, kSourceDtypes[index],
            source_shapes[index], "source plane " + std::to_string(index)
        );
    }

    const std::vector<std::vector<std::int64_t>> scratch_shapes = {
        {cells},
        {cells, 1},
        {cells, summary_abi::kStatusFieldCountV2},
        {cells, summary_abi::kIntegerFieldCountV2},
        {cells, summary_abi::kFloatFieldCountV2},
        {cells, summary_abi::kDigestFieldCountV2, summary_abi::kDigestBytesV2},
        {cells, summary_abi::kTextFieldCountV2, summary_abi::kTextRecordBytesV2},
    };
    for (std::int64_t index = 0; index < summary_abi::kPlaneCountV2; ++index) {
        check_tensor(
            scratch_planes[index], device, kPlaneDtypes[index],
            scratch_shapes[index], "scratch plane " + std::to_string(index)
        );
    }
    check_tensor(
        scratch_ready, device, c10::ScalarType::Byte, {cells}, "scratch_ready"
    );
    check_tensor(
        assemble_status, device, c10::ScalarType::Int, {cells}, "assemble_status"
    );
    check_tensor(
        producer_status, device, c10::ScalarType::Int, {cells}, "producer_status"
    );

    const torch::Tensor& output_validity = output_planes[summary_abi::kPlaneValidity];
    TORCH_CHECK(output_validity.dim() == 2, "output validity must be [G,C]");
    const std::int64_t generations = output_validity.size(0);
    TORCH_CHECK(
        output_validity.size(1) == cells,
        "output cell count must match source"
    );
    TORCH_CHECK(
        generation >= 0 && generation < generations,
        "generation is outside the output range"
    );
    const std::vector<std::vector<std::int64_t>> output_shapes = {
        {generations, cells},
        {generations, cells, 1},
        {generations, cells, summary_abi::kStatusFieldCountV2},
        {generations, cells, summary_abi::kIntegerFieldCountV2},
        {generations, cells, summary_abi::kFloatFieldCountV2},
        {generations, cells, summary_abi::kDigestFieldCountV2, summary_abi::kDigestBytesV2},
        {generations, cells, summary_abi::kTextFieldCountV2, summary_abi::kTextRecordBytesV2},
    };
    for (std::int64_t index = 0; index < summary_abi::kPlaneCountV2; ++index) {
        check_tensor(
            output_planes[index], device, kPlaneDtypes[index],
            output_shapes[index], "output plane " + std::to_string(index)
        );
    }

    c10::cuda::CUDAGuard device_guard(device);
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream(device.index());
    eiel_cuda_architecture_b_compact_summary_pipeline_v1_launch(
        reinterpret_cast<const summary_abi::DefinedMaskV1*>(source_planes[0].data_ptr()),
        source_planes[1].data_ptr<std::int32_t>(),
        source_planes[2].data_ptr<std::int64_t>(),
        source_planes[3].data_ptr<double>(),
        source_planes[4].data_ptr<std::uint8_t>(),
        source_planes[5].data_ptr<std::uint8_t>(),
        scratch_ready.data_ptr<std::uint8_t>(),
        scratch_planes[summary_abi::kPlaneValidity].data_ptr<std::uint8_t>(),
        reinterpret_cast<summary_abi::DefinedMaskV1*>(
            scratch_planes[summary_abi::kPlaneDefinedMask].data_ptr()
        ),
        scratch_planes[summary_abi::kPlaneStatusCodes].data_ptr<std::int32_t>(),
        scratch_planes[summary_abi::kPlaneIntegerValues].data_ptr<std::int64_t>(),
        scratch_planes[summary_abi::kPlaneFloatValues].data_ptr<double>(),
        scratch_planes[summary_abi::kPlaneDigestValues].data_ptr<std::uint8_t>(),
        scratch_planes[summary_abi::kPlaneTextValues].data_ptr<std::uint8_t>(),
        make_scratch_view(scratch_planes, cells),
        make_output_view(output_planes, generations, cells),
        assemble_status.data_ptr<std::int32_t>(),
        producer_status.data_ptr<std::int32_t>(),
        cells,
        generation,
        stream
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

pybind11::dict resource_receipt() {
    cudaFuncAttributes assembler_attributes{};
    cudaFuncAttributes gate_attributes{};
    C10_CUDA_CHECK(
        eiel_cuda_architecture_b_compact_summary_assembler_v1_attributes(
            &assembler_attributes
        )
    );
    C10_CUDA_CHECK(
        eiel_cuda_architecture_b_compact_summary_pipeline_v1_gate_attributes(
            &gate_attributes
        )
    );
    const finalizer::KernelResourceReceiptV2 finalizer_receipt =
        finalizer::query_compact_summary_finalizer_v2_resources();

    pybind11::dict assembler;
    assembler["binary_version"] = assembler_attributes.binaryVersion;
    assembler["ptx_version"] = assembler_attributes.ptxVersion;
    assembler["registers_per_thread"] = assembler_attributes.numRegs;
    assembler["local_bytes_per_thread"] = assembler_attributes.localSizeBytes;
    assembler["shared_bytes_per_block"] = assembler_attributes.sharedSizeBytes;

    pybind11::dict gate;
    gate["binary_version"] = gate_attributes.binaryVersion;
    gate["ptx_version"] = gate_attributes.ptxVersion;
    gate["registers_per_thread"] = gate_attributes.numRegs;
    gate["local_bytes_per_thread"] = gate_attributes.localSizeBytes;
    gate["shared_bytes_per_block"] = gate_attributes.sharedSizeBytes;

    pybind11::dict finish;
    finish["binary_version"] = finalizer_receipt.binary_version;
    finish["ptx_version"] = finalizer_receipt.ptx_version;
    finish["registers_per_thread"] = finalizer_receipt.registers_per_thread;
    finish["local_bytes_per_thread"] = finalizer_receipt.local_bytes_per_thread;
    finish["shared_bytes_per_block"] = finalizer_receipt.shared_bytes_per_block;
    finish["occupancy_blocks_per_multiprocessor"] =
        finalizer_receipt.occupancy_blocks_per_multiprocessor;

    pybind11::dict result;
    result["assembler"] = assembler;
    result["status_gate"] = gate;
    result["finalizer"] = finish;
    result["cuda_kernel_launches_per_pipeline_call"] = 3;
    result["intermediate_synchronizations"] = 0;
    result["capability_promoted"] = false;
    return result;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def(
        "run_into",
        &run_into,
        "Run compact assembler and finalizer on one CUDA stream"
    );
    module.def("resource_receipt", &resource_receipt);
}
