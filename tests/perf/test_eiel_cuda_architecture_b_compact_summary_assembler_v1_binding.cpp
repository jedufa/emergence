#include "cuda_architecture_b_compact_summary_assembler_v1.cuh"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace compact_abi = eiel::cuda_architecture_b::compact_summary_v2;

void eiel_cuda_architecture_b_compact_summary_assembler_v1_launch(
    const compact_abi::DefinedMaskV1* source_defined_masks,
    const std::int32_t* source_status_codes,
    const std::int64_t* source_integer_values,
    const double* source_float_values,
    const std::uint8_t* source_digest_values,
    const std::uint8_t* source_text_values,
    std::uint8_t* scratch_ready,
    std::uint8_t* scratch_validity,
    compact_abi::DefinedMaskV1* scratch_defined_masks,
    std::int32_t* scratch_status_codes,
    std::int64_t* scratch_integer_values,
    double* scratch_float_values,
    std::uint8_t* scratch_digest_values,
    std::uint8_t* scratch_text_values,
    std::int32_t* assemble_status,
    std::int64_t row_count,
    cudaStream_t stream
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

constexpr std::array<c10::ScalarType, compact_abi::kPlaneCountV2>
kScratchDtypes = {
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

void assemble_into(
    const std::vector<torch::Tensor>& source_planes,
    const std::vector<torch::Tensor>& scratch_planes,
    const torch::Tensor& scratch_ready,
    const torch::Tensor& assemble_status
) {
    TORCH_CHECK(
        source_planes.size() == 6,
        "source_planes must contain mask, status, integer, float, digest, text"
    );
    TORCH_CHECK(
        scratch_planes.size() == compact_abi::kPlaneCountV2,
        "scratch_planes must contain the seven ABI-v2 planes"
    );
    const torch::Tensor& source_mask = source_planes[0];
    TORCH_CHECK(source_mask.dim() == 2, "source mask must be [rows,1]");
    const std::int64_t rows = source_mask.size(0);
    TORCH_CHECK(rows > 0, "at least one source row is required");
    const torch::Device device = source_mask.device();

    const std::vector<std::vector<std::int64_t>> source_shapes = {
        {rows, 1},
        {rows, compact_abi::kStatusFieldCountV2},
        {rows, compact_abi::kIntegerFieldCountV2},
        {rows, compact_abi::kFloatFieldCountV2},
        {rows, compact_abi::kDigestFieldCountV2, compact_abi::kDigestBytesV2},
        {rows, compact_abi::kTextFieldCountV2,
         compact_abi::kTextRecordBytesV2},
    };
    for (std::int64_t index = 0; index < 6; ++index) {
        check_tensor(
            source_planes[index],
            device,
            kSourceDtypes[index],
            source_shapes[index],
            "source plane " + std::to_string(index)
        );
    }

    const std::vector<std::vector<std::int64_t>> scratch_shapes = {
        {rows},
        {rows, 1},
        {rows, compact_abi::kStatusFieldCountV2},
        {rows, compact_abi::kIntegerFieldCountV2},
        {rows, compact_abi::kFloatFieldCountV2},
        {rows, compact_abi::kDigestFieldCountV2, compact_abi::kDigestBytesV2},
        {rows, compact_abi::kTextFieldCountV2,
         compact_abi::kTextRecordBytesV2},
    };
    for (
        std::int64_t index = 0;
        index < compact_abi::kPlaneCountV2;
        ++index
    ) {
        check_tensor(
            scratch_planes[index],
            device,
            kScratchDtypes[index],
            scratch_shapes[index],
            "scratch plane " + std::to_string(index)
        );
    }
    check_tensor(
        scratch_ready,
        device,
        c10::ScalarType::Byte,
        {rows},
        "scratch_ready"
    );
    check_tensor(
        assemble_status,
        device,
        c10::ScalarType::Int,
        {rows},
        "assemble_status"
    );

    c10::cuda::CUDAGuard device_guard(device);
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream(device.index());
    eiel_cuda_architecture_b_compact_summary_assembler_v1_launch(
        reinterpret_cast<const compact_abi::DefinedMaskV1*>(
            source_planes[0].data_ptr()
        ),
        source_planes[1].data_ptr<std::int32_t>(),
        source_planes[2].data_ptr<std::int64_t>(),
        source_planes[3].data_ptr<double>(),
        source_planes[4].data_ptr<std::uint8_t>(),
        source_planes[5].data_ptr<std::uint8_t>(),
        scratch_ready.data_ptr<std::uint8_t>(),
        scratch_planes[compact_abi::kPlaneValidity].data_ptr<std::uint8_t>(),
        reinterpret_cast<compact_abi::DefinedMaskV1*>(
            scratch_planes[compact_abi::kPlaneDefinedMask].data_ptr()
        ),
        scratch_planes[compact_abi::kPlaneStatusCodes]
            .data_ptr<std::int32_t>(),
        scratch_planes[compact_abi::kPlaneIntegerValues]
            .data_ptr<std::int64_t>(),
        scratch_planes[compact_abi::kPlaneFloatValues].data_ptr<double>(),
        scratch_planes[compact_abi::kPlaneDigestValues]
            .data_ptr<std::uint8_t>(),
        scratch_planes[compact_abi::kPlaneTextValues]
            .data_ptr<std::uint8_t>(),
        assemble_status.data_ptr<std::int32_t>(),
        rows,
        stream
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

pybind11::dict resource_receipt() {
    cudaFuncAttributes attributes{};
    C10_CUDA_CHECK(
        eiel_cuda_architecture_b_compact_summary_assembler_v1_attributes(
            &attributes
        )
    );
    pybind11::dict result;
    result["binary_version"] = attributes.binaryVersion;
    result["ptx_version"] = attributes.ptxVersion;
    result["registers_per_thread"] = attributes.numRegs;
    result["local_bytes_per_thread"] = attributes.localSizeBytes;
    result["shared_bytes_per_block"] = attributes.sharedSizeBytes;
    result["constant_bytes"] = attributes.constSizeBytes;
    result["max_threads_per_block"] = attributes.maxThreadsPerBlock;
    return result;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def(
        "assemble_into",
        &assemble_into,
        "Assemble semantic boundary sources into compact ABI-v2 scratch rows"
    );
    module.def(
        "resource_receipt",
        &resource_receipt,
        "CUDA compact-summary assembler resource receipt"
    );
}
