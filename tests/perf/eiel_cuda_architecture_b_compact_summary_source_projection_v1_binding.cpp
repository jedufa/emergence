#include "cuda_architecture_b_compact_summary_source_projection_v1.cuh"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace projection =
    eiel::cuda_architecture_b::compact_summary_source_projection_v1;
namespace compact_abi = eiel::cuda_architecture_b::compact_summary_v2;

namespace {

constexpr std::array<c10::ScalarType, compact_abi::kPlaneCountV2> kPlaneDtypes = {
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

compact_abi::CompactSummaryViewV2 make_output_view(
    const std::vector<torch::Tensor>& planes,
    const std::int64_t generations,
    const std::int64_t cells
) {
    return {
        generations,
        cells,
        {planes[compact_abi::kPlaneValidity].data_ptr<std::uint8_t>(), cells, 1},
        {reinterpret_cast<compact_abi::DefinedMaskV1*>(
             planes[compact_abi::kPlaneDefinedMask].data_ptr()
         ),
         cells,
         1},
        {planes[compact_abi::kPlaneStatusCodes].data_ptr<std::int32_t>(),
         cells * compact_abi::kStatusFieldCountV2,
         compact_abi::kStatusFieldCountV2},
        {planes[compact_abi::kPlaneIntegerValues].data_ptr<std::int64_t>(),
         cells * compact_abi::kIntegerFieldCountV2,
         compact_abi::kIntegerFieldCountV2},
        {planes[compact_abi::kPlaneFloatValues].data_ptr<double>(),
         cells * compact_abi::kFloatFieldCountV2,
         compact_abi::kFloatFieldCountV2},
        {planes[compact_abi::kPlaneDigestValues].data_ptr<std::uint8_t>(),
         cells * compact_abi::kDigestFieldCountV2 * compact_abi::kDigestBytesV2,
         compact_abi::kDigestFieldCountV2 * compact_abi::kDigestBytesV2},
        {planes[compact_abi::kPlaneTextValues].data_ptr<std::uint8_t>(),
         cells * compact_abi::kTextFieldCountV2 * compact_abi::kTextRecordBytesV2,
         compact_abi::kTextFieldCountV2 * compact_abi::kTextRecordBytesV2},
    };
}

void project_into(
    const torch::Tensor& authority_bytes,
    const std::vector<torch::Tensor>& output_planes,
    const torch::Tensor& producer_status
) {
    TORCH_CHECK(
        output_planes.size() == compact_abi::kPlaneCountV2,
        "output_planes must contain the seven compact ABI-v2 planes"
    );
    TORCH_CHECK(
        authority_bytes.dim() == 2,
        "authority_bytes must be [generation*cell, authority_byte_width]"
    );
    const std::int64_t rows = authority_bytes.size(0);
    TORCH_CHECK(rows > 0, "at least one authority row is required");
    TORCH_CHECK(
        authority_bytes.size(1)
            == static_cast<std::int64_t>(projection::kAuthorityByteWidthV1),
        "authority byte width mismatch"
    );
    const torch::Device device = authority_bytes.device();
    TORCH_CHECK(authority_bytes.is_cuda(), "authority_bytes must be CUDA resident");
    TORCH_CHECK(authority_bytes.device() == device, "authority device mismatch");
    TORCH_CHECK(authority_bytes.is_contiguous(), "authority_bytes must be contiguous");
    TORCH_CHECK(
        authority_bytes.scalar_type() == c10::ScalarType::Byte,
        "authority_bytes must use torch.uint8"
    );

    const torch::Tensor& validity = output_planes[compact_abi::kPlaneValidity];
    TORCH_CHECK(validity.dim() == 2, "validity must be [generation,cell]");
    const std::int64_t generations = validity.size(0);
    const std::int64_t cells = validity.size(1);
    TORCH_CHECK(generations > 0 && cells > 0, "output dimensions must be positive");
    TORCH_CHECK(rows == generations * cells, "authority rows must cover every output row");

    const std::vector<std::vector<std::int64_t>> shapes = {
        {generations, cells},
        {generations, cells, 1},
        {generations, cells, compact_abi::kStatusFieldCountV2},
        {generations, cells, compact_abi::kIntegerFieldCountV2},
        {generations, cells, compact_abi::kFloatFieldCountV2},
        {generations, cells, compact_abi::kDigestFieldCountV2, compact_abi::kDigestBytesV2},
        {generations, cells, compact_abi::kTextFieldCountV2, compact_abi::kTextRecordBytesV2},
    };
    for (std::int64_t index = 0; index < compact_abi::kPlaneCountV2; ++index) {
        check_tensor(
            output_planes[index],
            device,
            kPlaneDtypes[index],
            shapes[index],
            "output plane " + std::to_string(index)
        );
    }
    check_tensor(
        producer_status,
        device,
        c10::ScalarType::Int,
        {generations, cells},
        "producer_status"
    );

    c10::cuda::CUDAGuard device_guard(device);
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream(device.index());
    C10_CUDA_CHECK(projection::launch_compact_summary_source_projection_v1(
        reinterpret_cast<
            const projection::CompactSummaryGenerationCellAuthorityV1*>(
            authority_bytes.data_ptr<std::uint8_t>()
        ),
        rows,
        make_output_view(output_planes, generations, cells),
        producer_status.data_ptr<std::int32_t>(),
        stream
    ));
}

pybind11::dict contract_receipt() {
    pybind11::dict result;
    result["source_version"] = projection::kAuthoritySourceVersionV1;
    result["authority_byte_width"] = projection::kAuthorityByteWidthV1;
    result["source_plane_count"] =
        static_cast<std::int64_t>(compact_abi::kPlaneCountV2);
    result["source_plane_logical_bytes"] =
        projection::kAuthoritySourcePlaneLogicalBytesV1;
    result["status_count"] =
        static_cast<std::int64_t>(compact_abi::kStatusFieldCountV2);
    result["integer_count"] =
        static_cast<std::int64_t>(compact_abi::kIntegerFieldCountV2);
    result["float_count"] =
        static_cast<std::int64_t>(compact_abi::kFloatFieldCountV2);
    result["digest_byte_count"] =
        compact_abi::kDigestFieldCountV2 * compact_abi::kDigestBytesV2;
    result["text_byte_count"] =
        compact_abi::kTextFieldCountV2 * compact_abi::kTextRecordBytesV2;
    result["status_success"] =
        static_cast<std::int32_t>(projection::kProjectionSuccess);
    result["status_code_invalid"] =
        static_cast<std::int32_t>(projection::kProjectionStatusCodeInvalid);
    result["status_index_invalid"] =
        static_cast<std::int32_t>(projection::kProjectionIndexInvalid);
    result["status_float_nonfinite"] =
        static_cast<std::int32_t>(projection::kProjectionFloatNonFinite);
    return result;
}

pybind11::dict resource_receipt() {
    cudaFuncAttributes attributes{};
    C10_CUDA_CHECK(
        projection::query_compact_summary_source_projection_v1_attributes(
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
        "project_into",
        &project_into,
        "Project complete native authority POD rows into preallocated ABI-v2 planes"
    );
    module.def("contract_receipt", &contract_receipt);
    module.def("resource_receipt", &resource_receipt);
}
