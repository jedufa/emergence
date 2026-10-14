#include "cuda_architecture_b_compact_summary_finalizer_v2.cuh"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace finalizer =
    eiel::cuda_architecture_b::compact_summary_finalizer_v2;
namespace summary_abi = eiel::cuda_architecture_b::compact_summary_v2;

namespace {

constexpr std::array<c10::ScalarType, summary_abi::kPlaneCountV2> kPlaneDtypes = {
    c10::ScalarType::Byte,
    c10::ScalarType::UInt64,
    c10::ScalarType::Int,
    c10::ScalarType::Long,
    c10::ScalarType::Double,
    c10::ScalarType::Byte,
    c10::ScalarType::Byte,
};

void check_tensor_base(
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

void check_shape(
    const torch::Tensor& tensor,
    const std::vector<std::int64_t>& expected,
    const std::string& name
) {
    TORCH_CHECK(
        tensor.sizes().vec() == expected,
        name,
        " must have shape ",
        expected,
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

void compact_summary_finalize_v2(
    const std::vector<torch::Tensor>& scratch_planes,
    const std::vector<torch::Tensor>& output_planes,
    torch::Tensor producer_status,
    const std::int64_t generation
) {
    TORCH_CHECK(
        scratch_planes.size() == summary_abi::kPlaneCountV2,
        "scratch_planes must contain the seven ABI v2 planes"
    );
    TORCH_CHECK(
        output_planes.size() == summary_abi::kPlaneCountV2,
        "output_planes must contain the seven ABI v2 planes"
    );
    const torch::Tensor& scratch_validity =
        scratch_planes[summary_abi::kPlaneValidity];
    TORCH_CHECK(scratch_validity.dim() == 1, "scratch validity must be [cells]");
    const std::int64_t cells = scratch_validity.size(0);
    TORCH_CHECK(cells > 0, "scratch must contain at least one cell");
    const torch::Device device = scratch_validity.device();

    const std::vector<std::vector<std::int64_t>> scratch_shapes = {
        {cells},
        {cells, 1},
        {cells, summary_abi::kStatusFieldCountV2},
        {cells, summary_abi::kIntegerFieldCountV2},
        {cells, summary_abi::kFloatFieldCountV2},
        {cells, summary_abi::kDigestFieldCountV2, summary_abi::kDigestBytesV2},
        {cells, summary_abi::kTextFieldCountV2, summary_abi::kTextRecordBytesV2},
    };
    for (std::int64_t plane = 0; plane < summary_abi::kPlaneCountV2; ++plane) {
        const std::string name = "scratch plane " + std::to_string(plane);
        check_tensor_base(
            scratch_planes[plane],
            device,
            kPlaneDtypes[plane],
            name
        );
        check_shape(scratch_planes[plane], scratch_shapes[plane], name);
    }

    const torch::Tensor& output_validity =
        output_planes[summary_abi::kPlaneValidity];
    TORCH_CHECK(output_validity.dim() == 2, "output validity must be [G,C]");
    const std::int64_t generations = output_validity.size(0);
    TORCH_CHECK(
        output_validity.size(1) == cells,
        "output validity cell count must match scratch"
    );
    TORCH_CHECK(
        generation >= 0 && generation < generations,
        "generation is outside the preallocated output range"
    );
    const std::vector<std::vector<std::int64_t>> output_shapes = {
        {generations, cells},
        {generations, cells, 1},
        {generations, cells, summary_abi::kStatusFieldCountV2},
        {generations, cells, summary_abi::kIntegerFieldCountV2},
        {generations, cells, summary_abi::kFloatFieldCountV2},
        {generations, cells, summary_abi::kDigestFieldCountV2,
         summary_abi::kDigestBytesV2},
        {generations, cells, summary_abi::kTextFieldCountV2,
         summary_abi::kTextRecordBytesV2},
    };
    for (std::int64_t plane = 0; plane < summary_abi::kPlaneCountV2; ++plane) {
        const std::string name = "output plane " + std::to_string(plane);
        check_tensor_base(
            output_planes[plane],
            device,
            kPlaneDtypes[plane],
            name
        );
        check_shape(output_planes[plane], output_shapes[plane], name);
    }

    check_tensor_base(
        producer_status,
        device,
        c10::ScalarType::Int,
        "producer_status"
    );
    check_shape(producer_status, {cells}, "producer_status");

    c10::cuda::CUDAGuard device_guard(device);
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream(device.index());
    finalizer::launch_compact_summary_finalizer_v2(
        make_scratch_view(scratch_planes, cells),
        make_output_view(output_planes, generations, cells),
        producer_status.data_ptr<std::int32_t>(),
        generation,
        stream
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

pybind11::dict resource_receipt_v2() {
    const finalizer::KernelResourceReceiptV2 receipt =
        finalizer::query_compact_summary_finalizer_v2_resources();
    pybind11::dict result;
    result["binary_version"] = receipt.binary_version;
    result["ptx_version"] = receipt.ptx_version;
    result["max_threads_per_block"] = receipt.max_threads_per_block;
    result["registers_per_thread"] = receipt.registers_per_thread;
    result["occupancy_blocks_per_multiprocessor"] =
        receipt.occupancy_blocks_per_multiprocessor;
    result["constant_bytes"] = receipt.constant_bytes;
    result["local_bytes_per_thread"] = receipt.local_bytes_per_thread;
    result["shared_bytes_per_block"] = receipt.shared_bytes_per_block;
    return result;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def(
        "finalize",
        &compact_summary_finalize_v2,
        "Allocation-free compact-summary ABI v2 scratch finalizer"
    );
    module.def(
        "resource_receipt",
        &resource_receipt_v2,
        "CUDA finalizer kernel resource receipt"
    );
}
