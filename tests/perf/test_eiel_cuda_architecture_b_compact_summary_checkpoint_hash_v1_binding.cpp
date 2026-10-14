#include "cuda_architecture_b_compact_summary_checkpoint_hash_v1.cuh"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <string>
#include <unordered_set>
#include <vector>

namespace checkpoint_hash =
    eiel::cuda_architecture_b::compact_summary_checkpoint_hash_v1;

namespace {

enum ProbeFaultV1 : std::int64_t {
    kNoProbeFaultV1 = 0,
    kNullDataProbeFaultV1 = 1,
    kInvalidDtypeProbeFaultV1 = 2,
    kInvalidModeProbeFaultV1 = 3,
    kInvalidLayoutProbeFaultV1 = 4,
    kNullSourceProbeFaultV1 = 5,
    kInvalidRawOutputProbeFaultV1 = 6,
    kInvalidCanonicalOutputProbeFaultV1 = 7,
};

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

bool supported_source_dtype(const c10::ScalarType dtype) {
    return dtype == c10::ScalarType::Bool
        || dtype == c10::ScalarType::Byte
        || dtype == c10::ScalarType::Char
        || dtype == c10::ScalarType::Short
        || dtype == c10::ScalarType::Int
        || dtype == c10::ScalarType::Long
        || dtype == c10::ScalarType::Float
        || dtype == c10::ScalarType::Double;
}

void require_same_size(
    const std::size_t actual,
    const std::size_t expected,
    const char* const name
) {
    TORCH_CHECK(actual == expected, name, " must contain ", expected, " entries");
}

void snapshot_into(
    const std::vector<torch::Tensor>& fields,
    const std::vector<std::int64_t>& modes,
    const std::vector<std::int64_t>& row_capacities,
    const std::vector<torch::Tensor>& alive_masks,
    const std::vector<torch::Tensor>& active_lengths,
    const std::vector<std::int64_t>& raw_output_indices,
    const std::vector<std::int64_t>& canonical_output_indices,
    const std::int64_t generation_index,
    torch::Tensor raw_digests,
    torch::Tensor canonical_digests,
    torch::Tensor status,
    torch::Tensor valid,
    const std::int64_t fault_field,
    const std::int64_t fault_kind
) {
    const std::size_t field_count = fields.size();
    TORCH_CHECK(field_count > 0, "at least one tensor field is required");
    require_same_size(modes.size(), field_count, "modes");
    require_same_size(row_capacities.size(), field_count, "row_capacities");
    require_same_size(alive_masks.size(), field_count, "alive_masks");
    require_same_size(active_lengths.size(), field_count, "active_lengths");
    require_same_size(raw_output_indices.size(), field_count, "raw_output_indices");
    require_same_size(
        canonical_output_indices.size(),
        field_count,
        "canonical_output_indices"
    );
    TORCH_CHECK(
        fault_field >= -1 && fault_field < static_cast<std::int64_t>(field_count),
        "fault_field is out of range"
    );
    TORCH_CHECK(
        fault_kind >= kNoProbeFaultV1
            && fault_kind <= kInvalidCanonicalOutputProbeFaultV1,
        "fault_kind is out of range"
    );
    TORCH_CHECK(
        (fault_field == -1) == (fault_kind == kNoProbeFaultV1),
        "fault_field and fault_kind must both select a fault or neither"
    );

    TORCH_CHECK(fields[0].defined() && fields[0].is_cuda(), "fields[0] must be CUDA");
    const torch::Device device = fields[0].device();
    TORCH_CHECK(raw_digests.dim() == 4, "raw_digests must be [G,C,F,32]");
    const std::int64_t generations = raw_digests.size(0);
    const std::int64_t cells = raw_digests.size(1);
    const std::int64_t output_fields = raw_digests.size(2);
    TORCH_CHECK(
        generations > 0 && cells > 0 && output_fields > 0
            && raw_digests.size(3) == checkpoint_hash::kDigestBytesV1,
        "raw_digests has an invalid extent"
    );
    TORCH_CHECK(
        generation_index >= 0 && generation_index < generations,
        "generation_index is out of range"
    );
    require_cuda_contiguous(raw_digests, device, "raw_digests");
    require_cuda_contiguous(canonical_digests, device, "canonical_digests");
    require_cuda_contiguous(status, device, "status");
    require_cuda_contiguous(valid, device, "valid");
    TORCH_CHECK(raw_digests.scalar_type() == c10::ScalarType::Byte, "raw_digests must be uint8");
    TORCH_CHECK(canonical_digests.scalar_type() == c10::ScalarType::Byte, "canonical_digests must be uint8");
    TORCH_CHECK(status.scalar_type() == c10::ScalarType::Int, "status must be int32");
    TORCH_CHECK(valid.scalar_type() == c10::ScalarType::Byte, "valid must be uint8");
    TORCH_CHECK(canonical_digests.sizes() == raw_digests.sizes(), "digest table shapes disagree");
    TORCH_CHECK(
        status.dim() == 3
            && status.size(0) == generations
            && status.size(1) == cells
            && status.size(2) == static_cast<std::int64_t>(field_count),
        "status must be [G,C,descriptor_fields]"
    );
    TORCH_CHECK(valid.sizes() == status.sizes(), "valid/status shapes disagree");

    std::unordered_set<std::int64_t> raw_indices;
    std::unordered_set<std::int64_t> canonical_indices;
    for (std::size_t index = 0; index < field_count; ++index) {
        TORCH_CHECK(
            raw_output_indices[index] >= 0 && raw_output_indices[index] < output_fields,
            "raw output index is out of range"
        );
        TORCH_CHECK(
            canonical_output_indices[index] >= 0
                && canonical_output_indices[index] < output_fields,
            "canonical output index is out of range"
        );
        TORCH_CHECK(raw_indices.insert(raw_output_indices[index]).second, "duplicate raw output index");
        TORCH_CHECK(
            canonical_indices.insert(canonical_output_indices[index]).second,
            "duplicate canonical output index"
        );
    }

    checkpoint_hash::SnapshotOutputsV1 outputs{
        raw_digests.data_ptr<std::uint8_t>(),
        canonical_digests.data_ptr<std::uint8_t>(),
        status.data_ptr<std::int32_t>(),
        valid.data_ptr<std::uint8_t>(),
        generations,
        cells,
        output_fields,
        static_cast<std::int64_t>(field_count),
        generation_index,
    };

    c10::cuda::CUDAGuard guard(device);
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream(device.index());
    for (std::size_t begin = 0; begin < field_count; begin += checkpoint_hash::kPointerBatchCapacityV1) {
        checkpoint_hash::TensorDigestPointerBatchV1 batch{};
        const std::size_t end = std::min(
            field_count,
            begin + static_cast<std::size_t>(checkpoint_hash::kPointerBatchCapacityV1)
        );
        batch.count = static_cast<std::int32_t>(end - begin);
        for (std::size_t index = begin; index < end; ++index) {
            const torch::Tensor& field = fields[index];
            require_cuda_contiguous(field, device, "field");
            TORCH_CHECK(field.dim() >= 1 && field.size(0) == cells, "field must start with cell dimension");
            TORCH_CHECK(supported_source_dtype(field.scalar_type()), "unsupported field dtype");
            TORCH_CHECK(row_capacities[index] > 0, "row capacity must be positive");
            TORCH_CHECK(field.numel() % cells == 0, "field extent is not cell-divisible");
            const std::uint64_t dtype_bytes = static_cast<std::uint64_t>(field.element_size());
            const std::uint64_t elements_per_cell = static_cast<std::uint64_t>(field.numel() / cells);
            TORCH_CHECK(
                elements_per_cell <= std::numeric_limits<std::uint64_t>::max() / dtype_bytes,
                "field byte extent overflows uint64"
            );
            const std::uint64_t bytes_per_cell = elements_per_cell * dtype_bytes;
            TORCH_CHECK(
                bytes_per_cell > 0
                    && bytes_per_cell % static_cast<std::uint64_t>(row_capacities[index]) == 0,
                "field bytes must divide exactly into logical rows"
            );
            const std::uint64_t row_bytes =
                bytes_per_cell / static_cast<std::uint64_t>(row_capacities[index]);

            const std::int64_t mode = modes[index];
            const std::uint8_t* mask_pointer = nullptr;
            const std::int64_t* length_pointer = nullptr;
            std::uint64_t mask_stride = 0U;
            if (mode == checkpoint_hash::kLiveRowsDeadZeroV1) {
                const torch::Tensor& mask = alive_masks[index];
                require_cuda_contiguous(mask, device, "alive mask");
                TORCH_CHECK(mask.scalar_type() == c10::ScalarType::Bool, "alive mask must be bool");
                TORCH_CHECK(
                    mask.dim() == 2
                        && mask.size(0) == cells
                        && mask.size(1) == row_capacities[index],
                    "alive mask must be [cells,row_capacity]"
                );
                mask_pointer = reinterpret_cast<const std::uint8_t*>(mask.data_ptr());
                mask_stride = static_cast<std::uint64_t>(row_capacities[index]);
            }
            if (
                mode == checkpoint_hash::kActivePrefixV1
                || mode
                    == checkpoint_hash::kTorchCpuMt19937PackedRowsToDenseActivePrefixV1
            ) {
                const torch::Tensor& lengths = active_lengths[index];
                require_cuda_contiguous(lengths, device, "active lengths");
                TORCH_CHECK(lengths.scalar_type() == c10::ScalarType::Long, "active lengths must be int64");
                TORCH_CHECK(
                    lengths.dim() == 1 && lengths.numel() == cells,
                    "active lengths must be [cells]"
                );
                length_pointer = lengths.data_ptr<std::int64_t>();
            }

            checkpoint_hash::TensorDigestPointerEntryV1 entry{
                reinterpret_cast<const std::uint8_t*>(field.data_ptr()),
                mask_pointer,
                length_pointer,
                bytes_per_cell,
                bytes_per_cell,
                static_cast<std::uint64_t>(row_capacities[index]),
                row_bytes,
                mask_stride,
                static_cast<std::uint32_t>(dtype_bytes),
                static_cast<std::int32_t>(mode),
                static_cast<std::int32_t>(raw_output_indices[index]),
                static_cast<std::int32_t>(canonical_output_indices[index]),
                static_cast<std::int32_t>(index),
            };
            if (static_cast<std::int64_t>(index) == fault_field) {
                if (fault_kind == kNullDataProbeFaultV1) {
                    entry.data = nullptr;
                } else if (fault_kind == kInvalidDtypeProbeFaultV1) {
                    entry.dtype_bytes = 3U;
                } else if (fault_kind == kInvalidModeProbeFaultV1) {
                    entry.canonicalization_mode = 99;
                } else if (fault_kind == kInvalidLayoutProbeFaultV1) {
                    entry.row_bytes += 1U;
                } else if (fault_kind == kNullSourceProbeFaultV1) {
                    entry.alive_mask = nullptr;
                    entry.active_lengths = nullptr;
                } else if (fault_kind == kInvalidRawOutputProbeFaultV1) {
                    entry.raw_output_index = static_cast<std::int32_t>(output_fields);
                } else if (fault_kind == kInvalidCanonicalOutputProbeFaultV1) {
                    entry.canonical_output_index = static_cast<std::int32_t>(output_fields);
                }
            }
            batch.entries[index - begin] = entry;
        }
        checkpoint_hash::launch_checkpoint_tensor_hash_snapshot_v1(batch, outputs, stream);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

pybind11::dict resource_receipt() {
    const checkpoint_hash::KernelResourceReceiptV1 receipt =
        checkpoint_hash::query_checkpoint_tensor_hash_snapshot_v1_resources();
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
        "snapshot_into",
        &snapshot_into,
        pybind11::arg("fields"),
        pybind11::arg("modes"),
        pybind11::arg("row_capacities"),
        pybind11::arg("alive_masks"),
        pybind11::arg("active_lengths"),
        pybind11::arg("raw_output_indices"),
        pybind11::arg("canonical_output_indices"),
        pybind11::arg("generation_index"),
        pybind11::arg("raw_digests"),
        pybind11::arg("canonical_digests"),
        pybind11::arg("status"),
        pybind11::arg("valid"),
        pybind11::arg("fault_field") = -1,
        pybind11::arg("fault_kind") = 0
    );
    module.def("resource_receipt", &resource_receipt);
    module.attr("contract_version") = checkpoint_hash::kContractVersionV1;
    module.attr("mode_exact_state") = static_cast<std::int32_t>(
        checkpoint_hash::kExactStateRawBytesV1
    );
    module.attr("mode_live_rows") = static_cast<std::int32_t>(
        checkpoint_hash::kLiveRowsDeadZeroV1
    );
    module.attr("mode_active_prefix") = static_cast<std::int32_t>(
        checkpoint_hash::kActivePrefixV1
    );
    module.attr("mode_torch_cpu_mt19937_packed_rows_to_dense_active_prefix") =
        static_cast<std::int32_t>(
            checkpoint_hash::kTorchCpuMt19937PackedRowsToDenseActivePrefixV1
        );
    module.attr("status_success") = static_cast<std::int32_t>(
        checkpoint_hash::kSnapshotSuccessV1
    );
}
