#include "cuda_architecture_b_compact_summary_boundary_v1.cuh"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include <array>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace boundary =
    eiel::cuda_architecture_b::compact_summary_boundary_v1;
namespace checkpoint_hash =
    eiel::cuda_architecture_b::compact_summary_checkpoint_hash_v1;
namespace reductions =
    eiel::cuda_architecture_b::compact_summary_reductions_v1;

namespace {

namespace abi = eiel::cuda_architecture_b::compact_summary_v2;

constexpr std::int64_t kTypedColumnCountV1 = 16;
constexpr std::int64_t kCursorSourceCountV1 = 9;
constexpr std::int64_t kIdentityScalarCountV1 = 3;
constexpr std::int32_t kMaximumProbeBatchesV1 = 4;

constexpr std::array<c10::ScalarType, kTypedColumnCountV1> kTypedDtypesV1 = {
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

constexpr std::array<c10::ScalarType, 6> kSourceDtypesV1 = {
    c10::ScalarType::UInt64,
    c10::ScalarType::Int,
    c10::ScalarType::Long,
    c10::ScalarType::Double,
    c10::ScalarType::Byte,
    c10::ScalarType::Byte,
};

constexpr std::array<c10::ScalarType, abi::kPlaneCountV2> kPlaneDtypesV1 = {
    c10::ScalarType::Byte,
    c10::ScalarType::UInt64,
    c10::ScalarType::Int,
    c10::ScalarType::Long,
    c10::ScalarType::Double,
    c10::ScalarType::Byte,
    c10::ScalarType::Byte,
};

void require_size(
    const std::size_t actual,
    const std::size_t expected,
    const char* const name
) {
    TORCH_CHECK(actual == expected, name, " must contain ", expected, " entries");
}

void require_tensor(
    const torch::Tensor& tensor,
    const torch::Device& device,
    const c10::ScalarType dtype,
    const std::vector<std::int64_t>& shape,
    const std::string& name
) {
    TORCH_CHECK(tensor.defined(), name, " must be defined");
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

abi::CompactSummaryViewV2 make_scratch_view(
    const std::vector<torch::Tensor>& planes,
    const std::int64_t cells
) {
    return {
        1,
        cells,
        {planes[abi::kPlaneValidity].data_ptr<std::uint8_t>(), 0,
         planes[abi::kPlaneValidity].stride(0)},
        {reinterpret_cast<abi::DefinedMaskV1*>(
             planes[abi::kPlaneDefinedMask].data_ptr()),
         0,
         planes[abi::kPlaneDefinedMask].stride(0)},
        {planes[abi::kPlaneStatusCodes].data_ptr<std::int32_t>(), 0,
         planes[abi::kPlaneStatusCodes].stride(0)},
        {planes[abi::kPlaneIntegerValues].data_ptr<std::int64_t>(), 0,
         planes[abi::kPlaneIntegerValues].stride(0)},
        {planes[abi::kPlaneFloatValues].data_ptr<double>(), 0,
         planes[abi::kPlaneFloatValues].stride(0)},
        {planes[abi::kPlaneDigestValues].data_ptr<std::uint8_t>(), 0,
         planes[abi::kPlaneDigestValues].stride(0)},
        {planes[abi::kPlaneTextValues].data_ptr<std::uint8_t>(), 0,
         planes[abi::kPlaneTextValues].stride(0)},
    };
}

abi::CompactSummaryViewV2 make_output_view(
    const std::vector<torch::Tensor>& planes,
    const std::int64_t generations,
    const std::int64_t cells
) {
    return {
        generations,
        cells,
        {planes[abi::kPlaneValidity].data_ptr<std::uint8_t>(),
         planes[abi::kPlaneValidity].stride(0),
         planes[abi::kPlaneValidity].stride(1)},
        {reinterpret_cast<abi::DefinedMaskV1*>(
             planes[abi::kPlaneDefinedMask].data_ptr()),
         planes[abi::kPlaneDefinedMask].stride(0),
         planes[abi::kPlaneDefinedMask].stride(1)},
        {planes[abi::kPlaneStatusCodes].data_ptr<std::int32_t>(),
         planes[abi::kPlaneStatusCodes].stride(0),
         planes[abi::kPlaneStatusCodes].stride(1)},
        {planes[abi::kPlaneIntegerValues].data_ptr<std::int64_t>(),
         planes[abi::kPlaneIntegerValues].stride(0),
         planes[abi::kPlaneIntegerValues].stride(1)},
        {planes[abi::kPlaneFloatValues].data_ptr<double>(),
         planes[abi::kPlaneFloatValues].stride(0),
         planes[abi::kPlaneFloatValues].stride(1)},
        {planes[abi::kPlaneDigestValues].data_ptr<std::uint8_t>(),
         planes[abi::kPlaneDigestValues].stride(0),
         planes[abi::kPlaneDigestValues].stride(1)},
        {planes[abi::kPlaneTextValues].data_ptr<std::uint8_t>(),
         planes[abi::kPlaneTextValues].stride(0),
         planes[abi::kPlaneTextValues].stride(1)},
    };
}

reductions::ReductionInputsV1 make_reduction_inputs(
    const std::vector<torch::Tensor>& typed_columns,
    const std::vector<torch::Tensor>& cursor_sources,
    const torch::Tensor& active_agent_ids,
    const std::vector<torch::Tensor>& identity_scalars,
    const std::int64_t slot_capacity,
    const torch::Device& device,
    const std::int64_t cells
) {
    require_size(typed_columns.size(), kTypedColumnCountV1, "typed_columns");
    require_size(cursor_sources.size(), kCursorSourceCountV1, "cursor_sources");
    require_size(identity_scalars.size(), kIdentityScalarCountV1, "identity_scalars");
    TORCH_CHECK(typed_columns[0].dim() == 2, "typed columns must be [C,E]");
    const std::int64_t event_capacity = typed_columns[0].size(1);
    for (std::int64_t index = 0; index < kTypedColumnCountV1; ++index) {
        require_tensor(
            typed_columns[index],
            device,
            kTypedDtypesV1[index],
            {cells, event_capacity},
            "typed column " + std::to_string(index)
        );
    }
    for (std::int64_t index = 0; index < kCursorSourceCountV1; ++index) {
        require_tensor(
            cursor_sources[index],
            device,
            c10::ScalarType::Long,
            {cells},
            "cursor source " + std::to_string(index)
        );
    }
    TORCH_CHECK(active_agent_ids.dim() == 2, "active ids must be [C,A]");
    const std::int64_t active_capacity = active_agent_ids.size(1);
    require_tensor(
        active_agent_ids,
        device,
        c10::ScalarType::Long,
        {cells, active_capacity},
        "active_agent_ids"
    );
    for (std::int64_t index = 0; index < kIdentityScalarCountV1; ++index) {
        require_tensor(
            identity_scalars[index],
            device,
            c10::ScalarType::Long,
            {cells},
            "identity scalar " + std::to_string(index)
        );
    }
    TORCH_CHECK(slot_capacity > 0, "slot_capacity must be positive");
    return {
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
        {
            cursor_sources[0].data_ptr<std::int64_t>(),
            cursor_sources[1].data_ptr<std::int64_t>(),
            cursor_sources[2].data_ptr<std::int64_t>(),
            cursor_sources[3].data_ptr<std::int64_t>(),
            cursor_sources[4].data_ptr<std::int64_t>(),
            cursor_sources[5].data_ptr<std::int64_t>(),
            cursor_sources[6].data_ptr<std::int64_t>(),
            cursor_sources[7].data_ptr<std::int64_t>(),
            cursor_sources[8].data_ptr<std::int64_t>(),
        },
        {
            active_agent_ids.data_ptr<std::int64_t>(),
            identity_scalars[0].data_ptr<std::int64_t>(),
            identity_scalars[1].data_ptr<std::int64_t>(),
            identity_scalars[2].data_ptr<std::int64_t>(),
            active_capacity,
            active_capacity,
            slot_capacity,
        },
        cells,
    };
}

struct ProbeCheckpointBatchesV1 {
    std::array<checkpoint_hash::TensorDigestPointerBatchV1,
               kMaximumProbeBatchesV1> batches;
    std::int32_t batch_count;
};

ProbeCheckpointBatchesV1 make_checkpoint_batches(
    const std::vector<torch::Tensor>& fields,
    const std::vector<std::int64_t>& modes,
    const std::vector<std::int64_t>& row_capacities,
    const std::vector<torch::Tensor>& alive_masks,
    const std::vector<torch::Tensor>& active_lengths,
    const torch::Device& device,
    const std::int64_t cells,
    const std::int64_t output_fields,
    const std::int64_t checkpoint_fault
) {
    const std::size_t field_count = fields.size();
    require_size(modes.size(), field_count, "checkpoint modes");
    require_size(row_capacities.size(), field_count, "checkpoint row capacities");
    require_size(alive_masks.size(), field_count, "checkpoint alive masks");
    require_size(active_lengths.size(), field_count, "checkpoint active lengths");
    TORCH_CHECK(checkpoint_fault >= 0 && checkpoint_fault <= 2, "invalid checkpoint fault");
    TORCH_CHECK(
        field_count <= static_cast<std::size_t>(
            kMaximumProbeBatchesV1 * checkpoint_hash::kPointerBatchCapacityV1
        ),
        "probe checkpoint field capacity exceeded"
    );
    ProbeCheckpointBatchesV1 result{};
    for (std::size_t index = 0; index < field_count; ++index) {
        const torch::Tensor& field = fields[index];
        TORCH_CHECK(field.dim() >= 1 && field.size(0) == cells, "checkpoint field must start with C");
        TORCH_CHECK(field.is_cuda() && field.device() == device && field.is_contiguous(), "checkpoint field storage mismatch");
        TORCH_CHECK(row_capacities[index] > 0, "checkpoint row capacity must be positive");
        TORCH_CHECK(field.numel() % cells == 0, "checkpoint field must be cell-divisible");
        const std::uint64_t dtype_bytes = field.element_size();
        const std::uint64_t bytes_per_cell =
            static_cast<std::uint64_t>(field.numel() / cells) * dtype_bytes;
        TORCH_CHECK(
            bytes_per_cell > 0
                && bytes_per_cell % static_cast<std::uint64_t>(row_capacities[index]) == 0,
            "checkpoint row layout mismatch"
        );
        const std::uint64_t row_bytes = bytes_per_cell
            / static_cast<std::uint64_t>(row_capacities[index]);
        const std::uint8_t* alive = nullptr;
        const std::int64_t* active = nullptr;
        std::uint64_t alive_stride = 0U;
        if (modes[index] == checkpoint_hash::kLiveRowsDeadZeroV1) {
            require_tensor(
                alive_masks[index], device, c10::ScalarType::Bool,
                {cells, row_capacities[index]}, "checkpoint alive mask"
            );
            alive = reinterpret_cast<const std::uint8_t*>(alive_masks[index].data_ptr());
            alive_stride = static_cast<std::uint64_t>(row_capacities[index]);
        }
        if (modes[index] == checkpoint_hash::kActivePrefixV1) {
            require_tensor(
                active_lengths[index], device, c10::ScalarType::Long,
                {cells}, "checkpoint active lengths"
            );
            active = active_lengths[index].data_ptr<std::int64_t>();
        }
        const std::size_t batch_index = index
            / checkpoint_hash::kPointerBatchCapacityV1;
        const std::size_t entry_index = index
            % checkpoint_hash::kPointerBatchCapacityV1;
        auto& entry = result.batches[batch_index].entries[entry_index];
        entry = {
            reinterpret_cast<const std::uint8_t*>(field.data_ptr()),
            alive,
            active,
            bytes_per_cell,
            bytes_per_cell,
            static_cast<std::uint64_t>(row_capacities[index]),
            row_bytes,
            alive_stride,
            static_cast<std::uint32_t>(dtype_bytes),
            static_cast<std::int32_t>(modes[index]),
            static_cast<std::int32_t>(index),
            static_cast<std::int32_t>(index),
            static_cast<std::int32_t>(index),
        };
        if (index == 0 && checkpoint_fault == 1) {
            entry.dtype_bytes = 3U;
        } else if (index == 0 && checkpoint_fault == 2) {
            entry.canonicalization_mode = 99;
        }
    }
    result.batch_count = static_cast<std::int32_t>(
        (field_count + checkpoint_hash::kPointerBatchCapacityV1 - 1)
        / checkpoint_hash::kPointerBatchCapacityV1
    );
    for (std::int32_t batch = 0; batch < result.batch_count; ++batch) {
        const std::size_t begin = static_cast<std::size_t>(batch)
            * checkpoint_hash::kPointerBatchCapacityV1;
        const std::size_t remaining = field_count - begin;
        result.batches[batch].count = static_cast<std::int32_t>(
            remaining < static_cast<std::size_t>(checkpoint_hash::kPointerBatchCapacityV1)
                ? remaining
                : checkpoint_hash::kPointerBatchCapacityV1
        );
    }
    TORCH_CHECK(
        output_fields == static_cast<std::int64_t>(field_count),
        "probe uses one output slot per checkpoint field"
    );
    return result;
}

pybind11::dict receipt_dict(
    const boundary::BoundaryLaunchReceiptV1& receipt
) {
    pybind11::dict result;
    result["status"] = receipt.status;
    result["provided_host_canonical_digest_mask"] =
        receipt.provided_host_canonical_digest_mask;
    result["missing_host_canonical_digest_mask"] =
        receipt.missing_host_canonical_digest_mask;
    result["unresolved_live_metadata_mask"] =
        receipt.unresolved_live_metadata_mask;
    result["canonical_row_submission_capable"] =
        receipt.canonical_row_submission_capable != 0U;
    result["runtime_capability_promoted"] =
        receipt.runtime_capability_promoted != 0U;
    result["validity_reset_kernel_launches"] =
        receipt.validity_reset_kernel_launches;
    result["reduction_kernel_launches"] = receipt.reduction_kernel_launches;
    result["checkpoint_hash_kernel_launches"] =
        receipt.checkpoint_hash_kernel_launches;
    result["prerequisite_gate_kernel_launches"] =
        receipt.prerequisite_gate_kernel_launches;
    result["accepted_pipeline_kernel_launches"] =
        receipt.accepted_pipeline_kernel_launches;
    result["total_kernel_launches"] = receipt.total_kernel_launches;
    result["intermediate_synchronizations"] =
        receipt.intermediate_synchronizations;
    result["device_allocations"] = receipt.device_allocations;
    result["host_allocations"] = receipt.host_allocations;
    result["retained_generation_index"] = receipt.retained.generation_index;
    result["retained_cell_count"] = receipt.retained.cell_count;
    result["retained_lifecycle_identity"] =
        receipt.retained.lifecycle_identity != nullptr;
    result["retained_checkpoint_outputs"] =
        receipt.retained.checkpoint_outputs.cell_count == receipt.retained.cell_count;
    result["retained_completed_host_digest_values"] =
        receipt.retained.completed_host_digest_values != nullptr;
    return result;
}

pybind11::dict run_into(
    const std::vector<torch::Tensor>& typed_columns,
    const std::vector<torch::Tensor>& cursor_sources,
    torch::Tensor active_agent_ids,
    const std::vector<torch::Tensor>& identity_scalars,
    const std::int64_t slot_capacity,
    const std::vector<torch::Tensor>& checkpoint_fields,
    const std::vector<std::int64_t>& checkpoint_modes,
    const std::vector<std::int64_t>& checkpoint_row_capacities,
    const std::vector<torch::Tensor>& checkpoint_alive_masks,
    const std::vector<torch::Tensor>& checkpoint_active_lengths,
    const std::vector<torch::Tensor>& source_planes,
    const std::vector<torch::Tensor>& scratch_planes,
    torch::Tensor scratch_ready,
    const std::vector<torch::Tensor>& output_planes,
    torch::Tensor reduction_result_bytes,
    torch::Tensor checkpoint_raw_digests,
    torch::Tensor checkpoint_canonical_digests,
    torch::Tensor checkpoint_status,
    torch::Tensor checkpoint_valid,
    torch::Tensor assemble_status,
    torch::Tensor producer_status,
    const std::int64_t generation_index,
    const std::uint32_t provided_host_digest_mask,
    const std::int64_t checkpoint_fault
) {
    require_size(source_planes.size(), 6, "source_planes");
    require_size(scratch_planes.size(), abi::kPlaneCountV2, "scratch_planes");
    require_size(output_planes.size(), abi::kPlaneCountV2, "output_planes");
    TORCH_CHECK(source_planes[0].dim() == 2, "source mask must be [C,1]");
    const std::int64_t cells = source_planes[0].size(0);
    TORCH_CHECK(cells > 0, "at least one cell is required");
    const torch::Device device = source_planes[0].device();

    const std::vector<std::vector<std::int64_t>> source_shapes = {
        {cells, 1},
        {cells, abi::kStatusFieldCountV2},
        {cells, abi::kIntegerFieldCountV2},
        {cells, abi::kFloatFieldCountV2},
        {cells, abi::kDigestFieldCountV2, abi::kDigestBytesV2},
        {cells, abi::kTextFieldCountV2, abi::kTextRecordBytesV2},
    };
    for (std::int64_t index = 0; index < 6; ++index) {
        require_tensor(
            source_planes[index], device, kSourceDtypesV1[index],
            source_shapes[index], "source plane " + std::to_string(index)
        );
    }
    const std::vector<std::vector<std::int64_t>> scratch_shapes = {
        {cells},
        {cells, 1},
        {cells, abi::kStatusFieldCountV2},
        {cells, abi::kIntegerFieldCountV2},
        {cells, abi::kFloatFieldCountV2},
        {cells, abi::kDigestFieldCountV2, abi::kDigestBytesV2},
        {cells, abi::kTextFieldCountV2, abi::kTextRecordBytesV2},
    };
    for (std::int64_t index = 0; index < abi::kPlaneCountV2; ++index) {
        require_tensor(
            scratch_planes[index], device, kPlaneDtypesV1[index],
            scratch_shapes[index], "scratch plane " + std::to_string(index)
        );
    }
    require_tensor(scratch_ready, device, c10::ScalarType::Byte, {cells}, "scratch_ready");
    require_tensor(assemble_status, device, c10::ScalarType::Int, {cells}, "assemble_status");
    require_tensor(producer_status, device, c10::ScalarType::Int, {cells}, "producer_status");

    TORCH_CHECK(output_planes[abi::kPlaneValidity].dim() == 2, "output validity must be [G,C]");
    const std::int64_t generations = output_planes[abi::kPlaneValidity].size(0);
    TORCH_CHECK(generation_index >= 0 && generation_index < generations, "generation out of range");
    const std::vector<std::vector<std::int64_t>> output_shapes = {
        {generations, cells},
        {generations, cells, 1},
        {generations, cells, abi::kStatusFieldCountV2},
        {generations, cells, abi::kIntegerFieldCountV2},
        {generations, cells, abi::kFloatFieldCountV2},
        {generations, cells, abi::kDigestFieldCountV2, abi::kDigestBytesV2},
        {generations, cells, abi::kTextFieldCountV2, abi::kTextRecordBytesV2},
    };
    for (std::int64_t index = 0; index < abi::kPlaneCountV2; ++index) {
        require_tensor(
            output_planes[index], device, kPlaneDtypesV1[index],
            output_shapes[index], "output plane " + std::to_string(index)
        );
    }

    require_tensor(
        reduction_result_bytes,
        device,
        c10::ScalarType::Byte,
        {generations, cells, static_cast<std::int64_t>(sizeof(reductions::ReductionResultV1))},
        "reduction_result_bytes"
    );
    TORCH_CHECK(checkpoint_raw_digests.dim() == 4, "checkpoint raw digests must be [G,C,F,32]");
    const std::int64_t checkpoint_fields_count = checkpoint_raw_digests.size(2);
    require_tensor(
        checkpoint_raw_digests, device, c10::ScalarType::Byte,
        {generations, cells, checkpoint_fields_count, checkpoint_hash::kDigestBytesV1},
        "checkpoint_raw_digests"
    );
    require_tensor(
        checkpoint_canonical_digests, device, c10::ScalarType::Byte,
        {generations, cells, checkpoint_fields_count, checkpoint_hash::kDigestBytesV1},
        "checkpoint_canonical_digests"
    );
    require_tensor(
        checkpoint_status, device, c10::ScalarType::Int,
        {generations, cells, checkpoint_fields_count}, "checkpoint_status"
    );
    require_tensor(
        checkpoint_valid, device, c10::ScalarType::Byte,
        {generations, cells, checkpoint_fields_count}, "checkpoint_valid"
    );

    const reductions::ReductionInputsV1 reduction_inputs = make_reduction_inputs(
        typed_columns,
        cursor_sources,
        active_agent_ids,
        identity_scalars,
        slot_capacity,
        device,
        cells
    );
    const ProbeCheckpointBatchesV1 checkpoint_batches = make_checkpoint_batches(
        checkpoint_fields,
        checkpoint_modes,
        checkpoint_row_capacities,
        checkpoint_alive_masks,
        checkpoint_active_lengths,
        device,
        cells,
        checkpoint_fields_count,
        checkpoint_fault
    );
    checkpoint_hash::SnapshotOutputsV1 checkpoint_outputs{
        checkpoint_raw_digests.data_ptr<std::uint8_t>(),
        checkpoint_canonical_digests.data_ptr<std::uint8_t>(),
        checkpoint_status.data_ptr<std::int32_t>(),
        checkpoint_valid.data_ptr<std::uint8_t>(),
        generations,
        cells,
        checkpoint_fields_count,
        checkpoint_fields_count,
        generation_index,
    };
    reductions::ReductionResultV1* const reduction_output =
        reinterpret_cast<reductions::ReductionResultV1*>(
            reduction_result_bytes.data_ptr<std::uint8_t>()
            + generation_index * cells * sizeof(reductions::ReductionResultV1)
        );
    const abi::CompactSummaryViewV2 scratch_view =
        make_scratch_view(scratch_planes, cells);
    const abi::CompactSummaryViewV2 output_view =
        make_output_view(output_planes, generations, cells);
    boundary::CompactSummaryBoundaryRequestV1 request{
        generation_index,
        cells,
        reduction_inputs,
        reduction_output,
        checkpoint_batches.batch_count == 0
            ? nullptr
            : checkpoint_batches.batches.data(),
        checkpoint_batches.batch_count,
        checkpoint_outputs,
        {
            reinterpret_cast<const abi::DefinedMaskV1*>(source_planes[0].data_ptr()),
            source_planes[1].data_ptr<std::int32_t>(),
            source_planes[2].data_ptr<std::int64_t>(),
            source_planes[3].data_ptr<double>(),
            source_planes[4].data_ptr<std::uint8_t>(),
            source_planes[5].data_ptr<std::uint8_t>(),
            scratch_ready.data_ptr<std::uint8_t>(),
            scratch_planes[abi::kPlaneValidity].data_ptr<std::uint8_t>(),
            reinterpret_cast<abi::DefinedMaskV1*>(
                scratch_planes[abi::kPlaneDefinedMask].data_ptr()
            ),
            scratch_planes[abi::kPlaneStatusCodes].data_ptr<std::int32_t>(),
            scratch_planes[abi::kPlaneIntegerValues].data_ptr<std::int64_t>(),
            scratch_planes[abi::kPlaneFloatValues].data_ptr<double>(),
            scratch_planes[abi::kPlaneDigestValues].data_ptr<std::uint8_t>(),
            scratch_planes[abi::kPlaneTextValues].data_ptr<std::uint8_t>(),
            scratch_view,
            output_view,
            assemble_status.data_ptr<std::int32_t>(),
            producer_status.data_ptr<std::int32_t>(),
        },
        {nullptr, nullptr, nullptr, 0, 0, 0, 0},
        provided_host_digest_mask,
    };

    c10::cuda::CUDAGuard guard(device);
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream(device.index());
    const boundary::BoundaryLaunchReceiptV1 receipt =
        boundary::launch_compact_summary_boundary_v1(request, stream);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return receipt_dict(receipt);
}

pybind11::dict resource_receipt() {
    const boundary::BoundaryKernelResourceReceiptV1 receipt =
        boundary::query_compact_summary_boundary_v1_resources();
    const auto reduction = reductions::query_compact_summary_reductions_v1_resources();
    const auto checkpoint = checkpoint_hash::query_checkpoint_tensor_hash_snapshot_v1_resources();
    auto one = [](const auto& value) {
        pybind11::dict result;
        result["binary_version"] = value.binary_version;
        result["ptx_version"] = value.ptx_version;
        result["registers_per_thread"] = value.registers_per_thread;
        result["local_bytes_per_thread"] = value.local_bytes_per_thread;
        result["shared_bytes_per_block"] = value.shared_bytes_per_block;
        return result;
    };
    pybind11::dict result;
    result["validity_reset"] = one(receipt.validity_reset);
    result["reduction"] = one(reduction);
    result["checkpoint_hash"] = one(checkpoint);
    result["prerequisite_gate"] = one(receipt.prerequisite_gate);
    result["intermediate_synchronizations"] = 0;
    result["device_allocations"] = 0;
    result["host_allocations"] = 0;
    result["runtime_capability_promoted"] = false;
    return result;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("run_into", &run_into);
    module.def("resource_receipt", &resource_receipt);
    module.attr("contract_version") = boundary::kContractVersionV1;
    module.attr("required_host_digest_mask") =
        boundary::kRequiredHostCanonicalDigestMaskV1;
    module.attr("status_success") = static_cast<std::int32_t>(
        boundary::kBoundaryLaunchSuccessV1
    );
    module.attr("status_host_digest_finalization_required") =
        static_cast<std::int32_t>(
            boundary::kBoundaryHostDigestFinalizationRequiredV1
        );
    module.attr("producer_reduction_failure") =
        static_cast<std::int32_t>(
            boundary::kBoundaryProducerReductionFailureV1
        );
    module.attr("producer_checkpoint_hash_failure") =
        static_cast<std::int32_t>(
            boundary::kBoundaryProducerCheckpointHashFailureV1
        );
    module.attr("producer_host_digest_finalization_required") =
        static_cast<std::int32_t>(
            boundary::kBoundaryProducerHostDigestFinalizationRequiredV1
        );
    module.attr("producer_terminal_sticky_tail") =
        static_cast<std::int32_t>(
            boundary::finalizer::kProducerStatusTerminalStickyTail
        );
}
