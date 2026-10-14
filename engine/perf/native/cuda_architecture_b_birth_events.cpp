#include <torch/extension.h>

#include <c10/cuda/CUDAGuard.h>

#include <array>
#include <cstdint>
#include <vector>

std::vector<torch::Tensor> eiel_cuda_architecture_b_event_tape_append_v1(
    const std::vector<torch::Tensor>& tape_columns,
    const std::vector<torch::Tensor>& event_rows,
    torch::Tensor active_count,
    torch::Tensor status
);

namespace {

constexpr std::array<c10::ScalarType, 22> EVENT_TAPE_DTYPES = {
    torch::kInt32,
    torch::kInt64,
    torch::kInt64,
    torch::kInt64,
    torch::kInt64,
    torch::kFloat32,
    torch::kInt32,
    torch::kInt64,
    torch::kInt64,
    torch::kInt64,
    torch::kInt64,
    torch::kInt64,
    torch::kInt64,
    torch::kInt64,
    torch::kInt64,
    torch::kInt64,
    torch::kInt64,
    torch::kInt32,
    torch::kInt32,
    torch::kInt32,
    torch::kInt8,
    torch::kFloat32,
};

void check_cuda_vector(
    const torch::Tensor& tensor,
    const char* label,
    const c10::ScalarType dtype,
    const c10::Device& device,
    const int64_t expected_rows
) {
    TORCH_CHECK(tensor.is_cuda(), label, " must be CUDA resident");
    TORCH_CHECK(tensor.device() == device, label, " must share the birth-event CUDA device");
    TORCH_CHECK(tensor.is_contiguous(), label, " must be contiguous");
    TORCH_CHECK(tensor.dim() == 1, label, " must be one-dimensional");
    TORCH_CHECK(tensor.scalar_type() == dtype, label, " has an invalid dtype");
    TORCH_CHECK(tensor.numel() == expected_rows, label, " has an invalid row count");
}

void check_scalar_or_birth_vector(
    const torch::Tensor& tensor,
    const char* label,
    const c10::Device& device,
    const int64_t birth_count
) {
    TORCH_CHECK(tensor.is_cuda(), label, " must be CUDA resident");
    TORCH_CHECK(tensor.device() == device, label, " must share the birth-event CUDA device");
    TORCH_CHECK(tensor.is_contiguous(), label, " must be contiguous");
    TORCH_CHECK(tensor.dim() == 1, label, " must be one-dimensional");
    TORCH_CHECK(tensor.scalar_type() == torch::kInt64, label, " must be int64");
    TORCH_CHECK(
        tensor.numel() == 1 || tensor.numel() == birth_count,
        label,
        " must contain one resident value or one value per birth"
    );
}

torch::Tensor expand_scalar_or_birth_vector(
    const torch::Tensor& tensor,
    const int64_t birth_count
) {
    return tensor.numel() == 1
        ? tensor.expand({birth_count}).contiguous()
        : tensor;
}

}  // namespace

std::vector<torch::Tensor> eiel_cuda_architecture_b_birth_events_append_v1(
    const std::vector<torch::Tensor>& event_tape_columns,
    torch::Tensor event_tape_active_count,
    torch::Tensor event_tape_status,
    torch::Tensor parent_agent_ids,
    torch::Tensor child_agent_ids,
    torch::Tensor parent_slots,
    torch::Tensor child_slots,
    torch::Tensor lineage_founder_ids,
    torch::Tensor lineage_depths,
    torch::Tensor generation,
    torch::Tensor tick,
    torch::Tensor mutation_occurred,
    torch::Tensor mutation_delta_total,
    torch::Tensor child_energy
) {
    TORCH_CHECK(parent_agent_ids.is_cuda(), "parent_agent_ids must be CUDA resident");
    const auto device = parent_agent_ids.device();
    const int64_t birth_count = parent_agent_ids.numel();
    c10::cuda::CUDAGuard device_guard(device);

    check_cuda_vector(
        parent_agent_ids,
        "parent_agent_ids",
        torch::kInt64,
        device,
        birth_count
    );
    check_cuda_vector(child_agent_ids, "child_agent_ids", torch::kInt64, device, birth_count);
    check_cuda_vector(parent_slots, "parent_slots", torch::kInt64, device, birth_count);
    check_cuda_vector(child_slots, "child_slots", torch::kInt64, device, birth_count);
    check_cuda_vector(
        lineage_founder_ids,
        "lineage_founder_ids",
        torch::kInt64,
        device,
        birth_count
    );
    check_cuda_vector(lineage_depths, "lineage_depths", torch::kInt32, device, birth_count);
    check_scalar_or_birth_vector(generation, "generation", device, birth_count);
    check_scalar_or_birth_vector(tick, "tick", device, birth_count);
    check_cuda_vector(
        mutation_occurred,
        "mutation_occurred",
        torch::kBool,
        device,
        birth_count
    );
    check_cuda_vector(
        mutation_delta_total,
        "mutation_delta_total",
        torch::kFloat32,
        device,
        birth_count
    );
    check_cuda_vector(child_energy, "child_energy", torch::kFloat32, device, birth_count);
    check_cuda_vector(
        event_tape_active_count,
        "event_tape_active_count",
        torch::kInt64,
        device,
        1
    );
    check_cuda_vector(
        event_tape_status,
        "event_tape_status",
        torch::kInt32,
        device,
        1
    );

    TORCH_CHECK(
        event_tape_columns.size() == EVENT_TAPE_DTYPES.size(),
        "birth-event tape requires 6 legacy and 16 typed columns"
    );
    const int64_t tape_capacity = event_tape_columns.front().numel();
    for (std::size_t index = 0; index < event_tape_columns.size(); ++index) {
        check_cuda_vector(
            event_tape_columns[index],
            "event_tape_column",
            EVENT_TAPE_DTYPES[index],
            device,
            tape_capacity
        );
    }

    const auto int64_options = parent_agent_ids.options().dtype(torch::kInt64);
    const auto int32_options = parent_agent_ids.options().dtype(torch::kInt32);
    const auto int8_options = parent_agent_ids.options().dtype(torch::kInt8);
    const auto bool_options = parent_agent_ids.options().dtype(torch::kBool);

    // The accepted owner emits [mutation, birth] per child and omits the
    // mutation row when no policy element changed. Build that order before one
    // all-column append so capacity failure cannot publish a partial child.
    const auto birth_indices = torch::arange(birth_count, int64_options);
    const auto duplicated_birth_indices = torch::stack(
        {birth_indices, birth_indices},
        1
    ).reshape({-1});
    const auto keep_rows = torch::stack(
        {mutation_occurred, torch::ones({birth_count}, bool_options)},
        1
    ).reshape({-1});
    const auto event_birth_indices = duplicated_birth_indices.masked_select(keep_rows);
    const auto all_row_is_birth = torch::stack(
        {
            torch::zeros({birth_count}, bool_options),
            torch::ones({birth_count}, bool_options),
        },
        1
    ).reshape({-1});
    const auto is_birth = all_row_is_birth.masked_select(keep_rows);
    const int64_t event_count = event_birth_indices.numel();

    const auto parent_by_event = parent_agent_ids.index_select(0, event_birth_indices);
    const auto child_by_event = child_agent_ids.index_select(0, event_birth_indices);
    const auto parent_slot_by_event = parent_slots.index_select(0, event_birth_indices);
    const auto child_slot_by_event = child_slots.index_select(0, event_birth_indices);
    const auto founder_by_event = lineage_founder_ids.index_select(0, event_birth_indices);
    const auto depth_by_event = lineage_depths.index_select(0, event_birth_indices);
    const auto generation_by_birth = expand_scalar_or_birth_vector(generation, birth_count);
    const auto tick_by_birth = expand_scalar_or_birth_vector(tick, birth_count);
    const auto generation_by_event = generation_by_birth.index_select(0, event_birth_indices);
    const auto tick_by_event = tick_by_birth.index_select(0, event_birth_indices);
    const auto mutation_value_by_event = mutation_delta_total.index_select(
        0,
        event_birth_indices
    );
    const auto birth_value_by_event = child_energy.index_select(0, event_birth_indices);

    const auto event_type = 2 - is_birth.to(torch::kInt32);
    const auto actor_agent_id = torch::where(is_birth, parent_by_event, child_by_event);
    const auto actor_slot = torch::where(is_birth, parent_slot_by_event, child_slot_by_event);
    const auto event_value = torch::where(
        is_birth,
        birth_value_by_event,
        mutation_value_by_event
    );
    const auto event_sequence = event_tape_active_count
        + torch::arange(event_count, int64_options);
    const auto reason_code = torch::full({event_count}, -1, int32_options);
    const auto success = torch::ones({event_count}, int8_options);

    std::vector<torch::Tensor> event_rows = {
        event_type,
        generation_by_event,
        tick_by_event,
        actor_agent_id,
        child_by_event,
        event_value,
        event_type,
        generation_by_event,
        tick_by_event,
        event_sequence,
        actor_agent_id,
        child_by_event,
        actor_slot,
        child_slot_by_event,
        parent_by_event,
        child_by_event,
        founder_by_event,
        depth_by_event,
        event_type,
        reason_code,
        success,
        event_value,
    };
    return eiel_cuda_architecture_b_event_tape_append_v1(
        event_tape_columns,
        event_rows,
        event_tape_active_count,
        event_tape_status
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.doc() = "Exact standalone CUDA Architecture B birth-event construction";
    module.def(
        "eiel_cuda_architecture_b_birth_events_append_v1",
        &eiel_cuda_architecture_b_birth_events_append_v1
    );
}
