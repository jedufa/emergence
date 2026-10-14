#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <array>
#include <cstdint>
#include <vector>

namespace {

constexpr int32_t STATUS_OK = 0;
constexpr int32_t STATUS_CAPACITY_EXCEEDED = 1;
constexpr int32_t STATUS_INVALID_SEMANTICS = 2;

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

struct EventTapeColumns {
    int32_t* event_type;
    int64_t* generation;
    int64_t* tick;
    int64_t* actor_id;
    int64_t* target_id;
    float* value;
    int32_t* typed_event_type;
    int64_t* typed_generation;
    int64_t* typed_tick;
    int64_t* typed_sequence;
    int64_t* typed_actor_id;
    int64_t* typed_target_id;
    int64_t* typed_actor_slot;
    int64_t* typed_target_slot;
    int64_t* typed_parent_id;
    int64_t* typed_child_id;
    int64_t* typed_founder_id;
    int32_t* typed_lineage_depth;
    int32_t* typed_subtype;
    int32_t* typed_reason;
    int8_t* typed_success;
    float* typed_value;
};

void check_cuda_vector(
    const torch::Tensor& tensor,
    const char* label,
    const c10::ScalarType dtype,
    const c10::Device& device
) {
    TORCH_CHECK(tensor.is_cuda(), label, " must be CUDA resident");
    TORCH_CHECK(tensor.device() == device, label, " must share one CUDA device");
    TORCH_CHECK(tensor.is_contiguous(), label, " must be contiguous");
    TORCH_CHECK(tensor.dim() == 1, label, " must be one-dimensional");
    TORCH_CHECK(tensor.scalar_type() == dtype, label, " has an invalid dtype");
}

void check_scalar(
    const torch::Tensor& tensor,
    const char* label,
    const c10::ScalarType dtype,
    const c10::Device& device
) {
    check_cuda_vector(tensor, label, dtype, device);
    TORCH_CHECK(tensor.numel() == 1, label, " must contain exactly one value");
}

__device__ __forceinline__ void publish_event(
    const EventTapeColumns columns,
    const int64_t row,
    const int32_t event_type,
    const int64_t generation,
    const int64_t tick,
    const int64_t actor_id,
    const int64_t target_id,
    const int64_t actor_slot,
    const int64_t target_slot,
    const int64_t parent_id,
    const int64_t child_id,
    const int64_t founder_id,
    const int32_t lineage_depth,
    const int32_t subtype,
    const int32_t reason,
    const int8_t success,
    const float value
) {
    columns.event_type[row] = event_type;
    columns.generation[row] = generation;
    columns.tick[row] = tick;
    columns.actor_id[row] = actor_id;
    columns.target_id[row] = target_id;
    columns.value[row] = value;
    columns.typed_event_type[row] = event_type;
    columns.typed_generation[row] = generation;
    columns.typed_tick[row] = tick;
    columns.typed_sequence[row] = row;
    columns.typed_actor_id[row] = actor_id;
    columns.typed_target_id[row] = target_id;
    columns.typed_actor_slot[row] = actor_slot;
    columns.typed_target_slot[row] = target_slot;
    columns.typed_parent_id[row] = parent_id;
    columns.typed_child_id[row] = child_id;
    columns.typed_founder_id[row] = founder_id;
    columns.typed_lineage_depth[row] = lineage_depth;
    columns.typed_subtype[row] = subtype;
    columns.typed_reason[row] = reason;
    columns.typed_success[row] = success;
    columns.typed_value[row] = value;
}

__global__ void preflight_birth_lifecycle_events_kernel(
    const int64_t* __restrict__ slot_to_agent_id,
    const int32_t* __restrict__ lineage_founder_id,
    const int32_t* __restrict__ lineage_depth,
    const int64_t* __restrict__ decision_parent_slots,
    const int32_t* __restrict__ decision_reason_codes,
    const int64_t* __restrict__ decision_count,
    const int64_t* __restrict__ parent_agent_ids,
    const int64_t* __restrict__ child_agent_ids,
    const int64_t* __restrict__ parent_slots,
    const int64_t* __restrict__ child_slots,
    const bool* __restrict__ mutation_occurred,
    const int64_t* __restrict__ successful_birth_count,
    const int64_t* __restrict__ active_count,
    int32_t* __restrict__ status,
    const int64_t slot_count,
    const int64_t decision_capacity,
    const int64_t success_capacity,
    const int64_t tape_capacity
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }

    status[0] = STATUS_OK;
    const int64_t decisions = decision_count[0];
    const int64_t successes = successful_birth_count[0];
    if (decisions < 0 || decisions > decision_capacity
        || successes < 0 || successes > success_capacity) {
        status[0] = STATUS_INVALID_SEMANTICS;
        return;
    }

    int64_t success_ordinal = 0;
    int64_t events = 0;
    for (int64_t decision = 0; decision < decisions; ++decision) {
        const int64_t parent_slot = decision_parent_slots[decision];
        const int32_t reason = decision_reason_codes[decision];
        if (parent_slot < 0 || parent_slot >= slot_count
            || slot_to_agent_id[parent_slot] < 0
            || lineage_depth[parent_slot] < 0) {
            status[0] = STATUS_INVALID_SEMANTICS;
            return;
        }

        if (reason == 0) {
            if (success_ordinal >= successes) {
                status[0] = STATUS_INVALID_SEMANTICS;
                return;
            }
            const int64_t committed_parent_slot = parent_slots[success_ordinal];
            const int64_t committed_child_slot = child_slots[success_ordinal];
            if (committed_parent_slot != parent_slot
                || committed_parent_slot < 0 || committed_parent_slot >= slot_count
                || committed_child_slot < 0 || committed_child_slot >= slot_count
                || parent_agent_ids[success_ordinal] < 0
                || child_agent_ids[success_ordinal] < 0
                || slot_to_agent_id[committed_parent_slot]
                    != parent_agent_ids[success_ordinal]
                || slot_to_agent_id[committed_child_slot]
                    != child_agent_ids[success_ordinal]
                || lineage_founder_id[committed_child_slot] < 0
                || lineage_depth[committed_child_slot] < 0) {
                status[0] = STATUS_INVALID_SEMANTICS;
                return;
            }
            events += mutation_occurred[success_ordinal] ? 2 : 1;
            ++success_ordinal;
            continue;
        }
        if (reason >= 1 && reason <= 3) {
            ++events;
            continue;
        }
        if (reason >= 7 && reason <= 9) {
            continue;
        }
        status[0] = STATUS_INVALID_SEMANTICS;
        return;
    }
    if (success_ordinal != successes) {
        status[0] = STATUS_INVALID_SEMANTICS;
        return;
    }

    const int64_t cursor = active_count[0];
    if (cursor < 0 || cursor > tape_capacity
        || events > tape_capacity - cursor) {
        status[0] = STATUS_CAPACITY_EXCEEDED;
    }
}

__global__ void append_birth_lifecycle_events_kernel(
    const EventTapeColumns columns,
    int64_t* __restrict__ active_count,
    const int32_t* __restrict__ status,
    const int64_t* __restrict__ slot_to_agent_id,
    const int32_t* __restrict__ lineage_founder_id,
    const int32_t* __restrict__ lineage_depth,
    const int64_t* __restrict__ decision_parent_slots,
    const int32_t* __restrict__ decision_reason_codes,
    const int64_t* __restrict__ decision_count,
    const int64_t* __restrict__ parent_agent_ids,
    const int64_t* __restrict__ child_agent_ids,
    const int64_t* __restrict__ parent_slots,
    const int64_t* __restrict__ child_slots,
    const bool* __restrict__ mutation_occurred,
    const float* __restrict__ mutation_delta_total,
    const float* __restrict__ child_energy,
    const int64_t* __restrict__ generation,
    const int64_t* __restrict__ tick
) {
    if (blockIdx.x != 0 || threadIdx.x != 0 || status[0] != STATUS_OK) {
        return;
    }

    int64_t output_row = active_count[0];
    int64_t success_ordinal = 0;
    for (int64_t decision = 0; decision < decision_count[0]; ++decision) {
        const int64_t decision_parent_slot = decision_parent_slots[decision];
        const int64_t decision_parent_id = slot_to_agent_id[decision_parent_slot];
        const int32_t reason = decision_reason_codes[decision];
        if (reason >= 1 && reason <= 3) {
            const int32_t raw_founder = lineage_founder_id[decision_parent_slot];
            publish_event(
                columns,
                output_row++,
                3,
                generation[0],
                tick[0],
                decision_parent_id,
                -1,
                decision_parent_slot,
                -1,
                -1,
                -1,
                raw_founder >= 0
                    ? static_cast<int64_t>(raw_founder)
                    : decision_parent_id,
                lineage_depth[decision_parent_slot],
                3,
                reason,
                0,
                static_cast<float>(reason)
            );
            continue;
        }
        if (reason != 0) {
            continue;
        }

        const int64_t parent_id = parent_agent_ids[success_ordinal];
        const int64_t child_id = child_agent_ids[success_ordinal];
        const int64_t parent_slot = parent_slots[success_ordinal];
        const int64_t child_slot = child_slots[success_ordinal];
        const int64_t founder = static_cast<int64_t>(lineage_founder_id[child_slot]);
        const int32_t depth = lineage_depth[child_slot];
        if (mutation_occurred[success_ordinal]) {
            publish_event(
                columns,
                output_row++,
                2,
                generation[0],
                tick[0],
                child_id,
                child_id,
                child_slot,
                child_slot,
                parent_id,
                child_id,
                founder,
                depth,
                2,
                -1,
                1,
                mutation_delta_total[success_ordinal]
            );
        }
        publish_event(
            columns,
            output_row++,
            1,
            generation[0],
            tick[0],
            parent_id,
            child_id,
            parent_slot,
            child_slot,
            parent_id,
            child_id,
            founder,
            depth,
            1,
            -1,
            1,
            child_energy[success_ordinal]
        );
        ++success_ordinal;
    }
    active_count[0] = output_row;
}

EventTapeColumns event_tape_pointers(const std::vector<torch::Tensor>& columns) {
    return {
        columns[0].data_ptr<int32_t>(),
        columns[1].data_ptr<int64_t>(),
        columns[2].data_ptr<int64_t>(),
        columns[3].data_ptr<int64_t>(),
        columns[4].data_ptr<int64_t>(),
        columns[5].data_ptr<float>(),
        columns[6].data_ptr<int32_t>(),
        columns[7].data_ptr<int64_t>(),
        columns[8].data_ptr<int64_t>(),
        columns[9].data_ptr<int64_t>(),
        columns[10].data_ptr<int64_t>(),
        columns[11].data_ptr<int64_t>(),
        columns[12].data_ptr<int64_t>(),
        columns[13].data_ptr<int64_t>(),
        columns[14].data_ptr<int64_t>(),
        columns[15].data_ptr<int64_t>(),
        columns[16].data_ptr<int64_t>(),
        columns[17].data_ptr<int32_t>(),
        columns[18].data_ptr<int32_t>(),
        columns[19].data_ptr<int32_t>(),
        columns[20].data_ptr<int8_t>(),
        columns[21].data_ptr<float>(),
    };
}

}  // namespace

std::vector<torch::Tensor> eiel_cuda_architecture_b_birth_lifecycle_events_append_v1(
    const std::vector<torch::Tensor>& event_tape_columns,
    torch::Tensor event_tape_active_count,
    torch::Tensor event_tape_status,
    torch::Tensor slot_to_agent_id,
    torch::Tensor lineage_founder_id,
    torch::Tensor lineage_depth,
    torch::Tensor decision_parent_slots,
    torch::Tensor decision_reason_codes,
    torch::Tensor decision_count,
    torch::Tensor parent_agent_ids,
    torch::Tensor child_agent_ids,
    torch::Tensor parent_slots,
    torch::Tensor child_slots,
    torch::Tensor mutation_occurred,
    torch::Tensor mutation_delta_total,
    torch::Tensor child_energy,
    torch::Tensor successful_birth_count,
    torch::Tensor generation,
    torch::Tensor tick
) {
    TORCH_CHECK(slot_to_agent_id.is_cuda(), "slot_to_agent_id must be CUDA resident");
    const auto device = slot_to_agent_id.device();
    c10::cuda::CUDAGuard device_guard(device);

    check_cuda_vector(slot_to_agent_id, "slot_to_agent_id", torch::kInt64, device);
    const int64_t slot_count = slot_to_agent_id.numel();
    TORCH_CHECK(slot_count > 0, "slot_to_agent_id must have positive fixed capacity");
    check_cuda_vector(
        lineage_founder_id,
        "lineage_founder_id",
        torch::kInt32,
        device
    );
    check_cuda_vector(lineage_depth, "lineage_depth", torch::kInt32, device);
    TORCH_CHECK(
        lineage_founder_id.numel() == slot_count && lineage_depth.numel() == slot_count,
        "lineage state must contain one row per slot"
    );

    check_cuda_vector(
        decision_parent_slots,
        "decision_parent_slots",
        torch::kInt64,
        device
    );
    const int64_t decision_capacity = decision_parent_slots.numel();
    TORCH_CHECK(decision_capacity > 0, "planner decision capacity must be positive");
    check_cuda_vector(
        decision_reason_codes,
        "decision_reason_codes",
        torch::kInt32,
        device
    );
    TORCH_CHECK(
        decision_reason_codes.numel() == decision_capacity,
        "planner decision tensors must share one fixed capacity"
    );
    check_scalar(decision_count, "decision_count", torch::kInt64, device);

    check_cuda_vector(parent_agent_ids, "parent_agent_ids", torch::kInt64, device);
    const int64_t success_capacity = parent_agent_ids.numel();
    for (const auto& item : std::vector<std::pair<const char*, torch::Tensor>>{
             {"child_agent_ids", child_agent_ids},
             {"parent_slots", parent_slots},
             {"child_slots", child_slots},
         }) {
        check_cuda_vector(item.second, item.first, torch::kInt64, device);
        TORCH_CHECK(
            item.second.numel() == success_capacity,
            item.first,
            " must share the successful birth fixed capacity"
        );
    }
    check_cuda_vector(
        mutation_occurred,
        "mutation_occurred",
        torch::kBool,
        device
    );
    check_cuda_vector(
        mutation_delta_total,
        "mutation_delta_total",
        torch::kFloat32,
        device
    );
    check_cuda_vector(child_energy, "child_energy", torch::kFloat32, device);
    TORCH_CHECK(
        mutation_occurred.numel() == success_capacity
            && mutation_delta_total.numel() == success_capacity
            && child_energy.numel() == success_capacity,
        "successful birth value tensors must share one fixed capacity"
    );
    check_scalar(
        successful_birth_count,
        "successful_birth_count",
        torch::kInt64,
        device
    );
    check_scalar(generation, "generation", torch::kInt64, device);
    check_scalar(tick, "tick", torch::kInt64, device);
    check_scalar(
        event_tape_active_count,
        "event_tape_active_count",
        torch::kInt64,
        device
    );
    check_scalar(
        event_tape_status,
        "event_tape_status",
        torch::kInt32,
        device
    );

    TORCH_CHECK(
        event_tape_columns.size() == EVENT_TAPE_DTYPES.size(),
        "birth lifecycle tape requires 6 legacy and 16 typed columns"
    );
    const int64_t tape_capacity = event_tape_columns.front().numel();
    for (std::size_t index = 0; index < event_tape_columns.size(); ++index) {
        check_cuda_vector(
            event_tape_columns[index],
            "event_tape_column",
            EVENT_TAPE_DTYPES[index],
            device
        );
        TORCH_CHECK(
            event_tape_columns[index].numel() == tape_capacity,
            "event tape columns must share one capacity"
        );
    }

    const EventTapeColumns pointers = event_tape_pointers(event_tape_columns);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    preflight_birth_lifecycle_events_kernel<<<1, 1, 0, stream>>>(
        slot_to_agent_id.data_ptr<int64_t>(),
        lineage_founder_id.data_ptr<int32_t>(),
        lineage_depth.data_ptr<int32_t>(),
        decision_parent_slots.data_ptr<int64_t>(),
        decision_reason_codes.data_ptr<int32_t>(),
        decision_count.data_ptr<int64_t>(),
        parent_agent_ids.data_ptr<int64_t>(),
        child_agent_ids.data_ptr<int64_t>(),
        parent_slots.data_ptr<int64_t>(),
        child_slots.data_ptr<int64_t>(),
        mutation_occurred.data_ptr<bool>(),
        successful_birth_count.data_ptr<int64_t>(),
        event_tape_active_count.data_ptr<int64_t>(),
        event_tape_status.data_ptr<int32_t>(),
        slot_count,
        decision_capacity,
        success_capacity,
        tape_capacity
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    append_birth_lifecycle_events_kernel<<<1, 1, 0, stream>>>(
        pointers,
        event_tape_active_count.data_ptr<int64_t>(),
        event_tape_status.data_ptr<int32_t>(),
        slot_to_agent_id.data_ptr<int64_t>(),
        lineage_founder_id.data_ptr<int32_t>(),
        lineage_depth.data_ptr<int32_t>(),
        decision_parent_slots.data_ptr<int64_t>(),
        decision_reason_codes.data_ptr<int32_t>(),
        decision_count.data_ptr<int64_t>(),
        parent_agent_ids.data_ptr<int64_t>(),
        child_agent_ids.data_ptr<int64_t>(),
        parent_slots.data_ptr<int64_t>(),
        child_slots.data_ptr<int64_t>(),
        mutation_occurred.data_ptr<bool>(),
        mutation_delta_total.data_ptr<float>(),
        child_energy.data_ptr<float>(),
        generation.data_ptr<int64_t>(),
        tick.data_ptr<int64_t>()
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {event_tape_active_count, event_tape_status};
}

#ifndef EIEL_CUDA_ARCHITECTURE_B_NO_PYBIND
PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.doc() = "Exact resident CUDA Architecture B asexual lifecycle publication";
    module.def(
        "eiel_cuda_architecture_b_birth_lifecycle_events_append_v1",
        &eiel_cuda_architecture_b_birth_lifecycle_events_append_v1
    );
}
#endif
