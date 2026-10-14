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
constexpr int32_t DEATH_COMMITTED_EVENT_TYPE = 4;
constexpr int64_t SENTINEL_ID = -1;

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

__global__ void preflight_death_lifecycle_events_kernel(
    const int64_t* __restrict__ death_agent_ids,
    const int64_t* __restrict__ death_slots,
    const int32_t* __restrict__ reason_codes,
    const int64_t* __restrict__ death_count,
    const int64_t* __restrict__ generation,
    const int64_t* __restrict__ tick,
    const int32_t* __restrict__ lineage_founder_id,
    const int32_t* __restrict__ lineage_depth,
    const int64_t* __restrict__ active_count,
    int32_t* __restrict__ status,
    const int64_t death_capacity,
    const int64_t slot_count,
    const int64_t tape_capacity
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }

    // Recognized prior tape failures are sticky across later stages and ticks.
    // Other values are treated as an uninitialized standalone-call sentinel.
    if (status[0] == STATUS_CAPACITY_EXCEEDED
        || status[0] == STATUS_INVALID_SEMANTICS) {
        return;
    }
    status[0] = STATUS_OK;
    const int64_t count = death_count[0];
    if (count < 0 || count > death_capacity
        || generation[0] < 0 || tick[0] < 0) {
        status[0] = STATUS_INVALID_SEMANTICS;
        return;
    }

    for (int64_t death = 0; death < count; ++death) {
        const int64_t agent_id = death_agent_ids[death];
        const int64_t slot = death_slots[death];
        const int32_t reason = reason_codes[death];
        if (agent_id < 0
            || slot < 0 || slot >= slot_count
            || (reason != 1 && reason != 2)
            || lineage_founder_id[slot] < 0
            || lineage_depth[slot] < 0) {
            status[0] = STATUS_INVALID_SEMANTICS;
            return;
        }
        for (int64_t previous = 0; previous < death; ++previous) {
            if (death_agent_ids[previous] == agent_id
                || death_slots[previous] == slot) {
                status[0] = STATUS_INVALID_SEMANTICS;
                return;
            }
        }
    }

    const int64_t cursor = active_count[0];
    if (cursor < 0 || cursor > tape_capacity
        || count > tape_capacity - cursor) {
        status[0] = STATUS_CAPACITY_EXCEEDED;
    }
}

__device__ __forceinline__ void publish_death_event(
    const EventTapeColumns columns,
    const int64_t row,
    const int64_t generation,
    const int64_t tick,
    const int64_t agent_id,
    const int64_t slot,
    const int64_t founder_id,
    const int32_t lineage_depth,
    const int32_t reason
) {
    const float value = static_cast<float>(reason);
    columns.event_type[row] = DEATH_COMMITTED_EVENT_TYPE;
    columns.generation[row] = generation;
    columns.tick[row] = tick;
    columns.actor_id[row] = agent_id;
    columns.target_id[row] = SENTINEL_ID;
    columns.value[row] = value;
    columns.typed_event_type[row] = DEATH_COMMITTED_EVENT_TYPE;
    columns.typed_generation[row] = generation;
    columns.typed_tick[row] = tick;
    columns.typed_sequence[row] = row;
    columns.typed_actor_id[row] = agent_id;
    columns.typed_target_id[row] = SENTINEL_ID;
    columns.typed_actor_slot[row] = slot;
    columns.typed_target_slot[row] = SENTINEL_ID;
    columns.typed_parent_id[row] = SENTINEL_ID;
    columns.typed_child_id[row] = SENTINEL_ID;
    columns.typed_founder_id[row] = founder_id;
    columns.typed_lineage_depth[row] = lineage_depth;
    columns.typed_subtype[row] = DEATH_COMMITTED_EVENT_TYPE;
    columns.typed_reason[row] = reason;
    columns.typed_success[row] = 1;
    columns.typed_value[row] = value;
}

__global__ void append_death_lifecycle_events_kernel(
    const EventTapeColumns columns,
    int64_t* __restrict__ active_count,
    const int32_t* __restrict__ status,
    const int64_t* __restrict__ death_agent_ids,
    const int64_t* __restrict__ death_slots,
    const int32_t* __restrict__ reason_codes,
    const int64_t* __restrict__ death_count,
    const int64_t* __restrict__ generation,
    const int64_t* __restrict__ tick,
    const int32_t* __restrict__ lineage_founder_id,
    const int32_t* __restrict__ lineage_depth
) {
    if (blockIdx.x != 0 || threadIdx.x != 0 || status[0] != STATUS_OK) {
        return;
    }

    const int64_t cursor = active_count[0];
    const int64_t count = death_count[0];
    for (int64_t death = 0; death < count; ++death) {
        const int64_t slot = death_slots[death];
        publish_death_event(
            columns,
            cursor + death,
            generation[0],
            tick[0],
            death_agent_ids[death],
            slot,
            static_cast<int64_t>(lineage_founder_id[slot]),
            lineage_depth[slot],
            reason_codes[death]
        );
    }
    active_count[0] = cursor + count;
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

std::vector<torch::Tensor> eiel_cuda_architecture_b_death_lifecycle_events_append_v1(
    const std::vector<torch::Tensor>& event_tape_columns,
    torch::Tensor event_tape_active_count,
    torch::Tensor event_tape_status,
    torch::Tensor death_agent_ids,
    torch::Tensor death_slots,
    torch::Tensor reason_codes,
    torch::Tensor death_count,
    torch::Tensor generation,
    torch::Tensor tick,
    torch::Tensor lineage_founder_id,
    torch::Tensor lineage_depth
) {
    TORCH_CHECK(death_agent_ids.is_cuda(), "death_agent_ids must be CUDA resident");
    const auto device = death_agent_ids.device();
    c10::cuda::CUDAGuard device_guard(device);

    check_cuda_vector(death_agent_ids, "death_agent_ids", torch::kInt64, device);
    const int64_t death_capacity = death_agent_ids.numel();
    check_cuda_vector(death_slots, "death_slots", torch::kInt64, device);
    check_cuda_vector(
        reason_codes,
        "reason_codes",
        torch::kInt32,
        device
    );
    TORCH_CHECK(
        death_slots.numel() == death_capacity
            && reason_codes.numel() == death_capacity,
        "death lifecycle inputs must share one fixed capacity"
    );
    check_scalar(death_count, "death_count", torch::kInt64, device);
    check_scalar(generation, "generation", torch::kInt64, device);
    check_scalar(tick, "tick", torch::kInt64, device);

    check_cuda_vector(
        lineage_founder_id,
        "lineage_founder_id",
        torch::kInt32,
        device
    );
    const int64_t slot_count = lineage_founder_id.numel();
    TORCH_CHECK(slot_count > 0, "lineage state must have positive fixed capacity");
    check_cuda_vector(lineage_depth, "lineage_depth", torch::kInt32, device);
    TORCH_CHECK(
        lineage_depth.numel() == slot_count,
        "lineage state must contain one row per resident slot"
    );

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
        "death lifecycle tape requires 6 legacy and 16 typed columns"
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
    preflight_death_lifecycle_events_kernel<<<1, 1, 0, stream>>>(
        death_agent_ids.data_ptr<int64_t>(),
        death_slots.data_ptr<int64_t>(),
        reason_codes.data_ptr<int32_t>(),
        death_count.data_ptr<int64_t>(),
        generation.data_ptr<int64_t>(),
        tick.data_ptr<int64_t>(),
        lineage_founder_id.data_ptr<int32_t>(),
        lineage_depth.data_ptr<int32_t>(),
        event_tape_active_count.data_ptr<int64_t>(),
        event_tape_status.data_ptr<int32_t>(),
        death_capacity,
        slot_count,
        tape_capacity
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    append_death_lifecycle_events_kernel<<<1, 1, 0, stream>>>(
        pointers,
        event_tape_active_count.data_ptr<int64_t>(),
        event_tape_status.data_ptr<int32_t>(),
        death_agent_ids.data_ptr<int64_t>(),
        death_slots.data_ptr<int64_t>(),
        reason_codes.data_ptr<int32_t>(),
        death_count.data_ptr<int64_t>(),
        generation.data_ptr<int64_t>(),
        tick.data_ptr<int64_t>(),
        lineage_founder_id.data_ptr<int32_t>(),
        lineage_depth.data_ptr<int32_t>()
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {event_tape_active_count, event_tape_status};
}

#ifndef EIEL_CUDA_ARCHITECTURE_B_NO_PYBIND
PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.doc() = "Exact fixed-capacity CUDA Architecture B death lifecycle publication";
    module.def(
        "eiel_cuda_architecture_b_death_lifecycle_events_append_v1",
        &eiel_cuda_architecture_b_death_lifecycle_events_append_v1
    );
}
#endif
