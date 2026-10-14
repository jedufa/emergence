#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <vector>

namespace {

constexpr int32_t kOk = 0;
constexpr int32_t kInvalidPendingCount = 1;
constexpr int32_t kPendingCapacityExceeded = 2;
constexpr int32_t kInvalidFreeQueueCount = 3;
constexpr int32_t kDuplicateFreeSlot = 4;
constexpr int32_t kFreeSlotStateContradiction = 5;
constexpr int32_t kEventCapacityExceeded = 6;
constexpr int32_t kEventSemanticsInvalid = 7;
constexpr int32_t kDuplicateDeathIdentity = 8;
constexpr int32_t kDeathCommittedEventType = 4;
constexpr int64_t kSentinelId = -1;

struct State {
    bool* alive;
    int64_t* agent_id;
    int64_t* pos_y;
    int64_t* pos_x;
    float* energy;
    float* health;
    int64_t* occupancy_agent_id;
    int64_t* occupancy_slot;
    int32_t* patch_agent_id;
    int32_t* agent_id_to_slot;
    bool* death_candidate;
    int32_t* death_reason;
    double* policy_energy_observation;
    int32_t* rest_streak;
    int64_t* action_cache_token;
    int32_t* action_cache_cursor;
    int32_t* action_cache_next;
    int64_t* active_agent_ids;
    int32_t* active_agent_slots;
    int64_t* active_count;
    int64_t* pending_free_agent_ids;
    int64_t* pending_free_slots;
    int64_t* pending_free_count;
    int64_t* death_agent_ids;
    int64_t* death_slots;
    int32_t* death_reason_codes;
    int64_t* death_count;
    int64_t* survivor_agent_ids;
    int64_t* free_queue;
    int64_t* free_queue_count;
    int64_t* lineage_founder_id;
    int32_t* lineage_depth;
    int64_t* birth_parent_ids;
    int64_t* birth_child_ids;
    int64_t* birth_count;
    bool* mutation_occurred;
    float* mutation_delta_total;
    int32_t* event_type;
    int64_t* event_generation;
    int64_t* event_tick;
    int64_t* event_actor_id;
    int64_t* event_target_id;
    float* event_value;
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
    int64_t* event_active_count;
    int32_t* event_status;
    int64_t batch;
    int64_t capacity;
    int64_t grid_size;
    int64_t event_capacity;
};

struct Staging {
    int64_t* death_agent_ids;
    int64_t* death_slots;
    int32_t* death_reason_codes;
    int64_t* death_count;
    int64_t* survivor_agent_ids;
    int32_t* survivor_slots;
    int64_t* survivor_count;
    int32_t* status;
};

__device__ __forceinline__ bool is_candidate(
    const State& state, const int64_t offset, const int64_t slot) {
    return state.alive[offset + slot]
        && state.agent_id[offset + slot] >= 0
        && state.death_candidate[offset + slot];
}

__global__ void preflight_kernel(State state, Staging staging) {
    if (threadIdx.x != 0) {
        return;
    }
    const int64_t world = static_cast<int64_t>(blockIdx.x);
    if (world >= state.batch) {
        return;
    }
    const int64_t offset = world * state.capacity;
    for (int64_t row = 0; row < state.capacity; ++row) {
        staging.death_agent_ids[offset + row] = -1;
        staging.death_slots[offset + row] = -1;
        staging.death_reason_codes[offset + row] = 0;
        staging.survivor_agent_ids[offset + row] = -1;
        staging.survivor_slots[offset + row] = -1;
    }
    staging.death_count[world] = 0;
    staging.survivor_count[world] = 0;
    staging.status[world] = kOk;

    const int64_t pending = state.pending_free_count[world];
    if (pending < 0 || pending > state.capacity) {
        staging.status[world] = kInvalidPendingCount;
        return;
    }
    const int64_t free_count = state.free_queue_count[world];
    if (free_count < 0 || free_count > state.capacity) {
        staging.status[world] = kInvalidFreeQueueCount;
        return;
    }

    // Validate the complete existing/free-to-append set before any mutation.
    for (int64_t row = 0; row < free_count + pending; ++row) {
        const bool from_queue = row < free_count;
        const int64_t source_row = from_queue ? row : row - free_count;
        const int64_t slot = from_queue
            ? state.free_queue[offset + source_row]
            : state.pending_free_slots[offset + source_row];
        if (slot < 0 || slot >= state.capacity) {
            staging.status[world] = kFreeSlotStateContradiction;
            return;
        }
        if (state.alive[offset + slot] || state.agent_id[offset + slot] >= 0) {
            staging.status[world] = kFreeSlotStateContradiction;
            return;
        }
        for (int64_t previous = 0; previous < row; ++previous) {
            const bool previous_from_queue = previous < free_count;
            const int64_t previous_row = previous_from_queue
                ? previous : previous - free_count;
            const int64_t previous_slot = previous_from_queue
                ? state.free_queue[offset + previous_row]
                : state.pending_free_slots[offset + previous_row];
            if (slot == previous_slot) {
                staging.status[world] = kDuplicateFreeSlot;
                return;
            }
        }
    }

    int64_t candidate_count = 0;
    for (int64_t slot = 0; slot < state.capacity; ++slot) {
        candidate_count += is_candidate(state, offset, slot) ? 1 : 0;
    }
    if (pending + candidate_count > state.capacity
        || free_count + pending + candidate_count > state.capacity) {
        staging.status[world] = kPendingCapacityExceeded;
        return;
    }

    int64_t previous_agent_id = INT64_MIN;
    int64_t previous_slot = -1;
    for (int64_t death = 0; death < candidate_count; ++death) {
        int64_t selected_agent_id = INT64_MAX;
        int64_t selected_slot = -1;
        for (int64_t slot = 0; slot < state.capacity; ++slot) {
            if (!is_candidate(state, offset, slot)) {
                continue;
            }
            const int64_t agent_id = state.agent_id[offset + slot];
            const bool follows = agent_id > previous_agent_id
                || (agent_id == previous_agent_id && slot > previous_slot);
            if (follows && (agent_id < selected_agent_id
                    || (agent_id == selected_agent_id && slot < selected_slot))) {
                selected_agent_id = agent_id;
                selected_slot = slot;
            }
        }
        if (selected_slot < 0) {
            staging.status[world] = kEventSemanticsInvalid;
            return;
        }
        if (death > 0
            && staging.death_agent_ids[offset + death - 1] == selected_agent_id) {
            staging.status[world] = kDuplicateDeathIdentity;
            return;
        }
        for (int64_t row = 0; row < free_count + pending; ++row) {
            const int64_t slot = row < free_count
                ? state.free_queue[offset + row]
                : state.pending_free_slots[offset + row - free_count];
            if (slot == selected_slot) {
                staging.status[world] = kDuplicateFreeSlot;
                return;
            }
        }
        staging.death_agent_ids[offset + death] = selected_agent_id;
        staging.death_slots[offset + death] = selected_slot;
        staging.death_reason_codes[offset + death] =
            state.health[offset + selected_slot] <= 0.0f ? 2 : 1;
        if (state.lineage_founder_id[offset + selected_slot] < 0
            || state.lineage_depth[offset + selected_slot] < 0) {
            staging.status[world] = kEventSemanticsInvalid;
            return;
        }
        previous_agent_id = selected_agent_id;
        previous_slot = selected_slot;
    }
    staging.death_count[world] = candidate_count;

    int64_t survivor_count = 0;
    previous_agent_id = INT64_MIN;
    previous_slot = -1;
    while (true) {
        int64_t selected_agent_id = INT64_MAX;
        int64_t selected_slot = -1;
        for (int64_t slot = 0; slot < state.capacity; ++slot) {
            if (!state.alive[offset + slot] || state.agent_id[offset + slot] < 0
                || state.death_candidate[offset + slot]) {
                continue;
            }
            const int64_t agent_id = state.agent_id[offset + slot];
            const bool follows = agent_id > previous_agent_id
                || (agent_id == previous_agent_id && slot > previous_slot);
            if (follows && (agent_id < selected_agent_id
                    || (agent_id == selected_agent_id && slot < selected_slot))) {
                selected_agent_id = agent_id;
                selected_slot = slot;
            }
        }
        if (selected_slot < 0) {
            break;
        }
        staging.survivor_agent_ids[offset + survivor_count] = selected_agent_id;
        staging.survivor_slots[offset + survivor_count] =
            static_cast<int32_t>(selected_slot);
        ++survivor_count;
        previous_agent_id = selected_agent_id;
        previous_slot = selected_slot;
    }
    staging.survivor_count[world] = survivor_count;

    const int32_t tape_status = state.event_status[world];
    if (tape_status == 1) {
        staging.status[world] = kEventCapacityExceeded;
        return;
    }
    if (tape_status == 2) {
        staging.status[world] = kEventSemanticsInvalid;
        return;
    }
    const int64_t cursor = state.event_active_count[world];
    if (cursor < 0 || cursor > state.event_capacity
        || candidate_count > state.event_capacity - cursor) {
        staging.status[world] = kEventCapacityExceeded;
        return;
    }
}

__device__ __forceinline__ void publish_death_event(
    const State& state,
    const Staging& staging,
    const int64_t world,
    const int64_t death,
    const int64_t generation,
    const int64_t tick) {
    const int64_t slot_offset = world * state.capacity;
    const int64_t event_offset = world * state.event_capacity;
    const int64_t cursor = state.event_active_count[world];
    const int64_t row = cursor + death;
    const int64_t event_index = event_offset + row;
    const int64_t agent_id = staging.death_agent_ids[slot_offset + death];
    const int64_t slot = staging.death_slots[slot_offset + death];
    const int32_t reason = staging.death_reason_codes[slot_offset + death];
    const float value = static_cast<float>(reason);
    state.event_type[event_index] = kDeathCommittedEventType;
    state.event_generation[event_index] = generation;
    state.event_tick[event_index] = tick;
    state.event_actor_id[event_index] = agent_id;
    state.event_target_id[event_index] = kSentinelId;
    state.event_value[event_index] = value;
    state.typed_event_type[event_index] = kDeathCommittedEventType;
    state.typed_generation[event_index] = generation;
    state.typed_tick[event_index] = tick;
    state.typed_sequence[event_index] = row;
    state.typed_actor_id[event_index] = agent_id;
    state.typed_target_id[event_index] = kSentinelId;
    state.typed_actor_slot[event_index] = slot;
    state.typed_target_slot[event_index] = kSentinelId;
    state.typed_parent_id[event_index] = kSentinelId;
    state.typed_child_id[event_index] = kSentinelId;
    state.typed_founder_id[event_index] = state.lineage_founder_id[slot_offset + slot];
    state.typed_lineage_depth[event_index] = state.lineage_depth[slot_offset + slot];
    state.typed_subtype[event_index] = kDeathCommittedEventType;
    state.typed_reason[event_index] = reason;
    state.typed_success[event_index] = 1;
    state.typed_value[event_index] = value;
}

__global__ void commit_kernel(
    State state, Staging staging, const int64_t generation, const int64_t tick) {
    if (threadIdx.x != 0) {
        return;
    }
    const int64_t world = static_cast<int64_t>(blockIdx.x);
    if (world >= state.batch) {
        return;
    }
    // The preceding kernel is a stream-wide boundary.  Every world checks the
    // complete status vector, so one failed world suppresses every mutation.
    for (int64_t other = 0; other < state.batch; ++other) {
        if (staging.status[other] != kOk) {
            return;
        }
    }

    const int64_t offset = world * state.capacity;
    const int64_t grid_offset = world * state.grid_size * state.grid_size;
    const int64_t pending = state.pending_free_count[world];
    const int64_t free_before = state.free_queue_count[world];
    const int64_t deaths = staging.death_count[world];
    const int64_t survivors = staging.survivor_count[world];
    for (int64_t row = 0; row < state.capacity; ++row) {
        state.death_agent_ids[offset + row] = -1;
        state.death_slots[offset + row] = -1;
        state.death_reason_codes[offset + row] = 0;
        state.survivor_agent_ids[offset + row] = -1;
        state.active_agent_ids[offset + row] = -1;
        state.active_agent_slots[offset + row] = -1;
        state.birth_parent_ids[offset + row] = -1;
        state.birth_child_ids[offset + row] = -1;
        state.mutation_occurred[offset + row] = false;
        state.mutation_delta_total[offset + row] = 0.0f;
    }
    state.birth_count[world] = 0;

    for (int64_t row = 0; row < pending; ++row) {
        state.free_queue[offset + free_before + row] =
            state.pending_free_slots[offset + row];
    }
    for (int64_t death = 0; death < deaths; ++death) {
        const int64_t agent_id = staging.death_agent_ids[offset + death];
        const int64_t slot = staging.death_slots[offset + death];
        state.pending_free_agent_ids[offset + pending + death] = agent_id;
        state.pending_free_slots[offset + pending + death] = slot;
        state.free_queue[offset + free_before + pending + death] = slot;
        state.death_agent_ids[offset + death] = agent_id;
        state.death_slots[offset + death] = slot;
        state.death_reason_codes[offset + death] =
            staging.death_reason_codes[offset + death];

        const int64_t y = state.pos_y[offset + slot];
        const int64_t x = state.pos_x[offset + slot];
        if (y >= 0 && y < state.grid_size && x >= 0 && x < state.grid_size) {
            const int64_t cell = grid_offset + y * state.grid_size + x;
            if (state.occupancy_agent_id[cell] == agent_id) {
                state.occupancy_agent_id[cell] = -1;
                state.occupancy_slot[cell] = -1;
                state.patch_agent_id[cell] = -1;
            }
        }
        if (agent_id >= 0 && agent_id < state.capacity) {
            state.agent_id_to_slot[offset + agent_id] = -1;
        }
        state.alive[offset + slot] = false;
        state.agent_id[offset + slot] = -1;
        state.pos_y[offset + slot] = -1;
        state.pos_x[offset + slot] = -1;
        state.energy[offset + slot] = 0.0f;
        state.health[offset + slot] = 0.0f;
        state.policy_energy_observation[offset + slot] = __longlong_as_double(
            static_cast<long long>(0x7ff8000000000000ULL));
        state.rest_streak[offset + slot] = 0;
        state.action_cache_token[offset + slot] = -1;
        state.action_cache_cursor[offset + slot] = -1;
        state.action_cache_next[offset + slot] = -1;
        state.death_candidate[offset + slot] = false;
        state.death_reason[offset + slot] = 0;
    }
    for (int64_t row = 0; row < survivors; ++row) {
        state.active_agent_ids[offset + row] = staging.survivor_agent_ids[offset + row];
        state.active_agent_slots[offset + row] = staging.survivor_slots[offset + row];
        state.survivor_agent_ids[offset + row] = staging.survivor_agent_ids[offset + row];
    }
    state.active_count[world] = survivors;
    state.death_count[world] = deaths;
    state.free_queue_count[world] = free_before + pending + deaths;
    state.pending_free_count[world] = 0;

    const int64_t event_before = state.event_active_count[world];
    for (int64_t death = 0; death < deaths; ++death) {
        publish_death_event(state, staging, world, death, generation, tick);
    }
    state.event_active_count[world] = event_before + deaths;
    state.event_status[world] = 0;
}

void require_tensor(
    const torch::Tensor& tensor,
    const torch::Device& device,
    const torch::ScalarType dtype,
    const std::vector<int64_t>& shape,
    const char* group) {
    TORCH_CHECK(tensor.is_cuda(), group, " tensors must be CUDA");
    TORCH_CHECK(tensor.device() == device, group, " tensors must share one device");
    TORCH_CHECK(tensor.is_contiguous(), group, " tensors must be contiguous");
    TORCH_CHECK(tensor.scalar_type() == dtype, group, " tensor dtype mismatch");
    TORCH_CHECK(tensor.sizes().vec() == shape, group, " tensor shape mismatch");
}

}  // namespace

void tensor_world_c4_resident_death_transaction_cuda(
    std::vector<torch::Tensor> a,
    std::vector<torch::Tensor> s,
    int64_t generation,
    int64_t tick) {
    TORCH_CHECK(a.size() == 61, "C4 resident death authoritative ABI is incomplete");
    TORCH_CHECK(s.size() == 8, "C4 resident death staging ABI is incomplete");
    TORCH_CHECK(a[0].dim() == 2, "C4 resident death slot state must be rank two");
    const int64_t batch = a[0].size(0);
    const int64_t capacity = a[0].size(1);
    TORCH_CHECK(batch > 0 && capacity > 0, "C4 resident death dimensions must be positive");
    TORCH_CHECK(a[6].dim() == 3 && a[6].size(0) == batch
        && a[6].size(1) == a[6].size(2), "C4 resident death grid is invalid");
    const int64_t grid_size = a[6].size(1);
    TORCH_CHECK(a[37].dim() == 2 && a[37].size(0) == batch,
        "C4 resident death event tape is invalid");
    const int64_t event_capacity = a[37].size(1);
    TORCH_CHECK(generation >= 0 && tick >= 0, "C4 resident death time is invalid");
    const auto device = a[0].device();
    c10::cuda::CUDAGuard guard(device);
    const std::vector<int64_t> slots{batch, capacity};
    const std::vector<int64_t> batch_shape{batch};
    const std::vector<int64_t> grid{batch, grid_size, grid_size};
    const std::vector<int64_t> events{batch, event_capacity};

    const torch::ScalarType slot_types[37] = {
        torch::kBool, torch::kInt64, torch::kInt64, torch::kInt64,
        torch::kFloat32, torch::kFloat32,
        torch::kInt64, torch::kInt64, torch::kInt32,
        torch::kInt32, torch::kBool, torch::kInt32, torch::kFloat64,
        torch::kInt32, torch::kInt64, torch::kInt32, torch::kInt32,
        torch::kInt64, torch::kInt32, torch::kInt64,
        torch::kInt64, torch::kInt64, torch::kInt64,
        torch::kInt64, torch::kInt64, torch::kInt32, torch::kInt64,
        torch::kInt64, torch::kInt64, torch::kInt64,
        torch::kInt64, torch::kInt32,
        torch::kInt64, torch::kInt64, torch::kInt64, torch::kBool,
        torch::kFloat32};
    for (int index = 0; index < 37; ++index) {
        const bool is_grid = index >= 6 && index <= 8;
        const bool is_batch = index == 19 || index == 22 || index == 26
            || index == 29 || index == 34;
        require_tensor(a[index], device, slot_types[index],
            is_grid ? grid : (is_batch ? batch_shape : slots), "authoritative");
    }
    const torch::ScalarType event_types[24] = {
        torch::kInt32, torch::kInt64, torch::kInt64, torch::kInt64,
        torch::kInt64, torch::kFloat32, torch::kInt32, torch::kInt64,
        torch::kInt64, torch::kInt64, torch::kInt64, torch::kInt64,
        torch::kInt64, torch::kInt64, torch::kInt64, torch::kInt64,
        torch::kInt64, torch::kInt32, torch::kInt32, torch::kInt32,
        torch::kInt8, torch::kFloat32, torch::kInt64, torch::kInt32};
    for (int index = 0; index < 24; ++index) {
        require_tensor(a[37 + index], device, event_types[index],
            index >= 22 ? batch_shape : events, "event");
    }
    const torch::ScalarType staging_types[8] = {
        torch::kInt64, torch::kInt64, torch::kInt32, torch::kInt64,
        torch::kInt64, torch::kInt32, torch::kInt64, torch::kInt32};
    for (int index = 0; index < 8; ++index) {
        require_tensor(s[index], device, staging_types[index],
            index == 3 || index >= 6 ? batch_shape : slots, "staging");
    }

    State state{
        a[0].data_ptr<bool>(), a[1].data_ptr<int64_t>(),
        a[2].data_ptr<int64_t>(), a[3].data_ptr<int64_t>(),
        a[4].data_ptr<float>(), a[5].data_ptr<float>(),
        a[6].data_ptr<int64_t>(), a[7].data_ptr<int64_t>(),
        a[8].data_ptr<int32_t>(), a[9].data_ptr<int32_t>(),
        a[10].data_ptr<bool>(), a[11].data_ptr<int32_t>(),
        a[12].data_ptr<double>(), a[13].data_ptr<int32_t>(),
        a[14].data_ptr<int64_t>(), a[15].data_ptr<int32_t>(),
        a[16].data_ptr<int32_t>(), a[17].data_ptr<int64_t>(),
        a[18].data_ptr<int32_t>(), a[19].data_ptr<int64_t>(),
        a[20].data_ptr<int64_t>(), a[21].data_ptr<int64_t>(),
        a[22].data_ptr<int64_t>(), a[23].data_ptr<int64_t>(),
        a[24].data_ptr<int64_t>(), a[25].data_ptr<int32_t>(),
        a[26].data_ptr<int64_t>(), a[27].data_ptr<int64_t>(),
        a[28].data_ptr<int64_t>(), a[29].data_ptr<int64_t>(),
        a[30].data_ptr<int64_t>(), a[31].data_ptr<int32_t>(),
        a[32].data_ptr<int64_t>(), a[33].data_ptr<int64_t>(),
        a[34].data_ptr<int64_t>(), a[35].data_ptr<bool>(),
        a[36].data_ptr<float>(), a[37].data_ptr<int32_t>(),
        a[38].data_ptr<int64_t>(), a[39].data_ptr<int64_t>(),
        a[40].data_ptr<int64_t>(), a[41].data_ptr<int64_t>(),
        a[42].data_ptr<float>(), a[43].data_ptr<int32_t>(),
        a[44].data_ptr<int64_t>(), a[45].data_ptr<int64_t>(),
        a[46].data_ptr<int64_t>(), a[47].data_ptr<int64_t>(),
        a[48].data_ptr<int64_t>(), a[49].data_ptr<int64_t>(),
        a[50].data_ptr<int64_t>(), a[51].data_ptr<int64_t>(),
        a[52].data_ptr<int64_t>(), a[53].data_ptr<int64_t>(),
        a[54].data_ptr<int32_t>(), a[55].data_ptr<int32_t>(),
        a[56].data_ptr<int32_t>(), a[57].data_ptr<int8_t>(),
        a[58].data_ptr<float>(), a[59].data_ptr<int64_t>(),
        a[60].data_ptr<int32_t>(),
        batch, capacity, grid_size, event_capacity};
    Staging staging{
        s[0].data_ptr<int64_t>(), s[1].data_ptr<int64_t>(),
        s[2].data_ptr<int32_t>(), s[3].data_ptr<int64_t>(),
        s[4].data_ptr<int64_t>(), s[5].data_ptr<int32_t>(),
        s[6].data_ptr<int64_t>(), s[7].data_ptr<int32_t>()};
    const auto stream = at::cuda::getCurrentCUDAStream(device.index());
    preflight_kernel<<<static_cast<unsigned int>(batch), 1, 0, stream>>>(state, staging);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    commit_kernel<<<static_cast<unsigned int>(batch), 1, 0, stream>>>(
        state, staging, generation, tick);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
