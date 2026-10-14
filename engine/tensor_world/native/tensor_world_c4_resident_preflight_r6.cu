#include "tensor_world_c4_resident_population_r5.cu"

namespace r6 {

constexpr int32_t kOk = 0;
constexpr int32_t kInvalidQueue = 50;
constexpr int32_t kInvalidIdentity = 51;
constexpr int32_t kInvalidOccupancy = 52;
constexpr int32_t kPeerFailed = 53;
constexpr int32_t kActiveCount = 60;
constexpr int32_t kActiveOrder = 61;
constexpr int32_t kActiveSlot = 62;
constexpr int32_t kNextIdentity = 63;
constexpr int32_t kCompatibilityProjection = 64;

__global__ void all_live_semantic_preflight(
    State live, int32_t* status, Config config) {
    if (threadIdx.x != 0) return;
    const int64_t world = blockIdx.x;
    if (world >= live.batch) return;
    const int64_t offset = world * live.capacity;
    const int64_t grid_offset = world * live.grid_size * live.grid_size;
    if (config.identity_mode == kIdentitySortedMonotonic && status[world] != kOk) return;
    status[world] = kOk;

    if (config.identity_mode == kIdentitySortedMonotonic) {
        const int64_t count = live.active_count[world];
        if (count < 0 || count > live.capacity) {
            status[world] = kActiveCount;
            return;
        }
        int64_t live_count = 0, previous = -1;
        for (int64_t slot = 0; slot < live.capacity; ++slot)
            live_count += live.alive[offset + slot] ? 1 : 0;
        if (live_count != count) {
            status[world] = kActiveCount;
            return;
        }
        for (int64_t row = 0; row < live.capacity; ++row) {
            const int64_t id = live.active_agent_ids[offset + row];
            const int64_t slot = live.active_agent_slots[offset + row];
            if (row >= count) {
                if (id != -1 || slot != -1) status[world] = kActiveCount;
                if (status[world] != kOk) return;
                continue;
            }
            if (id < 0 || (row > 0 && id <= previous)) {
                status[world] = kActiveOrder;
                return;
            }
            if (slot < 0 || slot >= live.capacity || !live.alive[offset + slot]
                || live.agent_id[offset + slot] != id) {
                status[world] = kActiveSlot;
                return;
            }
            previous = id;
        }
        if (live.next_agent_id[world] < 0
            || (count > 0 && live.next_agent_id[world] <= previous)) {
            status[world] = kNextIdentity;
            return;
        }
        for (int64_t id = 0; id < live.capacity; ++id) {
            if (live.agent_id_to_slot[offset + id]
                != r5::sorted_identity_slot(live, world, id)) {
                status[world] = kCompatibilityProjection;
                return;
            }
        }
    }

    const int64_t free_count = live.free_count[world];
    if (free_count < 0 || free_count > live.capacity) {
        status[world] = kInvalidQueue;
        return;
    }
    for (int64_t row = 0; row < free_count; ++row) {
        const int64_t slot = live.free_queue[offset + row];
        if (slot < 0 || slot >= live.capacity || live.alive[offset + slot]
            || live.agent_id[offset + slot] >= 0) {
            status[world] = kInvalidQueue;
            return;
        }
        for (int64_t prior = 0; prior < row; ++prior) {
            if (live.free_queue[offset + prior] == slot) {
                status[world] = kInvalidQueue;
                return;
            }
        }
    }

    for (int64_t slot = 0; slot < live.capacity; ++slot) {
        const bool alive = live.alive[offset + slot];
        const int64_t id = live.agent_id[offset + slot];
        if (alive != (id >= 0)) {
            status[world] = kInvalidIdentity;
            return;
        }
        if (live.slot_tokens[offset + slot] < 0) {
            status[world] = kInvalidIdentity;
            return;
        }
        if (!alive) continue;
        if (config.identity_mode == kIdentityLegacyDense
            && (id >= live.capacity || live.agent_id_to_slot[offset + id] != slot)) {
            status[world] = kInvalidIdentity;
            return;
        }
        const int64_t y = live.pos_y[offset + slot];
        const int64_t x = live.pos_x[offset + slot];
        if (y < 0 || y >= live.grid_size || x < 0 || x >= live.grid_size) {
            status[world] = kInvalidOccupancy;
            return;
        }
        const int64_t cell = grid_offset + y * live.grid_size + x;
        if (live.occupancy_agent[cell] != id
            || live.occupancy_slot[cell] != slot
            || live.patch_agent[cell] != static_cast<int32_t>(id)) {
            status[world] = kInvalidOccupancy;
            return;
        }
    }

    for (int64_t id = 0; config.identity_mode == kIdentityLegacyDense
         && id < live.capacity; ++id) {
        const int64_t slot = live.agent_id_to_slot[offset + id];
        if (slot < 0) continue;
        if (slot >= live.capacity || !live.alive[offset + slot]
            || live.agent_id[offset + slot] != id) {
            status[world] = kInvalidIdentity;
            return;
        }
    }

    for (int64_t local_cell = 0;
         local_cell < live.grid_size * live.grid_size; ++local_cell) {
        const int64_t cell = grid_offset + local_cell;
        const int64_t id = live.occupancy_agent[cell];
        const int64_t slot = live.occupancy_slot[cell];
        if ((id < 0) != (slot < 0)
            || live.patch_agent[cell] != static_cast<int32_t>(id)) {
            status[world] = kInvalidOccupancy;
            return;
        }
        if (id < 0) continue;
        if (slot >= live.capacity || !live.alive[offset + slot]
            || live.agent_id[offset + slot] != id) {
            status[world] = kInvalidOccupancy;
            return;
        }
        const int64_t y = local_cell / live.grid_size;
        const int64_t x = local_cell % live.grid_size;
        if (live.pos_y[offset + slot] != y || live.pos_x[offset + slot] != x) {
            status[world] = kInvalidOccupancy;
            return;
        }
    }
}

__global__ void merge_preflight_status(
    r4::R4Stage stage, r5::TurnStage turn, State live, int32_t* status) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    bool failed = false;
    for (int64_t world = 0; world < live.batch; ++world)
        failed |= status[world] != kOk;
    if (!failed) return;
    for (int64_t world = 0; world < live.batch; ++world) {
        const int32_t code = status[world];
        if (code == kOk) status[world] = kPeerFailed;
        turn.status[world] = code == kInvalidQueue
            ? r5::kInvalidQueue
            : (code == kOk ? r5::kPeerFailed : r5::kIdentityOccupancy);
        stage.base.status[world] = code == kOk
            ? r5::kPeerFailed : r4::kInvalidResident;
    }
}

}  // namespace r6

void tensor_world_c4_resident_preflight_r6_cuda(
    std::vector<torch::Tensor> primary, std::vector<torch::Tensor> death_tensors,
    std::vector<torch::Tensor> inherit, std::vector<torch::Tensor> zero,
    std::vector<torch::Tensor> minus_one, std::vector<torch::Tensor> half,
    std::vector<torch::Tensor> one, std::vector<torch::Tensor> scalar_rng,
    std::vector<torch::Tensor> replay, std::vector<torch::Tensor> staging,
    std::vector<torch::Tensor> turnover, torch::Tensor status,
    int64_t max_births, double parent_energy_share,
    double minimum_energy_buffer, int64_t cooldown_steps,
    double start_health, double max_energy, int64_t life_history_mode,
    int64_t maturity_start_ticks, int64_t maturity_rise_ticks,
    int64_t fertility_decline_start_ticks, int64_t fertility_decline_ticks,
    double elder_fertility_floor, int64_t generation, int64_t tick,
    int64_t environment_stream, int64_t runner_stream) {
    TORCH_CHECK(primary.size() == 101 && death_tensors.size() == 13,
        "C4 R6 authoritative ABI is incomplete");
    TORCH_CHECK(inherit.size() == 29 && zero.size() == 55 && minus_one.size() == 11
        && half.size() == 1 && one.size() == 1, "C4 R6 reset ABI is incomplete");
    TORCH_CHECK(scalar_rng.size() == 8 && replay.size() == 6
        && staging.size() == 49 && turnover.size() == 22,
        "C4 R6 continuation/staging ABI is incomplete");
    const int64_t batch = primary[0].size(0), capacity = primary[0].size(1);
    TORCH_CHECK(batch == 4 && capacity > 0, "C4 R6 is bounded to positive B4");
    const int64_t grid_size = primary[9].size(1);
    const int64_t event_capacity = primary[77].size(1);
    const int64_t streams = replay[0].size(1);
    const int64_t tape_capacity = scalar_rng[4].size(1);
    const auto device = primary[0].device();
    TORCH_CHECK(device.is_cuda(), "C4 R6 requires CUDA state");
    c10::cuda::CUDAGuard guard(device);
    r5::validate_abi(primary, death_tensors, inherit, zero, minus_one, half, one,
        scalar_rng, replay, staging, turnover, device, batch, capacity, grid_size,
        event_capacity, streams, tape_capacity);
    r5::exact(status, device, torch::kInt32, {batch}, "R6 status");

    State live = make_state(primary, batch, capacity, grid_size, event_capacity);
    live.policy_use_gru = inherit[11].data_ptr<bool>();
    r4::Genome genome{
        inherit[2].data_ptr<int32_t>(), inherit[3].data_ptr<int32_t>(),
        inherit[10].data_ptr<int64_t>(), inherit[11].data_ptr<bool>(),
        inherit[12].data_ptr<float>(), inherit[13].data_ptr<float>(),
        inherit[14].data_ptr<float>(), inherit[15].data_ptr<float>(),
        inherit[16].data_ptr<float>(), inherit[17].data_ptr<float>(),
        inherit[18].data_ptr<float>(), inherit[19].data_ptr<float>(),
        inherit[20].data_ptr<float>(), inherit[21].data_ptr<float>(),
        inherit[26].data_ptr<float>(), inherit[27].data_ptr<float>(),
        inherit[28].data_ptr<float>()};
    r4::RandomState python{
        scalar_rng[0].data_ptr<uint32_t>(), scalar_rng[1].data_ptr<int64_t>(),
        scalar_rng[2].data_ptr<bool>(), scalar_rng[3].data_ptr<double>(),
        scalar_rng[4].data_ptr<double>(), scalar_rng[5].data_ptr<double>(),
        scalar_rng[6].data_ptr<int64_t>(), scalar_rng[7].data_ptr<int64_t>(),
        tape_capacity};
    r4::Replay replay_state{
        replay[0].data_ptr<int64_t>(), replay[1].data_ptr<int64_t>(),
        replay[2].data_ptr<int64_t>(), replay[3].data_ptr<int64_t>(),
        replay[4].data_ptr<int64_t>(), streams, environment_stream, runner_stream};
    r4::R4State live_r4{live, genome, python, replay_state, zero[15].data_ptr<float>()};
    r4::R4Stage stage = r4::make_stage(staging);
    r5::DeathState death{
        death_tensors[0].data_ptr<bool>(), death_tensors[1].data_ptr<int32_t>(),
        death_tensors[2].data_ptr<int64_t>(), death_tensors[3].data_ptr<int64_t>(),
        death_tensors[4].data_ptr<int64_t>(), death_tensors[5].data_ptr<int64_t>(),
        death_tensors[6].data_ptr<int64_t>(), death_tensors[7].data_ptr<int32_t>(),
        death_tensors[8].data_ptr<int64_t>(), death_tensors[9].data_ptr<int32_t>(),
        death_tensors[10].data_ptr<int64_t>(), death_tensors[11].data_ptr<int32_t>(),
        death_tensors[12].data_ptr<int32_t>()};
    r5::TurnStage turn = r5::make_turn(turnover);
    Config config{max_births, parent_energy_share, minimum_energy_buffer,
        static_cast<int32_t>(cooldown_steps), static_cast<float>(start_health),
        static_cast<float>(max_energy), static_cast<int32_t>(life_history_mode),
        static_cast<int32_t>(maturity_start_ticks),
        static_cast<int32_t>(maturity_rise_ticks),
        static_cast<int32_t>(fertility_decline_start_ticks),
        static_cast<int32_t>(fertility_decline_ticks), elder_fertility_floor,
        generation, tick};
    const auto stream = at::cuda::getCurrentCUDAStream(device.index());

    r6::all_live_semantic_preflight<<<4, 1, 0, stream>>>(
        live, status.data_ptr<int32_t>(), config);
    r5::prepare_post_death_shadow<<<4, 1, 0, stream>>>(
        live, death, replay_state, turn, config);
    State shadow = live;
    shadow.alive = turn.alive;
    shadow.agent_id = turn.agent_id;
    shadow.pos_y = turn.pos_y;
    shadow.pos_x = turn.pos_x;
    shadow.energy = turn.energy;
    shadow.health = turn.health;
    shadow.occupancy_agent = turn.occupancy_agent;
    shadow.occupancy_slot = turn.occupancy_slot;
    shadow.patch_agent = turn.patch_agent;
    shadow.agent_id_to_slot = turn.id_to_slot;
    shadow.free_queue = turn.virtual_queue;
    shadow.free_count = turn.virtual_count;
    shadow.repro_cooldown = turn.repro_cooldown;
    shadow.policy_energy_observation = turn.policy_energy;
    shadow.event_active_count = turn.event_count;
    shadow.event_status = turn.event_status;
    r4::Replay shadow_replay = replay_state;
    shadow_replay.event_sequence = turn.replay_event_sequence;
    r4::R4State shadow_r4{
        shadow, genome, python, shadow_replay, zero[15].data_ptr<float>()};
    r4::preflight_kernel<<<4, 1, 0, stream>>>(shadow_r4, stage, config);
    r6::merge_preflight_status<<<1, 1, 0, stream>>>(
        stage, turn, live, status.data_ptr<int32_t>());
    r5::merge_all_status<<<1, 1, 0, stream>>>(stage, turn, live, config);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    for (auto& tensor : inherit)
        dispatch_inherit(tensor, stage.base, batch, capacity, stream);
    for (auto& tensor : zero)
        dispatch_fill(tensor, 0.0, stage.base, batch, capacity, stream);
    for (auto& tensor : minus_one)
        dispatch_fill(tensor, -1.0, stage.base, batch, capacity, stream);
    for (auto& tensor : half)
        dispatch_fill(tensor, 0.5, stage.base, batch, capacity, stream);
    for (auto& tensor : one)
        dispatch_fill(tensor, 1.0, stage.base, batch, capacity, stream);
    r5::commit_deaths<<<4, 1, 0, stream>>>(live, death, stage, turn, config);
    r4::commit_kernel<<<4, 1, 0, stream>>>(live_r4, stage, config);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
