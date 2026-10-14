#include "tensor_world_c4_resident_turnover_r3.cu"

namespace {

constexpr int32_t kR8Admission = 80;

__global__ void enforce_maxload_admission_r8(
    Stage birth, TurnStageR3 turn, const int64_t* replay_event_sequence,
    int64_t batch, int64_t capacity) {
    if (threadIdx.x != 0) return;
    const int64_t world = static_cast<int64_t>(blockIdx.x);
    if (world >= batch || turn.status[world] != 0 || birth.status[world] != 0) return;
    const int64_t offset = world * capacity;
    bool admitted = turn.death_count[world] == 8 && birth.planned_count[world] == 8;
    admitted &= replay_event_sequence[world]
        == turn.event_count[world] - turn.death_count[world];
    for (int64_t child = 0; child < birth.planned_count[world]; ++child) {
        bool reuses_new_death = false;
        for (int64_t death = 0; death < turn.death_count[world]; ++death) {
            reuses_new_death |= birth.child_slots[offset + child]
                == turn.death_slots[offset + death];
        }
        admitted &= reuses_new_death;
    }
    if (!admitted) turn.status[world] = kR8Admission;
}

__global__ void publish_replay_sequence_r8(
    Stage birth, TurnStageR3 turn, int64_t* replay_event_sequence, int64_t batch) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    for (int64_t world = 0; world < batch; ++world) {
        if (turn.status[world] != 0 || birth.status[world] != 0) return;
    }
    for (int64_t world = 0; world < batch; ++world) {
        replay_event_sequence[world] = birth.event_sequence[world];
    }
}

}  // namespace

void tensor_world_c4_resident_maxload_r8_cuda(
    std::vector<torch::Tensor> primary,
    std::vector<torch::Tensor> death_authoritative,
    std::vector<torch::Tensor> inherit,
    std::vector<torch::Tensor> zero,
    std::vector<torch::Tensor> minus_one,
    std::vector<torch::Tensor> half,
    std::vector<torch::Tensor> one,
    std::vector<torch::Tensor> birth_staging,
    std::vector<torch::Tensor> turnover_staging,
    torch::Tensor replay_event_sequence,
    int64_t max_births, double parent_energy_share, double minimum_energy_buffer,
    int64_t cooldown_steps, double start_health, double max_energy,
    int64_t life_history_mode, int64_t maturity_start_ticks, int64_t maturity_rise_ticks,
    int64_t fertility_decline_start_ticks, int64_t fertility_decline_ticks,
    double elder_fertility_floor, int64_t generation, int64_t tick) {
    TORCH_CHECK(primary.size() == 101 && death_authoritative.size() == 13,
        "C4 R8 authoritative ABI is incomplete");
    TORCH_CHECK(birth_staging.size() == 18 && turnover_staging.size() == 21,
        "C4 R8 staging ABI is incomplete");
    TORCH_CHECK(inherit.size() == 29 && zero.size() == 55 && minus_one.size() == 11
        && half.size() == 1 && one.size() == 1, "C4 R8 reset ABI is incomplete");
    TORCH_CHECK(primary[0].dim() == 2 && primary[0].size(0) == 4
        && primary[0].size(1) == 32, "C4 R8 is bounded to B4/C32");
    constexpr int64_t batch = 4;
    constexpr int64_t capacity = 32;
    TORCH_CHECK(primary[9].dim() == 3 && primary[9].size(0) == batch
        && primary[9].size(1) == 32 && primary[9].size(2) == 32,
        "C4 R8 is bounded to a 32x32 grid");
    TORCH_CHECK(primary[77].dim() == 2 && primary[77].size(0) == batch
        && primary[77].size(1) == 64, "C4 R8 requires 64 event rows per world");
    constexpr int64_t grid_size = 32;
    constexpr int64_t event_capacity = 64;
    const auto device = primary[0].device();
    TORCH_CHECK(device.is_cuda(), "C4 R8 requires CUDA state");
    c10::cuda::CUDAGuard guard(device);
    validate_abi_r3(primary, death_authoritative, inherit, zero, minus_one, half, one,
        birth_staging, turnover_staging, device, batch, capacity, grid_size, event_capacity);
    require_exact_r3(replay_event_sequence, device, torch::kInt64, {batch},
        "replay event sequence");
    TORCH_CHECK(max_births == 8 && cooldown_steps >= 0 && generation >= 0 && tick >= 0,
        "C4 R8 scalar configuration is outside the admitted envelope");

    State live = make_state(primary, batch, capacity, grid_size, event_capacity);
    live.policy_use_gru = inherit[11].data_ptr<bool>();
    Stage birth{
        birth_staging[0].data_ptr<double>(), birth_staging[1].data_ptr<int32_t>(),
        birth_staging[2].data_ptr<int32_t>(), birth_staging[3].data_ptr<int32_t>(),
        birth_staging[4].data_ptr<int32_t>(), birth_staging[5].data_ptr<float>(),
        birth_staging[6].data_ptr<int64_t>(), birth_staging[7].data_ptr<int64_t>(),
        birth_staging[8].data_ptr<int32_t>(), birth_staging[9].data_ptr<int32_t>(),
        birth_staging[10].data_ptr<int64_t>(), birth_staging[11].data_ptr<int32_t>(),
        birth_staging[12].data_ptr<int64_t>(), birth_staging[13].data_ptr<int64_t>(),
        birth_staging[14].data_ptr<int64_t>(), birth_staging[15].data_ptr<int64_t>(),
        birth_staging[16].data_ptr<int64_t>(), birth_staging[17].data_ptr<int32_t>()};
    DeathStateR3 death{
        death_authoritative[0].data_ptr<bool>(), death_authoritative[1].data_ptr<int32_t>(),
        death_authoritative[2].data_ptr<int64_t>(), death_authoritative[3].data_ptr<int64_t>(),
        death_authoritative[4].data_ptr<int64_t>(), death_authoritative[5].data_ptr<int64_t>(),
        death_authoritative[6].data_ptr<int64_t>(), death_authoritative[7].data_ptr<int32_t>(),
        death_authoritative[8].data_ptr<int64_t>(), death_authoritative[9].data_ptr<int32_t>(),
        death_authoritative[10].data_ptr<int64_t>(), death_authoritative[11].data_ptr<int32_t>(),
        death_authoritative[12].data_ptr<int32_t>()};
    TurnStageR3 turn{
        turnover_staging[0].data_ptr<int64_t>(), turnover_staging[1].data_ptr<int64_t>(),
        turnover_staging[2].data_ptr<int32_t>(), turnover_staging[3].data_ptr<int64_t>(),
        turnover_staging[4].data_ptr<int64_t>(), turnover_staging[5].data_ptr<int64_t>(),
        turnover_staging[6].data_ptr<int32_t>(), turnover_staging[7].data_ptr<bool>(),
        turnover_staging[8].data_ptr<int64_t>(), turnover_staging[9].data_ptr<int64_t>(),
        turnover_staging[10].data_ptr<int64_t>(), turnover_staging[11].data_ptr<float>(),
        turnover_staging[12].data_ptr<float>(), turnover_staging[13].data_ptr<int64_t>(),
        turnover_staging[14].data_ptr<int64_t>(), turnover_staging[15].data_ptr<int32_t>(),
        turnover_staging[16].data_ptr<int32_t>(), turnover_staging[17].data_ptr<int32_t>(),
        turnover_staging[18].data_ptr<double>(), turnover_staging[19].data_ptr<int64_t>(),
        turnover_staging[20].data_ptr<int32_t>()};
    Config config{max_births, parent_energy_share, minimum_energy_buffer,
        static_cast<int32_t>(cooldown_steps), static_cast<float>(start_health),
        static_cast<float>(max_energy), static_cast<int32_t>(life_history_mode),
        static_cast<int32_t>(maturity_start_ticks), static_cast<int32_t>(maturity_rise_ticks),
        static_cast<int32_t>(fertility_decline_start_ticks),
        static_cast<int32_t>(fertility_decline_ticks), elder_fertility_floor, generation, tick};

    const auto stream = at::cuda::getCurrentCUDAStream(device.index());
    prepare_shadow_r3<<<batch, 1, 0, stream>>>(live, death, birth, turn);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
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
    preflight_plan_kernel<<<batch, 1, 0, stream>>>(shadow, birth, config);
    enforce_maxload_admission_r8<<<batch, 1, 0, stream>>>(
        birth, turn, replay_event_sequence.data_ptr<int64_t>(), batch, capacity);
    merge_status_r3<<<1, 1, 0, stream>>>(birth, turn, batch);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    for (auto& tensor : inherit) dispatch_inherit(tensor, birth, batch, capacity, stream);
    for (auto& tensor : zero) dispatch_fill(tensor, 0.0, birth, batch, capacity, stream);
    for (auto& tensor : minus_one) dispatch_fill(tensor, -1.0, birth, batch, capacity, stream);
    for (auto& tensor : half) dispatch_fill(tensor, 0.5, birth, batch, capacity, stream);
    for (auto& tensor : one) dispatch_fill(tensor, 1.0, birth, batch, capacity, stream);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    combined_commit_r3<<<batch, 1, 0, stream>>>(live, death, birth, turn, config);
    publish_replay_sequence_r8<<<1, 1, 0, stream>>>(
        birth, turn, replay_event_sequence.data_ptr<int64_t>(), batch);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
