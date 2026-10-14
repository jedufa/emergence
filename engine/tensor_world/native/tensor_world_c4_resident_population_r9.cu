#include "tensor_world_c4_resident_preflight_r6.cu"

__global__ void publish_r9_status(r4::R4Stage stage, int32_t* status, int64_t batch) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    for (int64_t world = 0; world < batch; ++world)
        status[world] = stage.base.status[world];
}

void tensor_world_c4_resident_population_r9_cuda(
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
        "C4 R9 authoritative ABI is incomplete");
    TORCH_CHECK(inherit.size() == 29 && zero.size() == 55 && minus_one.size() == 11
        && half.size() == 1 && one.size() == 1, "C4 R9 reset ABI is incomplete");
    TORCH_CHECK(scalar_rng.size() == 8 && replay.size() == 6
        && staging.size() == 49 && turnover.size() == 22,
        "C4 R9 continuation/staging ABI is incomplete");
    const int64_t batch = primary[0].size(0), capacity = primary[0].size(1);
    TORCH_CHECK(batch == 48 && capacity > 0, "C4 R9 is bounded to positive B48");
    const int64_t grid_size = primary[9].size(1);
    const int64_t event_capacity = primary[77].size(1);
    const int64_t streams = replay[0].size(1);
    const int64_t tape_capacity = scalar_rng[4].size(1);
    const auto device = primary[0].device();
    TORCH_CHECK(device.is_cuda(), "C4 R9 requires CUDA state");
    c10::cuda::CUDAGuard guard(device);
    r5::validate_abi(primary, death_tensors, inherit, zero, minus_one, half, one,
        scalar_rng, replay, staging, turnover, device, batch, capacity, grid_size,
        event_capacity, streams, tape_capacity);
    r5::exact(status, device, torch::kInt32, {batch}, "R9 status");

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

    r6::all_live_semantic_preflight<<<batch, 1, 0, stream>>>(
        live, status.data_ptr<int32_t>(), config);
    r5::prepare_post_death_shadow<<<batch, 1, 0, stream>>>(
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
    r4::preflight_kernel<<<batch, 1, 0, stream>>>(shadow_r4, stage, config);
    r6::merge_preflight_status<<<1, 1, 0, stream>>>(
        stage, turn, live, status.data_ptr<int32_t>());
    r5::merge_all_status<<<1, 1, 0, stream>>>(stage, turn, live, config);
    publish_r9_status<<<1, 1, 0, stream>>>(
        stage, status.data_ptr<int32_t>(), batch);
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
    r5::commit_deaths<<<batch, 1, 0, stream>>>(live, death, stage, turn, config);
    r4::commit_kernel<<<batch, 1, 0, stream>>>(live_r4, stage, config);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
