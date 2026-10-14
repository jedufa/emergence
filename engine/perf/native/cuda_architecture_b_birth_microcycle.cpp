#include <torch/extension.h>

#include <cstdint>
#include <vector>

std::vector<torch::Tensor> eiel_cuda_architecture_b_asexual_birth_plan_v1(
    torch::Tensor slot_to_agent_id,
    torch::Tensor alive_mask,
    torch::Tensor policy_use_gru,
    torch::Tensor pos_y,
    torch::Tensor pos_x,
    torch::Tensor agent_grid,
    torch::Tensor energy,
    torch::Tensor policy_energy_observation,
    torch::Tensor health,
    torch::Tensor injury,
    torch::Tensor scar,
    torch::Tensor repro_cooldown,
    torch::Tensor repro_threshold,
    torch::Tensor slot_free_queue,
    torch::Tensor slot_free_queue_count,
    torch::Tensor slot_free_queue_head,
    torch::Tensor pcg64_state_words,
    torch::Tensor pcg64_buffer_words,
    torch::Tensor policy_np_semantic_draws,
    torch::Tensor runner_pcg64_state_words,
    torch::Tensor runner_pcg64_buffer_words,
    torch::Tensor metrics_np_semantic_draws,
    double parent_energy_share,
    double min_energy_buffer,
    double injury_repro_k_injury,
    double injury_repro_k_scar,
    double injury_birth_fail_scar,
    double injury_birth_fail_injury,
    int64_t max_births,
    bool env_policy_alias,
    bool requires_mate,
    bool injury_gate_active,
    bool conception_gate_active,
    bool gestation_gate_active
);

std::vector<torch::Tensor> eiel_cuda_architecture_b_fixed_mlp_birth_transaction_v1(
    std::vector<torch::Tensor> identity_state,
    torch::Tensor parent_slots,
    torch::Tensor spawn_positions,
    torch::Tensor planned_count,
    std::vector<torch::Tensor> scalar_rng_state,
    std::vector<torch::Tensor> genome_int32_fields,
    torch::Tensor policy_mem_dim,
    torch::Tensor policy_use_gru,
    std::vector<torch::Tensor> genome_float32_fields,
    std::vector<torch::Tensor> drive_state,
    std::vector<torch::Tensor> policy_state,
    double parent_energy_share,
    double minimum_energy_buffer,
    double start_health,
    int64_t cooldown_steps,
    int64_t generation,
    int64_t tick,
    int64_t sensor_radius_max,
    double max_energy
);

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
);

namespace {

constexpr int64_t kIdentityTensorCount = 35;
constexpr int64_t kScalarRngTensorCount = 8;
constexpr int64_t kGenomeInt32TensorCount = 8;
constexpr int64_t kGenomeFloat32TensorCount = 24;
constexpr int64_t kDriveTensorCount = 6;
constexpr int64_t kPolicyTensorCount = 9;
constexpr int64_t kEventColumnCount = 22;

std::vector<torch::Tensor> clone_all(const std::vector<torch::Tensor>& tensors) {
    std::vector<torch::Tensor> staged;
    staged.reserve(tensors.size());
    for (const auto& tensor : tensors) {
        staged.push_back(tensor.clone());
    }
    return staged;
}

void commit_if_success(
    const torch::Tensor& success,
    torch::Tensor destination,
    const torch::Tensor& staged
) {
    if (destination.scalar_type() == torch::kUInt32) {
        auto destination_bits = destination.view(torch::kInt32);
        auto staged_bits = staged.view(torch::kInt32);
        destination_bits.copy_(torch::where(success, staged_bits, destination_bits));
        return;
    }
    if (destination.scalar_type() == torch::kUInt64) {
        auto destination_bits = destination.view(torch::kInt64);
        auto staged_bits = staged.view(torch::kInt64);
        destination_bits.copy_(torch::where(success, staged_bits, destination_bits));
        return;
    }
    destination.copy_(torch::where(success, staged, destination));
}

void commit_vector_if_success(
    const torch::Tensor& success,
    const std::vector<torch::Tensor>& destination,
    const std::vector<torch::Tensor>& staged
) {
    TORCH_INTERNAL_ASSERT(destination.size() == staged.size());
    for (size_t index = 0; index < destination.size(); ++index) {
        commit_if_success(success, destination[index], staged[index]);
    }
}

}  // namespace

std::vector<torch::Tensor> eiel_cuda_architecture_b_birth_microcycle_v1(
    std::vector<torch::Tensor> identity_state,
    std::vector<torch::Tensor> scalar_rng_state,
    std::vector<torch::Tensor> genome_int32_fields,
    torch::Tensor policy_mem_dim,
    torch::Tensor policy_use_gru,
    std::vector<torch::Tensor> genome_float32_fields,
    std::vector<torch::Tensor> drive_state,
    std::vector<torch::Tensor> policy_state,
    torch::Tensor planner_pcg64_state_words,
    torch::Tensor planner_pcg64_buffer_words,
    torch::Tensor policy_np_semantic_draws,
    torch::Tensor runner_pcg64_state_words,
    torch::Tensor runner_pcg64_buffer_words,
    torch::Tensor metrics_np_semantic_draws,
    std::vector<torch::Tensor> event_tape_columns,
    torch::Tensor event_tape_active_count,
    torch::Tensor event_tape_status,
    torch::Tensor external_gate_status,
    double parent_energy_share,
    double minimum_energy_buffer,
    double start_health,
    int64_t cooldown_steps,
    int64_t generation,
    int64_t tick,
    int64_t sensor_radius_max,
    double max_energy,
    int64_t max_births,
    double injury_repro_k_injury,
    double injury_repro_k_scar,
    double injury_birth_fail_scar,
    double injury_birth_fail_injury,
    bool env_policy_alias,
    bool requires_mate,
    bool injury_gate_active,
    bool conception_gate_active,
    bool gestation_gate_active
) {
    TORCH_CHECK(identity_state.size() == kIdentityTensorCount, "identity_state must contain 35 tensors");
    TORCH_CHECK(scalar_rng_state.size() == kScalarRngTensorCount, "scalar_rng_state must contain 8 tensors");
    TORCH_CHECK(genome_int32_fields.size() == kGenomeInt32TensorCount, "genome_int32_fields must contain 8 tensors");
    TORCH_CHECK(genome_float32_fields.size() == kGenomeFloat32TensorCount, "genome_float32_fields must contain 24 tensors");
    TORCH_CHECK(drive_state.size() == kDriveTensorCount, "drive_state must contain 6 tensors");
    TORCH_CHECK(policy_state.size() == kPolicyTensorCount, "policy_state must contain 9 tensors");
    TORCH_CHECK(event_tape_columns.size() == kEventColumnCount, "event_tape_columns must contain 22 tensors");
    TORCH_CHECK(identity_state[13].is_alias_of(genome_float32_fields[16]), "repro_threshold must be one authoritative tensor");
    TORCH_CHECK(
        external_gate_status.is_cuda()
            && external_gate_status.is_contiguous()
            && external_gate_status.device() == event_tape_status.device()
            && external_gate_status.scalar_type() == torch::kInt32
            && external_gate_status.numel() == 1,
        "external_gate_status must be contiguous CUDA int32[1]"
    );

    auto staged_identity = clone_all(identity_state);
    auto staged_scalar_rng = scalar_rng_state;
    for (const int64_t index : {0, 1, 2, 3, 6, 7}) {
        staged_scalar_rng[index] = scalar_rng_state[index].clone();
    }
    auto staged_genome_int32 = clone_all(genome_int32_fields);
    auto staged_policy_mem_dim = policy_mem_dim.clone();
    auto staged_policy_use_gru = policy_use_gru.clone();
    auto staged_genome_float32 = clone_all(genome_float32_fields);
    staged_genome_float32[16] = staged_identity[13];
    auto staged_drive = clone_all(drive_state);
    auto staged_policy = policy_state;
    for (const int64_t index : {0, 1, 2, 7, 8}) {
        staged_policy[index] = policy_state[index].clone();
    }
    auto staged_planner_pcg64_state_words = planner_pcg64_state_words.clone();
    auto staged_planner_pcg64_buffer_words = planner_pcg64_buffer_words.clone();
    auto staged_policy_np_semantic_draws = policy_np_semantic_draws.clone();
    auto staged_runner_pcg64_state_words = runner_pcg64_state_words.clone();
    auto staged_runner_pcg64_buffer_words = runner_pcg64_buffer_words.clone();
    auto staged_metrics_np_semantic_draws = metrics_np_semantic_draws.clone();
    auto staged_event_tape_columns = clone_all(event_tape_columns);
    auto staged_event_tape_active_count = event_tape_active_count.clone();
    auto staged_event_tape_status = event_tape_status.clone();

    auto local_free_queue_head = torch::zeros_like(staged_identity[27]);
    const auto plan = eiel_cuda_architecture_b_asexual_birth_plan_v1(
        staged_identity[0],
        staged_identity[3],
        staged_policy_use_gru,
        staged_identity[4],
        staged_identity[5],
        staged_identity[6],
        staged_identity[7],
        staged_identity[20],
        staged_identity[8],
        staged_identity[9],
        staged_identity[10],
        staged_identity[12],
        staged_identity[13],
        staged_identity[26],
        staged_identity[27],
        local_free_queue_head,
        staged_planner_pcg64_state_words,
        staged_planner_pcg64_buffer_words,
        staged_policy_np_semantic_draws,
        staged_runner_pcg64_state_words,
        staged_runner_pcg64_buffer_words,
        staged_metrics_np_semantic_draws,
        parent_energy_share,
        minimum_energy_buffer,
        injury_repro_k_injury,
        injury_repro_k_scar,
        injury_birth_fail_scar,
        injury_birth_fail_injury,
        max_births,
        env_policy_alias,
        requires_mate,
        injury_gate_active,
        conception_gate_active,
        gestation_gate_active
    );

    const auto transaction = eiel_cuda_architecture_b_fixed_mlp_birth_transaction_v1(
        staged_identity,
        plan[5],
        plan[7],
        plan[17],
        staged_scalar_rng,
        staged_genome_int32,
        staged_policy_mem_dim,
        staged_policy_use_gru,
        staged_genome_float32,
        staged_drive,
        staged_policy,
        parent_energy_share,
        minimum_energy_buffer,
        start_health,
        cooldown_steps,
        generation,
        tick,
        sensor_radius_max,
        max_energy
    );

    const auto plan_ok = plan[21].eq(0);
    const auto transaction_ok = transaction[21].eq(0);
    const auto prior_ok = plan_ok.logical_and(transaction_ok);
    const auto zero_count = torch::zeros_like(plan[17]);
    const auto event_decision_count = torch::where(prior_ok, plan[18], zero_count);
    const auto event_birth_count = torch::where(prior_ok, transaction[3], zero_count);
    const auto generation_tensor = torch::full_like(plan[17], generation);
    const auto tick_tensor = torch::full_like(plan[17], tick);
    const auto event = eiel_cuda_architecture_b_birth_lifecycle_events_append_v1(
        staged_event_tape_columns,
        staged_event_tape_active_count,
        staged_event_tape_status,
        staged_identity[0],
        staged_identity[15],
        staged_identity[16],
        plan[9],
        plan[10],
        event_decision_count,
        transaction[0],
        transaction[1],
        plan[5],
        transaction[2],
        transaction[15],
        transaction[16],
        plan[8],
        event_birth_count,
        generation_tensor,
        tick_tensor
    );

    const auto event_ok = event[1].eq(0);
    const auto external_ok = external_gate_status.eq(0);
    const auto success = prior_ok.logical_and(event_ok).logical_and(external_ok);
    commit_vector_if_success(success, identity_state, staged_identity);
    for (const int64_t index : {0, 1, 2, 3, 6, 7}) {
        commit_if_success(success, scalar_rng_state[index], staged_scalar_rng[index]);
    }
    commit_vector_if_success(success, genome_int32_fields, staged_genome_int32);
    commit_if_success(success, policy_mem_dim, staged_policy_mem_dim);
    commit_if_success(success, policy_use_gru, staged_policy_use_gru);
    commit_vector_if_success(success, genome_float32_fields, staged_genome_float32);
    commit_vector_if_success(success, drive_state, staged_drive);
    for (const int64_t index : {0, 1, 2, 7, 8}) {
        commit_if_success(success, policy_state[index], staged_policy[index]);
    }
    commit_if_success(success, planner_pcg64_state_words, staged_planner_pcg64_state_words);
    commit_if_success(success, planner_pcg64_buffer_words, staged_planner_pcg64_buffer_words);
    commit_if_success(success, policy_np_semantic_draws, staged_policy_np_semantic_draws);
    commit_if_success(success, runner_pcg64_state_words, staged_runner_pcg64_state_words);
    commit_if_success(success, runner_pcg64_buffer_words, staged_runner_pcg64_buffer_words);
    commit_if_success(success, metrics_np_semantic_draws, staged_metrics_np_semantic_draws);
    commit_vector_if_success(success, event_tape_columns, staged_event_tape_columns);
    commit_if_success(success, event_tape_active_count, staged_event_tape_active_count);
    commit_if_success(success, event_tape_status, staged_event_tape_status);

    auto failed_stage = torch::zeros_like(event_tape_status);
    failed_stage = torch::where(
        failed_stage.eq(0).logical_and(plan_ok.logical_not()),
        torch::full_like(failed_stage, 1),
        failed_stage
    );
    failed_stage = torch::where(
        failed_stage.eq(0).logical_and(transaction_ok.logical_not()),
        torch::full_like(failed_stage, 2),
        failed_stage
    );
    failed_stage = torch::where(
        failed_stage.eq(0).logical_and(event_ok.logical_not()),
        torch::full_like(failed_stage, 3),
        failed_stage
    );
    failed_stage = torch::where(
        failed_stage.eq(0).logical_and(external_ok.logical_not()),
        torch::full_like(failed_stage, 4),
        failed_stage
    );

    return {
        plan[5],
        plan[6],
        plan[7],
        plan[8],
        plan[9],
        plan[10],
        plan[16],
        plan[17],
        plan[18],
        plan[19],
        plan[20],
        plan[21],
        transaction[0],
        transaction[1],
        transaction[2],
        transaction[3],
        transaction[4],
        transaction[5],
        transaction[6],
        transaction[7],
        transaction[8],
        transaction[9],
        transaction[10],
        transaction[11],
        transaction[12],
        transaction[13],
        transaction[14],
        transaction[15],
        transaction[16],
        transaction[17],
        transaction[18],
        transaction[19],
        transaction[20],
        transaction[21],
        event[0],
        event[1],
        failed_stage,
        plan[25],
        plan[26],
        plan[24],
    };
}

#ifndef EIEL_CUDA_ARCHITECTURE_B_NO_BIRTH_MICROCYCLE_PYBIND
PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.doc() = "Atomic resident CUDA Architecture B ordinary-birth microcycle";
    module.def(
        "eiel_cuda_architecture_b_birth_microcycle_v1",
        &eiel_cuda_architecture_b_birth_microcycle_v1
    );
}
#endif
