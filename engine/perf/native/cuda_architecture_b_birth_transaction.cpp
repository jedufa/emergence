#include <torch/extension.h>

#include <cstdint>
#include <vector>

std::vector<torch::Tensor> eiel_cuda_architecture_b_controlled_birth_identity_commit_v1(
    torch::Tensor slot_to_agent_id,
    torch::Tensor agent_id_to_slot,
    torch::Tensor slot_tokens,
    torch::Tensor alive_mask,
    torch::Tensor pos_y,
    torch::Tensor pos_x,
    torch::Tensor agent_grid,
    torch::Tensor energy,
    torch::Tensor health,
    torch::Tensor injury,
    torch::Tensor scar,
    torch::Tensor age,
    torch::Tensor repro_cooldown,
    torch::Tensor repro_threshold,
    torch::Tensor offspring_count,
    torch::Tensor lineage_founder_id,
    torch::Tensor lineage_depth,
    torch::Tensor lineage_birth_generation,
    torch::Tensor lineage_birth_tick,
    torch::Tensor patch_habitat_lineage_id,
    torch::Tensor policy_energy_observation,
    torch::Tensor survival_steps,
    torch::Tensor rest_streak,
    torch::Tensor action_cache_token,
    torch::Tensor action_cache_cursor,
    torch::Tensor action_cache_next,
    torch::Tensor slot_free_queue,
    torch::Tensor slot_free_queue_count,
    torch::Tensor population_next_id,
    torch::Tensor parent_slots,
    torch::Tensor spawn_positions,
    torch::Tensor planned_count,
    double parent_energy_share,
    double minimum_energy_buffer,
    double start_health,
    int64_t cooldown_steps,
    int64_t generation,
    int64_t tick
);

std::vector<torch::Tensor> eiel_cuda_architecture_b_turnover_refill_identity_commit_v1(
    torch::Tensor slot_to_agent_id,
    torch::Tensor agent_id_to_slot,
    torch::Tensor slot_tokens,
    torch::Tensor alive_mask,
    torch::Tensor pos_y,
    torch::Tensor pos_x,
    torch::Tensor agent_grid,
    torch::Tensor energy,
    torch::Tensor health,
    torch::Tensor injury,
    torch::Tensor scar,
    torch::Tensor age,
    torch::Tensor repro_cooldown,
    torch::Tensor repro_threshold,
    torch::Tensor offspring_count,
    torch::Tensor lineage_founder_id,
    torch::Tensor lineage_depth,
    torch::Tensor lineage_birth_generation,
    torch::Tensor lineage_birth_tick,
    torch::Tensor patch_habitat_lineage_id,
    torch::Tensor policy_energy_observation,
    torch::Tensor survival_steps,
    torch::Tensor rest_streak,
    torch::Tensor action_cache_token,
    torch::Tensor action_cache_cursor,
    torch::Tensor action_cache_next,
    torch::Tensor slot_free_queue,
    torch::Tensor slot_free_queue_count,
    torch::Tensor population_next_id,
    torch::Tensor parent_slots,
    torch::Tensor mate_slots,
    torch::Tensor spawn_positions,
    torch::Tensor planned_count,
    double start_energy,
    double start_health,
    int64_t generation
);

std::vector<torch::Tensor> eiel_cuda_architecture_b_fixed_mlp_child_scalar_genome_v1(
    torch::Tensor slot_to_agent_id,
    torch::Tensor alive_mask,
    torch::Tensor parent_slots,
    torch::Tensor child_slots,
    torch::Tensor planned_count,
    torch::Tensor python_random_words,
    torch::Tensor python_random_index,
    torch::Tensor python_random_has_gauss,
    torch::Tensor python_random_gauss_next,
    torch::Tensor python_random_gauss_primary_tape,
    torch::Tensor python_random_gauss_secondary_tape,
    torch::Tensor python_random_raw_draws,
    torch::Tensor mutation_py_semantic_draws,
    std::vector<torch::Tensor> genome_int32_fields,
    torch::Tensor policy_mem_dim,
    torch::Tensor policy_use_gru,
    std::vector<torch::Tensor> genome_float32_fields,
    int64_t sensor_radius_max
);

std::vector<torch::Tensor>
eiel_cuda_architecture_b_turnover_interleaved_fixed_mlp_child_scalar_genome_v1(
    torch::Tensor slot_to_agent_id,
    torch::Tensor alive_mask,
    torch::Tensor parent_pool_slots,
    torch::Tensor parent_pool_count,
    torch::Tensor child_slots,
    torch::Tensor planned_count,
    torch::Tensor python_random_words,
    torch::Tensor python_random_index,
    torch::Tensor python_random_has_gauss,
    torch::Tensor python_random_gauss_next,
    torch::Tensor python_random_gauss_primary_tape,
    torch::Tensor python_random_gauss_secondary_tape,
    torch::Tensor python_random_raw_draws,
    torch::Tensor mutation_py_semantic_draws,
    std::vector<torch::Tensor> genome_int32_fields,
    torch::Tensor policy_mem_dim,
    torch::Tensor policy_use_gru,
    std::vector<torch::Tensor> genome_float32_fields,
    int64_t sensor_radius_max
);

std::vector<torch::Tensor> eiel_cuda_architecture_b_fixed_mlp_child_drives_v1(
    torch::Tensor slot_to_agent_id,
    torch::Tensor alive_mask,
    torch::Tensor policy_use_gru,
    torch::Tensor parent_slots,
    torch::Tensor child_slots,
    torch::Tensor planned_count,
    torch::Tensor policy_energy_observation,
    torch::Tensor genome_drive_sp_hunger,
    torch::Tensor genome_drive_sp_pain,
    torch::Tensor genome_drive_sp_fatigue,
    torch::Tensor genome_drive_sp_fear,
    torch::Tensor genome_exp_prior_noise,
    torch::Tensor genome_drive_birth_noise,
    torch::Tensor drives,
    torch::Tensor birth_reset_shock,
    torch::Tensor pcg64_state_words,
    torch::Tensor pcg64_buffer_words,
    torch::Tensor mutation_np_semantic_draws,
    double max_energy
);

std::vector<torch::Tensor> eiel_cuda_architecture_b_newborn_chemistry_v1(
    torch::Tensor slot_to_agent_id,
    torch::Tensor alive_mask,
    torch::Tensor child_slots,
    torch::Tensor planned_count,
    torch::Tensor policy_energy_observation,
    torch::Tensor chemistry,
    double max_energy
);

std::vector<torch::Tensor> eiel_cuda_architecture_b_fixed_mlp_child_policy_v1(
    torch::Tensor slot_to_agent_id,
    torch::Tensor alive_mask,
    torch::Tensor parent_slots,
    torch::Tensor child_slots,
    torch::Tensor planned_count,
    torch::Tensor mlp_pack,
    torch::Tensor hmem,
    torch::Tensor policy_use_gru,
    torch::Tensor rng_manager_mutation_torch_cpu_state,
    torch::Tensor policy_birth_delta_tape,
    torch::Tensor policy_birth_mutation_mask_tape,
    torch::Tensor policy_birth_post_rng_state_packed_tape,
    torch::Tensor policy_birth_semantic_draws_tape,
    torch::Tensor policy_birth_tape_cursor,
    torch::Tensor mutation_torch_cpu_semantic_draws
);

std::vector<torch::Tensor> eiel_cuda_architecture_b_fixed_mlp_turnover_child_policy_v1(
    torch::Tensor slot_to_agent_id,
    torch::Tensor alive_mask,
    torch::Tensor parent_slots,
    torch::Tensor mate_slots,
    torch::Tensor child_slots,
    torch::Tensor planned_count,
    torch::Tensor mlp_pack,
    torch::Tensor hmem,
    torch::Tensor policy_use_gru,
    torch::Tensor rng_manager_mutation_torch_cpu_state,
    torch::Tensor policy_birth_delta_tape,
    torch::Tensor policy_birth_mutation_mask_tape,
    torch::Tensor policy_birth_crossover_mask_packed_tape,
    torch::Tensor policy_birth_post_rng_state_packed_tape,
    torch::Tensor policy_birth_semantic_draws_tape,
    torch::Tensor policy_birth_tape_cursor,
    torch::Tensor mutation_torch_cpu_semantic_draws
);

namespace {

constexpr int64_t kIdentityTensorCount = 35;
constexpr int64_t kScalarRngTensorCount = 8;
constexpr int64_t kScalarInt32FieldCount = 8;
constexpr int64_t kScalarFloat32FieldCount = 24;
constexpr int64_t kDriveTensorCount = 6;
constexpr int64_t kPolicyTensorCount = 9;
constexpr int64_t kTurnoverPolicyTensorCount = 10;

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

torch::Tensor gated_counter(
    const torch::Tensor& success,
    const torch::Tensor& value
) {
    return torch::where(success, value, torch::zeros_like(value));
}

}  // namespace

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
) {
    TORCH_CHECK(
        identity_state.size() == kIdentityTensorCount,
        "identity_state must contain 35 tensors"
    );
    for (const int64_t index : {29, 30, 31, 32, 33, 34}) {
        TORCH_CHECK(
            identity_state[index].is_cuda()
                && identity_state[index].scalar_type() == torch::kInt32
                && identity_state[index].dim() == 1
                && identity_state[index].numel() == identity_state[0].numel(),
            "birth action-history tensors must be CUDA int32[capacity]"
        );
    }
    TORCH_CHECK(
        scalar_rng_state.size() == kScalarRngTensorCount,
        "scalar_rng_state must contain 8 tensors"
    );
    TORCH_CHECK(
        genome_int32_fields.size() == kScalarInt32FieldCount,
        "genome_int32_fields must contain 8 tensors"
    );
    TORCH_CHECK(
        genome_float32_fields.size() == kScalarFloat32FieldCount,
        "genome_float32_fields must contain 24 tensors"
    );
    TORCH_CHECK(
        drive_state.size() == kDriveTensorCount,
        "drive_state must contain 6 tensors"
    );
    TORCH_CHECK(
        policy_state.size() == kPolicyTensorCount,
        "policy_state must contain 9 tensors"
    );
    TORCH_CHECK(
        identity_state[13].is_alias_of(genome_float32_fields[16]),
        "repro_threshold must be one authoritative tensor"
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

    const auto identity = eiel_cuda_architecture_b_controlled_birth_identity_commit_v1(
        staged_identity[0],
        staged_identity[1],
        staged_identity[2],
        staged_identity[3],
        staged_identity[4],
        staged_identity[5],
        staged_identity[6],
        staged_identity[7],
        staged_identity[8],
        staged_identity[9],
        staged_identity[10],
        staged_identity[11],
        staged_identity[12],
        staged_identity[13],
        staged_identity[14],
        staged_identity[15],
        staged_identity[16],
        staged_identity[17],
        staged_identity[18],
        staged_identity[19],
        staged_identity[20],
        staged_identity[21],
        staged_identity[22],
        staged_identity[23],
        staged_identity[24],
        staged_identity[25],
        staged_identity[26],
        staged_identity[27],
        staged_identity[28],
        parent_slots,
        spawn_positions,
        planned_count,
        parent_energy_share,
        minimum_energy_buffer,
        start_health,
        cooldown_steps,
        generation,
        tick
    );
    const auto& child_slots = identity[2];
    auto planned_rows = torch::arange(
        child_slots.numel(),
        child_slots.options().dtype(torch::kInt64)
    ).lt(planned_count);
    planned_rows.logical_and_(identity[6].eq(0));
    auto safe_child_slots = torch::where(
        planned_rows,
        child_slots,
        torch::zeros_like(child_slots)
    );
    auto newborn_counts = torch::zeros(
        {staged_identity[0].numel()},
        child_slots.options().dtype(torch::kInt32)
    );
    newborn_counts.index_add_(
        0,
        safe_child_slots,
        planned_rows.to(torch::kInt32)
    );
    auto newborn_mask = newborn_counts.gt(0);
    for (const int64_t index : {29, 30}) {
        staged_identity[index].copy_(torch::where(
            newborn_mask,
            torch::full_like(staged_identity[index], -1),
            staged_identity[index]
        ));
    }
    for (const int64_t index : {31, 32, 33, 34}) {
        staged_identity[index].copy_(torch::where(
            newborn_mask,
            torch::zeros_like(staged_identity[index]),
            staged_identity[index]
        ));
    }

    const auto scalar = eiel_cuda_architecture_b_fixed_mlp_child_scalar_genome_v1(
        staged_identity[0],
        staged_identity[3],
        parent_slots,
        child_slots,
        planned_count,
        staged_scalar_rng[0],
        staged_scalar_rng[1],
        staged_scalar_rng[2],
        staged_scalar_rng[3],
        staged_scalar_rng[4],
        staged_scalar_rng[5],
        staged_scalar_rng[6],
        staged_scalar_rng[7],
        staged_genome_int32,
        staged_policy_mem_dim,
        staged_policy_use_gru,
        staged_genome_float32,
        sensor_radius_max
    );

    const auto drives = eiel_cuda_architecture_b_fixed_mlp_child_drives_v1(
        staged_identity[0],
        staged_identity[3],
        staged_policy_use_gru,
        parent_slots,
        child_slots,
        planned_count,
        staged_identity[20],
        staged_genome_float32[9],
        staged_genome_float32[10],
        staged_genome_float32[11],
        staged_genome_float32[12],
        staged_genome_float32[14],
        staged_genome_float32[15],
        staged_drive[0],
        staged_drive[1],
        staged_drive[2],
        staged_drive[3],
        staged_drive[4],
        max_energy
    );
    const auto chemistry = eiel_cuda_architecture_b_newborn_chemistry_v1(
        staged_identity[0],
        staged_identity[3],
        child_slots,
        planned_count,
        staged_identity[20],
        staged_drive[5],
        max_energy
    );

    const auto policy = eiel_cuda_architecture_b_fixed_mlp_child_policy_v1(
        staged_identity[0],
        staged_identity[3],
        parent_slots,
        child_slots,
        planned_count,
        staged_policy[0],
        staged_policy[1],
        staged_policy_use_gru,
        staged_policy[2],
        staged_policy[3],
        staged_policy[4],
        staged_policy[5],
        staged_policy[6],
        staged_policy[7],
        staged_policy[8]
    );

    const auto identity_ok = identity[6].eq(0);
    const auto scalar_ok = scalar[2].eq(0);
    const auto physiology_status = torch::maximum(drives[11], chemistry[1]);
    const auto drives_ok = physiology_status.eq(0);
    const auto policy_ok = policy[7].eq(0);
    const auto success = identity_ok.logical_and(scalar_ok)
        .logical_and(drives_ok)
        .logical_and(policy_ok);
    const auto commit_state = success.logical_and(planned_count.ne(0));

    commit_vector_if_success(commit_state, identity_state, staged_identity);
    for (const int64_t index : {0, 1, 2, 3, 6, 7}) {
        commit_if_success(
            commit_state,
            scalar_rng_state[index],
            staged_scalar_rng[index]
        );
    }
    commit_vector_if_success(commit_state, genome_int32_fields, staged_genome_int32);
    commit_if_success(commit_state, policy_mem_dim, staged_policy_mem_dim);
    commit_if_success(commit_state, policy_use_gru, staged_policy_use_gru);
    commit_vector_if_success(commit_state, genome_float32_fields, staged_genome_float32);
    commit_vector_if_success(commit_state, drive_state, staged_drive);
    for (const int64_t index : {0, 1, 2, 7, 8}) {
        commit_if_success(commit_state, policy_state[index], staged_policy[index]);
    }

    auto failed_stage = torch::zeros_like(identity[6]);
    failed_stage = torch::where(
        identity[6].ne(0),
        torch::full_like(failed_stage, 1),
        failed_stage
    );
    failed_stage = torch::where(
        failed_stage.eq(0).logical_and(scalar[2].ne(0)),
        torch::full_like(failed_stage, 2),
        failed_stage
    );
    failed_stage = torch::where(
        failed_stage.eq(0).logical_and(physiology_status.ne(0)),
        torch::full_like(failed_stage, 3),
        failed_stage
    );
    failed_stage = torch::where(
        failed_stage.eq(0).logical_and(policy[7].ne(0)),
        torch::full_like(failed_stage, 4),
        failed_stage
    );

    return {
        identity[0],
        identity[1],
        identity[2],
        gated_counter(success, identity[3]),
        identity[4],
        gated_counter(success, identity[5]),
        gated_counter(success, scalar[1]),
        gated_counter(success, drives[6]),
        gated_counter(success, drives[7]),
        gated_counter(success, drives[8]),
        gated_counter(success, drives[9]),
        gated_counter(success, drives[10]),
        gated_counter(success, policy[1]),
        gated_counter(success, policy[2]),
        gated_counter(success, policy[3]),
        torch::where(success, policy[4], torch::zeros_like(policy[4])),
        torch::where(success, policy[5], torch::zeros_like(policy[5])),
        torch::where(success, policy[6], torch::zeros_like(policy[6])),
        identity[6],
        scalar[2],
        physiology_status,
        policy[7],
        failed_stage,
    };
}

std::vector<torch::Tensor>
eiel_cuda_architecture_b_fixed_mlp_turnover_refill_transaction_v1(
    std::vector<torch::Tensor> identity_state,
    torch::Tensor parent_pool_slots,
    torch::Tensor parent_pool_count,
    torch::Tensor spawn_positions,
    torch::Tensor planned_count,
    std::vector<torch::Tensor> scalar_rng_state,
    std::vector<torch::Tensor> genome_int32_fields,
    torch::Tensor policy_mem_dim,
    torch::Tensor policy_use_gru,
    std::vector<torch::Tensor> genome_float32_fields,
    std::vector<torch::Tensor> drive_state,
    std::vector<torch::Tensor> policy_state,
    double start_energy,
    double start_health,
    int64_t generation,
    int64_t sensor_radius_max,
    double max_energy
) {
    TORCH_CHECK(
        identity_state.size() == kIdentityTensorCount,
        "identity_state must contain 35 tensors"
    );
    for (const int64_t index : {29, 30, 31, 32, 33, 34}) {
        TORCH_CHECK(
            identity_state[index].is_cuda()
                && identity_state[index].scalar_type() == torch::kInt32
                && identity_state[index].dim() == 1
                && identity_state[index].numel() == identity_state[0].numel(),
            "turnover action-history tensors must be CUDA int32[capacity]"
        );
    }
    TORCH_CHECK(
        scalar_rng_state.size() == kScalarRngTensorCount,
        "scalar_rng_state must contain 8 tensors"
    );
    TORCH_CHECK(
        genome_int32_fields.size() == kScalarInt32FieldCount,
        "genome_int32_fields must contain 8 tensors"
    );
    TORCH_CHECK(
        genome_float32_fields.size() == kScalarFloat32FieldCount,
        "genome_float32_fields must contain 24 tensors"
    );
    TORCH_CHECK(
        drive_state.size() == kDriveTensorCount,
        "drive_state must contain 6 tensors"
    );
    TORCH_CHECK(
        policy_state.size() == kTurnoverPolicyTensorCount,
        "turnover policy_state must contain 10 tensors"
    );
    TORCH_CHECK(
        identity_state[13].is_alias_of(genome_float32_fields[16]),
        "repro_threshold must be one authoritative tensor"
    );
    TORCH_CHECK(
        identity_state[0].is_cuda()
            && identity_state[0].scalar_type() == torch::kInt64
            && identity_state[0].dim() == 1
            && identity_state[0].numel() >= 2,
        "turnover identity state must have CUDA int64 capacity >= 2"
    );
    const auto device = identity_state[0].device();
    const int64_t slot_count = identity_state[0].numel();
    TORCH_CHECK(
        parent_pool_slots.is_cuda()
            && parent_pool_slots.device() == device
            && parent_pool_slots.scalar_type() == torch::kInt64
            && parent_pool_slots.dim() == 1
            && parent_pool_slots.numel() == slot_count,
        "parent_pool_slots must be CUDA int64[slots] on the identity device"
    );
    TORCH_CHECK(
        parent_pool_count.is_cuda()
            && parent_pool_count.device() == device
            && parent_pool_count.scalar_type() == torch::kInt64
            && parent_pool_count.dim() == 1
            && parent_pool_count.numel() == 1,
        "parent_pool_count must be CUDA int64[1] on the identity device"
    );
    TORCH_CHECK(
        spawn_positions.is_cuda()
            && spawn_positions.device() == device
            && spawn_positions.scalar_type() == torch::kInt32
            && spawn_positions.dim() == 2
            && spawn_positions.size(0) == slot_count
            && spawn_positions.size(1) == 2,
        "spawn_positions must be CUDA int32[slots,2] on the identity device"
    );
    TORCH_CHECK(
        planned_count.is_cuda()
            && planned_count.device() == device
            && planned_count.scalar_type() == torch::kInt64
            && planned_count.dim() == 1
            && planned_count.numel() == 1,
        "planned_count must be CUDA int64[1] on the identity device"
    );

    auto staged_identity = clone_all(identity_state);
    auto allocation_identity = clone_all(identity_state);
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
    for (const int64_t index : {0, 1, 2, 8, 9}) {
        staged_policy[index] = policy_state[index].clone();
    }

    // Allocate child slots on a disposable identity clone. The scalar kernel
    // then owns pair selection and CPython consumption before identity is
    // replayed on the authoritative staged clone with those exact pairs.
    const int64_t plan_capacity = spawn_positions.size(0);
    auto provisional_parent_slots = parent_pool_slots.narrow(0, 0, 1)
        .expand({plan_capacity}).contiguous();
    auto provisional_mate_slots = parent_pool_slots.narrow(0, 1, 1)
        .expand({plan_capacity}).contiguous();
    const auto allocation = eiel_cuda_architecture_b_turnover_refill_identity_commit_v1(
        allocation_identity[0],
        allocation_identity[1],
        allocation_identity[2],
        allocation_identity[3],
        allocation_identity[4],
        allocation_identity[5],
        allocation_identity[6],
        allocation_identity[7],
        allocation_identity[8],
        allocation_identity[9],
        allocation_identity[10],
        allocation_identity[11],
        allocation_identity[12],
        allocation_identity[13],
        allocation_identity[14],
        allocation_identity[15],
        allocation_identity[16],
        allocation_identity[17],
        allocation_identity[18],
        allocation_identity[19],
        allocation_identity[20],
        allocation_identity[21],
        allocation_identity[22],
        allocation_identity[23],
        allocation_identity[24],
        allocation_identity[25],
        allocation_identity[26],
        allocation_identity[27],
        allocation_identity[28],
        provisional_parent_slots,
        provisional_mate_slots,
        spawn_positions,
        planned_count,
        start_energy,
        start_health,
        generation
    );
    const auto& allocated_child_slots = allocation[3];

    const auto scalar =
        eiel_cuda_architecture_b_turnover_interleaved_fixed_mlp_child_scalar_genome_v1(
            allocation_identity[0],
            allocation_identity[3],
            parent_pool_slots,
            parent_pool_count,
            allocated_child_slots,
            planned_count,
            staged_scalar_rng[0],
            staged_scalar_rng[1],
            staged_scalar_rng[2],
            staged_scalar_rng[3],
            staged_scalar_rng[4],
            staged_scalar_rng[5],
            staged_scalar_rng[6],
            staged_scalar_rng[7],
            staged_genome_int32,
            staged_policy_mem_dim,
            staged_policy_use_gru,
            staged_genome_float32,
            sensor_radius_max
        );
    const auto& parent_slots = scalar[0];
    const auto& mate_slots = scalar[1];
    auto scalar_repro_threshold = staged_identity[13].clone();

    const auto identity = eiel_cuda_architecture_b_turnover_refill_identity_commit_v1(
        staged_identity[0],
        staged_identity[1],
        staged_identity[2],
        staged_identity[3],
        staged_identity[4],
        staged_identity[5],
        staged_identity[6],
        staged_identity[7],
        staged_identity[8],
        staged_identity[9],
        staged_identity[10],
        staged_identity[11],
        staged_identity[12],
        staged_identity[13],
        staged_identity[14],
        staged_identity[15],
        staged_identity[16],
        staged_identity[17],
        staged_identity[18],
        staged_identity[19],
        staged_identity[20],
        staged_identity[21],
        staged_identity[22],
        staged_identity[23],
        staged_identity[24],
        staged_identity[25],
        staged_identity[26],
        staged_identity[27],
        staged_identity[28],
        parent_slots,
        mate_slots,
        spawn_positions,
        planned_count,
        start_energy,
        start_health,
        generation
    );
    const auto& child_slots = identity[3];
    auto planned_rows = torch::arange(
        child_slots.numel(),
        child_slots.options().dtype(torch::kInt64)
    ).lt(planned_count);
    auto child_slots_match = torch::where(
        planned_rows,
        child_slots.eq(allocated_child_slots),
        torch::ones_like(planned_rows)
    ).all().reshape({1});
    auto identity_status = torch::maximum(allocation[7], identity[7]);
    identity_status = torch::where(
        identity_status.eq(0).logical_and(child_slots_match.logical_not()),
        torch::full_like(identity_status, 10),
        identity_status
    );
    planned_rows.logical_and_(identity_status.eq(0));
    auto safe_child_slots = torch::where(
        planned_rows,
        child_slots,
        torch::zeros_like(child_slots)
    );
    auto newborn_counts = torch::zeros(
        {staged_identity[0].numel()},
        child_slots.options().dtype(torch::kInt32)
    );
    newborn_counts.index_add_(
        0,
        safe_child_slots,
        planned_rows.to(torch::kInt32)
    );
    auto newborn_mask = newborn_counts.gt(0);
    staged_identity[13].copy_(torch::where(
        newborn_mask,
        scalar_repro_threshold,
        staged_identity[13]
    ));
    for (const int64_t index : {29, 30}) {
        staged_identity[index].copy_(torch::where(
            newborn_mask,
            torch::full_like(staged_identity[index], -1),
            staged_identity[index]
        ));
    }
    for (const int64_t index : {31, 32, 33, 34}) {
        staged_identity[index].copy_(torch::where(
            newborn_mask,
            torch::zeros_like(staged_identity[index]),
            staged_identity[index]
        ));
    }

    // The ordinary drive kernel validates ascending unique parent IDs but uses
    // only child scalar rows after preflight. Turnover sampling permits repeated
    // parents in arbitrary order, so adapt the whole batch to synthetic
    // validation rows while preserving child order and exact PCG64 consumption.
    const int64_t drive_slot_count = plan_capacity == 0 ? 1 : 2 * plan_capacity;
    auto drive_slot_to_agent_id = torch::arange(
        drive_slot_count,
        staged_identity[0].options().dtype(torch::kInt64)
    );
    auto drive_alive_mask = torch::ones(
        {drive_slot_count},
        staged_identity[3].options().dtype(torch::kBool)
    );
    auto drive_policy_use_gru = torch::zeros(
        {drive_slot_count},
        staged_policy_use_gru.options().dtype(torch::kBool)
    );
    auto drive_parent_slots = torch::arange(
        plan_capacity,
        parent_slots.options().dtype(torch::kInt64)
    );
    auto drive_child_slots = drive_parent_slots + plan_capacity;
    const auto drive_child_scalar = [&](const torch::Tensor& source) {
        auto adapted = torch::zeros({drive_slot_count}, source.options());
        adapted.slice(0, plan_capacity, 2 * plan_capacity).copy_(
            source.index_select(0, safe_child_slots)
        );
        return adapted;
    };
    auto drive_policy_energy = drive_child_scalar(staged_identity[20]);
    auto drive_sp_hunger = drive_child_scalar(staged_genome_float32[9]);
    auto drive_sp_pain = drive_child_scalar(staged_genome_float32[10]);
    auto drive_sp_fatigue = drive_child_scalar(staged_genome_float32[11]);
    auto drive_sp_fear = drive_child_scalar(staged_genome_float32[12]);
    auto drive_exp_prior_noise = drive_child_scalar(staged_genome_float32[14]);
    auto drive_birth_noise = drive_child_scalar(staged_genome_float32[15]);
    auto adapted_drives = torch::zeros(
        {drive_slot_count, staged_drive[0].size(1)},
        staged_drive[0].options()
    );
    auto adapted_birth_reset_shock = torch::zeros(
        {drive_slot_count},
        staged_drive[1].options()
    );
    const auto drives = eiel_cuda_architecture_b_fixed_mlp_child_drives_v1(
        drive_slot_to_agent_id,
        drive_alive_mask,
        drive_policy_use_gru,
        drive_parent_slots,
        drive_child_slots,
        planned_count,
        drive_policy_energy,
        drive_sp_hunger,
        drive_sp_pain,
        drive_sp_fatigue,
        drive_sp_fear,
        drive_exp_prior_noise,
        drive_birth_noise,
        adapted_drives,
        adapted_birth_reset_shock,
        staged_drive[2],
        staged_drive[3],
        staged_drive[4],
        max_energy
    );
    auto planned_drive_rows = adapted_drives.slice(
        0,
        plan_capacity,
        2 * plan_capacity
    ) * planned_rows.to(staged_drive[0].scalar_type()).unsqueeze(1);
    auto scattered_drives = torch::zeros_like(staged_drive[0]);
    scattered_drives.index_add_(0, safe_child_slots, planned_drive_rows);
    staged_drive[0].copy_(torch::where(
        newborn_mask.unsqueeze(1),
        scattered_drives,
        staged_drive[0]
    ));
    auto planned_shock = adapted_birth_reset_shock.slice(
        0,
        plan_capacity,
        2 * plan_capacity
    ) * planned_rows.to(staged_drive[1].scalar_type());
    auto scattered_shock = torch::zeros_like(staged_drive[1]);
    scattered_shock.index_add_(0, safe_child_slots, planned_shock);
    staged_drive[1].copy_(torch::where(
        newborn_mask,
        scattered_shock,
        staged_drive[1]
    ));
    const auto chemistry = eiel_cuda_architecture_b_newborn_chemistry_v1(
        staged_identity[0],
        staged_identity[3],
        child_slots,
        planned_count,
        staged_identity[20],
        staged_drive[5],
        max_energy
    );

    const auto policy = eiel_cuda_architecture_b_fixed_mlp_turnover_child_policy_v1(
        staged_identity[0],
        staged_identity[3],
        parent_slots,
        mate_slots,
        child_slots,
        planned_count,
        staged_policy[0],
        staged_policy[1],
        staged_policy_use_gru,
        staged_policy[2],
        staged_policy[3],
        staged_policy[4],
        staged_policy[5],
        staged_policy[6],
        staged_policy[7],
        staged_policy[8],
        staged_policy[9]
    );

    const auto identity_ok = identity_status.eq(0);
    const auto scalar_ok = scalar[4].eq(0);
    const auto physiology_status = torch::maximum(drives[11], chemistry[1]);
    const auto drives_ok = physiology_status.eq(0);
    const auto policy_ok = policy[7].eq(0);
    const auto success = identity_ok.logical_and(scalar_ok)
        .logical_and(drives_ok)
        .logical_and(policy_ok);
    const auto commit_state = success.logical_and(planned_count.ne(0));

    commit_vector_if_success(commit_state, identity_state, staged_identity);
    for (const int64_t index : {0, 1, 2, 3, 6, 7}) {
        commit_if_success(
            commit_state,
            scalar_rng_state[index],
            staged_scalar_rng[index]
        );
    }
    commit_vector_if_success(commit_state, genome_int32_fields, staged_genome_int32);
    commit_if_success(commit_state, policy_mem_dim, staged_policy_mem_dim);
    commit_if_success(commit_state, policy_use_gru, staged_policy_use_gru);
    commit_vector_if_success(commit_state, genome_float32_fields, staged_genome_float32);
    commit_vector_if_success(commit_state, drive_state, staged_drive);
    for (const int64_t index : {0, 1, 2, 8, 9}) {
        commit_if_success(commit_state, policy_state[index], staged_policy[index]);
    }

    auto failed_stage = torch::zeros_like(identity_status);
    failed_stage = torch::where(
        identity_status.ne(0),
        torch::full_like(failed_stage, 1),
        failed_stage
    );
    failed_stage = torch::where(
        failed_stage.eq(0).logical_and(scalar[4].ne(0)),
        torch::full_like(failed_stage, 2),
        failed_stage
    );
    failed_stage = torch::where(
        failed_stage.eq(0).logical_and(physiology_status.ne(0)),
        torch::full_like(failed_stage, 3),
        failed_stage
    );
    failed_stage = torch::where(
        failed_stage.eq(0).logical_and(policy[7].ne(0)),
        torch::full_like(failed_stage, 4),
        failed_stage
    );

    return {
        identity[0],
        identity[1],
        identity[2],
        identity[3],
        gated_counter(success, identity[4]),
        identity[5],
        gated_counter(success, identity[6]),
        gated_counter(success, scalar[3]),
        gated_counter(success, drives[6]),
        gated_counter(success, drives[7]),
        gated_counter(success, drives[8]),
        gated_counter(success, drives[9]),
        gated_counter(success, drives[10]),
        gated_counter(success, policy[1]),
        gated_counter(success, policy[2]),
        gated_counter(success, policy[3]),
        torch::where(success, policy[4], torch::zeros_like(policy[4])),
        torch::where(success, policy[5], torch::zeros_like(policy[5])),
        torch::where(success, policy[6], torch::zeros_like(policy[6])),
        identity_status,
        scalar[4],
        physiology_status,
        policy[7],
        failed_stage,
    };
}

#ifndef EIEL_CUDA_ARCHITECTURE_B_NO_PYBIND
PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def(
        "eiel_cuda_architecture_b_fixed_mlp_birth_transaction_v1",
        &eiel_cuda_architecture_b_fixed_mlp_birth_transaction_v1,
        "Atomic resident CUDA fixed-MLP birth transaction"
    );
    module.def(
        "eiel_cuda_architecture_b_fixed_mlp_turnover_refill_transaction_v1",
        &eiel_cuda_architecture_b_fixed_mlp_turnover_refill_transaction_v1,
        "Atomic resident CUDA fixed-MLP turnover refill transaction"
    );
}
#endif
