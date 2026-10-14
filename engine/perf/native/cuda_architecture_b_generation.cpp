#include <torch/extension.h>

#include <ATen/ops/_assert_async.h>
#include <ATen/ops/searchsorted.h>

#include "cuda_architecture_b_generation_context.hpp"
#include "cuda_architecture_b_modular_tick.hpp"

#include <cmath>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

torch::Tensor eiel_cuda_owner_observation_frame_counted_v1(
    torch::Tensor agent_ids_by_alive_order,
    torch::Tensor active_count,
    torch::Tensor agent_id_to_slot,
    torch::Tensor pos_y,
    torch::Tensor pos_x,
    torch::Tensor agent_grid,
    torch::Tensor food_grid,
    torch::Tensor energy,
    torch::Tensor health,
    torch::Tensor patch_food_amount,
    torch::Tensor patch_cover,
    torch::Tensor patch_shelter,
    torch::Tensor patch_scent,
    torch::Tensor patch_waste,
    torch::Tensor patch_movement_cost,
    bool patch_enabled,
    int64_t view_radius,
    double max_energy,
    double start_health,
    double patch_food_amount_max,
    double patch_scent_max,
    double patch_waste_max,
    int64_t patch_density_radius,
    bool patch_density_include_self,
    bool patch_density_normalize,
    double patch_movement_cost_min,
    double patch_movement_cost_max
);

torch::Tensor eiel_cuda_owner_observation_frame_slotted_v1(
    torch::Tensor agent_ids_by_alive_order,
    torch::Tensor active_slots_by_alive_order,
    torch::Tensor active_count,
    torch::Tensor pos_y,
    torch::Tensor pos_x,
    torch::Tensor agent_grid,
    torch::Tensor food_grid,
    torch::Tensor energy,
    torch::Tensor health,
    torch::Tensor patch_food_amount,
    torch::Tensor patch_cover,
    torch::Tensor patch_shelter,
    torch::Tensor patch_scent,
    torch::Tensor patch_waste,
    torch::Tensor patch_movement_cost,
    bool patch_enabled,
    int64_t view_radius,
    double max_energy,
    double start_health,
    double patch_food_amount_max,
    double patch_scent_max,
    double patch_waste_max,
    int64_t patch_density_radius,
    bool patch_density_include_self,
    bool patch_density_normalize,
    double patch_movement_cost_min,
    double patch_movement_cost_max
);

std::vector<torch::Tensor> eiel_cuda_fixed_mlp_refill_counted_v1(
    torch::Tensor observations,
    torch::Tensor slots,
    torch::Tensor active_count,
    torch::Tensor mlp_pack,
    torch::Tensor drives,
    torch::Tensor attack_bias,
    torch::Tensor action_temperature,
    torch::Tensor drive_logit_gain,
    torch::Tensor slot_tokens,
    torch::Tensor action_cache_actions,
    torch::Tensor action_cache_token,
    torch::Tensor action_cache_cursor,
    torch::Tensor action_cache_next,
    torch::Tensor pcg64_state_words,
    torch::Tensor pcg64_buffer_words,
    int64_t hidden_size,
    int64_t action_count,
    int64_t cache_width,
    bool deterministic,
    int64_t rng_seed,
    int64_t rng_counter,
    double w_hunger_gather,
    double w_pain_rest,
    double w_fatigue_rest,
    double w_pain_attack,
    double w_fear_attack
);

std::vector<torch::Tensor>
eiel_cuda_fixed_mlp_refill_counted_with_residual_v1(
    torch::Tensor observations,
    torch::Tensor slots,
    torch::Tensor active_count,
    torch::Tensor logit_residuals,
    torch::Tensor mlp_pack,
    torch::Tensor drives,
    torch::Tensor attack_bias,
    torch::Tensor action_temperature,
    torch::Tensor drive_logit_gain,
    torch::Tensor slot_tokens,
    torch::Tensor action_cache_actions,
    torch::Tensor action_cache_token,
    torch::Tensor action_cache_cursor,
    torch::Tensor action_cache_next,
    torch::Tensor pcg64_state_words,
    torch::Tensor pcg64_buffer_words,
    int64_t hidden_size,
    int64_t action_count,
    int64_t cache_width,
    bool deterministic,
    int64_t rng_seed,
    int64_t rng_counter,
    double w_hunger_gather,
    double w_pain_rest,
    double w_fatigue_rest,
    double w_pain_attack,
    double w_fear_attack
);

std::vector<torch::Tensor> eiel_cuda_architecture_b_policy_cache_service_v1(
    torch::Tensor active_agent_ids,
    torch::Tensor active_count,
    torch::Tensor slots,
    torch::Tensor slot_tokens,
    torch::Tensor action_cache_actions,
    torch::Tensor action_cache_token,
    torch::Tensor action_cache_cursor,
    torch::Tensor action_cache_next,
    int64_t cache_width
);

torch::Tensor eiel_cuda_architecture_b_flat_food_spawn_v1(
    torch::Tensor food_grid,
    torch::Tensor patch_food_amount,
    torch::Tensor pcg64_state_words,
    torch::Tensor pcg64_buffer_words,
    torch::Tensor policy_np_semantic_draws,
    int64_t max_food,
    int64_t effective_spawn_count,
    bool patch_enabled,
    double patch_food_presence_threshold,
    double patch_food_initial_value
);

std::vector<torch::Tensor> eiel_cuda_architecture_b_lifecycle_death_cull_v1(
    torch::Tensor slot_to_agent_id,
    torch::Tensor agent_id_to_slot,
    torch::Tensor alive_mask,
    torch::Tensor pos_y,
    torch::Tensor pos_x,
    torch::Tensor agent_grid,
    torch::Tensor energy,
    torch::Tensor health,
    torch::Tensor policy_energy_observation,
    torch::Tensor rest_streak,
    torch::Tensor action_cache_token,
    torch::Tensor action_cache_cursor,
    torch::Tensor action_cache_next,
    torch::Tensor pending_free_agent_ids,
    torch::Tensor pending_free_slots,
    torch::Tensor pending_free_count
);

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
);

std::vector<torch::Tensor> eiel_cuda_architecture_b_release_pending_free_slots_v1(
    torch::Tensor slot_to_agent_id,
    torch::Tensor alive_mask,
    torch::Tensor slot_free_queue,
    torch::Tensor slot_free_queue_count,
    torch::Tensor pending_free_slots,
    torch::Tensor pending_free_count,
    torch::Tensor pending_release_cursor
);

std::vector<torch::Tensor> eiel_cuda_architecture_b_death_class_concat_v1(
    torch::Tensor action_agent_ids,
    torch::Tensor action_slots,
    torch::Tensor action_reason_codes,
    torch::Tensor action_count,
    torch::Tensor residual_agent_ids,
    torch::Tensor residual_slots,
    torch::Tensor residual_reason_codes,
    torch::Tensor residual_count
);

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
);

std::vector<torch::Tensor> eiel_cuda_architecture_b_turnover_plan_v1(
    torch::Tensor slot_to_agent_id,
    torch::Tensor alive_mask,
    torch::Tensor survival_steps,
    torch::Tensor energy_collected_exact,
    torch::Tensor offspring_count,
    torch::Tensor attacks,
    torch::Tensor early_survival_steps,
    torch::Tensor early_energy_collected_exact,
    torch::Tensor early_injury_gained,
    torch::Tensor early_life_weight,
    torch::Tensor action_switches,
    torch::Tensor action_oscillations,
    torch::Tensor drive_volatility,
    torch::Tensor selfharm_events,
    torch::Tensor pred_err_sum,
    torch::Tensor pred_err_count,
    int64_t population_size,
    double cull_fraction,
    double parent_fraction,
    double coherence_thrashing_weight,
    double coherence_drive_volatility_weight,
    double coherence_selfharm_weight,
    double coherence_pred_err_weight
);

std::vector<torch::Tensor> eiel_cuda_architecture_b_turnover_cull_release_v1(
    torch::Tensor slot_to_agent_id,
    torch::Tensor agent_id_to_slot,
    torch::Tensor alive_mask,
    torch::Tensor pos_y,
    torch::Tensor pos_x,
    torch::Tensor agent_grid,
    torch::Tensor energy,
    torch::Tensor health,
    torch::Tensor policy_energy_observation,
    torch::Tensor action_cache_token,
    torch::Tensor action_cache_cursor,
    torch::Tensor action_cache_next,
    torch::Tensor action_cache_actions,
    torch::Tensor slot_free_queue,
    torch::Tensor slot_free_queue_count,
    torch::Tensor pending_free_agent_ids,
    torch::Tensor pending_free_slots,
    torch::Tensor pending_free_count,
    torch::Tensor culled_agent_ids,
    torch::Tensor culled_slots,
    torch::Tensor culled_count,
    torch::Tensor turnover_plan_status
);

std::vector<torch::Tensor> eiel_cuda_architecture_b_turnover_refill_plan_v1(
    torch::Tensor slot_to_agent_id,
    torch::Tensor alive_mask,
    torch::Tensor agent_grid,
    torch::Tensor parent_pool_slots,
    torch::Tensor parent_pool_count,
    torch::Tensor refill_count,
    torch::Tensor turnover_plan_status,
    torch::Tensor python_random_words,
    torch::Tensor python_random_index,
    torch::Tensor python_random_raw_draws,
    torch::Tensor mutation_py_semantic_draws,
    torch::Tensor env_pcg64_state_words,
    torch::Tensor env_pcg64_buffer_words,
    torch::Tensor policy_np_semantic_draws,
    bool env_policy_alias
);

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
);

std::vector<torch::Tensor> eiel_cuda_architecture_b_runner_shuffle_counted_v1(
    torch::Tensor ordered_slots,
    torch::Tensor active_count,
    torch::Tensor pcg64_state_words,
    torch::Tensor pcg64_buffer_words
);

std::vector<torch::Tensor>
eiel_cuda_architecture_b_advance_precomputed_drive_noise_rng_v1(
    torch::Tensor ordered_slots,
    torch::Tensor active_count,
    torch::Tensor drive_noise,
    torch::Tensor pcg64_state_words,
    torch::Tensor pcg64_buffer_words
);

std::vector<torch::Tensor> eiel_cuda_architecture_b_chemistry_update_counted_v1(
    torch::Tensor chemistry,
    torch::Tensor energy,
    torch::Tensor health,
    torch::Tensor injury,
    torch::Tensor drives,
    torch::Tensor alive_mask,
    torch::Tensor ordered_slots,
    torch::Tensor active_count,
    torch::Tensor pcg64_state_words,
    torch::Tensor pcg64_buffer_words,
    torch::Tensor policy_np_semantic_draws,
    double max_energy,
    double start_health
);

std::vector<torch::Tensor> eiel_cuda_scripted_owner_action_generation_counted_v1(
    torch::Tensor agent_ids_by_alive_order,
    torch::Tensor active_count,
    torch::Tensor agent_id_to_slot,
    torch::Tensor actions_by_tick_slot,
    torch::Tensor pos_y,
    torch::Tensor pos_x,
    torch::Tensor agent_grid,
    torch::Tensor food_grid,
    torch::Tensor patch_food_amount,
    torch::Tensor patch_scent_emit_pending,
    torch::Tensor patch_waste_emit_pending,
    torch::Tensor patch_movement_cost,
    torch::Tensor energy,
    torch::Tensor health,
    bool patch_enabled,
    double move_cost,
    double gather_cost,
    double rest_cost,
    double attack_cost,
    double action_cost_mult,
    double rest_cost_mult,
    double max_energy,
    double food_energy,
    double patch_food_gather_amount,
    double patch_food_presence_threshold,
    double patch_scent_move_emit,
    double patch_waste_move_emit,
    double patch_scent_gather_emit,
    double patch_waste_consume_emit,
    bool patch_movement_cost_apply_enable,
    double patch_movement_cost_energy_scale
);

std::vector<torch::Tensor> eiel_cuda_scripted_owner_action_generation_slotted_v1(
    torch::Tensor agent_ids_by_alive_order,
    torch::Tensor active_slots_by_alive_order,
    torch::Tensor active_count,
    torch::Tensor actions_by_tick_slot,
    torch::Tensor pos_y,
    torch::Tensor pos_x,
    torch::Tensor agent_grid,
    torch::Tensor food_grid,
    torch::Tensor patch_food_amount,
    torch::Tensor patch_scent_emit_pending,
    torch::Tensor patch_waste_emit_pending,
    torch::Tensor patch_movement_cost,
    torch::Tensor energy,
    torch::Tensor health,
    bool patch_enabled,
    double move_cost,
    double gather_cost,
    double rest_cost,
    double attack_cost,
    double action_cost_mult,
    double rest_cost_mult,
    double max_energy,
    double food_energy,
    double patch_food_gather_amount,
    double patch_food_presence_threshold,
    double patch_scent_move_emit,
    double patch_waste_move_emit,
    double patch_scent_gather_emit,
    double patch_waste_consume_emit,
    bool patch_movement_cost_apply_enable,
    double patch_movement_cost_energy_scale
);

torch::Tensor eiel_cuda_architecture_b_patch_context_p2_v1(
    torch::Tensor patch_food_amount,
    torch::Tensor patch_fertility,
    torch::Tensor patch_scent,
    torch::Tensor patch_waste,
    torch::Tensor patch_scent_emit_pending,
    torch::Tensor patch_waste_emit_pending,
    torch::Tensor agent_grid,
    torch::Tensor patch_density,
    torch::Tensor food_grid,
    int64_t steps,
    double patch_scent_decay_per_step,
    double patch_waste_decay_per_step,
    double patch_scent_presence_emit,
    double patch_waste_presence_emit,
    double patch_waste_density_emit_scale,
    double patch_scent_max,
    double patch_waste_max,
    double patch_food_amount_max,
    double patch_food_regrow_per_step,
    bool patch_food_regrow_use_fertility,
    bool patch_food_regrow_cap_by_fertility,
    double patch_food_presence_threshold,
    int64_t patch_density_radius,
    bool patch_density_include_self,
    bool patch_density_normalize,
    bool patch_world_waste_consequence_enable,
    bool patch_waste_regrowth_penalty_enable,
    double patch_waste_regrowth_penalty_scale,
    double patch_waste_regrowth_penalty_max,
    bool patch_density_regrowth_penalty_enable,
    double patch_density_regrowth_penalty_scale,
    double patch_density_regrowth_penalty_max,
    double patch_regrowth_min_multiplier
);

std::vector<torch::Tensor> eiel_cuda_architecture_b_active_mechanics_v1(
    torch::Tensor slot_to_agent_id,
    torch::Tensor alive_mask,
    torch::Tensor actions_by_slot,
    torch::Tensor energy,
    torch::Tensor health,
    torch::Tensor injury,
    torch::Tensor age,
    torch::Tensor policy_energy_observation,
    torch::Tensor rest_streak,
    torch::Tensor survival_steps,
    torch::Tensor gru_upkeep_cost,
    torch::Tensor cost_charge_scalar,
    torch::Tensor cost_charge_maturation_scalar,
    torch::Tensor drives,
    torch::Tensor drive_inertia,
    torch::Tensor drive_noise,
    torch::Tensor prev_drive_delta,
    torch::Tensor drive_volatility,
    torch::Tensor drive_jitter,
    torch::Tensor episode0_drive_jitter_sum,
    torch::Tensor last_energy_delta,
    torch::Tensor last_health_delta,
    torch::Tensor last_injury_delta,
    torch::Tensor selfharm_events,
    torch::Tensor episode0_selfharm_events,
    torch::Tensor pre_energy,
    torch::Tensor pre_health,
    torch::Tensor pre_injury,
    torch::Tensor drive_noise_samples,
    torch::Tensor drive_noise_sample_valid,
    bool enable_rest_charge,
    int64_t rest_streak_free,
    double rest_charge_energy,
    double rest_charge_growth,
    double rest_deficit_to_health,
    int64_t developmental_maturity_age,
    int64_t episode0_k,
    double max_energy,
    double start_health,
    double drive_hunger_gain,
    double drive_hunger_decay,
    double drive_pain_gain,
    double drive_pain_decay,
    double drive_fatigue_gain,
    double drive_fatigue_rest_relief,
    double drive_fatigue_decay,
    double drive_fear_gain,
    double drive_fear_decay
);

namespace {

constexpr int64_t kPhaseEDevelopmentMatureStartStep = 20;
constexpr int64_t kPhaseEDevelopmentLateStartStep = 40;

std::vector<torch::Tensor> clone_tensor_vector(
    const std::vector<torch::Tensor>& tensors
) {
    std::vector<torch::Tensor> result;
    result.reserve(tensors.size());
    for (const auto& tensor : tensors) {
        result.push_back(tensor.clone());
    }
    return result;
}

void commit_tensor(
    const torch::Tensor& success,
    const torch::Tensor& target,
    const torch::Tensor& staged
) {
    if (
        target.scalar_type() == torch::kUInt32
        || target.scalar_type() == torch::kUInt64
    ) {
        auto signed_type = target.scalar_type() == torch::kUInt32
            ? torch::kInt32
            : torch::kInt64;
        auto target_bits = target.view(signed_type);
        auto staged_bits = staged.view(signed_type);
        target_bits.copy_(torch::where(success, staged_bits, target_bits));
        return;
    }
    target.copy_(torch::where(success, staged, target));
}

void commit_tensor_vector(
    const torch::Tensor& success,
    const std::vector<torch::Tensor>& targets,
    const std::vector<torch::Tensor>& staged
) {
    TORCH_CHECK(targets.size() == staged.size(), "staged tensor vector size mismatch");
    for (size_t index = 0; index < targets.size(); ++index) {
        commit_tensor(success, targets[index], staged[index]);
    }
}

void reset_newborn_slot_state(
    const torch::Tensor& child_slots,
    const torch::Tensor& committed_count,
    std::vector<torch::Tensor>& turnover_state,
    torch::Tensor action_cache_actions,
    torch::Tensor prev_drive_delta,
    torch::Tensor drive_volatility,
    torch::Tensor drive_jitter,
    torch::Tensor episode0_drive_jitter_sum,
    torch::Tensor last_energy_delta,
    torch::Tensor last_health_delta,
    torch::Tensor last_injury_delta,
    torch::Tensor selfharm_events,
    torch::Tensor episode0_selfharm_events
) {
    TORCH_CHECK(turnover_state.size() == 18, "turnover state must contain 18 tensors");
    auto rows = torch::arange(
        child_slots.numel(),
        child_slots.options().dtype(torch::kInt64)
    );
    auto valid_rows = rows.lt(committed_count);
    auto safe_slots = torch::where(
        valid_rows,
        child_slots,
        torch::zeros_like(child_slots)
    );
    auto newborn_counts = torch::zeros(
        {turnover_state[0].numel()},
        child_slots.options().dtype(torch::kInt32)
    );
    newborn_counts.index_add_(0, safe_slots, valid_rows.to(torch::kInt32));
    auto newborn = newborn_counts.gt(0);

    for (const int64_t index : {0, 1, 2, 3, 4, 5, 6, 8, 9, 12, 13, 14, 15, 17}) {
        turnover_state[index].copy_(torch::where(
            newborn,
            torch::zeros_like(turnover_state[index]),
            turnover_state[index]
        ));
    }
    turnover_state[7].copy_(torch::where(
        newborn,
        torch::ones_like(turnover_state[7]),
        turnover_state[7]
    ));
    turnover_state[10].copy_(torch::where(
        newborn,
        torch::full_like(turnover_state[10], -1),
        turnover_state[10]
    ));
    turnover_state[11].copy_(torch::where(
        newborn,
        torch::zeros_like(turnover_state[11]),
        turnover_state[11]
    ));
    turnover_state[16].copy_(torch::where(
        newborn,
        torch::ones_like(turnover_state[16]),
        turnover_state[16]
    ));

    auto row_mask = newborn.unsqueeze(1);
    action_cache_actions.copy_(torch::where(
        row_mask,
        torch::full_like(action_cache_actions, -1),
        action_cache_actions
    ));
    prev_drive_delta.copy_(torch::where(
        row_mask,
        torch::zeros_like(prev_drive_delta),
        prev_drive_delta
    ));
    for (auto tensor : {
             drive_volatility,
             drive_jitter,
             episode0_drive_jitter_sum,
             last_energy_delta,
             last_health_delta,
             last_injury_delta,
             selfharm_events,
             episode0_selfharm_events,
         }) {
        tensor.copy_(torch::where(newborn, torch::zeros_like(tensor), tensor));
    }
}

struct CompactIdentityPairs {
    torch::Tensor agent_ids;
    torch::Tensor slots;
    torch::Tensor active_count;
};

CompactIdentityPairs compact_identity_pairs_by_agent_id(
    const torch::Tensor& slot_to_agent_id,
    const torch::Tensor& alive_mask
) {
    const int64_t capacity = slot_to_agent_id.numel();
    auto inactive_key = torch::full_like(
        slot_to_agent_id,
        std::numeric_limits<int64_t>::max()
    );
    auto sort_keys = torch::where(alive_mask, slot_to_agent_id, inactive_key);
    auto order = torch::argsort(sort_keys);
    return {
        sort_keys.index_select(0, order).contiguous(),
        order.to(torch::kInt32).contiguous(),
        torch::sum(alive_mask.to(torch::kInt64)).reshape({1}),
    };
}

torch::Tensor lookup_sorted_values(
    const torch::Tensor& sorted_agent_ids,
    const torch::Tensor& sorted_values,
    const torch::Tensor& query_agent_ids
) {
    TORCH_CHECK(
        sorted_agent_ids.dim() == 1
            && sorted_values.dim() == 1
            && sorted_agent_ids.numel() == sorted_values.numel()
            && sorted_agent_ids.numel() > 0,
        "compact identity lookup requires aligned nonempty rank-one tensors"
    );
    auto positions = at::searchsorted(sorted_agent_ids, query_agent_ids);
    auto bounded_positions = torch::clamp(
        positions, 0, sorted_agent_ids.numel() - 1
    );
    auto matches = sorted_agent_ids.index_select(0, bounded_positions)
        .eq(query_agent_ids)
        .logical_and(positions.lt(sorted_agent_ids.numel()));
    return torch::where(
        matches,
        sorted_values.index_select(0, bounded_positions),
        torch::full_like(bounded_positions, -1)
    );
}

}  // namespace

namespace eiel {
namespace cuda_architecture_b {

class GenerationContext::Impl final : private GenerationContextArguments {
public:
    explicit Impl(GenerationContextArguments arguments)
        : GenerationContextArguments(std::move(arguments)) {
        prepare();
    }

    void prepare() {
    TORCH_CHECK(steps > 0, "Architecture B generation steps must be positive");
    TORCH_CHECK(agent_ids_by_alive_order.is_cuda(), "Architecture B generation state must be CUDA-resident");
    TORCH_CHECK(agent_ids_by_alive_order.dim() == 1, "agent_ids_by_alive_order must be rank 1");
    TORCH_CHECK(energy.dim() == 1, "energy must be a rank-1 slot bank");
    TORCH_CHECK(event_tape_columns.size() == 22, "generation event tape requires 6 legacy and 16 typed columns");
    TORCH_CHECK(max_births_per_tick >= 0, "max_births_per_tick must be non-negative");
    TORCH_CHECK(!enable_biomes, "Architecture B resident food spawn does not support biome weighting");
    TORCH_CHECK(!birth_enabled || resident_lifecycle_state_enabled, "birth path requires resident lifecycle state");
    TORCH_CHECK(
        !generation_turnover_enabled || resident_lifecycle_state_enabled,
        "turnover path requires resident lifecycle state"
    );
    TORCH_CHECK(
        metrics_np_semantic_draws.is_cuda()
            && metrics_np_semantic_draws.scalar_type() == torch::kInt64
            && metrics_np_semantic_draws.dim() == 1
            && metrics_np_semantic_draws.numel() == 1,
        "metrics_np_semantic_draws must be CUDA int64[1]"
    );
    if (resident_lifecycle_state_enabled) {
        TORCH_CHECK(birth_identity_state.size() == 35, "birth identity state must contain 35 tensors");
        TORCH_CHECK(birth_identity_state.at(0).is_alias_of(slot_to_agent_id), "birth slot identity must alias generation state");
        TORCH_CHECK(birth_identity_state.at(1).is_alias_of(agent_id_to_slot), "birth agent map must alias generation state");
        TORCH_CHECK(birth_identity_state.at(3).is_alias_of(alive_mask), "birth alive mask must alias generation state");
        TORCH_CHECK(birth_identity_state.at(26).size(0) >= energy.size(0), "birth free queue must cover slot capacity");
        TORCH_CHECK(
            birth_planner_pcg64_state_words.is_alias_of(pcg64_state_words)
                && birth_planner_pcg64_buffer_words.is_alias_of(pcg64_buffer_words),
            "birth planner must alias the shared env/policy PCG64 state"
        );
    }
    if (birth_enabled || generation_turnover_enabled) {
        TORCH_CHECK(birth_scalar_rng_state.size() == 8, "birth scalar RNG state must contain 8 tensors");
        TORCH_CHECK(birth_genome_int32_fields.size() == 8, "birth integer genome state must contain 8 tensors");
        TORCH_CHECK(birth_genome_float32_fields.size() == 24, "birth float genome state must contain 24 tensors");
        TORCH_CHECK(birth_drive_state.size() == 6, "birth drive state must contain 6 tensors");
        TORCH_CHECK(turnover_state.size() == 18, "birth lifecycle state must contain 18 turnover/stat tensors");
    }
    if (birth_enabled) {
        TORCH_CHECK(max_births_per_tick > 0, "enabled birth path requires positive max_births_per_tick");
        TORCH_CHECK(birth_policy_state.size() == 9, "birth policy state must contain 9 tensors");
    }
    if (generation_turnover_enabled) {
        TORCH_CHECK(population_size > 0, "turnover population_size must be positive");
        TORCH_CHECK(cull_fraction >= 0.0 && cull_fraction <= 1.0, "turnover cull_fraction must be in [0, 1]");
        TORCH_CHECK(parent_fraction >= 0.0 && parent_fraction <= 1.0, "turnover parent_fraction must be in [0, 1]");
        TORCH_CHECK(turnover_policy_state.size() == 10, "turnover policy state must contain 10 tensors");
        TORCH_CHECK(active_mechanics_enabled, "turnover requires active lifecycle mechanics");
    }
    if (active_mechanics_enabled) {
        TORCH_CHECK(
            chemistry.is_cuda()
                && chemistry.scalar_type() == torch::kFloat32
                && chemistry.dim() == 2
                && chemistry.size(0) == energy.size(0)
                && chemistry.size(1) >= 6,
            "active generation chemistry must be CUDA float32[capacity,>=6]"
        );
        TORCH_CHECK(
            runner_pcg64_state_words.is_cuda()
                && runner_pcg64_state_words.scalar_type() == torch::kUInt64
                && runner_pcg64_state_words.numel() == 4,
            "active generation runner PCG64 state must be CUDA uint64[4]"
        );
        TORCH_CHECK(
            runner_pcg64_buffer_words.is_cuda()
                && runner_pcg64_buffer_words.scalar_type() == torch::kInt64
                && runner_pcg64_buffer_words.numel() == 2,
            "active generation runner PCG64 buffer must be CUDA int64[2]"
        );
        TORCH_CHECK(
            pcg64_state_words.is_cuda()
                && pcg64_state_words.scalar_type() == torch::kUInt64
                && pcg64_state_words.numel() == 4
                && pcg64_buffer_words.is_cuda()
                && pcg64_buffer_words.scalar_type() == torch::kInt64
                && pcg64_buffer_words.numel() == 2,
            "active generation chemistry requires shared env/policy PCG64 state"
        );
        TORCH_CHECK(
            drive_noise_samples_by_tick_slot.dim() == 3
                && drive_noise_samples_by_tick_slot.size(0) >= start_tick + steps,
            "drive_noise_samples_by_tick_slot must provide every generation tick"
        );
        TORCH_CHECK(
            drive_noise_samples_valid_by_tick_slot.dim() == 2
                && drive_noise_samples_valid_by_tick_slot.size(0) >= start_tick + steps,
            "drive_noise_samples_valid_by_tick_slot must provide every generation tick"
        );
    }

    capacity = energy.size(0);
    for (const auto& tensor : {
             last_action,
             last_last_action,
             action_switches,
             action_oscillations,
             episode0_switches,
             episode0_oscillations,
         }) {
        TORCH_CHECK(
            tensor.is_cuda()
                && tensor.scalar_type() == torch::kInt32
                && tensor.dim() == 1
                && tensor.numel() == capacity,
            "action-history tensors must be CUDA int32[capacity]"
        );
    }
    TORCH_CHECK(
        agent_ids_by_alive_order.size(0) <= capacity,
        "agent_ids_by_alive_order exceeds the fixed slot capacity"
    );
    auto initial_identity_pairs = compact_identity_pairs_by_agent_id(
        slot_to_agent_id, alive_mask
    );
    initial_slots = initial_identity_pairs.slots.slice(
        0, 0, agent_ids_by_alive_order.size(0)
    ).contiguous();
    at::_assert_async(
        torch::all(
            initial_identity_pairs.agent_ids.slice(
                0, 0, agent_ids_by_alive_order.size(0)
            ) == agent_ids_by_alive_order
        ),
        "initial alive-order identities disagree with compact slot ownership"
    );
    active_agent_ids = torch::full(
        {capacity},
        -1,
        agent_ids_by_alive_order.options()
    );
    active_agent_ids.slice(0, 0, agent_ids_by_alive_order.size(0)).copy_(
        agent_ids_by_alive_order
    );
    active_count = torch::full(
        {1},
        agent_ids_by_alive_order.size(0),
        agent_ids_by_alive_order.options()
    );
    reporting_row_count = agent_ids_by_alive_order.size(0);
    reporting_status = torch::full(
        {1}, 1, energy.options().dtype(torch::kInt32)
    );
    reporting_valid = torch::zeros(
        {1}, energy.options().dtype(torch::kBool)
    );
    reporting_generation_counts = torch::zeros(
        {48}, agent_ids_by_alive_order.options().dtype(torch::kInt64)
    );
    reporting_peak_injury = torch::where(
        alive_mask,
        injury.to(torch::kFloat64),
        torch::zeros(injury.sizes(), injury.options().dtype(torch::kFloat64))
    ).max().reshape({1}).clone();
    reporting_agent_ids = agent_ids_by_alive_order.clone();
    reporting_cohort_codes = torch::zeros(
        {reporting_row_count}, energy.options().dtype(torch::kInt32)
    );
    reporting_action_counts = torch::zeros(
        {reporting_row_count, 7},
        agent_ids_by_alive_order.options().dtype(torch::kInt64)
    );
    reporting_start_injury = injury.index_select(
        0, initial_slots.to(torch::kLong)
    ).clone();
    reporting_survival_steps = torch::zeros(
        {reporting_row_count}, survival_steps.options()
    );
    reporting_energy_collected_exact = torch::zeros(
        {reporting_row_count}, energy.options().dtype(torch::kFloat64)
    );
    reporting_offspring_count = torch::zeros(
        {reporting_row_count}, energy.options().dtype(torch::kInt32)
    );
    reporting_attacks = torch::zeros_like(reporting_offspring_count);
    reporting_alive_end = torch::zeros(
        {reporting_row_count}, energy.options().dtype(torch::kBool)
    );
    reporting_injury_end = torch::zeros(
        {reporting_row_count}, injury.options()
    );
    reporting_aggression_bias = torch::zeros_like(reporting_injury_end);
    reporting_early_survival_steps = torch::zeros_like(
        reporting_offspring_count
    );
    reporting_early_energy_collected_exact = torch::zeros_like(
        reporting_energy_collected_exact
    );
    reporting_early_injury_gained = torch::zeros_like(
        reporting_injury_end
    );
    reporting_early_life_weight = torch::ones_like(
        reporting_injury_end
    );
    reporting_action_switches = torch::zeros_like(
        reporting_offspring_count
    );
    reporting_action_oscillations = torch::zeros_like(
        reporting_offspring_count
    );
    reporting_drive_jitter = torch::zeros_like(reporting_injury_end);
    reporting_selfharm_events = torch::zeros_like(
        reporting_offspring_count
    );
    reporting_pred_err_sum = torch::zeros_like(reporting_injury_end);
    reporting_pred_err_count = torch::zeros_like(
        reporting_offspring_count
    );
    reporting_birth_reset_shock = torch::zeros_like(reporting_injury_end);
    reporting_episode0_pred_err_sum = torch::zeros_like(
        reporting_injury_end
    );
    reporting_episode0_pred_err_count = torch::zeros_like(
        reporting_offspring_count
    );
    reporting_episode0_selfharm_events = torch::zeros_like(
        reporting_offspring_count
    );
    reporting_late_pred_err_sum = torch::zeros_like(reporting_injury_end);
    reporting_late_pred_err_count = torch::zeros_like(
        reporting_offspring_count
    );
    reporting_action_terminal_survival = torch::zeros_like(
        reporting_survival_steps
    );
    reporting_action_terminal_rows = torch::zeros(
        {reporting_row_count}, energy.options().dtype(torch::kInt32)
    );
    reporting_movement_blocked_bounds = torch::zeros(
        {1}, agent_ids_by_alive_order.options().dtype(torch::kInt64)
    );
    reporting_event_tape_start_count = event_tape_active_count.clone();
    actions_by_tick_slot = torch::full(
        {steps, capacity},
        5,
        energy.options().dtype(torch::kInt32)
    );
    died_any = torch::zeros(
        {agent_ids_by_alive_order.size(0)},
        energy.options().dtype(torch::kBool)
    );
    food_gained_total = torch::zeros(
        {agent_ids_by_alive_order.size(0)},
        energy.options()
    );
    moved_count_total = torch::zeros(
        {agent_ids_by_alive_order.size(0)},
        energy.options().dtype(torch::kInt32)
    );
    action_metrics_float = torch::zeros({6}, energy.options());
    action_metrics_int = torch::zeros({2}, energy.options().dtype(torch::kInt32));
    applied_actions = torch::zeros(
        {1}, energy.options().dtype(torch::kInt64)
    );
    moved = torch::zeros_like(applied_actions);
    gathered = torch::zeros_like(applied_actions);
    policy_refill_calls = torch::zeros_like(applied_actions);
    policy_refill_rows = torch::zeros_like(applied_actions);
    policy_cache_hit_rows = torch::zeros_like(applied_actions);
    policy_cache_hit_calls = torch::zeros_like(applied_actions);
    policy_cache_rows_requested = torch::zeros_like(applied_actions);
    policy_cache_rows_missed = torch::zeros_like(applied_actions);
    policy_cache_rows_refilled = torch::zeros_like(applied_actions);
    policy_cache_full_refill_ticks = torch::zeros_like(applied_actions);
    policy_cache_partial_refill_ticks = torch::zeros_like(applied_actions);
    runtime_bookkeeping_by_tick = torch::zeros(
        {steps, 5}, energy.options().dtype(torch::kInt64)
    );
    policy_cache_status_by_tick = torch::zeros(
        {steps}, energy.options().dtype(torch::kInt32)
    );
    patch_stats = patch_enabled
        ? torch::zeros({steps, 25}, energy.options().dtype(torch::kFloat64))
        : torch::empty({0, 25}, energy.options().dtype(torch::kFloat64));
    food_spawn_stats = torch::zeros(
        {steps, 3}, energy.options().dtype(torch::kInt64)
    );
    lifecycle_death_agent_ids = torch::full(
        {steps, capacity},
        -1,
        agent_ids_by_alive_order.options()
    );
    lifecycle_death_slots = torch::full(
        {steps, capacity},
        -1,
        agent_ids_by_alive_order.options()
    );
    lifecycle_death_reason_codes = torch::zeros(
        {steps, capacity},
        energy.options().dtype(torch::kInt32)
    );
    lifecycle_death_counts = torch::zeros(
        {steps},
        agent_ids_by_alive_order.options()
    );
    lifecycle_alive_counts = torch::zeros(
        {steps},
        agent_ids_by_alive_order.options()
    );
    lifecycle_status = torch::zeros(
        {steps},
        energy.options().dtype(torch::kInt32)
    );
    active_mechanics_status = torch::zeros(
        {steps},
        energy.options().dtype(torch::kInt32)
    );
    active_mechanics_counts = torch::zeros(
        {steps, 5},
        energy.options().dtype(torch::kInt64)
    );
    drive_noise_rng_semantic_draws_by_tick = torch::zeros(
        {steps}, energy.options().dtype(torch::kInt64)
    );
    drive_noise_rng_status_by_tick = torch::zeros(
        {steps}, energy.options().dtype(torch::kInt32)
    );
    runner_shuffle_semantic_draws_by_tick = torch::zeros(
        {steps},
        energy.options().dtype(torch::kInt64)
    );
    runner_shuffle_status_by_tick = torch::zeros(
        {steps},
        energy.options().dtype(torch::kInt32)
    );
    chemistry_semantic_draws_by_tick = torch::zeros_like(
        runner_shuffle_semantic_draws_by_tick
    );
    chemistry_status_by_tick = torch::zeros_like(
        runner_shuffle_status_by_tick
    );
    event_append_status_by_tick = torch::zeros(
        {steps},
        energy.options().dtype(torch::kInt32)
    );
    birth_reason_counts_by_tick = torch::zeros(
        {steps, 10},
        agent_ids_by_alive_order.options().dtype(torch::kInt64)
    );
    birth_planned_counts_by_tick = torch::zeros(
        {steps},
        agent_ids_by_alive_order.options().dtype(torch::kInt64)
    );
    birth_decision_counts_by_tick = torch::zeros_like(birth_planned_counts_by_tick);
    birth_committed_counts_by_tick = torch::zeros_like(birth_planned_counts_by_tick);
    birth_status_by_tick = torch::zeros(
        {steps},
        energy.options().dtype(torch::kInt32)
    );
    residual_release_counts_by_tick = torch::zeros_like(birth_planned_counts_by_tick);
    residual_release_status_by_tick = torch::zeros_like(birth_status_by_tick);
    residual_pending_free_agent_ids = torch::full(
        {capacity},
        -1,
        agent_ids_by_alive_order.options().dtype(torch::kInt64)
    );
    residual_pending_free_slots = torch::full_like(residual_pending_free_agent_ids, -1);
    residual_pending_free_count = torch::zeros(
        {1},
        agent_ids_by_alive_order.options().dtype(torch::kInt64)
    );
    policy_pcg64_state_after_action_by_tick = torch::zeros(
        {steps, 4},
        pcg64_state_words.options()
    );
    policy_pcg64_buffer_after_action_by_tick = torch::zeros(
        {steps, 2},
        pcg64_buffer_words.options()
    );
    planner_pcg64_state_before_birth_by_tick = torch::zeros_like(
        policy_pcg64_state_after_action_by_tick
    );
    planner_pcg64_buffer_before_birth_by_tick = torch::zeros_like(
        policy_pcg64_buffer_after_action_by_tick
    );
    sticky_event_status = event_tape_status.clone();
    observations = torch::empty({0, 0}, energy.options());
    logits = torch::empty({0, action_count}, energy.options());
    next_rng_counter = torch::full(
        {1},
        rng_counter,
        energy.options().dtype(torch::kInt64)
    );
    generation_policy_np_semantic_draws = resident_lifecycle_state_enabled
        ? birth_policy_np_semantic_draws
        : torch::zeros({1}, energy.options().dtype(torch::kInt64));
    effective_food_spawn_count = static_cast<int64_t>(
        std::llround(food_spawn_per_step * food_spawn_mult)
    );
    TORCH_CHECK(
        max_food <= 0 || effective_food_spawn_count <= 0 || env_policy_alias,
        "resident food spawn requires the accepted env/policy RNG alias"
    );
    TORCH_CHECK(
        max_food <= 0 || effective_food_spawn_count <= 0
            || resident_lifecycle_state_enabled,
        "resident food spawn requires persistent policy RNG semantic state"
    );
    if (start_tick == 0) {
        action_cache_actions.fill_(-1);
        action_cache_token.fill_(-1);
        action_cache_cursor.fill_(-1);
        action_cache_next.fill_(-1);
        for (torch::Tensor* tensor : {
                 &action_switches,
                 &action_oscillations,
                 &drive_volatility,
                 &last_energy_delta,
                 &last_health_delta,
                 &last_injury_delta,
                 &selfharm_events,
             }) {
            tensor->copy_(torch::where(
                alive_mask,
                torch::zeros_like(*tensor),
                *tensor
            ));
        }
        if (resident_lifecycle_state_enabled) {
            for (torch::Tensor* tensor : {
                     &turnover_state.at(8),
                     &turnover_state.at(9),
                 }) {
                tensor->copy_(torch::where(
                    alive_mask,
                    torch::zeros_like(*tensor),
                    *tensor
                ));
            }
        }
    }

    }

    void execute_pre_policy_setup(const int64_t tick) {
        if (patch_enabled) {
            auto step_stats = eiel_cuda_architecture_b_patch_context_p2_v1(
                patch_food_amount,
                patch_fertility,
                patch_scent,
                patch_waste,
                patch_scent_emit_pending,
                patch_waste_emit_pending,
                agent_grid,
                patch_density,
                food_grid,
                1,
                patch_scent_decay_per_step,
                patch_waste_decay_per_step,
                patch_scent_presence_emit,
                patch_waste_presence_emit,
                patch_waste_density_emit_scale,
                patch_scent_max,
                patch_waste_max,
                patch_food_amount_max,
                patch_food_regrow_per_step,
                patch_food_regrow_use_fertility,
                patch_food_regrow_cap_by_fertility,
                patch_food_presence_threshold,
                patch_density_radius,
                patch_density_include_self,
                patch_density_normalize,
                patch_world_waste_consequence_enable,
                patch_waste_regrowth_penalty_enable,
                patch_waste_regrowth_penalty_scale,
                patch_waste_regrowth_penalty_max,
                patch_density_regrowth_penalty_enable,
                patch_density_regrowth_penalty_scale,
                patch_density_regrowth_penalty_max,
                patch_regrowth_min_multiplier
            );
            patch_stats.select(0, tick).copy_(step_stats.select(0, 0));
        }
        auto step_food_spawn_stats = eiel_cuda_architecture_b_flat_food_spawn_v1(
            food_grid,
            patch_food_amount,
            pcg64_state_words,
            pcg64_buffer_words,
            generation_policy_np_semantic_draws,
            max_food,
            effective_food_spawn_count,
            patch_enabled,
            patch_food_presence_threshold,
            patch_food_initial_value
        );
        food_spawn_stats.select(0, tick).copy_(step_food_spawn_stats);
        auto identity_pairs = compact_identity_pairs_by_agent_id(
            slot_to_agent_id, alive_mask
        );
        active_count.copy_(identity_pairs.active_count);
        auto active_rows = torch::arange(capacity, active_agent_ids.options());
        active_mask = active_rows < torch::clamp(active_count, 0, capacity);
        active_agent_ids.copy_(torch::where(
            active_mask,
            identity_pairs.agent_ids,
            torch::full_like(identity_pairs.agent_ids, -1)
        ));
        safe_agent_ids = torch::where(
            active_mask,
            active_agent_ids,
            torch::zeros_like(active_agent_ids)
        ).contiguous();
        slots = torch::where(
            active_mask,
            identity_pairs.slots,
            torch::zeros_like(identity_pairs.slots)
        ).contiguous();
        pre_action_count = active_count.clone();
        if (resident_lifecycle_state_enabled) {
            turnover_state.at(11).copy_(
                turnover_state.at(11).logical_or(alive_mask)
            );
        }
        applied_actions.add_(pre_action_count);
    }

    GenerationPrePolicyBatchView pre_policy_batch_view() const {
        return {
            active_agent_ids,
            active_count,
            slots,
        };
    }

    void accept_policy_cache_batch_outputs(
        const int64_t tick,
        const std::vector<torch::Tensor>& policy_cache
    ) {
        TORCH_CHECK(
            policy_cache.size() == 9,
            "Architecture B policy cache batch must return nine tensors"
        );
        policy_miss_agent_ids = policy_cache.at(0);
        policy_miss_slots = policy_cache.at(1);
        auto policy_requested_rows = policy_cache.at(2);
        auto policy_hit_rows = policy_cache.at(3);
        policy_miss_rows = policy_cache.at(4);
        policy_cache_rows_requested.add_(policy_requested_rows);
        policy_cache_hit_rows.add_(policy_hit_rows);
        policy_cache_rows_missed.add_(policy_miss_rows);
        policy_cache_rows_refilled.add_(policy_miss_rows);
        policy_refill_rows.add_(policy_miss_rows);
        policy_refill_calls.add_(
            policy_miss_rows.gt(0).to(policy_refill_calls.scalar_type())
        );
        policy_cache_full_refill_ticks.add_(policy_cache.at(5));
        policy_cache_partial_refill_ticks.add_(policy_cache.at(6));
        policy_cache_hit_calls.add_(policy_cache.at(7));
        policy_cache_status_by_tick.select(0, tick).copy_(
            policy_cache.at(8).select(0, 0)
        );
        runtime_bookkeeping_by_tick.select(0, tick).select(0, 0).copy_(
            pre_action_count.select(0, 0)
        );
        runtime_bookkeeping_by_tick.select(0, tick).select(0, 3).copy_(
            policy_miss_rows.gt(0).to(torch::kInt64).select(0, 0)
        );
        runtime_bookkeeping_by_tick.select(0, tick).select(0, 4).copy_(
            policy_miss_rows.select(0, 0)
        );
        chemistry_order_slots = slots.clone();
        if (active_mechanics_enabled) {
            auto shuffle_outputs = eiel_cuda_architecture_b_runner_shuffle_counted_v1(
                chemistry_order_slots,
                pre_action_count,
                runner_pcg64_state_words,
                runner_pcg64_buffer_words
            );
            runner_shuffle_semantic_draws_by_tick.select(0, tick).copy_(
                shuffle_outputs.at(3).select(0, 0)
            );
            runner_shuffle_status_by_tick.select(0, tick).copy_(
                shuffle_outputs.at(4).select(0, 0)
            );
            metrics_np_semantic_draws.add_(torch::where(
                shuffle_outputs.at(4).eq(0),
                shuffle_outputs.at(3),
                torch::zeros_like(shuffle_outputs.at(3))
            ));
        }
    }

    void accept_observations(torch::Tensor batch_observations) {
        TORCH_CHECK(
            batch_observations.is_cuda(),
            "Architecture B observations must be CUDA resident"
        );
        observations = std::move(batch_observations);
    }

    void execute_pre_policy(const int64_t tick) {
        execute_pre_policy_setup(tick);
        auto policy_cache = eiel_cuda_architecture_b_policy_cache_service_v1(
            active_agent_ids,
            active_count,
            slots,
            slot_tokens,
            action_cache_actions,
            action_cache_token,
            action_cache_cursor,
            action_cache_next,
            cache_width
        );
        accept_policy_cache_batch_outputs(tick, policy_cache);
        accept_observations(eiel_cuda_owner_observation_frame_slotted_v1(
            policy_miss_agent_ids,
            policy_miss_slots,
            policy_miss_rows,
            pos_y,
            pos_x,
            agent_grid,
            food_grid,
            energy,
            health,
            patch_food_amount,
            patch_cover,
            patch_shelter,
            patch_scent,
            patch_waste,
            patch_movement_cost,
            patch_enabled,
            view_radius,
            max_energy,
            start_health,
            patch_food_amount_max,
            patch_scent_max,
            patch_waste_max,
            patch_density_radius,
            patch_density_include_self,
            patch_density_normalize,
            patch_movement_cost_min,
            patch_movement_cost_max
        ));
    }

    void execute_single_cell_policy(const int64_t tick) {
        auto policy_outputs = eiel_cuda_fixed_mlp_refill_counted_v1(
            observations,
            policy_miss_slots,
            policy_miss_rows,
            mlp_pack,
            drives,
            attack_bias,
            action_temperature,
            drive_logit_gain,
            slot_tokens,
            action_cache_actions,
            action_cache_token,
            action_cache_cursor,
            action_cache_next,
            pcg64_state_words,
            pcg64_buffer_words,
            hidden_size,
            action_count,
            cache_width,
            deterministic,
            rng_seed,
            rng_counter,
            w_hunger_gather,
            w_pain_rest,
            w_fatigue_rest,
            w_pain_attack,
            w_fear_attack
        );
        accept_policy_batch_outputs(tick, policy_outputs);
    }

    void execute_single_cell_policy_with_residual(
        const int64_t tick,
        torch::Tensor logit_residuals
    ) {
        auto policy_outputs =
            eiel_cuda_fixed_mlp_refill_counted_with_residual_v1(
                observations,
                policy_miss_slots,
                policy_miss_rows,
                std::move(logit_residuals),
                mlp_pack,
                drives,
                attack_bias,
                action_temperature,
                drive_logit_gain,
                slot_tokens,
                action_cache_actions,
                action_cache_token,
                action_cache_cursor,
                action_cache_next,
                pcg64_state_words,
                pcg64_buffer_words,
                hidden_size,
                action_count,
                cache_width,
                deterministic,
                rng_seed,
                rng_counter,
                w_hunger_gather,
                w_pain_rest,
                w_fatigue_rest,
                w_pain_attack,
                w_fear_attack
            );
        accept_policy_batch_outputs(tick, policy_outputs);
    }

    void execute_single_cell_modular_policy(
        const int64_t tick,
        const std::vector<torch::Tensor>& modular_planes
    ) {
        TORCH_CHECK(
            modular_planes.size()
                == eiel::cuda_architecture_b::kModularTickPlaneCount,
            "Phase E generation requires exactly 57 modular planes"
        );
        TORCH_CHECK(
            view_radius == 2,
            "Phase E modular observations require the frozen radius-2 width"
        );
        at::_assert_async(
            torch::all(torch::eq(modular_planes.at(0), alive_mask)),
            "Phase E modular occupancy disagrees with generation alive_mask"
        );
        at::_assert_async(
            torch::all(torch::eq(modular_planes.at(1), slot_to_agent_id)),
            "Phase E modular identity disagrees with generation slot ownership"
        );
        at::_assert_async(
            torch::all(torch::logical_or(
                torch::logical_not(alive_mask),
                torch::eq(modular_planes.at(2), slot_tokens)
            )),
            "Phase E modular slot tokens disagree with generation ownership"
        );
        at::_assert_async(
            torch::all(torch::eq(policy_miss_rows, active_count)),
            "Phase E cache-width-one lane requires every active row to miss"
        );

        auto active_observations = eiel_cuda_owner_observation_frame_slotted_v1(
            active_agent_ids,
            slots,
            active_count,
            pos_y,
            pos_x,
            agent_grid,
            food_grid,
            energy,
            health,
            patch_food_amount,
            patch_cover,
            patch_shelter,
            patch_scent,
            patch_waste,
            patch_movement_cost,
            patch_enabled,
            view_radius,
            max_energy,
            start_health,
            patch_food_amount_max,
            patch_scent_max,
            patch_waste_max,
            patch_density_radius,
            patch_density_include_self,
            patch_density_normalize,
            patch_movement_cost_min,
            patch_movement_cost_max
        );
        torch::Tensor modular_observations;
        if (patch_enabled) {
            TORCH_CHECK(
                active_observations.dim() == 2
                    && active_observations.size(1) >= 27,
                "Phase E patch observation surface is too narrow"
            );
            modular_observations = torch::cat(
                {
                    active_observations.slice(1, 0, 25),
                    active_observations.slice(
                        1,
                        active_observations.size(1) - 2,
                        active_observations.size(1)
                    ),
                },
                1
            ).contiguous();
        } else {
            TORCH_CHECK(
                active_observations.dim() == 2
                    && active_observations.size(1) == 27,
                "Phase E flat observation surface must be [capacity,27]"
            );
            modular_observations = active_observations;
        }

        auto slot_observations =
            eiel::cuda_architecture_b::scatter_active_rows_by_slot_v1(
                modular_observations,
                slots.to(torch::kInt64).contiguous(),
                active_mask.contiguous(),
                capacity
            );
        auto fatigue = drives.select(1, 2).contiguous();
        auto stage_indices = torch::where(
            survival_steps >= kPhaseEDevelopmentLateStartStep,
            torch::full_like(survival_steps, 2),
            torch::where(
                survival_steps >= kPhaseEDevelopmentMatureStartStep,
                torch::ones_like(survival_steps),
                torch::zeros_like(survival_steps)
            )
        ).to(torch::kLong).contiguous();
        auto modular_outputs = eiel::cuda_architecture_b::modular_tick_v1(
            modular_planes,
            slot_observations,
            fatigue,
            stage_indices,
            action_count
        );
        TORCH_CHECK(
            !modular_outputs.empty(),
            "Phase E native modular tick returned no residual"
        );
        auto compact_residuals = modular_outputs.at(0).index_select(
            0,
            policy_miss_slots.to(torch::kLong)
        ).contiguous();
        observations = std::move(modular_observations);
        execute_single_cell_policy_with_residual(
            tick,
            std::move(compact_residuals)
        );
    }

    GenerationPolicyBatchView policy_batch_view() const {
        return {
            observations,
            policy_miss_slots,
            policy_miss_rows,
            survival_steps,
        };
    }

    void accept_policy_batch_outputs(
        const int64_t tick,
        const std::vector<torch::Tensor>& policy_outputs
    ) {
        TORCH_CHECK(
            policy_outputs.size() == 9,
            "Architecture B policy batch must return nine tensors"
        );
        if (!deterministic) {
            generation_policy_np_semantic_draws.add_(
                policy_miss_rows * (cache_width * action_count)
            );
        }
        logits = policy_outputs.at(0);
        auto row_actions = policy_outputs.at(1);
        next_rng_counter = policy_outputs.at(6);
        policy_pcg64_state_after_action_by_tick.select(0, tick).copy_(
            policy_outputs.at(7)
        );
        policy_pcg64_buffer_after_action_by_tick.select(0, tick).copy_(
            policy_outputs.at(8)
        );

    }

    void execute_post_policy_before_action_lifecycle(const int64_t tick) {
        actions_by_slot = torch::full(
            {capacity},
            5,
            energy.options().dtype(torch::kInt32)
        );
        actions_by_slot.copy_(torch::where(
            alive_mask,
            action_cache_next,
            actions_by_slot
        ));
        actions_by_tick_slot.select(0, tick).copy_(actions_by_slot);
        auto reporting_rows = torch::arange(
            reporting_row_count,
            reporting_agent_ids.options().dtype(torch::kInt64)
        );
        auto initial_rows = lookup_sorted_values(
            reporting_agent_ids, reporting_rows, safe_agent_ids
        ).to(torch::kLong);
        auto valid_initial_rows = active_mask.logical_and(initial_rows.ge(0));
        auto safe_initial_rows = torch::where(
            valid_initial_rows,
            initial_rows,
            torch::zeros_like(initial_rows)
        );
        auto active_actions = actions_by_slot.index_select(
            0, slots.to(torch::kLong)
        );
        for (int64_t action_index = 0; action_index < 7; ++action_index) {
            reporting_action_counts.select(1, action_index).index_add_(
                0,
                safe_initial_rows,
                valid_initial_rows.logical_and(
                    active_actions.eq(action_index)
                ).to(torch::kInt64)
            );
        }
        auto active_y = pos_y.index_select(0, slots.to(torch::kLong));
        auto active_x = pos_x.index_select(0, slots.to(torch::kLong));
        auto target_y = torch::where(
            active_actions.eq(0),
            active_y - 1,
            torch::where(active_actions.eq(1), active_y + 1, active_y)
        );
        auto target_x = torch::where(
            active_actions.eq(2),
            active_x + 1,
            torch::where(active_actions.eq(3), active_x - 1, active_x)
        );
        auto movement_rows = valid_initial_rows
            .logical_and(active_actions.ge(0))
            .logical_and(active_actions.le(3));
        reporting_movement_blocked_bounds.add_(
            movement_rows.logical_and(
                target_y.lt(0)
                    .logical_or(target_y.ge(agent_grid.size(0)))
                    .logical_or(target_x.lt(0))
                    .logical_or(target_x.ge(agent_grid.size(1)))
            ).to(torch::kInt64).sum().reshape({1})
        );
        auto old_last_action = last_action.clone();
        auto old_last_last_action = last_last_action.clone();
        auto action_switch_mask = alive_mask
            .logical_and(old_last_action.ne(-1))
            .logical_and(actions_by_slot.ne(old_last_action));
        auto action_oscillation_mask = alive_mask
            .logical_and(old_last_last_action.ne(-1))
            .logical_and(actions_by_slot.eq(old_last_last_action))
            .logical_and(actions_by_slot.ne(old_last_action));
        action_switches.add_(action_switch_mask.to(action_switches.scalar_type()));
        action_oscillations.add_(
            action_oscillation_mask.to(action_oscillations.scalar_type())
        );
        auto episode0_mask = alive_mask.logical_and(survival_steps.lt(episode0_k));
        episode0_switches.add_(
            action_switch_mask.logical_and(episode0_mask).to(
                episode0_switches.scalar_type()
            )
        );
        episode0_oscillations.add_(
            action_oscillation_mask.logical_and(episode0_mask).to(
                episode0_oscillations.scalar_type()
            )
        );
        last_last_action.copy_(torch::where(
            alive_mask,
            old_last_action,
            last_last_action
        ));
        last_action.copy_(torch::where(
            alive_mask,
            actions_by_slot,
            last_action
        ));
        pre_energy = energy.clone();
        pre_health = health.clone();
        pre_injury = injury.clone();
        pre_survival_steps = survival_steps.clone();
        auto action_outputs = eiel_cuda_scripted_owner_action_generation_slotted_v1(
            active_agent_ids,
            slots,
            active_count,
            actions_by_slot.unsqueeze(0).contiguous(),
            pos_y,
            pos_x,
            agent_grid,
            food_grid,
            patch_food_amount,
            patch_scent_emit_pending,
            patch_waste_emit_pending,
            patch_movement_cost,
            energy,
            health,
            patch_enabled,
            move_cost,
            gather_cost,
            rest_cost,
            attack_cost,
            action_cost_mult,
            rest_cost_mult,
            max_energy,
            food_energy,
            patch_food_gather_amount,
            patch_food_presence_threshold,
            patch_scent_move_emit,
            patch_waste_move_emit,
            patch_scent_gather_emit,
            patch_waste_consume_emit,
            patch_movement_cost_apply_enable,
            patch_movement_cost_energy_scale
        );
        auto died_accumulator = died_any.to(torch::kInt32);
        died_accumulator.index_add_(
            0,
            safe_initial_rows,
            action_outputs.at(1).logical_and(valid_initial_rows).to(torch::kInt32)
        );
        died_any.copy_(died_accumulator.gt(0));
        food_gained_total.index_add_(
            0,
            safe_initial_rows,
            action_outputs.at(2) * valid_initial_rows.to(action_outputs.at(2).dtype())
        );
        moved_count_total.index_add_(
            0,
            safe_initial_rows,
            action_outputs.at(3) * valid_initial_rows.to(action_outputs.at(3).dtype())
        );
        action_metrics_float.add_(action_outputs.at(4));
        action_metrics_int.add_(action_outputs.at(5));
        auto tick_moved = action_outputs.at(3).to(torch::kInt64).sum().reshape({1});
        auto tick_gathered = action_outputs.at(2).gt(0).to(torch::kInt64)
            .sum().reshape({1});
        moved.add_(tick_moved);
        gathered.add_(tick_gathered);
        runtime_bookkeeping_by_tick.select(0, tick).select(0, 1).copy_(
            tick_moved.select(0, 0)
        );
        runtime_bookkeeping_by_tick.select(0, tick).select(0, 2).copy_(
            tick_gathered.select(0, 0)
        );
        reporting_peak_injury.copy_(torch::maximum(
            reporting_peak_injury,
            torch::where(
                alive_mask,
                injury.to(torch::kFloat64),
                torch::zeros(
                    injury.sizes(), injury.options().dtype(torch::kFloat64)
                )
            ).max().reshape({1})
        ));
        if (resident_lifecycle_state_enabled) {
            auto food_by_slot = torch::zeros_like(turnover_state.at(0));
            food_by_slot.index_add_(
                0,
                slots.to(torch::kLong),
                action_outputs.at(2).to(food_by_slot.scalar_type())
                    * active_mask.to(food_by_slot.scalar_type())
            );
            turnover_state.at(1).add_(food_by_slot.to(torch::kFloat64));
            turnover_state.at(0).copy_(turnover_state.at(1).to(torch::kFloat32));
            if (early_life_steps > 0) {
                auto early_rows = active_mask.logical_and(
                    pre_survival_steps.index_select(0, slots.to(torch::kLong))
                        .lt(early_life_steps)
                );
                auto early_food_by_slot = torch::zeros_like(turnover_state.at(4));
                early_food_by_slot.index_add_(
                    0,
                    slots.to(torch::kLong),
                    action_outputs.at(2).to(early_food_by_slot.scalar_type())
                        * early_rows.to(early_food_by_slot.scalar_type())
                );
                turnover_state.at(5).add_(
                    early_food_by_slot.to(torch::kFloat64)
                );
                turnover_state.at(4).copy_(
                    turnover_state.at(5).to(torch::kFloat32)
                );
            }
        }

    }

    void accept_action_lifecycle_outputs(
        const int64_t tick,
        const std::vector<torch::Tensor>& outputs
    ) {
        TORCH_CHECK(
            outputs.size() == 7,
            "Architecture B action lifecycle batch must return seven tensors"
        );
        action_lifecycle = outputs;
        auto action_death_positions = torch::arange(
            capacity, action_lifecycle.at(0).options().dtype(torch::kInt64)
        );
        auto action_death_valid = action_death_positions.lt(
            action_lifecycle.at(3)
        );
        auto safe_action_death_ids = torch::where(
            action_death_valid,
            action_lifecycle.at(0),
            torch::zeros_like(action_lifecycle.at(0))
        );
        auto reporting_rows = torch::arange(
            reporting_row_count,
            reporting_agent_ids.options().dtype(torch::kInt64)
        );
        auto action_death_initial_rows = lookup_sorted_values(
            reporting_agent_ids, reporting_rows, safe_action_death_ids
        ).to(torch::kLong);
        action_death_valid = action_death_valid.logical_and(
            action_death_initial_rows.ge(0)
        );
        auto safe_action_death_initial_rows = torch::where(
            action_death_valid,
            action_death_initial_rows,
            torch::zeros_like(action_death_initial_rows)
        );
        auto safe_action_death_slots = torch::where(
            action_death_valid,
            action_lifecycle.at(1),
            torch::zeros_like(action_lifecycle.at(1))
        ).to(torch::kLong);
        reporting_action_terminal_survival.index_add_(
            0,
            safe_action_death_initial_rows,
            (survival_steps.index_select(0, safe_action_death_slots) + 1)
                * action_death_valid.to(survival_steps.scalar_type())
        );
        reporting_action_terminal_rows.index_add_(
            0,
            safe_action_death_initial_rows,
            action_death_valid.to(torch::kInt32)
        );
        event_generation = torch::full_like(action_lifecycle.at(3), generation);
        event_tick = torch::full_like(action_lifecycle.at(3), start_tick + tick);
        auto action_event_append = eiel_cuda_architecture_b_death_lifecycle_events_append_v1(
            event_tape_columns,
            event_tape_active_count,
            sticky_event_status,
            action_lifecycle.at(0),
            action_lifecycle.at(1),
            action_lifecycle.at(2),
            action_lifecycle.at(3),
            event_generation,
            event_tick,
            lineage_founder_id,
            lineage_depth
        );
        sticky_event_status = torch::maximum(sticky_event_status, action_event_append.at(1));
        active_agent_ids.copy_(action_lifecycle.at(4));
        active_count.copy_(action_lifecycle.at(5));
    }

    GenerationActiveMechanicsBatchView active_mechanics_batch_view() const {
        return {
            actions_by_slot,
            pre_energy,
            pre_health,
            pre_injury,
        };
    }

    void execute_single_cell_active_mechanics(const int64_t tick) {
        std::vector<torch::Tensor> mechanics_outputs;
        if (active_mechanics_enabled) {
            mechanics_outputs = eiel_cuda_architecture_b_active_mechanics_v1(
                slot_to_agent_id,
                alive_mask,
                actions_by_slot,
                energy,
                health,
                injury,
                age,
                policy_energy_observation,
                rest_streak,
                survival_steps,
                gru_upkeep_cost,
                cost_charge_scalar,
                cost_charge_maturation_scalar,
                drives,
                drive_inertia,
                drive_noise,
                prev_drive_delta,
                drive_volatility,
                drive_jitter,
                episode0_drive_jitter_sum,
                last_energy_delta,
                last_health_delta,
                last_injury_delta,
                selfharm_events,
                episode0_selfharm_events,
                pre_energy,
                pre_health,
                pre_injury,
                drive_noise_samples_by_tick_slot.select(
                    0,
                    start_tick + tick
                ).contiguous(),
                drive_noise_samples_valid_by_tick_slot.select(
                    0,
                    start_tick + tick
                ).contiguous(),
                enable_rest_charge,
                rest_streak_free,
                rest_charge_energy,
                rest_charge_growth,
                rest_deficit_to_health,
                developmental_maturity_age,
                episode0_k,
                max_energy,
                start_health,
                drive_hunger_gain,
                drive_hunger_decay,
                drive_pain_gain,
                drive_pain_decay,
                drive_fatigue_gain,
                drive_fatigue_rest_relief,
                drive_fatigue_decay,
                drive_fear_gain,
                drive_fear_decay
            );
        }
        accept_active_mechanics_batch_outputs(tick, mechanics_outputs);
    }

    void accept_active_mechanics_batch_outputs(
        const int64_t tick,
        const std::vector<torch::Tensor>& outputs
    ) {
        if (active_mechanics_enabled) {
            TORCH_CHECK(
                outputs.size() == 2,
                "Architecture B active mechanics batch must return two tensors"
            );
            const auto& mechanics_outputs = outputs;
            active_mechanics_status.select(0, tick).copy_(
                mechanics_outputs.at(0).reshape({-1}).select(0, 0)
            );
            active_mechanics_counts.select(0, tick).copy_(mechanics_outputs.at(1));

            auto drive_noise_rng_outputs =
                eiel_cuda_architecture_b_advance_precomputed_drive_noise_rng_v1(
                    chemistry_order_slots,
                    pre_action_count,
                    drive_noise,
                    runner_pcg64_state_words,
                    runner_pcg64_buffer_words
                );
            drive_noise_rng_semantic_draws_by_tick.select(0, tick).copy_(
                drive_noise_rng_outputs.at(2).select(0, 0)
            );
            drive_noise_rng_status_by_tick.select(0, tick).copy_(
                drive_noise_rng_outputs.at(3).select(0, 0)
            );
            metrics_np_semantic_draws.add_(torch::where(
                drive_noise_rng_outputs.at(3).eq(0),
                drive_noise_rng_outputs.at(2),
                torch::zeros_like(drive_noise_rng_outputs.at(2))
            ));

            auto chemistry_outputs = eiel_cuda_architecture_b_chemistry_update_counted_v1(
                chemistry,
                energy,
                health,
                injury,
                drives,
                alive_mask,
                chemistry_order_slots,
                pre_action_count,
                pcg64_state_words,
                pcg64_buffer_words,
                generation_policy_np_semantic_draws,
                max_energy,
                start_health
            );
            chemistry_semantic_draws_by_tick.select(0, tick).copy_(
                chemistry_outputs.at(4).select(0, 0)
            );
            chemistry_status_by_tick.select(0, tick).copy_(
                chemistry_outputs.at(5).select(0, 0)
            );
            sticky_event_status = torch::maximum(
                sticky_event_status,
                torch::maximum(
                    runner_shuffle_status_by_tick.select(0, tick).reshape({1}),
                    torch::maximum(
                        drive_noise_rng_outputs.at(3),
                        chemistry_outputs.at(5)
                    )
                )
            );
            if (resident_lifecycle_state_enabled) {
                auto early_alive = alive_mask.logical_and(
                    pre_survival_steps.lt(early_life_steps)
                );
                if (early_life_steps > 0) {
                    turnover_state.at(3).add_(
                        early_alive.to(turnover_state.at(3).scalar_type())
                    );
                    turnover_state.at(6).add_(torch::where(
                        early_alive,
                        torch::clamp_min(last_injury_delta, 0.0),
                        torch::zeros_like(turnover_state.at(6))
                    ));
                }
                turnover_state.at(7).copy_(torch::where(
                    alive_mask,
                    torch::full_like(turnover_state.at(7), early_life_weight),
                    turnover_state.at(7)
                ));
            }
        }
        reporting_peak_injury.copy_(torch::maximum(
            reporting_peak_injury,
            torch::where(
                alive_mask,
                injury.to(torch::kFloat64),
                torch::zeros(
                    injury.sizes(), injury.options().dtype(torch::kFloat64)
                )
            ).max().reshape({1})
        ));

        residual_pending_free_agent_ids.fill_(-1);
        residual_pending_free_slots.fill_(-1);
        residual_pending_free_count.zero_();
    }

    GenerationResidualLifecycleBatchView residual_lifecycle_batch_view() const {
        return {
            residual_pending_free_agent_ids,
            residual_pending_free_slots,
            residual_pending_free_count,
        };
    }

    void accept_residual_lifecycle_batch_state(
        torch::Tensor batch_pending_free_agent_ids,
        torch::Tensor batch_pending_free_slots,
        torch::Tensor batch_pending_free_count
    ) {
        residual_pending_free_agent_ids = std::move(
            batch_pending_free_agent_ids
        );
        residual_pending_free_slots = std::move(batch_pending_free_slots);
        residual_pending_free_count = std::move(batch_pending_free_count);
    }

    void accept_residual_lifecycle_outputs(
        const int64_t tick,
        const std::vector<torch::Tensor>& outputs
    ) {
        TORCH_CHECK(
            outputs.size() == 7,
            "Architecture B residual lifecycle batch must return seven tensors"
        );
        residual_lifecycle = outputs;
        auto residual_event_append = eiel_cuda_architecture_b_death_lifecycle_events_append_v1(
            event_tape_columns,
            event_tape_active_count,
            sticky_event_status,
            residual_lifecycle.at(0),
            residual_lifecycle.at(1),
            residual_lifecycle.at(2),
            residual_lifecycle.at(3),
            event_generation,
            event_tick,
            lineage_founder_id,
            lineage_depth
        );
        sticky_event_status = torch::maximum(sticky_event_status, residual_event_append.at(1));
        active_agent_ids.copy_(residual_lifecycle.at(4));
        active_count.copy_(residual_lifecycle.at(5));
        if (resident_lifecycle_state_enabled && enable_combat) {
            turnover_state.at(2).add_(
                alive_mask.logical_and(actions_by_slot.eq(6)).to(
                    turnover_state.at(2).scalar_type()
                )
            );
        }

        auto tick_release_status = torch::zeros(
            {1},
            energy.options().dtype(torch::kInt32)
        );
        if (resident_lifecycle_state_enabled) {
            auto residual_release_cursor = torch::zeros_like(residual_pending_free_count);
            auto release_outputs = eiel_cuda_architecture_b_release_pending_free_slots_v1(
                slot_to_agent_id,
                alive_mask,
                birth_identity_state.at(26),
                birth_identity_state.at(27),
                residual_pending_free_slots,
                residual_pending_free_count,
                residual_release_cursor
            );
            residual_release_counts_by_tick.select(0, tick).copy_(
                release_outputs.at(2).select(0, 0)
            );
            residual_release_status_by_tick.select(0, tick).copy_(
                release_outputs.at(3).select(0, 0)
            );
            tick_release_status.copy_(release_outputs.at(3));
        }
        if (birth_enabled) {
            planner_pcg64_state_before_birth_by_tick.select(0, tick).copy_(
                birth_planner_pcg64_state_words
            );
            planner_pcg64_buffer_before_birth_by_tick.select(0, tick).copy_(
                birth_planner_pcg64_buffer_words
            );
            auto birth_outputs = eiel_cuda_architecture_b_birth_microcycle_v1(
                birth_identity_state,
                birth_scalar_rng_state,
                birth_genome_int32_fields,
                birth_policy_mem_dim,
                birth_policy_use_gru,
                birth_genome_float32_fields,
                birth_drive_state,
                birth_policy_state,
                birth_planner_pcg64_state_words,
                birth_planner_pcg64_buffer_words,
                birth_policy_np_semantic_draws,
                runner_pcg64_state_words,
                runner_pcg64_buffer_words,
                metrics_np_semantic_draws,
                event_tape_columns,
                event_tape_active_count,
                sticky_event_status,
                tick_release_status,
                reproduce_parent_energy_share,
                reproduce_min_energy_buffer,
                start_health,
                reproduce_cooldown_steps,
                generation,
                start_tick + tick,
                genome_sensor_radius_max,
                max_energy,
                max_births_per_tick,
                injury_repro_k_injury,
                injury_repro_k_scar,
                injury_birth_fail_scar,
                injury_birth_fail_injury,
                true,
                reproduce_requires_mate,
                false,
                injury_repro_k_injury > 0.0 || injury_repro_k_scar > 0.0,
                injury_birth_fail_scar > 0.0 || injury_birth_fail_injury > 0.0
            );
            birth_reason_counts_by_tick.select(0, tick).copy_(birth_outputs.at(6));
            birth_planned_counts_by_tick.select(0, tick).copy_(
                birth_outputs.at(7).select(0, 0)
            );
            birth_decision_counts_by_tick.select(0, tick).copy_(
                birth_outputs.at(8).select(0, 0)
            );
            birth_status_by_tick.select(0, tick).copy_(
                birth_outputs.at(36).select(0, 0)
            );
            auto birth_ok = birth_outputs.at(36).eq(0).logical_and(
                tick_release_status.eq(0)
            );
            auto safe_birth_committed_count = torch::where(
                birth_ok,
                birth_outputs.at(15),
                torch::zeros_like(birth_outputs.at(15))
            );
            birth_committed_counts_by_tick.select(0, tick).copy_(
                safe_birth_committed_count.select(0, 0)
            );
            reset_newborn_slot_state(
                birth_outputs.at(14),
                safe_birth_committed_count,
                turnover_state,
                action_cache_actions,
                prev_drive_delta,
                drive_volatility,
                drive_jitter,
                episode0_drive_jitter_sum,
                last_energy_delta,
                last_health_delta,
                last_injury_delta,
                selfharm_events,
                episode0_selfharm_events
            );
            auto birth_rows = torch::arange(
                birth_outputs.at(0).numel(),
                birth_outputs.at(0).options().dtype(torch::kInt64)
            );
            auto committed_birth_rows = birth_rows.lt(safe_birth_committed_count);
            auto safe_parent_slots = torch::where(
                committed_birth_rows,
                birth_outputs.at(0),
                torch::zeros_like(birth_outputs.at(0))
            );
            auto parent_birth_counts = torch::zeros(
                {capacity},
                safe_parent_slots.options().dtype(torch::kInt32)
            );
            parent_birth_counts.index_add_(
                0,
                safe_parent_slots,
                committed_birth_rows.to(torch::kInt32)
            );
            auto first_repro_missing = parent_birth_counts.gt(0).logical_and(
                turnover_state.at(10).lt(0)
            );
            turnover_state.at(10).copy_(torch::where(
                first_repro_missing,
                survival_steps,
                turnover_state.at(10)
            ));
            active_agent_ids.copy_(torch::where(
                birth_ok,
                birth_outputs.at(16),
                residual_lifecycle.at(4)
            ));
            active_count.copy_(torch::where(
                birth_ok,
                birth_outputs.at(17),
                residual_lifecycle.at(5)
            ));
            sticky_event_status = torch::maximum(sticky_event_status, birth_outputs.at(35));
        }

        // Match the CPU owner's post-service boundary projection: cursor points
        // at the next cached action, while the current tick consumed index zero.
        auto projected_cache_index = torch::clamp(
            action_cache_cursor.to(torch::kLong),
            0,
            action_cache_actions.size(1) - 1
        ).unsqueeze(1);
        auto projected_cache_next = action_cache_actions.gather(
            1, projected_cache_index
        ).squeeze(1);
        auto valid_cache_rows = alive_mask
            .logical_and(action_cache_token.ge(0))
            .logical_and(action_cache_cursor.ge(0));
        action_cache_next.copy_(torch::where(
            valid_cache_rows,
            projected_cache_next,
            torch::full_like(action_cache_next, -1)
        ));

        auto combined_deaths = eiel_cuda_architecture_b_death_class_concat_v1(
            action_lifecycle.at(0),
            action_lifecycle.at(1),
            action_lifecycle.at(2),
            action_lifecycle.at(3),
            residual_lifecycle.at(0),
            residual_lifecycle.at(1),
            residual_lifecycle.at(2),
            residual_lifecycle.at(3)
        );
        lifecycle_death_agent_ids.select(0, tick).copy_(combined_deaths.at(0));
        lifecycle_death_slots.select(0, tick).copy_(combined_deaths.at(1));
        lifecycle_death_reason_codes.select(0, tick).copy_(combined_deaths.at(2));
        lifecycle_death_counts.select(0, tick).copy_(combined_deaths.at(3).select(0, 0));
        lifecycle_alive_counts.select(0, tick).copy_(active_count.select(0, 0));
        auto resident_tick_status = torch::maximum(
            torch::maximum(
                runner_shuffle_status_by_tick.select(0, tick).reshape({1}),
                torch::maximum(
                    drive_noise_rng_status_by_tick.select(0, tick).reshape({1}),
                    chemistry_status_by_tick.select(0, tick).reshape({1})
                )
            ),
            torch::maximum(
                active_mechanics_status.select(0, tick).reshape({1}),
                policy_cache_status_by_tick.select(0, tick).reshape({1})
            )
        );
        lifecycle_status.select(0, tick).copy_(
            torch::maximum(
                torch::maximum(
                    torch::maximum(
                        torch::maximum(action_lifecycle.at(6), residual_lifecycle.at(6)),
                        combined_deaths.at(4)
                    ),
                    residual_release_status_by_tick.select(0, tick).reshape({1})
                ),
                resident_tick_status
            ).select(0, 0)
        );
        event_append_status_by_tick.slice(0, tick, steps).copy_(
            sticky_event_status.expand({steps - tick})
        );
    }

    void execute_post_policy_impl(
        const int64_t tick,
        const std::vector<torch::Tensor>* modular_planes
    ) {
        execute_post_policy_before_action_lifecycle(tick);
        auto action_outputs = eiel_cuda_architecture_b_lifecycle_death_cull_v1(
            slot_to_agent_id,
            agent_id_to_slot,
            alive_mask,
            pos_y,
            pos_x,
            agent_grid,
            energy,
            health,
            policy_energy_observation,
            rest_streak,
            action_cache_token,
            action_cache_cursor,
            action_cache_next,
            pending_free_agent_ids,
            pending_free_slots,
            pending_free_count
        );
        if (modular_planes != nullptr) {
            eiel::cuda_architecture_b::clear_modular_rows_v1(
                *modular_planes,
                action_outputs.at(1),
                action_outputs.at(3)
            );
        }
        accept_action_lifecycle_outputs(tick, action_outputs);
        execute_single_cell_active_mechanics(tick);
        auto residual_outputs = eiel_cuda_architecture_b_lifecycle_death_cull_v1(
            slot_to_agent_id,
            agent_id_to_slot,
            alive_mask,
            pos_y,
            pos_x,
            agent_grid,
            energy,
            health,
            policy_energy_observation,
            rest_streak,
            action_cache_token,
            action_cache_cursor,
            action_cache_next,
            residual_pending_free_agent_ids,
            residual_pending_free_slots,
            residual_pending_free_count
        );
        if (modular_planes != nullptr) {
            eiel::cuda_architecture_b::clear_modular_rows_v1(
                *modular_planes,
                residual_outputs.at(1),
                residual_outputs.at(3)
            );
        }
        accept_residual_lifecycle_outputs(tick, residual_outputs);
    }

    void execute_post_policy(const int64_t tick) {
        execute_post_policy_impl(tick, nullptr);
    }

    void execute_post_policy_modular(
        const int64_t tick,
        const std::vector<torch::Tensor>& modular_planes
    ) {
        execute_post_policy_impl(tick, &modular_planes);
    }

    std::vector<torch::Tensor> finalize(
        const bool phase_e_no_turnover_reporting = false
    ) {
    event_tape_status.copy_(sticky_event_status);

    auto generation_gate_status = torch::maximum(
        torch::maximum(
            lifecycle_status.max().reshape({1}),
            birth_status_by_tick.max().reshape({1})
        ),
        sticky_event_status
    ).ne(0).to(torch::kInt32) * 8;

    // Freeze the reporting view before turnover mutates or clears cohort slots.
    // Phase E uses the same complete reporting surface without executing turnover.
    if (generation_turnover_enabled || phase_e_no_turnover_reporting) {
        auto reporting_slots = initial_slots.to(torch::kLong);
        auto final_identity_pairs = compact_identity_pairs_by_agent_id(
            slot_to_agent_id, alive_mask
        );
        auto reporting_final_slots = lookup_sorted_values(
            final_identity_pairs.agent_ids,
            final_identity_pairs.slots.to(torch::kInt64),
            reporting_agent_ids
        ).to(torch::kLong);
        auto reporting_final_slot_valid = reporting_final_slots.ge(0);
        auto reporting_safe_final_slots = torch::where(
            reporting_final_slot_valid,
            reporting_final_slots,
            torch::zeros_like(reporting_final_slots)
        );
        reporting_alive_end = reporting_final_slot_valid.logical_and(
            alive_mask.index_select(0, reporting_safe_final_slots)
        ).clone();

        auto gathered_survival_steps = survival_steps.index_select(
            0, reporting_slots
        );
        reporting_survival_steps = torch::where(
            reporting_action_terminal_rows.gt(0),
            reporting_action_terminal_survival,
            gathered_survival_steps
        ).clone();
        reporting_energy_collected_exact = turnover_state.at(1).index_select(
            0, reporting_slots
        ).clone();
        reporting_offspring_count = birth_identity_state.at(14).index_select(
            0, reporting_slots
        ).clone();
        reporting_attacks = turnover_state.at(2).index_select(
            0, reporting_slots
        ).clone();
        reporting_injury_end = injury.index_select(0, reporting_slots).clone();
        reporting_aggression_bias = birth_genome_float32_fields.at(1)
            .index_select(0, reporting_slots).clone();
        reporting_early_survival_steps = turnover_state.at(3).index_select(
            0, reporting_slots
        ).clone();
        reporting_early_energy_collected_exact = turnover_state.at(5)
            .index_select(0, reporting_slots).clone();
        reporting_early_injury_gained = turnover_state.at(6).index_select(
            0, reporting_slots
        ).clone();
        reporting_early_life_weight = turnover_state.at(7).index_select(
            0, reporting_slots
        ).clone();
        reporting_action_switches = action_switches.index_select(
            0, reporting_slots
        ).clone();
        reporting_action_oscillations = action_oscillations.index_select(
            0, reporting_slots
        ).clone();
        reporting_drive_jitter = drive_jitter.index_select(
            0, reporting_slots
        ).clone();
        reporting_selfharm_events = selfharm_events.index_select(
            0, reporting_slots
        ).clone();
        reporting_pred_err_sum = turnover_state.at(8).index_select(
            0, reporting_slots
        ).clone();
        reporting_pred_err_count = turnover_state.at(9).index_select(
            0, reporting_slots
        ).clone();
        reporting_birth_reset_shock = birth_drive_state.at(1).index_select(
            0, reporting_slots
        ).clone();
        reporting_episode0_pred_err_sum = turnover_state.at(12).index_select(
            0, reporting_slots
        ).clone();
        reporting_episode0_pred_err_count = turnover_state.at(13).index_select(
            0, reporting_slots
        ).clone();
        reporting_episode0_selfharm_events = episode0_selfharm_events
            .index_select(0, reporting_slots).clone();
        reporting_late_pred_err_sum = turnover_state.at(14).index_select(
            0, reporting_slots
        ).clone();
        reporting_late_pred_err_count = turnover_state.at(15).index_select(
            0, reporting_slots
        ).clone();

        auto birth_reasons = birth_reason_counts_by_tick.sum(0);
        auto committed_birth_rows = birth_committed_counts_by_tick.sum()
            .reshape({1}).to(torch::kInt64);
        auto failed_birth_rows = birth_reasons.slice(0, 1, 7).sum()
            .reshape({1}).to(torch::kInt64);
        auto public_failed_birth_rows = (
            birth_reasons.select(0, 2)
            + birth_reasons.select(0, 3)
            + birth_reasons.select(0, 5)
            + birth_reasons.select(0, 6)
        ).reshape({1}).to(torch::kInt64);
        auto birth_attempt_rows = (
            birth_reasons.select(0, 0)
            + birth_reasons.select(0, 3)
            + birth_reasons.select(0, 5)
            + birth_reasons.select(0, 6)
        ).reshape({1}).to(torch::kInt64);
        auto action_totals = reporting_action_counts.sum(0);
        auto action_move_total = action_totals.slice(0, 0, 4).sum()
            .reshape({1});
        auto movement_blocked_occupied = action_move_total
            - reporting_movement_blocked_bounds
            - moved;
        auto preturnover_alive_count = active_count.clone().to(torch::kInt64);
        auto reporting_capacity = torch::full(
            {1}, capacity, preturnover_alive_count.options()
        );
        auto reporting_typed_event_rows = (
            event_tape_active_count - reporting_event_tape_start_count
        ).to(torch::kInt64);
        auto policy_loop_tick_calls = torch::full(
            {1}, steps, applied_actions.options()
        );
        auto zero_count = torch::zeros_like(applied_actions);
        reporting_generation_counts = torch::cat({
            policy_loop_tick_calls,
            lifecycle_death_counts.sum().reshape({1}).to(torch::kInt64),
            committed_birth_rows,
            failed_birth_rows,
            public_failed_birth_rows,
            birth_attempt_rows,
            birth_reasons.select(0, 7).reshape({1}).to(torch::kInt64),
            applied_actions,
            action_move_total,
            action_totals.select(0, 0).reshape({1}),
            action_totals.select(0, 1).reshape({1}),
            action_totals.select(0, 2).reshape({1}),
            action_totals.select(0, 3).reshape({1}),
            action_totals.select(0, 4).reshape({1}),
            action_totals.select(0, 5).reshape({1}),
            action_totals.select(0, 6).reshape({1}),
            moved,
            gathered,
            action_move_total,
            movement_blocked_occupied,
            reporting_movement_blocked_bounds,
            birth_reasons.select(0, 8).reshape({1}).to(torch::kInt64),
            birth_reasons.select(0, 9).reshape({1}).to(torch::kInt64),
            birth_reasons.select(0, 7).reshape({1}).to(torch::kInt64),
            birth_reasons.select(0, 1).reshape({1}).to(torch::kInt64),
            birth_reasons.select(0, 4).reshape({1}).to(torch::kInt64),
            birth_reasons.select(0, 1).reshape({1}).to(torch::kInt64),
            birth_reasons.select(0, 4).reshape({1}).to(torch::kInt64),
            birth_reasons.select(0, 2).reshape({1}).to(torch::kInt64),
            birth_reasons.select(0, 7).reshape({1}).to(torch::kInt64),
            zero_count,
            birth_reasons.select(0, 5).reshape({1}).to(torch::kInt64),
            birth_reasons.select(0, 6).reshape({1}).to(torch::kInt64),
            preturnover_alive_count,
            reporting_capacity,
            reporting_capacity - preturnover_alive_count,
            reporting_typed_event_rows,
            policy_loop_tick_calls,
            policy_refill_calls,
            policy_refill_rows,
            policy_cache_hit_rows,
            policy_cache_hit_calls,
            policy_cache_rows_requested,
            policy_cache_hit_rows,
            policy_cache_rows_missed,
            policy_cache_rows_refilled,
            policy_cache_full_refill_ticks,
            policy_cache_partial_refill_ticks,
        }).clone();
    }

    auto turnover_ranked_agent_ids = torch::full(
        {capacity}, -1, active_agent_ids.options()
    );
    auto turnover_culled_agent_ids = torch::full_like(
        turnover_ranked_agent_ids, -1
    );
    auto turnover_parent_pool_agent_ids = torch::full_like(
        turnover_ranked_agent_ids, -1
    );
    auto turnover_refill_parent_agent_ids = torch::full_like(
        turnover_ranked_agent_ids, -1
    );
    auto turnover_refill_mate_agent_ids = torch::full_like(
        turnover_ranked_agent_ids, -1
    );
    auto turnover_child_agent_ids = torch::full_like(
        turnover_ranked_agent_ids, -1
    );
    auto turnover_child_slots = torch::full_like(
        turnover_ranked_agent_ids, -1
    );
    auto turnover_counts = torch::zeros(
        {8}, active_count.options()
    );
    auto turnover_status = torch::zeros(
        {1}, energy.options().dtype(torch::kInt32)
    );
    auto turnover_next_generation = torch::full(
        {1}, generation, active_count.options()
    );
    auto turnover_next_tick = torch::full(
        {1}, start_tick + steps, active_count.options()
    );

    if (generation_turnover_enabled) {
        auto turnover_plan = eiel_cuda_architecture_b_turnover_plan_v1(
            slot_to_agent_id,
            alive_mask,
            survival_steps,
            turnover_state.at(1),
            birth_identity_state.at(14),
            turnover_state.at(2),
            turnover_state.at(3),
            turnover_state.at(5),
            turnover_state.at(6),
            turnover_state.at(7),
            action_switches,
            action_oscillations,
            drive_volatility,
            selfharm_events,
            turnover_state.at(8),
            turnover_state.at(9),
            population_size,
            cull_fraction,
            parent_fraction,
            coherence_thrashing_weight,
            coherence_drive_volatility_weight,
            coherence_selfharm_weight,
            coherence_pred_err_weight
        );

        auto staged_identity = clone_tensor_vector(birth_identity_state);
        auto staged_scalar_rng = clone_tensor_vector(birth_scalar_rng_state);
        auto staged_genome_int32 = clone_tensor_vector(
            birth_genome_int32_fields
        );
        auto staged_policy_mem_dim = birth_policy_mem_dim.clone();
        auto staged_policy_use_gru = birth_policy_use_gru.clone();
        auto staged_genome_float32 = clone_tensor_vector(
            birth_genome_float32_fields
        );
        staged_genome_float32.at(16) = staged_identity.at(13);
        auto staged_drive = clone_tensor_vector(birth_drive_state);
        auto staged_turnover_policy = clone_tensor_vector(
            turnover_policy_state
        );
        auto staged_turnover_state = clone_tensor_vector(turnover_state);
        auto staged_action_cache_actions = action_cache_actions.clone();
        auto staged_prev_drive_delta = prev_drive_delta.clone();
        auto staged_drive_volatility = drive_volatility.clone();
        auto staged_drive_jitter = drive_jitter.clone();
        auto staged_episode0_drive_jitter_sum =
            episode0_drive_jitter_sum.clone();
        auto staged_last_energy_delta = last_energy_delta.clone();
        auto staged_last_health_delta = last_health_delta.clone();
        auto staged_last_injury_delta = last_injury_delta.clone();
        auto staged_selfharm_events = selfharm_events.clone();
        auto staged_episode0_selfharm_events =
            episode0_selfharm_events.clone();
        auto staged_pending_free_agent_ids = pending_free_agent_ids.clone();
        auto staged_pending_free_slots = pending_free_slots.clone();
        auto staged_pending_free_count = pending_free_count.clone();
        auto staged_planner_state = birth_planner_pcg64_state_words.clone();
        auto staged_planner_buffer = birth_planner_pcg64_buffer_words.clone();
        auto staged_policy_np_semantic_draws =
            birth_policy_np_semantic_draws.clone();
        auto refill_plan_python_words = staged_scalar_rng.at(0).clone();
        auto refill_plan_python_index = staged_scalar_rng.at(1).clone();
        auto refill_plan_python_raw_draws = staged_scalar_rng.at(6).clone();
        auto refill_plan_python_semantic_draws = staged_scalar_rng.at(7).clone();

        auto cull_outputs = eiel_cuda_architecture_b_turnover_cull_release_v1(
            staged_identity.at(0),
            staged_identity.at(1),
            staged_identity.at(3),
            staged_identity.at(4),
            staged_identity.at(5),
            staged_identity.at(6),
            staged_identity.at(7),
            staged_identity.at(8),
            staged_identity.at(20),
            staged_identity.at(23),
            staged_identity.at(24),
            staged_identity.at(25),
            staged_action_cache_actions,
            staged_identity.at(26),
            staged_identity.at(27),
            staged_pending_free_agent_ids,
            staged_pending_free_slots,
            staged_pending_free_count,
            turnover_plan.at(5),
            turnover_plan.at(4),
            turnover_plan.at(6),
            turnover_plan.at(13)
        );
        auto refill_gate_status = torch::maximum(
            turnover_plan.at(13), cull_outputs.at(6)
        );
        auto refill_plan = eiel_cuda_architecture_b_turnover_refill_plan_v1(
            staged_identity.at(0),
            staged_identity.at(3),
            staged_identity.at(6),
            turnover_plan.at(7),
            turnover_plan.at(9),
            turnover_plan.at(11),
            refill_gate_status,
            refill_plan_python_words,
            refill_plan_python_index,
            refill_plan_python_raw_draws,
            refill_plan_python_semantic_draws,
            staged_planner_state,
            staged_planner_buffer,
            staged_policy_np_semantic_draws,
            env_policy_alias
        );

        auto ordinary_policy_draws = torch::zeros(
            {1}, active_count.options()
        );
        if (birth_enabled) {
            auto ordinary_rows = torch::arange(
                birth_policy_state.at(6).numel(),
                birth_policy_state.at(6).options().dtype(torch::kInt64)
            );
            auto consumed_rows = ordinary_rows.lt(birth_policy_state.at(7));
            ordinary_policy_draws.copy_(torch::where(
                consumed_rows,
                birth_policy_state.at(6),
                torch::zeros_like(birth_policy_state.at(6))
            ).sum().reshape({1}));
        }
        auto policy_branch_status = ordinary_policy_draws.ne(0).to(
            torch::kInt32
        );
        auto pretransaction_status = torch::maximum(
            torch::maximum(
                torch::maximum(refill_gate_status, refill_plan.at(9)),
                policy_branch_status
            ),
            generation_gate_status
        );
        auto safe_refill_count = torch::where(
            pretransaction_status.eq(0),
            refill_plan.at(4),
            torch::zeros_like(refill_plan.at(4))
        );
        auto refill_transaction =
            eiel_cuda_architecture_b_fixed_mlp_turnover_refill_transaction_v1(
                staged_identity,
                turnover_plan.at(7),
                turnover_plan.at(9),
                refill_plan.at(2),
                safe_refill_count,
                staged_scalar_rng,
                staged_genome_int32,
                staged_policy_mem_dim,
                staged_policy_use_gru,
                staged_genome_float32,
                staged_drive,
                staged_turnover_policy,
                start_energy,
                start_health,
                generation,
                genome_sensor_radius_max,
                max_energy
            );
        reset_newborn_slot_state(
            refill_transaction.at(3),
            refill_transaction.at(4),
            staged_turnover_state,
            staged_action_cache_actions,
            staged_prev_drive_delta,
            staged_drive_volatility,
            staged_drive_jitter,
            staged_episode0_drive_jitter_sum,
            staged_last_energy_delta,
            staged_last_health_delta,
            staged_last_injury_delta,
            staged_selfharm_events,
            staged_episode0_selfharm_events
        );

        auto inactive_slots = staged_identity.at(3).logical_not();
        const auto normalize_inactive = [&inactive_slots](
            torch::Tensor& tensor,
            const c10::Scalar& value
        ) {
            auto mask = inactive_slots;
            for (int64_t dimension = 1; dimension < tensor.dim(); ++dimension) {
                mask = mask.unsqueeze(-1);
            }
            tensor.copy_(torch::where(
                mask.expand_as(tensor),
                torch::full_like(tensor, value),
                tensor
            ));
        };
        for (const int64_t index : {7, 8, 9, 10, 11, 12, 14, 16, 17, 18,
                                    19, 21, 22, 29, 30, 31, 32, 33, 34}) {
            normalize_inactive(staged_identity.at(index), c10::Scalar(0));
        }
        normalize_inactive(staged_identity.at(13), c10::Scalar(30.0));
        normalize_inactive(staged_identity.at(15), c10::Scalar(-1));
        for (const int64_t index : {23, 24, 25}) {
            normalize_inactive(staged_identity.at(index), c10::Scalar(-1));
        }
        normalize_inactive(staged_genome_int32.at(0), c10::Scalar(64));
        normalize_inactive(staged_genome_int32.at(1), c10::Scalar(0));
        normalize_inactive(staged_genome_int32.at(2), c10::Scalar(0));
        normalize_inactive(staged_genome_int32.at(3), c10::Scalar(1));
        for (const int64_t index : {4, 5, 6, 7}) {
            normalize_inactive(staged_genome_int32.at(index), c10::Scalar(0));
        }
        normalize_inactive(staged_policy_mem_dim, c10::Scalar(0));
        normalize_inactive(staged_policy_use_gru, c10::Scalar(false));
        for (const int64_t index : {0, 1, 2, 18}) {
            normalize_inactive(staged_genome_float32.at(index), c10::Scalar(0));
        }
        normalize_inactive(staged_genome_float32.at(3), c10::Scalar(30.0));
        for (const int64_t index : {4, 7, 19, 20}) {
            normalize_inactive(staged_genome_float32.at(index), c10::Scalar(1.0));
        }
        for (const int64_t index : {5, 21}) {
            normalize_inactive(staged_genome_float32.at(index), c10::Scalar(0.85));
        }
        for (const int64_t index : {6, 22}) {
            normalize_inactive(staged_genome_float32.at(index), c10::Scalar(0.02));
        }
        for (const int64_t index : {8, 23}) {
            normalize_inactive(staged_genome_float32.at(index), c10::Scalar(0.08));
        }
        normalize_inactive(staged_genome_float32.at(9), c10::Scalar(0.55));
        normalize_inactive(staged_genome_float32.at(10), c10::Scalar(0.05));
        normalize_inactive(staged_genome_float32.at(11), c10::Scalar(0.15));
        normalize_inactive(staged_genome_float32.at(12), c10::Scalar(0.10));
        normalize_inactive(staged_genome_float32.at(13), c10::Scalar(0.35));
        normalize_inactive(staged_genome_float32.at(14), c10::Scalar(0.05));
        normalize_inactive(staged_genome_float32.at(15), c10::Scalar(0.03));
        normalize_inactive(staged_genome_float32.at(16), c10::Scalar(30.0));
        normalize_inactive(staged_drive.at(0), c10::Scalar(0));
        normalize_inactive(staged_drive.at(1), c10::Scalar(0));
        normalize_inactive(staged_drive.at(5), c10::Scalar(0));
        normalize_inactive(staged_turnover_policy.at(1), c10::Scalar(0));
        for (const int64_t index : {0, 1, 2, 3, 4, 5, 6, 8, 9, 12, 13,
                                    14, 15, 16, 17}) {
            normalize_inactive(staged_turnover_state.at(index), c10::Scalar(0));
        }
        normalize_inactive(staged_turnover_state.at(7), c10::Scalar(1.0));
        normalize_inactive(staged_turnover_state.at(10), c10::Scalar(-1));
        normalize_inactive(staged_turnover_state.at(11), c10::Scalar(false));
        normalize_inactive(staged_action_cache_actions, c10::Scalar(-1));
        for (torch::Tensor* tensor : {
                 &staged_prev_drive_delta,
                 &staged_drive_volatility,
                 &staged_drive_jitter,
                 &staged_episode0_drive_jitter_sum,
                 &staged_last_energy_delta,
                 &staged_last_health_delta,
                 &staged_last_injury_delta,
                 &staged_selfharm_events,
                 &staged_episode0_selfharm_events,
             }) {
            normalize_inactive(*tensor, c10::Scalar(0));
        }

        auto final_live_count = staged_identity.at(3).to(torch::kInt64)
            .sum().reshape({1});
        auto fixed_live_ids = torch::where(
            staged_identity.at(3),
            staged_identity.at(0),
            torch::full_like(staged_identity.at(0),
                std::numeric_limits<int64_t>::max())
        );
        auto sorted_live_ids = std::get<0>(torch::sort(fixed_live_ids));
        auto live_rows = torch::arange(
            capacity, sorted_live_ids.options().dtype(torch::kInt64)
        );
        auto staged_active_agent_ids = torch::where(
            live_rows.lt(final_live_count),
            sorted_live_ids,
            torch::full_like(sorted_live_ids, -1)
        );
        auto count_matches = refill_transaction.at(4).eq(
            turnover_plan.at(11)
        );
        auto population_matches = final_live_count.eq(population_size);
        turnover_status.copy_(torch::maximum(
            pretransaction_status,
            refill_transaction.at(22)
        ));
        turnover_status.copy_(torch::where(
            turnover_status.eq(0).logical_and(count_matches).logical_and(
                population_matches
            ),
            turnover_status,
            torch::where(
                turnover_status.eq(0),
                torch::full_like(turnover_status, 7),
                turnover_status
            )
        ));
        auto turnover_success = turnover_status.eq(0);

        commit_tensor_vector(turnover_success, birth_identity_state, staged_identity);
        commit_tensor_vector(turnover_success, birth_scalar_rng_state, staged_scalar_rng);
        commit_tensor_vector(turnover_success, birth_genome_int32_fields, staged_genome_int32);
        birth_policy_mem_dim.copy_(torch::where(
            turnover_success, staged_policy_mem_dim, birth_policy_mem_dim
        ));
        birth_policy_use_gru.copy_(torch::where(
            turnover_success, staged_policy_use_gru, birth_policy_use_gru
        ));
        commit_tensor_vector(turnover_success, birth_genome_float32_fields, staged_genome_float32);
        commit_tensor_vector(turnover_success, birth_drive_state, staged_drive);
        commit_tensor_vector(turnover_success, turnover_policy_state, staged_turnover_policy);
        commit_tensor_vector(turnover_success, turnover_state, staged_turnover_state);
        action_cache_actions.copy_(torch::where(
            turnover_success, staged_action_cache_actions, action_cache_actions
        ));
        prev_drive_delta.copy_(torch::where(
            turnover_success, staged_prev_drive_delta, prev_drive_delta
        ));
        drive_volatility.copy_(torch::where(
            turnover_success, staged_drive_volatility, drive_volatility
        ));
        drive_jitter.copy_(torch::where(
            turnover_success, staged_drive_jitter, drive_jitter
        ));
        episode0_drive_jitter_sum.copy_(torch::where(
            turnover_success,
            staged_episode0_drive_jitter_sum,
            episode0_drive_jitter_sum
        ));
        last_energy_delta.copy_(torch::where(
            turnover_success, staged_last_energy_delta, last_energy_delta
        ));
        last_health_delta.copy_(torch::where(
            turnover_success, staged_last_health_delta, last_health_delta
        ));
        last_injury_delta.copy_(torch::where(
            turnover_success, staged_last_injury_delta, last_injury_delta
        ));
        selfharm_events.copy_(torch::where(
            turnover_success, staged_selfharm_events, selfharm_events
        ));
        episode0_selfharm_events.copy_(torch::where(
            turnover_success,
            staged_episode0_selfharm_events,
            episode0_selfharm_events
        ));
        pending_free_agent_ids.copy_(torch::where(
            turnover_success,
            staged_pending_free_agent_ids,
            pending_free_agent_ids
        ));
        pending_free_slots.copy_(torch::where(
            turnover_success, staged_pending_free_slots, pending_free_slots
        ));
        pending_free_count.copy_(torch::where(
            turnover_success, staged_pending_free_count, pending_free_count
        ));
        commit_tensor(
            turnover_success,
            birth_planner_pcg64_state_words,
            staged_planner_state
        );
        birth_planner_pcg64_buffer_words.copy_(torch::where(
            turnover_success,
            staged_planner_buffer,
            birth_planner_pcg64_buffer_words
        ));
        birth_policy_np_semantic_draws.copy_(torch::where(
            turnover_success,
            staged_policy_np_semantic_draws,
            birth_policy_np_semantic_draws
        ));
        active_agent_ids.copy_(torch::where(
            turnover_success, staged_active_agent_ids, active_agent_ids
        ));
        active_count.copy_(torch::where(
            turnover_success, final_live_count, active_count
        ));

        turnover_ranked_agent_ids.copy_(turnover_plan.at(1));
        turnover_culled_agent_ids.copy_(turnover_plan.at(5));
        turnover_parent_pool_agent_ids.copy_(turnover_plan.at(8));
        turnover_refill_parent_agent_ids.copy_(refill_transaction.at(0));
        turnover_refill_mate_agent_ids.copy_(refill_transaction.at(1));
        turnover_child_agent_ids.copy_(refill_transaction.at(2));
        turnover_child_slots.copy_(refill_transaction.at(3));
        turnover_counts.copy_(torch::cat({
            turnover_plan.at(3),
            turnover_plan.at(6),
            turnover_plan.at(9),
            turnover_plan.at(11),
            cull_outputs.at(2),
            refill_plan.at(4),
            refill_transaction.at(4),
            final_live_count,
        }));
        turnover_next_generation.copy_(torch::where(
            turnover_success,
            torch::full_like(turnover_next_generation, generation + 1),
            turnover_next_generation
        ));
        turnover_next_tick.copy_(torch::where(
            turnover_success,
            torch::zeros_like(turnover_next_tick),
            turnover_next_tick
        ));
    }
    if (generation_turnover_enabled || phase_e_no_turnover_reporting) {
        reporting_status.zero_();
        if (birth_enabled) {
            reporting_status.fill_(2);
        }
        if (start_tick != 0) {
            reporting_status.fill_(3);
        }
        if (action_count != 7) {
            reporting_status.fill_(7);
        }
        reporting_status.copy_(torch::where(
            generation_gate_status.ne(0),
            torch::full_like(reporting_status, 5),
            reporting_status
        ));
        reporting_status.copy_(torch::where(
            turnover_status.ne(0),
            torch::full_like(reporting_status, 6),
            reporting_status
        ));
    }
    reporting_valid = reporting_status.eq(0).clone();
    return {
        observations,
        logits,
        actions_by_tick_slot,
        energy,
        died_any,
        food_gained_total,
        moved_count_total,
        action_metrics_float,
        action_metrics_int,
        initial_slots,
        patch_stats,
        food_spawn_stats,
        next_rng_counter,
        pcg64_state_words,
        pcg64_buffer_words,
        active_agent_ids,
        lifecycle_death_agent_ids,
        lifecycle_death_slots,
        lifecycle_death_reason_codes,
        lifecycle_death_counts,
        lifecycle_alive_counts,
        lifecycle_status,
        active_mechanics_status,
        active_mechanics_counts,
        drive_noise_rng_semantic_draws_by_tick,
        drive_noise_rng_status_by_tick,
        event_tape_active_count,
        event_tape_status,
        event_append_status_by_tick,
        active_count,
        birth_reason_counts_by_tick,
        birth_planned_counts_by_tick,
        birth_decision_counts_by_tick,
        birth_committed_counts_by_tick,
        birth_status_by_tick,
        residual_release_counts_by_tick,
        residual_release_status_by_tick,
        policy_pcg64_state_after_action_by_tick,
        policy_pcg64_buffer_after_action_by_tick,
        planner_pcg64_state_before_birth_by_tick,
        planner_pcg64_buffer_before_birth_by_tick,
        runner_shuffle_semantic_draws_by_tick,
        runner_shuffle_status_by_tick,
        chemistry_semantic_draws_by_tick,
        chemistry_status_by_tick,
        metrics_np_semantic_draws,
        turnover_ranked_agent_ids,
        turnover_culled_agent_ids,
        turnover_parent_pool_agent_ids,
        turnover_refill_parent_agent_ids,
        turnover_refill_mate_agent_ids,
        turnover_child_agent_ids,
        turnover_child_slots,
        turnover_counts,
        turnover_status,
        turnover_next_generation,
        turnover_next_tick,
        applied_actions,
        moved,
        gathered,
        policy_refill_calls,
        policy_refill_rows,
        runtime_bookkeeping_by_tick,
        reporting_status,
        reporting_valid,
        reporting_generation_counts,
        reporting_peak_injury,
        reporting_agent_ids,
        reporting_cohort_codes,
        reporting_action_counts,
        reporting_start_injury,
        reporting_survival_steps,
        reporting_energy_collected_exact,
        reporting_offspring_count,
        reporting_attacks,
        reporting_alive_end,
        reporting_injury_end,
        reporting_aggression_bias,
        reporting_early_survival_steps,
        reporting_early_energy_collected_exact,
        reporting_early_injury_gained,
        reporting_early_life_weight,
        reporting_action_switches,
        reporting_action_oscillations,
        reporting_drive_jitter,
        reporting_selfharm_events,
        reporting_pred_err_sum,
        reporting_pred_err_count,
        reporting_birth_reset_shock,
        reporting_episode0_pred_err_sum,
        reporting_episode0_pred_err_count,
        reporting_episode0_selfharm_events,
        reporting_late_pred_err_sum,
        reporting_late_pred_err_count,
    };
    }

private:
    int64_t capacity = 0;
    int64_t reporting_row_count = 0;
    int64_t effective_food_spawn_count = 0;

    torch::Tensor initial_slots;
    torch::Tensor active_agent_ids;
    torch::Tensor active_count;
    torch::Tensor reporting_status;
    torch::Tensor reporting_valid;
    torch::Tensor reporting_generation_counts;
    torch::Tensor reporting_peak_injury;
    torch::Tensor reporting_agent_ids;
    torch::Tensor reporting_cohort_codes;
    torch::Tensor reporting_action_counts;
    torch::Tensor reporting_start_injury;
    torch::Tensor reporting_survival_steps;
    torch::Tensor reporting_energy_collected_exact;
    torch::Tensor reporting_offspring_count;
    torch::Tensor reporting_attacks;
    torch::Tensor reporting_alive_end;
    torch::Tensor reporting_injury_end;
    torch::Tensor reporting_aggression_bias;
    torch::Tensor reporting_early_survival_steps;
    torch::Tensor reporting_early_energy_collected_exact;
    torch::Tensor reporting_early_injury_gained;
    torch::Tensor reporting_early_life_weight;
    torch::Tensor reporting_action_switches;
    torch::Tensor reporting_action_oscillations;
    torch::Tensor reporting_drive_jitter;
    torch::Tensor reporting_selfharm_events;
    torch::Tensor reporting_pred_err_sum;
    torch::Tensor reporting_pred_err_count;
    torch::Tensor reporting_birth_reset_shock;
    torch::Tensor reporting_episode0_pred_err_sum;
    torch::Tensor reporting_episode0_pred_err_count;
    torch::Tensor reporting_episode0_selfharm_events;
    torch::Tensor reporting_late_pred_err_sum;
    torch::Tensor reporting_late_pred_err_count;
    torch::Tensor reporting_action_terminal_survival;
    torch::Tensor reporting_action_terminal_rows;
    torch::Tensor reporting_movement_blocked_bounds;
    torch::Tensor reporting_event_tape_start_count;
    torch::Tensor actions_by_tick_slot;
    torch::Tensor died_any;
    torch::Tensor food_gained_total;
    torch::Tensor moved_count_total;
    torch::Tensor action_metrics_float;
    torch::Tensor action_metrics_int;
    torch::Tensor applied_actions;
    torch::Tensor moved;
    torch::Tensor gathered;
    torch::Tensor policy_refill_calls;
    torch::Tensor policy_refill_rows;
    torch::Tensor policy_cache_hit_rows;
    torch::Tensor policy_cache_hit_calls;
    torch::Tensor policy_cache_rows_requested;
    torch::Tensor policy_cache_rows_missed;
    torch::Tensor policy_cache_rows_refilled;
    torch::Tensor policy_cache_full_refill_ticks;
    torch::Tensor policy_cache_partial_refill_ticks;
    torch::Tensor runtime_bookkeeping_by_tick;
    torch::Tensor policy_cache_status_by_tick;
    torch::Tensor patch_stats;
    torch::Tensor food_spawn_stats;
    torch::Tensor lifecycle_death_agent_ids;
    torch::Tensor lifecycle_death_slots;
    torch::Tensor lifecycle_death_reason_codes;
    torch::Tensor lifecycle_death_counts;
    torch::Tensor lifecycle_alive_counts;
    torch::Tensor lifecycle_status;
    torch::Tensor active_mechanics_status;
    torch::Tensor active_mechanics_counts;
    torch::Tensor drive_noise_rng_semantic_draws_by_tick;
    torch::Tensor drive_noise_rng_status_by_tick;
    torch::Tensor runner_shuffle_semantic_draws_by_tick;
    torch::Tensor runner_shuffle_status_by_tick;
    torch::Tensor chemistry_semantic_draws_by_tick;
    torch::Tensor chemistry_status_by_tick;
    torch::Tensor event_append_status_by_tick;
    torch::Tensor birth_reason_counts_by_tick;
    torch::Tensor birth_planned_counts_by_tick;
    torch::Tensor birth_decision_counts_by_tick;
    torch::Tensor birth_committed_counts_by_tick;
    torch::Tensor birth_status_by_tick;
    torch::Tensor residual_release_counts_by_tick;
    torch::Tensor residual_release_status_by_tick;
    torch::Tensor residual_pending_free_agent_ids;
    torch::Tensor residual_pending_free_slots;
    torch::Tensor residual_pending_free_count;
    torch::Tensor policy_pcg64_state_after_action_by_tick;
    torch::Tensor policy_pcg64_buffer_after_action_by_tick;
    torch::Tensor planner_pcg64_state_before_birth_by_tick;
    torch::Tensor planner_pcg64_buffer_before_birth_by_tick;
    torch::Tensor sticky_event_status;
    torch::Tensor observations;
    torch::Tensor logits;
    torch::Tensor next_rng_counter;
    torch::Tensor generation_policy_np_semantic_draws;

    torch::Tensor active_mask;
    torch::Tensor safe_agent_ids;
    torch::Tensor slots;
    torch::Tensor pre_action_count;
    torch::Tensor policy_miss_agent_ids;
    torch::Tensor policy_miss_slots;
    torch::Tensor policy_miss_rows;
    torch::Tensor chemistry_order_slots;
    torch::Tensor actions_by_slot;
    torch::Tensor pre_energy;
    torch::Tensor pre_health;
    torch::Tensor pre_injury;
    torch::Tensor pre_survival_steps;
    torch::Tensor event_generation;
    torch::Tensor event_tick;
    std::vector<torch::Tensor> action_lifecycle;
    std::vector<torch::Tensor> residual_lifecycle;
};

std::unique_ptr<GenerationContext> GenerationContext::prepare(
    GenerationContextArguments arguments
) {
    return std::unique_ptr<GenerationContext>(new GenerationContext(
        std::make_unique<Impl>(std::move(arguments))
    ));
}

GenerationContext::GenerationContext(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}

GenerationContext::~GenerationContext() = default;

void GenerationContext::execute_pre_policy(const int64_t tick) {
    impl_->execute_pre_policy(tick);
}

void GenerationContext::execute_pre_policy_setup(const int64_t tick) {
    impl_->execute_pre_policy_setup(tick);
}

GenerationPrePolicyBatchView GenerationContext::pre_policy_batch_view() const {
    return impl_->pre_policy_batch_view();
}

void GenerationContext::accept_policy_cache_batch_outputs(
    const int64_t tick,
    const std::vector<torch::Tensor>& outputs
) {
    impl_->accept_policy_cache_batch_outputs(tick, outputs);
}

void GenerationContext::accept_observations(torch::Tensor observations) {
    impl_->accept_observations(std::move(observations));
}

void GenerationContext::execute_single_cell_policy(const int64_t tick) {
    impl_->execute_single_cell_policy(tick);
}

void GenerationContext::execute_single_cell_policy_with_residual(
    const int64_t tick,
    torch::Tensor logit_residuals
) {
    impl_->execute_single_cell_policy_with_residual(
        tick,
        std::move(logit_residuals)
    );
}

void GenerationContext::execute_single_cell_modular_policy(
    const int64_t tick,
    const std::vector<torch::Tensor>& modular_planes
) {
    impl_->execute_single_cell_modular_policy(tick, modular_planes);
}

GenerationPolicyBatchView GenerationContext::policy_batch_view() const {
    return impl_->policy_batch_view();
}

void GenerationContext::accept_policy_batch_outputs(
    const int64_t tick,
    const std::vector<torch::Tensor>& outputs
) {
    impl_->accept_policy_batch_outputs(tick, outputs);
}

void GenerationContext::execute_post_policy_before_action_lifecycle(
    const int64_t tick
) {
    impl_->execute_post_policy_before_action_lifecycle(tick);
}

void GenerationContext::accept_action_lifecycle_outputs(
    const int64_t tick,
    const std::vector<torch::Tensor>& outputs
) {
    impl_->accept_action_lifecycle_outputs(tick, outputs);
}

GenerationActiveMechanicsBatchView
GenerationContext::active_mechanics_batch_view() const {
    return impl_->active_mechanics_batch_view();
}

void GenerationContext::execute_single_cell_active_mechanics(
    const int64_t tick
) {
    impl_->execute_single_cell_active_mechanics(tick);
}

void GenerationContext::accept_active_mechanics_batch_outputs(
    const int64_t tick,
    const std::vector<torch::Tensor>& outputs
) {
    impl_->accept_active_mechanics_batch_outputs(tick, outputs);
}

GenerationResidualLifecycleBatchView
GenerationContext::residual_lifecycle_batch_view() const {
    return impl_->residual_lifecycle_batch_view();
}

void GenerationContext::accept_residual_lifecycle_batch_state(
    torch::Tensor pending_free_agent_ids,
    torch::Tensor pending_free_slots,
    torch::Tensor pending_free_count
) {
    impl_->accept_residual_lifecycle_batch_state(
        std::move(pending_free_agent_ids),
        std::move(pending_free_slots),
        std::move(pending_free_count)
    );
}

void GenerationContext::accept_residual_lifecycle_outputs(
    const int64_t tick,
    const std::vector<torch::Tensor>& outputs
) {
    impl_->accept_residual_lifecycle_outputs(tick, outputs);
}

void GenerationContext::execute_post_policy(const int64_t tick) {
    impl_->execute_post_policy(tick);
}

void GenerationContext::execute_post_policy_modular(
    const int64_t tick,
    const std::vector<torch::Tensor>& modular_planes
) {
    impl_->execute_post_policy_modular(tick, modular_planes);
}

std::vector<torch::Tensor> GenerationContext::finalize() {
    return impl_->finalize();
}

std::vector<torch::Tensor> GenerationContext::finalize_phase_e() {
    return impl_->finalize(true);
}

}  // namespace cuda_architecture_b
}  // namespace eiel

#define EIEL_GENERATION_DECLARE_TENSOR(name) torch::Tensor name,
#define EIEL_GENERATION_DECLARE_CONST_LIST(name) \
    const std::vector<torch::Tensor>& name,
#define EIEL_GENERATION_DECLARE_LIST(name) std::vector<torch::Tensor> name,
#define EIEL_GENERATION_DECLARE_SCALAR(type, name) type name,

std::vector<torch::Tensor> eiel_cuda_architecture_b_generation_v3(
    EIEL_CUDA_ARCH_B_GENERATION_CONTEXT_BATCH_ARGUMENTS(
        EIEL_GENERATION_DECLARE_TENSOR,
        EIEL_GENERATION_DECLARE_CONST_LIST,
        EIEL_GENERATION_DECLARE_LIST
    )
    EIEL_CUDA_ARCH_B_GENERATION_CONTEXT_SCALARS(
        EIEL_GENERATION_DECLARE_SCALAR
    )
    bool env_policy_alias
) {
#define EIEL_GENERATION_MOVE_TENSOR(name) std::move(name),
#define EIEL_GENERATION_COPY_CONST_LIST(name) name,
#define EIEL_GENERATION_MOVE_LIST(name) std::move(name),
#define EIEL_GENERATION_FORWARD_SCALAR(type, name) name,
    auto context = eiel::cuda_architecture_b::GenerationContext::prepare({
        EIEL_CUDA_ARCH_B_GENERATION_CONTEXT_BATCH_ARGUMENTS(
            EIEL_GENERATION_MOVE_TENSOR,
            EIEL_GENERATION_COPY_CONST_LIST,
            EIEL_GENERATION_MOVE_LIST
        )
        EIEL_CUDA_ARCH_B_GENERATION_CONTEXT_SCALARS(
            EIEL_GENERATION_FORWARD_SCALAR
        )
        env_policy_alias,
    });
#undef EIEL_GENERATION_MOVE_TENSOR
#undef EIEL_GENERATION_COPY_CONST_LIST
#undef EIEL_GENERATION_MOVE_LIST
#undef EIEL_GENERATION_FORWARD_SCALAR

    for (int64_t tick = 0; tick < steps; ++tick) {
        context->execute_pre_policy(tick);
        context->execute_single_cell_policy(tick);
        context->execute_post_policy(tick);
    }
    return context->finalize();
}

std::vector<torch::Tensor> eiel_cuda_architecture_b_generation_phase_e_v1(
    EIEL_CUDA_ARCH_B_GENERATION_CONTEXT_BATCH_ARGUMENTS(
        EIEL_GENERATION_DECLARE_TENSOR,
        EIEL_GENERATION_DECLARE_CONST_LIST,
        EIEL_GENERATION_DECLARE_LIST
    )
    EIEL_CUDA_ARCH_B_GENERATION_CONTEXT_SCALARS(
        EIEL_GENERATION_DECLARE_SCALAR
    )
    bool env_policy_alias,
    const std::vector<torch::Tensor>& modular_planes,
    int64_t modular_agent_capacity,
    int64_t modular_action_count,
    int64_t modular_observation_width,
    int64_t modular_cache_width,
    int64_t modular_policy_stride,
    int64_t modular_logical_module_slots,
    int64_t modular_logical_connection_slots,
    int64_t modular_physical_module_slots,
    int64_t modular_physical_connection_slots,
    int64_t modular_development_mature_start_step,
    int64_t modular_development_late_start_step,
    int64_t modular_max_births,
    int64_t modular_turnover_enabled
) {
    TORCH_CHECK(
        modular_planes.size()
            == eiel::cuda_architecture_b::kModularTickPlaneCount,
        "Phase E generation requires exactly 57 modular planes"
    );
    TORCH_CHECK(
        modular_agent_capacity > 0
            && modular_planes.at(0).dim() == 1
            && modular_planes.at(0).size(0) == modular_agent_capacity,
        "Phase E modular agent capacity mismatch"
    );
    TORCH_CHECK(
        modular_action_count == 7
            && action_count == modular_action_count
            && modular_observation_width == 27
            && modular_cache_width == 1
            && cache_width == modular_cache_width
            && modular_policy_stride == 1,
        "Phase E modular policy shape contract mismatch"
    );
    TORCH_CHECK(
        (modular_logical_module_slots == 12
             && modular_logical_connection_slots == 32)
            || (modular_logical_module_slots == 16
                && modular_logical_connection_slots == 48),
        "Phase E modular logical profile is unsupported"
    );
    TORCH_CHECK(
        modular_physical_module_slots == 16
            && modular_physical_connection_slots == 48,
        "Phase E modular physical envelope mismatch"
    );
    TORCH_CHECK(
        modular_development_mature_start_step
                == kPhaseEDevelopmentMatureStartStep
            && modular_development_late_start_step
                == kPhaseEDevelopmentLateStartStep,
        "Phase E modular development schedule must equal 20/40"
    );
    TORCH_CHECK(
        modular_max_births == 0
            && modular_turnover_enabled == 0
            && max_births_per_tick == 0
            && !birth_enabled
            && !generation_turnover_enabled,
        "Phase E first live generation lane forbids birth and turnover"
    );
    TORCH_CHECK(
        view_radius == 2,
        "Phase E modular observations require the frozen radius-2 width"
    );

    // This preflight must remain before GenerationContext::prepare. That
    // constructor clears caches and initializes generation reporting state.
    // Any contradictory Phase E envelope therefore fails before mutation.
    eiel::cuda_architecture_b::validate_modular_planes_v1(modular_planes);
    const int64_t capacity = modular_agent_capacity;
    const auto owner_device = alive_mask.device();
    TORCH_CHECK(
        alive_mask.is_cuda()
            && alive_mask.scalar_type() == torch::kBool
            && alive_mask.dim() == 1
            && alive_mask.size(0) == capacity
            && alive_mask.is_contiguous()
            && !alive_mask.requires_grad(),
        "Phase E alive_mask must be contiguous CUDA bool[capacity]"
    );
    TORCH_CHECK(
        energy.device() == owner_device
            && energy.scalar_type() == torch::kFloat32
            && energy.dim() == 1
            && energy.size(0) == capacity
            && energy.is_contiguous()
            && !energy.requires_grad(),
        "Phase E energy must be contiguous float32[capacity]"
    );
    TORCH_CHECK(
        slot_to_agent_id.device() == owner_device
            && slot_to_agent_id.scalar_type() == torch::kInt64
            && slot_to_agent_id.dim() == 1
            && slot_to_agent_id.size(0) == capacity
            && slot_to_agent_id.is_contiguous()
            && !slot_to_agent_id.requires_grad(),
        "Phase E slot_to_agent_id must be contiguous int64[capacity]"
    );
    TORCH_CHECK(
        slot_tokens.device() == owner_device
            && slot_tokens.scalar_type() == torch::kInt64
            && slot_tokens.dim() == 1
            && slot_tokens.size(0) == capacity
            && slot_tokens.is_contiguous()
            && !slot_tokens.requires_grad(),
        "Phase E slot_tokens must be contiguous int64[capacity]"
    );
    TORCH_CHECK(
        agent_id_to_slot.device() == owner_device
            && agent_id_to_slot.scalar_type() == torch::kInt32
            && agent_id_to_slot.dim() == 1
            && agent_id_to_slot.numel() > 0
            && agent_id_to_slot.is_contiguous()
            && !agent_id_to_slot.requires_grad(),
        "Phase E agent_id_to_slot must be a nonempty contiguous int32 vector"
    );
    TORCH_CHECK(
        agent_ids_by_alive_order.device() == owner_device
            && agent_ids_by_alive_order.scalar_type() == torch::kInt64
            && agent_ids_by_alive_order.dim() == 1
            && agent_ids_by_alive_order.size(0) <= capacity
            && agent_ids_by_alive_order.is_contiguous()
            && !agent_ids_by_alive_order.requires_grad(),
        "Phase E alive-order identities must be contiguous int64[<=capacity]"
    );
    TORCH_CHECK(
        action_cache_token.device() == owner_device
            && action_cache_token.scalar_type() == torch::kInt64
            && action_cache_token.dim() == 1
            && action_cache_token.size(0) == capacity
            && action_cache_token.is_contiguous()
            && !action_cache_token.requires_grad()
            && action_cache_cursor.device() == owner_device
            && action_cache_cursor.scalar_type() == torch::kInt32
            && action_cache_cursor.dim() == 1
            && action_cache_cursor.size(0) == capacity
            && action_cache_cursor.is_contiguous()
            && !action_cache_cursor.requires_grad()
            && action_cache_next.device() == owner_device
            && action_cache_next.scalar_type() == torch::kInt32
            && action_cache_next.dim() == 1
            && action_cache_next.size(0) == capacity
            && action_cache_next.is_contiguous()
            && !action_cache_next.requires_grad()
            && action_cache_actions.device() == owner_device
            && action_cache_actions.scalar_type() == torch::kInt32
            && action_cache_actions.dim() == 2
            && action_cache_actions.size(0) == capacity
            && action_cache_actions.size(1) >= modular_cache_width
            && action_cache_actions.is_contiguous()
            && !action_cache_actions.requires_grad(),
        "Phase E cache owner planes have an invalid fixed shape"
    );
    TORCH_CHECK(
        drives.device() == owner_device
            && drives.scalar_type() == torch::kFloat32
            && drives.dim() == 2
            && drives.size(0) == capacity
            && drives.size(1) >= 3
            && drives.is_contiguous()
            && !drives.requires_grad(),
        "Phase E drives must be contiguous float32[capacity,>=3]"
    );
    TORCH_CHECK(
        survival_steps.device() == owner_device
            && survival_steps.scalar_type() == torch::kInt32
            && survival_steps.dim() == 1
            && survival_steps.size(0) == capacity
            && survival_steps.is_contiguous()
            && !survival_steps.requires_grad(),
        "Phase E survival_steps must be contiguous int32[capacity]"
    );
    TORCH_CHECK(
        modular_planes.at(0).device() == owner_device,
        "Phase E modular planes and generation owner must share one device"
    );

    at::_assert_async(
        torch::all(torch::eq(modular_planes.at(0), alive_mask)),
        "Phase E modular occupancy disagrees with generation alive_mask"
    );
    at::_assert_async(
        torch::all(torch::eq(modular_planes.at(1), slot_to_agent_id)),
        "Phase E modular identity disagrees with generation slot ownership"
    );
    at::_assert_async(
        torch::all(torch::logical_or(
            torch::logical_not(alive_mask),
            torch::eq(modular_planes.at(2), slot_tokens)
        )),
        "Phase E modular slot tokens disagree with generation ownership"
    );

    at::_assert_async(
        torch::all(torch::logical_or(alive_mask, slot_to_agent_id == -1)),
        "Phase E dead generation slots must not retain an owner identity"
    );
    auto entry_identity_pairs = compact_identity_pairs_by_agent_id(
        slot_to_agent_id, alive_mask
    );
    at::_assert_async(
        entry_identity_pairs.active_count == agent_ids_by_alive_order.size(0),
        "Phase E alive-order count disagrees with alive_mask"
    );

    auto ordered_agent_ids = entry_identity_pairs.agent_ids.slice(
        0, 0, agent_ids_by_alive_order.size(0)
    );
    at::_assert_async(
        torch::all(agent_ids_by_alive_order == ordered_agent_ids),
        "Phase E alive-order identities disagree with compact slot ownership"
    );
    at::_assert_async(
        torch::all(torch::logical_and(
            agent_ids_by_alive_order >= 0,
            agent_ids_by_alive_order
                <= std::numeric_limits<int32_t>::max()
        )),
        "Phase E active identities exceed the int32 world-grid representation"
    );
    auto ordered_slots = entry_identity_pairs.slots.slice(
        0, 0, agent_ids_by_alive_order.size(0)
    );
    auto ordered_slots_long = ordered_slots.to(torch::kInt64);
    at::_assert_async(
        torch::all(torch::logical_and(ordered_slots >= 0, ordered_slots < capacity)),
        "Phase E alive-order slot is outside the fixed agent capacity"
    );
    at::_assert_async(
        torch::all(torch::logical_and(
            alive_mask.index_select(0, ordered_slots_long),
            slot_to_agent_id.index_select(0, ordered_slots_long)
                == agent_ids_by_alive_order
        )),
        "Phase E alive-order identities disagree with slot ownership"
    );
    auto ordered_slot_counts = torch::zeros(
        {capacity}, slot_to_agent_id.options().dtype(torch::kInt32)
    );
    ordered_slot_counts.index_add_(
        0,
        ordered_slots_long,
        torch::ones(
            ordered_slots.sizes(),
            slot_to_agent_id.options().dtype(torch::kInt32)
        )
    );
    at::_assert_async(
        torch::all(ordered_slot_counts <= 1),
        "Phase E alive-order slots must be unique"
    );
    if (start_tick != 0) {
        auto cache_hit = alive_mask
            .logical_and(action_cache_token == slot_tokens)
            .logical_and(action_cache_cursor >= 0)
            .logical_and(action_cache_cursor < modular_cache_width);
        at::_assert_async(
            torch::all(torch::logical_not(cache_hit)),
            "Phase E cache-width-one lane requires every active row to miss"
        );
    }

#define EIEL_GENERATION_MOVE_TENSOR(name) std::move(name),
#define EIEL_GENERATION_COPY_CONST_LIST(name) name,
#define EIEL_GENERATION_MOVE_LIST(name) std::move(name),
#define EIEL_GENERATION_FORWARD_SCALAR(type, name) name,
    auto context = eiel::cuda_architecture_b::GenerationContext::prepare({
        EIEL_CUDA_ARCH_B_GENERATION_CONTEXT_BATCH_ARGUMENTS(
            EIEL_GENERATION_MOVE_TENSOR,
            EIEL_GENERATION_COPY_CONST_LIST,
            EIEL_GENERATION_MOVE_LIST
        )
        EIEL_CUDA_ARCH_B_GENERATION_CONTEXT_SCALARS(
            EIEL_GENERATION_FORWARD_SCALAR
        )
        env_policy_alias,
    });
#undef EIEL_GENERATION_MOVE_TENSOR
#undef EIEL_GENERATION_COPY_CONST_LIST
#undef EIEL_GENERATION_MOVE_LIST
#undef EIEL_GENERATION_FORWARD_SCALAR

    for (int64_t tick = 0; tick < steps; ++tick) {
        context->execute_pre_policy(tick);
        context->execute_single_cell_modular_policy(tick, modular_planes);
        context->execute_post_policy_modular(tick, modular_planes);
    }
    return resident_lifecycle_state_enabled
        ? context->finalize_phase_e()
        : context->finalize();
}

#undef EIEL_GENERATION_DECLARE_TENSOR
#undef EIEL_GENERATION_DECLARE_CONST_LIST
#undef EIEL_GENERATION_DECLARE_LIST
#undef EIEL_GENERATION_DECLARE_SCALAR

void eiel_cuda_architecture_b_bind_generation_multicell(
    pybind11::module_& module
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.doc() = "Bounded CUDA Architecture B resident generation composition";
    module.def(
        "eiel_cuda_architecture_b_generation_v3",
        &eiel_cuda_architecture_b_generation_v3
    );
    module.def(
        "eiel_cuda_architecture_b_generation_phase_e_v1",
        &eiel_cuda_architecture_b_generation_phase_e_v1
    );
    eiel_cuda_architecture_b_bind_generation_multicell(module);
}
