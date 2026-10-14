#pragma once

#include <torch/extension.h>

#include <cstdint>
#include <memory>
#include <vector>

#define EIEL_CUDA_ARCH_B_GENERATION_CONTEXT_BATCH_ARGUMENTS(                 \
    TENSOR, CONST_LIST, LIST                                                \
)                                                                           \
    TENSOR(agent_ids_by_alive_order)                                        \
    TENSOR(agent_id_to_slot)                                                \
    TENSOR(slot_to_agent_id)                                                \
    TENSOR(alive_mask)                                                      \
    TENSOR(pos_y)                                                           \
    TENSOR(pos_x)                                                           \
    TENSOR(agent_grid)                                                      \
    TENSOR(food_grid)                                                       \
    TENSOR(energy)                                                          \
    TENSOR(health)                                                          \
    TENSOR(injury)                                                          \
    TENSOR(age)                                                             \
    TENSOR(policy_energy_observation)                                       \
    TENSOR(rest_streak)                                                     \
    TENSOR(survival_steps)                                                  \
    TENSOR(last_action)                                                     \
    TENSOR(last_last_action)                                                \
    TENSOR(action_switches)                                                 \
    TENSOR(action_oscillations)                                             \
    TENSOR(episode0_switches)                                               \
    TENSOR(episode0_oscillations)                                           \
    TENSOR(gru_upkeep_cost)                                                 \
    TENSOR(cost_charge_scalar)                                              \
    TENSOR(cost_charge_maturation_scalar)                                   \
    TENSOR(patch_food_amount)                                               \
    TENSOR(patch_fertility)                                                 \
    TENSOR(patch_cover)                                                     \
    TENSOR(patch_shelter)                                                   \
    TENSOR(patch_scent)                                                     \
    TENSOR(patch_waste)                                                     \
    TENSOR(patch_density)                                                   \
    TENSOR(patch_scent_emit_pending)                                        \
    TENSOR(patch_waste_emit_pending)                                        \
    TENSOR(patch_movement_cost)                                             \
    TENSOR(mlp_pack)                                                        \
    TENSOR(drives)                                                          \
    TENSOR(drive_inertia)                                                   \
    TENSOR(drive_noise)                                                     \
    TENSOR(prev_drive_delta)                                                \
    TENSOR(drive_volatility)                                                \
    TENSOR(drive_jitter)                                                    \
    TENSOR(episode0_drive_jitter_sum)                                       \
    TENSOR(last_energy_delta)                                               \
    TENSOR(last_health_delta)                                               \
    TENSOR(last_injury_delta)                                               \
    TENSOR(selfharm_events)                                                 \
    TENSOR(episode0_selfharm_events)                                        \
    TENSOR(drive_noise_samples_by_tick_slot)                                \
    TENSOR(drive_noise_samples_valid_by_tick_slot)                          \
    TENSOR(attack_bias)                                                     \
    TENSOR(action_temperature)                                              \
    TENSOR(drive_logit_gain)                                                \
    TENSOR(slot_tokens)                                                     \
    TENSOR(action_cache_actions)                                            \
    TENSOR(action_cache_token)                                              \
    TENSOR(action_cache_cursor)                                             \
    TENSOR(action_cache_next)                                               \
    TENSOR(pending_free_agent_ids)                                          \
    TENSOR(pending_free_slots)                                              \
    TENSOR(pending_free_count)                                              \
    TENSOR(pcg64_state_words)                                               \
    TENSOR(pcg64_buffer_words)                                              \
    TENSOR(lineage_founder_id)                                              \
    TENSOR(lineage_depth)                                                   \
    TENSOR(chemistry)                                                       \
    TENSOR(runner_pcg64_state_words)                                        \
    TENSOR(runner_pcg64_buffer_words)                                       \
    TENSOR(metrics_np_semantic_draws)                                       \
    CONST_LIST(event_tape_columns)                                          \
    TENSOR(event_tape_active_count)                                         \
    TENSOR(event_tape_status)                                               \
    LIST(birth_identity_state)                                              \
    LIST(birth_scalar_rng_state)                                            \
    LIST(birth_genome_int32_fields)                                         \
    TENSOR(birth_policy_mem_dim)                                            \
    TENSOR(birth_policy_use_gru)                                            \
    LIST(birth_genome_float32_fields)                                       \
    LIST(birth_drive_state)                                                 \
    LIST(birth_policy_state)                                                \
    LIST(turnover_policy_state)                                             \
    LIST(turnover_state)                                                    \
    TENSOR(birth_planner_pcg64_state_words)                                 \
    TENSOR(birth_planner_pcg64_buffer_words)                                \
    TENSOR(birth_policy_np_semantic_draws)

#define EIEL_CUDA_ARCH_B_GENERATION_CONTEXT_SCALARS(SCALAR) \
    SCALAR(bool, resident_lifecycle_state_enabled)           \
    SCALAR(bool, birth_enabled)                              \
    SCALAR(bool, generation_turnover_enabled)                \
    SCALAR(int64_t, steps)                                   \
    SCALAR(bool, patch_enabled)                              \
    SCALAR(bool, deterministic)                              \
    SCALAR(int64_t, generation)                              \
    SCALAR(int64_t, start_tick)                              \
    SCALAR(int64_t, view_radius)                             \
    SCALAR(double, max_energy)                               \
    SCALAR(double, start_health)                             \
    SCALAR(double, patch_food_amount_max)                    \
    SCALAR(double, patch_scent_max)                          \
    SCALAR(double, patch_waste_max)                          \
    SCALAR(int64_t, patch_density_radius)                    \
    SCALAR(bool, patch_density_include_self)                 \
    SCALAR(bool, patch_density_normalize)                    \
    SCALAR(double, patch_movement_cost_min)                  \
    SCALAR(double, patch_movement_cost_max)                  \
    SCALAR(int64_t, hidden_size)                             \
    SCALAR(int64_t, action_count)                            \
    SCALAR(int64_t, cache_width)                             \
    SCALAR(int64_t, rng_seed)                                \
    SCALAR(int64_t, rng_counter)                             \
    SCALAR(double, w_hunger_gather)                          \
    SCALAR(double, w_pain_rest)                              \
    SCALAR(double, w_fatigue_rest)                           \
    SCALAR(double, w_pain_attack)                            \
    SCALAR(double, w_fear_attack)                            \
    SCALAR(double, move_cost)                                \
    SCALAR(double, gather_cost)                              \
    SCALAR(double, rest_cost)                                \
    SCALAR(double, attack_cost)                              \
    SCALAR(double, action_cost_mult)                         \
    SCALAR(double, rest_cost_mult)                           \
    SCALAR(double, food_energy)                              \
    SCALAR(int64_t, max_food)                                \
    SCALAR(double, food_spawn_per_step)                      \
    SCALAR(double, food_spawn_mult)                          \
    SCALAR(bool, enable_biomes)                              \
    SCALAR(double, patch_food_initial_value)                 \
    SCALAR(double, patch_food_gather_amount)                 \
    SCALAR(double, patch_food_presence_threshold)            \
    SCALAR(double, patch_scent_move_emit)                    \
    SCALAR(double, patch_waste_move_emit)                    \
    SCALAR(double, patch_scent_gather_emit)                  \
    SCALAR(double, patch_waste_consume_emit)                 \
    SCALAR(bool, patch_movement_cost_apply_enable)           \
    SCALAR(double, patch_movement_cost_energy_scale)         \
    SCALAR(double, patch_scent_decay_per_step)               \
    SCALAR(double, patch_waste_decay_per_step)               \
    SCALAR(double, patch_scent_presence_emit)                \
    SCALAR(double, patch_waste_presence_emit)                \
    SCALAR(double, patch_waste_density_emit_scale)           \
    SCALAR(double, patch_food_regrow_per_step)               \
    SCALAR(bool, patch_food_regrow_use_fertility)            \
    SCALAR(bool, patch_food_regrow_cap_by_fertility)         \
    SCALAR(bool, patch_world_waste_consequence_enable)       \
    SCALAR(bool, patch_waste_regrowth_penalty_enable)        \
    SCALAR(double, patch_waste_regrowth_penalty_scale)       \
    SCALAR(double, patch_waste_regrowth_penalty_max)         \
    SCALAR(bool, patch_density_regrowth_penalty_enable)      \
    SCALAR(double, patch_density_regrowth_penalty_scale)     \
    SCALAR(double, patch_density_regrowth_penalty_max)       \
    SCALAR(double, patch_regrowth_min_multiplier)            \
    SCALAR(bool, active_mechanics_enabled)                   \
    SCALAR(bool, enable_rest_charge)                         \
    SCALAR(int64_t, rest_streak_free)                        \
    SCALAR(double, rest_charge_energy)                       \
    SCALAR(double, rest_charge_growth)                       \
    SCALAR(double, rest_deficit_to_health)                   \
    SCALAR(int64_t, developmental_maturity_age)              \
    SCALAR(int64_t, episode0_k)                              \
    SCALAR(double, drive_hunger_gain)                        \
    SCALAR(double, drive_hunger_decay)                       \
    SCALAR(double, drive_pain_gain)                          \
    SCALAR(double, drive_pain_decay)                         \
    SCALAR(double, drive_fatigue_gain)                       \
    SCALAR(double, drive_fatigue_rest_relief)                \
    SCALAR(double, drive_fatigue_decay)                      \
    SCALAR(double, drive_fear_gain)                          \
    SCALAR(double, drive_fear_decay)                         \
    SCALAR(double, reproduce_parent_energy_share)            \
    SCALAR(double, reproduce_min_energy_buffer)              \
    SCALAR(int64_t, reproduce_cooldown_steps)                \
    SCALAR(int64_t, genome_sensor_radius_max)                \
    SCALAR(double, injury_repro_k_injury)                    \
    SCALAR(double, injury_repro_k_scar)                      \
    SCALAR(double, injury_birth_fail_scar)                   \
    SCALAR(double, injury_birth_fail_injury)                 \
    SCALAR(int64_t, max_births_per_tick)                     \
    SCALAR(bool, reproduce_requires_mate)                    \
    SCALAR(int64_t, population_size)                         \
    SCALAR(double, cull_fraction)                            \
    SCALAR(double, parent_fraction)                          \
    SCALAR(double, start_energy)                             \
    SCALAR(int64_t, early_life_steps)                        \
    SCALAR(double, early_life_weight)                        \
    SCALAR(bool, enable_combat)                              \
    SCALAR(double, coherence_thrashing_weight)               \
    SCALAR(double, coherence_drive_volatility_weight)        \
    SCALAR(double, coherence_selfharm_weight)                \
    SCALAR(double, coherence_pred_err_weight)

namespace eiel {
namespace cuda_architecture_b {

struct GenerationContextArguments {
#define EIEL_CONTEXT_TENSOR_FIELD(name) torch::Tensor name;
#define EIEL_CONTEXT_CONST_LIST_FIELD(name) std::vector<torch::Tensor> name;
#define EIEL_CONTEXT_LIST_FIELD(name) std::vector<torch::Tensor> name;
#define EIEL_CONTEXT_SCALAR_FIELD(type, name) type name;
    EIEL_CUDA_ARCH_B_GENERATION_CONTEXT_BATCH_ARGUMENTS(
        EIEL_CONTEXT_TENSOR_FIELD,
        EIEL_CONTEXT_CONST_LIST_FIELD,
        EIEL_CONTEXT_LIST_FIELD
    )
    EIEL_CUDA_ARCH_B_GENERATION_CONTEXT_SCALARS(
        EIEL_CONTEXT_SCALAR_FIELD
    )
    bool env_policy_alias;
#undef EIEL_CONTEXT_TENSOR_FIELD
#undef EIEL_CONTEXT_CONST_LIST_FIELD
#undef EIEL_CONTEXT_LIST_FIELD
#undef EIEL_CONTEXT_SCALAR_FIELD
};

struct GenerationPolicyBatchView {
    torch::Tensor observations;
    torch::Tensor miss_slots;
    torch::Tensor miss_rows;
    torch::Tensor survival_steps;
};

struct GenerationPrePolicyBatchView {
    torch::Tensor active_agent_ids;
    torch::Tensor active_count;
    torch::Tensor slots;
};

struct GenerationResidualLifecycleBatchView {
    torch::Tensor pending_free_agent_ids;
    torch::Tensor pending_free_slots;
    torch::Tensor pending_free_count;
};

struct GenerationActiveMechanicsBatchView {
    torch::Tensor actions_by_slot;
    torch::Tensor pre_energy;
    torch::Tensor pre_health;
    torch::Tensor pre_injury;
};

class GenerationContext final {
public:
    static std::unique_ptr<GenerationContext> prepare(
        GenerationContextArguments arguments
    );

    ~GenerationContext();

    GenerationContext(const GenerationContext&) = delete;
    GenerationContext& operator=(const GenerationContext&) = delete;

    void execute_pre_policy(int64_t tick);
    void execute_pre_policy_setup(int64_t tick);
    GenerationPrePolicyBatchView pre_policy_batch_view() const;
    void accept_policy_cache_batch_outputs(
        int64_t tick,
        const std::vector<torch::Tensor>& outputs
    );
    void accept_observations(torch::Tensor observations);
    void execute_single_cell_policy(int64_t tick);
    void execute_single_cell_policy_with_residual(
        int64_t tick,
        torch::Tensor logit_residuals
    );
    void execute_single_cell_modular_policy(
        int64_t tick,
        const std::vector<torch::Tensor>& modular_planes
    );
    GenerationPolicyBatchView policy_batch_view() const;
    void accept_policy_batch_outputs(
        int64_t tick,
        const std::vector<torch::Tensor>& outputs
    );
    void execute_post_policy_before_action_lifecycle(int64_t tick);
    void accept_action_lifecycle_outputs(
        int64_t tick,
        const std::vector<torch::Tensor>& outputs
    );
    GenerationActiveMechanicsBatchView active_mechanics_batch_view() const;
    void execute_single_cell_active_mechanics(int64_t tick);
    void accept_active_mechanics_batch_outputs(
        int64_t tick,
        const std::vector<torch::Tensor>& outputs
    );
    GenerationResidualLifecycleBatchView residual_lifecycle_batch_view() const;
    void accept_residual_lifecycle_batch_state(
        torch::Tensor pending_free_agent_ids,
        torch::Tensor pending_free_slots,
        torch::Tensor pending_free_count
    );
    void accept_residual_lifecycle_outputs(
        int64_t tick,
        const std::vector<torch::Tensor>& outputs
    );
    void execute_post_policy(int64_t tick);
    void execute_post_policy_modular(
        int64_t tick,
        const std::vector<torch::Tensor>& modular_planes
    );
    std::vector<torch::Tensor> finalize();
    std::vector<torch::Tensor> finalize_phase_e();

private:
    class Impl;

    explicit GenerationContext(std::unique_ptr<Impl> impl);

    std::unique_ptr<Impl> impl_;
};

}  // namespace cuda_architecture_b
}  // namespace eiel
