#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace eiel::cuda_architecture_b::stackless_birth_generation {

constexpr int kThreadsPerCell = 128;

struct KernelResourceReceipt {
    int32_t binary_version;
    int32_t max_blocks_per_multiprocessor;
    int32_t max_threads_per_block;
    int32_t num_registers;
    int32_t ptx_version;
    int64_t constant_size_bytes;
    int64_t local_size_bytes;
    int64_t shared_size_bytes;
};

struct FlatFullTickState {
    int32_t* age;
    float* gru_upkeep_cost;
    const float* cost_charge_scalar;
    const float* cost_charge_maturation_scalar;
    float* drive_inertia;
    float* drive_noise;
    float* prev_drive_delta;
    float* drive_volatility;
    float* drive_jitter;
    float* episode0_drive_jitter_sum;
    float* last_energy_delta;
    float* last_health_delta;
    float* last_injury_delta;
    int32_t* selfharm_events;
    int32_t* episode0_selfharm_events;
    float* chemistry;
    uint64_t* runner_pcg64_state_words;
    int64_t* runner_pcg64_buffer_words;
    uint64_t* shared_env_policy_pcg64_state_words;
    int64_t* shared_env_policy_pcg64_buffer_words;
    const float* drive_noise_samples;
    const bool* drive_noise_samples_valid;
    int64_t* slot_free_queue;
    int64_t* slot_free_queue_count;
    int64_t* residual_pending_free_agent_ids;
    int64_t* residual_pending_free_slots;
    int64_t* residual_pending_free_count;
    int64_t* policy_np_semantic_draws;
    int64_t* metrics_np_semantic_draws;
    int64_t drive_noise_sample_ticks;
    int64_t chemistry_width;
};

struct FlatFullTickOutputs {
    int32_t* chemistry_order_slots;
    int64_t* runner_shuffle_draws;
    int32_t* runner_shuffle_status;
    int64_t* drive_noise_draws;
    int32_t* drive_noise_status;
    int64_t* chemistry_draws;
    int32_t* chemistry_status;
    int32_t* active_mechanics_status;
    int64_t* active_mechanics_counts;
    int64_t* residual_death_agent_ids;
    int64_t* residual_death_slots;
    int32_t* residual_death_reason_codes;
    int64_t* residual_death_count;
    int64_t* residual_release_count;
    int32_t* residual_release_status;
    uint64_t* policy_pcg64_state_after_action;
    int64_t* policy_pcg64_buffer_after_action;
    int64_t* food_spawn_stats;
};

struct FlatEventState {
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
    int64_t* active_count;
    int32_t* status;
    int32_t* lineage_founder_id;
    int32_t* lineage_depth;
    const int64_t* initial_id_to_row;
    int32_t* terminal_survival;
    int32_t* terminal_rows;
    int32_t* status_by_tick;
    int64_t capacity;
    int64_t reporting_rows;
};

struct FlatOrdinaryBirthState {
    float* scar;
    int32_t* repro_cooldown;
    float* repro_threshold;
    int32_t* offspring_count;
    int32_t* lineage_birth_generation;
    int32_t* lineage_birth_tick;
    int32_t* patch_habitat_lineage_id;
    int64_t* population_next_id;

    uint32_t* python_random_words;
    int64_t* python_random_index;
    bool* python_random_has_gauss;
    double* python_random_gauss_next;
    const double* python_random_gauss_primary_tape;
    const double* python_random_gauss_secondary_tape;
    int64_t* python_random_raw_draws;
    int64_t* mutation_py_semantic_draws;

    int32_t* genome_hidden_size;
    int32_t* genome_memory_dim;
    int32_t* genome_sensor_channel_mask;
    int32_t* genome_sensor_radius;
    int32_t* genome_introspection_channel_mask;
    int32_t* genome_self_regulation_control_mask;
    int32_t* genome_self_modification_target_mask;
    int32_t* genome_directed_inheritance_channel_mask;
    int64_t* policy_mem_dim;
    bool* policy_use_gru;

    float* genome_exploration_bias;
    float* genome_aggression_bias;
    float* genome_mutation_rate;
    float* genome_reproduction_threshold;
    float* genome_action_temperature;
    float* genome_drive_inertia;
    float* genome_drive_noise;
    float* genome_drive_logit_gain;
    float* genome_pred_lr;
    float* genome_drive_sp_hunger;
    float* genome_drive_sp_pain;
    float* genome_drive_sp_fatigue;
    float* genome_drive_sp_fear;
    float* genome_exp_prior_strength;
    float* genome_exp_prior_noise;
    float* genome_drive_birth_noise;
    float* pred_lr;

    float* birth_reset_shock;
    uint64_t* population_pcg64_state_words;
    int64_t* population_pcg64_buffer_words;
    int64_t* mutation_np_semantic_draws;

    float* hmem;
    uint8_t* mutation_torch_cpu_state;
    const float* policy_birth_delta_tape;
    const bool* policy_birth_mutation_mask_tape;
    const uint8_t* policy_birth_post_rng_state_packed_tape;
    const int64_t* policy_birth_semantic_draws_tape;
    int64_t* policy_birth_tape_cursor;
    int64_t* mutation_torch_cpu_semantic_draws;

    int32_t* attacks;
    int32_t* early_survival_steps;
    float* early_energy_collected;
    double* early_energy_collected_exact;
    float* early_injury_gained;
    float* early_life_weight;
    float* pred_err_sum;
    int32_t* pred_err_count;
    int32_t* first_repro_step;
    bool* dev_stage_initialized;
    float* episode0_pred_err_sum;
    int32_t* episode0_pred_err_count;
    float* late_pred_err_sum;
    int32_t* late_pred_err_count;
    float* chemistry_observation_fidelity;
    float* chemistry_observation_bias;
};

struct FlatOrdinaryBirthOutputs {
    int64_t* slot_free_queue_head;
    int32_t* planned_repro_cooldown;
    int64_t* planned_parent_slots;
    int64_t* planned_child_slots;
    int32_t* planned_spawn_positions;
    float* planned_child_energy;
    int64_t* decision_parent_slots;
    int32_t* decision_reason_codes;
    int32_t* decision_candidate_counts;
    int64_t* decision_draw_starts;
    int64_t* decision_draw_ends;
    int32_t* decision_selected_positions;
    bool* decision_selected_reserved;
    int64_t* reason_counts;
    int64_t* planned_count;
    int64_t* decision_count;
    int64_t* spawn_rng_draws;
    int64_t* spawn_rng_semantic_draws;
    int64_t* runner_rng_draws;
    int64_t* runner_rng_semantic_draws;
    int32_t* plan_status;
    int64_t* committed_parent_agent_ids;
    int64_t* committed_child_agent_ids;
    int64_t* committed_child_slots;
    int64_t* identity_committed_count;
    int32_t* identity_status;
    int64_t* scalar_committed_count;
    int64_t* scalar_semantic_draws;
    int32_t* scalar_status;
    int64_t* drive_committed_count;
    int64_t* birth_noise_rows_consumed;
    int64_t* exp_prior_rows_consumed;
    int64_t* standard_normal_rows;
    int64_t* drive_semantic_draws;
    int32_t* drive_status;
    int32_t* chemistry_status;
    int64_t* policy_committed_count;
    int64_t* policy_semantic_draws;
    int64_t* policy_mutation_events;
    int64_t* policy_mutated_elements;
    double* policy_mutation_delta_total;
    bool* policy_mutation_occurred;
    float* policy_mutation_delta;
    int32_t* policy_status;
    int32_t* combined_status;
    uint64_t* planner_pcg64_state_before;
    int64_t* planner_pcg64_buffer_before;
};

struct FlatStacklessBirthScratch {
    uint32_t* python_trial_words;
    int64_t words_per_cell;
};

struct FlatOrdinaryBirthConfig {
    bool enabled;
    int64_t max_births_per_tick;
    double parent_energy_share;
    float minimum_energy_buffer;
    int32_t cooldown_steps;
    int32_t sensor_radius_max;
    double injury_repro_k_injury;
    double injury_repro_k_scar;
    double injury_birth_fail_scar;
    double injury_birth_fail_injury;
    bool env_policy_alias;
    bool requires_mate;
    int64_t gauss_tape_capacity;
    int64_t policy_tape_capacity;
    int64_t torch_state_bytes;
    int64_t hmem_width;
};

struct FlatMechanicsConfig {
    bool enable_rest_charge;
    int32_t rest_streak_free;
    float rest_charge_energy;
    float rest_charge_growth;
    float rest_deficit_to_health;
    int32_t developmental_maturity_age;
    float drive_hunger_gain;
    float drive_hunger_decay;
    float drive_pain_gain;
    float drive_pain_decay;
    float drive_fatigue_gain;
    float drive_fatigue_rest_relief;
    float drive_fatigue_decay;
    float drive_fear_gain;
    float drive_fear_decay;
    int64_t generation;
    int64_t start_tick;
};

void launch_flat_dynamic_policy_food_stackless_birth_full_tick_v1(
    int64_t* active_agent_ids,
    int64_t* active_count_state,
    int64_t* slot_to_agent_id,
    int32_t* agent_id_to_slot,
    bool* alive_mask,
    int64_t* slot_tokens,
    int32_t* action_cache_actions,
    int64_t* action_cache_token,
    int32_t* action_cache_cursor,
    int32_t* action_cache_next,
    int32_t* pos_y,
    int32_t* pos_x,
    int32_t* agent_grid,
    int32_t* food_grid,
    float* energy,
    float* health,
    float* injury,
    double* policy_energy_observation,
    int32_t* rest_streak,
    int64_t* pending_free_agent_ids,
    int64_t* pending_free_slots,
    int64_t* pending_free_count,
    float* energy_collected,
    double* energy_collected_exact,
    int32_t* last_action,
    int32_t* last_last_action,
    int32_t* action_switches,
    int32_t* action_oscillations,
    int32_t* episode0_switches,
    int32_t* episode0_oscillations,
    int32_t* survival_steps,
    float* mlp_pack,
    float* drives,
    float* attack_bias,
    float* action_temperature,
    float* drive_logit_gain,
    float* food_gained_scratch,
    int64_t* ordered_agent_ids_out,
    int32_t* ordered_slots_out,
    float* observations_out,
    float* logits_out,
    int32_t* row_actions_out,
    int32_t* actions_by_slot_out,
    float* pre_energy_out,
    float* pre_health_out,
    float* pre_injury_out,
    int32_t* pre_survival_steps_out,
    int64_t* active_count_out,
    int64_t* policy_refill_rows_out,
    int64_t* policy_cache_hit_rows_out,
    int64_t* death_agent_ids_out,
    int64_t* death_slots_out,
    int32_t* death_reason_codes_out,
    int64_t* death_count_out,
    int64_t* survivor_agent_ids_out,
    int64_t* survivor_count_out,
    int32_t* lifecycle_status_out,
    int32_t* status_out,
    FlatFullTickState full_tick_state,
    FlatFullTickOutputs full_tick_outputs,
    FlatEventState event_state,
    FlatOrdinaryBirthState ordinary_birth_state,
    FlatOrdinaryBirthOutputs ordinary_birth_outputs,
    FlatStacklessBirthScratch ordinary_birth_scratch,
    FlatOrdinaryBirthConfig ordinary_birth_config,
    FlatMechanicsConfig mechanics_config,
    int64_t cells,
    int64_t ticks,
    int64_t active_capacity,
    int64_t agent_id_capacity,
    int64_t capacity,
    int64_t pending_capacity,
    int64_t height,
    int64_t width,
    int64_t view_radius,
    int64_t observation_size,
    int64_t hidden_size,
    int64_t action_count,
    int64_t packed_width,
    int64_t drive_width,
    int64_t cache_capacity,
    int64_t cache_width,
    bool deterministic_actions,
    int64_t max_food,
    int64_t effective_food_spawn_count,
    double max_energy,
    double start_health,
    double move_cost,
    double gather_cost,
    double rest_cost,
    double attack_cost,
    double action_cost_mult,
    double rest_cost_mult,
    double food_energy,
    int64_t episode0_k,
    float w_hunger_gather,
    float w_pain_rest,
    float w_fatigue_rest,
    float w_pain_attack,
    float w_fear_attack,
    cudaStream_t stream
);

KernelResourceReceipt
query_flat_dynamic_policy_food_stackless_birth_full_tick_resources_v1();

}  // namespace eiel::cuda_architecture_b::stackless_birth_generation
