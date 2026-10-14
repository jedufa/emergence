#include <torch/extension.h>

#include <ATen/MemoryOverlap.h>
#include <ATen/cuda/CUDAEvent.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

#include "cuda_architecture_b_generation_context.hpp"

#define EIEL_ARCH_B_GENERATION_BATCH_ARGUMENTS(TENSOR, CONST_LIST, LIST) \
    TENSOR(agent_ids_by_alive_order)                                      \
    TENSOR(agent_id_to_slot)                                              \
    TENSOR(slot_to_agent_id)                                              \
    TENSOR(alive_mask)                                                    \
    TENSOR(pos_y)                                                         \
    TENSOR(pos_x)                                                         \
    TENSOR(agent_grid)                                                    \
    TENSOR(food_grid)                                                     \
    TENSOR(energy)                                                        \
    TENSOR(health)                                                        \
    TENSOR(injury)                                                        \
    TENSOR(age)                                                           \
    TENSOR(policy_energy_observation)                                     \
    TENSOR(rest_streak)                                                   \
    TENSOR(survival_steps)                                                \
    TENSOR(last_action)                                                   \
    TENSOR(last_last_action)                                              \
    TENSOR(action_switches)                                               \
    TENSOR(action_oscillations)                                           \
    TENSOR(episode0_switches)                                             \
    TENSOR(episode0_oscillations)                                         \
    TENSOR(gru_upkeep_cost)                                               \
    TENSOR(cost_charge_scalar)                                            \
    TENSOR(cost_charge_maturation_scalar)                                 \
    TENSOR(patch_food_amount)                                             \
    TENSOR(patch_fertility)                                               \
    TENSOR(patch_cover)                                                   \
    TENSOR(patch_shelter)                                                 \
    TENSOR(patch_scent)                                                   \
    TENSOR(patch_waste)                                                   \
    TENSOR(patch_density)                                                 \
    TENSOR(patch_scent_emit_pending)                                      \
    TENSOR(patch_waste_emit_pending)                                      \
    TENSOR(patch_movement_cost)                                           \
    TENSOR(mlp_pack)                                                      \
    TENSOR(drives)                                                        \
    TENSOR(drive_inertia)                                                 \
    TENSOR(drive_noise)                                                   \
    TENSOR(prev_drive_delta)                                              \
    TENSOR(drive_volatility)                                              \
    TENSOR(drive_jitter)                                                  \
    TENSOR(episode0_drive_jitter_sum)                                     \
    TENSOR(last_energy_delta)                                             \
    TENSOR(last_health_delta)                                             \
    TENSOR(last_injury_delta)                                             \
    TENSOR(selfharm_events)                                               \
    TENSOR(episode0_selfharm_events)                                      \
    TENSOR(drive_noise_samples_by_tick_slot)                              \
    TENSOR(drive_noise_samples_valid_by_tick_slot)                        \
    TENSOR(attack_bias)                                                   \
    TENSOR(action_temperature)                                            \
    TENSOR(drive_logit_gain)                                              \
    TENSOR(slot_tokens)                                                   \
    TENSOR(action_cache_actions)                                          \
    TENSOR(action_cache_token)                                            \
    TENSOR(action_cache_cursor)                                           \
    TENSOR(action_cache_next)                                             \
    TENSOR(pending_free_agent_ids)                                        \
    TENSOR(pending_free_slots)                                            \
    TENSOR(pending_free_count)                                            \
    TENSOR(pcg64_state_words)                                             \
    TENSOR(pcg64_buffer_words)                                            \
    TENSOR(lineage_founder_id)                                            \
    TENSOR(lineage_depth)                                                 \
    TENSOR(chemistry)                                                     \
    TENSOR(runner_pcg64_state_words)                                      \
    TENSOR(runner_pcg64_buffer_words)                                     \
    TENSOR(metrics_np_semantic_draws)                                     \
    CONST_LIST(event_tape_columns)                                        \
    TENSOR(event_tape_active_count)                                       \
    TENSOR(event_tape_status)                                             \
    LIST(birth_identity_state)                                            \
    LIST(birth_scalar_rng_state)                                          \
    LIST(birth_genome_int32_fields)                                       \
    TENSOR(birth_policy_mem_dim)                                          \
    TENSOR(birth_policy_use_gru)                                          \
    LIST(birth_genome_float32_fields)                                     \
    LIST(birth_drive_state)                                               \
    LIST(birth_policy_state)                                              \
    LIST(turnover_policy_state)                                           \
    LIST(turnover_state)                                                  \
    TENSOR(birth_planner_pcg64_state_words)                               \
    TENSOR(birth_planner_pcg64_buffer_words)                              \
    TENSOR(birth_policy_np_semantic_draws)

#define EIEL_ARCH_B_GENERATION_SHARED_SCALARS(SCALAR)       \
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

#define EIEL_DECLARE_TENSOR(name) torch::Tensor name,
#define EIEL_DECLARE_CONST_LIST(name) const std::vector<torch::Tensor>& name,
#define EIEL_DECLARE_LIST(name) std::vector<torch::Tensor> name,
#define EIEL_DECLARE_SCALAR(type, name) type name,

std::vector<torch::Tensor> eiel_cuda_architecture_b_generation_v3(
    EIEL_ARCH_B_GENERATION_BATCH_ARGUMENTS(
        EIEL_DECLARE_TENSOR,
        EIEL_DECLARE_CONST_LIST,
        EIEL_DECLARE_LIST
    )
    EIEL_ARCH_B_GENERATION_SHARED_SCALARS(EIEL_DECLARE_SCALAR)
    bool env_policy_alias
);

std::vector<torch::Tensor> eiel_cuda_fixed_mlp_refill_batched_counted_v1(
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
eiel_cuda_architecture_b_policy_cache_service_batched_v1(
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

torch::Tensor eiel_cuda_owner_observation_frame_batched_counted_v1(
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

std::vector<torch::Tensor>
eiel_cuda_architecture_b_lifecycle_death_cull_batched_counted_v1(
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

std::vector<torch::Tensor>
eiel_cuda_architecture_b_active_mechanics_batched_counted_v1(
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

constexpr int64_t kActiveMechanicsBatchMinCells = 16;

constexpr int64_t CUDA_LOW_PRIORITY_POOL_SIZE = 32;

void check_batched_tensor(
    const torch::Tensor& tensor,
    const char* label,
    const int64_t cell_count,
    const c10::Device& device
) {
    TORCH_CHECK(tensor.defined(), label, " must be defined");
    TORCH_CHECK(tensor.is_cuda(), label, " must be CUDA resident");
    TORCH_CHECK(tensor.device() == device, label, " must share one CUDA device");
    TORCH_CHECK(tensor.dim() >= 1, label, " must have a leading cell dimension");
    TORCH_CHECK(
        tensor.size(0) == cell_count,
        label,
        " has leading dimension ",
        tensor.size(0),
        "; expected ",
        cell_count
    );
    TORCH_CHECK(
        at::has_internal_overlap(tensor) == at::MemOverlap::No,
        label,
        " must have non-overlapping cell storage"
    );
}

void check_batched_tensor_list(
    const std::vector<torch::Tensor>& tensors,
    const char* label,
    const int64_t cell_count,
    const c10::Device& device
) {
    for (std::size_t index = 0; index < tensors.size(); ++index) {
        const std::string indexed_label = std::string(label) + "["
            + std::to_string(index) + "]";
        check_batched_tensor(
            tensors[index],
            indexed_label.c_str(),
            cell_count,
            device
        );
    }
}

torch::Tensor select_cell(const torch::Tensor& tensor, const int64_t cell_index) {
    return tensor.select(0, cell_index);
}

std::vector<torch::Tensor> select_cell_list(
    const std::vector<torch::Tensor>& tensors,
    const int64_t cell_index
) {
    std::vector<torch::Tensor> selected;
    selected.reserve(tensors.size());
    for (const auto& tensor : tensors) {
        selected.push_back(select_cell(tensor, cell_index));
    }
    return selected;
}

void record_tensor_on_stream(
    const torch::Tensor& tensor,
    const c10::cuda::CUDAStream& stream
) {
    if (tensor.numel() != 0) {
        c10::cuda::CUDACachingAllocator::recordStream(
            tensor.storage().data_ptr(),
            stream
        );
    }
}

void record_tensor_list_on_stream(
    const std::vector<torch::Tensor>& tensors,
    const c10::cuda::CUDAStream& stream
) {
    for (const auto& tensor : tensors) {
        record_tensor_on_stream(tensor, stream);
    }
}

void join_worker_streams(
    const std::vector<c10::cuda::CUDAStream>& streams,
    const c10::cuda::CUDAStream& caller_stream
) {
    std::vector<at::cuda::CUDAEvent> completion_events;
    completion_events.reserve(streams.size());
    for (const auto& stream : streams) {
        completion_events.emplace_back(cudaEventDisableTiming);
        completion_events.back().record(stream);
        completion_events.back().block(caller_stream);
    }
}

void close_tick_barrier(
    const std::vector<c10::cuda::CUDAStream>& streams,
    const c10::cuda::CUDAStream caller_stream
) {
    join_worker_streams(streams, caller_stream);
    at::cuda::CUDAEvent all_workers_complete(cudaEventDisableTiming);
    all_workers_complete.record(caller_stream);
    for (const auto& stream : streams) {
        all_workers_complete.block(stream);
    }
}

}  // namespace

std::vector<std::vector<torch::Tensor>>
eiel_cuda_architecture_b_generation_multicell_v2(
    EIEL_ARCH_B_GENERATION_BATCH_ARGUMENTS(
        EIEL_DECLARE_TENSOR,
        EIEL_DECLARE_CONST_LIST,
        EIEL_DECLARE_LIST
    )
    EIEL_ARCH_B_GENERATION_SHARED_SCALARS(EIEL_DECLARE_SCALAR)
    bool env_policy_alias,
    int64_t max_streams
) {
    TORCH_CHECK(
        agent_ids_by_alive_order.defined(),
        "agent_ids_by_alive_order must be defined"
    );
    TORCH_CHECK(
        agent_ids_by_alive_order.is_cuda(),
        "multicell Architecture B generation requires CUDA tensors"
    );
    TORCH_CHECK(
        agent_ids_by_alive_order.dim() >= 1,
        "agent_ids_by_alive_order must have a leading cell dimension"
    );
    const int64_t cell_count = agent_ids_by_alive_order.size(0);
    TORCH_CHECK(cell_count >= 2, "multicell Architecture B generation requires at least 2 cells");
    TORCH_CHECK(max_streams > 0, "max_streams must be positive");

    const c10::Device device = agent_ids_by_alive_order.device();
#define EIEL_CHECK_BATCHED_TENSOR(name) \
    check_batched_tensor(name, #name, cell_count, device);
#define EIEL_CHECK_BATCHED_LIST(name) \
    check_batched_tensor_list(name, #name, cell_count, device);
    EIEL_ARCH_B_GENERATION_BATCH_ARGUMENTS(
        EIEL_CHECK_BATCHED_TENSOR,
        EIEL_CHECK_BATCHED_LIST,
        EIEL_CHECK_BATCHED_LIST
    )
#undef EIEL_CHECK_BATCHED_TENSOR
#undef EIEL_CHECK_BATCHED_LIST

    // Mirror generation_v2's top-level checks before any mutable cell dispatch.
    TORCH_CHECK(steps > 0, "Architecture B generation steps must be positive");
    TORCH_CHECK(
        agent_ids_by_alive_order.dim() == 2,
        "batched agent_ids_by_alive_order must be rank 2"
    );
    TORCH_CHECK(energy.dim() == 2, "batched energy must be a rank-2 slot bank");
    TORCH_CHECK(
        event_tape_columns.size() == 22,
        "generation event tape requires 6 legacy and 16 typed columns"
    );
    TORCH_CHECK(max_births_per_tick >= 0, "max_births_per_tick must be non-negative");
    TORCH_CHECK(
        !enable_biomes,
        "Architecture B resident food spawn does not support biome weighting"
    );
    TORCH_CHECK(
        !birth_enabled || resident_lifecycle_state_enabled,
        "birth path requires resident lifecycle state"
    );
    TORCH_CHECK(
        !generation_turnover_enabled || resident_lifecycle_state_enabled,
        "turnover path requires resident lifecycle state"
    );
    TORCH_CHECK(
        metrics_np_semantic_draws.scalar_type() == torch::kInt64
            && metrics_np_semantic_draws.dim() == 2
            && metrics_np_semantic_draws.size(1) == 1,
        "batched metrics_np_semantic_draws must be CUDA int64[cells,1]"
    );
    if (resident_lifecycle_state_enabled) {
        TORCH_CHECK(
            birth_identity_state.size() == 35,
            "birth identity state must contain 35 tensors"
        );
        TORCH_CHECK(
            birth_identity_state.at(0).is_alias_of(slot_to_agent_id),
            "birth slot identity must alias generation state"
        );
        TORCH_CHECK(
            birth_identity_state.at(1).is_alias_of(agent_id_to_slot),
            "birth agent map must alias generation state"
        );
        TORCH_CHECK(
            birth_identity_state.at(3).is_alias_of(alive_mask),
            "birth alive mask must alias generation state"
        );
        TORCH_CHECK(
            birth_identity_state.at(26).dim() >= 2
                && birth_identity_state.at(26).size(1) >= energy.size(1),
            "birth free queue must cover slot capacity"
        );
        TORCH_CHECK(
            birth_planner_pcg64_state_words.is_alias_of(pcg64_state_words)
                && birth_planner_pcg64_buffer_words.is_alias_of(pcg64_buffer_words),
            "birth planner must alias the shared env/policy PCG64 state"
        );
    }
    if (birth_enabled || generation_turnover_enabled) {
        TORCH_CHECK(
            birth_scalar_rng_state.size() == 8,
            "birth scalar RNG state must contain 8 tensors"
        );
        TORCH_CHECK(
            birth_genome_int32_fields.size() == 8,
            "birth integer genome state must contain 8 tensors"
        );
        TORCH_CHECK(
            birth_genome_float32_fields.size() == 24,
            "birth float genome state must contain 24 tensors"
        );
        TORCH_CHECK(
            birth_drive_state.size() == 6,
            "birth drive state must contain 6 tensors"
        );
        TORCH_CHECK(
            turnover_state.size() == 18,
            "birth lifecycle state must contain 18 turnover/stat tensors"
        );
    }
    if (birth_enabled) {
        TORCH_CHECK(
            max_births_per_tick > 0,
            "enabled birth path requires positive max_births_per_tick"
        );
        TORCH_CHECK(
            birth_policy_state.size() == 9,
            "birth policy state must contain 9 tensors"
        );
    }
    if (generation_turnover_enabled) {
        TORCH_CHECK(population_size > 0, "turnover population_size must be positive");
        TORCH_CHECK(
            cull_fraction >= 0.0 && cull_fraction <= 1.0,
            "turnover cull_fraction must be in [0, 1]"
        );
        TORCH_CHECK(
            parent_fraction >= 0.0 && parent_fraction <= 1.0,
            "turnover parent_fraction must be in [0, 1]"
        );
        TORCH_CHECK(
            turnover_policy_state.size() == 10,
            "turnover policy state must contain 10 tensors"
        );
        TORCH_CHECK(
            active_mechanics_enabled,
            "turnover requires active lifecycle mechanics"
        );
    }
    if (active_mechanics_enabled) {
        TORCH_CHECK(
            chemistry.scalar_type() == torch::kFloat32
                && chemistry.dim() == 3
                && chemistry.size(1) == energy.size(1)
                && chemistry.size(2) >= 6,
            "batched active generation chemistry must be CUDA float32[cells,capacity,>=6]"
        );
        TORCH_CHECK(
            runner_pcg64_state_words.scalar_type() == torch::kUInt64
                && select_cell(runner_pcg64_state_words, 0).numel() == 4,
            "batched active generation runner PCG64 state must be CUDA uint64[cells,4]"
        );
        TORCH_CHECK(
            runner_pcg64_buffer_words.scalar_type() == torch::kInt64
                && select_cell(runner_pcg64_buffer_words, 0).numel() == 2,
            "batched active generation runner PCG64 buffer must be CUDA int64[cells,2]"
        );
        TORCH_CHECK(
            pcg64_state_words.scalar_type() == torch::kUInt64
                && select_cell(pcg64_state_words, 0).numel() == 4
                && pcg64_buffer_words.scalar_type() == torch::kInt64
                && select_cell(pcg64_buffer_words, 0).numel() == 2,
            "active generation chemistry requires batched shared env/policy PCG64 state"
        );
        TORCH_CHECK(
            drive_noise_samples_by_tick_slot.dim() == 4
                && drive_noise_samples_by_tick_slot.size(1) >= start_tick + steps,
            "batched drive_noise_samples_by_tick_slot must provide every generation tick"
        );
        TORCH_CHECK(
            drive_noise_samples_valid_by_tick_slot.dim() == 3
                && drive_noise_samples_valid_by_tick_slot.size(1) >= start_tick + steps,
            "batched drive_noise_samples_valid_by_tick_slot must provide every generation tick"
        );
    }

    const int64_t capacity = energy.size(1);
    for (const auto& tensor : {
             last_action,
             last_last_action,
             action_switches,
             action_oscillations,
             episode0_switches,
             episode0_oscillations,
         }) {
        TORCH_CHECK(
            tensor.scalar_type() == torch::kInt32
                && tensor.dim() == 2
                && tensor.size(1) == capacity,
            "batched action-history tensors must be CUDA int32[cells,capacity]"
        );
    }
    TORCH_CHECK(
        agent_ids_by_alive_order.size(1) <= capacity,
        "agent_ids_by_alive_order exceeds the fixed slot capacity"
    );

    c10::cuda::CUDAGuard device_guard(device);
    const auto caller_stream = c10::cuda::getCurrentCUDAStream(device.index());
    const int64_t stream_count = std::min(
        {cell_count, max_streams, CUDA_LOW_PRIORITY_POOL_SIZE}
    );
    std::vector<c10::cuda::CUDAStream> worker_streams;
    worker_streams.reserve(static_cast<std::size_t>(stream_count));
    for (int64_t stream_index = 0; stream_index < stream_count; ++stream_index) {
        worker_streams.push_back(
            c10::cuda::getStreamFromPool(false, device.index())
        );
    }

    at::cuda::CUDAEvent inputs_ready(cudaEventDisableTiming);
    inputs_ready.record(caller_stream);
    for (const auto& worker_stream : worker_streams) {
        inputs_ready.block(worker_stream);
#define EIEL_RECORD_BATCHED_TENSOR(name) \
        record_tensor_on_stream(name, worker_stream);
#define EIEL_RECORD_BATCHED_LIST(name) \
        record_tensor_list_on_stream(name, worker_stream);
        EIEL_ARCH_B_GENERATION_BATCH_ARGUMENTS(
            EIEL_RECORD_BATCHED_TENSOR,
            EIEL_RECORD_BATCHED_LIST,
            EIEL_RECORD_BATCHED_LIST
        )
#undef EIEL_RECORD_BATCHED_TENSOR
#undef EIEL_RECORD_BATCHED_LIST
    }

    using eiel::cuda_architecture_b::GenerationContext;
    using eiel::cuda_architecture_b::GenerationActiveMechanicsBatchView;
    using eiel::cuda_architecture_b::GenerationPrePolicyBatchView;
    using eiel::cuda_architecture_b::GenerationResidualLifecycleBatchView;
    std::vector<std::unique_ptr<GenerationContext>> contexts;
    contexts.reserve(static_cast<std::size_t>(cell_count));
    std::vector<std::vector<torch::Tensor>> outputs(
        static_cast<std::size_t>(cell_count)
    );
    try {
        for (int64_t cell_index = 0; cell_index < cell_count; ++cell_index) {
            const auto& worker_stream = worker_streams.at(
                static_cast<std::size_t>(cell_index % stream_count)
            );
            c10::cuda::CUDAStreamGuard stream_guard(worker_stream);
#define EIEL_SELECT_BATCHED_TENSOR(name) select_cell(name, cell_index),
#define EIEL_SELECT_BATCHED_LIST(name) select_cell_list(name, cell_index),
#define EIEL_FORWARD_SCALAR(type, name) name,
            contexts.push_back(GenerationContext::prepare({
                EIEL_ARCH_B_GENERATION_BATCH_ARGUMENTS(
                    EIEL_SELECT_BATCHED_TENSOR,
                    EIEL_SELECT_BATCHED_LIST,
                    EIEL_SELECT_BATCHED_LIST
                )
                EIEL_ARCH_B_GENERATION_SHARED_SCALARS(EIEL_FORWARD_SCALAR)
                env_policy_alias,
            }));
#undef EIEL_SELECT_BATCHED_TENSOR
#undef EIEL_SELECT_BATCHED_LIST
#undef EIEL_FORWARD_SCALAR
        }

        for (int64_t tick = 0; tick < steps; ++tick) {
            std::vector<GenerationPrePolicyBatchView> pre_policy_views;
            pre_policy_views.reserve(static_cast<std::size_t>(cell_count));
            for (int64_t cell_index = 0; cell_index < cell_count; ++cell_index) {
                const auto& worker_stream = worker_streams.at(
                    static_cast<std::size_t>(cell_index % stream_count)
                );
                c10::cuda::CUDAStreamGuard stream_guard(worker_stream);
                contexts.at(static_cast<std::size_t>(cell_index))
                    ->execute_pre_policy_setup(tick);
                pre_policy_views.push_back(
                    contexts.at(static_cast<std::size_t>(cell_index))
                        ->pre_policy_batch_view()
                );
            }

            join_worker_streams(worker_streams, caller_stream);
            c10::cuda::CUDAStreamGuard caller_guard(caller_stream);
            std::vector<torch::Tensor> active_agent_id_rows;
            std::vector<torch::Tensor> active_count_rows;
            std::vector<torch::Tensor> slot_rows;
            active_agent_id_rows.reserve(static_cast<std::size_t>(cell_count));
            active_count_rows.reserve(static_cast<std::size_t>(cell_count));
            slot_rows.reserve(static_cast<std::size_t>(cell_count));
            for (const auto& view : pre_policy_views) {
                record_tensor_on_stream(view.active_agent_ids, caller_stream);
                record_tensor_on_stream(view.active_count, caller_stream);
                record_tensor_on_stream(view.slots, caller_stream);
                active_agent_id_rows.push_back(view.active_agent_ids);
                active_count_rows.push_back(view.active_count);
                slot_rows.push_back(view.slots);
            }
            auto batched_active_agent_ids = torch::stack(
                active_agent_id_rows,
                0
            ).contiguous();
            auto batched_active_counts = torch::stack(
                active_count_rows,
                0
            ).contiguous();
            auto batched_slots = torch::stack(slot_rows, 0).contiguous();
            auto policy_cache_outputs =
                eiel_cuda_architecture_b_policy_cache_service_batched_v1(
                    batched_active_agent_ids,
                    batched_active_counts,
                    batched_slots,
                    slot_tokens,
                    action_cache_actions,
                    action_cache_token,
                    action_cache_cursor,
                    action_cache_next,
                    cache_width
                );
            at::cuda::CUDAEvent cache_complete(cudaEventDisableTiming);
            cache_complete.record(caller_stream);
            for (const auto& worker_stream : worker_streams) {
                cache_complete.block(worker_stream);
            }
            for (int64_t cell_index = 0; cell_index < cell_count; ++cell_index) {
                const auto& worker_stream = worker_streams.at(
                    static_cast<std::size_t>(cell_index % stream_count)
                );
                c10::cuda::CUDAStreamGuard stream_guard(worker_stream);
                std::vector<torch::Tensor> cell_cache_outputs;
                cell_cache_outputs.reserve(policy_cache_outputs.size());
                for (const auto& output : policy_cache_outputs) {
                    cell_cache_outputs.push_back(select_cell(output, cell_index));
                }
                record_tensor_list_on_stream(cell_cache_outputs, worker_stream);
                contexts.at(static_cast<std::size_t>(cell_index))
                    ->accept_policy_cache_batch_outputs(tick, cell_cache_outputs);
            }

            join_worker_streams(worker_streams, caller_stream);
            auto batched_observations =
                eiel_cuda_owner_observation_frame_batched_counted_v1(
                    policy_cache_outputs.at(0),
                    policy_cache_outputs.at(4),
                    agent_id_to_slot,
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
            auto policy_outputs = eiel_cuda_fixed_mlp_refill_batched_counted_v1(
                batched_observations,
                policy_cache_outputs.at(1),
                policy_cache_outputs.at(4),
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
            at::cuda::CUDAEvent policy_complete(cudaEventDisableTiming);
            policy_complete.record(caller_stream);
            for (const auto& worker_stream : worker_streams) {
                policy_complete.block(worker_stream);
            }

            for (int64_t cell_index = 0; cell_index < cell_count; ++cell_index) {
                const auto& worker_stream = worker_streams.at(
                    static_cast<std::size_t>(cell_index % stream_count)
                );
                c10::cuda::CUDAStreamGuard stream_guard(worker_stream);
                std::vector<torch::Tensor> cell_policy_outputs;
                cell_policy_outputs.reserve(policy_outputs.size());
                for (const auto& output : policy_outputs) {
                    cell_policy_outputs.push_back(select_cell(output, cell_index));
                }
                record_tensor_list_on_stream(cell_policy_outputs, worker_stream);
                auto& context = contexts.at(static_cast<std::size_t>(cell_index));
                context->accept_observations(
                    select_cell(batched_observations, cell_index)
                );
                context->accept_policy_batch_outputs(tick, cell_policy_outputs);
                context->execute_post_policy_before_action_lifecycle(tick);
            }

            join_worker_streams(worker_streams, caller_stream);
            auto action_lifecycle_outputs =
                eiel_cuda_architecture_b_lifecycle_death_cull_batched_counted_v1(
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
            at::cuda::CUDAEvent action_lifecycle_complete(
                cudaEventDisableTiming
            );
            action_lifecycle_complete.record(caller_stream);
            for (const auto& worker_stream : worker_streams) {
                action_lifecycle_complete.block(worker_stream);
            }

            std::vector<GenerationActiveMechanicsBatchView> mechanics_views;
            mechanics_views.reserve(static_cast<std::size_t>(cell_count));
            for (int64_t cell_index = 0; cell_index < cell_count; ++cell_index) {
                const auto& worker_stream = worker_streams.at(
                    static_cast<std::size_t>(cell_index % stream_count)
                );
                c10::cuda::CUDAStreamGuard stream_guard(worker_stream);
                std::vector<torch::Tensor> cell_lifecycle_outputs;
                cell_lifecycle_outputs.reserve(action_lifecycle_outputs.size());
                for (const auto& output : action_lifecycle_outputs) {
                    cell_lifecycle_outputs.push_back(
                        select_cell(output, cell_index)
                    );
                }
                record_tensor_list_on_stream(
                    cell_lifecycle_outputs,
                    worker_stream
                );
                auto& context = contexts.at(
                    static_cast<std::size_t>(cell_index)
                );
                context->accept_action_lifecycle_outputs(
                    tick,
                    cell_lifecycle_outputs
                );
                mechanics_views.push_back(
                    context->active_mechanics_batch_view()
                );
            }

            std::vector<GenerationResidualLifecycleBatchView> residual_views;
            residual_views.reserve(static_cast<std::size_t>(cell_count));
            if (
                active_mechanics_enabled &&
                cell_count >= kActiveMechanicsBatchMinCells
            ) {
            join_worker_streams(worker_streams, caller_stream);
            std::vector<torch::Tensor> actions_by_slot_rows;
            std::vector<torch::Tensor> pre_energy_rows;
            std::vector<torch::Tensor> pre_health_rows;
            std::vector<torch::Tensor> pre_injury_rows;
            actions_by_slot_rows.reserve(static_cast<std::size_t>(cell_count));
            pre_energy_rows.reserve(static_cast<std::size_t>(cell_count));
            pre_health_rows.reserve(static_cast<std::size_t>(cell_count));
            pre_injury_rows.reserve(static_cast<std::size_t>(cell_count));
            for (const auto& view : mechanics_views) {
                record_tensor_on_stream(view.actions_by_slot, caller_stream);
                record_tensor_on_stream(view.pre_energy, caller_stream);
                record_tensor_on_stream(view.pre_health, caller_stream);
                record_tensor_on_stream(view.pre_injury, caller_stream);
                actions_by_slot_rows.push_back(view.actions_by_slot);
                pre_energy_rows.push_back(view.pre_energy);
                pre_health_rows.push_back(view.pre_health);
                pre_injury_rows.push_back(view.pre_injury);
            }
            auto batched_actions_by_slot = torch::stack(
                actions_by_slot_rows,
                0
            ).contiguous();
            auto batched_pre_energy = torch::stack(pre_energy_rows, 0).contiguous();
            auto batched_pre_health = torch::stack(pre_health_rows, 0).contiguous();
            auto batched_pre_injury = torch::stack(pre_injury_rows, 0).contiguous();
            std::vector<torch::Tensor> mechanics_outputs;
            if (active_mechanics_enabled) {
                mechanics_outputs =
                    eiel_cuda_architecture_b_active_mechanics_batched_counted_v1(
                        slot_to_agent_id,
                        alive_mask,
                        batched_actions_by_slot,
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
                        batched_pre_energy,
                        batched_pre_health,
                        batched_pre_injury,
                        drive_noise_samples_by_tick_slot.select(
                            1,
                            start_tick + tick
                        ).contiguous(),
                        drive_noise_samples_valid_by_tick_slot.select(
                            1,
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
            at::cuda::CUDAEvent mechanics_complete(cudaEventDisableTiming);
            mechanics_complete.record(caller_stream);
            for (const auto& worker_stream : worker_streams) {
                mechanics_complete.block(worker_stream);
            }

            for (int64_t cell_index = 0; cell_index < cell_count; ++cell_index) {
                const auto& worker_stream = worker_streams.at(
                    static_cast<std::size_t>(cell_index % stream_count)
                );
                c10::cuda::CUDAStreamGuard stream_guard(worker_stream);
                std::vector<torch::Tensor> cell_mechanics_outputs;
                cell_mechanics_outputs.reserve(mechanics_outputs.size());
                for (const auto& output : mechanics_outputs) {
                    cell_mechanics_outputs.push_back(
                        select_cell(output, cell_index)
                    );
                }
                record_tensor_list_on_stream(
                    cell_mechanics_outputs,
                    worker_stream
                );
                auto& context = contexts.at(
                    static_cast<std::size_t>(cell_index)
                );
                context->accept_active_mechanics_batch_outputs(
                    tick,
                    cell_mechanics_outputs
                );
                residual_views.push_back(
                    context->residual_lifecycle_batch_view()
                );
            }
            } else {
                for (int64_t cell_index = 0; cell_index < cell_count; ++cell_index) {
                    const auto& worker_stream = worker_streams.at(
                        static_cast<std::size_t>(cell_index % stream_count)
                    );
                    c10::cuda::CUDAStreamGuard stream_guard(worker_stream);
                    auto& context = contexts.at(
                        static_cast<std::size_t>(cell_index)
                    );
                    context->execute_single_cell_active_mechanics(tick);
                    residual_views.push_back(
                        context->residual_lifecycle_batch_view()
                    );
                }
            }

            join_worker_streams(worker_streams, caller_stream);
            std::vector<torch::Tensor> residual_agent_id_rows;
            std::vector<torch::Tensor> residual_slot_rows;
            std::vector<torch::Tensor> residual_count_rows;
            residual_agent_id_rows.reserve(static_cast<std::size_t>(cell_count));
            residual_slot_rows.reserve(static_cast<std::size_t>(cell_count));
            residual_count_rows.reserve(static_cast<std::size_t>(cell_count));
            for (const auto& view : residual_views) {
                record_tensor_on_stream(
                    view.pending_free_agent_ids,
                    caller_stream
                );
                record_tensor_on_stream(view.pending_free_slots, caller_stream);
                record_tensor_on_stream(view.pending_free_count, caller_stream);
                residual_agent_id_rows.push_back(view.pending_free_agent_ids);
                residual_slot_rows.push_back(view.pending_free_slots);
                residual_count_rows.push_back(view.pending_free_count);
            }
            auto batched_residual_agent_ids = torch::stack(
                residual_agent_id_rows,
                0
            ).contiguous();
            auto batched_residual_slots = torch::stack(
                residual_slot_rows,
                0
            ).contiguous();
            auto batched_residual_counts = torch::stack(
                residual_count_rows,
                0
            ).contiguous();
            auto residual_lifecycle_outputs =
                eiel_cuda_architecture_b_lifecycle_death_cull_batched_counted_v1(
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
                    batched_residual_agent_ids,
                    batched_residual_slots,
                    batched_residual_counts
                );
            at::cuda::CUDAEvent residual_lifecycle_complete(
                cudaEventDisableTiming
            );
            residual_lifecycle_complete.record(caller_stream);
            for (const auto& worker_stream : worker_streams) {
                residual_lifecycle_complete.block(worker_stream);
            }
            for (int64_t cell_index = 0; cell_index < cell_count; ++cell_index) {
                const auto& worker_stream = worker_streams.at(
                    static_cast<std::size_t>(cell_index % stream_count)
                );
                c10::cuda::CUDAStreamGuard stream_guard(worker_stream);
                std::vector<torch::Tensor> cell_lifecycle_outputs;
                cell_lifecycle_outputs.reserve(residual_lifecycle_outputs.size());
                for (const auto& output : residual_lifecycle_outputs) {
                    cell_lifecycle_outputs.push_back(
                        select_cell(output, cell_index)
                    );
                }
                record_tensor_list_on_stream(
                    cell_lifecycle_outputs,
                    worker_stream
                );
                auto& context = contexts.at(
                    static_cast<std::size_t>(cell_index)
                );
                context->accept_residual_lifecycle_batch_state(
                    select_cell(batched_residual_agent_ids, cell_index),
                    select_cell(batched_residual_slots, cell_index),
                    select_cell(batched_residual_counts, cell_index)
                );
                context->accept_residual_lifecycle_outputs(
                    tick,
                    cell_lifecycle_outputs
                );
            }
            close_tick_barrier(worker_streams, caller_stream);
        }

        for (int64_t cell_index = 0; cell_index < cell_count; ++cell_index) {
            const auto& worker_stream = worker_streams.at(
                static_cast<std::size_t>(cell_index % stream_count)
            );
            c10::cuda::CUDAStreamGuard stream_guard(worker_stream);
            outputs.at(static_cast<std::size_t>(cell_index)) =
                contexts.at(static_cast<std::size_t>(cell_index))->finalize();
            record_tensor_list_on_stream(
                outputs.at(static_cast<std::size_t>(cell_index)),
                caller_stream
            );
        }
    } catch (...) {
        join_worker_streams(worker_streams, caller_stream);
        throw;
    }
    join_worker_streams(worker_streams, caller_stream);
    return outputs;
}

void eiel_cuda_architecture_b_bind_generation_multicell(
    pybind11::module_& module
) {
    module.def(
        "eiel_cuda_architecture_b_generation_multicell_v2",
        &eiel_cuda_architecture_b_generation_multicell_v2
    );
}

#undef EIEL_DECLARE_TENSOR
#undef EIEL_DECLARE_CONST_LIST
#undef EIEL_DECLARE_LIST
#undef EIEL_DECLARE_SCALAR
#undef EIEL_ARCH_B_GENERATION_BATCH_ARGUMENTS
#undef EIEL_ARCH_B_GENERATION_SHARED_SCALARS
