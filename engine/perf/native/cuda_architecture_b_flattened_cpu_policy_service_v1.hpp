#pragma once

#include <torch/torch.h>

#include <cstdint>

#ifndef EIEL_OPTION_D_M0_TELEMETRY_ENABLED
#define EIEL_OPTION_D_M0_TELEMETRY_ENABLED 0
#endif

#if EIEL_OPTION_D_M0_TELEMETRY_ENABLED != 0 \
    && EIEL_OPTION_D_M0_TELEMETRY_ENABLED != 1
#error "EIEL_OPTION_D_M0_TELEMETRY_ENABLED must be 0 or 1"
#endif

namespace eiel::cuda_architecture_b::flattened_cpu_policy_service_v1 {

inline constexpr bool kOptionDM0TelemetryEnabledV1 =
    EIEL_OPTION_D_M0_TELEMETRY_ENABLED == 1;

struct FlattenedPatchFixedMlpCpuExactArgsV1 {
    torch::Tensor ordered_slots;
    torch::Tensor observations;
    torch::Tensor death_slots_miss_rows;
    torch::Tensor active_count;
    torch::Tensor policy_refill_rows;
    torch::Tensor policy_cache_hit_rows;
    torch::Tensor status;
    // Optional [cells] CUDA bool plane. True means the native tick owner
    // already supplied the authoritative action and policy cache accounting
    // does not apply to that cell.
    torch::Tensor policy_service_bypass_by_cell;

    torch::Tensor mlp_pack;
    torch::Tensor drives;
    torch::Tensor attack_bias;
    torch::Tensor action_temperature;
    torch::Tensor drive_logit_gain;
    torch::Tensor slot_tokens;

    torch::Tensor action_cache_actions;
    torch::Tensor action_cache_token;
    torch::Tensor action_cache_cursor;
    torch::Tensor action_cache_next;
    torch::Tensor logits;
    torch::Tensor row_actions;
    torch::Tensor actions_by_slot;
    torch::Tensor policy_pcg64_state_words;
    torch::Tensor policy_pcg64_buffer_words;
    torch::Tensor policy_np_semantic_draws;

    int64_t cells;
    int64_t controller_tick;
    int64_t output_tick;
    int64_t capacity;
    int64_t observation_size;
    int64_t hidden_size;
    int64_t action_count;
    int64_t packed_width;
    int64_t drive_width;
    int64_t cache_capacity;
    int64_t cache_width;
    bool deterministic_actions;

    double w_hunger_gather;
    double w_pain_rest;
    double w_fatigue_rest;
    double w_pain_attack;
    double w_fear_attack;
};

struct FlattenedPatchFixedMlpCpuExactReceiptV1 {
    int64_t controller_tick;
    int64_t output_tick;
    int64_t cells_serviced;
    int64_t active_rows;
    int64_t cache_hit_rows;
    int64_t refill_rows;
    int64_t gumbel_draws;
    int64_t device_to_host_operations;
    int64_t host_to_device_operations;
    int64_t aten_operations_lower_bound;
    int64_t host_synchronizations;
    bool option_d_m0_telemetry_enabled;
    int64_t host_wall_time_ns;
    int64_t device_to_host_time_ns;
    int64_t host_to_device_time_ns;
    int64_t cpu_mlp_time_ns;
    int64_t cpu_sampling_time_ns;
    int64_t final_stream_synchronization_time_ns;
    int64_t device_to_host_bytes_lower_bound;
    int64_t host_to_device_bytes_lower_bound;
    int64_t host_allocation_count_lower_bound;
    int64_t host_allocation_bytes_lower_bound;
};

FlattenedPatchFixedMlpCpuExactReceiptV1
service_flattened_patch_fixed_mlp_cpu_exact_v1(
    const FlattenedPatchFixedMlpCpuExactArgsV1& arguments
);

}  // namespace eiel::cuda_architecture_b::flattened_cpu_policy_service_v1
