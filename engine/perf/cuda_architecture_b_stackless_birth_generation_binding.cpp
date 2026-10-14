#include "cuda_architecture_b_stackless_birth_generation.hpp"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include "../native_runtime/authoritative_state/binary_identity_constants_v1.hpp"

#include <cmath>
#include <cstdint>
#include <tuple>
#include <utility>
#include <vector>

namespace {

void check_cuda_contiguous(
    const torch::Tensor& reference,
    const torch::Tensor& tensor,
    const char* name
) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be CUDA resident");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(tensor.device() == reference.device(), name, " must share one device");
}

void stackless_flat_dynamic_policy_food_birth_full_tick_v1(
    torch::Tensor active_agent_ids,
    torch::Tensor active_count_state,
    torch::Tensor slot_to_agent_id,
    torch::Tensor agent_id_to_slot,
    torch::Tensor alive_mask,
    torch::Tensor slot_tokens,
    torch::Tensor action_cache_actions,
    torch::Tensor action_cache_token,
    torch::Tensor action_cache_cursor,
    torch::Tensor action_cache_next,
    torch::Tensor pos_y,
    torch::Tensor pos_x,
    torch::Tensor agent_grid,
    torch::Tensor food_grid,
    torch::Tensor energy,
    torch::Tensor health,
    torch::Tensor injury,
    torch::Tensor policy_energy_observation,
    torch::Tensor rest_streak,
    torch::Tensor pending_free_agent_ids,
    torch::Tensor pending_free_slots,
    torch::Tensor pending_free_count,
    torch::Tensor energy_collected,
    torch::Tensor energy_collected_exact,
    torch::Tensor last_action,
    torch::Tensor last_last_action,
    torch::Tensor action_switches,
    torch::Tensor action_oscillations,
    torch::Tensor episode0_switches,
    torch::Tensor episode0_oscillations,
    torch::Tensor survival_steps,
    torch::Tensor mlp_pack,
    torch::Tensor drives,
    torch::Tensor attack_bias,
    torch::Tensor action_temperature,
    torch::Tensor drive_logit_gain,
    torch::Tensor food_gained_scratch,
    torch::Tensor ordered_agent_ids,
    torch::Tensor ordered_slots,
    torch::Tensor observations,
    torch::Tensor logits,
    torch::Tensor row_actions,
    torch::Tensor actions_by_slot,
    torch::Tensor pre_energy,
    torch::Tensor pre_health,
    torch::Tensor pre_injury,
    torch::Tensor pre_survival_steps,
    torch::Tensor active_count,
    torch::Tensor policy_refill_rows,
    torch::Tensor policy_cache_hit_rows,
    torch::Tensor death_agent_ids,
    torch::Tensor death_slots,
    torch::Tensor death_reason_codes,
    torch::Tensor death_count,
    torch::Tensor survivor_agent_ids,
    torch::Tensor survivor_count,
    torch::Tensor lifecycle_status,
    torch::Tensor status,
    std::vector<torch::Tensor> full_tick_state_tensors,
    std::vector<torch::Tensor> full_tick_output_tensors,
    std::vector<torch::Tensor> event_state_tensors,
    std::vector<torch::Tensor> ordinary_birth_state_tensors,
    std::vector<torch::Tensor> ordinary_birth_output_tensors,
    std::vector<torch::Tensor> ordinary_birth_scratch_tensors,
    const int64_t ticks,
    const int64_t view_radius,
    const int64_t hidden_size,
    const int64_t action_count,
    const int64_t cache_width,
    const bool deterministic_actions,
    const int64_t max_food,
    const double food_spawn_per_step,
    const double food_spawn_mult,
    const double max_energy,
    const double start_health,
    const double move_cost,
    const double gather_cost,
    const double rest_cost,
    const double attack_cost,
    const double action_cost_mult,
    const double rest_cost_mult,
    const double food_energy,
    const int64_t episode0_k,
    const double w_hunger_gather,
    const double w_pain_rest,
    const double w_fatigue_rest,
    const double w_pain_attack,
    const double w_fear_attack,
    const bool enable_rest_charge,
    const int64_t rest_streak_free,
    const double rest_charge_energy,
    const double rest_charge_growth,
    const double rest_deficit_to_health,
    const int64_t developmental_maturity_age,
    const double drive_hunger_gain,
    const double drive_hunger_decay,
    const double drive_pain_gain,
    const double drive_pain_decay,
    const double drive_fatigue_gain,
    const double drive_fatigue_rest_relief,
    const double drive_fatigue_decay,
    const double drive_fear_gain,
    const double drive_fear_decay,
    const int64_t generation,
    const int64_t start_tick,
    const int64_t max_births_per_tick,
    const double reproduce_parent_energy_share,
    const double reproduce_min_energy_buffer,
    const int64_t reproduce_cooldown_steps,
    const int64_t genome_sensor_radius_max,
    const double injury_repro_k_injury,
    const double injury_repro_k_scar,
    const double injury_birth_fail_scar,
    const double injury_birth_fail_injury,
    const bool env_policy_alias,
    const bool reproduce_requires_mate
) {
    TORCH_CHECK(ticks > 0 && ticks <= 80, "persistent ticks must be in [1, 80]");
    TORCH_CHECK(view_radius >= 0, "view_radius must be non-negative");
    TORCH_CHECK(hidden_size > 0 && hidden_size <= 64, "hidden_size must be in [1, 64]");
    TORCH_CHECK(action_count >= 7 && action_count <= 16, "action_count must be in [7, 16]");
    TORCH_CHECK(cache_width == 2, "bounded persistent cache_width must equal 2");
    TORCH_CHECK(max_food >= 0, "max_food must be non-negative");
    TORCH_CHECK(food_spawn_per_step >= 0.0, "food_spawn_per_step must be non-negative");
    TORCH_CHECK(food_spawn_mult >= 0.0, "food_spawn_mult must be non-negative");
    TORCH_CHECK(max_energy != 0.0 && start_health != 0.0, "observation denominators must be non-zero");
    TORCH_CHECK(episode0_k >= 0, "episode0_k must be non-negative");
    TORCH_CHECK(rest_streak_free >= 0, "rest_streak_free must be non-negative");
    TORCH_CHECK(developmental_maturity_age >= 0, "maturity age must be non-negative");
    TORCH_CHECK(generation >= 0 && start_tick >= 0, "generation/tick must be non-negative");
    TORCH_CHECK(
        max_births_per_tick == 0 || max_births_per_tick == 1,
        "bounded persistent ordinary births must be disabled or one per tick"
    );
    TORCH_CHECK(
        full_tick_state_tensors.size() == 29,
        "full_tick_state_tensors must contain 29 tensors"
    );
    TORCH_CHECK(
        full_tick_output_tensors.size() == 18,
        "full_tick_output_tensors must contain 18 tensors"
    );
    TORCH_CHECK(
        event_state_tensors.size() == 30,
        "event_state_tensors must contain 30 tensors"
    );
    const bool ordinary_births_enabled = max_births_per_tick > 0;
    TORCH_CHECK(
        ordinary_birth_state_tensors.size()
            == (ordinary_births_enabled ? 110 : 0),
        "ordinary_birth_state_tensors must be empty when births are disabled "
        "and contain 110 tensors when enabled"
    );
    TORCH_CHECK(
        ordinary_birth_output_tensors.size()
            == (ordinary_births_enabled ? 47 : 0),
        "ordinary_birth_output_tensors must be empty when births are disabled "
        "and contain 47 tensors when enabled"
    );
    TORCH_CHECK(
        ordinary_birth_scratch_tensors.size()
            == (ordinary_births_enabled ? 1 : 0),
        "ordinary_birth_scratch_tensors must be empty when births are disabled "
        "and contain one tensor when enabled"
    );

    for (const auto& named : std::vector<std::pair<const char*, torch::Tensor>>{
             {"active_agent_ids", active_agent_ids},
             {"active_count_state", active_count_state},
             {"slot_to_agent_id", slot_to_agent_id},
             {"agent_id_to_slot", agent_id_to_slot},
             {"alive_mask", alive_mask},
             {"slot_tokens", slot_tokens},
             {"action_cache_actions", action_cache_actions},
             {"action_cache_token", action_cache_token},
             {"action_cache_cursor", action_cache_cursor},
             {"action_cache_next", action_cache_next},
             {"pos_y", pos_y},
             {"pos_x", pos_x},
             {"agent_grid", agent_grid},
             {"food_grid", food_grid},
             {"energy", energy},
             {"health", health},
             {"injury", injury},
             {"policy_energy_observation", policy_energy_observation},
             {"rest_streak", rest_streak},
             {"pending_free_agent_ids", pending_free_agent_ids},
             {"pending_free_slots", pending_free_slots},
             {"pending_free_count", pending_free_count},
             {"energy_collected", energy_collected},
             {"energy_collected_exact", energy_collected_exact},
             {"last_action", last_action},
             {"last_last_action", last_last_action},
             {"action_switches", action_switches},
             {"action_oscillations", action_oscillations},
             {"episode0_switches", episode0_switches},
             {"episode0_oscillations", episode0_oscillations},
             {"survival_steps", survival_steps},
             {"mlp_pack", mlp_pack},
             {"drives", drives},
             {"attack_bias", attack_bias},
             {"action_temperature", action_temperature},
             {"drive_logit_gain", drive_logit_gain},
             {"food_gained_scratch", food_gained_scratch},
             {"ordered_agent_ids", ordered_agent_ids},
             {"ordered_slots", ordered_slots},
             {"observations", observations},
             {"logits", logits},
             {"row_actions", row_actions},
             {"actions_by_slot", actions_by_slot},
             {"pre_energy", pre_energy},
             {"pre_health", pre_health},
             {"pre_injury", pre_injury},
             {"pre_survival_steps", pre_survival_steps},
             {"active_count", active_count},
             {"policy_refill_rows", policy_refill_rows},
             {"policy_cache_hit_rows", policy_cache_hit_rows},
             {"death_agent_ids", death_agent_ids},
             {"death_slots", death_slots},
             {"death_reason_codes", death_reason_codes},
             {"death_count", death_count},
             {"survivor_agent_ids", survivor_agent_ids},
             {"survivor_count", survivor_count},
             {"lifecycle_status", lifecycle_status},
             {"status", status},
         }) {
        check_cuda_contiguous(active_agent_ids, named.second, named.first);
    }
    for (const auto& tensor : full_tick_state_tensors) {
        check_cuda_contiguous(active_agent_ids, tensor, "full_tick_state_tensor");
    }
    for (const auto& tensor : full_tick_output_tensors) {
        check_cuda_contiguous(active_agent_ids, tensor, "full_tick_output_tensor");
    }
    for (const auto& tensor : event_state_tensors) {
        check_cuda_contiguous(active_agent_ids, tensor, "event_state_tensor");
    }
    for (const auto& tensor : ordinary_birth_state_tensors) {
        check_cuda_contiguous(
            active_agent_ids,
            tensor,
            "ordinary_birth_state_tensor"
        );
    }
    for (const auto& tensor : ordinary_birth_output_tensors) {
        check_cuda_contiguous(
            active_agent_ids,
            tensor,
            "ordinary_birth_output_tensor"
        );
    }
    for (const auto& tensor : ordinary_birth_scratch_tensors) {
        check_cuda_contiguous(
            active_agent_ids,
            tensor,
            "ordinary_birth_scratch_tensor"
        );
    }
    TORCH_CHECK(
        active_agent_ids.scalar_type() == torch::kInt64 && active_agent_ids.dim() == 2,
        "active_agent_ids must be int64[cells,active_capacity]"
    );
    const int64_t cells = active_agent_ids.size(0);
    const int64_t active_capacity = active_agent_ids.size(1);
    TORCH_CHECK(cells > 0 && active_capacity > 0, "persistent challenger requires non-empty cells and active capacity");
    if (ordinary_births_enabled) {
        TORCH_CHECK(
            ordinary_birth_scratch_tensors.at(0).scalar_type() == torch::kUInt32
                && ordinary_birth_scratch_tensors.at(0).sizes()
                    == torch::IntArrayRef({cells, 624}),
            "stackless CPython trial scratch must be uint32[cells,624]"
        );
    }
    TORCH_CHECK(
        active_count_state.scalar_type() == torch::kInt64
            && active_count_state.sizes() == torch::IntArrayRef({cells}),
        "active_count_state must be int64[cells]"
    );
    TORCH_CHECK(
        agent_id_to_slot.scalar_type() == torch::kInt32
            && agent_id_to_slot.dim() == 2
            && agent_id_to_slot.size(0) == cells,
        "agent_id_to_slot must be int32[cells,agent_id_capacity]"
    );
    TORCH_CHECK(
        energy.scalar_type() == torch::kFloat32
            && energy.dim() == 2
            && energy.size(0) == cells,
        "energy must be float32[cells,capacity]"
    );
    const int64_t capacity = energy.size(1);
    const int64_t agent_id_capacity = agent_id_to_slot.size(1);
    TORCH_CHECK(
        active_capacity <= capacity,
        "active_agent_ids capacity must not exceed slot capacity"
    );
    TORCH_CHECK(
        !ordinary_births_enabled || active_capacity == capacity,
        "birth-enabled active_agent_ids must cover the complete slot capacity"
    );
    TORCH_CHECK(
        generation <= INT32_MAX && start_tick <= INT32_MAX
            && ticks - 1 <= INT32_MAX - start_tick,
        "birth generation and absolute ticks must fit int32"
    );
    const auto check_cell_capacity = [&](const torch::Tensor& tensor,
                                         const char* name,
                                         const c10::ScalarType dtype,
                                         const int64_t rank) {
        TORCH_CHECK(
            tensor.scalar_type() == dtype
                && tensor.dim() == rank
                && tensor.size(0) == cells
                && tensor.size(1) == capacity,
            name,
            " has an invalid cell/capacity shape"
        );
    };
    check_cell_capacity(pos_y, "pos_y", torch::kInt32, 2);
    check_cell_capacity(pos_x, "pos_x", torch::kInt32, 2);
    check_cell_capacity(slot_to_agent_id, "slot_to_agent_id", torch::kInt64, 2);
    check_cell_capacity(alive_mask, "alive_mask", torch::kBool, 2);
    check_cell_capacity(health, "health", torch::kFloat32, 2);
    check_cell_capacity(injury, "injury", torch::kFloat32, 2);
    check_cell_capacity(rest_streak, "rest_streak", torch::kInt32, 2);
    check_cell_capacity(slot_tokens, "slot_tokens", torch::kInt64, 2);
    check_cell_capacity(action_cache_token, "action_cache_token", torch::kInt64, 2);
    check_cell_capacity(action_cache_cursor, "action_cache_cursor", torch::kInt32, 2);
    check_cell_capacity(action_cache_next, "action_cache_next", torch::kInt32, 2);
    TORCH_CHECK(
        action_cache_actions.scalar_type() == torch::kInt32
            && action_cache_actions.dim() == 3
            && action_cache_actions.size(0) == cells
            && action_cache_actions.size(1) == capacity
            && cache_width <= action_cache_actions.size(2),
        "action_cache_actions must be int32[cells,capacity,cache_capacity]"
    );
    const int64_t cache_capacity = action_cache_actions.size(2);
    TORCH_CHECK(
        pending_free_agent_ids.scalar_type() == torch::kInt64
            && pending_free_agent_ids.dim() == 2
            && pending_free_agent_ids.size(0) == cells,
        "pending_free_agent_ids must be int64[cells,pending_capacity]"
    );
    const int64_t pending_capacity = pending_free_agent_ids.size(1);
    TORCH_CHECK(pending_capacity >= capacity, "pending free capacity must cover slot capacity");
    TORCH_CHECK(
        pending_free_slots.scalar_type() == torch::kInt64
            && pending_free_slots.sizes() == pending_free_agent_ids.sizes(),
        "pending_free_slots must match pending_free_agent_ids"
    );
    TORCH_CHECK(
        pending_free_count.scalar_type() == torch::kInt64
            && pending_free_count.sizes() == torch::IntArrayRef({cells}),
        "pending_free_count must be int64[cells]"
    );
    check_cell_capacity(
        policy_energy_observation,
        "policy_energy_observation",
        torch::kFloat64,
        2
    );
    check_cell_capacity(energy_collected, "energy_collected", torch::kFloat32, 2);
    check_cell_capacity(
        energy_collected_exact,
        "energy_collected_exact",
        torch::kFloat64,
        2
    );
    check_cell_capacity(last_action, "last_action", torch::kInt32, 2);
    check_cell_capacity(last_last_action, "last_last_action", torch::kInt32, 2);
    check_cell_capacity(action_switches, "action_switches", torch::kInt32, 2);
    check_cell_capacity(action_oscillations, "action_oscillations", torch::kInt32, 2);
    check_cell_capacity(episode0_switches, "episode0_switches", torch::kInt32, 2);
    check_cell_capacity(
        episode0_oscillations,
        "episode0_oscillations",
        torch::kInt32,
        2
    );
    check_cell_capacity(survival_steps, "survival_steps", torch::kInt32, 2);
    check_cell_capacity(food_gained_scratch, "food_gained_scratch", torch::kFloat32, 2);
    check_cell_capacity(mlp_pack, "mlp_pack", torch::kFloat32, 3);
    check_cell_capacity(drives, "drives", torch::kFloat32, 3);
    check_cell_capacity(attack_bias, "attack_bias", torch::kFloat32, 2);
    check_cell_capacity(action_temperature, "action_temperature", torch::kFloat32, 2);
    check_cell_capacity(drive_logit_gain, "drive_logit_gain", torch::kFloat32, 2);
    TORCH_CHECK(drives.size(2) >= 4, "drives must expose at least four channels");
    for (const int64_t index : {0, 13, 14}) {
        check_cell_capacity(
            full_tick_state_tensors.at(index),
            "full tick int32 slot state",
            torch::kInt32,
            2
        );
    }
    for (const int64_t index : {1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12}) {
        check_cell_capacity(
            full_tick_state_tensors.at(index),
            "full tick float32 slot state",
            torch::kFloat32,
            2
        );
    }
    TORCH_CHECK(
        full_tick_state_tensors.at(6).scalar_type() == torch::kFloat32
            && full_tick_state_tensors.at(6).sizes()
                == torch::IntArrayRef({cells, capacity, drives.size(2)}),
        "prev_drive_delta must match drives"
    );
    TORCH_CHECK(
        full_tick_state_tensors.at(15).scalar_type() == torch::kFloat32
            && full_tick_state_tensors.at(15).dim() == 3
            && full_tick_state_tensors.at(15).size(0) == cells
            && full_tick_state_tensors.at(15).size(1) == capacity
            && full_tick_state_tensors.at(15).size(2) >= 6,
        "chemistry must be float32[cells,capacity,width>=6]"
    );
    for (const int64_t index : {16, 18}) {
        TORCH_CHECK(
            full_tick_state_tensors.at(index).scalar_type() == torch::kUInt64
                && full_tick_state_tensors.at(index).sizes()
                    == torch::IntArrayRef({cells, 4}),
            "PCG64 state must be uint64[cells,4]"
        );
    }
    for (const int64_t index : {17, 19}) {
        TORCH_CHECK(
            full_tick_state_tensors.at(index).scalar_type() == torch::kInt64
                && full_tick_state_tensors.at(index).sizes()
                    == torch::IntArrayRef({cells, 2}),
            "PCG64 buffer must be int64[cells,2]"
        );
    }
    TORCH_CHECK(
        full_tick_state_tensors.at(20).scalar_type() == torch::kFloat32
            && full_tick_state_tensors.at(20).dim() == 4
            && full_tick_state_tensors.at(20).size(0) == cells
            && full_tick_state_tensors.at(20).size(2) == capacity
            && full_tick_state_tensors.at(20).size(3) == drives.size(2),
        "drive noise samples have an invalid shape"
    );
    const int64_t drive_noise_sample_ticks = full_tick_state_tensors.at(20).size(1);
    TORCH_CHECK(
        full_tick_state_tensors.at(21).scalar_type() == torch::kBool
            && full_tick_state_tensors.at(21).sizes()
                == torch::IntArrayRef({cells, drive_noise_sample_ticks, capacity}),
        "drive noise sample validity has an invalid shape"
    );
    for (const int64_t index : {22, 24, 25}) {
        TORCH_CHECK(
            full_tick_state_tensors.at(index).scalar_type() == torch::kInt64
                && full_tick_state_tensors.at(index).sizes()
                    == torch::IntArrayRef({cells, capacity}),
            "full tick queue/scratch must be int64[cells,capacity]"
        );
    }
    for (const int64_t index : {23, 26, 27, 28}) {
        TORCH_CHECK(
            full_tick_state_tensors.at(index).scalar_type() == torch::kInt64
                && full_tick_state_tensors.at(index).sizes()
                    == torch::IntArrayRef({cells}),
            "full tick scalar state must be int64[cells]"
        );
    }
    TORCH_CHECK(
        full_tick_state_tensors.at(24).data_ptr<int64_t>()
            != full_tick_output_tensors.at(9).data_ptr<int64_t>()
            && full_tick_state_tensors.at(25).data_ptr<int64_t>()
                != full_tick_output_tensors.at(10).data_ptr<int64_t>()
            && full_tick_state_tensors.at(26).data_ptr<int64_t>()
                != full_tick_output_tensors.at(12).data_ptr<int64_t>(),
        "residual pending scratch must not alias residual death outputs"
    );
    TORCH_CHECK(
        agent_grid.scalar_type() == torch::kInt32
            && agent_grid.dim() == 3
            && agent_grid.size(0) == cells,
        "agent_grid must be int32[cells,height,width]"
    );
    TORCH_CHECK(
        food_grid.scalar_type() == torch::kInt32
            && food_grid.sizes() == agent_grid.sizes(),
        "food_grid must match agent_grid"
    );
    const int64_t height = agent_grid.size(1);
    const int64_t width = agent_grid.size(2);
    const int64_t side = 2 * view_radius + 1;
    const int64_t observation_size = side * side + 2;
    const int64_t packed_width = hidden_size * observation_size
        + hidden_size
        + hidden_size * hidden_size
        + hidden_size
        + action_count * hidden_size
        + action_count;
    TORCH_CHECK(mlp_pack.size(2) == packed_width, "fixed MLP packed width mismatch");

    TORCH_CHECK(
        ordered_agent_ids.scalar_type() == torch::kInt64
            && ordered_agent_ids.sizes() == torch::IntArrayRef({cells, ticks, capacity}),
        "ordered_agent_ids output must be int64[cells,ticks,capacity]"
    );
    for (const auto& named : std::vector<std::pair<const char*, torch::Tensor>>{
             {"death_agent_ids", death_agent_ids},
             {"death_slots", death_slots},
             {"survivor_agent_ids", survivor_agent_ids},
         }) {
        TORCH_CHECK(
            named.second.scalar_type() == torch::kInt64
                && named.second.sizes() == torch::IntArrayRef({cells, ticks, capacity}),
            named.first,
            " output must be int64[cells,ticks,capacity]"
        );
    }
    TORCH_CHECK(
        death_reason_codes.scalar_type() == torch::kInt32
            && death_reason_codes.sizes() == torch::IntArrayRef({cells, ticks, capacity}),
        "death_reason_codes output must be int32[cells,ticks,capacity]"
    );
    for (const auto& named : std::vector<std::pair<const char*, torch::Tensor>>{
             {"ordered_slots", ordered_slots},
             {"row_actions", row_actions},
             {"actions_by_slot", actions_by_slot},
         }) {
        TORCH_CHECK(
            named.second.scalar_type() == torch::kInt32
                && named.second.sizes() == torch::IntArrayRef({cells, ticks, capacity}),
            named.first,
            " output must be int32[cells,ticks,capacity]"
        );
    }
    for (const auto& named : std::vector<std::pair<const char*, torch::Tensor>>{
             {"pre_energy", pre_energy},
             {"pre_health", pre_health},
             {"pre_injury", pre_injury},
         }) {
        TORCH_CHECK(
            named.second.scalar_type() == torch::kFloat32
                && named.second.sizes() == torch::IntArrayRef({cells, ticks, capacity}),
            named.first,
            " output must be float32[cells,ticks,capacity]"
        );
    }
    TORCH_CHECK(
        pre_survival_steps.scalar_type() == torch::kInt32
            && pre_survival_steps.sizes() == torch::IntArrayRef({cells, ticks, capacity}),
        "pre_survival_steps output must be int32[cells,ticks,capacity]"
    );
    TORCH_CHECK(
        observations.scalar_type() == torch::kFloat32
            && observations.sizes() == torch::IntArrayRef(
                {cells, ticks, capacity, observation_size}
            ),
        "observations output has an invalid shape or dtype"
    );
    TORCH_CHECK(
        logits.scalar_type() == torch::kFloat32
            && logits.sizes() == torch::IntArrayRef(
                {cells, ticks, capacity, action_count}
            ),
        "logits output has an invalid shape or dtype"
    );
    TORCH_CHECK(
        active_count.scalar_type() == torch::kInt64
            && active_count.sizes() == torch::IntArrayRef({cells, ticks}),
        "active_count output must be int64[cells,ticks]"
    );
    for (const auto& named : std::vector<std::pair<const char*, torch::Tensor>>{
             {"policy_refill_rows", policy_refill_rows},
             {"policy_cache_hit_rows", policy_cache_hit_rows},
             {"death_count", death_count},
             {"survivor_count", survivor_count},
         }) {
        TORCH_CHECK(
            named.second.scalar_type() == torch::kInt64
                && named.second.sizes() == torch::IntArrayRef({cells, ticks}),
            named.first,
            " output must be int64[cells,ticks]"
        );
    }
    TORCH_CHECK(
        lifecycle_status.scalar_type() == torch::kInt32
            && lifecycle_status.sizes() == torch::IntArrayRef({cells, ticks}),
        "lifecycle_status output must be int32[cells,ticks]"
    );
    TORCH_CHECK(
        status.scalar_type() == torch::kInt32
            && status.sizes() == torch::IntArrayRef({cells}),
        "status output must be int32[cells]"
    );

    TORCH_CHECK(
        full_tick_output_tensors.at(0).scalar_type() == torch::kInt32
            && full_tick_output_tensors.at(0).sizes()
                == torch::IntArrayRef({cells, ticks, capacity}),
        "chemistry order output has an invalid shape"
    );
    for (const int64_t index : {1, 3, 5, 12, 13}) {
        TORCH_CHECK(
            full_tick_output_tensors.at(index).scalar_type() == torch::kInt64
                && full_tick_output_tensors.at(index).sizes()
                    == torch::IntArrayRef({cells, ticks}),
            "full tick int64 counter output has an invalid shape"
        );
    }
    for (const int64_t index : {2, 4, 6, 7, 14}) {
        TORCH_CHECK(
            full_tick_output_tensors.at(index).scalar_type() == torch::kInt32
                && full_tick_output_tensors.at(index).sizes()
                    == torch::IntArrayRef({cells, ticks}),
            "full tick status output has an invalid shape"
        );
    }
    TORCH_CHECK(
        full_tick_output_tensors.at(8).scalar_type() == torch::kInt64
            && full_tick_output_tensors.at(8).sizes()
                == torch::IntArrayRef({cells, ticks, 5}),
        "active mechanics count output has an invalid shape"
    );
    for (const int64_t index : {9, 10}) {
        TORCH_CHECK(
            full_tick_output_tensors.at(index).scalar_type() == torch::kInt64
                && full_tick_output_tensors.at(index).sizes()
                    == torch::IntArrayRef({cells, ticks, capacity}),
            "residual death int64 output has an invalid shape"
        );
    }
    TORCH_CHECK(
        full_tick_output_tensors.at(11).scalar_type() == torch::kInt32
            && full_tick_output_tensors.at(11).sizes()
                == torch::IntArrayRef({cells, ticks, capacity}),
        "residual death reason output has an invalid shape"
    );
    TORCH_CHECK(
        full_tick_output_tensors.at(15).scalar_type() == torch::kUInt64
            && full_tick_output_tensors.at(15).sizes()
                == torch::IntArrayRef({cells, ticks, 4}),
        "policy PCG64 state snapshot output has an invalid shape"
    );
    TORCH_CHECK(
        full_tick_output_tensors.at(16).scalar_type() == torch::kInt64
            && full_tick_output_tensors.at(16).sizes()
                == torch::IntArrayRef({cells, ticks, 2}),
        "policy PCG64 buffer snapshot output has an invalid shape"
    );
    TORCH_CHECK(
        full_tick_output_tensors.at(17).scalar_type() == torch::kInt64
            && full_tick_output_tensors.at(17).sizes()
                == torch::IntArrayRef({cells, ticks, 3}),
        "food spawn statistics output has an invalid shape"
    );

    const std::vector<c10::ScalarType> event_column_dtypes{
        torch::kInt32, torch::kInt64, torch::kInt64, torch::kInt64,
        torch::kInt64, torch::kFloat32, torch::kInt32, torch::kInt64,
        torch::kInt64, torch::kInt64, torch::kInt64, torch::kInt64,
        torch::kInt64, torch::kInt64, torch::kInt64, torch::kInt64,
        torch::kInt64, torch::kInt32, torch::kInt32, torch::kInt32,
        torch::kInt8, torch::kFloat32,
    };
    const int64_t event_capacity = event_state_tensors.at(0).size(1);
    for (int64_t index = 0; index < 22; ++index) {
        TORCH_CHECK(
            event_state_tensors.at(index).scalar_type()
                    == event_column_dtypes.at(index)
                && event_state_tensors.at(index).sizes()
                    == torch::IntArrayRef({cells, event_capacity}),
            "event tape column has an invalid shape or dtype"
        );
    }
    TORCH_CHECK(event_capacity > 0, "event capacity must be positive");
    TORCH_CHECK(
        event_state_tensors.at(22).scalar_type() == torch::kInt64
            && event_state_tensors.at(22).sizes() == torch::IntArrayRef({cells}),
        "event active count must be int64[cells]"
    );
    TORCH_CHECK(
        event_state_tensors.at(23).scalar_type() == torch::kInt32
            && event_state_tensors.at(23).sizes() == torch::IntArrayRef({cells}),
        "event status must be int32[cells]"
    );
    for (const int64_t index : {24, 25}) {
        TORCH_CHECK(
            event_state_tensors.at(index).scalar_type() == torch::kInt32
                && event_state_tensors.at(index).sizes()
                    == torch::IntArrayRef({cells, capacity}),
            "lineage event state has an invalid shape"
        );
    }
    TORCH_CHECK(
        event_state_tensors.at(26).scalar_type() == torch::kInt64
            && event_state_tensors.at(26).sizes()
                == torch::IntArrayRef({cells, agent_id_capacity}),
        "initial_id_to_row has an invalid shape"
    );
    const int64_t reporting_rows = event_state_tensors.at(27).size(1);
    for (const int64_t index : {27, 28}) {
        TORCH_CHECK(
            event_state_tensors.at(index).scalar_type() == torch::kInt32
                && event_state_tensors.at(index).sizes()
                    == torch::IntArrayRef({cells, reporting_rows}),
            "terminal reporting state has an invalid shape"
        );
    }
    TORCH_CHECK(
        event_state_tensors.at(29).scalar_type() == torch::kInt32
            && event_state_tensors.at(29).sizes()
                == torch::IntArrayRef({cells, ticks}),
        "event status-by-tick has an invalid shape"
    );
    if (ordinary_births_enabled) {
        const auto require_exact_alias = [](
            const torch::Tensor& left,
            const torch::Tensor& right,
            const char* name
        ) {
            TORCH_CHECK(
                left.data_ptr() == right.data_ptr()
                    && left.storage_offset() == right.storage_offset()
                    && left.sizes() == right.sizes()
                    && left.strides() == right.strides()
                    && left.scalar_type() == right.scalar_type(),
                name,
                " must be one exact authoritative tensor view"
            );
        };
        const auto& birth = ordinary_birth_state_tensors;
        require_exact_alias(birth.at(0), slot_to_agent_id, "slot_to_agent_id");
        require_exact_alias(birth.at(1), agent_id_to_slot, "agent_id_to_slot");
        require_exact_alias(birth.at(2), slot_tokens, "slot_tokens");
        require_exact_alias(birth.at(3), alive_mask, "alive_mask");
        require_exact_alias(birth.at(4), pos_y, "pos_y");
        require_exact_alias(birth.at(5), pos_x, "pos_x");
        require_exact_alias(birth.at(6), agent_grid, "agent_grid");
        require_exact_alias(birth.at(7), energy, "energy");
        require_exact_alias(birth.at(8), health, "health");
        require_exact_alias(birth.at(9), injury, "injury");
        require_exact_alias(birth.at(11), full_tick_state_tensors.at(0), "age");
        require_exact_alias(
            birth.at(13),
            birth.at(69),
            "repro_threshold"
        );
        require_exact_alias(
            birth.at(15),
            event_state_tensors.at(24),
            "lineage_founder_id"
        );
        require_exact_alias(
            birth.at(16),
            event_state_tensors.at(25),
            "lineage_depth"
        );
        require_exact_alias(
            birth.at(20),
            policy_energy_observation,
            "policy_energy_observation"
        );
        require_exact_alias(birth.at(21), survival_steps, "survival_steps");
        require_exact_alias(birth.at(22), rest_streak, "rest_streak");
        require_exact_alias(
            birth.at(23),
            action_cache_token,
            "action_cache_token"
        );
        require_exact_alias(
            birth.at(24),
            action_cache_cursor,
            "action_cache_cursor"
        );
        require_exact_alias(
            birth.at(25),
            action_cache_next,
            "action_cache_next"
        );
        require_exact_alias(
            birth.at(26),
            full_tick_state_tensors.at(22),
            "slot_free_queue"
        );
        require_exact_alias(
            birth.at(27),
            full_tick_state_tensors.at(23),
            "slot_free_queue_count"
        );
        require_exact_alias(
            birth.at(61),
            birth.at(76),
            "pred_lr"
        );
        require_exact_alias(
            birth.at(70),
            full_tick_state_tensors.at(1),
            "gru_upkeep_cost"
        );
        require_exact_alias(birth.at(71), attack_bias, "attack_bias");
        require_exact_alias(
            birth.at(72),
            action_temperature,
            "action_temperature"
        );
        require_exact_alias(
            birth.at(73),
            drive_logit_gain,
            "drive_logit_gain"
        );
        require_exact_alias(
            birth.at(74),
            full_tick_state_tensors.at(4),
            "drive_inertia"
        );
        require_exact_alias(
            birth.at(75),
            full_tick_state_tensors.at(5),
            "drive_noise"
        );
        require_exact_alias(birth.at(77), drives, "drives");
        require_exact_alias(birth.at(83), mlp_pack, "mlp_pack");
        require_exact_alias(
            birth.at(92),
            energy_collected,
            "energy_collected"
        );
        require_exact_alias(
            birth.at(93),
            energy_collected_exact,
            "energy_collected_exact"
        );
    }

    c10::cuda::CUDAGuard device_guard(active_agent_ids.device());

    using namespace eiel::cuda_architecture_b::stackless_birth_generation;
    const FlatFullTickState full_tick_state{
        full_tick_state_tensors.at(0).data_ptr<int32_t>(),
        full_tick_state_tensors.at(1).data_ptr<float>(),
        full_tick_state_tensors.at(2).data_ptr<float>(),
        full_tick_state_tensors.at(3).data_ptr<float>(),
        full_tick_state_tensors.at(4).data_ptr<float>(),
        full_tick_state_tensors.at(5).data_ptr<float>(),
        full_tick_state_tensors.at(6).data_ptr<float>(),
        full_tick_state_tensors.at(7).data_ptr<float>(),
        full_tick_state_tensors.at(8).data_ptr<float>(),
        full_tick_state_tensors.at(9).data_ptr<float>(),
        full_tick_state_tensors.at(10).data_ptr<float>(),
        full_tick_state_tensors.at(11).data_ptr<float>(),
        full_tick_state_tensors.at(12).data_ptr<float>(),
        full_tick_state_tensors.at(13).data_ptr<int32_t>(),
        full_tick_state_tensors.at(14).data_ptr<int32_t>(),
        full_tick_state_tensors.at(15).data_ptr<float>(),
        full_tick_state_tensors.at(16).data_ptr<uint64_t>(),
        full_tick_state_tensors.at(17).data_ptr<int64_t>(),
        full_tick_state_tensors.at(18).data_ptr<uint64_t>(),
        full_tick_state_tensors.at(19).data_ptr<int64_t>(),
        full_tick_state_tensors.at(20).data_ptr<float>(),
        full_tick_state_tensors.at(21).data_ptr<bool>(),
        full_tick_state_tensors.at(22).data_ptr<int64_t>(),
        full_tick_state_tensors.at(23).data_ptr<int64_t>(),
        full_tick_state_tensors.at(24).data_ptr<int64_t>(),
        full_tick_state_tensors.at(25).data_ptr<int64_t>(),
        full_tick_state_tensors.at(26).data_ptr<int64_t>(),
        full_tick_state_tensors.at(27).data_ptr<int64_t>(),
        full_tick_state_tensors.at(28).data_ptr<int64_t>(),
        drive_noise_sample_ticks,
        full_tick_state_tensors.at(15).size(2),
    };
    const FlatFullTickOutputs full_tick_outputs{
        full_tick_output_tensors.at(0).data_ptr<int32_t>(),
        full_tick_output_tensors.at(1).data_ptr<int64_t>(),
        full_tick_output_tensors.at(2).data_ptr<int32_t>(),
        full_tick_output_tensors.at(3).data_ptr<int64_t>(),
        full_tick_output_tensors.at(4).data_ptr<int32_t>(),
        full_tick_output_tensors.at(5).data_ptr<int64_t>(),
        full_tick_output_tensors.at(6).data_ptr<int32_t>(),
        full_tick_output_tensors.at(7).data_ptr<int32_t>(),
        full_tick_output_tensors.at(8).data_ptr<int64_t>(),
        full_tick_output_tensors.at(9).data_ptr<int64_t>(),
        full_tick_output_tensors.at(10).data_ptr<int64_t>(),
        full_tick_output_tensors.at(11).data_ptr<int32_t>(),
        full_tick_output_tensors.at(12).data_ptr<int64_t>(),
        full_tick_output_tensors.at(13).data_ptr<int64_t>(),
        full_tick_output_tensors.at(14).data_ptr<int32_t>(),
        full_tick_output_tensors.at(15).data_ptr<uint64_t>(),
        full_tick_output_tensors.at(16).data_ptr<int64_t>(),
        full_tick_output_tensors.at(17).data_ptr<int64_t>(),
    };
    const FlatEventState event_state{
        event_state_tensors.at(0).data_ptr<int32_t>(),
        event_state_tensors.at(1).data_ptr<int64_t>(),
        event_state_tensors.at(2).data_ptr<int64_t>(),
        event_state_tensors.at(3).data_ptr<int64_t>(),
        event_state_tensors.at(4).data_ptr<int64_t>(),
        event_state_tensors.at(5).data_ptr<float>(),
        event_state_tensors.at(6).data_ptr<int32_t>(),
        event_state_tensors.at(7).data_ptr<int64_t>(),
        event_state_tensors.at(8).data_ptr<int64_t>(),
        event_state_tensors.at(9).data_ptr<int64_t>(),
        event_state_tensors.at(10).data_ptr<int64_t>(),
        event_state_tensors.at(11).data_ptr<int64_t>(),
        event_state_tensors.at(12).data_ptr<int64_t>(),
        event_state_tensors.at(13).data_ptr<int64_t>(),
        event_state_tensors.at(14).data_ptr<int64_t>(),
        event_state_tensors.at(15).data_ptr<int64_t>(),
        event_state_tensors.at(16).data_ptr<int64_t>(),
        event_state_tensors.at(17).data_ptr<int32_t>(),
        event_state_tensors.at(18).data_ptr<int32_t>(),
        event_state_tensors.at(19).data_ptr<int32_t>(),
        event_state_tensors.at(20).data_ptr<int8_t>(),
        event_state_tensors.at(21).data_ptr<float>(),
        event_state_tensors.at(22).data_ptr<int64_t>(),
        event_state_tensors.at(23).data_ptr<int32_t>(),
        event_state_tensors.at(24).data_ptr<int32_t>(),
        event_state_tensors.at(25).data_ptr<int32_t>(),
        event_state_tensors.at(26).data_ptr<int64_t>(),
        event_state_tensors.at(27).data_ptr<int32_t>(),
        event_state_tensors.at(28).data_ptr<int32_t>(),
        event_state_tensors.at(29).data_ptr<int32_t>(),
        event_capacity,
        reporting_rows,
    };
    FlatOrdinaryBirthState ordinary_birth_state{};
    FlatOrdinaryBirthOutputs ordinary_birth_outputs{};
    FlatStacklessBirthScratch ordinary_birth_scratch{};
    FlatOrdinaryBirthConfig ordinary_birth_config{
        ordinary_births_enabled,
        max_births_per_tick,
        reproduce_parent_energy_share,
        static_cast<float>(reproduce_min_energy_buffer),
        static_cast<int32_t>(reproduce_cooldown_steps),
        static_cast<int32_t>(genome_sensor_radius_max),
        injury_repro_k_injury,
        injury_repro_k_scar,
        injury_birth_fail_scar,
        injury_birth_fail_injury,
        env_policy_alias,
        reproduce_requires_mate,
        0,
        0,
        0,
        0,
    };
    if (ordinary_births_enabled) {
        const auto& state = ordinary_birth_state_tensors;
        const auto& output = ordinary_birth_output_tensors;
        ordinary_birth_state = {
            state.at(10).data_ptr<float>(),
            state.at(12).data_ptr<int32_t>(),
            state.at(13).data_ptr<float>(),
            state.at(14).data_ptr<int32_t>(),
            state.at(17).data_ptr<int32_t>(),
            state.at(18).data_ptr<int32_t>(),
            state.at(19).data_ptr<int32_t>(),
            state.at(28).data_ptr<int64_t>(),
            state.at(35).data_ptr<uint32_t>(),
            state.at(36).data_ptr<int64_t>(),
            state.at(37).data_ptr<bool>(),
            state.at(38).data_ptr<double>(),
            state.at(39).data_ptr<double>(),
            state.at(40).data_ptr<double>(),
            state.at(41).data_ptr<int64_t>(),
            state.at(42).data_ptr<int64_t>(),
            state.at(43).data_ptr<int32_t>(),
            state.at(44).data_ptr<int32_t>(),
            state.at(45).data_ptr<int32_t>(),
            state.at(46).data_ptr<int32_t>(),
            state.at(47).data_ptr<int32_t>(),
            state.at(48).data_ptr<int32_t>(),
            state.at(49).data_ptr<int32_t>(),
            state.at(50).data_ptr<int32_t>(),
            state.at(51).data_ptr<int64_t>(),
            state.at(52).data_ptr<bool>(),
            state.at(53).data_ptr<float>(),
            state.at(54).data_ptr<float>(),
            state.at(55).data_ptr<float>(),
            state.at(56).data_ptr<float>(),
            state.at(57).data_ptr<float>(),
            state.at(58).data_ptr<float>(),
            state.at(59).data_ptr<float>(),
            state.at(60).data_ptr<float>(),
            state.at(61).data_ptr<float>(),
            state.at(62).data_ptr<float>(),
            state.at(63).data_ptr<float>(),
            state.at(64).data_ptr<float>(),
            state.at(65).data_ptr<float>(),
            state.at(66).data_ptr<float>(),
            state.at(67).data_ptr<float>(),
            state.at(68).data_ptr<float>(),
            state.at(76).data_ptr<float>(),
            state.at(78).data_ptr<float>(),
            state.at(79).data_ptr<uint64_t>(),
            state.at(80).data_ptr<int64_t>(),
            state.at(81).data_ptr<int64_t>(),
            state.at(84).data_ptr<float>(),
            state.at(85).data_ptr<uint8_t>(),
            state.at(86).data_ptr<float>(),
            state.at(87).data_ptr<bool>(),
            state.at(88).data_ptr<uint8_t>(),
            state.at(89).data_ptr<int64_t>(),
            state.at(90).data_ptr<int64_t>(),
            state.at(91).data_ptr<int64_t>(),
            state.at(94).data_ptr<int32_t>(),
            state.at(95).data_ptr<int32_t>(),
            state.at(96).data_ptr<float>(),
            state.at(97).data_ptr<double>(),
            state.at(98).data_ptr<float>(),
            state.at(99).data_ptr<float>(),
            state.at(100).data_ptr<float>(),
            state.at(101).data_ptr<int32_t>(),
            state.at(102).data_ptr<int32_t>(),
            state.at(103).data_ptr<bool>(),
            state.at(104).data_ptr<float>(),
            state.at(105).data_ptr<int32_t>(),
            state.at(106).data_ptr<float>(),
            state.at(107).data_ptr<int32_t>(),
            state.at(108).data_ptr<float>(),
            state.at(109).data_ptr<float>(),
        };
        ordinary_birth_outputs = {
            output.at(0).data_ptr<int64_t>(),
            output.at(1).data_ptr<int32_t>(),
            output.at(2).data_ptr<int64_t>(),
            output.at(3).data_ptr<int64_t>(),
            output.at(4).data_ptr<int32_t>(),
            output.at(5).data_ptr<float>(),
            output.at(6).data_ptr<int64_t>(),
            output.at(7).data_ptr<int32_t>(),
            output.at(8).data_ptr<int32_t>(),
            output.at(9).data_ptr<int64_t>(),
            output.at(10).data_ptr<int64_t>(),
            output.at(11).data_ptr<int32_t>(),
            output.at(12).data_ptr<bool>(),
            output.at(13).data_ptr<int64_t>(),
            output.at(14).data_ptr<int64_t>(),
            output.at(15).data_ptr<int64_t>(),
            output.at(16).data_ptr<int64_t>(),
            output.at(17).data_ptr<int64_t>(),
            output.at(18).data_ptr<int64_t>(),
            output.at(19).data_ptr<int64_t>(),
            output.at(20).data_ptr<int32_t>(),
            output.at(21).data_ptr<int64_t>(),
            output.at(22).data_ptr<int64_t>(),
            output.at(23).data_ptr<int64_t>(),
            output.at(24).data_ptr<int64_t>(),
            output.at(25).data_ptr<int32_t>(),
            output.at(26).data_ptr<int64_t>(),
            output.at(27).data_ptr<int64_t>(),
            output.at(28).data_ptr<int32_t>(),
            output.at(29).data_ptr<int64_t>(),
            output.at(30).data_ptr<int64_t>(),
            output.at(31).data_ptr<int64_t>(),
            output.at(32).data_ptr<int64_t>(),
            output.at(33).data_ptr<int64_t>(),
            output.at(34).data_ptr<int32_t>(),
            output.at(35).data_ptr<int32_t>(),
            output.at(36).data_ptr<int64_t>(),
            output.at(37).data_ptr<int64_t>(),
            output.at(38).data_ptr<int64_t>(),
            output.at(39).data_ptr<int64_t>(),
            output.at(40).data_ptr<double>(),
            output.at(41).data_ptr<bool>(),
            output.at(42).data_ptr<float>(),
            output.at(43).data_ptr<int32_t>(),
            output.at(44).data_ptr<int32_t>(),
            output.at(45).data_ptr<uint64_t>(),
            output.at(46).data_ptr<int64_t>(),
        };
        ordinary_birth_config.gauss_tape_capacity = state.at(39).size(1);
        ordinary_birth_config.policy_tape_capacity = state.at(86).size(1);
        ordinary_birth_config.torch_state_bytes = state.at(85).size(1);
        ordinary_birth_config.hmem_width = state.at(84).size(2);
        ordinary_birth_scratch = {
            ordinary_birth_scratch_tensors.at(0).data_ptr<uint32_t>(),
            ordinary_birth_scratch_tensors.at(0).size(1),
        };
    }
    const FlatMechanicsConfig mechanics_config{
        enable_rest_charge,
        static_cast<int32_t>(rest_streak_free),
        static_cast<float>(rest_charge_energy),
        static_cast<float>(rest_charge_growth),
        static_cast<float>(rest_deficit_to_health),
        static_cast<int32_t>(developmental_maturity_age),
        static_cast<float>(drive_hunger_gain),
        static_cast<float>(drive_hunger_decay),
        static_cast<float>(drive_pain_gain),
        static_cast<float>(drive_pain_decay),
        static_cast<float>(drive_fatigue_gain),
        static_cast<float>(drive_fatigue_rest_relief),
        static_cast<float>(drive_fatigue_decay),
        static_cast<float>(drive_fear_gain),
        static_cast<float>(drive_fear_decay),
        generation,
        start_tick,
    };

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    launch_flat_dynamic_policy_food_stackless_birth_full_tick_v1(
        active_agent_ids.data_ptr<int64_t>(),
        active_count_state.data_ptr<int64_t>(),
        slot_to_agent_id.data_ptr<int64_t>(),
        agent_id_to_slot.data_ptr<int32_t>(),
        alive_mask.data_ptr<bool>(),
        slot_tokens.data_ptr<int64_t>(),
        action_cache_actions.data_ptr<int32_t>(),
        action_cache_token.data_ptr<int64_t>(),
        action_cache_cursor.data_ptr<int32_t>(),
        action_cache_next.data_ptr<int32_t>(),
        pos_y.data_ptr<int32_t>(),
        pos_x.data_ptr<int32_t>(),
        agent_grid.data_ptr<int32_t>(),
        food_grid.data_ptr<int32_t>(),
        energy.data_ptr<float>(),
        health.data_ptr<float>(),
        injury.data_ptr<float>(),
        policy_energy_observation.data_ptr<double>(),
        rest_streak.data_ptr<int32_t>(),
        pending_free_agent_ids.data_ptr<int64_t>(),
        pending_free_slots.data_ptr<int64_t>(),
        pending_free_count.data_ptr<int64_t>(),
        energy_collected.data_ptr<float>(),
        energy_collected_exact.data_ptr<double>(),
        last_action.data_ptr<int32_t>(),
        last_last_action.data_ptr<int32_t>(),
        action_switches.data_ptr<int32_t>(),
        action_oscillations.data_ptr<int32_t>(),
        episode0_switches.data_ptr<int32_t>(),
        episode0_oscillations.data_ptr<int32_t>(),
        survival_steps.data_ptr<int32_t>(),
        mlp_pack.data_ptr<float>(),
        drives.data_ptr<float>(),
        attack_bias.data_ptr<float>(),
        action_temperature.data_ptr<float>(),
        drive_logit_gain.data_ptr<float>(),
        food_gained_scratch.data_ptr<float>(),
        ordered_agent_ids.data_ptr<int64_t>(),
        ordered_slots.data_ptr<int32_t>(),
        observations.data_ptr<float>(),
        logits.data_ptr<float>(),
        row_actions.data_ptr<int32_t>(),
        actions_by_slot.data_ptr<int32_t>(),
        pre_energy.data_ptr<float>(),
        pre_health.data_ptr<float>(),
        pre_injury.data_ptr<float>(),
        pre_survival_steps.data_ptr<int32_t>(),
        active_count.data_ptr<int64_t>(),
        policy_refill_rows.data_ptr<int64_t>(),
        policy_cache_hit_rows.data_ptr<int64_t>(),
        death_agent_ids.data_ptr<int64_t>(),
        death_slots.data_ptr<int64_t>(),
        death_reason_codes.data_ptr<int32_t>(),
        death_count.data_ptr<int64_t>(),
        survivor_agent_ids.data_ptr<int64_t>(),
        survivor_count.data_ptr<int64_t>(),
        lifecycle_status.data_ptr<int32_t>(),
        status.data_ptr<int32_t>(),
        full_tick_state,
        full_tick_outputs,
        event_state,
        ordinary_birth_state,
        ordinary_birth_outputs,
        ordinary_birth_scratch,
        ordinary_birth_config,
        mechanics_config,
        cells,
        ticks,
        active_capacity,
        agent_id_capacity,
        capacity,
        pending_capacity,
        height,
        width,
        view_radius,
        observation_size,
        hidden_size,
        action_count,
        packed_width,
        drives.size(2),
        cache_capacity,
        cache_width,
        deterministic_actions,
        max_food,
        static_cast<int64_t>(std::llround(food_spawn_per_step * food_spawn_mult)),
        max_energy,
        start_health,
        move_cost,
        gather_cost,
        rest_cost,
        attack_cost,
        action_cost_mult,
        rest_cost_mult,
        food_energy,
        episode0_k,
        static_cast<float>(w_hunger_gather),
        static_cast<float>(w_pain_rest),
        static_cast<float>(w_fatigue_rest),
        static_cast<float>(w_pain_attack),
        static_cast<float>(w_fear_attack),
        stream
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

pybind11::dict
stackless_flat_dynamic_policy_food_birth_full_tick_resource_receipt_v1() {
    const auto receipt = eiel::cuda_architecture_b::stackless_birth_generation::
        query_flat_dynamic_policy_food_stackless_birth_full_tick_resources_v1();
    pybind11::dict out;
    out["kernel_symbol"] =
        "flat_dynamic_policy_food_stackless_birth_full_tick_kernel_v1";
    out["threads_per_cell"] = eiel::cuda_architecture_b::stackless_birth_generation::
        kThreadsPerCell;
    out["binary_version"] = receipt.binary_version;
    out["ptx_version"] = receipt.ptx_version;
    out["num_registers"] = receipt.num_registers;
    out["constant_size_bytes"] = receipt.constant_size_bytes;
    out["local_size_bytes"] = receipt.local_size_bytes;
    out["shared_size_bytes"] = receipt.shared_size_bytes;
    out["max_threads_per_block"] = receipt.max_threads_per_block;
    out["max_blocks_per_multiprocessor"] = receipt.max_blocks_per_multiprocessor;
    return out;
}

pybind11::dict eiel_authoritative_state_identity_v1() {
    using namespace eiel::authoritative_state::constants_v1;
    pybind11::dict out;
    out["contract_schema_version"] =
        "eiel_authoritative_state_binary_identity_v1";
    out["state_schema_version"] = kSchemaVersion;
    out["state_schema_sha256"] = kSchemaSha256;
    out["source_commit"] = kSourceCommit;
    out["source_inventory_sha256"] = kSourceInventorySha256;
    out["build_identity_sha256"] = kBuildIdentitySha256;
    out["abi_set_sha256"] = kAbiSetSha256;
    out["checkpoint_schema_version"] = "checkpoint_bundle_v2";
    out["checkpoint_schema_sha256"] = kCheckpointSchemaSha256;
    out["chronicle_schema_version"] =
        "eiel_authoritative_state_chronicle_schema_v1";
    out["chronicle_schema_sha256"] = kChronicleSchemaSha256;
    out["provider_capability_identity"] = kProviderCapabilityIdentity;
    out["overlay_version"] = "inheritance-cross-path-overlay-v1";
    out["supported_cache_widths"] = pybind11::make_tuple(1, 2);
    out["supported_profiles"] = pybind11::make_tuple(
        "eiel-production-runtime-v1"
    );
    out["build_toolchain_identity"] = __VERSION__;
#ifdef EIEL_AUTHORITATIVE_STATE_COMPILE_FLAGS
    out["compile_flags"] = EIEL_AUTHORITATIVE_STATE_COMPILE_FLAGS;
#else
    out["compile_flags"] = "extension-build-defaults";
#endif
    out["artifact_build_id"] = kArtifactBuildId;
    pybind11::list variants;
    for (const auto& row : kAbiVariants) {
        pybind11::dict variant;
        variant["entrypoint"] = row.variant_id;
        variant["abi_version"] = row.variant_id;
        variant["argument_count"] = row.argument_count;
        variant["argument_order_sha256"] = row.argument_order_sha256;
        variants.append(variant);
    }
    out["abi_variants"] = variants;
    return out;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.doc() = "Raw-CUDA Architecture B persistent dynamic Flat full tick";
    module.def(
        "eiel_cuda_architecture_b_stackless_flat_dynamic_policy_food_birth_full_tick_v1",
        &stackless_flat_dynamic_policy_food_birth_full_tick_v1
    );
    module.def(
        "eiel_cuda_architecture_b_stackless_flat_dynamic_policy_food_birth_full_tick_resource_receipt_v1",
        &stackless_flat_dynamic_policy_food_birth_full_tick_resource_receipt_v1
    );
    module.def(
        "eiel_authoritative_state_identity_v1",
        &eiel_authoritative_state_identity_v1
    );
}
