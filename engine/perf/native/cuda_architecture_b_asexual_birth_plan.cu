#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include "cuda_architecture_b_pcg64.cuh"

namespace {

using eiel::cuda_architecture_b::Pcg64State;
using eiel::cuda_architecture_b::pcg64_double01;
using eiel::cuda_architecture_b::pcg64_raw32;

constexpr int32_t STATUS_OK = 0;
constexpr int32_t STATUS_INVALID_RESIDENT_STATE = 1;
constexpr int32_t STATUS_INVALID_AGENT_IDS = 2;
constexpr int32_t STATUS_UNSUPPORTED_GRU = 3;
constexpr int32_t STATUS_INVALID_PCG64_STATE = 4;
constexpr int32_t STATUS_INVALID_CONFIG = 5;
constexpr int32_t STATUS_COUNTER_OVERFLOW = 6;

constexpr int32_t REASON_ACCEPTED = 0;
constexpr int32_t REASON_NO_SPACE = 1;
constexpr int32_t REASON_BUFFER = 2;
constexpr int32_t REASON_NO_FREE_SLOT = 3;
constexpr int32_t REASON_CONCEPTION = 5;
constexpr int32_t REASON_GESTATION = 6;
constexpr int32_t REASON_HEALTH = 7;
constexpr int32_t REASON_COOLDOWN = 8;
constexpr int32_t REASON_NO_ENERGY = 9;
constexpr int32_t REASON_COUNT = 10;

__device__ __forceinline__ uint32_t pcg64_bounded_uint32(
    Pcg64State& rng,
    const uint32_t max_value
) {
    if (max_value == 0xffffffffU) {
        return pcg64_raw32(rng);
    }
    uint32_t mask = max_value;
    mask |= mask >> 1U;
    mask |= mask >> 2U;
    mask |= mask >> 4U;
    mask |= mask >> 8U;
    mask |= mask >> 16U;
    uint32_t value = 0;
    do {
        value = pcg64_raw32(rng) & mask;
    } while (value > max_value);
    return value;
}

__device__ __forceinline__ bool selected_position_is_reserved(
    const int32_t y,
    const int32_t x,
    const int32_t* planned_spawn_positions,
    const int64_t planned_count
) {
    for (int64_t row = 0; row < planned_count; ++row) {
        if (planned_spawn_positions[row * 2] == y
            && planned_spawn_positions[row * 2 + 1] == x) {
            return true;
        }
    }
    return false;
}

__device__ __forceinline__ bool child_slot_already_planned(
    const int64_t slot,
    const int64_t* planned_child_slots,
    const int64_t planned_count
) {
    for (int64_t row = 0; row < planned_count; ++row) {
        if (planned_child_slots[row] == slot) {
            return true;
        }
    }
    return false;
}

__global__ void asexual_birth_plan_kernel(
    const int64_t* __restrict__ slot_to_agent_id,
    const bool* __restrict__ alive_mask,
    const bool* __restrict__ policy_use_gru,
    const int32_t* __restrict__ pos_y,
    const int32_t* __restrict__ pos_x,
    const int32_t* __restrict__ agent_grid,
    const float* __restrict__ energy,
    const double* __restrict__ policy_energy_observation,
    const float* __restrict__ health,
    const float* __restrict__ injury,
    const float* __restrict__ scar,
    int32_t* __restrict__ repro_cooldown,
    int32_t* __restrict__ planned_repro_cooldown,
    const float* __restrict__ repro_threshold,
    const int64_t* __restrict__ slot_free_queue,
    const int64_t* __restrict__ slot_free_queue_count,
    int64_t* __restrict__ slot_free_queue_head,
    uint64_t* __restrict__ pcg64_state_words,
    int64_t* __restrict__ pcg64_buffer_words,
    int64_t* __restrict__ policy_np_semantic_draws,
    uint64_t* __restrict__ runner_pcg64_state_words,
    int64_t* __restrict__ runner_pcg64_buffer_words,
    int64_t* __restrict__ metrics_np_semantic_draws,
    int64_t* __restrict__ planned_parent_slots,
    int64_t* __restrict__ planned_child_slots,
    int32_t* __restrict__ planned_spawn_positions,
    float* __restrict__ planned_child_energy,
    int64_t* __restrict__ decision_parent_slots,
    int32_t* __restrict__ decision_reason_codes,
    int32_t* __restrict__ decision_candidate_counts,
    int64_t* __restrict__ decision_draw_starts,
    int64_t* __restrict__ decision_draw_ends,
    int32_t* __restrict__ decision_selected_positions,
    bool* __restrict__ decision_selected_reserved,
    int64_t* __restrict__ reason_counts,
    int64_t* __restrict__ planned_count_out,
    int64_t* __restrict__ decision_count_out,
    int64_t* __restrict__ spawn_rng_draws_added_out,
    int64_t* __restrict__ spawn_rng_semantic_draws_added_out,
    int64_t* __restrict__ runner_rng_draws_added_out,
    int64_t* __restrict__ runner_rng_semantic_draws_added_out,
    int32_t* __restrict__ status,
    const int64_t slot_count,
    const int64_t height,
    const int64_t width,
    const int64_t free_queue_capacity,
    const int64_t max_births,
    const double parent_energy_share,
    const float min_energy_buffer,
    const double injury_repro_k_injury,
    const double injury_repro_k_scar,
    const double injury_birth_fail_scar,
    const double injury_birth_fail_injury,
    const bool env_policy_alias,
    const bool requires_mate,
    const bool injury_gate_active,
    const bool conception_gate_active,
    const bool gestation_gate_active
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }

    if (!::isfinite(parent_energy_share) || parent_energy_share < 0.0
        || parent_energy_share > 1.0 || !::isfinite(min_energy_buffer)
        || min_energy_buffer < 0.0f || max_births < 0 || max_births > slot_count
        || !::isfinite(injury_repro_k_injury) || injury_repro_k_injury < 0.0
        || !::isfinite(injury_repro_k_scar) || injury_repro_k_scar < 0.0
        || !::isfinite(injury_birth_fail_scar) || injury_birth_fail_scar < 0.0
        || !::isfinite(injury_birth_fail_injury) || injury_birth_fail_injury < 0.0
        || requires_mate || injury_gate_active
        || conception_gate_active
            != (injury_repro_k_injury > 0.0 || injury_repro_k_scar > 0.0)
        || gestation_gate_active
            != (injury_birth_fail_scar > 0.0 || injury_birth_fail_injury > 0.0)) {
        status[0] = STATUS_INVALID_CONFIG;
        return;
    }
    const int64_t free_queue_size = slot_free_queue_count[0];
    const int64_t initial_free_head = slot_free_queue_head[0];
    if (free_queue_size < 0 || free_queue_size > free_queue_capacity
        || initial_free_head < 0 || initial_free_head > free_queue_size
        || policy_np_semantic_draws[0] < 0 || metrics_np_semantic_draws[0] < 0) {
        status[0] = STATUS_INVALID_RESIDENT_STATE;
        return;
    }
    if (pcg64_buffer_words[0] < 0 || pcg64_buffer_words[0] > 1
        || pcg64_buffer_words[1] < 0
        || static_cast<uint64_t>(pcg64_buffer_words[1]) > 0xffffffffULL
        || (pcg64_state_words[2] & 1ULL) == 0ULL) {
        status[0] = STATUS_INVALID_PCG64_STATE;
        return;
    }
    if (runner_pcg64_buffer_words[0] < 0 || runner_pcg64_buffer_words[0] > 1
        || runner_pcg64_buffer_words[1] < 0
        || static_cast<uint64_t>(runner_pcg64_buffer_words[1]) > 0xffffffffULL
        || (runner_pcg64_state_words[2] & 1ULL) == 0ULL) {
        status[0] = STATUS_INVALID_PCG64_STATE;
        return;
    }

    int64_t alive_rows = 0;
    for (int64_t slot = 0; slot < slot_count; ++slot) {
        planned_repro_cooldown[slot] = repro_cooldown[slot];
        if (!alive_mask[slot]) {
            continue;
        }
        ++alive_rows;
        const int64_t agent_id = slot_to_agent_id[slot];
        if (agent_id < 0 || pos_y[slot] < 0 || pos_y[slot] >= height
            || pos_x[slot] < 0 || pos_x[slot] >= width
            || !::isfinite(energy[slot]) || !::isfinite(health[slot])
            || !::isfinite(injury[slot]) || injury[slot] < 0.0f
            || !::isfinite(scar[slot]) || scar[slot] < 0.0f
            || !::isfinite(repro_threshold[slot]) || repro_cooldown[slot] < 0) {
            status[0] = STATUS_INVALID_RESIDENT_STATE;
            return;
        }
        if (policy_use_gru[slot]) {
            status[0] = STATUS_UNSUPPORTED_GRU;
            return;
        }
        for (int64_t other = 0; other < slot; ++other) {
            if (alive_mask[other] && slot_to_agent_id[other] == agent_id) {
                status[0] = STATUS_INVALID_AGENT_IDS;
                return;
            }
        }
    }

    Pcg64State rng{
        pcg64_state_words[0],
        pcg64_state_words[1],
        pcg64_state_words[2],
        pcg64_state_words[3],
        static_cast<uint32_t>(pcg64_buffer_words[0]),
        static_cast<uint32_t>(pcg64_buffer_words[1]),
    };
    Pcg64State runner_rng{
        runner_pcg64_state_words[0],
        runner_pcg64_state_words[1],
        runner_pcg64_state_words[2],
        runner_pcg64_state_words[3],
        static_cast<uint32_t>(runner_pcg64_buffer_words[0]),
        static_cast<uint32_t>(runner_pcg64_buffer_words[1]),
    };
    int64_t free_head = initial_free_head;
    int64_t planned_count = 0;
    int64_t decision_count = 0;
    int64_t spawn_rng_draws = 0;
    int64_t spawn_rng_semantic_draws = 0;
    int64_t runner_rng_draws = 0;
    int64_t runner_rng_semantic_draws = 0;
    int64_t previous_agent_id = std::numeric_limits<int64_t>::min();
    int64_t previous_slot = -1;

    for (int64_t order = 0; order < alive_rows; ++order) {
        if (planned_count >= max_births) {
            break;
        }

        int64_t parent_slot = -1;
        int64_t parent_agent_id = std::numeric_limits<int64_t>::max();
        for (int64_t slot = 0; slot < slot_count; ++slot) {
            if (!alive_mask[slot]) {
                continue;
            }
            const int64_t agent_id = slot_to_agent_id[slot];
            const bool follows_previous = agent_id > previous_agent_id
                || (agent_id == previous_agent_id && slot > previous_slot);
            if (!follows_previous) {
                continue;
            }
            if (parent_slot < 0 || agent_id < parent_agent_id
                || (agent_id == parent_agent_id && slot < parent_slot)) {
                parent_slot = slot;
                parent_agent_id = agent_id;
            }
        }
        if (parent_slot < 0) {
            status[0] = STATUS_INVALID_AGENT_IDS;
            return;
        }
        previous_agent_id = parent_agent_id;
        previous_slot = parent_slot;

        const int64_t decision = decision_count++;
        decision_parent_slots[decision] = parent_slot;
        decision_draw_starts[decision] = spawn_rng_draws;
        decision_draw_ends[decision] = spawn_rng_draws;

        int32_t reason = REASON_ACCEPTED;
        if (health[parent_slot] <= 0.0f) {
            reason = REASON_HEALTH;
        } else {
            if (planned_repro_cooldown[parent_slot] > 0) {
                planned_repro_cooldown[parent_slot] -= 1;
            }
            if (planned_repro_cooldown[parent_slot] != 0) {
                reason = REASON_COOLDOWN;
            } else if (energy[parent_slot] < repro_threshold[parent_slot]) {
                reason = REASON_NO_ENERGY;
            }
        }

        int32_t candidate_y[4];
        int32_t candidate_x[4];
        int32_t candidate_count = 0;
        int32_t selected_y = -1;
        int32_t selected_x = -1;
        bool selected_reserved = false;

        if (reason == REASON_ACCEPTED) {
            constexpr int32_t DY[4] = {-1, 1, 0, 0};
            constexpr int32_t DX[4] = {0, 0, -1, 1};
            const int32_t y = pos_y[parent_slot];
            const int32_t x = pos_x[parent_slot];
            for (int32_t direction = 0; direction < 4; ++direction) {
                const int32_t ny = y + DY[direction];
                const int32_t nx = x + DX[direction];
                if (ny < 0 || ny >= height || nx < 0 || nx >= width
                    || agent_grid[static_cast<int64_t>(ny) * width + nx] != -1) {
                    continue;
                }
                candidate_y[candidate_count] = ny;
                candidate_x[candidate_count] = nx;
                ++candidate_count;
            }

            if (spawn_rng_semantic_draws
                > std::numeric_limits<int64_t>::max() - candidate_count) {
                status[0] = STATUS_COUNTER_OVERFLOW;
                return;
            }
            spawn_rng_semantic_draws += candidate_count;
            for (int32_t index = candidate_count - 1; index > 0; --index) {
                const int32_t selected = static_cast<int32_t>(
                    pcg64_bounded_uint32(rng, static_cast<uint32_t>(index))
                );
                ++spawn_rng_draws;
                const int32_t tmp_y = candidate_y[index];
                const int32_t tmp_x = candidate_x[index];
                candidate_y[index] = candidate_y[selected];
                candidate_x[index] = candidate_x[selected];
                candidate_y[selected] = tmp_y;
                candidate_x[selected] = tmp_x;
            }
            if (candidate_count > 0) {
                selected_y = candidate_y[0];
                selected_x = candidate_x[0];
                selected_reserved = selected_position_is_reserved(
                    selected_y,
                    selected_x,
                    planned_spawn_positions,
                    planned_count
                );
            }
            if (candidate_count == 0 || selected_reserved) {
                reason = REASON_NO_SPACE;
            }
        }

        if (reason == REASON_ACCEPTED) {
            const double parent_energy = ::isfinite(policy_energy_observation[parent_slot])
                ? policy_energy_observation[parent_slot]
                : static_cast<double>(energy[parent_slot]);
            const double parent_share = parent_energy * parent_energy_share;
            if ((parent_energy - parent_share) < static_cast<double>(min_energy_buffer)) {
                reason = REASON_BUFFER;
            }
            if (reason == REASON_ACCEPTED && conception_gate_active) {
                if (metrics_np_semantic_draws[0]
                    > std::numeric_limits<int64_t>::max() - runner_rng_semantic_draws - 1) {
                    status[0] = STATUS_COUNTER_OVERFLOW;
                    return;
                }
                const double parent_injury = static_cast<double>(injury[parent_slot]);
                const double parent_scar = static_cast<double>(scar[parent_slot]);
                const double conception_probability = ::exp(-(
                    injury_repro_k_injury * parent_injury
                    + injury_repro_k_scar * parent_scar
                ));
                const double conception_draw = pcg64_double01(runner_rng);
                ++runner_rng_draws;
                ++runner_rng_semantic_draws;
                if (conception_draw > conception_probability) {
                    reason = REASON_CONCEPTION;
                }
            }
            if (reason == REASON_ACCEPTED && gestation_gate_active) {
                if (metrics_np_semantic_draws[0]
                    > std::numeric_limits<int64_t>::max() - runner_rng_semantic_draws - 1) {
                    status[0] = STATUS_COUNTER_OVERFLOW;
                    return;
                }
                const double parent_injury = static_cast<double>(injury[parent_slot]);
                const double parent_scar = static_cast<double>(scar[parent_slot]);
                const double gestation_failure_probability = ::fmin(
                    0.95,
                    injury_birth_fail_scar * parent_scar
                        + injury_birth_fail_injury * parent_injury
                );
                const double gestation_draw = pcg64_double01(runner_rng);
                ++runner_rng_draws;
                ++runner_rng_semantic_draws;
                if (gestation_draw < gestation_failure_probability) {
                    reason = REASON_GESTATION;
                }
            }
            if (reason == REASON_ACCEPTED) {
                int64_t child_slot = -1;
                while (free_head < free_queue_size) {
                    const int64_t candidate_slot = slot_free_queue[free_head++];
                    if (candidate_slot >= 0 && candidate_slot < slot_count
                        && !alive_mask[candidate_slot]
                        && slot_to_agent_id[candidate_slot] < 0
                        && !child_slot_already_planned(
                            candidate_slot,
                            planned_child_slots,
                            planned_count
                        )) {
                        child_slot = candidate_slot;
                        break;
                    }
                }
                if (child_slot < 0) {
                    reason = REASON_NO_FREE_SLOT;
                } else {
                    planned_parent_slots[planned_count] = parent_slot;
                    planned_child_slots[planned_count] = child_slot;
                    planned_spawn_positions[planned_count * 2] = selected_y;
                    planned_spawn_positions[planned_count * 2 + 1] = selected_x;
                    planned_child_energy[planned_count] = static_cast<float>(parent_share);
                    ++planned_count;
                }
            }
        }

        decision_reason_codes[decision] = reason;
        decision_candidate_counts[decision] = candidate_count;
        decision_draw_ends[decision] = spawn_rng_draws;
        decision_selected_positions[decision * 2] = selected_y;
        decision_selected_positions[decision * 2 + 1] = selected_x;
        decision_selected_reserved[decision] = selected_reserved;
        if (reason >= 0 && reason < REASON_COUNT) {
            reason_counts[reason] += 1;
        }
        if (reason == REASON_NO_FREE_SLOT) {
            break;
        }
    }

    if (env_policy_alias && policy_np_semantic_draws[0]
        > std::numeric_limits<int64_t>::max() - spawn_rng_semantic_draws) {
        status[0] = STATUS_COUNTER_OVERFLOW;
        return;
    }
    if (metrics_np_semantic_draws[0]
        > std::numeric_limits<int64_t>::max() - runner_rng_semantic_draws) {
        status[0] = STATUS_COUNTER_OVERFLOW;
        return;
    }

    for (int64_t slot = 0; slot < slot_count; ++slot) {
        repro_cooldown[slot] = planned_repro_cooldown[slot];
    }
    slot_free_queue_head[0] = free_head;
    pcg64_state_words[0] = rng.state_lo;
    pcg64_state_words[1] = rng.state_hi;
    pcg64_state_words[2] = rng.inc_lo;
    pcg64_state_words[3] = rng.inc_hi;
    pcg64_buffer_words[0] = static_cast<int64_t>(rng.has_uint32);
    pcg64_buffer_words[1] = static_cast<int64_t>(rng.uinteger);
    if (env_policy_alias) {
        policy_np_semantic_draws[0] += spawn_rng_semantic_draws;
    }
    runner_pcg64_state_words[0] = runner_rng.state_lo;
    runner_pcg64_state_words[1] = runner_rng.state_hi;
    runner_pcg64_state_words[2] = runner_rng.inc_lo;
    runner_pcg64_state_words[3] = runner_rng.inc_hi;
    runner_pcg64_buffer_words[0] = static_cast<int64_t>(runner_rng.has_uint32);
    runner_pcg64_buffer_words[1] = static_cast<int64_t>(runner_rng.uinteger);
    metrics_np_semantic_draws[0] += runner_rng_semantic_draws;
    planned_count_out[0] = planned_count;
    decision_count_out[0] = decision_count;
    spawn_rng_draws_added_out[0] = spawn_rng_draws;
    spawn_rng_semantic_draws_added_out[0] = spawn_rng_semantic_draws;
    runner_rng_draws_added_out[0] = runner_rng_draws;
    runner_rng_semantic_draws_added_out[0] = runner_rng_semantic_draws;
    status[0] = STATUS_OK;
}

void check_cuda_contiguous(const torch::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

}  // namespace

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
    const double parent_energy_share,
    const double min_energy_buffer,
    const double injury_repro_k_injury,
    const double injury_repro_k_scar,
    const double injury_birth_fail_scar,
    const double injury_birth_fail_injury,
    const int64_t max_births,
    const bool env_policy_alias,
    const bool requires_mate,
    const bool injury_gate_active,
    const bool conception_gate_active,
    const bool gestation_gate_active
) {
    const std::vector<std::pair<const char*, torch::Tensor>> tensors = {
        {"slot_to_agent_id", slot_to_agent_id},
        {"alive_mask", alive_mask},
        {"policy_use_gru", policy_use_gru},
        {"pos_y", pos_y},
        {"pos_x", pos_x},
        {"agent_grid", agent_grid},
        {"energy", energy},
        {"policy_energy_observation", policy_energy_observation},
        {"health", health},
        {"injury", injury},
        {"scar", scar},
        {"repro_cooldown", repro_cooldown},
        {"repro_threshold", repro_threshold},
        {"slot_free_queue", slot_free_queue},
        {"slot_free_queue_count", slot_free_queue_count},
        {"slot_free_queue_head", slot_free_queue_head},
        {"pcg64_state_words", pcg64_state_words},
        {"pcg64_buffer_words", pcg64_buffer_words},
        {"policy_np_semantic_draws", policy_np_semantic_draws},
        {"runner_pcg64_state_words", runner_pcg64_state_words},
        {"runner_pcg64_buffer_words", runner_pcg64_buffer_words},
        {"metrics_np_semantic_draws", metrics_np_semantic_draws},
    };
    for (const auto& item : tensors) {
        check_cuda_contiguous(item.second, item.first);
    }
    const auto device = slot_to_agent_id.device();
    for (const auto& item : tensors) {
        TORCH_CHECK(item.second.device() == device, item.first, " must share one CUDA device");
    }

    TORCH_CHECK(slot_to_agent_id.scalar_type() == torch::kInt64 && slot_to_agent_id.dim() == 1, "slot_to_agent_id must be int64[slots]");
    const int64_t slot_count = slot_to_agent_id.size(0);
    TORCH_CHECK(alive_mask.scalar_type() == torch::kBool && alive_mask.dim() == 1 && alive_mask.numel() == slot_count, "alive_mask must be bool[slots]");
    TORCH_CHECK(policy_use_gru.scalar_type() == torch::kBool && policy_use_gru.dim() == 1 && policy_use_gru.numel() == slot_count, "policy_use_gru must be bool[slots]");
    TORCH_CHECK(pos_y.scalar_type() == torch::kInt32 && pos_y.dim() == 1 && pos_y.numel() == slot_count, "pos_y must be int32[slots]");
    TORCH_CHECK(pos_x.scalar_type() == torch::kInt32 && pos_x.dim() == 1 && pos_x.numel() == slot_count, "pos_x must be int32[slots]");
    TORCH_CHECK(agent_grid.scalar_type() == torch::kInt32 && agent_grid.dim() == 2 && agent_grid.size(0) > 0 && agent_grid.size(1) > 0, "agent_grid must be nonempty int32[height,width]");
    TORCH_CHECK(energy.scalar_type() == torch::kFloat32 && energy.dim() == 1 && energy.numel() == slot_count, "energy must be float32[slots]");
    TORCH_CHECK(policy_energy_observation.scalar_type() == torch::kFloat64 && policy_energy_observation.dim() == 1 && policy_energy_observation.numel() == slot_count, "policy_energy_observation must be float64[slots]");
    TORCH_CHECK(health.scalar_type() == torch::kFloat32 && health.dim() == 1 && health.numel() == slot_count, "health must be float32[slots]");
    TORCH_CHECK(injury.scalar_type() == torch::kFloat32 && injury.dim() == 1 && injury.numel() == slot_count, "injury must be float32[slots]");
    TORCH_CHECK(scar.scalar_type() == torch::kFloat32 && scar.dim() == 1 && scar.numel() == slot_count, "scar must be float32[slots]");
    TORCH_CHECK(repro_cooldown.scalar_type() == torch::kInt32 && repro_cooldown.dim() == 1 && repro_cooldown.numel() == slot_count, "repro_cooldown must be int32[slots]");
    TORCH_CHECK(repro_threshold.scalar_type() == torch::kFloat32 && repro_threshold.dim() == 1 && repro_threshold.numel() == slot_count, "repro_threshold must be float32[slots]");
    TORCH_CHECK(slot_free_queue.scalar_type() == torch::kInt64 && slot_free_queue.dim() == 1, "slot_free_queue must be int64[queue]");
    TORCH_CHECK(slot_free_queue_count.scalar_type() == torch::kInt64 && slot_free_queue_count.dim() == 1 && slot_free_queue_count.numel() == 1, "slot_free_queue_count must be int64[1]");
    TORCH_CHECK(slot_free_queue_head.scalar_type() == torch::kInt64 && slot_free_queue_head.dim() == 1 && slot_free_queue_head.numel() == 1, "slot_free_queue_head must be int64[1]");
    TORCH_CHECK(pcg64_state_words.scalar_type() == torch::kUInt64 && pcg64_state_words.dim() == 1 && pcg64_state_words.numel() == 4, "pcg64_state_words must be uint64[4]");
    TORCH_CHECK(pcg64_buffer_words.scalar_type() == torch::kInt64 && pcg64_buffer_words.dim() == 1 && pcg64_buffer_words.numel() == 2, "pcg64_buffer_words must be int64[2]");
    TORCH_CHECK(policy_np_semantic_draws.scalar_type() == torch::kInt64 && policy_np_semantic_draws.dim() == 1 && policy_np_semantic_draws.numel() == 1, "policy_np_semantic_draws must be int64[1]");
    TORCH_CHECK(runner_pcg64_state_words.scalar_type() == torch::kUInt64 && runner_pcg64_state_words.dim() == 1 && runner_pcg64_state_words.numel() == 4, "runner_pcg64_state_words must be uint64[4]");
    TORCH_CHECK(runner_pcg64_buffer_words.scalar_type() == torch::kInt64 && runner_pcg64_buffer_words.dim() == 1 && runner_pcg64_buffer_words.numel() == 2, "runner_pcg64_buffer_words must be int64[2]");
    TORCH_CHECK(metrics_np_semantic_draws.scalar_type() == torch::kInt64 && metrics_np_semantic_draws.dim() == 1 && metrics_np_semantic_draws.numel() == 1, "metrics_np_semantic_draws must be int64[1]");
    TORCH_CHECK(max_births >= 0 && max_births <= slot_count, "max_births must be within slot capacity");

    auto int64_options = slot_to_agent_id.options().dtype(torch::kInt64);
    auto int32_options = slot_to_agent_id.options().dtype(torch::kInt32);
    auto float_options = slot_to_agent_id.options().dtype(torch::kFloat32);
    auto bool_options = slot_to_agent_id.options().dtype(torch::kBool);
    auto planned_repro_cooldown = repro_cooldown.clone();
    auto planned_parent_slots = torch::full({max_births}, -1, int64_options);
    auto planned_child_slots = torch::full({max_births}, -1, int64_options);
    auto planned_spawn_positions = torch::full({max_births, 2}, -1, int32_options);
    auto planned_child_energy = torch::full(
        {max_births},
        std::numeric_limits<float>::quiet_NaN(),
        float_options
    );
    auto decision_parent_slots = torch::full({slot_count}, -1, int64_options);
    auto decision_reason_codes = torch::full({slot_count}, -1, int32_options);
    auto decision_candidate_counts = torch::zeros({slot_count}, int32_options);
    auto decision_draw_starts = torch::zeros({slot_count}, int64_options);
    auto decision_draw_ends = torch::zeros({slot_count}, int64_options);
    auto decision_selected_positions = torch::full({slot_count, 2}, -1, int32_options);
    auto decision_selected_reserved = torch::zeros({slot_count}, bool_options);
    auto reason_counts = torch::zeros({REASON_COUNT}, int64_options);
    auto planned_count = torch::zeros({1}, int64_options);
    auto decision_count = torch::zeros({1}, int64_options);
    auto spawn_rng_draws_added = torch::zeros({1}, int64_options);
    auto spawn_rng_semantic_draws_added = torch::zeros({1}, int64_options);
    auto runner_rng_draws_added = torch::zeros({1}, int64_options);
    auto runner_rng_semantic_draws_added = torch::zeros({1}, int64_options);
    auto status = torch::zeros({1}, int32_options);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    asexual_birth_plan_kernel<<<1, 1, 0, stream>>>(
        slot_to_agent_id.data_ptr<int64_t>(),
        alive_mask.data_ptr<bool>(),
        policy_use_gru.data_ptr<bool>(),
        pos_y.data_ptr<int32_t>(),
        pos_x.data_ptr<int32_t>(),
        agent_grid.data_ptr<int32_t>(),
        energy.data_ptr<float>(),
        policy_energy_observation.data_ptr<double>(),
        health.data_ptr<float>(),
        injury.data_ptr<float>(),
        scar.data_ptr<float>(),
        repro_cooldown.data_ptr<int32_t>(),
        planned_repro_cooldown.data_ptr<int32_t>(),
        repro_threshold.data_ptr<float>(),
        slot_free_queue.data_ptr<int64_t>(),
        slot_free_queue_count.data_ptr<int64_t>(),
        slot_free_queue_head.data_ptr<int64_t>(),
        pcg64_state_words.data_ptr<uint64_t>(),
        pcg64_buffer_words.data_ptr<int64_t>(),
        policy_np_semantic_draws.data_ptr<int64_t>(),
        runner_pcg64_state_words.data_ptr<uint64_t>(),
        runner_pcg64_buffer_words.data_ptr<int64_t>(),
        metrics_np_semantic_draws.data_ptr<int64_t>(),
        planned_parent_slots.data_ptr<int64_t>(),
        planned_child_slots.data_ptr<int64_t>(),
        planned_spawn_positions.data_ptr<int32_t>(),
        planned_child_energy.data_ptr<float>(),
        decision_parent_slots.data_ptr<int64_t>(),
        decision_reason_codes.data_ptr<int32_t>(),
        decision_candidate_counts.data_ptr<int32_t>(),
        decision_draw_starts.data_ptr<int64_t>(),
        decision_draw_ends.data_ptr<int64_t>(),
        decision_selected_positions.data_ptr<int32_t>(),
        decision_selected_reserved.data_ptr<bool>(),
        reason_counts.data_ptr<int64_t>(),
        planned_count.data_ptr<int64_t>(),
        decision_count.data_ptr<int64_t>(),
        spawn_rng_draws_added.data_ptr<int64_t>(),
        spawn_rng_semantic_draws_added.data_ptr<int64_t>(),
        runner_rng_draws_added.data_ptr<int64_t>(),
        runner_rng_semantic_draws_added.data_ptr<int64_t>(),
        status.data_ptr<int32_t>(),
        slot_count,
        agent_grid.size(0),
        agent_grid.size(1),
        slot_free_queue.size(0),
        max_births,
        parent_energy_share,
        static_cast<float>(min_energy_buffer),
        injury_repro_k_injury,
        injury_repro_k_scar,
        injury_birth_fail_scar,
        injury_birth_fail_injury,
        env_policy_alias,
        requires_mate,
        injury_gate_active,
        conception_gate_active,
        gestation_gate_active
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return {
        repro_cooldown,
        slot_free_queue_head,
        pcg64_state_words,
        pcg64_buffer_words,
        policy_np_semantic_draws,
        planned_parent_slots,
        planned_child_slots,
        planned_spawn_positions,
        planned_child_energy,
        decision_parent_slots,
        decision_reason_codes,
        decision_candidate_counts,
        decision_draw_starts,
        decision_draw_ends,
        decision_selected_positions,
        decision_selected_reserved,
        reason_counts,
        planned_count,
        decision_count,
        spawn_rng_draws_added,
        spawn_rng_semantic_draws_added,
        status,
        runner_pcg64_state_words,
        runner_pcg64_buffer_words,
        metrics_np_semantic_draws,
        runner_rng_draws_added,
        runner_rng_semantic_draws_added,
    };
}
