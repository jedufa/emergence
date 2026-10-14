#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace {

constexpr int64_t kObservationWidth = 74;
constexpr int64_t kHiddenWidth = 64;
constexpr int64_t kActions = 7;
constexpr int64_t kPolicyPackedWidth = 9415;
constexpr float kGumbelEps = 1.0e-6f;

void require_cuda_contiguous(const torch::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be CUDA resident");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

__device__ __forceinline__ float safe_norm(float value, float maximum) {
    if (maximum <= 0.0f) return 0.0f;
    return fminf(1.0f, fmaxf(0.0f, value / maximum));
}

__global__ void observe_kernel(
    float* __restrict__ out, const bool* __restrict__ alive,
    const int64_t* __restrict__ pos_y, const int64_t* __restrict__ pos_x,
    const int64_t* __restrict__ occupancy, const float* __restrict__ energy,
    const float* __restrict__ health, const float* __restrict__ food,
    const float* __restrict__ cover, const float* __restrict__ shelter,
    const float* __restrict__ scent, const float* __restrict__ waste,
    const float* __restrict__ movement, int64_t batch, int64_t capacity,
    int64_t width, float max_energy, float start_health, float food_max,
    float food_threshold, float scent_max, float waste_max, int64_t density_radius,
    bool density_include_self, bool density_normalize, float movement_min,
    float movement_max
) {
    const int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= batch * capacity) return;
    float* row_out = out + row * kObservationWidth;
    if (!alive[row]) {
        for (int64_t i = 0; i < kObservationWidth; ++i) row_out[i] = 0.0f;
        return;
    }
    const int64_t world = row / capacity;
    const int64_t grid_offset = world * width * width;
    const int64_t cy = pos_y[row];
    const int64_t cx = pos_x[row];
    const float movement_span = movement_max - movement_min;
    float density_denom = static_cast<float>(
        (2 * density_radius + 1) * (2 * density_radius + 1)
        - (density_include_self ? 0 : 1));
    if (density_denom <= 0.0f) density_denom = 1.0f;
    int64_t t = 0;
    for (int64_t dy = -1; dy <= 1; ++dy) {
        for (int64_t dx = -1; dx <= 1; ++dx, ++t) {
            const int64_t y = cy + dy;
            const int64_t x = cx + dx;
            float legacy = 0.0f, food_norm = 0.0f, c = 0.0f, sh = 0.0f;
            float scent_norm = 0.0f, waste_norm = 0.0f, density = 0.0f;
            float movement_norm = 0.0f;
            if (y >= 0 && y < width && x >= 0 && x < width) {
                const int64_t cell = grid_offset + y * width + x;
                if (occupancy[cell] != -1) legacy = -1.0f;
                else if (food[cell] > food_threshold) legacy = 1.0f;
                food_norm = safe_norm(food[cell], food_max);
                c = fminf(1.0f, fmaxf(0.0f, cover[cell]));
                sh = fminf(1.0f, fmaxf(0.0f, shelter[cell]));
                scent_norm = safe_norm(scent[cell], scent_max);
                waste_norm = safe_norm(waste[cell], waste_max);
                int count = 0;
                for (int64_t yy = max(int64_t(0), y - density_radius);
                     yy < min(width, y + density_radius + 1); ++yy) {
                    for (int64_t xx = max(int64_t(0), x - density_radius);
                         xx < min(width, x + density_radius + 1); ++xx) {
                        count += occupancy[grid_offset + yy * width + xx] != -1;
                    }
                }
                if (!density_include_self && occupancy[cell] != -1) --count;
                density = density_normalize
                    ? fminf(1.0f, fmaxf(0.0f, static_cast<float>(count)))
                    : safe_norm(static_cast<float>(count), density_denom);
                if (movement_span > 0.0f) {
                    movement_norm = fminf(1.0f, fmaxf(
                        0.0f, (movement[cell] - movement_min) / movement_span));
                }
            }
            row_out[t] = legacy;
            row_out[9 + t] = food_norm;
            row_out[18 + t] = c;
            row_out[27 + t] = sh;
            row_out[36 + t] = scent_norm;
            row_out[45 + t] = waste_norm;
            row_out[54 + t] = density;
            row_out[63 + t] = movement_norm;
        }
    }
    row_out[72] = energy[row] / max_energy;
    row_out[73] = health[row] / start_health;
}

// This row-local arithmetic intentionally mirrors the source-current
// fixed_mlp_policy_logits_from_pack_exact_v1 helper.  The serial order is part
// of the zero-tolerance C0 trajectory contract.
__device__ __forceinline__ void policy_logits_from_pack_exact(
    const float* const pack,
    const float* const observation,
    const float* const drives,
    const float attack_bias,
    const float action_temperature,
    const float drive_logit_gain,
    float* const row_logits,
    const float w_hunger_gather,
    const float w_pain_rest,
    const float w_fatigue_rest,
    const float w_pain_attack,
    const float w_fear_attack
) {
    float h1[kHiddenWidth];
    float h2[kHiddenWidth];
    int64_t offset = 0;
    for (int64_t hidden = 0; hidden < kHiddenWidth; ++hidden) {
        float value = pack[kHiddenWidth * kObservationWidth + hidden];
        const float* const weights = pack + hidden * kObservationWidth;
        for (int64_t feature = 0; feature < kObservationWidth; ++feature) {
            value += weights[feature] * observation[feature];
        }
        h1[hidden] = tanhf(value);
    }
    offset += kHiddenWidth * kObservationWidth + kHiddenWidth;
    for (int64_t hidden = 0; hidden < kHiddenWidth; ++hidden) {
        float value = pack[offset + kHiddenWidth * kHiddenWidth + hidden];
        const float* const weights = pack + offset + hidden * kHiddenWidth;
        for (int64_t input = 0; input < kHiddenWidth; ++input) {
            value += weights[input] * h1[input];
        }
        h2[hidden] = tanhf(value);
    }
    offset += kHiddenWidth * kHiddenWidth + kHiddenWidth;
    const float temperature = action_temperature < 0.05f
        ? 0.05f : action_temperature;
    const float gain = drive_logit_gain < 0.0f ? 0.0f : drive_logit_gain;
    for (int64_t action = 0; action < kActions; ++action) {
        const float* const weights = pack + offset + action * kHiddenWidth;
        float score = pack[offset + kActions * kHiddenWidth + action];
        for (int64_t hidden = 0; hidden < kHiddenWidth; ++hidden) {
            score += weights[hidden] * h2[hidden];
        }
        if (action == 6) score += attack_bias;
        score /= temperature;
        if (action == 4) {
            score += gain * (w_hunger_gather * drives[0]);
        } else if (action == 5) {
            score += gain * (w_pain_rest * drives[1]
                + w_fatigue_rest * drives[2]);
        } else if (action == 6) {
            score -= gain * (w_pain_attack * drives[1]
                + w_fear_attack * drives[3]);
        }
        row_logits[action] = score;
    }
}

__global__ void policy_logits_kernel(
    float* __restrict__ logits, const float* __restrict__ packed_weights,
    const float* __restrict__ observations, const float* __restrict__ drives,
    const float* __restrict__ attack_bias,
    const float* __restrict__ action_temperature,
    const float* __restrict__ drive_logit_gain, int64_t rows,
    float w_hunger_gather, float w_pain_rest, float w_fatigue_rest,
    float w_pain_attack, float w_fear_attack
) {
    const int64_t row = static_cast<int64_t>(blockIdx.x);
    if (threadIdx.x != 0 || row >= rows) return;
    policy_logits_from_pack_exact(
        packed_weights + row * kPolicyPackedWidth,
        observations + row * kObservationWidth,
        drives + row * 4,
        attack_bias[row],
        action_temperature[row],
        drive_logit_gain[row],
        logits + row * kActions,
        w_hunger_gather,
        w_pain_rest,
        w_fatigue_rest,
        w_pain_attack,
        w_fear_attack
    );
}

__global__ void select_kernel(
    int64_t* __restrict__ actions, const float* __restrict__ logits,
    const double* __restrict__ uniforms, const bool* __restrict__ alive,
    int64_t* __restrict__ cursors, int64_t batch, int64_t capacity
) {
    const int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= batch * capacity) return;
    if (!alive[row]) { actions[row] = -1; return; }
    int64_t best = 0;
    float u = static_cast<float>(uniforms[row * kActions]);
    u = fminf(1.0f - kGumbelEps, fmaxf(kGumbelEps, u));
    float best_score = logits[row * kActions] - logf(-logf(u));
    for (int64_t action = 1; action < kActions; ++action) {
        u = static_cast<float>(uniforms[row * kActions + action]);
        u = fminf(1.0f - kGumbelEps, fmaxf(kGumbelEps, u));
        const float score = logits[row * kActions + action] - logf(-logf(u));
        if (score > best_score) { best_score = score; best = action; }
    }
    actions[row] = best;
    if (row % capacity == 0) cursors[row / capacity] += 1;
}

__device__ __forceinline__ float base_action_cost(
    int64_t action, double move_cost, double gather_cost, double rest_cost,
    double attack_cost, double action_mult, double rest_mult
) {
    double cost = action <= 3 ? move_cost :
        (action == 4 ? gather_cost : (action == 5 ? rest_cost : attack_cost));
    if (action == 5) cost *= rest_mult;
    return static_cast<float>(cost * action_mult);
}

__global__ void settle_kernel(
    const int64_t* __restrict__ actions, const bool* __restrict__ alive,
    const int64_t* __restrict__ agent_ids, int64_t* __restrict__ pos_y,
    int64_t* __restrict__ pos_x, int64_t* __restrict__ id_grid,
    int64_t* __restrict__ slot_grid, float* __restrict__ food,
    const float* __restrict__ movement, float* __restrict__ energy,
    double* __restrict__ collected, bool* __restrict__ moved,
    bool* __restrict__ gathered, float* __restrict__ food_gained,
    int64_t* __restrict__ attempted,
    int64_t* __restrict__ committed, int64_t* __restrict__ gathers,
    int64_t capacity, int64_t width, double move_cost, double gather_cost,
    double rest_cost, double attack_cost, double action_mult,
    double rest_mult, double max_energy, double food_energy,
    double gather_amount, double food_threshold, bool movement_cost_apply,
    double movement_cost_scale
) {
    const int64_t world = blockIdx.x;
    if (threadIdx.x != 0) return;
    const int64_t row_offset = world * capacity;
    const int64_t grid_offset = world * width * width;
    attempted[world] = 0; committed[world] = 0; gathers[world] = 0;
    for (int64_t row = 0; row < capacity; ++row) {
        const int64_t flat = row_offset + row;
        moved[flat] = false; gathered[flat] = false; food_gained[flat] = 0.0f;
        if (!alive[flat]) continue;
        const int64_t action = actions[flat];
        if (action < 0 || action > 3) continue;
        ++attempted[world];
        int64_t y = pos_y[flat], x = pos_x[flat], ny = y, nx = x;
        if (action == 0) --ny; else if (action == 1) ++ny;
        else if (action == 2) ++nx; else --nx;
        if (ny < 0 || ny >= width || nx < 0 || nx >= width) continue;
        const int64_t destination = grid_offset + ny * width + nx;
        if (id_grid[destination] != -1) continue;
        const int64_t origin = grid_offset + y * width + x;
        if (id_grid[origin] == agent_ids[flat]) {
            id_grid[origin] = -1; slot_grid[origin] = -1;
        }
        id_grid[destination] = agent_ids[flat];
        slot_grid[destination] = row;
        pos_y[flat] = ny; pos_x[flat] = nx; moved[flat] = true;
        ++committed[world];
    }
    for (int64_t row = 0; row < capacity; ++row) {
        const int64_t flat = row_offset + row;
        if (!alive[flat] || actions[flat] != 4) continue;
        const int64_t cell = grid_offset + pos_y[flat] * width + pos_x[flat];
        const double amount = static_cast<double>(food[cell]);
        if (amount <= food_threshold) continue;
        const double harvest = fmin(amount, gather_amount > 0.0 ? gather_amount : 1.0);
        const double post = fmax(0.0, amount - harvest);
        food[cell] = static_cast<float>(post);
        const float gained = static_cast<float>(food_energy * harvest /
            (gather_amount > 0.0 ? gather_amount : 1.0));
        food_gained[flat] = gained;
        collected[flat] += static_cast<double>(gained);
        gathered[flat] = true; ++gathers[world];
    }
    for (int64_t row = 0; row < capacity; ++row) {
        const int64_t flat = row_offset + row;
        if (!alive[flat]) continue;
        float cost = base_action_cost(actions[flat], move_cost, gather_cost,
            rest_cost, attack_cost, action_mult, rest_mult);
        if (moved[flat] && movement_cost_apply) {
            const int64_t cell = grid_offset + pos_y[flat] * width + pos_x[flat];
            const float mult = fmaxf(0.0f, 1.0f + static_cast<float>(movement_cost_scale)
                * (movement[cell] - 1.0f));
            cost = fmaxf(0.0f, cost * mult);
        }
        energy[flat] = fminf(static_cast<float>(max_energy),
            fmaxf(0.0f, energy[flat] - cost + food_gained[flat]));
    }
}

}  // namespace

void tensor_world_c1_observe_cuda(
    torch::Tensor observations, torch::Tensor alive, torch::Tensor pos_y,
    torch::Tensor pos_x, torch::Tensor agent_id_grid, torch::Tensor energy,
    torch::Tensor health, torch::Tensor food_amount, torch::Tensor cover,
    torch::Tensor shelter, torch::Tensor scent, torch::Tensor waste,
    torch::Tensor movement_cost, double max_energy, double start_health,
    double food_max, double food_threshold, double scent_max, double waste_max,
    int64_t density_radius, bool density_include_self,
    bool density_normalize, double movement_min, double movement_max
) {
    for (const auto& item : {observations, alive, pos_y, pos_x, agent_id_grid,
            energy, health, food_amount, cover, shelter, scent, waste, movement_cost})
        require_cuda_contiguous(item, "Tensor World observation tensor");
    TORCH_CHECK(observations.scalar_type() == torch::kFloat32 && observations.dim() == 3
        && observations.size(2) == kObservationWidth, "observations must be float32[B,C,74]");
    const auto batch = alive.size(0), capacity = alive.size(1), width = food_amount.size(1);
    const int blocks = static_cast<int>((batch * capacity + 127) / 128);
    observe_kernel<<<blocks, 128, 0, at::cuda::getCurrentCUDAStream()>>>(
        observations.data_ptr<float>(), alive.data_ptr<bool>(), pos_y.data_ptr<int64_t>(),
        pos_x.data_ptr<int64_t>(), agent_id_grid.data_ptr<int64_t>(), energy.data_ptr<float>(),
        health.data_ptr<float>(), food_amount.data_ptr<float>(), cover.data_ptr<float>(),
        shelter.data_ptr<float>(), scent.data_ptr<float>(), waste.data_ptr<float>(),
        movement_cost.data_ptr<float>(), batch, capacity, width, static_cast<float>(max_energy),
        static_cast<float>(start_health), static_cast<float>(food_max),
        static_cast<float>(food_threshold),
        static_cast<float>(scent_max), static_cast<float>(waste_max), density_radius,
        density_include_self, density_normalize, static_cast<float>(movement_min),
        static_cast<float>(movement_max));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void tensor_world_c1_policy_logits_cuda(
    torch::Tensor logits, torch::Tensor packed_weights,
    torch::Tensor observations, torch::Tensor drives,
    torch::Tensor attack_bias, torch::Tensor action_temperature,
    torch::Tensor drive_logit_gain, double w_hunger_gather,
    double w_pain_rest, double w_fatigue_rest,
    double w_pain_attack, double w_fear_attack
) {
    for (const auto& item : {logits, packed_weights, observations, drives,
            attack_bias, action_temperature, drive_logit_gain})
        require_cuda_contiguous(item, "Tensor World policy tensor");
    const auto rows = packed_weights.size(0);
    TORCH_CHECK(
        logits.scalar_type() == torch::kFloat32 && logits.dim() == 2
            && logits.size(0) == rows && logits.size(1) == kActions,
        "logits must be float32[rows,7]");
    TORCH_CHECK(
        packed_weights.scalar_type() == torch::kFloat32
            && packed_weights.dim() == 2
            && packed_weights.size(1) == kPolicyPackedWidth,
        "packed_weights must be float32[rows,9415]");
    TORCH_CHECK(
        observations.scalar_type() == torch::kFloat32
            && observations.sizes() == torch::IntArrayRef({rows, kObservationWidth}),
        "observations must be float32[rows,74]");
    TORCH_CHECK(
        drives.scalar_type() == torch::kFloat32 && drives.dim() == 2
            && drives.size(0) == rows && drives.size(1) == 4,
        "drives must be float32[rows,4]");
    for (const auto& item : {attack_bias, action_temperature, drive_logit_gain})
        TORCH_CHECK(item.scalar_type() == torch::kFloat32 && item.dim() == 1
            && item.size(0) == rows, "policy scalar vectors must be float32[rows]");
    policy_logits_kernel<<<
        static_cast<unsigned int>(rows), 1, 0,
        at::cuda::getCurrentCUDAStream(packed_weights.get_device())
    >>>(
        logits.data_ptr<float>(), packed_weights.data_ptr<float>(),
        observations.data_ptr<float>(), drives.data_ptr<float>(),
        attack_bias.data_ptr<float>(), action_temperature.data_ptr<float>(),
        drive_logit_gain.data_ptr<float>(), rows,
        static_cast<float>(w_hunger_gather), static_cast<float>(w_pain_rest),
        static_cast<float>(w_fatigue_rest), static_cast<float>(w_pain_attack),
        static_cast<float>(w_fear_attack));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void tensor_world_c1_select_cuda(
    torch::Tensor actions, torch::Tensor logits, torch::Tensor uniforms,
    torch::Tensor alive, torch::Tensor action_cursor
) {
    for (const auto& item : {actions, logits, uniforms, alive, action_cursor})
        require_cuda_contiguous(item, "Tensor World action tensor");
    const auto batch = alive.size(0), capacity = alive.size(1);
    const int blocks = static_cast<int>((batch * capacity + 127) / 128);
    select_kernel<<<blocks, 128, 0, at::cuda::getCurrentCUDAStream()>>>(
        actions.data_ptr<int64_t>(), logits.data_ptr<float>(), uniforms.data_ptr<double>(),
        alive.data_ptr<bool>(), action_cursor.data_ptr<int64_t>(), batch, capacity);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void tensor_world_c1_settle_cuda(
    torch::Tensor actions, torch::Tensor alive, torch::Tensor agent_ids,
    torch::Tensor pos_y, torch::Tensor pos_x, torch::Tensor agent_id_grid,
    torch::Tensor slot_grid, torch::Tensor food_amount, torch::Tensor movement_cost,
    torch::Tensor energy, torch::Tensor energy_collected, torch::Tensor moved,
    torch::Tensor gathered, torch::Tensor food_gained, torch::Tensor moves_attempted,
    torch::Tensor moves_committed, torch::Tensor gathers_committed,
    double move_cost, double gather_cost, double rest_cost, double attack_cost,
    double action_cost_mult, double rest_cost_mult, double max_energy,
    double food_energy, double gather_amount, double food_threshold,
    bool movement_cost_apply, double movement_cost_scale
) {
    for (const auto& item : {actions, alive, agent_ids, pos_y, pos_x, agent_id_grid,
            slot_grid, food_amount, movement_cost, energy, energy_collected, moved,
            gathered, food_gained, moves_attempted, moves_committed, gathers_committed})
        require_cuda_contiguous(item, "Tensor World settlement tensor");
    settle_kernel<<<static_cast<unsigned int>(alive.size(0)), 1, 0,
        at::cuda::getCurrentCUDAStream()>>>(actions.data_ptr<int64_t>(), alive.data_ptr<bool>(),
        agent_ids.data_ptr<int64_t>(), pos_y.data_ptr<int64_t>(), pos_x.data_ptr<int64_t>(),
        agent_id_grid.data_ptr<int64_t>(), slot_grid.data_ptr<int64_t>(),
        food_amount.data_ptr<float>(), movement_cost.data_ptr<float>(), energy.data_ptr<float>(),
        energy_collected.data_ptr<double>(), moved.data_ptr<bool>(), gathered.data_ptr<bool>(),
        food_gained.data_ptr<float>(),
        moves_attempted.data_ptr<int64_t>(), moves_committed.data_ptr<int64_t>(),
        gathers_committed.data_ptr<int64_t>(), alive.size(1), food_amount.size(1), move_cost,
        gather_cost, rest_cost, attack_cost, action_cost_mult, rest_cost_mult, max_energy,
        food_energy, gather_amount, food_threshold, movement_cost_apply, movement_cost_scale);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
