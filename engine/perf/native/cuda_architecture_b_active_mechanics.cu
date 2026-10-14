#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

namespace {

constexpr int32_t ACT_GATHER = 4;
constexpr int32_t ACT_REST = 5;
constexpr int32_t ACT_ATTACK = 6;

constexpr int32_t STATUS_OK = 0;
constexpr int32_t STATUS_INVALID_SLOT_PARTITION = 1;
constexpr int32_t STATUS_NONFINITE_LIVE_STATE = 2;
constexpr int32_t STATUS_INVALID_DRIVE_CONTROL = 3;
constexpr int32_t STATUS_MISSING_DRIVE_NOISE = 4;
constexpr int32_t STATUS_COUNTER_OVERFLOW = 5;

constexpr int64_t COUNT_PROCESSED_ROWS = 0;
constexpr int64_t COUNT_SURVIVAL_ADVANCES = 1;
constexpr int64_t COUNT_REST_CHARGES = 2;
constexpr int64_t COUNT_REST_DEFICIT_DAMAGE = 3;
constexpr int64_t COUNT_DRIVE_NOISE_ROWS = 4;
constexpr int64_t COUNT_FIELDS = 5;

__device__ __forceinline__ float f32_add(const float left, const float right) {
    return __fadd_rn(left, right);
}

__device__ __forceinline__ float f32_sub(const float left, const float right) {
    return __fsub_rn(left, right);
}

__device__ __forceinline__ float f32_mul(const float left, const float right) {
    return __fmul_rn(left, right);
}

__device__ __forceinline__ float f32_div(const float left, const float right) {
    return __fdiv_rn(left, right);
}

__device__ __forceinline__ float clamp01(const float value) {
    return value < 0.0f ? 0.0f : (value > 1.0f ? 1.0f : value);
}

__device__ __forceinline__ float max_zero(const float value) {
    return value < 0.0f ? 0.0f : value;
}

__device__ __forceinline__ int32_t normalized_action(const int32_t action) {
    return action < 0 || action > ACT_ATTACK ? ACT_REST : action;
}

__global__ void active_mechanics_batched_kernel(
    const int64_t* __restrict__ slot_to_agent_id,
    const bool* __restrict__ alive_mask,
    const int32_t* __restrict__ actions_by_slot,
    float* __restrict__ energy,
    float* __restrict__ health,
    const float* __restrict__ injury,
    const int32_t* __restrict__ age,
    double* __restrict__ policy_energy_observation,
    int32_t* __restrict__ rest_streak,
    int32_t* __restrict__ survival_steps,
    const float* __restrict__ gru_upkeep_cost,
    const float* __restrict__ cost_charge_scalar,
    const float* __restrict__ cost_charge_maturation_scalar,
    float* __restrict__ drives,
    const float* __restrict__ drive_inertia,
    const float* __restrict__ drive_noise,
    float* __restrict__ prev_drive_delta,
    float* __restrict__ drive_volatility,
    float* __restrict__ drive_jitter,
    float* __restrict__ episode0_drive_jitter_sum,
    float* __restrict__ last_energy_delta,
    float* __restrict__ last_health_delta,
    float* __restrict__ last_injury_delta,
    int32_t* __restrict__ selfharm_events,
    int32_t* __restrict__ episode0_selfharm_events,
    const float* __restrict__ pre_energy,
    const float* __restrict__ pre_health,
    const float* __restrict__ pre_injury,
    const float* __restrict__ drive_noise_samples,
    const bool* __restrict__ drive_noise_sample_valid,
    int32_t* __restrict__ status,
    int64_t* __restrict__ counts,
    const int64_t capacity,
    const int64_t cell_count,
    const bool enable_rest_charge,
    const int32_t rest_streak_free,
    const float rest_charge_energy,
    const float rest_charge_growth,
    const float rest_deficit_to_health,
    const int32_t developmental_maturity_age,
    const int32_t episode0_k,
    const float max_energy,
    const float start_health,
    const float drive_hunger_gain,
    const float drive_hunger_decay,
    const float drive_pain_gain,
    const float drive_pain_decay,
    const float drive_fatigue_gain,
    const float drive_fatigue_rest_relief,
    const float drive_fatigue_decay,
    const float drive_fear_gain,
    const float drive_fear_decay
) {
    if (threadIdx.x != 0) {
        return;
    }

    const int64_t cell = static_cast<int64_t>(blockIdx.x);
    if (cell >= cell_count) {
        return;
    }
    const int64_t slot_offset = cell * capacity;
    const int64_t drive_offset = slot_offset * 4;
    slot_to_agent_id += slot_offset;
    alive_mask += slot_offset;
    actions_by_slot += slot_offset;
    energy += slot_offset;
    health += slot_offset;
    injury += slot_offset;
    age += slot_offset;
    policy_energy_observation += slot_offset;
    rest_streak += slot_offset;
    survival_steps += slot_offset;
    gru_upkeep_cost += slot_offset;
    cost_charge_scalar += slot_offset;
    cost_charge_maturation_scalar += slot_offset;
    drives += drive_offset;
    drive_inertia += slot_offset;
    drive_noise += slot_offset;
    prev_drive_delta += drive_offset;
    drive_volatility += slot_offset;
    drive_jitter += slot_offset;
    episode0_drive_jitter_sum += slot_offset;
    last_energy_delta += slot_offset;
    last_health_delta += slot_offset;
    last_injury_delta += slot_offset;
    selfharm_events += slot_offset;
    episode0_selfharm_events += slot_offset;
    pre_energy += slot_offset;
    pre_health += slot_offset;
    pre_injury += slot_offset;
    drive_noise_samples += drive_offset;
    drive_noise_sample_valid += slot_offset;
    status += cell;
    counts += cell * COUNT_FIELDS;

    status[0] = STATUS_OK;
    for (int64_t index = 0; index < COUNT_FIELDS; ++index) {
        counts[index] = 0;
    }

    // Validate every assigned row before the first authoritative write. Dead
    // rows may retain NaN diagnostic shadows and are deliberately ignored.
    for (int64_t slot = 0; slot < capacity; ++slot) {
        const bool assigned = slot_to_agent_id[slot] >= 0;
        if (alive_mask[slot] != assigned) {
            status[0] = STATUS_INVALID_SLOT_PARTITION;
            return;
        }
        if (!assigned) {
            continue;
        }
        const int64_t drive_offset = slot * 4;
        if (
            !isfinite(energy[slot]) || !isfinite(health[slot])
            || !isfinite(injury[slot]) || !isfinite(pre_energy[slot])
            || !isfinite(pre_health[slot]) || !isfinite(pre_injury[slot])
            || !isfinite(gru_upkeep_cost[slot])
            || !isfinite(cost_charge_scalar[slot])
            || !isfinite(cost_charge_maturation_scalar[slot])
            || !isfinite(drive_volatility[slot]) || !isfinite(drive_jitter[slot])
            || !isfinite(episode0_drive_jitter_sum[slot])
            || !isfinite(last_energy_delta[slot])
            || !isfinite(last_health_delta[slot])
            || !isfinite(last_injury_delta[slot])
        ) {
            status[0] = STATUS_NONFINITE_LIVE_STATE;
            return;
        }
        for (int64_t drive = 0; drive < 4; ++drive) {
            if (
                !isfinite(drives[drive_offset + drive])
                || !isfinite(prev_drive_delta[drive_offset + drive])
            ) {
                status[0] = STATUS_NONFINITE_LIVE_STATE;
                return;
            }
        }
        if (
            !isfinite(drive_inertia[slot]) || drive_inertia[slot] < 0.0f
            || !isfinite(drive_noise[slot]) || drive_noise[slot] < 0.0f
        ) {
            status[0] = STATUS_INVALID_DRIVE_CONTROL;
            return;
        }
        if (drive_noise[slot] > 0.0f) {
            if (!drive_noise_sample_valid[slot]) {
                status[0] = STATUS_MISSING_DRIVE_NOISE;
                return;
            }
            for (int64_t drive = 0; drive < 4; ++drive) {
                if (!isfinite(drive_noise_samples[drive_offset + drive])) {
                    status[0] = STATUS_MISSING_DRIVE_NOISE;
                    return;
                }
            }
        }
        if (
            rest_streak[slot] == std::numeric_limits<int32_t>::max()
            || survival_steps[slot] == std::numeric_limits<int32_t>::max()
            || selfharm_events[slot] == std::numeric_limits<int32_t>::max()
            || episode0_selfharm_events[slot] == std::numeric_limits<int32_t>::max()
        ) {
            status[0] = STATUS_COUNTER_OVERFLOW;
            return;
        }
        (void)age[slot];
    }

    for (int64_t slot = 0; slot < capacity; ++slot) {
        if (!alive_mask[slot] || slot_to_agent_id[slot] < 0) {
            continue;
        }
        const int32_t action = normalized_action(actions_by_slot[slot]);

        // The accepted CPU owner performs this rest gate after base action
        // mutation and before its action-time death decision.
        if (enable_rest_charge) {
            if (action == ACT_REST) {
                rest_streak[slot] += 1;
            } else {
                rest_streak[slot] = 0;
            }
            if (rest_streak[slot] > rest_streak_free) {
                const int32_t over = rest_streak[slot] - rest_streak_free;
                const float growth = f32_mul(
                    rest_charge_growth,
                    static_cast<float>(over - 1)
                );
                const float charge = f32_add(rest_charge_energy, growth);
                energy[slot] = f32_sub(energy[slot], charge);
                counts[COUNT_REST_CHARGES] += 1;
                if (energy[slot] < 0.0f) {
                    const float damage = f32_mul(-energy[slot], rest_deficit_to_health);
                    health[slot] = f32_sub(health[slot], damage);
                    energy[slot] = 0.0f;
                    counts[COUNT_REST_DEFICIT_DAMAGE] += 1;
                }
                policy_energy_observation[slot] = static_cast<double>(energy[slot]);
            }
        }

        // Rows terminal at the CPU action boundary do not receive upkeep or a
        // drive update. The following lifecycle slice owns their cull.
        if (energy[slot] <= 0.0f || health[slot] <= 0.0f) {
            continue;
        }

        const float gru_charge = max_zero(gru_upkeep_cost[slot]);
        if (gru_charge > 0.0f) {
            energy[slot] = max_zero(f32_sub(energy[slot], gru_charge));
        }
        float charge = max_zero(cost_charge_scalar[slot]);
        if (
            developmental_maturity_age > 0
            && survival_steps[slot] < developmental_maturity_age
        ) {
            charge = f32_add(
                charge,
                max_zero(cost_charge_maturation_scalar[slot])
            );
        }
        if (charge > 0.0f) {
            energy[slot] = max_zero(f32_sub(energy[slot], charge));
        }
        survival_steps[slot] += 1;
        policy_energy_observation[slot] = static_cast<double>(energy[slot]);

        const int64_t offset = slot * 4;
        const float prev0 = drives[offset + 0];
        const float prev1 = drives[offset + 1];
        const float prev2 = drives[offset + 2];
        const float prev3 = drives[offset + 3];
        const float e_normed = clamp01(f32_div(energy[slot], max_energy));
        const float h_normed = clamp01(f32_div(health[slot], start_health));
        const float de = f32_sub(energy[slot], pre_energy[slot]);
        const float dh = f32_sub(health[slot], pre_health[slot]);
        const float dinj = f32_sub(injury[slot], pre_injury[slot]);
        last_energy_delta[slot] = de;
        last_health_delta[slot] = dh;
        last_injury_delta[slot] = dinj;

        const float hunger_gap = f32_sub(1.0f, e_normed);
        const float hunger_gain_term = f32_mul(drive_hunger_gain, hunger_gap);
        const float target_hunger_start = f32_add(prev0, hunger_gain_term);
        const float hunger_delta_norm = f32_div(-de, max_energy);
        const float hunger_delta_term = f32_mul(0.04f, hunger_delta_norm);
        const float target_hunger_with_delta = f32_add(
            target_hunger_start,
            hunger_delta_term
        );
        float target_hunger = f32_sub(
            target_hunger_with_delta,
            drive_hunger_decay
        );
        if (action == ACT_GATHER && de > 0.0f) {
            const float gather_norm = f32_div(de, max_energy);
            const float gather_term = f32_mul(0.20f, gather_norm);
            target_hunger = f32_sub(target_hunger, gather_term);
        }
        target_hunger = clamp01(target_hunger);

        float target_pain = prev1;
        if (dinj > 0.0f) {
            target_pain = f32_add(
                target_pain,
                f32_mul(drive_pain_gain, dinj < 1.0f ? dinj : 1.0f)
            );
        }
        if (dh < 0.0f) {
            const float damage_norm = f32_div(-dh, start_health);
            target_pain = f32_add(
                target_pain,
                f32_mul(0.10f, damage_norm < 1.0f ? damage_norm : 1.0f)
            );
        }
        target_pain = clamp01(f32_sub(target_pain, drive_pain_decay));

        float target_fatigue = prev2;
        if (action == ACT_REST) {
            target_fatigue = f32_sub(target_fatigue, drive_fatigue_rest_relief);
        } else {
            target_fatigue = f32_add(target_fatigue, drive_fatigue_gain);
        }
        target_fatigue = clamp01(f32_sub(target_fatigue, drive_fatigue_decay));

        float target_fear = prev3;
        if (dinj > 0.0f) {
            target_fear = f32_add(
                target_fear,
                f32_mul(drive_fear_gain, dinj < 1.0f ? dinj : 1.0f)
            );
        }
        target_fear = clamp01(
            f32_sub(
                f32_add(
                    target_fear,
                    f32_mul(0.05f, f32_sub(1.0f, h_normed))
                ),
                drive_fear_decay
            )
        );

        float inertia = drive_inertia[slot];
        inertia = inertia < 0.0f ? 0.0f : (inertia > 0.999f ? 0.999f : inertia);
        const float inverse = f32_sub(1.0f, inertia);
        float next0 = f32_add(
            f32_mul(inertia, prev0),
            f32_mul(inverse, target_hunger)
        );
        float next1 = f32_add(
            f32_mul(inertia, prev1),
            f32_mul(inverse, target_pain)
        );
        float next2 = f32_add(
            f32_mul(inertia, prev2),
            f32_mul(inverse, target_fatigue)
        );
        float next3 = f32_add(
            f32_mul(inertia, prev3),
            f32_mul(inverse, target_fear)
        );
        const float noise_scale = drive_noise[slot];
        if (noise_scale > 0.0f) {
            next0 = f32_add(next0, f32_mul(noise_scale, drive_noise_samples[offset + 0]));
            next1 = f32_add(next1, f32_mul(noise_scale, drive_noise_samples[offset + 1]));
            next2 = f32_add(next2, f32_mul(noise_scale, drive_noise_samples[offset + 2]));
            next3 = f32_add(next3, f32_mul(noise_scale, drive_noise_samples[offset + 3]));
            counts[COUNT_DRIVE_NOISE_ROWS] += 1;
        }
        next0 = clamp01(next0);
        next1 = clamp01(next1);
        next2 = clamp01(next2);
        next3 = clamp01(next3);

        const float delta0 = f32_sub(next0, prev0);
        const float delta1 = f32_sub(next1, prev1);
        const float delta2 = f32_sub(next2, prev2);
        const float delta3 = f32_sub(next3, prev3);
        const float volatility_instant = static_cast<float>(
            fabs(static_cast<double>(delta0))
            + fabs(static_cast<double>(delta1))
            + fabs(static_cast<double>(delta2))
            + fabs(static_cast<double>(delta3))
        );
        drive_volatility[slot] = f32_add(
            drive_volatility[slot],
            volatility_instant
        );
        const float jitter_instant = static_cast<float>(
            fabs(static_cast<double>(delta0) - static_cast<double>(prev_drive_delta[offset + 0]))
            + fabs(static_cast<double>(delta1) - static_cast<double>(prev_drive_delta[offset + 1]))
            + fabs(static_cast<double>(delta2) - static_cast<double>(prev_drive_delta[offset + 2]))
            + fabs(static_cast<double>(delta3) - static_cast<double>(prev_drive_delta[offset + 3]))
        );
        drive_jitter[slot] = f32_add(drive_jitter[slot], jitter_instant);
        if (survival_steps[slot] <= episode0_k) {
            episode0_drive_jitter_sum[slot] = f32_add(
                episode0_drive_jitter_sum[slot],
                jitter_instant
            );
        }
        prev_drive_delta[offset + 0] = delta0;
        prev_drive_delta[offset + 1] = delta1;
        prev_drive_delta[offset + 2] = delta2;
        prev_drive_delta[offset + 3] = delta3;
        drives[offset + 0] = next0;
        drives[offset + 1] = next1;
        drives[offset + 2] = next2;
        drives[offset + 3] = next3;
        if (action == ACT_ATTACK && dinj > 0.0f) {
            selfharm_events[slot] += 1;
            if (survival_steps[slot] <= episode0_k) {
                episode0_selfharm_events[slot] += 1;
            }
        }
        counts[COUNT_PROCESSED_ROWS] += 1;
        counts[COUNT_SURVIVAL_ADVANCES] += 1;
    }
}

void check_cuda_contiguous(const torch::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be CUDA-resident");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void check_slot_tensor(
    const torch::Tensor& tensor,
    const torch::ScalarType dtype,
    const int64_t capacity,
    const char* name,
    const torch::Device& device
) {
    check_cuda_contiguous(tensor, name);
    TORCH_CHECK(tensor.device() == device, name, " must share the mechanics CUDA device");
    TORCH_CHECK(tensor.scalar_type() == dtype, name, " has the wrong dtype");
    TORCH_CHECK(
        tensor.dim() == 1 && tensor.size(0) == capacity,
        name,
        " must be a rank-1 slot bank"
    );
}

void check_batched_slot_tensor(
    const torch::Tensor& tensor,
    const torch::ScalarType dtype,
    const int64_t cell_count,
    const int64_t capacity,
    const char* name,
    const torch::Device& device
) {
    check_cuda_contiguous(tensor, name);
    TORCH_CHECK(tensor.device() == device, name, " must share the mechanics CUDA device");
    TORCH_CHECK(tensor.scalar_type() == dtype, name, " has the wrong dtype");
    TORCH_CHECK(
        tensor.dim() == 2
            && tensor.size(0) == cell_count
            && tensor.size(1) == capacity,
        name,
        " must be [cells,capacity]"
    );
}

double checked_finite(const double value, const char* name) {
    TORCH_CHECK(std::isfinite(value), name, " must be finite");
    return value;
}

}  // namespace

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
    const bool enable_rest_charge,
    const int64_t rest_streak_free,
    const double rest_charge_energy,
    const double rest_charge_growth,
    const double rest_deficit_to_health,
    const int64_t developmental_maturity_age,
    const int64_t episode0_k,
    const double max_energy,
    const double start_health,
    const double drive_hunger_gain,
    const double drive_hunger_decay,
    const double drive_pain_gain,
    const double drive_pain_decay,
    const double drive_fatigue_gain,
    const double drive_fatigue_rest_relief,
    const double drive_fatigue_decay,
    const double drive_fear_gain,
    const double drive_fear_decay
) {
    check_cuda_contiguous(slot_to_agent_id, "slot_to_agent_id");
    TORCH_CHECK(slot_to_agent_id.scalar_type() == torch::kInt64, "slot_to_agent_id must be int64");
    TORCH_CHECK(slot_to_agent_id.dim() == 1, "slot_to_agent_id must be rank 1");
    const int64_t capacity = slot_to_agent_id.size(0);
    TORCH_CHECK(capacity > 0, "mechanics slot capacity must be positive");
    const auto device = slot_to_agent_id.device();

    check_slot_tensor(alive_mask, torch::kBool, capacity, "alive_mask", device);
    check_slot_tensor(actions_by_slot, torch::kInt32, capacity, "actions_by_slot", device);
    check_slot_tensor(energy, torch::kFloat32, capacity, "energy", device);
    check_slot_tensor(health, torch::kFloat32, capacity, "health", device);
    check_slot_tensor(injury, torch::kFloat32, capacity, "injury", device);
    check_slot_tensor(age, torch::kInt32, capacity, "age", device);
    check_slot_tensor(
        policy_energy_observation,
        torch::kFloat64,
        capacity,
        "policy_energy_observation",
        device
    );
    check_slot_tensor(rest_streak, torch::kInt32, capacity, "rest_streak", device);
    check_slot_tensor(survival_steps, torch::kInt32, capacity, "survival_steps", device);
    check_slot_tensor(gru_upkeep_cost, torch::kFloat32, capacity, "gru_upkeep_cost", device);
    check_slot_tensor(cost_charge_scalar, torch::kFloat32, capacity, "cost_charge_scalar", device);
    check_slot_tensor(
        cost_charge_maturation_scalar,
        torch::kFloat32,
        capacity,
        "cost_charge_maturation_scalar",
        device
    );
    check_slot_tensor(drive_inertia, torch::kFloat32, capacity, "drive_inertia", device);
    check_slot_tensor(drive_noise, torch::kFloat32, capacity, "drive_noise", device);
    check_slot_tensor(drive_volatility, torch::kFloat32, capacity, "drive_volatility", device);
    check_slot_tensor(drive_jitter, torch::kFloat32, capacity, "drive_jitter", device);
    check_slot_tensor(
        episode0_drive_jitter_sum,
        torch::kFloat32,
        capacity,
        "episode0_drive_jitter_sum",
        device
    );
    check_slot_tensor(last_energy_delta, torch::kFloat32, capacity, "last_energy_delta", device);
    check_slot_tensor(last_health_delta, torch::kFloat32, capacity, "last_health_delta", device);
    check_slot_tensor(last_injury_delta, torch::kFloat32, capacity, "last_injury_delta", device);
    check_slot_tensor(selfharm_events, torch::kInt32, capacity, "selfharm_events", device);
    check_slot_tensor(
        episode0_selfharm_events,
        torch::kInt32,
        capacity,
        "episode0_selfharm_events",
        device
    );
    check_slot_tensor(pre_energy, torch::kFloat32, capacity, "pre_energy", device);
    check_slot_tensor(pre_health, torch::kFloat32, capacity, "pre_health", device);
    check_slot_tensor(pre_injury, torch::kFloat32, capacity, "pre_injury", device);
    check_slot_tensor(
        drive_noise_sample_valid,
        torch::kBool,
        capacity,
        "drive_noise_sample_valid",
        device
    );

    for (const auto& named : std::vector<std::pair<const char*, torch::Tensor>>{
             {"drives", drives},
             {"prev_drive_delta", prev_drive_delta},
             {"drive_noise_samples", drive_noise_samples},
         }) {
        check_cuda_contiguous(named.second, named.first);
        TORCH_CHECK(named.second.device() == device, named.first, " must share the mechanics CUDA device");
        TORCH_CHECK(named.second.scalar_type() == torch::kFloat32, named.first, " must be float32");
        TORCH_CHECK(
            named.second.dim() == 2
                && named.second.size(0) == capacity
                && named.second.size(1) == 4,
            named.first,
            " must be [capacity,4]"
        );
    }

    const std::vector<std::pair<const char*, torch::Tensor>> mutable_tensors = {
        {"energy", energy},
        {"health", health},
        {"policy_energy_observation", policy_energy_observation},
        {"rest_streak", rest_streak},
        {"survival_steps", survival_steps},
        {"drives", drives},
        {"prev_drive_delta", prev_drive_delta},
        {"drive_volatility", drive_volatility},
        {"drive_jitter", drive_jitter},
        {"episode0_drive_jitter_sum", episode0_drive_jitter_sum},
        {"last_energy_delta", last_energy_delta},
        {"last_health_delta", last_health_delta},
        {"last_injury_delta", last_injury_delta},
        {"selfharm_events", selfharm_events},
        {"episode0_selfharm_events", episode0_selfharm_events},
    };
    for (size_t left = 0; left < mutable_tensors.size(); ++left) {
        for (size_t right = left + 1; right < mutable_tensors.size(); ++right) {
            TORCH_CHECK(
                mutable_tensors[left].second.data_ptr()
                    != mutable_tensors[right].second.data_ptr(),
                mutable_tensors[left].first,
                " and ",
                mutable_tensors[right].first,
                " must not alias"
            );
        }
    }
    for (const auto& immutable : std::vector<std::pair<const char*, torch::Tensor>>{
             {"injury", injury},
             {"age", age},
             {"pre_energy", pre_energy},
             {"pre_health", pre_health},
             {"pre_injury", pre_injury},
             {"drive_noise_samples", drive_noise_samples},
         }) {
        for (const auto& mutable_tensor : mutable_tensors) {
            TORCH_CHECK(
                immutable.second.data_ptr() != mutable_tensor.second.data_ptr(),
                immutable.first,
                " must not alias mutable tensor ",
                mutable_tensor.first
            );
        }
    }

    TORCH_CHECK(
        rest_streak_free >= 0 && rest_streak_free <= std::numeric_limits<int32_t>::max(),
        "rest_streak_free must fit nonnegative int32"
    );
    TORCH_CHECK(
        developmental_maturity_age >= 0
            && developmental_maturity_age <= std::numeric_limits<int32_t>::max(),
        "developmental_maturity_age must fit nonnegative int32"
    );
    TORCH_CHECK(
        episode0_k >= 0 && episode0_k <= std::numeric_limits<int32_t>::max(),
        "episode0_k must fit nonnegative int32"
    );
    const double max_energy_value = checked_finite(max_energy, "max_energy");
    const double start_health_value = checked_finite(start_health, "start_health");
    TORCH_CHECK(max_energy_value > 0.0, "max_energy must be positive");
    TORCH_CHECK(start_health_value > 0.0, "start_health must be positive");
    const double rest_charge_value = checked_finite(rest_charge_energy, "rest_charge_energy");
    const double rest_growth_value = checked_finite(rest_charge_growth, "rest_charge_growth");
    const double deficit_value = checked_finite(rest_deficit_to_health, "rest_deficit_to_health");
    TORCH_CHECK(rest_charge_value >= 0.0, "rest_charge_energy must be nonnegative");
    TORCH_CHECK(rest_growth_value >= 0.0, "rest_charge_growth must be nonnegative");
    TORCH_CHECK(deficit_value >= 0.0, "rest_deficit_to_health must be nonnegative");

    const std::vector<std::pair<const char*, double>> drive_controls = {
        {"drive_hunger_gain", drive_hunger_gain},
        {"drive_hunger_decay", drive_hunger_decay},
        {"drive_pain_gain", drive_pain_gain},
        {"drive_pain_decay", drive_pain_decay},
        {"drive_fatigue_gain", drive_fatigue_gain},
        {"drive_fatigue_rest_relief", drive_fatigue_rest_relief},
        {"drive_fatigue_decay", drive_fatigue_decay},
        {"drive_fear_gain", drive_fear_gain},
        {"drive_fear_decay", drive_fear_decay},
    };
    for (const auto& named : drive_controls) {
        TORCH_CHECK(
            checked_finite(named.second, named.first) >= 0.0,
            named.first,
            " must be nonnegative"
        );
    }

    c10::cuda::CUDAGuard device_guard(device);
    auto status = torch::zeros({1}, energy.options().dtype(torch::kInt32));
    auto counts = torch::zeros({COUNT_FIELDS}, energy.options().dtype(torch::kInt64));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    active_mechanics_batched_kernel<<<1, 1, 0, stream>>>(
        slot_to_agent_id.data_ptr<int64_t>(),
        alive_mask.data_ptr<bool>(),
        actions_by_slot.data_ptr<int32_t>(),
        energy.data_ptr<float>(),
        health.data_ptr<float>(),
        injury.data_ptr<float>(),
        age.data_ptr<int32_t>(),
        policy_energy_observation.data_ptr<double>(),
        rest_streak.data_ptr<int32_t>(),
        survival_steps.data_ptr<int32_t>(),
        gru_upkeep_cost.data_ptr<float>(),
        cost_charge_scalar.data_ptr<float>(),
        cost_charge_maturation_scalar.data_ptr<float>(),
        drives.data_ptr<float>(),
        drive_inertia.data_ptr<float>(),
        drive_noise.data_ptr<float>(),
        prev_drive_delta.data_ptr<float>(),
        drive_volatility.data_ptr<float>(),
        drive_jitter.data_ptr<float>(),
        episode0_drive_jitter_sum.data_ptr<float>(),
        last_energy_delta.data_ptr<float>(),
        last_health_delta.data_ptr<float>(),
        last_injury_delta.data_ptr<float>(),
        selfharm_events.data_ptr<int32_t>(),
        episode0_selfharm_events.data_ptr<int32_t>(),
        pre_energy.data_ptr<float>(),
        pre_health.data_ptr<float>(),
        pre_injury.data_ptr<float>(),
        drive_noise_samples.data_ptr<float>(),
        drive_noise_sample_valid.data_ptr<bool>(),
        status.data_ptr<int32_t>(),
        counts.data_ptr<int64_t>(),
        capacity,
        1,
        enable_rest_charge,
        static_cast<int32_t>(rest_streak_free),
        static_cast<float>(rest_charge_value),
        static_cast<float>(rest_growth_value),
        static_cast<float>(deficit_value),
        static_cast<int32_t>(developmental_maturity_age),
        static_cast<int32_t>(episode0_k),
        static_cast<float>(max_energy_value),
        static_cast<float>(start_health_value),
        static_cast<float>(drive_hunger_gain),
        static_cast<float>(drive_hunger_decay),
        static_cast<float>(drive_pain_gain),
        static_cast<float>(drive_pain_decay),
        static_cast<float>(drive_fatigue_gain),
        static_cast<float>(drive_fatigue_rest_relief),
        static_cast<float>(drive_fatigue_decay),
        static_cast<float>(drive_fear_gain),
        static_cast<float>(drive_fear_decay)
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {status, counts};
}

std::vector<torch::Tensor> eiel_cuda_architecture_b_active_mechanics_batched_counted_v1(
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
    const bool enable_rest_charge,
    const int64_t rest_streak_free,
    const double rest_charge_energy,
    const double rest_charge_growth,
    const double rest_deficit_to_health,
    const int64_t developmental_maturity_age,
    const int64_t episode0_k,
    const double max_energy,
    const double start_health,
    const double drive_hunger_gain,
    const double drive_hunger_decay,
    const double drive_pain_gain,
    const double drive_pain_decay,
    const double drive_fatigue_gain,
    const double drive_fatigue_rest_relief,
    const double drive_fatigue_decay,
    const double drive_fear_gain,
    const double drive_fear_decay
) {
    check_cuda_contiguous(slot_to_agent_id, "slot_to_agent_id");
    TORCH_CHECK(
        slot_to_agent_id.scalar_type() == torch::kInt64,
        "slot_to_agent_id must be int64"
    );
    TORCH_CHECK(
        slot_to_agent_id.dim() == 2,
        "batched slot_to_agent_id must be [cells,capacity]"
    );
    const int64_t cell_count = slot_to_agent_id.size(0);
    const int64_t capacity = slot_to_agent_id.size(1);
    TORCH_CHECK(cell_count > 0, "batched mechanics requires at least one cell");
    TORCH_CHECK(capacity > 0, "mechanics slot capacity must be positive");
    TORCH_CHECK(
        cell_count <= std::numeric_limits<int32_t>::max(),
        "batched mechanics cell count exceeds the CUDA grid limit"
    );
    const auto device = slot_to_agent_id.device();

    check_batched_slot_tensor(
        alive_mask, torch::kBool, cell_count, capacity, "alive_mask", device
    );
    check_batched_slot_tensor(
        actions_by_slot,
        torch::kInt32,
        cell_count,
        capacity,
        "actions_by_slot",
        device
    );
    check_batched_slot_tensor(
        energy, torch::kFloat32, cell_count, capacity, "energy", device
    );
    check_batched_slot_tensor(
        health, torch::kFloat32, cell_count, capacity, "health", device
    );
    check_batched_slot_tensor(
        injury, torch::kFloat32, cell_count, capacity, "injury", device
    );
    check_batched_slot_tensor(
        age, torch::kInt32, cell_count, capacity, "age", device
    );
    check_batched_slot_tensor(
        policy_energy_observation,
        torch::kFloat64,
        cell_count,
        capacity,
        "policy_energy_observation",
        device
    );
    check_batched_slot_tensor(
        rest_streak, torch::kInt32, cell_count, capacity, "rest_streak", device
    );
    check_batched_slot_tensor(
        survival_steps,
        torch::kInt32,
        cell_count,
        capacity,
        "survival_steps",
        device
    );
    check_batched_slot_tensor(
        gru_upkeep_cost,
        torch::kFloat32,
        cell_count,
        capacity,
        "gru_upkeep_cost",
        device
    );
    check_batched_slot_tensor(
        cost_charge_scalar,
        torch::kFloat32,
        cell_count,
        capacity,
        "cost_charge_scalar",
        device
    );
    check_batched_slot_tensor(
        cost_charge_maturation_scalar,
        torch::kFloat32,
        cell_count,
        capacity,
        "cost_charge_maturation_scalar",
        device
    );
    check_batched_slot_tensor(
        drive_inertia,
        torch::kFloat32,
        cell_count,
        capacity,
        "drive_inertia",
        device
    );
    check_batched_slot_tensor(
        drive_noise,
        torch::kFloat32,
        cell_count,
        capacity,
        "drive_noise",
        device
    );
    check_batched_slot_tensor(
        drive_volatility,
        torch::kFloat32,
        cell_count,
        capacity,
        "drive_volatility",
        device
    );
    check_batched_slot_tensor(
        drive_jitter,
        torch::kFloat32,
        cell_count,
        capacity,
        "drive_jitter",
        device
    );
    check_batched_slot_tensor(
        episode0_drive_jitter_sum,
        torch::kFloat32,
        cell_count,
        capacity,
        "episode0_drive_jitter_sum",
        device
    );
    check_batched_slot_tensor(
        last_energy_delta,
        torch::kFloat32,
        cell_count,
        capacity,
        "last_energy_delta",
        device
    );
    check_batched_slot_tensor(
        last_health_delta,
        torch::kFloat32,
        cell_count,
        capacity,
        "last_health_delta",
        device
    );
    check_batched_slot_tensor(
        last_injury_delta,
        torch::kFloat32,
        cell_count,
        capacity,
        "last_injury_delta",
        device
    );
    check_batched_slot_tensor(
        selfharm_events,
        torch::kInt32,
        cell_count,
        capacity,
        "selfharm_events",
        device
    );
    check_batched_slot_tensor(
        episode0_selfharm_events,
        torch::kInt32,
        cell_count,
        capacity,
        "episode0_selfharm_events",
        device
    );
    check_batched_slot_tensor(
        pre_energy,
        torch::kFloat32,
        cell_count,
        capacity,
        "pre_energy",
        device
    );
    check_batched_slot_tensor(
        pre_health,
        torch::kFloat32,
        cell_count,
        capacity,
        "pre_health",
        device
    );
    check_batched_slot_tensor(
        pre_injury,
        torch::kFloat32,
        cell_count,
        capacity,
        "pre_injury",
        device
    );
    check_batched_slot_tensor(
        drive_noise_sample_valid,
        torch::kBool,
        cell_count,
        capacity,
        "drive_noise_sample_valid",
        device
    );

    for (const auto& named : std::vector<std::pair<const char*, torch::Tensor>>{
             {"drives", drives},
             {"prev_drive_delta", prev_drive_delta},
             {"drive_noise_samples", drive_noise_samples},
         }) {
        check_cuda_contiguous(named.second, named.first);
        TORCH_CHECK(
            named.second.device() == device,
            named.first,
            " must share the mechanics CUDA device"
        );
        TORCH_CHECK(
            named.second.scalar_type() == torch::kFloat32,
            named.first,
            " must be float32"
        );
        TORCH_CHECK(
            named.second.dim() == 3
                && named.second.size(0) == cell_count
                && named.second.size(1) == capacity
                && named.second.size(2) == 4,
            named.first,
            " must be [cells,capacity,4]"
        );
    }

    const std::vector<std::pair<const char*, torch::Tensor>> mutable_tensors = {
        {"energy", energy},
        {"health", health},
        {"policy_energy_observation", policy_energy_observation},
        {"rest_streak", rest_streak},
        {"survival_steps", survival_steps},
        {"drives", drives},
        {"prev_drive_delta", prev_drive_delta},
        {"drive_volatility", drive_volatility},
        {"drive_jitter", drive_jitter},
        {"episode0_drive_jitter_sum", episode0_drive_jitter_sum},
        {"last_energy_delta", last_energy_delta},
        {"last_health_delta", last_health_delta},
        {"last_injury_delta", last_injury_delta},
        {"selfharm_events", selfharm_events},
        {"episode0_selfharm_events", episode0_selfharm_events},
    };
    for (size_t left = 0; left < mutable_tensors.size(); ++left) {
        for (size_t right = left + 1; right < mutable_tensors.size(); ++right) {
            TORCH_CHECK(
                mutable_tensors[left].second.data_ptr()
                    != mutable_tensors[right].second.data_ptr(),
                mutable_tensors[left].first,
                " and ",
                mutable_tensors[right].first,
                " must not alias"
            );
        }
    }
    for (const auto& immutable : std::vector<std::pair<const char*, torch::Tensor>>{
             {"injury", injury},
             {"age", age},
             {"pre_energy", pre_energy},
             {"pre_health", pre_health},
             {"pre_injury", pre_injury},
             {"drive_noise_samples", drive_noise_samples},
         }) {
        for (const auto& mutable_tensor : mutable_tensors) {
            TORCH_CHECK(
                immutable.second.data_ptr() != mutable_tensor.second.data_ptr(),
                immutable.first,
                " must not alias mutable tensor ",
                mutable_tensor.first
            );
        }
    }

    TORCH_CHECK(
        rest_streak_free >= 0 && rest_streak_free <= std::numeric_limits<int32_t>::max(),
        "rest_streak_free must fit nonnegative int32"
    );
    TORCH_CHECK(
        developmental_maturity_age >= 0
            && developmental_maturity_age <= std::numeric_limits<int32_t>::max(),
        "developmental_maturity_age must fit nonnegative int32"
    );
    TORCH_CHECK(
        episode0_k >= 0 && episode0_k <= std::numeric_limits<int32_t>::max(),
        "episode0_k must fit nonnegative int32"
    );
    const double max_energy_value = checked_finite(max_energy, "max_energy");
    const double start_health_value = checked_finite(start_health, "start_health");
    TORCH_CHECK(max_energy_value > 0.0, "max_energy must be positive");
    TORCH_CHECK(start_health_value > 0.0, "start_health must be positive");
    const double rest_charge_value = checked_finite(rest_charge_energy, "rest_charge_energy");
    const double rest_growth_value = checked_finite(rest_charge_growth, "rest_charge_growth");
    const double deficit_value = checked_finite(rest_deficit_to_health, "rest_deficit_to_health");
    TORCH_CHECK(rest_charge_value >= 0.0, "rest_charge_energy must be nonnegative");
    TORCH_CHECK(rest_growth_value >= 0.0, "rest_charge_growth must be nonnegative");
    TORCH_CHECK(deficit_value >= 0.0, "rest_deficit_to_health must be nonnegative");

    const std::vector<std::pair<const char*, double>> drive_controls = {
        {"drive_hunger_gain", drive_hunger_gain},
        {"drive_hunger_decay", drive_hunger_decay},
        {"drive_pain_gain", drive_pain_gain},
        {"drive_pain_decay", drive_pain_decay},
        {"drive_fatigue_gain", drive_fatigue_gain},
        {"drive_fatigue_rest_relief", drive_fatigue_rest_relief},
        {"drive_fatigue_decay", drive_fatigue_decay},
        {"drive_fear_gain", drive_fear_gain},
        {"drive_fear_decay", drive_fear_decay},
    };
    for (const auto& named : drive_controls) {
        TORCH_CHECK(
            checked_finite(named.second, named.first) >= 0.0,
            named.first,
            " must be nonnegative"
        );
    }

    c10::cuda::CUDAGuard device_guard(device);
    auto status = torch::zeros({cell_count}, energy.options().dtype(torch::kInt32));
    auto counts = torch::zeros(
        {cell_count, COUNT_FIELDS}, energy.options().dtype(torch::kInt64)
    );
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    active_mechanics_batched_kernel<<<
        static_cast<unsigned int>(cell_count), 1, 0, stream
    >>>(
        slot_to_agent_id.data_ptr<int64_t>(),
        alive_mask.data_ptr<bool>(),
        actions_by_slot.data_ptr<int32_t>(),
        energy.data_ptr<float>(),
        health.data_ptr<float>(),
        injury.data_ptr<float>(),
        age.data_ptr<int32_t>(),
        policy_energy_observation.data_ptr<double>(),
        rest_streak.data_ptr<int32_t>(),
        survival_steps.data_ptr<int32_t>(),
        gru_upkeep_cost.data_ptr<float>(),
        cost_charge_scalar.data_ptr<float>(),
        cost_charge_maturation_scalar.data_ptr<float>(),
        drives.data_ptr<float>(),
        drive_inertia.data_ptr<float>(),
        drive_noise.data_ptr<float>(),
        prev_drive_delta.data_ptr<float>(),
        drive_volatility.data_ptr<float>(),
        drive_jitter.data_ptr<float>(),
        episode0_drive_jitter_sum.data_ptr<float>(),
        last_energy_delta.data_ptr<float>(),
        last_health_delta.data_ptr<float>(),
        last_injury_delta.data_ptr<float>(),
        selfharm_events.data_ptr<int32_t>(),
        episode0_selfharm_events.data_ptr<int32_t>(),
        pre_energy.data_ptr<float>(),
        pre_health.data_ptr<float>(),
        pre_injury.data_ptr<float>(),
        drive_noise_samples.data_ptr<float>(),
        drive_noise_sample_valid.data_ptr<bool>(),
        status.data_ptr<int32_t>(),
        counts.data_ptr<int64_t>(),
        capacity,
        cell_count,
        enable_rest_charge,
        static_cast<int32_t>(rest_streak_free),
        static_cast<float>(rest_charge_value),
        static_cast<float>(rest_growth_value),
        static_cast<float>(deficit_value),
        static_cast<int32_t>(developmental_maturity_age),
        static_cast<int32_t>(episode0_k),
        static_cast<float>(max_energy_value),
        static_cast<float>(start_health_value),
        static_cast<float>(drive_hunger_gain),
        static_cast<float>(drive_hunger_decay),
        static_cast<float>(drive_pain_gain),
        static_cast<float>(drive_pain_decay),
        static_cast<float>(drive_fatigue_gain),
        static_cast<float>(drive_fatigue_rest_relief),
        static_cast<float>(drive_fatigue_decay),
        static_cast<float>(drive_fear_gain),
        static_cast<float>(drive_fear_decay)
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {status, counts};
}
