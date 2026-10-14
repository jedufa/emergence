#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <type_traits>
#include <vector>

namespace {

constexpr int64_t kObservationWidth = 83;
constexpr int64_t kHiddenWidth = 64;
constexpr int64_t kActions = 7;
constexpr int64_t kDriveWidth = 4;
constexpr int64_t kPolicyPackedWidth = 9991;
constexpr int kThreadsPerRow = 64;

// Drop-in indices for c2_backend._native_state_v1.  The focused source test
// keeps this list synchronized with tensor_world_c2_kernels.cu.
enum T : int64_t {
    ALIVE, AGENT_ID, POS_Y, POS_X, ENERGY, HEALTH, AGENT_AGE,
    SURVIVAL, ENERGY_COLLECTED, ID_GRID_I64, SLOT_GRID_I64, ACTIONS,
    MOVED, GATHERED, FOOD_GAINED, MOVES_ATTEMPTED, MOVES_COMMITTED,
    GATHERS_COMMITTED, ACTION_UNIFORMS, ACTION_CURSOR, COVER, SHELTER,
    MOVEMENT, FOOD, SCENT, WASTE, DENSITY, FERTILITY, SCENT_PENDING,
    WASTE_PENDING, RECOVERY, VISIBILITY, AGENT_GRID_I32, FOOD_GRID_I32,
    MULTIPLIER_SCRATCH, WASTE_PENALTY_SCRATCH, DENSITY_PENALTY_SCRATCH,
    SORTED_MULTIPLIER_SCRATCH, RECOVERY_BOOST_SCRATCH,
    SORTED_RECOVERY_BOOST_SCRATCH, WDL_CUE, WDL_TIMER, WDL_ACTIVE,
    WDL_INTENSITY, WDL_AGE, WDL_EVENT_ID, WDL_PRIOR_CUE, WDL_BASE_SEED,
    WDL_NEXT_STEP, EED_A_AMOUNT, EED_B_AMOUNT, EED_A_CAPACITY,
    EED_B_CAPACITY, EED_A_RENEWAL, EED_B_RENEWAL, EED_A_PENDING,
    EED_B_PENDING, EED_A_LOAD, EED_B_LOAD, EED_CLASS, EED_ACTIVE_Y,
    EED_ACTIVE_X, EED_ACTIVE_CLASS, EED_PROGRESS, EED_AXIS, C2_PACK,
    JUNGLE_UNIFORMS, EFFECT_PRE, EFFECT_POST, HAZARD_FLAGS, EVENT_SCENT,
    EVENT_WASTE, EVENT_GATHER_AMOUNT, EVENT_GATHER_CLASS, OBSERVATIONS,
    LOGITS, PATCH_STATS, RECOVERY_STATS, JUNGLE_INTEGERS, JUNGLE_DOUBLES,
    WDL_INTEGERS, WDL_DELAY_SUM, WDL_ACTION_INTEGERS, EED_RECEIPT,
    DRIVES, ATTACK_BIAS, ACTION_TEMPERATURE, DRIVE_LOGIT_GAIN,
    TENSOR_COUNT
};

enum CF : int64_t {
    F_MAX_ENERGY, F_START_HEALTH, F_SCENT_DECAY, F_WASTE_DECAY,
    F_SCENT_PRESENCE, F_WASTE_PRESENCE, F_WASTE_DENSITY_EMIT,
    F_SCENT_MAX, F_WASTE_MAX, F_FOOD_MAX, F_FOOD_REGROW,
    F_FOOD_THRESHOLD, F_WASTE_PENALTY_SCALE, F_WASTE_PENALTY_MAX,
    F_DENSITY_PENALTY_SCALE, F_DENSITY_PENALTY_MAX, F_REGROWTH_MIN,
    F_RECOVERY_MIN, F_RECOVERY_MAX, F_FERTILITY_MIN, F_FERTILITY_MAX,
    F_RECOVERY_DECAY, F_RECOVERY_BASE, F_RECOVERY_COVER,
    F_RECOVERY_FERTILITY, F_RECOVERY_WASTE, F_RECOVERY_DENSITY,
    F_RECOVERY_SCENT, F_RECOVERY_BOOST_SCALE, F_RECOVERY_BOOST_MAX,
    F_JUNGLE_BASE_RISK, F_JUNGLE_SCENT, F_JUNGLE_WASTE,
    F_JUNGLE_DENSITY, F_JUNGLE_VISIBILITY, F_JUNGLE_COVER,
    F_JUNGLE_SHELTER, F_JUNGLE_RISK_MIN, F_JUNGLE_RISK_CAP,
    F_JUNGLE_DAMAGE, F_JUNGLE_JUVENILE_AGE, F_JUNGLE_JUVENILE_MULT,
    F_JUNGLE_COVER_DAMAGE, F_JUNGLE_SHELTER_DAMAGE, F_JUNGLE_TERRAIN_MIN,
    F_WDL_INTENSITY, F_WDL_DROPOUT, F_EED_DECAY, F_EED_MAX_LOAD,
    F_EED_IMPACT_PER_SUCCESS, F_EED_GATHER, F_EED_HANDLE_A,
    F_EED_HANDLE_B, F_EED_RETURN_A, F_EED_RETURN_B,
    F_MOVE_COST, F_GATHER_COST, F_REST_COST, F_ATTACK_COST,
    F_ACTION_MULT, F_REST_MULT, F_MOVEMENT_SCALE,
    F_SCENT_MOVE_EMIT, F_WASTE_MOVE_EMIT, F_SCENT_GATHER_EMIT,
    F_WASTE_GATHER_EMIT, F_W_HUNGER_GATHER, F_W_PAIN_REST,
    F_W_FATIGUE_REST, F_W_PAIN_ATTACK, F_W_FEAR_ATTACK, F_CONFIG_COUNT
};

template <typename P>
P ptr(const std::vector<torch::Tensor>& tensors, T index) {
    using Value = std::remove_cv_t<std::remove_pointer_t<P>>;
    return tensors[index].data_ptr<Value>();
}

// EIEL_C2_POLICY_PARALLEL_R12_KERNEL_BEGIN
__global__ void policy_parallel_kernel(
    float* logits,
    const float* pack,
    const float* observations,
    const float* drives,
    const float* attack_bias,
    const float* action_temperature,
    const float* drive_logit_gain,
    const bool* alive,
    const double* config_floats,
    int64_t rows
) {
    const int64_t row = blockIdx.x;
    const int64_t lane = threadIdx.x;
    if (row >= rows || lane >= kThreadsPerRow) {
        return;
    }
    float* const row_logits = logits + row * kActions;
    if (!alive[row]) {
        if (lane < kActions) {
            row_logits[lane] = 0.0f;
        }
        return;
    }

    __shared__ float h1[kHiddenWidth];
    __shared__ float h2[kHiddenWidth];
    const float* const row_pack = pack + row * kPolicyPackedWidth;
    const float* const row_observation = observations + row * kObservationWidth;
    const float* const row_drives = drives + row * kDriveWidth;

    // One thread owns one neuron.  The feature loop is deliberately serial and
    // ascending, exactly matching policy_exact in tensor_world_c2_kernels.cu.
    float value = row_pack[kHiddenWidth * kObservationWidth + lane];
    const float* weights = row_pack + lane * kObservationWidth;
    for (int64_t feature = 0; feature < kObservationWidth; ++feature) {
        value += weights[feature] * row_observation[feature];
    }
    h1[lane] = tanhf(value);
    __syncthreads();

    constexpr int64_t second_layer =
        kHiddenWidth * kObservationWidth + kHiddenWidth;
    value = row_pack[
        second_layer + kHiddenWidth * kHiddenWidth + lane
    ];
    weights = row_pack + second_layer + lane * kHiddenWidth;
    for (int64_t input = 0; input < kHiddenWidth; ++input) {
        value += weights[input] * h1[input];
    }
    h2[lane] = tanhf(value);
    __syncthreads();

    if (lane >= kActions) {
        return;
    }
    constexpr int64_t output_layer =
        second_layer + kHiddenWidth * kHiddenWidth + kHiddenWidth;
    weights = row_pack + output_layer + lane * kHiddenWidth;
    float score = row_pack[output_layer + kActions * kHiddenWidth + lane];
    for (int64_t hidden = 0; hidden < kHiddenWidth; ++hidden) {
        score += weights[hidden] * h2[hidden];
    }
    const float temperature = fmaxf(0.05f, action_temperature[row]);
    const float gain = fmaxf(0.0f, drive_logit_gain[row]);
    if (lane == 6) {
        score += attack_bias[row];
    }
    score /= temperature;
    if (lane == 4) {
        score += gain * config_floats[F_W_HUNGER_GATHER] * row_drives[0];
    } else if (lane == 5) {
        score += gain * (
            config_floats[F_W_PAIN_REST] * row_drives[1]
            + config_floats[F_W_FATIGUE_REST] * row_drives[2]
        );
    } else if (lane == 6) {
        score -= gain * (
            config_floats[F_W_PAIN_ATTACK] * row_drives[1]
            + config_floats[F_W_FEAR_ATTACK] * row_drives[3]
        );
    }
    row_logits[lane] = score;
}
// EIEL_C2_POLICY_PARALLEL_R12_KERNEL_END

void validate_policy_call(
    const std::vector<torch::Tensor>& tensors,
    const torch::Tensor& config_ints,
    const torch::Tensor& config_floats
) {
    TORCH_CHECK(
        tensors.size() == TENSOR_COUNT,
        "C2 parallel policy state tensor ABI mismatch"
    );
    TORCH_CHECK(
        config_ints.is_cuda() && config_ints.is_contiguous()
            && config_ints.scalar_type() == torch::kInt64
            && config_ints.numel() == 23,
        "C2 parallel policy integer config ABI mismatch"
    );
    TORCH_CHECK(
        config_floats.is_cuda() && config_floats.is_contiguous()
            && config_floats.scalar_type() == torch::kFloat64
            && config_floats.numel() == F_CONFIG_COUNT,
        "C2 parallel policy floating config ABI mismatch"
    );
    const auto& alive = tensors[ALIVE];
    TORCH_CHECK(
        alive.is_cuda() && alive.is_contiguous()
            && alive.scalar_type() == torch::kBool && alive.dim() == 2
            && alive.size(0) > 0 && alive.size(1) > 0,
        "C2 parallel policy alive ABI mismatch"
    );
    const int64_t batch = alive.size(0);
    const int64_t capacity = alive.size(1);
    const auto device = alive.device();
    const auto validate = [&](T index, torch::ScalarType dtype,
                              std::vector<int64_t> shape) {
        const auto& tensor = tensors[index];
        TORCH_CHECK(
            tensor.is_cuda() && tensor.device() == device
                && tensor.is_contiguous() && tensor.scalar_type() == dtype
                && tensor.sizes().vec() == shape,
            "C2 parallel policy tensor ABI mismatch at index ",
            static_cast<int64_t>(index)
        );
    };
    validate(C2_PACK, torch::kFloat32, {batch, capacity, kPolicyPackedWidth});
    validate(OBSERVATIONS, torch::kFloat32, {batch, capacity, kObservationWidth});
    validate(LOGITS, torch::kFloat32, {batch, capacity, kActions});
    validate(DRIVES, torch::kFloat32, {batch, capacity, kDriveWidth});
    validate(ATTACK_BIAS, torch::kFloat32, {batch, capacity});
    validate(ACTION_TEMPERATURE, torch::kFloat32, {batch, capacity});
    validate(DRIVE_LOGIT_GAIN, torch::kFloat32, {batch, capacity});
    TORCH_CHECK(
        config_ints.device() == device && config_floats.device() == device,
        "C2 parallel policy config/device mismatch"
    );
}

void c2_policy_parallel(
    std::vector<torch::Tensor> tensors,
    torch::Tensor config_ints,
    torch::Tensor config_floats
) {
    validate_policy_call(tensors, config_ints, config_floats);
    const int64_t rows = tensors[ALIVE].numel();
    policy_parallel_kernel<<<
        rows, kThreadsPerRow, 0, at::cuda::getCurrentCUDAStream()
    >>>(
        ptr<float*>(tensors, LOGITS),
        ptr<const float*>(tensors, C2_PACK),
        ptr<const float*>(tensors, OBSERVATIONS),
        ptr<const float*>(tensors, DRIVES),
        ptr<const float*>(tensors, ATTACK_BIAS),
        ptr<const float*>(tensors, ACTION_TEMPERATURE),
        ptr<const float*>(tensors, DRIVE_LOGIT_GAIN),
        ptr<const bool*>(tensors, ALIVE),
        config_floats.data_ptr<double>(),
        rows
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("policy_logits", &c2_policy_parallel);
}
