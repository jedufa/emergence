#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <vector>

namespace {

constexpr std::int64_t kObservationColumns = 12;
constexpr std::int8_t kCuePhase = 0;
constexpr std::int8_t kDelayPhase = 1;
constexpr std::int8_t kDecisionPhase = 2;
constexpr std::int8_t kConsequencePhase = 3;

struct GmnwConfigV1 {
    std::int64_t cue_alphabet;
    std::int64_t cue_ticks;
    std::int64_t delay_ticks;
    std::int64_t consequence_ticks;
    bool distractors_enabled;
    std::int64_t task_seed;
    std::int64_t grace_generations;
    std::int64_t severe_generations;
    float grace_damage;
    float severe_damage;
    float lethal_damage;
    float grace_reproduction_scale;
};

__device__ __forceinline__ std::uint64_t splitmix64(std::uint64_t value) {
    value += 0x9E3779B97F4A7C15ULL;
    value = (value ^ (value >> 30U)) * 0xBF58476D1CE4E5B9ULL;
    value = (value ^ (value >> 27U)) * 0x94D049BB133111EBULL;
    return value ^ (value >> 31U);
}

__device__ __forceinline__ std::uint64_t episode_rng_state(
    const GmnwConfigV1 config,
    const std::int64_t stream_id,
    const std::int64_t episode_index
) {
    const std::uint64_t mixed = static_cast<std::uint64_t>(config.task_seed)
        ^ (static_cast<std::uint64_t>(stream_id + 1) * 0xD6E8FEB86659FD93ULL)
        ^ (static_cast<std::uint64_t>(episode_index + 1) * 0xA5A3564E27F8862FULL);
    return splitmix64(mixed);
}

__device__ __forceinline__ std::int8_t balanced_cue(
    const GmnwConfigV1 config,
    const std::int64_t stream_id,
    const std::int64_t episode_index
) {
    const std::int64_t block = episode_index / config.cue_alphabet;
    const std::int64_t ordinal = episode_index % config.cue_alphabet;
    std::int8_t values[4] = {0, 1, 2, 3};
    std::uint64_t state = episode_rng_state(config, stream_id, block);
    for (std::int64_t upper = config.cue_alphabet - 1; upper > 0; --upper) {
        state = splitmix64(state);
        const std::int64_t swap = static_cast<std::int64_t>(
            state % static_cast<std::uint64_t>(upper + 1)
        );
        const std::int8_t temporary = values[upper];
        values[upper] = values[swap];
        values[swap] = temporary;
    }
    return values[ordinal];
}

__device__ __forceinline__ void initialize_episode(
    const GmnwConfigV1 config,
    const std::int64_t index,
    const std::int64_t next_episode,
    const std::int64_t stream,
    std::int64_t* episode_index,
    std::int8_t* cue_identity,
    std::int8_t* cue_phase,
    std::int32_t* phase_tick,
    std::int32_t* delay_remaining,
    bool* decision_due,
    std::int8_t* correct_action,
    std::int8_t* last_response,
    std::int8_t* correctness,
    std::int8_t* consequence_status,
    std::int64_t* task_rng_state,
    std::int64_t* task_rng_cursor
) {
    constexpr std::int8_t actions[4] = {0, 2, 1, 3};
    const std::int8_t cue = balanced_cue(config, stream, next_episode);
    episode_index[index] = next_episode;
    cue_identity[index] = cue;
    cue_phase[index] = kCuePhase;
    phase_tick[index] = 0;
    delay_remaining[index] = 0;
    decision_due[index] = false;
    correct_action[index] = actions[cue];
    last_response[index] = -1;
    correctness[index] = -1;
    consequence_status[index] = 0;
    task_rng_state[index] = static_cast<std::int64_t>(
        episode_rng_state(config, stream, next_episode)
    );
    task_rng_cursor[index] = next_episode + 1;
}

__device__ __forceinline__ void clear_slot(
    const std::int64_t index,
    bool* alive,
    std::int64_t* stream_id,
    std::int64_t* episode_index,
    std::int8_t* cue_identity,
    std::int8_t* cue_phase,
    std::int32_t* phase_tick,
    std::int32_t* delay_remaining,
    bool* decision_due,
    std::int8_t* correct_action,
    std::int8_t* last_response,
    std::int8_t* correctness,
    std::int8_t* consequence_status,
    std::int64_t* task_rng_state,
    std::int64_t* task_rng_cursor
) {
    alive[index] = false;
    stream_id[index] = -1;
    episode_index[index] = -1;
    cue_identity[index] = -1;
    cue_phase[index] = 0;
    phase_tick[index] = 0;
    delay_remaining[index] = 0;
    decision_due[index] = false;
    correct_action[index] = -1;
    last_response[index] = -1;
    correctness[index] = -1;
    consequence_status[index] = 0;
    task_rng_state[index] = 0;
    task_rng_cursor[index] = 0;
}

__global__ void gmnw_run_resident_v1(
    const std::int64_t* actions,
    const std::int64_t* generation_by_tick,
    const std::int64_t ticks,
    const std::int64_t cells,
    const std::int64_t capacity,
    const float start_health,
    const GmnwConfigV1 config,
    bool* alive,
    std::int64_t* stream_id,
    std::int64_t* episode_index,
    std::int8_t* cue_identity,
    std::int8_t* cue_phase,
    std::int32_t* phase_tick,
    std::int32_t* delay_remaining,
    bool* decision_due,
    std::int8_t* correct_action,
    std::int8_t* last_response,
    std::int8_t* correctness,
    std::int8_t* consequence_status,
    std::int64_t* task_rng_state,
    std::int64_t* task_rng_cursor,
    float* health,
    float* reproduction_scale,
    std::int64_t* counters,
    double* damage_total,
    float* observation_by_tick,
    std::int32_t* status
) {
    const std::int64_t cell = static_cast<std::int64_t>(blockIdx.x);
    if (threadIdx.x != 0 || cell >= cells || status[cell] != 0) return;
    const std::int64_t cell_offset = cell * capacity;
    for (std::int64_t slot = 0; slot < capacity; ++slot) {
        const std::int64_t index = cell_offset + slot;
        if (
            !isfinite(health[index]) || health[index] < 0.0f
            || !isfinite(reproduction_scale[index])
            || reproduction_scale[index] < 0.0f
            || reproduction_scale[index] > 1.0f
            || (alive[index] && (
                stream_id[index] < 0 || episode_index[index] < 0
                || cue_identity[index] < 0
                || cue_identity[index] >= config.cue_alphabet
                || cue_phase[index] < kCuePhase
                || cue_phase[index] > kConsequencePhase
            ))
        ) {
            status[cell] = 1;
            return;
        }
    }

    std::int64_t* const cell_counters = counters + cell * 6;
    for (std::int64_t tick = 0; tick < ticks; ++tick) {
        const std::int64_t generation = generation_by_tick[tick];
        if (generation < 0) {
            status[cell] = 2;
            return;
        }
        for (std::int64_t slot = 0; slot < capacity; ++slot) {
            const std::int64_t index = cell_offset + slot;
            float* const observation = observation_by_tick
                + ((tick * cells + cell) * capacity + slot) * kObservationColumns;
            for (std::int64_t column = 0; column < kObservationColumns; ++column) {
                observation[column] = 0.0f;
            }
            if (!alive[index]) continue;
            const std::int8_t phase = cue_phase[index];
            if (phase == kCuePhase) observation[cue_identity[index]] = 1.0f;
            observation[4 + phase] = 1.0f;
            if (phase == kDelayPhase && config.distractors_enabled) {
                const std::int64_t elapsed = config.delay_ticks - delay_remaining[index];
                const std::uint64_t distractor = splitmix64(
                    static_cast<std::uint64_t>(task_rng_state[index])
                    ^ (static_cast<std::uint64_t>(elapsed + 1) * 0x8CB92BA72F3D8DD7ULL)
                    ^ 0x47524D4E575F4453ULL
                );
                observation[8 + static_cast<std::int64_t>(distractor % 4ULL)] = 1.0f;
            }

            if (phase == kCuePhase) {
                phase_tick[index] += 1;
                if (phase_tick[index] >= config.cue_ticks) {
                    cue_phase[index] = kDelayPhase;
                    phase_tick[index] = 0;
                    delay_remaining[index] = static_cast<std::int32_t>(
                        config.delay_ticks
                    );
                }
                continue;
            }
            if (phase == kDelayPhase) {
                delay_remaining[index] -= 1;
                if (delay_remaining[index] > 0) {
                    phase_tick[index] += 1;
                } else {
                    cue_phase[index] = kDecisionPhase;
                    phase_tick[index] = 0;
                    delay_remaining[index] = 0;
                    decision_due[index] = true;
                }
                continue;
            }
            if (phase == kDecisionPhase) {
                const std::int64_t action = actions[(tick * cells + cell) * capacity + slot];
                const bool correct = action == correct_action[index];
                cue_phase[index] = kConsequencePhase;
                phase_tick[index] = 0;
                decision_due[index] = false;
                last_response[index] = static_cast<std::int8_t>(action);
                correctness[index] = correct ? 1 : 0;
                consequence_status[index] = correct ? 1 : 2;
                cell_counters[0] += 1;
                cell_counters[correct ? 1 : 2] += 1;
                if (correct) continue;

                float damage = config.lethal_damage * start_health;
                float scale = 0.0f;
                if (generation < config.grace_generations) {
                    damage = config.grace_damage * start_health;
                    scale = config.grace_reproduction_scale;
                    cell_counters[3] += 1;
                } else if (
                    generation < config.grace_generations + config.severe_generations
                ) {
                    damage = config.severe_damage * start_health;
                    cell_counters[4] += 1;
                } else {
                    cell_counters[4] += 1;
                }
                health[index] -= damage;
                reproduction_scale[index] = fminf(reproduction_scale[index], scale);
                damage_total[cell] += static_cast<double>(damage);
                if (health[index] <= 0.0f) {
                    health[index] = 0.0f;
                    reproduction_scale[index] = 0.0f;
                    cell_counters[5] += 1;
                    clear_slot(
                        index, alive, stream_id, episode_index, cue_identity,
                        cue_phase, phase_tick, delay_remaining, decision_due,
                        correct_action, last_response, correctness,
                        consequence_status, task_rng_state, task_rng_cursor
                    );
                }
                continue;
            }
            phase_tick[index] += 1;
            if (phase_tick[index] >= config.consequence_ticks) {
                reproduction_scale[index] = 1.0f;
                initialize_episode(
                    config, index, episode_index[index] + 1, stream_id[index],
                    episode_index, cue_identity, cue_phase, phase_tick,
                    delay_remaining, decision_due, correct_action, last_response,
                    correctness, consequence_status, task_rng_state,
                    task_rng_cursor
                );
            }
        }
    }
}

void require_cuda(const torch::Tensor& reference, const torch::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be CUDA resident");
    TORCH_CHECK(tensor.device() == reference.device(), name, " device mismatch");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(!tensor.requires_grad(), name, " must not require gradients");
}

void require_shape(
    const torch::Tensor& reference,
    const torch::Tensor& tensor,
    const torch::ScalarType dtype,
    const std::vector<std::int64_t>& shape,
    const char* name
) {
    require_cuda(reference, tensor, name);
    TORCH_CHECK(tensor.scalar_type() == dtype, name, " dtype mismatch");
    TORCH_CHECK(tensor.sizes().vec() == shape, name, " shape mismatch");
}

std::int64_t exact_int(const pybind11::dict& config, const char* name) {
    const pybind11::str key(name);
    TORCH_CHECK(config.contains(key), "GMNW CUDA config is missing ", name);
    const auto value = config[key];
    TORCH_CHECK(
        pybind11::isinstance<pybind11::int_>(value)
            && !pybind11::isinstance<pybind11::bool_>(value),
        "GMNW CUDA config field must be an exact integer: ", name
    );
    return pybind11::cast<std::int64_t>(value);
}

bool exact_bool(const pybind11::dict& config, const char* name) {
    const pybind11::str key(name);
    TORCH_CHECK(config.contains(key), "GMNW CUDA config is missing ", name);
    TORCH_CHECK(
        pybind11::isinstance<pybind11::bool_>(config[key]),
        "GMNW CUDA config field must be an exact bool: ", name
    );
    return pybind11::cast<bool>(config[key]);
}

float exact_float(const pybind11::dict& config, const char* name) {
    const pybind11::str key(name);
    TORCH_CHECK(config.contains(key), "GMNW CUDA config is missing ", name);
    TORCH_CHECK(
        pybind11::isinstance<pybind11::float_>(config[key]),
        "GMNW CUDA config field must be an exact float: ", name
    );
    return pybind11::cast<float>(config[key]);
}

}  // namespace

pybind11::dict eiel_gmnw_cuda_run_resident_v1(
    const torch::Tensor& actions,
    const torch::Tensor& generation_by_tick,
    const pybind11::dict& config,
    const pybind11::list& state,
    const double start_health,
    const bool cancel_requested
) {
    TORCH_CHECK(actions.is_cuda() && actions.is_contiguous(), "GMNW actions must be contiguous CUDA");
    TORCH_CHECK(actions.scalar_type() == torch::kInt64 && actions.dim() == 3, "GMNW actions must be int64[ticks,cells,capacity]");
    const std::int64_t ticks = actions.size(0);
    const std::int64_t cells = actions.size(1);
    const std::int64_t capacity = actions.size(2);
    TORCH_CHECK(ticks > 0 && cells > 0 && capacity > 0, "GMNW CUDA dimensions must be positive");
    TORCH_CHECK(state.size() == 19, "GMNW CUDA state tensor count mismatch");

    std::vector<torch::Tensor> tensors;
    tensors.reserve(state.size());
    for (const auto item : state) tensors.push_back(pybind11::cast<torch::Tensor>(item));
    const std::vector<std::int64_t> rows{cells, capacity};
    require_shape(actions, generation_by_tick, torch::kInt64, {ticks}, "generation_by_tick");
    require_shape(actions, tensors[0], torch::kBool, rows, "alive");
    require_shape(actions, tensors[1], torch::kInt64, rows, "stream_id");
    require_shape(actions, tensors[2], torch::kInt64, rows, "episode_index");
    require_shape(actions, tensors[3], torch::kInt8, rows, "cue_identity");
    require_shape(actions, tensors[4], torch::kInt8, rows, "cue_phase");
    require_shape(actions, tensors[5], torch::kInt32, rows, "phase_tick");
    require_shape(actions, tensors[6], torch::kInt32, rows, "delay_remaining");
    require_shape(actions, tensors[7], torch::kBool, rows, "decision_due");
    require_shape(actions, tensors[8], torch::kInt8, rows, "correct_action");
    require_shape(actions, tensors[9], torch::kInt8, rows, "last_response");
    require_shape(actions, tensors[10], torch::kInt8, rows, "correctness");
    require_shape(actions, tensors[11], torch::kInt8, rows, "consequence_status");
    require_shape(actions, tensors[12], torch::kInt64, rows, "task_rng_state");
    require_shape(actions, tensors[13], torch::kInt64, rows, "task_rng_cursor");
    require_shape(actions, tensors[14], torch::kFloat32, rows, "health");
    require_shape(actions, tensors[15], torch::kFloat32, rows, "reproduction_scale");
    require_shape(actions, tensors[16], torch::kInt64, {cells, 6}, "counters");
    require_shape(actions, tensors[17], torch::kFloat64, {cells}, "damage_total");
    require_shape(actions, tensors[18], torch::kFloat32, {ticks, cells, capacity, kObservationColumns}, "observation_by_tick");
    TORCH_CHECK(std::isfinite(start_health) && start_health > 0.0, "GMNW start health must be positive and finite");

    GmnwConfigV1 parsed{
        exact_int(config, "cue_alphabet"),
        exact_int(config, "cue_ticks"),
        exact_int(config, "delay_ticks"),
        exact_int(config, "consequence_ticks"),
        exact_bool(config, "distractors_enabled"),
        exact_int(config, "task_seed"),
        exact_int(config, "grace_generation_count"),
        exact_int(config, "severe_generation_count"),
        exact_float(config, "grace_damage_fraction"),
        exact_float(config, "severe_damage_fraction"),
        exact_float(config, "lethal_damage_fraction"),
        exact_float(config, "grace_reproduction_scale"),
    };
    TORCH_CHECK(parsed.cue_alphabet == 2 || parsed.cue_alphabet == 4, "GMNW cue alphabet is invalid");
    TORCH_CHECK(parsed.cue_ticks > 0 && parsed.delay_ticks > 0 && parsed.consequence_ticks > 0, "GMNW phase durations are invalid");
    TORCH_CHECK(parsed.grace_generations > 0 && parsed.severe_generations > 0, "GMNW pressure horizon is invalid");

    pybind11::dict receipt;
    receipt["schema_version"] = "eiel_gmnw_cuda_operation_v1";
    receipt["native_owner"] = true;
    receipt["device_resident"] = true;
    receipt["python_tick_or_generation_ownership"] = false;
    receipt["hidden_fallback"] = false;
    receipt["runtime_jit"] = false;
    receipt["tick_count"] = ticks;
    receipt["cell_count"] = cells;
    receipt["capacity"] = capacity;
    if (cancel_requested) {
        receipt["cancelled"] = true;
        receipt["kernel_launch_count"] = static_cast<std::int64_t>(0);
        return receipt;
    }

    c10::cuda::CUDAGuard guard(actions.device());
    auto status = torch::zeros(
        {cells},
        torch::TensorOptions().dtype(torch::kInt32).device(actions.device())
    );
    const auto stream = at::cuda::getCurrentCUDAStream(actions.device().index());
    gmnw_run_resident_v1<<<static_cast<unsigned int>(cells), 1, 0, stream.stream()>>>(
        actions.data_ptr<std::int64_t>(),
        generation_by_tick.data_ptr<std::int64_t>(),
        ticks,
        cells,
        capacity,
        static_cast<float>(start_health),
        parsed,
        tensors[0].data_ptr<bool>(),
        tensors[1].data_ptr<std::int64_t>(),
        tensors[2].data_ptr<std::int64_t>(),
        tensors[3].data_ptr<std::int8_t>(),
        tensors[4].data_ptr<std::int8_t>(),
        tensors[5].data_ptr<std::int32_t>(),
        tensors[6].data_ptr<std::int32_t>(),
        tensors[7].data_ptr<bool>(),
        tensors[8].data_ptr<std::int8_t>(),
        tensors[9].data_ptr<std::int8_t>(),
        tensors[10].data_ptr<std::int8_t>(),
        tensors[11].data_ptr<std::int8_t>(),
        tensors[12].data_ptr<std::int64_t>(),
        tensors[13].data_ptr<std::int64_t>(),
        tensors[14].data_ptr<float>(),
        tensors[15].data_ptr<float>(),
        tensors[16].data_ptr<std::int64_t>(),
        tensors[17].data_ptr<double>(),
        tensors[18].data_ptr<float>(),
        status.data_ptr<std::int32_t>()
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    receipt["cancelled"] = false;
    receipt["kernel_launch_count"] = static_cast<std::int64_t>(1);
    receipt["status"] = status;
    return receipt;
}

pybind11::dict eiel_gmnw_cuda_capability_v1() {
    pybind11::dict out;
    out["abi_version"] = 1;
    out["world"] = "GRU_MEMORY_NECESSITY_WORLD_V1";
    out["device_resident_private_state"] = true;
    out["single_dispatch_tick_and_generation_loop"] = true;
    out["observation_projection"] = true;
    out["consequence_lifecycle"] = true;
    out["integrated_resident_gru_controller"] = false;
    out["production_qualified"] = false;
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("run_resident", &eiel_gmnw_cuda_run_resident_v1);
    module.def("capability", &eiel_gmnw_cuda_capability_v1);
}
