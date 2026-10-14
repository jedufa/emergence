#include "cuda_architecture_b_flattened_cpu_policy_service_v1.hpp"

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

namespace eiel::cuda_architecture_b::flattened_cpu_policy_service_v1 {
namespace {

constexpr int64_t kGatherAction = 4;
constexpr int64_t kRestAction = 5;
constexpr int64_t kAttackAction = 6;
constexpr float kGumbelEpsilon = 1.0e-6f;

struct Pcg64StateV1 {
    unsigned __int128 state;
    unsigned __int128 increment;
    uint32_t has_uint32;
    uint32_t uinteger;
};

class Pcg64V1 {
public:
    explicit Pcg64V1(const Pcg64StateV1 state) : state_(state) {}

    uint64_t raw64() {
        const unsigned __int128 old_state = state_.state;
        constexpr unsigned __int128 multiplier =
            (static_cast<unsigned __int128>(2549297995355413924ULL) << 64U)
            | static_cast<unsigned __int128>(4865540595714422341ULL);
        state_.state = old_state * multiplier + state_.increment;
        const unsigned __int128 output_state = state_.state;
        const uint64_t xorshifted = static_cast<uint64_t>(
            (output_state >> 64U) ^ output_state
        );
        const uint32_t rotation = static_cast<uint32_t>(output_state >> 122U);
        if (rotation == 0U) {
            return xorshifted;
        }
        return (xorshifted >> rotation)
            | (xorshifted << ((64U - rotation) & 63U));
    }

    uint32_t raw32() {
        if (state_.has_uint32 != 0U) {
            state_.has_uint32 = 0U;
            return state_.uinteger;
        }
        const uint64_t raw = raw64();
        state_.uinteger = static_cast<uint32_t>(raw >> 32U);
        state_.has_uint32 = 1U;
        return static_cast<uint32_t>(raw & UINT64_C(0xffffffff));
    }

    float numpy_float32_01() {
        return static_cast<float>(raw32() >> 8U) * 0x1.0p-24f;
    }

    Pcg64StateV1 state() const {
        return state_;
    }

private:
    Pcg64StateV1 state_;
};

struct CellWorkV1 {
    int64_t cell;
    int64_t active_rows;
    int64_t refill_rows;
    torch::Tensor active_slots_cpu;
    torch::Tensor miss_rows_cpu;
    torch::Tensor miss_slots_cpu;
    torch::Tensor row_actions_cpu;
    torch::Tensor logits_cpu;
    torch::Tensor sampled_cache_cpu;
    torch::Tensor cache_tokens_cpu;
};

void require_cuda_tensor(
    const torch::Tensor& tensor,
    const c10::Device& device,
    const c10::ScalarType dtype,
    const torch::IntArrayRef shape,
    const char* name
) {
    TORCH_CHECK(tensor.defined(), name, " must be defined");
    TORCH_CHECK(tensor.is_cuda(), name, " must remain a CUDA tensor");
    TORCH_CHECK(tensor.device() == device, name, " must share the service CUDA device");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(tensor.scalar_type() == dtype, name, " has an invalid dtype");
    TORCH_CHECK(tensor.sizes() == shape, name, " has an invalid shape");
}

void require_finite_setting(const double value, const char* name) {
    TORCH_CHECK(std::isfinite(value), name, " must be finite");
}

torch::Tensor cpu_contiguous(const torch::Tensor& tensor) {
    return tensor.detach().to(torch::kCPU).contiguous();
}

int64_t steady_ns_v1() {
    if constexpr (kOptionDM0TelemetryEnabledV1) {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()
        ).count();
    }
    return 0;
}

int64_t tensor_bytes_v1(const torch::Tensor& tensor) {
    return tensor.numel() * static_cast<int64_t>(tensor.element_size());
}

torch::Tensor fixed_mlp_logits_cpu_exact_v1(
    const torch::Tensor& observations,
    const torch::Tensor& packed_parameters,
    const torch::Tensor& attack_bias,
    const torch::Tensor& action_temperature,
    const torch::Tensor& drives,
    const torch::Tensor& drive_gain,
    const FlattenedPatchFixedMlpCpuExactArgsV1& arguments
) {
    namespace idx = torch::indexing;

    const int64_t rows = observations.size(0);
    if (rows == 0) {
        return torch::empty(
            {0, arguments.action_count},
            torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU)
        );
    }

    int64_t offset = 0;
    auto w1 = packed_parameters
        .slice(
            1,
            offset,
            offset + arguments.hidden_size * arguments.observation_size
        )
        .view({rows, arguments.hidden_size, arguments.observation_size});
    offset += arguments.hidden_size * arguments.observation_size;
    auto b1 = packed_parameters
        .slice(1, offset, offset + arguments.hidden_size)
        .view({rows, arguments.hidden_size});
    offset += arguments.hidden_size;
    auto w2 = packed_parameters
        .slice(
            1,
            offset,
            offset + arguments.hidden_size * arguments.hidden_size
        )
        .view({rows, arguments.hidden_size, arguments.hidden_size});
    offset += arguments.hidden_size * arguments.hidden_size;
    auto b2 = packed_parameters
        .slice(1, offset, offset + arguments.hidden_size)
        .view({rows, arguments.hidden_size});
    offset += arguments.hidden_size;
    auto w3 = packed_parameters
        .slice(
            1,
            offset,
            offset + arguments.action_count * arguments.hidden_size
        )
        .view({rows, arguments.action_count, arguments.hidden_size});
    offset += arguments.action_count * arguments.hidden_size;
    auto b3 = packed_parameters
        .slice(1, offset, offset + arguments.action_count)
        .view({rows, arguments.action_count});

    auto h1 = torch::tanh(
        torch::bmm(w1, observations.unsqueeze(-1)).squeeze(-1) + b1
    );
    auto h2 = torch::tanh(
        torch::bmm(w2, h1.unsqueeze(-1)).squeeze(-1) + b2
    );
    auto logits_policy =
        torch::bmm(w3, h2.unsqueeze(-1)).squeeze(-1) + b3;
    logits_policy.index_put_(
        {idx::Slice(), kAttackAction},
        logits_policy.index({idx::Slice(), kAttackAction}) + attack_bias
    );
    logits_policy = logits_policy
        / torch::clamp(action_temperature, 0.05).unsqueeze(-1);

    auto logits = logits_policy.clone();
    const auto gain = torch::clamp(drive_gain, 0.0);
    logits.index_put_(
        {idx::Slice(), kGatherAction},
        logits.index({idx::Slice(), kGatherAction})
            + gain * (
                arguments.w_hunger_gather
                * drives.index({idx::Slice(), 0})
            )
    );
    logits.index_put_(
        {idx::Slice(), kRestAction},
        logits.index({idx::Slice(), kRestAction})
            + gain * (
                arguments.w_pain_rest * drives.index({idx::Slice(), 1})
                + arguments.w_fatigue_rest * drives.index({idx::Slice(), 2})
            )
    );
    logits.index_put_(
        {idx::Slice(), kAttackAction},
        logits.index({idx::Slice(), kAttackAction})
            - gain * (
                arguments.w_pain_attack * drives.index({idx::Slice(), 1})
                + arguments.w_fear_attack * drives.index({idx::Slice(), 3})
            )
    );
    return logits.contiguous();
}

torch::Tensor sample_actions_cpu_exact_v1(
    const torch::Tensor& logits,
    const int64_t cache_width,
    Pcg64V1& rng,
    const bool deterministic
) {
    const int64_t rows = logits.size(0);
    const int64_t action_count = logits.size(1);
    auto sampled = torch::empty(
        {rows, cache_width},
        torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU)
    );
    const auto logits_accessor = logits.accessor<float, 2>();
    auto sampled_accessor = sampled.accessor<int32_t, 2>();

    for (int64_t row = 0; row < rows; ++row) {
        if (deterministic) {
            int32_t best_action = 0;
            float best_score = logits_accessor[row][0];
            for (int64_t action = 1; action < action_count; ++action) {
                const float score = logits_accessor[row][action];
                if (score > best_score) {
                    best_score = score;
                    best_action = static_cast<int32_t>(action);
                }
            }
            for (int64_t cache_index = 0; cache_index < cache_width; ++cache_index) {
                sampled_accessor[row][cache_index] = best_action;
            }
            continue;
        }

        for (int64_t cache_index = 0; cache_index < cache_width; ++cache_index) {
            int32_t best_action = 0;
            float best_score = -std::numeric_limits<float>::infinity();
            for (int64_t action = 0; action < action_count; ++action) {
                float uniform = rng.numpy_float32_01();
                if (uniform < kGumbelEpsilon) {
                    uniform = kGumbelEpsilon;
                } else if (uniform > 1.0f - kGumbelEpsilon) {
                    uniform = 1.0f - kGumbelEpsilon;
                }
                const float gumbel = -std::log(-std::log(uniform));
                const float score = logits_accessor[row][action] + gumbel;
                if (action == 0 || score > best_score) {
                    best_score = score;
                    best_action = static_cast<int32_t>(action);
                }
            }
            sampled_accessor[row][cache_index] = best_action;
        }
    }
    return sampled;
}

Pcg64StateV1 load_pcg64_state_v1(
    const torch::TensorAccessor<uint64_t, 2>& state_words,
    const torch::TensorAccessor<int64_t, 2>& buffer_words,
    const int64_t cell
) {
    TORCH_CHECK(
        buffer_words[cell][0] == 0 || buffer_words[cell][0] == 1,
        "policy PCG64 has_uint32 must be 0 or 1 for cell ",
        cell
    );
    TORCH_CHECK(
        buffer_words[cell][1] >= 0
            && static_cast<uint64_t>(buffer_words[cell][1])
                <= std::numeric_limits<uint32_t>::max(),
        "policy PCG64 buffered uint32 is out of range for cell ",
        cell
    );
    return {
        (static_cast<unsigned __int128>(state_words[cell][1]) << 64U)
            | state_words[cell][0],
        (static_cast<unsigned __int128>(state_words[cell][3]) << 64U)
            | state_words[cell][2],
        static_cast<uint32_t>(buffer_words[cell][0]),
        static_cast<uint32_t>(buffer_words[cell][1]),
    };
}

void store_pcg64_state_v1(
    const Pcg64StateV1 state,
    torch::TensorAccessor<uint64_t, 2> state_words,
    torch::TensorAccessor<int64_t, 2> buffer_words,
    const int64_t cell
) {
    state_words[cell][0] = static_cast<uint64_t>(state.state);
    state_words[cell][1] = static_cast<uint64_t>(state.state >> 64U);
    state_words[cell][2] = static_cast<uint64_t>(state.increment);
    state_words[cell][3] = static_cast<uint64_t>(state.increment >> 64U);
    buffer_words[cell][0] = static_cast<int64_t>(state.has_uint32);
    buffer_words[cell][1] = static_cast<int64_t>(state.uinteger);
}

}  // namespace

FlattenedPatchFixedMlpCpuExactReceiptV1
service_flattened_patch_fixed_mlp_cpu_exact_v1(
    const FlattenedPatchFixedMlpCpuExactArgsV1& arguments
) {
    torch::InferenceMode inference_mode;
    const int64_t service_start_ns = steady_ns_v1();
    int64_t device_to_host_time_ns = 0;
    int64_t host_to_device_time_ns = 0;
    int64_t cpu_mlp_time_ns = 0;
    int64_t cpu_sampling_time_ns = 0;
    int64_t final_stream_synchronization_time_ns = 0;
    int64_t device_to_host_bytes_lower_bound = 0;
    int64_t host_to_device_bytes_lower_bound = 0;
    int64_t device_to_host_operations = 0;
    int64_t host_to_device_operations = 0;
    int64_t host_allocation_count_lower_bound = 0;
    int64_t host_allocation_bytes_lower_bound = 0;
    const auto record_host_allocation_lower_bound = [&](const int64_t bytes) {
        if constexpr (kOptionDM0TelemetryEnabledV1) {
            TORCH_CHECK(bytes >= 0, "host allocation byte count must be non-negative");
            if (bytes == 0) {
                return;
            }
            TORCH_CHECK(
                host_allocation_count_lower_bound
                    < std::numeric_limits<int64_t>::max(),
                "host allocation count lower bound overflows int64"
            );
            TORCH_CHECK(
                host_allocation_bytes_lower_bound
                    <= std::numeric_limits<int64_t>::max() - bytes,
                "host allocation byte lower bound overflows int64"
            );
            ++host_allocation_count_lower_bound;
            host_allocation_bytes_lower_bound += bytes;
        }
    };
    const auto timed_cpu_contiguous = [&](
                                          const torch::Tensor& tensor
                                      ) -> torch::Tensor {
        if constexpr (!kOptionDM0TelemetryEnabledV1) {
            return cpu_contiguous(tensor);
        }
        const int64_t start_ns = steady_ns_v1();
        auto out = cpu_contiguous(tensor);
        device_to_host_time_ns += steady_ns_v1() - start_ns;
        const int64_t bytes = tensor_bytes_v1(out);
        device_to_host_bytes_lower_bound += bytes;
        if (bytes > 0) {
            ++device_to_host_operations;
        }
        record_host_allocation_lower_bound(bytes);
        return out;
    };
    const auto timed_to_device = [&](
                                     const torch::Tensor& tensor,
                                     const c10::Device& target_device
                                 ) -> torch::Tensor {
        if constexpr (!kOptionDM0TelemetryEnabledV1) {
            return tensor.to(target_device);
        }
        const int64_t start_ns = steady_ns_v1();
        auto out = tensor.to(target_device);
        host_to_device_time_ns += steady_ns_v1() - start_ns;
        const int64_t bytes = tensor_bytes_v1(out);
        host_to_device_bytes_lower_bound += bytes;
        if (bytes > 0) {
            ++host_to_device_operations;
        }
        return out;
    };

    TORCH_CHECK(arguments.cells > 0, "CPU exact policy service requires cells > 0");
    TORCH_CHECK(
        arguments.controller_tick >= 0,
        "CPU exact policy service requires controller_tick >= 0"
    );
    TORCH_CHECK(arguments.capacity > 0, "CPU exact policy service requires capacity > 0");
    TORCH_CHECK(
        arguments.observation_size > 0 && arguments.observation_size <= 83,
        "CPU exact policy service requires observation_size in [1, 83]"
    );
    TORCH_CHECK(
        arguments.hidden_size > 0 && arguments.hidden_size <= 64,
        "CPU exact policy service requires hidden_size in [1, 64]"
    );
    TORCH_CHECK(
        arguments.action_count == 7,
        "CPU exact Patch policy service requires the authoritative seven actions"
    );
    TORCH_CHECK(
        arguments.drive_width >= 4,
        "CPU exact Patch policy service requires at least four drive channels"
    );
    TORCH_CHECK(
        arguments.cache_capacity > 0
            && arguments.cache_width > 0
            && arguments.cache_width <= arguments.cache_capacity,
        "CPU exact policy service cache dimensions are invalid"
    );
    const int64_t expected_packed_width =
        arguments.hidden_size * arguments.observation_size
        + arguments.hidden_size
        + arguments.hidden_size * arguments.hidden_size
        + arguments.hidden_size
        + arguments.action_count * arguments.hidden_size
        + arguments.action_count;
    TORCH_CHECK(
        arguments.packed_width == expected_packed_width,
        "CPU exact policy service fixed-MLP packed width mismatch"
    );
    for (const auto& setting : std::vector<std::pair<double, const char*>>{
             {arguments.w_hunger_gather, "w_hunger_gather"},
             {arguments.w_pain_rest, "w_pain_rest"},
             {arguments.w_fatigue_rest, "w_fatigue_rest"},
             {arguments.w_pain_attack, "w_pain_attack"},
             {arguments.w_fear_attack, "w_fear_attack"},
         }) {
        require_finite_setting(setting.first, setting.second);
    }

    TORCH_CHECK(
        arguments.ordered_slots.defined() && arguments.ordered_slots.is_cuda(),
        "ordered_slots must establish a CUDA service device"
    );
    TORCH_CHECK(
        arguments.ordered_slots.dim() == 3,
        "ordered_slots must be rank 3"
    );
    const c10::Device device = arguments.ordered_slots.device();
    const c10::cuda::CUDAGuard device_guard(device);
    const int64_t output_ticks = arguments.ordered_slots.size(1);
    TORCH_CHECK(output_ticks > 0, "CPU exact policy service requires output ticks");
    TORCH_CHECK(
        arguments.output_tick >= 0 && arguments.output_tick < output_ticks,
        "CPU exact policy service output_tick is out of range"
    );
    TORCH_CHECK(
        (output_ticks == 1 && arguments.output_tick == 0)
            || (
                output_ticks > 1
                && arguments.output_tick == arguments.controller_tick
            ),
        "controller_tick/output_tick do not match rolling or full output storage"
    );

    require_cuda_tensor(
        arguments.ordered_slots,
        device,
        torch::kInt32,
        {arguments.cells, output_ticks, arguments.capacity},
        "ordered_slots"
    );
    require_cuda_tensor(
        arguments.observations,
        device,
        torch::kFloat32,
        {
            arguments.cells,
            output_ticks,
            arguments.capacity,
            arguments.observation_size,
        },
        "observations"
    );
    require_cuda_tensor(
        arguments.death_slots_miss_rows,
        device,
        torch::kInt64,
        {arguments.cells, output_ticks, arguments.capacity},
        "death_slots_miss_rows"
    );
    for (const auto& named :
         std::vector<std::pair<const char*, const torch::Tensor*>>{
             {"active_count", &arguments.active_count},
             {"policy_refill_rows", &arguments.policy_refill_rows},
             {"policy_cache_hit_rows", &arguments.policy_cache_hit_rows},
         }) {
        require_cuda_tensor(
            *named.second,
            device,
            torch::kInt64,
            {arguments.cells, output_ticks},
            named.first
        );
    }
    require_cuda_tensor(
        arguments.mlp_pack,
        device,
        torch::kFloat32,
        {arguments.cells, arguments.capacity, arguments.packed_width},
        "mlp_pack"
    );
    require_cuda_tensor(
        arguments.drives,
        device,
        torch::kFloat32,
        {arguments.cells, arguments.capacity, arguments.drive_width},
        "drives"
    );
    for (const auto& named :
         std::vector<std::pair<const char*, const torch::Tensor*>>{
             {"attack_bias", &arguments.attack_bias},
             {"action_temperature", &arguments.action_temperature},
             {"drive_logit_gain", &arguments.drive_logit_gain},
         }) {
        require_cuda_tensor(
            *named.second,
            device,
            torch::kFloat32,
            {arguments.cells, arguments.capacity},
            named.first
        );
    }
    require_cuda_tensor(
        arguments.slot_tokens,
        device,
        torch::kInt64,
        {arguments.cells, arguments.capacity},
        "slot_tokens"
    );
    require_cuda_tensor(
        arguments.action_cache_actions,
        device,
        torch::kInt32,
        {arguments.cells, arguments.capacity, arguments.cache_capacity},
        "action_cache_actions"
    );
    require_cuda_tensor(
        arguments.action_cache_token,
        device,
        torch::kInt64,
        {arguments.cells, arguments.capacity},
        "action_cache_token"
    );
    require_cuda_tensor(
        arguments.action_cache_cursor,
        device,
        torch::kInt32,
        {arguments.cells, arguments.capacity},
        "action_cache_cursor"
    );
    require_cuda_tensor(
        arguments.action_cache_next,
        device,
        torch::kInt32,
        {arguments.cells, arguments.capacity},
        "action_cache_next"
    );
    require_cuda_tensor(
        arguments.logits,
        device,
        torch::kFloat32,
        {
            arguments.cells,
            output_ticks,
            arguments.capacity,
            arguments.action_count,
        },
        "logits"
    );
    for (const auto& named :
         std::vector<std::pair<const char*, const torch::Tensor*>>{
             {"row_actions", &arguments.row_actions},
             {"actions_by_slot", &arguments.actions_by_slot},
         }) {
        require_cuda_tensor(
            *named.second,
            device,
            torch::kInt32,
            {arguments.cells, output_ticks, arguments.capacity},
            named.first
        );
    }
    require_cuda_tensor(
        arguments.policy_pcg64_state_words,
        device,
        torch::kUInt64,
        {arguments.cells, 4},
        "policy_pcg64_state_words"
    );
    require_cuda_tensor(
        arguments.policy_pcg64_buffer_words,
        device,
        torch::kInt64,
        {arguments.cells, 2},
        "policy_pcg64_buffer_words"
    );
    require_cuda_tensor(
        arguments.policy_np_semantic_draws,
        device,
        torch::kInt64,
        {arguments.cells},
        "policy_np_semantic_draws"
    );
    require_cuda_tensor(
        arguments.status,
        device,
        torch::kInt32,
        {arguments.cells},
        "status"
    );
    const bool has_policy_service_bypass =
        arguments.policy_service_bypass_by_cell.defined();
    if (has_policy_service_bypass) {
        require_cuda_tensor(
            arguments.policy_service_bypass_by_cell,
            device,
            torch::kBool,
            {arguments.cells},
            "policy_service_bypass_by_cell"
        );
    }

    auto status_cpu = timed_cpu_contiguous(arguments.status);
    auto policy_service_bypass_cpu = has_policy_service_bypass
        ? timed_cpu_contiguous(arguments.policy_service_bypass_by_cell)
        : torch::zeros(
              {arguments.cells},
              torch::TensorOptions().dtype(torch::kBool).device(torch::kCPU)
          );
    if constexpr (kOptionDM0TelemetryEnabledV1) {
        if (!has_policy_service_bypass) {
            record_host_allocation_lower_bound(
                tensor_bytes_v1(policy_service_bypass_cpu)
            );
        }
    }
    auto active_counts_cpu = timed_cpu_contiguous(
        arguments.active_count.select(1, arguments.output_tick)
    );
    auto refill_counts_cpu = timed_cpu_contiguous(
        arguments.policy_refill_rows.select(1, arguments.output_tick)
    );
    auto hit_counts_cpu = timed_cpu_contiguous(
        arguments.policy_cache_hit_rows.select(1, arguments.output_tick)
    );
    auto state_words_after_cpu = timed_cpu_contiguous(
        arguments.policy_pcg64_state_words
    );
    auto buffer_words_after_cpu = timed_cpu_contiguous(
        arguments.policy_pcg64_buffer_words
    );
    auto semantic_draws_after_cpu = timed_cpu_contiguous(
        arguments.policy_np_semantic_draws
    );

    const auto active_counts = active_counts_cpu.accessor<int64_t, 1>();
    const auto refill_counts = refill_counts_cpu.accessor<int64_t, 1>();
    const auto hit_counts = hit_counts_cpu.accessor<int64_t, 1>();
    auto state_words_after =
        state_words_after_cpu.accessor<uint64_t, 2>();
    auto buffer_words_after =
        buffer_words_after_cpu.accessor<int64_t, 2>();
    auto semantic_draws_after =
        semantic_draws_after_cpu.accessor<int64_t, 1>();
    const auto status_values = status_cpu.accessor<int32_t, 1>();
    const auto policy_service_bypass =
        policy_service_bypass_cpu.accessor<bool, 1>();

    std::vector<CellWorkV1> work;
    work.reserve(static_cast<size_t>(arguments.cells));
    record_host_allocation_lower_bound(
        arguments.cells * static_cast<int64_t>(sizeof(CellWorkV1))
    );
    int64_t cells_serviced = 0;
    int64_t total_active_rows = 0;
    int64_t total_hit_rows = 0;
    int64_t total_refill_rows = 0;
    int64_t total_gumbel_draws = 0;
    if constexpr (!kOptionDM0TelemetryEnabledV1) {
        device_to_host_operations =
            7 + (has_policy_service_bypass ? 1 : 0);
        host_to_device_operations = 3;
    }
    for (int64_t cell = 0; cell < arguments.cells; ++cell) {
        if (status_values[cell] != 0 || policy_service_bypass[cell]) {
            continue;
        }
        ++cells_serviced;
        if constexpr (!kOptionDM0TelemetryEnabledV1) {
            device_to_host_operations += 10;
            host_to_device_operations += 6;
        }
        const int64_t active_rows = active_counts[cell];
        const int64_t refill_rows = refill_counts[cell];
        const int64_t cache_hit_rows = hit_counts[cell];
        TORCH_CHECK(
            active_rows >= 0 && active_rows <= arguments.capacity,
            "active row count is out of range for cell ",
            cell
        );
        TORCH_CHECK(
            refill_rows >= 0 && refill_rows <= active_rows,
            "refill row count is out of range for cell ",
            cell
        );
        TORCH_CHECK(
            cache_hit_rows >= 0
                && cache_hit_rows <= active_rows
                && cache_hit_rows + refill_rows == active_rows,
            "prepared cache accounting does not conserve active rows for cell ",
            cell
        );

        auto active_slots_cpu = timed_cpu_contiguous(
            arguments.ordered_slots
                .select(0, cell)
                .select(0, arguments.output_tick)
                .narrow(0, 0, active_rows)
        );
        auto row_actions_cpu = timed_cpu_contiguous(
            arguments.row_actions
                .select(0, cell)
                .select(0, arguments.output_tick)
                .narrow(0, 0, active_rows)
        );
        auto miss_rows_cpu = timed_cpu_contiguous(
            arguments.death_slots_miss_rows
                .select(0, cell)
                .select(0, arguments.output_tick)
                .narrow(0, 0, refill_rows)
        );

        const auto active_slots = active_slots_cpu.accessor<int32_t, 1>();
        const auto miss_row_values = miss_rows_cpu.accessor<int64_t, 1>();
        auto row_action_values = row_actions_cpu.accessor<int32_t, 1>();
        std::vector<uint8_t> slot_seen(
            static_cast<size_t>(arguments.capacity),
            0U
        );
        std::vector<uint8_t> miss_seen(
            static_cast<size_t>(active_rows),
            0U
        );
        record_host_allocation_lower_bound(arguments.capacity);
        if (active_rows > 0) {
            record_host_allocation_lower_bound(active_rows);
        }
        for (int64_t row = 0; row < active_rows; ++row) {
            const int64_t slot = active_slots[row];
            TORCH_CHECK(
                slot >= 0 && slot < arguments.capacity,
                "ordered active slot is out of range for cell ",
                cell,
                ", row ",
                row
            );
            TORCH_CHECK(
                slot_seen[static_cast<size_t>(slot)] == 0U,
                "ordered active slots contain a duplicate for cell ",
                cell
            );
            slot_seen[static_cast<size_t>(slot)] = 1U;
        }
        for (int64_t index = 0; index < refill_rows; ++index) {
            const int64_t row = miss_row_values[index];
            TORCH_CHECK(
                row >= 0 && row < active_rows,
                "prepared miss row is out of range for cell ",
                cell
            );
            TORCH_CHECK(
                miss_seen[static_cast<size_t>(row)] == 0U,
                "prepared miss rows contain a duplicate for cell ",
                cell
            );
            miss_seen[static_cast<size_t>(row)] = 1U;
        }
        for (int64_t row = 0; row < active_rows; ++row) {
            if (miss_seen[static_cast<size_t>(row)] == 0U) {
                TORCH_CHECK(
                    row_action_values[row] >= 0
                        && row_action_values[row] < arguments.action_count,
                    "prepare kernel did not materialize a valid cache-hit action "
                    "for cell ",
                    cell,
                    ", row ",
                    row
                );
            }
        }

        auto miss_slots_cpu = active_slots_cpu
            .index_select(0, miss_rows_cpu)
            .to(torch::kLong)
            .contiguous();
        record_host_allocation_lower_bound(tensor_bytes_v1(miss_slots_cpu));
        auto miss_rows_cuda = timed_to_device(miss_rows_cpu, device);
        auto miss_slots_cuda = timed_to_device(miss_slots_cpu, device);
        auto observations_cpu = timed_cpu_contiguous(
            arguments.observations
                .select(0, cell)
                .select(0, arguments.output_tick)
                .index_select(0, miss_rows_cuda)
        );
        auto packed_parameters_cpu = timed_cpu_contiguous(
            arguments.mlp_pack
                .select(0, cell)
                .index_select(0, miss_slots_cuda)
        );
        auto drives_cpu = timed_cpu_contiguous(
            arguments.drives
                .select(0, cell)
                .index_select(0, miss_slots_cuda)
        );
        auto attack_bias_cpu = timed_cpu_contiguous(
            arguments.attack_bias
                .select(0, cell)
                .index_select(0, miss_slots_cuda)
        );
        auto action_temperature_cpu = timed_cpu_contiguous(
            arguments.action_temperature
                .select(0, cell)
                .index_select(0, miss_slots_cuda)
        );
        auto drive_gain_cpu = timed_cpu_contiguous(
            arguments.drive_logit_gain
                .select(0, cell)
                .index_select(0, miss_slots_cuda)
        );
        auto cache_tokens_cpu = timed_cpu_contiguous(
            arguments.slot_tokens
                .select(0, cell)
                .index_select(0, miss_slots_cuda)
        );

        const int64_t mlp_start_ns = steady_ns_v1();
        auto logits_cpu = fixed_mlp_logits_cpu_exact_v1(
            observations_cpu,
            packed_parameters_cpu,
            attack_bias_cpu,
            action_temperature_cpu,
            drives_cpu,
            drive_gain_cpu,
            arguments
        );
        record_host_allocation_lower_bound(tensor_bytes_v1(logits_cpu));
        cpu_mlp_time_ns += steady_ns_v1() - mlp_start_ns;
        const Pcg64StateV1 pre_sample_state = load_pcg64_state_v1(
            state_words_after,
            buffer_words_after,
            cell
        );
        Pcg64V1 rng(pre_sample_state);
        const int64_t sampling_start_ns = steady_ns_v1();
        auto sampled_cache_cpu = sample_actions_cpu_exact_v1(
            logits_cpu,
            arguments.cache_width,
            rng,
            arguments.deterministic_actions
        );
        record_host_allocation_lower_bound(tensor_bytes_v1(sampled_cache_cpu));
        cpu_sampling_time_ns += steady_ns_v1() - sampling_start_ns;
        const auto sampled_actions =
            sampled_cache_cpu.accessor<int32_t, 2>();
        for (int64_t index = 0; index < refill_rows; ++index) {
            row_action_values[miss_row_values[index]] = sampled_actions[index][0];
        }
        for (int64_t row = 0; row < active_rows; ++row) {
            TORCH_CHECK(
                row_action_values[row] >= 0
                    && row_action_values[row] < arguments.action_count,
                "CPU exact service did not complete every active row action "
                "for cell ",
                cell
            );
        }

        int64_t gumbel_draws = 0;
        if (!arguments.deterministic_actions) {
            TORCH_CHECK(
                refill_rows
                    <= std::numeric_limits<int64_t>::max()
                        / arguments.cache_width
                        / arguments.action_count,
                "policy Gumbel draw count overflows int64 for cell ",
                cell
            );
            gumbel_draws =
                refill_rows * arguments.cache_width * arguments.action_count;
            TORCH_CHECK(
                semantic_draws_after[cell] >= 0
                    && semantic_draws_after[cell]
                        <= std::numeric_limits<int64_t>::max() - gumbel_draws,
                "policy semantic draw counter overflows for cell ",
                cell
            );
            semantic_draws_after[cell] += gumbel_draws;
            store_pcg64_state_v1(
                rng.state(),
                state_words_after,
                buffer_words_after,
                cell
            );
        }

        TORCH_CHECK(
            total_active_rows <= std::numeric_limits<int64_t>::max() - active_rows
                && total_hit_rows
                    <= std::numeric_limits<int64_t>::max() - cache_hit_rows
                && total_refill_rows
                    <= std::numeric_limits<int64_t>::max() - refill_rows
                && total_gumbel_draws
                    <= std::numeric_limits<int64_t>::max() - gumbel_draws,
            "CPU exact policy service receipt counter overflow"
        );
        total_active_rows += active_rows;
        total_hit_rows += cache_hit_rows;
        total_refill_rows += refill_rows;
        total_gumbel_draws += gumbel_draws;
        if constexpr (!kOptionDM0TelemetryEnabledV1) {
            if (refill_rows > 0) {
                host_to_device_operations += 4;
            }
        }
        work.push_back({
            cell,
            active_rows,
            refill_rows,
            std::move(active_slots_cpu),
            std::move(miss_rows_cpu),
            std::move(miss_slots_cpu),
            std::move(row_actions_cpu),
            std::move(logits_cpu),
            std::move(sampled_cache_cpu),
            std::move(cache_tokens_cpu),
        });
    }

    for (const CellWorkV1& cell_work : work) {
        const auto active_slots_long = cell_work.active_slots_cpu.to(torch::kLong);
        record_host_allocation_lower_bound(tensor_bytes_v1(active_slots_long));
        const auto active_slots_cuda = timed_to_device(active_slots_long, device);
        const auto miss_rows_cuda = timed_to_device(cell_work.miss_rows_cpu, device);
        const auto miss_slots_cuda = timed_to_device(cell_work.miss_slots_cpu, device);
        const auto complete_row_actions_cuda =
            timed_to_device(cell_work.row_actions_cpu, device);

        auto row_actions_view = arguments.row_actions
            .select(0, cell_work.cell)
            .select(0, arguments.output_tick);
        row_actions_view
            .narrow(0, 0, cell_work.active_rows)
            .copy_(complete_row_actions_cuda);
        arguments.actions_by_slot
            .select(0, cell_work.cell)
            .select(0, arguments.output_tick)
            .index_copy_(0, active_slots_cuda, complete_row_actions_cuda);

        if (cell_work.refill_rows == 0) {
            continue;
        }
        arguments.logits
            .select(0, cell_work.cell)
            .select(0, arguments.output_tick)
            .index_copy_(0, miss_rows_cuda, timed_to_device(cell_work.logits_cpu, device));

        auto cache_rows = arguments.action_cache_actions
            .select(0, cell_work.cell)
            .index_select(0, miss_slots_cuda)
            .contiguous();
        cache_rows
            .slice(1, 0, arguments.cache_width)
            .copy_(timed_to_device(cell_work.sampled_cache_cpu, device));
        arguments.action_cache_actions
            .select(0, cell_work.cell)
            .index_copy_(0, miss_slots_cuda, cache_rows);
        arguments.action_cache_token
            .select(0, cell_work.cell)
            .index_copy_(0, miss_slots_cuda, timed_to_device(cell_work.cache_tokens_cpu, device));
        arguments.action_cache_cursor
            .select(0, cell_work.cell)
            .index_fill_(0, miss_slots_cuda, 1);
        arguments.action_cache_next
            .select(0, cell_work.cell)
            .index_copy_(
                0,
                miss_slots_cuda,
                timed_to_device(cell_work.sampled_cache_cpu.select(1, 0), device)
            );
    }

    arguments.policy_pcg64_state_words.copy_(
        timed_to_device(state_words_after_cpu, device)
    );
    arguments.policy_pcg64_buffer_words.copy_(
        timed_to_device(buffer_words_after_cpu, device)
    );
    arguments.policy_np_semantic_draws.copy_(
        timed_to_device(semantic_draws_after_cpu, device)
    );
    const int64_t final_sync_start_ns = steady_ns_v1();
    c10::cuda::getCurrentCUDAStream(device.index()).synchronize();
    final_stream_synchronization_time_ns =
        steady_ns_v1() - final_sync_start_ns;

    return {
        arguments.controller_tick,
        arguments.output_tick,
        cells_serviced,
        total_active_rows,
        total_hit_rows,
        total_refill_rows,
        total_gumbel_draws,
        device_to_host_operations,
        host_to_device_operations,
        device_to_host_operations + host_to_device_operations,
        1,
        kOptionDM0TelemetryEnabledV1,
        steady_ns_v1() - service_start_ns,
        device_to_host_time_ns,
        host_to_device_time_ns,
        cpu_mlp_time_ns,
        cpu_sampling_time_ns,
        final_stream_synchronization_time_ns,
        device_to_host_bytes_lower_bound,
        host_to_device_bytes_lower_bound,
        host_allocation_count_lower_bound,
        host_allocation_bytes_lower_bound,
    };
}

}  // namespace eiel::cuda_architecture_b::flattened_cpu_policy_service_v1
