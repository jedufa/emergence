#include "../../engine/perf/cuda_architecture_b_compact_summary_metadata_retention_v1.cuh"

#include <cuda_runtime.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <exception>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace retention =
    eiel::cuda_architecture_b::compact_summary_metadata_retention_v1;

namespace {

constexpr std::int64_t kCells = 2;
constexpr std::int64_t kGenerations = 3;
constexpr std::int64_t kCapacity = 4;
constexpr std::int64_t kAgentIdCapacity = 32;
constexpr std::int64_t kEventCapacity = 64;
constexpr std::int64_t kPolicyTapeCapacity = 16;
constexpr std::int64_t kMutationStateBytes = 5056;
constexpr std::int64_t kDescriptorFields = 132;
constexpr std::int64_t kOutputFields = 132;
constexpr std::int64_t kStepsPerGeneration = 3;

enum I64Field : std::int32_t {
    kGeneration = 0,
    kTick,
    kActiveCount,
    kIdentityPrefix,
    kFreeQueueCount,
    kPendingFreeCount,
    kEventCursor,
    kPopulationNextId,
    kPolicyTapeCursor,
    kPythonIndex,
    kI64FieldCount,
};

struct InitialAuthority {
    std::int64_t runtime_event_cursor;
    std::int64_t lifecycle_event_cursor;
    std::int64_t generation_calls;
    std::int64_t restore_count;
    std::int64_t policy_py_semantic_draws;
};

enum PcgAuthority : std::int32_t {
    kSharedEnvPolicy = 0,
    kRunner,
    kPopulation,
    kBirthSpawn,
    kPcgAuthorityCount,
};

void check_cuda(const cudaError_t status, const char* const operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(
            std::string(operation) + ": " + cudaGetErrorString(status)
        );
    }
}

template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(const std::size_t count) : count_(count) {
        check_cuda(
            cudaMalloc(reinterpret_cast<void**>(&pointer_), count * sizeof(T)),
            "cudaMalloc"
        );
    }

    ~DeviceBuffer() {
        if (pointer_ != nullptr) {
            cudaFree(pointer_);
        }
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    T* data() noexcept { return pointer_; }
    const T* data() const noexcept { return pointer_; }
    std::size_t size() const noexcept { return count_; }

    void upload(const std::vector<T>& values) {
        if (values.size() != count_) {
            throw std::runtime_error("upload extent mismatch");
        }
        check_cuda(
            cudaMemcpy(
                pointer_,
                values.data(),
                count_ * sizeof(T),
                cudaMemcpyHostToDevice
            ),
            "cudaMemcpy host to device"
        );
    }

    void upload_bytes(const std::uint8_t* const values) {
        static_assert(sizeof(T) == 1);
        check_cuda(
            cudaMemcpy(
                pointer_, values, count_, cudaMemcpyHostToDevice
            ),
            "cudaMemcpy byte host to device"
        );
    }

    std::vector<T> download() const {
        std::vector<T> result(count_);
        check_cuda(
            cudaMemcpy(
                result.data(),
                pointer_,
                count_ * sizeof(T),
                cudaMemcpyDeviceToHost
            ),
            "cudaMemcpy device to host"
        );
        return result;
    }

private:
    T* pointer_ = nullptr;
    std::size_t count_ = 0;
};

struct HostState {
    std::vector<std::int32_t> upstream;
    std::vector<std::int64_t> i64;
    std::vector<std::uint64_t> pcg_state;
    std::vector<std::int64_t> pcg_buffer;
    std::vector<std::uint32_t> python_words;
    std::vector<std::uint8_t> python_has_gauss;
    std::vector<double> python_gauss_next;
    std::vector<std::int64_t> semantic;
    std::vector<std::int64_t> runtime_bookkeeping;
    std::vector<double> policy_energy;
    std::vector<std::uint8_t> alive;
    std::vector<std::int64_t> slot_to_agent_id;
};

struct DeviceState {
    DeviceBuffer<std::int32_t> upstream{kCells};
    DeviceBuffer<std::int64_t> i64{kI64FieldCount * kCells};
    DeviceBuffer<std::uint64_t> pcg_state{
        kPcgAuthorityCount * kCells * retention::kPcg64StateWordsV1
    };
    DeviceBuffer<std::int64_t> pcg_buffer{
        kPcgAuthorityCount * kCells * retention::kPcg64BufferWordsV1
    };
    DeviceBuffer<std::uint32_t> python_words{
        kCells * retention::kPythonMt19937WordsV1
    };
    DeviceBuffer<bool> python_has_gauss{kCells};
    DeviceBuffer<double> python_gauss_next{kCells};
    DeviceBuffer<std::int64_t> semantic{
        retention::kSemanticStreamCounterCountV1 * kCells
    };
    DeviceBuffer<std::int64_t> runtime_bookkeeping{
        kCells * kGenerations * kStepsPerGeneration
            * retention::kRuntimeBookkeepingColumnCountV1
    };
    DeviceBuffer<double> policy_energy{kCells * kCapacity};
    DeviceBuffer<bool> alive{kCells * kCapacity};
    DeviceBuffer<std::int64_t> slot_to_agent_id{kCells * kCapacity};
    DeviceBuffer<std::uint8_t> raw_digests{
        kGenerations * kCells * kOutputFields * retention::kDigestBytesV1
    };
    DeviceBuffer<std::int32_t> descriptor_status{
        kGenerations * kCells * kDescriptorFields
    };
    DeviceBuffer<std::uint8_t> descriptor_validity{
        kGenerations * kCells * kDescriptorFields
    };

    void upload_state(const HostState& state) {
        upstream.upload(state.upstream);
        i64.upload(state.i64);
        pcg_state.upload(state.pcg_state);
        pcg_buffer.upload(state.pcg_buffer);
        python_words.upload(state.python_words);
        python_has_gauss.upload_bytes(state.python_has_gauss.data());
        python_gauss_next.upload(state.python_gauss_next);
        semantic.upload(state.semantic);
        runtime_bookkeeping.upload(state.runtime_bookkeeping);
        policy_energy.upload(state.policy_energy);
        alive.upload_bytes(state.alive.data());
        slot_to_agent_id.upload(state.slot_to_agent_id);
    }

    retention::MetadataRetentionInputsV1 inputs(
        const std::int64_t generation_index,
        const InitialAuthority& initial
    ) const {
        const auto i64_pointer = [this](const I64Field field) {
            return i64.data() + static_cast<std::int64_t>(field) * kCells;
        };
        const auto state_pointer = [this](const PcgAuthority authority) {
            return pcg_state.data()
                + static_cast<std::int64_t>(authority)
                    * kCells * retention::kPcg64StateWordsV1;
        };
        const auto buffer_pointer = [this](const PcgAuthority authority) {
            return pcg_buffer.data()
                + static_cast<std::int64_t>(authority)
                    * kCells * retention::kPcg64BufferWordsV1;
        };
        const auto semantic_pointer = [this](const std::int32_t counter) {
            return semantic.data() + static_cast<std::int64_t>(counter) * kCells;
        };
        retention::MetadataRetentionInputsV1 result{};
        result.upstream_status = upstream.data();
        result.generation = i64_pointer(kGeneration);
        result.tick = i64_pointer(kTick);
        result.active_count = i64_pointer(kActiveCount);
        result.agent_id_to_slot_prefix_length = i64_pointer(kIdentityPrefix);
        result.slot_free_queue_count = i64_pointer(kFreeQueueCount);
        result.pending_free_count = i64_pointer(kPendingFreeCount);
        result.event_cursor = i64_pointer(kEventCursor);
        result.runtime_bookkeeping_by_tick = runtime_bookkeeping.data();
        result.population_next_id = i64_pointer(kPopulationNextId);
        result.policy_birth_tape_cursor = i64_pointer(kPolicyTapeCursor);
        result.shared_env_policy_pcg64_state_words =
            state_pointer(kSharedEnvPolicy);
        result.shared_env_policy_pcg64_buffer_words =
            buffer_pointer(kSharedEnvPolicy);
        result.runner_pcg64_state_words = state_pointer(kRunner);
        result.runner_pcg64_buffer_words = buffer_pointer(kRunner);
        result.population_pcg64_state_words = state_pointer(kPopulation);
        result.population_pcg64_buffer_words = buffer_pointer(kPopulation);
        result.birth_spawn_pcg64_state_words = state_pointer(kBirthSpawn);
        result.birth_spawn_pcg64_buffer_words = buffer_pointer(kBirthSpawn);
        result.python_random_words = python_words.data();
        result.python_random_index = i64_pointer(kPythonIndex);
        result.python_random_has_gauss = python_has_gauss.data();
        result.python_random_gauss_next = python_gauss_next.data();
        result.mutation_py_semantic_draws =
            semantic_pointer(retention::kSemanticMutationPyV1);
        result.mutation_np_semantic_draws =
            semantic_pointer(retention::kSemanticMutationNumpyV1);
        result.mutation_torch_cpu_semantic_draws =
            semantic_pointer(retention::kSemanticMutationTorchCpuV1);
        result.policy_np_semantic_draws =
            semantic_pointer(retention::kSemanticPolicyNumpyV1);
        result.metrics_np_semantic_draws =
            semantic_pointer(retention::kSemanticMetricsNumpyV1);
        result.checkpoint_raw_digests = raw_digests.data();
        result.checkpoint_descriptor_status = descriptor_status.data();
        result.checkpoint_descriptor_validity = descriptor_validity.data();
        result.policy_energy_observation = policy_energy.data();
        result.alive_mask = alive.data();
        result.slot_to_agent_id = slot_to_agent_id.data();
        result.initial_runtime_event_cursor = initial.runtime_event_cursor;
        result.initial_lifecycle_event_cursor = initial.lifecycle_event_cursor;
        result.initial_generation_calls = initial.generation_calls;
        result.restore_count = initial.restore_count;
        result.initial_policy_py_semantic_draws =
            initial.policy_py_semantic_draws;
        result.generation_index = generation_index;
        result.generation_count = kGenerations;
        result.cell_count = kCells;
        result.configured_steps_per_generation = kStepsPerGeneration;
        result.capacity = kCapacity;
        result.agent_id_capacity = kAgentIdCapacity;
        result.event_capacity = kEventCapacity;
        result.policy_birth_tape_capacity = kPolicyTapeCapacity;
        result.mutation_torch_cpu_state_bytes = kMutationStateBytes;
        result.checkpoint_output_field_count = kOutputFields;
        result.checkpoint_descriptor_count = kDescriptorFields;
        result.env_policy_alias = 1U;
        return result;
    }
};

std::int64_t& i64(
    HostState& state,
    const I64Field field,
    const std::int64_t cell
) {
    return state.i64[static_cast<std::int64_t>(field) * kCells + cell];
}

std::int64_t i64(
    const HostState& state,
    const I64Field field,
    const std::int64_t cell
) {
    return state.i64[static_cast<std::int64_t>(field) * kCells + cell];
}

std::uint64_t& pcg_state(
    HostState& state,
    const PcgAuthority authority,
    const std::int64_t cell,
    const std::int32_t word
) {
    return state.pcg_state[
        (static_cast<std::int64_t>(authority) * kCells + cell)
            * retention::kPcg64StateWordsV1
        + word
    ];
}

std::uint64_t pcg_state(
    const HostState& state,
    const PcgAuthority authority,
    const std::int64_t cell,
    const std::int32_t word
) {
    return state.pcg_state[
        (static_cast<std::int64_t>(authority) * kCells + cell)
            * retention::kPcg64StateWordsV1
        + word
    ];
}

std::int64_t& pcg_buffer(
    HostState& state,
    const PcgAuthority authority,
    const std::int64_t cell,
    const std::int32_t word
) {
    return state.pcg_buffer[
        (static_cast<std::int64_t>(authority) * kCells + cell)
            * retention::kPcg64BufferWordsV1
        + word
    ];
}

std::int64_t pcg_buffer(
    const HostState& state,
    const PcgAuthority authority,
    const std::int64_t cell,
    const std::int32_t word
) {
    return state.pcg_buffer[
        (static_cast<std::int64_t>(authority) * kCells + cell)
            * retention::kPcg64BufferWordsV1
        + word
    ];
}

std::int64_t semantic(
    const HostState& state,
    const std::int32_t counter,
    const std::int64_t cell
) {
    return state.semantic[static_cast<std::int64_t>(counter) * kCells + cell];
}

std::int64_t& runtime_bookkeeping(
    HostState& state,
    const std::int64_t cell,
    const std::int64_t generation_index,
    const std::int64_t tick,
    const std::int32_t column
) {
    return state.runtime_bookkeeping[
        ((cell * kGenerations + generation_index) * kStepsPerGeneration + tick)
            * retention::kRuntimeBookkeepingColumnCountV1
        + column
    ];
}

std::int64_t runtime_bookkeeping(
    const HostState& state,
    const std::int64_t cell,
    const std::int64_t generation_index,
    const std::int64_t tick,
    const std::int32_t column
) {
    return state.runtime_bookkeeping[
        ((cell * kGenerations + generation_index) * kStepsPerGeneration + tick)
            * retention::kRuntimeBookkeepingColumnCountV1
        + column
    ];
}

std::int64_t generation_bookkeeping_sum(
    const HostState& state,
    const std::int64_t cell,
    const std::int64_t generation_index,
    const std::int32_t column
) {
    std::int64_t result = 0;
    for (std::int64_t tick = 0; tick < kStepsPerGeneration; ++tick) {
        result += runtime_bookkeeping(
            state, cell, generation_index, tick, column
        );
    }
    return result;
}

std::int64_t last_tick_bookkeeping(
    const HostState& state,
    const std::int64_t cell,
    const std::int64_t generation_index,
    const std::int32_t column
) {
    return runtime_bookkeeping(
        state,
        cell,
        generation_index,
        kStepsPerGeneration - 1,
        column
    );
}

std::int64_t cumulative_applied_actions(
    const HostState& state,
    const std::int64_t cell,
    const std::int64_t generation_index
) {
    std::int64_t result = 0;
    for (std::int64_t generation = 0;
         generation <= generation_index;
         ++generation) {
        result += generation_bookkeeping_sum(state, cell, generation, 0);
    }
    return result;
}

double from_bits(const std::uint64_t bits) {
    double result = 0.0;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

HostState make_generation_state(const std::int64_t generation_index) {
    HostState state{
        std::vector<std::int32_t>(kCells, 0),
        std::vector<std::int64_t>(kI64FieldCount * kCells, 0),
        std::vector<std::uint64_t>(
            kPcgAuthorityCount * kCells * retention::kPcg64StateWordsV1,
            0U
        ),
        std::vector<std::int64_t>(
            kPcgAuthorityCount * kCells * retention::kPcg64BufferWordsV1,
            0
        ),
        std::vector<std::uint32_t>(
            kCells * retention::kPythonMt19937WordsV1,
            0U
        ),
        std::vector<std::uint8_t>(kCells, 0U),
        std::vector<double>(kCells, 0.0),
        std::vector<std::int64_t>(
            retention::kSemanticStreamCounterCountV1 * kCells,
            0
        ),
        std::vector<std::int64_t>(
            kCells * kGenerations * kStepsPerGeneration
                * retention::kRuntimeBookkeepingColumnCountV1,
            0
        ),
        std::vector<double>(kCells * kCapacity, 0.0),
        std::vector<std::uint8_t>(kCells * kCapacity, 0U),
        std::vector<std::int64_t>(kCells * kCapacity, -1),
    };

    for (std::int64_t cell = 0; cell < kCells; ++cell) {
        i64(state, kGeneration, cell) = generation_index + 1;
        i64(state, kTick, cell) = 0;
        i64(state, kActiveCount, cell) = cell == 0 ? 2 : 3;
        i64(state, kIdentityPrefix, cell) = cell == 0 ? 8 : 12;
        i64(state, kFreeQueueCount, cell) = cell == 0 ? 2 : 1;
        i64(state, kPendingFreeCount, cell) = 0;
        i64(state, kEventCursor, cell) = 7 + generation_index * 4 + cell;
        i64(state, kPopulationNextId, cell) = 10 + generation_index * 7 + cell * 4;
        i64(state, kPolicyTapeCursor, cell) = 3 + generation_index + cell;
        i64(state, kPythonIndex, cell) = 200 + generation_index * 37 + cell;

        for (std::int64_t generation = 0;
             generation < kGenerations;
             ++generation) {
            for (std::int64_t tick = 0; tick < kStepsPerGeneration; ++tick) {
                runtime_bookkeeping(state, cell, generation, tick, 0) =
                    2 + generation + cell + tick;
                runtime_bookkeeping(state, cell, generation, tick, 1) =
                    (generation + cell + tick) & 1;
                runtime_bookkeeping(state, cell, generation, tick, 2) =
                    generation + tick;
                runtime_bookkeeping(state, cell, generation, tick, 3) =
                    tick == 0 ? 1 : 0;
                runtime_bookkeeping(state, cell, generation, tick, 4) =
                    tick == 0 ? 2 + generation + cell : 0;
            }
        }

        for (std::int32_t authority = 0;
             authority < kPcgAuthorityCount;
             ++authority) {
            for (std::int32_t word = 0;
                 word < retention::kPcg64StateWordsV1;
                 ++word) {
                pcg_state(
                    state,
                    static_cast<PcgAuthority>(authority),
                    cell,
                    word
                ) = 0x8100000000000000ULL
                    + static_cast<std::uint64_t>(generation_index) * 0x100000ULL
                    + static_cast<std::uint64_t>(authority) * 0x10000ULL
                    + static_cast<std::uint64_t>(cell) * 0x100ULL
                    + static_cast<std::uint64_t>(word);
            }
            pcg_buffer(
                state,
                static_cast<PcgAuthority>(authority),
                cell,
                0
            ) = (authority + cell + generation_index) & 1;
            pcg_buffer(
                state,
                static_cast<PcgAuthority>(authority),
                cell,
                1
            ) = 1000 + generation_index * 100 + authority * 10 + cell;
        }

        for (std::int32_t word = 0;
             word < retention::kPythonMt19937WordsV1;
             ++word) {
            state.python_words[
                cell * retention::kPythonMt19937WordsV1 + word
            ] = static_cast<std::uint32_t>(
                generation_index * 100000 + cell * 1000 + word
            );
        }
        state.python_has_gauss[cell] = cell == 1 ? 1U : 0U;
        state.python_gauss_next[cell] =
            cell == 1 ? 1.25 + generation_index : from_bits(0x7ff8000000001234ULL);
        for (std::int32_t counter = 0;
             counter < retention::kSemanticStreamCounterCountV1;
             ++counter) {
            state.semantic[counter * kCells + cell] =
                10000 * generation_index + 100 * counter + cell;
        }
    }

    state.alive = {1U, 0U, 1U, 0U, 1U, 1U, 0U, 1U};
    state.slot_to_agent_id = {2, -1, 7, -1, 9, 10, -1, 11};
    const double canonical_nan = from_bits(0x7ff8000000000000ULL);
    if (generation_index == 0) {
        state.policy_energy = {
            10.0, canonical_nan, 20.0, canonical_nan,
            1.0, 2.0, canonical_nan, 3.0,
        };
    } else {
        state.policy_energy = {
            11.0, canonical_nan, -0.0, canonical_nan,
            4.5, 5.5, canonical_nan, 6.5,
        };
    }
    return state;
}

std::array<std::uint8_t, retention::kDigestBytesV1> digest_from_hex(
    const char* const hex
) {
    std::array<std::uint8_t, retention::kDigestBytesV1> result{};
    const auto nibble = [](const char value) -> std::uint8_t {
        if (value >= '0' && value <= '9') {
            return static_cast<std::uint8_t>(value - '0');
        }
        if (value >= 'a' && value <= 'f') {
            return static_cast<std::uint8_t>(10 + value - 'a');
        }
        throw std::runtime_error("invalid digest fixture");
    };
    for (std::int32_t index = 0; index < retention::kDigestBytesV1; ++index) {
        result[index] = static_cast<std::uint8_t>(
            (nibble(hex[index * 2]) << 4U) | nibble(hex[index * 2 + 1])
        );
    }
    return result;
}

std::array<std::uint8_t, retention::kDigestBytesV1> policy_digest_fixture(
    const std::int64_t generation_index,
    const std::int64_t cell
) {
    static constexpr const char* kDigests[2][2] = {
        {
            "ff87dbcff4d1a016081cbc5272dffe3eab7e1b5610a34036e2ebf367b2de731c",
            "2d81a19547e9694f88daace9873acb2665a4dc655dfecc3f557a5d777abd7e74",
        },
        {
            "95880fdce68160797b93d9749d26f6838f537cac1b36dc8987298be26d2302d4",
            "3453bac6a8ca2e38b03dd99032ba57c4ef9abd4a535218d43507c8a285e05f37",
        },
    };
    return digest_from_hex(kDigests[generation_index][cell]);
}

std::int64_t descriptor_status_offset(
    const std::int64_t generation_index,
    const std::int64_t cell
) {
    return (generation_index * kCells + cell) * kDescriptorFields
        + retention::kMutationTorchCpuDescriptorIndexV1;
}

std::int64_t digest_byte_offset(
    const std::int64_t generation_index,
    const std::int64_t cell
) {
    return (
        (generation_index * kCells + cell) * kOutputFields
        + retention::kMutationTorchCpuRawOutputIndexV1
    ) * retention::kDigestBytesV1;
}

retention::CompactSummaryMetadataRetentionV1 expected_row(
    const HostState& state,
    const std::vector<std::uint8_t>& raw_digests,
    const InitialAuthority& initial,
    const std::int64_t generation_index,
    const std::int64_t cell
) {
    retention::CompactSummaryMetadataRetentionV1 row{};
    row.source_version = retention::kSourceVersionV1;
    row.status = retention::kRetentionSuccessV1;
    row.validity = 1U;
    row.env_policy_alias = 1U;
    row.python_random_has_gauss = state.python_has_gauss[cell];
    row.mutation_torch_descriptor_validity = 1U;
    row.flags = retention::kRequiredRetentionFlagsV1;
    row.generation_index = generation_index;
    row.cell_index = cell;
    row.generation = i64(state, kGeneration, cell);
    row.tick = i64(state, kTick, cell);
    row.configured_steps_per_generation = kStepsPerGeneration;
    row.active_count = i64(state, kActiveCount, cell);
    row.capacity = kCapacity;
    row.agent_id_to_slot_prefix_length = i64(state, kIdentityPrefix, cell);
    row.agent_id_capacity = kAgentIdCapacity;
    row.slot_free_queue_count = i64(state, kFreeQueueCount, cell);
    row.event_cursor = i64(state, kEventCursor, cell);
    row.event_capacity = kEventCapacity;
    row.runtime_event_cursor = initial.runtime_event_cursor
        + cumulative_applied_actions(state, cell, generation_index)
        + row.event_cursor - initial.lifecycle_event_cursor;
    row.population_next_id = i64(state, kPopulationNextId, cell);
    row.policy_birth_tape_cursor = i64(state, kPolicyTapeCursor, cell);
    row.policy_birth_tape_capacity = kPolicyTapeCapacity;
    row.generation_calls = initial.generation_calls
        + (generation_index + 1) * kStepsPerGeneration;
    row.restore_count = initial.restore_count;
    row.last_applied_actions = last_tick_bookkeeping(
        state, cell, generation_index, 0
    );
    row.last_moved = last_tick_bookkeeping(
        state, cell, generation_index, 1
    );
    row.last_gathered = last_tick_bookkeeping(
        state, cell, generation_index, 2
    );
    row.last_policy_refill_calls = last_tick_bookkeeping(
        state, cell, generation_index, 3
    );
    row.last_policy_refill_rows = last_tick_bookkeeping(
        state, cell, generation_index, 4
    );
    row.mutation_torch_cpu_state_bytes = kMutationStateBytes;
    row.policy_energy_observation_capacity = kCapacity;
    row.policy_energy_live_count = i64(state, kActiveCount, cell);
    row.policy_energy_finite_live_count = i64(state, kActiveCount, cell);
    row.native_runtime_rng_state = static_cast<std::int64_t>(
        pcg_state(state, kSharedEnvPolicy, cell, 0) & 0x7fffffffffffffffULL
    );

    for (std::int32_t authority = 0;
         authority < retention::kPcg64AuthorityCountV1;
         ++authority) {
        for (std::int32_t word = 0;
             word < retention::kPcg64StateWordsV1;
             ++word) {
            row.pcg64[authority].state_words[word] = pcg_state(
                state,
                static_cast<PcgAuthority>(authority),
                cell,
                word
            );
        }
        for (std::int32_t word = 0;
             word < retention::kPcg64BufferWordsV1;
             ++word) {
            row.pcg64[authority].buffer_words[word] = pcg_buffer(
                state,
                static_cast<PcgAuthority>(authority),
                cell,
                word
            );
        }
    }
    for (std::int32_t word = 0;
         word < retention::kPythonMt19937WordsV1;
         ++word) {
        row.python_random_words[word] = state.python_words[
            cell * retention::kPythonMt19937WordsV1 + word
        ];
    }
    row.python_random_index = i64(state, kPythonIndex, cell);
    row.python_random_gauss_next = state.python_gauss_next[cell];
    for (std::int32_t counter = 0;
         counter < retention::kSemanticStreamCounterCountV1;
         ++counter) {
        row.semantic_stream_counters[counter] = semantic(state, counter, cell);
    }
    row.semantic_stream_counters[retention::kSemanticPolicyPyV1] =
        initial.policy_py_semantic_draws
        + 8 * (
            generation_index
            + (initial.generation_calls > 0 ? 1 : 0)
        );
    row.mutation_torch_descriptor_index =
        retention::kMutationTorchCpuDescriptorIndexV1;
    row.mutation_torch_raw_output_index =
        retention::kMutationTorchCpuRawOutputIndexV1;
    row.mutation_torch_dtype_code = retention::kDtypeUint8V1;
    row.mutation_torch_rank = retention::kRankOneV1;
    const std::int64_t mutation_offset = digest_byte_offset(generation_index, cell);
    const auto policy_digest = policy_digest_fixture(generation_index, cell);
    for (std::int32_t index = 0; index < retention::kDigestBytesV1; ++index) {
        row.mutation_torch_raw_sha256[index] = raw_digests[mutation_offset + index];
        row.policy_energy_tagged_sha256[index] = policy_digest[index];
    }
    return row;
}

void launch(
    const retention::MetadataRetentionInputsV1& inputs,
    DeviceBuffer<retention::CompactSummaryMetadataRetentionV1>& output
) {
    retention::launch_compact_summary_metadata_retention_v1(
        inputs,
        output.data(),
        nullptr  // CUDA's default stream is a valid stream handle.
    );
    check_cuda(cudaGetLastError(), "retention kernel launch");
    check_cuda(cudaDeviceSynchronize(), "retention kernel completion");
}

void require_bytes_equal(
    const retention::CompactSummaryMetadataRetentionV1& actual,
    const retention::CompactSummaryMetadataRetentionV1& expected,
    const char* const label
) {
    if (std::memcmp(&actual, &expected, sizeof(actual)) != 0) {
        const auto* actual_bytes = reinterpret_cast<const std::uint8_t*>(&actual);
        const auto* expected_bytes = reinterpret_cast<const std::uint8_t*>(&expected);
        std::size_t offset = 0;
        while (
            offset < sizeof(actual)
            && actual_bytes[offset] == expected_bytes[offset]
        ) {
            ++offset;
        }
        throw std::runtime_error(
            std::string(label) + " byte mismatch at offset "
            + std::to_string(offset)
        );
    }
}

void require_failed_row(
    const retention::CompactSummaryMetadataRetentionV1& actual,
    const retention::RetentionStatusV1 status,
    const std::int64_t generation_index,
    const std::int64_t cell
) {
    retention::CompactSummaryMetadataRetentionV1 expected{};
    expected.source_version = retention::kSourceVersionV1;
    expected.status = status;
    expected.generation_index = generation_index;
    expected.cell_index = cell;
    require_bytes_equal(actual, expected, "failed row");
}

void require_vectors_equal_bytes(
    const void* const actual,
    const void* const expected,
    const std::size_t bytes,
    const char* const label
) {
    if (std::memcmp(actual, expected, bytes) != 0) {
        throw std::runtime_error(std::string(label) + " was mutated");
    }
}

}  // namespace

int main() {
    int device_count = 0;
    const cudaError_t device_status = cudaGetDeviceCount(&device_count);
    if (device_status != cudaSuccess || device_count == 0) {
        std::cerr << "SKIP: no CUDA device\n";
        return 77;
    }

    try {
        DeviceState device;
        DeviceBuffer<retention::CompactSummaryMetadataRetentionV1> output{
            kGenerations * kCells
        };

        std::vector<std::uint8_t> raw_digests(
            kGenerations * kCells * kOutputFields * retention::kDigestBytesV1,
            0U
        );
        std::vector<std::int32_t> descriptor_status(
            kGenerations * kCells * kDescriptorFields,
            0
        );
        std::vector<std::uint8_t> descriptor_validity(
            kGenerations * kCells * kDescriptorFields,
            0U
        );
        for (std::int64_t generation = 0;
             generation < kGenerations;
             ++generation) {
            for (std::int64_t cell = 0; cell < kCells; ++cell) {
                descriptor_validity[descriptor_status_offset(generation, cell)] = 1U;
                const std::int64_t offset = digest_byte_offset(generation, cell);
                for (std::int32_t index = 0;
                     index < retention::kDigestBytesV1;
                     ++index) {
                    raw_digests[offset + index] = static_cast<std::uint8_t>(
                        generation * 80 + cell * 32 + index
                    );
                }
            }
        }
        device.raw_digests.upload(raw_digests);
        device.descriptor_status.upload(descriptor_status);
        device.descriptor_validity.upload(descriptor_validity);
        check_cuda(
            cudaMemset(
                output.data(),
                0xcd,
                output.size()
                    * sizeof(retention::CompactSummaryMetadataRetentionV1)
            ),
            "initialize output"
        );

        const InitialAuthority pristine_initial{20, 5, 0, 3, 400};
        const InitialAuthority continuation_initial{75, 5, 7, 4, 900};

        HostState generation_zero = make_generation_state(0);
        device.upload_state(generation_zero);
        launch(device.inputs(0, pristine_initial), output);
        auto retained = output.download();
        for (std::int64_t cell = 0; cell < kCells; ++cell) {
            require_bytes_equal(
                retained[cell],
                expected_row(
                    generation_zero,
                    raw_digests,
                    pristine_initial,
                    0,
                    cell
                ),
                "generation zero"
            );
        }
        const std::array<retention::CompactSummaryMetadataRetentionV1, kCells>
            generation_zero_snapshot = {retained[0], retained[1]};

        HostState generation_one = make_generation_state(1);
        device.upload_state(generation_one);
        launch(device.inputs(1, pristine_initial), output);
        retained = output.download();
        for (std::int64_t cell = 0; cell < kCells; ++cell) {
            require_bytes_equal(
                retained[cell],
                generation_zero_snapshot[cell],
                "generation zero isolation"
            );
            require_bytes_equal(
                retained[kCells + cell],
                expected_row(
                    generation_one,
                    raw_digests,
                    pristine_initial,
                    1,
                    cell
                ),
                "generation one"
            );
        }
        if (std::memcmp(&retained[0], &retained[1], sizeof(retained[0])) == 0) {
            throw std::runtime_error("cell rows are not isolated");
        }

        const auto pcg_after = device.pcg_state.download();
        const auto mt_after = device.python_words.download();
        const auto bookkeeping_after = device.runtime_bookkeeping.download();
        const auto policy_after = device.policy_energy.download();
        require_vectors_equal_bytes(
            pcg_after.data(),
            generation_one.pcg_state.data(),
            pcg_after.size() * sizeof(pcg_after[0]),
            "PCG state"
        );
        require_vectors_equal_bytes(
            mt_after.data(),
            generation_one.python_words.data(),
            mt_after.size() * sizeof(mt_after[0]),
            "MT19937 state"
        );
        require_vectors_equal_bytes(
            bookkeeping_after.data(),
            generation_one.runtime_bookkeeping.data(),
            bookkeeping_after.size() * sizeof(bookkeeping_after[0]),
            "runtime bookkeeping"
        );
        require_vectors_equal_bytes(
            policy_after.data(),
            generation_one.policy_energy.data(),
            policy_after.size() * sizeof(policy_after[0]),
            "policy energy authority"
        );

        DeviceBuffer<retention::CompactSummaryMetadataRetentionV1>
            continuation_output{kGenerations * kCells};
        check_cuda(
            cudaMemset(
                continuation_output.data(),
                0,
                continuation_output.size()
                    * sizeof(retention::CompactSummaryMetadataRetentionV1)
            ),
            "initialize continuation output"
        );
        device.upload_state(generation_zero);
        launch(device.inputs(0, continuation_initial), continuation_output);
        auto continued = continuation_output.download();
        for (std::int64_t cell = 0; cell < kCells; ++cell) {
            require_bytes_equal(
                continued[cell],
                expected_row(
                    generation_zero,
                    raw_digests,
                    continuation_initial,
                    0,
                    cell
                ),
                "checkpoint continuation generation zero"
            );
        }
        const std::array<retention::CompactSummaryMetadataRetentionV1, kCells>
            continued_generation_zero_snapshot = {continued[0], continued[1]};
        device.upload_state(generation_one);
        launch(device.inputs(1, continuation_initial), continuation_output);
        continued = continuation_output.download();
        for (std::int64_t cell = 0; cell < kCells; ++cell) {
            require_bytes_equal(
                continued[cell],
                continued_generation_zero_snapshot[cell],
                "checkpoint continuation generation zero isolation"
            );
            require_bytes_equal(
                continued[kCells + cell],
                expected_row(
                    generation_one,
                    raw_digests,
                    continuation_initial,
                    1,
                    cell
                ),
                "checkpoint continuation generation one"
            );
        }

        generation_one.upstream[1] = 1;
        i64(generation_one, kActiveCount, 0) = kCapacity + 1;
        device.upload_state(generation_one);
        launch(device.inputs(2, pristine_initial), output);
        retained = output.download();
        require_failed_row(
            retained[2 * kCells],
            retention::kRetentionInvalidActiveCountV1,
            2,
            0
        );

        i64(generation_one, kActiveCount, 0) = 2;
        pcg_buffer(generation_one, kSharedEnvPolicy, 0, 0) = 2;
        device.upload_state(generation_one);
        launch(device.inputs(2, pristine_initial), output);
        retained = output.download();
        require_failed_row(
            retained[2 * kCells],
            retention::kRetentionInvalidPcg64BufferV1,
            2,
            0
        );

        pcg_buffer(generation_one, kSharedEnvPolicy, 0, 0) = 1;
        device.upload_state(generation_one);
        descriptor_validity[descriptor_status_offset(2, 0)] = 0U;
        device.descriptor_validity.upload(descriptor_validity);
        launch(device.inputs(2, pristine_initial), output);
        retained = output.download();
        require_failed_row(
            retained[2 * kCells],
            retention::kRetentionInvalidMutationDescriptorV1,
            2,
            0
        );

        descriptor_validity[descriptor_status_offset(2, 0)] = 1U;
        device.descriptor_validity.upload(descriptor_validity);
        generation_one.policy_energy[1] = 9.0;
        device.upload_state(generation_one);
        launch(device.inputs(2, pristine_initial), output);
        retained = output.download();
        require_failed_row(
            retained[2 * kCells],
            retention::kRetentionInvalidPolicyEnergyAuthorityV1,
            2,
            0
        );

        generation_one.policy_energy[1] = from_bits(0x7ff8000000000000ULL);
        device.upload_state(generation_one);
        auto null_input = device.inputs(2, pristine_initial);
        null_input.runtime_bookkeeping_by_tick = nullptr;
        launch(null_input, output);
        retained = output.download();
        require_failed_row(
            retained[2 * kCells],
            retention::kRetentionNullRequiredInputV1,
            2,
            0
        );

        auto backward_lifecycle = device.inputs(2, pristine_initial);
        backward_lifecycle.initial_lifecycle_event_cursor =
            i64(generation_one, kEventCursor, 0) + 1;
        launch(backward_lifecycle, output);
        retained = output.download();
        require_failed_row(
            retained[2 * kCells],
            retention::kRetentionInvalidRuntimeEventCursorV1,
            2,
            0
        );

        runtime_bookkeeping(generation_one, 0, 2, 1, 2) = -1;
        device.upload_state(generation_one);
        launch(device.inputs(2, pristine_initial), output);
        retained = output.download();
        require_failed_row(
            retained[2 * kCells],
            retention::kRetentionInvalidBookkeepingCounterV1,
            2,
            0
        );
        runtime_bookkeeping(generation_one, 0, 2, 1, 2) = 3;
        device.upload_state(generation_one);

        auto runtime_overflow = device.inputs(2, pristine_initial);
        runtime_overflow.initial_runtime_event_cursor =
            std::numeric_limits<std::int64_t>::max();
        launch(runtime_overflow, output);
        retained = output.download();
        require_failed_row(
            retained[2 * kCells],
            retention::kRetentionCounterOverflowV1,
            2,
            0
        );

        auto generation_overflow = device.inputs(2, pristine_initial);
        generation_overflow.initial_generation_calls =
            std::numeric_limits<std::int64_t>::max();
        launch(generation_overflow, output);
        retained = output.download();
        require_failed_row(
            retained[2 * kCells],
            retention::kRetentionCounterOverflowV1,
            2,
            0
        );

        auto policy_overflow = device.inputs(2, pristine_initial);
        policy_overflow.initial_policy_py_semantic_draws =
            std::numeric_limits<std::int64_t>::max() - 1;
        launch(policy_overflow, output);
        retained = output.download();
        require_failed_row(
            retained[2 * kCells],
            retention::kRetentionCounterOverflowV1,
            2,
            0
        );

        std::cout
            << "compact_summary_metadata_retention_v1: exact_rows=8 "
            << "checkpoint_continuation=true g0_g1_bookkeeping=true "
            << "generation_isolation=true cell_isolation=true "
            << "input_mutation=false failure_cases=10\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
