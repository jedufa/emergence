#include "cuda_architecture_b_compact_summary_metadata_retention_v1.cuh"

#include "cuda_architecture_b_sha256_v1.cuh"

#include <cmath>
#include <cstdint>
#include <limits>

namespace eiel::cuda_architecture_b::compact_summary_metadata_retention_v1 {

namespace {

namespace sha256 = eiel::cuda_architecture_b::sha256_v1;

inline constexpr std::int32_t kPcgSharedEnvPolicyV1 = 0;
inline constexpr std::int32_t kPcgRunnerV1 = 1;
inline constexpr std::int32_t kPcgPopulationV1 = 2;
inline constexpr std::int32_t kPcgBirthSpawnV1 = 3;
inline constexpr std::int64_t kMaximumInt64V1 =
    std::numeric_limits<std::int64_t>::max();

static __device__ __constant__ const std::uint8_t
    kPolicyEnergyTagPrefixV1[] = {
        '{', '"', 'd', 't', 'y', 'p', 'e', '"', ':', '"',
        't', 'o', 'r', 'c', 'h', '.', 'f', 'l', 'o', 'a', 't', '6', '4',
        '"', ',', '"', 's', 'h', 'a', 'p', 'e', '"', ':', '[',
    };
static __device__ __constant__ const std::uint8_t
    kPolicyEnergyTagSuffixV1[] = {']', '}', 0U};

__device__ __forceinline__ bool multiply_nonnegative_exact(
    const std::int64_t left,
    const std::int64_t right,
    std::int64_t* const product
) noexcept {
    if (left < 0 || right < 0 || (left != 0 && right > kMaximumInt64V1 / left)) {
        return false;
    }
    *product = left * right;
    return true;
}

__device__ __forceinline__ bool add_nonnegative_exact(
    const std::int64_t left,
    const std::int64_t right,
    std::int64_t* const sum
) noexcept {
    if (left < 0 || right < 0 || right > kMaximumInt64V1 - left) {
        return false;
    }
    *sum = left + right;
    return true;
}

__device__ __forceinline__ bool required_pointers_present(
    const MetadataRetentionInputsV1& inputs
) noexcept {
    return inputs.upstream_status != nullptr
        && inputs.generation != nullptr
        && inputs.tick != nullptr
        && inputs.active_count != nullptr
        && inputs.agent_id_to_slot_prefix_length != nullptr
        && inputs.slot_free_queue_count != nullptr
        && inputs.pending_free_count != nullptr
        && inputs.event_cursor != nullptr
        && inputs.runtime_bookkeeping_by_tick != nullptr
        && inputs.population_next_id != nullptr
        && inputs.policy_birth_tape_cursor != nullptr
        && inputs.shared_env_policy_pcg64_state_words != nullptr
        && inputs.shared_env_policy_pcg64_buffer_words != nullptr
        && inputs.runner_pcg64_state_words != nullptr
        && inputs.runner_pcg64_buffer_words != nullptr
        && inputs.population_pcg64_state_words != nullptr
        && inputs.population_pcg64_buffer_words != nullptr
        && inputs.birth_spawn_pcg64_state_words != nullptr
        && inputs.birth_spawn_pcg64_buffer_words != nullptr
        && inputs.python_random_words != nullptr
        && inputs.python_random_index != nullptr
        && inputs.python_random_has_gauss != nullptr
        && inputs.python_random_gauss_next != nullptr
        && inputs.mutation_py_semantic_draws != nullptr
        && inputs.mutation_np_semantic_draws != nullptr
        && inputs.mutation_torch_cpu_semantic_draws != nullptr
        && inputs.policy_np_semantic_draws != nullptr
        && inputs.metrics_np_semantic_draws != nullptr
        && inputs.checkpoint_raw_digests != nullptr
        && inputs.checkpoint_descriptor_status != nullptr
        && inputs.checkpoint_descriptor_validity != nullptr
        && inputs.policy_energy_observation != nullptr
        && inputs.alive_mask != nullptr
        && inputs.slot_to_agent_id != nullptr;
}

__device__ __forceinline__ void clear_row(
    CompactSummaryMetadataRetentionV1* const row
) noexcept {
    std::uint8_t* const bytes = reinterpret_cast<std::uint8_t*>(row);
    for (std::int32_t index = 0;
         index < static_cast<std::int32_t>(sizeof(*row));
         ++index) {
        bytes[index] = 0U;
    }
}

__device__ __forceinline__ void fail_row(
    CompactSummaryMetadataRetentionV1* const row,
    const RetentionStatusV1 status
) noexcept {
    row->status = static_cast<std::int32_t>(status);
    __threadfence();
    row->validity = 0U;
}

__device__ __forceinline__ bool pcg64_buffer_valid(
    const std::int64_t* const words,
    const std::int64_t cell
) noexcept {
    const std::int64_t offset = cell * kPcg64BufferWordsV1;
    return (words[offset] == 0 || words[offset] == 1)
        && words[offset + 1] >= 0
        && static_cast<std::uint64_t>(words[offset + 1]) <= 0xffffffffULL;
}

__device__ __forceinline__ void copy_pcg64(
    const std::uint64_t* const state,
    const std::int64_t* const buffer,
    const std::int64_t cell,
    Pcg64RetentionV1* const output
) noexcept {
    const std::int64_t state_offset = cell * kPcg64StateWordsV1;
    const std::int64_t buffer_offset = cell * kPcg64BufferWordsV1;
#pragma unroll
    for (std::int32_t word = 0; word < kPcg64StateWordsV1; ++word) {
        output->state_words[word] = state[state_offset + word];
    }
#pragma unroll
    for (std::int32_t word = 0; word < kPcg64BufferWordsV1; ++word) {
        output->buffer_words[word] = buffer[buffer_offset + word];
    }
}

__device__ __forceinline__ void policy_energy_tagged_digest(
    const double* const values,
    const std::int64_t capacity,
    std::uint8_t output[kDigestBytesV1]
) noexcept {
    sha256::Sha256ContextV1 context;
    sha256::sha256_init_v1(&context);
    sha256::sha256_update_v1(
        &context,
        kPolicyEnergyTagPrefixV1,
        sizeof(kPolicyEnergyTagPrefixV1)
    );

    std::uint8_t reverse_digits[20];
    std::int32_t digit_count = 0;
    std::uint64_t remaining = static_cast<std::uint64_t>(capacity);
    do {
        reverse_digits[digit_count++] = static_cast<std::uint8_t>(
            '0' + static_cast<std::uint8_t>(remaining % 10U)
        );
        remaining /= 10U;
    } while (remaining != 0U);
    for (std::int32_t index = digit_count - 1; index >= 0; --index) {
        sha256::sha256_update_v1(&context, reverse_digits + index, 1U);
    }
    sha256::sha256_update_v1(
        &context,
        kPolicyEnergyTagSuffixV1,
        sizeof(kPolicyEnergyTagSuffixV1)
    );
    sha256::sha256_update_v1(
        &context,
        reinterpret_cast<const std::uint8_t*>(values),
        static_cast<std::uint64_t>(capacity) * sizeof(double)
    );
    sha256::sha256_final_v1(&context, output);
}

}  // namespace

__global__ void retain_compact_summary_metadata_v1(
    const MetadataRetentionInputsV1 inputs,
    CompactSummaryMetadataRetentionV1* const output
) {
    const std::int64_t cell =
        static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (output == nullptr || cell < 0 || cell >= inputs.cell_count) {
        return;
    }
    if (
        inputs.cell_count <= 0
        || inputs.generation_count <= 0
        || inputs.generation_index < 0
        || inputs.generation_index >= inputs.generation_count
        || inputs.generation_count > kMaximumInt64V1 / inputs.cell_count
    ) {
        return;
    }

    const std::int64_t row_index =
        inputs.generation_index * inputs.cell_count + cell;
    CompactSummaryMetadataRetentionV1* const row = output + row_index;
    clear_row(row);
    row->source_version = kSourceVersionV1;
    row->generation_index = inputs.generation_index;
    row->cell_index = cell;

    if (!required_pointers_present(inputs)) {
        fail_row(row, kRetentionNullRequiredInputV1);
        return;
    }
    if (inputs.upstream_status[cell] != 0) {
        fail_row(row, kRetentionUpstreamFailureV1);
        return;
    }
    if (inputs.env_policy_alias != 1U) {
        fail_row(row, kRetentionEnvPolicyAliasRequiredV1);
        return;
    }
    if (
        inputs.configured_steps_per_generation <= 0
        || inputs.generation[cell] < 0
        || inputs.tick[cell] != 0
    ) {
        fail_row(row, kRetentionInvalidBoundaryCoordinateV1);
        return;
    }
    const std::int64_t mutation_descriptor_index =
        mutation_torch_cpu_descriptor_index_v1(
            inputs.checkpoint_descriptor_count
        );
    const std::int64_t mutation_raw_output_index =
        mutation_torch_cpu_raw_output_index_v1(
            inputs.checkpoint_descriptor_count
        );
    if (
        inputs.capacity <= 0
        || inputs.agent_id_capacity <= 0
        || inputs.event_capacity <= 0
        || inputs.policy_birth_tape_capacity < 0
        || inputs.mutation_torch_cpu_state_bytes <= 0
        || mutation_descriptor_index < 0
        || mutation_raw_output_index < 0
        || inputs.checkpoint_output_field_count
            <= mutation_raw_output_index
        || inputs.checkpoint_descriptor_count
            <= mutation_descriptor_index
    ) {
        fail_row(row, kRetentionInvalidCapacityV1);
        return;
    }

    std::int64_t policy_offset = 0;
    std::int64_t python_offset = 0;
    std::int64_t descriptor_rows = 0;
    std::int64_t descriptor_status_offset = 0;
    std::int64_t digest_fields = 0;
    std::int64_t digest_field_offset = 0;
    std::int64_t ticks_per_cell = 0;
    std::int64_t generation_tick_start = 0;
    std::int64_t generation_tick_end = 0;
    std::int64_t cell_tick_offset = 0;
    std::int64_t bookkeeping_tick_end = 0;
    if (
        !multiply_nonnegative_exact(cell, inputs.capacity, &policy_offset)
        || !multiply_nonnegative_exact(
            cell,
            static_cast<std::int64_t>(kPythonMt19937WordsV1),
            &python_offset
        )
        || !multiply_nonnegative_exact(
            row_index,
            inputs.checkpoint_descriptor_count,
            &descriptor_rows
        )
        || descriptor_rows > kMaximumInt64V1 - mutation_descriptor_index
        || !multiply_nonnegative_exact(
            row_index,
            inputs.checkpoint_output_field_count,
            &digest_fields
        )
        || digest_fields > kMaximumInt64V1 - mutation_raw_output_index
        || !multiply_nonnegative_exact(
            inputs.generation_count,
            inputs.configured_steps_per_generation,
            &ticks_per_cell
        )
        || !multiply_nonnegative_exact(
            inputs.generation_index,
            inputs.configured_steps_per_generation,
            &generation_tick_start
        )
        || !add_nonnegative_exact(
            generation_tick_start,
            inputs.configured_steps_per_generation,
            &generation_tick_end
        )
        || generation_tick_end > ticks_per_cell
        || !multiply_nonnegative_exact(cell, ticks_per_cell, &cell_tick_offset)
        || !add_nonnegative_exact(
            cell_tick_offset,
            generation_tick_end,
            &bookkeeping_tick_end
        )
        || bookkeeping_tick_end
            > kMaximumInt64V1 / kRuntimeBookkeepingColumnCountV1
    ) {
        fail_row(row, kRetentionAddressOverflowV1);
        return;
    }
    descriptor_status_offset =
        descriptor_rows + mutation_descriptor_index;
    digest_field_offset =
        digest_fields + mutation_raw_output_index;
    if (digest_field_offset > kMaximumInt64V1 / kDigestBytesV1) {
        fail_row(row, kRetentionAddressOverflowV1);
        return;
    }
    const std::int64_t digest_byte_offset =
        digest_field_offset * kDigestBytesV1;

    const std::int64_t active_count = inputs.active_count[cell];
    if (active_count < 0 || active_count > inputs.capacity) {
        fail_row(row, kRetentionInvalidActiveCountV1);
        return;
    }
    const std::int64_t identity_prefix =
        inputs.agent_id_to_slot_prefix_length[cell];
    if (identity_prefix < 1 || identity_prefix > inputs.agent_id_capacity) {
        fail_row(row, kRetentionInvalidIdentityPrefixV1);
        return;
    }
    const std::int64_t population_next_id = inputs.population_next_id[cell];
    if (population_next_id < 1 || population_next_id < identity_prefix) {
        fail_row(row, kRetentionInvalidPopulationNextIdV1);
        return;
    }
    const std::int64_t free_count = inputs.slot_free_queue_count[cell];
    if (free_count < 0 || free_count > inputs.capacity
        || free_count != inputs.capacity - active_count) {
        fail_row(row, kRetentionInvalidFreeQueueCountV1);
        return;
    }
    if (inputs.pending_free_count[cell] != 0) {
        fail_row(row, kRetentionPendingFreeStateV1);
        return;
    }
    const std::int64_t event_cursor = inputs.event_cursor[cell];
    if (event_cursor < 0 || event_cursor > inputs.event_capacity) {
        fail_row(row, kRetentionInvalidEventCursorV1);
        return;
    }
    if (
        inputs.initial_runtime_event_cursor < 0
        || inputs.initial_lifecycle_event_cursor < 0
        || inputs.initial_runtime_event_cursor
            < inputs.initial_lifecycle_event_cursor
        || event_cursor < inputs.initial_lifecycle_event_cursor
    ) {
        fail_row(row, kRetentionInvalidRuntimeEventCursorV1);
        return;
    }
    const std::int64_t policy_cursor = inputs.policy_birth_tape_cursor[cell];
    if (policy_cursor < 0 || policy_cursor > inputs.policy_birth_tape_capacity) {
        fail_row(row, kRetentionInvalidPolicyTapeCursorV1);
        return;
    }

    if (inputs.initial_generation_calls < 0 || inputs.restore_count < 0) {
        fail_row(row, kRetentionInvalidBookkeepingCounterV1);
        return;
    }
    if (inputs.initial_policy_py_semantic_draws < 0) {
        fail_row(row, kRetentionInvalidSemanticCounterV1);
        return;
    }

    std::int64_t last_bookkeeping[kRuntimeBookkeepingColumnCountV1] = {};
    std::int64_t cumulative_applied_actions = 0;
    for (std::int64_t tick_offset = 0;
         tick_offset < generation_tick_end;
         ++tick_offset) {
        const std::int64_t bookkeeping_offset =
            (cell_tick_offset + tick_offset)
            * kRuntimeBookkeepingColumnCountV1;
        const std::int64_t applied_actions =
            inputs.runtime_bookkeeping_by_tick[bookkeeping_offset];
        if (
            applied_actions < 0
            || !add_nonnegative_exact(
                cumulative_applied_actions,
                applied_actions,
                &cumulative_applied_actions
            )
        ) {
            fail_row(
                row,
                applied_actions < 0
                    ? kRetentionInvalidBookkeepingCounterV1
                    : kRetentionCounterOverflowV1
            );
            return;
        }
        if (tick_offset < generation_tick_start) {
            continue;
        }
#pragma unroll
        for (std::int32_t column = 0;
             column < kRuntimeBookkeepingColumnCountV1;
             ++column) {
            const std::int64_t value =
                inputs.runtime_bookkeeping_by_tick[bookkeeping_offset + column];
            if (value < 0) {
                fail_row(row, kRetentionInvalidBookkeepingCounterV1);
                return;
            }
            last_bookkeeping[column] = value;
        }
    }

    std::int64_t completed_generation_count = 0;
    std::int64_t generation_call_increment = 0;
    std::int64_t generation_calls = 0;
    if (
        !add_nonnegative_exact(
            inputs.generation_index,
            1,
            &completed_generation_count
        )
        || !multiply_nonnegative_exact(
            completed_generation_count,
            inputs.configured_steps_per_generation,
            &generation_call_increment
        )
        || !add_nonnegative_exact(
            inputs.initial_generation_calls,
            generation_call_increment,
            &generation_calls
        )
    ) {
        fail_row(row, kRetentionCounterOverflowV1);
        return;
    }

    std::int64_t lifecycle_event_additions = 0;
    std::int64_t runtime_event_cursor = 0;
    std::int64_t runtime_with_actions = 0;
    lifecycle_event_additions =
        event_cursor - inputs.initial_lifecycle_event_cursor;
    if (
        !add_nonnegative_exact(
            inputs.initial_runtime_event_cursor,
            cumulative_applied_actions,
            &runtime_with_actions
        )
        || !add_nonnegative_exact(
            runtime_with_actions,
            lifecycle_event_additions,
            &runtime_event_cursor
        )
    ) {
        fail_row(row, kRetentionCounterOverflowV1);
        return;
    }

    std::int64_t policy_py_generation_calls = inputs.generation_index;
    std::int64_t policy_py_increment = 0;
    std::int64_t policy_py_semantic_draws = 0;
    if (
        (
            inputs.initial_generation_calls > 0
            && !add_nonnegative_exact(
                policy_py_generation_calls,
                1,
                &policy_py_generation_calls
            )
        )
        || !multiply_nonnegative_exact(
            policy_py_generation_calls,
            8,
            &policy_py_increment
        )
        || !add_nonnegative_exact(
            inputs.initial_policy_py_semantic_draws,
            policy_py_increment,
            &policy_py_semantic_draws
        )
    ) {
        fail_row(row, kRetentionCounterOverflowV1);
        return;
    }

    if (runtime_event_cursor < event_cursor || generation_calls <= 0) {
        fail_row(row, kRetentionCounterOverflowV1);
        return;
    }

    if (last_bookkeeping[0] > cumulative_applied_actions) {
        fail_row(row, kRetentionInvalidBookkeepingCounterV1);
        return;
    }

    for (std::int32_t index = 0;
         index < kRuntimeBookkeepingColumnCountV1;
         ++index) {
        if (last_bookkeeping[index] < 0) {
            fail_row(row, kRetentionInvalidBookkeepingCounterV1);
            return;
        }
    }

    if (
        !pcg64_buffer_valid(inputs.shared_env_policy_pcg64_buffer_words, cell)
        || !pcg64_buffer_valid(inputs.runner_pcg64_buffer_words, cell)
        || !pcg64_buffer_valid(inputs.population_pcg64_buffer_words, cell)
        || !pcg64_buffer_valid(inputs.birth_spawn_pcg64_buffer_words, cell)
    ) {
        fail_row(row, kRetentionInvalidPcg64BufferV1);
        return;
    }
    const std::int64_t python_index = inputs.python_random_index[cell];
    const bool python_has_gauss = inputs.python_random_has_gauss[cell];
    const double python_gauss_next = inputs.python_random_gauss_next[cell];
    if (
        python_index < 0
        || python_index > kPythonMt19937WordsV1
        || (python_has_gauss && !::isfinite(python_gauss_next))
    ) {
        fail_row(row, kRetentionInvalidPythonMt19937StateV1);
        return;
    }

    const std::int64_t semantic_counters[kSemanticStreamCounterCountV1] = {
        inputs.mutation_py_semantic_draws[cell],
        inputs.mutation_np_semantic_draws[cell],
        inputs.mutation_torch_cpu_semantic_draws[cell],
        inputs.policy_np_semantic_draws[cell],
        policy_py_semantic_draws,
        inputs.metrics_np_semantic_draws[cell],
    };
#pragma unroll
    for (std::int32_t index = 0;
         index < kSemanticStreamCounterCountV1;
         ++index) {
        if (semantic_counters[index] < 0) {
            fail_row(row, kRetentionInvalidSemanticCounterV1);
            return;
        }
    }
    if (
        inputs.checkpoint_descriptor_status[descriptor_status_offset] != 0
        || inputs.checkpoint_descriptor_validity[descriptor_status_offset] != 1U
    ) {
        fail_row(row, kRetentionInvalidMutationDescriptorV1);
        return;
    }

    std::int64_t policy_live_count = 0;
    std::int64_t policy_finite_live_count = 0;
    for (std::int64_t slot = 0; slot < inputs.capacity; ++slot) {
        const bool alive = inputs.alive_mask[policy_offset + slot];
        const bool assigned = inputs.slot_to_agent_id[policy_offset + slot] >= 0;
        const bool finite = ::isfinite(
            inputs.policy_energy_observation[policy_offset + slot]
        );
        if (alive != assigned || (alive && !finite) || (!alive && finite)) {
            fail_row(row, kRetentionInvalidPolicyEnergyAuthorityV1);
            return;
        }
        if (alive) {
            ++policy_live_count;
            if (finite) {
                ++policy_finite_live_count;
            }
        }
    }
    if (
        policy_live_count != active_count
        || policy_finite_live_count != active_count
    ) {
        fail_row(row, kRetentionInvalidPolicyEnergyAuthorityV1);
        return;
    }

    std::uint8_t policy_digest[kDigestBytesV1];
    policy_energy_tagged_digest(
        inputs.policy_energy_observation + policy_offset,
        inputs.capacity,
        policy_digest
    );

    row->env_policy_alias = 1U;
    row->python_random_has_gauss = python_has_gauss ? 1U : 0U;
    row->mutation_torch_descriptor_validity = 1U;
    row->flags = kRequiredRetentionFlagsV1;
    row->generation = inputs.generation[cell];
    row->tick = inputs.tick[cell];
    row->configured_steps_per_generation =
        inputs.configured_steps_per_generation;
    row->active_count = active_count;
    row->capacity = inputs.capacity;
    row->agent_id_to_slot_prefix_length = identity_prefix;
    row->agent_id_capacity = inputs.agent_id_capacity;
    row->slot_free_queue_count = free_count;
    row->event_cursor = event_cursor;
    row->event_capacity = inputs.event_capacity;
    row->runtime_event_cursor = runtime_event_cursor;
    row->population_next_id = population_next_id;
    row->policy_birth_tape_cursor = policy_cursor;
    row->policy_birth_tape_capacity = inputs.policy_birth_tape_capacity;
    row->generation_calls = generation_calls;
    row->restore_count = inputs.restore_count;
    row->last_applied_actions = last_bookkeeping[0];
    row->last_moved = last_bookkeeping[1];
    row->last_gathered = last_bookkeeping[2];
    row->last_policy_refill_calls = last_bookkeeping[3];
    row->last_policy_refill_rows = last_bookkeeping[4];
    row->mutation_torch_cpu_state_bytes =
        inputs.mutation_torch_cpu_state_bytes;
    row->policy_energy_observation_capacity = inputs.capacity;
    row->policy_energy_live_count = policy_live_count;
    row->policy_energy_finite_live_count = policy_finite_live_count;
    row->native_runtime_rng_state = static_cast<std::int64_t>(
        inputs.shared_env_policy_pcg64_state_words[
            cell * kPcg64StateWordsV1
        ] & 0x7fffffffffffffffULL
    );

    copy_pcg64(
        inputs.shared_env_policy_pcg64_state_words,
        inputs.shared_env_policy_pcg64_buffer_words,
        cell,
        row->pcg64 + kPcgSharedEnvPolicyV1
    );
    copy_pcg64(
        inputs.runner_pcg64_state_words,
        inputs.runner_pcg64_buffer_words,
        cell,
        row->pcg64 + kPcgRunnerV1
    );
    copy_pcg64(
        inputs.population_pcg64_state_words,
        inputs.population_pcg64_buffer_words,
        cell,
        row->pcg64 + kPcgPopulationV1
    );
    copy_pcg64(
        inputs.birth_spawn_pcg64_state_words,
        inputs.birth_spawn_pcg64_buffer_words,
        cell,
        row->pcg64 + kPcgBirthSpawnV1
    );

    for (std::int32_t word = 0; word < kPythonMt19937WordsV1; ++word) {
        row->python_random_words[word] =
            inputs.python_random_words[python_offset + word];
    }
    row->python_random_index = python_index;
    row->python_random_gauss_next = python_gauss_next;
#pragma unroll
    for (std::int32_t index = 0;
         index < kSemanticStreamCounterCountV1;
         ++index) {
        row->semantic_stream_counters[index] = semantic_counters[index];
    }

    row->mutation_torch_descriptor_index =
        static_cast<std::uint32_t>(mutation_descriptor_index);
    row->mutation_torch_raw_output_index =
        static_cast<std::uint32_t>(mutation_raw_output_index);
    row->mutation_torch_dtype_code = kDtypeUint8V1;
    row->mutation_torch_rank = kRankOneV1;
#pragma unroll
    for (std::int32_t index = 0; index < kDigestBytesV1; ++index) {
        row->mutation_torch_raw_sha256[index] =
            inputs.checkpoint_raw_digests[digest_byte_offset + index];
        row->policy_energy_tagged_sha256[index] = policy_digest[index];
    }

    row->status = kRetentionSuccessV1;
    __threadfence();
    row->validity = 1U;
}

void launch_compact_summary_metadata_retention_v1(
    const MetadataRetentionInputsV1 inputs,
    CompactSummaryMetadataRetentionV1* const output,
    void* const stream_handle
) {
    if (
        output == nullptr
        || inputs.cell_count <= 0
        || inputs.generation_count <= 0
        || inputs.generation_index < 0
        || inputs.generation_index >= inputs.generation_count
    ) {
        return;
    }
    constexpr std::int32_t threads_per_block = 64;
    const unsigned int blocks = static_cast<unsigned int>(
        (inputs.cell_count + threads_per_block - 1) / threads_per_block
    );
    retain_compact_summary_metadata_v1<<<
        blocks,
        threads_per_block,
        0,
        reinterpret_cast<cudaStream_t>(stream_handle)
    >>>(inputs, output);
}

}  // namespace eiel::cuda_architecture_b::compact_summary_metadata_retention_v1
