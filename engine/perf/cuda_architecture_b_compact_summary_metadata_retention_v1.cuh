#ifndef EIEL_CUDA_ARCHITECTURE_B_COMPACT_SUMMARY_METADATA_RETENTION_V1_CUH_
#define EIEL_CUDA_ARCHITECTURE_B_COMPACT_SUMMARY_METADATA_RETENTION_V1_CUH_

#if defined(__CUDACC__)
#include <cuda_runtime.h>
#endif

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace eiel::cuda_architecture_b::compact_summary_metadata_retention_v1 {

inline constexpr char kContractVersionV1[] =
    "eiel_cuda_architecture_b_compact_summary_metadata_retention_v1";
inline constexpr std::uint32_t kSourceVersionV1 = 2U;
inline constexpr std::int32_t kPcg64StateWordsV1 = 4;
inline constexpr std::int32_t kPcg64BufferWordsV1 = 2;
inline constexpr std::int32_t kPcg64AuthorityCountV1 = 4;
inline constexpr std::int32_t kPythonMt19937WordsV1 = 624;
inline constexpr std::int32_t kSemanticStreamCounterCountV1 = 6;
inline constexpr std::int32_t kRuntimeBookkeepingColumnCountV1 = 5;
inline constexpr std::int32_t kDigestBytesV1 = 32;
inline constexpr std::int32_t kFlatDescriptorCountV1 = 132;
inline constexpr std::int32_t kPatchP2DescriptorCountV1 = 144;
inline constexpr std::int32_t kMutationTorchCpuDescriptorIndexV1 = 109;
inline constexpr std::int32_t kMutationTorchCpuRawOutputIndexV1 = 109;
inline constexpr std::int32_t kPatchP2MutationTorchCpuDescriptorIndexV1 = 121;
inline constexpr std::int32_t kPatchP2MutationTorchCpuRawOutputIndexV1 = 121;

#if defined(__CUDACC__)
__host__ __device__
#endif
inline constexpr std::int32_t mutation_torch_cpu_descriptor_index_v1(
    const std::int64_t descriptor_count
) noexcept {
    return descriptor_count == kFlatDescriptorCountV1
        ? kMutationTorchCpuDescriptorIndexV1
        : (
              descriptor_count == kPatchP2DescriptorCountV1
                  ? kPatchP2MutationTorchCpuDescriptorIndexV1
                  : -1
          );
}

#if defined(__CUDACC__)
__host__ __device__
#endif
inline constexpr std::int32_t mutation_torch_cpu_raw_output_index_v1(
    const std::int64_t descriptor_count
) noexcept {
    return descriptor_count == kFlatDescriptorCountV1
        ? kMutationTorchCpuRawOutputIndexV1
        : (
              descriptor_count == kPatchP2DescriptorCountV1
                  ? kPatchP2MutationTorchCpuRawOutputIndexV1
                  : -1
          );
}
inline constexpr std::uint32_t kDtypeUint8V1 = 1U;
inline constexpr std::uint32_t kRankOneV1 = 1U;

enum RetentionStatusV1 : std::int32_t {
    kRetentionSuccessV1 = 0,
    kRetentionInvalidLaunchShapeV1 = 1,
    kRetentionNullRequiredInputV1 = 2,
    kRetentionUpstreamFailureV1 = 3,
    kRetentionInvalidBoundaryCoordinateV1 = 4,
    kRetentionInvalidCapacityV1 = 5,
    kRetentionInvalidActiveCountV1 = 6,
    kRetentionInvalidIdentityPrefixV1 = 7,
    kRetentionInvalidFreeQueueCountV1 = 8,
    kRetentionPendingFreeStateV1 = 9,
    kRetentionInvalidEventCursorV1 = 10,
    kRetentionInvalidRuntimeEventCursorV1 = 11,
    kRetentionInvalidPopulationNextIdV1 = 12,
    kRetentionInvalidPolicyTapeCursorV1 = 13,
    kRetentionInvalidBookkeepingCounterV1 = 14,
    kRetentionInvalidPcg64BufferV1 = 15,
    kRetentionInvalidPythonMt19937StateV1 = 16,
    kRetentionInvalidSemanticCounterV1 = 17,
    kRetentionInvalidMutationDescriptorV1 = 18,
    kRetentionInvalidPolicyEnergyAuthorityV1 = 19,
    kRetentionAddressOverflowV1 = 20,
    kRetentionEnvPolicyAliasRequiredV1 = 21,
    kRetentionCounterOverflowV1 = 22,
};

enum SemanticStreamCounterV1 : std::int32_t {
    kSemanticMutationPyV1 = 0,
    kSemanticMutationNumpyV1,
    kSemanticMutationTorchCpuV1,
    kSemanticPolicyNumpyV1,
    kSemanticPolicyPyV1,
    kSemanticMetricsNumpyV1,
};
static_assert(kSemanticMetricsNumpyV1 + 1 == kSemanticStreamCounterCountV1);

enum RetentionFlagsV1 : std::uint32_t {
    kRetentionFlatWorldV1 = 1U << 0U,
    kRetentionGenerationBoundaryV1 = 1U << 1U,
    kRetentionPolicyEnergyLiveFiniteV1 = 1U << 2U,
    kRetentionMutationDescriptorLinkedV1 = 1U << 3U,
    kRetentionExportNonConsumingV1 = 1U << 4U,
};
inline constexpr std::uint32_t kRequiredRetentionFlagsV1 =
    kRetentionFlatWorldV1
    | kRetentionGenerationBoundaryV1
    | kRetentionPolicyEnergyLiveFiniteV1
    | kRetentionMutationDescriptorLinkedV1
    | kRetentionExportNonConsumingV1;

struct Pcg64RetentionV1 {
    std::uint64_t state_words[kPcg64StateWordsV1];
    std::int64_t buffer_words[kPcg64BufferWordsV1];
};

// All pointers address already-resident, cell-major state. Runtime bookkeeping
// is [cell_count, generation_count * configured_steps_per_generation, 5]. The
// checkpoint digest planes are generation-major, then cell-major, then
// descriptor-major. Initial counters and restore_count are pre-dispatch scalar
// authority shared by the resident cells.
struct MetadataRetentionInputsV1 {
    const std::int32_t* upstream_status;
    const std::int64_t* generation;
    const std::int64_t* tick;
    const std::int64_t* active_count;
    // Source version 2 retains this ABI slot as the compact active-pair count.
    const std::int64_t* agent_id_to_slot_prefix_length;
    const std::int64_t* slot_free_queue_count;
    const std::int64_t* pending_free_count;
    const std::int64_t* event_cursor;
    const std::int64_t* runtime_bookkeeping_by_tick;
    const std::int64_t* population_next_id;
    const std::int64_t* policy_birth_tape_cursor;

    const std::uint64_t* shared_env_policy_pcg64_state_words;
    const std::int64_t* shared_env_policy_pcg64_buffer_words;
    const std::uint64_t* runner_pcg64_state_words;
    const std::int64_t* runner_pcg64_buffer_words;
    const std::uint64_t* population_pcg64_state_words;
    const std::int64_t* population_pcg64_buffer_words;
    const std::uint64_t* birth_spawn_pcg64_state_words;
    const std::int64_t* birth_spawn_pcg64_buffer_words;

    const std::uint32_t* python_random_words;
    const std::int64_t* python_random_index;
    const bool* python_random_has_gauss;
    const double* python_random_gauss_next;

    const std::int64_t* mutation_py_semantic_draws;
    const std::int64_t* mutation_np_semantic_draws;
    const std::int64_t* mutation_torch_cpu_semantic_draws;
    const std::int64_t* policy_np_semantic_draws;
    const std::int64_t* metrics_np_semantic_draws;

    const std::uint8_t* checkpoint_raw_digests;
    const std::int32_t* checkpoint_descriptor_status;
    const std::uint8_t* checkpoint_descriptor_validity;

    const double* policy_energy_observation;
    const bool* alive_mask;
    const std::int64_t* slot_to_agent_id;

    std::int64_t initial_runtime_event_cursor;
    std::int64_t initial_lifecycle_event_cursor;
    std::int64_t initial_generation_calls;
    std::int64_t restore_count;
    std::int64_t initial_policy_py_semantic_draws;
    std::int64_t generation_index;
    std::int64_t generation_count;
    std::int64_t cell_count;
    std::int64_t configured_steps_per_generation;
    std::int64_t capacity;
    std::int64_t agent_id_capacity;
    std::int64_t event_capacity;
    std::int64_t policy_birth_tape_capacity;
    std::int64_t mutation_torch_cpu_state_bytes;
    std::int64_t checkpoint_output_field_count;
    std::int64_t checkpoint_descriptor_count;
    std::uint8_t env_policy_alias;
    std::uint8_t reserved[7];
};

// Fixed generation-major [G,C] retained metadata. Validity is published last.
struct alignas(8) CompactSummaryMetadataRetentionV1 {
    std::uint32_t source_version;
    std::int32_t status;
    std::uint8_t validity;
    std::uint8_t env_policy_alias;
    std::uint8_t python_random_has_gauss;
    std::uint8_t mutation_torch_descriptor_validity;
    std::uint32_t flags;

    std::int64_t generation_index;
    std::int64_t cell_index;
    std::int64_t generation;
    std::int64_t tick;
    std::int64_t configured_steps_per_generation;
    std::int64_t active_count;
    std::int64_t capacity;
    // Source version 2 retains this ABI slot as the compact active-pair count.
    std::int64_t agent_id_to_slot_prefix_length;
    std::int64_t agent_id_capacity;
    std::int64_t slot_free_queue_count;
    std::int64_t event_cursor;
    std::int64_t event_capacity;
    std::int64_t runtime_event_cursor;
    std::int64_t population_next_id;
    std::int64_t policy_birth_tape_cursor;
    std::int64_t policy_birth_tape_capacity;
    std::int64_t generation_calls;
    std::int64_t restore_count;
    std::int64_t last_applied_actions;
    std::int64_t last_moved;
    std::int64_t last_gathered;
    std::int64_t last_policy_refill_calls;
    std::int64_t last_policy_refill_rows;
    std::int64_t mutation_torch_cpu_state_bytes;
    std::int64_t policy_energy_observation_capacity;
    std::int64_t policy_energy_live_count;
    std::int64_t policy_energy_finite_live_count;
    std::int64_t native_runtime_rng_state;

    Pcg64RetentionV1 pcg64[kPcg64AuthorityCountV1];
    std::uint32_t python_random_words[kPythonMt19937WordsV1];
    std::int64_t python_random_index;
    double python_random_gauss_next;
    std::int64_t semantic_stream_counters[kSemanticStreamCounterCountV1];

    std::uint32_t mutation_torch_descriptor_index;
    std::uint32_t mutation_torch_raw_output_index;
    std::uint32_t mutation_torch_dtype_code;
    std::uint32_t mutation_torch_rank;
    std::uint8_t mutation_torch_raw_sha256[kDigestBytesV1];
    std::uint8_t policy_energy_tagged_sha256[kDigestBytesV1];
};

static_assert(sizeof(bool) == 1);
static_assert(sizeof(double) == 8);
static_assert(sizeof(Pcg64RetentionV1) == 48);
static_assert(alignof(Pcg64RetentionV1) == 8);
static_assert(std::is_standard_layout_v<Pcg64RetentionV1>);
static_assert(std::is_trivially_copyable_v<Pcg64RetentionV1>);
static_assert(std::is_standard_layout_v<MetadataRetentionInputsV1>);
static_assert(std::is_trivially_copyable_v<MetadataRetentionInputsV1>);
static_assert(std::is_standard_layout_v<CompactSummaryMetadataRetentionV1>);
static_assert(std::is_trivially_copyable_v<CompactSummaryMetadataRetentionV1>);
static_assert(alignof(CompactSummaryMetadataRetentionV1) == 8);
static_assert(sizeof(CompactSummaryMetadataRetentionV1) == 3072);
static_assert(offsetof(CompactSummaryMetadataRetentionV1, source_version) == 0);
static_assert(offsetof(CompactSummaryMetadataRetentionV1, status) == 4);
static_assert(offsetof(CompactSummaryMetadataRetentionV1, validity) == 8);
static_assert(offsetof(CompactSummaryMetadataRetentionV1, flags) == 12);
static_assert(offsetof(CompactSummaryMetadataRetentionV1, generation_index) == 16);
static_assert(offsetof(CompactSummaryMetadataRetentionV1, generation) == 32);
static_assert(offsetof(CompactSummaryMetadataRetentionV1, active_count) == 56);
static_assert(offsetof(CompactSummaryMetadataRetentionV1, pcg64) == 240);
static_assert(offsetof(CompactSummaryMetadataRetentionV1, python_random_words) == 432);
static_assert(offsetof(CompactSummaryMetadataRetentionV1, python_random_index) == 2928);
static_assert(offsetof(CompactSummaryMetadataRetentionV1, semantic_stream_counters) == 2944);
static_assert(offsetof(CompactSummaryMetadataRetentionV1, mutation_torch_descriptor_index) == 2992);
static_assert(offsetof(CompactSummaryMetadataRetentionV1, mutation_torch_raw_sha256) == 3008);
static_assert(offsetof(CompactSummaryMetadataRetentionV1, policy_energy_tagged_sha256) == 3040);

#if defined(__CUDACC__)
__global__ void retain_compact_summary_metadata_v1(
    MetadataRetentionInputsV1 inputs,
    CompactSummaryMetadataRetentionV1* output
);
#endif

// Keep this contract usable by strict host-only ABI tests. The CUDA translation
// unit owns the stream cast and kernel launch.
void launch_compact_summary_metadata_retention_v1(
    MetadataRetentionInputsV1 inputs,
    CompactSummaryMetadataRetentionV1* output,
    void* stream_handle
);

}  // namespace eiel::cuda_architecture_b::compact_summary_metadata_retention_v1

#endif
