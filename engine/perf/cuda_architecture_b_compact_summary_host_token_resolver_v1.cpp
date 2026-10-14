#include "cuda_architecture_b_compact_summary_host_token_resolver_v1.hpp"

#include <cmath>
#include <cstddef>
#include <cstring>
#include <limits>

namespace eiel::cuda_architecture_b::compact_summary_host_token_resolver_v1 {
namespace {

namespace renderer = compact_summary_host_template_renderer_v1;

using FailureComponent = HostTokenResolverFailureComponentV1;
using Status = HostTokenResolverStatusV1;

inline constexpr std::uint64_t kDigestBytesV1 = 32U;
inline constexpr std::uint64_t kSha256HexBytesV1 = 64U;
inline constexpr std::uint64_t kSurfaceCountV1 = 6U;

enum CanonicalizationModeV1 : std::int64_t {
    kExactModeV1 = 0,
    kDeadRowZeroModeV1 = 1,
    kActivePrefixModeV1 = 2,
};

enum class PrefixAuthorityV1 : std::uint32_t {
    kNone = 0,
    kActiveCount = 1,
    kEventCursor = 2,
    kPolicyBirthTapeCursor = 3,
    kSlotFreeQueueCount = 4,
};

struct DescriptorPlanV1 {
    std::int64_t descriptor_count;
    std::int64_t identity_descriptor;
    std::int64_t policy_tape_first_descriptor;
    std::int64_t policy_tape_last_descriptor;
    std::int64_t policy_energy_descriptor;
    std::int64_t mutation_torch_cpu_descriptor;
    std::int64_t slot_free_queue_descriptor;
    std::int64_t typed_lifecycle_first_descriptor;
    std::int64_t typed_lifecycle_last_descriptor;
};

inline constexpr DescriptorPlanV1 kFlatDescriptorPlanV1 = {
    checkpoint::kFlatDescriptorCountV1,
    10,
    92,
    96,
    98,
    109,
    112,
    116,
    131,
};

inline constexpr DescriptorPlanV1 kPatchP2DescriptorPlanV1 = {
    checkpoint::kPatchP2DescriptorCountV1,
    10,
    104,
    108,
    110,
    121,
    124,
    128,
    143,
};

const DescriptorPlanV1* descriptor_plan(
    const std::int64_t descriptor_count
) noexcept {
    if (descriptor_count == kFlatDescriptorPlanV1.descriptor_count) {
        return &kFlatDescriptorPlanV1;
    }
    if (descriptor_count == kPatchP2DescriptorPlanV1.descriptor_count) {
        return &kPatchP2DescriptorPlanV1;
    }
    return nullptr;
}

enum SourceKindV1 : std::uint64_t {
    kCheckpointTrajectoryDigestV1 = 1,
    kCheckpointTrajectoryShapeV1,
    kCheckpointCanonicalDigestV1,
    kCheckpointCanonicalShapeV1,
    kCheckpointSourceManifestDigestV1,
    kCheckpointSourceManifestShapeV1,
    kCheckpointStoredManifestShapeV1,
    kCheckpointStoredManifestNumelV1,
    kCheckpointStoredManifestBytesV1,
    kProjectionTopLevelIntegerV1,
    kProjectionMetadataIntegerV1,
    kProjectionMetadataBooleanV1,
    kPcg64StateV1,
    kPcg64IncV1,
    kPcg64BufferFlagV1,
    kPcg64BufferValueV1,
    kPythonMtWordV1,
    kPythonMtIndexV1,
    kSemanticStreamCounterV1,
    kMetadataImportDigestV1,
    kPolicyEnergyDigestV1,
    kPolicyEnergyShapeV1,
    kIdentityCountV1,
    kIdentityDigestV1,
    kPolicyMutationTapeCursorV1,
    kPolicyMutationTapeDigestV1,
    kPolicyMutationTapeShapeV1,
    kPythonGaussCacheV1 = 28,
};

enum SelectorV1 : std::uint64_t {
    kSelectorNoneV1 = 0,
    kSelectorRawV1 = 1,
    kSelectorCanonicalV1 = 2,
};

struct TokenCoordinateV1 {
    std::uint64_t resolver_row;
    std::uint32_t generation;
    std::uint32_t cell;
    std::uint32_t surface;
    std::uint32_t token;
};

struct ResolvedTokenV1 {
    std::uint32_t token_type;
    std::uint32_t shape_rank;
    std::uint64_t scalar_low;
    std::uint64_t scalar_high;
    std::uint64_t shape_dimensions[2];
    const std::uint8_t* digest;
};

HostTokenResolverFailureV1 success_failure() noexcept {
    return {
        static_cast<std::uint32_t>(Status::kSuccess),
        static_cast<std::uint32_t>(FailureComponent::kNone),
        kNoResolverRowV1,
        kNoResolverCoordinateV1,
        kNoResolverCoordinateV1,
        kNoResolverCoordinateV1,
        kNoResolverCoordinateV1,
        kNoDescriptorIndexV1,
    };
}

Status fail(
    HostTokenResolverFailureV1* const failure,
    const Status status,
    const FailureComponent component,
    const TokenCoordinateV1 coordinate = {
        kNoResolverRowV1,
        kNoResolverCoordinateV1,
        kNoResolverCoordinateV1,
        kNoResolverCoordinateV1,
        kNoResolverCoordinateV1,
    },
    const std::int64_t descriptor = kNoDescriptorIndexV1
) noexcept {
    if (failure != nullptr) {
        *failure = {
            static_cast<std::uint32_t>(status),
            static_cast<std::uint32_t>(component),
            coordinate.resolver_row,
            coordinate.generation,
            coordinate.cell,
            coordinate.surface,
            coordinate.token,
            descriptor,
        };
    }
    return status;
}

bool checked_multiply(
    const std::uint64_t left,
    const std::uint64_t right,
    std::uint64_t* const result
) noexcept {
    if (result == nullptr
        || (left != 0U && right > UINT64_MAX / left)) {
        return false;
    }
    *result = left * right;
    return true;
}

bool checked_index_extent(const std::uint64_t value) noexcept {
    return value <= static_cast<std::uint64_t>(
        std::numeric_limits<std::size_t>::max()
    );
}

const std::int64_t* descriptor_spec(
    const HostTokenResolverRequestV1& request,
    const std::int64_t descriptor
) noexcept {
    return request.descriptor_specs
        + static_cast<std::size_t>(descriptor)
            * static_cast<std::size_t>(checkpoint::kDescriptorSpecWidthV1);
}

PrefixAuthorityV1 prefix_authority(
    const DescriptorPlanV1& plan,
    const std::int64_t descriptor
) noexcept {
    if (descriptor >= 8 && descriptor <= plan.identity_descriptor) {
        return PrefixAuthorityV1::kActiveCount;
    }
    if (descriptor >= 39 && descriptor <= 44) {
        return PrefixAuthorityV1::kEventCursor;
    }
    if (descriptor >= plan.policy_tape_first_descriptor
        && descriptor <= plan.policy_tape_last_descriptor) {
        return PrefixAuthorityV1::kPolicyBirthTapeCursor;
    }
    if (descriptor == plan.slot_free_queue_descriptor) {
        return PrefixAuthorityV1::kSlotFreeQueueCount;
    }
    if (descriptor >= plan.typed_lifecycle_first_descriptor
        && descriptor <= plan.typed_lifecycle_last_descriptor) {
        return PrefixAuthorityV1::kEventCursor;
    }
    return PrefixAuthorityV1::kNone;
}

bool is_policy_tape_descriptor(
    const DescriptorPlanV1& plan,
    const std::int64_t descriptor
) noexcept {
    return descriptor >= plan.policy_tape_first_descriptor
        && descriptor <= plan.policy_tape_last_descriptor;
}

bool nonnegative_u64(
    const std::int64_t value,
    std::uint64_t* const output
) noexcept {
    if (value < 0 || output == nullptr) {
        return false;
    }
    *output = static_cast<std::uint64_t>(value);
    return true;
}

bool metadata_prefix(
    const HostTokenResolverRequestV1& request,
    const metadata::CompactSummaryMetadataRetentionV1& row,
    const std::int64_t descriptor,
    std::uint64_t* const output
) noexcept {
    const DescriptorPlanV1* const plan = descriptor_plan(
        request.descriptor_count
    );
    if (plan == nullptr) {
        return false;
    }
    std::int64_t value = -1;
    switch (prefix_authority(*plan, descriptor)) {
        case PrefixAuthorityV1::kActiveCount:
            value = row.active_count;
            break;
        case PrefixAuthorityV1::kEventCursor:
            value = row.event_cursor;
            break;
        case PrefixAuthorityV1::kPolicyBirthTapeCursor:
            value = row.policy_birth_tape_cursor;
            break;
        case PrefixAuthorityV1::kSlotFreeQueueCount:
            value = row.slot_free_queue_count;
            break;
        case PrefixAuthorityV1::kNone:
            return false;
    }
    return nonnegative_u64(value, output);
}

Status validate_dimensions_and_runtime(
    const HostTokenResolverRequestV1& request
) noexcept {
    if (request.resolver_rows == nullptr
        || request.host_template_runtime == nullptr
        || request.descriptor_specs == nullptr
        || request.checkpoint_raw_digests == nullptr
        || request.checkpoint_canonical_digests == nullptr
        || request.checkpoint_status == nullptr
        || request.checkpoint_validity == nullptr
        || request.metadata_rows == nullptr
        || request.import_digests == nullptr) {
        return fail(
            request.failure,
            Status::kNullArgument,
            FailureComponent::kRequest
        );
    }
    if (request.generation_count <= 0
        || request.active_generation_count <= 0
        || request.active_generation_count > request.generation_count
        || request.cell_count <= 0
        || !checkpoint::is_supported_descriptor_count_v1(
            request.descriptor_count
        )
        || static_cast<std::uint64_t>(request.generation_count) > UINT32_MAX
        || static_cast<std::uint64_t>(request.cell_count) > UINT32_MAX) {
        return fail(
            request.failure,
            Status::kInvalidDimensions,
            FailureComponent::kRequest
        );
    }

    std::uint64_t generation_cells = 0;
    std::uint64_t checkpoint_rows = 0;
    std::uint64_t digest_bytes = 0;
    std::uint64_t program_count = 0;
    if (!checked_multiply(
            static_cast<std::uint64_t>(request.generation_count),
            static_cast<std::uint64_t>(request.cell_count),
            &generation_cells
        )
        || !checked_multiply(
            generation_cells,
            static_cast<std::uint64_t>(request.descriptor_count),
            &checkpoint_rows
        )
        || !checked_multiply(checkpoint_rows, kDigestBytesV1, &digest_bytes)
        || !checked_multiply(generation_cells, kSurfaceCountV1, &program_count)
        || !checked_index_extent(generation_cells)
        || !checked_index_extent(checkpoint_rows)
        || !checked_index_extent(digest_bytes)
        || !checked_index_extent(program_count)) {
        return fail(
            request.failure,
            Status::kArithmeticOverflow,
            FailureComponent::kRequest
        );
    }
    if (runtime::host_template_runtime_status_v1(
            request.host_template_runtime
        ) != runtime::HostTemplateRuntimeStatusV1::kSuccess
        || runtime::host_template_program_count_v1(
            request.host_template_runtime
        ) != program_count) {
        return fail(
            request.failure,
            Status::kRuntimeNotReady,
            FailureComponent::kRuntime
        );
    }
    return Status::kSuccess;
}

Status validate_descriptor_specs(
    const HostTokenResolverRequestV1& request
) noexcept {
    const DescriptorPlanV1* const plan = descriptor_plan(
        request.descriptor_count
    );
    if (plan == nullptr) {
        return fail(
            request.failure,
            Status::kInvalidDimensions,
            FailureComponent::kRequest
        );
    }
    bool raw_outputs[checkpoint::kDescriptorCapacityV1]{};
    bool canonical_outputs[checkpoint::kDescriptorCapacityV1]{};
    for (std::int64_t descriptor = 0;
         descriptor < request.descriptor_count;
         ++descriptor) {
        const std::int64_t* const spec = descriptor_spec(request, descriptor);
        const std::int64_t row_capacity = spec[checkpoint::kRowCapacityV1];
        const std::int64_t row_bytes = spec[checkpoint::kRowBytesV1];
        const std::int64_t dtype_bytes = spec[checkpoint::kDtypeBytesV1];
        const std::int64_t raw_output = spec[checkpoint::kRawOutputIndexV1];
        const std::int64_t canonical_output =
            spec[checkpoint::kCanonicalOutputIndexV1];
        const std::int64_t mode = spec[checkpoint::kCanonicalizationModeV1];
        const bool active_prefix =
            prefix_authority(*plan, descriptor) != PrefixAuthorityV1::kNone;
        std::uint64_t bytes_per_cell = 0;
        if (spec[checkpoint::kDescriptorIndexV1] != descriptor
            || row_capacity <= 0 || row_bytes <= 0 || dtype_bytes <= 0
            || row_bytes % dtype_bytes != 0
            || !checked_multiply(
                static_cast<std::uint64_t>(row_capacity),
                static_cast<std::uint64_t>(row_bytes),
                &bytes_per_cell
            )
            || bytes_per_cell > static_cast<std::uint64_t>(INT64_MAX)
            || spec[checkpoint::kBytesPerCellV1]
                != static_cast<std::int64_t>(bytes_per_cell)
            || spec[checkpoint::kCellStrideBytesV1]
                != spec[checkpoint::kBytesPerCellV1]
            || raw_output < 0 || raw_output >= request.descriptor_count
            || canonical_output < 0
            || canonical_output >= request.descriptor_count
            || raw_outputs[raw_output]
            || canonical_outputs[canonical_output]
            || mode < kExactModeV1 || mode > kActivePrefixModeV1
            || ((mode == kActivePrefixModeV1) != active_prefix)
            || (descriptor == plan->policy_energy_descriptor
                && mode != kDeadRowZeroModeV1)
            || (descriptor == plan->mutation_torch_cpu_descriptor
                && (mode != kExactModeV1
                    || dtype_bytes != metadata::kDtypeUint8V1))) {
            return fail(
                request.failure,
                Status::kDescriptorContradiction,
                FailureComponent::kDescriptorSpecs,
                {},
                descriptor
            );
        }
        raw_outputs[raw_output] = true;
        canonical_outputs[canonical_output] = true;
    }
    return Status::kSuccess;
}

Status validate_checkpoint_planes(
    const HostTokenResolverRequestV1& request
) noexcept {
    const std::uint64_t row_count =
        static_cast<std::uint64_t>(request.active_generation_count)
        * static_cast<std::uint64_t>(request.cell_count)
        * static_cast<std::uint64_t>(request.descriptor_count);
    for (std::uint64_t row = 0; row < row_count; ++row) {
        if (request.checkpoint_status[row] != 0
            || request.checkpoint_validity[row] != 1U) {
            return fail(
                request.failure,
                Status::kCheckpointInvalid,
                FailureComponent::kCheckpointPlanes,
                {},
                static_cast<std::int64_t>(
                    row % static_cast<std::uint64_t>(request.descriptor_count)
                )
            );
        }
    }
    return Status::kSuccess;
}

bool nonnegative_metadata_scalars(
    const metadata::CompactSummaryMetadataRetentionV1& row
) noexcept {
    return row.generation >= 0
        && row.tick >= 0
        && row.configured_steps_per_generation > 0
        && row.active_count >= 0
        && row.capacity > 0
        && row.agent_id_to_slot_prefix_length > 0
        && row.agent_id_capacity > 0
        && row.slot_free_queue_count >= 0
        && row.event_cursor >= 0
        && row.event_capacity >= 0
        && row.runtime_event_cursor >= 0
        && row.population_next_id > 0
        && row.policy_birth_tape_cursor >= 0
        && row.policy_birth_tape_capacity >= 0
        && row.generation_calls >= 0
        && row.restore_count >= 0
        && row.last_applied_actions >= 0
        && row.last_moved >= 0
        && row.last_gathered >= 0
        && row.last_policy_refill_calls >= 0
        && row.last_policy_refill_rows >= 0
        && row.mutation_torch_cpu_state_bytes >= 0
        && row.policy_energy_observation_capacity > 0
        && row.policy_energy_live_count >= 0
        && row.policy_energy_finite_live_count >= 0
        && row.native_runtime_rng_state >= 0;
}

bool valid_pcg64_buffers(
    const metadata::CompactSummaryMetadataRetentionV1& row
) noexcept {
    for (std::int32_t authority = 0;
         authority < metadata::kPcg64AuthorityCountV1;
         ++authority) {
        const std::int64_t flag = row.pcg64[authority].buffer_words[0];
        const std::int64_t value = row.pcg64[authority].buffer_words[1];
        if ((flag != 0 && flag != 1)
            || value < 0
            || static_cast<std::uint64_t>(value) > UINT32_MAX) {
            return false;
        }
    }
    return true;
}

Status validate_metadata_rows(
    const HostTokenResolverRequestV1& request
) noexcept {
    const DescriptorPlanV1* const plan = descriptor_plan(
        request.descriptor_count
    );
    if (plan == nullptr) {
        return fail(
            request.failure,
            Status::kInvalidDimensions,
            FailureComponent::kRequest
        );
    }
    const std::int64_t* const energy_spec =
        descriptor_spec(request, plan->policy_energy_descriptor);
    const std::int64_t* const mutation_spec = descriptor_spec(
        request,
        plan->mutation_torch_cpu_descriptor
    );
    for (std::int64_t generation = 0;
         generation < request.active_generation_count;
         ++generation) {
        for (std::int64_t cell = 0; cell < request.cell_count; ++cell) {
            const std::uint64_t row_index =
                static_cast<std::uint64_t>(generation)
                    * static_cast<std::uint64_t>(request.cell_count)
                + static_cast<std::uint64_t>(cell);
            const auto& row = request.metadata_rows[row_index];
            bool counters_valid = true;
            for (std::int32_t stream = 0;
                 stream < metadata::kSemanticStreamCounterCountV1;
                 ++stream) {
                counters_valid = counters_valid
                    && row.semantic_stream_counters[stream] >= 0;
            }
            bool prefixes_valid = true;
            for (std::int64_t descriptor = 0;
                 descriptor < request.descriptor_count;
                 ++descriptor) {
                const std::int64_t* const spec = descriptor_spec(
                    request,
                    descriptor
                );
                if (spec[checkpoint::kCanonicalizationModeV1]
                    != kActivePrefixModeV1) {
                    continue;
                }
                std::uint64_t prefix = 0;
                prefixes_valid = prefixes_valid
                    && metadata_prefix(request, row, descriptor, &prefix)
                    && prefix <= static_cast<std::uint64_t>(
                        spec[checkpoint::kRowCapacityV1]
                    );
            }
            if (row.source_version != metadata::kSourceVersionV1
                || row.status != metadata::kRetentionSuccessV1
                || row.validity != 1U
                || (row.flags & metadata::kRequiredRetentionFlagsV1)
                    != metadata::kRequiredRetentionFlagsV1
                || row.env_policy_alias != 1U
                || row.python_random_has_gauss > 1U
                || (row.python_random_has_gauss == 1U
                    && !std::isfinite(row.python_random_gauss_next))
                || row.mutation_torch_descriptor_validity != 1U
                || row.generation_index != generation
                || row.cell_index != cell
                || !nonnegative_metadata_scalars(row)
                || row.active_count > row.capacity
                || row.agent_id_to_slot_prefix_length != row.active_count
                || row.agent_id_to_slot_prefix_length > row.agent_id_capacity
                || row.slot_free_queue_count != row.capacity - row.active_count
                || row.event_cursor > row.event_capacity
                || row.runtime_event_cursor < row.event_cursor
                || row.policy_birth_tape_cursor
                    > row.policy_birth_tape_capacity
                || row.population_next_id
                    < row.agent_id_to_slot_prefix_length
                || row.policy_energy_live_count != row.active_count
                || row.policy_energy_finite_live_count != row.active_count
                || row.policy_energy_observation_capacity
                    != energy_spec[checkpoint::kRowCapacityV1]
                || row.python_random_index < 0
                || row.python_random_index > metadata::kPythonMt19937WordsV1
                || !valid_pcg64_buffers(row)
                || !counters_valid
                || !prefixes_valid
                || row.mutation_torch_descriptor_index
                    != static_cast<std::uint32_t>(
                        plan->mutation_torch_cpu_descriptor
                    )
                || row.mutation_torch_raw_output_index
                    != static_cast<std::uint32_t>(
                        mutation_spec[checkpoint::kRawOutputIndexV1]
                    )
                || row.mutation_torch_dtype_code != metadata::kDtypeUint8V1
                || row.mutation_torch_rank != metadata::kRankOneV1) {
                return fail(
                    request.failure,
                    Status::kMetadataInvalid,
                    FailureComponent::kMetadata,
                    {
                        kNoResolverRowV1,
                        static_cast<std::uint32_t>(generation),
                        static_cast<std::uint32_t>(cell),
                        kNoResolverCoordinateV1,
                        kNoResolverCoordinateV1,
                    }
                );
            }
        }
    }
    return Status::kSuccess;
}

Status selected_output_index(
    const HostTokenResolverRequestV1& request,
    const std::int64_t descriptor,
    const std::uint64_t selector,
    const TokenCoordinateV1 coordinate,
    std::int64_t* const output
) noexcept {
    if (descriptor < 0 || descriptor >= request.descriptor_count
        || output == nullptr
        || (selector != kSelectorRawV1
            && selector != kSelectorCanonicalV1)) {
        return fail(
            request.failure,
            Status::kResolverContradiction,
            FailureComponent::kResolverPlan,
            coordinate,
            descriptor
        );
    }
    const std::int64_t* const spec = descriptor_spec(request, descriptor);
    *output = spec[
        selector == kSelectorRawV1
            ? checkpoint::kRawOutputIndexV1
            : checkpoint::kCanonicalOutputIndexV1
    ];
    return Status::kSuccess;
}

Status descriptor_shape(
    const HostTokenResolverRequestV1& request,
    const metadata::CompactSummaryMetadataRetentionV1& metadata_row,
    const std::int64_t descriptor,
    const std::uint64_t selector,
    const std::uint32_t rank,
    const TokenCoordinateV1 coordinate,
    ResolvedTokenV1* const resolved
) noexcept {
    std::int64_t unused_output = 0;
    const Status selector_status = selected_output_index(
        request,
        descriptor,
        selector,
        coordinate,
        &unused_output
    );
    if (selector_status != Status::kSuccess) {
        return selector_status;
    }
    if (resolved == nullptr || (rank != 1U && rank != 2U)) {
        return fail(
            request.failure,
            Status::kResolverContradiction,
            FailureComponent::kResolverPlan,
            coordinate,
            descriptor
        );
    }
    const std::int64_t* const spec = descriptor_spec(request, descriptor);
    std::uint64_t first = static_cast<std::uint64_t>(
        spec[checkpoint::kRowCapacityV1]
    );
    if (selector == kSelectorCanonicalV1
        && spec[checkpoint::kCanonicalizationModeV1] == kActivePrefixModeV1
        && (!metadata_prefix(request, metadata_row, descriptor, &first)
            || first > static_cast<std::uint64_t>(
                spec[checkpoint::kRowCapacityV1]
            ))) {
        return fail(
            request.failure,
            Status::kMetadataInvalid,
            FailureComponent::kMetadata,
            coordinate,
            descriptor
        );
    }
    const std::uint64_t inner = static_cast<std::uint64_t>(
        spec[checkpoint::kRowBytesV1] / spec[checkpoint::kDtypeBytesV1]
    );
    if ((rank == 1U && inner != 1U) || (rank == 2U && inner == 0U)) {
        return fail(
            request.failure,
            Status::kDescriptorContradiction,
            FailureComponent::kDescriptorSpecs,
            coordinate,
            descriptor
        );
    }
    resolved->shape_rank = rank;
    resolved->shape_dimensions[0] = first;
    resolved->shape_dimensions[1] = rank == 2U ? inner : 0U;
    return Status::kSuccess;
}

Status descriptor_numel_or_bytes(
    const HostTokenResolverRequestV1& request,
    const metadata::CompactSummaryMetadataRetentionV1& metadata_row,
    const std::int64_t descriptor,
    const std::uint64_t selector,
    const bool bytes,
    const TokenCoordinateV1 coordinate,
    std::uint64_t* const output
) noexcept {
    std::int64_t unused_output = 0;
    const Status selector_status = selected_output_index(
        request,
        descriptor,
        selector,
        coordinate,
        &unused_output
    );
    if (selector_status != Status::kSuccess) {
        return selector_status;
    }
    const std::int64_t* const spec = descriptor_spec(request, descriptor);
    std::uint64_t first = static_cast<std::uint64_t>(
        spec[checkpoint::kRowCapacityV1]
    );
    if (selector == kSelectorCanonicalV1
        && spec[checkpoint::kCanonicalizationModeV1] == kActivePrefixModeV1
        && (!metadata_prefix(request, metadata_row, descriptor, &first)
            || first > static_cast<std::uint64_t>(
                spec[checkpoint::kRowCapacityV1]
            ))) {
        return fail(
            request.failure,
            Status::kMetadataInvalid,
            FailureComponent::kMetadata,
            coordinate,
            descriptor
        );
    }
    const std::uint64_t width = bytes
        ? static_cast<std::uint64_t>(spec[checkpoint::kRowBytesV1])
        : static_cast<std::uint64_t>(
            spec[checkpoint::kRowBytesV1]
                / spec[checkpoint::kDtypeBytesV1]
        );
    if (!checked_multiply(first, width, output)) {
        return fail(
            request.failure,
            Status::kArithmeticOverflow,
            FailureComponent::kDescriptorSpecs,
            coordinate,
            descriptor
        );
    }
    return Status::kSuccess;
}

const metadata::CompactSummaryMetadataRetentionV1& metadata_at(
    const HostTokenResolverRequestV1& request,
    const std::uint32_t generation,
    const std::uint32_t cell
) noexcept {
    return request.metadata_rows[
        static_cast<std::size_t>(generation)
            * static_cast<std::size_t>(request.cell_count)
        + cell
    ];
}

const std::uint8_t* checkpoint_digest_at(
    const HostTokenResolverRequestV1& request,
    const std::uint8_t* const plane,
    const std::uint32_t generation,
    const std::uint32_t cell,
    const std::int64_t output
) noexcept {
    const std::size_t row =
        (static_cast<std::size_t>(generation)
             * static_cast<std::size_t>(request.cell_count)
         + cell)
            * static_cast<std::size_t>(request.descriptor_count)
        + static_cast<std::size_t>(output);
    return plane + row * static_cast<std::size_t>(kDigestBytesV1);
}

Status checkpoint_digest(
    const HostTokenResolverRequestV1& request,
    const std::int64_t descriptor,
    const std::uint64_t selector,
    const TokenCoordinateV1 coordinate,
    const std::uint8_t** const digest
) noexcept {
    std::int64_t output = 0;
    const Status status = selected_output_index(
        request,
        descriptor,
        selector,
        coordinate,
        &output
    );
    if (status != Status::kSuccess) {
        return status;
    }
    *digest = checkpoint_digest_at(
        request,
        selector == kSelectorRawV1
            ? request.checkpoint_raw_digests
            : request.checkpoint_canonical_digests,
        coordinate.generation,
        coordinate.cell,
        output
    );
    return Status::kSuccess;
}

std::int32_t pcg_authority_slot(const std::uint64_t authority) noexcept {
    switch (authority) {
        case 1U: return 3;  // birth_spawn
        case 2U: return 0;  // env, aliases policy
        case 3U: return 0;  // policy, aliases env
        case 4U: return 2;  // population
        case 5U: return 1;  // runner
        default: return -1;
    }
}

Status require_declaration(
    const HostTokenResolverRequestV1& request,
    const TokenCoordinateV1 coordinate,
    const std::uint64_t primary,
    const std::uint64_t secondary,
    const std::uint64_t rank,
    const std::uint64_t primary_minimum,
    const std::uint64_t primary_maximum,
    const std::uint64_t required_secondary,
    const std::uint64_t required_rank
) noexcept {
    if (primary < primary_minimum || primary > primary_maximum
        || secondary != required_secondary || rank != required_rank) {
        return fail(
            request.failure,
            Status::kResolverContradiction,
            FailureComponent::kResolverPlan,
            coordinate,
            primary <= static_cast<std::uint64_t>(INT64_MAX)
                ? static_cast<std::int64_t>(primary)
                : kNoDescriptorIndexV1
        );
    }
    return Status::kSuccess;
}

Status resolve_source_value(
    const HostTokenResolverRequestV1& request,
    const std::uint64_t* const row,
    const TokenCoordinateV1 coordinate,
    ResolvedTokenV1* const resolved
) noexcept {
    const std::uint64_t kind = row[kResolverSourceKindColumnV1];
    const std::uint64_t primary = row[kResolverPrimaryIndexColumnV1];
    const std::uint64_t secondary = row[kResolverSecondaryCodeColumnV1];
    const std::uint64_t rank = row[kResolverRankColumnV1];
    const auto& metadata_row = metadata_at(
        request,
        coordinate.generation,
        coordinate.cell
    );
    const DescriptorPlanV1* const plan = descriptor_plan(
        request.descriptor_count
    );
    *resolved = {};

    const auto descriptor_value = [&]() noexcept -> std::int64_t {
        return primary <= static_cast<std::uint64_t>(INT64_MAX)
            ? static_cast<std::int64_t>(primary)
            : kNoDescriptorIndexV1;
    };
    const auto generic_descriptor = [&]() noexcept -> Status {
        if (primary >= static_cast<std::uint64_t>(request.descriptor_count)
            || plan == nullptr
            || is_policy_tape_descriptor(*plan, descriptor_value())
            || (secondary != kSelectorRawV1
                && secondary != kSelectorCanonicalV1)) {
            return fail(
                request.failure,
                Status::kResolverContradiction,
                FailureComponent::kResolverPlan,
                coordinate,
                descriptor_value()
            );
        }
        return Status::kSuccess;
    };

    switch (kind) {
        case kCheckpointTrajectoryDigestV1:
        case kCheckpointCanonicalDigestV1:
        case kCheckpointSourceManifestDigestV1: {
            const Status declaration = generic_descriptor();
            if (declaration != Status::kSuccess || rank != 0U) {
                return declaration != Status::kSuccess
                    ? declaration
                    : fail(
                        request.failure,
                        Status::kResolverContradiction,
                        FailureComponent::kResolverPlan,
                        coordinate,
                        descriptor_value()
                    );
            }
            resolved->token_type = static_cast<std::uint32_t>(
                renderer::HostTemplateTokenTypeV1::kSha256Hex
            );
            return checkpoint_digest(
                request,
                descriptor_value(),
                secondary,
                coordinate,
                &resolved->digest
            );
        }
        case kCheckpointTrajectoryShapeV1:
        case kCheckpointCanonicalShapeV1:
        case kCheckpointSourceManifestShapeV1:
        case kCheckpointStoredManifestShapeV1: {
            const Status declaration = generic_descriptor();
            if (declaration != Status::kSuccess) {
                return declaration;
            }
            resolved->token_type = static_cast<std::uint32_t>(
                renderer::HostTemplateTokenTypeV1::kTensorShape
            );
            return descriptor_shape(
                request,
                metadata_row,
                descriptor_value(),
                secondary,
                static_cast<std::uint32_t>(rank),
                coordinate,
                resolved
            );
        }
        case kCheckpointStoredManifestNumelV1:
        case kCheckpointStoredManifestBytesV1: {
            const Status declaration = generic_descriptor();
            if (declaration != Status::kSuccess || rank != 0U) {
                return declaration != Status::kSuccess
                    ? declaration
                    : fail(
                        request.failure,
                        Status::kResolverContradiction,
                        FailureComponent::kResolverPlan,
                        coordinate,
                        descriptor_value()
                    );
            }
            resolved->token_type = static_cast<std::uint32_t>(
                renderer::HostTemplateTokenTypeV1::kInteger
            );
            return descriptor_numel_or_bytes(
                request,
                metadata_row,
                descriptor_value(),
                secondary,
                kind == kCheckpointStoredManifestBytesV1,
                coordinate,
                &resolved->scalar_low
            );
        }
        case kProjectionTopLevelIntegerV1: {
            const Status declaration = require_declaration(
                request, coordinate, primary, secondary, rank,
                1U, 9U, kSelectorNoneV1, 0U
            );
            if (declaration != Status::kSuccess) {
                return declaration;
            }
            const std::int64_t values[9] = {
                metadata_row.restore_count,
                metadata_row.generation_calls,
                metadata_row.event_cursor,
                metadata_row.runtime_event_cursor,
                metadata_row.last_applied_actions,
                metadata_row.last_moved,
                metadata_row.last_gathered,
                metadata_row.last_policy_refill_calls,
                metadata_row.last_policy_refill_rows,
            };
            resolved->token_type = static_cast<std::uint32_t>(
                renderer::HostTemplateTokenTypeV1::kInteger
            );
            resolved->scalar_low = static_cast<std::uint64_t>(values[primary - 1U]);
            return Status::kSuccess;
        }
        case kProjectionMetadataIntegerV1:
        case kProjectionMetadataBooleanV1: {
            const std::uint64_t expected_primary =
                kind == kProjectionMetadataBooleanV1 ? 6U : primary;
            const Status declaration = require_declaration(
                request,
                coordinate,
                primary,
                secondary,
                rank,
                kind == kProjectionMetadataBooleanV1 ? 6U : 1U,
                kind == kProjectionMetadataBooleanV1 ? 6U : 13U,
                kSelectorNoneV1,
                0U
            );
            if (declaration != Status::kSuccess
                || (kind == kProjectionMetadataIntegerV1
                    && expected_primary == 6U)) {
                return declaration != Status::kSuccess
                    ? declaration
                    : fail(
                        request.failure,
                        Status::kResolverContradiction,
                        FailureComponent::kResolverPlan,
                        coordinate
                    );
            }
            if (kind == kProjectionMetadataBooleanV1) {
                resolved->token_type = static_cast<std::uint32_t>(
                    renderer::HostTemplateTokenTypeV1::kBoolean
                );
                resolved->scalar_low = request.descriptor_count
                        == kPatchP2DescriptorPlanV1.descriptor_count
                    ? 1U
                    : 0U;
                return Status::kSuccess;
            }
            const std::int64_t values[13] = {
                metadata_row.generation,
                metadata_row.active_count,
                metadata_row.event_cursor,
                metadata_row.event_cursor,
                0,  // Compact checkpoints retain no inactive event-tape tail.
                0,
                metadata_row.native_runtime_rng_state,
                metadata_row.policy_energy_finite_live_count,
                metadata_row.policy_energy_live_count,
                metadata_row.population_next_id,
                metadata_row.event_cursor,
                metadata_row.event_cursor,
                metadata_row.configured_steps_per_generation,
            };
            resolved->token_type = static_cast<std::uint32_t>(
                renderer::HostTemplateTokenTypeV1::kInteger
            );
            resolved->scalar_low = static_cast<std::uint64_t>(values[primary - 1U]);
            return Status::kSuccess;
        }
        case kPcg64StateV1:
        case kPcg64IncV1:
        case kPcg64BufferFlagV1:
        case kPcg64BufferValueV1: {
            const Status declaration = require_declaration(
                request, coordinate, primary, secondary, rank,
                1U, 5U, kSelectorNoneV1, 0U
            );
            if (declaration != Status::kSuccess) {
                return declaration;
            }
            const std::int32_t slot = pcg_authority_slot(primary);
            const auto& pcg = metadata_row.pcg64[slot];
            if (kind == kPcg64StateV1 || kind == kPcg64IncV1) {
                const std::int32_t offset = kind == kPcg64StateV1 ? 0 : 2;
                resolved->token_type = static_cast<std::uint32_t>(
                    renderer::HostTemplateTokenTypeV1::kPcgUint128
                );
                resolved->scalar_low = pcg.state_words[offset];
                resolved->scalar_high = pcg.state_words[offset + 1];
            } else {
                const std::int32_t offset =
                    kind == kPcg64BufferFlagV1 ? 0 : 1;
                resolved->token_type = static_cast<std::uint32_t>(
                    renderer::HostTemplateTokenTypeV1::kInteger
                );
                resolved->scalar_low = static_cast<std::uint64_t>(
                    pcg.buffer_words[offset]
                );
            }
            return Status::kSuccess;
        }
        case kPythonMtWordV1: {
            const Status declaration = require_declaration(
                request, coordinate, primary, secondary, rank,
                0U,
                static_cast<std::uint64_t>(metadata::kPythonMt19937WordsV1 - 1),
                kSelectorNoneV1,
                0U
            );
            if (declaration != Status::kSuccess) {
                return declaration;
            }
            resolved->token_type = static_cast<std::uint32_t>(
                renderer::HostTemplateTokenTypeV1::kPythonMtWord
            );
            resolved->scalar_low = metadata_row.python_random_words[primary];
            return Status::kSuccess;
        }
        case kPythonMtIndexV1: {
            const Status declaration = require_declaration(
                request, coordinate, primary, secondary, rank,
                0U, 0U, kSelectorNoneV1, 0U
            );
            if (declaration != Status::kSuccess) {
                return declaration;
            }
            resolved->token_type = static_cast<std::uint32_t>(
                renderer::HostTemplateTokenTypeV1::kInteger
            );
            resolved->scalar_low = static_cast<std::uint64_t>(
                metadata_row.python_random_index
            );
            return Status::kSuccess;
        }
        case kPythonGaussCacheV1: {
            const Status declaration = require_declaration(
                request, coordinate, primary, secondary, rank,
                0U, 0U, kSelectorNoneV1, 0U
            );
            if (declaration != Status::kSuccess) {
                return declaration;
            }
            resolved->token_type = static_cast<std::uint32_t>(
                renderer::HostTemplateTokenTypeV1::kNullableFloat64
            );
            resolved->scalar_high = metadata_row.python_random_has_gauss;
            if (resolved->scalar_high == 1U) {
                std::memcpy(
                    &resolved->scalar_low,
                    &metadata_row.python_random_gauss_next,
                    sizeof(metadata_row.python_random_gauss_next)
                );
            }
            return Status::kSuccess;
        }
        case kSemanticStreamCounterV1: {
            const Status declaration = require_declaration(
                request, coordinate, primary, secondary, rank,
                1U, 16U, kSelectorNoneV1, 0U
            );
            if (declaration != Status::kSuccess) {
                return declaration;
            }
            std::int32_t retained = -1;
            switch (primary) {
                case 2U: retained = metadata::kSemanticMetricsNumpyV1; break;
                case 5U: retained = metadata::kSemanticMutationPyV1; break;
                case 6U: retained = metadata::kSemanticMutationNumpyV1; break;
                case 7U: retained = metadata::kSemanticMutationTorchCpuV1; break;
                case 9U: retained = metadata::kSemanticPolicyPyV1; break;
                case 10U: retained = metadata::kSemanticPolicyNumpyV1; break;
                case 1U:
                case 3U:
                case 4U:
                case 8U:
                case 11U:
                case 12U:
                case 13U:
                case 14U:
                case 15U:
                case 16U:
                    break;  // Ten explicit zero streams in the Flat envelope.
                default:
                    return fail(
                        request.failure,
                        Status::kResolverContradiction,
                        FailureComponent::kResolverPlan,
                        coordinate
                    );
            }
            resolved->token_type = static_cast<std::uint32_t>(
                renderer::HostTemplateTokenTypeV1::kInteger
            );
            resolved->scalar_low = retained < 0
                ? 0U
                : static_cast<std::uint64_t>(
                    metadata_row.semantic_stream_counters[retained]
                );
            return Status::kSuccess;
        }
        case kMetadataImportDigestV1: {
            const Status declaration = require_declaration(
                request, coordinate, primary, secondary, rank,
                0U, 0U, kSelectorNoneV1, 0U
            );
            if (declaration != Status::kSuccess) {
                return declaration;
            }
            resolved->token_type = static_cast<std::uint32_t>(
                renderer::HostTemplateTokenTypeV1::kSha256Hex
            );
            resolved->digest = request.import_digests
                + static_cast<std::size_t>(coordinate.cell)
                    * static_cast<std::size_t>(kDigestBytesV1);
            return Status::kSuccess;
        }
        case kPolicyEnergyDigestV1:
        case kPolicyEnergyShapeV1: {
            const std::uint64_t energy_descriptor = plan == nullptr
                ? UINT64_MAX
                : static_cast<std::uint64_t>(
                    plan->policy_energy_descriptor
                );
            const Status declaration = require_declaration(
                request,
                coordinate,
                primary,
                secondary,
                rank,
                energy_descriptor,
                energy_descriptor,
                kSelectorRawV1,
                kind == kPolicyEnergyDigestV1 ? 0U : 1U
            );
            if (declaration != Status::kSuccess) {
                return declaration;
            }
            if (kind == kPolicyEnergyDigestV1) {
                resolved->token_type = static_cast<std::uint32_t>(
                    renderer::HostTemplateTokenTypeV1::kSha256Hex
                );
                resolved->digest = metadata_row.policy_energy_tagged_sha256;
                return Status::kSuccess;
            }
            resolved->token_type = static_cast<std::uint32_t>(
                renderer::HostTemplateTokenTypeV1::kTensorShape
            );
            return descriptor_shape(
                request,
                metadata_row,
                descriptor_value(),
                kSelectorRawV1,
                1U,
                coordinate,
                resolved
            );
        }
        case kIdentityCountV1:
        case kIdentityDigestV1: {
            const std::uint64_t identity_descriptor = plan == nullptr
                ? UINT64_MAX
                : static_cast<std::uint64_t>(plan->identity_descriptor);
            const Status declaration = require_declaration(
                request,
                coordinate,
                primary,
                secondary,
                rank,
                identity_descriptor,
                identity_descriptor,
                kSelectorCanonicalV1,
                0U
            );
            if (declaration != Status::kSuccess) {
                return declaration;
            }
            if (kind == kIdentityCountV1) {
                resolved->token_type = static_cast<std::uint32_t>(
                    renderer::HostTemplateTokenTypeV1::kInteger
                );
                resolved->scalar_low = static_cast<std::uint64_t>(
                    metadata_row.active_count
                );
                return Status::kSuccess;
            }
            resolved->token_type = static_cast<std::uint32_t>(
                renderer::HostTemplateTokenTypeV1::kSha256Hex
            );
            return checkpoint_digest(
                request,
                descriptor_value(),
                kSelectorCanonicalV1,
                coordinate,
                &resolved->digest
            );
        }
        case kPolicyMutationTapeCursorV1: {
            const Status declaration = require_declaration(
                request, coordinate, primary, secondary, rank,
                0U, 0U, kSelectorNoneV1, 0U
            );
            if (declaration != Status::kSuccess) {
                return declaration;
            }
            resolved->token_type = static_cast<std::uint32_t>(
                renderer::HostTemplateTokenTypeV1::kInteger
            );
            resolved->scalar_low = static_cast<std::uint64_t>(
                metadata_row.policy_birth_tape_cursor
            );
            return Status::kSuccess;
        }
        case kPolicyMutationTapeDigestV1:
        case kPolicyMutationTapeShapeV1: {
            if (plan == nullptr
                || !is_policy_tape_descriptor(*plan, descriptor_value())
                || secondary != kSelectorCanonicalV1
                || (kind == kPolicyMutationTapeDigestV1 && rank != 0U)
                || (kind == kPolicyMutationTapeShapeV1
                    && rank != 1U && rank != 2U)) {
                return fail(
                    request.failure,
                    Status::kResolverContradiction,
                    FailureComponent::kResolverPlan,
                    coordinate,
                    descriptor_value()
                );
            }
            if (kind == kPolicyMutationTapeDigestV1) {
                resolved->token_type = static_cast<std::uint32_t>(
                    renderer::HostTemplateTokenTypeV1::kSha256Hex
                );
                return checkpoint_digest(
                    request,
                    descriptor_value(),
                    kSelectorCanonicalV1,
                    coordinate,
                    &resolved->digest
                );
            }
            resolved->token_type = static_cast<std::uint32_t>(
                renderer::HostTemplateTokenTypeV1::kTensorShape
            );
            return descriptor_shape(
                request,
                metadata_row,
                descriptor_value(),
                kSelectorCanonicalV1,
                static_cast<std::uint32_t>(rank),
                coordinate,
                resolved
            );
        }
        default:
            return fail(
                request.failure,
                Status::kUnsupportedSourceKind,
                FailureComponent::kResolverPlan,
                coordinate
            );
    }
}

void binary_digest_to_hex(
    const std::uint8_t* const digest,
    std::uint8_t* const output
) noexcept {
    constexpr std::uint8_t hexadecimal[] = "0123456789abcdef";
    for (std::uint32_t byte = 0; byte < kDigestBytesV1; ++byte) {
        output[byte * 2U] = hexadecimal[digest[byte] >> 4U];
        output[byte * 2U + 1U] = hexadecimal[digest[byte] & 0x0fU];
    }
}

Status validate_or_write_token(
    const HostTokenResolverRequestV1& request,
    const std::uint64_t* const row,
    const runtime::HostTemplateProgramBindingV1& binding,
    const TokenCoordinateV1 coordinate,
    const bool write
) noexcept {
    if (row[kResolverAbiVersionColumnV1] != kResolverAbiVersionV1) {
        return fail(
            request.failure,
            Status::kResolverAbiMismatch,
            FailureComponent::kResolverPlan,
            coordinate
        );
    }
    if (row[kResolverReservedZeroColumnV1] != 0U) {
        return fail(
            request.failure,
            Status::kResolverReservedNonzero,
            FailureComponent::kResolverPlan,
            coordinate
        );
    }
    if (row[kResolverRankColumnV1] > UINT32_MAX
        || binding.program == nullptr
        || binding.program->tokens == nullptr
        || binding.program->tokens[coordinate.token].rank
            != static_cast<std::uint32_t>(row[kResolverRankColumnV1])) {
        return fail(
            request.failure,
            Status::kResolverContradiction,
            FailureComponent::kResolverPlan,
            coordinate
        );
    }

    ResolvedTokenV1 resolved{};
    const Status source_status = resolve_source_value(
        request,
        row,
        coordinate,
        &resolved
    );
    if (source_status != Status::kSuccess) {
        return source_status;
    }
    runtime::HostTemplateTokenStorageV1 storage =
        runtime::lookup_host_template_token_storage_v1(
            request.host_template_runtime,
            coordinate.generation,
            coordinate.cell,
            coordinate.surface,
            coordinate.token
        );
    if (storage.value == nullptr
        || storage.value->token_type != resolved.token_type
        || binding.program->tokens[coordinate.token].token_type
            != resolved.token_type
        || (resolved.token_type == static_cast<std::uint32_t>(
                renderer::HostTemplateTokenTypeV1::kTensorShape
            )
            && (storage.shape_dimensions == nullptr
                || storage.shape_dimension_capacity != resolved.shape_rank))
        || (resolved.token_type == static_cast<std::uint32_t>(
                renderer::HostTemplateTokenTypeV1::kSha256Hex
            )
            && (resolved.digest == nullptr
                || storage.sha256_hex == nullptr
                || storage.sha256_hex_capacity != kSha256HexBytesV1))) {
        return fail(
            request.failure,
            Status::kTokenStorageContradiction,
            FailureComponent::kTokenStorage,
            coordinate
        );
    }
    if (!write) {
        return Status::kSuccess;
    }

    storage.value->scalar_low = resolved.scalar_low;
    storage.value->scalar_high = resolved.scalar_high;
    storage.value->reserved = 0U;
    if (resolved.token_type == static_cast<std::uint32_t>(
            renderer::HostTemplateTokenTypeV1::kTensorShape
        )) {
        for (std::uint32_t dimension = 0;
             dimension < resolved.shape_rank;
             ++dimension) {
            storage.shape_dimensions[dimension] =
                resolved.shape_dimensions[dimension];
        }
    } else if (resolved.token_type == static_cast<std::uint32_t>(
                   renderer::HostTemplateTokenTypeV1::kSha256Hex
               )) {
        binary_digest_to_hex(resolved.digest, storage.sha256_hex);
    }
    return Status::kSuccess;
}

Status token_count(
    const HostTokenResolverRequestV1& request,
    std::uint64_t* const output
) noexcept {
    std::uint64_t count = 0;
    for (std::uint32_t generation = 0;
         generation
             < static_cast<std::uint32_t>(request.active_generation_count);
         ++generation) {
        for (std::uint32_t cell = 0;
             cell < static_cast<std::uint32_t>(request.cell_count);
             ++cell) {
            for (std::uint32_t surface = 1U;
                 surface <= kSurfaceCountV1;
                 ++surface) {
                const runtime::HostTemplateProgramBindingV1 binding =
                    runtime::lookup_host_template_program_v1(
                        request.host_template_runtime,
                        generation,
                        cell,
                        surface
                    );
                if (binding.program == nullptr
                    || binding.token_value_count != binding.program->token_count
                    || UINT64_MAX - count < binding.token_value_count) {
                    return fail(
                        request.failure,
                        binding.program == nullptr
                            ? Status::kRuntimeNotReady
                            : Status::kArithmeticOverflow,
                        FailureComponent::kRuntime,
                        {
                            kNoResolverRowV1,
                            generation,
                            cell,
                            surface,
                            kNoResolverCoordinateV1,
                        }
                    );
                }
                count += binding.token_value_count;
            }
        }
    }
    *output = count;
    return Status::kSuccess;
}

Status traverse_tokens(
    const HostTokenResolverRequestV1& request,
    const bool write
) noexcept {
    std::uint64_t resolver_row = 0;
    for (std::uint32_t generation = 0;
         generation
             < static_cast<std::uint32_t>(request.active_generation_count);
         ++generation) {
        for (std::uint32_t cell = 0;
             cell < static_cast<std::uint32_t>(request.cell_count);
             ++cell) {
            for (std::uint32_t surface = 1U;
                 surface <= kSurfaceCountV1;
                 ++surface) {
                const runtime::HostTemplateProgramBindingV1 binding =
                    runtime::lookup_host_template_program_v1(
                        request.host_template_runtime,
                        generation,
                        cell,
                        surface
                    );
                for (std::uint32_t token = 0;
                     token < binding.token_value_count;
                     ++token, ++resolver_row) {
                    const TokenCoordinateV1 coordinate = {
                        resolver_row,
                        generation,
                        cell,
                        surface,
                        token,
                    };
                    const std::uint64_t* const row = request.resolver_rows
                        + static_cast<std::size_t>(resolver_row)
                            * kResolverColumnCountV1;
                    const Status status = validate_or_write_token(
                        request,
                        row,
                        binding,
                        coordinate,
                        write
                    );
                    if (status != Status::kSuccess) {
                        return status;
                    }
                }
            }
        }
    }
    return Status::kSuccess;
}

}  // namespace

HostTokenResolverStatusV1 resolve_compact_summary_host_tokens_v1(
    const HostTokenResolverRequestV1* const request
) noexcept {
    if (request == nullptr || request->failure == nullptr) {
        return Status::kNullArgument;
    }
    *request->failure = success_failure();
    Status status = validate_dimensions_and_runtime(*request);
    if (status != Status::kSuccess) {
        return status;
    }
    status = validate_descriptor_specs(*request);
    if (status != Status::kSuccess) {
        return status;
    }
    status = validate_checkpoint_planes(*request);
    if (status != Status::kSuccess) {
        return status;
    }
    status = validate_metadata_rows(*request);
    if (status != Status::kSuccess) {
        return status;
    }
    std::uint64_t expected_rows = 0;
    status = token_count(*request, &expected_rows);
    if (status != Status::kSuccess) {
        return status;
    }
    if (request->resolver_row_count < expected_rows
        || (request->active_generation_count == request->generation_count
            && request->resolver_row_count != expected_rows)
        || !checked_index_extent(expected_rows)
        || expected_rows > UINT64_MAX / kResolverColumnCountV1) {
        return fail(
            request->failure,
            request->resolver_row_count < expected_rows
                    || (request->active_generation_count
                            == request->generation_count
                        && request->resolver_row_count != expected_rows)
                ? Status::kResolverRowCountMismatch
                : Status::kArithmeticOverflow,
            FailureComponent::kResolverPlan
        );
    }

    // The first traversal validates every row and every destination. Therefore
    // malformed input cannot leave a partially updated runtime.
    status = traverse_tokens(*request, false);
    if (status != Status::kSuccess) {
        return status;
    }
    status = traverse_tokens(*request, true);
    if (status == Status::kSuccess) {
        *request->failure = success_failure();
    }
    return status;
}

const char* host_token_resolver_status_name_v1(
    const HostTokenResolverStatusV1 status
) noexcept {
    switch (status) {
        case Status::kSuccess: return "success";
        case Status::kNullArgument: return "null_argument";
        case Status::kInvalidDimensions: return "invalid_dimensions";
        case Status::kArithmeticOverflow: return "arithmetic_overflow";
        case Status::kRuntimeNotReady: return "runtime_not_ready";
        case Status::kResolverRowCountMismatch:
            return "resolver_row_count_mismatch";
        case Status::kResolverAbiMismatch: return "resolver_abi_mismatch";
        case Status::kResolverReservedNonzero:
            return "resolver_reserved_nonzero";
        case Status::kUnsupportedSourceKind:
            return "unsupported_source_kind";
        case Status::kResolverContradiction: return "resolver_contradiction";
        case Status::kDescriptorContradiction:
            return "descriptor_contradiction";
        case Status::kCheckpointInvalid: return "checkpoint_invalid";
        case Status::kMetadataInvalid: return "metadata_invalid";
        case Status::kTokenStorageContradiction:
            return "token_storage_contradiction";
    }
    return "unknown";
}

}  // namespace eiel::cuda_architecture_b::compact_summary_host_token_resolver_v1
