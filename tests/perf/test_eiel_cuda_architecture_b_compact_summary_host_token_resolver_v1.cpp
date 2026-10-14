#include "cuda_architecture_b_compact_summary_host_token_resolver_v1.hpp"

#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <new>
#include <stdexcept>
#include <string>
#include <vector>

namespace resolver =
    eiel::cuda_architecture_b::compact_summary_host_token_resolver_v1;
namespace runtime =
    eiel::cuda_architecture_b::compact_summary_host_template_runtime_v1;
namespace renderer =
    eiel::cuda_architecture_b::compact_summary_host_template_renderer_v1;
namespace checkpoint =
    eiel::cuda_architecture_b::compact_summary_checkpoint_runtime_v1;
namespace metadata =
    eiel::cuda_architecture_b::compact_summary_metadata_retention_v1;

namespace allocation_probe {
std::atomic<bool> enabled{false};
std::atomic<std::uint64_t> count{0};

[[gnu::noinline]] void* allocate(const std::size_t size) {
    return std::malloc(size == 0U ? 1U : size);
}

[[gnu::noinline]] void* allocate_aligned(
    const std::size_t size,
    const std::size_t alignment
) {
    void* value = nullptr;
    return posix_memalign(&value, alignment, size == 0U ? 1U : size) == 0
        ? value
        : nullptr;
}

[[gnu::noinline]] void deallocate(void* const value) noexcept {
    std::free(value);
}
}  // namespace allocation_probe

void* operator new(const std::size_t size) {
    if (allocation_probe::enabled.load(std::memory_order_relaxed)) {
        allocation_probe::count.fetch_add(1U, std::memory_order_relaxed);
    }
    if (void* const value = allocation_probe::allocate(size)) {
        return value;
    }
    throw std::bad_alloc();
}

void* operator new[](const std::size_t size) {
    return ::operator new(size);
}

void* operator new(const std::size_t size, const std::align_val_t alignment) {
    if (allocation_probe::enabled.load(std::memory_order_relaxed)) {
        allocation_probe::count.fetch_add(1U, std::memory_order_relaxed);
    }
    if (void* const value = allocation_probe::allocate_aligned(
            size,
            static_cast<std::size_t>(alignment)
        )) {
        return value;
    }
    throw std::bad_alloc();
}

void* operator new[](
    const std::size_t size,
    const std::align_val_t alignment
) {
    return ::operator new(size, alignment);
}

void operator delete(void* const value) noexcept {
    allocation_probe::deallocate(value);
}

void operator delete[](void* const value) noexcept {
    allocation_probe::deallocate(value);
}

void operator delete(void* const value, std::size_t) noexcept {
    allocation_probe::deallocate(value);
}

void operator delete[](void* const value, std::size_t) noexcept {
    allocation_probe::deallocate(value);
}

void operator delete(void* const value, std::align_val_t) noexcept {
    allocation_probe::deallocate(value);
}

void operator delete[](void* const value, std::align_val_t) noexcept {
    allocation_probe::deallocate(value);
}

void operator delete(
    void* const value,
    std::size_t,
    std::align_val_t
) noexcept {
    allocation_probe::deallocate(value);
}

void operator delete[](
    void* const value,
    std::size_t,
    std::align_val_t
) noexcept {
    allocation_probe::deallocate(value);
}

namespace {

inline constexpr std::uint32_t kGenerationCount = 2U;
inline constexpr std::uint32_t kCellCount = 1U;
inline constexpr std::int64_t kFlatDescriptorCount =
    checkpoint::kFlatDescriptorCountV1;
inline constexpr std::size_t kDigestBytes = 32U;

struct TestDescriptorPlan {
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

inline constexpr TestDescriptorPlan kFlatPlan = {
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

inline constexpr TestDescriptorPlan kPatchP2Plan = {
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

const TestDescriptorPlan& test_descriptor_plan(
    const std::int64_t descriptor_count
) {
    if (descriptor_count == kFlatPlan.descriptor_count) {
        return kFlatPlan;
    }
    if (descriptor_count == kPatchP2Plan.descriptor_count) {
        return kPatchP2Plan;
    }
    throw std::runtime_error("unsupported test descriptor count");
}

template <typename T>
std::vector<T> read_binary(const std::string& path) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) {
        throw std::runtime_error("cannot open " + path);
    }
    const std::streamsize byte_count = stream.tellg();
    if (byte_count < 0
        || byte_count % static_cast<std::streamsize>(sizeof(T)) != 0) {
        throw std::runtime_error("invalid element width in " + path);
    }
    stream.seekg(0);
    std::vector<T> values(
        static_cast<std::size_t>(byte_count) / sizeof(T)
    );
    if (byte_count != 0
        && !stream.read(reinterpret_cast<char*>(values.data()), byte_count)) {
        throw std::runtime_error("cannot read " + path);
    }
    return values;
}

void require(const bool condition, const char* const message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

runtime::HostTemplateWireTableV1 table(
    const std::vector<std::uint64_t>& values,
    const std::uint32_t columns
) {
    require(values.size() % columns == 0U, "wire table has a partial row");
    return {
        values.empty() ? nullptr : values.data(),
        values.size() / columns,
        columns,
        0U,
    };
}

bool active_prefix_descriptor(
    const TestDescriptorPlan& plan,
    const std::int64_t descriptor
) {
    return descriptor == 8
        || (descriptor >= 9 && descriptor <= plan.identity_descriptor)
        || (descriptor >= 39 && descriptor <= 44)
        || (descriptor >= plan.policy_tape_first_descriptor
            && descriptor <= plan.policy_tape_last_descriptor)
        || descriptor == plan.slot_free_queue_descriptor
        || (descriptor >= plan.typed_lifecycle_first_descriptor
            && descriptor <= plan.typed_lifecycle_last_descriptor);
}

std::int64_t row_capacity(
    const TestDescriptorPlan& plan,
    const std::int64_t descriptor
) {
    if (descriptor >= 8 && descriptor <= 10) {
        return 250;
    }
    if ((descriptor >= 39 && descriptor <= 44)
        || (descriptor >= plan.typed_lifecycle_first_descriptor
            && descriptor <= plan.typed_lifecycle_last_descriptor)) {
        return 300;
    }
    if (descriptor >= plan.policy_tape_first_descriptor
        && descriptor <= plan.policy_tape_last_descriptor) {
        return 50;
    }
    return 200;
}

std::int64_t* spec_row(
    std::vector<std::int64_t>& specs,
    const std::int64_t descriptor
) {
    return specs.data()
        + static_cast<std::size_t>(descriptor)
            * static_cast<std::size_t>(checkpoint::kDescriptorSpecWidthV1);
}

const std::int64_t* spec_row(
    const std::vector<std::int64_t>& specs,
    const std::int64_t descriptor
) {
    return specs.data()
        + static_cast<std::size_t>(descriptor)
            * static_cast<std::size_t>(checkpoint::kDescriptorSpecWidthV1);
}

std::vector<std::int64_t> make_descriptor_specs(
    const std::vector<std::uint64_t>& resolver_rows,
    const TestDescriptorPlan& plan
) {
    bool rank_two[checkpoint::kDescriptorCapacityV1]{};
    const std::size_t row_count =
        resolver_rows.size() / resolver::kResolverColumnCountV1;
    for (std::size_t row = 0; row < row_count; ++row) {
        const std::uint64_t* const source = resolver_rows.data()
            + row * resolver::kResolverColumnCountV1;
        const std::uint64_t kind =
            source[resolver::kResolverSourceKindColumnV1];
        const std::uint64_t descriptor =
            source[resolver::kResolverPrimaryIndexColumnV1];
        const std::uint64_t rank = source[resolver::kResolverRankColumnV1];
        if ((kind == 2U || kind == 4U || kind == 6U || kind == 7U
             || kind == 22U || kind == 27U)
            && descriptor < static_cast<std::uint64_t>(
                plan.descriptor_count
            )
            && rank == 2U) {
            rank_two[descriptor] = true;
        }
    }

    std::vector<std::int64_t> specs(
        static_cast<std::size_t>(plan.descriptor_count)
            * static_cast<std::size_t>(checkpoint::kDescriptorSpecWidthV1),
        0
    );
    for (std::int64_t descriptor = 0;
         descriptor < plan.descriptor_count;
         ++descriptor) {
        std::int64_t* const spec = spec_row(specs, descriptor);
        const std::int64_t dtype_bytes =
            descriptor == plan.mutation_torch_cpu_descriptor ? 1 : 8;
        const std::int64_t row_bytes = rank_two[descriptor]
            ? 3 * dtype_bytes
            : dtype_bytes;
        const std::int64_t capacity = row_capacity(plan, descriptor);
        spec[checkpoint::kCellStrideBytesV1] = capacity * row_bytes;
        spec[checkpoint::kBytesPerCellV1] = capacity * row_bytes;
        spec[checkpoint::kRowCapacityV1] = capacity;
        spec[checkpoint::kRowBytesV1] = row_bytes;
        spec[checkpoint::kDtypeBytesV1] = dtype_bytes;
        spec[checkpoint::kCanonicalizationModeV1] =
            active_prefix_descriptor(plan, descriptor)
                ? 2
                : (descriptor == plan.policy_energy_descriptor ? 1 : 0);
        spec[checkpoint::kRawOutputIndexV1] = descriptor;
        spec[checkpoint::kCanonicalOutputIndexV1] = descriptor;
        spec[checkpoint::kDescriptorIndexV1] = descriptor;
    }
    // Prove output lookup is indirect without disturbing the retained descriptor.
    spec_row(specs, 0)[checkpoint::kRawOutputIndexV1] = 1;
    spec_row(specs, 1)[checkpoint::kRawOutputIndexV1] = 0;
    spec_row(specs, 2)[checkpoint::kCanonicalOutputIndexV1] = 3;
    spec_row(specs, 3)[checkpoint::kCanonicalOutputIndexV1] = 2;
    return specs;
}

std::uint8_t digest_byte(
    const bool canonical,
    const std::uint32_t generation,
    const std::uint32_t cell,
    const std::int64_t output,
    const std::size_t byte
) {
    const std::uint64_t seed = canonical ? 0x91U : 0x17U;
    return static_cast<std::uint8_t>(
        (seed + generation * 13U + cell * 7U
         + static_cast<std::uint64_t>(output) * 3U + byte)
        & 0xffU
    );
}

std::size_t checkpoint_row(
    const std::uint32_t generation,
    const std::uint32_t cell,
    const std::int64_t output,
    const std::int64_t descriptor_count
) {
    return (static_cast<std::size_t>(generation) * kCellCount + cell)
            * static_cast<std::size_t>(descriptor_count)
        + static_cast<std::size_t>(output);
}

void fill_checkpoint_digests(
    std::vector<std::uint8_t>& raw,
    std::vector<std::uint8_t>& canonical,
    const std::int64_t descriptor_count
) {
    for (std::uint32_t generation = 0;
         generation < kGenerationCount;
         ++generation) {
        for (std::uint32_t cell = 0; cell < kCellCount; ++cell) {
            for (std::int64_t output = 0;
                 output < descriptor_count;
                 ++output) {
                const std::size_t offset =
                    checkpoint_row(
                        generation,
                        cell,
                        output,
                        descriptor_count
                    ) * kDigestBytes;
                for (std::size_t byte = 0; byte < kDigestBytes; ++byte) {
                    raw[offset + byte] = digest_byte(
                        false, generation, cell, output, byte
                    );
                    canonical[offset + byte] = digest_byte(
                        true, generation, cell, output, byte
                    );
                }
            }
        }
    }
}

std::vector<metadata::CompactSummaryMetadataRetentionV1> make_metadata(
    const std::vector<std::int64_t>& specs,
    const std::vector<std::uint8_t>& raw_digests,
    const TestDescriptorPlan& plan
) {
    std::vector<metadata::CompactSummaryMetadataRetentionV1> rows(
        kGenerationCount * kCellCount
    );
    for (std::uint32_t generation = 0;
         generation < kGenerationCount;
         ++generation) {
        for (std::uint32_t cell = 0; cell < kCellCount; ++cell) {
            auto& row = rows[generation * kCellCount + cell];
            row.source_version = metadata::kSourceVersionV1;
            row.status = metadata::kRetentionSuccessV1;
            row.validity = 1U;
            row.env_policy_alias = 1U;
            row.python_random_has_gauss = generation == 0U ? 0U : 1U;
            row.python_random_gauss_next = generation == 0U ? 0.0 : 9.999e-5;
            row.mutation_torch_descriptor_validity = 1U;
            row.flags = metadata::kRequiredRetentionFlagsV1;
            row.generation_index = generation;
            row.cell_index = cell;
            row.generation = 40 + generation;
            row.tick = 100 + generation;
            row.configured_steps_per_generation = 10;
            row.active_count = 7 + generation;
            row.capacity = 200;
            row.agent_id_to_slot_prefix_length = row.active_count;
            row.agent_id_capacity = 250;
            row.slot_free_queue_count = row.capacity - row.active_count;
            row.event_cursor = 11 + generation;
            row.event_capacity = 300;
            row.runtime_event_cursor = 31 + generation;
            row.population_next_id = 1000 + generation;
            row.policy_birth_tape_cursor = 5 + generation;
            row.policy_birth_tape_capacity = 50;
            row.generation_calls = 20 + generation;
            row.restore_count = 2;
            row.last_applied_actions = 70 + generation;
            row.last_moved = 60 + generation;
            row.last_gathered = 50 + generation;
            row.last_policy_refill_calls = 4 + generation;
            row.last_policy_refill_rows = 17 + generation;
            row.mutation_torch_cpu_state_bytes = 1234;
            row.policy_energy_observation_capacity = spec_row(
                specs,
                plan.policy_energy_descriptor
            )[checkpoint::kRowCapacityV1];
            row.policy_energy_live_count = row.active_count;
            row.policy_energy_finite_live_count = row.active_count;
            row.native_runtime_rng_state = 0x1234 + generation;
            for (std::int32_t authority = 0;
                 authority < metadata::kPcg64AuthorityCountV1;
                 ++authority) {
                for (std::int32_t word = 0;
                     word < metadata::kPcg64StateWordsV1;
                     ++word) {
                    row.pcg64[authority].state_words[word] =
                        10000U * generation + 100U * authority + word + 1U;
                }
                row.pcg64[authority].buffer_words[0] = authority % 2;
                row.pcg64[authority].buffer_words[1] =
                    200 + authority + generation;
            }
            for (std::int32_t word = 0;
                 word < metadata::kPythonMt19937WordsV1;
                 ++word) {
                row.python_random_words[word] = static_cast<std::uint32_t>(
                    generation * 1000U + static_cast<std::uint32_t>(word)
                );
            }
            row.python_random_index = 17 + generation;
            for (std::int32_t stream = 0;
                 stream < metadata::kSemanticStreamCounterCountV1;
                 ++stream) {
                row.semantic_stream_counters[stream] =
                    300 + generation * 10 + stream;
            }
            row.mutation_torch_descriptor_index =
                static_cast<std::uint32_t>(
                    plan.mutation_torch_cpu_descriptor
                );
            row.mutation_torch_raw_output_index = static_cast<std::uint32_t>(
                spec_row(specs, plan.mutation_torch_cpu_descriptor)[
                    checkpoint::kRawOutputIndexV1
                ]
            );
            row.mutation_torch_dtype_code = metadata::kDtypeUint8V1;
            row.mutation_torch_rank = metadata::kRankOneV1;
            const std::size_t mutation_offset = checkpoint_row(
                generation,
                cell,
                row.mutation_torch_raw_output_index,
                plan.descriptor_count
            ) * kDigestBytes;
            for (std::size_t byte = 0; byte < kDigestBytes; ++byte) {
                row.mutation_torch_raw_sha256[byte] =
                    raw_digests[mutation_offset + byte];
                row.policy_energy_tagged_sha256[byte] =
                    static_cast<std::uint8_t>(0xc0U + generation + byte);
            }
        }
    }
    return rows;
}

const std::uint8_t* selected_checkpoint_digest(
    const resolver::HostTokenResolverRequestV1& request,
    const std::uint64_t primary,
    const std::uint64_t selector,
    const std::uint32_t generation,
    const std::uint32_t cell
) {
    const std::int64_t* const descriptor = request.descriptor_specs
        + primary * checkpoint::kDescriptorSpecWidthV1;
    const std::int64_t output = descriptor[
        selector == 1U
            ? checkpoint::kRawOutputIndexV1
            : checkpoint::kCanonicalOutputIndexV1
    ];
    const std::uint8_t* const plane = selector == 1U
        ? request.checkpoint_raw_digests
        : request.checkpoint_canonical_digests;
    return plane
        + checkpoint_row(
            generation,
            cell,
            output,
            request.descriptor_count
        ) * kDigestBytes;
}

void check_hex(
    const runtime::HostTemplateTokenStorageV1 storage,
    const std::uint8_t* const digest
) {
    constexpr char hexadecimal[] = "0123456789abcdef";
    require(storage.sha256_hex != nullptr, "missing SHA token storage");
    require(storage.sha256_hex_capacity == 64U, "wrong SHA token capacity");
    for (std::size_t byte = 0; byte < kDigestBytes; ++byte) {
        require(
            storage.sha256_hex[byte * 2U]
                == static_cast<std::uint8_t>(hexadecimal[digest[byte] >> 4U]),
            "SHA high nibble was not lowercase hex"
        );
        require(
            storage.sha256_hex[byte * 2U + 1U]
                == static_cast<std::uint8_t>(hexadecimal[digest[byte] & 0x0fU]),
            "SHA low nibble was not lowercase hex"
        );
    }
}

std::uint64_t expected_prefix(
    const TestDescriptorPlan& plan,
    const metadata::CompactSummaryMetadataRetentionV1& row,
    const std::int64_t descriptor,
    const std::uint64_t selector,
    const std::int64_t capacity
) {
    if (selector != 2U || !active_prefix_descriptor(plan, descriptor)) {
        return static_cast<std::uint64_t>(capacity);
    }
    if (descriptor >= 8 && descriptor <= plan.identity_descriptor) {
        return static_cast<std::uint64_t>(row.active_count);
    }
    if ((descriptor >= 39 && descriptor <= 44)
        || (descriptor >= plan.typed_lifecycle_first_descriptor
            && descriptor <= plan.typed_lifecycle_last_descriptor)) {
        return static_cast<std::uint64_t>(row.event_cursor);
    }
    if (descriptor >= plan.policy_tape_first_descriptor
        && descriptor <= plan.policy_tape_last_descriptor) {
        return static_cast<std::uint64_t>(row.policy_birth_tape_cursor);
    }
    return static_cast<std::uint64_t>(row.slot_free_queue_count);
}

std::int32_t expected_pcg_slot(const std::uint64_t authority) {
    const std::int32_t slots[5] = {3, 0, 0, 2, 1};
    return slots[authority - 1U];
}

std::int32_t expected_semantic_slot(const std::uint64_t code) {
    switch (code) {
        case 2U: return metadata::kSemanticMetricsNumpyV1;
        case 5U: return metadata::kSemanticMutationPyV1;
        case 6U: return metadata::kSemanticMutationNumpyV1;
        case 7U: return metadata::kSemanticMutationTorchCpuV1;
        case 9U: return metadata::kSemanticPolicyPyV1;
        case 10U: return metadata::kSemanticPolicyNumpyV1;
        default: return -1;
    }
}

void verify_token(
    const resolver::HostTokenResolverRequestV1& request,
    const std::uint64_t* const plan,
    const std::uint32_t generation,
    const std::uint32_t cell,
    const runtime::HostTemplateTokenStorageV1 storage
) {
    const std::uint64_t kind = plan[resolver::kResolverSourceKindColumnV1];
    const std::uint64_t primary = plan[resolver::kResolverPrimaryIndexColumnV1];
    const std::uint64_t selector = plan[resolver::kResolverSecondaryCodeColumnV1];
    const std::uint64_t rank = plan[resolver::kResolverRankColumnV1];
    const TestDescriptorPlan& descriptor_plan = test_descriptor_plan(
        request.descriptor_count
    );
    const auto& row = request.metadata_rows[generation * kCellCount + cell];
    const std::int64_t* const descriptor =
        primary < static_cast<std::uint64_t>(request.descriptor_count)
        ? request.descriptor_specs
            + primary * checkpoint::kDescriptorSpecWidthV1
        : nullptr;
    require(storage.value != nullptr, "token storage missing");

    if (kind == 1U || kind == 3U || kind == 5U
        || kind == 24U || kind == 26U) {
        check_hex(
            storage,
            selected_checkpoint_digest(
                request, primary, selector, generation, cell
            )
        );
        return;
    }
    if (kind == 20U) {
        check_hex(storage, request.import_digests + cell * kDigestBytes);
        return;
    }
    if (kind == 21U) {
        check_hex(storage, row.policy_energy_tagged_sha256);
        return;
    }
    if (kind == 2U || kind == 4U || kind == 6U || kind == 7U
        || kind == 22U || kind == 27U) {
        require(descriptor != nullptr, "shape descriptor missing");
        const std::uint64_t first = expected_prefix(
            descriptor_plan,
            row,
            static_cast<std::int64_t>(primary),
            selector,
            descriptor[checkpoint::kRowCapacityV1]
        );
        require(storage.shape_dimensions != nullptr, "shape storage missing");
        require(storage.shape_dimension_capacity == rank, "shape rank drift");
        require(storage.shape_dimensions[0] == first, "shape prefix drift");
        if (rank == 2U) {
            require(
                storage.shape_dimensions[1]
                    == static_cast<std::uint64_t>(
                        descriptor[checkpoint::kRowBytesV1]
                            / descriptor[checkpoint::kDtypeBytesV1]
                    ),
                "shape inner dimension drift"
            );
        }
        return;
    }

    std::uint64_t expected_low = 0U;
    std::uint64_t expected_high = 0U;
    if (kind == 8U || kind == 9U) {
        require(descriptor != nullptr, "manifest descriptor missing");
        const std::uint64_t first = expected_prefix(
            descriptor_plan,
            row,
            static_cast<std::int64_t>(primary),
            selector,
            descriptor[checkpoint::kRowCapacityV1]
        );
        const std::uint64_t width = kind == 9U
            ? static_cast<std::uint64_t>(descriptor[checkpoint::kRowBytesV1])
            : static_cast<std::uint64_t>(
                descriptor[checkpoint::kRowBytesV1]
                    / descriptor[checkpoint::kDtypeBytesV1]
            );
        expected_low = first * width;
    } else if (kind == 10U) {
        const std::int64_t values[9] = {
            row.restore_count,
            row.generation_calls,
            row.event_cursor,
            row.runtime_event_cursor,
            row.last_applied_actions,
            row.last_moved,
            row.last_gathered,
            row.last_policy_refill_calls,
            row.last_policy_refill_rows,
        };
        expected_low = static_cast<std::uint64_t>(values[primary - 1U]);
    } else if (kind == 11U) {
        const std::int64_t values[13] = {
            row.generation,
            row.active_count,
            row.event_cursor,
            row.event_cursor,
            0,
            0,
            row.native_runtime_rng_state,
            row.policy_energy_finite_live_count,
            row.policy_energy_live_count,
            row.population_next_id,
            row.event_cursor,
            row.event_cursor,
            row.configured_steps_per_generation,
        };
        expected_low = static_cast<std::uint64_t>(values[primary - 1U]);
    } else if (kind == 12U) {
        expected_low = request.descriptor_count
                == kPatchP2Plan.descriptor_count
            ? 1U
            : 0U;
    } else if (kind >= 13U && kind <= 16U) {
        const auto& pcg = row.pcg64[expected_pcg_slot(primary)];
        if (kind == 13U || kind == 14U) {
            const std::int32_t offset = kind == 13U ? 0 : 2;
            expected_low = pcg.state_words[offset];
            expected_high = pcg.state_words[offset + 1];
        } else {
            expected_low = static_cast<std::uint64_t>(
                pcg.buffer_words[kind == 15U ? 0 : 1]
            );
        }
    } else if (kind == 17U) {
        expected_low = row.python_random_words[primary];
    } else if (kind == 18U) {
        expected_low = static_cast<std::uint64_t>(row.python_random_index);
    } else if (kind == 19U) {
        const std::int32_t retained = expected_semantic_slot(primary);
        expected_low = retained < 0
            ? 0U
            : static_cast<std::uint64_t>(
                row.semantic_stream_counters[retained]
            );
    } else if (kind == 23U) {
        expected_low = static_cast<std::uint64_t>(row.active_count);
    } else if (kind == 25U) {
        expected_low = static_cast<std::uint64_t>(
            row.policy_birth_tape_cursor
        );
    } else if (kind == 28U) {
        require(
            storage.value->token_type == static_cast<std::uint32_t>(
                renderer::HostTemplateTokenTypeV1::kNullableFloat64
            ),
            "gauss cache token type drift"
        );
        expected_high = row.python_random_has_gauss;
        if (expected_high == 1U) {
            std::memcpy(
                &expected_low,
                &row.python_random_gauss_next,
                sizeof(row.python_random_gauss_next)
            );
        }
    } else {
        throw std::runtime_error("integer source kind was not verified");
    }
    require(storage.value->scalar_low == expected_low, "scalar low word drift");
    require(storage.value->scalar_high == expected_high, "scalar high word drift");
}

std::uint64_t runtime_fingerprint(runtime::HostTemplateRuntimeV1& prepared) {
    std::uint64_t hash = 1469598103934665603ULL;
    const auto mix = [&hash](const std::uint64_t value) {
        hash ^= value;
        hash *= 1099511628211ULL;
    };
    for (std::uint32_t generation = 0;
         generation < kGenerationCount;
         ++generation) {
        for (std::uint32_t surface = 1U; surface <= 6U; ++surface) {
            const auto binding = runtime::lookup_host_template_program_v1(
                &prepared, generation, 0U, surface
            );
            for (std::uint32_t token = 0;
                 token < binding.token_value_count;
                 ++token) {
                const auto storage = runtime::lookup_host_template_token_storage_v1(
                    &prepared, generation, 0U, surface, token
                );
                mix(storage.value->token_type);
                mix(storage.value->scalar_low);
                mix(storage.value->scalar_high);
                for (std::uint32_t dimension = 0;
                     dimension < storage.shape_dimension_capacity;
                     ++dimension) {
                    mix(storage.shape_dimensions[dimension]);
                }
                for (std::uint32_t byte = 0;
                     byte < storage.sha256_hex_capacity;
                     ++byte) {
                    mix(storage.sha256_hex[byte]);
                }
            }
        }
    }
    return hash;
}

void verify_all_sources_and_render(
    const resolver::HostTokenResolverRequestV1& request
) {
    bool seen[29]{};
    bool saw_event_tape_inactive_tail = false;
    std::uint64_t resolver_row = 0;
    for (std::uint32_t generation = 0;
         generation < kGenerationCount;
         ++generation) {
        for (std::uint32_t surface = 1U; surface <= 6U; ++surface) {
            const auto binding = runtime::lookup_host_template_program_v1(
                request.host_template_runtime, generation, 0U, surface
            );
            for (std::uint32_t token = 0;
                 token < binding.token_value_count;
                 ++token, ++resolver_row) {
                const std::uint64_t* const plan = request.resolver_rows
                    + resolver_row * resolver::kResolverColumnCountV1;
                const std::uint64_t kind =
                    plan[resolver::kResolverSourceKindColumnV1];
                require(kind >= 1U && kind <= 28U, "source kind out of range");
                seen[kind] = true;
                const auto storage =
                    runtime::lookup_host_template_token_storage_v1(
                        request.host_template_runtime,
                        generation,
                        0U,
                        surface,
                        token
                    );
                verify_token(
                    request,
                    plan,
                    generation,
                    0U,
                    storage
                );
                if (kind == 11U
                    && plan[resolver::kResolverPrimaryIndexColumnV1] == 5U) {
                    const auto& row = request.metadata_rows[
                        generation * kCellCount
                    ];
                    require(
                        storage.value->scalar_low == 0U,
                        "compact checkpoint retained an inactive event-tape tail"
                    );
                    require(
                        row.event_capacity != row.capacity,
                        "focused inactive-tail fixture does not distinguish capacities"
                    );
                    saw_event_tape_inactive_tail = true;
                }
            }
            renderer::HostTemplateResultV1 result{};
            renderer::HostTemplateRequestV1 render{};
            render.abi_version = renderer::kHostTemplateRendererAbiVersionV1;
            render.mode = static_cast<std::uint32_t>(
                renderer::HostTemplateRenderModeV1::kHashOnly
            );
            render.program = binding.program;
            render.token_values = binding.token_values;
            render.token_value_count = binding.token_value_count;
            render.result = &result;
            require(
                renderer::render_and_hash_host_template_v1(&render)
                    == renderer::HostTemplateStatusV1::kSuccess,
                "resolved program did not satisfy renderer constraints"
            );
        }
    }
    require(resolver_row == request.resolver_row_count, "resolver traversal drift");
    for (std::uint64_t kind = 1U; kind <= 28U; ++kind) {
        require(seen[kind], "one source kind was not covered");
    }
    require(
        saw_event_tape_inactive_tail,
        "projection metadata code 5 was not covered"
    );
}

void expect_status(
    resolver::HostTokenResolverRequestV1& request,
    const resolver::HostTokenResolverStatusV1 expected,
    const char* const message
) {
    const auto actual = resolver::resolve_compact_summary_host_tokens_v1(&request);
    if (actual != expected) {
        std::cerr
            << message << ": expected="
            << resolver::host_token_resolver_status_name_v1(expected)
            << " actual="
            << resolver::host_token_resolver_status_name_v1(actual)
            << " component=" << request.failure->component
            << " row=" << request.failure->resolver_row
            << " descriptor=" << request.failure->descriptor_index
            << '\n';
        throw std::runtime_error(message);
    }
}

std::size_t find_resolver_kind(
    const std::vector<std::uint64_t>& rows,
    const std::uint64_t kind
) {
    for (std::size_t row = 0;
         row < rows.size() / resolver::kResolverColumnCountV1;
         ++row) {
        if (rows[row * resolver::kResolverColumnCountV1
                 + resolver::kResolverSourceKindColumnV1] == kind) {
            return row;
        }
    }
    throw std::runtime_error("resolver source kind not found");
}

std::uint32_t verify_patch_p2_descriptor_mapping(
    runtime::HostTemplateRuntimeV1& prepared,
    std::vector<std::uint64_t>& resolver_rows
) {
    std::vector<std::int64_t> specs = make_descriptor_specs(
        resolver_rows,
        kPatchP2Plan
    );
    const std::size_t checkpoint_rows =
        kGenerationCount * kCellCount
        * static_cast<std::size_t>(kPatchP2Plan.descriptor_count);
    std::vector<std::uint8_t> raw(checkpoint_rows * kDigestBytes);
    std::vector<std::uint8_t> canonical(checkpoint_rows * kDigestBytes);
    fill_checkpoint_digests(
        raw,
        canonical,
        kPatchP2Plan.descriptor_count
    );
    std::vector<std::int32_t> statuses(checkpoint_rows, 0);
    std::vector<std::uint8_t> validity(checkpoint_rows, 1U);
    std::vector<metadata::CompactSummaryMetadataRetentionV1> metadata_rows =
        make_metadata(specs, raw, kPatchP2Plan);
    std::vector<std::uint8_t> import_digests(kCellCount * kDigestBytes);
    for (std::size_t byte = 0; byte < import_digests.size(); ++byte) {
        import_digests[byte] = static_cast<std::uint8_t>(0x70U + byte);
    }

    resolver::HostTokenResolverFailureV1 failure{};
    resolver::HostTokenResolverRequestV1 request = {
        resolver_rows.data(),
        resolver_rows.size() / resolver::kResolverColumnCountV1,
        &prepared,
        specs.data(),
        raw.data(),
        canonical.data(),
        statuses.data(),
        validity.data(),
        metadata_rows.data(),
        import_digests.data(),
        kGenerationCount,
        kGenerationCount,
        kCellCount,
        kPatchP2Plan.descriptor_count,
        &failure,
    };

    expect_status(
        request,
        resolver::HostTokenResolverStatusV1::kSuccess,
        "valid Patch P2 resolver request"
    );
    verify_all_sources_and_render(request);
    const std::uint64_t settled_fingerprint = runtime_fingerprint(prepared);
    std::uint32_t malformed_count = 0U;

    request.descriptor_count = checkpoint::kFlatDescriptorCountV1 + 1;
    expect_status(
        request,
        resolver::HostTokenResolverStatusV1::kInvalidDimensions,
        "unsupported descriptor count 133"
    );
    ++malformed_count;
    request.descriptor_count = checkpoint::kPatchP2DescriptorCountV1 - 1;
    expect_status(
        request,
        resolver::HostTokenResolverStatusV1::kInvalidDimensions,
        "unsupported descriptor count 143"
    );
    ++malformed_count;
    request.descriptor_count = kPatchP2Plan.descriptor_count;

    const auto resolver_primary_offset = [&resolver_rows](
        const std::uint64_t kind
    ) {
        return find_resolver_kind(resolver_rows, kind)
                * resolver::kResolverColumnCountV1
            + resolver::kResolverPrimaryIndexColumnV1;
    };

    const std::size_t energy_primary = resolver_primary_offset(21U);
    const std::uint64_t saved_energy = resolver_rows[energy_primary];
    resolver_rows[energy_primary] = 98U;
    expect_status(
        request,
        resolver::HostTokenResolverStatusV1::kResolverContradiction,
        "Patch energy descriptor mapped to Flat index"
    );
    resolver_rows[energy_primary] = saved_energy;
    ++malformed_count;

    const std::size_t identity_primary = resolver_primary_offset(24U);
    const std::uint64_t saved_identity = resolver_rows[identity_primary];
    resolver_rows[identity_primary] = 12U;
    expect_status(
        request,
        resolver::HostTokenResolverStatusV1::kResolverContradiction,
        "Patch identity descriptor mapping drift"
    );
    resolver_rows[identity_primary] = saved_identity;
    ++malformed_count;

    const std::size_t policy_tape_primary = resolver_primary_offset(26U);
    const std::uint64_t saved_policy_tape = resolver_rows[policy_tape_primary];
    resolver_rows[policy_tape_primary] = 92U;
    expect_status(
        request,
        resolver::HostTokenResolverStatusV1::kResolverContradiction,
        "Patch policy-tape descriptor mapped to pre-shift index"
    );
    resolver_rows[policy_tape_primary] = saved_policy_tape;
    ++malformed_count;

    std::int64_t* const policy_tape_spec = spec_row(
        specs,
        kPatchP2Plan.policy_tape_first_descriptor
    );
    policy_tape_spec[checkpoint::kCanonicalizationModeV1] = 0;
    expect_status(
        request,
        resolver::HostTokenResolverStatusV1::kDescriptorContradiction,
        "Patch active-prefix mode drift"
    );
    policy_tape_spec[checkpoint::kCanonicalizationModeV1] = 2;
    ++malformed_count;

    const std::uint32_t saved_mutation_descriptor =
        metadata_rows[0].mutation_torch_descriptor_index;
    metadata_rows[0].mutation_torch_descriptor_index =
        metadata::kMutationTorchCpuDescriptorIndexV1;
    expect_status(
        request,
        resolver::HostTokenResolverStatusV1::kMetadataInvalid,
        "Patch retained mutation descriptor mapped to Flat index"
    );
    metadata_rows[0].mutation_torch_descriptor_index =
        saved_mutation_descriptor;
    ++malformed_count;

    require(
        runtime_fingerprint(prepared) == settled_fingerprint,
        "malformed Patch request partially updated token storage"
    );
    expect_status(
        request,
        resolver::HostTokenResolverStatusV1::kSuccess,
        "restored valid Patch P2 request"
    );
    require(
        runtime_fingerprint(prepared) == settled_fingerprint,
        "restored Patch request changed valid output"
    );
    return malformed_count;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        require(argc == 2, "expected one fixture directory argument");
        const std::string root = argv[1];
        const std::vector<std::uint64_t> programs =
            read_binary<std::uint64_t>(root + "/programs.bin");
        const std::vector<std::uint64_t> tokens =
            read_binary<std::uint64_t>(root + "/tokens.bin");
        const std::vector<std::uint64_t> parts =
            read_binary<std::uint64_t>(root + "/parts.bin");
        const std::vector<std::uint64_t> constraints =
            read_binary<std::uint64_t>(root + "/constraints.bin");
        const std::vector<std::uint8_t> byte_pool =
            read_binary<std::uint8_t>(root + "/byte_pool.bin");
        std::vector<std::uint64_t> resolver_rows =
            read_binary<std::uint64_t>(root + "/resolver.bin");
        std::vector<std::uint64_t> patch_resolver_rows =
            read_binary<std::uint64_t>(root + "/resolver_patch_p2.bin");
        require(
            resolver_rows.size() % resolver::kResolverColumnCountV1 == 0U,
            "resolver tensor has a partial row"
        );
        require(
            patch_resolver_rows.size() == resolver_rows.size(),
            "Patch resolver tensor row count drift"
        );

        const runtime::HostTemplateWireViewV1 wire = {
            runtime::kHostTemplateWireAbiVersionV1,
            0U,
            kGenerationCount,
            kCellCount,
            table(programs, runtime::kProgramWireColumnCountV1),
            table(tokens, runtime::kTokenWireColumnCountV1),
            table(parts, runtime::kPartWireColumnCountV1),
            table(constraints, runtime::kConstraintWireColumnCountV1),
            {
                byte_pool.empty() ? nullptr : byte_pool.data(),
                byte_pool.size(),
                0U,
            },
            0U,
        };
        runtime::HostTemplateRuntimeV1 prepared;
        require(
            runtime::prepare_host_template_runtime_v1(wire, &prepared)
                == runtime::HostTemplateRuntimeStatusV1::kSuccess,
            "host template runtime preparation failed"
        );

        std::vector<std::int64_t> specs = make_descriptor_specs(
            resolver_rows,
            kFlatPlan
        );
        const std::size_t checkpoint_rows =
            kGenerationCount * kCellCount
            * static_cast<std::size_t>(kFlatDescriptorCount);
        std::vector<std::uint8_t> raw(checkpoint_rows * kDigestBytes);
        std::vector<std::uint8_t> canonical(checkpoint_rows * kDigestBytes);
        fill_checkpoint_digests(raw, canonical, kFlatDescriptorCount);
        std::vector<std::int32_t> statuses(checkpoint_rows, 0);
        std::vector<std::uint8_t> validity(checkpoint_rows, 1U);
        std::vector<metadata::CompactSummaryMetadataRetentionV1> metadata_rows =
            make_metadata(specs, raw, kFlatPlan);
        std::vector<std::uint8_t> import_digests(kCellCount * kDigestBytes);
        for (std::size_t byte = 0; byte < import_digests.size(); ++byte) {
            import_digests[byte] = static_cast<std::uint8_t>(0x60U + byte);
        }

        resolver::HostTokenResolverFailureV1 failure{};
        resolver::HostTokenResolverRequestV1 request = {
            resolver_rows.data(),
            resolver_rows.size() / resolver::kResolverColumnCountV1,
            &prepared,
            specs.data(),
            raw.data(),
            canonical.data(),
            statuses.data(),
            validity.data(),
            metadata_rows.data(),
            import_digests.data(),
            kGenerationCount,
            kGenerationCount,
            kCellCount,
            kFlatDescriptorCount,
            &failure,
        };

        expect_status(
            request,
            resolver::HostTokenResolverStatusV1::kSuccess,
            "valid resolver request"
        );
        verify_all_sources_and_render(request);
        const std::uint64_t settled_fingerprint = runtime_fingerprint(prepared);

        allocation_probe::count.store(0U, std::memory_order_relaxed);
        allocation_probe::enabled.store(true, std::memory_order_relaxed);
        const auto repeat_one = resolver::resolve_compact_summary_host_tokens_v1(
            &request
        );
        const auto repeat_two = resolver::resolve_compact_summary_host_tokens_v1(
            &request
        );
        allocation_probe::enabled.store(false, std::memory_order_relaxed);
        require(
            repeat_one == resolver::HostTokenResolverStatusV1::kSuccess
                && repeat_two == resolver::HostTokenResolverStatusV1::kSuccess,
            "repeat resolve failed"
        );
        require(
            allocation_probe::count.load(std::memory_order_relaxed) == 0U,
            "resolve allocated host storage"
        );
        require(
            runtime_fingerprint(prepared) == settled_fingerprint,
            "repeat resolve was not deterministic"
        );

        std::uint32_t malformed_count = 0U;
        --request.resolver_row_count;
        expect_status(
            request,
            resolver::HostTokenResolverStatusV1::kResolverRowCountMismatch,
            "short resolver plan"
        );
        ++request.resolver_row_count;
        ++malformed_count;

        const std::size_t last_row = request.resolver_row_count - 1U;
        const std::size_t last_abi =
            last_row * resolver::kResolverColumnCountV1
            + resolver::kResolverAbiVersionColumnV1;
        const std::uint64_t saved_abi = resolver_rows[last_abi];
        resolver_rows[last_abi] = 2U;
        expect_status(
            request,
            resolver::HostTokenResolverStatusV1::kResolverAbiMismatch,
            "resolver ABI drift"
        );
        require(
            runtime_fingerprint(prepared) == settled_fingerprint,
            "malformed late row partially updated token storage"
        );
        resolver_rows[last_abi] = saved_abi;
        ++malformed_count;

        const std::size_t first_reserved =
            resolver::kResolverReservedZeroColumnV1;
        resolver_rows[first_reserved] = 1U;
        expect_status(
            request,
            resolver::HostTokenResolverStatusV1::kResolverReservedNonzero,
            "resolver reserved field"
        );
        resolver_rows[first_reserved] = 0U;
        ++malformed_count;

        const std::size_t first_kind = resolver::kResolverSourceKindColumnV1;
        const std::uint64_t saved_kind = resolver_rows[first_kind];
        resolver_rows[first_kind] = 99U;
        expect_status(
            request,
            resolver::HostTokenResolverStatusV1::kUnsupportedSourceKind,
            "unknown source kind"
        );
        resolver_rows[first_kind] = saved_kind;
        ++malformed_count;

        const std::size_t digest_row = find_resolver_kind(resolver_rows, 1U);
        const std::size_t digest_selector =
            digest_row * resolver::kResolverColumnCountV1
            + resolver::kResolverSecondaryCodeColumnV1;
        const std::uint64_t saved_selector = resolver_rows[digest_selector];
        resolver_rows[digest_selector] = 0U;
        expect_status(
            request,
            resolver::HostTokenResolverStatusV1::kResolverContradiction,
            "invalid digest selector"
        );
        resolver_rows[digest_selector] = saved_selector;
        ++malformed_count;

        const std::size_t gauss_row = find_resolver_kind(resolver_rows, 28U);
        const std::size_t gauss_primary =
            gauss_row * resolver::kResolverColumnCountV1
            + resolver::kResolverPrimaryIndexColumnV1;
        resolver_rows[gauss_primary] = 1U;
        expect_status(
            request,
            resolver::HostTokenResolverStatusV1::kResolverContradiction,
            "invalid gauss cache declaration"
        );
        resolver_rows[gauss_primary] = 0U;
        ++malformed_count;

        std::int64_t* const descriptor_zero = spec_row(specs, 0);
        const std::int64_t saved_descriptor =
            descriptor_zero[checkpoint::kDescriptorIndexV1];
        descriptor_zero[checkpoint::kDescriptorIndexV1] = 1;
        expect_status(
            request,
            resolver::HostTokenResolverStatusV1::kDescriptorContradiction,
            "descriptor index drift"
        );
        descriptor_zero[checkpoint::kDescriptorIndexV1] = saved_descriptor;
        ++malformed_count;

        statuses[0] = 1;
        expect_status(
            request,
            resolver::HostTokenResolverStatusV1::kCheckpointInvalid,
            "checkpoint status"
        );
        statuses[0] = 0;
        ++malformed_count;

        validity[0] = 0U;
        expect_status(
            request,
            resolver::HostTokenResolverStatusV1::kCheckpointInvalid,
            "checkpoint validity"
        );
        validity[0] = 1U;
        ++malformed_count;

        metadata_rows[0].status = metadata::kRetentionUpstreamFailureV1;
        expect_status(
            request,
            resolver::HostTokenResolverStatusV1::kMetadataInvalid,
            "metadata status"
        );
        metadata_rows[0].status = metadata::kRetentionSuccessV1;
        ++malformed_count;

        metadata_rows[0].validity = 0U;
        expect_status(
            request,
            resolver::HostTokenResolverStatusV1::kMetadataInvalid,
            "metadata validity"
        );
        metadata_rows[0].validity = 1U;
        ++malformed_count;

        const std::uint32_t saved_flags = metadata_rows[0].flags;
        metadata_rows[0].flags &= ~metadata::kRetentionExportNonConsumingV1;
        expect_status(
            request,
            resolver::HostTokenResolverStatusV1::kMetadataInvalid,
            "metadata required flags"
        );
        metadata_rows[0].flags = saved_flags;
        ++malformed_count;

        metadata_rows[0].env_policy_alias = 0U;
        expect_status(
            request,
            resolver::HostTokenResolverStatusV1::kMetadataInvalid,
            "env policy alias"
        );
        metadata_rows[0].env_policy_alias = 1U;
        ++malformed_count;

        metadata_rows[0].python_random_has_gauss = 1U;
        metadata_rows[0].python_random_gauss_next =
            std::numeric_limits<double>::infinity();
        expect_status(
            request,
            resolver::HostTokenResolverStatusV1::kMetadataInvalid,
            "non-finite python gauss cache"
        );
        metadata_rows[0].python_random_has_gauss = 2U;
        metadata_rows[0].python_random_gauss_next = 0.0;
        expect_status(
            request,
            resolver::HostTokenResolverStatusV1::kMetadataInvalid,
            "invalid python gauss presence"
        );
        metadata_rows[0].python_random_has_gauss = 0U;
        malformed_count += 2U;

        const std::int64_t saved_capacity =
            descriptor_zero[checkpoint::kRowCapacityV1];
        const std::int64_t saved_row_bytes =
            descriptor_zero[checkpoint::kRowBytesV1];
        descriptor_zero[checkpoint::kRowCapacityV1] =
            std::numeric_limits<std::int64_t>::max();
        descriptor_zero[checkpoint::kRowBytesV1] = 3;
        expect_status(
            request,
            resolver::HostTokenResolverStatusV1::kDescriptorContradiction,
            "descriptor arithmetic overflow"
        );
        descriptor_zero[checkpoint::kRowCapacityV1] = saved_capacity;
        descriptor_zero[checkpoint::kRowBytesV1] = saved_row_bytes;
        ++malformed_count;

        expect_status(
            request,
            resolver::HostTokenResolverStatusV1::kSuccess,
            "restored valid request"
        );
        require(
            runtime_fingerprint(prepared) == settled_fingerprint,
            "restored request changed valid output"
        );
        const std::uint32_t patch_malformed_count =
            verify_patch_p2_descriptor_mapping(
                prepared,
                patch_resolver_rows
            );

        std::cout
            << "status=success source_kinds=28 resolver_rows="
            << request.resolver_row_count
            << " malformed=" << malformed_count
            << " patch_malformed=" << patch_malformed_count
            << " allocation_count="
            << allocation_probe::count.load(std::memory_order_relaxed)
            << '\n';
        return 0;
    } catch (const std::exception& error) {
        allocation_probe::enabled.store(false, std::memory_order_relaxed);
        std::cerr << error.what() << '\n';
        return 2;
    }
}
