#include "compact_reducer_core_v1.h"
#include "compact_reducer_internal_v1.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>

namespace {

using BirthReceipt = eiel_native_compact_birth_batch_receipt_v1;
using Status = eiel_native_compact_reducer_status_v1;

constexpr std::size_t kPolicyValues = 224U;
constexpr std::size_t kPolicyLociPerHaplotype = 112U;
constexpr std::size_t kTraitLociPerHaplotype = 16U;
constexpr std::size_t kTraitOffset = kPolicyValues;
constexpr std::size_t kGenomeBytes = EIEL_COMPACT_GENOME_ROW_BYTES_V1;
constexpr std::size_t kProvenanceBytes =
    EIEL_COMPACT_PACKED_BIRTH_PROVENANCE_BYTES_V1;
constexpr std::size_t kLinkedLoci = EIEL_COMPACT_LINKED_LOCI_V1;
constexpr std::size_t kMetricFields = EIEL_COMPACT_DEATH_METRIC_FIELDS_V1;
constexpr std::size_t kMetricRowBytes = EIEL_COMPACT_DEATH_METRIC_ROW_BYTES_V1;

static_assert(std::is_standard_layout_v<BirthReceipt>);
static_assert(std::is_trivially_copyable_v<BirthReceipt>);
static_assert(sizeof(std::int8_t) == 1U);
static_assert(sizeof(std::int16_t) == 2U);

[[nodiscard]] std::uint32_t read_u32_le(const std::uint8_t* const source) noexcept {
    return static_cast<std::uint32_t>(source[0])
        | (static_cast<std::uint32_t>(source[1]) << 8U)
        | (static_cast<std::uint32_t>(source[2]) << 16U)
        | (static_cast<std::uint32_t>(source[3]) << 24U);
}

[[nodiscard]] std::uint64_t read_u64_le(const std::uint8_t* const source) noexcept {
    std::uint64_t value = 0U;
    for (std::size_t byte = 0U; byte < 8U; ++byte) {
        value |= static_cast<std::uint64_t>(source[byte]) << (byte * 8U);
    }
    return value;
}

[[nodiscard]] std::int64_t read_i64_le(const std::uint8_t* const source) noexcept {
    const std::uint64_t bits = read_u64_le(source);
    std::int64_t value = 0;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

[[nodiscard]] std::int8_t read_i8(const std::uint8_t* const source) noexcept {
    std::int8_t value = 0;
    std::memcpy(&value, source, sizeof(value));
    return value;
}

[[nodiscard]] std::int16_t read_i16_le(const std::uint8_t* const source) noexcept {
    const std::uint16_t bits = static_cast<std::uint16_t>(source[0])
        | static_cast<std::uint16_t>(static_cast<std::uint16_t>(source[1]) << 8U);
    std::int16_t value = 0;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

void write_i8(std::uint8_t* const destination, const std::int8_t value) noexcept {
    std::memcpy(destination, &value, sizeof(value));
}

void write_i16_le(
    std::uint8_t* const destination, const std::int16_t value) noexcept {
    std::uint16_t bits = 0U;
    std::memcpy(&bits, &value, sizeof(bits));
    destination[0] = static_cast<std::uint8_t>(bits & 0xffU);
    destination[1] = static_cast<std::uint8_t>((bits >> 8U) & 0xffU);
}

[[nodiscard]] bool add_u64_checked(
    std::uint64_t* const target, const std::uint64_t value) noexcept {
    if (*target > std::numeric_limits<std::uint64_t>::max() - value) return false;
    *target += value;
    return true;
}

[[nodiscard]] bool add_i64_checked(
    std::int64_t* const target, const std::int64_t value) noexcept {
    if ((value > 0 && *target > std::numeric_limits<std::int64_t>::max() - value)
        || (value < 0 && *target < std::numeric_limits<std::int64_t>::min() - value)) {
        return false;
    }
    *target += value;
    return true;
}

}  // namespace

namespace eiel_compact_reducer_internal_v1 {

Status reconstruct_birth(
    const std::uint8_t* const parent_a,
    const std::uint8_t* const parent_b,
    const std::uint8_t* const provenance,
    const std::int32_t mutation_step,
    std::uint8_t* const child,
    BirthReceipt* const receipt) noexcept {
    if (mutation_step < 1) return EIEL_NATIVE_COMPACT_REDUCER_INVALID_MUTATION_STEP_V1;
    const std::uint32_t recombination = read_u32_le(provenance + 64U);
    std::uint8_t staged_child[kGenomeBytes]{};
    BirthReceipt staged_receipt = *receipt;
    if (!add_u64_checked(&staged_receipt.reconstructed_birth_count, 1U)) {
        return EIEL_NATIVE_COMPACT_REDUCER_COUNTER_OVERFLOW_V1;
    }
    if (recombination != 0U
        && !add_u64_checked(&staged_receipt.recombination_nonzero_pattern_count, 1U)) {
        return EIEL_NATIVE_COMPACT_REDUCER_COUNTER_OVERFLOW_V1;
    }
    for (std::size_t choice = 0U;
         choice < EIEL_COMPACT_RECOMBINATION_CHOICES_V1; ++choice) {
        if (((recombination >> choice) & 1U) != 0U
            && !add_u64_checked(
                &staged_receipt.recombination_choice_one_counts[choice], 1U)) {
            return EIEL_NATIVE_COMPACT_REDUCER_COUNTER_OVERFLOW_V1;
        }
    }
    for (std::size_t linked_locus = 0U; linked_locus < kLinkedLoci;
         ++linked_locus) {
        const std::size_t block = linked_locus / 8U;
        const std::size_t haplotype_a =
            static_cast<std::size_t>((recombination >> (2U * block)) & 1U);
        const std::size_t haplotype_b =
            static_cast<std::size_t>((recombination >> (2U * block + 1U)) & 1U);
        const std::uint8_t packed_codes = provenance[linked_locus / 2U];
        const std::uint8_t mutation_codes = static_cast<std::uint8_t>(
            linked_locus % 2U == 0U ? packed_codes & 0x0fU : packed_codes >> 4U);
        const bool policy = linked_locus < kPolicyLociPerHaplotype;
        const std::size_t locus = policy
            ? linked_locus
            : linked_locus - kPolicyLociPerHaplotype;
        const std::size_t loci_per_haplotype = policy
            ? kPolicyLociPerHaplotype
            : kTraitLociPerHaplotype;
        const std::int64_t minimum = policy ? -127 : -32767;
        const std::int64_t maximum = policy ? 127 : 32767;
        for (std::size_t allele = 0U; allele < 2U; ++allele) {
            const std::uint8_t code = static_cast<std::uint8_t>(
                (mutation_codes >> (2U * allele)) & 0x03U);
            if (code == 3U) {
                return EIEL_NATIVE_COMPACT_REDUCER_RESERVED_MUTATION_CODE_V1;
            }
            const std::int64_t direction = code == 0U ? 0 : (code == 1U ? 1 : -1);
            const std::size_t haplotype = allele == 0U ? haplotype_a : haplotype_b;
            const std::size_t source_index = haplotype * loci_per_haplotype + locus;
            const std::size_t destination_index = allele * loci_per_haplotype + locus;
            const std::uint8_t* const parent = allele == 0U ? parent_a : parent_b;
            std::int64_t before = 0;
            if (policy) {
                before = static_cast<std::int64_t>(read_i8(parent + source_index));
            } else {
                before = static_cast<std::int64_t>(read_i16_le(
                    parent + kTraitOffset + source_index * 2U));
            }
            const std::int64_t candidate = before
                + direction * static_cast<std::int64_t>(mutation_step);
            const std::int64_t after = std::max(minimum, std::min(maximum, candidate));
            if (policy) {
                write_i8(
                    staged_child + destination_index,
                    static_cast<std::int8_t>(after));
            } else {
                write_i16_le(
                    staged_child + kTraitOffset + destination_index * 2U,
                    static_cast<std::int16_t>(after));
            }
            if (direction != 0) {
                if (!add_u64_checked(&staged_receipt.mutation_origin_count, 1U)
                    || !add_u64_checked(
                        &staged_receipt
                             .mutation_origin_counts_by_linked_locus[linked_locus],
                        1U)
                    || !add_i64_checked(
                        &staged_receipt
                             .mutation_delta_totals_by_linked_locus[linked_locus],
                        after - before)) {
                    return EIEL_NATIVE_COMPACT_REDUCER_COUNTER_OVERFLOW_V1;
                }
                if (after == before
                    && !add_u64_checked(&staged_receipt.saturated_noop_count, 1U)) {
                    return EIEL_NATIVE_COMPACT_REDUCER_COUNTER_OVERFLOW_V1;
                }
            }
        }
    }
    std::memcpy(child, staged_child, kGenomeBytes);
    *receipt = staged_receipt;
    return EIEL_NATIVE_COMPACT_REDUCER_SUCCESS_V1;
}

}  // namespace eiel_compact_reducer_internal_v1

namespace {

void store_i128(
    const __int128 value, eiel_native_compact_i128_v1* const output) noexcept {
    const unsigned __int128 bits = static_cast<unsigned __int128>(value);
    output->low = static_cast<std::uint64_t>(bits);
    const std::uint64_t high_bits = static_cast<std::uint64_t>(bits >> 64U);
    std::memcpy(&output->high, &high_bits, sizeof(high_bits));
}

}  // namespace

extern "C" {

std::uint32_t eiel_native_compact_reducer_abi_version_v1(void) {
    return EIEL_NATIVE_COMPACT_REDUCER_ABI_VERSION_V1;
}

std::int64_t eiel_native_compact_reducer_birth_receipt_size_v1(void) {
    return static_cast<std::int64_t>(sizeof(BirthReceipt));
}

std::int64_t eiel_native_compact_reducer_active_record_size_v1(void) {
    return static_cast<std::int64_t>(
        sizeof(eiel_native_compact_active_record_v1));
}

std::int64_t eiel_native_compact_reducer_world_receipt_size_v1(void) {
    return static_cast<std::int64_t>(
        sizeof(eiel_native_compact_world_receipt_v1));
}

std::int64_t eiel_native_compact_reducer_strategy_receipt_size_v1(void) {
    return static_cast<std::int64_t>(
        sizeof(eiel_native_compact_strategy_receipt_v1));
}

std::int32_t eiel_native_compact_reconstruct_birth_v1(
    const std::uint8_t* const parent_a_genome,
    const std::uint8_t* const parent_b_genome,
    const std::uint8_t* const packed_birth_provenance,
    const std::int32_t mutation_step,
    std::uint8_t* const child_genome,
    BirthReceipt* const receipt) {
    if (parent_a_genome == nullptr || parent_b_genome == nullptr
        || packed_birth_provenance == nullptr || child_genome == nullptr
        || receipt == nullptr) {
        return EIEL_NATIVE_COMPACT_REDUCER_NULL_POINTER_V1;
    }
    BirthReceipt empty{};
    *receipt = empty;
    return eiel_compact_reducer_internal_v1::reconstruct_birth(
        parent_a_genome, parent_b_genome, packed_birth_provenance,
        mutation_step, child_genome, receipt);
}

std::int32_t eiel_native_compact_reconstruct_birth_batch_v1(
    const std::uint8_t* const parent_a_genomes,
    const std::uint8_t* const parent_b_genomes,
    const std::uint8_t* const packed_birth_provenance_rows,
    const std::int32_t* const mutation_steps,
    const std::int64_t row_count,
    std::uint8_t* const child_genomes,
    BirthReceipt* const receipt) {
    if (receipt == nullptr) return EIEL_NATIVE_COMPACT_REDUCER_NULL_POINTER_V1;
    BirthReceipt empty{};
    *receipt = empty;
    if (row_count < 0) return EIEL_NATIVE_COMPACT_REDUCER_INVALID_EXTENT_V1;
    if (row_count == 0) return EIEL_NATIVE_COMPACT_REDUCER_SUCCESS_V1;
    if (parent_a_genomes == nullptr || parent_b_genomes == nullptr
        || packed_birth_provenance_rows == nullptr || mutation_steps == nullptr
        || child_genomes == nullptr) {
        return EIEL_NATIVE_COMPACT_REDUCER_NULL_POINTER_V1;
    }
    for (std::int64_t row = 0; row < row_count; ++row) {
        const auto index = static_cast<std::size_t>(row);
        const Status status = eiel_compact_reducer_internal_v1::reconstruct_birth(
            parent_a_genomes + index * kGenomeBytes,
            parent_b_genomes + index * kGenomeBytes,
            packed_birth_provenance_rows + index * kProvenanceBytes,
            mutation_steps[index], child_genomes + index * kGenomeBytes,
            receipt);
        if (status != EIEL_NATIVE_COMPACT_REDUCER_SUCCESS_V1) {
            BirthReceipt reset{};
            *receipt = reset;
            return status;
        }
    }
    return EIEL_NATIVE_COMPACT_REDUCER_SUCCESS_V1;
}

std::int32_t eiel_native_compact_accumulate_death_metrics_v1(
    const std::uint8_t* const metric_rows,
    const std::int64_t row_count,
    eiel_native_compact_i128_v1 totals[EIEL_COMPACT_DEATH_METRIC_FIELDS_V1]) {
    if (totals == nullptr) return EIEL_NATIVE_COMPACT_REDUCER_NULL_POINTER_V1;
    for (std::size_t field = 0U; field < kMetricFields; ++field) {
        totals[field] = eiel_native_compact_i128_v1{};
    }
    if (row_count < 0) return EIEL_NATIVE_COMPACT_REDUCER_INVALID_EXTENT_V1;
    if (row_count == 0) return EIEL_NATIVE_COMPACT_REDUCER_SUCCESS_V1;
    if (metric_rows == nullptr) return EIEL_NATIVE_COMPACT_REDUCER_NULL_POINTER_V1;
    __int128 accumulators[kMetricFields]{};
    for (std::int64_t row = 0; row < row_count; ++row) {
        const std::uint8_t* cursor = metric_rows
            + static_cast<std::size_t>(row) * kMetricRowBytes;
        for (std::size_t field = 0U; field < kMetricFields; ++field) {
            if (field == 17U) {
                accumulators[field] += static_cast<__int128>(read_i64_le(cursor));
                cursor += 8U;
            } else {
                accumulators[field] += static_cast<__int128>(read_u32_le(cursor));
                cursor += 4U;
            }
        }
    }
    for (std::size_t field = 0U; field < kMetricFields; ++field) {
        store_i128(accumulators[field], &totals[field]);
    }
    return EIEL_NATIVE_COMPACT_REDUCER_SUCCESS_V1;
}

}  // extern "C"
