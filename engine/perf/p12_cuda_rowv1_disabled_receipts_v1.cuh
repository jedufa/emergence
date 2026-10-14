#pragma once

#include <cstdint>

namespace eiel::p12::cuda_rowv1_disabled_receipts_v1 {

inline constexpr std::int64_t kRowV1AbiVersion = 3;
inline constexpr std::int64_t kRowV1SourceLayoutVersion = 2;
inline constexpr std::int64_t kRowV1Width = 64;
inline constexpr std::int64_t kRawOperationRefineDescendantParameter = 2;
inline constexpr std::int64_t kRawStatusNoop = 4;
inline constexpr std::int64_t kRawReasonRefinedValueUnchanged = 4;

__device__ __forceinline__ void write_cell(
    std::int64_t* const rows,
    std::int64_t* const counts,
    std::int32_t* const overflow,
    const std::int32_t* const disabled_reason_codes,
    const std::int64_t* const parent_agent_ids,
    const std::int64_t* const parent_slots,
    const std::int64_t* const parent_slot_tokens,
    const std::int64_t* const child_agent_ids,
    const std::int64_t* const child_slots,
    const std::int64_t* const child_slot_tokens,
    const std::int64_t* const generation,
    const std::int64_t* const tick,
    const std::int64_t cell,
    const std::int64_t capacity,
    const std::int64_t attempts,
    const std::int64_t row_width
) {
    std::int64_t sequence = 0;
    overflow[cell] = 0;
    counts[cell] = 0;
    for (std::int64_t attempt = 0; attempt < attempts; ++attempt) {
        const std::int64_t index = cell * attempts + attempt;
        const std::int32_t reason_code = disabled_reason_codes[index];
        if (reason_code == 0) {
            continue;
        }
        if (reason_code < 1 || reason_code > 6) {
            overflow[cell] = 1;
            continue;
        }
        if (sequence >= capacity) {
            overflow[cell] = 1;
            continue;
        }
        std::int64_t* const row = rows + (cell * capacity + sequence) * row_width;
        for (std::int64_t column = 0; column < row_width; ++column) {
            row[column] = -1;
        }
        row[0] = kRowV1AbiVersion;
        row[1] = kRowV1SourceLayoutVersion;
        row[2] = sequence;
        row[3] = generation[cell];
        row[4] = tick[cell];
        row[5] = parent_agent_ids[index];
        row[6] = parent_slots[index];
        row[7] = parent_slot_tokens[index];
        row[8] = child_agent_ids[index];
        row[9] = child_slots[index];
        row[10] = child_slot_tokens[index];
        row[11] = attempt;
        row[12] = 0;
        row[13] = 0;
        row[14] = kRawOperationRefineDescendantParameter;
        row[15] = kRawStatusNoop;
        row[16] = kRawReasonRefinedValueUnchanged;
        row[17] = -1;
        row[18] = -1;
        row[19] = 0;
        row[20] = 0;
        row[21] = 0;
        row[22] = reason_code;
        ++sequence;
    }
    counts[cell] = sequence;
}

}  // namespace eiel::p12::cuda_rowv1_disabled_receipts_v1

