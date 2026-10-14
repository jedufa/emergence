#ifndef EIEL_NATIVE_COMPACT_REDUCER_INTERNAL_V1_H
#define EIEL_NATIVE_COMPACT_REDUCER_INTERNAL_V1_H

#include "compact_reducer_core_v1.h"

#include <cstdint>

namespace eiel_compact_reducer_internal_v1 {

[[nodiscard]] eiel_native_compact_reducer_status_v1 reconstruct_birth(
    const std::uint8_t* parent_a,
    const std::uint8_t* parent_b,
    const std::uint8_t* provenance,
    std::int32_t mutation_step,
    std::uint8_t* child,
    eiel_native_compact_birth_batch_receipt_v1* receipt) noexcept;

}  // namespace eiel_compact_reducer_internal_v1

#endif
