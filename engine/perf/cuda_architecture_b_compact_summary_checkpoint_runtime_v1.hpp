#pragma once

#include <cstdint>
#include <type_traits>

namespace eiel::cuda_architecture_b::compact_summary_checkpoint_runtime_v1 {

inline constexpr char kContractVersionV1[] =
    "eiel_cuda_architecture_b_compact_summary_checkpoint_runtime_v1";
inline constexpr std::int64_t kFlatDescriptorCountV1 = 132;
inline constexpr std::int64_t kPatchP2DescriptorCountV1 = 144;
inline constexpr std::int64_t kDescriptorCapacityV1 = kPatchP2DescriptorCountV1;

// Compatibility name for Flat-only consumers. New consumers must select the
// exact count for their world and use kDescriptorCapacityV1 for fixed storage.
inline constexpr std::int64_t kDescriptorCountV1 = kFlatDescriptorCountV1;

inline constexpr std::int64_t expected_descriptor_count_v1(
    const bool patch_p2
) noexcept {
    return patch_p2 ? kPatchP2DescriptorCountV1 : kFlatDescriptorCountV1;
}

inline constexpr bool is_supported_descriptor_count_v1(
    const std::int64_t descriptor_count
) noexcept {
    return descriptor_count == kFlatDescriptorCountV1
        || descriptor_count == kPatchP2DescriptorCountV1;
}
inline constexpr std::int64_t kAgentIdToSlotAgentIdsDescriptorIndexV1 = 8;
inline constexpr std::int64_t kAgentIdToSlotSlotsDescriptorIndexV1 = 9;

enum DescriptorSpecColumnV1 : std::int64_t {
    kDataTensorIndexV1 = 0,
    kAliveMaskTensorIndexV1,
    kActiveLengthTensorIndexV1,
    kCellStrideBytesV1,
    kBytesPerCellV1,
    kRowCapacityV1,
    kRowBytesV1,
    kAliveMaskCellStrideV1,
    kDtypeBytesV1,
    kCanonicalizationModeV1,
    kRawOutputIndexV1,
    kCanonicalOutputIndexV1,
    kDescriptorIndexV1,
    kDescriptorSpecWidthV1,
};

static_assert(kDescriptorSpecWidthV1 == 13);
static_assert(expected_descriptor_count_v1(false) == 132);
static_assert(expected_descriptor_count_v1(true) == 144);
static_assert(kDescriptorCapacityV1 >= kFlatDescriptorCountV1);
static_assert(kDescriptorCapacityV1 >= kPatchP2DescriptorCountV1);
static_assert(std::is_same_v<std::underlying_type_t<DescriptorSpecColumnV1>, std::int64_t>);

}  // namespace eiel::cuda_architecture_b::compact_summary_checkpoint_runtime_v1
