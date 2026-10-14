#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <atomic>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace {

constexpr int kWorldWidth = 32;
constexpr int kSites = 1024;
constexpr int kInitialPopulation = 256;
constexpr int kCapacity = 512;
constexpr int kPolicyLoci = 112;
constexpr int kTraitLoci = 16;
constexpr int kLinkedLoci = 128;
constexpr int kFeatures = 16;
constexpr int kActions = 7;
constexpr int kRecombinationBlocks = 16;
constexpr int kThreads = 256;
constexpr int kWarpSize = 32;
constexpr int kWarpsPerBlock = kThreads / kWarpSize;
constexpr int kMaxSitesPerLane = (kSites + kThreads - 1) / kThreads;
constexpr int kMaxAgentsPerLane = (kCapacity + kThreads - 1) / kThreads;
constexpr int kStateBytes = 265472;
constexpr int kReferenceScratchBytes = 8192;
constexpr int kEventRowBytes = 48;
constexpr int kEventRowsPerTick = 2 * kCapacity;
constexpr int kSparseEventRowBytesV3 = 32;
constexpr int kScienceBirthCapacity = kCapacity / 2;
constexpr int kScienceWorldControlWidthV1 = 22;
constexpr int kScienceAgentMetricWidthV1 = 21;
constexpr int kScienceBirthProvenanceWidthV1 = kLinkedLoci + 4;
constexpr std::uint64_t kMutationThresholdV2 = 2048u;
constexpr std::uint16_t kEmpty = 0xffffu;
constexpr std::uint64_t kRngSemanticEpoch = 3u;
constexpr std::uint64_t kFullCapabilityMask = (1u << 7) - 1u;
constexpr int kPatchDeltaShift = 0;
constexpr int kWdlDeltaShift = 8;
constexpr int kJungleDeltaShift = 16;
constexpr int kEedDeltaShift = 24;
constexpr int kDeathDeltaShift = 32;
static_assert(
    (kMaxSitesPerLane + kMaxAgentsPerLane) * kWarpSize < 256,
    "packed per-warp counter deltas must not carry between byte lanes"
);

constexpr int kHeaderOffset = 0;
constexpr int kOccupancyOffset = 192;
constexpr int kResourceOffset = 2240;
constexpr int kScentOffset = 4288;
constexpr int kWasteOffset = 6336;
constexpr int kEedImpactOffset = 8384;
constexpr int kEedPendingOffset = 10432;
constexpr int kWdlDueOffset = 12480;
constexpr int kWdlActiveOffset = 20672;
constexpr int kTerrainOffset = 21696;
constexpr int kJungleOffset = 22720;
constexpr int kPositionOffset = 23744;
constexpr int kEnergyOffset = 24768;
constexpr int kHealthOffset = 25792;
constexpr int kAgeOffset = 26816;
constexpr int kAbsoluteBirthTickOffset = 27840;
constexpr int kAliveWordsOffset = 31936;
constexpr int kAgentIdOffset = 32000;
constexpr int kParentAIdOffset = 36096;
constexpr int kParentBIdOffset = 40192;
constexpr int kPolicyAllelesOffset = 44288;
constexpr int kTraitAllelesOffset = 158976;
constexpr int kExpressedPolicyOffset = 191744;
constexpr int kExpressedTraitOffset = 249088;

enum HeaderWord : int {
    kWorldUid = 0,
    kNextAbsoluteTick = 1,
    kNextAgentId = 2,
    kActiveCount = 3,
    kStatus = 4,
    kTicksCompleted = 5,
    kPatchCounter = 6,
    kWdlCounter = 7,
    kJungleCounter = 8,
    kEedCounter = 9,
    kBirthCapabilityCounter = 10,
    kMutationCapabilityCounter = 11,
    kRecombinationCapabilityCounter = 12,
    kRngKey = 13,
    kRngEpoch = 14,
    kBirthCount = 15,
    kDeathCount = 16,
    kMutationCount = 17,
    kRecombinationCount = 18,
    kWdlDueCount = 19,
    kJungleActiveCount = 20,
    kEedPendingCount = 21,
    kReserved0 = 22,
    kReserved1 = 23,
};

enum CapabilityBit : std::uint64_t {
    kPatch = 1u << 0,
    kWdl = 1u << 1,
    kJungle = 1u << 2,
    kEed = 1u << 3,
    kBirth = 1u << 4,
    kMutation = 1u << 5,
    kRecombination = 1u << 6,
};

enum ScienceWorldControlFieldV1 : int {
    kScienceCapabilityMask = 0,
    kScienceMutationThreshold = 1,
    kScienceMutationStep = 2,
    kScienceResourceCarryingBase = 3,
    kScienceResourceCarryingVariation = 4,
    kScienceResourceGrowthBase = 5,
    kScienceResourceGrowthVariation = 6,
    kScienceResourceMultiplierQ16 = 7,
    kScienceMovementCost = 8,
    kScienceResourceUptakeBase = 9,
    kScienceWdlDamage = 10,
    kScienceJungleDamageThreshold = 11,
    kScienceEedLossShift = 12,
    kScienceReproductionEnergyThreshold = 13,
    kScienceReproductionEnergyCost = 14,
    kScienceSelectionPressureQ16 = 15,
    kScienceSeedEcology = 16,
    kScienceSeedFounder = 17,
    kScienceSeedPolicy = 18,
    kScienceSeedMutation = 19,
    kScienceSeedTaskEnvironment = 20,
    kScienceSeedInitialization = 21,
};

enum ScienceAgentMetricFieldV1 : int {
    kScienceAction0Count = 0,
    kScienceAction1Count = 1,
    kScienceAction2Count = 2,
    kScienceAction3Count = 3,
    kScienceAction4Count = 4,
    kScienceAction5Count = 5,
    kScienceAction6Count = 6,
    kScienceAliveTicks = 7,
    kScienceReproductionEligible = 8,
    kScienceMateCandidate = 9,
    kScienceBirthAttempt = 10,
    kScienceSuccessfulParent = 11,
    kScienceResourceUptake = 12,
    kScienceMovementCostTotal = 13,
    kScienceMovementAttempt = 14,
    kScienceMovementSuccess = 15,
    kScienceHealthDamage = 16,
    kScienceEnergyDelta = 17,
    kScienceAttackAttempt = 18,
    kScienceAttackHit = 19,
    kScienceBirthsAsParent = 20,
};

enum RngDomain : std::uint64_t {
    kDomainInitialSite = 0x21u,
    kDomainInitialPolicy = 0x22u,
    kDomainInitialTrait = 0x23u,
    kDomainSiteEnvironment = 0x30u,
    kDomainRecombination = 0x41u,
    kDomainMutation = 0x42u,
};

enum SiteRngLane : int {
    kSitePatchCarryingLane = 0,
    kSitePatchGrowthLane = 1,
    kSiteWdlLane = 2,
    kSiteJungleLane = 3,
};

struct LifecycleEventRow {
    std::uint64_t absolute_tick;
    std::uint64_t event_id;
    std::uint64_t agent_id;
    std::uint64_t parent_a_id;
    std::uint64_t parent_b_id;
    std::uint16_t event_kind;
    std::uint16_t site_index;
    std::uint16_t status;
    std::uint16_t reserved;
};
static_assert(sizeof(LifecycleEventRow) == kEventRowBytes, "event row ABI drift");

struct SparseLifecycleEventV3 {
    std::uint64_t agent_id;
    std::uint64_t parent_a_id;
    std::uint64_t parent_b_id;
    std::uint64_t packed_key_and_reserved;
};
static_assert(
    sizeof(SparseLifecycleEventV3) == kSparseEventRowBytesV3,
    "sparse event row ABI drift"
);

enum SparseEventValidationBitV3 : std::int32_t {
    kSparseInvalidOffsets = 1 << 0,
    kSparseInvalidStatus = 1 << 1,
    kSparseNonzeroUnusedRow = 1 << 2,
    kSparseInvalidTick = 1 << 3,
    kSparseInvalidKindOrId = 1 << 4,
    kSparseInvalidAgentIdentity = 1 << 5,
    kSparseInvalidParentIdentity = 1 << 6,
    kSparseInvalidBirthParents = 1 << 7,
    kSparseInvalidSiteOrReserved = 1 << 8,
    kSparseOutputOverflow = 1 << 9,
    kSparseCountMismatch = 1 << 10,
    kSparseDirectOverflow = 1 << 11,
    kSparseDirectOrder = 1 << 12,
};

template <typename T>
__device__ __forceinline__ T* field(std::uint8_t* row, const int offset) {
    return reinterpret_cast<T*>(row + offset);
}

template <typename T>
__device__ __forceinline__ const T* field(const std::uint8_t* row, const int offset) {
    return reinterpret_cast<const T*>(row + offset);
}

__device__ __forceinline__ std::uint32_t rotate_left32(
    const std::uint32_t value, const int shift
) {
    return (value << shift) | (value >> (32 - shift));
}

__device__ __forceinline__ uint2 rng_tick_key_values(
    const std::uint64_t rng_key,
    const std::uint64_t world_uid,
    const std::uint64_t absolute_tick
) {
    const std::uint32_t key_low = static_cast<std::uint32_t>(rng_key);
    const std::uint32_t key_high = static_cast<std::uint32_t>(rng_key >> 32);
    const std::uint32_t uid_low = static_cast<std::uint32_t>(world_uid);
    const std::uint32_t uid_high = static_cast<std::uint32_t>(world_uid >> 32);
    const std::uint32_t tick_low = static_cast<std::uint32_t>(absolute_tick);
    const std::uint32_t tick_high = static_cast<std::uint32_t>(absolute_tick >> 32);
    std::uint32_t first = key_low ^ rotate_left32(key_high, 7) ^
        rotate_left32(uid_low, 13) ^ rotate_left32(uid_high, 23) ^
        rotate_left32(tick_low, 3) ^ rotate_left32(tick_high, 17) ^
        0x9e3779b9u ^ static_cast<std::uint32_t>(kRngSemanticEpoch);
    std::uint32_t second = key_high ^ rotate_left32(key_low, 11) ^
        rotate_left32(uid_high, 5) ^ rotate_left32(uid_low, 19) ^
        rotate_left32(tick_high, 29) ^ rotate_left32(tick_low, 9) ^
        0xbb67ae85u;
    first += rotate_left32(second, 15);
    second ^= rotate_left32(first, 21);
    first += rotate_left32(second, 9);
    return make_uint2(first, second);
}

__device__ __forceinline__ uint2 rng_tick_key(
    const std::uint64_t* header,
    const std::uint64_t absolute_tick
) {
    return rng_tick_key_values(
        header[kRngKey], header[kWorldUid], absolute_tick
    );
}

__device__ __forceinline__ uint4 rng_address_counter(
    const std::uint64_t domain,
    const std::uint64_t subject,
    const std::uint64_t operation,
    const std::uint64_t draw
) {
    const std::uint32_t domain_low = static_cast<std::uint32_t>(domain);
    const std::uint32_t domain_high = static_cast<std::uint32_t>(domain >> 32);
    const std::uint32_t subject_low = static_cast<std::uint32_t>(subject);
    const std::uint32_t subject_high = static_cast<std::uint32_t>(subject >> 32);
    const std::uint32_t operation_low = static_cast<std::uint32_t>(operation);
    const std::uint32_t operation_high = static_cast<std::uint32_t>(operation >> 32);
    const std::uint32_t draw_low = static_cast<std::uint32_t>(draw);
    const std::uint32_t draw_high = static_cast<std::uint32_t>(draw >> 32);
    return make_uint4(
        subject_low ^ rotate_left32(domain_high, 5) ^
            rotate_left32(operation_low, 17) ^ rotate_left32(draw_high, 27),
        subject_high ^ rotate_left32(domain_low, 7) ^
            rotate_left32(operation_high, 19) ^ rotate_left32(draw_low, 29),
        domain_low ^ rotate_left32(subject_high, 11) ^
            rotate_left32(operation_low, 23) ^ rotate_left32(draw_high, 3),
        domain_high ^ rotate_left32(subject_low, 13) ^
            rotate_left32(operation_high, 25) ^ rotate_left32(draw_low, 5)
    );
}

__device__ __forceinline__ uint4 philox4x32_10(uint4 counter, uint2 key) {
    constexpr std::uint32_t kPhiloxMultiplier0 = 0xd2511f53u;
    constexpr std::uint32_t kPhiloxMultiplier1 = 0xcd9e8d57u;
    constexpr std::uint32_t kPhiloxWeyl0 = 0x9e3779b9u;
    constexpr std::uint32_t kPhiloxWeyl1 = 0xbb67ae85u;
#pragma unroll
    for (int round = 0; round < 10; ++round) {
        const std::uint32_t low0 = kPhiloxMultiplier0 * counter.x;
        const std::uint32_t high0 = __umulhi(kPhiloxMultiplier0, counter.x);
        const std::uint32_t low1 = kPhiloxMultiplier1 * counter.z;
        const std::uint32_t high1 = __umulhi(kPhiloxMultiplier1, counter.z);
        counter = make_uint4(
            high1 ^ counter.y ^ key.x,
            low1,
            high0 ^ counter.w ^ key.y,
            low0
        );
        key.x += kPhiloxWeyl0;
        key.y += kPhiloxWeyl1;
    }
    return counter;
}

__device__ __forceinline__ uint4 counter_rng4x32(
    const uint2 tick_key,
    const std::uint64_t domain,
    const std::uint64_t subject,
    const std::uint64_t operation,
    const std::uint64_t draw
) {
    return philox4x32_10(
        rng_address_counter(domain, subject, operation, draw),
        tick_key
    );
}

__device__ __forceinline__ std::uint32_t counter_rng32(
    const uint2 tick_key,
    const std::uint64_t domain,
    const std::uint64_t subject,
    const std::uint64_t operation,
    const std::uint64_t draw
) {
    const uint4 block = counter_rng4x32(
        tick_key, domain, subject, operation, draw
    );
    return block.x ^ block.y ^ block.z ^ block.w;
}

__device__ __forceinline__ int clamp_int(const int value, const int low, const int high) {
    return value < low ? low : (value > high ? high : value);
}

__device__ __forceinline__ std::int8_t clamp_i8(const int value) {
    return static_cast<std::int8_t>(clamp_int(value, -127, 127));
}

__device__ __forceinline__ std::int16_t clamp_i16(const int value) {
    return static_cast<std::int16_t>(clamp_int(value, -32767, 32767));
}

__device__ __forceinline__ bool is_alive(const std::uint32_t* words, const int slot) {
    return (words[slot >> 5] & (1u << (slot & 31))) != 0u;
}

__device__ __forceinline__ void write_event_payload(
    LifecycleEventRow* events,
    const int row_index,
    const std::uint64_t tick,
    const std::uint64_t agent_id,
    const std::uint64_t parent_a_id,
    const std::uint64_t parent_b_id,
    const std::uint16_t event_kind,
    const std::uint64_t canonical_subject_ordinal,
    const std::uint16_t site
) {
    LifecycleEventRow& event = events[row_index];
    event.absolute_tick = tick;
    event.event_id = (static_cast<std::uint64_t>(event_kind) << 48) |
        (canonical_subject_ordinal & 0x0000ffffffffffffULL);
    event.agent_id = agent_id;
    event.parent_a_id = parent_a_id;
    event.parent_b_id = parent_b_id;
    event.event_kind = event_kind;
    event.site_index = site;
    event.reserved = 0u;
    __threadfence_block();
    event.status = 1u;
}

__device__ __forceinline__ void write_event(
    LifecycleEventRow* events,
    std::int32_t* event_counts,
    const int world,
    const int row_index,
    const std::uint64_t tick,
    const std::uint64_t agent_id,
    const std::uint64_t parent_a_id,
    const std::uint64_t parent_b_id,
    const std::uint16_t event_kind,
    const std::uint64_t canonical_subject_ordinal,
    const std::uint16_t site
) {
    write_event_payload(
        events, row_index, tick, agent_id, parent_a_id, parent_b_id,
        event_kind, canonical_subject_ordinal, site
    );
    ++event_counts[world];
}

__device__ __forceinline__ void clear_event(LifecycleEventRow* events, const int row_index) {
    LifecycleEventRow& event = events[row_index];
    event.absolute_tick = 0u;
    event.event_id = 0u;
    event.agent_id = 0u;
    event.parent_a_id = 0u;
    event.parent_b_id = 0u;
    event.event_kind = 0u;
    event.site_index = 0u;
    event.status = 0u;
    event.reserved = 0u;
}

__device__ __forceinline__ void set_alive(std::uint32_t* words, const int slot) {
    atomicOr(words + (slot >> 5), 1u << (slot & 31));
}

__device__ __forceinline__ void clear_alive(std::uint32_t* words, const int slot) {
    atomicAnd(words + (slot >> 5), ~(1u << (slot & 31)));
}

__device__ __forceinline__ int policy_allele_index(
    const int allele, const int locus, const int slot
) {
    return (allele * kPolicyLoci + locus) * kCapacity + slot;
}

__device__ __forceinline__ int trait_allele_index(
    const int allele, const int locus, const int slot
) {
    return (allele * kTraitLoci + locus) * kCapacity + slot;
}

__device__ __forceinline__ int expressed_policy_index(
    const int action, const int group, const int slot
) {
    return (action * 4 + group) * kCapacity + slot;
}

__device__ __forceinline__ int expressed_trait_index(const int locus, const int slot) {
    return locus * kCapacity + slot;
}

__device__ __forceinline__ std::int8_t midpoint_i8(
    const std::int8_t first, const std::int8_t second
) {
    return static_cast<std::int8_t>(
        (static_cast<int>(first) + static_cast<int>(second)) / 2
    );
}

__device__ __forceinline__ std::int16_t midpoint_i16(
    const std::int16_t first, const std::int16_t second
) {
    return static_cast<std::int16_t>(
        (static_cast<int>(first) + static_cast<int>(second)) / 2
    );
}

__device__ void express_genome(std::uint8_t* row, const int slot) {
    const auto* policy = field<std::int8_t>(row, kPolicyAllelesOffset);
    const auto* traits = field<std::int16_t>(row, kTraitAllelesOffset);
    auto* expressed_policy = field<std::int32_t>(row, kExpressedPolicyOffset);
    auto* expressed_traits = field<std::int16_t>(row, kExpressedTraitOffset);
    for (int action = 0; action < kActions; ++action) {
        for (int group = 0; group < 4; ++group) {
            std::uint32_t packed = 0u;
            for (int lane = 0; lane < 4; ++lane) {
                const int feature_index = group * 4 + lane;
                const int locus = feature_index * kActions + action;
                const std::int8_t value = midpoint_i8(
                    policy[policy_allele_index(0, locus, slot)],
                    policy[policy_allele_index(1, locus, slot)]
                );
                packed |= static_cast<std::uint32_t>(
                    static_cast<std::uint8_t>(value)
                ) << (lane * 8);
            }
            expressed_policy[expressed_policy_index(action, group, slot)] =
                static_cast<std::int32_t>(packed);
        }
    }
    for (int locus = 0; locus < kTraitLoci; ++locus) {
        expressed_traits[expressed_trait_index(locus, slot)] = midpoint_i16(
            traits[trait_allele_index(0, locus, slot)],
            traits[trait_allele_index(1, locus, slot)]
        );
    }
}

__device__ __forceinline__ std::uint32_t pack_features(
    const std::int8_t* features, const int group
) {
    std::uint32_t packed = 0u;
    for (int lane = 0; lane < 4; ++lane) {
        packed |= static_cast<std::uint32_t>(
            static_cast<std::uint8_t>(features[group * 4 + lane])
        ) << (lane * 8);
    }
    return packed;
}

__device__ int controller_action(const std::uint8_t* row, const int slot) {
    const auto* position = field<std::uint16_t>(row, kPositionOffset);
    const auto* energy = field<std::int16_t>(row, kEnergyOffset);
    const auto* health = field<std::int16_t>(row, kHealthOffset);
    const auto* age = field<std::uint16_t>(row, kAgeOffset);
    const auto* resource = field<std::uint16_t>(row, kResourceOffset);
    const auto* scent = field<std::uint16_t>(row, kScentOffset);
    const auto* waste = field<std::uint16_t>(row, kWasteOffset);
    const auto* impact = field<std::uint16_t>(row, kEedImpactOffset);
    const auto* pending = field<std::uint16_t>(row, kEedPendingOffset);
    const auto* wdl = field<std::uint8_t>(row, kWdlActiveOffset);
    const auto* terrain = field<std::uint8_t>(row, kTerrainOffset);
    const auto* jungle = field<std::uint8_t>(row, kJungleOffset);
    const auto* occupancy = field<std::uint16_t>(row, kOccupancyOffset);
    const auto* ids = field<std::uint64_t>(row, kAgentIdOffset);
    const auto* header = field<std::uint64_t>(row, kHeaderOffset);
    const auto* policy = field<std::int32_t>(row, kExpressedPolicyOffset);
    const int site = static_cast<int>(position[slot]);
    const int x = site & (kWorldWidth - 1);
    const int y = site >> 5;
    const int north = ((y + kWorldWidth - 1) & (kWorldWidth - 1)) * kWorldWidth + x;
    const int east = y * kWorldWidth + ((x + 1) & (kWorldWidth - 1));
    std::int8_t features[kFeatures];
    features[0] = 32;
    features[1] = clamp_i8(static_cast<int>(energy[slot]) / 8 - 16);
    features[2] = clamp_i8(static_cast<int>(health[slot]) / 4 - 16);
    features[3] = clamp_i8(static_cast<int>(resource[site]) / 4 - 16);
    features[4] = clamp_i8(static_cast<int>(wdl[site]) * 8 - 16);
    features[5] = clamp_i8(static_cast<int>(jungle[site]) - 16);
    features[6] = clamp_i8(static_cast<int>(impact[site]) / 4 - 16);
    features[7] = clamp_i8(static_cast<int>(pending[site]) / 4 - 16);
    features[8] = clamp_i8(static_cast<int>(scent[site]) / 4 - 16);
    features[9] = clamp_i8(static_cast<int>(waste[site]) / 4 - 16);
    features[10] = clamp_i8(static_cast<int>(age[slot]) / 8 - 16);
    features[11] = occupancy[north] == kEmpty ? 16 : -16;
    features[12] = occupancy[east] == kEmpty ? 16 : -16;
    features[13] = static_cast<std::int8_t>((ids[slot] & 1u) ? 16 : -16);
    features[14] = static_cast<std::int8_t>((header[kNextAbsoluteTick] & 1u) ? 16 : -16);
    features[15] = clamp_i8(
        static_cast<int>(resource[site]) - static_cast<int>(impact[site]) -
        static_cast<int>(terrain[site]) * 8
    );
    int selected = 0;
    int selected_score = std::numeric_limits<int>::min();
    for (int action = 0; action < kActions; ++action) {
        int score = 0;
        for (int group = 0; group < 4; ++group) {
            score = __dp4a(
                policy[expressed_policy_index(action, group, slot)],
                static_cast<std::int32_t>(pack_features(features, group)),
                score
            );
        }
        if (score > selected_score) {
            selected_score = score;
            selected = action;
        }
    }
    return selected;
}

__device__ __forceinline__ std::uint16_t movement_target(
    const std::uint16_t source, const int action
) {
    const int x = static_cast<int>(source) & (kWorldWidth - 1);
    const int y = static_cast<int>(source) >> 5;
    if (action == 1) {
        return static_cast<std::uint16_t>(
            ((y + kWorldWidth - 1) & (kWorldWidth - 1)) * kWorldWidth + x
        );
    }
    if (action == 2) {
        return static_cast<std::uint16_t>(
            y * kWorldWidth + ((x + 1) & (kWorldWidth - 1))
        );
    }
    if (action == 3) {
        return static_cast<std::uint16_t>(
            ((y + 1) & (kWorldWidth - 1)) * kWorldWidth + x
        );
    }
    if (action == 4) {
        return static_cast<std::uint16_t>(
            y * kWorldWidth + ((x + kWorldWidth - 1) & (kWorldWidth - 1))
        );
    }
    return source;
}

__device__ __forceinline__ bool sites_are_cardinal_neighbors(
    const std::uint16_t first, const std::uint16_t second
) {
    const int first_x = static_cast<int>(first) & (kWorldWidth - 1);
    const int first_y = static_cast<int>(first) >> 5;
    const int second_x = static_cast<int>(second) & (kWorldWidth - 1);
    const int second_y = static_cast<int>(second) >> 5;
    const int raw_dx = abs(first_x - second_x);
    const int raw_dy = abs(first_y - second_y);
    const int dx = min(raw_dx, kWorldWidth - raw_dx);
    const int dy = min(raw_dy, kWorldWidth - raw_dy);
    return dx + dy == 1;
}

__device__ std::uint64_t update_site(
    std::uint8_t* row,
    const int site,
    const std::uint64_t tick,
    const uint2 tick_rng_key,
    const std::uint64_t capability_mask,
    const std::int64_t* science_world_control,
    const bool defer_counters
) {
    auto* header = field<std::uint64_t>(row, kHeaderOffset);
    auto* resource = field<std::uint16_t>(row, kResourceOffset);
    auto* scent = field<std::uint16_t>(row, kScentOffset);
    auto* waste = field<std::uint16_t>(row, kWasteOffset);
    auto* impact = field<std::uint16_t>(row, kEedImpactOffset);
    auto* pending = field<std::uint16_t>(row, kEedPendingOffset);
    auto* due = field<std::uint64_t>(row, kWdlDueOffset);
    auto* active = field<std::uint8_t>(row, kWdlActiveOffset);
    const auto* terrain = field<std::uint8_t>(row, kTerrainOffset);
    auto* jungle = field<std::uint8_t>(row, kJungleOffset);
    const uint4 site_draws = counter_rng4x32(
        tick_rng_key, kDomainSiteEnvironment,
        static_cast<std::uint64_t>(site), 0u, 0u
    );
    std::uint64_t packed_deltas = 0u;
    auto add_counter = [&] __device__ (const int index, const int shift) {
        if (defer_counters) {
            packed_deltas += 1ULL << shift;
        } else {
            ++header[index];
        }
    };
    if ((capability_mask & kEed) != 0u) {
        if (pending[site] != 0u) {
            impact[site] = static_cast<std::uint16_t>(
                min(4095, static_cast<int>(impact[site]) + static_cast<int>(pending[site]))
            );
            pending[site] = 0u;
            add_counter(kEedCounter, kEedDeltaShift);
        }
        if (impact[site] != 0u) {
            const int loss_shift = science_world_control == nullptr
                ? 5 : static_cast<int>(science_world_control[kScienceEedLossShift]);
            const int loss = max(1, static_cast<int>(impact[site]) >> loss_shift);
            resource[site] = static_cast<std::uint16_t>(
                max(0, static_cast<int>(resource[site]) - loss)
            );
            --impact[site];
        }
    }
    if (scent[site] != 0u) {
        --scent[site];
    }
    if (waste[site] != 0u) {
        --waste[site];
    }
    if ((capability_mask & kPatch) != 0u) {
        const int before = static_cast<int>(resource[site]);
        const int carrying_base = science_world_control == nullptr
            ? 96 : static_cast<int>(
                science_world_control[kScienceResourceCarryingBase]
            );
        const int carrying_variation = science_world_control == nullptr
            ? 63 : static_cast<int>(
                science_world_control[kScienceResourceCarryingVariation]
            );
        const int growth_base = science_world_control == nullptr
            ? 1 : static_cast<int>(
                science_world_control[kScienceResourceGrowthBase]
            );
        const int growth_variation = science_world_control == nullptr
            ? 3 : static_cast<int>(
                science_world_control[kScienceResourceGrowthVariation]
            );
        const std::int64_t resource_multiplier_q16 =
            science_world_control == nullptr ? 65536 :
                science_world_control[kScienceResourceMultiplierQ16];
        const int raw_carrying = carrying_base + static_cast<int>(
            site_draws.x % static_cast<std::uint32_t>(carrying_variation + 1)
        ) - static_cast<int>(terrain[site]) * 8;
        const int carrying = clamp_int(
            static_cast<int>((static_cast<std::int64_t>(max(0, raw_carrying)) *
                resource_multiplier_q16) >> 16), 0, 65535
        );
        const int raw_growth = growth_base + static_cast<int>(
            site_draws.y % static_cast<std::uint32_t>(growth_variation + 1)
        ) + (terrain[site] == 0u ? 1 : 0);
        const int growth = clamp_int(
            static_cast<int>((static_cast<std::int64_t>(max(0, raw_growth)) *
                resource_multiplier_q16) >> 16), 0, 65535
        );
        const int waste_penalty = static_cast<int>(waste[site]) >> 5;
        resource[site] = static_cast<std::uint16_t>(
            min(carrying, before + max(0, growth - waste_penalty))
        );
        if (static_cast<int>(resource[site]) != before) {
            add_counter(kPatchCounter, kPatchDeltaShift);
        }
    }
    if ((capability_mask & kWdl) != 0u) {
        if (active[site] != 0u) {
            --active[site];
        }
        if (due[site] != 0u && due[site] <= tick) {
            due[site] = 0u;
            active[site] = 4u;
            add_counter(kWdlCounter, kWdlDeltaShift);
        }
        if (due[site] == 0u && active[site] == 0u &&
            (site_draws.z & 63u) == 0u) {
            const std::uint64_t max_tick = static_cast<std::uint64_t>(
                std::numeric_limits<std::int64_t>::max()
            );
            due[site] = tick <= max_tick - 2u ? tick + 2u : max_tick;
            add_counter(kWdlCounter, kWdlDeltaShift);
        }
    }
    if ((capability_mask & kJungle) != 0u) {
        const std::uint32_t draw = site_draws.w;
        const int before = static_cast<int>(jungle[site]);
        int after = before;
        if ((draw & 7u) == 0u) {
            after = min(63, before + 1);
        } else if ((draw & 15u) == 1u) {
            after = max(0, before - 1);
        }
        jungle[site] = static_cast<std::uint8_t>(after);
        if (after != before) {
            add_counter(kJungleCounter, kJungleDeltaShift);
        }
    }
    return packed_deltas;
}

__device__ std::uint64_t apply_agent_effects(
    std::uint8_t* row,
    const int slot,
    const int action,
    const std::uint64_t capability_mask,
    const std::int64_t* science_world_control,
    const bool defer_counters,
    LifecycleEventRow* events,
    std::int32_t* event_counts,
    const int world,
    const int death_event_row,
    const std::uint64_t tick
) {
    auto* header = field<std::uint64_t>(row, kHeaderOffset);
    auto* occupancy = field<std::uint16_t>(row, kOccupancyOffset);
    auto* resource = field<std::uint16_t>(row, kResourceOffset);
    auto* scent = field<std::uint16_t>(row, kScentOffset);
    auto* waste = field<std::uint16_t>(row, kWasteOffset);
    auto* pending = field<std::uint16_t>(row, kEedPendingOffset);
    const auto* wdl = field<std::uint8_t>(row, kWdlActiveOffset);
    const auto* terrain = field<std::uint8_t>(row, kTerrainOffset);
    const auto* jungle = field<std::uint8_t>(row, kJungleOffset);
    auto* position = field<std::uint16_t>(row, kPositionOffset);
    auto* energy = field<std::int16_t>(row, kEnergyOffset);
    auto* health = field<std::int16_t>(row, kHealthOffset);
    auto* age = field<std::uint16_t>(row, kAgeOffset);
    auto* alive = field<std::uint32_t>(row, kAliveWordsOffset);
    const auto* traits = field<std::int16_t>(row, kExpressedTraitOffset);
    const auto* ids = field<std::uint64_t>(row, kAgentIdOffset);
    const auto* parent_a_ids = field<std::uint64_t>(row, kParentAIdOffset);
    const auto* parent_b_ids = field<std::uint64_t>(row, kParentBIdOffset);
    std::uint64_t packed_deltas = 0u;
    const int site = static_cast<int>(position[slot]);
    int next_energy = static_cast<int>(energy[slot]) - 1;
    int next_health = static_cast<int>(health[slot]);
    scent[site] = static_cast<std::uint16_t>(
        min(4095, static_cast<int>(scent[site]) + 2)
    );
    waste[site] = static_cast<std::uint16_t>(
        min(4095, static_cast<int>(waste[site]) + 1)
    );
    if (action >= 1 && action <= 4) {
        const int movement_cost = science_world_control == nullptr
            ? 1 : static_cast<int>(science_world_control[kScienceMovementCost]);
        next_energy -= movement_cost + static_cast<int>(terrain[site]) +
            (abs(static_cast<int>(traits[expressed_trait_index(1, slot)])) & 1);
    } else if (action == 0) {
        next_energy += 1;
    } else if (action == 5 && resource[site] != 0u) {
        const int uptake_base = science_world_control == nullptr
            ? 4 : static_cast<int>(
                science_world_control[kScienceResourceUptakeBase]
            );
        const int uptake = min(
            static_cast<int>(resource[site]),
            uptake_base +
                (abs(static_cast<int>(traits[expressed_trait_index(2, slot)])) & 3)
        );
        resource[site] = static_cast<std::uint16_t>(static_cast<int>(resource[site]) - uptake);
        waste[site] = static_cast<std::uint16_t>(
            min(4095, static_cast<int>(waste[site]) + uptake)
        );
        next_energy += uptake;
        if ((capability_mask & kEed) != 0u) {
            pending[site] = static_cast<std::uint16_t>(
                min(4095, static_cast<int>(pending[site]) + uptake)
            );
            if (defer_counters) {
                packed_deltas += 1ULL << kEedDeltaShift;
            } else {
                ++header[kEedCounter];
            }
        }
    }
    if ((capability_mask & kWdl) != 0u && wdl[site] != 0u) {
        const int resistance = abs(static_cast<int>(traits[expressed_trait_index(3, slot)])) & 3;
        const int damage = science_world_control == nullptr
            ? 1 : static_cast<int>(science_world_control[kScienceWdlDamage]);
        next_health -= damage * max(1, static_cast<int>(wdl[site]) - resistance);
    }
    const int jungle_threshold = science_world_control == nullptr
        ? 24 : static_cast<int>(
            science_world_control[kScienceJungleDamageThreshold]
        );
    if ((capability_mask & kJungle) != 0u &&
        static_cast<int>(jungle[site]) > jungle_threshold) {
        const int resistance = abs(static_cast<int>(traits[expressed_trait_index(4, slot)])) & 7;
        if (static_cast<int>(jungle[site]) > jungle_threshold + resistance) {
            --next_health;
        }
    }
    age[slot] = static_cast<std::uint16_t>(min(65535, static_cast<int>(age[slot]) + 1));
    energy[slot] = clamp_i16(next_energy);
    health[slot] = clamp_i16(next_health);
    const int lifespan = 384 + (abs(static_cast<int>(traits[expressed_trait_index(6, slot)])) & 127);
    if (next_energy <= 0 || next_health <= 0 || static_cast<int>(age[slot]) > lifespan) {
        clear_alive(alive, slot);
        occupancy[site] = kEmpty;
        if (defer_counters) {
            if (events != nullptr) {
                write_event_payload(
                    events, death_event_row, tick,
                    ids[slot], parent_a_ids[slot], parent_b_ids[slot], 1u,
                    static_cast<std::uint64_t>(slot), static_cast<std::uint16_t>(site)
                );
            }
            packed_deltas += 1ULL << kDeathDeltaShift;
        } else {
            write_event(
                events, event_counts, world, death_event_row, tick,
                ids[slot], parent_a_ids[slot], parent_b_ids[slot], 1u,
                static_cast<std::uint64_t>(slot), static_cast<std::uint16_t>(site)
            );
            ++header[kDeathCount];
        }
    }
    return packed_deltas;
}

__device__ __forceinline__ std::uint32_t child_recombination_bits(
    const uint2 tick_rng_key,
    const std::uint64_t child_id,
    const std::uint64_t parent_a_id,
    const std::uint64_t parent_b_id,
    const int linked_block
) {
    const uint4 block = counter_rng4x32(
        tick_rng_key,
        kDomainRecombination |
            (static_cast<std::uint64_t>(linked_block) << 32),
        child_id,
        parent_a_id,
        parent_b_id
    );
    return block.x ^ block.y ^ block.z ^ block.w;
}

__device__ __forceinline__ uint2 child_mutation_draws(
    const uint2 tick_rng_key,
    const std::uint64_t child_id,
    const int linked_locus
) {
    const uint4 block = counter_rng4x32(
        tick_rng_key,
        kDomainMutation,
        child_id,
        static_cast<std::uint64_t>(linked_locus),
        0u
    );
    return make_uint2(block.x ^ block.z, block.y ^ block.w);
}

__device__ __forceinline__ bool apply_mutation_draw(
    std::uint8_t* row,
    const int child,
    const int allele,
    const int linked_locus,
    const std::uint32_t draw,
    const std::uint32_t mutation_threshold = kMutationThresholdV2,
    const int mutation_step = 1
) {
    if ((draw & 0xffffu) >= mutation_threshold) {
        return false;
    }
    const int delta = ((draw >> 16) & 1u) != 0u
        ? mutation_step : -mutation_step;
    if (linked_locus < kPolicyLoci) {
        auto* policy = field<std::int8_t>(row, kPolicyAllelesOffset);
        const int index = policy_allele_index(allele, linked_locus, child);
        policy[index] = clamp_i8(static_cast<int>(policy[index]) + delta);
    } else {
        auto* traits = field<std::int16_t>(row, kTraitAllelesOffset);
        const int locus = linked_locus - kPolicyLoci;
        const int index = trait_allele_index(allele, locus, child);
        traits[index] = clamp_i16(static_cast<int>(traits[index]) + delta);
    }
    return true;
}

template <typename ActionGetter>
__device__ void commit_attacks_serial(std::uint8_t* row, ActionGetter action_get) {
    const auto* occupancy = field<std::uint16_t>(row, kOccupancyOffset);
    const auto* position = field<std::uint16_t>(row, kPositionOffset);
    auto* health = field<std::int16_t>(row, kHealthOffset);
    const auto* alive = field<std::uint32_t>(row, kAliveWordsOffset);
    for (int attacker = 0; attacker < kCapacity; ++attacker) {
        if (!is_alive(alive, attacker) || action_get(attacker) != 6) {
            continue;
        }
        const std::uint16_t target_site = movement_target(position[attacker], 2);
        const std::uint16_t target = occupancy[target_site];
        if (target != kEmpty && target != static_cast<std::uint16_t>(attacker) &&
            is_alive(alive, static_cast<int>(target))) {
            health[target] = clamp_i16(static_cast<int>(health[target]) - 3);
        }
    }
}

__device__ __forceinline__ void commit_attack_for_slot(
    std::uint8_t* row, const int attacker, const int action
) {
    if (action != 6) {
        return;
    }
    const auto* occupancy = field<std::uint16_t>(row, kOccupancyOffset);
    const auto* position = field<std::uint16_t>(row, kPositionOffset);
    auto* health = field<std::int16_t>(row, kHealthOffset);
    const auto* alive = field<std::uint32_t>(row, kAliveWordsOffset);
    if (!is_alive(alive, attacker)) {
        return;
    }
    const std::uint16_t target_site = movement_target(position[attacker], 2);
    const std::uint16_t target = occupancy[target_site];
    if (target != kEmpty && target != static_cast<std::uint16_t>(attacker) &&
        is_alive(alive, static_cast<int>(target))) {
        // East-neighbor projection is a bijection on the toroidal site grid.
        // Sole occupancy therefore makes target writes injective across attackers.
        health[target] = clamp_i16(static_cast<int>(health[target]) - 3);
    }
}

template <typename ActionGetter>
__device__ std::uint16_t choose_min_uid_mate(
    const std::uint8_t* row,
    const int slot,
    ActionGetter action_get,
    const int reproduction_energy_threshold = 100
) {
    const auto* occupancy = field<std::uint16_t>(row, kOccupancyOffset);
    const auto* position = field<std::uint16_t>(row, kPositionOffset);
    const auto* energy = field<std::int16_t>(row, kEnergyOffset);
    const auto* alive = field<std::uint32_t>(row, kAliveWordsOffset);
    const auto* ids = field<std::uint64_t>(row, kAgentIdOffset);
    const auto* expressed_traits = field<std::int16_t>(row, kExpressedTraitOffset);
    if (!is_alive(alive, slot) || action_get(slot) != 0 ||
        energy[slot] < reproduction_energy_threshold) {
        return kEmpty;
    }
    const bool sex = expressed_traits[expressed_trait_index(0, slot)] < 0;
    std::uint16_t selected = kEmpty;
    std::uint64_t selected_uid = std::numeric_limits<std::uint64_t>::max();
    for (int direction = 1; direction <= 4; ++direction) {
        const std::uint16_t candidate = occupancy[
            movement_target(position[slot], direction)
        ];
        if (candidate == kEmpty || candidate == static_cast<std::uint16_t>(slot) ||
            !is_alive(alive, static_cast<int>(candidate)) ||
            action_get(static_cast<int>(candidate)) != 0 ||
            energy[candidate] < reproduction_energy_threshold ||
            (expressed_traits[expressed_trait_index(0, candidate)] < 0) == sex) {
            continue;
        }
        if (ids[candidate] < selected_uid) {
            selected = candidate;
            selected_uid = ids[candidate];
        }
    }
    return selected;
}

__device__ std::uint16_t minimum_empty_birth_site(
    const std::uint8_t* row,
    const int parent_a,
    const int parent_b
) {
    const auto* occupancy = field<std::uint16_t>(row, kOccupancyOffset);
    const auto* position = field<std::uint16_t>(row, kPositionOffset);
    std::uint16_t selected = kEmpty;
    for (int parent_index = 0; parent_index < 2; ++parent_index) {
        const int parent = parent_index == 0 ? parent_a : parent_b;
        for (int direction = 1; direction <= 4; ++direction) {
            const std::uint16_t site = movement_target(position[parent], direction);
            if (occupancy[site] == kEmpty &&
                (selected == kEmpty || site < selected)) {
                selected = site;
            }
        }
    }
    return selected;
}

__device__ void inherit_child_genome_reference(
    std::uint8_t* row,
    const int parent_a,
    const int parent_b,
    const int child,
    const uint2 tick_rng_key,
    const std::uint64_t capability_mask,
    const std::uint64_t child_id
) {
    auto* header = field<std::uint64_t>(row, kHeaderOffset);
    auto* policy = field<std::int8_t>(row, kPolicyAllelesOffset);
    auto* traits = field<std::int16_t>(row, kTraitAllelesOffset);
    for (int block = 0; block < kRecombinationBlocks; ++block) {
        int hap_a = 0;
        int hap_b = 0;
        if ((capability_mask & kRecombination) != 0u) {
            const auto* ids = field<std::uint64_t>(row, kAgentIdOffset);
            const std::uint32_t recombination = child_recombination_bits(
                tick_rng_key, child_id, ids[parent_a], ids[parent_b], block
            );
            hap_a = static_cast<int>(recombination & 1u);
            hap_b = static_cast<int>((recombination >> 1) & 1u);
            ++header[kRecombinationCapabilityCounter];
            ++header[kRecombinationCount];
        }
        for (int lane = 0; lane < 8; ++lane) {
            const int linked_locus = block * 8 + lane;
            if (linked_locus < kPolicyLoci) {
                policy[policy_allele_index(0, linked_locus, child)] =
                    policy[policy_allele_index(hap_a, linked_locus, parent_a)];
                policy[policy_allele_index(1, linked_locus, child)] =
                    policy[policy_allele_index(hap_b, linked_locus, parent_b)];
            } else {
                const int locus = linked_locus - kPolicyLoci;
                traits[trait_allele_index(0, locus, child)] =
                    traits[trait_allele_index(hap_a, locus, parent_a)];
                traits[trait_allele_index(1, locus, child)] =
                    traits[trait_allele_index(hap_b, locus, parent_b)];
            }
            if ((capability_mask & kMutation) != 0u) {
                const uint2 mutation = child_mutation_draws(
                    tick_rng_key, child_id, linked_locus
                );
                if (apply_mutation_draw(row, child, 0, linked_locus, mutation.x)) {
                    ++header[kMutationCapabilityCounter];
                    ++header[kMutationCount];
                }
                if (apply_mutation_draw(row, child, 1, linked_locus, mutation.y)) {
                    ++header[kMutationCapabilityCounter];
                    ++header[kMutationCount];
                }
            }
        }
    }
    express_genome(row, child);
}

template <typename ActionGetter, typename ActionSetter>
__device__ void commit_births_reference(
    std::uint8_t* row,
    const std::uint64_t tick,
    const uint2 tick_rng_key,
    const std::uint64_t capability_mask,
    ActionGetter action_get,
    ActionSetter action_set,
    int* site_winner_slots,
    int* slot_plans,
    LifecycleEventRow* events,
    std::int32_t* event_counts,
    const int world,
    const int event_tick_base
) {
    if ((capability_mask & kBirth) == 0u) {
        return;
    }
    auto* header = field<std::uint64_t>(row, kHeaderOffset);
    auto* occupancy = field<std::uint16_t>(row, kOccupancyOffset);
    auto* position = field<std::uint16_t>(row, kPositionOffset);
    auto* energy = field<std::int16_t>(row, kEnergyOffset);
    auto* health = field<std::int16_t>(row, kHealthOffset);
    auto* age = field<std::uint16_t>(row, kAgeOffset);
    auto* absolute_birth_tick = field<std::uint64_t>(row, kAbsoluteBirthTickOffset);
    auto* alive = field<std::uint32_t>(row, kAliveWordsOffset);
    auto* ids = field<std::uint64_t>(row, kAgentIdOffset);
    auto* parent_a_ids = field<std::uint64_t>(row, kParentAIdOffset);
    auto* parent_b_ids = field<std::uint64_t>(row, kParentBIdOffset);
    for (int slot = 0; slot < kCapacity; ++slot) {
        site_winner_slots[slot] = static_cast<int>(
            choose_min_uid_mate(row, slot, action_get)
        );
    }
    for (int slot = 0; slot < kCapacity; ++slot) {
        const int mate = site_winner_slots[slot];
        slot_plans[slot] = -1;
        if (mate == static_cast<int>(kEmpty) || site_winner_slots[mate] != slot ||
            ids[slot] >= ids[mate]) {
            continue;
        }
        const std::uint16_t site = minimum_empty_birth_site(row, slot, mate);
        if (site != kEmpty) {
            slot_plans[slot] = (mate << 16) | static_cast<int>(site);
        }
    }
    for (int site = 0; site < kSites; ++site) {
        site_winner_slots[site] = -1;
    }
    for (int leader = 0; leader < kCapacity; ++leader) {
        if (slot_plans[leader] < 0) {
            continue;
        }
        const int site = slot_plans[leader] & 0xffff;
        const int incumbent = site_winner_slots[site];
        if (incumbent < 0 || ids[leader] < ids[incumbent]) {
            site_winner_slots[site] = leader;
        }
    }
    while (true) {
        int leader = -1;
        std::uint64_t leader_uid = std::numeric_limits<std::uint64_t>::max();
        for (int candidate = 0; candidate < kCapacity; ++candidate) {
            if (slot_plans[candidate] < 0 ||
                site_winner_slots[slot_plans[candidate] & 0xffff] != candidate) {
                continue;
            }
            if (ids[candidate] < leader_uid) {
                leader = candidate;
                leader_uid = ids[candidate];
            }
        }
        if (leader < 0) {
            break;
        }
        const int parent_b = slot_plans[leader] >> 16;
        const int site = slot_plans[leader] & 0xffff;
        slot_plans[leader] = -1;
        int child = -1;
        for (int slot = 0; slot < kCapacity; ++slot) {
            if (!is_alive(alive, slot)) {
                child = slot;
                break;
            }
        }
        if (child < 0) {
            break;
        }
        const std::uint64_t child_id = header[kNextAgentId];
        if (child_id == 0u || child_id >= static_cast<std::uint64_t>(
                std::numeric_limits<std::int64_t>::max())) {
            header[kStatus] = 1u;
            return;
        }
        energy[leader] = static_cast<std::int16_t>(energy[leader] - 30);
        energy[parent_b] = static_cast<std::int16_t>(energy[parent_b] - 30);
        header[kNextAgentId] = child_id + 1u;
        position[child] = static_cast<std::uint16_t>(site);
        energy[child] = 80;
        health[child] = 100;
        age[child] = 0u;
        absolute_birth_tick[child] = tick;
        ids[child] = child_id;
        parent_a_ids[child] = ids[leader];
        parent_b_ids[child] = ids[parent_b];
        inherit_child_genome_reference(
            row, leader, parent_b, child, tick_rng_key, capability_mask, child_id
        );
        __threadfence_block();
        occupancy[site] = static_cast<std::uint16_t>(child);
        set_alive(alive, child);
        action_set(child, 0);
        write_event(
            events, event_counts, world, event_tick_base + kCapacity + child,
            tick, child_id, ids[leader], ids[parent_b], 2u,
            static_cast<std::uint64_t>(child),
            static_cast<std::uint16_t>(site)
        );
        ++header[kBirthCapabilityCounter];
        ++header[kBirthCount];
    }
}

__device__ void commit_child_warp(
    std::uint8_t* row,
    const int parent_a,
    const int parent_b,
    const int child,
    const std::uint16_t site,
    const int birth_rank,
    const std::uint64_t child_id,
    const std::uint64_t tick,
    const uint2 recombination_tick_rng_key,
    const uint2 mutation_tick_rng_key,
    const std::uint64_t capability_mask,
    const std::uint32_t mutation_threshold,
    const int mutation_step,
    std::int64_t* science_agent_metrics,
    std::uint8_t* science_agent_birth_provenance,
    std::uint8_t* actions,
    std::uint16_t* mutation_counts,
    LifecycleEventRow* events,
    const int event_tick_base,
    std::int64_t* science_birth_child_ids,
    std::int64_t* science_birth_parent_a_ids,
    std::int64_t* science_birth_parent_b_ids,
    std::int32_t* science_birth_mutation_counts,
    std::int32_t* science_birth_child_slots,
    std::int32_t* science_birth_sites,
    const std::int64_t science_birth_base
) {
    constexpr unsigned int kFullWarpMask = 0xffffffffu;
    const int lane = static_cast<int>(threadIdx.x) & (kWarpSize - 1);
    auto* occupancy = field<std::uint16_t>(row, kOccupancyOffset);
    auto* position = field<std::uint16_t>(row, kPositionOffset);
    auto* energy = field<std::int16_t>(row, kEnergyOffset);
    auto* health = field<std::int16_t>(row, kHealthOffset);
    auto* age = field<std::uint16_t>(row, kAgeOffset);
    auto* absolute_birth_tick = field<std::uint64_t>(row, kAbsoluteBirthTickOffset);
    auto* alive = field<std::uint32_t>(row, kAliveWordsOffset);
    auto* ids = field<std::uint64_t>(row, kAgentIdOffset);
    auto* parent_a_ids = field<std::uint64_t>(row, kParentAIdOffset);
    auto* parent_b_ids = field<std::uint64_t>(row, kParentBIdOffset);
    auto* policy = field<std::int8_t>(row, kPolicyAllelesOffset);
    auto* traits = field<std::int16_t>(row, kTraitAllelesOffset);
    auto* expressed_policy = field<std::int32_t>(row, kExpressedPolicyOffset);
    auto* expressed_traits = field<std::int16_t>(row, kExpressedTraitOffset);
    const std::uint64_t parent_a_uid = ids[parent_a];
    const std::uint64_t parent_b_uid = ids[parent_b];
    if (lane == 0) {
        position[child] = site;
        energy[child] = 80;
        health[child] = 100;
        age[child] = 0u;
        absolute_birth_tick[child] = tick;
        ids[child] = child_id;
        parent_a_ids[child] = parent_a_uid;
        parent_b_ids[child] = parent_b_uid;
    }
    std::uint32_t recombination = 0u;
    if (lane < kRecombinationBlocks &&
        (capability_mask & kRecombination) != 0u) {
        recombination = child_recombination_bits(
            recombination_tick_rng_key,
            child_id, parent_a_uid, parent_b_uid, lane
        );
    }
    std::uint32_t mutation_count = 0u;
    for (int locus_offset = 0; locus_offset < 4; ++locus_offset) {
        const int linked_locus = lane + locus_offset * kWarpSize;
        const int block = linked_locus >> 3;
        const std::uint32_t block_recombination = __shfl_sync(
            kFullWarpMask, recombination, block
        );
        const int hap_a = (capability_mask & kRecombination) != 0u
            ? static_cast<int>(block_recombination & 1u) : 0;
        const int hap_b = (capability_mask & kRecombination) != 0u
            ? static_cast<int>((block_recombination >> 1) & 1u) : 0;
        if (linked_locus < kPolicyLoci) {
            policy[policy_allele_index(0, linked_locus, child)] =
                policy[policy_allele_index(hap_a, linked_locus, parent_a)];
            policy[policy_allele_index(1, linked_locus, child)] =
                policy[policy_allele_index(hap_b, linked_locus, parent_b)];
        } else {
            const int trait_locus = linked_locus - kPolicyLoci;
            traits[trait_allele_index(0, trait_locus, child)] =
                traits[trait_allele_index(hap_a, trait_locus, parent_a)];
            traits[trait_allele_index(1, trait_locus, child)] =
                traits[trait_allele_index(hap_b, trait_locus, parent_b)];
        }
        std::uint8_t mutation_provenance = 0u;
        if ((capability_mask & kMutation) != 0u) {
            const uint2 mutation = child_mutation_draws(
                mutation_tick_rng_key, child_id, linked_locus
            );
            const bool allele_zero_mutated = apply_mutation_draw(
                row, child, 0, linked_locus, mutation.x,
                mutation_threshold, mutation_step
            );
            const bool allele_one_mutated = apply_mutation_draw(
                row, child, 1, linked_locus, mutation.y,
                mutation_threshold, mutation_step
            );
            mutation_count += allele_zero_mutated ? 1u : 0u;
            mutation_count += allele_one_mutated ? 1u : 0u;
            mutation_provenance |= allele_zero_mutated
                ? (((mutation.x >> 16) & 1u) != 0u ? 1u : 2u)
                : 0u;
            mutation_provenance |= static_cast<std::uint8_t>(
                allele_one_mutated
                    ? (((mutation.y >> 16) & 1u) != 0u ? 1u : 2u) << 2
                    : 0u
            );
        }
        if (science_agent_birth_provenance != nullptr) {
            science_agent_birth_provenance[
                static_cast<std::int64_t>(child) *
                    kScienceBirthProvenanceWidthV1 + linked_locus
            ] = mutation_provenance;
        }
    }
    const unsigned int recombination_haplotype_a = __ballot_sync(
        kFullWarpMask,
        lane < kRecombinationBlocks && (recombination & 1u) != 0u
    );
    const unsigned int recombination_haplotype_b = __ballot_sync(
        kFullWarpMask,
        lane < kRecombinationBlocks && (recombination & 2u) != 0u
    );
    for (int offset = kWarpSize / 2; offset != 0; offset >>= 1) {
        mutation_count += __shfl_down_sync(kFullWarpMask, mutation_count, offset);
    }
    __syncwarp(kFullWarpMask);
    if (lane < kActions * 4) {
        const int action = lane >> 2;
        const int group = lane & 3;
        std::uint32_t packed = 0u;
        for (int component = 0; component < 4; ++component) {
            const int locus = (group * 4 + component) * kActions + action;
            const std::int8_t value = midpoint_i8(
                policy[policy_allele_index(0, locus, child)],
                policy[policy_allele_index(1, locus, child)]
            );
            packed |= static_cast<std::uint32_t>(
                static_cast<std::uint8_t>(value)
            ) << (component * 8);
        }
        expressed_policy[expressed_policy_index(action, group, child)] =
            static_cast<std::int32_t>(packed);
    }
    if (lane < kTraitLoci) {
        expressed_traits[expressed_trait_index(lane, child)] = midpoint_i16(
            traits[trait_allele_index(0, lane, child)],
            traits[trait_allele_index(1, lane, child)]
        );
    }
    if (science_agent_metrics != nullptr && lane < kScienceAgentMetricWidthV1) {
        science_agent_metrics[
            static_cast<std::int64_t>(child) * kScienceAgentMetricWidthV1 + lane
        ] = 0;
    }
    __syncwarp(kFullWarpMask);
    if (lane == 0) {
        if (science_agent_birth_provenance != nullptr) {
            std::uint32_t packed_recombination = 0u;
            for (int block = 0; block < kRecombinationBlocks; ++block) {
                packed_recombination |=
                    ((recombination_haplotype_a >> block) & 1u) << (2 * block);
                packed_recombination |=
                    ((recombination_haplotype_b >> block) & 1u) << (2 * block + 1);
            }
            std::uint8_t* provenance = science_agent_birth_provenance +
                static_cast<std::int64_t>(child) *
                    kScienceBirthProvenanceWidthV1;
            for (int byte = 0; byte < 4; ++byte) {
                provenance[kLinkedLoci + byte] = static_cast<std::uint8_t>(
                    packed_recombination >> (8 * byte)
                );
            }
        }
        mutation_counts[birth_rank] = static_cast<std::uint16_t>(mutation_count);
        if (science_birth_child_ids != nullptr) {
            const std::int64_t output_index = science_birth_base + birth_rank;
            science_birth_child_ids[output_index] = static_cast<std::int64_t>(child_id);
            science_birth_parent_a_ids[output_index] = static_cast<std::int64_t>(parent_a_uid);
            science_birth_parent_b_ids[output_index] = static_cast<std::int64_t>(parent_b_uid);
            science_birth_mutation_counts[output_index] = static_cast<std::int32_t>(mutation_count);
            science_birth_child_slots[output_index] = child;
            science_birth_sites[output_index] = static_cast<int>(site);
        }
        __threadfence_block();
        occupancy[site] = static_cast<std::uint16_t>(child);
        set_alive(alive, child);
        actions[child] = 0u;
        if (events != nullptr) {
            write_event_payload(
                events, event_tick_base + kCapacity + child,
                tick, child_id, parent_a_uid, parent_b_uid, 2u,
                static_cast<std::uint64_t>(child), site
            );
        }
    }
}

__device__ void append_direct_sparse_events_v3(
    const std::uint8_t* row,
    const std::uint8_t* event_flags,
    const int tick_offset,
    const std::uint16_t event_kind,
    SparseLifecycleEventV3* world_events,
    const int event_capacity,
    int* direct_event_count,
    std::int32_t* validation,
    int* warp_counts,
    int* warp_bases,
    int* tile_base_shared,
    int* tile_count_shared,
    const std::int64_t* science_agent_metrics,
    std::int64_t* science_event_metrics,
    const std::uint8_t* science_agent_birth_provenance,
    std::uint8_t* science_event_birth_provenance
) {
    constexpr unsigned int kFullWarpMask = 0xffffffffu;
    const int lane = static_cast<int>(threadIdx.x) & (kWarpSize - 1);
    const int warp = static_cast<int>(threadIdx.x) / kWarpSize;
    const auto* position = field<std::uint16_t>(row, kPositionOffset);
    const auto* ids = field<std::uint64_t>(row, kAgentIdOffset);
    const auto* parent_a_ids = field<std::uint64_t>(row, kParentAIdOffset);
    const auto* parent_b_ids = field<std::uint64_t>(row, kParentBIdOffset);
    for (int slot_base = 0; slot_base < kCapacity; slot_base += kThreads) {
        const int slot = slot_base + static_cast<int>(threadIdx.x);
        const bool committed = slot < kCapacity && event_flags[slot] != 0u;
        const unsigned int committed_mask = __ballot_sync(kFullWarpMask, committed);
        const unsigned int lower_lane_mask =
            lane == 0 ? 0u : ((1u << lane) - 1u);
        const int rank_in_warp = __popc(committed_mask & lower_lane_mask);
        if (lane == 0) {
            warp_counts[warp] = __popc(committed_mask);
        }
        __syncthreads();
        if (threadIdx.x < kWarpsPerBlock) {
            int prefix = 0;
            for (int prior = 0; prior < static_cast<int>(threadIdx.x); ++prior) {
                prefix += warp_counts[prior];
            }
            warp_bases[threadIdx.x] = prefix;
            if (threadIdx.x == kWarpsPerBlock - 1) {
                *tile_count_shared = prefix + warp_counts[threadIdx.x];
            }
        }
        if (threadIdx.x == 0) {
            *tile_base_shared = *direct_event_count;
        }
        __syncthreads();
        if (committed) {
            const int output_index = *tile_base_shared +
                warp_bases[warp] + rank_in_warp;
            if (output_index < event_capacity) {
                SparseLifecycleEventV3 packed{};
                packed.agent_id = ids[slot];
                packed.parent_a_id = parent_a_ids[slot];
                packed.parent_b_id = parent_b_ids[slot];
                packed.packed_key_and_reserved =
                    static_cast<std::uint64_t>(tick_offset) |
                    (static_cast<std::uint64_t>(event_kind) << 6) |
                    (static_cast<std::uint64_t>(slot) << 8) |
                    (static_cast<std::uint64_t>(position[slot]) << 17);
                world_events[output_index] = packed;
                if (science_event_metrics != nullptr) {
                    for (int metric = 0; metric < kScienceAgentMetricWidthV1;
                         ++metric) {
                        science_event_metrics[
                            static_cast<std::int64_t>(output_index) *
                                kScienceAgentMetricWidthV1 + metric
                        ] = event_kind == 1u
                            ? science_agent_metrics[
                                static_cast<std::int64_t>(slot) *
                                    kScienceAgentMetricWidthV1 + metric
                            ]
                            : 0;
                    }
                }
                if (science_event_birth_provenance != nullptr) {
                    for (int field = 0; field < kScienceBirthProvenanceWidthV1;
                         ++field) {
                        science_event_birth_provenance[
                            static_cast<std::int64_t>(output_index) *
                                kScienceBirthProvenanceWidthV1 + field
                        ] = event_kind == 2u
                            ? science_agent_birth_provenance[
                                static_cast<std::int64_t>(slot) *
                                    kScienceBirthProvenanceWidthV1 + field
                            ]
                            : 0u;
                    }
                }
            } else {
                atomicOr(validation, kSparseDirectOverflow);
            }
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            *direct_event_count += *tile_count_shared;
        }
        __syncthreads();
    }
}

__device__ void finalize_tick(std::uint8_t* row) {
    auto* header = field<std::uint64_t>(row, kHeaderOffset);
    const auto* alive = field<std::uint32_t>(row, kAliveWordsOffset);
    const auto* due = field<std::uint64_t>(row, kWdlDueOffset);
    const auto* jungle = field<std::uint8_t>(row, kJungleOffset);
    const auto* pending = field<std::uint16_t>(row, kEedPendingOffset);
    std::uint64_t active_count = 0u;
    std::uint64_t due_count = 0u;
    std::uint64_t jungle_count = 0u;
    std::uint64_t pending_count = 0u;
    for (int slot = 0; slot < kCapacity; ++slot) {
        active_count += is_alive(alive, slot) ? 1u : 0u;
    }
    for (int site = 0; site < kSites; ++site) {
        due_count += due[site] != 0u ? 1u : 0u;
        jungle_count += jungle[site] != 0u ? 1u : 0u;
        pending_count += pending[site] != 0u ? 1u : 0u;
    }
    const std::uint64_t completed_tick = header[kNextAbsoluteTick];
    header[kActiveCount] = active_count;
    header[kWdlDueCount] = due_count;
    header[kJungleActiveCount] = jungle_count;
    header[kEedPendingCount] = pending_count;
    ++header[kTicksCompleted];
    header[kStatus] = 0u;
    __threadfence();
    header[kNextAbsoluteTick] = completed_tick + 1u;
}

__device__ void finalize_tick_parallel(
    std::uint8_t* row,
    std::uint32_t* warp_summary_counts
) {
    auto* header = field<std::uint64_t>(row, kHeaderOffset);
    const auto* alive = field<std::uint32_t>(row, kAliveWordsOffset);
    const auto* due = field<std::uint64_t>(row, kWdlDueOffset);
    const auto* jungle = field<std::uint8_t>(row, kJungleOffset);
    const auto* pending = field<std::uint16_t>(row, kEedPendingOffset);
    std::uint32_t active_count = 0u;
    std::uint32_t due_count = 0u;
    std::uint32_t jungle_count = 0u;
    std::uint32_t pending_count = 0u;
    for (int slot = static_cast<int>(threadIdx.x); slot < kCapacity; slot += kThreads) {
        active_count += is_alive(alive, slot) ? 1u : 0u;
    }
    for (int site = static_cast<int>(threadIdx.x); site < kSites; site += kThreads) {
        due_count += due[site] != 0u ? 1u : 0u;
        jungle_count += jungle[site] != 0u ? 1u : 0u;
        pending_count += pending[site] != 0u ? 1u : 0u;
    }
    constexpr unsigned int kFullWarpMask = 0xffffffffu;
    for (int offset = kWarpSize / 2; offset != 0; offset >>= 1) {
        active_count += __shfl_down_sync(kFullWarpMask, active_count, offset);
        due_count += __shfl_down_sync(kFullWarpMask, due_count, offset);
        jungle_count += __shfl_down_sync(kFullWarpMask, jungle_count, offset);
        pending_count += __shfl_down_sync(kFullWarpMask, pending_count, offset);
    }
    const int lane = static_cast<int>(threadIdx.x) & (kWarpSize - 1);
    const int warp = static_cast<int>(threadIdx.x) / kWarpSize;
    if (lane == 0) {
        warp_summary_counts[warp] = active_count;
        warp_summary_counts[kWarpsPerBlock + warp] = due_count;
        warp_summary_counts[2 * kWarpsPerBlock + warp] = jungle_count;
        warp_summary_counts[3 * kWarpsPerBlock + warp] = pending_count;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        std::uint64_t totals[4] = {0u, 0u, 0u, 0u};
        for (int warp_index = 0; warp_index < kWarpsPerBlock; ++warp_index) {
            totals[0] += warp_summary_counts[warp_index];
            totals[1] += warp_summary_counts[kWarpsPerBlock + warp_index];
            totals[2] += warp_summary_counts[2 * kWarpsPerBlock + warp_index];
            totals[3] += warp_summary_counts[3 * kWarpsPerBlock + warp_index];
        }
        const std::uint64_t completed_tick = header[kNextAbsoluteTick];
        header[kActiveCount] = totals[0];
        header[kWdlDueCount] = totals[1];
        header[kJungleActiveCount] = totals[2];
        header[kEedPendingCount] = totals[3];
        ++header[kTicksCompleted];
        header[kStatus] = 0u;
        __threadfence();
        header[kNextAbsoluteTick] = completed_tick + 1u;
    }
}

__global__ void initialize_kernel(
    std::uint8_t* state,
    const std::int64_t row_stride,
    const std::int64_t* world_uids,
    const std::uint64_t seed,
    const std::uint64_t next_absolute_tick,
    const std::int64_t* science_world_controls,
    const std::int64_t science_world_control_stride
) {
    std::uint8_t* row = state + static_cast<std::int64_t>(blockIdx.x) * row_stride;
    auto* header = field<std::uint64_t>(row, kHeaderOffset);
    auto* occupancy = field<std::uint16_t>(row, kOccupancyOffset);
    auto* resource = field<std::uint16_t>(row, kResourceOffset);
    auto* scent = field<std::uint16_t>(row, kScentOffset);
    auto* waste = field<std::uint16_t>(row, kWasteOffset);
    auto* impact = field<std::uint16_t>(row, kEedImpactOffset);
    auto* pending = field<std::uint16_t>(row, kEedPendingOffset);
    auto* due = field<std::uint64_t>(row, kWdlDueOffset);
    auto* active = field<std::uint8_t>(row, kWdlActiveOffset);
    auto* terrain = field<std::uint8_t>(row, kTerrainOffset);
    auto* jungle = field<std::uint8_t>(row, kJungleOffset);
    auto* position = field<std::uint16_t>(row, kPositionOffset);
    auto* energy = field<std::int16_t>(row, kEnergyOffset);
    auto* health = field<std::int16_t>(row, kHealthOffset);
    auto* age = field<std::uint16_t>(row, kAgeOffset);
    auto* absolute_birth_tick = field<std::uint64_t>(row, kAbsoluteBirthTickOffset);
    auto* alive = field<std::uint32_t>(row, kAliveWordsOffset);
    auto* ids = field<std::uint64_t>(row, kAgentIdOffset);
    auto* parent_a = field<std::uint64_t>(row, kParentAIdOffset);
    auto* parent_b = field<std::uint64_t>(row, kParentBIdOffset);
    auto* policy = field<std::int8_t>(row, kPolicyAllelesOffset);
    auto* traits = field<std::int16_t>(row, kTraitAllelesOffset);
    const std::int64_t* science_world_control =
        science_world_controls == nullptr ? nullptr : science_world_controls +
            static_cast<std::int64_t>(blockIdx.x) * science_world_control_stride;
    if (threadIdx.x < 24) {
        header[threadIdx.x] = 0u;
    }
    if (threadIdx.x < 16) {
        alive[threadIdx.x] = 0u;
    }
    if (threadIdx.x == 0) {
        const std::uint64_t uid = static_cast<std::uint64_t>(world_uids[blockIdx.x]);
        header[kWorldUid] = uid;
        header[kNextAbsoluteTick] = next_absolute_tick;
        header[kNextAgentId] = static_cast<std::uint64_t>(kInitialPopulation + 1);
        header[kActiveCount] = kInitialPopulation;
        header[kRngKey] = science_world_control == nullptr ? seed :
            static_cast<std::uint64_t>(
                science_world_control[kScienceSeedInitialization]
            );
        header[kRngEpoch] = kRngSemanticEpoch;
    }
    __syncthreads();
    const uint2 initial_site_rng_key = science_world_control == nullptr
        ? rng_tick_key(header, next_absolute_tick)
        : rng_tick_key_values(
            static_cast<std::uint64_t>(
                science_world_control[kScienceSeedEcology]
            ),
            static_cast<std::uint64_t>(
                science_world_control[kScienceSeedTaskEnvironment]
            ),
            next_absolute_tick
        );
    const uint2 initial_policy_rng_key = science_world_control == nullptr
        ? initial_site_rng_key
        : rng_tick_key_values(
            static_cast<std::uint64_t>(
                science_world_control[kScienceSeedPolicy]
            ),
            static_cast<std::uint64_t>(
                science_world_control[kScienceSeedTaskEnvironment]
            ),
            next_absolute_tick
        );
    for (int site = static_cast<int>(threadIdx.x); site < kSites; site += kThreads) {
        occupancy[site] = kEmpty;
        const std::uint32_t site_draw = counter_rng32(
            initial_site_rng_key, kDomainInitialSite,
            static_cast<std::uint64_t>(site), 0u, 0u
        );
        resource[site] = static_cast<std::uint16_t>(48u + (site_draw & 31u));
        scent[site] = 0u;
        waste[site] = 0u;
        impact[site] = 0u;
        pending[site] = 0u;
        const std::uint64_t delay = static_cast<std::uint64_t>(site % 3 + 1);
        const std::uint64_t max_tick = static_cast<std::uint64_t>(
            std::numeric_limits<std::int64_t>::max()
        );
        due[site] = (site % 97 == 0)
            ? (next_absolute_tick <= max_tick - delay
                ? next_absolute_tick + delay : max_tick)
            : 0u;
        active[site] = 0u;
        terrain[site] = static_cast<std::uint8_t>((site_draw >> 16) & 3u);
        jungle[site] = static_cast<std::uint8_t>((site_draw >> 8) & 31u);
    }
    for (int slot = static_cast<int>(threadIdx.x); slot < kCapacity; slot += kThreads) {
        position[slot] = 0u;
        energy[slot] = 0;
        health[slot] = 0;
        age[slot] = 0u;
        absolute_birth_tick[slot] = 0u;
        ids[slot] = 0u;
        parent_a[slot] = 0u;
        parent_b[slot] = 0u;
        for (int allele = 0; allele < 2; ++allele) {
            for (int locus = 0; locus < kPolicyLoci; ++locus) {
                policy[policy_allele_index(allele, locus, slot)] = 0;
            }
            for (int locus = 0; locus < kTraitLoci; ++locus) {
                traits[trait_allele_index(allele, locus, slot)] = 0;
            }
        }
    }
    __syncthreads();
    const int slot = static_cast<int>(threadIdx.x);
    if (slot < kInitialPopulation) {
        const int initial_site = slot == 34 ? 256 : slot;
        position[slot] = static_cast<std::uint16_t>(initial_site);
        occupancy[initial_site] = static_cast<std::uint16_t>(slot);
        energy[slot] = static_cast<std::int16_t>(slot == 2 || slot == 3 ? 300 : 160 + slot % 32);
        health[slot] = static_cast<std::int16_t>(slot == 0 ? 2 : 100);
        age[slot] = 0u;
        absolute_birth_tick[slot] = next_absolute_tick;
        ids[slot] = static_cast<std::uint64_t>(slot + 1);
        int chosen_action = slot % kActions;
        if (slot == 2 || slot == 3) {
            chosen_action = 0;
        }
        for (int allele = 0; allele < 2; ++allele) {
            for (int locus = 0; locus < kPolicyLoci; ++locus) {
                const std::uint32_t draw = counter_rng32(
                    initial_policy_rng_key, kDomainInitialPolicy,
                    ids[slot], static_cast<std::uint64_t>(allele),
                    static_cast<std::uint64_t>(locus)
                );
                int value = static_cast<int>(draw % 5u) - 2;
                if (locus / kActions == 0 && locus % kActions == chosen_action) {
                    value = 100;
                }
                policy[policy_allele_index(allele, locus, slot)] =
                    static_cast<std::int8_t>(value);
            }
            for (int locus = 0; locus < kTraitLoci; ++locus) {
                const std::uint32_t draw = counter_rng32(
                    initial_policy_rng_key, kDomainInitialTrait,
                    ids[slot], static_cast<std::uint64_t>(allele),
                    static_cast<std::uint64_t>(locus)
                );
                traits[trait_allele_index(allele, locus, slot)] =
                    static_cast<std::int16_t>(
                        locus == 0 && (ids[slot] & 1u) != 0u
                            ? -(8 + static_cast<int>(draw % 25u))
                            : 8 + static_cast<int>(draw % 25u)
                    );
            }
        }
        set_alive(alive, slot);
    }
    __syncthreads();
    for (int cache_slot = static_cast<int>(threadIdx.x); cache_slot < kCapacity;
         cache_slot += kThreads) {
        express_genome(row, cache_slot);
    }
}

__global__ void advance_reference_kernel(
    std::uint8_t* state,
    const std::int64_t row_stride,
    std::uint8_t* scratch,
    const std::int64_t scratch_stride,
    LifecycleEventRow* events,
    const std::int64_t event_stride_rows,
    std::int32_t* event_counts,
    const int temporal_k,
    const std::uint64_t capability_mask
) {
    if (threadIdx.x != 0) {
        return;
    }
    std::uint8_t* row = state + static_cast<std::int64_t>(blockIdx.x) * row_stride;
    auto* plan_winner = reinterpret_cast<std::int32_t*>(
        scratch + static_cast<std::int64_t>(blockIdx.x) * scratch_stride
    );
    auto* slot_plan = plan_winner + kSites;
    auto* header = field<std::uint64_t>(row, kHeaderOffset);
    auto* occupancy = field<std::uint16_t>(row, kOccupancyOffset);
    auto* position = field<std::uint16_t>(row, kPositionOffset);
    auto* energy = field<std::int16_t>(row, kEnergyOffset);
    auto* alive = field<std::uint32_t>(row, kAliveWordsOffset);
    const auto* ids = field<std::uint64_t>(row, kAgentIdOffset);
    LifecycleEventRow* world_events = events +
        static_cast<std::int64_t>(blockIdx.x) * event_stride_rows;
    const std::uint64_t max_signed = static_cast<std::uint64_t>(
        std::numeric_limits<std::int64_t>::max()
    );
    if (header[kStatus] != 0u) {
        return;
    }
    if (header[kNextAbsoluteTick] > max_signed - static_cast<std::uint64_t>(temporal_k)) {
        header[kStatus] = 1u;
        return;
    }
    const std::uint64_t id_headroom = static_cast<std::uint64_t>(
        kCapacity * temporal_k
    );
    if (header[kNextAgentId] == 0u || header[kNextAgentId] > max_signed - id_headroom) {
        header[kStatus] = 2u;
        return;
    }
    event_counts[blockIdx.x] = 0;
    for (std::int64_t row_index = 0; row_index < event_stride_rows; ++row_index) {
        clear_event(world_events, static_cast<int>(row_index));
    }
    for (int temporal = 0; temporal < temporal_k; ++temporal) {
        const std::uint64_t tick = header[kNextAbsoluteTick];
        if (tick >= static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) {
            header[kStatus] = 1u;
            return;
        }
        if (header[kNextAgentId] == 0u || header[kNextAgentId] >
                static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()) -
                    static_cast<std::uint64_t>(kCapacity)) {
            header[kStatus] = 2u;
            return;
        }
        const int event_tick_base = temporal * kEventRowsPerTick;
        const uint2 tick_rng_key = rng_tick_key(header, tick);
        for (int site = 0; site < kSites; ++site) {
            (void)update_site(
                row, site, tick, tick_rng_key, capability_mask, nullptr, false
            );
            plan_winner[site] = std::numeric_limits<int>::max();
        }
        for (int slot = 0; slot < kCapacity; ++slot) {
            int action = 0;
            std::uint16_t target = position[slot];
            if (is_alive(alive, slot)) {
                action = controller_action(row, slot);
                target = movement_target(position[slot], action);
                if (action >= 1 && action <= 4 && occupancy[target] == kEmpty) {
                    const int old_winner = plan_winner[target];
                    if (old_winner == std::numeric_limits<int>::max() ||
                        ids[slot] < ids[old_winner]) {
                        plan_winner[target] = slot;
                    }
                }
            }
            slot_plan[slot] = (action << 16) | static_cast<int>(target);
        }
        for (int slot = 0; slot < kCapacity; ++slot) {
            const int action = (slot_plan[slot] >> 16) & 0xff;
            const int target = slot_plan[slot] & 0xffff;
            if (is_alive(alive, slot) && action >= 1 && action <= 4 &&
                plan_winner[target] == slot) {
                occupancy[position[slot]] = kEmpty;
            }
        }
        for (int site = 0; site < kSites; ++site) {
            const int winner = plan_winner[site];
            if (winner != std::numeric_limits<int>::max()) {
                occupancy[site] = static_cast<std::uint16_t>(winner);
                position[winner] = static_cast<std::uint16_t>(site);
            }
        }
        auto attack_action_get = [=] __device__ (const int slot) {
            return (slot_plan[slot] >> 16) & 0xff;
        };
        commit_attacks_serial(row, attack_action_get);
        for (int slot = 0; slot < kCapacity; ++slot) {
            if (is_alive(alive, slot)) {
                (void)apply_agent_effects(
                    row, slot, (slot_plan[slot] >> 16) & 0xff,
                    capability_mask, nullptr, false, world_events,
                    event_counts, static_cast<int>(blockIdx.x),
                    event_tick_base + slot, tick
                );
            }
        }
        auto action_get = [=] __device__ (const int slot) {
            return (slot_plan[slot] >> 16) & 0xff;
        };
        auto action_set = [=] __device__ (const int slot, const int action) {
            slot_plan[slot] = action << 16;
        };
        commit_births_reference(
            row, tick, tick_rng_key, capability_mask,
            action_get, action_set, plan_winner, slot_plan,
            world_events, event_counts, static_cast<int>(blockIdx.x), event_tick_base
        );
        if (header[kStatus] != 0u) {
            return;
        }
        finalize_tick(row);
    }
}

__global__ __launch_bounds__(kThreads, 3) void advance_optimized_kernel(
    std::uint8_t* state,
    const std::int64_t row_stride,
    LifecycleEventRow* events,
    const std::int64_t event_stride_rows,
    std::int32_t* event_counts,
    SparseLifecycleEventV3* direct_sparse_events,
    const std::int64_t direct_sparse_stride_rows,
    const int direct_sparse_capacity_rows,
    std::int32_t* direct_sparse_validation,
    std::int64_t* science_action_ids,
    std::uint8_t* science_action_values,
    std::int64_t* science_birth_child_ids,
    std::int64_t* science_birth_parent_a_ids,
    std::int64_t* science_birth_parent_b_ids,
    std::int32_t* science_birth_mutation_counts,
    std::int32_t* science_birth_child_slots,
    std::int32_t* science_birth_sites,
    std::int32_t* science_birth_counts,
    const std::int64_t* science_world_controls,
    const std::int64_t science_world_control_stride,
    std::int64_t* science_agent_metrics,
    const std::int64_t science_agent_metric_stride,
    std::int64_t* science_event_metrics,
    const std::int64_t science_event_metric_stride,
    std::uint8_t* science_agent_birth_provenance,
    const std::int64_t science_agent_birth_provenance_stride,
    std::uint8_t* science_event_birth_provenance,
    const std::int64_t science_event_birth_provenance_stride,
    const int temporal_k,
    const std::uint64_t uniform_capability_mask
) {
    __shared__ unsigned long long winner_uids[kSites];
    __shared__ std::uint8_t actions[kCapacity];
    __shared__ std::uint16_t targets[kCapacity];
    __shared__ std::uint16_t mate_choices[kCapacity];
    __shared__ std::uint16_t pair_sites[kCapacity];
    __shared__ std::uint16_t leader_slots_by_rank[kCapacity / 2];
    __shared__ std::uint16_t child_slots_by_rank[kCapacity / 2];
    __shared__ std::uint16_t mutation_counts_by_rank[kCapacity / 2];
    __shared__ std::int16_t science_energy_before[kCapacity];
    __shared__ std::int16_t science_health_before[kCapacity];
    __shared__ std::uint32_t warp_summary_counts[4 * kWarpsPerBlock];
    __shared__ std::uint64_t warp_counter_deltas[kWarpsPerBlock];
    __shared__ uint2 site_tick_rng_key;
    __shared__ uint2 recombination_tick_rng_key;
    __shared__ uint2 mutation_tick_rng_key;
    __shared__ std::uint64_t birth_id_base;
    __shared__ int accepted_birth_count;
    __shared__ std::uint8_t direct_event_flags[kCapacity];
    __shared__ int direct_event_count;
    __shared__ int direct_warp_counts[kWarpsPerBlock];
    __shared__ int direct_warp_bases[kWarpsPerBlock];
    __shared__ int direct_tile_base;
    __shared__ int direct_tile_count;
    std::uint8_t* row = state + static_cast<std::int64_t>(blockIdx.x) * row_stride;
    auto* header = field<std::uint64_t>(row, kHeaderOffset);
    auto* occupancy = field<std::uint16_t>(row, kOccupancyOffset);
    auto* position = field<std::uint16_t>(row, kPositionOffset);
    auto* energy = field<std::int16_t>(row, kEnergyOffset);
    auto* health = field<std::int16_t>(row, kHealthOffset);
    const auto* terrain = field<std::uint8_t>(row, kTerrainOffset);
    const auto* resource = field<std::uint16_t>(row, kResourceOffset);
    const auto* expressed_traits = field<std::int16_t>(
        row, kExpressedTraitOffset
    );
    auto* alive = field<std::uint32_t>(row, kAliveWordsOffset);
    const auto* ids = field<std::uint64_t>(row, kAgentIdOffset);
    const std::int64_t* science_world_control =
        science_world_controls == nullptr ? nullptr : science_world_controls +
            static_cast<std::int64_t>(blockIdx.x) * science_world_control_stride;
    std::int64_t* world_science_agent_metrics =
        science_agent_metrics == nullptr ? nullptr : science_agent_metrics +
            static_cast<std::int64_t>(blockIdx.x) * science_agent_metric_stride;
    std::int64_t* world_science_event_metrics =
        science_event_metrics == nullptr ? nullptr : science_event_metrics +
            static_cast<std::int64_t>(blockIdx.x) * science_event_metric_stride;
    std::uint8_t* world_science_agent_birth_provenance =
        science_agent_birth_provenance == nullptr
        ? nullptr
        : science_agent_birth_provenance +
            static_cast<std::int64_t>(blockIdx.x) *
                science_agent_birth_provenance_stride;
    std::uint8_t* world_science_event_birth_provenance =
        science_event_birth_provenance == nullptr
        ? nullptr
        : science_event_birth_provenance +
            static_cast<std::int64_t>(blockIdx.x) *
                science_event_birth_provenance_stride;
    const std::uint64_t capability_mask = science_world_control == nullptr
        ? uniform_capability_mask
        : static_cast<std::uint64_t>(
            science_world_control[kScienceCapabilityMask]
        );
    const std::uint32_t mutation_threshold = science_world_control == nullptr
        ? static_cast<std::uint32_t>(kMutationThresholdV2)
        : static_cast<std::uint32_t>(
            science_world_control[kScienceMutationThreshold]
        );
    const int mutation_step = science_world_control == nullptr
        ? 1 : static_cast<int>(science_world_control[kScienceMutationStep]);
    const int reproduction_energy_threshold = science_world_control == nullptr
        ? 100 : clamp_int(
            static_cast<int>((
                science_world_control[kScienceReproductionEnergyThreshold] *
                science_world_control[kScienceSelectionPressureQ16]
            ) >> 16), 1, 32767
        );
    const int reproduction_energy_cost = science_world_control == nullptr
        ? 30 : static_cast<int>(
            science_world_control[kScienceReproductionEnergyCost]
        );
    const bool direct_sparse = direct_sparse_events != nullptr;
    LifecycleEventRow* world_events = events == nullptr ? nullptr : events +
        static_cast<std::int64_t>(blockIdx.x) * event_stride_rows;
    SparseLifecycleEventV3* world_direct_sparse_events =
        direct_sparse_events == nullptr ? nullptr : direct_sparse_events +
        static_cast<std::int64_t>(blockIdx.x) * direct_sparse_stride_rows;
    const std::uint64_t max_signed = static_cast<std::uint64_t>(
        std::numeric_limits<std::int64_t>::max()
    );
    if (threadIdx.x == 0) {
        event_counts[blockIdx.x] = 0;
        direct_event_count = 0;
        if (direct_sparse) {
            direct_sparse_validation[blockIdx.x] = 0;
        }
    }
    __syncthreads();
    if (header[kStatus] != 0u) {
        return;
    }
    if (header[kNextAbsoluteTick] > max_signed - static_cast<std::uint64_t>(temporal_k)) {
        if (threadIdx.x == 0) {
            header[kStatus] = 1u;
        }
        return;
    }
    const std::uint64_t id_headroom = static_cast<std::uint64_t>(
        kCapacity * temporal_k
    );
    if (header[kNextAgentId] == 0u || header[kNextAgentId] > max_signed - id_headroom) {
        if (threadIdx.x == 0) {
            header[kStatus] = 2u;
        }
        return;
    }
    if (!direct_sparse) {
        for (std::int64_t row_index = static_cast<std::int64_t>(threadIdx.x);
             row_index < event_stride_rows; row_index += kThreads) {
            clear_event(world_events, static_cast<int>(row_index));
        }
    }
    __syncthreads();
    for (int temporal = 0; temporal < temporal_k; ++temporal) {
        const std::uint64_t tick = header[kNextAbsoluteTick];
        if (tick >= static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) {
            if (threadIdx.x == 0) {
                header[kStatus] = 1u;
            }
            return;
        }
        if (header[kNextAgentId] == 0u || header[kNextAgentId] >
                static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()) -
                    static_cast<std::uint64_t>(kCapacity)) {
            if (threadIdx.x == 0) {
                header[kStatus] = 2u;
            }
            return;
        }
        const int event_tick_base = temporal * kEventRowsPerTick;
        const std::int64_t science_tick_base =
            (static_cast<std::int64_t>(blockIdx.x) * temporal_k + temporal) * kCapacity;
        const std::int64_t science_birth_base =
            (static_cast<std::int64_t>(blockIdx.x) * temporal_k + temporal) *
            kScienceBirthCapacity;
        if (threadIdx.x == 0 && science_birth_counts != nullptr) {
            science_birth_counts[
                static_cast<std::int64_t>(blockIdx.x) * temporal_k + temporal
            ] = 0;
        }
        if (threadIdx.x == 0) {
            if (science_world_control == nullptr) {
                site_tick_rng_key = rng_tick_key(header, tick);
                recombination_tick_rng_key = site_tick_rng_key;
                mutation_tick_rng_key = site_tick_rng_key;
            } else {
                const std::uint64_t rng_subject = static_cast<std::uint64_t>(
                    science_world_control[kScienceSeedTaskEnvironment]
                );
                site_tick_rng_key = rng_tick_key_values(
                    static_cast<std::uint64_t>(
                        science_world_control[kScienceSeedEcology]
                    ),
                    rng_subject,
                    tick
                );
                recombination_tick_rng_key = rng_tick_key_values(
                    static_cast<std::uint64_t>(
                        science_world_control[kScienceSeedPolicy]
                    ),
                    rng_subject,
                    tick
                );
                mutation_tick_rng_key = rng_tick_key_values(
                    static_cast<std::uint64_t>(
                        science_world_control[kScienceSeedMutation]
                    ),
                    rng_subject,
                    tick
                );
            }
        }
        __syncthreads();
        std::uint64_t packed_tick_deltas = 0u;
        for (int site = static_cast<int>(threadIdx.x); site < kSites; site += kThreads) {
            packed_tick_deltas += update_site(
                row, site, tick, site_tick_rng_key, capability_mask,
                science_world_control, true
            );
        }
        __syncthreads();
        for (int slot = static_cast<int>(threadIdx.x); slot < kCapacity; slot += kThreads) {
            int action = 0;
            std::uint16_t target = position[slot];
            if (is_alive(alive, slot)) {
                action = controller_action(row, slot);
                target = movement_target(position[slot], action);
                if (world_science_agent_metrics != nullptr) {
                    std::int64_t* metrics = world_science_agent_metrics +
                        static_cast<std::int64_t>(slot) *
                            kScienceAgentMetricWidthV1;
                    ++metrics[kScienceAction0Count + action];
                    ++metrics[kScienceAliveTicks];
                    science_energy_before[slot] = energy[slot];
                    science_health_before[slot] = health[slot];
                    if (action >= 1 && action <= 4) {
                        ++metrics[kScienceMovementAttempt];
                        metrics[kScienceMovementCostTotal] +=
                            (science_world_control == nullptr ? 1 :
                                science_world_control[kScienceMovementCost]) +
                            static_cast<std::int64_t>(terrain[position[slot]]) +
                            (abs(static_cast<int>(expressed_traits[
                                expressed_trait_index(1, slot)
                            ])) & 1);
                    } else if (action == 5 && resource[position[slot]] != 0u) {
                        const int uptake_base = science_world_control == nullptr
                            ? 4 : static_cast<int>(science_world_control[
                                kScienceResourceUptakeBase
                            ]);
                        metrics[kScienceResourceUptake] += min(
                            static_cast<int>(resource[position[slot]]),
                            uptake_base + (abs(static_cast<int>(expressed_traits[
                                expressed_trait_index(2, slot)
                            ])) & 3)
                        );
                    }
                }
            }
            actions[slot] = static_cast<std::uint8_t>(action);
            targets[slot] = target;
            if (science_action_ids != nullptr) {
                science_action_ids[science_tick_base + slot] =
                    is_alive(alive, slot) ? static_cast<std::int64_t>(ids[slot]) : 0;
                science_action_values[science_tick_base + slot] =
                    is_alive(alive, slot) ? static_cast<std::uint8_t>(action) : 0u;
            }
        }
        __syncthreads();
        for (int site = static_cast<int>(threadIdx.x); site < kSites; site += kThreads) {
            winner_uids[site] = std::numeric_limits<unsigned long long>::max();
        }
        __syncthreads();
        for (int slot = static_cast<int>(threadIdx.x); slot < kCapacity; slot += kThreads) {
            if (is_alive(alive, slot) && actions[slot] >= 1u && actions[slot] <= 4u &&
                occupancy[targets[slot]] == kEmpty) {
                atomicMin(winner_uids + targets[slot],
                    static_cast<unsigned long long>(ids[slot]));
            }
        }
        __syncthreads();
        for (int slot = static_cast<int>(threadIdx.x); slot < kCapacity; slot += kThreads) {
            if (is_alive(alive, slot) && actions[slot] >= 1u && actions[slot] <= 4u &&
                winner_uids[targets[slot]] == ids[slot]) {
                occupancy[position[slot]] = kEmpty;
            }
        }
        __syncthreads();
        for (int slot = static_cast<int>(threadIdx.x); slot < kCapacity; slot += kThreads) {
            if (is_alive(alive, slot) && actions[slot] >= 1u && actions[slot] <= 4u &&
                winner_uids[targets[slot]] == ids[slot]) {
                occupancy[targets[slot]] = static_cast<std::uint16_t>(slot);
                position[slot] = targets[slot];
                if (world_science_agent_metrics != nullptr) {
                    ++world_science_agent_metrics[
                        static_cast<std::int64_t>(slot) *
                            kScienceAgentMetricWidthV1 +
                        kScienceMovementSuccess
                    ];
                }
            }
        }
        __syncthreads();
        if (header[kStatus] != 0u) {
            return;
        }
        if (direct_sparse) {
            for (int slot = static_cast<int>(threadIdx.x); slot < kCapacity;
                 slot += kThreads) {
                direct_event_flags[slot] = 0u;
            }
            __syncthreads();
        }
        for (int attacker = static_cast<int>(threadIdx.x); attacker < kCapacity;
             attacker += kThreads) {
            if (world_science_agent_metrics != nullptr &&
                is_alive(alive, attacker) && actions[attacker] == 6u) {
                std::int64_t* metrics = world_science_agent_metrics +
                    static_cast<std::int64_t>(attacker) *
                        kScienceAgentMetricWidthV1;
                ++metrics[kScienceAttackAttempt];
                const std::uint16_t target_site = movement_target(
                    position[attacker], 2
                );
                const std::uint16_t target = occupancy[target_site];
                if (target != kEmpty &&
                    target != static_cast<std::uint16_t>(attacker) &&
                    is_alive(alive, static_cast<int>(target))) {
                    ++metrics[kScienceAttackHit];
                }
            }
            commit_attack_for_slot(row, attacker, static_cast<int>(actions[attacker]));
        }
        __syncthreads();
        for (int slot = static_cast<int>(threadIdx.x); slot < kCapacity; slot += kThreads) {
            if (is_alive(alive, slot)) {
                const std::uint64_t agent_deltas = apply_agent_effects(
                    row, slot, static_cast<int>(actions[slot]), capability_mask,
                    science_world_control, true,
                    world_events, event_counts, static_cast<int>(blockIdx.x),
                    event_tick_base + slot, tick
                );
                packed_tick_deltas += agent_deltas;
                if (world_science_agent_metrics != nullptr) {
                    std::int64_t* metrics = world_science_agent_metrics +
                        static_cast<std::int64_t>(slot) *
                            kScienceAgentMetricWidthV1;
                    metrics[kScienceEnergyDelta] +=
                        static_cast<std::int64_t>(energy[slot]) -
                        science_energy_before[slot];
                    metrics[kScienceHealthDamage] += max(
                        0,
                        static_cast<int>(science_health_before[slot]) -
                            static_cast<int>(health[slot])
                    );
                }
                if (direct_sparse) {
                    direct_event_flags[slot] = static_cast<std::uint8_t>(
                        (agent_deltas >> kDeathDeltaShift) & 1u
                    );
                }
            }
        }
        constexpr unsigned int kFullWarpMask = 0xffffffffu;
        for (int offset = kWarpSize / 2; offset != 0; offset >>= 1) {
            packed_tick_deltas += __shfl_down_sync(
                kFullWarpMask, packed_tick_deltas, offset
            );
        }
        const int lane = static_cast<int>(threadIdx.x) & (kWarpSize - 1);
        const int warp = static_cast<int>(threadIdx.x) / kWarpSize;
        if (lane == 0) {
            warp_counter_deltas[warp] = packed_tick_deltas;
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            for (int counter = 0; counter < 4; ++counter) {
                std::uint64_t total = 0u;
                const int shift = counter * 8;
                for (int warp_index = 0; warp_index < kWarpsPerBlock; ++warp_index) {
                    total += (warp_counter_deltas[warp_index] >> shift) & 0xffu;
                }
                header[kPatchCounter + counter] += total;
            }
            std::uint64_t death_total = 0u;
            for (int warp_index = 0; warp_index < kWarpsPerBlock; ++warp_index) {
                death_total +=
                    (warp_counter_deltas[warp_index] >> kDeathDeltaShift) & 0xffu;
            }
            header[kDeathCount] += death_total;
            event_counts[blockIdx.x] += static_cast<std::int32_t>(death_total);
        }
        __syncthreads();
        if (direct_sparse) {
            append_direct_sparse_events_v3(
                row, direct_event_flags, temporal, 1u,
                world_direct_sparse_events, direct_sparse_capacity_rows,
                &direct_event_count, direct_sparse_validation + blockIdx.x,
                direct_warp_counts, direct_warp_bases,
                &direct_tile_base, &direct_tile_count,
                world_science_agent_metrics, world_science_event_metrics,
                world_science_agent_birth_provenance,
                world_science_event_birth_provenance
            );
            for (int slot = static_cast<int>(threadIdx.x); slot < kCapacity;
                 slot += kThreads) {
                direct_event_flags[slot] = 0u;
            }
            __syncthreads();
        }
        if ((capability_mask & kBirth) != 0u) {
            auto action_get = [=] __device__ (const int slot) {
                return static_cast<int>(actions[slot]);
            };
            for (int site = static_cast<int>(threadIdx.x); site < kSites;
                 site += kThreads) {
                winner_uids[site] = std::numeric_limits<unsigned long long>::max();
            }
            for (int slot = static_cast<int>(threadIdx.x); slot < kCapacity;
                 slot += kThreads) {
                mate_choices[slot] = choose_min_uid_mate(
                    row, slot, action_get, reproduction_energy_threshold
                );
                if (world_science_agent_metrics != nullptr &&
                    is_alive(alive, slot) && actions[slot] == 0u &&
                    energy[slot] >= reproduction_energy_threshold) {
                    std::int64_t* metrics = world_science_agent_metrics +
                        static_cast<std::int64_t>(slot) *
                            kScienceAgentMetricWidthV1;
                    ++metrics[kScienceReproductionEligible];
                    if (mate_choices[slot] != kEmpty) {
                        ++metrics[kScienceMateCandidate];
                    }
                }
                pair_sites[slot] = kEmpty;
            }
            __syncthreads();
            for (int leader = static_cast<int>(threadIdx.x); leader < kCapacity;
                 leader += kThreads) {
                const std::uint16_t mate = mate_choices[leader];
                if (mate == kEmpty || mate_choices[mate] != leader ||
                    ids[leader] >= ids[mate]) {
                    continue;
                }
                if (world_science_agent_metrics != nullptr) {
                    ++world_science_agent_metrics[
                        static_cast<std::int64_t>(leader) *
                            kScienceAgentMetricWidthV1 + kScienceBirthAttempt
                    ];
                    ++world_science_agent_metrics[
                        static_cast<std::int64_t>(mate) *
                            kScienceAgentMetricWidthV1 + kScienceBirthAttempt
                    ];
                }
                const std::uint16_t site = minimum_empty_birth_site(
                    row, leader, static_cast<int>(mate)
                );
                pair_sites[leader] = site;
                if (site != kEmpty) {
                    atomicMin(
                        winner_uids + site,
                        static_cast<unsigned long long>(ids[leader])
                    );
                }
            }
            __syncthreads();
            if (threadIdx.x == 0) {
                int pair_count = 0;
                int free_count = 0;
                for (int slot = 0; slot < kCapacity; ++slot) {
                    free_count += is_alive(alive, slot) ? 0 : 1;
                    if (pair_sites[slot] != kEmpty &&
                        winner_uids[pair_sites[slot]] == ids[slot]) {
                        ++pair_count;
                    }
                }
                accepted_birth_count = min(pair_count, free_count);
                birth_id_base = header[kNextAgentId];
            }
            __syncthreads();
            for (int leader = static_cast<int>(threadIdx.x); leader < kCapacity;
                 leader += kThreads) {
                if (pair_sites[leader] == kEmpty ||
                    winner_uids[pair_sites[leader]] != ids[leader]) {
                    continue;
                }
                int rank = 0;
                for (int other = 0; other < kCapacity; ++other) {
                    rank += pair_sites[other] != kEmpty &&
                        winner_uids[pair_sites[other]] == ids[other] &&
                        ids[other] < ids[leader] ? 1 : 0;
                }
                if (rank >= accepted_birth_count) {
                    continue;
                }
                int free_rank = 0;
                int child = -1;
                for (int slot = 0; slot < kCapacity; ++slot) {
                    if (is_alive(alive, slot)) {
                        continue;
                    }
                    if (free_rank == rank) {
                        child = slot;
                        break;
                    }
                    ++free_rank;
                }
                leader_slots_by_rank[rank] = static_cast<std::uint16_t>(leader);
                child_slots_by_rank[rank] = static_cast<std::uint16_t>(child);
                const int mate = static_cast<int>(mate_choices[leader]);
                if (world_science_agent_metrics != nullptr) {
                    ++world_science_agent_metrics[
                        static_cast<std::int64_t>(leader) *
                            kScienceAgentMetricWidthV1 +
                        kScienceSuccessfulParent
                    ];
                    ++world_science_agent_metrics[
                        static_cast<std::int64_t>(mate) *
                            kScienceAgentMetricWidthV1 +
                        kScienceSuccessfulParent
                    ];
                    ++world_science_agent_metrics[
                        static_cast<std::int64_t>(leader) *
                            kScienceAgentMetricWidthV1 +
                        kScienceBirthsAsParent
                    ];
                    ++world_science_agent_metrics[
                        static_cast<std::int64_t>(mate) *
                            kScienceAgentMetricWidthV1 +
                        kScienceBirthsAsParent
                    ];
                    world_science_agent_metrics[
                        static_cast<std::int64_t>(leader) *
                            kScienceAgentMetricWidthV1 +
                        kScienceEnergyDelta
                    ] -= reproduction_energy_cost;
                    world_science_agent_metrics[
                        static_cast<std::int64_t>(mate) *
                            kScienceAgentMetricWidthV1 +
                        kScienceEnergyDelta
                    ] -= reproduction_energy_cost;
                }
                energy[leader] = static_cast<std::int16_t>(
                    energy[leader] - reproduction_energy_cost
                );
                energy[mate] = static_cast<std::int16_t>(
                    energy[mate] - reproduction_energy_cost
                );
            }
        } else if (threadIdx.x == 0) {
            accepted_birth_count = 0;
        }
        __syncthreads();
        for (int rank_base = 0; rank_base < accepted_birth_count;
             rank_base += kWarpsPerBlock) {
            const int rank = rank_base + warp;
            if (rank < accepted_birth_count) {
                const int leader = static_cast<int>(leader_slots_by_rank[rank]);
                const int mate = static_cast<int>(mate_choices[leader]);
                commit_child_warp(
                    row, leader, mate,
                    static_cast<int>(child_slots_by_rank[rank]),
                    pair_sites[leader], rank,
                    birth_id_base + static_cast<std::uint64_t>(rank),
                    tick, recombination_tick_rng_key, mutation_tick_rng_key,
                    capability_mask, mutation_threshold, mutation_step,
                    world_science_agent_metrics,
                    world_science_agent_birth_provenance,
                    actions, mutation_counts_by_rank,
                    world_events, event_tick_base,
                    science_birth_child_ids,
                    science_birth_parent_a_ids,
                    science_birth_parent_b_ids,
                    science_birth_mutation_counts,
                    science_birth_child_slots,
                    science_birth_sites,
                    science_birth_base
                );
            }
            __syncthreads();
        }
        if (direct_sparse) {
            for (int rank = static_cast<int>(threadIdx.x);
                 rank < accepted_birth_count; rank += kThreads) {
                direct_event_flags[child_slots_by_rank[rank]] = 1u;
            }
            __syncthreads();
        }
        if (threadIdx.x == 0 && accepted_birth_count != 0) {
            std::uint64_t mutation_total = 0u;
            for (int rank = 0; rank < accepted_birth_count; ++rank) {
                mutation_total += mutation_counts_by_rank[rank];
            }
            header[kNextAgentId] = birth_id_base +
                static_cast<std::uint64_t>(accepted_birth_count);
            header[kBirthCapabilityCounter] += accepted_birth_count;
            header[kBirthCount] += accepted_birth_count;
            if ((capability_mask & kRecombination) != 0u) {
                const std::uint64_t recombination_total =
                    static_cast<std::uint64_t>(accepted_birth_count) *
                    static_cast<std::uint64_t>(kRecombinationBlocks);
                header[kRecombinationCapabilityCounter] += recombination_total;
                header[kRecombinationCount] += recombination_total;
            }
            header[kMutationCapabilityCounter] += mutation_total;
            header[kMutationCount] += mutation_total;
            event_counts[blockIdx.x] += accepted_birth_count;
            if (science_birth_counts != nullptr) {
                science_birth_counts[
                    static_cast<std::int64_t>(blockIdx.x) * temporal_k + temporal
                ] = accepted_birth_count;
            }
        }
        __syncthreads();
        if (direct_sparse) {
            append_direct_sparse_events_v3(
                row, direct_event_flags, temporal, 2u,
                world_direct_sparse_events, direct_sparse_capacity_rows,
                &direct_event_count, direct_sparse_validation + blockIdx.x,
                direct_warp_counts, direct_warp_bases,
                &direct_tile_base, &direct_tile_count,
                world_science_agent_metrics, world_science_event_metrics,
                world_science_agent_birth_provenance,
                world_science_event_birth_provenance
            );
        }
        if (header[kStatus] != 0u) {
            return;
        }
        finalize_tick_parallel(row, warp_summary_counts);
        __syncthreads();
    }
    if (direct_sparse && threadIdx.x == 0 &&
        direct_event_count != event_counts[blockIdx.x]) {
        atomicOr(direct_sparse_validation + blockIdx.x, kSparseCountMismatch);
    }
}

__global__ void force_test_conditions_kernel(
    std::uint8_t* state, const std::int64_t row_stride
) {
    if (threadIdx.x != 0) {
        return;
    }
    std::uint8_t* row = state + static_cast<std::int64_t>(blockIdx.x) * row_stride;
    auto* header = field<std::uint64_t>(row, kHeaderOffset);
    auto* occupancy = field<std::uint16_t>(row, kOccupancyOffset);
    auto* due = field<std::uint64_t>(row, kWdlDueOffset);
    auto* active = field<std::uint8_t>(row, kWdlActiveOffset);
    auto* jungle = field<std::uint8_t>(row, kJungleOffset);
    auto* pending = field<std::uint16_t>(row, kEedPendingOffset);
    auto* position = field<std::uint16_t>(row, kPositionOffset);
    auto* energy = field<std::int16_t>(row, kEnergyOffset);
    auto* health = field<std::int16_t>(row, kHealthOffset);
    auto* alive = field<std::uint32_t>(row, kAliveWordsOffset);
    auto* ids = field<std::uint64_t>(row, kAgentIdOffset);
    auto* policy = field<std::int8_t>(row, kPolicyAllelesOffset);
    auto* traits = field<std::int16_t>(row, kTraitAllelesOffset);
    for (int slot = 0; slot < kInitialPopulation; ++slot) {
        energy[slot] = 80;
        for (int allele = 0; allele < 2; ++allele) {
            for (int locus = 0; locus < kPolicyLoci; ++locus) {
                policy[policy_allele_index(allele, locus, slot)] = 0;
            }
            policy[policy_allele_index(allele, 0, slot)] = 100;
        }
        express_genome(row, slot);
    }
    health[0] = 1;
    due[position[0]] = header[kNextAbsoluteTick];
    active[position[0]] = 4u;
    jungle[position[0]] = 63u;
    pending[position[5]] = 8u;
    for (int pair_index = 0; pair_index < 10; ++pair_index) {
        const int first = pair_index < 7
            ? 2 + pair_index * 4
            : (pair_index == 7 ? 33 : 242 + (pair_index - 8) * 4);
        const int second = pair_index == 7 ? 65 : first + 1;
        const std::uint64_t first_uid = ids[first];
        ids[first] = ids[second];
        ids[second] = first_uid;
        for (int pair_member = 0; pair_member < 2; ++pair_member) {
            const int slot = pair_member == 0 ? first : second;
            set_alive(alive, slot);
            occupancy[position[slot]] = static_cast<std::uint16_t>(slot);
            energy[slot] = 300;
            health[slot] = 120;
            for (int allele = 0; allele < 2; ++allele) {
                for (int locus = 0; locus < kPolicyLoci; ++locus) {
                    policy[policy_allele_index(allele, locus, slot)] = 0;
                }
                policy[policy_allele_index(allele, 0, slot)] = 100;
                traits[trait_allele_index(allele, 0, slot)] =
                    static_cast<std::int16_t>(pair_member == 0 ? -16 : 16);
            }
            express_genome(row, slot);
        }
    }
}

__global__ void force_full_capacity_benchmark_fixture_kernel(
    std::uint8_t* state, const std::int64_t row_stride
) {
    if (threadIdx.x != 0) {
        return;
    }
    std::uint8_t* row = state + static_cast<std::int64_t>(blockIdx.x) * row_stride;
    auto* header = field<std::uint64_t>(row, kHeaderOffset);
    auto* occupancy = field<std::uint16_t>(row, kOccupancyOffset);
    auto* position = field<std::uint16_t>(row, kPositionOffset);
    auto* energy = field<std::int16_t>(row, kEnergyOffset);
    auto* health = field<std::int16_t>(row, kHealthOffset);
    auto* age = field<std::uint16_t>(row, kAgeOffset);
    auto* absolute_birth_tick = field<std::uint64_t>(
        row, kAbsoluteBirthTickOffset
    );
    auto* alive = field<std::uint32_t>(row, kAliveWordsOffset);
    auto* ids = field<std::uint64_t>(row, kAgentIdOffset);
    auto* parent_a = field<std::uint64_t>(row, kParentAIdOffset);
    auto* parent_b = field<std::uint64_t>(row, kParentBIdOffset);
    auto* policy = field<std::int8_t>(row, kPolicyAllelesOffset);
    auto* traits = field<std::int16_t>(row, kTraitAllelesOffset);
    bool fresh = header[kStatus] == 0u && header[kTicksCompleted] == 0u &&
        header[kActiveCount] == static_cast<std::uint64_t>(kInitialPopulation) &&
        header[kNextAgentId] == static_cast<std::uint64_t>(kInitialPopulation + 1) &&
        health[0] == 2 && energy[0] == 160 && ids[0] == 1u && ids[2] == 3u;
    for (int index = kPatchCounter; index <= kRecombinationCapabilityCounter;
         ++index) {
        fresh = fresh && header[index] == 0u;
    }
    for (int index = kBirthCount; index <= kEedPendingCount; ++index) {
        fresh = fresh && header[index] == 0u;
    }
    for (int slot = kInitialPopulation; slot < kCapacity; ++slot) {
        const int site = slot + kCapacity;
        fresh = fresh && !is_alive(alive, slot) && occupancy[site] == kEmpty &&
            ids[slot] == 0u && parent_a[slot] == 0u && parent_b[slot] == 0u;
    }
    for (int slot = 0; slot < kInitialPopulation; ++slot) {
        fresh = fresh && is_alive(alive, slot) && ids[slot] != 0u &&
            position[slot] < kSites && occupancy[position[slot]] == slot;
    }
    if (!fresh) {
        header[kStatus] = 7u;
        return;
    }
    const std::uint64_t next_tick = header[kNextAbsoluteTick];
    const uint2 fixture_rng_key = rng_tick_key(header, next_tick);
    for (int slot = kInitialPopulation; slot < kCapacity; ++slot) {
        const int site = slot + kCapacity;
        const std::uint64_t biological_uid = static_cast<std::uint64_t>(slot + 1);
        position[slot] = static_cast<std::uint16_t>(site);
        occupancy[site] = static_cast<std::uint16_t>(slot);
        energy[slot] = static_cast<std::int16_t>(160 + slot % 32);
        health[slot] = 100;
        age[slot] = 0u;
        absolute_birth_tick[slot] = next_tick;
        ids[slot] = biological_uid;
        parent_a[slot] = 0u;
        parent_b[slot] = 0u;
        const int chosen_action = slot % kActions;
        for (int allele = 0; allele < 2; ++allele) {
            for (int locus = 0; locus < kPolicyLoci; ++locus) {
                const std::uint32_t draw = counter_rng32(
                    fixture_rng_key, kDomainInitialPolicy, biological_uid,
                    static_cast<std::uint64_t>(allele),
                    static_cast<std::uint64_t>(locus)
                );
                int value = static_cast<int>(draw % 5u) - 2;
                if (locus / kActions == 0 && locus % kActions == chosen_action) {
                    value = 100;
                }
                policy[policy_allele_index(allele, locus, slot)] =
                    static_cast<std::int8_t>(value);
            }
            for (int locus = 0; locus < kTraitLoci; ++locus) {
                const std::uint32_t draw = counter_rng32(
                    fixture_rng_key, kDomainInitialTrait, biological_uid,
                    static_cast<std::uint64_t>(allele),
                    static_cast<std::uint64_t>(locus)
                );
                traits[trait_allele_index(allele, locus, slot)] =
                    static_cast<std::int16_t>(
                        locus == 0 && (biological_uid & 1u) != 0u
                            ? -(8 + static_cast<int>(draw % 25u))
                            : 8 + static_cast<int>(draw % 25u)
                    );
            }
        }
        express_genome(row, slot);
        set_alive(alive, slot);
    }
    header[kNextAgentId] = static_cast<std::uint64_t>(kCapacity + 1);
    header[kActiveCount] = static_cast<std::uint64_t>(kCapacity);
}

__global__ void rebuild_derived_caches_kernel(
    std::uint8_t* state, const std::int64_t row_stride
) {
    std::uint8_t* row = state + static_cast<std::int64_t>(blockIdx.x) * row_stride;
    for (int slot = static_cast<int>(threadIdx.x); slot < kCapacity; slot += kThreads) {
        express_genome(row, slot);
    }
}

__global__ __launch_bounds__(kThreads) void compact_events_v3_kernel(
    const std::uint8_t* state,
    const std::int64_t state_stride,
    const LifecycleEventRow* dense_events,
    const std::int64_t dense_event_stride_rows,
    const std::int32_t* event_counts,
    const std::int64_t* world_offsets,
    SparseLifecycleEventV3* packed_events,
    const std::int64_t packed_event_count,
    std::int32_t* validation,
    const std::uint64_t start_tick,
    const int tick_count
) {
    __shared__ std::int64_t segment_begin;
    __shared__ std::int64_t segment_end;
    __shared__ std::int64_t tile_base;
    __shared__ int tile_count;
    __shared__ int warp_counts[kWarpsPerBlock];
    __shared__ int warp_bases[kWarpsPerBlock];

    const int world = static_cast<int>(blockIdx.x);
    const int lane = static_cast<int>(threadIdx.x) & (kWarpSize - 1);
    const int warp = static_cast<int>(threadIdx.x) / kWarpSize;
    const std::uint8_t* state_row =
        state + static_cast<std::int64_t>(world) * state_stride;
    const auto* header = field<std::uint64_t>(state_row, kHeaderOffset);
    const LifecycleEventRow* world_events = dense_events +
        static_cast<std::int64_t>(world) * dense_event_stride_rows;

    if (threadIdx.x == 0) {
        validation[world] = 0;
        const std::int64_t begin = world_offsets[world];
        const std::int64_t end = world_offsets[world + 1];
        const std::int64_t declared_count =
            static_cast<std::int64_t>(event_counts[world]);
        const bool valid_offsets = begin >= 0 && end >= begin &&
            end <= packed_event_count && end - begin == declared_count;
        if (!valid_offsets) {
            validation[world] |= kSparseInvalidOffsets;
            segment_begin = 0;
            segment_end = 0;
        } else {
            segment_begin = begin;
            segment_end = end;
        }
        tile_base = segment_begin;
    }
    __syncthreads();

    const std::uint64_t final_next_agent_id = header[kNextAgentId];
    if (final_next_agent_id == 0u ||
        final_next_agent_id >
            static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) {
        if (threadIdx.x == 0) {
            atomicOr(validation + world, kSparseInvalidAgentIdentity);
        }
    }

    for (std::int64_t tile = 0; tile < dense_event_stride_rows; tile += kThreads) {
        const std::int64_t row_index = tile + static_cast<std::int64_t>(threadIdx.x);
        bool committed = false;
        SparseLifecycleEventV3 packed{};
        std::int32_t errors = 0;

        if (row_index < dense_event_stride_rows) {
            const LifecycleEventRow& event = world_events[row_index];
            const auto* words = reinterpret_cast<const std::uint64_t*>(&event);
            if (event.status != 0u && event.status != 1u) {
                errors |= kSparseInvalidStatus;
            }
            committed = event.status == 1u;
            if (!committed) {
                if ((words[0] | words[1] | words[2] | words[3] | words[4] |
                     words[5]) != 0u) {
                    errors |= kSparseNonzeroUnusedRow;
                }
            } else {
                const std::int64_t tick_offset = row_index / kEventRowsPerTick;
                const int within_tick = static_cast<int>(row_index % kEventRowsPerTick);
                const int subject_slot = within_tick % kCapacity;
                const std::uint16_t expected_kind =
                    within_tick < kCapacity ? 1u : 2u;
                const std::uint64_t expected_event_id =
                    (static_cast<std::uint64_t>(expected_kind) << 48) |
                    static_cast<std::uint64_t>(subject_slot);
                if (tick_offset < 0 || tick_offset >= tick_count ||
                    event.absolute_tick !=
                        start_tick + static_cast<std::uint64_t>(tick_offset)) {
                    errors |= kSparseInvalidTick;
                }
                if (event.event_kind != expected_kind ||
                    event.event_id != expected_event_id) {
                    errors |= kSparseInvalidKindOrId;
                }
                if (event.agent_id == 0u ||
                    event.agent_id >= final_next_agent_id) {
                    errors |= kSparseInvalidAgentIdentity;
                }
                if (event.parent_a_id >= final_next_agent_id ||
                    event.parent_b_id >= final_next_agent_id) {
                    errors |= kSparseInvalidParentIdentity;
                }
                if (expected_kind == 2u &&
                    (event.parent_a_id == 0u || event.parent_b_id == 0u ||
                     event.parent_a_id == event.parent_b_id)) {
                    errors |= kSparseInvalidBirthParents;
                }
                if (event.site_index >= kSites || event.reserved != 0u) {
                    errors |= kSparseInvalidSiteOrReserved;
                }
                packed.agent_id = event.agent_id;
                packed.parent_a_id = event.parent_a_id;
                packed.parent_b_id = event.parent_b_id;
                packed.packed_key_and_reserved =
                    static_cast<std::uint64_t>(tick_offset) |
                    (static_cast<std::uint64_t>(expected_kind) << 6) |
                    (static_cast<std::uint64_t>(subject_slot) << 8) |
                    (static_cast<std::uint64_t>(event.site_index) << 17);
            }
        }
        if (errors != 0) {
            atomicOr(validation + world, errors);
        }

        const unsigned int committed_mask = __ballot_sync(0xffffffffu, committed);
        const unsigned int lower_lane_mask =
            lane == 0 ? 0u : ((1u << lane) - 1u);
        const int rank_in_warp = __popc(committed_mask & lower_lane_mask);
        if (lane == 0) {
            warp_counts[warp] = __popc(committed_mask);
        }
        __syncthreads();
        if (threadIdx.x < kWarpsPerBlock) {
            int prefix = 0;
            for (int prior = 0; prior < static_cast<int>(threadIdx.x); ++prior) {
                prefix += warp_counts[prior];
            }
            warp_bases[threadIdx.x] = prefix;
            if (threadIdx.x == kWarpsPerBlock - 1) {
                tile_count = prefix + warp_counts[threadIdx.x];
            }
        }
        __syncthreads();
        if (committed) {
            const std::int64_t output_index = tile_base +
                static_cast<std::int64_t>(warp_bases[warp] + rank_in_warp);
            if (output_index >= segment_begin && output_index < segment_end &&
                output_index < packed_event_count) {
                packed_events[output_index] = packed;
            } else {
                atomicOr(validation + world, kSparseOutputOverflow);
            }
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            tile_base += static_cast<std::int64_t>(tile_count);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0 &&
        (tile_base != segment_end ||
         tile_base - segment_begin != static_cast<std::int64_t>(event_counts[world]))) {
        atomicOr(validation + world, kSparseCountMismatch);
    }
}

__global__ __launch_bounds__(kThreads) void pack_direct_sparse_events_v3_kernel(
    const std::uint8_t* state,
    const std::int64_t state_stride,
    const SparseLifecycleEventV3* direct_events,
    const std::int64_t direct_event_stride_rows,
    const int direct_event_capacity_rows,
    const std::int32_t* event_counts,
    const std::int32_t* advance_validation,
    const std::int64_t* world_offsets,
    SparseLifecycleEventV3* packed_events,
    const std::int64_t packed_event_count,
    const std::int64_t* science_event_metrics,
    const std::int64_t science_event_metric_stride,
    std::int64_t* packed_science_event_metrics,
    const std::uint8_t* science_event_birth_provenance,
    const std::int64_t science_event_birth_provenance_stride,
    std::uint8_t* packed_science_event_birth_provenance,
    std::int32_t* validation,
    const int tick_count
) {
    __shared__ std::int64_t segment_begin;
    __shared__ int source_count;
    const int world = static_cast<int>(blockIdx.x);
    const std::uint8_t* state_row =
        state + static_cast<std::int64_t>(world) * state_stride;
    const auto* header = field<std::uint64_t>(state_row, kHeaderOffset);
    const SparseLifecycleEventV3* world_events = direct_events +
        static_cast<std::int64_t>(world) * direct_event_stride_rows;
    const std::int64_t* world_science_event_metrics =
        science_event_metrics == nullptr ? nullptr : science_event_metrics +
            static_cast<std::int64_t>(world) * science_event_metric_stride;
    const std::uint8_t* world_science_event_birth_provenance =
        science_event_birth_provenance == nullptr
        ? nullptr
        : science_event_birth_provenance +
            static_cast<std::int64_t>(world) *
                science_event_birth_provenance_stride;
    if (threadIdx.x == 0) {
        validation[world] = advance_validation[world];
        const std::int64_t begin = world_offsets[world];
        const std::int64_t end = world_offsets[world + 1];
        const std::int64_t declared_count =
            static_cast<std::int64_t>(event_counts[world]);
        if (declared_count < 0 || declared_count > direct_event_capacity_rows ||
            begin < 0 || end < begin || end > packed_event_count ||
            end - begin != declared_count ||
            (world == 0 && begin != 0) ||
            (world == static_cast<int>(gridDim.x) - 1 &&
             end != packed_event_count) ||
            direct_event_stride_rows < direct_event_capacity_rows) {
            validation[world] |= kSparseInvalidOffsets;
            segment_begin = 0;
            source_count = 0;
        } else {
            segment_begin = begin;
            source_count = static_cast<int>(declared_count);
        }
    }
    __syncthreads();

    const std::uint64_t final_next_agent_id = header[kNextAgentId];
    if (final_next_agent_id == 0u ||
        final_next_agent_id >
            static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) {
        if (threadIdx.x == 0) {
            atomicOr(validation + world, kSparseInvalidAgentIdentity);
        }
    }
    for (int source_index = static_cast<int>(threadIdx.x);
         source_index < source_count; source_index += kThreads) {
        const SparseLifecycleEventV3 event = world_events[source_index];
        const std::uint64_t key = event.packed_key_and_reserved;
        const int tick_offset = static_cast<int>(key & 0x3fu);
        const int event_kind = static_cast<int>((key >> 6) & 0x3u);
        const int subject_slot = static_cast<int>((key >> 8) & 0x1ffu);
        const int site = static_cast<int>((key >> 17) & 0x3ffu);
        std::int32_t errors = 0;
        if ((key >> 27) != 0u || site >= kSites) {
            errors |= kSparseInvalidSiteOrReserved;
        }
        if (tick_offset >= tick_count) {
            errors |= kSparseInvalidTick;
        }
        if ((event_kind != 1 && event_kind != 2) ||
            subject_slot >= kCapacity) {
            errors |= kSparseInvalidKindOrId;
        }
        if (event.agent_id == 0u || event.agent_id >= final_next_agent_id) {
            errors |= kSparseInvalidAgentIdentity;
        }
        if (event.parent_a_id >= final_next_agent_id ||
            event.parent_b_id >= final_next_agent_id) {
            errors |= kSparseInvalidParentIdentity;
        }
        if (event_kind == 2 &&
            (event.parent_a_id == 0u || event.parent_b_id == 0u ||
             event.parent_a_id == event.parent_b_id)) {
            errors |= kSparseInvalidBirthParents;
        }
        if (source_index != 0) {
            const std::uint64_t prior_key =
                world_events[source_index - 1].packed_key_and_reserved;
            const int prior_tick = static_cast<int>(prior_key & 0x3fu);
            const int prior_kind = static_cast<int>((prior_key >> 6) & 0x3u);
            const int prior_slot = static_cast<int>((prior_key >> 8) & 0x1ffu);
            const bool strictly_canonical =
                prior_tick < tick_offset ||
                (prior_tick == tick_offset && prior_kind < event_kind) ||
                (prior_tick == tick_offset && prior_kind == event_kind &&
                 prior_slot < subject_slot);
            if (!strictly_canonical) {
                errors |= kSparseDirectOrder;
            }
        }
        const std::int64_t output_index =
            segment_begin + static_cast<std::int64_t>(source_index);
        if (output_index < 0 || output_index >= packed_event_count) {
            errors |= kSparseOutputOverflow;
        } else {
            packed_events[output_index] = event;
            if (packed_science_event_metrics != nullptr) {
                for (int metric = 0; metric < kScienceAgentMetricWidthV1;
                     ++metric) {
                    packed_science_event_metrics[
                        output_index * kScienceAgentMetricWidthV1 + metric
                    ] = world_science_event_metrics[
                        static_cast<std::int64_t>(source_index) *
                            kScienceAgentMetricWidthV1 + metric
                    ];
                }
            }
            if (packed_science_event_birth_provenance != nullptr) {
                for (int field = 0; field < kScienceBirthProvenanceWidthV1;
                     ++field) {
                    packed_science_event_birth_provenance[
                        output_index * kScienceBirthProvenanceWidthV1 + field
                    ] = world_science_event_birth_provenance[
                        static_cast<std::int64_t>(source_index) *
                            kScienceBirthProvenanceWidthV1 + field
                    ];
                }
            }
        }
        if (errors != 0) {
            atomicOr(validation + world, errors);
        }
    }
}

void check_state(const torch::Tensor& state) {
    TORCH_CHECK(state.is_cuda(), "state must be CUDA");
    TORCH_CHECK(state.scalar_type() == torch::kUInt8, "state must be uint8");
    TORCH_CHECK(state.is_contiguous(), "state must be contiguous");
    TORCH_CHECK(state.dim() == 2 && state.size(0) >= 1 && state.size(1) == kStateBytes,
        "state must be uint8[B,265472]");
}

void check_science_world_controls_v1(
    const torch::Tensor& state,
    const torch::Tensor& science_world_controls
) {
    TORCH_CHECK(
        science_world_controls.is_cuda() &&
        science_world_controls.scalar_type() == torch::kInt64 &&
        science_world_controls.is_contiguous() &&
        science_world_controls.dim() == 2 &&
        science_world_controls.size(0) == state.size(0) &&
        science_world_controls.size(1) == kScienceWorldControlWidthV1,
        "science world controls must be exactly int64[B,22] on CUDA"
    );
    TORCH_CHECK(
        science_world_controls.device() == state.device(),
        "science world controls device mismatch"
    );
}

void check_compact_science_metrics_v1(
    const torch::Tensor& state,
    const torch::Tensor& science_agent_metrics,
    const torch::Tensor& science_event_metrics,
    const std::int64_t event_capacity_rows
) {
    TORCH_CHECK(
        science_agent_metrics.is_cuda() &&
        science_agent_metrics.scalar_type() == torch::kInt64 &&
        science_agent_metrics.is_contiguous() &&
        science_agent_metrics.dim() == 3 &&
        science_agent_metrics.size(0) == state.size(0) &&
        science_agent_metrics.size(1) == kCapacity &&
        science_agent_metrics.size(2) == kScienceAgentMetricWidthV1,
        "science agent metrics must be exactly int64[B,512,21] on CUDA"
    );
    TORCH_CHECK(
        science_event_metrics.is_cuda() &&
        science_event_metrics.scalar_type() == torch::kInt64 &&
        science_event_metrics.is_contiguous() &&
        science_event_metrics.dim() == 3 &&
        science_event_metrics.size(0) == state.size(0) &&
        science_event_metrics.size(1) == event_capacity_rows &&
        science_event_metrics.size(2) == kScienceAgentMetricWidthV1,
        "science event metrics must be exactly int64[B,capacity,21] on CUDA"
    );
    TORCH_CHECK(
        science_agent_metrics.device() == state.device() &&
        science_event_metrics.device() == state.device(),
        "compact science metrics device mismatch"
    );
}

void check_compact_science_birth_provenance_v1(
    const torch::Tensor& state,
    const torch::Tensor& science_agent_birth_provenance,
    const torch::Tensor& science_event_birth_provenance,
    const std::int64_t event_capacity_rows
) {
    TORCH_CHECK(
        science_agent_birth_provenance.is_cuda() &&
        science_agent_birth_provenance.scalar_type() == torch::kUInt8 &&
        science_agent_birth_provenance.is_contiguous() &&
        science_agent_birth_provenance.dim() == 3 &&
        science_agent_birth_provenance.size(0) == state.size(0) &&
        science_agent_birth_provenance.size(1) == kCapacity &&
        science_agent_birth_provenance.size(2) ==
            kScienceBirthProvenanceWidthV1,
        "science agent birth provenance must be exactly uint8[B,512,132] on CUDA"
    );
    TORCH_CHECK(
        science_event_birth_provenance.is_cuda() &&
        science_event_birth_provenance.scalar_type() == torch::kUInt8 &&
        science_event_birth_provenance.is_contiguous() &&
        science_event_birth_provenance.dim() == 3 &&
        science_event_birth_provenance.size(0) == state.size(0) &&
        science_event_birth_provenance.size(1) == event_capacity_rows &&
        science_event_birth_provenance.size(2) ==
            kScienceBirthProvenanceWidthV1,
        "science event birth provenance must be exactly uint8[B,capacity,132] on CUDA"
    );
    TORCH_CHECK(
        science_agent_birth_provenance.device() == state.device() &&
        science_event_birth_provenance.device() == state.device(),
        "compact science birth provenance device mismatch"
    );
}

bool tensors_overlap(const torch::Tensor& left, const torch::Tensor& right) {
    const std::uintptr_t left_begin = reinterpret_cast<std::uintptr_t>(
        left.data_ptr()
    );
    const std::uintptr_t right_begin = reinterpret_cast<std::uintptr_t>(
        right.data_ptr()
    );
    const std::uint64_t left_bytes = static_cast<std::uint64_t>(left.numel()) *
        static_cast<std::uint64_t>(left.element_size());
    const std::uint64_t right_bytes = static_cast<std::uint64_t>(right.numel()) *
        static_cast<std::uint64_t>(right.element_size());
    TORCH_CHECK(left_bytes <= std::numeric_limits<std::uintptr_t>::max() - left_begin &&
        right_bytes <= std::numeric_limits<std::uintptr_t>::max() - right_begin,
        "tensor storage range exceeds uintptr");
    return left_begin < right_begin + right_bytes &&
        right_begin < left_begin + left_bytes;
}

void check_events(
    const torch::Tensor& state,
    const torch::Tensor& events,
    const torch::Tensor& event_counts,
    const std::int64_t temporal_k,
    const std::int64_t event_capacity_k
) {
    TORCH_CHECK(event_capacity_k >= 1 && event_capacity_k <= 64,
        "event_capacity_k must be in [1,64]");
    TORCH_CHECK(temporal_k <= event_capacity_k,
        "dispatch temporal_k exceeds event capacity");
    TORCH_CHECK(events.is_cuda() && events.scalar_type() == torch::kUInt8 &&
        events.is_contiguous() && events.dim() == 3 &&
        events.size(0) == state.size(0) &&
        events.size(1) == kEventRowsPerTick * event_capacity_k &&
        events.size(2) == kEventRowBytes, "events must be exactly uint8[B,2*C*K,48] on CUDA");
    TORCH_CHECK(event_counts.is_cuda() && event_counts.scalar_type() == torch::kInt32 &&
        event_counts.is_contiguous() && event_counts.dim() == 1 &&
        event_counts.size(0) == state.size(0), "event_counts must be int32[B] on CUDA");
    TORCH_CHECK(events.device() == state.device() && event_counts.device() == state.device(),
        "event staging device mismatch");
}

void check_direct_sparse_events_v3(
    const torch::Tensor& state,
    const torch::Tensor& direct_events,
    const torch::Tensor& event_counts,
    const torch::Tensor& validation,
    const std::int64_t event_capacity_rows
) {
    TORCH_CHECK(event_capacity_rows >= 1 &&
        event_capacity_rows <= kEventRowsPerTick * 64,
        "direct sparse event capacity must be in [1,2*C*64]");
    TORCH_CHECK(direct_events.is_cuda() &&
        direct_events.scalar_type() == torch::kUInt8 &&
        direct_events.is_contiguous() && direct_events.dim() == 3 &&
        direct_events.size(0) == state.size(0) &&
        direct_events.size(1) == event_capacity_rows &&
        direct_events.size(2) == kSparseEventRowBytesV3,
        "direct sparse events must be exactly uint8[B,capacity,32] on CUDA");
    TORCH_CHECK(event_counts.is_cuda() &&
        event_counts.scalar_type() == torch::kInt32 &&
        event_counts.is_contiguous() && event_counts.dim() == 1 &&
        event_counts.size(0) == state.size(0),
        "direct sparse event_counts must be int32[B] on CUDA");
    TORCH_CHECK(validation.is_cuda() &&
        validation.scalar_type() == torch::kInt32 &&
        validation.is_contiguous() && validation.dim() == 1 &&
        validation.size(0) == state.size(0),
        "direct sparse validation must be int32[B] on CUDA");
    TORCH_CHECK(direct_events.device() == state.device() &&
        event_counts.device() == state.device() &&
        validation.device() == state.device(),
        "direct sparse event staging device mismatch");
    TORCH_CHECK(!tensors_overlap(state, direct_events) &&
        !tensors_overlap(state, event_counts) &&
        !tensors_overlap(state, validation) &&
        !tensors_overlap(direct_events, event_counts) &&
        !tensors_overlap(direct_events, validation) &&
        !tensors_overlap(event_counts, validation),
        "direct sparse owner tensors must not overlap");
}

void check_science_capture_v1(
    const torch::Tensor& state,
    const torch::Tensor& action_ids,
    const torch::Tensor& action_values,
    const torch::Tensor& birth_child_ids,
    const torch::Tensor& birth_parent_a_ids,
    const torch::Tensor& birth_parent_b_ids,
    const torch::Tensor& birth_mutation_counts,
    const torch::Tensor& birth_child_slots,
    const torch::Tensor& birth_sites,
    const torch::Tensor& birth_counts,
    const std::int64_t temporal_k
);

void initialize(
    torch::Tensor state,
    torch::Tensor world_uids,
    const std::int64_t seed,
    const std::int64_t next_tick
) {
    check_state(state);
    TORCH_CHECK(world_uids.is_cuda() && world_uids.scalar_type() == torch::kInt64 &&
        world_uids.is_contiguous() && world_uids.dim() == 1 &&
        world_uids.size(0) == state.size(0), "world_uids must be int64[B] on CUDA");
    TORCH_CHECK(world_uids.device() == state.device(), "world_uids device mismatch");
    TORCH_CHECK(seed >= 0, "seed must be nonnegative signed int64");
    TORCH_CHECK(next_tick >= 0, "next_absolute_tick must be nonnegative signed int64");
    const c10::cuda::CUDAGuard guard(state.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    initialize_kernel<<<state.size(0), kThreads, 0, stream>>>(
        state.data_ptr<std::uint8_t>(), state.stride(0),
        world_uids.data_ptr<std::int64_t>(),
        static_cast<std::uint64_t>(seed), static_cast<std::uint64_t>(next_tick),
        nullptr, 0
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void initialize_science_worlds_v1(
    torch::Tensor state,
    torch::Tensor world_uids,
    torch::Tensor science_world_controls,
    const std::int64_t next_tick
) {
    check_state(state);
    TORCH_CHECK(world_uids.is_cuda() && world_uids.scalar_type() == torch::kInt64 &&
        world_uids.is_contiguous() && world_uids.dim() == 1 &&
        world_uids.size(0) == state.size(0),
        "world_uids must be int64[B] on CUDA");
    TORCH_CHECK(world_uids.device() == state.device(), "world_uids device mismatch");
    check_science_world_controls_v1(state, science_world_controls);
    TORCH_CHECK(next_tick >= 0, "next_absolute_tick must be nonnegative signed int64");
    const c10::cuda::CUDAGuard guard(state.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    initialize_kernel<<<state.size(0), kThreads, 0, stream>>>(
        state.data_ptr<std::uint8_t>(), state.stride(0),
        world_uids.data_ptr<std::int64_t>(), 0u,
        static_cast<std::uint64_t>(next_tick),
        science_world_controls.data_ptr<std::int64_t>(),
        science_world_controls.stride(0)
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void advance_reference(
    torch::Tensor state,
    torch::Tensor scratch,
    torch::Tensor events,
    torch::Tensor event_counts,
    const std::int64_t temporal_k,
    const std::int64_t event_capacity_k,
    const std::int64_t capability_mask
) {
    check_state(state);
    TORCH_CHECK(scratch.is_cuda() && scratch.scalar_type() == torch::kUInt8 &&
        scratch.is_contiguous() && scratch.dim() == 2 &&
        scratch.size(0) == state.size(0) && scratch.size(1) == kReferenceScratchBytes,
        "reference scratch must be uint8[B,8192] on CUDA");
    TORCH_CHECK(scratch.device() == state.device(), "reference scratch device mismatch");
    TORCH_CHECK(temporal_k >= 1 && temporal_k <= 64, "temporal_k must be in [1,64]");
    check_events(state, events, event_counts, temporal_k, event_capacity_k);
    TORCH_CHECK(capability_mask >= 0 && capability_mask <= kFullCapabilityMask,
        "capability mask is invalid");
    const c10::cuda::CUDAGuard guard(state.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    advance_reference_kernel<<<state.size(0), 1, 0, stream>>>(
        state.data_ptr<std::uint8_t>(), state.stride(0),
        scratch.data_ptr<std::uint8_t>(), scratch.stride(0),
        reinterpret_cast<LifecycleEventRow*>(events.data_ptr<std::uint8_t>()),
        events.stride(0) / kEventRowBytes,
        event_counts.data_ptr<std::int32_t>(),
        static_cast<int>(temporal_k), static_cast<std::uint64_t>(capability_mask)
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void advance(
    torch::Tensor state,
    torch::Tensor events,
    torch::Tensor event_counts,
    const std::int64_t temporal_k,
    const std::int64_t event_capacity_k,
    const std::int64_t capability_mask
) {
    check_state(state);
    TORCH_CHECK(temporal_k >= 1 && temporal_k <= 64, "temporal_k must be in [1,64]");
    check_events(state, events, event_counts, temporal_k, event_capacity_k);
    TORCH_CHECK(capability_mask >= 0 && capability_mask <= kFullCapabilityMask,
        "capability mask is invalid");
    const c10::cuda::CUDAGuard guard(state.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    advance_optimized_kernel<<<state.size(0), kThreads, 0, stream>>>(
        state.data_ptr<std::uint8_t>(), state.stride(0),
        reinterpret_cast<LifecycleEventRow*>(events.data_ptr<std::uint8_t>()),
        events.stride(0) / kEventRowBytes,
        event_counts.data_ptr<std::int32_t>(),
        nullptr, 0, 0, nullptr,
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
        nullptr, 0,
        nullptr, 0, nullptr, 0,
        nullptr, 0, nullptr, 0,
        static_cast<int>(temporal_k), static_cast<std::uint64_t>(capability_mask)
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void advance_direct_sparse_v3(
    torch::Tensor state,
    torch::Tensor direct_events,
    torch::Tensor event_counts,
    torch::Tensor validation,
    const std::int64_t temporal_k,
    const std::int64_t event_capacity_rows,
    const std::int64_t capability_mask
) {
    check_state(state);
    TORCH_CHECK(temporal_k >= 1 && temporal_k <= 64,
        "temporal_k must be in [1,64]");
    check_direct_sparse_events_v3(
        state, direct_events, event_counts, validation, event_capacity_rows
    );
    TORCH_CHECK(capability_mask >= 0 && capability_mask <= kFullCapabilityMask,
        "capability mask is invalid");
    const c10::cuda::CUDAGuard guard(state.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    advance_optimized_kernel<<<state.size(0), kThreads, 0, stream>>>(
        state.data_ptr<std::uint8_t>(), state.stride(0), nullptr, 0,
        event_counts.data_ptr<std::int32_t>(),
        reinterpret_cast<SparseLifecycleEventV3*>(
            direct_events.data_ptr<std::uint8_t>()
        ),
        direct_events.stride(0) / kSparseEventRowBytesV3,
        static_cast<int>(event_capacity_rows),
        validation.data_ptr<std::int32_t>(),
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
        nullptr, 0,
        nullptr, 0, nullptr, 0,
        nullptr, 0, nullptr, 0,
        static_cast<int>(temporal_k),
        static_cast<std::uint64_t>(capability_mask)
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void advance_direct_sparse_worlds_v1(
    torch::Tensor state,
    torch::Tensor direct_events,
    torch::Tensor event_counts,
    torch::Tensor validation,
    torch::Tensor science_world_controls,
    const std::int64_t temporal_k,
    const std::int64_t event_capacity_rows
) {
    check_state(state);
    TORCH_CHECK(temporal_k >= 1 && temporal_k <= 64,
        "temporal_k must be in [1,64]");
    check_direct_sparse_events_v3(
        state, direct_events, event_counts, validation, event_capacity_rows
    );
    check_science_world_controls_v1(state, science_world_controls);
    const c10::cuda::CUDAGuard guard(state.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    advance_optimized_kernel<<<state.size(0), kThreads, 0, stream>>>(
        state.data_ptr<std::uint8_t>(), state.stride(0), nullptr, 0,
        event_counts.data_ptr<std::int32_t>(),
        reinterpret_cast<SparseLifecycleEventV3*>(
            direct_events.data_ptr<std::uint8_t>()
        ),
        direct_events.stride(0) / kSparseEventRowBytesV3,
        static_cast<int>(event_capacity_rows),
        validation.data_ptr<std::int32_t>(),
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
        science_world_controls.data_ptr<std::int64_t>(),
        science_world_controls.stride(0),
        nullptr, 0, nullptr, 0,
        nullptr, 0, nullptr, 0,
        static_cast<int>(temporal_k), kFullCapabilityMask
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void advance_direct_sparse_compact_science_worlds_v1(
    torch::Tensor state,
    torch::Tensor direct_events,
    torch::Tensor event_counts,
    torch::Tensor validation,
    torch::Tensor science_world_controls,
    torch::Tensor science_agent_metrics,
    torch::Tensor science_event_metrics,
    torch::Tensor science_agent_birth_provenance,
    torch::Tensor science_event_birth_provenance,
    const std::int64_t temporal_k,
    const std::int64_t event_capacity_rows
) {
    check_state(state);
    TORCH_CHECK(temporal_k >= 1 && temporal_k <= 64,
        "temporal_k must be in [1,64]");
    check_direct_sparse_events_v3(
        state, direct_events, event_counts, validation, event_capacity_rows
    );
    check_science_world_controls_v1(state, science_world_controls);
    check_compact_science_metrics_v1(
        state, science_agent_metrics, science_event_metrics,
        event_capacity_rows
    );
    check_compact_science_birth_provenance_v1(
        state, science_agent_birth_provenance,
        science_event_birth_provenance, event_capacity_rows
    );
    const c10::cuda::CUDAGuard guard(state.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    science_event_metrics.zero_();
    science_event_birth_provenance.zero_();
    advance_optimized_kernel<<<state.size(0), kThreads, 0, stream>>>(
        state.data_ptr<std::uint8_t>(), state.stride(0), nullptr, 0,
        event_counts.data_ptr<std::int32_t>(),
        reinterpret_cast<SparseLifecycleEventV3*>(
            direct_events.data_ptr<std::uint8_t>()
        ),
        direct_events.stride(0) / kSparseEventRowBytesV3,
        static_cast<int>(event_capacity_rows),
        validation.data_ptr<std::int32_t>(),
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
        science_world_controls.data_ptr<std::int64_t>(),
        science_world_controls.stride(0),
        science_agent_metrics.data_ptr<std::int64_t>(),
        science_agent_metrics.stride(0),
        science_event_metrics.data_ptr<std::int64_t>(),
        science_event_metrics.stride(0),
        science_agent_birth_provenance.data_ptr<std::uint8_t>(),
        science_agent_birth_provenance.stride(0),
        science_event_birth_provenance.data_ptr<std::uint8_t>(),
        science_event_birth_provenance.stride(0),
        static_cast<int>(temporal_k), kFullCapabilityMask
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void advance_direct_sparse_science_v1(
    torch::Tensor state,
    torch::Tensor direct_events,
    torch::Tensor event_counts,
    torch::Tensor validation,
    torch::Tensor science_action_ids,
    torch::Tensor science_action_values,
    torch::Tensor science_birth_child_ids,
    torch::Tensor science_birth_parent_a_ids,
    torch::Tensor science_birth_parent_b_ids,
    torch::Tensor science_birth_mutation_counts,
    torch::Tensor science_birth_child_slots,
    torch::Tensor science_birth_sites,
    torch::Tensor science_birth_counts,
    const std::int64_t temporal_k,
    const std::int64_t event_capacity_rows,
    const std::int64_t capability_mask
) {
    check_state(state);
    TORCH_CHECK(temporal_k >= 1 && temporal_k <= 64,
        "temporal_k must be in [1,64]");
    check_direct_sparse_events_v3(
        state, direct_events, event_counts, validation, event_capacity_rows
    );
    check_science_capture_v1(
        state, science_action_ids, science_action_values,
        science_birth_child_ids, science_birth_parent_a_ids,
        science_birth_parent_b_ids, science_birth_mutation_counts,
        science_birth_child_slots, science_birth_sites, science_birth_counts,
        temporal_k
    );
    TORCH_CHECK(capability_mask >= 0 && capability_mask <= kFullCapabilityMask,
        "capability mask is invalid");
    const c10::cuda::CUDAGuard guard(state.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    science_action_ids.zero_();
    science_action_values.zero_();
    science_birth_counts.zero_();
    science_birth_child_ids.zero_();
    science_birth_parent_a_ids.zero_();
    science_birth_parent_b_ids.zero_();
    science_birth_mutation_counts.zero_();
    science_birth_child_slots.zero_();
    science_birth_sites.zero_();
    advance_optimized_kernel<<<state.size(0), kThreads, 0, stream>>>(
        state.data_ptr<std::uint8_t>(), state.stride(0), nullptr, 0,
        event_counts.data_ptr<std::int32_t>(),
        reinterpret_cast<SparseLifecycleEventV3*>(
            direct_events.data_ptr<std::uint8_t>()
        ),
        direct_events.stride(0) / kSparseEventRowBytesV3,
        static_cast<int>(event_capacity_rows),
        validation.data_ptr<std::int32_t>(),
        science_action_ids.data_ptr<std::int64_t>(),
        science_action_values.data_ptr<std::uint8_t>(),
        science_birth_child_ids.data_ptr<std::int64_t>(),
        science_birth_parent_a_ids.data_ptr<std::int64_t>(),
        science_birth_parent_b_ids.data_ptr<std::int64_t>(),
        science_birth_mutation_counts.data_ptr<std::int32_t>(),
        science_birth_child_slots.data_ptr<std::int32_t>(),
        science_birth_sites.data_ptr<std::int32_t>(),
        science_birth_counts.data_ptr<std::int32_t>(),
        nullptr, 0,
        nullptr, 0, nullptr, 0,
        nullptr, 0, nullptr, 0,
        static_cast<int>(temporal_k),
        static_cast<std::uint64_t>(capability_mask)
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void advance_direct_sparse_science_worlds_v1(
    torch::Tensor state,
    torch::Tensor direct_events,
    torch::Tensor event_counts,
    torch::Tensor validation,
    torch::Tensor science_action_ids,
    torch::Tensor science_action_values,
    torch::Tensor science_birth_child_ids,
    torch::Tensor science_birth_parent_a_ids,
    torch::Tensor science_birth_parent_b_ids,
    torch::Tensor science_birth_mutation_counts,
    torch::Tensor science_birth_child_slots,
    torch::Tensor science_birth_sites,
    torch::Tensor science_birth_counts,
    torch::Tensor science_world_controls,
    const std::int64_t temporal_k,
    const std::int64_t event_capacity_rows
) {
    check_state(state);
    TORCH_CHECK(temporal_k >= 1 && temporal_k <= 64,
        "temporal_k must be in [1,64]");
    check_direct_sparse_events_v3(
        state, direct_events, event_counts, validation, event_capacity_rows
    );
    check_science_capture_v1(
        state, science_action_ids, science_action_values,
        science_birth_child_ids, science_birth_parent_a_ids,
        science_birth_parent_b_ids, science_birth_mutation_counts,
        science_birth_child_slots, science_birth_sites, science_birth_counts,
        temporal_k
    );
    check_science_world_controls_v1(state, science_world_controls);
    const c10::cuda::CUDAGuard guard(state.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    science_action_ids.zero_();
    science_action_values.zero_();
    science_birth_counts.zero_();
    science_birth_child_ids.zero_();
    science_birth_parent_a_ids.zero_();
    science_birth_parent_b_ids.zero_();
    science_birth_mutation_counts.zero_();
    science_birth_child_slots.zero_();
    science_birth_sites.zero_();
    advance_optimized_kernel<<<state.size(0), kThreads, 0, stream>>>(
        state.data_ptr<std::uint8_t>(), state.stride(0), nullptr, 0,
        event_counts.data_ptr<std::int32_t>(),
        reinterpret_cast<SparseLifecycleEventV3*>(
            direct_events.data_ptr<std::uint8_t>()
        ),
        direct_events.stride(0) / kSparseEventRowBytesV3,
        static_cast<int>(event_capacity_rows),
        validation.data_ptr<std::int32_t>(),
        science_action_ids.data_ptr<std::int64_t>(),
        science_action_values.data_ptr<std::uint8_t>(),
        science_birth_child_ids.data_ptr<std::int64_t>(),
        science_birth_parent_a_ids.data_ptr<std::int64_t>(),
        science_birth_parent_b_ids.data_ptr<std::int64_t>(),
        science_birth_mutation_counts.data_ptr<std::int32_t>(),
        science_birth_child_slots.data_ptr<std::int32_t>(),
        science_birth_sites.data_ptr<std::int32_t>(),
        science_birth_counts.data_ptr<std::int32_t>(),
        science_world_controls.data_ptr<std::int64_t>(),
        science_world_controls.stride(0),
        nullptr, 0, nullptr, 0,
        nullptr, 0, nullptr, 0,
        static_cast<int>(temporal_k), kFullCapabilityMask
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void compact_events_v3(
    torch::Tensor state,
    torch::Tensor dense_events,
    torch::Tensor event_counts,
    torch::Tensor world_offsets,
    torch::Tensor packed_events,
    torch::Tensor validation,
    const std::int64_t start_tick,
    const std::int64_t tick_count,
    const std::int64_t event_capacity_k
) {
    check_state(state);
    TORCH_CHECK(start_tick >= 0, "start_tick must be nonnegative signed int64");
    TORCH_CHECK(tick_count >= 1 && tick_count <= event_capacity_k,
        "tick_count must be in [1,event_capacity_k]");
    TORCH_CHECK(start_tick <= std::numeric_limits<std::int64_t>::max() - tick_count,
        "sparse event tick range exceeds signed int64");
    check_events(state, dense_events, event_counts, tick_count, event_capacity_k);
    TORCH_CHECK(world_offsets.is_cuda() &&
        world_offsets.scalar_type() == torch::kInt64 &&
        world_offsets.is_contiguous() && world_offsets.dim() == 1 &&
        world_offsets.size(0) == state.size(0) + 1,
        "world_offsets must be int64[B+1] on CUDA");
    TORCH_CHECK(packed_events.is_cuda() &&
        packed_events.scalar_type() == torch::kUInt8 &&
        packed_events.is_contiguous() && packed_events.dim() == 2 &&
        packed_events.size(1) == kSparseEventRowBytesV3,
        "packed_events must be uint8[N,32] on CUDA");
    TORCH_CHECK(validation.is_cuda() &&
        validation.scalar_type() == torch::kInt32 &&
        validation.is_contiguous() && validation.dim() == 1 &&
        validation.size(0) == state.size(0),
        "validation must be int32[B] on CUDA");
    TORCH_CHECK(world_offsets.device() == state.device() &&
        packed_events.device() == state.device() &&
        validation.device() == state.device(),
        "sparse event staging device mismatch");
    const c10::cuda::CUDAGuard guard(state.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    compact_events_v3_kernel<<<state.size(0), kThreads, 0, stream>>>(
        state.data_ptr<std::uint8_t>(), state.stride(0),
        reinterpret_cast<const LifecycleEventRow*>(
            dense_events.data_ptr<std::uint8_t>()
        ),
        dense_events.stride(0) / kEventRowBytes,
        event_counts.data_ptr<std::int32_t>(),
        world_offsets.data_ptr<std::int64_t>(),
        reinterpret_cast<SparseLifecycleEventV3*>(
            packed_events.data_ptr<std::uint8_t>()
        ),
        packed_events.size(0), validation.data_ptr<std::int32_t>(),
        static_cast<std::uint64_t>(start_tick), static_cast<int>(tick_count)
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void pack_direct_sparse_events_v3(
    torch::Tensor state,
    torch::Tensor direct_events,
    torch::Tensor event_counts,
    torch::Tensor advance_validation,
    torch::Tensor world_offsets,
    torch::Tensor packed_events,
    torch::Tensor validation,
    const std::int64_t tick_count,
    const std::int64_t event_capacity_rows
) {
    check_state(state);
    TORCH_CHECK(tick_count >= 1 && tick_count <= 64,
        "tick_count must be in [1,64]");
    check_direct_sparse_events_v3(
        state, direct_events, event_counts, advance_validation,
        event_capacity_rows
    );
    TORCH_CHECK(world_offsets.is_cuda() &&
        world_offsets.scalar_type() == torch::kInt64 &&
        world_offsets.is_contiguous() && world_offsets.dim() == 1 &&
        world_offsets.size(0) == state.size(0) + 1,
        "world_offsets must be int64[B+1] on CUDA");
    TORCH_CHECK(packed_events.is_cuda() &&
        packed_events.scalar_type() == torch::kUInt8 &&
        packed_events.is_contiguous() && packed_events.dim() == 2 &&
        packed_events.size(1) == kSparseEventRowBytesV3,
        "packed_events must be uint8[N,32] on CUDA");
    TORCH_CHECK(validation.is_cuda() &&
        validation.scalar_type() == torch::kInt32 &&
        validation.is_contiguous() && validation.dim() == 1 &&
        validation.size(0) == state.size(0),
        "validation must be int32[B] on CUDA");
    TORCH_CHECK(world_offsets.device() == state.device() &&
        packed_events.device() == state.device() &&
        validation.device() == state.device(),
        "direct sparse packing device mismatch");
    TORCH_CHECK(!tensors_overlap(state, world_offsets) &&
        !tensors_overlap(state, packed_events) &&
        !tensors_overlap(state, validation) &&
        !tensors_overlap(direct_events, world_offsets) &&
        !tensors_overlap(direct_events, packed_events) &&
        !tensors_overlap(direct_events, validation) &&
        !tensors_overlap(event_counts, world_offsets) &&
        !tensors_overlap(event_counts, packed_events) &&
        !tensors_overlap(event_counts, validation) &&
        !tensors_overlap(advance_validation, world_offsets) &&
        !tensors_overlap(advance_validation, packed_events) &&
        !tensors_overlap(advance_validation, validation) &&
        !tensors_overlap(world_offsets, packed_events) &&
        !tensors_overlap(world_offsets, validation) &&
        !tensors_overlap(packed_events, validation),
        "direct sparse pack tensors must not overlap");
    const c10::cuda::CUDAGuard guard(state.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    pack_direct_sparse_events_v3_kernel<<<state.size(0), kThreads, 0, stream>>>(
        state.data_ptr<std::uint8_t>(), state.stride(0),
        reinterpret_cast<const SparseLifecycleEventV3*>(
            direct_events.data_ptr<std::uint8_t>()
        ),
        direct_events.stride(0) / kSparseEventRowBytesV3,
        static_cast<int>(event_capacity_rows),
        event_counts.data_ptr<std::int32_t>(),
        advance_validation.data_ptr<std::int32_t>(),
        world_offsets.data_ptr<std::int64_t>(),
        reinterpret_cast<SparseLifecycleEventV3*>(
            packed_events.data_ptr<std::uint8_t>()
        ),
        packed_events.size(0), nullptr, 0, nullptr,
        nullptr, 0, nullptr,
        validation.data_ptr<std::int32_t>(),
        static_cast<int>(tick_count)
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void pack_direct_sparse_science_events_v1(
    torch::Tensor state,
    torch::Tensor direct_events,
    torch::Tensor event_counts,
    torch::Tensor advance_validation,
    torch::Tensor world_offsets,
    torch::Tensor packed_events,
    torch::Tensor validation,
    torch::Tensor science_agent_metrics,
    torch::Tensor science_event_metrics,
    torch::Tensor packed_science_event_metrics,
    torch::Tensor science_agent_birth_provenance,
    torch::Tensor science_event_birth_provenance,
    torch::Tensor packed_science_event_birth_provenance,
    const std::int64_t tick_count,
    const std::int64_t event_capacity_rows
) {
    check_state(state);
    TORCH_CHECK(tick_count >= 1 && tick_count <= 64,
        "tick_count must be in [1,64]");
    check_direct_sparse_events_v3(
        state, direct_events, event_counts, advance_validation,
        event_capacity_rows
    );
    check_compact_science_metrics_v1(
        state, science_agent_metrics, science_event_metrics,
        event_capacity_rows
    );
    check_compact_science_birth_provenance_v1(
        state, science_agent_birth_provenance,
        science_event_birth_provenance, event_capacity_rows
    );
    TORCH_CHECK(world_offsets.is_cuda() &&
        world_offsets.scalar_type() == torch::kInt64 &&
        world_offsets.is_contiguous() && world_offsets.dim() == 1 &&
        world_offsets.size(0) == state.size(0) + 1,
        "world_offsets must be int64[B+1] on CUDA");
    TORCH_CHECK(packed_events.is_cuda() &&
        packed_events.scalar_type() == torch::kUInt8 &&
        packed_events.is_contiguous() && packed_events.dim() == 2 &&
        packed_events.size(1) == kSparseEventRowBytesV3,
        "packed_events must be uint8[N,32] on CUDA");
    TORCH_CHECK(validation.is_cuda() &&
        validation.scalar_type() == torch::kInt32 &&
        validation.is_contiguous() && validation.dim() == 1 &&
        validation.size(0) == state.size(0),
        "validation must be int32[B] on CUDA");
    TORCH_CHECK(packed_science_event_metrics.is_cuda() &&
        packed_science_event_metrics.scalar_type() == torch::kInt64 &&
        packed_science_event_metrics.is_contiguous() &&
        packed_science_event_metrics.dim() == 2 &&
        packed_science_event_metrics.size(0) == packed_events.size(0) &&
        packed_science_event_metrics.size(1) == kScienceAgentMetricWidthV1,
        "packed science event metrics must be int64[N,21] on CUDA");
    TORCH_CHECK(packed_science_event_birth_provenance.is_cuda() &&
        packed_science_event_birth_provenance.scalar_type() == torch::kUInt8 &&
        packed_science_event_birth_provenance.is_contiguous() &&
        packed_science_event_birth_provenance.dim() == 2 &&
        packed_science_event_birth_provenance.size(0) == packed_events.size(0) &&
        packed_science_event_birth_provenance.size(1) ==
            kScienceBirthProvenanceWidthV1,
        "packed science birth provenance must be uint8[N,132] on CUDA");
    TORCH_CHECK(world_offsets.device() == state.device() &&
        packed_events.device() == state.device() &&
        validation.device() == state.device() &&
        packed_science_event_metrics.device() == state.device() &&
        packed_science_event_birth_provenance.device() == state.device(),
        "direct sparse science packing device mismatch");
    const c10::cuda::CUDAGuard guard(state.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    pack_direct_sparse_events_v3_kernel<<<state.size(0), kThreads, 0, stream>>>(
        state.data_ptr<std::uint8_t>(), state.stride(0),
        reinterpret_cast<const SparseLifecycleEventV3*>(
            direct_events.data_ptr<std::uint8_t>()
        ),
        direct_events.stride(0) / kSparseEventRowBytesV3,
        static_cast<int>(event_capacity_rows),
        event_counts.data_ptr<std::int32_t>(),
        advance_validation.data_ptr<std::int32_t>(),
        world_offsets.data_ptr<std::int64_t>(),
        reinterpret_cast<SparseLifecycleEventV3*>(
            packed_events.data_ptr<std::uint8_t>()
        ),
        packed_events.size(0),
        science_event_metrics.data_ptr<std::int64_t>(),
        science_event_metrics.stride(0),
        packed_science_event_metrics.data_ptr<std::int64_t>(),
        science_event_birth_provenance.data_ptr<std::uint8_t>(),
        science_event_birth_provenance.stride(0),
        packed_science_event_birth_provenance.data_ptr<std::uint8_t>(),
        validation.data_ptr<std::int32_t>(),
        static_cast<int>(tick_count)
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

bool is_lower_sha256(const std::string& value) {
    if (value.size() != 64) {
        return false;
    }
    for (const char character : value) {
        if (!((character >= '0' && character <= '9') ||
              (character >= 'a' && character <= 'f'))) {
            return false;
        }
    }
    return true;
}

const char* graph_node_type_name(const cudaGraphNodeType type) {
    switch (type) {
        case cudaGraphNodeTypeKernel:
            return "kernel";
        case cudaGraphNodeTypeMemcpy:
            return "memcpy";
        case cudaGraphNodeTypeMemset:
            return "memset";
        case cudaGraphNodeTypeHost:
            return "host";
        case cudaGraphNodeTypeGraph:
            return "child_graph";
        case cudaGraphNodeTypeEmpty:
            return "empty";
        case cudaGraphNodeTypeWaitEvent:
            return "event_wait";
        case cudaGraphNodeTypeEventRecord:
            return "event_record";
        default:
            return "other";
    }
}

std::atomic<std::uint64_t> execution_plan_ordinal{0};

class CudaFullEvolutionExecutionPlanV2 final {
public:
    CudaFullEvolutionExecutionPlanV2(
        std::string mode,
        torch::Tensor state,
        torch::Tensor events,
        torch::Tensor event_counts,
        const std::int64_t dispatch_temporal_k,
        const std::int64_t event_capacity_k,
        const std::int64_t capability_mask,
        std::string profile_id,
        std::string artifact_sha256,
        std::string manifest_sha256
    ) :
        mode_(std::move(mode)),
        state_(std::move(state)),
        events_(std::move(events)),
        event_counts_(std::move(event_counts)),
        dispatch_temporal_k_(dispatch_temporal_k),
        event_capacity_k_(event_capacity_k),
        capability_mask_(static_cast<std::uint64_t>(capability_mask)),
        profile_id_(std::move(profile_id)),
        artifact_sha256_(std::move(artifact_sha256)),
        manifest_sha256_(std::move(manifest_sha256)),
        native_plan_ordinal_(execution_plan_ordinal.fetch_add(1) + 1) {
        TORCH_CHECK(
            mode_ == "eager_monolith" || mode_ == "cuda_graph_monolith",
            "execution mode must be eager_monolith or cuda_graph_monolith"
        );
        TORCH_CHECK(
            dispatch_temporal_k_ >= 1 && dispatch_temporal_k_ <= 64,
            "dispatch_temporal_k must be in [1,64]"
        );
        TORCH_CHECK(
            event_capacity_k_ >= dispatch_temporal_k_ && event_capacity_k_ <= 64,
            "event_capacity_k must cover dispatch_temporal_k and be in [1,64]"
        );
        TORCH_CHECK(
            capability_mask >= 0 && capability_mask <= kFullCapabilityMask,
            "capability mask is invalid"
        );
        TORCH_CHECK(
            profile_id_ == "eiel_full_evolution_cuda_native_v2",
            "execution plan profile identity mismatch"
        );
        TORCH_CHECK(
            is_lower_sha256(artifact_sha256_) && is_lower_sha256(manifest_sha256_),
            "execution plan artifact identity must contain two lowercase SHA-256 values"
        );
        check_state(state_);
        check_events(
            state_, events_, event_counts_, dispatch_temporal_k_, event_capacity_k_
        );
        device_index_ = state_.get_device();
        batch_size_ = state_.size(0);
        state_pointer_ = state_.data_ptr<std::uint8_t>();
        state_stride_ = state_.stride(0);
        event_pointer_ = reinterpret_cast<LifecycleEventRow*>(
            events_.data_ptr<std::uint8_t>()
        );
        event_stride_rows_ = events_.stride(0) / kEventRowBytes;
        event_count_pointer_ = event_counts_.data_ptr<std::int32_t>();
        kernel_temporal_k_ = static_cast<int>(dispatch_temporal_k_);
        kernel_capability_mask_ = capability_mask_;

        const c10::cuda::CUDAGuard guard(state_.device());
        construction_stream_ = at::cuda::getCurrentCUDAStream();
        if (mode_ == "cuda_graph_monolith") {
            try {
                create_graph();
                inspect_graph();
            } catch (...) {
                close_noexcept();
                throw;
            }
        }
    }

    CudaFullEvolutionExecutionPlanV2(
        const CudaFullEvolutionExecutionPlanV2&
    ) = delete;
    CudaFullEvolutionExecutionPlanV2& operator=(
        const CudaFullEvolutionExecutionPlanV2&
    ) = delete;

    ~CudaFullEvolutionExecutionPlanV2() {
        close_noexcept();
    }

    void launch(
        const std::int64_t start_next_absolute_tick,
        const std::int64_t ticks
    ) {
        TORCH_CHECK(!closed_, "execution plan is closed");
        TORCH_CHECK(start_next_absolute_tick >= 0,
            "start_next_absolute_tick must be nonnegative signed int64");
        TORCH_CHECK(
            ticks == dispatch_temporal_k_,
            "execution-plan launch ticks must equal its immutable dispatch_temporal_k"
        );
        validate_binding();
        TORCH_CHECK(
            start_next_absolute_tick <=
                std::numeric_limits<std::int64_t>::max() - ticks,
            "execution receipt absolute-tick extent exceeds signed int64"
        );
        const c10::cuda::CUDAGuard guard(state_.device());
        const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
        TORCH_CHECK(
            stream == construction_stream_,
            "execution plan is bound to one immutable CUDA stream, including "
            "the valid default stream 0"
        );
        if (mode_ == "cuda_graph_monolith") {
            C10_CUDA_CHECK(cudaGraphLaunch(graph_exec_, stream));
        } else {
            advance_optimized_kernel<<<batch_size_, kThreads, 0, stream>>>(
                state_pointer_, state_stride_, event_pointer_, event_stride_rows_,
                event_count_pointer_, direct_sparse_pointer_,
                direct_sparse_stride_rows_, direct_sparse_capacity_rows_,
                direct_sparse_validation_pointer_,
                nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                nullptr, 0,
                nullptr, 0, nullptr, 0,
                nullptr, 0, nullptr, 0,
                static_cast<int>(ticks),
                kernel_capability_mask_
            );
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
        last_launch_stream_ = stream;
        ++launch_ordinal_;
        last_start_next_absolute_tick_ = start_next_absolute_tick;
        last_tick_count_ = ticks;
    }

    py::dict topology_manifest() {
        TORCH_CHECK(!closed_, "execution plan is closed");
        validate_binding();
        if (mode_ == "cuda_graph_monolith") {
            inspect_graph();
        }
        py::dict result;
        result["schema_version"] =
            "eiel_full_evolution_cuda_execution_plan_topology_v2";
        result["execution_mode"] = mode_;
        result["profile_id"] = profile_id_;
        result["artifact_sha256"] = artifact_sha256_;
        result["manifest_sha256"] = manifest_sha256_;
        result["native_plan_ordinal"] = native_plan_ordinal_;
        result["device_index"] = device_index_;
        result["B"] = batch_size_;
        result["dispatch_temporal_k"] = dispatch_temporal_k_;
        result["event_capacity_k"] = event_capacity_k_;
        result["capability_mask"] = capability_mask_;
        result["construction_kind"] = mode_ == "cuda_graph_monolith"
            ? "explicit_cuda_graph_api_no_stream_capture"
            : "eager_kernel_launch";
        result["construction_stream_handle"] = reinterpret_cast<std::uintptr_t>(
            construction_stream_
        );
        result["graph_handle"] = reinterpret_cast<std::uintptr_t>(graph_);
        result["graph_exec_handle"] = reinterpret_cast<std::uintptr_t>(graph_exec_);
        result["graph_node_count"] = graph_node_count_;
        result["execution_kernel_count"] = 1;
        py::list node_types;
        if (mode_ == "cuda_graph_monolith") {
            node_types.append(graph_node_type_name(graph_node_type_));
        }
        result["node_types"] = node_types;
        py::list kernel_nodes;
        py::dict kernel;
        kernel["name"] = "advance_optimized_kernel";
        kernel["node_type"] = "kernel";
        kernel["runtime_graph_node"] = mode_ == "cuda_graph_monolith";
        kernel["function_pointer_verified"] = mode_ == "cuda_graph_monolith"
            ? graph_function_verified_ : true;
        kernel["grid"] = py::make_tuple(batch_size_, 1, 1);
        kernel["block"] = py::make_tuple(kThreads, 1, 1);
        kernel["shared_memory_bytes"] = 0;
        py::dict parameters;
        parameters["state_pointer"] = reinterpret_cast<std::uintptr_t>(state_pointer_);
        parameters["state_stride_bytes"] = state_stride_;
        parameters["event_pointer"] = reinterpret_cast<std::uintptr_t>(event_pointer_);
        parameters["event_stride_rows"] = event_stride_rows_;
        parameters["event_count_pointer"] = reinterpret_cast<std::uintptr_t>(
            event_count_pointer_
        );
        parameters["dispatch_temporal_k"] = dispatch_temporal_k_;
        parameters["capability_mask"] = capability_mask_;
        kernel["parameters"] = parameters;
        kernel_nodes.append(kernel);
        result["kernel_nodes"] = kernel_nodes;
        result["dependency_edges"] = py::list();
        py::dict pointers;
        pointers["state"] = reinterpret_cast<std::uintptr_t>(state_pointer_);
        pointers["events"] = reinterpret_cast<std::uintptr_t>(event_pointer_);
        pointers["event_counts"] = reinterpret_cast<std::uintptr_t>(
            event_count_pointer_
        );
        result["pointer_bindings"] = pointers;
        result["strong_tensor_references"] = true;
        result["pointer_substitution_allowed"] = false;
        result["runtime_inspected"] = mode_ == "cuda_graph_monolith";
        result["declaration_is_execution_evidence"] = false;
        result["launch_stream_policy"] =
            "immutable_construction_stream_including_default_stream_0";
        return result;
    }

    py::object last_execution_receipt() const {
        TORCH_CHECK(!closed_, "execution plan is closed");
        if (launch_ordinal_ == 0) {
            return py::none();
        }
        py::dict result;
        result["schema_version"] =
            "eiel_full_evolution_cuda_native_execution_receipt_v2";
        result["execution_mode"] = mode_;
        result["profile_id"] = profile_id_;
        result["artifact_sha256"] = artifact_sha256_;
        result["manifest_sha256"] = manifest_sha256_;
        result["native_plan_ordinal"] = native_plan_ordinal_;
        result["launch_ordinal"] = launch_ordinal_;
        result["device_index"] = device_index_;
        result["launch_stream_handle"] = reinterpret_cast<std::uintptr_t>(
            last_launch_stream_
        );
        result["construction_stream_handle"] = reinterpret_cast<std::uintptr_t>(
            construction_stream_
        );
        result["launch_stream_policy"] =
            "immutable_construction_stream_including_default_stream_0";
        result["start_next_absolute_tick"] = last_start_next_absolute_tick_;
        result["end_next_absolute_tick"] =
            last_start_next_absolute_tick_ + last_tick_count_;
        result["tick_count"] = last_tick_count_;
        result["B"] = batch_size_;
        result["dispatch_temporal_k"] = dispatch_temporal_k_;
        result["event_capacity_k"] = event_capacity_k_;
        result["capability_mask"] = capability_mask_;
        result["enqueued_world_ticks"] = batch_size_ * last_tick_count_;
        result["committed_world_ticks"] = 0;
        result["kernel"] = "advance_optimized_kernel";
        result["graph_launched"] = mode_ == "cuda_graph_monolith";
        result["topology_graph_node_count"] = graph_node_count_;
        result["topology_execution_kernel_count"] = 1;
        py::list node_types;
        if (mode_ == "cuda_graph_monolith") {
            node_types.append(graph_node_type_name(graph_node_type_));
        }
        result["topology_node_types"] = node_types;
        py::list kernel_nodes;
        py::dict kernel_node;
        kernel_node["name"] = "advance_optimized_kernel";
        kernel_node["node_type"] = "kernel";
        kernel_node["runtime_graph_node"] = mode_ == "cuda_graph_monolith";
        kernel_node["function_pointer_verified"] = mode_ == "cuda_graph_monolith"
            ? graph_function_verified_ : true;
        kernel_node["grid"] = py::make_tuple(batch_size_, 1, 1);
        kernel_node["block"] = py::make_tuple(kThreads, 1, 1);
        kernel_node["shared_memory_bytes"] = 0;
        py::dict parameters;
        parameters["state_pointer"] = reinterpret_cast<std::uintptr_t>(state_pointer_);
        parameters["state_stride_bytes"] = state_stride_;
        parameters["event_pointer"] = reinterpret_cast<std::uintptr_t>(event_pointer_);
        parameters["event_stride_rows"] = event_stride_rows_;
        parameters["event_count_pointer"] = reinterpret_cast<std::uintptr_t>(
            event_count_pointer_
        );
        parameters["dispatch_temporal_k"] = dispatch_temporal_k_;
        parameters["capability_mask"] = capability_mask_;
        kernel_node["parameters"] = parameters;
        kernel_nodes.append(kernel_node);
        result["topology_kernel_nodes"] = kernel_nodes;
        result["topology_dependency_edges"] = py::list();
        py::dict pointers;
        pointers["state"] = reinterpret_cast<std::uintptr_t>(state_pointer_);
        pointers["events"] = reinterpret_cast<std::uintptr_t>(event_pointer_);
        pointers["event_counts"] = reinterpret_cast<std::uintptr_t>(
            event_count_pointer_
        );
        result["pointer_bindings"] = pointers;
        result["strong_tensor_references"] = true;
        result["pointer_substitution_allowed"] = false;
        result["runtime_inspected"] = mode_ == "cuda_graph_monolith";
        result["native_status"] = "enqueued_unvalidated";
        result["event_status"] = "not_validated_by_native_launch";
        result["per_node_timing"] = py::none();
        result["inter_node_gap_timing"] = py::none();
        result["timing_fabrication_allowed"] = false;
        return std::move(result);
    }

    void close() {
        if (closed_) {
            return;
        }
        const c10::cuda::CUDAGuard guard(state_.device());
        if (launch_ordinal_ != 0) {
            C10_CUDA_CHECK(cudaStreamSynchronize(construction_stream_));
        }
        if (graph_exec_ != nullptr) {
            C10_CUDA_CHECK(cudaGraphExecDestroy(graph_exec_));
            graph_exec_ = nullptr;
        }
        if (graph_ != nullptr) {
            C10_CUDA_CHECK(cudaGraphDestroy(graph_));
            graph_ = nullptr;
        }
        state_ = torch::Tensor();
        events_ = torch::Tensor();
        event_counts_ = torch::Tensor();
        closed_ = true;
    }

    bool closed() const {
        return closed_;
    }

private:
    void create_graph() {
        C10_CUDA_CHECK(cudaGraphCreate(&graph_, 0));
        void* arguments[] = {
            &state_pointer_, &state_stride_, &event_pointer_, &event_stride_rows_,
            &event_count_pointer_, &direct_sparse_pointer_,
            &direct_sparse_stride_rows_, &direct_sparse_capacity_rows_,
            &direct_sparse_validation_pointer_,
            &science_action_ids_pointer_, &science_action_values_pointer_,
            &science_birth_child_ids_pointer_, &science_birth_parent_a_ids_pointer_,
            &science_birth_parent_b_ids_pointer_, &science_birth_mutation_counts_pointer_,
            &science_birth_child_slots_pointer_, &science_birth_sites_pointer_,
            &science_birth_counts_pointer_,
            &science_world_controls_pointer_, &science_world_control_stride_,
            &science_agent_metrics_pointer_, &science_agent_metric_stride_,
            &science_event_metrics_pointer_, &science_event_metric_stride_,
            &science_agent_birth_provenance_pointer_,
            &science_agent_birth_provenance_stride_,
            &science_event_birth_provenance_pointer_,
            &science_event_birth_provenance_stride_,
            &kernel_temporal_k_,
            &kernel_capability_mask_,
        };
        cudaKernelNodeParams parameters{};
        parameters.func = reinterpret_cast<void*>(advance_optimized_kernel);
        parameters.gridDim = dim3(static_cast<unsigned int>(batch_size_), 1, 1);
        parameters.blockDim = dim3(kThreads, 1, 1);
        parameters.sharedMemBytes = 0;
        parameters.kernelParams = arguments;
        parameters.extra = nullptr;
        C10_CUDA_CHECK(cudaGraphAddKernelNode(
            &kernel_node_, graph_, nullptr, 0, &parameters
        ));
        C10_CUDA_CHECK(cudaGraphInstantiate(
            &graph_exec_, graph_, nullptr, nullptr, 0
        ));
    }

    void inspect_graph() {
        TORCH_CHECK(graph_ != nullptr && graph_exec_ != nullptr,
            "CUDA Graph execution plan is incomplete");
        std::size_t node_count = 0;
        C10_CUDA_CHECK(cudaGraphGetNodes(graph_, nullptr, &node_count));
        TORCH_CHECK(node_count == 1,
            "cuda_graph_monolith must contain exactly one runtime node");
        std::vector<cudaGraphNode_t> nodes(node_count);
        C10_CUDA_CHECK(cudaGraphGetNodes(graph_, nodes.data(), &node_count));
        cudaGraphNodeType node_type{};
        C10_CUDA_CHECK(cudaGraphNodeGetType(nodes.front(), &node_type));
        TORCH_CHECK(node_type == cudaGraphNodeTypeKernel,
            "cuda_graph_monolith runtime node must be a kernel");
        cudaKernelNodeParams parameters{};
        C10_CUDA_CHECK(cudaGraphKernelNodeGetParams(nodes.front(), &parameters));
        TORCH_CHECK(
            parameters.func == reinterpret_cast<void*>(advance_optimized_kernel) &&
            parameters.gridDim.x == static_cast<unsigned int>(batch_size_) &&
            parameters.gridDim.y == 1 && parameters.gridDim.z == 1 &&
            parameters.blockDim.x == kThreads &&
            parameters.blockDim.y == 1 && parameters.blockDim.z == 1 &&
            parameters.sharedMemBytes == 0,
            "cuda_graph_monolith kernel node parameters changed"
        );
        std::size_t edge_count = 0;
        C10_CUDA_CHECK(cudaGraphGetEdges(graph_, nullptr, nullptr, &edge_count));
        TORCH_CHECK(edge_count == 0,
            "cuda_graph_monolith must not contain dependency edges");
        graph_node_count_ = static_cast<std::int64_t>(node_count);
        graph_node_type_ = node_type;
        graph_function_verified_ = true;
    }

    void validate_binding() const {
        TORCH_CHECK(!closed_, "execution plan is closed");
        check_state(state_);
        check_events(
            state_, events_, event_counts_, dispatch_temporal_k_, event_capacity_k_
        );
        TORCH_CHECK(
            state_.get_device() == device_index_ && state_.size(0) == batch_size_ &&
            state_.data_ptr<std::uint8_t>() == state_pointer_ &&
            state_.stride(0) == state_stride_ &&
            events_.data_ptr<std::uint8_t>() ==
                reinterpret_cast<std::uint8_t*>(event_pointer_) &&
            events_.stride(0) / kEventRowBytes == event_stride_rows_ &&
            event_counts_.data_ptr<std::int32_t>() == event_count_pointer_,
            "execution plan tensor pointer, shape, stride, batch, or device binding changed"
        );
    }

    void close_noexcept() noexcept {
        if (closed_) {
            return;
        }
        int prior_device = -1;
        cudaGetDevice(&prior_device);
        if (device_index_ >= 0) {
            cudaSetDevice(device_index_);
        }
        if (launch_ordinal_ != 0) {
            cudaStreamSynchronize(construction_stream_);
        }
        if (graph_exec_ != nullptr) {
            cudaGraphExecDestroy(graph_exec_);
            graph_exec_ = nullptr;
        }
        if (graph_ != nullptr) {
            cudaGraphDestroy(graph_);
            graph_ = nullptr;
        }
        if (prior_device >= 0 && prior_device != device_index_) {
            cudaSetDevice(prior_device);
        }
        closed_ = true;
    }

    std::string mode_;
    torch::Tensor state_;
    torch::Tensor events_;
    torch::Tensor event_counts_;
    std::int64_t dispatch_temporal_k_ = 0;
    std::int64_t event_capacity_k_ = 0;
    std::uint64_t capability_mask_ = 0;
    std::string profile_id_;
    std::string artifact_sha256_;
    std::string manifest_sha256_;
    std::uint64_t native_plan_ordinal_ = 0;
    int device_index_ = -1;
    std::int64_t batch_size_ = 0;
    std::uint8_t* state_pointer_ = nullptr;
    std::int64_t state_stride_ = 0;
    LifecycleEventRow* event_pointer_ = nullptr;
    std::int64_t event_stride_rows_ = 0;
    std::int32_t* event_count_pointer_ = nullptr;
    SparseLifecycleEventV3* direct_sparse_pointer_ = nullptr;
    std::int64_t direct_sparse_stride_rows_ = 0;
    int direct_sparse_capacity_rows_ = 0;
    std::int32_t* direct_sparse_validation_pointer_ = nullptr;
    std::int64_t* science_action_ids_pointer_ = nullptr;
    std::uint8_t* science_action_values_pointer_ = nullptr;
    std::int64_t* science_birth_child_ids_pointer_ = nullptr;
    std::int64_t* science_birth_parent_a_ids_pointer_ = nullptr;
    std::int64_t* science_birth_parent_b_ids_pointer_ = nullptr;
    std::int32_t* science_birth_mutation_counts_pointer_ = nullptr;
    std::int32_t* science_birth_child_slots_pointer_ = nullptr;
    std::int32_t* science_birth_sites_pointer_ = nullptr;
    std::int32_t* science_birth_counts_pointer_ = nullptr;
    const std::int64_t* science_world_controls_pointer_ = nullptr;
    std::int64_t science_world_control_stride_ = 0;
    std::int64_t* science_agent_metrics_pointer_ = nullptr;
    std::int64_t science_agent_metric_stride_ = 0;
    std::int64_t* science_event_metrics_pointer_ = nullptr;
    std::int64_t science_event_metric_stride_ = 0;
    std::uint8_t* science_agent_birth_provenance_pointer_ = nullptr;
    std::int64_t science_agent_birth_provenance_stride_ = 0;
    std::uint8_t* science_event_birth_provenance_pointer_ = nullptr;
    std::int64_t science_event_birth_provenance_stride_ = 0;
    int kernel_temporal_k_ = 0;
    std::uint64_t kernel_capability_mask_ = 0;
    cudaStream_t construction_stream_ = nullptr;
    cudaStream_t last_launch_stream_ = nullptr;
    cudaGraph_t graph_ = nullptr;
    cudaGraphExec_t graph_exec_ = nullptr;
    cudaGraphNode_t kernel_node_ = nullptr;
    std::int64_t graph_node_count_ = 0;
    cudaGraphNodeType graph_node_type_ = cudaGraphNodeTypeEmpty;
    bool graph_function_verified_ = false;
    std::uint64_t launch_ordinal_ = 0;
    std::int64_t last_start_next_absolute_tick_ = 0;
    std::int64_t last_tick_count_ = 0;
    bool closed_ = false;
};

std::shared_ptr<CudaFullEvolutionExecutionPlanV2> create_execution_plan(
    const std::string& mode,
    torch::Tensor state,
    torch::Tensor events,
    torch::Tensor event_counts,
    const std::int64_t dispatch_temporal_k,
    const std::int64_t event_capacity_k,
    const std::int64_t capability_mask,
    const std::string& profile_id,
    const std::string& artifact_sha256,
    const std::string& manifest_sha256
) {
    return std::make_shared<CudaFullEvolutionExecutionPlanV2>(
        mode, std::move(state), std::move(events), std::move(event_counts),
        dispatch_temporal_k, event_capacity_k, capability_mask, profile_id,
        artifact_sha256, manifest_sha256
    );
}

void check_science_capture_v1(
    const torch::Tensor& state,
    const torch::Tensor& action_ids,
    const torch::Tensor& action_values,
    const torch::Tensor& birth_child_ids,
    const torch::Tensor& birth_parent_a_ids,
    const torch::Tensor& birth_parent_b_ids,
    const torch::Tensor& birth_mutation_counts,
    const torch::Tensor& birth_child_slots,
    const torch::Tensor& birth_sites,
    const torch::Tensor& birth_counts,
    const std::int64_t temporal_k
) {
    TORCH_CHECK(temporal_k >= 1 && temporal_k <= 64,
        "science capture temporal_k must be in [1,64]");
    const auto batch = state.size(0);
    TORCH_CHECK(action_ids.is_cuda() && action_ids.scalar_type() == torch::kInt64 &&
        action_ids.is_contiguous() && action_ids.dim() == 3 &&
        action_ids.size(0) == batch && action_ids.size(1) == temporal_k &&
        action_ids.size(2) == kCapacity,
        "science action ids must be int64[B,K,C] on CUDA");
    TORCH_CHECK(action_values.is_cuda() && action_values.scalar_type() == torch::kUInt8 &&
        action_values.is_contiguous() && action_values.dim() == 3 &&
        action_values.size(0) == batch && action_values.size(1) == temporal_k &&
        action_values.size(2) == kCapacity,
        "science action values must be uint8[B,K,C] on CUDA");
    TORCH_CHECK(birth_child_ids.is_cuda() && birth_child_ids.scalar_type() == torch::kInt64 &&
        birth_child_ids.is_contiguous() && birth_child_ids.dim() == 3 &&
        birth_child_ids.size(0) == batch && birth_child_ids.size(1) == temporal_k &&
        birth_child_ids.size(2) == kScienceBirthCapacity,
        "science birth child ids must be int64[B,K,C/2] on CUDA");
    TORCH_CHECK(birth_parent_a_ids.is_cuda() && birth_parent_a_ids.scalar_type() == torch::kInt64 &&
        birth_parent_a_ids.is_contiguous() && birth_parent_a_ids.dim() == 3 &&
        birth_parent_a_ids.size(0) == batch && birth_parent_a_ids.size(1) == temporal_k &&
        birth_parent_a_ids.size(2) == kScienceBirthCapacity,
        "science birth parent-a ids must be int64[B,K,C/2] on CUDA");
    TORCH_CHECK(birth_parent_b_ids.is_cuda() && birth_parent_b_ids.scalar_type() == torch::kInt64 &&
        birth_parent_b_ids.is_contiguous() && birth_parent_b_ids.dim() == 3 &&
        birth_parent_b_ids.size(0) == batch && birth_parent_b_ids.size(1) == temporal_k &&
        birth_parent_b_ids.size(2) == kScienceBirthCapacity,
        "science birth parent-b ids must be int64[B,K,C/2] on CUDA");
    for (const auto& tensor : {birth_mutation_counts, birth_child_slots, birth_sites}) {
        TORCH_CHECK(tensor.is_cuda() && tensor.scalar_type() == torch::kInt32 &&
            tensor.is_contiguous() && tensor.dim() == 3 &&
            tensor.size(0) == batch && tensor.size(1) == temporal_k &&
            tensor.size(2) == kScienceBirthCapacity,
            "science birth metadata must be int32[B,K,C/2] on CUDA");
    }
    TORCH_CHECK(birth_counts.is_cuda() && birth_counts.scalar_type() == torch::kInt32 &&
        birth_counts.is_contiguous() && birth_counts.dim() == 2 &&
        birth_counts.size(0) == batch && birth_counts.size(1) == temporal_k,
        "science birth counts must be int32[B,K] on CUDA");
    for (const auto& tensor : {
        action_ids, action_values, birth_child_ids, birth_parent_a_ids,
        birth_parent_b_ids, birth_mutation_counts, birth_child_slots,
        birth_sites, birth_counts
    }) {
        TORCH_CHECK(tensor.device() == state.device(),
            "science capture tensor device mismatch");
        TORCH_CHECK(!tensors_overlap(state, tensor),
            "science capture tensor overlaps resident state");
    }
}

void rebuild_derived_caches(torch::Tensor state) {
    check_state(state);
    const c10::cuda::CUDAGuard guard(state.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    rebuild_derived_caches_kernel<<<state.size(0), kThreads, 0, stream>>>(
        state.data_ptr<std::uint8_t>(), state.stride(0)
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void force_test_conditions(torch::Tensor state) {
    check_state(state);
    const c10::cuda::CUDAGuard guard(state.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    force_test_conditions_kernel<<<state.size(0), 1, 0, stream>>>(
        state.data_ptr<std::uint8_t>(), state.stride(0)
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void force_full_capacity_benchmark_fixture_v2(torch::Tensor state) {
    check_state(state);
    const c10::cuda::CUDAGuard guard(state.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    force_full_capacity_benchmark_fixture_kernel<<<state.size(0), 1, 0, stream>>>(
        state.data_ptr<std::uint8_t>(), state.stride(0)
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void add_field(
    py::list& fields,
    const char* name,
    const char* dtype,
    const py::tuple& shape,
    const int offset,
    const int bytes
) {
    py::dict value;
    value["name"] = name;
    value["dtype"] = dtype;
    value["shape"] = py::list(shape);
    value["offset_bytes"] = offset;
    value["bytes"] = bytes;
    fields.append(value);
}

py::dict layout_manifest() {
    py::dict result;
    result["schema_version"] = "eiel_full_evolution_cuda_native_layout_manifest_v2";
    result["alignment_bytes"] = 64;
    result["state_bytes_per_world"] = kStateBytes;
    result["canonical_checkpoint_bytes_per_world"] = 191744;
    result["derived_cache_bytes_per_world"] = 73728;
    result["reference_scratch_bytes_per_world"] = kReferenceScratchBytes;
    result["optimized_kernel_global_scratch_bytes"] = 0;
    py::dict counter_safety;
    counter_safety["max_capability_counter_increment_per_tick"] = 65536;
    counter_safety["max_total_ticks"] = static_cast<std::uint64_t>(140737488355327ULL);
    counter_safety["counter_dtype"] = "signed_int64_projection_of_u64_header";
    result["counter_safety"] = counter_safety;
    py::dict rng_address;
    py::list rng_fields;
    for (const char* name : {
        "rng_key", "rng_semantic_epoch", "world_uid", "absolute_tick",
        "domain", "subject", "operation", "draw"
    }) {
        rng_fields.append(name);
    }
    rng_address["fields"] = rng_fields;
    rng_address["semantic_epoch"] = kRngSemanticEpoch;
    rng_address["generator"] = "philox4x32_10";
    rng_address["composition"] = "arx_folded_full_u64_counter_address_v3";
    rng_address["arithmetic"] = "integer_only";
    rng_address["output_width_bits"] = 32;
    rng_address["site_bundle_domain"] = "site_environment";
    rng_address["site_bundle_subject"] = "canonical_site_index";
    py::list site_bundle_outputs;
    for (const char* name : {
        "patch_carrying", "patch_growth", "wdl_schedule", "jungle_transition"
    }) {
        site_bundle_outputs.append(name);
    }
    rng_address["site_bundle_outputs"] = site_bundle_outputs;
    py::dict child_genome_bundle;
    child_genome_bundle["persistent_subject"] = "child_biological_uid";
    child_genome_bundle["recombination"] =
        "one_philox_block_per_linked_block_parent_uids_in_operation_draw";
    child_genome_bundle["mutation"] =
        "one_philox_block_per_linked_locus_two_allele_outputs";
    child_genome_bundle["forbidden_subject"] = "canonical_agent_slot";
    rng_address["child_genome_bundle"] = child_genome_bundle;
    py::list forbidden_rng_inputs;
    for (const char* name : {
        "agent_slot", "batch_index", "cta", "warp", "lane", "launch_order"
    }) {
        forbidden_rng_inputs.append(name);
    }
    rng_address["forbidden_inputs"] = forbidden_rng_inputs;
    result["rng_address"] = rng_address;
    py::list header_names;
    for (const char* name : {
        "world_uid", "next_absolute_tick", "next_agent_id", "active_count",
        "status", "ticks_completed", "capability_patch_counter",
        "capability_wdl_delayed_hazard_counter", "capability_jungle_counter",
        "capability_eed_counter", "capability_genome_v2_ordinary_birth_counter",
        "capability_genome_v2_mutation_counter",
        "capability_recombination_profile_counter", "rng_key",
        "rng_semantic_epoch", "birth_count", "death_count", "mutation_count",
        "recombination_count", "wdl_due_count", "jungle_active_count",
        "eed_pending_count", "reserved_0", "reserved_1"
    }) {
        header_names.append(name);
    }
    result["header_names"] = header_names;
    py::dict event_row;
    event_row["bytes"] = kEventRowBytes;
    event_row["rows_per_world_per_tick"] = kEventRowsPerTick;
    py::list identity_key;
    identity_key.append("world_uid");
    identity_key.append("absolute_tick");
    identity_key.append("event_id");
    event_row["complete_identity_key"] = identity_key;
    py::dict event_id_codec;
    event_id_codec["kind_bits"] = 16;
    event_id_codec["ordinal_bits"] = 48;
    event_id_codec["subject_ordinal"] = "canonical_agent_slot";
    event_row["event_id_codec"] = event_id_codec;
    py::dict event_kinds;
    event_kinds["none"] = 0;
    event_kinds["death"] = 1;
    event_kinds["birth"] = 2;
    event_row["event_kinds"] = event_kinds;
    py::list event_fields;
    const char* event_field_names[] = {
        "absolute_tick", "event_id", "agent_id", "parent_a_id", "parent_b_id",
        "event_kind", "site_index", "status", "reserved"
    };
    const char* event_field_dtypes[] = {
        "u64", "u64", "u64", "u64", "u64", "u16", "u16", "u16", "u16"
    };
    for (int index = 0; index < 9; ++index) {
        py::dict event_field;
        event_field["name"] = event_field_names[index];
        event_field["dtype"] = event_field_dtypes[index];
        event_fields.append(event_field);
    }
    event_row["fields"] = event_fields;
    py::dict row_index;
    row_index["death"] = "slot";
    row_index["birth"] = "512+slot";
    event_row["row_index"] = row_index;
    py::dict event_status;
    event_status["unused"] = 0;
    event_status["committed"] = 1;
    event_row["status"] = event_status;
    event_row["unused_row"] = "all_zero";
    event_row["reserved_value"] = 0;
    result["lifecycle_event_row"] = event_row;
    py::dict sparse_event;
    sparse_event["wire_abi"] =
        "eiel_full_evolution_cuda_sparse_lifecycle_event_v3";
    sparse_event["bytes"] = kSparseEventRowBytesV3;
    py::list sparse_fields;
    for (const char* name : {
        "agent_id", "parent_a_id", "parent_b_id", "packed_key_and_reserved"
    }) {
        py::dict sparse_field;
        sparse_field["name"] = name;
        sparse_field["dtype"] = "u64";
        sparse_fields.append(sparse_field);
    }
    sparse_event["fields"] = sparse_fields;
    py::dict packed_key;
    py::dict tick_offset;
    tick_offset["bits"] = 6;
    tick_offset["shift"] = 0;
    packed_key["tick_offset"] = tick_offset;
    py::dict sparse_kind;
    sparse_kind["bits"] = 2;
    sparse_kind["shift"] = 6;
    packed_key["event_kind"] = sparse_kind;
    py::dict subject_slot;
    subject_slot["bits"] = 9;
    subject_slot["shift"] = 8;
    packed_key["subject_slot"] = subject_slot;
    py::dict site_index;
    site_index["bits"] = 10;
    site_index["shift"] = 17;
    packed_key["site_index"] = site_index;
    packed_key["used_bits"] = 27;
    packed_key["high_reserved_value"] = 0;
    sparse_event["packed_key"] = packed_key;
    py::list sparse_order;
    for (const char* name : {
        "world_uid", "tick_offset", "death_before_birth", "subject_slot"
    }) {
        sparse_order.append(name);
    }
    sparse_event["canonical_order"] = sparse_order;
    sparse_event["world_offsets"] = "u64[B+1]_zero_origin_exact_terminal_N";
    result["sparse_lifecycle_event_v3"] = sparse_event;
    py::list fields;
    add_field(fields, "scalar_header_counters", "u64", py::make_tuple(24), 0, 192);
    add_field(fields, "occupancy", "u16", py::make_tuple(1024), 192, 2048);
    add_field(fields, "resource", "u16", py::make_tuple(1024), 2240, 2048);
    add_field(fields, "scent", "u16", py::make_tuple(1024), 4288, 2048);
    add_field(fields, "waste", "u16", py::make_tuple(1024), 6336, 2048);
    add_field(fields, "eed_impact", "u16", py::make_tuple(1024), 8384, 2048);
    add_field(fields, "eed_pending", "u16", py::make_tuple(1024), 10432, 2048);
    add_field(fields, "wdl_due", "u64", py::make_tuple(1024), 12480, 8192);
    add_field(fields, "wdl_active", "u8", py::make_tuple(1024), 20672, 1024);
    add_field(fields, "terrain", "u8", py::make_tuple(1024), 21696, 1024);
    add_field(fields, "jungle", "u8", py::make_tuple(1024), 22720, 1024);
    add_field(fields, "position", "u16", py::make_tuple(512), 23744, 1024);
    add_field(fields, "energy", "i16", py::make_tuple(512), 24768, 1024);
    add_field(fields, "health", "i16", py::make_tuple(512), 25792, 1024);
    add_field(fields, "age", "u16", py::make_tuple(512), 26816, 1024);
    add_field(fields, "absolute_birth_tick", "u64", py::make_tuple(512), 27840, 4096);
    add_field(fields, "alive_words", "u32", py::make_tuple(16), 31936, 64);
    add_field(fields, "agent_id", "u64", py::make_tuple(512), 32000, 4096);
    add_field(fields, "parent_a_id", "u64", py::make_tuple(512), 36096, 4096);
    add_field(fields, "parent_b_id", "u64", py::make_tuple(512), 40192, 4096);
    add_field(fields, "policy_alleles", "i8", py::make_tuple(2, 112, 512), 44288, 114688);
    add_field(fields, "trait_alleles", "i16", py::make_tuple(2, 16, 512), 158976, 32768);
    add_field(fields, "expressed_policy_cache", "i32", py::make_tuple(7, 4, 512), 191744, 57344);
    add_field(fields, "expressed_trait_cache", "i16", py::make_tuple(16, 512), 249088, 16384);
    result["fields"] = fields;
    return result;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.doc() = "CUDA-golden Full Evolution V2 raw CUDA vertical slice";
    module.def("abi_version", []() {
        return std::string("eiel_full_evolution_cuda_native_abi_v1");
    });
    module.def("extension_version", []() {
        return std::string("eiel_full_evolution_cuda_native_extension_v2");
    });
    module.def("architecture_id", []() {
        return std::string("EIEL_FULL_EVOLUTION_CUDA_NATIVE_V2");
    });
    module.def("route_id", []() {
        return std::string("tensor_world_cuda_full_evolution_v2");
    });
    module.def("layout_manifest", &layout_manifest);
    py::class_<
        CudaFullEvolutionExecutionPlanV2,
        std::shared_ptr<CudaFullEvolutionExecutionPlanV2>
    >(module, "CudaFullEvolutionExecutionPlanV2", py::module_local())
        .def("launch", &CudaFullEvolutionExecutionPlanV2::launch)
        .def("topology_manifest", &CudaFullEvolutionExecutionPlanV2::topology_manifest)
        .def(
            "last_execution_receipt",
            &CudaFullEvolutionExecutionPlanV2::last_execution_receipt
        )
        .def("close", &CudaFullEvolutionExecutionPlanV2::close)
        .def_property_readonly("closed", &CudaFullEvolutionExecutionPlanV2::closed);
    module.def("create_execution_plan", &create_execution_plan);
    module.def("initialize", &initialize);
    module.def("initialize_science_worlds_v1", &initialize_science_worlds_v1);
    module.def("advance_reference", &advance_reference);
    module.def("advance", &advance);
    module.def("advance_direct_sparse_v3", &advance_direct_sparse_v3);
    module.def("advance_direct_sparse_worlds_v1", &advance_direct_sparse_worlds_v1);
    module.def(
        "advance_direct_sparse_compact_science_worlds_v1",
        &advance_direct_sparse_compact_science_worlds_v1
    );
    module.def("advance_direct_sparse_science_v1", &advance_direct_sparse_science_v1);
    module.def(
        "advance_direct_sparse_science_worlds_v1",
        &advance_direct_sparse_science_worlds_v1
    );
    module.def("compact_events_v3", &compact_events_v3);
    module.def("pack_direct_sparse_events_v3", &pack_direct_sparse_events_v3);
    module.def(
        "pack_direct_sparse_science_events_v1",
        &pack_direct_sparse_science_events_v1
    );
    module.def("rebuild_derived_caches", &rebuild_derived_caches);
    module.def("force_test_conditions", &force_test_conditions);
    module.def(
        "force_full_capacity_benchmark_fixture_v2",
        &force_full_capacity_benchmark_fixture_v2
    );
}
