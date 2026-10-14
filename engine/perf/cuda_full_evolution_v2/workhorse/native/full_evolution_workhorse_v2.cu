#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

namespace py = pybind11;

#ifndef EIEL_WORKHORSE_WIDTH
#error "EIEL_WORKHORSE_WIDTH is required"
#endif
#ifndef EIEL_WORKHORSE_HEIGHT
#error "EIEL_WORKHORSE_HEIGHT is required"
#endif
#ifndef EIEL_WORKHORSE_INITIAL_POPULATION
#error "EIEL_WORKHORSE_INITIAL_POPULATION is required"
#endif
#ifndef EIEL_WORKHORSE_CAPACITY
#error "EIEL_WORKHORSE_CAPACITY is required"
#endif
#ifndef EIEL_WORKHORSE_BACKEND_KIND
#error "EIEL_WORKHORSE_BACKEND_KIND is required"
#endif

namespace {

constexpr int kWidth = EIEL_WORKHORSE_WIDTH;
constexpr int kHeight = EIEL_WORKHORSE_HEIGHT;
constexpr int kSites = kWidth * kHeight;
constexpr int kInitialPopulation = EIEL_WORKHORSE_INITIAL_POPULATION;
constexpr int kCapacity = EIEL_WORKHORSE_CAPACITY;
constexpr int kStateBytes = EIEL_WORKHORSE_STATE_BYTES;
constexpr int kScratchBytes = EIEL_WORKHORSE_SCRATCH_BYTES;
constexpr int kMaximumTemporalK = EIEL_WORKHORSE_MAX_K;
constexpr int kBackendKind = EIEL_WORKHORSE_BACKEND_KIND;
constexpr int kThreads = 256;
constexpr int kWarpSize = 32;
constexpr int kWarpsPerBlock = kThreads / kWarpSize;
constexpr int kGenomeLoci = 32;
constexpr int kEventRowBytes = 48;
constexpr int kEventRowsPerTick = 2 * kCapacity;
constexpr std::uint32_t kEmpty = 0xffffffffu;
constexpr std::uint64_t kEmptyUid = 0xffffffffffffffffull;
constexpr std::uint64_t kRngEpoch = 4;

constexpr int kHeaderOffset = EIEL_STATE_HEADER_OFFSET;
constexpr int kOccupancyOffset = EIEL_STATE_OCCUPANCY_OFFSET;
constexpr int kResourceOffset = EIEL_STATE_RESOURCE_OFFSET;
constexpr int kPatchFieldOffset = EIEL_STATE_PATCH_FIELD_OFFSET;
constexpr int kPatchPendingOffset = EIEL_STATE_PATCH_PENDING_OFFSET;
constexpr int kWdlDelay0Offset = EIEL_STATE_WDL_DELAY_0_OFFSET;
constexpr int kWdlDelay1Offset = EIEL_STATE_WDL_DELAY_1_OFFSET;
constexpr int kWdlActiveOffset = EIEL_STATE_WDL_ACTIVE_OFFSET;
constexpr int kJungleOffset = EIEL_STATE_JUNGLE_OFFSET;
constexpr int kEedFieldOffset = EIEL_STATE_EED_FIELD_OFFSET;
constexpr int kAliveOffset = EIEL_STATE_ALIVE_OFFSET;
constexpr int kActiveSlotsByUidOffset = EIEL_STATE_ACTIVE_SLOTS_BY_UID_OFFSET;
constexpr int kPositionOffset = EIEL_STATE_POSITION_OFFSET;
constexpr int kEnergyOffset = EIEL_STATE_ENERGY_OFFSET;
constexpr int kHealthOffset = EIEL_STATE_HEALTH_OFFSET;
constexpr int kAgeOffset = EIEL_STATE_AGE_OFFSET;
constexpr int kBirthTickOffset = EIEL_STATE_BIRTH_TICK_OFFSET;
constexpr int kAgentIdOffset = EIEL_STATE_AGENT_ID_OFFSET;
constexpr int kParentAIdOffset = EIEL_STATE_PARENT_A_ID_OFFSET;
constexpr int kParentBIdOffset = EIEL_STATE_PARENT_B_ID_OFFSET;
constexpr int kGenomeAOffset = EIEL_STATE_GENOME_A_OFFSET;
constexpr int kGenomeBOffset = EIEL_STATE_GENOME_B_OFFSET;
constexpr int kExpressedOffset = EIEL_STATE_EXPRESSED_OFFSET;

constexpr int kMoveTargetOffset = EIEL_SCRATCH_MOVE_TARGET_OFFSET;
constexpr int kMoveAcceptedOffset = EIEL_SCRATCH_MOVE_ACCEPTED_OFFSET;
constexpr int kMoveWinnerUidOffset = EIEL_SCRATCH_MOVE_WINNER_UID_OFFSET;
constexpr int kBirthWinnerUidOffset = EIEL_SCRATCH_BIRTH_WINNER_UID_OFFSET;
constexpr int kGatherAmountOffset = EIEL_SCRATCH_GATHER_AMOUNT_OFFSET;
constexpr int kMateSlotOffset = EIEL_SCRATCH_MATE_SLOT_OFFSET;
constexpr int kSpawnSiteOffset = EIEL_SCRATCH_SPAWN_SITE_OFFSET;
constexpr int kBirthAcceptedOffset = EIEL_SCRATCH_BIRTH_ACCEPTED_OFFSET;
constexpr int kBirthRankOffset = EIEL_SCRATCH_BIRTH_RANK_OFFSET;
constexpr int kActiveSlotsOffset = EIEL_SCRATCH_ACTIVE_SLOTS_OFFSET;
constexpr int kFreeSlotsOffset = EIEL_SCRATCH_FREE_SLOTS_OFFSET;
constexpr int kTileActiveCountOffset = EIEL_SCRATCH_TILE_ACTIVE_COUNT_OFFSET;
constexpr int kTileFreeCountOffset = EIEL_SCRATCH_TILE_FREE_COUNT_OFFSET;
constexpr int kTileBirthCountOffset = EIEL_SCRATCH_TILE_BIRTH_COUNT_OFFSET;
constexpr int kWorldControlOffset = EIEL_SCRATCH_WORLD_CONTROL_OFFSET;
constexpr int kScienceMutationThresholdControl = 16;
constexpr int kScienceResourceMultiplierQ16Control = 17;
constexpr int kScienceControlsApplied = 18;
constexpr std::uint64_t kDefaultMutationThresholdU16 = 8192;
constexpr std::uint64_t kDefaultResourceMultiplierQ16 = 65536;

constexpr int kSiteTiles = (kSites + kThreads - 1) / kThreads;
constexpr int kAgentTiles = (kCapacity + kThreads - 1) / kThreads;
constexpr int kEnvironmentTiles =
    kSiteTiles > (2 * kAgentTiles) ? kSiteTiles : (2 * kAgentTiles);

static_assert(kSites > 0 && static_cast<std::uint64_t>(kSites) < kEmpty, "site u32 overflow");
static_assert(kCapacity > 0 && static_cast<std::uint64_t>(kCapacity) < kEmpty, "slot u32 overflow");
static_assert(kInitialPopulation <= kCapacity, "P exceeds C");
static_assert(kInitialPopulation <= kSites, "P exceeds surface");
static_assert(kBackendKind >= 0 && kBackendKind <= 2, "unknown backend kind");
static_assert(kThreads % kWarpSize == 0, "CTA must contain complete warps");

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
    kRngEpochWord = 14,
    kBirthCount = 15,
    kDeathCount = 16,
    kMutationCount = 17,
    kRecombinationCount = 18,
    kLastEventCount = 19,
    kFixtureFlags = 23,
};

enum CapabilityBit : std::uint32_t {
    kDisablePatch = 1u << 0,
    kDisableWdl = 1u << 1,
    kDisableJungle = 1u << 2,
    kDisableEed = 1u << 3,
    kDisableBirth = 1u << 4,
    kDisableMutation = 1u << 5,
    kDisableRecombination = 1u << 6,
};

constexpr std::uint64_t kFixtureForceMutation = 1ull << 0;
constexpr std::uint64_t kFixtureBirthCommitConflictAfterDeath = 1ull << 1;

enum RngDomain : std::uint64_t {
    kDomainInitialGenome = 0x21u,
    kDomainMovement = 0x31u,
    kDomainRecombinationA = 0x41u,
    kDomainRecombinationB = 0x42u,
    kDomainMutation = 0x43u,
};

struct LifecycleEventRow {
    std::uint64_t absolute_tick;
    std::uint64_t event_id;
    std::uint64_t agent_id;
    std::uint64_t parent_a_id;
    std::uint64_t parent_b_id;
    std::uint32_t site_index;
    std::uint16_t event_kind;
    std::uint8_t status;
    std::uint8_t reserved;
};
static_assert(sizeof(LifecycleEventRow) == kEventRowBytes, "event ABI drift");

struct PhysiologyCounterDeltasV2 {
    std::uint32_t patch;
    std::uint32_t wdl;
    std::uint32_t jungle;
    std::uint32_t eed;
};

struct BirthCounterDeltasV2 {
    std::uint32_t births;
    std::uint32_t mutations;
    std::uint32_t recombination_births;
};

template <typename T>
__device__ __forceinline__ T* field(std::uint8_t* row, int offset) {
    return reinterpret_cast<T*>(row + offset);
}

template <typename T>
__device__ __forceinline__ const T* field(const std::uint8_t* row, int offset) {
    return reinterpret_cast<const T*>(row + offset);
}

__device__ __forceinline__ std::uint32_t rotate_left32(
    std::uint32_t value, int shift
) {
    return (value << shift) | (value >> (32 - shift));
}

__device__ __forceinline__ uint2 rng_tick_key(
    const std::uint64_t* header, std::uint64_t absolute_tick
) {
    const std::uint64_t rng_key = header[kRngKey];
    const std::uint64_t world_uid = header[kWorldUid];
    const std::uint32_t key_low = static_cast<std::uint32_t>(rng_key);
    const std::uint32_t key_high = static_cast<std::uint32_t>(rng_key >> 32);
    const std::uint32_t uid_low = static_cast<std::uint32_t>(world_uid);
    const std::uint32_t uid_high = static_cast<std::uint32_t>(world_uid >> 32);
    const std::uint32_t tick_low = static_cast<std::uint32_t>(absolute_tick);
    const std::uint32_t tick_high = static_cast<std::uint32_t>(absolute_tick >> 32);
    std::uint32_t first = key_low ^ rotate_left32(key_high, 7) ^
        rotate_left32(uid_low, 13) ^ rotate_left32(uid_high, 23) ^
        rotate_left32(tick_low, 3) ^ rotate_left32(tick_high, 17) ^
        0x9e3779b9u ^ static_cast<std::uint32_t>(kRngEpoch);
    std::uint32_t second = key_high ^ rotate_left32(key_low, 11) ^
        rotate_left32(uid_high, 5) ^ rotate_left32(uid_low, 19) ^
        rotate_left32(tick_high, 29) ^ rotate_left32(tick_low, 9) ^
        0xbb67ae85u;
    first += rotate_left32(second, 15);
    second ^= rotate_left32(first, 21);
    first += rotate_left32(second, 9);
    return make_uint2(first, second);
}

__device__ __forceinline__ uint4 rng_address_counter(
    std::uint64_t domain,
    std::uint64_t subject,
    std::uint64_t operation,
    std::uint64_t draw
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
    constexpr std::uint32_t kMultiplier0 = 0xd2511f53u;
    constexpr std::uint32_t kMultiplier1 = 0xcd9e8d57u;
    constexpr std::uint32_t kWeyl0 = 0x9e3779b9u;
    constexpr std::uint32_t kWeyl1 = 0xbb67ae85u;
#pragma unroll
    for (int round = 0; round < 10; ++round) {
        const std::uint32_t low0 = kMultiplier0 * counter.x;
        const std::uint32_t high0 = __umulhi(kMultiplier0, counter.x);
        const std::uint32_t low1 = kMultiplier1 * counter.z;
        const std::uint32_t high1 = __umulhi(kMultiplier1, counter.z);
        counter = make_uint4(
            high1 ^ counter.y ^ key.x,
            low1,
            high0 ^ counter.w ^ key.y,
            low0
        );
        key.x += kWeyl0;
        key.y += kWeyl1;
    }
    return counter;
}

__device__ __forceinline__ std::uint32_t addressed_rng(
    const std::uint64_t* header,
    std::uint64_t absolute_tick,
    std::uint64_t domain,
    std::uint64_t subject,
    std::uint64_t operation,
    std::uint64_t draw
) {
    return philox4x32_10(
        rng_address_counter(domain, subject, operation, draw),
        rng_tick_key(header, absolute_tick)
    ).x;
}

__device__ __forceinline__ std::uint32_t neighbor_site(
    std::uint32_t site, std::uint32_t direction
) {
    const std::uint32_t x = site % kWidth;
    const std::uint32_t y = site / kWidth;
    if ((direction & 3u) == 0u) return y * kWidth + ((x + 1) % kWidth);
    if ((direction & 3u) == 1u) return y * kWidth + ((x + kWidth - 1) % kWidth);
    if ((direction & 3u) == 2u) return ((y + 1) % kHeight) * kWidth + x;
    return ((y + kHeight - 1) % kHeight) * kWidth + x;
}

__device__ __forceinline__ void atomic_add_u64(std::uint64_t* value, std::uint64_t add) {
    atomicAdd(reinterpret_cast<unsigned long long*>(value),
              static_cast<unsigned long long>(add));
}

__device__ __forceinline__ std::uint32_t warp_sum_u32(std::uint32_t value) {
#pragma unroll
    for (int offset = kWarpSize / 2; offset != 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    return value;
}

__device__ __forceinline__ PhysiologyCounterDeltasV2 warp_sum_physiology(
    PhysiologyCounterDeltasV2 value
) {
#pragma unroll
    for (int offset = kWarpSize / 2; offset != 0; offset >>= 1) {
        value.patch += __shfl_down_sync(0xffffffffu, value.patch, offset);
        value.wdl += __shfl_down_sync(0xffffffffu, value.wdl, offset);
        value.jungle += __shfl_down_sync(0xffffffffu, value.jungle, offset);
        value.eed += __shfl_down_sync(0xffffffffu, value.eed, offset);
    }
    return value;
}

__device__ __forceinline__ BirthCounterDeltasV2 warp_sum_birth(
    BirthCounterDeltasV2 value
) {
#pragma unroll
    for (int offset = kWarpSize / 2; offset != 0; offset >>= 1) {
        value.births += __shfl_down_sync(0xffffffffu, value.births, offset);
        value.mutations += __shfl_down_sync(0xffffffffu, value.mutations, offset);
        value.recombination_births += __shfl_down_sync(
            0xffffffffu, value.recombination_births, offset
        );
    }
    return value;
}

__device__ __forceinline__ void commit_physiology_cta_totals(
    std::uint8_t* row,
    PhysiologyCounterDeltasV2 local,
    PhysiologyCounterDeltasV2* warp_totals
) {
    local = warp_sum_physiology(local);
    const int lane = static_cast<int>(threadIdx.x) & (kWarpSize - 1);
    const int warp = static_cast<int>(threadIdx.x) / kWarpSize;
    if (lane == 0) warp_totals[warp] = local;
    __syncthreads();
    if (threadIdx.x == 0) {
        PhysiologyCounterDeltasV2 total{};
        for (int index = 0; index < kWarpsPerBlock; ++index) {
            total.patch += warp_totals[index].patch;
            total.wdl += warp_totals[index].wdl;
            total.jungle += warp_totals[index].jungle;
            total.eed += warp_totals[index].eed;
        }
        auto* header = field<std::uint64_t>(row, kHeaderOffset);
        if (kBackendKind == 2) {
            header[kPatchCounter] += total.patch;
            header[kWdlCounter] += total.wdl;
            header[kJungleCounter] += total.jungle;
            header[kEedCounter] += total.eed;
        } else {
            if (total.patch) atomic_add_u64(header + kPatchCounter, total.patch);
            if (total.wdl) atomic_add_u64(header + kWdlCounter, total.wdl);
            if (total.jungle) atomic_add_u64(header + kJungleCounter, total.jungle);
            if (total.eed) atomic_add_u64(header + kEedCounter, total.eed);
        }
    }
}

__device__ __forceinline__ std::uint32_t reduce_cta_u32(
    std::uint32_t local, std::uint32_t* warp_totals
) {
    local = warp_sum_u32(local);
    const int lane = static_cast<int>(threadIdx.x) & (kWarpSize - 1);
    const int warp = static_cast<int>(threadIdx.x) / kWarpSize;
    if (lane == 0) warp_totals[warp] = local;
    __syncthreads();
    if (threadIdx.x != 0) return 0;
    std::uint32_t total = 0;
    for (int index = 0; index < kWarpsPerBlock; ++index) {
        total += warp_totals[index];
    }
    return total;
}

__device__ __forceinline__ BirthCounterDeltasV2 reduce_cta_birth(
    BirthCounterDeltasV2 local, BirthCounterDeltasV2* warp_totals
) {
    local = warp_sum_birth(local);
    const int lane = static_cast<int>(threadIdx.x) & (kWarpSize - 1);
    const int warp = static_cast<int>(threadIdx.x) / kWarpSize;
    if (lane == 0) warp_totals[warp] = local;
    __syncthreads();
    BirthCounterDeltasV2 total{};
    if (threadIdx.x == 0) {
        for (int index = 0; index < kWarpsPerBlock; ++index) {
            total.births += warp_totals[index].births;
            total.mutations += warp_totals[index].mutations;
            total.recombination_births +=
                warp_totals[index].recombination_births;
        }
    }
    return total;
}

__device__ __forceinline__ void set_first_status(std::uint8_t* row, std::uint64_t code) {
    auto* status = field<std::uint64_t>(row, kHeaderOffset) + kStatus;
    atomicCAS(
        reinterpret_cast<unsigned long long*>(status),
        0ull,
        static_cast<unsigned long long>(code)
    );
}

__device__ __forceinline__ void clear_event(LifecycleEventRow* row) {
    row->absolute_tick = 0;
    row->event_id = 0;
    row->agent_id = 0;
    row->parent_a_id = 0;
    row->parent_b_id = 0;
    row->site_index = 0;
    row->event_kind = 0;
    row->status = 0;
    row->reserved = 0;
}

__device__ __forceinline__ void invalidate_event_status(
    LifecycleEventRow* row
) {
    // Active writers publish status last. Performance backends invalidate the
    // single resident 2*C page once at the start of a timed dispatch. Exact
    // inactive-row zeroing remains an ACK-tail operation.
    row->status = 0;
}

__device__ __forceinline__ void write_event(
    LifecycleEventRow* row,
    std::uint64_t tick,
    std::uint16_t kind,
    std::uint32_t slot,
    std::uint32_t site,
    std::uint64_t agent_id,
    std::uint64_t parent_a,
    std::uint64_t parent_b
) {
    row->absolute_tick = tick;
    // Full event key is (world_uid, absolute_tick, event_id); this field is the
    // transparent injective per-tick codec required by the event ABI.
    row->event_id = (static_cast<std::uint64_t>(kind) << 48) | slot;
    row->agent_id = agent_id;
    row->parent_a_id = parent_a;
    row->parent_b_id = parent_b;
    row->site_index = site;
    row->event_kind = kind;
    row->reserved = 0;
    __threadfence();
    row->status = 1;
}

__global__ void canonicalize_inactive_events_v2(
    LifecycleEventRow* events, std::uint64_t row_count
) {
    const std::uint64_t row_index =
        static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row_index < row_count && events[row_index].status == 0) {
        clear_event(events + row_index);
    }
}

__global__ void initialize_worlds_v2(
    std::uint8_t* state,
    std::uint8_t* scratch,
    const std::int64_t* exact_world_uids,
    const std::int64_t* exact_rng_keys,
    std::uint64_t world_uid_base,
    std::uint64_t start_tick,
    std::uint64_t rng_key,
    bool use_exact_world_identity
) {
    const int world = blockIdx.y;
    std::uint8_t* row = state + static_cast<std::int64_t>(world) * kStateBytes;
    std::uint8_t* work = scratch + static_cast<std::int64_t>(world) * kScratchBytes;
    auto* header = field<std::uint64_t>(row, kHeaderOffset);
    auto* occupancy = field<std::uint32_t>(row, kOccupancyOffset);
    auto* resource = field<std::uint16_t>(row, kResourceOffset);
    auto* patch = field<std::uint16_t>(row, kPatchFieldOffset);
    auto* patch_pending = field<std::uint16_t>(row, kPatchPendingOffset);
    auto* jungle = field<std::uint8_t>(row, kJungleOffset);
    auto* alive = field<std::uint8_t>(row, kAliveOffset);
    auto* position = field<std::uint32_t>(row, kPositionOffset);
    auto* active_slots_by_uid = field<std::uint32_t>(row, kActiveSlotsByUidOffset);
    auto* energy = field<std::int32_t>(row, kEnergyOffset);
    auto* health = field<std::int32_t>(row, kHealthOffset);
    auto* age = field<std::uint32_t>(row, kAgeOffset);
    auto* birth_tick = field<std::uint64_t>(row, kBirthTickOffset);
    auto* agent_id = field<std::uint64_t>(row, kAgentIdOffset);
    auto* parent_a = field<std::uint64_t>(row, kParentAIdOffset);
    auto* parent_b = field<std::uint64_t>(row, kParentBIdOffset);
    auto* genome_a = field<std::uint8_t>(row, kGenomeAOffset);
    auto* genome_b = field<std::uint8_t>(row, kGenomeBOffset);
    auto* expressed = field<std::uint8_t>(row, kExpressedOffset);
    if (threadIdx.x == 0) {
        header[kWorldUid] = use_exact_world_identity
            ? static_cast<std::uint64_t>(exact_world_uids[world])
            : world_uid_base + world;
        header[kNextAbsoluteTick] = start_tick;
        header[kNextAgentId] = static_cast<std::uint64_t>(kInitialPopulation + 1);
        header[kActiveCount] = kInitialPopulation;
        header[kStatus] = 0;
        header[kTicksCompleted] = 0;
        header[kRngKey] = use_exact_world_identity
            ? static_cast<std::uint64_t>(exact_rng_keys[world])
            : rng_key;
        header[kRngEpochWord] = kRngEpoch;
    }
    __syncthreads();
    for (int site = threadIdx.x; site < kSites; site += blockDim.x) {
        occupancy[site] = kEmpty;
        resource[site] = static_cast<std::uint16_t>(96 + (site % 31));
        patch[site] = static_cast<std::uint16_t>(8 + (site % 17));
        jungle[site] = static_cast<std::uint8_t>(
            (static_cast<std::uint64_t>(site) * 13 + header[kWorldUid]) & 0x7f
        );
    }
    __syncthreads();
    for (int slot = threadIdx.x; slot < kCapacity; slot += blockDim.x) {
        const bool live = slot < kInitialPopulation;
        const std::uint32_t site = live
            ? static_cast<std::uint32_t>((static_cast<std::uint64_t>(slot) * kSites) /
                                         kInitialPopulation)
            : kEmpty;
        alive[slot] = live ? 1 : 0;
        active_slots_by_uid[slot] = live ? static_cast<std::uint32_t>(slot) : kEmpty;
        position[slot] = site;
        energy[slot] = live ? 160 + (slot & 31) : 0;
        health[slot] = live ? 256 : 0;
        age[slot] = 0;
        birth_tick[slot] = live ? start_tick : 0;
        agent_id[slot] = live ? static_cast<std::uint64_t>(slot + 1) : 0;
        parent_a[slot] = 0;
        parent_b[slot] = 0;
        if (live) occupancy[site] = static_cast<std::uint32_t>(slot);
        for (int locus = 0; locus < kGenomeLoci; ++locus) {
            const int index = slot * kGenomeLoci + locus;
            const std::uint8_t first = static_cast<std::uint8_t>(addressed_rng(
                header, start_tick, kDomainInitialGenome,
                static_cast<std::uint64_t>(slot + 1), locus, 0
            ));
            const std::uint8_t second = static_cast<std::uint8_t>(addressed_rng(
                header, start_tick, kDomainInitialGenome,
                static_cast<std::uint64_t>(slot + 1), locus, 1
            ));
            genome_a[index] = live ? first : 0;
            genome_b[index] = live ? second : 0;
            expressed[index] = live
                ? static_cast<std::uint8_t>((static_cast<unsigned>(first) + second) / 2)
                : 0;
        }
    }
    for (int index = threadIdx.x; index < kScratchBytes; index += blockDim.x) {
        work[index] = 0;
    }
}

__device__ __forceinline__ void environment_site(
    std::uint8_t* row, std::uint8_t* work, int site, std::uint32_t disable_mask
) {
    auto* resource = field<std::uint16_t>(row, kResourceOffset);
    auto* patch = field<std::uint16_t>(row, kPatchFieldOffset);
    auto* patch_pending = field<std::uint16_t>(row, kPatchPendingOffset);
    auto* delay0 = field<std::uint16_t>(row, kWdlDelay0Offset);
    auto* delay1 = field<std::uint16_t>(row, kWdlDelay1Offset);
    auto* active = field<std::uint16_t>(row, kWdlActiveOffset);
    auto* eed_field = field<std::uint16_t>(row, kEedFieldOffset);
    auto* move_winner = field<std::uint64_t>(work, kMoveWinnerUidOffset);
    auto* birth_winner = field<std::uint64_t>(work, kBirthWinnerUidOffset);
    const auto* science_control = field<std::uint64_t>(work, kWorldControlOffset);
    const std::uint64_t resource_multiplier_q16 =
        science_control[kScienceControlsApplied] == 0
            ? kDefaultResourceMultiplierQ16
            : science_control[kScienceResourceMultiplierQ16Control];
    if (!(disable_mask & kDisablePatch)) {
        patch[site] = static_cast<std::uint16_t>(min(4095u,
            static_cast<unsigned>(patch[site]) + patch_pending[site]));
        patch_pending[site] = 0;
        const std::uint64_t unscaled_growth = 1u + patch[site] / 64u;
        const std::uint64_t scaled_growth =
            (unscaled_growth * resource_multiplier_q16) >> 16;
        resource[site] = static_cast<std::uint16_t>(min(
            static_cast<std::uint64_t>(4095),
            static_cast<std::uint64_t>(resource[site]) + scaled_growth));
    }
    if (!(disable_mask & kDisableWdl)) {
        active[site] = delay0[site];
        delay0[site] = delay1[site];
        delay1[site] = 0;
    } else {
        active[site] = 0;
    }
    eed_field[site] = static_cast<std::uint16_t>(eed_field[site] - eed_field[site] / 16u);
    move_winner[site] = kEmptyUid;
    birth_winner[site] = kEmptyUid;
}

__device__ __forceinline__ void clear_agent_scratch(std::uint8_t* work, int slot) {
    field<std::uint32_t>(work, kMoveTargetOffset)[slot] = kEmpty;
    field<std::uint8_t>(work, kMoveAcceptedOffset)[slot] = 0;
    field<std::uint16_t>(work, kGatherAmountOffset)[slot] = 0;
    field<std::uint32_t>(work, kMateSlotOffset)[slot] = kEmpty;
    field<std::uint32_t>(work, kSpawnSiteOffset)[slot] = kEmpty;
    field<std::uint8_t>(work, kBirthAcceptedOffset)[slot] = 0;
    field<std::uint32_t>(work, kBirthRankOffset)[slot] = kEmpty;
}

__global__ void environment_phase_v2(
    std::uint8_t* state,
    std::uint8_t* scratch,
    LifecycleEventRow* events,
    std::int64_t* event_counts,
    int local_tick,
    int event_k_stride,
    std::uint32_t disable_mask
) {
    const int world = blockIdx.y;
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    std::uint8_t* row = state + static_cast<std::int64_t>(world) * kStateBytes;
    std::uint8_t* work = scratch + static_cast<std::int64_t>(world) * kScratchBytes;
    if (kBackendKind != 0 && local_tick == 0 && index < kEventRowsPerTick) {
        LifecycleEventRow* dispatch_events = events +
            static_cast<std::int64_t>(world) * kEventRowsPerTick;
        invalidate_event_status(dispatch_events + index);
    }
    if (field<std::uint64_t>(row, kHeaderOffset)[kStatus] != 0) return;
    if (index < kSites) environment_site(row, work, index, disable_mask);
    if (index < kCapacity) clear_agent_scratch(work, index);
    if (kBackendKind == 0 && index < kEventRowsPerTick) {
        LifecycleEventRow* base = events +
            static_cast<std::int64_t>(world) * kEventRowsPerTick;
        clear_event(base + index);
    }
}

__device__ __forceinline__ void propose_move(
    std::uint8_t* row,
    std::uint8_t* work,
    int slot,
    std::uint64_t tick
) {
    auto* alive = field<std::uint8_t>(row, kAliveOffset);
    if (!alive[slot]) return;
    auto* position = field<std::uint32_t>(row, kPositionOffset);
    auto* agent_id = field<std::uint64_t>(row, kAgentIdOffset);
    auto* header = field<std::uint64_t>(row, kHeaderOffset);
    auto* move_target = field<std::uint32_t>(work, kMoveTargetOffset);
    auto* move_winner = field<std::uint64_t>(work, kMoveWinnerUidOffset);
    const std::uint32_t direction = addressed_rng(
        header, tick, kDomainMovement, agent_id[slot], 0, 0
    ) & 3u;
    const std::uint32_t target = neighbor_site(position[slot], direction);
    move_target[slot] = target;
    atomicMin(reinterpret_cast<unsigned long long*>(move_winner + target),
              static_cast<unsigned long long>(agent_id[slot]));
}

__device__ __forceinline__ std::uint32_t slot_for_ordinal(
    const std::uint8_t* row, std::uint32_t ordinal
) {
    const auto* header = field<std::uint64_t>(row, kHeaderOffset);
    if (ordinal >= header[kActiveCount]) return kEmpty;
    return field<std::uint32_t>(row, kActiveSlotsByUidOffset)[ordinal];
}

__global__ void agent_intent_phase_v2(
    std::uint8_t* state, std::uint8_t* scratch, std::uint64_t tick
) {
    const int world = blockIdx.y;
    const int ordinal = blockIdx.x * blockDim.x + threadIdx.x;
    if (ordinal >= kCapacity) return;
    std::uint8_t* row = state + static_cast<std::int64_t>(world) * kStateBytes;
    if (field<std::uint64_t>(row, kHeaderOffset)[kStatus] != 0) return;
    const std::uint32_t slot = slot_for_ordinal(row, ordinal);
    if (slot == kEmpty || slot >= static_cast<std::uint32_t>(kCapacity)) return;
    tick = field<std::uint64_t>(row, kHeaderOffset)[kNextAbsoluteTick];
    propose_move(
        row,
        scratch + static_cast<std::int64_t>(world) * kScratchBytes,
        static_cast<int>(slot),
        tick
    );
}

__device__ __forceinline__ void decide_move(
    std::uint8_t* row, std::uint8_t* work, int slot
) {
    auto* alive = field<std::uint8_t>(row, kAliveOffset);
    if (!alive[slot]) return;
    const auto* occupancy = field<std::uint32_t>(row, kOccupancyOffset);
    const auto* agent_id = field<std::uint64_t>(row, kAgentIdOffset);
    const auto* target = field<std::uint32_t>(work, kMoveTargetOffset);
    const auto* winner = field<std::uint64_t>(work, kMoveWinnerUidOffset);
    field<std::uint8_t>(work, kMoveAcceptedOffset)[slot] =
        target[slot] != kEmpty && occupancy[target[slot]] == kEmpty &&
        winner[target[slot]] == agent_id[slot];
}

__global__ void movement_accept_phase_v2(std::uint8_t* state, std::uint8_t* scratch) {
    const int world = blockIdx.y;
    const int ordinal = blockIdx.x * blockDim.x + threadIdx.x;
    if (ordinal < kCapacity) {
        std::uint8_t* row = state + static_cast<std::int64_t>(world) * kStateBytes;
        if (field<std::uint64_t>(row, kHeaderOffset)[kStatus] != 0) return;
        const std::uint32_t slot = slot_for_ordinal(row, ordinal);
        if (slot == kEmpty || slot >= static_cast<std::uint32_t>(kCapacity)) return;
        decide_move(
            row,
            scratch + static_cast<std::int64_t>(world) * kScratchBytes,
            static_cast<int>(slot)
        );
    }
}

__device__ __forceinline__ void release_move(
    std::uint8_t* row, std::uint8_t* work, int slot
) {
    if (!field<std::uint8_t>(row, kAliveOffset)[slot] ||
        !field<std::uint8_t>(work, kMoveAcceptedOffset)[slot]) return;
    auto* occupancy = field<std::uint32_t>(row, kOccupancyOffset);
    const std::uint32_t old_site = field<std::uint32_t>(row, kPositionOffset)[slot];
    if (atomicCAS(occupancy + old_site, static_cast<std::uint32_t>(slot), kEmpty) !=
        static_cast<std::uint32_t>(slot)) {
        set_first_status(row, 1);
    }
}

__global__ void movement_release_phase_v2(std::uint8_t* state, std::uint8_t* scratch) {
    const int world = blockIdx.y;
    const int ordinal = blockIdx.x * blockDim.x + threadIdx.x;
    if (ordinal < kCapacity) {
        std::uint8_t* row = state + static_cast<std::int64_t>(world) * kStateBytes;
        if (field<std::uint64_t>(row, kHeaderOffset)[kStatus] != 0) return;
        const std::uint32_t slot = slot_for_ordinal(row, ordinal);
        if (slot == kEmpty || slot >= static_cast<std::uint32_t>(kCapacity)) return;
        release_move(
            row,
            scratch + static_cast<std::int64_t>(world) * kScratchBytes,
            static_cast<int>(slot)
        );
    }
}

__device__ __forceinline__ void commit_move(
    std::uint8_t* row, std::uint8_t* work, int slot
) {
    if (!field<std::uint8_t>(row, kAliveOffset)[slot] ||
        !field<std::uint8_t>(work, kMoveAcceptedOffset)[slot]) return;
    auto* occupancy = field<std::uint32_t>(row, kOccupancyOffset);
    auto* position = field<std::uint32_t>(row, kPositionOffset);
    const std::uint32_t target = field<std::uint32_t>(work, kMoveTargetOffset)[slot];
    if (atomicCAS(occupancy + target, kEmpty, static_cast<std::uint32_t>(slot)) == kEmpty) {
        position[slot] = target;
    } else {
        set_first_status(row, 2);
    }
}

__global__ void movement_commit_phase_v2(std::uint8_t* state, std::uint8_t* scratch) {
    const int world = blockIdx.y;
    const int ordinal = blockIdx.x * blockDim.x + threadIdx.x;
    if (ordinal < kCapacity) {
        std::uint8_t* row = state + static_cast<std::int64_t>(world) * kStateBytes;
        if (field<std::uint64_t>(row, kHeaderOffset)[kStatus] != 0) return;
        const std::uint32_t slot = slot_for_ordinal(row, ordinal);
        if (slot == kEmpty || slot >= static_cast<std::uint32_t>(kCapacity)) return;
        commit_move(
            row,
            scratch + static_cast<std::int64_t>(world) * kScratchBytes,
            static_cast<int>(slot)
        );
    }
}

__device__ __forceinline__ PhysiologyCounterDeltasV2 interact_and_physiology(
    std::uint8_t* row,
    std::uint8_t* work,
    int slot,
    std::uint32_t disable_mask
) {
    PhysiologyCounterDeltasV2 deltas{};
    auto* alive = field<std::uint8_t>(row, kAliveOffset);
    if (!alive[slot]) return deltas;
    const std::uint32_t site = field<std::uint32_t>(row, kPositionOffset)[slot];
    auto* resource = field<std::uint16_t>(row, kResourceOffset);
    auto* patch_pending = field<std::uint16_t>(row, kPatchPendingOffset);
    auto* delay1 = field<std::uint16_t>(row, kWdlDelay1Offset);
    auto* active = field<std::uint16_t>(row, kWdlActiveOffset);
    auto* jungle = field<std::uint8_t>(row, kJungleOffset);
    auto* eed_field = field<std::uint16_t>(row, kEedFieldOffset);
    auto* energy = field<std::int32_t>(row, kEnergyOffset);
    auto* health = field<std::int32_t>(row, kHealthOffset);
    auto* expressed = field<std::uint8_t>(row, kExpressedOffset);
    const int genome = slot * kGenomeLoci;
    const int eed_bonus = (disable_mask & kDisableEed) ? 0 : expressed[genome + 31] / 64;
    const int wanted = 1 + eed_bonus;
    const int gathered = min(wanted, static_cast<int>(resource[site]));
    resource[site] = static_cast<std::uint16_t>(resource[site] - gathered);
    field<std::uint16_t>(work, kGatherAmountOffset)[slot] = gathered;
    energy[slot] += gathered * 6;
    if (!(disable_mask & kDisablePatch)) {
        patch_pending[site] = static_cast<std::uint16_t>(min(4095u,
            static_cast<unsigned>(patch_pending[site]) + 1u + expressed[genome] / 64u));
    }
    if (!(disable_mask & kDisableWdl)) {
        delay1[site] = static_cast<std::uint16_t>(min(4095u,
            static_cast<unsigned>(delay1[site]) + 1u + expressed[genome + 3] / 64u));
    }
    if (!(disable_mask & kDisableEed)) {
        eed_field[site] = static_cast<std::uint16_t>(min(4095u,
            static_cast<unsigned>(eed_field[site]) + 1u + eed_bonus));
    }
    int cost = 1;
    if (!(disable_mask & kDisableJungle)) cost += jungle[site] / 32;
    if (!(disable_mask & kDisableWdl)) cost += active[site] / 8;
    if (!(disable_mask & kDisableJungle)) health[slot] -= jungle[site] / 64;
    if (!(disable_mask & kDisableWdl)) health[slot] -= active[site] / 8;
    energy[slot] -= cost;
    if (energy[slot] < 0) {
        health[slot] += energy[slot];
        energy[slot] = 0;
    }
    field<std::uint32_t>(row, kAgeOffset)[slot] += 1;
    deltas.patch = !(disable_mask & kDisablePatch);
    deltas.wdl = !(disable_mask & kDisableWdl) && active[site];
    deltas.jungle = !(disable_mask & kDisableJungle) && jungle[site];
    deltas.eed = !(disable_mask & kDisableEed) && eed_bonus;
    return deltas;
}

__global__ void interaction_physiology_phase_v2(
    std::uint8_t* state, std::uint8_t* scratch, std::uint32_t disable_mask
) {
    __shared__ PhysiologyCounterDeltasV2 warp_totals[kWarpsPerBlock];
    const int world = blockIdx.y;
    const int ordinal = blockIdx.x * blockDim.x + threadIdx.x;
    std::uint8_t* row = state + static_cast<std::int64_t>(world) * kStateBytes;
    PhysiologyCounterDeltasV2 deltas{};
    if (ordinal < kCapacity &&
        field<std::uint64_t>(row, kHeaderOffset)[kStatus] == 0) {
        const std::uint32_t slot = slot_for_ordinal(row, ordinal);
        if (slot != kEmpty && slot < static_cast<std::uint32_t>(kCapacity)) {
            deltas = interact_and_physiology(
                row,
                scratch + static_cast<std::int64_t>(world) * kScratchBytes,
                static_cast<int>(slot),
                disable_mask
            );
        }
    }
    commit_physiology_cta_totals(row, deltas, warp_totals);
}

__device__ __forceinline__ bool commit_death(
    std::uint8_t* row,
    LifecycleEventRow* event_base,
    int slot,
    std::uint64_t tick
) {
    auto* alive = field<std::uint8_t>(row, kAliveOffset);
    if (!alive[slot]) return false;
    auto* energy = field<std::int32_t>(row, kEnergyOffset);
    auto* health = field<std::int32_t>(row, kHealthOffset);
    auto* age = field<std::uint32_t>(row, kAgeOffset);
    if (health[slot] > 0 && energy[slot] > 0 && age[slot] < 10000) return false;
    const std::uint32_t site = field<std::uint32_t>(row, kPositionOffset)[slot];
    auto* occupancy = field<std::uint32_t>(row, kOccupancyOffset);
    atomicCAS(occupancy + site, static_cast<std::uint32_t>(slot), kEmpty);
    alive[slot] = 0;
    if (event_base != nullptr) {
        write_event(
            event_base + slot,
            tick,
            1,
            slot,
            site,
            field<std::uint64_t>(row, kAgentIdOffset)[slot],
            field<std::uint64_t>(row, kParentAIdOffset)[slot],
            field<std::uint64_t>(row, kParentBIdOffset)[slot]
        );
    }
    return true;
}

__global__ void death_phase_v2(
    std::uint8_t* state,
    std::uint8_t* scratch,
    LifecycleEventRow* events,
    std::int64_t* event_counts,
    int local_tick,
    int event_k_stride,
    std::uint64_t tick
) {
    __shared__ std::uint32_t warp_totals[kWarpsPerBlock];
    const int world = blockIdx.y;
    const int ordinal = blockIdx.x * blockDim.x + threadIdx.x;
    std::uint8_t* row = state + static_cast<std::int64_t>(world) * kStateBytes;
    std::uint8_t* work = scratch + static_cast<std::int64_t>(world) * kScratchBytes;
    std::uint32_t committed = 0;
    auto* event_base = kBackendKind == 0
        ? events + static_cast<std::int64_t>(world) * kEventRowsPerTick
        : nullptr;
    if (ordinal < kCapacity &&
        field<std::uint64_t>(row, kHeaderOffset)[kStatus] == 0) {
        const std::uint32_t slot = slot_for_ordinal(row, ordinal);
        if (slot != kEmpty && slot < static_cast<std::uint32_t>(kCapacity)) {
            tick = field<std::uint64_t>(row, kHeaderOffset)[kNextAbsoluteTick];
            committed = commit_death(
                row, event_base, static_cast<int>(slot), tick
            );
            field<std::uint8_t>(work, kMoveAcceptedOffset)[slot] =
                static_cast<std::uint8_t>(committed);
        }
    }
    const std::uint32_t cta_total = reduce_cta_u32(committed, warp_totals);
    if (threadIdx.x == 0 && cta_total) {
        if (kBackendKind == 0) {
            atomic_add_u64(
                reinterpret_cast<std::uint64_t*>(event_counts + world),
                cta_total
            );
        }
        atomic_add_u64(
            field<std::uint64_t>(row, kHeaderOffset) + kDeathCount, cta_total
        );
    }
}

template <int EventKind>
__device__ void append_sparse_page_events_v3(
    const std::uint8_t* row,
    const std::uint8_t* work,
    LifecycleEventRow* events,
    std::int64_t* event_counts,
    std::uint64_t tick,
    std::uint32_t* scan,
    std::int64_t* tile_base
) {
    const int world = blockIdx.y;
    const auto* death_flags = field<std::uint8_t>(work, kMoveAcceptedOffset);
    const auto* alive = field<std::uint8_t>(row, kAliveOffset);
    const auto* birth_tick = field<std::uint64_t>(row, kBirthTickOffset);
    const auto* position = field<std::uint32_t>(row, kPositionOffset);
    const auto* agent_id = field<std::uint64_t>(row, kAgentIdOffset);
    const auto* parent_a_id = field<std::uint64_t>(row, kParentAIdOffset);
    const auto* parent_b_id = field<std::uint64_t>(row, kParentBIdOffset);
    LifecycleEventRow* page = events +
        static_cast<std::int64_t>(world) * kEventRowsPerTick;
    for (int slot_base = 0; slot_base < kCapacity; slot_base += blockDim.x) {
        const int slot = slot_base + threadIdx.x;
        const bool committed = slot < kCapacity && (
            EventKind == 1
                ? death_flags[slot] != 0
                : alive[slot] != 0 && birth_tick[slot] == tick + 1
        );
        scan[threadIdx.x] = committed ? 1u : 0u;
        __syncthreads();
        for (int offset = 1; offset < blockDim.x; offset <<= 1) {
            const std::uint32_t add =
                threadIdx.x >= offset ? scan[threadIdx.x - offset] : 0u;
            __syncthreads();
            scan[threadIdx.x] += add;
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            *tile_base = event_counts[world];
        }
        __syncthreads();
        if (committed) {
            const std::int64_t output_index =
                *tile_base + static_cast<std::int64_t>(scan[threadIdx.x] - 1u);
            if (output_index < kEventRowsPerTick) {
                write_event(
                    page + output_index,
                    tick,
                    static_cast<std::uint16_t>(EventKind),
                    static_cast<std::uint32_t>(slot),
                    position[slot],
                    agent_id[slot],
                    parent_a_id[slot],
                    parent_b_id[slot]
                );
            }
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            event_counts[world] = *tile_base + scan[kThreads - 1];
        }
        __syncthreads();
    }
}

template <int EventKind>
__global__ void compact_phase_events_v3(
    const std::uint8_t* state,
    const std::uint8_t* scratch,
    LifecycleEventRow* events,
    std::int64_t* event_counts
) {
    __shared__ std::uint32_t scan[kThreads];
    __shared__ std::int64_t tile_base;
    const int world = blockIdx.y;
    const std::uint8_t* row = state +
        static_cast<std::int64_t>(world) * kStateBytes;
    if (field<std::uint64_t>(row, kHeaderOffset)[kStatus] != 0) return;
    const std::uint8_t* work = scratch +
        static_cast<std::int64_t>(world) * kScratchBytes;
    const std::uint64_t tick =
        field<std::uint64_t>(row, kHeaderOffset)[kNextAbsoluteTick];
    append_sparse_page_events_v3<EventKind>(
        row, work, events, event_counts, tick, scan, &tile_base
    );
}

__device__ __forceinline__ void propose_birth_from_survivor_order(
    std::uint8_t* row,
    std::uint8_t* work,
    std::uint32_t survivor_ordinal,
    std::uint32_t disable_mask
) {
    if (disable_mask & kDisableBirth) return;
    const auto* control = field<std::uint64_t>(work, kWorldControlOffset);
    if ((survivor_ordinal & 1u) || survivor_ordinal + 1 >= control[0]) return;
    const auto* survivor_order = field<std::uint32_t>(work, kActiveSlotsOffset);
    const std::uint32_t slot = survivor_order[survivor_ordinal];
    const std::uint32_t mate = survivor_order[survivor_ordinal + 1];
    if (slot == kEmpty || mate == kEmpty || slot >= static_cast<std::uint32_t>(kCapacity) ||
        mate >= static_cast<std::uint32_t>(kCapacity)) return;
    auto* alive = field<std::uint8_t>(row, kAliveOffset);
    if (!alive[slot] || !alive[mate]) return;
    auto* energy = field<std::int32_t>(row, kEnergyOffset);
    if (energy[slot] < 256 || energy[mate] < 256) return;
    auto* position = field<std::uint32_t>(row, kPositionOffset);
    const std::uint32_t spawn = neighbor_site(position[slot], 2);
    if (field<std::uint32_t>(row, kOccupancyOffset)[spawn] != kEmpty) return;
    field<std::uint32_t>(work, kMateSlotOffset)[slot] = mate;
    field<std::uint32_t>(work, kSpawnSiteOffset)[slot] = spawn;
    atomicMin(
        reinterpret_cast<unsigned long long*>(
            field<std::uint64_t>(work, kBirthWinnerUidOffset) + spawn
        ),
        static_cast<unsigned long long>(field<std::uint64_t>(row, kAgentIdOffset)[slot])
    );
}

__device__ __forceinline__ bool accepted_birth_candidate(
    std::uint8_t* row, std::uint8_t* work, int slot
);

__global__ void mate_spawn_phase_v2(
    std::uint8_t* state, std::uint8_t* scratch, std::uint32_t disable_mask
) {
    const int world = blockIdx.y;
    std::uint8_t* row = state + static_cast<std::int64_t>(world) * kStateBytes;
    std::uint8_t* work = scratch + static_cast<std::int64_t>(world) * kScratchBytes;
    if (field<std::uint64_t>(row, kHeaderOffset)[kStatus] != 0) return;
    auto* control = field<std::uint64_t>(work, kWorldControlOffset);
    const std::uint32_t survivor_total = static_cast<std::uint32_t>(control[0]);
    for (std::uint32_t survivor_ordinal = threadIdx.x;
         survivor_ordinal < survivor_total;
         survivor_ordinal += blockDim.x) {
        propose_birth_from_survivor_order(
            row, work, survivor_ordinal, disable_mask
        );
    }
    __syncthreads();

    // Rank accepted first parents in compacted biological-UID order.  A
    // block-tiled scan avoids both raw-slot ordering and nondeterministic
    // atomic allocation while retaining one CTA per world.
    __shared__ std::uint32_t accepted_scan[kThreads];
    __shared__ std::uint32_t accepted_base;
    if (threadIdx.x == 0) accepted_base = 0;
    __syncthreads();
    const auto* survivor_order = field<std::uint32_t>(work, kActiveSlotsOffset);
    for (std::uint32_t tile = 0; tile < survivor_total; tile += blockDim.x) {
        const std::uint32_t survivor_ordinal = tile + threadIdx.x;
        const std::uint32_t slot = survivor_ordinal < survivor_total
            ? survivor_order[survivor_ordinal] : kEmpty;
        const bool accepted = survivor_ordinal < survivor_total &&
            (survivor_ordinal & 1u) == 0u && slot < kCapacity &&
            accepted_birth_candidate(row, work, static_cast<int>(slot));
        accepted_scan[threadIdx.x] = accepted ? 1u : 0u;
        __syncthreads();
        for (int offset = 1; offset < blockDim.x; offset <<= 1) {
            const std::uint32_t add = threadIdx.x >= offset
                ? accepted_scan[threadIdx.x - offset] : 0u;
            __syncthreads();
            accepted_scan[threadIdx.x] += add;
            __syncthreads();
        }
        const std::uint32_t tile_base = accepted_base;
        if (accepted) {
            field<std::uint32_t>(work, kBirthRankOffset)[slot] =
                tile_base + accepted_scan[threadIdx.x] - 1u;
            field<std::uint8_t>(work, kBirthAcceptedOffset)[slot] = 1;
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            accepted_base += accepted_scan[blockDim.x - 1];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        control[2] = min(
            static_cast<std::uint64_t>(accepted_base), control[1]
        );
    }
}

__device__ __forceinline__ bool accepted_birth_candidate(
    std::uint8_t* row, std::uint8_t* work, int slot
) {
    if (!field<std::uint8_t>(row, kAliveOffset)[slot]) return false;
    const std::uint32_t spawn = field<std::uint32_t>(work, kSpawnSiteOffset)[slot];
    if (spawn == kEmpty) return false;
    return field<std::uint64_t>(work, kBirthWinnerUidOffset)[spawn] ==
        field<std::uint64_t>(row, kAgentIdOffset)[slot];
}

__global__ void rank_tiles_phase_v2(std::uint8_t* state, std::uint8_t* scratch) {
    __shared__ std::uint32_t active_scan[kThreads];
    __shared__ std::uint32_t free_scan[kThreads];
    const int world = blockIdx.y;
    const int ordinal = blockIdx.x * blockDim.x + threadIdx.x;
    std::uint8_t* row = state + static_cast<std::int64_t>(world) * kStateBytes;
    std::uint8_t* work = scratch + static_cast<std::int64_t>(world) * kScratchBytes;
    if (field<std::uint64_t>(row, kHeaderOffset)[kStatus] != 0) return;
    const std::uint32_t ordered_slot = ordinal < kCapacity
        ? slot_for_ordinal(row, ordinal) : kEmpty;
    const bool live = ordered_slot != kEmpty && ordered_slot < kCapacity &&
        field<std::uint8_t>(row, kAliveOffset)[ordered_slot];
    active_scan[threadIdx.x] = live;
    free_scan[threadIdx.x] = ordinal < kCapacity &&
        !field<std::uint8_t>(row, kAliveOffset)[ordinal];
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            active_scan[threadIdx.x] += active_scan[threadIdx.x + stride];
            free_scan[threadIdx.x] += free_scan[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        field<std::uint32_t>(work, kTileActiveCountOffset)[blockIdx.x] = active_scan[0];
        field<std::uint32_t>(work, kTileFreeCountOffset)[blockIdx.x] = free_scan[0];
        field<std::uint32_t>(work, kTileBirthCountOffset)[blockIdx.x] = 0;
    }
}

__global__ void rank_prefix_phase_v2(std::uint8_t* state, std::uint8_t* scratch) {
    const int world = blockIdx.y;
    if (threadIdx.x != 0) return;
    std::uint8_t* row = state + static_cast<std::int64_t>(world) * kStateBytes;
    std::uint8_t* work = scratch + static_cast<std::int64_t>(world) * kScratchBytes;
    if (field<std::uint64_t>(row, kHeaderOffset)[kStatus] != 0) return;
    auto* active = field<std::uint32_t>(work, kTileActiveCountOffset);
    auto* free = field<std::uint32_t>(work, kTileFreeCountOffset);
    auto* birth = field<std::uint32_t>(work, kTileBirthCountOffset);
    std::uint32_t active_total = 0;
    std::uint32_t free_total = 0;
    for (int tile = 0; tile < kAgentTiles; ++tile) {
        const std::uint32_t active_count = active[tile];
        const std::uint32_t free_count = free[tile];
        active[tile] = active_total;
        free[tile] = free_total;
        birth[tile] = 0;
        active_total += active_count;
        free_total += free_count;
    }
    auto* control = field<std::uint64_t>(work, kWorldControlOffset);
    control[0] = active_total;
    control[1] = free_total;
    control[2] = 0;
    control[3] = field<std::uint64_t>(row, kHeaderOffset)[kNextAgentId];
}

__global__ void scatter_ranks_phase_v2(std::uint8_t* state, std::uint8_t* scratch) {
    __shared__ std::uint32_t active_scan[kThreads];
    __shared__ std::uint32_t free_scan[kThreads];
    const int world = blockIdx.y;
    const int ordinal = blockIdx.x * blockDim.x + threadIdx.x;
    std::uint8_t* row = state + static_cast<std::int64_t>(world) * kStateBytes;
    std::uint8_t* work = scratch + static_cast<std::int64_t>(world) * kScratchBytes;
    if (field<std::uint64_t>(row, kHeaderOffset)[kStatus] != 0) return;
    const std::uint32_t ordered_slot = ordinal < kCapacity
        ? slot_for_ordinal(row, ordinal) : kEmpty;
    const bool live = ordered_slot != kEmpty && ordered_slot < kCapacity &&
        field<std::uint8_t>(row, kAliveOffset)[ordered_slot];
    active_scan[threadIdx.x] = live;
    free_scan[threadIdx.x] = ordinal < kCapacity &&
        !field<std::uint8_t>(row, kAliveOffset)[ordinal];
    __syncthreads();
    for (int offset = 1; offset < blockDim.x; offset <<= 1) {
        const std::uint32_t add_active =
            threadIdx.x >= offset ? active_scan[threadIdx.x - offset] : 0;
        const std::uint32_t add_free =
            threadIdx.x >= offset ? free_scan[threadIdx.x - offset] : 0;
        __syncthreads();
        active_scan[threadIdx.x] += add_active;
        free_scan[threadIdx.x] += add_free;
        __syncthreads();
    }
    if (ordinal >= kCapacity) return;
    if (live) {
        const std::uint32_t index =
            field<std::uint32_t>(work, kTileActiveCountOffset)[blockIdx.x] +
            active_scan[threadIdx.x] - 1;
        field<std::uint32_t>(work, kActiveSlotsOffset)[index] = ordered_slot;
    }
    if (!field<std::uint8_t>(row, kAliveOffset)[ordinal]) {
        const std::uint32_t index =
            field<std::uint32_t>(work, kTileFreeCountOffset)[blockIdx.x] +
            free_scan[threadIdx.x] - 1;
        field<std::uint32_t>(work, kFreeSlotsOffset)[index] = ordinal;
    }
}

__device__ __forceinline__ BirthCounterDeltasV2 construct_child(
    std::uint8_t* row,
    std::uint8_t* work,
    LifecycleEventRow* event_base,
    int parent_slot,
    std::uint64_t tick,
    std::uint32_t disable_mask
) {
    BirthCounterDeltasV2 deltas{};
    const std::uint32_t rank = field<std::uint32_t>(work, kBirthRankOffset)[parent_slot];
    const auto* control = field<std::uint64_t>(work, kWorldControlOffset);
    if (rank == kEmpty || rank >= control[2] || rank >= control[1]) return deltas;
    const std::uint32_t mate_slot = field<std::uint32_t>(work, kMateSlotOffset)[parent_slot];
    const std::uint32_t child_slot = field<std::uint32_t>(work, kFreeSlotsOffset)[rank];
    const std::uint32_t spawn = field<std::uint32_t>(work, kSpawnSiteOffset)[parent_slot];
    auto* header = field<std::uint64_t>(row, kHeaderOffset);
    auto* agent_id = field<std::uint64_t>(row, kAgentIdOffset);
    const std::uint64_t child_id = control[3] + rank;
    auto* genome_a = field<std::uint8_t>(row, kGenomeAOffset);
    auto* genome_b = field<std::uint8_t>(row, kGenomeBOffset);
    auto* expressed = field<std::uint8_t>(row, kExpressedOffset);
    int mutation_locus = -1;
    const bool fixture = (header[kFixtureFlags] & kFixtureForceMutation) != 0;
    const std::uint64_t mutation_threshold =
        control[kScienceControlsApplied] == 0
            ? kDefaultMutationThresholdU16
            : control[kScienceMutationThresholdControl];
    if (!(disable_mask & kDisableMutation)) {
        const std::uint32_t mutation_draw = addressed_rng(
            header, tick, kDomainMutation, child_id, 0, 0
        );
        const bool default_rate = mutation_threshold == kDefaultMutationThresholdU16;
        const bool selected = default_rate
            ? (mutation_draw & 7u) == 0u
            : (mutation_draw & 0xffffu) < mutation_threshold;
        if (fixture || selected) {
            mutation_locus = mutation_draw % kGenomeLoci;
        }
    }
    for (int locus = 0; locus < kGenomeLoci; ++locus) {
        const int parent_index = parent_slot * kGenomeLoci + locus;
        const int mate_index = mate_slot * kGenomeLoci + locus;
        const int child_index = child_slot * kGenomeLoci + locus;
        std::uint8_t first = genome_a[parent_index];
        std::uint8_t second = genome_a[mate_index];
        if (!(disable_mask & kDisableRecombination)) {
            if (addressed_rng(header, tick, kDomainRecombinationA, child_id, locus, 0) & 1u) {
                first = genome_b[parent_index];
            }
            if (addressed_rng(header, tick, kDomainRecombinationB, child_id, locus, 0) & 1u) {
                second = genome_b[mate_index];
            }
        }
        if (locus == mutation_locus) first ^= 0x80u;
        genome_a[child_index] = first;
        genome_b[child_index] = second;
        expressed[child_index] = static_cast<std::uint8_t>(
            (static_cast<unsigned>(first) + second) / 2
        );
    }
    field<std::uint32_t>(row, kPositionOffset)[child_slot] = spawn;
    field<std::int32_t>(row, kEnergyOffset)[child_slot] = 128;
    field<std::int32_t>(row, kHealthOffset)[child_slot] = 256;
    field<std::uint32_t>(row, kAgeOffset)[child_slot] = 0;
    field<std::uint64_t>(row, kBirthTickOffset)[child_slot] = tick + 1;
    agent_id[child_slot] = child_id;
    field<std::uint64_t>(row, kParentAIdOffset)[child_slot] = agent_id[parent_slot];
    field<std::uint64_t>(row, kParentBIdOffset)[child_slot] = agent_id[mate_slot];
    field<std::int32_t>(row, kEnergyOffset)[parent_slot] -= 64;
    field<std::int32_t>(row, kEnergyOffset)[mate_slot] -= 64;
    auto* occupancy = field<std::uint32_t>(row, kOccupancyOffset);
    if ((header[kFixtureFlags] & kFixtureBirthCommitConflictAfterDeath) != 0 &&
        rank == 0) {
        occupancy[spawn] = static_cast<std::uint32_t>(parent_slot);
    }
    if (atomicCAS(occupancy + spawn, kEmpty, child_slot) != kEmpty) {
        set_first_status(row, 4);
        return deltas;
    }
    __threadfence();
    field<std::uint8_t>(row, kAliveOffset)[child_slot] = 1;
    field<std::uint32_t>(work, kActiveSlotsOffset)[control[0] + rank] = child_slot;
    if (event_base != nullptr) {
        write_event(
            event_base + kCapacity + child_slot,
            tick,
            2,
            child_slot,
            spawn,
            child_id,
            agent_id[parent_slot],
            agent_id[mate_slot]
        );
    }
    deltas.births = 1;
    if (!(disable_mask & kDisableRecombination)) {
        deltas.recombination_births = 1;
    }
    if (mutation_locus >= 0) {
        deltas.mutations = 1;
    }
    return deltas;
}

__global__ void birth_commit_phase_v2(
    std::uint8_t* state,
    std::uint8_t* scratch,
    LifecycleEventRow* events,
    std::int64_t* event_counts,
    int local_tick,
    int event_k_stride,
    std::uint64_t tick,
    std::uint32_t disable_mask
) {
    __shared__ BirthCounterDeltasV2 warp_totals[kWarpsPerBlock];
    const int world = blockIdx.y;
    const int slot = blockIdx.x * blockDim.x + threadIdx.x;
    std::uint8_t* row = state + static_cast<std::int64_t>(world) * kStateBytes;
    BirthCounterDeltasV2 deltas{};
    auto* event_base = kBackendKind == 0
        ? events + static_cast<std::int64_t>(world) * kEventRowsPerTick
        : nullptr;
    if (slot < kCapacity &&
        field<std::uint64_t>(row, kHeaderOffset)[kStatus] == 0) {
        tick = field<std::uint64_t>(row, kHeaderOffset)[kNextAbsoluteTick];
        deltas = construct_child(
            row,
            scratch + static_cast<std::int64_t>(world) * kScratchBytes,
            event_base,
            slot,
            tick,
            disable_mask
        );
    }
    const BirthCounterDeltasV2 total = reduce_cta_birth(deltas, warp_totals);
    if (threadIdx.x == 0) {
        auto* header = field<std::uint64_t>(row, kHeaderOffset);
        if (total.births) {
            if (kBackendKind == 0) {
                atomic_add_u64(
                    reinterpret_cast<std::uint64_t*>(event_counts + world),
                    total.births
                );
            }
            atomic_add_u64(header + kBirthCount, total.births);
            atomic_add_u64(header + kBirthCapabilityCounter, total.births);
        }
        if (total.mutations) {
            atomic_add_u64(header + kMutationCount, total.mutations);
            atomic_add_u64(header + kMutationCapabilityCounter, total.mutations);
        }
        if (total.recombination_births) {
            atomic_add_u64(
                header + kRecombinationCount,
                static_cast<std::uint64_t>(total.recombination_births) *
                    kGenomeLoci * 2
            );
            atomic_add_u64(
                header + kRecombinationCapabilityCounter,
                total.recombination_births
            );
        }
    }
}

__global__ void publish_phase_v2(
    std::uint8_t* state, std::uint8_t* scratch, std::int64_t* event_counts, std::uint64_t tick
) {
    const int world = blockIdx.y;
    std::uint8_t* row = state + static_cast<std::int64_t>(world) * kStateBytes;
    std::uint8_t* work = scratch + static_cast<std::int64_t>(world) * kScratchBytes;
    auto* header = field<std::uint64_t>(row, kHeaderOffset);
    tick = header[kNextAbsoluteTick];
    const auto* control = field<std::uint64_t>(work, kWorldControlOffset);
    if (header[kStatus] != 0) return;
    const std::uint64_t next_active = control[0] + control[2];
    auto* active_order = field<std::uint32_t>(row, kActiveSlotsByUidOffset);
    const auto* next_order = field<std::uint32_t>(work, kActiveSlotsOffset);
    for (std::uint64_t ordinal = threadIdx.x; ordinal < next_active; ordinal += blockDim.x) {
        active_order[ordinal] = next_order[ordinal];
    }
    for (std::uint64_t ordinal = next_active + threadIdx.x;
         ordinal < static_cast<std::uint64_t>(kCapacity);
         ordinal += blockDim.x) {
        active_order[ordinal] = kEmpty;
    }
    __syncthreads();
    if (threadIdx.x != 0) return;
    if (header[kStatus] == 0) {
        header[kActiveCount] = next_active;
        header[kNextAgentId] = control[3] + control[2];
        header[kTicksCompleted] += 1;
        header[kLastEventCount] = header[kBirthCount] + header[kDeathCount];
        __threadfence();
        header[kNextAbsoluteTick] = tick + 1;
    }
}

// The one-CTA backend has a real single-kernel temporal path. Canonical active,
// free-slot, and birth ranks use deterministic block-tiled scans.
__global__ void fused_world_temporal_v2(
    std::uint8_t* state,
    std::uint8_t* scratch,
    LifecycleEventRow* events,
    std::int64_t* event_counts,
    int temporal_k,
    std::uint32_t disable_mask
) {
    __shared__ PhysiologyCounterDeltasV2 physiology_warp_totals[kWarpsPerBlock];
    __shared__ std::uint32_t death_warp_totals[kWarpsPerBlock];
    __shared__ BirthCounterDeltasV2 birth_counter_warp_totals[kWarpsPerBlock];
    __shared__ std::uint32_t active_scan[kThreads];
    __shared__ std::uint32_t free_scan[kThreads];
    __shared__ std::uint32_t accepted_scan[kThreads];
    __shared__ std::uint32_t active_base;
    __shared__ std::uint32_t free_base;
    __shared__ std::uint32_t accepted_base;
    __shared__ std::int64_t event_tile_base;
    const int world = blockIdx.y;
    std::uint8_t* row = state + static_cast<std::int64_t>(world) * kStateBytes;
    std::uint8_t* work = scratch + static_cast<std::int64_t>(world) * kScratchBytes;
    auto* header = field<std::uint64_t>(row, kHeaderOffset);
    LifecycleEventRow* dispatch_events = events +
        static_cast<std::int64_t>(world) * kEventRowsPerTick;
    const std::int64_t dispatch_event_rows = kEventRowsPerTick;
    for (std::int64_t index = threadIdx.x; index < dispatch_event_rows;
         index += blockDim.x) {
        invalidate_event_status(dispatch_events + index);
    }
    __syncthreads();
    if (header[kStatus] != 0) return;
    for (int local_tick = 0; local_tick < temporal_k; ++local_tick) {
        const std::uint64_t tick = header[kNextAbsoluteTick];
        LifecycleEventRow* event_base = nullptr;
        for (int site = threadIdx.x; site < kSites; site += blockDim.x) {
            environment_site(row, work, site, disable_mask);
        }
        for (int slot = threadIdx.x; slot < kCapacity; slot += blockDim.x) {
            clear_agent_scratch(work, slot);
        }
        __syncthreads();
        if (header[kStatus] != 0) return;
        for (int ordinal = threadIdx.x; ordinal < kCapacity; ordinal += blockDim.x) {
            const std::uint32_t slot = slot_for_ordinal(row, ordinal);
            if (slot != kEmpty && slot < static_cast<std::uint32_t>(kCapacity))
                propose_move(row, work, slot, tick);
        }
        __syncthreads();
        if (header[kStatus] != 0) return;
        for (int ordinal = threadIdx.x; ordinal < kCapacity; ordinal += blockDim.x) {
            const std::uint32_t slot = slot_for_ordinal(row, ordinal);
            if (slot != kEmpty && slot < static_cast<std::uint32_t>(kCapacity))
                decide_move(row, work, slot);
        }
        __syncthreads();
        if (header[kStatus] != 0) return;
        for (int ordinal = threadIdx.x; ordinal < kCapacity; ordinal += blockDim.x) {
            const std::uint32_t slot = slot_for_ordinal(row, ordinal);
            if (slot != kEmpty && slot < static_cast<std::uint32_t>(kCapacity))
                release_move(row, work, slot);
        }
        __syncthreads();
        if (header[kStatus] != 0) return;
        for (int ordinal = threadIdx.x; ordinal < kCapacity; ordinal += blockDim.x) {
            const std::uint32_t slot = slot_for_ordinal(row, ordinal);
            if (slot != kEmpty && slot < static_cast<std::uint32_t>(kCapacity))
                commit_move(row, work, slot);
        }
        __syncthreads();
        if (header[kStatus] != 0) return;
        PhysiologyCounterDeltasV2 physiology_deltas{};
        for (int ordinal = threadIdx.x; ordinal < kCapacity; ordinal += blockDim.x) {
            const std::uint32_t slot = slot_for_ordinal(row, ordinal);
            if (slot != kEmpty && slot < static_cast<std::uint32_t>(kCapacity)) {
                const PhysiologyCounterDeltasV2 one = interact_and_physiology(
                    row, work, slot, disable_mask
                );
                physiology_deltas.patch += one.patch;
                physiology_deltas.wdl += one.wdl;
                physiology_deltas.jungle += one.jungle;
                physiology_deltas.eed += one.eed;
            }
        }
        commit_physiology_cta_totals(
            row, physiology_deltas, physiology_warp_totals
        );
        __syncthreads();
        if (header[kStatus] != 0) return;
        auto* death_flags = field<std::uint8_t>(work, kMoveAcceptedOffset);
        for (int slot = threadIdx.x; slot < kCapacity; slot += blockDim.x) {
            death_flags[slot] = 0;
        }
        __syncthreads();
        std::uint32_t death_deltas = 0;
        for (int ordinal = threadIdx.x; ordinal < kCapacity; ordinal += blockDim.x) {
            const std::uint32_t slot = slot_for_ordinal(row, ordinal);
            if (slot != kEmpty && slot < static_cast<std::uint32_t>(kCapacity)) {
                const bool committed = commit_death(row, event_base, slot, tick);
                death_flags[slot] = static_cast<std::uint8_t>(committed);
                death_deltas += committed;
            }
        }
        const std::uint32_t death_total = reduce_cta_u32(
            death_deltas, death_warp_totals
        );
        if (threadIdx.x == 0 && death_total) {
            header[kDeathCount] += death_total;
        }
        __syncthreads();
        append_sparse_page_events_v3<1>(
            row,
            work,
            events,
            event_counts,
            tick,
            active_scan,
            &event_tile_base
        );
        if (header[kStatus] != 0) return;
        auto* active_slots = field<std::uint32_t>(work, kActiveSlotsOffset);
        auto* free_slots = field<std::uint32_t>(work, kFreeSlotsOffset);
        auto* tile_active = field<std::uint32_t>(work, kTileActiveCountOffset);
        auto* tile_free = field<std::uint32_t>(work, kTileFreeCountOffset);
        auto* tile_birth = field<std::uint32_t>(work, kTileBirthCountOffset);
        const auto* prior_order = field<std::uint32_t>(
            row, kActiveSlotsByUidOffset
        );
        const auto* alive = field<std::uint8_t>(row, kAliveOffset);
        const std::uint64_t old_active = header[kActiveCount];
        if (threadIdx.x == 0) {
            active_base = 0;
            free_base = 0;
        }
        __syncthreads();
        for (int tile = 0; tile < kAgentTiles; ++tile) {
            const int ordinal = tile * kThreads + threadIdx.x;
            const std::uint32_t ordered_slot =
                static_cast<std::uint64_t>(ordinal) < old_active
                ? prior_order[ordinal] : kEmpty;
            const bool live = ordered_slot != kEmpty && ordered_slot < kCapacity &&
                alive[ordered_slot];
            const bool free = ordinal < kCapacity && !alive[ordinal];
            if (threadIdx.x == 0) {
                tile_active[tile] = active_base;
                tile_free[tile] = free_base;
                tile_birth[tile] = 0;
            }
            active_scan[threadIdx.x] = live ? 1u : 0u;
            free_scan[threadIdx.x] = free ? 1u : 0u;
            __syncthreads();
            for (int offset = 1; offset < blockDim.x; offset <<= 1) {
                const std::uint32_t active_add = threadIdx.x >= offset
                    ? active_scan[threadIdx.x - offset] : 0u;
                const std::uint32_t free_add = threadIdx.x >= offset
                    ? free_scan[threadIdx.x - offset] : 0u;
                __syncthreads();
                active_scan[threadIdx.x] += active_add;
                free_scan[threadIdx.x] += free_add;
                __syncthreads();
            }
            const std::uint32_t tile_active_base = active_base;
            const std::uint32_t tile_free_base = free_base;
            if (live) {
                active_slots[tile_active_base + active_scan[threadIdx.x] - 1u] =
                    ordered_slot;
            }
            if (free) {
                free_slots[tile_free_base + free_scan[threadIdx.x] - 1u] =
                    static_cast<std::uint32_t>(ordinal);
            }
            __syncthreads();
            if (threadIdx.x == 0) {
                active_base += active_scan[kThreads - 1];
                free_base += free_scan[kThreads - 1];
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            auto* mutable_control = field<std::uint64_t>(
                work, kWorldControlOffset
            );
            mutable_control[0] = active_base;
            mutable_control[1] = free_base;
            mutable_control[2] = 0;
            mutable_control[3] = header[kNextAgentId];
        }
        __syncthreads();
        const auto* control = field<std::uint64_t>(work, kWorldControlOffset);
        for (std::uint32_t survivor_ordinal = threadIdx.x;
             survivor_ordinal < control[0];
             survivor_ordinal += blockDim.x) {
            propose_birth_from_survivor_order(
                row, work, survivor_ordinal, disable_mask
            );
        }
        __syncthreads();
        if (threadIdx.x == 0) accepted_base = 0;
        __syncthreads();
        auto* birth_rank = field<std::uint32_t>(work, kBirthRankOffset);
        auto* birth_accepted = field<std::uint8_t>(work, kBirthAcceptedOffset);
        for (std::uint32_t tile = 0; tile < control[0]; tile += blockDim.x) {
            const std::uint32_t survivor_ordinal = tile + threadIdx.x;
            const std::uint32_t slot = survivor_ordinal < control[0]
                ? active_slots[survivor_ordinal] : kEmpty;
            const bool accepted = survivor_ordinal < control[0] &&
                (survivor_ordinal & 1u) == 0u && slot < kCapacity &&
                accepted_birth_candidate(row, work, static_cast<int>(slot));
            accepted_scan[threadIdx.x] = accepted ? 1u : 0u;
            __syncthreads();
            for (int offset = 1; offset < blockDim.x; offset <<= 1) {
                const std::uint32_t add = threadIdx.x >= offset
                    ? accepted_scan[threadIdx.x - offset] : 0u;
                __syncthreads();
                accepted_scan[threadIdx.x] += add;
                __syncthreads();
            }
            const std::uint32_t tile_base = accepted_base;
            if (accepted) {
                birth_rank[slot] = tile_base + accepted_scan[threadIdx.x] - 1u;
                birth_accepted[slot] = 1;
            }
            __syncthreads();
            if (threadIdx.x == 0) {
                accepted_base += accepted_scan[kThreads - 1];
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            field<std::uint64_t>(work, kWorldControlOffset)[2] = min(
                static_cast<std::uint64_t>(accepted_base), control[1]
            );
        }
        __syncthreads();
        BirthCounterDeltasV2 birth_deltas{};
        for (int slot = threadIdx.x; slot < kCapacity; slot += blockDim.x) {
            const BirthCounterDeltasV2 one = construct_child(
                row, work, event_base, slot, tick, disable_mask
            );
            birth_deltas.births += one.births;
            birth_deltas.mutations += one.mutations;
            birth_deltas.recombination_births += one.recombination_births;
        }
        const BirthCounterDeltasV2 birth_total = reduce_cta_birth(
            birth_deltas, birth_counter_warp_totals
        );
        if (threadIdx.x == 0) {
            header[kBirthCount] += birth_total.births;
            header[kBirthCapabilityCounter] += birth_total.births;
            header[kMutationCount] += birth_total.mutations;
            header[kMutationCapabilityCounter] += birth_total.mutations;
            header[kRecombinationCount] +=
                static_cast<std::uint64_t>(birth_total.recombination_births) *
                kGenomeLoci * 2;
            header[kRecombinationCapabilityCounter] +=
                birth_total.recombination_births;
        }
        __syncthreads();
        append_sparse_page_events_v3<2>(
            row,
            work,
            events,
            event_counts,
            tick,
            active_scan,
            &event_tile_base
        );
        if (header[kStatus] != 0) return;
        if (header[kStatus] == 0) {
            const auto* control = field<std::uint64_t>(work, kWorldControlOffset);
            const std::uint64_t next_active = control[0] + control[2];
            auto* active_order = field<std::uint32_t>(row, kActiveSlotsByUidOffset);
            const auto* next_order = field<std::uint32_t>(work, kActiveSlotsOffset);
            for (std::uint64_t ordinal = threadIdx.x; ordinal < next_active;
                 ordinal += blockDim.x) {
                active_order[ordinal] = next_order[ordinal];
            }
            for (std::uint64_t ordinal = next_active + threadIdx.x;
                 ordinal < static_cast<std::uint64_t>(kCapacity);
                 ordinal += blockDim.x) {
                active_order[ordinal] = kEmpty;
            }
        }
        __syncthreads();
        if (threadIdx.x == 0 && header[kStatus] == 0) {
            const auto* control = field<std::uint64_t>(work, kWorldControlOffset);
            header[kActiveCount] = control[0] + control[2];
            header[kNextAgentId] = control[3] + control[2];
            header[kTicksCompleted] += 1;
            header[kLastEventCount] = header[kBirthCount] + header[kDeathCount];
            __threadfence();
            header[kNextAbsoluteTick] = tick + 1;
        }
        __syncthreads();
    }
}

__global__ void force_causal_fixture_v2(std::uint8_t* state) {
    const int world = blockIdx.y;
    std::uint8_t* row = state + static_cast<std::int64_t>(world) * kStateBytes;
    if (threadIdx.x != 0) return;
    auto* header = field<std::uint64_t>(row, kHeaderOffset);
    auto* alive = field<std::uint8_t>(row, kAliveOffset);
    auto* occupancy = field<std::uint32_t>(row, kOccupancyOffset);
    auto* resource = field<std::uint16_t>(row, kResourceOffset);
    auto* patch = field<std::uint16_t>(row, kPatchFieldOffset);
    auto* patch_pending = field<std::uint16_t>(row, kPatchPendingOffset);
    auto* position = field<std::uint32_t>(row, kPositionOffset);
    auto* energy = field<std::int32_t>(row, kEnergyOffset);
    auto* health = field<std::int32_t>(row, kHealthOffset);
    auto* genome_a = field<std::uint8_t>(row, kGenomeAOffset);
    auto* genome_b = field<std::uint8_t>(row, kGenomeBOffset);
    auto* expressed = field<std::uint8_t>(row, kExpressedOffset);
    header[kFixtureFlags] = 1;
    for (int slot = 0; slot < kCapacity; ++slot) {
        if (alive[slot]) energy[slot] = 160;
    }
    energy[0] = 600;
    energy[1] = 600;
    health[2] = 0;
    auto* fixture_wdl_due = field<std::uint16_t>(row, kWdlDelay0Offset);
    for (int site = 0; site < kSites; ++site) {
        fixture_wdl_due[site] = 64;
        resource[site] = 0;
        patch[site] = 0;
        patch_pending[site] = 64;
    }
    field<std::uint8_t>(row, kJungleOffset)[position[1]] = 127;
    for (int locus = 0; locus < kGenomeLoci; ++locus) {
        genome_a[locus] = static_cast<std::uint8_t>(8 + locus);
        genome_b[locus] = static_cast<std::uint8_t>(72 + locus);
        genome_a[kGenomeLoci + locus] = static_cast<std::uint8_t>(136 + locus);
        genome_b[kGenomeLoci + locus] = static_cast<std::uint8_t>(200 + locus);
        expressed[locus] = static_cast<std::uint8_t>((genome_a[locus] + genome_b[locus]) / 2);
        expressed[kGenomeLoci + locus] = static_cast<std::uint8_t>(
            (static_cast<unsigned>(genome_a[kGenomeLoci + locus]) +
             genome_b[kGenomeLoci + locus]) / 2
        );
    }
    genome_a[31] = 255;
    genome_b[31] = 255;
    genome_a[kGenomeLoci + 31] = 255;
    genome_b[kGenomeLoci + 31] = 255;
    expressed[31] = 255;
    expressed[kGenomeLoci + 31] = 255;
    const std::uint32_t origin = position[0];
    for (std::uint32_t direction = 0; direction < 4; ++direction) {
        const std::uint32_t possible_position = neighbor_site(origin, direction);
        const std::uint32_t possible_spawn = neighbor_site(possible_position, 2);
        if (possible_spawn == origin || occupancy[possible_spawn] == kEmpty) continue;
        const std::uint32_t occupant = occupancy[possible_spawn];
        occupancy[possible_spawn] = kEmpty;
        if (occupant < static_cast<std::uint32_t>(kCapacity) && occupant > 1) {
            alive[occupant] = 0;
            position[occupant] = kEmpty;
        }
    }
}

__global__ void force_birth_overflow_fixture_v2(std::uint8_t* state) {
    const int world = blockIdx.y;
    if (threadIdx.x != 0) return;
    std::uint8_t* row = state + static_cast<std::int64_t>(world) * kStateBytes;
    auto* header = field<std::uint64_t>(row, kHeaderOffset);
    auto* occupancy = field<std::uint32_t>(row, kOccupancyOffset);
    auto* alive = field<std::uint8_t>(row, kAliveOffset);
    auto* order = field<std::uint32_t>(row, kActiveSlotsByUidOffset);
    auto* position = field<std::uint32_t>(row, kPositionOffset);
    auto* energy = field<std::int32_t>(row, kEnergyOffset);
    auto* health = field<std::int32_t>(row, kHealthOffset);
    auto* age = field<std::uint32_t>(row, kAgeOffset);
    auto* birth_tick = field<std::uint64_t>(row, kBirthTickOffset);
    auto* agent_id = field<std::uint64_t>(row, kAgentIdOffset);
    auto* parent_a = field<std::uint64_t>(row, kParentAIdOffset);
    auto* parent_b = field<std::uint64_t>(row, kParentBIdOffset);
    for (int site = 0; site < kSites; ++site) occupancy[site] = kEmpty;
    for (int slot = 0; slot < kCapacity; ++slot) {
        const bool live = slot + 1 < kCapacity;
        const std::uint32_t site = live
            ? static_cast<std::uint32_t>(
                  (2 * (slot / kWidth)) * kWidth + (slot % kWidth)
              )
            : kEmpty;
        alive[slot] = live;
        order[slot] = live ? static_cast<std::uint32_t>(slot) : kEmpty;
        position[slot] = site;
        energy[slot] = live ? 600 : 0;
        health[slot] = live ? 256 : 0;
        age[slot] = 0;
        birth_tick[slot] = live ? header[kNextAbsoluteTick] : 0;
        agent_id[slot] = live ? static_cast<std::uint64_t>(slot + 1) : 0;
        parent_a[slot] = 0;
        parent_b[slot] = 0;
        if (live) occupancy[site] = static_cast<std::uint32_t>(slot);
    }
    header[kActiveCount] = kCapacity - 1;
    header[kNextAgentId] = kCapacity;
    header[kStatus] = 0;
    header[kFixtureFlags] = 0;
}

// Decisive ordering sentinel: B dies after physiology, leaving [A,C,D].  The
// frozen law must pair A-C.  The obsolete raw-ordinal law would suppress A-B
// and then reject C-D because D is below the energy threshold.
__global__ void force_middle_survivor_death_fixture_v2(std::uint8_t* state) {
    const int world = blockIdx.y;
    if (threadIdx.x != 0) return;
    std::uint8_t* row = state + static_cast<std::int64_t>(world) * kStateBytes;
    auto* header = field<std::uint64_t>(row, kHeaderOffset);
    auto* occupancy = field<std::uint32_t>(row, kOccupancyOffset);
    auto* alive = field<std::uint8_t>(row, kAliveOffset);
    auto* order = field<std::uint32_t>(row, kActiveSlotsByUidOffset);
    auto* position = field<std::uint32_t>(row, kPositionOffset);
    auto* energy = field<std::int32_t>(row, kEnergyOffset);
    auto* health = field<std::int32_t>(row, kHealthOffset);
    auto* age = field<std::uint32_t>(row, kAgeOffset);
    auto* birth_tick = field<std::uint64_t>(row, kBirthTickOffset);
    auto* agent_id = field<std::uint64_t>(row, kAgentIdOffset);
    auto* parent_a = field<std::uint64_t>(row, kParentAIdOffset);
    auto* parent_b = field<std::uint64_t>(row, kParentBIdOffset);
    auto* genome_a = field<std::uint8_t>(row, kGenomeAOffset);
    auto* genome_b = field<std::uint8_t>(row, kGenomeBOffset);
    auto* expressed = field<std::uint8_t>(row, kExpressedOffset);
    for (int site = 0; site < kSites; ++site) occupancy[site] = kEmpty;
    for (int slot = 0; slot < kCapacity; ++slot) {
        const bool live = slot < 4;
        const std::uint32_t site = live
            ? static_cast<std::uint32_t>((2 * slot) * kWidth + 2 * slot)
            : kEmpty;
        alive[slot] = live;
        order[slot] = live ? static_cast<std::uint32_t>(slot) : kEmpty;
        position[slot] = site;
        energy[slot] = live ? (slot == 3 ? 64 : 600) : 0;
        health[slot] = live ? (slot == 1 ? 0 : 256) : 0;
        age[slot] = 0;
        birth_tick[slot] = live ? header[kNextAbsoluteTick] : 0;
        agent_id[slot] = live ? static_cast<std::uint64_t>(101 + slot) : 0;
        parent_a[slot] = 0;
        parent_b[slot] = 0;
        if (live) occupancy[site] = static_cast<std::uint32_t>(slot);
        for (int locus = 0; locus < kGenomeLoci; ++locus) {
            const int index = slot * kGenomeLoci + locus;
            const std::uint8_t first = live
                ? static_cast<std::uint8_t>(11 + slot + locus) : 0;
            const std::uint8_t second = live
                ? static_cast<std::uint8_t>(71 + slot + locus) : 0;
            genome_a[index] = first;
            genome_b[index] = second;
            expressed[index] = static_cast<std::uint8_t>(
                (static_cast<unsigned>(first) + second) / 2
            );
        }
    }
    header[kActiveCount] = 4;
    header[kNextAgentId] = 105;
    header[kStatus] = 0;
    header[kBirthCount] = 0;
    header[kDeathCount] = 0;
    header[kLastEventCount] = 0;
    header[kFixtureFlags] = 0;
}

__global__ void force_transaction_error_fixture_v2(std::uint8_t* state) {
    const int world = blockIdx.y;
    if (threadIdx.x != 0) return;
    std::uint8_t* row = state + static_cast<std::int64_t>(world) * kStateBytes;
    const std::uint32_t site = field<std::uint32_t>(row, kPositionOffset)[0];
    auto* occupancy = field<std::uint32_t>(row, kOccupancyOffset);
    auto* alive = field<std::uint8_t>(row, kAliveOffset);
    auto* position = field<std::uint32_t>(row, kPositionOffset);
    for (std::uint32_t direction = 0; direction < 4; ++direction) {
        const std::uint32_t target = neighbor_site(site, direction);
        const std::uint32_t occupant = occupancy[target];
        if (occupant != kEmpty && occupant != 0 &&
            occupant < static_cast<std::uint32_t>(kCapacity)) {
            occupancy[target] = kEmpty;
            alive[occupant] = 0;
            position[occupant] = kEmpty;
        }
    }
    occupancy[site] = 1;
}

__global__ void enable_event_then_transaction_error_fixture_v2(
    std::uint8_t* state
) {
    const int world = blockIdx.y;
    if (threadIdx.x != 0) return;
    std::uint8_t* row = state + static_cast<std::int64_t>(world) * kStateBytes;
    field<std::uint64_t>(row, kHeaderOffset)[kFixtureFlags] |=
        kFixtureBirthCommitConflictAfterDeath;
}

__global__ void clear_derived_v2(std::uint8_t* state, std::uint8_t* scratch) {
    const int world = blockIdx.y;
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    std::uint8_t* row = state + static_cast<std::int64_t>(world) * kStateBytes;
    std::uint8_t* work = scratch + static_cast<std::int64_t>(world) * kScratchBytes;
    if (index < kSites) field<std::uint32_t>(row, kOccupancyOffset)[index] = kEmpty;
    for (int offset = index; offset < kScratchBytes; offset += gridDim.x * blockDim.x) {
        work[offset] = 0;
    }
}

__global__ void restore_agents_v2(std::uint8_t* state) {
    const int world = blockIdx.y;
    const int slot = blockIdx.x * blockDim.x + threadIdx.x;
    if (slot >= kCapacity) return;
    std::uint8_t* row = state + static_cast<std::int64_t>(world) * kStateBytes;
    if (!field<std::uint8_t>(row, kAliveOffset)[slot]) return;
    const std::uint32_t site = field<std::uint32_t>(row, kPositionOffset)[slot];
    if (site >= static_cast<std::uint32_t>(kSites) ||
        atomicCAS(field<std::uint32_t>(row, kOccupancyOffset) + site,
                  kEmpty, static_cast<std::uint32_t>(slot)) != kEmpty) {
        set_first_status(row, 5);
    }
}

__global__ void validate_active_order_v2(std::uint8_t* state) {
    const int world = blockIdx.y;
    if (threadIdx.x != 0) return;
    std::uint8_t* row = state + static_cast<std::int64_t>(world) * kStateBytes;
    auto* header = field<std::uint64_t>(row, kHeaderOffset);
    const std::uint64_t active_count = header[kActiveCount];
    if (active_count > static_cast<std::uint64_t>(kCapacity)) {
        set_first_status(row, 6);
        return;
    }
    const auto* order = field<std::uint32_t>(row, kActiveSlotsByUidOffset);
    const auto* alive = field<std::uint8_t>(row, kAliveOffset);
    const auto* ids = field<std::uint64_t>(row, kAgentIdOffset);
    std::uint64_t previous_uid = 0;
    for (std::uint64_t ordinal = 0; ordinal < active_count; ++ordinal) {
        const std::uint32_t slot = order[ordinal];
        if (slot >= static_cast<std::uint32_t>(kCapacity) || !alive[slot] ||
            ids[slot] <= previous_uid) {
            set_first_status(row, 7);
            return;
        }
        previous_uid = ids[slot];
    }
    std::uint64_t observed_alive = 0;
    for (int slot = 0; slot < kCapacity; ++slot) observed_alive += alive[slot] != 0;
    if (observed_alive != active_count) set_first_status(row, 8);
}

void validate_tensors(
    const torch::Tensor& state,
    const torch::Tensor& scratch,
    const torch::Tensor* events = nullptr,
    const torch::Tensor* event_counts = nullptr
) {
    TORCH_CHECK(state.is_cuda() && state.scalar_type() == torch::kUInt8 &&
                state.is_contiguous() && state.dim() == 2 &&
                state.size(1) == kStateBytes,
                "state must be contiguous CUDA uint8[B,state_bytes]");
    TORCH_CHECK(scratch.is_cuda() && scratch.scalar_type() == torch::kUInt8 &&
                scratch.is_contiguous() && scratch.dim() == 2 &&
                scratch.size(0) == state.size(0) && scratch.size(1) == kScratchBytes,
                "scratch must be contiguous CUDA uint8[B,scratch_bytes]");
    TORCH_CHECK(state.device() == scratch.device(), "state and scratch device mismatch");
    if (events) {
        TORCH_CHECK(events->is_cuda() && events->scalar_type() == torch::kUInt8 &&
                    events->is_contiguous() && events->dim() == 3 &&
                    events->size(0) == state.size(0) &&
                    events->size(1) == kEventRowsPerTick &&
                    events->size(2) == kEventRowBytes,
                    "events must be exact CUDA uint8[B,2*C,48] sparse pages");
        TORCH_CHECK(events->device() == state.device(), "events device mismatch");
    }
    if (event_counts) {
        TORCH_CHECK(event_counts->is_cuda() &&
                    event_counts->scalar_type() == torch::kInt64 &&
                    event_counts->is_contiguous() && event_counts->dim() == 1 &&
                    event_counts->size(0) == state.size(0),
                    "event_counts must be exact CUDA int64[B]");
        TORCH_CHECK(event_counts->device() == state.device(), "event count device mismatch");
    }
}

void initialize(
    torch::Tensor state,
    torch::Tensor scratch,
    std::uint64_t world_uid_base,
    std::uint64_t start_tick,
    std::uint64_t rng_key
) {
    validate_tensors(state, scratch);
    c10::cuda::CUDAGuard guard(state.device());
    state.zero_();
    scratch.zero_();
    const auto stream = at::cuda::getCurrentCUDAStream(state.get_device());
    const dim3 grid(1, static_cast<unsigned>(state.size(0)), 1);
    initialize_worlds_v2<<<grid, kThreads, 0, stream>>>(
        state.data_ptr<std::uint8_t>(),
        scratch.data_ptr<std::uint8_t>(),
        nullptr,
        nullptr,
        world_uid_base,
        start_tick,
        rng_key,
        false
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void initialize_exact(
    torch::Tensor state,
    torch::Tensor scratch,
    torch::Tensor world_uids,
    std::uint64_t start_tick,
    torch::Tensor rng_keys
) {
    validate_tensors(state, scratch);
    TORCH_CHECK(world_uids.is_cuda() &&
                world_uids.scalar_type() == torch::kInt64 &&
                world_uids.is_contiguous() && world_uids.dim() == 1 &&
                world_uids.size(0) == state.size(0),
                "world_uids must be contiguous CUDA int64[B]");
    TORCH_CHECK(rng_keys.is_cuda() &&
                rng_keys.scalar_type() == torch::kInt64 &&
                rng_keys.is_contiguous() && rng_keys.dim() == 1 &&
                rng_keys.size(0) == state.size(0),
                "rng_keys must be contiguous CUDA int64[B]");
    TORCH_CHECK(world_uids.device() == state.device() &&
                rng_keys.device() == state.device(),
                "world identity vectors must share the state device");
    c10::cuda::CUDAGuard guard(state.device());
    const auto world_uids_cpu = world_uids.to(torch::kCPU).contiguous();
    const auto rng_keys_cpu = rng_keys.to(torch::kCPU).contiguous();
    const auto* world_uid_values = world_uids_cpu.data_ptr<std::int64_t>();
    const auto* rng_key_values = rng_keys_cpu.data_ptr<std::int64_t>();
    std::unordered_set<std::int64_t> unique_world_uids;
    for (std::int64_t world = 0; world < state.size(0); ++world) {
        TORCH_CHECK(world_uid_values[world] > 0,
                    "world_uids must be positive signed int64");
        TORCH_CHECK(rng_key_values[world] >= 0,
                    "rng_keys must be nonnegative signed int64");
        TORCH_CHECK(unique_world_uids.insert(world_uid_values[world]).second,
                    "world_uids must be unique within B");
    }
    state.zero_();
    scratch.zero_();
    const auto stream = at::cuda::getCurrentCUDAStream(state.get_device());
    const dim3 grid(1, static_cast<unsigned>(state.size(0)), 1);
    initialize_worlds_v2<<<grid, kThreads, 0, stream>>>(
        state.data_ptr<std::uint8_t>(),
        scratch.data_ptr<std::uint8_t>(),
        world_uids.data_ptr<std::int64_t>(),
        rng_keys.data_ptr<std::int64_t>(),
        0u,
        start_tick,
        0u,
        true
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void validate_dispatch(
    const torch::Tensor& state,
    const torch::Tensor& scratch,
    const torch::Tensor& events,
    const torch::Tensor& event_counts,
    int temporal_k,
    std::uint32_t disable_mask
) {
    validate_tensors(state, scratch, &events, &event_counts);
    TORCH_CHECK(temporal_k > 0 && temporal_k <= kMaximumTemporalK,
                "temporal_k exceeds compiled shape envelope");
    TORCH_CHECK(kBackendKind != 0 || temporal_k == 1,
                "reference_phased_v2 requires K=1");
    // The allocation law is exactly one 2*C*48-byte page per world,
    // independent of temporal K. Counts retain the complete required extent.
    TORCH_CHECK(disable_mask < (1u << 7), "unknown diagnostic ablation bits");
    TORCH_CHECK(state.size(0) > 0 && state.size(0) <= 65535,
                "B must fit the grid.y workhorse ABI");
}

void canonicalize_events(torch::Tensor events) {
    TORCH_CHECK(events.is_cuda() && events.scalar_type() == torch::kUInt8 &&
                events.is_contiguous() && events.dim() == 3 &&
                events.size(0) > 0 && events.size(0) <= 65535 &&
                events.size(1) == kEventRowsPerTick &&
                events.size(2) == kEventRowBytes,
                "events must be exact CUDA uint8[B,2*C,48] sparse pages");
    c10::cuda::CUDAGuard guard(events.device());
    const std::uint64_t row_count = static_cast<std::uint64_t>(
        events.numel() / kEventRowBytes
    );
    const std::uint64_t block_count =
        (row_count + kThreads - 1) / kThreads;
    TORCH_CHECK(
        block_count <= static_cast<std::uint64_t>(
            std::numeric_limits<unsigned>::max()
        ),
        "event canonicalization grid exceeds CUDA grid.x"
    );
    const auto stream = at::cuda::getCurrentCUDAStream(events.get_device());
    canonicalize_inactive_events_v2<<<
        static_cast<unsigned>(block_count), kThreads, 0, stream
    >>>(
        reinterpret_cast<LifecycleEventRow*>(events.data_ptr<std::uint8_t>()),
        row_count
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void enqueue_dispatch(
    std::uint8_t* state_ptr,
    std::uint8_t* scratch_ptr,
    LifecycleEventRow* event_ptr,
    std::int64_t* count_ptr,
    unsigned B,
    int temporal_k,
    std::uint32_t disable_mask,
    cudaStream_t stream
) {
    // Status is owner-lifetime state.  Initialization creates zero; restore
    // validation or a failed dispatch poisons it permanently.  In particular,
    // advance must never erase a pending restore/owner error.
    C10_CUDA_CHECK(cudaMemsetAsync(
        count_ptr, 0, static_cast<std::size_t>(B) * sizeof(std::int64_t), stream
    ));
    if (kBackendKind == 2) {
        fused_world_temporal_v2<<<dim3(1, B, 1), kThreads, 0, stream>>>(
            state_ptr, scratch_ptr, event_ptr, count_ptr, temporal_k, disable_mask
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return;
    }
    for (int local_tick = 0; local_tick < temporal_k; ++local_tick) {
        // The graph backend captures this exact multi-CTA DAG once and replays it.
        // No host intervention occurs between nodes after capture.
        const std::uint64_t tick = 0;  // Kernels read the authoritative state frontier below.
        environment_phase_v2<<<dim3(kEnvironmentTiles, B, 1), kThreads, 0, stream>>>(
            state_ptr, scratch_ptr, event_ptr, count_ptr, local_tick, temporal_k,
            disable_mask
        );
        // Passing zero means each phase reads header[kNextAbsoluteTick].  The explicit
        // value is retained in signatures to keep RNG/event addressing visible.
        agent_intent_phase_v2<<<dim3(kAgentTiles, B, 1), kThreads, 0, stream>>>(
            state_ptr, scratch_ptr, tick
        );
        movement_accept_phase_v2<<<dim3(kAgentTiles, B, 1), kThreads, 0, stream>>>(
            state_ptr, scratch_ptr
        );
        movement_release_phase_v2<<<dim3(kAgentTiles, B, 1), kThreads, 0, stream>>>(
            state_ptr, scratch_ptr
        );
        movement_commit_phase_v2<<<dim3(kAgentTiles, B, 1), kThreads, 0, stream>>>(
            state_ptr, scratch_ptr
        );
        interaction_physiology_phase_v2<<<dim3(kAgentTiles, B, 1), kThreads, 0, stream>>>(
            state_ptr, scratch_ptr, disable_mask
        );
        death_phase_v2<<<dim3(kAgentTiles, B, 1), kThreads, 0, stream>>>(
            state_ptr, scratch_ptr, event_ptr, count_ptr, local_tick, temporal_k, tick
        );
        if (kBackendKind == 1) {
            compact_phase_events_v3<1><<<dim3(1, B, 1), kThreads, 0, stream>>>(
                state_ptr, scratch_ptr, event_ptr, count_ptr
            );
        }
        rank_tiles_phase_v2<<<dim3(kAgentTiles, B, 1), kThreads, 0, stream>>>(
            state_ptr, scratch_ptr
        );
        rank_prefix_phase_v2<<<dim3(1, B, 1), kThreads, 0, stream>>>(
            state_ptr, scratch_ptr
        );
        scatter_ranks_phase_v2<<<dim3(kAgentTiles, B, 1), kThreads, 0, stream>>>(
            state_ptr, scratch_ptr
        );
        mate_spawn_phase_v2<<<dim3(1, B, 1), kThreads, 0, stream>>>(
            state_ptr, scratch_ptr, disable_mask
        );
        birth_commit_phase_v2<<<dim3(kAgentTiles, B, 1), kThreads, 0, stream>>>(
            state_ptr, scratch_ptr, event_ptr, count_ptr, local_tick, temporal_k,
            tick, disable_mask
        );
        if (kBackendKind == 1) {
            compact_phase_events_v3<2><<<dim3(1, B, 1), kThreads, 0, stream>>>(
                state_ptr, scratch_ptr, event_ptr, count_ptr
            );
        }
        publish_phase_v2<<<dim3(1, B, 1), kThreads, 0, stream>>>(
            state_ptr, scratch_ptr, count_ptr, tick
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
}

struct PreparedGraphV2 {
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executable = nullptr;
    std::uint8_t* state = nullptr;
    std::uint8_t* scratch = nullptr;
    LifecycleEventRow* events = nullptr;
    std::int64_t* event_counts = nullptr;
    int device = -1;
    unsigned B = 0;
    int temporal_k = 0;
    std::uint32_t disable_mask = 0;
    std::size_t graph_object_device_bytes = 0;
    std::size_t graph_exec_device_bytes = 0;
};

PreparedGraphV2 prepared_graph_v2;
std::mutex prepared_graph_mutex_v2;

std::string shape_id();

void destroy_prepared_graph_v2() {
    if (prepared_graph_v2.executable != nullptr) {
        C10_CUDA_CHECK(cudaGraphExecDestroy(prepared_graph_v2.executable));
    }
    if (prepared_graph_v2.graph != nullptr) {
        C10_CUDA_CHECK(cudaGraphDestroy(prepared_graph_v2.graph));
    }
    prepared_graph_v2 = PreparedGraphV2{};
}

void prepare_graph(
    torch::Tensor state,
    torch::Tensor scratch,
    torch::Tensor events,
    torch::Tensor event_counts,
    int temporal_k,
    std::uint32_t disable_mask
) {
    TORCH_CHECK(kBackendKind == 1, "prepare_graph requires phase_graph_v2");
    validate_dispatch(state, scratch, events, event_counts, temporal_k, disable_mask);
    c10::cuda::CUDAGuard guard(state.device());
    const auto selected_stream = at::cuda::getCurrentCUDAStream(state.get_device());
    cudaStream_t stream = selected_stream;
    const unsigned B = static_cast<unsigned>(state.size(0));
    auto* state_ptr = state.data_ptr<std::uint8_t>();
    auto* scratch_ptr = scratch.data_ptr<std::uint8_t>();
    auto* event_ptr = reinterpret_cast<LifecycleEventRow*>(events.data_ptr<std::uint8_t>());
    auto* count_ptr = event_counts.data_ptr<std::int64_t>();
    std::lock_guard<std::mutex> lock(prepared_graph_mutex_v2);
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    destroy_prepared_graph_v2();
    std::size_t free_before_capture = 0;
    std::size_t total_before_capture = 0;
    C10_CUDA_CHECK(cudaMemGetInfo(&free_before_capture, &total_before_capture));
    cudaGraph_t captured = nullptr;
    // PyTorch normally enters this call on its legacy default stream. CUDA
    // forbids capturing that stream, so capture the immutable dispatch DAG on
    // a private non-blocking stream after the current stream is settled. The
    // resulting graph is stream-independent and is launched later on the
    // caller's current stream.
    cudaStream_t capture_stream = nullptr;
    C10_CUDA_CHECK(cudaStreamCreateWithFlags(
        &capture_stream, cudaStreamNonBlocking
    ));
    bool capture_active = false;
    try {
        C10_CUDA_CHECK(cudaStreamBeginCapture(
            capture_stream, cudaStreamCaptureModeThreadLocal
        ));
        capture_active = true;
        enqueue_dispatch(
            state_ptr, scratch_ptr, event_ptr, count_ptr, B, temporal_k,
            disable_mask, capture_stream
        );
        C10_CUDA_CHECK(cudaStreamEndCapture(capture_stream, &captured));
        capture_active = false;
        C10_CUDA_CHECK(cudaStreamDestroy(capture_stream));
        capture_stream = nullptr;
    } catch (...) {
        if (capture_active) {
            cudaGraph_t abandoned = nullptr;
            cudaStreamEndCapture(capture_stream, &abandoned);
            if (abandoned != nullptr) cudaGraphDestroy(abandoned);
        }
        if (capture_stream != nullptr) cudaStreamDestroy(capture_stream);
        throw;
    }
    TORCH_CHECK(captured != nullptr, "phase_graph_v2 stream capture returned no graph");
    std::size_t free_after_capture = 0;
    std::size_t total_after_capture = 0;
    C10_CUDA_CHECK(cudaMemGetInfo(&free_after_capture, &total_after_capture));
    TORCH_CHECK(total_after_capture == total_before_capture,
                "device VRAM total changed during graph capture");
    cudaGraphExec_t executable = nullptr;
    cudaError_t instantiate_status = cudaGraphInstantiate(
        &executable, captured, nullptr, nullptr, 0
    );
    if (instantiate_status != cudaSuccess) {
        cudaGraphDestroy(captured);
        C10_CUDA_CHECK(instantiate_status);
    }
    std::size_t free_after_instantiate = 0;
    std::size_t total_after_instantiate = 0;
    C10_CUDA_CHECK(cudaMemGetInfo(&free_after_instantiate, &total_after_instantiate));
    TORCH_CHECK(total_after_instantiate == total_before_capture,
                "device VRAM total changed during graph instantiation");
    prepared_graph_v2.graph = captured;
    prepared_graph_v2.executable = executable;
    prepared_graph_v2.state = state_ptr;
    prepared_graph_v2.scratch = scratch_ptr;
    prepared_graph_v2.events = event_ptr;
    prepared_graph_v2.event_counts = count_ptr;
    prepared_graph_v2.device = state.get_device();
    prepared_graph_v2.B = B;
    prepared_graph_v2.temporal_k = temporal_k;
    prepared_graph_v2.disable_mask = disable_mask;
    prepared_graph_v2.graph_object_device_bytes =
        free_before_capture > free_after_capture ?
            free_before_capture - free_after_capture : 0;
    prepared_graph_v2.graph_exec_device_bytes =
        free_after_capture > free_after_instantiate ?
            free_after_capture - free_after_instantiate : 0;
}

const char* graph_kernel_name_v2(void* function) {
    if (function == reinterpret_cast<void*>(environment_phase_v2))
        return "environment_phase_v2";
    if (function == reinterpret_cast<void*>(agent_intent_phase_v2))
        return "agent_intent_phase_v2";
    if (function == reinterpret_cast<void*>(movement_accept_phase_v2))
        return "movement_accept_phase_v2";
    if (function == reinterpret_cast<void*>(movement_release_phase_v2))
        return "movement_release_phase_v2";
    if (function == reinterpret_cast<void*>(movement_commit_phase_v2))
        return "movement_commit_phase_v2";
    if (function == reinterpret_cast<void*>(interaction_physiology_phase_v2))
        return "interaction_physiology_phase_v2";
    if (function == reinterpret_cast<void*>(death_phase_v2))
        return "death_phase_v2";
    if (function == reinterpret_cast<void*>(compact_phase_events_v3<1>))
        return "compact_death_events_v3";
    if (function == reinterpret_cast<void*>(mate_spawn_phase_v2))
        return "mate_spawn_phase_v2";
    if (function == reinterpret_cast<void*>(rank_tiles_phase_v2))
        return "rank_tiles_phase_v2";
    if (function == reinterpret_cast<void*>(rank_prefix_phase_v2))
        return "rank_prefix_phase_v2";
    if (function == reinterpret_cast<void*>(scatter_ranks_phase_v2))
        return "scatter_ranks_phase_v2";
    if (function == reinterpret_cast<void*>(birth_commit_phase_v2))
        return "birth_commit_phase_v2";
    if (function == reinterpret_cast<void*>(compact_phase_events_v3<2>))
        return "compact_birth_events_v3";
    if (function == reinterpret_cast<void*>(publish_phase_v2))
        return "publish_phase_v2";
    return "UNKNOWN_KERNEL";
}

int graph_node_index_v2(
    const std::vector<cudaGraphNode_t>& nodes, cudaGraphNode_t target
) {
    for (std::size_t index = 0; index < nodes.size(); ++index) {
        if (nodes[index] == target) return static_cast<int>(index);
    }
    return -1;
}

py::dict graph_inspect() {
    TORCH_CHECK(kBackendKind == 1, "graph_inspect requires phase_graph_v2");
    std::lock_guard<std::mutex> lock(prepared_graph_mutex_v2);
    TORCH_CHECK(prepared_graph_v2.graph != nullptr &&
                prepared_graph_v2.executable != nullptr,
                "phase_graph_v2 is not prepared");
    int previous_device = -1;
    C10_CUDA_CHECK(cudaGetDevice(&previous_device));
    C10_CUDA_CHECK(cudaSetDevice(prepared_graph_v2.device));
    std::size_t node_count = 0;
    std::size_t edge_count = 0;
    C10_CUDA_CHECK(cudaGraphGetNodes(prepared_graph_v2.graph, nullptr, &node_count));
    C10_CUDA_CHECK(cudaGraphGetEdges(
        prepared_graph_v2.graph, nullptr, nullptr, &edge_count
    ));
    std::vector<cudaGraphNode_t> nodes(node_count);
    std::vector<cudaGraphNode_t> from(edge_count);
    std::vector<cudaGraphNode_t> to(edge_count);
    C10_CUDA_CHECK(cudaGraphGetNodes(
        prepared_graph_v2.graph, nodes.data(), &node_count
    ));
    C10_CUDA_CHECK(cudaGraphGetEdges(
        prepared_graph_v2.graph, from.data(), to.data(), &edge_count
    ));
    std::vector<int> successor(node_count, -1);
    std::vector<int> indegree(node_count, 0);
    bool linear = edge_count + 1 == node_count;
    for (std::size_t edge = 0; edge < edge_count; ++edge) {
        const int source = graph_node_index_v2(nodes, from[edge]);
        const int target = graph_node_index_v2(nodes, to[edge]);
        if (source < 0 || target < 0 || successor[source] != -1) {
            linear = false;
            continue;
        }
        successor[source] = target;
        indegree[target] += 1;
        if (indegree[target] > 1) linear = false;
    }
    int root = -1;
    for (std::size_t index = 0; index < node_count; ++index) {
        if (indegree[index] == 0) {
            if (root != -1) linear = false;
            root = static_cast<int>(index);
        }
    }
    std::vector<int> order;
    for (int current = root; current >= 0 && order.size() < node_count;
         current = successor[current]) {
        order.push_back(current);
    }
    if (order.size() != node_count) linear = false;
    py::list inspected_nodes;
    py::list kernel_sequence;
    int kernel_nodes = 0;
    int memset_nodes = 0;
    bool geometry_exact = true;
    for (std::size_t rank = 0; rank < order.size(); ++rank) {
        cudaGraphNodeType type{};
        C10_CUDA_CHECK(cudaGraphNodeGetType(nodes[order[rank]], &type));
        py::dict row;
        row["topological_rank"] = rank;
        if (type == cudaGraphNodeTypeKernel) {
            cudaKernelNodeParams params{};
            C10_CUDA_CHECK(cudaGraphKernelNodeGetParams(nodes[order[rank]], &params));
            const char* name = graph_kernel_name_v2(params.func);
            cudaFuncAttributes attributes{};
            C10_CUDA_CHECK(cudaFuncGetAttributes(&attributes, params.func));
            row["node_type"] = "kernel";
            row["kernel"] = name;
            row["grid"] = py::make_tuple(params.gridDim.x, params.gridDim.y, params.gridDim.z);
            row["block"] = py::make_tuple(params.blockDim.x, params.blockDim.y, params.blockDim.z);
            row["dynamic_shared_bytes"] = params.sharedMemBytes;
            row["registers_per_thread"] = attributes.numRegs;
            row["static_shared_bytes"] = attributes.sharedSizeBytes;
            row["local_bytes_per_thread"] = attributes.localSizeBytes;
            row["maximum_threads_per_block"] = attributes.maxThreadsPerBlock;
            kernel_sequence.append(name);
            ++kernel_nodes;
            const bool single_cta_world_phase =
                std::string(name) == "rank_prefix_phase_v2" ||
                std::string(name) == "mate_spawn_phase_v2" ||
                std::string(name) == "compact_death_events_v3" ||
                std::string(name) == "compact_birth_events_v3" ||
                std::string(name) == "publish_phase_v2";
            const unsigned expected_x = single_cta_world_phase ? 1u :
                (std::string(name) == "environment_phase_v2" ?
                    static_cast<unsigned>(kEnvironmentTiles) :
                    static_cast<unsigned>(kAgentTiles));
            geometry_exact = geometry_exact && std::string(name) != "UNKNOWN_KERNEL" &&
                params.gridDim.x == expected_x &&
                params.gridDim.y == prepared_graph_v2.B &&
                params.gridDim.z == 1 && params.blockDim.x == kThreads &&
                params.blockDim.y == 1 && params.blockDim.z == 1;
        } else if (type == cudaGraphNodeTypeMemset) {
            cudaMemsetParams params{};
            C10_CUDA_CHECK(cudaGraphMemsetNodeGetParams(nodes[order[rank]], &params));
            row["node_type"] = "memset";
            row["name"] = "event_count_memset_v2";
            row["pitch"] = params.pitch;
            row["value"] = params.value;
            row["element_size"] = params.elementSize;
            row["width"] = params.width;
            row["height"] = params.height;
            ++memset_nodes;
        } else {
            row["node_type"] = "unexpected";
            row["cuda_node_type"] = static_cast<int>(type);
            linear = false;
        }
        inspected_nodes.append(row);
    }
    py::list expected_kernel_sequence;
    const char* expected_names[] = {
        "environment_phase_v2", "agent_intent_phase_v2",
        "movement_accept_phase_v2", "movement_release_phase_v2",
        "movement_commit_phase_v2", "interaction_physiology_phase_v2",
        "death_phase_v2", "compact_death_events_v3",
        "rank_tiles_phase_v2", "rank_prefix_phase_v2",
        "scatter_ranks_phase_v2", "mate_spawn_phase_v2",
        "birth_commit_phase_v2", "compact_birth_events_v3",
        "publish_phase_v2",
    };
    for (int local_tick = 0; local_tick < prepared_graph_v2.temporal_k; ++local_tick) {
        for (const char* name : expected_names) expected_kernel_sequence.append(name);
    }
    py::dict result;
    result["schema_version"] = "eiel_full_evolution_cuda_graph_inspection_v2";
    result["runtime_inspected"] = true;
    result["shape_id"] = shape_id();
    result["backend_id"] = EIEL_WORKHORSE_BACKEND_ID;
    result["B"] = prepared_graph_v2.B;
    result["temporal_k"] = prepared_graph_v2.temporal_k;
    result["disable_mask"] = prepared_graph_v2.disable_mask;
    result["graph_object_device_bytes"] =
        prepared_graph_v2.graph_object_device_bytes;
    result["graph_exec_device_bytes"] = prepared_graph_v2.graph_exec_device_bytes;
    result["node_count"] = node_count;
    result["edge_count"] = edge_count;
    result["kernel_node_count"] = kernel_nodes;
    result["memset_node_count"] = memset_nodes;
    result["linear_chain"] = linear;
    result["kernel_sequence"] = kernel_sequence;
    result["expected_kernel_sequence"] = expected_kernel_sequence;
    result["kernel_sequence_exact"] = kernel_sequence.equal(expected_kernel_sequence);
    result["kernel_geometry_exact"] = geometry_exact;
    result["nodes"] = inspected_nodes;
    C10_CUDA_CHECK(cudaSetDevice(previous_device));
    return result;
}

void release_graph() {
    TORCH_CHECK(kBackendKind == 1, "release_graph requires phase_graph_v2");
    std::lock_guard<std::mutex> lock(prepared_graph_mutex_v2);
    if (prepared_graph_v2.graph == nullptr) return;
    int previous_device = -1;
    C10_CUDA_CHECK(cudaGetDevice(&previous_device));
    C10_CUDA_CHECK(cudaSetDevice(prepared_graph_v2.device));
    destroy_prepared_graph_v2();
    C10_CUDA_CHECK(cudaSetDevice(previous_device));
}

void launch_advance(
    torch::Tensor state,
    torch::Tensor scratch,
    torch::Tensor events,
    torch::Tensor event_counts,
    int temporal_k,
    std::uint32_t disable_mask
) {
    validate_dispatch(state, scratch, events, event_counts, temporal_k, disable_mask);
    c10::cuda::CUDAGuard guard(state.device());
    const auto selected_stream = at::cuda::getCurrentCUDAStream(state.get_device());
    cudaStream_t stream = selected_stream;
    const unsigned B = static_cast<unsigned>(state.size(0));
    auto* state_ptr = state.data_ptr<std::uint8_t>();
    auto* scratch_ptr = scratch.data_ptr<std::uint8_t>();
    auto* event_ptr = reinterpret_cast<LifecycleEventRow*>(events.data_ptr<std::uint8_t>());
    auto* count_ptr = event_counts.data_ptr<std::int64_t>();
    if (kBackendKind == 1) {
        std::lock_guard<std::mutex> lock(prepared_graph_mutex_v2);
        TORCH_CHECK(prepared_graph_v2.executable != nullptr,
                    "phase_graph_v2 requires prepare_graph before advance");
        TORCH_CHECK(prepared_graph_v2.state == state_ptr &&
                    prepared_graph_v2.scratch == scratch_ptr &&
                    prepared_graph_v2.events == event_ptr &&
                    prepared_graph_v2.event_counts == count_ptr &&
                    prepared_graph_v2.device == state.get_device() &&
                    prepared_graph_v2.B == B &&
                    prepared_graph_v2.temporal_k == temporal_k &&
                    prepared_graph_v2.disable_mask == disable_mask,
                    "phase_graph_v2 prepared graph identity/pointers do not match");
        C10_CUDA_CHECK(cudaGraphLaunch(prepared_graph_v2.executable, stream));
        return;
    }
    enqueue_dispatch(
        state_ptr, scratch_ptr, event_ptr, count_ptr, B, temporal_k, disable_mask, stream
    );
}

void advance(
    torch::Tensor state,
    torch::Tensor scratch,
    torch::Tensor events,
    torch::Tensor event_counts,
    int temporal_k
) {
    launch_advance(state, scratch, events, event_counts, temporal_k, 0);
}

void advance_diagnostic_ablation(
    torch::Tensor state,
    torch::Tensor scratch,
    torch::Tensor events,
    torch::Tensor event_counts,
    int temporal_k,
    std::uint32_t disable_mask
) {
    launch_advance(state, scratch, events, event_counts, temporal_k, disable_mask);
}

void force_causal_fixture(torch::Tensor state, torch::Tensor scratch) {
    validate_tensors(state, scratch);
    c10::cuda::CUDAGuard guard(state.device());
    const auto stream = at::cuda::getCurrentCUDAStream(state.get_device());
    force_causal_fixture_v2<<<dim3(1, static_cast<unsigned>(state.size(0)), 1),
                              kThreads, 0, stream>>>(state.data_ptr<std::uint8_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void force_birth_overflow_fixture(torch::Tensor state, torch::Tensor scratch) {
    validate_tensors(state, scratch);
    c10::cuda::CUDAGuard guard(state.device());
    const auto stream = at::cuda::getCurrentCUDAStream(state.get_device());
    force_birth_overflow_fixture_v2<<<
        dim3(1, static_cast<unsigned>(state.size(0)), 1), 1, 0, stream
    >>>(state.data_ptr<std::uint8_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void force_middle_survivor_death_fixture(
    torch::Tensor state, torch::Tensor scratch
) {
    validate_tensors(state, scratch);
    c10::cuda::CUDAGuard guard(state.device());
    const auto stream = at::cuda::getCurrentCUDAStream(state.get_device());
    force_middle_survivor_death_fixture_v2<<<
        dim3(1, static_cast<unsigned>(state.size(0)), 1), 1, 0, stream
    >>>(state.data_ptr<std::uint8_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void force_transaction_error_fixture(torch::Tensor state, torch::Tensor scratch) {
    validate_tensors(state, scratch);
    c10::cuda::CUDAGuard guard(state.device());
    const auto stream = at::cuda::getCurrentCUDAStream(state.get_device());
    force_transaction_error_fixture_v2<<<
        dim3(1, static_cast<unsigned>(state.size(0)), 1), 1, 0, stream
    >>>(state.data_ptr<std::uint8_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void force_event_then_transaction_error_fixture(
    torch::Tensor state, torch::Tensor scratch
) {
    validate_tensors(state, scratch);
    c10::cuda::CUDAGuard guard(state.device());
    const auto stream = at::cuda::getCurrentCUDAStream(state.get_device());
    force_middle_survivor_death_fixture_v2<<<
        dim3(1, static_cast<unsigned>(state.size(0)), 1), 1, 0, stream
    >>>(state.data_ptr<std::uint8_t>());
    enable_event_then_transaction_error_fixture_v2<<<
        dim3(1, static_cast<unsigned>(state.size(0)), 1), 1, 0, stream
    >>>(state.data_ptr<std::uint8_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

__global__ void apply_science_controls_v1_kernel(
    std::uint8_t* scratch,
    const std::int64_t* controls,
    std::int64_t control_stride
) {
    const int world = blockIdx.x;
    if (threadIdx.x != 0) return;
    std::uint8_t* work = scratch + static_cast<std::int64_t>(world) * kScratchBytes;
    auto* target = field<std::uint64_t>(work, kWorldControlOffset);
    target[kScienceMutationThresholdControl] =
        static_cast<std::uint64_t>(controls[world * control_stride]);
    target[kScienceResourceMultiplierQ16Control] =
        static_cast<std::uint64_t>(controls[world * control_stride + 1]);
    __threadfence();
    target[kScienceControlsApplied] = 1;
}

__global__ void validate_science_controls_v1_kernel(
    const std::uint8_t* state,
    const std::uint8_t* scratch,
    const std::int64_t* controls,
    std::int64_t control_stride,
    std::uint64_t expected_next_absolute_tick,
    std::int64_t* statuses
) {
    const int world = blockIdx.x;
    if (threadIdx.x != 0) return;
    const std::uint8_t* row =
        state + static_cast<std::int64_t>(world) * kStateBytes;
    const std::uint8_t* work =
        scratch + static_cast<std::int64_t>(world) * kScratchBytes;
    const auto* header = field<std::uint64_t>(row, kHeaderOffset);
    const auto* prior = field<std::uint64_t>(work, kWorldControlOffset);
    std::int64_t status = 0;
    if (header[kNextAbsoluteTick] != expected_next_absolute_tick) status = 1;
    else if (header[kStatus] != 0) status = 2;
    else if (prior[kScienceControlsApplied] != 0) status = 3;
    else if (controls[world * control_stride] < 0 ||
             controls[world * control_stride] > 65536) status = 4;
    else if (controls[world * control_stride + 1] < 0 ||
             controls[world * control_stride + 1] > 16 * 65536) status = 5;
    statuses[world] = status;
}

__global__ void read_science_controls_v1_kernel(
    const std::uint8_t* scratch,
    std::int64_t* controls,
    std::int64_t control_stride
) {
    const int world = blockIdx.x;
    if (threadIdx.x != 0) return;
    const std::uint8_t* work =
        scratch + static_cast<std::int64_t>(world) * kScratchBytes;
    const auto* source = field<std::uint64_t>(work, kWorldControlOffset);
    controls[world * control_stride] = static_cast<std::int64_t>(
        source[kScienceControlsApplied] == 0
            ? kDefaultMutationThresholdU16
            : source[kScienceMutationThresholdControl]
    );
    controls[world * control_stride + 1] = static_cast<std::int64_t>(
        source[kScienceControlsApplied] == 0
            ? kDefaultResourceMultiplierQ16
            : source[kScienceResourceMultiplierQ16Control]
    );
    controls[world * control_stride + 2] = static_cast<std::int64_t>(
        source[kScienceControlsApplied]
    );
}

void apply_science_controls_v1(
    torch::Tensor state,
    torch::Tensor scratch,
    torch::Tensor controls,
    std::uint64_t expected_next_absolute_tick
) {
    validate_tensors(state, scratch);
    TORCH_CHECK(
        state.size(0) > 0 && state.size(0) <= 65535,
        "scientific-control B must be in [1,65535]"
    );
    TORCH_CHECK(
        controls.is_cuda() && controls.scalar_type() == torch::kInt64 &&
        controls.is_contiguous() && controls.dim() == 2 &&
        controls.size(0) == state.size(0) && controls.size(1) == 2 &&
        controls.device() == state.device(),
        "science controls must be exact CUDA int64[B,2]"
    );
    const torch::Tensor host_controls = controls.to(torch::kCPU);
    const auto values = host_controls.accessor<std::int64_t, 2>();
    for (std::int64_t world = 0; world < host_controls.size(0); ++world) {
        TORCH_CHECK(
            values[world][0] >= 0 && values[world][0] <= 65536,
            "mutation threshold must be in [0,65536]"
        );
        TORCH_CHECK(
            values[world][1] >= 0 && values[world][1] <= 16 * 65536,
            "resource multiplier Q16 must be in [0,1048576]"
        );
    }
    c10::cuda::CUDAGuard guard(state.device());
    const auto stream = at::cuda::getCurrentCUDAStream(state.get_device());
    std::lock_guard<std::mutex> graph_lock(prepared_graph_mutex_v2);
    TORCH_CHECK(
        prepared_graph_v2.graph == nullptr &&
        prepared_graph_v2.executable == nullptr,
        "scientific controls require a fresh owner before graph preparation"
    );
    torch::Tensor statuses = torch::zeros(
        {state.size(0)},
        torch::TensorOptions().dtype(torch::kInt64).device(state.device())
    );
    validate_science_controls_v1_kernel<<<
        static_cast<unsigned>(state.size(0)), 1, 0, stream
    >>>(
        state.data_ptr<std::uint8_t>(), scratch.data_ptr<std::uint8_t>(),
        controls.data_ptr<std::int64_t>(), controls.stride(0),
        expected_next_absolute_tick, statuses.data_ptr<std::int64_t>()
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    const torch::Tensor host_statuses = statuses.to(torch::kCPU);
    const auto status_values = host_statuses.accessor<std::int64_t, 1>();
    for (std::int64_t world = 0; world < host_statuses.size(0); ++world) {
        TORCH_CHECK(
            status_values[world] == 0,
            "scientific-control application rejected world ", world,
            " with guard status ", status_values[world]
        );
    }
    apply_science_controls_v1_kernel<<<
        static_cast<unsigned>(state.size(0)), 1, 0, stream
    >>>(
        scratch.data_ptr<std::uint8_t>(), controls.data_ptr<std::int64_t>(),
        controls.stride(0)
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

torch::Tensor science_controls_readback_v1(torch::Tensor scratch) {
    TORCH_CHECK(
        scratch.is_cuda() && scratch.scalar_type() == torch::kUInt8 &&
        scratch.is_contiguous() && scratch.dim() == 2 &&
        scratch.size(0) > 0 && scratch.size(0) <= 65535 &&
        scratch.size(1) == kScratchBytes,
        "scratch must be exact CUDA uint8[B,scratch_bytes]"
    );
    c10::cuda::CUDAGuard guard(scratch.device());
    torch::Tensor result = torch::empty(
        {scratch.size(0), 3},
        torch::TensorOptions().dtype(torch::kInt64).device(scratch.device())
    );
    const auto stream = at::cuda::getCurrentCUDAStream(scratch.get_device());
    read_science_controls_v1_kernel<<<
        static_cast<unsigned>(scratch.size(0)), 1, 0, stream
    >>>(
        scratch.data_ptr<std::uint8_t>(), result.data_ptr<std::int64_t>(),
        result.stride(0)
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return result;
}

std::int64_t science_control_abi_version_v1() {
    return 1;
}

void restore_derived(torch::Tensor state, torch::Tensor scratch) {
    validate_tensors(state, scratch);
    c10::cuda::CUDAGuard guard(state.device());
    const auto stream = at::cuda::getCurrentCUDAStream(state.get_device());
    const unsigned B = static_cast<unsigned>(state.size(0));
    clear_derived_v2<<<dim3(kEnvironmentTiles, B, 1), kThreads, 0, stream>>>(
        state.data_ptr<std::uint8_t>(), scratch.data_ptr<std::uint8_t>()
    );
    restore_agents_v2<<<dim3(kAgentTiles, B, 1), kThreads, 0, stream>>>(
        state.data_ptr<std::uint8_t>()
    );
    validate_active_order_v2<<<dim3(1, B, 1), kThreads, 0, stream>>>(
        state.data_ptr<std::uint8_t>()
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    // Restore is an explicit, non-timed owner boundary. Surface validation
    // synchronously here; advance never clears these error codes.
    std::vector<std::uint64_t> statuses(B, 0);
    C10_CUDA_CHECK(cudaMemcpy2DAsync(
        statuses.data(),
        sizeof(std::uint64_t),
        state.data_ptr<std::uint8_t>() + kHeaderOffset +
            kStatus * sizeof(std::uint64_t),
        kStateBytes,
        sizeof(std::uint64_t),
        B,
        cudaMemcpyDeviceToHost,
        stream
    ));
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    for (unsigned world = 0; world < B; ++world) {
        TORCH_CHECK(
            statuses[world] == 0,
            "restore validation failed for world ", world,
            " with preserved status ", statuses[world]
        );
    }
}

std::string shape_id() {
    return "w" + std::to_string(kWidth) + "h" + std::to_string(kHeight) + "p" +
        std::to_string(kInitialPopulation) + "c" + std::to_string(kCapacity);
}

py::list temporal_k_candidates() {
    py::list values;
    values.append(1);
    if (kMaximumTemporalK >= 2) values.append(2);
    if (kMaximumTemporalK >= 4) values.append(4);
    if (kMaximumTemporalK >= 8) values.append(8);
    if (kMaximumTemporalK >= 16) values.append(16);
    if (kMaximumTemporalK >= 32) values.append(32);
    if (kMaximumTemporalK >= 64) values.append(64);
    return values;
}

py::dict compile_shape() {
    py::dict value;
    value["shape_id"] = shape_id();
    value["width"] = kWidth;
    value["height"] = kHeight;
    value["sites"] = kSites;
    value["initial_population"] = kInitialPopulation;
    value["capacity"] = kCapacity;
    value["index_width_bits"] = 32;
    value["state_bytes_per_world"] = kStateBytes;
    value["scratch_bytes_per_world"] = kScratchBytes;
    value["event_row_bytes"] = kEventRowBytes;
    value["maximum_temporal_k"] = kMaximumTemporalK;
    value["temporal_k_candidates"] = temporal_k_candidates();
    return value;
}

py::dict kernel_topology() {
    py::dict value;
    value["schema_version"] = "eiel_full_evolution_workhorse_backend_topology_v2";
    value["backend_id"] = EIEL_WORKHORSE_BACKEND_ID;
    value["kind"] = kBackendKind == 0 ? "correctness_reference" : "performance";
    value["shape_id"] = shape_id();
    value["threads"] = kThreads;
    value["world_axis"] = "grid.y";
    value["world_isolation"] = "all offsets begin at blockIdx.y world row";
    py::list kernels;
    if (kBackendKind == 2) {
        kernels.append("fused_world_temporal_v2");
        value["kernels_per_dispatch"] = kernels;
        value["edges_per_dispatch"] = 0;
        value["ctas_per_world"] = 1;
        value["temporal_k_inside_kernel"] = true;
        value["shape_strided_site_and_agent_loops"] = true;
        value["event_transition_preparation"] =
            "single_2C_sparse_page_status_invalidation_once_inside_fused_kernel_then_stable_slot_compaction_each_tick";
    } else {
        kernels.append("environment_phase_v2");
        kernels.append("agent_intent_phase_v2");
        kernels.append("movement_accept_phase_v2");
        kernels.append("movement_release_phase_v2");
        kernels.append("movement_commit_phase_v2");
        kernels.append("interaction_physiology_phase_v2");
        kernels.append("death_phase_v2");
        if (kBackendKind == 1) kernels.append("compact_death_events_v3");
        kernels.append("rank_tiles_phase_v2");
        kernels.append("rank_prefix_phase_v2");
        kernels.append("scatter_ranks_phase_v2");
        kernels.append("mate_spawn_phase_v2");
        kernels.append("birth_commit_phase_v2");
        if (kBackendKind == 1) kernels.append("compact_birth_events_v3");
        kernels.append("publish_phase_v2");
        value["kernels_per_tick"] = kernels;
        value["edges_per_tick"] = kBackendKind == 1 ? 14 : 12;
        value["site_tiles"] = kSiteTiles;
        value["agent_tiles"] = kAgentTiles;
        py::list single_cta_world_phases;
        single_cta_world_phases.append("rank_prefix_phase_v2");
        single_cta_world_phases.append("mate_spawn_phase_v2");
        if (kBackendKind == 1) {
            single_cta_world_phases.append("compact_death_events_v3");
            single_cta_world_phases.append("compact_birth_events_v3");
        }
        single_cta_world_phases.append("publish_phase_v2");
        value["single_cta_world_phases"] = single_cta_world_phases;
        value["event_transition_preparation"] =
            kBackendKind == 0
                ? "reference_k1_single_page_dense_zero"
                : "single_2C_sparse_page_status_invalidation_once_inside_first_environment_phase_then_stable_slot_compaction";
        value["capture"] = kBackendKind == 1;
        if (kBackendKind == 1) {
            value["capture_owner"] = "native_prepared_graph_v2";
            py::list boundary_nodes;
            boundary_nodes.append("event_count_memset_v2");
            value["dispatch_boundary_nodes"] = boundary_nodes;
            value["graph_kernel_nodes_per_temporal_k"] = 15;
            value["graph_edge_law"] = "linear_stream_capture_nodes_minus_one";
            value["runtime_inspection"] =
                "cudaGraphGetNodes_edges_kernel_and_function_attributes";
        }
    }
    return value;
}

py::dict compiled_kernel_resource_v2(const char* name, const void* function) {
    cudaFuncAttributes attributes{};
    C10_CUDA_CHECK(cudaFuncGetAttributes(&attributes, function));
    int active_blocks_per_sm = 0;
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &active_blocks_per_sm, function, kThreads, 0
    ));
    py::dict row;
    row["kernel"] = name;
    row["registers_per_thread"] = attributes.numRegs;
    row["static_shared_bytes"] = attributes.sharedSizeBytes;
    row["local_bytes_per_thread"] = attributes.localSizeBytes;
    row["maximum_threads_per_block"] = attributes.maxThreadsPerBlock;
    row["active_blocks_per_sm_at_compiled_block"] = active_blocks_per_sm;
    return row;
}

py::dict runtime_inspect() {
    int device = -1;
    C10_CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp properties{};
    C10_CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    C10_CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    std::ostringstream uuid;
    uuid << std::hex << std::setfill('0');
    for (unsigned char value : properties.uuid.bytes) {
        uuid << std::setw(2) << static_cast<unsigned>(value);
    }
    py::dict value;
    value["device_index"] = device;
    value["target_sm"] = "sm_" + std::to_string(properties.major) +
        std::to_string(properties.minor);
    value["multiprocessor_count"] = properties.multiProcessorCount;
    value["device_uuid"] = uuid.str();
    value["total_device_bytes"] = static_cast<std::uint64_t>(properties.totalGlobalMem);
    value["free_device_bytes"] = static_cast<std::uint64_t>(free_bytes);
    value["maximum_grid_x"] = properties.maxGridSize[0];
    value["maximum_grid_y"] = properties.maxGridSize[1];
    value["maximum_threads_per_block"] = properties.maxThreadsPerBlock;
    value["maximum_threads_per_sm"] = properties.maxThreadsPerMultiProcessor;
    value["registers_per_sm"] = properties.regsPerMultiprocessor;
    value["shared_memory_per_sm"] =
        static_cast<std::uint64_t>(properties.sharedMemPerMultiprocessor);
    value["maximum_blocks_per_sm"] = properties.maxBlocksPerMultiProcessor;
    value["warp_size"] = properties.warpSize;
    value["backend_id"] = EIEL_WORKHORSE_BACKEND_ID;
    value["profile_id"] = EIEL_WORKHORSE_PROFILE_ID;
    value["route_id"] = EIEL_WORKHORSE_ROUTE_ID;
    value["site_ctas_per_world"] = kSiteTiles;
    value["agent_ctas_per_world"] = kAgentTiles;
    value["environment_ctas_per_world"] = kEnvironmentTiles;
    py::list resources;
    if (kBackendKind == 2) {
        resources.append(compiled_kernel_resource_v2(
            "fused_world_temporal_v2",
            reinterpret_cast<const void*>(fused_world_temporal_v2)
        ));
    } else {
        resources.append(compiled_kernel_resource_v2(
            "environment_phase_v2",
            reinterpret_cast<const void*>(environment_phase_v2)
        ));
        resources.append(compiled_kernel_resource_v2(
            "agent_intent_phase_v2",
            reinterpret_cast<const void*>(agent_intent_phase_v2)
        ));
        resources.append(compiled_kernel_resource_v2(
            "movement_accept_phase_v2",
            reinterpret_cast<const void*>(movement_accept_phase_v2)
        ));
        resources.append(compiled_kernel_resource_v2(
            "movement_release_phase_v2",
            reinterpret_cast<const void*>(movement_release_phase_v2)
        ));
        resources.append(compiled_kernel_resource_v2(
            "movement_commit_phase_v2",
            reinterpret_cast<const void*>(movement_commit_phase_v2)
        ));
        resources.append(compiled_kernel_resource_v2(
            "interaction_physiology_phase_v2",
            reinterpret_cast<const void*>(interaction_physiology_phase_v2)
        ));
        resources.append(compiled_kernel_resource_v2(
            "death_phase_v2", reinterpret_cast<const void*>(death_phase_v2)
        ));
        if (kBackendKind == 1) {
            resources.append(compiled_kernel_resource_v2(
                "compact_death_events_v3",
                reinterpret_cast<const void*>(compact_phase_events_v3<1>)
            ));
        }
        resources.append(compiled_kernel_resource_v2(
            "rank_tiles_phase_v2", reinterpret_cast<const void*>(rank_tiles_phase_v2)
        ));
        resources.append(compiled_kernel_resource_v2(
            "rank_prefix_phase_v2", reinterpret_cast<const void*>(rank_prefix_phase_v2)
        ));
        resources.append(compiled_kernel_resource_v2(
            "scatter_ranks_phase_v2", reinterpret_cast<const void*>(scatter_ranks_phase_v2)
        ));
        resources.append(compiled_kernel_resource_v2(
            "mate_spawn_phase_v2", reinterpret_cast<const void*>(mate_spawn_phase_v2)
        ));
        resources.append(compiled_kernel_resource_v2(
            "birth_commit_phase_v2", reinterpret_cast<const void*>(birth_commit_phase_v2)
        ));
        if (kBackendKind == 1) {
            resources.append(compiled_kernel_resource_v2(
                "compact_birth_events_v3",
                reinterpret_cast<const void*>(compact_phase_events_v3<2>)
            ));
        }
        resources.append(compiled_kernel_resource_v2(
            "publish_phase_v2", reinterpret_cast<const void*>(publish_phase_v2)
        ));
    }
    value["compiled_kernel_resources"] = resources;
    value["runtime_inspected"] = true;
    return value;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("profile_family", [] { return "EIEL_FULL_EVOLUTION_CUDA_WORKHORSE_V2"; });
    module.def("profile_id", [] { return std::string(EIEL_WORKHORSE_PROFILE_ID); });
    module.def("route_id", [] { return std::string(EIEL_WORKHORSE_ROUTE_ID); });
    module.def("abi_version", [] { return "eiel_full_evolution_cuda_workhorse_abi_v2"; });
    module.def("event_abi_version", [] {
        return "eiel_full_evolution_workhorse_sparse_lifecycle_event_v3";
    });
    module.def("semantic_transition_id", [] { return "eiel_full_evolution_workhorse_transition_v2"; });
    module.def("semantic_transition_sha256", [] { return std::string(EIEL_WORKHORSE_SEMANTIC_HASH); });
    module.def("backend_id", [] { return std::string(EIEL_WORKHORSE_BACKEND_ID); });
    module.def("backend_topology_sha256", [] { return std::string(EIEL_WORKHORSE_TOPOLOGY_HASH); });
    module.def("layout_sha256", [] { return std::string(EIEL_WORKHORSE_LAYOUT_HASH); });
    module.def("compile_shape", &compile_shape);
    module.def("kernel_topology", &kernel_topology);
    module.def("runtime_inspect", &runtime_inspect);
    module.def("initialize", &initialize);
    module.def("initialize_exact", &initialize_exact);
    module.def("prepare_graph", &prepare_graph);
    module.def("graph_inspect", &graph_inspect);
    module.def("release_graph", &release_graph);
    module.def("advance", &advance);
    module.def("advance_diagnostic_ablation", &advance_diagnostic_ablation);
    module.def("apply_science_controls_v1", &apply_science_controls_v1);
    module.def("science_control_abi_version_v1", &science_control_abi_version_v1);
    module.def("science_controls_readback_v1", &science_controls_readback_v1);
    module.def("canonicalize_events", &canonicalize_events);
    module.def("restore_derived", &restore_derived);
    module.def("force_causal_fixture", &force_causal_fixture);
    module.def("force_birth_overflow_fixture", &force_birth_overflow_fixture);
    module.def(
        "force_middle_survivor_death_fixture",
        &force_middle_survivor_death_fixture
    );
    module.def("force_transaction_error_fixture", &force_transaction_error_fixture);
    module.def(
        "force_event_then_transaction_error_fixture",
        &force_event_then_transaction_error_fixture
    );
}
