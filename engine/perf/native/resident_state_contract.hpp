#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

#if defined(__CUDACC__)
#define EIEL_ARCHB_HOST_DEVICE __host__ __device__
#else
#define EIEL_ARCHB_HOST_DEVICE
#endif

namespace eiel::architecture_b {

inline constexpr std::uint32_t kResidentStateContractVersion = 1;
inline constexpr std::size_t kSha256Bytes = 32;
inline constexpr std::int64_t kMaxGeneration =
    static_cast<std::int64_t>(0x7FFFFFFFFFFFFFFFULL);

enum class BackendKind : std::uint32_t {
    kInvalid = 0,
    kCpuOracle = 1,
    kCudaResident = 2,
};

enum class DeviceKind : std::uint32_t {
    kInvalid = 0,
    kHostCpu = 1,
    kCuda = 2,
};

enum class CellBankLayout : std::uint32_t {
    kInvalid = 0,
    kContiguousCellMajor = 1,
};

enum class RngAlgorithm : std::uint32_t {
    kInvalid = 0,
    kPcg64 = 1,
    kCounter64 = 2,
};

enum class BoundarySnapshotPolicy : std::uint32_t {
    kInvalid = 0,
    kNoSnapshot = 1,
    kSingleImmutableSnapshot = 2,
};

enum class CapabilityFlag : std::uint64_t {
    kResidentState = 1ULL << 0,
    kCompleteGeneration = 1ULL << 1,
    kFlatWorld = 1ULL << 2,
    kPatchWorld = 1ULL << 3,
    kFixedMlpPolicy = 1ULL << 4,
    kLifecycle = 1ULL << 5,
    kCheckpointResume = 1ULL << 6,
    kExactPublicProjection = 1ULL << 7,
    kExactRngSchedule = 1ULL << 8,
    kMultiCell = 1ULL << 9,
    kImmutableBoundarySnapshot = 1ULL << 10,
    kNoHostTrajectoryCallbacks = 1ULL << 11,
    kNoHiddenFallback = 1ULL << 12,
};

using CapabilityMask = std::uint64_t;

EIEL_ARCHB_HOST_DEVICE constexpr CapabilityMask capability_mask(
    const CapabilityFlag flag
) noexcept {
    return static_cast<CapabilityMask>(flag);
}

inline constexpr CapabilityMask kKnownCapabilityMask =
    capability_mask(CapabilityFlag::kResidentState)
    | capability_mask(CapabilityFlag::kCompleteGeneration)
    | capability_mask(CapabilityFlag::kFlatWorld)
    | capability_mask(CapabilityFlag::kPatchWorld)
    | capability_mask(CapabilityFlag::kFixedMlpPolicy)
    | capability_mask(CapabilityFlag::kLifecycle)
    | capability_mask(CapabilityFlag::kCheckpointResume)
    | capability_mask(CapabilityFlag::kExactPublicProjection)
    | capability_mask(CapabilityFlag::kExactRngSchedule)
    | capability_mask(CapabilityFlag::kMultiCell)
    | capability_mask(CapabilityFlag::kImmutableBoundarySnapshot)
    | capability_mask(CapabilityFlag::kNoHostTrajectoryCallbacks)
    | capability_mask(CapabilityFlag::kNoHiddenFallback);

inline constexpr CapabilityMask kRequiredDispatchCapabilities =
    capability_mask(CapabilityFlag::kResidentState)
    | capability_mask(CapabilityFlag::kCompleteGeneration)
    | capability_mask(CapabilityFlag::kImmutableBoundarySnapshot)
    | capability_mask(CapabilityFlag::kNoHostTrajectoryCallbacks)
    | capability_mask(CapabilityFlag::kNoHiddenFallback);

// The accepted tensor manifest remains an external authority. Native adapters
// pass its schema, record count, and digest without duplicating field names here.
struct ManifestAuthority {
    std::uint32_t schema_version;
    std::uint32_t field_count;
    std::uint8_t sha256[kSha256Bytes];
};

struct BackendCapabilities {
    std::uint32_t contract_version;
    BackendKind backend;
    DeviceKind device;
    std::uint32_t device_ordinal;
    std::uint32_t max_cells;
    std::uint32_t reserved;
    CapabilityMask flags;
};

// state_address is an opaque backend-owned state/view table. The external
// manifest defines its tensor records; this contract defines dispatch metadata.
struct CellBankViewMetadata {
    std::uint32_t contract_version;
    BackendKind backend;
    DeviceKind device;
    std::uint32_t device_ordinal;
    CellBankLayout layout;
    std::uint32_t leading_cell_axis;
    std::uint32_t cell_count;
    std::uint32_t capacity_per_cell;
    std::uint32_t tensor_record_count;
    std::uint32_t reserved;
    std::uint64_t state_epoch;
    std::uintptr_t state_address;
    std::uint64_t state_bytes;
    ManifestAuthority manifest;
};

// The algorithm selects the interpretation of these fixed-width words. Keeping
// the stream state POD permits identical host and device bank descriptors.
struct RngStreamPodState {
    std::uint64_t words[4];
    std::uint64_t draw_count;
};

struct RngBankViewMetadata {
    std::uint32_t contract_version;
    RngAlgorithm algorithm;
    DeviceKind device;
    std::uint32_t device_ordinal;
    std::uint32_t cell_count;
    std::uint32_t streams_per_cell;
    std::uint64_t state_count;
    std::uintptr_t states_address;
    std::uint64_t state_stride_bytes;
    std::uint64_t state_bytes;
};

// Counters cover one complete generation. Initial import is intentionally
// outside this scope; boundary snapshot counters are separate from trajectory
// copies so one logical snapshot may contain several tensor copy operations.
struct GenerationCopyCounters {
    std::uint64_t trajectory_host_to_device_operations;
    std::uint64_t trajectory_host_to_device_bytes;
    std::uint64_t trajectory_device_to_host_operations;
    std::uint64_t trajectory_device_to_host_bytes;
    std::uint64_t trajectory_device_synchronizations;
    std::uint64_t trajectory_device_allocations;
    std::uint64_t trajectory_device_allocation_bytes;
    std::uint64_t boundary_snapshot_count;
    std::uint64_t boundary_snapshot_bytes;
    std::uint64_t boundary_device_to_host_operations;
    std::uint64_t boundary_device_to_host_bytes;
    std::uint64_t boundary_device_synchronizations;
    std::uint64_t host_trajectory_callbacks;
    std::uint64_t hidden_fallbacks;
};

struct GenerationBoundary {
    std::uint32_t contract_version;
    std::uint32_t reserved;
    std::uint64_t boundary_ordinal;
    std::int64_t completed_generation;
    std::int64_t next_generation;
    std::uint64_t completed_ticks;
    std::uint64_t state_epoch_before;
    std::uint64_t state_epoch_after;
    std::uint64_t rng_draw_count_before;
    std::uint64_t rng_draw_count_after;
    GenerationCopyCounters copies;
};

struct ResidentDispatchContract {
    BackendCapabilities capabilities;
    CellBankViewMetadata cell_bank;
    RngBankViewMetadata rng_bank;
};

enum class ContractError : std::uint32_t {
    kUninitialized = 0,
    kOk = 1,
    kContractVersionMismatch = 2,
    kUnknownBackend = 3,
    kUnknownDevice = 4,
    kBackendDeviceMismatch = 5,
    kUnknownCapability = 6,
    kMissingCapability = 7,
    kCellCountOutOfRange = 8,
    kLayoutMismatch = 9,
    kManifestAuthorityInvalid = 10,
    kManifestMismatch = 11,
    kStateViewInvalid = 12,
    kRngAlgorithmInvalid = 13,
    kRngBankInvalid = 14,
    kRngCellCountMismatch = 15,
    kBoundarySequenceInvalid = 16,
    kCounterPairInvalid = 17,
    kTrajectoryHostTransfer = 18,
    kTrajectorySynchronization = 19,
    kTrajectoryHostCallback = 20,
    kHiddenFallback = 21,
    kBoundarySnapshotInvalid = 22,
    kBackendCounterMismatch = 23,
};

struct ValidationResult {
    ContractError error;

    EIEL_ARCHB_HOST_DEVICE constexpr bool ok() const noexcept {
        return error == ContractError::kOk;
    }
};

EIEL_ARCHB_HOST_DEVICE constexpr ValidationResult valid_contract() noexcept {
    return {ContractError::kOk};
}

EIEL_ARCHB_HOST_DEVICE constexpr ValidationResult invalid_contract(
    const ContractError error
) noexcept {
    return {error};
}

EIEL_ARCHB_HOST_DEVICE constexpr bool is_known_backend(
    const BackendKind backend
) noexcept {
    return backend == BackendKind::kCpuOracle
        || backend == BackendKind::kCudaResident;
}

EIEL_ARCHB_HOST_DEVICE constexpr bool is_known_device(
    const DeviceKind device
) noexcept {
    return device == DeviceKind::kHostCpu || device == DeviceKind::kCuda;
}

EIEL_ARCHB_HOST_DEVICE constexpr bool backend_matches_device(
    const BackendKind backend,
    const DeviceKind device
) noexcept {
    return (backend == BackendKind::kCpuOracle && device == DeviceKind::kHostCpu)
        || (backend == BackendKind::kCudaResident && device == DeviceKind::kCuda);
}

EIEL_ARCHB_HOST_DEVICE constexpr bool digest_is_nonzero(
    const std::uint8_t (&digest)[kSha256Bytes]
) noexcept {
    for (std::size_t index = 0; index < kSha256Bytes; ++index) {
        if (digest[index] != 0) {
            return true;
        }
    }
    return false;
}

EIEL_ARCHB_HOST_DEVICE constexpr bool digests_equal(
    const std::uint8_t (&left)[kSha256Bytes],
    const std::uint8_t (&right)[kSha256Bytes]
) noexcept {
    for (std::size_t index = 0; index < kSha256Bytes; ++index) {
        if (left[index] != right[index]) {
            return false;
        }
    }
    return true;
}

EIEL_ARCHB_HOST_DEVICE constexpr ValidationResult validate_manifest_authority(
    const ManifestAuthority& actual,
    const ManifestAuthority& expected
) noexcept {
    if (actual.schema_version == 0 || actual.field_count == 0
        || expected.schema_version == 0 || expected.field_count == 0
        || !digest_is_nonzero(actual.sha256)
        || !digest_is_nonzero(expected.sha256)) {
        return invalid_contract(ContractError::kManifestAuthorityInvalid);
    }
    if (actual.schema_version != expected.schema_version
        || actual.field_count != expected.field_count
        || !digests_equal(actual.sha256, expected.sha256)) {
        return invalid_contract(ContractError::kManifestMismatch);
    }
    return valid_contract();
}

EIEL_ARCHB_HOST_DEVICE constexpr ValidationResult validate_backend_capabilities(
    const BackendCapabilities& capabilities,
    const CapabilityMask required_flags,
    const std::uint32_t requested_cells
) noexcept {
    if (capabilities.contract_version != kResidentStateContractVersion) {
        return invalid_contract(ContractError::kContractVersionMismatch);
    }
    if (!is_known_backend(capabilities.backend)) {
        return invalid_contract(ContractError::kUnknownBackend);
    }
    if (!is_known_device(capabilities.device)) {
        return invalid_contract(ContractError::kUnknownDevice);
    }
    if (!backend_matches_device(capabilities.backend, capabilities.device)) {
        return invalid_contract(ContractError::kBackendDeviceMismatch);
    }
    if ((capabilities.flags & ~kKnownCapabilityMask) != 0
        || (required_flags & ~kKnownCapabilityMask) != 0) {
        return invalid_contract(ContractError::kUnknownCapability);
    }
    const CapabilityMask effective_required =
        required_flags | kRequiredDispatchCapabilities;
    if ((capabilities.flags & effective_required) != effective_required) {
        return invalid_contract(ContractError::kMissingCapability);
    }
    if (requested_cells == 0 || capabilities.max_cells == 0
        || requested_cells > capabilities.max_cells) {
        return invalid_contract(ContractError::kCellCountOutOfRange);
    }
    if (requested_cells > 1
        && (capabilities.flags & capability_mask(CapabilityFlag::kMultiCell)) == 0) {
        return invalid_contract(ContractError::kMissingCapability);
    }
    return valid_contract();
}

EIEL_ARCHB_HOST_DEVICE constexpr ValidationResult validate_cell_bank_view(
    const CellBankViewMetadata& view,
    const ManifestAuthority& expected_manifest
) noexcept {
    if (view.contract_version != kResidentStateContractVersion) {
        return invalid_contract(ContractError::kContractVersionMismatch);
    }
    if (!is_known_backend(view.backend)) {
        return invalid_contract(ContractError::kUnknownBackend);
    }
    if (!is_known_device(view.device)) {
        return invalid_contract(ContractError::kUnknownDevice);
    }
    if (!backend_matches_device(view.backend, view.device)) {
        return invalid_contract(ContractError::kBackendDeviceMismatch);
    }
    if (view.layout != CellBankLayout::kContiguousCellMajor
        || view.leading_cell_axis != 0) {
        return invalid_contract(ContractError::kLayoutMismatch);
    }
    if (view.cell_count == 0 || view.capacity_per_cell == 0) {
        return invalid_contract(ContractError::kCellCountOutOfRange);
    }
    if (view.state_address == 0 || view.state_bytes == 0
        || view.tensor_record_count == 0
        || view.tensor_record_count != view.manifest.field_count) {
        return invalid_contract(ContractError::kStateViewInvalid);
    }
    return validate_manifest_authority(view.manifest, expected_manifest);
}

EIEL_ARCHB_HOST_DEVICE constexpr ValidationResult validate_rng_bank_view(
    const RngBankViewMetadata& view
) noexcept {
    if (view.contract_version != kResidentStateContractVersion) {
        return invalid_contract(ContractError::kContractVersionMismatch);
    }
    if (view.algorithm != RngAlgorithm::kPcg64
        && view.algorithm != RngAlgorithm::kCounter64) {
        return invalid_contract(ContractError::kRngAlgorithmInvalid);
    }
    if (!is_known_device(view.device)) {
        return invalid_contract(ContractError::kUnknownDevice);
    }
    if (view.cell_count == 0 || view.streams_per_cell == 0) {
        return invalid_contract(ContractError::kRngBankInvalid);
    }
    const std::uint64_t expected_state_count =
        static_cast<std::uint64_t>(view.cell_count) * view.streams_per_cell;
    const std::uint64_t expected_state_bytes =
        expected_state_count * sizeof(RngStreamPodState);
    if (view.state_count != expected_state_count || view.states_address == 0
        || view.state_stride_bytes != sizeof(RngStreamPodState)
        || view.state_bytes != expected_state_bytes
        || view.states_address % alignof(RngStreamPodState) != 0) {
        return invalid_contract(ContractError::kRngBankInvalid);
    }
    return valid_contract();
}

EIEL_ARCHB_HOST_DEVICE constexpr ValidationResult validate_dispatch_contract(
    const ResidentDispatchContract& contract,
    const ManifestAuthority& expected_manifest,
    const CapabilityMask required_flags
) noexcept {
    ValidationResult result = validate_backend_capabilities(
        contract.capabilities,
        required_flags,
        contract.cell_bank.cell_count
    );
    if (!result.ok()) {
        return result;
    }
    result = validate_cell_bank_view(contract.cell_bank, expected_manifest);
    if (!result.ok()) {
        return result;
    }
    result = validate_rng_bank_view(contract.rng_bank);
    if (!result.ok()) {
        return result;
    }
    if (contract.capabilities.backend != contract.cell_bank.backend
        || contract.capabilities.device != contract.cell_bank.device
        || contract.capabilities.device != contract.rng_bank.device
        || contract.capabilities.device_ordinal != contract.cell_bank.device_ordinal
        || contract.capabilities.device_ordinal != contract.rng_bank.device_ordinal) {
        return invalid_contract(ContractError::kBackendDeviceMismatch);
    }
    if (contract.cell_bank.cell_count != contract.rng_bank.cell_count) {
        return invalid_contract(ContractError::kRngCellCountMismatch);
    }
    return valid_contract();
}

EIEL_ARCHB_HOST_DEVICE constexpr bool count_bytes_pair_is_valid(
    const std::uint64_t count,
    const std::uint64_t bytes
) noexcept {
    return (count == 0) == (bytes == 0);
}

EIEL_ARCHB_HOST_DEVICE constexpr ValidationResult validate_generation_boundary(
    const GenerationBoundary& boundary,
    const BackendKind backend,
    const BoundarySnapshotPolicy snapshot_policy
) noexcept {
    if (boundary.contract_version != kResidentStateContractVersion) {
        return invalid_contract(ContractError::kContractVersionMismatch);
    }
    if (!is_known_backend(backend)) {
        return invalid_contract(ContractError::kUnknownBackend);
    }
    if (boundary.boundary_ordinal == 0 || boundary.completed_generation < 0
        || boundary.completed_generation == kMaxGeneration
        || boundary.next_generation != boundary.completed_generation + 1
        || boundary.completed_ticks == 0
        || boundary.state_epoch_after <= boundary.state_epoch_before
        || boundary.rng_draw_count_after < boundary.rng_draw_count_before) {
        return invalid_contract(ContractError::kBoundarySequenceInvalid);
    }
    const GenerationCopyCounters& copies = boundary.copies;
    if (!count_bytes_pair_is_valid(
            copies.trajectory_host_to_device_operations,
            copies.trajectory_host_to_device_bytes
        )
        || !count_bytes_pair_is_valid(
            copies.trajectory_device_to_host_operations,
            copies.trajectory_device_to_host_bytes
        )
        || !count_bytes_pair_is_valid(
            copies.trajectory_device_allocations,
            copies.trajectory_device_allocation_bytes
        )
        || !count_bytes_pair_is_valid(
            copies.boundary_snapshot_count,
            copies.boundary_snapshot_bytes
        )
        || !count_bytes_pair_is_valid(
            copies.boundary_device_to_host_operations,
            copies.boundary_device_to_host_bytes
        )) {
        return invalid_contract(ContractError::kCounterPairInvalid);
    }
    if (copies.trajectory_host_to_device_operations != 0
        || copies.trajectory_device_to_host_operations != 0) {
        return invalid_contract(ContractError::kTrajectoryHostTransfer);
    }
    if (copies.trajectory_device_synchronizations != 0) {
        return invalid_contract(ContractError::kTrajectorySynchronization);
    }
    if (copies.host_trajectory_callbacks != 0) {
        return invalid_contract(ContractError::kTrajectoryHostCallback);
    }
    if (copies.hidden_fallbacks != 0) {
        return invalid_contract(ContractError::kHiddenFallback);
    }
    if (snapshot_policy == BoundarySnapshotPolicy::kNoSnapshot) {
        if (copies.boundary_snapshot_count != 0
            || copies.boundary_device_to_host_operations != 0) {
            return invalid_contract(ContractError::kBoundarySnapshotInvalid);
        }
    } else if (snapshot_policy == BoundarySnapshotPolicy::kSingleImmutableSnapshot) {
        if (copies.boundary_snapshot_count != 1
            || copies.boundary_snapshot_bytes == 0) {
            return invalid_contract(ContractError::kBoundarySnapshotInvalid);
        }
        if (backend == BackendKind::kCudaResident
            && (copies.boundary_device_to_host_operations == 0
                || copies.boundary_device_to_host_bytes == 0
                || copies.boundary_device_to_host_bytes
                    > copies.boundary_snapshot_bytes)) {
            return invalid_contract(ContractError::kBoundarySnapshotInvalid);
        }
    } else {
        return invalid_contract(ContractError::kBoundarySnapshotInvalid);
    }
    if (backend == BackendKind::kCpuOracle
        && (copies.trajectory_device_allocations != 0
            || copies.boundary_device_to_host_operations != 0
            || copies.boundary_device_synchronizations != 0)) {
        return invalid_contract(ContractError::kBackendCounterMismatch);
    }
    return valid_contract();
}

static_assert(std::is_standard_layout_v<ManifestAuthority>);
static_assert(std::is_trivially_copyable_v<ManifestAuthority>);
static_assert(std::is_standard_layout_v<BackendCapabilities>);
static_assert(std::is_trivially_copyable_v<BackendCapabilities>);
static_assert(std::is_standard_layout_v<CellBankViewMetadata>);
static_assert(std::is_trivially_copyable_v<CellBankViewMetadata>);
static_assert(std::is_standard_layout_v<RngStreamPodState>);
static_assert(std::is_trivially_copyable_v<RngStreamPodState>);
static_assert(std::is_standard_layout_v<RngBankViewMetadata>);
static_assert(std::is_trivially_copyable_v<RngBankViewMetadata>);
static_assert(std::is_standard_layout_v<GenerationCopyCounters>);
static_assert(std::is_trivially_copyable_v<GenerationCopyCounters>);
static_assert(std::is_standard_layout_v<GenerationBoundary>);
static_assert(std::is_trivially_copyable_v<GenerationBoundary>);
static_assert(std::is_standard_layout_v<ResidentDispatchContract>);
static_assert(std::is_trivially_copyable_v<ResidentDispatchContract>);

}  // namespace eiel::architecture_b

#undef EIEL_ARCHB_HOST_DEVICE
