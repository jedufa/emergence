#pragma once

#include <ATen/CPUGeneratorImpl.h>
#include <ATen/ops/rand.h>
#include <ATen/ops/randn.h>
#include <numpy/random/distributions.h>
#include <torch/extension.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "cuda_architecture_b_option_d_m2_policy_shadow_v1.cuh"
#include "cuda_architecture_b_option_d_m4_policy_refill_v1.cuh"
#include "cuda_architecture_b_torch_rng_state_pack_v1.cuh"
#include "../hapb_rngv2/native/hapb_rngv2.cuh"

#ifndef EIEL_OPTION_D_M0_TELEMETRY_ENABLED
#define EIEL_OPTION_D_M0_TELEMETRY_ENABLED 0
#endif

#ifndef EIEL_OPTION_D_M2_POLICY_SHADOW_ENABLED
#define EIEL_OPTION_D_M2_POLICY_SHADOW_ENABLED 0
#endif

#ifndef EIEL_OPTION_D_M4_DEVICE_POLICY_REFILL_ENABLED
#define EIEL_OPTION_D_M4_DEVICE_POLICY_REFILL_ENABLED 0
#endif

#ifndef EIEL_OPTION_D_M7_BATCHED_CPU_POLICY_REFILL_ENABLED
#define EIEL_OPTION_D_M7_BATCHED_CPU_POLICY_REFILL_ENABLED 0
#endif

#ifndef EIEL_OPTION_D_M8_RESIDENT_POLICY_SUFFIX_REUSE_ENABLED
#define EIEL_OPTION_D_M8_RESIDENT_POLICY_SUFFIX_REUSE_ENABLED 0
#endif

#if EIEL_OPTION_D_M0_TELEMETRY_ENABLED != 0 \
    && EIEL_OPTION_D_M0_TELEMETRY_ENABLED != 1
#error "EIEL_OPTION_D_M0_TELEMETRY_ENABLED must be 0 or 1"
#endif

#if EIEL_OPTION_D_M2_POLICY_SHADOW_ENABLED != 0 \
    && EIEL_OPTION_D_M2_POLICY_SHADOW_ENABLED != 1
#error "EIEL_OPTION_D_M2_POLICY_SHADOW_ENABLED must be 0 or 1"
#endif

#if EIEL_OPTION_D_M4_DEVICE_POLICY_REFILL_ENABLED != 0 \
    && EIEL_OPTION_D_M4_DEVICE_POLICY_REFILL_ENABLED != 1
#error "EIEL_OPTION_D_M4_DEVICE_POLICY_REFILL_ENABLED must be 0 or 1"
#endif

#if EIEL_OPTION_D_M7_BATCHED_CPU_POLICY_REFILL_ENABLED != 0 \
    && EIEL_OPTION_D_M7_BATCHED_CPU_POLICY_REFILL_ENABLED != 1
#error "EIEL_OPTION_D_M7_BATCHED_CPU_POLICY_REFILL_ENABLED must be 0 or 1"
#endif

#if EIEL_OPTION_D_M8_RESIDENT_POLICY_SUFFIX_REUSE_ENABLED != 0 \
    && EIEL_OPTION_D_M8_RESIDENT_POLICY_SUFFIX_REUSE_ENABLED != 1
#error "EIEL_OPTION_D_M8_RESIDENT_POLICY_SUFFIX_REUSE_ENABLED must be 0 or 1"
#endif

#if EIEL_OPTION_D_M8_RESIDENT_POLICY_SUFFIX_REUSE_ENABLED == 1 \
    && EIEL_OPTION_D_M7_BATCHED_CPU_POLICY_REFILL_ENABLED != 1
#error "Option D M8 suffix reuse requires the exact M7 physical-refill fallback"
#endif

#if EIEL_OPTION_D_M4_DEVICE_POLICY_REFILL_ENABLED == 1 \
    && EIEL_OPTION_D_M7_BATCHED_CPU_POLICY_REFILL_ENABLED == 1
#error "Option D M4 and M7 policy-refill candidates are mutually exclusive"
#endif

namespace eiel::cuda_architecture_b::resident_window_refill_v1 {

namespace torch_rng_pack =
    eiel::cuda_architecture_b::torch_rng_state_pack_v1;
namespace option_d_m2_policy_shadow =
    eiel::cuda_architecture_b::option_d_m2_policy_shadow_v1;
namespace option_d_m4_policy_refill =
    eiel::cuda_architecture_b::option_d_m4_policy_refill_v1;

constexpr int64_t kCpythonStateWordsV1 = 624;
constexpr int64_t kCpythonGaussCapacityLimitV1 = 10549248;
constexpr int64_t kPolicyEpisodeCapacityLimitV1 = 20604;
inline constexpr bool kOptionDM0TelemetryEnabledV1 =
    EIEL_OPTION_D_M0_TELEMETRY_ENABLED == 1;
inline constexpr bool kOptionDM2PolicyShadowEnabledV1 =
    EIEL_OPTION_D_M2_POLICY_SHADOW_ENABLED == 1;
inline constexpr bool kOptionDM4DevicePolicyRefillEnabledV1 =
    EIEL_OPTION_D_M4_DEVICE_POLICY_REFILL_ENABLED == 1;
inline constexpr bool kOptionDM7BatchedCpuPolicyRefillEnabledV1 =
    EIEL_OPTION_D_M7_BATCHED_CPU_POLICY_REFILL_ENABLED == 1;
inline constexpr bool kOptionDM8ResidentPolicySuffixReuseEnabledV1 =
    EIEL_OPTION_D_M8_RESIDENT_POLICY_SUFFIX_REUSE_ENABLED == 1;
// SHA-256("eiel_hapb_frag_identity_compact_policy_v1").  This identifies
// the exact producer schema/encoding used by the repair; native build identity
// remains independently bound by the extension manifest.
inline constexpr const char* kFragIdentityProducerSchemaSha256V1 =
    "70a4cce6bb169effb4a8def5f4437e3ea23f57552ce0fcab8bf681d0d1c2ab6f";

inline int64_t steady_ns_v1() {
    if constexpr (kOptionDM0TelemetryEnabledV1) {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()
        ).count();
    }
    return 0;
}

inline bool hapb_frag_a_force_physical_policy_refill_v1() {
    const char* const value =
        std::getenv("EIEL_HAPB_FRAG_A_FORCE_PHYSICAL_POLICY_REFILL");
    return value != nullptr && value[0] == '1' && value[1] == '\0';
}

inline bool hapb_rngv2_policy_window_enabled_v1() {
    const char* const value = std::getenv("EIEL_HAPB_RNG_SEMANTICS_VERSION");
    return value != nullptr && value[0] == '2' && value[1] == '\0';
}

// 213/214 reuse-on H4 failed G3≠checkpoint (cells 23 then 33). 215 with
// FORCE_PHYSICAL=1 matched Gate-6 oracle 203. Production reuse is opt-in.
inline bool hapb_frag_a_policy_window_reuse_enabled_v1() {
    const char* const enable =
        std::getenv("EIEL_HAPB_FRAG_A_ENABLE_POLICY_WINDOW_REUSE");
    if (enable == nullptr || enable[0] != '1' || enable[1] != '\0') {
        return false;
    }
    return !hapb_frag_a_force_physical_policy_refill_v1();
}

inline bool hapb_frag_identity_trace_enabled_v1() {
    const char* const value =
        std::getenv("EIEL_HAPB_FRAG_IDENTITY_TRACE");
    return value != nullptr && value[0] == '1' && value[1] == '\0';
}

inline uint64_t hapb_frag_identity_fnv1a64_v1(
    const uint8_t* const bytes,
    const int64_t byte_count
) {
    uint64_t value = UINT64_C(14695981039346656037);
    for (int64_t index = 0; index < byte_count; ++index) {
        value ^= static_cast<uint64_t>(bytes[index]);
        value *= UINT64_C(1099511628211);
    }
    return value;
}

inline bool hapb_frag_a_policy_window_reuse_requested_v1() {
    const char* const enable =
        std::getenv("EIEL_HAPB_FRAG_A_ENABLE_POLICY_WINDOW_REUSE");
    return enable != nullptr && enable[0] == '1' && enable[1] == '\0';
}

struct PolicyWindowTensorIdentityV1 {
    const void* storage_impl = nullptr;
    const void* data = nullptr;
    int64_t storage_offset = 0;
    int32_t device_type = -1;
    int32_t device_index = -1;
    int32_t scalar_type = -1;
    uint32_t version = 0;
    std::vector<int64_t> sizes;
    std::vector<int64_t> strides;
};

struct PolicyWindowConfigurationIdentityV1 {
    int64_t cells = 0;
    int64_t capacity = 0;
    int64_t policy_row_width = 0;
    int64_t mutation_width = 0;
    bool mutation_enable = false;
    std::string mutation_mode;
    double mutation_rate = 0.0;
    double mutation_std = 0.0;
    std::string producer_schema_sha256;
    std::string crossover_encoding = "bitpack_lsb0_v1";
    std::string rng_replay_abi = torch_rng_pack::kAbiFingerprintSha256;
};

enum class PolicyWindowPublicationOriginV1 : uint8_t {
    kUninitialized = 0,
    kAllocatorHydration = 1,
    kPhysicalRefill = 2,
};

// Transient proof that one resident allocation owns the bytes addressed by
// window_base/cursor.  It is intentionally absent from the tensor ABI and
// checkpoints.  A new binding/owner receives a new instance.
struct PolicyWindowIdentityStateV1 {
    bool initialized = false;
    int64_t tape_storage_generation = 0;
    int64_t rng_lineage_epoch = 0;
    std::vector<uint8_t> window_valid;
    std::vector<int64_t> window_epoch;
    std::vector<int64_t> window_rng_lineage_epoch;
    std::vector<int64_t> window_base;
    std::vector<int64_t> window_base_semantic_draws;
    PolicyWindowConfigurationIdentityV1 configuration;
    std::array<PolicyWindowTensorIdentityV1, 9> tensors;
    const void* execution_lease_crossover_storage_impl = nullptr;
    PolicyWindowPublicationOriginV1 publication_origin =
        PolicyWindowPublicationOriginV1::kUninitialized;
};

// Non-serializable owner retained only by one prepared HAPB execution owner.
// The lease rejects concurrent/re-entrant use while native code has released
// the GIL, closing the admitted writer set across same-owner route re-entry.
struct PolicyWindowIdentityOwnerV1 {
    PolicyWindowIdentityStateV1 state;
    std::shared_ptr<std::mutex> execution_lease;
    int64_t entry_count = 0;
    int64_t last_issued_validation_token = 0;
    int64_t pending_validation_token = 0;
    std::string allocator_boundary_marker;
    std::string next_boundary_marker;
};

inline constexpr const char* kPolicyWindowIdentityOwnerCapsuleNameV1 =
    "eiel.hapb.policy_window_identity_owner.v1";

inline thread_local PolicyWindowIdentityStateV1*
    active_policy_window_identity_state_v1 = nullptr;
inline thread_local std::string active_policy_window_identity_boundary_v1 =
    "public_untrusted_entry";

inline std::mutex policy_window_storage_lease_registry_mutex_v1;
inline std::unordered_map<const void*, std::weak_ptr<std::mutex>>
    policy_window_storage_lease_registry_v1;

inline std::shared_ptr<std::mutex> policy_window_storage_execution_lease_v1(
    const torch::Tensor& crossover_mask_packed
) {
    TORCH_CHECK(
        crossover_mask_packed.defined(),
        "policy-window storage lease requires a defined crossover tape"
    );
    const void* const key =
        crossover_mask_packed.storage().unsafeGetStorageImpl();
    TORCH_CHECK(key != nullptr, "policy-window crossover storage is null");
    const std::lock_guard<std::mutex> guard(
        policy_window_storage_lease_registry_mutex_v1
    );
    auto& weak = policy_window_storage_lease_registry_v1[key];
    auto lease = weak.lock();
    if (!lease) {
        lease = std::make_shared<std::mutex>();
        weak = lease;
    }
    return lease;
}

class ScopedPolicyWindowIdentityStateV1 {
public:
    explicit ScopedPolicyWindowIdentityStateV1(
        PolicyWindowIdentityStateV1* const state,
        std::string boundary = active_policy_window_identity_boundary_v1
    ) : previous_state_(active_policy_window_identity_state_v1),
        previous_boundary_(active_policy_window_identity_boundary_v1) {
        active_policy_window_identity_state_v1 = state;
        active_policy_window_identity_boundary_v1 = std::move(boundary);
    }

    ~ScopedPolicyWindowIdentityStateV1() {
        active_policy_window_identity_state_v1 = previous_state_;
        active_policy_window_identity_boundary_v1 =
            std::move(previous_boundary_);
    }

    ScopedPolicyWindowIdentityStateV1(
        const ScopedPolicyWindowIdentityStateV1&
    ) = delete;
    ScopedPolicyWindowIdentityStateV1& operator=(
        const ScopedPolicyWindowIdentityStateV1&
    ) = delete;

private:
    PolicyWindowIdentityStateV1* previous_state_;
    std::string previous_boundary_;
};

inline PolicyWindowIdentityOwnerV1* policy_window_identity_owner_from_capsule_v1(
    const pybind11::handle capsule
) {
    TORCH_CHECK(
        PyCapsule_CheckExact(capsule.ptr()),
        "HAPB policy-window identity owner must be an opaque capsule"
    );
    const char* const name = PyCapsule_GetName(capsule.ptr());
    TORCH_CHECK(
        name != nullptr
            && std::string_view(name)
                == kPolicyWindowIdentityOwnerCapsuleNameV1,
        "HAPB policy-window identity owner capsule has the wrong type"
    );
    auto* const owner = static_cast<PolicyWindowIdentityOwnerV1*>(
        PyCapsule_GetPointer(
            capsule.ptr(), kPolicyWindowIdentityOwnerCapsuleNameV1
        )
    );
    TORCH_CHECK(owner != nullptr, "HAPB policy-window identity owner is null");
    return owner;
}

inline std::mutex& policy_window_identity_owner_execution_mutex_v1(
    PolicyWindowIdentityOwnerV1* const owner
) {
    TORCH_CHECK(
        owner != nullptr && owner->execution_lease != nullptr,
        "HAPB policy-window identity owner has no storage lease"
    );
    return *owner->execution_lease;
}

class PolicyWindowIdentityOwnerExecutionLeaseV1 {
public:
    explicit PolicyWindowIdentityOwnerExecutionLeaseV1(
        PolicyWindowIdentityOwnerV1* const owner
    ) : owner_(owner), lock_(
            policy_window_identity_owner_execution_mutex_v1(owner),
            std::try_to_lock
        ) {
        TORCH_CHECK(
            lock_.owns_lock(),
            "HAPB policy-window identity owner is already executing"
        );
        TORCH_CHECK(
            owner_->entry_count < std::numeric_limits<int64_t>::max(),
            "HAPB policy-window identity entry count overflow"
        );
        if (owner_->pending_validation_token != 0) {
            std::fill(
                owner_->state.window_valid.begin(),
                owner_->state.window_valid.end(),
                0U
            );
            owner_->pending_validation_token = 0;
            owner_->next_boundary_marker =
                "unvalidated_prior_entry_reentry";
        }
        boundary_ = owner_->entry_count == 0
            ? owner_->allocator_boundary_marker
            : (
                owner_->next_boundary_marker.empty()
                ? "same_owner_route_reentry"
                : owner_->next_boundary_marker
            );
        owner_->next_boundary_marker.clear();
        TORCH_CHECK(
            owner_->last_issued_validation_token
                < std::numeric_limits<int64_t>::max(),
            "HAPB policy-window validation token overflow"
        );
        entry_validation_token_ = ++owner_->last_issued_validation_token;
        ++owner_->entry_count;
    }

    ~PolicyWindowIdentityOwnerExecutionLeaseV1() {
        if (!native_returned_) {
            std::fill(
                owner_->state.window_valid.begin(),
                owner_->state.window_valid.end(),
                0U
            );
            owner_->pending_validation_token = 0;
            owner_->next_boundary_marker = "native_exception_reentry";
        }
    }

    PolicyWindowIdentityStateV1* state() const { return &owner_->state; }
    const std::string& boundary() const { return boundary_; }
    int64_t mark_native_returned_pending_python_validation() {
        TORCH_CHECK(
            owner_->pending_validation_token == 0,
            "HAPB policy-window identity has a pending validation result"
        );
        owner_->pending_validation_token = entry_validation_token_;
        native_returned_ = true;
        return entry_validation_token_;
    }

private:
    PolicyWindowIdentityOwnerV1* owner_;
    std::unique_lock<std::mutex> lock_;
    std::string boundary_;
    int64_t entry_validation_token_ = 0;
    bool native_returned_ = false;
};

inline void commit_policy_window_identity_owner_entry_v1(
    const pybind11::handle capsule,
    const int64_t validation_token
) {
    auto* const owner = policy_window_identity_owner_from_capsule_v1(capsule);
    const std::unique_lock<std::mutex> lock(
        policy_window_identity_owner_execution_mutex_v1(owner),
        std::try_to_lock
    );
    TORCH_CHECK(
        lock.owns_lock(),
        "HAPB policy-window identity validation raced an active execution"
    );
    TORCH_CHECK(
        validation_token > 0
            && owner->pending_validation_token == validation_token,
        "HAPB policy-window identity validation token is stale or absent"
    );
    owner->pending_validation_token = 0;
}

inline void require_policy_window_execution_lease_storage_v1(
    const PolicyWindowIdentityStateV1* const state,
    const torch::Tensor& crossover_mask_packed
) {
    if (
        state == nullptr
        || state->execution_lease_crossover_storage_impl == nullptr
    ) {
        return;
    }
    TORCH_CHECK(
        crossover_mask_packed.defined()
            && crossover_mask_packed.storage().unsafeGetStorageImpl()
                == state->execution_lease_crossover_storage_impl,
        "HAPB policy-window live crossover storage differs from its execution lease"
    );
}

inline const char* policy_window_publication_origin_name_v1(
    const PolicyWindowPublicationOriginV1 origin
) {
    switch (origin) {
        case PolicyWindowPublicationOriginV1::kUninitialized:
            return "uninitialized";
        case PolicyWindowPublicationOriginV1::kAllocatorHydration:
            return "allocator_hydration_adopted";
        case PolicyWindowPublicationOriginV1::kPhysicalRefill:
            return "physical_refill_published";
    }
    TORCH_CHECK(false, "unknown policy-window publication origin");
    return "unknown";
}

inline PolicyWindowTensorIdentityV1 policy_window_tensor_identity_v1(
    const torch::Tensor& tensor
) {
    TORCH_CHECK(tensor.defined(), "policy-window identity tensor is undefined");
    PolicyWindowTensorIdentityV1 out;
    out.storage_impl = tensor.storage().unsafeGetStorageImpl();
    out.data = tensor.data_ptr();
    out.storage_offset = tensor.storage_offset();
    out.device_type = static_cast<int32_t>(tensor.device().type());
    out.device_index = static_cast<int32_t>(tensor.device().index());
    out.scalar_type = static_cast<int32_t>(tensor.scalar_type());
    out.version = tensor.unsafeGetTensorImpl()
        ->version_counter().current_version();
    out.sizes.assign(tensor.sizes().begin(), tensor.sizes().end());
    out.strides.assign(tensor.strides().begin(), tensor.strides().end());
    return out;
}

inline bool policy_window_tensor_identity_matches_v1(
    const PolicyWindowTensorIdentityV1& expected,
    const torch::Tensor& actual
) {
    if (!actual.defined()) {
        return false;
    }
    return expected.storage_impl == actual.storage().unsafeGetStorageImpl()
        && expected.data == actual.data_ptr()
        && expected.storage_offset == actual.storage_offset()
        && expected.device_type
            == static_cast<int32_t>(actual.device().type())
        && expected.device_index
            == static_cast<int32_t>(actual.device().index())
        && expected.scalar_type
            == static_cast<int32_t>(actual.scalar_type())
        && expected.version == actual.unsafeGetTensorImpl()
            ->version_counter().current_version()
        && expected.sizes.size()
            == static_cast<std::size_t>(actual.dim())
        && std::equal(
            expected.sizes.begin(),
            expected.sizes.end(),
            actual.sizes().begin()
        )
        && expected.strides.size()
            == static_cast<std::size_t>(actual.dim())
        && std::equal(
            expected.strides.begin(),
            expected.strides.end(),
            actual.strides().begin()
        );
}

inline std::array<torch::Tensor, 9> policy_window_identity_tensors_v1(
    const torch::Tensor& source_state,
    const torch::Tensor& delta,
    const torch::Tensor& mutation_mask,
    const torch::Tensor& crossover_mask_packed,
    const torch::Tensor& post_state_packed,
    const torch::Tensor& semantic_tape,
    const torch::Tensor& absolute_cursor,
    const torch::Tensor& semantic_counter,
    const torch::Tensor& window_base
) {
    return {
        source_state,
        delta,
        mutation_mask,
        crossover_mask_packed,
        post_state_packed,
        semantic_tape,
        absolute_cursor,
        semantic_counter,
        window_base,
    };
}

inline PolicyWindowConfigurationIdentityV1
policy_window_configuration_identity_v1(
    const torch::Tensor& source_state,
    const torch::Tensor& delta,
    const int64_t policy_row_width,
    const bool mutation_enable,
    const std::string& mutation_mode,
    const double mutation_rate,
    const double mutation_std,
    const std::string& execution_route_sha256
) {
    PolicyWindowConfigurationIdentityV1 out;
    out.cells = source_state.size(0);
    out.capacity = delta.size(1);
    out.policy_row_width = policy_row_width;
    out.mutation_width = delta.size(2);
    out.mutation_enable = mutation_enable;
    out.mutation_mode = mutation_mode;
    out.mutation_rate = mutation_rate;
    out.mutation_std = mutation_std;
    out.producer_schema_sha256 = execution_route_sha256;
    return out;
}

inline bool policy_window_configuration_identity_matches_v1(
    const PolicyWindowConfigurationIdentityV1& expected,
    const PolicyWindowConfigurationIdentityV1& actual
) {
    return expected.cells == actual.cells
        && expected.capacity == actual.capacity
        && expected.policy_row_width == actual.policy_row_width
        && expected.mutation_width == actual.mutation_width
        && expected.mutation_enable == actual.mutation_enable
        && expected.mutation_mode == actual.mutation_mode
        && expected.mutation_rate == actual.mutation_rate
        && expected.mutation_std == actual.mutation_std
        && expected.producer_schema_sha256 == actual.producer_schema_sha256
        && expected.crossover_encoding == actual.crossover_encoding
        && expected.rng_replay_abi == actual.rng_replay_abi;
}

inline void invalidate_policy_window_identity_v1(
    PolicyWindowIdentityStateV1* const state,
    const bool rotate_rng_lineage = false,
    const bool rotate_tape_generation = false
) {
    if (state == nullptr) {
        return;
    }
    std::fill(state->window_valid.begin(), state->window_valid.end(), 0U);
    if (rotate_rng_lineage) {
        TORCH_CHECK(
            state->rng_lineage_epoch < std::numeric_limits<int64_t>::max(),
            "policy-window RNG lineage epoch overflow"
        );
        state->rng_lineage_epoch = std::max<int64_t>(
            1, state->rng_lineage_epoch + 1
        );
    }
    if (rotate_tape_generation) {
        TORCH_CHECK(
            state->tape_storage_generation
                < std::numeric_limits<int64_t>::max(),
            "policy-window tape generation overflow"
        );
        state->tape_storage_generation = std::max<int64_t>(
            1, state->tape_storage_generation + 1
        );
    }
}

inline void publish_policy_window_identity_v1(
    PolicyWindowIdentityStateV1* const state,
    const std::array<torch::Tensor, 9>& tensors,
    const PolicyWindowConfigurationIdentityV1& configuration,
    const std::vector<int64_t>& cursor,
    const std::vector<int64_t>& semantic_draws,
    const PolicyWindowPublicationOriginV1 publication_origin =
        PolicyWindowPublicationOriginV1::kPhysicalRefill
) {
    if (state == nullptr) {
        return;
    }
    const int64_t cells = configuration.cells;
    TORCH_CHECK(
        cells > 0
            && cursor.size() == static_cast<std::size_t>(cells)
            && semantic_draws.size() == static_cast<std::size_t>(cells),
        "policy-window identity publication shape mismatch"
    );
    state->initialized = true;
    state->tape_storage_generation = std::max<int64_t>(
        1, state->tape_storage_generation
    );
    state->rng_lineage_epoch = std::max<int64_t>(1, state->rng_lineage_epoch);
    state->configuration = configuration;
    for (std::size_t index = 0; index < tensors.size(); ++index) {
        state->tensors[index] = policy_window_tensor_identity_v1(tensors[index]);
    }
    state->window_valid.assign(static_cast<std::size_t>(cells), 0U);
    state->window_epoch.assign(
        static_cast<std::size_t>(cells), state->tape_storage_generation
    );
    state->window_rng_lineage_epoch.assign(
        static_cast<std::size_t>(cells), state->rng_lineage_epoch
    );
    state->window_base = cursor;
    state->window_base_semantic_draws = semantic_draws;
    state->publication_origin = publication_origin;
    // Validity is published last, after every descriptor and epoch is bound.
    std::fill(state->window_valid.begin(), state->window_valid.end(), 1U);
}

inline void require_batched_tensor_v1(
    const torch::Tensor& tensor,
    const torch::Tensor& reference,
    torch::ScalarType dtype,
    const std::vector<int64_t>& shape,
    const char* name
);

inline pybind11::list int64_values_to_list_v1(
    const std::vector<int64_t>& values
);

struct Pcg64StateV1 {
    unsigned __int128 state;
    unsigned __int128 increment;
    uint32_t has_uint32;
    uint32_t buffered_uint32;
};

class Pcg64V1 {
public:
    explicit Pcg64V1(Pcg64StateV1 state) : state_(state) {}

    uint64_t raw64() {
        constexpr unsigned __int128 multiplier =
            (static_cast<unsigned __int128>(2549297995355413924ULL) << 64U)
            | static_cast<unsigned __int128>(4865540595714422341ULL);
        state_.state = state_.state * multiplier + state_.increment;
        const uint64_t low = static_cast<uint64_t>(state_.state);
        const uint64_t high = static_cast<uint64_t>(state_.state >> 64U);
        const uint64_t xorshifted = high ^ low;
        const uint32_t rotation = static_cast<uint32_t>(high >> 58U);
        return rotation == 0U
            ? xorshifted
            : (xorshifted >> rotation)
                | (xorshifted << ((64U - rotation) & 63U));
    }

    uint32_t raw32() {
        if (state_.has_uint32 != 0U) {
            state_.has_uint32 = 0U;
            return state_.buffered_uint32;
        }
        const uint64_t raw = raw64();
        state_.buffered_uint32 = static_cast<uint32_t>(raw >> 32U);
        state_.has_uint32 = 1U;
        return static_cast<uint32_t>(raw);
    }

    double double01() {
        return static_cast<double>(raw64() >> 11U)
            * (1.0 / 9007199254740992.0);
    }

    uint64_t numpy_interval(const uint64_t maximum) {
        auto bitgen = bit_generator();
        return random_interval(&bitgen, maximum);
    }

    std::vector<float> numpy_standard_normal_float32(const int64_t count) {
        TORCH_CHECK(count >= 0, "normal sample count must be nonnegative");
        std::vector<double> values64(static_cast<std::size_t>(count), 0.0);
        if (count > 0) {
            auto bitgen = bit_generator();
            random_standard_normal_fill(
                &bitgen,
                static_cast<npy_intp>(count),
                values64.data()
            );
        }
        std::vector<float> values32(static_cast<std::size_t>(count), 0.0f);
        for (std::size_t index = 0; index < values32.size(); ++index) {
            values32[index] = static_cast<float>(values64[index]);
        }
        return values32;
    }

private:
    static uint64_t next_uint64(void* state) {
        return static_cast<Pcg64V1*>(state)->raw64();
    }

    static uint32_t next_uint32(void* state) {
        return static_cast<Pcg64V1*>(state)->raw32();
    }

    static double next_double(void* state) {
        return static_cast<Pcg64V1*>(state)->double01();
    }

    static uint64_t next_raw(void* state) {
        return static_cast<Pcg64V1*>(state)->raw64();
    }

    bitgen_t bit_generator() {
        bitgen_t bitgen;
        bitgen.state = this;
        bitgen.next_uint64 = &next_uint64;
        bitgen.next_uint32 = &next_uint32;
        bitgen.next_double = &next_double;
        bitgen.next_raw = &next_raw;
        return bitgen;
    }

    Pcg64StateV1 state_;
};

inline Pcg64StateV1 pcg64_state_v1(
    const uint64_t* words,
    const int64_t* buffer
) {
    TORCH_CHECK((words[2] & 1ULL) == 1ULL, "runner PCG64 increment must be odd");
    TORCH_CHECK(buffer[0] == 0 || buffer[0] == 1, "runner PCG64 buffer flag must be binary");
    TORCH_CHECK(
        buffer[1] >= 0
            && static_cast<uint64_t>(buffer[1]) <= 0xffffffffULL,
        "runner PCG64 buffered uint32 is outside range"
    );
    return {
        (static_cast<unsigned __int128>(words[1]) << 64U) | words[0],
        (static_cast<unsigned __int128>(words[3]) << 64U) | words[2],
        static_cast<uint32_t>(buffer[0]),
        static_cast<uint32_t>(buffer[1]),
    };
}

inline pybind11::dict refill_drive_noise_window_v1(
    const torch::Tensor& active_agent_ids,
    const torch::Tensor& active_count,
    const torch::Tensor& agent_id_to_slot,
    const torch::Tensor& alive_mask,
    const torch::Tensor& drive_noise,
    const torch::Tensor& runner_state_words,
    const torch::Tensor& runner_buffer_words,
    torch::Tensor samples,
    torch::Tensor samples_valid
) {
    TORCH_CHECK(
        active_agent_ids.defined() && active_agent_ids.dim() == 2
            && active_agent_ids.scalar_type() == torch::kInt64,
        "active_agent_ids must be int64[cells,capacity]"
    );
    const int64_t cells = active_agent_ids.size(0);
    const int64_t capacity = active_agent_ids.size(1);
    TORCH_CHECK(cells > 0 && capacity > 0, "drive-noise dimensions must be positive");
    require_batched_tensor_v1(
        active_count,
        active_agent_ids,
        torch::kInt64,
        {cells},
        "active_count"
    );
    TORCH_CHECK(
        agent_id_to_slot.defined() && agent_id_to_slot.dim() == 2
            && agent_id_to_slot.size(0) == cells
            && agent_id_to_slot.scalar_type() == torch::kInt32
            && agent_id_to_slot.device() == active_agent_ids.device(),
        "agent_id_to_slot must be int32[cells,id_capacity] on the resident device"
    );
    require_batched_tensor_v1(
        alive_mask,
        active_agent_ids,
        torch::kBool,
        {cells, capacity},
        "alive_mask"
    );
    require_batched_tensor_v1(
        drive_noise,
        active_agent_ids,
        torch::kFloat32,
        {cells, capacity},
        "drive_noise"
    );
    require_batched_tensor_v1(
        runner_state_words,
        active_agent_ids,
        torch::kUInt64,
        {cells, 4},
        "runner_pcg64_state_words"
    );
    require_batched_tensor_v1(
        runner_buffer_words,
        active_agent_ids,
        torch::kInt64,
        {cells, 2},
        "runner_pcg64_buffer_words"
    );
    TORCH_CHECK(
        samples.defined() && samples.dim() == 4
            && samples.size(0) == cells && samples.size(1) > 0
            && samples.size(2) == capacity && samples.size(3) == 4
            && samples.scalar_type() == torch::kFloat32
            && samples.device() == active_agent_ids.device()
            && samples.is_contiguous() && !samples.requires_grad(),
        "drive-noise samples must be contiguous float32[cells,ticks,capacity,4]"
    );
    const int64_t ticks = samples.size(1);
    require_batched_tensor_v1(
        samples_valid,
        active_agent_ids,
        torch::kBool,
        {cells, ticks, capacity},
        "drive_noise_samples_valid"
    );

    const auto active_host = active_agent_ids.detach().cpu().contiguous();
    const auto count_host = active_count.detach().cpu().contiguous();
    const auto map_host = agent_id_to_slot.detach().cpu().contiguous();
    const auto alive_host = alive_mask.detach().cpu().contiguous();
    const auto noise_host = drive_noise.detach().cpu().contiguous();
    const auto state_host = runner_state_words.detach().cpu().contiguous();
    const auto buffer_host = runner_buffer_words.detach().cpu().contiguous();
    auto staged_samples = torch::zeros(
        {cells, ticks, capacity, 4},
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU)
    );
    auto staged_valid = torch::zeros(
        {cells, ticks, capacity},
        torch::TensorOptions().dtype(torch::kBool).device(torch::kCPU)
    );
    const auto* counts = count_host.data_ptr<int64_t>();
    const auto* active = active_host.data_ptr<int64_t>();
    const auto* slots = map_host.data_ptr<int32_t>();
    const auto* alive = alive_host.data_ptr<bool>();
    const auto* noise = noise_host.data_ptr<float>();
    const auto* state = state_host.data_ptr<uint64_t>();
    const auto* buffer = buffer_host.data_ptr<int64_t>();
    auto* staged = staged_samples.data_ptr<float>();
    auto* valid = staged_valid.data_ptr<bool>();
    const int64_t id_capacity = agent_id_to_slot.size(1);
    std::vector<int64_t> active_counts;
    active_counts.reserve(static_cast<std::size_t>(cells));
    for (int64_t cell = 0; cell < cells; ++cell) {
        const int64_t count = counts[cell];
        TORCH_CHECK(count >= 0 && count <= capacity, "active_count is outside capacity");
        active_counts.push_back(count);
        bool any_noise = false;
        std::vector<int64_t> base_ids(static_cast<std::size_t>(count));
        for (int64_t row = 0; row < count; ++row) {
            const int64_t agent_id = active[cell * capacity + row];
            TORCH_CHECK(
                agent_id >= 0 && agent_id < id_capacity,
                "active agent id is outside agent_id_to_slot capacity"
            );
            const int32_t slot = slots[cell * id_capacity + agent_id];
            TORCH_CHECK(
                slot >= 0 && static_cast<int64_t>(slot) < capacity
                    && alive[cell * capacity + slot],
                "active agent does not resolve to one live slot"
            );
            base_ids[static_cast<std::size_t>(row)] = agent_id;
            any_noise = any_noise || noise[cell * capacity + slot] > 0.0f;
        }
        if (!any_noise) {
            continue;
        }
        Pcg64V1 rng(pcg64_state_v1(state + cell * 4, buffer + cell * 2));
        for (int64_t tick = 0; tick < ticks; ++tick) {
            auto ordered_ids = base_ids;
            for (int64_t row = count - 1; row > 0; --row) {
                const int64_t other = static_cast<int64_t>(
                    rng.numpy_interval(static_cast<uint64_t>(row))
                );
                std::swap(
                    ordered_ids[static_cast<std::size_t>(row)],
                    ordered_ids[static_cast<std::size_t>(other)]
                );
            }
            const auto normals = rng.numpy_standard_normal_float32(count * 4);
            for (int64_t row = 0; row < count; ++row) {
                const int64_t agent_id = ordered_ids[static_cast<std::size_t>(row)];
                const int64_t slot = slots[cell * id_capacity + agent_id];
                const int64_t valid_index =
                    (cell * ticks + tick) * capacity + slot;
                valid[valid_index] = true;
                const int64_t sample_index = valid_index * 4;
                for (int64_t drive = 0; drive < 4; ++drive) {
                    staged[sample_index + drive] = normals[
                        static_cast<std::size_t>(row * 4 + drive)
                    ];
                }
            }
        }
    }
    samples.copy_(staged_samples);
    samples_valid.copy_(staged_valid);

    pybind11::dict out;
    out["schema_version"] =
        "eiel_cuda_architecture_b_native_drive_noise_window_refill_v1";
    out["operation"] = "refill_drive_noise_window_v1";
    out["batch_cells"] = cells;
    out["physical_tick_capacity"] = ticks;
    out["active_counts"] = int64_values_to_list_v1(active_counts);
    out["source_rng_state_preserved"] = true;
    out["native_owner"] = true;
    return out;
}

class CpythonMt19937V1 {
public:
    CpythonMt19937V1(const torch::Tensor& words, const int64_t index) {
        TORCH_CHECK(!words.is_cuda(), "CPython random words must be CPU resident");
        TORCH_CHECK(words.scalar_type() == torch::kUInt32, "CPython random words must be uint32");
        TORCH_CHECK(words.is_contiguous(), "CPython random words must be contiguous");
        TORCH_CHECK(
            words.dim() == 1 && words.size(0) == kCpythonStateWordsV1,
            "CPython random words must have shape [624]"
        );
        TORCH_CHECK(index >= 0 && index <= kCpythonStateWordsV1, "CPython random index must be in [0, 624]");
        const auto* source = words.data_ptr<uint32_t>();
        std::copy(source, source + kCpythonStateWordsV1, state_.begin());
        index_ = static_cast<int32_t>(index);
    }

    uint32_t next_u32() {
        if (index_ >= kCpythonStateWordsV1) {
            twist();
        }
        uint32_t value = state_[static_cast<std::size_t>(index_++)];
        value ^= value >> 11U;
        value ^= (value << 7U) & 0x9d2c5680U;
        value ^= (value << 15U) & 0xefc60000U;
        value ^= value >> 18U;
        return value;
    }

    int64_t index() const {
        return static_cast<int64_t>(index_);
    }

private:
    void twist() {
        constexpr uint32_t matrix_a = 0x9908b0dfU;
        constexpr uint32_t upper_mask = 0x80000000U;
        constexpr uint32_t lower_mask = 0x7fffffffU;
        for (std::size_t offset = 0; offset < 227; ++offset) {
            const uint32_t mixed =
                (state_[offset] & upper_mask) | (state_[offset + 1] & lower_mask);
            state_[offset] = state_[offset + 397] ^ (mixed >> 1U)
                ^ ((mixed & 1U) != 0U ? matrix_a : 0U);
        }
        for (std::size_t offset = 227; offset < 623; ++offset) {
            const uint32_t mixed =
                (state_[offset] & upper_mask) | (state_[offset + 1] & lower_mask);
            state_[offset] = state_[offset - 227] ^ (mixed >> 1U)
                ^ ((mixed & 1U) != 0U ? matrix_a : 0U);
        }
        const uint32_t mixed =
            (state_[623] & upper_mask) | (state_[0] & lower_mask);
        state_[623] = state_[396] ^ (mixed >> 1U)
            ^ ((mixed & 1U) != 0U ? matrix_a : 0U);
        index_ = 0;
    }

    std::array<uint32_t, kCpythonStateWordsV1> state_{};
    int32_t index_ = static_cast<int32_t>(kCpythonStateWordsV1);
};

struct PreparedCpythonGaussianWindowV1 {
    torch::Tensor primary;
    torch::Tensor secondary;
    int64_t staging_post_index;
    int64_t tape_allocation_time_ns = 0;
    int64_t raw_generation_time_ns = 0;
    int64_t transcendental_conversion_time_ns = 0;
};

inline PreparedCpythonGaussianWindowV1 make_cpython_gaussian_window_v1(
    const torch::Tensor& words_cpu,
    const int64_t index,
    const int64_t capacity
) {
    TORCH_CHECK(capacity >= 0, "Gaussian tape capacity must be nonnegative");
    TORCH_CHECK(
        capacity <= kCpythonGaussCapacityLimitV1,
        "Gaussian tape capacity exceeds the bounded limit"
    );
    auto options = torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU);
    const int64_t allocation_start_ns = steady_ns_v1();
    auto primary = torch::empty({capacity}, options);
    auto secondary = torch::empty({capacity}, options);
    const int64_t tape_allocation_time_ns =
        steady_ns_v1() - allocation_start_ns;
    if (capacity == 0) {
        return {
            std::move(primary),
            std::move(secondary),
            index,
            tape_allocation_time_ns,
            0,
            0,
        };
    }

    CpythonMt19937V1 random(words_cpu, index);
    std::vector<uint32_t> raw_words(static_cast<std::size_t>(capacity + 3));
    const int64_t raw_start_ns = steady_ns_v1();
    for (auto& word : raw_words) {
        word = random.next_u32();
    }
    const int64_t raw_generation_time_ns = steady_ns_v1() - raw_start_ns;
    auto* primary_values = primary.data_ptr<double>();
    auto* secondary_values = secondary.data_ptr<double>();
    constexpr double double_scale = 1.0 / 9007199254740992.0;
    constexpr double two_pi = 6.2831853071795864769252867665590057683943387987502;
    const int64_t conversion_start_ns = steady_ns_v1();
    for (int64_t offset = 0; offset < capacity; ++offset) {
        const auto position = static_cast<std::size_t>(offset);
        const double first = (
            static_cast<double>(raw_words[position] >> 5U) * 67108864.0
            + static_cast<double>(raw_words[position + 1] >> 6U)
        ) * double_scale;
        const double second = (
            static_cast<double>(raw_words[position + 2] >> 5U) * 67108864.0
            + static_cast<double>(raw_words[position + 3] >> 6U)
        ) * double_scale;
        const double x2pi = first * two_pi;
        const double radius = std::sqrt(-2.0 * std::log(1.0 - second));
        primary_values[position] = std::cos(x2pi) * radius;
        secondary_values[position] = std::sin(x2pi) * radius;
    }
    const int64_t transcendental_conversion_time_ns =
        steady_ns_v1() - conversion_start_ns;

    return {
        std::move(primary),
        std::move(secondary),
        random.index(),
        tape_allocation_time_ns,
        raw_generation_time_ns,
        transcendental_conversion_time_ns,
    };
}

inline pybind11::dict prepare_cpython_gaussian_window_v1(
    const torch::Tensor& words_cpu,
    const int64_t index,
    const int64_t capacity
) {
    auto prepared = make_cpython_gaussian_window_v1(words_cpu, index, capacity);
    pybind11::dict out;
    out["python_random_gauss_primary_tape"] = prepared.primary;
    out["python_random_gauss_secondary_tape"] = prepared.secondary;
    out["staging_post_index"] = prepared.staging_post_index;
    out["source_state_preserved"] = true;
    return out;
}

struct PreparedFixedMlpPolicyWindowV1 {
    torch::Tensor source;
    torch::Tensor delta;
    torch::Tensor mutation_mask;
    torch::Tensor crossover_mask_packed;
    torch::Tensor post_state_packed;
    torch::Tensor semantic_draws;
    int64_t torch_random_generation_time_ns = 0;
    int64_t torch_compare_time_ns = 0;
    int64_t crossover_pack_time_ns = 0;
    int64_t torch_randn_generation_time_ns = 0;
    int64_t stack_allocation_time_ns = 0;
};

inline PreparedFixedMlpPolicyWindowV1 make_fixed_mlp_policy_window_rngv2_v1(
    const torch::Tensor& state_cpu,
    const int64_t episodes,
    const int64_t width,
    const int64_t mutation_width,
    const bool mutation_enable,
    const double mutation_rate,
    const double mutation_std,
    const uint32_t world_uid,
    const uint64_t semantic_origin,
    const uint64_t namespace_key = 0xCA05117E450A51C0ull,
    const uint64_t absolute_tick = 0
) {
    TORCH_CHECK(episodes >= 0, "V2 policy episodes must be nonnegative");
    TORCH_CHECK(width >= 0 && width <= 65535, "V2 policy width must fit operation_id");
    TORCH_CHECK(mutation_width == 1 || mutation_width == width,
        "V2 policy mutation tape width must be compact width 1 or full policy width");
    const auto source = state_cpu.defined()
        ? state_cpu.contiguous().clone()
        : torch::empty({0}, torch::TensorOptions().dtype(torch::kUInt8));
    const auto float_options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
    const auto bool_options = torch::TensorOptions().dtype(torch::kBool).device(torch::kCPU);
    const int64_t row_bytes = (width + 7) / 8;
    auto crossover_packed = torch::zeros(
        {episodes, row_bytes},
        torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU)
    );
    const bool enabled = mutation_enable && mutation_std > 0.0 && mutation_width == width;
    auto delta = torch::zeros({episodes, mutation_width}, float_options);
    auto mutation_mask = torch::zeros({episodes, mutation_width}, bool_options);
    auto post_state_packed = torch::empty(
        {episodes, 0},
        torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU)
    );
    auto semantic_draws = torch::full(
        {episodes},
        width,
        torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU)
    );
    auto* const packed = crossover_packed.data_ptr<uint8_t>();
    auto* const delta_ptr = delta.data_ptr<float>();
    auto* const mask_ptr = mutation_mask.data_ptr<bool>();
    for (int64_t episode = 0; episode < episodes; ++episode) {
        const uint64_t subject = semantic_origin + static_cast<uint64_t>(episode);
        for (int64_t column = 0; column < width; ++column) {
            const int32_t field_status = hapb_rngv2::validate_fields(
                world_uid,
                hapb_rngv2::DOMAIN_POLICY_WINDOW_CROSSOVER,
                1u,
                subject,
                absolute_tick,
                static_cast<uint32_t>(column),
                0
            );
            TORCH_CHECK(field_status == tw_rngv2::RNG_OK, "HAPB RNGV2 policy address is invalid");
            const uint32_t word = hapb_rngv2::word_at(
                namespace_key,
                world_uid,
                hapb_rngv2::DOMAIN_POLICY_WINDOW_CROSSOVER,
                1u,
                subject,
                absolute_tick,
                static_cast<uint32_t>(column),
                0
            );
            if (tw_rngv2::u32_to_f32_half_open(word) < 0.5f) {
                packed[episode * row_bytes + (column >> 3)] =
                    static_cast<uint8_t>(
                        packed[episode * row_bytes + (column >> 3)]
                        | static_cast<uint8_t>(1u << (column & 7))
                    );
            }
            if (!enabled) {
                continue;
            }
            bool keep = true;
            if (mutation_rate < 1.0) {
                const uint32_t mask_word = hapb_rngv2::word_at(
                    namespace_key,
                    world_uid,
                    hapb_rngv2::DOMAIN_POLICY_WINDOW_MASK,
                    1u,
                    subject,
                    absolute_tick,
                    static_cast<uint32_t>(column),
                    0
                );
                keep = tw_rngv2::u32_to_f32_half_open(mask_word) < static_cast<float>(mutation_rate);
            }
            mask_ptr[episode * mutation_width + column] = keep;
            if (!keep) {
                continue;
            }
            uint32_t k0 = 0;
            uint32_t k1 = 0;
            uint32_t c0 = 0;
            uint32_t c1 = 0;
            uint32_t c2 = 0;
            uint32_t c3 = 0;
            tw_rngv2::encode_key(
                namespace_key,
                world_uid,
                hapb_rngv2::DOMAIN_POLICY_WINDOW_MUTATION,
                1u,
                &k0,
                &k1
            );
            tw_rngv2::encode_counter(
                subject,
                absolute_tick,
                static_cast<uint32_t>(column),
                0,
                &c0,
                &c1,
                &c2,
                &c3
            );
            const tw_rngv2::PhiloxWords words =
                tw_rngv2::philox4x32_10(c0, c1, c2, c3, k0, k1);
            uint64_t z0_bits = 0;
            uint64_t z1_bits = 0;
            int32_t normal_status = tw_rngv2::RNG_OK;
            tw_rngv2::box_muller(words, &z0_bits, &z1_bits, &normal_status);
            TORCH_CHECK(normal_status == tw_rngv2::RNG_OK, "HAPB RNGV2 policy normal is non-finite");
            double z0 = 0.0;
            std::memcpy(&z0, &z0_bits, sizeof(z0));
            delta_ptr[episode * mutation_width + column] =
                static_cast<float>(static_cast<double>(mutation_std) * z0);
        }
    }
    return {
        std::move(source),
        std::move(delta),
        std::move(mutation_mask),
        std::move(crossover_packed),
        std::move(post_state_packed),
        std::move(semantic_draws),
        0,
        0,
        0,
        0,
        0,
    };
}

inline int64_t packed_crossover_row_bytes_v1(const int64_t logical_width) {
    TORCH_CHECK(logical_width >= 0, "policy crossover logical width must be nonnegative");
    return (logical_width + 7) / 8;
}

inline torch::Tensor pack_crossover_mask_lsb0_v1(const torch::Tensor& mask) {
    TORCH_CHECK(mask.defined() && mask.dim() == 1, "policy crossover row must be rank-1");
    TORCH_CHECK(mask.scalar_type() == torch::kBool, "policy crossover row must be bool");
    TORCH_CHECK(mask.is_contiguous(), "policy crossover row must be contiguous");
    const int64_t width = mask.numel();
    const int64_t row_bytes = packed_crossover_row_bytes_v1(width);
    auto packed = torch::zeros({row_bytes}, torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU));
    const auto* const source = mask.data_ptr<bool>();
    auto* const target = packed.data_ptr<uint8_t>();
    for (int64_t column = 0; column < width; ++column) {
        if (source[column]) {
            target[column / 8] = static_cast<uint8_t>(
                target[column / 8] | static_cast<uint8_t>(1U << (column & 7))
            );
        }
    }
    return packed;
}

inline torch::Tensor pack_crossover_mask_lsb0_batched_v1(
    const torch::Tensor& mask
) {
    TORCH_CHECK(
        mask.defined() && mask.dim() == 2,
        "batched policy crossover mask must be rank-2"
    );
    TORCH_CHECK(
        mask.scalar_type() == torch::kBool,
        "batched policy crossover mask must be bool"
    );
    TORCH_CHECK(
        !mask.is_cuda() && mask.is_contiguous(),
        "batched policy crossover mask must be contiguous CPU storage"
    );
    const int64_t rows = mask.size(0);
    const int64_t width = mask.size(1);
    const int64_t row_bytes = packed_crossover_row_bytes_v1(width);
    auto packed = torch::zeros(
        {rows, row_bytes},
        torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU)
    );
    for (int64_t bit = 0; bit < 8; ++bit) {
        auto bit_plane = mask.slice(1, bit, width, 8).to(torch::kUInt8);
        if (bit_plane.numel() == 0) {
            continue;
        }
        bit_plane.mul_(static_cast<int64_t>(1U << bit));
        packed.slice(1, 0, bit_plane.size(1)).bitwise_or_(bit_plane);
    }
    return packed;
}

inline uint32_t read_u32_le_v1(const uint8_t* const source) {
    return static_cast<uint32_t>(source[0])
        | (static_cast<uint32_t>(source[1]) << 8U)
        | (static_cast<uint32_t>(source[2]) << 16U)
        | (static_cast<uint32_t>(source[3]) << 24U);
}

inline uint64_t read_u64_le_v1(const uint8_t* const source) {
    return static_cast<uint64_t>(read_u32_le_v1(source))
        | (static_cast<uint64_t>(read_u32_le_v1(source + 4)) << 32U);
}

inline void require_canonical_torch_cpu_rng_metadata_v1(
    const uint8_t* const source
) {
    const uint32_t left = read_u32_le_v1(source + 8);
    const uint32_t seeded = read_u32_le_v1(source + 12);
    const uint64_t next = read_u64_le_v1(source + 16);
    const bool current_position =
        (left == 1U && next == 0U)
        || (
            left >= 1U && left <= 624U
            && next >= 1U && next <= 624U
            && static_cast<uint64_t>(left) + next == 625U
        );
    TORCH_CHECK(
        seeded == 1U && current_position,
        "Torch CPU RNG state has noncanonical seeded/left/next metadata"
    );
    const uint32_t double_normal_valid = read_u32_le_v1(source + 5040);
    TORCH_CHECK(
        double_normal_valid <= 1U && source[5052] <= 1U,
        "Torch CPU RNG state has noncanonical normal-cache validity metadata"
    );
    TORCH_CHECK(
        source[5044] == 0U && source[5045] == 0U
            && source[5046] == 0U && source[5047] == 0U
            && source[5053] == 0U && source[5054] == 0U
            && source[5055] == 0U,
        "Torch CPU RNG state has nonzero ABI padding"
    );
}

inline torch::Tensor pack_torch_cpu_rng_state_v1(
    const torch::Tensor& full_state
) {
    TORCH_CHECK(!full_state.is_cuda(), "Torch CPU RNG state must be CPU resident");
    TORCH_CHECK(
        full_state.scalar_type() == torch::kUInt8 && full_state.dim() == 1
            && full_state.numel() == torch_rng_pack::kFullStateBytes
            && full_state.is_contiguous(),
        "Torch CPU RNG state must be contiguous uint8[5056]"
    );
    auto packed = torch::empty(
        {torch_rng_pack::kPackedStateBytes},
        torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU)
    );
    const auto* const source = full_state.data_ptr<uint8_t>();
    require_canonical_torch_cpu_rng_metadata_v1(source);
    auto* const target = packed.data_ptr<uint8_t>();
    for (int64_t index = 0; index < torch_rng_pack::kFullStateBytes; ++index) {
        const int64_t packed_index = torch_rng_pack::packed_byte_index_v1(index);
        if (packed_index < 0) {
            TORCH_CHECK(
                source[index] == 0U,
                "Torch CPU RNG state has nonzero omitted upper 32-bit lane bytes"
            );
        } else {
            target[packed_index] = source[index];
        }
    }
    return packed;
}

inline bool is_lower_hex_sha256_v1(const std::string& value) {
    if (value.size() != 64U) {
        return false;
    }
    return std::all_of(
        value.begin(),
        value.end(),
        [](const char item) {
            return (item >= '0' && item <= '9')
                || (item >= 'a' && item <= 'f');
        }
    );
}

inline pybind11::dict torch_rng_post_state_abi_receipt_v1(
    const std::string& execution_route_sha256
) {
    const bool route_bound = !execution_route_sha256.empty();
    TORCH_CHECK(
        !route_bound || is_lower_hex_sha256_v1(execution_route_sha256),
        "execution_route_sha256 must be empty or one lowercase SHA-256 hex digest"
    );
    pybind11::dict out;
    out["abi_schema_version"] = torch_rng_pack::kAbiSchemaVersion;
    out["codec_schema_version"] = torch_rng_pack::kEncoding;
    out["abi_fingerprint_sha256"] = torch_rng_pack::kAbiFingerprintSha256;
    out["torch_version"] = torch_rng_pack::kQualifiedTorchVersion;
    out["torch_git_version"] = torch_rng_pack::kQualifiedTorchGitVersion;
    out["torch_build_config_sha256"] =
        torch_rng_pack::kQualifiedTorchBuildConfigSha256;
    out["byteorder"] = torch_rng_pack::kQualifiedByteOrder;
    out["full_state_bytes"] = torch_rng_pack::kFullStateBytes;
    out["packed_state_bytes"] = torch_rng_pack::kPackedStateBytes;
    out["portable_checkpoint_format"] = false;
    out["execution_tape_encoding"] = true;
    out["route_binding_status"] =
        route_bound ? "execution_route_bound" : "unbound_native_refill_probe";
    out["execution_route_sha256"] = execution_route_sha256;
    return out;
}

inline PreparedFixedMlpPolicyWindowV1 make_fixed_mlp_policy_window_v1(
    const torch::Tensor& state_cpu,
    const int64_t episodes,
    const int64_t width,
    const int64_t mutation_width,
    const bool mutation_enable,
    std::string mutation_mode,
    const double mutation_rate,
    const double mutation_std,
    const std::string& execution_route_sha256 = std::string()
) {
    TORCH_CHECK(!state_cpu.is_cuda(), "policy RNG state must be CPU resident");
    TORCH_CHECK(state_cpu.scalar_type() == torch::kUInt8, "policy RNG state must be uint8");
    TORCH_CHECK(state_cpu.dim() == 1, "policy RNG state must be rank-1");
    TORCH_CHECK(
        state_cpu.numel() == torch_rng_pack::kFullStateBytes,
        "policy RNG state must use the canonical 5056-byte Torch layout"
    );
    TORCH_CHECK(episodes >= 0 && episodes <= kPolicyEpisodeCapacityLimitV1, "policy episode capacity is outside the bounded range");
    TORCH_CHECK(width >= 0, "policy row width must be nonnegative");
    TORCH_CHECK(
        mutation_width == 1 || mutation_width == width,
        "policy mutation tape width must be compact width 1 or full policy width"
    );
    TORCH_CHECK(std::isfinite(mutation_rate), "policy mutation rate must be finite");
    TORCH_CHECK(std::isfinite(mutation_std), "policy mutation standard deviation must be finite");

    const std::string requested_mutation_mode = mutation_mode;
    std::transform(
        mutation_mode.begin(),
        mutation_mode.end(),
        mutation_mode.begin(),
        [](const unsigned char value) { return static_cast<char>(std::tolower(value)); }
    );
    if (mutation_mode.empty() || mutation_mode == "gaussian") {
        mutation_mode = "dense";
    }
    TORCH_CHECK(mutation_mode == "dense" || mutation_mode == "sparse", "policy mutation mode must be dense or sparse");
    const double rate = std::clamp(mutation_rate, 0.0, 1.0);
    const double standard_deviation = std::max(0.0, mutation_std);
    const bool enabled = mutation_enable && standard_deviation > 0.0;
    const bool materialize_post_state_rows = mutation_width != 1;
    TORCH_CHECK(
        !enabled || mutation_width == width,
        "compact policy mutation tape requires disabled policy mutation"
    );

    const auto source = state_cpu.contiguous().clone();
    at::Generator generator = at::detail::createCPUGenerator(0);
    generator.set_state(source);
    const auto float_options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
    const auto bool_options = torch::TensorOptions().dtype(torch::kBool).device(torch::kCPU);
    const bool m7_batched_cpu_surface_supported =
        !mutation_enable && !enabled && mutation_width == 1
        && requested_mutation_mode == "dense";
    if constexpr (kOptionDM7BatchedCpuPolicyRefillEnabledV1) {
    if (m7_batched_cpu_surface_supported) {
        const int64_t allocation_start_ns = steady_ns_v1();
        auto delta = torch::zeros({episodes, 1}, float_options);
        auto mutation_mask = torch::zeros({episodes, 1}, bool_options);
        auto post_state_packed = torch::empty(
            {episodes, 0},
            torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU)
        );
        auto semantic_draws = torch::full(
            {episodes},
            width,
            torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU)
        );
        const int64_t stack_allocation_time_ns =
            steady_ns_v1() - allocation_start_ns;

        const int64_t crossover_rand_start_ns = steady_ns_v1();
        auto crossover_draws = at::rand(
            {episodes, width}, generator, float_options
        );
        const int64_t torch_random_generation_time_ns =
            steady_ns_v1() - crossover_rand_start_ns;

        const int64_t crossover_compare_start_ns = steady_ns_v1();
        auto crossover = crossover_draws.lt(0.5).contiguous();
        const int64_t torch_compare_time_ns =
            steady_ns_v1() - crossover_compare_start_ns;

        const int64_t pack_start_ns = steady_ns_v1();
        auto crossover_packed =
            pack_crossover_mask_lsb0_batched_v1(crossover);
        const int64_t crossover_pack_time_ns = steady_ns_v1() - pack_start_ns;

        return {
            std::move(source),
            std::move(delta),
            std::move(mutation_mask),
            std::move(crossover_packed),
            std::move(post_state_packed),
            std::move(semantic_draws),
            torch_random_generation_time_ns,
            torch_compare_time_ns,
            crossover_pack_time_ns,
            0,
            stack_allocation_time_ns,
        };
    }
    }
    std::vector<torch::Tensor> crossover_packed_rows;
    std::vector<torch::Tensor> delta_rows;
    std::vector<torch::Tensor> mask_rows;
    std::vector<torch::Tensor> packed_post_state_rows;
    std::vector<int64_t> semantic_draw_rows;
    crossover_packed_rows.reserve(static_cast<std::size_t>(episodes));
    delta_rows.reserve(static_cast<std::size_t>(episodes));
    mask_rows.reserve(static_cast<std::size_t>(episodes));
    if (materialize_post_state_rows) {
        packed_post_state_rows.reserve(static_cast<std::size_t>(episodes));
    }
    semantic_draw_rows.reserve(static_cast<std::size_t>(episodes));
    int64_t torch_random_generation_time_ns = 0;
    int64_t torch_compare_time_ns = 0;
    int64_t crossover_pack_time_ns = 0;
    int64_t torch_randn_generation_time_ns = 0;
    for (int64_t episode = 0; episode < episodes; ++episode) {
        const int64_t crossover_rand_start_ns = steady_ns_v1();
        auto crossover_draws = at::rand({width}, generator, float_options);
        if constexpr (kOptionDM0TelemetryEnabledV1) {
            torch_random_generation_time_ns +=
                steady_ns_v1() - crossover_rand_start_ns;
        }
        const int64_t crossover_compare_start_ns = steady_ns_v1();
        auto crossover = crossover_draws.lt(0.5);
        if constexpr (kOptionDM0TelemetryEnabledV1) {
            torch_compare_time_ns +=
                steady_ns_v1() - crossover_compare_start_ns;
        }
        int64_t semantic_draws = width;
        torch::Tensor mask;
        torch::Tensor delta;
        if (enabled) {
            if (mutation_mode == "sparse") {
                const int64_t mask_rand_start_ns = steady_ns_v1();
                auto mask_draws = at::rand({width}, generator, float_options);
                if constexpr (kOptionDM0TelemetryEnabledV1) {
                    torch_random_generation_time_ns +=
                        steady_ns_v1() - mask_rand_start_ns;
                }
                const int64_t mask_compare_start_ns = steady_ns_v1();
                mask = mask_draws.lt(rate);
                if constexpr (kOptionDM0TelemetryEnabledV1) {
                    torch_compare_time_ns +=
                        steady_ns_v1() - mask_compare_start_ns;
                }
                semantic_draws += width;
            } else {
                mask = torch::ones({width}, bool_options);
            }
            const int64_t randn_start_ns = steady_ns_v1();
            delta = at::randn({width}, generator, float_options)
                * standard_deviation;
            if constexpr (kOptionDM0TelemetryEnabledV1) {
                torch_randn_generation_time_ns +=
                    steady_ns_v1() - randn_start_ns;
            }
            semantic_draws += width;
            delta = torch::where(mask, delta, torch::zeros_like(delta));
        } else {
            mask = torch::zeros({mutation_width}, bool_options);
            delta = torch::zeros({mutation_width}, float_options);
        }
        const int64_t pack_start_ns = steady_ns_v1();
        crossover_packed_rows.push_back(
            pack_crossover_mask_lsb0_v1(crossover.contiguous())
        );
        if constexpr (kOptionDM0TelemetryEnabledV1) {
            crossover_pack_time_ns += steady_ns_v1() - pack_start_ns;
        }
        delta_rows.push_back(delta.contiguous());
        mask_rows.push_back(mask.contiguous());
        if (materialize_post_state_rows) {
            packed_post_state_rows.push_back(
                pack_torch_cpu_rng_state_v1(generator.get_state().contiguous())
            );
        }
        semantic_draw_rows.push_back(semantic_draws);
    }

    const int64_t stack_start_ns = steady_ns_v1();
    const auto stack_or_empty = [](const std::vector<torch::Tensor>& rows, torch::Tensor empty) {
        return rows.empty() ? std::move(empty) : torch::stack(rows).contiguous();
    };
    auto crossover_packed = stack_or_empty(
        crossover_packed_rows,
        torch::empty(
            {0, packed_crossover_row_bytes_v1(width)},
            torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU)
        )
    );
    auto delta = stack_or_empty(
        delta_rows,
        torch::empty({0, mutation_width}, float_options)
    );
    auto mutation_mask = stack_or_empty(
        mask_rows,
        torch::empty({0, mutation_width}, bool_options)
    );
    const std::vector<int64_t> post_state_empty_shape =
        materialize_post_state_rows
        ? std::vector<int64_t>{0, torch_rng_pack::kPackedStateBytes}
        : std::vector<int64_t>{episodes, 0};
    auto post_state_packed = stack_or_empty(
        packed_post_state_rows,
        torch::empty(
            post_state_empty_shape,
            torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU)
        )
    );
    auto semantic_draws = torch::tensor(
        semantic_draw_rows,
        torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU)
    );
    const int64_t stack_allocation_time_ns = steady_ns_v1() - stack_start_ns;
    return {
        std::move(source),
        std::move(delta),
        std::move(mutation_mask),
        std::move(crossover_packed),
        std::move(post_state_packed),
        std::move(semantic_draws),
        torch_random_generation_time_ns,
        torch_compare_time_ns,
        crossover_pack_time_ns,
        torch_randn_generation_time_ns,
        stack_allocation_time_ns,
    };
}

inline pybind11::dict prepare_fixed_mlp_policy_window_v1(
    const torch::Tensor& state_cpu,
    const int64_t episodes,
    const int64_t width,
    const bool mutation_enable,
    std::string mutation_mode,
    const double mutation_rate,
    const double mutation_std,
    const std::string& execution_route_sha256 = std::string()
) {
    auto prepared = make_fixed_mlp_policy_window_v1(
        state_cpu,
        episodes,
        width,
        width,
        mutation_enable,
        std::move(mutation_mode),
        mutation_rate,
        mutation_std,
        execution_route_sha256
    );
    auto abi_receipt = torch_rng_post_state_abi_receipt_v1(
        execution_route_sha256
    );
    pybind11::dict out;
    out["rng_manager_mutation_torch_cpu_state"] = prepared.source;
    out["policy_birth_delta_tape"] = prepared.delta;
    out["policy_birth_mutation_mask_tape"] = prepared.mutation_mask;
    out["policy_birth_crossover_mask_packed_tape"] = prepared.crossover_mask_packed;
    out["policy_birth_crossover_mask_logical_width"] = width;
    out["policy_birth_crossover_mask_encoding"] = "bitpack_lsb0_v1";
    out["policy_birth_post_rng_state_packed_tape"] = prepared.post_state_packed;
    out["policy_birth_post_rng_state_encoding"] = torch_rng_pack::kEncoding;
    out["policy_birth_post_rng_state_abi_schema_version"] =
        torch_rng_pack::kAbiSchemaVersion;
    out["policy_birth_post_rng_state_abi_fingerprint_sha256"] =
        torch_rng_pack::kAbiFingerprintSha256;
    out["policy_birth_post_rng_state_abi"] = abi_receipt;
    out["policy_birth_post_rng_state_route_binding_status"] =
        abi_receipt["route_binding_status"];
    out["policy_birth_post_rng_state_execution_route_sha256"] =
        abi_receipt["execution_route_sha256"];
    out["policy_birth_post_rng_state_qualified_torch_version"] =
        torch_rng_pack::kQualifiedTorchVersion;
    out["policy_birth_post_rng_state_qualified_torch_git_version"] =
        torch_rng_pack::kQualifiedTorchGitVersion;
    out["policy_birth_post_rng_state_qualified_build_config_sha256"] =
        torch_rng_pack::kQualifiedTorchBuildConfigSha256;
    out["policy_birth_post_rng_state_byteorder"] =
        torch_rng_pack::kQualifiedByteOrder;
    out["policy_birth_post_rng_state_logical_row_bytes"] =
        torch_rng_pack::kFullStateBytes;
    out["policy_birth_post_rng_state_packed_row_bytes"] =
        torch_rng_pack::kPackedStateBytes;
    out["policy_birth_post_rng_state_dense_allocation_materialized"] = false;
    out["policy_birth_semantic_draws_tape"] = prepared.semantic_draws;
    out["option_d_m7_batched_cpu_policy_refill_enabled"] =
        kOptionDM7BatchedCpuPolicyRefillEnabledV1;
    out["option_d_m7_batched_cpu_policy_refill_used"] = false;
    out["option_d_m7_batched_cpu_policy_refill_authority"] =
        kOptionDM7BatchedCpuPolicyRefillEnabledV1
        ? "compiled_candidate_not_used_for_full_mutation_width_surface"
        : "disabled";
    out["source_state_preserved"] = torch::equal(
        prepared.source,
        state_cpu.contiguous()
    );
    if constexpr (kOptionDM0TelemetryEnabledV1) {
        out["torch_random_generation_time_ns"] =
            prepared.torch_random_generation_time_ns;
        out["torch_compare_time_ns"] = prepared.torch_compare_time_ns;
        out["crossover_pack_time_ns"] = prepared.crossover_pack_time_ns;
        out["torch_compare_pack_time_ns"] =
            prepared.torch_compare_time_ns + prepared.crossover_pack_time_ns;
        out["torch_randn_generation_time_ns"] =
            prepared.torch_randn_generation_time_ns;
        out["policy_allocation_stack_copy_time_ns"] =
            prepared.stack_allocation_time_ns;
    }
    return out;
}

inline void require_batched_tensor_v1(
    const torch::Tensor& tensor,
    const torch::Tensor& reference,
    const torch::ScalarType dtype,
    const std::vector<int64_t>& shape,
    const char* name
) {
    TORCH_CHECK(tensor.defined(), name, " must be defined");
    TORCH_CHECK(tensor.scalar_type() == dtype, name, " has the wrong dtype");
    TORCH_CHECK(tensor.sizes().vec() == shape, name, " has the wrong shape");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(tensor.device() == reference.device(), name, " must share one device");
    TORCH_CHECK(!tensor.requires_grad(), name, " must not require gradients");
}

inline std::vector<int64_t> host_int64_values_v1(
    const torch::Tensor& tensor,
    const char* name
) {
    auto host = tensor.detach().cpu().contiguous();
    TORCH_CHECK(host.scalar_type() == torch::kInt64, name, " must be int64");
    std::vector<int64_t> values(static_cast<std::size_t>(host.numel()));
    const auto* source = host.data_ptr<int64_t>();
    std::copy(source, source + host.numel(), values.begin());
    return values;
}

inline pybind11::list int64_values_to_list_v1(const std::vector<int64_t>& values) {
    pybind11::list out;
    for (const auto value : values) {
        out.append(value);
    }
    return out;
}

inline std::vector<int64_t> int64_values_from_list_v1(
    const pybind11::handle value,
    const int64_t expected_size,
    const char* name
) {
    TORCH_CHECK(
        PyList_Check(value.ptr()),
        name,
        " receipt field must be a list"
    );
    const auto rows = pybind11::reinterpret_borrow<pybind11::list>(value);
    TORCH_CHECK(
        pybind11::len(rows) == expected_size,
        name,
        " receipt field has the wrong size"
    );
    std::vector<int64_t> out;
    out.reserve(static_cast<std::size_t>(expected_size));
    for (const pybind11::handle item : rows) {
        TORCH_CHECK(
            PyLong_Check(item.ptr()) && !PyBool_Check(item.ptr()),
            name,
            " receipt field must contain exact integers"
        );
        out.push_back(pybind11::cast<int64_t>(item));
    }
    return out;
}

inline int64_t policy_window_identity_checkpoint_int64_v1(
    const pybind11::dict& mapping,
    const char* const name
) {
    TORCH_CHECK(
        mapping.contains(name),
        "policy-window identity checkpoint lacks ",
        name
    );
    const auto value = mapping[pybind11::str(name)];
    TORCH_CHECK(
        PyLong_Check(value.ptr()) && !PyBool_Check(value.ptr()),
        "policy-window identity checkpoint field ",
        name,
        " must be an exact integer"
    );
    return pybind11::cast<int64_t>(value);
}

inline bool policy_window_identity_checkpoint_bool_v1(
    const pybind11::dict& mapping,
    const char* const name
) {
    TORCH_CHECK(
        mapping.contains(name),
        "policy-window identity checkpoint lacks ",
        name
    );
    const auto value = mapping[pybind11::str(name)];
    TORCH_CHECK(
        PyBool_Check(value.ptr()),
        "policy-window identity checkpoint field ",
        name,
        " must be a boolean"
    );
    return pybind11::cast<bool>(value);
}

inline std::string policy_window_identity_checkpoint_string_v1(
    const pybind11::dict& mapping,
    const char* const name
) {
    TORCH_CHECK(
        mapping.contains(name),
        "policy-window identity checkpoint lacks ",
        name
    );
    const auto value = mapping[pybind11::str(name)];
    TORCH_CHECK(
        PyUnicode_Check(value.ptr()),
        "policy-window identity checkpoint field ",
        name,
        " must be a string"
    );
    return pybind11::cast<std::string>(value);
}

inline pybind11::dict export_policy_window_identity_owner_checkpoint_v1(
    const pybind11::handle capsule
) {
    auto* const owner = policy_window_identity_owner_from_capsule_v1(capsule);
    const std::unique_lock<std::mutex> lock(
        policy_window_identity_owner_execution_mutex_v1(owner),
        std::try_to_lock
    );
    TORCH_CHECK(
        lock.owns_lock(),
        "HAPB policy-window identity checkpoint raced an active execution"
    );
    TORCH_CHECK(
        owner->pending_validation_token == 0,
        "HAPB policy-window identity checkpoint has an unvalidated entry"
    );
    const auto& state = owner->state;
    TORCH_CHECK(
        state.initialized,
        "HAPB policy-window identity checkpoint requires a published state"
    );
    TORCH_CHECK(
        state.configuration.cells > 0
            && state.window_valid.size()
                == static_cast<std::size_t>(state.configuration.cells)
            && state.window_epoch.size()
                == static_cast<std::size_t>(state.configuration.cells)
            && state.window_rng_lineage_epoch.size()
                == static_cast<std::size_t>(state.configuration.cells)
            && state.window_base.size()
                == static_cast<std::size_t>(state.configuration.cells)
            && state.window_base_semantic_draws.size()
                == static_cast<std::size_t>(state.configuration.cells),
        "HAPB policy-window identity checkpoint state is incomplete"
    );
    pybind11::dict configuration;
    configuration["cells"] = state.configuration.cells;
    configuration["capacity"] = state.configuration.capacity;
    configuration["policy_row_width"] = state.configuration.policy_row_width;
    configuration["mutation_width"] = state.configuration.mutation_width;
    configuration["mutation_enable"] = state.configuration.mutation_enable;
    configuration["mutation_mode"] = state.configuration.mutation_mode;
    configuration["mutation_rate"] = state.configuration.mutation_rate;
    configuration["mutation_std"] = state.configuration.mutation_std;
    configuration["producer_schema_sha256"] =
        state.configuration.producer_schema_sha256;
    configuration["crossover_encoding"] =
        state.configuration.crossover_encoding;
    configuration["rng_replay_abi"] = state.configuration.rng_replay_abi;
    pybind11::list valid;
    for (const auto value : state.window_valid) {
        valid.append(static_cast<int64_t>(value));
    }
    pybind11::dict out;
    out["schema_version"] =
        "eiel_hapb_policy_window_identity_checkpoint_v1";
    out["initialized"] = state.initialized;
    out["tape_storage_generation"] = state.tape_storage_generation;
    out["rng_lineage_epoch"] = state.rng_lineage_epoch;
    out["window_valid"] = valid;
    out["window_epoch"] = int64_values_to_list_v1(state.window_epoch);
    out["window_rng_lineage_epoch"] = int64_values_to_list_v1(
        state.window_rng_lineage_epoch
    );
    out["window_base"] = int64_values_to_list_v1(state.window_base);
    out["window_base_semantic_draws"] = int64_values_to_list_v1(
        state.window_base_semantic_draws
    );
    out["configuration"] = configuration;
    out["publication_origin"] = policy_window_publication_origin_name_v1(
        state.publication_origin
    );
    return out;
}

inline void restore_policy_window_identity_owner_checkpoint_v1(
    const pybind11::handle capsule,
    const torch::Tensor& source_state,
    const torch::Tensor& delta,
    const torch::Tensor& mutation_mask,
    const torch::Tensor& crossover_mask_packed,
    const torch::Tensor& post_state_packed,
    const torch::Tensor& semantic_tape,
    const torch::Tensor& absolute_cursor,
    const torch::Tensor& semantic_counter,
    const torch::Tensor& window_base,
    const int64_t policy_row_width,
    const pybind11::dict& checkpoint
) {
    auto* const owner = policy_window_identity_owner_from_capsule_v1(capsule);
    const std::unique_lock<std::mutex> lock(
        policy_window_identity_owner_execution_mutex_v1(owner),
        std::try_to_lock
    );
    TORCH_CHECK(
        lock.owns_lock(),
        "HAPB policy-window identity restore raced an active execution"
    );
    TORCH_CHECK(
        owner->pending_validation_token == 0,
        "HAPB policy-window identity restore has an unvalidated entry"
    );
    TORCH_CHECK(
        policy_window_identity_checkpoint_string_v1(
            checkpoint, "schema_version"
        ) == "eiel_hapb_policy_window_identity_checkpoint_v1",
        "HAPB policy-window identity checkpoint schema is incompatible"
    );
    TORCH_CHECK(
        policy_window_identity_checkpoint_bool_v1(checkpoint, "initialized"),
        "HAPB policy-window identity checkpoint is not initialized"
    );
    const auto raw_configuration = checkpoint[pybind11::str("configuration")];
    TORCH_CHECK(
        PyMapping_Check(raw_configuration.ptr()),
        "HAPB policy-window identity checkpoint configuration is malformed"
    );
    const auto configuration = pybind11::reinterpret_borrow<pybind11::dict>(
        raw_configuration
    );
    PolicyWindowConfigurationIdentityV1 restored_configuration;
    restored_configuration.cells = policy_window_identity_checkpoint_int64_v1(
        configuration, "cells"
    );
    restored_configuration.capacity =
        policy_window_identity_checkpoint_int64_v1(configuration, "capacity");
    restored_configuration.policy_row_width =
        policy_window_identity_checkpoint_int64_v1(
            configuration, "policy_row_width"
        );
    restored_configuration.mutation_width =
        policy_window_identity_checkpoint_int64_v1(
            configuration, "mutation_width"
        );
    restored_configuration.mutation_enable =
        policy_window_identity_checkpoint_bool_v1(
            configuration, "mutation_enable"
        );
    restored_configuration.mutation_mode =
        policy_window_identity_checkpoint_string_v1(
            configuration, "mutation_mode"
        );
    TORCH_CHECK(
        configuration.contains("mutation_rate")
            && PyFloat_Check(configuration[pybind11::str("mutation_rate")].ptr()),
        "HAPB policy-window identity mutation rate is malformed"
    );
    TORCH_CHECK(
        configuration.contains("mutation_std")
            && PyFloat_Check(configuration[pybind11::str("mutation_std")].ptr()),
        "HAPB policy-window identity mutation std is malformed"
    );
    restored_configuration.mutation_rate = pybind11::cast<double>(
        configuration[pybind11::str("mutation_rate")]
    );
    restored_configuration.mutation_std = pybind11::cast<double>(
        configuration[pybind11::str("mutation_std")]
    );
    restored_configuration.producer_schema_sha256 =
        policy_window_identity_checkpoint_string_v1(
            configuration, "producer_schema_sha256"
        );
    restored_configuration.crossover_encoding =
        policy_window_identity_checkpoint_string_v1(
            configuration, "crossover_encoding"
        );
    restored_configuration.rng_replay_abi =
        policy_window_identity_checkpoint_string_v1(
            configuration, "rng_replay_abi"
        );
    TORCH_CHECK(
        source_state.defined() && source_state.dim() == 2
            && source_state.scalar_type() == torch::kUInt8
            && source_state.size(0) == restored_configuration.cells
            && source_state.size(1) == torch_rng_pack::kFullStateBytes,
        "HAPB policy-window identity restore source RNG shape differs"
    );
    TORCH_CHECK(
        delta.defined() && delta.dim() == 3
            && delta.size(0) == restored_configuration.cells
            && delta.size(1) == restored_configuration.capacity
            && delta.size(2) == restored_configuration.mutation_width
            && policy_row_width == restored_configuration.policy_row_width,
        "HAPB policy-window identity restore policy tape shape differs"
    );
    TORCH_CHECK(
        restored_configuration.mutation_width == 1
            && mutation_mask.defined()
            && crossover_mask_packed.defined()
            && post_state_packed.defined()
            && semantic_tape.defined()
            && absolute_cursor.defined()
            && semantic_counter.defined()
            && window_base.defined(),
        "HAPB policy-window identity restore lacks its complete tape ABI"
    );
    const auto tensors = policy_window_identity_tensors_v1(
        source_state,
        delta,
        mutation_mask,
        crossover_mask_packed,
        post_state_packed,
        semantic_tape,
        absolute_cursor,
        semantic_counter,
        window_base
    );
    const auto cells = restored_configuration.cells;
    const auto tape_generation = policy_window_identity_checkpoint_int64_v1(
        checkpoint, "tape_storage_generation"
    );
    const auto rng_lineage_epoch = policy_window_identity_checkpoint_int64_v1(
        checkpoint, "rng_lineage_epoch"
    );
    TORCH_CHECK(
        tape_generation > 0 && rng_lineage_epoch > 0,
        "HAPB policy-window identity restore epochs must be positive"
    );
    const auto window_valid = int64_values_from_list_v1(
        checkpoint[pybind11::str("window_valid")], cells, "window_valid"
    );
    const auto window_epoch = int64_values_from_list_v1(
        checkpoint[pybind11::str("window_epoch")], cells, "window_epoch"
    );
    const auto window_rng_lineage_epoch = int64_values_from_list_v1(
        checkpoint[pybind11::str("window_rng_lineage_epoch")],
        cells,
        "window_rng_lineage_epoch"
    );
    const auto window_base_values = int64_values_from_list_v1(
        checkpoint[pybind11::str("window_base")], cells, "window_base"
    );
    const auto window_base_semantic_draws = int64_values_from_list_v1(
        checkpoint[pybind11::str("window_base_semantic_draws")],
        cells,
        "window_base_semantic_draws"
    );
    for (int64_t cell = 0; cell < cells; ++cell) {
        const auto offset = static_cast<std::size_t>(cell);
        TORCH_CHECK(
            window_valid[offset] == 1
                && window_epoch[offset] == tape_generation
                && window_rng_lineage_epoch[offset] == rng_lineage_epoch
                && window_base_values[offset] >= 0
                && window_base_semantic_draws[offset] >= 0,
            "HAPB policy-window identity restore contains an invalid window"
        );
    }
    const auto publication_origin =
        policy_window_identity_checkpoint_string_v1(
            checkpoint, "publication_origin"
        );
    PolicyWindowPublicationOriginV1 restored_origin;
    if (publication_origin == "allocator_hydration_adopted") {
        restored_origin = PolicyWindowPublicationOriginV1::kAllocatorHydration;
    } else if (publication_origin == "physical_refill_published") {
        restored_origin = PolicyWindowPublicationOriginV1::kPhysicalRefill;
    } else {
        TORCH_CHECK(
            false,
            "HAPB policy-window identity restore has an unsupported publication origin"
        );
    }
    auto& state = owner->state;
    state.initialized = true;
    state.tape_storage_generation = tape_generation;
    state.rng_lineage_epoch = rng_lineage_epoch;
    state.window_valid.assign(
        window_valid.begin(),
        window_valid.end()
    );
    state.window_epoch = window_epoch;
    state.window_rng_lineage_epoch = window_rng_lineage_epoch;
    state.window_base = window_base_values;
    state.window_base_semantic_draws = window_base_semantic_draws;
    state.configuration = restored_configuration;
    for (std::size_t index = 0; index < tensors.size(); ++index) {
        state.tensors[index] = policy_window_tensor_identity_v1(tensors[index]);
    }
    state.execution_lease_crossover_storage_impl =
        crossover_mask_packed.storage().unsafeGetStorageImpl();
    state.publication_origin = restored_origin;
    owner->next_boundary_marker = "checkpoint_rehydrated_identity_owner";
}

inline pybind11::capsule create_untrusted_policy_window_identity_owner_v1(
    const torch::Tensor& source_state,
    const torch::Tensor& delta,
    const torch::Tensor& mutation_mask,
    const torch::Tensor& crossover_mask_packed,
    const torch::Tensor& post_state_packed,
    const torch::Tensor& semantic_tape,
    const torch::Tensor& absolute_cursor,
    const torch::Tensor& semantic_counter,
    const torch::Tensor& window_base,
    const int64_t policy_row_width,
    std::string allocator_boundary_marker
) {
    TORCH_CHECK(
        source_state.defined() && source_state.dim() == 2
            && source_state.scalar_type() == torch::kUInt8
            && source_state.is_contiguous() && !source_state.requires_grad(),
        "allocator policy RNG state must be contiguous uint8 rank-2"
    );
    const int64_t cells = source_state.size(0);
    TORCH_CHECK(
        cells > 0
            && source_state.size(1) == torch_rng_pack::kFullStateBytes,
        "allocator policy RNG state has the wrong shape"
    );
    TORCH_CHECK(
        delta.defined() && delta.dim() == 3 && delta.size(2) == 1,
        "policy-window identity owner requires compact policy rows"
    );
    const int64_t capacity = delta.size(1);
    TORCH_CHECK(
        capacity > 0 && capacity <= kPolicyEpisodeCapacityLimitV1
            && policy_row_width > 0,
        "policy-window identity dimensions are outside the bounded surface"
    );
    require_batched_tensor_v1(
        delta, source_state, torch::kFloat32, {cells, capacity, 1},
        "policy_birth_delta_tape"
    );
    require_batched_tensor_v1(
        mutation_mask, source_state, torch::kBool, {cells, capacity, 1},
        "policy_birth_mutation_mask_tape"
    );
    require_batched_tensor_v1(
        crossover_mask_packed,
        source_state,
        torch::kUInt8,
        {cells, capacity, packed_crossover_row_bytes_v1(policy_row_width)},
        "policy_birth_crossover_mask_packed_tape"
    );
    require_batched_tensor_v1(
        post_state_packed,
        source_state,
        torch::kUInt8,
        {cells, capacity, 0},
        "policy_birth_post_rng_state_packed_tape"
    );
    require_batched_tensor_v1(
        semantic_tape, source_state, torch::kInt64, {cells, capacity},
        "policy_birth_semantic_draws_tape"
    );
    require_batched_tensor_v1(
        absolute_cursor, source_state, torch::kInt64, {cells},
        "policy_birth_tape_cursor"
    );
    require_batched_tensor_v1(
        semantic_counter, source_state, torch::kInt64, {cells},
        "mutation_torch_cpu_semantic_draws"
    );
    require_batched_tensor_v1(
        window_base, source_state, torch::kInt64, {cells},
        "policy_birth_window_base"
    );
    TORCH_CHECK(
        allocator_boundary_marker == "fresh_allocator_owner"
            || allocator_boundary_marker
                == "checkpoint_reconstructed_allocator_owner",
        "policy-window identity owner requires an exact lifecycle marker"
    );

    auto* const owner = new PolicyWindowIdentityOwnerV1();
    try {
        owner->allocator_boundary_marker =
            std::move(allocator_boundary_marker);
        owner->execution_lease = policy_window_storage_execution_lease_v1(
            crossover_mask_packed
        );
        owner->state.execution_lease_crossover_storage_impl =
            crossover_mask_packed.storage().unsafeGetStorageImpl();
        const std::unique_lock<std::mutex> storage_guard(
            policy_window_identity_owner_execution_mutex_v1(owner),
            std::try_to_lock
        );
        TORCH_CHECK(
            storage_guard.owns_lock(),
            "policy-window identity creation raced an active writer"
        );
        // The allocator's pre-existing bytes are not a content proof.  The
        // first admitted ensure must physically materialize from canonical
        // RNG/counter state and publish identity only after that succeeds.
        invalidate_policy_window_identity_v1(&owner->state, true, true);
    } catch (...) {
        delete owner;
        throw;
    }
    return pybind11::capsule(
        owner,
        kPolicyWindowIdentityOwnerCapsuleNameV1,
        [](PyObject* capsule) {
            auto* const value = static_cast<PolicyWindowIdentityOwnerV1*>(
                PyCapsule_GetPointer(
                    capsule, kPolicyWindowIdentityOwnerCapsuleNameV1
                )
            );
            delete value;
        }
    );
}

inline pybind11::dict refill_cpython_gaussian_window_v1(
    const torch::Tensor& words,
    const torch::Tensor& index,
    const torch::Tensor& has_gauss,
    const torch::Tensor& gauss_next,
    torch::Tensor primary,
    torch::Tensor secondary,
    const torch::Tensor& absolute_raw_draws,
    const torch::Tensor& semantic_draws,
    torch::Tensor window_base
) {
    const int64_t refill_start_ns = steady_ns_v1();
    TORCH_CHECK(words.defined() && words.dim() == 2, "CPython random words must be batched rank-2");
    TORCH_CHECK(words.scalar_type() == torch::kUInt32, "CPython random words must be uint32");
    TORCH_CHECK(words.is_contiguous(), "CPython random words must be contiguous");
    TORCH_CHECK(!words.requires_grad(), "CPython random words must not require gradients");
    const int64_t cells = words.size(0);
    TORCH_CHECK(cells > 0 && words.size(1) == kCpythonStateWordsV1, "CPython random words must have shape [cells,624]");
    TORCH_CHECK(primary.defined() && primary.dim() == 2, "primary Gaussian tape must be batched rank-2");
    const int64_t capacity = primary.size(1);
    TORCH_CHECK(capacity > 0 && capacity <= kCpythonGaussCapacityLimitV1, "Gaussian physical capacity is outside the bounded range");
    require_batched_tensor_v1(index, words, torch::kInt64, {cells}, "python_random_index");
    require_batched_tensor_v1(has_gauss, words, torch::kBool, {cells}, "python_random_has_gauss");
    require_batched_tensor_v1(gauss_next, words, torch::kFloat64, {cells}, "python_random_gauss_next");
    require_batched_tensor_v1(primary, words, torch::kFloat64, {cells, capacity}, "python_random_gauss_primary_tape");
    require_batched_tensor_v1(secondary, words, torch::kFloat64, {cells, capacity}, "python_random_gauss_secondary_tape");
    require_batched_tensor_v1(absolute_raw_draws, words, torch::kInt64, {cells}, "python_random_raw_draws");
    require_batched_tensor_v1(semantic_draws, words, torch::kInt64, {cells}, "mutation_py_semantic_draws");
    require_batched_tensor_v1(window_base, words, torch::kInt64, {cells}, "python_random_window_base");

    const int64_t d2h_start_ns = steady_ns_v1();
    auto words_host = words.detach().cpu().contiguous();
    auto gauss_next_host = gauss_next.detach().cpu().contiguous();
    const auto indices = host_int64_values_v1(index, "python_random_index");
    const auto raw_before = host_int64_values_v1(absolute_raw_draws, "python_random_raw_draws");
    const auto semantic_before = host_int64_values_v1(semantic_draws, "mutation_py_semantic_draws");
    const auto base_before = host_int64_values_v1(window_base, "python_random_window_base");
    const int64_t device_to_host_time_ns = steady_ns_v1() - d2h_start_ns;
    const auto* gauss_next_values = gauss_next_host.data_ptr<double>();
    std::vector<torch::Tensor> primary_rows;
    std::vector<torch::Tensor> secondary_rows;
    primary_rows.reserve(static_cast<std::size_t>(cells));
    secondary_rows.reserve(static_cast<std::size_t>(cells));
    int64_t gaussian_raw_generation_time_ns = 0;
    int64_t gaussian_transcendental_conversion_time_ns = 0;
    int64_t gaussian_state_preparation_time_ns = 0;
    int64_t gaussian_tape_allocation_time_ns = 0;
    for (int64_t cell = 0; cell < cells; ++cell) {
        const int64_t state_preparation_start_ns = steady_ns_v1();
        TORCH_CHECK(indices[static_cast<std::size_t>(cell)] >= 0 && indices[static_cast<std::size_t>(cell)] <= kCpythonStateWordsV1, "python_random_index must be in [0,624]");
        TORCH_CHECK(std::isfinite(gauss_next_values[cell]), "python_random_gauss_next must be finite");
        TORCH_CHECK(raw_before[static_cast<std::size_t>(cell)] >= 0, "python_random_raw_draws must be nonnegative");
        TORCH_CHECK(semantic_before[static_cast<std::size_t>(cell)] >= 0, "mutation_py_semantic_draws must be nonnegative");
        TORCH_CHECK(base_before[static_cast<std::size_t>(cell)] >= 0 && base_before[static_cast<std::size_t>(cell)] <= raw_before[static_cast<std::size_t>(cell)], "python_random_window_base must be between zero and the absolute raw-draw counter");
        auto words_row = words_host[cell].clone().contiguous();
        gaussian_state_preparation_time_ns +=
            steady_ns_v1() - state_preparation_start_ns;
        auto prepared = make_cpython_gaussian_window_v1(
            words_row,
            indices[static_cast<std::size_t>(cell)],
            capacity
        );
        gaussian_tape_allocation_time_ns += prepared.tape_allocation_time_ns;
        gaussian_raw_generation_time_ns += prepared.raw_generation_time_ns;
        gaussian_transcendental_conversion_time_ns +=
            prepared.transcendental_conversion_time_ns;
        primary_rows.push_back(std::move(prepared.primary));
        secondary_rows.push_back(std::move(prepared.secondary));
    }
    const int64_t allocation_start_ns = steady_ns_v1();
    auto staged_primary = torch::stack(primary_rows).contiguous();
    auto staged_secondary = torch::stack(secondary_rows).contiguous();
    const int64_t gaussian_stack_pack_time_ns =
        steady_ns_v1() - allocation_start_ns;
    const int64_t primary_h2d_start_ns = steady_ns_v1();
    primary.copy_(staged_primary);
    const int64_t primary_h2d_submission_time_ns =
        steady_ns_v1() - primary_h2d_start_ns;
    const int64_t secondary_h2d_start_ns = steady_ns_v1();
    secondary.copy_(staged_secondary);
    const int64_t secondary_h2d_submission_time_ns =
        steady_ns_v1() - secondary_h2d_start_ns;
    const int64_t window_base_h2d_start_ns = steady_ns_v1();
    window_base.copy_(absolute_raw_draws);
    const int64_t window_base_h2d_submission_time_ns =
        steady_ns_v1() - window_base_h2d_start_ns;
    const int64_t host_to_device_time_ns =
        primary_h2d_submission_time_ns
        + secondary_h2d_submission_time_ns
        + window_base_h2d_submission_time_ns;

    pybind11::dict out;
    out["schema_version"] = "eiel_cuda_architecture_b_native_resident_window_refill_v1";
    out["operation"] = "refill_cpython_gaussian_window_v1";
    if constexpr (kOptionDM0TelemetryEnabledV1) {
        out["option_d_m0_telemetry_enabled"] = true;
    }
    out["batch_cells"] = cells;
    out["physical_capacity"] = capacity;
    out["absolute_raw_draws_before"] = int64_values_to_list_v1(raw_before);
    out["semantic_draws_before"] = int64_values_to_list_v1(semantic_before);
    out["window_base_before"] = int64_values_to_list_v1(base_before);
    out["window_base_after"] = int64_values_to_list_v1(raw_before);
    out["absolute_raw_draws_preserved"] = true;
    out["semantic_draws_preserved"] = true;
    out["source_rng_state_preserved"] = true;
    out["host_to_device_operations_lower_bound"] = static_cast<int64_t>(3);
    out["device_to_host_operations_lower_bound"] = static_cast<int64_t>(6);
    out["primary_segment_bytes"] =
        static_cast<int64_t>(primary.numel() * primary.element_size());
    out["secondary_segment_bytes"] =
        static_cast<int64_t>(secondary.numel() * secondary.element_size());
    out["window_base_bytes"] =
        static_cast<int64_t>(window_base.numel() * window_base.element_size());
    out["host_wall_time_ns"] = steady_ns_v1() - refill_start_ns;
    out["device_to_host_time_ns"] = device_to_host_time_ns;
    out["host_to_device_time_ns"] = host_to_device_time_ns;
    out["gaussian_raw_generation_time_ns"] = gaussian_raw_generation_time_ns;
    out["gaussian_transcendental_conversion_time_ns"] =
        gaussian_transcendental_conversion_time_ns;
    out["gaussian_state_preparation_time_ns"] =
        gaussian_state_preparation_time_ns;
    out["gaussian_tape_allocation_time_ns"] =
        gaussian_tape_allocation_time_ns;
    out["gaussian_stack_pack_time_ns"] = gaussian_stack_pack_time_ns;
    out["primary_h2d_submission_time_ns"] =
        primary_h2d_submission_time_ns;
    out["secondary_h2d_submission_time_ns"] =
        secondary_h2d_submission_time_ns;
    out["window_base_h2d_submission_time_ns"] =
        window_base_h2d_submission_time_ns;
    out["gaussian_allocation_copy_time_ns"] =
        gaussian_stack_pack_time_ns + host_to_device_time_ns;
    return out;
}

inline pybind11::dict ensure_cpython_gaussian_window_v1(
    const torch::Tensor& words,
    const torch::Tensor& index,
    const torch::Tensor& has_gauss,
    const torch::Tensor& gauss_next,
    torch::Tensor primary,
    torch::Tensor secondary,
    const torch::Tensor& absolute_raw_draws,
    const torch::Tensor& semantic_draws,
    torch::Tensor window_base,
    const int64_t required_future_draws
) {
    const int64_t ensure_start_ns = steady_ns_v1();
    TORCH_CHECK(
        primary.defined() && primary.dim() == 2,
        "primary Gaussian tape must be batched rank-2"
    );
    const int64_t cells = primary.size(0);
    const int64_t capacity = primary.size(1);
    TORCH_CHECK(
        cells > 0 && capacity > 0
            && capacity <= kCpythonGaussCapacityLimitV1,
        "Gaussian physical capacity is outside the bounded range"
    );
    TORCH_CHECK(
        required_future_draws >= 0 && required_future_draws <= capacity,
        "required future Gaussian draws must fit the physical tape"
    );
    TORCH_CHECK(
        words.defined() && words.dim() == 2
            && words.scalar_type() == torch::kUInt32
            && words.size(0) == cells
            && words.size(1) == kCpythonStateWordsV1
            && words.is_contiguous() && !words.requires_grad(),
        "CPython random words must be contiguous uint32[cells,624]"
    );
    require_batched_tensor_v1(
        index, words, torch::kInt64, {cells}, "python_random_index"
    );
    require_batched_tensor_v1(
        has_gauss, words, torch::kBool, {cells}, "python_random_has_gauss"
    );
    require_batched_tensor_v1(
        gauss_next,
        words,
        torch::kFloat64,
        {cells},
        "python_random_gauss_next"
    );
    require_batched_tensor_v1(
        primary,
        words,
        torch::kFloat64,
        {cells, capacity},
        "python_random_gauss_primary_tape"
    );
    require_batched_tensor_v1(
        secondary,
        words,
        torch::kFloat64,
        {cells, capacity},
        "python_random_gauss_secondary_tape"
    );
    require_batched_tensor_v1(
        absolute_raw_draws,
        words,
        torch::kInt64,
        {cells},
        "python_random_raw_draws"
    );
    require_batched_tensor_v1(
        semantic_draws,
        words,
        torch::kInt64,
        {cells},
        "mutation_py_semantic_draws"
    );
    require_batched_tensor_v1(
        window_base,
        words,
        torch::kInt64,
        {cells},
        "python_random_window_base"
    );

    const int64_t coverage_check_start_ns = steady_ns_v1();
    const auto raw_before = host_int64_values_v1(
        absolute_raw_draws, "python_random_raw_draws"
    );
    const auto base_before = host_int64_values_v1(
        window_base, "python_random_window_base"
    );
    const int64_t coverage_check_time_ns =
        steady_ns_v1() - coverage_check_start_ns;
    bool refill_required = false;
    for (int64_t cell = 0; cell < cells; ++cell) {
        const auto offset = static_cast<std::size_t>(cell);
        TORCH_CHECK(
            base_before[offset] >= 0
                && raw_before[offset] >= base_before[offset]
                && raw_before[offset] - base_before[offset] <= capacity,
            "Gaussian cursor lies outside the resident physical tape"
        );
        const int64_t remaining = capacity
            - (raw_before[offset] - base_before[offset]);
        refill_required = refill_required
            || remaining < required_future_draws;
    }
    if (!refill_required) {
        pybind11::dict out;
        out["schema_version"] =
            "eiel_cuda_architecture_b_native_resident_window_refill_v1";
        out["operation"] = "reuse_cpython_gaussian_window_v1";
        if constexpr (kOptionDM0TelemetryEnabledV1) {
            out["option_d_m0_telemetry_enabled"] = true;
        }
        out["batch_cells"] = cells;
        out["physical_capacity"] = capacity;
        out["required_future_draws"] = required_future_draws;
        out["refill_performed"] = false;
        out["resident_window_reused"] = true;
        out["absolute_raw_draws_before"] = int64_values_to_list_v1(raw_before);
        out["window_base_before"] = int64_values_to_list_v1(base_before);
        out["window_base_after"] = int64_values_to_list_v1(base_before);
        out["absolute_raw_draws_preserved"] = true;
        out["semantic_draws_preserved"] = true;
        out["source_rng_state_preserved"] = true;
        out["host_to_device_operations_lower_bound"] = static_cast<int64_t>(0);
        out["device_to_host_operations_lower_bound"] = static_cast<int64_t>(2);
        out["primary_segment_bytes"] = static_cast<int64_t>(0);
        out["secondary_segment_bytes"] = static_cast<int64_t>(0);
        out["window_base_bytes"] = static_cast<int64_t>(0);
        out["host_wall_time_ns"] = steady_ns_v1() - ensure_start_ns;
        out["device_to_host_time_ns"] = coverage_check_time_ns;
        out["host_to_device_time_ns"] = static_cast<int64_t>(0);
        out["gaussian_raw_generation_time_ns"] = static_cast<int64_t>(0);
        out["gaussian_transcendental_conversion_time_ns"] =
            static_cast<int64_t>(0);
        out["gaussian_allocation_copy_time_ns"] = static_cast<int64_t>(0);
        out["gaussian_state_preparation_time_ns"] = static_cast<int64_t>(0);
        out["gaussian_tape_allocation_time_ns"] = static_cast<int64_t>(0);
        out["gaussian_stack_pack_time_ns"] = static_cast<int64_t>(0);
        out["primary_h2d_submission_time_ns"] = static_cast<int64_t>(0);
        out["secondary_h2d_submission_time_ns"] = static_cast<int64_t>(0);
        out["window_base_h2d_submission_time_ns"] = static_cast<int64_t>(0);
        return out;
    }

    pybind11::dict out = refill_cpython_gaussian_window_v1(
        words,
        index,
        has_gauss,
        gauss_next,
        std::move(primary),
        std::move(secondary),
        absolute_raw_draws,
        semantic_draws,
        std::move(window_base)
    );
    out["required_future_draws"] = required_future_draws;
    out["refill_performed"] = true;
    out["resident_window_reused"] = false;
    out["device_to_host_operations_lower_bound"] =
        pybind11::cast<int64_t>(out["device_to_host_operations_lower_bound"])
        + 2;
    out["device_to_host_time_ns"] =
        pybind11::cast<int64_t>(out["device_to_host_time_ns"])
        + coverage_check_time_ns;
    out["host_wall_time_ns"] = steady_ns_v1() - ensure_start_ns;
    return out;
}

inline pybind11::dict refill_fixed_mlp_policy_window_v1(
    const torch::Tensor& source_state,
    torch::Tensor delta,
    torch::Tensor mutation_mask,
    torch::Tensor crossover_mask_packed,
    torch::Tensor post_state_packed,
    torch::Tensor semantic_tape,
    const torch::Tensor& absolute_cursor,
    const torch::Tensor& semantic_counter,
    torch::Tensor window_base,
    const int64_t policy_row_width,
    const bool mutation_enable,
    const std::string& mutation_mode,
    const double mutation_rate,
    const double mutation_std,
    const std::string& execution_route_sha256 = std::string()
) {
    const int64_t refill_start_ns = steady_ns_v1();
    TORCH_CHECK(source_state.defined() && source_state.dim() == 2, "policy RNG state must be batched rank-2");
    TORCH_CHECK(source_state.scalar_type() == torch::kUInt8, "policy RNG state must be uint8");
    TORCH_CHECK(source_state.is_contiguous(), "policy RNG state must be contiguous");
    TORCH_CHECK(!source_state.requires_grad(), "policy RNG state must not require gradients");
    TORCH_CHECK(delta.defined() && delta.dim() == 3, "policy delta tape must be batched rank-3");
    const int64_t cells = source_state.size(0);
    const int64_t state_bytes = source_state.size(1);
    const int64_t episodes = delta.size(1);
    const int64_t mutation_width = delta.size(2);
    TORCH_CHECK(crossover_mask_packed.defined() && crossover_mask_packed.dim() == 3, "packed policy crossover mask must be batched rank-3");
    TORCH_CHECK(crossover_mask_packed.scalar_type() == torch::kUInt8, "packed policy crossover mask must be uint8");
    const int64_t packed_width = crossover_mask_packed.size(2);
    const int64_t width = policy_row_width;
    TORCH_CHECK(cells > 0 && episodes > 0 && mutation_width > 0 && width > 0, "policy resident dimensions must be positive");
    TORCH_CHECK(state_bytes == torch_rng_pack::kFullStateBytes, "policy RNG source state must use canonical 5056-byte rows");
    TORCH_CHECK(
        mutation_width == 1 || mutation_width == width,
        "policy mutation tape width must be compact width 1 or full policy width"
    );
    TORCH_CHECK(episodes <= kPolicyEpisodeCapacityLimitV1, "policy episode capacity exceeds the bounded range");
    TORCH_CHECK(std::isfinite(mutation_rate) && mutation_rate >= 0.0 && mutation_rate <= 1.0, "policy mutation rate must be in [0,1]");
    TORCH_CHECK(std::isfinite(mutation_std) && mutation_std >= 0.0, "policy mutation standard deviation must be nonnegative");
    const bool enabled = mutation_enable && mutation_std > 0.0;
    auto abi_receipt = torch_rng_post_state_abi_receipt_v1(
        execution_route_sha256
    );
    TORCH_CHECK(
        !enabled || mutation_width == width,
        "compact policy mutation tape requires disabled policy mutation"
    );
    require_batched_tensor_v1(delta, source_state, torch::kFloat32, {cells, episodes, mutation_width}, "policy_birth_delta_tape");
    require_batched_tensor_v1(mutation_mask, source_state, torch::kBool, {cells, episodes, mutation_width}, "policy_birth_mutation_mask_tape");
    require_batched_tensor_v1(crossover_mask_packed, source_state, torch::kUInt8, {cells, episodes, packed_crossover_row_bytes_v1(width)}, "policy_birth_crossover_mask_packed_tape");
    const bool replay_post_state_rows = mutation_width == 1;
    if (replay_post_state_rows) {
        require_batched_tensor_v1(post_state_packed, source_state, torch::kUInt8, {cells, episodes, 0}, "policy_birth_post_rng_state_packed_tape");
    } else {
        require_batched_tensor_v1(post_state_packed, source_state, torch::kUInt8, {cells, episodes, torch_rng_pack::kPackedStateBytes}, "policy_birth_post_rng_state_packed_tape");
    }
    require_batched_tensor_v1(semantic_tape, source_state, torch::kInt64, {cells, episodes}, "policy_birth_semantic_draws_tape");
    require_batched_tensor_v1(absolute_cursor, source_state, torch::kInt64, {cells}, "policy_birth_tape_cursor");
    require_batched_tensor_v1(semantic_counter, source_state, torch::kInt64, {cells}, "mutation_torch_cpu_semantic_draws");
    require_batched_tensor_v1(window_base, source_state, torch::kInt64, {cells}, "policy_birth_window_base");

    const bool m4_device_refill_surface_supported =
        !mutation_enable && !enabled && mutation_width == 1
        && replay_post_state_rows && mutation_mode == "dense";
    if constexpr (kOptionDM4DevicePolicyRefillEnabledV1) {
    if (m4_device_refill_surface_supported) {
        const int64_t device_refill_start_ns = steady_ns_v1();
        const auto result =
            option_d_m4_policy_refill::generate_compact_policy_window_v1(
                source_state,
                delta,
                mutation_mask,
                crossover_mask_packed,
                post_state_packed,
                semantic_tape,
                absolute_cursor,
                semantic_counter,
                window_base,
                width
            );
        const int64_t device_refill_time_ns =
            steady_ns_v1() - device_refill_start_ns;
        TORCH_CHECK(
            result.invalid_state_count == 0
                && result.invalid_counter_count == 0,
            "Option D M4 device policy refill rejected source state or counters: "
            "invalid_states=",
            result.invalid_state_count,
            " invalid_counters=",
            result.invalid_counter_count
        );

        option_d_m2_policy_shadow::PolicySegmentShadowResultV1 shadow_result{};
        if constexpr (kOptionDM2PolicyShadowEnabledV1) {
            shadow_result =
                option_d_m2_policy_shadow::verify_policy_segment_shadow_v1(
                    source_state,
                    crossover_mask_packed,
                    semantic_tape,
                    width
                );
            TORCH_CHECK(
                shadow_result.mismatch_count == 0
                    && shadow_result.invalid_state_count == 0
                    && shadow_result.invalid_semantic_count == 0,
                "Option D M2 policy segment shadow mismatch after M4 refill: mismatches=",
                shadow_result.mismatch_count,
                " invalid_states=",
                shadow_result.invalid_state_count,
                " invalid_semantic_rows=",
                shadow_result.invalid_semantic_count
            );
        }

        pybind11::dict out;
        out["schema_version"] =
            "eiel_cuda_architecture_b_native_resident_window_refill_v1";
        out["operation"] = "refill_fixed_mlp_policy_window_v1";
        if constexpr (kOptionDM0TelemetryEnabledV1) {
            out["option_d_m0_telemetry_enabled"] = true;
        }
        out["option_d_m2_policy_shadow_enabled"] =
            kOptionDM2PolicyShadowEnabledV1;
        out["option_d_m2_policy_shadow_authority"] =
            kOptionDM2PolicyShadowEnabledV1
            ? "non_authoritative_compare_only"
            : "disabled";
        out["option_d_m4_device_policy_refill_enabled"] = true;
        out["option_d_m4_device_policy_refill_authority"] =
            "candidate_device_generated_compact_refill";
        out["option_d_m4_device_policy_refill_schema_version"] =
            "eiel_option_d_m4_device_policy_refill_v1";
        out["option_d_m4_device_policy_refill_generated_rows"] =
            result.generated_rows;
        out["option_d_m4_device_policy_refill_generated_bytes"] =
            result.generated_bytes;
        out["option_d_m4_device_policy_refill_invalid_state_count"] =
            result.invalid_state_count;
        out["option_d_m4_device_policy_refill_invalid_counter_count"] =
            result.invalid_counter_count;
        out["option_d_m4_device_policy_refill_materialized_float32_crossover"] =
            false;
        out["option_d_m4_device_policy_refill_persistent_rng_state_written"] =
            false;
        out["option_d_m7_batched_cpu_policy_refill_enabled"] = false;
        out["option_d_m7_batched_cpu_policy_refill_used"] = false;
        out["option_d_m7_batched_cpu_policy_refill_authority"] = "disabled";
        out["batch_cells"] = cells;
        out["physical_capacity"] = episodes;
        out["policy_row_width"] = width;
        out["policy_crossover_mask_encoding"] = "bitpack_lsb0_v1";
        out["policy_crossover_packed_row_bytes"] =
            packed_crossover_row_bytes_v1(width);
        out["policy_birth_post_rng_state_encoding"] = torch_rng_pack::kEncoding;
        out["policy_birth_post_rng_state_abi_schema_version"] =
            torch_rng_pack::kAbiSchemaVersion;
        out["policy_birth_post_rng_state_abi_fingerprint_sha256"] =
            torch_rng_pack::kAbiFingerprintSha256;
        out["policy_birth_post_rng_state_abi"] = abi_receipt;
        out["policy_birth_post_rng_state_route_binding_status"] =
            abi_receipt["route_binding_status"];
        out["policy_birth_post_rng_state_execution_route_sha256"] =
            abi_receipt["execution_route_sha256"];
        out["policy_birth_post_rng_state_qualified_torch_version"] =
            torch_rng_pack::kQualifiedTorchVersion;
        out["policy_birth_post_rng_state_qualified_torch_git_version"] =
            torch_rng_pack::kQualifiedTorchGitVersion;
        out["policy_birth_post_rng_state_qualified_build_config_sha256"] =
            torch_rng_pack::kQualifiedTorchBuildConfigSha256;
        out["policy_birth_post_rng_state_byteorder"] =
            torch_rng_pack::kQualifiedByteOrder;
        out["policy_birth_post_rng_state_logical_row_bytes"] =
            torch_rng_pack::kFullStateBytes;
        out["policy_birth_post_rng_state_packed_row_bytes"] =
            torch_rng_pack::kPackedStateBytes;
        out["policy_birth_post_rng_state_dense_allocation_materialized"] = false;
        out["policy_birth_post_rng_state_resident_rows_materialized"] = false;
        out["policy_birth_post_rng_state_replay_schema_version"] =
            "eiel_torch_cpu_mt19937_replay_v1";
        out["absolute_cursor_preserved"] = true;
        out["semantic_draws_preserved"] = true;
        out["source_rng_state_preserved"] = true;
        out["host_wall_time_ns"] = steady_ns_v1() - refill_start_ns;
        out["device_to_host_time_ns"] = device_refill_time_ns;
        out["host_to_device_time_ns"] = static_cast<int64_t>(0);
        out["torch_random_generation_time_ns"] = static_cast<int64_t>(0);
        out["torch_compare_pack_time_ns"] = static_cast<int64_t>(0);
        out["torch_randn_generation_time_ns"] = static_cast<int64_t>(0);
        out["policy_allocation_stack_copy_time_ns"] = static_cast<int64_t>(0);
        if constexpr (kOptionDM0TelemetryEnabledV1) {
            out["torch_compare_time_ns"] = static_cast<int64_t>(0);
            out["crossover_pack_time_ns"] = static_cast<int64_t>(0);
        }
        out["device_to_host_operations_lower_bound"] = static_cast<int64_t>(1);
        out["host_to_device_operations_lower_bound"] = static_cast<int64_t>(0);
        out["device_to_host_bytes_lower_bound"] =
            static_cast<int64_t>(4 * sizeof(int64_t));
        out["host_to_device_bytes_lower_bound"] = static_cast<int64_t>(0);
        if constexpr (kOptionDM2PolicyShadowEnabledV1) {
            out["option_d_m2_policy_shadow_schema_version"] =
                "eiel_option_d_m2_policy_segment_shadow_v1";
            out["option_d_m2_policy_shadow_checked_rows"] =
                shadow_result.checked_rows;
            out["option_d_m2_policy_shadow_checked_bytes"] =
                shadow_result.checked_bytes;
            out["option_d_m2_policy_shadow_mismatch_count"] =
                shadow_result.mismatch_count;
            out["option_d_m2_policy_shadow_invalid_state_count"] =
                shadow_result.invalid_state_count;
            out["option_d_m2_policy_shadow_invalid_semantic_count"] =
                shadow_result.invalid_semantic_count;
            out["option_d_m2_policy_shadow_materialized_float32_crossover"] =
                false;
            out["option_d_m2_policy_shadow_persistent_rng_state_written"] =
                false;
        }
        return out;
    }
    }

    const int64_t d2h_start_ns = steady_ns_v1();
    auto source_host = source_state.detach().cpu().contiguous();
    const auto cursor_before = host_int64_values_v1(absolute_cursor, "policy_birth_tape_cursor");
    const auto semantic_before = host_int64_values_v1(semantic_counter, "mutation_torch_cpu_semantic_draws");
    const auto base_before = host_int64_values_v1(window_base, "policy_birth_window_base");
    const int64_t device_to_host_time_ns = steady_ns_v1() - d2h_start_ns;
    std::vector<torch::Tensor> delta_rows;
    std::vector<torch::Tensor> mutation_rows;
    std::vector<torch::Tensor> crossover_packed_rows;
    std::vector<torch::Tensor> packed_post_state_rows;
    std::vector<torch::Tensor> semantic_rows;
    delta_rows.reserve(static_cast<std::size_t>(cells));
    mutation_rows.reserve(static_cast<std::size_t>(cells));
    crossover_packed_rows.reserve(static_cast<std::size_t>(cells));
    packed_post_state_rows.reserve(static_cast<std::size_t>(cells));
    semantic_rows.reserve(static_cast<std::size_t>(cells));
    int64_t torch_random_generation_time_ns = 0;
    int64_t torch_compare_time_ns = 0;
    int64_t crossover_pack_time_ns = 0;
    int64_t torch_randn_generation_time_ns = 0;
    int64_t stack_allocation_time_ns = 0;
    for (int64_t cell = 0; cell < cells; ++cell) {
        TORCH_CHECK(cursor_before[static_cast<std::size_t>(cell)] >= 0, "policy_birth_tape_cursor must be nonnegative");
        TORCH_CHECK(semantic_before[static_cast<std::size_t>(cell)] >= 0, "mutation_torch_cpu_semantic_draws must be nonnegative");
        TORCH_CHECK(
            base_before[static_cast<std::size_t>(cell)] >= 0
                && base_before[static_cast<std::size_t>(cell)]
                    <= cursor_before[static_cast<std::size_t>(cell)],
            "policy_birth_window_base must be a nonnegative physical origin "
            "at most the tape cursor"
        );
        // RNGV2 subjects use the canonical semantic origin carried by the
        // mutation stream.  The policy-birth tape cursor is a physical tape
        // position and can be rebased when a fresh owner hydrates a
        // checkpoint; using it as the subject changes the semantic address at
        // a continuation boundary.
        auto prepared = hapb_rngv2_policy_window_enabled_v1()
            ? make_fixed_mlp_policy_window_rngv2_v1(
                source_host[cell].clone().contiguous(),
                episodes,
                width,
                mutation_width,
                mutation_enable,
                mutation_rate,
                mutation_std,
                // World uid is the world, never the physical cell index.
                // Addressing by slot made force-vs-reuse diverge after turnover.
                0u,
                static_cast<uint64_t>(semantic_before[static_cast<std::size_t>(cell)])
            )
            : make_fixed_mlp_policy_window_v1(
                source_host[cell].clone().contiguous(),
                episodes,
                width,
                mutation_width,
                mutation_enable,
                mutation_mode,
                mutation_rate,
                mutation_std,
                execution_route_sha256
            );
        torch_random_generation_time_ns += prepared.torch_random_generation_time_ns;
        torch_compare_time_ns += prepared.torch_compare_time_ns;
        crossover_pack_time_ns += prepared.crossover_pack_time_ns;
        torch_randn_generation_time_ns += prepared.torch_randn_generation_time_ns;
        stack_allocation_time_ns += prepared.stack_allocation_time_ns;
        delta_rows.push_back(std::move(prepared.delta));
        mutation_rows.push_back(std::move(prepared.mutation_mask));
        crossover_packed_rows.push_back(std::move(prepared.crossover_mask_packed));
        packed_post_state_rows.push_back(std::move(prepared.post_state_packed));
        semantic_rows.push_back(std::move(prepared.semantic_draws));
    }
    int64_t host_to_device_time_ns = 0;
    const int64_t stack_start_ns = steady_ns_v1();
    auto stacked_delta = torch::stack(delta_rows).contiguous();
    auto stacked_mutation = torch::stack(mutation_rows).contiguous();
    auto stacked_crossover = torch::stack(crossover_packed_rows).contiguous();
    auto stacked_post_state = torch::stack(packed_post_state_rows).contiguous();
    auto stacked_semantic = torch::stack(semantic_rows).contiguous();
    if constexpr (kOptionDM0TelemetryEnabledV1) {
        stack_allocation_time_ns += steady_ns_v1() - stack_start_ns;
    }
    const int64_t copy_start_ns = steady_ns_v1();
    delta.copy_(stacked_delta);
    mutation_mask.copy_(stacked_mutation);
    crossover_mask_packed.copy_(stacked_crossover);
    post_state_packed.copy_(stacked_post_state);
    semantic_tape.copy_(stacked_semantic);
    window_base.copy_(absolute_cursor);
    if constexpr (kOptionDM0TelemetryEnabledV1) {
        host_to_device_time_ns += steady_ns_v1() - copy_start_ns;
    }

    option_d_m2_policy_shadow::PolicySegmentShadowResultV1 shadow_result{};
    if constexpr (kOptionDM2PolicyShadowEnabledV1) {
        shadow_result =
            option_d_m2_policy_shadow::verify_policy_segment_shadow_v1(
                source_state,
                crossover_mask_packed,
                semantic_tape,
                width
            );
        TORCH_CHECK(
            shadow_result.mismatch_count == 0
                && shadow_result.invalid_state_count == 0
                && shadow_result.invalid_semantic_count == 0,
            "Option D M2 policy segment shadow mismatch: mismatches=",
            shadow_result.mismatch_count,
            " invalid_states=",
            shadow_result.invalid_state_count,
            " invalid_semantic_rows=",
            shadow_result.invalid_semantic_count
        );
    }

    pybind11::dict out;
    out["schema_version"] = "eiel_cuda_architecture_b_native_resident_window_refill_v1";
    out["operation"] = "refill_fixed_mlp_policy_window_v1";
    if constexpr (kOptionDM0TelemetryEnabledV1) {
        out["option_d_m0_telemetry_enabled"] = true;
    }
    out["option_d_m2_policy_shadow_enabled"] =
        kOptionDM2PolicyShadowEnabledV1;
    out["option_d_m2_policy_shadow_authority"] =
        kOptionDM2PolicyShadowEnabledV1
        ? "non_authoritative_compare_only"
        : "disabled";
    out["option_d_m4_device_policy_refill_enabled"] =
        kOptionDM4DevicePolicyRefillEnabledV1;
    out["option_d_m4_device_policy_refill_authority"] =
        kOptionDM4DevicePolicyRefillEnabledV1
        ? "compiled_candidate_not_used_for_surface"
        : "disabled";
    const bool m7_batched_cpu_refill_used =
        kOptionDM7BatchedCpuPolicyRefillEnabledV1
        && !mutation_enable && !enabled && mutation_width == 1
        && replay_post_state_rows && mutation_mode == "dense";
    out["option_d_m7_batched_cpu_policy_refill_enabled"] =
        kOptionDM7BatchedCpuPolicyRefillEnabledV1;
    out["option_d_m7_batched_cpu_policy_refill_used"] =
        m7_batched_cpu_refill_used;
    out["option_d_m7_batched_cpu_policy_refill_authority"] =
        m7_batched_cpu_refill_used
        ? "candidate_batched_exact_cpu_compact_refill"
        : (
            kOptionDM7BatchedCpuPolicyRefillEnabledV1
            ? "compiled_candidate_not_used_for_surface"
            : "disabled"
        );
    if (m7_batched_cpu_refill_used) {
        out["option_d_m7_batched_cpu_policy_refill_schema_version"] =
            "eiel_option_d_m7_batched_cpu_policy_refill_v1";
    } else {
        out["option_d_m7_batched_cpu_policy_refill_schema_version"] =
            pybind11::none();
    }
    out["batch_cells"] = cells;
    out["physical_capacity"] = episodes;
    out["policy_row_width"] = width;
    out["policy_crossover_mask_encoding"] = "bitpack_lsb0_v1";
    out["policy_crossover_packed_row_bytes"] = packed_crossover_row_bytes_v1(width);
    out["policy_birth_post_rng_state_encoding"] = torch_rng_pack::kEncoding;
    out["policy_birth_post_rng_state_abi_schema_version"] =
        torch_rng_pack::kAbiSchemaVersion;
    out["policy_birth_post_rng_state_abi_fingerprint_sha256"] =
        torch_rng_pack::kAbiFingerprintSha256;
    out["policy_birth_post_rng_state_abi"] = abi_receipt;
    out["policy_birth_post_rng_state_route_binding_status"] =
        abi_receipt["route_binding_status"];
    out["policy_birth_post_rng_state_execution_route_sha256"] =
        abi_receipt["execution_route_sha256"];
    out["policy_birth_post_rng_state_qualified_torch_version"] =
        torch_rng_pack::kQualifiedTorchVersion;
    out["policy_birth_post_rng_state_qualified_torch_git_version"] =
        torch_rng_pack::kQualifiedTorchGitVersion;
    out["policy_birth_post_rng_state_qualified_build_config_sha256"] =
        torch_rng_pack::kQualifiedTorchBuildConfigSha256;
    out["policy_birth_post_rng_state_byteorder"] =
        torch_rng_pack::kQualifiedByteOrder;
    out["policy_birth_post_rng_state_logical_row_bytes"] =
        torch_rng_pack::kFullStateBytes;
    out["policy_birth_post_rng_state_packed_row_bytes"] =
        torch_rng_pack::kPackedStateBytes;
    out["policy_birth_post_rng_state_dense_allocation_materialized"] = false;
    out["policy_birth_post_rng_state_resident_rows_materialized"] =
        !replay_post_state_rows;
    if constexpr (kOptionDM2PolicyShadowEnabledV1) {
        out["option_d_m2_policy_shadow_schema_version"] =
            "eiel_option_d_m2_policy_segment_shadow_v1";
        out["option_d_m2_policy_shadow_checked_rows"] =
            shadow_result.checked_rows;
        out["option_d_m2_policy_shadow_checked_bytes"] =
            shadow_result.checked_bytes;
        out["option_d_m2_policy_shadow_mismatch_count"] =
            shadow_result.mismatch_count;
        out["option_d_m2_policy_shadow_invalid_state_count"] =
            shadow_result.invalid_state_count;
        out["option_d_m2_policy_shadow_invalid_semantic_count"] =
            shadow_result.invalid_semantic_count;
        out["option_d_m2_policy_shadow_materialized_float32_crossover"] =
            false;
        out["option_d_m2_policy_shadow_persistent_rng_state_written"] =
            false;
    }
    if (replay_post_state_rows) {
        out["policy_birth_post_rng_state_replay_schema_version"] =
            "eiel_torch_cpu_mt19937_replay_v1";
    } else {
        out["policy_birth_post_rng_state_replay_schema_version"] =
            pybind11::none();
    }
    out["absolute_cursor_before"] = int64_values_to_list_v1(cursor_before);
    out["semantic_draws_before"] = int64_values_to_list_v1(semantic_before);
    out["window_base_before"] = int64_values_to_list_v1(base_before);
    out["window_base_after"] = int64_values_to_list_v1(cursor_before);
    out["absolute_cursor_preserved"] = true;
    out["semantic_draws_preserved"] = true;
    out["source_rng_state_preserved"] = true;
    out["host_wall_time_ns"] = steady_ns_v1() - refill_start_ns;
    out["device_to_host_time_ns"] = device_to_host_time_ns;
    out["host_to_device_time_ns"] = host_to_device_time_ns;
    out["torch_random_generation_time_ns"] = torch_random_generation_time_ns;
    out["torch_compare_pack_time_ns"] =
        torch_compare_time_ns + crossover_pack_time_ns;
    out["torch_randn_generation_time_ns"] = torch_randn_generation_time_ns;
    out["policy_allocation_stack_copy_time_ns"] =
        stack_allocation_time_ns + host_to_device_time_ns;
    if constexpr (kOptionDM0TelemetryEnabledV1) {
        out["torch_compare_time_ns"] = torch_compare_time_ns;
        out["crossover_pack_time_ns"] = crossover_pack_time_ns;
    }
    out["device_to_host_operations_lower_bound"] = static_cast<int64_t>(4);
    out["host_to_device_operations_lower_bound"] = static_cast<int64_t>(6);
    out["device_to_host_bytes_lower_bound"] =
        static_cast<int64_t>(source_host.numel() * source_host.element_size())
        + static_cast<int64_t>(absolute_cursor.numel() * absolute_cursor.element_size())
        + static_cast<int64_t>(semantic_counter.numel() * semantic_counter.element_size())
        + static_cast<int64_t>(window_base.numel() * window_base.element_size());
    out["host_to_device_bytes_lower_bound"] =
        static_cast<int64_t>(delta.numel() * delta.element_size())
        + static_cast<int64_t>(mutation_mask.numel() * mutation_mask.element_size())
        + static_cast<int64_t>(crossover_mask_packed.numel() * crossover_mask_packed.element_size())
        + static_cast<int64_t>(post_state_packed.numel() * post_state_packed.element_size())
        + static_cast<int64_t>(semantic_tape.numel() * semantic_tape.element_size())
        + static_cast<int64_t>(window_base.numel() * window_base.element_size());
    return out;
}

inline pybind11::dict refill_fixed_mlp_policy_window_with_identity_v1(
    const torch::Tensor& source_state,
    torch::Tensor delta,
    torch::Tensor mutation_mask,
    torch::Tensor crossover_mask_packed,
    torch::Tensor post_state_packed,
    torch::Tensor semantic_tape,
    const torch::Tensor& absolute_cursor,
    const torch::Tensor& semantic_counter,
    torch::Tensor window_base,
    const int64_t policy_row_width,
    const bool mutation_enable,
    const std::string& mutation_mode,
    const double mutation_rate,
    const double mutation_std,
    const std::string& execution_route_sha256,
    PolicyWindowIdentityStateV1* const frag_identity_state
) {
    const bool identity_surface =
        frag_identity_state != nullptr
        && hapb_frag_a_policy_window_reuse_requested_v1()
        && delta.defined() && delta.dim() == 3 && delta.size(2) == 1
        && !mutation_enable && mutation_mode == "dense"
        && mutation_rate == 0.0 && mutation_std == 0.0;
    PolicyWindowConfigurationIdentityV1 configuration;
    std::array<torch::Tensor, 9> tensors;
    if (identity_surface) {
        require_policy_window_execution_lease_storage_v1(
            frag_identity_state, crossover_mask_packed
        );
        configuration = policy_window_configuration_identity_v1(
            source_state,
            delta,
            policy_row_width,
            mutation_enable,
            mutation_mode,
            mutation_rate,
            mutation_std,
            execution_route_sha256
        );
        tensors = policy_window_identity_tensors_v1(
            source_state,
            delta,
            mutation_mask,
            crossover_mask_packed,
            post_state_packed,
            semantic_tape,
            absolute_cursor,
            semantic_counter,
            window_base
        );
        // Invalidate before any tape/base write.  An exception leaves the
        // companion invalid and cannot authorize an arithmetic reuse.
        invalidate_policy_window_identity_v1(
            frag_identity_state, false, true
        );
    }
    pybind11::dict out = refill_fixed_mlp_policy_window_v1(
        source_state,
        std::move(delta),
        std::move(mutation_mask),
        std::move(crossover_mask_packed),
        std::move(post_state_packed),
        std::move(semantic_tape),
        absolute_cursor,
        semantic_counter,
        std::move(window_base),
        policy_row_width,
        mutation_enable,
        mutation_mode,
        mutation_rate,
        mutation_std,
        execution_route_sha256
    );
    if (identity_surface) {
        const auto cursor = int64_values_from_list_v1(
            out["absolute_cursor_before"],
            configuration.cells,
            "absolute_cursor_before"
        );
        const auto semantic_draws = int64_values_from_list_v1(
            out["semantic_draws_before"],
            configuration.cells,
            "semantic_draws_before"
        );
        publish_policy_window_identity_v1(
            frag_identity_state,
            tensors,
            configuration,
            cursor,
            semantic_draws
        );
        out["frag_identity_contract_applied"] = true;
        out["frag_identity_result"] = "physical_refill_identity_published";
        out["frag_identity_tape_generation"] =
            frag_identity_state->tape_storage_generation;
        out["frag_identity_rng_lineage_epoch"] =
            frag_identity_state->rng_lineage_epoch;
    }
    return out;
}

inline pybind11::dict ensure_fixed_mlp_policy_window_with_identity_v1(
    const torch::Tensor& source_state,
    torch::Tensor delta,
    torch::Tensor mutation_mask,
    torch::Tensor crossover_mask_packed,
    torch::Tensor post_state_packed,
    torch::Tensor semantic_tape,
    const torch::Tensor& absolute_cursor,
    const torch::Tensor& semantic_counter,
    torch::Tensor window_base,
    const int64_t policy_row_width,
    const bool mutation_enable,
    const std::string& mutation_mode,
    const double mutation_rate,
    const double mutation_std,
    const int64_t required_future_episodes,
    const std::string& execution_route_sha256 = std::string(),
    PolicyWindowIdentityStateV1* const frag_identity_state = nullptr,
    const std::string& identity_boundary_event = "ensure_after_consume"
) {
    const int64_t ensure_start_ns = steady_ns_v1();
    std::optional<pybind11::list> frag_identity_trace;
    if constexpr (kOptionDM8ResidentPolicySuffixReuseEnabledV1) {
        TORCH_CHECK(
            !hapb_frag_a_policy_window_reuse_requested_v1(),
            "HAPB identity-bound reuse cannot be combined with legacy M8"
        );
    }
    if constexpr (!kOptionDM8ResidentPolicySuffixReuseEnabledV1) {
        const bool frag_identity_requested =
            hapb_frag_a_policy_window_reuse_requested_v1();
        const bool compact_disabled_mutation =
            delta.defined() && delta.dim() == 3 && delta.size(2) == 1
            && !mutation_enable && mutation_mode == "dense"
            && mutation_rate == 0.0 && mutation_std == 0.0
            && required_future_episodes >= 0;
        const bool frag_identity_contract_applied =
            frag_identity_requested && compact_disabled_mutation;
        if (frag_identity_contract_applied && frag_identity_state != nullptr) {
            require_policy_window_execution_lease_storage_v1(
                frag_identity_state, crossover_mask_packed
            );
        }
        std::vector<int64_t> cursor_before;
        std::vector<int64_t> semantic_before;
        std::vector<int64_t> base_before;
        std::vector<int64_t> remaining_episodes_by_cell;
        std::vector<std::string_view> decision_reasons;
        std::vector<uint8_t> cell_identity_verified;
        int64_t minimum_remaining_episodes = 0;
        int64_t refill_trigger_cell_count = 0;
        bool physical_refill_required = true;
        int64_t identity_check_device_to_host_time_ns = 0;
        PolicyWindowConfigurationIdentityV1 current_configuration;
        std::array<torch::Tensor, 9> current_tensors;
        const bool identity_trace_enabled =
            frag_identity_contract_applied
            && hapb_frag_identity_trace_enabled_v1();
        const bool force_physical_refill =
            hapb_frag_a_force_physical_policy_refill_v1();
        if (frag_identity_contract_applied) {
            const int64_t cells = source_state.size(0);
            const int64_t capacity = delta.size(1);
            TORCH_CHECK(
                required_future_episodes <= capacity,
                "required future policy episodes must fit the physical tape"
            );
            const int64_t identity_check_started_ns = steady_ns_v1();
            cursor_before = host_int64_values_v1(
                absolute_cursor, "policy_birth_tape_cursor"
            );
            semantic_before = host_int64_values_v1(
                semantic_counter, "mutation_torch_cpu_semantic_draws"
            );
            base_before = host_int64_values_v1(
                window_base, "policy_birth_window_base"
            );
            identity_check_device_to_host_time_ns =
                steady_ns_v1() - identity_check_started_ns;
            current_configuration = policy_window_configuration_identity_v1(
                source_state,
                delta,
                policy_row_width,
                mutation_enable,
                mutation_mode,
                mutation_rate,
                mutation_std,
                execution_route_sha256
            );
            current_tensors = policy_window_identity_tensors_v1(
                source_state,
                delta,
                mutation_mask,
                crossover_mask_packed,
                post_state_packed,
                semantic_tape,
                absolute_cursor,
                semantic_counter,
                window_base
            );
            physical_refill_required = force_physical_refill;
            minimum_remaining_episodes = capacity;
            remaining_episodes_by_cell.reserve(static_cast<std::size_t>(cells));
            if (identity_trace_enabled) {
                decision_reasons.reserve(static_cast<std::size_t>(cells));
                cell_identity_verified.reserve(static_cast<std::size_t>(cells));
            }
            bool tensor_identity_matches = frag_identity_state != nullptr;
            bool configuration_identity_matches = frag_identity_state != nullptr;
            if (frag_identity_state != nullptr) {
                configuration_identity_matches =
                    frag_identity_state->initialized
                    && policy_window_configuration_identity_matches_v1(
                        frag_identity_state->configuration,
                        current_configuration
                    );
                for (std::size_t index = 0; index < current_tensors.size(); ++index) {
                    tensor_identity_matches = tensor_identity_matches
                        && policy_window_tensor_identity_matches_v1(
                            frag_identity_state->tensors[index],
                            current_tensors[index]
                        );
                }
            }
            for (int64_t cell = 0; cell < cells; ++cell) {
                const auto offset = static_cast<std::size_t>(cell);
                bool identity_verified = true;
                std::string_view reason = "identity_and_suffix_verified";
                TORCH_CHECK(
                    base_before[offset] >= 0,
                    "policy_birth_window_base must be a nonnegative physical origin"
                );
                {
                    TORCH_CHECK(
                        cursor_before[offset] >= base_before[offset]
                            && cursor_before[offset] - base_before[offset]
                                <= capacity,
                        "policy cursor lies outside the resident physical tape"
                    );
                    const int64_t physical_cursor =
                        cursor_before[offset] - base_before[offset];
                    const int64_t remaining = capacity - physical_cursor;
                    minimum_remaining_episodes = std::min(
                        minimum_remaining_episodes, remaining
                    );
                    remaining_episodes_by_cell.push_back(remaining);
                    if (frag_identity_state == nullptr) {
                        identity_verified = false;
                        reason = "identity_state_missing";
                    } else if (!frag_identity_state->initialized) {
                        identity_verified = false;
                        reason = "identity_state_uninitialized";
                    } else if (!configuration_identity_matches) {
                        identity_verified = false;
                        reason = "configuration_identity_mismatch";
                    } else if (!tensor_identity_matches) {
                        identity_verified = false;
                        reason = "tensor_storage_identity_mismatch";
                    } else if (
                        frag_identity_state->window_valid.size()
                            != static_cast<std::size_t>(cells)
                        || frag_identity_state->window_epoch.size()
                            != static_cast<std::size_t>(cells)
                        || frag_identity_state->window_rng_lineage_epoch.size()
                            != static_cast<std::size_t>(cells)
                        || frag_identity_state->window_base.size()
                            != static_cast<std::size_t>(cells)
                        || frag_identity_state->window_base_semantic_draws.size()
                            != static_cast<std::size_t>(cells)
                    ) {
                        identity_verified = false;
                        reason = "identity_state_shape_mismatch";
                    } else if (!frag_identity_state->window_valid[offset]) {
                        identity_verified = false;
                        reason = "window_invalid";
                    } else if (
                        frag_identity_state->tape_storage_generation <= 0
                        || frag_identity_state->window_epoch[offset]
                            != frag_identity_state->tape_storage_generation
                    ) {
                        identity_verified = false;
                        reason = "tape_generation_mismatch";
                    } else if (
                        frag_identity_state->rng_lineage_epoch <= 0
                        || frag_identity_state->window_rng_lineage_epoch[offset]
                            != frag_identity_state->rng_lineage_epoch
                    ) {
                        identity_verified = false;
                        reason = "rng_lineage_mismatch";
                    } else if (
                        frag_identity_state->window_base[offset]
                            != base_before[offset]
                    ) {
                        identity_verified = false;
                        reason = "window_base_identity_mismatch";
                    } else {
                        TORCH_CHECK(
                            frag_identity_state
                                    ->window_base_semantic_draws[offset]
                                >= 0
                                && semantic_before[offset] >= 0,
                            "policy-window semantic identity must be nonnegative"
                        );
                        TORCH_CHECK(
                            policy_row_width == 0
                                || physical_cursor
                                    <= (
                                        std::numeric_limits<int64_t>::max()
                                        - frag_identity_state
                                            ->window_base_semantic_draws[offset]
                                    ) / policy_row_width,
                            "policy-window semantic identity overflow"
                        );
                        const int64_t expected_semantic =
                            frag_identity_state
                                ->window_base_semantic_draws[offset]
                            + physical_cursor * policy_row_width;
                        if (semantic_before[offset] != expected_semantic) {
                            identity_verified = false;
                            reason = "semantic_schedule_identity_mismatch";
                        } else if (remaining < required_future_episodes) {
                            reason = "insufficient_physical_suffix";
                        }
                    }
                }
                const bool cell_requires_refill =
                    !identity_verified
                    || remaining_episodes_by_cell[offset]
                        < required_future_episodes
                    || force_physical_refill;
                if (cell_requires_refill) {
                    physical_refill_required = true;
                    ++refill_trigger_cell_count;
                }
                if (identity_trace_enabled) {
                    cell_identity_verified.push_back(
                        identity_verified ? 1U : 0U
                    );
                    decision_reasons.push_back(
                        force_physical_refill
                        ? std::string_view("forced_physical_control")
                        : reason
                    );
                }
            }
            if (identity_trace_enabled) {
                frag_identity_trace.emplace();
                for (int64_t cell = 0; cell < cells; ++cell) {
                    const auto offset = static_cast<std::size_t>(cell);
                    pybind11::dict row;
                    row["schema_version"] =
                        "eiel_hapb_frag_identity_trace_v1";
                    row["phase"] = "identity_bound_reuse";
                    row["event"] = identity_boundary_event;
                    row["cell"] = cell;
                    row["logical_cursor_before"] = cursor_before[offset];
                    row["logical_cursor_after"] = cursor_before[offset];
                    row["requested_future_episodes"] =
                        required_future_episodes;
                    row["window_base"] = base_before[offset];
                    row["capacity"] = capacity;
                    row["remaining_suffix"] =
                        remaining_episodes_by_cell[offset];
                    row["window_valid"] =
                        frag_identity_state != nullptr
                        && frag_identity_state->window_valid.size()
                            == static_cast<std::size_t>(cells)
                        ? pybind11::cast<bool>(
                            frag_identity_state->window_valid[offset] != 0U
                        )
                        : pybind11::none();
                    row["window_epoch"] =
                        frag_identity_state != nullptr
                        && frag_identity_state->window_epoch.size()
                            == static_cast<std::size_t>(cells)
                        ? pybind11::cast(
                            frag_identity_state->window_epoch[offset]
                        )
                        : pybind11::none();
                    row["rng_identity"] = cell_identity_verified[offset]
                        ? "lineage_and_schedule_verified"
                        : "identity_rejected";
                    // Observation must not add a full source-state D2H
                    // synchronization. The predicate already binds the
                    // closed-writer lineage and semantic schedule.
                    row["source_rng_state_fnv1a64"] = pybind11::none();
                    row["source_rng_content_read_performed"] = false;
                    row["source_rng_identity_basis"] =
                        "closed_writer_lineage_and_semantic_schedule";
                    row["semantic_counter"] = semantic_before[offset];
                    row["tape_identity"] = cell_identity_verified[offset]
                        ? "storage_generation_and_configuration_verified"
                        : "identity_rejected";
                    row["decision"] = physical_refill_required
                        ? "physical_refill"
                        : "reuse";
                    row["reason"] =
                        physical_refill_required
                            && decision_reasons[offset]
                                == "identity_and_suffix_verified"
                        ? "batch_peer_forced_refill"
                        : std::string(decision_reasons[offset]);
                    row["checkpoint_reentry_marker"] =
                        active_policy_window_identity_boundary_v1;
                    row["identity_publication_origin"] =
                        frag_identity_state != nullptr
                        ? policy_window_publication_origin_name_v1(
                            frag_identity_state->publication_origin
                        )
                        : "identity_state_missing";
                    frag_identity_trace->append(row);
                }
            }
            if (!physical_refill_required) {
                pybind11::dict out;
                out["schema_version"] =
                    "eiel_cuda_architecture_b_native_resident_window_refill_v1";
                out["operation"] = "reuse_fixed_mlp_policy_window_v1";
                out["hapb_frag_a_policy_window_reuse_used"] = true;
                out["frag_identity_contract_applied"] = true;
                out["frag_identity_result"] = "identity_and_suffix_verified";
                out["frag_identity_tape_generation"] =
                    frag_identity_state->tape_storage_generation;
                out["frag_identity_rng_lineage_epoch"] =
                    frag_identity_state->rng_lineage_epoch;
                out["option_d_m8_resident_policy_suffix_reuse_enabled"] = false;
                out["option_d_m8_resident_policy_suffix_reuse_used"] = false;
                out["option_d_m8_resident_policy_suffix_reuse_authority"] =
                    "hapb_frag_a_identity_bound_reuse_independent_of_option_d_m8";
                out["option_d_m8_resident_policy_suffix_reuse_schema_version"] =
                    pybind11::none();
                out["batch_cells"] = cells;
                out["physical_capacity"] = capacity;
                out["policy_row_width"] = policy_row_width;
                out["required_future_episodes"] = required_future_episodes;
                out["minimum_remaining_episodes"] = minimum_remaining_episodes;
                out["remaining_episodes_by_cell"] =
                    int64_values_to_list_v1(remaining_episodes_by_cell);
                out["refill_trigger_cell_count"] = refill_trigger_cell_count;
                out["physical_refill_performed"] = false;
                out["resident_window_reused"] = true;
                out["policy_window_hydrated"] = true;
                out["absolute_cursor_before"] =
                    int64_values_to_list_v1(cursor_before);
                out["window_base_before"] =
                    int64_values_to_list_v1(base_before);
                out["window_base_after"] =
                    int64_values_to_list_v1(base_before);
                out["absolute_cursor_preserved"] = true;
                out["semantic_draws_preserved"] = true;
                out["source_rng_state_preserved"] = true;
                out["physical_rows_generated"] = static_cast<int64_t>(0);
                out["host_to_device_tape_bytes"] = static_cast<int64_t>(0);
                out["host_wall_time_ns"] = steady_ns_v1() - ensure_start_ns;
                out["device_to_host_time_ns"] =
                    identity_check_device_to_host_time_ns;
                out["host_to_device_time_ns"] = static_cast<int64_t>(0);
                out["torch_random_generation_time_ns"] = static_cast<int64_t>(0);
                out["torch_compare_pack_time_ns"] = static_cast<int64_t>(0);
                out["torch_randn_generation_time_ns"] = static_cast<int64_t>(0);
                out["policy_allocation_stack_copy_time_ns"] =
                    static_cast<int64_t>(0);
                out["device_to_host_bytes_lower_bound"] =
                    static_cast<int64_t>(
                        absolute_cursor.numel() * absolute_cursor.element_size()
                        + semantic_counter.numel() * semantic_counter.element_size()
                        + window_base.numel() * window_base.element_size()
                    );
                out["host_to_device_bytes_lower_bound"] = static_cast<int64_t>(0);
                out["device_to_host_operations_lower_bound"] =
                    static_cast<int64_t>(3);
                out["host_to_device_operations_lower_bound"] =
                    static_cast<int64_t>(0);
                if (frag_identity_trace.has_value()) {
                    out["frag_identity_trace"] = *frag_identity_trace;
                }
                return out;
            }
        }
        const int64_t physical_rows_generated =
            source_state.size(0) * delta.size(1);
        pybind11::dict out = refill_fixed_mlp_policy_window_with_identity_v1(
            source_state,
            delta,
            mutation_mask,
            crossover_mask_packed,
            post_state_packed,
            semantic_tape,
            absolute_cursor,
            semantic_counter,
            window_base,
            policy_row_width,
            mutation_enable,
            mutation_mode,
            mutation_rate,
            mutation_std,
            execution_route_sha256,
            frag_identity_state
        );
        if (frag_identity_contract_applied && frag_identity_state != nullptr) {
            if (identity_trace_enabled && frag_identity_trace.has_value()) {
                const int64_t cells = source_state.size(0);
                for (int64_t cell = 0; cell < cells; ++cell) {
                    const auto offset = static_cast<std::size_t>(cell);
                    pybind11::dict row;
                    row["schema_version"] =
                        "eiel_hapb_frag_identity_trace_v1";
                    row["phase"] = "identity_bound_reuse";
                    row["event"] = "physical_refill_published";
                    row["cell"] = cell;
                    row["logical_cursor_before"] = cursor_before[offset];
                    row["logical_cursor_after"] = cursor_before[offset];
                    row["requested_future_episodes"] =
                        required_future_episodes;
                    row["window_base"] = cursor_before[offset];
                    row["capacity"] = current_configuration.capacity;
                    row["remaining_suffix"] = current_configuration.capacity;
                    row["window_valid"] = true;
                    row["window_epoch"] =
                        frag_identity_state->window_epoch[offset];
                    row["rng_identity"] = "lineage_bound";
                    row["source_rng_state_fnv1a64"] = pybind11::none();
                    row["semantic_counter"] = semantic_before[offset];
                    row["tape_identity"] =
                        "storage_generation_and_configuration_published";
                    row["decision"] = "hydrate";
                    row["reason"] = "physical_refill_completed";
                    row["checkpoint_reentry_marker"] =
                        active_policy_window_identity_boundary_v1;
                    row["identity_publication_origin"] =
                        "physical_refill_published";
                    frag_identity_trace->append(row);
                }
            }
        }
        out["option_d_m8_resident_policy_suffix_reuse_enabled"] = false;
        out["option_d_m8_resident_policy_suffix_reuse_used"] = false;
        out["option_d_m8_resident_policy_suffix_reuse_authority"] = "disabled";
        out["option_d_m8_resident_policy_suffix_reuse_schema_version"] =
            pybind11::none();
        out["hapb_frag_a_policy_window_reuse_used"] = false;
        out["frag_identity_contract_applied"] = frag_identity_contract_applied;
        out["frag_identity_result"] = frag_identity_contract_applied
            ? (
                frag_identity_state != nullptr
                ? "physical_refill_identity_published"
                : "physical_refill_identity_state_missing"
            )
            : "not_requested";
        if (frag_identity_contract_applied && frag_identity_state != nullptr) {
            out["frag_identity_tape_generation"] =
                frag_identity_state->tape_storage_generation;
            out["frag_identity_rng_lineage_epoch"] =
                frag_identity_state->rng_lineage_epoch;
        }
        out["required_future_episodes"] = required_future_episodes;
        out["physical_refill_performed"] = true;
        out["resident_window_reused"] = false;
        out["refill_trigger_cell_count"] = frag_identity_contract_applied
            ? refill_trigger_cell_count
            : source_state.size(0);
        out["physical_rows_generated"] = physical_rows_generated;
        out["host_to_device_tape_bytes"] = pybind11::cast<int64_t>(
            out["host_to_device_bytes_lower_bound"]
        );
        if (frag_identity_contract_applied) {
            out["device_to_host_bytes_lower_bound"] =
                pybind11::cast<int64_t>(
                    out["device_to_host_bytes_lower_bound"]
                )
                + static_cast<int64_t>(
                    absolute_cursor.numel() * absolute_cursor.element_size()
                    + semantic_counter.numel() * semantic_counter.element_size()
                    + window_base.numel() * window_base.element_size()
                );
            out["device_to_host_time_ns"] =
                pybind11::cast<int64_t>(out["device_to_host_time_ns"])
                + identity_check_device_to_host_time_ns;
            out["device_to_host_operations_lower_bound"] =
                pybind11::cast<int64_t>(
                    out["device_to_host_operations_lower_bound"]
                ) + 3;
        }
        out["host_wall_time_ns"] = steady_ns_v1() - ensure_start_ns;
        if (frag_identity_trace.has_value()) {
            out["frag_identity_trace"] = *frag_identity_trace;
        }
        return out;
    }

    static_assert(
        !kOptionDM8ResidentPolicySuffixReuseEnabledV1
            || kOptionDM7BatchedCpuPolicyRefillEnabledV1,
        "M8 requires M7 as its physical-refill fallback"
    );
    TORCH_CHECK(
        source_state.defined() && source_state.dim() == 2
            && source_state.scalar_type() == torch::kUInt8
            && source_state.is_contiguous() && !source_state.requires_grad(),
        "policy RNG state must be contiguous uint8 rank-2"
    );
    TORCH_CHECK(
        delta.defined() && delta.dim() == 3,
        "policy delta tape must be batched rank-3"
    );
    const int64_t cells = source_state.size(0);
    const int64_t capacity = delta.size(1);
    const int64_t mutation_width = delta.size(2);
    TORCH_CHECK(
        cells > 0 && capacity > 0 && capacity <= kPolicyEpisodeCapacityLimitV1,
        "policy episode capacity is outside the bounded range"
    );
    TORCH_CHECK(
        required_future_episodes >= 0
            && required_future_episodes <= capacity,
        "required future policy episodes must fit the physical tape"
    );
    TORCH_CHECK(
        policy_row_width > 0 && mutation_width == 1
            && !mutation_enable && mutation_mode == "dense"
            && mutation_std == 0.0,
        "Option D M8 requires the compact disabled-mutation dense policy surface"
    );
    TORCH_CHECK(
        source_state.size(1) == torch_rng_pack::kFullStateBytes,
        "policy RNG source state must use canonical 5056-byte rows"
    );
    require_batched_tensor_v1(
        delta,
        source_state,
        torch::kFloat32,
        {cells, capacity, 1},
        "policy_birth_delta_tape"
    );
    require_batched_tensor_v1(
        mutation_mask,
        source_state,
        torch::kBool,
        {cells, capacity, 1},
        "policy_birth_mutation_mask_tape"
    );
    require_batched_tensor_v1(
        crossover_mask_packed,
        source_state,
        torch::kUInt8,
        {cells, capacity, packed_crossover_row_bytes_v1(policy_row_width)},
        "policy_birth_crossover_mask_packed_tape"
    );
    require_batched_tensor_v1(
        post_state_packed,
        source_state,
        torch::kUInt8,
        {cells, capacity, 0},
        "policy_birth_post_rng_state_packed_tape"
    );
    require_batched_tensor_v1(
        semantic_tape,
        source_state,
        torch::kInt64,
        {cells, capacity},
        "policy_birth_semantic_draws_tape"
    );
    require_batched_tensor_v1(
        absolute_cursor,
        source_state,
        torch::kInt64,
        {cells},
        "policy_birth_tape_cursor"
    );
    require_batched_tensor_v1(
        semantic_counter,
        source_state,
        torch::kInt64,
        {cells},
        "mutation_torch_cpu_semantic_draws"
    );
    require_batched_tensor_v1(
        window_base,
        source_state,
        torch::kInt64,
        {cells},
        "policy_birth_window_base"
    );

    const int64_t coverage_check_start_ns = steady_ns_v1();
    const auto cursor_before = host_int64_values_v1(
        absolute_cursor, "policy_birth_tape_cursor"
    );
    const auto base_before = host_int64_values_v1(
        window_base, "policy_birth_window_base"
    );
    const int64_t coverage_check_time_ns =
        steady_ns_v1() - coverage_check_start_ns;
    bool physical_refill_required = false;
    int64_t minimum_remaining_episodes = capacity;
    int64_t refill_trigger_cell_count = 0;
    std::vector<int64_t> remaining_episodes_by_cell;
    remaining_episodes_by_cell.reserve(static_cast<std::size_t>(cells));
    for (int64_t cell = 0; cell < cells; ++cell) {
        const auto offset = static_cast<std::size_t>(cell);
        TORCH_CHECK(
            base_before[offset] >= 0
                && cursor_before[offset] >= base_before[offset]
                && cursor_before[offset] - base_before[offset] <= capacity,
            "policy cursor lies outside the resident physical tape"
        );
        const int64_t remaining = capacity
            - (cursor_before[offset] - base_before[offset]);
        minimum_remaining_episodes = std::min(
            minimum_remaining_episodes, remaining
        );
        remaining_episodes_by_cell.push_back(remaining);
        if (remaining < required_future_episodes) {
            physical_refill_required = true;
            ++refill_trigger_cell_count;
        }
    }

    if (!physical_refill_required) {
        pybind11::dict out;
        out["schema_version"] =
            "eiel_cuda_architecture_b_native_resident_window_refill_v1";
        out["operation"] = "reuse_fixed_mlp_policy_window_v1";
        if constexpr (kOptionDM0TelemetryEnabledV1) {
            out["option_d_m0_telemetry_enabled"] = true;
        }
        out["option_d_m7_batched_cpu_policy_refill_enabled"] = true;
        out["option_d_m7_batched_cpu_policy_refill_used"] = false;
        out["option_d_m7_batched_cpu_policy_refill_authority"] =
            "compiled_candidate_not_used_for_surface";
        out["option_d_m7_batched_cpu_policy_refill_schema_version"] =
            pybind11::none();
        out["option_d_m8_resident_policy_suffix_reuse_enabled"] = true;
        out["option_d_m8_resident_policy_suffix_reuse_used"] = true;
        out["option_d_m8_resident_policy_suffix_reuse_authority"] =
            "candidate_resident_policy_suffix_reuse";
        out["option_d_m8_resident_policy_suffix_reuse_schema_version"] =
            "eiel_option_d_m8_resident_policy_suffix_reuse_v1";
        out["batch_cells"] = cells;
        out["physical_capacity"] = capacity;
        out["policy_row_width"] = policy_row_width;
        out["required_future_episodes"] = required_future_episodes;
        out["minimum_remaining_episodes"] = minimum_remaining_episodes;
        out["remaining_episodes_by_cell"] =
            int64_values_to_list_v1(remaining_episodes_by_cell);
        out["refill_trigger_cell_count"] = refill_trigger_cell_count;
        out["physical_refill_performed"] = false;
        out["resident_window_reused"] = true;
        out["absolute_cursor_before"] = int64_values_to_list_v1(cursor_before);
        out["window_base_before"] = int64_values_to_list_v1(base_before);
        out["window_base_after"] = int64_values_to_list_v1(base_before);
        out["absolute_cursor_preserved"] = true;
        out["semantic_draws_preserved"] = true;
        out["source_rng_state_preserved"] = true;
        out["host_wall_time_ns"] = steady_ns_v1() - ensure_start_ns;
        out["device_to_host_time_ns"] = coverage_check_time_ns;
        out["host_to_device_time_ns"] = static_cast<int64_t>(0);
        out["torch_random_generation_time_ns"] = static_cast<int64_t>(0);
        out["torch_compare_pack_time_ns"] = static_cast<int64_t>(0);
        out["torch_randn_generation_time_ns"] = static_cast<int64_t>(0);
        out["policy_allocation_stack_copy_time_ns"] = static_cast<int64_t>(0);
        out["device_to_host_bytes_lower_bound"] =
            static_cast<int64_t>(
                absolute_cursor.numel() * absolute_cursor.element_size()
                + window_base.numel() * window_base.element_size()
            );
        out["host_to_device_bytes_lower_bound"] = static_cast<int64_t>(0);
        if constexpr (kOptionDM0TelemetryEnabledV1) {
            out["torch_compare_time_ns"] = static_cast<int64_t>(0);
            out["crossover_pack_time_ns"] = static_cast<int64_t>(0);
            out["device_to_host_operations_lower_bound"] = static_cast<int64_t>(2);
            out["host_to_device_operations_lower_bound"] = static_cast<int64_t>(0);
        }
        return out;
    }

    pybind11::dict out = refill_fixed_mlp_policy_window_v1(
        source_state,
        std::move(delta),
        std::move(mutation_mask),
        std::move(crossover_mask_packed),
        std::move(post_state_packed),
        std::move(semantic_tape),
        absolute_cursor,
        semantic_counter,
        std::move(window_base),
        policy_row_width,
        mutation_enable,
        mutation_mode,
        mutation_rate,
        mutation_std,
        execution_route_sha256
    );
    out["option_d_m8_resident_policy_suffix_reuse_enabled"] = true;
    out["option_d_m8_resident_policy_suffix_reuse_used"] = false;
    out["option_d_m8_resident_policy_suffix_reuse_authority"] =
        "compiled_candidate_physical_refill_fallback";
    out["option_d_m8_resident_policy_suffix_reuse_schema_version"] =
        "eiel_option_d_m8_resident_policy_suffix_reuse_v1";
    out["required_future_episodes"] = required_future_episodes;
    out["minimum_remaining_episodes"] = minimum_remaining_episodes;
    out["remaining_episodes_by_cell"] =
        int64_values_to_list_v1(remaining_episodes_by_cell);
    out["refill_trigger_cell_count"] = refill_trigger_cell_count;
    out["physical_refill_performed"] = true;
    out["resident_window_reused"] = false;
    if constexpr (kOptionDM0TelemetryEnabledV1) {
        out["device_to_host_operations_lower_bound"] =
            pybind11::cast<int64_t>(out["device_to_host_operations_lower_bound"])
            + 2;
        out["device_to_host_time_ns"] =
            pybind11::cast<int64_t>(out["device_to_host_time_ns"])
            + coverage_check_time_ns;
    }
    out["device_to_host_bytes_lower_bound"] =
        pybind11::cast<int64_t>(out["device_to_host_bytes_lower_bound"])
        + static_cast<int64_t>(2 * cells * sizeof(int64_t));
    out["host_wall_time_ns"] = steady_ns_v1() - ensure_start_ns;
    return out;
}

inline pybind11::dict ensure_fixed_mlp_policy_window_v1(
    const torch::Tensor& source_state,
    torch::Tensor delta,
    torch::Tensor mutation_mask,
    torch::Tensor crossover_mask_packed,
    torch::Tensor post_state_packed,
    torch::Tensor semantic_tape,
    const torch::Tensor& absolute_cursor,
    const torch::Tensor& semantic_counter,
    torch::Tensor window_base,
    const int64_t policy_row_width,
    const bool mutation_enable,
    const std::string& mutation_mode,
    const double mutation_rate,
    const double mutation_std,
    const int64_t required_future_episodes,
    const std::string& execution_route_sha256 = std::string()
) {
    return ensure_fixed_mlp_policy_window_with_identity_v1(
        source_state,
        std::move(delta),
        std::move(mutation_mask),
        std::move(crossover_mask_packed),
        std::move(post_state_packed),
        std::move(semantic_tape),
        absolute_cursor,
        semantic_counter,
        std::move(window_base),
        policy_row_width,
        mutation_enable,
        mutation_mode,
        mutation_rate,
        mutation_std,
        required_future_episodes,
        execution_route_sha256,
        nullptr
    );
}

}  // namespace eiel::cuda_architecture_b::resident_window_refill_v1
