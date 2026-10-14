#ifndef EIEL_CUDA_ARCHITECTURE_B_HAPB_POLICY_CONTEXT_V1_HPP_
#define EIEL_CUDA_ARCHITECTURE_B_HAPB_POLICY_CONTEXT_V1_HPP_

#include <cstdint>
#include <functional>

#include <torch/types.h>

namespace eiel::cuda_architecture_b::hapb_policy_context_v1 {

inline constexpr std::int64_t kCellsV1 = 48;
inline constexpr std::int64_t kPhysicalCellsMinV1 = 1;
inline constexpr std::int64_t kPhysicalCellsMaxV1 = 256;
std::int64_t physical_cells_v1(const torch::Tensor& bank);
inline constexpr std::int64_t kCapacityV1 = 512;
inline constexpr std::int64_t kHotPrefixRowsV1 = 256;
inline constexpr std::int64_t kPolicyWidthV1 = 9991;
inline constexpr std::int64_t kPolicyRowBytesV1 = 39964;
inline constexpr std::int64_t kTrafficPolicyH2DRowsV1 = 0;
inline constexpr std::int64_t kTrafficPolicyH2DBytesV1 = 1;
inline constexpr std::int64_t kTrafficPolicyD2HRowsV1 = 2;
inline constexpr std::int64_t kTrafficPolicyD2HBytesV1 = 3;
inline constexpr std::int64_t kTrafficControlD2HBytesV1 = 4;
inline constexpr std::int64_t kTrafficControlH2DBytesV1 = 5;
inline constexpr std::int64_t kTrafficSynchronizationCountV1 = 6;
inline constexpr std::int64_t kTrafficGpuStallNsV1 = 7;
inline constexpr std::int64_t kTrafficCounterCountV1 = 8;

// This is transaction state only.  The canonical bank is CPU authority; the
// device prefix and tile cannot be serialized or reused outside the installed
// prepare/refill/resume interval.
struct HapbPolicyContextV1 {
    torch::Tensor canonical_host_policy_bank;
    torch::Tensor pinned_staging_tile;
    torch::Tensor hot_prefix_cache;
    torch::Tensor cuda_tile_arena;
    torch::Tensor stable_overflow_rows;
    torch::Tensor stable_overflow_active_rows;
    torch::Tensor stable_overflow_content_rows;
    torch::Tensor content_row_map;
    torch::Tensor jungle_pre_predator_policy_pcg64_state;
    torch::Tensor jungle_pre_predator_policy_pcg64_buffer;
    torch::Tensor policy_traffic_counters;
    bool content_alias_enabled = false;
};

// The direct WDL/Jungle/EED dispatcher owns this route context for the whole
// native call.  It deliberately excludes tick-local row maps and the Jungle
// cursor snapshot: each prepare/resume boundary materializes a separate
// HapbPolicyContextV1 from this stable owner plus freshly prepared state.
// Keeping the two scopes distinct prevents a prior tick's map/cursor from
// leaking into a checkpoint or being reused after the next classifier pass.
using HapbOrdinaryBirthTransactionV1 = std::function<void(
    const torch::Tensor&,
    const torch::Tensor&,
    const torch::Tensor&,
    const torch::Tensor&,
    const torch::Tensor&,
    const torch::Tensor&,
    const torch::Tensor&,
    const torch::Tensor&
)>;

// Invoked after the native turnover shadow/cull/refill plan is complete and
// before native identity/state publish. The canonical host bank remains the
// only policy authority; the CUDA route receives only bounded plan/tape views.
using HapbTurnoverTransactionV1 = std::function<void(
    const torch::Tensor&,
    const torch::Tensor&,
    const torch::Tensor&,
    const torch::Tensor&,
    const torch::Tensor&,
    const torch::Tensor&,
    const torch::Tensor&,
    const torch::Tensor&,
    const torch::Tensor&
)>;

struct HapbDirectRouteContextV1 {
    torch::Tensor canonical_host_policy_bank;
    torch::Tensor pinned_staging_tile;
    torch::Tensor hot_prefix_cache;
    torch::Tensor cuda_tile_arena;
    // CPU logical-slot to immutable hot-pool-content relationship.  It is a
    // reconstructible cache descriptor; the canonical host bank remains the
    // sole checkpoint authority.
    torch::Tensor content_row_map;
    torch::Tensor policy_traffic_counters;
    bool content_alias_enabled = true;
    // Invoked only after native ordinary-birth preparation has validated the
    // exact plan and before the resident birth commit makes a child eligible.
    // It receives CUDA views; the guarded host bridge owns all bounded D2H,
    // host-authoritative mutation, and H2D cache refresh work.
    HapbOrdinaryBirthTransactionV1 ordinary_birth_transaction;
    // Invoked after native turnover planning and before native turnover
    // commit. It commits exact crossover/mutation rows to host authority.
    HapbTurnoverTransactionV1 turnover_transaction;
};

// Tick-local prepared rows supplied by the resident controller after its
// classifier-only prepare launch.  This surface intentionally contains no
// resident MLP tensor: policy authority is obtained only through the active
// direct-route context.
struct HapbPreparedPolicyBoundaryV1 {
    torch::Tensor ordered_slots;
    torch::Tensor observations;
    torch::Tensor stable_miss_rows;
    torch::Tensor active_row_counts;
    torch::Tensor miss_row_counts;
    torch::Tensor status;
    torch::Tensor policy_service_bypass_by_cell;
    torch::Tensor drives;
    torch::Tensor attack_bias;
    torch::Tensor action_temperature;
    torch::Tensor drive_logit_gain;
    torch::Tensor slot_tokens;
    torch::Tensor action_cache_actions;
    torch::Tensor action_cache_token;
    torch::Tensor action_cache_cursor;
    torch::Tensor action_cache_next;
    torch::Tensor logits;
    torch::Tensor row_actions;
    torch::Tensor actions_by_slot;
    torch::Tensor policy_pcg64_state_words;
    torch::Tensor policy_pcg64_buffer_words;
    torch::Tensor policy_np_semantic_draws;
    float w_hunger_gather;
    float w_pain_rest;
    float w_fatigue_rest;
    float w_pain_attack;
    float w_fear_attack;
};

void validate_hapb_policy_context_v1(const HapbPolicyContextV1& context);
void install_hapb_policy_context_v1(const HapbPolicyContextV1& context);
void clear_hapb_policy_context_v1() noexcept;
const HapbPolicyContextV1& current_hapb_policy_context_v1();

void validate_hapb_direct_route_context_v1(
    const HapbDirectRouteContextV1& context
);
void install_hapb_direct_route_context_v1(
    const HapbDirectRouteContextV1& context
);
void clear_hapb_direct_route_context_v1() noexcept;
const HapbDirectRouteContextV1& current_hapb_direct_route_context_v1();
HapbPolicyContextV1 materialize_hapb_policy_context_v1(
    const torch::Tensor& stable_overflow_rows,
    const torch::Tensor& stable_overflow_active_rows,
    const torch::Tensor& jungle_pre_predator_policy_pcg64_state,
    const torch::Tensor& jungle_pre_predator_policy_pcg64_buffer,
    const torch::Tensor& stable_overflow_content_rows
);
bool hapb_direct_route_context_active_v1() noexcept;
torch::Tensor execute_hapb_prepared_policy_boundary_v1(
    const HapbPreparedPolicyBoundaryV1& boundary
);

class HapbPolicyContextGuardV1 {
public:
    explicit HapbPolicyContextGuardV1(const HapbPolicyContextV1& context);
    ~HapbPolicyContextGuardV1();
    HapbPolicyContextGuardV1(const HapbPolicyContextGuardV1&) = delete;
    HapbPolicyContextGuardV1& operator=(const HapbPolicyContextGuardV1&) = delete;

private:
    bool installed_ = false;
};

class HapbDirectRouteContextGuardV1 {
public:
    explicit HapbDirectRouteContextGuardV1(
        const HapbDirectRouteContextV1& context
    );
    ~HapbDirectRouteContextGuardV1();
    HapbDirectRouteContextGuardV1(const HapbDirectRouteContextGuardV1&) = delete;
    HapbDirectRouteContextGuardV1& operator=(
        const HapbDirectRouteContextGuardV1&
    ) = delete;

private:
    bool installed_ = false;
};

}  // namespace eiel::cuda_architecture_b::hapb_policy_context_v1

#endif
