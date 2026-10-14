#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace eiel::recurrent::modular_v1 {

constexpr std::size_t kPhysicalModuleSlots = 16U;
constexpr std::size_t kPhysicalConnectionSlots = 48U;
constexpr std::size_t kBindingsPerModule = 2U;
constexpr std::size_t kActionWidth = 7U;
constexpr std::uint16_t kLegacyLogicalModules = 12U;
constexpr std::uint16_t kLegacyLogicalConnections = 32U;
constexpr std::uint16_t kExpandedLogicalModules = 16U;
constexpr std::uint16_t kExpandedLogicalConnections = 48U;

enum class Status : std::uint8_t {
    kOk = 0,
    kInvalidArgument,
    kInvalidProfile,
    kCapacityExceeded,
    kDuplicateIdentity,
    kDanglingConnection,
    kInvalidSemantic,
    kInvalidNumeric,
    kInvalidMask,
    kNoEligibleModule,
    kDependencyBlocked,
    kUnsafeBoundary,
    kUnauthorized,
    kRollbackUnverified,
    kRngChanged,
    kOldSubspaceMismatch,
};

enum class CapacityProfile : std::uint8_t {
    kLegacy12x32 = 0,
    kExpanded16x48 = 1,
};

enum class Operator : std::uint8_t {
    kNone = 0,
    kIntegrator = 1,
    kGatedScalar = 2,
};

enum class InputSemantic : std::uint8_t {
    kSelfEnergy = 0,
    kSelfHealth = 1,
    kInvalid = 255,
};

enum class OutputSemantic : std::uint8_t {
    kMoveNorth = 0,
    kMoveSouth = 1,
    kMoveEast = 2,
    kMoveWest = 3,
    kRest = 4,
    kInvalid = 255,
};

enum class MutationOperation : std::uint8_t {
    kToggleDormancy = 0,
    kPruneEligible = 1,
    kRefineParameter = 2,
    kRefineRoutingOrExpression = 3,
};

struct CapacityLimits {
    std::uint16_t module_slots;
    std::uint16_t connection_slots;
};

struct IntegratorParameters {
    float expression_scale;
    float retention;
    float input_gain;
    float output_gain;
};

struct GatedParameters {
    float reset_bias;
    float reset_input_gain;
    float reset_state_gain;
    float update_bias;
    float update_input_gain;
    float update_state_gain;
    float candidate_bias;
    float candidate_input_gain;
    float candidate_state_gain;
    float output_gain;
};

struct InputBinding {
    InputSemantic semantic;
    float gain;
    bool enabled;
};

struct OutputBinding {
    OutputSemantic semantic;
    float gain;
    bool enabled;
};

struct SpecializationMetadata {
    std::uint64_t source_module_instance_id;
    std::uint32_t changed_dimensions;
    bool duplicate_descendant;
};

struct ModuleRecord {
    std::uint64_t module_instance_id;
    std::uint64_t homology_id;
    std::uint64_t parent_module_instance_id;
    std::uint64_t topology_key;
    std::uint16_t local_id;
    Operator op;
    IntegratorParameters integrator;
    GatedParameters gated;
    std::array<InputBinding, kBindingsPerModule> input_bindings;
    std::array<OutputBinding, kBindingsPerModule> output_bindings;
    std::array<std::uint64_t, kBindingsPerModule> prerequisite_homology_ids;
    std::uint8_t input_binding_count;
    std::uint8_t output_binding_count;
    std::uint8_t prerequisite_count;
    SpecializationMetadata specialization;
};

struct ConnectionRecord {
    std::uint16_t local_id;
    std::uint16_t source_module_slot;
    std::uint16_t target_module_slot;
    float weight;
};

struct FixedEnvelope {
    CapacityProfile profile;
    std::uint64_t capacity_epoch_id;
    std::uint64_t capacity_epoch_generation;
    std::array<ModuleRecord, kPhysicalModuleSlots> modules;
    std::array<ConnectionRecord, kPhysicalConnectionSlots> connections;
    std::array<bool, kPhysicalModuleSlots> module_occupied;
    std::array<bool, kPhysicalModuleSlots> module_dormant;
    std::array<bool, kPhysicalModuleSlots> module_state_present;
    std::array<bool, kPhysicalModuleSlots> module_output_present;
    std::array<std::uint16_t, kPhysicalModuleSlots> module_execution_order;
    std::array<bool, kPhysicalModuleSlots> development_explicit_present;
    std::array<bool, kPhysicalModuleSlots> development_latch;
    std::array<bool, kPhysicalModuleSlots> development_effective_gain_present;
    std::array<float, kPhysicalModuleSlots> development_effective_gain;
    std::array<float, kPhysicalModuleSlots> module_state;
    std::array<float, kPhysicalModuleSlots> module_previous_output;
    std::array<bool, kPhysicalConnectionSlots> connection_occupied;
    std::uint16_t module_count;
    std::uint16_t connection_count;
};

struct SemanticInputs {
    float self_energy;
    float self_health;
};

struct GatedStep {
    float input_aggregate;
    float reset_pre;
    float reset_gate;
    float update_pre;
    float update_gate;
    float candidate_pre;
    float candidate;
    float state_after;
    float base_output;
    float expressed_output;
    float effective_output;
};

struct ModuleStep {
    float routed_input;
    float state_before;
    float state_after;
    float output;
    GatedStep gated;
};

struct MutationResult {
    Status status;
    MutationOperation operation;
    std::uint16_t selected_slot;
    std::uint32_t changed_dimension;
    std::uint64_t selected_module_instance_id;
    std::uint64_t selected_homology_id;
    float value_before;
    float value_after;
    bool changed;
};

struct CostCounts {
    std::uint64_t allocated_modules;
    std::uint64_t executing_integrators;
    std::uint64_t executing_gated;
    std::uint64_t evaluated_connections;
    std::uint64_t active_input_bindings;
    std::uint64_t active_output_bindings;
    std::uint64_t active_dependency_edges;
    std::uint64_t dormant_modules;
};

struct CostSummary {
    CostCounts counts;
    std::array<std::uint64_t, 8U> components;
    std::uint64_t total;
    bool reporting_only;
};

enum class BoundaryKind : std::uint8_t {
    kUnknown = 0,
    kCompletedGenerationBeforeNextGenerationRngDraw = 1,
};

struct CapacityEpochRequest {
    bool authorized;
    bool rollback_checkpoint_verified;
    BoundaryKind boundary;
    CapacityProfile source_profile;
    CapacityProfile target_profile;
    std::uint64_t authorized_generation;
    std::uint64_t next_generation;
    std::uint64_t rollback_checkpoint_digest;
    std::uint64_t rng_digest_before;
    std::uint64_t rng_digest_after;
    std::uint64_t old_subspace_digest_before;
    std::uint64_t old_subspace_digest_after;
    std::uint64_t latest_observed_generation;
    std::uint32_t cap_contacts_in_window;
    std::uint32_t newly_blocked_legal_mutations_in_window;
    std::uint32_t near_cap_lineage_count;
    std::array<float, 3U> active_capacity_fraction;
    std::array<bool, 3U> low_bloat;
    std::array<bool, 3U> compression_relief_failed;
    std::uint8_t pressure_window_samples;
};

struct CapacityEpochReceipt {
    Status status;
    CapacityProfile source_profile;
    CapacityProfile target_profile;
    std::uint16_t newly_legal_module_begin;
    std::uint16_t newly_legal_connection_begin;
    std::uint32_t migration_rng_draws;
    std::uint32_t physical_resize_count;
    bool tail_canonical_empty;
    std::uint64_t source_capacity_epoch_id;
    std::uint64_t target_capacity_epoch_id;
    std::uint64_t capacity_epoch_generation;
};

[[nodiscard]] CapacityLimits limits_for(CapacityProfile profile) noexcept;
void initialize_empty(FixedEnvelope* envelope, CapacityProfile profile) noexcept;
[[nodiscard]] Status validate(const FixedEnvelope& envelope) noexcept;
[[nodiscard]] Status canonicalize_checkpoint(FixedEnvelope* envelope) noexcept;
[[nodiscard]] std::uint64_t canonical_digest(const FixedEnvelope& envelope) noexcept;
[[nodiscard]] Status project_checkpoint(
    const FixedEnvelope& owner,
    FixedEnvelope* checkpoint_out) noexcept;
[[nodiscard]] Status restore_checkpoint(
    FixedEnvelope* fresh_owner,
    const FixedEnvelope& checkpoint) noexcept;

[[nodiscard]] Status activate_module(
    FixedEnvelope* envelope,
    const ModuleRecord& module,
    bool dormant,
    std::uint16_t* slot_out) noexcept;
[[nodiscard]] Status neutral_duplicate(
    FixedEnvelope* envelope,
    std::uint16_t source_slot,
    std::uint64_t new_instance_id,
    std::uint64_t new_homology_id,
    std::uint64_t new_topology_key,
    std::uint16_t* slot_out) noexcept;
[[nodiscard]] Status set_dormant(FixedEnvelope* envelope, std::uint16_t slot, bool dormant) noexcept;
[[nodiscard]] Status set_module_development_state(
    FixedEnvelope* envelope,
    std::uint16_t slot,
    bool explicit_present,
    bool latch,
    bool effective_gain_present,
    float effective_gain) noexcept;
[[nodiscard]] Status prune_module(FixedEnvelope* envelope, std::uint16_t slot) noexcept;

[[nodiscard]] MutationResult mutate_two_draws(
    FixedEnvelope* envelope,
    float draw0,
    float draw1) noexcept;
[[nodiscard]] std::uint8_t select_inheritance_donor(float externally_supplied_draw) noexcept;
[[nodiscard]] Status inherit_module_atomically(
    FixedEnvelope* child,
    const FixedEnvelope& parent_a,
    const FixedEnvelope& parent_b,
    std::uint16_t parent_slot,
    float externally_supplied_draw,
    std::uint16_t* child_slot_out) noexcept;

[[nodiscard]] Status route_input(
    const ModuleRecord& module,
    const SemanticInputs& inputs,
    float* routed_input) noexcept;
[[nodiscard]] Status step_module(
    FixedEnvelope* envelope,
    std::uint16_t slot,
    float external_input,
    float routed_previous,
    ModuleStep* result) noexcept;
[[nodiscard]] Status route_outputs(
    const FixedEnvelope& envelope,
    float legacy_rest_residual,
    std::array<float, kActionWidth>* residuals) noexcept;

[[nodiscard]] CostSummary structural_cost(const FixedEnvelope& envelope) noexcept;
[[nodiscard]] CapacityEpochReceipt expand_capacity_epoch(
    FixedEnvelope* envelope,
    const CapacityEpochRequest& request) noexcept;

}  // namespace eiel::recurrent::modular_v1
