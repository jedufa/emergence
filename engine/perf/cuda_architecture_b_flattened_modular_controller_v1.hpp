#ifndef EIEL_CUDA_ARCHITECTURE_B_FLATTENED_MODULAR_CONTROLLER_V1_HPP_
#define EIEL_CUDA_ARCHITECTURE_B_FLATTENED_MODULAR_CONTROLLER_V1_HPP_

#include <cstdint>

namespace eiel::cuda_architecture_b::flattened_modular_controller_v1 {

inline constexpr std::int32_t kAbiVersionV1 = 1;
inline constexpr std::int32_t kLifecycleAbiVersionV2 = 2;
inline constexpr std::int64_t kPlaneCountV1 = 57;
inline constexpr std::int64_t kModuleSlotsV1 = 16;
inline constexpr std::int64_t kConnectionSlotsV1 = 48;
inline constexpr std::int64_t kBindingsPerModuleV1 = 2;
inline constexpr std::int64_t kActionWidthV1 = 7;
inline constexpr std::int64_t kTurnoverMutationReceiptWidthV1 = 21;
inline constexpr std::int64_t kTurnoverMutationSequenceCapacityV1 = 3;
inline constexpr std::int64_t kOrdinaryBirthRowV1ReceiptWidthV3 = 64;

enum ModularPlaneIndexV1 : std::int32_t {
    kAgentOccupiedV1 = 0,
    kAgentIdV1,
    kAgentSlotTokenV1,
    kModuleCountV1,
    kConnectionCountV1,
    kModuleOccupiedV1,
    kModuleLocalIdIndexV1,
    kModuleInstanceIdIndexV1,
    kModuleHomologyIdIndexV1,
    kModuleTypeIndexV1,
    kModuleTypeCodeV1,
    kIdentityOriginCodeV1,
    kTopologyPositionV1,
    kModuleExecutionOrderV1,
    kDependencyTopologicalOrderV1,
    kOperatorCodeV1,
    kIntegratorParametersV1,
    kGatedParametersV1,
    kModuleDormantV1,
    kModuleStatePresentV1,
    kModuleStateV1,
    kModuleOutputPresentV1,
    kModulePreviousOutputV1,
    kInputBindingCountV1,
    kInputSemanticCodeV1,
    kInputSemanticIdIndexV1,
    kInputGainV1,
    kInputEnabledV1,
    kOutputBindingCountV1,
    kOutputSemanticCodeV1,
    kOutputSemanticIdIndexV1,
    kOutputActionIndexV1,
    kOutputGainV1,
    kOutputEnabledV1,
    kDevelopmentExplicitV1,
    kDevelopmentStageGainsV1,
    kDevelopmentPrerequisiteCountV1,
    kDevelopmentPrerequisiteHomologyIndexV1,
    kDevelopmentPrerequisiteModuleSlotV1,
    kDevelopmentLatchEnabledV1,
    kDevelopmentLatchPresentV1,
    kDevelopmentLatchStateV1,
    kDevelopmentEffectiveGainPresentV1,
    kDevelopmentEffectiveGainV1,
    kConnectionOccupiedV1,
    kConnectionLocalIdIndexV1,
    kConnectionSrcModuleSlotV1,
    kConnectionDstModuleSlotV1,
    kConnectionSrcSlotIndexV1,
    kConnectionDstSlotIndexV1,
    kConnectionValueTypeIndexV1,
    kConnectionChannelIndexV1,
    kConnectionWeightV1,
    kConnectionExecutionOrderV1,
    kStructuralCostCountsV1,
    kStructuralCostComponentsV1,
    kStructuralCostTotalV1,
};

struct ModularTensorTripletV2 {
    std::uint8_t* source_address;
    std::uint8_t* shadow_address;
    std::int64_t bytes_per_cell;
    std::int64_t bytes_per_row;
    std::int64_t fill_byte;
};

struct FlatModularTensorTableV2 {
    const ModularTensorTripletV2* tensors;
};

static_assert(sizeof(ModularTensorTripletV2) == 5 * sizeof(std::int64_t));

// The first production slice consumes the already-committed 57-plane ABI but
// retains only pointers needed by the tick equations. All tensors are
// cell-major and remain CUDA-resident for the native trajectory.
struct FlatModularContextV1 {
    bool enabled;
    bool lifecycle_enabled;
    FlatModularTensorTableV2 lifecycle_tensors;
    std::int64_t* cold_origin_slot_source;
    std::int64_t* cold_origin_slot_shadow;
    bool* agent_occupied;
    std::int64_t* agent_id;
    std::int64_t* agent_slot_token;
    bool* module_occupied;
    std::int32_t* module_homology_id_index;
    std::int8_t* module_type_code;
    std::int32_t* module_execution_order;
    std::int32_t* dependency_topological_order;
    std::int8_t* operator_code;
    float* integrator_parameters;
    float* gated_parameters;
    bool* module_dormant;
    bool* module_state_present;
    float* module_state;
    bool* module_output_present;
    float* module_previous_output;
    std::int8_t* input_binding_count;
    std::int8_t* input_semantic_code;
    float* input_gain;
    bool* input_enabled;
    std::int8_t* output_binding_count;
    std::int8_t* output_action_index;
    float* output_gain;
    bool* output_enabled;
    float* development_stage_gains;
    std::int8_t* development_prerequisite_count;
    std::int32_t* development_prerequisite_homology_index;
    std::int32_t* development_prerequisite_module_slot;
    bool* development_latch_enabled;
    bool* development_latch_present;
    bool* development_latch_state;
    bool* development_effective_gain_present;
    float* development_effective_gain;
    bool* connection_occupied;
    std::int32_t* connection_src_module_slot;
    std::int32_t* connection_dst_module_slot;
    float* connection_weight;
    std::int32_t* connection_execution_order;
    float* action_residual_trace;
    std::int64_t action_residual_trace_ticks;
    bool turnover_mutation_enabled;
    bool ordinary_birth_mutation_enabled;
    const double* turnover_mutation_draws;
    bool* duplicate_descendant_mask_source;
    bool* duplicate_descendant_mask_shadow;
    std::int64_t* turnover_mutation_receipt_rows;
    std::int64_t* turnover_mutation_receipt_counts;
    std::int32_t* turnover_mutation_receipt_overflow;
    std::int64_t* ordinary_birth_rowv1_receipt_rows;
    std::int64_t* ordinary_birth_rowv1_receipt_counts;
    std::int32_t* ordinary_birth_rowv1_receipt_overflow;
    std::int64_t turnover_mutation_first_generation;
    std::int64_t cells;
    std::int64_t capacity;
    std::int32_t mature_start_step;
    std::int32_t late_start_step;
};

void install_flat_modular_context_v1(const FlatModularContextV1& context);
void clear_flat_modular_context_v1() noexcept;
const FlatModularContextV1& current_flat_modular_context_v1() noexcept;

}  // namespace eiel::cuda_architecture_b::flattened_modular_controller_v1

#endif
