#ifndef EIEL_MODULAR_RECURRENT_CPU_ORACLE_C_API_V1_H
#define EIEL_MODULAR_RECURRENT_CPU_ORACLE_C_API_V1_H

#include <stdint.h>

#if defined(__GNUC__) || defined(__clang__)
#define EIEL_MODULAR_ORACLE_EXPORT __attribute__((visibility("default")))
#else
#define EIEL_MODULAR_ORACLE_EXPORT
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define EIEL_MODULAR_RECURRENT_CPU_ORACLE_ABI_VERSION_V1 1U
#define EIEL_MODULAR_RECURRENT_PLANE_COUNT_V1 57
#define EIEL_MODULAR_RECURRENT_MODULE_SLOTS_V1 16
#define EIEL_MODULAR_RECURRENT_CONNECTION_SLOTS_V1 48
#define EIEL_MODULAR_RECURRENT_BINDINGS_V1 2
#define EIEL_MODULAR_RECURRENT_ACTION_WIDTH_V1 7
#define EIEL_MODULAR_RECURRENT_OBSERVATION_WIDTH_V1 27
#define EIEL_MODULAR_RECURRENT_COST_FIELDS_V1 10

enum eiel_modular_recurrent_status_v1 {
    EIEL_MODULAR_RECURRENT_SUCCESS_V1 = 0,
    EIEL_MODULAR_RECURRENT_NULL_POINTER_V1 = 1,
    EIEL_MODULAR_RECURRENT_ABI_MISMATCH_V1 = 2,
    EIEL_MODULAR_RECURRENT_EXTENT_MISMATCH_V1 = 3,
    EIEL_MODULAR_RECURRENT_INVALID_PROFILE_V1 = 4,
    EIEL_MODULAR_RECURRENT_OWNER_MISMATCH_V1 = 5,
    EIEL_MODULAR_RECURRENT_INVALID_PLANE_V1 = 6,
    EIEL_MODULAR_RECURRENT_NONFINITE_V1 = 7,
    EIEL_MODULAR_RECURRENT_INVALID_ORDER_V1 = 8,
    EIEL_MODULAR_RECURRENT_CAPACITY_EXCEEDED_V1 = 9,
    EIEL_MODULAR_RECURRENT_INVALID_MUTATION_V1 = 10,
    EIEL_MODULAR_RECURRENT_INVALID_CHECKPOINT_V1 = 11,
    EIEL_MODULAR_RECURRENT_UNSUPPORTED_V1 = 12
};

enum eiel_modular_recurrent_profile_v1 {
    EIEL_MODULAR_RECURRENT_PROFILE_12X32_V1 = 0,
    EIEL_MODULAR_RECURRENT_PROFILE_16X48_V1 = 1
};

enum eiel_modular_recurrent_mutation_field_v1 {
    EIEL_MODULAR_MUTATE_INTEGRATOR_PARAMETER_V1 = 0,
    EIEL_MODULAR_MUTATE_GATED_PARAMETER_V1 = 1,
    EIEL_MODULAR_MUTATE_INPUT_GAIN_V1 = 2,
    EIEL_MODULAR_MUTATE_OUTPUT_GAIN_V1 = 3,
    EIEL_MODULAR_MUTATE_DEVELOPMENT_GAIN_V1 = 4,
    EIEL_MODULAR_MUTATE_CONNECTION_WEIGHT_V1 = 5
};

typedef struct eiel_modular_recurrent_row_v1 {
    uint8_t agent_occupied;
    int64_t agent_id;
    int64_t agent_slot_token;
    int32_t module_count;
    int32_t connection_count;
    uint8_t module_occupied[16];
    int32_t module_local_id_index[16];
    int32_t module_instance_id_index[16];
    int32_t module_homology_id_index[16];
    int32_t module_type_index[16];
    int8_t module_type_code[16];
    int8_t identity_origin_code[16];
    float topology_position[32];
    int32_t module_execution_order[16];
    int32_t dependency_topological_order[16];
    int8_t operator_code[16];
    float integrator_parameters[64];
    float gated_parameters[160];
    uint8_t module_dormant[16];
    uint8_t module_state_present[16];
    float module_state[16];
    uint8_t module_output_present[16];
    float module_previous_output[16];
    int8_t input_binding_count[16];
    int8_t input_semantic_code[32];
    int32_t input_semantic_id_index[32];
    float input_gain[32];
    uint8_t input_enabled[32];
    int8_t output_binding_count[16];
    int8_t output_semantic_code[32];
    int32_t output_semantic_id_index[32];
    int8_t output_action_index[32];
    float output_gain[32];
    uint8_t output_enabled[32];
    uint8_t development_explicit[16];
    float development_stage_gains[48];
    int8_t development_prerequisite_count[16];
    int32_t development_prerequisite_homology_index[32];
    int32_t development_prerequisite_module_slot[32];
    uint8_t development_latch_enabled[16];
    uint8_t development_latch_present[16];
    uint8_t development_latch_state[16];
    uint8_t development_effective_gain_present[16];
    float development_effective_gain[16];
    uint8_t connection_occupied[48];
    int32_t connection_local_id_index[48];
    int32_t connection_src_module_slot[48];
    int32_t connection_dst_module_slot[48];
    int32_t connection_src_slot_index[48];
    int32_t connection_dst_slot_index[48];
    int32_t connection_value_type_index[48];
    int32_t connection_channel_index[48];
    float connection_weight[48];
    int32_t connection_execution_order[48];
    int64_t structural_cost_counts[10];
    double structural_cost_components[10];
    double structural_cost_total;
} eiel_modular_recurrent_row_v1;

typedef struct eiel_modular_recurrent_owner_view_v1 {
    int64_t* slot_to_agent_id;
    int64_t* slot_tokens;
    uint8_t* alive_mask;
    int64_t row_count;
} eiel_modular_recurrent_owner_view_v1;

typedef struct eiel_modular_recurrent_const_owner_view_v1 {
    const int64_t* slot_to_agent_id;
    const int64_t* slot_tokens;
    const uint8_t* alive_mask;
    int64_t row_count;
} eiel_modular_recurrent_const_owner_view_v1;

typedef struct eiel_modular_recurrent_mutation_v1 {
    uint32_t field;
    int32_t outer_index;
    int32_t inner_index;
    float additive_delta;
} eiel_modular_recurrent_mutation_v1;

typedef struct eiel_modular_recurrent_canonical_receipt_binding_v1 {
    uint32_t binding_version;
    uint32_t canonical_receipt_contract_version;
    uint64_t canonical_receipt_sha256_words[4];
    uint64_t canonical_receipt_bytes_sha256_words[4];
    uint64_t draw0_float64_bits;
    uint64_t draw1_float64_bits;
    int64_t transaction_ordinal;
    int64_t parent_agent_id;
    int64_t parent_slot;
    int64_t parent_slot_token;
    int64_t child_agent_id;
    int64_t child_slot;
    int64_t child_slot_token;
    int32_t selected_instance_id_index;
    int32_t selected_homology_id_index;
    int32_t operation_code;
    int32_t status_code;
    int32_t reason_code;
    uint8_t changed;
    uint8_t reserved[7];
    uint64_t structural_payload_sha256_words[4];
    int32_t changed_dimension_code;
    uint8_t native_receipt_construction_required;
    uint8_t reserved_tail[3];
} eiel_modular_recurrent_canonical_receipt_binding_v1;

typedef struct eiel_modular_recurrent_canonical_receipt_source_v1 {
    uint32_t source_version;
    uint32_t reserved0;
    uint64_t source_packed_sha256_words[4];
    uint64_t ancestry_tape_sha256_words[4];
} eiel_modular_recurrent_canonical_receipt_source_v1;

typedef struct eiel_modular_recurrent_tick_receipt_v1 {
    int64_t committed_rows;
    int64_t structured_agents;
    int64_t neutral_agents;
    int64_t executing_modules;
    int64_t changing_state_modules;
    int64_t rng_draws;
    uint8_t production_routing_qualified;
    uint8_t publication_qualified;
} eiel_modular_recurrent_tick_receipt_v1;

typedef struct eiel_modular_recurrent_cost_capacity_receipt_v1 {
    int32_t logical_module_slots;
    int32_t logical_connection_slots;
    int32_t occupied_modules;
    int32_t occupied_connections;
    int64_t counts[10];
    double components[10];
    double total;
    uint8_t reporting_only;
    uint8_t at_module_capacity;
    uint8_t at_connection_capacity;
} eiel_modular_recurrent_cost_capacity_receipt_v1;

typedef struct eiel_modular_recurrent_birth_receipt_v1 {
    uint32_t receipt_version;
    int32_t lifecycle_status;
    uint8_t committed;
    uint8_t owner_checked;
    uint8_t parent_preserved;
    uint8_t child_runtime_reset;
    uint8_t canonical_receipt_bound;
    uint8_t canonical_receipt_identity_checked;
    uint8_t recurrent_receipt_bytes_bound;
    uint8_t canonical_receipt_reserved;
    int64_t destination_row;
    int64_t parent_a_row;
    int64_t parent_b_row;
    int64_t selected_parent_row;
    int64_t expected_destination_token;
    int64_t expected_parent_a_token;
    int64_t expected_parent_b_token;
    int64_t parent_a_agent_id;
    int64_t parent_b_agent_id;
    int64_t child_agent_id;
    int64_t publication_token;
    uint32_t donor_draw_float32_bits;
    int64_t mutation_count;
    uint64_t draw0_float64_bits;
    uint64_t draw1_float64_bits;
    uint64_t canonical_recurrent_receipt_sha256_words[4];
    uint64_t canonical_recurrent_receipt_bytes_sha256_words[4];
    int64_t transaction_ordinal;
    int32_t selected_instance_id_index;
    int32_t selected_homology_id_index;
    int32_t operation_code;
    int32_t status_code;
    int32_t reason_code;
    uint8_t changed;
    uint8_t canonical_receipt_reserved_tail[7];
    uint64_t structural_payload_sha256_words[4];
    uint8_t structural_payload_sha256_bound;
    uint8_t structural_payload_identity_checked;
    uint8_t structural_payload_reserved_tail[6];
} eiel_modular_recurrent_birth_receipt_v1;

EIEL_MODULAR_ORACLE_EXPORT uint32_t
eiel_modular_recurrent_cpu_oracle_abi_version_v1(void);
EIEL_MODULAR_ORACLE_EXPORT int32_t
eiel_modular_recurrent_cpu_oracle_plane_count_v1(void);
EIEL_MODULAR_ORACLE_EXPORT int64_t
eiel_modular_recurrent_cpu_oracle_row_size_v1(void);

EIEL_MODULAR_ORACLE_EXPORT int32_t eiel_modular_recurrent_validate_v1(
    const eiel_modular_recurrent_row_v1* rows,
    int64_t row_count,
    uint32_t profile,
    const eiel_modular_recurrent_const_owner_view_v1* owners);

EIEL_MODULAR_ORACLE_EXPORT int32_t
eiel_modular_recurrent_row_structural_payload_sha256_v1(
    const eiel_modular_recurrent_row_v1* row,
    uint64_t structural_payload_sha256_words[4]);

EIEL_MODULAR_ORACLE_EXPORT int32_t
eiel_modular_recurrent_canonical_receipt_sha256_v1(
    uint32_t profile,
    const eiel_modular_recurrent_canonical_receipt_source_v1* source,
    const eiel_modular_recurrent_canonical_receipt_binding_v1* binding,
    uint64_t canonical_receipt_sha256_words[4],
    uint64_t canonical_receipt_bytes_sha256_words[4]);

EIEL_MODULAR_ORACLE_EXPORT int32_t eiel_modular_recurrent_tick_v1(
    eiel_modular_recurrent_row_v1* rows,
    int64_t row_count,
    uint32_t profile,
    const eiel_modular_recurrent_const_owner_view_v1* owners,
    const float* observations,
    int64_t observation_element_count,
    const float* fatigue,
    int64_t fatigue_count,
    const int64_t* stage_indices,
    int64_t stage_count,
    eiel_modular_recurrent_row_v1* scratch_rows,
    int64_t scratch_row_count,
    float* scratch_action_residuals,
    int64_t scratch_action_count,
    float* action_residuals,
    int64_t action_count,
    eiel_modular_recurrent_tick_receipt_v1* receipt_out);

EIEL_MODULAR_ORACLE_EXPORT int32_t eiel_modular_recurrent_founder_init_v1(
    eiel_modular_recurrent_row_v1* rows,
    int64_t row_count,
    uint32_t profile,
    eiel_modular_recurrent_owner_view_v1* owners,
    int64_t destination_row,
    int64_t expected_destination_token,
    int64_t founder_agent_id,
    int64_t publication_token,
    const eiel_modular_recurrent_row_v1* founder_structure);

EIEL_MODULAR_ORACLE_EXPORT int32_t eiel_modular_recurrent_newborn_inherit_v1(
    eiel_modular_recurrent_row_v1* rows,
    int64_t row_count,
    uint32_t profile,
    eiel_modular_recurrent_owner_view_v1* owners,
    int64_t destination_row,
    int64_t expected_destination_token,
    int64_t parent_a_row,
    int64_t expected_parent_a_token,
    int64_t parent_b_row,
    int64_t expected_parent_b_token,
    float donor_draw,
    int64_t child_agent_id,
    int64_t publication_token,
    const eiel_modular_recurrent_mutation_v1* mutations,
    int64_t mutation_count);

EIEL_MODULAR_ORACLE_EXPORT int32_t
eiel_modular_recurrent_newborn_inherit_with_receipt_v1(
    eiel_modular_recurrent_row_v1* rows,
    int64_t row_count,
    uint32_t profile,
    eiel_modular_recurrent_owner_view_v1* owners,
    int64_t destination_row,
    int64_t expected_destination_token,
    int64_t parent_a_row,
    int64_t expected_parent_a_token,
    int64_t parent_b_row,
    int64_t expected_parent_b_token,
    float donor_draw,
    int64_t child_agent_id,
    int64_t publication_token,
    const eiel_modular_recurrent_mutation_v1* mutations,
    int64_t mutation_count,
    eiel_modular_recurrent_birth_receipt_v1* receipt_out);

EIEL_MODULAR_ORACLE_EXPORT int32_t
eiel_modular_recurrent_newborn_inherit_with_canonical_receipt_v1(
    eiel_modular_recurrent_row_v1* rows,
    int64_t row_count,
    uint32_t profile,
    eiel_modular_recurrent_owner_view_v1* owners,
    int64_t destination_row,
    int64_t expected_destination_token,
    int64_t parent_a_row,
    int64_t expected_parent_a_token,
    int64_t parent_b_row,
    int64_t expected_parent_b_token,
    float donor_draw,
    int64_t child_agent_id,
    int64_t publication_token,
    const eiel_modular_recurrent_mutation_v1* mutations,
    int64_t mutation_count,
    const eiel_modular_recurrent_canonical_receipt_source_v1* canonical_source,
    const eiel_modular_recurrent_canonical_receipt_binding_v1* canonical_binding,
    eiel_modular_recurrent_birth_receipt_v1* receipt_out);

EIEL_MODULAR_ORACLE_EXPORT int32_t eiel_modular_recurrent_death_cleanup_v1(
    eiel_modular_recurrent_row_v1* rows,
    int64_t row_count,
    uint32_t profile,
    eiel_modular_recurrent_owner_view_v1* owners,
    int64_t row_index,
    int64_t expected_agent_id,
    int64_t expected_token);

EIEL_MODULAR_ORACLE_EXPORT int32_t eiel_modular_recurrent_same_slot_reuse_v1(
    eiel_modular_recurrent_row_v1* rows,
    int64_t row_count,
    uint32_t profile,
    eiel_modular_recurrent_owner_view_v1* owners,
    int64_t row_index,
    int64_t expected_agent_id,
    int64_t expected_token,
    int64_t structural_parent_row,
    int64_t expected_parent_token,
    int64_t child_agent_id,
    int64_t publication_token,
    const eiel_modular_recurrent_mutation_v1* mutations,
    int64_t mutation_count);

EIEL_MODULAR_ORACLE_EXPORT int32_t
eiel_modular_recurrent_same_slot_reuse_with_receipt_v1(
    eiel_modular_recurrent_row_v1* rows,
    int64_t row_count,
    uint32_t profile,
    eiel_modular_recurrent_owner_view_v1* owners,
    int64_t row_index,
    int64_t expected_agent_id,
    int64_t expected_token,
    int64_t structural_parent_row,
    int64_t expected_parent_token,
    int64_t child_agent_id,
    int64_t publication_token,
    const eiel_modular_recurrent_mutation_v1* mutations,
    int64_t mutation_count,
    eiel_modular_recurrent_birth_receipt_v1* receipt_out);

EIEL_MODULAR_ORACLE_EXPORT int32_t
eiel_modular_recurrent_same_slot_reuse_with_canonical_receipt_v1(
    eiel_modular_recurrent_row_v1* rows,
    int64_t row_count,
    uint32_t profile,
    eiel_modular_recurrent_owner_view_v1* owners,
    int64_t row_index,
    int64_t expected_agent_id,
    int64_t expected_token,
    int64_t structural_parent_row,
    int64_t expected_parent_token,
    int64_t child_agent_id,
    int64_t publication_token,
    const eiel_modular_recurrent_mutation_v1* mutations,
    int64_t mutation_count,
    const eiel_modular_recurrent_canonical_receipt_source_v1* canonical_source,
    const eiel_modular_recurrent_canonical_receipt_binding_v1* canonical_binding,
    eiel_modular_recurrent_birth_receipt_v1* receipt_out);

EIEL_MODULAR_ORACLE_EXPORT int32_t eiel_modular_recurrent_checkpoint_project_v1(
    const eiel_modular_recurrent_row_v1* rows,
    int64_t row_count,
    uint32_t profile,
    const eiel_modular_recurrent_const_owner_view_v1* owners,
    int64_t row_index,
    int64_t expected_token,
    eiel_modular_recurrent_row_v1* checkpoint_out);

EIEL_MODULAR_ORACLE_EXPORT int32_t eiel_modular_recurrent_checkpoint_restore_v1(
    const eiel_modular_recurrent_row_v1* checkpoint,
    eiel_modular_recurrent_row_v1* rows,
    int64_t row_count,
    uint32_t profile,
    eiel_modular_recurrent_owner_view_v1* owners,
    int64_t destination_row,
    int64_t expected_destination_token);

EIEL_MODULAR_ORACLE_EXPORT int32_t eiel_modular_recurrent_cost_capacity_v1(
    const eiel_modular_recurrent_row_v1* row,
    uint32_t profile,
    eiel_modular_recurrent_cost_capacity_receipt_v1* receipt_out);

#ifdef __cplusplus
}
#endif

#endif
