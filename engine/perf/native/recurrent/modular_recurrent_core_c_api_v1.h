#ifndef EIEL_MODULAR_RECURRENT_CORE_C_API_V1_H
#define EIEL_MODULAR_RECURRENT_CORE_C_API_V1_H

#include <stdint.h>

#if defined(__GNUC__) || defined(__clang__)
#define EIEL_MODULAR_CORE_EXPORT __attribute__((visibility("default")))
#else
#define EIEL_MODULAR_CORE_EXPORT
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define EIEL_MODULAR_RECURRENT_CORE_ABI_VERSION_V1 1U
#define EIEL_MODULAR_RECURRENT_CORE_ORDINARY_BIRTH_RECEIPT_LAYOUT_VERSION_V2 2U
#define EIEL_MODULAR_RECURRENT_CORE_ORDINARY_BIRTH_RECEIPT_LAYOUT_VERSION_V4 4U

enum eiel_modular_recurrent_core_profile_v1 {
    EIEL_MODULAR_RECURRENT_CORE_PROFILE_12X32_V1 = 0,
    EIEL_MODULAR_RECURRENT_CORE_PROFILE_16X48_V1 = 1
};

enum eiel_modular_recurrent_core_birth_transaction_status_v1 {
    EIEL_MODULAR_RECURRENT_CORE_BIRTH_COMMITTED_V1 = 0,
    EIEL_MODULAR_RECURRENT_CORE_BIRTH_CAPACITY_UNAVAILABLE_V1 = 1,
    EIEL_MODULAR_RECURRENT_CORE_BIRTH_PUBLICATION_REJECTED_V1 = 2,
    EIEL_MODULAR_RECURRENT_CORE_BIRTH_OWNER_MISMATCH_V1 = 3,
    EIEL_MODULAR_RECURRENT_CORE_BIRTH_INVALID_DRAW_V1 = 4,
    EIEL_MODULAR_RECURRENT_CORE_BIRTH_NATIVE_MUTATION_REJECTED_V1 = 5,
    EIEL_MODULAR_RECURRENT_CORE_BIRTH_MUTATION_DISABLED_V1 = 6
};

enum eiel_modular_recurrent_core_disabled_reason_v1 {
    EIEL_MODULAR_RECURRENT_CORE_DISABLED_REASON_NONE_V1 = 0,
    EIEL_MODULAR_RECURRENT_CORE_DISABLED_REASON_OPERATOR_DISABLED_V1 = 1,
    EIEL_MODULAR_RECURRENT_CORE_DISABLED_REASON_STRUCTURAL_MUTATION_DISABLED_V1 = 2,
    EIEL_MODULAR_RECURRENT_CORE_DISABLED_REASON_RECURRENT_MUTATION_DISABLED_V1 = 3,
    EIEL_MODULAR_RECURRENT_CORE_DISABLED_REASON_OPERATOR_FAMILY_DISABLED_V1 = 4,
    EIEL_MODULAR_RECURRENT_CORE_DISABLED_REASON_ZERO_MUTATION_RATE_V1 = 5,
    EIEL_MODULAR_RECURRENT_CORE_DISABLED_REASON_MUTATION_ATTEMPT_BUDGET_ZERO_V1 = 6
};

typedef struct eiel_modular_recurrent_core_mutation_receipt_v1 {
    int32_t status;
    int32_t operation;
    int32_t selected_slot;
    uint32_t changed_dimension;
    uint8_t changed;
    uint8_t reserved[7];
    uint64_t digest_before;
    uint64_t digest_after;
} eiel_modular_recurrent_core_mutation_receipt_v1;

typedef struct eiel_modular_recurrent_core_ordinary_birth_receipt_v1 {
    int32_t transaction_status;
    int32_t native_status;
    int32_t operation;
    int32_t selected_slot;
    uint32_t changed_dimension;
    uint8_t changed;
    uint8_t committed;
    uint8_t capacity_available;
    uint8_t publish_success;
    uint8_t reserved[4];
    int64_t parent_agent_id;
    int64_t parent_slot;
    int64_t parent_slot_token;
    int64_t child_agent_id;
    int64_t child_slot;
    int64_t expected_child_slot_token;
    int64_t child_publication_token;
    uint64_t draw0_float64_bits;
    uint64_t draw1_float64_bits;
    uint64_t parent_digest_before;
    uint64_t child_digest_before;
    uint64_t proposal_digest;
    uint64_t child_digest_after;
    uint64_t selected_module_instance_id;
    uint64_t selected_homology_id;
    float value_before_float32;
    float value_after_float32;
} eiel_modular_recurrent_core_ordinary_birth_receipt_v1;

typedef eiel_modular_recurrent_core_ordinary_birth_receipt_v1
    eiel_modular_recurrent_core_ordinary_birth_receipt_v2;

typedef struct eiel_modular_recurrent_core_ordinary_birth_receipt_v4 {
    int32_t transaction_status;
    int32_t native_status;
    int32_t operation;
    int32_t selected_slot;
    uint32_t changed_dimension;
    uint8_t changed;
    uint8_t committed;
    uint8_t capacity_available;
    uint8_t publish_success;
    uint8_t reserved[4];
    int64_t parent_agent_id;
    int64_t parent_slot;
    int64_t parent_slot_token;
    int64_t child_agent_id;
    int64_t child_slot;
    int64_t expected_child_slot_token;
    int64_t child_publication_token;
    uint64_t draw0_float64_bits;
    uint64_t draw1_float64_bits;
    uint64_t parent_digest_before;
    uint64_t child_digest_before;
    uint64_t proposal_digest;
    uint64_t child_digest_after;
    uint64_t selected_module_instance_id;
    uint64_t selected_homology_id;
    float value_before_float32;
    float value_after_float32;
    int32_t disabled_reason_code;
    uint8_t reserved_v4[4];
} eiel_modular_recurrent_core_ordinary_birth_receipt_v4;

EIEL_MODULAR_CORE_EXPORT uint32_t
eiel_modular_recurrent_core_abi_version_v1(void);

EIEL_MODULAR_CORE_EXPORT int64_t
eiel_modular_recurrent_core_fixed_envelope_size_v1(void);

EIEL_MODULAR_CORE_EXPORT int64_t
eiel_modular_recurrent_core_fixed_envelope_alignment_v1(void);

EIEL_MODULAR_CORE_EXPORT int32_t eiel_modular_recurrent_core_initialize_empty_v1(
    void* envelope,
    int64_t envelope_bytes,
    uint32_t profile);

EIEL_MODULAR_CORE_EXPORT int32_t eiel_modular_recurrent_core_validate_v1(
    const void* envelope,
    int64_t envelope_bytes);

EIEL_MODULAR_CORE_EXPORT uint64_t eiel_modular_recurrent_core_canonical_digest_v1(
    const void* envelope,
    int64_t envelope_bytes);

EIEL_MODULAR_CORE_EXPORT int32_t eiel_modular_recurrent_core_module_count_v1(
    const void* envelope,
    int64_t envelope_bytes);

EIEL_MODULAR_CORE_EXPORT int32_t eiel_modular_recurrent_core_activate_integrator_v1(
    void* envelope,
    int64_t envelope_bytes,
    uint64_t module_instance_id,
    uint64_t homology_id,
    uint64_t topology_key,
    float expression_scale,
    float retention,
    float input_gain,
    float output_gain,
    uint8_t dormant,
    uint16_t* slot_out);

EIEL_MODULAR_CORE_EXPORT int32_t eiel_modular_recurrent_core_neutral_duplicate_v1(
    void* envelope,
    int64_t envelope_bytes,
    uint16_t source_slot,
    uint64_t new_instance_id,
    uint64_t new_homology_id,
    uint64_t new_topology_key,
    uint16_t* slot_out);

EIEL_MODULAR_CORE_EXPORT int32_t eiel_modular_recurrent_core_mutate_two_draws_v1(
    void* envelope,
    int64_t envelope_bytes,
    float draw0,
    float draw1,
    eiel_modular_recurrent_core_mutation_receipt_v1* receipt_out);

EIEL_MODULAR_CORE_EXPORT int32_t
eiel_modular_recurrent_core_ordinary_birth_two_draws_v1(
    const void* parent_envelope,
    int64_t parent_envelope_bytes,
    void* child_envelope,
    int64_t child_envelope_bytes,
    int64_t parent_agent_id,
    int64_t parent_slot,
    int64_t parent_slot_token,
    int64_t child_agent_id,
    int64_t child_slot,
    int64_t expected_child_slot_token,
    int64_t child_publication_token,
    uint64_t draw0_float64_bits,
    uint64_t draw1_float64_bits,
    uint8_t capacity_available,
    uint8_t publish_success,
    eiel_modular_recurrent_core_ordinary_birth_receipt_v1* receipt_out);

EIEL_MODULAR_CORE_EXPORT uint32_t
eiel_modular_recurrent_core_ordinary_birth_receipt_layout_version_v2(void);

EIEL_MODULAR_CORE_EXPORT int64_t
eiel_modular_recurrent_core_ordinary_birth_receipt_size_v2(void);

EIEL_MODULAR_CORE_EXPORT int64_t
eiel_modular_recurrent_core_ordinary_birth_receipt_alignment_v2(void);

EIEL_MODULAR_CORE_EXPORT int32_t
eiel_modular_recurrent_core_ordinary_birth_two_draws_v2(
    const void* parent_envelope,
    int64_t parent_envelope_bytes,
    void* child_envelope,
    int64_t child_envelope_bytes,
    int64_t parent_agent_id,
    int64_t parent_slot,
    int64_t parent_slot_token,
    int64_t child_agent_id,
    int64_t child_slot,
    int64_t expected_child_slot_token,
    int64_t child_publication_token,
    uint64_t draw0_float64_bits,
    uint64_t draw1_float64_bits,
    uint8_t capacity_available,
    uint8_t publish_success,
    eiel_modular_recurrent_core_ordinary_birth_receipt_v2* receipt_out);

EIEL_MODULAR_CORE_EXPORT int32_t
eiel_modular_recurrent_core_ordinary_birth_two_draws_v3(
    const void* parent_envelope,
    int64_t parent_envelope_bytes,
    void* child_envelope,
    int64_t child_envelope_bytes,
    int64_t parent_agent_id,
    int64_t parent_slot,
    int64_t parent_slot_token,
    int64_t child_agent_id,
    int64_t child_slot,
    int64_t expected_child_slot_token,
    int64_t child_publication_token,
    uint64_t draw0_float64_bits,
    uint64_t draw1_float64_bits,
    uint8_t mutation_enabled,
    uint8_t capacity_available,
    uint8_t publish_success,
    eiel_modular_recurrent_core_ordinary_birth_receipt_v2* receipt_out);

EIEL_MODULAR_CORE_EXPORT uint32_t
eiel_modular_recurrent_core_ordinary_birth_receipt_layout_version_v4(void);

EIEL_MODULAR_CORE_EXPORT int64_t
eiel_modular_recurrent_core_ordinary_birth_receipt_size_v4(void);

EIEL_MODULAR_CORE_EXPORT int64_t
eiel_modular_recurrent_core_ordinary_birth_receipt_alignment_v4(void);

EIEL_MODULAR_CORE_EXPORT int32_t
eiel_modular_recurrent_core_ordinary_birth_two_draws_v4(
    const void* parent_envelope,
    int64_t parent_envelope_bytes,
    void* child_envelope,
    int64_t child_envelope_bytes,
    int64_t parent_agent_id,
    int64_t parent_slot,
    int64_t parent_slot_token,
    int64_t child_agent_id,
    int64_t child_slot,
    int64_t expected_child_slot_token,
    int64_t child_publication_token,
    uint64_t draw0_float64_bits,
    uint64_t draw1_float64_bits,
    uint32_t disabled_reason_code,
    uint8_t capacity_available,
    uint8_t publish_success,
    eiel_modular_recurrent_core_ordinary_birth_receipt_v4* receipt_out);

#ifdef __cplusplus
}
#endif

#endif
