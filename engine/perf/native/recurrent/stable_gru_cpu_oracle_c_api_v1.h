#pragma once

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#define EIEL_STABLE_GRU_EXPORT __declspec(dllexport)
#else
#define EIEL_STABLE_GRU_EXPORT __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define EIEL_STABLE_GRU_CPU_ORACLE_ABI_VERSION_V1 1
#define EIEL_STABLE_GRU_HIDDEN_WIDTH_V1 64
#define EIEL_STABLE_GRU_ACTION_WIDTH_V1 7

typedef struct {
    uint32_t contract_version;
    int64_t observation_size;
    int64_t active_memory_size;
    float aggression;
    float temperature;
    float drives[4];
    float gain;
    float hunger_gather_weight;
    float pain_rest_weight;
    float fatigue_rest_weight;
    float pain_attack_weight;
    float fear_attack_weight;
} eiel_stable_gru_step_config_v1;

typedef struct {
    uint32_t contract_version;
    int64_t cell_index;
    int64_t agent_index;
    int64_t hidden_size;
    uint64_t token;
    float hidden[EIEL_STABLE_GRU_HIDDEN_WIDTH_V1];
} eiel_stable_gru_checkpoint_row_v1;

EIEL_STABLE_GRU_EXPORT uint32_t eiel_stable_gru_cpu_oracle_abi_version_v1(void);

EIEL_STABLE_GRU_EXPORT int32_t eiel_stable_gru_packed_parameter_count_v1(
    int64_t observation_size,
    int64_t* parameter_count_out
);

EIEL_STABLE_GRU_EXPORT int32_t eiel_stable_gru_forward_v1(
    const eiel_stable_gru_step_config_v1* config,
    const float* observation,
    int64_t observation_count,
    const float* hidden_before,
    int64_t hidden_before_count,
    const float* packed_parameters,
    int64_t packed_parameter_count,
    float* hidden_after,
    int64_t hidden_after_count,
    float* logits,
    int64_t logits_count,
    int32_t* action_out
);

/*
 * Transactional multi-row adapter. Every row computes into caller-owned
 * scratch first; outputs remain byte-unchanged unless every row succeeds.
 * Scratch spans must not overlap each other or any public output span.
 */
EIEL_STABLE_GRU_EXPORT int32_t eiel_stable_gru_forward_batch_v1(
    const eiel_stable_gru_step_config_v1* configs,
    int64_t config_count,
    const float* observations,
    int64_t observation_element_count,
    const float* hidden_before,
    int64_t hidden_before_element_count,
    const float* packed_parameters,
    int64_t packed_parameter_element_count,
    int64_t row_count,
    float* scratch_hidden,
    int64_t scratch_hidden_element_count,
    float* scratch_logits,
    int64_t scratch_logits_element_count,
    int32_t* scratch_actions,
    int64_t scratch_action_count,
    float* hidden_after,
    int64_t hidden_after_element_count,
    float* logits,
    int64_t logits_element_count,
    int32_t* actions,
    int64_t action_count
);

EIEL_STABLE_GRU_EXPORT int32_t eiel_stable_gru_lifecycle_v1(
    float* hidden,
    int64_t hidden_element_count,
    uint64_t* tokens,
    int64_t token_count,
    int64_t row_count,
    int64_t row_index,
    uint64_t expected_token,
    uint64_t publication_token,
    uint32_t operation
);

EIEL_STABLE_GRU_EXPORT int32_t eiel_stable_gru_mutate_v1(
    const float* parent,
    int64_t parameter_count,
    const float* additive_draws,
    const uint8_t* mutation_mask,
    float scale,
    float* scratch,
    int64_t scratch_count,
    float* child,
    int64_t child_count
);

EIEL_STABLE_GRU_EXPORT int32_t eiel_stable_gru_checkpoint_project_v1(
    const float* hidden,
    int64_t hidden_element_count,
    const uint64_t* tokens,
    int64_t token_count,
    int64_t row_count,
    int64_t row_index,
    uint64_t expected_token,
    eiel_stable_gru_checkpoint_row_v1* checkpoint_out
);

EIEL_STABLE_GRU_EXPORT int32_t eiel_stable_gru_checkpoint_restore_v1(
    const eiel_stable_gru_checkpoint_row_v1* checkpoint,
    float* hidden,
    int64_t hidden_element_count,
    uint64_t* tokens,
    int64_t token_count,
    int64_t row_count,
    uint64_t expected_destination_token
);

#ifdef __cplusplus
}
#endif

#undef EIEL_STABLE_GRU_EXPORT
