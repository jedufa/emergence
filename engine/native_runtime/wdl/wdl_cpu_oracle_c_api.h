#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define EIEL_WDL_CPU_ORACLE_ABI_VERSION_V1 1

typedef struct eiel_wdl_cpu_oracle_handle_v1 eiel_wdl_cpu_oracle_handle_v1;

typedef struct {
    int32_t schema_version;
    int32_t height;
    int32_t width;
    int32_t delay_steps;
    int32_t cue_duration_steps;
    int32_t active_duration_steps;
    int32_t event_period_steps;
    int32_t radius;
    float intensity;
    float observation_dropout_rate;
    uint8_t enabled;
    uint8_t cue_enabled;
    uint8_t observation_enabled;
    uint8_t damage_enabled;
    uint8_t deterministic_schedule_enabled;
    int32_t view_radius;
    uint64_t max_scheduled_events;
    uint64_t max_damage_events;
} eiel_wdl_cpu_oracle_config_v1;

typedef struct {
    int64_t cue_cell_count;
    int64_t active_cell_count;
    int64_t event_count;
    int64_t cue_without_active_count;
    int64_t active_without_prior_cue_count;
    double cue_to_active_delay_sum;
    int64_t cue_to_active_delay_count;
} eiel_wdl_tick_metrics_v1;

typedef struct {
    int64_t active_exposure_count;
    int64_t agents_damaged;
    int64_t deaths;
    double damage_total;
    float damage_max;
} eiel_wdl_damage_metrics_v1;

typedef struct {
    int64_t event_id;
    int64_t abs_step;
    int32_t center_y;
    int32_t center_x;
    int64_t scheduled_cell_count;
    float intensity;
} eiel_wdl_scheduled_event_v1;

typedef struct {
    int64_t abs_step;
    int32_t agent_id;
    int32_t y;
    int32_t x;
    float raw_intensity;
    float damage_scale;
    float effective_damage;
    float health_before;
    float health_after;
    uint8_t died;
} eiel_wdl_damage_event_v1;

typedef struct {
    int64_t abs_step;
    eiel_wdl_tick_metrics_v1 tick;
    eiel_wdl_damage_metrics_v1 damage;
    eiel_wdl_scheduled_event_v1 scheduled_event;
    uint64_t observation_rows;
    uint64_t observation_width;
} eiel_wdl_tick_receipt_v1;

typedef struct {
    int64_t event_count;
    double cue_cell_count_mean;
    double active_cell_count_mean;
    int64_t agent_cue_exposure_count;
    int64_t agent_active_exposure_count;
    double damage_total;
    float damage_max;
    int64_t agents_damaged;
    double cue_to_active_delay_mean;
    int64_t cue_without_active_count;
    int64_t active_without_prior_cue_count;
    int64_t enabled;
    int64_t tick_count;
} eiel_wdl_generation_summary_v1;

enum eiel_wdl_state_field_v1 {
    EIEL_WDL_STATE_CUE_V1 = 0,
    EIEL_WDL_STATE_TIMER_V1 = 1,
    EIEL_WDL_STATE_ACTIVE_V1 = 2,
    EIEL_WDL_STATE_INTENSITY_V1 = 3,
    EIEL_WDL_STATE_AGE_V1 = 4,
    EIEL_WDL_STATE_EVENT_ID_V1 = 5,
    EIEL_WDL_STATE_PRIOR_CUE_V1 = 6
};

int32_t eiel_wdl_cpu_oracle_abi_version_v1(void);
eiel_wdl_cpu_oracle_handle_v1* eiel_wdl_cpu_oracle_create_v1(
    const eiel_wdl_cpu_oracle_config_v1* config,
    int64_t base_seed,
    uint64_t agent_count,
    int32_t* status_out
);
void eiel_wdl_cpu_oracle_destroy_v1(eiel_wdl_cpu_oracle_handle_v1* handle);
int32_t eiel_wdl_cpu_oracle_set_agents_v1(
    eiel_wdl_cpu_oracle_handle_v1* handle,
    const uint8_t* alive,
    const int32_t* pos_y,
    const int32_t* pos_x,
    const float* health,
    const float* damage_scale,
    uint64_t agent_count
);
int32_t eiel_wdl_cpu_oracle_step_v1(
    eiel_wdl_cpu_oracle_handle_v1* handle,
    int64_t abs_step,
    const int32_t* post_action_y,
    const int32_t* post_action_x,
    uint64_t position_count,
    uint8_t cancel_requested,
    eiel_wdl_tick_receipt_v1* receipt
);
int32_t eiel_wdl_cpu_oracle_begin_generation_v1(
    eiel_wdl_cpu_oracle_handle_v1* handle,
    uint8_t clear_event_journals
);
uint64_t eiel_wdl_cpu_oracle_observation_count_v1(const eiel_wdl_cpu_oracle_handle_v1* handle);
uint64_t eiel_wdl_cpu_oracle_observation_row_count_v1(const eiel_wdl_cpu_oracle_handle_v1* handle);
int32_t eiel_wdl_cpu_oracle_copy_observation_agent_ids_v1(
    const eiel_wdl_cpu_oracle_handle_v1* handle,
    int32_t* output,
    uint64_t capacity
);
int32_t eiel_wdl_cpu_oracle_copy_observation_v1(
    const eiel_wdl_cpu_oracle_handle_v1* handle,
    float* output,
    uint64_t capacity
);
uint64_t eiel_wdl_cpu_oracle_state_field_bytes_v1(
    const eiel_wdl_cpu_oracle_handle_v1* handle,
    int32_t field
);
/*
 * Raw state-field copies use the local host's byte order and floating-point
 * representation. They are an in-process ABI view, not a portable wire or
 * checkpoint format. Use the checkpoint API for the versioned little-endian
 * serialization contract.
 */
int32_t eiel_wdl_cpu_oracle_copy_state_field_v1(
    const eiel_wdl_cpu_oracle_handle_v1* handle,
    int32_t field,
    void* output,
    uint64_t capacity_bytes
);
int32_t eiel_wdl_cpu_oracle_copy_agents_v1(
    const eiel_wdl_cpu_oracle_handle_v1* handle,
    uint8_t* alive,
    int32_t* pos_y,
    int32_t* pos_x,
    float* health,
    uint64_t capacity
);
int32_t eiel_wdl_cpu_oracle_generation_summary_v1(
    const eiel_wdl_cpu_oracle_handle_v1* handle,
    eiel_wdl_generation_summary_v1* summary
);
uint64_t eiel_wdl_cpu_oracle_scheduled_event_count_v1(const eiel_wdl_cpu_oracle_handle_v1* handle);
int32_t eiel_wdl_cpu_oracle_copy_scheduled_events_v1(
    const eiel_wdl_cpu_oracle_handle_v1* handle,
    eiel_wdl_scheduled_event_v1* output,
    uint64_t capacity
);
uint64_t eiel_wdl_cpu_oracle_damage_event_count_v1(const eiel_wdl_cpu_oracle_handle_v1* handle);
int32_t eiel_wdl_cpu_oracle_copy_damage_events_v1(
    const eiel_wdl_cpu_oracle_handle_v1* handle,
    eiel_wdl_damage_event_v1* output,
    uint64_t capacity
);
uint64_t eiel_wdl_cpu_oracle_checkpoint_size_v1(const eiel_wdl_cpu_oracle_handle_v1* handle);
int32_t eiel_wdl_cpu_oracle_write_checkpoint_v1(
    const eiel_wdl_cpu_oracle_handle_v1* handle,
    uint8_t* output,
    uint64_t capacity,
    uint64_t* bytes_written
);
int32_t eiel_wdl_cpu_oracle_restore_checkpoint_v1(
    eiel_wdl_cpu_oracle_handle_v1* handle,
    const uint8_t* input,
    uint64_t size
);

#ifdef __cplusplus
}
#endif
