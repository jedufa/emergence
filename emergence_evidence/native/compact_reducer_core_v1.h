#ifndef EIEL_NATIVE_COMPACT_REDUCER_CORE_V1_H
#define EIEL_NATIVE_COMPACT_REDUCER_CORE_V1_H

#include <stdint.h>

#if defined(__GNUC__) || defined(__clang__)
#define EIEL_COMPACT_REDUCER_EXPORT __attribute__((visibility("default")))
#else
#define EIEL_COMPACT_REDUCER_EXPORT
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define EIEL_NATIVE_COMPACT_REDUCER_ABI_VERSION_V1 1U
#define EIEL_COMPACT_GENOME_ROW_BYTES_V1 288
#define EIEL_COMPACT_PACKED_BIRTH_PROVENANCE_BYTES_V1 68
#define EIEL_COMPACT_LINKED_LOCI_V1 128
#define EIEL_COMPACT_RECOMBINATION_CHOICES_V1 32
#define EIEL_COMPACT_DEATH_METRIC_FIELDS_V1 21
#define EIEL_COMPACT_DEATH_METRIC_ROW_BYTES_V1 88
#define EIEL_COMPACT_STRATEGY_MAX_IDS_V1 64
#define EIEL_COMPACT_STRATEGY_MAX_THRESHOLDS_V1 16
#define EIEL_COMPACT_STRATEGY_MAX_GROUPS_V1 256
#define EIEL_COMPACT_STRATEGY_MAX_PARENT_PAIRS_V1 1024
#define EIEL_COMPACT_SNAPSHOT_MAX_POLICY_LOCI_V1 112
#define EIEL_COMPACT_SNAPSHOT_MAX_TRAIT_LOCI_V1 16

enum eiel_native_compact_reducer_status_v1 {
    EIEL_NATIVE_COMPACT_REDUCER_SUCCESS_V1 = 0,
    EIEL_NATIVE_COMPACT_REDUCER_NULL_POINTER_V1 = 1,
    EIEL_NATIVE_COMPACT_REDUCER_INVALID_EXTENT_V1 = 2,
    EIEL_NATIVE_COMPACT_REDUCER_INVALID_MUTATION_STEP_V1 = 3,
    EIEL_NATIVE_COMPACT_REDUCER_RESERVED_MUTATION_CODE_V1 = 4,
    EIEL_NATIVE_COMPACT_REDUCER_COUNTER_OVERFLOW_V1 = 5,
    EIEL_NATIVE_COMPACT_REDUCER_ALLOCATION_FAILURE_V1 = 6,
    EIEL_NATIVE_COMPACT_REDUCER_INVALID_STATE_V1 = 7,
    EIEL_NATIVE_COMPACT_REDUCER_INVALID_WORLD_V1 = 8,
    EIEL_NATIVE_COMPACT_REDUCER_NONCONTIGUOUS_TICK_V1 = 9,
    EIEL_NATIVE_COMPACT_REDUCER_INVALID_EVENT_V1 = 10,
    EIEL_NATIVE_COMPACT_REDUCER_INACTIVE_AGENT_V1 = 11,
    EIEL_NATIVE_COMPACT_REDUCER_DUPLICATE_AGENT_V1 = 12,
    EIEL_NATIVE_COMPACT_REDUCER_CAPACITY_EXCEEDED_V1 = 13,
    EIEL_NATIVE_COMPACT_REDUCER_AUXILIARY_CURSOR_V1 = 14,
    EIEL_NATIVE_COMPACT_REDUCER_POISONED_V1 = 15,
    EIEL_NATIVE_COMPACT_REDUCER_THREAD_FAILURE_V1 = 16
};

typedef struct eiel_native_compact_birth_batch_receipt_v1 {
    uint64_t reconstructed_birth_count;
    uint64_t mutation_origin_count;
    uint64_t saturated_noop_count;
    uint64_t recombination_nonzero_pattern_count;
    uint64_t mutation_origin_counts_by_linked_locus[128];
    int64_t mutation_delta_totals_by_linked_locus[128];
    uint64_t recombination_choice_one_counts[32];
} eiel_native_compact_birth_batch_receipt_v1;

typedef struct eiel_native_compact_i128_v1 {
    uint64_t low;
    int64_t high;
} eiel_native_compact_i128_v1;

typedef struct eiel_native_compact_active_record_v1 {
    uint64_t agent_id;
    uint64_t generation;
    uint64_t strategy_mask;
    int64_t birth_tick;
    uint8_t mutation_lineage;
    uint8_t born;
    uint8_t reproduced;
    uint8_t founder;
    uint8_t latest_metrics_present;
    uint8_t reserved[3];
    uint8_t genome[288];
} eiel_native_compact_active_record_v1;

typedef struct eiel_native_compact_strategy_group_count_v1 {
    uint64_t strategy_mask;
    uint64_t count;
} eiel_native_compact_strategy_group_count_v1;

typedef struct eiel_native_compact_strategy_parent_pair_count_v1 {
    uint64_t first_strategy_mask;
    uint64_t second_strategy_mask;
    uint64_t count;
} eiel_native_compact_strategy_parent_pair_count_v1;

typedef struct eiel_native_compact_strategy_receipt_v1 {
    uint8_t enabled;
    uint8_t finalized;
    uint8_t strategy_count;
    uint8_t threshold_count;
    uint8_t reserved[4];
    int64_t early_life_tick;
    int64_t first_birth_tick;
    int64_t first_death_tick;
    int64_t extinction_tick;
    uint64_t total_agent_count;
    uint64_t founder_count;
    uint64_t birth_count;
    uint64_t death_count;
    uint64_t maximum_lineage_depth;
    uint64_t mixed_parent_birth_count;
    int64_t threshold_ticks[16];
    uint64_t agent_counts_by_strategy[64];
    uint64_t offspring_born_by_strategy[64];
    uint64_t offspring_survived_by_strategy_threshold[1024];
    uint64_t maximum_lineage_depth_by_strategy[64];
    uint64_t parent_birth_events_by_strategy[64];
    uint64_t death_events_by_strategy[64];
    uint64_t persistence_to_early_life_by_strategy[64];
    uint64_t terminal_alive_by_strategy[64];
    eiel_native_compact_i128_v1 terminal_metric_totals_by_strategy[1344];
    uint64_t offspring_group_count;
    eiel_native_compact_strategy_group_count_v1 offspring_groups[256];
    uint64_t parent_pair_count;
    eiel_native_compact_strategy_parent_pair_count_v1 parent_pairs[1024];
} eiel_native_compact_strategy_receipt_v1;

typedef struct eiel_native_compact_world_receipt_v1 {
    uint64_t birth_count;
    uint64_t death_count;
    uint64_t recombination_birth_count;
    uint64_t carrier_reproduction_birth_count;
    uint64_t mutation_carrier_reproduction_birth_count;
    uint64_t unique_born_agents_reproduced_count;
    uint64_t unique_mutation_carriers_reproduced_count;
    uint64_t maximum_lineage_depth;
    uint64_t dispatch_count;
    uint64_t event_count;
    uint64_t dispatch_tick_count;
    int64_t next_absolute_tick;
    uint64_t active_agent_count;
    eiel_native_compact_birth_batch_receipt_v1 births;
    eiel_native_compact_i128_v1 completed_lifetime_metric_totals[21];
} eiel_native_compact_world_receipt_v1;

typedef struct eiel_native_compact_failure_v1 {
    int32_t status;
    uint32_t world_index;
    int64_t absolute_tick;
    int64_t event_row_index;
    uint64_t agent_id;
    uint64_t parent_a_id;
    uint64_t parent_b_id;
} eiel_native_compact_failure_v1;

EIEL_COMPACT_REDUCER_EXPORT uint32_t
eiel_native_compact_reducer_abi_version_v1(void);

EIEL_COMPACT_REDUCER_EXPORT int64_t
eiel_native_compact_reducer_birth_receipt_size_v1(void);

EIEL_COMPACT_REDUCER_EXPORT int64_t
eiel_native_compact_reducer_active_record_size_v1(void);

EIEL_COMPACT_REDUCER_EXPORT int64_t
eiel_native_compact_reducer_world_receipt_size_v1(void);

EIEL_COMPACT_REDUCER_EXPORT int64_t
eiel_native_compact_reducer_strategy_receipt_size_v1(void);

EIEL_COMPACT_REDUCER_EXPORT int32_t
eiel_native_compact_reconstruct_birth_v1(
    const uint8_t* parent_a_genome,
    const uint8_t* parent_b_genome,
    const uint8_t* packed_birth_provenance,
    int32_t mutation_step,
    uint8_t* child_genome,
    eiel_native_compact_birth_batch_receipt_v1* receipt);

EIEL_COMPACT_REDUCER_EXPORT int32_t
eiel_native_compact_reconstruct_birth_batch_v1(
    const uint8_t* parent_a_genomes,
    const uint8_t* parent_b_genomes,
    const uint8_t* packed_birth_provenance_rows,
    const int32_t* mutation_steps,
    int64_t row_count,
    uint8_t* child_genomes,
    eiel_native_compact_birth_batch_receipt_v1* receipt);

EIEL_COMPACT_REDUCER_EXPORT int32_t
eiel_native_compact_accumulate_death_metrics_v1(
    const uint8_t* metric_rows,
    int64_t row_count,
    eiel_native_compact_i128_v1 totals[21]);

EIEL_COMPACT_REDUCER_EXPORT int32_t
eiel_native_compact_state_create_v1(
    int64_t world_count,
    const uint32_t* resident_capacities,
    const int64_t* start_next_absolute_ticks,
    const int64_t* mutation_segment_offsets,
    const int64_t* mutation_segment_start_ticks,
    const int32_t* mutation_segment_steps,
    int64_t mutation_segment_count,
    int32_t max_worker_threads,
    void** output_handle);

EIEL_COMPACT_REDUCER_EXPORT void
eiel_native_compact_state_destroy_v1(void* handle);

EIEL_COMPACT_REDUCER_EXPORT int32_t
eiel_native_compact_state_initialize_world_v1(
    void* handle,
    int64_t world_index,
    const uint64_t* agent_ids,
    const uint8_t* packed_genomes,
    int64_t active_agent_count);

EIEL_COMPACT_REDUCER_EXPORT int32_t
eiel_native_compact_state_configure_snapshot_world_v1(
    void* handle,
    int64_t world_index,
    const uint8_t* cell_id,
    int64_t cell_id_bytes,
    const uint16_t* tracked_policy_loci,
    int32_t tracked_policy_locus_count,
    const uint16_t* tracked_trait_loci,
    int32_t tracked_trait_locus_count,
    int32_t strategy_count,
    const uint8_t* strategy_policy_sha256_rows,
    const uint64_t* strategy_policy_masks,
    int32_t strategy_policy_count,
    const int64_t* threshold_ticks,
    int32_t threshold_count,
    int64_t early_life_tick);

EIEL_COMPACT_REDUCER_EXPORT int32_t
eiel_native_compact_state_configure_genotype_attribution_v1(
    void* handle,
    int64_t world_index);

EIEL_COMPACT_REDUCER_EXPORT int32_t
eiel_native_compact_state_configure_genotype_focal_ids_v1(
    void* handle,
    int64_t world_index,
    const uint8_t* focal_ids,
    int64_t focal_id_count);

EIEL_COMPACT_REDUCER_EXPORT int32_t
eiel_native_compact_state_configure_hereditary_birth_output_v1(
    void* handle,
    int64_t world_index,
    const uint8_t* path,
    int64_t path_bytes);

EIEL_COMPACT_REDUCER_EXPORT int32_t
eiel_native_compact_state_finalize_hereditary_birth_output_v1(
    void* handle,
    int64_t world_index);

EIEL_COMPACT_REDUCER_EXPORT int32_t
eiel_native_compact_state_process_snapshot_v1(
    void* handle,
    int64_t absolute_tick,
    const uint8_t* canonical_json_lines,
    int64_t canonical_json_bytes,
    int64_t row_count);

EIEL_COMPACT_REDUCER_EXPORT int32_t
eiel_native_compact_state_snapshot_result_size_v1(
    const void* handle,
    int64_t* output_size);

EIEL_COMPACT_REDUCER_EXPORT int32_t
eiel_native_compact_state_copy_snapshot_result_v1(
    const void* handle,
    uint8_t* output,
    int64_t output_capacity);

EIEL_COMPACT_REDUCER_EXPORT int32_t
eiel_native_compact_state_process_dispatch_v1(
    void* handle,
    int64_t start_next_absolute_tick,
    int64_t end_next_absolute_tick,
    const uint8_t* event_rows,
    int64_t event_bytes,
    const uint8_t* death_metric_rows,
    int64_t death_metric_bytes,
    const uint8_t* packed_birth_provenance_rows,
    int64_t birth_provenance_bytes,
    const int64_t* world_offsets,
    int64_t world_offset_count);

EIEL_COMPACT_REDUCER_EXPORT int32_t
eiel_native_compact_state_world_receipt_v1(
    const void* handle,
    int64_t world_index,
    eiel_native_compact_world_receipt_v1* output_receipt);

EIEL_COMPACT_REDUCER_EXPORT int32_t
eiel_native_compact_state_export_active_v1(
    const void* handle,
    int64_t world_index,
    eiel_native_compact_active_record_v1* output_records,
    int64_t record_capacity,
    int64_t* output_record_count);

EIEL_COMPACT_REDUCER_EXPORT int32_t
eiel_native_compact_state_failure_v1(
    const void* handle,
    eiel_native_compact_failure_v1* output_failure);

EIEL_COMPACT_REDUCER_EXPORT int32_t
eiel_native_compact_state_configure_strategy_v1(
    void* handle,
    int64_t world_index,
    int32_t strategy_count,
    const uint64_t* founder_agent_ids,
    const uint64_t* founder_strategy_masks,
    int64_t founder_count,
    const int64_t* threshold_ticks,
    int32_t threshold_count,
    int64_t early_life_tick);

EIEL_COMPACT_REDUCER_EXPORT int32_t
eiel_native_compact_state_observe_strategy_snapshot_v1(
    void* handle,
    int64_t world_index,
    const uint64_t* active_agent_ids,
    const uint8_t* active_metric_rows,
    int64_t active_agent_count);

EIEL_COMPACT_REDUCER_EXPORT int32_t
eiel_native_compact_state_finalize_strategy_v1(
    void* handle,
    int64_t world_index,
    int64_t terminal_next_absolute_tick);

EIEL_COMPACT_REDUCER_EXPORT int32_t
eiel_native_compact_state_strategy_receipt_v1(
    const void* handle,
    int64_t world_index,
    eiel_native_compact_strategy_receipt_v1* output_receipt);

EIEL_COMPACT_REDUCER_EXPORT int32_t
eiel_native_compact_state_genotype_result_size_v1(
    const void* handle,
    int64_t world_index,
    int64_t* output_size);

EIEL_COMPACT_REDUCER_EXPORT int32_t
eiel_native_compact_state_copy_genotype_result_v1(
    const void* handle,
    int64_t world_index,
    uint8_t* output,
    int64_t output_capacity);

#ifdef __cplusplus
}
#endif

#endif
