#include "cuda_architecture_b_compact_summary_reporting_authority_v1.cuh"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace eiel::cuda_architecture_b::compact_summary_reporting_authority_v1 {
namespace {

struct FixtureSourceStorageV1 {
    std::int64_t slot_to_agent_id[4];
    double energy_collected_exact[4];
    double early_energy_collected_exact[4];
    float injury[4];
    float aggression_bias[4];
    float early_injury_gained[4];
    float early_life_weight[4];
    float drive_jitter[4];
    float pred_err_sum[4];
    float birth_reset_shock[4];
    float episode0_pred_err_sum[4];
    float late_pred_err_sum[4];
    std::int32_t lifespan_steps[4];
    std::int32_t offspring_count[4];
    std::int32_t attacks_made[4];
    std::int32_t early_survival_steps[4];
    std::int32_t action_switches[4];
    std::int32_t action_oscillations[4];
    std::int32_t selfharm_events[4];
    std::int32_t pred_err_count[4];
    std::int32_t episode0_pred_err_count[4];
    std::int32_t episode0_selfharm_events[4];
    std::int32_t late_pred_err_count[4];
    double policy_energy_observation[4];
    float health[4];
    float drives[16];
    float identity_drift_ema[4];
    float exploration_bias[4];
    std::int32_t pos_y[4];
    std::int32_t pos_x[4];
    std::int32_t lineage_founder_id[4];
    std::int32_t lineage_birth_generation[4];
    std::int32_t lineage_birth_tick[4];
    std::int32_t lineage_depth[4];
    std::int32_t first_repro_step[4];
    std::int32_t patch_habitat_lineage_id[4];
    std::int32_t genome_hidden_size[4];
    std::int32_t genome_memory_dim[4];
    std::uint8_t policy_use_gru[4];
};

__device__ __forceinline__ GenerationAuthorityAccessV1 access_at(
    std::uint8_t* const retained_bytes,
    std::int32_t* const slot_to_behavior_row,
    double* const effective_fitness_by_row,
    const std::int32_t generation,
    const std::int32_t row_capacity,
    const std::int32_t slot_capacity
) noexcept {
    const std::size_t byte_stride = sizeof(GenerationAuthorityHeaderV1)
        + static_cast<std::size_t>(row_capacity)
            * (sizeof(BehaviorAuthorityRowV1) + sizeof(LineageAuthorityRowV1));
    std::uint8_t* const row_base =
        retained_bytes + static_cast<std::size_t>(generation) * byte_stride;
    auto* const header =
        reinterpret_cast<GenerationAuthorityHeaderV1*>(row_base);
    auto* const behavior = reinterpret_cast<BehaviorAuthorityRowV1*>(
        row_base + sizeof(GenerationAuthorityHeaderV1)
    );
    auto* const lineage = reinterpret_cast<LineageAuthorityRowV1*>(
        row_base + sizeof(GenerationAuthorityHeaderV1)
        + static_cast<std::size_t>(row_capacity)
            * sizeof(BehaviorAuthorityRowV1)
    );
    return {
        header,
        behavior,
        lineage,
        slot_to_behavior_row
            + static_cast<std::int64_t>(generation) * slot_capacity,
        effective_fitness_by_row
            + static_cast<std::int64_t>(generation) * row_capacity,
        row_capacity,
        slot_capacity,
    };
}

__device__ __forceinline__ void initialize_fixture_sources(
    FixtureSourceStorageV1* const storage,
    const std::int64_t first_id,
    const std::int64_t second_id
) noexcept {
    clear_pod_v1(storage);
    for (std::int32_t slot = 0; slot < 4; ++slot) {
        storage->slot_to_agent_id[slot] = -1;
        storage->early_life_weight[slot] = 1.0f;
        storage->health[slot] = 1.0f;
        storage->lineage_founder_id[slot] = slot;
        storage->lineage_birth_generation[slot] = 0;
        storage->lineage_birth_tick[slot] = slot;
        storage->first_repro_step[slot] = -1;
        storage->genome_hidden_size[slot] = 8;
        storage->genome_memory_dim[slot] = 4;
    }
    storage->slot_to_agent_id[0] = first_id;
    storage->slot_to_agent_id[1] = second_id;
    for (std::int32_t slot = 0; slot < 2; ++slot) {
        storage->energy_collected_exact[slot] = 1.5 + slot;
        storage->early_energy_collected_exact[slot] = 0.25 + slot;
        storage->injury[slot] = 0.125f * static_cast<float>(slot);
        storage->aggression_bias[slot] = slot == 0 ? 0.123456789f : -0.25f;
        storage->drive_jitter[slot] = 0.03125f * static_cast<float>(slot + 1);
        storage->pred_err_sum[slot] = 0.5f * static_cast<float>(slot + 1);
        storage->episode0_pred_err_sum[slot] = 0.25f * static_cast<float>(slot + 1);
        storage->late_pred_err_sum[slot] = 0.125f * static_cast<float>(slot + 1);
        storage->lifespan_steps[slot] = 5 + slot;
        storage->offspring_count[slot] = slot;
        storage->attacks_made[slot] = slot;
        storage->early_survival_steps[slot] = 2 + slot;
        storage->action_switches[slot] = slot;
        storage->action_oscillations[slot] = slot + 1;
        storage->selfharm_events[slot] = slot;
        storage->pred_err_count[slot] = slot + 1;
        storage->episode0_pred_err_count[slot] = slot + 1;
        storage->episode0_selfharm_events[slot] = slot;
        storage->late_pred_err_count[slot] = slot + 1;
        storage->policy_energy_observation[slot] = 40.0 + slot;
        storage->drives[slot * 4 + 0] = 0.1f + slot;
        storage->drives[slot * 4 + 1] = 0.2f + slot;
        storage->drives[slot * 4 + 2] = 0.3f + slot;
        storage->drives[slot * 4 + 3] = 0.4f + slot;
        storage->identity_drift_ema[slot] = 0.015625f * static_cast<float>(slot + 1);
        storage->exploration_bias[slot] = -0.125f * static_cast<float>(slot + 1);
        storage->pos_y[slot] = 2 + slot;
        storage->pos_x[slot] = 3 + slot;
        storage->lineage_founder_id[slot] = slot == 0 ? 12 : 11;
        storage->lineage_birth_generation[slot] = slot == 0 ? 0 : 0;
        storage->lineage_birth_tick[slot] = slot == 0 ? 3 : 0;
        storage->lineage_depth[slot] = slot == 0 ? 1 : 0;
        storage->first_repro_step[slot] = slot == 0 ? -1 : 4;
        storage->patch_habitat_lineage_id[slot] = 7 + slot;
        storage->genome_hidden_size[slot] = 8 + slot;
        storage->genome_memory_dim[slot] = 4 + slot;
        storage->policy_use_gru[slot] = static_cast<std::uint8_t>(slot);
    }
}

__device__ __forceinline__ ResidentGenerationSourcesV1 source_view(
    const FixtureSourceStorageV1& storage,
    const std::int32_t slot_capacity
) noexcept {
    return {
        storage.slot_to_agent_id,
        storage.energy_collected_exact,
        storage.early_energy_collected_exact,
        storage.injury,
        storage.aggression_bias,
        storage.early_injury_gained,
        storage.early_life_weight,
        storage.drive_jitter,
        storage.pred_err_sum,
        storage.birth_reset_shock,
        storage.episode0_pred_err_sum,
        storage.late_pred_err_sum,
        storage.lifespan_steps,
        storage.offspring_count,
        storage.attacks_made,
        storage.early_survival_steps,
        storage.action_switches,
        storage.action_oscillations,
        storage.selfharm_events,
        storage.pred_err_count,
        storage.episode0_pred_err_count,
        storage.episode0_selfharm_events,
        storage.late_pred_err_count,
        storage.policy_energy_observation,
        storage.health,
        storage.drives,
        storage.identity_drift_ema,
        storage.exploration_bias,
        storage.pos_y,
        storage.pos_x,
        storage.lineage_founder_id,
        storage.lineage_birth_generation,
        storage.lineage_birth_tick,
        storage.lineage_depth,
        storage.first_repro_step,
        storage.patch_habitat_lineage_id,
        storage.genome_hidden_size,
        storage.genome_memory_dim,
        storage.policy_use_gru,
        1,
        slot_capacity,
    };
}

__device__ __forceinline__ ReportingFitnessConfigV1 fixture_config() noexcept {
    return {
        0.25,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.15,
        0.08,
        3.0,
        1.25,
        0.35,
        0.75,
        0.0,
        0,
        0,
        1,
        {0, 0, 0, 0, 0},
    };
}

__device__ __forceinline__ BehaviorTerminalValuesV1 death_terminal() noexcept {
    BehaviorTerminalValuesV1 value{};
    value.energy_collected_exact = 0.75;
    value.early_energy_collected_exact = 0.25;
    value.injury_end = 0.5f;
    value.aggression_bias_f32 = 0.375f;
    value.early_life_weight = 1.0f;
    value.lifespan_steps = 3;
    value.early_survival_steps = 2;
    value.pred_err_count = 1;
    value.episode0_pred_err_count = 1;
    value.late_pred_err_count = 1;
    return value;
}

__device__ __forceinline__ void run_normal_fixture(
    const GenerationAuthorityAccessV1& access
) noexcept {
    const std::int64_t ids[2] = {10, 11};
    const std::int32_t slots[2] = {0, 1};
    const float injury[2] = {-0.0f, 0.25f};
    if (!initialize_generation_v1(access, 12, ids, slots, injury, 2)) {
        return;
    }
    count_action_v1(access, 0, 10, 0);
    count_action_v1(access, 0, 10, 0);
    count_action_v1(access, 1, 11, 6);
    seal_death_before_slot_reuse_v1(access, 0, 10, death_terminal());
    append_ordinary_birth_v1(access, 12, 0, 0.0f);
    count_action_v1(access, 0, 12, 4);

    FixtureSourceStorageV1 sources{};
    initialize_fixture_sources(&sources, 12, 11);
    finalize_pre_turnover_v1(
        access, source_view(sources, access.slot_capacity), fixture_config()
    );
}

__device__ __forceinline__ void run_empty_fixture(
    const GenerationAuthorityAccessV1& access
) noexcept {
    initialize_generation_v1(access, 0, nullptr, nullptr, nullptr, 0);
    FixtureSourceStorageV1 sources{};
    initialize_fixture_sources(&sources, -1, -1);
    finalize_pre_turnover_v1(
        access, source_view(sources, access.slot_capacity), fixture_config()
    );
}

__device__ __forceinline__ void run_overflow_fixture(
    const GenerationAuthorityAccessV1& access
) noexcept {
    std::int64_t ids[kStrictRowCapacityV1];
    std::int32_t slots[kStrictRowCapacityV1];
    float injury[kStrictRowCapacityV1];
    for (std::int32_t index = 0; index < kStrictRowCapacityV1; ++index) {
        ids[index] = index;
        slots[index] = index;
        injury[index] = 0.0f;
    }
    if (!initialize_generation_v1(
            access,
            kStrictRowCapacityV1,
            ids,
            slots,
            injury,
            kStrictRowCapacityV1
        )) {
        return;
    }
    append_ordinary_birth_v1(
        access, kStrictRowCapacityV1, kStrictRowCapacityV1, 0.0f
    );
}

__device__ __forceinline__ void run_duplicate_fixture(
    const GenerationAuthorityAccessV1& access
) noexcept {
    const std::int64_t ids[1] = {0};
    const std::int32_t slots[1] = {0};
    const float injury[1] = {0.0f};
    initialize_generation_v1(access, 1, ids, slots, injury, 1);
    append_ordinary_birth_v1(access, 1, 1, 0.0f);
    append_ordinary_birth_v1(access, 1, 2, 0.0f);
}

__device__ __forceinline__ void run_signed_zero_fixture(
    const GenerationAuthorityAccessV1& access
) noexcept {
    const std::int64_t ids[1] = {0};
    const std::int32_t slots[1] = {0};
    const float injury[1] = {-0.0f};
    initialize_generation_v1(access, 1, ids, slots, injury, 1);
    FixtureSourceStorageV1 sources{};
    initialize_fixture_sources(&sources, 0, -1);
    sources.energy_collected_exact[0] = -0.0;
    sources.early_energy_collected_exact[0] = -0.0;
    sources.injury[0] = -0.0f;
    sources.aggression_bias[0] = -0.0f;
    sources.policy_energy_observation[0] = -0.0;
    sources.health[0] = -0.0f;
    sources.exploration_bias[0] = -0.0f;
    finalize_pre_turnover_v1(
        access, source_view(sources, access.slot_capacity), fixture_config()
    );
}

__device__ __forceinline__ void run_identity_drift_unavailable_fixture(
    const GenerationAuthorityAccessV1& access
) noexcept {
    const std::int64_t ids[1] = {0};
    const std::int32_t slots[1] = {0};
    const float injury[1] = {0.0f};
    initialize_generation_v1(access, 1, ids, slots, injury, 1);
    FixtureSourceStorageV1 sources{};
    initialize_fixture_sources(&sources, 0, -1);
    ResidentGenerationSourcesV1 view = source_view(sources, access.slot_capacity);
    view.identity_drift_ema = nullptr;
    view.identity_drift_ema_available = 0;
    finalize_pre_turnover_v1(access, view, fixture_config());
}

__device__ __forceinline__ void run_missing_lineage_source_fixture(
    const GenerationAuthorityAccessV1& access
) noexcept {
    const std::int64_t ids[1] = {0};
    const std::int32_t slots[1] = {0};
    const float injury[1] = {0.0f};
    initialize_generation_v1(access, 1, ids, slots, injury, 1);
    FixtureSourceStorageV1 sources{};
    initialize_fixture_sources(&sources, 0, -1);
    ResidentGenerationSourcesV1 view = source_view(sources, access.slot_capacity);
    view.exploration_bias = nullptr;
    finalize_pre_turnover_v1(access, view, fixture_config());
}

__device__ __forceinline__ void run_malformed_lineage_source_fixture(
    const GenerationAuthorityAccessV1& access
) noexcept {
    const std::int64_t ids[1] = {0};
    const std::int32_t slots[1] = {0};
    const float injury[1] = {0.0f};
    initialize_generation_v1(access, 1, ids, slots, injury, 1);
    FixtureSourceStorageV1 sources{};
    initialize_fixture_sources(&sources, 0, -1);
    sources.identity_drift_ema[0] = __int_as_float(0x7fc00000);
    finalize_pre_turnover_v1(
        access, source_view(sources, access.slot_capacity), fixture_config()
    );
}

__device__ __forceinline__ void run_isolated_generation(
    const GenerationAuthorityAccessV1& access,
    const std::int64_t agent_id,
    const std::int32_t action
) noexcept {
    const std::int64_t ids[1] = {agent_id};
    const std::int32_t slots[1] = {0};
    const float injury[1] = {0.0f};
    initialize_generation_v1(access, agent_id + 1, ids, slots, injury, 1);
    count_action_v1(access, 0, agent_id, action);
    if (action == 6) {
        count_action_v1(access, 0, agent_id, action);
    }
    FixtureSourceStorageV1 sources{};
    initialize_fixture_sources(&sources, agent_id, -1);
    sources.lineage_founder_id[0] = static_cast<std::int32_t>(agent_id);
    finalize_pre_turnover_v1(
        access, source_view(sources, access.slot_capacity), fixture_config()
    );
}

__global__ void reporting_authority_fixture_kernel_v1(
    std::uint8_t* const retained_bytes,
    std::int32_t* const slot_to_behavior_row,
    double* const effective_fitness_by_row,
    const std::int64_t generations,
    const std::int32_t row_capacity,
    const std::int32_t slot_capacity,
    const std::int32_t fixture
) {
    if (blockIdx.x != 0 || threadIdx.x != 0 || generations <= 0) {
        return;
    }
    GenerationAuthorityAccessV1 access = access_at(
        retained_bytes,
        slot_to_behavior_row,
        effective_fitness_by_row,
        0,
        row_capacity,
        slot_capacity
    );
    switch (fixture) {
        case kFixtureStartBornDiedReuseV1:
            run_normal_fixture(access);
            break;
        case kFixtureEmptyV1:
            run_empty_fixture(access);
            break;
        case kFixtureOverflowV1:
            run_overflow_fixture(access);
            break;
        case kFixtureDuplicateV1:
            run_duplicate_fixture(access);
            break;
        case kFixtureSignedZeroV1:
            run_signed_zero_fixture(access);
            break;
        case kFixtureG2IsolationV1:
            run_isolated_generation(access, 0, 0);
            if (generations >= 2) {
                access = access_at(
                    retained_bytes,
                    slot_to_behavior_row,
                    effective_fitness_by_row,
                    1,
                    row_capacity,
                    slot_capacity
                );
                run_isolated_generation(access, 10, 6);
            }
            break;
        case kFixtureWriteAfterSealV1:
            run_empty_fixture(access);
            count_action_v1(access, 0, 0, 0);
            break;
        case kFixtureIdentityDriftUnavailableV1:
            run_identity_drift_unavailable_fixture(access);
            break;
        case kFixtureMissingLineageSourceV1:
            run_missing_lineage_source_fixture(access);
            break;
        case kFixtureMalformedLineageSourceV1:
            run_malformed_lineage_source_fixture(access);
            break;
        default:
            initialize_generation_v1(access, 0, nullptr, nullptr, nullptr, 0);
            record_failure_v1(access.header, kLedgerInvalidInputV1);
            break;
    }
}

}  // namespace

void launch_reporting_authority_fixture_v1(
    std::uint8_t* const retained_bytes,
    std::int32_t* const slot_to_behavior_row,
    double* const effective_fitness_by_row,
    const std::int64_t generations,
    const std::int32_t row_capacity,
    const std::int32_t slot_capacity,
    const std::int32_t fixture,
    cudaStream_t stream
) {
    reporting_authority_fixture_kernel_v1<<<1, kThreadsPerBlockV1, 0, stream>>>(
        retained_bytes,
        slot_to_behavior_row,
        effective_fitness_by_row,
        generations,
        row_capacity,
        slot_capacity,
        fixture
    );
}

KernelResourceReceiptV1 query_reporting_authority_fixture_v1_resources() {
    cudaFuncAttributes attributes{};
    cudaError_t error = cudaFuncGetAttributes(
        &attributes, reporting_authority_fixture_kernel_v1
    );
    if (error != cudaSuccess) {
        return {-1, -1, -1, -1, -1, -1, -1, -1};
    }
    std::int32_t occupancy = -1;
    error = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &occupancy,
        reporting_authority_fixture_kernel_v1,
        kThreadsPerBlockV1,
        0
    );
    if (error != cudaSuccess) {
        occupancy = -1;
    }
    return {
        attributes.binaryVersion,
        attributes.ptxVersion,
        attributes.maxThreadsPerBlock,
        attributes.numRegs,
        occupancy,
        static_cast<std::int64_t>(attributes.constSizeBytes),
        static_cast<std::int64_t>(attributes.localSizeBytes),
        static_cast<std::int64_t>(attributes.sharedSizeBytes),
    };
}

}  // namespace eiel::cuda_architecture_b::compact_summary_reporting_authority_v1
