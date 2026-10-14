#ifndef EIEL_CUDA_ARCHITECTURE_B_FLATTENED_MODULAR_CONTROLLER_V1_CUH_
#define EIEL_CUDA_ARCHITECTURE_B_FLATTENED_MODULAR_CONTROLLER_V1_CUH_

#include "cuda_architecture_b_flattened_modular_controller_v1.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>

namespace eiel::cuda_architecture_b::flattened_modular_controller_v1 {
namespace device {

constexpr std::int64_t kOrdinaryBirthRowV1DisabledReasonAbsent = 0;
constexpr std::int64_t kOrdinaryBirthRowV1DisabledReasonOperatorDisabled = 1;
constexpr std::int64_t kOrdinaryBirthRowV1DisabledReasonStructuralDisabled = 2;
constexpr std::int64_t kOrdinaryBirthRowV1DisabledReasonRecurrentDisabled = 3;
constexpr std::int64_t kOrdinaryBirthRowV1DisabledReasonOperatorFamilyDisabled = 4;
constexpr std::int64_t kOrdinaryBirthRowV1DisabledReasonZeroMutationRate = 5;
constexpr std::int64_t kOrdinaryBirthRowV1DisabledReasonAttemptBudgetZero = 6;

__device__ __forceinline__ bool valid_ordinary_birth_rowv1_disabled_reason_v3(
    const std::int64_t raw_disabled_reason
) {
    return raw_disabled_reason >= kOrdinaryBirthRowV1DisabledReasonAbsent
        && raw_disabled_reason <= kOrdinaryBirthRowV1DisabledReasonAttemptBudgetZero;
}

__device__ __forceinline__ void clear_ordinary_birth_rowv1_tail_v3(
    std::int64_t* const rowv1
) {
    for (
        std::int64_t index = 23;
        index < kOrdinaryBirthRowV1ReceiptWidthV3;
        ++index
    ) {
        rowv1[index] = -1;
    }
}

__device__ __forceinline__ bool write_ordinary_birth_rowv1_receipt_v3(
    const FlatModularContextV1 context,
    const std::int64_t cell,
    const std::int64_t sequence_index,
    const std::int64_t commit_generation,
    const std::int64_t commit_tick,
    const std::int64_t parent_agent_id,
    const std::int64_t parent_slot,
    const std::int64_t parent_slot_token,
    const std::int64_t child_agent_id,
    const std::int64_t child_slot,
    const std::int64_t child_slot_token,
    const std::int64_t transaction_ordinal,
    const std::int64_t draw0_bits,
    const std::int64_t draw1_bits,
    const std::int64_t operation,
    const std::int64_t status,
    const std::int64_t reason,
    const std::int64_t selected_module,
    const std::int64_t selected_instance,
    const std::int64_t selected_homology,
    const std::int64_t changed_dimension,
    const std::int64_t changed,
    const std::int64_t raw_disabled_reason
) {
    if (
        !valid_ordinary_birth_rowv1_disabled_reason_v3(raw_disabled_reason)
        || context.ordinary_birth_rowv1_receipt_rows == nullptr
        || context.ordinary_birth_rowv1_receipt_counts == nullptr
        || context.ordinary_birth_rowv1_receipt_overflow == nullptr
        || cell < 0 || cell >= context.cells
        || sequence_index < 0
        || sequence_index >= kTurnoverMutationSequenceCapacityV1
    ) {
        return false;
    }
    if (context.ordinary_birth_rowv1_receipt_counts[cell] != sequence_index) {
        context.ordinary_birth_rowv1_receipt_overflow[cell] = 1;
        return false;
    }
    std::int64_t* const rowv1 = context.ordinary_birth_rowv1_receipt_rows
        + (cell * kTurnoverMutationSequenceCapacityV1 + sequence_index)
            * kOrdinaryBirthRowV1ReceiptWidthV3;
    for (
        std::int64_t index = 0;
        index < kOrdinaryBirthRowV1ReceiptWidthV3;
        ++index
    ) {
        rowv1[index] = -1;
    }
    rowv1[0] = 3;
    rowv1[1] = 2;
    rowv1[2] = sequence_index;
    rowv1[3] = commit_generation;
    rowv1[4] = commit_tick;
    rowv1[5] = parent_agent_id;
    rowv1[6] = parent_slot;
    rowv1[7] = parent_slot_token;
    rowv1[8] = child_agent_id;
    rowv1[9] = child_slot;
    rowv1[10] = child_slot_token;
    rowv1[11] = transaction_ordinal;
    rowv1[12] = draw0_bits;
    rowv1[13] = draw1_bits;
    rowv1[14] = operation;
    rowv1[15] = status;
    rowv1[16] = reason;
    rowv1[17] = selected_module;
    rowv1[18] = selected_instance;
    rowv1[19] = selected_homology;
    rowv1[20] = changed_dimension;
    rowv1[21] = changed;
    rowv1[22] = raw_disabled_reason;
    clear_ordinary_birth_rowv1_tail_v3(rowv1);
    context.ordinary_birth_rowv1_receipt_counts[cell] = sequence_index + 1;
    return true;
}

__device__ __forceinline__ bool write_ordinary_birth_rowv1_disabled_event_v3(
    const FlatModularContextV1 context,
    const std::int64_t cell,
    const std::int64_t sequence_index,
    const std::int64_t commit_generation,
    const std::int64_t commit_tick,
    const std::int64_t parent_agent_id,
    const std::int64_t parent_slot,
    const std::int64_t parent_slot_token,
    const std::int64_t child_agent_id,
    const std::int64_t child_slot,
    const std::int64_t child_slot_token,
    const std::int64_t transaction_ordinal,
    const std::int64_t raw_disabled_reason
) {
    if (raw_disabled_reason == kOrdinaryBirthRowV1DisabledReasonAbsent) {
        return false;
    }
    return write_ordinary_birth_rowv1_receipt_v3(
        context,
        cell,
        sequence_index,
        commit_generation,
        commit_tick,
        parent_agent_id,
        parent_slot,
        parent_slot_token,
        child_agent_id,
        child_slot,
        child_slot_token,
        transaction_ordinal,
        0,
        0,
        -1,
        3,
        2,
        -1,
        -1,
        -1,
        0,
        0,
        raw_disabled_reason
    );
}

__device__ __forceinline__ bool runtime_plane_v1(const std::int32_t plane) {
    return plane == kModuleStatePresentV1 || plane == kModuleStateV1
        || plane == kModuleOutputPresentV1 || plane == kModulePreviousOutputV1
        || plane == kDevelopmentLatchPresentV1
        || plane == kDevelopmentLatchStateV1
        || plane == kDevelopmentEffectiveGainPresentV1
        || plane == kDevelopmentEffectiveGainV1;
}

__device__ __forceinline__ bool owner_plane_v1(const std::int32_t plane) {
    return plane == kAgentOccupiedV1 || plane == kAgentIdV1
        || plane == kAgentSlotTokenV1;
}

__device__ __forceinline__ const ModularTensorTripletV2& lifecycle_entry_v2(
    const FlatModularContextV1 context,
    const std::int32_t plane
) {
    return context.lifecycle_tensors.tensors[plane];
}

__device__ __forceinline__ std::uint8_t* lifecycle_row_v2(
    const FlatModularContextV1 context,
    const std::int32_t plane,
    const bool shadow,
    const std::int64_t cell,
    const std::int64_t slot
) {
    const ModularTensorTripletV2& entry = lifecycle_entry_v2(context, plane);
    std::uint8_t* const base = shadow ? entry.shadow_address : entry.source_address;
    return base + cell * entry.bytes_per_cell + slot * entry.bytes_per_row;
}

__device__ __forceinline__ bool lifecycle_table_valid_v2(
    const FlatModularContextV1 context
) {
    if (
        !context.enabled || !context.lifecycle_enabled
        || context.lifecycle_tensors.tensors == nullptr || context.cells <= 0
        || context.capacity <= 0 || context.cold_origin_slot_source == nullptr
        || context.cold_origin_slot_shadow == nullptr
    ) {
        return false;
    }
    for (std::int32_t plane = 0; plane < kPlaneCountV1; ++plane) {
        const ModularTensorTripletV2& entry = lifecycle_entry_v2(context, plane);
        if (
            entry.source_address == nullptr || entry.shadow_address == nullptr
            || entry.bytes_per_row <= 0
            || entry.bytes_per_cell != context.capacity * entry.bytes_per_row
            || (entry.fill_byte != 0 && entry.fill_byte != 0xff)
        ) {
            return false;
        }
    }
    if (
        (context.turnover_mutation_enabled
         && context.ordinary_birth_mutation_enabled)
        ||
        (context.turnover_mutation_enabled
         || context.ordinary_birth_mutation_enabled)
        && (
            context.turnover_mutation_draws == nullptr
            || context.duplicate_descendant_mask_source == nullptr
            || context.duplicate_descendant_mask_shadow == nullptr
            || context.turnover_mutation_receipt_rows == nullptr
            || context.turnover_mutation_receipt_counts == nullptr
            || context.turnover_mutation_receipt_overflow == nullptr
            || (
                context.ordinary_birth_mutation_enabled
                && (
                    context.ordinary_birth_rowv1_receipt_rows == nullptr
                    || context.ordinary_birth_rowv1_receipt_counts == nullptr
                    || context.ordinary_birth_rowv1_receipt_overflow == nullptr
                )
            )
        )
    ) {
        return false;
    }
    return true;
}

__device__ __forceinline__ void copy_lifecycle_cell_source_to_shadow_v2(
    const FlatModularContextV1 context,
    const std::int64_t cell,
    const std::int64_t lane,
    const std::int64_t lanes
) {
    for (std::int32_t plane = 0; plane < kPlaneCountV1; ++plane) {
        const ModularTensorTripletV2& entry = lifecycle_entry_v2(context, plane);
        const std::uint8_t* const source =
            entry.source_address + cell * entry.bytes_per_cell;
        std::uint8_t* const shadow =
            entry.shadow_address + cell * entry.bytes_per_cell;
        for (
            std::int64_t byte = lane;
            byte < entry.bytes_per_cell;
            byte += lanes
        ) {
            shadow[byte] = source[byte];
        }
    }
    for (std::int64_t slot = lane; slot < context.capacity; slot += lanes) {
        const std::int64_t index = cell * context.capacity + slot;
        context.cold_origin_slot_shadow[index] =
            context.cold_origin_slot_source[index];
    }
    if (
        context.turnover_mutation_enabled
        || context.ordinary_birth_mutation_enabled
    ) {
        const std::int64_t mask_values = context.capacity * kModuleSlotsV1;
        for (std::int64_t index = lane; index < mask_values; index += lanes) {
            const std::int64_t offset = cell * mask_values + index;
            context.duplicate_descendant_mask_shadow[offset] =
                context.duplicate_descendant_mask_source[offset];
        }
    }
}

__device__ __forceinline__ void copy_lifecycle_cell_shadow_to_source_v2(
    const FlatModularContextV1 context,
    const std::int64_t cell,
    const std::int64_t lane,
    const std::int64_t lanes
) {
    for (std::int32_t plane = 0; plane < kPlaneCountV1; ++plane) {
        const ModularTensorTripletV2& entry = lifecycle_entry_v2(context, plane);
        const std::uint8_t* const shadow =
            entry.shadow_address + cell * entry.bytes_per_cell;
        std::uint8_t* const source =
            entry.source_address + cell * entry.bytes_per_cell;
        for (
            std::int64_t byte = lane;
            byte < entry.bytes_per_cell;
            byte += lanes
        ) {
            source[byte] = shadow[byte];
        }
    }
    for (std::int64_t slot = lane; slot < context.capacity; slot += lanes) {
        const std::int64_t index = cell * context.capacity + slot;
        context.cold_origin_slot_source[index] =
            context.cold_origin_slot_shadow[index];
    }
    if (
        context.turnover_mutation_enabled
        || context.ordinary_birth_mutation_enabled
    ) {
        const std::int64_t mask_values = context.capacity * kModuleSlotsV1;
        for (std::int64_t index = lane; index < mask_values; index += lanes) {
            const std::int64_t offset = cell * mask_values + index;
            context.duplicate_descendant_mask_source[offset] =
                context.duplicate_descendant_mask_shadow[offset];
        }
    }
}

__device__ __forceinline__ void fill_lifecycle_row_v2(
    const FlatModularContextV1 context,
    const bool shadow,
    const std::int64_t cell,
    const std::int64_t slot
) {
    for (std::int32_t plane = 0; plane < kPlaneCountV1; ++plane) {
        const ModularTensorTripletV2& entry = lifecycle_entry_v2(context, plane);
        std::uint8_t* const row = lifecycle_row_v2(
            context, plane, shadow, cell, slot
        );
        const std::uint8_t fill = static_cast<std::uint8_t>(entry.fill_byte);
        for (std::int64_t byte = 0; byte < entry.bytes_per_row; ++byte) {
            row[byte] = fill;
        }
    }
    const std::int64_t index = cell * context.capacity + slot;
    (shadow ? context.cold_origin_slot_shadow
            : context.cold_origin_slot_source)[index] = -1;
    if (
        context.turnover_mutation_enabled
        || context.ordinary_birth_mutation_enabled
    ) {
        bool* const mask = shadow
            ? context.duplicate_descendant_mask_shadow
            : context.duplicate_descendant_mask_source;
        const std::int64_t base =
            (cell * context.capacity + slot) * kModuleSlotsV1;
        for (std::int64_t module = 0; module < kModuleSlotsV1; ++module) {
            mask[base + module] = false;
        }
    }
}

__device__ __forceinline__ void reset_lifecycle_runtime_row_v2(
    const FlatModularContextV1 context,
    const bool shadow,
    const std::int64_t cell,
    const std::int64_t slot
) {
    for (std::int32_t plane = 0; plane < kPlaneCountV1; ++plane) {
        if (!runtime_plane_v1(plane)) {
            continue;
        }
        const ModularTensorTripletV2& entry = lifecycle_entry_v2(context, plane);
        std::uint8_t* const row = lifecycle_row_v2(
            context, plane, shadow, cell, slot
        );
        const std::uint8_t fill = static_cast<std::uint8_t>(entry.fill_byte);
        for (std::int64_t byte = 0; byte < entry.bytes_per_row; ++byte) {
            row[byte] = fill;
        }
    }
}

__device__ __forceinline__ void copy_lifecycle_structural_row_v2(
    const FlatModularContextV1 context,
    const bool shadow,
    const std::int64_t cell,
    const std::int64_t parent_slot,
    const std::int64_t child_slot
) {
    for (std::int32_t plane = 0; plane < kPlaneCountV1; ++plane) {
        if (runtime_plane_v1(plane) || owner_plane_v1(plane)) {
            continue;
        }
        const ModularTensorTripletV2& entry = lifecycle_entry_v2(context, plane);
        const std::uint8_t* const parent = lifecycle_row_v2(
            context, plane, shadow, cell, parent_slot
        );
        std::uint8_t* const child = lifecycle_row_v2(
            context, plane, shadow, cell, child_slot
        );
        for (std::int64_t byte = 0; byte < entry.bytes_per_row; ++byte) {
            child[byte] = parent[byte];
        }
    }
    if (
        context.turnover_mutation_enabled
        || context.ordinary_birth_mutation_enabled
    ) {
        bool* const mask = shadow
            ? context.duplicate_descendant_mask_shadow
            : context.duplicate_descendant_mask_source;
        const std::int64_t parent_base =
            (cell * context.capacity + parent_slot) * kModuleSlotsV1;
        const std::int64_t child_base =
            (cell * context.capacity + child_slot) * kModuleSlotsV1;
        for (std::int64_t module = 0; module < kModuleSlotsV1; ++module) {
            mask[child_base + module] = mask[parent_base + module];
        }
    }
}

__device__ __forceinline__ void bind_lifecycle_owner_v2(
    const FlatModularContextV1 context,
    const bool shadow,
    const std::int64_t cell,
    const std::int64_t slot,
    const std::int64_t agent_id,
    const std::int64_t slot_token
) {
    *reinterpret_cast<bool*>(lifecycle_row_v2(
        context, kAgentOccupiedV1, shadow, cell, slot
    )) = true;
    *reinterpret_cast<std::int64_t*>(lifecycle_row_v2(
        context, kAgentIdV1, shadow, cell, slot
    )) = agent_id;
    *reinterpret_cast<std::int64_t*>(lifecycle_row_v2(
        context, kAgentSlotTokenV1, shadow, cell, slot
    )) = slot_token;
}

__device__ __forceinline__ bool inherit_lifecycle_newborn_v2(
    const FlatModularContextV1 context,
    const bool shadow,
    const std::int64_t cell,
    const std::int64_t parent_slot,
    const std::int64_t mate_slot,
    const std::int64_t child_slot,
    const std::int64_t child_agent_id,
    const std::int64_t child_slot_token
) {
    if (
        !lifecycle_table_valid_v2(context) || cell < 0 || cell >= context.cells
        || parent_slot < 0 || parent_slot >= context.capacity
        || child_slot < 0 || child_slot >= context.capacity
        || child_agent_id < 0 || child_slot_token < 0
        || (mate_slot >= 0 && mate_slot >= context.capacity)
    ) {
        return false;
    }
    std::int64_t* const origins = shadow
        ? context.cold_origin_slot_shadow : context.cold_origin_slot_source;
    const std::int64_t parent_origin =
        origins[cell * context.capacity + parent_slot];
    if (parent_origin < 0 || parent_origin >= context.capacity) {
        return false;
    }
    copy_lifecycle_structural_row_v2(
        context, shadow, cell, parent_slot, child_slot
    );
    reset_lifecycle_runtime_row_v2(context, shadow, cell, child_slot);
    bind_lifecycle_owner_v2(
        context, shadow, cell, child_slot, child_agent_id, child_slot_token
    );
    origins[cell * context.capacity + child_slot] = parent_origin;
    return true;
}

__device__ __forceinline__ bool stage_ordinary_lifecycle_newborn_v3(
    const FlatModularContextV1 context,
    const std::int64_t cell,
    const std::int64_t parent_slot,
    const std::int64_t child_slot,
    const std::int64_t child_agent_id,
    const std::int64_t child_slot_token
) {
    if (
        !lifecycle_table_valid_v2(context)
        || !context.ordinary_birth_mutation_enabled
        || cell < 0 || cell >= context.cells
        || parent_slot < 0 || parent_slot >= context.capacity
        || child_slot < 0 || child_slot >= context.capacity
        || child_agent_id < 0 || child_slot_token < 0
    ) {
        return false;
    }
    for (std::int32_t plane = 0; plane < kPlaneCountV1; ++plane) {
        if (runtime_plane_v1(plane) || owner_plane_v1(plane)) {
            continue;
        }
        const ModularTensorTripletV2& entry = lifecycle_entry_v2(context, plane);
        const std::uint8_t* const parent = lifecycle_row_v2(
            context, plane, false, cell, parent_slot
        );
        std::uint8_t* const child = lifecycle_row_v2(
            context, plane, true, cell, child_slot
        );
        for (std::int64_t byte = 0; byte < entry.bytes_per_row; ++byte) {
            child[byte] = parent[byte];
        }
    }
    reset_lifecycle_runtime_row_v2(context, true, cell, child_slot);
    bind_lifecycle_owner_v2(
        context, true, cell, child_slot, child_agent_id, child_slot_token
    );
    const std::int64_t parent_index =
        cell * context.capacity + parent_slot;
    const std::int64_t child_index =
        cell * context.capacity + child_slot;
    const std::int64_t parent_origin =
        context.cold_origin_slot_source[parent_index];
    if (parent_origin < 0 || parent_origin >= context.capacity) {
        return false;
    }
    context.cold_origin_slot_shadow[child_index] = parent_origin;
    const std::int64_t parent_mask =
        parent_index * kModuleSlotsV1;
    const std::int64_t child_mask =
        child_index * kModuleSlotsV1;
    for (std::int64_t module = 0; module < kModuleSlotsV1; ++module) {
        context.duplicate_descendant_mask_shadow[child_mask + module] =
            context.duplicate_descendant_mask_source[parent_mask + module];
    }
    return true;
}

__device__ __forceinline__ void publish_ordinary_lifecycle_newborn_v3(
    const FlatModularContextV1 context,
    const std::int64_t cell,
    const std::int64_t child_slot
) {
    for (std::int32_t plane = 0; plane < kPlaneCountV1; ++plane) {
        const ModularTensorTripletV2& entry = lifecycle_entry_v2(context, plane);
        const std::uint8_t* const shadow = lifecycle_row_v2(
            context, plane, true, cell, child_slot
        );
        std::uint8_t* const source = lifecycle_row_v2(
            context, plane, false, cell, child_slot
        );
        for (std::int64_t byte = 0; byte < entry.bytes_per_row; ++byte) {
            source[byte] = shadow[byte];
        }
    }
    const std::int64_t child_index =
        cell * context.capacity + child_slot;
    context.cold_origin_slot_source[child_index] =
        context.cold_origin_slot_shadow[child_index];
    const std::int64_t child_mask =
        child_index * kModuleSlotsV1;
    for (std::int64_t module = 0; module < kModuleSlotsV1; ++module) {
        context.duplicate_descendant_mask_source[child_mask + module] =
            context.duplicate_descendant_mask_shadow[child_mask + module];
    }
}

__device__ __forceinline__ bool mutate_descendant_parameter_v1(
    const FlatModularContextV1 context,
    const bool shadow,
    const bool ordinary_birth,
    const std::int64_t cell,
    const std::int64_t transaction_ordinal,
    const std::int64_t parent_slot,
    const std::int64_t child_slot,
    const std::int64_t parent_agent_id,
    const std::int64_t child_agent_id,
    const std::int64_t child_slot_token,
    const std::int64_t commit_generation,
    const std::int64_t commit_tick
) {
    const bool enabled = ordinary_birth
        ? context.ordinary_birth_mutation_enabled
        : context.turnover_mutation_enabled;
    if (!enabled) {
        return true;
    }
    if (
        !lifecycle_table_valid_v2(context) || cell < 0 || cell >= context.cells
        || transaction_ordinal < 0 || transaction_ordinal >= context.capacity
        || parent_slot < 0 || parent_slot >= context.capacity
        || child_slot < 0 || child_slot >= context.capacity
        || parent_agent_id < 0 || child_agent_id < 0 || child_slot_token < 0
        || (!ordinary_birth
            && (transaction_ordinal != 0 || commit_tick != 0))
        || context.turnover_mutation_first_generation < (ordinary_birth ? 0 : 1)
    ) {
        return false;
    }
    const std::int64_t sequence_index =
        context.turnover_mutation_receipt_counts[cell];
    if (
        sequence_index < 0
        || sequence_index >= kTurnoverMutationSequenceCapacityV1
    ) {
        context.turnover_mutation_receipt_overflow[cell] = 1;
        if (ordinary_birth) {
            context.ordinary_birth_rowv1_receipt_overflow[cell] = 1;
        }
        return false;
    }
    if (
        ordinary_birth
        && context.ordinary_birth_rowv1_receipt_counts[cell] != sequence_index
    ) {
        context.ordinary_birth_rowv1_receipt_overflow[cell] = 1;
        return false;
    }
    if (
        (
            ordinary_birth
            && commit_generation != context.turnover_mutation_first_generation
        )
        || (
            !ordinary_birth
            && commit_generation
                != context.turnover_mutation_first_generation + sequence_index
        )
    ) {
        return false;
    }
    const double* const draws = context.turnover_mutation_draws
        + (cell * kTurnoverMutationSequenceCapacityV1 + sequence_index) * 2;
    const double draw0 = draws[0];
    const double draw1 = draws[1];
    if (
        !isfinite(draw0) || !isfinite(draw1) || draw0 < 0.0 || draw0 >= 1.0
        || draw1 < 0.0 || draw1 >= 1.0
    ) {
        return false;
    }
    const std::int64_t operation = static_cast<std::int64_t>(draw0 * 4.0);
    // This qualification ABI intentionally supports one exact Python operator.
    if (operation != 2) {
        return false;
    }

    const bool* const module_occupied = reinterpret_cast<const bool*>(
        lifecycle_row_v2(context, kModuleOccupiedV1, shadow, cell, child_slot)
    );
    const std::int8_t* const module_type = reinterpret_cast<const std::int8_t*>(
        lifecycle_row_v2(context, kModuleTypeCodeV1, shadow, cell, child_slot)
    );
    const std::int8_t* const operator_code = reinterpret_cast<const std::int8_t*>(
        lifecycle_row_v2(context, kOperatorCodeV1, shadow, cell, child_slot)
    );
    const bool* const duplicate = (
        shadow ? context.duplicate_descendant_mask_shadow
               : context.duplicate_descendant_mask_source
    )
        + (cell * context.capacity + child_slot) * kModuleSlotsV1;
    std::int64_t eligible_count = 0;
    for (std::int64_t module = 0; module < kModuleSlotsV1; ++module) {
        if (
            module_occupied[module] && module_type[module] == 1
            && duplicate[module] && (operator_code[module] == 1 || operator_code[module] == 2)
        ) {
            ++eligible_count;
        }
    }

    std::int64_t status = 3;
    std::int64_t reason = 2;
    std::int64_t selected_module = -1;
    std::int64_t selected_instance = -1;
    std::int64_t selected_homology = -1;
    std::int64_t changed_dimension = 0;
    if (eligible_count > 0) {
        const double module_scaled = draw1 * static_cast<double>(eligible_count);
        const std::int64_t selected_ordinal = static_cast<std::int64_t>(module_scaled);
        const double module_fraction =
            module_scaled - static_cast<double>(selected_ordinal);
        std::int64_t ordinal = 0;
        for (std::int64_t module = 0; module < kModuleSlotsV1; ++module) {
            if (
                !module_occupied[module] || module_type[module] != 1
                || !duplicate[module]
                || (operator_code[module] != 1 && operator_code[module] != 2)
            ) {
                continue;
            }
            if (ordinal++ == selected_ordinal) {
                selected_module = module;
                break;
            }
        }
        if (
            selected_module < 0
            || (operator_code[selected_module] != 1
                && operator_code[selected_module] != 2)
        ) {
            return false;
        }
        const bool gated = operator_code[selected_module] == 2;
        const std::int64_t field_count = gated ? 10 : 4;
        const double field_scaled =
            module_fraction * static_cast<double>(field_count);
        std::int64_t field = static_cast<std::int64_t>(field_scaled);
        if (field >= field_count) field = field_count - 1;
        const double value_fraction = field_scaled - static_cast<double>(field);
        const double lower = gated && field != 9 ? -4.0
            : gated ? -2.0
            : field == 1 ? 0.0 : -2.0;
        const double upper = gated && field != 9 ? 4.0
            : gated ? 2.0
            : field == 1 ? 1.0 : 2.0;
        float* const parameters = reinterpret_cast<float*>(lifecycle_row_v2(
            context,
            gated ? kGatedParametersV1 : kIntegratorParametersV1,
                shadow,
            cell,
            child_slot
        ));
        float* const value =
            parameters + selected_module * field_count + field;
        const double replacement =
            lower + value_fraction * (upper - lower);
        if (replacement == static_cast<double>(*value)) {
            status = 4;
            reason = 4;
            selected_module = -1;
        } else {
            *value = static_cast<float>(replacement);
            const std::int32_t* const instance_indexes =
                reinterpret_cast<const std::int32_t*>(lifecycle_row_v2(
                    context, kModuleInstanceIdIndexV1, shadow, cell, child_slot
                ));
            const std::int32_t* const homology_indexes =
                reinterpret_cast<const std::int32_t*>(lifecycle_row_v2(
                    context, kModuleHomologyIdIndexV1, shadow, cell, child_slot
                ));
            selected_instance = instance_indexes[selected_module];
            selected_homology = homology_indexes[selected_module];
            if (selected_instance < 0 || selected_homology < 0) {
                return false;
            }
            status = 1;
            reason = 1;
            changed_dimension = gated ? field + 5 : field + 1;
        }
    }

    std::int64_t* const receipt = context.turnover_mutation_receipt_rows
        + (cell * kTurnoverMutationSequenceCapacityV1 + sequence_index)
            * kTurnoverMutationReceiptWidthV1;
    receipt[0] = 1;
    receipt[1] = sequence_index;
    receipt[2] = commit_generation;
    receipt[3] = commit_tick;
    receipt[4] = parent_agent_id;
    receipt[5] = parent_slot;
    receipt[6] = *reinterpret_cast<const std::int64_t*>(lifecycle_row_v2(
        context,
        kAgentSlotTokenV1,
        ordinary_birth ? false : shadow,
        cell,
        parent_slot
    ));
    receipt[7] = child_agent_id;
    receipt[8] = child_slot;
    receipt[9] = child_slot_token;
    receipt[10] = transaction_ordinal;
    receipt[11] = static_cast<std::int64_t>(__double_as_longlong(draw0));
    receipt[12] = static_cast<std::int64_t>(__double_as_longlong(draw1));
    receipt[13] = operation;
    receipt[14] = status;
    receipt[15] = reason;
    receipt[16] = selected_module;
    receipt[17] = selected_instance;
    receipt[18] = selected_homology;
    receipt[19] = changed_dimension;
    receipt[20] = 1;
    if (ordinary_birth) {
        if (!write_ordinary_birth_rowv1_receipt_v3(
                context,
                cell,
                sequence_index,
                commit_generation,
                commit_tick,
                parent_agent_id,
                parent_slot,
                receipt[6],
                child_agent_id,
                child_slot,
                child_slot_token,
                transaction_ordinal,
                receipt[11],
                receipt[12],
                operation,
                status,
                reason,
                selected_module,
                selected_instance,
                selected_homology,
                changed_dimension,
                receipt[20],
                kOrdinaryBirthRowV1DisabledReasonAbsent
            )) {
            return false;
        }
    }
    context.turnover_mutation_receipt_counts[cell] = sequence_index + 1;
    return true;
}

__device__ __forceinline__ bool mutate_turnover_descendant_parameter_v1(
    const FlatModularContextV1 context,
    const std::int64_t cell,
    const std::int64_t transaction_ordinal,
    const std::int64_t parent_slot,
    const std::int64_t child_slot,
    const std::int64_t parent_agent_id,
    const std::int64_t child_agent_id,
    const std::int64_t child_slot_token,
    const std::int64_t commit_generation,
    const std::int64_t commit_tick
) {
    return mutate_descendant_parameter_v1(
        context,
        true,
        false,
        cell,
        transaction_ordinal,
        parent_slot,
        child_slot,
        parent_agent_id,
        child_agent_id,
        child_slot_token,
        commit_generation,
        commit_tick
    );
}

__device__ __forceinline__ bool mutate_ordinary_birth_descendant_parameter_v1(
    const FlatModularContextV1 context,
    const std::int64_t cell,
    const std::int64_t transaction_ordinal,
    const std::int64_t parent_slot,
    const std::int64_t child_slot,
    const std::int64_t parent_agent_id,
    const std::int64_t child_agent_id,
    const std::int64_t child_slot_token,
    const std::int64_t commit_generation,
    const std::int64_t commit_tick
) {
    return mutate_descendant_parameter_v1(
        context,
        true,
        true,
        cell,
        transaction_ordinal,
        parent_slot,
        child_slot,
        parent_agent_id,
        child_agent_id,
        child_slot_token,
        commit_generation,
        commit_tick
    );
}

__device__ __forceinline__ float clamp_f32(
    const float value,
    const float low,
    const float high
) {
    return value < low ? low : (value > high ? high : value);
}

__device__ __forceinline__ float stable_sigmoid_f32(const float value) {
    const float clipped = clamp_f32(value, -16.0F, 16.0F);
    if (clipped >= 0.0F) {
        const float exponent = expf(-clipped);
        return 1.0F / (1.0F + exponent);
    }
    const float exponent = expf(clipped);
    return exponent / (1.0F + exponent);
}

__device__ __noinline__ bool action_residuals_exact_v1(
    const FlatModularContextV1 context,
    const std::int64_t cell,
    const std::int64_t slot,
    const std::int64_t expected_agent_id,
    const std::int64_t expected_slot_token,
    const float self_energy,
    const float self_health,
    const float fatigue,
    const std::int32_t survival_steps,
    float* const residuals
) {
    constexpr std::int64_t modules = kModuleSlotsV1;
    constexpr std::int64_t connections = kConnectionSlotsV1;
    for (std::int64_t action = 0; action < kActionWidthV1; ++action) {
        residuals[action] = 0.0F;
    }
    if (!context.enabled) {
        return true;
    }
    if (
        cell < 0 || cell >= context.cells || slot < 0 || slot >= context.capacity
        || context.agent_occupied == nullptr || context.agent_id == nullptr
        || context.agent_slot_token == nullptr || context.module_occupied == nullptr
        || context.module_homology_id_index == nullptr
        || context.module_type_code == nullptr
        || context.module_execution_order == nullptr
        || context.dependency_topological_order == nullptr
        || context.operator_code == nullptr
        || context.integrator_parameters == nullptr
        || context.gated_parameters == nullptr || context.module_dormant == nullptr
        || context.module_state_present == nullptr || context.module_state == nullptr
        || context.module_output_present == nullptr
        || context.module_previous_output == nullptr
        || context.input_binding_count == nullptr
        || context.input_semantic_code == nullptr || context.input_gain == nullptr
        || context.input_enabled == nullptr || context.output_binding_count == nullptr
        || context.output_action_index == nullptr || context.output_gain == nullptr
        || context.output_enabled == nullptr
        || context.development_stage_gains == nullptr
        || context.development_prerequisite_count == nullptr
        || context.development_prerequisite_homology_index == nullptr
        || context.development_prerequisite_module_slot == nullptr
        || context.development_latch_enabled == nullptr
        || context.development_latch_present == nullptr
        || context.development_latch_state == nullptr
        || context.development_effective_gain_present == nullptr
        || context.development_effective_gain == nullptr
        || context.connection_occupied == nullptr
        || context.connection_src_module_slot == nullptr
        || context.connection_dst_module_slot == nullptr
        || context.connection_weight == nullptr
        || context.connection_execution_order == nullptr
        || !isfinite(self_energy) || !isfinite(self_health) || !isfinite(fatigue)
        || survival_steps < 0
    ) {
        return false;
    }

    const std::int64_t row = cell * context.capacity + slot;
    if (
        !context.agent_occupied[row] || context.agent_id[row] != expected_agent_id
        || context.agent_slot_token[row] != expected_slot_token
    ) {
        return false;
    }
    const std::int64_t module_base = row * modules;
    const std::int64_t connection_base = row * connections;
    bool generic[modules];
    bool development_seen[modules]{};
    bool execution_seen[modules]{};
    float effective_gain[modules]{};
    float routed_observation[modules]{};
    float routed_previous[modules]{};
    float old_state[modules]{};
    float previous_output[modules]{};
    float next_state[modules]{};
    float next_output[modules]{};

    for (std::int64_t module = 0; module < modules; ++module) {
        const std::int64_t index = module_base + module;
        generic[module] = context.module_occupied[index]
            && context.module_type_code[index] == 1;
        if (
            context.module_occupied[index] && !generic[module]
            || (generic[module] && context.operator_code[index] != 1
                && context.operator_code[index] != 2)
        ) {
            return false;
        }
        old_state[module] = context.module_state_present[index]
            ? context.module_state[index] : 0.0F;
        previous_output[module] = context.module_previous_output[index];
        next_state[module] = context.module_state[index];
        next_output[module] = previous_output[module];
    }

    const std::int32_t stage = survival_steps >= context.late_start_step
        ? 2 : (survival_steps >= context.mature_start_step ? 1 : 0);
    for (std::int64_t order = 0; order < modules; ++order) {
        const std::int32_t module = context.dependency_topological_order[
            module_base + order
        ];
        if (module < 0) {
            continue;
        }
        if (module >= modules || !generic[module] || development_seen[module]) {
            return false;
        }
        development_seen[module] = true;
        const std::int64_t index = module_base + module;
        const std::int8_t prerequisite_count =
            context.development_prerequisite_count[index];
        if (prerequisite_count < 0 || prerequisite_count > 2) {
            return false;
        }
        bool ready = true;
        for (std::int64_t prerequisite = 0; prerequisite < prerequisite_count;
             ++prerequisite) {
            const std::int32_t source =
                context.development_prerequisite_module_slot[index * 2 + prerequisite];
            const std::int32_t homology =
                context.development_prerequisite_homology_index[
                    index * 2 + prerequisite
                ];
            if (
                source < 0 || source >= modules || homology < 0
                || context.module_homology_id_index[module_base + source] != homology
            ) {
                return false;
            }
            if (effective_gain[source] <= 0.0F) {
                ready = false;
            }
        }
        const bool previously_latched = context.development_latch_present[index]
            && context.development_latch_state[index];
        const bool gate_ready = ready || previously_latched;
        const bool next_latched = context.development_latch_enabled[index]
            && (previously_latched || ready);
        const float requested = context.development_stage_gains[index * 3 + stage];
        effective_gain[module] = gate_ready ? requested : 0.0F;
        context.development_latch_present[index] = next_latched;
        context.development_latch_state[index] = next_latched;
        context.development_effective_gain_present[index] = true;
        context.development_effective_gain[index] = effective_gain[module];
    }
    for (std::int64_t module = 0; module < modules; ++module) {
        if (development_seen[module] != generic[module]) {
            return false;
        }
        if (!generic[module]) {
            continue;
        }
        const std::int64_t index = module_base + module;
        const std::int8_t count = context.input_binding_count[index];
        if (count < 0 || count > 2) {
            return false;
        }
        float routed = 0.0F;
        for (std::int64_t binding = 0; binding < count; ++binding) {
            const std::int64_t binding_index = index * 2 + binding;
            if (!context.input_enabled[binding_index]) {
                continue;
            }
            const std::int8_t semantic = context.input_semantic_code[binding_index];
            if (semantic < 0 || semantic > 1) {
                return false;
            }
            const float source = semantic == 0 ? self_energy : self_health;
            routed += context.input_gain[binding_index] * source;
        }
        routed_observation[module] = clamp_f32(routed, -1.0F, 1.0F);
    }

    bool connection_seen[connections]{};
    for (std::int64_t order = 0; order < connections; ++order) {
        std::int32_t selected = -1;
        for (std::int64_t connection = 0; connection < connections; ++connection) {
            const std::int64_t index = connection_base + connection;
            if (
                context.connection_occupied[index]
                && context.connection_execution_order[index] == order
            ) {
                if (selected >= 0) {
                    return false;
                }
                selected = static_cast<std::int32_t>(connection);
            }
        }
        if (selected < 0) {
            continue;
        }
        connection_seen[selected] = true;
        const std::int64_t index = connection_base + selected;
        const std::int32_t source = context.connection_src_module_slot[index];
        const std::int32_t destination = context.connection_dst_module_slot[index];
        if (source < 0 || source >= modules || destination < 0 || destination >= modules) {
            return false;
        }
        const bool source_active = generic[source]
            && !context.module_dormant[module_base + source];
        const bool destination_active = generic[destination]
            && !context.module_dormant[module_base + destination]
            && effective_gain[destination] > 0.0F;
        if (source_active && destination_active) {
            const float source_value = effective_gain[source] > 0.0F
                ? previous_output[source] : 0.0F;
            routed_previous[destination] +=
                context.connection_weight[index] * source_value;
        }
    }
    for (std::int64_t connection = 0; connection < connections; ++connection) {
        if (
            connection_seen[connection]
            != context.connection_occupied[connection_base + connection]
        ) {
            return false;
        }
    }

    for (std::int64_t order = 0; order < modules; ++order) {
        const std::int32_t module = context.module_execution_order[module_base + order];
        if (module < 0) {
            continue;
        }
        if (module >= modules || !generic[module] || execution_seen[module]) {
            return false;
        }
        execution_seen[module] = true;
        const std::int64_t index = module_base + module;
        const float gain = effective_gain[module];
        const bool executing = !context.module_dormant[index] && gain > 0.0F;
        const float external = routed_observation[module] == 0.0F
            ? fatigue : fatigue + routed_observation[module];
        float state = old_state[module];
        float output = 0.0F;
        if (executing && context.operator_code[index] == 1) {
            const float* const parameters = context.integrator_parameters + index * 4;
            state = parameters[1] * old_state[module];
            state += parameters[2] * external;
            state = clamp_f32(state + routed_previous[module], -1.0F, 1.0F);
            output = parameters[0] * parameters[3];
            output = clamp_f32(output * state, -1.0F, 1.0F);
            if (gain != 1.0F) {
                output = clamp_f32(gain * output, -1.0F, 1.0F);
            }
        } else if (executing && context.operator_code[index] == 2) {
            const float* const p = context.gated_parameters + index * 10;
            const float aggregate = clamp_f32(
                external + routed_previous[module], -1.0F, 1.0F
            );
            const float reset_pre = clamp_f32(
                p[0] + p[1] * aggregate + p[2] * old_state[module],
                -16.0F,
                16.0F
            );
            const float reset = stable_sigmoid_f32(reset_pre);
            const float update_pre = clamp_f32(
                p[3] + p[4] * aggregate + p[5] * old_state[module],
                -16.0F,
                16.0F
            );
            const float update = stable_sigmoid_f32(update_pre);
            const float candidate_pre = clamp_f32(
                p[6] + p[7] * aggregate + p[8] * (reset * old_state[module]),
                -16.0F,
                16.0F
            );
            const float candidate = tanhf(candidate_pre);
            state = clamp_f32(
                (1.0F - update) * old_state[module] + update * candidate,
                -1.0F,
                1.0F
            );
            output = clamp_f32(p[9] * state, -1.0F, 1.0F);
            output = clamp_f32(
                context.integrator_parameters[index * 4] * output,
                -1.0F,
                1.0F
            );
            if (gain != 1.0F) {
                output = clamp_f32(gain * output, -1.0F, 1.0F);
            }
        }
        next_state[module] = state;
        next_output[module] = output;
    }
    for (std::int64_t module = 0; module < modules; ++module) {
        if (execution_seen[module] != generic[module]) {
            return false;
        }
        if (generic[module]) {
            const std::int64_t index = module_base + module;
            context.module_state_present[index] = true;
            context.module_output_present[index] = true;
            context.module_state[index] = next_state[module];
            context.module_previous_output[index] = next_output[module];
        }
    }

    float legacy_rest = 0.0F;
    for (std::int64_t order = 0; order < modules; ++order) {
        const std::int32_t module = context.module_execution_order[module_base + order];
        if (module < 0) {
            continue;
        }
        const std::int64_t index = module_base + module;
        legacy_rest += next_output[module];
        const std::int8_t count = context.output_binding_count[index];
        if (count < 0 || count > 2) {
            return false;
        }
        for (std::int64_t binding = 0; binding < count; ++binding) {
            const std::int64_t binding_index = index * 2 + binding;
            if (!context.output_enabled[binding_index]) {
                continue;
            }
            const std::int8_t action = context.output_action_index[binding_index];
            if (action < 0 || action >= kActionWidthV1) {
                return false;
            }
            residuals[action] +=
                context.output_gain[binding_index] * next_output[module];
        }
    }
    legacy_rest = clamp_f32(legacy_rest, -0.25F, 0.25F);
    for (std::int64_t action = 0; action < kActionWidthV1; ++action) {
        residuals[action] = clamp_f32(residuals[action], -0.25F, 0.25F);
    }
    residuals[5] = residuals[5] == 0.0F
        ? legacy_rest
        : clamp_f32(legacy_rest + residuals[5], -0.25F, 0.25F);
    return true;
}

}  // namespace device
}  // namespace eiel::cuda_architecture_b::flattened_modular_controller_v1

#endif
