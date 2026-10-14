#include "modular_recurrent_cpu_oracle_c_api_v1.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

namespace {

void clear_row(eiel_modular_recurrent_row_v1* const row) {
    std::memset(row, 0, sizeof(*row));
    row->agent_id = -1;
    row->agent_slot_token = -1;
    for (std::int32_t slot = 0; slot < 16; ++slot) {
        row->module_local_id_index[slot] = -1;
        row->module_instance_id_index[slot] = -1;
        row->module_homology_id_index[slot] = -1;
        row->module_type_index[slot] = -1;
        row->module_execution_order[slot] = -1;
        row->dependency_topological_order[slot] = -1;
        for (std::int32_t binding = 0; binding < 2; ++binding) {
            const std::int32_t index = slot * 2 + binding;
            row->input_semantic_code[index] = -1;
            row->input_semantic_id_index[index] = -1;
            row->output_semantic_code[index] = -1;
            row->output_semantic_id_index[index] = -1;
            row->output_action_index[index] = -1;
            row->development_prerequisite_homology_index[index] = -1;
            row->development_prerequisite_module_slot[index] = -1;
        }
    }
    for (std::int32_t slot = 0; slot < 48; ++slot) {
        row->connection_local_id_index[slot] = -1;
        row->connection_src_module_slot[slot] = -1;
        row->connection_dst_module_slot[slot] = -1;
        row->connection_src_slot_index[slot] = -1;
        row->connection_dst_slot_index[slot] = -1;
        row->connection_value_type_index[slot] = -1;
        row->connection_channel_index[slot] = -1;
        row->connection_execution_order[slot] = -1;
    }
}

eiel_modular_recurrent_row_v1 fixture_row() {
    eiel_modular_recurrent_row_v1 row{};
    clear_row(&row);
    row.agent_occupied = 1U;
    row.agent_id = 41;
    row.agent_slot_token = 9001;
    row.module_count = 1;
    row.module_occupied[0] = 1U;
    row.module_local_id_index[0] = 10;
    row.module_instance_id_index[0] = 20;
    row.module_homology_id_index[0] = 30;
    row.module_type_index[0] = 40;
    row.module_type_code[0] = 1;
    row.identity_origin_code[0] = 2;
    row.module_execution_order[0] = 0;
    row.dependency_topological_order[0] = 0;
    row.operator_code[0] = 1;
    row.integrator_parameters[0] = 0.75F;
    row.integrator_parameters[1] = 0.5F;
    row.integrator_parameters[2] = 0.75F;
    row.integrator_parameters[3] = 0.5F;
    row.module_state_present[0] = 1U;
    row.module_state[0] = 0.2F;
    row.module_output_present[0] = 1U;
    row.module_previous_output[0] = 0.1F;
    row.input_binding_count[0] = 1;
    row.input_semantic_code[0] = 0;
    row.input_semantic_id_index[0] = 50;
    row.input_gain[0] = 1.0F;
    row.input_enabled[0] = 1U;
    row.output_binding_count[0] = 1;
    row.output_semantic_code[0] = 4;
    row.output_semantic_id_index[0] = 60;
    row.output_action_index[0] = 5;
    row.output_gain[0] = 1.0F;
    row.output_enabled[0] = 1U;
    row.development_explicit[0] = 1U;
    row.development_stage_gains[0] = 1.0F;
    row.development_stage_gains[1] = 1.0F;
    row.development_stage_gains[2] = 1.0F;
    row.structural_cost_counts[0] = 1;
    row.structural_cost_counts[1] = 1;
    row.structural_cost_counts[4] = 1;
    row.structural_cost_counts[5] = 1;
    row.structural_cost_components[0] = 1.0;
    row.structural_cost_components[1] = 4.0;
    row.structural_cost_components[4] = 1.0;
    row.structural_cost_components[5] = 2.0;
    row.structural_cost_total = 8.0;
    return row;
}

}  // namespace

int main() {
    if (eiel_modular_recurrent_cpu_oracle_abi_version_v1() != 1U
        || eiel_modular_recurrent_cpu_oracle_plane_count_v1() != 57
        || eiel_modular_recurrent_cpu_oracle_row_size_v1()
            != static_cast<std::int64_t>(sizeof(eiel_modular_recurrent_row_v1))) {
        return 1;
    }
    std::array<eiel_modular_recurrent_row_v1, 2> rows{};
    rows[0] = fixture_row();
    clear_row(&rows[1]);
    std::array<std::int64_t, 2> ids{41, -1};
    std::array<std::int64_t, 2> tokens{9001, 7002};
    std::array<std::uint8_t, 2> alive{1U, 0U};
    eiel_modular_recurrent_owner_view_v1 owners{
        ids.data(), tokens.data(), alive.data(), 2};
    eiel_modular_recurrent_const_owner_view_v1 const_owners{
        ids.data(), tokens.data(), alive.data(), 2};
    if (eiel_modular_recurrent_validate_v1(rows.data(), 2, 1, &const_owners) != 0) {
        return 2;
    }
    std::array<float, 54> observations{};
    observations[25] = 0.8F;
    observations[26] = 0.6F;
    std::array<float, 2> fatigue{0.1F, 0.0F};
    std::array<std::int64_t, 2> stages{0, 0};
    std::array<eiel_modular_recurrent_row_v1, 2> scratch{};
    std::array<float, 14> scratch_actions{};
    std::array<float, 14> actions{};
    eiel_modular_recurrent_tick_receipt_v1 receipt{};
    if (eiel_modular_recurrent_tick_v1(
            rows.data(), 2, 1, &const_owners, observations.data(), 54,
            fatigue.data(), 2, stages.data(), 2, scratch.data(), 2,
            scratch_actions.data(), 14, actions.data(), 14, &receipt) != 0
        || receipt.committed_rows != 2 || actions[5] == 0.0F
        || rows[0].module_state[0] == 0.2F) {
        return 3;
    }
    const auto before = rows;
    observations[0] = std::numeric_limits<float>::quiet_NaN();
    actions.fill(91.0F);
    if (eiel_modular_recurrent_tick_v1(
            rows.data(), 2, 1, &const_owners, observations.data(), 54,
            fatigue.data(), 2, stages.data(), 2, scratch.data(), 2,
            scratch_actions.data(), 14, actions.data(), 14, &receipt) != 7
        || std::memcmp(rows.data(), before.data(), sizeof(rows)) != 0
        || actions[0] != 91.0F) {
        return 4;
    }
    eiel_modular_recurrent_row_v1 checkpoint{};
    if (eiel_modular_recurrent_checkpoint_project_v1(
            rows.data(), 2, 1, &const_owners, 0, 9001, &checkpoint) != 0) {
        return 5;
    }
    if (eiel_modular_recurrent_death_cleanup_v1(
            rows.data(), 2, 1, &owners, 0, 41, 9001) != 0
        || alive[0] != 0U || ids[0] != -1) {
        return 6;
    }
    if (eiel_modular_recurrent_checkpoint_restore_v1(
            &checkpoint, rows.data(), 2, 1, &owners, 0, 9001) != 0
        || std::memcmp(&checkpoint, &rows[0], sizeof(checkpoint)) != 0) {
        return 7;
    }
    eiel_modular_recurrent_cost_capacity_receipt_v1 cost{};
    if (eiel_modular_recurrent_cost_capacity_v1(&rows[0], 1, &cost) != 0
        || cost.total != 8.0 || cost.occupied_modules != 1
        || cost.logical_module_slots != 16 || cost.reporting_only != 1U) {
        return 8;
    }
    return 0;
}
