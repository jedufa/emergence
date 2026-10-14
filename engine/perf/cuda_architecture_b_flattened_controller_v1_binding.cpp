#include "cuda_architecture_b_flattened_controller_v1.hpp"
#include "cuda_architecture_b_flattened_birth_scratch_v2.cuh"
#include "cuda_architecture_b_resident_event_archive_v1.hpp"
#include "cuda_architecture_b_resident_run_control_v2.hpp"
#include "cuda_architecture_b_resident_run_control_v2_binding.hpp"
#include "cuda_architecture_b_resident_window_controller_v1.hpp"
#include "cuda_architecture_b_torch_rng_state_pack_v1.cuh"
#include "native/cuda_architecture_b_flattened_cpu_policy_service_v1.hpp"
#ifdef EIEL_WDL_JUNGLE_EED_HAPB_FIXED_MLP_NATIVE_BINDING_V1
#include "cuda_architecture_b_hapb_policy_context_v1.hpp"
#endif

#ifndef EIEL_CUDA_STAGE3_PARALLEL_PATCH_POLICY_ROWS
#define EIEL_CUDA_STAGE3_PARALLEL_PATCH_POLICY_ROWS 0
#endif
#ifndef EIEL_TW_LIFECYCLE_LINEAR_FAST_PATH_ENABLED
#define EIEL_TW_LIFECYCLE_LINEAR_FAST_PATH_ENABLED 0
#endif
#ifndef EIEL_TW_LIFECYCLE_LINEAR_FAST_PATH_TEST_SURFACE_ENABLED
#define EIEL_TW_LIFECYCLE_LINEAR_FAST_PATH_TEST_SURFACE_ENABLED 0
#endif
#include "cuda_architecture_b_resident_window_refill_v1.hpp"
#include "cuda_architecture_b_compact_summary_checkpoint_hash_v1.cuh"
#include "cuda_architecture_b_compact_summary_checkpoint_runtime_v1.hpp"
#include "cuda_architecture_b_compact_summary_host_settlement_v1.hpp"
#include "cuda_architecture_b_compact_summary_host_template_runtime_v1.hpp"
#include "cuda_architecture_b_compact_summary_lifecycle_digest_v1.cuh"
#include "cuda_architecture_b_compact_summary_metadata_retention_v1.cuh"
#include "cuda_architecture_b_compact_summary_reductions_v1.cuh"
#include "cuda_architecture_b_compact_summary_settlement_v1.hpp"
#include "cuda_architecture_b_compact_summary_v2.cuh"
#include "genome_upgrade_v2_q10_v1.cuh"
#include "../tw_rngv2/native/tw_rngv2.cuh"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#ifndef EIEL_CUDA_ARCHITECTURE_B_T160_PARTIAL_SPAN_ALLOWED
#define EIEL_CUDA_ARCHITECTURE_B_T160_PARTIAL_SPAN_ALLOWED false
#endif

#include "../native_runtime/authoritative_state/binary_identity_constants_v1.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace {

namespace compact_summary = eiel::cuda_architecture_b::compact_summary_v2;
namespace checkpoint_hash =
    eiel::cuda_architecture_b::compact_summary_checkpoint_hash_v1;
namespace checkpoint_runtime =
    eiel::cuda_architecture_b::compact_summary_checkpoint_runtime_v1;
namespace lifecycle_digest =
    eiel::cuda_architecture_b::compact_summary_lifecycle_digest_v1;
namespace metadata_retention =
    eiel::cuda_architecture_b::compact_summary_metadata_retention_v1;
namespace reductions =
    eiel::cuda_architecture_b::compact_summary_reductions_v1;
namespace host_settlement =
    eiel::cuda_architecture_b::compact_summary_host_settlement_v1;
namespace host_template_runtime =
    eiel::cuda_architecture_b::compact_summary_host_template_runtime_v1;
namespace settlement =
    eiel::cuda_architecture_b::compact_summary_settlement_v1;
namespace projection =
    eiel::cuda_architecture_b::compact_summary_source_projection_v1;
namespace reporting_authority =
    eiel::cuda_architecture_b::compact_summary_reporting_authority_v1;
namespace controller = eiel::cuda_architecture_b::flattened_controller_v1;
namespace turnover = eiel::cuda_architecture_b::flattened_turnover_v1;
namespace torch_rng_pack =
    eiel::cuda_architecture_b::torch_rng_state_pack_v1;
namespace genome_v2_causal_output =
    eiel::cuda_architecture_b::genome_v2_causal_output_v1;
namespace resident_window_refill =
    eiel::cuda_architecture_b::resident_window_refill_v1;
using resident_window_refill::ScopedPolicyWindowIdentityStateV1;
using resident_window_refill::active_policy_window_identity_state_v1;
namespace resident_event_archive =
    eiel::cuda_architecture_b::resident_event_archive_v1;
namespace resident_window_controller =
    eiel::cuda_architecture_b::resident_window_controller_v1;
namespace resident_run_control =
    eiel::cuda_architecture_b::resident_run_control_v2;
namespace resident_run_control_binding =
    eiel::cuda_architecture_b::resident_run_control_v2_binding;
#ifdef EIEL_WDL_JUNGLE_EED_HAPB_FIXED_MLP_NATIVE_BINDING_V1
namespace hapb_policy_context =
    eiel::cuda_architecture_b::hapb_policy_context_v1;
#endif

bool option_d_m6_birth_diagnostics_enabled_v1() {
    const char* const value = std::getenv(
        "EIEL_OPTION_D_M6_BIRTH_DIAGNOSTICS"
    );
    return value != nullptr
        && std::strcmp(value, "") != 0
        && std::strcmp(value, "0") != 0
        && std::strcmp(value, "false") != 0
        && std::strcmp(value, "FALSE") != 0;
}

pybind11::list int64_vector_to_pylist_v1(
    const std::vector<std::int64_t>& values,
    const std::size_t offset,
    const std::size_t count
) {
    pybind11::list out;
    for (std::size_t index = 0; index < count; ++index) {
        out.append(values.at(offset + index));
    }
    return out;
}

pybind11::list int32_vector_to_pylist_v1(
    const std::vector<std::int32_t>& values,
    const std::size_t offset,
    const std::size_t count
) {
    pybind11::list out;
    for (std::size_t index = 0; index < count; ++index) {
        out.append(values.at(offset + index));
    }
    return out;
}

std::vector<std::int64_t> copy_int64_tensor_host_v1(
    const torch::Tensor& tensor
) {
    std::vector<std::int64_t> values(
        static_cast<std::size_t>(tensor.numel())
    );
    if (!values.empty()) {
        C10_CUDA_CHECK(cudaMemcpy(
            values.data(),
            tensor.data_ptr<std::int64_t>(),
            values.size() * sizeof(std::int64_t),
            cudaMemcpyDeviceToHost
        ));
    }
    return values;
}

std::vector<std::int32_t> copy_int32_tensor_host_v1(
    const torch::Tensor& tensor
) {
    std::vector<std::int32_t> values(
        static_cast<std::size_t>(tensor.numel())
    );
    if (!values.empty()) {
        C10_CUDA_CHECK(cudaMemcpy(
            values.data(),
            tensor.data_ptr<std::int32_t>(),
            values.size() * sizeof(std::int32_t),
            cudaMemcpyDeviceToHost
        ));
    }
    return values;
}

std::int64_t sum_int64_vector_v1(const std::vector<std::int64_t>& values) {
    std::int64_t total = 0;
    for (const std::int64_t value : values) {
        total += value;
    }
    return total;
}
namespace exact_cpu_policy =
    eiel::cuda_architecture_b::flattened_cpu_policy_service_v1;
namespace genome_v2 = eiel::genome_upgrade_v2_q10_v1;

static_assert(
    resident_window_refill::kOptionDM0TelemetryEnabledV1
        == exact_cpu_policy::kOptionDM0TelemetryEnabledV1,
    "Option D M0 telemetry must be consistently enabled across native owners"
);

#ifndef EIEL_CUDA_ARCHITECTURE_B_EXACT_CPU_POLICY_CONTEXT_ALLOWED
#define EIEL_CUDA_ARCHITECTURE_B_EXACT_CPU_POLICY_CONTEXT_ALLOWED true
#endif

#ifndef EIEL_OPTION_D_M9_BIRTH_PREPARE_COMMIT_FUSED_ENABLED
#define EIEL_OPTION_D_M9_BIRTH_PREPARE_COMMIT_FUSED_ENABLED 0
#endif

thread_local std::shared_ptr<resident_run_control::RunControlV2>
    active_direct_run_control_v2;

struct CrossPathInvocationContextV1 {
    std::vector<torch::Tensor> tensors;
    std::string registry_sha256;
};

thread_local const CrossPathInvocationContextV1*
    active_cross_path_invocation_v1 = nullptr;

eiel::cuda_architecture_b::cross_path_resident_control_v1::
    FlatCrossPathResidentControlV1
cross_path_resident_control_v1() {
    using Control = eiel::cuda_architecture_b::cross_path_resident_control_v1::
        FlatCrossPathResidentControlV1;
    if (active_cross_path_invocation_v1 == nullptr) {
        return Control{};
    }
    const auto& value = active_cross_path_invocation_v1->tensors;
    TORCH_CHECK(
        value.size() == 65,
        "cross-path invocation requires exactly 65 resident tensors"
    );
    const int64_t rows_per_cell = value.at(22).size(1);
    return Control{
        value.at(3).data_ptr<int8_t>(),
        value.at(4).data_ptr<int8_t>(),
        value.at(5).data_ptr<int8_t>(),
        value.at(9).data_ptr<bool>(),
        value.at(10).data_ptr<bool>(),
        value.at(13).data_ptr<bool>(),
        value.at(17).data_ptr<uint64_t>(),
        value.at(18).data_ptr<double>(),
        value.at(20).data_ptr<int64_t>(),
        value.at(21).data_ptr<int64_t>(),
        value.at(21).size(1),
        rows_per_cell,
        value.at(22).data_ptr<int64_t>(),
        value.at(23).data_ptr<int64_t>(),
        value.at(24).data_ptr<int64_t>(),
        value.at(25).data_ptr<uint8_t>(),
        value.at(26).data_ptr<uint8_t>(),
        value.at(27).data_ptr<int64_t>(),
        value.at(28).data_ptr<int64_t>(),
        value.at(29).data_ptr<int64_t>(),
        value.at(30).data_ptr<int32_t>(),
        {
            value.at(31).data_ptr<bool>(),
            value.at(32).data_ptr<int8_t>(),
            value.at(33).data_ptr<int8_t>(),
            value.at(34).data_ptr<int64_t>(),
            value.at(35).data_ptr<int64_t>(),
            value.at(36).data_ptr<int64_t>(),
            value.at(37).data_ptr<int64_t>(),
            value.at(38).data_ptr<int64_t>(),
            value.at(39).data_ptr<bool>(),
            value.at(40).data_ptr<int32_t>(),
            value.at(50).data_ptr<int64_t>(),
            value.at(41).data_ptr<int8_t>(),
            value.at(42).data_ptr<int64_t>(),
            value.at(43).data_ptr<int64_t>(),
            value.at(44).data_ptr<int64_t>(),
            value.at(45).data_ptr<int64_t>(),
            value.at(46).data_ptr<bool>(),
            value.at(47).data_ptr<int64_t>(),
            value.at(48).data_ptr<int64_t>(),
            value.at(49).data_ptr<int64_t>(),
            value.at(51).data_ptr<int64_t>(),
            value.at(52).data_ptr<int64_t>(),
            value.at(53).data_ptr<int32_t>(),
            value.at(54).data_ptr<int64_t>(),
            value.at(55).data_ptr<int64_t>(),
            value.at(56).data_ptr<uint8_t>(),
            value.at(57).data_ptr<int64_t>(),
            value.at(58).data_ptr<bool>(),
            value.at(59).data_ptr<int64_t>(),
            value.at(60).data_ptr<int8_t>(),
            value.at(61).data_ptr<int64_t>(),
            value.at(62).data_ptr<int64_t>(),
            value.at(63).data_ptr<int64_t>(),
            value.at(64).data_ptr<int64_t>(),
            value.at(33).size(1),
            value.at(41).size(1),
            value.at(45).size(2),
            value.at(46).size(1),
            value.at(60).size(0),
        },
    };
}

torch::Tensor cross_path_exact_cpu_policy_bypass_v1() {
    if (active_cross_path_invocation_v1 == nullptr) {
        return torch::Tensor();
    }
    const auto& value = active_cross_path_invocation_v1->tensors;
    TORCH_CHECK(
        value.size() == 65,
        "cross-path invocation requires exactly 65 resident tensors"
    );
    return value.at(10);
}

std::string settle_generation_boundary_v2(
    const std::shared_ptr<resident_run_control::RunControlV2>& run_control,
    const int64_t absolute_generation,
    cudaStream_t stream
) {
    const cudaError_t boundary_status = cudaStreamSynchronize(stream);
    TORCH_CHECK(
        boundary_status == cudaSuccess,
        "CUDA V2 complete-generation boundary synchronization failed: ",
        cudaGetErrorString(boundary_status)
    );
    const auto manifest = run_control->manifest();
    const bool cancellation_pending =
        run_control->cancellation_pending_for_generation(absolute_generation);
    resident_run_control::BoundaryMetadataV2 boundary;
    boundary.completed_generation = absolute_generation;
    boundary.next_generation = absolute_generation + 1;
    boundary.boundary_ordinal =
        absolute_generation - manifest.start_generation + 1;
    boundary.trajectory_commit_frontier = absolute_generation;
    boundary.snapshot_frontier = absolute_generation - 1;
    boundary.output_ack_frontier = absolute_generation - 1;
    boundary.cancellation_checkpoint_requested = cancellation_pending;
    boundary.cancellation_checkpoint_persisted = false;
    return run_control->consume_cancellation_at_boundary(boundary).disposition;
}

pybind11::dict execute_direct_resident_windows_v2(
    const pybind11::function& direct_v1,
    const std::shared_ptr<resident_run_control::RunControlV2>& run_control,
    const pybind11::args& arguments
) {
    TORCH_CHECK(run_control, "CUDA V2 direct execution requires one run control");
    TORCH_CHECK(
        pybind11::len(arguments) == 177,
        "CUDA V2 direct execution requires 172 prepared ABI arguments and "
        "five direct-trajectory arguments"
    );
    TORCH_CHECK(
        !active_direct_run_control_v2,
        "one thread cannot nest CUDA V2 direct trajectories"
    );
    TORCH_CHECK(
        !run_control->manifest().run_started,
        "CUDA V2 direct execution requires a fresh single-use run control"
    );

    active_direct_run_control_v2 = run_control;
    PyObject* raw_result = PyObject_CallObject(direct_v1.ptr(), arguments.ptr());
    active_direct_run_control_v2.reset();
    if (raw_result == nullptr) {
        throw pybind11::error_already_set();
    }
    const auto result = pybind11::reinterpret_steal<pybind11::object>(raw_result);
    TORCH_CHECK(
        pybind11::isinstance<pybind11::dict>(result),
        "CUDA V2 direct trajectory must return one native result mapping"
    );
    pybind11::dict out = pybind11::cast<pybind11::dict>(result);
    out["native_run_control_v2"] =
        resident_run_control_binding::manifest_to_dict_v2(
            run_control->manifest()
        );
    out["native_run_control_v2_trajectory_owner"] = true;
    out["internal_pybind_trajectory_entry_trampoline"] = true;
    out["python_tick_or_generation_ownership"] = false;
    return out;
}

enum CompactSettlementTensorIndexV1 : std::size_t {
    kSettlementAuthorityDeviceV1 = 0,
    kSettlementProducerStatusDeviceV1,
    kSettlementLifecycleDigestDeviceV1,
    kSettlementLifecycleStatusDeviceV1,
    kSettlementLifecycleValidityDeviceV1,
    kSettlementReductionDeviceV1,
    kSettlementMetadataRetentionDeviceV1,
    kSettlementMetadataInitialHostV1,
    kSettlementCompletedGenerationsHostV1,
    kSettlementAuthorityHostV1,
    kSettlementReportingHostV1,
    kSettlementCheckpointRawHostV1,
    kSettlementCheckpointCanonicalHostV1,
    kSettlementCheckpointStatusHostV1,
    kSettlementCheckpointValidityHostV1,
    kSettlementGenerationSummaryHostV1,
    kSettlementPatchMetricsHostV1,
    kSettlementLifecycleDigestHostV1,
    kSettlementLifecycleStatusHostV1,
    kSettlementLifecycleValidityHostV1,
    kSettlementReductionHostV1,
    kSettlementMetadataRetentionHostV1,
    kSettlementCheckpointDueHostV1,
    kSettlementImportDigestHostV1,
    kSettlementHostTemplateProgramsHostV1,
    kSettlementHostTemplateTokensHostV1,
    kSettlementHostTemplatePartsHostV1,
    kSettlementHostTemplateConstraintsHostV1,
    kSettlementHostTemplateBytePoolHostV1,
    kSettlementHostTokenResolverPlanHostV1,
    kSettlementTensorCountV1,
};

static_assert(
    reporting_authority::retained_bytes_per_generation_cell_v1(
        reporting_authority::kDynamicRowCapacityV1
    ) == 43024
);

void check_cuda_contiguous(
    const torch::Tensor& reference,
    const torch::Tensor& tensor,
    const char* name
) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be CUDA resident");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(tensor.device() == reference.device(), name, " must share one device");
}

pybind11::dict compact_summary_abi_metadata_v2() {
    pybind11::list planes;
    const auto append_plane = [&planes](
                                  const char* name,
                                  const char* dtype,
                                  pybind11::tuple shape_suffix
                              ) {
        pybind11::dict plane;
        plane["name"] = name;
        plane["dtype"] = dtype;
        plane["shape_prefix"] = pybind11::make_tuple(
            "generation_count",
            "cells"
        );
        plane["shape_suffix"] = std::move(shape_suffix);
        planes.append(std::move(plane));
    };
    append_plane("validity", "uint8", pybind11::make_tuple());
    append_plane(
        "defined_mask",
        "uint64",
        pybind11::make_tuple(
            static_cast<int64_t>(compact_summary::kDefinedMaskFieldCountV2)
        )
    );
    append_plane(
        "status_codes",
        "int32",
        pybind11::make_tuple(
            static_cast<int64_t>(compact_summary::kStatusFieldCountV2)
        )
    );
    append_plane(
        "integer_values",
        "int64",
        pybind11::make_tuple(
            static_cast<int64_t>(compact_summary::kIntegerFieldCountV2)
        )
    );
    append_plane(
        "float_values",
        "float64",
        pybind11::make_tuple(
            static_cast<int64_t>(compact_summary::kFloatFieldCountV2)
        )
    );
    append_plane(
        "digest_values",
        "uint8",
        pybind11::make_tuple(
            static_cast<int64_t>(compact_summary::kDigestFieldCountV2),
            compact_summary::kDigestBytesV2
        )
    );
    append_plane(
        "text_values",
        "uint8",
        pybind11::make_tuple(
            static_cast<int64_t>(compact_summary::kTextFieldCountV2),
            compact_summary::kTextRecordBytesV2
        )
    );

    pybind11::dict out;
    out["schema_version"] = compact_summary::kNativeAbiSchemaVersionV2;
    out["native_abi_version"] = compact_summary::kNativeAbiVersionV2;
    out["wire_schema_version"] = compact_summary::kWireSchemaVersionNameV1;
    out["codebook_version"] = compact_summary::kCodebookSchemaVersionV1;
    out["defined_mask_version"] =
        compact_summary::kDefinedMaskSchemaVersionV1;
    out["plane_layout_version"] =
        compact_summary::kPlaneLayoutSchemaVersionV2;
    out["projection_contract_version"] =
        compact_summary::kProjectionContractSchemaVersionV2;
    out["invalid_row_contract_version"] =
        compact_summary::kInvalidRowContractSchemaVersionV2;
    out["plane_count"] = static_cast<int64_t>(compact_summary::kPlaneCountV2);
    out["planes"] = std::move(planes);
    out["logical_bytes_per_generation_cell"] =
        compact_summary::kLogicalBytesPerGenerationCellV2;
    out["producer_status"] = "qualified_patch_p2_g1_g2_g8_g32_g64_g128";
    out["invalid_row_status"] = "all_invalid_noop";
    out["validity_written_last"] =
        compact_summary::kInvalidRowContractV2.validity_written_last;
    return out;
}

pybind11::dict flattened_flat_dynamic_policy_food_birth_controller_v1(
    torch::Tensor active_agent_ids,
    torch::Tensor active_agent_slots,
    torch::Tensor active_count_state,
    torch::Tensor slot_to_agent_id,
    torch::Tensor agent_id_to_slot,
    torch::Tensor alive_mask,
    torch::Tensor slot_tokens,
    torch::Tensor action_cache_actions,
    torch::Tensor action_cache_token,
    torch::Tensor action_cache_cursor,
    torch::Tensor action_cache_next,
    torch::Tensor pos_y,
    torch::Tensor pos_x,
    torch::Tensor agent_grid,
    torch::Tensor food_grid,
    torch::Tensor energy,
    torch::Tensor health,
    torch::Tensor injury,
    torch::Tensor policy_energy_observation,
    torch::Tensor rest_streak,
    torch::Tensor pending_free_agent_ids,
    torch::Tensor pending_free_slots,
    torch::Tensor pending_free_count,
    torch::Tensor energy_collected,
    torch::Tensor energy_collected_exact,
    torch::Tensor last_action,
    torch::Tensor last_last_action,
    torch::Tensor action_switches,
    torch::Tensor action_oscillations,
    torch::Tensor episode0_switches,
    torch::Tensor episode0_oscillations,
    torch::Tensor survival_steps,
    torch::Tensor mlp_pack,
    torch::Tensor drives,
    torch::Tensor attack_bias,
    torch::Tensor action_temperature,
    torch::Tensor drive_logit_gain,
    torch::Tensor food_gained_scratch,
    torch::Tensor ordered_agent_ids,
    torch::Tensor ordered_slots,
    torch::Tensor observations,
    torch::Tensor logits,
    torch::Tensor row_actions,
    torch::Tensor actions_by_slot,
    torch::Tensor pre_energy,
    torch::Tensor pre_health,
    torch::Tensor pre_injury,
    torch::Tensor pre_survival_steps,
    torch::Tensor active_count,
    torch::Tensor policy_refill_rows,
    torch::Tensor policy_cache_hit_rows,
    torch::Tensor runtime_bookkeeping_by_tick,
    torch::Tensor death_agent_ids,
    torch::Tensor death_slots,
    torch::Tensor death_reason_codes,
    torch::Tensor death_count,
    torch::Tensor survivor_agent_ids,
    torch::Tensor survivor_count,
    torch::Tensor lifecycle_status,
    torch::Tensor status,
    std::vector<torch::Tensor> full_tick_state_tensors,
    std::vector<torch::Tensor> full_tick_output_tensors,
    std::vector<torch::Tensor> compact_summary_reporting_tensors,
    std::vector<torch::Tensor> compact_summary_checkpoint_source_tensors,
    torch::Tensor compact_summary_checkpoint_descriptor_specs,
    std::vector<torch::Tensor> compact_summary_checkpoint_output_tensors,
    std::vector<torch::Tensor> compact_summary_settlement_tensors,
    std::vector<torch::Tensor> event_state_tensors,
    std::vector<torch::Tensor> ordinary_birth_state_tensors,
    std::vector<torch::Tensor> ordinary_birth_output_tensors,
    std::vector<torch::Tensor> ordinary_birth_scratch_tensors,
    torch::Tensor turnover_selection_seed_by_cell,
    std::vector<torch::Tensor> turnover_state_tensors,
    std::vector<torch::Tensor> turnover_shadow_tensors,
    std::vector<torch::Tensor> turnover_output_tensors,
    std::vector<torch::Tensor> turnover_pointer_table_tensors,
    std::vector<torch::Tensor> patch_p2_state_tensors,
    std::vector<torch::Tensor> patch_p2_scratch_tensors,
    std::vector<torch::Tensor> patch_p2_stats_tensors,
    pybind11::dict patch_p2_config,
    const int64_t ticks,
    const int64_t view_radius,
    const int64_t hidden_size,
    const int64_t action_count,
    const int64_t cache_width,
    const bool deterministic_actions,
    const int64_t max_food,
    const double food_spawn_per_step,
    const double food_spawn_mult,
    const double max_energy,
    const double start_energy,
    const double start_health,
    torch::Tensor move_cost_by_cell,
    const double gather_cost,
    const double rest_cost,
    const double attack_cost,
    const double action_cost_mult,
    const double rest_cost_mult,
    const double food_energy,
    const int64_t episode0_k,
    const double w_hunger_gather,
    const double w_pain_rest,
    const double w_fatigue_rest,
    const double w_pain_attack,
    const double w_fear_attack,
    const bool enable_rest_charge,
    const int64_t rest_streak_free,
    const double rest_charge_energy,
    const double rest_charge_growth,
    const double rest_deficit_to_health,
    const int64_t developmental_maturity_age,
    const int64_t life_history_mode,
    const int64_t life_history_maturity_start_ticks,
    const int64_t life_history_maturity_rise_ticks,
    const int64_t life_history_fertility_decline_start_ticks,
    const int64_t life_history_fertility_decline_ticks,
    const double life_history_elder_fertility_floor,
    const int64_t life_history_senescence_start_ticks,
    const int64_t life_history_senescence_ramp_ticks,
    const double life_history_senescence_energy_cost,
    const double life_history_senescence_health_cost,
    const int64_t life_history_max_age_ticks,
    const double drive_hunger_gain,
    const double drive_hunger_decay,
    const double drive_pain_gain,
    const double drive_pain_decay,
    const double drive_fatigue_gain,
    const double drive_fatigue_rest_relief,
    const double drive_fatigue_decay,
    const double drive_fear_gain,
    const double drive_fear_decay,
    const int64_t generation,
    const int64_t start_tick,
    const bool ordinary_births_enabled,
    const int64_t max_births_per_tick,
    const double reproduce_parent_energy_share,
    const double reproduce_min_energy_buffer,
    const int64_t reproduce_cooldown_steps,
    const int64_t genome_sensor_radius_max,
    const double injury_repro_k_injury,
    const double injury_repro_k_scar,
    const double injury_birth_fail_scar,
    const double injury_birth_fail_injury,
    const bool env_policy_alias,
    const bool reproduce_requires_mate,
    const bool generation_turnover_enabled,
    const int64_t lifecycle_mode,
    const int64_t turnover_population_size,
    const double turnover_cull_fraction,
    const double turnover_parent_fraction,
    const int64_t turnover_selection_mode,
    const double coherence_thrashing_weight,
    const double coherence_drive_volatility_weight,
    const double coherence_selfharm_weight,
    const double coherence_pred_err_weight,
    const int64_t generation_count,
    const double reporting_parent_fraction,
    const double reporting_episode0_fitness_weight,
    const double reporting_episode0_pred_err_fitness_weight,
    const double reporting_episode0_thrashing_fitness_weight,
    const double reporting_episode0_drive_jitter_fitness_weight,
    const double reporting_episode0_selfharm_fitness_weight,
    const double reporting_coherence_thrashing_weight,
    const double reporting_coherence_drive_jitter_weight,
    const double reporting_coherence_selfharm_weight,
    const double reporting_coherence_pred_err_weight,
    const double reporting_coherence_reset_shock_weight,
    const double reporting_coherence_episode0_pred_err_weight,
    const double reporting_late_pred_cost_weight,
    const bool reporting_late_pred_cost_enable,
    const bool reporting_disable_prediction_error_selection_pressure,
    torch::Tensor first_failure_records
) {
    const bool exact_tick_native_span =
        patch_p2_config.contains("tw_rngv2_semantics_version")
        && pybind11::cast<int32_t>(
            patch_p2_config["tw_rngv2_semantics_version"]
        ) == 2
        && generation_count == 1
        && generation == 0
        && start_tick >= 0
        && ticks > 0
        && !generation_turnover_enabled;
    const bool exact_partial_tick_span =
        generation_count == 1
        && ticks > 0
        && ticks <= 160
        && start_tick >= 0
        && start_tick < 160
        && ticks <= 160 - start_tick
        && (
            start_tick > 0
            || (ticks != 80 && ticks != 160)
            || (EIEL_CUDA_ARCHITECTURE_B_T160_PARTIAL_SPAN_ALLOWED && ticks < 160)
        );
    TORCH_CHECK(
        ticks == 80 || ticks == 160 || exact_partial_tick_span
            || exact_tick_native_span,
        "persistent Patch P2 ticks must be a full admitted interval or one "
        "exact absolute-tick native span"
    );
    TORCH_CHECK(
        generation_count > 0 && generation_count <= 128,
        "generation_count must be in [1, 128]"
    );
    TORCH_CHECK(view_radius >= 0, "view_radius must be non-negative");
    TORCH_CHECK(hidden_size > 0 && hidden_size <= 64, "hidden_size must be in [1, 64]");
    TORCH_CHECK(
        action_count == 7,
        "the qualified flattened-controller envelope requires action_count=7"
    );
#ifndef EIEL_CUDA_ARCHITECTURE_B_CACHE_WIDTH_VALID
#define EIEL_CUDA_ARCHITECTURE_B_CACHE_WIDTH_VALID (cache_width == 2)
#endif
    TORCH_CHECK(
        EIEL_CUDA_ARCHITECTURE_B_CACHE_WIDTH_VALID,
        "bounded persistent cache_width is outside the qualified route"
    );
    TORCH_CHECK(max_food >= 0, "max_food must be non-negative");
    TORCH_CHECK(food_spawn_per_step >= 0.0, "food_spawn_per_step must be non-negative");
    TORCH_CHECK(food_spawn_mult >= 0.0, "food_spawn_mult must be non-negative");
    TORCH_CHECK(max_energy != 0.0 && start_health != 0.0, "observation denominators must be non-zero");
    TORCH_CHECK(episode0_k >= 0, "episode0_k must be non-negative");
    TORCH_CHECK(rest_streak_free >= 0, "rest_streak_free must be non-negative");
    TORCH_CHECK(developmental_maturity_age >= 0, "maturity age must be non-negative");
    TORCH_CHECK(
        life_history_mode == 0 || life_history_mode == 1,
        "life_history_mode must be legacy=0 or integrated=1"
    );
    TORCH_CHECK(
        life_history_maturity_start_ticks >= 0
            && life_history_maturity_rise_ticks >= 0
            && life_history_fertility_decline_start_ticks >= 0
            && life_history_fertility_decline_ticks >= 0
            && life_history_senescence_start_ticks >= 0
            && life_history_senescence_ramp_ticks >= 0
            && life_history_max_age_ticks >= 0,
        "life-history tick parameters must be non-negative"
    );
    TORCH_CHECK(
        std::isfinite(life_history_elder_fertility_floor)
            && life_history_elder_fertility_floor >= 0.0
            && life_history_elder_fertility_floor <= 1.0,
        "life_history_elder_fertility_floor must be finite in [0, 1]"
    );
    TORCH_CHECK(
        std::isfinite(life_history_senescence_energy_cost)
            && life_history_senescence_energy_cost >= 0.0
            && std::isfinite(life_history_senescence_health_cost)
            && life_history_senescence_health_cost >= 0.0,
        "life-history senescence costs must be finite and non-negative"
    );
    TORCH_CHECK(generation >= 0 && start_tick >= 0, "generation/tick must be non-negative");
    TORCH_CHECK(
        max_births_per_tick >= 0,
        "flattened ordinary births per tick must be non-negative"
    );
    if (generation_turnover_enabled) {
        TORCH_CHECK(
            lifecycle_mode == 0 || lifecycle_mode == 1,
            "lifecycle mode must be legacy=0 or hybrid-overlapping=1"
        );
        TORCH_CHECK(
            turnover_population_size > 0,
            "turnover population_size must be positive"
        );
        TORCH_CHECK(
            std::isfinite(start_energy) && start_energy > 0.0,
            "turnover start_energy must be finite and positive"
        );
        TORCH_CHECK(
            turnover_cull_fraction >= 0.0 && turnover_cull_fraction <= 1.0,
            "turnover cull_fraction must be in [0, 1]"
        );
        TORCH_CHECK(
            turnover_parent_fraction >= 0.0 && turnover_parent_fraction <= 1.0,
            "turnover parent_fraction must be in [0, 1]"
        );
    }
    TORCH_CHECK(
        full_tick_state_tensors.size() == 29,
        "full_tick_state_tensors must contain 29 tensors"
    );
    const bool generation_summary_scalar_enabled =
        full_tick_output_tensors.size() == 20
        || full_tick_output_tensors.size() == 27;
    const bool compact_summary_planes_enabled =
        full_tick_output_tensors.size() == 26
        || full_tick_output_tensors.size() == 27;
    TORCH_CHECK(
        full_tick_output_tensors.size() == 19
            || full_tick_output_tensors.size() == 20
            || full_tick_output_tensors.size() == 26
            || full_tick_output_tensors.size() == 27,
        "full_tick_output_tensors must contain 19 base tensors, 20 with "
        "the Phase A generation-summary scalar, 26 with seven compact-summary "
        "ABI v2 planes, or 27 with Phase A and the seven compact planes"
    );
    const torch::Tensor generation_summary_scalar =
        generation_summary_scalar_enabled
        ? full_tick_output_tensors.at(19)
        : torch::Tensor();
    const size_t compact_summary_plane_offset =
        19 + (generation_summary_scalar_enabled ? 1 : 0);
    TORCH_CHECK(
        event_state_tensors.size() == 31,
        "event_state_tensors must contain 31 tensors"
    );
    TORCH_CHECK(
        !ordinary_births_enabled || max_births_per_tick > 0,
        "enabled ordinary births require positive scratch capacity"
    );
    TORCH_CHECK(
        turnover_selection_mode == 0 || turnover_selection_mode == 1,
        "turnover selection mode must be fitness=0 or uniform-neutral=1"
    );
    const bool full_patch_lifecycle_state_required =
        patch_p2_state_tensors.size() == 10
        && patch_p2_scratch_tensors.size() == 1
        && (
            patch_p2_stats_tensors.size() == 1
            || (compact_summary_planes_enabled
                && patch_p2_stats_tensors.size() == 2)
        )
        && (patch_p2_config.size() == 34 || patch_p2_config.size() == 35
            || patch_p2_config.size() == 40 || patch_p2_config.size() == 41
            || patch_p2_config.size() == 44 || patch_p2_config.size() == 45
            || patch_p2_config.size() == 46 || patch_p2_config.size() == 47);
    const bool lifecycle_state_enabled = ordinary_births_enabled
        || generation_turnover_enabled
        || full_patch_lifecycle_state_required;
    const bool lifecycle_scratch_enabled = ordinary_births_enabled
        || generation_turnover_enabled
        || full_patch_lifecycle_state_required;
    const bool genome_upgrade_v2_state_enabled =
        lifecycle_state_enabled && ordinary_birth_state_tensors.size() == 139;
    const bool diploid_state_enabled =
        lifecycle_state_enabled
        && (
            ordinary_birth_state_tensors.size() == 122
            || genome_upgrade_v2_state_enabled
        );
    TORCH_CHECK(
        !(diploid_state_enabled && active_cross_path_invocation_v1 != nullptr),
        "diploid ordinary-birth authority and cross-path resident authority "
        "are mutually exclusive within one native invocation"
    );
    TORCH_CHECK(
        ordinary_birth_state_tensors.size()
            == (lifecycle_state_enabled
                    ? (
                        genome_upgrade_v2_state_enabled
                            ? 139
                            : (diploid_state_enabled ? 122 : 113)
                    )
                    : 0),
        "ordinary_birth_state_tensors must be empty when lifecycle is disabled "
        "and contain 113 legacy tensors, 122 diploid tensors, or 139 fixed-Q10 "
        "genome tensors when births, turnover, or full Patch are enabled"
    );
    TORCH_CHECK(
        ordinary_birth_output_tensors.size()
            == ((ordinary_births_enabled || compact_summary_planes_enabled)
                    ? 63
                    : 0),
        "ordinary_birth_output_tensors must contain 63 tensors when births or "
        "compact summary reporting are enabled and be empty otherwise"
    );
    TORCH_CHECK(
        ordinary_birth_scratch_tensors.size()
            == (lifecycle_scratch_enabled ? 1 : 0),
        "ordinary_birth_scratch_tensors must contain one tensor when births, "
        "turnover, or full Patch lifecycle state are enabled and be empty "
        "otherwise"
    );
    TORCH_CHECK(
        turnover_state_tensors.size()
            == (generation_turnover_enabled
                    ? turnover::kTurnoverMutableTensorCount
                    : 0)
            && turnover_shadow_tensors.size()
                == (generation_turnover_enabled
                        ? turnover::kTurnoverMutableTensorCount
                        : 0)
            && turnover_output_tensors.size()
                == (generation_turnover_enabled
                        ? 16
                        : (turnover_output_tensors.empty() ? 0 : 16))
            && turnover_pointer_table_tensors.size()
                == (generation_turnover_enabled ? 1 : 0),
        "turnover mutable tensor groups must be empty when disabled; the "
        "16-output status group may remain for resident controller reporting"
    );
    const bool patch_p2_disabled = patch_p2_state_tensors.empty()
        && patch_p2_scratch_tensors.empty()
        && patch_p2_stats_tensors.empty()
        && patch_p2_config.size() == 0;
    const bool patch_p2_context_probe_mode =
        patch_p2_state_tensors.size() == 7
        && patch_p2_scratch_tensors.size() == 1
        && patch_p2_stats_tensors.size() == 1
        && patch_p2_config.size() == 23;
    const bool patch_p2_full_mode = patch_p2_state_tensors.size() == 10
        && patch_p2_scratch_tensors.size() == 1
        && (
            patch_p2_stats_tensors.size() == 1
            || (compact_summary_planes_enabled
                && patch_p2_stats_tensors.size() == 2)
        )
        && (patch_p2_config.size() == 34 || patch_p2_config.size() == 35
            || patch_p2_config.size() == 40 || patch_p2_config.size() == 41
            || patch_p2_config.size() == 44 || patch_p2_config.size() == 45
            || patch_p2_config.size() == 46 || patch_p2_config.size() == 47);
    TORCH_CHECK(
        patch_p2_disabled
            || patch_p2_context_probe_mode
            || patch_p2_full_mode,
        "Patch P2 groups must all be empty when disabled, contain the closed "
        "7-state/1-scratch/1-stats/23-config b2/t1 context-probe ABI, or "
        "contain the full 10-state/1-scratch/1-or-2-stats/34-config Patch owner ABI"
    );
    const bool patch_p2_enabled = patch_p2_context_probe_mode
        || patch_p2_full_mode;
    if (generation_count > 1) {
        TORCH_CHECK(
            !patch_p2_context_probe_mode,
            "the Patch P2 context probe does not support resident multigeneration"
        );
        TORCH_CHECK(
            !patch_p2_full_mode
                || generation_count == 2
                || generation_count == 8
                || generation_count == 32
                || generation_count == 64
                || generation_count == 128,
            "resident Patch P2 validation admits generation_count in {2, 8, 32, 64, 128}"
        );
        TORCH_CHECK(
            !patch_p2_full_mode
                || generation_count == 2
                || compact_summary_planes_enabled
                || generation_summary_scalar_enabled,
            "resident Patch P2 g8/g32/g64/g128 requires a generation summary output"
        );
        TORCH_CHECK(
            (ticks == 80 || ticks == 160) && start_tick == 0,
            "resident multigeneration requires t80 or t160 generation boundaries"
        );
        TORCH_CHECK(
            generation_turnover_enabled || lifecycle_mode == 1,
            "resident multigeneration requires generation turnover or "
            "hybrid-overlapping lifecycle mode"
        );
    }
    if (patch_p2_enabled) {
        for (const char* key : {
                 "patch_scent_decay_per_step",
                 "patch_waste_decay_per_step",
                 "patch_scent_presence_emit",
                 "patch_waste_presence_emit",
                 "patch_waste_density_emit_scale",
                 "patch_scent_max",
                 "patch_waste_max",
                 "patch_food_amount_max",
                 "patch_food_regrow_per_step",
                 "patch_food_regrow_use_fertility",
                 "patch_food_regrow_cap_by_fertility",
                 "patch_food_presence_threshold",
                 "patch_density_radius",
                 "patch_density_include_self",
                 "patch_density_normalize",
                 "patch_world_waste_consequence_enable",
                 "patch_waste_regrowth_penalty_enable",
                 "patch_waste_regrowth_penalty_scale",
                 "patch_waste_regrowth_penalty_max",
                 "patch_density_regrowth_penalty_enable",
                 "patch_density_regrowth_penalty_scale",
                 "patch_density_regrowth_penalty_max",
                 "patch_regrowth_min_multiplier",
             }) {
            TORCH_CHECK(
                patch_p2_config.contains(pybind11::str(key)),
                "Patch P2 config is missing ",
                key
            );
        }
    }
    if (patch_p2_full_mode) {
        for (const char* key : {
                 "patch_food_amount_initial_food_cell_value",
                 "patch_food_gather_amount",
                 "patch_scent_move_emit",
                 "patch_waste_move_emit",
                 "patch_scent_gather_emit",
                 "patch_waste_consume_emit",
                 "patch_movement_cost_apply_enable",
                 "patch_movement_cost_energy_scale",
                 "patch_movement_cost_energy_scale_by_cell",
                 "patch_food_gather_amount_by_cell",
                 "patch_food_regrow_per_step_by_cell",
                 "reproduce_parent_energy_share_by_cell",
                 "reproduce_min_energy_buffer_by_cell",
                 "reproduce_cooldown_steps_by_cell",
                 "reproduce_try_prob_by_cell",
                 "patch_movement_cost_min",
                 "patch_movement_cost_max",
             }) {
            TORCH_CHECK(
                patch_p2_config.contains(pybind11::str(key)),
                "full Patch P2 config is missing ",
                key
            );
        }
    }

    for (const auto& named : std::vector<std::pair<const char*, torch::Tensor>>{
             {"active_agent_ids", active_agent_ids},
             {"active_agent_slots", active_agent_slots},
             {"active_count_state", active_count_state},
             {"slot_to_agent_id", slot_to_agent_id},
             {"agent_id_to_slot", agent_id_to_slot},
             {"alive_mask", alive_mask},
             {"slot_tokens", slot_tokens},
             {"action_cache_actions", action_cache_actions},
             {"action_cache_token", action_cache_token},
             {"action_cache_cursor", action_cache_cursor},
             {"action_cache_next", action_cache_next},
             {"pos_y", pos_y},
             {"pos_x", pos_x},
             {"agent_grid", agent_grid},
             {"food_grid", food_grid},
             {"energy", energy},
             {"health", health},
             {"injury", injury},
             {"policy_energy_observation", policy_energy_observation},
             {"rest_streak", rest_streak},
             {"pending_free_agent_ids", pending_free_agent_ids},
             {"pending_free_slots", pending_free_slots},
             {"pending_free_count", pending_free_count},
             {"energy_collected", energy_collected},
             {"energy_collected_exact", energy_collected_exact},
             {"last_action", last_action},
             {"last_last_action", last_last_action},
             {"action_switches", action_switches},
             {"action_oscillations", action_oscillations},
             {"episode0_switches", episode0_switches},
             {"episode0_oscillations", episode0_oscillations},
             {"survival_steps", survival_steps},
             {"mlp_pack", mlp_pack},
             {"drives", drives},
             {"attack_bias", attack_bias},
             {"action_temperature", action_temperature},
             {"drive_logit_gain", drive_logit_gain},
             {"food_gained_scratch", food_gained_scratch},
             {"ordered_agent_ids", ordered_agent_ids},
             {"ordered_slots", ordered_slots},
             {"observations", observations},
             {"logits", logits},
             {"row_actions", row_actions},
             {"actions_by_slot", actions_by_slot},
             {"pre_energy", pre_energy},
             {"pre_health", pre_health},
             {"pre_injury", pre_injury},
             {"pre_survival_steps", pre_survival_steps},
             {"active_count", active_count},
             {"policy_refill_rows", policy_refill_rows},
             {"policy_cache_hit_rows", policy_cache_hit_rows},
             {"runtime_bookkeeping_by_tick", runtime_bookkeeping_by_tick},
             {"death_agent_ids", death_agent_ids},
             {"death_slots", death_slots},
             {"death_reason_codes", death_reason_codes},
             {"death_count", death_count},
             {"survivor_agent_ids", survivor_agent_ids},
             {"survivor_count", survivor_count},
             {"lifecycle_status", lifecycle_status},
             {"status", status},
             {"first_failure_records", first_failure_records},
         }) {
        check_cuda_contiguous(active_agent_ids, named.second, named.first);
    }
    for (const auto& tensor : full_tick_state_tensors) {
        check_cuda_contiguous(active_agent_ids, tensor, "full_tick_state_tensor");
    }
    for (const auto& tensor : full_tick_output_tensors) {
        check_cuda_contiguous(active_agent_ids, tensor, "full_tick_output_tensor");
    }
    for (const auto& tensor : event_state_tensors) {
        check_cuda_contiguous(active_agent_ids, tensor, "event_state_tensor");
    }
    for (const auto& tensor : ordinary_birth_state_tensors) {
        check_cuda_contiguous(
            active_agent_ids,
            tensor,
            "ordinary_birth_state_tensor"
        );
    }
    for (const auto& tensor : ordinary_birth_output_tensors) {
        check_cuda_contiguous(
            active_agent_ids,
            tensor,
            "ordinary_birth_output_tensor"
        );
    }
    for (const auto& tensor : ordinary_birth_scratch_tensors) {
        check_cuda_contiguous(
            active_agent_ids,
            tensor,
            "ordinary_birth_scratch_tensor"
        );
    }
    for (const auto& tensor : turnover_state_tensors) {
        check_cuda_contiguous(active_agent_ids, tensor, "turnover_state_tensor");
    }
    for (const auto& tensor : turnover_shadow_tensors) {
        check_cuda_contiguous(active_agent_ids, tensor, "turnover_shadow_tensor");
    }
    for (const auto& tensor : turnover_output_tensors) {
        check_cuda_contiguous(active_agent_ids, tensor, "turnover_output_tensor");
    }
    for (const auto& tensor : turnover_pointer_table_tensors) {
        check_cuda_contiguous(
            active_agent_ids, tensor, "turnover_pointer_table_tensor"
        );
    }
    for (const auto& tensor : patch_p2_state_tensors) {
        check_cuda_contiguous(active_agent_ids, tensor, "Patch P2 state tensor");
    }
    for (const auto& tensor : patch_p2_scratch_tensors) {
        check_cuda_contiguous(active_agent_ids, tensor, "Patch P2 scratch tensor");
    }
    for (const auto& tensor : patch_p2_stats_tensors) {
        check_cuda_contiguous(active_agent_ids, tensor, "Patch P2 stats tensor");
    }
    TORCH_CHECK(
        active_agent_ids.scalar_type() == torch::kInt64 && active_agent_ids.dim() == 2,
        "active_agent_ids must be int64[cells,active_capacity]"
    );
    const int64_t cells = active_agent_ids.size(0);
    if (generation_count > 1) {
        TORCH_CHECK(
            cells <= 72,
            "resident multigeneration candidate supports at most batch_cells=72"
        );
    }
    const int64_t active_capacity = active_agent_ids.size(1);
    TORCH_CHECK(cells > 0 && active_capacity > 0, "persistent challenger requires non-empty cells and active capacity");
    TORCH_CHECK(
        move_cost_by_cell.scalar_type() == torch::kFloat64
            && move_cost_by_cell.dim() == 1
            && move_cost_by_cell.size(0) == cells
            && move_cost_by_cell.is_cuda()
            && move_cost_by_cell.is_contiguous()
            && move_cost_by_cell.device() == active_agent_ids.device(),
        "move costs must be contiguous CUDA float64[cells]"
    );
    TORCH_CHECK(
        turnover_selection_seed_by_cell.scalar_type() == torch::kUInt64
            && turnover_selection_seed_by_cell.dim() == 1
            && turnover_selection_seed_by_cell.size(0) == cells
            && turnover_selection_seed_by_cell.is_cuda()
            && turnover_selection_seed_by_cell.is_contiguous(),
        "turnover selection seeds must be contiguous CUDA uint64[cells]"
    );
    if (lifecycle_scratch_enabled) {
        TORCH_CHECK(
            ordinary_birth_scratch_tensors.at(0).scalar_type() == torch::kUInt8
                && ordinary_birth_scratch_tensors.at(0).dim() == 2
                && ordinary_birth_scratch_tensors.at(0).size(0) == cells,
            "flattened lifecycle scratch must be uint8[cells,bytes_per_cell]"
        );
    }
    if (generation_turnover_enabled) {
        TORCH_CHECK(
            turnover_population_size <= turnover::kMaximumCapacity,
            "bounded flattened turnover population_size must not exceed "
            "turnover::kMaximumCapacity"
        );
        for (
            int64_t index = 0;
            index < turnover::kTurnoverMutableTensorCount;
            ++index
        ) {
            const auto& source = turnover_state_tensors.at(index);
            const auto& shadow = turnover_shadow_tensors.at(index);
            const bool compact_mlp_shadow =
                index == turnover::kMlpPack
                && source.dim() == 3 && shadow.dim() == 3
                && shadow.size(0) == cells
                && shadow.size(1) >= turnover_population_size
                && shadow.size(2) == source.size(2);
            const bool deferred_mlp_shadow =
                index == turnover::kMlpPack
                && source.dim() == 3 && shadow.dim() == 3
                && shadow.size(0) == cells
                && shadow.size(1) == 0
                && shadow.size(2) == source.size(2);
            TORCH_CHECK(
                source.dim() >= 1 && source.size(0) == cells
                    && (compact_mlp_shadow || deferred_mlp_shadow
                        || (source.sizes() == shadow.sizes()
                            && source.strides() == shadow.strides()))
                    && source.scalar_type() == shadow.scalar_type()
                    && source.data_ptr() != shadow.data_ptr(),
                "turnover source/shadow tensors must be disjoint compatible "
                "contiguous cell-major tensors"
            );
        }
        TORCH_CHECK(
            turnover_pointer_table_tensors.at(0).scalar_type() == torch::kInt64
                && turnover_pointer_table_tensors.at(0).numel()
                    == turnover::kTurnoverMutableTensorCount * 4,
            "turnover pointer table must contain four int64 fields per tensor"
        );
    }
    TORCH_CHECK(
        active_count_state.scalar_type() == torch::kInt64
            && active_count_state.sizes() == torch::IntArrayRef({cells}),
        "active_count_state must be int64[cells]"
    );
    TORCH_CHECK(
        active_agent_slots.scalar_type() == torch::kInt32
            && active_agent_slots.sizes()
                == torch::IntArrayRef({cells, active_capacity}),
        "active_agent_slots must be int32[cells,active_capacity]"
    );
    TORCH_CHECK(
        agent_id_to_slot.scalar_type() == torch::kInt32
            && agent_id_to_slot.dim() == 2
            && agent_id_to_slot.size(0) == cells,
        "agent_id_to_slot must be int32[cells,agent_id_capacity]"
    );
    TORCH_CHECK(
        energy.scalar_type() == torch::kFloat32
            && energy.dim() == 2
            && energy.size(0) == cells,
        "energy must be float32[cells,capacity]"
    );
    const int64_t capacity = energy.size(1);
    const int64_t agent_id_capacity = agent_id_to_slot.size(1);
    TORCH_CHECK(
        agent_id_to_slot.sizes()
            == torch::IntArrayRef({cells, capacity}),
        "agent_id_to_slot is a slot-capacity compatibility shadow"
    );
    TORCH_CHECK(
        active_capacity <= capacity,
        "active_agent_ids capacity must not exceed slot capacity"
    );
    TORCH_CHECK(
        !lifecycle_state_enabled || active_capacity == capacity,
        "lifecycle-enabled active_agent_ids must cover the complete slot capacity"
    );
    TORCH_CHECK(
        max_births_per_tick <= capacity,
        "flattened ordinary births per tick must not exceed slot capacity"
    );
    TORCH_CHECK(
        generation <= INT32_MAX && start_tick <= INT32_MAX
            && ticks - 1 <= INT32_MAX - start_tick,
        "birth generation and absolute ticks must fit int32"
    );
    const auto check_cell_capacity = [&](const torch::Tensor& tensor,
                                         const char* name,
                                         const c10::ScalarType dtype,
                                         const int64_t rank) {
        TORCH_CHECK(
            tensor.scalar_type() == dtype
                && tensor.dim() == rank
                && tensor.size(0) == cells
                && tensor.size(1) == capacity,
            name,
            " has an invalid cell/capacity shape"
        );
    };
#ifdef EIEL_WDL_JUNGLE_EED_HAPB_FIXED_MLP_NATIVE_BINDING_V1
    const bool hapb_direct_mlp_sentinel =
        hapb_policy_context::hapb_direct_route_context_active_v1();
#else
    const bool hapb_direct_mlp_sentinel = false;
#endif
    check_cell_capacity(pos_y, "pos_y", torch::kInt32, 2);
    check_cell_capacity(pos_x, "pos_x", torch::kInt32, 2);
    check_cell_capacity(slot_to_agent_id, "slot_to_agent_id", torch::kInt64, 2);
    check_cell_capacity(alive_mask, "alive_mask", torch::kBool, 2);
    check_cell_capacity(health, "health", torch::kFloat32, 2);
    check_cell_capacity(injury, "injury", torch::kFloat32, 2);
    check_cell_capacity(rest_streak, "rest_streak", torch::kInt32, 2);
    check_cell_capacity(slot_tokens, "slot_tokens", torch::kInt64, 2);
    check_cell_capacity(action_cache_token, "action_cache_token", torch::kInt64, 2);
    check_cell_capacity(action_cache_cursor, "action_cache_cursor", torch::kInt32, 2);
    check_cell_capacity(action_cache_next, "action_cache_next", torch::kInt32, 2);
    TORCH_CHECK(
        action_cache_actions.scalar_type() == torch::kInt32
            && action_cache_actions.dim() == 3
            && action_cache_actions.size(0) == cells
            && action_cache_actions.size(1) == capacity
            && cache_width <= action_cache_actions.size(2),
        "action_cache_actions must be int32[cells,capacity,cache_capacity]"
    );
    const int64_t cache_capacity = action_cache_actions.size(2);
    TORCH_CHECK(
        pending_free_agent_ids.scalar_type() == torch::kInt64
            && pending_free_agent_ids.dim() == 2
            && pending_free_agent_ids.size(0) == cells,
        "pending_free_agent_ids must be int64[cells,pending_capacity]"
    );
    const int64_t pending_capacity = pending_free_agent_ids.size(1);
    TORCH_CHECK(pending_capacity >= capacity, "pending free capacity must cover slot capacity");
    TORCH_CHECK(
        pending_free_slots.scalar_type() == torch::kInt64
            && pending_free_slots.sizes() == pending_free_agent_ids.sizes(),
        "pending_free_slots must match pending_free_agent_ids"
    );
    TORCH_CHECK(
        pending_free_count.scalar_type() == torch::kInt64
            && pending_free_count.sizes() == torch::IntArrayRef({cells}),
        "pending_free_count must be int64[cells]"
    );
    check_cell_capacity(
        policy_energy_observation,
        "policy_energy_observation",
        torch::kFloat64,
        2
    );
    check_cell_capacity(energy_collected, "energy_collected", torch::kFloat32, 2);
    check_cell_capacity(
        energy_collected_exact,
        "energy_collected_exact",
        torch::kFloat64,
        2
    );
    check_cell_capacity(last_action, "last_action", torch::kInt32, 2);
    check_cell_capacity(last_last_action, "last_last_action", torch::kInt32, 2);
    check_cell_capacity(action_switches, "action_switches", torch::kInt32, 2);
    check_cell_capacity(action_oscillations, "action_oscillations", torch::kInt32, 2);
    check_cell_capacity(episode0_switches, "episode0_switches", torch::kInt32, 2);
    check_cell_capacity(
        episode0_oscillations,
        "episode0_oscillations",
        torch::kInt32,
        2
    );
    check_cell_capacity(survival_steps, "survival_steps", torch::kInt32, 2);
    check_cell_capacity(food_gained_scratch, "food_gained_scratch", torch::kFloat32, 2);
    if (hapb_direct_mlp_sentinel) {
        TORCH_CHECK(
            mlp_pack.scalar_type() == torch::kFloat32
                && mlp_pack.dim() == 3
                && mlp_pack.size(0) == cells
                && mlp_pack.size(1) == 1
                && mlp_pack.size(2) == 9991,
            "HAPB direct policy-only route requires the nonresident MLP ABI sentinel"
        );
    } else {
        check_cell_capacity(mlp_pack, "mlp_pack", torch::kFloat32, 3);
    }
    check_cell_capacity(drives, "drives", torch::kFloat32, 3);
    check_cell_capacity(attack_bias, "attack_bias", torch::kFloat32, 2);
    check_cell_capacity(action_temperature, "action_temperature", torch::kFloat32, 2);
    check_cell_capacity(drive_logit_gain, "drive_logit_gain", torch::kFloat32, 2);
    TORCH_CHECK(drives.size(2) >= 4, "drives must expose at least four channels");
    for (const int64_t index : {0, 13, 14}) {
        check_cell_capacity(
            full_tick_state_tensors.at(index),
            "full tick int32 slot state",
            torch::kInt32,
            2
        );
    }
    for (const int64_t index : {1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12}) {
        check_cell_capacity(
            full_tick_state_tensors.at(index),
            "full tick float32 slot state",
            torch::kFloat32,
            2
        );
    }
    TORCH_CHECK(
        full_tick_state_tensors.at(6).scalar_type() == torch::kFloat32
            && full_tick_state_tensors.at(6).sizes()
                == torch::IntArrayRef({cells, capacity, drives.size(2)}),
        "prev_drive_delta must match drives"
    );
    TORCH_CHECK(
        full_tick_state_tensors.at(15).scalar_type() == torch::kFloat32
            && full_tick_state_tensors.at(15).dim() == 3
            && full_tick_state_tensors.at(15).size(0) == cells
            && full_tick_state_tensors.at(15).size(1) == capacity
            && full_tick_state_tensors.at(15).size(2) >= 6,
        "chemistry must be float32[cells,capacity,width>=6]"
    );
    for (const int64_t index : {16, 18}) {
        TORCH_CHECK(
            full_tick_state_tensors.at(index).scalar_type() == torch::kUInt64
                && full_tick_state_tensors.at(index).sizes()
                    == torch::IntArrayRef({cells, 4}),
            "PCG64 state must be uint64[cells,4]"
        );
    }
    for (const int64_t index : {17, 19}) {
        TORCH_CHECK(
            full_tick_state_tensors.at(index).scalar_type() == torch::kInt64
                && full_tick_state_tensors.at(index).sizes()
                    == torch::IntArrayRef({cells, 2}),
            "PCG64 buffer must be int64[cells,2]"
        );
    }
    TORCH_CHECK(
        full_tick_state_tensors.at(20).scalar_type() == torch::kFloat32
            && full_tick_state_tensors.at(20).dim() == 4
            && full_tick_state_tensors.at(20).size(0) == cells
            && full_tick_state_tensors.at(20).size(2) == capacity
            && full_tick_state_tensors.at(20).size(3) == drives.size(2),
        "drive noise samples have an invalid shape"
    );
    const int64_t drive_noise_sample_ticks = full_tick_state_tensors.at(20).size(1);
    TORCH_CHECK(
        full_tick_state_tensors.at(21).scalar_type() == torch::kBool
            && full_tick_state_tensors.at(21).sizes()
                == torch::IntArrayRef({cells, drive_noise_sample_ticks, capacity}),
        "drive noise sample validity has an invalid shape"
    );
    for (const int64_t index : {22, 24, 25}) {
        TORCH_CHECK(
            full_tick_state_tensors.at(index).scalar_type() == torch::kInt64
                && full_tick_state_tensors.at(index).sizes()
                    == torch::IntArrayRef({cells, capacity}),
            "full tick queue/scratch must be int64[cells,capacity]"
        );
    }
    for (const int64_t index : {23, 26, 27, 28}) {
        TORCH_CHECK(
            full_tick_state_tensors.at(index).scalar_type() == torch::kInt64
                && full_tick_state_tensors.at(index).sizes()
                    == torch::IntArrayRef({cells}),
            "full tick scalar state must be int64[cells]"
        );
    }
    TORCH_CHECK(
        full_tick_state_tensors.at(24).data_ptr<int64_t>()
            != full_tick_output_tensors.at(10).data_ptr<int64_t>()
            && full_tick_state_tensors.at(25).data_ptr<int64_t>()
                != full_tick_output_tensors.at(11).data_ptr<int64_t>()
            && full_tick_state_tensors.at(26).data_ptr<int64_t>()
                != full_tick_output_tensors.at(13).data_ptr<int64_t>(),
        "residual pending scratch must not alias residual death outputs"
    );
    TORCH_CHECK(
        agent_grid.scalar_type() == torch::kInt32
            && agent_grid.dim() == 3
            && agent_grid.size(0) == cells,
        "agent_grid must be int32[cells,height,width]"
    );
    TORCH_CHECK(
        food_grid.scalar_type() == torch::kInt32
            && food_grid.sizes() == agent_grid.sizes(),
        "food_grid must match agent_grid"
    );
    const int64_t height = agent_grid.size(1);
    const int64_t width = agent_grid.size(2);
    TORCH_CHECK(height > 0 && width > 0, "grid dimensions must be positive");
    if (patch_p2_enabled) {
        TORCH_CHECK(
            height <= eiel::cuda_architecture_b::persistent_patch_p2::
                    kMaximumExactSerialCells
                / width,
            "bounded exact Patch P2 serial reduction supports at most 1024 grid cells"
        );
        const std::vector<const char*> patch_p2_state_names{
            "patch_food_amount",
            "patch_fertility",
            "patch_scent",
            "patch_waste",
            "patch_scent_emit_pending",
            "patch_waste_emit_pending",
            "patch_density",
        };
        std::vector<const char*> active_patch_p2_state_names{
            patch_p2_state_names
        };
        if (patch_p2_full_mode) {
            active_patch_p2_state_names.insert(
                active_patch_p2_state_names.end(),
                {
                    "patch_cover",
                    "patch_shelter",
                    "patch_movement_cost",
                }
            );
        }
        for (
            size_t index = 0;
            index < active_patch_p2_state_names.size();
            ++index
        ) {
            TORCH_CHECK(
                patch_p2_state_tensors.at(index).scalar_type() == torch::kFloat32
                    && patch_p2_state_tensors.at(index).sizes()
                        == torch::IntArrayRef({cells, height, width}),
                active_patch_p2_state_names.at(index),
                " must be float32[cells,H,W]"
            );
        }
        TORCH_CHECK(
            patch_p2_scratch_tensors.at(0).scalar_type() == torch::kFloat32
                && patch_p2_scratch_tensors.at(0).sizes()
                    == torch::IntArrayRef({cells, 4, height, width}),
            "Patch P2 scratch must be float32[cells,4,H,W]"
        );
        TORCH_CHECK(
            patch_p2_stats_tensors.at(0).scalar_type() == torch::kFloat64
                && patch_p2_stats_tensors.at(0).sizes()
                    == torch::IntArrayRef({
                        cells,
                        generation_count * ticks,
                        eiel::cuda_architecture_b::persistent_patch_p2::kStatCount,
                    }),
            "Patch P2 stats must be float64[cells,generation_count*ticks,25]"
        );
        if (patch_p2_stats_tensors.size() == 2) {
            TORCH_CHECK(
                patch_p2_full_mode
                    && compact_summary_planes_enabled
                    && patch_p2_stats_tensors.at(1).scalar_type()
                        == torch::kFloat64
                    && patch_p2_stats_tensors.at(1).sizes()
                        == torch::IntArrayRef({
                            generation_count,
                            cells,
                            controller::kPatchCompactMetricCount,
                        }),
                "Patch P2 compact metrics must be "
                "float64[generation_count,cells,19]"
            );
        }
        std::vector<std::pair<const char*, torch::Tensor>> patch_p2_arrays{
            {"patch_food_amount", patch_p2_state_tensors.at(0)},
            {"patch_fertility", patch_p2_state_tensors.at(1)},
            {"patch_scent", patch_p2_state_tensors.at(2)},
            {"patch_waste", patch_p2_state_tensors.at(3)},
            {"patch_scent_emit_pending", patch_p2_state_tensors.at(4)},
            {"patch_waste_emit_pending", patch_p2_state_tensors.at(5)},
            {"patch_density", patch_p2_state_tensors.at(6)},
            {"agent_grid", agent_grid},
            {"food_grid", food_grid},
            {"Patch P2 scratch", patch_p2_scratch_tensors.at(0)},
            {"Patch P2 stats", patch_p2_stats_tensors.at(0)},
        };
        if (patch_p2_stats_tensors.size() == 2) {
            patch_p2_arrays.emplace_back(
                "Patch P2 compact metrics",
                patch_p2_stats_tensors.at(1)
            );
        }
        if (patch_p2_full_mode) {
            patch_p2_arrays.insert(
                patch_p2_arrays.end() - 2,
                {
                    {"patch_cover", patch_p2_state_tensors.at(7)},
                    {"patch_shelter", patch_p2_state_tensors.at(8)},
                    {"patch_movement_cost", patch_p2_state_tensors.at(9)},
                }
            );
        }
        for (size_t left = 0; left < patch_p2_arrays.size(); ++left) {
            for (size_t right = left + 1; right < patch_p2_arrays.size(); ++right) {
                TORCH_CHECK(
                    patch_p2_arrays.at(left).second.data_ptr()
                        != patch_p2_arrays.at(right).second.data_ptr(),
                    patch_p2_arrays.at(left).first,
                    " and ",
                    patch_p2_arrays.at(right).first,
                    " must not alias"
                );
            }
        }
    }
    if (generation_turnover_enabled) {
        TORCH_CHECK(
            capacity <= turnover::kMaximumCapacity,
            "bounded flattened turnover supports capacity <= ",
            turnover::kMaximumCapacity
        );
        for (int64_t index = 0; index < 7; ++index) {
            TORCH_CHECK(
                turnover_output_tensors.at(index).scalar_type() == torch::kInt64
                    && turnover_output_tensors.at(index).sizes()
                        == torch::IntArrayRef({cells, capacity}),
                "turnover identifier outputs must be int64[cells,capacity]"
            );
        }
        TORCH_CHECK(
            turnover_output_tensors.at(7).scalar_type() == torch::kInt64
                && turnover_output_tensors.at(7).sizes()
                    == torch::IntArrayRef({cells, 8}),
            "turnover counts must be int64[cells,8]"
        );
        TORCH_CHECK(
            turnover_output_tensors.at(8).scalar_type() == torch::kInt32
                && turnover_output_tensors.at(8).sizes()
                    == torch::IntArrayRef({cells}),
            "turnover status must be int32[cells]"
        );
        for (const int64_t index : {9, 10}) {
            TORCH_CHECK(
                turnover_output_tensors.at(index).scalar_type() == torch::kInt64
                    && turnover_output_tensors.at(index).sizes()
                        == torch::IntArrayRef({cells}),
                "turnover next-boundary outputs must be int64[cells]"
            );
        }
        const int64_t causal_capacity =
            turnover_output_tensors.at(11).size(1);
        TORCH_CHECK(
            (causal_capacity == 0
             || causal_capacity == genome_v2_causal_output::kRecordCapacity)
                && turnover_output_tensors.at(11).scalar_type() == torch::kInt64
                && turnover_output_tensors.at(11).sizes()
                    == torch::IntArrayRef(
                        {cells,
                         causal_capacity,
                         genome_v2_causal_output::kRecordWidth}
                    )
                && turnover_output_tensors.at(12).scalar_type() == torch::kBool
                && turnover_output_tensors.at(12).sizes()
                    == torch::IntArrayRef({cells, causal_capacity})
                && turnover_output_tensors.at(13).scalar_type() == torch::kInt64
                && turnover_output_tensors.at(13).sizes()
                    == torch::IntArrayRef({cells})
                && turnover_output_tensors.at(14).scalar_type() == torch::kInt64
                && turnover_output_tensors.at(14).sizes()
                    == torch::IntArrayRef({cells})
                && turnover_output_tensors.at(15).scalar_type() == torch::kBool
                && turnover_output_tensors.at(15).sizes()
                    == torch::IntArrayRef({cells}),
            "turnover Genome V2 causal outputs have an invalid bounded ABI"
        );
    }
    const int64_t side = 2 * view_radius + 1;
#ifndef EIEL_CUDA_ARCHITECTURE_B_OBSERVATION_CHANNEL_COUNT
#define EIEL_CUDA_ARCHITECTURE_B_OBSERVATION_CHANNEL_COUNT \
    (patch_p2_full_mode ? 8 : 1)
#endif
#ifndef EIEL_CUDA_ARCHITECTURE_B_OBSERVATION_EXTRA_COLUMNS
#define EIEL_CUDA_ARCHITECTURE_B_OBSERVATION_EXTRA_COLUMNS 0
#endif
    const int64_t observation_size =
        EIEL_CUDA_ARCHITECTURE_B_OBSERVATION_CHANNEL_COUNT * side * side + 2
        + EIEL_CUDA_ARCHITECTURE_B_OBSERVATION_EXTRA_COLUMNS;
    const int64_t packed_width = hidden_size * observation_size
        + hidden_size
        + hidden_size * hidden_size
        + hidden_size
        + action_count * hidden_size
        + action_count;
    TORCH_CHECK(mlp_pack.size(2) == packed_width, "fixed MLP packed width mismatch");
    if (lifecycle_state_enabled) {
        const auto& birth = ordinary_birth_state_tensors;
        const auto& torch_state = birth.at(85);
        TORCH_CHECK(
            torch_state.scalar_type() == torch::kUInt8
                && torch_state.dim() == 2
                && torch_state.size(0) == cells
                && torch_state.size(1) == torch_rng_pack::kFullStateBytes,
            "birth policy Torch RNG state must be canonical uint8[cells,5056]"
        );
        const int64_t torch_state_bytes = torch_state.size(1);
        const int64_t hmem_width = birth.at(84).size(2);
        const auto generic_birth_layout =
            eiel::cuda_architecture_b::flattened_birth_scratch_v2::make_layout(
                max_births_per_tick,
                capacity,
                packed_width,
                hmem_width,
                torch_state_bytes
            );
        TORCH_CHECK(
            generic_birth_layout.ok(),
            "runtime-sized flattened birth scratch layout is invalid"
        );
        const int64_t legacy_birth_scratch_bytes =
            eiel::cuda_architecture_b::flattened_controller_v1::
                kBirthScratchBytesPerCell;
        TORCH_CHECK(
            generic_birth_layout.required_bytes
                <= std::numeric_limits<int64_t>::max()
                    - legacy_birth_scratch_bytes,
            "runtime-sized flattened birth scratch byte extent overflows int64"
        );
        const int64_t required_birth_scratch_bytes =
            legacy_birth_scratch_bytes + generic_birth_layout.required_bytes
            + (diploid_state_enabled ? 8448 : 0);
        TORCH_CHECK(
            ordinary_birth_scratch_tensors.at(0).size(1)
                == required_birth_scratch_bytes,
            "flattened lifecycle scratch byte extent does not match the "
            "qualified runtime birth layout"
        );
        const auto& delta_tape = birth.at(86);
        const bool explicit_rngv2_birth_coordinates =
            patch_p2_config.contains("tw_rngv2_semantics_version")
            && patch_p2_config.contains("tw_rngv2_world_uid")
            && patch_p2_config.contains("tw_rngv2_world_uid_by_cell")
            && patch_p2_config.contains("tw_rngv2_namespace_key64")
            && patch_p2_config.contains("tw_rngv2_semantic_epoch")
            && pybind11::cast<int32_t>(
                patch_p2_config["tw_rngv2_semantics_version"]
            ) == 2;
        TORCH_CHECK(
            delta_tape.scalar_type() == torch::kFloat32
                && delta_tape.dim() == 3
                && delta_tape.size(0) == cells
                && (
                    explicit_rngv2_birth_coordinates
                        ? delta_tape.size(1) == 0
                        : delta_tape.size(1) > 0
                )
                && (
                    delta_tape.size(2) == packed_width
                    || delta_tape.size(2) == 1
                ),
            "birth policy delta tape must be "
            "float32[cells,policy_tape_capacity,packed_width|1] with zero "
            "capacity only for explicit RNG V2"
        );
        const int64_t policy_tape_capacity = delta_tape.size(1);
        const int64_t policy_mutation_tape_width = delta_tape.size(2);
        const int64_t policy_crossover_packed_row_bytes = (packed_width + 7) / 8;
        const auto check_birth_policy_tensor = [&birth](
            const int64_t index,
            const c10::ScalarType dtype,
            const torch::IntArrayRef shape,
            const char* name
        ) {
            const auto& tensor = birth.at(index);
            TORCH_CHECK(
                tensor.scalar_type() == dtype && tensor.sizes() == shape,
                name,
                " has an invalid dtype, shape, or extent"
            );
        };
        check_birth_policy_tensor(
            87,
            torch::kBool,
            {cells, policy_tape_capacity, policy_mutation_tape_width},
            "birth policy mutation mask tape"
        );
        check_birth_policy_tensor(
            88,
            torch::kUInt8,
            {cells, policy_tape_capacity, policy_crossover_packed_row_bytes},
            "birth packed policy crossover mask tape"
        );
        {
            const auto& post_state_tape = birth.at(89);
            const bool replay_backed_post_state =
                policy_mutation_tape_width == 1
                && post_state_tape.scalar_type() == torch::kUInt8
                && post_state_tape.sizes()
                    == torch::IntArrayRef({cells, policy_tape_capacity, 0});
            TORCH_CHECK(
                replay_backed_post_state
                    || (
                        post_state_tape.scalar_type() == torch::kUInt8
                        && post_state_tape.sizes()
                            == torch::IntArrayRef(
                                {cells, policy_tape_capacity,
                                 torch_rng_pack::kPackedStateBytes}
                            )
                    ),
                "birth packed policy post-RNG-state tape has an invalid "
                "dtype, shape, or extent"
            );
        }
        check_birth_policy_tensor(
            90,
            torch::kInt64,
            {cells, policy_tape_capacity},
            "birth policy semantic-draws tape"
        );
        for (const int64_t index : {91, 92}) {
            const auto& tensor = birth.at(index);
            TORCH_CHECK(
                tensor.scalar_type() == torch::kInt64
                    && tensor.size(0) == cells
                    && tensor.numel() == cells,
                "birth policy cursor/counter must contain one int64 per cell"
            );
        }
    }

    TORCH_CHECK(
        ordered_agent_ids.dim() == 3,
        "ordered_agent_ids output must be rank 3"
    );
    const int64_t output_ticks = ordered_agent_ids.size(1);
    TORCH_CHECK(
        output_ticks == 1 || output_ticks == ticks,
        "output tick storage must contain one rolling row or every tick"
    );
    if (ordinary_births_enabled) {
        const auto check_birth_output = [&](const int64_t index,
                                            const c10::ScalarType dtype,
                                            const torch::IntArrayRef shape,
                                            const char* name) {
            const auto& tensor = ordinary_birth_output_tensors.at(index);
            TORCH_CHECK(
                tensor.scalar_type() == dtype && tensor.sizes() == shape,
                name,
                " has an invalid dtype or shape"
            );
        };
        TORCH_CHECK(
            ordinary_birth_output_tensors.at(0).scalar_type() == torch::kInt64
                && ordinary_birth_output_tensors.at(0).sizes()
                    == torch::IntArrayRef({cells}),
            "birth queue-head output must be int64[cells]"
        );
        TORCH_CHECK(
            ordinary_birth_output_tensors.at(1).scalar_type() == torch::kInt32
                && ordinary_birth_output_tensors.at(1).sizes()
                    == torch::IntArrayRef({cells, capacity}),
            "birth cooldown output must be int32[cells,capacity]"
        );
        for (const int64_t index : {2, 3, 21, 22, 23}) {
            check_birth_output(
                index,
                torch::kInt64,
                {cells, output_ticks, max_births_per_tick},
                "birth int64 per-birth output"
            );
        }
        check_birth_output(
            4,
            torch::kInt32,
            {cells, output_ticks, max_births_per_tick, 2},
            "birth spawn-position output"
        );
        check_birth_output(
            5,
            torch::kFloat32,
            {cells, output_ticks, max_births_per_tick},
            "birth child-energy output"
        );
        const int64_t causal_capacity =
            ordinary_birth_output_tensors.at(58).size(1);
        TORCH_CHECK(
            (causal_capacity == 0
             || causal_capacity == genome_v2_causal_output::kRecordCapacity)
                && ordinary_birth_output_tensors.at(58).scalar_type()
                    == torch::kInt64
                && ordinary_birth_output_tensors.at(58).sizes()
                    == torch::IntArrayRef(
                        {cells,
                         causal_capacity,
                         genome_v2_causal_output::kRecordWidth}
                    )
                && ordinary_birth_output_tensors.at(59).scalar_type()
                    == torch::kBool
                && ordinary_birth_output_tensors.at(59).sizes()
                    == torch::IntArrayRef({cells, causal_capacity})
                && ordinary_birth_output_tensors.at(60).scalar_type()
                    == torch::kInt64
                && ordinary_birth_output_tensors.at(60).sizes()
                    == torch::IntArrayRef({cells})
                && ordinary_birth_output_tensors.at(61).scalar_type()
                    == torch::kInt64
                && ordinary_birth_output_tensors.at(61).sizes()
                    == torch::IntArrayRef({cells})
                && ordinary_birth_output_tensors.at(62).scalar_type()
                    == torch::kBool
                && ordinary_birth_output_tensors.at(62).sizes()
                    == torch::IntArrayRef({cells}),
            "ordinary-birth Genome V2 causal outputs have an invalid bounded ABI"
        );
        check_birth_output(
            6, torch::kInt64, {cells, output_ticks, capacity},
            "birth decision-parent output"
        );
        for (const int64_t index : {7, 8}) {
            check_birth_output(
                index, torch::kInt32, {cells, output_ticks, capacity},
                "birth int32 decision output"
            );
        }
        for (const int64_t index : {9, 10}) {
            check_birth_output(
                index, torch::kInt64, {cells, output_ticks, capacity},
                "birth int64 decision output"
            );
        }
        check_birth_output(
            11, torch::kInt32, {cells, output_ticks, capacity, 2},
            "birth selected-position output"
        );
        check_birth_output(
            12, torch::kBool, {cells, output_ticks, capacity},
            "birth selected-reservation output"
        );
        check_birth_output(
            13, torch::kInt64, {cells, output_ticks, 10},
            "birth reason-count output"
        );
        for (const int64_t index : {
                 14, 15, 16, 17, 18, 19, 24, 26, 27, 29, 30, 31, 32, 33,
                 36, 37, 38, 39,
             }) {
            check_birth_output(
                index, torch::kInt64, {cells, output_ticks},
                "birth int64 tick output"
            );
        }
        for (const int64_t index : {20, 25, 28, 34, 35, 43, 44}) {
            check_birth_output(
                index, torch::kInt32, {cells, output_ticks},
                "birth int32 tick output"
            );
        }
        check_birth_output(
            40, torch::kFloat64, {cells, output_ticks},
            "birth mutation-total output"
        );
        check_birth_output(
            41, torch::kBool, {cells, output_ticks, max_births_per_tick},
            "birth mutation-occurred output"
        );
        check_birth_output(
            42, torch::kFloat32, {cells, output_ticks, max_births_per_tick},
            "birth mutation-delta output"
        );
        check_birth_output(
            45, torch::kUInt64, {cells, output_ticks, 4},
            "birth planner-state output"
        );
        check_birth_output(
            46, torch::kInt64, {cells, output_ticks, 2},
            "birth planner-buffer output"
        );
        for (const int64_t index : {47, 48}) {
            check_birth_output(
                index,
                torch::kInt64,
                {cells, output_ticks, max_births_per_tick},
                "birth mate per-birth output"
            );
        }
        check_birth_output(
            49, torch::kFloat64, {cells, capacity},
            "birth mate planned-energy output"
        );
        for (const int64_t index : {50, 51, 52}) {
            check_birth_output(
                index,
                torch::kFloat64,
                {cells, output_ticks, max_births_per_tick},
                "birth mate exact-energy output"
            );
        }
        check_birth_output(
            53, torch::kInt64, {cells, output_ticks, capacity},
            "birth decision-mate output"
        );
        for (const int64_t index : {54, 55}) {
            check_birth_output(
                index, torch::kInt32, {cells, output_ticks, capacity},
                "birth mate int32 decision output"
            );
        }
        check_birth_output(
            56, torch::kInt64, {cells, output_ticks, 12},
            "birth mate reason-count output"
        );
        check_birth_output(
            57, torch::kInt32, {cells, output_ticks, capacity},
            "birth mate decision-reason output"
        );
    }
    TORCH_CHECK(
        ordered_agent_ids.scalar_type() == torch::kInt64
            && ordered_agent_ids.sizes()
                == torch::IntArrayRef({cells, output_ticks, capacity}),
        "ordered_agent_ids output must be int64[cells,output_ticks,capacity]"
    );
    for (const auto& named : std::vector<std::pair<const char*, torch::Tensor>>{
             {"death_agent_ids", death_agent_ids},
             {"death_slots", death_slots},
             {"survivor_agent_ids", survivor_agent_ids},
         }) {
        TORCH_CHECK(
            named.second.scalar_type() == torch::kInt64
                && named.second.sizes()
                    == torch::IntArrayRef({cells, output_ticks, capacity}),
            named.first,
            " output must be int64[cells,output_ticks,capacity]"
        );
    }
    TORCH_CHECK(
        death_reason_codes.scalar_type() == torch::kInt32
            && death_reason_codes.sizes()
                == torch::IntArrayRef({cells, output_ticks, capacity}),
        "death_reason_codes output must be int32[cells,output_ticks,capacity]"
    );
    for (const auto& named : std::vector<std::pair<const char*, torch::Tensor>>{
             {"ordered_slots", ordered_slots},
             {"row_actions", row_actions},
             {"actions_by_slot", actions_by_slot},
         }) {
        TORCH_CHECK(
            named.second.scalar_type() == torch::kInt32
                && named.second.sizes()
                    == torch::IntArrayRef({cells, output_ticks, capacity}),
            named.first,
            " output must be int32[cells,output_ticks,capacity]"
        );
    }
    for (const auto& named : std::vector<std::pair<const char*, torch::Tensor>>{
             {"pre_energy", pre_energy},
             {"pre_health", pre_health},
             {"pre_injury", pre_injury},
         }) {
        TORCH_CHECK(
            named.second.scalar_type() == torch::kFloat32
                && named.second.sizes()
                    == torch::IntArrayRef({cells, output_ticks, capacity}),
            named.first,
            " output must be float32[cells,output_ticks,capacity]"
        );
    }
    TORCH_CHECK(
        pre_survival_steps.scalar_type() == torch::kInt32
            && pre_survival_steps.sizes()
                == torch::IntArrayRef({cells, output_ticks, capacity}),
        "pre_survival_steps output must be int32[cells,output_ticks,capacity]"
    );
    TORCH_CHECK(
        observations.scalar_type() == torch::kFloat32
            && observations.sizes() == torch::IntArrayRef(
                {cells, output_ticks, capacity, observation_size}
            ),
        "observations output has an invalid shape or dtype"
    );
    TORCH_CHECK(
        logits.scalar_type() == torch::kFloat32
            && logits.sizes() == torch::IntArrayRef(
                {cells, output_ticks, capacity, action_count}
            ),
        "logits output has an invalid shape or dtype"
    );
    TORCH_CHECK(
        active_count.scalar_type() == torch::kInt64
            && active_count.sizes()
                == torch::IntArrayRef({cells, output_ticks}),
        "active_count output must be int64[cells,output_ticks]"
    );
    TORCH_CHECK(
        runtime_bookkeeping_by_tick.scalar_type() == torch::kInt64
            && runtime_bookkeeping_by_tick.sizes()
                == torch::IntArrayRef({cells, generation_count * ticks, 5}),
        "runtime_bookkeeping_by_tick output must cover all resident ticks"
    );
    if (generation_summary_scalar_enabled) {
        TORCH_CHECK(
            generation_summary_scalar.scalar_type() == torch::kFloat64
                && generation_summary_scalar.sizes() == torch::IntArrayRef({
                    generation_count,
                    cells,
                    controller::kGenerationSummaryScalarWidthV1,
                }),
            "generation summary scalar output must be contiguous float64"
            "[generation_count,cells,31]"
        );
    }
    int64_t compact_reporting_row_capacity = 0;
    int64_t compact_reporting_retained_bytes_per_generation_cell = 0;
    if (compact_summary_planes_enabled) {
        TORCH_CHECK(
            output_ticks == ticks,
            "compact summary authority currently requires full tick outputs"
        );
        TORCH_CHECK(
            cells == 1,
            "the compact authority producer is qualified only for b1"
        );
        TORCH_CHECK(
            lifecycle_state_enabled
                && (generation_turnover_enabled || lifecycle_mode == 1),
            "the Flat g2 compact authority producer requires lifecycle and "
            "turnover or hybrid-overlapping lifecycle mode"
        );
        TORCH_CHECK(
            ordinary_births_enabled && generation_summary_scalar_enabled,
            "the Flat g2 compact authority producer requires births and generation summaries"
        );
        const torch::Tensor& validity = full_tick_output_tensors.at(
            compact_summary_plane_offset + compact_summary::kPlaneValidity
        );
        const torch::Tensor& defined_mask = full_tick_output_tensors.at(
            compact_summary_plane_offset + compact_summary::kPlaneDefinedMask
        );
        const torch::Tensor& status_codes = full_tick_output_tensors.at(
            compact_summary_plane_offset + compact_summary::kPlaneStatusCodes
        );
        const torch::Tensor& integer_values = full_tick_output_tensors.at(
            compact_summary_plane_offset + compact_summary::kPlaneIntegerValues
        );
        const torch::Tensor& float_values = full_tick_output_tensors.at(
            compact_summary_plane_offset + compact_summary::kPlaneFloatValues
        );
        const torch::Tensor& digest_values = full_tick_output_tensors.at(
            compact_summary_plane_offset + compact_summary::kPlaneDigestValues
        );
        const torch::Tensor& text_values = full_tick_output_tensors.at(
            compact_summary_plane_offset + compact_summary::kPlaneTextValues
        );
        for (const auto& named : std::vector<
                 std::pair<const char*, torch::Tensor>>{
                 {"compact summary validity", validity},
                 {"compact summary defined_mask", defined_mask},
                 {"compact summary status_codes", status_codes},
                 {"compact summary integer_values", integer_values},
                 {"compact summary float_values", float_values},
                 {"compact summary digest_values", digest_values},
                 {"compact summary text_values", text_values},
             }) {
            check_cuda_contiguous(active_agent_ids, named.second, named.first);
        }
        TORCH_CHECK(
            validity.scalar_type() == torch::kUInt8
                && validity.sizes()
                    == torch::IntArrayRef({generation_count, cells}),
            "compact summary validity must be uint8[generation_count,cells]"
        );
        TORCH_CHECK(
            defined_mask.scalar_type() == torch::kUInt64
                && defined_mask.sizes() == torch::IntArrayRef({
                    generation_count,
                    cells,
                    compact_summary::kDefinedMaskFieldCountV2,
                }),
            "compact summary defined_mask must be "
            "uint64[generation_count,cells,1]"
        );
        TORCH_CHECK(
            status_codes.scalar_type() == torch::kInt32
                && status_codes.sizes() == torch::IntArrayRef({
                    generation_count,
                    cells,
                    compact_summary::kStatusFieldCountV2,
                }),
            "compact summary status_codes must be "
            "int32[generation_count,cells,12]"
        );
        TORCH_CHECK(
            integer_values.scalar_type() == torch::kInt64
                && integer_values.sizes() == torch::IntArrayRef({
                    generation_count,
                    cells,
                    compact_summary::kIntegerFieldCountV2,
                }),
            "compact summary integer_values must be "
            "int64[generation_count,cells,76]"
        );
        TORCH_CHECK(
            float_values.scalar_type() == torch::kFloat64
                && float_values.sizes() == torch::IntArrayRef({
                    generation_count,
                    cells,
                    compact_summary::kFloatFieldCountV2,
                }),
            "compact summary float_values must be "
            "float64[generation_count,cells,21]"
        );
        TORCH_CHECK(
            digest_values.scalar_type() == torch::kUInt8
                && digest_values.sizes() == torch::IntArrayRef({
                    generation_count,
                    cells,
                    compact_summary::kDigestFieldCountV2,
                    compact_summary::kDigestBytesV2,
                }),
            "compact summary digest_values must be "
            "uint8[generation_count,cells,10,32]"
        );
        TORCH_CHECK(
            text_values.scalar_type() == torch::kUInt8
                && text_values.sizes() == torch::IntArrayRef({
                    generation_count,
                    cells,
                    compact_summary::kTextFieldCountV2,
                    compact_summary::kTextRecordBytesV2,
                }),
            "compact summary text_values must be "
            "uint8[generation_count,cells,1,65]"
        );
    }
    TORCH_CHECK(
        compact_summary_reporting_tensors.size()
            == (compact_summary_planes_enabled ? 6 : 0),
        "compact_summary_reporting_tensors must be empty when compact planes "
        "are disabled and contain retained bytes, slot-map scratch, effective-"
        "fitness scratch, and three event cursors when enabled"
    );
    if (compact_summary_planes_enabled) {
        TORCH_CHECK(
            capacity <= std::numeric_limits<int32_t>::max(),
            "compact summary reporting slot capacity exceeds int32"
        );
        const auto& reporting_retained =
            compact_summary_reporting_tensors.at(0);
        const auto& reporting_slot_map =
            compact_summary_reporting_tensors.at(1);
        const auto& reporting_effective_fitness =
            compact_summary_reporting_tensors.at(2);
        const auto& event_cursor_start =
            compact_summary_reporting_tensors.at(3);
        const auto& event_cursor_pre_turnover =
            compact_summary_reporting_tensors.at(4);
        const auto& event_cursor_end =
            compact_summary_reporting_tensors.at(5);
        for (const auto& named : std::vector<
                 std::pair<const char*, torch::Tensor>>{
                 {"compact summary reporting retained bytes", reporting_retained},
                 {"compact summary reporting slot map", reporting_slot_map},
                 {
                     "compact summary reporting effective fitness",
                     reporting_effective_fitness,
                 },
                 {"compact summary event cursor start", event_cursor_start},
                 {
                     "compact summary event cursor pre-turnover",
                     event_cursor_pre_turnover,
                 },
                 {"compact summary event cursor end", event_cursor_end},
             }) {
            check_cuda_contiguous(active_agent_ids, named.second, named.first);
        }
        TORCH_CHECK(
            reporting_effective_fitness.scalar_type() == torch::kFloat64
                && reporting_effective_fitness.dim() == 3
                && reporting_effective_fitness.size(0) == generation_count
                && reporting_effective_fitness.size(1) == cells
                && reporting_effective_fitness.size(2) > 0
                && reporting_effective_fitness.size(2)
                    <= reporting_authority::kMaximumDynamicRowCapacityV1,
            "compact summary reporting effective fitness must be contiguous "
            "float64[generation_count,cells,runtime_row_capacity]"
        );
        compact_reporting_row_capacity = reporting_effective_fitness.size(2);
        compact_reporting_retained_bytes_per_generation_cell =
            static_cast<int64_t>(
                reporting_authority::retained_bytes_per_generation_cell_v1(
                    static_cast<int32_t>(compact_reporting_row_capacity)
                )
            );
        TORCH_CHECK(
            reporting_retained.scalar_type() == torch::kUInt8
                && reporting_retained.sizes() == torch::IntArrayRef({
                    generation_count,
                    cells,
                    compact_reporting_retained_bytes_per_generation_cell,
                }),
            "compact summary reporting retained bytes must be contiguous "
            "uint8[generation_count,cells,runtime_retained_bytes]"
        );
        TORCH_CHECK(
            reporting_slot_map.scalar_type() == torch::kInt32
                && reporting_slot_map.sizes() == torch::IntArrayRef({
                    generation_count,
                    cells,
                    capacity,
                }),
            "compact summary reporting slot map must be contiguous "
            "int32[generation_count,cells,capacity]"
        );
        for (const auto& named : std::vector<
                 std::pair<const char*, torch::Tensor>>{
                 {"event_cursor_start", event_cursor_start},
                 {"event_cursor_pre_turnover", event_cursor_pre_turnover},
                 {"event_cursor_end", event_cursor_end},
             }) {
            TORCH_CHECK(
                named.second.scalar_type() == torch::kInt64
                    && named.second.sizes()
                        == torch::IntArrayRef({generation_count, cells}),
                named.first,
                " must be contiguous int64[generation_count,cells]"
            );
        }
        for (const double value : {
                 reporting_parent_fraction,
                 reporting_episode0_fitness_weight,
                 reporting_episode0_pred_err_fitness_weight,
                 reporting_episode0_thrashing_fitness_weight,
                 reporting_episode0_drive_jitter_fitness_weight,
                 reporting_episode0_selfharm_fitness_weight,
                 reporting_coherence_thrashing_weight,
                 reporting_coherence_drive_jitter_weight,
                 reporting_coherence_selfharm_weight,
                 reporting_coherence_pred_err_weight,
                 reporting_coherence_reset_shock_weight,
                 reporting_coherence_episode0_pred_err_weight,
                 reporting_late_pred_cost_weight,
             }) {
            TORCH_CHECK(
                std::isfinite(value),
                "compact summary reporting fitness values must be finite"
            );
        }
        TORCH_CHECK(
            reporting_parent_fraction >= 0.0
                && reporting_parent_fraction <= 1.0,
            "reporting_parent_fraction must be in [0, 1]"
        );
    }
    TORCH_CHECK(
        compact_summary_checkpoint_descriptor_specs.defined()
            && !compact_summary_checkpoint_descriptor_specs.is_cuda()
            && compact_summary_checkpoint_descriptor_specs.is_contiguous()
            && compact_summary_checkpoint_descriptor_specs.is_pinned()
            && compact_summary_checkpoint_descriptor_specs.scalar_type()
                == torch::kInt64
            && compact_summary_checkpoint_descriptor_specs.dim() == 2
            && compact_summary_checkpoint_descriptor_specs.size(1)
                == checkpoint_runtime::kDescriptorSpecWidthV1,
        "compact checkpoint descriptor specs must be contiguous pinned CPU "
        "int64[descriptor_count,13]"
    );
    const int64_t compact_checkpoint_descriptor_count =
        compact_summary_checkpoint_descriptor_specs.size(0);
    TORCH_CHECK(
        compact_checkpoint_descriptor_count
            <= std::numeric_limits<int32_t>::max(),
        "compact checkpoint descriptor count exceeds int32"
    );
    const bool compact_checkpoint_hash_enabled =
        compact_checkpoint_descriptor_count > 0;
    host_template_runtime::HostTemplateRuntimeV1 compact_host_template_runtime;
    TORCH_CHECK(
        compact_checkpoint_hash_enabled
            ? (
                  compact_summary_planes_enabled
                  && !compact_summary_checkpoint_source_tensors.empty()
                  && compact_summary_checkpoint_output_tensors.size() == 4
              )
            : (
                  compact_summary_checkpoint_source_tensors.empty()
                  && compact_summary_checkpoint_output_tensors.empty()
              ),
        "compact checkpoint hashing must provide specs, source tensors, and "
        "four outputs together, and requires compact planes"
    );
    TORCH_CHECK(
        compact_summary_settlement_tensors.size()
            == (compact_summary_planes_enabled ? kSettlementTensorCountV1 : 0),
        "compact_summary_settlement_tensors must be empty when compact planes "
        "are disabled and contain the complete preallocated settlement ABI when enabled"
    );
    if (compact_summary_planes_enabled) {
        TORCH_CHECK(
            compact_checkpoint_hash_enabled,
            "the Flat g2 compact authority producer requires checkpoint hashing"
        );
        const auto& authority_device = compact_summary_settlement_tensors.at(
            kSettlementAuthorityDeviceV1
        );
        const auto& producer_status_device =
            compact_summary_settlement_tensors.at(kSettlementProducerStatusDeviceV1);
        const auto& lifecycle_digest_device = compact_summary_settlement_tensors.at(
            kSettlementLifecycleDigestDeviceV1
        );
        const auto& lifecycle_status_device = compact_summary_settlement_tensors.at(
            kSettlementLifecycleStatusDeviceV1
        );
        const auto& lifecycle_validity_device = compact_summary_settlement_tensors.at(
            kSettlementLifecycleValidityDeviceV1
        );
        const auto& reduction_device = compact_summary_settlement_tensors.at(
            kSettlementReductionDeviceV1
        );
        const auto& metadata_retention_device =
            compact_summary_settlement_tensors.at(
                kSettlementMetadataRetentionDeviceV1
            );
        const auto& metadata_initial_host =
            compact_summary_settlement_tensors.at(
                kSettlementMetadataInitialHostV1
            );
        const auto& completed_generations_host =
            compact_summary_settlement_tensors.at(
                kSettlementCompletedGenerationsHostV1
            );
        check_cuda_contiguous(
            active_agent_ids, authority_device, "compact authority device"
        );
        check_cuda_contiguous(
            active_agent_ids,
            producer_status_device,
            "compact producer status device"
        );
        check_cuda_contiguous(
            active_agent_ids,
            lifecycle_digest_device,
            "compact lifecycle digest device"
        );
        check_cuda_contiguous(
            active_agent_ids,
            lifecycle_status_device,
            "compact lifecycle status device"
        );
        check_cuda_contiguous(
            active_agent_ids,
            lifecycle_validity_device,
            "compact lifecycle validity device"
        );
        check_cuda_contiguous(
            active_agent_ids, reduction_device, "compact reduction device"
        );
        check_cuda_contiguous(
            active_agent_ids,
            metadata_retention_device,
            "compact metadata retention device"
        );
        TORCH_CHECK(
            authority_device.scalar_type() == torch::kUInt8
                && authority_device.sizes() == torch::IntArrayRef({
                    generation_count,
                    cells,
                    1256,
                }),
            "compact authority device must be uint8[generation_count,cells,1256]"
        );
        TORCH_CHECK(
            producer_status_device.scalar_type() == torch::kInt32
                && producer_status_device.sizes()
                    == torch::IntArrayRef({generation_count, cells}),
            "compact producer status device must be int32[generation_count,cells]"
        );
        TORCH_CHECK(
            lifecycle_digest_device.scalar_type() == torch::kUInt8
                && lifecycle_digest_device.sizes() == torch::IntArrayRef({
                    generation_count,
                    cells,
                    lifecycle_digest::kDigestBytesV1,
                })
                && lifecycle_status_device.scalar_type() == torch::kInt32
                && lifecycle_status_device.sizes()
                    == torch::IntArrayRef({generation_count, cells})
                && lifecycle_validity_device.scalar_type() == torch::kUInt8
                && lifecycle_validity_device.sizes()
                    == torch::IntArrayRef({generation_count, cells}),
            "compact lifecycle settlement device tensors have an invalid contract"
        );
        TORCH_CHECK(
            reduction_device.scalar_type() == torch::kUInt8
                && reduction_device.sizes() == torch::IntArrayRef({
                    generation_count,
                    cells,
                    static_cast<std::int64_t>(sizeof(reductions::ReductionResultV1)),
                }),
            "compact reduction device must be uint8[generation_count,cells,456]"
        );
        TORCH_CHECK(
            metadata_retention_device.scalar_type() == torch::kUInt8
                && metadata_retention_device.sizes() == torch::IntArrayRef({
                    generation_count,
                    cells,
                    3072,
                }),
            "compact metadata retention device must be "
            "uint8[generation_count,cells,3072]"
        );
        TORCH_CHECK(
            !metadata_initial_host.is_cuda()
                && metadata_initial_host.is_contiguous()
                && metadata_initial_host.is_pinned()
                && metadata_initial_host.scalar_type() == torch::kInt64
                && metadata_initial_host.sizes()
                    == torch::IntArrayRef({cells, 5}),
            "compact initial metadata must be pinned CPU int64[cells,5]"
        );
        TORCH_CHECK(
            !completed_generations_host.is_cuda()
                && completed_generations_host.is_contiguous()
                && completed_generations_host.is_pinned()
                && completed_generations_host.scalar_type() == torch::kInt64
                && completed_generations_host.sizes()
                    == torch::IntArrayRef({cells}),
            "compact completed generations must be pinned CPU int64[cells]"
        );
        for (std::size_t index = kSettlementMetadataInitialHostV1;
             index < compact_summary_settlement_tensors.size();
             ++index) {
            const auto& tensor = compact_summary_settlement_tensors.at(index);
            TORCH_CHECK(
                !tensor.is_cuda() && tensor.is_contiguous()
                    && tensor.is_pinned(),
                "compact settlement host tensors must be contiguous pinned CPU storage"
            );
        }
        TORCH_CHECK(
            compact_summary_settlement_tensors.at(kSettlementAuthorityHostV1)
                        .scalar_type()
                    == torch::kUInt8
                && compact_summary_settlement_tensors.at(kSettlementAuthorityHostV1)
                       .sizes()
                    == authority_device.sizes(),
            "compact authority host staging must match the device authority"
        );
        TORCH_CHECK(
            compact_summary_settlement_tensors.at(kSettlementReportingHostV1)
                    .sizes()
                == compact_summary_reporting_tensors.at(0).sizes(),
            "compact reporting host staging shape mismatch"
        );
        for (std::size_t index = 0; index < 4; ++index) {
            TORCH_CHECK(
                compact_summary_settlement_tensors.at(
                    kSettlementCheckpointRawHostV1 + index
                ).sizes()
                        == compact_summary_checkpoint_output_tensors.at(index).sizes()
                    && compact_summary_settlement_tensors.at(
                           kSettlementCheckpointRawHostV1 + index
                       ).scalar_type()
                        == compact_summary_checkpoint_output_tensors.at(index)
                               .scalar_type(),
                "compact checkpoint host staging contract mismatch"
            );
        }
        TORCH_CHECK(
            compact_summary_settlement_tensors.at(kSettlementGenerationSummaryHostV1)
                        .scalar_type()
                    == torch::kFloat64
                && compact_summary_settlement_tensors.at(
                       kSettlementGenerationSummaryHostV1
                   ).sizes()
                    == generation_summary_scalar.sizes(),
            "compact generation summary host staging contract mismatch"
        );
        const auto& patch_metrics_host =
            compact_summary_settlement_tensors.at(kSettlementPatchMetricsHostV1);
        TORCH_CHECK(
            patch_metrics_host.scalar_type() == torch::kFloat64
                && patch_metrics_host.sizes() == torch::IntArrayRef({
                    generation_count,
                    cells,
                    controller::kPatchCompactMetricCount,
                }),
            "compact Patch metrics host staging must be "
            "float64[generation_count,cells,19]"
        );
        const std::pair<std::size_t, std::size_t> lifecycle_mirrors[] = {
            {kSettlementLifecycleDigestDeviceV1, kSettlementLifecycleDigestHostV1},
            {kSettlementLifecycleStatusDeviceV1, kSettlementLifecycleStatusHostV1},
            {
                kSettlementLifecycleValidityDeviceV1,
                kSettlementLifecycleValidityHostV1,
            },
        };
        for (const auto [device_index, host_index] : lifecycle_mirrors) {
            TORCH_CHECK(
                compact_summary_settlement_tensors.at(host_index).sizes()
                        == compact_summary_settlement_tensors.at(device_index).sizes()
                    && compact_summary_settlement_tensors.at(host_index).scalar_type()
                        == compact_summary_settlement_tensors.at(device_index)
                               .scalar_type(),
                "compact lifecycle host staging contract mismatch"
            );
        }
        for (const auto [device_index, host_index] : {
                 std::pair<std::size_t, std::size_t>{
                     kSettlementReductionDeviceV1,
                     kSettlementReductionHostV1,
                 },
                 std::pair<std::size_t, std::size_t>{
                     kSettlementMetadataRetentionDeviceV1,
                     kSettlementMetadataRetentionHostV1,
                 },
             }) {
            TORCH_CHECK(
                compact_summary_settlement_tensors.at(host_index).sizes()
                        == compact_summary_settlement_tensors.at(device_index).sizes()
                    && compact_summary_settlement_tensors.at(host_index).scalar_type()
                        == compact_summary_settlement_tensors.at(device_index)
                               .scalar_type(),
                "compact retained metadata host staging contract mismatch"
            );
        }
        const auto& checkpoint_due_host =
            compact_summary_settlement_tensors.at(kSettlementCheckpointDueHostV1);
        TORCH_CHECK(
            checkpoint_due_host.scalar_type() == torch::kBool
                && checkpoint_due_host.sizes()
                    == torch::IntArrayRef({generation_count, cells}),
            "compact checkpoint due authority must be pinned CPU "
            "bool[generation_count,cells]"
        );
        const auto& import_digest_host =
            compact_summary_settlement_tensors.at(kSettlementImportDigestHostV1);
        TORCH_CHECK(
            import_digest_host.scalar_type() == torch::kUInt8
                && import_digest_host.sizes() == torch::IntArrayRef({cells, 32}),
            "compact import digest authority must be pinned CPU uint8[cells,32]"
        );
        const auto& host_template_programs =
            compact_summary_settlement_tensors.at(
                kSettlementHostTemplateProgramsHostV1
            );
        const auto& host_template_tokens =
            compact_summary_settlement_tensors.at(
                kSettlementHostTemplateTokensHostV1
            );
        const auto& host_template_parts =
            compact_summary_settlement_tensors.at(
                kSettlementHostTemplatePartsHostV1
            );
        const auto& host_template_constraints =
            compact_summary_settlement_tensors.at(
                kSettlementHostTemplateConstraintsHostV1
            );
        const auto& host_template_byte_pool =
            compact_summary_settlement_tensors.at(
                kSettlementHostTemplateBytePoolHostV1
            );
        const auto& host_token_resolver_plan =
            compact_summary_settlement_tensors.at(
                kSettlementHostTokenResolverPlanHostV1
            );
        TORCH_CHECK(
            host_template_programs.scalar_type() == torch::kUInt64
                && host_template_programs.dim() == 2
                && host_template_programs.size(0)
                    == generation_count * cells * 6
                && host_template_programs.size(1) == 13,
            "compact host-template programs must be pinned CPU "
            "uint64[generation_count*cells*6,13]"
        );
        for (const auto& named : std::vector<
                 std::pair<const char*, torch::Tensor>>{
                 {"tokens", host_template_tokens},
                 {"parts", host_template_parts},
                 {"constraints", host_template_constraints},
             }) {
            const int64_t expected_width =
                std::string(named.first) == "tokens" ? 13 : 5;
            TORCH_CHECK(
                named.second.scalar_type() == torch::kUInt64
                    && named.second.dim() == 2
                    && named.second.size(1) == expected_width,
                "compact host-template ", named.first,
                " must be a pinned CPU uint64 row matrix"
            );
        }
        TORCH_CHECK(
            host_template_byte_pool.scalar_type() == torch::kUInt8
                && host_template_byte_pool.dim() == 1,
            "compact host-template byte pool must be pinned CPU uint8[bytes]"
        );
        TORCH_CHECK(
            host_token_resolver_plan.scalar_type() == torch::kUInt64
                && host_token_resolver_plan.dim() == 2
                && host_token_resolver_plan.size(0)
                    == host_template_tokens.size(0)
                && host_token_resolver_plan.size(1) == 6,
            "compact host-token resolver plan must be pinned CPU "
            "uint64[token_count,6] aligned with host-template tokens"
        );
        const auto host_template_runtime_status =
            host_template_runtime::prepare_host_template_runtime_v1(
                host_template_runtime::HostTemplateWireViewV1{
                    host_template_runtime::kHostTemplateWireAbiVersionV1,
                    0,
                    static_cast<std::uint32_t>(generation_count),
                    static_cast<std::uint32_t>(cells),
                    {
                        host_template_programs.data_ptr<std::uint64_t>(),
                        static_cast<std::uint64_t>(
                            host_template_programs.size(0)
                        ),
                        13,
                        0,
                    },
                    {
                        host_template_tokens.data_ptr<std::uint64_t>(),
                        static_cast<std::uint64_t>(host_template_tokens.size(0)),
                        13,
                        0,
                    },
                    {
                        host_template_parts.data_ptr<std::uint64_t>(),
                        static_cast<std::uint64_t>(host_template_parts.size(0)),
                        5,
                        0,
                    },
                    {
                        host_template_constraints.data_ptr<std::uint64_t>(),
                        static_cast<std::uint64_t>(
                            host_template_constraints.size(0)
                        ),
                        5,
                        0,
                    },
                    {
                        host_template_byte_pool.data_ptr<std::uint8_t>(),
                        static_cast<std::uint64_t>(host_template_byte_pool.numel()),
                        0,
                    },
                    0,
                },
                &compact_host_template_runtime
            );
        TORCH_CHECK(
            host_template_runtime_status
                == host_template_runtime::HostTemplateRuntimeStatusV1::kSuccess,
            "compact host-template runtime preparation failed: ",
            host_template_runtime::host_template_runtime_status_name_v1(
                host_template_runtime_status
            )
        );
    }
    if (compact_checkpoint_hash_enabled) {
        for (const auto& tensor : compact_summary_checkpoint_source_tensors) {
            check_cuda_contiguous(
                active_agent_ids,
                tensor,
                "compact checkpoint source tensor"
            );
            TORCH_CHECK(
                tensor.dim() >= 1 && tensor.size(0) == cells,
                "compact checkpoint source tensors must be cell-major"
            );
        }
        const auto& raw_digests = compact_summary_checkpoint_output_tensors.at(0);
        const auto& canonical_digests =
            compact_summary_checkpoint_output_tensors.at(1);
        const auto& checkpoint_status =
            compact_summary_checkpoint_output_tensors.at(2);
        const auto& checkpoint_valid =
            compact_summary_checkpoint_output_tensors.at(3);
        for (const auto& named : std::vector<
                 std::pair<const char*, torch::Tensor>>{
                 {"compact checkpoint raw digests", raw_digests},
                 {"compact checkpoint canonical digests", canonical_digests},
                 {"compact checkpoint status", checkpoint_status},
                 {"compact checkpoint valid", checkpoint_valid},
             }) {
            check_cuda_contiguous(active_agent_ids, named.second, named.first);
        }
        TORCH_CHECK(
            raw_digests.scalar_type() == torch::kUInt8
                && raw_digests.sizes() == torch::IntArrayRef({
                    generation_count,
                    cells,
                    compact_checkpoint_descriptor_count,
                    checkpoint_hash::kDigestBytesV1,
                })
                && canonical_digests.scalar_type() == torch::kUInt8
                && canonical_digests.sizes() == raw_digests.sizes(),
            "compact checkpoint digest outputs must be contiguous "
            "uint8[generation_count,cells,descriptor_count,32]"
        );
        TORCH_CHECK(
            checkpoint_status.scalar_type() == torch::kInt32
                && checkpoint_status.sizes() == torch::IntArrayRef({
                    generation_count,
                    cells,
                    compact_checkpoint_descriptor_count,
                })
                && checkpoint_valid.scalar_type() == torch::kUInt8
                && checkpoint_valid.sizes() == checkpoint_status.sizes(),
            "compact checkpoint status/valid outputs must be contiguous "
            "[generation_count,cells,descriptor_count]"
        );
    }
    for (const auto& named : std::vector<std::pair<const char*, torch::Tensor>>{
             {"policy_refill_rows", policy_refill_rows},
             {"policy_cache_hit_rows", policy_cache_hit_rows},
             {"death_count", death_count},
             {"survivor_count", survivor_count},
         }) {
        TORCH_CHECK(
            named.second.scalar_type() == torch::kInt64
                && named.second.sizes()
                    == torch::IntArrayRef({cells, output_ticks}),
            named.first,
            " output must be int64[cells,output_ticks]"
        );
    }
    TORCH_CHECK(
        lifecycle_status.scalar_type() == torch::kInt32
            && lifecycle_status.sizes()
                == torch::IntArrayRef({cells, output_ticks}),
        "lifecycle_status output must be int32[cells,output_ticks]"
    );
    TORCH_CHECK(
        status.scalar_type() == torch::kInt32
            && status.sizes() == torch::IntArrayRef({cells}),
        "status output must be int32[cells]"
    );
    TORCH_CHECK(
        first_failure_records.scalar_type() == torch::kInt64
            && first_failure_records.sizes() == torch::IntArrayRef({
                cells,
                controller::kFirstFailureRecordWidthV1,
            }),
        "first_failure_records output must be int64[cells,10]"
    );

    TORCH_CHECK(
        full_tick_output_tensors.at(0).scalar_type() == torch::kInt32
            && full_tick_output_tensors.at(0).sizes()
                == torch::IntArrayRef({cells, output_ticks, capacity}),
        "chemistry order output has an invalid shape"
    );
    for (const int64_t index : {1, 3, 4, 6, 13, 14}) {
        TORCH_CHECK(
            full_tick_output_tensors.at(index).scalar_type() == torch::kInt64
                && full_tick_output_tensors.at(index).sizes()
                    == torch::IntArrayRef({cells, output_ticks}),
            "full tick int64 counter output has an invalid shape"
        );
    }
    for (const int64_t index : {2, 5, 7, 8, 15}) {
        TORCH_CHECK(
            full_tick_output_tensors.at(index).scalar_type() == torch::kInt32
                && full_tick_output_tensors.at(index).sizes()
                    == torch::IntArrayRef({cells, output_ticks}),
            "full tick status output has an invalid shape"
        );
    }
    TORCH_CHECK(
        full_tick_output_tensors.at(9).scalar_type() == torch::kInt64
            && full_tick_output_tensors.at(9).sizes()
                == torch::IntArrayRef({cells, output_ticks, 5}),
        "active mechanics count output has an invalid shape"
    );
    for (const int64_t index : {10, 11}) {
        TORCH_CHECK(
            full_tick_output_tensors.at(index).scalar_type() == torch::kInt64
                && full_tick_output_tensors.at(index).sizes()
                    == torch::IntArrayRef({cells, output_ticks, capacity}),
            "residual death int64 output has an invalid shape"
        );
    }
    TORCH_CHECK(
        full_tick_output_tensors.at(12).scalar_type() == torch::kInt32
            && full_tick_output_tensors.at(12).sizes()
                == torch::IntArrayRef({cells, output_ticks, capacity}),
        "residual death reason output has an invalid shape"
    );
    TORCH_CHECK(
        full_tick_output_tensors.at(16).scalar_type() == torch::kUInt64
            && full_tick_output_tensors.at(16).sizes()
                == torch::IntArrayRef({cells, output_ticks, 4}),
        "policy PCG64 state snapshot output has an invalid shape"
    );
    TORCH_CHECK(
        full_tick_output_tensors.at(17).scalar_type() == torch::kInt64
            && full_tick_output_tensors.at(17).sizes()
                == torch::IntArrayRef({cells, output_ticks, 2}),
        "policy PCG64 buffer snapshot output has an invalid shape"
    );
    TORCH_CHECK(
        full_tick_output_tensors.at(18).scalar_type() == torch::kInt64
            && full_tick_output_tensors.at(18).sizes()
                == torch::IntArrayRef({cells, output_ticks, 3}),
        "food spawn statistics output has an invalid shape"
    );

    const std::vector<c10::ScalarType> event_column_dtypes{
        torch::kInt32, torch::kInt64, torch::kInt64, torch::kInt64,
        torch::kInt64, torch::kFloat32, torch::kInt32, torch::kInt64,
        torch::kInt64, torch::kInt64, torch::kInt64, torch::kInt64,
        torch::kInt64, torch::kInt64, torch::kInt64, torch::kInt64,
        torch::kInt64, torch::kInt32, torch::kInt32, torch::kInt32,
        torch::kInt8, torch::kFloat32,
    };
    const int64_t event_capacity = event_state_tensors.at(0).size(1);
    for (int64_t index = 0; index < 22; ++index) {
        TORCH_CHECK(
            event_state_tensors.at(index).scalar_type()
                    == event_column_dtypes.at(index)
                && event_state_tensors.at(index).sizes()
                    == torch::IntArrayRef({cells, event_capacity}),
            "event tape column has an invalid shape or dtype"
        );
    }
    TORCH_CHECK(event_capacity > 0, "event capacity must be positive");
    TORCH_CHECK(
        event_state_tensors.at(22).scalar_type() == torch::kInt64
            && event_state_tensors.at(22).sizes() == torch::IntArrayRef({cells}),
        "event active count must be int64[cells]"
    );
    TORCH_CHECK(
        event_state_tensors.at(23).scalar_type() == torch::kInt32
            && event_state_tensors.at(23).sizes() == torch::IntArrayRef({cells}),
        "event status must be int32[cells]"
    );
    for (const int64_t index : {24, 25}) {
        TORCH_CHECK(
            event_state_tensors.at(index).scalar_type() == torch::kInt32
                && event_state_tensors.at(index).sizes()
                    == torch::IntArrayRef({cells, capacity}),
            "lineage event state has an invalid shape"
        );
    }
    const torch::Tensor& initial_agent_ids_by_row =
        event_state_tensors.at(26);
    const int64_t reporting_rows = event_state_tensors.at(27).size(1);
    TORCH_CHECK(
        initial_agent_ids_by_row.scalar_type() == torch::kInt64
            && initial_agent_ids_by_row.sizes()
                == torch::IntArrayRef({cells, reporting_rows}),
        "initial agent IDs by reporting row have an invalid shape"
    );
    for (const int64_t index : {27, 28}) {
        TORCH_CHECK(
            event_state_tensors.at(index).scalar_type() == torch::kInt32
                && event_state_tensors.at(index).sizes()
                    == torch::IntArrayRef({cells, reporting_rows}),
            "terminal reporting state has an invalid shape"
        );
    }
    TORCH_CHECK(
        event_state_tensors.at(29).scalar_type() == torch::kInt32
            && event_state_tensors.at(29).sizes()
                == torch::IntArrayRef({cells, output_ticks}),
        "event status-by-tick has an invalid shape"
    );
    TORCH_CHECK(
        event_state_tensors.at(30).scalar_type() == torch::kInt64
            && event_state_tensors.at(30).sizes()
                == torch::IntArrayRef({cells}),
        "event sequence base must be int64[cells]"
    );
    if (lifecycle_state_enabled) {
        for (const int64_t index : {111, 112}) {
            TORCH_CHECK(
                ordinary_birth_state_tensors.at(index).scalar_type()
                        == torch::kInt64
                    && ordinary_birth_state_tensors.at(index).sizes()
                        == torch::IntArrayRef({cells}),
                "ordinary birth cursor base must be int64[cells]"
            );
        }
        if (diploid_state_enabled) {
            TORCH_CHECK(
                ordinary_birth_state_tensors.at(113).scalar_type()
                        == torch::kInt32
                    && ordinary_birth_state_tensors.at(113).sizes()
                        == torch::IntArrayRef({cells}),
                "diploid inheritance mode must be int32[cells]"
            );
            for (const int64_t index : {114, 115}) {
                TORCH_CHECK(
                    ordinary_birth_state_tensors.at(index).scalar_type()
                            == torch::kBool
                        && ordinary_birth_state_tensors.at(index).sizes()
                            == torch::IntArrayRef({cells}),
                    "diploid inheritance flags must be bool[cells]"
                );
            }
            TORCH_CHECK(
                ordinary_birth_state_tensors.at(116).scalar_type()
                        == torch::kFloat64
                    && ordinary_birth_state_tensors.at(116).sizes()
                        == torch::IntArrayRef({cells}),
                "diploid mutation sigma multiplier must be float64[cells]"
            );
            TORCH_CHECK(
                ordinary_birth_state_tensors.at(117).scalar_type()
                        == torch::kFloat32
                    && ordinary_birth_state_tensors.at(117).sizes()
                        == torch::IntArrayRef({cells, 34, capacity}),
                "diploid allele storage must be float32[cells,34,capacity]"
            );
            TORCH_CHECK(
                ordinary_birth_state_tensors.at(118).scalar_type()
                        == torch::kFloat32
                    && ordinary_birth_state_tensors.at(118).sizes()
                        == torch::IntArrayRef({cells, capacity}),
                "resource-handling phenotype must be float32[cells,capacity]"
            );
            TORCH_CHECK(
                ordinary_birth_state_tensors.at(119).scalar_type()
                        == torch::kFloat32
                    && ordinary_birth_state_tensors.at(119).sizes()
                        == torch::IntArrayRef({cells, 34, capacity}),
                "diploid allele shadow must be float32[cells,34,capacity]"
            );
            TORCH_CHECK(
                ordinary_birth_state_tensors.at(120).scalar_type()
                        == torch::kFloat32
                    && ordinary_birth_state_tensors.at(120).sizes()
                        == torch::IntArrayRef({cells, capacity}),
                "resource-handling phenotype shadow must be "
                "float32[cells,capacity]"
            );
            TORCH_CHECK(
                ordinary_birth_state_tensors.at(121).scalar_type()
                        == torch::kUInt8
                    && ordinary_birth_state_tensors.at(121).sizes()
                        == torch::IntArrayRef({cells, capacity, 32}),
                "canonical hereditary digest state must be "
                "uint8[cells,capacity,32]"
            );
            if (genome_upgrade_v2_state_enabled) {
                TORCH_CHECK(
                    ordinary_birth_state_tensors.at(122).scalar_type()
                            == torch::kBool
                        && ordinary_birth_state_tensors.at(122).sizes()
                            == torch::IntArrayRef({cells}),
                    "GENOME-UPGRADE-V2 enable state must be bool[cells]"
                );
                for (const int64_t index : {123, 124}) {
                    TORCH_CHECK(
                        ordinary_birth_state_tensors.at(index).scalar_type()
                                == torch::kInt64
                            && ordinary_birth_state_tensors.at(index).sizes()
                                == torch::IntArrayRef({cells}),
                        "GENOME-UPGRADE-V2 mutation controls must be "
                        "int64[cells]"
                    );
                }
                const auto check_genome_v2_plane = [&](
                    const int64_t index,
                    const torch::IntArrayRef shape,
                    const char* name
                ) {
                    TORCH_CHECK(
                        ordinary_birth_state_tensors.at(index).scalar_type()
                                == torch::kInt64
                            && ordinary_birth_state_tensors.at(index).sizes()
                                == shape,
                        name
                    );
                };
                for (const int64_t index : {125, 132}) {
                    check_genome_v2_plane(
                        index,
                        {cells, 2, 12, 4, capacity},
                        "GENOME-UPGRADE-V2 allele planes must be "
                        "int64[cells,2,12,4,capacity]"
                    );
                }
                for (const int64_t index : {126, 133}) {
                    check_genome_v2_plane(
                        index,
                        {cells, 2, 12, capacity},
                        "GENOME-UPGRADE-V2 modifier planes must be "
                        "int64[cells,2,12,capacity]"
                    );
                }
                for (const int64_t index : {127, 134}) {
                    check_genome_v2_plane(
                        index,
                        {cells, 12, capacity},
                        "GENOME-UPGRADE-V2 product planes must be "
                        "int64[cells,12,capacity]"
                    );
                }
                for (const int64_t index : {128, 135}) {
                    check_genome_v2_plane(
                        index,
                        {cells, 10, capacity},
                        "GENOME-UPGRADE-V2 phenotype planes must be "
                        "int64[cells,10,capacity]"
                    );
                }
                for (
                    const int64_t index : {129, 130, 131, 136, 137, 138}
                ) {
                    check_genome_v2_plane(
                        index,
                        {cells, capacity},
                        "GENOME-UPGRADE-V2 scalar planes must be "
                        "int64[cells,capacity]"
                    );
                }
            }
        }
        const auto require_exact_alias = [](
            const torch::Tensor& left,
            const torch::Tensor& right,
            const char* name
        ) {
            TORCH_CHECK(
                left.data_ptr() == right.data_ptr()
                    && left.storage_offset() == right.storage_offset()
                    && left.sizes() == right.sizes()
                    && left.strides() == right.strides()
                    && left.scalar_type() == right.scalar_type(),
                name,
                " must be one exact authoritative tensor view"
            );
        };
        const auto& birth = ordinary_birth_state_tensors;
        if (generation_turnover_enabled) {
            const auto& turnover_state = turnover_state_tensors;
            require_exact_alias(
                turnover_state.at(turnover::kActiveAgentIds),
                active_agent_ids,
                "active_agent_ids"
            );
            require_exact_alias(
                turnover_state.at(turnover::kActiveAgentSlots),
                active_agent_slots,
                "active_agent_slots"
            );
            require_exact_alias(
                turnover_state.at(turnover::kActiveCountState),
                active_count_state,
                "active_count_state"
            );
        }
        require_exact_alias(birth.at(0), slot_to_agent_id, "slot_to_agent_id");
        require_exact_alias(birth.at(1), agent_id_to_slot, "agent_id_to_slot");
        require_exact_alias(birth.at(2), slot_tokens, "slot_tokens");
        require_exact_alias(birth.at(3), alive_mask, "alive_mask");
        require_exact_alias(birth.at(4), pos_y, "pos_y");
        require_exact_alias(birth.at(5), pos_x, "pos_x");
        require_exact_alias(birth.at(6), agent_grid, "agent_grid");
        require_exact_alias(birth.at(7), energy, "energy");
        require_exact_alias(birth.at(8), health, "health");
        require_exact_alias(birth.at(9), injury, "injury");
        require_exact_alias(birth.at(11), full_tick_state_tensors.at(0), "age");
        require_exact_alias(
            birth.at(13),
            birth.at(69),
            "repro_threshold"
        );
        require_exact_alias(
            birth.at(15),
            event_state_tensors.at(24),
            "lineage_founder_id"
        );
        require_exact_alias(
            birth.at(16),
            event_state_tensors.at(25),
            "lineage_depth"
        );
        require_exact_alias(
            birth.at(20),
            policy_energy_observation,
            "policy_energy_observation"
        );
        require_exact_alias(birth.at(21), survival_steps, "survival_steps");
        require_exact_alias(birth.at(22), rest_streak, "rest_streak");
        require_exact_alias(
            birth.at(23),
            action_cache_token,
            "action_cache_token"
        );
        require_exact_alias(
            birth.at(24),
            action_cache_cursor,
            "action_cache_cursor"
        );
        require_exact_alias(
            birth.at(25),
            action_cache_next,
            "action_cache_next"
        );
        require_exact_alias(
            birth.at(26),
            full_tick_state_tensors.at(22),
            "slot_free_queue"
        );
        require_exact_alias(
            birth.at(27),
            full_tick_state_tensors.at(23),
            "slot_free_queue_count"
        );
        require_exact_alias(
            birth.at(70),
            full_tick_state_tensors.at(1),
            "gru_upkeep_cost"
        );
        require_exact_alias(birth.at(71), attack_bias, "attack_bias");
        require_exact_alias(
            birth.at(72),
            action_temperature,
            "action_temperature"
        );
        require_exact_alias(
            birth.at(73),
            drive_logit_gain,
            "drive_logit_gain"
        );
        require_exact_alias(
            birth.at(74),
            full_tick_state_tensors.at(4),
            "drive_inertia"
        );
        require_exact_alias(
            birth.at(75),
            full_tick_state_tensors.at(5),
            "drive_noise"
        );
        require_exact_alias(birth.at(77), drives, "drives");
        require_exact_alias(birth.at(83), mlp_pack, "mlp_pack");
        require_exact_alias(
            birth.at(93),
            energy_collected,
            "energy_collected"
        );
        require_exact_alias(
            birth.at(94),
            energy_collected_exact,
            "energy_collected_exact"
        );
    }

    const double effective_food_spawn_value =
        food_spawn_per_step * food_spawn_mult;
    const int64_t effective_food_spawn_count = static_cast<int64_t>(
        std::llround(effective_food_spawn_value)
    );
    eiel::cuda_architecture_b::persistent_patch_p2::PatchP2Config
        patch_p2_device_config{};
    double patch_food_gather_amount_value = 0.0;
    double patch_food_initial_value = 0.0;
    double patch_scent_move_emit_value = 0.0;
    double patch_waste_move_emit_value = 0.0;
    double patch_scent_gather_emit_value = 0.0;
    double patch_waste_consume_emit_value = 0.0;
    bool patch_movement_cost_apply_enable_value = false;
    double patch_movement_cost_energy_scale_value = 0.0;
    torch::Tensor patch_movement_cost_energy_scale_by_cell;
    torch::Tensor patch_food_gather_amount_by_cell;
    torch::Tensor patch_food_regrow_per_step_by_cell;
    torch::Tensor reproduce_parent_energy_share_by_cell;
    torch::Tensor reproduce_min_energy_buffer_by_cell;
    torch::Tensor reproduce_cooldown_steps_by_cell;
    torch::Tensor reproduce_try_prob_by_cell;
    torch::Tensor tw_rngv2_world_uid_by_cell;
    double patch_movement_cost_min_value = 0.0;
    double patch_movement_cost_max_value = 0.0;
    if (patch_p2_enabled) {
        const auto finite_config = [&](const char* name) {
            const double value = pybind11::cast<double>(
                patch_p2_config[pybind11::str(name)]
            );
            TORCH_CHECK(std::isfinite(value), name, " must be finite");
            return value;
        };
        const auto bool_config = [&](const char* name) {
            return pybind11::cast<bool>(
                patch_p2_config[pybind11::str(name)]
            );
        };
        const double scent_decay = std::clamp(
            finite_config("patch_scent_decay_per_step"), 0.0, 1.0
        );
        const double waste_decay = std::clamp(
            finite_config("patch_waste_decay_per_step"), 0.0, 1.0
        );
        const double scent_presence = std::max(
            0.0, finite_config("patch_scent_presence_emit")
        );
        const double waste_presence = std::max(
            0.0, finite_config("patch_waste_presence_emit")
        );
        const double density_emit = std::max(
            0.0, finite_config("patch_waste_density_emit_scale")
        );
        const double scent_max_value = std::max(
            0.0, finite_config("patch_scent_max")
        );
        const double waste_max_value = std::max(
            0.0, finite_config("patch_waste_max")
        );
        const double food_max_value = std::max(
            0.0, finite_config("patch_food_amount_max")
        );
        const double regrow_value = std::max(
            0.0, finite_config("patch_food_regrow_per_step")
        );
        const double threshold_value = std::clamp(
            finite_config("patch_food_presence_threshold"),
            0.0,
            food_max_value
        );
        const int64_t density_radius = pybind11::cast<int64_t>(
            patch_p2_config[pybind11::str("patch_density_radius")]
        );
        TORCH_CHECK(
            density_radius >= 0,
            "patch_density_radius must be nonnegative"
        );
        const double waste_scale = std::max(
            0.0, finite_config("patch_waste_regrowth_penalty_scale")
        );
        const double waste_cap = std::clamp(
            finite_config("patch_waste_regrowth_penalty_max"), 0.0, 1.0
        );
        const double density_scale = std::max(
            0.0, finite_config("patch_density_regrowth_penalty_scale")
        );
        const double density_cap = std::clamp(
            finite_config("patch_density_regrowth_penalty_max"), 0.0, 1.0
        );
        const double minimum_multiplier = std::clamp(
            finite_config("patch_regrowth_min_multiplier"), 0.0, 1.0
        );
        if (patch_p2_context_probe_mode) {
            TORCH_CHECK(
                std::isfinite(effective_food_spawn_value)
                    && effective_food_spawn_count == 0,
                "closed Patch P2 b2/t1 context-probe mode requires zero "
                "effective Flat food spawn"
            );
        }
        if (patch_p2_full_mode) {
            patch_food_initial_value = finite_config(
                "patch_food_amount_initial_food_cell_value"
            );
            patch_food_gather_amount_value = finite_config(
                "patch_food_gather_amount"
            );
            patch_scent_move_emit_value = finite_config(
                "patch_scent_move_emit"
            );
            patch_waste_move_emit_value = finite_config(
                "patch_waste_move_emit"
            );
            patch_scent_gather_emit_value = finite_config(
                "patch_scent_gather_emit"
            );
            patch_waste_consume_emit_value = finite_config(
                "patch_waste_consume_emit"
            );
            patch_movement_cost_apply_enable_value = bool_config(
                "patch_movement_cost_apply_enable"
            );
            patch_movement_cost_energy_scale_value = finite_config(
                "patch_movement_cost_energy_scale"
            );
            patch_movement_cost_energy_scale_by_cell = patch_p2_config[
                pybind11::str("patch_movement_cost_energy_scale_by_cell")
            ].cast<torch::Tensor>();
            check_cuda_contiguous(
                active_agent_ids,
                patch_movement_cost_energy_scale_by_cell,
                "patch_movement_cost_energy_scale_by_cell"
            );
            TORCH_CHECK(
                patch_movement_cost_energy_scale_by_cell.scalar_type()
                        == torch::kFloat64
                    && patch_movement_cost_energy_scale_by_cell.sizes()
                        == torch::IntArrayRef({cells}),
                "patch_movement_cost_energy_scale_by_cell must be "
                "float64[cells]"
            );
            if (
                patch_p2_config.size() == 40 || patch_p2_config.size() == 41
                || patch_p2_config.size() == 44
                || patch_p2_config.size() == 45
                || patch_p2_config.size() == 46
                || patch_p2_config.size() == 47
            ) {
                const auto exact_float64_by_cell = [&](const char* name) {
                    auto tensor = patch_p2_config[pybind11::str(name)]
                        .cast<torch::Tensor>();
                    check_cuda_contiguous(active_agent_ids, tensor, name);
                    TORCH_CHECK(
                        tensor.scalar_type() == torch::kFloat64
                            && tensor.sizes() == torch::IntArrayRef({cells})
                            && torch::isfinite(tensor).all().item<bool>()
                            && !torch::any(tensor < 0.0).item<bool>(),
                        name,
                        " must be finite nonnegative float64[cells]"
                    );
                    return tensor;
                };
                patch_food_gather_amount_by_cell = exact_float64_by_cell(
                    "patch_food_gather_amount_by_cell"
                );
                patch_food_regrow_per_step_by_cell = exact_float64_by_cell(
                    "patch_food_regrow_per_step_by_cell"
                );
                reproduce_parent_energy_share_by_cell = exact_float64_by_cell(
                    "reproduce_parent_energy_share_by_cell"
                );
                reproduce_min_energy_buffer_by_cell = exact_float64_by_cell(
                    "reproduce_min_energy_buffer_by_cell"
                );
                reproduce_try_prob_by_cell = exact_float64_by_cell(
                    "reproduce_try_prob_by_cell"
                );
                reproduce_cooldown_steps_by_cell = patch_p2_config[
                    pybind11::str("reproduce_cooldown_steps_by_cell")
                ].cast<torch::Tensor>();
                check_cuda_contiguous(
                    active_agent_ids,
                    reproduce_cooldown_steps_by_cell,
                    "reproduce_cooldown_steps_by_cell"
                );
                TORCH_CHECK(
                    reproduce_cooldown_steps_by_cell.scalar_type()
                            == torch::kInt32
                        && reproduce_cooldown_steps_by_cell.sizes()
                            == torch::IntArrayRef({cells})
                        && !torch::any(reproduce_cooldown_steps_by_cell < 0)
                            .item<bool>(),
                    "reproduce_cooldown_steps_by_cell must be "
                    "nonnegative int32[cells]"
                );
                TORCH_CHECK(
                    torch::all(reproduce_parent_energy_share_by_cell <= 1.0)
                            .item<bool>()
                        && torch::all(reproduce_try_prob_by_cell <= 1.0)
                            .item<bool>()
                        && torch::all(patch_food_gather_amount_by_cell > 0.0)
                            .item<bool>(),
                    "per-cell reproduction probabilities/shares must be in "
                    "[0,1] and food gather amount must be positive"
                );
            }
            patch_movement_cost_min_value = finite_config(
                "patch_movement_cost_min"
            );
            patch_movement_cost_max_value = finite_config(
                "patch_movement_cost_max"
            );
            TORCH_CHECK(
                patch_food_initial_value > threshold_value
                    && patch_food_initial_value <= food_max_value,
                "patch_food_amount_initial_food_cell_value must be greater "
                "than the presence threshold and no greater than the maximum"
            );
            TORCH_CHECK(
                patch_food_gather_amount_value > 0.0,
                "patch_food_gather_amount must be positive"
            );
            for (const auto& named : {
                     std::pair<const char*, double>{
                         "patch_scent_move_emit",
                         patch_scent_move_emit_value,
                     },
                     {"patch_waste_move_emit", patch_waste_move_emit_value},
                     {"patch_scent_gather_emit", patch_scent_gather_emit_value},
                     {"patch_waste_consume_emit", patch_waste_consume_emit_value},
                     {
                         "patch_movement_cost_energy_scale",
                         patch_movement_cost_energy_scale_value,
                     },
                 }) {
                TORCH_CHECK(
                    named.second >= 0.0,
                    named.first,
                    " must be non-negative"
                );
            }
            TORCH_CHECK(
                patch_movement_cost_max_value >= patch_movement_cost_min_value,
                "patch_movement_cost_max must be greater than or equal to "
                "patch_movement_cost_min"
            );
        }
        patch_p2_device_config = {
            height,
            width,
            static_cast<float>(1.0 - scent_decay),
            static_cast<float>(1.0 - waste_decay),
            scent_presence,
            waste_presence,
            static_cast<float>(density_emit),
            static_cast<float>(scent_max_value),
            static_cast<float>(waste_max_value),
            static_cast<float>(food_max_value),
            static_cast<float>(regrow_value),
            bool_config("patch_food_regrow_use_fertility"),
            bool_config("patch_food_regrow_cap_by_fertility"),
            static_cast<float>(threshold_value),
            density_radius,
            bool_config("patch_density_include_self"),
            bool_config("patch_density_normalize"),
            bool_config("patch_world_waste_consequence_enable"),
            bool_config("patch_waste_regrowth_penalty_enable"),
            static_cast<float>(waste_scale),
            static_cast<float>(waste_cap),
            bool_config("patch_density_regrowth_penalty_enable"),
            static_cast<float>(density_scale),
            static_cast<float>(density_cap),
            static_cast<float>(minimum_multiplier),
        };
    }

    c10::cuda::CUDAGuard device_guard(active_agent_ids.device());

    using namespace eiel::cuda_architecture_b::flattened_controller_v1;
    const FlatFullTickState full_tick_state{
        full_tick_state_tensors.at(0).data_ptr<int32_t>(),
        full_tick_state_tensors.at(1).data_ptr<float>(),
        full_tick_state_tensors.at(2).data_ptr<float>(),
        full_tick_state_tensors.at(3).data_ptr<float>(),
        full_tick_state_tensors.at(4).data_ptr<float>(),
        full_tick_state_tensors.at(5).data_ptr<float>(),
        full_tick_state_tensors.at(6).data_ptr<float>(),
        full_tick_state_tensors.at(7).data_ptr<float>(),
        full_tick_state_tensors.at(8).data_ptr<float>(),
        full_tick_state_tensors.at(9).data_ptr<float>(),
        full_tick_state_tensors.at(10).data_ptr<float>(),
        full_tick_state_tensors.at(11).data_ptr<float>(),
        full_tick_state_tensors.at(12).data_ptr<float>(),
        full_tick_state_tensors.at(13).data_ptr<int32_t>(),
        full_tick_state_tensors.at(14).data_ptr<int32_t>(),
        full_tick_state_tensors.at(15).data_ptr<float>(),
        full_tick_state_tensors.at(16).data_ptr<uint64_t>(),
        full_tick_state_tensors.at(17).data_ptr<int64_t>(),
        full_tick_state_tensors.at(18).data_ptr<uint64_t>(),
        full_tick_state_tensors.at(19).data_ptr<int64_t>(),
        full_tick_state_tensors.at(20).data_ptr<float>(),
        full_tick_state_tensors.at(21).data_ptr<bool>(),
        full_tick_state_tensors.at(22).data_ptr<int64_t>(),
        full_tick_state_tensors.at(23).data_ptr<int64_t>(),
        full_tick_state_tensors.at(24).data_ptr<int64_t>(),
        full_tick_state_tensors.at(25).data_ptr<int64_t>(),
        full_tick_state_tensors.at(26).data_ptr<int64_t>(),
        full_tick_state_tensors.at(27).data_ptr<int64_t>(),
        full_tick_state_tensors.at(28).data_ptr<int64_t>(),
        drive_noise_sample_ticks,
        full_tick_state_tensors.at(15).size(2),
    };
    const FlatFullTickOutputs full_tick_outputs{
        full_tick_output_tensors.at(0).data_ptr<int32_t>(),
        full_tick_output_tensors.at(1).data_ptr<int64_t>(),
        full_tick_output_tensors.at(2).data_ptr<int32_t>(),
        full_tick_output_tensors.at(3).data_ptr<int64_t>(),
        full_tick_output_tensors.at(4).data_ptr<int64_t>(),
        full_tick_output_tensors.at(5).data_ptr<int32_t>(),
        full_tick_output_tensors.at(6).data_ptr<int64_t>(),
        full_tick_output_tensors.at(7).data_ptr<int32_t>(),
        full_tick_output_tensors.at(8).data_ptr<int32_t>(),
        full_tick_output_tensors.at(9).data_ptr<int64_t>(),
        full_tick_output_tensors.at(10).data_ptr<int64_t>(),
        full_tick_output_tensors.at(11).data_ptr<int64_t>(),
        full_tick_output_tensors.at(12).data_ptr<int32_t>(),
        full_tick_output_tensors.at(13).data_ptr<int64_t>(),
        full_tick_output_tensors.at(14).data_ptr<int64_t>(),
        full_tick_output_tensors.at(15).data_ptr<int32_t>(),
        full_tick_output_tensors.at(16).data_ptr<uint64_t>(),
        full_tick_output_tensors.at(17).data_ptr<int64_t>(),
        full_tick_output_tensors.at(18).data_ptr<int64_t>(),
    };
    const FlatEventState event_state{
        event_state_tensors.at(0).data_ptr<int32_t>(),
        event_state_tensors.at(1).data_ptr<int64_t>(),
        event_state_tensors.at(2).data_ptr<int64_t>(),
        event_state_tensors.at(3).data_ptr<int64_t>(),
        event_state_tensors.at(4).data_ptr<int64_t>(),
        event_state_tensors.at(5).data_ptr<float>(),
        event_state_tensors.at(6).data_ptr<int32_t>(),
        event_state_tensors.at(7).data_ptr<int64_t>(),
        event_state_tensors.at(8).data_ptr<int64_t>(),
        event_state_tensors.at(9).data_ptr<int64_t>(),
        event_state_tensors.at(10).data_ptr<int64_t>(),
        event_state_tensors.at(11).data_ptr<int64_t>(),
        event_state_tensors.at(12).data_ptr<int64_t>(),
        event_state_tensors.at(13).data_ptr<int64_t>(),
        event_state_tensors.at(14).data_ptr<int64_t>(),
        event_state_tensors.at(15).data_ptr<int64_t>(),
        event_state_tensors.at(16).data_ptr<int64_t>(),
        event_state_tensors.at(17).data_ptr<int32_t>(),
        event_state_tensors.at(18).data_ptr<int32_t>(),
        event_state_tensors.at(19).data_ptr<int32_t>(),
        event_state_tensors.at(20).data_ptr<int8_t>(),
        event_state_tensors.at(21).data_ptr<float>(),
        event_state_tensors.at(22).data_ptr<int64_t>(),
        event_state_tensors.at(30).data_ptr<int64_t>(),
        event_state_tensors.at(23).data_ptr<int32_t>(),
        event_state_tensors.at(24).data_ptr<int32_t>(),
        event_state_tensors.at(25).data_ptr<int32_t>(),
        initial_agent_ids_by_row.data_ptr<int64_t>(),
        event_state_tensors.at(27).data_ptr<int32_t>(),
        event_state_tensors.at(28).data_ptr<int32_t>(),
        event_state_tensors.at(29).data_ptr<int32_t>(),
        event_capacity,
        reporting_rows,
    };
    FlatOrdinaryBirthState ordinary_birth_state{};
    FlatOrdinaryBirthOutputs ordinary_birth_outputs{};
    FlatFlattenedBirthScratch ordinary_birth_scratch{};
    FlatOrdinaryBirthConfig ordinary_birth_config{
        ordinary_births_enabled,
        max_births_per_tick,
        reproduce_parent_energy_share,
        static_cast<float>(reproduce_min_energy_buffer),
        static_cast<int32_t>(reproduce_cooldown_steps),
        static_cast<int32_t>(genome_sensor_radius_max),
        injury_repro_k_injury,
        injury_repro_k_scar,
        injury_birth_fail_scar,
        injury_birth_fail_injury,
        env_policy_alias,
        reproduce_requires_mate,
        0,
        0,
        0,
        0,
        0,
        cross_path_resident_control_v1(),
    };
    if (
        patch_p2_config.size() == 40 || patch_p2_config.size() == 41
        || patch_p2_config.size() == 44 || patch_p2_config.size() == 45
        || patch_p2_config.size() == 46 || patch_p2_config.size() == 47
    ) {
        ordinary_birth_config.parent_energy_share_by_cell =
            reproduce_parent_energy_share_by_cell.data_ptr<double>();
        ordinary_birth_config.minimum_energy_buffer_by_cell =
            reproduce_min_energy_buffer_by_cell.data_ptr<double>();
        ordinary_birth_config.cooldown_steps_by_cell =
            reproduce_cooldown_steps_by_cell.data_ptr<int32_t>();
        ordinary_birth_config.reproduce_try_probability_by_cell =
            reproduce_try_prob_by_cell.data_ptr<double>();
    }
    ordinary_birth_config.rng_semantics_version = 1;
    ordinary_birth_config.world_uid = 0;
    ordinary_birth_config.world_uid_by_cell = nullptr;
    ordinary_birth_config.rng_namespace_key = tw_rngv2::NAMESPACE_DEFAULT;
    ordinary_birth_config.rng_semantic_epoch =
        tw_rngv2::SEMANTIC_EPOCH_V1;
    const bool has_tw_rngv2_coordinates =
        patch_p2_config.contains("tw_rngv2_semantics_version");
    TORCH_CHECK(
        has_tw_rngv2_coordinates
            == patch_p2_config.contains("tw_rngv2_world_uid")
        && has_tw_rngv2_coordinates
            == patch_p2_config.contains("tw_rngv2_world_uid_by_cell")
        && has_tw_rngv2_coordinates
            == patch_p2_config.contains("tw_rngv2_namespace_key64")
        && has_tw_rngv2_coordinates
            == patch_p2_config.contains("tw_rngv2_semantic_epoch"),
        "patch_p2_config must provide the complete TW RNG coordinate contract"
    );
    if (has_tw_rngv2_coordinates) {
        ordinary_birth_config.rng_semantics_version = pybind11::cast<int32_t>(
            patch_p2_config["tw_rngv2_semantics_version"]
        );
        ordinary_birth_config.world_uid = pybind11::cast<uint32_t>(
            patch_p2_config["tw_rngv2_world_uid"]
        );
        tw_rngv2_world_uid_by_cell = patch_p2_config[
            "tw_rngv2_world_uid_by_cell"
        ].cast<torch::Tensor>();
        check_cuda_contiguous(
            active_agent_ids,
            tw_rngv2_world_uid_by_cell,
            "tw_rngv2_world_uid_by_cell"
        );
        TORCH_CHECK(
            tw_rngv2_world_uid_by_cell.scalar_type() == torch::kInt64
                && tw_rngv2_world_uid_by_cell.sizes()
                    == torch::IntArrayRef({cells})
                && !torch::any(tw_rngv2_world_uid_by_cell < 0).item<bool>()
                && !torch::any(
                    tw_rngv2_world_uid_by_cell
                        > static_cast<int64_t>(
                            std::numeric_limits<uint32_t>::max()
                        )
                ).item<bool>(),
            "tw_rngv2_world_uid_by_cell must be uint32-bounded int64[cells]"
        );
        if (ordinary_birth_config.rng_semantics_version == 2 && cells > 1) {
            const torch::Tensor logical_world_uids_cpu =
                tw_rngv2_world_uid_by_cell.to(torch::kCPU).contiguous();
            const int64_t* const logical_world_uid_values =
                logical_world_uids_cpu.data_ptr<int64_t>();
            std::set<int64_t> unique_logical_world_uids;
            for (int64_t cell = 0; cell < cells; ++cell) {
                unique_logical_world_uids.insert(logical_world_uid_values[cell]);
            }
            TORCH_CHECK(
                static_cast<int64_t>(unique_logical_world_uids.size()) == cells,
                "multi-cell TW RNG V2 world UIDs must be unique"
            );
        }
        ordinary_birth_config.world_uid_by_cell =
            tw_rngv2_world_uid_by_cell.data_ptr<int64_t>();
        ordinary_birth_config.rng_namespace_key = pybind11::cast<uint64_t>(
            patch_p2_config["tw_rngv2_namespace_key64"]
        );
        ordinary_birth_config.rng_semantic_epoch = pybind11::cast<uint32_t>(
            patch_p2_config["tw_rngv2_semantic_epoch"]
        );
    }
    if (lifecycle_state_enabled) {
        const auto& state = ordinary_birth_state_tensors;
        const auto& output = ordinary_birth_output_tensors;
        ordinary_birth_state = {
            state.at(10).data_ptr<float>(),
            state.at(12).data_ptr<int32_t>(),
            state.at(13).data_ptr<float>(),
            state.at(14).data_ptr<int32_t>(),
            state.at(17).data_ptr<int32_t>(),
            state.at(18).data_ptr<int32_t>(),
            state.at(19).data_ptr<int32_t>(),
            state.at(28).data_ptr<int64_t>(),
            state.at(35).data_ptr<uint32_t>(),
            state.at(36).data_ptr<int64_t>(),
            state.at(37).data_ptr<bool>(),
            state.at(38).data_ptr<double>(),
            state.at(39).data_ptr<double>(),
            state.at(40).data_ptr<double>(),
            state.at(41).data_ptr<int64_t>(),
            state.at(42).data_ptr<int64_t>(),
            state.at(43).data_ptr<int32_t>(),
            state.at(44).data_ptr<int32_t>(),
            state.at(45).data_ptr<int32_t>(),
            state.at(46).data_ptr<int32_t>(),
            state.at(47).data_ptr<int32_t>(),
            state.at(48).data_ptr<int32_t>(),
            state.at(49).data_ptr<int32_t>(),
            state.at(50).data_ptr<int32_t>(),
            state.at(51).data_ptr<int64_t>(),
            state.at(52).data_ptr<bool>(),
            state.at(53).data_ptr<float>(),
            state.at(54).data_ptr<float>(),
            state.at(55).data_ptr<float>(),
            state.at(56).data_ptr<float>(),
            state.at(57).data_ptr<float>(),
            state.at(58).data_ptr<float>(),
            state.at(59).data_ptr<float>(),
            state.at(60).data_ptr<float>(),
            state.at(61).data_ptr<float>(),
            state.at(62).data_ptr<float>(),
            state.at(63).data_ptr<float>(),
            state.at(64).data_ptr<float>(),
            state.at(65).data_ptr<float>(),
            state.at(66).data_ptr<float>(),
            state.at(67).data_ptr<float>(),
            state.at(68).data_ptr<float>(),
            state.at(76).data_ptr<float>(),
            state.at(78).data_ptr<float>(),
            state.at(79).data_ptr<uint64_t>(),
            state.at(80).data_ptr<int64_t>(),
            state.at(81).data_ptr<int64_t>(),
            state.at(84).data_ptr<float>(),
            state.at(85).data_ptr<uint8_t>(),
            state.at(86).data_ptr<float>(),
            state.at(87).data_ptr<bool>(),
            state.at(88).data_ptr<uint8_t>(),
            state.at(89).numel() == 0
                ? nullptr
                : state.at(89).data_ptr<uint8_t>(),
            state.at(90).data_ptr<int64_t>(),
            state.at(91).data_ptr<int64_t>(),
            state.at(92).data_ptr<int64_t>(),
            state.at(95).data_ptr<int32_t>(),
            state.at(96).data_ptr<int32_t>(),
            state.at(97).data_ptr<float>(),
            state.at(98).data_ptr<double>(),
            state.at(99).data_ptr<float>(),
            state.at(100).data_ptr<float>(),
            state.at(101).data_ptr<float>(),
            state.at(102).data_ptr<int32_t>(),
            state.at(103).data_ptr<int32_t>(),
            state.at(104).data_ptr<bool>(),
            state.at(105).data_ptr<float>(),
            state.at(106).data_ptr<int32_t>(),
            state.at(107).data_ptr<float>(),
            state.at(108).data_ptr<int32_t>(),
            state.at(109).data_ptr<float>(),
            state.at(110).data_ptr<float>(),
            state.at(111).data_ptr<int64_t>(),
            state.at(112).data_ptr<int64_t>(),
            diploid_state_enabled
                ? state.at(113).data_ptr<int32_t>()
                : nullptr,
            diploid_state_enabled
                ? state.at(114).data_ptr<bool>()
                : nullptr,
            diploid_state_enabled
                ? state.at(115).data_ptr<bool>()
                : nullptr,
            diploid_state_enabled
                ? state.at(116).data_ptr<double>()
                : nullptr,
            diploid_state_enabled
                ? state.at(117).data_ptr<float>()
                : nullptr,
            diploid_state_enabled
                ? state.at(118).data_ptr<float>()
                : nullptr,
            diploid_state_enabled
                ? state.at(119).data_ptr<float>()
                : nullptr,
            diploid_state_enabled
                ? state.at(120).data_ptr<float>()
                : nullptr,
            genome_upgrade_v2_state_enabled
                ? state.at(122).data_ptr<bool>()
                : nullptr,
            genome_upgrade_v2_state_enabled
                ? state.at(123).data_ptr<int64_t>()
                : nullptr,
            genome_upgrade_v2_state_enabled
                ? state.at(124).data_ptr<int64_t>()
                : nullptr,
            genome_upgrade_v2_state_enabled
                ? state.at(125).data_ptr<int64_t>()
                : nullptr,
            genome_upgrade_v2_state_enabled
                ? state.at(126).data_ptr<int64_t>()
                : nullptr,
            genome_upgrade_v2_state_enabled
                ? state.at(127).data_ptr<int64_t>()
                : nullptr,
            genome_upgrade_v2_state_enabled
                ? state.at(128).data_ptr<int64_t>()
                : nullptr,
            genome_upgrade_v2_state_enabled
                ? state.at(129).data_ptr<int64_t>()
                : nullptr,
            genome_upgrade_v2_state_enabled
                ? state.at(130).data_ptr<int64_t>()
                : nullptr,
            genome_upgrade_v2_state_enabled
                ? state.at(131).data_ptr<int64_t>()
                : nullptr,
            genome_upgrade_v2_state_enabled
                ? state.at(132).data_ptr<int64_t>()
                : nullptr,
            genome_upgrade_v2_state_enabled
                ? state.at(133).data_ptr<int64_t>()
                : nullptr,
            genome_upgrade_v2_state_enabled
                ? state.at(134).data_ptr<int64_t>()
                : nullptr,
            genome_upgrade_v2_state_enabled
                ? state.at(135).data_ptr<int64_t>()
                : nullptr,
            genome_upgrade_v2_state_enabled
                ? state.at(136).data_ptr<int64_t>()
                : nullptr,
            genome_upgrade_v2_state_enabled
                ? state.at(137).data_ptr<int64_t>()
                : nullptr,
            genome_upgrade_v2_state_enabled
                ? state.at(138).data_ptr<int64_t>()
                : nullptr,
            diploid_state_enabled
                ? state.at(121).data_ptr<uint8_t>()
                : nullptr,
        };
        if (ordinary_births_enabled) {
            ordinary_birth_outputs = {
            output.at(0).data_ptr<int64_t>(),
            output.at(1).data_ptr<int32_t>(),
            output.at(2).data_ptr<int64_t>(),
            output.at(3).data_ptr<int64_t>(),
            output.at(4).data_ptr<int32_t>(),
            output.at(5).data_ptr<float>(),
            output.at(6).data_ptr<int64_t>(),
            output.at(7).data_ptr<int32_t>(),
            output.at(8).data_ptr<int32_t>(),
            output.at(9).data_ptr<int64_t>(),
            output.at(10).data_ptr<int64_t>(),
            output.at(11).data_ptr<int32_t>(),
            output.at(12).data_ptr<bool>(),
            output.at(13).data_ptr<int64_t>(),
            output.at(14).data_ptr<int64_t>(),
            output.at(15).data_ptr<int64_t>(),
            output.at(16).data_ptr<int64_t>(),
            output.at(17).data_ptr<int64_t>(),
            output.at(18).data_ptr<int64_t>(),
            output.at(19).data_ptr<int64_t>(),
            output.at(20).data_ptr<int32_t>(),
            output.at(21).data_ptr<int64_t>(),
            output.at(22).data_ptr<int64_t>(),
            output.at(23).data_ptr<int64_t>(),
            output.at(24).data_ptr<int64_t>(),
            output.at(25).data_ptr<int32_t>(),
            output.at(26).data_ptr<int64_t>(),
            output.at(27).data_ptr<int64_t>(),
            output.at(28).data_ptr<int32_t>(),
            output.at(29).data_ptr<int64_t>(),
            output.at(30).data_ptr<int64_t>(),
            output.at(31).data_ptr<int64_t>(),
            output.at(32).data_ptr<int64_t>(),
            output.at(33).data_ptr<int64_t>(),
            output.at(34).data_ptr<int32_t>(),
            output.at(35).data_ptr<int32_t>(),
            output.at(36).data_ptr<int64_t>(),
            output.at(37).data_ptr<int64_t>(),
            output.at(38).data_ptr<int64_t>(),
            output.at(39).data_ptr<int64_t>(),
            output.at(40).data_ptr<double>(),
            output.at(41).data_ptr<bool>(),
            output.at(42).data_ptr<float>(),
            output.at(43).data_ptr<int32_t>(),
            output.at(44).data_ptr<int32_t>(),
            output.at(45).data_ptr<uint64_t>(),
            output.at(46).data_ptr<int64_t>(),
            output.at(47).data_ptr<int64_t>(),
            output.at(48).data_ptr<int64_t>(),
            output.at(49).data_ptr<double>(),
            output.at(50).data_ptr<double>(),
            output.at(51).data_ptr<double>(),
            output.at(52).data_ptr<double>(),
            output.at(53).data_ptr<int64_t>(),
            output.at(54).data_ptr<int32_t>(),
            output.at(55).data_ptr<int32_t>(),
            output.at(56).data_ptr<int64_t>(),
            output.at(57).data_ptr<int32_t>(),
            output.at(58).data_ptr<int64_t>(),
            output.at(59).data_ptr<bool>(),
            output.at(60).data_ptr<int64_t>(),
            output.at(61).data_ptr<int64_t>(),
            output.at(62).data_ptr<bool>(),
            output.at(58).size(1),
            };
        }
        ordinary_birth_config.gauss_tape_capacity = state.at(39).size(1);
        ordinary_birth_config.policy_tape_capacity = state.at(86).size(1);
        ordinary_birth_config.policy_mutation_tape_width = state.at(86).size(2);
        ordinary_birth_config.torch_state_bytes = state.at(85).size(1);
        ordinary_birth_config.hmem_width = state.at(84).size(2);
        if (ordinary_birth_config.rng_semantics_version == 2) {
            TORCH_CHECK(
                ordinary_birth_config.gauss_tape_capacity == 0
                    && ordinary_birth_config.policy_tape_capacity == 0,
                "TW RNG V2 forbids host-refilled Gaussian and policy tapes"
            );
        }
        if (lifecycle_scratch_enabled) {
            ordinary_birth_scratch = {
                ordinary_birth_scratch_tensors.at(0).data_ptr<uint8_t>(),
                ordinary_birth_scratch_tensors.at(0).size(1),
            };
        }
    }
    turnover::FlatTurnoverMutableTensorTable turnover_tensors{};
    turnover::FlatTurnoverScratch turnover_scratch{};
    turnover::FlatTurnoverOutputs turnover_outputs{};
#ifdef EIEL_WDL_JUNGLE_EED_HAPB_FIXED_MLP_NATIVE_BINDING_V1
    const bool hapb_direct_policy =
        hapb_policy_context::hapb_direct_route_context_active_v1();
#else
    const bool hapb_direct_policy = false;
#endif
    const auto turnover_birth_layout =
        eiel::cuda_architecture_b::flattened_birth_scratch_v2::make_layout(
            max_births_per_tick,
            capacity,
            packed_width,
            ordinary_birth_config.hmem_width,
            ordinary_birth_config.torch_state_bytes
        );
    const int32_t* const turnover_qualification_fault_stage =
        lifecycle_scratch_enabled && turnover_birth_layout.ok()
        ? reinterpret_cast<const int32_t*>(
            ordinary_birth_scratch_tensors.at(0).data_ptr<uint8_t>()
            + controller::kBirthScratchBytesPerCell
            + turnover_birth_layout.segment(
                eiel::cuda_architecture_b::flattened_birth_scratch_v2::
                    kGenomeUpgradeV2QualificationFaultStage
            ).offset
        )
        : nullptr;
    const turnover::FlatTurnoverConfig turnover_config{
        generation_turnover_enabled,
        static_cast<int32_t>(lifecycle_mode),
        turnover_population_size,
        turnover_cull_fraction,
        turnover_parent_fraction,
        coherence_thrashing_weight,
        coherence_drive_volatility_weight,
        coherence_selfharm_weight,
        coherence_pred_err_weight,
        status.data_ptr<int32_t>(),
        turnover_qualification_fault_stage,
        lifecycle_scratch_enabled
            ? ordinary_birth_scratch_tensors.at(0).size(1)
            : 0,
        ordinary_birth_config.cross_path_control,
        static_cast<int32_t>(turnover_selection_mode),
        turnover_selection_seed_by_cell.data_ptr<uint64_t>(),
        hapb_direct_policy,
    };
    if (generation_turnover_enabled) {
        turnover_tensors = {
            reinterpret_cast<const turnover::TurnoverMutableTensorTriplet*>(
                turnover_pointer_table_tensors.at(0).data_ptr<int64_t>()
            ),
        };
        turnover_scratch = {
            ordinary_birth_scratch_tensors.at(0).data_ptr<uint8_t>(),
            ordinary_birth_scratch_tensors.at(0).size(1),
        };
        turnover_outputs = {
            turnover_output_tensors.at(0).data_ptr<int64_t>(),
            turnover_output_tensors.at(1).data_ptr<int64_t>(),
            turnover_output_tensors.at(2).data_ptr<int64_t>(),
            turnover_output_tensors.at(3).data_ptr<int64_t>(),
            turnover_output_tensors.at(4).data_ptr<int64_t>(),
            turnover_output_tensors.at(5).data_ptr<int64_t>(),
            turnover_output_tensors.at(6).data_ptr<int64_t>(),
            turnover_output_tensors.at(7).data_ptr<int64_t>(),
            turnover_output_tensors.at(8).data_ptr<int32_t>(),
            turnover_output_tensors.at(9).data_ptr<int64_t>(),
            turnover_output_tensors.at(10).data_ptr<int64_t>(),
            turnover_output_tensors.at(11).data_ptr<int64_t>(),
            turnover_output_tensors.at(12).data_ptr<bool>(),
            turnover_output_tensors.at(13).data_ptr<int64_t>(),
            turnover_output_tensors.at(14).data_ptr<int64_t>(),
            turnover_output_tensors.at(15).data_ptr<bool>(),
            turnover_output_tensors.at(11).size(1),
        };
    }
    const FlatMechanicsConfig mechanics_config{
        enable_rest_charge,
        static_cast<int32_t>(rest_streak_free),
        static_cast<float>(rest_charge_energy),
        static_cast<float>(rest_charge_growth),
        static_cast<float>(rest_deficit_to_health),
        static_cast<int32_t>(developmental_maturity_age),
        static_cast<int32_t>(life_history_mode),
        static_cast<int32_t>(life_history_maturity_start_ticks),
        static_cast<int32_t>(life_history_maturity_rise_ticks),
        static_cast<int32_t>(life_history_fertility_decline_start_ticks),
        static_cast<int32_t>(life_history_fertility_decline_ticks),
        static_cast<float>(life_history_elder_fertility_floor),
        static_cast<int32_t>(life_history_senescence_start_ticks),
        static_cast<int32_t>(life_history_senescence_ramp_ticks),
        static_cast<float>(life_history_senescence_energy_cost),
        static_cast<float>(life_history_senescence_health_cost),
        static_cast<int32_t>(life_history_max_age_ticks),
        static_cast<float>(drive_hunger_gain),
        static_cast<float>(drive_hunger_decay),
        static_cast<float>(drive_pain_gain),
        static_cast<float>(drive_pain_decay),
        static_cast<float>(drive_fatigue_gain),
        static_cast<float>(drive_fatigue_rest_relief),
        static_cast<float>(drive_fatigue_decay),
        static_cast<float>(drive_fear_gain),
        static_cast<float>(drive_fear_decay),
        generation,
        start_tick,
        exact_tick_native_span ? 0 : start_tick,
    };
    FlatPatchP2Context patch_p2_context{};
    patch_p2_context.enabled = patch_p2_enabled;
    patch_p2_context.mode = patch_p2_full_mode
        ? FlatPatchP2Mode::kFullPatchP2
        : (patch_p2_context_probe_mode
               ? FlatPatchP2Mode::kContextProbeB2T1
               : FlatPatchP2Mode::kDisabled);
    patch_p2_context.stats_ticks_per_cell = generation_count * ticks;
    if (patch_p2_enabled) {
        patch_p2_context.state = {
            patch_p2_state_tensors.at(0).data_ptr<float>(),
            patch_p2_state_tensors.at(1).data_ptr<float>(),
            patch_p2_state_tensors.at(2).data_ptr<float>(),
            patch_p2_state_tensors.at(3).data_ptr<float>(),
            patch_p2_state_tensors.at(4).data_ptr<float>(),
            patch_p2_state_tensors.at(5).data_ptr<float>(),
            agent_grid.data_ptr<int32_t>(),
            patch_p2_state_tensors.at(6).data_ptr<float>(),
            food_grid.data_ptr<int32_t>(),
        };
        float* const patch_p2_scratch_base =
            patch_p2_scratch_tensors.at(0).data_ptr<float>();
        const int64_t patch_p2_grid_cells = height * width;
        patch_p2_context.scratch = {
            patch_p2_scratch_base,
            patch_p2_scratch_base + patch_p2_grid_cells,
            patch_p2_scratch_base + 2 * patch_p2_grid_cells,
            patch_p2_scratch_base + 3 * patch_p2_grid_cells,
        };
        patch_p2_context.outputs = {
            patch_p2_stats_tensors.at(0).data_ptr<double>(),
        };
        patch_p2_context.compact_metrics =
            patch_p2_stats_tensors.size() == 2
            ? patch_p2_stats_tensors.at(1).data_ptr<double>()
            : nullptr;
        patch_p2_context.config = patch_p2_device_config;
    }
    if (patch_p2_full_mode) {
        patch_p2_context.patch_cover =
            patch_p2_state_tensors.at(7).data_ptr<float>();
        patch_p2_context.patch_shelter =
            patch_p2_state_tensors.at(8).data_ptr<float>();
        patch_p2_context.patch_movement_cost =
            patch_p2_state_tensors.at(9).data_ptr<float>();
        patch_p2_context.patch_food_initial_value = patch_food_initial_value;
        patch_p2_context.patch_food_gather_amount =
            patch_food_gather_amount_value;
        patch_p2_context.patch_scent_move_emit = patch_scent_move_emit_value;
        patch_p2_context.patch_waste_move_emit = patch_waste_move_emit_value;
        patch_p2_context.patch_scent_gather_emit =
            patch_scent_gather_emit_value;
        patch_p2_context.patch_waste_consume_emit =
            patch_waste_consume_emit_value;
        patch_p2_context.patch_movement_cost_apply_enable =
            patch_movement_cost_apply_enable_value;
        patch_p2_context.patch_movement_cost_energy_scale =
            patch_movement_cost_energy_scale_value;
        patch_p2_context.patch_movement_cost_energy_scale_by_cell =
            patch_movement_cost_energy_scale_by_cell.data_ptr<double>();
        patch_p2_context.patch_movement_cost_min =
            patch_movement_cost_min_value;
        patch_p2_context.patch_movement_cost_max =
            patch_movement_cost_max_value;
        patch_p2_context.patch_food_gather_amount_by_cell =
            patch_food_gather_amount_by_cell.defined()
            ? patch_food_gather_amount_by_cell.data_ptr<double>()
            : nullptr;
        patch_p2_context.patch_food_regrow_per_step_by_cell =
            patch_food_regrow_per_step_by_cell.defined()
            ? patch_food_regrow_per_step_by_cell.data_ptr<double>()
            : nullptr;
    }

    FlatCompactSummaryContextV1 compact_summary_context{};
    if (compact_summary_planes_enabled) {
        compact_summary_context = {
            true,
            compact_summary_reporting_tensors.at(0).data_ptr<uint8_t>(),
            compact_summary_reporting_tensors.at(1).data_ptr<int32_t>(),
            compact_summary_reporting_tensors.at(2).data_ptr<double>(),
            compact_summary_reporting_tensors.at(3).data_ptr<int64_t>(),
            compact_summary_reporting_tensors.at(4).data_ptr<int64_t>(),
            compact_summary_reporting_tensors.at(5).data_ptr<int64_t>(),
            reporting_authority::ReportingFitnessConfigV1{
                reporting_parent_fraction,
                reporting_episode0_fitness_weight,
                reporting_episode0_pred_err_fitness_weight,
                reporting_episode0_thrashing_fitness_weight,
                reporting_episode0_drive_jitter_fitness_weight,
                reporting_episode0_selfharm_fitness_weight,
                reporting_coherence_thrashing_weight,
                reporting_coherence_drive_jitter_weight,
                reporting_coherence_selfharm_weight,
                reporting_coherence_pred_err_weight,
                reporting_coherence_reset_shock_weight,
                reporting_coherence_episode0_pred_err_weight,
                reporting_late_pred_cost_weight,
                static_cast<uint8_t>(reporting_late_pred_cost_enable),
                static_cast<uint8_t>(
                    reporting_disable_prediction_error_selection_pressure
                ),
                static_cast<uint8_t>(patch_p2_enabled),
                {},
            },
            0,
            generation_count,
            cells,
            static_cast<int32_t>(compact_reporting_row_capacity),
            static_cast<int32_t>(capacity),
        };
    }

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    std::vector<checkpoint_hash::TensorDigestPointerBatchV1>
        compact_checkpoint_batches;
    bool compact_checkpoint_active_ids_bound = false;
    bool compact_checkpoint_active_slots_bound = false;
    if (compact_checkpoint_hash_enabled) {
        TORCH_CHECK(
            compact_checkpoint_descriptor_count
                == checkpoint_runtime::expected_descriptor_count_v1(
                    patch_p2_full_mode
                ),
            "compact checkpoint descriptor count must match the frozen ABI"
        );
        const int64_t* const specs =
            compact_summary_checkpoint_descriptor_specs.data_ptr<int64_t>();
        compact_checkpoint_batches.reserve(static_cast<size_t>(
            (compact_checkpoint_descriptor_count
                 + checkpoint_hash::kPointerBatchCapacityV1 - 1)
            / checkpoint_hash::kPointerBatchCapacityV1
        ));
        for (
            int64_t descriptor = 0;
            descriptor < compact_checkpoint_descriptor_count;
            ++descriptor
        ) {
            const int64_t* const spec = specs
                + descriptor * checkpoint_runtime::kDescriptorSpecWidthV1;
            const int64_t data_index =
                spec[checkpoint_runtime::kDataTensorIndexV1];
            const int64_t alive_index =
                spec[checkpoint_runtime::kAliveMaskTensorIndexV1];
            const int64_t active_length_index =
                spec[checkpoint_runtime::kActiveLengthTensorIndexV1];
            const int64_t cell_stride =
                spec[checkpoint_runtime::kCellStrideBytesV1];
            const int64_t bytes_per_cell =
                spec[checkpoint_runtime::kBytesPerCellV1];
            const int64_t row_capacity =
                spec[checkpoint_runtime::kRowCapacityV1];
            const int64_t row_bytes =
                spec[checkpoint_runtime::kRowBytesV1];
            const int64_t alive_stride =
                spec[checkpoint_runtime::kAliveMaskCellStrideV1];
            const int64_t dtype_bytes =
                spec[checkpoint_runtime::kDtypeBytesV1];
            const int64_t mode =
                spec[checkpoint_runtime::kCanonicalizationModeV1];
            const int64_t raw_output_index =
                spec[checkpoint_runtime::kRawOutputIndexV1];
            const int64_t canonical_output_index =
                spec[checkpoint_runtime::kCanonicalOutputIndexV1];
            const int64_t descriptor_index =
                spec[checkpoint_runtime::kDescriptorIndexV1];
            TORCH_CHECK(
                data_index >= 0
                    && data_index
                        < static_cast<int64_t>(
                            compact_summary_checkpoint_source_tensors.size()
                        )
                    && (alive_index == -1
                        || (alive_index >= 0
                            && alive_index
                                < static_cast<int64_t>(
                                    compact_summary_checkpoint_source_tensors.size()
                                )))
                    && (active_length_index == -1
                        || (active_length_index >= 0
                            && active_length_index
                                < static_cast<int64_t>(
                                    compact_summary_checkpoint_source_tensors.size()
                                )))
                    && cell_stride > 0
                    && bytes_per_cell > 0
                    && row_capacity > 0
                    && row_bytes > 0
                    && alive_stride >= 0
                    && (dtype_bytes == 1 || dtype_bytes == 2
                        || dtype_bytes == 4 || dtype_bytes == 8)
                    && mode >= checkpoint_hash::kExactStateRawBytesV1
                    && mode <= checkpoint_hash::kActivePrefixV1
                    && raw_output_index == descriptor
                    && canonical_output_index == descriptor
                    && descriptor_index == descriptor,
                "compact checkpoint descriptor spec is malformed at index ",
                descriptor
            );
            TORCH_CHECK(
                row_capacity
                    <= std::numeric_limits<int64_t>::max() / row_bytes
                    && row_capacity * row_bytes == bytes_per_cell,
                "compact checkpoint descriptor row layout is inconsistent at index ",
                descriptor
            );
            const torch::Tensor& data =
                compact_summary_checkpoint_source_tensors.at(data_index);
            TORCH_CHECK(
                static_cast<int64_t>(data.element_size()) == dtype_bytes
                    && data.numel()
                        <= std::numeric_limits<int64_t>::max() / dtype_bytes
                    && data.numel() * dtype_bytes == cells * cell_stride,
                "compact checkpoint descriptor data layout is inconsistent at index ",
                descriptor
            );
            if (
                descriptor
                == checkpoint_runtime::kAgentIdToSlotAgentIdsDescriptorIndexV1
            ) {
                TORCH_CHECK(
                    data.scalar_type() == torch::kInt64
                        && data.sizes()
                            == torch::IntArrayRef({cells, capacity})
                        && data.data_ptr<int64_t>()
                            == active_agent_ids.data_ptr<int64_t>(),
                    "compact checkpoint active IDs must alias resident "
                    "active_agent_ids"
                );
                compact_checkpoint_active_ids_bound = true;
            }
            if (
                descriptor
                == checkpoint_runtime::kAgentIdToSlotSlotsDescriptorIndexV1
            ) {
                TORCH_CHECK(
                    data.scalar_type() == torch::kInt32
                        && data.sizes()
                            == torch::IntArrayRef({cells, capacity})
                        && data.data_ptr<int32_t>()
                            == active_agent_slots.data_ptr<int32_t>(),
                    "compact checkpoint active slots must alias resident "
                    "active_agent_slots"
                );
                compact_checkpoint_active_slots_bound = true;
            }

            const std::uint8_t* alive_pointer = nullptr;
            if (alive_index >= 0) {
                const torch::Tensor& mask =
                    compact_summary_checkpoint_source_tensors.at(alive_index);
                TORCH_CHECK(
                    (mask.scalar_type() == torch::kBool
                        || mask.scalar_type() == torch::kUInt8)
                        && mask.numel() == cells * alive_stride
                        && alive_stride >= row_capacity,
                    "compact checkpoint alive-mask layout is inconsistent at index ",
                    descriptor
                );
                alive_pointer = reinterpret_cast<const std::uint8_t*>(
                    mask.data_ptr()
                );
            }
            const int64_t* active_length_pointer = nullptr;
            if (active_length_index >= 0) {
                const torch::Tensor& lengths =
                    compact_summary_checkpoint_source_tensors.at(
                        active_length_index
                    );
                TORCH_CHECK(
                    lengths.scalar_type() == torch::kInt64
                        && lengths.sizes() == torch::IntArrayRef({cells}),
                    "compact checkpoint active-length source must be int64[cells]"
                );
                active_length_pointer = lengths.data_ptr<int64_t>();
                if (
                    descriptor
                        == checkpoint_runtime::kAgentIdToSlotAgentIdsDescriptorIndexV1
                    || descriptor
                        == checkpoint_runtime::kAgentIdToSlotSlotsDescriptorIndexV1
                ) {
                    TORCH_CHECK(
                        active_length_pointer
                            == active_count_state.data_ptr<int64_t>(),
                        "compact checkpoint identity pairs must share "
                        "active_count_state authority"
                    );
                }
            }

            const int64_t batch_index = descriptor
                / checkpoint_hash::kPointerBatchCapacityV1;
            const int64_t batch_field = descriptor
                % checkpoint_hash::kPointerBatchCapacityV1;
            if (batch_field == 0) {
                compact_checkpoint_batches.push_back({});
            }
            auto& batch = compact_checkpoint_batches.at(
                static_cast<size_t>(batch_index)
            );
            batch.entries[batch_field] = {
                reinterpret_cast<const std::uint8_t*>(data.data_ptr()),
                alive_pointer,
                active_length_pointer,
                static_cast<std::uint64_t>(cell_stride),
                static_cast<std::uint64_t>(bytes_per_cell),
                static_cast<std::uint64_t>(row_capacity),
                static_cast<std::uint64_t>(row_bytes),
                static_cast<std::uint64_t>(alive_stride),
                static_cast<std::uint32_t>(dtype_bytes),
                static_cast<std::int32_t>(mode),
                static_cast<std::int32_t>(raw_output_index),
                static_cast<std::int32_t>(canonical_output_index),
                static_cast<std::int32_t>(descriptor_index),
            };
            batch.count = static_cast<std::int32_t>(batch_field + 1);
        }
        TORCH_CHECK(
            compact_checkpoint_active_ids_bound
                && compact_checkpoint_active_slots_bound,
            "compact checkpoint active identity pair descriptors are missing"
        );
    }
    std::vector<turnover::TurnoverMutableTensorTriplet>
        turnover_pointer_entries;
    if (generation_turnover_enabled) {
        turnover_pointer_entries.reserve(turnover::kTurnoverMutableTensorCount);
        for (
            int64_t index = 0;
            index < turnover::kTurnoverMutableTensorCount;
            ++index
        ) {
            const auto& source = turnover_state_tensors.at(index);
            const auto& shadow = turnover_shadow_tensors.at(index);
            TORCH_CHECK(
                source.numel() % cells == 0,
                "turnover tensor element count must divide evenly by cells"
            );
            const int64_t source_bytes_per_cell = source.numel() / cells
                * static_cast<int64_t>(source.element_size());
            const int64_t shadow_bytes_per_cell = shadow.numel() / cells
                * static_cast<int64_t>(shadow.element_size());
            turnover_pointer_entries.push_back({
                static_cast<uint8_t*>(source.data_ptr()),
                static_cast<uint8_t*>(shadow.data_ptr()),
                source_bytes_per_cell,
                shadow_bytes_per_cell,
            });
        }
        C10_CUDA_CHECK(cudaMemcpyAsync(
            turnover_pointer_table_tensors.at(0).data_ptr<int64_t>(),
            turnover_pointer_entries.data(),
            turnover_pointer_entries.size()
                * sizeof(turnover::TurnoverMutableTensorTriplet),
            cudaMemcpyHostToDevice,
            stream
        ));
    }
    ControllerRunReceipt run_receipt{
        1,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        static_cast<int32_t>(cudaSuccess),
    };
    int64_t completed_generations = 0;
    int64_t exact_cpu_policy_service_calls = 0;
    int64_t exact_cpu_policy_device_to_host_operations = 0;
    int64_t exact_cpu_policy_host_to_device_operations = 0;
    int64_t exact_cpu_policy_aten_operations_lower_bound = 0;
    int64_t exact_cpu_policy_host_wall_time_ns = 0;
    int64_t exact_cpu_policy_device_to_host_time_ns = 0;
    int64_t exact_cpu_policy_host_to_device_time_ns = 0;
    int64_t exact_cpu_policy_mlp_time_ns = 0;
    int64_t exact_cpu_policy_sampling_time_ns = 0;
    int64_t exact_cpu_policy_final_sync_time_ns = 0;
    int64_t exact_cpu_policy_device_to_host_bytes = 0;
    int64_t exact_cpu_policy_host_to_device_bytes = 0;
    int64_t exact_cpu_policy_host_allocation_count_lower_bound = 0;
    int64_t exact_cpu_policy_host_allocation_bytes_lower_bound = 0;
    pybind11::list gaussian_segment_refill_receipts;
    pybind11::list gaussian_segment_boundary_receipts;
    int64_t gaussian_segment_refill_calls = 0;
    int64_t gaussian_segment_reuse_calls = 0;
    int64_t gaussian_segment_device_to_host_operations = 0;
    int64_t gaussian_segment_host_to_device_operations = 0;
    int64_t gaussian_segment_host_wall_time_ns = 0;
    int64_t gaussian_segment_device_to_host_time_ns = 0;
    int64_t gaussian_segment_host_to_device_time_ns = 0;
    int64_t gaussian_segment_raw_generation_time_ns = 0;
    int64_t gaussian_segment_transcendental_time_ns = 0;
    int64_t gaussian_segment_allocation_copy_time_ns = 0;
    int64_t gaussian_segment_state_preparation_time_ns = 0;
    int64_t gaussian_segment_tape_allocation_time_ns = 0;
    int64_t gaussian_segment_stack_pack_time_ns = 0;
    int64_t gaussian_segment_primary_h2d_submission_time_ns = 0;
    int64_t gaussian_segment_secondary_h2d_submission_time_ns = 0;
    int64_t gaussian_segment_window_base_h2d_submission_time_ns = 0;
    pybind11::list policy_birth_segment_refill_receipts;
    pybind11::dict policy_birth_identity_entry_ensure_receipt;
    bool policy_birth_identity_entry_ensure_performed = false;
    bool policy_birth_identity_entry_physical_refill_performed = false;
    resident_window_refill::PolicyWindowIdentityStateV1
        local_policy_window_identity_state;
    auto* const effective_policy_window_identity_state =
        active_policy_window_identity_state_v1 != nullptr
        ? active_policy_window_identity_state_v1
        : (
            resident_window_refill::
                hapb_frag_a_policy_window_reuse_requested_v1()
            ? &local_policy_window_identity_state
            : nullptr
        );
    int64_t policy_birth_segment_refill_calls = 0;
    int64_t policy_birth_segment_physical_refill_calls = 0;
    int64_t policy_birth_segment_resident_suffix_reuse_calls = 0;
    int64_t policy_birth_segment_device_to_host_operations = 0;
    int64_t policy_birth_segment_host_to_device_operations = 0;
    int64_t policy_birth_segment_host_wall_time_ns = 0;
    int64_t policy_birth_segment_device_to_host_time_ns = 0;
    int64_t policy_birth_segment_host_to_device_time_ns = 0;
    int64_t policy_birth_segment_torch_random_time_ns = 0;
    int64_t policy_birth_segment_compare_pack_time_ns = 0;
    int64_t policy_birth_segment_allocation_copy_time_ns = 0;
    int64_t policy_birth_identity_ensure_calls_total = 0;
    int64_t policy_birth_identity_physical_refill_calls_total = 0;
    int64_t policy_birth_identity_resident_suffix_reuse_calls_total = 0;
    int64_t policy_birth_identity_device_to_host_operations_total = 0;
    int64_t policy_birth_identity_host_to_device_operations_total = 0;
    int64_t policy_birth_identity_device_to_host_bytes_total = 0;
    int64_t policy_birth_identity_host_to_device_bytes_total = 0;
    int64_t policy_birth_identity_host_wall_time_ns_total = 0;
    int64_t policy_birth_identity_device_to_host_time_ns_total = 0;
    int64_t policy_birth_identity_host_to_device_time_ns_total = 0;
    const auto accumulate_policy_birth_identity_receipt = [&] (
        const pybind11::dict& value
    ) {
        ++policy_birth_identity_ensure_calls_total;
        const bool physical = pybind11::cast<bool>(
            value["physical_refill_performed"]
        );
        if (physical) {
            ++policy_birth_identity_physical_refill_calls_total;
        } else {
            ++policy_birth_identity_resident_suffix_reuse_calls_total;
        }
        const auto add_if_present = [&] (
            const char* field,
            int64_t* const total
        ) {
            if (value.contains(field)) {
                *total += pybind11::cast<int64_t>(value[field]);
            }
        };
        add_if_present(
            "device_to_host_operations_lower_bound",
            &policy_birth_identity_device_to_host_operations_total
        );
        add_if_present(
            "host_to_device_operations_lower_bound",
            &policy_birth_identity_host_to_device_operations_total
        );
        add_if_present(
            "device_to_host_bytes_lower_bound",
            &policy_birth_identity_device_to_host_bytes_total
        );
        add_if_present(
            "host_to_device_bytes_lower_bound",
            &policy_birth_identity_host_to_device_bytes_total
        );
        add_if_present(
            "host_wall_time_ns",
            &policy_birth_identity_host_wall_time_ns_total
        );
        add_if_present(
            "device_to_host_time_ns",
            &policy_birth_identity_device_to_host_time_ns_total
        );
        add_if_present(
            "host_to_device_time_ns",
            &policy_birth_identity_host_to_device_time_ns_total
        );
    };
    int64_t compact_checkpoint_projection_launches = 0;
    int64_t compact_checkpoint_hash_launches = 0;
    int64_t compact_lifecycle_digest_launches = 0;
    int64_t compact_reduction_launches = 0;
    int64_t compact_metadata_retention_launches = 0;
    double* const generation_summary_scalar_base =
        generation_summary_scalar_enabled
        ? generation_summary_scalar.data_ptr<double>()
        : nullptr;
    TORCH_CHECK(
        ordinary_birth_config.rng_semantics_version == 2
            ? full_tick_state.drive_noise_sample_ticks == 0
            : full_tick_state.drive_noise_sample_ticks >= ticks,
        ordinary_birth_config.rng_semantics_version == 2
            ? "TW RNGV2 drive noise forbids a future-value tape"
            : "drive-noise tape does not cover one generation"
    );
    for (
        int64_t generation_offset = 0;
        generation_offset < generation_count;
        ++generation_offset
    ) {
        int64_t absolute_generation =
            mechanics_config.generation + generation_offset;
        const auto run_control = active_direct_run_control_v2;
        if (run_control) {
            const auto manifest = run_control->manifest();
            if (manifest.generation_transaction_open) {
                TORCH_CHECK(
                    manifest.active_generation == absolute_generation,
                    "CUDA V2 run control generation identity drifted"
                );
            } else {
                if (manifest.last_boundary) {
                    absolute_generation = manifest.last_boundary->next_generation;
                }
                run_control->begin_generation(absolute_generation);
            }
        }
        FlatPatchP2Context generation_patch_p2_context = patch_p2_context;
        if (patch_p2_enabled) {
            generation_patch_p2_context.outputs.stats =
                patch_p2_stats_tensors.at(0).data_ptr<double>()
                + generation_offset * ticks
                    * eiel::cuda_architecture_b::persistent_patch_p2::kStatCount;
            generation_patch_p2_context.compact_metrics =
                patch_p2_stats_tensors.size() == 2
                ? patch_p2_stats_tensors.at(1).data_ptr<double>()
                    + generation_offset * cells
                        * controller::kPatchCompactMetricCount
                : nullptr;
        }

        FlatMechanicsConfig generation_mechanics_config = mechanics_config;
        generation_mechanics_config.generation = absolute_generation;
        generation_mechanics_config.start_tick =
            generation_offset == 0 ? mechanics_config.start_tick : 0;
        FlatFullTickState generation_full_tick_state = full_tick_state;
        FlatCompactSummaryContextV1 generation_compact_summary_context =
            compact_summary_context;
        generation_compact_summary_context.generation_index =
            generation_offset;

        static const std::function<void(int64_t, int64_t, int64_t)>
            no_inter_tick_control;
        static const std::function<void(int64_t, int64_t)>
            no_exact_cpu_policy_refill;
        static const std::function<void(int64_t, int64_t)>
            no_hapb_ordinary_birth_commit;
        static const std::function<void()>
            no_hapb_turnover_commit;
        static const std::function<void(int64_t, int64_t, int64_t)>
            no_gaussian_segment_refill;
        const auto birth_cell_scalar = [cells](
                                           const torch::Tensor& value,
                                           const char* name
                                       ) {
            TORCH_CHECK(
                value.defined() && value.numel() == cells,
                name,
                " must contain exactly one scalar per cell"
            );
            return value.reshape({cells});
        };
        std::function<void(int64_t, int64_t)> exact_cpu_policy_refill =
            no_exact_cpu_policy_refill;
        std::function<void(int64_t, int64_t)> hapb_ordinary_birth_commit =
            no_hapb_ordinary_birth_commit;
        std::function<void()> hapb_turnover_commit = no_hapb_turnover_commit;
        const int64_t policy_birth_segment_capacity =
            ordinary_birth_config.policy_tape_capacity;
        const bool policy_birth_segment_refill_enabled =
            ordinary_birth_config.enabled
            && policy_birth_segment_capacity > 0
            && policy_birth_segment_capacity
                < ticks * ordinary_birth_config.max_births_per_tick;
        const bool policy_birth_identity_entry_ensure_enabled =
            ordinary_birth_config.enabled
            && policy_birth_segment_capacity > 0
            && resident_window_refill::
                hapb_frag_a_policy_window_reuse_requested_v1();
        std::function<void(int64_t, int64_t, int64_t)>
            policy_birth_segment_refill = no_gaussian_segment_refill;
        if (
            policy_birth_segment_refill_enabled
            || policy_birth_identity_entry_ensure_enabled
        ) {
            TORCH_CHECK(
                ordinary_birth_config.policy_mutation_tape_width == 1
                    && ordinary_birth_state_tensors.at(
                        resident_window_controller::kPolicyPackedPostStateTapeV1
                    ).size(2) == 0,
                "policy birth segment refill requires compact disabled-mutation "
                "policy tape"
            );
            TORCH_CHECK(
                policy_birth_segment_capacity
                    >= ordinary_birth_config.max_births_per_tick,
                "policy birth segment capacity must cover one ordinary tick"
            );
        }
        if (policy_birth_identity_entry_ensure_enabled) {
            policy_birth_identity_entry_ensure_receipt =
                resident_window_refill::
                    ensure_fixed_mlp_policy_window_with_identity_v1(
                        ordinary_birth_state_tensors.at(
                            resident_window_controller::kPolicyRngStateV1
                        ),
                        ordinary_birth_state_tensors.at(
                            resident_window_controller::kPolicyDeltaTapeV1
                        ),
                        ordinary_birth_state_tensors.at(
                            resident_window_controller::
                                kPolicyMutationMaskTapeV1
                        ),
                        ordinary_birth_state_tensors.at(
                            resident_window_controller::
                                kPolicyCrossoverMaskPackedTapeV1
                        ),
                        ordinary_birth_state_tensors.at(
                            resident_window_controller::
                                kPolicyPackedPostStateTapeV1
                        ),
                        ordinary_birth_state_tensors.at(
                            resident_window_controller::kPolicySemanticTapeV1
                        ),
                        birth_cell_scalar(
                            ordinary_birth_state_tensors.at(
                                resident_window_controller::
                                    kPolicyAbsoluteCursorV1
                            ),
                            "policy_birth_tape_cursor"
                        ),
                        birth_cell_scalar(
                            ordinary_birth_state_tensors.at(
                                resident_window_controller::
                                    kMutationTorchSemanticDrawsV1
                            ),
                            "mutation_torch_cpu_semantic_draws"
                        ),
                        birth_cell_scalar(
                            ordinary_birth_state_tensors.at(
                                resident_window_controller::
                                    kPolicyWindowBaseV1
                            ),
                            "policy_birth_window_base"
                        ),
                        packed_width,
                        false,
                        "dense",
                        0.0,
                        0.0,
                        ordinary_birth_config.max_births_per_tick,
                        resident_window_refill::
                            kFragIdentityProducerSchemaSha256V1,
                        effective_policy_window_identity_state,
                        "ensure_before_first_consume"
                    );
            policy_birth_identity_entry_ensure_performed = true;
            policy_birth_identity_entry_physical_refill_performed =
                pybind11::cast<bool>(
                    policy_birth_identity_entry_ensure_receipt[
                        "physical_refill_performed"
                    ]
                );
            accumulate_policy_birth_identity_receipt(
                policy_birth_identity_entry_ensure_receipt
            );
        }
        if (policy_birth_segment_refill_enabled) {
            policy_birth_segment_refill = [&](
                                              const int64_t generation,
                                              const int64_t completed_tick,
                                              const int64_t next_tick
                                          ) {
                const bool turnover_boundary = next_tick == ticks;
                const int64_t required_future_episodes =
                    turnover_boundary
                    ? policy_birth_segment_capacity
                    : ordinary_birth_config.max_births_per_tick;
                auto refill_receipt =
                    resident_window_refill::
                        ensure_fixed_mlp_policy_window_with_identity_v1(
                        ordinary_birth_state_tensors.at(
                            resident_window_controller::kPolicyRngStateV1
                        ),
                        ordinary_birth_state_tensors.at(
                            resident_window_controller::kPolicyDeltaTapeV1
                        ),
                        ordinary_birth_state_tensors.at(
                            resident_window_controller::kPolicyMutationMaskTapeV1
                        ),
                        ordinary_birth_state_tensors.at(
                            resident_window_controller::
                                kPolicyCrossoverMaskPackedTapeV1
                        ),
                        ordinary_birth_state_tensors.at(
                            resident_window_controller::kPolicyPackedPostStateTapeV1
                        ),
                        ordinary_birth_state_tensors.at(
                            resident_window_controller::kPolicySemanticTapeV1
                        ),
                        birth_cell_scalar(
                            ordinary_birth_state_tensors.at(
                                resident_window_controller::kPolicyAbsoluteCursorV1
                            ),
                            "policy_birth_tape_cursor"
                        ),
                        birth_cell_scalar(
                            ordinary_birth_state_tensors.at(
                                resident_window_controller::
                                    kMutationTorchSemanticDrawsV1
                            ),
                            "mutation_torch_cpu_semantic_draws"
                        ),
                        birth_cell_scalar(
                            ordinary_birth_state_tensors.at(
                                resident_window_controller::kPolicyWindowBaseV1
                            ),
                            "policy_birth_window_base"
                        ),
                        packed_width,
                        false,
                        "dense",
                        0.0,
                        0.0,
                        required_future_episodes,
                        resident_window_refill::
                            kFragIdentityProducerSchemaSha256V1,
                        effective_policy_window_identity_state,
                        "ensure_after_consume"
                    );
                refill_receipt["segment_schema"] =
                    "eiel_cuda_architecture_b_policy_birth_segment_refill_v1";
                refill_receipt["segment_boundary"] = "native_birth_boundary_v1";
                refill_receipt["generation"] = generation;
                refill_receipt["completed_tick"] = completed_tick;
                refill_receipt["next_tick"] = next_tick;
                refill_receipt["turnover_boundary"] = turnover_boundary;
                if (refill_receipt.contains("frag_identity_trace")) {
                    for (auto item : pybind11::cast<pybind11::list>(
                            refill_receipt["frag_identity_trace"]
                        )) {
                        auto row = pybind11::cast<pybind11::dict>(item);
                        row["generation"] = generation;
                        row["completed_tick"] = completed_tick;
                        row["next_tick"] = next_tick;
                        row["turnover_boundary"] = turnover_boundary;
                    }
                }
                policy_birth_segment_refill_receipts.append(refill_receipt);
                accumulate_policy_birth_identity_receipt(refill_receipt);
                ++policy_birth_segment_refill_calls;
                if (pybind11::cast<bool>(
                        refill_receipt["physical_refill_performed"]
                    )) {
                    ++policy_birth_segment_physical_refill_calls;
                } else {
                    ++policy_birth_segment_resident_suffix_reuse_calls;
                }
                if constexpr (resident_window_refill::kOptionDM0TelemetryEnabledV1) {
                    policy_birth_segment_host_to_device_operations +=
                        pybind11::cast<int64_t>(
                            refill_receipt["host_to_device_operations_lower_bound"]
                        );
                    policy_birth_segment_device_to_host_operations +=
                        pybind11::cast<int64_t>(
                            refill_receipt["device_to_host_operations_lower_bound"]
                        );
                    policy_birth_segment_host_wall_time_ns += pybind11::cast<int64_t>(
                        refill_receipt["host_wall_time_ns"]
                    );
                    policy_birth_segment_device_to_host_time_ns += pybind11::cast<int64_t>(
                        refill_receipt["device_to_host_time_ns"]
                    );
                    policy_birth_segment_host_to_device_time_ns += pybind11::cast<int64_t>(
                        refill_receipt["host_to_device_time_ns"]
                    );
                    policy_birth_segment_torch_random_time_ns += pybind11::cast<int64_t>(
                        refill_receipt["torch_random_generation_time_ns"]
                    );
                    policy_birth_segment_compare_pack_time_ns += pybind11::cast<int64_t>(
                        refill_receipt["torch_compare_pack_time_ns"]
                    );
                    policy_birth_segment_allocation_copy_time_ns += pybind11::cast<int64_t>(
                        refill_receipt["policy_allocation_stack_copy_time_ns"]
                    );
                }
            };
        }
#ifdef EIEL_WDL_JUNGLE_EED_HAPB_FIXED_MLP_NATIVE_BINDING_V1
        TORCH_CHECK(
            ordinary_birth_config.rng_semantics_version != 2
                || !hapb_direct_policy,
            "TW RNG V2 tick-native production forbids host-authoritative policy service"
        );
        if (hapb_direct_policy) {
            TORCH_CHECK(
                patch_p2_full_mode && !deterministic_actions
                    && cells >= hapb_policy_context::kPhysicalCellsMinV1
                    && cells <= hapb_policy_context::kPhysicalCellsMaxV1
                    && capacity == hapb_policy_context::kCapacityV1
                    && observation_size == 83 && hidden_size == 64
                    && action_count == 7 && cache_width == 2
                    && output_ticks == 1,
                "HAPB direct policy service requires the WDL/P256/C512 ABI with parameterized B"
            );
            if (turnover_config.enabled) {
                const auto& direct_context =
                    hapb_policy_context::current_hapb_direct_route_context_v1();
                TORCH_CHECK(
                    static_cast<bool>(direct_context.turnover_transaction),
                    "HAPB direct generation turnover requires its installed "
                    "host-authoritative plan/commit transaction"
                );
                const auto turnover_scratch_tensor =
                    ordinary_birth_scratch_tensors.at(0);
                const int64_t turnover_scratch_stride =
                    turnover_scratch_tensor.size(1) / sizeof(int64_t);
                hapb_turnover_commit = [
                    transaction = direct_context.turnover_transaction,
                    turnover_scratch_tensor,
                    turnover_scratch_stride,
                    cells,
                    capacity,
                    &ordinary_birth_state_tensors,
                    &birth_cell_scalar
                ] {
                    const auto make_view = [
                        turnover_scratch_tensor,
                        turnover_scratch_stride,
                        cells,
                        capacity
                    ](const int64_t byte_offset) {
                        return torch::from_blob(
                            turnover_scratch_tensor.data_ptr<uint8_t>()
                                + byte_offset,
                            {cells, capacity},
                            {turnover_scratch_stride, 1},
                            torch::TensorOptions()
                                .dtype(torch::kInt64)
                                .device(turnover_scratch_tensor.device())
                        );
                    };
                    const auto committed_count = torch::from_blob(
                        turnover_scratch_tensor.data_ptr<uint8_t>()
                            + turnover::FlatTurnoverWorkLayout::kScalarCounters
                            + turnover::kAggregateCommittedCount
                                * static_cast<int64_t>(sizeof(int64_t)),
                        {cells},
                        {turnover_scratch_stride},
                        torch::TensorOptions()
                            .dtype(torch::kInt64)
                            .device(turnover_scratch_tensor.device())
                    );
                    transaction(
                        make_view(
                            turnover::FlatTurnoverWorkLayout::kRefillParentSlots
                        ),
                        make_view(
                            turnover::FlatTurnoverWorkLayout::kRefillMateSlots
                        ),
                        make_view(
                            turnover::FlatTurnoverWorkLayout::kCommittedChildSlots
                        ),
                        committed_count,
                        ordinary_birth_state_tensors.at(
                            resident_window_controller::kPolicyDeltaTapeV1
                        ),
                        ordinary_birth_state_tensors.at(
                            resident_window_controller::kPolicyMutationMaskTapeV1
                        ),
                        ordinary_birth_state_tensors.at(
                            resident_window_controller::
                                kPolicyCrossoverMaskPackedTapeV1
                        ),
                        birth_cell_scalar(
                            ordinary_birth_state_tensors.at(
                                resident_window_controller::kPolicyAbsoluteCursorV1
                            ),
                            "hapb_policy_absolute_cursor"
                        ),
                        birth_cell_scalar(
                            ordinary_birth_state_tensors.at(
                                resident_window_controller::kPolicyWindowBaseV1
                            ),
                            "hapb_policy_window_base"
                        )
                    );
                };
            }
            if (ordinary_birth_config.enabled) {
                const auto& direct_context =
                    hapb_policy_context::current_hapb_direct_route_context_v1();
                TORCH_CHECK(
                    static_cast<bool>(direct_context.ordinary_birth_transaction),
                    "HAPB direct ordinary birth requires its installed host-authoritative "
                    "plan/commit transaction"
                );
                hapb_ordinary_birth_commit = [&, transaction =
                    direct_context.ordinary_birth_transaction](
                                                const int64_t controller_tick,
                                                const int64_t output_tick
                                            ) {
                    transaction(
                        ordinary_birth_output_tensors.at(2).select(1, output_tick),
                        ordinary_birth_output_tensors.at(3).select(1, output_tick),
                        ordinary_birth_output_tensors.at(14).select(1, output_tick),
                        ordinary_birth_state_tensors.at(
                            resident_window_controller::kPolicyDeltaTapeV1
                        ),
                        ordinary_birth_state_tensors.at(
                            resident_window_controller::kPolicyMutationMaskTapeV1
                        ),
                        ordinary_birth_state_tensors.at(
                            resident_window_controller::kPolicySemanticTapeV1
                        ),
                        birth_cell_scalar(ordinary_birth_state_tensors.at(
                            resident_window_controller::kPolicyAbsoluteCursorV1
                        ), "hapb_policy_absolute_cursor"),
                        birth_cell_scalar(ordinary_birth_state_tensors.at(
                            resident_window_controller::kPolicyWindowBaseV1
                        ), "hapb_policy_window_base")
                    );
                    static_cast<void>(controller_tick);
                };
            }
            exact_cpu_policy_refill = [&](
                                          const int64_t controller_tick,
                                          const int64_t output_tick
                                      ) {
                static_cast<void>(
                    hapb_policy_context::execute_hapb_prepared_policy_boundary_v1({
                        ordered_slots.select(1, output_tick),
                        observations.select(1, output_tick),
                        death_slots.select(1, output_tick),
                        active_count.select(1, output_tick),
                        policy_refill_rows.select(1, output_tick),
                        status,
                        cross_path_exact_cpu_policy_bypass_v1(),
                        drives,
                        attack_bias,
                        action_temperature,
                        drive_logit_gain,
                        slot_tokens,
                        action_cache_actions,
                        action_cache_token,
                        action_cache_cursor,
                        action_cache_next,
                        logits.select(1, output_tick),
                        row_actions.select(1, output_tick),
                        actions_by_slot.select(1, output_tick),
                        full_tick_output_tensors.at(16)
                            .select(1, output_tick),
                        full_tick_output_tensors.at(17)
                            .select(1, output_tick),
                        full_tick_state_tensors.at(27),
                        static_cast<float>(w_hunger_gather),
                        static_cast<float>(w_pain_rest),
                        static_cast<float>(w_fatigue_rest),
                        static_cast<float>(w_pain_attack),
                        static_cast<float>(w_fear_attack),
                    })
                );
                ++exact_cpu_policy_service_calls;
                static_cast<void>(controller_tick);
            };
        } else
#endif
        if (
            patch_p2_full_mode
            && !deterministic_actions
            && ordinary_birth_config.rng_semantics_version != 2
            && EIEL_CUDA_ARCHITECTURE_B_EXACT_CPU_POLICY_CONTEXT_ALLOWED
        ) {
            exact_cpu_policy_refill = [&](
                                          const int64_t controller_tick,
                                          const int64_t output_tick
                                      ) {
                const auto service_receipt = exact_cpu_policy::
                    service_flattened_patch_fixed_mlp_cpu_exact_v1({
                        ordered_slots,
                        observations,
                        death_slots,
                        active_count,
                        policy_refill_rows,
                        policy_cache_hit_rows,
                        status,
                        cross_path_exact_cpu_policy_bypass_v1(),
                        mlp_pack,
                        drives,
                        attack_bias,
                        action_temperature,
                        drive_logit_gain,
                        slot_tokens,
                        action_cache_actions,
                        action_cache_token,
                        action_cache_cursor,
                        action_cache_next,
                        logits,
                        row_actions,
                        actions_by_slot,
                        full_tick_state_tensors.at(18),
                        full_tick_state_tensors.at(19),
                        full_tick_state_tensors.at(27),
                        cells,
                        controller_tick,
                        output_tick,
                        capacity,
                        observation_size,
                        hidden_size,
                        action_count,
                        packed_width,
                        drives.size(2),
                        cache_capacity,
                        cache_width,
                        deterministic_actions,
                        w_hunger_gather,
                        w_pain_rest,
                        w_fatigue_rest,
                        w_pain_attack,
                        w_fear_attack,
                    });
                ++exact_cpu_policy_service_calls;
                exact_cpu_policy_device_to_host_operations +=
                    service_receipt.device_to_host_operations;
                exact_cpu_policy_host_to_device_operations +=
                    service_receipt.host_to_device_operations;
                exact_cpu_policy_aten_operations_lower_bound +=
                    service_receipt.aten_operations_lower_bound;
                if constexpr (exact_cpu_policy::kOptionDM0TelemetryEnabledV1) {
                    exact_cpu_policy_host_wall_time_ns +=
                        service_receipt.host_wall_time_ns;
                    exact_cpu_policy_device_to_host_time_ns +=
                        service_receipt.device_to_host_time_ns;
                    exact_cpu_policy_host_to_device_time_ns +=
                        service_receipt.host_to_device_time_ns;
                    exact_cpu_policy_mlp_time_ns +=
                        service_receipt.cpu_mlp_time_ns;
                    exact_cpu_policy_sampling_time_ns +=
                        service_receipt.cpu_sampling_time_ns;
                    exact_cpu_policy_final_sync_time_ns +=
                        service_receipt.final_stream_synchronization_time_ns;
                    exact_cpu_policy_device_to_host_bytes +=
                        service_receipt.device_to_host_bytes_lower_bound;
                    exact_cpu_policy_host_to_device_bytes +=
                        service_receipt.host_to_device_bytes_lower_bound;
                    exact_cpu_policy_host_allocation_count_lower_bound +=
                        service_receipt.host_allocation_count_lower_bound;
                    exact_cpu_policy_host_allocation_bytes_lower_bound +=
                        service_receipt.host_allocation_bytes_lower_bound;
                }
            };
        }
        std::function<void(int64_t, int64_t, int64_t)>
            gaussian_segment_refill = no_gaussian_segment_refill;
        const int64_t ordinary_birth_gaussian_horizon =
            ordinary_birth_config.enabled
            && ordinary_birth_config.max_births_per_tick > 0
            && ticks <= std::numeric_limits<int64_t>::max()
                    / ordinary_birth_config.max_births_per_tick
            && ticks * ordinary_birth_config.max_births_per_tick
                    <= std::numeric_limits<int64_t>::max() / 1024
                ? ticks * ordinary_birth_config.max_births_per_tick * 1024
                : 0;
        const bool gaussian_segment_refill_enabled =
            ordinary_birth_gaussian_horizon > 0
            && ordinary_birth_config.gauss_tape_capacity > 0
            && ordinary_birth_config.gauss_tape_capacity
                < ordinary_birth_gaussian_horizon;
        if (gaussian_segment_refill_enabled) {
            gaussian_segment_refill = [&](
                                          const int64_t generation,
                                          const int64_t completed_tick,
                                          const int64_t next_tick
                                      ) {
                auto refill_receipt =
                    resident_window_refill::ensure_cpython_gaussian_window_v1(
                        ordinary_birth_state_tensors.at(
                            resident_window_controller::kPythonRandomWordsV1
                        ),
                        birth_cell_scalar(
                            ordinary_birth_state_tensors.at(
                                resident_window_controller::kPythonRandomIndexV1
                            ),
                            "python_random_index"
                        ),
                        birth_cell_scalar(
                            ordinary_birth_state_tensors.at(
                                resident_window_controller::kPythonRandomHasGaussV1
                            ),
                            "python_random_has_gauss"
                        ),
                        birth_cell_scalar(
                            ordinary_birth_state_tensors.at(
                                resident_window_controller::kPythonRandomGaussNextV1
                            ),
                            "python_random_gauss_next"
                        ),
                        ordinary_birth_state_tensors.at(
                            resident_window_controller::kPythonRandomPrimaryTapeV1
                        ),
                        ordinary_birth_state_tensors.at(
                            resident_window_controller::kPythonRandomSecondaryTapeV1
                        ),
                        birth_cell_scalar(
                            ordinary_birth_state_tensors.at(
                                resident_window_controller::kPythonRandomRawDrawsV1
                            ),
                            "python_random_raw_draws"
                        ),
                        birth_cell_scalar(
                            ordinary_birth_state_tensors.at(
                                resident_window_controller::kMutationPySemanticDrawsV1
                            ),
                            "mutation_py_semantic_draws"
                        ),
                        birth_cell_scalar(
                            ordinary_birth_state_tensors.at(
                                resident_window_controller::kPythonRandomWindowBaseV1
                            ),
                            "python_random_window_base"
                        ),
                        ordinary_birth_config.max_births_per_tick * 1024
                    );
                refill_receipt["segment_schema"] =
                    "host_prepared_exact_gaussian_segment_v1";
                refill_receipt["segment_boundary"] = "native_tick_boundary_v1";
                refill_receipt["generation"] = generation;
                refill_receipt["completed_tick"] = completed_tick;
                refill_receipt["next_tick"] = next_tick;
                gaussian_segment_boundary_receipts.append(refill_receipt);
                if (pybind11::cast<bool>(refill_receipt["refill_performed"])) {
                    gaussian_segment_refill_receipts.append(refill_receipt);
                    ++gaussian_segment_refill_calls;
                } else {
                    ++gaussian_segment_reuse_calls;
                }
                gaussian_segment_host_to_device_operations +=
                    pybind11::cast<int64_t>(
                        refill_receipt["host_to_device_operations_lower_bound"]
                    );
                gaussian_segment_device_to_host_operations +=
                    pybind11::cast<int64_t>(
                        refill_receipt["device_to_host_operations_lower_bound"]
                    );
                if constexpr (resident_window_refill::kOptionDM0TelemetryEnabledV1) {
                    gaussian_segment_host_wall_time_ns += pybind11::cast<int64_t>(
                        refill_receipt["host_wall_time_ns"]
                    );
                    gaussian_segment_device_to_host_time_ns += pybind11::cast<int64_t>(
                        refill_receipt["device_to_host_time_ns"]
                    );
                    gaussian_segment_host_to_device_time_ns += pybind11::cast<int64_t>(
                        refill_receipt["host_to_device_time_ns"]
                    );
                    gaussian_segment_raw_generation_time_ns += pybind11::cast<int64_t>(
                        refill_receipt["gaussian_raw_generation_time_ns"]
                    );
                    gaussian_segment_transcendental_time_ns += pybind11::cast<int64_t>(
                        refill_receipt["gaussian_transcendental_conversion_time_ns"]
                    );
                    gaussian_segment_allocation_copy_time_ns += pybind11::cast<int64_t>(
                        refill_receipt["gaussian_allocation_copy_time_ns"]
                    );
                    gaussian_segment_state_preparation_time_ns +=
                        pybind11::cast<int64_t>(
                            refill_receipt["gaussian_state_preparation_time_ns"]
                        );
                    gaussian_segment_tape_allocation_time_ns +=
                        pybind11::cast<int64_t>(
                            refill_receipt["gaussian_tape_allocation_time_ns"]
                        );
                    gaussian_segment_stack_pack_time_ns +=
                        pybind11::cast<int64_t>(
                            refill_receipt["gaussian_stack_pack_time_ns"]
                        );
                    gaussian_segment_primary_h2d_submission_time_ns +=
                        pybind11::cast<int64_t>(
                            refill_receipt["primary_h2d_submission_time_ns"]
                        );
                    gaussian_segment_secondary_h2d_submission_time_ns +=
                        pybind11::cast<int64_t>(
                            refill_receipt["secondary_h2d_submission_time_ns"]
                        );
                    gaussian_segment_window_base_h2d_submission_time_ns +=
                        pybind11::cast<int64_t>(
                            refill_receipt["window_base_h2d_submission_time_ns"]
                        );
                }
            };
        }
        TORCH_CHECK(
            ordinary_birth_config.rng_semantics_version != 2
                || (
                    !exact_cpu_policy_refill
                    && !hapb_ordinary_birth_commit
                    && !hapb_turnover_commit
                    && !policy_birth_segment_refill
                    && !gaussian_segment_refill
                ),
            "TW RNG V2 tick-native production forbids every CPU refill or host policy callback"
        );
        std::function<void(int64_t, int64_t, int64_t)> inter_tick_control =
            no_inter_tick_control;
        if (run_control) {
            inter_tick_control = [run_control](
                                     const int64_t generation,
                                     const int64_t completed_tick,
                                     const int64_t next_tick
                                 ) {
                pybind11::gil_scoped_release release;
                run_control->inter_tick_control_point(
                    generation,
                    completed_tick,
                    next_tick
                );
            };
        }
        const auto generation_receipt =
            run_flat_dynamic_policy_food_flattened_controller_v1(
        active_agent_ids.data_ptr<int64_t>(),
        active_agent_slots.data_ptr<int32_t>(),
        active_count_state.data_ptr<int64_t>(),
        slot_to_agent_id.data_ptr<int64_t>(),
        agent_id_to_slot.data_ptr<int32_t>(),
        alive_mask.data_ptr<bool>(),
        slot_tokens.data_ptr<int64_t>(),
        action_cache_actions.data_ptr<int32_t>(),
        action_cache_token.data_ptr<int64_t>(),
        action_cache_cursor.data_ptr<int32_t>(),
        action_cache_next.data_ptr<int32_t>(),
        pos_y.data_ptr<int32_t>(),
        pos_x.data_ptr<int32_t>(),
        agent_grid.data_ptr<int32_t>(),
        food_grid.data_ptr<int32_t>(),
        energy.data_ptr<float>(),
        health.data_ptr<float>(),
        injury.data_ptr<float>(),
        policy_energy_observation.data_ptr<double>(),
        rest_streak.data_ptr<int32_t>(),
        pending_free_agent_ids.data_ptr<int64_t>(),
        pending_free_slots.data_ptr<int64_t>(),
        pending_free_count.data_ptr<int64_t>(),
        energy_collected.data_ptr<float>(),
        energy_collected_exact.data_ptr<double>(),
        last_action.data_ptr<int32_t>(),
        last_last_action.data_ptr<int32_t>(),
        action_switches.data_ptr<int32_t>(),
        action_oscillations.data_ptr<int32_t>(),
        episode0_switches.data_ptr<int32_t>(),
        episode0_oscillations.data_ptr<int32_t>(),
        survival_steps.data_ptr<int32_t>(),
        mlp_pack.data_ptr<float>(),
        drives.data_ptr<float>(),
        attack_bias.data_ptr<float>(),
        action_temperature.data_ptr<float>(),
        drive_logit_gain.data_ptr<float>(),
        food_gained_scratch.data_ptr<float>(),
        ordered_agent_ids.data_ptr<int64_t>(),
        ordered_slots.data_ptr<int32_t>(),
        observations.data_ptr<float>(),
        logits.data_ptr<float>(),
        row_actions.data_ptr<int32_t>(),
        actions_by_slot.data_ptr<int32_t>(),
        pre_energy.data_ptr<float>(),
        pre_health.data_ptr<float>(),
        pre_injury.data_ptr<float>(),
        pre_survival_steps.data_ptr<int32_t>(),
        active_count.data_ptr<int64_t>(),
        policy_refill_rows.data_ptr<int64_t>(),
        policy_cache_hit_rows.data_ptr<int64_t>(),
        runtime_bookkeeping_by_tick.data_ptr<int64_t>()
            + generation_offset * ticks * 5,
        death_agent_ids.data_ptr<int64_t>(),
        death_slots.data_ptr<int64_t>(),
        death_reason_codes.data_ptr<int32_t>(),
        death_count.data_ptr<int64_t>(),
        survivor_agent_ids.data_ptr<int64_t>(),
        survivor_count.data_ptr<int64_t>(),
        lifecycle_status.data_ptr<int32_t>(),
        status.data_ptr<int32_t>(),
        first_failure_records.data_ptr<int64_t>(),
        generation_summary_scalar_enabled
            ? generation_summary_scalar_base
                + generation_offset * cells
                    * controller::kGenerationSummaryScalarWidthV1
            : nullptr,
        generation_full_tick_state,
        full_tick_outputs,
        event_state,
        ordinary_birth_state,
        ordinary_birth_outputs,
        ordinary_birth_scratch,
        ordinary_birth_config,
        turnover_tensors,
        turnover_scratch,
        turnover_outputs,
        turnover_config,
        generation_mechanics_config,
        generation_patch_p2_context,
        generation_compact_summary_context,
        cells,
        ticks,
        output_ticks,
        generation_count * ticks,
        active_capacity,
        agent_id_capacity,
        capacity,
        pending_capacity,
        height,
        width,
        view_radius,
        observation_size,
        hidden_size,
        action_count,
        packed_width,
        drives.size(2),
        cache_capacity,
        cache_width,
        deterministic_actions,
        max_food,
        effective_food_spawn_count,
        max_energy,
        start_energy,
        start_health,
        move_cost_by_cell.data_ptr<double>(),
        gather_cost,
        rest_cost,
        attack_cost,
        action_cost_mult,
        rest_cost_mult,
        food_energy,
        episode0_k,
        static_cast<float>(w_hunger_gather),
        static_cast<float>(w_pain_rest),
        static_cast<float>(w_fatigue_rest),
        static_cast<float>(w_pain_attack),
        static_cast<float>(w_fear_attack),
        exact_cpu_policy_refill,
        hapb_ordinary_birth_commit,
        policy_birth_segment_refill,
        gaussian_segment_refill,
        inter_tick_control,
        hapb_turnover_commit,
        stream,
        false
        );
        run_receipt.native_tick_iterations +=
            generation_receipt.native_tick_iterations;
        run_receipt.phase_1_launches += generation_receipt.phase_1_launches;
        run_receipt.birth_prepare_launches +=
            generation_receipt.birth_prepare_launches;
        run_receipt.birth_commit_launches +=
            generation_receipt.birth_commit_launches;
        run_receipt.birth_prepare_commit_fused_launches +=
            generation_receipt.birth_prepare_commit_fused_launches;
        run_receipt.turnover_shadow_plan_launches +=
            generation_receipt.turnover_shadow_plan_launches;
        run_receipt.turnover_cull_refill_plan_launches +=
            generation_receipt.turnover_cull_refill_plan_launches;
        run_receipt.turnover_commit_launches +=
            generation_receipt.turnover_commit_launches;
        run_receipt.turnover_publish_launches +=
            generation_receipt.turnover_publish_launches;
        run_receipt.patch_density_refresh_launches +=
            generation_receipt.patch_density_refresh_launches;
        run_receipt.gru_cleanup_launches +=
            generation_receipt.gru_cleanup_launches;
        run_receipt.generation_summary_finalize_launches +=
            generation_receipt.generation_summary_finalize_launches;
        run_receipt.host_synchronizations_inside_loop +=
            generation_receipt.host_synchronizations_inside_loop;
#if EIEL_OPTION_D_M0_TELEMETRY_ENABLED == 1
        run_receipt.gaussian_pre_refill_synchronization_time_ns.insert(
            run_receipt.gaussian_pre_refill_synchronization_time_ns.end(),
            generation_receipt.gaussian_pre_refill_synchronization_time_ns.begin(),
            generation_receipt.gaussian_pre_refill_synchronization_time_ns.end()
        );
        run_receipt.gaussian_post_copy_synchronization_time_ns.insert(
            run_receipt.gaussian_post_copy_synchronization_time_ns.end(),
            generation_receipt.gaussian_post_copy_synchronization_time_ns.begin(),
            generation_receipt.gaussian_post_copy_synchronization_time_ns.end()
        );
#endif
        if (
            generation_receipt.cuda_status
            != static_cast<int32_t>(cudaSuccess)
        ) {
            run_receipt.cuda_status = generation_receipt.cuda_status;
            break;
        }
        if (
            (exact_partial_tick_span || exact_tick_native_span)
            && !generation_turnover_enabled
        ) {
            turnover_output_tensors.at(9).fill_(absolute_generation);
            turnover_output_tensors.at(10).fill_(start_tick + ticks);
        }
        if (compact_checkpoint_hash_enabled) {
            checkpoint_hash::SnapshotOutputsV1 checkpoint_outputs{
                compact_summary_checkpoint_output_tensors.at(0)
                    .data_ptr<std::uint8_t>(),
                compact_summary_checkpoint_output_tensors.at(1)
                    .data_ptr<std::uint8_t>(),
                compact_summary_checkpoint_output_tensors.at(2)
                    .data_ptr<std::int32_t>(),
                compact_summary_checkpoint_output_tensors.at(3)
                    .data_ptr<std::uint8_t>(),
                generation_count,
                cells,
                compact_checkpoint_descriptor_count,
                compact_checkpoint_descriptor_count,
                generation_offset,
            };
            for (const auto& batch : compact_checkpoint_batches) {
                checkpoint_hash::launch_checkpoint_tensor_hash_snapshot_v1(
                    batch,
                    checkpoint_outputs,
                    stream
                );
                ++compact_checkpoint_hash_launches;
            }
            const cudaError_t checkpoint_launch_status = cudaPeekAtLastError();
            if (checkpoint_launch_status != cudaSuccess) {
                run_receipt.cuda_status =
                    static_cast<std::int32_t>(checkpoint_launch_status);
                break;
            }
        }
        if (compact_summary_planes_enabled) {
            const reductions::ReductionLaunchStatusV1 reduction_launch_status =
                reductions::launch_compact_summary_reductions_v1(
                    reductions::ReductionInputsV1{
                        {
                            {
                                event_state.typed_event_type,
                                event_state.typed_generation,
                                event_state.typed_tick,
                                event_state.typed_sequence,
                                event_state.typed_actor_id,
                                event_state.typed_target_id,
                                event_state.typed_actor_slot,
                                event_state.typed_target_slot,
                                event_state.typed_parent_id,
                                event_state.typed_child_id,
                                event_state.typed_founder_id,
                                event_state.typed_lineage_depth,
                                event_state.typed_subtype,
                                event_state.typed_reason,
                                event_state.typed_success,
                                event_state.typed_value,
                                event_state.capacity,
                                event_state.capacity,
                            },
                            event_state.active_count,
                        },
                        {
                            generation_summary_scalar_base,
                            generation_count,
                            cells,
                            controller::kGenerationSummaryScalarWidthV1,
                        },
                        {
                            active_agent_ids.data_ptr<std::int64_t>(),
                            active_count_state.data_ptr<std::int64_t>(),
                            ordinary_birth_state_tensors.at(28)
                                .data_ptr<std::int64_t>(),
                            ordinary_birth_state_tensors.at(27)
                                .data_ptr<std::int64_t>(),
                            capacity,
                            capacity,
                        },
                        {
                            ordinary_birth_output_tensors.at(15)
                                .data_ptr<std::int64_t>(),
                            ordinary_birth_output_tensors.at(13)
                                .data_ptr<std::int64_t>(),
                            ticks,
                            ticks,
                            ticks * reductions::kBirthPlanReasonCountV1,
                            reductions::kBirthPlanReasonCountV1,
                            reductions::kBirthPlanReasonCountV1,
                        },
                        generation_offset,
                        cells,
                    },
                    reductions::ReductionOutputViewV1{
                        reinterpret_cast<reductions::ReductionResultV1*>(
                            compact_summary_settlement_tensors
                                .at(kSettlementReductionDeviceV1)
                                .data_ptr<std::uint8_t>()
                        ),
                        generation_count,
                        cells,
                    },
                    stream
                );
            if (reduction_launch_status != reductions::kReductionLaunchSuccessV1) {
                run_receipt.cuda_status =
                    static_cast<std::int32_t>(cudaErrorInvalidValue);
                break;
            }
            ++compact_reduction_launches;
            const cudaError_t reduction_kernel_status = cudaPeekAtLastError();
            if (reduction_kernel_status != cudaSuccess) {
                run_receipt.cuda_status =
                    static_cast<std::int32_t>(reduction_kernel_status);
                break;
            }
            const lifecycle_digest::FlatTypedLifecycleColumnsV1 columns{
                event_state.typed_event_type,
                event_state.typed_generation,
                event_state.typed_tick,
                event_state.typed_sequence,
                event_state.typed_actor_id,
                event_state.typed_target_id,
                event_state.typed_actor_slot,
                event_state.typed_target_slot,
                event_state.typed_parent_id,
                event_state.typed_child_id,
                event_state.typed_founder_id,
                event_state.typed_lineage_depth,
                event_state.typed_subtype,
                event_state.typed_reason,
                event_state.typed_success,
                event_state.typed_value,
            };
            lifecycle_digest::launch_compact_summary_lifecycle_digest_v1(
                lifecycle_digest::FlatLifecycleDigestInputsV1{
                    columns,
                    event_state.active_count,
                    event_state.capacity,
                },
                lifecycle_digest::LifecycleDigestOutputsV1{
                    compact_summary_settlement_tensors
                        .at(kSettlementLifecycleDigestDeviceV1)
                        .data_ptr<std::uint8_t>(),
                    compact_summary_settlement_tensors
                        .at(kSettlementLifecycleStatusDeviceV1)
                        .data_ptr<std::int32_t>(),
                    compact_summary_settlement_tensors
                        .at(kSettlementLifecycleValidityDeviceV1)
                        .data_ptr<std::uint8_t>(),
                    generation_count,
                    cells,
                    generation_offset,
                },
                stream
            );
            ++compact_lifecycle_digest_launches;
            const cudaError_t lifecycle_launch_status = cudaPeekAtLastError();
            if (lifecycle_launch_status != cudaSuccess) {
                run_receipt.cuda_status =
                    static_cast<std::int32_t>(lifecycle_launch_status);
                break;
            }

            const auto& metadata_initial_host =
                compact_summary_settlement_tensors.at(
                    kSettlementMetadataInitialHostV1
                );
            const std::int64_t* const metadata_initial =
                metadata_initial_host.data_ptr<std::int64_t>();
            metadata_retention::MetadataRetentionInputsV1 metadata_inputs{};
            metadata_inputs.upstream_status = status.data_ptr<std::int32_t>();
            metadata_inputs.generation =
                turnover_output_tensors.at(9).data_ptr<std::int64_t>();
            metadata_inputs.tick =
                turnover_output_tensors.at(10).data_ptr<std::int64_t>();
            metadata_inputs.active_count =
                active_count_state.data_ptr<std::int64_t>();
            metadata_inputs.agent_id_to_slot_prefix_length =
                active_count_state.data_ptr<std::int64_t>();
            metadata_inputs.slot_free_queue_count =
                ordinary_birth_state_tensors.at(27).data_ptr<std::int64_t>();
            metadata_inputs.pending_free_count =
                pending_free_count.data_ptr<std::int64_t>();
            metadata_inputs.event_cursor =
                generation_compact_summary_context.event_cursor_end
                + generation_offset * cells;
            metadata_inputs.runtime_bookkeeping_by_tick =
                runtime_bookkeeping_by_tick.data_ptr<std::int64_t>();
            metadata_inputs.population_next_id =
                ordinary_birth_state_tensors.at(28).data_ptr<std::int64_t>();
            metadata_inputs.policy_birth_tape_cursor =
                ordinary_birth_state_tensors.at(91).data_ptr<std::int64_t>();
            metadata_inputs.shared_env_policy_pcg64_state_words =
                full_tick_state_tensors.at(18).data_ptr<std::uint64_t>();
            metadata_inputs.shared_env_policy_pcg64_buffer_words =
                full_tick_state_tensors.at(19).data_ptr<std::int64_t>();
            metadata_inputs.runner_pcg64_state_words =
                full_tick_state_tensors.at(16).data_ptr<std::uint64_t>();
            metadata_inputs.runner_pcg64_buffer_words =
                full_tick_state_tensors.at(17).data_ptr<std::int64_t>();
            metadata_inputs.population_pcg64_state_words =
                ordinary_birth_state_tensors.at(79).data_ptr<std::uint64_t>();
            metadata_inputs.population_pcg64_buffer_words =
                ordinary_birth_state_tensors.at(80).data_ptr<std::int64_t>();
            metadata_inputs.birth_spawn_pcg64_state_words =
                metadata_inputs.shared_env_policy_pcg64_state_words;
            metadata_inputs.birth_spawn_pcg64_buffer_words =
                metadata_inputs.shared_env_policy_pcg64_buffer_words;
            metadata_inputs.python_random_words =
                ordinary_birth_state_tensors.at(35).data_ptr<std::uint32_t>();
            metadata_inputs.python_random_index =
                ordinary_birth_state_tensors.at(36).data_ptr<std::int64_t>();
            metadata_inputs.python_random_has_gauss =
                ordinary_birth_state_tensors.at(37).data_ptr<bool>();
            metadata_inputs.python_random_gauss_next =
                ordinary_birth_state_tensors.at(38).data_ptr<double>();
            metadata_inputs.mutation_py_semantic_draws =
                ordinary_birth_state_tensors.at(42).data_ptr<std::int64_t>();
            metadata_inputs.mutation_np_semantic_draws =
                ordinary_birth_state_tensors.at(81).data_ptr<std::int64_t>();
            metadata_inputs.mutation_torch_cpu_semantic_draws =
                ordinary_birth_state_tensors.at(92).data_ptr<std::int64_t>();
            metadata_inputs.policy_np_semantic_draws =
                full_tick_state_tensors.at(27).data_ptr<std::int64_t>();
            metadata_inputs.metrics_np_semantic_draws =
                full_tick_state_tensors.at(28).data_ptr<std::int64_t>();
            metadata_inputs.checkpoint_raw_digests =
                compact_summary_checkpoint_output_tensors.at(0)
                    .data_ptr<std::uint8_t>();
            metadata_inputs.checkpoint_descriptor_status =
                compact_summary_checkpoint_output_tensors.at(2)
                    .data_ptr<std::int32_t>();
            metadata_inputs.checkpoint_descriptor_validity =
                compact_summary_checkpoint_output_tensors.at(3)
                    .data_ptr<std::uint8_t>();
            metadata_inputs.policy_energy_observation =
                policy_energy_observation.data_ptr<double>();
            metadata_inputs.alive_mask = alive_mask.data_ptr<bool>();
            metadata_inputs.slot_to_agent_id =
                slot_to_agent_id.data_ptr<std::int64_t>();
            metadata_inputs.initial_lifecycle_event_cursor = metadata_initial[0];
            metadata_inputs.initial_runtime_event_cursor = metadata_initial[1];
            metadata_inputs.initial_generation_calls = metadata_initial[2];
            metadata_inputs.restore_count = metadata_initial[3];
            metadata_inputs.initial_policy_py_semantic_draws = metadata_initial[4];
            metadata_inputs.generation_index = generation_offset;
            metadata_inputs.generation_count = generation_count;
            metadata_inputs.cell_count = cells;
            metadata_inputs.configured_steps_per_generation = ticks;
            metadata_inputs.capacity = capacity;
            metadata_inputs.agent_id_capacity = agent_id_capacity;
            metadata_inputs.event_capacity = event_capacity;
            metadata_inputs.policy_birth_tape_capacity =
                ordinary_birth_state_tensors.at(86).size(1);
            metadata_inputs.mutation_torch_cpu_state_bytes =
                ordinary_birth_state_tensors.at(85).size(1);
            metadata_inputs.checkpoint_output_field_count =
                compact_summary_checkpoint_output_tensors.at(0).size(2);
            metadata_inputs.checkpoint_descriptor_count =
                compact_checkpoint_descriptor_count;
            metadata_inputs.env_policy_alias = env_policy_alias ? 1U : 0U;
            metadata_retention::launch_compact_summary_metadata_retention_v1(
                metadata_inputs,
                reinterpret_cast<
                    metadata_retention::CompactSummaryMetadataRetentionV1*>(
                    compact_summary_settlement_tensors
                        .at(kSettlementMetadataRetentionDeviceV1)
                        .data_ptr<std::uint8_t>()
                ),
                reinterpret_cast<void*>(stream)
            );
            ++compact_metadata_retention_launches;
            const cudaError_t metadata_launch_status = cudaPeekAtLastError();
            if (metadata_launch_status != cudaSuccess) {
                run_receipt.cuda_status =
                    static_cast<std::int32_t>(metadata_launch_status);
                break;
            }
        }
        ++completed_generations;
        if (
            run_control
            && settle_generation_boundary_v2(
                run_control,
                absolute_generation,
                stream
            ) != "continue"
        ) {
            break;
        }
    }
    host_settlement::HostSettlementContextV1 compact_host_settlement_context{};
    settlement::SettlementReceiptV1 compact_settlement_receipt{
        settlement::kSettlementInvalidArgumentsV1,
        0,
        0,
        0,
        0,
        0,
        0,
    };
    if (
        compact_summary_planes_enabled
        && completed_generations == generation_count
        && run_receipt.cuda_status == static_cast<int32_t>(cudaSuccess)
    ) {
        const auto transfer = [](
                                  const torch::Tensor& device_source,
                                  const torch::Tensor& host_destination
                              ) -> settlement::ByteTransferV1 {
            return {
                device_source.data_ptr(),
                host_destination.data_ptr(),
                static_cast<std::size_t>(
                    device_source.numel()
                    * static_cast<int64_t>(device_source.element_size())
                ),
            };
        };
        const auto& authority_device = compact_summary_settlement_tensors.at(
            kSettlementAuthorityDeviceV1
        );
        const auto& authority_host = compact_summary_settlement_tensors.at(
            kSettlementAuthorityHostV1
        );
        compact_host_settlement_context.authority_rows = reinterpret_cast<
            projection::CompactSummaryGenerationCellAuthorityV1*>(
            authority_host.data_ptr<std::uint8_t>()
        );
        compact_host_settlement_context.reporting_retained_bytes =
            compact_summary_settlement_tensors
                .at(kSettlementReportingHostV1)
                .data_ptr<std::uint8_t>();
        compact_host_settlement_context.generation_summary_values =
            compact_summary_settlement_tensors
                .at(kSettlementGenerationSummaryHostV1)
                .data_ptr<double>();
        compact_host_settlement_context.patch_metrics_values =
            patch_p2_full_mode
            ? compact_summary_settlement_tensors
                .at(kSettlementPatchMetricsHostV1)
                .data_ptr<double>()
            : nullptr;
        compact_host_settlement_context.lifecycle_digest_values =
            compact_summary_settlement_tensors
                .at(kSettlementLifecycleDigestHostV1)
                .data_ptr<std::uint8_t>();
        compact_host_settlement_context.lifecycle_status_values =
            compact_summary_settlement_tensors
                .at(kSettlementLifecycleStatusHostV1)
                .data_ptr<std::int32_t>();
        compact_host_settlement_context.lifecycle_validity_values =
            compact_summary_settlement_tensors
                .at(kSettlementLifecycleValidityHostV1)
                .data_ptr<std::uint8_t>();
        compact_host_settlement_context.checkpoint_raw_digests =
            compact_summary_settlement_tensors
                .at(kSettlementCheckpointRawHostV1)
                .data_ptr<std::uint8_t>();
        compact_host_settlement_context.checkpoint_canonical_digests =
            compact_summary_settlement_tensors
                .at(kSettlementCheckpointCanonicalHostV1)
                .data_ptr<std::uint8_t>();
        compact_host_settlement_context.checkpoint_status_values =
            compact_summary_settlement_tensors
                .at(kSettlementCheckpointStatusHostV1)
                .data_ptr<std::int32_t>();
        compact_host_settlement_context.checkpoint_validity_values =
            compact_summary_settlement_tensors
                .at(kSettlementCheckpointValidityHostV1)
                .data_ptr<std::uint8_t>();
        compact_host_settlement_context.reduction_rows = reinterpret_cast<
            const reductions::ReductionResultV1*>(
            compact_summary_settlement_tensors
                .at(kSettlementReductionHostV1)
                .data_ptr<std::uint8_t>()
        );
        compact_host_settlement_context.metadata_rows = reinterpret_cast<
            const metadata_retention::CompactSummaryMetadataRetentionV1*>(
            compact_summary_settlement_tensors
                .at(kSettlementMetadataRetentionHostV1)
                .data_ptr<std::uint8_t>()
        );
        compact_host_settlement_context.checkpoint_due_values =
            compact_summary_settlement_tensors
                .at(kSettlementCheckpointDueHostV1)
                .data_ptr<bool>();
        compact_host_settlement_context.import_digest_values =
            compact_summary_settlement_tensors
                .at(kSettlementImportDigestHostV1)
                .data_ptr<std::uint8_t>();
        compact_host_settlement_context.host_template_runtime =
            &compact_host_template_runtime;
        compact_host_settlement_context.completed_generation_values =
            compact_summary_settlement_tensors
                .at(kSettlementCompletedGenerationsHostV1)
                .data_ptr<std::int64_t>();
        compact_host_settlement_context.host_token_resolver_rows =
            compact_summary_settlement_tensors
                .at(kSettlementHostTokenResolverPlanHostV1)
                .data_ptr<std::uint64_t>();
        compact_host_settlement_context.checkpoint_descriptor_specs =
            compact_summary_checkpoint_descriptor_specs.data_ptr<std::int64_t>();
        compact_host_settlement_context.host_token_resolver_row_count =
            static_cast<std::uint64_t>(
                compact_summary_settlement_tensors
                    .at(kSettlementHostTokenResolverPlanHostV1)
                    .size(0)
            );
        compact_host_settlement_context.token_resolver_failure = {};
        compact_host_settlement_context.initial_generation = generation;
        compact_host_settlement_context.generation_count = generation_count;
        compact_host_settlement_context.cell_count = cells;
        compact_host_settlement_context.checkpoint_field_count =
            compact_summary_checkpoint_output_tensors.at(0).size(2);
        compact_host_settlement_context.checkpoint_descriptor_count =
            compact_checkpoint_descriptor_count;
        compact_host_settlement_context.reporting_retained_row_bytes =
            compact_reporting_retained_bytes_per_generation_cell;
        compact_host_settlement_context.generation_summary_width =
            controller::kGenerationSummaryScalarWidthV1;
        compact_host_settlement_context.patch_metrics_width =
            patch_p2_full_mode
            ? host_settlement::kPatchMetricWidthV1
            : 0;
        compact_host_settlement_context.world_code = patch_p2_full_mode
            ? compact_summary::kWorldPatch
            : compact_summary::kWorldFlat;
        compact_host_settlement_context.reporting_row_capacity =
            compact_reporting_row_capacity;
        compact_host_settlement_context.completed_rows = 0;
        compact_host_settlement_context.first_failed_row = -1;
        compact_host_settlement_context.status =
            host_settlement::kHostSettlementInvalidArgumentsV1;

        const auto& validity = full_tick_output_tensors.at(
            compact_summary_plane_offset + compact_summary::kPlaneValidity
        );
        const auto& defined_mask = full_tick_output_tensors.at(
            compact_summary_plane_offset + compact_summary::kPlaneDefinedMask
        );
        const auto& status_codes = full_tick_output_tensors.at(
            compact_summary_plane_offset + compact_summary::kPlaneStatusCodes
        );
        const auto& integer_values = full_tick_output_tensors.at(
            compact_summary_plane_offset + compact_summary::kPlaneIntegerValues
        );
        const auto& float_values = full_tick_output_tensors.at(
            compact_summary_plane_offset + compact_summary::kPlaneFloatValues
        );
        const auto& digest_values = full_tick_output_tensors.at(
            compact_summary_plane_offset + compact_summary::kPlaneDigestValues
        );
        const auto& text_values = full_tick_output_tensors.at(
            compact_summary_plane_offset + compact_summary::kPlaneTextValues
        );
        const compact_summary::CompactSummaryViewV2 compact_output_view{
            generation_count,
            cells,
            {validity.data_ptr<std::uint8_t>(), cells, 1},
            {defined_mask.data_ptr<compact_summary::DefinedMaskV1>(), cells, 1},
            {
                status_codes.data_ptr<std::int32_t>(),
                cells * compact_summary::kStatusFieldCountV2,
                compact_summary::kStatusFieldCountV2,
            },
            {
                integer_values.data_ptr<std::int64_t>(),
                cells * compact_summary::kIntegerFieldCountV2,
                compact_summary::kIntegerFieldCountV2,
            },
            {
                float_values.data_ptr<double>(),
                cells * compact_summary::kFloatFieldCountV2,
                compact_summary::kFloatFieldCountV2,
            },
            {
                digest_values.data_ptr<std::uint8_t>(),
                cells * compact_summary::kDigestFieldCountV2
                    * compact_summary::kDigestBytesV2,
                compact_summary::kDigestFieldCountV2
                    * compact_summary::kDigestBytesV2,
            },
            {
                text_values.data_ptr<std::uint8_t>(),
                cells * compact_summary::kTextFieldCountV2
                    * compact_summary::kTextRecordBytesV2,
                compact_summary::kTextFieldCountV2
                    * compact_summary::kTextRecordBytesV2,
            },
        };
        compact_settlement_receipt = settlement::enqueue_settlement_v1(
            settlement::SettlementRequestV1{
                transfer(authority_device, authority_host),
                transfer(
                    compact_summary_reporting_tensors.at(0),
                    compact_summary_settlement_tensors.at(kSettlementReportingHostV1)
                ),
                transfer(
                    compact_summary_checkpoint_output_tensors.at(0),
                    compact_summary_settlement_tensors.at(
                        kSettlementCheckpointRawHostV1
                    )
                ),
                transfer(
                    compact_summary_checkpoint_output_tensors.at(1),
                    compact_summary_settlement_tensors.at(
                        kSettlementCheckpointCanonicalHostV1
                    )
                ),
                transfer(
                    compact_summary_checkpoint_output_tensors.at(2),
                    compact_summary_settlement_tensors.at(
                        kSettlementCheckpointStatusHostV1
                    )
                ),
                transfer(
                    compact_summary_checkpoint_output_tensors.at(3),
                    compact_summary_settlement_tensors.at(
                        kSettlementCheckpointValidityHostV1
                    )
                ),
                transfer(
                    generation_summary_scalar,
                    compact_summary_settlement_tensors.at(
                        kSettlementGenerationSummaryHostV1
                    )
                ),
                patch_p2_full_mode
                ? transfer(
                      patch_p2_stats_tensors.at(1),
                      compact_summary_settlement_tensors.at(
                          kSettlementPatchMetricsHostV1
                      )
                  )
                : settlement::ByteTransferV1{nullptr, nullptr, 0},
                transfer(
                    compact_summary_settlement_tensors.at(
                        kSettlementLifecycleDigestDeviceV1
                    ),
                    compact_summary_settlement_tensors.at(
                        kSettlementLifecycleDigestHostV1
                    )
                ),
                transfer(
                    compact_summary_settlement_tensors.at(
                        kSettlementLifecycleStatusDeviceV1
                    ),
                    compact_summary_settlement_tensors.at(
                        kSettlementLifecycleStatusHostV1
                    )
                ),
                transfer(
                    compact_summary_settlement_tensors.at(
                        kSettlementLifecycleValidityDeviceV1
                    ),
                    compact_summary_settlement_tensors.at(
                        kSettlementLifecycleValidityHostV1
                    )
                ),
                transfer(
                    compact_summary_settlement_tensors.at(
                        kSettlementReductionDeviceV1
                    ),
                    compact_summary_settlement_tensors.at(
                        kSettlementReductionHostV1
                    )
                ),
                transfer(
                    compact_summary_settlement_tensors.at(
                        kSettlementMetadataRetentionDeviceV1
                    ),
                    compact_summary_settlement_tensors.at(
                        kSettlementMetadataRetentionHostV1
                    )
                ),
                transfer(
                    turnover_output_tensors.at(9),
                    compact_summary_settlement_tensors.at(
                        kSettlementCompletedGenerationsHostV1
                    )
                ),
                &compact_host_settlement_context,
                host_settlement::settle_compact_summary_rows_callback_v1,
                reinterpret_cast<
                    const projection::CompactSummaryGenerationCellAuthorityV1*>(
                    authority_host.data_ptr<std::uint8_t>()
                ),
                reinterpret_cast<
                    projection::CompactSummaryGenerationCellAuthorityV1*>(
                    authority_device.data_ptr<std::uint8_t>()
                ),
                generation_count * cells,
                compact_output_view,
                compact_summary_settlement_tensors
                    .at(kSettlementProducerStatusDeviceV1)
                    .data_ptr<std::int32_t>(),
            },
            stream
        );
        if (compact_settlement_receipt.status != settlement::kSettlementSuccessV1) {
            run_receipt.cuda_status = static_cast<std::int32_t>(cudaErrorUnknown);
        }
    }
    const auto final_synchronization_started =
        std::chrono::steady_clock::now();
    const cudaError_t final_synchronization_status = cudaStreamSynchronize(stream);
    const auto final_synchronization_finished =
        std::chrono::steady_clock::now();
    const auto final_stream_synchronization_time_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            final_synchronization_finished - final_synchronization_started
        ).count();
    const auto last_useful_kernel_completion_observed_host_time_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            final_synchronization_finished.time_since_epoch()
        ).count();
    run_receipt.final_host_synchronizations = 1;
    if (
        run_receipt.cuda_status == static_cast<int32_t>(cudaSuccess)
        && final_synchronization_status != cudaSuccess
    ) {
        run_receipt.cuda_status =
            static_cast<int32_t>(final_synchronization_status);
    }
    std::vector<int64_t> completed_generations_by_cell;
    if (
        run_receipt.cuda_status == static_cast<int32_t>(cudaSuccess)
        && !generation_turnover_enabled
    ) {
        completed_generations_by_cell.assign(
            static_cast<size_t>(cells),
            generation + completed_generations
        );
    }
    if (
        run_receipt.cuda_status == static_cast<int32_t>(cudaSuccess)
        && generation_turnover_enabled
    ) {
        completed_generations_by_cell.resize(static_cast<size_t>(cells));
        cudaError_t completion_copy_status = cudaSuccess;
        if (compact_summary_planes_enabled) {
            const auto& completed_generations_host =
                compact_summary_settlement_tensors.at(
                    kSettlementCompletedGenerationsHostV1
                );
            std::memcpy(
                completed_generations_by_cell.data(),
                completed_generations_host.data_ptr<std::int64_t>(),
                static_cast<size_t>(cells) * sizeof(int64_t)
            );
        } else {
            completion_copy_status = cudaMemcpy(
                completed_generations_by_cell.data(),
                turnover_output_tensors.at(9).data_ptr<int64_t>(),
                static_cast<size_t>(cells) * sizeof(int64_t),
                cudaMemcpyDeviceToHost
            );
        }
        if (completion_copy_status != cudaSuccess) {
            run_receipt.cuda_status =
                static_cast<int32_t>(completion_copy_status);
        } else {
            completed_generations = generation_count;
            for (const int64_t next_generation : completed_generations_by_cell) {
                const int64_t cell_completed = std::clamp<int64_t>(
                    next_generation - generation,
                    0,
                    generation_count
                );
                completed_generations = std::min(
                    completed_generations,
                    cell_completed
                );
            }
        }
    }
    std::vector<int64_t> first_failure_records_host(
        static_cast<size_t>(cells * controller::kFirstFailureRecordWidthV1)
    );
    if (run_receipt.cuda_status == static_cast<int32_t>(cudaSuccess)) {
        const cudaError_t failure_copy_status = cudaMemcpy(
            first_failure_records_host.data(),
            first_failure_records.data_ptr<int64_t>(),
            first_failure_records_host.size() * sizeof(int64_t),
            cudaMemcpyDeviceToHost
        );
        if (failure_copy_status != cudaSuccess) {
            run_receipt.cuda_status = static_cast<int32_t>(failure_copy_status);
        }
    }
    TORCH_CHECK(
        run_receipt.cuda_status == static_cast<int32_t>(cudaSuccess),
        "flattened CUDA controller failed at its final boundary: ",
        cudaGetErrorString(static_cast<cudaError_t>(run_receipt.cuda_status))
    );
    pybind11::dict receipt;
    pybind11::list first_failure_by_cell;
    for (int64_t cell = 0; cell < cells; ++cell) {
        const int64_t* const record = first_failure_records_host.data()
            + cell * controller::kFirstFailureRecordWidthV1;
        pybind11::dict published_record;
        published_record["generation"] =
            record[controller::kFirstFailureGeneration];
        published_record["tick"] = record[controller::kFirstFailureTick];
        published_record["stage"] = record[controller::kFirstFailureStage];
        published_record["stage_name"] =
            controller::controller_failure_stage_name_v1(
                record[controller::kFirstFailureStage]
            );
        published_record["status"] = record[controller::kFirstFailureStatus];
        published_record["failure_code"] =
            record[controller::kFirstFailureCode];
        const bool tick_core_status_failure =
            record[controller::kFirstFailureStage]
                == controller::kControllerFailureStageTickCoreAndBirthPlan
            && record[controller::kFirstFailureStatus] == 1;
        const int64_t tick_core_predicate_mask = tick_core_status_failure
            ? record[controller::kFirstFailureCode]
            : controller::kFirstFailureNoFailureSentinelV1;
        pybind11::list tick_core_predicate_names;
        if (tick_core_status_failure && tick_core_predicate_mask > 0) {
            for (
                int64_t bit = 0;
                bit < controller::kTickCorePredicateCountV1;
                ++bit
            ) {
                if (
                    (tick_core_predicate_mask
                        & (static_cast<int64_t>(1) << bit))
                    != 0
                ) {
                    tick_core_predicate_names.append(
                        controller::tick_core_predicate_name_v1(bit)
                    );
                }
            }
        }
        published_record["tick_core_predicate_mask"] =
            tick_core_predicate_mask;
        published_record["tick_core_predicate_names"] =
            std::move(tick_core_predicate_names);
        published_record["tick_core_predicate_mask_is_first_failure"] =
            tick_core_status_failure;
        published_record["cell"] = record[controller::kFirstFailureCell];
        published_record["attempted_index"] =
            record[controller::kFirstFailureAttemptedIndex];
        published_record["required_capacity"] =
            record[controller::kFirstFailureRequiredCapacity];
        published_record["available_capacity"] =
            record[controller::kFirstFailureAvailableCapacity];
        published_record["sequence"] =
            record[controller::kFirstFailureSequence];
        first_failure_by_cell.append(std::move(published_record));
    }
    receipt["first_failure_by_cell"] = std::move(first_failure_by_cell);
    receipt["tick_core_predicate_diagnostic_schema_version"] =
        controller::kTickCorePredicateDiagnosticSchemaVersionV1;
    receipt["tick_core_predicate_mask_authority"] =
        "diagnostic_only_first_short_circuited_predicate";
    receipt["first_failure_no_failure_sentinel"] =
        controller::kFirstFailureNoFailureSentinelV1;
    if (
        option_d_m6_birth_diagnostics_enabled_v1()
        && (!ordinary_births_enabled || ordinary_birth_output_tensors.size() < 45)
    ) {
        pybind11::dict birth_diagnostics;
        birth_diagnostics["schema_version"] =
            "eiel_option_d_m6_direct_birth_window_diagnostics_v1";
        birth_diagnostics["authority"] =
            "diagnostic_only_no_execution_authority";
        birth_diagnostics["environment_knob"] =
            "EIEL_OPTION_D_M6_BIRTH_DIAGNOSTICS";
        birth_diagnostics["capture_status"] =
            "skipped_birth_entrypoint_surface_not_enabled";
        birth_diagnostics["ordinary_births_enabled_argument"] =
            ordinary_births_enabled;
        birth_diagnostics["ordinary_birth_config_enabled"] =
            ordinary_birth_config.enabled;
        birth_diagnostics["ordinary_birth_output_tensor_count"] =
            static_cast<std::int64_t>(ordinary_birth_output_tensors.size());
        birth_diagnostics["max_births_per_tick"] =
            ordinary_birth_config.max_births_per_tick;
        birth_diagnostics["gauss_tape_capacity"] =
            ordinary_birth_config.gauss_tape_capacity;
        birth_diagnostics["policy_tape_capacity"] =
            ordinary_birth_config.policy_tape_capacity;
        birth_diagnostics["policy_mutation_tape_width"] =
            ordinary_birth_config.policy_mutation_tape_width;
        birth_diagnostics["torch_state_bytes"] =
            ordinary_birth_config.torch_state_bytes;
        birth_diagnostics["hmem_width"] = ordinary_birth_config.hmem_width;
        birth_diagnostics["cells"] = cells;
        birth_diagnostics["semantic_ticks"] = ticks;
        receipt["option_d_m6_birth_diagnostics"] =
            std::move(birth_diagnostics);
    }
    if (
        option_d_m6_birth_diagnostics_enabled_v1()
        && ordinary_births_enabled
        && ordinary_birth_output_tensors.size() >= 45
    ) {
        const std::int64_t diagnostic_ticks =
            ordinary_birth_output_tensors.at(14).dim() >= 2
            ? ordinary_birth_output_tensors.at(14).size(1)
            : ticks;
        const std::int64_t birth_tick_count = cells * diagnostic_ticks;
        const std::int64_t reason_width =
            ordinary_birth_output_tensors.at(13).dim() >= 3
            ? ordinary_birth_output_tensors.at(13).size(2)
            : 0;
        const std::int64_t sampled_cell_count = std::min<std::int64_t>(
            cells,
            4
        );
        const std::int64_t sampled_tick_count = std::min<std::int64_t>(
            diagnostic_ticks,
            8
        );
        const auto reason_counts_host =
            copy_int64_tensor_host_v1(ordinary_birth_output_tensors.at(13));
        const auto planned_count_host =
            copy_int64_tensor_host_v1(ordinary_birth_output_tensors.at(14));
        const auto decision_count_host =
            copy_int64_tensor_host_v1(ordinary_birth_output_tensors.at(15));
        const auto identity_committed_count_host =
            copy_int64_tensor_host_v1(ordinary_birth_output_tensors.at(24));
        const auto scalar_committed_count_host =
            copy_int64_tensor_host_v1(ordinary_birth_output_tensors.at(27));
        const auto drive_committed_count_host =
            copy_int64_tensor_host_v1(ordinary_birth_output_tensors.at(29));
        const auto policy_committed_count_host =
            copy_int64_tensor_host_v1(ordinary_birth_output_tensors.at(38));
        const auto plan_status_host =
            copy_int32_tensor_host_v1(ordinary_birth_output_tensors.at(20));
        const auto identity_status_host =
            copy_int32_tensor_host_v1(ordinary_birth_output_tensors.at(25));
        const auto scalar_status_host =
            copy_int32_tensor_host_v1(ordinary_birth_output_tensors.at(28));
        const auto drive_status_host =
            copy_int32_tensor_host_v1(ordinary_birth_output_tensors.at(34));
        const auto policy_status_host =
            copy_int32_tensor_host_v1(ordinary_birth_output_tensors.at(43));
        const auto combined_status_host =
            copy_int32_tensor_host_v1(ordinary_birth_output_tensors.at(44));
        const auto active_count_state_host =
            copy_int64_tensor_host_v1(active_count_state);
        const auto slot_free_queue_host =
            copy_int64_tensor_host_v1(full_tick_state_tensors.at(22));
        const auto slot_free_queue_count_host =
            copy_int64_tensor_host_v1(full_tick_state_tensors.at(23));

        auto nonzero_count = [](const std::vector<std::int32_t>& values) {
            std::int64_t total = 0;
            for (const std::int32_t value : values) {
                if (value != 0) {
                    ++total;
                }
            }
            return total;
        };
        pybind11::list reason_totals;
        for (
            std::int64_t reason_index = 0;
            reason_index < reason_width;
            ++reason_index
        ) {
            std::int64_t reason_total = 0;
            for (
                std::int64_t tick_index = 0;
                tick_index < birth_tick_count;
                ++tick_index
            ) {
                reason_total += reason_counts_host.at(
                    static_cast<std::size_t>(
                        tick_index * reason_width + reason_index
                    )
                );
            }
            reason_totals.append(reason_total);
        }

        pybind11::list first_ticks;
        for (
            std::int64_t cell_index = 0;
            cell_index < sampled_cell_count;
            ++cell_index
        ) {
            pybind11::list cell_ticks;
            for (
                std::int64_t tick_index = 0;
                tick_index < sampled_tick_count;
                ++tick_index
            ) {
                const std::size_t flat_index = static_cast<std::size_t>(
                    cell_index * diagnostic_ticks + tick_index
                );
                const std::size_t reason_offset = static_cast<std::size_t>(
                    (cell_index * diagnostic_ticks + tick_index) * reason_width
                );
                pybind11::dict row;
                row["tick"] = tick_index;
                row["planned_count"] = planned_count_host.at(flat_index);
                row["decision_count"] = decision_count_host.at(flat_index);
                row["identity_committed_count"] =
                    identity_committed_count_host.at(flat_index);
                row["scalar_committed_count"] =
                    scalar_committed_count_host.at(flat_index);
                row["drive_committed_count"] =
                    drive_committed_count_host.at(flat_index);
                row["policy_committed_count"] =
                    policy_committed_count_host.at(flat_index);
                row["plan_status"] = plan_status_host.at(flat_index);
                row["identity_status"] = identity_status_host.at(flat_index);
                row["scalar_status"] = scalar_status_host.at(flat_index);
                row["drive_status"] = drive_status_host.at(flat_index);
                row["policy_status"] = policy_status_host.at(flat_index);
                row["combined_status"] = combined_status_host.at(flat_index);
                row["reason_counts"] = int64_vector_to_pylist_v1(
                    reason_counts_host,
                    reason_offset,
                    static_cast<std::size_t>(reason_width)
                );
                cell_ticks.append(std::move(row));
            }
            pybind11::dict cell_row;
            cell_row["cell"] = cell_index;
            cell_row["ticks"] = std::move(cell_ticks);
            first_ticks.append(std::move(cell_row));
        }

        pybind11::dict status_nonzero_counts;
        status_nonzero_counts["plan_status"] = nonzero_count(plan_status_host);
        status_nonzero_counts["identity_status"] =
            nonzero_count(identity_status_host);
        status_nonzero_counts["scalar_status"] =
            nonzero_count(scalar_status_host);
        status_nonzero_counts["drive_status"] =
            nonzero_count(drive_status_host);
        status_nonzero_counts["policy_status"] =
            nonzero_count(policy_status_host);
        status_nonzero_counts["combined_status"] =
            nonzero_count(combined_status_host);

        pybind11::dict totals;
        totals["planned_count"] = sum_int64_vector_v1(planned_count_host);
        totals["decision_count"] = sum_int64_vector_v1(decision_count_host);
        totals["identity_committed_count"] =
            sum_int64_vector_v1(identity_committed_count_host);
        totals["scalar_committed_count"] =
            sum_int64_vector_v1(scalar_committed_count_host);
        totals["drive_committed_count"] =
            sum_int64_vector_v1(drive_committed_count_host);
        totals["policy_committed_count"] =
            sum_int64_vector_v1(policy_committed_count_host);

        pybind11::dict birth_diagnostics;
        birth_diagnostics["schema_version"] =
            "eiel_option_d_m6_direct_birth_window_diagnostics_v1";
        birth_diagnostics["authority"] =
            "diagnostic_only_no_execution_authority";
        birth_diagnostics["environment_knob"] =
            "EIEL_OPTION_D_M6_BIRTH_DIAGNOSTICS";
        birth_diagnostics["capture_status"] = "captured";
        birth_diagnostics["ordinary_births_enabled_argument"] =
            ordinary_births_enabled;
        birth_diagnostics["ordinary_birth_config_enabled"] =
            ordinary_birth_config.enabled;
        birth_diagnostics["max_births_per_tick"] =
            ordinary_birth_config.max_births_per_tick;
        birth_diagnostics["gauss_tape_capacity"] =
            ordinary_birth_config.gauss_tape_capacity;
        birth_diagnostics["policy_tape_capacity"] =
            ordinary_birth_config.policy_tape_capacity;
        birth_diagnostics["policy_mutation_tape_width"] =
            ordinary_birth_config.policy_mutation_tape_width;
        birth_diagnostics["torch_state_bytes"] =
            ordinary_birth_config.torch_state_bytes;
        birth_diagnostics["hmem_width"] = ordinary_birth_config.hmem_width;
        birth_diagnostics["cells"] = cells;
        birth_diagnostics["semantic_ticks"] = ticks;
        birth_diagnostics["output_ticks"] = diagnostic_ticks;
        birth_diagnostics["sampled_cell_count"] = sampled_cell_count;
        birth_diagnostics["sampled_tick_count"] = sampled_tick_count;
        birth_diagnostics["reason_width"] = reason_width;
        birth_diagnostics["totals"] = std::move(totals);
        birth_diagnostics["status_nonzero_counts"] =
            std::move(status_nonzero_counts);
        birth_diagnostics["reason_totals"] = std::move(reason_totals);
        birth_diagnostics["first_ticks_by_cell"] = std::move(first_ticks);
        birth_diagnostics["active_count_state_after"] =
            int64_vector_to_pylist_v1(
                active_count_state_host,
                0,
                active_count_state_host.size()
            );
        pybind11::list slot_free_queue_by_cell;
        const std::int64_t queue_sample_cells =
            std::min<std::int64_t>(cells, 4);
        const std::array<std::int64_t, 28> interesting_slots{
            128, 135, 144, 146, 149, 150, 155, 160,
            216, 217, 218, 219, 220, 221, 222, 223,
            224, 225, 226, 228, 231, 232, 234, 235,
            237, 238, 239, 240,
        };
        for (
            std::int64_t cell_index = 0;
            cell_index < queue_sample_cells;
            ++cell_index
        ) {
            const std::int64_t queue_count =
                slot_free_queue_count_host.at(
                    static_cast<std::size_t>(cell_index)
                );
            const std::size_t cell_offset = static_cast<std::size_t>(
                cell_index * capacity
            );
            const std::size_t bounded_count = static_cast<std::size_t>(
                std::max<std::int64_t>(
                    0,
                    std::min<std::int64_t>(queue_count, capacity)
                )
            );
            pybind11::dict queue_row;
            queue_row["cell"] = cell_index;
            queue_row["count"] = queue_count;
            queue_row["prefix"] = int64_vector_to_pylist_v1(
                slot_free_queue_host,
                cell_offset,
                std::min<std::size_t>(bounded_count, 24)
            );
            const std::size_t suffix_count =
                std::min<std::size_t>(bounded_count, 24);
            queue_row["suffix"] = int64_vector_to_pylist_v1(
                slot_free_queue_host,
                cell_offset + bounded_count - suffix_count,
                suffix_count
            );
            pybind11::dict positions;
            for (const std::int64_t value : interesting_slots) {
                std::int64_t position = -1;
                for (
                    std::size_t queue_index = 0;
                    queue_index < bounded_count;
                    ++queue_index
                ) {
                    if (
                        slot_free_queue_host.at(cell_offset + queue_index)
                        == value
                    ) {
                        position = static_cast<std::int64_t>(queue_index);
                        break;
                    }
                }
                if (position < 0) {
                    positions[pybind11::str(std::to_string(value))] =
                        pybind11::none();
                } else {
                    positions[pybind11::str(std::to_string(value))] =
                        position;
                }
            }
            queue_row["interesting_value_positions"] = std::move(positions);
            slot_free_queue_by_cell.append(std::move(queue_row));
        }
        birth_diagnostics["slot_free_queue_by_cell"] =
            std::move(slot_free_queue_by_cell);
        receipt["option_d_m6_birth_diagnostics"] =
            std::move(birth_diagnostics);
    }
    receipt["native_dispatches"] = run_receipt.native_dispatches;
    receipt["native_tick_iterations"] = run_receipt.native_tick_iterations;
    receipt["phase_1_launches"] = run_receipt.phase_1_launches;
    receipt["birth_prepare_launches"] = run_receipt.birth_prepare_launches;
    receipt["birth_commit_launches"] = run_receipt.birth_commit_launches;
    receipt["birth_prepare_commit_fused_launches"] =
        run_receipt.birth_prepare_commit_fused_launches;
    receipt["turnover_shadow_plan_launches"] =
        run_receipt.turnover_shadow_plan_launches;
    receipt["turnover_cull_refill_plan_launches"] =
        run_receipt.turnover_cull_refill_plan_launches;
    receipt["turnover_commit_launches"] =
        run_receipt.turnover_commit_launches;
    receipt["turnover_publish_launches"] =
        run_receipt.turnover_publish_launches;
    receipt["patch_density_refresh_launches"] =
        run_receipt.patch_density_refresh_launches;
    receipt["gru_cleanup_launches"] = run_receipt.gru_cleanup_launches;
    receipt["generation_summary_finalize_launches"] =
        run_receipt.generation_summary_finalize_launches;
    receipt["compact_checkpoint_hash_launches"] =
        compact_checkpoint_hash_launches;
    receipt["compact_checkpoint_projection_launches"] =
        compact_checkpoint_projection_launches;
    receipt["compact_lifecycle_digest_launches"] =
        compact_lifecycle_digest_launches;
    receipt["compact_reduction_launches"] = compact_reduction_launches;
    receipt["compact_metadata_retention_launches"] =
        compact_metadata_retention_launches;
    receipt["compact_checkpoint_hash_batches_per_generation"] =
        static_cast<int64_t>(compact_checkpoint_batches.size());
    receipt["compact_checkpoint_descriptor_count"] =
        compact_checkpoint_descriptor_count;
    receipt["compact_checkpoint_hash_status"] =
        compact_checkpoint_hash_enabled
            ? "device_tensor_hashes_retained_host_metadata_pending"
            : "disabled";
    receipt["total_kernel_launches"] =
        run_receipt.phase_1_launches
        + run_receipt.birth_prepare_launches
        + run_receipt.birth_commit_launches
        + run_receipt.birth_prepare_commit_fused_launches
        + run_receipt.turnover_shadow_plan_launches
        + run_receipt.turnover_cull_refill_plan_launches
        + run_receipt.turnover_commit_launches
        + run_receipt.turnover_publish_launches
        + run_receipt.patch_density_refresh_launches
        + run_receipt.gru_cleanup_launches
        + run_receipt.generation_summary_finalize_launches
        + compact_checkpoint_projection_launches
        + compact_checkpoint_hash_launches
        + compact_reduction_launches
        + compact_lifecycle_digest_launches
        + compact_metadata_retention_launches;
    receipt["host_synchronizations_inside_loop"] =
        run_receipt.host_synchronizations_inside_loop;
    if constexpr (resident_window_refill::kOptionDM0TelemetryEnabledV1) {
        receipt["option_d_m0_telemetry_enabled"] = true;
    }
    receipt["native_host_gaussian_segment_refill_used"] =
        gaussian_segment_refill_calls > 0;
    receipt["native_host_policy_birth_segment_refill_used"] =
        policy_birth_segment_refill_calls > 0;
    receipt["policy_birth_segment_refill_calls_inside_tick_loop"] =
        policy_birth_segment_refill_calls;
    receipt["option_d_m8_resident_policy_suffix_reuse_enabled"] =
        resident_window_refill::kOptionDM8ResidentPolicySuffixReuseEnabledV1;
    receipt["policy_birth_segment_coverage_check_calls_inside_tick_loop"] =
        (
            resident_window_refill::kOptionDM8ResidentPolicySuffixReuseEnabledV1
            || resident_window_refill::
                hapb_frag_a_policy_window_reuse_requested_v1()
        )
            ? policy_birth_segment_refill_calls
            : 0;
    receipt["policy_birth_segment_physical_refill_calls_inside_tick_loop"] =
        policy_birth_segment_physical_refill_calls;
    receipt["native_host_policy_birth_segment_physical_refill_used"] =
        policy_birth_segment_physical_refill_calls > 0;
    receipt["policy_birth_segment_resident_suffix_reuse_calls_inside_tick_loop"] =
        policy_birth_segment_resident_suffix_reuse_calls;
    receipt["policy_birth_segment_refill_receipts"] =
        std::move(policy_birth_segment_refill_receipts);
    receipt["policy_birth_identity_entry_ensure_performed"] =
        policy_birth_identity_entry_ensure_performed;
    receipt["policy_birth_identity_entry_physical_refill_performed"] =
        policy_birth_identity_entry_physical_refill_performed;
    receipt["policy_birth_identity_entry_ensure_receipt"] =
        std::move(policy_birth_identity_entry_ensure_receipt);
    receipt["policy_birth_identity_ensure_calls_total"] =
        policy_birth_identity_ensure_calls_total;
    receipt["policy_birth_identity_materialization_calls_total"] =
        static_cast<int64_t>(0);
    receipt["policy_birth_identity_physical_refill_calls_total"] =
        policy_birth_identity_physical_refill_calls_total;
    receipt["policy_birth_identity_resident_suffix_reuse_calls_total"] =
        policy_birth_identity_resident_suffix_reuse_calls_total;
    receipt["policy_birth_identity_device_to_host_operations_lower_bound"] =
        policy_birth_identity_device_to_host_operations_total;
    receipt["policy_birth_identity_host_to_device_operations_lower_bound"] =
        policy_birth_identity_host_to_device_operations_total;
    receipt["policy_birth_identity_device_to_host_bytes_lower_bound"] =
        policy_birth_identity_device_to_host_bytes_total;
    receipt["policy_birth_identity_host_to_device_bytes_lower_bound"] =
        policy_birth_identity_host_to_device_bytes_total;
    receipt["policy_birth_identity_host_wall_time_ns"] =
        policy_birth_identity_host_wall_time_ns_total;
    receipt["policy_birth_identity_device_to_host_time_ns"] =
        policy_birth_identity_device_to_host_time_ns_total;
    receipt["policy_birth_identity_host_to_device_time_ns"] =
        policy_birth_identity_host_to_device_time_ns_total;
    receipt["policy_birth_segment_device_to_host_operations_lower_bound"] =
        policy_birth_segment_device_to_host_operations;
    receipt["policy_birth_segment_host_to_device_operations_lower_bound"] =
        policy_birth_segment_host_to_device_operations;
    receipt["policy_birth_segment_host_wall_time_ns"] =
        policy_birth_segment_host_wall_time_ns;
    receipt["policy_birth_segment_device_to_host_time_ns"] =
        policy_birth_segment_device_to_host_time_ns;
    receipt["policy_birth_segment_host_to_device_time_ns"] =
        policy_birth_segment_host_to_device_time_ns;
    receipt["policy_birth_segment_torch_random_generation_time_ns"] =
        policy_birth_segment_torch_random_time_ns;
    receipt["policy_birth_segment_torch_compare_pack_time_ns"] =
        policy_birth_segment_compare_pack_time_ns;
    receipt["policy_birth_segment_allocation_stack_copy_time_ns"] =
        policy_birth_segment_allocation_copy_time_ns;
    receipt["gaussian_segment_refill_calls_inside_tick_loop"] =
        gaussian_segment_refill_calls;
    receipt["gaussian_segment_refill_receipts"] =
        std::move(gaussian_segment_refill_receipts);
    receipt["gaussian_segment_boundary_receipts"] =
        std::move(gaussian_segment_boundary_receipts);
    receipt["gaussian_segment_reuse_calls_inside_tick_loop"] =
        gaussian_segment_reuse_calls;
    receipt["gaussian_segment_device_to_host_operations_lower_bound"] =
        gaussian_segment_device_to_host_operations;
    receipt["gaussian_segment_host_to_device_operations_lower_bound"] =
        gaussian_segment_host_to_device_operations;
    receipt["gaussian_segment_host_wall_time_ns"] =
        gaussian_segment_host_wall_time_ns;
    receipt["gaussian_segment_device_to_host_time_ns"] =
        gaussian_segment_device_to_host_time_ns;
    receipt["gaussian_segment_host_to_device_time_ns"] =
        gaussian_segment_host_to_device_time_ns;
    receipt["gaussian_segment_raw_generation_time_ns"] =
        gaussian_segment_raw_generation_time_ns;
    receipt["gaussian_segment_transcendental_conversion_time_ns"] =
        gaussian_segment_transcendental_time_ns;
    receipt["gaussian_segment_allocation_copy_time_ns"] =
        gaussian_segment_allocation_copy_time_ns;
    receipt["gaussian_segment_state_preparation_time_ns"] =
        gaussian_segment_state_preparation_time_ns;
    receipt["gaussian_segment_tape_allocation_time_ns"] =
        gaussian_segment_tape_allocation_time_ns;
    receipt["gaussian_segment_stack_pack_time_ns"] =
        gaussian_segment_stack_pack_time_ns;
    receipt["gaussian_segment_primary_h2d_submission_time_ns"] =
        gaussian_segment_primary_h2d_submission_time_ns;
    receipt["gaussian_segment_secondary_h2d_submission_time_ns"] =
        gaussian_segment_secondary_h2d_submission_time_ns;
    receipt["gaussian_segment_window_base_h2d_submission_time_ns"] =
        gaussian_segment_window_base_h2d_submission_time_ns;
#if EIEL_OPTION_D_M0_TELEMETRY_ENABLED == 1
    receipt["gaussian_pre_refill_synchronization_time_ns"] =
        run_receipt.gaussian_pre_refill_synchronization_time_ns;
    receipt["gaussian_post_copy_synchronization_time_ns"] =
        run_receipt.gaussian_post_copy_synchronization_time_ns;
#endif
    receipt["final_host_synchronizations"] =
        run_receipt.final_host_synchronizations;
    receipt["final_stream_synchronization_time_ns"] =
        final_stream_synchronization_time_ns;
    receipt["last_useful_kernel_completion_observed_host_time_ns"] =
        last_useful_kernel_completion_observed_host_time_ns;
    receipt["requested_generations"] = generation_count;
    receipt["completed_generations"] = completed_generations;
    receipt["completed_generations_by_cell"] = completed_generations_by_cell;
    receipt["python_generation_iterations"] = 0;
    receipt["intermediate_device_to_host_operations"] =
        exact_cpu_policy_device_to_host_operations
        + policy_birth_segment_device_to_host_operations
        + gaussian_segment_device_to_host_operations;
    receipt["final_device_to_host_operations"] =
        (generation_turnover_enabled
             ? (compact_summary_planes_enabled ? 1 : 2)
             : 1)
        + compact_settlement_receipt.d2h_operations;
    receipt["intermediate_host_to_device_operations"] =
        exact_cpu_policy_host_to_device_operations
        + policy_birth_segment_host_to_device_operations
        + gaussian_segment_host_to_device_operations;
    receipt["final_host_to_device_operations"] =
        compact_settlement_receipt.h2d_operations;
    receipt["intermediate_host_synchronizations"] =
        run_receipt.host_synchronizations_inside_loop;
    receipt["python_tick_iterations"] = 0;
    receipt["aten_operations_inside_tick_loop"] =
        exact_cpu_policy_aten_operations_lower_bound;
    receipt["aten_operations_inside_tick_loop_count_semantics"] =
        exact_cpu_policy_service_calls > 0
            ? "lower_bound_from_explicit_native_host_transfers"
            : "exact_zero";
    receipt["exact_cpu_policy_service_calls_inside_tick_loop"] =
        exact_cpu_policy_service_calls;
    receipt["native_host_policy_service_used"] =
        exact_cpu_policy_service_calls > 0;
    if constexpr (!exact_cpu_policy::kOptionDM0TelemetryEnabledV1) {
        pybind11::dict exact_cpu_policy_observation_receipt;
        exact_cpu_policy_observation_receipt["schema_version"] =
            "eiel_option_d_cpu_policy_service_observation_v1";
        exact_cpu_policy_observation_receipt["authority"] =
            "measurement_only_not_option_d_rng_authority";
        exact_cpu_policy_observation_receipt["optimization_performed"] = false;
        exact_cpu_policy_observation_receipt["calls_inside_tick_loop"] =
            exact_cpu_policy_service_calls;
        exact_cpu_policy_observation_receipt["host_wall_time_ns"] =
            exact_cpu_policy_host_wall_time_ns;
        exact_cpu_policy_observation_receipt["device_to_host_time_ns"] =
            exact_cpu_policy_device_to_host_time_ns;
        exact_cpu_policy_observation_receipt["host_to_device_time_ns"] =
            exact_cpu_policy_host_to_device_time_ns;
        exact_cpu_policy_observation_receipt["cpu_mlp_time_ns"] =
            exact_cpu_policy_mlp_time_ns;
        exact_cpu_policy_observation_receipt["cpu_sampling_time_ns"] =
            exact_cpu_policy_sampling_time_ns;
        exact_cpu_policy_observation_receipt[
            "final_stream_synchronization_time_ns"
        ] = exact_cpu_policy_final_sync_time_ns;
        exact_cpu_policy_observation_receipt[
            "device_to_host_bytes_lower_bound"
        ] = exact_cpu_policy_device_to_host_bytes;
        exact_cpu_policy_observation_receipt[
            "host_to_device_bytes_lower_bound"
        ] = exact_cpu_policy_host_to_device_bytes;
        exact_cpu_policy_observation_receipt["device_to_host_operations"] =
            exact_cpu_policy_device_to_host_operations;
        exact_cpu_policy_observation_receipt["host_to_device_operations"] =
            exact_cpu_policy_host_to_device_operations;
        exact_cpu_policy_observation_receipt["host_synchronizations"] =
            exact_cpu_policy_service_calls;
        receipt["exact_cpu_policy_service_observation_receipt"] =
            std::move(exact_cpu_policy_observation_receipt);
    }
    if constexpr (exact_cpu_policy::kOptionDM0TelemetryEnabledV1) {
        pybind11::dict raw_policy_metrics;
        raw_policy_metrics["calls_inside_tick_loop"] =
            exact_cpu_policy_service_calls;
        raw_policy_metrics["host_wall_time_ns"] =
            exact_cpu_policy_host_wall_time_ns;
        raw_policy_metrics["device_to_host_time_ns"] =
            exact_cpu_policy_device_to_host_time_ns;
        raw_policy_metrics["host_to_device_time_ns"] =
            exact_cpu_policy_host_to_device_time_ns;
        raw_policy_metrics["cpu_mlp_time_ns"] =
            exact_cpu_policy_mlp_time_ns;
        raw_policy_metrics["cpu_sampling_time_ns"] =
            exact_cpu_policy_sampling_time_ns;
        raw_policy_metrics["final_stream_synchronization_time_ns"] =
            exact_cpu_policy_final_sync_time_ns;
        raw_policy_metrics["device_to_host_bytes_lower_bound"] =
            exact_cpu_policy_device_to_host_bytes;
        raw_policy_metrics["host_to_device_bytes_lower_bound"] =
            exact_cpu_policy_host_to_device_bytes;
        raw_policy_metrics["device_to_host_operations"] =
            exact_cpu_policy_device_to_host_operations;
        raw_policy_metrics["host_to_device_operations"] =
            exact_cpu_policy_host_to_device_operations;
        raw_policy_metrics["explicit_final_stream_synchronizations"] =
            exact_cpu_policy_service_calls;
        raw_policy_metrics[
            "implicit_blocking_device_to_host_transfer_waits_lower_bound"
        ] = exact_cpu_policy_device_to_host_operations;
        raw_policy_metrics["host_allocation_count_lower_bound"] =
            exact_cpu_policy_host_allocation_count_lower_bound;
        raw_policy_metrics["host_allocation_bytes_lower_bound"] =
            exact_cpu_policy_host_allocation_bytes_lower_bound;
        raw_policy_metrics["host_allocation_measurement_scope"] =
            "lower_bound_of_nonempty_explicit_service_owned_host_tensor_and_"
            "vector_outputs";
        raw_policy_metrics["transfer_operation_measurement_scope"] =
            "nonempty_tensor_copy_operations_only";
        raw_policy_metrics["synchronization_measurement_scope"] =
            "explicit_final_cuda_stream_synchronizations_and_lower_bound_"
            "blocking_device_to_host_tensor_transfers";
        receipt["option_d_m0_exact_cpu_policy_service_raw_metrics"] =
            std::move(raw_policy_metrics);
    }
    receipt["runtime_jit"] = false;
    receipt["hidden_fallback"] = false;
    receipt["compact_summary_planes_enabled"] =
        compact_summary_planes_enabled;
    const bool compact_summary_rows_complete =
        compact_summary_planes_enabled
        && compact_host_settlement_context.status
            == host_settlement::kHostSettlementSuccessV1
        && compact_host_settlement_context.completed_rows
            == generation_count * cells;
    const bool compact_summary_rows_partial =
        compact_summary_planes_enabled
        && compact_host_settlement_context.status
            == host_settlement::kHostSettlementPartialPrefixV1
        && compact_host_settlement_context.completed_rows
            == completed_generations * cells;
    receipt["compact_summary_planes_status"] =
        compact_summary_rows_complete
        ? "native_settlement_callback_complete_unqualified"
        : (compact_summary_rows_partial
              ? "native_settlement_callback_partial_prefix"
        : (
              compact_summary_planes_enabled
                  ? "native_settlement_callback_failed_closed"
                  : "disabled"
          ));
    receipt["compact_summary_rows_status"] =
        compact_summary_rows_complete
        ? "retained_complete_unqualified"
        : (compact_summary_rows_partial
              ? "retained_completed_prefix"
              : (compact_summary_planes_enabled
                    ? "invalid_fail_closed"
                    : "not_allocated"));
    receipt["compact_settlement_status"] = compact_settlement_receipt.status;
    receipt["compact_settlement_d2h_operations"] =
        compact_settlement_receipt.d2h_operations;
    receipt["compact_settlement_host_callback_launches"] =
        compact_settlement_receipt.host_callback_launches;
    receipt["compact_settlement_h2d_operations"] =
        compact_settlement_receipt.h2d_operations;
    receipt["compact_settlement_projection_launches"] =
        compact_settlement_receipt.projection_launches;
    receipt["compact_settlement_synchronizations"] =
        compact_settlement_receipt.synchronizations;
    receipt["compact_settlement_allocations"] =
        compact_settlement_receipt.allocations;
    receipt["compact_host_settlement_status"] =
        compact_host_settlement_context.status;
    receipt["compact_host_settlement_completed_rows"] =
        compact_host_settlement_context.completed_rows;
    receipt["compact_host_settlement_first_failed_row"] =
        compact_host_settlement_context.first_failed_row;
    receipt["compact_host_token_resolver_status"] =
        compact_host_settlement_context.token_resolver_failure.status;
    receipt["compact_host_token_resolver_component"] =
        compact_host_settlement_context.token_resolver_failure.component;
    receipt["compact_host_token_resolver_row"] =
        compact_host_settlement_context.token_resolver_failure.resolver_row;
    receipt["compact_host_token_resolver_generation"] =
        compact_host_settlement_context.token_resolver_failure.generation;
    receipt["compact_host_token_resolver_cell"] =
        compact_host_settlement_context.token_resolver_failure.cell;
    receipt["compact_host_token_resolver_surface"] =
        compact_host_settlement_context.token_resolver_failure.surface;
    receipt["compact_host_token_resolver_token"] =
        compact_host_settlement_context.token_resolver_failure.token_index;
    receipt["compact_host_token_resolver_descriptor"] =
        compact_host_settlement_context.token_resolver_failure.descriptor_index;
    receipt["compact_summary_abi_v2"] = compact_summary_abi_metadata_v2();
    const int64_t reporting_generation_cell_rows = generation_count * cells;
    const int64_t allocated_reporting_rows = compact_summary_planes_enabled
        ? reporting_generation_cell_rows
        : 0;
    const int64_t retained_bytes_per_generation_cell =
        compact_summary_planes_enabled
        ? compact_reporting_retained_bytes_per_generation_cell
        : 0;
    const int64_t reporting_retained_bytes =
        allocated_reporting_rows * retained_bytes_per_generation_cell;
    const int64_t reporting_slot_map_bytes =
        allocated_reporting_rows * capacity * static_cast<int64_t>(sizeof(int32_t));
    const int64_t reporting_effective_fitness_bytes =
        allocated_reporting_rows
        * compact_reporting_row_capacity
        * static_cast<int64_t>(sizeof(double));
    const int64_t reporting_event_cursor_bytes =
        allocated_reporting_rows * 3 * static_cast<int64_t>(sizeof(int64_t));
    pybind11::dict final_transfer_contract;
    final_transfer_contract["boundary"] =
        "stream_ordered_before_one_final_device_synchronization";
    final_transfer_contract["contract_host_transfer_count"] =
        compact_summary_planes_enabled
        ? (patch_p2_full_mode ? 13 : 12)
        : 0;
    final_transfer_contract["observed_host_transfer_count"] =
        compact_settlement_receipt.d2h_operations;
    final_transfer_contract["source_tensor"] =
        "bounded_compact_settlement_sources";
    final_transfer_contract["dtype"] = "uint8";
    final_transfer_contract["shape"] = pybind11::make_tuple(
        generation_count,
        cells,
        retained_bytes_per_generation_cell
    );
    final_transfer_contract["bytes"] = reporting_retained_bytes;
    final_transfer_contract["scratch_included"] = false;
    final_transfer_contract["event_cursors_included"] = false;
    final_transfer_contract["status"] = compact_summary_planes_enabled
        ? "native_stream_ordered_callback_wired"
        : "disabled";
    pybind11::dict reporting_storage;
    reporting_storage["schema_version"] =
        "eiel_cuda_architecture_b_compact_summary_reporting_integration_v1";
    reporting_storage["retained_layout_version"] =
        reporting_authority::kRetainedLayoutVersionV2;
    reporting_storage["enabled"] = compact_summary_planes_enabled;
    reporting_storage["generation_major"] = true;
    reporting_storage["generation_cell_rows"] = reporting_generation_cell_rows;
    reporting_storage["reporting_row_capacity"] =
        compact_reporting_row_capacity;
    reporting_storage["slot_capacity"] = capacity;
    reporting_storage["retained_bytes_per_generation_cell"] =
        retained_bytes_per_generation_cell;
    reporting_storage["retained_bytes"] = reporting_retained_bytes;
    reporting_storage["slot_map_bytes"] = reporting_slot_map_bytes;
    reporting_storage["effective_fitness_bytes"] =
        reporting_effective_fitness_bytes;
    reporting_storage["event_cursor_bytes"] = reporting_event_cursor_bytes;
    reporting_storage["total_preallocated_bytes"] =
        reporting_retained_bytes
        + reporting_slot_map_bytes
        + reporting_effective_fitness_bytes
        + reporting_event_cursor_bytes;
    reporting_storage["preallocated_before_native_dispatch"] =
        compact_summary_planes_enabled;
    reporting_storage["native_producer_supported"] = false;
    reporting_storage["final_transfer"] = std::move(final_transfer_contract);
    receipt["compact_summary_reporting_storage"] =
        std::move(reporting_storage);
    return receipt;
}

pybind11::dict execute_resident_windows_direct_v1(
    torch::Tensor active_agent_ids,
    torch::Tensor active_agent_slots,
    torch::Tensor active_count_state,
    torch::Tensor slot_to_agent_id,
    torch::Tensor agent_id_to_slot,
    torch::Tensor alive_mask,
    torch::Tensor slot_tokens,
    torch::Tensor action_cache_actions,
    torch::Tensor action_cache_token,
    torch::Tensor action_cache_cursor,
    torch::Tensor action_cache_next,
    torch::Tensor pos_y,
    torch::Tensor pos_x,
    torch::Tensor agent_grid,
    torch::Tensor food_grid,
    torch::Tensor energy,
    torch::Tensor health,
    torch::Tensor injury,
    torch::Tensor policy_energy_observation,
    torch::Tensor rest_streak,
    torch::Tensor pending_free_agent_ids,
    torch::Tensor pending_free_slots,
    torch::Tensor pending_free_count,
    torch::Tensor energy_collected,
    torch::Tensor energy_collected_exact,
    torch::Tensor last_action,
    torch::Tensor last_last_action,
    torch::Tensor action_switches,
    torch::Tensor action_oscillations,
    torch::Tensor episode0_switches,
    torch::Tensor episode0_oscillations,
    torch::Tensor survival_steps,
    torch::Tensor mlp_pack,
    torch::Tensor drives,
    torch::Tensor attack_bias,
    torch::Tensor action_temperature,
    torch::Tensor drive_logit_gain,
    torch::Tensor food_gained_scratch,
    torch::Tensor ordered_agent_ids,
    torch::Tensor ordered_slots,
    torch::Tensor observations,
    torch::Tensor logits,
    torch::Tensor row_actions,
    torch::Tensor actions_by_slot,
    torch::Tensor pre_energy,
    torch::Tensor pre_health,
    torch::Tensor pre_injury,
    torch::Tensor pre_survival_steps,
    torch::Tensor active_count,
    torch::Tensor policy_refill_rows,
    torch::Tensor policy_cache_hit_rows,
    torch::Tensor runtime_bookkeeping_by_tick,
    torch::Tensor death_agent_ids,
    torch::Tensor death_slots,
    torch::Tensor death_reason_codes,
    torch::Tensor death_count,
    torch::Tensor survivor_agent_ids,
    torch::Tensor survivor_count,
    torch::Tensor lifecycle_status,
    torch::Tensor status,
    std::vector<torch::Tensor> full_tick_state_tensors,
    std::vector<torch::Tensor> full_tick_output_tensors,
    std::vector<torch::Tensor> compact_summary_reporting_tensors,
    std::vector<torch::Tensor> compact_summary_checkpoint_source_tensors,
    torch::Tensor compact_summary_checkpoint_descriptor_specs,
    std::vector<torch::Tensor> compact_summary_checkpoint_output_tensors,
    std::vector<torch::Tensor> compact_summary_settlement_tensors,
    std::vector<torch::Tensor> event_state_tensors,
    std::vector<torch::Tensor> ordinary_birth_state_tensors,
    std::vector<torch::Tensor> ordinary_birth_output_tensors,
    std::vector<torch::Tensor> ordinary_birth_scratch_tensors,
    torch::Tensor turnover_selection_seed_by_cell,
    std::vector<torch::Tensor> turnover_state_tensors,
    std::vector<torch::Tensor> turnover_shadow_tensors,
    std::vector<torch::Tensor> turnover_output_tensors,
    std::vector<torch::Tensor> turnover_pointer_table_tensors,
    std::vector<torch::Tensor> patch_p2_state_tensors,
    std::vector<torch::Tensor> patch_p2_scratch_tensors,
    std::vector<torch::Tensor> patch_p2_stats_tensors,
    pybind11::dict patch_p2_config,
    const int64_t ticks,
    const int64_t view_radius,
    const int64_t hidden_size,
    const int64_t action_count,
    const int64_t cache_width,
    const bool deterministic_actions,
    const int64_t max_food,
    const double food_spawn_per_step,
    const double food_spawn_mult,
    const double max_energy,
    const double start_energy,
    const double start_health,
    torch::Tensor move_cost_by_cell,
    const double gather_cost,
    const double rest_cost,
    const double attack_cost,
    const double action_cost_mult,
    const double rest_cost_mult,
    const double food_energy,
    const int64_t episode0_k,
    const double w_hunger_gather,
    const double w_pain_rest,
    const double w_fatigue_rest,
    const double w_pain_attack,
    const double w_fear_attack,
    const bool enable_rest_charge,
    const int64_t rest_streak_free,
    const double rest_charge_energy,
    const double rest_charge_growth,
    const double rest_deficit_to_health,
    const int64_t developmental_maturity_age,
    const int64_t life_history_mode,
    const int64_t life_history_maturity_start_ticks,
    const int64_t life_history_maturity_rise_ticks,
    const int64_t life_history_fertility_decline_start_ticks,
    const int64_t life_history_fertility_decline_ticks,
    const double life_history_elder_fertility_floor,
    const int64_t life_history_senescence_start_ticks,
    const int64_t life_history_senescence_ramp_ticks,
    const double life_history_senescence_energy_cost,
    const double life_history_senescence_health_cost,
    const int64_t life_history_max_age_ticks,
    const double drive_hunger_gain,
    const double drive_hunger_decay,
    const double drive_pain_gain,
    const double drive_pain_decay,
    const double drive_fatigue_gain,
    const double drive_fatigue_rest_relief,
    const double drive_fatigue_decay,
    const double drive_fear_gain,
    const double drive_fear_decay,
    const int64_t generation,
    const int64_t start_tick,
    const bool ordinary_births_enabled,
    const int64_t max_births_per_tick,
    const double reproduce_parent_energy_share,
    const double reproduce_min_energy_buffer,
    const int64_t reproduce_cooldown_steps,
    const int64_t genome_sensor_radius_max,
    const double injury_repro_k_injury,
    const double injury_repro_k_scar,
    const double injury_birth_fail_scar,
    const double injury_birth_fail_injury,
    const bool env_policy_alias,
    const bool reproduce_requires_mate,
    const bool generation_turnover_enabled,
    const int64_t lifecycle_mode,
    const int64_t turnover_population_size,
    const double turnover_cull_fraction,
    const double turnover_parent_fraction,
    const int64_t turnover_selection_mode,
    const double coherence_thrashing_weight,
    const double coherence_drive_volatility_weight,
    const double coherence_selfharm_weight,
    const double coherence_pred_err_weight,
    const int64_t generation_count,
    const double reporting_parent_fraction,
    const double reporting_episode0_fitness_weight,
    const double reporting_episode0_pred_err_fitness_weight,
    const double reporting_episode0_thrashing_fitness_weight,
    const double reporting_episode0_drive_jitter_fitness_weight,
    const double reporting_episode0_selfharm_fitness_weight,
    const double reporting_coherence_thrashing_weight,
    const double reporting_coherence_drive_jitter_weight,
    const double reporting_coherence_selfharm_weight,
    const double reporting_coherence_pred_err_weight,
    const double reporting_coherence_reset_shock_weight,
    const double reporting_coherence_episode0_pred_err_weight,
    const double reporting_late_pred_cost_weight,
    const bool reporting_late_pred_cost_enable,
    const bool reporting_disable_prediction_error_selection_pressure,
    torch::Tensor first_failure_records,
    const int64_t logical_generation_count,
    const bool policy_weight_mutation_enable,
    const std::string& policy_weight_mutation_mode,
    const double policy_weight_mutation_rate,
    const double policy_weight_mutation_std
) {
    TORCH_CHECK(
        full_tick_output_tensors.size() == 20
            || full_tick_output_tensors.size() == 27,
        "direct resident dispatch requires scalar generation summaries"
    );
    TORCH_CHECK(
        !patch_p2_stats_tensors.empty(),
        "direct resident dispatch requires Patch P2 statistics"
    );
    TORCH_CHECK(
        turnover_output_tensors.size()
            > resident_window_controller::kTurnoverStatusOutputV1,
        "direct resident dispatch requires generation turnover outputs"
    );
    const auto run_control = active_direct_run_control_v2;
    const int64_t policy_row_width = mlp_pack.size(2);
    TORCH_CHECK(policy_row_width > 0, "direct resident policy row width must be positive");
    resident_window_refill::PolicyWindowIdentityStateV1
        local_policy_window_identity_state;
    auto* const policy_window_identity_state =
        active_policy_window_identity_state_v1 != nullptr
        ? active_policy_window_identity_state_v1
        : &local_policy_window_identity_state;
    if (run_control) {
        const auto manifest = run_control->manifest();
        TORCH_CHECK(
            !manifest.run_started,
            "CUDA V2 run control is single-use and already started"
        );
        run_control->start_run(generation, logical_generation_count);
    }
    const auto submit_window = [&](
        const int64_t generation_base,
        const int64_t requested_window_generations
    ) {
        auto window_full_tick_output_tensors = full_tick_output_tensors;
        window_full_tick_output_tensors[
            resident_window_controller::kGenerationSummaryOutputV1
        ] = full_tick_output_tensors[
            resident_window_controller::kGenerationSummaryOutputV1
        ].narrow(0, 0, requested_window_generations);
        auto window_patch_p2_stats_tensors = patch_p2_stats_tensors;
        window_patch_p2_stats_tensors[0] = patch_p2_stats_tensors[0].narrow(
            1,
            0,
            requested_window_generations * ticks
        ).contiguous();
        if (window_patch_p2_stats_tensors.size() == 2) {
            window_patch_p2_stats_tensors[1] = patch_p2_stats_tensors[1].narrow(
                0,
                0,
                requested_window_generations
            );
        }
        const auto window_runtime_bookkeeping_by_tick =
            runtime_bookkeeping_by_tick.narrow(
                1,
                0,
                requested_window_generations * ticks
            ).contiguous();
        const ScopedPolicyWindowIdentityStateV1 identity_scope(
            policy_window_identity_state
        );
        auto receipt = flattened_flat_dynamic_policy_food_birth_controller_v1(
            active_agent_ids,
            active_agent_slots,
            active_count_state,
            slot_to_agent_id,
            agent_id_to_slot,
            alive_mask,
            slot_tokens,
            action_cache_actions,
            action_cache_token,
            action_cache_cursor,
            action_cache_next,
            pos_y,
            pos_x,
            agent_grid,
            food_grid,
            energy,
            health,
            injury,
            policy_energy_observation,
            rest_streak,
            pending_free_agent_ids,
            pending_free_slots,
            pending_free_count,
            energy_collected,
            energy_collected_exact,
            last_action,
            last_last_action,
            action_switches,
            action_oscillations,
            episode0_switches,
            episode0_oscillations,
            survival_steps,
            mlp_pack,
            drives,
            attack_bias,
            action_temperature,
            drive_logit_gain,
            food_gained_scratch,
            ordered_agent_ids,
            ordered_slots,
            observations,
            logits,
            row_actions,
            actions_by_slot,
            pre_energy,
            pre_health,
            pre_injury,
            pre_survival_steps,
            active_count,
            policy_refill_rows,
            policy_cache_hit_rows,
            window_runtime_bookkeeping_by_tick,
            death_agent_ids,
            death_slots,
            death_reason_codes,
            death_count,
            survivor_agent_ids,
            survivor_count,
            lifecycle_status,
            status,
            full_tick_state_tensors,
            window_full_tick_output_tensors,
            compact_summary_reporting_tensors,
            compact_summary_checkpoint_source_tensors,
            compact_summary_checkpoint_descriptor_specs,
            compact_summary_checkpoint_output_tensors,
            compact_summary_settlement_tensors,
            event_state_tensors,
            ordinary_birth_state_tensors,
            ordinary_birth_output_tensors,
            ordinary_birth_scratch_tensors,
            turnover_selection_seed_by_cell,
            turnover_state_tensors,
            turnover_shadow_tensors,
            turnover_output_tensors,
            turnover_pointer_table_tensors,
            patch_p2_state_tensors,
            patch_p2_scratch_tensors,
            window_patch_p2_stats_tensors,
            patch_p2_config,
            ticks,
            view_radius,
            hidden_size,
            action_count,
            cache_width,
            deterministic_actions,
            max_food,
            food_spawn_per_step,
            food_spawn_mult,
            max_energy,
            start_energy,
            start_health,
            move_cost_by_cell,
            gather_cost,
            rest_cost,
            attack_cost,
            action_cost_mult,
            rest_cost_mult,
            food_energy,
            episode0_k,
            w_hunger_gather,
            w_pain_rest,
            w_fatigue_rest,
            w_pain_attack,
            w_fear_attack,
            enable_rest_charge,
            rest_streak_free,
            rest_charge_energy,
            rest_charge_growth,
            rest_deficit_to_health,
            developmental_maturity_age,
            life_history_mode,
            life_history_maturity_start_ticks,
            life_history_maturity_rise_ticks,
            life_history_fertility_decline_start_ticks,
            life_history_fertility_decline_ticks,
            life_history_elder_fertility_floor,
            life_history_senescence_start_ticks,
            life_history_senescence_ramp_ticks,
            life_history_senescence_energy_cost,
            life_history_senescence_health_cost,
            life_history_max_age_ticks,
            drive_hunger_gain,
            drive_hunger_decay,
            drive_pain_gain,
            drive_pain_decay,
            drive_fatigue_gain,
            drive_fatigue_rest_relief,
            drive_fatigue_decay,
            drive_fear_gain,
            drive_fear_decay,
            generation_base,
            start_tick,
            ordinary_births_enabled,
            max_births_per_tick,
            reproduce_parent_energy_share,
            reproduce_min_energy_buffer,
            reproduce_cooldown_steps,
            genome_sensor_radius_max,
            injury_repro_k_injury,
            injury_repro_k_scar,
            injury_birth_fail_scar,
            injury_birth_fail_injury,
            env_policy_alias,
            reproduce_requires_mate,
            generation_turnover_enabled,
            lifecycle_mode,
            turnover_population_size,
            turnover_cull_fraction,
            turnover_parent_fraction,
            turnover_selection_mode,
            coherence_thrashing_weight,
            coherence_drive_volatility_weight,
            coherence_selfharm_weight,
            coherence_pred_err_weight,
            requested_window_generations,
            reporting_parent_fraction,
            reporting_episode0_fitness_weight,
            reporting_episode0_pred_err_fitness_weight,
            reporting_episode0_thrashing_fitness_weight,
            reporting_episode0_drive_jitter_fitness_weight,
            reporting_episode0_selfharm_fitness_weight,
            reporting_coherence_thrashing_weight,
            reporting_coherence_drive_jitter_weight,
            reporting_coherence_selfharm_weight,
            reporting_coherence_pred_err_weight,
            reporting_coherence_reset_shock_weight,
            reporting_coherence_episode0_pred_err_weight,
            reporting_late_pred_cost_weight,
            reporting_late_pred_cost_enable,
            reporting_disable_prediction_error_selection_pressure,
            first_failure_records
        );
        // Narrowing the tick axis of a multicell [cells,160,*] buffer for an
        // odd one-generation tail is not contiguous.  The physical ABI needs
        // contiguous inputs, so the tensors above can be private copies.  Put
        // their native results back into the resident window buffers before
        // the owner snapshots the completed prefix.
        runtime_bookkeeping_by_tick.narrow(
            1,
            0,
            requested_window_generations * ticks
        ).copy_(window_runtime_bookkeeping_by_tick);
        patch_p2_stats_tensors[0].narrow(
            1,
            0,
            requested_window_generations * ticks
        ).copy_(window_patch_p2_stats_tensors[0]);
        if (run_control) {
            const auto manifest = run_control->manifest();
            if (manifest.last_boundary) {
                receipt["native_run_control_v2_next_generation"] =
                    manifest.last_boundary->next_generation;
            }
            receipt["native_run_control_v2_stop_after_completed_prefix"] =
                manifest.awaiting_cancellation_checkpoint || manifest.sealed;
        }
        return receipt;
    };
    std::string event_history_sink_path;
    if (patch_p2_config.contains(pybind11::str("event_history_sink_path"))) {
        event_history_sink_path = pybind11::cast<std::string>(
            patch_p2_config[pybind11::str("event_history_sink_path")]
        );
        TORCH_CHECK(
            !event_history_sink_path.empty(),
            "event_history_sink_path must not be empty"
        );
    }
    try {
        auto result = resident_window_controller::execute_resident_windows_core_v1(
            submit_window,
            generation,
            generation_count,
            logical_generation_count,
            full_tick_output_tensors[
                resident_window_controller::kGenerationSummaryOutputV1
            ],
            runtime_bookkeeping_by_tick,
            patch_p2_stats_tensors.front(),
            event_state_tensors,
            ordinary_birth_state_tensors,
            status,
            lifecycle_status,
            turnover_output_tensors[
                resident_window_controller::kTurnoverCountsOutputV1
            ],
            turnover_output_tensors[
                resident_window_controller::kTurnoverStatusOutputV1
            ],
            policy_row_width,
            policy_weight_mutation_enable,
            policy_weight_mutation_mode,
            policy_weight_mutation_rate,
            policy_weight_mutation_std,
            false,
            true,
            true,
            run_control != nullptr,
            policy_window_identity_state,
            event_history_sink_path
        );
        if constexpr (resident_window_refill::kOptionDM0TelemetryEnabledV1) {
            result["option_d_m0_telemetry_enabled"] = true;
        }
        return result;
    } catch (const std::exception& error) {
        if (run_control) {
            const auto manifest = run_control->manifest();
            run_control->publish_failure(resident_run_control::FailureMetadataV2{
                "CUDA_DIRECT_RESIDENT_EXECUTION_FAILED",
                error.what(),
                manifest.active_generation,
                manifest.last_boundary
                    ? manifest.last_boundary->trajectory_commit_frontier
                    : generation - 1,
            });
        }
        throw;
    }
}

pybind11::dict eiel_authoritative_state_build_provenance_v1();

pybind11::dict
flattened_flat_dynamic_policy_food_birth_controller_resource_receipt_v1() {
    const auto phase_1 = eiel::cuda_architecture_b::flattened_controller_v1::
        query_flattened_controller_tick_core_resources_v1();
    const auto birth_prepare = eiel::cuda_architecture_b::flattened_controller_v1::
        query_flattened_controller_birth_prepare_resources_v1();
    const auto birth_commit = eiel::cuda_architecture_b::flattened_controller_v1::
        query_flattened_controller_birth_commit_resources_v1();
    const auto birth_prepare_commit =
        eiel::cuda_architecture_b::flattened_controller_v1::
            query_flattened_controller_birth_prepare_commit_resources_v1();
    const auto summary_finalize =
        eiel::cuda_architecture_b::flattened_controller_v1::
            query_generation_summary_finalize_resources_v1();
    const auto turnover_shadow_plan =
        turnover::query_flattened_turnover_shadow_plan_resources_v1();
    const auto turnover_cull_refill_plan =
        turnover::query_flattened_turnover_cull_refill_plan_resources_v1();
    const auto turnover_commit =
        turnover::query_flattened_turnover_commit_resources_v1();
    const auto turnover_publish =
        turnover::query_flattened_turnover_publish_resources_v1();
    const auto compact_checkpoint_hash =
        checkpoint_hash::query_checkpoint_tensor_hash_snapshot_v1_resources();
    const auto resource_dict = [](const auto& receipt, const char* symbol) {
        pybind11::dict out;
        out["kernel_symbol"] = symbol;
        out["binary_version"] = receipt.binary_version;
        out["ptx_version"] = receipt.ptx_version;
        out["num_registers"] = receipt.num_registers;
        out["constant_size_bytes"] = receipt.constant_size_bytes;
        out["local_size_bytes"] = receipt.local_size_bytes;
        out["shared_size_bytes"] = receipt.shared_size_bytes;
        out["max_threads_per_block"] = receipt.max_threads_per_block;
        out["max_blocks_per_multiprocessor"] =
            receipt.max_blocks_per_multiprocessor;
        return out;
    };
    pybind11::dict out;
    out["threads_per_cell"] = eiel::cuda_architecture_b::flattened_controller_v1::
        kThreadsPerCell;
    out["scratch_bytes_per_cell"] =
        eiel::cuda_architecture_b::flattened_controller_v1::
            kBirthScratchBytesPerCell;
    out["parallel_patch_policy_rows_challenger"] =
        EIEL_CUDA_STAGE3_PARALLEL_PATCH_POLICY_ROWS != 0;
    out["lifecycle_linear_fast_path_enabled"] =
        EIEL_TW_LIFECYCLE_LINEAR_FAST_PATH_ENABLED != 0;
    out["lifecycle_linear_fast_path_test_surface_enabled"] =
        EIEL_TW_LIFECYCLE_LINEAR_FAST_PATH_TEST_SURFACE_ENABLED != 0;
    out["option_d_m0_telemetry_enabled"] =
        resident_window_refill::kOptionDM0TelemetryEnabledV1;
    out["option_d_m2_policy_shadow_enabled"] =
        resident_window_refill::kOptionDM2PolicyShadowEnabledV1;
    out["option_d_m4_device_policy_refill_enabled"] =
        resident_window_refill::kOptionDM4DevicePolicyRefillEnabledV1;
    out["option_d_m7_batched_cpu_policy_refill_enabled"] =
        resident_window_refill::kOptionDM7BatchedCpuPolicyRefillEnabledV1;
    out["option_d_m8_resident_policy_suffix_reuse_enabled"] =
        resident_window_refill::kOptionDM8ResidentPolicySuffixReuseEnabledV1;
    out["option_d_m6_device_policy_context_enabled"] =
        !static_cast<bool>(EIEL_CUDA_ARCHITECTURE_B_EXACT_CPU_POLICY_CONTEXT_ALLOWED);
    out["exact_cpu_policy_context_allowed"] =
        static_cast<bool>(EIEL_CUDA_ARCHITECTURE_B_EXACT_CPU_POLICY_CONTEXT_ALLOWED);
    out["option_d_m9_birth_prepare_commit_fused_enabled"] =
        static_cast<bool>(EIEL_OPTION_D_M9_BIRTH_PREPARE_COMMIT_FUSED_ENABLED);
    out["authoritative_state_build_provenance_v1"] =
        eiel_authoritative_state_build_provenance_v1();
    out["phase_1"] = resource_dict(
        phase_1,
        "flattened_controller_tick_core_and_birth_plan_kernel_v1"
    );
    out["birth_prepare"] = resource_dict(
        birth_prepare,
        "flattened_controller_birth_prepare_kernel_v1"
    );
    out["birth_commit"] = resource_dict(
        birth_commit,
        "flattened_controller_birth_transaction_and_settlement_kernel_v1"
    );
    out["birth_prepare_commit_fused"] = resource_dict(
        birth_prepare_commit,
        "flattened_controller_birth_prepare_and_transaction_kernel_v1"
    );
    out["generation_summary_finalize"] = resource_dict(
        summary_finalize,
        "finalize_generation_summary_scalar_v1"
    );
    out["turnover_shadow_plan"] = resource_dict(
        turnover_shadow_plan,
        "flattened_turnover_shadow_plan_kernel_v1"
    );
    out["turnover_cull_refill_plan"] = resource_dict(
        turnover_cull_refill_plan,
        "flattened_turnover_cull_refill_plan_kernel_v1"
    );
    out["turnover_commit"] = resource_dict(
        turnover_commit,
        "flattened_turnover_commit_kernel_v1"
    );
    out["turnover_publish"] = resource_dict(
        turnover_publish,
        "flattened_turnover_publish_kernel_v1"
    );
    pybind11::dict compact_checkpoint_hash_resource;
    compact_checkpoint_hash_resource["kernel_symbol"] =
        "checkpoint_tensor_hash_snapshot_kernel_v1";
    compact_checkpoint_hash_resource["binary_version"] =
        compact_checkpoint_hash.binary_version;
    compact_checkpoint_hash_resource["ptx_version"] =
        compact_checkpoint_hash.ptx_version;
    compact_checkpoint_hash_resource["num_registers"] =
        compact_checkpoint_hash.registers_per_thread;
    compact_checkpoint_hash_resource["constant_size_bytes"] =
        compact_checkpoint_hash.constant_bytes;
    compact_checkpoint_hash_resource["local_size_bytes"] =
        compact_checkpoint_hash.local_bytes_per_thread;
    compact_checkpoint_hash_resource["shared_size_bytes"] =
        compact_checkpoint_hash.shared_bytes_per_block;
    compact_checkpoint_hash_resource["max_threads_per_block"] =
        compact_checkpoint_hash.max_threads_per_block;
    compact_checkpoint_hash_resource["max_blocks_per_multiprocessor"] =
        compact_checkpoint_hash.occupancy_blocks_per_multiprocessor;
    out["compact_checkpoint_hash"] =
        std::move(compact_checkpoint_hash_resource);
    return out;
}

pybind11::dict
flattened_flat_dynamic_policy_food_birth_controller_generation_count_capability_v1() {
    pybind11::dict limits;
    limits["max_generation_count"] = 128;
    limits["max_physical_batch_cells"] = 72;
    limits["max_ticks_per_generation"] = 160;
    limits["max_total_native_ticks"] = 10240;
    limits["max_turnover_count"] = 128;
    limits["preallocated_d2d_snapshot_slots"] = 0;
    limits["max_policy_episode_capacity"] = 20604;
    limits["max_gaussian_values_per_plane"] = 10549248;
    limits["max_additional_event_rows"] = 5268992;
    limits["max_identity_growth"] = 18176;

    pybind11::dict out;
    out["schema_version"] =
        "eiel_cuda_architecture_b_flattened_generation_count_capability_v1";
    out["generation_count_supported"] = true;
    // Keep the legacy scalar for old Flat callers; new consumers must use the
    // qualified envelope below instead of inferring broad world support.
    out["supported_world"] = "flat";
    pybind11::list supported_worlds;
    supported_worlds.append("flat");
    supported_worlds.append("patch");
    out["supported_worlds"] = std::move(supported_worlds);
    pybind11::list supported_generation_counts;
    for (const int64_t count : {1, 2, 8, 32, 64, 128}) {
        supported_generation_counts.append(count);
    }
    out["supported_generation_counts"] =
        std::move(supported_generation_counts);
    out["resident_generation_summaries_supported"] = true;
    out["eiel_compact_generation_summary_v1_supported"] = true;
    out["compact_summary_plane_plumbing_supported"] = true;
    out["compact_summary_native_producer_status"] =
        "qualified_patch_p2_g1_g2_g8_g32_g64_g128";
    out["compact_summary_reporting_retained_layout_version"] =
        reporting_authority::kRetainedLayoutVersionV2;
    out["compact_summary_reporting_dynamic_row_capacity"] =
        reporting_authority::kDynamicRowCapacityV1;
    out["compact_summary_reporting_maximum_dynamic_row_capacity"] =
        reporting_authority::kMaximumDynamicRowCapacityV1;
    out["compact_summary_reporting_retained_bytes_per_generation_cell"] =
        static_cast<int64_t>(
            reporting_authority::retained_bytes_per_generation_cell_v1(
                reporting_authority::kDynamicRowCapacityV1
            )
        );
    out["compact_summary_reporting_native_producer_supported"] = true;
    out["compact_summary_abi_v2"] = compact_summary_abi_metadata_v2();
    out["generation_summary_scalar_layout_version"] =
        controller::kGenerationSummaryScalarLayoutVersionV1;
    out["generation_summary_scalar_layout_width"] =
        static_cast<int64_t>(controller::kGenerationSummaryScalarWidthV1);
    pybind11::list generation_summary_scalar_columns;
    for (const char* name : controller::kGenerationSummaryScalarColumnNamesV1) {
        generation_summary_scalar_columns.append(name);
    }
    out["generation_summary_scalar_columns"] =
        generation_summary_scalar_columns;
    out["canonical_final_checkpoint_projection_supported"] = true;
    out["same_process_repeat_supported"] = true;
    out["rolling_tick_output_storage_supported"] = true;
    out["rolling_tick_output_storage_rows"] = 1;
    out["compact_summary_max_physical_batch_cells"] = 1;
    pybind11::dict patch_p2;
    patch_p2["qualified"] = true;
    patch_p2["profile"] = "P2";
    patch_p2["profile_sha256"] =
        "08897ccd52faf20af319d2e8bdd426518c7a6546c5066bbb1e23df56af6ca050";
    patch_p2["world"] = "patch";
    patch_p2["policy"] = "stochastic_fixed_mlp";
    patch_p2["batch_cells"] = 1;
    patch_p2["population"] = 64;
    patch_p2["grid_size"] = 16;
    pybind11::list patch_supported_geometry_envelopes;
    pybind11::dict patch_w16p64_geometry;
    patch_w16p64_geometry["population"] = 64;
    patch_w16p64_geometry["grid_size"] = 16;
    patch_supported_geometry_envelopes.append(
        std::move(patch_w16p64_geometry));
    pybind11::dict patch_w16p128_geometry;
    patch_w16p128_geometry["population"] = 128;
    patch_w16p128_geometry["grid_size"] = 16;
    patch_supported_geometry_envelopes.append(
        std::move(patch_w16p128_geometry));
    pybind11::dict patch_w32p256_geometry;
    patch_w32p256_geometry["population"] = 256;
    patch_w32p256_geometry["grid_size"] = 32;
    patch_supported_geometry_envelopes.append(
        std::move(patch_w32p256_geometry));
    pybind11::dict patch_w32p512_geometry;
    patch_w32p512_geometry["population"] = 512;
    patch_w32p512_geometry["grid_size"] = 32;
    patch_supported_geometry_envelopes.append(
        std::move(patch_w32p512_geometry));
    patch_p2["supported_geometry_envelopes"] =
        std::move(patch_supported_geometry_envelopes);
    pybind11::list patch_p2_admitted_ticks;
    patch_p2_admitted_ticks.append(80);
    patch_p2_admitted_ticks.append(160);
    patch_p2["ticks_per_generation"] = 80;
    patch_p2["admitted_ticks_per_generation"] = std::move(patch_p2_admitted_ticks);
    patch_p2["max_births_per_tick"] = 1;
    patch_p2["mechanically_supported_max_births_per_tick"] = 64;
    patch_p2["supported_reproduction_mode"] = "asexual";
    patch_p2["generation_turnover"] = true;
    pybind11::list patch_supported_lifecycle_modes;
    patch_supported_lifecycle_modes.append(0);
    patch_supported_lifecycle_modes.append(1);
    patch_p2["supported_lifecycle_modes"] =
        std::move(patch_supported_lifecycle_modes);
    patch_p2["compact_summary_enabled"] = true;
    pybind11::list patch_supported_generation_counts;
    for (const int64_t count : {1, 2, 8, 32, 64, 128}) {
        patch_supported_generation_counts.append(count);
    }
    patch_p2["supported_generation_counts"] =
        std::move(patch_supported_generation_counts);
    patch_p2["checkpoint_descriptor_count"] =
        checkpoint_runtime::kPatchP2DescriptorCountV1;
    patch_p2["sticky_capacity_failure_qualified"] = true;
    patch_p2["qualification_gate"] =
        "CUDA-STAGE3-PATCH-P2-G64-G128-EXACTNESS";
    out["patch_p2_qualification"] = std::move(patch_p2);
    pybind11::dict patch_p2_b2_candidate;
    patch_p2_b2_candidate["available"] = true;
    patch_p2_b2_candidate["qualified"] = true;
    patch_p2_b2_candidate["batch_cells"] = 2;
    patch_p2_b2_candidate["generation_count"] = 2;
    pybind11::list patch_p2_b2_candidate_supported_generation_counts;
    for (const int64_t count : {2, 8, 32, 64, 128}) {
        patch_p2_b2_candidate_supported_generation_counts.append(count);
    }
    patch_p2_b2_candidate["candidate_supported_generation_counts"] =
        std::move(patch_p2_b2_candidate_supported_generation_counts);
    pybind11::list patch_p2_b2_qualified_generation_counts;
    patch_p2_b2_qualified_generation_counts.append(2);
    patch_p2_b2_qualified_generation_counts.append(32);
    patch_p2_b2_candidate["qualified_generation_counts"] =
        std::move(patch_p2_b2_qualified_generation_counts);
    patch_p2_b2_candidate["g32_qualification_scope"] =
        "two_independent_cells_t80_rolling_single_row_exact";
    patch_p2_b2_candidate["g32_evidence_sha256"] =
        "ae3582724a341f44a8a1920ef4d962545f3fedea31e441e949719bda282763b7";
    patch_p2_b2_candidate["ticks_per_generation"] = 80;
    patch_p2_b2_candidate["population"] = 64;
    patch_p2_b2_candidate["grid_size"] = 16;
    patch_p2_b2_candidate["generation_summary_scalar_enabled"] = true;
    patch_p2_b2_candidate["compact_summary_enabled"] = false;
    patch_p2_b2_candidate["independent_cells_required"] = true;
    patch_p2_b2_candidate["qualification_gate"] =
        "CUDA-STAGE3-PATCH-P2-B2-G2-INDEPENDENT-CELLS";
    patch_p2_b2_candidate["evidence_sha256"] =
        "4ca866a2fc1401940d8a61d3b12ba444a2d71595e09a73cf1c8c41800";
    out["patch_p2_b2_candidate"] = std::move(patch_p2_b2_candidate);
    pybind11::dict patch_p2_b8_qualification;
    patch_p2_b8_qualification["available"] = true;
    patch_p2_b8_qualification["qualified"] = true;
    patch_p2_b8_qualification["batch_cells"] = 8;
    patch_p2_b8_qualification["generation_count"] = 2;
    pybind11::list patch_p2_b8_candidate_supported_generation_counts;
    for (const int64_t count : {2, 8, 32, 64, 128}) {
        patch_p2_b8_candidate_supported_generation_counts.append(count);
    }
    patch_p2_b8_qualification["candidate_supported_generation_counts"] =
        std::move(patch_p2_b8_candidate_supported_generation_counts);
    pybind11::list patch_p2_b8_qualified_generation_counts;
    patch_p2_b8_qualified_generation_counts.append(2);
    patch_p2_b8_qualified_generation_counts.append(8);
    patch_p2_b8_qualification["qualified_generation_counts"] =
        std::move(patch_p2_b8_qualified_generation_counts);
    patch_p2_b8_qualification["g8_qualification_scope"] =
        "eight_independent_cells_t80_rolling_single_row_delta_validated";
    patch_p2_b8_qualification["g8_candidate_evidence_sha256"] =
        "49094686b0d9b06561e66451cd7d6f55a7f6f82cc0c4041805c054ec8683e1ef";
    patch_p2_b8_qualification["g8_metadata_repair_evidence_sha256"] =
        "988970ef2fc488bedeae6ffd2bd7d6daf3a1458e83ddbd7fc670f1b3bc98e045";
    patch_p2_b8_qualification["g8_metadata_repair_commit"] =
        "3d6e36b2224a31e471cd649593ba7f2b3d6f04fd";
    patch_p2_b8_qualification["ticks_per_generation"] = 80;
    patch_p2_b8_qualification["population"] = 64;
    patch_p2_b8_qualification["grid_size"] = 16;
    patch_p2_b8_qualification["generation_summary_scalar_enabled"] = true;
    patch_p2_b8_qualification["compact_summary_enabled"] = false;
    patch_p2_b8_qualification["independent_cells_required"] = true;
    patch_p2_b8_qualification["qualification_gate"] =
        "CUDA-STAGE3-PATCH-P2-B8-G2-INDEPENDENT-CELLS";
    patch_p2_b8_qualification["evidence_sha256"] =
        "a64ed670c2e372e2a394f6a9a886129b870d12b9863d4993e43bbea8cb6e0737";
    out["patch_p2_b8_qualification"] =
        std::move(patch_p2_b8_qualification);
    pybind11::dict patch_p2_b4_candidate;
    patch_p2_b4_candidate["available"] = true;
    patch_p2_b4_candidate["qualified"] = false;
    patch_p2_b4_candidate["batch_cells"] = 4;
    patch_p2_b4_candidate["generation_count"] = 2;
    pybind11::list patch_p2_b4_candidate_supported_generation_counts;
    for (const int64_t count : {2, 8, 32, 64, 128}) {
        patch_p2_b4_candidate_supported_generation_counts.append(count);
    }
    patch_p2_b4_candidate["candidate_supported_generation_counts"] =
        std::move(patch_p2_b4_candidate_supported_generation_counts);
    pybind11::list patch_p2_b4_qualified_generation_counts;
    patch_p2_b4_candidate["qualified_generation_counts"] =
        std::move(patch_p2_b4_qualified_generation_counts);
    patch_p2_b4_candidate["ticks_per_generation"] = 80;
    patch_p2_b4_candidate["population"] = 64;
    patch_p2_b4_candidate["grid_size"] = 16;
    patch_p2_b4_candidate["generation_summary_scalar_enabled"] = true;
    patch_p2_b4_candidate["compact_summary_enabled"] = false;
    patch_p2_b4_candidate["independent_cells_required"] = true;
    patch_p2_b4_candidate["qualification_gate"] =
        "CUDA-STAGE3-PATCH-P2-B4-G2-INDEPENDENT-CELLS";
    out["patch_p2_b4_candidate"] = std::move(patch_p2_b4_candidate);
    pybind11::dict patch_p2_b48_candidate;
    patch_p2_b48_candidate["available"] = true;
    patch_p2_b48_candidate["qualified"] = true;
    patch_p2_b48_candidate["batch_cells"] = 48;
    patch_p2_b48_candidate["generation_count"] = 2;
    pybind11::list patch_p2_b48_candidate_supported_generation_counts;
    for (const int64_t count : {2, 8, 32, 64, 128}) {
        patch_p2_b48_candidate_supported_generation_counts.append(count);
    }
    patch_p2_b48_candidate["candidate_supported_generation_counts"] =
        std::move(patch_p2_b48_candidate_supported_generation_counts);
    pybind11::list patch_p2_b48_qualified_generation_counts;
    patch_p2_b48_qualified_generation_counts.append(2);
    patch_p2_b48_candidate["qualified_generation_counts"] =
        std::move(patch_p2_b48_qualified_generation_counts);
    patch_p2_b48_candidate["ticks_per_generation"] = 80;
    patch_p2_b48_candidate["population"] = 64;
    patch_p2_b48_candidate["grid_size"] = 16;
    patch_p2_b48_candidate["generation_summary_scalar_enabled"] = true;
    patch_p2_b48_candidate["compact_summary_enabled"] = false;
    patch_p2_b48_candidate["independent_cells_required"] = true;
    patch_p2_b48_candidate["rolling_tick_outputs_required"] = true;
    patch_p2_b48_candidate["full_tick_history_qualified"] = false;
    patch_p2_b48_candidate["unique_seed_semantic_matrix_qualified"] = false;
    patch_p2_b48_candidate["qualification_scope"] =
        "paired_capacity_isolation_with_b8_semantic_evidence_reuse";
    patch_p2_b48_candidate["qualification_gate"] =
        "CUDA-STAGE3-PATCH-P2-B48-G2-PAIRED-CAPACITY-ISOLATION";
    patch_p2_b48_candidate["evidence_sha256"] =
        "272f4e5c859d752513ddfb615d6d18e075414e6ba94b5bfc4fd0f3f45a596fbe";
    patch_p2_b48_candidate["reused_b8_evidence_sha256"] =
        "a64ed670c2e372e2a394f6a9a886129b870d12b9863d4993e43bbea8cb6e0737";
    out["patch_p2_b48_candidate"] = std::move(patch_p2_b48_candidate);
    pybind11::dict patch_p2_b72_candidate;
    patch_p2_b72_candidate["available"] = true;
    patch_p2_b72_candidate["qualified"] = true;
    patch_p2_b72_candidate["batch_cells"] = 72;
    patch_p2_b72_candidate["generation_count"] = 2;
    pybind11::list patch_p2_b72_candidate_supported_generation_counts;
    patch_p2_b72_candidate_supported_generation_counts.append(2);
    patch_p2_b72_candidate["candidate_supported_generation_counts"] =
        std::move(patch_p2_b72_candidate_supported_generation_counts);
    pybind11::list patch_p2_b72_qualified_generation_counts;
    patch_p2_b72_qualified_generation_counts.append(2);
    patch_p2_b72_candidate["qualified_generation_counts"] =
        std::move(patch_p2_b72_qualified_generation_counts);
    patch_p2_b72_candidate["ticks_per_generation"] = 80;
    patch_p2_b72_candidate["population"] = 64;
    patch_p2_b72_candidate["grid_size"] = 16;
    patch_p2_b72_candidate["generation_summary_scalar_enabled"] = true;
    patch_p2_b72_candidate["compact_summary_enabled"] = false;
    patch_p2_b72_candidate["independent_cells_required"] = true;
    patch_p2_b72_candidate["rolling_tick_outputs_required"] = true;
    patch_p2_b72_candidate["full_tick_history_qualified"] = false;
    patch_p2_b72_candidate["unique_seed_semantic_matrix_qualified"] = false;
    patch_p2_b72_candidate["qualification_scope"] =
        "triplicated_capacity_isolation_with_b8_semantic_evidence_reuse";
    patch_p2_b72_candidate["qualification_gate"] =
        "CUDA-STAGE4-PATCH-P2-B72-G2-TRIPLICATED-CAPACITY-ISOLATION";
    patch_p2_b72_candidate["evidence_sha256"] =
        "299216cc2cc6a548a1025ab9cd87cbe699e4e0e50b3ea16c111efcbb7d1d179c";
    patch_p2_b72_candidate["reused_b8_evidence_sha256"] =
        "a64ed670c2e372e2a394f6a9a886129b870d12b9863d4993e43bbea8cb6e0737";
    out["patch_p2_b72_candidate"] = std::move(patch_p2_b72_candidate);
    pybind11::list unsupported_mechanics;
    for (const char* value : {
             "jungle_world",
             "gru_policy",
             "delayed_world_state",
             "multi_cell_batching_above_b72",
             "profile_shape_overrides",
             "generation_count_outside_1_2_8_32_64_128",
         }) {
        unsupported_mechanics.append(value);
    }
    out["unsupported_mechanics"] = std::move(unsupported_mechanics);
    out["production_routing_enabled"] = false;
    out["qualification_metadata_only"] = true;
    out["capacity_limits"] = limits;
    return out;
}

struct GenomeV2C1BatchOutputsV1 {
    torch::Tensor genotype;
    torch::Tensor context;
    torch::Tensor product;
    torch::Tensor regulatory;
    torch::Tensor phenotype;
    torch::Tensor runtime;
    torch::Tensor cost;
};

void validate_genome_v2_c1_inputs_v1(
    const torch::Tensor& genotype,
    const torch::Tensor& context,
    const bool require_cuda
) {
    TORCH_CHECK(
        genotype.defined() && context.defined(),
        "Genome V2 C1 inputs must be defined tensors"
    );
    TORCH_CHECK(
        genotype.scalar_type() == torch::kInt64
            && context.scalar_type() == torch::kInt64,
        "Genome V2 C1 inputs must use int64"
    );
    TORCH_CHECK(
        genotype.dim() == 2
            && genotype.size(1) == genome_v2::kC1GenotypeWidth,
        "Genome V2 C1 genotype must have shape [batch, 120]"
    );
    TORCH_CHECK(
        context.dim() == 2
            && context.size(1) == genome_v2::kContextCount,
        "Genome V2 C1 context must have shape [batch, 8]"
    );
    TORCH_CHECK(
        genotype.size(0) > 0 && genotype.size(0) == context.size(0),
        "Genome V2 C1 inputs require one non-empty matched batch"
    );
    TORCH_CHECK(
        genotype.is_contiguous() && context.is_contiguous(),
        "Genome V2 C1 inputs must be contiguous"
    );
    TORCH_CHECK(
        !genotype.requires_grad() && !context.requires_grad(),
        "Genome V2 C1 inputs must not require gradients"
    );
    TORCH_CHECK(
        genotype.is_cuda() == require_cuda
            && context.is_cuda() == require_cuda,
        require_cuda
            ? "Genome V2 C1 CUDA endpoint requires CUDA tensors"
            : "Genome V2 C1 host endpoint requires CPU tensors"
    );
    TORCH_CHECK(
        genotype.device() == context.device(),
        "Genome V2 C1 inputs must share one device"
    );
}

GenomeV2C1BatchOutputsV1 allocate_genome_v2_c1_outputs_v1(
    const torch::Tensor& reference
) {
    const int64_t batch = reference.size(0);
    const auto options = reference.options().requires_grad(false);
    return {
        torch::empty(
            {batch, genome_v2::kC1GenotypeWidth}, options
        ),
        torch::empty({batch, genome_v2::kContextCount}, options),
        torch::empty({batch, genome_v2::kC1ProductWidth}, options),
        torch::empty({batch, genome_v2::kC1RegulatoryWidth}, options),
        torch::empty({batch, genome_v2::kPhenotypeCount}, options),
        torch::empty({batch, genome_v2::kC1RuntimeWidth}, options),
        torch::empty({batch, genome_v2::kC1CostWidth}, options),
    };
}

pybind11::dict genome_v2_c1_output_dict_v1(
    GenomeV2C1BatchOutputsV1 outputs,
    const char* const backend
) {
    pybind11::dict result;
    result["schema_version"] = "genome_v2_p10_c1_artifact_batch_v1";
    result["backend"] = backend;
    result["genotype"] = std::move(outputs.genotype);
    result["context"] = std::move(outputs.context);
    result["product"] = std::move(outputs.product);
    result["regulatory"] = std::move(outputs.regulatory);
    result["phenotype"] = std::move(outputs.phenotype);
    result["runtime_consequence"] = std::move(outputs.runtime);
    result["cost"] = std::move(outputs.cost);
    result["no_jit"] = true;
    result["no_fallback"] = true;
    result["production_artifact_endpoint"] = true;
    return result;
}

pybind11::dict genome_upgrade_v2_c1_host_batch_v1(
    torch::Tensor genotype,
    torch::Tensor context
) {
    validate_genome_v2_c1_inputs_v1(genotype, context, false);
    auto outputs = allocate_genome_v2_c1_outputs_v1(genotype);
    const int64_t* const genotype_input = genotype.data_ptr<int64_t>();
    const int64_t* const context_input = context.data_ptr<int64_t>();
    for (int64_t row = 0; row < genotype.size(0); ++row) {
        genome_v2::GenotypeV1 value{};
        const int64_t* const genotype_row =
            genotype_input + row * genome_v2::kC1GenotypeWidth;
        int32_t genotype_index = 0;
        for (
            int32_t homolog = 0;
            homolog < genome_v2::kHomologCount;
            ++homolog
        ) {
            for (
                int32_t product = 0;
                product < genome_v2::kProductCount;
                ++product
            ) {
                for (
                    int32_t parameter = 0;
                    parameter < genome_v2::kParameterCount;
                    ++parameter
                ) {
                    value.alleles[homolog][product][parameter] =
                        genotype_row[genotype_index++];
                }
            }
        }
        for (
            int32_t homolog = 0;
            homolog < genome_v2::kHomologCount;
            ++homolog
        ) {
            for (
                int32_t product = 0;
                product < genome_v2::kProductCount;
                ++product
            ) {
                value.modifiers[homolog][product] =
                    genotype_row[genotype_index++];
            }
        }
        const int64_t* const context_row =
            context_input + row * genome_v2::kContextCount;
        const int32_t input_status =
            genome_v2::c1_input_status_v1(value, context_row);
        TORCH_CHECK(
            input_status == 0,
            "Genome V2 C1 host input violates the frozen Q10 contract: ",
            input_status
        );
        genome_v2::ExpressionV1 expression{};
        genome_v2::RegulatoryDiagnosticsV1 diagnostics{};
        genome_v2::express_v1(
            value, context_row, &expression, &diagnostics
        );
        const genome_v2::RuntimeProjectionV1 runtime =
            genome_v2::project_runtime_traits_v1(expression);
        genome_v2::write_c1_dimensions_v1(
            value,
            context_row,
            expression,
            diagnostics,
            runtime,
            outputs.genotype.data_ptr<int64_t>()
                + row * genome_v2::kC1GenotypeWidth,
            outputs.context.data_ptr<int64_t>()
                + row * genome_v2::kContextCount,
            outputs.product.data_ptr<int64_t>()
                + row * genome_v2::kC1ProductWidth,
            outputs.regulatory.data_ptr<int64_t>()
                + row * genome_v2::kC1RegulatoryWidth,
            outputs.phenotype.data_ptr<int64_t>()
                + row * genome_v2::kPhenotypeCount,
            outputs.runtime.data_ptr<int64_t>()
                + row * genome_v2::kC1RuntimeWidth,
            outputs.cost.data_ptr<int64_t>()
                + row * genome_v2::kC1CostWidth
        );
    }
    return genome_v2_c1_output_dict_v1(std::move(outputs), "native");
}

pybind11::dict genome_upgrade_v2_c1_cuda_batch_v1(
    torch::Tensor genotype,
    torch::Tensor context
) {
    validate_genome_v2_c1_inputs_v1(genotype, context, true);
    c10::cuda::CUDAGuard device_guard(genotype.device());
    auto outputs = allocate_genome_v2_c1_outputs_v1(genotype);
    const torch::Tensor status = torch::empty(
        {genotype.size(0)},
        genotype.options().dtype(torch::kInt32).requires_grad(false)
    );
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    controller::launch_genome_upgrade_v2_c1_batch_v1(
        genotype.data_ptr<int64_t>(),
        context.data_ptr<int64_t>(),
        outputs.genotype.data_ptr<int64_t>(),
        outputs.context.data_ptr<int64_t>(),
        outputs.product.data_ptr<int64_t>(),
        outputs.regulatory.data_ptr<int64_t>(),
        outputs.phenotype.data_ptr<int64_t>(),
        outputs.runtime.data_ptr<int64_t>(),
        outputs.cost.data_ptr<int64_t>(),
        status.data_ptr<int32_t>(),
        genotype.size(0),
        stream
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    std::vector<int32_t> host_status(
        static_cast<size_t>(genotype.size(0))
    );
    C10_CUDA_CHECK(cudaMemcpyAsync(
        host_status.data(),
        status.data_ptr<int32_t>(),
        host_status.size() * sizeof(int32_t),
        cudaMemcpyDeviceToHost,
        stream
    ));
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    for (const int32_t value : host_status) {
        TORCH_CHECK(
            value == 0,
            "Genome V2 C1 CUDA input violates the frozen Q10 contract: ",
            value
        );
    }
    return genome_v2_c1_output_dict_v1(std::move(outputs), "cuda");
}

pybind11::dict eiel_authoritative_state_identity_v1() {
    using namespace eiel::authoritative_state::constants_v1;
    pybind11::dict out;
    out["contract_schema_version"] =
        "eiel_authoritative_state_binary_identity_v1";
    out["state_schema_version"] = kSchemaVersion;
    out["state_schema_sha256"] = kSchemaSha256;
    out["source_commit"] = kSourceCommit;
    out["source_inventory_sha256"] = kSourceInventorySha256;
    out["build_identity_sha256"] = kBuildIdentitySha256;
    out["abi_set_sha256"] = kAbiSetSha256;
    out["checkpoint_schema_version"] = "checkpoint_bundle_v2";
    out["checkpoint_schema_sha256"] = kCheckpointSchemaSha256;
    out["chronicle_schema_version"] =
        "eiel_authoritative_state_chronicle_schema_v1";
    out["chronicle_schema_sha256"] = kChronicleSchemaSha256;
    out["provider_capability_identity"] = kProviderCapabilityIdentity;
    out["overlay_version"] = "inheritance-cross-path-overlay-v1";
    out["supported_cache_widths"] = pybind11::make_tuple(1, 2);
    out["supported_profiles"] = pybind11::make_tuple(
        "eiel-production-runtime-v1"
    );
    out["build_toolchain_identity"] = __VERSION__;
#ifdef EIEL_AUTHORITATIVE_STATE_COMPILE_FLAGS
    out["compile_flags"] = EIEL_AUTHORITATIVE_STATE_COMPILE_FLAGS;
#else
    out["compile_flags"] = "extension-build-defaults";
#endif
    out["artifact_build_id"] = kArtifactBuildId;
    pybind11::list variants;
    for (const auto& row : kAbiVariants) {
        pybind11::dict variant;
        variant["entrypoint"] = row.variant_id;
        variant["abi_version"] = row.variant_id;
        variant["argument_count"] = row.argument_count;
        variant["argument_order_sha256"] = row.argument_order_sha256;
        variants.append(variant);
    }
    out["abi_variants"] = variants;
    return out;
}

pybind11::dict eiel_authoritative_state_build_provenance_v1() {
#if !defined(EIEL_AUTHORITATIVE_STATE_BUILD_SOURCE_COMMIT)
#error "authoritative-state build source commit must be injected by the builder"
#endif
#if !defined(EIEL_AUTHORITATIVE_STATE_COMPILED_SOURCE_SNAPSHOT_SHA256)
#error "authoritative-state compiled source snapshot must be injected by the builder"
#endif
    pybind11::dict out;
    out["schema_version"] =
        "eiel_authoritative_state_binary_build_provenance_v1";
    out["build_source_commit"] =
        EIEL_AUTHORITATIVE_STATE_BUILD_SOURCE_COMMIT;
    out["compiled_source_snapshot_sha256"] =
        EIEL_AUTHORITATIVE_STATE_COMPILED_SOURCE_SNAPSHOT_SHA256;
    return out;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.doc() = "Raw-CUDA Architecture B flattened dynamic Flat controller";
    module.def(
        "eiel_cuda_architecture_b_genome_upgrade_v2_c1_native_batch_v1",
        &genome_upgrade_v2_c1_host_batch_v1
    );
    module.def(
        "eiel_cuda_architecture_b_genome_upgrade_v2_c1_cuda_batch_v1",
        &genome_upgrade_v2_c1_cuda_batch_v1
    );
    module.def(
        "eiel_cuda_architecture_b_flattened_controller_v1",
        &flattened_flat_dynamic_policy_food_birth_controller_v1
    );
    module.def(
        "eiel_cuda_architecture_b_flattened_controller_resource_receipt_v1",
        &flattened_flat_dynamic_policy_food_birth_controller_resource_receipt_v1
    );
    module.def(
        "eiel_cuda_architecture_b_flattened_generation_count_capability_v1",
        &flattened_flat_dynamic_policy_food_birth_controller_generation_count_capability_v1
    );
    module.def(
        "eiel_authoritative_state_identity_v1",
        &eiel_authoritative_state_identity_v1
    );
    module.def(
        "eiel_authoritative_state_build_provenance_v1",
        &eiel_authoritative_state_build_provenance_v1
    );
    module.def(
        "eiel_cuda_architecture_b_prepare_cpython_gaussian_window_v1",
        &resident_window_refill::prepare_cpython_gaussian_window_v1
    );
    module.def(
        "eiel_cuda_architecture_b_prepare_fixed_mlp_policy_window_v1",
        &resident_window_refill::prepare_fixed_mlp_policy_window_v1,
        pybind11::arg("state_cpu"),
        pybind11::arg("episodes"),
        pybind11::arg("width"),
        pybind11::arg("mutation_enable"),
        pybind11::arg("mutation_mode"),
        pybind11::arg("mutation_rate"),
        pybind11::arg("mutation_std"),
        pybind11::arg("execution_route_sha256") = ""
    );
    module.def(
        "eiel_cuda_architecture_b_refill_cpython_gaussian_window_v1",
        &resident_window_refill::refill_cpython_gaussian_window_v1
    );
    module.def(
        "eiel_cuda_architecture_b_refill_or_reuse_cpython_gaussian_window_v1",
        &resident_window_refill::ensure_cpython_gaussian_window_v1
    );
    module.def(
        "eiel_cuda_architecture_b_refill_fixed_mlp_policy_window_v1",
        [](const torch::Tensor& source_state,
           torch::Tensor delta,
           torch::Tensor mutation_mask,
           torch::Tensor crossover_mask_packed,
           torch::Tensor post_state_packed,
           torch::Tensor semantic_tape,
           const torch::Tensor& absolute_cursor,
           const torch::Tensor& semantic_counter,
           torch::Tensor window_base,
           const int64_t policy_row_width,
           const bool mutation_enable,
           const std::string& mutation_mode,
           const double mutation_rate,
           const double mutation_std,
           const std::string& execution_route_sha256) {
            const auto lease = resident_window_refill::
                policy_window_storage_execution_lease_v1(
                    crossover_mask_packed
                );
            const std::unique_lock<std::mutex> guard(
                *lease, std::try_to_lock
            );
            TORCH_CHECK(
                guard.owns_lock(),
                "public policy-window refill raced an active owner"
            );
            return resident_window_refill::refill_fixed_mlp_policy_window_v1(
                source_state,
                std::move(delta),
                std::move(mutation_mask),
                std::move(crossover_mask_packed),
                std::move(post_state_packed),
                std::move(semantic_tape),
                absolute_cursor,
                semantic_counter,
                std::move(window_base),
                policy_row_width,
                mutation_enable,
                mutation_mode,
                mutation_rate,
                mutation_std,
                execution_route_sha256
            );
        },
        pybind11::arg("source_state"),
        pybind11::arg("delta"),
        pybind11::arg("mutation_mask"),
        pybind11::arg("crossover_mask_packed"),
        pybind11::arg("post_state_packed"),
        pybind11::arg("semantic_tape"),
        pybind11::arg("absolute_cursor"),
        pybind11::arg("semantic_counter"),
        pybind11::arg("window_base"),
        pybind11::arg("policy_row_width"),
        pybind11::arg("mutation_enable"),
        pybind11::arg("mutation_mode"),
        pybind11::arg("mutation_rate"),
        pybind11::arg("mutation_std"),
        pybind11::arg("execution_route_sha256") = ""
    );
    module.def(
        "eiel_cuda_architecture_b_refill_or_reuse_fixed_mlp_policy_window_v1",
        [](const torch::Tensor& source_state,
           torch::Tensor delta,
           torch::Tensor mutation_mask,
           torch::Tensor crossover_mask_packed,
           torch::Tensor post_state_packed,
           torch::Tensor semantic_tape,
           const torch::Tensor& absolute_cursor,
           const torch::Tensor& semantic_counter,
           torch::Tensor window_base,
           const int64_t policy_row_width,
           const bool mutation_enable,
           const std::string& mutation_mode,
           const double mutation_rate,
           const double mutation_std,
           const int64_t required_future_episodes,
           const std::string& execution_route_sha256) {
            const auto lease = resident_window_refill::
                policy_window_storage_execution_lease_v1(
                    crossover_mask_packed
                );
            const std::unique_lock<std::mutex> guard(
                *lease, std::try_to_lock
            );
            TORCH_CHECK(
                guard.owns_lock(),
                "public policy-window ensure raced an active owner"
            );
            return resident_window_refill::ensure_fixed_mlp_policy_window_v1(
                source_state,
                std::move(delta),
                std::move(mutation_mask),
                std::move(crossover_mask_packed),
                std::move(post_state_packed),
                std::move(semantic_tape),
                absolute_cursor,
                semantic_counter,
                std::move(window_base),
                policy_row_width,
                mutation_enable,
                mutation_mode,
                mutation_rate,
                mutation_std,
                required_future_episodes,
                execution_route_sha256
            );
        },
        pybind11::arg("source_state"),
        pybind11::arg("delta"),
        pybind11::arg("mutation_mask"),
        pybind11::arg("crossover_mask_packed"),
        pybind11::arg("post_state_packed"),
        pybind11::arg("semantic_tape"),
        pybind11::arg("absolute_cursor"),
        pybind11::arg("semantic_counter"),
        pybind11::arg("window_base"),
        pybind11::arg("policy_row_width"),
        pybind11::arg("mutation_enable"),
        pybind11::arg("mutation_mode"),
        pybind11::arg("mutation_rate"),
        pybind11::arg("mutation_std"),
        pybind11::arg("required_future_episodes"),
        pybind11::arg("execution_route_sha256") = ""
    );
    module.def(
        "eiel_cuda_architecture_b_refill_drive_noise_window_v1",
        &resident_window_refill::refill_drive_noise_window_v1
    );
    module.def(
        "eiel_cuda_architecture_b_drain_event_window_once_v1",
        &resident_event_archive::drain_event_window_once_v1
    );
    module.def(
        "eiel_cuda_architecture_b_drain_event_window_once_with_budget_v1",
        &resident_event_archive::drain_event_window_once_with_budget_v1
    );
    module.def(
        "eiel_cuda_architecture_b_execute_resident_windows_v1",
        &resident_window_controller::execute_resident_windows_v1
    );
    module.def(
        "eiel_cuda_architecture_b_execute_direct_resident_windows_v1",
        &execute_resident_windows_direct_v1
    );
    resident_run_control_binding::bind_run_control_v2(module);
    const auto direct_v1 = module.attr(
        "eiel_cuda_architecture_b_execute_direct_resident_windows_v1"
    ).cast<pybind11::function>();
    module.def(
        "eiel_cuda_architecture_b_execute_direct_resident_windows_v2",
        [direct_v1](
            const std::shared_ptr<resident_run_control::RunControlV2>& run_control,
            const pybind11::args& arguments
        ) {
            return execute_direct_resident_windows_v2(
                direct_v1,
                run_control,
                arguments
            );
        }
    );
}
