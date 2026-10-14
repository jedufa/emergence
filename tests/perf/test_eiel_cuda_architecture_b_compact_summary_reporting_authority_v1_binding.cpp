#include "cuda_architecture_b_compact_summary_reporting_authority_v1.cuh"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include <cstddef>
#include <cstdint>

namespace authority =
    eiel::cuda_architecture_b::compact_summary_reporting_authority_v1;

torch::Tensor run_fixture(
    const torch::Tensor& device_anchor,
    const std::int64_t fixture,
    const std::int64_t generations,
    const std::int64_t row_capacity,
    const std::int64_t slot_capacity
) {
    TORCH_CHECK(device_anchor.is_cuda(), "device_anchor must be CUDA");
    TORCH_CHECK(device_anchor.is_contiguous(), "device_anchor must be contiguous");
    TORCH_CHECK(generations > 0 && generations <= 2, "generations must be 1 or 2");
    TORCH_CHECK(
        row_capacity > 0 && row_capacity <= authority::kDynamicRowCapacityV1,
        "row_capacity must be in [1, 224]"
    );
    TORCH_CHECK(
        slot_capacity > authority::kStrictRowCapacityV1,
        "slot_capacity must leave one free slot for overflow qualification"
    );
    TORCH_CHECK(slot_capacity <= 1024, "slot_capacity is unreasonably large");
    TORCH_CHECK(
        fixture >= authority::kFixtureStartBornDiedReuseV1
            && fixture <= authority::kFixtureMalformedLineageSourceV1,
        "unknown qualification fixture"
    );
    TORCH_CHECK(
        fixture != authority::kFixtureG2IsolationV1 || generations == 2,
        "g2 isolation requires exactly two generations"
    );
    TORCH_CHECK(
        fixture == authority::kFixtureG2IsolationV1 || generations == 1,
        "non-g2 fixtures require exactly one generation"
    );
    TORCH_CHECK(
        fixture != authority::kFixtureOverflowV1
            || row_capacity == authority::kStrictRowCapacityV1,
        "overflow qualification requires the strict 64-row capacity"
    );

    c10::cuda::CUDAGuard guard(device_anchor.device());
    const auto byte_stride = static_cast<std::int64_t>(
        authority::retained_bytes_per_generation_cell_v1(
            static_cast<std::int32_t>(row_capacity)
        )
    );
    const torch::TensorOptions byte_options = device_anchor.options()
        .dtype(torch::kUInt8)
        .requires_grad(false);
    const torch::TensorOptions int_options = device_anchor.options()
        .dtype(torch::kInt32)
        .requires_grad(false);
    const torch::TensorOptions double_options = device_anchor.options()
        .dtype(torch::kFloat64)
        .requires_grad(false);
    torch::Tensor retained = torch::zeros(
        {generations, byte_stride}, byte_options
    );
    torch::Tensor slot_to_row = torch::empty(
        {generations, slot_capacity}, int_options
    );
    torch::Tensor effective = torch::empty(
        {generations, row_capacity}, double_options
    );

    const cudaStream_t stream =
        at::cuda::getCurrentCUDAStream(device_anchor.device().index()).stream();
    authority::launch_reporting_authority_fixture_v1(
        retained.data_ptr<std::uint8_t>(),
        slot_to_row.data_ptr<std::int32_t>(),
        effective.data_ptr<double>(),
        generations,
        static_cast<std::int32_t>(row_capacity),
        static_cast<std::int32_t>(slot_capacity),
        static_cast<std::int32_t>(fixture),
        stream
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return retained;
}

pybind11::dict resource_receipt() {
    const authority::KernelResourceReceiptV1 receipt =
        authority::query_reporting_authority_fixture_v1_resources();
    pybind11::dict result;
    result["contract_version"] = authority::kContractVersionV1;
    result["retained_layout_version"] = authority::kRetainedLayoutVersionV2;
    result["binary_version"] = receipt.binary_version;
    result["ptx_version"] = receipt.ptx_version;
    result["max_threads_per_block"] = receipt.max_threads_per_block;
    result["registers_per_thread"] = receipt.registers_per_thread;
    result["occupancy_blocks_per_multiprocessor"] =
        receipt.occupancy_blocks_per_multiprocessor;
    result["constant_bytes"] = receipt.constant_bytes;
    result["local_bytes_per_thread"] = receipt.local_bytes_per_thread;
    result["shared_bytes_per_block"] = receipt.shared_bytes_per_block;
    result["header_bytes"] = sizeof(authority::GenerationAuthorityHeaderV1);
    result["behavior_row_bytes"] = sizeof(authority::BehaviorAuthorityRowV1);
    result["lineage_row_bytes"] = sizeof(authority::LineageAuthorityRowV1);
    result["strict_bytes_per_generation_cell"] =
        authority::retained_bytes_per_generation_cell_v1(
            authority::kStrictRowCapacityV1
        );
    result["dynamic_bytes_per_generation_cell"] =
        authority::retained_bytes_per_generation_cell_v1(
            authority::kDynamicRowCapacityV1
        );
    pybind11::dict lineage_offsets;
    lineage_offsets["agent_id"] = offsetof(authority::LineageAuthorityRowV1, agent_id);
    lineage_offsets["energy"] = offsetof(authority::LineageAuthorityRowV1, energy);
    lineage_offsets["energy_collected_exact"] =
        offsetof(authority::LineageAuthorityRowV1, energy_collected_exact);
    lineage_offsets["episode0_pred_err_mean"] =
        offsetof(authority::LineageAuthorityRowV1, episode0_pred_err_mean);
    lineage_offsets["identity_drift_ema"] =
        offsetof(authority::LineageAuthorityRowV1, identity_drift_ema);
    lineage_offsets["health"] = offsetof(authority::LineageAuthorityRowV1, health);
    lineage_offsets["injury"] = offsetof(authority::LineageAuthorityRowV1, injury);
    lineage_offsets["drives"] = offsetof(authority::LineageAuthorityRowV1, drives);
    lineage_offsets["pred_err_sum"] =
        offsetof(authority::LineageAuthorityRowV1, pred_err_sum);
    lineage_offsets["drive_jitter"] =
        offsetof(authority::LineageAuthorityRowV1, drive_jitter);
    lineage_offsets["aggression_bias_f32"] =
        offsetof(authority::LineageAuthorityRowV1, aggression_bias_f32);
    lineage_offsets["exploration_bias_f32"] =
        offsetof(authority::LineageAuthorityRowV1, exploration_bias_f32);
    lineage_offsets["y"] = offsetof(authority::LineageAuthorityRowV1, y);
    lineage_offsets["x"] = offsetof(authority::LineageAuthorityRowV1, x);
    lineage_offsets["founder_id"] =
        offsetof(authority::LineageAuthorityRowV1, founder_id);
    lineage_offsets["birth_generation"] =
        offsetof(authority::LineageAuthorityRowV1, birth_generation);
    lineage_offsets["birth_tick"] =
        offsetof(authority::LineageAuthorityRowV1, birth_tick);
    lineage_offsets["lineage_depth"] =
        offsetof(authority::LineageAuthorityRowV1, lineage_depth);
    lineage_offsets["offspring_count"] =
        offsetof(authority::LineageAuthorityRowV1, offspring_count);
    lineage_offsets["survival_steps"] =
        offsetof(authority::LineageAuthorityRowV1, survival_steps);
    lineage_offsets["first_repro_step"] =
        offsetof(authority::LineageAuthorityRowV1, first_repro_step);
    lineage_offsets["pred_err_count"] =
        offsetof(authority::LineageAuthorityRowV1, pred_err_count);
    lineage_offsets["patch_habitat_lineage_id"] =
        offsetof(authority::LineageAuthorityRowV1, patch_habitat_lineage_id);
    lineage_offsets["hidden_size"] =
        offsetof(authority::LineageAuthorityRowV1, hidden_size);
    lineage_offsets["memory_dim"] =
        offsetof(authority::LineageAuthorityRowV1, memory_dim);
    lineage_offsets["use_gru"] = offsetof(authority::LineageAuthorityRowV1, use_gru);
    lineage_offsets["reserved"] =
        offsetof(authority::LineageAuthorityRowV1, reserved);
    result["lineage_field_offsets"] = lineage_offsets;
    return result;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def(
        "run_fixture",
        &run_fixture,
        pybind11::arg("device_anchor"),
        pybind11::arg("fixture"),
        pybind11::arg("generations") = 1,
        pybind11::arg("row_capacity") = authority::kStrictRowCapacityV1,
        pybind11::arg("slot_capacity") = authority::kDynamicRowCapacityV1 + 1
    );
    module.def("resource_receipt", &resource_receipt);
}
