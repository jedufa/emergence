#include "eed_resource_impact_native_v1.hpp"

#include <torch/extension.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace eiel::eed_resource_impact_native_v1 {
namespace {

Config validated_config(
    const pybind11::dict& raw,
    const std::int32_t expected_schema_version = kSchemaVersion
) {
    const auto schema_version =
        pybind11::cast<std::int32_t>(raw["schema_version"]);
    TORCH_CHECK(
        schema_version == expected_schema_version,
        "EED native schema version mismatch"
    );
    const auto impact = pybind11::cast<float>(raw["impact_per_success"]);
    const auto decay = pybind11::cast<float>(raw["decay_per_step"]);
    const auto maximum = pybind11::cast<float>(raw["maximum_load"]);
    const auto applied = pybind11::cast<bool>(raw["impact_applied"]);
    const auto ecology_frozen = pybind11::cast<bool>(raw["ecology_frozen"]);
    const auto phenotype_fixed = pybind11::cast<bool>(raw["phenotype_fixed"]);
    const auto fixed_axis_mode =
        pybind11::cast<std::int32_t>(raw["fixed_axis_mode"]);
    TORCH_CHECK(
        std::isfinite(impact) && impact >= 0.0F,
        "EED impact_per_success must be finite and nonnegative"
    );
    TORCH_CHECK(
        std::isfinite(decay) && decay >= 0.0F && decay <= 1.0F,
        "EED decay_per_step must be finite and in [0,1]"
    );
    TORCH_CHECK(
        std::isfinite(maximum) && maximum >= 0.0F,
        "EED maximum_load must be finite and nonnegative"
    );
    TORCH_CHECK(
        fixed_axis_mode >= 0 && fixed_axis_mode <= 3,
        "EED fixed_axis_mode must be in [0,3]"
    );
    return {
        impact,
        decay,
        maximum,
        applied,
        ecology_frozen,
        phenotype_fixed,
        fixed_axis_mode,
    };
}

float fixed_axis(const Config config, const std::int64_t index) {
    if (config.fixed_axis_mode == 0) {
        return 0.0F;
    }
    if (config.fixed_axis_mode == 1) {
        return 1.0F;
    }
    if (config.fixed_axis_mode == 2) {
        return -1.0F;
    }
    return index % 2 == 0 ? 1.0F : -1.0F;
}

void validate_inputs(
    const std::vector<torch::Tensor>& state,
    const torch::Tensor& resource_classes,
    const torch::Tensor& flat_indices,
    const torch::Tensor& handling_axes
) {
    TORCH_CHECK(
        state.size() == kStateTensorCount,
        "EED native transition requires ten state tensors"
    );
    const auto& reference = state.front();
    TORCH_CHECK(
        reference.scalar_type() == torch::kFloat32
            && reference.dim() == 3 && reference.is_contiguous()
            && !reference.requires_grad(),
        "EED state tensors must be contiguous float32[cells,height,width]"
    );
    TORCH_CHECK(
        reference.size(0) > 0 && reference.size(1) > 0
            && reference.size(2) > 0,
        "EED state tensor grid must be nonempty"
    );
    for (const auto& tensor : state) {
        TORCH_CHECK(
            tensor.scalar_type() == torch::kFloat32
                && tensor.sizes() == reference.sizes()
                && tensor.device() == reference.device()
                && tensor.is_contiguous() && !tensor.requires_grad(),
            "all EED state tensors must share dtype, shape, and device"
        );
    }
    const auto events = resource_classes.numel();
    TORCH_CHECK(
        resource_classes.scalar_type() == torch::kInt32
            && flat_indices.scalar_type() == torch::kInt64
            && handling_axes.scalar_type() == torch::kFloat32
            && resource_classes.dim() == 1 && flat_indices.dim() == 1
            && handling_axes.dim() == 1
            && flat_indices.numel() == events
            && handling_axes.numel() == events,
        "EED successful-gather event tensors have invalid dtype or shape"
    );
    for (const auto& tensor : {
             resource_classes, flat_indices, handling_axes
         }) {
        TORCH_CHECK(
            tensor.device() == reference.device() && tensor.is_contiguous()
                && !tensor.requires_grad(),
            "EED successful-gather event tensors must share the state device"
        );
    }
    const auto classes_cpu =
        resource_classes.to(torch::kCPU).contiguous();
    const auto indices_cpu = flat_indices.to(torch::kCPU).contiguous();
    const auto axes_cpu = handling_axes.to(torch::kCPU).contiguous();
    const auto* const classes = classes_cpu.data_ptr<std::int32_t>();
    const auto* const indices = indices_cpu.data_ptr<std::int64_t>();
    const auto* const axes = axes_cpu.data_ptr<float>();
    for (std::int64_t event = 0; event < events; ++event) {
        TORCH_CHECK(
            classes[event] == kResourceA || classes[event] == kResourceB,
            "EED successful-gather resource class must be A or B"
        );
        TORCH_CHECK(
            indices[event] >= 0 && indices[event] < reference.numel(),
            "EED successful-gather flat index is out of range"
        );
        TORCH_CHECK(
            std::isfinite(axes[event]),
            "EED successful-gather handling axis must be finite"
        );
    }
}

float bounded(float value, float low, float high) {
    return std::fmin(high, std::fmax(low, value));
}

void transition_cpu_v1(
    const std::vector<torch::Tensor>& state,
    const torch::Tensor& resource_classes,
    const torch::Tensor& flat_indices,
    const torch::Tensor& handling_axes,
    const Config config
) {
    const auto total = state.front().numel();
    auto* const a_pending = state.at(6).data_ptr<float>();
    auto* const b_pending = state.at(7).data_ptr<float>();
    const auto* const classes = resource_classes.data_ptr<std::int32_t>();
    const auto* const indices = flat_indices.data_ptr<std::int64_t>();
    const auto* const axes = handling_axes.data_ptr<float>();
    for (std::int64_t event = 0; event < resource_classes.numel(); ++event) {
        const auto index = indices[event];
        TORCH_CHECK(
            index >= 0 && index < total,
            "EED successful-gather flat index is out of range"
        );
        const auto resource_class = classes[event];
        TORCH_CHECK(
            resource_class == kResourceA || resource_class == kResourceB,
            "EED successful-gather resource class must be A or B"
        );
        TORCH_CHECK(
            std::isfinite(axes[event]),
            "EED successful-gather handling axis must be finite"
        );
        if (config.impact_applied && !config.ecology_frozen) {
            const float axis = config.phenotype_fixed
                ? fixed_axis(config, index)
                : axes[event];
            const float signed_axis = resource_class == kResourceA
                ? axis
                : -axis;
            const float impact =
                config.impact_per_success * std::fmax(0.0F, signed_axis);
            auto* const pending =
                resource_class == kResourceA ? a_pending : b_pending;
            pending[index] = pending[index] + impact;
        }
    }

    const float retained = 1.0F - config.decay_per_step;
    for (std::int64_t resource = 0; resource < 2; ++resource) {
        auto* const amount = state.at(resource).data_ptr<float>();
        const auto* const capacity = state.at(2 + resource).data_ptr<float>();
        const auto* const renewal = state.at(4 + resource).data_ptr<float>();
        auto* const pending = state.at(6 + resource).data_ptr<float>();
        auto* const load = state.at(8 + resource).data_ptr<float>();
        for (std::int64_t index = 0; index < total; ++index) {
            if (config.ecology_frozen) {
                continue;
            }
            load[index] = config.impact_applied
                ? bounded(
                      load[index] * retained + pending[index],
                      0.0F,
                      config.maximum_load
                  )
                : 0.0F;
            pending[index] = 0.0F;
            const float effective =
                renewal[index] * (1.0F - bounded(load[index], 0.0F, 1.0F));
            amount[index] = std::fmin(
                capacity[index],
                std::fmax(0.0F, amount[index] + effective)
            );
        }
    }
}

pybind11::dict transition(
    std::vector<torch::Tensor> state,
    torch::Tensor resource_classes,
    torch::Tensor flat_indices,
    torch::Tensor handling_axes,
    const pybind11::dict& raw_config
) {
    validate_inputs(state, resource_classes, flat_indices, handling_axes);
    const auto config = validated_config(raw_config);
    if (state.front().is_cuda()) {
        transition_cuda_v1(
            state, resource_classes, flat_indices, handling_axes, config
        );
    } else {
        TORCH_CHECK(
            state.front().device().is_cpu(),
            "EED native transition supports CPU or CUDA only"
        );
        transition_cpu_v1(
            state, resource_classes, flat_indices, handling_axes, config
        );
    }
    pybind11::dict out;
    out["schema_version"] = kSchemaVersion;
    out["owner"] = "native_cpp_cuda";
    out["state_tensor_count"] = kStateTensorCount;
    out["successful_gather_event_count"] = resource_classes.numel();
    out["impact_applied"] = config.impact_applied;
    out["ecology_frozen"] = config.ecology_frozen;
    out["phenotype_fixed"] = config.phenotype_fixed;
    out["new_rng_draws"] = 0;
    out["state_tensors"] = state;
    return out;
}

pybind11::dict capability() {
    pybind11::dict out;
    out["schema_version"] = kSchemaVersion;
    out["overlay_id"] = "world.overlay.eed_resource_impact.v1";
    out["cpu_oracle"] = true;
    out["cuda_owner"] = true;
    out["state_tensor_count"] = kStateTensorCount;
    out["deterministic_settlement"] = true;
    out["new_rng_draws"] = 0;
    out["production_route_integrated"] = false;
    return out;
}

}  // namespace

ResidentContextV1 validated_resident_context_v1(
    const pybind11::list& state_tensors,
    const pybind11::dict& raw_config
) {
    TORCH_CHECK(
        state_tensors.size() == 15,
        "EED resident route requires fifteen state tensors"
    );
    std::vector<torch::Tensor> tensors;
    tensors.reserve(15);
    for (const auto item : state_tensors) {
        tensors.push_back(pybind11::cast<torch::Tensor>(item));
    }
    const auto& reference = tensors.front();
    TORCH_CHECK(
        reference.is_cuda() && reference.scalar_type() == torch::kFloat32
            && reference.dim() == 3 && reference.is_contiguous(),
        "EED resident grid state must be contiguous CUDA float32[B,H,W]"
    );
    const auto cells = reference.size(0);
    const auto height = reference.size(1);
    const auto width = reference.size(2);
    TORCH_CHECK(
        cells > 0 && height > 0 && width > 0,
        "EED resident grid state must be nonempty"
    );
    for (std::size_t index = 0; index <= 10; ++index) {
        const auto& tensor = tensors.at(index);
        TORCH_CHECK(
            tensor.is_cuda() && tensor.device() == reference.device()
                && tensor.scalar_type() == torch::kFloat32
                && tensor.sizes() == reference.sizes()
                && tensor.is_contiguous() && !tensor.requires_grad(),
            "EED resident grid tensors must share CUDA dtype and shape"
        );
    }
    const auto capacity = tensors.at(11).size(1);
    TORCH_CHECK(capacity > 0, "EED resident agent capacity must be positive");
    for (std::size_t index = 11; index <= 13; ++index) {
        const auto& tensor = tensors.at(index);
        TORCH_CHECK(
            tensor.is_cuda() && tensor.device() == reference.device()
                && tensor.scalar_type() == torch::kInt32 && tensor.dim() == 2
                && tensor.size(0) == cells && tensor.size(1) == capacity
                && tensor.is_contiguous() && !tensor.requires_grad(),
            "EED resident handling cursor tensors must be int32[B,K]"
        );
    }
    const auto& progress = tensors.at(14);
    TORCH_CHECK(
        progress.is_cuda() && progress.device() == reference.device()
            && progress.scalar_type() == torch::kFloat32
            && progress.sizes() == tensors.at(11).sizes()
            && progress.is_contiguous() && !progress.requires_grad(),
        "EED resident handling progress must be float32[B,K]"
    );
    const auto fixed_axis_mode_by_cell =
        pybind11::cast<torch::Tensor>(raw_config["fixed_axis_mode_by_cell"]);
    TORCH_CHECK(
        fixed_axis_mode_by_cell.is_cuda()
            && fixed_axis_mode_by_cell.device() == reference.device()
            && fixed_axis_mode_by_cell.scalar_type() == torch::kInt32
            && fixed_axis_mode_by_cell.dim() == 1
            && fixed_axis_mode_by_cell.size(0) == cells
            && fixed_axis_mode_by_cell.is_contiguous()
            && !fixed_axis_mode_by_cell.requires_grad(),
        "EED fixed-axis modes must be contiguous CUDA int32[B]"
    );
    const auto minimum_fixed_axis_mode =
        fixed_axis_mode_by_cell.min().item<std::int32_t>();
    const auto maximum_fixed_axis_mode =
        fixed_axis_mode_by_cell.max().item<std::int32_t>();
    TORCH_CHECK(
        minimum_fixed_axis_mode >= 0 && maximum_fixed_axis_mode <= 3,
        "EED per-cell fixed-axis modes must be in [0,3]"
    );
    torch::Tensor gather_amount_by_cell;
    if (raw_config.contains("gather_amount_by_cell")) {
        gather_amount_by_cell = pybind11::cast<torch::Tensor>(
            raw_config["gather_amount_by_cell"]
        );
        TORCH_CHECK(
            gather_amount_by_cell.is_cuda()
                && gather_amount_by_cell.device() == reference.device()
                && gather_amount_by_cell.scalar_type() == torch::kFloat64
                && gather_amount_by_cell.sizes() == torch::IntArrayRef({cells})
                && gather_amount_by_cell.is_contiguous()
                && torch::isfinite(gather_amount_by_cell).all().item<bool>()
                && torch::all(gather_amount_by_cell > 0.0).item<bool>(),
            "EED gather_amount_by_cell must be positive finite float64[B]"
        );
    }
    torch::Tensor regrow_amount_by_cell;
    if (raw_config.contains("regrow_amount_by_cell")) {
        regrow_amount_by_cell = pybind11::cast<torch::Tensor>(
            raw_config["regrow_amount_by_cell"]
        );
        TORCH_CHECK(
            regrow_amount_by_cell.is_cuda()
                && regrow_amount_by_cell.device() == reference.device()
                && regrow_amount_by_cell.scalar_type() == torch::kFloat64
                && regrow_amount_by_cell.sizes() == torch::IntArrayRef({cells})
                && regrow_amount_by_cell.is_contiguous()
                && torch::isfinite(regrow_amount_by_cell).all().item<bool>()
                && torch::all(regrow_amount_by_cell >= 0.0).item<bool>(),
            "EED regrow_amount_by_cell must be nonnegative finite float64[B]"
        );
    }

    const auto impact = validated_config(
        raw_config, kResidentSchemaVersion
    );
    const auto exact_float = [&raw_config](const char* const name) {
        const auto value = pybind11::cast<float>(raw_config[name]);
        TORCH_CHECK(std::isfinite(value), "EED config ", name, " is nonfinite");
        return value;
    };
    const auto presence = exact_float("food_presence_threshold");
    const auto gather = exact_float("gather_amount");
    const auto threshold_a = exact_float("resource_a_handling_threshold");
    const auto threshold_b = exact_float("resource_b_handling_threshold");
    const auto return_a = exact_float("resource_a_return");
    const auto return_b = exact_float("resource_b_return");
    TORCH_CHECK(
        presence >= 0.0F && gather > 0.0F && threshold_a > 0.0F
            && threshold_b > 0.0F && return_a >= 0.0F && return_b >= 0.0F,
        "EED resident handling config is outside its bounded domain"
    );
    const auto enabled = pybind11::cast<bool>(raw_config["enabled"]);
    TORCH_CHECK(enabled, "EED resident entrypoint requires enabled=true");
    const auto equalized =
        pybind11::cast<bool>(raw_config["equalized_sham"]);
    return {
        true,
        {
            kResidentSchemaVersion,
            presence,
            gather,
            threshold_a,
            threshold_b,
            return_a,
            return_b,
            impact,
            equalized,
            true,
        },
        fixed_axis_mode_by_cell.data_ptr<std::int32_t>(),
        gather_amount_by_cell.defined()
            ? gather_amount_by_cell.data_ptr<double>() : nullptr,
        regrow_amount_by_cell.defined()
            ? regrow_amount_by_cell.data_ptr<double>() : nullptr,
        tensors.at(0).data_ptr<float>(),
        tensors.at(1).data_ptr<float>(),
        tensors.at(2).data_ptr<float>(),
        tensors.at(3).data_ptr<float>(),
        tensors.at(4).data_ptr<float>(),
        tensors.at(5).data_ptr<float>(),
        tensors.at(6).data_ptr<float>(),
        tensors.at(7).data_ptr<float>(),
        tensors.at(8).data_ptr<float>(),
        tensors.at(9).data_ptr<float>(),
        tensors.at(10).data_ptr<float>(),
        tensors.at(11).data_ptr<std::int32_t>(),
        tensors.at(12).data_ptr<std::int32_t>(),
        tensors.at(13).data_ptr<std::int32_t>(),
        tensors.at(14).data_ptr<float>(),
        cells,
        height,
        width,
        capacity,
    };
}

void bind_eed_resource_impact_native_v1(pybind11::module_& module) {
    module.def(
        "eiel_eed_resource_impact_transition_v1",
        &transition,
        pybind11::arg("state_tensors"),
        pybind11::arg("successful_resource_classes"),
        pybind11::arg("successful_flat_indices"),
        pybind11::arg("successful_handling_axes"),
        pybind11::arg("config")
    );
    module.def(
        "eiel_eed_resource_impact_capability_v1",
        &capability
    );
}

}  // namespace eiel::eed_resource_impact_native_v1
