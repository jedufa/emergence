#include <torch/extension.h>

#include <string>
#include <vector>

#include "cuda_architecture_b_flattened_gru_controller_v3.hpp"
#include "cuda_architecture_b_flattened_gmnw_controller_v1.hpp"
#include "cuda_architecture_b_flattened_jungle_controller_v4.hpp"
#include "cuda_architecture_b_flattened_modular_controller_v1.hpp"
#include "cuda_architecture_b_flattened_wdl_controller_v2.hpp"
#ifdef EIEL_WDL_JUNGLE_EED_HAPB_FIXED_MLP_NATIVE_BINDING_V1
#include "cuda_architecture_b_hapb_policy_context_v1.hpp"
#include "wdl_jungle_eed_hapb_fixed_mlp_native_binding_v1.hpp"
#endif
#ifndef EIEL_NON_EED_WDL_JUNGLE_ONLY
#include "eed_resource_impact_native_v1.hpp"
#endif

#ifndef EIEL_TW_LIFECYCLE_LINEAR_FAST_PATH_TEST_SURFACE_ENABLED
#define EIEL_TW_LIFECYCLE_LINEAR_FAST_PATH_TEST_SURFACE_ENABLED 0
#endif

#if EIEL_TW_LIFECYCLE_LINEAR_FAST_PATH_TEST_SURFACE_ENABLED
namespace eiel::cuda_architecture_b::flattened_controller_v1 {
torch::Tensor test_only_lifecycle_death_cull_cell_v3(
    const std::vector<torch::Tensor>& state,
    int64_t slot_count,
    int64_t active_capacity,
    int64_t agent_id_capacity,
    int64_t grid_height,
    int64_t grid_width,
    int64_t pending_capacity
);
}  // namespace eiel::cuda_architecture_b::flattened_controller_v1
#endif

#if EIEL_CUDA_ARCHITECTURE_B_EXACT_CPU_POLICY_CONTEXT_ALLOWED
#define EIEL_WDL_COMPILED_EXACT_CPU_POLICY_CONTEXT_ALLOWED_V1 1
#else
#define EIEL_WDL_COMPILED_EXACT_CPU_POLICY_CONTEXT_ALLOWED_V1 0
#endif

#pragma push_macro("PYBIND11_MODULE")
#pragma push_macro("EIEL_CUDA_ARCHITECTURE_B_OBSERVATION_CHANNEL_COUNT")
#pragma push_macro("EIEL_CUDA_ARCHITECTURE_B_OBSERVATION_EXTRA_COLUMNS")
#pragma push_macro("EIEL_CUDA_ARCHITECTURE_B_CACHE_WIDTH_VALID")
#pragma push_macro("EIEL_CUDA_ARCHITECTURE_B_EXACT_CPU_POLICY_CONTEXT_ALLOWED")
#pragma push_macro("EIEL_CUDA_ARCHITECTURE_B_T160_PARTIAL_SPAN_ALLOWED")
#undef PYBIND11_MODULE
#define PYBIND11_MODULE(name, variable) \
    void bind_flattened_controller_v1_for_wdl_v2(pybind11::module_& variable)
#define EIEL_CUDA_ARCHITECTURE_B_OBSERVATION_CHANNEL_COUNT \
    (eiel::cuda_architecture_b::flattened_wdl_controller_v2:: \
             current_flat_wdl_context_v2() \
         .enabled \
     ? 9 \
     : (patch_p2_full_mode ? 8 : 1))
#define EIEL_CUDA_ARCHITECTURE_B_OBSERVATION_EXTRA_COLUMNS \
    (eiel::cuda_architecture_b::flattened_gmnw_controller_v1:: \
             current_flat_gmnw_context_v1() \
         .enabled \
     ? 12 \
     : 0)
#define EIEL_CUDA_ARCHITECTURE_B_CACHE_WIDTH_VALID \
    ((eiel::cuda_architecture_b::flattened_gmnw_controller_v1:: \
              current_flat_gmnw_context_v1() \
          .enabled || \
      eiel::cuda_architecture_b::flattened_modular_controller_v1:: \
              current_flat_modular_context_v1() \
          .enabled) \
     ? cache_width == 1 \
     : cache_width == 2)
#ifdef EIEL_WDL_JUNGLE_EED_HAPB_FIXED_MLP_NATIVE_BINDING_V1
#define EIEL_CUDA_ARCHITECTURE_B_HAPB_DIRECT_POLICY_CONTEXT_ALLOWED \
    (eiel::cuda_architecture_b::hapb_policy_context_v1:: \
        hapb_direct_route_context_active_v1())
#else
#define EIEL_CUDA_ARCHITECTURE_B_HAPB_DIRECT_POLICY_CONTEXT_ALLOWED false
#endif
#define EIEL_CUDA_ARCHITECTURE_B_EXACT_CPU_POLICY_CONTEXT_ALLOWED \
    (EIEL_WDL_COMPILED_EXACT_CPU_POLICY_CONTEXT_ALLOWED_V1 && \
     (EIEL_CUDA_ARCHITECTURE_B_HAPB_DIRECT_POLICY_CONTEXT_ALLOWED || \
      !eiel::cuda_architecture_b::flattened_jungle_controller_v4:: \
             current_flat_jungle_context_v4() \
         .enabled \
     && !eiel::cuda_architecture_b::flattened_gru_controller_v3:: \
              current_flat_gru_context_v3() \
          .enabled \
     && !eiel::cuda_architecture_b::flattened_gmnw_controller_v1:: \
              current_flat_gmnw_context_v1() \
          .enabled \
     && !eiel::cuda_architecture_b::flattened_modular_controller_v1:: \
              current_flat_modular_context_v1() \
          .enabled))
#define EIEL_CUDA_ARCHITECTURE_B_T160_PARTIAL_SPAN_ALLOWED \
    (eiel::cuda_architecture_b::flattened_wdl_controller_v2:: \
         current_flat_wdl_context_v2().enabled)
#include "cuda_architecture_b_flattened_controller_v1_binding.cpp"
#pragma pop_macro("EIEL_CUDA_ARCHITECTURE_B_T160_PARTIAL_SPAN_ALLOWED")
#pragma pop_macro("EIEL_CUDA_ARCHITECTURE_B_EXACT_CPU_POLICY_CONTEXT_ALLOWED")
#undef EIEL_CUDA_ARCHITECTURE_B_HAPB_DIRECT_POLICY_CONTEXT_ALLOWED
#pragma pop_macro("EIEL_CUDA_ARCHITECTURE_B_CACHE_WIDTH_VALID")
#pragma pop_macro("EIEL_CUDA_ARCHITECTURE_B_OBSERVATION_EXTRA_COLUMNS")
#pragma pop_macro("EIEL_CUDA_ARCHITECTURE_B_OBSERVATION_CHANNEL_COUNT")
#pragma pop_macro("PYBIND11_MODULE")
#undef EIEL_WDL_COMPILED_EXACT_CPU_POLICY_CONTEXT_ALLOWED_V1

#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace {

namespace wdl_controller =
    eiel::cuda_architecture_b::flattened_wdl_controller_v2;
namespace gru_controller =
    eiel::cuda_architecture_b::flattened_gru_controller_v3;
namespace gmnw_controller =
    eiel::cuda_architecture_b::flattened_gmnw_controller_v1;
namespace jungle_controller =
    eiel::cuda_architecture_b::flattened_jungle_controller_v4;
namespace modular_controller =
    eiel::cuda_architecture_b::flattened_modular_controller_v1;
#ifndef EIEL_NON_EED_WDL_JUNGLE_ONLY
namespace eed_controller = eiel::eed_resource_impact_native_v1;
#endif

constexpr std::size_t kDirectResidentPreparedArgumentCountV1 = 172;
constexpr std::size_t kDirectResidentDirectTrajectoryArgumentCountV1 =
    kDirectResidentPreparedArgumentCountV1 + 5;
constexpr std::size_t kDirectResidentPhysicalTrajectoryArgumentCountV1 =
    kDirectResidentPreparedArgumentCountV1;
// The Python preparation contract and the physical C++ ABI both place
// patch_p2_config at slot 79.  Keep the sink attestation at this exact owner
// boundary; do not add a second event-drain implementation here.
constexpr std::size_t kPatchP2ConfigArgumentIndexV1 = 79;
// The shared direct-resident ABI places ticks_per_generation immediately
// after patch_p2_config.  The tick-span wrapper must report the value that
// actually crossed the native boundary, rather than relying on the full
// generation controller's receipt.
constexpr std::size_t kTicksArgumentIndexV1 = 80;
constexpr std::size_t kRuntimeBookkeepingArgumentIndexV2 = 51;
constexpr const char* kResidentEventSinkStorageModeV1 =
    "eiel_resident_event_sink_v1";
constexpr const char* kResidentEventSinkAttestationSchemaV1 =
    "eiel_native_event_sink_attestation_v1";
constexpr const char* kResidentEventSinkAttestationStageV1 =
    "resident_window_controller_v1";

std::optional<std::string> authenticated_event_sink_path_v1(
    const pybind11::args& arguments,
    const bool required,
    const char* route
) {
    const char* stage = "abi_argument_count";
    TORCH_CHECK(
        pybind11::len(arguments) > kPatchP2ConfigArgumentIndexV1,
        "native event sink binding failed; route=",
        route,
        " last_attested_stage=",
        stage,
        " patch_p2_config ABI slot is absent"
    );
    stage = "abi_patch_p2_config";
    const auto patch_config_handle = arguments[kPatchP2ConfigArgumentIndexV1];
    TORCH_CHECK(
        pybind11::isinstance<pybind11::dict>(patch_config_handle),
        "native event sink binding failed; route=",
        route,
        " last_attested_stage=",
        stage,
        " patch_p2_config is not a mapping"
    );
    const pybind11::dict patch_config =
        pybind11::reinterpret_borrow<pybind11::dict>(patch_config_handle);
    const pybind11::str sink_key("event_history_sink_path");
    if (!patch_config.contains(sink_key)) {
        TORCH_CHECK(
            !required,
            "native event sink binding failed; route=",
            route,
            " last_attested_stage=",
            stage,
            " event_history_sink_path is absent from the physical ABI"
        );
        return std::nullopt;
    }
    stage = "abi_event_history_sink_path";
    const auto sink_handle = patch_config[sink_key];
    TORCH_CHECK(
        pybind11::isinstance<pybind11::str>(sink_handle),
        "native event sink binding failed; route=",
        route,
        " last_attested_stage=",
        stage,
        " event_history_sink_path is not a string"
    );
    const auto sink_path = pybind11::cast<std::string>(sink_handle);
    TORCH_CHECK(
        !sink_path.empty(),
        "native event sink binding failed; route=",
        route,
        " last_attested_stage=",
        stage,
        " event_history_sink_path is empty"
    );
    return sink_path;
}

pybind11::dict attest_resident_event_sink_result_v1(
    pybind11::dict result,
    const std::optional<std::string>& sink_path,
    const bool required,
    const char* route
) {
    if (!sink_path.has_value()) {
        TORCH_CHECK(
            !required,
            "native event sink attestation failed; route=",
            route,
            " last_attested_stage=abi_event_history_sink_path missing"
        );
        return result;
    }
    const char* stage = "resident_window_controller_result";
    const pybind11::str storage_mode_key("event_history_storage_mode");
    TORCH_CHECK(
        result.contains(storage_mode_key)
            && pybind11::isinstance<pybind11::str>(result[storage_mode_key]),
        "native event sink attestation failed; route=",
        route,
        " last_attested_stage=",
        stage,
        " storage mode is absent or malformed"
    );
    TORCH_CHECK(
        pybind11::cast<std::string>(result[storage_mode_key])
            == kResidentEventSinkStorageModeV1,
        "native event sink attestation failed; route=",
        route,
        " last_attested_stage=",
        stage,
        " resident controller did not return the external sink mode"
    );
    TORCH_CHECK(
        !result.contains(pybind11::str("event_histories")),
        "native event sink attestation failed; route=",
        route,
        " last_attested_stage=",
        stage,
        " inline cumulative event_histories were returned"
    );
    const pybind11::str reference_key("event_history_reference");
    TORCH_CHECK(
        result.contains(reference_key)
            && pybind11::isinstance<pybind11::dict>(result[reference_key]),
        "native event sink attestation failed; route=",
        route,
        " last_attested_stage=",
        stage,
        " canonical sink reference is absent or malformed"
    );
    const pybind11::dict reference =
        pybind11::reinterpret_borrow<pybind11::dict>(result[reference_key]);
    const pybind11::str reference_path_key("path");
    TORCH_CHECK(
        reference.contains(reference_path_key)
            && pybind11::isinstance<pybind11::str>(reference[reference_path_key])
            && pybind11::cast<std::string>(reference[reference_path_key])
                == *sink_path,
        "native event sink attestation failed; route=",
        route,
        " last_attested_stage=",
        stage,
        " canonical sink reference differs from the authenticated ABI path"
    );
    for (const char* key : {
             "event_drain_receipts",
             "event_history_frontier_by_cell",
             "event_pending_rows_by_cell",
         }) {
        const pybind11::str field(key);
        TORCH_CHECK(
            result.contains(field)
                && pybind11::isinstance<pybind11::list>(result[field]),
            "native event sink attestation failed; route=",
            route,
            " last_attested_stage=",
            stage,
            " required resident sink field is absent or malformed: ",
            key
        );
    }
    const pybind11::str materialized_key("host_event_materialized_bytes");
    TORCH_CHECK(
        result.contains(materialized_key)
            && pybind11::isinstance<pybind11::int_>(result[materialized_key])
            && pybind11::cast<int64_t>(result[materialized_key]) == 0,
        "native event sink attestation failed; route=",
        route,
        " last_attested_stage=",
        stage,
        " host_event_materialized_bytes is nonzero or absent"
    );
    result["native_event_sink_attestation_schema_version"] =
        kResidentEventSinkAttestationSchemaV1;
    result["native_event_sink_last_attested_stage"] =
        kResidentEventSinkAttestationStageV1;
    result["native_event_sink_abi_argument_index"] =
        static_cast<int64_t>(kPatchP2ConfigArgumentIndexV1);
    result["native_event_sink_path"] = *sink_path;
    return result;
}

class WdlContextGuardV2 final {
  public:
    explicit WdlContextGuardV2(
        const wdl_controller::FlatWdlContextV2& context
    ) {
        wdl_controller::install_flat_wdl_context_v2(context);
    }

    ~WdlContextGuardV2() {
        wdl_controller::clear_flat_wdl_context_v2();
    }

    WdlContextGuardV2(const WdlContextGuardV2&) = delete;
    WdlContextGuardV2& operator=(const WdlContextGuardV2&) = delete;
};

class GruContextGuardV3 final {
  public:
    explicit GruContextGuardV3(
        const gru_controller::FlatGruContextV3& context
    ) {
        gru_controller::install_flat_gru_context_v3(context);
    }

    ~GruContextGuardV3() {
        gru_controller::clear_flat_gru_context_v3();
    }

    GruContextGuardV3(const GruContextGuardV3&) = delete;
    GruContextGuardV3& operator=(const GruContextGuardV3&) = delete;
};

class GmnwContextGuardV1 final {
  public:
    explicit GmnwContextGuardV1(
        const gmnw_controller::FlatGmnwContextV1& context
    ) {
        gmnw_controller::install_flat_gmnw_context_v1(context);
    }

    ~GmnwContextGuardV1() {
        gmnw_controller::clear_flat_gmnw_context_v1();
    }

    GmnwContextGuardV1(const GmnwContextGuardV1&) = delete;
    GmnwContextGuardV1& operator=(const GmnwContextGuardV1&) = delete;
};

class JungleContextGuardV4 final {
  public:
    explicit JungleContextGuardV4(
        const jungle_controller::FlatJungleContextV4& context
    ) {
        jungle_controller::install_flat_jungle_context_v4(context);
    }

    ~JungleContextGuardV4() {
        jungle_controller::clear_flat_jungle_context_v4();
    }

    JungleContextGuardV4(const JungleContextGuardV4&) = delete;
    JungleContextGuardV4& operator=(const JungleContextGuardV4&) = delete;
};

class ModularContextGuardV1 final {
  public:
    explicit ModularContextGuardV1(
        const modular_controller::FlatModularContextV1& context
    ) {
        modular_controller::install_flat_modular_context_v1(context);
    }

    ~ModularContextGuardV1() {
        modular_controller::clear_flat_modular_context_v1();
    }

    ModularContextGuardV1(const ModularContextGuardV1&) = delete;
    ModularContextGuardV1& operator=(const ModularContextGuardV1&) = delete;
};

#ifndef EIEL_NON_EED_WDL_JUNGLE_ONLY
class EedContextGuardV1 final {
  public:
    explicit EedContextGuardV1(
        const eed_controller::ResidentContextV1& context
    ) {
        eed_controller::install_resident_context_v1(context);
    }

    ~EedContextGuardV1() {
        eed_controller::clear_resident_context_v1();
    }

    EedContextGuardV1(const EedContextGuardV1&) = delete;
    EedContextGuardV1& operator=(const EedContextGuardV1&) = delete;
};
#endif

torch::Tensor exact_tensor(
    const pybind11::list& tensors,
    const std::size_t index,
    const char* const name
) {
    TORCH_CHECK(index < tensors.size(), "missing WDL tensor: ", name);
    auto tensor = pybind11::cast<torch::Tensor>(tensors[index]);
    TORCH_CHECK(tensor.is_cuda(), "WDL tensor ", name, " must be CUDA resident");
    TORCH_CHECK(tensor.is_contiguous(), "WDL tensor ", name, " must be contiguous");
    return tensor;
}

torch::Tensor exact_config_tensor(
    const pybind11::dict& config,
    const char* const name
) {
    const pybind11::str key(name);
    TORCH_CHECK(config.contains(key), "WDL config is missing ", name);
    auto tensor = pybind11::cast<torch::Tensor>(config[key]);
    TORCH_CHECK(tensor.is_cuda(), "WDL config tensor ", name, " must be CUDA resident");
    TORCH_CHECK(tensor.is_contiguous(), "WDL config tensor ", name, " must be contiguous");
    return tensor;
}

void same_device(
    const torch::Tensor& reference,
    const torch::Tensor& tensor,
    const char* const name
) {
    TORCH_CHECK(
        tensor.device() == reference.device(),
        "WDL tensor ",
        name,
        " must share one CUDA device"
    );
}

std::int64_t exact_int(
    const pybind11::dict& config,
    const char* const name
) {
    const pybind11::str key(name);
    TORCH_CHECK(config.contains(key), "WDL config is missing ", name);
    const auto value = config[key];
    TORCH_CHECK(
        pybind11::isinstance<pybind11::int_>(value)
            && !pybind11::isinstance<pybind11::bool_>(value),
        "WDL config ",
        name,
        " must be an exact integer"
    );
    return pybind11::cast<std::int64_t>(value);
}

std::int32_t exact_int32(
    const pybind11::dict& config,
    const char* const name
) {
    const std::int64_t value = exact_int(config, name);
    TORCH_CHECK(
        value >= std::numeric_limits<std::int32_t>::min()
            && value <= std::numeric_limits<std::int32_t>::max(),
        "WDL config ",
        name,
        " is outside int32 range"
    );
    return static_cast<std::int32_t>(value);
}

bool exact_bool(const pybind11::dict& config, const char* const name) {
    const pybind11::str key(name);
    TORCH_CHECK(config.contains(key), "WDL config is missing ", name);
    TORCH_CHECK(
        pybind11::isinstance<pybind11::bool_>(config[key]),
        "WDL config ",
        name,
        " must be an exact bool"
    );
    return pybind11::cast<bool>(config[key]);
}

float exact_float(const pybind11::dict& config, const char* const name) {
    const pybind11::str key(name);
    TORCH_CHECK(config.contains(key), "WDL config is missing ", name);
    TORCH_CHECK(
        pybind11::isinstance<pybind11::float_>(config[key]),
        "WDL config ",
        name,
        " must be an exact float"
    );
    const double value = pybind11::cast<double>(config[key]);
    TORCH_CHECK(std::isfinite(value), "WDL config ", name, " must be finite");
    return static_cast<float>(value);
}

double exact_double(const pybind11::dict& config, const char* const name) {
    const pybind11::str key(name);
    TORCH_CHECK(config.contains(key), "GRU config is missing ", name);
    TORCH_CHECK(
        pybind11::isinstance<pybind11::float_>(config[key]),
        "GRU config ",
        name,
        " must be an exact float"
    );
    const double value = pybind11::cast<double>(config[key]);
    TORCH_CHECK(std::isfinite(value), "GRU config ", name, " must be finite");
    return value;
}

modular_controller::FlatModularContextV1 validated_modular_context_v1(
    const pybind11::list& tensors,
    const pybind11::dict& config
) {
    using c10::ScalarType;
    constexpr std::array<ScalarType, modular_controller::kPlaneCountV1> dtypes{{
        torch::kBool, torch::kInt64, torch::kInt64, torch::kInt32, torch::kInt32,
        torch::kBool, torch::kInt32, torch::kInt32, torch::kInt32, torch::kInt32,
        torch::kInt8, torch::kInt8, torch::kFloat32, torch::kInt32, torch::kInt32,
        torch::kInt8, torch::kFloat32, torch::kFloat32, torch::kBool, torch::kBool,
        torch::kFloat32, torch::kBool, torch::kFloat32, torch::kInt8, torch::kInt8,
        torch::kInt32, torch::kFloat32, torch::kBool, torch::kInt8, torch::kInt8,
        torch::kInt32, torch::kInt8, torch::kFloat32, torch::kBool, torch::kBool,
        torch::kFloat32, torch::kInt8, torch::kInt32, torch::kInt32, torch::kBool,
        torch::kBool, torch::kBool, torch::kBool, torch::kFloat32, torch::kBool,
        torch::kInt32, torch::kInt32, torch::kInt32, torch::kInt32, torch::kInt32,
        torch::kInt32, torch::kInt32, torch::kFloat32, torch::kInt32, torch::kInt64,
        torch::kFloat64, torch::kFloat64,
    }};
    const pybind11::str lifecycle_key("lifecycle_enabled");
    const bool lifecycle_enabled = config.contains(lifecycle_key)
        ? exact_bool(config, "lifecycle_enabled") : false;
    const std::int64_t expected_tensor_count = lifecycle_enabled
        ? 2 * modular_controller::kPlaneCountV1
        : modular_controller::kPlaneCountV1;
    TORCH_CHECK(
        tensors.size() == expected_tensor_count,
        "modular execution requires exactly ",
        expected_tensor_count,
        " ordered source/shadow tensors"
    );
    const auto cells = exact_int(config, "cells");
    const auto capacity = exact_int(config, "capacity");
    const auto mature_start_step = exact_int32(config, "mature_start_step");
    const auto late_start_step = exact_int32(config, "late_start_step");
    TORCH_CHECK(
        exact_bool(config, "enabled") && cells > 0 && capacity > 0
            && mature_start_step == 20 && late_start_step == 40,
        "modular v1 configuration is outside the bounded production contract"
    );

    std::array<torch::Tensor, modular_controller::kPlaneCountV1> values;
    std::array<torch::Tensor, modular_controller::kPlaneCountV1> shadows;
    for (std::size_t index = 0; index < values.size(); ++index) {
        values[index] = pybind11::cast<torch::Tensor>(tensors[index]);
        TORCH_CHECK(
            values[index].is_cuda() && values[index].is_contiguous()
                && !values[index].requires_grad(),
            "modular tensor ", index, " must be contiguous no-grad CUDA storage"
        );
        TORCH_CHECK(
            values[index].scalar_type() == dtypes[index],
            "modular tensor ", index, " has the wrong dtype"
        );
        TORCH_CHECK(
            values[index].dim() >= 2 && values[index].size(0) == cells
                && values[index].size(1) == capacity,
            "modular tensor ", index, " has the wrong cell/capacity prefix"
        );
        TORCH_CHECK(
            values[index].device() == values[0].device(),
            "all modular tensors must share one CUDA device"
        );
        if (lifecycle_enabled) {
            shadows[index] = pybind11::cast<torch::Tensor>(
                tensors[index + modular_controller::kPlaneCountV1]
            );
            TORCH_CHECK(
                shadows[index].is_cuda() && shadows[index].is_contiguous()
                    && !shadows[index].requires_grad()
                    && shadows[index].scalar_type() == dtypes[index]
                    && shadows[index].sizes() == values[index].sizes()
                    && shadows[index].device() == values[0].device(),
                "modular shadow tensor ", index,
                " must exactly match its source layout"
            );
        }
    }
    const auto expect_shape = [&](
        const std::size_t index,
        const std::initializer_list<std::int64_t> suffix
    ) {
        std::vector<std::int64_t> expected{cells, capacity};
        expected.insert(expected.end(), suffix.begin(), suffix.end());
        TORCH_CHECK(
            values[index].sizes() == torch::IntArrayRef(expected),
            "modular tensor ", index, " has the wrong fixed-envelope shape"
        );
    };
    for (const std::size_t index : {0U, 1U, 2U, 3U, 4U, 56U}) {
        expect_shape(index, {});
    }
    for (const std::size_t index : {
             5U, 6U, 7U, 8U, 9U, 10U, 11U, 13U, 14U, 15U, 18U, 19U,
             21U, 23U, 28U, 34U, 36U, 39U, 40U, 41U, 42U, 43U,
         }) {
        expect_shape(index, {modular_controller::kModuleSlotsV1});
    }
    for (const std::size_t index : {12U, 24U, 25U, 26U, 27U, 29U, 30U, 31U, 32U, 33U, 37U, 38U}) {
        expect_shape(
            index,
            {modular_controller::kModuleSlotsV1,
             index == 12U ? 2 : modular_controller::kBindingsPerModuleV1}
        );
    }
    expect_shape(16, {modular_controller::kModuleSlotsV1, 4});
    expect_shape(17, {modular_controller::kModuleSlotsV1, 10});
    expect_shape(20, {modular_controller::kModuleSlotsV1, 1});
    expect_shape(22, {modular_controller::kModuleSlotsV1, 1});
    expect_shape(35, {modular_controller::kModuleSlotsV1, 3});
    for (std::size_t index = 44; index <= 53; ++index) {
        expect_shape(index, {modular_controller::kConnectionSlotsV1});
    }
    expect_shape(54, {10});
    expect_shape(55, {10});

    modular_controller::FlatModularTensorTableV2 lifecycle_table{};
    torch::Tensor cold_origin_slots;
    torch::Tensor cold_origin_slots_shadow;
    torch::Tensor action_residual_trace;
    std::int64_t action_residual_trace_ticks = 0;
    torch::Tensor turnover_mutation_draws;
    torch::Tensor duplicate_descendant_mask;
    torch::Tensor duplicate_descendant_mask_shadow;
    torch::Tensor turnover_mutation_receipt_rows;
    torch::Tensor turnover_mutation_receipt_counts;
    torch::Tensor turnover_mutation_receipt_overflow;
    torch::Tensor ordinary_birth_rowv1_receipt_rows;
    torch::Tensor ordinary_birth_rowv1_receipt_counts;
    torch::Tensor ordinary_birth_rowv1_receipt_overflow;
    std::int64_t turnover_mutation_first_generation = 0;
    if (lifecycle_enabled) {
        const pybind11::str pointer_table_key("lifecycle_pointer_table");
        TORCH_CHECK(
            config.contains(pointer_table_key),
            "modular lifecycle pointer table is missing"
        );
        const auto pointer_table = pybind11::cast<torch::Tensor>(
            config[pointer_table_key]
        );
        TORCH_CHECK(
            pointer_table.is_cuda() && pointer_table.is_contiguous()
                && !pointer_table.requires_grad()
                && pointer_table.scalar_type() == torch::kInt64
                && pointer_table.sizes() == torch::IntArrayRef({
                    modular_controller::kPlaneCountV1,
                    static_cast<std::int64_t>(5),
                })
                && pointer_table.device() == values[0].device(),
            "modular lifecycle pointer table must be int64[57,5] on the source device"
        );
        constexpr std::array<std::int32_t, 23> negative_fill_planes{{
            1, 2, 6, 7, 8, 9, 13, 14, 24, 25, 29, 30, 31,
            37, 38, 45, 46, 47, 48, 49, 50, 51, 53,
        }};
        std::array<
            modular_controller::ModularTensorTripletV2,
            modular_controller::kPlaneCountV1
        > entries{};
        for (std::size_t index = 0; index < entries.size(); ++index) {
            TORCH_CHECK(
                values[index].numel() % (cells * capacity) == 0,
                "modular lifecycle plane ", index,
                " does not divide into fixed agent rows"
            );
            const std::int64_t bytes_per_row =
                values[index].numel() / (cells * capacity)
                * static_cast<std::int64_t>(values[index].element_size());
            const bool negative_fill = std::find(
                negative_fill_planes.begin(), negative_fill_planes.end(),
                static_cast<std::int32_t>(index)
            ) != negative_fill_planes.end();
            entries[index] = {
                static_cast<std::uint8_t*>(values[index].data_ptr()),
                static_cast<std::uint8_t*>(shadows[index].data_ptr()),
                capacity * bytes_per_row,
                bytes_per_row,
                negative_fill ? 0xff : 0,
            };
        }
        C10_CUDA_CHECK(cudaMemcpy(
            pointer_table.data_ptr<std::int64_t>(),
            entries.data(),
            sizeof(entries),
            cudaMemcpyHostToDevice
        ));
        lifecycle_table.tensors = reinterpret_cast<
            const modular_controller::ModularTensorTripletV2*
        >(pointer_table.data_ptr<std::int64_t>());
        cold_origin_slots = pybind11::cast<torch::Tensor>(
            config[pybind11::str("cold_origin_slots")]
        );
        cold_origin_slots_shadow = pybind11::cast<torch::Tensor>(
            config[pybind11::str("cold_origin_slots_shadow")]
        );
        const auto validate_cold_origin = [&](
            const torch::Tensor& tensor,
            const char* const name
        ) {
            TORCH_CHECK(
                tensor.is_cuda() && tensor.is_contiguous()
                    && !tensor.requires_grad()
                    && tensor.scalar_type() == torch::kInt64
                    && tensor.sizes()
                        == torch::IntArrayRef({cells, capacity})
                    && tensor.device() == values[0].device(),
                "modular lifecycle ", name,
                " must be contiguous int64[cells,capacity] on the source device"
            );
        };
        validate_cold_origin(cold_origin_slots, "cold_origin_slots");
        validate_cold_origin(
            cold_origin_slots_shadow, "cold_origin_slots_shadow"
        );
    }
    const pybind11::str action_trace_key("action_residual_trace");
    if (config.contains(action_trace_key)) {
        action_residual_trace = pybind11::cast<torch::Tensor>(
            config[action_trace_key]
        );
        TORCH_CHECK(
            action_residual_trace.is_cuda()
                && action_residual_trace.is_contiguous()
                && !action_residual_trace.requires_grad()
                && action_residual_trace.scalar_type() == torch::kFloat32
                && action_residual_trace.dim() == 4
                && action_residual_trace.size(0) == cells
                && action_residual_trace.size(1) > 0
                && action_residual_trace.size(2) == capacity
                && action_residual_trace.size(3)
                    == modular_controller::kActionWidthV1
                && action_residual_trace.device() == values[0].device(),
            "modular action residual trace must be contiguous "
            "float32[cells,ticks,capacity,7] on the source device"
        );
        action_residual_trace_ticks = action_residual_trace.size(1);
    }
    const pybind11::str turnover_mutation_key("turnover_mutation_draws");
    const pybind11::str ordinary_birth_mutation_key(
        "ordinary_birth_mutation_draws"
    );
    const bool turnover_mutation_enabled = config.contains(turnover_mutation_key);
    const bool ordinary_birth_mutation_enabled =
        config.contains(ordinary_birth_mutation_key);
    TORCH_CHECK(
        !(turnover_mutation_enabled && ordinary_birth_mutation_enabled),
        "modular turnover and ordinary-birth mutation modes are mutually exclusive"
    );
    const bool modular_mutation_enabled =
        turnover_mutation_enabled || ordinary_birth_mutation_enabled;
    if (modular_mutation_enabled) {
        const auto mutation_draw_key = ordinary_birth_mutation_enabled
            ? ordinary_birth_mutation_key : turnover_mutation_key;
        const char* const receipt_prefix = ordinary_birth_mutation_enabled
            ? "ordinary_birth_mutation" : "turnover_mutation";
        turnover_mutation_draws = pybind11::cast<torch::Tensor>(
            config[mutation_draw_key]
        );
        duplicate_descendant_mask = pybind11::cast<torch::Tensor>(
            config[pybind11::str("duplicate_descendant_mask")]
        );
        duplicate_descendant_mask_shadow = pybind11::cast<torch::Tensor>(
            config[pybind11::str("duplicate_descendant_mask_shadow")]
        );
        turnover_mutation_receipt_rows = pybind11::cast<torch::Tensor>(
            config[pybind11::str(
                std::string(receipt_prefix) + "_receipt_rows"
            )]
        );
        turnover_mutation_receipt_counts = pybind11::cast<torch::Tensor>(
            config[pybind11::str(
                std::string(receipt_prefix) + "_receipt_counts"
            )]
        );
        turnover_mutation_receipt_overflow = pybind11::cast<torch::Tensor>(
            config[pybind11::str(
                std::string(receipt_prefix) + "_receipt_overflow"
            )]
        );
        if (ordinary_birth_mutation_enabled) {
            ordinary_birth_rowv1_receipt_rows = pybind11::cast<torch::Tensor>(
                config[pybind11::str("ordinary_birth_rowv1_receipt_rows")]
            );
            ordinary_birth_rowv1_receipt_counts = pybind11::cast<torch::Tensor>(
                config[pybind11::str("ordinary_birth_rowv1_receipt_counts")]
            );
            ordinary_birth_rowv1_receipt_overflow = pybind11::cast<torch::Tensor>(
                config[pybind11::str("ordinary_birth_rowv1_receipt_overflow")]
            );
        }
        turnover_mutation_first_generation = exact_int(
            config,
            ordinary_birth_mutation_enabled
                ? "ordinary_birth_mutation_first_generation"
                : "turnover_mutation_first_generation"
        );
        TORCH_CHECK(
            turnover_mutation_first_generation
                >= (ordinary_birth_mutation_enabled ? 0 : 1),
            "modular mutation first generation is outside the supported domain"
        );
        const auto same_cuda_tensor = [&](const torch::Tensor& tensor) {
            return tensor.is_cuda() && tensor.is_contiguous()
                && !tensor.requires_grad() && tensor.device() == values[0].device();
        };
        TORCH_CHECK(
            lifecycle_enabled && same_cuda_tensor(turnover_mutation_draws)
                && turnover_mutation_draws.scalar_type() == torch::kFloat64
                && turnover_mutation_draws.sizes() == torch::IntArrayRef({
                    cells,
                    modular_controller::kTurnoverMutationSequenceCapacityV1,
                    2,
                }),
            "modular mutation draws must be contiguous "
            "float64[cells,capacity,2] CUDA storage"
        );
        TORCH_CHECK(
            same_cuda_tensor(duplicate_descendant_mask)
                && same_cuda_tensor(duplicate_descendant_mask_shadow)
                && duplicate_descendant_mask.scalar_type() == torch::kBool
                && duplicate_descendant_mask_shadow.scalar_type() == torch::kBool
                && duplicate_descendant_mask.sizes()
                    == torch::IntArrayRef({
                        cells, capacity, modular_controller::kModuleSlotsV1
                    })
                && duplicate_descendant_mask_shadow.sizes()
                    == duplicate_descendant_mask.sizes(),
            "modular duplicate-descendant masks must be matching contiguous "
            "bool[cells,capacity,16] CUDA storage"
        );
        TORCH_CHECK(
            same_cuda_tensor(turnover_mutation_receipt_rows)
                && turnover_mutation_receipt_rows.scalar_type() == torch::kInt64
                && turnover_mutation_receipt_rows.sizes()
                    == torch::IntArrayRef({
                        cells,
                        modular_controller::kTurnoverMutationSequenceCapacityV1,
                        modular_controller::kTurnoverMutationReceiptWidthV1,
                    })
                && same_cuda_tensor(turnover_mutation_receipt_counts)
                && turnover_mutation_receipt_counts.scalar_type() == torch::kInt64
                && turnover_mutation_receipt_counts.sizes()
                    == torch::IntArrayRef({cells})
                && same_cuda_tensor(turnover_mutation_receipt_overflow)
                && turnover_mutation_receipt_overflow.scalar_type()
                    == torch::kInt32
                && turnover_mutation_receipt_overflow.sizes()
                    == torch::IntArrayRef({cells}),
            "modular mutation receipts have the wrong CUDA layout"
        );
        if (ordinary_birth_mutation_enabled) {
            TORCH_CHECK(
                same_cuda_tensor(ordinary_birth_rowv1_receipt_rows)
                    && ordinary_birth_rowv1_receipt_rows.scalar_type()
                        == torch::kInt64
                    && ordinary_birth_rowv1_receipt_rows.sizes()
                        == torch::IntArrayRef({
                            cells,
                            modular_controller::kTurnoverMutationSequenceCapacityV1,
                            modular_controller::kOrdinaryBirthRowV1ReceiptWidthV3,
                        })
                    && same_cuda_tensor(ordinary_birth_rowv1_receipt_counts)
                    && ordinary_birth_rowv1_receipt_counts.scalar_type()
                        == torch::kInt64
                    && ordinary_birth_rowv1_receipt_counts.sizes()
                        == torch::IntArrayRef({cells})
                    && same_cuda_tensor(ordinary_birth_rowv1_receipt_overflow)
                    && ordinary_birth_rowv1_receipt_overflow.scalar_type()
                        == torch::kInt32
                    && ordinary_birth_rowv1_receipt_overflow.sizes()
                        == torch::IntArrayRef({cells}),
                "ordinary-birth RowV1 v3 receipts have the wrong CUDA layout"
            );
        }
    }

    return {
        true,
        lifecycle_enabled,
        lifecycle_table,
        lifecycle_enabled
            ? cold_origin_slots.data_ptr<std::int64_t>() : nullptr,
        lifecycle_enabled
            ? cold_origin_slots_shadow.data_ptr<std::int64_t>() : nullptr,
        values[0].data_ptr<bool>(),
        values[1].data_ptr<std::int64_t>(),
        values[2].data_ptr<std::int64_t>(),
        values[5].data_ptr<bool>(),
        values[8].data_ptr<std::int32_t>(),
        values[10].data_ptr<std::int8_t>(),
        values[13].data_ptr<std::int32_t>(),
        values[14].data_ptr<std::int32_t>(),
        values[15].data_ptr<std::int8_t>(),
        values[16].data_ptr<float>(),
        values[17].data_ptr<float>(),
        values[18].data_ptr<bool>(),
        values[19].data_ptr<bool>(),
        values[20].data_ptr<float>(),
        values[21].data_ptr<bool>(),
        values[22].data_ptr<float>(),
        values[23].data_ptr<std::int8_t>(),
        values[24].data_ptr<std::int8_t>(),
        values[26].data_ptr<float>(),
        values[27].data_ptr<bool>(),
        values[28].data_ptr<std::int8_t>(),
        values[31].data_ptr<std::int8_t>(),
        values[32].data_ptr<float>(),
        values[33].data_ptr<bool>(),
        values[35].data_ptr<float>(),
        values[36].data_ptr<std::int8_t>(),
        values[37].data_ptr<std::int32_t>(),
        values[38].data_ptr<std::int32_t>(),
        values[39].data_ptr<bool>(),
        values[40].data_ptr<bool>(),
        values[41].data_ptr<bool>(),
        values[42].data_ptr<bool>(),
        values[43].data_ptr<float>(),
        values[44].data_ptr<bool>(),
        values[46].data_ptr<std::int32_t>(),
        values[47].data_ptr<std::int32_t>(),
        values[52].data_ptr<float>(),
        values[53].data_ptr<std::int32_t>(),
        action_residual_trace.defined()
            ? action_residual_trace.data_ptr<float>() : nullptr,
        action_residual_trace_ticks,
        turnover_mutation_enabled,
        ordinary_birth_mutation_enabled,
        modular_mutation_enabled
            ? turnover_mutation_draws.data_ptr<double>() : nullptr,
        modular_mutation_enabled
            ? duplicate_descendant_mask.data_ptr<bool>() : nullptr,
        modular_mutation_enabled
            ? duplicate_descendant_mask_shadow.data_ptr<bool>() : nullptr,
        modular_mutation_enabled
            ? turnover_mutation_receipt_rows.data_ptr<std::int64_t>() : nullptr,
        modular_mutation_enabled
            ? turnover_mutation_receipt_counts.data_ptr<std::int64_t>() : nullptr,
        modular_mutation_enabled
            ? turnover_mutation_receipt_overflow.data_ptr<std::int32_t>() : nullptr,
        ordinary_birth_mutation_enabled
            ? ordinary_birth_rowv1_receipt_rows.data_ptr<std::int64_t>() : nullptr,
        ordinary_birth_mutation_enabled
            ? ordinary_birth_rowv1_receipt_counts.data_ptr<std::int64_t>() : nullptr,
        ordinary_birth_mutation_enabled
            ? ordinary_birth_rowv1_receipt_overflow.data_ptr<std::int32_t>() : nullptr,
        turnover_mutation_first_generation,
        cells,
        capacity,
        mature_start_step,
        late_start_step,
    };
}

jungle_controller::FlatJungleContextV4 validated_jungle_context_v4(
    const pybind11::list& state_tensors,
    const pybind11::list& reporting_tensors,
    const pybind11::dict& config
) {
    TORCH_CHECK(
        state_tensors.size() == 2,
        "Jungle v4 requires recovery-activity and visibility state tensors"
    );
    TORCH_CHECK(
        reporting_tensors.size() == 5,
        "Jungle v4 requires scratch, recovery stats, integer metrics, "
        "double metrics, and predator multiplier tape tensors"
    );
    const auto recovery = exact_tensor(
        state_tensors, 0, "Jungle recovery activity"
    );
    const auto visibility = exact_tensor(
        state_tensors, 1, "Jungle visibility"
    );
    TORCH_CHECK(
        recovery.scalar_type() == torch::kFloat32
            && recovery.dim() == 3 && recovery.is_contiguous(),
        "Jungle recovery activity must be contiguous float32[cells,H,W]"
    );
    TORCH_CHECK(
        visibility.scalar_type() == torch::kFloat32
            && visibility.sizes() == recovery.sizes()
            && visibility.is_contiguous(),
        "Jungle visibility must match recovery activity"
    );
    const auto scratch = exact_tensor(reporting_tensors, 0, "Jungle scratch");
    const auto recovery_stats = exact_tensor(
        reporting_tensors, 1, "Jungle recovery stats"
    );
    const auto integer_metrics = exact_tensor(
        reporting_tensors, 2, "Jungle integer metrics"
    );
    const auto double_metrics = exact_tensor(
        reporting_tensors, 3, "Jungle double metrics"
    );
    const auto multiplier_tape = exact_tensor(
        reporting_tensors, 4, "Jungle predator multiplier tape"
    );
    const auto cells = recovery.size(0);
    const auto height = recovery.size(1);
    const auto width = recovery.size(2);
    TORCH_CHECK(
        cells > 0 && height > 0 && width > 0,
        "Jungle state dimensions must be positive"
    );
    TORCH_CHECK(
        scratch.scalar_type() == torch::kFloat32
            && scratch.sizes() == torch::IntArrayRef({cells, 2, height, width}),
        "Jungle scratch must be float32[cells,2,H,W]"
    );
    TORCH_CHECK(
        recovery_stats.scalar_type() == torch::kFloat64
            && recovery_stats.dim() == 3
            && recovery_stats.size(0) == cells
            && recovery_stats.size(2)
                == eiel::cuda_architecture_b::persistent_patch_p2::
                    kRecoveryStatCount,
        "Jungle recovery stats must be float64[cells,ticks,7]"
    );
    const auto stats_ticks = recovery_stats.size(1);
    TORCH_CHECK(stats_ticks > 0, "Jungle reporting tick capacity must be positive");
    const auto ticks_per_generation = exact_int(config, "ticks_per_generation");
    const auto generation_base = exact_int(config, "generation_base");
    TORCH_CHECK(
        ticks_per_generation > 0 && stats_ticks % ticks_per_generation == 0,
        "Jungle reporting capacity must contain complete generation rows"
    );
    const auto generation_rows = stats_ticks / ticks_per_generation;
    TORCH_CHECK(
        integer_metrics.scalar_type() == torch::kInt64
            && integer_metrics.sizes() == torch::IntArrayRef({
                cells,
                stats_ticks,
                jungle_controller::kTickIntegerMetricCountV4,
            }),
        "Jungle integer metrics must be int64[cells,ticks,9]"
    );
    TORCH_CHECK(
        double_metrics.scalar_type() == torch::kFloat64
            && double_metrics.sizes() == torch::IntArrayRef({
                cells,
                stats_ticks,
                jungle_controller::kTickDoubleMetricCountV4,
            }),
        "Jungle double metrics must be float64[cells,ticks,7]"
    );
    TORCH_CHECK(
        multiplier_tape.scalar_type() == torch::kFloat32
            && multiplier_tape.dim() == 3
            && multiplier_tape.size(0) == cells
            && multiplier_tape.size(1) == stats_ticks
            && multiplier_tape.size(2) > 0,
        "Jungle multiplier tape must be float32[cells,ticks,capacity]"
    );
    for (const auto& tensor : {
             visibility,
             scratch,
             recovery_stats,
             integer_metrics,
             double_metrics,
             multiplier_tape,
         }) {
        same_device(recovery, tensor, "Jungle context tensor");
    }

    jungle_controller::FlatJungleConfigV4 parsed{
        exact_bool(config, "enabled"),
        exact_bool(config, "predator_pressure_enabled"),
        exact_bool(config, "juvenile_vulnerability_enabled"),
        exact_bool(config, "terrain_protection_enabled"),
        exact_bool(config, "terrain_protection_apply_to_damage_enabled"),
        exact_bool(config, "birth_emissions_enabled"),
        exact_bool(config, "birth_emission_clamp_enabled"),
        exact_bool(config, "ecological_recovery_enabled"),
        exact_bool(config, "visibility_from_cover_enabled"),
        exact_bool(config, "recovery_regrowth_boost_enabled"),
        exact_int32(config, "juvenile_age_threshold"),
        exact_float(config, "scent_max"),
        exact_float(config, "waste_max"),
        exact_float(config, "density_denominator"),
        exact_float(config, "visibility_min"),
        exact_float(config, "visibility_max"),
        exact_float(config, "visibility_default"),
        exact_float(config, "predator_base_risk"),
        exact_float(config, "predator_scent_weight"),
        exact_float(config, "predator_waste_weight"),
        exact_float(config, "predator_density_weight"),
        exact_float(config, "predator_visibility_weight"),
        exact_float(config, "predator_cover_protection_weight"),
        exact_float(config, "predator_shelter_protection_weight"),
        exact_float(config, "predator_risk_min"),
        exact_float(config, "predator_risk_cap"),
        exact_double(config, "predator_health_damage"),
        exact_double(config, "juvenile_damage_multiplier"),
        exact_float(config, "cover_damage_reduction_weight"),
        exact_float(config, "shelter_damage_reduction_weight"),
        exact_float(config, "terrain_damage_multiplier_min"),
        exact_double(config, "birth_scent_emit"),
        exact_double(config, "birth_waste_emit"),
        exact_float(config, "recovery_activity_min"),
        exact_float(config, "recovery_activity_max"),
        exact_float(config, "fertility_min"),
        exact_float(config, "fertility_max"),
        exact_float(config, "recovery_decay_factor"),
        exact_float(config, "recovery_base_growth"),
        exact_float(config, "recovery_cover_weight"),
        exact_float(config, "recovery_fertility_weight"),
        exact_float(config, "recovery_waste_penalty_weight"),
        exact_float(config, "recovery_density_penalty_weight"),
        exact_float(config, "recovery_scent_penalty_weight"),
        exact_float(config, "recovery_regrowth_boost_scale"),
        exact_float(config, "recovery_regrowth_boost_max"),
    };
    TORCH_CHECK(parsed.enabled, "Jungle v4 context must be explicitly enabled");
    TORCH_CHECK(
        parsed.juvenile_age_threshold >= 0
            && parsed.scent_max >= 0.0f && parsed.waste_max >= 0.0f
            && parsed.density_denominator > 0.0f
            && parsed.visibility_min <= parsed.visibility_max
            && parsed.predator_risk_min >= 0.0f
            && parsed.predator_risk_cap <= 1.0f
            && parsed.predator_risk_min <= parsed.predator_risk_cap
            && parsed.predator_health_damage >= 0.0
            && parsed.juvenile_damage_multiplier >= 1.0
            && parsed.terrain_damage_multiplier_min >= 0.0f
            && parsed.terrain_damage_multiplier_min <= 1.0f
            && parsed.birth_scent_emit >= 0.0f
            && parsed.birth_waste_emit >= 0.0f
            && parsed.recovery_activity_min <= parsed.recovery_activity_max
            && parsed.fertility_min <= parsed.fertility_max
            && parsed.recovery_decay_factor >= 0.0f
            && parsed.recovery_decay_factor <= 1.0f
            && parsed.recovery_regrowth_boost_scale >= 0.0f
            && parsed.recovery_regrowth_boost_max >= 0.0f,
        "Jungle v4 config is outside the accepted finite envelope"
    );
    const auto optional_nonnegative_float64_by_cell = [&](const char* name) {
        if (!config.contains(pybind11::str(name))) {
            return torch::Tensor();
        }
        const auto tensor = exact_config_tensor(config, name);
        TORCH_CHECK(
            tensor.scalar_type() == torch::kFloat64
                && tensor.sizes() == torch::IntArrayRef({cells})
                && tensor.is_contiguous()
                && torch::isfinite(tensor).all().item<bool>()
                && !torch::any(tensor < 0.0).item<bool>(),
            name,
            " must be contiguous finite nonnegative float64[cells]"
        );
        same_device(recovery, tensor, name);
        return tensor;
    };
    const auto predator_base_risk_by_cell =
        optional_nonnegative_float64_by_cell("predator_base_risk_by_cell");
    const auto predator_health_damage_by_cell =
        optional_nonnegative_float64_by_cell("predator_health_damage_by_cell");
    const auto predator_risk_cap_by_cell =
        optional_nonnegative_float64_by_cell("predator_risk_cap_by_cell");
    const auto recovery_base_growth_by_cell =
        optional_nonnegative_float64_by_cell("recovery_base_growth_by_cell");
    const auto recovery_decay_factor_by_cell =
        optional_nonnegative_float64_by_cell("recovery_decay_factor_by_cell");
    const auto recovery_fertility_weight_by_cell =
        optional_nonnegative_float64_by_cell("recovery_fertility_weight_by_cell");
    const auto recovery_waste_penalty_weight_by_cell =
        optional_nonnegative_float64_by_cell(
            "recovery_waste_penalty_weight_by_cell"
        );
    const int per_cell_experiment_vector_count =
        static_cast<int>(predator_base_risk_by_cell.defined())
        + static_cast<int>(predator_health_damage_by_cell.defined())
        + static_cast<int>(predator_risk_cap_by_cell.defined())
        + static_cast<int>(recovery_base_growth_by_cell.defined())
        + static_cast<int>(recovery_decay_factor_by_cell.defined())
        + static_cast<int>(recovery_fertility_weight_by_cell.defined())
        + static_cast<int>(recovery_waste_penalty_weight_by_cell.defined());
    TORCH_CHECK(
        per_cell_experiment_vector_count == 0
            || per_cell_experiment_vector_count == 7,
        "Jungle experiment vectors must be supplied as one complete cell-ordered set"
    );
    TORCH_CHECK(
        (!predator_risk_cap_by_cell.defined()
            || torch::all(predator_risk_cap_by_cell <= 1.0).item<bool>())
            && (!recovery_decay_factor_by_cell.defined()
                || torch::all(recovery_decay_factor_by_cell <= 1.0).item<bool>()),
        "per-cell Jungle risk cap and recovery decay factor must be in [0,1]"
    );
    return {
        true,
        parsed,
        recovery.data_ptr<float>(),
        visibility.data_ptr<float>(),
        scratch.data_ptr<float>(),
        recovery_stats.data_ptr<double>(),
        integer_metrics.data_ptr<std::int64_t>(),
        double_metrics.data_ptr<double>(),
        multiplier_tape.data_ptr<float>(),
        ticks_per_generation,
        generation_rows,
        generation_base,
        multiplier_tape.size(2),
        predator_base_risk_by_cell.defined()
            ? predator_base_risk_by_cell.data_ptr<double>() : nullptr,
        predator_health_damage_by_cell.defined()
            ? predator_health_damage_by_cell.data_ptr<double>() : nullptr,
        predator_risk_cap_by_cell.defined()
            ? predator_risk_cap_by_cell.data_ptr<double>() : nullptr,
        recovery_base_growth_by_cell.defined()
            ? recovery_base_growth_by_cell.data_ptr<double>() : nullptr,
        recovery_decay_factor_by_cell.defined()
            ? recovery_decay_factor_by_cell.data_ptr<double>() : nullptr,
        recovery_fertility_weight_by_cell.defined()
            ? recovery_fertility_weight_by_cell.data_ptr<double>() : nullptr,
        recovery_waste_penalty_weight_by_cell.defined()
            ? recovery_waste_penalty_weight_by_cell.data_ptr<double>() : nullptr,
    };
}

wdl_controller::FlatWdlContextV2 validated_wdl_context_v2(
    const pybind11::list& state_tensors,
    const pybind11::list& schedule_tensors,
    const pybind11::list& reporting_tensors,
    const pybind11::dict& config
) {
    TORCH_CHECK(state_tensors.size() == 7, "WDL v3 requires seven state tensors");
    TORCH_CHECK(schedule_tensors.size() == 2, "WDL v3 requires two schedule tensors");
    TORCH_CHECK(reporting_tensors.size() == 10, "WDL v3 requires ten reporting tensors");

    const auto cue = exact_tensor(state_tensors, 0, "cue");
    const auto timer = exact_tensor(state_tensors, 1, "timer");
    const auto active = exact_tensor(state_tensors, 2, "active");
    const auto intensity = exact_tensor(state_tensors, 3, "intensity");
    const auto age = exact_tensor(state_tensors, 4, "age");
    const auto event_id = exact_tensor(state_tensors, 5, "event_id");
    const auto prior_cue = exact_tensor(state_tensors, 6, "prior_cue");
    TORCH_CHECK(
        cue.scalar_type() == torch::kFloat32 && cue.dim() == 3,
        "WDL cue must be float32[cells,height,width]"
    );
    const auto grid_shape = cue.sizes();
    for (const auto& item : {
             std::pair{timer, torch::kInt32},
             std::pair{active, torch::kFloat32},
             std::pair{intensity, torch::kFloat32},
             std::pair{age, torch::kInt32},
             std::pair{event_id, torch::kInt64},
             std::pair{prior_cue, torch::kInt8},
         }) {
        TORCH_CHECK(
            item.first.scalar_type() == item.second
                && item.first.sizes() == grid_shape,
            "all seven WDL state tensors must share [cells,height,width]"
        );
        same_device(cue, item.first, "state");
    }
    const std::int64_t cells = cue.size(0);
    TORCH_CHECK(cells > 0 && cue.size(1) > 0 && cue.size(2) > 0, "WDL grid is empty");

    const auto radius_by_cell = exact_config_tensor(config, "radius_by_cell");
    TORCH_CHECK(
        radius_by_cell.scalar_type() == torch::kInt32
            && radius_by_cell.sizes() == torch::IntArrayRef({cells}),
        "WDL radius_by_cell must be int32[cells]"
    );
    same_device(cue, radius_by_cell, "radius_by_cell");

    const auto base_seed = exact_tensor(schedule_tensors, 0, "base_seed");
    const auto next_step = exact_tensor(schedule_tensors, 1, "next_absolute_step");
    for (const auto& tensor : {base_seed, next_step}) {
        TORCH_CHECK(
            tensor.scalar_type() == torch::kInt64
                && tensor.sizes() == torch::IntArrayRef({cells}),
            "WDL schedule tensors must be int64[cells]"
        );
        same_device(cue, tensor, "schedule");
    }

    const auto tick_integers = exact_tensor(reporting_tensors, 0, "tick_integers");
    const auto tick_delay_sum = exact_tensor(reporting_tensors, 1, "tick_delay_sum");
    const auto event_integers = exact_tensor(reporting_tensors, 2, "event_integers");
    const auto event_intensity = exact_tensor(reporting_tensors, 3, "event_intensity");
    const auto status_by_tick = exact_tensor(reporting_tensors, 4, "status_by_tick");
    const auto generation_metrics = exact_tensor(reporting_tensors, 5, "generation_metrics");
    const auto agent_history_keys = exact_tensor(
        reporting_tensors, 6, "agent_history_keys"
    );
    const auto agent_history_flags = exact_tensor(
        reporting_tensors, 7, "agent_history_flags"
    );
    const auto lineage_history_keys = exact_tensor(
        reporting_tensors, 8, "lineage_history_keys"
    );
    const auto lineage_history_damage = exact_tensor(
        reporting_tensors, 9, "lineage_history_damage"
    );
    for (const auto& tensor : {
             tick_integers,
             tick_delay_sum,
             event_integers,
             event_intensity,
             status_by_tick,
             generation_metrics,
             agent_history_keys,
             agent_history_flags,
             lineage_history_keys,
             lineage_history_damage,
         }) {
        same_device(cue, tensor, "reporting");
    }
    TORCH_CHECK(
        tick_integers.scalar_type() == torch::kInt64
            && tick_integers.dim() == 4
            && tick_integers.size(0) == cells
            && tick_integers.size(3)
                == wdl_controller::kWdlTickIntegerMetricCountV2,
        "WDL tick integers must be int64[cells,generations,ticks,6]"
    );
    const std::int64_t generations = tick_integers.size(1);
    const std::int64_t ticks = tick_integers.size(2);
    TORCH_CHECK(generations > 0 && ticks > 0, "WDL reporting window is empty");
    TORCH_CHECK(
        tick_delay_sum.scalar_type() == torch::kFloat64
            && tick_delay_sum.sizes()
                == torch::IntArrayRef({cells, generations, ticks}),
        "WDL tick delay sum has an invalid shape or dtype"
    );
    TORCH_CHECK(
        event_integers.scalar_type() == torch::kInt64
            && event_integers.sizes() == torch::IntArrayRef(
                {
                    cells,
                    generations,
                    ticks,
                    wdl_controller::kWdlScheduledEventIntegerCountV2,
                }
            ),
        "WDL scheduled event integers have an invalid shape or dtype"
    );
    TORCH_CHECK(
        event_intensity.scalar_type() == torch::kFloat32
            && event_intensity.sizes()
                == torch::IntArrayRef({cells, generations, ticks}),
        "WDL scheduled event intensity has an invalid shape or dtype"
    );
    TORCH_CHECK(
        status_by_tick.scalar_type() == torch::kInt32
            && status_by_tick.sizes()
                == torch::IntArrayRef({cells, generations, ticks}),
        "WDL status-by-tick has an invalid shape or dtype"
    );
    TORCH_CHECK(
        generation_metrics.scalar_type() == torch::kFloat64
            && generation_metrics.sizes() == torch::IntArrayRef(
                {
                    cells,
                    generations,
                    wdl_controller::kWdlGenerationMetricCountV2,
                }
            ),
        "WDL generation metrics have an invalid shape or dtype"
    );
    TORCH_CHECK(
        agent_history_keys.scalar_type() == torch::kInt64
            && agent_history_keys.dim() == 3
            && agent_history_keys.size(0) == cells
            && agent_history_keys.size(1) == generations,
        "WDL agent history keys must be int64[cells,generations,history_capacity]"
    );
    const std::int64_t history_capacity = agent_history_keys.size(2);
    TORCH_CHECK(history_capacity > 0, "WDL history capacity must be positive");
    TORCH_CHECK(
        agent_history_flags.scalar_type() == torch::kUInt8
            && agent_history_flags.sizes()
                == torch::IntArrayRef({cells, generations, history_capacity}),
        "WDL agent history flags have an invalid shape or dtype"
    );
    TORCH_CHECK(
        lineage_history_keys.scalar_type() == torch::kInt64
            && lineage_history_keys.sizes()
                == torch::IntArrayRef({cells, generations, history_capacity}),
        "WDL lineage history keys have an invalid shape or dtype"
    );
    TORCH_CHECK(
        lineage_history_damage.scalar_type() == torch::kFloat64
            && lineage_history_damage.sizes()
                == torch::IntArrayRef({cells, generations, history_capacity}),
        "WDL lineage history damage has an invalid shape or dtype"
    );

    const bool enabled = exact_bool(config, "enabled");
    TORCH_CHECK(enabled, "WDL v3 entrypoint requires enabled=true");
    const std::int64_t history_abi_version =
        exact_int(config, "history_abi_version");
    const std::int64_t slot_capacity = exact_int(config, "slot_capacity");
    const std::int64_t max_births_per_tick =
        exact_int(config, "max_births_per_tick");
    const std::int64_t ticks_per_generation =
        exact_int(config, "ticks_per_generation");
    TORCH_CHECK(
        history_abi_version == wdl_controller::kAbiVersionV3,
        "WDL compact generation-history ABI must be v3"
    );
    TORCH_CHECK(
        slot_capacity > 0 && max_births_per_tick >= 0
            && ticks_per_generation == ticks
            && max_births_per_tick
                <= (INT64_MAX - slot_capacity) / ticks_per_generation
            && history_capacity
                == slot_capacity
                    + ticks_per_generation * max_births_per_tick
            && exact_int(config, "history_capacity") == history_capacity,
        "WDL compact generation-history capacity formula mismatch"
    );
    return {
        true,
        {
            exact_int32(config, "schema_version"),
            exact_int32(config, "delay_steps"),
            exact_int32(config, "cue_duration_steps"),
            exact_int32(config, "active_duration_steps"),
            exact_int32(config, "event_period_steps"),
            exact_int32(config, "radius"),
            exact_float(config, "intensity"),
            exact_float(config, "observation_dropout_rate"),
            static_cast<std::uint8_t>(enabled),
            static_cast<std::uint8_t>(exact_bool(config, "cue_enabled")),
            static_cast<std::uint8_t>(exact_bool(config, "observation_enabled")),
            static_cast<std::uint8_t>(exact_bool(config, "damage_enabled")),
            static_cast<std::uint8_t>(
                exact_bool(config, "deterministic_schedule_enabled")
            ),
        },
        radius_by_cell.data_ptr<std::int32_t>(),
        cue.data_ptr<float>(),
        timer.data_ptr<std::int32_t>(),
        active.data_ptr<float>(),
        intensity.data_ptr<float>(),
        age.data_ptr<std::int32_t>(),
        event_id.data_ptr<std::int64_t>(),
        prior_cue.data_ptr<std::int8_t>(),
        base_seed.data_ptr<std::int64_t>(),
        next_step.data_ptr<std::int64_t>(),
        tick_integers.data_ptr<std::int64_t>(),
        tick_delay_sum.data_ptr<double>(),
        event_integers.data_ptr<std::int64_t>(),
        event_intensity.data_ptr<float>(),
        status_by_tick.data_ptr<std::int32_t>(),
        generation_metrics.data_ptr<double>(),
        agent_history_keys.data_ptr<std::int64_t>(),
        agent_history_flags.data_ptr<std::uint8_t>(),
        lineage_history_keys.data_ptr<std::int64_t>(),
        lineage_history_damage.data_ptr<double>(),
        ticks,
        generations,
        exact_int(config, "generation_base"),
        history_capacity,
    };
}

gru_controller::FlatGruContextV3 validated_gru_context_v3(
    const pybind11::list& tensors,
    const pybind11::dict& config
) {
    TORCH_CHECK(tensors.size() == 3, "GRU v3 requires pack, scratch, and status tensors");
    const auto pack = exact_tensor(tensors, 0, "gru_pack");
    const auto scratch = exact_tensor(tensors, 1, "gru_scratch");
    const auto status = exact_tensor(tensors, 2, "gru_status");
    const pybind11::str shared_key("immutable_shared_policy");
    const bool immutable_shared_policy = config.contains(shared_key)
        ? exact_bool(config, "immutable_shared_policy")
        : false;
    TORCH_CHECK(
        pack.scalar_type() == torch::kFloat32
            && (pack.dim() == 3 || (immutable_shared_policy && pack.dim() == 1)),
        "GRU pack must be dense float32[cells,capacity,packed_width] or "
        "frozen shared float32[packed_width]"
    );
    TORCH_CHECK(
        scratch.scalar_type() == torch::kFloat32 && scratch.dim() == 3
            && scratch.size(2) == gru_controller::kStableScratchWidthV3,
        "GRU scratch must be float32[cells,capacity,519]"
    );
    const std::int64_t cells = scratch.size(0);
    const std::int64_t capacity = scratch.size(1);
    const std::int64_t packed_width = immutable_shared_policy
        ? pack.size(0)
        : pack.size(2);
    TORCH_CHECK(
        cells > 0 && capacity > 0 && packed_width > 0,
        "GRU pack extents must be positive"
    );
    TORCH_CHECK(
        scratch.sizes() == torch::IntArrayRef(
            {cells, capacity, gru_controller::kStableScratchWidthV3}
        ),
        "GRU scratch must be float32[cells,capacity,519]"
    );
    TORCH_CHECK(
        immutable_shared_policy
            || (pack.size(0) == cells && pack.size(1) == capacity),
        "dense GRU pack extents must match scratch"
    );
    TORCH_CHECK(
        status.scalar_type() == torch::kInt32
            && status.sizes() == torch::IntArrayRef({cells}),
        "GRU status must be int32[cells]"
    );
    same_device(pack, scratch, "gru_scratch");
    same_device(pack, status, "gru_status");
    const double upkeep_cost_base = exact_double(config, "upkeep_cost_base");
    const double upkeep_cost_per_dim = exact_double(
        config, "upkeep_cost_per_dim"
    );
    TORCH_CHECK(
        upkeep_cost_base >= 0.0 && upkeep_cost_per_dim >= 0.0,
        "GRU upkeep costs must be nonnegative"
    );
    return {
        true,
        immutable_shared_policy,
        pack.data_ptr<float>(),
        packed_width,
        scratch.data_ptr<float>(),
        status.data_ptr<std::int32_t>(),
        cells,
        capacity,
        upkeep_cost_base,
        upkeep_cost_per_dim,
    };
}

gmnw_controller::FlatGmnwContextV1 validated_gmnw_context_v1(
    const pybind11::list& tensors,
    const pybind11::dict& config
) {
    const pybind11::str control_enabled_key("control_matrix_enabled");
    const pybind11::str trace_enabled_key("qualification_trace_enabled");
    const pybind11::str decision_tape_enabled_key(
        "qualification_decision_tape_enabled"
    );
    const pybind11::str selection_metrics_enabled_key(
        "selection_metrics_enabled"
    );
    const bool has_control_tensors = config.contains(control_enabled_key)
        ? exact_bool(config, "control_matrix_enabled")
        : tensors.size() == 27 || tensors.size() == 34;
    const bool has_trace_tensors = config.contains(trace_enabled_key)
        ? exact_bool(config, "qualification_trace_enabled")
        : false;
    const bool has_decision_tape_tensors = config.contains(
        decision_tape_enabled_key
    ) ? exact_bool(config, "qualification_decision_tape_enabled") : false;
    const bool has_selection_metrics_tensors = config.contains(
        selection_metrics_enabled_key
    ) ? exact_bool(config, "selection_metrics_enabled") : false;
    const std::size_t expected_tensor_count = 23U
        + (has_control_tensors ? 4U : 0U)
        + (has_trace_tensors ? 7U : 0U)
        + (has_decision_tape_tensors ? 3U : 0U)
        + (has_selection_metrics_tensors ? 3U : 0U);
    TORCH_CHECK(
        tensors.size() == expected_tensor_count,
        "GMNW v1 tensor inventory disagrees with configured optional surfaces"
    );
    std::vector<torch::Tensor> values;
    values.reserve(tensors.size());
    for (std::size_t index = 0; index < tensors.size(); ++index) {
        values.push_back(exact_tensor(tensors, index, "gmnw_state"));
    }
    const auto& stream_id = values[0];
    TORCH_CHECK(
        stream_id.scalar_type() == torch::kInt64 && stream_id.dim() == 2,
        "GMNW stream IDs must be int64[cells,capacity]"
    );
    const std::int64_t cells = stream_id.size(0);
    const std::int64_t capacity = stream_id.size(1);
    TORCH_CHECK(cells > 0 && capacity > 0, "GMNW extents must be positive");
    const std::array<std::int64_t, 2> row_shape{cells, capacity};
    const auto rows = torch::IntArrayRef(row_shape);
    const auto require_rows = [&](const std::size_t index, const torch::ScalarType type) {
        TORCH_CHECK(
            values[index].scalar_type() == type && values[index].sizes() == rows,
            "GMNW per-slot tensor ", index, " has an invalid dtype or shape"
        );
        same_device(stream_id, values[index], "gmnw_state");
    };
    require_rows(1, torch::kInt64);
    require_rows(2, torch::kInt8);
    require_rows(3, torch::kInt8);
    require_rows(4, torch::kInt32);
    require_rows(5, torch::kInt32);
    require_rows(6, torch::kBool);
    require_rows(7, torch::kInt8);
    require_rows(8, torch::kInt8);
    require_rows(9, torch::kInt8);
    require_rows(10, torch::kInt8);
    require_rows(11, torch::kInt64);
    require_rows(12, torch::kInt64);
    require_rows(13, torch::kFloat32);

    const std::int64_t generation_rows = exact_int(
        config, "generation_rows_per_cell"
    );
    TORCH_CHECK(generation_rows > 0, "GMNW generation rows must be positive");
    TORCH_CHECK(
        values[14].scalar_type() == torch::kInt64
            && values[14].sizes() == torch::IntArrayRef(
                {cells, generation_rows, gmnw_controller::kCounterColumnsV1}
            ),
        "GMNW counters must be int64[cells,generations,6]"
    );
    TORCH_CHECK(
        values[15].scalar_type() == torch::kFloat64
            && values[15].sizes() == torch::IntArrayRef({cells, generation_rows}),
        "GMNW damage totals must be float64[cells,generations]"
    );
    TORCH_CHECK(
        values[16].scalar_type() == torch::kInt32
            && values[16].sizes() == torch::IntArrayRef({cells}),
        "GMNW status must be int32[cells]"
    );
    TORCH_CHECK(
        values[17].scalar_type() == torch::kInt32
            && values[17].sizes() == torch::IntArrayRef({cells}),
        "GMNW cue alphabet must be int32[cells]"
    );
    TORCH_CHECK(
        values[18].scalar_type() == torch::kInt32
            && values[18].sizes() == torch::IntArrayRef({cells}),
        "GMNW delay ticks must be int32[cells]"
    );
    for (const std::size_t index : {19U, 20U}) {
        TORCH_CHECK(
            values[index].scalar_type() == torch::kBool
                && values[index].sizes() == torch::IntArrayRef({cells}),
            "GMNW boolean cell configuration has an invalid dtype or shape"
        );
    }
    TORCH_CHECK(
        values[21].scalar_type() == torch::kInt64
            && values[21].sizes() == torch::IntArrayRef({cells}),
        "GMNW task seeds must be int64[cells]"
    );
    TORCH_CHECK(
        values[22].scalar_type() == torch::kInt32
            && values[22].sizes() == torch::IntArrayRef({cells}),
        "GMNW VENV consequence profiles must be int32[cells]"
    );
    const std::int32_t minimum_venv_profile =
        values[22].min().item<std::int32_t>();
    const std::int32_t maximum_venv_profile =
        values[22].max().item<std::int32_t>();
    TORCH_CHECK(
        minimum_venv_profile >= gmnw_controller::kVenvConsequenceLegacyV1
            && maximum_venv_profile <= gmnw_controller::kVenvConsequenceP3V1,
        "GMNW VENV consequence profiles contain an unknown code"
    );
    for (std::size_t index = 14; index < values.size(); ++index) {
        same_device(stream_id, values[index], "gmnw_state");
    }

    namespace controls = eiel::cuda_architecture_b::gmnw_control_matrix_v1;
    controls::GmnwControlMatrixContextV1 control_context{};
    if (has_control_tensors) {
        TORCH_CHECK(
            values[23].scalar_type() == torch::kInt8
                && values[23].sizes() == torch::IntArrayRef({cells}),
            "GMNW control policy arms must be int8[cells]"
        );
        TORCH_CHECK(
            values[24].scalar_type() == torch::kFloat32
                && values[24].sizes() == torch::IntArrayRef(
                    {controls::kMatchedMlpPackedWidthV1}
                ),
            "GMNW matched MLP pack has an invalid dtype or width"
        );
        TORCH_CHECK(
            values[25].scalar_type() == torch::kFloat32
                && values[25].sizes() == torch::IntArrayRef(
                    {cells, controls::kMatchedMlpScratchWidthV1}
                ),
            "GMNW matched MLP scratch has an invalid float32 cell layout"
        );
        TORCH_CHECK(
            values[26].scalar_type() == torch::kInt32
                && values[26].sizes() == torch::IntArrayRef({cells}),
            "GMNW control status must be int32[cells]"
        );
        control_context = {
            true,
            values[23].data_ptr<std::int8_t>(),
            values[24].data_ptr<float>(),
            values[25].data_ptr<float>(),
            values[26].data_ptr<std::int32_t>(),
            cells,
        };
    }

    const bool trace_enabled = config.contains(trace_enabled_key)
        ? exact_bool(config, "qualification_trace_enabled")
        : false;
    TORCH_CHECK(
        trace_enabled == has_trace_tensors,
        "GMNW qualification trace enablement and tensor suffix disagree"
    );
    gmnw_controller::GmnwQualificationTraceContextV1 trace_context{};
    if (has_trace_tensors) {
        const std::size_t trace_base = has_control_tensors ? 27U : 23U;
        const std::int64_t trace_ticks = exact_int(
            config, "qualification_trace_output_ticks"
        );
        TORCH_CHECK(
            trace_ticks == 80 || trace_ticks == 160,
            "GMNW qualification trace output ticks must be 80 or 160"
        );
        TORCH_CHECK(
            values[trace_base].scalar_type() == torch::kInt64
                && values[trace_base].sizes()
                    == torch::IntArrayRef({cells, trace_ticks}),
            "GMNW trace active count must be int64[cells,ticks]"
        );
        for (const std::size_t offset : {1U, 2U}) {
            const auto expected_type = offset == 1U ? torch::kInt64 : torch::kInt32;
            TORCH_CHECK(
                values[trace_base + offset].scalar_type() == expected_type
                    && values[trace_base + offset].sizes()
                        == torch::IntArrayRef({cells, trace_ticks, capacity}),
                "GMNW trace active order tensor has an invalid dtype or shape"
            );
        }
        for (const std::size_t offset : {3U, 4U}) {
            TORCH_CHECK(
                values[trace_base + offset].scalar_type() == torch::kFloat32
                    && values[trace_base + offset].sizes()
                        == torch::IntArrayRef({
                            cells,
                            trace_ticks,
                            capacity,
                            gmnw_controller::kQualificationTraceHiddenWidthV1,
                        }),
                "GMNW trace hidden tensor has an invalid dtype or shape"
            );
        }
        for (const std::size_t offset : {5U, 6U}) {
            TORCH_CHECK(
                values[trace_base + offset].scalar_type() == torch::kInt64
                    && values[trace_base + offset].sizes()
                        == torch::IntArrayRef({
                            cells,
                            trace_ticks,
                            capacity,
                            gmnw_controller::kQualificationTraceTaskStateColumnsV1,
                        }),
                "GMNW trace task-state tensor has an invalid dtype or shape"
            );
        }
        trace_context = {
            true,
            trace_ticks,
            values[trace_base].data_ptr<std::int64_t>(),
            values[trace_base + 1].data_ptr<std::int64_t>(),
            values[trace_base + 2].data_ptr<std::int32_t>(),
            values[trace_base + 3].data_ptr<float>(),
            values[trace_base + 4].data_ptr<float>(),
            values[trace_base + 5].data_ptr<std::int64_t>(),
            values[trace_base + 6].data_ptr<std::int64_t>(),
        };
    }

    gmnw_controller::GmnwQualificationDecisionTapeContextV1 decision_tape_context{};
    if (has_decision_tape_tensors) {
        const std::size_t tape_base = 23U
            + (has_control_tensors ? 4U : 0U)
            + (has_trace_tensors ? 7U : 0U);
        TORCH_CHECK(
            values[tape_base].scalar_type() == torch::kInt8
                && values[tape_base].sizes() == torch::IntArrayRef({
                    cells,
                    gmnw_controller::kQualificationDecisionRowsV1,
                }),
            "GMNW qualification decision presence must be int8[cells,3072]"
        );
        TORCH_CHECK(
            values[tape_base + 1].scalar_type() == torch::kInt64
                && values[tape_base + 1].sizes() == torch::IntArrayRef({
                    cells,
                    gmnw_controller::kQualificationDecisionRowsV1,
                    gmnw_controller::kQualificationDecisionColumnsV1,
                }),
            "GMNW qualification decision rows must be int64[cells,3072,6]"
        );
        TORCH_CHECK(
            values[tape_base + 2].scalar_type() == torch::kInt32
                && values[tape_base + 2].sizes() == torch::IntArrayRef({cells}),
            "GMNW qualification decision status must be int32[cells]"
        );
        decision_tape_context = {
            true,
            values[tape_base].data_ptr<std::int8_t>(),
            values[tape_base + 1].data_ptr<std::int64_t>(),
            values[tape_base + 2].data_ptr<std::int32_t>(),
        };
    }

    gmnw_controller::GmnwSelectionMetricsContextV1 selection_metrics_context{};
    if (has_selection_metrics_tensors) {
        const std::size_t metrics_base = 23U
            + (has_control_tensors ? 4U : 0U)
            + (has_trace_tensors ? 7U : 0U)
            + (has_decision_tape_tensors ? 3U : 0U);
        TORCH_CHECK(
            values[metrics_base].scalar_type() == torch::kInt64
                && values[metrics_base].sizes() == torch::IntArrayRef({
                    cells,
                    generation_rows,
                    gmnw_controller::kSelectionPolicyFamilyCountV1,
                    gmnw_controller::kSelectionCountMetricCountV1,
                }),
            "GMNW selection counts must be int64[cells,generations,2,7]"
        );
        TORCH_CHECK(
            values[metrics_base + 1].scalar_type() == torch::kFloat64
                && values[metrics_base + 1].sizes() == torch::IntArrayRef({
                    cells,
                    generation_rows,
                    gmnw_controller::kSelectionPolicyFamilyCountV1,
                }),
            "GMNW selection upkeep totals must be float64[cells,generations,2]"
        );
        TORCH_CHECK(
            values[metrics_base + 2].scalar_type() == torch::kInt32
                && values[metrics_base + 2].sizes()
                    == torch::IntArrayRef({cells}),
            "GMNW selection metrics status must be int32[cells]"
        );
        selection_metrics_context = {
            true,
            values[metrics_base].data_ptr<std::int64_t>(),
            values[metrics_base + 1].data_ptr<double>(),
            values[metrics_base + 2].data_ptr<std::int32_t>(),
        };
    }

    const std::int32_t cue_ticks = exact_int32(config, "cue_ticks");
    const std::int32_t consequence_ticks = exact_int32(
        config, "consequence_ticks"
    );
    const std::int32_t grace_generations = exact_int32(
        config, "grace_generation_count"
    );
    const std::int32_t severe_generations = exact_int32(
        config, "severe_generation_count"
    );
    const double grace_damage = exact_double(config, "grace_damage_fraction");
    const double severe_damage = exact_double(config, "severe_damage_fraction");
    const double lethal_damage = exact_double(config, "lethal_damage_fraction");
    const float grace_scale = exact_float(config, "grace_reproduction_scale");
    const pybind11::str terminal_low_population_key(
        "terminal_low_population_no_reinit"
    );
    const bool terminal_low_population_no_reinit = config.contains(
        terminal_low_population_key
    ) ? exact_bool(config, "terminal_low_population_no_reinit")
      : has_selection_metrics_tensors;
    TORCH_CHECK(
        exact_bool(config, "enabled") && cue_ticks > 0 && consequence_ticks > 0
            && grace_generations > 0 && severe_generations > 0
            && grace_damage >= 0.0F && severe_damage >= 0.0F
            && lethal_damage >= 0.0F && grace_scale >= 0.0F
            && grace_scale <= 1.0F,
        "GMNW configuration is outside the frozen contract"
    );
    return {
        true,
        values[0].data_ptr<std::int64_t>(),
        values[1].data_ptr<std::int64_t>(),
        values[2].data_ptr<std::int8_t>(),
        values[3].data_ptr<std::int8_t>(),
        values[4].data_ptr<std::int32_t>(),
        values[5].data_ptr<std::int32_t>(),
        values[6].data_ptr<bool>(),
        values[7].data_ptr<std::int8_t>(),
        values[8].data_ptr<std::int8_t>(),
        values[9].data_ptr<std::int8_t>(),
        values[10].data_ptr<std::int8_t>(),
        values[11].data_ptr<std::int64_t>(),
        values[12].data_ptr<std::int64_t>(),
        values[13].data_ptr<float>(),
        values[14].data_ptr<std::int64_t>(),
        values[15].data_ptr<double>(),
        values[16].data_ptr<std::int32_t>(),
        values[17].data_ptr<std::int32_t>(),
        values[18].data_ptr<std::int32_t>(),
        values[19].data_ptr<bool>(),
        values[20].data_ptr<bool>(),
        values[21].data_ptr<std::int64_t>(),
        values[22].data_ptr<std::int32_t>(),
        cells,
        capacity,
        exact_int(config, "generation_base"),
        generation_rows,
        cue_ticks,
        consequence_ticks,
        grace_generations,
        severe_generations,
        grace_damage,
        severe_damage,
        lethal_damage,
        grace_scale,
        terminal_low_population_no_reinit,
        trace_context,
        decision_tape_context,
        control_context,
        selection_metrics_context,
    };
}

pybind11::dict execute_direct_resident_gru_v3(
    const pybind11::function& direct_v1,
    const std::shared_ptr<resident_run_control::RunControlV2>& run_control,
    const pybind11::list& gru_tensors,
    const pybind11::dict& gru_config,
    const pybind11::args& arguments
) {
    TORCH_CHECK(
        pybind11::len(arguments) ==
            kDirectResidentDirectTrajectoryArgumentCountV1,
        "CUDA GRU v3 execution requires 172 prepared ABI arguments and five "
        "direct-trajectory arguments"
    );
    const auto context = validated_gru_context_v3(gru_tensors, gru_config);
    const c10::cuda::CUDAGuard device_guard(
        pybind11::cast<torch::Tensor>(gru_tensors[0]).device()
    );
    const GruContextGuardV3 context_guard(context);
    pybind11::dict result = execute_direct_resident_windows_v2(
        direct_v1, run_control, arguments
    );
    result["gru_native_owner"] = true;
    result["gru_native_controller_abi_version"] = gru_controller::kAbiVersionV3;
    result["gru_state_tensors"] = gru_tensors;
    result["gru_python_tick_or_generation_ownership"] = false;
    return result;
}

pybind11::dict execute_direct_resident_gru_physical_v3(
    const pybind11::function& physical_v1,
    const pybind11::list& gru_tensors,
    const pybind11::dict& gru_config,
    const pybind11::args& arguments
) {
    TORCH_CHECK(
        pybind11::len(arguments) ==
            kDirectResidentPhysicalTrajectoryArgumentCountV1,
        "CUDA GRU v3 physical execution requires the active-slot 172-argument "
        "physical trajectory ABI"
    );
    const auto context = validated_gru_context_v3(gru_tensors, gru_config);
    const c10::cuda::CUDAGuard device_guard(
        pybind11::cast<torch::Tensor>(gru_tensors[0]).device()
    );
    const GruContextGuardV3 context_guard(context);
    PyObject* raw_result = PyObject_CallObject(
        physical_v1.ptr(), arguments.ptr()
    );
    if (raw_result == nullptr) {
        throw pybind11::error_already_set();
    }
    const auto result = pybind11::reinterpret_steal<pybind11::object>(raw_result);
    TORCH_CHECK(
        pybind11::isinstance<pybind11::dict>(result),
        "CUDA GRU v3 physical trajectory must return one native result mapping"
    );
    pybind11::dict out = pybind11::cast<pybind11::dict>(result);
    out["gru_native_owner"] = true;
    out["gru_native_controller_abi_version"] = gru_controller::kAbiVersionV3;
    out["gru_physical_gate_entrypoint"] = true;
    out["gru_python_tick_or_generation_ownership"] = false;
    return out;
}

pybind11::dict execute_direct_resident_gmnw_gru_v1(
    const pybind11::function& direct_v1,
    const std::shared_ptr<resident_run_control::RunControlV2>& run_control,
    const pybind11::list& gru_tensors,
    const pybind11::dict& gru_config,
    const pybind11::list& gmnw_tensors,
    const pybind11::dict& gmnw_config,
    const pybind11::args& arguments
) {
    TORCH_CHECK(
        pybind11::len(arguments) ==
            kDirectResidentDirectTrajectoryArgumentCountV1,
        "CUDA GMNW+GRU v1 execution requires 172 prepared ABI arguments and "
        "five direct-trajectory arguments"
    );
    const auto gru_context = validated_gru_context_v3(gru_tensors, gru_config);
    const auto gmnw_context = validated_gmnw_context_v1(
        gmnw_tensors, gmnw_config
    );
    const auto gru_device = pybind11::cast<torch::Tensor>(gru_tensors[0]).device();
    TORCH_CHECK(
        pybind11::cast<torch::Tensor>(gmnw_tensors[0]).device() == gru_device,
        "GMNW and GRU state must share one CUDA device"
    );
    const c10::cuda::CUDAGuard device_guard(gru_device);
    const GruContextGuardV3 gru_guard(gru_context);
    const GmnwContextGuardV1 gmnw_guard(gmnw_context);
    pybind11::dict result = execute_direct_resident_windows_v2(
        direct_v1, run_control, arguments
    );
    result["gru_native_owner"] = true;
    result["gru_native_controller_abi_version"] = gru_controller::kAbiVersionV3;
    result["gru_state_tensors"] = gru_tensors;
    result["gmnw_native_owner"] = true;
    result["gmnw_native_controller_abi_version"] = gmnw_controller::kAbiVersionV2;
    result["gmnw_state_tensors"] = gmnw_tensors;
    if (gmnw_context.selection_metrics.enabled) {
        result["gmnw_selection_metrics_version"] =
            gmnw_controller::kSelectionMetricsVersionV1;
    }
    result["gru_python_tick_or_generation_ownership"] = false;
    result["gmnw_python_tick_or_generation_ownership"] = false;
    return result;
}

pybind11::dict execute_direct_resident_gmnw_gru_physical_v1(
    const pybind11::function& physical_v1,
    const pybind11::list& gru_tensors,
    const pybind11::dict& gru_config,
    const pybind11::list& gmnw_tensors,
    const pybind11::dict& gmnw_config,
    const pybind11::args& arguments
) {
    TORCH_CHECK(
        pybind11::len(arguments) ==
            kDirectResidentPhysicalTrajectoryArgumentCountV1,
        "CUDA GMNW+GRU v1 physical execution requires the active-slot "
        "172-argument trajectory ABI"
    );
    const auto gru_context = validated_gru_context_v3(gru_tensors, gru_config);
    const auto gmnw_context = validated_gmnw_context_v1(
        gmnw_tensors, gmnw_config
    );
    const auto gru_device = pybind11::cast<torch::Tensor>(gru_tensors[0]).device();
    TORCH_CHECK(
        pybind11::cast<torch::Tensor>(gmnw_tensors[0]).device() == gru_device,
        "GMNW and GRU state must share one CUDA device"
    );
    const c10::cuda::CUDAGuard device_guard(gru_device);
    const GruContextGuardV3 gru_guard(gru_context);
    const GmnwContextGuardV1 gmnw_guard(gmnw_context);
    PyObject* raw_result = PyObject_CallObject(
        physical_v1.ptr(), arguments.ptr()
    );
    if (raw_result == nullptr) {
        throw pybind11::error_already_set();
    }
    const auto result = pybind11::reinterpret_steal<pybind11::object>(raw_result);
    TORCH_CHECK(
        pybind11::isinstance<pybind11::dict>(result),
        "CUDA GMNW+GRU v1 physical trajectory returned a non-mapping result"
    );
    pybind11::dict out = pybind11::cast<pybind11::dict>(result);
    out["gru_native_owner"] = true;
    out["gru_native_controller_abi_version"] = gru_controller::kAbiVersionV3;
    out["gmnw_native_owner"] = true;
    out["gmnw_native_controller_abi_version"] = gmnw_controller::kAbiVersionV2;
    if (gmnw_context.selection_metrics.enabled) {
        out["gmnw_selection_metrics_version"] =
            gmnw_controller::kSelectionMetricsVersionV1;
    }
    out["gmnw_gru_physical_gate_entrypoint"] = true;
    out["gru_python_tick_or_generation_ownership"] = false;
    out["gmnw_python_tick_or_generation_ownership"] = false;
    return out;
}

pybind11::dict execute_direct_resident_wdl_gru_v3(
    const pybind11::function& direct_v1,
    const std::shared_ptr<resident_run_control::RunControlV2>& run_control,
    const pybind11::list& gru_tensors,
    const pybind11::dict& gru_config,
    const pybind11::list& state_tensors,
    const pybind11::list& schedule_tensors,
    const pybind11::list& reporting_tensors,
    const pybind11::dict& config,
    const pybind11::args& arguments
) {
    TORCH_CHECK(
        pybind11::len(arguments) ==
            kDirectResidentDirectTrajectoryArgumentCountV1,
        "CUDA WDL+GRU v3 execution requires 172 prepared ABI arguments and "
        "five direct-trajectory arguments"
    );
    const auto gru_context = validated_gru_context_v3(gru_tensors, gru_config);
    const auto wdl_context = validated_wdl_context_v2(
        state_tensors, schedule_tensors, reporting_tensors, config
    );
    const c10::cuda::CUDAGuard device_guard(
        pybind11::cast<torch::Tensor>(gru_tensors[0]).device()
    );
    const GruContextGuardV3 gru_guard(gru_context);
    const WdlContextGuardV2 wdl_guard(wdl_context);
    pybind11::dict result = execute_direct_resident_windows_v2(
        direct_v1, run_control, arguments
    );
    result["gru_native_owner"] = true;
    result["gru_native_controller_abi_version"] = gru_controller::kAbiVersionV3;
    result["gru_state_tensors"] = gru_tensors;
    result["wdl_native_owner"] = true;
    result["wdl_native_controller_abi_version"] = wdl_controller::kAbiVersionV3;
    result["wdl_state_tensors"] = state_tensors;
    result["wdl_schedule_tensors"] = schedule_tensors;
    result["wdl_reporting_tensors"] = reporting_tensors;
    result["gru_python_tick_or_generation_ownership"] = false;
    result["wdl_python_tick_or_generation_ownership"] = false;
    return result;
}

pybind11::dict execute_direct_resident_wdl_gru_physical_v3(
    const pybind11::function& physical_v1,
    const pybind11::list& gru_tensors,
    const pybind11::dict& gru_config,
    const pybind11::list& state_tensors,
    const pybind11::list& schedule_tensors,
    const pybind11::list& reporting_tensors,
    const pybind11::dict& config,
    const pybind11::args& arguments
) {
    TORCH_CHECK(
        pybind11::len(arguments) ==
            kDirectResidentPhysicalTrajectoryArgumentCountV1,
        "CUDA WDL+GRU v3 physical execution requires the active-slot "
        "172-argument trajectory ABI"
    );
    const auto gru_context = validated_gru_context_v3(gru_tensors, gru_config);
    const auto wdl_context = validated_wdl_context_v2(
        state_tensors, schedule_tensors, reporting_tensors, config
    );
    const c10::cuda::CUDAGuard device_guard(
        pybind11::cast<torch::Tensor>(gru_tensors[0]).device()
    );
    const GruContextGuardV3 gru_guard(gru_context);
    const WdlContextGuardV2 wdl_guard(wdl_context);
    PyObject* raw_result = PyObject_CallObject(
        physical_v1.ptr(), arguments.ptr()
    );
    if (raw_result == nullptr) {
        throw pybind11::error_already_set();
    }
    const auto result = pybind11::reinterpret_steal<pybind11::object>(raw_result);
    TORCH_CHECK(
        pybind11::isinstance<pybind11::dict>(result),
        "CUDA WDL+GRU v3 physical trajectory must return one native result mapping"
    );
    pybind11::dict out = pybind11::cast<pybind11::dict>(result);
    out["gru_native_owner"] = true;
    out["gru_native_controller_abi_version"] = gru_controller::kAbiVersionV3;
    out["wdl_native_owner"] = true;
    out["wdl_native_controller_abi_version"] = wdl_controller::kAbiVersionV3;
    out["wdl_gru_physical_gate_entrypoint"] = true;
    out["gru_python_tick_or_generation_ownership"] = false;
    out["wdl_python_tick_or_generation_ownership"] = false;
    return out;
}

pybind11::dict gru_capability_v3() {
    pybind11::dict out;
    out["abi_version"] = gru_controller::kAbiVersionV3;
    out["policy_family"] = "stable_gru";
    out["hidden_width"] = gru_controller::kStableHiddenWidthV3;
    out["action_count"] = gru_controller::kStableActionCountV3;
    out["resident_hidden_state"] = true;
    out["cache_refill_rng_contract"] = "one_gumbel_vector_then_repeat";
    out["native_tick_ownership"] = true;
    out["runtime_jit_required"] = false;
    out["production_qualified"] = false;
    return out;
}

pybind11::dict gmnw_gru_capability_v1() {
    pybind11::dict out;
    out["abi_version"] = gmnw_controller::kAbiVersionV2;
    out["world"] = "GRU_MEMORY_NECESSITY_WORLD_V1";
    out["policy_family"] = "stable_gru";
    out["resident_private_task_state"] = true;
    out["resident_hidden_state"] = true;
    out["per_cell_task_configuration"] = true;
    out["exact_consequence_lifecycle"] = true;
    out["exact_reproduction_opportunity"] = true;
    out["public_world_action_neutralization"] = true;
    out["native_tick_ownership"] = true;
    out["native_generation_ownership"] = true;
    out["runtime_jit_required"] = false;
    out["hidden_fallback"] = false;
    out["production_qualified"] = false;
    out["qualification_trace_version"] =
        gmnw_controller::kQualificationTraceVersionV1;
    out["qualification_decision_tape_version"] =
        gmnw_controller::kQualificationDecisionTapeVersionV1;
    out["selection_metrics_version"] =
        gmnw_controller::kSelectionMetricsVersionV1;
    return out;
}

pybind11::dict execute_direct_resident_wdl_v2(
    const pybind11::function& direct_v1,
    const std::shared_ptr<resident_run_control::RunControlV2>& run_control,
    const pybind11::list& state_tensors,
    const pybind11::list& schedule_tensors,
    const pybind11::list& reporting_tensors,
    const pybind11::dict& config,
    const pybind11::args& arguments
) {
    TORCH_CHECK(
        pybind11::len(arguments) ==
            kDirectResidentDirectTrajectoryArgumentCountV1,
        "CUDA WDL v2 execution requires 172 prepared ABI arguments and five "
        "direct-trajectory arguments"
    );
    const auto context = validated_wdl_context_v2(
        state_tensors, schedule_tensors, reporting_tensors, config
    );
    const c10::cuda::CUDAGuard device_guard(
        pybind11::cast<torch::Tensor>(state_tensors[0]).device()
    );
    const WdlContextGuardV2 context_guard(context);
    pybind11::dict result = execute_direct_resident_windows_v2(
        direct_v1, run_control, arguments
    );
    result["wdl_native_owner"] = true;
    result["wdl_native_controller_abi_version"] =
        wdl_controller::kAbiVersionV3;
    result["wdl_state_tensors"] = state_tensors;
    result["wdl_schedule_tensors"] = schedule_tensors;
    result["wdl_reporting_tensors"] = reporting_tensors;
    result["wdl_python_tick_or_generation_ownership"] = false;
    return result;
}

pybind11::dict execute_direct_resident_wdl_jungle_v1(
    const pybind11::function& direct_v1,
    const std::shared_ptr<resident_run_control::RunControlV2>& run_control,
    const pybind11::list& wdl_state_tensors,
    const pybind11::list& wdl_schedule_tensors,
    const pybind11::list& wdl_reporting_tensors,
    const pybind11::dict& wdl_config,
    const pybind11::list& jungle_state_tensors,
    const pybind11::list& jungle_reporting_tensors,
    const pybind11::dict& jungle_config,
    const pybind11::args& arguments
) {
    TORCH_CHECK(
        pybind11::len(arguments) ==
            kDirectResidentDirectTrajectoryArgumentCountV1,
        "CUDA WDL/Jungle v1 execution requires 172 prepared ABI arguments and "
        "five direct-trajectory arguments"
    );
    const auto wdl_context = validated_wdl_context_v2(
        wdl_state_tensors, wdl_schedule_tensors, wdl_reporting_tensors, wdl_config
    );
    const auto jungle_context = validated_jungle_context_v4(
        jungle_state_tensors, jungle_reporting_tensors, jungle_config
    );
    const auto wdl_device = pybind11::cast<torch::Tensor>(wdl_state_tensors[0]).device();
    const auto jungle_device = pybind11::cast<torch::Tensor>(jungle_state_tensors[0]).device();
    TORCH_CHECK(
        wdl_device == jungle_device,
        "CUDA WDL/Jungle v1 contexts must use one physical CUDA device"
    );
    const c10::cuda::CUDAGuard device_guard(wdl_device);
    const WdlContextGuardV2 wdl_context_guard(wdl_context);
    const JungleContextGuardV4 jungle_context_guard(jungle_context);
    pybind11::dict result = execute_direct_resident_windows_v2(
        direct_v1, run_control, arguments
    );
    result["wdl_jungle_native_owner"] = true;
    result["wdl_native_controller_abi_version"] = wdl_controller::kAbiVersionV3;
    result["jungle_native_controller_abi_version"] = jungle_controller::kAbiVersionV4;
    result["wdl_jungle_python_tick_or_generation_ownership"] = false;
    return result;
}

#ifndef EIEL_NON_EED_WDL_JUNGLE_ONLY
pybind11::dict execute_direct_resident_wdl_jungle_eed_v1(
    const pybind11::function& direct_v1,
    const std::shared_ptr<resident_run_control::RunControlV2>& run_control,
    const pybind11::list& wdl_state_tensors,
    const pybind11::list& wdl_schedule_tensors,
    const pybind11::list& wdl_reporting_tensors,
    const pybind11::dict& wdl_config,
    const pybind11::list& jungle_state_tensors,
    const pybind11::list& jungle_reporting_tensors,
    const pybind11::dict& jungle_config,
    const pybind11::list& eed_state_tensors,
    const pybind11::dict& eed_config,
    const pybind11::args& arguments
) {
    const auto event_sink_path = authenticated_event_sink_path_v1(
        arguments,
        false,
        "wdl_jungle_eed_v1"
    );
    const auto eed_context =
        eed_controller::validated_resident_context_v1(eed_state_tensors, eed_config);
    const auto wdl_device = pybind11::cast<torch::Tensor>(wdl_state_tensors[0]).device();
    const auto eed_device = pybind11::cast<torch::Tensor>(eed_state_tensors[0]).device();
    TORCH_CHECK(
        wdl_device == eed_device,
        "CUDA WDL/Jungle/EED contexts must use one physical CUDA device"
    );
    const EedContextGuardV1 eed_context_guard(eed_context);
    pybind11::dict result = execute_direct_resident_wdl_jungle_v1(
        direct_v1,
        run_control,
        wdl_state_tensors,
        wdl_schedule_tensors,
        wdl_reporting_tensors,
        wdl_config,
        jungle_state_tensors,
        jungle_reporting_tensors,
        jungle_config,
        arguments
    );
    result["eed_native_owner"] = true;
    result["eed_native_controller_abi_version"] =
        eed_controller::kResidentSchemaVersion;
    result["eed_state_tensors"] = eed_state_tensors;
    result["eed_python_tick_or_generation_ownership"] = false;
    return attest_resident_event_sink_result_v1(
        std::move(result),
        event_sink_path,
        false,
        "wdl_jungle_eed_v1"
    );
}

pybind11::dict execute_direct_resident_eed_v1(
    const pybind11::function& direct_v1,
    const std::shared_ptr<resident_run_control::RunControlV2>& run_control,
    const pybind11::list& state_tensors,
    const pybind11::dict& config,
    const pybind11::args& arguments
) {
    TORCH_CHECK(
        pybind11::len(arguments) ==
            kDirectResidentDirectTrajectoryArgumentCountV1,
        "CUDA EED v1 execution requires 172 prepared ABI arguments and five "
        "direct-trajectory arguments"
    );
    const auto context =
        eed_controller::validated_resident_context_v1(state_tensors, config);
    const c10::cuda::CUDAGuard device_guard(
        pybind11::cast<torch::Tensor>(state_tensors[0]).device()
    );
    const EedContextGuardV1 context_guard(context);
    pybind11::dict result = execute_direct_resident_windows_v2(
        direct_v1, run_control, arguments
    );
    result["eed_native_owner"] = true;
    result["eed_native_controller_abi_version"] =
        eed_controller::kResidentSchemaVersion;
    result["eed_state_tensors"] = state_tensors;
    result["eed_python_tick_or_generation_ownership"] = false;
    return result;
}

pybind11::dict execute_direct_resident_wdl_jungle_eed_tick_dispatch_v2(
    const pybind11::function& direct_v1,
    const pybind11::list& wdl_state_tensors,
    const pybind11::list& wdl_schedule_tensors,
    const pybind11::list& wdl_reporting_tensors,
    const pybind11::dict& wdl_config,
    const pybind11::list& jungle_state_tensors,
    const pybind11::list& jungle_reporting_tensors,
    const pybind11::dict& jungle_config,
    const pybind11::list& eed_state_tensors,
    const pybind11::dict& eed_config,
    const pybind11::list& source_windows
) {
    TORCH_CHECK(
        pybind11::len(source_windows) > 0,
        "CUDA WDL/Jungle/EED tick dispatch requires bounded source windows"
    );
    const auto wdl_context = validated_wdl_context_v2(
        wdl_state_tensors, wdl_schedule_tensors, wdl_reporting_tensors, wdl_config
    );
    const auto jungle_context = validated_jungle_context_v4(
        jungle_state_tensors, jungle_reporting_tensors, jungle_config
    );
    const auto eed_context =
        eed_controller::validated_resident_context_v1(eed_state_tensors, eed_config);
    const auto device = pybind11::cast<torch::Tensor>(wdl_state_tensors[0]).device();
    TORCH_CHECK(
        device == pybind11::cast<torch::Tensor>(jungle_state_tensors[0]).device()
            && device == pybind11::cast<torch::Tensor>(eed_state_tensors[0]).device(),
        "CUDA WDL/Jungle/EED tick-span contexts must use one physical device"
    );
    const c10::cuda::CUDAGuard device_guard(device);
    const WdlContextGuardV2 wdl_context_guard(wdl_context);
    const JungleContextGuardV4 jungle_context_guard(jungle_context);
    const EedContextGuardV1 eed_context_guard(eed_context);
    std::optional<std::string> event_sink_path;
    pybind11::dict out;
    pybind11::list event_drain_receipts;
    torch::Tensor runtime_totals;
    torch::Tensor runtime_last;
    int64_t native_tick_iterations = 0;
    int64_t final_host_synchronizations = 0;
    int64_t final_device_to_host_operations = 0;
    int64_t final_stream_synchronization_time_ns = 0;
    int64_t intermediate_device_to_host_operations = 0;
    int64_t intermediate_host_to_device_operations = 0;
    for (const auto source_window : source_windows) {
        TORCH_CHECK(
            pybind11::isinstance<pybind11::tuple>(source_window),
            "CUDA WDL/Jungle/EED source-window arguments must be tuples"
        );
        const auto arguments =
            pybind11::reinterpret_borrow<pybind11::args>(source_window);
        TORCH_CHECK(
            pybind11::len(arguments)
                == kDirectResidentDirectTrajectoryArgumentCountV1,
            "CUDA WDL/Jungle/EED source window requires the exact "
            "172-argument physical trajectory ABI plus five direct arguments"
        );
        const auto window_sink_path = authenticated_event_sink_path_v1(
            arguments,
            true,
            "wdl_jungle_eed_tick_dispatch_v2"
        );
        TORCH_CHECK(window_sink_path.has_value(), "tick dispatch lacks event sink");
        if (!event_sink_path.has_value()) {
            event_sink_path = window_sink_path;
        } else {
            TORCH_CHECK(
                event_sink_path == window_sink_path,
                "tick dispatch source windows bind different event sinks"
            );
        }
        const auto run_control =
            std::make_shared<resident_run_control::RunControlV2>();
        auto window_result = execute_direct_resident_windows_v2(
            direct_v1,
            run_control,
            arguments
        );
        const int64_t window_ticks =
            pybind11::cast<int64_t>(arguments[kTicksArgumentIndexV1]);
        TORCH_CHECK(window_ticks > 0, "tick dispatch window is empty");
        TORCH_CHECK(
            native_tick_iterations
                <= std::numeric_limits<int64_t>::max() - window_ticks,
            "tick dispatch iteration count overflow"
        );
        native_tick_iterations += window_ticks;
        const auto runtime = pybind11::cast<torch::Tensor>(
            arguments[kRuntimeBookkeepingArgumentIndexV2]
        );
        TORCH_CHECK(
            runtime.dim() == 3 && runtime.size(1) == window_ticks
                && runtime.size(2) == 5,
            "tick dispatch runtime bookkeeping must be int64[B,ticks,5]"
        );
        auto window_totals = runtime.sum(1);
        runtime_totals = runtime_totals.defined()
            ? runtime_totals + window_totals
            : window_totals.clone();
        runtime_last = runtime.select(1, window_ticks - 1).clone();
        const auto drains = pybind11::cast<pybind11::list>(
            window_result["event_drain_receipts"]
        );
        for (const auto receipt : drains) {
            event_drain_receipts.append(receipt);
        }
        const auto accumulate = [&window_result](
                                    const char* name,
                                    int64_t& total
                                ) {
            const int64_t value = pybind11::cast<int64_t>(window_result[name]);
            TORCH_CHECK(
                value >= 0 && total <= std::numeric_limits<int64_t>::max() - value,
                "tick dispatch telemetry overflow: ", name
            );
            total += value;
        };
        accumulate("final_host_synchronizations", final_host_synchronizations);
        accumulate(
            "final_device_to_host_operations", final_device_to_host_operations
        );
        accumulate(
            "final_stream_synchronization_time_ns",
            final_stream_synchronization_time_ns
        );
        accumulate(
            "intermediate_device_to_host_operations",
            intermediate_device_to_host_operations
        );
        accumulate(
            "intermediate_host_to_device_operations",
            intermediate_host_to_device_operations
        );
        out = std::move(window_result);
    }
    TORCH_CHECK(event_sink_path.has_value(), "tick dispatch has no event sink");
    out["event_drain_receipts"] = std::move(event_drain_receipts);
    out["runtime_bookkeeping_totals"] = std::move(runtime_totals);
    out["runtime_bookkeeping_last"] = std::move(runtime_last);
    out["final_host_synchronizations"] = final_host_synchronizations;
    out["final_device_to_host_operations"] = final_device_to_host_operations;
    out["final_stream_synchronization_time_ns"] =
        final_stream_synchronization_time_ns;
    out["intermediate_device_to_host_operations"] =
        intermediate_device_to_host_operations;
    out["intermediate_host_to_device_operations"] =
        intermediate_host_to_device_operations;
    out["wdl_jungle_native_owner"] = true;
    out["wdl_native_controller_abi_version"] = wdl_controller::kAbiVersionV3;
    out["jungle_native_controller_abi_version"] = jungle_controller::kAbiVersionV4;
    out["eed_native_owner"] = true;
    out["eed_native_controller_abi_version"] = eed_controller::kResidentSchemaVersion;
    out["absolute_tick_span_native_owner"] = true;
    out["native_tick_iterations"] = native_tick_iterations;
    out["source_window_count"] = pybind11::len(source_windows);
    out["wdl_jungle_python_tick_or_generation_ownership"] = false;
    return attest_resident_event_sink_result_v1(
        std::move(out),
        event_sink_path,
        true,
        "wdl_jungle_eed_tick_dispatch_v2"
    );
}

#endif

pybind11::dict execute_direct_resident_modular_v1(
    const pybind11::function& direct_v1,
    const std::shared_ptr<resident_run_control::RunControlV2>& run_control,
    const pybind11::list& modular_tensors,
    const pybind11::dict& modular_config,
    const pybind11::args& arguments,
    const bool expected_lifecycle_enabled
) {
    TORCH_CHECK(
        pybind11::len(arguments) ==
            kDirectResidentDirectTrajectoryArgumentCountV1,
        "CUDA modular v1 execution requires 172 prepared ABI arguments and "
        "five direct-trajectory arguments"
    );
    const c10::cuda::CUDAGuard device_guard(
        pybind11::cast<torch::Tensor>(modular_tensors[0]).device()
    );
    const auto context = validated_modular_context_v1(
        modular_tensors, modular_config
    );
    TORCH_CHECK(
        context.lifecycle_enabled == expected_lifecycle_enabled,
        "modular entrypoint ABI mode contradicts lifecycle_enabled"
    );
    const ModularContextGuardV1 context_guard(context);
    pybind11::dict result = execute_direct_resident_windows_v2(
        direct_v1, run_control, arguments
    );
    result["modular_native_owner"] = true;
    result["modular_native_controller_abi_version"] = expected_lifecycle_enabled
        ? modular_controller::kLifecycleAbiVersionV2
        : modular_controller::kAbiVersionV1;
    result["modular_state_tensors"] = modular_tensors;
    result["modular_python_tick_or_generation_ownership"] = false;
    result["modular_lifecycle_supported"] = context.lifecycle_enabled;
    return result;
}

pybind11::dict execute_direct_resident_modular_physical_v1(
    const pybind11::function& physical_v1,
    const pybind11::list& modular_tensors,
    const pybind11::dict& modular_config,
    const pybind11::args& arguments,
    const bool expected_lifecycle_enabled
) {
    TORCH_CHECK(
        pybind11::len(arguments) ==
            kDirectResidentPhysicalTrajectoryArgumentCountV1,
        "CUDA modular v1 physical execution requires the active-slot "
        "172-argument physical trajectory ABI"
    );
    const c10::cuda::CUDAGuard device_guard(
        pybind11::cast<torch::Tensor>(modular_tensors[0]).device()
    );
    const auto context = validated_modular_context_v1(
        modular_tensors, modular_config
    );
    TORCH_CHECK(
        context.lifecycle_enabled == expected_lifecycle_enabled,
        "modular physical entrypoint ABI mode contradicts lifecycle_enabled"
    );
    const ModularContextGuardV1 context_guard(context);
    PyObject* raw_result = PyObject_CallObject(
        physical_v1.ptr(), arguments.ptr()
    );
    if (raw_result == nullptr) {
        throw pybind11::error_already_set();
    }
    const auto result = pybind11::reinterpret_steal<pybind11::object>(raw_result);
    TORCH_CHECK(
        pybind11::isinstance<pybind11::dict>(result),
        "CUDA modular v1 physical trajectory must return one native result mapping"
    );
    pybind11::dict out = pybind11::cast<pybind11::dict>(result);
    out["modular_native_owner"] = true;
    out["modular_native_controller_abi_version"] = expected_lifecycle_enabled
        ? modular_controller::kLifecycleAbiVersionV2
        : modular_controller::kAbiVersionV1;
    out["modular_physical_gate_entrypoint"] = true;
    out["modular_python_tick_or_generation_ownership"] = false;
    out["modular_lifecycle_supported"] = context.lifecycle_enabled;
    return out;
}

pybind11::dict execute_direct_resident_jungle_v4(
    const pybind11::function& direct_v1,
    const std::shared_ptr<resident_run_control::RunControlV2>& run_control,
    const pybind11::list& state_tensors,
    const pybind11::list& reporting_tensors,
    const pybind11::dict& config,
    const pybind11::args& arguments
) {
    TORCH_CHECK(
        pybind11::len(arguments) ==
            kDirectResidentDirectTrajectoryArgumentCountV1,
        "CUDA Jungle v4 execution requires 172 prepared ABI arguments and "
        "five direct-trajectory arguments"
    );
    const auto context = validated_jungle_context_v4(
        state_tensors, reporting_tensors, config
    );
    const c10::cuda::CUDAGuard device_guard(
        pybind11::cast<torch::Tensor>(state_tensors[0]).device()
    );
    const JungleContextGuardV4 context_guard(context);
    pybind11::dict result = execute_direct_resident_windows_v2(
        direct_v1, run_control, arguments
    );
    result["jungle_native_owner"] = true;
    result["jungle_native_controller_abi_version"] =
        jungle_controller::kAbiVersionV4;
    result["jungle_state_tensors"] = state_tensors;
    result["jungle_reporting_tensors"] = reporting_tensors;
    result["jungle_python_tick_or_generation_ownership"] = false;
    return result;
}

pybind11::dict execute_direct_resident_jungle_physical_v4(
    const pybind11::function& physical_v1,
    const pybind11::list& state_tensors,
    const pybind11::list& reporting_tensors,
    const pybind11::dict& config,
    const pybind11::args& arguments
) {
    TORCH_CHECK(
        pybind11::len(arguments) ==
            kDirectResidentPhysicalTrajectoryArgumentCountV1,
        "CUDA Jungle v4 physical execution requires the active-slot "
        "172-argument physical trajectory ABI"
    );
    const auto context = validated_jungle_context_v4(
        state_tensors, reporting_tensors, config
    );
    const c10::cuda::CUDAGuard device_guard(
        pybind11::cast<torch::Tensor>(state_tensors[0]).device()
    );
    const JungleContextGuardV4 context_guard(context);
    PyObject* raw_result = PyObject_CallObject(
        physical_v1.ptr(), arguments.ptr()
    );
    if (raw_result == nullptr) {
        throw pybind11::error_already_set();
    }
    const auto result = pybind11::reinterpret_steal<pybind11::object>(raw_result);
    TORCH_CHECK(
        pybind11::isinstance<pybind11::dict>(result),
        "CUDA Jungle v4 physical trajectory must return one native result mapping"
    );
    pybind11::dict out = pybind11::cast<pybind11::dict>(result);
    out["jungle_native_owner"] = true;
    out["jungle_native_controller_abi_version"] =
        jungle_controller::kAbiVersionV4;
    out["jungle_physical_gate_entrypoint"] = true;
    out["jungle_python_tick_or_generation_ownership"] = false;
    return out;
}

pybind11::dict execute_direct_resident_wdl_physical_v2(
    const pybind11::function& physical_v1,
    const pybind11::list& state_tensors,
    const pybind11::list& schedule_tensors,
    const pybind11::list& reporting_tensors,
    const pybind11::dict& config,
    const pybind11::args& arguments
) {
    TORCH_CHECK(
        pybind11::len(arguments) ==
            kDirectResidentPhysicalTrajectoryArgumentCountV1,
        "CUDA WDL v2 physical execution requires the active-slot 172-argument "
        "physical trajectory ABI"
    );
    const auto context = validated_wdl_context_v2(
        state_tensors, schedule_tensors, reporting_tensors, config
    );
    const c10::cuda::CUDAGuard device_guard(
        pybind11::cast<torch::Tensor>(state_tensors[0]).device()
    );
    const WdlContextGuardV2 context_guard(context);
    PyObject* raw_result = PyObject_CallObject(
        physical_v1.ptr(), arguments.ptr()
    );
    if (raw_result == nullptr) {
        throw pybind11::error_already_set();
    }
    const auto result = pybind11::reinterpret_steal<pybind11::object>(raw_result);
    TORCH_CHECK(
        pybind11::isinstance<pybind11::dict>(result),
        "CUDA WDL v2 physical trajectory must return one native result mapping"
    );
    pybind11::dict out = pybind11::cast<pybind11::dict>(result);
    out["wdl_native_owner"] = true;
    out["wdl_native_controller_abi_version"] = wdl_controller::kAbiVersionV3;
    out["wdl_physical_gate_entrypoint"] = true;
    out["wdl_python_tick_or_generation_ownership"] = false;
    return out;
}

#ifndef EIEL_NON_EED_WDL_JUNGLE_ONLY
pybind11::dict execute_direct_resident_eed_physical_v1(
    const pybind11::function& physical_v1,
    const pybind11::list& state_tensors,
    const pybind11::dict& config,
    const pybind11::args& arguments
) {
    TORCH_CHECK(
        pybind11::len(arguments) ==
            kDirectResidentPhysicalTrajectoryArgumentCountV1,
        "CUDA EED v1 physical execution requires the active-slot 172-argument "
        "physical trajectory ABI"
    );
    const auto context =
        eed_controller::validated_resident_context_v1(state_tensors, config);
    const c10::cuda::CUDAGuard device_guard(
        pybind11::cast<torch::Tensor>(state_tensors[0]).device()
    );
    const EedContextGuardV1 context_guard(context);
    PyObject* raw_result = PyObject_CallObject(
        physical_v1.ptr(), arguments.ptr()
    );
    if (raw_result == nullptr) {
        throw pybind11::error_already_set();
    }
    const auto result = pybind11::reinterpret_steal<pybind11::object>(raw_result);
    TORCH_CHECK(
        pybind11::isinstance<pybind11::dict>(result),
        "CUDA EED v1 physical trajectory must return one native result mapping"
    );
    pybind11::dict out = pybind11::cast<pybind11::dict>(result);
    out["eed_native_owner"] = true;
    out["eed_native_controller_abi_version"] =
        eed_controller::kResidentSchemaVersion;
    out["eed_physical_gate_entrypoint"] = true;
    out["eed_python_tick_or_generation_ownership"] = false;
    return out;
}
#endif

#ifndef EIEL_NON_EED_WDL_JUNGLE_ONLY
pybind11::dict eed_capability_v1() {
    pybind11::dict out =
        flattened_flat_dynamic_policy_food_birth_controller_generation_count_capability_v1();
    out["abi_version"] = eed_controller::kResidentSchemaVersion;
    out["world"] = "patch_eed_resource_impact";
    out["resident_state"] = true;
    out["observation_size"] = 74;
    out["exact_respec_handling"] = true;
    out["exact_resource_impact_settlement"] = true;
    out["native_tick_ownership"] = true;
    out["runtime_jit_required"] = false;
    out["production_qualified"] = false;
    return out;
}
#endif

pybind11::dict wdl_capability_v3() {
    pybind11::dict out;
    out["abi_version"] = wdl_controller::kAbiVersionV3;
    out["world"] = "patch_wdl";
    out["resident_state"] = true;
    out["observation_size"] =
        wdl_controller::kPatchWdlRadiusOneObservationSizeV2;
    out["exact_phase_schedule"] = true;
    out["exact_cue_observation"] = true;
    out["exact_delayed_damage"] = true;
    out["native_tick_ownership"] = true;
    out["generation_history_abi"] = "compact_keyed_generation_history_v3";
    out["generation_history_capacity_formula"] =
        "slot_capacity+ticks_per_generation*max_births_per_tick";
    out["generation_history_no_eviction"] = true;
    out["runtime_jit_required"] = false;
    out["production_qualified"] = false;
    return out;
}

pybind11::dict modular_capability_v1() {
    pybind11::dict out;
    out["abi_version"] = modular_controller::kAbiVersionV1;
    out["resident_state"] = true;
    out["physical_module_slots"] = modular_controller::kModuleSlotsV1;
    out["physical_connection_slots"] = modular_controller::kConnectionSlotsV1;
    out["action_width"] = modular_controller::kActionWidthV1;
    out["deterministic_fixed_mlp"] = true;
    out["cache_width"] = 1;
    out["births"] = false;
    out["turnover"] = false;
    out["native_tick_ownership"] = true;
    out["native_generation_ownership"] = true;
    out["runtime_jit_required"] = false;
    out["hidden_fallback"] = false;
    out["production_qualified"] = false;
    out["optional_action_residual_trace"] = true;
    return out;
}

pybind11::dict modular_lifecycle_capability_v2() {
    pybind11::dict out;
    out["abi_version"] = modular_controller::kLifecycleAbiVersionV2;
    out["resident_state"] = true;
    out["physical_module_slots"] = modular_controller::kModuleSlotsV1;
    out["physical_connection_slots"] = modular_controller::kConnectionSlotsV1;
    out["action_width"] = modular_controller::kActionWidthV1;
    out["deterministic_fixed_mlp"] = true;
    out["cache_width"] = 1;
    out["births"] = true;
    out["turnover"] = true;
    out["no_mutation_inheritance"] = true;
    out["heterogeneous_parent_structure"] = true;
    out["native_tick_ownership"] = true;
    out["native_generation_ownership"] = true;
    out["runtime_jit_required"] = false;
    out["hidden_fallback"] = false;
    out["production_qualified"] = false;
    out["optional_action_residual_trace"] = true;
    out["turnover_refine_descendant_parameter_qualification"] = true;
    out["turnover_mutation_receipt_raw_layout_version"] = 2;
    out["turnover_mutation_receipt_sequence_capacity"] =
        modular_controller::kTurnoverMutationSequenceCapacityV1;
    out["mutation_receipt_sequence_v1"] = true;
    out["turnover_mutation_qualified_generation_horizon"] = 3;
    out["turnover_mutation_overflow_test_generation"] = 4;
    out["mutated_checkpoint_fresh_owner"] = true;
    out["ordinary_birth_mutation"] = true;
    out["ordinary_birth_gated_parameter_refinement"] = true;
    out["ordinary_birth_mutation_receipt_raw_layout_version"] = 2;
    out["ordinary_birth_mutation_receipt_capacity"] =
        modular_controller::kTurnoverMutationSequenceCapacityV1;
    out["ordinary_birth_rowv1_receipt_abi_version"] = 3;
    out["ordinary_birth_rowv1_receipt_width"] =
        modular_controller::kOrdinaryBirthRowV1ReceiptWidthV3;
    out["ordinary_birth_rowv1_receipt_capacity"] =
        modular_controller::kTurnoverMutationSequenceCapacityV1;
    out["ordinary_birth_rowv1_receipt_default_production"] = false;
    out["ordinary_birth_rowv1_receipt_kernel_mirror_present"] = true;
    out["ordinary_birth_rowv1_receipt_kernel_write_qualified"] = true;
    out["ordinary_birth_rowv1_receipt_disabled_event_writer"] = true;
    out["ordinary_birth_rowv1_receipt_device_writer_sanitized"] = true;
    out["ordinary_birth_mutation_default_production"] = false;
    out["turnover_mutation_g8_g32"] = false;
    out["turnover_mutation_default_production"] = false;
    return out;
}

pybind11::dict jungle_capability_v4() {
    pybind11::dict out;
    out["abi_version"] = jungle_controller::kAbiVersionV4;
    out["world"] = "patch_jungle";
    out["resident_state"] = true;
    out["exact_predator_pressure"] = true;
    out["exact_birth_emissions"] = true;
    out["exact_ecological_recovery"] = true;
    out["native_tick_ownership"] = true;
    out["runtime_jit_required"] = false;
    out["production_qualified"] = false;
    return out;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    bind_flattened_controller_v1_for_wdl_v2(module);
#if EIEL_TW_LIFECYCLE_LINEAR_FAST_PATH_TEST_SURFACE_ENABLED
    module.def(
        "_test_only_lifecycle_death_cull_cell_v3",
        &eiel::cuda_architecture_b::flattened_controller_v1::
            test_only_lifecycle_death_cull_cell_v3
    );
#endif
#ifndef EIEL_NON_EED_WDL_JUNGLE_ONLY
    eiel::eed_resource_impact_native_v1::bind_eed_resource_impact_native_v1(
        module
    );
#endif
#ifdef EIEL_WDL_JUNGLE_EED_HAPB_FIXED_MLP_NATIVE_BINDING_V1
    eiel::cuda_architecture_b::hapb_fixed_mlp_native_binding_v1::
        bind_hapb_fixed_mlp_native_v1(module);
#endif
    const auto direct_v1 = module.attr(
        "eiel_cuda_architecture_b_execute_direct_resident_windows_v1"
    ).cast<pybind11::function>();
    const auto physical_v1 = module.attr(
        "eiel_cuda_architecture_b_flattened_controller_v1"
    ).cast<pybind11::function>();
    module.def(
        "eiel_cuda_architecture_b_execute_direct_resident_modular_v1",
        [direct_v1](
            const std::shared_ptr<resident_run_control::RunControlV2>& run_control,
            const pybind11::list& modular_tensors,
            const pybind11::dict& modular_config,
            const pybind11::args& arguments
        ) {
            return execute_direct_resident_modular_v1(
                direct_v1,
                run_control,
                modular_tensors,
                modular_config,
                arguments,
                false
            );
        }
    );
    module.def(
        "eiel_cuda_architecture_b_execute_direct_resident_modular_physical_v1",
        [physical_v1](
            const pybind11::list& modular_tensors,
            const pybind11::dict& modular_config,
            const pybind11::args& arguments
        ) {
            return execute_direct_resident_modular_physical_v1(
                physical_v1, modular_tensors, modular_config, arguments, false
            );
        }
    );
    module.def(
        "eiel_cuda_architecture_b_flattened_modular_capability_v1",
        &modular_capability_v1
    );
    module.def(
        "eiel_cuda_architecture_b_execute_direct_resident_modular_lifecycle_v2",
        [direct_v1](
            const std::shared_ptr<resident_run_control::RunControlV2>& run_control,
            const pybind11::list& modular_tensors,
            const pybind11::dict& modular_config,
            const pybind11::args& arguments
        ) {
            return execute_direct_resident_modular_v1(
                direct_v1,
                run_control,
                modular_tensors,
                modular_config,
                arguments,
                true
            );
        }
    );
    module.def(
        "eiel_cuda_architecture_b_execute_direct_resident_modular_lifecycle_physical_v2",
        [physical_v1](
            const pybind11::list& modular_tensors,
            const pybind11::dict& modular_config,
            const pybind11::args& arguments
        ) {
            return execute_direct_resident_modular_physical_v1(
                physical_v1, modular_tensors, modular_config, arguments, true
            );
        }
    );
    module.def(
        "eiel_cuda_architecture_b_flattened_modular_lifecycle_capability_v2",
        &modular_lifecycle_capability_v2
    );
    module.def(
        "eiel_cuda_architecture_b_execute_direct_resident_jungle_v4",
        [direct_v1](
            const std::shared_ptr<resident_run_control::RunControlV2>& run_control,
            const pybind11::list& state_tensors,
            const pybind11::list& reporting_tensors,
            const pybind11::dict& config,
            const pybind11::args& arguments
        ) {
            return execute_direct_resident_jungle_v4(
                direct_v1,
                run_control,
                state_tensors,
                reporting_tensors,
                config,
                arguments
            );
        }
    );
    module.def(
        "eiel_cuda_architecture_b_execute_direct_resident_jungle_physical_v4",
        [physical_v1](
            const pybind11::list& state_tensors,
            const pybind11::list& reporting_tensors,
            const pybind11::dict& config,
            const pybind11::args& arguments
        ) {
            return execute_direct_resident_jungle_physical_v4(
                physical_v1,
                state_tensors,
                reporting_tensors,
                config,
                arguments
            );
        }
    );
    module.def(
        "eiel_cuda_architecture_b_flattened_jungle_capability_v4",
        &jungle_capability_v4
    );
    module.def(
        "eiel_cuda_architecture_b_execute_direct_resident_wdl_v3",
        [direct_v1](
            const std::shared_ptr<resident_run_control::RunControlV2>& run_control,
            const pybind11::list& state_tensors,
            const pybind11::list& schedule_tensors,
            const pybind11::list& reporting_tensors,
            const pybind11::dict& config,
            const pybind11::args& arguments
        ) {
            return execute_direct_resident_wdl_v2(
                direct_v1,
                run_control,
                state_tensors,
                schedule_tensors,
                reporting_tensors,
                config,
                arguments
            );
        }
    );
    module.def(
        "eiel_cuda_architecture_b_execute_direct_resident_wdl_jungle_v1",
        [direct_v1](
            const std::shared_ptr<resident_run_control::RunControlV2>& run_control,
            const pybind11::list& wdl_state_tensors,
            const pybind11::list& wdl_schedule_tensors,
            const pybind11::list& wdl_reporting_tensors,
            const pybind11::dict& wdl_config,
            const pybind11::list& jungle_state_tensors,
            const pybind11::list& jungle_reporting_tensors,
            const pybind11::dict& jungle_config,
            const pybind11::args& arguments
        ) {
            return execute_direct_resident_wdl_jungle_v1(
                direct_v1,
                run_control,
                wdl_state_tensors,
                wdl_schedule_tensors,
                wdl_reporting_tensors,
                wdl_config,
                jungle_state_tensors,
                jungle_reporting_tensors,
                jungle_config,
                arguments
            );
        }
    );
#ifndef EIEL_NON_EED_WDL_JUNGLE_ONLY
    module.def(
        "eiel_cuda_architecture_b_execute_direct_resident_wdl_jungle_eed_v1",
        [direct_v1](
            const std::shared_ptr<resident_run_control::RunControlV2>& run_control,
            const pybind11::list& wdl_state_tensors,
            const pybind11::list& wdl_schedule_tensors,
            const pybind11::list& wdl_reporting_tensors,
            const pybind11::dict& wdl_config,
            const pybind11::list& jungle_state_tensors,
            const pybind11::list& jungle_reporting_tensors,
            const pybind11::dict& jungle_config,
            const pybind11::list& eed_state_tensors,
            const pybind11::dict& eed_config,
            const pybind11::args& arguments
        ) {
            return execute_direct_resident_wdl_jungle_eed_v1(
                direct_v1,
                run_control,
                wdl_state_tensors,
                wdl_schedule_tensors,
                wdl_reporting_tensors,
                wdl_config,
                jungle_state_tensors,
                jungle_reporting_tensors,
                jungle_config,
                eed_state_tensors,
                eed_config,
                arguments
            );
        }
    );
    module.def(
        "eiel_cuda_architecture_b_execute_direct_resident_wdl_jungle_eed_tick_dispatch_v2",
        [direct_v1](
            const pybind11::list& wdl_state_tensors,
            const pybind11::list& wdl_schedule_tensors,
            const pybind11::list& wdl_reporting_tensors,
            const pybind11::dict& wdl_config,
            const pybind11::list& jungle_state_tensors,
            const pybind11::list& jungle_reporting_tensors,
            const pybind11::dict& jungle_config,
            const pybind11::list& eed_state_tensors,
            const pybind11::dict& eed_config,
            const pybind11::list& source_windows
        ) {
            return execute_direct_resident_wdl_jungle_eed_tick_dispatch_v2(
                direct_v1,
                wdl_state_tensors,
                wdl_schedule_tensors,
                wdl_reporting_tensors,
                wdl_config,
                jungle_state_tensors,
                jungle_reporting_tensors,
                jungle_config,
                eed_state_tensors,
                eed_config,
                source_windows
            );
        }
    );
#endif
    module.def(
        "eiel_cuda_architecture_b_execute_direct_resident_wdl_physical_v3",
        [physical_v1](
            const pybind11::list& state_tensors,
            const pybind11::list& schedule_tensors,
            const pybind11::list& reporting_tensors,
            const pybind11::dict& config,
            const pybind11::args& arguments
        ) {
            return execute_direct_resident_wdl_physical_v2(
                physical_v1,
                state_tensors,
                schedule_tensors,
                reporting_tensors,
                config,
                arguments
            );
        }
    );
    module.def(
        "eiel_cuda_architecture_b_flattened_wdl_capability_v3",
        &wdl_capability_v3
    );
#ifndef EIEL_NON_EED_WDL_JUNGLE_ONLY
    module.def(
        "eiel_cuda_architecture_b_execute_direct_resident_eed_v1",
        [direct_v1](
            const std::shared_ptr<resident_run_control::RunControlV2>& run_control,
            const pybind11::list& state_tensors,
            const pybind11::dict& config,
            const pybind11::args& arguments
        ) {
            return execute_direct_resident_eed_v1(
                direct_v1,
                run_control,
                state_tensors,
                config,
                arguments
            );
        }
    );
    module.def(
        "eiel_cuda_architecture_b_execute_direct_resident_eed_physical_v1",
        [physical_v1](
            const pybind11::list& state_tensors,
            const pybind11::dict& config,
            const pybind11::args& arguments
        ) {
            return execute_direct_resident_eed_physical_v1(
                physical_v1, state_tensors, config, arguments
            );
        }
    );
    module.def(
        "eiel_cuda_architecture_b_flattened_eed_capability_v1",
        &eed_capability_v1
    );
#endif
    module.def(
        "eiel_cuda_architecture_b_execute_direct_resident_gru_v3",
        [direct_v1](
            const std::shared_ptr<resident_run_control::RunControlV2>& run_control,
            const pybind11::list& gru_tensors,
            const pybind11::dict& gru_config,
            const pybind11::args& arguments
        ) {
            return execute_direct_resident_gru_v3(
                direct_v1, run_control, gru_tensors, gru_config, arguments
            );
        }
    );
    module.def(
        "eiel_cuda_architecture_b_execute_direct_resident_gru_physical_v3",
        [physical_v1](
            const pybind11::list& gru_tensors,
            const pybind11::dict& gru_config,
            const pybind11::args& arguments
        ) {
            return execute_direct_resident_gru_physical_v3(
                physical_v1, gru_tensors, gru_config, arguments
            );
        }
    );
    module.def(
        "eiel_cuda_architecture_b_execute_direct_resident_wdl_gru_v3",
        [direct_v1](
            const std::shared_ptr<resident_run_control::RunControlV2>& run_control,
            const pybind11::list& gru_tensors,
            const pybind11::dict& gru_config,
            const pybind11::list& state_tensors,
            const pybind11::list& schedule_tensors,
            const pybind11::list& reporting_tensors,
            const pybind11::dict& config,
            const pybind11::args& arguments
        ) {
            return execute_direct_resident_wdl_gru_v3(
                direct_v1,
                run_control,
                gru_tensors,
                gru_config,
                state_tensors,
                schedule_tensors,
                reporting_tensors,
                config,
                arguments
            );
        }
    );
    module.def(
        "eiel_cuda_architecture_b_execute_direct_resident_wdl_gru_physical_v3",
        [physical_v1](
            const pybind11::list& gru_tensors,
            const pybind11::dict& gru_config,
            const pybind11::list& state_tensors,
            const pybind11::list& schedule_tensors,
            const pybind11::list& reporting_tensors,
            const pybind11::dict& config,
            const pybind11::args& arguments
        ) {
            return execute_direct_resident_wdl_gru_physical_v3(
                physical_v1,
                gru_tensors,
                gru_config,
                state_tensors,
                schedule_tensors,
                reporting_tensors,
                config,
                arguments
            );
        }
    );
    module.def(
        "eiel_cuda_architecture_b_flattened_gru_capability_v3",
        &gru_capability_v3
    );
    module.def(
        "eiel_cuda_architecture_b_execute_direct_resident_gmnw_gru_v1",
        [direct_v1](
            const std::shared_ptr<resident_run_control::RunControlV2>& run_control,
            const pybind11::list& gru_tensors,
            const pybind11::dict& gru_config,
            const pybind11::list& gmnw_tensors,
            const pybind11::dict& gmnw_config,
            const pybind11::args& arguments
        ) {
            return execute_direct_resident_gmnw_gru_v1(
                direct_v1,
                run_control,
                gru_tensors,
                gru_config,
                gmnw_tensors,
                gmnw_config,
                arguments
            );
        }
    );
    module.def(
        "eiel_cuda_architecture_b_execute_direct_resident_gmnw_gru_physical_v1",
        [physical_v1](
            const pybind11::list& gru_tensors,
            const pybind11::dict& gru_config,
            const pybind11::list& gmnw_tensors,
            const pybind11::dict& gmnw_config,
            const pybind11::args& arguments
        ) {
            return execute_direct_resident_gmnw_gru_physical_v1(
                physical_v1,
                gru_tensors,
                gru_config,
                gmnw_tensors,
                gmnw_config,
                arguments
            );
        }
    );
    module.def(
        "eiel_cuda_architecture_b_flattened_gmnw_gru_capability_v1",
        &gmnw_gru_capability_v1
    );
}
