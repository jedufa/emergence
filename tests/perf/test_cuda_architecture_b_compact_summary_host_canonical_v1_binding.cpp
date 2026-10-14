#include "cuda_architecture_b_compact_summary_reporting_authority_v1.cuh"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>

#include "../../engine/perf/cuda_architecture_b_compact_summary_host_canonical_v1.cpp"

namespace py = pybind11;
namespace authority =
    eiel::cuda_architecture_b::compact_summary_reporting_authority_v1;
namespace canonical =
    eiel::cuda_architecture_b::compact_summary_host_canonical_v1;

namespace {

struct FixtureV1 {
    std::int64_t generation;
    authority::GenerationAuthorityHeaderV1 header;
    std::array<authority::BehaviorAuthorityRowV1, 4> behavior;
    std::array<authority::LineageAuthorityRowV1, 4> lineage;
};

authority::BehaviorAuthorityRowV1 behavior_row(
    const std::int64_t agent_id,
    const double energy,
    const double fitness,
    const float injury_start,
    const float injury_end,
    const float aggression,
    const std::int32_t lifespan,
    const std::int32_t offspring,
    const std::int32_t attacks,
    const std::array<std::uint8_t, 7>& actions,
    const std::uint8_t flags
) {
    authority::BehaviorAuthorityRowV1 row{};
    row.agent_id = agent_id;
    row.energy_collected_exact = energy;
    row.base_fitness = fitness;
    row.injury_start = injury_start;
    row.injury_end = injury_end;
    row.aggression_bias_f32 = aggression;
    row.lifespan_steps = lifespan;
    row.offspring = offspring;
    row.attacks_made = attacks;
    std::memcpy(row.action_counts, actions.data(), actions.size());
    row.flags = flags;
    return row;
}

authority::LineageAuthorityRowV1 lineage_row(
    const std::int64_t agent_id,
    const std::int32_t founder_id
) {
    authority::LineageAuthorityRowV1 row{};
    row.agent_id = agent_id;
    row.founder_id = founder_id;
    row.first_repro_step = -1;
    return row;
}

FixtureV1 signed_zero_fixture() {
    FixtureV1 fixture{};
    fixture.generation = 0;
    fixture.header = {1, 1, 1, 2};
    fixture.behavior[0] = behavior_row(
        0, -0.0, 0.0, -0.0F, -0.0F, -0.0F, 0, 0, 0,
        {0, 0, 0, 0, 0, 0, 0}, 0x0fU
    );
    fixture.lineage[0] = lineage_row(0, 0);
    fixture.lineage[0].energy = -0.0;
    fixture.lineage[0].energy_collected_exact = -0.0;
    fixture.lineage[0].health = -0.0F;
    fixture.lineage[0].injury = -0.0F;
    fixture.lineage[0].drives[0] = -0.0F;
    fixture.lineage[0].drives[2] = -0.0F;
    fixture.lineage[0].pred_err_sum = -0.0F;
    fixture.lineage[0].aggression_bias_f32 = -0.0F;
    fixture.lineage[0].exploration_bias_f32 = -0.0F;
    return fixture;
}

FixtureV1 real_shaped_fixture() {
    FixtureV1 fixture{};
    fixture.generation = 4;
    fixture.header = {12, 3, 2, 2};
    fixture.behavior[0] = behavior_row(
        10, 0.75, 1.5, -0.0F, 0.5F, 0.375F, 3, 0, 0,
        {2, 0, 0, 0, 0, 0, 0}, 0x08U
    );
    fixture.behavior[1] = behavior_row(
        11, 2.5, 8.25, 0.25F, 0.125F, -0.25F, 6, 1, 1,
        {0, 0, 0, 0, 0, 0, 1}, 0x0fU
    );
    fixture.behavior[2] = behavior_row(
        12, 1.5, 5.125, 0.0F, 0.0F, 0.42961761F, 5, 0, 0,
        {0, 0, 0, 0, 1, 0, 0}, 0x09U
    );

    fixture.lineage[0] = lineage_row(11, 11);
    fixture.lineage[0].energy = 41.0;
    fixture.lineage[0].energy_collected_exact = 2.5;
    fixture.lineage[0].episode0_pred_err_mean = 0.25;
    fixture.lineage[0].identity_drift_ema = 0.03125;
    fixture.lineage[0].health = 1.0F;
    fixture.lineage[0].injury = 0.125F;
    fixture.lineage[0].drives[0] = 1.1F;
    fixture.lineage[0].drives[1] = 1.2F;
    fixture.lineage[0].drives[2] = 1.3F;
    fixture.lineage[0].drives[3] = 1.4F;
    fixture.lineage[0].pred_err_sum = 1.0F;
    fixture.lineage[0].drive_jitter = 0.0625F;
    fixture.lineage[0].aggression_bias_f32 = -0.25F;
    fixture.lineage[0].exploration_bias_f32 = -0.25F;
    fixture.lineage[0].y = 3;
    fixture.lineage[0].x = 4;
    fixture.lineage[0].offspring_count = 1;
    fixture.lineage[0].survival_steps = 6;
    fixture.lineage[0].first_repro_step = 4;
    fixture.lineage[0].pred_err_count = 2;
    fixture.lineage[0].patch_habitat_lineage_id = 8;
    fixture.lineage[0].hidden_size = 9;
    fixture.lineage[0].memory_dim = 5;
    fixture.lineage[0].use_gru = 1;

    fixture.lineage[1] = lineage_row(12, 12);
    fixture.lineage[1].energy = 40.0;
    fixture.lineage[1].energy_collected_exact = 1.5;
    fixture.lineage[1].episode0_pred_err_mean = 0.25;
    fixture.lineage[1].identity_drift_ema = 0.015625;
    fixture.lineage[1].health = 1.0F;
    fixture.lineage[1].drives[0] = 0.1F;
    fixture.lineage[1].drives[1] = 0.2F;
    fixture.lineage[1].drives[2] = 0.3F;
    fixture.lineage[1].drives[3] = 0.4F;
    fixture.lineage[1].pred_err_sum = 0.5F;
    fixture.lineage[1].drive_jitter = 0.03125F;
    fixture.lineage[1].aggression_bias_f32 = 0.123456789F;
    fixture.lineage[1].exploration_bias_f32 = -0.42961761F;
    fixture.lineage[1].y = 2;
    fixture.lineage[1].x = 3;
    fixture.lineage[1].birth_tick = 3;
    fixture.lineage[1].lineage_depth = 1;
    fixture.lineage[1].survival_steps = 5;
    fixture.lineage[1].pred_err_count = 1;
    fixture.lineage[1].patch_habitat_lineage_id = 7;
    fixture.lineage[1].hidden_size = 8;
    fixture.lineage[1].memory_dim = 4;
    return fixture;
}

py::dict result_to_dict(const canonical::HostCanonicalResultV1& result) {
    py::dict output;
    output["status"] = result.status;
    output["failed_field"] = result.failed_field;
    output["failed_row"] = result.failed_row;
    output["behavior_digest"] = py::bytes(
        reinterpret_cast<const char*>(result.behavior_rows_digest_sha256),
        canonical::kDigestBytesV1
    );
    output["lineage_digest"] = py::bytes(
        reinterpret_cast<const char*>(result.lineage_records_digest_sha256),
        canonical::kDigestBytesV1
    );

    py::dict behavior;
    behavior["row_count"] = result.behavior.row_count;
    behavior["cohort_size"] = result.behavior.cohort_size;
    behavior["cohort_survivors"] = result.behavior.cohort_survivors;
    behavior["action_total"] = result.behavior.action_total;
    behavior["move_total"] = result.behavior.move_total;
    behavior["move_n"] = result.behavior.move_n;
    behavior["move_s"] = result.behavior.move_s;
    behavior["move_e"] = result.behavior.move_e;
    behavior["move_w"] = result.behavior.move_w;
    behavior["gather"] = result.behavior.gather;
    behavior["rest"] = result.behavior.rest;
    behavior["attack"] = result.behavior.attack;
    output["behavior"] = behavior;

    const auto& lineage = result.lineage;
    py::dict lineage_output;
    lineage_output["defined_mask"] = lineage.compact_abi_v2_defined_mask;
    lineage_output["record_count"] = lineage.record_count;
    lineage_output["active_lineages"] = lineage.active_lineages;
    lineage_output["dominant_population_share"] =
        lineage.dominant_population_share;
    lineage_output["top3_population_share"] = lineage.top3_population_share;
    lineage_output["population_entropy"] = lineage.population_entropy;
    if ((lineage.compact_abi_v2_defined_mask
         & canonical::kCompactAbiV2DefinedLineageReproductionEntropyV1) != 0U) {
        lineage_output["reproduction_entropy"] = lineage.reproduction_entropy;
        lineage_output["dominant_reproduction_share"] =
            lineage.dominant_reproduction_share;
    } else {
        lineage_output["reproduction_entropy"] = py::none();
        lineage_output["dominant_reproduction_share"] = py::none();
    }
    lineage_output["rare_persistent_lineages"] =
        lineage.rare_persistent_lineages;
    lineage_output["collapse_streak"] = lineage.collapse_streak;
    lineage_output["collapse_warning_code"] = lineage.collapse_warning_code;
    if ((lineage.compact_abi_v2_defined_mask
         & canonical::kCompactAbiV2DefinedLineageDominantTextV1) != 0U) {
        lineage_output["dominant_lineage_id"] = py::str(
            reinterpret_cast<const char*>(lineage.dominant_lineage_text_record + 1U),
            lineage.dominant_lineage_text_record[0]
        );
    } else {
        lineage_output["dominant_lineage_id"] = py::none();
    }
    lineage_output["dominant_lineage_text_record"] = py::bytes(
        reinterpret_cast<const char*>(lineage.dominant_lineage_text_record),
        canonical::kDominantLineageTextRecordBytesV1
    );
    output["lineage"] = lineage_output;

    py::dict state;
    state["collapse_streak"] = result.lineage_state_after.collapse_streak;
    state["population_dominance_streak"] =
        result.lineage_state_after.population_dominance_streak;
    state["reproduction_dominance_streak"] =
        result.lineage_state_after.reproduction_dominance_streak;
    output["lineage_state_after"] = state;
    return output;
}

py::dict run_fixture(const std::string& name, const bool callback) {
    FixtureV1 fixture = name == "signed_zero" ? signed_zero_fixture() : real_shaped_fixture();
    if (name == "unsealed") {
        fixture.header.status = 1;
    } else if (name == "count") {
        fixture.header.behavior_row_count = 5;
    } else if (name == "behavior_order") {
        fixture.behavior[1].agent_id = 9;
    } else if (name == "behavior_nonfinite") {
        fixture.behavior[0].base_fitness = std::numeric_limits<double>::infinity();
    } else if (name == "lineage_order") {
        fixture.lineage[1].agent_id = 10;
    } else if (name == "lineage_nonfinite") {
        fixture.lineage[0].drives[2] = std::numeric_limits<float>::quiet_NaN();
    } else if (name == "lineage_mismatch") {
        fixture.lineage[0].agent_id = 10;
    } else if (name == "alive_count_mismatch") {
        fixture.behavior[2].flags = 0x08U;
    } else if (name == "inactive_nonzero") {
        fixture.behavior[3].agent_id = 99;
    } else if (name != "signed_zero" && name != "real_shaped") {
        throw py::value_error("unknown fixture");
    }

    canonical::HostCanonicalResultV1 result{};
    if (callback) {
        canonical::HostCanonicalRequestV1 request{
            fixture.generation,
            &fixture.header,
            fixture.behavior.data(),
            fixture.lineage.data(),
            4,
            0,
            canonical::kDefaultLineageMetricsInputV1,
            &result,
        };
        canonical::canonicalize_retained_digests_callback_v1(&request);
    } else {
        canonical::canonicalize_retained_digests_v1(
            fixture.generation,
            &fixture.header,
            fixture.behavior.data(),
            fixture.lineage.data(),
            4,
            &result
        );
    }
    return result_to_dict(result);
}

py::dict run_rows(
    const std::int64_t generation,
    const std::int64_t start_id_ceiling,
    const py::list& behavior_input,
    const py::list& lineage_input,
    const py::dict& state_input,
    const py::dict& config_input,
    const bool metrics_defined
) {
    constexpr std::size_t kCapacity = 144;
    const std::size_t behavior_count = py::len(behavior_input);
    const std::size_t lineage_count = py::len(lineage_input);
    if (behavior_count > kCapacity || lineage_count > kCapacity) {
        throw py::value_error("row capacity exceeds 144");
    }
    std::array<authority::BehaviorAuthorityRowV1, kCapacity> behavior{};
    std::array<authority::LineageAuthorityRowV1, kCapacity> lineage{};
    for (std::size_t index = 0; index < behavior_count; ++index) {
        const py::dict source = py::cast<py::dict>(behavior_input[index]);
        const std::int64_t agent_id = py::cast<std::int64_t>(source["agent_id"]);
        const bool alive = py::cast<bool>(source["alive"]);
        const py::sequence actions = py::cast<py::sequence>(source["actions"]);
        if (py::len(actions) != 7) {
            throw py::value_error("behavior actions must have length 7");
        }
        std::array<std::uint8_t, 7> action_counts{};
        for (std::size_t action = 0; action < action_counts.size(); ++action) {
            const int value = py::cast<int>(actions[action]);
            if (value < 0 || value > 255) {
                throw py::value_error("behavior action count is outside uint8");
            }
            action_counts[action] = static_cast<std::uint8_t>(value);
        }
        behavior[index] = behavior_row(
            agent_id, 0.0, 0.0, 0.0F, 0.0F, 0.0F, 0, 0, 0,
            action_counts,
            static_cast<std::uint8_t>(0x08U | (alive ? 0x01U : 0x00U))
        );
    }
    for (std::size_t index = 0; index < lineage_count; ++index) {
        const py::dict source = py::cast<py::dict>(lineage_input[index]);
        lineage[index] = lineage_row(
            py::cast<std::int64_t>(source["agent_id"]),
            py::cast<std::int32_t>(source["founder_id"])
        );
        lineage[index].lineage_depth =
            py::cast<std::int32_t>(source["lineage_depth"]);
        lineage[index].offspring_count =
            py::cast<std::int32_t>(source["offspring_count"]);
    }

    const std::size_t required_capacity =
        behavior_count > lineage_count ? behavior_count : lineage_count;
    const std::int32_t row_capacity = static_cast<std::int32_t>(
        required_capacity == 0U ? 1U : required_capacity
    );
    authority::GenerationAuthorityHeaderV1 header{
        start_id_ceiling,
        static_cast<std::uint16_t>(behavior_count),
        static_cast<std::uint16_t>(lineage_count),
        2,
    };
    canonical::LineageMetricsInputV1 metrics{};
    metrics.initial_state.collapse_streak =
        py::cast<std::int64_t>(state_input["collapse_streak"]);
    metrics.initial_state.population_dominance_streak =
        py::cast<std::int64_t>(state_input["population_dominance_streak"]);
    metrics.initial_state.reproduction_dominance_streak =
        py::cast<std::int64_t>(state_input["reproduction_dominance_streak"]);
    metrics.config.population_dominance_threshold = py::cast<double>(
        config_input["lineage_dominance_threshold"]
    );
    metrics.config.entropy_minimum =
        py::cast<double>(config_input["lineage_entropy_minimum"]);
    metrics.config.reproduction_dominance_threshold = py::cast<double>(
        config_input["lineage_reproduction_dominance_threshold"]
    );
    metrics.config.population_dominance_streak_required = py::cast<std::int64_t>(
        config_input["lineage_dominance_streak_required"]
    );
    metrics.config.minimum_active_lineages =
        py::cast<std::int64_t>(config_input["lineage_min_active_lineages"]);
    metrics.config.reproduction_dominance_streak_required =
        py::cast<std::int64_t>(
            config_input["lineage_reproduction_dominance_streak_required"]
        );
    metrics.config.metrics_defined = metrics_defined ? 1U : 0U;

    canonical::HostCanonicalResultV1 result{};
    canonical::canonicalize_retained_summary_v1(
        generation, &header, behavior.data(), lineage.data(), row_capacity,
        &metrics, &result
    );
    return result_to_dict(result);
}

std::string format_float(const double value) {
    char output[96];
    std::size_t length = 0;
    if (!canonical::detail::format_python_json_float_v1(
            value, output, sizeof(output), &length
        )) {
        throw py::value_error("value is not a canonical finite float");
    }
    return std::string(output, length);
}

py::dict pod_receipt() {
    py::dict result;
    result["authority_header_bytes"] = sizeof(authority::GenerationAuthorityHeaderV1);
    result["authority_behavior_bytes"] = sizeof(authority::BehaviorAuthorityRowV1);
    result["authority_lineage_bytes"] = sizeof(authority::LineageAuthorityRowV1);
    result["result_bytes"] = sizeof(canonical::HostCanonicalResultV1);
    result["request_bytes"] = sizeof(canonical::HostCanonicalRequestV1);
    result["lineage_input_bytes"] = sizeof(canonical::LineageMetricsInputV1);
    result["behavior_output_bytes"] = sizeof(canonical::BehaviorAggregateOutputV1);
    result["lineage_output_bytes"] = sizeof(canonical::LineageSummaryOutputV1);
    result["lineage_state_bytes"] = sizeof(canonical::LineageMetricsStateV1);
    result["dominant_text_record_bytes"] =
        canonical::kDominantLineageTextRecordBytesV1;
    result["digest_bytes"] = canonical::kDigestBytesV1;
    return result;
}

py::bytes sha256_bytes(const py::bytes& input) {
    const std::string value = input;
    canonical::Sha256ContextV1 context{};
    std::uint8_t digest[canonical::kDigestBytesV1]{};
    canonical::sha256_init(&context);
    canonical::sha256_update(&context, value.data(), value.size());
    canonical::sha256_final(&context, digest);
    return py::bytes(
        reinterpret_cast<const char*>(digest), canonical::kDigestBytesV1
    );
}

}  // namespace

PYBIND11_MODULE(cuda_architecture_b_compact_summary_host_canonical_v1_probe, module) {
    module.def("run_fixture", &run_fixture, py::arg("name"), py::arg("callback") = false);
    module.def(
        "run_rows", &run_rows,
        py::arg("generation"), py::arg("start_id_ceiling"),
        py::arg("behavior_rows"), py::arg("lineage_rows"),
        py::arg("state"), py::arg("config"), py::arg("metrics_defined") = true
    );
    module.def("format_float", &format_float);
    module.def("pod_receipt", &pod_receipt);
    module.def("sha256_bytes", &sha256_bytes);
}
